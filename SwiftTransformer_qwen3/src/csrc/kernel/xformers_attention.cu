#include "xformers_attention.h"

#include <cassert>
#include <iostream>
#include <cmath>
#include <mutex>

#include <ATen/Context.h>
#include <ATen/ScalarOps.h>
#include <ATen/Tensor.h>
#include <ATen/core/Generator.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Optional.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <torch/torch.h>

#include "xformers/xformers/csrc/attention/cuda/fmha/autogen/cutlassF.h"
#include "xformers/xformers/csrc/attention/cuda/fmha/kernel_forward.h"
#include "xformers/xformers/csrc/attention/cuda/fmha/pytorch_utils.h"

#include "util/cuda_utils.h"
#include "util/debug_utils.h"

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
  There are 2 modes for using this function.
  (Mode BMHK) With all the heads having the same seqlen
  (Mode 1MHK) `batch=1` with all tokens across batches concatenated
*/
std::tuple<at::Tensor, at::Tensor, int64_t, int64_t>
static efficient_attention_forward_cutlass(
    const at::Tensor& query, // [b, seqlen, num_heads, K]
    const at::Tensor& key, // [b, seqlen, num_heads, K]
    const at::Tensor& value, // [b, seqlen, num_heads, Kv]
    const c10::optional<at::Tensor>& bias, // [b, num_heads, seqlen, seqlen]
    // (Mode 1MHK only) [b+1]: cu_seqlens_q[b] contains the
    // position of the first query token for batch $b
    const c10::optional<at::Tensor>& seqstart_q,
    // (Mode 1MHK only) [b+1]: cu_seqlen_k[b] contains the
    // position of the first key token for batch $b
    const c10::optional<at::Tensor>& seqstart_k,
    // (Mode 1MHK only) Maximum sequence length across batches
    const c10::optional<int64_t> max_seqlen_q_,
    double dropout_p, // attention matrix dropout probability
    bool compute_logsumexp,
    int64_t custom_mask_type,
    c10::optional<double> scale,
    const c10::optional<at::Tensor>& seqlen_k,
    const c10::optional<int64_t> window_size,
	at::Tensor& res) {
  TORCH_CHECK(query.dim() == 4);
  TORCH_CHECK(key.dim() == 4);
  TORCH_CHECK(value.dim() == 4);

  // Batch sizes
  TORCH_CHECK(query.size(0) == key.size(0));
  TORCH_CHECK(query.size(0) == value.size(0));

  // Sequence length
  TORCH_CHECK(key.size(1) == value.size(1));

  // Num heads
  TORCH_CHECK(query.size(2) == key.size(2));
  TORCH_CHECK(query.size(2) == value.size(2));

  // Embedding per head
  TORCH_CHECK(query.size(3) == key.size(3));

  int64_t max_seqlen_q, max_seqlen_k;
  TORCH_CHECK(seqstart_q.has_value() == seqstart_k.has_value());
  if (seqstart_q.has_value()) {
    TORCH_CHECK(seqstart_q->scalar_type() == at::ScalarType::Int);
    TORCH_CHECK(seqstart_k->scalar_type() == at::ScalarType::Int);
    TORCH_CHECK(seqstart_q->dim() == 1 && seqstart_k->dim() == 1);
    CHECK_NOSPARSE_CONTIGUOUS_CUDA((*seqstart_q));
    CHECK_NOSPARSE_CONTIGUOUS_CUDA((*seqstart_k));
    TORCH_CHECK(seqstart_q->size(0) == seqstart_k->size(0));
    TORCH_CHECK(query.size(0) == 1, "cu_seqlen only supports batch_size=1");
    TORCH_CHECK(max_seqlen_q_.has_value());
    max_seqlen_q = *max_seqlen_q_;
    max_seqlen_k = 0; // Will be set inside the kernel
  } else {
    max_seqlen_q = query.size(1);
    max_seqlen_k = key.size(1);
  }

  CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(query);
  CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(key);
  CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(value);

  at::cuda::CUDAGuard device_guard(query.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t N = key.size(1);
  int64_t num_heads = query.size(-2);
  int64_t K = query.size(-1);
  int64_t Kv = value.size(-1);

  at::Tensor logsumexp;

  const bool use_dropout = std::fpclassify(dropout_p) != FP_ZERO;
  at::PhiloxCudaState rng_engine_inputs;
  if (use_dropout) {
    at::CUDAGeneratorImpl* gen =
        at::get_generator_or_default<at::CUDAGeneratorImpl>(
            c10::nullopt, at::cuda::detail::getDefaultCUDAGenerator());

    std::lock_guard<std::mutex> lock(gen->mutex_);
    // if using dropout, we produce 1 random number for each element of the
    // attention tensor
    rng_engine_inputs = gen->philox_cuda_state(B * num_heads * M * N);
  }

  cudaDeviceProp* p = at::cuda::getDeviceProperties(query.device().index());
  const int computeCapability = p->major * 10 + p->minor;

  bool kernel_launched = false;
  const auto maxShmem = p->sharedMemPerBlockOptin;

  auto launchKernel = [&](auto _k, auto kernel_fn) {
    using Kernel = decltype(_k);
    using scalar_t = typename Kernel::scalar_t;
    (void)_k;

    if (kernel_launched) {
      return;
    }
    // Check if this kernel is compatible
    if (!Kernel::kSupportsDropout && use_dropout) {
      return;
    }
    if (!Kernel::kSupportsBias && bias.has_value()) {
      return;
    }

    if (value.size(3) > Kernel::kMaxK || key.size(3) > Kernel::kMaxK) {
      return;
    }
    // Alignment
    if ((query.stride(2) % Kernel::kAlignmentQ) ||
        (key.stride(2) % Kernel::kAlignmentK) ||
        (value.stride(2) % Kernel::kAlignmentV)) {
      return;
    }
    // Uses too much shmem
    size_t smem_bytes = sizeof(typename Kernel::SharedStorage);
    if (smem_bytes > maxShmem) {
      return;
    }
    kernel_launched = true;

    // res = at::empty(
    //     {B, M, num_heads, Kv},
    //     query.options().dtype(
    //         CutlassToAtenDtype<typename Kernel::output_t>::atScalarType()));

    // NOTE: Should be aligned (by padding) in case M is
    // not a good number for loading during backward
    constexpr decltype(M) kAlignLSE = Kernel::kAlignLSE;
    logsumexp = at::empty(
        {seqstart_q.has_value() ? seqstart_q->size(0) - 1 : B,
         num_heads,
         compute_logsumexp ? ceil_div(max_seqlen_q, kAlignLSE) * kAlignLSE : 0},
        query.options().dtype(at::ScalarType::Float));

    typename Kernel::Params p;
    p.query_ptr = (scalar_t*)query.data_ptr();
    p.key_ptr = (scalar_t*)key.data_ptr();
    p.value_ptr = (scalar_t*)value.data_ptr();
    p.logsumexp_ptr = compute_logsumexp
        ? (typename Kernel::lse_scalar_t*)logsumexp.data_ptr()
        : nullptr;
    at::Tensor output_accum;
    if (Kernel::kNeedsOutputAccumulatorBuffer) {
      output_accum = at::empty(
          {B, M, num_heads, Kv},
          query.options().dtype(
              CutlassToAtenDtype<
                  typename Kernel::output_accum_t>::atScalarType()));
      p.output_accum_ptr =
          (typename Kernel::output_accum_t*)output_accum.data_ptr();
    } else {
      p.output_accum_ptr = nullptr;
    }
    p.output_ptr = (typename Kernel::output_t*)res.data_ptr();

    if (seqstart_q.has_value()) {
      p.seqstart_q_ptr = (int32_t*)seqstart_q->data_ptr();
      p.seqstart_k_ptr = (int32_t*)seqstart_k->data_ptr();
    }

    p.num_heads = num_heads;
    p.head_dim = query.size(3);
    p.head_dim_value = value.size(3);
    p.num_queries = max_seqlen_q;
    p.num_keys = max_seqlen_k;
    p.num_batches = seqstart_q.has_value() ? seqstart_q->size(0) - 1 : B;
    p.custom_mask_type = custom_mask_type;

    p.seqlen_k_ptr = nullptr;
    if (seqlen_k.has_value()) {
      CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(seqlen_k.value());
      TORCH_CHECK(seqlen_k->scalar_type() == at::ScalarType::Int);
      p.seqlen_k_ptr = (int32_t*)seqlen_k->data_ptr();
    }

    if (window_size.has_value()) {
      p.window_size = *window_size;
    }

    if (scale.has_value()) {
      p.scale = float(*scale);
    } else {
      p.scale = float(1.0 / std::sqrt(float(p.head_dim)));
    }

    ASSIGN_CHECK_OVERFLOW(p.q_strideB, query.stride(0));
    ASSIGN_CHECK_OVERFLOW(p.k_strideB, key.stride(0));
    ASSIGN_CHECK_OVERFLOW(p.v_strideB, value.stride(0));
    ASSIGN_CHECK_OVERFLOW(p.q_strideM, query.stride(1));
    ASSIGN_CHECK_OVERFLOW(p.k_strideM, key.stride(1));
    ASSIGN_CHECK_OVERFLOW(p.v_strideM, value.stride(1));
    ASSIGN_CHECK_OVERFLOW(p.q_strideH, query.stride(2));
    ASSIGN_CHECK_OVERFLOW(p.k_strideH, key.stride(2));
    ASSIGN_CHECK_OVERFLOW(p.v_strideH, value.stride(2));
    ASSIGN_CHECK_OVERFLOW(p.o_strideM, res.stride(1));

    if (bias.has_value()) {
      CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA((*bias));
      TORCH_CHECK(
          bias->scalar_type() == CutlassToAtenDtype<scalar_t>::atScalarType(),
          "invalid dtype for bias - should match query's dtype");
      p.attn_bias_ptr = (scalar_t*)bias->data_ptr();

      TORCH_CHECK(bias->dim() == 4, "Bias expected in BMHK format");
      TORCH_CHECK(
          bias->size(0) == query.size(0),
          "attn_bias: wrong shape (batch dimension)");
      TORCH_CHECK(
          bias->size(1) == query.size(2),
          "attn_bias: wrong shape (head dimension)");
      TORCH_CHECK(
          bias->size(2) == query.size(1),
          "attn_bias: wrong shape (seqlenQ dimension)");
      TORCH_CHECK(
          bias->size(3) == key.size(1),
          "attn_bias: wrong shape (seqlenKV dimension)");
      ASSIGN_CHECK_OVERFLOW(p.bias_strideB, bias->stride(0));
      ASSIGN_CHECK_OVERFLOW(p.bias_strideH, bias->stride(1));
      ASSIGN_CHECK_OVERFLOW(p.bias_strideM, bias->stride(2));
      TORCH_CHECK(
          bias->stride(3) == 1,
          "attn_bias: wrong alignment (last dimension must be contiguous)");
    }

    p.use_dropout = use_dropout;
    if (p.use_dropout) {
    //   p.rng_engine_inputs = rng_engine_inputs; We do not need dropout
      p.dropout_prob = dropout_p;
    }

    if (smem_bytes > 0xc000) {
      auto err = cudaFuncSetAttribute(
          kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
      XFORMERS_CHECK(
          err != cudaErrorInvalidValue,
          "This GPU does not have enough shared-memory (kernel requires ",
          smem_bytes / 1024,
          " kb)");
      AT_CUDA_CHECK(err);
    }
    auto blocks = p.getBlocksGrid();
    if (blocks.x * blocks.y * blocks.z == 0 || key.size(1) == 0) {
      res.zero_();
      return;
    }
    Kernel::check_supported(p);
    kernel_fn<<<blocks, p.getThreadsGrid(), smem_bytes, stream>>>(p);
  };

  // Dispatch to the right kernel
  DISPATCH_TYPES(query, ([&]() {
                   dispatch_cutlassF<scalar_t>(launchKernel, computeCapability);
                 }));
  TORCH_CHECK(kernel_launched, "cutlassF: no kernel found to launch!");
  AT_CUDA_CHECK(cudaGetLastError());

  // uint64_t -> int64_t bitwise casting as PyTorch don't support uint64_t
  // so just fake it as a int64_t
  int64_t seed, offset;
  if (use_dropout) {
    std::memcpy(&seed, &rng_engine_inputs.seed_, sizeof(seed));
    std::memcpy(&offset, &rng_engine_inputs.offset_.val, sizeof(offset));
  }

  return std::make_tuple(res, logsumexp, seed, offset);
}

namespace st::kernel {

template<typename T>
void xformersContextStageAttention(
	T* __restrict__ result,
	const T* __restrict__ qkvs,	// [num_tokens, 3*num_heads, head_dim]
	const float qk_scale,
	const int64_t* __restrict__ input_lens,
	const int64_t num_context_reqs,
	const int64_t* __restrict__ ith_context_req_req_index,		// WARNING currently this kernel only support ith_context_req_req_index[i] == i
	const int32_t* __restrict__ ith_context_req_token_index,	// batch_size+1
	const int64_t num_q_heads,
	const int64_t num_kv_heads,
	const int64_t head_dim,
	const int64_t num_tokens,
	const int64_t max_context_req_len
) {
	if constexpr (std::is_same_v<T, float>) {
		// Does not support float now!
		assert_whenever(0);
	} else {
		assert_whenever (num_q_heads == num_kv_heads);
		const int64_t num_heads = num_q_heads;
		auto getTensor = [](void* data, torch::IntArrayRef sizes, const std::vector<int64_t> &dimension_strides, torch::ScalarType dtype, torch::Device device = torch::kCUDA) {
			const int64_t dim = dimension_strides.size();
			std::vector<int64_t> strides(dim);
			for (int64_t i = dim-1; i >= 0; --i) {
				int64_t last_dim_stride = i == dim-1 ? 1 : strides[i+1];
				int64_t last_dim_size = i == dim-1 ? 1 : sizes[i+1];
				strides[i] = last_dim_stride * last_dim_size * dimension_strides[i];
			}
			auto options = torch::TensorOptions().dtype(dtype).device(device);
			return torch::from_blob(data, sizes, strides, [](void*) {}, options);
		};
		at::Tensor q_tensor = getTensor(
			const_cast<T*>(qkvs),
			{ 1, num_tokens, num_heads, head_dim },
			{ 1, 3, 1, 1 },
			torch::kHalf
		);
		at::Tensor k_tensor = getTensor(
			const_cast<T*>(qkvs) + num_heads * head_dim,
			{ 1, num_tokens, num_heads, head_dim },
			{ 1, 3, 1, 1 },
			torch::kHalf
		);
		at::Tensor v_tensor = getTensor(
			const_cast<T*>(qkvs) + 2 * num_heads * head_dim,
			{ 1, num_tokens, num_heads, head_dim },
			{ 1, 3, 1, 1 },
			torch::kHalf
		);
		at::Tensor seqstart = getTensor(
			const_cast<int32_t*>(ith_context_req_token_index),
			{ num_context_reqs + 1 },
			{ 1 },
			torch::kInt32
		);
		at::Tensor result_tensor = getTensor(
			result,
			{ 1, num_tokens, num_heads, head_dim },
			{ 1, 1, 1, 1 },
			torch::kHalf
		);
		efficient_attention_forward_cutlass(
			q_tensor,
			k_tensor,
			v_tensor,
			c10::nullopt,
			seqstart,
			seqstart,
			max_context_req_len,
			0.0,
			false,
			2,	// TO CHECK
			(double)qk_scale,
			c10::nullopt,
			c10::nullopt,
			result_tensor
		);
		sync_check_cuda_error();
	}
}

#define INSTANTIALIZE_XFORMERS_CONTEXT_STAGE_ATTENTION(T) \
	template void xformersContextStageAttention<T>( \
		T* __restrict__ result, \
		const T* __restrict__ qkvs,	\
		const float qk_scale, \
		const int64_t* __restrict__ input_lens, \
		const int64_t num_context_reqs, \
		const int64_t* __restrict__ ith_context_req_req_index, \
		const int32_t* __restrict__ ith_context_req_token_index, \
		const int64_t num_q_heads, \
		const int64_t num_kv_heads, \
		const int64_t head_dim, \
		const int64_t num_tokens, \
		const int64_t max_context_req_len \
	);

INSTANTIALIZE_XFORMERS_CONTEXT_STAGE_ATTENTION(float)
INSTANTIALIZE_XFORMERS_CONTEXT_STAGE_ATTENTION(half)

}	// namespace st::kernel
