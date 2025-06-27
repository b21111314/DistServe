#include "attention_ref.h"

#include <iostream>
#include <torch/torch.h>

#include "../kernel/attention_ref.h"
#include "../kernel/kvcache_mgmt_ref.h"

namespace st::reference::layer {

using torch::Tensor;

Tensor rmsnormRef(const Tensor& input, const Tensor& weight, float eps) {
    // input: [num_tokens, hidden_size]
    // weight: [num_heads, head_dim] or [hidden_size]

    auto x = input.to(torch::kFloat32);  // ensure float for precision
    auto variance = x.pow(2).mean(-1, true);  // [num_tokens, 1]
    auto normed = input / torch::sqrt(variance + eps);  // [num_tokens, hidden_size]

    if (weight.dim() == 1) {
        // Normal case: weight is [hidden_size]
        return normed * weight.unsqueeze(0);  // [1, hidden_size] broadcast
    } else if (weight.dim() == 2) {
        // Qwen3 case: weight is [num_heads, head_dim]
        int64_t num_heads = weight.size(0);
        int64_t head_dim = weight.size(1);
        int64_t num_tokens = normed.size(0);

        // reshape [num_tokens, hidden_size] -> [num_tokens, num_heads, head_dim]
        normed = normed.view({num_tokens, num_heads, head_dim});
        auto normed_weighted = normed * weight.view({1, num_heads, head_dim});
        return normed_weighted.view({num_tokens, num_heads * head_dim});
    } else {
        throw std::runtime_error("Unsupported weight shape in rmsnormRef");
    }
}

void attentionLayerRef(
	Tensor &result,		// [num_tokens, hidden_size]
	Tensor &k_cache,	// [num_blocks, num_layers, num_kv_heads, block_size, head_dim]
	Tensor &v_cache,	// [num_blocks, num_layers, num_kv_heads, block_size, head_dim]

	const Tensor &input,	// [num_tokens, hidden_size]
	const Tensor &input_len_cpu,	// [num_reqs]
	const Tensor &is_context_stage_cpu,	// [num_reqs]
	const Tensor &block_table_cpu,	// [num_reqs, max_num_block_per_seq]

	const float qk_scale, 

	const Tensor &qkv_weight_kernel,	// [hidden_size, num_q_heads + 2*num_kv_heads, head_dim]
	const Tensor &qkv_weight_bias,		// [num_q_heads+2*num_kv_heads, head_dim]
	const Tensor &out_weight_kernel,	// [num_q_heads, head_dim, hidden_size]
	const Tensor &out_weight_bias,		// [hidden_size]
	const Tensor &q_norm_weight,        // 新增
    const Tensor &k_norm_weight,         // 新增

	const int64_t layer_id
) {
	const int64_t num_tokens = input.size(0);
	const int64_t num_q_heads = out_weight_kernel.size(0);
	const int64_t num_kv_heads = k_cache.size(2);
	const int64_t head_dim = qkv_weight_kernel.size(2);
	const int64_t hidden_size = qkv_weight_kernel.size(0);

	// Step 1. QKV GEMM
	Tensor qkvs;

	if (q_norm_weight.defined() && k_norm_weight.defined()) {
		// === Qwen3 path: Apply RMSNorm on Q/K separately, then GEMM ===
		Tensor q_input = st::reference::kernel::rmsnormRef(input, q_norm_weight, 1e-5);
		Tensor k_input = st::reference::kernel::rmsnormRef(input, k_norm_weight, 1e-5);

		int64_t q_size = num_q_heads * head_dim;
		int64_t k_size = num_kv_heads * head_dim;
		int64_t v_size = num_kv_heads * head_dim;

		Tensor Wq = qkv_weight_kernel.slice(1, 0, q_size / head_dim);
		Tensor Wk = qkv_weight_kernel.slice(1, q_size / head_dim, (q_size + k_size) / head_dim);
		Tensor Wv = qkv_weight_kernel.slice(1, (q_size + k_size) / head_dim, (q_size + k_size + v_size) / head_dim);

		Tensor bq = qkv_weight_bias.slice(0, 0, q_size / head_dim);
		Tensor bk = qkv_weight_bias.slice(0, q_size / head_dim, (q_size + k_size) / head_dim);
		Tensor bv = qkv_weight_bias.slice(0, (q_size + k_size) / head_dim, (q_size + k_size + v_size) / head_dim);

		Tensor Q = torch::matmul(q_input, Wq.view({hidden_size, q_size})) + bq.view({q_size});
		Tensor K = torch::matmul(k_input, Wk.view({hidden_size, k_size})) + bk.view({k_size});
		Tensor V = torch::matmul(input, Wv.view({hidden_size, v_size})) + bv.view({v_size});

		Q = Q.view({num_tokens, num_q_heads, head_dim});
		K = K.view({num_tokens, num_kv_heads, head_dim});
		V = V.view({num_tokens, num_kv_heads, head_dim});

		qkvs = torch::cat({Q, K, V}, 1);  // [num_tokens, num_q_heads + 2*num_kv_heads, head_dim]
	} else {
		// === Default path: 1-shot QKV GEMM ===
		qkvs = torch::matmul(input, qkv_weight_kernel.view({hidden_size, (num_q_heads+2*num_kv_heads)*head_dim}));
		qkvs += qkv_weight_bias.view({(num_q_heads+2*num_kv_heads)*head_dim});
		qkvs = qkvs.view({num_tokens, num_q_heads+2*num_kv_heads, head_dim});
	}

	// Step 2. Attention
	// result: [num_tokens, hidden_size]
	st::reference::kernel::attentionKernelRef(
		result,
		k_cache,
		v_cache,

		qkvs,
		qk_scale,
		block_table_cpu,
		input_len_cpu,
		is_context_stage_cpu,

		true, true
	);

	// Step 3. Save KV Cache
	st::reference::kernel::saveContextStageKVCacheKernelRef(
		k_cache,
		v_cache,
		qkvs,
		block_table_cpu,
		input_len_cpu,
		is_context_stage_cpu,
		layer_id
	);

	// Step 4. Output GEMM
	result = torch::matmul(result, out_weight_kernel.view({hidden_size, hidden_size})) + out_weight_bias;
}

}