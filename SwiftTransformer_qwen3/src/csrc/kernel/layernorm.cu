#include "layernorm.h"

#include <cassert>
#include <cstdlib>

#include "util/cuda_utils.h"
#include "kernel/reduction.cuh"
#include "util/debug_utils.h"

namespace st::kernel {

constexpr int WARP_SIZE = 32;
constexpr int NUM_THREADS = 256;
constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;

template<typename T, bool HAVE_PRE_LAYERNORM_BIAS>
__global__ void layernormKernel(
	T* __restrict__ out,			// [num_tokens, hidden_size]
	const T* __restrict__ input,	// [num_tokens, hidden_size]
	const T* __restrict__ weight,	// [hidden_size]
	const T* __restrict__ bias,		// [hidden_size]
	const float epsilon,
	const int64_t num_tokens,
	const int64_t hidden_size,
	T* __restrict__ biased_input,	// [num_tokens, hidden_size]
	const T* __restrict__ pre_layernorm_bias	// [hidden_size]
) {
	typedef std::conditional_t<std::is_same<T, half>::value, half2, float2> T2;
	assert(reinterpret_cast<uintptr_t>(input) % sizeof(T2) == 0);
    assert(reinterpret_cast<uintptr_t>(weight) % sizeof(T2) == 0);
    assert(reinterpret_cast<uintptr_t>(out) % sizeof(T2) == 0);

	extern __shared__ float shared_mem[];
	T2* input_buf = (T2*)shared_mem;

	__shared__ float s_mean, s_variance;
	float mean = 0.0, variance = 0.0;

	for (int64_t idx = threadIdx.x; idx < hidden_size / 2; idx += blockDim.x) {
		T2 elem = ((T2*)input)[blockIdx.x * hidden_size / 2 + idx];
		if constexpr (HAVE_PRE_LAYERNORM_BIAS) {
			if (pre_layernorm_bias != nullptr && biased_input != nullptr) {
			    const T2 pre_bias = ((T2*)pre_layernorm_bias)[blockIdx.x * hidden_size / 2 + idx];
			    elem.x += pre_bias.x;
			    elem.y += pre_bias.y;
			    ((T2*)biased_input)[blockIdx.x * hidden_size / 2 + idx] = elem;
			}
		}
		input_buf[idx] = elem;
		mean += float(elem.x) + float(elem.y);
		variance += float(elem.x) * float(elem.x) + float(elem.y) * float(elem.y);
	}

	for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
		mean += __shfl_down_sync(0xffffffff, mean, offset);
		variance += __shfl_down_sync(0xffffffff, variance, offset);
	}

	static __shared__ float reduction_wksp[2][NUM_WARPS];
	if ((threadIdx.x & 31) == 0) {
		reduction_wksp[0][threadIdx.x >> 5] = mean;
		reduction_wksp[1][threadIdx.x >> 5] = variance;
	}
	__syncthreads();

	if (threadIdx.x < NUM_WARPS) {
		mean = reduction_wksp[0][threadIdx.x];
		variance = reduction_wksp[1][threadIdx.x];
	}
	for (int offset = NUM_WARPS / 2; offset > 0; offset /= 2) {
		mean += __shfl_down_sync(0xffffffff, mean, offset);
		variance += __shfl_down_sync(0xffffffff, variance, offset);
	}

	if (threadIdx.x == 0) {
		float hidden_size_fp = float(hidden_size);
		s_mean = mean / hidden_size_fp;
		s_variance = rsqrtf(variance / hidden_size_fp - s_mean * s_mean + epsilon);
	}
	__syncthreads();

	T final_mean = (T)s_mean;
	T final_variance = (T)s_variance;
	for (int64_t idx = threadIdx.x; idx < hidden_size / 2; idx += blockDim.x) {
		T2 x = input_buf[idx];
		T2 weight_elem = ((T2*)weight)[idx];
		T2 bias_elem = {0.0f, 0.0f};
        if (bias != nullptr) {
	        bias_elem = ((T2*)bias)[idx];
        }
		((T2*)out)[blockIdx.x * hidden_size / 2 + idx] = {
			((x.x - final_mean) * final_variance) * weight_elem.x + bias_elem.x,
			((x.y - final_mean) * final_variance) * weight_elem.y + bias_elem.y
		};
	}
}

template<typename T>
void layernorm(
	T* out,
	const T* input,
	const T* weight,
	const T* bias,
	const float epsilon,
	const int64_t num_tokens,
	const int64_t hidden_size,
	T* biased_input,
	const T* pre_layernorm_bias
) {
	// 防止 misaligned T2 访问！
	if (hidden_size % 2 != 0) {
        throw std::invalid_argument("hidden_size must be a multiple of 2");
    }
	dim3 grid(num_tokens);
	dim3 block(NUM_THREADS);

	using T2 = std::conditional_t<std::is_same<T, half>::value, half2, float2>;
	size_t smem_size = (hidden_size / 2) * sizeof(T2)+2 * NUM_WARPS * sizeof(float);

	if (pre_layernorm_bias == nullptr) {
		assert(biased_input == nullptr);
		layernormKernel<T, false><<<grid, block, smem_size>>>(
			out, input, weight, bias, epsilon, num_tokens, hidden_size,
			nullptr, nullptr
		);
	} else {
		assert(biased_input != nullptr);
		layernormKernel<T, true><<<grid, block, smem_size>>>(
			out, input, weight, bias, epsilon, num_tokens, hidden_size,
			biased_input, pre_layernorm_bias
		);
	}
}

template void layernorm(
	float* out, const float* input,
	const float* weight, const float* bias, const float epsilon,
	const int64_t num_tokens, const int64_t hidden_size,
	float* biased_input, const float* pre_layernorm_bias
);

template void layernorm(
	half* out, const half* input,
	const half* weight, const half* bias, const float epsilon,
	const int64_t num_tokens, const int64_t hidden_size,
	half* biased_input, const half* pre_layernorm_bias
);

}  // namespace st::kernel