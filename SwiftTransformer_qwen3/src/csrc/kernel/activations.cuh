#pragma once

#include <cassert>
#include "activation_types.h"
#include "util/cuda_utils.h"

namespace st::kernel {

template<typename T, ActivationType activation_type>
__forceinline__ __device__ T applyActivation(const T &x) {
	if constexpr (activation_type == ActivationType::RELU) {
		return x > (T)0 ? x : (T)0;
	}
	else if constexpr (activation_type == ActivationType::SILU) {
		return (T)((float)x / (1.0f + __expf((float)-x)));
	}
	else if constexpr (activation_type == ActivationType::GELU) {
		const float x3 = (float) (x * x * x);
		const T t = (T) tanhf((T) (0.79788456f * (float) (x + (T) (0.044715f * x3))));
		return ((T) 0.5) * x * (((T) 1.0) + t);
	}
	else {
		assert(false);
	}
}

__device__ __forceinline__ half applySwiGLU(const half& x1, const half& x2) {
	float a = __half2float(x1);
	float b = __half2float(x2);
	float silu = a / (1.0f + __expf(-a));
	return __float2half(silu * b);
}

template<typename T>
__device__ __forceinline__ T applySwiGLU(const T& x1, const T& x2) {
	float a = (float)x1;
	float b = (float)x2;
	float silu = a / (1.0f + __expf(-a));
	return (T)(silu * b);
}

}