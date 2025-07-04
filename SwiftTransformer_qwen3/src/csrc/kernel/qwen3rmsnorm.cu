#include "kernel/qwen3rmsnorm.h"
#include <cuda_fp16.h>
#include <math.h>
#include "util/cuda_utils.h"

namespace st::kernel {

template <typename T>
__global__ void rmsnorm_qwen3_kernel(
    T* out,
    const T* input,
    const T* weight,
    int64_t num_tokens,
    int64_t num_heads,
    int64_t head_dim
) {
    int token_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid = threadIdx.x;

    if (tid >= head_dim) return;

    int64_t offset = token_idx * num_heads * head_dim + head_idx * head_dim;

    float sum = 0.0f;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float val = static_cast<float>(input[offset + i]);
        sum += val * val;
    }

    __shared__ float shared_sum;
    if (threadIdx.x == 0) shared_sum = 0.0f;
    __syncthreads();
    atomicAdd(&shared_sum, sum);
    __syncthreads();

    float rms = rsqrtf(shared_sum / head_dim + 1e-5f);
    float val = static_cast<float>(input[offset + tid]);
    float w = static_cast<float>(weight[head_idx * head_dim + tid]);
    out[offset + tid] = static_cast<T>(val * rms * w);
}

template <typename T>
void qwen3_rmsnorm_launch(
    T* out,
    const T* input,
    const T* weight,
    int64_t num_tokens,
    int64_t num_heads,
    int64_t head_dim
) {
    dim3 grid(num_tokens, num_heads);
    dim3 block(std::min<int64_t>(256, head_dim));
    rmsnorm_qwen3_kernel<T><<<grid, block>>>(
        out, input, weight, num_tokens, num_heads, head_dim
    );
    sync_check_cuda_error();
}

// 显式实例化
template void qwen3_rmsnorm_launch<float>(
    float*, const float*, const float*, int64_t, int64_t, int64_t);
template void qwen3_rmsnorm_launch<__half>(
    __half*, const __half*, const __half*, int64_t, int64_t, int64_t);

}  // namespace st::kernel