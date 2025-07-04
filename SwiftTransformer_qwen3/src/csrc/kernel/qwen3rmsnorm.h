#pragma once

#include <cstdint>

namespace st::kernel {

// 声明模板函数
template <typename T>
void qwen3_rmsnorm_launch(
    T* out,
    const T* input,
    const T* weight,
    int64_t num_tokens,
    int64_t num_heads,
    int64_t head_dim
);

}  // namespace st::kernel