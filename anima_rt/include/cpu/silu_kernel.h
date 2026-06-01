// SiLU (Swish) activation, extracted from PyTorch aten/src/ATen/native/cpu/Activation.cpp
// Original copyright: Copyright (c) 2016- Facebook, Inc. See NOTICE.
#pragma once
#include <cmath>

namespace anima {
namespace cpu {

inline float silu_scalar(float x) {
    return x / (1.0f + std::exp(-x));
}

// FP32 in → FP32 out, element-wise.
inline void silu_kernel(const float* x, float* y, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        y[i] = silu_scalar(x[i]);
    }
}

} // namespace cpu
} // namespace anima
