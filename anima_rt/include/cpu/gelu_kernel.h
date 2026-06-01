// GELU activation — exact erf-based, extracted from PyTorch aten/src/ATen/native/cpu/Gelu.h
// Original copyright: Copyright (c) 2016- Facebook, Inc. See NOTICE.
#pragma once
#include <cmath>

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440
#endif

namespace anima {
namespace cpu {

inline float gelu_scalar(float x) {
    const float kAlpha = (float)M_SQRT1_2;
    return x * 0.5f * (1.0f + std::erf(x * kAlpha));
}

// FP32 in → FP32 out, element-wise.
inline void gelu_kernel(const float* x, float* y, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        y[i] = gelu_scalar(x[i]);
    }
}

} // namespace cpu
} // namespace anima
