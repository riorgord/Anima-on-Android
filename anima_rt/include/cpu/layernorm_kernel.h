// LayerNorm kernel, derived from PyTorch
// aten/src/ATen/native/cpu/layer_norm_kernel.cpp (LayerNormKernelImplInternal).
// Two-pass algorithm: Welford moments → normalize + affine.
// Original copyright: Copyright (c) 2016- Facebook, Inc. See NOTICE.
#pragma once
#include "welford.h"

namespace anima {
namespace cpu {

// LayerNorm along last dimension.
// X[M,D], gamma[D]|null, beta[D]|null → Y[M,D]
inline void layernorm_kernel(const float* X,
                             const float* gamma,
                             const float* beta,
                             float* Y,
                             int64_t M, int64_t D, float eps) {
    for (int64_t row = 0; row < M; row++) {
        const float* x_row = X + row * D;
        float*       y_row = Y + row * D;

        // Pass 1: compute mean & rstd via Welford
        auto [mean, var] = rowwise_moments(x_row, D);
        float rstd = 1.0f / std::sqrt(var + eps);

        // Pass 2: normalize + affine
        for (int64_t j = 0; j < D; j++) {
            float v = (x_row[j] - mean) * rstd;
            if (gamma) v *= gamma[j];
            if (beta)  v += beta[j];
            y_row[j] = v;
        }
    }
}

} // namespace cpu
} // namespace anima
