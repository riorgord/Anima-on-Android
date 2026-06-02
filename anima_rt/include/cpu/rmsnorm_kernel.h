// RMSNorm kernel — exact formula from PyTorch layer_norm.cpp:265 rms_norm_composite.
// PT doesn't have a dedicated kernel; it composites: rsqrt(mean(x^2) + eps) * weight.
// We replicate this EXACT formula (sum/N instead of Welford) for bit-identical results.
// Formula: Y = X * rsqrt(mean(X^2) + eps) * weight
#pragma once
#include <cmath>

namespace anima {
namespace cpu {

inline void rmsnorm_kernel(const float* X, const float* weight,
                           float* Y, int64_t M, int64_t D, float eps) {
    for (int64_t row = 0; row < M; row++) {
        const float* x_row = X + row * D;
        float*       y_row = Y + row * D;

        // PT formula: mean(x^2) — simple sum/N, NOT Welford
        float sum_sq = 0.0f;
        for (int64_t j = 0; j < D; j++) {
            float v = x_row[j];
            sum_sq += v * v;
        }
        float rms = sum_sq / (float)D;
        float inv_rms = 1.0f / std::sqrt(rms + eps);

        for (int64_t j = 0; j < D; j++) {
            float v = x_row[j] * inv_rms;
            if (weight) v *= weight[j];
            y_row[j] = v;
        }
    }
}

} // namespace cpu
} // namespace anima
