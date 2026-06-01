// RMSNorm kernel — dedicated implementation.
// PyTorch implements RMSNorm as a composite (pow→mean→rsqrt→mul),
// not a dedicated kernel. We write a fused kernel using the same math.
// Formula: Y = X * rsqrt(mean(X^2) + eps) * weight
#pragma once
#include "welford.h"

namespace anima {
namespace cpu {

// RMSNorm along last dimension.
// X[M,D], weight[D] → Y[M,D]
inline void rmsnorm_kernel(const float* X, const float* weight,
                           float* Y, int64_t M, int64_t D, float eps) {
    for (int64_t row = 0; row < M; row++) {
        const float* x_row = X + row * D;
        float*       y_row = Y + row * D;

        float rms = rowwise_rms(x_row, D, eps);

        for (int64_t j = 0; j < D; j++) {
            float v = x_row[j] * rms;
            if (weight) v *= weight[j];
            y_row[j] = v;
        }
    }
}

} // namespace cpu
} // namespace anima
