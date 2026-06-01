// Welford algorithm for numerically-stable mean & variance.
// Simplified scalar version derived from PyTorch
// aten/src/ATen/native/cpu/moments_utils.h (RowwiseMoments).
// Original copyright: Copyright (c) 2016- Facebook, Inc. See NOTICE.
#pragma once
#include <cmath>
#include <utility>

namespace anima {
namespace cpu {

// Compute mean and variance of a row using Welford's online algorithm.
// Returns {mean, variance} (variance = population variance, ddof=0).
inline std::pair<float, float> rowwise_moments(const float* x, int64_t n) {
    float mean = 0.0f;
    float m2   = 0.0f;  // sum of squared diffs
    int64_t count = 0;

    for (int64_t i = 0; i < n; i++) {
        float v = x[i];
        count++;
        float delta = v - mean;
        mean += delta / (float)count;
        float delta2 = v - mean;
        m2 += delta * delta2;
    }

    float variance = (count > 0) ? m2 / (float)count : 0.0f;
    return {mean, variance};
}

// RMS (root-mean-square) of a row. Uses the same Welford pattern but for x^2.
inline float rowwise_rms(const float* x, int64_t n, float eps) {
    float mean_sq = 0.0f;
    int64_t count = 0;

    for (int64_t i = 0; i < n; i++) {
        float v = x[i] * x[i];  // x^2
        count++;
        float delta = v - mean_sq;
        mean_sq += delta / (float)count;
    }

    return 1.0f / std::sqrt(mean_sq + eps);
}

} // namespace cpu
} // namespace anima
