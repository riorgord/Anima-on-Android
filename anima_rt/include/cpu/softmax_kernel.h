// Softmax along last dimension, extracted from PyTorch
// aten/src/ATen/native/cpu/SoftMaxKernel.cpp (the _vec_softmax_lastdim path).
// Original copyright: Copyright (c) 2016- Facebook, Inc. See NOTICE.
#pragma once
#include <cmath>
#include <algorithm>
#include <memory>

namespace anima {
namespace cpu {

// Softmax along the last dimension.
// Input:  X[outer_size, dim_size]  row-major
// Output: Y[outer_size, dim_size]  = softmax(X, dim=-1)
inline void softmax_kernel(const float* X, float* Y,
                           int64_t outer_size, int64_t dim_size) {
    for (int64_t i = 0; i < outer_size; i++) {
        const float* row_in  = X + i * dim_size;
        float*       row_out = Y + i * dim_size;

        // 1. Find max for numerical stability
        float max_val = row_in[0];
        for (int64_t j = 1; j < dim_size; j++) {
            if (row_in[j] > max_val) max_val = row_in[j];
        }

        // 2. Compute exp(x - max) and sum
        float sum = 0.0f;
        for (int64_t j = 0; j < dim_size; j++) {
            float v = std::exp(row_in[j] - max_val);
            row_out[j] = v;
            sum += v;
        }

        // 3. Normalize
        float inv_sum = 1.0f / sum;
        for (int64_t j = 0; j < dim_size; j++) {
            row_out[j] *= inv_sum;
        }
    }
}

} // namespace cpu
} // namespace anima
