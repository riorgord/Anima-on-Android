/* Flash Attention — tiled online softmax algorithm.
 *
 * Extracted from PyTorch aten/src/ATen/native/cpu/FlashAttentionKernel.cpp
 * This is the algorithm PT 2.11 uses by default for F.scaled_dot_product_attention.
 *
 * The math backend (matmul→softmax→matmul) produces DIFFERENT results (0.96 max_err
 * per attention call).  Flash attention uses online softmax which changes accumulation
 * order → different floating-point results → different image output.
 *
 * Algorithm (scalar, single-threaded):
 *   For each query block:
 *     Initialize output=0, max_vals=-inf, sum_vals=0
 *     For each KV block:
 *       scores = Q_block @ K_block^T
 *       scores *= scale
 *       Apply causal mask
 *       For each row:
 *         new_max = max(old_max, row_max(scores))
 *         output_row *= exp(old_max - new_max)
 *         sum_row = sum_row * exp(old_max - new_max) + sum(exp(scores_row - new_max))
 *         output_row += softmax_row @ V_block
 *         old_max = new_max
 *     output_row /= sum_row
 */

#pragma once
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cfloat>

namespace anima {
namespace cpu {

/* ── GEMM function pointer (matches anima_rt_run_gemm_fp32) ──── */
typedef bool (*gemm_fn)(const float* A, const float* B, float* C, int M, int N, int K);

/* ── Flash attention: tiled online softmax ──────────────────────────
 *
 * Q, K, V: [B*H, S, D]  (all heads + batch flattened)
 * S_q, S_kv: query and key/value sequence lengths
 * D: head dimension
 * scale: 1/sqrt(D)
 * is_causal: apply causal mask
 * gemm: GEMM function for matmul (OpenBLAS or pure C++)
 *
 * Tiling parameters (matching PT's FlashAttentionKernel.cpp):
 *   qSplitSize = (S_q >= 768) ? 256 : (S_q >= 192) ? 64 : 32
 *   kvSplitSize = 512
 *
 * Temporary buffer needed (caller allocates):
 *   scores: [qSplitSize * kvSplitSize]
 *   output: same as function output
 */
inline void flash_attention(
    const float* Q, const float* K, const float* V,
    float* output,           // [BH * S_q * D]
    int BH, int S_q, int S_kv, int D,
    float scale, bool is_causal)
{
    // Determine tiling (matching PT's logic)
    int qSplitSize = (S_q >= 768) ? 256 : (S_q >= 192) ? 64 : 32;
    int kvSplitSize = 512;

    // Limit split sizes to actual sequence lengths
    if (qSplitSize > S_q) qSplitSize = S_q;
    if (kvSplitSize > S_kv) kvSplitSize = S_kv;

    int qSlice = (S_q + qSplitSize - 1) / qSplitSize;
    int kvSlice = (S_kv + kvSplitSize - 1) / kvSplitSize;

    // Allocate per-thread temp buffers
    float* scores = new float[qSplitSize * kvSplitSize];
    float* max_vals = new float[qSplitSize];
    float* sum_vals = new float[qSplitSize];
    // output is pre-allocated by caller

    for (int bh = 0; bh < BH; bh++) {
        const float* Q_head = Q + bh * S_q * D;
        const float* K_head = K + bh * S_kv * D;
        const float* V_head = V + bh * S_kv * D;
        float* out_head = output + bh * S_q * D;

        for (int qi = 0; qi < qSlice; qi++) {
            int qStart = qi * qSplitSize;
            int qBlockSize = (qStart + qSplitSize <= S_q) ? qSplitSize : (S_q - qStart);

            // Initialize per-block state
            for (int r = 0; r < qBlockSize; r++) {
                max_vals[r] = -INFINITY;
                sum_vals[r] = 0.0f;
                for (int d = 0; d < D; d++) {
                    out_head[(qStart + r) * D + d] = 0.0f;
                }
            }

            int num_keys = is_causal ? std::min(qStart + qBlockSize, S_kv) : S_kv;

            for (int kvi = 0; kvi < kvSlice; kvi++) {
                int kvStart = kvi * kvSplitSize;
                int kvBlockSize = (kvStart + kvSplitSize <= S_kv) ? kvSplitSize : (S_kv - kvStart);
                if (kvStart >= num_keys) break;

                // Adjust kvBlockSize for causal masking
                int effective_kvSize = kvBlockSize;
                if (is_causal) {
                    int last_key = num_keys - kvStart;
                    if (last_key < kvBlockSize) effective_kvSize = last_key;
                    if (effective_kvSize <= 0) break;
                }

                // ── Q_block @ K_block^T → scores [qBlockSize, kvBlockSize] ──
                const float* Q_block = Q_head + qStart * D;
                const float* K_block = K_head + kvStart * D;
                // gemm_fp32: C[M,N] = A[M,K] @ B^T[N,K]
                // We want scores[qBlockSize, kvBlockSize] = Q_block[qBlockSize,D] @ K_block[kvBlockSize,D]^T
                for (int i = 0; i < qBlockSize; i++) {
                    for (int j = 0; j < effective_kvSize; j++) {
                        float s = 0.0f;
                        for (int d = 0; d < D; d++) {
                            s += Q_block[i * D + d] * K_block[j * D + d];
                        }
                        scores[i * kvBlockSize + j] = s * scale;
                    }
                    // Fill unused with -inf (for causal or padding)
                    for (int j = effective_kvSize; j < kvBlockSize; j++) {
                        scores[i * kvBlockSize + j] = -INFINITY;
                    }
                }

                // ── Apply causal mask ──
                if (is_causal) {
                    for (int i = 0; i < qBlockSize; i++) {
                        int global_row = qStart + i;
                        for (int j = 0; j < kvBlockSize; j++) {
                            int global_col = kvStart + j;
                            if (global_col > global_row) {
                                scores[i * kvBlockSize + j] = -INFINITY;
                            }
                        }
                    }
                }

                // ── Online softmax update ──
                for (int i = 0; i < qBlockSize; i++) {
                    // Find row max
                    float row_max = -INFINITY;
                    for (int j = 0; j < kvBlockSize; j++) {
                        if (scores[i * kvBlockSize + j] > row_max)
                            row_max = scores[i * kvBlockSize + j];
                    }

                    // If all -inf, skip this row
                    if (row_max == -INFINITY) continue;

                    float new_max = (max_vals[i] > row_max) ? max_vals[i] : row_max;
                    float exp_diff = (max_vals[i] == -INFINITY) ? 0.0f :
                                     std::exp(max_vals[i] - new_max);

                    // Rescale existing output
                    if (kvStart > 0) {  // not first KV block
                        for (int d = 0; d < D; d++) {
                            out_head[(qStart + i) * D + d] *= exp_diff;
                        }
                    }

                    // Update sum
                    sum_vals[i] = sum_vals[i] * exp_diff;

                    // Compute exp(scores - new_max) and accumulate sum
                    float row_sum = 0.0f;
                    for (int j = 0; j < kvBlockSize; j++) {
                        float exp_val = std::exp(scores[i * kvBlockSize + j] - new_max);
                        scores[i * kvBlockSize + j] = exp_val;
                        row_sum += exp_val;
                    }
                    sum_vals[i] += row_sum;

                    // Accumulate: output += P @ V_block
                    for (int j = 0; j < kvBlockSize; j++) {
                        float p_val = scores[i * kvBlockSize + j];
                        if (p_val == 0.0f) continue;
                        const float* V_row = V_head + (kvStart + j) * D;
                        for (int d = 0; d < D; d++) {
                            out_head[(qStart + i) * D + d] += p_val * V_row[d];
                        }
                    }

                    max_vals[i] = new_max;
                }
            }

            // ── Final normalization: output /= sum ──
            for (int i = 0; i < qBlockSize; i++) {
                if (sum_vals[i] > 0.0f) {
                    float inv_sum = 1.0f / sum_vals[i];
                    for (int d = 0; d < D; d++) {
                        out_head[(qStart + i) * D + d] *= inv_sum;
                    }
                }
            }
        }
    }

    delete[] scores;
    delete[] max_vals;
    delete[] sum_vals;
}

} // namespace cpu
} // namespace anima
