/* SDPA math backend — exact formula from PyTorch
 *   aten/src/ATen/native/transformers/attention.cpp:826
 *   _scaled_dot_product_attention_math
 *
 * Formula:
 *   scale = 1/sqrt(head_dim)
 *   Q = Q * scale,  K = K * scale
 *   attn = Q @ K^T              [B*H, S_q, S_kv]
 *   if causal: attn += causal_mask
 *   attn = softmax(attn, dim=-1)
 *   output = attn @ V           [B*H, S_q, D]
 *
 * We compose this from our existing GEMM + Softmax kernels.
 */
#pragma once
#include <cmath>
#include <cstring>
#include <cfloat>
#include <algorithm>

namespace anima {
namespace cpu {

/* ── Causal mask helper ───────────────────────────────────────────── */
// Fills upper triangle of mask[S_q, S_kv] with -inf, lower with 0.
inline void fill_causal_mask(float* mask, int S_q, int S_kv) {
    for (int i = 0; i < S_q; i++) {
        for (int j = 0; j < S_kv; j++) {
            mask[i * S_kv + j] = (j > i) ? -INFINITY : 0.0f;
        }
    }
}

/* ── Scale tensor in-place: x[i] *= scale ───────────────────────── */
inline void scale_inplace(float* x, int64_t n, float scale) {
    for (int64_t i = 0; i < n; i++) x[i] *= scale;
}

/* ── Add mask to attention scores: attn[i] += mask[i] ────────────── */
inline void add_mask_inplace(float* attn, const float* mask, int64_t n) {
    for (int64_t i = 0; i < n; i++) attn[i] += mask[i];
}

/* ── SDPA math backend ───────────────────────────────────────────────
 *
 * Q, K, V: shape [B*H, S, D] (batched: all heads & batch flattened)
 * S_q, S_kv: query and key/value sequence lengths
 * output: [B*H, S_q, D]
 *
 * The caller must provide GEMM and Softmax function pointers.
 * Temporary buffers:
 *   - QK buffer: [B*H, S_q, S_kv]  (attn scores)
 *   - causal_mask: [S_q, S_kv] if causal, else nullptr
 * Memory for QK buffer must be pre-allocated by caller.
 */
inline void sdpa_math(
    const float* Q, const float* K, const float* V,
    float* output,
    float* qk_buffer,       // temp: [BH * S_q * S_kv] floats
    float* causal_mask_buf, // temp: [S_q * S_kv] floats (only if causal)
    int BH, int S_q, int S_kv, int D,
    float scale,
    bool is_causal,
    // Function pointers — provided by the caller (our GEMM/Softmax backends)
    bool (*gemm)(const float*, const float*, float*, int, int, int),
    bool (*softmax)(const float*, float*, int64_t, int64_t)
) {
    const int64_t Q_size = (int64_t)BH * S_q * D;
    const int64_t K_size = (int64_t)BH * S_kv * D;
    const int64_t V_size = (int64_t)BH * S_kv * D;
    const int64_t QK_size = (int64_t)BH * S_q * S_kv;
    const int64_t out_size = (int64_t)BH * S_q * D;

    // Step 1: Copy Q → Q_scaled, K → K_scaled (we can scale in-place via copies)
    // For now, we modify in-place via temporary copies — caller provides buffers
    // Actually, to avoid modifying inputs, we work directly.
    // But we need to scale Q and K. Let's make copies.
    // For efficiency, the caller can pre-scale Q and K before calling.
    // For simplicity: scale Q_copy and K_copy.
    float* Q_scaled = new float[Q_size];
    float* K_scaled = new float[K_size];
    std::memcpy(Q_scaled, Q, Q_size * sizeof(float));
    std::memcpy(K_scaled, K, K_size * sizeof(float));
    scale_inplace(Q_scaled, Q_size, scale);
    scale_inplace(K_scaled, K_size, scale);

    // Step 2: attn = Q_scaled @ K_scaled^T
    // Q_scaled: [BH*S_q, D] in practice but we need batched matmul.
    // Each head is independent: Q_h[S_q, D] @ K_h^T[D, S_kv] = attn_h[S_q, S_kv]
    // We process each head separately and call gemm for each.
    for (int h = 0; h < BH; h++) {
        const float* Q_h = Q_scaled + h * S_q * D;
        const float* K_h = K_scaled + h * S_kv * D;
        float* attn_h = qk_buffer + h * S_q * S_kv;

        // gemm: attn_h[S_q, S_kv] = Q_h[S_q, D] @ K_h^T[S_kv, D]
        if (!gemm(Q_h, K_h, attn_h, S_q, S_kv, D)) {
            // Fallback: manual matmul
            for (int i = 0; i < S_q; i++) {
                for (int j = 0; j < S_kv; j++) {
                    float sum = 0.0f;
                    for (int d = 0; d < D; d++) {
                        sum += Q_h[i * D + d] * K_h[j * D + d];
                    }
                    attn_h[i * S_kv + j] = sum;
                }
            }
        }
    }

    // Step 3: Apply causal mask if needed
    if (is_causal) {
        fill_causal_mask(causal_mask_buf, S_q, S_kv);
        for (int h = 0; h < BH; h++) {
            float* attn_h = qk_buffer + h * S_q * S_kv;
            add_mask_inplace(attn_h, causal_mask_buf, (int64_t)S_q * S_kv);
        }
    }

    // Step 4: Softmax along last dim for each head
    for (int h = 0; h < BH; h++) {
        float* attn_h = qk_buffer + h * S_q * S_kv;
        // softmax each row independently — treat as [S_q, S_kv] → softmax dim=-1
        if (!softmax(attn_h, attn_h, S_q, S_kv)) {
            // Fallback: manual softmax
            for (int i = 0; i < S_q; i++) {
                float* row = attn_h + i * S_kv;
                float max_val = -INFINITY;
                for (int j = 0; j < S_kv; j++) max_val = std::max(max_val, row[j]);
                float sum = 0.0f;
                for (int j = 0; j < S_kv; j++) {
                    row[j] = std::exp(row[j] - max_val);
                    sum += row[j];
                }
                for (int j = 0; j < S_kv; j++) row[j] /= sum;
            }
        }
    }

    // Step 5: output = attn @ V  (for each head)
    for (int h = 0; h < BH; h++) {
        const float* attn_h = qk_buffer + h * S_q * S_kv;
        const float* V_h = V + h * S_kv * D;
        float* out_h = output + h * S_q * D;

        // gemm: out_h[S_q, D] = attn_h[S_q, S_kv] @ V_h[S_kv, D]
        if (!gemm(attn_h, V_h, out_h, S_q, D, S_kv)) {
            // Fallback: manual matmul
            for (int i = 0; i < S_q; i++) {
                for (int d = 0; d < D; d++) {
                    float sum = 0.0f;
                    for (int j = 0; j < S_kv; j++) {
                        sum += attn_h[i * S_kv + j] * V_h[j * D + d];
                    }
                    out_h[i * D + d] = sum;
                }
            }
        }
    }

    delete[] Q_scaled;
    delete[] K_scaled;
}

} // namespace cpu
} // namespace anima
