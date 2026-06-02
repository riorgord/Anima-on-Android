/* RoPE (Rotary Position Embedding) kernel
 *
 * Extracted from predict2.py's apply_rotary_pos_emb:
 *
 *   t_ = t.reshape(*t.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2).float()
 *   t_out = freqs[..., 0] * t_[..., 0] + freqs[..., 1] * t_[..., 1]
 *   t_out = t_out.movedim(-1, -2).reshape(*t.shape).type_as(t)
 *
 * This is the "rotate_half" RoPE variant (common in HuggingFace models):
 *   For each pair (x_r, x_c) at the input:
 *     out_r = cos * x_r + sin * x_c
 *   Then interleave back to original shape.
 *
 * t:    shape [B, S, H, D] or [M, D] where D is divisible by 2
 * freqs: cos/sin values, shape compatible with t's broadcast
 *        For each position, freqs has 2 values: (cos, sin)
 */
#pragma once
#include <cmath>
#include <cstring>

namespace anima {
namespace cpu {

/* ── FP32 RoPE, element-wise formula ─────────────────────────────────
 *
 * Loop over all elements, pair-wise.
 * For each pair (x_r, x_c) at position (b, s, h, 2*p):
 *   out_r = freqs_cos * x_r + freqs_sin * x_c
 *
 * The output is interleaved back to original dim order.
 */
inline void rope_kernel(
    const float* t,          // [B, S, H, D]
    const float* freqs,      // cos/sin values, shape compatible
    float* out,              // [B, S, H, D]
    int B, int S, int H, int D,
    int freq_stride          // stride in freqs for each (cos,sin) pair
) {
    int half_D = D / 2;
    int64_t total_pairs = (int64_t)B * S * H * half_D;

    for (int64_t idx = 0; idx < total_pairs; idx++) {
        // Decompose flat index
        int64_t tmp = idx;
        int pair_idx = tmp % half_D;   tmp /= half_D;
        int h_idx    = tmp % H;        tmp /= H;
        int s_idx    = tmp % S;        tmp /= S;
        int b_idx    = tmp;

        // Source indices in t: x_r and x_c are adjacent in the reshaped layout
        // After reshape(*.shape[:-1], 2, -1).movedim(-2, -1):
        // The pair (x_r, x_c) at position p maps to the original D elements.
        // x_r = element at flat index * 2 within the pair
        // x_c = element at flat index * 2 + 1 within the pair
        int64_t base_t = ((int64_t)b_idx * S * H + (int64_t)s_idx * H + h_idx) * D;
        float x_r = t[base_t + pair_idx * 2];
        float x_c = t[base_t + pair_idx * 2 + 1];

        // freqs: for this position, find cos and sin
        // freqs has shape [L, D] or compatible — the indexing depends on position
        int freq_idx = s_idx * freq_stride + pair_idx * 2;
        float cos_val = freqs[freq_idx];
        float sin_val = freqs[freq_idx + 1];

        // Formula: out_r = cos * x_r + sin * x_c
        float out_val = cos_val * x_r + sin_val * x_c;

        // Place in output: same position as x_r
        out[base_t + pair_idx * 2] = out_val;
        // The x_c position in output — need to compute separately
        // Actually the output shape matches input, so we need both values.
        // But the formula only computes ONE value per pair...
        // Let me re-examine the Python reshape.
    }
}

/* For now, we keep RoPE as a simple element-wise operation that mirrors
 * the exact Python formula. Since the formula involves reshape/movedim,
 * we implement it directly in Python and call this as a helper.
 *
 * The pure-C version above is INCOMPLETE — it needs the correct interleaving
 * logic that matches predict2.py:31-38. We'll finalize it when testing on phone.
 */

} // namespace cpu
} // namespace anima
