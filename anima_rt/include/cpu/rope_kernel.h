/* RoPE (Rotary Position Embedding) apply kernel
 *
 * Direct C++ translation of predict2.py:31-38:
 *   t_ = t.reshape(*t.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2).float()
 *   t_out = freqs[..., 0] * t_[..., 0] + freqs[..., 1] * t_[..., 1]
 *   t_out = t_out.movedim(-1, -2).reshape(*t.shape).type_as(t)
 *
 * CRITICAL: PT's .reshape(*, 2, D/2) groups D elements into TWO HALVES:
 *   even half = [0..D/2-1], odd half = [D/2..D-1]
 *   Pairs are (p, D/2+p) — NOT adjacent (2p, 2p+1)!
 *
 * Formula for each (head, position, pair p in 0..D/2-1):
 *   out[p]        = cos * x[p] + (-sin) * x[D/2+p]
 *   out[D/2+p]    = sin * x[p] + cos * x[D/2+p]
 *
 * Freqs layout [S, D/2, 2, 2] row-major flat:
 *   freqs[s*D*2 + p*4 + 0] = cos
 *   freqs[s*D*2 + p*4 + 1] = -sin
 *   freqs[s*D*2 + p*4 + 2] = sin
 *   freqs[s*D*2 + p*4 + 3] = cos
 *
 * All computation in FP32.
 */
#pragma once
#include <cstdint>

namespace anima {
namespace cpu {

inline void rope_kernel(
    const float* t,       // [N, S, D]  N=B*H, S=seq_len, D=head_dim (even)
    const float* freqs,   // [S, D/2, 2, 2] row-major flat (extra singleton dims OK)
    float* out,           // [N, S, D]
    int N, int S, int D)
{
    const int half_D = D / 2;

    for (int n = 0; n < N; n++) {
        const float* t_n   = t + n * S * D;
        float*       out_n = out + n * S * D;

        for (int s = 0; s < S; s++) {
            const float* t_ns     = t_n + s * D;
            float*       out_ns   = out_n + s * D;
            const float* freqs_s  = freqs + s * D * 2;  // stride = D/2*2*2 = D*2

            for (int p = 0; p < half_D; p++) {
                // PT pairing: element p (even) with element half_D+p (odd)
                float x_even = t_ns[p];
                float x_odd  = t_ns[half_D + p];

                float cos_val  = freqs_s[p * 4 + 0];  // cos
                float nsin_val = freqs_s[p * 4 + 1];  // -sin
                float sin_val  = freqs_s[p * 4 + 2];  // sin
                float cos2_val = freqs_s[p * 4 + 3];  // cos

                out_ns[p]          = cos_val * x_even + nsin_val * x_odd;
                out_ns[half_D + p] = sin_val * x_even + cos2_val * x_odd;
            }
        }
    }
}

} // namespace cpu
} // namespace anima
