// Head and tail ops for DiT — C++ CPU implementations
// Replaces PyTorch x_embedder, t_embedder, final_layer, unpatchify
// All computations in fp32 for numerical parity with PyTorch.

#pragma once

#include <cmath>
#include <cstring>
#include <cstdint>
#include <algorithm>

namespace head_tail {

// ── Helpers ──

// BF16 unpack: uint16_t → float32 (exact: just shift)
static inline float bf16_to_f32(uint16_t bf16) {
    uint32_t bits = ((uint32_t)bf16) << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

// FP32 → FP16 (for final output conversion)
static inline uint16_t f32_to_f16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000;
    uint32_t exp  = (bits >> 23) & 0xFF;
    uint32_t mant = bits & 0x7FFFFF;

    if (exp == 0) return (uint16_t)sign;           // zero / subnormal
    if (exp >= 0x8F) {                              // overflow / Inf / NaN
        if (exp == 0xFF && mant) return (uint16_t)(sign | 0x7E00 | (mant >> 13)); // NaN
        return (uint16_t)(sign | 0x7C00);            // Inf
    }
    if (exp <= 0x70) return (uint16_t)sign;         // underflow

    uint32_t f16_exp = exp - 0x70;                  // fp32 bias 127 → fp16 bias 15
    uint32_t f16_mant = (mant + 0x1000) >> 13;       // 23-bit → 10-bit with rounding
    return (uint16_t)(sign | (f16_exp << 10) | f16_mant);
}

// Simple CPU GEMM: C[M,N] = A[M,K] @ B[N,K]^T  (all fp32)
// B is stored as BF16 uint16_t[].  alpha is scaling factor.
static inline void cpu_gemm_bf16(int M, int N, int K,
                                  const float* A, const uint16_t* B_bf16,
                                  float* C, float alpha = 1.0f) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * bf16_to_f32(B_bf16[n * K + k]);
            }
            C[m * N + n] = sum * alpha;
        }
    }
}

// Simple CPU GEMM: C[M,N] = A[M,K] @ B[K,N]  (B is col-major equiv, all fp32)
// Used for cases where B is stored in [K,N] layout
static inline void cpu_gemm_fp32(int M, int N, int K,
                                  const float* A, const float* B,
                                  float* C, float alpha = 1.0f) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum * alpha;
        }
    }
}

// SiLU activation (in-place on fp32)
static inline void silu_inplace(float* x, int N) {
    for (int i = 0; i < N; i++) {
        x[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

// RMS Norm (simple, no affine): y = x * rsqrt(mean(x^2) + eps)
// x: [rows, D], D elements per row, in-place, fp32
static inline void rms_norm_inplace(float* x, int rows, int D, float eps = 1e-6f) {
    for (int r = 0; r < rows; r++) {
        float* row = x + r * D;
        float sq = 0.0f;
        for (int i = 0; i < D; i++) sq += row[i] * row[i];
        float rms = 1.0f / sqrtf(sq / (float)D + eps);
        for (int i = 0; i < D; i++) row[i] *= rms;
    }
}

// LayerNorm (no affine): y = (x - mean) / sqrt(var + eps), fp32
static inline void layernorm(const float* x, float* out, int rows, int D, float eps = 1e-6f) {
    for (int r = 0; r < rows; r++) {
        const float* row = x + r * D;
        float sum = 0.0f;
        for (int i = 0; i < D; i++) sum += row[i];
        float mean = sum / (float)D;
        float sq = 0.0f;
        for (int i = 0; i < D; i++) { float d = row[i] - mean; sq += d * d; }
        float inv_std = 1.0f / sqrtf(sq / (float)D + eps);
        float* o = out + r * D;
        for (int i = 0; i < D; i++) o[i] = (row[i] - mean) * inv_std;
    }
}

// ═══════════════════════════════════════════════════════════════
// HEAD OPS
// ═══════════════════════════════════════════════════════════════

// x_embedder: PatchEmbed
// Input:  x [B, C+1, T, H, W] fp16 format (uint16_t[])
//          where C=in_channels (16), +1 for padding mask
//          spatial_patch=2, temporal_patch=1
// Output: x_emb [MS, out_channels] where MS=B*T*H_patches*W_patches, out_channels=2048
//          stored as fp32
//
// Step 1: rearrange [B, C+1, T, H, W] → [B, T, H/2, W/2, (C+1)*4]
// Step 2: Linear: [MS, 68] @ W[2048, 68]^T → [MS, 2048]
//
// W_x_proj: BF16 weight [2048, 68] stored as uint16_t[]
static bool x_embed(const uint16_t* x_fp16,       // [B, C+1, T, H_pix, W_pix]
                     const uint16_t* w_proj_bf16,  // [out_C, dim] = [2048, 68]
                     float* x_emb_out_fp32,         // [B*T*Hp*Wp, out_C] fp32
                     int B, int C_in, int T, int H_pix, int W_pix,
                     int patch_spatial, int patch_temporal, int out_C) {
    int C_pad = C_in + 1;  // extra channel for padding mask
    int H_patches = H_pix / patch_spatial;
    int W_patches = W_pix / patch_spatial;
    int in_dim = C_pad * patch_spatial * patch_spatial * patch_temporal; // 17*2*2*1 = 68
    int patch_elems = patch_temporal * patch_spatial * patch_spatial; // 1*2*2=4

    int MS = B * T * H_patches * W_patches;  // 2*1*16*16 = 512

    // Allocate temp buffer for rearranged input [MS, in_dim] fp32
    float* x_rearr = new (std::nothrow) float[MS * in_dim];
    if (!x_rearr) return false;

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int hp = 0; hp < H_patches; hp++) {
                for (int wp = 0; wp < W_patches; wp++) {
                    int out_row = ((b * T + t) * H_patches + hp) * W_patches + wp;
                    float* dst = x_rearr + out_row * in_dim;
                    int didx = 0;
                    // Real image channels (C_in = 16): read from input buffer
                    for (int c = 0; c < C_in; c++) {
                        for (int pt = 0; pt < patch_temporal; pt++) {
                            for (int ph = 0; ph < patch_spatial; ph++) {
                                for (int pw = 0; pw < patch_spatial; pw++) {
                                    // Input layout: [B, C_in, T, H_pix, W_pix]
                                    int in_idx = (((b * C_in + c) * T + (t * patch_temporal + pt)) * H_pix
                                                  + (hp * patch_spatial + ph)) * W_pix
                                                  + (wp * patch_spatial + pw);
                                    uint16_t h = x_fp16[in_idx];
                                    // fp16 → fp32
                                    uint32_t sign = (h >> 15) & 1;
                                    uint32_t exp  = (h >> 10) & 0x1F;
                                    uint32_t mant = h & 0x3FF;
                                    float val;
                                    if (exp == 0) val = 0.0f;
                                    else if (exp == 31) val = (mant == 0) ? INFINITY : NAN;
                                    else {
                                        uint32_t f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
                                        memcpy(&val, &f32, sizeof(val));
                                    }
                                    dst[didx++] = val;
                                }
                            }
                        }
                    }
                    // Padding mask channel: all zeros
                    for (int i = 0; i < patch_elems; i++) dst[didx++] = 0.0f;
                }
            }
        }
    }

    // Linear: [MS, in_dim] @ W[out_C, in_dim]^T → [MS, out_C]
    cpu_gemm_bf16(MS, out_C, in_dim, x_rearr, w_proj_bf16, x_emb_out_fp32);
    delete[] x_rearr;
    return true;
}

// t_embedder: Timesteps + TimestepEmbedding
// Input: sigma (single float, one timestep for all batch items)
// Output: t_emb [M, D] where M=batch (2 for CFG), D=2048
//         adaln_lora [M, 3*D] = [M, 6144]
// Weights: w_t1 [2048, 2048] BF16, w_t2 [6144, 2048] BF16
//          (t_embedding_norm.weight [2048] BF16 handled separately)
static bool t_embed(const float* sigmas, int M,      // M sigma values (all same for CFG)
                     const uint16_t* w_t1_bf16,      // [2048, 2048]
                     const uint16_t* w_t2_bf16,      // [6144, 2048]
                     float* t_emb_out,                // [M, 2048] fp32
                     float* adaln_lora_out) {         // [M, 6144] fp32
    const int D = 2048;
    const int half = D / 2;  // 1024

    // Step 1: Sinusoidal embedding (Timesteps)
    // emb = cos(sin) embeddings: [M, 2048]
    float* emb = new (std::nothrow) float[M * D];
    if (!emb) return false;

    for (int m = 0; m < M; m++) {
        float sigma = sigmas[m];
        for (int i = 0; i < half; i++) {
            float exponent = -logf(10000.0f) * (float)i / (float)(half);
            float freq = expf(exponent);
            float val = sigma * freq;
            // IMPORTANT: PyTorch Timesteps uses torch.cat([sin_emb, cos_emb], dim=-1)
            // so sin goes to [0:half], cos goes to [half:D]
            emb[m * D + i] = sinf(val);
            emb[m * D + half + i] = cosf(val);
        }
    }

    // Step 2: SiLU → Linear1(2048→2048) → SiLU → Linear2(2048→6144)
    // Linear1: [M, D] @ W1[D, D]^T → [M, D]
    float* h1 = new (std::nothrow) float[M * D];
    if (!h1) { delete[] emb; return false; }
    cpu_gemm_bf16(M, D, D, emb, w_t1_bf16, h1);
    silu_inplace(h1, M * D);

    // Linear2: [M, D] @ W2[3*D, D]^T → [M, 3*D]
    cpu_gemm_bf16(M, 3 * D, D, h1, w_t2_bf16, adaln_lora_out);

    // t_emb is the raw embedding (before SiLU+Linear)
    memcpy(t_emb_out, emb, M * D * sizeof(float));

    delete[] h1;
    delete[] emb;
    return true;
}

// t_embedding_norm: RMSNorm on t_emb [M, D]
// Weight: w_norm [D] BF16
static void t_embedding_norm(float* t_emb,        // [M, D] fp32 in/out
                              const uint16_t* w,  // [D] BF16
                              int M, int D, float eps = 1e-6f) {
    rms_norm_inplace(t_emb, M, D, eps);
    // Apply weight
    for (int r = 0; r < M; r++) {
        for (int d = 0; d < D; d++) {
            t_emb[r * D + d] *= bf16_to_f32(w[d]);
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// TAIL OPS
// ═══════════════════════════════════════════════════════════════

// final_layer: LN → AdaLN → Linear → patches
// Input:  x [MS, D] fp32  (block output)
//         t_emb [M, D] fp32
//         adaln_lora [M, 3*D] fp32
// Output: patches [MS, out_dim] fp32 where out_dim = patch^2 * out_channels = 4*16 = 64
// Weights: w_fa1 [256, D] BF16, w_fa2 [2*D, 256] BF16, w_fl [out_dim, D] BF16
static bool final_layer(const float* x_fp32,          // [MS, D]
                         const float* t_emb_fp32,      // [M, D]
                         const float* adaln_lora_fp32, // [M, 3*D]
                         const uint16_t* w_fa1_bf16,   // [256, D]
                         const uint16_t* w_fa2_bf16,   // [2*D, 256]
                         const uint16_t* w_fl_bf16,    // [out_dim, D]
                         float* out_fp32,              // [MS, out_dim]
                         int MS, int M, int D, int out_dim,
                         float eps = 1e-6f) {

    int S = MS / M;  // tokens per batch item
    int n_adaln_chunks = 2;  // final_layer uses only first 2 chunks of adaln_lora
    int adaln_dim = n_adaln_chunks * D;  // 4096

    // 1. LayerNorm(x) [MS, D]
    float* x_norm = new (std::nothrow) float[MS * D];
    if (!x_norm) return false;
    layernorm(x_fp32, x_norm, MS, D, eps);

    // 2. AdaLN modulation from t_emb:
    //    h = silu(t_emb @ W_fa1^T) @ W_fa2^T  → [M, 4096]
    float* h1 = new (std::nothrow) float[M * 256];  // [M, 256]
    float* h2 = new (std::nothrow) float[M * adaln_dim];  // [M, 4096]
    if (!h1 || !h2) { delete[] x_norm; delete[] h1; delete[] h2; return false; }

    cpu_gemm_bf16(M, 256, D, t_emb_fp32, w_fa1_bf16, h1);  // SiLU input
    silu_inplace(h1, M * 256);
    cpu_gemm_bf16(M, adaln_dim, 256, h1, w_fa2_bf16, h2);  // [M, 4096]

    // 3. Combine with adaln_lora: h2 += adaln_lora[:, :4096]
    for (int i = 0; i < M * adaln_dim; i++) {
        h2[i] += adaln_lora_fp32[i];
    }

    // 4. Apply scale/shift per token
    //    shift = h2[:, :D], scale = h2[:, D:] + 1.0
    //    broadcast [M, D] → [MS, D]
    float* scale = new (std::nothrow) float[MS * D];
    float* shift_buf = new (std::nothrow) float[MS * D];
    if (!scale || !shift_buf) { delete[] x_norm; delete[] h1; delete[] h2; delete[] scale; delete[] shift_buf; return false; }

    for (int m = 0; m < M; m++) {
        for (int s = 0; s < S; s++) {
            for (int d = 0; d < D; d++) {
                scale[(m * S + s) * D + d] = h2[m * adaln_dim + D + d] + 1.0f;
                shift_buf[(m * S + s) * D + d] = h2[m * adaln_dim + d];
            }
        }
    }

    // x_mod = x_norm * scale + shift
    float* x_mod = new (std::nothrow) float[MS * D];
    if (!x_mod) { delete[] x_norm; delete[] h1; delete[] h2; delete[] scale; delete[] shift_buf; return false; }
    for (int i = 0; i < MS * D; i++) {
        x_mod[i] = fmaf(x_norm[i], scale[i], shift_buf[i]);
    }

    // 5. Final Linear: [MS, D] @ W_f_linear[out_dim, D]^T → [MS, out_dim]
    cpu_gemm_bf16(MS, out_dim, D, x_mod, w_fl_bf16, out_fp32);

    delete[] x_norm;
    delete[] h1;
    delete[] h2;
    delete[] scale;
    delete[] shift_buf;
    delete[] x_mod;
    return true;
}

// unpatchify: rearrange [B, T, H_patches, W_patches, out_dim] → [B, C_out, T, H_pix, W_pix]
// out_dim = patch^2 * C_out  (e.g., 4 * 16 = 64)
static void unpatchify(const float* in_fp32,        // [B, T, Hp, Wp, out_dim]
                        float* out_fp32,              // [B, C_out, T, H_pix, W_pix]
                        int B, int T, int Hp, int Wp,
                        int patch_spatial, int patch_temporal, int C_out) {
    int out_dim = patch_spatial * patch_spatial * patch_temporal * C_out;  // 2*2*1*16=64
    int H_pix = Hp * patch_spatial;
    int W_pix = Wp * patch_spatial;

    // Zero output
    int out_total = B * C_out * T * patch_temporal * H_pix * W_pix;
    memset(out_fp32, 0, out_total * sizeof(float));

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int hp = 0; hp < Hp; hp++) {
                for (int wp = 0; wp < Wp; wp++) {
                    for (int pt = 0; pt < patch_temporal; pt++) {
                        for (int ph = 0; ph < patch_spatial; ph++) {
                            for (int pw = 0; pw < patch_spatial; pw++) {
                                for (int c = 0; c < C_out; c++) {
                                    // in: [b, t, hp, wp, (c * pt_max * ph_max * pw_max) + pt*ph*pw + ph*pw + pw]
                                    int in_idx = ((((b * T + t) * Hp + hp) * Wp + wp) * out_dim
                                                   + ((c * patch_temporal + pt) * patch_spatial + ph) * patch_spatial + pw);
                                    int out_idx = ((((b * C_out + c) * (T * patch_temporal) + (t * patch_temporal + pt))
                                                     * H_pix + (hp * patch_spatial + ph)) * W_pix
                                                     + (wp * patch_spatial + pw));
                                    out_fp32[out_idx] = in_fp32[in_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

} // namespace head_tail
