// Public C API — extern "C" functions callable via Python ctypes.
// Mirrors the proven pattern from hybridops/vulkan/hybrid_engine.cpp.
#include "anima_backend.h"
#include "cpu/flash_attention.h"
#include <cstdint>
#include <cmath>

static anima::InferenceBackend* g_backend = nullptr;

extern "C" {

bool anima_rt_init(void) {
    if (g_backend) return true;  // already initialised
    g_backend = anima::InferenceBackend::create_cpu();
    if (!g_backend) return false;
    // Init GEMM backend (loads OpenBLAS if available)
    if (g_backend->gemm.init) g_backend->gemm.init(g_backend->gemm.ctx);
    return true;
}

// ── Activation ────────────────────────────────────────────────────

bool anima_rt_run_gelu(const float* x, float* out, int n) {
    if (!g_backend || !g_backend->activation.gelu) return false;
    return g_backend->activation.gelu(g_backend->activation.ctx, x, out, n);
}

bool anima_rt_run_silu(const float* x, float* out, int n) {
    if (!g_backend || !g_backend->activation.silu) return false;
    return g_backend->activation.silu(g_backend->activation.ctx, x, out, n);
}

// ── Normalization ─────────────────────────────────────────────────

bool anima_rt_run_layernorm(const float* x, float* out,
                            int m, int d, float eps) {
    if (!g_backend || !g_backend->layernorm.compute) return false;
    // elementwise_affine=False → gamma=null, beta=null
    return g_backend->layernorm.compute(g_backend->layernorm.ctx,
                                        x, nullptr, nullptr,
                                        out, m, d, eps);
}

bool anima_rt_run_rmsnorm(const float* x, const float* w, float* out,
                          int m, int d, float eps) {
    if (!g_backend || !g_backend->rmsnorm.compute) return false;
    return g_backend->rmsnorm.compute(g_backend->rmsnorm.ctx,
                                      x, w, out, m, d, eps);
}

bool anima_rt_run_softmax(const float* x, float* out,
                          int outer, int dim) {
    if (!g_backend || !g_backend->softmax.compute) return false;
    return g_backend->softmax.compute(g_backend->softmax.ctx,
                                      x, out, outer, dim);
}

// ── GEMM ──────────────────────────────────────────────────────────

// FP32 GEMM: C[M,N] = A[M,K] @ B_weight^T[N,K]
// A: FP32, B_weight: FP32, C: FP32
// Declared in gemm_backend.cpp
extern bool anima_rt_run_gemm_fp32(const float* A, const float* B,
                                   float* C, int M, int N, int K);

// BF16 GEMM: C[M,N] = A[M,K] @ W_bf16^T[N,K]
// A: FP32, W_bf16: BF16 as uint16_t (packed), C: FP32
// Declared in gemm_backend.cpp
extern bool anima_rt_run_gemm_bf16(const float* A, const uint16_t* W,
                                   float* C, int M, int N, int K);

// ── SDPA (Scaled Dot-Product Attention) math backend ──────────────
// Composed from our GEMM + Softmax kernels.
// Q, K, V: [B*H, S, D] — batch*heads flattened
// S_q = query seq len, S_kv = key/value seq len, D = head_dim

extern bool anima_rt_run_sdpa(
    const float* Q, const float* K, const float* V,
    float* output,
    int BH, int S_q, int S_kv, int D,
    float scale, bool is_causal)
{
    if (!g_backend) return false;

    // Allocate temporary buffers
    int64_t qk_size = (int64_t)BH * S_q * S_kv;
    int64_t mask_size = is_causal ? (int64_t)S_q * S_kv : 0;
    float* qk_buf = new float[qk_size];
    float* mask_buf = mask_size > 0 ? new float[mask_size] : nullptr;

    // Scale Q and K in temporary copies
    int64_t Q_size = (int64_t)BH * S_q * D;
    int64_t K_size = (int64_t)BH * S_kv * D;
    float* Q_s = new float[Q_size];
    float* K_s = new float[K_size];
    for (int64_t i = 0; i < Q_size; i++) Q_s[i] = Q[i] * scale;
    for (int64_t i = 0; i < K_size; i++) K_s[i] = K[i] * scale;

    // Per-head loop
    for (int h = 0; h < BH; h++) {
        const float* Q_h = Q_s + h * S_q * D;
        const float* K_h = K_s + h * S_kv * D;
        const float* V_h = V   + h * S_kv * D;
        float* attn_h = qk_buf + h * S_q * S_kv;
        float* out_h  = output + h * S_q * D;

        // Step 1: attn = Q_h @ K_h^T  [S_q, S_kv]
        anima_rt_run_gemm_fp32(Q_h, K_h, attn_h, S_q, S_kv, D);

        // Step 2: causal mask
        if (is_causal) {
            for (int i = 0; i < S_q; i++)
                for (int j = 0; j < S_kv; j++)
                    mask_buf[i * S_kv + j] = (j > i) ? -INFINITY : 0.0f;
            for (int64_t i = 0; i < (int64_t)S_q * S_kv; i++)
                attn_h[i] += mask_buf[i];
        }

        // Step 3: softmax along last dim
        g_backend->softmax.compute(g_backend->softmax.ctx,
                                   attn_h, attn_h, S_q, S_kv);

        // Step 4: output = attn @ V  [S_q, D] = attn[S_q,S_kv] @ V[S_kv,D]
        // gemm_fp32 computes A @ B^T, but we need A @ B. Manual matmul.
        for (int i = 0; i < S_q; i++) {
            for (int j = 0; j < D; j++) {
                float sum = 0.0f;
                for (int k = 0; k < S_kv; k++)
                    sum += attn_h[i * S_kv + k] * V_h[k * D + j];
                out_h[i * D + j] = sum;
            }
        }
    }

    delete[] Q_s;
    delete[] K_s;
    delete[] qk_buf;
    delete[] mask_buf;
    return true;
}

// ── SDPA Flash Attention (tiled online softmax) ────────────────────
// Matches PT 2.11's default F.scaled_dot_product_attention behavior.
// Same interface as math backend, different algorithm.

extern bool anima_rt_run_sdpa_flash(
    const float* Q, const float* K, const float* V,
    float* output,
    int BH, int S_q, int S_kv, int D,
    float scale, bool is_causal)
{
    anima::cpu::flash_attention(Q, K, V, output,
                                BH, S_q, S_kv, D, scale, is_causal);
    return true;
}

void anima_rt_destroy(void) {
    if (g_backend) {
        anima::InferenceBackend::destroy(g_backend);
        g_backend = nullptr;
    }
}

} // extern "C"
