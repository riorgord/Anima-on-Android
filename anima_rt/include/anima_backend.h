// Pluggable backend interface — function-pointer tables.
// CPU is the default; Vulkan/QNN swap in by replacing function pointers.
#pragma once
#include <cstdint>

namespace anima {

// Opaque backend context (CPU = nothing, Vulkan = VkDevice+shaders, QNN = QNN backend)
struct BackendContext { virtual ~BackendContext() = default; };

// ── Per-op backend tables ──────────────────────────────────────────

struct GemmBackend {
    BackendContext* ctx = nullptr;
    bool (*init)(BackendContext* ctx) = nullptr;
    // C[M,N] += alpha * A[M,K] @ weight[N,K]^T
    // weight is BF16 (uint16_t), A/C are FP32 (float)
    bool (*compute_bf16)(BackendContext* ctx,
                         const uint16_t* weight, const float* A,
                         float* C, int64_t M, int64_t N, int64_t K,
                         float alpha) = nullptr;
    void (*destroy)(BackendContext* ctx) = nullptr;
};

struct LayerNormBackend {
    BackendContext* ctx = nullptr;
    bool (*init)(BackendContext* ctx) = nullptr;
    // Y[M,D] = LayerNorm(X[M,D]; gamma[D]|null, beta[D]|null, eps)
    bool (*compute)(BackendContext* ctx,
                    const float* X, const float* gamma, const float* beta,
                    float* Y, int64_t M, int64_t D, float eps) = nullptr;
    void (*destroy)(BackendContext* ctx) = nullptr;
};

struct RMSNormBackend {
    BackendContext* ctx = nullptr;
    bool (*init)(BackendContext* ctx) = nullptr;
    // Y[M,D] = X[M,D] * rsqrt(mean(X^2, dim=-1) + eps) * weight[D]
    bool (*compute)(BackendContext* ctx,
                    const float* X, const float* weight,
                    float* Y, int64_t M, int64_t D, float eps) = nullptr;
    void (*destroy)(BackendContext* ctx) = nullptr;
};

struct ActivationBackend {
    BackendContext* ctx = nullptr;
    bool (*init)(BackendContext* ctx) = nullptr;
    bool (*gelu)(BackendContext* ctx, const float* X, float* Y, int64_t N) = nullptr;
    bool (*silu)(BackendContext* ctx, const float* X, float* Y, int64_t N) = nullptr;
    void (*destroy)(BackendContext* ctx) = nullptr;
};

struct SoftmaxBackend {
    BackendContext* ctx = nullptr;
    bool (*init)(BackendContext* ctx) = nullptr;
    // Softmax along last dimension
    bool (*compute)(BackendContext* ctx,
                    const float* X, float* Y,
                    int64_t outer_size, int64_t dim_size) = nullptr;
    void (*destroy)(BackendContext* ctx) = nullptr;
};

// ── Top-level backend ──────────────────────────────────────────────

struct InferenceBackend {
    GemmBackend       gemm;
    LayerNormBackend  layernorm;
    RMSNormBackend    rmsnorm;
    ActivationBackend activation;
    SoftmaxBackend    softmax;

    static InferenceBackend* create_cpu();
    // Future: static InferenceBackend* create_vulkan();
    // Future: static InferenceBackend* create_qnn();
    static void destroy(InferenceBackend* b);
};

} // namespace anima
