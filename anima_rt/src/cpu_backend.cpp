// CPU backend — fills the InferenceBackend function pointers
// with the extracted kernel implementations.
#include "anima_backend.h"
#include "cpu/gelu_kernel.h"
#include "cpu/silu_kernel.h"
#include "cpu/softmax_kernel.h"
#include "cpu/layernorm_kernel.h"
#include "cpu/rmsnorm_kernel.h"

namespace anima {

// ── CPU backend context (stateless) ────────────────────────────────

struct CPUBackendContext : BackendContext {};

// ── Activation ─────────────────────────────────────────────────────

static bool cpu_gelu(BackendContext*, const float* X, float* Y, int64_t N) {
    cpu::gelu_kernel(X, Y, N);
    return true;
}

static bool cpu_silu(BackendContext*, const float* X, float* Y, int64_t N) {
    cpu::silu_kernel(X, Y, N);
    return true;
}

// ── LayerNorm ──────────────────────────────────────────────────────

static bool cpu_layernorm(BackendContext*,
                          const float* X, const float* gamma, const float* beta,
                          float* Y, int64_t M, int64_t D, float eps) {
    cpu::layernorm_kernel(X, gamma, beta, Y, M, D, eps);
    return true;
}

// ── RMSNorm ────────────────────────────────────────────────────────

static bool cpu_rmsnorm(BackendContext*,
                        const float* X, const float* weight,
                        float* Y, int64_t M, int64_t D, float eps) {
    cpu::rmsnorm_kernel(X, weight, Y, M, D, eps);
    return true;
}

// ── Softmax ────────────────────────────────────────────────────────

static bool cpu_softmax(BackendContext*,
                        const float* X, float* Y,
                        int64_t outer, int64_t dim) {
    cpu::softmax_kernel(X, Y, outer, dim);
    return true;
}

// ── Factory ────────────────────────────────────────────────────────

InferenceBackend* InferenceBackend::create_cpu() {
    auto* ctx = new CPUBackendContext();
    auto* b   = new InferenceBackend();

    b->activation.ctx     = ctx;
    b->activation.gelu    = cpu_gelu;
    b->activation.silu    = cpu_silu;
    b->activation.destroy = [](BackendContext* c) { delete (CPUBackendContext*)c; };

    b->layernorm.ctx     = nullptr;  // stateless, reuse ctx would need sharing
    b->layernorm.compute = cpu_layernorm;

    b->rmsnorm.ctx     = nullptr;
    b->rmsnorm.compute = cpu_rmsnorm;

    b->softmax.ctx     = nullptr;
    b->softmax.compute = cpu_softmax;

    b->gemm.ctx     = nullptr;
    b->gemm.compute_bf16 = nullptr;  // GEMM stays in PyTorch for now

    return b;
}

void InferenceBackend::destroy(InferenceBackend* b) {
    if (!b) return;
    if (b->activation.destroy && b->activation.ctx)
        b->activation.destroy(b->activation.ctx);
    delete b;
}

} // namespace anima
