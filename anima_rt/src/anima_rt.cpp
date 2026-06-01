// Public C API — extern "C" functions callable via Python ctypes.
// Mirrors the proven pattern from hybridops/vulkan/hybrid_engine.cpp.
#include "anima_backend.h"

static anima::InferenceBackend* g_backend = nullptr;

extern "C" {

bool anima_rt_init(void) {
    if (g_backend) return true;  // already initialised
    g_backend = anima::InferenceBackend::create_cpu();
    return g_backend != nullptr;
}

bool anima_rt_run_gelu(const float* x, float* out, int n) {
    if (!g_backend || !g_backend->activation.gelu) return false;
    return g_backend->activation.gelu(g_backend->activation.ctx, x, out, n);
}

bool anima_rt_run_silu(const float* x, float* out, int n) {
    if (!g_backend || !g_backend->activation.silu) return false;
    return g_backend->activation.silu(g_backend->activation.ctx, x, out, n);
}

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

void anima_rt_destroy(void) {
    if (g_backend) {
        anima::InferenceBackend::destroy(g_backend);
        g_backend = nullptr;
    }
}

} // extern "C"
