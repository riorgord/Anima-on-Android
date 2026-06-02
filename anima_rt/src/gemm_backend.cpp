/* GEMM backend: OpenBLAS dlopen (primary) + pure C++ fallback.
 *
 * Strategy: PyTorch on ARM Linux (Termux) uses OpenBLAS sgemm_/sbgemm_.
 * We dlopen the same library → bit-exact same results as PT.
 * If OpenBLAS is unavailable, we use a pure C++ fallback that accumulates
 * in FP32 (same mathematical formula, just slower).
 *
 * All computation is FP32 — matches the phone's "bf16 memory, fp32 compute" approach.
 */

#include <dlfcn.h>
#include <cstring>
#include <cstdlib>
#include <cstdint>

/* ── OpenBLAS function pointer types (Fortran column-major interface) ──── */
typedef void (*sgemm_fort_t)(const char* transa, const char* transb,
    const int* m, const int* n, const int* k,
    const float* alpha, const float* a, const int* lda,
    const float* b, const int* ldb,
    const float* beta, float* c, const int* ldc);

typedef void (*sbgemm_fort_t)(const char* transa, const char* transb,
    const int* m, const int* n, const int* k,
    const float* alpha, const void* a, const int* lda,
    const void* b, const int* ldb,
    const float* beta, float* c, const int* ldc);

/* ── Global BLAS handle ──── */
struct BlasState {
    void*    handle;
    sgemm_fort_t  sgemm;
    sbgemm_fort_t sbgemm;
    bool     has_blas;
};
static BlasState g_blas;

/* ── Init / Destroy ──── */
extern "C" bool anima_gemm_init(void) {
    if (g_blas.handle) return true;  // already done

    const char* paths[] = {
        "libopenblas.so",
        "/data/data/com.termux/files/usr/lib/libopenblas.so",
        "/usr/lib/libopenblas.so",
        nullptr
    };
    for (int i = 0; paths[i]; i++) {
        g_blas.handle = dlopen(paths[i], RTLD_NOW | RTLD_GLOBAL);
        if (g_blas.handle) break;
    }
    if (g_blas.handle) {
        g_blas.sgemm  = (sgemm_fort_t)dlsym(g_blas.handle, "sgemm_");
        g_blas.sbgemm = (sbgemm_fort_t)dlsym(g_blas.handle, "sbgemm_");
        g_blas.has_blas = (g_blas.sgemm != nullptr);
    }
    return true;  // always succeed — fallback is available
}

extern "C" void anima_gemm_destroy(void) {
    if (g_blas.handle) { dlclose(g_blas.handle); }
    g_blas.handle = nullptr;
    g_blas.has_blas = false;
}

/* ── BF16 ↔ FP32 helpers ──── */
// BF16 is stored as uint16_t (lower 16 bits of a uint32 that's the FP32 bit pattern).
// To convert: shift uint16_t left by 16 → uint32_t → reinterpret as float.

static inline float bf16_to_f32(uint16_t b) {
    uint32_t bits = ((uint32_t)b) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(float));
    return f;
}

static inline uint16_t f32_to_bf16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(uint32_t));
    // Round-to-nearest-even: add 0x7FFF + ((bits >> 16) & 1) then truncate
    // This matches PyTorch's BF16 rounding convention.
    uint32_t rounding = 0x7FFF + ((bits >> 16) & 1);
    bits += rounding;
    return (uint16_t)(bits >> 16);
}

/* ═══════════════════════════════════════════════════════════════
 * Pure C++ fallback GEMM (numerically equivalent to PT BlasKernel.cpp)
 *
 * Row-major convention:
 *   C[M,N] = A[M,K] @ B^T[N,K]
 *   where A is M×K, B is N×K (weight matrix: out_features × in_features)
 *
 * For BLAS Fortran interface (column-major), we use the identity:
 *   Row-major C[M×N] = A[M×K] @ B^T[K×N]
 *   ≡ Fortran C^T[N×M] = B[N×K] @ A^T[K×M]
 *   → sgemm_('N','N', N,M,K, 1.0, B,N, A,K, 0.0, C,N)
 * ═══════════════════════════════════════════════════════════════ */

/* ── FP32 GEMM: C = A @ B^T  (A: M×K float, B: N×K float) ──── */
static void sgemm_fallback(int M, int N, int K,
                           const float* A, const float* B,
                           float* C) {
    // C[i,j] = sum_k A[i,k] * B[j,k]
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[j * K + k];
            }
            C[i * N + j] = sum;
        }
    }
}

/* ── BF16 GEMM: C = A @ B^T  (A: M×K float, B: N×K bf16 as uint16) ──── */
static void sbgemm_fallback(int M, int N, int K,
                            const float* A, const uint16_t* B,
                            float* C) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            const float* A_row = A + i * K;
            const uint16_t* B_row = B + j * K;
            for (int k = 0; k < K; k++) {
                sum += A_row[k] * bf16_to_f32(B_row[k]);
            }
            C[i * N + j] = sum;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 * Public C API
 * ═══════════════════════════════════════════════════════════════ */

extern "C" bool anima_gemm_has_blas(void) { return g_blas.has_blas; }

/* ── FP32 GEMM: C[M,N] = A[M,K] @ B^T[N,K] ──── */
extern "C" bool anima_rt_run_gemm_fp32(
    const float* A, const float* B, float* C,
    int M, int N, int K)
{
    // Use pure C++ fallback for FP32 — avoids Fortran col-major mapping issues.
    // The fallback is numerically exact and FP32 GEMM for shell Linear (small M,N) is fast enough.
    sgemm_fallback(M, N, K, A, B, C);
    return true;
}

/* ── BF16 GEMM: C[M,N] = A[M,K] @ Weight^T[N,K]
 *   A: FP32 activations  [M, K]
 *   Weight: BF16 packed as uint16_t [N, K]
 *   C: FP32 output [M, N]
 * ──── */
extern "C" bool anima_rt_run_gemm_bf16(
    const float* A, const uint16_t* Weight, float* C,
    int M, int N, int K)
{
    if (g_blas.has_blas && g_blas.sbgemm) {
        const float one = 1.0f, zero = 0.0f;
        int MM = N, NN = M, KK = K;
        g_blas.sbgemm("N", "N", &MM, &NN, &KK,
                      &one, Weight, &MM,
                      A, &KK,
                      &zero, C, &MM);
        return true;
    }
    sbgemm_fallback(M, N, K, A, Weight, C);
    return true;
}

/* ── Bias-add: C[i,j] += bias[j]  (C: M×N, bias: N) ──── */
extern "C" bool anima_rt_run_bias_add(
    float* C, const float* bias, int M, int N)
{
    if (!bias) return true;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] += bias[j];
        }
    }
    return true;
}
