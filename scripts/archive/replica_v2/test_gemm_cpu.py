"""Phase 2b: Verify cpu_gemm_bf16 against PyTorch F.linear.
Tests the C++ head_tail_ops.h GEMM implementation.

Usage (WSL):
  cd /mnt/d/AI/anima_phone && python scripts/replica/test_gemm_cpu.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from replica.common import *
import torch.nn.functional as F

def cpp_style_gemm_bf16(A_fp32, B_bf16, M, N, K, alpha=1.0):
    """Exact replica of C++ head_tail_ops.h cpu_gemm_bf16.
    C[M,N] = A[M,K] @ B[N,K]^T * alpha
    B is stored as uint16_t[] with BF16 bit pattern.
    """
    C = np.zeros((M, N), dtype=np.float32)
    for m in range(M):
        for n in range(N):
            s = 0.0
            for k in range(K):
                a = A_fp32[m * K + k]
                # BF16 unpack (uint16_t → float)
                b_bits = int(B_bf16[n * K + k])
                f32 = struct.unpack('f', struct.pack('I', b_bits << 16))[0]
                s = fma_f32(a, f32, s)
            C[m, n] = s * alpha
    return C

def fma_f32(a, b, c):
    """Emulate fma(a, b, c) = a*b + c with single rounding.
    In Python: we use float (double precision in Python), then cast to np.float32.
    """
    return np.float32(np.float64(a) * np.float64(b) + np.float64(c))

def main():
    print("=" * 70)
    print("Phase 2b: cpu_gemm_bf16 Verification")
    print("=" * 70)

    sd = load_weights()

    tests = [
        ("Q_proj [512,2048]@[2048,2048]^T", MS, D, D, "blocks.0.self_attn.q_proj.weight"),
        ("MLP fc1 [512,2048]@[8192,2048]^T", MS, D, MLP_HIDDEN, "blocks.0.mlp.layer1.weight"),
        ("AdaLN down [2,2048]@[256,2048]^T", M, D, ADALN_LORA_DIM, "blocks.0.adaln_modulation_self_attn.1.weight"),
        ("cross K/V [1024,1024]@[2048,1024]^T", M*NCTX, CTXD, D, "blocks.0.cross_attn.k_proj.weight"),
    ]

    for label, Mv, Kv, Nv, wkey in tests:
        print(f"\n── {label} ──")
        A = rand_input(Mv, Kv, device="cpu", dtype=torch.float32).numpy()
        w_fp32 = sd[wkey].to(torch.float32)
        # Convert to BF16 and back to get uint16 bit pattern (simulating BF16 storage)
        w_bf16 = w_fp32.to(torch.bfloat16)
        B_bf16 = w_bf16.view(torch.uint16).numpy().reshape(Nv, Kv).ravel()  # BF16 as uint16 raw

        # C++ style
        t0 = time.time()
        C_cpp = cpp_style_gemm_bf16(A.ravel(), B_bf16.ravel(), Mv, Nv, Kv)
        t_cpp = time.time() - t0

        # PyTorch reference: use BF16→FP32 weight (matching C++ behavior)
        A_t = torch.from_numpy(A).reshape(Mv, Kv)
        B_t = w_fp32.reshape(Nv, Kv)  # already FP32 from BF16 source
        t1 = time.time()
        C_pt = F.linear(A_t, B_t).numpy()
        t_pt = time.time() - t1

        r = compare(C_cpp, C_pt, label)
        status = "✓ PASS" if r['max_err'] < 1e-6 else f"⚠️ max_err={r['max_err']:.2e}"
        print(f"  C++: {t_cpp*1000:.1f}ms, PT: {t_pt*1000:.1f}ms")
        print(f"  max_err={r['max_err']:.2e}, mean_err={r['mean_err']:.2e} {status}")

        # Check against baseline
        baseline_path = os.path.join(REPLICA_DIR, "cpu_cuda_baseline.json")
        if os.path.exists(baseline_path):
            with open(baseline_path) as f:
                baseline = json.load(f)
            for b in baseline:
                blabel = b.get("label", "")
                if ("GEMM" in blabel and str(Kv) in blabel) or \
                   ("QK^T" in blabel and label.startswith("QK")):
                    target = b["target_1_5x"]
                    if r['max_err'] <= target:
                        print(f"  ✓ Within 1.5× baseline ({target:.2e})")
                    else:
                        print(f"  ⚠️ Exceeds baseline target ({target:.2e})")
                    break

if __name__ == "__main__":
    main()
