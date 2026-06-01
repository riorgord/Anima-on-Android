"""Phase 2a: Verify t_embed (sin/cos embedding + SiLU + Linear chain).
Tests that our C++ head_tail_ops.h implementation matches PyTorch exactly.

BUG FOUND: head_tail_ops.h:209-214 swaps sin and cos halves.
PyTorch: torch.cat([sin_emb, cos_emb], dim=-1) → [sin | cos]
C++:     [cos | sin] ← WRONG

Usage (WSL):
  cd /mnt/d/AI/anima_phone && python scripts/replica/test_t_embed.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from replica.common import *
import torch.nn.functional as F

def cpp_t_embed_wrong(sigmas, M_val, D_val, w1_bf16_np, w2_bf16_np):
    """Replicate the WRONG C++ head_tail_ops.h t_embed (sin/cos swapped)."""
    half = D_val // 2
    emb = np.zeros((M_val, D_val), dtype=np.float32)

    # Step 1: Sinusoidal embedding — BUG: cos goes to [0:half], sin goes to [half:D]
    for m in range(M_val):
        s = float(sigmas[m])
        for i in range(half):
            exponent = -np.log(10000.0) * i / half
            freq = np.exp(exponent)
            val = s * freq
            emb[m, i] = np.cos(val)          # WRONG: cos in first half
            emb[m, half + i] = np.sin(val)   # WRONG: sin in second half

    # Step 2: SiLU(emb @ w1^T)
    w1 = torch.from_numpy(w1_bf16_np).to(torch.float32)  # [D, D]
    emb_t = torch.from_numpy(emb)
    h1 = F.linear(emb_t, w1)
    h1 = F.silu(h1)

    # Step 3: h1 @ w2^T → adaln_lora [M, 3D], t_emb = raw emb
    w2 = torch.from_numpy(w2_bf16_np).to(torch.float32)  # [3D, D]
    adaln_lora = F.linear(h1, w2).numpy()
    return emb, adaln_lora  # raw emb [M, D], lora [M, 3D]

def pt_t_embed_correct(sigmas, M_val, D_val, w1_bf16_np, w2_bf16_np):
    """Correct PyTorch version: sin→[0:half], cos→[half:D]."""
    half = D_val // 2
    emb = np.zeros((M_val, D_val), dtype=np.float32)

    # Step 1: Sinusoidal embedding — CORRECT order: sin first, cos second
    for m in range(M_val):
        s = float(sigmas[m])
        for i in range(half):
            exponent = -np.log(10000.0) * i / half
            freq = np.exp(exponent)
            val = s * freq
            emb[m, i] = np.sin(val)           # CORRECT: sin in first half
            emb[m, half + i] = np.cos(val)    # CORRECT: cos in second half

    # Step 2: SiLU(emb @ w1^T)
    w1 = torch.from_numpy(w1_bf16_np).to(torch.float32)  # [D, D]
    emb_t = torch.from_numpy(emb)
    h1 = F.linear(emb_t, w1)
    h1 = F.silu(h1)

    # Step 3: h1 @ w2^T → adaln_lora [M, 3D], t_emb = raw emb
    w2 = torch.from_numpy(w2_bf16_np).to(torch.float32)  # [3D, D]
    adaln_lora = F.linear(h1, w2).numpy()
    return emb, adaln_lora

def main():
    print("=" * 70)
    print("Phase 2a: t_embed Verification")
    print("=" * 70)

    sd = load_weights()
    w1_key = "t_embedder.1.linear_1.weight"  # [2048, 2048]
    w2_key = "t_embedder.1.linear_2.weight"  # [6144, 2048]

    w1_bf16 = sd[w1_key].to(torch.float32).numpy()
    w2_bf16 = sd[w2_key].to(torch.float32).numpy()

    sigmas = [1.0, 1.0]  # CFG batch

    # 1. Test wrong (C++ current) version
    emb_wrong, lora_wrong = cpp_t_embed_wrong(sigmas, M, D, w1_bf16, w2_bf16)

    # 2. Test correct (PyTorch) version via our helper
    emb_correct, lora_correct = pt_t_embed_correct(sigmas, M, D, w1_bf16, w2_bf16)

    # 3. Reference via actual PyTorch model class
    emb_ref = pt_timesteps(sigmas).numpy()
    emb_t = torch.from_numpy(emb_ref)
    h1 = F.linear(emb_t, torch.from_numpy(w1_bf16).to(torch.float32))
    h1 = F.silu(h1)
    lora_ref = F.linear(h1, torch.from_numpy(w2_bf16).to(torch.float32)).numpy()

    # 4. Compare
    print("\n── Raw sin/cos embedding ──")
    r1 = compare(emb_wrong, emb_ref, "C++ WRONG vs PT ref")
    r2 = compare(emb_correct, emb_ref, "C++ FIXED vs PT ref")
    print(f"  WRONG (cos|sin): max_err={r1['max_err']:.6f} {'❌ BROKEN' if r1['max_err'] > 1e-3 else '✓'}")
    print(f"  FIXED (sin|cos): max_err={r2['max_err']:.6f} {'✓ CORRECT' if r2['max_err'] < 1e-6 else '⚠️'}")

    print("\n── After SiLU + Linear chain (adaln_lora) ──")
    r3 = compare(lora_wrong, lora_ref, "C++ WRONG vs PT ref")
    r4 = compare(lora_correct, lora_ref, "C++ FIXED vs PT ref")
    print(f"  WRONG: max_err={r3['max_err']:.6f} {'❌ BROKEN' if r3['max_err'] > 1e-3 else '✓'}")
    print(f"  FIXED: max_err={r4['max_err']:.6f} {'✓ CORRECT' if r4['max_err'] < 1e-6 else '⚠️'}")

    print(f"\n  WRONG lora range: [{lora_wrong.min():.4f}, {lora_wrong.max():.4f}]")
    print(f"  FIXED lora range: [{lora_correct.min():.4f}, {lora_correct.max():.4f}]")
    print(f"  REF   lora range: [{lora_ref.min():.4f}, {lora_ref.max():.4f}]")

    # Save fixed version for C++ implementation verification
    save_npy(os.path.join(REPLICA_DIR, "t_emb_ref.npy"), emb_correct)
    save_npy(os.path.join(REPLICA_DIR, "adaln_lora_ref.npy"), lora_correct)
    print("\nSaved reference outputs to scripts/replica/")

    # Check against baseline
    baseline_path = os.path.join(REPLICA_DIR, "cpu_cuda_baseline.json")
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline = json.load(f)
        for b in baseline:
            if "Sin/Cos" in b.get("label", ""):
                target = b["target_1_5x"]
                print(f"\n  Baseline target for timesteps: {target:.2e}")
                print(f"  Fixed version max_err: {r2['max_err']:.2e}")
                if r2['max_err'] <= target:
                    print(f"  ✓ PASS (within 1.5× CPU-vs-CUDA)")
                else:
                    print(f"  ⚠️ EXCEEDS target")
                break

if __name__ == "__main__":
    main()
