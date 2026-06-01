"""Phase 1: Establish CPU vs CUDA baseline error for each DiT op.
Measures PyTorch's own CPU fp32 vs CUDA fp32 error — this is our target ceiling.
Our Vulkan vs CUDA error target = baseline * 1.5.

Usage (WSL):
  source /home/riorg/miniconda3/etc/profile.d/conda.sh
  conda activate /home/riorg/anima-work/.conda
  cd /mnt/d/AI/anima_phone && python scripts/replica/bench_cpu_cuda.py
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from replica.common import *

N_RUNS = 10  # number of runs per op for stability

def bench_op(label, cpu_fn, cuda_fn, n_runs=N_RUNS):
    """Run CPU and CUDA versions, return max error across runs."""
    max_errs = []
    for _ in range(n_runs):
        cpu_out = cpu_fn().cpu().numpy()
        cuda_out = cuda_fn().cpu().numpy()
        r = compare(cpu_out, cuda_out, label)
        if not np.isnan(r["max_err"]):
            max_errs.append(r["max_err"])
    if not max_errs:
        return {"label": label, "max_err": float('nan'), "mean_err": float('nan'), "n_runs": 0}
    return {
        "label": label,
        "max_err": max(max_errs),
        "mean_err": float(np.mean(max_errs)),
        "n_runs": len(max_errs),
        "target_1_5x": max(max_errs) * 1.5,
    }

def main():
    print("=" * 70)
    print("Phase 1: CPU vs CUDA Baseline Error Measurement")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This script needs GPU.")
        print("Run on WSL with CUDA-capable GPU.")
        sys.exit(1)

    device_cpu = "cpu"
    device_cuda = "cuda"
    dtype = torch.float32

    print(f"\nLoading weights from {SF_PATH}...")
    sd = load_weights(SF_PATH, device="cpu", dtype=dtype)  # keep on CPU for CPU path

    results = []

    # ═══════════════════════════════════════════════════════
    # 1. GEMM — various K sizes
    # ═══════════════════════════════════════════════════════
    print("\n── GEMM ──")
    # Pick the first block's self_attn.q_proj.weight as [D, D] test weight
    w_key = "blocks.0.self_attn.q_proj.weight"
    w_dd = get_weight(sd, w_key, device="cpu", dtype=dtype)  # [2048, 2048]

    for label, Mv, Kv, Nv in [
        ("GEMM K=2048 (Q_proj)", MS, D, D),
        ("GEMM K=1024 (cross K/V)", M*NCTX, CTXD, D),
        ("GEMM K=8192 (MLP fc1)", MS, D, MLP_HIDDEN),
        ("GEMM K=256 (AdaLN up)", M, ADALN_LORA_DIM, D),
    ]:
        # Use a random submatrix of actual weights for realistic value distribution
        x = rand_input(Mv, Kv, device="cpu", dtype=dtype)
        if Nv == D and Kv == D:
            w = w_dd.to(device="cpu")
        elif Nv == D and Kv == MLP_HIDDEN:
            w = get_weight(sd, "blocks.0.mlp.layer1.weight", "cpu", dtype)  # [8192, 2048]
            w = w[:Nv, :Kv].contiguous()
        elif Nv == D and Kv == CTXD:
            w = get_weight(sd, "blocks.0.cross_attn.k_proj.weight", "cpu", dtype)
        elif Nv == D and Kv == ADALN_LORA_DIM:
            w = get_weight(sd, "blocks.0.adaln_modulation_self_attn.2.weight", "cpu", dtype)
            w = w[:D, :Kv].contiguous()
        else:
            w = rand_bf16_weight(Nv, Kv, device="cpu")

        def make_fn(x, w, Mv, Nv, Kv):
            def fn(dev):
                xx = x.to(dev); ww = w.to(dev)
                return F.linear(xx[:Mv, :Kv].reshape(Mv, Kv), ww[:Nv, :Kv].reshape(Nv, Kv))
            return fn

        cpu_f = make_fn(x, w, Mv, Nv, Kv)
        x_c = x.clone(); w_c = w.clone()
        cuda_f = lambda: F.linear(x_c.to(device_cuda)[:Mv, :Kv].reshape(Mv, Kv),
                                   w_c.to(device_cuda)[:Nv, :Kv].reshape(Nv, Kv))

        r = bench_op(label, lambda: cpu_f(device_cpu), cuda_f)
        print(f"  {label}: max_err={r['max_err']:.2e}, target(1.5×)={r['target_1_5x']:.2e}")
        results.append(r)

    # ═══════════════════════════════════════════════════════
    # 2. LayerNorm
    # ═══════════════════════════════════════════════════════
    print("\n── LayerNorm ──")
    for label, rows, elems in [("LN 512×2048 (block)", MS, D), ("LN 2×2048 (t_emb)", M, D)]:
        x = rand_input(rows, elems, device="cpu", dtype=dtype)
        def cpu_f(): return pt_layernorm(x, rows, elems)
        x_c = x.clone()
        cuda_f = lambda: pt_layernorm(x_c.to(device_cuda), rows, elems)
        r = bench_op(label, cpu_f, cuda_f)
        print(f"  {label}: max_err={r['max_err']:.2e}, target(1.5×)={r['target_1_5x']:.2e}")
        results.append(r)

    # ═══════════════════════════════════════════════════════
    # 3. RMSNorm
    # ═══════════════════════════════════════════════════════
    print("\n── RMSNorm ──")
    for label, rows, elems, wkey in [
        ("RMSNorm 8192×128 (Q/K)", MS*NH, HD, "blocks.0.self_attn.q_norm.weight"),
        ("RMSNorm 2×2048 (t_norm)", M, D, "t_embedding_norm.weight"),
    ]:
        x = rand_input(rows, elems, device="cpu", dtype=dtype)
        w_cpu = get_weight(sd, wkey, "cpu", dtype)
        def cpu_f(): return pt_rmsnorm(x, rows, elems, w_cpu)
        x_c, w_c = x.clone(), w_cpu.clone()
        cuda_f = lambda: pt_rmsnorm(x_c.to(device_cuda), rows, elems,
                                      w_c.to(device_cuda))
        r = bench_op(label, cpu_f, cuda_f)
        print(f"  {label}: max_err={r['max_err']:.2e}, target(1.5×)={r['target_1_5x']:.2e}")
        results.append(r)

    # ═══════════════════════════════════════════════════════
    # 4. GELU
    # ═══════════════════════════════════════════════════════
    print("\n── GELU ──")
    x = rand_input(MS * MLP_HIDDEN, device="cpu", dtype=dtype) * 2.0  # realistic range
    def cpu_f(): return pt_gelu(x)
    x_c = x.clone()
    cuda_f = lambda: pt_gelu(x_c.to(device_cuda))
    r = bench_op("GELU MS×8192", cpu_f, cuda_f)
    print(f"  GELU: max_err={r['max_err']:.2e}, target(1.5×)={r['target_1_5x']:.2e}")
    results.append(r)

    # ═══════════════════════════════════════════════════════
    # 5. SiLU
    # ═══════════════════════════════════════════════════════
    print("\n── SiLU ──")
    x = rand_input(M * D, device="cpu", dtype=dtype) * 2.0
    def cpu_f(): return pt_silu(x)
    x_c = x.clone()
    cuda_f = lambda: pt_silu(x_c.to(device_cuda))
    r = bench_op("SiLU 2×2048", cpu_f, cuda_f)
    print(f"  SiLU: max_err={r['max_err']:.2e}, target(1.5×)={r['target_1_5x']:.2e}")
    results.append(r)

    # ═══════════════════════════════════════════════════════
    # 6. Softmax
    # ═══════════════════════════════════════════════════════
    print("\n── Softmax ──")
    # Per-batch self-attention: S query tokens × H heads × S key tokens
    x = rand_input(S * NH, S, device="cpu", dtype=dtype)
    M_q, M_kv, Hv = S, S, NH
    def cpu_f(): return pt_softmax_lastdim(x.reshape(M_q, Hv, M_kv), M_q, Hv, M_kv)
    x_c = x.clone()
    cuda_f = lambda: pt_softmax_lastdim(x_c.to(device_cuda).reshape(M_q, Hv, M_kv), M_q, Hv, M_kv)
    r = bench_op("Softmax 256×256", cpu_f, cuda_f)
    print(f"  Softmax: max_err={r['max_err']:.2e}, target(1.5×)={r['target_1_5x']:.2e}")
    results.append(r)

    # ═══════════════════════════════════════════════════════
    # 7. Sin/Cos embedding
    # ═══════════════════════════════════════════════════════
    print("\n── Sin/Cos embedding (Timesteps) ──")
    sigma = 1.0
    def cpu_f(): return pt_timesteps(sigma)
    def cuda_f(): return pt_timesteps(torch.tensor([sigma, sigma], device=device_cuda).unsqueeze(1))
    r = bench_op("Sin/Cos embedding sigma=1.0", cpu_f, cuda_f)
    print(f"  Timesteps: max_err={r['max_err']:.2e}, target(1.5×)={r['target_1_5x']:.2e}")
    results.append(r)

    # ═══════════════════════════════════════════════════════
    # 8. AdaLN chain (SiLU→Linear→Linear→chunk)
    # ═══════════════════════════════════════════════════════
    print("\n── AdaLN chain ──")
    t_emb = torch.from_numpy(pt_timesteps(sigma).numpy()).to(device_cpu)
    def cpu_f():
        s, sc, g = pt_adaln_chain(t_emb, sd, 0, "self_attn", "cpu", dtype)
        return torch.cat([s, sc, g], dim=-1)
    t_emb_c = t_emb.clone()
    # Move a copy of weights to GPU
    sd_cuda = {k: v.to(device_cuda) for k, v in sd.items()
               if "blocks.0.adaln_modulation_self_attn" in k}
    def cuda_f():
        s, sc, g = pt_adaln_chain(t_emb_c.to(device_cuda), sd_cuda, 0, "self_attn", device_cuda, dtype)
        return torch.cat([s, sc, g], dim=-1)
    r = bench_op("AdaLN SA chain block 0", cpu_f, cuda_f)
    print(f"  AdaLN chain: max_err={r['max_err']:.2e}, target(1.5×)={r['target_1_5x']:.2e}")
    results.append(r)

    # ═══════════════════════════════════════════════════════
    # 9. Attention QK^T + Softmax + AV (single head pair)
    # ═══════════════════════════════════════════════════════
    print("\n── Attention QK^T ──")
    Q = rand_input(S, HD, device="cpu", dtype=dtype)
    K = rand_input(S, HD, device="cpu", dtype=dtype)
    def cpu_f(): return (Q @ K.T) / np.sqrt(HD)
    Q_c, K_c = Q.clone(), K.clone()
    cuda_f = lambda: (Q_c.to(device_cuda) @ K_c.to(device_cuda).T) / np.sqrt(HD)
    r = bench_op("QK^T S×D @ D×S", cpu_f, cuda_f)
    print(f"  QK^T: max_err={r['max_err']:.2e}, target(1.5×)={r['target_1_5x']:.2e}")
    results.append(r)

    print("\n── Attention AV ──")
    Attn = torch.softmax(rand_input(S, S, device="cpu", dtype=dtype), dim=-1)
    V = rand_input(S, HD, device="cpu", dtype=dtype)
    def cpu_f(): return Attn @ V
    Attn_c, V_c = Attn.clone(), V.clone()
    cuda_f = lambda: Attn_c.to(device_cuda) @ V_c.to(device_cuda)
    r = bench_op("AV S×S @ S×D", cpu_f, cuda_f)
    print(f"  AV: max_err={r['max_err']:.2e}, target(1.5×)={r['target_1_5x']:.2e}")
    results.append(r)

    # ═══════════════════════════════════════════════════════
    # Summary & Save
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("BASELINE SUMMARY: CPU vs CUDA fp32 max_err")
    print("=" * 70)
    for r in results:
        print(f"  {r['label']:<40s} max_err={r['max_err']:.2e}  target={r['target_1_5x']:.2e}")

    # Find the ceiling for each op category
    out_path = os.path.join(REPLICA_DIR, "cpu_cuda_baseline.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")
    print("Done.")

if __name__ == "__main__":
    main()
