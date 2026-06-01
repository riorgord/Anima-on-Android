"""Real pipeline comparison: C++ engine vs PyTorch manual whitebox.

Both sides use:
  - Same real pipeline inputs (x_flat, ctx_flat from gen_real_inputs.py)
  - Same C++-style t_emb and lora
  - Same C++-style RoPE frequencies (compute_rope_freqs replicated)

The only difference: PyTorch manual ops (F.layer_norm, F.linear, etc.) vs C++ Vulkan shaders.

Usage (WSL2):
    python /mnt/d/AI/anima_phone/scripts/compare_realpipe.py [--phone-dir output/realpipe]
"""
import sys, os, argparse
sys.path.insert(0, '/mnt/d/AI/anima_phone/src')
import torch, torch.nn.functional as F, numpy as np, time

DEV = 'cuda'; DTYPE = torch.float16
M, S, D = 2, 256, 2048; MS = M*S; NH = 16; HD = 128
NCTX, CTXD = 512, 1024; MLP_HIDDEN = 8192; ADALN_LORA_DIM = 256
HP = 16; HALF_DIM = HD // 2; SCALE_ATTN = 1.0 / np.sqrt(HD)
EPS = 1e-6

DIM_H = HD // 6 * 2; DIM_W = DIM_H; DIM_T = HD - 2 * DIM_H

RPDIR = '/mnt/d/AI/anima_phone/output/realpipe'
OUTDIR = RPDIR

# ── C++-style RoPE (from pc_whitebox_ref.py) ──
def compute_rope_freqs_cpp():
    h_ntk = 4.0 ** (DIM_H / (DIM_H - 2))
    w_ntk = 4.0 ** (DIM_W / (DIM_W - 2))
    t_ntk = 1.0 ** (DIM_T / (DIM_T - 2))
    h_theta = 10000.0 * h_ntk; w_theta = 10000.0 * w_ntk; t_theta = 10000.0 * t_ntk
    half_dim = HD // 2
    half_dim_t = DIM_T // 2; half_dim_h = DIM_H // 2; half_dim_w = DIM_W // 2

    freqs = torch.zeros(half_dim, dtype=torch.float32)
    for j in range(half_dim):
        if j < half_dim_t:
            freqs[j] = 1.0 / (t_theta ** (2.0 * j / DIM_T))
        elif j < half_dim_t + half_dim_h:
            freqs[j] = 1.0 / (h_theta ** (2.0 * (j - half_dim_t) / DIM_H))
        else:
            freqs[j] = 1.0 / (w_theta ** (2.0 * (j - half_dim_t - half_dim_h) / DIM_W))

    h_idx = torch.arange(S) // HP; w_idx = torch.arange(S) % HP
    angles = torch.zeros(S, half_dim, dtype=torch.float32)
    for j in range(half_dim):
        if j < half_dim_t: angles[:, j] = 0.0
        elif j < half_dim_t + half_dim_h: angles[:, j] = h_idx.float() * freqs[j]
        else: angles[:, j] = w_idx.float() * freqs[j]
    cos_vals = torch.cos(angles); sin_vals = torch.sin(angles)

    n_rows = M * S * NH
    freqs_out = torch.zeros(n_rows, half_dim, 4, dtype=torch.float16)
    for mb in range(M):
        for p in range(S):
            for h in range(NH):
                dst = mb * S * NH + p * NH + h
                for j in range(half_dim):
                    c, s = cos_vals[p, j].item(), sin_vals[p, j].item()
                    freqs_out[dst, j, 0] = c
                    freqs_out[dst, j, 1] = -s
                    freqs_out[dst, j, 2] = s
                    freqs_out[dst, j, 3] = c
    return freqs_out.to(DEV)

def apply_rope_cpp(q_or_k, freqs):
    ph = q_or_k.shape[0]
    x = q_or_k.view(ph, HALF_DIM, 2).float()
    f = freqs.view(ph, HALF_DIM, 2, 2).float()
    out_r = x[..., 0] * f[..., 0, 0] + x[..., 1] * f[..., 0, 1]
    out_i = x[..., 0] * f[..., 1, 0] + x[..., 1] * f[..., 1, 1]
    return torch.stack([out_r, out_i], dim=-1).view(ph, HD).to(DTYPE)

# ── C++-style t_emb ──
def compute_t_emb_lora(sd, sigma=1.0):
    w1 = sd['t_embedder.1.linear_1.weight'].float().to(DEV)
    w2 = sd['t_embedder.1.linear_2.weight'].float().to(DEV)
    w_ln = sd['t_embedding_norm.weight'].float().to(DEV)
    halfD = D // 2
    j = torch.arange(halfD, dtype=torch.float32, device=DEV)
    freqs = sigma * torch.exp(-torch.log(torch.tensor(10000.0)) * j / halfD)
    sin_emb = torch.zeros(M, D, dtype=torch.float32, device=DEV)
    sin_emb[:, :halfD] = torch.cos(freqs).unsqueeze(0)
    sin_emb[:, halfD:] = torch.sin(freqs).unsqueeze(0)
    rms = torch.sqrt((sin_emb * sin_emb).mean(-1, keepdim=True) + 1e-6)
    t_emb = (sin_emb * w_ln.unsqueeze(0) / rms).to(DTYPE)
    lora = (F.silu(sin_emb @ w1.T) @ w2.T).to(DTYPE)
    return t_emb, lora

# ── AdaLN (matching C++ engine) ──
def compute_adaln(sd, b, t_emb, lora):
    result = {}
    for key_prefix, weight_prefix in [
        ('sa', 'self_attn'), ('cx', 'cross_attn'), ('mlp', 'mlp')
    ]:
        w1 = sd[f'blocks.{b}.adaln_modulation_{weight_prefix}.1.weight'].float().to(DEV)
        w2 = sd[f'blocks.{b}.adaln_modulation_{weight_prefix}.2.weight'].float().to(DEV)
        t_f32, l_f32 = t_emb.float(), lora.float()
        h = F.silu(t_f32) @ w1.T
        out = h @ w2.T + l_f32
        shift, scale, gate = torch.chunk(out, 3, dim=-1)
        result[f'{key_prefix}_shift'] = shift.to(DTYPE)
        result[f'{key_prefix}_scale'] = scale.to(DTYPE) + 1.0
        result[f'{key_prefix}_gate'] = gate.to(DTYPE)
    return result

# ── Per-batch self-attention ──
def self_attention(q_roped, k_roped, v_flat):
    attn_out = torch.zeros(MS * NH, HD, dtype=DTYPE, device=DEV)
    for mb in range(M):
        base = mb * S * NH
        q_mb = q_roped[base:base + S * NH].view(S, NH, HD).permute(1, 0, 2)
        k_mb = k_roped[base:base + S * NH].view(S, NH, HD).permute(1, 0, 2)
        v_mb = v_flat[base:base + S * NH].view(S, NH, HD).permute(1, 0, 2)
        scores = torch.bmm(q_mb, k_mb.transpose(1, 2)) * SCALE_ATTN
        attn_w = F.softmax(scores.float(), dim=-1).to(DTYPE)
        attn_o = torch.bmm(attn_w, v_mb).permute(1, 0, 2).reshape(S * NH, HD)
        attn_out[base:base + S * NH] = attn_o
    return attn_out

# ── Per-batch cross-attention ──
def cross_attention(q_norm, k_norm, v_flat):
    attn_out = torch.zeros(MS * NH, HD, dtype=DTYPE, device=DEV)
    for mb in range(M):
        q_mb = q_norm[mb * S * NH:(mb + 1) * S * NH].view(S, NH, HD).permute(1, 0, 2)
        k_mb = k_norm[mb * NCTX * NH:(mb + 1) * NCTX * NH].view(NCTX, NH, HD).permute(1, 0, 2)
        v_mb = v_flat[mb * NCTX * NH:(mb + 1) * NCTX * NH].view(NCTX, NH, HD).permute(1, 0, 2)
        scores = torch.bmm(q_mb, k_mb.transpose(1, 2)) * SCALE_ATTN
        attn_w = F.softmax(scores.float(), dim=-1).to(DTYPE)
        attn_o = torch.bmm(attn_w, v_mb).permute(1, 0, 2).reshape(S * NH, HD)
        attn_out[mb * S * NH: (mb+1) * S * NH] = attn_o
    return attn_out

# ── White-box block forward ──
def whitebox_block(b, x_flat, ctx_flat, adaln, sd, rope_freqs):
    ph = MS * NH; ph_cross = M * NCTX * NH; MS_kv = M * NCTX
    bc = lambda key: adaln[key].repeat_interleave(S, dim=0)

    # Self-attn
    q_w = sd[f'blocks.{b}.self_attn.q_proj.weight'].to(DEV).to(DTYPE)
    k_w = sd[f'blocks.{b}.self_attn.k_proj.weight'].to(DEV).to(DTYPE)
    v_w = sd[f'blocks.{b}.self_attn.v_proj.weight'].to(DEV).to(DTYPE)
    o_w = sd[f'blocks.{b}.self_attn.output_proj.weight'].to(DEV).to(DTYPE)
    qn_w = sd[f'blocks.{b}.self_attn.q_norm.weight'].to(DEV).to(DTYPE)
    kn_w = sd[f'blocks.{b}.self_attn.k_norm.weight'].to(DEV).to(DTYPE)

    y = F.layer_norm(x_flat, (D,), weight=None, bias=None, eps=EPS)
    y = y * bc('sa_scale') + bc('sa_shift')
    q = F.linear(y, q_w); k = F.linear(y, k_w); v = F.linear(y, v_w)
    q_n = F.rms_norm(q.view(ph, HD), (HD,), weight=qn_w, eps=EPS)
    k_n = F.rms_norm(k.view(ph, HD), (HD,), weight=kn_w, eps=EPS)
    v_flat = v.view(ph, HD)

    q_roped = apply_rope_cpp(q_n, rope_freqs)
    k_roped = apply_rope_cpp(k_n, rope_freqs)
    attn_o = self_attention(q_roped, k_roped, v_flat)
    sa_out = F.linear(attn_o.view(MS, D), o_w)
    x_sa = x_flat + bc('sa_gate') * sa_out

    # Cross-attn
    cx_q_w = sd[f'blocks.{b}.cross_attn.q_proj.weight'].to(DEV).to(DTYPE)
    cx_k_w = sd[f'blocks.{b}.cross_attn.k_proj.weight'].to(DEV).to(DTYPE)
    cx_v_w = sd[f'blocks.{b}.cross_attn.v_proj.weight'].to(DEV).to(DTYPE)
    cx_o_w = sd[f'blocks.{b}.cross_attn.output_proj.weight'].to(DEV).to(DTYPE)
    cx_qn_w = sd[f'blocks.{b}.cross_attn.q_norm.weight'].to(DEV).to(DTYPE)
    cx_kn_w = sd[f'blocks.{b}.cross_attn.k_norm.weight'].to(DEV).to(DTYPE)

    y = F.layer_norm(x_sa, (D,), weight=None, bias=None, eps=EPS)
    y = y * bc('cx_scale') + bc('cx_shift')
    q_cx = F.linear(y, cx_q_w)
    k_cx = F.linear(ctx_flat, cx_k_w)
    v_cx = F.linear(ctx_flat, cx_v_w)
    q_cx_n = F.rms_norm(q_cx.view(ph, HD), (HD,), weight=cx_qn_w, eps=EPS)
    k_cx_n = F.rms_norm(k_cx.view(ph_cross, HD), (HD,), weight=cx_kn_w, eps=EPS)

    cx_attn_o = cross_attention(q_cx_n, k_cx_n, v_cx.view(ph_cross, HD))
    cx_out = F.linear(cx_attn_o.view(MS, D), cx_o_w)
    x_cx = x_sa + bc('cx_gate') * cx_out

    # MLP
    l1_w = sd[f'blocks.{b}.mlp.layer1.weight'].to(DEV).to(DTYPE)
    l2_w = sd[f'blocks.{b}.mlp.layer2.weight'].to(DEV).to(DTYPE)

    y = F.layer_norm(x_cx, (D,), weight=None, bias=None, eps=EPS)
    y = y * bc('mlp_scale') + bc('mlp_shift')
    fc1 = F.gelu(F.linear(y, l1_w))
    fc2 = F.linear(fc1, l2_w)
    x_out = x_cx + bc('mlp_gate') * fc2

    return x_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phone-dir', type=str, default=None,
                        help='Directory with phone C++ dump block_*_cpp.npy')
    args = parser.parse_args()

    # ── Load weights ──
    print("Loading weights...")
    t0 = time.time()
    sd_r = torch.load('/mnt/d/AI/anima_phone/models/diffusion_weights_fp16.pt',
                      weights_only=True, map_location='cpu')
    sd = {}
    for k, v in sd_r.items():
        ck = k
        while ck.startswith('net.'): ck = ck[4:]
        sd[ck] = v
    del sd_r
    print(f"  {len(sd)} keys in {time.time()-t0:.1f}s")

    # ── Load real pipeline inputs ──
    print(f"\nLoading real pipeline inputs from {RPDIR}/")
    x_np = np.load(f'{RPDIR}/x_flat.npy').astype(np.float16)
    ctx_np = np.load(f'{RPDIR}/ctx_flat.npy').astype(np.float16)
    x_in = torch.from_numpy(x_np.astype(np.float32)).to(DEV, DTYPE)
    ctx_in = torch.from_numpy(ctx_np.astype(np.float32)).to(DEV, DTYPE)
    print(f"  x:   {x_in.shape} [{x_in.min():.4f}, {x_in.max():.4f}]")
    print(f"  ctx: {ctx_in.shape} [{ctx_in.min():.4f}, {ctx_in.max():.4f}]")

    # ── Compute C++-style t_emb, lora, RoPE ──
    print("\nComputing C++-style t_emb, lora, RoPE...")
    t_emb, lora = compute_t_emb_lora(sd, sigma=1.0)
    rope_freqs = compute_rope_freqs_cpp()
    print(f"  t_emb: [{t_emb.min():.4f}, {t_emb.max():.4f}]")
    print(f"  lora:  [{lora.min():.4f}, {lora.max():.4f}]")
    print(f"  RoPE:  {rope_freqs.shape}")

    # ── Run white-box 28 blocks ──
    print("\nRunning whitebox 28 blocks (real inputs, C++ RoPE)...")
    t_loop = time.time()
    x = x_in.clone()
    whitebox_outs = []

    for b in range(28):
        adaln = compute_adaln(sd, b, t_emb, lora)
        x = whitebox_block(b, x, ctx_in, adaln, sd, rope_freqs)
        out_np = x.cpu().numpy().astype(np.float16)
        whitebox_outs.append(out_np)
        np.save(f'{OUTDIR}/block_{b:02d}_pt_wb.npy', out_np)

        if b < 5 or b > 22:
            f = out_np[np.isfinite(out_np)]
            nans = np.sum(np.isnan(out_np))
            print(f"  Block {b:2d}: [{f.min():.1f}, {f.max():.1f}] nan={nans}")
        elif b == 5:
            print(f"  ... (blocks 5-22 skipped)")

    dt = time.time() - t_loop
    print(f"  28 blocks in {dt:.1f}s")
    final = whitebox_outs[-1]
    print(f"  Final: [{final.min():.1f}, {final.max():.1f}]")

    # ── Compare with phone dumps ──
    phone_dir = args.phone_dir or RPDIR
    print(f"\n{'='*70}")
    print(f"Comparing with phone dumps: {phone_dir}")
    print(f"{'Block':<6} {'C++':<30} {'Whitebox':<30} {'max_err':<10}")

    for b in range(28):
        cpp_path = f'{phone_dir}/block_{b:02d}_cpp.npy'
        if not os.path.exists(cpp_path):
            continue
        cpp = np.load(cpp_path).astype(np.float32).reshape(512, 2048)
        pt = whitebox_outs[b].astype(np.float32)
        ok = np.isfinite(cpp) & np.isfinite(pt)
        if ok.sum() > 0:
            diff = np.abs(cpp[ok] - pt[ok])
            cr = f'[{cpp[ok].min():.1f},{cpp[ok].max():.1f}]'
            pr = f'[{pt[ok].min():.1f},{pt[ok].max():.1f}]'
            print(f"  {b:2d}   {cr:<30} {pr:<30} {diff.max():<10.2f}")
        else:
            print(f"  {b:2d}   ALL NaN")

    print("\nDONE.")


if __name__ == '__main__':
    main()
