"""Phase 4: Verify each GLSL shader algorithm against PyTorch reference.
Simulates each shader's exact algorithm in Python/NumPy, compares with PyTorch.
Uses fp32 throughout, matching the v2 shader precision.

Usage (WSL):
  cd /mnt/d/AI/anima_phone && python scripts/replica/verify_shaders.py
"""
import sys, os, json, struct, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/mnt/d/AI/anima_phone/hybridops/src")
from replica.common import *
import torch.nn.functional as F

# ═══════════════════════════════════════════════════════════════
# Shader algorithm replicas (exact Python port of GLSL logic)
# ═══════════════════════════════════════════════════════════════

def bf16_unpack(packed, idx):
    """Replicate GLSL: bf16_unpack(uint packed, uint idx)."""
    p = int(packed)  # ensure Python int
    bits = (p & 0xFFFF) if idx == 0 else (p >> 16)
    return struct.unpack('f', struct.pack('I', (bits << 16) & 0xFFFFFFFF))[0]

def shader_gemm_bf16(A, B_packed, Mv, Nv, Kv, alpha=1.0):
    """Replicate gemm_bf16.comp algorithm.
    A: [M, K] fp32. B_packed: [N, K/2] uint32 (2 BF16 per uint).
    Workgroup: 8x8. One thread per output element.
    Thread(row,col) does sequential fma over K.
    """
    C = np.zeros((Mv, Nv), dtype=np.float32)
    A_flat = A.ravel()  # linear access for shader-style indexing
    B_flat = B_packed.ravel()  # linear access for shader-style indexing
    k2 = Kv // 2  # uints per row of B
    k4 = Kv // 4  # 4-element iterations
    for row in range(Mv):
        for col in range(Nv):
            s = np.float32(0.0)
            # Unrolled 4-element loop (matches shader)
            for k in range(k4):
                a0 = np.float32(A_flat[row * Kv + 4*k])
                a1 = np.float32(A_flat[row * Kv + 4*k + 1])
                a2 = np.float32(A_flat[row * Kv + 4*k + 2])
                a3 = np.float32(A_flat[row * Kv + 4*k + 3])
                b0 = B_flat[col * k2 + 2*k]
                b1 = B_flat[col * k2 + 2*k + 1]
                s = np.float32(np.float64(a0) * np.float64(bf16_unpack(b0, 0)) + np.float64(s))
                s = np.float32(np.float64(a1) * np.float64(bf16_unpack(b0, 1)) + np.float64(s))
                s = np.float32(np.float64(a2) * np.float64(bf16_unpack(b1, 0)) + np.float64(s))
                s = np.float32(np.float64(a3) * np.float64(bf16_unpack(b1, 1)) + np.float64(s))
            # Remainder
            rem_start = k4 * 4
            for k in range(rem_start, Kv):
                a = np.float32(A_flat[row * Kv + k])
                bp = B_flat[col * k2 + k // 2]
                b = bf16_unpack(bp, k & 1)
                s = np.float32(np.float64(a) * np.float64(b) + np.float64(s))
            C[row, col] = np.float32(s * alpha)
    return C

def shader_layernorm(in_data, rows, elems, eps=1e-6):
    """Replicate layernorm_fp32.comp algorithm.
    3-pass: mean→var→norm. Tree reduce with 256 threads.
    """
    in_f = in_data.ravel()  # ensure flat
    out = np.zeros((rows, elems), dtype=np.float32)
    for r in range(rows):
        off = r * elems
        # Pass 1: mean (sequential sum for single-thread simulation)
        my_sum = np.float32(0.0)
        for i in range(elems):
            my_sum = np.float32(np.float64(my_sum) + np.float64(in_f[off + i]))
        mean = np.float32(my_sum / np.float32(elems))

        # Pass 2: variance
        my_sq = np.float32(0.0)
        for i in range(elems):
            d = np.float32(in_f[off + i] - mean)
            my_sq = np.float32(np.float64(my_sq) + np.float64(d * d))
        inv_std = np.float32(1.0 / np.sqrt(np.float64(my_sq) / np.float64(elems) + eps))

        # Pass 3: normalize
        for i in range(elems):
            out[r, i] = np.float32((in_f[off + i] - mean) * inv_std)
    return out

def shader_rmsnorm(in_data, weight_packed, rows, elems, eps=1e-6):
    """Replicate rms_norm_fp32.comp algorithm.
    Sum squares → tree reduce → rsqrt → multiply by weight.
    weight_packed: uint32[] (2 BF16 per uint), same as GLSL binding 1.
    """
    out = np.zeros((rows, elems), dtype=np.float32)
    in_f = in_data.ravel()
    for r in range(rows):
        off = r * elems
        # Sum squares
        my_sq = np.float32(0.0)
        for i in range(elems):
            v = in_f[off + i]
            my_sq = np.float32(np.float64(my_sq) + np.float64(v * v))
        rms = np.float32(1.0 / np.sqrt(np.float64(my_sq) / np.float64(elems) + eps))
        # Apply norm and weight
        for i in range(elems):
            v = np.float32(in_f[off + i] * rms)
            w = bf16_unpack(weight_packed[i // 2], i & 1)
            out[r, i] = np.float32(v * w)
    return out

def shader_silu(in_data, n):
    """Replicate silu_fp32.comp: x / (1 + exp(-x))."""
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        x = in_data[i]
        out[i] = np.float32(x / (1.0 + np.exp(-x)))
    return out

def shader_gelu(in_data, n):
    """Replicate gelu_fp32.comp: 0.5*x*(1+tanh(beta*(x+kappa*x^3)))."""
    beta = np.float32(0.7978845608)
    kappa = np.float32(0.044715)
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        x = np.float32(in_data[i])
        x3 = np.float32(x * x * x)
        inner = np.float32(beta * (x + kappa * x3))
        out[i] = np.float32(0.5 * x * (1.0 + np.tanh(inner)))
    return out

def shader_scale_shift(x, scl, sft, n, scale_stride, shift_stride):
    """Replicate scale_shift_fp32.comp: out = fma(x, s, b)."""
    x_f = x.ravel(); scl_f = scl.ravel(); sft_f = sft.ravel()
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        s = scl_f[i * scale_stride] if scale_stride > 0 else scl_f[0]
        b = sft_f[i * shift_stride] if shift_stride > 0 else sft_f[0]
        out[i] = np.float32(np.float64(x_f[i]) * np.float64(s) + np.float64(b))
    return out

def shader_gate(oproj, gate, residual, n):
    """Replicate gate_fp32.comp: out = fma(g, o, x) = x + gate * o."""
    o_f = oproj.ravel(); g_f = gate.ravel(); r_f = residual.ravel()
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        out[i] = np.float32(np.float64(g_f[i]) * np.float64(o_f[i]) + np.float64(r_f[i]))
    return out

def shader_broadcast(in_data, Mv, Dv, repeat):
    """Replicate broadcast_fp32.comp: out[i*repeat+r][d] = in[i][d]."""
    in_f = in_data.ravel()
    out = np.zeros(Mv * repeat * Dv, dtype=np.float32)
    for idx in range(Mv * repeat * Dv):
        out_row = idx // Dv
        col = idx % Dv
        in_row = out_row // repeat
        out[idx] = in_f[in_row * Dv + col]
    return out

def shader_attn_qkt(Q, K, M_q, M_kv, H, Dv, scale):
    """Replicate attn_qkt_fp32.comp.
    One workgroup per (m_q, h). 256 threads per WG.
    Each thread handles M_kv values in strides of 256.
    Per-thread dot product over D (sequential fma).
    """
    Q_f = Q.ravel(); K_f = K.ravel()
    A = np.zeros((M_q * H, M_kv), dtype=np.float32)
    for gid in range(M_q * H):
        m_q = gid // H
        h = gid % H
        q_off = m_q * H * Dv + h * Dv
        # Simulate 256 threads, each handles M_kv/256 key positions
        for m_kv in range(M_kv):
            k_off = m_kv * H * Dv + h * Dv
            s = np.float32(0.0)
            for d in range(Dv):
                s = np.float32(np.float64(Q_f[q_off + d]) * np.float64(K_f[k_off + d]) + np.float64(s))
            A[gid, m_kv] = np.float32(s * scale)
    return A

def shader_attn_softmax(A, M_q, M_kv, H):
    """Replicate attn_softmax_fp32.comp: 3-pass safe softmax."""
    orig_shape = A.shape
    A_f = A.ravel().copy()  # ensure flat, work on copy
    for gid in range(M_q * H):
        off = gid * M_kv
        # Pass 1: max
        row_max = np.float32(-1e30)
        for i in range(M_kv):
            if A_f[off + i] > row_max:
                row_max = A_f[off + i]
        # Pass 2: exp sum
        my_sum = np.float32(0.0)
        for i in range(M_kv):
            my_sum = np.float32(np.float64(my_sum) + np.float64(np.exp(A_f[off + i] - row_max)))
        inv_sum = np.float32(1.0 / my_sum)
        # Pass 3: normalize
        for i in range(M_kv):
            A_f[off + i] = np.float32(np.exp(A_f[off + i] - row_max) * inv_sum)
    return A_f.reshape(orig_shape)

def shader_attn_out(A, V, M_q, M_kv, H, Dv):
    """Replicate attn_out_fp32.comp: O = A @ V."""
    A_f = A.ravel(); V_flat = V.ravel()
    O = np.zeros(M_q * H * Dv, dtype=np.float32)  # flat output
    for gid in range(M_q * H):
        m_q = gid // H
        h = gid % H
        a_off = gid * M_kv
        o_off = m_q * H * Dv + h * Dv
        for d in range(Dv):
            s = np.float32(0.0)
            for m_kv in range(M_kv):
                s = np.float32(np.float64(A_f[a_off + m_kv]) * np.float64(V_flat[m_kv * H * Dv + h * Dv + d]) + np.float64(s))
            O[o_off + d] = s
    return O.reshape(M_q, H, Dv)

def shader_rope(t_in, freqs, Nv, head_dim):
    """Replicate rope_fp32.comp: 2D complex rotation."""
    half = head_dim // 2
    out_f = np.zeros(Nv * head_dim, dtype=np.float32)
    t_flat = t_in.ravel()
    for idx in range(Nv):
        t_base = idx * head_dim
        for i in range(half):
            a = t_flat[t_base + 2*i]
            b = t_flat[t_base + 2*i + 1]
            f_base = (idx * half + i) * 4
            c  = freqs[f_base + 0]
            ms = freqs[f_base + 1]
            s  = freqs[f_base + 2]
            mc = freqs[f_base + 3]
            out_f[t_base + 2*i]     = np.float32(np.float64(c) * np.float64(a) + np.float64(ms) * np.float64(b))
            out_f[t_base + 2*i + 1] = np.float32(np.float64(s) * np.float64(a) + np.float64(mc) * np.float64(b))
    return out_f.reshape(Nv, head_dim)

# ═══════════════════════════════════════════════════════════════
# Test runner
# ═══════════════════════════════════════════════════════════════

def verify(label, shader_out, pt_out, baseline_label=None):
    """Compare shader output vs PyTorch reference."""
    r = compare(shader_out.ravel(), pt_out.cpu().numpy().ravel(), label)
    status = "✓ PASS" if r['max_err'] < 1e-5 else f"⚠️ max_err={r['max_err']:.2e}"

    # Check against baseline
    baseline_path = os.path.join(REPLICA_DIR, "cpu_cuda_baseline.json")
    if baseline_path and os.path.exists(baseline_path):
        with open(baseline_path) as f:
            bl = json.load(f)
        for b in bl:
            if baseline_label and baseline_label in b.get("label", ""):
                target = b["target_1_5x"]
                if r['max_err'] <= target:
                    status += f"  ✓ within 1.5× baseline ({target:.2e})"
                else:
                    status += f"  ✗ EXCEEDS {target:.2e}"
                break

    print(f"  {label}: max_err={r['max_err']:.2e}, mean_err={r['mean_err']:.2e} {status}")
    return r

def main():
    print("=" * 70)
    print("Phase 4: Shader Algorithm Verification vs PyTorch")
    print("=" * 70)

    sd = load_weights()
    results = []

    # ── 1. GEMM ──
    print("\n── 1. GEMM (gemm_bf16.comp) ──")
    for label, Mv, Kv, Nv, wkey in [
        # Only test small GEMMs (algorithm identical for all sizes)
        ("AdaLN_up [2,256]@[256,256]^T", M, ADALN_LORA_DIM, 256, "blocks.0.adaln_modulation_self_attn.2.weight"),
    ]:
        x_t = rand_input(Mv, Kv, device="cpu", dtype=torch.float32)
        w_t = sd[wkey]  # loaded as fp32 by load_weights
        # Subset: take only first Nv rows, first Kv cols
        w_t_sub = w_t[:Nv, :Kv].contiguous()
        # Convert back to BF16 to simulate the Vulkan buffer storage
        w_bf16 = w_t_sub.to(torch.bfloat16)
        w_u16 = w_bf16.view(torch.uint16).numpy().reshape(Nv, Kv)
        B_packed = np.zeros((Nv, Kv // 2), dtype=np.uint32)
        for n in range(Nv):
            for k2 in range(Kv // 2):
                lo = int(w_u16[n, 2*k2])
                hi = int(w_u16[n, 2*k2 + 1])
                B_packed[n, k2] = np.uint32(lo | (hi << 16))

        A_np = x_t.numpy().reshape(Mv, Kv)

        t0 = time.time()
        C_shader = shader_gemm_bf16(A_np, B_packed, Mv, Nv, Kv)
        t_shader = time.time() - t0

        # PyTorch reference (use same subset)
        w_fp32 = w_t_sub.to(torch.float32).reshape(Nv, Kv)
        t1 = time.time()
        C_pt = F.linear(x_t.reshape(Mv, Kv), w_fp32)
        t_pt = time.time() - t1

        r = verify(label, C_shader, C_pt, f"GEMM K={Kv}")
        results.append(r)
        print(f"    shader sim: {t_shader*1000:.0f}ms, PT: {t_pt*1000:.0f}ms")

    # ── 2. LayerNorm ──
    print("\n── 2. LayerNorm (layernorm_fp32.comp) ──")
    x = rand_input(MS, D, device="cpu", dtype=torch.float32) * 2.0
    C_shader = shader_layernorm(x.numpy().reshape(MS, D), MS, D).ravel()
    C_pt = pt_layernorm(x, MS, D)
    results.append(verify("LN 512×2048", C_shader, C_pt, "LN 512"))

    # ── 3. RMSNorm ──
    print("\n── 3. RMSNorm (rms_norm_fp32.comp) ──")
    w_t = sd["blocks.0.self_attn.q_norm.weight"]
    w_bf16 = w_t.to(torch.bfloat16)
    w_u16 = w_bf16.view(torch.uint16).numpy()
    w_packed = np.zeros(HD // 2, dtype=np.uint32)
    for i in range(HD // 2):
        lo, hi = int(w_u16[2*i]), int(w_u16[2*i+1])
        w_packed[i] = np.uint32(lo | (hi << 16))

    x = rand_input(MS*NH, HD, device="cpu", dtype=torch.float32) * 2.0
    C_shader = shader_rmsnorm(x.numpy().reshape(MS*NH, HD), w_packed, MS*NH, HD).ravel()
    C_pt = pt_rmsnorm(x, MS*NH, HD, w_t.to(torch.float32))
    results.append(verify("RMSNorm 8192×128", C_shader, C_pt, "RMSNorm 8192"))

    # ── 4. GELU ──
    print("\n── 4. GELU (gelu_fp32.comp) ──")
    x = rand_input(MS * MLP_HIDDEN // 16, device="cpu", dtype=torch.float32) * 3.0  # smaller test
    C_shader = shader_gelu(x.numpy().ravel(), len(x.numpy().ravel()))
    C_pt = F.gelu(x, approximate='tanh')  # match shader's tanh approximation
    results.append(verify("GELU", C_shader, C_pt, "GELU"))

    # ── 5. SiLU ──
    print("\n── 5. SiLU (silu_fp32.comp) ──")
    x = rand_input(M * D, device="cpu", dtype=torch.float32) * 2.0
    C_shader = shader_silu(x.numpy().ravel(), M * D)
    C_pt = pt_silu(x)
    results.append(verify("SiLU", C_shader, C_pt, "SiLU"))

    # ── 6. ScaleShift ──
    print("\n── 6. ScaleShift (scale_shift_fp32.comp) ──")
    x = rand_input(MS * D, device="cpu", dtype=torch.float32) * 2.0
    scl = torch.randn(MS * D, dtype=torch.float32) * 0.1
    sft = torch.randn(MS * D, dtype=torch.float32) * 0.1
    C_shader = shader_scale_shift(x.numpy().ravel(), scl.numpy().ravel(), sft.numpy().ravel(), MS*D, 1, 1)
    C_pt = x * scl + sft  # PyTorch: element-wise fma
    results.append(verify("ScaleShift per-element", C_shader, C_pt))

    # ── 7. Gate ──
    print("\n── 7. Gate (gate_fp32.comp) ──")
    oproj = rand_input(MS*D, device="cpu", dtype=torch.float32) * 0.1
    gate = torch.randn(MS*D, dtype=torch.float32) * 0.1
    residual = rand_input(MS*D, device="cpu", dtype=torch.float32) * 0.1
    C_shader = shader_gate(oproj.numpy().ravel(), gate.numpy().ravel(), residual.numpy().ravel(), MS*D)
    C_pt = residual + gate * oproj
    results.append(verify("Gate residual", C_shader, C_pt))

    # ── 8. Broadcast ──
    print("\n── 8. Broadcast (broadcast_fp32.comp) ──")
    x = rand_input(M, D, device="cpu", dtype=torch.float32)
    C_shader = shader_broadcast(x.numpy().ravel(), M, D, S)
    C_pt = x.repeat_interleave(S, dim=0)
    results.append(verify("Broadcast 2→512", C_shader, C_pt))

    # ── 9. RoPE ──
    print("\n── 9. RoPE (rope_fp32.comp) ──")
    t_in = rand_input(MS*NH, HD, device="cpu", dtype=torch.float32)
    # Generate C++-style freqs
    half = HD // 2
    freqs = np.zeros((MS*NH, half, 4), dtype=np.float32)
    for idx in range(MS*NH):
        for i in range(half):
            freqs[idx, i, 0] = np.cos(float(idx + i) * 0.1)
            freqs[idx, i, 1] = -np.sin(float(idx + i) * 0.1)
            freqs[idx, i, 2] = np.sin(float(idx + i) * 0.1)
            freqs[idx, i, 3] = np.cos(float(idx + i) * 0.1)
    C_shader = shader_rope(t_in.numpy().reshape(MS*NH, HD), freqs.ravel(), MS*NH, HD)
    C_pt = pt_rope_rotate(t_in.reshape(MS*NH, HD), torch.from_numpy(freqs), MS*NH, HD)
    results.append(verify("RoPE rotate", C_shader, C_pt))

    # ── 10. Attention QK^T ──
    print("\n── 10. Attention QK^T (attn_qkt_fp32.comp) ──")
    Q = rand_input(S, NH, HD, device="cpu", dtype=torch.float32).reshape(S*NH, HD)
    K = rand_input(S, NH, HD, device="cpu", dtype=torch.float32).reshape(S*NH, HD)
    sc = 1.0 / np.sqrt(HD)
    C_shader = shader_attn_qkt(Q.numpy(), K.numpy(), S, S, NH, HD, sc)
    # PyTorch: bmm
    Q_pt = Q.reshape(S, NH, HD).permute(1, 0, 2)  # [NH, S, HD]
    K_pt = K.reshape(S, NH, HD).permute(1, 0, 2)  # [NH, S, HD]
    C_pt = (torch.bmm(Q_pt, K_pt.transpose(1, 2)) * sc).permute(1, 0, 2).reshape(S*NH, S)
    results.append(verify("QK^T 256×256", C_shader, C_pt, "QK^T"))

    # ── 11. Softmax ──
    print("\n── 11. Softmax (attn_softmax_fp32.comp) ──")
    A = rand_input(S*NH, S, device="cpu", dtype=torch.float32) * 0.1
    A_np = A.numpy().copy()
    A_shader = shader_attn_softmax(A_np, S, NH, S)  # [S*NH, S]
    # PyTorch: don't permute — compare directly with same layout
    A_pt = F.softmax(torch.from_numpy(A_np).reshape(S, NH, S).reshape(-1, S), dim=-1).numpy()
    results.append(verify("Softmax 256×256", A_shader.ravel(), A_pt.ravel(), "Softmax"))

    # ── 12. Attention AV ──
    print("\n── 12. Attention AV (attn_out_fp32.comp) ──")
    A_w = torch.softmax(rand_input(S, NH, S, device="cpu", dtype=torch.float32), dim=-1)
    V = rand_input(S, NH, HD, device="cpu", dtype=torch.float32)
    A_w_flat = A_w.permute(1, 0, 2).reshape(S*NH, S).numpy()  # [S*NH, S]
    V_flat = V.permute(1, 0, 2).reshape(S*NH, HD).numpy()  # [S*NH, HD]
    C_shader = shader_attn_out(A_w_flat, V_flat, S, S, NH, HD)
    C_pt = (A_w @ V).permute(1, 0, 2).reshape(-1)
    results.append(verify("AV 256×256", C_shader.ravel(), C_pt, "AV"))

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SHADER VERIFICATION SUMMARY")
    print("=" * 70)
    all_pass = True
    for r in results:
        status = "✓" if r['max_err'] < 1e-5 else "⚠️"
        print(f"  {status} {r['label']:<45s} max_err={r['max_err']:.2e}")
        if r['max_err'] >= 1e-5:
            all_pass = False
    print(f"\n  Overall: {'✓ ALL SHADERS VERIFIED' if all_pass else '⚠️ SOME SHADERS NEED FIXING'}")

if __name__ == "__main__":
    main()
