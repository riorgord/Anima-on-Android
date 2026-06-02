"""Simulate C++ cpu_gemm_bf16 in Python and compare against PyTorch F.linear."""
import numpy as np, torch, torch.nn.functional as F, safetensors.torch, struct, os

M, D = 2, 2048
AL_DIM = 256  # adaln_lora_dim
SF = '/mnt/d/AI/手坤的anima/models/diffusion_model/anima-base-v1.0.safetensors'

# ── Load t_emb from phone capture to match C++ input exactly ──
capture_dir = '/mnt/d/AI/anima_phone/output/cmp_v2'
t_emb_cap = torch.from_numpy(np.load(f'{capture_dir}/b0_temb.npy').reshape(M, D))

# ── Load weights ──
st = safetensors.torch.load_file(SF, device='cpu')
sd = {k[4:] if k.startswith('net.') else k: v.to(torch.float32) for k, v in st.items()}
del st

# ── C++ SiLU equivalent ──
silu_cpu = t_emb_cap.numpy() / (1.0 + np.exp(-t_emb_cap.numpy()))
silu_pt  = F.silu(t_emb_cap)
print(f"SiLU: CPU vs PT max_err = {np.abs(silu_cpu - silu_pt.numpy()).max():.10f}")

# ── Helper: read raw BF16 bytes from safetensors ──
# We need to re-open the safetensors to get raw bytes (not torch-converted)
import safetensors
with safetensors.safe_open(SF, framework='pt', device='cpu') as f:
    pass  # just to check format

# Actually, let's use safetensors metadata to get byte offsets and read raw
def get_raw_bf16(sf_path, key):
    """Read raw BF16 weight bytes from safetensors, return as numpy uint16 array."""
    import json, struct
    with open(sf_path, 'rb') as f:
        # Read header size (8 bytes: uint64)
        header_len_bytes = f.read(8)
        header_len = struct.unpack('<Q', header_len_bytes)[0]
        # Read header JSON
        header_json = f.read(header_len)
        header = json.loads(header_json)
        # Data starts after 8 + header_len
        data_start = 8 + header_len
        # Find the key (with or without prefix)
        if key in header:
            info = header[key]
        elif 'net.' + key in header:
            info = header['net.' + key]
        else:
            raise KeyError(f"Key not found: {key}")
        offset = info['data_offsets'][0]
        length = info['data_offsets'][1] - offset
        dtype = info['dtype']
        assert dtype == 'BF16', f"Expected BF16, got {dtype}"
        f.seek(data_start + offset)
        raw = f.read(length)
        arr = np.frombuffer(raw, dtype=np.uint16).copy()
        return arr

# ── Simulate C++ cpu_gemm_bf16 EXACTLY ──
def bf16_to_f32(bf16):
    """Exact C++ implementation: shift uint16 left by 16, reinterpret as float."""
    bits = int(bf16) << 16
    return struct.unpack('f', struct.pack('I', bits))[0]

def cpu_gemm_bf16_sim(A_np, B_bf16_uint16, Mv, Nv, Kv):
    """Exact simulation of C++ cpu_gemm_bf16.
    A: [Mv, Kv] float32
    B: uint16 array indexed as B[n * Kv + k]
    Returns C: [Mv, Nv] float32
    """
    C = np.zeros((Mv, Nv), dtype=np.float32)
    for m in range(Mv):
        for n in range(Nv):
            s = 0.0
            for k in range(Kv):
                w = bf16_to_f32(B_bf16_uint16[n * Kv + k])
                s += A_np[m, k] * w
            C[m, n] = s
    return C

# ── Test on adaln w0 (LoRA down: [256, 2048]) ──
w0_keys = [
    'blocks.0.adaln_modulation_self_attn.1.weight',
]

for wkey in w0_keys:
    print(f"\n=== Testing {wkey} ===")

    # Get raw BF16 weight
    raw_bf16 = get_raw_bf16(SF, wkey)
    print(f"  Raw BF16: {len(raw_bf16)} uint16 values")
    print(f"  First 5 raw: {raw_bf16[:5]}")
    print(f"  First 5 as f32: {[bf16_to_f32(raw_bf16[i]) for i in range(5)]}")

    # Get PyTorch weight for comparison
    pt_w = sd[wkey]  # already float32
    print(f"  PT weight shape: {pt_w.shape}")
    print(f"  PT weight[0,0:5]: {pt_w[0,:5].numpy()}")

    # Convert first 5 from PT back to BF16 and check
    # PT float32 → BF16: round to nearest, truncate mantissa
    pt_w_bf16 = pt_w.to(torch.bfloat16)
    print(f"  PT as bf16[0,0:5]: {[float(pt_w_bf16[0,i]) for i in range(5)]}")

    # Check if raw BF16 matches PT bf16
    # Get the uint16 representation of PT's bf16
    pt_bf16_uint16 = pt_w_bf16.view(torch.uint16).numpy().flatten()
    # Compare first few
    for i in range(5):
        raw_val = bf16_to_f32(raw_bf16[i])
        pt_val = float(pt_w_bf16[0, i])
        raw_uint = int(raw_bf16[i])
        pt_uint = int(pt_bf16_uint16[i])
        match = "✓" if raw_uint == pt_uint else "✗ DIFFER!"
        print(f"    W[0,{i}]: raw=0x{raw_uint:04X} ({raw_val:.6f})  pt=0x{pt_uint:04X} ({pt_val:.6f}) {match}")

    # Compare the FULL weight
    # raw_bf16 is stored row-major [256, 2048]
    raw_w = raw_bf16.reshape(256, 2048)
    pt_bf16_np = pt_bf16_uint16.reshape(256, 2048)
    n_diff = np.sum(raw_w != pt_bf16_np)
    print(f"  Total elements: {256*2048}, different: {n_diff} ({100*n_diff/(256*2048):.4f}%)")

    if n_diff > 0:
        print(f"  FIRST MISMATCH at [{np.where(raw_w != pt_bf16_np)[0][0]}, {np.where(raw_w != pt_bf16_np)[1][0]}]")
        idx = np.where((raw_w != pt_bf16_np).any(axis=1))[0][0]
        for j in range(min(5, 2048)):
            if raw_w[idx, j] != pt_bf16_np[idx, j]:
                raw_val = bf16_to_f32(raw_w[idx, j])
                pt_val = float(pt_w_bf16[idx, j])
                print(f"    W[{idx},{j}]: raw=0x{raw_w[idx,j]:04X} ({raw_val:.6f})  pt=0x{pt_bf16_np[idx,j]:04X} ({pt_val:.6f})")

    # Compute GEMM both ways
    # C++ AdaLN: SiLU(emb) → GEMM(SiLU_out, W0) → t1
    Mv, Nv, Kv = M, AL_DIM, D
    t1_cpu_sim = cpu_gemm_bf16_sim(silu_cpu, raw_bf16, Mv, Nv, Kv)
    # PT: F.silu first, THEN F.linear
    t1_pt = F.linear(F.silu(t_emb_cap), pt_w[:Nv, :Kv])

    print(f"\n  --- GEMM output comparison (C++ vs PT, both SiLU first) ---")
    print(f"  CPU sim t1[0,0:4] = {t1_cpu_sim[0,:4]}")
    print(f"  PT     t1[0,0:4] = {t1_pt[0,:4].numpy()}")
    max_err = np.abs(t1_cpu_sim - t1_pt.numpy()).max()
    print(f"  t1 max_err = {max_err:.6f}")

    # Also test: raw BF16 weight loaded as float32, used in PT GEMM
    raw_bf16_f32 = np.array([bf16_to_f32(raw_bf16[i]) for i in range(len(raw_bf16))], dtype=np.float32).reshape(Nv, Kv)
    t1_raw = F.linear(F.silu(t_emb_cap), torch.from_numpy(raw_bf16_f32))
    max_err_raw = (t1_raw - t1_pt).abs().max().item()
    print(f"  PT(w_from_raw_bf16) vs PT(original_w): max_err={max_err_raw:.10f}")

    # Also compare: cpu_gemm_bf16_sim vs PT with raw bf16 weights
    max_err_cpu_vs_raw = np.abs(t1_cpu_sim - t1_raw.numpy()).max()
    print(f"  CPU sim vs PT(raw_w): max_err={max_err_cpu_vs_raw:.6f}")

    # Test phone debug values vs our simulation
    phone_t1 = np.array([1.484372, -1.694637, -1.690686, -0.306566], dtype=np.float32)
    sim_err = np.abs(t1_cpu_sim[0,:4] - phone_t1).max()
    print(f"  CPU sim vs Phone C++ t1: max_err={sim_err:.6f}")

    if max_err < 0.001 and max_err_raw < 0.001 and max_err_cpu_vs_raw < 0.001:
        print(f"\n  ✓ C++ GEMM matches PT perfectly (max_err < 0.001)!")
    else:
        print(f"\n  ⚠️ DISCREPANCY! Need further investigation.")
