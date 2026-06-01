"""Dump PyTorch block 0 intermediate values for Vulkan comparison.
BF16 weights loaded, fp32 compute, same inputs as phone pipeline step 1.
"""
import sys, struct, json, os
import torch, torch.nn.functional as F
import numpy as np

sys.path.insert(0, "/mnt/d/AI/anima_phone/hybridops/src")
import predict2

DTYPE = torch.float32  # fp32 compute, same as Vulkan v2
DEV = "cpu"
M, D, S, Nctx, CtxD = 2, 2048, 256, 512, 1024
MS = M * S
HEAD_DIM = 128
N_HEADS = 16
ADALN_LORA_DIM = 256

# ── Safetensors reader ──
class SafetensorsReader:
    def __init__(self, path):
        self.path = path
        self._f = open(path, 'rb')
        header_bytes = self._f.read(8)
        header_len = struct.unpack('<Q', header_bytes)[0]
        header_json = self._f.read(header_len).decode('utf-8')
        self.header = json.loads(header_json)
        self.data_start = 8 + header_len

    def get_tensor(self, key):
        info = self.header[key]
        off = self.data_start + info['data_offsets'][0]
        end = self.data_start + info['data_offsets'][1]
        size = end - off
        self._f.seek(off)
        buf = self._f.read(size)
        d = info['dtype']
        if d == 'BF16':
            arr = np.frombuffer(buf, dtype=np.uint16).reshape(info['shape']).copy()
            return torch.from_numpy(arr).view(torch.bfloat16).to(torch.float32)
        elif d == 'F16':
            arr = np.frombuffer(buf, dtype=np.uint16).reshape(info['shape']).copy()
            return torch.from_numpy(arr).view(torch.float16).to(torch.float32)
        elif d == 'F32':
            arr = np.frombuffer(buf, dtype=np.float32).reshape(info['shape']).copy()
            return torch.from_numpy(arr).to(torch.float32)
        return None

    def keys(self): return list(self.header.keys())
    def close(self):
        if self._f: self._f.close(); self._f = None

# ── Load weights ──
# WSL2 paths
MODEL_PATH = "/mnt/d/AI/手坤的anima/models/diffusion_model/anima-base-v1.0.safetensors"
CTX_DIR = "/mnt/d/AI/anima_phone/hybridops/models"
OUT_DIR = "/mnt/d/AI/anima_phone/output/block0_ref"
path = MODEL_PATH
if not os.path.exists(path):
    print(f"Cannot find safetensors at {path}")
    sys.exit(1)

print(f"Loading: {path}")
st = SafetensorsReader(path)

# Detect prefix
all_keys = st.keys()
tensor_keys = [k for k in all_keys if k != "__metadata__"]
PREFIX = ""
if tensor_keys and '.' in tensor_keys[0]:
    first_part = tensor_keys[0].split('.')[0]
    if first_part not in ('blocks', 'x_embedder', 't_embedder', 'final_layer',
                          't_embedding_norm', 'pos_embedder', 'llm_adapter'):
        PREFIX = first_part + '.'
        print(f"  Detected prefix: '{PREFIX}'")

def wk(name):
    """Get weight tensor by clean key (without prefix)."""
    full_key = PREFIX + name if PREFIX else name
    if full_key not in st.header:
        # Try without prefix
        if name in st.header:
            full_key = name
        else:
            print(f"Key not found: {full_key} (also tried: {name})")
            print(f"Available keys sample: {all_keys[:5]}")
            return None
    return st.get_tensor(full_key)

# ── Create model ──
config = dict(max_img_h=240, max_img_w=240, max_frames=128, in_channels=16,
    out_channels=16, patch_spatial=2, patch_temporal=1, concat_padding_mask=True,
    model_channels=2048, num_blocks=28, num_heads=16, mlp_ratio=4.0,
    crossattn_emb_channels=1024, pos_emb_cls="rope3d", pos_emb_learnable=True,
    pos_emb_interpolation="crop", min_fps=1, max_fps=30,
    use_adaln_lora=True, adaln_lora_dim=256,
    rope_h_extrapolation_ratio=4.0, rope_w_extrapolation_ratio=4.0,
    rope_t_extrapolation_ratio=1.0, extra_per_block_abs_pos_emb=False,
    rope_enable_fps_modulation=False)

class RefOps:
    Linear = torch.nn.Linear
    LayerNorm = torch.nn.LayerNorm
    RMSNorm = torch.nn.RMSNorm
    Embedding = torch.nn.Embedding
    GELU = torch.nn.GELU

dit = predict2.MiniTrainDIT(**config, device=DEV, dtype=DTYPE, operations=RefOps)

# Load state dict
sd = {}
for k in st.keys():
    if k == "__metadata__": continue
    clean = k[len(PREFIX):] if PREFIX and k.startswith(PREFIX) else k
    t = st.get_tensor(k)
    if t is not None: sd[clean] = t
dit.load_state_dict(sd, strict=False)
dit.eval()
st.close()
print("Model loaded.")

# ── Generate inputs (matching phone pipeline step 1, seed=6666) ──
gen = torch.Generator(device=DEV).manual_seed(6666)
x = torch.randn(1, 16, 32, 32, generator=gen, dtype=DTYPE)
sigma = 1.0  # first sigma from scheduler
ts = torch.tensor([sigma], dtype=DTYPE)

# Load contexts
ctx_cond = torch.load(os.path.join(CTX_DIR, "context_cond.pt"),
                       weights_only=True, map_location=DEV).to(DTYPE)
ctx_uncond = torch.load(os.path.join(CTX_DIR, "context_uncond.pt"),
                         weights_only=True, map_location=DEV).to(DTYPE)

# CFG batch
x_b = x.unsqueeze(2).repeat(2, 1, 1, 1, 1)  # [2,16,1,32,32]
ctx_b = torch.cat([ctx_cond, ctx_uncond], dim=0)  # [2,512,1024]
ts_b = ts.repeat(2)  # [2]

# x_embedder: exactly what C++ head_x_embed does
padding_mask = torch.zeros(2, 1, 1, 32, 32, dtype=DTYPE)
x_pad = torch.cat([x_b, padding_mask], dim=1)  # [2,17,1,32,32]
from einops import rearrange
x_patches = rearrange(x_pad, "b c (t r) (h m) (w n) -> b t h w (c r m n)", r=1, m=2, n=2)
x_flat = x_patches.reshape(MS, 68)  # [512, 68]
w_x_proj = wk("x_embedder.proj.1.weight")
x_emb = F.linear(x_flat, w_x_proj)  # [512, 2048]
print(f"x_emb: [{x_emb.min():.4f}, {x_emb.max():.4f}]")

# t_embedder: exactly what C++ head_t_embed does
half = D // 2
freq = 1.0 / (10000.0 ** (torch.arange(half, dtype=torch.float32) / (half - 1)))
emb = ts_b[:, None].float() * freq[None, :]
emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)  # [2, 2048]
t_emb_raw = emb

w_t1 = wk("t_embedder.1.linear_1.weight")
w_t2 = wk("t_embedder.1.linear_2.weight")
h = F.linear(t_emb_raw, w_t1)
h = F.silu(h)
adaln_lora = F.linear(h, w_t2)  # [2, 6144]

w_tn = wk("t_embedding_norm.weight")
t_emb_norm = t_emb_raw * w_tn * torch.rsqrt(t_emb_raw.pow(2).mean(-1, keepdim=True) + 1e-6)
print(f"t_emb: [{t_emb_norm.min():.4f}, {t_emb_norm.max():.4f}]")
print(f"adaln_lora: [{adaln_lora.min():.4f}, {adaln_lora.max():.4f}]")

# ctx: [2, 512, 1024] → [1024, 2048] for K/V projection
ctx_flat = ctx_b.reshape(M * Nctx, CtxD)
print(f"ctx: [{ctx_flat.min():.4f}, {ctx_flat.max():.4f}]")

# ── Block 0 manual forward ──
b = 0
out_dir = OUT_DIR
os.makedirs(out_dir, exist_ok=True)

def save(name, t):
    np.save(os.path.join(out_dir, f"{name}.npy"), t.detach().float().numpy())
    print(f"  {name}: shape={list(t.shape)} range=[{t.min():.4f},{t.max():.4f}]")

save("00_input", x_emb)
save("01_t_emb", t_emb_norm)
save("02_adaln_lora", adaln_lora)

# ── AdaLN self ──
e = t_emb_norm  # [M, D]
w1_s = wk(f"blocks.{b}.adaln_modulation_self_attn.1.weight")
w2_s = wk(f"blocks.{b}.adaln_modulation_self_attn.2.weight")
h1 = F.silu(e)                                                        # [2,2048]
h2 = F.linear(h1, w1_s)                                               # [2,256]
h3 = F.linear(h2, w2_s)                                               # [2,6144] (3*D)
h3 = h3 + adaln_lora                                                  # +lora
shift_s, scale_s, gate_s = h3.chunk(3, dim=-1)                       # each [2,2048]
scale_s1 = scale_s + 1.0

save("03_sa_shift_pre_bcast", shift_s)
save("04_sa_scale_pre_bcast", scale_s)
save("05_sa_gate_pre_bcast", gate_s)

# Broadcast [M,D] → [MS,D]: repeat S times
def broadcast(v):
    return v.unsqueeze(1).repeat(1, S, 1).reshape(MS, D)

shift_s_b = broadcast(shift_s)
scale_s1_b = broadcast(scale_s1)
gate_s_b = broadcast(gate_s)

save("06_sa_shift", shift_s_b)
save("07_sa_scale", scale_s1_b)
save("08_sa_gate", gate_s_b)

# ── Self-Attention ──
w_q = wk(f"blocks.{b}.self_attn.q_proj.weight")
w_k = wk(f"blocks.{b}.self_attn.k_proj.weight")
w_v = wk(f"blocks.{b}.self_attn.v_proj.weight")
w_o = wk(f"blocks.{b}.self_attn.output_proj.weight")
w_qn = wk(f"blocks.{b}.self_attn.q_norm.weight")
w_kn = wk(f"blocks.{b}.self_attn.k_norm.weight")

x_flat_2d = x_emb  # [MS, D]
x_ln = F.layer_norm(x_flat_2d, (D,), None, None, 1e-6)
x_mod = x_ln * scale_s1_b + shift_s_b
save("09_sa_ln_out", x_ln)
save("10_sa_modulated", x_mod)

q = F.linear(x_mod, w_q)  # [MS, 2048]
k = F.linear(x_mod, w_k)
v = F.linear(x_mod, w_v)
save("11_sa_q", q)
save("12_sa_k", k)
save("13_sa_v", v)

# Reshape Q/K/V: [MS, H*D] → [MS*H, head_dim]
q_r = q.reshape(MS, N_HEADS, HEAD_DIM).reshape(MS * N_HEADS, HEAD_DIM)
k_r = k.reshape(MS, N_HEADS, HEAD_DIM).reshape(MS * N_HEADS, HEAD_DIM)
v_r = v.reshape(MS, N_HEADS, HEAD_DIM).reshape(MS * N_HEADS, HEAD_DIM)

# RMSNorm Q/K
q_n = q_r * w_qn.repeat(MS, 1).reshape(MS * N_HEADS, HEAD_DIM) * \
      torch.rsqrt(q_r.pow(2).mean(-1, keepdim=True) + 1e-6)
k_n = k_r * w_kn.repeat(MS, 1).reshape(MS * N_HEADS, HEAD_DIM) * \
      torch.rsqrt(k_r.pow(2).mean(-1, keepdim=True) + 1e-6)
save("14_sa_q_norm", q_n)
save("15_sa_k_norm", k_n)

# RoPE: simplified (full RoPE uses 3D position embeddings)
# Compute RoPE freqs same way Vulkan does
def compute_rope_freqs_pytorch():
    head_dim = HEAD_DIM
    dim_h = head_dim // 6 * 2
    dim_w = dim_h
    dim_t = head_dim - 2 * dim_h
    half_dim = head_dim // 2
    H_patches = 16
    W_patches = 16

    h_ntk = 4.0 ** (dim_h / (dim_h - 2))
    w_ntk = 4.0 ** (dim_w / (dim_w - 2))
    t_ntk = 1.0 ** (dim_t / (dim_t - 2))
    h_theta = 10000.0 * h_ntk
    w_theta = 10000.0 * w_ntk
    t_theta = 10000.0 * t_ntk

    # Per-position [256, half_dim, 4]
    pos_freqs = torch.zeros(S, half_dim, 4, dtype=torch.float32)
    for p in range(S):
        h_idx = p // W_patches
        w_idx = p % W_patches
        for j in range(half_dim):
            if j < dim_t // 2:
                freq = 1.0 / (t_theta ** (2 * j / dim_t))
                angle = 0.0
            elif j < dim_t // 2 + dim_h // 2:
                jh = j - dim_t // 2
                freq = 1.0 / (h_theta ** (2 * jh / dim_h))
                angle = h_idx * freq
            elif j < dim_t // 2 + dim_h // 2 + dim_w // 2:
                jw = j - dim_t // 2 - dim_h // 2
                freq = 1.0 / (w_theta ** (2 * jw / dim_w))
                angle = w_idx * freq
            else:
                jt = j - dim_t // 2 - dim_h // 2 - dim_w // 2
                freq = 1.0 / (t_theta ** (2 * jt / dim_t))
                angle = 0.0
            pos_freqs[p, j, 0] = torch.cos(torch.tensor(angle))
            pos_freqs[p, j, 1] = -torch.sin(torch.tensor(angle))
            pos_freqs[p, j, 2] = torch.sin(torch.tensor(angle))
            pos_freqs[p, j, 3] = torch.cos(torch.tensor(angle))

    # Replicate: [M * S * H, half_dim, 4]
    all_freqs = torch.zeros(M * S * N_HEADS, half_dim, 4, dtype=torch.float32)
    for mb in range(M):
        for p in range(S):
            for h in range(N_HEADS):
                dst = mb * S * N_HEADS + p * N_HEADS + h
                all_freqs[dst] = pos_freqs[p]
    return all_freqs

rope_freqs = compute_rope_freqs_pytorch()

def apply_rope(t, freqs):
    """t: [N, head_dim], freqs: [N, half_dim, 4]"""
    N_rows, hd = t.shape
    half = hd // 2
    out = torch.zeros_like(t)
    for i in range(half):
        a = t[:, 2*i]
        b = t[:, 2*i+1]
        c  = freqs[:, i, 0]  # cos
        ms = freqs[:, i, 1]  # -sin
        s  = freqs[:, i, 2]  # sin
        mc = freqs[:, i, 3]  # cos
        out[:, 2*i]   = c * a + ms * b
        out[:, 2*i+1] = s * a + mc * b
    return out

q_rope = apply_rope(q_n, rope_freqs)
k_rope = apply_rope(k_n, rope_freqs)
save("16_sa_q_rope", q_rope)
save("17_sa_k_rope", k_rope)

# Attention: QK^T → softmax → AV
attn_scale = 1.0 / np.sqrt(HEAD_DIM)
# Reshape to [M, H, S, D] for per-batch attention
q_bh = q_rope.reshape(M, S, N_HEADS, HEAD_DIM).permute(0, 2, 1, 3)  # [2, 16, 256, 128]
k_bh = k_rope.reshape(M, S, N_HEADS, HEAD_DIM).permute(0, 2, 1, 3)
v_bh = v_r.reshape(M, S, N_HEADS, HEAD_DIM).permute(0, 2, 1, 3)

attn_out = torch.zeros(M * S, N_HEADS * HEAD_DIM, dtype=torch.float32)
for mb in range(M):
    scores = torch.matmul(q_bh[mb], k_bh[mb].transpose(-2, -1)) * attn_scale  # [16,256,256]
    attn_w = F.softmax(scores, dim=-1)
    attn_o = torch.matmul(attn_w, v_bh[mb])  # [16,256,128]
    attn_out[mb*S:(mb+1)*S] = attn_o.permute(1, 0, 2).reshape(S, N_HEADS*HEAD_DIM)

save("18_sa_attn_out", attn_out)

o_proj = F.linear(attn_out, w_o)  # [MS, D]
save("19_sa_o_proj", o_proj)

x_sa = x_flat_2d + gate_s_b * o_proj
save("20_sa_residual", x_sa)

# ── Cross-Attention ──
w_ca_q = wk(f"blocks.{b}.cross_attn.q_proj.weight")
w_ca_k = wk(f"blocks.{b}.cross_attn.k_proj.weight")
w_ca_v = wk(f"blocks.{b}.cross_attn.v_proj.weight")
w_ca_o = wk(f"blocks.{b}.cross_attn.output_proj.weight")
w_ca_qn = wk(f"blocks.{b}.cross_attn.q_norm.weight")
w_ca_kn = wk(f"blocks.{b}.cross_attn.k_norm.weight")

# AdaLN cross (already computed in seg_adaln)
w1_c = wk(f"blocks.{b}.adaln_modulation_cross_attn.1.weight")
w2_c = wk(f"blocks.{b}.adaln_modulation_cross_attn.2.weight")
h1c = F.silu(e)
h2c = F.linear(h1c, w1_c)
h3c = F.linear(h2c, w2_c) + adaln_lora
shift_c, scale_c, gate_c = h3c.chunk(3, dim=-1)
scale_c1 = scale_c + 1.0
shift_c_b = broadcast(shift_c)
scale_c1_b = broadcast(scale_c1)
gate_c_b = broadcast(gate_c)

x_ca_ln = F.layer_norm(x_sa, (D,), None, None, 1e-6)
x_ca_mod = x_ca_ln * scale_c1_b + shift_c_b
save("21_ca_ln_out", x_ca_ln)
save("22_ca_modulated", x_ca_mod)

q_ca = F.linear(x_ca_mod, w_ca_q)
k_ca = F.linear(ctx_flat, w_ca_k)   # from ctx, [1024, 2048]
v_ca = F.linear(ctx_flat, w_ca_v)
save("23_ca_q", q_ca)
save("24_ca_k", k_ca)
save("25_ca_v", v_ca)

q_ca_r = q_ca.reshape(MS, N_HEADS, HEAD_DIM).reshape(MS * N_HEADS, HEAD_DIM)
k_ca_r = k_ca.reshape(M * Nctx, N_HEADS, HEAD_DIM).reshape(M * Nctx * N_HEADS, HEAD_DIM)
v_ca_r = v_ca.reshape(M * Nctx, N_HEADS, HEAD_DIM).reshape(M * Nctx * N_HEADS, HEAD_DIM)

q_ca_n = q_ca_r * w_ca_qn.repeat(MS, 1).reshape(MS * N_HEADS, HEAD_DIM) * \
         torch.rsqrt(q_ca_r.pow(2).mean(-1, keepdim=True) + 1e-6)
k_ca_n = k_ca_r * w_ca_kn.repeat(M * Nctx, 1).reshape(M * Nctx * N_HEADS, HEAD_DIM) * \
         torch.rsqrt(k_ca_r.pow(2).mean(-1, keepdim=True) + 1e-6)
save("26_ca_q_norm", q_ca_n)
save("27_ca_k_norm", k_ca_n)

# Cross-attention (no RoPE for cross Q/K from ctx in this implementation)
# Actually, looking at the C++ code, cross-attn DOES use RoPE for Q but not K from ctx
# Wait, let me check... In C++, seg_cross_pre does rmsnorm but NOT rope for cross-attn.
# So cross-attn Q/K are RMSNorm'd but not RoPE'd. The attention uses the raw normalized Q/K.
q_ca_bh = q_ca_n.reshape(M, S, N_HEADS, HEAD_DIM).permute(0, 2, 1, 3)
k_ca_bh = k_ca_n.reshape(M, Nctx, N_HEADS, HEAD_DIM).permute(0, 2, 1, 3)

attn_ca_out = torch.zeros(M * S, N_HEADS * HEAD_DIM, dtype=torch.float32)
for mb in range(M):
    scores = torch.matmul(q_ca_bh[mb], k_ca_bh[mb].transpose(-2, -1)) * attn_scale
    attn_w = F.softmax(scores, dim=-1)
    v_ca_bh = v_ca_r.reshape(M, Nctx, N_HEADS, HEAD_DIM).permute(0, 2, 1, 3)[mb]
    attn_o = torch.matmul(attn_w, v_ca_bh)
    attn_ca_out[mb*S:(mb+1)*S] = attn_o.permute(1, 0, 2).reshape(S, N_HEADS*HEAD_DIM)

save("28_ca_attn_out", attn_ca_out)

o_ca_proj = F.linear(attn_ca_out, w_ca_o)
save("29_ca_o_proj", o_ca_proj)

x_ca = x_sa + gate_c_b * o_ca_proj
save("30_ca_residual", x_ca)

# ── MLP ──
w_m1 = wk(f"blocks.{b}.mlp.layer1.weight")
w_m2 = wk(f"blocks.{b}.mlp.layer2.weight")

# AdaLN MLP
w1_m = wk(f"blocks.{b}.adaln_modulation_mlp.1.weight")
w2_m = wk(f"blocks.{b}.adaln_modulation_mlp.2.weight")
h1m = F.silu(e)
h2m = F.linear(h1m, w1_m)
h3m = F.linear(h2m, w2_m) + adaln_lora
shift_m, scale_m, gate_m = h3m.chunk(3, dim=-1)
scale_m1 = scale_m + 1.0
shift_m_b = broadcast(shift_m)
scale_m1_b = broadcast(scale_m1)
gate_m_b = broadcast(gate_m)

x_mlp_ln = F.layer_norm(x_ca, (D,), None, None, 1e-6)
x_mlp_mod = x_mlp_ln * scale_m1_b + shift_m_b
save("31_mlp_ln_out", x_mlp_ln)
save("32_mlp_modulated", x_mlp_mod)

fc1 = F.linear(x_mlp_mod, w_m1)
fc1_gelu = F.gelu(fc1)
save("33_mlp_fc1", fc1)
save("34_mlp_gelu", fc1_gelu)

fc2 = F.linear(fc1_gelu, w_m2)
save("35_mlp_fc2", fc2)

x_out = x_ca + gate_m_b * fc2
save("36_block0_out", x_out)

print(f"\nDumped 37 intermediates to {out_dir}")
print(f"Block 0 output range: [{x_out.min():.4f}, {x_out.max():.4f}]")
print("\nRun on phone with C++ capture to compare.")
