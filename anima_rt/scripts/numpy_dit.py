"""numpy_dit.py — Anima DiT forward, zero torch, BF16-storage FP32-compute.
Forked from predict2.py. Like RTX 20-series: BF16 weights, FP32 math.
"""
import math, numpy as np

# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _pad_to_patch_size(x_np, pt, ps):
    B, C, T, H, W = x_np.shape
    ph = (ps - H % ps) % ps; pw = (ps - W % ps) % ps; pt2 = (pt - T % pt) % pt
    if ph == 0 and pw == 0 and pt2 == 0: return x_np
    return np.pad(x_np, ((0,0),(0,0),(0,pt2),(0,ph),(0,pw)))

def _rearrange_patchify(x_np, ps=2, pt=1):
    """predict2.PatchEmbed rearrange: b c (t r) (h m) (w n) -> b t h w (c r m n)"""
    B, C, Tr, Hm, Wn = x_np.shape
    T, H, W = Tr//pt, Hm//ps, Wn//ps
    x = x_np.reshape(B, C, T, pt, H, ps, W, ps)
    x = x.transpose(0, 2, 4, 6, 1, 3, 5, 7)
    return x.reshape(B, T, H, W, C*pt*ps*ps)

def _rearrange_unpatchify(x_np, ps=2, pt=1, out_ch=16):
    """predict2.unpatchify: b t h w (p1 p2 t C) -> b C (T t) (H p1) (W p2)"""
    B, T, H, W, M = x_np.shape
    x = x_np.reshape(B, T, H, W, ps, ps, pt, out_ch)
    # einsum: "B T H W p1 p2 t C -> B C (T t) (H p1) (W p2)"
    # transpose to [B, C, T, t, H, p1, W, p2] = dims 0,7,1,6,2,4,3,5
    x = x.transpose(0, 7, 1, 6, 2, 4, 3, 5)
    return x.reshape(B, out_ch, T*pt, H*ps, W*ps)

def _numpy_rope(t_np, freqs_np):
    """Line-by-line numpy mirror of predict2.apply_rotary_pos_emb."""
    half_D = t_np.shape[-1] // 2; t_shape = t_np.shape
    t_ = np.expand_dims(np.moveaxis(t_np.reshape(*t_shape[:-1], 2, half_D), -2, -1), -2)
    return np.moveaxis(freqs_np[...,0]*t_[...,0] + freqs_np[...,1]*t_[...,1], -1, -2).reshape(*t_shape)

def _numpy_timesteps(sigma, num_channels=2048):
    """Mirrors predict2.Timesteps.forward. sigma: scalar or [B] array."""
    half = num_channels // 2
    exponent = -math.log(10000) * np.arange(half, dtype=np.float32) / float(half)
    emb = np.exp(exponent)
    sigma_arr = np.atleast_1d(np.asarray(sigma, dtype=np.float32))
    emb = sigma_arr.reshape(-1,1) * emb.reshape(1,-1)
    return np.concatenate([np.cos(emb), np.sin(emb)], axis=-1)

def _generate_rope_freqs(H, W, T, head_dim):
    """Mirrors VideoRopePosition3DEmb.generate_embeddings."""
    dim = head_dim; dim_h = dim//6*2; dim_w = dim_h; dim_t = dim - 2*dim_h
    h_ntk = 4.0**(dim_h/(dim_h-2)); w_ntk = 4.0**(dim_w/(dim_w-2))
    hf = 1.0/(10000.0*h_ntk)**(np.arange(0,dim_h,2,dtype=np.float32)[:dim_h//2]/dim_h)
    wf = 1.0/(10000.0*w_ntk)**(np.arange(0,dim_w,2,dtype=np.float32)[:dim_w//2]/dim_w)
    tf = 1.0/10000.0**(np.arange(0,dim_t,2,dtype=np.float32)[:dim_t//2]/dim_t)
    m = lambda x: np.stack([np.cos(x), -np.sin(x), np.sin(x), np.cos(x)], axis=-1)
    hh = np.tile(m(np.outer(np.arange(H,dtype=np.float32),hf))[None,:,None,:,:], (T,1,W,1,1))
    wh = np.tile(m(np.outer(np.arange(W,dtype=np.float32),wf))[None,None,:,:,:], (T,H,1,1,1))
    th = np.tile(m(np.outer(np.arange(T,dtype=np.float32),tf))[:,None,None,:,:], (1,H,W,1,1))
    em = np.concatenate([th,hh,wh], axis=-2)
    return em.reshape(T,H,W,dim//2,2,2).reshape(T*H*W,dim//2,2,2)


# ═══════════════════════════════════════════════════════════════
# NumpyDiT
# ═══════════════════════════════════════════════════════════════

class NumpyDiT:
    def __init__(self, vk, rt):
        self._vk = vk; self._rt = rt
        self.D = 2048; self.num_blocks = 28; self.num_heads = 16
        self.head_dim = 128; self.mlp_hidden = 8192
        self.ps = 2; self.pt = 1; self.out_ch = 16

        self.x_embedder = NumpyPatchEmbed(self._vk)
        self.t_embedder = NumpyTimestepEmbedding(self.D, self._vk, self._rt)
        self.final_layer = NumpyFinalLayer(self.D, self._vk, self._rt)
        self.blocks = [NumpyBlock(i, self.D, 1024, self.num_heads, self.head_dim,
                                   self.mlp_hidden, self._vk, self._rt)
                       for i in range(self.num_blocks)]

    def forward(self, x_np, sigma, ctx_np):
        """x_np: [B,C,T,H,W], sigma: scalar or [B], ctx: [B,N,1024].
        Returns: [B,C,Tout,Hout,Wout] FP32.
        Resets Vulkan descriptor pool internally — call freely without vk management."""

        # Reset Vulkan descriptor pool (needed each step)
        import ctypes, vk_ops
        vk_ops._lib.vk_reset_pool()
        B, C, T_in, Hi, Wi = x_np.shape
        orig = [B, C, T_in, Hi, Wi]

        x_np = _pad_to_patch_size(x_np, self.pt, self.ps)
        B, C, T, H_img, W_img = x_np.shape

        # concat_padding_mask (predict2.prepare_embedded_sequence lines 798-807)
        padding = np.zeros((B, 1, T, H_img, W_img), dtype=np.float32)
        x_np = np.concatenate([x_np, padding], axis=1)  # [B, C+1, T, H, W]

        # x_embedder → [B*T, S, D] fp32
        x_bt_s_d = self.x_embedder.forward(x_np.astype(np.float32))
        H, W = H_img//self.ps, W_img//self.ps

        # reshape to [B, T, H, W, D] — residual stream in fp32
        x_bt_hw_d = x_bt_s_d.reshape(B, T, H, W, self.D).astype(np.float32)

        # rope freqs → fp32
        rope_raw = _generate_rope_freqs(H, W, T, self.head_dim).astype(np.float32)
        rope_emb = rope_raw[np.newaxis, :, np.newaxis, :, :, :]

        # t_embedder: timesteps → fp32 emb
        sigma_arr = np.atleast_1d(np.asarray(sigma, dtype=np.float32))
        B_sig = sigma_arr.shape[0]
        t_emb_np = _numpy_timesteps(sigma_arr, self.D)  # [B,2048] fp32
        t_emb, adaln_lora = self.t_embedder.forward(t_emb_np)
        t_emb = self._rt.run_rmsnorm(t_emb, 't_embedding_norm')

        ctx_f32 = ctx_np.astype(np.float32)

        # 28 blocks — all fp32
        for blk in self.blocks:
            x_bt_hw_d = blk.forward(x_bt_hw_d, t_emb, ctx_f32, rope_emb, adaln_lora)

        # final_layer — fp32
        x_out = self.final_layer.forward(x_bt_hw_d, t_emb, adaln_lora)

        # unpatchify + crop
        x_out = _rearrange_unpatchify(x_out, self.ps, self.pt, self.out_ch)
        x_out = x_out[:, :, :orig[-3], :orig[-2], :orig[-1]]
        return x_out.astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# NumpyPatchEmbed
# ═══════════════════════════════════════════════════════════════

class NumpyPatchEmbed:
    def __init__(self, vk): self._vk = vk
    def forward(self, x_np):
        x = _rearrange_patchify(x_np, ps=2, pt=1)
        B, T, H, W, K_in = x.shape
        M = B*T*H*W
        x_flat = x.reshape(M, K_in).astype(np.float32)
        out = np.zeros((M, 2048), dtype=np.float32)
        self._vk.vk_run_gemm('x_embedder.proj.1.weight', x_flat, out, M, 2048, K_in)
        return out.reshape(B*T, H*W, 2048)


# ═══════════════════════════════════════════════════════════════
# NumpyTimestepEmbedding
# ═══════════════════════════════════════════════════════════════

class NumpyTimestepEmbedding:
    def __init__(self, D, vk, rt):
        self._vk = vk; self._rt = rt; self.D = D
    def forward(self, emb_np):
        B, D = emb_np.shape; M = B
        x = emb_np.reshape(M, D).astype(np.float32)
        out1 = np.zeros((M, D), dtype=np.float32)
        self._vk.vk_run_gemm('t_embedder.1.linear_1.weight', x, out1, M, D, D)
        out1 = self._rt.run_silu(out1.reshape(-1)).reshape(M, D)
        out2 = np.zeros((M, 3*D), dtype=np.float32)
        self._vk.vk_run_gemm('t_embedder.1.linear_2.weight', out1, out2, M, 3*D, D)
        return emb_np.reshape(B, 1, D), out2.reshape(B, 1, 3*D)


# ═══════════════════════════════════════════════════════════════
# NumpyFinalLayer
# ═══════════════════════════════════════════════════════════════

class NumpyFinalLayer:
    """predict2.FinalLayer — LN→scale/shift→SiLU→Linear (n_adaln_chunks=2)."""
    def __init__(self, D, vk, rt):
        self._vk = vk; self._rt = rt; self.D = D; self.out_dim = 64

    def forward(self, x_bt_hw_d, emb, adaln_lora):
        B, T, H, W, D = x_bt_hw_d.shape; M = B*T
        # AdaLN modulation: PT Sequential(SiLU, Linear, Linear) — ONE SiLU
        emb_f = emb.reshape(M, D).astype(np.float32)
        mod = self._rt.run_silu(emb_f.reshape(-1)).reshape(M, D)
        m1 = np.zeros((M, 256), dtype=np.float32)
        self._vk.vk_run_gemm('final_layer.adaln_modulation.1.weight', mod, m1, M, 256, D)
        # No SiLU between the two Linears
        m2 = np.zeros((M, 2*D), dtype=np.float32)
        self._vk.vk_run_gemm('final_layer.adaln_modulation.2.weight', m1, m2, M, 2*D, 256)
        if adaln_lora is not None:
            m2 = m2 + adaln_lora.reshape(M, 3*D).astype(np.float32)[:, :2*D]
        m2 = m2.reshape(B, T, 2*D)
        shift, scale = np.split(m2, 2, axis=-1)

        # LN + modulate (NO SiLU here — PT FinalLayer is LN→modulate→Linear)
        x_f32 = x_bt_hw_d.reshape(B*T*H*W, D).astype(np.float32)
        x_n = self._rt.run_layernorm(x_f32, D).reshape(B, T, H, W, D)
        x_m = x_n*(1.0+scale.reshape(B,T,1,1,D)) + shift.reshape(B,T,1,1,D)

        # Linear
        x_f = x_m.reshape(B*T*H*W, D)
        out = np.zeros((B*T*H*W, self.out_dim), dtype=np.float32)
        self._vk.vk_run_gemm('final_layer.linear.weight', x_f, out, B*T*H*W, self.out_dim, D)
        return out.reshape(B, T, H, W, self.out_dim)


# ═══════════════════════════════════════════════════════════════
# NumpyBlock
# ═══════════════════════════════════════════════════════════════

class NumpyBlock:
    def __init__(self, idx, D, ctx_dim, num_heads, head_dim, mlp_hidden, vk, rt):
        self._vk = vk; self._rt = rt
        self.idx = idx; self.D = D; self.num_heads = num_heads; self.head_dim = head_dim
        self.sa = NumpyAttention(f'blocks.{idx}.self_attn', D, None, num_heads, head_dim, vk, rt)
        self.cx = NumpyAttention(f'blocks.{idx}.cross_attn', D, ctx_dim, num_heads, head_dim, vk, rt)
        self.mlp = NumpyMLP(f'blocks.{idx}.mlp', D, mlp_hidden, vk, rt)
        self._sa_pfx = f'blocks.{idx}.adaln_modulation_self_attn'
        self._cx_pfx = f'blocks.{idx}.adaln_modulation_cross_attn'
        self._mlp_pfx = f'blocks.{idx}.adaln_modulation_mlp'

    def _adaln(self, emb, lora, prefix, n_chunks):
        """Mirrors PT Sequential(SiLU, Linear, Linear) — ONE SiLU at start only."""
        B, T, Di = emb.shape; M = B*T
        mod = self._rt.run_silu(emb.reshape(M,Di).astype(np.float32).reshape(-1)).reshape(M,Di)
        m1 = np.zeros((M, 256), dtype=np.float32)
        self._vk.vk_run_gemm(f'{prefix}.1.weight', mod, m1, M, 256, Di)
        # No SiLU here! PT has Sequential(SiLU, Linear, Linear) — only ONE SiLU
        m2 = np.zeros((M, n_chunks*Di), dtype=np.float32)
        self._vk.vk_run_gemm(f'{prefix}.2.weight', m1, m2, M, n_chunks*Di, 256)
        if lora is not None:
            m2 = m2 + lora.reshape(M, -1).astype(np.float32)[:, :n_chunks*Di]
        return np.split(m2.reshape(B, T, n_chunks*Di), n_chunks, axis=-1)

    def forward(self, x, emb, ctx, rope_emb, adaln_lora):
        B, T, H, W, D = x.shape

        # self-attention
        shift, scale, gate = self._adaln(emb, adaln_lora, self._sa_pfx, 3)
        x_n = self._rt.run_layernorm(x.reshape(B*T*H*W,D).astype(np.float32),D).reshape(B,T,H,W,D)
        x_m = x_n*(1.0+scale.reshape(B,T,1,1,D)) + shift.reshape(B,T,1,1,D)
        sa_out = self.sa.forward(x_m.reshape(B, T*H*W, D), None, rope_emb).reshape(B,T,H,W,D)
        x = x + gate.reshape(B,T,1,1,D)*sa_out

        # cross-attention
        shift, scale, gate = self._adaln(emb, adaln_lora, self._cx_pfx, 3)
        x_n = self._rt.run_layernorm(x.reshape(B*T*H*W,D).astype(np.float32),D).reshape(B,T,H,W,D)
        x_m = x_n*(1.0+scale.reshape(B,T,1,1,D)) + shift.reshape(B,T,1,1,D)
        cx_out = self.cx.forward(x_m.reshape(B, T*H*W, D), ctx, None).reshape(B,T,H,W,D)
        x = x + gate.reshape(B,T,1,1,D)*cx_out

        # MLP
        shift, scale, gate = self._adaln(emb, adaln_lora, self._mlp_pfx, 3)
        x_n = self._rt.run_layernorm(x.reshape(B*T*H*W,D).astype(np.float32),D).reshape(B,T,H,W,D)
        x_m = x_n*(1.0+scale.reshape(B,T,1,1,D)) + shift.reshape(B,T,1,1,D)
        mlp_out = self.mlp.forward(x_m)
        x = x + gate.reshape(B,T,1,1,D)*mlp_out
        return x


# ═══════════════════════════════════════════════════════════════
# NumpyAttention
# ═══════════════════════════════════════════════════════════════

class NumpyAttention:
    def __init__(self, prefix, D, ctx_dim, num_heads, head_dim, vk, rt):
        self._vk = vk; self._rt = rt; self.prefix = prefix
        self.D = D; self.ctx_dim = ctx_dim or D
        self.num_heads = num_heads; self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.is_selfattn = ctx_dim is None

    def forward(self, x, context, rope_emb):
        B, S, _ = x.shape; D, H, hd = self.D, self.num_heads, self.head_dim
        ctx = x if context is None else context; S_kv = ctx.shape[1]

        # Q projection [B*S, D] @ W_Q^T[D, inner_dim] → [B*S, inner_dim]
        x_f = x.reshape(B*S, D).astype(np.float32)
        q = np.zeros((B*S, self.inner_dim), dtype=np.float32)
        self._vk.vk_run_gemm(f'{self.prefix}.q_proj.weight', x_f, q, B*S, self.inner_dim, D)

        ctx_f = ctx.reshape(B*S_kv, self.ctx_dim).astype(np.float32)
        k = np.zeros((B*S_kv, self.inner_dim), dtype=np.float32)
        self._vk.vk_run_gemm(f'{self.prefix}.k_proj.weight', ctx_f, k, B*S_kv, self.inner_dim, self.ctx_dim)
        v = np.zeros((B*S_kv, self.inner_dim), dtype=np.float32)
        self._vk.vk_run_gemm(f'{self.prefix}.v_proj.weight', ctx_f, v, B*S_kv, self.inner_dim, self.ctx_dim)

        q = q.reshape(B, S, H, hd); k = k.reshape(B, S_kv, H, hd); v = v.reshape(B, S_kv, H, hd)

        q = self._rt.run_rmsnorm(q, f'{self.prefix}.q_norm')
        k = self._rt.run_rmsnorm(k, f'{self.prefix}.k_norm')

        if self.is_selfattn and rope_emb is not None:
            q = _numpy_rope(q, rope_emb); k = _numpy_rope(k, rope_emb)

        q_sdpa = q.transpose(0,2,1,3); k_sdpa = k.transpose(0,2,1,3); v_sdpa = v.transpose(0,2,1,3)
        attn_out = self._rt.run_sdpa(q_sdpa, k_sdpa, v_sdpa)

        attn_f = attn_out.transpose(0,2,1,3).reshape(B*S, self.inner_dim).astype(np.float32)
        o = np.zeros((B*S, D), dtype=np.float32)
        self._vk.vk_run_gemm(f'{self.prefix}.output_proj.weight', attn_f, o, B*S, D, self.inner_dim)
        return o.reshape(B, S, D)


# ═══════════════════════════════════════════════════════════════
# NumpyMLP
# ═══════════════════════════════════════════════════════════════

class NumpyMLP:
    def __init__(self, prefix, D, hidden, vk, rt):
        self._vk = vk; self._rt = rt; self.prefix = prefix
        self.D = D; self.hidden = hidden

    def forward(self, x):
        B, T, H, W, D = x.shape; M = B*T*H*W
        x_f = x.reshape(M, D).astype(np.float32)
        fc1 = np.zeros((M, self.hidden), dtype=np.float32)
        self._vk.vk_run_gemm(f'{self.prefix}.layer1.weight', x_f, fc1, M, self.hidden, D)
        gelu = self._rt.run_gelu(fc1.reshape(-1)).reshape(M, self.hidden)
        fc2 = np.zeros((M, D), dtype=np.float32)
        self._vk.vk_run_gemm(f'{self.prefix}.layer2.weight', gelu, fc2, M, D, self.hidden)
        return fc2.reshape(B, T, H, W, D)
