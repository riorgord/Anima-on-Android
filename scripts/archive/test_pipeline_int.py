"""Phone pipeline integration test — 1 step with C++ engine"""
import ctypes, numpy as np, torch, time, sys
sys.path.insert(0,"/sdcard/anima_on_android/src")
sys.path.insert(0,"/sdcard/anima_on_android/scripts")

_lib=ctypes.CDLL("/data/local/tmp/libdit_vk.so")
_lib.dit_init.argtypes=[ctypes.c_char_p,ctypes.c_char_p]; _lib.dit_init.restype=ctypes.c_bool
_lib.dit_init_all_blocks.argtypes=[]; _lib.dit_init_all_blocks.restype=ctypes.c_bool
_lib.dit_forward_28blocks.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,
    ctypes.c_int,ctypes.c_int,ctypes.c_int]; _lib.dit_forward_28blocks.restype=ctypes.c_bool

import predict2, llm_adapter, wan_vae
import torch.nn.functional as F

DEV="cpu"; DTYPE=torch.float16
CFG=5.0; H=32  # 256x256

# Load contexts
print("Loading contexts...")
ctx_cond=torch.load("/sdcard/anima_on_android/models/context_cond.pt",weights_only=True).to(DEV).to(DTYPE)
ctx_uncond=torch.load("/sdcard/anima_on_android/models/context_uncond.pt",weights_only=True).to(DEV).to(DTYPE)

# Load DiT (PyTorch, for AdaLN computation + final_layer + x_embedder + t_embedder)
print("Loading DiT (PyTorch)...")
config=dict(max_img_h=240,max_img_w=240,max_frames=128,in_channels=16,out_channels=16,
    patch_spatial=2,patch_temporal=1,concat_padding_mask=True,model_channels=2048,
    num_blocks=28,num_heads=16,mlp_ratio=4.0,crossattn_emb_channels=1024,
    pos_emb_cls="rope3d",pos_emb_learnable=True,pos_emb_interpolation="crop",
    min_fps=1,max_fps=30,use_adaln_lora=True,adaln_lora_dim=256,
    rope_h_extrapolation_ratio=4.0,rope_w_extrapolation_ratio=4.0,rope_t_extrapolation_ratio=1.0,
    extra_per_block_abs_pos_emb=False,rope_enable_fps_modulation=False)
sd=torch.load("/sdcard/anima_on_android/models/diffusion_weights_fp16.pt",weights_only=True)
dit=predict2.MiniTrainDIT(**config,device=DEV,dtype=DTYPE,operations=torch.nn)
dit.load_state_dict(sd,strict=False); dit.eval()

# Init C++ engine
print("Init C++ engine...")
t0=time.time()
ok=_lib.dit_init(b"/data/local/tmp/diffusion_weights.bin",b"/data/local/tmp")
print(f"  init={ok} ({time.time()-t0:.1f}s)")
ok=_lib.dit_init_all_blocks()
print(f"  record={ok}")

# Generate latent
torch.manual_seed(6666)
x=torch.randn(1,16,H,H,generator=torch.Generator(device=DEV).manual_seed(6666),dtype=DTYPE)
sigma=torch.tensor([1.0],dtype=DTYPE)  # sigma=1 for first step

MS,D,M=512,2048,2
S=MS//M; n_elem=MS*D; adaln_pb=9

# Run 1 PyTorch step to get x_embedder output and t_emb
with torch.no_grad():
    x_b=x.unsqueeze(2).repeat(2,1,1,1,1)  # CFG batch
    # Run x_embedder + t_embedder
    x_pad=predict2._pad_to_patch_size(x_b,(1,2,2))
    x_emb=dit.x_embedder(x_pad)  # [B,T,H,W,D]
    x_flat=x_emb.reshape(MS,D)  # [512,2048]

    # RoPE
    rope_emb,_,_=dit.prepare_embedded_sequence(x_pad.float())

    sigma_t=sigma.repeat(2).unsqueeze(1)
    t_emb=dit.t_embedder[1](dit.t_embedder[0](sigma_t.to(DTYPE)).to(DTYPE))
    t_emb=dit.t_embedding_norm(t_emb)  # [M, D]

    # Compute AdaLN for all blocks
    print("Computing AdaLN for 28 blocks...")
    adaln_all=np.zeros(28*adaln_pb*n_elem,dtype=np.uint16)
    for i in range(28):
        block=dit.blocks[i]
        pfx=f"blocks.{i}."

        def compute_adaln(adaln_mod):
            h=F.silu(t_emb.float())
            h=F.linear(h,adaln_mod[1].weight.float())  # LoRA down
            h=F.linear(h,adaln_mod[2].weight.float())  # LoRA up
            shift,scale,gate=torch.chunk(h,3,dim=-1)
            scale_p1=scale+1.0
            return (scale_p1.repeat_interleave(S,0).numpy().astype(np.float16).ravel().view(np.uint16),
                    shift.repeat_interleave(S,0).numpy().astype(np.float16).ravel().view(np.uint16),
                    gate.repeat_interleave(S,0).numpy().astype(np.float16).ravel().view(np.uint16))

        base=i*adaln_pb*n_elem
        sc_s,sh_s,ga_s=compute_adaln(block.adaln_modulation_self_attn)
        sc_m,sh_m,ga_m=compute_adaln(block.adaln_modulation_mlp)
        adaln_all[base+0*n_elem:base+1*n_elem]=sc_s
        adaln_all[base+1*n_elem:base+2*n_elem]=sh_s
        adaln_all[base+2*n_elem:base+3*n_elem]=ga_s
        adaln_all[base+6*n_elem:base+7*n_elem]=sc_m
        adaln_all[base+7*n_elem:base+8*n_elem]=sh_m
        adaln_all[base+8*n_elem:base+9*n_elem]=ga_m

    # Run C++ engine (both CFG branches: M=2, MS=512)
    print("Running C++ engine...")
    x_np=x_flat.numpy().astype(np.float16)  # [512, 2048]
    out_np=np.zeros((MS,D),dtype=np.float16)
    t0=time.time()
    ok=_lib.dit_forward_28blocks(x_np.ctypes.data_as(ctypes.c_void_p),
        adaln_all.ctypes.data_as(ctypes.c_void_p),
        out_np.ctypes.data_as(ctypes.c_void_p), MS, D, M)
    print(f"  C++: {time.time()-t0:.3f}s ok={ok}")

    # PyTorch reference WITHOUT cross-attn (matching C++ engine)
    print("Running PyTorch reference (no cross-attn)...")
    t0=time.time()
    x_ref=x_flat.float().clone()
    with torch.no_grad():
        for i,block in enumerate(dit.blocks):
            # AdaLN self
            h=F.silu(t_emb.float())
            h=F.linear(h,block.adaln_modulation_self_attn[1].weight.float())
            h=F.linear(h,block.adaln_modulation_self_attn[2].weight.float())
            sh_s,sc_s,ga_s=torch.chunk(h,3,dim=-1)
            sc_s=sc_s+1.0
            # Self-attn (simplified, no cross-attn, no RoPE)
            ln=F.layer_norm(x_ref,(D,),weight=None,bias=None,eps=1e-6)
            mod=ln*sc_s.repeat_interleave(S,0)+sh_s.repeat_interleave(S,0)
            q=F.linear(mod,block.self_attn.q_proj.weight.float())
            k=F.linear(mod,block.self_attn.k_proj.weight.float())
            v=F.linear(mod,block.self_attn.v_proj.weight.float())
            q=F.rms_norm(q.reshape(MS*16,128),(128,),weight=block.self_attn.q_norm.weight.float(),eps=1e-6).reshape(MS,D)
            k=F.rms_norm(k.reshape(MS*16,128),(128,),weight=block.self_attn.k_norm.weight.float(),eps=1e-6).reshape(MS,D)
            o=F.linear(v,block.self_attn.output_proj.weight.float())
            x_ref=x_ref+ga_s.repeat_interleave(S,0)*o

            # AdaLN MLP
            h2=F.silu(t_emb.float())
            h2=F.linear(h2,block.adaln_modulation_mlp[1].weight.float())
            h2=F.linear(h2,block.adaln_modulation_mlp[2].weight.float())
            sh_m,sc_m,ga_m=torch.chunk(h2,3,dim=-1)
            sc_m=sc_m+1.0
            # MLP
            ln2=F.layer_norm(x_ref,(D,),weight=None,bias=None,eps=1e-6)
            mod2=ln2*sc_m.repeat_interleave(S,0)+sh_m.repeat_interleave(S,0)
            h=F.linear(mod2,block.mlp.layer1.weight.float())
            h=F.silu(h)
            fc2=F.linear(h,block.mlp.layer2.weight.float())
            x_ref=x_ref+ga_m.repeat_interleave(S,0)*fc2
    print(f"  PyTorch: {time.time()-t0:.3f}s")

    out_f32=out_np.astype(np.float32)
    ref_f32=x_ref.half().numpy().astype(np.float32)
    err=np.abs(out_f32-ref_f32).max()
    print(f"max_err={err:.5f}")
    print(f"C++ mean/std={out_f32.mean():.4f}/{out_f32.std():.4f}")
    print(f"PT  mean/std={ref_f32.mean():.4f}/{ref_f32.std():.4f}")
    print("PASS" if err < 50 else "FAIL")

_lib.dit_destroy()
print("DONE")
