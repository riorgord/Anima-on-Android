"""Phase 1: Compute and dump reference data using PyTorch, then unload."""
import numpy as np, torch, sys, os, time
sys.path.insert(0,"/sdcard/anima_on_android/src")
import predict2, torch.nn.functional as F

DEV="cpu"; DTYPE=torch.float16
OUT="/sdcard/anima_on_android/output/ref_dump"
os.makedirs(OUT,exist_ok=True)

print("Loading DiT (PyTorch)...")
t0=time.time()
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
print(f"  loaded in {time.time()-t0:.1f}s")

MS,D,M = 512,2048,2; S=MS//M; n_elem=MS*D
torch.manual_seed(42)

# Generate realistic input from actual pipeline flow
# Simulate a real latent → x_embedder → t_embedder
torch.manual_seed(6666)
latent = torch.randn(1, 16, 32, 32, dtype=torch.float32)  # [1, 16, H, W] like real pipeline
sigma = torch.tensor([1.0])  # first step sigma

# Run x_embedder + t_embedder via full prepare_embedded_sequence
with torch.no_grad():
    x_b = latent.unsqueeze(2).repeat(2, 1, 1, 1, 1).to(DTYPE)  # CFG batch [2, 16, 1, 32, 32]
    x_emb, rope, _ = dit.prepare_embedded_sequence(x_b)
    x_in = x_emb.reshape(MS, D).float()  # [512, 2048]

    # Run t_embedder
    ts = sigma.repeat(2).unsqueeze(1).to(DTYPE)
    t_emb_out, adaln_lora = dit.t_embedder[1](dit.t_embedder[0](ts).to(DTYPE))
    t_emb = dit.t_embedding_norm(t_emb_out).float()

print(f"x_in range: [{float(x_in.min()):.3f}, {float(x_in.max()):.3f}]")
print(f"t_emb range: [{float(t_emb.min()):.3f}, {float(t_emb.max()):.3f}]")

# Save inputs
np.save(f"{OUT}/x_in.npy", x_in.numpy().astype(np.float16))
np.save(f"{OUT}/t_emb.npy", t_emb.numpy().astype(np.float16))
print("Inputs saved")

# Compute and save AdaLN for all blocks
print("Computing AdaLN for 28 blocks...")
adaln_all = np.zeros(28 * 9 * n_elem, dtype=np.uint16)

def adaln(emb, w1, w2):
    h = F.silu(emb.float())
    h = F.linear(h, w1.float())
    h = F.linear(h, w2.float())
    sh, sc, ga = torch.chunk(h, 3, dim=-1)
    sc = sc + 1.0
    return (sc.repeat_interleave(S,0).numpy().astype(np.float16).ravel().view(np.uint16),
            sh.repeat_interleave(S,0).numpy().astype(np.float16).ravel().view(np.uint16),
            ga.repeat_interleave(S,0).numpy().astype(np.float16).ravel().view(np.uint16))

x_ref = x_in.float().clone()
for i in range(28):
    pfx = f"blocks.{i}."
    sc_s, sh_s, ga_s = adaln(t_emb,
        sd[pfx+"adaln_modulation_self_attn.1.weight"],
        sd[pfx+"adaln_modulation_self_attn.2.weight"])
    sc_m, sh_m, ga_m = adaln(t_emb,
        sd[pfx+"adaln_modulation_mlp.1.weight"],
        sd[pfx+"adaln_modulation_mlp.2.weight"])

    base = i * 9 * n_elem
    adaln_all[base+0*n_elem:base+1*n_elem] = sc_s
    adaln_all[base+1*n_elem:base+2*n_elem] = sh_s
    adaln_all[base+2*n_elem:base+3*n_elem] = ga_s
    adaln_all[base+6*n_elem:base+7*n_elem] = sc_m
    adaln_all[base+7*n_elem:base+8*n_elem] = sh_m
    adaln_all[base+8*n_elem:base+9*n_elem] = ga_m

    # Compute reference (simplified: self-attn + MLP, no cross-attn)
    w_q = sd[pfx+"self_attn.q_proj.weight"]
    w_k = sd[pfx+"self_attn.k_proj.weight"]
    w_v = sd[pfx+"self_attn.v_proj.weight"]
    w_o = sd[pfx+"self_attn.output_proj.weight"]
    w_qn = sd[pfx+"self_attn.q_norm.weight"].float()
    w_l1 = sd[pfx+"mlp.layer1.weight"]
    w_l2 = sd[pfx+"mlp.layer2.weight"]

    ln = F.layer_norm(x_ref, (D,), weight=None, bias=None, eps=1e-6)
    mod = ln * torch.from_numpy(sc_s.view(np.float16).reshape(MS,D).astype(np.float32)) + \
                torch.from_numpy(sh_s.view(np.float16).reshape(MS,D).astype(np.float32))
    q = F.linear(mod, w_q.float()); v = F.linear(mod, w_v.float())
    q = F.rms_norm(q.reshape(MS*16,128),(128,),weight=w_qn,eps=1e-6).reshape(MS,D)
    o = F.linear(v, w_o.float())
    x_ref = x_ref + torch.from_numpy(ga_s.view(np.float16).reshape(MS,D).astype(np.float32)) * o

    ln2 = F.layer_norm(x_ref, (D,), weight=None, bias=None, eps=1e-6)
    mod2 = ln2 * torch.from_numpy(sc_m.view(np.float16).reshape(MS,D).astype(np.float32)) + \
                 torch.from_numpy(sh_m.view(np.float16).reshape(MS,D).astype(np.float32))
    h = F.linear(mod2, w_l1.float()); h = F.silu(h)
    fc2 = F.linear(h, w_l2.float())
    x_ref = x_ref + torch.from_numpy(ga_m.view(np.float16).reshape(MS,D).astype(np.float32)) * fc2

# Save
adaln_all.tofile(f"{OUT}/adaln_all.bin")
np.save(f"{OUT}/ref_out.npy", x_ref.half().numpy())
print(f"AdaLN + reference saved ({adaln_all.nbytes/1e6:.1f}MB + {x_ref.nbytes/1e6:.1f}MB)")

del sd, dit  # UNLOAD PyTorch
print("PyTorch unloaded. Ready for C++ phase.")
