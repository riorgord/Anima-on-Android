"""Compare a specific weight tensor between .pt and .bin — bit-exact check."""
import torch, numpy as np, struct

sd = torch.load('/mnt/d/AI/anima_phone/models/diffusion_weights_fp16.pt', weights_only=True, map_location='cpu')

# Find the PT weight
pt_key = None
for k in sd:
    if 'blocks.0.self_attn.q_proj' in k:
        pt_key = k
        break

if pt_key:
    pt_f32 = sd[pt_key].numpy().astype(np.float32)
    pt_u16 = sd[pt_key].numpy().view(np.uint16).flatten()
    print(f'.pt: {pt_key} shape={pt_f32.shape} first5_fp16={pt_u16[:5]}')

with open('/mnt/d/AI/anima_phone/models/diffusion_weights.bin', 'rb') as f:
    N = struct.unpack('<I', f.read(4))[0]
    for i in range(N):
        nl = struct.unpack('<H', f.read(2))[0]
        name = f.read(nl).decode()
        nd = struct.unpack('<B', f.read(1))[0]
        sh = []; elems = 1
        for d in range(nd): s = struct.unpack('<I', f.read(4))[0]; sh.append(s); elems *= s
        data = f.read(elems * 2)
        if 'blocks.0.self_attn.q_proj' in name:
            bin_u16 = np.frombuffer(data, dtype=np.uint16)
            print(f'.bin: {name} shape={sh} first5={bin_u16[:5]}')
            diff = (pt_u16 != bin_u16).sum()
            print(f'  DIFFER: {diff}/{len(bin_u16)} ({100*diff/len(bin_u16):.4f}%)')
            if diff > 0:
                first = np.where(pt_u16 != bin_u16)[0][:3]
                for idx in first:
                    print(f'  idx={idx}: .pt={pt_u16[idx]:04x} .bin={bin_u16[idx]:04x}')
            break
