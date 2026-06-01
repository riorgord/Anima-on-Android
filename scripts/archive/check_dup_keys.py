import torch
sd = torch.load('/mnt/d/AI/anima_phone/models/diffusion_weights_fp16.pt', weights_only=True, map_location='cpu')
stripped = {}
dups = 0
for k in sd:
    ck = k
    while ck.startswith('net.'): ck = ck[4:]
    if ck in stripped:
        dups += 1
        print(f'DUPLICATE: {k} -> {ck}')
        print(f'  already from: {stripped[ck]}')
    else:
        stripped[ck] = k
print(f'\nTotal: {len(sd)} keys, unique stripped: {len(stripped)}, duplicates: {dups}')

# Show adaln self-attn weights for block 0
print('\nBlock 0 adaln self-attn weights:')
for k in sorted(stripped):
    if 'blocks.0.adaln' in k:
        v = sd[stripped[k]]
        print(f'  {k}: shape={list(v.shape)} range=[{v.min():.4f},{v.max():.4f}]')
