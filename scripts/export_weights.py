"""Export DiT weights from PyTorch .pt to flat binary for C++ Vulkan engine.
Format: [4B num_tensors N] then for each:
  [2B name_len] [name bytes] [1B ndim] [ndim*4B shape] [raw fp16 data]
"""
import sys, struct, torch

def export(pt_path, bin_path):
    sd = torch.load(pt_path, weights_only=True, map_location='cpu')
    with open(bin_path, 'wb') as f:
        f.write(struct.pack('<I', len(sd)))
        for name, tensor in sd.items():
            # Strip "net." prefix to match C++ engine expectations
            clean_name = name
            while clean_name.startswith("net."):
                clean_name = clean_name[4:]
            # Convert to fp16 if needed
            t = tensor.detach().cpu()
            if t.dtype != torch.float16:
                t = t.half()
            data = t.numpy().tobytes()
            name_bytes = clean_name.encode('utf-8')
            f.write(struct.pack('<H', len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack('<B', len(t.shape)))
            for d in t.shape:
                f.write(struct.pack('<I', d))
            f.write(data)
    print(f"Exported {len(sd)} tensors, {bin_path}")

if __name__ == '__main__':
    src = sys.argv[1] if len(sys.argv) > 1 else 'D:/AI/anima_phone/models/diffusion_weights_fp16.pt'
    dst = sys.argv[2] if len(sys.argv) > 2 else 'D:/AI/anima_phone/models/diffusion_weights.bin'
    export(src, dst)
