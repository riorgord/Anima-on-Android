"""Quick check if image is snow or clean."""
from PIL import Image; import numpy as np; import sys
img = Image.open(sys.argv[1])
arr = np.array(img, dtype=np.float32)
hdiff = np.abs(arr[:, 1:, :] - arr[:, :-1, :]).mean()
vdiff = np.abs(arr[1:, :, :] - arr[:-1, :, :]).mean()
print(f"Shape: {arr.shape}  range=[{arr.min()},{arr.max()}]")
print(f"Horizontal diff mean: {hdiff:.1f}  (snow ~40-80, clean ~5-20)")
print(f"Vertical diff mean: {vdiff:.1f}")
print(f"Channel means: R={arr[:,:,0].mean():.1f} G={arr[:,:,1].mean():.1f} B={arr[:,:,2].mean():.1f}")
hist = np.histogram(arr[:,:,0], bins=10, range=(0,255))[0]
print(f"R histogram: {hist}")
print("Uniform distribution = snow, Clustered = clean image")
