"""Quick compare: phone C++ dump vs PC whitebox reference."""
import numpy as np, os, sys

CMP = "/mnt/d/AI/anima_phone/output/cmp"
REF = "/mnt/d/AI/anima_phone/output/whitebox"

# Load phone Block 0 intermediates
phone = {}
for f in ["x_phone", "ctx_phone"]:
    phone[f] = np.load(f"{CMP}/{f}.npy").astype(np.float32)

for f in ["b0_sa", "b0_cx", "b0_mlp", "b0_q_norm", "b0_k_norm", "b0_v_raw", "b0_scores", "b0_attn_o"]:
    phone[f] = np.load(f"{CMP}/{f}.npy").astype(np.float32)

# Load PC whitebox intermediates
pt = {}
for name in ["sa_residual", "cx_residual", "mlp_residual",
             "sa_q_norm", "sa_k_norm", "sa_v_raw",
             "sa_attn_o", "sa_q_roped", "sa_k_roped",
             "sa_ln", "sa_modulated", "sa_q_raw", "sa_k_raw",
             "sa_o_proj"]:
    path = f"{REF}/b0/intermediates/{name}.npy"
    if os.path.exists(path):
        pt[name] = np.load(path).astype(np.float32)

# Block 0 intermediate comparison
mapping = [
    ("b0_sa", "sa_residual", "SA residual"),
    ("b0_cx", "cx_residual", "CX residual"),
    ("b0_mlp", "mlp_residual", "MLP residual"),
    ("b0_q_norm", "sa_q_norm", "Q after RMSNorm"),
    ("b0_k_norm", "sa_k_norm", "K after RMSNorm"),
    ("b0_v_raw", "sa_v_raw", "V raw"),
    ("b0_attn_o", "sa_attn_o", "Attn output"),
]

print("=" * 80)
print("Block 0 intermediate comparison: Phone C++ vs PC White-box")
print(f"{'Variable':<20s} {'Phone range':<32s} {'PT range':<32s} {'max_err':<10s} {'mean_err':<10s}")
print("-" * 104)

for ph_name, pt_name, label in mapping:
    ph = phone[ph_name]
    ref = pt[pt_name].flatten()
    min_len = min(len(ph), len(ref))
    ph_a = ph[:min_len]
    ref_a = ref[:min_len]
    ok = np.isfinite(ph_a) & np.isfinite(ref_a)
    if ok.sum() == 0:
        print(f"  {label:<18s}  ALL NaN")
        continue
    diff = np.abs(ph_a[ok] - ref_a[ok])
    max_e = diff.max()
    mean_e = diff.mean()
    ph_rng = f"[{ph_a[ok].min():.4f}, {ph_a[ok].max():.4f}]"
    pt_rng = f"[{ref_a[ok].min():.4f}, {ref_a[ok].max():.4f}]"
    flag = " ⚠️" if max_e > 1 else (" ✓" if max_e < 0.02 else "")
    print(f"  {label:<18s}  {ph_rng:<32s} {pt_rng:<32s} {max_e:<10.4f} {mean_e:<10.6f}{flag}")

# Input comparison
print("\nInput comparison:")
for inp_name, lbl in [("x_phone","x_input"), ("ctx_phone","ctx_input")]:
    ph = phone[inp_name].flatten()
    pt_inp = np.load(f"{REF}/{lbl}.npy").astype(np.float32).flatten()
    min_len = min(len(ph), len(pt_inp))
    ok = np.isfinite(ph[:min_len]) & np.isfinite(pt_inp[:min_len])
    diff = np.abs(ph[:min_len][ok] - pt_inp[:min_len][ok])
    flag = " ✓ identical" if diff.max() < 0.001 else " ⚠️ DIFFER"
    print(f"  {lbl}: max_err={diff.max():.6f}  mean_err={diff.mean():.8f}{flag}")

# Detail drill-down: find WHERE the max error occurs in Q_norm
print("\n" + "=" * 80)
print("Drill-down: Q after RMSNorm — max error locations")
print("=" * 80)
ph_q = phone["b0_q_norm"].flatten()
pt_q = np.load(f"{REF}/b0/intermediates/sa_q_norm.npy").astype(np.float32).flatten()
min_len = min(len(ph_q), len(pt_q))
ok = np.isfinite(ph_q[:min_len]) & np.isfinite(pt_q[:min_len])
diff = np.abs(ph_q[:min_len][ok] - pt_q[:min_len][ok])
# Find top-10 largest error indices
top_idx = np.argsort(-diff)[:10]
print(f"  Q_norm total elements compared: {ok.sum()}")
print(f"  Top-10 worst elements:")
print(f"  {'idx':<10s} {'flat_idx':<12s} {'Phone':<14s} {'PT':<14s} {'abs_err':<12s}")
# Decode flat index to (row, col)
HEAD_DIM = 128
for rank, idx in enumerate(top_idx):
    actual_idx = np.where(ok)[0][idx]
    # row = actual_idx // HEAD_DIM, col = actual_idx % HEAD_DIM
    print(f"  #{rank}:     {actual_idx:<10d}  {ph_q[actual_idx]:.6f}      {pt_q[actual_idx]:.6f}      {diff[actual_idx]:.6f}")

# Check: are the errors clustered at specific rows or columns?
ph_rows = ph_q.reshape(-1, HEAD_DIM)
pt_rows = pt_q.reshape(-1, HEAD_DIM)
min_rows = min(len(ph_rows), len(pt_rows))
row_max_err = np.zeros(min_rows)
for r in range(min_rows):
    ok_r = np.isfinite(ph_rows[r]) & np.isfinite(pt_rows[r])
    if ok_r.sum() > 0:
        row_max_err[r] = np.abs(ph_rows[r][ok_r] - pt_rows[r][ok_r]).max()
bad_rows = np.where(row_max_err > 0.02)[0]
print(f"\n  Rows with max_err > 0.02: {len(bad_rows)}/{min_rows}")
if len(bad_rows) > 0:
    print(f"  First 10 bad row indices: {bad_rows[:10]}")
    print(f"  Bad row max_err range: [{row_max_err[bad_rows].min():.4f}, {row_max_err[bad_rows].max():.4f}]")
    # Are bad rows contiguous?
    gaps = np.diff(bad_rows)
    print(f"  Bad row gaps: min={gaps.min()}, max={gaps.max()}, mean={gaps.mean():.1f}")

# Per-block output comparison
print("\n" + "=" * 80)
print("Per-block output comparison")
print(f"{'Block':<6} {'Phone C++ range':<34} {'PC PT range':<34} {'max_err':<10}")
print("-" * 84)

for b in range(28):
    cpp = np.load(f"{CMP}/block_{b:02d}_cpp.npy").astype(np.float32).flatten()
    pt_blk = np.load(f"{REF}/block_{b:02d}_pt.npy").astype(np.float32).flatten()
    ok = np.isfinite(cpp) & np.isfinite(pt_blk)
    if ok.sum() == 0:
        print(f"  {b:2d}    ALL NaN")
        continue
    diff = np.abs(cpp[ok] - pt_blk[ok])
    cpp_rng = f"[{cpp[ok].min():.2f}, {cpp[ok].max():.2f}]"
    pt_rng = f"[{pt_blk[ok].min():.2f}, {pt_blk[ok].max():.2f}]"
    flag = " ⚠️" if diff.max() > 100 else (" ⚠️ small" if diff.max() > 1 else "")
    print(f"  {b:2d}    {cpp_rng:<34} {pt_rng:<34} {diff.max():<10.2f}{flag}")

print("\nDone.")
