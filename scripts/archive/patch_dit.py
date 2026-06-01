"""Replace the B1 segment call in dit_engine.cpp to split into B1a+B1b."""
import sys

path = '/mnt/d/AI/anima_phone/vulkan/dit_engine.cpp'
with open(path, 'r') as f:
    lines = f.readlines()

# Find the B1 segment call line
for i, line in enumerate(lines):
    if 'record_segment_self_pre(rc, b, inBuf);' in line:
        print(f'Found at line {i+1}: {line.rstrip()}')

        # Replace line i with B1a call
        lines[i] = line.replace('record_segment_self_pre(rc, b, inBuf);',
                                'record_segment_self_pre_a(rc, b, inBuf);')

        # Fix the LOGE message on line i+1
        if 'SegB1' in lines[i+1]:
            lines[i+1] = lines[i+1].replace('SegB1', 'SegB1a')

        # Replace the GPU executed comment and capture block
        # Find the old capture block (lines i+2 to i+10 approx)
        # Remove old captures from line i+2 to i+8 (the B1 captures)
        # and insert new B1a_captures + B1b call + B1b captures

        # Find the "if (b == 0) {" for B1 captures
        b1_cap_start = None
        b1_cap_end = None
        for j in range(i+2, min(i+15, len(lines))):
            if 'if (b == 0)' in lines[j] and 'g_b0_q' in lines[j+1]:
                b1_cap_start = j
            if b1_cap_start and 'g_b0_v' in lines[j] and 'memcpy' in lines[j]:
                b1_cap_end = j + 1  # include the closing }
                break

        if b1_cap_start and b1_cap_end:
            # Build replacement
            indent = '            '
            replacement = [
                f'{indent}if (b == 0) {{\n',
                f'{indent}    if (g_b0_mod)   memcpy(g_b0_mod,   g_nBuf.mapped, g_block_out_size);\n',
                f'{indent}    if (g_b0_q_raw) memcpy(g_b0_q_raw, g_tQ.mapped, g_block_out_size);\n',
                f'{indent}    if (g_b0_v)     memcpy(g_b0_v,     g_tV.mapped, MS * N_HEADS * HEAD_DIM * 2);\n',
                f'{indent}}}\n',
                f'{indent}// Segment B1b: RMSNorm Q/K -> RoPE\n',
                f'{indent}record_segment_self_pre_b(rc, b);\n',
                f'{indent}if (!submit_segment()) {{ LOGE("SegB1b[%d] submit failed", b); return false; }}\n',
                f'{indent}if (b == 0) {{\n',
                f'{indent}    size_t qkv_sz = MS * N_HEADS * HEAD_DIM * 2;\n',
                f'{indent}    if (g_b0_q)        memcpy(g_b0_q,        g_tQ.mapped, qkv_sz);\n',
                f'{indent}    if (g_b0_k)        memcpy(g_b0_k,        g_tK.mapped, qkv_sz);\n',
                f'{indent}    if (g_b0_q_roped)  memcpy(g_b0_q_roped,  g_rBuf.mapped, qkv_sz);\n',
                f'{indent}    if (g_b0_k_roped)  memcpy(g_b0_k_roped,  g_attnO.mapped, qkv_sz);\n',
                f'{indent}}}\n',
            ]
            lines[b1_cap_start:b1_cap_end] = replacement
            print(f'Replaced lines {b1_cap_start+1}-{b1_cap_end}')
        else:
            print('Could not find B1 capture block')

        break
else:
    print('NOT FOUND: record_segment_self_pre')
    sys.exit(1)

with open(path, 'w') as f:
    f.writelines(lines)
print('Done. dit_engine.cpp updated.')
