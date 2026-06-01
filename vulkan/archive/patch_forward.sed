# Replace B1 segment call + all captures with split version
/            \/\/ Segment B1: LN→AdaLN→QKV→RMSNorm→RoPE (pre-attention)/{
    s/record_segment_self_pre(rc, b, inBuf);/record_segment_self_pre_a(rc, b, inBuf);/
    s/SegB1/SegB1a/
    # Change next line to new capture
    /GPU executed.*V_raw/{
        c\            if (b == 0) {\n                if (g_b0_mod)   memcpy(g_b0_mod,   g_nBuf.mapped, g_block_out_size);\n                if (g_b0_q_raw) memcpy(g_b0_q_raw, g_tQ.mapped, g_block_out_size);\n                if (g_b0_v)     memcpy(g_b0_v,     g_tV.mapped, MS * N_HEADS * HEAD_DIM * 2);\n            }\n            record_segment_self_pre_b(rc, b);\n            if (!submit_segment()) { LOGE(""SegB1b[%d] submit failed"", b); return false; }\n            if (b == 0) {\n                size_t qkv_sz = MS * N_HEADS * HEAD_DIM * 2;\n                if (g_b0_q)        memcpy(g_b0_q,        g_tQ.mapped, qkv_sz);\n                if (g_b0_k)        memcpy(g_b0_k,        g_tK.mapped, qkv_sz);\n                if (g_b0_q_roped)  memcpy(g_b0_q_roped,  g_rBuf.mapped, qkv_sz);\n                if (g_b0_k_roped)  memcpy(g_b0_k_roped,  g_attnO.mapped, qkv_sz);\n            }
    }
}
