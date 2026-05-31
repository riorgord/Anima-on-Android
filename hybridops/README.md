# HybridOps Pipeline (57s/step, May 25-26 2026)

## Files

### scripts/
- phone_pipeline.py: 主管线 (sigma直传, num_blocks=28, HybridOps)
- vk_ops.py: HybridOps 类 (替换 torch nn.Linear/nn.LayerNorm 等)
- vk_linear.py: GEMM via libvk_gemm.so
- vk_hybrid_ops.py: LayerNorm/RMSNorm/GELU/SiLU via libdit_vk.so
- vk_bridge.py: Vulkan bridge helpers
- vk_ops_diag.py: 诊断版本

### src/
- predict2.py: DiT 模型 (MiniTrainDIT, Block, Attention, GPT2FeedForward)
- llm_adapter.py: LLMAdapter
- position_embedding.py: VideoRopePosition3DEmb
- wan_vae.py: WanVAE decoder (无 latent norm bug 版本)

### vulkan/
- gemm.comp: 原始 generic tiled GEMM (14 GFLOPS)

## 速度
- 50-57s/step (HybridOps + GEMM 149 GFLOPS)
- 出图 81KB PNG (干净)

## 与当前 C++ 引擎的区别
- HybridOps: Python ctypes 逐层调用 Vulkan, PyTorch 做 RoPE/final_layer
- 当前 C++ 引擎: 全 C++ per-step recording, 28 block 全 GPU

## 已知问题（统一两份管线时需处理）

| 问题 | 说明 |
|------|------|
| 权重双重加载 | PyTorch small.pt + C++ .bin 都含 t_embedder |
| 两次模型构建 | PyTorch DiT(num_blocks=0) + C++ 28 blocks |
| PyTorch 残留 | x_embedder, final_layer 仍用 PyTorch CPU, 待 C++ 化 |
| GEMM 精度 | HybridOps fp16 dot 累加 vs 当前 fp32 fma (当前更优) |
| RoPE | HybridOps 全 PyTorch CPU vs 当前 C++ GPU (当前更优) |
| 出图 | HybridOps 81KB vs 当前 105KB vs PC 74KB |
