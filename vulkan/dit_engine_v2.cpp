// DiT Vulkan Inference Engine v2
// BF16 weights (safetensors direct) + FP32 compute throughout
// Per-call segmented dispatch (TDR-safe).  All 28 blocks, self+cross+MLP.
// Target: Snapdragon 8+ Gen 1 (Adreno 730), Android NDK.
//
// Ported from dit_engine.cpp — same dispatch architecture, upgraded precision.
// Gate anti-overflow scaling REMOVED — fp32 doesn't need it (range ~3.4e38).

#include <vulkan/vulkan.h>
#include <android/log.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <string>
#include <cmath>

#include "safetensors_reader.h"
#include "head_tail_ops.h"

#define LOG_TAG "DiT_VKv2"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// ── Model constants ──
static const uint32_t D          = 2048;
static const uint32_t CtxD       = 1024;
static const uint32_t Nctx       = 512;
static const uint32_t N_HEADS    = 16;
static const uint32_t HEAD_DIM   = 128;
static const uint32_t MLP_HIDDEN = 8192;
static const uint32_t ADALN_LORA_DIM = 256;
static const uint32_t D3         = 6144;  // 3*D

static uint32_t S  = 256;   // tokens per batch item
static uint32_t M  = 2;     // CFG batch
static uint32_t MS = 512;   // total tokens

// ── Push constant structs ──
struct PC_Gemm      { uint32_t M, N, K, batch; float alpha; };
struct PC_LayerNorm { uint32_t n_rows, n_elems; float eps; };
struct PC_RmsNorm   { uint32_t n_rows, n_elems; float eps; };
struct PC_Element   { uint32_t n_total; };
struct PC_ScaleShift{ uint32_t n_total, scale_stride, shift_stride; };
struct PC_Rope      { uint32_t N, head_dim; };
struct PC_Broadcast { uint32_t M, D, repeat; };
struct PC_AttnQKT   { uint32_t M_q, M_kv, H, D; float scale; };
struct PC_AttnSoftmax { uint32_t M_q, M_kv, H; };
struct PC_AttnOut   { uint32_t M_q, M_kv, H, D; };
struct PC_Gate      { uint32_t n_total; };

// ── Buffer ──
struct Buffer {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    size_t size = 0;
    void* mapped = nullptr;
};

// ── Shader pipeline ──
struct ShaderPipe {
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
};

// ── Vulkan context ──
struct VKCtx {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physDev = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t qFamily = 0;
    VkCommandPool cmdPool = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkFence stepFence = VK_NULL_HANDLE;
    VkDescriptorPool descPool  = VK_NULL_HANDLE;  // pre-recorded blocks
    VkDescriptorPool stepPool  = VK_NULL_HANDLE;  // per-step segments
    VkPhysicalDeviceMemoryProperties memProps = {};

    ShaderPipe gemm_bf16, layernorm_fp32, rms_norm_fp32, silu_fp32, gelu_fp32, scale_shift_fp32, rope_fp32, broadcast_fp32;
    ShaderPipe attn_qkt_fp32, attn_softmax_fp32, attn_out_fp32, gate_fp32;
};

// ── Helpers ──
static uint32_t find_mem_type(VKCtx& ctx, uint32_t typeBits, VkMemoryPropertyFlags props) {
    for (uint32_t i = 0; i < ctx.memProps.memoryTypeCount; i++)
        if ((typeBits & (1u << i)) && (ctx.memProps.memoryTypes[i].propertyFlags & props) == props)
            return i;
    return ~0u;
}

static bool create_buf(VKCtx& ctx, size_t size, VkBufferUsageFlags usage, Buffer& buf) {
    VkBufferCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = size; info.usage = usage; info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(ctx.device, &info, nullptr, &buf.buf) != VK_SUCCESS) return false;
    VkMemoryRequirements reqs;
    vkGetBufferMemoryRequirements(ctx.device, buf.buf, &reqs);
    VkMemoryAllocateInfo alloc = {};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = reqs.size;
    alloc.memoryTypeIndex = find_mem_type(ctx, reqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    if (alloc.memoryTypeIndex == ~0u)
        alloc.memoryTypeIndex = find_mem_type(ctx, reqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (vkAllocateMemory(ctx.device, &alloc, nullptr, &buf.mem) != VK_SUCCESS) return false;
    if (vkBindBufferMemory(ctx.device, buf.buf, buf.mem, 0) != VK_SUCCESS) return false;
    buf.size = size;
    vkMapMemory(ctx.device, buf.mem, 0, size, 0, &buf.mapped);
    return true;
}

static std::vector<uint32_t> load_spv(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { LOGE("Cannot open %s", path); return {}; }
    fseek(f, 0, SEEK_END); size_t size = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint32_t> code(size / 4);
    if (fread(code.data(), 1, size, f) != size) { fclose(f); return {}; }
    fclose(f); return code;
}

static bool create_pipe(VKCtx& ctx, const char* spv_path, uint32_t nBindings,
                        size_t pushSize, ShaderPipe& sp) {
    auto code = load_spv(spv_path);
    if (code.empty()) { LOGE("Shader not found: %s", spv_path); return false; }

    VkShaderModuleCreateInfo smInfo = {};
    smInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smInfo.codeSize = code.size() * 4; smInfo.pCode = code.data();
    if (vkCreateShaderModule(ctx.device, &smInfo, nullptr, &sp.shader) != VK_SUCCESS) return false;

    std::vector<VkDescriptorSetLayoutBinding> binds(nBindings);
    for (uint32_t i = 0; i < nBindings; i++) {
        binds[i].binding = i;
        binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo dslInfo = {};
    dslInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslInfo.bindingCount = nBindings; dslInfo.pBindings = binds.data();
    if (vkCreateDescriptorSetLayout(ctx.device, &dslInfo, nullptr, &sp.dsl) != VK_SUCCESS) return false;

    VkPipelineLayoutCreateInfo plInfo = {};
    plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plInfo.setLayoutCount = 1; plInfo.pSetLayouts = &sp.dsl;
    VkPushConstantRange pcRange = { VK_SHADER_STAGE_COMPUTE_BIT, 0, (uint32_t)pushSize };
    if (pushSize > 0) { plInfo.pushConstantRangeCount = 1; plInfo.pPushConstantRanges = &pcRange; }
    if (vkCreatePipelineLayout(ctx.device, &plInfo, nullptr, &sp.layout) != VK_SUCCESS) return false;

    VkComputePipelineCreateInfo cpInfo = {};
    cpInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpInfo.stage.module = sp.shader;
    cpInfo.stage.pName = "main";
    cpInfo.layout = sp.layout;
    if (vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &cpInfo, nullptr, &sp.pipeline) != VK_SUCCESS)
        return false;
    return true;
}

// ============================================================
// Vulkan init
// ============================================================
static bool init_vulkan(VKCtx& ctx) {
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "DiT_VKv2"; appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo instInfo = {};
    instInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instInfo.pApplicationInfo = &appInfo;
    if (vkCreateInstance(&instInfo, nullptr, &ctx.instance) != VK_SUCCESS) {
        LOGE("vkCreateInstance failed"); return false;
    }

    uint32_t devCount = 0;
    vkEnumeratePhysicalDevices(ctx.instance, &devCount, nullptr);
    std::vector<VkPhysicalDevice> devs(devCount);
    vkEnumeratePhysicalDevices(ctx.instance, &devCount, devs.data());
    for (auto d : devs) {
        VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(d, &props);
        LOGI("GPU: %s", props.deviceName);
        uint32_t qfCount; vkGetPhysicalDeviceQueueFamilyProperties(d, &qfCount, nullptr);
        std::vector<VkQueueFamilyProperties> qfProps(qfCount);
        vkGetPhysicalDeviceQueueFamilyProperties(d, &qfCount, qfProps.data());
        for (uint32_t i = 0; i < qfCount; i++)
            if (qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
                { ctx.physDev = d; ctx.qFamily = i; break; }
        if (ctx.physDev) break;
    }
    if (!ctx.physDev) { LOGE("No compute GPU"); return false; }
    vkGetPhysicalDeviceMemoryProperties(ctx.physDev, &ctx.memProps);

    float priority = 1.0f;
    VkDeviceQueueCreateInfo qInfo = {};
    qInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qInfo.queueFamilyIndex = ctx.qFamily; qInfo.queueCount = 1; qInfo.pQueuePriorities = &priority;
    VkDeviceCreateInfo devInfo = {};
    devInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    devInfo.queueCreateInfoCount = 1; devInfo.pQueueCreateInfos = &qInfo;
    if (vkCreateDevice(ctx.physDev, &devInfo, nullptr, &ctx.device) != VK_SUCCESS) {
        LOGE("vkCreateDevice failed"); return false;
    }
    vkGetDeviceQueue(ctx.device, ctx.qFamily, 0, &ctx.queue);

    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = ctx.qFamily;
    if (vkCreateCommandPool(ctx.device, &poolInfo, nullptr, &ctx.cmdPool) != VK_SUCCESS) return false;

    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (vkCreateFence(ctx.device, &fenceInfo, nullptr, &ctx.fence) != VK_SUCCESS) return false;
    if (vkCreateFence(ctx.device, &fenceInfo, nullptr, &ctx.stepFence) != VK_SUCCESS) return false;

    LOGI("Vulkan init OK");
    return true;
}

// ============================================================
// Pipelines — all fp32 shaders
// ============================================================
static bool create_all_pipelines(VKCtx& ctx, const char* spv_dir) {
    char p[256];
    #define CP(name, bindings, pushSize) \
        snprintf(p, sizeof(p), "%s/%s.spv", spv_dir, #name); \
        if (!create_pipe(ctx, p, bindings, pushSize, ctx.name)) { \
            LOGE("Pipeline %s failed", #name); return false; \
        }
    CP(gemm_bf16, 3, sizeof(PC_Gemm));
    CP(layernorm_fp32, 2, sizeof(PC_LayerNorm));
    CP(rms_norm_fp32, 3, sizeof(PC_RmsNorm));
    CP(silu_fp32, 2, sizeof(PC_Element));
    CP(gelu_fp32, 2, sizeof(PC_Element));
    CP(scale_shift_fp32, 4, sizeof(PC_ScaleShift));
    CP(rope_fp32, 3, sizeof(PC_Rope));
    CP(broadcast_fp32, 2, sizeof(PC_Broadcast));
    CP(attn_qkt_fp32, 3, sizeof(PC_AttnQKT));
    CP(attn_softmax_fp32, 1, sizeof(PC_AttnSoftmax));
    CP(attn_out_fp32, 3, sizeof(PC_AttnOut));
    CP(gate_fp32, 4, sizeof(PC_Gate));
    #undef CP
    LOGI("All 12 fp32 pipelines created");
    return true;
}

static bool create_descriptor_pools(VKCtx& ctx) {
    VkDescriptorPoolSize ps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 12000 };
    VkDescriptorPoolCreateInfo dpInfo = {};
    dpInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpInfo.maxSets = 3000; dpInfo.poolSizeCount = 1; dpInfo.pPoolSizes = &ps;
    if (vkCreateDescriptorPool(ctx.device, &dpInfo, nullptr, &ctx.descPool) != VK_SUCCESS) return false;
    if (vkCreateDescriptorPool(ctx.device, &dpInfo, nullptr, &ctx.stepPool) != VK_SUCCESS) return false;
    LOGI("Descriptor pools created");
    return true;
}

// ============================================================
// Safetensors weight loading — BF16 raw → Vulkan buffer
// ============================================================
static bool load_weights_safetensors(VKCtx& ctx, const char* sf_path,
                                      std::unordered_map<std::string, Buffer>& weights) {
    SafetensorsReader reader;
    if (!reader.open(sf_path)) { LOGE("Cannot open safetensors: %s", sf_path); return false; }

    std::string prefix = reader.detect_prefix();
    if (!prefix.empty()) LOGI("Detected prefix: '%s' — will strip", prefix.c_str());

    auto strip = [&](const std::string& k) -> std::string {
        return (!prefix.empty() && k.compare(0, prefix.size(), prefix) == 0)
               ? k.substr(prefix.size()) : k;
    };

    auto& keys = reader.keys();
    size_t n_loaded = 0, total_bytes = 0;
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    for (auto& key : keys) {
        if (key == "__metadata__") continue;
        std::string clean = strip(key);
        auto& info = reader.info(key);
        if (info.shape.empty()) continue;

        auto& w = weights[clean];
        w.size = info.data_len;
        if (!create_buf(ctx, w.size, usage, w)) {
            LOGE("Buffer alloc failed for %s (%.1f MB)", clean.c_str(), (double)w.size / 1e6);
            reader.close(); return false;
        }

        if (!reader.read_tensor(key, w.mapped, w.size)) {
            LOGE("Failed to read tensor %s", clean.c_str());
            reader.close(); return false;
        }

        total_bytes += w.size;
        n_loaded++;
    }

    reader.close();
    LOGI("Loaded %zu weight tensors (%.1f MB total) from safetensors — BF16 raw stored",
         n_loaded, (double)total_bytes / 1e6);
    return true;
}

// ============================================================
// Recording context — dispatch primitives
// ============================================================
struct RC {
    VKCtx* vk;
    VkCommandBuffer cmd;
    std::unordered_map<std::string, Buffer>* weights;

    void barrier() {
        VkMemoryBarrier b = {};
        b.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
    }

    void barrier_buf(VkBuffer buf) {
        VkBufferMemoryBarrier b = {};
        b.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        b.buffer = buf; b.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &b, 0, nullptr);
    }

    VkDescriptorSet alloc_set(VkDescriptorSetLayout dsl) {
        VkDescriptorSetAllocateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        info.descriptorPool = vk->descPool; info.descriptorSetCount = 1; info.pSetLayouts = &dsl;
        VkDescriptorSet ds = VK_NULL_HANDLE;
        VkResult res = vkAllocateDescriptorSets(vk->device, &info, &ds);
        if (res != VK_SUCCESS) {
            LOGE("alloc_set FAILED: VkResult=%d (pool exhausted or fragmented)", (int)res);
        }
        return ds;
    }

    void bind_buf(VkDescriptorSet ds, uint32_t binding, Buffer& buf, size_t off, size_t len) {
        VkDescriptorBufferInfo bi = { buf.buf, off, len };
        VkWriteDescriptorSet w = {};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = ds; w.dstBinding = binding; w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.pBufferInfo = &bi;
        vkUpdateDescriptorSets(vk->device, 1, &w, 0, nullptr);
    }

    // ── Dispatch methods ──

    void gemm(Buffer& A, const char* wname, uint32_t Mv, uint32_t Nv, uint32_t Kv, Buffer& C) {
        auto it = weights->find(wname);
        if (it == weights->end()) { LOGE("Weight not found: %s", wname); return; }
        auto& sp = vk->gemm_bf16;
        auto ds = alloc_set(sp.dsl);
        bind_buf(ds, 0, A, 0, VK_WHOLE_SIZE);
        bind_buf(ds, 1, it->second, 0, VK_WHOLE_SIZE);
        bind_buf(ds, 2, C, 0, VK_WHOLE_SIZE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sp.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sp.layout, 0, 1, &ds, 0, nullptr);
        PC_Gemm pc = { Mv, Nv, Kv, 1, 1.0f };
        vkCmdPushConstants(cmd, sp.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, (Nv + 7) / 8, (Mv + 7) / 8, 1);
        barrier();
    }

    void gemm_sub(Buffer& A, const char* wname, uint32_t Mv, uint32_t Nv, uint32_t Kv,
                   Buffer& C, size_t wSubOff) {
        auto it = weights->find(wname);
        if (it == weights->end()) { LOGE("Weight not found: %s", wname); return; }
        auto& sp = vk->gemm_bf16;
        auto ds = alloc_set(sp.dsl);
        bind_buf(ds, 0, A, 0, VK_WHOLE_SIZE);
        bind_buf(ds, 1, it->second, wSubOff, VK_WHOLE_SIZE);
        bind_buf(ds, 2, C, 0, VK_WHOLE_SIZE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sp.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sp.layout, 0, 1, &ds, 0, nullptr);
        PC_Gemm pc = { Mv, Nv, Kv, 1, 1.0f };
        vkCmdPushConstants(cmd, sp.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, (Nv + 7) / 8, (Mv + 7) / 8, 1);
        barrier();
    }

    void layernorm(Buffer& in, Buffer& out, uint32_t rows, uint32_t elems, float eps) {
        auto& sp = vk->layernorm_fp32;
        auto ds = alloc_set(sp.dsl);
        bind_buf(ds, 0, in, 0, VK_WHOLE_SIZE); bind_buf(ds, 1, out, 0, VK_WHOLE_SIZE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sp.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sp.layout, 0, 1, &ds, 0, nullptr);
        PC_LayerNorm pc = { rows, elems, eps };
        vkCmdPushConstants(cmd, sp.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, rows, 1, 1);
        barrier();
    }

    void rmsnorm(Buffer& in, const char* wname, Buffer& out, uint32_t rows, uint32_t elems, float eps) {
        auto it = weights->find(wname);
        if (it == weights->end()) { LOGE("Weight not found: %s", wname); return; }
        auto& sp = vk->rms_norm_fp32;
        auto ds = alloc_set(sp.dsl);
        bind_buf(ds, 0, in, 0, VK_WHOLE_SIZE);
        bind_buf(ds, 1, it->second, 0, VK_WHOLE_SIZE);
        bind_buf(ds, 2, out, 0, VK_WHOLE_SIZE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sp.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sp.layout, 0, 1, &ds, 0, nullptr);
        PC_RmsNorm pc = { rows, elems, eps };
        vkCmdPushConstants(cmd, sp.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, rows, 1, 1);
        barrier();
    }

    void silu(Buffer& in, Buffer& out, uint32_t n) {
        auto& sp = vk->silu_fp32;
        auto ds = alloc_set(sp.dsl);
        bind_buf(ds, 0, in, 0, VK_WHOLE_SIZE); bind_buf(ds, 1, out, 0, VK_WHOLE_SIZE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sp.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sp.layout, 0, 1, &ds, 0, nullptr);
        PC_Element pc = { n };
        vkCmdPushConstants(cmd, sp.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, (n + 255) / 256, 1, 1);
        barrier();
    }

    void gelu(Buffer& in, Buffer& out, uint32_t n) {
        auto& sp = vk->gelu_fp32;
        auto ds = alloc_set(sp.dsl);
        bind_buf(ds, 0, in, 0, VK_WHOLE_SIZE); bind_buf(ds, 1, out, 0, VK_WHOLE_SIZE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sp.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sp.layout, 0, 1, &ds, 0, nullptr);
        PC_Element pc = { n };
        vkCmdPushConstants(cmd, sp.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, (n + 255) / 256, 1, 1);
        barrier();
    }

    void scale_shift(Buffer& x, Buffer& scl, Buffer& sft, Buffer& out,
                     uint32_t n, uint32_t sS, uint32_t fS) {
        auto& sp = vk->scale_shift_fp32;
        auto ds = alloc_set(sp.dsl);
        bind_buf(ds, 0, x, 0, VK_WHOLE_SIZE); bind_buf(ds, 1, scl, 0, VK_WHOLE_SIZE);
        bind_buf(ds, 2, sft, 0, VK_WHOLE_SIZE); bind_buf(ds, 3, out, 0, VK_WHOLE_SIZE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sp.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sp.layout, 0, 1, &ds, 0, nullptr);
        PC_ScaleShift pc = { n, sS, fS };
        vkCmdPushConstants(cmd, sp.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, (n + 255) / 256, 1, 1);
        barrier();
    }

    void scale_shift_off(Buffer& x, Buffer& scl, size_t sclOff, Buffer& sft, size_t sftOff,
                          Buffer& out, uint32_t n, uint32_t sS, uint32_t fS) {
        auto& sp = vk->scale_shift_fp32;
        auto ds = alloc_set(sp.dsl);
        bind_buf(ds, 0, x, 0, VK_WHOLE_SIZE);
        bind_buf(ds, 1, scl, sclOff, VK_WHOLE_SIZE);
        bind_buf(ds, 2, sft, sftOff, VK_WHOLE_SIZE);
        bind_buf(ds, 3, out, 0, VK_WHOLE_SIZE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sp.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sp.layout, 0, 1, &ds, 0, nullptr);
        PC_ScaleShift pc = { n, sS, fS };
        vkCmdPushConstants(cmd, sp.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, (n + 255) / 256, 1, 1);
        barrier();
    }

    void rope(Buffer& t, Buffer& freqs, Buffer& out, uint32_t Nv, uint32_t hd) {
        auto& sp = vk->rope_fp32;
        auto ds = alloc_set(sp.dsl);
        bind_buf(ds, 0, t, 0, VK_WHOLE_SIZE); bind_buf(ds, 1, freqs, 0, VK_WHOLE_SIZE);
        bind_buf(ds, 2, out, 0, VK_WHOLE_SIZE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sp.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sp.layout, 0, 1, &ds, 0, nullptr);
        PC_Rope pc = { Nv, hd };
        vkCmdPushConstants(cmd, sp.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, (Nv + 255) / 256, 1, 1);
        barrier();
    }

    void broadcast_off(Buffer& in, Buffer& out, size_t outByteOff,
                        uint32_t Mv, uint32_t Dv, uint32_t rpt) {
        auto& sp = vk->broadcast_fp32;
        auto ds = alloc_set(sp.dsl);
        bind_buf(ds, 0, in, 0, VK_WHOLE_SIZE);
        bind_buf(ds, 1, out, outByteOff, VK_WHOLE_SIZE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sp.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sp.layout, 0, 1, &ds, 0, nullptr);
        PC_Broadcast pc = { Mv, Dv, rpt };
        vkCmdPushConstants(cmd, sp.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, (Mv * rpt * Dv + 255) / 256, 1, 1);
        barrier();
    }

    void record_attn_3pass(Buffer& Q, Buffer& K, Buffer& V, Buffer& A, Buffer& O,
                            uint32_t M_q, uint32_t M_kv, uint32_t H, uint32_t Dv, float scale,
                            size_t q_base_off = 0, size_t kv_base_off = 0,
                            size_t a_base_off = 0, size_t o_base_off = 0) {
        uint32_t batch_q = (M_q > 128) ? 128 : M_q;
        uint32_t n_batches = (M_q + batch_q - 1) / batch_q;
        size_t kv_row_bytes = (size_t)M_kv * H * Dv * 4;  // fp32

        for (uint32_t batch = 0; batch < n_batches; batch++) {
            uint32_t q_start = batch * batch_q;
            uint32_t this_q = (q_start + batch_q <= M_q) ? batch_q : (M_q - q_start);
            size_t qOff = q_base_off + (size_t)q_start * H * Dv * 4;
            size_t aOff = a_base_off + (size_t)q_start * H * M_kv * 4;
            size_t oOff = o_base_off + (size_t)q_start * H * Dv * 4;
            size_t qBytes = (size_t)this_q * H * Dv * 4;
            size_t aBytes = (size_t)this_q * H * M_kv * 4;
            size_t oBytes = (size_t)this_q * H * Dv * 4;

            // QK^T
            { auto ds = alloc_set(vk->attn_qkt_fp32.dsl);
              bind_buf(ds, 0, Q, qOff, qBytes); bind_buf(ds, 1, K, kv_base_off, kv_row_bytes);
              bind_buf(ds, 2, A, aOff, aBytes);
              vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk->attn_qkt_fp32.pipeline);
              vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk->attn_qkt_fp32.layout, 0, 1, &ds, 0, nullptr);
              PC_AttnQKT pc1 = { this_q, M_kv, H, Dv, scale };
              vkCmdPushConstants(cmd, vk->attn_qkt_fp32.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc1), &pc1);
              vkCmdDispatch(cmd, this_q * H, 1, 1); barrier(); }

            // softmax
            { auto ds = alloc_set(vk->attn_softmax_fp32.dsl);
              bind_buf(ds, 0, A, aOff, aBytes);
              vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk->attn_softmax_fp32.pipeline);
              vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk->attn_softmax_fp32.layout, 0, 1, &ds, 0, nullptr);
              PC_AttnSoftmax pc2 = { this_q, M_kv, H };
              vkCmdPushConstants(cmd, vk->attn_softmax_fp32.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc2), &pc2);
              vkCmdDispatch(cmd, this_q * H, 1, 1); barrier(); }

            // AV
            { auto ds = alloc_set(vk->attn_out_fp32.dsl);
              bind_buf(ds, 0, A, aOff, aBytes); bind_buf(ds, 1, V, kv_base_off, kv_row_bytes);
              bind_buf(ds, 2, O, oOff, oBytes);
              vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk->attn_out_fp32.pipeline);
              vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk->attn_out_fp32.layout, 0, 1, &ds, 0, nullptr);
              PC_AttnOut pc3 = { this_q, M_kv, H, Dv };
              vkCmdPushConstants(cmd, vk->attn_out_fp32.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc3), &pc3);
              vkCmdDispatch(cmd, this_q * H, 1, 1);
              if (batch < n_batches - 1) barrier(); }
        }
    }

    void gate_residual(Buffer& inBuf, Buffer& gateBuf, size_t gateOff,
                       Buffer& oproj, Buffer& out, uint32_t n) {
        auto& sp = vk->gate_fp32;
        auto ds = alloc_set(sp.dsl);
        bind_buf(ds, 0, oproj, 0, VK_WHOLE_SIZE);          // O_proj (fp32)
        bind_buf(ds, 1, gateBuf, gateOff, VK_WHOLE_SIZE);   // gate (fp32)
        bind_buf(ds, 2, inBuf, 0, VK_WHOLE_SIZE);           // residual (fp32)
        bind_buf(ds, 3, out, 0, VK_WHOLE_SIZE);             // output (fp32)
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sp.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sp.layout, 0, 1, &ds, 0, nullptr);
        PC_Gate pc = { n };
        vkCmdPushConstants(cmd, sp.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, (n + 255) / 256, 1, 1);
        barrier();
    }

    // ── GPU-side AdaLN: SiLU → LoRA down → LoRA up ×3 → scale+1 → broadcast ──
    // bcBuf layout: [scale+1, shift, gate] per module at slots (base, base+1, base+2)
    // base_comp: 0=self, 3=cross, 6=mlp
    // W2 is [3*D, lora_dim] BF16 — 3 components at offsets 0, W2_COMP, 2*W2_COMP
    // loraBuf: [3, M, D] fp32 — external lora from t_embedder
    void adaln_gpu(const char* w0_name, const char* w2_name, int base_comp,
                    Buffer& tEmb, Buffer& aBuf, Buffer& t1,
                    Buffer& tQ, Buffer& tK, Buffer& tV,
                    Buffer& bcBuf, Buffer& onesBuf, Buffer& loraBuf) {
        static const size_t W2C = (size_t)D * ADALN_LORA_DIM * 2;  // BF16 bytes per W2 component
        auto boff = [](int c) -> size_t { return (size_t)c * MS * D * 4; };
        size_t loraComp = (size_t)M * D * 4;

        // 1. SiLU(t_emb) → aBuf
        silu(tEmb, aBuf, M * D);
        barrier_buf(aBuf.buf);

        // 2. LoRA down: aBuf @ W0^T → t1 [M, 256]
        gemm(aBuf, w0_name, M, ADALN_LORA_DIM, D, t1);
        barrier_buf(t1.buf);

        // 3-5. LoRA up ×3 with weight sub-offsets
        gemm_sub(t1, w2_name, M, D, ADALN_LORA_DIM, tQ, 0);          // shift
        gemm_sub(t1, w2_name, M, D, ADALN_LORA_DIM, tK, W2C);        // scale
        gemm_sub(t1, w2_name, M, D, ADALN_LORA_DIM, tV, W2C * 2);    // gate

        // Add external lora: tQ += lora[0], tK += lora[1], tV += lora[2]
        // loraBuf layout: [3, M, D] = M*D floats per component
        scale_shift_off(tQ, onesBuf, 0, loraBuf, 0,           tQ, M * D, 0, 1);
        scale_shift_off(tK, onesBuf, 0, loraBuf, loraComp,     tK, M * D, 0, 1);
        scale_shift_off(tV, onesBuf, 0, loraBuf, loraComp * 2, tV, M * D, 0, 1);
        barrier_buf(tQ.buf); barrier_buf(tK.buf); barrier_buf(tV.buf);

        // 6. scale+1: aBuf = tK + 1.0
        scale_shift_off(tK, onesBuf, 0, onesBuf, 0, aBuf, M * D, 0, 0);

        // 7-9. Broadcast to bcBuf: [scale+1, shift, gate] at (base, base+1, base+2)
        broadcast_off(tQ, bcBuf, boff(base_comp + 1), M, D, S);   // shift
        broadcast_off(aBuf, bcBuf, boff(base_comp + 0), M, D, S);  // scale+1
        broadcast_off(tV, bcBuf, boff(base_comp + 2), M, D, S);   // gate
    }

    // AdaLN split into sub-segments to work around Adreno barrier bug.
    // Adreno ignores VkMemoryBarrier within same cmd buffer for compute dispatches.
    // Each sub-segment gets its own begin→record→end→submit+wait cycle.
    void adaln_gpu_split(const char* w0_name, const char* w2_name, int base_comp,
                          Buffer& tEmb, Buffer& aBuf, Buffer& t1,
                          Buffer& tQ, Buffer& tK, Buffer& tV,
                          Buffer& bcBuf, Buffer& onesBuf, Buffer& loraBuf) {
        static const size_t W2C = (size_t)D * ADALN_LORA_DIM * 2;
        auto boff = [](int c) -> size_t { return (size_t)c * MS * D * 4; };
        size_t loraComp = (size_t)M * D * 4;

        // ── SubA1: SiLU + LoRA down ──
        silu(tEmb, aBuf, M * D);
        gemm(aBuf, w0_name, M, ADALN_LORA_DIM, D, t1);

        // ── SubA2: LoRA up ×3 ──
        gemm_sub(t1, w2_name, M, D, ADALN_LORA_DIM, tQ, 0);          // shift
        gemm_sub(t1, w2_name, M, D, ADALN_LORA_DIM, tK, W2C);        // scale
        gemm_sub(t1, w2_name, M, D, ADALN_LORA_DIM, tV, W2C * 2);    // gate

        // ── SubA3: add lora + scale+1 ──
        scale_shift_off(tQ, onesBuf, 0, loraBuf, 0,           tQ, M * D, 0, 1);
        scale_shift_off(tK, onesBuf, 0, loraBuf, loraComp,     tK, M * D, 0, 1);
        scale_shift_off(tV, onesBuf, 0, loraBuf, loraComp * 2, tV, M * D, 0, 1);
        // scale+1: aBuf = tK + 1.0
        scale_shift_off(tK, onesBuf, 0, onesBuf, 0, aBuf, M * D, 0, 0);

        // ── SubA4: broadcast to bcBuf ──
        broadcast_off(tQ, bcBuf, boff(base_comp + 1), M, D, S);   // shift
        broadcast_off(aBuf, bcBuf, boff(base_comp + 0), M, D, S);  // scale+1
        broadcast_off(tV, bcBuf, boff(base_comp + 2), M, D, S);   // gate
    }
};

// ============================================================
// Global state
// ============================================================
static VKCtx g_vk;
static VkCommandBuffer g_segCmdBuf = VK_NULL_HANDLE;  // per-segment recording
static Buffer g_xBuf, g_tEmbBuf, g_ctxBuf, g_outBuf;
static Buffer g_t1, g_tQ, g_tK, g_tV, g_tO, g_rBuf, g_aBuf, g_nBuf, g_gBuf, g_bcBuf;
static Buffer g_attnA, g_attnO;
static Buffer g_onesBuf, g_loraBuf, g_ropeFreqsBuf;
static std::unordered_map<std::string, Buffer> g_weights;
static bool g_init = false;

// Block 0 capture (host memory)
static float *g_b0_x=nullptr, *g_b0_ctx=nullptr, *g_b0_temb=nullptr, *g_b0_lora=nullptr;
static float *g_b0_ln=nullptr, *g_b0_mod=nullptr, *g_b0_q=nullptr, *g_b0_k=nullptr, *g_b0_v=nullptr;
static float *g_b0_qn=nullptr, *g_b0_kn=nullptr, *g_b0_qr=nullptr, *g_b0_kr=nullptr;
static float *g_b0_scores=nullptr, *g_b0_attn_o=nullptr, *g_b0_oproj=nullptr;
static float *g_b0_fc1=nullptr, *g_b0_fc2=nullptr;
static float *g_b0_sa=nullptr, *g_b0_cx=nullptr, *g_b0_mlp=nullptr;
static float *g_b0_nbuf=nullptr;
static float *g_b0_bcbuf=nullptr;  // AdaLN modulation buffer

// RoPE host-side frequencies
static std::vector<float> g_ropeFreqsHost;
static size_t g_ropeFreqsSize = 0;

// ============================================================
// Per-step segmented recording helpers
// ============================================================
static VkDescriptorPool g_savedPool = VK_NULL_HANDLE;

static bool submit_segment(void) {
    vkResetFences(g_vk.device, 1, &g_vk.stepFence);
    VkSubmitInfo si = {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &g_segCmdBuf;
    VkResult sr = vkQueueSubmit(g_vk.queue, 1, &si, g_vk.stepFence);
    if (sr != VK_SUCCESS) {
        LOGE("submit_segment: vkQueueSubmit FAILED VkResult=%d", (int)sr);
        return false;
    }
    VkResult wr = vkWaitForFences(g_vk.device, 1, &g_vk.stepFence, VK_TRUE, UINT64_MAX);
    if (wr != VK_SUCCESS) {
        LOGE("submit_segment: vkWaitForFences FAILED VkResult=%d", (int)wr);
        return false;
    }
    return true;
}

static void begin_segment(RC& rc) {
    vkResetDescriptorPool(g_vk.device, g_vk.stepPool, 0);
    g_savedPool = g_vk.descPool;
    g_vk.descPool = g_vk.stepPool;
    VkCommandBufferBeginInfo bi = {};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(g_segCmdBuf, &bi);
    rc.cmd = g_segCmdBuf;
}

static void end_segment(void) {
    vkEndCommandBuffer(g_segCmdBuf);
    g_vk.descPool = g_savedPool;
}

// ============================================================
// TDR-safe segmentation: 4 segments per block (each < 40 dispatches)
// Segment A: AdaLN all 3 modules (36 dispatches)
// Segment B: Self-attn full chain (17 dispatches)
// Segment C: Cross-attn full chain (20 dispatches)
// Segment D: MLP (6 dispatches)
// ============================================================

// ── Segment A: AdaLN all 3 modules → bcBuf ──
// Each module is split into 4 sub-segments (3-4 dispatches each)
// to work around Adreno ignoring barriers within a single cmd buffer.
static void seg_adaln(RC& rc, int b) {
    char s0[128],s2[128],c0[128],c2[128],m0[128],m2[128];
    snprintf(s0,sizeof(s0),"blocks.%d.adaln_modulation_self_attn.1.weight",b);
    snprintf(s2,sizeof(s2),"blocks.%d.adaln_modulation_self_attn.2.weight",b);
    snprintf(c0,sizeof(c0),"blocks.%d.adaln_modulation_cross_attn.1.weight",b);
    snprintf(c2,sizeof(c2),"blocks.%d.adaln_modulation_cross_attn.2.weight",b);
    snprintf(m0,sizeof(m0),"blocks.%d.adaln_modulation_mlp.1.weight",b);
    snprintf(m2,sizeof(m2),"blocks.%d.adaln_modulation_mlp.2.weight",b);

    auto boff = [](int c) -> size_t { return (size_t)c * MS * D * 4; };
    size_t loraComp = (size_t)M * D * 4;
    static const size_t W2C = (size_t)D * ADALN_LORA_DIM * 2;

    // Helper: run adaln for one module in 4 sub-segments
    auto run_adaln_module = [&](const char* w0n, const char* w2n, int base) {
        // SubA1: SiLU + LoRA down (2 dispatches)
        begin_segment(rc);
        rc.silu(g_tEmbBuf, g_aBuf, M * D);
        rc.gemm(g_aBuf, w0n, M, ADALN_LORA_DIM, D, g_t1);
        end_segment(); submit_segment();

        // SubA2: LoRA up ×3 (3 dispatches)
        begin_segment(rc);
        rc.gemm_sub(g_t1, w2n, M, D, ADALN_LORA_DIM, g_tQ, 0);
        rc.gemm_sub(g_t1, w2n, M, D, ADALN_LORA_DIM, g_tK, W2C);
        rc.gemm_sub(g_t1, w2n, M, D, ADALN_LORA_DIM, g_tV, W2C * 2);
        end_segment(); submit_segment();

        // SubA3: Add lora + scale+1 (4 dispatches)
        begin_segment(rc);
        rc.scale_shift_off(g_tQ, g_onesBuf, 0, g_loraBuf, 0,           g_tQ, M*D, 0, 1);
        rc.scale_shift_off(g_tK, g_onesBuf, 0, g_loraBuf, loraComp,     g_tK, M*D, 0, 1);
        rc.scale_shift_off(g_tV, g_onesBuf, 0, g_loraBuf, loraComp * 2, g_tV, M*D, 0, 1);
        rc.scale_shift_off(g_tK, g_onesBuf, 0, g_onesBuf, 0, g_aBuf, M*D, 0, 0);
        end_segment(); submit_segment();

        // SubA4: Broadcast to bcBuf (3 dispatches)
        begin_segment(rc);
        rc.broadcast_off(g_tQ,   g_bcBuf, boff(base + 1), M, D, S);
        rc.broadcast_off(g_aBuf, g_bcBuf, boff(base + 0), M, D, S);
        rc.broadcast_off(g_tV,   g_bcBuf, boff(base + 2), M, D, S);
        end_segment(); submit_segment();
    };

    run_adaln_module(s0, s2, 0);  // SA
    run_adaln_module(c0, c2, 3);  // CX
    run_adaln_module(m0, m2, 6);  // MLP
}

// ── Debug: Block 0 AdaLN split into 3 independent segments (test barrier hypothesis) ──
static void seg_adaln_split(RC& rc) {
    // Each adaln_gpu call gets its own cmd buffer + submit + fence wait
    // If this fixes bcBuf, barrier within single cmd buffer is the root cause.
    begin_segment(rc);
    rc.adaln_gpu("blocks.0.adaln_modulation_self_attn.1.weight",
                 "blocks.0.adaln_modulation_self_attn.2.weight",
                 0, g_tEmbBuf, g_aBuf, g_t1, g_tQ, g_tK, g_tV,
                 g_bcBuf, g_onesBuf, g_loraBuf);
    end_segment(); submit_segment();

    begin_segment(rc);
    rc.adaln_gpu("blocks.0.adaln_modulation_cross_attn.1.weight",
                 "blocks.0.adaln_modulation_cross_attn.2.weight",
                 3, g_tEmbBuf, g_aBuf, g_t1, g_tQ, g_tK, g_tV,
                 g_bcBuf, g_onesBuf, g_loraBuf);
    end_segment(); submit_segment();

    begin_segment(rc);
    rc.adaln_gpu("blocks.0.adaln_modulation_mlp.1.weight",
                 "blocks.0.adaln_modulation_mlp.2.weight",
                 6, g_tEmbBuf, g_aBuf, g_t1, g_tQ, g_tK, g_tV,
                 g_bcBuf, g_onesBuf, g_loraBuf);
    end_segment(); submit_segment();
}

// ── Segment B: Self-attn LN→AdaLN→QKV→RMSNorm→RoPE→attn→O_proj→gate ──
static void seg_self_attn(RC& rc, int b, Buffer& inBuf) {
    auto boff = [](int c) -> size_t { return (size_t)c * MS * D * 4; };
    float scl = 1.0f / sqrtf((float)HEAD_DIM);
    uint32_t ph = MS * N_HEADS, S_per = MS / M;
    char qw[128],kw[128],vw[128],ow[128],qn[128],kn[128];
    snprintf(qw,sizeof(qw),"blocks.%d.self_attn.q_proj.weight",b);
    snprintf(kw,sizeof(kw),"blocks.%d.self_attn.k_proj.weight",b);
    snprintf(vw,sizeof(vw),"blocks.%d.self_attn.v_proj.weight",b);
    snprintf(ow,sizeof(ow),"blocks.%d.self_attn.output_proj.weight",b);
    snprintf(qn,sizeof(qn),"blocks.%d.self_attn.q_norm.weight",b);
    snprintf(kn,sizeof(kn),"blocks.%d.self_attn.k_norm.weight",b);

    begin_segment(rc);
    rc.layernorm(inBuf, g_nBuf, MS, D, 1e-6f);
    rc.scale_shift_off(g_nBuf, g_bcBuf, boff(0), g_bcBuf, boff(1), g_nBuf, MS*D, 1, 1);
    rc.gemm(g_nBuf, qw, MS, D, D, g_tQ);
    rc.gemm(g_nBuf, kw, MS, D, D, g_tK);
    rc.gemm(g_nBuf, vw, MS, D, D, g_tV);
    rc.rmsnorm(g_tQ, qn, g_tQ, ph, HEAD_DIM, 1e-6f);
    rc.rmsnorm(g_tK, kn, g_tK, ph, HEAD_DIM, 1e-6f);
    rc.rope(g_tQ, g_ropeFreqsBuf, g_rBuf, ph, HEAD_DIM);
    rc.rope(g_tK, g_ropeFreqsBuf, g_attnO, ph, HEAD_DIM);
    for (uint32_t mb = 0; mb < M; mb++) {
        size_t off = (size_t)mb * S_per * N_HEADS * HEAD_DIM * 4;
        rc.record_attn_3pass(g_rBuf, g_attnO, g_tV, g_attnA, g_attnO,
                             S_per, S_per, N_HEADS, HEAD_DIM, scl, off, off, off, off);
    }
    rc.gemm(g_attnO, ow, MS, D, D, g_tO);
    rc.gate_residual(inBuf, g_bcBuf, boff(2), g_tO, g_tV, MS * D);
    end_segment();
    submit_segment();
}

// ── Debug: Block 0 self-attn split into sub-segments for intermediate capture ──
static void seg_self_attn_debug(RC& rc, Buffer& inBuf) {
    auto boff = [](int c) -> size_t { return (size_t)c * MS * D * 4; };
    float scl = 1.0f / sqrtf((float)HEAD_DIM);
    uint32_t ph = MS * N_HEADS, S_per = MS / M;
    size_t qkv_bytes = (size_t)MS * N_HEADS * HEAD_DIM * 4;
    size_t d_bytes = (size_t)MS * D * 4;

    // ── Sub-segment B1a: LN only (1 dispatch) ──
    {   begin_segment(rc);
        rc.layernorm(inBuf, g_nBuf, MS, D, 1e-6f);
        end_segment(); submit_segment();
    }

    // ── Sub-segment B1b: scale_shift (LN→modulated) (1 dispatch) ──
    {   begin_segment(rc);
        rc.scale_shift_off(g_nBuf, g_bcBuf, boff(0), g_bcBuf, boff(1), g_nBuf, MS*D, 1, 1);
        end_segment(); submit_segment();
        if (g_b0_nbuf) memcpy(g_b0_nbuf, g_nBuf.mapped, d_bytes);  // capture modulated output
    }

    // ── Sub-segment B1c: QKV GEMM (3 dispatches) ──
    {   begin_segment(rc);
        rc.gemm(g_nBuf, "blocks.0.self_attn.q_proj.weight", MS, D, D, g_tQ);
        rc.gemm(g_nBuf, "blocks.0.self_attn.k_proj.weight", MS, D, D, g_tK);
        rc.gemm(g_nBuf, "blocks.0.self_attn.v_proj.weight", MS, D, D, g_tV);
        end_segment(); submit_segment();
        if (g_b0_q)  memcpy(g_b0_q,  g_tQ.mapped, qkv_bytes);
        if (g_b0_k)  memcpy(g_b0_k,  g_tK.mapped, qkv_bytes);
        if (g_b0_v)  memcpy(g_b0_v,  g_tV.mapped, qkv_bytes);
    }

    // ── Sub-segment B2: RMSNorm Q/K ──
    {   begin_segment(rc);
        rc.rmsnorm(g_tQ, "blocks.0.self_attn.q_norm.weight", g_tQ, ph, HEAD_DIM, 1e-6f);
        rc.rmsnorm(g_tK, "blocks.0.self_attn.k_norm.weight", g_tK, ph, HEAD_DIM, 1e-6f);
        end_segment(); submit_segment();
        // Capture Q, K after RMSNorm (g_tQ/g_tK modified in-place)
        if (g_b0_qn) memcpy(g_b0_qn, g_tQ.mapped, qkv_bytes);
        if (g_b0_kn) memcpy(g_b0_kn, g_tK.mapped, qkv_bytes);
    }

    // ── Sub-segment B3: RoPE ──
    {   begin_segment(rc);
        rc.rope(g_tQ, g_ropeFreqsBuf, g_rBuf, ph, HEAD_DIM);
        rc.rope(g_tK, g_ropeFreqsBuf, g_attnO, ph, HEAD_DIM);
        end_segment(); submit_segment();
        // Capture Q, K after RoPE
        if (g_b0_qr) memcpy(g_b0_qr, g_rBuf.mapped, qkv_bytes);
        if (g_b0_kr) memcpy(g_b0_kr, g_attnO.mapped, qkv_bytes);
    }

    // ── Sub-segment B4: Attention + O_proj + gate ──
    {   begin_segment(rc);
        // Self-attention: per-batch Q·K^T + softmax + AV
        for (uint32_t mb = 0; mb < M; mb++) {
            size_t off = (size_t)mb * S_per * N_HEADS * HEAD_DIM * 4;
            rc.record_attn_3pass(g_rBuf, g_attnO, g_tV, g_attnA, g_attnO,
                                 S_per, S_per, N_HEADS, HEAD_DIM, scl,
                                 off, off, off, off);
        }
        rc.gemm(g_attnO, "blocks.0.self_attn.output_proj.weight", MS, D, D, g_tO);
        rc.gate_residual(inBuf, g_bcBuf, boff(2), g_tO, g_tV, MS * D);
        end_segment(); submit_segment();
        // Capture attention output, O_proj, SA residual
        if (g_b0_attn_o) memcpy(g_b0_attn_o, g_attnO.mapped, qkv_bytes);
        if (g_b0_oproj)  memcpy(g_b0_oproj,  g_tO.mapped,    d_bytes);
    }
}

// ── Segment C: Cross-attn LN→AdaLN→QKV→RMSNorm→attn→O_proj→gate ──
static void seg_cross_attn(RC& rc, int b) {
    auto boff = [](int c) -> size_t { return (size_t)c * MS * D * 4; };
    float scl = 1.0f / sqrtf((float)HEAD_DIM);
    uint32_t ph = MS * N_HEADS, ph_cross = M * Nctx * N_HEADS, MS_kv = M * Nctx, S_per = MS / M;
    char qw[128],kw[128],vw[128],ow[128],qn[128],kn[128];
    snprintf(qw,sizeof(qw),"blocks.%d.cross_attn.q_proj.weight",b);
    snprintf(kw,sizeof(kw),"blocks.%d.cross_attn.k_proj.weight",b);
    snprintf(vw,sizeof(vw),"blocks.%d.cross_attn.v_proj.weight",b);
    snprintf(ow,sizeof(ow),"blocks.%d.cross_attn.output_proj.weight",b);
    snprintf(qn,sizeof(qn),"blocks.%d.cross_attn.q_norm.weight",b);
    snprintf(kn,sizeof(kn),"blocks.%d.cross_attn.k_norm.weight",b);

    begin_segment(rc);
    rc.layernorm(g_tV, g_nBuf, MS, D, 1e-6f);
    rc.scale_shift_off(g_nBuf, g_bcBuf, boff(3), g_bcBuf, boff(4), g_nBuf, MS*D, 1, 1);
    rc.gemm(g_nBuf, qw, MS, D, D, g_tQ);
    rc.gemm(g_ctxBuf, kw, MS_kv, D, CtxD, g_t1);
    rc.gemm(g_ctxBuf, vw, MS_kv, D, CtxD, g_rBuf);
    rc.rmsnorm(g_tQ, qn, g_tQ, ph, HEAD_DIM, 1e-6f);
    rc.rmsnorm(g_t1, kn, g_t1, ph_cross, HEAD_DIM, 1e-6f);
    for (uint32_t mb = 0; mb < M; mb++) {
        size_t q_off = (size_t)mb * S_per * N_HEADS * HEAD_DIM * 4;
        size_t kv_off = (size_t)mb * Nctx * N_HEADS * HEAD_DIM * 4;
        rc.record_attn_3pass(g_tQ, g_t1, g_rBuf, g_attnA, g_attnO,
                             S_per, Nctx, N_HEADS, HEAD_DIM, scl, q_off, kv_off, q_off, q_off);
    }
    rc.gemm(g_attnO, ow, MS, D, D, g_tO);
    rc.gate_residual(g_tV, g_bcBuf, boff(5), g_tO, g_gBuf, MS * D);
    end_segment();
    submit_segment();
}

// ── Segment D: MLP LN→AdaLN→fc1→GELU→fc2→gate ──
static void seg_mlp(RC& rc, int b) {
    auto boff = [](int c) -> size_t { return (size_t)c * MS * D * 4; };
    char l1w[128],l2w[128];
    snprintf(l1w,sizeof(l1w),"blocks.%d.mlp.layer1.weight",b);
    snprintf(l2w,sizeof(l2w),"blocks.%d.mlp.layer2.weight",b);

    begin_segment(rc);
    rc.layernorm(g_gBuf, g_nBuf, MS, D, 1e-6f);
    rc.scale_shift_off(g_nBuf, g_bcBuf, boff(6), g_bcBuf, boff(7), g_nBuf, MS*D, 1, 1);
    rc.gemm(g_nBuf, l1w, MS, MLP_HIDDEN, D, g_t1);
    rc.gelu(g_t1, g_t1, MS * MLP_HIDDEN);
    rc.gemm(g_t1, l2w, MS, D, MLP_HIDDEN, g_nBuf);
    rc.gate_residual(g_gBuf, g_bcBuf, boff(8), g_nBuf, g_outBuf, MS * D);
    end_segment();
    submit_segment();
}

// ── CPU AdaLN: compute bcBuf on CPU (bypasses Adreno multi-dispatch issues) ──
static void seg_adaln_cpu(int b) {
    auto boff = [](int c) -> size_t { return (size_t)c * MS * D * 4; };
    size_t loraComp = (size_t)M * D * 4;
    static const size_t W2C = (size_t)D * ADALN_LORA_DIM * 2;

    auto cpu_adaln_module = [&](const char* w0_key, const char* w2_key, int base) {
        // Read t_emb from GPU buffer (HOST_COHERENT — unified memory, no DMA needed)
        float emb[M*D], lora[M*D3];
        memcpy(emb,  g_tEmbBuf.mapped, M*D*4);
        memcpy(lora, g_loraBuf.mapped, M*D3*4);

        // 1. SiLU
        float aBuf[M*D];
        for (int i = 0; i < M*D; i++) aBuf[i] = emb[i] / (1.0f + expf(-emb[i]));

        // 2. LoRA down: aBuf @ W0^T → t1 [M, 256]
        auto it_w0 = g_weights.find(w0_key);
        float t1[M * ADALN_LORA_DIM];
        head_tail::cpu_gemm_bf16(M, ADALN_LORA_DIM, D, aBuf,
            (const uint16_t*)it_w0->second.mapped, t1);

        // 3. LoRA up: t1 @ W2^T → up [M, 3*D]
        auto it_w2 = g_weights.find(w2_key);
        float up[M * 3*D];
        head_tail::cpu_gemm_bf16(M, 3*D, ADALN_LORA_DIM, t1,
            (const uint16_t*)it_w2->second.mapped, up);

        // 4. Add lora
        for (int i = 0; i < M*3*D; i++) up[i] += lora[i];

        // 5. scale+1 for component 1
        float scale_plus1[M*D];
        for (int i = 0; i < M*D; i++) scale_plus1[i] = up[D + i] + 1.0f;

        // 6. Broadcast to bcBuf (CPU-side, write directly to mapped buffer)
        float* bc = (float*)g_bcBuf.mapped;
        for (int m = 0; m < M; m++) {
            for (int s = 0; s < (int)S; s++) {
                size_t dst_row = (size_t)(m*S + s) * D;
                size_t src_m = (size_t)m * D;
                memcpy(bc + boff(base+1)/4 + dst_row, up + m*3*D, D*4);              // shift
                memcpy(bc + boff(base+0)/4 + dst_row, scale_plus1 + src_m, D*4);  // scale+1
                memcpy(bc + boff(base+2)/4 + dst_row, up + 2*D + m*3*D, D*4);     // gate
            }
        }
    };

    char s0[128],s2[128],c0[128],c2[128],m0[128],m2[128];
    snprintf(s0,sizeof(s0),"blocks.%d.adaln_modulation_self_attn.1.weight",b);
    snprintf(s2,sizeof(s2),"blocks.%d.adaln_modulation_self_attn.2.weight",b);
    snprintf(c0,sizeof(c0),"blocks.%d.adaln_modulation_cross_attn.1.weight",b);
    snprintf(c2,sizeof(c2),"blocks.%d.adaln_modulation_cross_attn.2.weight",b);
    snprintf(m0,sizeof(m0),"blocks.%d.adaln_modulation_mlp.1.weight",b);
    snprintf(m2,sizeof(m2),"blocks.%d.adaln_modulation_mlp.2.weight",b);

    cpu_adaln_module(s0, s2, 0);  // SA
    cpu_adaln_module(c0, c2, 3);  // CX
    cpu_adaln_module(m0, m2, 6);  // MLP

    // Log first few bcBuf values for debug comparison
    float* bc = (float*)g_bcBuf.mapped;
    LOGI("CPU AdaLN bcBuf[0..4] (SA scale+1): %.4f %.4f %.4f %.4f",
         (double)bc[0], (double)bc[1], (double)bc[2], (double)bc[3]);
    LOGI("CPU AdaLN bcBuf[0..4] (SA shift):   %.4f %.4f %.4f %.4f",
         (double)bc[boff(1)/4], (double)bc[boff(1)/4+1],
         (double)bc[boff(1)/4+2], (double)bc[boff(1)/4+3]);
}

// ============================================================
// dit_forward_step — 28 blocks, per-step per-block segmented
// ============================================================
static bool dit_forward_step(void* x_data, void* ctx_data, void* out_data,
                              int _MS, int _D, int _M, int _Nctx, int _CtxD) {
    if (!g_init) return false;
    MS = (uint32_t)_MS; M = (uint32_t)_M; S = MS / M;

    // Upload inputs
    memcpy(g_xBuf.mapped, x_data, (size_t)MS * D * 4);    // x: fp32
    memcpy(g_ctxBuf.mapped, ctx_data, (size_t)_M * (size_t)_Nctx * (size_t)_CtxD * 4);

    // Upload RoPE freqs if needed
    if (!g_ropeFreqsHost.empty() && g_ropeFreqsBuf.buf == VK_NULL_HANDLE) {
        VkBufferUsageFlags u = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        if (!create_buf(g_vk, g_ropeFreqsSize * sizeof(float), u, g_ropeFreqsBuf)) {
            LOGE("RoPE freq buffer creation failed");
        } else {
            memcpy(g_ropeFreqsBuf.mapped, g_ropeFreqsHost.data(), g_ropeFreqsSize * sizeof(float));
        }
    }

    // 28 blocks, 4 segments each = 112 submits (TDR-safe: each < 40 dispatches)
    RC rc; rc.vk = &g_vk; rc.cmd = VK_NULL_HANDLE; rc.weights = &g_weights;

    for (int b = 0; b < 28; b++) {
        if (b == 0) {
            seg_adaln_split(rc);  // Test: each module in own cmd buffer
            if (g_b0_bcbuf)
                memcpy(g_b0_bcbuf, g_bcBuf.mapped, 9u * (size_t)MS * D * 4);
        } else {
            seg_adaln_cpu(b);
        }
        if (b == 0)
            seg_self_attn_debug(rc, g_xBuf);
        else
            seg_self_attn(rc, b, g_xBuf);
        if (b == 0 && g_b0_sa) memcpy(g_b0_sa, g_tV.mapped, (size_t)MS * D * 4);  // SA residual in g_tV

        seg_cross_attn(rc, b);
        if (b == 0 && g_b0_cx) memcpy(g_b0_cx, g_gBuf.mapped, (size_t)MS * D * 4); // CX residual in g_gBuf

        seg_mlp(rc, b);
        if (b == 0) {
            if (g_b0_mlp) memcpy(g_b0_mlp, g_outBuf.mapped, (size_t)MS * D * 4);
            if (g_b0_x)   memcpy(g_b0_x, g_xBuf.mapped, (size_t)MS * D * 4);  // original input
        }

        memcpy(g_xBuf.mapped, g_outBuf.mapped, (size_t)MS * D * 4);
    }

    // Clean up temp RoPE buffer
    if (g_ropeFreqsBuf.buf != VK_NULL_HANDLE) {
        vkDestroyBuffer(g_vk.device, g_ropeFreqsBuf.buf, nullptr);
        vkFreeMemory(g_vk.device, g_ropeFreqsBuf.mem, nullptr);
        g_ropeFreqsBuf.buf = VK_NULL_HANDLE; g_ropeFreqsBuf.mem = VK_NULL_HANDLE; g_ropeFreqsBuf.mapped = nullptr;
    }

    memcpy(out_data, g_outBuf.mapped, (size_t)MS * D * 4);
    LOGI("Forward step complete (%d blocks)", 28);
    return true;
}

// ============================================================
// RoPE frequency computation (replicates VideoRopePosition3DEmb)
// Ported from dit_engine.cpp, upgraded to fp32 output.
// ============================================================
static bool compute_rope_freqs(void) {
    uint32_t head_dim = HEAD_DIM;
    uint32_t T = 1;
    uint32_t dim_h = head_dim / 6 * 2;   // 42
    uint32_t dim_w = dim_h;               // 42
    uint32_t dim_t = head_dim - 2 * dim_h; // 44
    uint32_t half_dim = head_dim / 2;     // 64

    // NTK factors: extrapolation_ratio ^ (dim / (dim-2))
    float h_ntk = powf(4.0f, (float)dim_h / (float)(dim_h - 2));
    float w_ntk = powf(4.0f, (float)dim_w / (float)(dim_w - 2));
    float t_ntk = powf(1.0f, (float)dim_t / (float)(dim_t - 2));

    // Theta with NTK scaling applied INSIDE the frequency base
    float h_theta = 10000.0f * h_ntk;
    float w_theta = 10000.0f * w_ntk;
    float t_theta = 10000.0f * t_ntk;

    // Compute per-position freqs [S, half_dim, 4] in fp32
    // S=256 spatial positions (H_patches=16 × W_patches=16)
    std::vector<float> pos_freqs((size_t)S * half_dim * 4);

    for (uint32_t p = 0; p < S; p++) {
        uint32_t h_idx = p / (uint32_t)sqrtf((float)S);  // row in 16×16 grid
        uint32_t w_idx = p % (uint32_t)sqrtf((float)S);  // col
        uint32_t t_idx = 0;  // T=1 for image

        for (uint32_t j = 0; j < half_dim; j++) {
            float cos_val, sin_val;

            if (j < dim_t / 2) {
                // Temporal component (T=1 → angle=0 for all)
                float freq = 1.0f / powf(t_theta, (float)(2 * j) / (float)dim_t);
                float angle = (float)t_idx * freq;
                cos_val = cosf(angle); sin_val = sinf(angle);
            } else if (j < dim_t / 2 + dim_h / 2) {
                // Height component
                uint32_t jh = j - dim_t / 2;
                float freq = 1.0f / powf(h_theta, (float)(2 * jh) / (float)dim_h);
                float angle = (float)h_idx * freq;
                cos_val = cosf(angle); sin_val = sinf(angle);
            } else if (j < dim_t / 2 + dim_h / 2 + dim_w / 2) {
                // Width component
                uint32_t jw = j - dim_t / 2 - dim_h / 2;
                float freq = 1.0f / powf(w_theta, (float)(2 * jw) / (float)dim_w);
                float angle = (float)w_idx * freq;
                cos_val = cosf(angle); sin_val = sinf(angle);
            } else {
                // Remaining: all T→temporal (angle=0 for T=1)
                float freq = 1.0f / powf(t_theta, (float)(2 * (j - dim_t/2 - dim_h/2 - dim_w/2)) / (float)dim_t);
                float angle = (float)t_idx * freq;
                cos_val = cosf(angle); sin_val = sinf(angle);
            }

            // Store [cos, -sin, sin, cos] per pair (matching rope shader)
            size_t base = ((size_t)p * half_dim + j) * 4;
            pos_freqs[base + 0] = cos_val;
            pos_freqs[base + 1] = -sin_val;
            pos_freqs[base + 2] = sin_val;
            pos_freqs[base + 3] = cos_val;
        }
    }

    // Replicate per-head and per-batch: [S, half_dim, 4] → [M*S*H, half_dim, 4]
    // Layout: token-major (batch, pos, head) — matches Vulkan GEMM output
    uint32_t n_rows = M * S * N_HEADS;
    g_ropeFreqsSize = (size_t)n_rows * half_dim * 4;  // fp32 floats
    g_ropeFreqsHost.resize(g_ropeFreqsSize);

    for (uint32_t mb = 0; mb < M; mb++) {
        for (uint32_t p = 0; p < S; p++) {
            for (uint32_t h = 0; h < N_HEADS; h++) {
                uint32_t dst_row = mb * S * N_HEADS + p * N_HEADS + h;
                uint32_t src_row = p;  // same freqs for all heads at this position
                memcpy(&g_ropeFreqsHost[dst_row * half_dim * 4],
                       &pos_freqs[src_row * half_dim * 4],
                       half_dim * 4 * sizeof(float));
            }
        }
    }

    LOGI("RoPE freqs: %u rows × %u pairs, %zu floats (%.1f MB)",
         n_rows, half_dim, g_ropeFreqsSize,
         (double)(g_ropeFreqsSize * sizeof(float)) / 1e6);
    return true;
}

// Forward decls
static void dit_alloc_captures(void);
static void dit_free_captures(void);

// ============================================================
// Public API
// ============================================================
extern "C" {

bool dit_load_safetensors(const char* sf_path, const char* spv_dir) {
    if (g_init) return true;

    if (!init_vulkan(g_vk)) { LOGE("Vulkan init failed"); return false; }
    if (!create_all_pipelines(g_vk, spv_dir)) { LOGE("Pipelines failed"); return false; }
    if (!create_descriptor_pools(g_vk)) { LOGE("Descriptor pools failed"); return false; }

    if (!load_weights_safetensors(g_vk, sf_path, g_weights)) {
        LOGE("Weight loading failed"); return false;
    }

    // Per-segment command buffer
    VkCommandBufferAllocateInfo cbInfo = {};
    cbInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbInfo.commandPool = g_vk.cmdPool; cbInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbInfo.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(g_vk.device, &cbInfo, &g_segCmdBuf) != VK_SUCCESS) {
        LOGE("Segment cmd buf alloc failed"); return false;
    }

    // Allocate fp32 buffers
    VkBufferUsageFlags u = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    size_t bSz      = (size_t)MS * D * 4;               // 4MB
    size_t attSz    = (size_t)MS * N_HEADS * HEAD_DIM * 4;   // 4MB
    size_t mlpSz    = (size_t)MS * MLP_HIDDEN * 4;          // 16MB
    size_t bcSz     = 9u * (size_t)MS * D * 4;              // 36MB
    size_t attnASz  = (size_t)MS * N_HEADS * MS * 4;       // 32MB (max for cross: S*H*Nctx=256*16*1024)
    size_t crossKVSz= (size_t)M * Nctx * N_HEADS * HEAD_DIM * 4;  // 8MB (cross K/V max)

    if (!create_buf(g_vk, bSz, u, g_xBuf))       return false;
    if (!create_buf(g_vk, bSz, u, g_outBuf))     return false;
    if (!create_buf(g_vk, (size_t)M * D * 4, u, g_tEmbBuf)) return false;
    if (!create_buf(g_vk, crossKVSz, u, g_ctxBuf)) return false;
    if (!create_buf(g_vk, mlpSz, u, g_t1))       return false;
    if (!create_buf(g_vk, attSz, u, g_tQ))       return false;
    if (!create_buf(g_vk, attSz, u, g_tK))       return false;
    if (!create_buf(g_vk, attSz, u, g_tV))       return false;
    if (!create_buf(g_vk, bSz, u, g_tO))         return false;
    if (!create_buf(g_vk, bSz, u, g_nBuf))       return false;
    if (!create_buf(g_vk, bSz, u, g_gBuf))       return false;
    if (!create_buf(g_vk, crossKVSz, u, g_rBuf)) return false;
    if (!create_buf(g_vk, (size_t)M * D3 * 4, u, g_aBuf)) return false;
    if (!create_buf(g_vk, bcSz, u, g_bcBuf))     return false;
    if (!create_buf(g_vk, attSz, u, g_attnO))    return false;
    if (!create_buf(g_vk, attnASz, u, g_attnA))  return false;
    if (!create_buf(g_vk, 4096, u, g_onesBuf))   return false;
    if (!create_buf(g_vk, 3u * (size_t)M * D * 4, u, g_loraBuf)) return false;

    // Fill ones buffer
    float* ones = (float*)g_onesBuf.mapped;
    for (int i = 0; i < 1024; i++) ones[i] = 1.0f;

    // Pre-compute RoPE freqencies
    compute_rope_freqs();

    LOGI("dit_load_safetensors OK — %zu weight tensors, fp32 activations (~%.0f MB total)",
         g_weights.size(),
         (double)(bSz*4 + mlpSz + bcSz + attnASz + crossKVSz*3) / 1e6);

    // ── GPU vs CPU GEMM smoke test ──
    {
        int tM=4, tN=128, tK=128;  // small test
        float* testA = (float*)malloc(tM * tK * 4);
        float* testC_gpu = (float*)malloc(tM * tN * 4);
        float* testC_cpu = (float*)malloc(tM * tN * 4);
        for (int i = 0; i < tM*tK; i++) testA[i] = (float)((i % 127) - 63) / 63.0f;

        // Run GPU GEMM
        RC rc; rc.vk = &g_vk; rc.cmd = VK_NULL_HANDLE; rc.weights = &g_weights;
        begin_segment(rc);
        // Find a weight to test with — use first block's q_proj (first 128 cols)
        auto it = g_weights.find("blocks.0.self_attn.q_proj.weight");
        if (it != g_weights.end()) {
            // Copy test input to g_t1
            memcpy(g_t1.mapped, testA, tM * tK * 4);
            rc.gemm(g_t1, "blocks.0.self_attn.q_proj.weight", tM, tN, tK, g_tO);
            end_segment(); submit_segment();
            memcpy(testC_gpu, g_tO.mapped, tM * tN * 4);

            // Run CPU GEMM on same weight
            head_tail::cpu_gemm_bf16(tM, tN, tK, testA,
                (const uint16_t*)it->second.mapped, testC_cpu);

            float max_err = 0.0f;
            for (int i = 0; i < tM*tN; i++) {
                float d = fabsf(testC_gpu[i] - testC_cpu[i]);
                if (d > max_err) max_err = d;
            }
            LOGI("GEMM smoke test: GPU vs CPU max_err=%.6f %s",
                 (double)max_err, max_err < 1e-3f ? "OK" : "MISMATCH!");
            if (max_err >= 1e-3f) {
                LOGI("  GPU[0..4]: %.4f %.4f %.4f %.4f",
                     (double)testC_gpu[0], (double)testC_gpu[1],
                     (double)testC_gpu[2], (double)testC_gpu[3]);
                LOGI("  CPU[0..4]: %.4f %.4f %.4f %.4f",
                     (double)testC_cpu[0], (double)testC_cpu[1],
                     (double)testC_cpu[2], (double)testC_cpu[3]);
            }
        }
        free(testA); free(testC_gpu); free(testC_cpu);

        // Second smoke test: larger GEMM (K=2048) — same dimensions as block Q_proj
        int tM2=16, tN2=128, tK2=2048;
        float* testA2 = (float*)malloc(tM2 * tK2 * 4);
        float* testC_gpu2 = (float*)malloc(tM2 * tN2 * 4);
        float* testC_cpu2 = (float*)malloc(tM2 * tN2 * 4);
        for (int i = 0; i < tM2*tK2; i++) testA2[i] = (float)((i % 1021) - 510) / 510.0f;

        begin_segment(rc);
        memcpy(g_t1.mapped, testA2, tM2 * tK2 * 4);
        rc.gemm(g_t1, "blocks.0.self_attn.q_proj.weight", tM2, tN2, tK2, g_tO);
        end_segment(); submit_segment();
        memcpy(testC_gpu2, g_tO.mapped, tM2 * tN2 * 4);
        head_tail::cpu_gemm_bf16(tM2, tN2, tK2, testA2,
            (const uint16_t*)it->second.mapped, testC_cpu2);

        float max_err2 = 0.0f;
        for (int i = 0; i < tM2*tN2; i++) {
            float d = fabsf(testC_gpu2[i] - testC_cpu2[i]);
            if (d > max_err2) max_err2 = d;
        }
        LOGI("GEMM smoke K=2048: GPU vs CPU max_err=%.6f %s",
             (double)max_err2, max_err2 < 1e-3f ? "OK" : "MISMATCH!");
        if (max_err2 >= 1e-3f) {
            LOGI("  GPU[0..4]: %.4f %.4f %.4f %.4f",
                 (double)testC_gpu2[0], (double)testC_gpu2[1],
                 (double)testC_gpu2[2], (double)testC_gpu2[3]);
            LOGI("  CPU[0..4]: %.4f %.4f %.4f %.4f",
                 (double)testC_cpu2[0], (double)testC_cpu2[1],
                 (double)testC_cpu2[2], (double)testC_cpu2[3]);
        }

        free(testA2); free(testC_gpu2); free(testC_cpu2);

        // Third smoke test: SiLU + GEMM chain (same as adaln_gpu step 1-2)
        // SiLU(tEmb) → aBuf, then gemm(aBuf, W0) → t1
        float testEmb[4*D];  // simulate t_emb [M=2, D], repeat for consistent test
        for (int i = 0; i < 2*D; i++) testEmb[i] = (float)((i % 1021) - 510) / 510.0f;
        memcpy(g_tEmbBuf.mapped, testEmb, 2*D*4);

        begin_segment(rc);
        rc.silu(g_tEmbBuf, g_aBuf, 2*D);
        rc.barrier_buf(g_aBuf.buf);
        rc.gemm(g_aBuf, "blocks.0.adaln_modulation_self_attn.1.weight", 2, ADALN_LORA_DIM, D, g_t1);
        end_segment(); submit_segment();

        // CPU version
        {
            float* cpu_silu = (float*)malloc(2*D*4);
            float* cpu_gemm = (float*)malloc(2*ADALN_LORA_DIM*4);
            for (int i = 0; i < 2*D; i++) cpu_silu[i] = testEmb[i] / (1.0f + expf(-testEmb[i]));
            auto it2 = g_weights.find("blocks.0.adaln_modulation_self_attn.1.weight");
            head_tail::cpu_gemm_bf16(2, ADALN_LORA_DIM, D, cpu_silu,
                (const uint16_t*)it2->second.mapped, cpu_gemm);

            float max_err3 = 0.0f;
            float* gpu_out = (float*)g_t1.mapped;
            for (int i = 0; i < 2*ADALN_LORA_DIM; i++) {
                float d = fabsf(gpu_out[i] - cpu_gemm[i]);
                if (d > max_err3) max_err3 = d;
            }
            LOGI("SiLU+GEMM chain smoke: GPU vs CPU max_err=%.6f %s",
                 (double)max_err3, max_err3 < 1e-3f ? "OK" : "MISMATCH!");
            if (max_err3 >= 1e-3f) {
                LOGI("  GPU[0..4]: %.4f %.4f %.4f %.4f",
                     (double)gpu_out[0], (double)gpu_out[1],
                     (double)gpu_out[2], (double)gpu_out[3]);
                LOGI("  CPU[0..4]: %.4f %.4f %.4f %.4f",
                     (double)cpu_gemm[0], (double)cpu_gemm[1],
                     (double)cpu_gemm[2], (double)cpu_gemm[3]);
                // Also check SiLU output alone
                float* gpu_silu = (float*)g_aBuf.mapped;
                float max_silu_err = 0.0f;
                for (int i = 0; i < 2*D; i++) {
                    float d = fabsf(gpu_silu[i] - cpu_silu[i]);
                    if (d > max_silu_err) max_silu_err = d;
                }
                LOGI("  SiLU alone: max_err=%.6f GPU[0]=%.4f CPU[0]=%.4f",
                     (double)max_silu_err, (double)gpu_silu[0], (double)cpu_silu[0]);
            }
            free(cpu_silu); free(cpu_gemm);
        }

        // Fourth smoke test: FULL AdaLN chain — using SPLIT version (each sub-step own segment)
        {
            // Fill test t_emb data (same as before)
            memcpy(g_tEmbBuf.mapped, testEmb, 2*D*4);
            // Fill test lora data
            float testLora[2*D3];
            for (int i = 0; i < 2*D3; i++) testLora[i] = (float)((i % 997) - 498) / 1000.0f;
            memcpy(g_loraBuf.mapped, testLora, 2*D3*4);

            // Run with 1 dispatch PER segment — ultimate barrier test
            static const size_t W2C2 = (size_t)D * ADALN_LORA_DIM * 2;
            size_t loraComp2 = (size_t)M * D * 4;
            auto boff2 = [](int c) -> size_t { return (size_t)c * MS * D * 4; };

            // 1 dispatch per segment: SiLU
            begin_segment(rc); rc.silu(g_tEmbBuf, g_aBuf, M*D); end_segment(); submit_segment();
            // 1 dispatch: GEMM down
            begin_segment(rc); rc.gemm(g_aBuf, "blocks.0.adaln_modulation_self_attn.1.weight", M, ADALN_LORA_DIM, D, g_t1); end_segment(); submit_segment();
            // 1 dispatch: gemm_sub shift
            begin_segment(rc); rc.gemm_sub(g_t1, "blocks.0.adaln_modulation_self_attn.2.weight", M, D, ADALN_LORA_DIM, g_tQ, 0); end_segment(); submit_segment();
            // 1 dispatch: gemm_sub scale
            begin_segment(rc); rc.gemm_sub(g_t1, "blocks.0.adaln_modulation_self_attn.2.weight", M, D, ADALN_LORA_DIM, g_tK, W2C2); end_segment(); submit_segment();
            // 1 dispatch: gemm_sub gate
            begin_segment(rc); rc.gemm_sub(g_t1, "blocks.0.adaln_modulation_self_attn.2.weight", M, D, ADALN_LORA_DIM, g_tV, W2C2*2); end_segment(); submit_segment();
            // 1 dispatch: add lora to shift
            begin_segment(rc); rc.scale_shift_off(g_tQ, g_onesBuf, 0, g_loraBuf, 0, g_tQ, M*D, 0, 1); end_segment(); submit_segment();
            // 1 dispatch: add lora to scale
            begin_segment(rc); rc.scale_shift_off(g_tK, g_onesBuf, 0, g_loraBuf, loraComp2, g_tK, M*D, 0, 1); end_segment(); submit_segment();
            // 1 dispatch: add lora to gate
            begin_segment(rc); rc.scale_shift_off(g_tV, g_onesBuf, 0, g_loraBuf, loraComp2*2, g_tV, M*D, 0, 1); end_segment(); submit_segment();
            // 1 dispatch: scale+1
            begin_segment(rc); rc.scale_shift_off(g_tK, g_onesBuf, 0, g_onesBuf, 0, g_aBuf, M*D, 0, 0); end_segment(); submit_segment();
            // 1 dispatch: broadcast shift
            begin_segment(rc); rc.broadcast_off(g_tQ, g_bcBuf, boff2(1), M, D, S); end_segment(); submit_segment();
            // 1 dispatch: broadcast scale+1
            begin_segment(rc); rc.broadcast_off(g_aBuf, g_bcBuf, boff2(0), M, D, S); end_segment(); submit_segment();
            // 1 dispatch: broadcast gate
            begin_segment(rc); rc.broadcast_off(g_tV, g_bcBuf, boff2(2), M, D, S); end_segment(); submit_segment();

            // CPU reference
            {
                float* cpu_silu = (float*)malloc(2*D*4);
                float* cpu_down = (float*)malloc(2*ADALN_LORA_DIM*4);
                float* cpu_up_all = (float*)malloc(2*3*D*4);
                for (int i = 0; i < 2*D; i++) cpu_silu[i] = testEmb[i] / (1.0f + expf(-testEmb[i]));
                auto it_w0 = g_weights.find("blocks.0.adaln_modulation_self_attn.1.weight");
                auto it_w2 = g_weights.find("blocks.0.adaln_modulation_self_attn.2.weight");
                head_tail::cpu_gemm_bf16(2, ADALN_LORA_DIM, D, cpu_silu,
                    (const uint16_t*)it_w0->second.mapped, cpu_down);
                head_tail::cpu_gemm_bf16(2, 3*D, ADALN_LORA_DIM, cpu_down,
                    (const uint16_t*)it_w2->second.mapped, cpu_up_all);

                // Add lora: each component += lora[component]
                for (int c = 0; c < 3; c++) {
                    for (int m = 0; m < 2; m++) {
                        for (int d = 0; d < D; d++) {
                            cpu_up_all[(m*3+c)*D + d] += testLora[(m*3+c)*D + d];
                        }
                    }
                }

                // Scale+1 for component 1 (scale): scale = up_all[1] + 1.0
                float* cpu_scale_plus1 = (float*)malloc(2*D*4);
                for (int i = 0; i < 2*D; i++) cpu_scale_plus1[i] = cpu_up_all[D + i] + 1.0f;

                // Compare bcBuf content:
                // Slot 0 = scale+1, Slot 1 = shift, Slot 2 = gate
                size_t slot_bytes = 2u * S * D * 4;
                float* gpu_bc = (float*)g_bcBuf.mapped;
                float max_bc_err = 0.0f;
                for (int slot = 0; slot < 3; slot++) {
                    float* gpu_slot = gpu_bc + slot * 2*S*D;
                    float* cpu_slot = (slot == 0) ? cpu_scale_plus1 :
                                     (slot == 1) ? (cpu_up_all) :
                                     (cpu_up_all + 2*D);  // gate
                    for (int mb = 0; mb < 2; mb++) {
                        for (int s = 0; s < S; s++) {
                            for (int d = 0; d < D; d++) {
                                float diff = fabsf(gpu_slot[(mb*S+s)*D + d] - cpu_slot[mb*D + d]);
                                if (diff > max_bc_err) max_bc_err = diff;
                            }
                        }
                    }
                }
                LOGI("Full AdaLN smoke: bcBuf max_err=%.6f %s",
                     (double)max_bc_err, max_bc_err < 1e-3f ? "OK" : "MISMATCH!");

                // Also check intermediate outputs AFTER each step
                float max_t1_err = 0.0f, max_shift_err = 0.0f, max_scale_err = 0.0f;
                // Compute expected gemm_sub outputs BEFORE scale_shift adds lora
                float* cpu_up_raw = (float*)malloc(2*3*D*4);
                float* cpu_up_all_ref = (float*)malloc(2*3*D*4);
                // t1 @ W2^T without lora
                head_tail::cpu_gemm_bf16(2, 3*D, ADALN_LORA_DIM, cpu_down,
                    (const uint16_t*)it_w2->second.mapped, cpu_up_raw);
                // t1 @ W2^T + lora
                memcpy(cpu_up_all_ref, cpu_up_raw, 2*3*D*4);
                for (int c = 0; c < 3; c++)
                    for (int m = 0; m < 2; m++)
                        for (int d = 0; d < D; d++)
                            cpu_up_all_ref[(m*3+c)*D + d] += testLora[(m*3+c)*D + d];
                float* gpu_t1 = (float*)g_t1.mapped;
                float* gpu_tQ = (float*)g_tQ.mapped;  // shift + lora[0]
                float* gpu_tK = (float*)g_tK.mapped;  // scale + lora[1]
                float* gpu_aBuf = (float*)g_aBuf.mapped;  // scale + 1.0 + lora[1]
                for (int i = 0; i < 2*ADALN_LORA_DIM; i++) {
                    float d = fabsf(gpu_t1[i] - cpu_down[i]);
                    if (d > max_t1_err) max_t1_err = d;
                }
                for (int i = 0; i < 2*D; i++) {
                    float d = fabsf(gpu_tQ[i] - cpu_up_all[i]);  // shift
                    if (d > max_shift_err) max_shift_err = d;
                }
                for (int i = 0; i < 2*D; i++) {
                    float d = fabsf(gpu_tK[i] - cpu_up_all[D + i]);  // scale
                    if (d > max_scale_err) max_scale_err = d;
                }
                LOGI("  t1 (LoRA down): max_err=%.6f %s", (double)max_t1_err,
                     max_t1_err < 1e-3f ? "OK" : "MISMATCH!");
                LOGI("  shift (up+lora): max_err=%.6f %s", (double)max_shift_err,
                     max_shift_err < 1e-3f ? "OK" : "MISMATCH!");
                LOGI("  scale (up+lora): max_err=%.6f %s", (double)max_scale_err,
                     max_scale_err < 1e-3f ? "OK" : "MISMATCH!");
                if (max_shift_err >= 1e-3f || max_scale_err >= 1e-3f) {
                    LOGI("    GPU shift[0..4]: %.4f %.4f %.4f %.4f",
                         (double)gpu_tQ[0], (double)gpu_tQ[1], (double)gpu_tQ[2], (double)gpu_tQ[3]);
                    LOGI("    CPU shift[0..4]: %.4f %.4f %.4f %.4f",
                         (double)cpu_up_all[0], (double)cpu_up_all[1], (double)cpu_up_all[2], (double)cpu_up_all[3]);
                    LOGI("    GPU scale[0..4]: %.4f %.4f %.4f %.4f",
                         (double)gpu_tK[0], (double)gpu_tK[1], (double)gpu_tK[2], (double)gpu_tK[3]);
                    LOGI("    CPU scale[0..4]: %.4f %.4f %.4f %.4f",
                         (double)cpu_up_all[D], (double)cpu_up_all[D+1], (double)cpu_up_all[D+2], (double)cpu_up_all[D+3]);
                    // Check if GPU shift == GPU scale (offset bug symptom)
                    float diff_same = fabsf(gpu_tQ[0] - gpu_tK[0]);
                    LOGI("    GPU shift[0] vs scale[0]: diff=%.6f %s",
                         (double)diff_same, diff_same < 1e-6f ? "(IDENTICAL! offset bug?)" : "");
                }

                free(cpu_silu); free(cpu_down); free(cpu_up_all); free(cpu_scale_plus1);
            }
        }
    }

        // Fifth smoke test: gemm_sub alone (test buffer offset binding)
        {
            RC rc5; rc5.vk = &g_vk; rc5.cmd = VK_NULL_HANDLE; rc5.weights = &g_weights;
            size_t w2c = (size_t)D * ADALN_LORA_DIM * 2;
            float testT1[2*ADALN_LORA_DIM];
            for (int i = 0; i < 2*ADALN_LORA_DIM; i++) testT1[i] = (float)((i % 251) - 125) / 125.0f;
            memcpy(g_t1.mapped, testT1, 2*ADALN_LORA_DIM*4);

            float* cpu_sub[3]; float* gpu_sub[3];
            size_t offsets[3] = {0, w2c, w2c*2};
            for (int c = 0; c < 3; c++) {
                cpu_sub[c] = (float*)malloc(2*D*4);
                gpu_sub[c] = (float*)malloc(2*D*4);
            }

            auto it_w2 = g_weights.find("blocks.0.adaln_modulation_self_attn.2.weight");
            for (int c = 0; c < 3; c++) {
                head_tail::cpu_gemm_bf16(2, D, ADALN_LORA_DIM, testT1,
                    (const uint16_t*)((uint8_t*)it_w2->second.mapped + offsets[c]),
                    cpu_sub[c]);
            }

            begin_segment(rc5);
            rc5.gemm_sub(g_t1, "blocks.0.adaln_modulation_self_attn.2.weight", 2, D, ADALN_LORA_DIM, g_tQ, 0);
            rc5.gemm_sub(g_t1, "blocks.0.adaln_modulation_self_attn.2.weight", 2, D, ADALN_LORA_DIM, g_tK, w2c);
            rc5.gemm_sub(g_t1, "blocks.0.adaln_modulation_self_attn.2.weight", 2, D, ADALN_LORA_DIM, g_tV, w2c*2);
            end_segment(); submit_segment();

            memcpy(gpu_sub[0], g_tQ.mapped, 2*D*4);
            memcpy(gpu_sub[1], g_tK.mapped, 2*D*4);
            memcpy(gpu_sub[2], g_tV.mapped, 2*D*4);

            for (int c = 0; c < 3; c++) {
                float max_err = 0.0f;
                for (int i = 0; i < 2*D; i++) {
                    float d = fabsf(gpu_sub[c][i] - cpu_sub[c][i]);
                    if (d > max_err) max_err = d;
                }
                const char* names[] = {"shift","scale","gate"};
                LOGI("gemm_sub[%s] offset=%zu: max_err=%.6f %s",
                     names[c], offsets[c], (double)max_err,
                     max_err < 1e-3f ? "OK" : "MISMATCH!");
            }

            for (int c = 0; c < 3; c++) { free(cpu_sub[c]); free(gpu_sub[c]); }
        }

        // Sixth smoke test: scale_shift + broadcast
        {
            RC rc6; rc6.vk = &g_vk; rc6.cmd = VK_NULL_HANDLE; rc6.weights = &g_weights;
            // Test scale_shift with stride=0 for scale (scalar broadcast from onesBuf)
            float testIn[512];
            for (int i = 0; i < 512; i++) testIn[i] = (float)((i % 127) - 63) / 10.0f;
            memcpy(g_t1.mapped, testIn, 512*4);

            begin_segment(rc6);
            // scale_shift: out = x*ones[0] + testIn[0] (scalar shift, stride=0)
            // Using onesBuf for scale, g_t1 for shift (but as scalar), output to g_tO
            rc6.scale_shift_off(g_t1, g_onesBuf, 0, g_t1, 0, g_tO, 512, 0, 0);
            end_segment(); submit_segment();

            float* gpu_out = (float*)g_tO.mapped;
            float max_ss_err = 0.0f;
            for (int i = 0; i < 512; i++) {
                // Expected: out[i] = x[i]*1.0 + x[0] (scalar shift = first element)
                float expected = testIn[i] + testIn[0];
                float d = fabsf(gpu_out[i] - expected);
                if (d > max_ss_err) max_ss_err = d;
            }
            LOGI("scale_shift scalar: max_err=%.6f %s",
                 (double)max_ss_err, max_ss_err < 1e-6f ? "OK" : "MISMATCH!");
            if (max_ss_err >= 1e-6f) {
                LOGI("  GPU[0..4]: %.4f %.4f %.4f %.4f",
                     (double)gpu_out[0], (double)gpu_out[1], (double)gpu_out[2], (double)gpu_out[3]);
                LOGI("  EX[0..4]:  %.4f %.4f %.4f %.4f",
                     (double)(testIn[0]+testIn[0]), (double)(testIn[1]+testIn[0]),
                     (double)(testIn[2]+testIn[0]), (double)(testIn[3]+testIn[0]));
            }

            // Test broadcast: duplicate [2, 16] → [32, 16] (repeat=16, like S=256 but smaller)
            float testBC[2*16];
            for (int i = 0; i < 2*16; i++) testBC[i] = (float)(i * 1.1f);
            memcpy(g_t1.mapped, testBC, 2*16*4);

            begin_segment(rc6);
            rc6.broadcast_off(g_t1, g_tO, 0, 2, 16, 16);
            end_segment(); submit_segment();

            float* gpu_bc = (float*)g_tO.mapped;
            float max_bc_err = 0.0f;
            for (int i = 0; i < 2*16*16; i++) {
                int in_row = (i / 16) / 16;  // input row = output_row / repeat
                float expected = testBC[in_row * 16 + (i % 16)];
                float d = fabsf(gpu_bc[i] - expected);
                if (d > max_bc_err) max_bc_err = d;
            }
            LOGI("broadcast 2x16->32x16: max_err=%.6f %s",
                 (double)max_bc_err, max_bc_err < 1e-6f ? "OK" : "MISMATCH!");
        }

        // Seventh smoke test: LN alone on GPU vs CPU
        {
            RC rc7; rc7.vk = &g_vk; rc7.cmd = VK_NULL_HANDLE; rc7.weights = &g_weights;
            float* testLN = (float*)malloc(4 * D * 4);     // [4, 2048]
            float* cpuLN = (float*)malloc(4 * D * 4);
            for (int i = 0; i < 4*D; i++) testLN[i] = (float)((i % 1021) - 510) / 200.0f;
            memcpy(g_xBuf.mapped, testLN, 4*D*4);

            // CPU LN
            head_tail::layernorm(testLN, cpuLN, 4, D);

            // GPU LN
            begin_segment(rc7);
            rc7.layernorm(g_xBuf, g_outBuf, 4, D, 1e-6f);
            end_segment(); submit_segment();

            float max_ln_err = 0.0f;
            float* gpuLN = (float*)g_outBuf.mapped;
            for (int i = 0; i < 4*D; i++) {
                float d = fabsf(gpuLN[i] - cpuLN[i]);
                if (d > max_ln_err) max_ln_err = d;
            }
            LOGI("LN smoke (4x%d): max_err=%.6f %s", D, (double)max_ln_err,
                 max_ln_err < 1e-3f ? "OK" : "MISMATCH!");
            if (max_ln_err >= 1e-3f) {
                LOGI("  GPU[0..4]: %.4f %.4f %.4f %.4f",
                     (double)gpuLN[0], (double)gpuLN[1], (double)gpuLN[2], (double)gpuLN[3]);
                LOGI("  CPU[0..4]: %.4f %.4f %.4f %.4f",
                     (double)cpuLN[0], (double)cpuLN[1], (double)cpuLN[2], (double)cpuLN[3]);
            }

            // Also test LN with 512×2048 (same as block)
            float* testLN2 = (float*)malloc(MS * D * 4);
            float* cpuLN2 = (float*)malloc(MS * D * 4);
            for (int i = 0; i < MS*D; i++) testLN2[i] = (float)((i % 1021) - 510) / 200.0f;
            memcpy(g_xBuf.mapped, testLN2, MS*D*4);
            head_tail::layernorm(testLN2, cpuLN2, MS, D);

            begin_segment(rc7);
            rc7.layernorm(g_xBuf, g_outBuf, MS, D, 1e-6f);
            end_segment(); submit_segment();

            float max_ln2_err = 0.0f;
            float* gpuLN2 = (float*)g_outBuf.mapped;
            for (int i = 0; i < MS*D; i++) {
                float d = fabsf(gpuLN2[i] - cpuLN2[i]);
                if (d > max_ln2_err) max_ln2_err = d;
            }
            LOGI("LN smoke (512x%d): max_err=%.6f %s", D, (double)max_ln2_err,
                 max_ln2_err < 1e-3f ? "OK" : "MISMATCH!");

            free(testLN); free(cpuLN); free(testLN2); free(cpuLN2);
        }

        // Eighth smoke test: CPU AdaLN vs PyTorch reference
        {
            // Use same test data as Full AdaLN test for comparison
            float testEmb[2*D];
            for (int i = 0; i < 2*D; i++) testEmb[i] = (float)((i % 1021) - 510) / 510.0f;
            float testLora[2*D3];
            for (int i = 0; i < 2*D3; i++) testLora[i] = (float)((i % 997) - 498) / 1000.0f;

            // Run CPU AdaLN (same logic as seg_adaln_cpu)
            auto cpu_adaln_smoke = [&](const char* w0_key, const char* w2_key, float* bcBuf_out, int base) {
                // SiLU
                float aBuf[2*D];
                for (int i = 0; i < 2*D; i++) aBuf[i] = testEmb[i] / (1.0f + expf(-testEmb[i]));

                // LoRA down: aBuf @ W0^T
                auto it_w0 = g_weights.find(w0_key);
                float t1[2 * ADALN_LORA_DIM];
                head_tail::cpu_gemm_bf16(2, ADALN_LORA_DIM, D, aBuf,
                    (const uint16_t*)it_w0->second.mapped, t1);

                // LoRA up: t1 @ W2^T
                auto it_w2 = g_weights.find(w2_key);
                float up[2 * 3*D];
                head_tail::cpu_gemm_bf16(2, 3*D, ADALN_LORA_DIM, t1,
                    (const uint16_t*)it_w2->second.mapped, up);

                // Add lora
                for (int i = 0; i < 2*3*D; i++) up[i] += testLora[i];

                // scale+1
                float scale_plus1[2*D];
                for (int i = 0; i < 2*D; i++) scale_plus1[i] = up[D + i] + 1.0f;

                // Store to bcBuf (broadcast: each row repeated S times)
                size_t slot_bytes = (size_t)2 * S * D * 4;
                for (int m = 0; m < 2; m++) {
                    for (int s = 0; s < (int)S; s++) {
                        memcpy(bcBuf_out + base*slot_bytes/4 + 1*slot_bytes/4 + (m*S+s)*D,
                               up + m*D, D*4);  // shift → slot base+1
                        memcpy(bcBuf_out + base*slot_bytes/4 + 0*slot_bytes/4 + (m*S+s)*D,
                               scale_plus1 + m*D, D*4);  // scale+1 → slot base+0
                        memcpy(bcBuf_out + base*slot_bytes/4 + 2*slot_bytes/4 + (m*S+s)*D,
                               up + 2*D + m*D, D*4);  // gate → slot base+2
                    }
                }
            };

            float* bc = (float*)malloc(9u * 2*S*D * 4);
            memset(bc, 0, 9u * 2*S*D * 4);
            cpu_adaln_smoke("blocks.0.adaln_modulation_self_attn.1.weight",
                           "blocks.0.adaln_modulation_self_attn.2.weight", bc, 0);
            LOGI("CPU AdaLN smoke: shift[0]=%.4f scale+1[0]=%.4f gate[0]=%.4f",
                 (double)bc[2*S*D], (double)bc[0], (double)bc[4*S*D]);
            LOGI("  Shift[0..3]: %.4f %.4f %.4f %.4f",
                 (double)bc[2*S*D], (double)bc[2*S*D+1], (double)bc[2*S*D+2], (double)bc[2*S*D+3]);
            LOGI("  Scale+1[0..3]: %.4f %.4f %.4f %.4f",
                 (double)bc[0], (double)bc[1], (double)bc[2], (double)bc[3]);
            free(bc);

            // Compare with PyTorch: we need the WSL values as reference
            // These are computed offline and hardcoded here for comparison
            // PT SA shift[0..3] from WSL: see scripts/replica/cmp_bcbuf.py
            // For test input: shift ≈ t1@W2[0:D,:] + lora[0]
            float max_cpu_err = 0.0f;
            // We can't run PT here, but we already know from earlier tests
            // that the GPU Full AdaLN got max_err=68.88 with same test data.
            // If CPU gets different values, it's a CPU bug.
        }

    dit_alloc_captures();
    g_init = true;
    return true;
}

bool dit_compute_timestep(const float* sigmas, int M_val,
                           const char* w_t1_key, const char* w_t2_key, const char* w_tn_key,
                           void* t_emb_out, void* adaln_lora_out) {
    if (!g_init) return false;
    auto it1 = g_weights.find(w_t1_key);
    auto it2 = g_weights.find(w_t2_key);
    auto itn = g_weights.find(w_tn_key);
    if (it1 == g_weights.end() || it2 == g_weights.end() || itn == g_weights.end())
        return false;

    if (!head_tail::t_embed(sigmas, M_val,
                             (const uint16_t*)it1->second.mapped,
                             (const uint16_t*)it2->second.mapped,
                             (float*)t_emb_out, (float*)adaln_lora_out))
        return false;

    // Capture raw embedding BEFORE normalization
    float* raw_emb = (float*)malloc((size_t)M_val * D * 4);
    if (raw_emb) {
        memcpy(raw_emb, t_emb_out, (size_t)M_val * D * 4);
        FILE* fd = fopen("/sdcard/anima_on_android/output/cmp_v2/b0_raw_emb.bin", "wb");
        if (fd) { fwrite(raw_emb, 4, M_val*D, fd); fclose(fd); }
        free(raw_emb);
    }

    head_tail::t_embedding_norm((float*)t_emb_out, (const uint16_t*)itn->second.mapped, M_val, D);

    // Upload NORMALIZED t_emb to Vulkan (PyTorch passes norm'd t_emb to blocks)
    memcpy(g_tEmbBuf.mapped, t_emb_out, (size_t)M_val * D * 4);
    memcpy(g_loraBuf.mapped, adaln_lora_out, (size_t)M_val * D3 * 4);

    // Capture for comparison
    if (g_b0_temb) memcpy(g_b0_temb, t_emb_out, (size_t)M_val * D * 4);
    if (g_b0_lora) memcpy(g_b0_lora, adaln_lora_out, (size_t)M_val * D3 * 4);
    return true;
}

bool dit_head_x_embed(void* x_fp16,
                       const char* w_proj_key,
                       void* out_fp32,
                       int B, int C_in, int T, int H_pix, int W_pix) {
    if (!g_init) return false;
    auto it = g_weights.find(w_proj_key);
    if (it == g_weights.end()) { LOGE("Weight not found: %s", w_proj_key); return false; }
    return head_tail::x_embed((const uint16_t*)x_fp16, (const uint16_t*)it->second.mapped,
                               (float*)out_fp32, B, C_in, T, H_pix, W_pix, 2, 1, D);
}

bool dit_tail_final_layer(void* x_fp32, void* t_emb_fp32, void* adaln_lora_fp32,
                           const char* w_fa1, const char* w_fa2, const char* w_fl,
                           void* out_fp32, int MS_val, int M_val) {
    if (!g_init) return false;
    auto it1 = g_weights.find(w_fa1), it2 = g_weights.find(w_fa2), it3 = g_weights.find(w_fl);
    if (it1 == g_weights.end() || it2 == g_weights.end() || it3 == g_weights.end())
        return false;
    return head_tail::final_layer((const float*)x_fp32, (const float*)t_emb_fp32,
                                   (const float*)adaln_lora_fp32,
                                   (const uint16_t*)it1->second.mapped,
                                   (const uint16_t*)it2->second.mapped,
                                   (const uint16_t*)it3->second.mapped,
                                   (float*)out_fp32, MS_val, M_val, D, 64);
}

void dit_tail_unpatchify(void* in_fp32, void* out_fp32, int B, int T, int Hp, int Wp) {
    head_tail::unpatchify((const float*)in_fp32, (float*)out_fp32, B, T, Hp, Wp, 2, 1, 16);
}

bool dit_forward(void* x_data, void* ctx_data, void* out_data,
                  int MS_val, int D_val, int M_val, int Nctx_val, int CtxD_val) {
    return dit_forward_step(x_data, ctx_data, out_data, MS_val, D_val, M_val, Nctx_val, CtxD_val);
}

void dit_alloc_captures(void) {
    size_t sz = (size_t)MS * D * 4;
    size_t qkv_sz = (size_t)MS * N_HEADS * HEAD_DIM * 4;
    size_t score_sz = (size_t)(MS/M) * N_HEADS * (MS/M) * 4;
    size_t fc1_sz = (size_t)MS * MLP_HIDDEN * 4;
    g_b0_x  = (float*)malloc(sz);
    g_b0_sa = (float*)malloc(sz);
    g_b0_cx = (float*)malloc(sz);
    g_b0_mlp= (float*)malloc(sz);
    g_b0_temb = (float*)malloc((size_t)M * D * 4);
    g_b0_lora = (float*)malloc((size_t)M * D3 * 4);
    // Fine-grained intermediates
    g_b0_q  = (float*)malloc(qkv_sz);    // Q after GEMM [MS*NH*HD]
    g_b0_k  = (float*)malloc(qkv_sz);    // K after GEMM
    g_b0_v  = (float*)malloc(qkv_sz);    // V after GEMM
    g_b0_qn = (float*)malloc(qkv_sz);    // Q after RMSNorm
    g_b0_kn = (float*)malloc(qkv_sz);    // K after RMSNorm
    g_b0_qr = (float*)malloc(qkv_sz);    // Q after RoPE
    g_b0_kr = (float*)malloc(qkv_sz);    // K after RoPE
    g_b0_attn_o = (float*)malloc(qkv_sz);// Attention output
    g_b0_oproj = (float*)malloc(sz);     // O_proj output
    g_b0_fc1 = (float*)malloc(fc1_sz);  // MLP fc1 output
    g_b0_nbuf = (float*)malloc(sz);     // g_nBuf (LN+AdaLN output)
    g_b0_bcbuf = (float*)malloc(9u * (size_t)MS * D * 4);  // bcBuf (9*MS*D)
    LOGI("Block 0 capture buffers allocated (%.1f MB)", (double)(sz*7 + qkv_sz*7 + fc1_sz + 9u*MS*D*4) / 1e6);
}

void dit_dump_captures(const char* dir) {
    if (!g_b0_sa) return;
    auto save = [&](const char* name, float* data, size_t n) {
        char path[256]; snprintf(path, sizeof(path), "%s/%s.npy", dir, name);
        FILE* f = fopen(path, "wb");
        if (!f) { LOGE("Cannot open %s", path); return; }
        // Minimal .npy header (float32, 1-D)
        char header[128];
        int hdr_len = snprintf(header+10, sizeof(header)-10,
            "{'descr':'<f4','fortran_order':False,'shape':(%zu,),}", n);
        // NPY magic
        header[0] = (char)0x93; memcpy(header+1, "NUMPY", 5);
        header[6] = 1; header[7] = 0;
        uint16_t hdr_short = (uint16_t)(hdr_len + 1); // +1 for \n
        memcpy(header+8, &hdr_short, 2);
        header[10 + hdr_len] = '\n';
        size_t total = 10 + hdr_len + 1;
        fwrite(header, 1, total, f);
        fwrite(data, 4, n, f);
        fclose(f);
    };
    save("b0_x",     g_b0_x,  MS*D);
    save("b0_sa",    g_b0_sa, MS*D);
    save("b0_cx",    g_b0_cx, MS*D);
    save("b0_mlp",   g_b0_mlp,MS*D);
    save("b0_temb",  g_b0_temb, M*D);
    save("b0_lora",  g_b0_lora, M*D3);
    if (g_b0_q)  save("b0_q",  g_b0_q,  MS*N_HEADS*HEAD_DIM);
    if (g_b0_k)  save("b0_k",  g_b0_k,  MS*N_HEADS*HEAD_DIM);
    if (g_b0_v)  save("b0_v",  g_b0_v,  MS*N_HEADS*HEAD_DIM);
    if (g_b0_qn) save("b0_qn", g_b0_qn, MS*N_HEADS*HEAD_DIM);
    if (g_b0_kn) save("b0_kn", g_b0_kn, MS*N_HEADS*HEAD_DIM);
    if (g_b0_qr) save("b0_qr", g_b0_qr, MS*N_HEADS*HEAD_DIM);
    if (g_b0_kr) save("b0_kr", g_b0_kr, MS*N_HEADS*HEAD_DIM);
    if (g_b0_attn_o) save("b0_attn_o", g_b0_attn_o, MS*N_HEADS*HEAD_DIM);
    if (g_b0_oproj) save("b0_oproj", g_b0_oproj, MS*D);
    if (g_b0_fc1) save("b0_fc1", g_b0_fc1, MS*MLP_HIDDEN);
    if (g_b0_nbuf) save("b0_nbuf", g_b0_nbuf, MS*D);
    if (g_b0_bcbuf) save("b0_bcbuf", g_b0_bcbuf, 9u*MS*D);
    LOGI("Block 0 captures saved to %s", dir);
}

void dit_free_captures(void) {
    free(g_b0_x); free(g_b0_sa); free(g_b0_cx); free(g_b0_mlp);
    free(g_b0_temb); free(g_b0_lora);
    g_b0_x=g_b0_sa=g_b0_cx=g_b0_mlp=g_b0_temb=g_b0_lora=nullptr;
}

void dit_reset_pool(void) {
    if (!g_init) return;
    vkResetDescriptorPool(g_vk.device, g_vk.stepPool, 0);
}

void dit_destroy(void) {
    auto free_buf = [](Buffer& b) {
        if (b.mapped) vkUnmapMemory(g_vk.device, b.mem);
        if (b.buf) vkDestroyBuffer(g_vk.device, b.buf, nullptr);
        if (b.mem) vkFreeMemory(g_vk.device, b.mem, nullptr);
    };
    auto free_sp = [](ShaderPipe& sp) {
        if (sp.pipeline) vkDestroyPipeline(g_vk.device, sp.pipeline, nullptr);
        if (sp.layout) vkDestroyPipelineLayout(g_vk.device, sp.layout, nullptr);
        if (sp.dsl) vkDestroyDescriptorSetLayout(g_vk.device, sp.dsl, nullptr);
        if (sp.shader) vkDestroyShaderModule(g_vk.device, sp.shader, nullptr);
    };

    for (auto& kv : g_weights) free_buf(kv.second);
    g_weights.clear();

    free_sp(g_vk.gemm_bf16); free_sp(g_vk.layernorm_fp32); free_sp(g_vk.rms_norm_fp32);
    free_sp(g_vk.silu_fp32); free_sp(g_vk.gelu_fp32); free_sp(g_vk.scale_shift_fp32);
    free_sp(g_vk.rope_fp32); free_sp(g_vk.broadcast_fp32);
    free_sp(g_vk.attn_qkt_fp32); free_sp(g_vk.attn_softmax_fp32); free_sp(g_vk.attn_out_fp32);
    free_sp(g_vk.gate_fp32);

    free_buf(g_xBuf); free_buf(g_outBuf); free_buf(g_tEmbBuf); free_buf(g_ctxBuf);
    free_buf(g_t1); free_buf(g_tQ); free_buf(g_tK); free_buf(g_tV); free_buf(g_tO);
    free_buf(g_rBuf); free_buf(g_aBuf); free_buf(g_nBuf); free_buf(g_gBuf);
    free_buf(g_bcBuf); free_buf(g_onesBuf); free_buf(g_loraBuf);
    free_buf(g_attnA); free_buf(g_attnO);
    free_buf(g_ropeFreqsBuf);

    if (g_segCmdBuf) vkFreeCommandBuffers(g_vk.device, g_vk.cmdPool, 1, &g_segCmdBuf);
    if (g_vk.descPool) vkDestroyDescriptorPool(g_vk.device, g_vk.descPool, nullptr);
    if (g_vk.stepPool) vkDestroyDescriptorPool(g_vk.device, g_vk.stepPool, nullptr);
    if (g_vk.cmdPool) vkDestroyCommandPool(g_vk.device, g_vk.cmdPool, nullptr);
    if (g_vk.fence) vkDestroyFence(g_vk.device, g_vk.fence, nullptr);
    if (g_vk.stepFence) vkDestroyFence(g_vk.device, g_vk.stepFence, nullptr);
    if (g_vk.device) vkDestroyDevice(g_vk.device, nullptr);
    if (g_vk.instance) vkDestroyInstance(g_vk.instance, nullptr);

    g_init = false;
    g_ropeFreqsHost.clear();
    g_ropeFreqsSize = 0;
    LOGI("Engine destroyed");
}

} // extern "C"