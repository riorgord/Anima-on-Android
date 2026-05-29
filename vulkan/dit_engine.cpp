// DiT Vulkan Inference Engine
// Full DiT transformer block in one pre-recorded command buffer.
// Target: Snapdragon 8+ Gen 1 (Adreno 730), Android NDK.
#include <vulkan/vulkan.h>
#include <android/log.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <string>

#define LOG_TAG "DiT_VK"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// ============================================================
// Anima DiT model constants
// ============================================================
static const uint32_t D       = 2048;
static const uint32_t CtxD    = 1024;
static const uint32_t Nctx    = 512;
static const uint32_t N_HEADS = 16;
static const uint32_t HEAD_DIM = 128;
static const uint32_t MLP_HIDDEN = 8192;
static const uint32_t ADALN_LORA_DIM = 256;
static const uint32_t D3      = 6144;  // 3 * D

static uint32_t S  = 256;
static uint32_t M  = 2;
static uint32_t MS = 512;

// ============================================================
// Push constant structs (must match shader layouts exactly)
// ============================================================
struct PC_Gemm      { uint32_t M, N, K, batch; float alpha; };
struct PC_RmsNorm   { uint32_t n_rows, n_elems; float eps; };
struct PC_LayerNorm { uint32_t n_rows, n_elems; float eps; };
struct PC_Silu      { uint32_t n_total; };
struct PC_ScaleShift{ uint32_t n_total, scale_stride, shift_stride; };
struct PC_Rope      { uint32_t N, head_dim; };
struct PC_Attention { uint32_t total_q, total_kv, head_dim, S_kv; float scale; };
struct PC_Broadcast { uint32_t M, D, repeat; };
struct PC_AttnQKT     { uint32_t M_q, M_kv, H, D; float scale; };
struct PC_AttnSoftmax { uint32_t M_q, M_kv, H; };
struct PC_AttnOut     { uint32_t M_q, M_kv, H, D; };

// ============================================================
// Vulkan resource structs
// ============================================================
struct Buffer {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    size_t size = 0;
    void* mapped = nullptr;
};

// Per-shader-type pipeline bundle
struct ShaderPipe {
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
};

struct VulkanCtx {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queueFamily = 0;
    VkCommandPool cmdPool = VK_NULL_HANDLE;
    VkCommandBuffer cmd[28] = {};
    VkFence fence = VK_NULL_HANDLE;
    VkDescriptorPool descPool = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memProps = {};

    ShaderPipe gemm, rms_norm, layer_norm, silu, scale_shift, rope, attention, broadcast, gelu;
    ShaderPipe attn_qkt, attn_softmax, attn_out;
};

struct WeightInfo {
    std::string name;
    Buffer buf;  // per-tensor buffer
    size_t size;
    uint32_t dims[4];
    uint32_t ndim;
};

// ============================================================
// Helpers
// ============================================================
static uint32_t find_memory_type(VulkanCtx& ctx, uint32_t typeBits, VkMemoryPropertyFlags props) {
    for (uint32_t i = 0; i < ctx.memProps.memoryTypeCount; i++)
        if ((typeBits & (1u << i)) && (ctx.memProps.memoryTypes[i].propertyFlags & props) == props)
            return i;
    return ~0u;
}

static bool create_buffer(VulkanCtx& ctx, size_t size, VkBufferUsageFlags usage, Buffer& buf) {
    VkBufferCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = size; info.usage = usage; info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(ctx.device, &info, nullptr, &buf.buf) != VK_SUCCESS) return false;
    VkMemoryRequirements reqs;
    vkGetBufferMemoryRequirements(ctx.device, buf.buf, &reqs);
    VkMemoryAllocateInfo alloc = {};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = reqs.size;
    alloc.memoryTypeIndex = find_memory_type(ctx, reqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    if (alloc.memoryTypeIndex == ~0u)
        alloc.memoryTypeIndex = find_memory_type(ctx, reqs.memoryTypeBits,
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

// ============================================================
// Vulkan init
// ============================================================
static bool init_vulkan(VulkanCtx& ctx) {
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "DiT_VK"; appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo instInfo = {};
    instInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instInfo.pApplicationInfo = &appInfo;
    if (vkCreateInstance(&instInfo, nullptr, &ctx.instance) != VK_SUCCESS) return false;

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
            if (qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { ctx.physicalDevice = d; ctx.queueFamily = i; break; }
        if (ctx.physicalDevice) break;
    }
    if (!ctx.physicalDevice) { LOGE("No compute GPU"); return false; }
    vkGetPhysicalDeviceMemoryProperties(ctx.physicalDevice, &ctx.memProps);

    float priority = 1.0f;
    VkDeviceQueueCreateInfo qInfo = {};
    qInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qInfo.queueFamilyIndex = ctx.queueFamily; qInfo.queueCount = 1; qInfo.pQueuePriorities = &priority;
    VkDeviceCreateInfo devInfo = {};
    devInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    devInfo.queueCreateInfoCount = 1; devInfo.pQueueCreateInfos = &qInfo;
    if (vkCreateDevice(ctx.physicalDevice, &devInfo, nullptr, &ctx.device) != VK_SUCCESS) return false;
    vkGetDeviceQueue(ctx.device, ctx.queueFamily, 0, &ctx.queue);

    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = ctx.queueFamily;
    if (vkCreateCommandPool(ctx.device, &poolInfo, nullptr, &ctx.cmdPool) != VK_SUCCESS) return false;
    VkCommandBufferAllocateInfo cbInfo = {};
    cbInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbInfo.commandPool = ctx.cmdPool; cbInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbInfo.commandBufferCount = 28;
    if (vkAllocateCommandBuffers(ctx.device, &cbInfo, ctx.cmd) != VK_SUCCESS) return false;

    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (vkCreateFence(ctx.device, &fenceInfo, nullptr, &ctx.fence) != VK_SUCCESS) return false;

    LOGI("Vulkan init OK");
    return true;
}

// ============================================================
// Shader & pipeline creation
// ============================================================
static bool create_shader_pipe(VulkanCtx& ctx, const char* path, uint32_t nBindings,
                                size_t pushSize, ShaderPipe& sp) {
    auto code = load_spv(path);
    if (code.empty()) return false;
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
    if (vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &cpInfo, nullptr, &sp.pipeline) != VK_SUCCESS) return false;

    return true;
}

static bool create_all_pipelines(VulkanCtx& ctx, const char* spv_dir) {
    char p[256];
    #define CP(name, bindings, pushSize) \
        snprintf(p, sizeof(p), "%s/%s.spv", spv_dir, #name); \
        if (!create_shader_pipe(ctx, p, bindings, pushSize, ctx.name)) return false;
    CP(gemm, 3, sizeof(PC_Gemm));
    CP(rms_norm, 3, sizeof(PC_RmsNorm));
    CP(layer_norm, 2, sizeof(PC_LayerNorm));
    CP(silu, 2, sizeof(PC_Silu));
    CP(scale_shift, 4, sizeof(PC_ScaleShift));
    CP(rope, 3, sizeof(PC_Rope));
    CP(attention, 4, sizeof(PC_Attention));
    CP(broadcast, 2, sizeof(PC_Broadcast));
    #undef CP
    LOGI("All 8 pipelines created");
    return true;
}

static bool create_descriptor_pool(VulkanCtx& ctx) {
    VkDescriptorPoolSize poolSize = {};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 12000;  // 28 blocks × 53 dispatches × ~4 bindings
    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 3000;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    if (vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &ctx.descPool) != VK_SUCCESS) return false;
    LOGI("Descriptor pool created");
    return true;
}

// ============================================================
// Weight loading
// ============================================================
static bool load_weights(VulkanCtx& ctx, const char* path,
                          std::unordered_map<std::string, WeightInfo>& weights) {
    FILE* f = fopen(path, "rb");
    if (!f) { LOGE("Cannot open %s", path); return false; }

    uint32_t N;
    if (fread(&N, sizeof(N), 1, f) != 1) { fclose(f); return false; }
    LOGI("Loading %u weight tensors...", N);

    // Read all tensor headers + data
    struct RawTensor { std::string name; size_t offset; size_t size; uint32_t dims[4]; uint32_t ndim; };
    std::vector<RawTensor> raw(N);
    size_t totalData = 0;
    for (uint32_t i = 0; i < N; i++) {
        uint16_t nl; fread(&nl, sizeof(nl), 1, f);
        char* tmp = (char*)alloca(nl+1);
        fread(tmp, 1, nl, f); tmp[nl] = 0;
        raw[i].name.assign(tmp, nl);
        uint8_t nd; fread(&nd, 1, 1, f);
        raw[i].ndim = nd;
        uint32_t sh[4] = {1,1,1,1}; size_t elems = 1;
        for (uint8_t d = 0; d < nd; d++) {
            fread(&sh[d], 4, 1, f); elems *= sh[d];
        }
        memcpy(raw[i].dims, sh, sizeof(sh));
        raw[i].size = elems * 2;
        raw[i].offset = totalData;
        totalData += raw[i].size;
        fseek(f, (long)(elems * 2), SEEK_CUR);
    }

    // Read all raw data into one temp buffer, then split into per-tensor Vulkan buffers
    uint8_t* allData = new (std::nothrow) uint8_t[totalData];
    if (!allData) { LOGE("OOM for %.1f MB temp buffer", totalData/1e6); fclose(f); return false; }
    fseek(f, 4, SEEK_SET);
    for (uint32_t i = 0; i < N; i++) {
        uint16_t nl; fread(&nl, sizeof(nl), 1, f);
        fseek(f, nl + 1, SEEK_CUR);
        uint32_t sh[4]; size_t elems = 1;
        for (uint8_t d = 0; d < raw[i].ndim; d++) { fread(&sh[d], 4, 1, f); elems *= sh[d]; }
        fread(allData + raw[i].offset, 2, elems, f);
    }
    fclose(f);
    LOGI("Read %.1f MB raw data", totalData/1e6);

    // Create per-tensor Vulkan buffers
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    for (uint32_t i = 0; i < N; i++) {
        auto& w = weights[raw[i].name];
        w.name = raw[i].name;
        w.size = raw[i].size;
        w.ndim = raw[i].ndim;
        memcpy(w.dims, raw[i].dims, sizeof(w.dims));
        if (!create_buffer(ctx, raw[i].size, usage, w.buf)) {
            LOGE("Failed to create buffer for %s (%.1f MB)", raw[i].name.c_str(), raw[i].size/1e6);
            delete[] allData;
            return false;
        }
        memcpy(w.buf.mapped, allData + raw[i].offset, raw[i].size);
    }
    delete[] allData;
    LOGI("Loaded %u weights into %u Vulkan buffers", N, (uint32_t)weights.size());
    return true;
}

// ============================================================
// Lightweight init — AdaLN only (no block GEMM weights)
// ============================================================
static bool load_adaln_weights(VulkanCtx& ctx, const char* path,
                                std::unordered_map<std::string, WeightInfo>& weights) {
    FILE* f = fopen(path, "rb");
    if (!f) { LOGE("Cannot open %s", path); return false; }

    uint32_t N;
    if (fread(&N, sizeof(N), 1, f) != 1) { fclose(f); return false; }

    // First pass: find all adaln-related tensors
    std::vector<size_t> offsets, sizes;
    std::vector<std::string> names;
    size_t total = 0;
    for (uint32_t i = 0; i < N; i++) {
        uint16_t nl; fread(&nl, sizeof(nl), 1, f);
        char* tmp = (char*)alloca(nl+1);
        fread(tmp, 1, nl, f); tmp[nl] = 0;
        std::string name(tmp, nl);
        uint8_t nd; fread(&nd, 1, 1, f);
        uint32_t sh[4] = {1,1,1,1}; size_t elems = 1;
        for (uint8_t d = 0; d < nd; d++) { fread(&sh[d], 4, 1, f); elems *= sh[d]; }

        bool want = (name.find("adaln_modulation") != std::string::npos)
                 || (name.find("t_embedder") != std::string::npos)
                 || (name == "t_embedding_norm.weight");
        if (want) {
            offsets.push_back(ftell(f));  // data starts here
            sizes.push_back(elems * 2);
            names.push_back(name);
            total += elems * 2;
        }
        fseek(f, (long)(elems * 2), SEEK_CUR);
    }
    LOGI("AdaLN-only: %u tensors (%.1f KB)", (uint32_t)names.size(), total/1e3);

    // Second pass: read selected tensors
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    uint8_t* buf = new (std::nothrow) uint8_t[total];
    if (!buf) { fclose(f); return false; }
    size_t off = 0;
    for (size_t j = 0; j < names.size(); j++) {
        fseek(f, (long)offsets[j], SEEK_SET);
        fread(buf + off, 1, sizes[j], f);
        auto& w = weights[names[j]];
        w.name = names[j];
        w.size = sizes[j];
        w.ndim = 2;
        if (!create_buffer(ctx, sizes[j], usage, w.buf)) {
            LOGE("Failed buffer for %s", names[j].c_str());
            delete[] buf; fclose(f); return false;
        }
        memcpy(w.buf.mapped, buf + off, sizes[j]);
        off += sizes[j];
    }
    delete[] buf; fclose(f);
    LOGI("AdaLN weights loaded: %u tensors", (uint32_t)weights.size());
    return true;
}

static bool create_adaln_pipelines(VulkanCtx& ctx, const char* spv_dir) {
    char p[256];
    #define CP(name, bindings, pushSize) \
        snprintf(p, sizeof(p), "%s/%s.spv", spv_dir, #name); \
        if (!create_shader_pipe(ctx, p, bindings, pushSize, ctx.name)) return false;
    CP(gemm, 3, sizeof(PC_Gemm));
    CP(silu, 2, sizeof(PC_Silu));
    CP(scale_shift, 4, sizeof(PC_ScaleShift));
    CP(broadcast, 2, sizeof(PC_Broadcast));
    CP(layer_norm, 2, sizeof(PC_LayerNorm));
    CP(rms_norm, 3, sizeof(PC_RmsNorm));
    CP(gelu, 2, sizeof(PC_Silu));
    CP(attn_qkt, 3, sizeof(PC_AttnQKT));
    CP(attn_softmax, 1, sizeof(PC_AttnSoftmax));
    CP(attn_out, 3, sizeof(PC_AttnOut));
    #undef CP
    LOGI("AdaLN pipelines created (9 types + 3 attn)");
    return true;
}

// ============================================================
// Forward declaration for adaln_gpu access
static Buffer g_loraBuf;
static Buffer g_onesBuf;
static Buffer g_lnInBuf;   // FP32 LayerNorm input
static Buffer g_lnOutBuf;  // FP32 LayerNorm output
static Buffer g_rmsInBuf;   // FP16 RMSNorm input (max M=16384,D=128 → 4.2MB)
static Buffer g_rmsOutBuf;  // FP16 RMSNorm output
static Buffer g_rmsWgtBuf;  // FP16 RMSNorm weight (max D=2048)
static Buffer g_geluInBuf;  // FP16 GELU input (M=512,D=8192 → 8.4MB)
static Buffer g_geluOutBuf; // FP16 GELU output
// Attention FP16 buffers — allocated at init, sized for cross-attn worst case
static Buffer g_attnQ, g_attnK, g_attnV, g_attnA, g_attnO;

struct RC {
    VulkanCtx* vk;
    VkCommandBuffer cmd;
    std::unordered_map<std::string, WeightInfo>* weights;
    Buffer *xBuf, *tEmbBuf, *ctxBuf, *outBuf;
    Buffer *t1, *tQ, *tK, *tV, *tO, *rBuf, *aBuf, *nBuf, *gBuf, *bcBuf, *onesBuf;

    void barrier() {
        VkMemoryBarrier b = {};
        b.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
    }

    VkDescriptorSet alloc_set(VkDescriptorSetLayout dsl) {
        VkDescriptorSetAllocateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        info.descriptorPool = vk->descPool;
        info.descriptorSetCount = 1;
        info.pSetLayouts = &dsl;
        VkDescriptorSet ds;
        vkAllocateDescriptorSets(vk->device, &info, &ds);
        return ds;
    }

    void bind_buf(VkDescriptorSet ds, uint32_t binding, Buffer& buf, size_t off, size_t len) {
        VkDescriptorBufferInfo bi = { buf.buf, off, len };
        VkWriteDescriptorSet w = {};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = ds;
        w.dstBinding = binding;
        w.dstArrayElement = 0;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.pBufferInfo = &bi;
        vkUpdateDescriptorSets(vk->device, 1, &w, 0, nullptr);
    }

    void dispatch_gemm(Buffer& A, const char* wname, uint32_t Mv, uint32_t Nv, uint32_t Kv, Buffer& C,
                        size_t wSubOff = 0) {
        auto it = weights->find(wname);
        if (it == weights->end()) { LOGE("Weight not found: %s", wname); return; }
        Buffer& wbuf = it->second.buf;
        auto& sp = vk->gemm;
        auto ds = alloc_set(sp.dsl);
        bind_buf(ds, 0, A, 0, VK_WHOLE_SIZE);
        bind_buf(ds, 1, wbuf, wSubOff, VK_WHOLE_SIZE);
        bind_buf(ds, 2, C, 0, VK_WHOLE_SIZE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sp.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sp.layout, 0, 1, &ds, 0, nullptr);
        PC_Gemm pc = { Mv, Nv, Kv, 1, 1.0f };
        vkCmdPushConstants(cmd, sp.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, (Nv + 7) / 8, (Mv + 7) / 8, 1);
        barrier();
    }

    void dispatch_silu(Buffer& in, Buffer& out, uint32_t n) {
        auto& sp = vk->silu;
        auto ds = alloc_set(sp.dsl);
        bind_buf(ds, 0, in, 0, VK_WHOLE_SIZE);
        bind_buf(ds, 1, out, 0, VK_WHOLE_SIZE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sp.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sp.layout, 0, 1, &ds, 0, nullptr);
        PC_Silu pc = { n };
        vkCmdPushConstants(cmd, sp.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, (n + 255) / 256, 1, 1);
        barrier();
    }

    void dispatch_layernorm(Buffer& in, Buffer& out, uint32_t rows, uint32_t elems, float eps) {
        auto& sp = vk->layer_norm;
        auto ds = alloc_set(sp.dsl);
        bind_buf(ds, 0, in, 0, VK_WHOLE_SIZE);
        bind_buf(ds, 1, out, 0, VK_WHOLE_SIZE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sp.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sp.layout, 0, 1, &ds, 0, nullptr);
        PC_LayerNorm pc = { rows, elems, eps };
        vkCmdPushConstants(cmd, sp.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, rows, 1, 1);
        barrier();
    }

    void dispatch_rmsnorm(Buffer& in, const char* wname, Buffer& out, uint32_t rows, uint32_t elems, float eps) {
        auto it = weights->find(wname);
        if (it == weights->end()) { LOGE("Weight not found: %s", wname); return; }
        Buffer& wbuf = it->second.buf;
        auto& sp = vk->rms_norm;
        auto ds = alloc_set(sp.dsl);
        bind_buf(ds, 0, in, 0, VK_WHOLE_SIZE);
        bind_buf(ds, 1, wbuf, 0, VK_WHOLE_SIZE);
        bind_buf(ds, 2, out, 0, VK_WHOLE_SIZE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sp.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sp.layout, 0, 1, &ds, 0, nullptr);
        PC_RmsNorm pc = { rows, elems, eps };
        vkCmdPushConstants(cmd, sp.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, rows, 1, 1);
        barrier();
    }

    void dispatch_scale_shift(Buffer& x, Buffer& scl, size_t sclOff, Buffer& sft, size_t sftOff,
                               Buffer& out, uint32_t n, uint32_t sS, uint32_t fS) {
        auto& sp = vk->scale_shift;
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

    // Convenience: scale_shift with offset=0
    void dispatch_scale_shift(Buffer& x, Buffer& scl, Buffer& sft, Buffer& out,
                               uint32_t n, uint32_t sS, uint32_t fS) {
        dispatch_scale_shift(x, scl, 0, sft, 0, out, n, sS, fS);
    }

    void dispatch_broadcast(Buffer& in, Buffer& out, size_t outOff, uint32_t Mv, uint32_t Dv, uint32_t rpt) {
        auto& sp = vk->broadcast;
        auto ds = alloc_set(sp.dsl);
        bind_buf(ds, 0, in, 0, VK_WHOLE_SIZE);
        bind_buf(ds, 1, out, outOff, VK_WHOLE_SIZE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sp.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sp.layout, 0, 1, &ds, 0, nullptr);
        PC_Broadcast pc = { Mv, Dv, rpt };
        vkCmdPushConstants(cmd, sp.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, (Mv * rpt * Dv + 255) / 256, 1, 1);
        barrier();
    }

    void dispatch_rope(Buffer& t, Buffer& freqs, Buffer& out, uint32_t Nv, uint32_t hd) {
        auto& sp = vk->rope;
        auto ds = alloc_set(sp.dsl);
        bind_buf(ds, 0, t, 0, VK_WHOLE_SIZE);
        bind_buf(ds, 1, freqs, 0, VK_WHOLE_SIZE);
        bind_buf(ds, 2, out, 0, VK_WHOLE_SIZE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sp.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sp.layout, 0, 1, &ds, 0, nullptr);
        PC_Rope pc = { Nv, hd };
        vkCmdPushConstants(cmd, sp.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, (Nv + 255) / 256, 1, 1);
        barrier();
    }

    void dispatch_attention(Buffer& Q, Buffer& K, Buffer& V, Buffer& O,
                             uint32_t tq, uint32_t tkv, uint32_t hd, uint32_t Skv, float scl) {
        auto& sp = vk->attention;
        auto ds = alloc_set(sp.dsl);
        bind_buf(ds, 0, Q, 0, VK_WHOLE_SIZE);
        bind_buf(ds, 1, K, 0, VK_WHOLE_SIZE);
        bind_buf(ds, 2, V, 0, VK_WHOLE_SIZE);
        bind_buf(ds, 3, O, 0, VK_WHOLE_SIZE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sp.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sp.layout, 0, 1, &ds, 0, nullptr);
        PC_Attention pc = { tq, tkv, hd, Skv, scl };
        vkCmdPushConstants(cmd, sp.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, tq, 1, 1);
        barrier();
    }

    // GPU-side AdaLN for one channel (self_attn, cross_attn, or mlp)
    // Computes: SiLU(t_emb) → LoRA down → 3×LoRA up → scale+1 → 3×broadcast
    // Writes broadcasted shift/scale/gate to bcBuf at comp slots (base, base+1, base+2)
    // Uses: aBuf(SiLU temp & scale+1 temp), t1(LoRA down), tQ/tK/tV(shift/scale/gate)
    void adaln_gpu(const char* w0_name, const char* w2_name, int base_comp) {
        size_t comp_sz = D * ADALN_LORA_DIM * 2;
        size_t shiftSlot = (size_t)(base_comp + 1) * MS * D * 2;
        size_t scaleSlot = (size_t)(base_comp + 0) * MS * D * 2;
        size_t gateSlot  = (size_t)(base_comp + 2) * MS * D * 2;

        // 1. SiLU(t_emb) → aBuf [M*D]
        dispatch_silu(*tEmbBuf, *aBuf, M * D);
        // 2. LoRA down: aBuf @ W0^T → t1 [M, 256]
        dispatch_gemm(*aBuf, w0_name, M, ADALN_LORA_DIM, D, *t1);
        // 3-5. LoRA up × 3: t1 @ W2 components → tQ(shift), tK(scale), tV(gate)
        dispatch_gemm(*t1, w2_name, M, D, ADALN_LORA_DIM, *tQ, 0);
        dispatch_gemm(*t1, w2_name, M, D, ADALN_LORA_DIM, *tK, comp_sz);
        dispatch_gemm(*t1, w2_name, M, D, ADALN_LORA_DIM, *tV, comp_sz * 2);
        // 5b. Add external lora: tQ += lora_shift, tK += lora_scale, tV += lora_gate
        // lora layout: [3, M, D] — 3 components, each [M,D] contiguous
        size_t loraComp = M * D * 2;  // bytes per component
        dispatch_scale_shift(*tQ, *onesBuf, 0, g_loraBuf, 0,          *tQ, M*D, 0, 1);
        dispatch_scale_shift(*tK, *onesBuf, 0, g_loraBuf, loraComp,   *tK, M*D, 0, 1);
        dispatch_scale_shift(*tV, *onesBuf, 0, g_loraBuf, loraComp*2, *tV, M*D, 0, 1);

        // 6. scale+1: tK + 1.0 → aBuf (temporary)
        dispatch_scale_shift(*tK, *onesBuf, *onesBuf, *aBuf, M * D, 0, 0);
        // 7-9. Broadcast [M,D] → [MS,D] to bcBuf slots
        dispatch_broadcast(*tQ, *bcBuf, shiftSlot, M, D, S);
        dispatch_broadcast(*aBuf, *bcBuf, scaleSlot, M, D, S);  // scale+1
        dispatch_broadcast(*tV, *bcBuf, gateSlot, M, D, S);
    }

    // Weight buffer lookup
    Buffer* wbuf(const char* name) {
        auto it = weights->find(name);
        if (it == weights->end()) { LOGE("Weight not found: %s", name); return nullptr; }
        return &it->second.buf;
    }
};

// ============================================================
// Global state
// ============================================================
static VulkanCtx g_vk;
static VkCommandBuffer g_lnCmdBuf = VK_NULL_HANDLE;  // dedicated LN cmd buf (not in g_vk.cmd[])
static Buffer g_xBuf, g_tEmbBuf, g_ctxBuf, g_outBuf;
static Buffer g_t1, g_tQ, g_tK, g_tV, g_tO, g_rBuf, g_aBuf, g_nBuf, g_gBuf, g_bcBuf;
static std::unordered_map<std::string, WeightInfo> g_weights;
static bool g_init = false;

// ============================================================
// Public C API
// ============================================================
extern "C" {

bool dit_init(const char* weight_path, const char* spv_dir) {
    if (g_init) return true;
    LOGI("dit_init: weights=%s spv=%s", weight_path ? weight_path : "(none)", spv_dir);

    if (!init_vulkan(g_vk)) { LOGE("Vulkan init failed"); return false; }
    if (!create_all_pipelines(g_vk, spv_dir)) { LOGE("Pipeline creation failed"); return false; }
    if (!create_descriptor_pool(g_vk)) { LOGE("Descriptor pool failed"); return false; }

    if (weight_path && weight_path[0]) {
        if (!load_weights(g_vk, weight_path, g_weights)) {
            LOGE("Weight loading failed"); return false;
        }
    } else {
        LOGI("No weight path — weightless init");
    }

    // Allocate I/O buffers
    VkBufferUsageFlags u = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    size_t bSz = MS * D * 2;
    size_t bigSz = MS * MLP_HIDDEN * 2;
    size_t attSz = MS * N_HEADS * HEAD_DIM * 2;
    size_t ropeSz = MS * N_HEADS * (HEAD_DIM/2) * 4 * 2;
    size_t adalnSz = M * D3 * 2;
    size_t bcastSz = 9 * MS * D * 2;  // 9 AdaLN components × [MS,D] fp16 (shared per-block)

    if (!create_buffer(g_vk, bSz, u, g_xBuf)) return false;
    if (!create_buffer(g_vk, M * D * 2, u, g_tEmbBuf)) return false;
    if (!create_buffer(g_vk, M * Nctx * CtxD * 2, u, g_ctxBuf)) return false;
    if (!create_buffer(g_vk, bSz, u, g_outBuf)) return false;
    if (!create_buffer(g_vk, bigSz, u, g_t1)) return false;
    if (!create_buffer(g_vk, attSz, u, g_tQ)) return false;
    if (!create_buffer(g_vk, attSz, u, g_tK)) return false;
    if (!create_buffer(g_vk, attSz, u, g_tV)) return false;
    if (!create_buffer(g_vk, attSz, u, g_tO)) return false;
    if (!create_buffer(g_vk, ropeSz, u, g_rBuf)) return false;
    if (!create_buffer(g_vk, adalnSz, u, g_aBuf)) return false;
    if (!create_buffer(g_vk, bSz, u, g_nBuf)) return false;
    if (!create_buffer(g_vk, bSz, u, g_gBuf)) return false;
    if (!create_buffer(g_vk, bcastSz, u, g_bcBuf)) return false;

    // Lora buffer: [3, M, D] fp16 (pre-computed per sigma, CPU→GPU upload)
    if (!create_buffer(g_vk, 3 * M * D * 2, u, g_loraBuf)) return false;

    // Ones buffer: filled with fp16(1.0)
    if (!create_buffer(g_vk, 4096, u, g_onesBuf)) return false;
    uint16_t* ones = (uint16_t*)g_onesBuf.mapped;
    for (int i = 0; i < 2048; i++) ones[i] = 0x3C00;  // fp16 1.0

    LOGI("dit_init OK — %u buffers allocated", 15);
    g_init = true;
    return true;
}

// Forward declaration for pre-recording AdaLN blocks
static bool record_adaln_block(int blockIdx, int cmdIdx);
bool dit_record_all_adaln_blocks(void);

// Lightweight init: AdaLN weights only (~340KB), no GEMM/attention weights.
bool dit_init_adaln_only(const char* weight_path, const char* spv_dir) {
    if (g_init) return true;
    LOGI("dit_init_adaln_only: weights=%s spv=%s", weight_path, spv_dir);

    if (!init_vulkan(g_vk)) { LOGE("Vulkan init failed"); return false; }
    if (!create_adaln_pipelines(g_vk, spv_dir)) { LOGE("Pipeline creation failed"); return false; }
    if (!create_descriptor_pool(g_vk)) { LOGE("Descriptor pool failed"); return false; }

    if (weight_path && weight_path[0]) {
        if (!load_adaln_weights(g_vk, weight_path, g_weights)) {
            LOGE("AdaLN weight loading failed"); return false;
        }
    }

    VkBufferUsageFlags u = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    size_t bSz = MS * D * 2;
    size_t adalnSz = M * D3 * 2;
    size_t bcastSz = 9 * MS * D * 2;

    if (!create_buffer(g_vk, M * D * 2, u, g_tEmbBuf)) return false;
    if (!create_buffer(g_vk, adalnSz, u, g_aBuf)) return false;
    if (!create_buffer(g_vk, M * ADALN_LORA_DIM * 2, u, g_t1)) return false;
    if (!create_buffer(g_vk, M * D * 2, u, g_tQ)) return false;
    if (!create_buffer(g_vk, M * D * 2, u, g_tK)) return false;
    if (!create_buffer(g_vk, M * D * 2, u, g_tV)) return false;
    if (!create_buffer(g_vk, bcastSz, u, g_bcBuf)) return false;

    if (!create_buffer(g_vk, 3 * M * D * 2, u, g_loraBuf)) return false;

    // FP32 LayerNorm I/O buffers (M_max=MS=512, D=2048 → 4MB each)
    if (!create_buffer(g_vk, MS * D * 4, u, g_lnInBuf)) return false;
    if (!create_buffer(g_vk, MS * D * 4, u, g_lnOutBuf)) return false;

    // FP16 RMSNorm I/O buffers (cross-attn K norm: M=16384, D=128 → 4.2MB)
    size_t rmsMaxSz = MS * N_HEADS * D * 2;  // 512*16*2048*2 = 32MB, way oversized; double head_dim*Nctx=32768*128*2=8MB
    size_t rmsSz = 5 * 1024 * 1024;  // 5MB — enough for 16384×128 fp16
    if (!create_buffer(g_vk, rmsSz, u, g_rmsInBuf)) return false;
    if (!create_buffer(g_vk, rmsSz, u, g_rmsOutBuf)) return false;

    // FP16 RMSNorm weight buffer (max D=2048 → 4KB)
    if (!create_buffer(g_vk, D * 2, u, g_rmsWgtBuf)) return false;

    // FP16 GELU I/O buffers (M=512, MLP_HIDDEN=8192 → 8.4MB each)
    size_t geluSz = MS * MLP_HIDDEN * 2;
    if (!create_buffer(g_vk, geluSz, u, g_geluInBuf)) return false;
    if (!create_buffer(g_vk, geluSz, u, g_geluOutBuf)) return false;

    // Attention FP16 buffers (sized for cross-attn: M_q=512, M_kv=1024, H=16, D=128)
    // Q: M_q*H*D*2=2MB, K: M_kv*H*D*2=4MB, V: 4MB, A: M_q*H*M_kv*2=16.8MB, O: 2MB
    if (!create_buffer(g_vk, MS * N_HEADS * HEAD_DIM * 2, u, g_attnQ)) return false;
    if (!create_buffer(g_vk, M * Nctx * N_HEADS * HEAD_DIM * 2, u, g_attnK)) return false;
    if (!create_buffer(g_vk, M * Nctx * N_HEADS * HEAD_DIM * 2, u, g_attnV)) return false;
    if (!create_buffer(g_vk, MS * N_HEADS * (M * Nctx) * 2, u, g_attnA)) return false;
    if (!create_buffer(g_vk, MS * N_HEADS * HEAD_DIM * 2, u, g_attnO)) return false;

    if (!create_buffer(g_vk, 4096, u, g_onesBuf)) return false;
    uint16_t* ones = (uint16_t*)g_onesBuf.mapped;
    for (int i = 0; i < 2048; i++) ones[i] = 0x3C00;

    // Dummy buffers for xBuf/ctxBuf/outBuf (RC references them, not used by adaln_gpu)
    if (!create_buffer(g_vk, bSz, u, g_xBuf)) return false;
    if (!create_buffer(g_vk, bSz, u, g_outBuf)) return false;
    if (!create_buffer(g_vk, M * Nctx * CtxD * 2, u, g_ctxBuf)) return false;

    // Allocate dedicated LN command buffer (avoid overwriting AdaLN cmd[i])
    VkCommandBufferAllocateInfo lnCbInfo = {};
    lnCbInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    lnCbInfo.commandPool = g_vk.cmdPool;
    lnCbInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    lnCbInfo.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(g_vk.device, &lnCbInfo, &g_lnCmdBuf) != VK_SUCCESS) {
        LOGE("LN cmd buffer alloc failed"); return false;
    }

    LOGI("dit_init_adaln_only OK — %u buffers", 15);

    // Pre-record all 28 AdaLN blocks (avoids per-step recording / pool exhaustion)
    if (!dit_record_all_adaln_blocks()) {
        LOGE("Failed to pre-record AdaLN blocks");
        return false;
    }

    g_init = true;
    return true;
}

// Compute t_emb and lora on CPU from sigma, replace PC pre-compute entirely.
// Reads t_embedder weights from loaded weight buffers, writes results to
// g_tEmbBuf (t_emb) and g_loraBuf (lora [3,M,D]). ~0.5ms on Adreno A710.
bool dit_compute_timestep(float sigma) {
    if (!g_init) return false;

    // ---- 1. sinusoidal embedding: [M, D] ----
    auto ws = g_weights.find("t_embedder.1.linear_2.weight");
    if (ws == g_weights.end()) { LOGE("t_embedder weights not found"); return false; }
    auto w_ln_it = g_weights.find("t_embedding_norm.weight");
    if (w_ln_it == g_weights.end()) { LOGE("t_embedding_norm not found"); return false; }

    uint32_t halfD = D / 2u;  // 1024
    uint32_t D3 = 3u * D;     // 6144

    // Read t_embedder weights from GPU mapped buffers (fp16 → fp32)
    auto load_f32 = [](const Buffer& buf, size_t n) {
        std::vector<float> out(n);
        const uint16_t* src = (const uint16_t*)buf.mapped;
        for (size_t i = 0; i < n; i++) {
            // fp16 to fp32 manually
            uint32_t h = src[i];
            uint32_t sign = (h >> 15) & 1;
            uint32_t exp  = (h >> 10) & 0x1f;
            uint32_t mant = h & 0x3ff;
            uint32_t f32;
            if (exp == 0) {
                if (mant == 0) f32 = sign << 31;
                else { /* subnormal — approximate to zero */ f32 = sign << 31; }
            } else if (exp == 31) {
                f32 = (sign << 31) | 0x7f800000 | (mant << 13);  // NaN/Inf
            } else {
                f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
            }
            out[i] = *(float*)&f32;
        }
        return out;
    };

    // Weight shapes: w1 = [D, D] = [2048, 2048], w2 = [3*D, D] = [6144, 2048]
    size_t w1_n = (size_t)D * D;
    size_t w2_n = (size_t)D3 * D;
    size_t w_ln_n = D;

    // Get weights — need linear_1 too
    auto w1_it = g_weights.find("t_embedder.1.linear_1.weight");
    auto w2_it = g_weights.find("t_embedder.1.linear_2.weight");
    if (w1_it == g_weights.end() || w2_it == g_weights.end()) {
        LOGE("t_embedder weights missing"); return false;
    }

    std::vector<float> w1 = load_f32(w1_it->second.buf, w1_n);
    std::vector<float> w2 = load_f32(w2_it->second.buf, w2_n);
    std::vector<float> w_ln = load_f32(w_ln_it->second.buf, w_ln_n);

    // ---- 2. sinusoidal embedding [M, D] ----
    std::vector<float> sin_emb(M * D);
    double log10000 = log(10000.0);
    for (uint32_t b = 0; b < M; b++) {
        float* row = &sin_emb[b * D];
        for (uint32_t j = 0; j < halfD; j++) {
            double freq = sigma * exp(-log10000 * (double)j / (double)halfD);
            row[j] = (float)cos(freq);
            row[halfD + j] = (float)sin(freq);
        }
    }

    // ---- 3. t_emb = RMSNorm(sinusoidal, w_ln, eps=1e-6) ----
    for (uint32_t b = 0; b < M; b++) {
        float* row = &sin_emb[b * D];
        double sq_sum = 0.0;
        for (uint32_t i = 0; i < D; i++) sq_sum += (double)row[i] * row[i];
        double rms = sqrt(sq_sum / (double)D + 1e-6);
        uint16_t* out = (uint16_t*)g_tEmbBuf.mapped + b * D;
        for (uint32_t i = 0; i < D; i++) {
            float val = (float)((double)row[i] * (double)w_ln[i] / rms);
            // fp32 → fp16 (simple rounding)
            uint32_t bits = *(uint32_t*)&val;
            uint32_t sign16 = (bits >> 16) & 0x8000;
            uint32_t exp32 = (bits >> 23) & 0xff;
            uint32_t mant32 = bits & 0x7fffff;
            uint32_t half;
            if (exp32 == 0) { half = sign16; }
            else if (exp32 >= 143) { half = sign16 | 0x7c00; }  // overflow → inf
            else if (exp32 <= 112) { half = sign16; }
            else {
                uint32_t exp16 = exp32 - 112;
                half = sign16 | (exp16 << 10) | ((mant32 + 0x1000) >> 13);
            }
            out[i] = (uint16_t)half;
        }
    }

    // ---- 4. h = SiLU(sin_emb @ w1^T) ----
    std::vector<float> h1(M * D);
    for (uint32_t b = 0; b < M; b++) {
        for (uint32_t o = 0; o < D; o++) {
            double sum = 0.0;
            const float* w1_row = &w1[o * D];
            const float* in_row = &sin_emb[b * D];
            for (uint32_t k = 0; k < D; k++) sum += (double)in_row[k] * w1_row[k];
            float x = (float)sum;
            h1[b * D + o] = x / (1.0f + expf(-x));  // SiLU
        }
    }

    // ---- 5. lora = h1 @ w2^T → [M, 3D] → chunk → [3, M, D] ----
    uint16_t* lora_out = (uint16_t*)g_loraBuf.mapped;
    for (uint32_t b = 0; b < M; b++) {
        for (uint32_t o = 0; o < D3; o++) {
            double sum = 0.0;
            const float* w2_row = &w2[o * D];
            const float* in_row = &h1[b * D];
            for (uint32_t k = 0; k < D; k++) sum += (double)in_row[k] * w2_row[k];
            float val = (float)sum;
            // fp32 → fp16
            uint32_t bits = *(uint32_t*)&val;
            uint32_t sign16 = (bits >> 16) & 0x8000;
            uint32_t exp32 = (bits >> 23) & 0xff;
            uint32_t mant32 = bits & 0x7fffff;
            uint32_t half;
            if (exp32 == 0) { half = sign16; }
            else if (exp32 >= 143) { half = sign16 | 0x7c00; }
            else if (exp32 <= 112) { half = sign16; }
            else {
                uint32_t exp16 = exp32 - 112;
                half = sign16 | (exp16 << 10) | ((mant32 + 0x1000) >> 13);
            }
            // Chunk into [3, M, D]: shift=0, scale=D, gate=2D
            uint32_t comp = o / D;  // 0=shift, 1=scale, 2=gate
            uint32_t col  = o % D;
            lora_out[comp * M * D + b * D + col] = (uint16_t)half;
        }
    }

    return true;
}

void dit_write_lora(void* data) {
    if (!g_init) return;
    memcpy(g_loraBuf.mapped, data, 3 * M * D * 2);
}

bool dit_forward(void* x_data, void* t_emb_data, void* ctx_data, void* out_data,
                  int _MS, int _D, int _M, int _Nctx, int _CtxD) {
    if (!g_init) return false;
    S = (uint32_t)_MS / (uint32_t)_M;
    MS = (uint32_t)_MS; M = (uint32_t)_M;

    size_t xBytes = MS * _D * 2;
    size_t tBytes = M * _D * 2;
    size_t ctxBytes = M * (uint32_t)_Nctx * (uint32_t)_CtxD * 2;
    size_t outBytes = MS * _D * 2;

    memcpy(g_xBuf.mapped, x_data, xBytes);
    memcpy(g_tEmbBuf.mapped, t_emb_data, tBytes);
    memcpy(g_ctxBuf.mapped, ctx_data, ctxBytes);

    vkResetFences(g_vk.device, 1, &g_vk.fence);

    VkSubmitInfo submit = {};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &g_vk.cmd[0];

    if (vkQueueSubmit(g_vk.queue, 1, &submit, g_vk.fence) != VK_SUCCESS) {
        LOGE("Queue submit failed"); return false;
    }
    vkWaitForFences(g_vk.device, 1, &g_vk.fence, VK_TRUE, UINT64_MAX);

    memcpy(out_data, g_outBuf.mapped, outBytes);
    return true;
}

bool dit_record_gemm_test(const char* weight_name, uint32_t Mv, uint32_t Nv, uint32_t Kv) {
    if (!g_init) { LOGE("Not initialized"); return false; }
    if (g_weights.find(weight_name) == g_weights.end()) {
        LOGE("Weight not found: %s", weight_name); return false;
    }
    LOGI("GEMM test: %s M=%u N=%u K=%u", weight_name, Mv, Nv, Kv);

    VkCommandBufferBeginInfo begin = {};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(g_vk.cmd[0], &begin) != VK_SUCCESS) return false;

    RC rc; memset(&rc, 0, sizeof(rc));
    rc.vk = &g_vk; rc.cmd = g_vk.cmd[0]; rc.weights = &g_weights;
    rc.xBuf = &g_xBuf; rc.tEmbBuf = &g_tEmbBuf; rc.ctxBuf = &g_ctxBuf; rc.outBuf = &g_outBuf;
    rc.t1 = &g_t1; rc.tQ = &g_tQ; rc.tK = &g_tK; rc.tV = &g_tV; rc.tO = &g_tO;
    rc.rBuf = &g_rBuf; rc.aBuf = &g_aBuf; rc.nBuf = &g_nBuf; rc.gBuf = &g_gBuf; rc.bcBuf = &g_bcBuf; rc.onesBuf = &g_onesBuf;

    rc.dispatch_gemm(*rc.xBuf, weight_name, Mv, Nv, Kv, *rc.outBuf);

    if (vkEndCommandBuffer(g_vk.cmd[0]) != VK_SUCCESS) return false;
    LOGI("GEMM test recorded");
    return true;
}

bool dit_record_oneshot(void) {
    if (!g_init) { LOGE("dit_record_oneshot: not initialized"); return false; }

    VkCommandBufferBeginInfo begin = {};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(g_vk.cmd[0], &begin) != VK_SUCCESS) return false;

    RC rc; memset(&rc, 0, sizeof(rc));
    rc.vk = &g_vk; rc.cmd = g_vk.cmd[0]; rc.weights = &g_weights;
    rc.xBuf = &g_xBuf; rc.tEmbBuf = &g_tEmbBuf; rc.ctxBuf = &g_ctxBuf; rc.outBuf = &g_outBuf;
    rc.t1 = &g_t1; rc.tQ = &g_tQ; rc.tK = &g_tK; rc.tV = &g_tV; rc.tO = &g_tO;
    rc.rBuf = &g_rBuf; rc.aBuf = &g_aBuf; rc.nBuf = &g_nBuf; rc.gBuf = &g_gBuf; rc.bcBuf = &g_bcBuf; rc.onesBuf = &g_onesBuf;

    rc.dispatch_layernorm(*rc.xBuf, *rc.outBuf, MS, D, 1e-6f);

    if (vkEndCommandBuffer(g_vk.cmd[0]) != VK_SUCCESS) return false;
    LOGI("Command buffer recorded (layernorm test, %u rows)", MS);
    return true;
}

bool dit_record_self_attn(int blockIdx) {
    // Record a simplified self-attention path for one block:
    // LN → Q_proj → K_proj → V_proj → Q_norm → K_norm → RoPE_Q → RoPE_K → Attn → O_proj
    if (!g_init) { LOGE("Not initialized"); return false; }
    char q_w[128], k_w[128], v_w[128], o_w[128];
    snprintf(q_w, sizeof(q_w), "blocks.%d.self_attn.q_proj.weight", blockIdx);
    snprintf(k_w, sizeof(k_w), "blocks.%d.self_attn.k_proj.weight", blockIdx);
    snprintf(v_w, sizeof(v_w), "blocks.%d.self_attn.v_proj.weight", blockIdx);
    snprintf(o_w, sizeof(o_w), "blocks.%d.self_attn.output_proj.weight", blockIdx);

    VkCommandBufferBeginInfo begin = {};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(g_vk.cmd[0], &begin) != VK_SUCCESS) return false;

    RC rc; memset(&rc, 0, sizeof(rc));
    rc.vk = &g_vk; rc.cmd = g_vk.cmd[0]; rc.weights = &g_weights;
    rc.xBuf = &g_xBuf; rc.tEmbBuf = &g_tEmbBuf; rc.ctxBuf = &g_ctxBuf; rc.outBuf = &g_outBuf;
    rc.t1 = &g_t1; rc.tQ = &g_tQ; rc.tK = &g_tK; rc.tV = &g_tV; rc.tO = &g_tO;
    rc.rBuf = &g_rBuf; rc.aBuf = &g_aBuf; rc.nBuf = &g_nBuf; rc.gBuf = &g_gBuf; rc.bcBuf = &g_bcBuf; rc.onesBuf = &g_onesBuf;

    rc.dispatch_layernorm(*rc.xBuf, *rc.nBuf, MS, D, 1e-6f);
    rc.dispatch_gemm(*rc.nBuf, q_w, MS, D, D, *rc.tQ);
    rc.dispatch_gemm(*rc.nBuf, k_w, MS, D, D, *rc.tK);
    rc.dispatch_gemm(*rc.nBuf, v_w, MS, D, D, *rc.tV);

    // Q/K norms: RMSNorm per head (n_rows=MS*16=8192, n_elems=head_dim=128)
    char qn_w[128], kn_w[128];
    snprintf(qn_w, sizeof(qn_w), "blocks.%d.self_attn.q_norm.weight", blockIdx);
    snprintf(kn_w, sizeof(kn_w), "blocks.%d.self_attn.k_norm.weight", blockIdx);
    uint32_t ph = MS * N_HEADS;  // per_head_rows = 512*16 = 8192
    rc.dispatch_rmsnorm(*rc.tQ, qn_w, *rc.tQ, ph, HEAD_DIM, 1e-6f);
    rc.dispatch_rmsnorm(*rc.tK, kn_w, *rc.tK, ph, HEAD_DIM, 1e-6f);

    // O proj: skip attention for now, just project V as a test
    // V is already [MS, D] flat, treat as "attention output"
    rc.dispatch_gemm(*rc.tV, o_w, MS, D, D, *rc.outBuf);

    if (vkEndCommandBuffer(g_vk.cmd[0]) != VK_SUCCESS) return false;
    LOGI("Self-attn (Q/K/V/O+norms) block %d recorded", blockIdx);
    return true;
}

// Buffer upload helpers for pre-computed CPU data
// buf_id: 0=xBuf, 1=tEmbBuf, 2=ctxBuf, 3=outBuf, 4=bcastScale, 5=bcastShift, 6=bcastGate
bool dit_write_buf(int buf_id, void* data, size_t size) {
    if (!g_init) return false;
    Buffer* bufs[] = { &g_xBuf, &g_tEmbBuf, &g_ctxBuf, &g_outBuf,
                       &g_bcBuf /*scale*/, &g_aBuf /*shift*/, &g_gBuf /*gate*/, &g_nBuf };
    if (buf_id < 0 || buf_id >= 8) return false;
    Buffer* b = bufs[buf_id];
    if (size > b->size) size = b->size;
    memcpy(b->mapped, data, size);
    return true;
}

bool dit_read_buf(int buf_id, void* out, size_t size) {
    if (!g_init) return false;
    Buffer* bufs[] = { &g_xBuf, &g_tEmbBuf, &g_ctxBuf, &g_outBuf,
                       &g_bcBuf, &g_aBuf, &g_gBuf, &g_nBuf, &g_loraBuf };
    if (buf_id < 0 || buf_id >= 9) return false;
    Buffer* b = bufs[buf_id];
    if (size > b->size) size = b->size;
    memcpy(out, b->mapped, size);
    return true;
}

// GPU adaln test: record just self-attn adaln into cmd[0], writes to bcBuf[0..2]
bool dit_record_adaln_gpu_test(int blockIdx) {
    if (!g_init) { LOGE("Not initialized"); return false; }
    char w0[128], w2[128];
    snprintf(w0, sizeof(w0), "blocks.%d.adaln_modulation_self_attn.1.weight", blockIdx);
    snprintf(w2, sizeof(w2), "blocks.%d.adaln_modulation_self_attn.2.weight", blockIdx);

    VkCommandBufferBeginInfo begin = {};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(g_vk.cmd[0], &begin) != VK_SUCCESS) return false;

    RC rc; memset(&rc, 0, sizeof(rc));
    rc.vk = &g_vk; rc.cmd = g_vk.cmd[0]; rc.weights = &g_weights;
    rc.xBuf = &g_xBuf; rc.tEmbBuf = &g_tEmbBuf; rc.ctxBuf = &g_ctxBuf; rc.outBuf = &g_outBuf;
    rc.t1 = &g_t1; rc.tQ = &g_tQ; rc.tK = &g_tK; rc.tV = &g_tV; rc.tO = &g_tO;
    rc.rBuf = &g_rBuf; rc.aBuf = &g_aBuf; rc.nBuf = &g_nBuf; rc.gBuf = &g_gBuf;
    rc.bcBuf = &g_bcBuf; rc.onesBuf = &g_onesBuf;

    rc.adaln_gpu(w0, w2, 0);  // writes to bcBuf[0,1,2]

    if (vkEndCommandBuffer(g_vk.cmd[0]) != VK_SUCCESS) return false;
    LOGI("GPU adaln recorded for block %d", blockIdx);
    return true;
}

bool dit_record_mlp(int blockIdx) {
    // MLP: LN → fc1(2048→8192) → SiLU → fc2(8192→2048)
    if (!g_init) { LOGE("Not initialized"); return false; }
    char l1_w[128], l2_w[128];
    snprintf(l1_w, sizeof(l1_w), "blocks.%d.mlp.layer1.weight", blockIdx);
    snprintf(l2_w, sizeof(l2_w), "blocks.%d.mlp.layer2.weight", blockIdx);

    VkCommandBufferBeginInfo begin = {};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(g_vk.cmd[0], &begin) != VK_SUCCESS) return false;

    RC rc; memset(&rc, 0, sizeof(rc));
    rc.vk = &g_vk; rc.cmd = g_vk.cmd[0]; rc.weights = &g_weights;
    rc.xBuf = &g_xBuf; rc.tEmbBuf = &g_tEmbBuf; rc.ctxBuf = &g_ctxBuf; rc.outBuf = &g_outBuf;
    rc.t1 = &g_t1; rc.tQ = &g_tQ; rc.tK = &g_tK; rc.tV = &g_tV; rc.tO = &g_tO;
    rc.rBuf = &g_rBuf; rc.aBuf = &g_aBuf; rc.nBuf = &g_nBuf; rc.gBuf = &g_gBuf; rc.bcBuf = &g_bcBuf; rc.onesBuf = &g_onesBuf;

    rc.dispatch_layernorm(*rc.xBuf, *rc.nBuf, MS, D, 1e-6f);
    rc.dispatch_gemm(*rc.nBuf, l1_w, MS, MLP_HIDDEN, D, *rc.t1);
    rc.dispatch_silu(*rc.t1, *rc.t1, MS * MLP_HIDDEN);
    rc.dispatch_gemm(*rc.t1, l2_w, MS, D, MLP_HIDDEN, *rc.outBuf);

    if (vkEndCommandBuffer(g_vk.cmd[0]) != VK_SUCCESS) return false;
    LOGI("MLP block %d recorded (LN+fc1+SiLU+fc2)", blockIdx);
    return true;
}

bool dit_record_self_attn_full(int blockIdx) {
    // Self-attention with pre-uploaded AdaLN data (scale in bcBuf, shift in aBuf, gate in gBuf)
    // Path: LN → AdaLN apply → Q/K/V proj → Q/K norms → V→O proj → gate+residual
    if (!g_init) { LOGE("Not initialized"); return false; }
    char qw[128], kw[128], vw[128], ow[128], qnw[128], knw[128];
    snprintf(qw, sizeof(qw), "blocks.%d.self_attn.q_proj.weight", blockIdx);
    snprintf(kw, sizeof(kw), "blocks.%d.self_attn.k_proj.weight", blockIdx);
    snprintf(vw, sizeof(vw), "blocks.%d.self_attn.v_proj.weight", blockIdx);
    snprintf(ow, sizeof(ow), "blocks.%d.self_attn.output_proj.weight", blockIdx);
    snprintf(qnw, sizeof(qnw), "blocks.%d.self_attn.q_norm.weight", blockIdx);
    snprintf(knw, sizeof(knw), "blocks.%d.self_attn.k_norm.weight", blockIdx);

    VkCommandBufferBeginInfo begin = {};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(g_vk.cmd[0], &begin) != VK_SUCCESS) return false;

    RC rc; memset(&rc, 0, sizeof(rc));
    rc.vk = &g_vk; rc.cmd = g_vk.cmd[0]; rc.weights = &g_weights;
    rc.xBuf = &g_xBuf; rc.tEmbBuf = &g_tEmbBuf; rc.ctxBuf = &g_ctxBuf; rc.outBuf = &g_outBuf;
    rc.t1 = &g_t1; rc.tQ = &g_tQ; rc.tK = &g_tK; rc.tV = &g_tV; rc.tO = &g_tO;
    rc.rBuf = &g_rBuf; rc.aBuf = &g_aBuf; rc.nBuf = &g_nBuf; rc.gBuf = &g_gBuf; rc.bcBuf = &g_bcBuf; rc.onesBuf = &g_onesBuf;

    // 1. LayerNorm(x) → nBuf
    rc.dispatch_layernorm(*rc.xBuf, *rc.nBuf, MS, D, 1e-6f);

    // 2. AdaLN apply: nBuf * scale + shift → nBuf
    //    bcBuf layout: [scale_bcast (MS*D) | shift_bcast (MS*D) | gate_bcast (MS*D)]
    size_t scaleOff = 0;
    size_t shiftOff = MS * D * 2;       // after scale
    size_t gateOff  = MS * D * 2 * 2;   // after scale+shift
    rc.dispatch_scale_shift(*rc.nBuf, *rc.bcBuf, scaleOff, *rc.bcBuf, shiftOff,
                            *rc.nBuf, MS * D, 1, 1);

    // 3. Q/K/V proj from nBuf
    rc.dispatch_gemm(*rc.nBuf, qw, MS, D, D, *rc.tQ);
    rc.dispatch_gemm(*rc.nBuf, kw, MS, D, D, *rc.tK);
    rc.dispatch_gemm(*rc.nBuf, vw, MS, D, D, *rc.tV);

    // 4. Q/K norms
    uint32_t ph = MS * N_HEADS;
    rc.dispatch_rmsnorm(*rc.tQ, qnw, *rc.tQ, ph, HEAD_DIM, 1e-6f);
    rc.dispatch_rmsnorm(*rc.tK, knw, *rc.tK, ph, HEAD_DIM, 1e-6f);

    // 5. Skip attention — use V directly as "attention output"
    //    O_proj: V → [MS, D]
    rc.dispatch_gemm(*rc.tV, ow, MS, D, D, *rc.tO);

    // 6. Gate + residual → outBuf: out = x + gate * attn_out
    //    ScaleShift: tO * gate(bcBuf[gateOff:]) + xBuf → outBuf
    rc.dispatch_scale_shift(*rc.tO, *rc.bcBuf, gateOff, *rc.xBuf, 0, *rc.outBuf, MS * D, 1, 1);

    if (vkEndCommandBuffer(g_vk.cmd[0]) != VK_SUCCESS) return false;
    LOGI("Self-attn full block %d recorded (LN+AdaLN+QKV+norms+V→O+gate)", blockIdx);
    return true;
}

bool dit_record_adaln_only(void) {
    // Minimal test: LN → AdaLN apply (no QKV, no norms)
    // bcBuf layout: [scale_bcast (MS*D) | shift_bcast (MS*D)]
    if (!g_init) { LOGE("Not initialized"); return false; }
    size_t scaleOff = 0, shiftOff = MS * D * 2;

    VkCommandBufferBeginInfo begin = {};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(g_vk.cmd[0], &begin) != VK_SUCCESS) return false;

    RC rc; memset(&rc, 0, sizeof(rc));
    rc.vk = &g_vk; rc.cmd = g_vk.cmd[0]; rc.weights = &g_weights;
    rc.xBuf = &g_xBuf; rc.tEmbBuf = &g_tEmbBuf; rc.ctxBuf = &g_ctxBuf; rc.outBuf = &g_outBuf;
    rc.t1 = &g_t1; rc.tQ = &g_tQ; rc.tK = &g_tK; rc.tV = &g_tV; rc.tO = &g_tO;
    rc.rBuf = &g_rBuf; rc.aBuf = &g_aBuf; rc.nBuf = &g_nBuf; rc.gBuf = &g_gBuf; rc.bcBuf = &g_bcBuf; rc.onesBuf = &g_onesBuf;

    rc.dispatch_layernorm(*rc.xBuf, *rc.nBuf, MS, D, 1e-6f);
    rc.dispatch_scale_shift(*rc.nBuf, *rc.bcBuf, scaleOff, *rc.bcBuf, shiftOff,
                            *rc.outBuf, MS * D, 1, 1);

    if (vkEndCommandBuffer(g_vk.cmd[0]) != VK_SUCCESS) return false;
    LOGI("AdaLN-only recorded (LN+ScaleShift)");
    return true;
}

bool dit_record_adaln_gemm(int blockIdx) {
    // LN + AdaLN apply + Q_proj only
    // bcBuf: [scale_bcast (MS*D) | shift_bcast (MS*D)]
    if (!g_init) return false;
    size_t scaleOff = 0, shiftOff = MS * D * 2;
    char qw[128]; snprintf(qw, sizeof(qw), "blocks.%d.self_attn.q_proj.weight", blockIdx);

    VkCommandBufferBeginInfo begin = {};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(g_vk.cmd[0], &begin) != VK_SUCCESS) return false;

    RC rc; memset(&rc, 0, sizeof(rc));
    rc.vk = &g_vk; rc.cmd = g_vk.cmd[0]; rc.weights = &g_weights;
    rc.xBuf = &g_xBuf; rc.tEmbBuf = &g_tEmbBuf; rc.ctxBuf = &g_ctxBuf; rc.outBuf = &g_outBuf;
    rc.t1 = &g_t1; rc.tQ = &g_tQ; rc.tK = &g_tK; rc.tV = &g_tV; rc.tO = &g_tO;
    rc.rBuf = &g_rBuf; rc.aBuf = &g_aBuf; rc.nBuf = &g_nBuf; rc.gBuf = &g_gBuf; rc.bcBuf = &g_bcBuf; rc.onesBuf = &g_onesBuf;

    rc.dispatch_layernorm(*rc.xBuf, *rc.nBuf, MS, D, 1e-6f);
    rc.dispatch_scale_shift(*rc.nBuf, *rc.bcBuf, scaleOff, *rc.bcBuf, shiftOff,
                            *rc.nBuf, MS * D, 1, 1);
    rc.dispatch_gemm(*rc.nBuf, qw, MS, D, D, *rc.outBuf);

    if (vkEndCommandBuffer(g_vk.cmd[0]) != VK_SUCCESS) return false;
    LOGI("AdaLN+GEMM recorded");
    return true;
}

bool dit_record_block_full(int blockIdx) {
    // Full block: self-attn → MLP (skip cross-attn for now)
    // Buffer pipeline: nBuf=scratch, tQ/K/V=QKV, tO=attn_out, t1=residual_x, then tQ=fc1_out
    // bcBuf layout: [0]scale_self [1]shift_self [2]gate_self [3]scale_mlp [4]shift_mlp [5]gate_mlp
    if (!g_init) return false;
    auto adaln_off = [](int section, int comp) -> size_t {
        return (size_t)(section * 3 + comp) * MS * D * 2;
    };
    int b = blockIdx;
    char qw[128],kw[128],vw[128],ow[128],qnw[128],knw[128],l1w[128],l2w[128];
    snprintf(qw,sizeof(qw),"blocks.%d.self_attn.q_proj.weight",b);
    snprintf(kw,sizeof(kw),"blocks.%d.self_attn.k_proj.weight",b);
    snprintf(vw,sizeof(vw),"blocks.%d.self_attn.v_proj.weight",b);
    snprintf(ow,sizeof(ow),"blocks.%d.self_attn.output_proj.weight",b);
    snprintf(qnw,sizeof(qnw),"blocks.%d.self_attn.q_norm.weight",b);
    snprintf(knw,sizeof(knw),"blocks.%d.self_attn.k_norm.weight",b);
    snprintf(l1w,sizeof(l1w),"blocks.%d.mlp.layer1.weight",b);
    snprintf(l2w,sizeof(l2w),"blocks.%d.mlp.layer2.weight",b);

    VkCommandBufferBeginInfo bi = {};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(g_vk.cmd[0], &bi) != VK_SUCCESS) return false;

    RC rc; memset(&rc,0,sizeof(rc));
    rc.vk=&g_vk; rc.cmd=g_vk.cmd[0]; rc.weights=&g_weights;
    rc.xBuf=&g_xBuf; rc.tEmbBuf=&g_tEmbBuf; rc.ctxBuf=&g_ctxBuf; rc.outBuf=&g_outBuf;
    rc.t1=&g_t1; rc.tQ=&g_tQ; rc.tK=&g_tK; rc.tV=&g_tV; rc.tO=&g_tO;
    rc.rBuf=&g_rBuf; rc.aBuf=&g_aBuf; rc.nBuf=&g_nBuf; rc.gBuf=&g_gBuf; rc.bcBuf=&g_bcBuf; rc.onesBuf=&g_onesBuf;

    uint32_t ph = MS * N_HEADS;

    // === Self-Attention ===
    rc.dispatch_layernorm(*rc.xBuf, *rc.nBuf, MS, D, 1e-6f);
    rc.dispatch_scale_shift(*rc.nBuf,*rc.bcBuf,adaln_off(0,0),*rc.bcBuf,adaln_off(0,1),
                            *rc.nBuf, MS*D,1,1);
    rc.dispatch_gemm(*rc.nBuf,qw,MS,D,D,*rc.tQ);
    rc.dispatch_gemm(*rc.nBuf,kw,MS,D,D,*rc.tK);
    rc.dispatch_gemm(*rc.nBuf,vw,MS,D,D,*rc.tV);
    rc.dispatch_rmsnorm(*rc.tQ,qnw,*rc.tQ,ph,HEAD_DIM,1e-6f);
    rc.dispatch_rmsnorm(*rc.tK,knw,*rc.tK,ph,HEAD_DIM,1e-6f);
    rc.dispatch_gemm(*rc.tV,ow,MS,D,D,*rc.tO);        // tO = V @ Wo
    // Self-attn residual → tV (reuse V buffer, no longer needed after O_proj)
    rc.dispatch_scale_shift(*rc.tO,*rc.bcBuf,adaln_off(0,2),*rc.xBuf,0,
                            *rc.tV, MS*D,1,1);         // tV = x + gate*attn_out

    // === MLP ===
    rc.dispatch_layernorm(*rc.tV,*rc.nBuf,MS,D,1e-6f);
    rc.dispatch_scale_shift(*rc.nBuf,*rc.bcBuf,adaln_off(1,0),*rc.bcBuf,adaln_off(1,1),
                            *rc.nBuf, MS*D,1,1);
    rc.dispatch_gemm(*rc.nBuf,l1w,MS,MLP_HIDDEN,D,*rc.t1);     // t1(8MB) = fc1
    rc.dispatch_silu(*rc.t1,*rc.t1,MS*MLP_HIDDEN);
    rc.dispatch_gemm(*rc.t1,l2w,MS,D,MLP_HIDDEN,*rc.nBuf);    // nBuf = fc2
    rc.dispatch_scale_shift(*rc.nBuf,*rc.bcBuf,adaln_off(1,2),*rc.tV,0,
                            *rc.outBuf, MS*D,1,1);             // out = fc2*gate + residual(tV)

    if (vkEndCommandBuffer(g_vk.cmd[0]) != VK_SUCCESS) return false;
    LOGI("Full block %d recorded (16 dispatches)", blockIdx);
    return true;
}

static void record_one_block(RC& rc, int b, Buffer& inBuf, Buffer& outBuf) {
    // Shared bcBuf layout: 9 components at comp * MS*D*2 (no per-block offset)
    //   [0]=scale_self [1]=shift_self [2]=gate_self
    //   [3]=scale_cross [4]=shift_cross [5]=gate_cross
    //   [6]=scale_mlp [7]=shift_mlp [8]=gate_mlp
    auto off = [&](int comp) -> size_t {
        return (size_t)comp * MS * D * 2;
    };
    uint32_t ph = MS * N_HEADS;       // per-head rows for self-attn
    uint32_t ph_cross = M * Nctx * N_HEADS;  // per-head rows for cross-attn (1024*16=16384)
    uint32_t MS_kv = M * Nctx;        // KV tokens for cross-attn (1024)

    char qw[128],kw[128],vw[128],ow[128],qnw[128],knw[128];
    char cx_kw[128],cx_vw[128],cx_ow[128],cx_knw[128];
    char l1w[128],l2w[128];
    snprintf(qw,sizeof(qw),"blocks.%d.self_attn.q_proj.weight",b);
    snprintf(kw,sizeof(kw),"blocks.%d.self_attn.k_proj.weight",b);
    snprintf(vw,sizeof(vw),"blocks.%d.self_attn.v_proj.weight",b);
    snprintf(ow,sizeof(ow),"blocks.%d.self_attn.output_proj.weight",b);
    snprintf(qnw,sizeof(qnw),"blocks.%d.self_attn.q_norm.weight",b);
    snprintf(knw,sizeof(knw),"blocks.%d.self_attn.k_norm.weight",b);
    snprintf(cx_kw,sizeof(cx_kw),"blocks.%d.cross_attn.k_proj.weight",b);
    snprintf(cx_vw,sizeof(cx_vw),"blocks.%d.cross_attn.v_proj.weight",b);
    snprintf(cx_ow,sizeof(cx_ow),"blocks.%d.cross_attn.output_proj.weight",b);
    snprintf(cx_knw,sizeof(cx_knw),"blocks.%d.cross_attn.k_norm.weight",b);
    snprintf(l1w,sizeof(l1w),"blocks.%d.mlp.layer1.weight",b);
    snprintf(l2w,sizeof(l2w),"blocks.%d.mlp.layer2.weight",b);

    // GPU-side AdaLN for all three channels (writes to bcBuf[0..8])
    // Weight names for block b's AdaLN modules
    char adaln_s0[128], adaln_s2[128], adaln_c0[128], adaln_c2[128], adaln_m0[128], adaln_m2[128];
    snprintf(adaln_s0,sizeof(adaln_s0),"blocks.%d.adaln_modulation_self_attn.1.weight",b);
    snprintf(adaln_s2,sizeof(adaln_s2),"blocks.%d.adaln_modulation_self_attn.2.weight",b);
    snprintf(adaln_c0,sizeof(adaln_c0),"blocks.%d.adaln_modulation_cross_attn.1.weight",b);
    snprintf(adaln_c2,sizeof(adaln_c2),"blocks.%d.adaln_modulation_cross_attn.2.weight",b);
    snprintf(adaln_m0,sizeof(adaln_m0),"blocks.%d.adaln_modulation_mlp.1.weight",b);
    snprintf(adaln_m2,sizeof(adaln_m2),"blocks.%d.adaln_modulation_mlp.2.weight",b);

    // Self-attn AdaLN → bcBuf[0,1,2]
    rc.adaln_gpu(adaln_s0, adaln_s2, 0);
    // ===== Self-attn: LN→AdaLN→QKV→norms→V→O→gate+residual → tV =====
    rc.dispatch_layernorm(inBuf, *rc.nBuf, MS, D, 1e-6f);
    rc.dispatch_scale_shift(*rc.nBuf,*rc.bcBuf,off(0),*rc.bcBuf,off(1),*rc.nBuf, MS*D,1,1);
    rc.dispatch_gemm(*rc.nBuf,qw,MS,D,D,*rc.tQ);
    rc.dispatch_gemm(*rc.nBuf,kw,MS,D,D,*rc.tK);
    rc.dispatch_gemm(*rc.nBuf,vw,MS,D,D,*rc.tV);
    rc.dispatch_rmsnorm(*rc.tQ,qnw,*rc.tQ,ph,HEAD_DIM,1e-6f);
    rc.dispatch_rmsnorm(*rc.tK,knw,*rc.tK,ph,HEAD_DIM,1e-6f);
    rc.dispatch_gemm(*rc.tV,ow,MS,D,D,*rc.tO);
    rc.dispatch_scale_shift(*rc.tO,*rc.bcBuf,off(2),inBuf,0,
                            *rc.tV, MS*D,1,1);                    // tV = x + gate*self_attn

    // Cross-attn AdaLN → bcBuf[3,4,5]
    rc.adaln_gpu(adaln_c0, adaln_c2, 3);
    // ===== Cross-attn: LN→AdaLN→Q(from x)→K/V(from ctx)→norms→O→gate+residual =====
    rc.dispatch_layernorm(*rc.tV, *rc.nBuf, MS, D, 1e-6f);
    rc.dispatch_scale_shift(*rc.nBuf,*rc.bcBuf,off(3),*rc.bcBuf,off(4),*rc.nBuf, MS*D,1,1);
    rc.dispatch_gemm(*rc.nBuf,qw,MS,D,D,*rc.tQ);
    rc.dispatch_gemm(*rc.ctxBuf,cx_kw,MS_kv,D,CtxD,*rc.t1);
    rc.dispatch_gemm(*rc.ctxBuf,cx_vw,MS_kv,D,CtxD,*rc.rBuf);
    rc.dispatch_rmsnorm(*rc.tQ,qnw,*rc.tQ,ph,HEAD_DIM,1e-6f);
    rc.dispatch_rmsnorm(*rc.t1,cx_knw,*rc.t1,ph_cross,HEAD_DIM,1e-6f);
    rc.dispatch_gemm(*rc.rBuf,cx_ow,MS,D,D,*rc.tO);
    rc.dispatch_scale_shift(*rc.tO,*rc.bcBuf,off(5),*rc.tV,0,
                            *rc.gBuf, MS*D,1,1);

    // MLP AdaLN → bcBuf[6,7,8]
    rc.adaln_gpu(adaln_m0, adaln_m2, 6);
    // ===== MLP: LN→AdaLN→fc1→SiLU→fc2→gate+residual =====
    rc.dispatch_layernorm(*rc.gBuf,*rc.nBuf,MS,D,1e-6f);
    rc.dispatch_scale_shift(*rc.nBuf,*rc.bcBuf,off(6),*rc.bcBuf,off(7),*rc.nBuf, MS*D,1,1);
    rc.dispatch_gemm(*rc.nBuf,l1w,MS,MLP_HIDDEN,D,*rc.t1);
    rc.dispatch_silu(*rc.t1,*rc.t1,MS*MLP_HIDDEN);
    rc.dispatch_gemm(*rc.t1,l2w,MS,D,MLP_HIDDEN,*rc.nBuf);
    rc.dispatch_scale_shift(*rc.nBuf,*rc.bcBuf,off(8),*rc.gBuf,0,
                            outBuf, MS*D,1,1);
}

bool dit_record_n_blocks(int n) {
    if (!g_init || n < 1 || n > 28) return false;
    VkCommandBufferBeginInfo bi = {};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(g_vk.cmd[0], &bi) != VK_SUCCESS) return false;

    RC rc; memset(&rc,0,sizeof(rc));
    rc.vk=&g_vk; rc.cmd=g_vk.cmd[0]; rc.weights=&g_weights;
    rc.xBuf=&g_xBuf; rc.tEmbBuf=&g_tEmbBuf; rc.ctxBuf=&g_ctxBuf; rc.outBuf=&g_outBuf;
    rc.t1=&g_t1; rc.tQ=&g_tQ; rc.tK=&g_tK; rc.tV=&g_tV; rc.tO=&g_tO;
    rc.rBuf=&g_rBuf; rc.aBuf=&g_aBuf; rc.nBuf=&g_nBuf; rc.gBuf=&g_gBuf; rc.bcBuf=&g_bcBuf; rc.onesBuf=&g_onesBuf;

    for (int i = 0; i < n; i++) {
        Buffer* out, *in;
        if (i == 0) in = rc.xBuf;
        else if (i % 2 == 0) in = rc.xBuf;
        else in = rc.tV;

        if (i == n-1) out = rc.outBuf;
        else if (i % 2 == 0) out = rc.tV;
        else out = rc.xBuf;

        record_one_block(rc, i, *in, *out);
    }

    if (vkEndCommandBuffer(g_vk.cmd[0]) != VK_SUCCESS) return false;
    LOGI("%d blocks recorded (%d dispatches)", n, n*16);
    return true;
}

bool dit_record_block_to(int blockIdx, int cmdIdx) {
    // Record one block into cmd[cmdIdx], using shared bcBuf offsets
    if (!g_init || cmdIdx < 0 || cmdIdx >= 28) return false;
    VkCommandBufferBeginInfo bi = {};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(g_vk.cmd[cmdIdx], &bi) != VK_SUCCESS) return false;

    RC rc; memset(&rc,0,sizeof(rc));
    rc.vk=&g_vk; rc.cmd=g_vk.cmd[cmdIdx]; rc.weights=&g_weights;
    rc.xBuf=&g_xBuf; rc.tEmbBuf=&g_tEmbBuf; rc.ctxBuf=&g_ctxBuf; rc.outBuf=&g_outBuf;
    rc.t1=&g_t1; rc.tQ=&g_tQ; rc.tK=&g_tK; rc.tV=&g_tV; rc.tO=&g_tO;
    rc.rBuf=&g_rBuf; rc.aBuf=&g_aBuf; rc.nBuf=&g_nBuf; rc.gBuf=&g_gBuf; rc.bcBuf=&g_bcBuf; rc.onesBuf=&g_onesBuf;

    // Block reads from xBuf, writes to outBuf (caller copies outBuf→xBuf between blocks)
    record_one_block(rc, blockIdx, *rc.xBuf, *rc.outBuf);

    if (vkEndCommandBuffer(g_vk.cmd[cmdIdx]) != VK_SUCCESS) return false;
    return true;
}

bool dit_init_all_blocks(void) {
    if (!g_init) return false;
    for (int i = 0; i < 28; i++) {
        if (!dit_record_block_to(i, i)) { LOGE("Failed to record block %d", i); return false; }
    }
    LOGI("All 28 blocks recorded (1 per cmd buffer)");
    return true;
}

bool dit_forward_28blocks(void* x_data, void* t_emb_data, void* ctx_data, void* out_data,
                           int _MS, int _D, int _M, int _Nctx, int _CtxD) {
    if (!g_init) return false;
    MS = (uint32_t)_MS; M = (uint32_t)_M; S = MS / M;
    size_t xBytes = MS * D * 2;
    size_t tBytes = M * D * 2;
    size_t ctxBytes = M * (uint32_t)_Nctx * (uint32_t)_CtxD * 2;

    // Upload inputs (t_emb and ctx stay constant across blocks)
    memcpy(g_xBuf.mapped, x_data, xBytes);
    memcpy(g_tEmbBuf.mapped, t_emb_data, tBytes);
    if (ctx_data) memcpy(g_ctxBuf.mapped, ctx_data, ctxBytes);

    for (int i = 0; i < 28; i++) {
        vkResetFences(g_vk.device, 1, &g_vk.fence);
        VkSubmitInfo submit = {};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &g_vk.cmd[i];
        if (vkQueueSubmit(g_vk.queue, 1, &submit, g_vk.fence) != VK_SUCCESS) {
            LOGE("Submit failed at block %d", i); return false;
        }
        vkWaitForFences(g_vk.device, 1, &g_vk.fence, VK_TRUE, UINT64_MAX);
        memcpy(g_xBuf.mapped, g_outBuf.mapped, xBytes);
    }

    memcpy(out_data, g_outBuf.mapped, xBytes);
    return true;
}

static bool record_adaln_block(int blockIdx, int cmdIdx) {
    char w0_s[128], w2_s[128], w0_c[128], w2_c[128], w0_m[128], w2_m[128];
    snprintf(w0_s, sizeof(w0_s), "blocks.%d.adaln_modulation_self_attn.1.weight", blockIdx);
    snprintf(w2_s, sizeof(w2_s), "blocks.%d.adaln_modulation_self_attn.2.weight", blockIdx);
    snprintf(w0_c, sizeof(w0_c), "blocks.%d.adaln_modulation_cross_attn.1.weight", blockIdx);
    snprintf(w2_c, sizeof(w2_c), "blocks.%d.adaln_modulation_cross_attn.2.weight", blockIdx);
    snprintf(w0_m, sizeof(w0_m), "blocks.%d.adaln_modulation_mlp.1.weight", blockIdx);
    snprintf(w2_m, sizeof(w2_m), "blocks.%d.adaln_modulation_mlp.2.weight", blockIdx);

    VkCommandBufferBeginInfo bi = {};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(g_vk.cmd[cmdIdx], &bi) != VK_SUCCESS) return false;

    RC rc; memset(&rc, 0, sizeof(rc));
    rc.vk = &g_vk; rc.cmd = g_vk.cmd[cmdIdx]; rc.weights = &g_weights;
    rc.xBuf = &g_xBuf; rc.tEmbBuf = &g_tEmbBuf; rc.ctxBuf = &g_ctxBuf; rc.outBuf = &g_outBuf;
    rc.t1 = &g_t1; rc.tQ = &g_tQ; rc.tK = &g_tK; rc.tV = &g_tV; rc.tO = &g_tO;
    rc.rBuf = &g_rBuf; rc.aBuf = &g_aBuf; rc.nBuf = &g_nBuf; rc.gBuf = &g_gBuf;
    rc.bcBuf = &g_bcBuf; rc.onesBuf = &g_onesBuf;

    rc.adaln_gpu(w0_s, w2_s, 0);  // self:  bcBuf[0,1,2]
    rc.adaln_gpu(w0_c, w2_c, 3);  // cross: bcBuf[3,4,5]
    rc.adaln_gpu(w0_m, w2_m, 6);  // mlp:   bcBuf[6,7,8]

    return vkEndCommandBuffer(g_vk.cmd[cmdIdx]) == VK_SUCCESS;
}

bool dit_record_all_adaln_blocks(void) {
    // Called during init before g_init is set, so no g_init check here.
    for (int i = 0; i < 28; i++) {
        if (!record_adaln_block(i, i)) {
            LOGE("Failed to record adaln block %d", i);
            return false;
        }
    }
    LOGI("All 28 AdaLN blocks pre-recorded");
    return true;
}

bool dit_adaln_one_block(int blockIdx, void* out_9MD) {
    // Submit pre-recorded cmd[blockIdx], wait, read bcBuf back.
    // tEmbBuf + g_loraBuf must already be uploaded before calling.
    // Output [9, M, D] fp16 — reordered to [shift, scale, gate] per channel.
    // Scale values have +1 undone (adaln_gpu adds 1 for broadcast; Block.forward adds it back).
    if (!g_init || blockIdx < 0 || blockIdx >= 28) return false;

    vkResetFences(g_vk.device, 1, &g_vk.fence);
    VkSubmitInfo submit = {};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &g_vk.cmd[blockIdx];
    if (vkQueueSubmit(g_vk.queue, 1, &submit, g_vk.fence) != VK_SUCCESS) return false;
    vkWaitForFences(g_vk.device, 1, &g_vk.fence, VK_TRUE, UINT64_MAX);

    // Read first M rows from each bcBuf component
    // bcBuf stores [scale+1, shift, gate] per channel; reorder to [shift, scale, gate]
    // Reorder map: out[0]=bc[1] out[1]=bc[0] out[2]=bc[2] out[3]=bc[4] out[4]=bc[3] out[5]=bc[5] out[6]=bc[7] out[7]=bc[6] out[8]=bc[8]
    static const int reorder[9] = {1, 0, 2, 4, 3, 5, 7, 6, 8};
    // Scale components after reorder are at indices 1, 4, 7 (need -1 undo)
    static const bool is_scale[9] = {false, true, false, false, true, false, false, true, false};

    size_t mBytes = (size_t)M * D * 2;
    uint16_t* out = (uint16_t*)out_9MD;
    uint16_t* bc = (uint16_t*)g_bcBuf.mapped;

    for (int c = 0; c < 9; c++) {
        int src = reorder[c];
        memcpy(out + c * M * D, bc + src * MS * D, mBytes);
    }

    // Undo scale+1 on components 1, 4, 7
    for (int c = 0; c < 9; c++) {
        if (!is_scale[c]) continue;
        uint16_t* row = out + c * M * D;
        for (size_t i = 0; i < (size_t)M * D; i++) {
            // fp16 → fp32, subtract 1, fp32 → fp16
            uint32_t h = row[i];
            uint32_t sign = (h >> 15) & 1;
            uint32_t exp  = (h >> 10) & 0x1f;
            uint32_t mant = h & 0x3ff;
            float val;
            if (exp == 0) {
                val = 0.0f;  // subnormals → 0 (scale values ~0-2, never subnormal)
            } else if (exp == 31) {
                continue;  // NaN/Inf, leave as-is
            } else {
                uint32_t f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
                val = *(float*)&f32;
            }
            val -= 1.0f;
            // fp32 → fp16
            uint32_t bits = *(uint32_t*)&val;
            uint32_t s16 = (bits >> 16) & 0x8000;
            uint32_t e32 = (bits >> 23) & 0xff;
            uint32_t m32 = bits & 0x7fffff;
            if (e32 == 0) { row[i] = (uint16_t)s16; }
            else if (e32 >= 143) { row[i] = (uint16_t)(s16 | 0x7c00); }
            else if (e32 <= 112) { row[i] = (uint16_t)s16; }
            else { row[i] = (uint16_t)(s16 | ((e32 - 112) << 10) | ((m32 + 0x1000) >> 13)); }
        }
    }

    return true;
}

bool dit_run_layernorm(void* in_fp32, void* out_fp32, int _M, int _D, float eps) {
    // Single LayerNorm dispatch: upload FP32 input, run shader, download FP32 output.
    // Uses cmd[0] — one-shot recording, submit, wait, read back.
    if (!g_init) return false;

    uint32_t Mv = (uint32_t)_M;
    uint32_t Dv = (uint32_t)_D;
    size_t dataBytes = Mv * Dv * 4;  // float32 = 4 bytes

    // Upload input to g_lnInBuf
    memcpy(g_lnInBuf.mapped, in_fp32, dataBytes);

    // Record into dedicated LN command buffer (don't overwrite AdaLN pre-recorded cmds)
    VkCommandBufferBeginInfo bi = {};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(g_lnCmdBuf, &bi) != VK_SUCCESS) return false;

    // Allocate descriptor set for LN
    VkDescriptorSetAllocateInfo dsInfo = {};
    dsInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsInfo.descriptorPool = g_vk.descPool;
    dsInfo.descriptorSetCount = 1;
    dsInfo.pSetLayouts = &g_vk.layer_norm.dsl;
    VkDescriptorSet ds;
    if (vkAllocateDescriptorSets(g_vk.device, &dsInfo, &ds) != VK_SUCCESS) {
        LOGE("LN: descriptor set alloc failed");
        return false;
    }

    // Bind input buffer
    VkDescriptorBufferInfo inInfo = { g_lnInBuf.buf, 0, dataBytes };
    VkWriteDescriptorSet wIn = {};
    wIn.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wIn.dstSet = ds;
    wIn.dstBinding = 0;
    wIn.descriptorCount = 1;
    wIn.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wIn.pBufferInfo = &inInfo;

    // Bind output buffer
    VkDescriptorBufferInfo outInfo = { g_lnOutBuf.buf, 0, dataBytes };
    VkWriteDescriptorSet wOut = {};
    wOut.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wOut.dstSet = ds;
    wOut.dstBinding = 1;
    wOut.descriptorCount = 1;
    wOut.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wOut.pBufferInfo = &outInfo;

    VkWriteDescriptorSet writes[] = { wIn, wOut };
    vkUpdateDescriptorSets(g_vk.device, 2, writes, 0, nullptr);

    // Record dispatch
    vkCmdBindPipeline(g_lnCmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, g_vk.layer_norm.pipeline);
    vkCmdBindDescriptorSets(g_lnCmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
        g_vk.layer_norm.layout, 0, 1, &ds, 0, nullptr);

    PC_LayerNorm pc = { Mv, Dv, eps };
    vkCmdPushConstants(g_lnCmdBuf, g_vk.layer_norm.layout,
        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(g_lnCmdBuf, Mv, 1, 1);

    // Barrier: shader write → host read
    VkMemoryBarrier mb = {};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(g_lnCmdBuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);

    if (vkEndCommandBuffer(g_lnCmdBuf) != VK_SUCCESS) return false;

    // Submit and wait
    vkResetFences(g_vk.device, 1, &g_vk.fence);
    VkSubmitInfo submit = {};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &g_lnCmdBuf;
    if (vkQueueSubmit(g_vk.queue, 1, &submit, g_vk.fence) != VK_SUCCESS) return false;
    vkWaitForFences(g_vk.device, 1, &g_vk.fence, VK_TRUE, UINT64_MAX);

    // Download result
    memcpy(out_fp32, g_lnOutBuf.mapped, dataBytes);

    // Free descriptor set
    vkFreeDescriptorSets(g_vk.device, g_vk.descPool, 1, &ds);

    return true;
}

bool dit_run_attention(void* Q_fp16, void* K_fp16, void* V_fp16, void* O_fp16,
                        int _M_q, int _M_kv, int _H, int _D, float scale) {
    // 3-pass attention: QK^T → softmax → A@V. One cmd buffer, one submit.
    if (!g_init) return false;
    uint32_t M_q = (uint32_t)_M_q, M_kv = (uint32_t)_M_kv, H = (uint32_t)_H, D = (uint32_t)_D;

    // Upload Q/K/V
    size_t qBytes = M_q * H * D * 2, kvBytes = M_kv * H * D * 2;
    size_t aBytes = M_q * H * M_kv * 2;
    memcpy(g_attnQ.mapped, Q_fp16, qBytes);
    memcpy(g_attnK.mapped, K_fp16, kvBytes);
    memcpy(g_attnV.mapped, V_fp16, kvBytes);

    VkCommandBufferBeginInfo bi = {};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(g_lnCmdBuf, &bi) != VK_SUCCESS) return false;

    // ── Pass 1: QK^T ──
    VkDescriptorSetAllocateInfo dsa = {};
    dsa.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsa.descriptorPool = g_vk.descPool;
    dsa.descriptorSetCount = 1;

    VkDescriptorSet ds1;
    dsa.pSetLayouts = &g_vk.attn_qkt.dsl;
    vkAllocateDescriptorSets(g_vk.device, &dsa, &ds1);
    VkDescriptorBufferInfo bQ = {g_attnQ.buf, 0, qBytes};
    VkDescriptorBufferInfo bK = {g_attnK.buf, 0, kvBytes};
    VkDescriptorBufferInfo bA = {g_attnA.buf, 0, aBytes};
    VkWriteDescriptorSet w1[3] = {};
    for (int i=0;i<3;i++){w1[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;w1[i].dstSet=ds1;w1[i].dstBinding=(uint32_t)i;w1[i].descriptorCount=1;w1[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;}
    w1[0].pBufferInfo=&bQ; w1[1].pBufferInfo=&bK; w1[2].pBufferInfo=&bA;
    vkUpdateDescriptorSets(g_vk.device,3,w1,0,nullptr);
    vkCmdBindPipeline(g_lnCmdBuf,VK_PIPELINE_BIND_POINT_COMPUTE,g_vk.attn_qkt.pipeline);
    vkCmdBindDescriptorSets(g_lnCmdBuf,VK_PIPELINE_BIND_POINT_COMPUTE,g_vk.attn_qkt.layout,0,1,&ds1,0,nullptr);
    PC_AttnQKT pc1={M_q,M_kv,H,D,scale};
    vkCmdPushConstants(g_lnCmdBuf,g_vk.attn_qkt.layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(pc1),&pc1);
    vkCmdDispatch(g_lnCmdBuf, M_q*H, 1, 1);
    {
        VkMemoryBarrier mb={};
        mb.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(g_lnCmdBuf,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&mb,0,nullptr,0,nullptr);
    }

    // ── Pass 2: softmax in-place ──
    VkDescriptorSet ds2;
    dsa.pSetLayouts=&g_vk.attn_softmax.dsl;
    vkAllocateDescriptorSets(g_vk.device,&dsa,&ds2);
    {
        VkWriteDescriptorSet w2={};
        w2.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w2.dstSet=ds2; w2.dstBinding=0; w2.descriptorCount=1;
        w2.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w2.pBufferInfo=&bA;
        vkUpdateDescriptorSets(g_vk.device,1,&w2,0,nullptr);
    }
    vkCmdBindPipeline(g_lnCmdBuf,VK_PIPELINE_BIND_POINT_COMPUTE,g_vk.attn_softmax.pipeline);
    vkCmdBindDescriptorSets(g_lnCmdBuf,VK_PIPELINE_BIND_POINT_COMPUTE,g_vk.attn_softmax.layout,0,1,&ds2,0,nullptr);
    PC_AttnSoftmax pc2={M_q,M_kv,H};
    vkCmdPushConstants(g_lnCmdBuf,g_vk.attn_softmax.layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(pc2),&pc2);
    vkCmdDispatch(g_lnCmdBuf, M_q*H, 1, 1);
    {
        VkMemoryBarrier mb={};
        mb.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(g_lnCmdBuf,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&mb,0,nullptr,0,nullptr);
    }

    // ── Pass 3: A @ V ──
    VkDescriptorSet ds3;
    dsa.pSetLayouts=&g_vk.attn_out.dsl;
    vkAllocateDescriptorSets(g_vk.device,&dsa,&ds3);
    VkDescriptorBufferInfo bV={g_attnV.buf,0,kvBytes}, bO={g_attnO.buf,0,qBytes};
    VkWriteDescriptorSet w3[3] = {};
    for (int i=0;i<3;i++){w3[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;w3[i].dstSet=ds3;w3[i].dstBinding=(uint32_t)i;w3[i].descriptorCount=1;w3[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;}
    w3[0].pBufferInfo=&bA; w3[1].pBufferInfo=&bV; w3[2].pBufferInfo=&bO;
    vkUpdateDescriptorSets(g_vk.device,3,w3,0,nullptr);
    vkCmdBindPipeline(g_lnCmdBuf,VK_PIPELINE_BIND_POINT_COMPUTE,g_vk.attn_out.pipeline);
    vkCmdBindDescriptorSets(g_lnCmdBuf,VK_PIPELINE_BIND_POINT_COMPUTE,g_vk.attn_out.layout,0,1,&ds3,0,nullptr);
    PC_AttnOut pc3={M_q,M_kv,H,D};
    vkCmdPushConstants(g_lnCmdBuf,g_vk.attn_out.layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(pc3),&pc3);
    vkCmdDispatch(g_lnCmdBuf, M_q*H, 1, 1);
    {
        VkMemoryBarrier mb2={};
        mb2.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb2.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;
        mb2.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(g_lnCmdBuf,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&mb2,0,nullptr,0,nullptr);
    }

    if (vkEndCommandBuffer(g_lnCmdBuf) != VK_SUCCESS) return false;

    vkResetFences(g_vk.device, 1, &g_vk.fence);
    VkSubmitInfo submit = {};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &g_lnCmdBuf;
    if (vkQueueSubmit(g_vk.queue, 1, &submit, g_vk.fence) != VK_SUCCESS) return false;
    vkWaitForFences(g_vk.device, 1, &g_vk.fence, VK_TRUE, UINT64_MAX);

    memcpy(O_fp16, g_attnO.mapped, qBytes);
    vkFreeDescriptorSets(g_vk.device, g_vk.descPool, 1, &ds1);
    vkFreeDescriptorSets(g_vk.device, g_vk.descPool, 1, &ds2);
    vkFreeDescriptorSets(g_vk.device, g_vk.descPool, 1, &ds3);
    return true;
}

bool dit_run_gelu(void* in_fp16, void* out_fp16, int _N) {
    if (!g_init) return false;
    uint32_t N = (uint32_t)_N;
    size_t bytes = N * 2;
    memcpy(g_geluInBuf.mapped, in_fp16, bytes);

    VkCommandBufferBeginInfo bi = {};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(g_lnCmdBuf, &bi) != VK_SUCCESS) return false;

    VkDescriptorSetAllocateInfo dsInfo = {};
    dsInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsInfo.descriptorPool = g_vk.descPool;
    dsInfo.descriptorSetCount = 1;
    dsInfo.pSetLayouts = &g_vk.gelu.dsl;
    VkDescriptorSet ds;
    if (vkAllocateDescriptorSets(g_vk.device, &dsInfo, &ds) != VK_SUCCESS) return false;

    VkDescriptorBufferInfo bIn  = { g_geluInBuf.buf, 0, bytes };
    VkDescriptorBufferInfo bOut = { g_geluOutBuf.buf, 0, bytes };
    VkWriteDescriptorSet w[2] = {};
    w[0].sType = w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[0].dstSet = w[1].dstSet = ds;
    w[0].dstBinding = 0; w[1].dstBinding = 1;
    w[0].descriptorCount = w[1].descriptorCount = 1;
    w[0].descriptorType = w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[0].pBufferInfo = &bIn; w[1].pBufferInfo = &bOut;
    vkUpdateDescriptorSets(g_vk.device, 2, w, 0, nullptr);

    vkCmdBindPipeline(g_lnCmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, g_vk.gelu.pipeline);
    vkCmdBindDescriptorSets(g_lnCmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, g_vk.gelu.layout, 0, 1, &ds, 0, nullptr);
    PC_Silu pc = { N };
    vkCmdPushConstants(g_lnCmdBuf, g_vk.gelu.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(g_lnCmdBuf, (N + 255) / 256, 1, 1);

    VkMemoryBarrier mb = {};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(g_lnCmdBuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    if (vkEndCommandBuffer(g_lnCmdBuf) != VK_SUCCESS) return false;

    vkResetFences(g_vk.device, 1, &g_vk.fence);
    VkSubmitInfo submit = {};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &g_lnCmdBuf;
    if (vkQueueSubmit(g_vk.queue, 1, &submit, g_vk.fence) != VK_SUCCESS) return false;
    vkWaitForFences(g_vk.device, 1, &g_vk.fence, VK_TRUE, UINT64_MAX);

    memcpy(out_fp16, g_geluOutBuf.mapped, bytes);
    vkFreeDescriptorSets(g_vk.device, g_vk.descPool, 1, &ds);
    return true;
}

bool dit_run_rmsnorm(void* in_fp16, void* weight_fp16, int wlen, void* out_fp16, int _M, int _D, float eps) {
    // Single RMSNorm dispatch: FP16 I/O + weight. Uses g_lnCmdBuf (shared w/ LN, sequential).
    if (!g_init) return false;

    uint32_t Mv = (uint32_t)_M;
    uint32_t Dv = (uint32_t)_D;
    size_t inBytes = Mv * Dv * 2;       // fp16 = 2 bytes

    // Upload to dedicated RMSNorm buffers
    memcpy(g_rmsInBuf.mapped, in_fp16, inBytes);
    memcpy(g_rmsWgtBuf.mapped, weight_fp16, (size_t)wlen * 2);
    memset(g_rmsOutBuf.mapped, 0, inBytes);

    VkCommandBufferBeginInfo bi = {};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(g_lnCmdBuf, &bi) != VK_SUCCESS) return false;

    VkDescriptorSetAllocateInfo dsInfo = {};
    dsInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsInfo.descriptorPool = g_vk.descPool;
    dsInfo.descriptorSetCount = 1;
    dsInfo.pSetLayouts = &g_vk.rms_norm.dsl;
    VkDescriptorSet ds;
    if (vkAllocateDescriptorSets(g_vk.device, &dsInfo, &ds) != VK_SUCCESS) return false;

    VkDescriptorBufferInfo bIn  = { g_rmsInBuf.buf, 0, inBytes };
    VkDescriptorBufferInfo bW   = { g_rmsWgtBuf.buf, 0, (VkDeviceSize)wlen * 2 };
    VkDescriptorBufferInfo bOut = { g_rmsOutBuf.buf, 0, inBytes };

    VkWriteDescriptorSet w[3] = {};
    for (int i = 0; i < 3; i++) { w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i].dstSet = ds; w[i].dstBinding = (uint32_t)i; w[i].descriptorCount = 1; w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; }
    w[0].pBufferInfo = &bIn; w[1].pBufferInfo = &bW; w[2].pBufferInfo = &bOut;
    vkUpdateDescriptorSets(g_vk.device, 3, w, 0, nullptr);

    vkCmdBindPipeline(g_lnCmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, g_vk.rms_norm.pipeline);
    vkCmdBindDescriptorSets(g_lnCmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, g_vk.rms_norm.layout, 0, 1, &ds, 0, nullptr);
    PC_RmsNorm pc = { Mv, Dv, eps };
    vkCmdPushConstants(g_lnCmdBuf, g_vk.rms_norm.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(g_lnCmdBuf, Mv, 1, 1);

    VkMemoryBarrier mb = {};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(g_lnCmdBuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);

    if (vkEndCommandBuffer(g_lnCmdBuf) != VK_SUCCESS) return false;

    vkResetFences(g_vk.device, 1, &g_vk.fence);
    VkSubmitInfo submit = {};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &g_lnCmdBuf;
    if (vkQueueSubmit(g_vk.queue, 1, &submit, g_vk.fence) != VK_SUCCESS) return false;
    vkWaitForFences(g_vk.device, 1, &g_vk.fence, VK_TRUE, UINT64_MAX);

    memcpy(out_fp16, g_rmsOutBuf.mapped, inBytes);
    vkFreeDescriptorSets(g_vk.device, g_vk.descPool, 1, &ds);
    return true;
}

void dit_destroy() {
    auto free_buf = [&](Buffer& b) {
        if (b.mapped) vkUnmapMemory(g_vk.device, b.mem);
        if (b.buf) vkDestroyBuffer(g_vk.device, b.buf, nullptr);
        if (b.mem) vkFreeMemory(g_vk.device, b.mem, nullptr);
    };
    auto free_sp = [&](ShaderPipe& sp) {
        if (sp.pipeline) vkDestroyPipeline(g_vk.device, sp.pipeline, nullptr);
        if (sp.layout) vkDestroyPipelineLayout(g_vk.device, sp.layout, nullptr);
        if (sp.dsl) vkDestroyDescriptorSetLayout(g_vk.device, sp.dsl, nullptr);
        if (sp.shader) vkDestroyShaderModule(g_vk.device, sp.shader, nullptr);
    };
    free_sp(g_vk.gemm); free_sp(g_vk.rms_norm); free_sp(g_vk.layer_norm);
    free_sp(g_vk.silu); free_sp(g_vk.scale_shift); free_sp(g_vk.rope);
    free_sp(g_vk.attention); free_sp(g_vk.broadcast); free_sp(g_vk.gelu);
    free_sp(g_vk.attn_qkt); free_sp(g_vk.attn_softmax); free_sp(g_vk.attn_out);

    if (g_vk.descPool) vkDestroyDescriptorPool(g_vk.device, g_vk.descPool, nullptr);
    if (g_lnCmdBuf) vkFreeCommandBuffers(g_vk.device, g_vk.cmdPool, 1, &g_lnCmdBuf);
    if (g_vk.cmd[0]) vkFreeCommandBuffers(g_vk.device, g_vk.cmdPool, 28, g_vk.cmd);
    if (g_vk.cmdPool) vkDestroyCommandPool(g_vk.device, g_vk.cmdPool, nullptr);
    if (g_vk.fence) vkDestroyFence(g_vk.device, g_vk.fence, nullptr);

    // Free per-tensor weight buffers
    for (auto& kv : g_weights) free_buf(kv.second.buf);
    g_weights.clear();

    free_buf(g_xBuf); free_buf(g_tEmbBuf); free_buf(g_ctxBuf);
    free_buf(g_outBuf); free_buf(g_t1); free_buf(g_tQ); free_buf(g_tK);
    free_buf(g_tV); free_buf(g_tO); free_buf(g_rBuf); free_buf(g_aBuf);
    free_buf(g_nBuf); free_buf(g_gBuf); free_buf(g_bcBuf);
    free_buf(g_onesBuf); free_buf(g_loraBuf);
    free_buf(g_lnInBuf); free_buf(g_lnOutBuf);
    free_buf(g_geluInBuf); free_buf(g_geluOutBuf);
    free_buf(g_attnQ); free_buf(g_attnK); free_buf(g_attnV); free_buf(g_attnA); free_buf(g_attnO);
    free_buf(g_rmsInBuf); free_buf(g_rmsOutBuf); free_buf(g_rmsWgtBuf);

    if (g_vk.device) vkDestroyDevice(g_vk.device, nullptr);
    if (g_vk.instance) vkDestroyInstance(g_vk.instance, nullptr);
    g_init = false;
    LOGI("dit_destroy complete");
}

} // extern "C"
