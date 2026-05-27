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

    ShaderPipe gemm, rms_norm, layer_norm, silu, scale_shift, rope, attention, broadcast;
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
    poolSize.descriptorCount = 6000;
    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 1400;
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
// Command buffer recording context
// ============================================================
struct RC {
    VulkanCtx* vk;
    VkCommandBuffer cmd;
    std::unordered_map<std::string, WeightInfo>* weights;
    Buffer *xBuf, *tEmbBuf, *ctxBuf, *outBuf;
    Buffer *t1, *tQ, *tK, *tV, *tO, *rBuf, *aBuf, *nBuf, *gBuf, *bcBuf;

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

    void dispatch_gemm(Buffer& A, const char* wname, uint32_t Mv, uint32_t Nv, uint32_t Kv, Buffer& C) {
        auto it = weights->find(wname);
        if (it == weights->end()) { LOGE("Weight not found: %s", wname); return; }
        Buffer& wbuf = it->second.buf;
        auto& sp = vk->gemm;
        auto ds = alloc_set(sp.dsl);
        bind_buf(ds, 0, A, 0, VK_WHOLE_SIZE);
        bind_buf(ds, 1, wbuf, 0, VK_WHOLE_SIZE);
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

    void dispatch_broadcast(Buffer& in, Buffer& out, uint32_t Mv, uint32_t Dv, uint32_t rpt) {
        auto& sp = vk->broadcast;
        auto ds = alloc_set(sp.dsl);
        bind_buf(ds, 0, in, 0, VK_WHOLE_SIZE);
        bind_buf(ds, 1, out, 0, VK_WHOLE_SIZE);
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
static Buffer g_xBuf, g_tEmbBuf, g_ctxBuf, g_outBuf;
static Buffer g_t1, g_tQ, g_tK, g_tV, g_tO, g_rBuf, g_aBuf, g_nBuf, g_gBuf, g_bcBuf;
static Buffer g_onesBuf;  // small buffer: [1.0f, 1.0f] for scale+1 trick
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

    // Ones buffer: two fp16(1.0) values for scale+1 trick
    if (!create_buffer(g_vk, 4, u, g_onesBuf)) return false;
    uint16_t ones[2] = { 0x3C00 /*fp16 1.0*/, 0x3C00 };
    memcpy(g_onesBuf.mapped, ones, 4);

    LOGI("dit_init OK — %u buffers allocated", 15);
    g_init = true;
    return true;
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
    rc.rBuf = &g_rBuf; rc.aBuf = &g_aBuf; rc.nBuf = &g_nBuf; rc.gBuf = &g_gBuf; rc.bcBuf = &g_bcBuf;

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
    rc.rBuf = &g_rBuf; rc.aBuf = &g_aBuf; rc.nBuf = &g_nBuf; rc.gBuf = &g_gBuf; rc.bcBuf = &g_bcBuf;

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
    rc.rBuf = &g_rBuf; rc.aBuf = &g_aBuf; rc.nBuf = &g_nBuf; rc.gBuf = &g_gBuf; rc.bcBuf = &g_bcBuf;

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
    rc.rBuf = &g_rBuf; rc.aBuf = &g_aBuf; rc.nBuf = &g_nBuf; rc.gBuf = &g_gBuf; rc.bcBuf = &g_bcBuf;

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
    rc.rBuf = &g_rBuf; rc.aBuf = &g_aBuf; rc.nBuf = &g_nBuf; rc.gBuf = &g_gBuf; rc.bcBuf = &g_bcBuf;

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
    rc.rBuf = &g_rBuf; rc.aBuf = &g_aBuf; rc.nBuf = &g_nBuf; rc.gBuf = &g_gBuf; rc.bcBuf = &g_bcBuf;

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
    rc.rBuf = &g_rBuf; rc.aBuf = &g_aBuf; rc.nBuf = &g_nBuf; rc.gBuf = &g_gBuf; rc.bcBuf = &g_bcBuf;

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
    rc.rBuf=&g_rBuf; rc.aBuf=&g_aBuf; rc.nBuf=&g_nBuf; rc.gBuf=&g_gBuf; rc.bcBuf=&g_bcBuf;

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
    auto off = [&](int comp) -> size_t {
        return (size_t)comp * MS * D * 2;
    };
    uint32_t ph = MS * N_HEADS;
    char qw[128],kw[128],vw[128],ow[128],qnw[128],knw[128],l1w[128],l2w[128];
    snprintf(qw,sizeof(qw),"blocks.%d.self_attn.q_proj.weight",b);
    snprintf(kw,sizeof(kw),"blocks.%d.self_attn.k_proj.weight",b);
    snprintf(vw,sizeof(vw),"blocks.%d.self_attn.v_proj.weight",b);
    snprintf(ow,sizeof(ow),"blocks.%d.self_attn.output_proj.weight",b);
    snprintf(qnw,sizeof(qnw),"blocks.%d.self_attn.q_norm.weight",b);
    snprintf(knw,sizeof(knw),"blocks.%d.self_attn.k_norm.weight",b);
    snprintf(l1w,sizeof(l1w),"blocks.%d.mlp.layer1.weight",b);
    snprintf(l2w,sizeof(l2w),"blocks.%d.mlp.layer2.weight",b);

    // Self-attn: LN→AdaLN→QKV→norms→V→O→gate+residual → tV (using V as "attn out")
    rc.dispatch_layernorm(inBuf, *rc.nBuf, MS, D, 1e-6f);
    rc.dispatch_scale_shift(*rc.nBuf,*rc.bcBuf,off(0),*rc.bcBuf,off(1),*rc.nBuf, MS*D,1,1);
    rc.dispatch_gemm(*rc.nBuf,qw,MS,D,D,*rc.tQ);
    rc.dispatch_gemm(*rc.nBuf,kw,MS,D,D,*rc.tK);
    rc.dispatch_gemm(*rc.nBuf,vw,MS,D,D,*rc.tV);
    rc.dispatch_rmsnorm(*rc.tQ,qnw,*rc.tQ,ph,HEAD_DIM,1e-6f);
    rc.dispatch_rmsnorm(*rc.tK,knw,*rc.tK,ph,HEAD_DIM,1e-6f);
    rc.dispatch_gemm(*rc.tV,ow,MS,D,D,*rc.tO);                   // tO = V@Wo
    rc.dispatch_scale_shift(*rc.tO,*rc.bcBuf,off(2),inBuf,0,
                            *rc.tV, MS*D,1,1);                    // tV = x + gate*O

    // MLP: LN→AdaLN→fc1→SiLU→fc2→gate+residual
    rc.dispatch_layernorm(*rc.tV,*rc.nBuf,MS,D,1e-6f);
    rc.dispatch_scale_shift(*rc.nBuf,*rc.bcBuf,off(6),*rc.bcBuf,off(7),*rc.nBuf, MS*D,1,1);
    rc.dispatch_gemm(*rc.nBuf,l1w,MS,MLP_HIDDEN,D,*rc.t1);
    rc.dispatch_silu(*rc.t1,*rc.t1,MS*MLP_HIDDEN);
    rc.dispatch_gemm(*rc.t1,l2w,MS,D,MLP_HIDDEN,*rc.nBuf);
    rc.dispatch_scale_shift(*rc.nBuf,*rc.bcBuf,off(8),*rc.tV,0,
                            outBuf, MS*D,1,1);                    // out = x + gate*mlp
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
    rc.rBuf=&g_rBuf; rc.aBuf=&g_aBuf; rc.nBuf=&g_nBuf; rc.gBuf=&g_gBuf; rc.bcBuf=&g_bcBuf;

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
    rc.rBuf=&g_rBuf; rc.aBuf=&g_aBuf; rc.nBuf=&g_nBuf; rc.gBuf=&g_gBuf; rc.bcBuf=&g_bcBuf;

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

bool dit_forward_28blocks(void* x_data, void* adaln_all, void* out_data,
                           int _MS, int _D, int _M) {
    // adaln_all: 28 blocks × 9 comps × [MS, D] fp16 = 504MB
    // For each block: upload its AdaLN to bcBuf, submit cmd[i], copy out→x
    if (!g_init) return false;
    MS = (uint32_t)_MS; M = (uint32_t)_M; S = MS / M;
    size_t xBytes = MS * D * 2;
    size_t adalnPerBlock = 9 * MS * D * 2;

    memcpy(g_xBuf.mapped, x_data, xBytes);

    uint8_t* adaln = (uint8_t*)adaln_all;
    for (int i = 0; i < 28; i++) {
        // Upload this block's AdaLN data
        memcpy(g_bcBuf.mapped, adaln + i * adalnPerBlock, adalnPerBlock);

        // Submit
        vkResetFences(g_vk.device, 1, &g_vk.fence);
        VkSubmitInfo submit = {};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &g_vk.cmd[i];
        if (vkQueueSubmit(g_vk.queue, 1, &submit, g_vk.fence) != VK_SUCCESS) {
            LOGE("Submit failed at block %d", i); return false;
        }
        vkWaitForFences(g_vk.device, 1, &g_vk.fence, VK_TRUE, UINT64_MAX);

        // Copy output → xBuf for next block
        memcpy(g_xBuf.mapped, g_outBuf.mapped, xBytes);
    }

    memcpy(out_data, g_outBuf.mapped, xBytes);
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
    free_sp(g_vk.attention); free_sp(g_vk.broadcast);

    if (g_vk.descPool) vkDestroyDescriptorPool(g_vk.device, g_vk.descPool, nullptr);
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

    if (g_vk.device) vkDestroyDevice(g_vk.device, nullptr);
    if (g_vk.instance) vkDestroyInstance(g_vk.instance, nullptr);
    g_init = false;
    LOGI("dit_destroy complete");
}

} // extern "C"
