// Hybrid Vulkan Inference Engine
// Weight storage (BF16→FP16) + per-call GEMM/LN/RMSNorm/GELU dispatch.
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

#define LOG_TAG "Hybrid_VK"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// ── Model constants ──
static const uint32_t D          = 2048;
static const uint32_t MLP_HIDDEN = 8192;
static const uint32_t MS         = 512;  // B*T*H*W = 2*1*16*16

// ── Push constant structs (must match shaders) ──
struct PC_Gemm      { uint32_t M, N, K, batch; float alpha; };
struct PC_LayerNorm { uint32_t n_rows, n_elems; float eps; };
struct PC_RmsNorm   { uint32_t n_rows, n_elems; float eps; };
struct PC_Element   { uint32_t n_total; };

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
    VkCommandBuffer cmdBuf = VK_NULL_HANDLE;
    VkDescriptorPool descPool = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memProps = {};

    ShaderPipe gemm, layernorm, rmsnorm, gelu;
};

// ── Global state ──
static VKCtx g_ctx;
static std::unordered_map<std::string, Buffer> g_weights;
static bool g_init = false;
static bool g_finalized = false;

// Scratch buffers (allocated at finalize)
static Buffer g_gemmIn;     // M*K*2 fp16 (max 8MB)
static Buffer g_gemmOut;    // M*N*2 fp16 (max 8MB)
static Buffer g_lnIn;       // M*D*4 fp32 (max 4MB)
static Buffer g_lnOut;      // M*D*4 fp32 (max 4MB)
static Buffer g_rmsIn;      // M*D*2 fp16 (max 4MB)
static Buffer g_rmsOut;     // M*D*2 fp16 (max 4MB)
static Buffer g_rmsW;       // D*2 fp16 max weight (4KB)
static Buffer g_geluIn;     // N*2 fp16 (max 8MB)
static Buffer g_geluOut;    // N*2 fp16 (max 8MB)

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
    // Type 6: DEVICE_LOCAL + HOST_VISIBLE + HOST_COHERENT + HOST_CACHED
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
    if (vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &cpInfo, nullptr, &sp.pipeline) != VK_SUCCESS)
        return false;
    return true;
}

// ── BF16 → FP16 conversion ──
static inline uint16_t bf16_to_fp16(uint16_t bf16) {
    uint32_t sign = (bf16 >> 15) & 1;
    uint32_t exp  = (bf16 >> 7) & 0xFF;
    uint32_t mant = bf16 & 0x7F;

    if (exp == 0) return (uint16_t)(sign << 15);           // zero / subnormal → 0
    if (exp == 0xFF) {                                      // Inf / NaN
        if (mant == 0) return (uint16_t)((sign << 15) | (0x1F << 10));
        return (uint16_t)((sign << 15) | (0x1F << 10) | 1);
    }
    int fp16_exp = (int)exp - 112;                          // BF16 bias 127 → FP16 bias 15
    if (fp16_exp >= 31) return (uint16_t)((sign << 15) | (0x1E << 10) | 0x3FF);  // overflow → max
    if (fp16_exp <= 0) return (uint16_t)(sign << 15);       // underflow → 0
    uint32_t fp16_mant = mant << 3;                         // 7-bit → 10-bit mantissa
    return (uint16_t)((sign << 15) | (fp16_exp << 10) | fp16_mant);
}

// ============================================================
// Public API
// ============================================================
extern "C" {

bool vk_engine_init(void) {
    if (g_init) return true;

    // Instance
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "HybridEngine"; appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo instInfo = {};
    instInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instInfo.pApplicationInfo = &appInfo;
    if (vkCreateInstance(&instInfo, nullptr, &g_ctx.instance) != VK_SUCCESS) {
        LOGE("vkCreateInstance failed"); return false;
    }

    // Physical device
    uint32_t devCount = 0;
    vkEnumeratePhysicalDevices(g_ctx.instance, &devCount, nullptr);
    std::vector<VkPhysicalDevice> devs(devCount);
    vkEnumeratePhysicalDevices(g_ctx.instance, &devCount, devs.data());
    for (auto d : devs) {
        VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(d, &props);
        LOGI("GPU: %s", props.deviceName);
        uint32_t qfCount; vkGetPhysicalDeviceQueueFamilyProperties(d, &qfCount, nullptr);
        std::vector<VkQueueFamilyProperties> qfProps(qfCount);
        vkGetPhysicalDeviceQueueFamilyProperties(d, &qfCount, qfProps.data());
        for (uint32_t i = 0; i < qfCount; i++)
            if (qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
                { g_ctx.physDev = d; g_ctx.qFamily = i; break; }
        if (g_ctx.physDev) break;
    }
    if (!g_ctx.physDev) { LOGE("No compute GPU"); return false; }
    vkGetPhysicalDeviceMemoryProperties(g_ctx.physDev, &g_ctx.memProps);

    // Device
    float priority = 1.0f;
    VkDeviceQueueCreateInfo qInfo = {};
    qInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qInfo.queueFamilyIndex = g_ctx.qFamily; qInfo.queueCount = 1; qInfo.pQueuePriorities = &priority;
    VkDeviceCreateInfo devInfo = {};
    devInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    devInfo.queueCreateInfoCount = 1; devInfo.pQueueCreateInfos = &qInfo;
    if (vkCreateDevice(g_ctx.physDev, &devInfo, nullptr, &g_ctx.device) != VK_SUCCESS) {
        LOGE("vkCreateDevice failed"); return false;
    }
    vkGetDeviceQueue(g_ctx.device, g_ctx.qFamily, 0, &g_ctx.queue);

    // Command pool
    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = g_ctx.qFamily;
    if (vkCreateCommandPool(g_ctx.device, &poolInfo, nullptr, &g_ctx.cmdPool) != VK_SUCCESS) {
        LOGE("Command pool failed"); return false;
    }

    // Per-call command buffer
    VkCommandBufferAllocateInfo cbInfo = {};
    cbInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbInfo.commandPool = g_ctx.cmdPool;
    cbInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbInfo.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(g_ctx.device, &cbInfo, &g_ctx.cmdBuf) != VK_SUCCESS) {
        LOGE("Command buffer alloc failed"); return false;
    }

    // Fence
    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (vkCreateFence(g_ctx.device, &fenceInfo, nullptr, &g_ctx.fence) != VK_SUCCESS) {
        LOGE("Fence failed"); return false;
    }

    LOGI("Vulkan init OK");
    g_init = true;
    return true;
}

int vk_weight_add(const char* name, const void* data, int dtype,
                  const int* shape, int ndim) {
    if (!g_init) return -1;
    if (g_finalized) { LOGE("Cannot add weights after finalize"); return -2; }

    // Calculate total elements
    size_t elems = 1;
    for (int i = 0; i < ndim; i++) elems *= (size_t)shape[i];

    // Allocate Vulkan buffer (always FP16 storage = 2 bytes/elem)
    auto& w = g_weights[name];
    w.size = elems * 2;
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    if (!create_buf(g_ctx, w.size, usage, w)) {
        LOGE("Buffer alloc failed for %s (%.1f MB)", name, (double)w.size / 1e6);
        g_weights.erase(name);
        return -3;
    }

    uint16_t* dst = (uint16_t*)w.mapped;
    if (dtype == 2) {
        // BF16 → FP16 conversion
        const uint16_t* src = (const uint16_t*)data;
        for (size_t i = 0; i < elems; i++)
            dst[i] = bf16_to_fp16(src[i]);
    } else if (dtype == 1) {
        // Already FP16 — direct copy
        memcpy(dst, data, w.size);
    } else {
        // FP32 → FP16 conversion (unlikely for safetensors but handle it)
        const float* src = (const float*)data;
        for (size_t i = 0; i < elems; i++) {
            float v = src[i];
            uint32_t bits = *(uint32_t*)&v;
            uint32_t s16 = (bits >> 16) & 0x8000;
            uint32_t e32 = (bits >> 23) & 0xFF;
            uint32_t m32 = bits & 0x7FFFFF;
            if (e32 == 0) { dst[i] = (uint16_t)s16; }
            else if (e32 >= 143) { dst[i] = (uint16_t)(s16 | 0x7C00); }
            else if (e32 <= 112) { dst[i] = (uint16_t)s16; }
            else { dst[i] = (uint16_t)(s16 | ((e32 - 112) << 10) | ((m32 + 0x1000) >> 13)); }
        }
    }
    return 0;
}

bool vk_engine_finalize(void) {
    if (g_finalized) return true;
    if (!g_init) return false;

    LOGI("Finalizing: %zu weights, %.1f MB total",
         g_weights.size(),
         [&](){ size_t t=0; for(auto&kv:g_weights)t+=kv.second.size; return (double)t/1e6; }());

    // Descriptor pool (generous for per-call dispatch)
    VkDescriptorPoolSize ps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 12000 };
    VkDescriptorPoolCreateInfo dpInfo = {};
    dpInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpInfo.maxSets = 3000;
    dpInfo.poolSizeCount = 1;
    dpInfo.pPoolSizes = &ps;
    if (vkCreateDescriptorPool(g_ctx.device, &dpInfo, nullptr, &g_ctx.descPool) != VK_SUCCESS) {
        LOGE("Descriptor pool failed"); return false;
    }

    const char* spv_dir = "/data/local/tmp";

    // Create pipelines
    char path[256];
    #define CP(name, spv_file, nBindings, pushSize) \
        snprintf(path, sizeof(path), "%s/%s", spv_dir, spv_file); \
        if (!create_pipe(g_ctx, path, nBindings, pushSize, g_ctx.name)) { \
            LOGE("Pipeline %s failed", spv_file); return false; \
        }
    CP(gemm, "gemm_fp16.spv", 3, sizeof(PC_Gemm));
    CP(layernorm, "layernorm_fp32.spv", 2, sizeof(PC_LayerNorm));
    CP(rmsnorm, "rms_norm_fp16.spv", 3, sizeof(PC_RmsNorm));
    CP(gelu, "gelu_fp16.spv", 2, sizeof(PC_Element));
    #undef CP
    LOGI("All 4 pipelines created");

    // Scratch buffers
    VkBufferUsageFlags u = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    size_t gemmMax = MS * MLP_HIDDEN * 2;   // 8MB (M=512, K=8192 → fp16)
    size_t lnMax   = MS * D * 4;             // 4MB FP32
    size_t rmsMax  = MS * D * 2;             // 2MB FP16
    size_t geluMax = MS * MLP_HIDDEN * 2;    // 8MB

    if (!create_buf(g_ctx, gemmMax, u, g_gemmIn))  return false;
    if (!create_buf(g_ctx, gemmMax, u, g_gemmOut)) return false;
    if (!create_buf(g_ctx, lnMax, u, g_lnIn))      return false;
    if (!create_buf(g_ctx, lnMax, u, g_lnOut))     return false;
    if (!create_buf(g_ctx, rmsMax, u, g_rmsIn))    return false;
    if (!create_buf(g_ctx, rmsMax, u, g_rmsOut))   return false;
    if (!create_buf(g_ctx, D * 2, u, g_rmsW))      return false;
    if (!create_buf(g_ctx, geluMax, u, g_geluIn))  return false;
    if (!create_buf(g_ctx, geluMax, u, g_geluOut)) return false;

    LOGI("Scratch buffers allocated (%.1f MB total)",
         (double)(gemmMax*2 + lnMax*2 + rmsMax*2 + D*2 + geluMax*2) / 1e6);

    g_finalized = true;
    return true;
}

// ── Internal: one-shot dispatch helper ──
// Records dispatch into g_ctx.cmdBuf, submits, waits, returns.
static bool one_shot(ShaderPipe& sp, uint32_t nBindings,
                     VkDescriptorBufferInfo* bindInfos,
                     const void* pushData, size_t pushSize,
                     uint32_t gx, uint32_t gy, uint32_t gz) {
    // Begin
    VkCommandBufferBeginInfo bi = {};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(g_ctx.cmdBuf, &bi) != VK_SUCCESS) return false;

    // Allocate descriptor set
    VkDescriptorSetAllocateInfo dsa = {};
    dsa.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsa.descriptorPool = g_ctx.descPool;
    dsa.descriptorSetCount = 1;
    dsa.pSetLayouts = &sp.dsl;
    VkDescriptorSet ds;
    if (vkAllocateDescriptorSets(g_ctx.device, &dsa, &ds) != VK_SUCCESS) {
        vkEndCommandBuffer(g_ctx.cmdBuf);
        return false;
    }

    // Write bindings
    for (uint32_t i = 0; i < nBindings; i++) {
        VkWriteDescriptorSet w = {};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = ds; w.dstBinding = i; w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.pBufferInfo = &bindInfos[i];
        vkUpdateDescriptorSets(g_ctx.device, 1, &w, 0, nullptr);
    }

    // Bind pipeline + push constants + dispatch
    vkCmdBindPipeline(g_ctx.cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, sp.pipeline);
    vkCmdBindDescriptorSets(g_ctx.cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            sp.layout, 0, 1, &ds, 0, nullptr);
    if (pushSize > 0)
        vkCmdPushConstants(g_ctx.cmdBuf, sp.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, (uint32_t)pushSize, pushData);
    vkCmdDispatch(g_ctx.cmdBuf, gx, gy, gz);

    // Barrier: shader write → host read
    VkMemoryBarrier mb = {};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(g_ctx.cmdBuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);

    if (vkEndCommandBuffer(g_ctx.cmdBuf) != VK_SUCCESS) return false;

    // Submit + wait
    vkResetFences(g_ctx.device, 1, &g_ctx.fence);
    VkSubmitInfo submit = {};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &g_ctx.cmdBuf;
    if (vkQueueSubmit(g_ctx.queue, 1, &submit, g_ctx.fence) != VK_SUCCESS) {
        LOGE("Queue submit failed");
        return false;
    }
    if (vkWaitForFences(g_ctx.device, 1, &g_ctx.fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS)
        return false;

    return true;
}

// ── Reset descriptor pool (call between denoising steps) ──
bool vk_reset_pool(void) {
    if (!g_finalized) return false;
    vkResetDescriptorPool(g_ctx.device, g_ctx.descPool, 0);
    return true;
}

// ── GEMM: C[M,N] = A[M,K] @ weight_name[N,K]^T ──
bool vk_run_gemm(const char* weight_name, void* x_fp16, void* out_fp16,
                 int _M, int _N, int _K) {
    if (!g_finalized) return false;

    auto it = g_weights.find(weight_name);
    if (it == g_weights.end()) {
        LOGE("Weight not found: %s", weight_name);
        return false;
    }

    uint32_t Mv = (uint32_t)_M, Nv = (uint32_t)_N, Kv = (uint32_t)_K;
    size_t inBytes  = Mv * Kv * 2;  // fp16
    size_t outBytes = Mv * Nv * 2;

    // Upload input
    memcpy(g_gemmIn.mapped, x_fp16, inBytes);

    // Bind: A(input), B(weight), C(output)
    VkDescriptorBufferInfo bA = { g_gemmIn.buf, 0, inBytes };
    VkDescriptorBufferInfo bB = { it->second.buf, 0, it->second.size };
    VkDescriptorBufferInfo bC = { g_gemmOut.buf, 0, outBytes };
    VkDescriptorBufferInfo infos[3] = { bA, bB, bC };

    PC_Gemm pc = { Mv, Nv, Kv, 1, 1.0f };
    uint32_t gx = (Nv + 7) / 8;  // gemm_fp16 uses 8×8 workgroups
    uint32_t gy = (Mv + 7) / 8;
    if (!one_shot(g_ctx.gemm, 3, infos, &pc, sizeof(pc), gx, gy, 1))
        return false;

    // Download output
    memcpy(out_fp16, g_gemmOut.mapped, outBytes);
    return true;
}

// ── LayerNorm (FP32 I/O) ──
bool vk_run_layernorm(void* x_fp32, void* out_fp32, int _M, int _D, float eps) {
    if (!g_finalized) return false;
    uint32_t Mv = (uint32_t)_M, Dv = (uint32_t)_D;
    size_t bytes = Mv * Dv * 4;

    memcpy(g_lnIn.mapped, x_fp32, bytes);

    VkDescriptorBufferInfo bIn  = { g_lnIn.buf, 0, bytes };
    VkDescriptorBufferInfo bOut = { g_lnOut.buf, 0, bytes };
    VkDescriptorBufferInfo infos[2] = { bIn, bOut };

    PC_LayerNorm pc = { Mv, Dv, eps };
    if (!one_shot(g_ctx.layernorm, 2, infos, &pc, sizeof(pc), Mv, 1, 1))
        return false;

    memcpy(out_fp32, g_lnOut.mapped, bytes);
    return true;
}

// ── RMSNorm (FP16 I/O, with weight) ──
bool vk_run_rmsnorm(void* x_fp16, void* w_fp16, int wlen, void* out_fp16,
                    int _M, int _D, float eps) {
    if (!g_finalized) return false;
    uint32_t Mv = (uint32_t)_M, Dv = (uint32_t)_D;
    size_t inBytes = Mv * Dv * 2;
    size_t wBytes  = (size_t)wlen * 2;

    memcpy(g_rmsIn.mapped, x_fp16, inBytes);
    memcpy(g_rmsW.mapped, w_fp16, wBytes);

    VkDescriptorBufferInfo bIn  = { g_rmsIn.buf, 0, inBytes };
    VkDescriptorBufferInfo bW   = { g_rmsW.buf, 0, wBytes };
    VkDescriptorBufferInfo bOut = { g_rmsOut.buf, 0, inBytes };
    VkDescriptorBufferInfo infos[3] = { bIn, bW, bOut };

    PC_RmsNorm pc = { Mv, Dv, eps };
    if (!one_shot(g_ctx.rmsnorm, 3, infos, &pc, sizeof(pc), Mv, 1, 1))
        return false;

    memcpy(out_fp16, g_rmsOut.mapped, inBytes);
    return true;
}

// ── GELU (FP16 I/O, element-wise) ──
bool vk_run_gelu(void* x_fp16, void* out_fp16, int _N) {
    if (!g_finalized) return false;
    uint32_t n = (uint32_t)_N;
    size_t bytes = n * 2;

    memcpy(g_geluIn.mapped, x_fp16, bytes);

    VkDescriptorBufferInfo bIn  = { g_geluIn.buf, 0, bytes };
    VkDescriptorBufferInfo bOut = { g_geluOut.buf, 0, bytes };
    VkDescriptorBufferInfo infos[2] = { bIn, bOut };

    PC_Element pc = { n };
    if (!one_shot(g_ctx.gelu, 2, infos, &pc, sizeof(pc), (n + 255) / 256, 1, 1))
        return false;

    memcpy(out_fp16, g_geluOut.mapped, bytes);
    return true;
}

void vk_engine_destroy(void) {
    auto free_buf = [](Buffer& b) {
        if (b.mapped) vkUnmapMemory(g_ctx.device, b.mem);
        if (b.buf) vkDestroyBuffer(g_ctx.device, b.buf, nullptr);
        if (b.mem) vkFreeMemory(g_ctx.device, b.mem, nullptr);
    };
    auto free_sp = [](ShaderPipe& sp) {
        if (sp.pipeline) vkDestroyPipeline(g_ctx.device, sp.pipeline, nullptr);
        if (sp.layout) vkDestroyPipelineLayout(g_ctx.device, sp.layout, nullptr);
        if (sp.dsl) vkDestroyDescriptorSetLayout(g_ctx.device, sp.dsl, nullptr);
        if (sp.shader) vkDestroyShaderModule(g_ctx.device, sp.shader, nullptr);
    };

    // Free weight buffers
    for (auto& kv : g_weights) free_buf(kv.second);
    g_weights.clear();

    // Free pipelines
    free_sp(g_ctx.gemm); free_sp(g_ctx.layernorm);
    free_sp(g_ctx.rmsnorm); free_sp(g_ctx.gelu);

    // Free scratch buffers
    free_buf(g_gemmIn); free_buf(g_gemmOut);
    free_buf(g_lnIn); free_buf(g_lnOut);
    free_buf(g_rmsIn); free_buf(g_rmsOut); free_buf(g_rmsW);
    free_buf(g_geluIn); free_buf(g_geluOut);

    // Free Vulkan resources
    if (g_ctx.descPool) vkDestroyDescriptorPool(g_ctx.device, g_ctx.descPool, nullptr);
    if (g_ctx.cmdBuf) vkFreeCommandBuffers(g_ctx.device, g_ctx.cmdPool, 1, &g_ctx.cmdBuf);
    if (g_ctx.cmdPool) vkDestroyCommandPool(g_ctx.device, g_ctx.cmdPool, nullptr);
    if (g_ctx.fence) vkDestroyFence(g_ctx.device, g_ctx.fence, nullptr);
    if (g_ctx.device) vkDestroyDevice(g_ctx.device, nullptr);
    if (g_ctx.instance) vkDestroyInstance(g_ctx.instance, nullptr);

    g_init = false;
    g_finalized = false;
    LOGI("Engine destroyed");
}

} // extern "C"
