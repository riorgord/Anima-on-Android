// libvk_hybrid.so — Generic Vulkan compute dispatch wrapper
// Single function per operation, Python ctypes friendly.
// Target: Snapdragon 8+ Gen 1 (Adreno 730), Android NDK.
#include <vulkan/vulkan.h>
#include <android/log.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>

#define LOG_TAG "VK_Hybrid"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// ============================================================
// Structs
// ============================================================
struct GpuBuffer {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    size_t size = 0;
    void* mapped = nullptr;
};

struct Pipeline {
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorSet descSet = VK_NULL_HANDLE;
    std::vector<GpuBuffer> bufs;
    int numBufs = 0;
    int pushSize = 0;
    bool valid = false;
};

// ============================================================
// Global Vulkan state
// ============================================================
static VkInstance        g_instance = VK_NULL_HANDLE;
static VkPhysicalDevice  g_physDev  = VK_NULL_HANDLE;
static VkDevice          g_device   = VK_NULL_HANDLE;
static VkQueue           g_queue    = VK_NULL_HANDLE;
static uint32_t          g_qfIndex  = 0;
static VkCommandPool     g_cmdPool  = VK_NULL_HANDLE;
static VkCommandBuffer   g_cmdBuf   = VK_NULL_HANDLE;
static VkFence           g_fence    = VK_NULL_HANDLE;
static VkDescriptorPool  g_descPool = VK_NULL_HANDLE;
static VkPhysicalDeviceMemoryProperties g_memProps = {};
static bool              g_init     = false;
static std::vector<Pipeline> g_pipes;
static VkQueryPool       g_queryPool = VK_NULL_HANDLE;
static float             g_timestampPeriod = 0.0f;
static double            g_lastGpuUs = 0.0;
static bool              g_hasTimestamps = false;

// ============================================================
// Helpers
// ============================================================
static uint32_t find_mem_type(uint32_t typeBits, VkMemoryPropertyFlags props) {
    for (uint32_t i = 0; i < g_memProps.memoryTypeCount; i++)
        if ((typeBits & (1u << i)) && (g_memProps.memoryTypes[i].propertyFlags & props) == props)
            return i;
    return ~0u;
}

static void free_buf(GpuBuffer& b) {
    if (b.mapped) vkUnmapMemory(g_device, b.mem);
    if (b.buf) vkDestroyBuffer(g_device, b.buf, nullptr);
    if (b.mem) vkFreeMemory(g_device, b.mem, nullptr);
    b.mapped = nullptr; b.buf = VK_NULL_HANDLE; b.mem = VK_NULL_HANDLE; b.size = 0;
}

static bool create_buf(size_t size, VkBufferUsageFlags usage, GpuBuffer& buf) {
    VkBufferCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = size; info.usage = usage; info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(g_device, &info, nullptr, &buf.buf) != VK_SUCCESS) return false;
    VkMemoryRequirements reqs;
    vkGetBufferMemoryRequirements(g_device, buf.buf, &reqs);
    VkMemoryAllocateInfo alloc = {};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = reqs.size;
    alloc.memoryTypeIndex = find_mem_type(reqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (alloc.memoryTypeIndex == ~0u) {
        alloc.memoryTypeIndex = find_mem_type(reqs.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
            VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    }
    if (vkAllocateMemory(g_device, &alloc, nullptr, &buf.mem) != VK_SUCCESS) return false;
    if (vkBindBufferMemory(g_device, buf.buf, buf.mem, 0) != VK_SUCCESS) return false;
    buf.size = size;
    vkMapMemory(g_device, buf.mem, 0, size, 0, &buf.mapped);
    return true;
}

static std::vector<uint32_t> load_spv(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { LOGE("Cannot open %s", path); return {}; }
    fseek(f, 0, SEEK_END); size_t sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint32_t> code(sz / 4);
    if (fread(code.data(), 1, sz, f) != sz) { fclose(f); return {}; }
    fclose(f); return code;
}

// ============================================================
// Public API
// ============================================================
extern "C" {

bool vk_hybrid_init(void) {
    if (g_init) return true;

    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "VK_Hybrid"; appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo instInfo = {};
    instInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instInfo.pApplicationInfo = &appInfo;
    if (vkCreateInstance(&instInfo, nullptr, &g_instance) != VK_SUCCESS) {
        LOGE("vkCreateInstance failed"); return false;
    }

    uint32_t dc = 0;
    vkEnumeratePhysicalDevices(g_instance, &dc, nullptr);
    std::vector<VkPhysicalDevice> devs(dc);
    vkEnumeratePhysicalDevices(g_instance, &dc, devs.data());
    for (auto d : devs) {
        VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(d, &props);
        LOGI("GPU: %s", props.deviceName);
        uint32_t qc; vkGetPhysicalDeviceQueueFamilyProperties(d, &qc, nullptr);
        std::vector<VkQueueFamilyProperties> qps(qc);
        vkGetPhysicalDeviceQueueFamilyProperties(d, &qc, qps.data());
        for (uint32_t i = 0; i < qc; i++)
            if (qps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { g_physDev = d; g_qfIndex = i; break; }
        if (g_physDev) break;
    }
    if (!g_physDev) { LOGE("No compute GPU"); return false; }
    vkGetPhysicalDeviceMemoryProperties(g_physDev, &g_memProps);

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qInfo = {};
    qInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qInfo.queueFamilyIndex = g_qfIndex; qInfo.queueCount = 1; qInfo.pQueuePriorities = &prio;
    VkDeviceCreateInfo devInfo = {};
    devInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    devInfo.queueCreateInfoCount = 1; devInfo.pQueueCreateInfos = &qInfo;
    if (vkCreateDevice(g_physDev, &devInfo, nullptr, &g_device) != VK_SUCCESS) {
        LOGE("vkCreateDevice failed"); return false;
    }
    vkGetDeviceQueue(g_device, g_qfIndex, 0, &g_queue);

    VkCommandPoolCreateInfo cpInfo = {};
    cpInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpInfo.queueFamilyIndex = g_qfIndex;
    if (vkCreateCommandPool(g_device, &cpInfo, nullptr, &g_cmdPool) != VK_SUCCESS) {
        LOGE("Command pool failed"); return false;
    }

    VkCommandBufferAllocateInfo cbInfo = {};
    cbInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbInfo.commandPool = g_cmdPool;
    cbInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbInfo.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(g_device, &cbInfo, &g_cmdBuf) != VK_SUCCESS) {
        LOGE("Command buffer alloc failed"); return false;
    }

    VkFenceCreateInfo fInfo = {};
    fInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (vkCreateFence(g_device, &fInfo, nullptr, &g_fence) != VK_SUCCESS) {
        LOGE("Fence creation failed"); return false;
    }

    VkDescriptorPoolSize ps = {};
    ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ps.descriptorCount = 256;
    VkDescriptorPoolCreateInfo dpInfo = {};
    dpInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpInfo.maxSets = 64;
    dpInfo.poolSizeCount = 1;
    dpInfo.pPoolSizes = &ps;
    if (vkCreateDescriptorPool(g_device, &dpInfo, nullptr, &g_descPool) != VK_SUCCESS) {
        LOGE("Descriptor pool failed"); return false;
    }

    // Timestamp query pool
    VkPhysicalDeviceProperties devProps;
    vkGetPhysicalDeviceProperties(g_physDev, &devProps);
    if (devProps.limits.timestampComputeAndGraphics) {
        g_timestampPeriod = devProps.limits.timestampPeriod;
        VkQueryPoolCreateInfo qpInfo = {};
        qpInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        qpInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpInfo.queryCount = 2;
        if (vkCreateQueryPool(g_device, &qpInfo, nullptr, &g_queryPool) == VK_SUCCESS) {
            g_hasTimestamps = true;
            LOGI("Timestamp queries enabled (period=%.1f ns)", g_timestampPeriod);
        }
    }

    g_init = true;
    LOGI("vk_hybrid_init OK");
    return true;
}

int vk_hybrid_load(const char* spv_path, int num_bufs, int push_sz) {
    if (!g_init) { LOGE("Not initialized"); return -1; }

    Pipeline p;
    p.numBufs = num_bufs;
    p.pushSize = push_sz;

    // Load SPIR-V
    auto code = load_spv(spv_path);
    if (code.empty()) return -1;

    VkShaderModuleCreateInfo smInfo = {};
    smInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smInfo.codeSize = code.size() * 4;
    smInfo.pCode = code.data();
    if (vkCreateShaderModule(g_device, &smInfo, nullptr, &p.shader) != VK_SUCCESS) {
        LOGE("Shader module creation failed"); return -1;
    }

    // Descriptor set layout: all storage buffers
    std::vector<VkDescriptorSetLayoutBinding> bindings(num_bufs);
    for (int i = 0; i < num_bufs; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo dslInfo = {};
    dslInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslInfo.bindingCount = (uint32_t)num_bufs;
    dslInfo.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(g_device, &dslInfo, nullptr, &p.dsl) != VK_SUCCESS) {
        LOGE("DSL creation failed"); vkDestroyShaderModule(g_device, p.shader, nullptr); return -1;
    }

    // Pipeline layout
    VkPipelineLayoutCreateInfo plInfo = {};
    plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plInfo.setLayoutCount = 1; plInfo.pSetLayouts = &p.dsl;
    VkPushConstantRange pcRange = { VK_SHADER_STAGE_COMPUTE_BIT, 0, (uint32_t)push_sz };
    if (push_sz > 0) { plInfo.pushConstantRangeCount = 1; plInfo.pPushConstantRanges = &pcRange; }
    if (vkCreatePipelineLayout(g_device, &plInfo, nullptr, &p.layout) != VK_SUCCESS) {
        LOGE("Pipeline layout failed");
        vkDestroyDescriptorSetLayout(g_device, p.dsl, nullptr);
        vkDestroyShaderModule(g_device, p.shader, nullptr);
        return -1;
    }

    // Compute pipeline
    VkComputePipelineCreateInfo cpInfo = {};
    cpInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpInfo.stage.module = p.shader;
    cpInfo.stage.pName = "main";
    cpInfo.layout = p.layout;
    if (vkCreateComputePipelines(g_device, VK_NULL_HANDLE, 1, &cpInfo, nullptr, &p.pipeline) != VK_SUCCESS) {
        LOGE("Pipeline creation failed");
        vkDestroyPipelineLayout(g_device, p.layout, nullptr);
        vkDestroyDescriptorSetLayout(g_device, p.dsl, nullptr);
        vkDestroyShaderModule(g_device, p.shader, nullptr);
        return -1;
    }

    // Allocate descriptor set
    VkDescriptorSetAllocateInfo dsInfo = {};
    dsInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsInfo.descriptorPool = g_descPool;
    dsInfo.descriptorSetCount = 1;
    dsInfo.pSetLayouts = &p.dsl;
    if (vkAllocateDescriptorSets(g_device, &dsInfo, &p.descSet) != VK_SUCCESS) {
        LOGE("Descriptor set alloc failed");
        vkDestroyPipeline(g_device, p.pipeline, nullptr);
        vkDestroyPipelineLayout(g_device, p.layout, nullptr);
        vkDestroyDescriptorSetLayout(g_device, p.dsl, nullptr);
        vkDestroyShaderModule(g_device, p.shader, nullptr);
        return -1;
    }

    // Allocate buffers
    p.bufs.resize(num_bufs);

    p.valid = true;
    int handle = (int)g_pipes.size();
    g_pipes.push_back(std::move(p));
    LOGI("Loaded pipeline handle=%d, %d buffers, %dB push", handle, num_bufs, push_sz);
    return handle;
}

bool vk_hybrid_upload(int handle, int binding, void* data, size_t bytes) {
    if (handle < 0 || handle >= (int)g_pipes.size()) { LOGE("Bad handle"); return false; }
    auto& p = g_pipes[handle];
    if (!p.valid || binding < 0 || binding >= p.numBufs) { LOGE("Bad binding"); return false; }

    auto& buf = p.bufs[binding];
    if (buf.size < bytes) {
        free_buf(buf);
        if (!create_buf(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, buf)) {
            LOGE("Buffer alloc failed: %zu bytes", bytes); return false;
        }
    }
    if (data) memcpy(buf.mapped, data, bytes);

    // Update descriptor
    VkDescriptorBufferInfo bi = { buf.buf, 0, bytes > 0 ? bytes : VK_WHOLE_SIZE };
    VkWriteDescriptorSet w = {};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet = p.descSet;
    w.dstBinding = (uint32_t)binding;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w.pBufferInfo = &bi;
    vkUpdateDescriptorSets(g_device, 1, &w, 0, nullptr);

    return true;
}

bool vk_hybrid_download(int handle, int binding, void* out, size_t bytes) {
    if (handle < 0 || handle >= (int)g_pipes.size()) return false;
    auto& p = g_pipes[handle];
    if (!p.valid || binding < 0 || binding >= p.numBufs) return false;

    auto& buf = p.bufs[binding];
    if (!buf.mapped || buf.size < bytes) return false;
    memcpy(out, buf.mapped, bytes);
    return true;
}

bool vk_hybrid_run(int handle, uint32_t dx, uint32_t dy, uint32_t dz, void* push_data) {
    if (handle < 0 || handle >= (int)g_pipes.size()) return false;
    auto& p = g_pipes[handle];
    if (!p.valid) return false;

    // Record command buffer
    VkCommandBufferBeginInfo begin = {};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(g_cmdBuf, &begin) != VK_SUCCESS) return false;

    // Timestamp: reset query pool
    if (g_hasTimestamps)
        vkCmdResetQueryPool(g_cmdBuf, g_queryPool, 0, 2);

    vkCmdBindPipeline(g_cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
    vkCmdBindDescriptorSets(g_cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &p.descSet, 0, nullptr);
    if (p.pushSize > 0 && push_data)
        vkCmdPushConstants(g_cmdBuf, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, (uint32_t)p.pushSize, push_data);

    // Timestamp: before dispatch
    if (g_hasTimestamps)
        vkCmdWriteTimestamp(g_cmdBuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, g_queryPool, 0);

    vkCmdDispatch(g_cmdBuf, dx, dy, dz);

    // Timestamp: after dispatch
    if (g_hasTimestamps)
        vkCmdWriteTimestamp(g_cmdBuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, g_queryPool, 1);

    // Memory barrier
    VkMemoryBarrier mb = {};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(g_cmdBuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);

    if (vkEndCommandBuffer(g_cmdBuf) != VK_SUCCESS) return false;

    // Submit and wait
    vkResetFences(g_device, 1, &g_fence);
    VkSubmitInfo submit = {};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &g_cmdBuf;
    if (vkQueueSubmit(g_queue, 1, &submit, g_fence) != VK_SUCCESS) return false;
    if (vkWaitForFences(g_device, 1, &g_fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) return false;

    // Read GPU timestamps
    g_lastGpuUs = -1.0;
    if (g_hasTimestamps) {
        uint64_t ts[2] = {0, 0};
        if (vkGetQueryPoolResults(g_device, g_queryPool, 0, 2,
                sizeof(ts), ts, sizeof(uint64_t),
                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS) {
            if (ts[1] > ts[0])
                g_lastGpuUs = (double)(ts[1] - ts[0]) * g_timestampPeriod * 1e-3;  // ns → us
        }
    }

    return true;
}

double vk_hybrid_last_gpu_us(void) {
    return g_lastGpuUs;
}

void vk_hybrid_destroy(void) {
    for (auto& p : g_pipes) {
        for (auto& b : p.bufs) free_buf(b);
        if (p.descSet)   vkFreeDescriptorSets(g_device, g_descPool, 1, &p.descSet);
        if (p.pipeline)  vkDestroyPipeline(g_device, p.pipeline, nullptr);
        if (p.layout)    vkDestroyPipelineLayout(g_device, p.layout, nullptr);
        if (p.dsl)       vkDestroyDescriptorSetLayout(g_device, p.dsl, nullptr);
        if (p.shader)    vkDestroyShaderModule(g_device, p.shader, nullptr);
    }
    g_pipes.clear();

    if (g_queryPool) vkDestroyQueryPool(g_device, g_queryPool, nullptr);
    if (g_fence)    vkDestroyFence(g_device, g_fence, nullptr);
    if (g_cmdBuf)   vkFreeCommandBuffers(g_device, g_cmdPool, 1, &g_cmdBuf);
    if (g_cmdPool)  vkDestroyCommandPool(g_device, g_cmdPool, nullptr);
    if (g_descPool) vkDestroyDescriptorPool(g_device, g_descPool, nullptr);
    if (g_device)   vkDestroyDevice(g_device, nullptr);
    if (g_instance) vkDestroyInstance(g_instance, nullptr);

    g_init = false;
    LOGI("vk_hybrid_destroy complete");
}

} // extern "C"
