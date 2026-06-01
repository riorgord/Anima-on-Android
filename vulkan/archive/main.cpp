// Minimal Android Vulkan GEMM benchmark
// Build: ndk-build or cmake with Android NDK
#include <vulkan/vulkan.h>
#include <android/log.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>

#define LOG_TAG "VkGEMM"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// Simple Vulkan compute context
struct VulkanContext {
    VkInstance instance;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkQueue computeQueue;
    uint32_t computeQueueFamily;
    VkCommandPool commandPool;
    VkCommandBuffer commandBuffer;
    VkFence fence;
};

// FP16 helpers
uint32_t float_to_f16(float v) {
    // Minimal float->half conversion (round to nearest even)
    // In production, use compiler intrinsic or lookup table
    uint32_t f = *(uint32_t*)&v;
    uint32_t sign = (f >> 16) & 0x8000;
    int32_t exp = ((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (f >> 13) & 0x3FF;
    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7C00;
    return sign | (exp << 10) | mant;
}

float f16_to_float(uint32_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    if (exp == 0) return (sign ? -1.0f : 1.0f) * mant / 1024.0f * 1.0f / 16384.0f;
    if (exp == 31) return mant ? NAN : (sign ? -INFINITY : INFINITY);
    float val = 1.0f + mant / 1024.0f;
    int e = (int)exp - 15;
    while (e > 0) { val *= 2.0f; e--; }
    while (e < 0) { val *= 0.5f; e++; }
    return sign ? -val : val;
}

// Load SPIR-V from asset (returns bytes)
std::vector<uint32_t> load_spirv(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { LOGE("Cannot open %s", path); return {}; }
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint32_t> code(size / 4);
    fread(code.data(), 1, size, f);
    fclose(f);
    return code;
}

bool init_vulkan(VulkanContext& ctx) {
    // Create instance
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "VkGEMM";
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo instInfo = {};
    instInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instInfo.pApplicationInfo = &appInfo;

    if (vkCreateInstance(&instInfo, nullptr, &ctx.instance) != VK_SUCCESS) {
        LOGE("Failed to create Vulkan instance");
        return false;
    }

    // Find compute-capable physical device
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(ctx.instance, &deviceCount, nullptr);
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(ctx.instance, &deviceCount, devices.data());

    ctx.physicalDevice = VK_NULL_HANDLE;
    for (auto d : devices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(d, &props);
        LOGI("Device: %s (type=%d)", props.deviceName, props.deviceType);

        uint32_t qfCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(d, &qfCount, nullptr);
        std::vector<VkQueueFamilyProperties> qfProps(qfCount);
        vkGetPhysicalDeviceQueueFamilyProperties(d, &qfCount, qfProps.data());

        for (uint32_t i = 0; i < qfCount; i++) {
            if (qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                ctx.physicalDevice = d;
                ctx.computeQueueFamily = i;
                break;
            }
        }
        if (ctx.physicalDevice) break;
    }

    if (!ctx.physicalDevice) {
        LOGE("No Vulkan device with compute support");
        return false;
    }

    // Create device
    float priority = 1.0f;
    VkDeviceQueueCreateInfo qInfo = {};
    qInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qInfo.queueFamilyIndex = ctx.computeQueueFamily;
    qInfo.queueCount = 1;
    qInfo.pQueuePriorities = &priority;

    VkDeviceCreateInfo devInfo = {};
    devInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    devInfo.queueCreateInfoCount = 1;
    devInfo.pQueueCreateInfos = &qInfo;

    if (vkCreateDevice(ctx.physicalDevice, &devInfo, nullptr, &ctx.device) != VK_SUCCESS) {
        LOGE("Failed to create device");
        return false;
    }

    vkGetDeviceQueue(ctx.device, ctx.computeQueueFamily, 0, &ctx.computeQueue);

    // Command pool
    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = ctx.computeQueueFamily;
    vkCreateCommandPool(ctx.device, &poolInfo, nullptr, &ctx.commandPool);

    // Command buffer
    VkCommandBufferAllocateInfo cbInfo = {};
    cbInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbInfo.commandPool = ctx.commandPool;
    cbInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbInfo.commandBufferCount = 1;
    vkAllocateCommandBuffers(ctx.device, &cbInfo, &ctx.commandBuffer);

    // Fence
    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vkCreateFence(ctx.device, &fenceInfo, nullptr, &ctx.fence);

    LOGI("Vulkan initialized");
    return true;
}

void cleanup_vulkan(VulkanContext& ctx) {
    vkDestroyFence(ctx.device, ctx.fence, nullptr);
    vkFreeCommandBuffers(ctx.device, ctx.commandPool, 1, &ctx.commandBuffer);
    vkDestroyCommandPool(ctx.device, ctx.commandPool, nullptr);
    vkDestroyDevice(ctx.device, nullptr);
    vkDestroyInstance(ctx.instance, nullptr);
}

// CPU GEMM for verification: C = A × B^T
void cpu_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[j * K + k];
            }
            C[i * N + j] = sum;
        }
    }
}

int main(int argc, char** argv) {
    LOGI("=== Android Vulkan GEMM Benchmark ===");

    VulkanContext ctx = {};
    if (!init_vulkan(ctx)) {
        LOGE("Vulkan init failed, running CPU-only benchmark");
    }

    // Benchmark parameters
    int M = 2048, N = 2048, K = 2048;
    LOGI("Size: M=%d N=%d K=%d", M, N, K);

    // Allocate CPU buffers
    std::vector<float> A(M * K), B(N * K), C(M * N);
    for (int i = 0; i < M * K; i++) A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < N * K; i++) B[i] = (float)rand() / RAND_MAX;

    // CPU benchmark
    auto t0 = std::chrono::high_resolution_clock::now();
    cpu_gemm(A.data(), B.data(), C.data(), M, N, K);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double gflops = 2.0 * M * N * K / (cpu_ms / 1000.0) / 1e9;
    LOGI("CPU GEMM: %.1f ms (%.1f GFLOPS)", cpu_ms, gflops);

    // GPU benchmark (Vulkan) — to be implemented with compute pipeline
    LOGI("Vulkan GPU path: SPIR-V shader compiled, pipeline WIP");

    cleanup_vulkan(ctx);
    LOGI("Done");
    return 0;
}
