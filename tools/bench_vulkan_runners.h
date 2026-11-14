#pragma once

#include "bench_memory.h"
#include "bench_types.h"
#include "vk_minimal_loader.h"

#include <chrono>
#include <cstring>
#include <string>
#include <vector>

namespace nacelle_bench {

struct VkDeviceInfo {
    uint32_t index = 0;
    std::string backendName;
};

inline bool initVulkanInstance(vkmin::Loader& vk, vkmin::VkInstance& outInst, std::string& outErr) {
    if (!vk.loadLibrary()) {
        outErr = "SKIPPED (libvulkan not installed)";
        return false;
    }

    vkmin::VkApplicationInfo appInfo{};
    appInfo.sType = vkmin::VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "nacelle-bench";
    appInfo.apiVersion = (1u << 22) | (2u << 12);

    const char* portabilityExt = "VK_KHR_portability_enumeration";
    vkmin::VkInstanceCreateInfo ici{};
    ici.sType = vkmin::VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.flags = 0x00000001u;
    ici.pApplicationInfo = &appInfo;
    ici.enabledExtensionCount = 1;
    ici.ppEnabledExtensionNames = &portabilityExt;

    vkmin::VkResult instRes = vk.vkCreateInstance(&ici, nullptr, &outInst);
    if (instRes != vkmin::VK_SUCCESS) {
        ici.flags = 0;
        ici.enabledExtensionCount = 0;
        ici.ppEnabledExtensionNames = nullptr;
        instRes = vk.vkCreateInstance(&ici, nullptr, &outInst);
    }
    if (instRes != vkmin::VK_SUCCESS || !vk.loadInstanceSyms(outInst)) {
        if (outInst && vk.vkDestroyInstance) vk.vkDestroyInstance(outInst, nullptr);
        outInst = nullptr;
        outErr = instRes != vkmin::VK_SUCCESS
            ? ("SKIPPED (vkCreateInstance failed: " + std::to_string(instRes) + ")")
            : "SKIPPED (Vulkan instance symbols missing)";
        return false;
    }
    return true;
}

inline std::string queryVulkanBackendName(vkmin::Loader& vk, vkmin::VkPhysicalDevice pd) {
    vkmin::VkPhysicalDeviceProperties props{};
    std::string driverName;
    if (vk.vkGetPhysicalDeviceProperties2) {
        vkmin::VkPhysicalDeviceDriverProperties drvProps{};
        drvProps.sType = 1000196000u;
        vkmin::VkPhysicalDeviceProperties2 props2{};
        props2.sType = 1000059001u;
        props2.pNext = &drvProps;
        vk.vkGetPhysicalDeviceProperties2(pd, &props2);
        props = props2.properties;
        driverName = drvProps.driverName;
    } else {
        vk.vkGetPhysicalDeviceProperties(pd, &props);
    }
    std::string devName = props.deviceName;
    if (!driverName.empty() && devName.find(driverName) == std::string::npos) {
        return "Vulkan (" + driverName + "): " + devName;
    }
    return "Vulkan: " + devName;
}

inline std::vector<VkDeviceInfo> listVulkanDevices() {
    std::vector<VkDeviceInfo> out;
    vkmin::Loader vk;
    vkmin::VkInstance inst = nullptr;
    std::string err;
    if (!initVulkanInstance(vk, inst, err)) return out;

    uint32_t physCount = 0;
    vk.vkEnumeratePhysicalDevices(inst, &physCount, nullptr);
    std::vector<vkmin::VkPhysicalDevice> physDevs(physCount);
    if (physCount > 0) vk.vkEnumeratePhysicalDevices(inst, &physCount, physDevs.data());
    for (uint32_t i = 0; i < physCount; ++i) {
        out.push_back({i, queryVulkanBackendName(vk, physDevs[i])});
    }
    vk.vkDestroyInstance(inst, nullptr);
    return out;
}

inline std::vector<BenchRunResult> runVulkanComputeBench(
    const BenchmarkSpec& spec,
    const std::vector<uint8_t>& spv,
    uint32_t width,
    uint32_t height,
    uint32_t maxSteps,
    uint32_t iters,
    int32_t targetDeviceIdx = -1) {
    std::vector<BenchRunResult> results;
    vkmin::Loader vk;
    vkmin::VkInstance inst = nullptr;
    std::string err;
    if (!initVulkanInstance(vk, inst, err)) {
        BenchRunResult r;
        r.benchId = spec.id;
        r.benchName = spec.name;
        r.category = categoryName(spec.category);
        r.backend = "Vulkan (Lavapipe/SwiftShader)";
        r.shaderFormat = "SPIR-V";
        r.status = err;
        results.push_back(r);
        return results;
    }

    uint32_t physCount = 0;
    vk.vkEnumeratePhysicalDevices(inst, &physCount, nullptr);
    std::vector<vkmin::VkPhysicalDevice> physDevs(physCount);
    if (physCount > 0) vk.vkEnumeratePhysicalDevices(inst, &physCount, physDevs.data());

    for (uint32_t pdIdx = 0; pdIdx < physCount; ++pdIdx) {
        if (targetDeviceIdx >= 0 && static_cast<int32_t>(pdIdx) != targetDeviceIdx) continue;
        auto pd = physDevs[pdIdx];

        BenchRunResult r;
        r.benchId = spec.id;
        r.benchName = spec.name;
        r.category = categoryName(spec.category);
        r.backend = queryVulkanBackendName(vk, pd);
        r.shaderFormat = "SPIR-V";

        MemorySnapshot memBefore = queryProcessMemory();

        uint32_t qfCount = 0;
        vk.vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfCount, nullptr);
        std::vector<vkmin::VkQueueFamilyProperties> qfs(qfCount);
        vk.vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfCount, qfs.data());
        uint32_t computeQF = UINT32_MAX;
        for (uint32_t q = 0; q < qfCount; ++q) {
            if (qfs[q].queueFlags & vkmin::VK_QUEUE_COMPUTE_BIT) {
                computeQF = q;
                break;
            }
        }
        if (computeQF == UINT32_MAX) {
            r.status = "SKIPPED (no compute queue)";
            results.push_back(r);
            continue;
        }

        float prio = 1.0f;
        vkmin::VkDeviceQueueCreateInfo dqci{vkmin::VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0, computeQF, 1, &prio};
        vkmin::VkDeviceCreateInfo dci{vkmin::VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, nullptr, 0, 1, &dqci, 0, nullptr, 0, nullptr, nullptr};
        vkmin::VkDevice dev = nullptr;
        if (vk.vkCreateDevice(pd, &dci, nullptr, &dev) != vkmin::VK_SUCCESS) {
            r.status = "FAILED (vkCreateDevice)";
            results.push_back(r);
            continue;
        }
        vkmin::VkQueue queue = nullptr;
        vk.vkGetDeviceQueue(dev, computeQF, 0, &queue);

        auto t0 = std::chrono::steady_clock::now();
        vkmin::VkShaderModuleCreateInfo smci{vkmin::VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0, spv.size(), reinterpret_cast<const uint32_t*>(spv.data())};
        vkmin::VkShaderModule sm = 0;
        vk.vkCreateShaderModule(dev, &smci, nullptr, &sm);

        // Bindings matching DXC -fvk-t-shift 0 0 -fvk-u-shift 8 0 -fvk-b-shift 16 0
        vkmin::VkDescriptorSetLayoutBinding bindings[5]{};
        bindings[0] = {16, vkmin::VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, vkmin::VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[1] = {0, vkmin::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, vkmin::VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[2] = {1, vkmin::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, vkmin::VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[3] = {8, vkmin::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, vkmin::VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[4] = {9, vkmin::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, vkmin::VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

        vkmin::VkDescriptorSetLayoutCreateInfo dslci{vkmin::VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 5, bindings};
        vkmin::VkDescriptorSetLayout dsl = 0;
        vk.vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl);

        vkmin::VkPipelineLayoutCreateInfo plci{vkmin::VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &dsl, 0, nullptr};
        vkmin::VkPipelineLayout pl = 0;
        vk.vkCreatePipelineLayout(dev, &plci, nullptr, &pl);

        const char* entryName = spec.entryOrProfile.empty() ? "CSMain" : spec.entryOrProfile.c_str();
        vkmin::VkComputePipelineCreateInfo cpci{};
        cpci.sType = vkmin::VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage.sType = vkmin::VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpci.stage.stage = vkmin::VK_SHADER_STAGE_COMPUTE_BIT;
        cpci.stage.module = sm;
        cpci.stage.pName = entryName;
        cpci.layout = pl;
        vkmin::VkPipeline pipe = 0;
        vkmin::VkResult pipeRes = vk.vkCreateComputePipelines(dev, 0, 1, &cpci, nullptr, &pipe);
        auto t1 = std::chrono::steady_clock::now();
        r.compileMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        MemorySnapshot memAfterCompile = queryProcessMemory();

        if (pipeRes != vkmin::VK_SUCCESS || !pipe) {
            r.status = "FAILED (vkCreateComputePipelines)";
            vk.vkDestroyPipelineLayout(dev, pl, nullptr);
            vk.vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
            vk.vkDestroyShaderModule(dev, sm, nullptr);
            vk.vkDestroyDevice(dev, nullptr);
            results.push_back(r);
            continue;
        }

        vkmin::VkPhysicalDeviceMemoryProperties memProps{};
        vk.vkGetPhysicalDeviceMemoryProperties(pd, &memProps);
        auto findMem = [&](uint32_t bits) -> uint32_t {
            for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
                if ((bits & (1u << i)) &&
                    (memProps.memoryTypes[i].propertyFlags &
                     (vkmin::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vkmin::VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
                        (vkmin::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vkmin::VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
                    return i;
                }
            }
            return 0;
        };

        struct BufAlloc { vkmin::VkBuffer buf = 0; vkmin::VkDeviceMemory mem = 0; };
        auto mkBuf = [&](size_t sz, vkmin::VkFlags usage) -> BufAlloc {
            BufAlloc ba;
            vkmin::VkBufferCreateInfo bci{vkmin::VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, sz, usage, vkmin::VK_SHARING_MODE_EXCLUSIVE, 0, nullptr};
            vk.vkCreateBuffer(dev, &bci, nullptr, &ba.buf);
            vkmin::VkMemoryRequirements mr{};
            vk.vkGetBufferMemoryRequirements(dev, ba.buf, &mr);
            vkmin::VkMemoryAllocateInfo mai{vkmin::VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, mr.size, findMem(mr.memoryTypeBits)};
            vk.vkAllocateMemory(dev, &mai, nullptr, &ba.mem);
            vk.vkBindBufferMemory(dev, ba.buf, ba.mem, 0);
            return ba;
        };

        size_t maxOutBytes = 16 * 1024 * 1024;
        if (spec.id == "bench_raymarch") {
            maxOutBytes = std::max<size_t>(maxOutBytes, static_cast<size_t>(width * 2) * (height * 2) * 16);
        }

        auto cb = mkBuf(256, vkmin::VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        auto srv0 = mkBuf(4 * 1024 * 1024, vkmin::VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        auto srv1 = mkBuf(4 * 1024 * 1024, vkmin::VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        auto uav0 = mkBuf(maxOutBytes, vkmin::VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        auto uav1 = mkBuf(4 * 1024 * 1024, vkmin::VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

        void* ptr = nullptr;
        if (spec.id == "bench_raymarch") {
            vk.vkMapMemory(dev, srv0.mem, 0, 64, 0, &ptr);
            float spheres[16] = {
                0.0f, 0.2f, 0.5f, 0.8f,
                -1.3f, 0.0f, 1.0f, 0.6f,
                1.3f, 0.1f, 0.9f, 0.65f,
                0.0f, 1.1f, 1.5f, 0.5f,
            };
            std::memcpy(ptr, spheres, sizeof(spheres));
            vk.vkUnmapMemory(dev, srv0.mem);
        } else if (spec.id == "matmul") {
            vk.vkMapMemory(dev, srv0.mem, 0, 256 * 256 * 4, 0, &ptr);
            float* a = static_cast<float*>(ptr);
            for (size_t i = 0; i < 256 * 256; ++i) a[i] = static_cast<float>((i % 17) + 1) * 0.01f;
            vk.vkUnmapMemory(dev, srv0.mem);
            vk.vkMapMemory(dev, srv1.mem, 0, 256 * 256 * 4, 0, &ptr);
            float* b = static_cast<float*>(ptr);
            for (size_t i = 0; i < 256 * 256; ++i) b[i] = static_cast<float>((i % 13) + 1) * 0.01f;
            vk.vkUnmapMemory(dev, srv1.mem);
        } else if (spec.id == "prefix_sum_groupshared") {
            vk.vkMapMemory(dev, srv0.mem, 0, 65536 * 4 * 4, 0, &ptr);
            uint32_t* p = static_cast<uint32_t*>(ptr);
            for (size_t i = 0; i < 65536 * 4; ++i) p[i] = static_cast<uint32_t>((i & 63u) + 1u);
            vk.vkUnmapMemory(dev, srv0.mem);
        }

        vkmin::VkDescriptorPoolSize psz[2] = {
            {vkmin::VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
            {vkmin::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4},
        };
        vkmin::VkDescriptorPoolCreateInfo dpci{vkmin::VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, 1, 2, psz};
        vkmin::VkDescriptorPool dp = 0;
        vk.vkCreateDescriptorPool(dev, &dpci, nullptr, &dp);

        vkmin::VkDescriptorSetAllocateInfo dsai{vkmin::VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, dp, 1, &dsl};
        vkmin::VkDescriptorSet ds = 0;
        vk.vkAllocateDescriptorSets(dev, &dsai, &ds);

        vkmin::VkDescriptorBufferInfo bi[5] = {
            {cb.buf, 0, 256},
            {srv0.buf, 0, 4 * 1024 * 1024},
            {srv1.buf, 0, 4 * 1024 * 1024},
            {uav0.buf, 0, maxOutBytes},
            {uav1.buf, 0, 4 * 1024 * 1024},
        };
        vkmin::VkWriteDescriptorSet wds[5]{};
        wds[0] = {vkmin::VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 16, 0, 1, vkmin::VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &bi[0], nullptr};
        wds[1] = {vkmin::VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 0, 0, 1, vkmin::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &bi[1], nullptr};
        wds[2] = {vkmin::VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 1, 0, 1, vkmin::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &bi[2], nullptr};
        wds[3] = {vkmin::VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 8, 0, 1, vkmin::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &bi[3], nullptr};
        wds[4] = {vkmin::VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 9, 0, 1, vkmin::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &bi[4], nullptr};
        vk.vkUpdateDescriptorSets(dev, 5, wds, 0, nullptr);

        vkmin::VkCommandPoolCreateInfo cpci2{vkmin::VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr, vkmin::VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, computeQF};
        vkmin::VkCommandPool cpool = nullptr;
        vk.vkCreateCommandPool(dev, &cpci2, nullptr, &cpool);
        vkmin::VkCommandBufferAllocateInfo cbai{vkmin::VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, cpool, vkmin::VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
        vkmin::VkCommandBuffer cmdBuf = nullptr;
        vk.vkAllocateCommandBuffers(dev, &cbai, &cmdBuf);

        auto dispatchVK = [&](uint32_t gx, uint32_t gy, uint32_t cbW, uint32_t cbH, uint32_t cbSteps) -> double {
            vk.vkMapMemory(dev, cb.mem, 0, 256, 0, &ptr);
            if (spec.id == "bench_raymarch") {
                uint32_t vals[4] = {cbW, cbH, cbSteps, 0};
                std::memcpy(ptr, vals, sizeof(vals));
            } else if (spec.id == "matmul") {
                uint32_t vals[4] = {cbW, cbH, cbSteps, 0};
                std::memcpy(ptr, vals, sizeof(vals));
            }
            vk.vkUnmapMemory(dev, cb.mem);

            auto s0 = std::chrono::steady_clock::now();
            vk.vkResetCommandBuffer(cmdBuf, 0);
            vkmin::VkCommandBufferBeginInfo cbbi{vkmin::VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, vkmin::VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
            vk.vkBeginCommandBuffer(cmdBuf, &cbbi);
            vk.vkCmdBindPipeline(cmdBuf, vkmin::VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
            vk.vkCmdBindDescriptorSets(cmdBuf, vkmin::VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);
            vk.vkCmdDispatch(cmdBuf, gx, gy, 1);
            vk.vkEndCommandBuffer(cmdBuf);
            vkmin::VkSubmitInfo si{vkmin::VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cmdBuf, 0, nullptr};
            vk.vkQueueSubmit(queue, 1, &si, 0);
            vk.vkQueueWaitIdle(queue);
            auto s1 = std::chrono::steady_clock::now();
            return std::chrono::duration<double, std::milli>(s1 - s0).count();
        };

        auto getDims = [&](int mode, uint32_t& gx, uint32_t& gy, uint32_t& cbW, uint32_t& cbH, uint32_t& cbSteps) {
            if (spec.id == "bench_raymarch") {
                cbW = (mode == 0) ? 64 : (mode == 1 ? width : width * 2);
                cbH = (mode == 0) ? 64 : (mode == 1 ? height : height * 2);
                cbSteps = (mode == 0) ? 16 : (mode == 1 ? maxSteps : maxSteps * 2);
                gx = (cbW + 7) / 8;
                gy = (cbH + 7) / 8;
            } else if (spec.id == "matmul") {
                uint32_t n = (mode == 0) ? 64 : (mode == 1 ? 128 : 256);
                cbW = n; cbH = n; cbSteps = n;
                gx = (n + 7) / 8;
                gy = (n + 7) / 8;
            } else {
                gx = (mode == 0) ? 64 : (mode == 1 ? 512 : 4096);
                gy = 1;
                cbW = gx * 64; cbH = 1; cbSteps = 1;
            }
        };

        uint32_t gx = 1, gy = 1, cbW = 1, cbH = 1, cbSteps = 1;
        getDims(1, gx, gy, cbW, cbH, cbSteps);

        r.firstDispatchMs = dispatchVK(gx, gy, cbW, cbH, cbSteps);
        std::vector<double> samples;
        samples.reserve(iters);
        for (uint32_t i = 0; i < iters; ++i) {
            samples.push_back(dispatchVK(gx, gy, cbW, cbH, cbSteps));
        }

        double sum = 0.0;
        if (spec.id == "bench_raymarch") {
            vk.vkMapMemory(dev, uav0.mem, 0, static_cast<size_t>(width) * height * 16, 0, &ptr);
            const float* fp = static_cast<const float*>(ptr);
            for (size_t i = 0; i < static_cast<size_t>(width) * height * 4; ++i) sum += fp[i];
            vk.vkUnmapMemory(dev, uav0.mem);
        } else if (spec.id == "matmul") {
            vk.vkMapMemory(dev, uav0.mem, 0, 128 * 128 * 4, 0, &ptr);
            const float* fp = static_cast<const float*>(ptr);
            for (size_t i = 0; i < 128 * 128; ++i) sum += fp[i];
            vk.vkUnmapMemory(dev, uav0.mem);
        } else {
            vk.vkMapMemory(dev, uav0.mem, 0, 4096, 0, &ptr);
            const uint32_t* up = static_cast<const uint32_t*>(ptr);
            for (size_t i = 0; i < 1024; ++i) sum += static_cast<double>(up[i] & 0xFFFFu);
            vk.vkUnmapMemory(dev, uav0.mem);
        }
        r.checksum = sum;

        getDims(0, gx, gy, cbW, cbH, cbSteps);
        r.shortJobMs = dispatchVK(gx, gy, cbW, cbH, cbSteps);
        getDims(2, gx, gy, cbW, cbH, cbSteps);
        r.longJobMs = dispatchVK(gx, gy, cbW, cbH, cbSteps);

        finalizeTimingAndMemory(r, memBefore, memAfterCompile, samples);
        results.push_back(r);

        vk.vkDestroyCommandPool(dev, cpool, nullptr);
        vk.vkDestroyDescriptorPool(dev, dp, nullptr);
        for (auto& b : {cb, srv0, srv1, uav0, uav1}) {
            vk.vkDestroyBuffer(dev, b.buf, nullptr);
            vk.vkFreeMemory(dev, b.mem, nullptr);
        }
        vk.vkDestroyPipeline(dev, pipe, nullptr);
        vk.vkDestroyPipelineLayout(dev, pl, nullptr);
        vk.vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
        vk.vkDestroyShaderModule(dev, sm, nullptr);
        vk.vkDestroyDevice(dev, nullptr);
    }

    vk.vkDestroyInstance(inst, nullptr);
    return results;
}

inline std::vector<BenchRunResult> runVulkanGraphicsBench(
    const BenchmarkSpec& spec,
    const std::vector<uint8_t>& vsSpv,
    const std::vector<uint8_t>& psSpv,
    uint32_t width,
    uint32_t height,
    uint32_t iters,
    int32_t targetDeviceIdx = -1) {
    std::vector<BenchRunResult> results;
    vkmin::Loader vk;
    vkmin::VkInstance inst = nullptr;
    std::string err;
    if (!initVulkanInstance(vk, inst, err)) {
        return results;
    }

    uint32_t physCount = 0;
    vk.vkEnumeratePhysicalDevices(inst, &physCount, nullptr);
    std::vector<vkmin::VkPhysicalDevice> physDevs(physCount);
    if (physCount > 0) vk.vkEnumeratePhysicalDevices(inst, &physCount, physDevs.data());

    for (uint32_t pdIdx = 0; pdIdx < physCount; ++pdIdx) {
        if (targetDeviceIdx >= 0 && static_cast<int32_t>(pdIdx) != targetDeviceIdx) continue;
        auto pd = physDevs[pdIdx];

        BenchRunResult r;
        r.benchId = spec.id;
        r.benchName = spec.name;
        r.category = categoryName(spec.category);
        r.backend = queryVulkanBackendName(vk, pd);
        r.shaderFormat = "SPIR-V";

        MemorySnapshot memBefore = queryProcessMemory();

        uint32_t qfCount = 0;
        vk.vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfCount, nullptr);
        std::vector<vkmin::VkQueueFamilyProperties> qfs(qfCount);
        vk.vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfCount, qfs.data());
        uint32_t gfxQF = UINT32_MAX;
        for (uint32_t q = 0; q < qfCount; ++q) {
            if (qfs[q].queueFlags & vkmin::VK_QUEUE_GRAPHICS_BIT) {
                gfxQF = q;
                break;
            }
        }
        if (gfxQF == UINT32_MAX) {
            r.status = "SKIPPED (no graphics queue)";
            results.push_back(r);
            continue;
        }

        float prio = 1.0f;
        vkmin::VkDeviceQueueCreateInfo dqci{vkmin::VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0, gfxQF, 1, &prio};
        vkmin::VkDeviceCreateInfo dci{vkmin::VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, nullptr, 0, 1, &dqci, 0, nullptr, 0, nullptr, nullptr};
        vkmin::VkDevice dev = nullptr;
        if (vk.vkCreateDevice(pd, &dci, nullptr, &dev) != vkmin::VK_SUCCESS) {
            r.status = "FAILED (vkCreateDevice)";
            results.push_back(r);
            continue;
        }
        vkmin::VkQueue queue = nullptr;
        vk.vkGetDeviceQueue(dev, gfxQF, 0, &queue);

        vkmin::VkPhysicalDeviceMemoryProperties memProps{};
        vk.vkGetPhysicalDeviceMemoryProperties(pd, &memProps);
        auto findMem = [&](uint32_t bits, vkmin::VkFlags req) -> uint32_t {
            for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
                if ((bits & (1u << i)) && ((memProps.memoryTypes[i].propertyFlags & req) == req)) return i;
            }
            for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
                if (bits & (1u << i)) return i;
            }
            return 0;
        };

        uint32_t maxW = (spec.category == BenchCategory::Pixel) ? width * 2 : 128;
        uint32_t maxH = (spec.category == BenchCategory::Pixel) ? height * 2 : 128;

        // Render pass (clear + store color attachment)
        vkmin::VkAttachmentDescription colorAtt{};
        colorAtt.format = vkmin::VK_FORMAT_R8G8B8A8_UNORM;
        colorAtt.samples = vkmin::VK_SAMPLE_COUNT_1_BIT;
        colorAtt.loadOp = vkmin::VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAtt.storeOp = vkmin::VK_ATTACHMENT_STORE_OP_STORE;
        colorAtt.stencilLoadOp = vkmin::VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAtt.stencilStoreOp = vkmin::VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAtt.initialLayout = vkmin::VK_IMAGE_LAYOUT_UNDEFINED;
        colorAtt.finalLayout = vkmin::VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

        vkmin::VkAttachmentReference colorRef{0, vkmin::VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        vkmin::VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = vkmin::VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorRef;

        vkmin::VkRenderPassCreateInfo rpci{vkmin::VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO, nullptr, 0, 1, &colorAtt, 1, &subpass, 0, nullptr};
        vkmin::VkRenderPass rp = 0;
        vk.vkCreateRenderPass(dev, &rpci, nullptr, &rp);

        auto t0 = std::chrono::steady_clock::now();
        vkmin::VkShaderModuleCreateInfo vsSmci{vkmin::VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0, vsSpv.size(), reinterpret_cast<const uint32_t*>(vsSpv.data())};
        vkmin::VkShaderModuleCreateInfo psSmci{vkmin::VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0, psSpv.size(), reinterpret_cast<const uint32_t*>(psSpv.data())};
        vkmin::VkShaderModule vsMod = 0, psMod = 0;
        vk.vkCreateShaderModule(dev, &vsSmci, nullptr, &vsMod);
        vk.vkCreateShaderModule(dev, &psSmci, nullptr, &psMod);

        vkmin::VkDescriptorSetLayoutBinding uboBind{16, vkmin::VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, vkmin::VK_SHADER_STAGE_VERTEX_BIT | vkmin::VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        vkmin::VkDescriptorSetLayoutCreateInfo dslci{vkmin::VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 1, &uboBind};
        vkmin::VkDescriptorSetLayout dsl = 0;
        vk.vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl);

        vkmin::VkPipelineLayoutCreateInfo plci{vkmin::VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &dsl, 0, nullptr};
        vkmin::VkPipelineLayout pl = 0;
        vk.vkCreatePipelineLayout(dev, &plci, nullptr, &pl);

        vkmin::VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = vkmin::VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = vkmin::VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vsMod;
        stages[0].pName = "VSMain";
        stages[1].sType = vkmin::VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = vkmin::VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = psMod;
        stages[1].pName = "PSMain";

        bool usesQuadVB = (spec.category == BenchCategory::Pixel);
        vkmin::VkVertexInputBindingDescription vbd{0, 12, 0};
        vkmin::VkVertexInputAttributeDescription vad{0, 0, vkmin::VK_FORMAT_R32G32B32_SFLOAT, 0};
        vkmin::VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType = vkmin::VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        if (usesQuadVB) {
            vi.vertexBindingDescriptionCount = 1;
            vi.pVertexBindingDescriptions = &vbd;
            vi.vertexAttributeDescriptionCount = 1;
            vi.pVertexAttributeDescriptions = &vad;
        }

        vkmin::VkPipelineInputAssemblyStateCreateInfo ia{vkmin::VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, nullptr, 0, vkmin::VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0};
        vkmin::VkViewport vp{0.0f, 0.0f, static_cast<float>(maxW), static_cast<float>(maxH), 0.0f, 1.0f};
        vkmin::VkRect2D sc{{0, 0}, {maxW, maxH}};
        vkmin::VkPipelineViewportStateCreateInfo vps{vkmin::VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, nullptr, 0, 1, &vp, 1, &sc};
        vkmin::VkPipelineRasterizationStateCreateInfo rs{vkmin::VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, nullptr, 0, 0, 0, vkmin::VK_POLYGON_MODE_FILL, vkmin::VK_CULL_MODE_NONE, vkmin::VK_FRONT_FACE_COUNTER_CLOCKWISE, 0, 0, 0, 0, 1.0f};
        vkmin::VkPipelineMultisampleStateCreateInfo ms{vkmin::VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, nullptr, 0, vkmin::VK_SAMPLE_COUNT_1_BIT, 0, 1.0f, nullptr, 0, 0};
        vkmin::VkPipelineColorBlendAttachmentState cba{};
        cba.colorWriteMask = 0xF;
        vkmin::VkPipelineColorBlendStateCreateInfo cbState{vkmin::VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, nullptr, 0, 0, 0, 1, &cba, {0, 0, 0, 0}};

        vkmin::VkGraphicsPipelineCreateInfo gpci{};
        gpci.sType = vkmin::VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        gpci.stageCount = 2;
        gpci.pStages = stages;
        gpci.pVertexInputState = &vi;
        gpci.pInputAssemblyState = &ia;
        gpci.pViewportState = &vps;
        gpci.pRasterizationState = &rs;
        gpci.pMultisampleState = &ms;
        gpci.pColorBlendState = &cbState;
        gpci.layout = pl;
        gpci.renderPass = rp;
        gpci.subpass = 0;

        vkmin::VkPipeline pipe = 0;
        vkmin::VkResult pipeRes = vk.vkCreateGraphicsPipelines(dev, 0, 1, &gpci, nullptr, &pipe);
        auto t1 = std::chrono::steady_clock::now();
        r.compileMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        MemorySnapshot memAfterCompile = queryProcessMemory();

        if (pipeRes != vkmin::VK_SUCCESS || !pipe) {
            r.status = "FAILED (vkCreateGraphicsPipelines)";
            vk.vkDestroyPipelineLayout(dev, pl, nullptr);
            vk.vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
            vk.vkDestroyShaderModule(dev, psMod, nullptr);
            vk.vkDestroyShaderModule(dev, vsMod, nullptr);
            vk.vkDestroyRenderPass(dev, rp, nullptr);
            vk.vkDestroyDevice(dev, nullptr);
            results.push_back(r);
            continue;
        }

        // Image + ImageView + Framebuffer
        vkmin::VkImageCreateInfo ici{vkmin::VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, nullptr, 0, vkmin::VK_IMAGE_TYPE_2D, vkmin::VK_FORMAT_R8G8B8A8_UNORM, {maxW, maxH, 1}, 1, 1, vkmin::VK_SAMPLE_COUNT_1_BIT, vkmin::VK_IMAGE_TILING_OPTIMAL, vkmin::VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | vkmin::VK_IMAGE_USAGE_TRANSFER_SRC_BIT, vkmin::VK_SHARING_MODE_EXCLUSIVE, 0, nullptr, vkmin::VK_IMAGE_LAYOUT_UNDEFINED};
        vkmin::VkImage img = 0;
        vk.vkCreateImage(dev, &ici, nullptr, &img);
        vkmin::VkMemoryRequirements imr{};
        vk.vkGetImageMemoryRequirements(dev, img, &imr);
        vkmin::VkMemoryAllocateInfo imai{vkmin::VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, imr.size, findMem(imr.memoryTypeBits, vkmin::VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
        vkmin::VkDeviceMemory imgMem = 0;
        vk.vkAllocateMemory(dev, &imai, nullptr, &imgMem);
        vk.vkBindImageMemory(dev, img, imgMem, 0);

        vkmin::VkImageViewCreateInfo ivci{vkmin::VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, nullptr, 0, img, vkmin::VK_IMAGE_VIEW_TYPE_2D, vkmin::VK_FORMAT_R8G8B8A8_UNORM, {0, 0, 0, 0}, {vkmin::VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        vkmin::VkImageView iv = 0;
        vk.vkCreateImageView(dev, &ivci, nullptr, &iv);

        vkmin::VkFramebufferCreateInfo fbci{vkmin::VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, nullptr, 0, rp, 1, &iv, maxW, maxH, 1};
        vkmin::VkFramebuffer fb = 0;
        vk.vkCreateFramebuffer(dev, &fbci, nullptr, &fb);

        auto mkBuf = [&](size_t sz, vkmin::VkFlags usage) {
            struct { vkmin::VkBuffer buf = 0; vkmin::VkDeviceMemory mem = 0; } ba;
            vkmin::VkBufferCreateInfo bci{vkmin::VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, sz, usage, vkmin::VK_SHARING_MODE_EXCLUSIVE, 0, nullptr};
            vk.vkCreateBuffer(dev, &bci, nullptr, &ba.buf);
            vkmin::VkMemoryRequirements mr{};
            vk.vkGetBufferMemoryRequirements(dev, ba.buf, &mr);
            vkmin::VkMemoryAllocateInfo mai{vkmin::VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, mr.size, findMem(mr.memoryTypeBits, vkmin::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vkmin::VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
            vk.vkAllocateMemory(dev, &mai, nullptr, &ba.mem);
            vk.vkBindBufferMemory(dev, ba.buf, ba.mem, 0);
            return ba;
        };

        auto cb = mkBuf(256, vkmin::VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        auto vb = mkBuf(18 * sizeof(float), vkmin::VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        auto rb = mkBuf(static_cast<size_t>(maxW) * maxH * 4, vkmin::VK_BUFFER_USAGE_TRANSFER_DST_BIT);

        void* ptr = nullptr;
        vk.vkMapMemory(dev, vb.mem, 0, 18 * sizeof(float), 0, &ptr);
        const float quadVerts[18] = {
            -1.0f, -1.0f, 0.0f,  1.0f, -1.0f, 0.0f,  1.0f,  1.0f, 0.0f,
            -1.0f, -1.0f, 0.0f,  1.0f,  1.0f, 0.0f, -1.0f,  1.0f, 0.0f,
        };
        std::memcpy(ptr, quadVerts, sizeof(quadVerts));
        vk.vkUnmapMemory(dev, vb.mem);

        vkmin::VkDescriptorPoolSize psz{vkmin::VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1};
        vkmin::VkDescriptorPoolCreateInfo dpci{vkmin::VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, 1, 1, &psz};
        vkmin::VkDescriptorPool dp = 0;
        vk.vkCreateDescriptorPool(dev, &dpci, nullptr, &dp);
        vkmin::VkDescriptorSetAllocateInfo dsai{vkmin::VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, dp, 1, &dsl};
        vkmin::VkDescriptorSet ds = 0;
        vk.vkAllocateDescriptorSets(dev, &dsai, &ds);
        vkmin::VkDescriptorBufferInfo bi{cb.buf, 0, 256};
        vkmin::VkWriteDescriptorSet wds{vkmin::VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 16, 0, 1, vkmin::VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &bi, nullptr};
        vk.vkUpdateDescriptorSets(dev, 1, &wds, 0, nullptr);

        vkmin::VkCommandPoolCreateInfo cpci2{vkmin::VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr, vkmin::VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, gfxQF};
        vkmin::VkCommandPool cpool = nullptr;
        vk.vkCreateCommandPool(dev, &cpci2, nullptr, &cpool);
        vkmin::VkCommandBufferAllocateInfo cbai{vkmin::VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, cpool, vkmin::VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
        vkmin::VkCommandBuffer cmdBuf = nullptr;
        vk.vkAllocateCommandBuffers(dev, &cbai, &cmdBuf);

        auto drawVK = [&](uint32_t rw, uint32_t rh, uint32_t vc) -> double {
            vk.vkMapMemory(dev, cb.mem, 0, 256, 0, &ptr);
            if (spec.category == BenchCategory::Pixel) {
                float cbData[8] = {static_cast<float>(rw), static_cast<float>(rh), 1.0f, 1.25f, 0.0f, 0.0f, 0.0f, 0.0f};
                std::memcpy(ptr, cbData, sizeof(cbData));
            } else {
                struct { float mvp[16]; float time; uint32_t vc; float pad[2]; } vcb{};
                vcb.mvp[0] = 1.0f; vcb.mvp[5] = 1.0f; vcb.mvp[10] = 1.0f; vcb.mvp[15] = 1.0f;
                vcb.time = 0.5f;
                vcb.vc = vc;
                std::memcpy(ptr, &vcb, sizeof(vcb));
            }
            vk.vkUnmapMemory(dev, cb.mem);

            auto s0 = std::chrono::steady_clock::now();
            vk.vkResetCommandBuffer(cmdBuf, 0);
            vkmin::VkCommandBufferBeginInfo cbbi{vkmin::VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, vkmin::VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
            vk.vkBeginCommandBuffer(cmdBuf, &cbbi);
            vkmin::VkClearValue cv{};
            cv.float32[3] = 1.0f;
            vkmin::VkRenderPassBeginInfo rpbi{vkmin::VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, nullptr, rp, fb, {{0, 0}, {rw, rh}}, 1, &cv};
            vk.vkCmdBeginRenderPass(cmdBuf, &rpbi, vkmin::VK_SUBPASS_CONTENTS_INLINE);
            vk.vkCmdBindPipeline(cmdBuf, vkmin::VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
            vk.vkCmdBindDescriptorSets(cmdBuf, vkmin::VK_PIPELINE_BIND_POINT_GRAPHICS, pl, 0, 1, &ds, 0, nullptr);
            if (usesQuadVB) {
                vkmin::VkDeviceSize off = 0;
                vk.vkCmdBindVertexBuffers(cmdBuf, 0, 1, &vb.buf, &off);
            }
            vk.vkCmdDraw(cmdBuf, vc, 1, 0, 0);
            vk.vkCmdEndRenderPass(cmdBuf);
            vk.vkEndCommandBuffer(cmdBuf);
            vkmin::VkSubmitInfo si{vkmin::VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cmdBuf, 0, nullptr};
            vk.vkQueueSubmit(queue, 1, &si, 0);
            vk.vkQueueWaitIdle(queue);
            auto s1 = std::chrono::steady_clock::now();
            return std::chrono::duration<double, std::milli>(s1 - s0).count();
        };

        uint32_t rw = (spec.category == BenchCategory::Pixel) ? width : 128;
        uint32_t rh = (spec.category == BenchCategory::Pixel) ? height : 128;
        uint32_t vc = (spec.category == BenchCategory::Pixel) ? 6 : 24000;

        r.firstDispatchMs = drawVK(rw, rh, vc);
        std::vector<double> samples;
        samples.reserve(iters);
        for (uint32_t i = 0; i < iters; ++i) {
            samples.push_back(drawVK(rw, rh, vc));
        }

        // Copy image to readback for checksum
        vk.vkResetCommandBuffer(cmdBuf, 0);
        vkmin::VkCommandBufferBeginInfo cbbi{vkmin::VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, vkmin::VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
        vk.vkBeginCommandBuffer(cmdBuf, &cbbi);
        vkmin::VkBufferImageCopy bic{0, 0, 0, {vkmin::VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {rw, rh, 1}};
        vk.vkCmdCopyImageToBuffer(cmdBuf, img, vkmin::VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb.buf, 1, &bic);
        vk.vkEndCommandBuffer(cmdBuf);
        vkmin::VkSubmitInfo si{vkmin::VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cmdBuf, 0, nullptr};
        vk.vkQueueSubmit(queue, 1, &si, 0);
        vk.vkQueueWaitIdle(queue);
        vk.vkMapMemory(dev, rb.mem, 0, static_cast<size_t>(rw) * rh * 4, 0, &ptr);
        const uint8_t* px = static_cast<const uint8_t*>(ptr);
        double sum = 0.0;
        for (size_t i = 0; i < static_cast<size_t>(rw) * rh * 4; ++i) sum += px[i];
        vk.vkUnmapMemory(dev, rb.mem);
        r.checksum = sum;

        if (spec.category == BenchCategory::Pixel) {
            r.shortJobMs = drawVK(64, 64, 6);
            r.longJobMs = drawVK(width * 2, height * 2, 6);
        } else {
            r.shortJobMs = drawVK(128, 128, 3000);
            r.longJobMs = drawVK(128, 128, 96000);
        }

        finalizeTimingAndMemory(r, memBefore, memAfterCompile, samples);
        results.push_back(r);

        vk.vkDestroyCommandPool(dev, cpool, nullptr);
        vk.vkDestroyDescriptorPool(dev, dp, nullptr);
        for (auto& b : {cb, vb, rb}) {
            vk.vkDestroyBuffer(dev, b.buf, nullptr);
            vk.vkFreeMemory(dev, b.mem, nullptr);
        }
        vk.vkDestroyFramebuffer(dev, fb, nullptr);
        vk.vkDestroyImageView(dev, iv, nullptr);
        vk.vkDestroyImage(dev, img, nullptr);
        vk.vkFreeMemory(dev, imgMem, nullptr);
        vk.vkDestroyPipeline(dev, pipe, nullptr);
        vk.vkDestroyPipelineLayout(dev, pl, nullptr);
        vk.vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
        vk.vkDestroyShaderModule(dev, psMod, nullptr);
        vk.vkDestroyShaderModule(dev, vsMod, nullptr);
        vk.vkDestroyRenderPass(dev, rp, nullptr);
        vk.vkDestroyDevice(dev, nullptr);
    }

    vk.vkDestroyInstance(inst, nullptr);
    return results;
}

} // namespace nacelle_bench
