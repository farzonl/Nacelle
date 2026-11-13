#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <nacelle/nacelle.h>
#include "../src/compiler/jit_engine.h"
#include "../src/compiler/shader_compiler.h"
#include "vk_minimal_loader.h"

namespace {

struct BenchParams {
    uint32_t width;
    uint32_t height;
    uint32_t maxSteps;
    float time;
};

struct BenchResult {
    std::string backend;
    std::string shaderFormat;
    uint32_t simdWidth = 0;
    double compileMs = 0.0;
    double meanMs = 0.0;
    double medianMs = 0.0;
    double minMs = 0.0;
    double checksum = 0.0;
    bool ran = false;
    std::string status = "OK";
};

bool readBinary(const std::string& path, std::vector<uint8_t>& out) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) return false;
    auto sz = ifs.tellg();
    if (sz <= 0) return false;
    out.resize(static_cast<size_t>(sz));
    ifs.seekg(0, std::ios::beg);
    return static_cast<bool>(ifs.read(reinterpret_cast<char*>(out.data()), sz));
}

bool validateSpirvBinary(const std::vector<uint8_t>& spv, std::string& outErr) {
    if (spv.size() < 20 || (spv.size() % 4) != 0) {
        outErr = "invalid SPIR-V byte size";
        return false;
    }
    uint32_t magic = 0;
    std::memcpy(&magic, spv.data(), 4);
    if (magic != 0x07230203u) {
        outErr = "invalid SPIR-V magic";
        return false;
    }
    return true;
}

double computeMedian(std::vector<double> vals) {
    if (vals.empty()) return 0.0;
    std::sort(vals.begin(), vals.end());
    size_t mid = vals.size() / 2;
    if (vals.size() % 2 == 0) {
        return 0.5 * (vals[mid - 1] + vals[mid]);
    }
    return vals[mid];
}

BenchResult runNacelleBench(
    const std::vector<uint8_t>& dxil,
    uint32_t simdWidth,
    uint32_t width,
    uint32_t height,
    uint32_t maxSteps,
    uint32_t iters) {
    BenchResult res;
    res.backend = "Nacelle (W=" + std::to_string(simdWidth) + ")";
    res.shaderFormat = "DXIL";
    res.simdWidth = simdWidth;

    nacelle::setSimdWidthOverride(simdWidth);
    nacelle::clearInMemoryPSOCacheForTesting();

    ID3D12Device* dev = nullptr;
    if (FAILED(NacelleCreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&dev))) || !dev) {
        res.status = "FAILED (CreateDevice)";
        return res;
    }

    D3D12_COMMAND_QUEUE_DESC qDesc{};
    qDesc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    ID3D12CommandQueue* queue = nullptr;
    dev->CreateCommandQueue(&qDesc, IID_PPV_ARGS(&queue));

    ID3D12CommandAllocator* alloc = nullptr;
    dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&alloc));

    ID3D12GraphicsCommandList* cmd = nullptr;
    dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, alloc, nullptr, IID_PPV_ARGS(&cmd));

    ID3D12Fence* fence = nullptr;
    dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));

    // Root signature: b0 (CBV), t0 (SRV), u0 (UAV)
    D3D12_ROOT_PARAMETER params[3]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    params[0].Descriptor.ShaderRegister = 0;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[1].Descriptor.ShaderRegister = 0;
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[2].Descriptor.ShaderRegister = 0;

    D3D12_ROOT_SIGNATURE_DESC rsDesc{};
    rsDesc.NumParameters = 3;
    rsDesc.pParameters = params;

    ID3DBlob* rsBlob = nullptr;
    NacelleSerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1_0, &rsBlob, nullptr);
    ID3D12RootSignature* rootSig = nullptr;
    dev->CreateRootSignature(0, rsBlob->GetBufferPointer(), rsBlob->GetBufferSize(), IID_PPV_ARGS(&rootSig));
    rsBlob->Release();

    auto t0 = std::chrono::steady_clock::now();
    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc{};
    psoDesc.pRootSignature = rootSig;
    psoDesc.CS = {dxil.data(), dxil.size()};
    ID3D12PipelineState* pso = nullptr;
    HRESULT hr = dev->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&pso));
    nacelle::waitForAllAsyncJITTasks();
    auto t1 = std::chrono::steady_clock::now();
    res.compileMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (FAILED(hr) || !pso) {
        res.status = "FAILED (CreateComputePipelineState)";
        return res;
    }

    auto createBuf = [&](size_t bytes) -> ID3D12Resource* {
        D3D12_HEAP_PROPERTIES hp{};
        hp.Type = D3D12_HEAP_TYPE_CUSTOM;
        hp.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
        hp.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
        D3D12_RESOURCE_DESC rd{};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = bytes;
        rd.Height = 1;
        rd.DepthOrArraySize = 1;
        rd.MipLevels = 1;
        rd.SampleDesc.Count = 1;
        rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        ID3D12Resource* r = nullptr;
        dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&r));
        return r;
    };

    ID3D12Resource* cbBuf = createBuf(256);
    ID3D12Resource* inBuf = createBuf(4 * 16);
    size_t outBytes = static_cast<size_t>(width) * height * 16;
    ID3D12Resource* outBuf = createBuf(outBytes);

    BenchParams* cbPtr = nullptr;
    cbBuf->Map(0, nullptr, reinterpret_cast<void**>(&cbPtr));
    cbPtr->width = width;
    cbPtr->height = height;
    cbPtr->maxSteps = maxSteps;
    cbPtr->time = 0.0f;
    cbBuf->Unmap(0, nullptr);

    float* sphPtr = nullptr;
    inBuf->Map(0, nullptr, reinterpret_cast<void**>(&sphPtr));
    float spheres[16] = {
        0.0f, 0.2f, 0.5f, 0.8f,
        -1.3f, 0.0f, 1.0f, 0.6f,
        1.3f, 0.1f, 0.9f, 0.65f,
        0.0f, 1.1f, 1.5f, 0.5f,
    };
    std::memcpy(sphPtr, spheres, sizeof(spheres));
    inBuf->Unmap(0, nullptr);

    uint32_t gx = (width + 7) / 8;
    uint32_t gy = (height + 7) / 8;

    auto dispatchOnce = [&](uint64_t fenceVal) {
        cmd->SetComputeRootSignature(rootSig);
        cmd->SetPipelineState(pso);
        cmd->SetComputeRootConstantBufferView(0, cbBuf->GetGPUVirtualAddress());
        cmd->SetComputeRootShaderResourceView(1, inBuf->GetGPUVirtualAddress());
        cmd->SetComputeRootUnorderedAccessView(2, outBuf->GetGPUVirtualAddress());
        cmd->Dispatch(gx, gy, 1);
        cmd->Close();
        ID3D12CommandList* lists[] = {cmd};
        queue->ExecuteCommandLists(1, lists);
        queue->Signal(fence, fenceVal);
        while (fence->GetCompletedValue() < fenceVal) {}
        alloc->Reset();
        cmd->Reset(alloc, nullptr);
    };

    // Warmup
    dispatchOnce(1);

    std::vector<double> samples;
    samples.reserve(iters);
    for (uint32_t i = 0; i < iters; ++i) {
        auto s0 = std::chrono::steady_clock::now();
        dispatchOnce(i + 2);
        auto s1 = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(s1 - s0).count());
    }

    float* outPtr = nullptr;
    outBuf->Map(0, nullptr, reinterpret_cast<void**>(&outPtr));
    double sum = 0.0;
    size_t count = static_cast<size_t>(width) * height * 4;
    for (size_t i = 0; i < count; ++i) sum += outPtr[i];
    outBuf->Unmap(0, nullptr);

    double total = 0.0;
    for (double v : samples) total += v;
    res.meanMs = total / static_cast<double>(samples.size());
    res.medianMs = computeMedian(samples);
    res.minMs = *std::min_element(samples.begin(), samples.end());
    res.checksum = sum;
    res.ran = true;

    outBuf->Release();
    inBuf->Release();
    cbBuf->Release();
    pso->Release();
    rootSig->Release();
    fence->Release();
    cmd->Release();
    alloc->Release();
    queue->Release();
    dev->Release();
    return res;
}

std::vector<BenchResult> runVulkanBenchmarks(
    const std::vector<uint8_t>& spv,
    uint32_t width,
    uint32_t height,
    uint32_t maxSteps,
    uint32_t iters) {
    std::vector<BenchResult> results;
    vkmin::Loader vk;
    if (!vk.loadLibrary()) {
        BenchResult r;
        r.backend = "Vulkan (Lavapipe/SwiftShader)";
        r.shaderFormat = "SPIR-V";
        r.status = "SKIPPED (libvulkan not installed)";
        results.push_back(r);
        return results;
    }

    vkmin::VkApplicationInfo appInfo{};
    appInfo.sType = vkmin::VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "nacelle-bench";
    appInfo.apiVersion = (1u << 22) | (1u << 12);

    const char* portabilityExt = "VK_KHR_portability_enumeration";
    vkmin::VkInstanceCreateInfo ici{};
    ici.sType = vkmin::VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.flags = 0x00000001u; // VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
    ici.pApplicationInfo = &appInfo;
    ici.enabledExtensionCount = 1;
    ici.ppEnabledExtensionNames = &portabilityExt;

    vkmin::VkInstance inst = nullptr;
    vkmin::VkResult instRes = vk.vkCreateInstance(&ici, nullptr, &inst);
    if (instRes != vkmin::VK_SUCCESS) {
        ici.flags = 0;
        ici.enabledExtensionCount = 0;
        ici.ppEnabledExtensionNames = nullptr;
        instRes = vk.vkCreateInstance(&ici, nullptr, &inst);
    }
    if (instRes != vkmin::VK_SUCCESS || !vk.loadInstanceSyms(inst)) {
        BenchResult r;
        r.backend = "Vulkan (Lavapipe/SwiftShader)";
        r.shaderFormat = "SPIR-V";
        r.status = "SKIPPED (vkCreateInstance failed)";
        results.push_back(r);
        return results;
    }

    uint32_t physCount = 0;
    vk.vkEnumeratePhysicalDevices(inst, &physCount, nullptr);
    std::vector<vkmin::VkPhysicalDevice> physDevs(physCount);
    if (physCount > 0) vk.vkEnumeratePhysicalDevices(inst, &physCount, physDevs.data());

    for (auto pd : physDevs) {
        vkmin::VkPhysicalDeviceProperties props{};
        vk.vkGetPhysicalDeviceProperties(pd, &props);
        std::string devName = props.deviceName;

        BenchResult r;
        r.backend = "Vulkan: " + devName;
        r.shaderFormat = "SPIR-V";

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
        if (computeQF == UINT32_MAX) continue;

        float prio = 1.0f;
        vkmin::VkDeviceQueueCreateInfo dqci{};
        dqci.sType = vkmin::VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        dqci.queueFamilyIndex = computeQF;
        dqci.queueCount = 1;
        dqci.pQueuePriorities = &prio;

        vkmin::VkDeviceCreateInfo dci{};
        dci.sType = vkmin::VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &dqci;

        vkmin::VkDevice dev = nullptr;
        if (vk.vkCreateDevice(pd, &dci, nullptr, &dev) != vkmin::VK_SUCCESS) continue;
        vkmin::VkQueue queue = nullptr;
        vk.vkGetDeviceQueue(dev, computeQF, 0, &queue);

        auto t0 = std::chrono::steady_clock::now();
        vkmin::VkShaderModuleCreateInfo smci{};
        smci.sType = vkmin::VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = spv.size();
        smci.pCode = reinterpret_cast<const uint32_t*>(spv.data());
        vkmin::VkShaderModule sm = vkmin::VK_NULL_HANDLE;
        vk.vkCreateShaderModule(dev, &smci, nullptr, &sm);

        vkmin::VkDescriptorSetLayoutBinding bindings[3]{};
        bindings[0] = {16, vkmin::VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, vkmin::VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[1] = {0, vkmin::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, vkmin::VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[2] = {8, vkmin::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, vkmin::VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

        vkmin::VkDescriptorSetLayoutCreateInfo dslci{};
        dslci.sType = vkmin::VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dslci.bindingCount = 3;
        dslci.pBindings = bindings;
        vkmin::VkDescriptorSetLayout dsl = vkmin::VK_NULL_HANDLE;
        vk.vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl);

        vkmin::VkPipelineLayoutCreateInfo plci{};
        plci.sType = vkmin::VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &dsl;
        vkmin::VkPipelineLayout pl = vkmin::VK_NULL_HANDLE;
        vk.vkCreatePipelineLayout(dev, &plci, nullptr, &pl);

        vkmin::VkComputePipelineCreateInfo cpci{};
        cpci.sType = vkmin::VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage.sType = vkmin::VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpci.stage.stage = vkmin::VK_SHADER_STAGE_COMPUTE_BIT;
        cpci.stage.module = sm;
        cpci.stage.pName = "main";
        cpci.layout = pl;
        vkmin::VkPipeline pipe = vkmin::VK_NULL_HANDLE;
        vk.vkCreateComputePipelines(dev, vkmin::VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe);
        auto t1 = std::chrono::steady_clock::now();
        r.compileMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

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

        struct BufAlloc {
            vkmin::VkBuffer buf = 0;
            vkmin::VkDeviceMemory mem = 0;
        };
        auto mkBuf = [&](size_t sz, vkmin::VkFlags usage) -> BufAlloc {
            BufAlloc ba;
            vkmin::VkBufferCreateInfo bci{};
            bci.sType = vkmin::VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bci.size = sz;
            bci.usage = usage;
            bci.sharingMode = vkmin::VK_SHARING_MODE_EXCLUSIVE;
            vk.vkCreateBuffer(dev, &bci, nullptr, &ba.buf);
            vkmin::VkMemoryRequirements mr{};
            vk.vkGetBufferMemoryRequirements(dev, ba.buf, &mr);
            vkmin::VkMemoryAllocateInfo mai{};
            mai.sType = vkmin::VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            mai.allocationSize = mr.size;
            mai.memoryTypeIndex = findMem(mr.memoryTypeBits);
            vk.vkAllocateMemory(dev, &mai, nullptr, &ba.mem);
            vk.vkBindBufferMemory(dev, ba.buf, ba.mem, 0);
            return ba;
        };

        auto cb = mkBuf(256, vkmin::VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        auto inB = mkBuf(64, vkmin::VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        size_t outBytes = static_cast<size_t>(width) * height * 16;
        auto outB = mkBuf(outBytes, vkmin::VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

        void* ptr = nullptr;
        vk.vkMapMemory(dev, cb.mem, 0, 256, 0, &ptr);
        BenchParams bp{width, height, maxSteps, 0.0f};
        std::memcpy(ptr, &bp, sizeof(bp));
        vk.vkUnmapMemory(dev, cb.mem);

        vk.vkMapMemory(dev, inB.mem, 0, 64, 0, &ptr);
        float spheres[16] = {
            0.0f, 0.2f, 0.5f, 0.8f,
            -1.3f, 0.0f, 1.0f, 0.6f,
            1.3f, 0.1f, 0.9f, 0.65f,
            0.0f, 1.1f, 1.5f, 0.5f,
        };
        std::memcpy(ptr, spheres, sizeof(spheres));
        vk.vkUnmapMemory(dev, inB.mem);

        vkmin::VkDescriptorPoolSize psz[2] = {
            {vkmin::VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
            {vkmin::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2},
        };
        vkmin::VkDescriptorPoolCreateInfo dpci{};
        dpci.sType = vkmin::VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets = 1;
        dpci.poolSizeCount = 2;
        dpci.pPoolSizes = psz;
        vkmin::VkDescriptorPool dp = 0;
        vk.vkCreateDescriptorPool(dev, &dpci, nullptr, &dp);

        vkmin::VkDescriptorSetAllocateInfo dsai{};
        dsai.sType = vkmin::VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool = dp;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &dsl;
        vkmin::VkDescriptorSet ds = 0;
        vk.vkAllocateDescriptorSets(dev, &dsai, &ds);

        vkmin::VkDescriptorBufferInfo bi[3] = {
            {cb.buf, 0, 256},
            {inB.buf, 0, 64},
            {outB.buf, 0, outBytes},
        };
        vkmin::VkWriteDescriptorSet wds[3]{};
        wds[0] = {vkmin::VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 16, 0, 1, vkmin::VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &bi[0], nullptr};
        wds[1] = {vkmin::VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 0, 0, 1, vkmin::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &bi[1], nullptr};
        wds[2] = {vkmin::VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 8, 0, 1, vkmin::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &bi[2], nullptr};
        vk.vkUpdateDescriptorSets(dev, 3, wds, 0, nullptr);

        vkmin::VkCommandPoolCreateInfo cpci2{vkmin::VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr, vkmin::VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, computeQF};
        vkmin::VkCommandPool cpool = nullptr;
        vk.vkCreateCommandPool(dev, &cpci2, nullptr, &cpool);

        vkmin::VkCommandBufferAllocateInfo cbai{vkmin::VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, cpool, vkmin::VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
        vkmin::VkCommandBuffer cmdBuf = nullptr;
        vk.vkAllocateCommandBuffers(dev, &cbai, &cmdBuf);

        uint32_t gx = (width + 7) / 8;
        uint32_t gy = (height + 7) / 8;

        auto dispatchVK = [&]() {
            vk.vkResetCommandBuffer(cmdBuf, 0);
            vkmin::VkCommandBufferBeginInfo cbbi{vkmin::VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, vkmin::VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
            vk.vkBeginCommandBuffer(cmdBuf, &cbbi);
            vk.vkCmdBindPipeline(cmdBuf, 1 /*COMPUTE*/, pipe);
            vk.vkCmdBindDescriptorSets(cmdBuf, 1 /*COMPUTE*/, pl, 0, 1, &ds, 0, nullptr);
            vk.vkCmdDispatch(cmdBuf, gx, gy, 1);
            vk.vkEndCommandBuffer(cmdBuf);
            vkmin::VkSubmitInfo si{};
            si.sType = vkmin::VK_STRUCTURE_TYPE_SUBMIT_INFO;
            si.commandBufferCount = 1;
            si.pCommandBuffers = &cmdBuf;
            vk.vkQueueSubmit(queue, 1, &si, 0);
            vk.vkQueueWaitIdle(queue);
        };

        dispatchVK();
        std::vector<double> samples;
        samples.reserve(iters);
        for (uint32_t i = 0; i < iters; ++i) {
            auto s0 = std::chrono::steady_clock::now();
            dispatchVK();
            auto s1 = std::chrono::steady_clock::now();
            samples.push_back(std::chrono::duration<double, std::milli>(s1 - s0).count());
        }

        vk.vkMapMemory(dev, outB.mem, 0, outBytes, 0, &ptr);
        const float* fp = static_cast<const float*>(ptr);
        double sum = 0.0;
        for (size_t i = 0; i < static_cast<size_t>(width) * height * 4; ++i) sum += fp[i];
        vk.vkUnmapMemory(dev, outB.mem);

        double tot = 0.0;
        for (double v : samples) tot += v;
        r.meanMs = tot / static_cast<double>(samples.size());
        r.medianMs = computeMedian(samples);
        r.minMs = *std::min_element(samples.begin(), samples.end());
        r.checksum = sum;
        r.ran = true;
        results.push_back(r);

        vk.vkDestroyCommandPool(dev, cpool, nullptr);
        vk.vkDestroyDescriptorPool(dev, dp, nullptr);
        vk.vkDestroyBuffer(dev, outB.buf, nullptr);
        vk.vkFreeMemory(dev, outB.mem, nullptr);
        vk.vkDestroyBuffer(dev, inB.buf, nullptr);
        vk.vkFreeMemory(dev, inB.mem, nullptr);
        vk.vkDestroyBuffer(dev, cb.buf, nullptr);
        vk.vkFreeMemory(dev, cb.mem, nullptr);
        vk.vkDestroyPipeline(dev, pipe, nullptr);
        vk.vkDestroyPipelineLayout(dev, pl, nullptr);
        vk.vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
        vk.vkDestroyShaderModule(dev, sm, nullptr);
        vk.vkDestroyDevice(dev, nullptr);
    }

    if (results.empty()) {
        BenchResult r;
        r.backend = "Vulkan (Lavapipe/SwiftShader)";
        r.shaderFormat = "SPIR-V";
        r.status = "SKIPPED (no software Vulkan ICD found)";
        results.push_back(r);
    }

    vk.vkDestroyInstance(inst, nullptr);
    return results;
}

void writeJsonReport(const std::string& path, const std::vector<BenchResult>& results) {
    std::ofstream ofs(path);
    ofs << "{\n  \"benchmarks\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        ofs << "    {\n"
            << "      \"backend\": \"" << r.backend << "\",\n"
            << "      \"shader_format\": \"" << r.shaderFormat << "\",\n"
            << "      \"simd_width\": " << r.simdWidth << ",\n"
            << "      \"ran\": " << (r.ran ? "true" : "false") << ",\n"
            << "      \"status\": \"" << r.status << "\",\n"
            << "      \"compile_ms\": " << std::fixed << std::setprecision(4) << r.compileMs << ",\n"
            << "      \"mean_ms\": " << r.meanMs << ",\n"
            << "      \"median_ms\": " << r.medianMs << ",\n"
            << "      \"min_ms\": " << r.minMs << ",\n"
            << "      \"checksum\": " << std::setprecision(6) << r.checksum << "\n"
            << "    }" << (i + 1 < results.size() ? "," : "") << "\n";
    }
    ofs << "  ]\n}\n";
}

} // namespace

int main(int argc, char** argv) {
    std::vector<uint32_t> widths = {4, 8, 16};
    uint32_t width = 256;
    uint32_t height = 256;
    uint32_t maxSteps = 48;
    uint32_t iters = 5;
    std::string jsonOut;
    std::string shaderDir = NACELLE_SHADER_BIN_DIR;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--quick") {
            width = 128;
            height = 128;
            maxSteps = 24;
            iters = 2;
        } else if (arg.rfind("--simd-width=", 0) == 0) {
            widths.clear();
            std::stringstream ss(arg.substr(13));
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                if (!tok.empty()) widths.push_back(static_cast<uint32_t>(std::stoul(tok)));
            }
        } else if (arg == "--iters" && i + 1 < argc) {
            iters = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--json" && i + 1 < argc) {
            jsonOut = argv[++i];
        } else if (arg == "--shader-dir" && i + 1 < argc) {
            shaderDir = argv[++i];
        }
    }

    std::string dxilPath = shaderDir + "/bench_raymarch.dxil";
    std::string spvPath = shaderDir + "/bench_raymarch.spv";

    std::vector<uint8_t> dxil, spv;
    if (!readBinary(dxilPath, dxil)) {
        std::cerr << "Failed to read DXIL benchmark shader: " << dxilPath << "\n";
        return 1;
    }
    if (!readBinary(spvPath, spv)) {
        std::cerr << "Failed to read SPIR-V benchmark shader: " << spvPath << "\n";
        return 1;
    }

    std::string spvErr;
    if (!validateSpirvBinary(spv, spvErr)) {
        std::cerr << "Invalid SPIR-V binary (" << spvPath << "): " << spvErr << "\n";
        return 1;
    }

    std::vector<BenchResult> allResults;
    bool anyNacelleFailed = false;
    for (uint32_t w : widths) {
        auto r = runNacelleBench(dxil, w, width, height, maxSteps, iters);
        if (!r.ran) anyNacelleFailed = true;
        allResults.push_back(r);
    }

#if defined(_WIN32)
    {
        BenchResult warpRes;
        warpRes.backend = "Microsoft WARP";
        warpRes.shaderFormat = "DXIL";
        warpRes.status = "SKIPPED (run via D3D12 WARP adapter on Windows)";
        allResults.push_back(warpRes);
    }
#else
    {
        BenchResult warpRes;
        warpRes.backend = "Microsoft WARP";
        warpRes.shaderFormat = "DXIL";
        warpRes.status = "SKIPPED (Windows-only)";
        allResults.push_back(warpRes);
    }
#endif

    auto vkResults = runVulkanBenchmarks(spv, width, height, maxSteps, iters);
    allResults.insert(allResults.end(), vkResults.begin(), vkResults.end());

    std::cout << "\n=== Nacelle Cross-Backend Benchmark (" << width << "x" << height
              << ", steps=" << maxSteps << ", iters=" << iters << ") ===\n";
    std::cout << "SPIR-V validated: " << spvPath << " (" << spv.size() << " bytes)\n\n";
    std::cout << std::left << std::setw(32) << "Backend"
              << std::setw(8) << "Fmt"
              << std::setw(12) << "Compile(ms)"
              << std::setw(12) << "Median(ms)"
              << std::setw(12) << "Min(ms)"
              << std::setw(14) << "Checksum"
              << "Status\n";
    std::cout << std::string(104, '-') << "\n";

    for (const auto& r : allResults) {
        std::cout << std::left << std::setw(32) << r.backend
                  << std::setw(8) << r.shaderFormat;
        if (r.ran) {
            std::cout << std::fixed << std::setprecision(2)
                      << std::setw(12) << r.compileMs
                      << std::setw(12) << r.medianMs
                      << std::setw(12) << r.minMs
                      << std::setprecision(1) << std::setw(14) << r.checksum
                      << r.status << "\n";
        } else {
            std::cout << std::setw(12) << "-"
                      << std::setw(12) << "-"
                      << std::setw(12) << "-"
                      << std::setw(14) << "-"
                      << r.status << "\n";
        }
    }

    if (!jsonOut.empty()) {
        writeJsonReport(jsonOut, allResults);
        std::cout << "\nJSON report written to: " << jsonOut << "\n";
    }

    return anyNacelleFailed ? 1 : 0;
}
