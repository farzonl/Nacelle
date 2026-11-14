#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#define VK_PTR __stdcall
#else
#include <dlfcn.h>
#define VK_PTR
#endif

namespace vkmin {

using VkResult = int32_t;
using VkFlags = uint32_t;
using VkDeviceSize = uint64_t;
using VkBool32 = uint32_t;

using VkInstance = struct VkInstance_T*;
using VkPhysicalDevice = struct VkPhysicalDevice_T*;
using VkDevice = struct VkDevice_T*;
using VkQueue = struct VkQueue_T*;
using VkCommandPool = struct VkCommandPool_T*;
using VkCommandBuffer = struct VkCommandBuffer_T*;
using VkBuffer = uint64_t;
using VkDeviceMemory = uint64_t;
using VkShaderModule = uint64_t;
using VkDescriptorSetLayout = uint64_t;
using VkPipelineLayout = uint64_t;
using VkPipeline = uint64_t;
using VkPipelineCache = uint64_t;
using VkDescriptorPool = uint64_t;
using VkDescriptorSet = uint64_t;
using VkFence = uint64_t;
using VkImage = uint64_t;
using VkImageView = uint64_t;
using VkRenderPass = uint64_t;
using VkFramebuffer = uint64_t;

constexpr VkResult VK_SUCCESS = 0;
constexpr uint64_t VK_NULL_HANDLE = 0;
constexpr VkDeviceSize VK_WHOLE_SIZE = ~0ULL;

constexpr uint32_t VK_STRUCTURE_TYPE_APPLICATION_INFO = 0;
constexpr uint32_t VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1;
constexpr uint32_t VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO = 2;
constexpr uint32_t VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO = 3;
constexpr uint32_t VK_STRUCTURE_TYPE_SUBMIT_INFO = 4;
constexpr uint32_t VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO = 5;
constexpr uint32_t VK_STRUCTURE_TYPE_FENCE_CREATE_INFO = 8;
constexpr uint32_t VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO = 12;
constexpr uint32_t VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO = 16;
constexpr uint32_t VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO = 29;
constexpr uint32_t VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO = 18;
constexpr uint32_t VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO = 30;
constexpr uint32_t VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO = 32;
constexpr uint32_t VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO = 33;
constexpr uint32_t VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO = 34;
constexpr uint32_t VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET = 35;
constexpr uint32_t VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO = 39;
constexpr uint32_t VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO = 40;
constexpr uint32_t VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO = 42;
constexpr uint32_t VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO = 14;
constexpr uint32_t VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO = 15;
constexpr uint32_t VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO = 19;
constexpr uint32_t VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO = 20;
constexpr uint32_t VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO = 22;
constexpr uint32_t VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO = 23;
constexpr uint32_t VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO = 24;
constexpr uint32_t VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO = 26;
constexpr uint32_t VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO = 27;
constexpr uint32_t VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO = 28;
constexpr uint32_t VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO = 37;
constexpr uint32_t VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO = 38;
constexpr uint32_t VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO = 43;

constexpr VkFlags VK_QUEUE_GRAPHICS_BIT = 0x00000001;
constexpr VkFlags VK_QUEUE_COMPUTE_BIT = 0x00000002;
constexpr VkFlags VK_BUFFER_USAGE_TRANSFER_DST_BIT = 0x00000002;
constexpr VkFlags VK_BUFFER_USAGE_VERTEX_BUFFER_BIT = 0x00000080;
constexpr VkFlags VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT = 0x00000010;
constexpr VkFlags VK_BUFFER_USAGE_STORAGE_BUFFER_BIT = 0x00000020;
constexpr VkFlags VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT = 0x00000001;
constexpr VkFlags VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT = 0x00000002;
constexpr VkFlags VK_MEMORY_PROPERTY_HOST_COHERENT_BIT = 0x00000004;
constexpr uint32_t VK_SHARING_MODE_EXCLUSIVE = 0;
constexpr uint32_t VK_SHADER_STAGE_VERTEX_BIT = 0x00000001;
constexpr uint32_t VK_SHADER_STAGE_FRAGMENT_BIT = 0x00000010;
constexpr uint32_t VK_SHADER_STAGE_COMPUTE_BIT = 0x00000020;
constexpr uint32_t VK_PIPELINE_BIND_POINT_GRAPHICS = 0;
constexpr uint32_t VK_PIPELINE_BIND_POINT_COMPUTE = 1;
constexpr uint32_t VK_FORMAT_R8G8B8A8_UNORM = 37;
constexpr uint32_t VK_FORMAT_R32G32B32_SFLOAT = 106;
constexpr uint32_t VK_IMAGE_TYPE_2D = 1;
constexpr uint32_t VK_IMAGE_TILING_OPTIMAL = 0;
constexpr uint32_t VK_IMAGE_USAGE_TRANSFER_SRC_BIT = 0x00000001;
constexpr uint32_t VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT = 0x00000010;
constexpr uint32_t VK_IMAGE_LAYOUT_UNDEFINED = 0;
constexpr uint32_t VK_IMAGE_LAYOUT_GENERAL = 1;
constexpr uint32_t VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL = 2;
constexpr uint32_t VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL = 6;
constexpr uint32_t VK_IMAGE_VIEW_TYPE_2D = 1;
constexpr uint32_t VK_IMAGE_ASPECT_COLOR_BIT = 0x00000001;
constexpr uint32_t VK_SAMPLE_COUNT_1_BIT = 0x00000001;
constexpr uint32_t VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST = 3;
constexpr uint32_t VK_POLYGON_MODE_FILL = 0;
constexpr uint32_t VK_CULL_MODE_NONE = 0;
constexpr uint32_t VK_FRONT_FACE_COUNTER_CLOCKWISE = 0;
constexpr uint32_t VK_ATTACHMENT_LOAD_OP_CLEAR = 1;
constexpr uint32_t VK_ATTACHMENT_STORE_OP_STORE = 0;
constexpr uint32_t VK_ATTACHMENT_LOAD_OP_DONT_CARE = 2;
constexpr uint32_t VK_ATTACHMENT_STORE_OP_DONT_CARE = 1;
constexpr uint32_t VK_SUBPASS_CONTENTS_INLINE = 0;
constexpr uint32_t VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER = 6;
constexpr uint32_t VK_DESCRIPTOR_TYPE_STORAGE_BUFFER = 7;
constexpr uint32_t VK_COMMAND_BUFFER_LEVEL_PRIMARY = 0;
constexpr uint32_t VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT = 0x00000001;
constexpr uint32_t VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT = 0x00000002;

struct VkApplicationInfo {
    uint32_t sType;
    const void* pNext;
    const char* pApplicationName;
    uint32_t applicationVersion;
    const char* pEngineName;
    uint32_t engineVersion;
    uint32_t apiVersion;
};

struct VkInstanceCreateInfo {
    uint32_t sType;
    const void* pNext;
    VkFlags flags;
    const VkApplicationInfo* pApplicationInfo;
    uint32_t enabledLayerCount;
    const char* const* ppEnabledLayerNames;
    uint32_t enabledExtensionCount;
    const char* const* ppEnabledExtensionNames;
};

struct VkPhysicalDeviceLimits {
    alignas(8) uint8_t raw[504];
};

struct VkPhysicalDeviceSparseProperties {
    VkBool32 residencyStandard2DBlockShape;
    VkBool32 residencyStandard2DMultisampleBlockShape;
    VkBool32 residencyStandard3DBlockShape;
    VkBool32 residencyAlignedMipSize;
    VkBool32 residencyNonResidentStrict;
};

struct VkPhysicalDeviceProperties {
    uint32_t apiVersion;
    uint32_t driverVersion;
    uint32_t vendorID;
    uint32_t deviceID;
    uint32_t deviceType;
    char deviceName[256];
    uint8_t pipelineCacheUUID[16];
    VkPhysicalDeviceLimits limits;
    VkPhysicalDeviceSparseProperties sparseProperties;
};

struct VkConformanceVersion {
    uint8_t major;
    uint8_t minor;
    uint8_t subminor;
    uint8_t patch;
};

struct VkPhysicalDeviceDriverProperties {
    uint32_t sType;
    void* pNext;
    uint32_t driverID;
    char driverName[256];
    char driverInfo[256];
    VkConformanceVersion conformanceVersion;
};

struct VkPhysicalDeviceProperties2 {
    uint32_t sType;
    void* pNext;
    VkPhysicalDeviceProperties properties;
};

struct VkQueueFamilyProperties {
    VkFlags queueFlags;
    uint32_t queueCount;
    uint32_t timestampValidBits;
    uint32_t minImageTransferGranularity[3];
};

struct VkMemoryType {
    VkFlags propertyFlags;
    uint32_t heapIndex;
};

struct VkMemoryHeap {
    VkDeviceSize size;
    VkFlags flags;
};

struct VkPhysicalDeviceMemoryProperties {
    uint32_t memoryTypeCount;
    VkMemoryType memoryTypes[32];
    uint32_t memoryHeapCount;
    VkMemoryHeap memoryHeaps[16];
};

struct VkDeviceQueueCreateInfo {
    uint32_t sType;
    const void* pNext;
    VkFlags flags;
    uint32_t queueFamilyIndex;
    uint32_t queueCount;
    const float* pQueuePriorities;
};

struct VkDeviceCreateInfo {
    uint32_t sType;
    const void* pNext;
    VkFlags flags;
    uint32_t queueCreateInfoCount;
    const VkDeviceQueueCreateInfo* pQueueCreateInfos;
    uint32_t enabledLayerCount;
    const char* const* ppEnabledLayerNames;
    uint32_t enabledExtensionCount;
    const char* const* ppEnabledExtensionNames;
    const void* pEnabledFeatures;
};

struct VkBufferCreateInfo {
    uint32_t sType;
    const void* pNext;
    VkFlags flags;
    VkDeviceSize size;
    VkFlags usage;
    uint32_t sharingMode;
    uint32_t queueFamilyIndexCount;
    const uint32_t* pQueueFamilyIndices;
};

struct VkMemoryRequirements {
    VkDeviceSize size;
    VkDeviceSize alignment;
    uint32_t memoryTypeBits;
};

struct VkMemoryAllocateInfo {
    uint32_t sType;
    const void* pNext;
    VkDeviceSize allocationSize;
    uint32_t memoryTypeIndex;
};

struct VkShaderModuleCreateInfo {
    uint32_t sType;
    const void* pNext;
    VkFlags flags;
    size_t codeSize;
    const uint32_t* pCode;
};

struct VkDescriptorSetLayoutBinding {
    uint32_t binding;
    uint32_t descriptorType;
    uint32_t descriptorCount;
    VkFlags stageFlags;
    const void* pImmutableSamplers;
};

struct VkDescriptorSetLayoutCreateInfo {
    uint32_t sType;
    const void* pNext;
    VkFlags flags;
    uint32_t bindingCount;
    const VkDescriptorSetLayoutBinding* pBindings;
};

struct VkPipelineLayoutCreateInfo {
    uint32_t sType;
    const void* pNext;
    VkFlags flags;
    uint32_t setLayoutCount;
    const VkDescriptorSetLayout* pSetLayouts;
    uint32_t pushConstantRangeCount;
    const void* pPushConstantRanges;
};

struct VkPipelineShaderStageCreateInfo {
    uint32_t sType;
    const void* pNext;
    VkFlags flags;
    VkFlags stage;
    VkShaderModule module;
    const char* pName;
    const void* pSpecializationInfo;
};

struct VkComputePipelineCreateInfo {
    uint32_t sType;
    const void* pNext;
    VkFlags flags;
    VkPipelineShaderStageCreateInfo stage;
    VkPipelineLayout layout;
    VkPipeline basePipelineHandle;
    int32_t basePipelineIndex;
};

struct VkDescriptorPoolSize {
    uint32_t type;
    uint32_t descriptorCount;
};

struct VkDescriptorPoolCreateInfo {
    uint32_t sType;
    const void* pNext;
    VkFlags flags;
    uint32_t maxSets;
    uint32_t poolSizeCount;
    const VkDescriptorPoolSize* pPoolSizes;
};

struct VkDescriptorSetAllocateInfo {
    uint32_t sType;
    const void* pNext;
    VkDescriptorPool descriptorPool;
    uint32_t descriptorSetCount;
    const VkDescriptorSetLayout* pSetLayouts;
};

struct VkDescriptorBufferInfo {
    VkBuffer buffer;
    VkDeviceSize offset;
    VkDeviceSize range;
};

struct VkWriteDescriptorSet {
    uint32_t sType;
    const void* pNext;
    VkDescriptorSet dstSet;
    uint32_t dstBinding;
    uint32_t dstArrayElement;
    uint32_t descriptorCount;
    uint32_t descriptorType;
    const void* pImageInfo;
    const VkDescriptorBufferInfo* pBufferInfo;
    const void* pTexelBufferView;
};

struct VkCommandPoolCreateInfo {
    uint32_t sType;
    const void* pNext;
    VkFlags flags;
    uint32_t queueFamilyIndex;
};

struct VkCommandBufferAllocateInfo {
    uint32_t sType;
    const void* pNext;
    VkCommandPool commandPool;
    uint32_t level;
    uint32_t commandBufferCount;
};

struct VkCommandBufferBeginInfo {
    uint32_t sType;
    const void* pNext;
    VkFlags flags;
    const void* pInheritanceInfo;
};

struct VkSubmitInfo {
    uint32_t sType;
    const void* pNext;
    uint32_t waitSemaphoreCount;
    const void* pWaitSemaphores;
    const VkFlags* pWaitDstStageMask;
    uint32_t commandBufferCount;
    const VkCommandBuffer* pCommandBuffers;
    uint32_t signalSemaphoreCount;
    const void* pSignalSemaphores;
};

struct VkFenceCreateInfo {
    uint32_t sType;
    const void* pNext;
    VkFlags flags;
};

struct VkExtent3D {
    uint32_t width;
    uint32_t height;
    uint32_t depth;
};

struct VkExtent2D {
    uint32_t width;
    uint32_t height;
};

struct VkOffset2D {
    int32_t x;
    int32_t y;
};

struct VkOffset3D {
    int32_t x;
    int32_t y;
    int32_t z;
};

struct VkRect2D {
    VkOffset2D offset;
    VkExtent2D extent;
};

struct VkViewport {
    float x;
    float y;
    float width;
    float height;
    float minDepth;
    float maxDepth;
};

struct VkImageCreateInfo {
    uint32_t sType;
    const void* pNext;
    VkFlags flags;
    uint32_t imageType;
    uint32_t format;
    VkExtent3D extent;
    uint32_t mipLevels;
    uint32_t arrayLayers;
    uint32_t samples;
    uint32_t tiling;
    VkFlags usage;
    uint32_t sharingMode;
    uint32_t queueFamilyIndexCount;
    const uint32_t* pQueueFamilyIndices;
    uint32_t initialLayout;
};

struct VkImageSubresourceRange {
    VkFlags aspectMask;
    uint32_t baseMipLevel;
    uint32_t levelCount;
    uint32_t baseArrayLayer;
    uint32_t layerCount;
};

struct VkComponentMapping {
    uint32_t r, g, b, a;
};

struct VkImageViewCreateInfo {
    uint32_t sType;
    const void* pNext;
    VkFlags flags;
    VkImage image;
    uint32_t viewType;
    uint32_t format;
    VkComponentMapping components;
    VkImageSubresourceRange subresourceRange;
};

struct VkAttachmentDescription {
    VkFlags flags;
    uint32_t format;
    uint32_t samples;
    uint32_t loadOp;
    uint32_t storeOp;
    uint32_t stencilLoadOp;
    uint32_t stencilStoreOp;
    uint32_t initialLayout;
    uint32_t finalLayout;
};

struct VkAttachmentReference {
    uint32_t attachment;
    uint32_t layout;
};

struct VkSubpassDescription {
    VkFlags flags;
    uint32_t pipelineBindPoint;
    uint32_t inputAttachmentCount;
    const VkAttachmentReference* pInputAttachments;
    uint32_t colorAttachmentCount;
    const VkAttachmentReference* pColorAttachments;
    const VkAttachmentReference* pResolveAttachments;
    const VkAttachmentReference* pDepthStencilAttachment;
    uint32_t preserveAttachmentCount;
    const uint32_t* pPreserveAttachments;
};

struct VkRenderPassCreateInfo {
    uint32_t sType;
    const void* pNext;
    VkFlags flags;
    uint32_t attachmentCount;
    const VkAttachmentDescription* pAttachments;
    uint32_t subpassCount;
    const VkSubpassDescription* pSubpasses;
    uint32_t dependencyCount;
    const void* pDependencies;
};

struct VkFramebufferCreateInfo {
    uint32_t sType;
    const void* pNext;
    VkFlags flags;
    VkRenderPass renderPass;
    uint32_t attachmentCount;
    const VkImageView* pAttachments;
    uint32_t width;
    uint32_t height;
    uint32_t layers;
};

struct VkVertexInputBindingDescription {
    uint32_t binding;
    uint32_t stride;
    uint32_t inputRate;
};

struct VkVertexInputAttributeDescription {
    uint32_t location;
    uint32_t binding;
    uint32_t format;
    uint32_t offset;
};

struct VkPipelineVertexInputStateCreateInfo {
    uint32_t sType;
    const void* pNext;
    VkFlags flags;
    uint32_t vertexBindingDescriptionCount;
    const VkVertexInputBindingDescription* pVertexBindingDescriptions;
    uint32_t vertexAttributeDescriptionCount;
    const VkVertexInputAttributeDescription* pVertexAttributeDescriptions;
};

struct VkPipelineInputAssemblyStateCreateInfo {
    uint32_t sType;
    const void* pNext;
    VkFlags flags;
    uint32_t topology;
    VkBool32 primitiveRestartEnable;
};

struct VkPipelineViewportStateCreateInfo {
    uint32_t sType;
    const void* pNext;
    VkFlags flags;
    uint32_t viewportCount;
    const VkViewport* pViewports;
    uint32_t scissorCount;
    const VkRect2D* pScissors;
};

struct VkPipelineRasterizationStateCreateInfo {
    uint32_t sType;
    const void* pNext;
    VkFlags flags;
    VkBool32 depthClampEnable;
    VkBool32 rasterizerDiscardEnable;
    uint32_t polygonMode;
    VkFlags cullMode;
    uint32_t frontFace;
    VkBool32 depthBiasEnable;
    float depthBiasConstantFactor;
    float depthBiasClamp;
    float depthBiasSlopeFactor;
    float lineWidth;
};

struct VkPipelineMultisampleStateCreateInfo {
    uint32_t sType;
    const void* pNext;
    VkFlags flags;
    uint32_t rasterizationSamples;
    VkBool32 sampleShadingEnable;
    float minSampleShading;
    const uint32_t* pSampleMask;
    VkBool32 alphaToCoverageEnable;
    VkBool32 alphaToOneEnable;
};

struct VkPipelineColorBlendAttachmentState {
    VkBool32 blendEnable;
    uint32_t srcColorBlendFactor;
    uint32_t dstColorBlendFactor;
    uint32_t colorBlendOp;
    uint32_t srcAlphaBlendFactor;
    uint32_t dstAlphaBlendFactor;
    uint32_t alphaBlendOp;
    VkFlags colorWriteMask;
};

struct VkPipelineColorBlendStateCreateInfo {
    uint32_t sType;
    const void* pNext;
    VkFlags flags;
    VkBool32 logicOpEnable;
    uint32_t logicOp;
    uint32_t attachmentCount;
    const VkPipelineColorBlendAttachmentState* pAttachments;
    float blendConstants[4];
};

struct VkGraphicsPipelineCreateInfo {
    uint32_t sType;
    const void* pNext;
    VkFlags flags;
    uint32_t stageCount;
    const VkPipelineShaderStageCreateInfo* pStages;
    const VkPipelineVertexInputStateCreateInfo* pVertexInputState;
    const VkPipelineInputAssemblyStateCreateInfo* pInputAssemblyState;
    const void* pTessellationState;
    const VkPipelineViewportStateCreateInfo* pViewportState;
    const VkPipelineRasterizationStateCreateInfo* pRasterizationState;
    const VkPipelineMultisampleStateCreateInfo* pMultisampleState;
    const void* pDepthStencilState;
    const VkPipelineColorBlendStateCreateInfo* pColorBlendState;
    const void* pDynamicState;
    VkPipelineLayout layout;
    VkRenderPass renderPass;
    uint32_t subpass;
    VkPipeline basePipelineHandle;
    int32_t basePipelineIndex;
};

union VkClearValue {
    float float32[4];
    int32_t int32[4];
    uint32_t uint32[4];
};

struct VkRenderPassBeginInfo {
    uint32_t sType;
    const void* pNext;
    VkRenderPass renderPass;
    VkFramebuffer framebuffer;
    VkRect2D renderArea;
    uint32_t clearValueCount;
    const VkClearValue* pClearValues;
};

struct VkImageSubresourceLayers {
    VkFlags aspectMask;
    uint32_t mipLevel;
    uint32_t baseArrayLayer;
    uint32_t layerCount;
};

struct VkBufferImageCopy {
    VkDeviceSize bufferOffset;
    uint32_t bufferRowLength;
    uint32_t bufferImageHeight;
    VkImageSubresourceLayers imageSubresource;
    VkOffset3D imageOffset;
    VkExtent3D imageExtent;
};

struct Loader {
#if defined(_WIN32)
    HMODULE lib = nullptr;
#else
    void* lib = nullptr;
#endif
    using PFN_vkGetInstanceProcAddr = void* (VK_PTR*)(VkInstance, const char*);
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;

    VkResult (VK_PTR* vkCreateInstance)(const VkInstanceCreateInfo*, const void*, VkInstance*) = nullptr;
    void (VK_PTR* vkDestroyInstance)(VkInstance, const void*) = nullptr;
    VkResult (VK_PTR* vkEnumeratePhysicalDevices)(VkInstance, uint32_t*, VkPhysicalDevice*) = nullptr;
    void (VK_PTR* vkGetPhysicalDeviceProperties)(VkPhysicalDevice, VkPhysicalDeviceProperties*) = nullptr;
    void (VK_PTR* vkGetPhysicalDeviceProperties2)(VkPhysicalDevice, VkPhysicalDeviceProperties2*) = nullptr;
    void (VK_PTR* vkGetPhysicalDeviceQueueFamilyProperties)(VkPhysicalDevice, uint32_t*, VkQueueFamilyProperties*) = nullptr;
    void (VK_PTR* vkGetPhysicalDeviceMemoryProperties)(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties*) = nullptr;
    VkResult (VK_PTR* vkCreateDevice)(VkPhysicalDevice, const VkDeviceCreateInfo*, const void*, VkDevice*) = nullptr;
    void (VK_PTR* vkDestroyDevice)(VkDevice, const void*) = nullptr;
    void (VK_PTR* vkGetDeviceQueue)(VkDevice, uint32_t, uint32_t, VkQueue*) = nullptr;
    VkResult (VK_PTR* vkCreateBuffer)(VkDevice, const VkBufferCreateInfo*, const void*, VkBuffer*) = nullptr;
    void (VK_PTR* vkDestroyBuffer)(VkDevice, VkBuffer, const void*) = nullptr;
    void (VK_PTR* vkGetBufferMemoryRequirements)(VkDevice, VkBuffer, VkMemoryRequirements*) = nullptr;
    VkResult (VK_PTR* vkAllocateMemory)(VkDevice, const VkMemoryAllocateInfo*, const void*, VkDeviceMemory*) = nullptr;
    void (VK_PTR* vkFreeMemory)(VkDevice, VkDeviceMemory, const void*) = nullptr;
    VkResult (VK_PTR* vkBindBufferMemory)(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize) = nullptr;
    VkResult (VK_PTR* vkMapMemory)(VkDevice, VkDeviceMemory, VkDeviceSize, VkDeviceSize, VkFlags, void**) = nullptr;
    void (VK_PTR* vkUnmapMemory)(VkDevice, VkDeviceMemory) = nullptr;
    VkResult (VK_PTR* vkCreateShaderModule)(VkDevice, const VkShaderModuleCreateInfo*, const void*, VkShaderModule*) = nullptr;
    void (VK_PTR* vkDestroyShaderModule)(VkDevice, VkShaderModule, const void*) = nullptr;
    VkResult (VK_PTR* vkCreateDescriptorSetLayout)(VkDevice, const VkDescriptorSetLayoutCreateInfo*, const void*, VkDescriptorSetLayout*) = nullptr;
    void (VK_PTR* vkDestroyDescriptorSetLayout)(VkDevice, VkDescriptorSetLayout, const void*) = nullptr;
    VkResult (VK_PTR* vkCreatePipelineLayout)(VkDevice, const VkPipelineLayoutCreateInfo*, const void*, VkPipelineLayout*) = nullptr;
    void (VK_PTR* vkDestroyPipelineLayout)(VkDevice, VkPipelineLayout, const void*) = nullptr;
    VkResult (VK_PTR* vkCreateComputePipelines)(VkDevice, VkPipelineCache, uint32_t, const VkComputePipelineCreateInfo*, const void*, VkPipeline*) = nullptr;
    void (VK_PTR* vkDestroyPipeline)(VkDevice, VkPipeline, const void*) = nullptr;
    VkResult (VK_PTR* vkCreateDescriptorPool)(VkDevice, const VkDescriptorPoolCreateInfo*, const void*, VkDescriptorPool*) = nullptr;
    void (VK_PTR* vkDestroyDescriptorPool)(VkDevice, VkDescriptorPool, const void*) = nullptr;
    VkResult (VK_PTR* vkAllocateDescriptorSets)(VkDevice, const VkDescriptorSetAllocateInfo*, VkDescriptorSet*) = nullptr;
    void (VK_PTR* vkUpdateDescriptorSets)(VkDevice, uint32_t, const VkWriteDescriptorSet*, uint32_t, const void*) = nullptr;
    VkResult (VK_PTR* vkCreateCommandPool)(VkDevice, const VkCommandPoolCreateInfo*, const void*, VkCommandPool*) = nullptr;
    void (VK_PTR* vkDestroyCommandPool)(VkDevice, VkCommandPool, const void*) = nullptr;
    VkResult (VK_PTR* vkAllocateCommandBuffers)(VkDevice, const VkCommandBufferAllocateInfo*, VkCommandBuffer*) = nullptr;
    VkResult (VK_PTR* vkBeginCommandBuffer)(VkCommandBuffer, const VkCommandBufferBeginInfo*) = nullptr;
    VkResult (VK_PTR* vkEndCommandBuffer)(VkCommandBuffer) = nullptr;
    void (VK_PTR* vkCmdBindPipeline)(VkCommandBuffer, uint32_t, VkPipeline) = nullptr;
    void (VK_PTR* vkCmdBindDescriptorSets)(VkCommandBuffer, uint32_t, VkPipelineLayout, uint32_t, uint32_t, const VkDescriptorSet*, uint32_t, const uint32_t*) = nullptr;
    void (VK_PTR* vkCmdDispatch)(VkCommandBuffer, uint32_t, uint32_t, uint32_t) = nullptr;
    VkResult (VK_PTR* vkQueueSubmit)(VkQueue, uint32_t, const VkSubmitInfo*, VkFence) = nullptr;
    VkResult (VK_PTR* vkQueueWaitIdle)(VkQueue) = nullptr;
    VkResult (VK_PTR* vkResetCommandBuffer)(VkCommandBuffer, VkFlags) = nullptr;
    VkResult (VK_PTR* vkCreateImage)(VkDevice, const VkImageCreateInfo*, const void*, VkImage*) = nullptr;
    void (VK_PTR* vkDestroyImage)(VkDevice, VkImage, const void*) = nullptr;
    void (VK_PTR* vkGetImageMemoryRequirements)(VkDevice, VkImage, VkMemoryRequirements*) = nullptr;
    VkResult (VK_PTR* vkBindImageMemory)(VkDevice, VkImage, VkDeviceMemory, VkDeviceSize) = nullptr;
    VkResult (VK_PTR* vkCreateImageView)(VkDevice, const VkImageViewCreateInfo*, const void*, VkImageView*) = nullptr;
    void (VK_PTR* vkDestroyImageView)(VkDevice, VkImageView, const void*) = nullptr;
    VkResult (VK_PTR* vkCreateRenderPass)(VkDevice, const VkRenderPassCreateInfo*, const void*, VkRenderPass*) = nullptr;
    void (VK_PTR* vkDestroyRenderPass)(VkDevice, VkRenderPass, const void*) = nullptr;
    VkResult (VK_PTR* vkCreateFramebuffer)(VkDevice, const VkFramebufferCreateInfo*, const void*, VkFramebuffer*) = nullptr;
    void (VK_PTR* vkDestroyFramebuffer)(VkDevice, VkFramebuffer, const void*) = nullptr;
    VkResult (VK_PTR* vkCreateGraphicsPipelines)(VkDevice, VkPipelineCache, uint32_t, const VkGraphicsPipelineCreateInfo*, const void*, VkPipeline*) = nullptr;
    void (VK_PTR* vkCmdBeginRenderPass)(VkCommandBuffer, const VkRenderPassBeginInfo*, uint32_t) = nullptr;
    void (VK_PTR* vkCmdEndRenderPass)(VkCommandBuffer) = nullptr;
    void (VK_PTR* vkCmdBindVertexBuffers)(VkCommandBuffer, uint32_t, uint32_t, const VkBuffer*, const VkDeviceSize*) = nullptr;
    void (VK_PTR* vkCmdDraw)(VkCommandBuffer, uint32_t, uint32_t, uint32_t, uint32_t) = nullptr;
    void (VK_PTR* vkCmdCopyImageToBuffer)(VkCommandBuffer, VkImage, uint32_t, VkBuffer, uint32_t, const VkBufferImageCopy*) = nullptr;

    ~Loader() {
#if defined(_WIN32)
        if (lib) FreeLibrary(lib);
#else
        if (lib) dlclose(lib);
#endif
    }

    bool loadLibrary() {
#if defined(_WIN32)
        lib = LoadLibraryA("vulkan-1.dll");
        if (!lib) return false;
        vkGetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetProcAddress(lib, "vkGetInstanceProcAddr"));
#elif defined(__APPLE__)
        const char* candidates[] = {"libvulkan.1.dylib", "libvulkan.dylib", "/usr/local/lib/libvulkan.1.dylib", "/opt/homebrew/lib/libvulkan.1.dylib"};
        for (const char* p : candidates) {
            lib = dlopen(p, RTLD_NOW | RTLD_LOCAL);
            if (lib) break;
        }
        if (!lib) return false;
        vkGetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(lib, "vkGetInstanceProcAddr"));
#else
        const char* candidates[] = {"libvulkan.so.1", "libvulkan.so"};
        for (const char* p : candidates) {
            lib = dlopen(p, RTLD_NOW | RTLD_LOCAL);
            if (lib) break;
        }
        if (!lib) return false;
        vkGetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(lib, "vkGetInstanceProcAddr"));
#endif
        if (!vkGetInstanceProcAddr) return false;
        vkCreateInstance = reinterpret_cast<decltype(vkCreateInstance)>(vkGetInstanceProcAddr(nullptr, "vkCreateInstance"));
        return vkCreateInstance != nullptr;
    }

    template <typename Fn>
    void loadSym(VkInstance inst, Fn& out, const char* name) {
        out = reinterpret_cast<Fn>(vkGetInstanceProcAddr(inst, name));
    }

    bool loadInstanceSyms(VkInstance inst) {
        loadSym(inst, vkDestroyInstance, "vkDestroyInstance");
        loadSym(inst, vkEnumeratePhysicalDevices, "vkEnumeratePhysicalDevices");
        loadSym(inst, vkGetPhysicalDeviceProperties, "vkGetPhysicalDeviceProperties");
        loadSym(inst, vkGetPhysicalDeviceProperties2, "vkGetPhysicalDeviceProperties2");
        if (!vkGetPhysicalDeviceProperties2) {
            loadSym(inst, vkGetPhysicalDeviceProperties2, "vkGetPhysicalDeviceProperties2KHR");
        }
        loadSym(inst, vkGetPhysicalDeviceQueueFamilyProperties, "vkGetPhysicalDeviceQueueFamilyProperties");
        loadSym(inst, vkGetPhysicalDeviceMemoryProperties, "vkGetPhysicalDeviceMemoryProperties");
        loadSym(inst, vkCreateDevice, "vkCreateDevice");
        loadSym(inst, vkDestroyDevice, "vkDestroyDevice");
        loadSym(inst, vkGetDeviceQueue, "vkGetDeviceQueue");
        loadSym(inst, vkCreateBuffer, "vkCreateBuffer");
        loadSym(inst, vkDestroyBuffer, "vkDestroyBuffer");
        loadSym(inst, vkGetBufferMemoryRequirements, "vkGetBufferMemoryRequirements");
        loadSym(inst, vkAllocateMemory, "vkAllocateMemory");
        loadSym(inst, vkFreeMemory, "vkFreeMemory");
        loadSym(inst, vkBindBufferMemory, "vkBindBufferMemory");
        loadSym(inst, vkMapMemory, "vkMapMemory");
        loadSym(inst, vkUnmapMemory, "vkUnmapMemory");
        loadSym(inst, vkCreateShaderModule, "vkCreateShaderModule");
        loadSym(inst, vkDestroyShaderModule, "vkDestroyShaderModule");
        loadSym(inst, vkCreateDescriptorSetLayout, "vkCreateDescriptorSetLayout");
        loadSym(inst, vkDestroyDescriptorSetLayout, "vkDestroyDescriptorSetLayout");
        loadSym(inst, vkCreatePipelineLayout, "vkCreatePipelineLayout");
        loadSym(inst, vkDestroyPipelineLayout, "vkDestroyPipelineLayout");
        loadSym(inst, vkCreateComputePipelines, "vkCreateComputePipelines");
        loadSym(inst, vkDestroyPipeline, "vkDestroyPipeline");
        loadSym(inst, vkCreateDescriptorPool, "vkCreateDescriptorPool");
        loadSym(inst, vkDestroyDescriptorPool, "vkDestroyDescriptorPool");
        loadSym(inst, vkAllocateDescriptorSets, "vkAllocateDescriptorSets");
        loadSym(inst, vkUpdateDescriptorSets, "vkUpdateDescriptorSets");
        loadSym(inst, vkCreateCommandPool, "vkCreateCommandPool");
        loadSym(inst, vkDestroyCommandPool, "vkDestroyCommandPool");
        loadSym(inst, vkAllocateCommandBuffers, "vkAllocateCommandBuffers");
        loadSym(inst, vkBeginCommandBuffer, "vkBeginCommandBuffer");
        loadSym(inst, vkEndCommandBuffer, "vkEndCommandBuffer");
        loadSym(inst, vkCmdBindPipeline, "vkCmdBindPipeline");
        loadSym(inst, vkCmdBindDescriptorSets, "vkCmdBindDescriptorSets");
        loadSym(inst, vkCmdDispatch, "vkCmdDispatch");
        loadSym(inst, vkQueueSubmit, "vkQueueSubmit");
        loadSym(inst, vkQueueWaitIdle, "vkQueueWaitIdle");
        loadSym(inst, vkResetCommandBuffer, "vkResetCommandBuffer");
        loadSym(inst, vkCreateImage, "vkCreateImage");
        loadSym(inst, vkDestroyImage, "vkDestroyImage");
        loadSym(inst, vkGetImageMemoryRequirements, "vkGetImageMemoryRequirements");
        loadSym(inst, vkBindImageMemory, "vkBindImageMemory");
        loadSym(inst, vkCreateImageView, "vkCreateImageView");
        loadSym(inst, vkDestroyImageView, "vkDestroyImageView");
        loadSym(inst, vkCreateRenderPass, "vkCreateRenderPass");
        loadSym(inst, vkDestroyRenderPass, "vkDestroyRenderPass");
        loadSym(inst, vkCreateFramebuffer, "vkCreateFramebuffer");
        loadSym(inst, vkDestroyFramebuffer, "vkDestroyFramebuffer");
        loadSym(inst, vkCreateGraphicsPipelines, "vkCreateGraphicsPipelines");
        loadSym(inst, vkCmdBeginRenderPass, "vkCmdBeginRenderPass");
        loadSym(inst, vkCmdEndRenderPass, "vkCmdEndRenderPass");
        loadSym(inst, vkCmdBindVertexBuffers, "vkCmdBindVertexBuffers");
        loadSym(inst, vkCmdDraw, "vkCmdDraw");
        loadSym(inst, vkCmdCopyImageToBuffer, "vkCmdCopyImageToBuffer");
        return vkEnumeratePhysicalDevices && vkCreateDevice && vkCreateComputePipelines && vkCmdDispatch;
    }
};

} // namespace vkmin
