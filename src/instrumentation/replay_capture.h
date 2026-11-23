#pragma once
#if defined(NACELLE_ENABLE_INSTRUMENTATION)
#include <nacelle/replay.h>
#include "replay_format.h"
#include "../api/command_list.h"

namespace nacelle::replay {
struct Transaction;
struct Execution {
    explicit Execution(std::string& error);
    ~Execution();
    bool valid = false;
};
struct Submission {
    explicit Submission(NacelleCommandQueue* queue);
    ~Submission();
    std::shared_ptr<Transaction> transaction;
    Transaction* previous = nullptr;
};
void observeCommand(const RecordedCommand& command, const ExecutionState& state);
void observeEntry(NacellePipelineState* pipeline, std::string_view nativeId, uint32_t tier);
bool provenanceEnabled();
void observeList(bool eligible, bool unsupported, D3D12_COMMAND_LIST_TYPE type, ID3D12Device* device);
void observeResource(NacelleResource* resource, const D3D12_HEAP_PROPERTIES& properties);
void observeRoot(NacelleRootSignature* root, const void* bytes, size_t size);
void observePipeline(NacellePipelineState* pipeline, const D3D12_COMPUTE_PIPELINE_STATE_DESC& desc);
void observeHeap(NacelleDescriptorHeap* heap, const D3D12_DESCRIPTOR_HEAP_DESC& desc);
void observeQueue(NacelleCommandQueue* queue);
void observeCBV(const D3D12_CONSTANT_BUFFER_VIEW_DESC* desc, D3D12_CPU_DESCRIPTOR_HANDLE destination);
void observeSRV(ID3D12Resource* resource, const D3D12_SHADER_RESOURCE_VIEW_DESC* desc, D3D12_CPU_DESCRIPTOR_HANDLE destination);
void observeUAV(ID3D12Resource* resource, ID3D12Resource* counter, const D3D12_UNORDERED_ACCESS_VIEW_DESC* desc,
                D3D12_CPU_DESCRIPTOR_HANDLE destination);
void observeDescriptorCopy(UINT destinations, const D3D12_CPU_DESCRIPTOR_HANDLE* destinationStarts,
    const UINT* destinationSizes, UINT sources, const D3D12_CPU_DESCRIPTOR_HANDLE* sourceStarts, const UINT* sourceSizes);
void observeMutation(const char* operation);
void observeMap(NacelleResource* resource, bool mapping);
Environment currentEnvironment();
bool sameEnvironment(const Environment& first, const Environment& second);
bool validateShader(std::span<const char> bytes, bool sourceConsent, std::string& error);
bool validateSemantics(const Bundle& bundle, std::string& error);
bool executeBundle(const Bundle& bundle, uint32_t repetitions, std::string& error);
}
#endif
