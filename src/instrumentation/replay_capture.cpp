#include "replay_capture.h"
#include "../compiler/jit_engine.h"
#include <algorithm>
#include <atomic>
#include <bit>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <new>
#include <set>
#include <tuple>

namespace nacelle::replay {
namespace {
template <class T> std::shared_ptr<T> retain(T* object) {
    object->AddRef();
    return {object, [](T* value) { value->Release(); }};
}
struct ResourceRecord { std::shared_ptr<NacelleResource> object; uint32_t heapType = 0, maps = 0; };
struct RootRecord { std::shared_ptr<NacelleRootSignature> object; std::string bytes; };
struct PipelineRecord {
    std::shared_ptr<NacellePipelineState> object;
    std::string bytes, error;
    uint32_t nodeMask = 0, flags = 0;
};
struct HeapRecord {
    std::shared_ptr<NacelleDescriptorHeap> object;
    uint32_t flags = 0;
    std::map<uint32_t, View> views;
};
struct Store {
    NacelleReplayConfiguration configuration;
    uint64_t resourceBytes = 0, shaderBytes = 0, descriptorSlots = 0;
    std::string error;
    std::map<NacelleResource*, ResourceRecord> resources;
    std::map<NacelleRootSignature*, RootRecord> roots;
    std::map<NacellePipelineState*, PipelineRecord> pipelines;
    std::map<NacelleDescriptorHeap*, HeapRecord> heaps;
    std::map<NacelleCommandQueue*, std::pair<std::shared_ptr<NacelleCommandQueue>, Environment>> queues;
};
std::mutex mutex;
std::condition_variable drained;
std::shared_ptr<Store> configured;
std::shared_ptr<Transaction> active;
std::atomic<bool> enabled{false};
thread_local Transaction* submission = nullptr;
thread_local bool replaying = false;
uint32_t executors = 0;
void log(const std::string& error) { std::fprintf(stderr, "[NacelleReplay] %s\n", error.c_str()); }
}

struct Transaction {
    std::shared_ptr<Store> store;
    Bundle bundle;
    ID3D12Device* device = nullptr;
    std::shared_ptr<NacelleCommandQueue> queue;
    uint64_t inFlight = 0, commandWords = 0;
    bool accepting = true, stopped = false;
    std::string error;
    std::map<NacelleResource*, uint32_t> resources;
    std::map<NacelleRootSignature*, uint32_t> roots;
    std::map<NacellePipelineState*, uint32_t> pipelines;
    std::map<NacelleDescriptorHeap*, uint32_t> heaps;
    void fail(std::string message) {
        if (error.empty()) { error = std::move(message); log(error); }
    }
    void command(std::string kind, std::vector<uint64_t> data = {}) {
        if (!error.empty()) return;
        if (bundle.metadata.commands.size() == maxCommands ||
            data.size() > (store->configuration.command_bytes / 8) - commandWords) {
            fail("Replay command budget exceeded");
            return;
        }
        commandWords += data.size();
        bundle.metadata.commands.push_back({std::move(kind), std::move(data)});
    }
    uint32_t append(std::string_view bytes, const char* role) {
        if (bundle.bytes.size() > profile::maxProfileBytes - bytes.size()) {
            fail("Replay aggregate payload exceeds 256 MiB");
            return 0;
        }
        const auto id = static_cast<uint32_t>(bundle.metadata.blobs.size() + 1);
        bundle.metadata.blobs.push_back({id, bundle.bytes.size(), bytes.size(), role, contentDigest(bytes)});
        bundle.bytes.append(bytes);
        return id;
    }
    uint32_t resource(NacelleResource* value) {
        const auto found = resources.find(value);
        if (found == resources.end()) { fail("Replay resource was not declared or has no creation provenance"); return 0; }
        return found->second;
    }
    std::pair<uint32_t, uint64_t> address(uint64_t value) {
        for (const auto& [object, id] : resources) {
            const uint64_t start = reinterpret_cast<uint64_t>(object->basePtr());
            if (value >= start && value - start < object->totalBytes()) return {id, value - start};
        }
        fail("Replay address is outside declared owned buffers");
        return {};
    }
    uint32_t root(NacelleRootSignature* value) {
        if (!value || value->parentDevice() != device) { fail("Replay root signature/device is invalid"); return 0; }
        if (const auto found = roots.find(value); found != roots.end()) return found->second;
        const auto source = store->roots.find(value);
        if (source == store->roots.end()) { fail("Replay root signature lacks creation provenance"); return 0; }
        const auto& layout = value->layout();
        if (!layout.staticSamplers.empty()) { fail("Replay static samplers are unsupported"); return 0; }
        for (const auto& parameter : layout.params)
            for (const auto& range : parameter.ranges)
                if (range.numDescriptors == UINT32_MAX || range.type == D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER) {
                    fail("Replay unbounded/sampler root ranges are unsupported");
                    return 0;
                }
        const auto id = static_cast<uint32_t>(roots.size() + 1);
        roots[value] = id;
        bundle.metadata.roots.push_back({id, append(source->second.bytes, "root-signature")});
        return id;
    }
    uint32_t pipeline(NacellePipelineState* value) {
        if (!value || value->parentDevice() != device || !value->isCompute()) {
            fail("Replay supports compute pipelines on the captured device only"); return 0;
        }
        const auto source = store->pipelines.find(value);
        if (source == store->pipelines.end()) { fail("Replay pipeline lacks creation provenance"); return 0; }
        if (!source->second.error.empty()) { fail(source->second.error); return 0; }
        const auto* entry = value->computePSO()->profileEntries.load();
        if (!entry || !entry->fn || entry->tier != 2 || entry->artifactId.empty() ||
            value->computePSO()->simdWidth != bundle.manifest.environment.simdWidth) {
            fail("Replay requires quiesced stable Tier-2 code with exact identity and SIMD"); return 0;
        }
        if (const auto found = pipelines.find(value); found != pipelines.end()) {
            if (bundle.metadata.pipelines[found->second - 1].nativeId != entry->artifactId)
                fail("Replay executed code identity changed");
            return found->second;
        }
        const auto rootId = root(value->rootSignature());
        if (!rootId) return 0;
        const auto id = static_cast<uint32_t>(pipelines.size() + 1);
        const auto blob = append(source->second.bytes, "dxil");
        pipelines[value] = id;
        bundle.metadata.pipelines.push_back({id, rootId, blob, source->second.nodeMask, source->second.flags,
                                            contentDigest(source->second.bytes), entry->artifactId});
        return id;
    }
    uint32_t heap(NacelleDescriptorHeap* value) {
        if (!value || value->parentDevice() != device) { fail("Replay descriptor heap/device is invalid"); return 0; }
        if (const auto found = heaps.find(value); found != heaps.end()) return found->second;
        const auto source = store->heaps.find(value);
        if (source == store->heaps.end()) { fail("Replay heap lacks supported creation provenance"); return 0; }
        const auto id = static_cast<uint32_t>(heaps.size() + 1);
        heaps[value] = id;
        bundle.metadata.heaps.push_back({id, value->count(), source->second.flags});
        return id;
    }
    std::pair<uint32_t, uint32_t> descriptor(uint64_t handle, bool retainView = true) {
        for (const auto& [object, source] : store->heaps) {
            const auto start = reinterpret_cast<uint64_t>(object->data());
            if (handle < start || handle - start >= uint64_t(object->count()) * sizeof(NacelleDescriptor) ||
                (handle - start) % sizeof(NacelleDescriptor)) continue;
            const auto slot = static_cast<uint32_t>((handle - start) / sizeof(NacelleDescriptor));
            if (!retainView) return {heap(object), slot};
            const auto found = source.views.find(slot);
            if (found == source.views.end()) { fail("Replay descriptor has no buffer-view provenance"); return {}; }
            auto view = found->second;
            const auto sourceResource = sourceResources.find({object, slot});
            if (sourceResource == sourceResources.end()) { fail("Replay descriptor resource provenance is missing"); return {}; }
            view.resource = resource(sourceResource->second);
            view.heap = heap(object);
            if (!view.resource || !view.heap) return {};
            if (std::none_of(bundle.metadata.views.begin(), bundle.metadata.views.end(),
                [&](const View& item) { return item.heap == view.heap && item.slot == slot; })) {
                if (bundle.metadata.views.size() == maxCommands) { fail("Replay descriptor limit exceeded"); return {}; }
                view.slot = slot;
                bundle.metadata.views.push_back(view);
            }
            return {view.heap, slot};
        }
        fail("Replay descriptor handle is outside retained heaps");
        return {};
    }
    std::map<std::pair<NacelleDescriptorHeap*, uint32_t>, NacelleResource*> sourceResources;
};

namespace {
std::map<std::pair<NacelleDescriptorHeap*, uint32_t>, NacelleResource*> viewResources;
void failure(Store& store, const std::string& error) {
    if (store.error.empty()) { store.error = error; log(error); }
    if (active) active->fail(error);
}
template <class F> void observe(F&& function) {
    if (replaying) return;
    if (!enabled.load(std::memory_order_relaxed)) return;
    std::lock_guard lock(mutex);
    if (!configured || !configured->error.empty()) return;
    if (active) { active->fail("Replay object/descriptor mutation during an active transaction"); return; }
    try { function(*configured); }
    catch (const std::bad_alloc&) { failure(*configured, "Insufficient memory retaining replay provenance"); }
}
std::pair<HeapRecord*, uint32_t> slot(Store& store, uint64_t handle) {
    for (auto& [object, heap] : store.heaps) {
        const auto start = reinterpret_cast<uint64_t>(object->data());
        if (handle >= start && handle - start < uint64_t(object->count()) * sizeof(NacelleDescriptor) &&
            (handle - start) % sizeof(NacelleDescriptor) == 0)
            return {&heap, static_cast<uint32_t>((handle - start) / sizeof(NacelleDescriptor))};
    }
    failure(store, "Replay descriptor creation/copy targets an untracked heap");
    return {};
}
void view(Store& store, uint64_t destination, NacelleResource* object, View value) {
    auto [heap, index] = slot(store, destination);
    if (!heap) return;
    if (!object || !store.resources.contains(object)) {
        heap->views.erase(index);
        viewResources.erase({heap->object.get(), index});
        return;
    }
    value.slot = index;
    heap->views[index] = std::move(value);
    viewResources[{heap->object.get(), index}] = object;
}
}

void observeResource(NacelleResource* object, const D3D12_HEAP_PROPERTIES& properties) {
    observe([&](Store& store) {
        if (object->resourceDesc().Dimension != D3D12_RESOURCE_DIMENSION_BUFFER || object->isReserved() ||
            properties.Type < D3D12_HEAP_TYPE_DEFAULT || properties.Type > D3D12_HEAP_TYPE_READBACK) return;
        if (store.resources.size() == maxObjects || object->totalBytes() > store.configuration.snapshot_bytes / 2 -
            std::min(store.resourceBytes, store.configuration.snapshot_bytes / 2)) {
            failure(store, "Replay retained resource budget/object limit exceeded"); return;
        }
        store.resourceBytes += object->totalBytes();
        store.resources.emplace(object, ResourceRecord{retain(object), static_cast<uint32_t>(properties.Type)});
    });
}
void observeRoot(NacelleRootSignature* object, const void* bytes, size_t size) {
    observe([&](Store& store) {
        if (store.roots.size() == maxObjects || size > store.configuration.shader_bytes - store.shaderBytes) {
            failure(store, "Replay root/shader retention budget exceeded"); return;
        }
        store.shaderBytes += size;
        store.roots.emplace(object, RootRecord{retain(object), {static_cast<const char*>(bytes), size}});
    });
}
void observePipeline(NacellePipelineState* object, const D3D12_COMPUTE_PIPELINE_STATE_DESC& desc) {
    observe([&](Store& store) {
        if (store.pipelines.size() == maxObjects ||
            desc.CS.BytecodeLength > store.configuration.shader_bytes - store.shaderBytes) {
            failure(store, "Replay shader retention budget exceeded"); return;
        }
        PipelineRecord record{retain(object), {}, {}, desc.NodeMask, static_cast<uint32_t>(desc.Flags)};
        const std::span<const char> bytes{static_cast<const char*>(desc.CS.pShaderBytecode), desc.CS.BytecodeLength};
        if (validateShader(bytes, (store.configuration.flags & NACELLE_REPLAY_EMBEDDED_SOURCES) != 0, record.error)) {
            record.bytes.assign(bytes.data(), bytes.size());
            store.shaderBytes += bytes.size();
        }
        store.pipelines.emplace(object, std::move(record));
    });
}
void observeHeap(NacelleDescriptorHeap* object, const D3D12_DESCRIPTOR_HEAP_DESC& desc) {
    observe([&](Store& store) {
        if (desc.Type != D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) return;
        if (store.heaps.size() == maxObjects || desc.NumDescriptors > maxCommands - store.descriptorSlots) {
            failure(store, "Replay descriptor heap limit exceeded"); return;
        }
        store.descriptorSlots += desc.NumDescriptors;
        store.heaps.emplace(object, HeapRecord{retain(object), static_cast<uint32_t>(desc.Flags), {}});
    });
}
void observeQueue(NacelleCommandQueue* object) {
    observe([&](Store& store) {
        if (store.queues.size() == maxObjects) { failure(store, "Replay queue provenance limit exceeded"); return; }
        store.queues.emplace(object, std::make_pair(retain(object), currentEnvironment()));
    });
}
void observeCBV(const D3D12_CONSTANT_BUFFER_VIEW_DESC* desc, D3D12_CPU_DESCRIPTOR_HANDLE destination) {
    observe([&](Store& store) {
        if (!desc) return;
        for (const auto& [object, record] : store.resources) {
            const auto start = reinterpret_cast<uint64_t>(object->basePtr());
            if (desc->BufferLocation >= start && desc->BufferLocation - start < object->totalBytes()) {
                View value;
                value.kind = "cbv";
                value.first = desc->BufferLocation - start;
                value.bytes = desc->SizeInBytes;
                view(store, destination.ptr, object, value);
                return;
            }
        }
        view(store, destination.ptr, nullptr, {});
    });
}
void observeSRV(ID3D12Resource* resource, const D3D12_SHADER_RESOURCE_VIEW_DESC* desc, D3D12_CPU_DESCRIPTOR_HANDLE destination) {
    observe([&](Store& store) {
        auto* object = static_cast<NacelleResource*>(resource);
        if (!store.resources.contains(object) || !desc || desc->ViewDimension != D3D12_SRV_DIMENSION_BUFFER) {
            view(store, destination.ptr, nullptr, {}); return;
        }
        View value;
        value.kind = "srv";
        value.first = desc->Buffer.FirstElement;
        value.count = desc->Buffer.NumElements;
        value.stride = desc->Buffer.StructureByteStride ? desc->Buffer.StructureByteStride :
            desc->Format == DXGI_FORMAT_UNKNOWN ? 4 : getFormatInfo(desc->Format).bytesPerBlock;
        value.format = desc->Format;
        value.flags = desc->Buffer.Flags;
        if (value.flags) value.stride = 4;
        view(store, destination.ptr, object, value);
    });
}
void observeUAV(ID3D12Resource* resource, ID3D12Resource* counter, const D3D12_UNORDERED_ACCESS_VIEW_DESC* desc,
                D3D12_CPU_DESCRIPTOR_HANDLE destination) {
    observe([&](Store& store) {
        auto* object = static_cast<NacelleResource*>(resource);
        if (!store.resources.contains(object) || counter || !desc || desc->ViewDimension != D3D12_UAV_DIMENSION_BUFFER ||
            desc->Buffer.CounterOffsetInBytes) { view(store, destination.ptr, nullptr, {}); return; }
        View value;
        value.kind = "uav";
        value.first = desc->Buffer.FirstElement;
        value.count = desc->Buffer.NumElements;
        value.stride = desc->Buffer.StructureByteStride ? desc->Buffer.StructureByteStride :
            desc->Format == DXGI_FORMAT_UNKNOWN ? 4 : getFormatInfo(desc->Format).bytesPerBlock;
        value.format = desc->Format;
        value.flags = desc->Buffer.Flags;
        if (value.flags) value.stride = 4;
        view(store, destination.ptr, object, value);
    });
}
void observeDescriptorCopy(UINT destinations, const D3D12_CPU_DESCRIPTOR_HANDLE* destinationStarts,
    const UINT* destinationSizes, UINT sources, const D3D12_CPU_DESCRIPTOR_HANDLE* sourceStarts, const UINT* sourceSizes) {
    observe([&](Store& store) {
        UINT sourceRange = 0, sourceIndex = 0;
        for (UINT range = 0; range < destinations; ++range) {
            const UINT count = destinationSizes ? destinationSizes[range] : 1;
            for (UINT index = 0; index < count; ++index) {
                while (sourceRange < sources && sourceIndex >= (sourceSizes ? sourceSizes[sourceRange] : 1)) {
                    ++sourceRange; sourceIndex = 0;
                }
                if (sourceRange == sources) { failure(store, "Replay descriptor copy ranges disagree"); return; }
                auto [source, slotIndex] = slot(store, sourceStarts[sourceRange].ptr + sourceIndex * sizeof(NacelleDescriptor));
                if (!source) return;
                const auto found = source->views.find(slotIndex);
                if (found == source->views.end()) { failure(store, "Replay copied descriptor has no provenance"); return; }
                const auto object = viewResources.at({source->object.get(), slotIndex});
                view(store, destinationStarts[range].ptr + index * sizeof(NacelleDescriptor), object, found->second);
                ++sourceIndex;
            }
        }
    });
}
void observeMutation(const char* operation) {
    if (replaying) return;
    if (!enabled.load(std::memory_order_relaxed)) return;
    std::lock_guard lock(mutex);
    if (active) active->fail(std::string("Unsupported mutation during replay capture: ") + operation);
}
void observeMap(NacelleResource* resource, bool mapping) {
    if (replaying) return;
    if (!enabled.load(std::memory_order_relaxed)) return;
    std::lock_guard lock(mutex);
    if (!configured) return;
    if (active) active->fail("Resource Map/Unmap during replay capture");
    if (const auto found = configured->resources.find(resource); found != configured->resources.end()) {
        if (mapping && found->second.maps == UINT32_MAX) failure(*configured, "Replay open-map counter limit exceeded");
        else if (mapping) ++found->second.maps;
        else if (found->second.maps) --found->second.maps;
    }
}

bool provenanceEnabled() { return enabled.load(std::memory_order_relaxed); }
Execution::Execution(std::string& error) {
    std::lock_guard lock(mutex);
    if (active || replaying) { error = "Replay execution cannot overlap capture or recursive execution"; return; }
    replaying = true;
    ++executors;
    valid = true;
}
Execution::~Execution() {
    if (!valid) return;
    std::lock_guard lock(mutex);
    replaying = false;
    --executors;
}
void observeList(bool eligible, bool unsupported, D3D12_COMMAND_LIST_TYPE type, ID3D12Device* device) {
    if (!submission) return;
    std::lock_guard lock(mutex);
    if (!eligible || unsupported || device != submission->device ||
        (type != D3D12_COMMAND_LIST_TYPE_DIRECT && type != D3D12_COMMAND_LIST_TYPE_COMPUTE))
        submission->fail("Replay command list was untracked or used unsupported commands/type");
}

Submission::Submission(NacelleCommandQueue* queue) {
    previous = submission;
    if (!enabled.load(std::memory_order_relaxed)) return;
    std::lock_guard lock(mutex);
    if (!active) return;
    if (!active->accepting) { active->fail("Submission after replay capture admission closed"); return; }
    transaction = active;
    ++transaction->inFlight;
    if (!transaction->queue) transaction->queue = retain(queue);
    if (transaction->queue.get() != queue || queue->parentDevice() != transaction->device)
        transaction->fail("Replay supports one queue on one captured device only");
    const auto provenance = transaction->store->queues.find(queue);
    if (provenance == transaction->store->queues.end() ||
        !sameEnvironment(provenance->second.second, transaction->bundle.manifest.environment))
        transaction->fail("Replay queue lacks matching creation/FP-environment provenance");
    transaction->command("submission");
    submission = transaction.get();
}
Submission::~Submission() {
    submission = previous;
    if (!transaction) return;
    std::lock_guard lock(mutex);
    --transaction->inFlight;
    drained.notify_all();
}

void observeEntry(NacellePipelineState* pipeline, std::string_view nativeId, uint32_t tier) {
    if (!submission) return;
    std::lock_guard lock(mutex);
    const auto found = submission->pipelines.find(pipeline);
    if (tier != 2 || found == submission->pipelines.end() ||
        submission->bundle.metadata.pipelines[found->second - 1].nativeId != nativeId)
        submission->fail("Replay exact executed entry changed from stable Tier-2 capture identity");
}

void observeCommand(const RecordedCommand& command, const ExecutionState& state) {
    if (!submission) return;
    std::lock_guard lock(mutex);
    auto& capture = *submission;
    if (!capture.error.empty()) return;
    try {
        std::visit([&](const auto& value) {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, CmdSetPipelineState>)
                capture.command("pipeline", {capture.pipeline(value.pso)});
            else if constexpr (std::is_same_v<T, CmdSetRootSignature>) {
                if (!value.compute) capture.fail("Replay graphics root bindings are unsupported");
                else capture.command("root", {capture.root(value.sig)});
            } else if constexpr (std::is_same_v<T, CmdSetDescriptorHeaps>) {
                if (value.sampHeap || !value.resHeap) capture.fail("Replay requires a resource-only descriptor heap");
                else capture.command("heap", {capture.heap(value.resHeap)});
            } else if constexpr (std::is_same_v<T, CmdSetRootDescTable>) {
                if (!value.compute) capture.fail("Replay graphics descriptor tables are unsupported");
                else {
                    const auto [heap, slot] = capture.descriptor(value.handle.ptr, false);
                    capture.command("table", {value.idx, heap, slot});
                }
            } else if constexpr (std::is_same_v<T, CmdSetRoot32BitConsts>) {
                if (!value.compute) capture.fail("Replay graphics constants are unsupported");
                else {
                    std::vector<uint64_t> words{value.idx, value.destOffset};
                    words.insert(words.end(), value.data.begin(), value.data.end());
                    capture.command("constants", std::move(words));
                }
            } else if constexpr (std::is_same_v<T, CmdSetRootView>) {
                if (!value.compute) capture.fail("Replay graphics root views are unsupported");
                else {
                    const auto [resource, offset] = capture.address(value.addr);
                    capture.command("view", {value.idx, resource, offset});
                }
            } else if constexpr (std::is_same_v<T, CmdDispatch>) {
                capture.pipeline(state.pso);
                if (state.pso && state.pso->rootSignature())
                    for (const auto& parameter : state.pso->rootSignature()->layout().params)
                        for (const auto& range : parameter.ranges) {
                            for (uint32_t index = 0; index < range.numDescriptors; ++index) {
                                if (range.numDescriptors > maxCommands) { capture.fail("Replay root descriptor range limit exceeded"); break; }
                                const auto parameterIndex = &parameter - state.pso->rootSignature()->layout().params.data();
                                const auto handle = reinterpret_cast<uint64_t>(state.computeCtx.root_slots[parameterIndex]);
                                capture.descriptor(handle + uint64_t(range.offsetInTable + index) * sizeof(NacelleDescriptor));
                            }
                        }
                capture.command("dispatch", {value.x, value.y, value.z});
            } else if constexpr (std::is_same_v<T, CmdCopyBufferRegion>)
                capture.command("copy", {capture.resource(value.dst), value.dstOff, capture.resource(value.src), value.srcOff, value.bytes});
            else if constexpr (std::is_same_v<T, CmdCopyResource>)
                capture.command("copy", {capture.resource(value.dst), 0, capture.resource(value.src), 0,
                    value.src && capture.resources.contains(value.src) ? value.src->totalBytes() : 0});
            else if constexpr (std::is_same_v<T, CmdClearUAV>) {
                if (!value.rects.empty()) { capture.fail("Replay partial buffer UAV clears are unsupported"); return; }
                const auto fields = [](const NacelleDescriptor& descriptor) {
                    return std::tie(descriptor.base_ptr, descriptor.width, descriptor.height, descriptor.depth_or_array,
                        descriptor.row_pitch, descriptor.format, descriptor.mip_levels, descriptor.sampler_flags,
                        descriptor.elem_stride, descriptor.byte_size, descriptor.extra_ptr, descriptor.mip_lod_bias,
                        descriptor.view_flags);
                };
                for (const auto& [heapObject, source] : capture.store->heaps)
                    for (const auto& [slot, view] : source.views) {
                        if (view.kind != "uav" || fields(heapObject->data()[slot]) != fields(value.desc)) continue;
                        const auto found = capture.sourceResources.find({heapObject, slot});
                        if (found == capture.sourceResources.end() || found->second != value.res) continue;
                        const auto [heap, index] = capture.descriptor(reinterpret_cast<uint64_t>(&heapObject->data()[slot]));
                        std::vector<uint64_t> words{heap, index, capture.resource(value.res), value.isFloat ? 1u : 0u};
                        for (size_t component = 0; component < 4; ++component)
                            words.push_back(value.isFloat ? std::bit_cast<uint32_t>(value.fVal[component]) : value.uVal[component]);
                        capture.command("clear", std::move(words));
                        return;
                    }
                capture.fail("Replay UAV clear has no unchanged descriptor provenance");
            } else if constexpr (std::is_same_v<T, CmdReplayBarrier>) {
                for (const auto& barrier : value.barriers) {
                    if (barrier.Type == D3D12_RESOURCE_BARRIER_TYPE_TRANSITION)
                        capture.command("barrier", {0, capture.resource(static_cast<NacelleResource*>(barrier.Transition.pResource)),
                            barrier.Transition.Subresource, static_cast<uint32_t>(barrier.Transition.StateBefore),
                            static_cast<uint32_t>(barrier.Transition.StateAfter), static_cast<uint32_t>(barrier.Flags)});
                    else if (barrier.Type == D3D12_RESOURCE_BARRIER_TYPE_UAV)
                        capture.command("barrier", {1, barrier.UAV.pResource ?
                            capture.resource(static_cast<NacelleResource*>(barrier.UAV.pResource)) : 0, 0, 0, 0,
                            static_cast<uint32_t>(barrier.Flags)});
                    else capture.fail("Replay aliasing barriers are unsupported");
                }
            } else capture.fail("Unsupported command in compute replay transaction");
        }, command);
    } catch (const std::bad_alloc&) { capture.fail("Insufficient memory recording replay commands"); }
}
}

struct NacelleReplayCapture { std::shared_ptr<nacelle::replay::Transaction> transaction; };

extern "C" HRESULT NacelleConfigureReplay(const NacelleReplayConfiguration* configuration) {
    using namespace nacelle::replay;
    if (!configuration || configuration->struct_size != sizeof(*configuration) || configuration->version != 1 ||
        (configuration->flags & ~7u) || (configuration->flags && (configuration->flags & 3u) != 3u) ||
        !configuration->snapshot_bytes || configuration->snapshot_bytes > maxSnapshotBytes ||
        !configuration->shader_bytes || configuration->shader_bytes > maxShaderBytes ||
        !configuration->command_bytes || configuration->command_bytes > maxCommandBytes) {
        log("Invalid replay configuration or missing snapshot/shader consent"); return E_INVALIDARG;
    }
    std::shared_ptr<Store> old;
    try {
        std::lock_guard lock(mutex);
        if (active || executors) { log("Cannot reconfigure active replay capture/execution"); return E_UNEXPECTED; }
        auto next = configuration->flags ? std::make_shared<Store>() : nullptr;
        if (next) next->configuration = *configuration;
        old = std::move(configured);
        configured = std::move(next);
        viewResources.clear();
        enabled.store(bool(configured), std::memory_order_relaxed);
    } catch (const std::bad_alloc&) { log("Insufficient memory configuring replay"); return E_OUTOFMEMORY; }
    return S_OK;
}

extern "C" HRESULT NacelleBeginReplayCapture(const NacelleReplayCaptureOptions* options, NacelleReplayCapture** output) {
    using namespace nacelle::replay;
    if (output) *output = nullptr;
    if (!output || !options || options->struct_size != sizeof(*options) || options->version != 1 ||
        options->contracts != 3 || !options->resources || !options->resource_count || options->resource_count > maxObjects) {
        log("Replay requires explicit race-free/host-quiescent declarations and bounded resources"); return E_INVALIDARG;
    }
    try {
        std::lock_guard lock(mutex);
        if (active || executors || !configured || !configured->error.empty()) {
            log(configured && !configured->error.empty() ? configured->error : "Replay is disabled or another capture is active");
            return E_UNEXPECTED;
        }
        auto capture = std::make_shared<Transaction>();
        capture->store = configured;
        capture->bundle.manifest.environment = currentEnvironment();
        if (capture->bundle.manifest.environment.fpPolicy != "flush-denormals") {
            log("Replay cannot fingerprint the active FP environment on this target"); return E_UNEXPECTED;
        }
        if (std::find(capture->bundle.manifest.environment.compilerOptions.begin(),
            capture->bundle.manifest.environment.compilerOptions.end(), "NACELLE_GEMM_BACKEND=unsupported") !=
            capture->bundle.manifest.environment.compilerOptions.end()) {
            log("Replay requires a recognized GEMM backend setting"); return E_UNEXPECTED;
        }
        const auto* gemm = std::getenv("NACELLE_DISABLE_GEMM_FASTPATH");
        if (!gemm || std::string_view(gemm) != "1") {
            log("Initial replay requires NACELLE_DISABLE_GEMM_FASTPATH=1 to exclude external BLAS execution");
            return E_UNEXPECTED;
        }
        capture->bundle.manifest.sourceConsent = (configured->configuration.flags & NACELLE_REPLAY_EMBEDDED_SOURCES) != 0;
        capture->bundle.manifest.snapshotBytes = configured->configuration.snapshot_bytes;
        capture->bundle.manifest.shaderBytes = configured->configuration.shader_bytes;
        capture->bundle.manifest.commandBytes = configured->configuration.command_bytes;
        capture->sourceResources = viewResources;
        uint64_t bytes = 0;
        for (uint32_t index = 0; index < options->resource_count; ++index) {
            const auto found = std::find_if(configured->resources.begin(), configured->resources.end(),
                [&](const auto& entry) {
                    return static_cast<ID3D12Resource*>(entry.first) == options->resources[index].resource;
                });
            auto* object = found == configured->resources.end() ? nullptr : found->first;
            const auto access = options->resources[index].access;
            if (found == configured->resources.end() || found->second.maps || !access || (access & ~3u) ||
                capture->resources.contains(object) ||
                (capture->device && capture->device != object->parentDevice())) {
                log("Replay resource is untracked, mapped, duplicated, cross-device or has invalid access");
                return E_INVALIDARG;
            }
            capture->device = object->parentDevice();
            const auto size = object->totalBytes();
            const auto needed = size * ((access & 2) ? 2 : 1);
            if (needed > configured->configuration.snapshot_bytes - bytes) {
                log("Replay initial/expected snapshots exceed declared budget"); return E_OUTOFMEMORY;
            }
            bytes += needed;
            const auto id = index + 1;
            capture->resources[object] = id;
            capture->bundle.metadata.resources.push_back({id, found->second.heapType,
                static_cast<uint32_t>(object->resourceDesc().Flags), access, size,
                capture->append({reinterpret_cast<const char*>(object->basePtr()), static_cast<size_t>(size)}, "initial"), 0});
        }
        auto handle = std::make_unique<NacelleReplayCapture>();
        handle->transaction = capture;
        active = std::move(capture);
        *output = handle.release();
        return S_OK;
    } catch (const std::bad_alloc&) { log("Insufficient memory snapshotting replay inputs"); return E_OUTOFMEMORY; }
}

namespace nacelle::replay {
static void stop(const std::shared_ptr<Transaction>& capture, bool expected) {
    std::unique_lock lock(mutex);
    if (capture->stopped) return;
    capture->accepting = false;
    drained.wait(lock, [&] { return capture->inFlight == 0; });
    auto queue = capture->queue;
    lock.unlock();
    std::unique_lock<std::mutex> queueLock;
    if (queue) queueLock = std::unique_lock(queue->replayMutex());
    lock.lock();
    if (expected && capture->error.empty()) {
        for (const auto& [object, id] : capture->pipelines)
            if (capture->bundle.metadata.pipelines[id - 1].nativeId != object->computePSO()->profileEntries.load()->artifactId)
                capture->fail("Replay code identity changed before capture end");
        if (capture->error.empty())
            for (const auto& [object, id] : capture->resources)
                if (capture->bundle.metadata.resources[id - 1].access & 2)
                    capture->bundle.metadata.resources[id - 1].expectedBlob =
                        capture->append({reinterpret_cast<const char*>(object->basePtr()), static_cast<size_t>(object->totalBytes())}, "expected");
    }
    if (active == capture) active.reset();
    capture->stopped = true;
}
}

extern "C" HRESULT NacelleEndReplayCapture(NacelleReplayCapture* handle, const char* path) {
    using namespace nacelle::replay;
    if (!handle || !path || !*path) { log("Invalid replay capture/output path"); return E_INVALIDARG; }
    try {
        stop(handle->transaction, true);
        if (!handle->transaction->error.empty()) return E_FAIL;
        std::string error;
        if (!validateSemantics(handle->transaction->bundle, error) ||
            !writeBundle(path, handle->transaction->bundle, error)) { log(error); return E_FAIL; }
        return S_OK;
    } catch (const std::bad_alloc&) { log("Insufficient memory saving replay"); return E_OUTOFMEMORY; }
      catch (const std::filesystem::filesystem_error& error) { log(error.what()); return E_FAIL; }
}
extern "C" void NacelleDestroyReplayCapture(NacelleReplayCapture* handle) {
    if (!handle) return;
    nacelle::replay::stop(handle->transaction, false);
    delete handle;
}
extern "C" HRESULT NacelleWaitForReplayCompilation() {
    using namespace nacelle::replay;
    try {
        std::string error;
        Execution execution(error);
        if (!execution.valid) { log(error); return E_UNEXPECTED; }
        nacelle::waitForAllAsyncJITTasks();
        return S_OK;
    } catch (const std::bad_alloc&) { log("Insufficient memory draining replay compilation"); return E_OUTOFMEMORY; }
      catch (const std::filesystem::filesystem_error& error) { log(error.what()); return E_FAIL; }
}
extern "C" HRESULT NacelleReplayBundle(const char* path, uint32_t consent, uint32_t repetitions) {
    using namespace nacelle::replay;
    if (!path || !*path || consent > 1 || !repetitions || repetitions > 100) {
        log("Invalid replay execution consent/repetition count"); return E_INVALIDARG;
    }
    try {
        Bundle bundle;
        std::string error;
        if (!readBundle(path, bundle, error)) { log(error); return E_FAIL; }
        if (!validateSemantics(bundle, error)) { log(error); return E_FAIL; }
        if (consent) {
            Execution execution(error);
            if (!execution.valid || !executeBundle(bundle, repetitions, error)) { log(error); return E_FAIL; }
        }
        return S_OK;
    } catch (const std::bad_alloc&) { log("Insufficient memory validating/replaying bundle"); return E_OUTOFMEMORY; }
      catch (const std::filesystem::filesystem_error& error) { log(error.what()); return E_FAIL; }
}
