#include <nacelle/instrumentation.h>
#include "instrumentation/capture.h"
#include "instrumentation/profile_format.h"
#include "compiler/shader_compiler.h"

#include <miniz.h>
#include <zstd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <iterator>
#include <new>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
// Faults affect one allocation on the API-calling thread, never concurrent workers.
thread_local size_t allocationFailureMinimum = 0;
thread_local size_t failedAllocationBytes = 0;
}

void* operator new(size_t bytes) {
    if (allocationFailureMinimum && bytes >= allocationFailureMinimum) {
        allocationFailureMinimum = 0;
        failedAllocationBytes = bytes;
        throw std::bad_alloc();
    }
    if (void* memory = std::malloc(bytes ? bytes : 1)) return memory;
    throw std::bad_alloc();
}

void* operator new[](size_t bytes) { return ::operator new(bytes); }
void operator delete(void* memory) noexcept { std::free(memory); }
void operator delete[](void* memory) noexcept { std::free(memory); }
void operator delete(void* memory, size_t) noexcept { std::free(memory); }
void operator delete[](void* memory, size_t) noexcept { std::free(memory); }

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct Capture {
    NacelleProfileCapture* value = nullptr;
    ~Capture() { NacelleDestroyProfileCapture(value); }
    void begin(uint64_t bytes = 1024 * 1024) {
        NacelleProfileOptions options;
        options.event_buffer_bytes = bytes;
        require(NacelleBeginProfileCapture(&options, &value) == S_OK, "Begin capture");
    }
};

nacelle::profile::Profile read(const std::filesystem::path& path) {
    nacelle::profile::Profile result;
    std::string error;
    require(nacelle::profile::readProfile(path.string(), result, error), error.c_str());
    return result;
}

std::string zipEntry(const std::filesystem::path& path, const char* name) {
    std::ifstream input(path, std::ios::binary);
    require(static_cast<bool>(input), "Open profile for ZIP inspection");
    const std::string bytes((std::istreambuf_iterator<char>(input)),
                            std::istreambuf_iterator<char>());
    require(!input.bad(), "Read profile for ZIP inspection");
    mz_zip_archive zip{};
    require(mz_zip_reader_init_mem(&zip, bytes.data(), bytes.size(), 0),
            "Open ZIP for entry inspection");
    const int index = mz_zip_reader_locate_file(&zip, name, nullptr, 0);
    mz_zip_archive_file_stat stat{};
    const bool found = index >= 0 &&
        mz_zip_reader_file_stat(&zip, static_cast<mz_uint>(index), &stat);
    std::string entry(found ? static_cast<size_t>(stat.m_uncomp_size) : 0, '\0');
    const bool extracted = found && mz_zip_reader_extract_to_mem(
        &zip, static_cast<mz_uint>(index), entry.data(), entry.size(), 0);
    mz_zip_reader_end(&zip);
    require(extracted, "Extract profile ZIP entry");
    return entry;
}

uint64_t manifestSize(const std::filesystem::path& path) {
    return zipEntry(path, "manifest.json").size();
}
}

int main(int argc, char** argv) {
    const auto directory = std::filesystem::temp_directory_path() /
        ("nacelle-profile-test-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        require(std::filesystem::create_directory(directory), "Create test directory");
        {
            nacelle::GraphicsEntryVersions<nacelle::CompiledPixelQuadFn> entries;
            const nacelle::CompiledPixelQuadFn initial = [](const nacelle::PixelBatchArgs* args) {
                *args->tileDepthSOA = 1;
            };
            const nacelle::CompiledPixelQuadFn promoted = [](const nacelle::PixelBatchArgs* args) {
                *args->tileDepthSOA = 2;
            };
            entries.store(initial, std::memory_order_release, 1, nacelle::profile::Cache::Miss);
            Capture capture;
            capture.begin();
            const auto pipeline = nacelle::profile::allocatePipelineIdentity();
            {
                nacelle::profile::Scope root("Graphics snapshot root");
                root.setAttributes({1, 2, 8, nacelle::profile::Stage::Graphics});
                root.beginGraphicsSelection(pipeline);
                std::promise<void> selected, resume;
                auto resumed = resume.get_future();
                std::jthread worker([&, context = root.context()] {
                    const auto* snapshot = entries.snapshot();
                    selected.set_value();
                    resumed.wait();
                    float result = 0;
                    nacelle::PixelBatchArgs args;
                    args.tileDepthSOA = &result;
                    snapshot->fn(&args);
                    require(result == 1 && snapshot->tier == 1,
                            "Delayed graphics execution retains old callable and version after promotion");
                    nacelle::profile::Scope old(context, "Old pixel entry");
                    old.setGraphicsVersion(pipeline, nacelle::profile::Stage::Pixel,
                                           snapshot->version, snapshot->tier, snapshot->cache);
                });
                selected.get_future().wait();
                entries.store(promoted, std::memory_order_release, 2, nacelle::profile::Cache::Miss);
                resume.set_value();
                worker.join();
                const auto* current = entries.snapshot();
                root.setGraphicsVersion(pipeline, nacelle::profile::Stage::Pixel,
                                        current->version, current->tier, current->cache);
                nacelle::profile::Scope vertex("Vertex entry");
                vertex.setGraphicsVersion(pipeline, nacelle::profile::Stage::Vertex,
                                          1, 1, nacelle::profile::Cache::Hit);
            }
            NacelleProfileSummary summary;
            const auto path = directory / "graphics-snapshots.nacelle-profile";
            require(NacelleEndProfileCapture(capture.value, path.string().c_str(), &summary) == S_OK,
                    "Save graphics snapshots across independent stages");
            const auto profile = read(path);
            require(profile.shaders.schemaVersion == 2 && profile.shaders.codeVersions.size() == 3,
                    "Graphics identities distinguish stages and retain both pixel versions");
            require(profile.shaders.codeSelections && profile.shaders.codeSelections->size() == 1 &&
                    profile.shaders.codeSelections->front().codeVersionIds.size() == 3,
                    "Delayed workers contribute their old entry to the draw's composite selection");
            for (const auto& event : profile.trace.traceEvents) {
                require(event.args.codeVersionId && event.args.pipelineId == 1,
                        "Graphics trace references capture-local version metadata");
            }
        }
        {
                NacelleProfileArtifactOptions options;
                options.flags = NACELLE_PROFILE_RETAIN_SHADER_TEXT | NACELLE_PROFILE_EMBED_SHADER_SOURCES;
                require(NacelleConfigureProfileArtifacts(&options) == S_OK, "Explicit artifact/source retention");
                const auto store = nacelle::profile::artifactStore();
                const std::string owner(64, 'a');
                nacelle::profile::TextArtifact source;
                source.kind = "hlsl";
                source.label = "source-1.hlsl";
                source.text = "float value = 1;\n";
                const auto sourceId = owner + "-hlsl-" + nacelle::profile::artifactDigest(source.text);
                nacelle::profile::TextArtifact dxil;
                dxil.kind = "dxil";
                dxil.label = "shader.dxil";
                dxil.text = "%value = fadd float 0.0, 1.0\n";
                dxil.mappings.push_back({1, 1, sourceId});
                nacelle::profile::retainArtifacts(store, owner, {source, dxil});
                const std::string nativeOwner(64, 'b');
                nacelle::profile::TextArtifact assembly;
                assembly.kind = "assembly";
                assembly.label = "text-1.asm";
                assembly.text = "0: ret\n";
                assembly.mappings.push_back({1, 0, "", 0, 1});
                nacelle::profile::retainArtifacts(store, nativeOwner, {assembly});
                Capture capture;
                capture.begin();
                const auto pipeline = nacelle::profile::allocatePipelineIdentity();
                {
                    nacelle::profile::Scope scope("Artifact entry selection");
                    scope.setAttributes({1, 2, 8, nacelle::profile::Stage::Compute, 1});
                    scope.setComputeVersion(pipeline, 1, 1,
                        nacelle::profile::Cache::Miss, owner);
                }
                {
                    nacelle::profile::Scope scope("Promoted artifact entry selection");
                    scope.setAttributes({1, 2, 8, nacelle::profile::Stage::Compute, 2});
                    scope.setComputeVersion(pipeline, 2, 2, nacelle::profile::Cache::Miss, owner, nativeOwner);
                }
                options.flags = 0;
                require(NacelleConfigureProfileArtifacts(&options) == S_OK, "Disable retention without invalidating capture snapshot");
                NacelleProfileSummary summary;
                const auto path = directory / "source-artifacts.nacelle-profile";
                require(NacelleEndProfileCapture(capture.value, path.string().c_str(), &summary) == S_OK,
                        "Save selected source/IR artifacts after store reset");
                auto profile = read(path);
                require(profile.manifest.schemaVersion == 2 && profile.artifacts &&
                        profile.artifacts->payloads.size() == 3 && profile.artifacts->selections.size() == 2 &&
                        profile.artifacts->selections[0].nativeAvailability == "tier1-unavailable" &&
                        profile.artifacts->selections[1].nativeAvailability == "available",
                        "Declared artifact payloads resolve exact selected entry");
                std::string error;
                profile.artifacts->payloads[1].mappings.front().sourceLine = 999;
                require(!nacelle::profile::validateArtifacts(*profile.artifacts, profile.shaders,
                    profile.manifest.captureOptions, error),
                        "Reject source mappings outside embedded source");
                profile.artifacts->payloads[1].mappings.front().sourceLine = 1;
                profile.manifest.captureOptions.embeddedSources = false;
                require(!nacelle::profile::validateArtifacts(*profile.artifacts, profile.shaders,
                    profile.manifest.captureOptions, error),
                        "Reject source payloads without explicit consent");
                options.flags = NACELLE_PROFILE_RETAIN_SHADER_TEXT;
                options.artifact_bytes = 1024;
                require(NacelleConfigureProfileArtifacts(&options) == S_OK, "Configure downward retention budget");
                const auto bounded = nacelle::profile::artifactStore();
                dxil.mappings.clear();
                const auto identityBytes = owner.size() * 2 + dxil.kind.size() * 2 + 2 + 16 + dxil.label.size();
                dxil.text.assign(1024 - identityBytes, 'a');
                nacelle::profile::retainArtifacts(bounded, owner, {dxil});
                require(!bounded->overflow && bounded->artifactBytes == 1024 && bounded->groups.size() == 1,
                        "Exact retention budget is admitted");
                nacelle::profile::retainArtifacts(bounded, nativeOwner, {dxil});
                require(bounded->overflow && bounded->groups.size() == 1, "Overflow admits no additional payload");
                Capture overflow;
                overflow.begin();
                const auto failed = directory / "artifact-overflow.nacelle-profile";
                require(FAILED(NacelleEndProfileCapture(overflow.value, failed.string().c_str(), &summary)) &&
                        !std::filesystem::exists(failed), "Retention overflow is explicit and creates no partial output");
                options.flags = 0;
                require(NacelleConfigureProfileArtifacts(&options) == S_OK, "Clear test retention");
        }
        {
            NacelleProfileSpatialOptions options;
            options.flags = NACELLE_PROFILE_SPATIAL_COMPUTE;
            options.record_bytes = 2 * sizeof(nacelle::profile::SpatialRecord);
            options.sample_stride = 2;
            require(NacelleConfigureProfileSpatial(&options) == S_OK, "Configure bounded spatial capture");
            Capture capture;
            capture.begin();
            require(NacelleConfigureProfileSpatial(&options) == E_UNEXPECTED,
                    "Spatial settings cannot change during active capture");
            {
                nacelle::profile::Scope scope("Spatial synthetic dispatch");
                scope.setAttributes({1, 2, 8, nacelle::profile::Stage::Compute, 1});
                scope.setComputeVersion(nacelle::profile::allocatePipelineIdentity(), 1, 1,
                    nacelle::profile::Cache::Miss);
                nacelle::profile::SpatialOperation operation;
                operation.stage = "compute";
                operation.dimensions = operation.extent = {3, 2, 1};
                operation.threadsPerGroup = {4, 2, 1};
                operation.granularity = "threadgroup";
                operation.orientation = "group-xyz";
                operation.availability = "available";
                const auto id = scope.addSpatialOperation(std::move(operation));
                for (uint32_t y = 0; y < 2; ++y)
                    for (uint32_t x = 0; x < 3; ++x)
                        if (auto* record = scope.spatial()->reserve(id, {x, y, 0})) {
                            record->wallNs = 0; // An observed zero is distinct from a missing cell.
                            record->invocations = 8;
                        }
                require(scope.spatial()->complete(), "Exact record-pool boundary stays complete");
                require(!scope.spatial()->reserve(id, {0, 0, 0}), "Next selected record explicitly drops");
            }
            NacelleProfileSummary summary;
            const auto path = directory / "spatial-sampled.nacelle-profile";
            require(NacelleEndProfileCapture(capture.value, path.string().c_str(), &summary) == S_FALSE &&
                    !summary.complete && summary.dropped_events == 0,
                    "Spatial overflow reports incomplete independently of event drops");
            auto profile = read(path);
            require(profile.spatial && !profile.manifest.complete && profile.spatial->recorded == 2 &&
                    profile.spatial->skipped == 4 && profile.spatial->dropped == 1 &&
                    profile.spatial->operations[0].cells.size() == 2 &&
                    profile.spatial->operations[0].cells[0].wallNs == 0 &&
                    profile.spatial->operations[0].cells[0].status == "partial",
                    "Sampled sparse cells preserve observed zeros and exact drops");
            std::string error;
            profile.spatial->operations[0].cells[0].xyz[0] = 3;
            require(!nacelle::profile::validateSpatial(*profile.spatial, profile.shaders,
                    profile.manifest.captureOptions, error), "Spatial reader rejects out-of-bounds coordinates");
            options.flags = 0;
            require(NacelleConfigureProfileSpatial(&options) == S_OK, "Disable spatial collection");
            options.flags = NACELLE_PROFILE_SPATIAL_COMPUTE;
            options.record_bytes = 16ull * 1024 * 1024;
            require(NacelleConfigureProfileSpatial(&options) == S_OK, "Configure spatial allocation failure case");
            allocationFailureMinimum = options.record_bytes / sizeof(nacelle::profile::SpatialRecord) *
                                       sizeof(nacelle::profile::SpatialRecord);
            NacelleProfileOptions eventOptions;
            eventOptions.event_buffer_bytes = 1024 * 1024;
            NacelleProfileCapture* failed = nullptr;
            require(NacelleBeginProfileCapture(&eventOptions, &failed) == E_OUTOFMEMORY && !failed,
                    "Spatial-pool allocation failure leaves no active capture");
            options.flags = 0;
            require(NacelleConfigureProfileSpatial(&options) == S_OK, "Reset spatial allocation settings");
            options.record_bytes = 1;
            require(NacelleConfigureProfileSpatial(&options) == E_INVALIDARG, "Reject invalid spatial record budget");
        }
        {
            NacelleProfileOptions options;
            options.event_buffer_bytes = 1024 * 1024;
            const auto eventBytes = options.event_buffer_bytes / sizeof(nacelle::profile::Event) *
                                    sizeof(nacelle::profile::Event);
            NacelleProfileCapture* failed = nullptr;
            allocationFailureMinimum = eventBytes;
            const auto result = NacelleBeginProfileCapture(&options, &failed);
            require(result == E_OUTOFMEMORY && !failed && failedAllocationBytes == eventBytes,
                    "Event-pool allocation failure is explicit and leaves no active capture");

            Capture capture;
            capture.begin();
            {
                NACELLE_PROFILE_SCOPE("Serialization allocation retry");
            }
            const auto failedPath = directory / "allocation-failed.nacelle-profile";
            const auto retryPath = directory / "allocation-retried.nacelle-profile";
            const auto failedName = failedPath.string();
            const auto retryName = retryPath.string();
            NacelleProfileSummary summary;
            allocationFailureMinimum = 1;
            require(NacelleEndProfileCapture(capture.value, failedName.c_str(), &summary) == E_OUTOFMEMORY &&
                    !std::filesystem::exists(failedPath),
                    "Serialization allocation failure produces no success-shaped or partial bundle");
            require(NacelleEndProfileCapture(capture.value, retryName.c_str(), &summary) == S_OK &&
                    summary.recorded_events == 1 && summary.dropped_events == 0 && summary.complete &&
                    read(retryPath).trace.traceEvents.size() == 1,
                    "Stopped capture retries after allocation failure without losing events");
        }
        {
            nacelle::ComputeEntryVersions entries;
            const nacelle::CompiledComputeFn initial = [](const nacelle::DriverRootContext*,
                uint8_t* result, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) {
                *result = 1;
            };
            const nacelle::CompiledComputeFn promoted = [](const nacelle::DriverRootContext*,
                uint8_t* result, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) {
                *result = 2;
            };
            entries.initialize(initial, 1, nacelle::profile::Cache::Miss);
            for (int session = 0; session < 2; ++session) {
                Capture capture;
                capture.begin();
                {
                    nacelle::profile::Scope root("Compute version test");
                    std::promise<void> selected, resume;
                    auto resumed = resume.get_future();
                    std::jthread worker([&, context = root.context()] {
                        const auto* snapshot = entries.load();
                        selected.set_value();
                        resumed.wait();
                        uint8_t result = 0;
                        snapshot->fn(nullptr, &result, 0, 0, 0, 1, 1, 1);
                        require(result == snapshot->tier, "Selected callable matches immutable tier");
                        nacelle::profile::Scope dispatch(context, "Versioned dispatch");
                        dispatch.setAttributes({1, 2, 8, nacelle::profile::Stage::Compute});
                        dispatch.setComputeVersion(entries.pipelineIdentity, snapshot->version,
                                                   snapshot->tier, snapshot->cache);
                        nacelle::profile::Scope child(dispatch.context(), "Versioned worker");
                    });
                    selected.get_future().wait();
                    if (session == 0) entries.promote(promoted);
                    resume.set_value();
                    worker.join();
                    const auto* snapshot = entries.load();
                    root.setAttributes({1, 2, 8, nacelle::profile::Stage::Compute});
                    root.setComputeVersion(entries.pipelineIdentity, snapshot->version,
                                           snapshot->tier, snapshot->cache);
                }
                NacelleProfileSummary summary;
                const auto path = directory /
                    ("compute-versions-" + std::to_string(session) + ".nacelle-profile");
                require(NacelleEndProfileCapture(capture.value, path.string().c_str(), &summary) == S_OK,
                        "Save selected compute entry versions");
                const auto profile = read(path);
                require(profile.shaders.codeVersions.size() == (session == 0 ? 2u : 1u),
                        "Retain old versions across promotion and reset capture-local IDs");
                for (const auto& event : profile.trace.traceEvents) {
                    require(event.cat == "runtime" && event.args.pipelineId == 1 &&
                            event.args.codeVersionId.has_value(),
                            "Dispatch and worker carry capture-local pipeline and code identity");
                    const auto& code = profile.shaders.codeVersions[*event.args.codeVersionId - 1];
                    require(code.version == code.tier && code.cache == "miss",
                            "Trace resolves to the actual selected entry, not the latest promotion");
                }
            }
        }
        {
            Capture capture;
            capture.begin();
            NacelleProfileCapture* duplicate = nullptr;
            NacelleProfileOptions options;
            require(FAILED(NacelleBeginProfileCapture(&options, &duplicate)), "Reject concurrent capture");
            require(!duplicate, "Clear failed begin output");
            std::thread worker;
            {
                nacelle::profile::Scope root("Compile test shader");
                root.setAttributes({1, 2, 8, nacelle::profile::Stage::Compute, 1,
                                    nacelle::profile::Cache::Miss});
                worker = std::thread([context = root.context()] {
                    nacelle::profile::Scope child(context, "Worker test shader");
                    nacelle::profile::Scope nested("Nested test stage");
                });
            }
            NacelleProfileSummary summary;
            const auto path = directory / "complete.nacelle-profile";
            require(NacelleEndProfileCapture(capture.value, path.string().c_str(), &summary) == S_OK,
                    "End waits for captured async work");
            worker.join();
            auto profile = read(path);
            require(summary.recorded_events == 3 && summary.complete, "All nested events recorded");
            const auto& events = profile.trace.traceEvents;
            const auto root = std::find_if(events.begin(), events.end(), [](const auto& event) {
                return event.name == "Compile test shader";
            });
            require(root != events.end(), "Root event exists");
            require(std::count_if(events.begin(), events.end(), [&](const auto& event) {
                return event.args.parent == root->args.id;
            }) == 1, "Async child retains parent");
            require(events.front().args.pso == "00000000000000020000000000000001",
                    "Persistent PSO identity");
            require(FAILED(NacelleEndProfileCapture(capture.value, path.string().c_str(), &summary)),
                    "Refuse to overwrite existing bundle");
            require(read(path).manifest.complete, "Existing bundle preserved");
            if (argc == 2) {
                const std::string destination = argv[1];
                std::filesystem::copy_file(path,
                    std::filesystem::path(std::u8string(destination.begin(), destination.end())));
            }
        }
        {
            Capture capture;
            capture.begin();
            nacelle::profile::Context escaped;
            {
                nacelle::profile::Scope root("Ownership root");
                root.setAttributes({1, 2, 8, nacelle::profile::Stage::Compute, 1,
                                    nacelle::profile::Cache::Miss});
                const auto owner = root.context();
                const auto references = owner.work->session.use_count();
                nacelle::profile::Scope middle("Ownership middle");
                nacelle::profile::Scope leaf("Ownership leaf");
                require(owner.work->session.use_count() == references,
                        "Nested scopes borrow the root without session reference-count traffic");
                escaped = leaf.context();
            }
            {
                const auto work = escaped.work;
                const auto references = work->session.use_count();
                nacelle::profile::Scope async(escaped, "Ownership async");
                escaped = {};
                require(work->session.use_count() == references,
                        "Async scope retains Work without a redundant session reference");
                nacelle::profile::Scope nested("Ownership async nested");
                const auto descendant = nested.context();
                require(descendant.work->session == work->session,
                        "Nested async context retains the original session after its root exits");
            }
            NacelleProfileSummary summary;
            const auto path = directory / "scope-ownership.nacelle-profile";
            require(NacelleEndProfileCapture(capture.value, path.string().c_str(), &summary) == S_OK,
                    "Save escaped nested context after owning roots exit");
            const auto profile = read(path);
            require(summary.recorded_events == 5 && summary.dropped_events == 0,
                    "Borrowing preserves every nested and escaped event");
            const auto& events = profile.trace.traceEvents;
            for (const auto& event : events) {
                require(event.args.pso == "00000000000000020000000000000001" &&
                        event.args.stage == "compute", "Borrowed scopes preserve attributes");
                if (event.args.parent) {
                    require(std::any_of(events.begin(), events.end(), [&](const auto& parent) {
                        return parent.args.id == event.args.parent;
                    }), "Escaped nested contexts preserve parent correlation");
                }
            }
        }
        {
            Capture capture;
            capture.begin(sizeof(nacelle::profile::Event));
            for (int i = 0; i < 10; ++i) {
                NACELLE_PROFILE_SCOPE("Overflow test");
            }
            NacelleProfileSummary summary;
            const auto path = directory / "overflow.nacelle-profile";
            require(NacelleEndProfileCapture(capture.value, path.string().c_str(), &summary) == S_FALSE,
                    "Overflow explicitly incomplete");
            auto profile = read(path);
            require(!profile.manifest.complete && summary.recorded_events == 1 &&
                    summary.dropped_events == 9, "Bounded recorder reports exact drops");
        }
        {
            for (int session = 0; session < 2; ++session) {
                Capture capture;
                capture.begin();
                std::vector<std::jthread> workers;
                {
                    nacelle::profile::Scope root("Concurrent root");
                    const auto context = root.context();
                    for (int worker = 0; worker < 4; ++worker) {
                        workers.emplace_back([context] {
                            nacelle::profile::Scope child(context, "Concurrent worker");
                            for (int i = 0; i < 97; ++i) {
                                nacelle::profile::Scope outer("Concurrent outer");
                                nacelle::profile::Scope inner("Concurrent inner");
                            }
                        });
                    }
                }
                NacelleProfileSummary summary;
                const auto path = directory /
                    ("concurrent-" + std::to_string(session) + ".nacelle-profile");
                require(NacelleEndProfileCapture(capture.value, path.string().c_str(), &summary) == S_OK,
                        "Drain concurrent events across repeated sessions");
                for (auto& worker : workers) worker.join();
                const auto profile = read(path);
                constexpr uint64_t expected = 1 + 4 * (1 + 2 * 97);
                require(summary.recorded_events == expected &&
                        profile.manifest.recordedEvents == expected &&
                        profile.trace.traceEvents.size() == expected,
                        "Count every completed concurrent event, including partial blocks");
                std::set<uint64_t> ids;
                for (const auto& event : profile.trace.traceEvents) {
                    require(event.args.id != 0 && ids.insert(event.args.id).second,
                            "Concurrent event IDs are nonzero and unique");
                }
                size_t roots = 0;
                for (const auto& event : profile.trace.traceEvents) {
                    if (event.args.parent == 0) ++roots;
                    else require(ids.contains(event.args.parent), "Concurrent parent ID exists");
                }
                require(roots == 1, "Concurrent scopes retain one root");
            }
        }
        {
            Capture capture;
            capture.begin(65 * sizeof(nacelle::profile::Event));
            std::vector<std::jthread> workers;
            for (int worker = 0; worker < 4; ++worker) {
                workers.emplace_back([] {
                    for (int i = 0; i < 130; ++i) {
                        NACELLE_PROFILE_SCOPE("Concurrent overflow");
                    }
                });
            }
            for (auto& worker : workers) worker.join();
            NacelleProfileSummary summary;
            const auto path = directory / "concurrent-overflow.nacelle-profile";
            require(NacelleEndProfileCapture(capture.value, path.string().c_str(), &summary) == S_FALSE,
                    "Concurrent overflow explicitly incomplete");
            const auto profile = read(path);
            require(summary.recorded_events == 65 && summary.dropped_events == 455 &&
                    profile.manifest.recordedEvents == 65 &&
                    profile.manifest.droppedEvents == 455 &&
                    profile.trace.traceEvents.size() == 65,
                    "Exact concurrent accounting with a partially available final block");
        }
        {
            Capture capture;
            capture.begin();
            NacelleProfileSummary summary;
            const auto path = directory / "empty.nacelle-profile";
            require(NacelleEndProfileCapture(capture.value, path.string().c_str(), &summary) == S_OK,
                    "End empty capture");
            const auto profile = read(path);
            require(summary.recorded_events == 0 && summary.dropped_events == 0 &&
                    profile.trace.traceEvents.empty(), "Empty capture has no stale events");
        }
        {
            nacelle::profile::Scope inactive("Inactive metadata");
            require(!inactive.needsCompilerConfiguration(), "No compiler metadata work while idle");
        }
        for (int session = 0; session < 2; ++session) {
            Capture capture;
            capture.begin();
            {
                nacelle::profile::Scope root("Concurrent compiler metadata");
                require(root.needsCompilerConfiguration(), "Compiler metadata is capture-local");
                std::vector<std::jthread> workers;
                for (int worker = 0; worker < 4; ++worker) {
                    workers.emplace_back([context = root.context(), worker] {
                        nacelle::profile::Scope child(context, "Compiler metadata worker");
                        if (child.needsCompilerConfiguration()) {
                            const auto hash = "compiler-" + std::to_string(worker);
                            nacelle::profile::setCompilerConfiguration(
                                hash, hash + "-triple", hash + "-cpu", {hash + "-feature"});
                        }
                    });
                }
                for (auto& worker : workers) worker.join();
                require(!root.needsCompilerConfiguration(), "Reuse recorded compiler configuration");
            }
            NacelleProfileSummary summary;
            const auto path = directory /
                ("compiler-metadata-" + std::to_string(session) + ".nacelle-profile");
            require(NacelleEndProfileCapture(capture.value, path.string().c_str(), &summary) == S_OK,
                    "Save concurrent compiler metadata");
            const auto profile = read(path);
            const auto& build = profile.manifest.build;
            require(build.compilerHash.starts_with("compiler-") &&
                    build.targetTriple == build.compilerHash + "-triple" &&
                    build.cpu == build.compilerHash + "-cpu" &&
                    build.cpuFeatures == std::vector<std::string>{build.compilerHash + "-feature"},
                    "Concurrent callers publish one consistent compiler configuration");
        }
        {
            Capture cancelled;
            cancelled.begin();
            NACELLE_PROFILE_SCOPE("Cancelled event");
        }
        {
            Capture capture;
            capture.begin();
            {
                NACELLE_PROFILE_SCOPE("Write error test");
            }
            NacelleProfileSummary summary;
            const auto invalid = directory / "missing" / "capture.nacelle-profile";
            require(FAILED(NacelleEndProfileCapture(capture.value, invalid.string().c_str(), &summary)),
                    "Write failures reported");
            const auto retry = directory / "retry.nacelle-profile";
            require(NacelleEndProfileCapture(capture.value, retry.string().c_str(), &summary) == S_OK,
                    "Stopped capture can retry output");
            require(read(retry).manifest.recordedEvents == 1, "Repeated sessions have no stale records");
        }
        {
            nacelle::profile::Session session(8 * sizeof(nacelle::profile::Event));
            auto& late = session.events[0];
            late.id = 3;
            late.startNs = 2501;
            late.durationNs = 999;
            late.thread = 2;
            late.attributes = {0, 0, 4, nacelle::profile::Stage::Pixel};
            std::strcpy(late.name, "Quote \" slash \\ newline \n");
            auto& early = session.events[4];
            early.id = 1;
            early.startNs = 1000;
            early.durationNs = 1501;
            early.thread = 1;
            early.attributes = {1, 2, 8, nacelle::profile::Stage::Compute, 1,
                                nacelle::profile::Cache::Miss};
            std::strcpy(early.name, "First event");
            auto& middle = session.events[7];
            middle = early;
            middle.id = 2;
            middle.parent = 1;
            middle.startNs = 2000;
            middle.durationNs = 0;
            middle.attributes.tier = 2;
            middle.attributes.cache = nacelle::profile::Cache::Hit;
            middle.attributes.pipelineId = middle.attributes.codeVersionId = 1;
            std::strcpy(middle.name, "Versioned event");
            session.shaders.codeVersions.push_back({
                1, 1, "00000000000000020000000000000001", "compute", 8, 1, 2, "hit"});
            session.nextBlock = session.capacity;
            session.wait();
            const auto path = directory / "lazy-trace.nacelle-profile";
            std::string error;
            require(nacelle::profile::writeProfile(path.string(), session, error), error.c_str());
            const auto profile = read(path);
            const auto& events = profile.trace.traceEvents;
            require(events.size() == 3 && events[0].args.id == 1 &&
                    events[1].args.id == 2 && events[2].args.id == 3,
                    "Lazy serialization sorts sparse completed event slots");
            require(events[2].name == late.name && events[2].ts == 2.501 &&
                    events[2].dur == 0.999 && events[1].dur == 0 &&
                    events[1].args.codeVersionId == 1,
                    "Lazy serialization preserves escaping, fractional units and version references");
            const auto compressed = zipEntry(path, "trace.json.zst");
            const auto size = ZSTD_getFrameContentSize(compressed.data(), compressed.size());
            require(size != ZSTD_CONTENTSIZE_ERROR && size != ZSTD_CONTENTSIZE_UNKNOWN,
                    "Trace frame declares its decoded size");
            std::string decoded(static_cast<size_t>(size), '\0');
            require(ZSTD_decompress(decoded.data(), decoded.size(),
                                    compressed.data(), compressed.size()) == decoded.size(),
                    "Decompress serialized trace");
            const auto exported = directory / "lazy-trace.json";
            require(nacelle::profile::exportTrace(exported.string(), profile.trace, error), error.c_str());
            std::ifstream input(exported, std::ios::binary);
            require(decoded == std::string(std::istreambuf_iterator<char>(input), {}),
                    "Lazy trace bytes match ordinary typed Trace serialization exactly");
        }
        {
            constexpr uint64_t manifestLimit = 1024 * 1024;
            nacelle::profile::Session session(sizeof(nacelle::profile::Event));
            session.wait();
            std::string error;
            const auto baseline = directory / "metadata-baseline.nacelle-profile";
            require(nacelle::profile::writeProfile(baseline.string(), session, error),
                    error.c_str());
            const auto overhead = manifestSize(baseline);
            require(overhead < manifestLimit, "Baseline manifest fits documented limit");
            session.compilerHash.assign(static_cast<size_t>(manifestLimit - overhead), 'a');
            const auto boundary = directory / "metadata-boundary.nacelle-profile";
            require(nacelle::profile::writeProfile(boundary.string(), session, error),
                    error.c_str());
            require(manifestSize(boundary) == manifestLimit &&
                    read(boundary).manifest.build.compilerHash == session.compilerHash,
                    "Exactly 1 MiB manifest round-trips");

            session.compilerHash.push_back('a');
            const auto oversized = directory / "metadata-oversized.nacelle-profile";
            error.clear();
            require(!nacelle::profile::writeProfile(oversized.string(), session, error) &&
                    error.find("manifest") != std::string::npos &&
                    !std::filesystem::exists(oversized),
                    "Reject a manifest one byte over the limit without creating a bundle");

            session.compilerHash.assign(static_cast<size_t>(manifestLimit / 2), '"');
            const auto escaped = directory / "metadata-escaped.nacelle-profile";
            error.clear();
            require(!nacelle::profile::writeProfile(escaped.string(), session, error) &&
                    error.find("manifest") != std::string::npos &&
                    !std::filesystem::exists(escaped),
                    "Manifest limit includes JSON escaping expansion");
        }
        {
            Capture capture;
            capture.begin();
            {
                NACELLE_PROFILE_SCOPE("Oversized metadata");
                nacelle::profile::setCompilerConfiguration(std::string(1024 * 1024, 'a'),
                                                           "", "", {});
            }
            NacelleProfileSummary summary;
            const auto path = directory / "metadata-api-error.nacelle-profile";
            require(FAILED(NacelleEndProfileCapture(capture.value, path.string().c_str(), &summary)) &&
                    !std::filesystem::exists(path),
                    "Public API reports oversized metadata instead of saving an unreadable bundle");
        }
        {
            const auto path = directory / "corrupt.nacelle-profile";
            std::ofstream(path, std::ios::binary) << "not a ZIP archive";
            nacelle::profile::Profile profile;
            std::string error;
            require(!nacelle::profile::readProfile(path.string(), profile, error) && !error.empty(),
                    "Reject corrupt archive explicitly");
        }
        {
            Capture capture;
            capture.begin();
            { NACELLE_PROFILE_SCOPE("ZIP header validation"); }
            NacelleProfileSummary summary;
            const auto path = directory / "zip-headers.nacelle-profile";
            require(NacelleEndProfileCapture(capture.value, path.string().c_str(), &summary) == S_OK,
                    "Save ZIP header fixture");
            std::ifstream input(path, std::ios::binary);
            const std::string original(std::istreambuf_iterator<char>(input), {});
            nacelle::profile::Profile profile;
            std::string error;
            require(nacelle::profile::readProfileBytes(original, profile, error), error.c_str());
            for (const auto offset : {6, 8, 14, 30}) {
                auto corrupt = original;
                corrupt[offset] ^= 1;
                error.clear();
                require(!nacelle::profile::readProfileBytes(corrupt, profile, error) && !error.empty(),
                        "Reject local ZIP flag, method, CRC and name conflicts before decoding");
            }
        }
        {
            nacelle::profile::Session session(1024);
            session.shaders.codeVersions.push_back({
                2, 1, std::string(32, 'a'), "compute", 8, 1, 1, "miss"});
            const auto path = directory / "invalid-code-identity.nacelle-profile";
            std::string error;
            require(nacelle::profile::writeProfile(path.string(), session, error), error.c_str());
            nacelle::profile::Profile profile;
            require(!nacelle::profile::readProfile(path.string(), profile, error) &&
                    error.find("code-version") != std::string::npos, "Reject invalid code-version IDs");
        }
        {
            nacelle::profile::Session session(1024);
            session.shaders.schemaVersion = 2;
            session.shaders.codeVersions.push_back({
                1, 1, std::string(32, 'a'), "hull", 8, 1, 1, "miss", "hull-control-point"});
            session.shaders.codeVersions.push_back({
                2, 1, std::string(32, 'a'), "hull", 8, 1, 1, "miss", "hull-patch-constant"});
            session.shaders.codeSelections = std::vector<nacelle::profile::CodeSelection>{{1, 1, {1, 2}}};
            std::string error;
            const auto path = directory / "hull-selection.nacelle-profile";
            require(nacelle::profile::writeProfile(path.string(), session, error), error.c_str());
            require(read(path).shaders.codeVersions.size() == 2, "Distinct hull roles resolve the same version");
            session.shaders.codeSelections->front().codeVersionIds = {1, 1};
            const auto invalid = directory / "duplicate-selection.nacelle-profile";
            require(nacelle::profile::writeProfile(invalid.string(), session, error), error.c_str());
            nacelle::profile::Profile profile;
            require(!nacelle::profile::readProfile(invalid.string(), profile, error) &&
                    error.find("selection") != std::string::npos, "Reject duplicate composite entry references");
        }
        {
            nacelle::profile::Session session(1024);
            session.shaders.codeVersions.push_back({
                1, 1, "00000000000000020000000000000001", "compute", 8, 1, 1, "miss"});
            nacelle::profile::Event event{};
            event.id = 1;
            event.attributes = {1, 2, 8, nacelle::profile::Stage::Compute, 1,
                                nacelle::profile::Cache::Miss, 1, 2};
            session.record(event);
            session.wait();
            nacelle::profile::Profile profile;
            std::string error;
            const auto dangling = directory / "dangling-code-reference.nacelle-profile";
            require(nacelle::profile::writeProfile(dangling.string(), session, error), error.c_str());
            require(!nacelle::profile::readProfile(dangling.string(), profile, error) &&
                    error.find("reference") != std::string::npos, "Reject dangling trace code references");
            session.events[0].attributes.codeVersionId = 1;
            session.events[0].attributes.tier = 2;
            const auto mismatch = directory / "mismatched-code-reference.nacelle-profile";
            require(nacelle::profile::writeProfile(mismatch.string(), session, error), error.c_str());
            require(!nacelle::profile::readProfile(mismatch.string(), profile, error) &&
                    error.find("match") != std::string::npos, "Reject trace/version tier mismatches");
        }
        {
            Capture capture;
            capture.begin();
            const uint64_t pipeline = nacelle::profile::allocatePipelineIdentity();
            for (uint32_t i = 0; i < nacelle::profile::maxCodeVersions / 2; ++i) {
                nacelle::profile::Scope scope("Selection limit");
                scope.beginGraphicsSelection(pipeline);
            }
            for (uint32_t i = 0; i <= nacelle::profile::maxCodeVersions / 2; ++i) {
                nacelle::profile::Scope scope("Metadata limit");
                scope.setAttributes({1, 2, 8, nacelle::profile::Stage::Compute});
                scope.setComputeVersion(pipeline + i, 1, 1, nacelle::profile::Cache::Miss);
            }
            NacelleProfileSummary summary;
            const auto path = directory / "code-metadata-overflow.nacelle-profile";
            require(FAILED(NacelleEndProfileCapture(capture.value, path.string().c_str(), &summary)) &&
                    !std::filesystem::exists(path), "Code versions and selections share one admission limit");
        }
        NacelleProfileOptions invalidOptions;
        invalidOptions.flags = 1;
        NacelleProfileCapture* invalid = nullptr;
        require(FAILED(NacelleBeginProfileCapture(&invalidOptions, &invalid)), "Reject unsupported options");
        std::filesystem::remove_all(directory);
        std::cout << "Capture lifecycle, async correlation, allocation retry, overflow, write errors, and corrupt input: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "instrumentation_test: " << error.what() << "\nArtifacts: " << directory << '\n';
        return 1;
    }
}
