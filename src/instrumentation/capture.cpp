#include "capture.h"
#include "profile_format.h"
#include <nacelle/instrumentation.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <new>

namespace {
std::mutex activeMutex;
std::shared_ptr<nacelle::profile::Session> activeSession;
std::atomic<bool> recording{false};
std::atomic<uint64_t> generation{1};
thread_local nacelle::profile::Scope* currentScope = nullptr;
NacelleProfileSpatialOptions spatialOptions;

HRESULT fail(HRESULT code, const char* message) {
    std::fprintf(stderr, "[NacelleProfile] %s\n", message);
    return code;
}
}

struct NacelleProfileCapture {
    std::shared_ptr<nacelle::profile::Session> session;
    bool stopped = false;
};

namespace nacelle::profile {

uint64_t allocatePipelineIdentity() {
    static std::atomic<uint64_t> next{1};
    return next.fetch_add(1, std::memory_order_relaxed);
}

Session::Session(uint64_t bytes)
    : generation(::generation.fetch_add(1)), bufferBytes(bytes),
      capacity(static_cast<size_t>(bytes / sizeof(Event))),
      events(std::make_unique<Event[]>(capacity)) {}

void Session::releaseWork() {
    if (outstanding.fetch_sub(1) == 1) {
        std::lock_guard lock(waitMutex);
        drained.notify_all();
    }
}

void Session::wait() {
    std::unique_lock lock(waitMutex);
    drained.wait(lock, [&] { return outstanding.load() == 0; });
    const auto end = events.get() + std::min(capacity, nextBlock.load());
    eventCount = static_cast<uint64_t>(std::count_if(events.get(), end,
        [](const Event& event) { return event.id != 0; }));
}

uint64_t Session::allocateEventId() {
    struct ThreadIds {
        uint64_t generation = 0;
        uint64_t cursor = 0, end = 0;
    };
    thread_local ThreadIds ids;
    if (ids.generation != generation || ids.cursor == ids.end) {
        constexpr uint64_t blockSize = 64;
        ids.generation = generation;
        ids.cursor = nextId.fetch_add(blockSize, std::memory_order_relaxed);
        ids.end = ids.cursor + blockSize;
    }
    return ids.cursor++;
}

void Session::record(Event event) {
    struct ThreadBuffer {
        uint64_t generation = 0;
        size_t cursor = 0, end = 0;
        uint32_t thread = 0;
    };
    thread_local ThreadBuffer buffer;
    if (buffer.generation != generation) {
        buffer = {};
        buffer.generation = generation;
        buffer.thread = nextThread.fetch_add(1, std::memory_order_relaxed);
    }
    if (buffer.cursor == buffer.end) {
        constexpr size_t blockSize = 64;
        buffer.cursor = nextBlock.fetch_add(blockSize, std::memory_order_relaxed);
        buffer.end = std::min(capacity, buffer.cursor + blockSize);
    }
    if (buffer.cursor >= capacity) {
        dropped.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    event.thread = buffer.thread;
    events[buffer.cursor++] = event;
}

Work::Work(std::shared_ptr<Session> value) : session(std::move(value)) {
    session->outstanding.fetch_add(1);
}
Work::~Work() { session->releaseWork(); }

Scope::Scope(const char* name) {
    if (currentScope) {
        session_ = currentScope->session_;
    } else if (recording.load(std::memory_order_acquire)) {
        std::lock_guard lock(activeMutex);
        rootSession_ = activeSession;
        session_ = rootSession_.get();
        if (session_) {
            session_->outstanding.fetch_add(1);
            ownsWork_ = true;
        }
    }
    if (session_) {
        begin(name);
        if (previous_) {
            event_.parent = previous_->event_.id;
            event_.attributes = previous_->event_.attributes;
        }
    }
}

Scope::Scope(const Context& context, const char* name, uint8_t tier) {
    if (!context.work) return;
    work_ = context.work;
    session_ = work_->session.get();
    begin(name);
    event_.parent = context.parent;
    event_.attributes = context.attributes;
    if (tier) event_.attributes.tier = tier;
}

void Scope::begin(const char* name) {
    event_ = {};
    previous_ = currentScope;
    currentScope = this;
    event_.id = session_->allocateEventId();
    event_.startNs = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - session_->origin).count());
    std::strncpy(event_.name, name, sizeof(event_.name) - 1);
}

Scope::~Scope() {
    if (!session_) return;
    const auto now = std::chrono::steady_clock::now() - session_->origin;
    event_.durationNs = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(now).count()) - event_.startNs;
    session_->record(event_);
    currentScope = previous_;
    if (ownsWork_) session_->releaseWork();
}

Context Scope::context() const {
    if (!session_) return {};
    // Nested scopes borrow their live stack parent; only roots and async scopes own the session.
    const Scope* owner = this;
    while (!owner->rootSession_ && !owner->work_) owner = owner->previous_;
    const auto& session = owner->rootSession_ ? owner->rootSession_ : owner->work_->session;
    return {std::make_shared<Work>(session), event_.id, event_.attributes};
}

uint64_t Scope::addSpatialOperation(SpatialOperation operation) const {
    if (!spatial() || !event_.attributes.codeVersionId) return 0;
    operation.eventId = event_.id;
    operation.pipelineId = event_.attributes.pipelineId;
    operation.codeVersionId = event_.attributes.codeVersionId;
    operation.pso = formatProfileHash(event_.attributes.hashLo, event_.attributes.hashHi);
    operation.simdWidth = event_.attributes.simdWidth;
    return spatial()->addOperation(std::move(operation));
}

void Scope::setComputeVersion(uint64_t pipeline, uint32_t version, uint8_t tier, Cache cache,
                              const std::string& shaderId, const std::string& artifactId) {
    if (session_) event_.attributes.codeSelectionId = 0;
    setGraphicsVersion(pipeline, Stage::Compute, version, tier, cache, shaderId, EntryRole::Default, artifactId);
}

void Scope::setGraphicsVersion(uint64_t pipeline, Stage stage, uint32_t version,
                               uint8_t tier, Cache cache, const std::string& shaderId, EntryRole role,
                               const std::string& artifactId) {
    if (!session_)
        return;
    auto &session = *session_;
    std::lock_guard lock(session.metadataMutex);
    if (session.shaderMetadataOverflow)
        return;
    if ((!shaderId.empty() &&
         (shaderId.size() != 64 || shaderId.find_first_not_of("0123456789abcdef") != std::string::npos)) ||
        (!artifactId.empty() && (artifactId.size() != 64 ||
                                 artifactId.find_first_not_of("0123456789abcdef") != std::string::npos))) {
        session.shaderMetadataOverflow = true;
        fail(E_INVALIDARG, "Profile shader/native identity is not a 64-character SHA-256 identity");
        return;
    }
    try {
        auto &attributes = event_.attributes;
        const auto key = std::make_tuple(pipeline, stage, role, version);
        auto found = session.codeVersionIds.find(key);
        if (found == session.codeVersionIds.end()) {
            if (session.shaders.codeVersions.size() + session.codeSelectionParents.size() ==
                maxCodeVersions) {
                session.shaderMetadataOverflow = true;
                fail(E_FAIL, "Profile code-version/selection metadata limit exceeded");
                return;
            }
            const auto pipelineIt =
                session.pipelineIds.try_emplace(pipeline, session.pipelineIds.size() + 1).first;
            const uint64_t id = session.shaders.codeVersions.size() + 1;
            constexpr const char *caches[] = {"unknown", "hit", "miss"};
            constexpr const char *stages[] = {"unknown", "compute", "graphics",     "vertex",
                                              "pixel",   "mesh",    "raytracing",   "geometry",
                                              "hull",    "domain",  "amplification"};
            const char *stageName = stages[static_cast<unsigned>(stage)];
            CodeVersion code{id,
                             pipelineIt->second,
                             formatProfileHash(attributes.hashLo, attributes.hashHi),
                             stageName,
                             attributes.simdWidth,
                             version,
                             tier,
                             caches[static_cast<unsigned>(cache)]};
            if (stage != Stage::Compute) {
                code.entryRole = role == EntryRole::HullControlPoint    ? "hull-control-point"
                                 : role == EntryRole::HullPatchConstant ? "hull-patch-constant"
                                                                        : stageName;
                session.shaders.schemaVersion = 2;
            }
            if (!shaderId.empty())
                code.shaderId = shaderId;
            if (!artifactId.empty())
                code.artifactId = artifactId;
            if (code.shaderId || code.artifactId)
                session.shaders.schemaVersion = 2;
            session.shaders.codeVersions.push_back(std::move(code));
            found = session.codeVersionIds.emplace(key, id).first;
        }
        attributes.pipelineId = session.shaders.codeVersions[found->second - 1].pipelineId;
        attributes.codeVersionId = found->second;
        attributes.stage = stage;
        attributes.tier = tier;
        attributes.cache = cache;
        for (auto selection = attributes.codeSelectionId; selection;
             selection = session.codeSelectionParents[selection - 1]) {
            auto &ids = session.shaders.codeSelections->at(selection - 1).codeVersionIds;
            if (std::find(ids.begin(), ids.end(), found->second) == ids.end()) {
                if (ids.size() == maxSelectionEntries) {
                    session.shaderMetadataOverflow = true;
                    fail(E_FAIL, "Profile graphics selection exceeds eight entries");
                } else
                    ids.push_back(found->second);
            }
        }
    } catch (const std::bad_alloc &) {
        session.shaderMetadataOverflow = true;
        session.metadataAllocationFailure = true;
        fail(E_OUTOFMEMORY, "Could not retain profile code-version metadata; capture cannot be saved");
    }
}

void Scope::beginGraphicsSelection(uint64_t pipeline) {
    if (!session_)
        return;
    auto &session = *session_;
    std::lock_guard lock(session.metadataMutex);
    if (session.shaderMetadataOverflow)
        return;
    try {
        if (session.shaders.codeVersions.size() + session.codeSelectionParents.size() == maxCodeVersions) {
            session.shaderMetadataOverflow = true;
            fail(E_FAIL, "Profile code-version/selection metadata limit exceeded");
            return;
        }
        const auto pipelineId =
            session.pipelineIds.try_emplace(pipeline, session.pipelineIds.size() + 1).first->second;
        if (!session.shaders.codeSelections)
            session.shaders.codeSelections.emplace();
        const auto id = session.codeSelectionParents.size() + 1;
        session.shaders.codeSelections->push_back({id, pipelineId, {}});
        const auto parent = event_.attributes.codeSelectionId;
        session.codeSelectionParents.push_back(
            parent && session.shaders.codeSelections->at(parent - 1).pipelineId == pipelineId ? parent : 0);
        session.shaders.schemaVersion = 2;
        event_.attributes.pipelineId = pipelineId;
        event_.attributes.codeVersionId = 0;
        event_.attributes.codeSelectionId = id;
    } catch (const std::bad_alloc &) {
        session.shaderMetadataOverflow = true;
        session.metadataAllocationFailure = true;
        fail(E_OUTOFMEMORY, "Could not retain profile graphics selection; capture cannot be saved");
    }
}

void Scope::observeGraphicsVersion(uint64_t pipeline, Stage stage, uint32_t version, uint8_t tier,
                                   Cache cache, const std::string &shaderId, EntryRole role,
                                   const std::string &artifactId) {
    const auto originalStage = attributes().stage;
    setGraphicsVersion(pipeline, stage, version, tier, cache, shaderId, role, artifactId);
    if (!session_)
        return;
    event_.attributes.stage = originalStage;
    event_.attributes.codeVersionId = 0;
    event_.attributes.tier = 0;
    event_.attributes.cache = Cache::Unknown;
}
Context captureContext() { return currentScope ? currentScope->context() : Context{}; }

void setCompilerConfiguration(const std::string &hash, const std::string &triple, const std::string &cpu,
                              const std::vector<std::string> &features) {
    if (!currentScope)
        return;
    auto &session = *currentScope->session_;
    std::lock_guard lock(session.metadataMutex);
    if (session.compilerConfigurationRecorded.load(std::memory_order_relaxed))
        return;
    if (session.shaderMetadataOverflow)
        return;
    size_t bytes = hash.size() + triple.size() + cpu.size();
    for (const auto &feature : features) {
        if (bytes > maxManifestBytes || feature.size() + 3 > maxManifestBytes - bytes) {
            session.shaderMetadataOverflow = true;
            fail(E_FAIL, "Profile compiler configuration exceeds the manifest format limit");
            return;
        }
        bytes += feature.size() + 3;
    }
    if (bytes > maxManifestBytes) {
        session.shaderMetadataOverflow = true;
        fail(E_FAIL, "Profile compiler configuration exceeds the manifest format limit");
        return;
    }
    try {
        session.compilerHash = hash;
        session.targetTriple = triple;
        session.cpu = cpu;
        session.cpuFeatures = features;
        session.compilerConfigurationRecorded.store(true, std::memory_order_release);
    } catch (const std::bad_alloc &) {
        session.shaderMetadataOverflow = true;
        session.metadataAllocationFailure = true;
        fail(E_OUTOFMEMORY, "Could not retain profile compiler configuration; capture cannot be saved");
    }
}

} // namespace nacelle::profile

extern "C" HRESULT NacelleConfigureProfileSpatial(const NacelleProfileSpatialOptions *options) {
    if (!options || options->struct_size != sizeof(*options) || options->version != 1 ||
        (options->flags & ~3u) || options->record_bytes < sizeof(nacelle::profile::SpatialRecord) ||
        options->record_bytes > 16ull * 1024 * 1024 || !options->sample_stride)
        return fail(E_INVALIDARG, "Invalid spatial options (record budget must be <=16 MiB)");
    for (size_t axis = 0; axis < 3; ++axis)
        if (options->min_xyz[axis] >= options->max_xyz[axis])
            return fail(E_INVALIDARG, "Spatial ROI must be nonempty and half-open");
    std::lock_guard lock(activeMutex);
    if (activeSession) return fail(E_UNEXPECTED, "Configure spatial observation before Begin");
    spatialOptions = *options;
    return S_OK;
}

extern "C" HRESULT NacelleBeginProfileCapture(
    const NacelleProfileOptions* options, NacelleProfileCapture** capture) {
    if (!capture) return fail(E_INVALIDARG, "Capture output is null");
    *capture = nullptr;
    if (!options || options->struct_size != sizeof(*options) || options->version != 1 ||
        options->flags != 0 || options->event_buffer_bytes < sizeof(nacelle::profile::Event) ||
        options->event_buffer_bytes > 64ull * 1024 * 1024) {
        return fail(E_INVALIDARG, "Invalid profile options (event budget must be <= 64 MiB)");
    }
    std::lock_guard lock(activeMutex);
    if (activeSession) return fail(E_UNEXPECTED, "A profile capture is already active");
    try {
        auto handle = std::make_unique<NacelleProfileCapture>();
        handle->session = std::make_shared<nacelle::profile::Session>(options->event_buffer_bytes);
        handle->session->retainedArtifacts = nacelle::profile::artifactStore();
        if (spatialOptions.flags)
            handle->session->spatial = std::make_unique<nacelle::profile::SpatialState>(spatialOptions);
        activeSession = handle->session;
        *capture = handle.release();
        recording.store(true, std::memory_order_release);
        return S_OK;
    } catch (const std::bad_alloc&) {
        return fail(E_OUTOFMEMORY, "Could not allocate profile capture buffers");
    }
}

extern "C" HRESULT NacelleEndProfileCapture(
    NacelleProfileCapture* capture, const char* path, NacelleProfileSummary* summary) {
    if (!capture || !path || !*path || !summary ||
        summary->struct_size != sizeof(*summary) || summary->version != 1) {
        return fail(E_INVALIDARG, "Invalid profile end arguments");
    }
    if (currentScope) return fail(E_UNEXPECTED, "Cannot end capture inside captured work");
    {
        std::lock_guard lock(activeMutex);
        if (!capture->stopped) {
            if (activeSession != capture->session)
                return fail(E_UNEXPECTED, "Capture is not active");
            recording.store(false, std::memory_order_release);
            activeSession.reset();
            capture->stopped = true;
        }
    }
    capture->session->wait();
    summary->recorded_events = capture->session->eventCount;
    summary->dropped_events = capture->session->dropped.load();
    summary->complete = summary->dropped_events == 0 &&
        !capture->session->shaderMetadataOverflow &&
        (!capture->session->spatial || capture->session->spatial->complete());
    if (capture->session->spatial && !capture->session->spatial->complete())
        std::fprintf(stderr, "[NacelleProfile] Spatial observations incomplete; inspect spatial drop counters\n");
    std::string error;
    if (capture->session->metadataAllocationFailure)
        return fail(E_OUTOFMEMORY, "Profile metadata allocation failed; begin a new capture");
    try {
        if (!nacelle::profile::writeProfile(path, *capture->session, error))
            return fail(E_FAIL, error.c_str());
    } catch (const std::bad_alloc&) {
        return fail(E_OUTOFMEMORY, "Could not serialize profile");
    } catch (const std::filesystem::filesystem_error& error) {
        return fail(E_FAIL, error.what());
    }
    return summary->complete ? S_OK : S_FALSE;
}

extern "C" void NacelleDestroyProfileCapture(NacelleProfileCapture* capture) {
    if (!capture) return;
    if (currentScope) {
        fail(E_UNEXPECTED, "Cannot destroy capture inside captured work");
        return;
    }
    {
        std::lock_guard lock(activeMutex);
        if (activeSession == capture->session) {
            recording.store(false, std::memory_order_release);
            activeSession.reset();
        }
    }
    capture->session->wait();
    delete capture;
}
