#include "source_debug.h"
#include "native_debug.h"

#if defined(NACELLE_ENABLE_INSTRUMENTATION)
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <limits>
#include <map>
#include <mutex>
#include <new>
#include <utility>

namespace nacelle::source_debug {
namespace {
struct Session {
    uint64_t id = 0;
    NacelleSourceDebugSelection selection;
    NacelleSourceDebugSummary summary;
    bool admission = true, observing = true, ending = false, inFlight = false;
    bool step = false, activeSeen = false, acknowledged = false;
    bool operationSelected = false, operationActive = false;
    uint64_t pixelInvocation = 0;
    uint32_t acknowledgedAction = UINT32_MAX;
    std::vector<std::pair<std::string, uint32_t>> breakpoints;
};
struct State {
    std::mutex mutex;
    std::condition_variable drained;
    bool enabled = false, frozen = false, owned = false;
    uint64_t sourceLimit = 8ull * 1024 * 1024, metadataLimit = 16ull * 1024 * 1024;
    uint64_t sourceUsed = 0, metadataUsed = 0, work = 0, sessionSequence = 0, stopSequence = 0;
    std::map<std::string, std::string> sources;
    std::deque<Site> sites;
    Session session;
    NacelleSourceDebugSnapshot snapshot;
};
std::atomic<bool> sourceEnabled{false};
State& state() {
    static State value;
    return value;
}
thread_local unsigned callbackDepth = 0;

HRESULT diagnostic(const char* message, HRESULT result = E_INVALIDARG) {
    std::fprintf(stderr, "source-debug: %s (HRESULT 0x%08x)\n", message,
                 static_cast<unsigned>(result));
    return result;
}
bool identity(const char* text) {
    if (!text) return false;
    for (unsigned i = 0; i < 64; ++i)
        if (!((text[i] >= '0' && text[i] <= '9') || (text[i] >= 'a' && text[i] <= 'f')))
            return false;
    return text[64] == '\0';
}
template<size_t N> bool label(const char (&text)[N], bool allowEmpty = false) {
    for (size_t i = 0; i < N; ++i) {
        const auto c = static_cast<unsigned char>(text[i]);
        if (!c) return allowEmpty || i != 0;
        if (c < 32 || c == 127) return false;
    }
    return false;
}
template<typename T> bool versioned(const T* value) {
    return value && value->struct_size == sizeof(T) && value->version == 1;
}
bool current(const State& data, uint64_t id) {
    return id && data.session.id == id;
}
void cancel(State& data) {
    data.session.observing = false;
    data.session.admission = false;
    data.session.summary.cancelled = 1;
}
HRESULT control(State& data, uint64_t session, uint64_t stop, uint32_t action) {
    if (!current(data, session) || action > NACELLE_SOURCE_CANCEL)
        return diagnostic("Invalid source session or control action");
    // Stop zero cancels a current session between stops, including after an acknowledged stop.
    if (action == NACELLE_SOURCE_CANCEL && !stop && !data.owned) {
        cancel(data);
        return S_OK;
    }
    if (!data.owned || !stop || data.snapshot.stop != stop ||
        data.snapshot.session != session || data.session.acknowledged ||
        (data.session.ending && action != NACELLE_SOURCE_CANCEL))
        return diagnostic("Stale, acknowledged, or unavailable source stop");
    data.session.acknowledged = true;
    data.session.acknowledgedAction = action;
    if (action == NACELLE_SOURCE_CANCEL) cancel(data);
    else data.session.step = action == NACELLE_SOURCE_STEP;
    return S_OK;
}
}

bool enabled() {
    return sourceEnabled.load(std::memory_order_acquire);
}
bool freeze() {
    auto& data = state();
    std::lock_guard lock(data.mutex);
    data.frozen = true;
    return data.enabled;
}
bool addSource(const std::string& id, const std::string& text, std::string& error) {
    try {
        auto& data = state();
        std::lock_guard lock(data.mutex);
        if (!data.enabled) { error = "Source debugging is not enabled"; return false; }
        if (id.size() != 64 || !identity(id.c_str())) {
            error = "Source identity must be 64 lowercase hexadecimal characters"; return false;
        }
        const auto found = data.sources.find(id);
        if (found != data.sources.end()) {
            if (found->second == text) return true;
            error = "Source identity already has different immutable contents"; return false;
        }
        // Include the map node, both string records, and their terminators.
        const uint64_t overhead = sizeof(decltype(data.sources)::value_type) + 4 * sizeof(void*) + 66;
        if (text.size() > data.sourceLimit - std::min(data.sourceUsed, data.sourceLimit) ||
            overhead > data.sourceLimit - data.sourceUsed - text.size()) {
            error = "Embedded source budget exhausted"; return false;
        }
        data.sources.emplace(id, text);
        data.sourceUsed += overhead + text.size();
        return true;
    } catch (const std::bad_alloc&) {
        error = "Out of memory retaining embedded source";
        diagnostic(error.c_str(), E_OUTOFMEMORY);
        return false;
    }
}
uint32_t addSite(Site site, std::string& error) {
    try {
        auto& data = state();
        std::lock_guard lock(data.mutex);
        const auto& location = site.location;
        if (!data.enabled || !versioned(&location) || !identity(location.shader_id) ||
            !identity(location.source_id) || !label(location.label) || !location.line ||
            (location.stage != NACELLE_SOURCE_COMPUTE && location.stage != NACELLE_SOURCE_PIXEL) ||
            site.locals.size() > 64 || location.inline_depth >= 8) {
            error = "Invalid source site, stage, label, identity, or local count"; return 0;
        }
        for (uint32_t i = 0; i <= location.inline_depth; ++i) {
            const auto& frame = location.frames[i];
            const auto embedded = identity(frame.source_id) ? data.sources.find(frame.source_id) : data.sources.end();
            if (embedded == data.sources.end() || !label(frame.function) || !frame.row ||
                frame.row > 1 + std::count(embedded->second.begin(), embedded->second.end(), '\n')) {
                error = "Invalid embedded source inline-frame mapping"; return 0;
            }
        }
        const auto source = data.sources.find(location.source_id);
        if (source == data.sources.end() ||
            location.line > 1 + std::count(source->second.begin(), source->second.end(), '\n')) {
            error = "Source site does not refer to an embedded source line"; return 0;
        }
        for (const auto& local : site.locals) {
            if (!label(local.name) || !label(local.reason, local.kind != NACELLE_SOURCE_UNAVAILABLE) ||
                local.kind > NACELLE_SOURCE_BOOL ||
                (local.kind != NACELLE_SOURCE_UNAVAILABLE &&
                 local.bits != 1 && local.bits != 8 && local.bits != 16 &&
                 local.bits != 32 && local.bits != 64)) {
                error = "Invalid local name, reason, kind, or bit width"; return 0;
            }
        }
        for (const auto& existing : data.sites)
            if (!std::strcmp(existing.location.shader_id, location.shader_id) &&
                existing.location.stage != location.stage) {
                error = "Shader identity has conflicting stages"; return 0;
            }
        // Charge deque node/map slack as well as each fixed-size metadata record.
        const uint64_t bytes = sizeof(Site) + 512 + site.locals.size() * sizeof(NacelleSourceDebugLocal);
        if (bytes > data.metadataLimit - data.metadataUsed ||
            data.sites.size() >= std::numeric_limits<uint32_t>::max()) {
            error = "Source site metadata budget exhausted"; return 0;
        }
        // Do not retain an arbitrarily large capacity supplied by the compiler.
        std::vector<NacelleSourceDebugLocal>(site.locals.begin(), site.locals.end()).swap(site.locals);
        site.location.id = static_cast<uint32_t>(data.sites.size() + 1);
        site.location.local_count = static_cast<uint32_t>(site.locals.size());
        data.sites.push_back(std::move(site));
        data.metadataUsed += bytes;
        return data.sites.back().location.id;
    } catch (const std::bad_alloc&) {
        error = "Out of memory retaining source site";
        diagnostic(error.c_str(), E_OUTOFMEMORY);
        return 0;
    }
}

void publishShader(const std::string& identity) {
    if (!enabled()) return;
    auto& data = state();
    std::lock_guard lock(data.mutex);
    for (auto& site : data.sites)
        if (site.location.shader_id == identity) site.location.available = 1;
}

Operation::Operation(std::string_view shader) {
    if (!enabled()) return;
    auto& data = state();
    std::lock_guard lock(data.mutex);
    auto& session = data.session;
    if (!session.id || !session.observing || shader != session.selection.shader_id) return;
    if (session.operationActive) {
        diagnostic("Concurrent shader operation is not captured", S_FALSE);
        session.summary.concurrent_work = 1;
        cancel(data);
    } else if (!session.operationSelected) {
        admitted = true;
        session.operationSelected = session.operationActive = true;
    }
}
Operation::~Operation() {
    if (!admitted) return;
    auto& data = state();
    std::lock_guard lock(data.mutex);
    data.session.operationActive = false;
    data.session.observing = false;
}

Work::Work() {
    if (!sourceEnabled.load(std::memory_order_acquire)) return;
    auto& data = state();
    std::lock_guard lock(data.mutex);
    if (!data.enabled) return;
    if (data.session.id) {
        if (data.session.admission && !data.work) {
            admitted = true;
            data.session.inFlight = true;
        } else if (data.session.inFlight) {
            data.session.summary.concurrent_work = 1;
            diagnostic("Concurrent submission is not captured", S_FALSE);
            cancel(data);
        }
    }
    ++data.work;
    tracked = true;
}
Work::~Work() {
    if (!tracked) return;
    auto& data = state();
    std::lock_guard lock(data.mutex);
    if (admitted && data.session.id) {
        data.session.admission = false;
        data.session.observing = false;
        data.session.inFlight = false;
    }
    --data.work;
    data.drained.notify_all();
}
}

using namespace nacelle::source_debug;

extern "C" NACELLE_PROFILE_API HRESULT NacelleConfigureSourceDebug(const NacelleSourceDebugOptions* options) {
    try {
        if (!versioned(options) || options->enabled > 1 || options->consent > 3 ||
            (options->enabled && options->consent != 3) ||
            !options->source_bytes || options->source_bytes > 16ull * 1024 * 1024 ||
            !options->metadata_bytes || options->metadata_bytes > 32ull * 1024 * 1024)
            return diagnostic("Invalid configuration, privacy consent, or metadata budget");
#if !defined(__linux__) || !defined(__x86_64__)
        if (options->enabled) return diagnostic("Source debugging requires Linux x86_64");
#endif
        auto& data = state();
        std::lock_guard lock(data.mutex);
        if (data.frozen) return diagnostic("Configure before pipeline compilation or cache admission");
        if (!data.sources.empty() || !data.sites.empty() || data.session.id)
            return diagnostic("Cannot reconfigure retained source metadata");
        if (options->enabled && !nacelle::native_debug::enabled()) {
            NacelleNativeDebugOptions native;
            native.enabled = 1;
            const auto result = NacelleConfigureNativeDebug(&native);
            if (FAILED(result)) return diagnostic("Native debugging could not be enabled", result);
        }
        data.enabled = options->enabled != 0;
        data.sourceLimit = options->source_bytes;
        data.metadataLimit = options->metadata_bytes;
        sourceEnabled.store(data.enabled, std::memory_order_release);
        return S_OK;
    } catch (const std::bad_alloc&) { return diagnostic("Configuration allocation failed", E_OUTOFMEMORY); }
}

extern "C" NACELLE_PROFILE_API HRESULT NacelleGetSourceDebugSite(uint32_t id, NacelleSourceDebugSite* site) {
    try {
        if (!id || !versioned(site)) return diagnostic("Invalid site query");
        auto& data = state();
        std::lock_guard lock(data.mutex);
        *site = {};
        if (id > data.sites.size()) return S_FALSE;
        *site = data.sites[id - 1].location;
        return S_OK;
    } catch (const std::bad_alloc&) { return diagnostic("Site query allocation failed", E_OUTOFMEMORY); }
}
extern "C" NACELLE_PROFILE_API HRESULT NacelleGetSourceDebugLocal(
    uint32_t id, uint32_t index, NacelleSourceDebugLocal* local) {
    try {
        if (!id || !local) return diagnostic("Invalid local query");
        auto& data = state();
        std::lock_guard lock(data.mutex);
        *local = {};
        if (id > data.sites.size() || index >= data.sites[id - 1].locals.size()) return S_FALSE;
        *local = data.sites[id - 1].locals[index];
        return S_OK;
    } catch (const std::bad_alloc&) { return diagnostic("Local query allocation failed", E_OUTOFMEMORY); }
}
extern "C" NACELLE_PROFILE_API HRESULT NacelleGetSourceDebugSource(
    const char* id, char* bytes, uint64_t* size) {
    try {
        if (!identity(id) || !size) return diagnostic("Invalid embedded source query");
        auto& data = state();
        std::lock_guard lock(data.mutex);
        const auto source = data.sources.find(id);
        const auto capacity = *size;
        *size = source == data.sources.end() ? 0 : source->second.size();
        if (source == data.sources.end()) return diagnostic("Embedded source unavailable", S_FALSE);
        if (!bytes) return S_OK;
        if (capacity < *size) return diagnostic("Embedded source buffer is too small");
        std::memcpy(bytes, source->second.data(), *size);
        return S_OK;
    } catch (const std::bad_alloc&) { return diagnostic("Source query allocation failed", E_OUTOFMEMORY); }
}
extern "C" NACELLE_PROFILE_API HRESULT NacelleBeginSourceDebug(
    const NacelleSourceDebugSelection* selection, uint64_t* session) {
    try {
        if (!versioned(selection) || !identity(selection->shader_id) || !session)
            return diagnostic("Invalid source selection");
        *session = 0;
        auto& data = state();
        std::lock_guard lock(data.mutex);
        if (!data.enabled || !nacelle::native_debug::enabled()) return diagnostic("Source/native debugging is disabled");
        if (data.session.id || data.work || data.owned) return diagnostic("Begin requires host-quiescent work and no session");
        const auto found = std::find_if(data.sites.begin(), data.sites.end(), [&](const Site& site) {
            return site.location.available && !std::strcmp(site.location.shader_id, selection->shader_id);
        });
        if (found == data.sites.end()) return diagnostic("Selected shader has no compiled source metadata");
        if (data.sessionSequence == UINT64_MAX) return diagnostic("Session identity space exhausted");
        data.session = {};
        data.snapshot = {};
        data.session.selection = *selection;
        data.session.id = ++data.sessionSequence;
        *session = data.session.id;
        return S_OK;
    } catch (const std::bad_alloc&) { return diagnostic("Session allocation failed", E_OUTOFMEMORY); }
}
extern "C" NACELLE_PROFILE_API HRESULT NacelleSourceDebugBreakpoint(
    uint64_t session, const char* source, uint32_t line) {
    try {
        if (!identity(source) || !line) return diagnostic("Invalid source breakpoint");
        auto& data = state();
        std::lock_guard lock(data.mutex);
        if (!current(data, session) || data.session.ending || !data.session.observing)
            return diagnostic("Stale or closed source session");
        const auto found = std::find_if(data.sites.begin(), data.sites.end(), [&](const Site& site) {
            return !std::strcmp(site.location.shader_id, data.session.selection.shader_id) &&
                !std::strcmp(site.location.source_id, source) && site.location.line == line;
        });
        if (found == data.sites.end()) return diagnostic("Breakpoint line is not executable for the selected shader");
        auto& breakpoints = data.session.breakpoints;
        for (const auto& breakpoint : breakpoints)
            if (breakpoint.first == source && breakpoint.second == line) return S_OK;
        if (breakpoints.size() == 4096) return diagnostic("Source breakpoint limit exceeded");
        breakpoints.emplace_back(source, line);
        return S_OK;
    } catch (const std::bad_alloc&) { return diagnostic("Breakpoint allocation failed", E_OUTOFMEMORY); }
}
extern "C" NACELLE_PROFILE_API HRESULT NacelleGetSourceDebugSnapshot(
    uint64_t session, NacelleSourceDebugSnapshot* snapshot) {
    try {
        if (!versioned(snapshot)) return diagnostic("Invalid snapshot query");
        auto& data = state();
        std::lock_guard lock(data.mutex);
        if (!current(data, session)) return diagnostic("Stale source session");
        *snapshot = {};
        if (!data.snapshot.stop || data.snapshot.session != session) return S_FALSE;
        *snapshot = data.snapshot;
        return S_OK;
    } catch (const std::bad_alloc&) { return diagnostic("Snapshot query allocation failed", E_OUTOFMEMORY); }
}
extern "C" NACELLE_PROFILE_API HRESULT NacelleControlSourceDebug(
    uint64_t session, uint64_t stop, uint32_t action) {
    try {
        auto& data = state();
        std::lock_guard lock(data.mutex);
        return control(data, session, stop, action);
    } catch (const std::bad_alloc&) { return diagnostic("Control allocation failed", E_OUTOFMEMORY); }
}
extern "C" NACELLE_PROFILE_API HRESULT NacelleEndSourceDebug(
    uint64_t session, NacelleSourceDebugSummary* summary) {
    try {
        if (!summary || callbackDepth) return diagnostic("End must be outside an observation callback");
        auto& data = state();
        std::unique_lock lock(data.mutex);
        if (!current(data, session) || data.session.ending) return diagnostic("Stale or already ending source session");
        data.session.ending = true;
        data.session.admission = false;
        data.session.observing = false;
        data.drained.wait(lock, [&] { return data.work == 0 && !data.owned; });
        *summary = data.session.summary;
        const bool complete = data.session.activeSeen && !summary->cancelled &&
            !summary->concurrent_work && !data.session.step;
        data.session = {};
        return complete ? S_OK : diagnostic("Session incomplete: cancelled, missing active target/step, or rejected work", S_FALSE);
    } catch (const std::bad_alloc&) { return diagnostic("End allocation failed", E_OUTOFMEMORY); }
}

#if defined(_MSC_VER)
#define NACELLE_SOURCE_NOINLINE __declspec(noinline)
#else
#define NACELLE_SOURCE_NOINLINE __attribute__((noinline))
#endif
extern "C" NACELLE_PROFILE_API NACELLE_SOURCE_NOINLINE void nacelle_source_debug_stop(
    const NacelleSourceDebugSnapshot* snapshot, NacelleSourceDebugCommand* command) {
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : "r"(snapshot), "r"(command) : "memory");
#else
    (void)snapshot;
    (void)command;
#endif
}

extern "C" NACELLE_SOURCE_NOINLINE void nacelle_source_debug_observe(
    uint32_t site, uint32_t width, uint32_t active, uint32_t real,
    const uint32_t* xyz, const uint64_t* values, const uint64_t* invocations) try {
#if defined(__GNUC__) || defined(__clang__)
    const auto caller = reinterpret_cast<uint64_t>(__builtin_extract_return_addr(__builtin_return_address(0)));
#else
    const uint64_t caller = 0;
#endif
    struct Callback {
        Callback() { ++callbackDepth; }
        ~Callback() { --callbackDepth; }
    } callback;
    auto& data = state();
    uint64_t sessionId = 0, stopId = 0;
    {
        std::lock_guard lock(data.mutex);
        auto& session = data.session;
        if (!session.id || !session.observing || !session.operationActive || !data.work) return;
        if (!site || site > data.sites.size() || !width || width > 16 || !xyz) {
            diagnostic("Invalid source observation"); cancel(data); return;
        }
        const auto& metadata = data.sites[site - 1];
        if (!std::strcmp(metadata.location.shader_id, session.selection.shader_id)) {
            if (!values && !metadata.locals.empty()) {
                diagnostic("Source observation is missing local values"); cancel(data); return;
            }
            uint32_t lane = width;
            for (uint32_t i = 0; i < width; ++i) {
                if (std::equal(session.selection.xyz, session.selection.xyz + 3, xyz + i * 3)) {
                    session.summary.target_seen = 1;
                    if ((active & (1u << i)) && (real & (1u << i))) lane = i;
                }
            }
            if (lane == width) return;
            if (metadata.location.stage == NACELLE_SOURCE_PIXEL) {
                if (!invocations || !invocations[lane]) {
                    diagnostic("Pixel observation has no primitive lifetime identity");
                    cancel(data); return;
                }
                if (session.pixelInvocation && session.pixelInvocation != invocations[lane]) return;
                session.pixelInvocation = invocations[lane];
            }
            session.activeSeen = true;
            bool hit = session.step && metadata.location.inline_depth <= data.snapshot.site.inline_depth;
            for (const auto& breakpoint : session.breakpoints)
                hit |= breakpoint.second == metadata.location.line &&
                    breakpoint.first == metadata.location.source_id;
            if (!hit || data.owned) return;
            NacelleNativeDebugCode code;
            if (NacelleGetNativeDebugCode(caller, &code) != S_OK || !code.code_version ||
                !identity(code.object_sha256)) {
                diagnostic("Observation caller has no live native code identity");
                cancel(data); return;
            }
            if (data.stopSequence == UINT64_MAX) {
                diagnostic("Stop identity space exhausted"); cancel(data); return;
            }
            data.snapshot = {};
            auto& snapshot = data.snapshot;
            snapshot.session = sessionId = session.id;
            snapshot.stop = stopId = ++data.stopSequence;
            snapshot.site = metadata.location;
            std::copy(metadata.locals.begin(), metadata.locals.end(), snapshot.locals);
            snapshot.code = code;
            snapshot.width = width;
            snapshot.active_mask = active & ((1u << width) - 1);
            snapshot.real_mask = real & ((1u << width) - 1);
            snapshot.selected_lane = lane;
            std::memcpy(snapshot.xyz, xyz, width * 3 * sizeof(uint32_t));
            for (size_t local = 0; local < metadata.locals.size(); ++local)
                std::memcpy(snapshot.values[local], values + local * width, width * sizeof(uint64_t));
            data.owned = true;
            session.acknowledged = false;
            session.acknowledgedAction = UINT32_MAX;
            session.step = false;
            ++session.summary.stops;
        } else return;
    }
    // LLDB owns all-stop behavior. No shader thread is parked by this runtime.
    NacelleSourceDebugCommand command;
    command.session = sessionId;
    command.stop = stopId;
    nacelle_source_debug_stop(&data.snapshot, &command);
    {
        std::lock_guard lock(data.mutex);
        if (data.owned && current(data, sessionId) && data.snapshot.stop == stopId) {
            if (command.session != sessionId || command.stop != stopId ||
                (command.action > NACELLE_SOURCE_CANCEL && command.action != UINT32_MAX)) {
                diagnostic("Debugger mailbox has stale identities or an invalid action");
                cancel(data);
            } else if (command.action != UINT32_MAX) {
                if (data.session.acknowledged) {
                    if (command.action != data.session.acknowledgedAction) {
                        diagnostic("Debugger mailbox conflicts with acknowledged controller action");
                        cancel(data);
                    }
                } else if (FAILED(control(data, command.session, command.stop, command.action))) {
                    cancel(data);
                }
            }
            if (!data.session.acknowledged) {
                diagnostic("Debugger rendezvous returned without control acknowledgement; cancelling observations", S_FALSE);
                cancel(data);
            }
            data.owned = false;
            data.drained.notify_all();
        }
    }
} catch (const std::bad_alloc&) {
    diagnostic("Observation allocation failed; cancelling observations", E_OUTOFMEMORY);
    auto& data = state();
    std::lock_guard lock(data.mutex);
    if (data.session.id) cancel(data);
}
#endif
