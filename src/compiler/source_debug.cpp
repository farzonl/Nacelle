#include "source_debug.h"
#include "native_debug.h"

#if defined(NACELLE_ENABLE_INSTRUMENTATION)
#include "shader_compiler.h"
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
    uint32_t inlineDepth = 0;
    uint64_t compilation = 0;
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
    NacelleOptimizedSourceDebugSnapshot optimizedSnapshot;
    bool compilationFailed = false;
    uint64_t tier1Compilations = 0, tier2Compilations = 0;
};
std::atomic<bool> sourceEnabled{false};
std::atomic<bool> optimizedEnabled{false};
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
bool materialized(const Site& site) {
    if (!site.location.available) return false;
    if (!optimizedEnabled.load(std::memory_order_acquire)) return true;
    NacelleNativeDebugCode live;
    return site.compilation && site.entryVersion &&
        NacelleGetNativeDebugCode(site.code.begin, &live) == S_OK &&
        live.code_version == site.code.code_version &&
        !std::strcmp(live.object_sha256, site.code.object_sha256);
}
template<typename Location> bool validOrigin(const State& data, const Location& location) {
    if (!identity(location.source_id) || !location.line || location.inline_depth >= 8) return false;
    const auto source = data.sources.find(location.source_id);
    if (source == data.sources.end() ||
        location.line > 1 + std::count(source->second.begin(), source->second.end(), '\n')) return false;
    for (uint32_t i = 0; i <= location.inline_depth; ++i) {
        const auto& frame = location.frames[i];
        const auto embedded = identity(frame.source_id) ? data.sources.find(frame.source_id) : data.sources.end();
        if (embedded == data.sources.end() || !label(frame.function) || !frame.row ||
            frame.row > 1 + std::count(embedded->second.begin(), embedded->second.end(), '\n')) return false;
    }
    return true;
}
bool sameOrigin(const NacelleOptimizedSourceDebugOrigin& a, const NacelleOptimizedSourceDebugOrigin& b) {
    return a.line == b.line && a.column == b.column && a.discriminator == b.discriminator &&
        a.inline_depth == b.inline_depth && !std::strcmp(a.source_id, b.source_id) &&
        std::equal(a.frames, a.frames + a.inline_depth + 1, b.frames, [](const auto& x, const auto& y) {
            return x.row == y.row && x.column == y.column && x.discriminator == y.discriminator &&
                !std::strcmp(x.source_id, y.source_id) && !std::strcmp(x.function, y.function);
        });
}
void copyOrigins(const Site& site, uint32_t& count, NacelleOptimizedSourceDebugOrigin (&origins)[8]) {
    count = site.origins.empty() ? 1 : static_cast<uint32_t>(site.origins.size());
    if (site.origins.empty()) origins[0] = primaryOrigin(site.location);
    else std::copy(site.origins.begin(), site.origins.end(), origins);
}
bool matches(const Site& site, const char* source, uint32_t line) {
    if (site.location.line == line && !std::strcmp(site.location.source_id, source)) return true;
    return std::any_of(site.origins.begin(), site.origins.end(), [&](const auto& origin) {
        return origin.line == line && !std::strcmp(origin.source_id, source);
    });
}
uint32_t inlineDepth(const Site& site) {
    auto depth = site.location.inline_depth;
    for (const auto& origin : site.origins) depth = std::min(depth, origin.inline_depth);
    return depth;
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

NacelleOptimizedSourceDebugOrigin primaryOrigin(const NacelleSourceDebugSite& site) {
    NacelleOptimizedSourceDebugOrigin origin;
    origin.line = site.line;
    origin.column = site.column;
    origin.discriminator = site.discriminator;
    origin.inline_depth = site.inline_depth;
    std::memcpy(origin.source_id, site.source_id, sizeof(origin.source_id));
    std::copy_n(site.frames, 8, origin.frames);
    return origin;
}
bool enabled() {
    return sourceEnabled.load(std::memory_order_acquire);
}
bool optimized() {
    return optimizedEnabled.load(std::memory_order_acquire);
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
            site.locals.size() > 64 || site.origins.size() > 8 || location.inline_depth >= 8 ||
            (!site.availability.empty() && site.availability.size() != site.locals.size()) ||
            site.mapping < NACELLE_SOURCE_MAPPING_EXACT || site.mapping > NACELLE_SOURCE_MAPPING_UNAVAILABLE) {
            error = "Invalid source site, stage, label, identity, or local count"; return 0;
        }
        if (!validOrigin(data, location)) {
            error = "Invalid embedded source line or inline-frame mapping"; return 0;
        }
        bool primaryFound = site.origins.empty();
        const auto primary = primaryOrigin(location);
        for (size_t i = 0; i < site.origins.size(); ++i) {
            if (!validOrigin(data, site.origins[i])) {
                error = "Invalid merged source origin"; return 0;
            }
            primaryFound |= sameOrigin(primary, site.origins[i]);
            for (size_t j = 0; j < i; ++j)
                if (sameOrigin(site.origins[j], site.origins[i])) {
                    error = "Duplicate merged source origin"; return 0;
                }
        }
        if (!primaryFound || (site.mapping == NACELLE_SOURCE_MAPPING_MERGED && site.origins.size() < 2) ||
            (site.mapping == NACELLE_SOURCE_MAPPING_EXACT && site.origins.size() > 1)) {
            error = "Source mapping classification does not match its complete origin set"; return 0;
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
        for (size_t i = 0; i < site.availability.size(); ++i) {
            const auto availability = site.availability[i];
            if (availability < NACELLE_SOURCE_VALUE_AVAILABLE ||
                availability > NACELLE_SOURCE_VALUE_PARTIAL ||
                (availability == NACELLE_SOURCE_VALUE_AVAILABLE) !=
                    (site.locals[i].kind != NACELLE_SOURCE_UNAVAILABLE)) {
                error = "Local kind and optimized availability are inconsistent"; return 0;
            }
        }
        for (const auto& existing : data.sites)
            if (!std::strcmp(existing.location.shader_id, location.shader_id) &&
                existing.location.stage != location.stage) {
                error = "Shader identity has conflicting stages"; return 0;
            }
        // Charge deque node/map slack as well as each fixed-size metadata record.
        const uint64_t bytes = sizeof(Site) + 512 + site.origins.size() * sizeof(NacelleOptimizedSourceDebugOrigin) +
            site.locals.size() *
            (sizeof(NacelleSourceDebugLocal) + sizeof(uint32_t));
        if (bytes > data.metadataLimit - data.metadataUsed ||
            data.sites.size() >= std::numeric_limits<uint32_t>::max()) {
            error = "Source site metadata budget exhausted"; return 0;
        }
        // Do not retain an arbitrarily large capacity supplied by the compiler.
        std::vector<NacelleSourceDebugLocal>(site.locals.begin(), site.locals.end()).swap(site.locals);
        std::vector<uint32_t>(site.availability.begin(), site.availability.end()).swap(site.availability);
        std::vector<NacelleOptimizedSourceDebugOrigin>(site.origins.begin(), site.origins.end()).swap(site.origins);
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

bool readSite(uint32_t id, Site& site) {
    auto& data = state();
    std::lock_guard lock(data.mutex);
    if (!id || id > data.sites.size()) return false;
    site = data.sites[id - 1];
    return true;
}
bool bindCompilation(uint64_t compilation, uint64_t address, uint64_t entryVersion, std::string& error) {
    if (!optimized() || !compilation) return true;
    NacelleNativeDebugCode code;
    if (NacelleGetNativeDebugCode(address, &code) != S_OK || !entryVersion) {
        error = "Optimized source sites have no live native entry identity"; return false;
    }
    auto& data = state();
    std::lock_guard lock(data.mutex);
    for (auto& site : data.sites) {
        if (site.compilation != compilation) continue;
        if (site.entryVersion || site.code.code_version) {
            error = "Optimized source compilation was already bound"; return false;
        }
    }
    for (auto& site : data.sites) {
        if (site.compilation != compilation) continue;
        site.code = code;
        site.entryVersion = entryVersion;
        site.location.available = site.mapping != NACELLE_SOURCE_MAPPING_UNAVAILABLE;
    }
    const auto found = std::find_if(data.sites.begin(), data.sites.end(),
        [&](const Site& site) { return site.compilation == compilation; });
    if (found != data.sites.end()) {
        if (found->tier == 1) ++data.tier1Compilations;
        if (found->tier == 2) ++data.tier2Compilations;
    }
    return true;
}
void reportCompilationFailure(const std::string& error) {
    if (!optimized()) return;
    auto& data = state();
    std::lock_guard lock(data.mutex);
    data.compilationFailed = true;
    diagnostic(error.empty() ? "Optimized debug compilation failed" : error.c_str(), E_FAIL);
}

void publishShader(const std::string& identity) {
    if (!enabled() || optimized()) return;
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

namespace {
HRESULT configureSource(const NacelleSourceDebugOptions* options, bool optimize) {
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
        if (optimize && nacelle::native_debug::enabled() && !optimized())
            return diagnostic("Pinned native debugging conflicts with optimized source configuration");
        if (options->enabled && !nacelle::native_debug::enabled()) {
            NacelleNativeDebugOptions native;
            native.enabled = 1;
            const auto result = NacelleConfigureNativeDebug(&native);
            if (FAILED(result)) return diagnostic("Native debugging could not be enabled", result);
        }
        data.enabled = options->enabled != 0;
        data.sourceLimit = options->source_bytes;
        data.metadataLimit = options->metadata_bytes;
        optimizedEnabled.store(options->enabled && optimize, std::memory_order_release);
        sourceEnabled.store(data.enabled, std::memory_order_release);
        return S_OK;
    } catch (const std::bad_alloc&) { return diagnostic("Configuration allocation failed", E_OUTOFMEMORY); }
}
}
extern "C" NACELLE_PROFILE_API HRESULT NacelleConfigureSourceDebug(const NacelleSourceDebugOptions* options) {
    return configureSource(options, false);
}
extern "C" NACELLE_PROFILE_API HRESULT NacelleConfigureOptimizedSourceDebug(
    const NacelleOptimizedSourceDebugOptions* options) {
    if (!versioned(options)) return diagnostic("Invalid optimized source configuration");
    NacelleSourceDebugOptions base;
    base.enabled = options->enabled;
    base.consent = options->consent;
    base.source_bytes = options->source_bytes;
    base.metadata_bytes = options->metadata_bytes;
    return configureSource(&base, true);
}

extern "C" NACELLE_PROFILE_API HRESULT NacelleGetOptimizedSourceDebugSite(
    uint32_t id, NacelleOptimizedSourceDebugSite* site) {
    if (!id || !versioned(site)) return diagnostic("Invalid optimized site query");
    auto& data = state();
    std::lock_guard lock(data.mutex);
    *site = {};
    if (id > data.sites.size()) return S_FALSE;
    const auto& stored = data.sites[id - 1];
    site->site = stored.location;
    site->compilation = stored.compilation;
    site->tier = stored.tier;
    site->entry_version = stored.entryVersion;
    site->mapping = stored.mapping;
    copyOrigins(stored, site->origin_count, site->origins);
    site->code = stored.code;
    if (site->site.available && !materialized(stored)) {
        site->site.available = 0;
        site->mapping = NACELLE_SOURCE_MAPPING_UNAVAILABLE;
    }
    return S_OK;
}
extern "C" NACELLE_PROFILE_API HRESULT NacelleGetOptimizedSourceDebugStatus(
    NacelleOptimizedSourceDebugStatus* status) {
    if (!versioned(status) || !optimized()) return diagnostic("Invalid optimized status query");
    auto& data = state();
    std::lock_guard lock(data.mutex);
    *status = {};
    status->compilation_failed = data.compilationFailed;
    status->tier1_compilations = data.tier1Compilations;
    status->tier2_compilations = data.tier2Compilations;
    status->retained_source_bytes = data.sourceUsed;
    status->retained_metadata_bytes = data.metadataUsed;
    return data.compilationFailed ? S_FALSE : S_OK;
}

extern "C" NACELLE_PROFILE_API HRESULT NacelleGetSourceDebugSite(uint32_t id, NacelleSourceDebugSite* site) {
    try {
        if (!id || !versioned(site)) return diagnostic("Invalid site query");
        auto& data = state();
        std::lock_guard lock(data.mutex);
        *site = {};
        if (id > data.sites.size()) return S_FALSE;
        *site = data.sites[id - 1].location;
        site->available = materialized(data.sites[id - 1]);
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
            return materialized(site) && !std::strcmp(site.location.shader_id, selection->shader_id);
        });
        if (found == data.sites.end()) return diagnostic("Selected shader has no compiled source metadata");
        if (data.sessionSequence == UINT64_MAX) return diagnostic("Session identity space exhausted");
        data.session = {};
        data.snapshot = {};
        data.optimizedSnapshot = {};
        data.session.selection = *selection;
        data.session.id = ++data.sessionSequence;
        *session = data.session.id;
        return S_OK;
    } catch (const std::bad_alloc&) { return diagnostic("Session allocation failed", E_OUTOFMEMORY); }
}
extern "C" NACELLE_PROFILE_API HRESULT NacelleGetOptimizedSourceDebugSnapshot(
    uint64_t session, NacelleOptimizedSourceDebugSnapshot* snapshot) {
    if (!versioned(snapshot)) return diagnostic("Invalid optimized snapshot query");
    auto& data = state();
    std::lock_guard lock(data.mutex);
    if (!current(data, session) || !optimized()) return diagnostic("Stale or non-optimized source session");
    *snapshot = {};
    if (!data.optimizedSnapshot.snapshot.stop) return S_FALSE;
    *snapshot = data.optimizedSnapshot;
    return S_OK;
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
            return (!optimized() || materialized(site)) &&
                !std::strcmp(site.location.shader_id, data.session.selection.shader_id) &&
                matches(site, source, line);
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
            !summary->concurrent_work && !data.session.step && !data.compilationFailed;
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
extern "C" NACELLE_PROFILE_API NACELLE_SOURCE_NOINLINE void nacelle_optimized_source_debug_stop(
    const NacelleOptimizedSourceDebugSnapshot* snapshot, NacelleSourceDebugCommand* command) {
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
            if (optimized()) {
                if (!metadata.location.available || !metadata.compilation) {
                    diagnostic("Optimized source observation is not bound to a materialized entry");
                    cancel(data); return;
                }
                if (session.compilation && session.compilation != metadata.compilation) return;
                session.compilation = metadata.compilation;
            }
            if (metadata.location.stage == NACELLE_SOURCE_PIXEL) {
                if (!invocations || !invocations[lane]) {
                    diagnostic("Pixel observation has no primitive lifetime identity");
                    cancel(data); return;
                }
                if (session.pixelInvocation && session.pixelInvocation != invocations[lane]) return;
                session.pixelInvocation = invocations[lane];
            }
            session.activeSeen = true;
            bool hit = session.step && inlineDepth(metadata) <= session.inlineDepth;
            for (const auto& breakpoint : session.breakpoints)
                hit |= matches(metadata, breakpoint.first.c_str(), breakpoint.second);
            if (!hit || data.owned) return;
            NacelleNativeDebugCode code;
            if (NacelleGetNativeDebugCode(caller, &code) != S_OK || !code.code_version ||
                !identity(code.object_sha256)) {
                diagnostic("Observation caller has no live native code identity");
                cancel(data); return;
            }
            if (optimized() && (code.code_version != metadata.code.code_version ||
                std::strcmp(code.object_sha256, metadata.code.object_sha256))) {
                diagnostic("Optimized observation belongs to a stale or different native object");
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
            if (optimized()) {
                auto& detail = data.optimizedSnapshot;
                detail = {};
                detail.snapshot = snapshot;
                detail.compilation = metadata.compilation;
                detail.entry_version = metadata.entryVersion;
                detail.tier = metadata.tier;
                detail.mapping = metadata.mapping;
                copyOrigins(metadata, detail.origin_count, detail.origins);
                for (size_t local = 0; local < metadata.locals.size(); ++local) {
                    detail.valid_mask[local] = metadata.locals[local].kind ? snapshot.active_mask : 0;
                    detail.availability[local] = metadata.availability.empty() ?
                        (metadata.locals[local].kind ? NACELLE_SOURCE_VALUE_AVAILABLE : NACELLE_SOURCE_VALUE_UNAVAILABLE) :
                        metadata.availability[local];
                }
            }
            data.owned = true;
            session.acknowledged = false;
            session.acknowledgedAction = UINT32_MAX;
            session.step = false;
            session.inlineDepth = inlineDepth(metadata);
            ++session.summary.stops;
        } else return;
    }
    // LLDB owns all-stop behavior. No shader thread is parked by this runtime.
    NacelleSourceDebugCommand command;
    command.session = sessionId;
    command.stop = stopId;
    if (optimized()) nacelle_optimized_source_debug_stop(&data.optimizedSnapshot, &command);
    else nacelle_source_debug_stop(&data.snapshot, &command);
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

extern "C" bool nacelle_source_debug_read_pixel_context(const nacelle::PixelBatchArgs* args,
    uint64_t batch, uint32_t width, uint32_t* xyz, uint32_t* masks, uint64_t* invocations) noexcept {
    if (!xyz || !masks || !invocations || (width != 4 && width != 8 && width != 16)) return false;
    std::fill_n(xyz, width * 3, UINT32_MAX);
    std::fill_n(invocations, width, 0);
    masks[0] = masks[1] = 0;
    if (!args || !args->batchCount || args->batchCount > 64 || batch >= args->batchCount ||
        !args->pixelX || !args->pixelY || args->sampleCoverageMasks ||
        (!args->coverageMasks && batch != 0)) return false;
    const uint32_t real = (args->coverageMasks ? args->coverageMasks[batch] : args->coverageMask) &
        ((1u << width) - 1);
    uint32_t active = 0;
    for (uint32_t quad = 0; quad < width / 4; ++quad) {
        const uint32_t quadMask = 15u << (quad * 4);
        if (!(real & quadMask)) continue;
        const float* identity = args->triInvW;
        if (args->triData) {
            if (!args->quadTriIndices) return false;
            const auto triangle = args->quadTriIndices[batch * (width / 4) + quad];
            // PixelBatcher owns at most one descriptor per non-padding quad.
            if (triangle >= args->batchCount * (width / 4)) return false;
            identity = args->triData[triangle].triInvW;
        }
        if (!identity) return false;
        active |= quadMask;
        for (uint32_t lane = quad * 4; lane < quad * 4 + 4; ++lane)
            invocations[lane] = reinterpret_cast<uintptr_t>(identity);
    }
    for (uint32_t lane = 0; lane < width; ++lane) {
        const float coordinates[]{args->pixelX[batch * width + lane], args->pixelY[batch * width + lane]};
        for (uint32_t dimension = 0; dimension < 2; ++dimension)
            if (coordinates[dimension] >= 0.0f && coordinates[dimension] < 4294967296.0f)
                xyz[lane * 3 + dimension] = static_cast<uint32_t>(coordinates[dimension]);
        xyz[lane * 3 + 2] = 0;
    }
    masks[0] = active;
    masks[1] = real;
    return true;
}
#endif
