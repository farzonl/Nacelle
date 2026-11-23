#pragma once
#if defined(NACELLE_ENABLE_INSTRUMENTATION)
#include "native_debug.h"

enum NacelleSourceDebugStage : uint32_t { NACELLE_SOURCE_COMPUTE = 1, NACELLE_SOURCE_PIXEL = 2 };
enum NacelleSourceDebugAction : uint32_t {
    NACELLE_SOURCE_CONTINUE = 0, NACELLE_SOURCE_STEP = 1, NACELLE_SOURCE_CANCEL = 2
};
enum NacelleSourceDebugValueKind : uint32_t {
    NACELLE_SOURCE_UNAVAILABLE = 0, NACELLE_SOURCE_UNSIGNED = 1,
    NACELLE_SOURCE_SIGNED = 2, NACELLE_SOURCE_FLOAT = 3, NACELLE_SOURCE_BOOL = 4
};
struct NacelleSourceDebugOptions {
    uint32_t struct_size = sizeof(NacelleSourceDebugOptions), version = 1;
    uint32_t enabled = 0, consent = 0; // 1: embedded source; 2: local values.
    uint64_t source_bytes = 8ull * 1024 * 1024, metadata_bytes = 16ull * 1024 * 1024;
};
struct NacelleSourceDebugInlineFrame {
    uint32_t row = 0, column = 0, discriminator = 0;
    char source_id[65]{}, function[96]{};
};
struct NacelleSourceDebugSite {
    uint32_t struct_size = sizeof(NacelleSourceDebugSite), version = 1;
    uint32_t id = 0, stage = 0, line = 0, column = 0, local_count = 0, available = 0;
    uint32_t threads_per_group[3]{1, 1, 1};
    char shader_id[65]{}, source_id[65]{}, label[64]{};
    uint32_t discriminator = 0, inline_depth = 0;
    NacelleSourceDebugInlineFrame frames[8]{};
};
struct NacelleSourceDebugLocal {
    uint32_t kind = 0, bits = 0;
    char name[96]{}, reason[96]{};
};
struct NacelleSourceDebugSelection {
    uint32_t struct_size = sizeof(NacelleSourceDebugSelection), version = 1;
    char shader_id[65]{};
    uint32_t xyz[3]{};
};
struct NacelleSourceDebugSnapshot {
    uint32_t struct_size = sizeof(NacelleSourceDebugSnapshot), version = 1;
    uint64_t session = 0, stop = 0;
    NacelleSourceDebugSite site;
    NacelleNativeDebugCode code;
    uint32_t width = 0, active_mask = 0, real_mask = 0, selected_lane = 0;
    uint32_t xyz[16][3]{};
    NacelleSourceDebugLocal locals[64]{};
    uint64_t values[64][16]{};
};
struct NacelleSourceDebugSummary {
    uint64_t stops = 0;
    uint32_t cancelled = 0, target_seen = 0, concurrent_work = 0;
};
struct NacelleSourceDebugCommand {
    uint64_t session = 0, stop = 0;
    uint32_t action = UINT32_MAX;
};
enum NacelleSourceDebugMapping : uint32_t {
    NACELLE_SOURCE_MAPPING_EXACT = 1, NACELLE_SOURCE_MAPPING_MERGED = 2,
    NACELLE_SOURCE_MAPPING_UNAVAILABLE = 3
};
enum NacelleSourceDebugAvailability : uint32_t {
    NACELLE_SOURCE_VALUE_AVAILABLE = 1, NACELLE_SOURCE_VALUE_UNAVAILABLE = 2,
    NACELLE_SOURCE_VALUE_OPTIMIZED_OUT = 3, NACELLE_SOURCE_VALUE_OUT_OF_SCOPE = 4,
    NACELLE_SOURCE_VALUE_UNSUPPORTED = 5, NACELLE_SOURCE_VALUE_PARTIAL = 6
};
struct NacelleOptimizedSourceDebugOptions {
    uint32_t struct_size = sizeof(NacelleOptimizedSourceDebugOptions), version = 1;
    uint32_t enabled = 0, consent = 0;
    uint64_t source_bytes = 8ull * 1024 * 1024, metadata_bytes = 16ull * 1024 * 1024;
};
struct NacelleOptimizedSourceDebugOrigin {
    uint32_t line = 0, column = 0, discriminator = 0, inline_depth = 0;
    char source_id[65]{};
    NacelleSourceDebugInlineFrame frames[8]{};
};
struct NacelleOptimizedSourceDebugSite {
    uint32_t struct_size = sizeof(NacelleOptimizedSourceDebugSite), version = 1;
    NacelleSourceDebugSite site;
    uint64_t compilation = 0, entry_version = 0;
    uint32_t tier = 0, mapping = NACELLE_SOURCE_MAPPING_UNAVAILABLE;
    uint32_t origin_count = 0;
    NacelleOptimizedSourceDebugOrigin origins[8]{};
    NacelleNativeDebugCode code;
};
struct NacelleOptimizedSourceDebugSnapshot {
    uint32_t struct_size = sizeof(NacelleOptimizedSourceDebugSnapshot), version = 1;
    NacelleSourceDebugSnapshot snapshot;
    uint64_t compilation = 0, entry_version = 0;
    uint32_t tier = 0, mapping = NACELLE_SOURCE_MAPPING_UNAVAILABLE;
    uint32_t origin_count = 0;
    NacelleOptimizedSourceDebugOrigin origins[8]{};
    uint32_t valid_mask[64]{}, availability[64]{};
};
struct NacelleOptimizedSourceDebugStatus {
    uint32_t struct_size = sizeof(NacelleOptimizedSourceDebugStatus), version = 1;
    uint32_t compilation_failed = 0;
    uint64_t tier1_compilations = 0, tier2_compilations = 0;
    uint64_t retained_source_bytes = 0, retained_metadata_bytes = 0;
};
static_assert(sizeof(NacelleSourceDebugSnapshot) <= 64 * 1024);
static_assert(sizeof(NacelleOptimizedSourceDebugSnapshot) <= 64 * 1024);
extern "C" {
NACELLE_PROFILE_API HRESULT NacelleConfigureSourceDebug(const NacelleSourceDebugOptions* options);
// Explicit opt-in; V1 source/native configuration retains its pinned compilation policy.
NACELLE_PROFILE_API HRESULT NacelleConfigureOptimizedSourceDebug(const NacelleOptimizedSourceDebugOptions* options);
NACELLE_PROFILE_API HRESULT NacelleGetOptimizedSourceDebugSite(uint32_t id, NacelleOptimizedSourceDebugSite* site);
NACELLE_PROFILE_API HRESULT NacelleGetOptimizedSourceDebugSnapshot(uint64_t session, NacelleOptimizedSourceDebugSnapshot* snapshot);
NACELLE_PROFILE_API HRESULT NacelleGetOptimizedSourceDebugStatus(NacelleOptimizedSourceDebugStatus* status);
// Enumerate site IDs from 1 until S_FALSE. Sources are embedded, never read from paths.
NACELLE_PROFILE_API HRESULT NacelleGetSourceDebugSite(uint32_t id, NacelleSourceDebugSite* site);
NACELLE_PROFILE_API HRESULT NacelleGetSourceDebugLocal(uint32_t site, uint32_t index, NacelleSourceDebugLocal* local);
NACELLE_PROFILE_API HRESULT NacelleGetSourceDebugSource(const char* source_id, char* bytes, uint64_t* size);
NACELLE_PROFILE_API HRESULT NacelleBeginSourceDebug(const NacelleSourceDebugSelection* selection, uint64_t* session);
NACELLE_PROFILE_API HRESULT NacelleSourceDebugBreakpoint(uint64_t session, const char* source_id, uint32_t line);
NACELLE_PROFILE_API HRESULT NacelleGetSourceDebugSnapshot(uint64_t session, NacelleSourceDebugSnapshot* snapshot);
NACELLE_PROFILE_API HRESULT NacelleControlSourceDebug(uint64_t session, uint64_t stop, uint32_t action);
NACELLE_PROFILE_API HRESULT NacelleEndSourceDebug(uint64_t session, NacelleSourceDebugSummary* summary);
// LLDB all-stop breakpoint target. Returns immediately without a debugger.
NACELLE_PROFILE_API void nacelle_source_debug_stop(
    const NacelleSourceDebugSnapshot* snapshot, NacelleSourceDebugCommand* command);
NACELLE_PROFILE_API void nacelle_optimized_source_debug_stop(
    const NacelleOptimizedSourceDebugSnapshot* snapshot, NacelleSourceDebugCommand* command);
}
#endif
