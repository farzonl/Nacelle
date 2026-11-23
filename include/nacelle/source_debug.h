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
static_assert(sizeof(NacelleSourceDebugSnapshot) <= 64 * 1024);
extern "C" {
NACELLE_PROFILE_API HRESULT NacelleConfigureSourceDebug(const NacelleSourceDebugOptions* options);
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
}
#endif
