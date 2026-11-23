#pragma once
#if defined(NACELLE_ENABLE_INSTRUMENTATION)
#include "instrumentation.h"

struct NacelleReplayCapture;
enum NacelleReplayConsent : uint32_t {
    NACELLE_REPLAY_SNAPSHOTS = 1,
    NACELLE_REPLAY_SHADER_BINARIES = 2,
    NACELLE_REPLAY_EMBEDDED_SOURCES = 4,
};
struct NacelleReplayConfiguration {
    uint32_t struct_size = sizeof(NacelleReplayConfiguration), version = 1, flags = 0;
    uint64_t snapshot_bytes = 64ull * 1024 * 1024;
    uint64_t shader_bytes = 16ull * 1024 * 1024;
    uint64_t command_bytes = 8ull * 1024 * 1024;
};
enum NacelleReplayAccess : uint32_t { NACELLE_REPLAY_READ = 1, NACELLE_REPLAY_WRITE = 2 };
struct NacelleReplayResource {
    ID3D12Resource* resource = nullptr;
    uint32_t access = 0;
};
enum NacelleReplayContract : uint32_t {
    NACELLE_REPLAY_RACE_FREE_INPUTS = 1,
    NACELLE_REPLAY_HOST_QUIESCENT = 2,
};
struct NacelleReplayCaptureOptions {
    uint32_t struct_size = sizeof(NacelleReplayCaptureOptions), version = 1, contracts = 0;
    uint32_t resource_count = 0;
    const NacelleReplayResource* resources = nullptr;
};
extern "C" {
// Enable before participating object creation; snapshot and shader consent are separate.
NACELLE_PROFILE_API HRESULT NacelleConfigureReplay(const NacelleReplayConfiguration* configuration);
NACELLE_PROFILE_API HRESULT NacelleWaitForReplayCompilation();
NACELLE_PROFILE_API HRESULT NacelleBeginReplayCapture(
    const NacelleReplayCaptureOptions* options, NacelleReplayCapture** capture);
NACELLE_PROFILE_API HRESULT NacelleEndReplayCapture(NacelleReplayCapture* capture, const char* output_path_utf8);
NACELLE_PROFILE_API void NacelleDestroyReplayCapture(NacelleReplayCapture* capture);
// execution_consent must be 1 to invoke trusted shader code; 0 validates only.
NACELLE_PROFILE_API HRESULT NacelleReplayBundle(
    const char* input_path_utf8, uint32_t execution_consent, uint32_t repetitions);
}
#endif
