#pragma once

#if defined(NACELLE_ENABLE_INSTRUMENTATION)
#include "nacelle.h"

#if defined(_WIN32) && defined(NACELLE_PROFILE_EXPORTS)
#define NACELLE_PROFILE_API __declspec(dllexport)
#else
#define NACELLE_PROFILE_API
#endif

struct NacelleProfileCapture;

enum NacelleProfileSpatialFlags : uint32_t {
    NACELLE_PROFILE_SPATIAL_COMPUTE = 1,
    NACELLE_PROFILE_SPATIAL_PIXEL = 2,
};

struct NacelleProfileSpatialOptions {
    uint32_t struct_size = sizeof(NacelleProfileSpatialOptions);
    uint32_t version = 1;
    uint32_t flags = 0;
    uint64_t record_bytes = 8ull * 1024 * 1024;
    uint32_t sample_stride = 1;
    // Half-open ROI in tile/threadgroup coordinates; defaults include all cells.
    uint32_t min_xyz[3]{0, 0, 0};
    uint32_t max_xyz[3]{UINT32_MAX, UINT32_MAX, UINT32_MAX};
};

enum NacelleProfileArtifactFlags : uint32_t {
    NACELLE_PROFILE_RETAIN_SHADER_TEXT = 1,
    NACELLE_PROFILE_EMBED_SHADER_SOURCES = 2,
};

struct NacelleProfileArtifactOptions {
    uint32_t struct_size = sizeof(NacelleProfileArtifactOptions);
    uint32_t version = 1;
    uint32_t flags = 0;
    uint64_t artifact_bytes = 32ull * 1024 * 1024;
    uint64_t source_bytes = 8ull * 1024 * 1024;
};

struct NacelleProfileOptions {
    uint32_t struct_size = sizeof(NacelleProfileOptions);
    uint32_t version = 1;
    uint64_t event_buffer_bytes = 64ull * 1024 * 1024;
    uint32_t flags = 0;
};

struct NacelleProfileSummary {
    uint32_t struct_size = sizeof(NacelleProfileSummary);
    uint32_t version = 1;
    uint64_t recorded_events = 0;
    uint64_t dropped_events = 0;
    uint32_t complete = 0;
};

// End/Destroy must be called outside captured work, and serialized for each handle.
extern "C" {
// Configure before Begin. Selection limits observation only, never shader execution.
NACELLE_PROFILE_API HRESULT NacelleConfigureProfileSpatial(
    const NacelleProfileSpatialOptions* options);
// Configure before creating pipelines. Sources are private; both opt-ins default OFF.
// Disabling/reconfiguring releases retained data; captured sessions retain their snapshot.
NACELLE_PROFILE_API HRESULT NacelleConfigureProfileArtifacts(
    const NacelleProfileArtifactOptions* options);
NACELLE_PROFILE_API HRESULT NacelleBeginProfileCapture(
    const NacelleProfileOptions* options, NacelleProfileCapture** capture);
NACELLE_PROFILE_API HRESULT NacelleEndProfileCapture(
    NacelleProfileCapture* capture, const char* output_path_utf8,
    NacelleProfileSummary* summary);
NACELLE_PROFILE_API void NacelleDestroyProfileCapture(NacelleProfileCapture* capture);
}
#endif
