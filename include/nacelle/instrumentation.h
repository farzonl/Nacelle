#pragma once

#if defined(NACELLE_ENABLE_INSTRUMENTATION)
#include "nacelle.h"

#if defined(_WIN32) && defined(NACELLE_PROFILE_EXPORTS)
#define NACELLE_PROFILE_API __declspec(dllexport)
#else
#define NACELLE_PROFILE_API
#endif

struct NacelleProfileCapture;

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
NACELLE_PROFILE_API HRESULT NacelleBeginProfileCapture(
    const NacelleProfileOptions* options, NacelleProfileCapture** capture);
NACELLE_PROFILE_API HRESULT NacelleEndProfileCapture(
    NacelleProfileCapture* capture, const char* output_path_utf8,
    NacelleProfileSummary* summary);
NACELLE_PROFILE_API void NacelleDestroyProfileCapture(NacelleProfileCapture* capture);
}
#endif
