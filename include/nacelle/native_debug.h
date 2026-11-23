#pragma once

#if defined(NACELLE_ENABLE_INSTRUMENTATION)
#include "instrumentation.h"

struct NacelleNativeDebugOptions {
    uint32_t struct_size = sizeof(NacelleNativeDebugOptions);
    uint32_t version = 1;
    uint32_t enabled = 0;
    uint64_t object_bytes = 32ull * 1024 * 1024;
};

struct NacelleNativeDebugCode {
    uint32_t struct_size = sizeof(NacelleNativeDebugCode);
    uint32_t version = 1;
    uint64_t begin = 0;
    uint64_t end = 0;
    uint64_t code_version = 0;
    char symbol[192]{};
    char object_sha256[65]{};
};

extern "C" {
// Configure before any pipeline compilation/cache lookup. Policy is then frozen.
NACELLE_PROFILE_API HRESULT NacelleConfigureNativeDebug(const NacelleNativeDebugOptions* options);
// S_FALSE means no live native-debug shader range contains address; output is cleared.
NACELLE_PROFILE_API HRESULT NacelleGetNativeDebugCode(uint64_t address, NacelleNativeDebugCode* code);
}
#endif
