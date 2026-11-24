#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>

#if defined(NACELLE_PROFILE_MEMORY_DIAGNOSTICS)
namespace nacelle::profile {
void observeProfileMemory(const char* stage, size_t retainedBytes);
void observeProfileTiming(const char* stage, uint64_t nanoseconds);
void inspectProfileStaging(FILE* file);
}
#define NACELLE_PROFILE_MEMORY(stage, bytes) \
    ::nacelle::profile::observeProfileMemory(stage, bytes)
#define NACELLE_PROFILE_PRIVATE_FILE(file) ::nacelle::profile::inspectProfileStaging(file)
#else
#define NACELLE_PROFILE_MEMORY(stage, bytes)
#define NACELLE_PROFILE_PRIVATE_FILE(file)
#endif
