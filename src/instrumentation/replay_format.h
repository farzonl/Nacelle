#pragma once
#include "profile_format.h"

namespace nacelle::replay {
inline constexpr uint64_t maxSnapshotBytes = 192ull * 1024 * 1024;
inline constexpr uint64_t maxShaderBytes = 32ull * 1024 * 1024;
inline constexpr uint64_t maxCommandBytes = 16ull * 1024 * 1024;
inline constexpr uint64_t maxObjects = 4096, maxCommands = 65536;

struct Blob {
    uint32_t id = 0;
    uint64_t offset = 0, size = 0;
    std::string role, digest;
};
struct Resource {
    uint32_t id = 0, heapType = 0, flags = 0, access = 0;
    uint64_t size = 0;
    uint32_t initialBlob = 0, expectedBlob = 0;
};
struct RootSignature { uint32_t id = 0, blob = 0; };
struct Pipeline {
    uint32_t id = 0, root = 0, shaderBlob = 0, nodeMask = 0, flags = 0;
    std::string shaderId, nativeId;
};
struct Heap { uint32_t id = 0, count = 0, flags = 0; };
struct View {
    uint32_t heap = 0, slot = 0, resource = 0;
    std::string kind;
    uint64_t first = 0, bytes = 0;
    uint32_t count = 0, stride = 0, format = 0, flags = 0;
};
struct Command {
    std::string kind;
    std::vector<uint64_t> data;
};
struct Environment {
    profile::Build build;
    uint32_t simdWidth = 0, pointerBits = 0, rounding = 0, fpControl = 0;
    std::string endian = "little", fpPolicy = "flush-denormals";
    std::vector<std::string> compilerOptions;
};
struct Manifest {
    uint32_t schemaVersion = 1;
    std::string kind = "compute-replay", capability = "single-queue-owned-buffers",
                guarantee = "same-environment-complete-race-free-inputs";
    bool complete = true, snapshotConsent = true, shaderConsent = true, sourceConsent = false,
         raceFree = true, hostQuiescent = true;
    uint64_t snapshotBytes = 64ull * 1024 * 1024, shaderBytes = 16ull * 1024 * 1024,
             commandBytes = 8ull * 1024 * 1024;
    Environment environment;
    std::vector<profile::Entry> entries;
};
struct Metadata {
    std::vector<Blob> blobs;
    std::vector<Resource> resources;
    std::vector<RootSignature> roots;
    std::vector<Pipeline> pipelines;
    std::vector<Heap> heaps;
    std::vector<View> views;
    std::vector<Command> commands;
};
struct Bundle { Manifest manifest; Metadata metadata; std::string bytes; };

std::string contentDigest(std::span<const char> bytes);
bool validateBundle(const Bundle& bundle, std::string& error);
bool writeBundle(const std::string& path, Bundle& bundle, std::string& error);
bool readBundleBytes(std::span<const char> bytes, Bundle& bundle, std::string& error);
bool readBundle(const std::string& path, Bundle& bundle, std::string& error);
std::span<const char> blobBytes(const Bundle& bundle, uint32_t id);
}
