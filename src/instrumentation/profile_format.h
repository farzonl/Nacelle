#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace nacelle::profile {

inline constexpr uint64_t maxProfileBytes = 256ull * 1024 * 1024;
inline constexpr uint64_t maxManifestBytes = 1024ull * 1024;
inline constexpr std::size_t maxCodeVersions = 65536;
std::string formatProfileHash(uint64_t lo, uint64_t hi);

struct Entry {
    std::string path;
    std::string encoding;
    uint64_t uncompressedSize = 0;
};

struct Build {
    std::string compiler;
    std::string operatingSystem;
    std::string architecture;
    std::string revision;
    std::string compilerHash;
    std::string targetTriple;
    std::string cpu;
    std::vector<std::string> cpuFeatures;
};

struct CaptureOptions {
    uint64_t eventBufferBytes = 0;
    bool embeddedSources = false;
    bool resourceSnapshots = false;
};

struct Manifest {
    uint32_t schemaVersion = 1;
    std::string kind = "profile";
    bool replayable = false;
    bool complete = true;
    uint64_t recordedEvents = 0;
    uint64_t droppedEvents = 0;
    std::string clock = "monotonic";
    std::string timestampUnit = "ns";
    Build build;
    CaptureOptions captureOptions;
    std::vector<Entry> entries;
};

struct TraceArgs {
    uint64_t id = 0;
    uint64_t parent = 0;
    std::string pso;
    std::string stage;
    uint32_t simdWidth = 0;
    uint32_t tier = 0;
    std::string cache;
    std::optional<uint64_t> pipelineId;
    std::optional<uint64_t> codeVersionId;
};

struct CodeVersion {
    uint64_t id = 0;
    uint64_t pipelineId = 0;
    std::string pso;
    std::string stage = "compute";
    uint32_t simdWidth = 0;
    uint32_t version = 0;
    uint8_t tier = 0;
    std::string cache;
};

struct ShaderMetadata {
    uint32_t schemaVersion = 1;
    std::vector<CodeVersion> codeVersions;
};

struct TraceEvent {
    std::string name;
    std::string cat;
    std::string ph = "X";
    double ts = 0;
    double dur = 0;
    uint32_t pid = 1;
    uint32_t tid = 0;
    TraceArgs args;
};

struct Trace {
    std::vector<TraceEvent> traceEvents;
    std::string displayTimeUnit = "ms";
};

struct Profile {
    Manifest manifest;
    Trace trace;
    ShaderMetadata shaders;
};

bool readProfile(const std::string& path, Profile& profile, std::string& error);
bool exportTrace(const std::string& path, const Trace& trace, std::string& error);
bool writeProfile(const std::string& path, const struct Session& session, std::string& error);

} // namespace nacelle::profile
