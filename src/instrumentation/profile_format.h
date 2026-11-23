#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace nacelle::profile {

inline constexpr uint64_t maxProfileBytes = 256ull * 1024 * 1024;
inline constexpr uint64_t maxManifestBytes = 1024ull * 1024;
inline constexpr std::size_t maxCodeVersions = 65536;
inline constexpr std::size_t maxSelectionEntries = 8;
std::string formatProfileHash(uint64_t lo, uint64_t hi);

struct Entry {
    std::string path;
    std::string encoding;
    uint64_t uncompressedSize = 0;
    std::optional<std::string> digest;
};

struct ArtifactMapping {
    uint32_t line = 0;
    uint32_t sourceLine = 0;
    std::string source;
    std::optional<uint64_t> offset;
    std::optional<uint32_t> size;
};

struct TextArtifact {
    std::string id;
    std::string owner;
    std::string kind;
    std::string label;
    std::string text;
    std::vector<ArtifactMapping> mappings;
};

struct ArtifactSelection {
    uint64_t codeVersionId = 0;
    std::vector<std::string> payloadIds;
    std::string shaderAvailability;
    std::string nativeAvailability;
    std::string mappingAvailability = "partial";
    std::string laneAvailability = "unavailable";
};

struct ArtifactMetadata {
    uint32_t schemaVersion = 1;
    std::vector<TextArtifact> payloads;
    std::vector<ArtifactSelection> selections;
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
    std::optional<uint64_t> artifactBytes;
    std::optional<uint64_t> sourceBytes;
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
    std::optional<uint64_t> codeSelectionId;
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
    std::optional<std::string> entryRole;
    std::optional<std::string> shaderId;
    std::optional<std::string> artifactId;
};

struct CodeSelection {
    uint64_t id = 0;
    uint64_t pipelineId = 0;
    std::vector<uint64_t> codeVersionIds;
};

struct ShaderMetadata {
    uint32_t schemaVersion = 1;
    std::vector<CodeVersion> codeVersions;
    std::optional<std::vector<CodeSelection>> codeSelections;
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
    std::optional<ArtifactMetadata> artifacts;
};

bool readProfile(const std::string& path, Profile& profile, std::string& error);
bool readProfileBytes(std::span<const char> archive, Profile& profile, std::string& error);
bool exportTrace(const std::string& path, const Trace& trace, std::string& error);
bool writeProfile(const std::string& path, const struct Session& session, std::string& error);

} // namespace nacelle::profile
