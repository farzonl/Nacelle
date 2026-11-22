#pragma once

#if defined(NACELLE_ENABLE_INSTRUMENTATION)
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include "profile_format.h"

namespace nacelle::profile {

enum class Stage : uint8_t { Unknown, Compute, Graphics, Vertex, Pixel, Mesh, Raytracing };
enum class Cache : uint8_t { Unknown, Hit, Miss };

struct Attributes {
    uint64_t hashLo;
    uint64_t hashHi;
    uint32_t simdWidth;
    Stage stage;
    uint8_t tier;
    Cache cache;
    uint64_t pipelineId = 0;
    uint64_t codeVersionId = 0;
};

uint64_t allocatePipelineIdentity();

struct Event {
    uint64_t startNs;
    uint64_t durationNs;
    uint64_t id;
    uint64_t parent;
    Attributes attributes;
    uint32_t thread;
    char name[48];
};

struct Session {
    explicit Session(uint64_t bytes);
    void releaseWork();
    void wait();
    uint64_t allocateEventId();
    void record(Event event);

    uint64_t generation;
    uint64_t bufferBytes;
    size_t capacity;
    std::unique_ptr<Event[]> events;
    std::chrono::steady_clock::time_point origin = std::chrono::steady_clock::now();
    alignas(64) std::atomic<size_t> nextBlock{0};
    uint64_t eventCount = 0; // Finalized only after all writers drain.
    alignas(64) std::atomic<uint64_t> dropped{0};
    alignas(64) std::atomic<uint64_t> nextId{1};
    alignas(64) std::atomic<uint64_t> outstanding{0};
    alignas(64) std::atomic<uint32_t> nextThread{1};
    std::mutex waitMutex;
    std::condition_variable drained;
    std::mutex metadataMutex;
    std::atomic<bool> compilerConfigurationRecorded{false};
    std::string compilerHash, targetTriple, cpu;
    std::vector<std::string> cpuFeatures;
    std::map<uint64_t, uint64_t> pipelineIds;
    std::map<std::pair<uint64_t, uint32_t>, uint64_t> codeVersionIds;
    ShaderMetadata shaders;
    bool shaderMetadataOverflow = false;
};

struct Work {
    explicit Work(std::shared_ptr<Session> session);
    ~Work();
    std::shared_ptr<Session> session;
};

struct Context {
    std::shared_ptr<Work> work;
    uint64_t parent = 0;
    Attributes attributes{};
};

class Scope {
public:
    explicit Scope(const char* name);
    Scope(const Context& context, const char* name, uint8_t tier = 0);
    ~Scope();
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
    void setAttributes(Attributes attributes) { if (session_) event_.attributes = attributes; }
    void setComputeVersion(uint64_t pipeline, uint32_t version, uint8_t tier, Cache cache);
    Attributes attributes() const { return session_ ? event_.attributes : Attributes{}; }
    bool enabled() const { return static_cast<bool>(session_); }
    bool needsCompilerConfiguration() const {
        return session_ && !session_->compilerConfigurationRecorded.load(std::memory_order_acquire);
    }
    Context context() const;

private:
    void begin(const char* name);
    Session* session_ = nullptr;
    std::shared_ptr<Session> rootSession_;
    std::shared_ptr<Work> work_;
    bool ownsWork_ = false;
    Event event_;
    Scope* previous_ = nullptr;
};

Context captureContext();
void setCompilerConfiguration(const std::string& hash, const std::string& triple,
                              const std::string& cpu, const std::vector<std::string>& features);

} // namespace nacelle::profile

#define NACELLE_PROFILE_JOIN_IMPL(a, b) a##b
#define NACELLE_PROFILE_JOIN(a, b) NACELLE_PROFILE_JOIN_IMPL(a, b)
#define NACELLE_PROFILE_SCOPE(name) \
    ::nacelle::profile::Scope NACELLE_PROFILE_JOIN(nacelleProfileScope, __LINE__)(name)
#define NACELLE_PROFILE_CAPTURE_CONTEXT \
    , nacelleProfileContext = ::nacelle::profile::captureContext()
#define NACELLE_PROFILE_ASYNC_SCOPE(name) \
    ::nacelle::profile::Scope NACELLE_PROFILE_JOIN(nacelleProfileScope, __LINE__)(nacelleProfileContext, name, 2)
#else
#define NACELLE_PROFILE_SCOPE(name)
#define NACELLE_PROFILE_CAPTURE_CONTEXT
#define NACELLE_PROFILE_ASYNC_SCOPE(name)
#endif
