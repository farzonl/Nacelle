#pragma once
#if defined(NACELLE_ENABLE_INSTRUMENTATION)
#include <nacelle/instrumentation.h>
#include "profile_format.h"
#include <atomic>
#include <memory>
#include <mutex>

namespace nacelle::profile {
inline constexpr size_t maxSpatialRecords = 131072;
inline constexpr size_t maxSpatialOperations = 4096;

struct SpatialRecord {
    uint64_t operation = 0, wallNs = 0, invocations = 0;
    std::array<uint32_t, 3> xyz{};
    std::array<uint64_t, 16> covered{}, companion{}, padding{};
};
static_assert(sizeof(SpatialRecord) == 424);

struct SpatialCounters {
    std::atomic<uint64_t> recorded{0}, skipped{0}, dropped{0};
};

struct SpatialState {
    explicit SpatialState(const NacelleProfileSpatialOptions& options);
    uint64_t addOperation(SpatialOperation operation);
    SpatialRecord* reserve(uint64_t operation, std::array<uint32_t, 3> xyz);
    bool complete() const;
    bool aggregate(SpatialMetadata& result, std::string& error) const;
    const NacelleProfileSpatialOptions options;
    const size_t capacity;
    std::unique_ptr<SpatialRecord[]> records;
    std::unique_ptr<SpatialCounters[]> counters;
    std::atomic<uint64_t> next{0};
    std::atomic<bool> operationOverflow{false};
    std::mutex operationMutex;
    std::vector<SpatialOperation> operations;
};

bool validateSpatial(const SpatialMetadata& metadata, const ShaderMetadata& shaders,
                     const CaptureOptions& options, std::string& error);
bool validateSpatialReferences(const SpatialMetadata& metadata, const Trace& trace,
                               bool eventsComplete, std::string& error);
}
#endif
