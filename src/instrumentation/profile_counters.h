#pragma once

#include "cpu_counters.h"
#include "profile_format.h"
#include <atomic>
#include <memory>

namespace nacelle::profile {

enum class CpuCounterRegion : uint32_t { ComputeWorker, PixelWorker, ComputeDirect };

struct CounterRecord {
    uint64_t eventId = 0, codeVersionId = 0;
    CpuCounterResult result;
    CpuCounterRegion region = CpuCounterRegion::ComputeWorker;
    uint32_t reserved = 0;
};
static_assert(sizeof(CounterRecord) == 64);
inline constexpr uint64_t maxCounterBytes = 16ull * 1024 * 1024;

class CounterState {
public:
    explicit CounterState(uint64_t bytes, const CpuCounterProvider& provider = cpuCounterProvider());
    CounterRecord* reserve(uint64_t event, uint64_t version, CpuCounterRegion region);
    bool complete() const;
    const char* failureReason() const;
    CounterMetadata metadata() const;
    const CpuCounterProvider& provider;
private:
    uint64_t bytes_;
    size_t capacity_;
    std::unique_ptr<CounterRecord[]> records_;
    std::atomic<uint64_t> attempted_{0};
};

bool validateCounters(const CounterMetadata& counters, const ShaderMetadata& shaders,
                      const CaptureOptions& options, std::string& error);
bool validateCounterReferences(const CounterMetadata& counters, const Trace& trace,
                               bool eventsComplete, std::string& error);
}
