#pragma once

#include <cstddef>
#include <cstdint>

namespace nacelle::profile {

enum class CpuCounterStatus : uint8_t {
    Available,
    PermissionDenied,
    UnsupportedProvider,
    UnsupportedEvent,
    NotRunning,
    IoFailure,
    InvalidReading,
    ResourceUnavailable,
    UnsupportedScope
};

const char* cpuCounterStatusName(CpuCounterStatus status) noexcept;

struct CpuCounterResult {
    // Unavailable results carry only status/nativeError; counts and times are zero.
    CpuCounterStatus status = CpuCounterStatus::UnsupportedProvider;
    int32_t nativeError = 0;
    uint64_t cycles = 0;
    uint64_t instructions = 0;
    uint64_t timeEnabledNs = 0;
    uint64_t timeRunningNs = 0;
};

class LinuxCpuCounterProvider;

// A group belongs to the thread that begins it; end must run on that thread.
// Like other handle owners, concurrent access to a group requires synchronization.
class CpuCounterGroup {
public:
    CpuCounterGroup() noexcept = default;
    ~CpuCounterGroup();
    CpuCounterGroup(const CpuCounterGroup&) = delete;
    CpuCounterGroup& operator=(const CpuCounterGroup&) = delete;
    void reset() noexcept;

private:
    friend class LinuxCpuCounterProvider;
    int cyclesFd_ = -1;
    int instructionsFd_ = -1;
    uint64_t cyclesId_ = 0;
    uint64_t instructionsId_ = 0;
    int32_t ownerThread_ = 0;
};

struct CpuCounterCapabilities {
    bool implemented = false;
    // Bit 0: cycles; bit 1: retired instructions. Availability requires begin().
    uint32_t metricMask = 0;
};

class CpuCounterProvider {
public:
    virtual ~CpuCounterProvider() = default;
    virtual const char* name() const noexcept = 0;
    virtual CpuCounterCapabilities capabilities() const noexcept = 0;
    virtual CpuCounterResult begin(CpuCounterGroup& group) const noexcept = 0;
    virtual CpuCounterResult end(CpuCounterGroup& group) const noexcept = 0;
};

const CpuCounterProvider& cpuCounterProvider() noexcept;

namespace detail {
// Native grouped-read words: count, enabled, running, (value, id) pairs.
// Raw counts are not scaled; running/enabled reports any multiplexing.
CpuCounterResult validateCpuCounterGroupRead(const uint64_t* words, size_t bytes,
                                            uint64_t cyclesId,
                                            uint64_t instructionsId) noexcept;
}

}
