#include "cpu_counters.h"

#if defined(NACELLE_ENABLE_CPU_COUNTERS) && NACELLE_ENABLE_CPU_COUNTERS == 1 && defined(__linux__)
#include <cerrno>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace nacelle::profile {

const char* cpuCounterStatusName(CpuCounterStatus status) noexcept {
    switch (status) {
    case CpuCounterStatus::Available: return "available";
    case CpuCounterStatus::PermissionDenied: return "permission-denied";
    case CpuCounterStatus::UnsupportedProvider: return "unsupported-provider";
    case CpuCounterStatus::UnsupportedEvent: return "unsupported-event";
    case CpuCounterStatus::NotRunning: return "not-running";
    case CpuCounterStatus::IoFailure: return "io-failure";
    case CpuCounterStatus::InvalidReading: return "invalid-reading";
    case CpuCounterStatus::ResourceUnavailable: return "resource-unavailable";
    case CpuCounterStatus::UnsupportedScope: return "unsupported-scope";
    }
    return "unknown";
}

CpuCounterResult detail::validateCpuCounterGroupRead(const uint64_t* words, size_t bytes,
                                                    uint64_t cyclesId,
                                                    uint64_t instructionsId) noexcept {
    CpuCounterResult result;
    result.status = CpuCounterStatus::InvalidReading;
    if (!words || bytes != 7 * sizeof(uint64_t) || cyclesId == instructionsId)
        return result;
    if (words[0] != 2 || words[2] > words[1])
        return result;
    const bool cyclesFirst = words[4] == cyclesId && words[6] == instructionsId;
    const bool instructionsFirst = words[4] == instructionsId && words[6] == cyclesId;
    if (!cyclesFirst && !instructionsFirst)
        return result;
    if (words[2] == 0) {
        result.status = CpuCounterStatus::NotRunning;
        return result;
    }
    result.timeEnabledNs = words[1];
    result.timeRunningNs = words[2];
    result.cycles = cyclesFirst ? words[3] : words[5];
    result.instructions = cyclesFirst ? words[5] : words[3];
    result.status = CpuCounterStatus::Available;
    return result;
}

CpuCounterGroup::~CpuCounterGroup() {
    reset();
}

void CpuCounterGroup::reset() noexcept {
#if defined(NACELLE_ENABLE_CPU_COUNTERS) && NACELLE_ENABLE_CPU_COUNTERS == 1 && defined(__linux__)
    // Linux releases the descriptor even when close reports EINTR; never retry it.
    if (instructionsFd_ >= 0) ::close(instructionsFd_);
    if (cyclesFd_ >= 0) ::close(cyclesFd_);
#endif
    cyclesFd_ = instructionsFd_ = -1;
    cyclesId_ = instructionsId_ = 0;
    ownerThread_ = 0;
}

#if defined(NACELLE_ENABLE_CPU_COUNTERS) && NACELLE_ENABLE_CPU_COUNTERS == 1 && defined(__linux__)
namespace {
constexpr unsigned maxAttempts = 8;

CpuCounterResult failure(int error, bool opening = false) noexcept {
    CpuCounterResult result;
    result.nativeError = error;
    switch (error) {
    case EACCES:
    case EPERM: result.status = CpuCounterStatus::PermissionDenied; break;
    case ENOSYS: result.status = CpuCounterStatus::UnsupportedProvider; break;
    case ENODEV:
    case ENOENT:
    case EOPNOTSUPP: result.status = CpuCounterStatus::UnsupportedEvent; break;
    case EINVAL:
        result.status = opening ? CpuCounterStatus::UnsupportedEvent : CpuCounterStatus::IoFailure;
        break;
    case EBUSY:
    case EAGAIN:
    case EMFILE:
    case ENFILE:
    case ENOMEM:
    case ENOSPC: result.status = CpuCounterStatus::ResourceUnavailable; break;
    default: result.status = CpuCounterStatus::IoFailure; break;
    }
    return result;
}

int openEvent(perf_event_attr& attributes, int leader) noexcept {
    for (unsigned attempt = 0; attempt < maxAttempts; ++attempt) {
        const auto fd = ::syscall(SYS_perf_event_open, &attributes, 0, -1, leader,
                                  PERF_FLAG_FD_CLOEXEC);
        if (fd >= 0 || errno != EINTR) return static_cast<int>(fd);
    }
    return -1;
}

int eventIoctl(int fd, unsigned long request, unsigned long argument) noexcept {
    for (unsigned attempt = 0; attempt < maxAttempts; ++attempt) {
        const int result = ::ioctl(fd, request, argument);
        if (result >= 0 || errno != EINTR) return result;
    }
    return -1;
}

ssize_t readGroup(int fd, uint64_t (&words)[7]) noexcept {
    for (unsigned attempt = 0; attempt < maxAttempts; ++attempt) {
        const auto bytes = ::read(fd, words, sizeof(words));
        if (bytes >= 0 || errno != EINTR) return bytes;
    }
    return -1;
}
}

class LinuxCpuCounterProvider final : public CpuCounterProvider {
public:
    const char* name() const noexcept override { return "linux-perf-event"; }
    CpuCounterCapabilities capabilities() const noexcept override { return {true, 3}; }

    CpuCounterResult begin(CpuCounterGroup& group) const noexcept override {
        group.reset();
        perf_event_attr attributes{};
        attributes.size = sizeof(attributes);
        attributes.type = PERF_TYPE_HARDWARE;
        attributes.config = PERF_COUNT_HW_CPU_CYCLES;
        attributes.disabled = 1;
        attributes.inherit = 0;
        attributes.exclude_kernel = 1;
        attributes.exclude_hv = 1;
        attributes.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_ID |
                                 PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
        group.ownerThread_ = static_cast<int32_t>(::syscall(SYS_gettid));
        group.cyclesFd_ = openEvent(attributes, -1);
        if (group.cyclesFd_ < 0) return failAndReset(group, errno, true);
        attributes.config = PERF_COUNT_HW_INSTRUCTIONS;
        group.instructionsFd_ = openEvent(attributes, group.cyclesFd_);
        if (group.instructionsFd_ < 0) return failAndReset(group, errno, true);
        if (eventIoctl(group.cyclesFd_, PERF_EVENT_IOC_ID,
                       reinterpret_cast<unsigned long>(&group.cyclesId_)) < 0 ||
            eventIoctl(group.instructionsFd_, PERF_EVENT_IOC_ID,
                       reinterpret_cast<unsigned long>(&group.instructionsId_)) < 0)
            return failAndReset(group, errno);
        if (group.cyclesId_ == group.instructionsId_) {
            group.reset();
            CpuCounterResult result;
            result.status = CpuCounterStatus::InvalidReading;
            return result;
        }
        if (eventIoctl(group.cyclesFd_, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) < 0 ||
            eventIoctl(group.cyclesFd_, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) < 0)
            return failAndReset(group, errno);
        CpuCounterResult result;
        result.status = CpuCounterStatus::Available;
        return result;
    }

    CpuCounterResult end(CpuCounterGroup& group) const noexcept override {
        CpuCounterResult result;
        if (group.cyclesFd_ < 0 || group.instructionsFd_ < 0) {
            group.reset();
            result.status = CpuCounterStatus::NotRunning;
            return result;
        }
        if (group.ownerThread_ != static_cast<int32_t>(::syscall(SYS_gettid))) {
            group.reset();
            result.status = CpuCounterStatus::UnsupportedScope;
            return result;
        }
        if (eventIoctl(group.cyclesFd_, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) < 0)
            return failAndReset(group, errno);
        uint64_t words[7]{};
        const auto bytes = readGroup(group.cyclesFd_, words);
        if (bytes < 0) return failAndReset(group, errno);
        result = detail::validateCpuCounterGroupRead(words, static_cast<size_t>(bytes),
                                                    group.cyclesId_, group.instructionsId_);
        group.reset();
        return result;
    }

private:
    static CpuCounterResult failAndReset(CpuCounterGroup& group, int error,
                                         bool opening = false) noexcept {
        const auto result = failure(error, opening);
        group.reset();
        return result;
    }
};
#else
namespace {
class UnsupportedCpuCounterProvider final : public CpuCounterProvider {
public:
    const char* name() const noexcept override { return "unsupported"; }
    CpuCounterCapabilities capabilities() const noexcept override { return {}; }
    CpuCounterResult begin(CpuCounterGroup& group) const noexcept override {
        group.reset();
        return {};
    }
    CpuCounterResult end(CpuCounterGroup& group) const noexcept override {
        group.reset();
        return {};
    }
};
}
#endif

const CpuCounterProvider& cpuCounterProvider() noexcept {
#if defined(NACELLE_ENABLE_CPU_COUNTERS) && NACELLE_ENABLE_CPU_COUNTERS == 1 && defined(__linux__)
    static const LinuxCpuCounterProvider provider;
#else
    static const UnsupportedCpuCounterProvider provider;
#endif
    return provider;
}

}
