#include "instrumentation/cpu_counters.h"

#include <cstdio>
#include <cstring>
#include <type_traits>

using namespace nacelle::profile;

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "CHECK failed at line %d: %s\n", __LINE__, #condition); \
        return false; \
    } \
} while (false)

namespace {
bool validateReadings() {
    uint64_t words[] = {2, 100, 100, 1234, 11, 5678, 22};
    auto result = detail::validateCpuCounterGroupRead(words, sizeof(words), 11, 22);
    CHECK(result.status == CpuCounterStatus::Available);
    CHECK(result.cycles == 1234 && result.instructions == 5678 && result.nativeError == 0);
    CHECK(result.timeEnabledNs == 100 && result.timeRunningNs == 100);
    words[3] = 5678; words[4] = 22; words[5] = 1234; words[6] = 11;
    words[2] = 25;
    result = detail::validateCpuCounterGroupRead(words, sizeof(words), 11, 22);
    CHECK(result.status == CpuCounterStatus::Available);
    CHECK(result.cycles == 1234 && result.instructions == 5678);
    CHECK(result.timeEnabledNs == 100 && result.timeRunningNs == 25);
    words[2] = 0;
    result = detail::validateCpuCounterGroupRead(words, sizeof(words), 11, 22);
    CHECK(result.status == CpuCounterStatus::NotRunning);
    CHECK(result.cycles == 0 && result.instructions == 0 &&
          result.timeEnabledNs == 0 && result.timeRunningNs == 0);
    words[1] = 0;
    CHECK(detail::validateCpuCounterGroupRead(words, sizeof(words), 11, 22).status ==
          CpuCounterStatus::NotRunning);
    words[2] = 1;
    CHECK(detail::validateCpuCounterGroupRead(words, sizeof(words), 11, 22).status ==
          CpuCounterStatus::InvalidReading);
    words[1] = 100;
    words[0] = 1;
    CHECK(detail::validateCpuCounterGroupRead(words, sizeof(words), 11, 22).status ==
          CpuCounterStatus::InvalidReading);
    words[0] = 2;
    words[6] = 22;
    CHECK(detail::validateCpuCounterGroupRead(words, sizeof(words), 11, 22).status ==
          CpuCounterStatus::InvalidReading);
    words[6] = 99;
    CHECK(detail::validateCpuCounterGroupRead(words, sizeof(words), 11, 22).status ==
          CpuCounterStatus::InvalidReading);
    words[6] = 11;
    CHECK(detail::validateCpuCounterGroupRead(words, sizeof(words), 11, 11).status ==
          CpuCounterStatus::InvalidReading);
    const size_t invalidSizes[] = {0, sizeof(words) - 1, sizeof(words) + 1};
    for (size_t bytes : invalidSizes) {
        CHECK(detail::validateCpuCounterGroupRead(words, bytes, 11, 22).status ==
              CpuCounterStatus::InvalidReading);
    }
    CHECK(detail::validateCpuCounterGroupRead(nullptr, sizeof(words), 11, 22).status ==
          CpuCounterStatus::InvalidReading);
    return true;
}

void printResult(const char* stage, const CpuCounterResult& result) {
    std::printf("provider=%s stage=%s status=%s nativeError=%d cycles=%llu instructions=%llu "
                "timeEnabledNs=%llu timeRunningNs=%llu\n",
                cpuCounterProvider().name(), stage, cpuCounterStatusName(result.status),
                result.nativeError, static_cast<unsigned long long>(result.cycles),
                static_cast<unsigned long long>(result.instructions),
                static_cast<unsigned long long>(result.timeEnabledNs),
                static_cast<unsigned long long>(result.timeRunningNs));
}

bool expectedUnavailable(CpuCounterStatus status) {
    return status == CpuCounterStatus::PermissionDenied ||
           status == CpuCounterStatus::UnsupportedProvider ||
           status == CpuCounterStatus::UnsupportedEvent ||
           status == CpuCounterStatus::ResourceUnavailable ||
           status == CpuCounterStatus::NotRunning;
}

void printSkip(const CpuCounterResult& result) {
    std::printf("SKIP: real-provider success NOT certified: %s (nativeError=%d).\n",
                cpuCounterStatusName(result.status), result.nativeError);
}

bool probeProvider(bool& realSucceeded) {
    realSucceeded = false;
    const auto& provider = cpuCounterProvider();
    CpuCounterGroup group;
    group.reset();
    group.reset();
    auto result = provider.end(group);
    CHECK(result.status == CpuCounterStatus::NotRunning ||
          result.status == CpuCounterStatus::UnsupportedProvider);
    result = provider.begin(group);
    printResult("begin", result);
    if (result.status != CpuCounterStatus::Available) {
        CHECK(expectedUnavailable(result.status));
        CHECK(result.cycles == 0 && result.instructions == 0 &&
              result.timeEnabledNs == 0 && result.timeRunningNs == 0);
        printSkip(result);
        result = provider.end(group);
        CHECK(result.status == CpuCounterStatus::NotRunning ||
              result.status == CpuCounterStatus::UnsupportedProvider);
        return true;
    }
    volatile uint64_t arithmetic = 1;
    for (uint64_t i = 0; i < 1000000; ++i)
        arithmetic = arithmetic * 6364136223846793005ULL + i;
    result = provider.end(group);
    printResult("end", result);
    if (expectedUnavailable(result.status)) {
        CHECK(result.timeEnabledNs == 0 && result.timeRunningNs == 0 &&
              result.cycles == 0 && result.instructions == 0);
        printSkip(result);
        return true;
    }
    CHECK(result.status == CpuCounterStatus::Available);
    CHECK(result.cycles > 0 && result.instructions > 0 && result.nativeError == 0);
    CHECK(result.timeRunningNs > 0 && result.timeRunningNs <= result.timeEnabledNs);
    CHECK(provider.end(group).status == CpuCounterStatus::NotRunning);
    realSucceeded = true;
    result = provider.begin(group);
    CHECK(result.status == CpuCounterStatus::Available || expectedUnavailable(result.status));
    group.reset();
    CHECK(provider.end(group).status == CpuCounterStatus::NotRunning);
    std::puts("Real per-thread user-mode counter probe passed.");
    return true;
}
}

int main(int argc, char** argv) {
    static_assert(sizeof(CpuCounterResult) == 40);
    static_assert(!std::is_copy_constructible_v<CpuCounterGroup>);
    static_assert(!std::is_move_constructible_v<CpuCounterGroup>);
    const bool requireReal = argc == 2 && std::strcmp(argv[1], "--require-real") == 0;
    if (argc != 1 && !requireReal) {
        std::fprintf(stderr, "Usage: %s [--require-real]\n", argv[0]);
        return 1;
    }
    bool realSucceeded = false;
    if (!validateReadings() || !probeProvider(realSucceeded)) return 1;
    if (requireReal && !realSucceeded) return 77;
    std::puts("CPU counter provider tests passed.");
    return 0;
}
