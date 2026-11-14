#pragma once

#include <cstdint>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <sys/resource.h>
#else
#include <fstream>
#include <string>
#include <sys/resource.h>
#endif

namespace nacelle_bench {

struct MemorySnapshot {
    double currentRssMb = 0.0;
    double peakRssMb = 0.0;
};

inline MemorySnapshot queryProcessMemory() {
    MemorySnapshot snap{};
    constexpr double kBytesPerMb = 1024.0 * 1024.0;

#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS_EX pmc{};
    pmc.cb = sizeof(pmc);
    if (GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof(pmc))) {
        snap.currentRssMb = static_cast<double>(pmc.WorkingSetSize) / kBytesPerMb;
        snap.peakRssMb = static_cast<double>(pmc.PeakWorkingSetSize) / kBytesPerMb;
    }
#elif defined(__APPLE__)
    mach_task_basic_info_data_t info{};
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS) {
        snap.currentRssMb = static_cast<double>(info.resident_size) / kBytesPerMb;
        snap.peakRssMb = static_cast<double>(info.resident_size_max) / kBytesPerMb;
    }
    struct rusage ru{};
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
        // On macOS ru_maxrss is in bytes
        double ruPeakMb = static_cast<double>(ru.ru_maxrss) / kBytesPerMb;
        if (ruPeakMb > snap.peakRssMb) snap.peakRssMb = ruPeakMb;
    }
#else
    std::ifstream ifs("/proc/self/status");
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            uint64_t kb = 0;
            if (sscanf(line.c_str() + 6, "%llu", reinterpret_cast<unsigned long long*>(&kb)) == 1) {
                snap.currentRssMb = static_cast<double>(kb) / 1024.0;
            }
        } else if (line.rfind("VmHWM:", 0) == 0) {
            uint64_t kb = 0;
            if (sscanf(line.c_str() + 6, "%llu", reinterpret_cast<unsigned long long*>(&kb)) == 1) {
                snap.peakRssMb = static_cast<double>(kb) / 1024.0;
            }
        }
    }
    if (snap.peakRssMb <= 0.0) {
        struct rusage ru{};
        if (getrusage(RUSAGE_SELF, &ru) == 0) {
            snap.peakRssMb = static_cast<double>(ru.ru_maxrss) / 1024.0;
        }
    }
#endif

    return snap;
}

} // namespace nacelle_bench
