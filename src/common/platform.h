#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>

#if !defined(_WIN32)
#include <sys/mman.h>
#include <unistd.h>
#include <wsl/winadapter.h>
#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif
#else
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <unknwn.h>
#endif

#include <directx/d3d12.h>
#include <directx/d3d12sdklayers.h>
#include <directx/d3dx12.h>
#include <directx/dxcore_interface.h>
#include <dxguids/dxguids.h>

#ifdef interface
#undef interface
#endif

#if !defined(_WIN32) && defined(__CRT_UUID_DECL) && !defined(NACELLE_ID3D10BLOB_UUID_DEFINED)
#define NACELLE_ID3D10BLOB_UUID_DEFINED
__CRT_UUID_DECL(ID3D10Blob, 0x8ba5fb08, 0x5195, 0x40e2, 0xac, 0x58, 0x0d, 0x98, 0x9c, 0x3a, 0x01, 0x02)
#endif

namespace nacelle {

inline void* alignedAlloc64(size_t size) {
    if (size == 0) size = 64;
    const size_t aligned = (size + 63) & ~size_t(63);
#if defined(_WIN32)
    return _aligned_malloc(aligned, 64);
#else
    return std::aligned_alloc(64, aligned);
#endif
}

inline void alignedFree(void* ptr) {
    if (!ptr) return;
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

inline constexpr size_t kDemandPagedThresholdBytes = 64 * 1024;

/// Allocates 64-byte aligned zero-initialized memory. Large allocations use
/// OS demand-zeroed virtual memory so untouched pages consume no physical RSS.
inline void* alignedAllocZeroed64(size_t size) {
    if (size == 0) size = 64;
    const size_t aligned = (size + 63) & ~size_t(63);
    if (aligned >= kDemandPagedThresholdBytes) {
#if defined(_WIN32)
        if (void* p = VirtualAlloc(nullptr, aligned, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)) {
            return p;
        }
#else
        void* p = mmap(nullptr, aligned, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p != MAP_FAILED) {
            return p;
        }
#endif
    }
    void* p = alignedAlloc64(aligned);
    if (p) {
        std::memset(p, 0, aligned);
    }
    return p;
}

inline void alignedFreeZeroed(void* ptr, size_t size) {
    if (!ptr) return;
    if (size == 0) size = 64;
    const size_t aligned = (size + 63) & ~size_t(63);
    if (aligned >= kDemandPagedThresholdBytes) {
#if defined(_WIN32)
        VirtualFree(ptr, 0, MEM_RELEASE);
        return;
#else
        munmap(ptr, aligned);
        return;
#endif
    }
    alignedFree(ptr);
}

inline std::optional<std::string> getEnv(const char* name) {
    if (!name) return std::nullopt;
#if defined(_WIN32)
    char* buf = nullptr;
    size_t len = 0;
    if (_dupenv_s(&buf, &len, name) != 0 || !buf) {
        return std::nullopt;
    }
    std::string result(buf);
    std::free(buf);
    return result;
#else
    const char* val = std::getenv(name);
    if (!val) return std::nullopt;
    return std::string(val);
#endif
}

inline std::filesystem::path getCacheDirectory() {
    auto tryUse = [](const std::filesystem::path& p) -> bool {
        std::error_code ec;
        std::filesystem::create_directories(p, ec);
        return !ec;
    };
    if (auto overrideDir = getEnv("NACELLE_CACHE_DIR"); overrideDir && !overrideDir->empty()) {
        std::filesystem::path p(*overrideDir);
        if (tryUse(p)) return p;
    }
    std::filesystem::path candidate;
#if defined(_WIN32)
    if (auto localAppData = getEnv("LOCALAPPDATA"); localAppData && !localAppData->empty()) {
        candidate = std::filesystem::path(*localAppData) / "nacelle";
    }
#else
    if (auto xdg = getEnv("XDG_CACHE_HOME"); xdg && !xdg->empty()) {
        candidate = std::filesystem::path(*xdg) / "nacelle";
    } else if (auto home = getEnv("HOME"); home && !home->empty()) {
        candidate = std::filesystem::path(*home) / ".cache" / "nacelle";
    }
#endif
    if (!candidate.empty() && tryUse(candidate)) {
        return candidate;
    }
    std::filesystem::path fallback = std::filesystem::temp_directory_path() / "nacelle";
    std::error_code ec;
    std::filesystem::create_directories(fallback, ec);
    return fallback;
}

inline std::string wideToUtf8(const wchar_t* wstr) {
    if (!wstr) return {};
    std::string out;
    while (*wstr) {
        uint32_t cp = static_cast<uint32_t>(*wstr++);
        if (cp < 0x80) {
            out.push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }
    return out;
}

inline void signalOsEvent(HANDLE hEvent) {
    if (!hEvent) return;
#if defined(_WIN32)
    ::SetEvent(hEvent);
#else
    int fd = static_cast<int>(reinterpret_cast<intptr_t>(hEvent));
    if (fd >= 0) {
        uint64_t one = 1;
        ssize_t written = ::write(fd, &one, sizeof(one));
        (void)written;
    }
#endif
}

} // namespace nacelle
