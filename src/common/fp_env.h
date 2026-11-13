#pragma once

#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <xmmintrin.h>
#endif

namespace nacelle {

/// Scoped RAII guard enabling hardware flush-to-zero (FTZ) and denormals-are-zero (DAZ)
/// to match D3D12 / HLSL Shader Model 6.x default single-precision denormal behavior.
class ScopedFlushDenormals {
public:
    ScopedFlushDenormals() noexcept {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
        savedState_ = _mm_getcsr();
        // Bit 15: FTZ (0x8000), Bit 6: DAZ (0x0040)
        _mm_setcsr(savedState_ | 0x8040u);
#elif defined(__aarch64__) && (defined(__clang__) || defined(__GNUC__))
        __asm__ volatile("mrs %0, fpcr" : "=r"(savedState_));
        // Bit 24: FZ (flush-to-zero for inputs and outputs on AArch64)
        uint64_t next = savedState_ | (uint64_t(1) << 24);
        __asm__ volatile("msr fpcr, %0" : : "r"(next));
#endif
    }

    ~ScopedFlushDenormals() noexcept {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
        _mm_setcsr(savedState_);
#elif defined(__aarch64__) && (defined(__clang__) || defined(__GNUC__))
        __asm__ volatile("msr fpcr, %0" : : "r"(savedState_));
#endif
    }

    ScopedFlushDenormals(const ScopedFlushDenormals&) = delete;
    ScopedFlushDenormals& operator=(const ScopedFlushDenormals&) = delete;

private:
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    unsigned int savedState_ = 0;
#elif defined(__aarch64__) && (defined(__clang__) || defined(__GNUC__))
    uint64_t savedState_ = 0;
#endif
};

} // namespace nacelle
