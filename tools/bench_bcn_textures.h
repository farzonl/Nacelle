#pragma once

#include <cstdint>
#include <vector>

namespace nacelle_bench {

// Generates a deterministic 256x256 BC1_UNORM texture (64x64 blocks, 8 bytes/block = 32 KiB).
inline std::vector<uint8_t> buildTestBC1Texture256() {
    constexpr uint32_t kBlocksX = 64u;
    constexpr uint32_t kBlocksY = 64u;
    std::vector<uint8_t> out(kBlocksX * kBlocksY * 8u);
    for (uint32_t by = 0; by < kBlocksY; ++by) {
        for (uint32_t bx = 0; bx < kBlocksX; ++bx) {
            uint8_t* blk = out.data() + (by * kBlocksX + bx) * 8u;
            uint16_t r0 = static_cast<uint16_t>((bx * 31u / 63u) & 0x1Fu);
            uint16_t g0 = static_cast<uint16_t>((by * 63u / 63u) & 0x3Fu);
            uint16_t b0 = static_cast<uint16_t>((((bx ^ by) * 31u) / 63u) & 0x1Fu);
            uint16_t c0 = static_cast<uint16_t>((r0 << 11) | (g0 << 5) | b0);
            uint16_t c1 = static_cast<uint16_t>(((31u - r0) << 11) | ((63u - g0) << 5) | (31u - b0));
            if (c0 < c1) {
                uint16_t tmp = c0;
                c0 = c1;
                c1 = tmp;
            }
            blk[0] = static_cast<uint8_t>(c0 & 0xFFu);
            blk[1] = static_cast<uint8_t>(c0 >> 8);
            blk[2] = static_cast<uint8_t>(c1 & 0xFFu);
            blk[3] = static_cast<uint8_t>(c1 >> 8);
            uint32_t idx = 0xE4E4E4E4u ^ ((bx + by * 7u) * 0x11223344u);
            blk[4] = static_cast<uint8_t>(idx & 0xFFu);
            blk[5] = static_cast<uint8_t>((idx >> 8) & 0xFFu);
            blk[6] = static_cast<uint8_t>((idx >> 16) & 0xFFu);
            blk[7] = static_cast<uint8_t>((idx >> 24) & 0xFFu);
        }
    }
    return out;
}

// Generates a deterministic 256x256 BC7_UNORM texture using Mode 6 (64x64 blocks, 16 bytes/block = 64 KiB).
inline std::vector<uint8_t> buildTestBC7Texture256() {
    constexpr uint32_t kBlocksX = 64u;
    constexpr uint32_t kBlocksY = 64u;
    std::vector<uint8_t> out(kBlocksX * kBlocksY * 16u, 0u);
    for (uint32_t by = 0; by < kBlocksY; ++by) {
        for (uint32_t bx = 0; bx < kBlocksX; ++bx) {
            uint8_t* blk = out.data() + (by * kBlocksX + bx) * 16u;
            // Mode 6: bit 6 set (0x40), followed by 7-bit RGBA endpoints + p-bits + 4-bit indices.
            uint64_t lo = 0x40ull; // mode 6
            uint64_t r0 = (bx * 2u) & 0x7Full;
            uint64_t r1 = (127u - r0) & 0x7Full;
            uint64_t g0 = (by * 2u) & 0x7Full;
            uint64_t g1 = (127u - g0) & 0x7Full;
            uint64_t b0 = ((bx + by) & 0x7Full);
            uint64_t b1 = (127u - b0) & 0x7Full;
            uint64_t a0 = 127ull;
            uint64_t a1 = 127ull;
            lo |= (r0 << 7) | (r1 << 14) | (g0 << 21) | (g1 << 28) | (b0 << 35) | (b1 << 42) | (a0 << 49) | (a1 << 56);
            uint64_t hi = (a1 >> 8) | (1ull << 0) | (1ull << 1); // p-bits = 1
            hi |= (0x0123456789ABCDEFull << 2) ^ (static_cast<uint64_t>(bx * 17u + by * 31u) << 8);
            for (int i = 0; i < 8; ++i) {
                blk[i] = static_cast<uint8_t>((lo >> (i * 8)) & 0xFFu);
                blk[8 + i] = static_cast<uint8_t>((hi >> (i * 8)) & 0xFFu);
            }
        }
    }
    return out;
}

} // namespace nacelle_bench
