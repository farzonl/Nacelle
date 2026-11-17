#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

namespace nacelle_bench {

struct GDeflateStreamData {
    std::vector<uint8_t> inputBytes;
    std::vector<uint32_t> controlWords;
    uint32_t numTiles = 0;
    uint32_t expectedOutBytes = 0;
    double expectedChecksum = 0.0;
};

namespace detail {

inline void appendTileBitsMSB(
    std::vector<uint32_t>& words,
    uint32_t& curWord,
    uint32_t& bitPos,
    uint32_t codeMSB,
    uint32_t numBits) {
    for (int i = static_cast<int>(numBits) - 1; i >= 0; --i) {
        uint32_t bit = (codeMSB >> i) & 1u;
        curWord |= (bit << bitPos);
        ++bitPos;
        if (bitPos == 32u) {
            words.push_back(curWord);
            curWord = 0u;
            bitPos = 0u;
        }
    }
}

} // namespace detail

inline GDeflateStreamData buildGDeflateStream(uint32_t numTiles) {
    constexpr uint32_t kNumLanes = 32u;
    constexpr uint32_t kRounds = 2048u;
    constexpr uint32_t kTileBytes = kRounds * kNumLanes;

    GDeflateStreamData data;
    data.numTiles = numTiles;
    data.expectedOutBytes = numTiles * kTileBytes;
    data.controlWords = {1u, 0u, 0u};

    std::vector<std::vector<uint32_t>> tileWords(numTiles);
    std::vector<uint8_t> expectedAll;
    expectedAll.reserve(data.expectedOutBytes);

    for (uint32_t t = 0; t < numTiles; ++t) {
        std::vector<uint32_t> laneWords[kNumLanes];
        uint32_t curWord[kNumLanes]{};
        uint32_t bitPos[kNumLanes]{};
        curWord[0] = 0x3u;
        bitPos[0] = 3u;

        for (uint32_t r = 0; r < kRounds; ++r) {
            for (uint32_t tid = 0; tid < kNumLanes; ++tid) {
                uint8_t val = static_cast<uint8_t>(((r * 37u + tid * 13u + t * 7u) % 143u) + 1u);
                expectedAll.push_back(val);
                detail::appendTileBitsMSB(laneWords[tid], curWord[tid], bitPos[tid], 0x30u + val, 8u);
            }
        }
        detail::appendTileBitsMSB(laneWords[0], curWord[0], bitPos[0], 0u, 7u);
        for (uint32_t tid = 0; tid < kNumLanes; ++tid) {
            if (bitPos[tid] > 0u) {
                laneWords[tid].push_back(curWord[tid]);
            }
            laneWords[tid].push_back(0u);
        }

        uint32_t nextWordIdx[kNumLanes]{};
        uint32_t cnt[kNumLanes]{};
        for (uint32_t tid = 0; tid < kNumLanes; ++tid) {
            tileWords[t].push_back(laneWords[tid][nextWordIdx[tid]++]);
            cnt[tid] = 32u;
        }

        auto simWaveEat = [&](auto&& eatPred, auto&& eatBits) {
            for (uint32_t l = 0; l < kNumLanes; ++l) {
                if (eatPred(l)) {
                    cnt[l] -= eatBits(l);
                }
            }
            for (uint32_t l = 0; l < kNumLanes; ++l) {
                if (eatPred(l) && cnt[l] < 32u) {
                    uint32_t w = (nextWordIdx[l] < laneWords[l].size()) ? laneWords[l][nextWordIdx[l]++] : 0u;
                    tileWords[t].push_back(w);
                    cnt[l] += 32u;
                }
            }
        };

        // 3-bit block header on lane 0
        simWaveEat([](uint32_t l) { return l == 0u; }, [](uint32_t) { return 3u; });
        // 2048 literal rounds across all 32 lanes
        for (uint32_t r = 0; r < kRounds; ++r) {
            simWaveEat([](uint32_t) { return true; }, [](uint32_t) { return 8u; });
        }
        // Final EOB symbol on lane 0 (7 bits)
        simWaveEat([](uint32_t l) { return l == 0u; }, [](uint32_t) { return 7u; });
    }

    // Assemble TileStream: [word1, word2, tileTable[numTiles], tile0Words, tile1Words, ...]
    const uint32_t headerBytes = 8u + numTiles * 4u;
    uint32_t totalDataBytes = 0u;
    std::vector<uint32_t> tileTable(numTiles, 0u);
    for (uint32_t t = 0; t < numTiles; ++t) {
        const uint32_t tileBytes = static_cast<uint32_t>(tileWords[t].size() * 4u);
        if (t > 0u) {
            tileTable[t] = totalDataBytes;
        }
        if (t + 1u == numTiles) {
            tileTable[0] = tileBytes; // TileStream convention: tileTable[0] holds last tile's byte size
        }
        totalDataBytes += tileBytes;
    }

    data.inputBytes.resize(headerBytes + totalDataBytes);
    const uint32_t word1 = (numTiles << 16) | (0x04u << 8);
    const uint32_t word2 = 0u; // lastTileSize = 0 => 64 KiB
    std::memcpy(data.inputBytes.data() + 0, &word1, 4);
    std::memcpy(data.inputBytes.data() + 4, &word2, 4);
    std::memcpy(data.inputBytes.data() + 8, tileTable.data(), numTiles * 4u);

    size_t writeOff = headerBytes;
    for (uint32_t t = 0; t < numTiles; ++t) {
        const size_t bytes = tileWords[t].size() * 4u;
        std::memcpy(data.inputBytes.data() + writeOff, tileWords[t].data(), bytes);
        writeOff += bytes;
    }

    // Expected checksum: sum of (uint32_t_word & 0xFFFFu) over first 4096 dwords of output
    double sum = 0.0;
    const size_t checkWords = std::min<size_t>(4096u, expectedAll.size() / 4u);
    for (size_t i = 0; i < checkWords; ++i) {
        uint32_t w = 0;
        std::memcpy(&w, expectedAll.data() + i * 4, 4);
        sum += static_cast<double>(w & 0xFFFFu);
    }
    data.expectedChecksum = sum;
    return data;
}

} // namespace nacelle_bench
