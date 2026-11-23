#include "replay_format.h"
#include "archive.h"
#include "artifacts.h"
#include <glaze/glaze.hpp>
#include <algorithm>
#include <bit>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>

namespace nacelle::replay {
std::string contentDigest(std::span<const char> bytes) {
    constexpr std::array<uint32_t, 64> constants{
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    std::array<uint32_t, 8> hash{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,
                                 0x9b05688c,0x1f83d9ab,0x5be0cd19};
    const uint64_t blocks = (bytes.size() + 9 + 63) / 64;
    for (uint64_t block = 0; block < blocks; ++block) {
        std::array<uint32_t, 64> words{};
        for (uint64_t index = 0; index < 64; ++index) {
            const uint64_t position = block * 64 + index;
            uint8_t value = 0;
            if (position < bytes.size()) value = static_cast<uint8_t>(bytes[position]);
            else if (position == bytes.size()) value = 0x80;
            else if (position >= blocks * 64 - 8)
                value = static_cast<uint8_t>((uint64_t(bytes.size()) * 8) >> ((blocks * 64 - 1 - position) * 8));
            words[index / 4] |= uint32_t(value) << (24 - (index % 4) * 8);
        }
        for (size_t index = 16; index < 64; ++index) {
            const auto a = words[index - 15], b = words[index - 2];
            words[index] = words[index - 16] + (std::rotr(a, 7) ^ std::rotr(a, 18) ^ (a >> 3)) +
                words[index - 7] + (std::rotr(b, 17) ^ std::rotr(b, 19) ^ (b >> 10));
        }
        auto work = hash;
        for (size_t index = 0; index < 64; ++index) {
            const uint32_t first = work[7] + (std::rotr(work[4], 6) ^ std::rotr(work[4], 11) ^ std::rotr(work[4], 25)) +
                ((work[4] & work[5]) ^ (~work[4] & work[6])) + constants[index] + words[index];
            const uint32_t second = (std::rotr(work[0], 2) ^ std::rotr(work[0], 13) ^ std::rotr(work[0], 22)) +
                ((work[0] & work[1]) ^ (work[0] & work[2]) ^ (work[1] & work[2]));
            for (size_t index = 7; index > 0; --index) work[index] = work[index - 1];
            work[4] += first;
            work[0] = first + second;
        }
        for (size_t index = 0; index < 8; ++index) hash[index] += work[index];
    }
    std::ostringstream result;
    result << std::hex << std::setfill('0');
    for (auto value : hash) result << std::setw(8) << value;
    return result.str();
}

std::span<const char> blobBytes(const Bundle& bundle, uint32_t id) {
    const auto& blob = bundle.metadata.blobs.at(id - 1);
    return {bundle.bytes.data() + blob.offset, static_cast<size_t>(blob.size)};
}

bool validateBundle(const Bundle& bundle, std::string& error) {
    const auto fail = [&](const char* reason) { error = reason; return false; };
    const auto& manifest = bundle.manifest;
    const auto& data = bundle.metadata;
    const auto& environment = manifest.environment;
    if (manifest.schemaVersion != 1 || manifest.kind != "compute-replay" ||
        manifest.capability != "single-queue-owned-buffers" ||
        manifest.guarantee != "same-environment-complete-race-free-inputs" ||
        !manifest.complete || !manifest.snapshotConsent || !manifest.shaderConsent ||
        !manifest.raceFree || !manifest.hostQuiescent ||
        !manifest.snapshotBytes || manifest.snapshotBytes > maxSnapshotBytes ||
        !manifest.shaderBytes || manifest.shaderBytes > maxShaderBytes ||
        !manifest.commandBytes || manifest.commandBytes > maxCommandBytes ||
        environment.endian != "little" || environment.fpPolicy != "flush-denormals" ||
        (environment.pointerBits != 32 && environment.pointerBits != 64) ||
        (environment.simdWidth != 4 && environment.simdWidth != 8 && environment.simdWidth != 16) ||
        environment.build.compiler.empty() || environment.build.revision.empty() ||
        environment.build.operatingSystem.empty() || environment.build.architecture.empty() ||
        environment.build.compilerHash.empty() || environment.build.cpu.empty() ||
        environment.build.targetTriple.empty() || environment.compilerOptions.size() > 64 ||
        environment.build.cpuFeatures.size() > 256 ||
        data.resources.empty() || data.resources.size() > maxObjects || data.roots.size() > maxObjects ||
        data.pipelines.size() > maxObjects || data.heaps.size() > maxObjects ||
        data.views.size() > maxCommands || data.commands.empty() || data.commands.size() > maxCommands ||
        data.blobs.size() > 4 * maxObjects || bundle.bytes.size() > profile::maxProfileBytes)
        return fail("Unsupported replay capability, consent, environment or limits");
    uint64_t snapshots = 0, shaders = 0, offset = 0;
    for (size_t index = 0; index < data.blobs.size(); ++index) {
        const auto& blob = data.blobs[index];
        if (blob.id != index + 1 || !blob.size || blob.offset != offset ||
            blob.offset > bundle.bytes.size() || blob.size > bundle.bytes.size() - blob.offset ||
            blob.digest != contentDigest(blobBytes(bundle, blob.id)))
            return fail("Replay blob identity, bounds or checksum mismatch");
        offset += blob.size;
        if (blob.role == "initial" || blob.role == "expected") snapshots += blob.size;
        else if (blob.role == "dxil" || blob.role == "root-signature") shaders += blob.size;
        else return fail("Unsupported replay blob role");
    }
    if (offset != bundle.bytes.size() || snapshots > manifest.snapshotBytes || shaders > manifest.shaderBytes)
        return fail("Replay blob payload exceeds declared budgets");
    std::set<uint32_t> used;
    const auto blob = [&](uint32_t id, const char* role, uint64_t size = 0) {
        if (!id || id > data.blobs.size()) return false;
        const auto& value = data.blobs[id - 1];
        used.insert(id);
        return value.role == role && (!size || value.size == size);
    };
    for (size_t index = 0; index < data.resources.size(); ++index) {
        const auto& resource = data.resources[index];
        if (resource.id != index + 1 || !resource.size || resource.size > manifest.snapshotBytes ||
            resource.heapType < 1 || resource.heapType > 3 || (resource.flags & ~0xcu) ||
            !resource.access || (resource.access & ~3u) ||
            !blob(resource.initialBlob, "initial", resource.size) ||
            ((resource.access & 2) ? !blob(resource.expectedBlob, "expected", resource.size) : resource.expectedBlob != 0))
            return fail("Invalid replay resource or incomplete snapshot");
    }
    for (size_t index = 0; index < data.roots.size(); ++index)
        if (data.roots[index].id != index + 1 || !blob(data.roots[index].blob, "root-signature"))
            return fail("Invalid replay root signature");
    const auto digest = [](const std::string& value) {
        return value.size() == 64 && value.find_first_not_of("0123456789abcdef") == std::string::npos;
    };
    for (size_t index = 0; index < data.pipelines.size(); ++index) {
        const auto& pipeline = data.pipelines[index];
        if (pipeline.id != index + 1 || !pipeline.root || pipeline.root > data.roots.size() ||
            !blob(pipeline.shaderBlob, "dxil") || !digest(pipeline.nativeId) ||
            pipeline.shaderId != data.blobs[pipeline.shaderBlob - 1].digest ||
            pipeline.nodeMask > 1 || pipeline.flags)
            return fail("Invalid replay compute pipeline or exact code identity");
    }
    if (used.size() != data.blobs.size()) return fail("Unused replay blobs are not permitted");
    for (size_t index = 0; index < data.heaps.size(); ++index) {
        const auto& heap = data.heaps[index];
        if (heap.id != index + 1 || !heap.count || heap.count > maxCommands || heap.flags > 1)
            return fail("Invalid replay descriptor heap");
    }
    std::set<std::pair<uint32_t, uint32_t>> slots;
    for (const auto& view : data.views) {
        if (!view.heap || view.heap > data.heaps.size() || view.slot >= data.heaps[view.heap - 1].count ||
            !view.resource || view.resource > data.resources.size() || !slots.emplace(view.heap, view.slot).second)
            return fail("Invalid or duplicate replay descriptor slot");
        const auto& resource = data.resources[view.resource - 1];
        if (view.kind == "cbv") {
            if (!view.bytes || view.bytes > resource.size || view.first > resource.size - view.bytes ||
                view.count || view.stride || view.flags || view.format)
                return fail("Invalid replay CBV range");
        } else if (view.kind == "srv" || view.kind == "uav") {
            if (!view.count || !view.stride || view.bytes ||
                uint64_t(view.count) * view.stride > resource.size ||
                view.first > (resource.size - uint64_t(view.count) * view.stride) / view.stride ||
                view.flags > 1 || view.format > 191 ||
                (view.kind == "uav" && !(resource.access & 2)))
                return fail("Invalid replay buffer view range/access");
        } else return fail("Unsupported replay descriptor kind");
    }
    bool submitting = false;
    uint64_t pipeline = 0, root = 0, heap = 0;
    uint64_t dispatches = 0;
    for (const auto& command : data.commands) {
        const auto& words = command.data;
        if (words.size() > maxCommands ||
            std::any_of(words.begin(), words.end(), [](auto value) { return value > (1ull << 53) - 1; }))
            return fail("Replay command exceeds exact integer or payload limits");
        const auto resourceRange = [&](size_t index, uint64_t size, uint32_t access) {
            return words[index] && words[index] <= data.resources.size() &&
                (data.resources[words[index] - 1].access & access) == access &&
                size <= data.resources[words[index] - 1].size &&
                words[index + 1] <= data.resources[words[index] - 1].size - size;
        };
        if (command.kind == "submission") {
            if (!words.empty()) return fail("Invalid replay submission");
            submitting = true;
            pipeline = root = heap = 0;
        } else if (!submitting) return fail("Replay command outside a submission");
        else if (command.kind == "pipeline") {
            if (words.size() != 1 || !words[0] || words[0] > data.pipelines.size()) return fail("Invalid replay pipeline binding");
            pipeline = words[0];
        } else if (command.kind == "root") {
            if (words.size() != 1 || !words[0] || words[0] > data.roots.size()) return fail("Invalid replay root binding");
            root = words[0];
        } else if (command.kind == "heap") {
            if (words.size() != 1 || !words[0] || words[0] > data.heaps.size()) return fail("Invalid replay heap binding");
            heap = words[0];
        } else if (command.kind == "table") {
            if (words.size() != 3 || words[0] >= 64 || words[1] != heap || !heap ||
                words[2] >= data.heaps[heap - 1].count)
                return fail("Invalid replay table binding");
        } else if (command.kind == "constants") {
            if (words.size() < 3 || words[0] >= 64 || words[1] > 256 || words.size() - 2 > 256 - words[1] ||
                std::any_of(words.begin() + 2, words.end(), [](auto value) { return value > UINT32_MAX; }))
                return fail("Invalid replay root constants");
        } else if (command.kind == "view") {
            if (words.size() != 3 || words[0] >= 64 || !resourceRange(1, 1, 0)) return fail("Invalid replay root view");
        } else if (command.kind == "dispatch") {
            if (words.size() != 3 || !pipeline || !root || data.pipelines[pipeline - 1].root != root ||
                std::any_of(words.begin(), words.end(), [](auto value) { return !value || value > 65535; }))
                return fail("Invalid replay dispatch or pipeline/root mismatch");
            ++dispatches;
        } else if (command.kind == "copy") {
            if (words.size() != 5 || !resourceRange(0, words[4], 2) || !resourceRange(2, words[4], 1))
                return fail("Invalid replay buffer copy");
        } else if (command.kind == "clear") {
            if (words.size() != 8 || !words[0] || words[0] > data.heaps.size() || !words[2] ||
                words[2] > data.resources.size() || !(data.resources[words[2] - 1].access & 2) ||
                words[1] > UINT32_MAX || words[3] > 1 ||
                !slots.contains({static_cast<uint32_t>(words[0]), static_cast<uint32_t>(words[1])}) ||
                std::any_of(words.begin() + 3, words.end(), [](auto value) { return value > UINT32_MAX; }))
                return fail("Invalid replay UAV clear");
        } else if (command.kind == "barrier") {
            if (words.size() != 6 || words[0] > 1 || words[1] > data.resources.size() ||
                (words[0] == 0 && (!words[1] || (words[2] != 0 && words[2] != UINT32_MAX))) ||
                (words[0] == 1 && (words[2] || words[3] || words[4])) ||
                words[3] > UINT32_MAX || words[4] > UINT32_MAX || words[5] > 2)
                return fail("Invalid replay host-memory barrier");
        } else return fail("Unsupported replay command");
    }
    return dispatches ? true : fail("Replay transaction has no compute dispatch");
}

bool writeBundle(const std::string& path, Bundle& bundle, std::string& error) {
    if (!validateBundle(bundle, error)) return false;
    std::string metadata, manifest;
    if (const auto ec = glz::write_json(bundle.metadata, metadata)) {
        error = "Cannot serialize replay metadata: " + glz::format_error(ec, metadata);
        return false;
    }
    if (metadata.size() > bundle.manifest.commandBytes ||
        metadata.size() + bundle.bytes.size() > profile::maxProfileBytes) {
        error = "Replay metadata or decoded aggregate exceeds budget";
        return false;
    }
    bundle.manifest.entries = {
        {"replay.json", "json", metadata.size(), profile::artifactDigest(metadata)},
        {"blobs.bin", "binary", bundle.bytes.size(), profile::artifactDigest(bundle.bytes)}};
    if (const auto ec = glz::write_json(bundle.manifest, manifest)) {
        error = "Cannot serialize replay manifest: " + glz::format_error(ec, manifest);
        return false;
    }
    if (manifest.size() > profile::maxManifestBytes) {
        error = "Replay manifest exceeds 1 MiB";
        return false;
    }
    return profile::writeStoredArchive(path, {{"manifest.json", manifest}, {"replay.json", metadata},
                                              {"blobs.bin", bundle.bytes}}, error);
}

bool readBundleBytes(std::span<const char> bytes, Bundle& bundle, std::string& error) {
    if (!profile::validateStoredArchive(bytes, 3, 3, error)) return false;
    Bundle parsed;
    std::string manifest, metadata;
    if (!profile::readStoredEntry(bytes, "manifest.json", profile::maxManifestBytes, manifest, error)) return false;
    if (const auto ec = glz::read<glz::opts{.error_on_missing_keys = true}>(parsed.manifest, manifest)) {
        error = "Invalid replay manifest: " + glz::format_error(ec, manifest);
        return false;
    }
    if (parsed.manifest.entries.size() != 2 ||
        parsed.manifest.entries[0].path != "replay.json" || parsed.manifest.entries[0].encoding != "json" ||
        parsed.manifest.entries[1].path != "blobs.bin" || parsed.manifest.entries[1].encoding != "binary" ||
        !parsed.manifest.commandBytes || parsed.manifest.commandBytes > maxCommandBytes ||
        parsed.manifest.entries[0].uncompressedSize > parsed.manifest.commandBytes ||
        parsed.manifest.entries[1].uncompressedSize > profile::maxProfileBytes ||
        parsed.manifest.entries[0].uncompressedSize >
            profile::maxProfileBytes - parsed.manifest.entries[1].uncompressedSize) {
        error = "Replay entry declarations exceed format limits";
        return false;
    }
    if (!profile::readStoredEntry(bytes, "replay.json", parsed.manifest.commandBytes, metadata, error) ||
        !profile::readStoredEntry(bytes, "blobs.bin", parsed.manifest.entries[1].uncompressedSize, parsed.bytes, error))
        return false;
    if (metadata.size() != parsed.manifest.entries[0].uncompressedSize ||
        parsed.bytes.size() != parsed.manifest.entries[1].uncompressedSize ||
        parsed.manifest.entries[0].digest != profile::artifactDigest(metadata) ||
        parsed.manifest.entries[1].digest != profile::artifactDigest(parsed.bytes)) {
        error = "Replay entry size or checksum mismatch";
        return false;
    }
    if (const auto ec = glz::read<glz::opts{.error_on_missing_keys = true}>(parsed.metadata, metadata)) {
        error = "Invalid replay metadata: " + glz::format_error(ec, metadata);
        return false;
    }
    if (!validateBundle(parsed, error)) return false;
    bundle = std::move(parsed);
    return true;
}

bool readBundle(const std::string& path, Bundle& bundle, std::string& error) {
    std::ifstream file(std::filesystem::path(std::u8string(path.begin(), path.end())), std::ios::binary | std::ios::ate);
    if (!file || file.tellg() < 0 || static_cast<uint64_t>(file.tellg()) > profile::maxProfileBytes) {
        error = "Cannot read replay archive or exceeds 256 MiB";
        return false;
    }
    std::string bytes(static_cast<size_t>(file.tellg()), '\0');
    file.seekg(0);
    if (!file.read(bytes.data(), static_cast<std::streamsize>(bytes.size()))) {
        error = "Cannot read complete replay archive";
        return false;
    }
    return readBundleBytes(bytes, bundle, error);
}
}
