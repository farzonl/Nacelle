#include "capture.h"
#include "profile_format.h"

#include <glaze/glaze.hpp>
#include <miniz.h>
#include <zstd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <ranges>
#include <set>
#include <sstream>

namespace nacelle::profile {
std::string formatProfileHash(uint64_t lo, uint64_t hi) {
    if (lo == 0 && hi == 0) return {};
    std::ostringstream stream;
    stream << std::hex << std::setfill('0') << std::setw(16) << hi << std::setw(16) << lo;
    return stream.str();
}

namespace {

struct Zip {
    mz_zip_archive archive{};
    bool writing = false;
    ~Zip() {
        if (writing) mz_zip_writer_end(&archive);
        else mz_zip_reader_end(&archive);
    }
};

bool safePath(const std::string& path) {
    if (path.empty() || path.front() == '/' || path.find('\\') != std::string::npos ||
        path.find(':') != std::string::npos || path.find('\0') != std::string::npos)
        return false;
    std::istringstream parts(path);
    std::string part;
    while (std::getline(parts, part, '/')) {
        if (part.empty() || part == "." || part == "..") return false;
    }
    return path.back() != '/';
}

template <typename T>
bool toJson(const T& value, std::string& json, std::string& error) {
    if (const auto ec = glz::write_json(value, json)) {
        error = "Glaze serialization failed: " + glz::format_error(ec, json);
        return false;
    }
    return true;
}

bool manifestToJson(const Manifest& manifest, std::string& json, std::string& error) {
    if (!toJson(manifest, json, error)) return false;
    if (json.size() > maxManifestBytes) {
        error = "Profile manifest exceeds the 1 MiB format limit";
        return false;
    }
    return true;
}

bool writeExclusive(const std::string& path, const void* bytes, size_t size, std::string& error) {
    const auto nativePath = std::filesystem::path(std::u8string(path.begin(), path.end()));
    FILE* file = nullptr;
#if defined(_WIN32)
    if (_wfopen_s(&file, nativePath.c_str(), L"wbx") != 0) file = nullptr;
#else
    file = std::fopen(nativePath.c_str(), "wbx");
#endif
    if (!file) {
        error = "Cannot create profile (destination must not already exist): " + path;
        return false;
    }
    const bool written = std::fwrite(bytes, 1, size, file) == size;
    const bool closed = std::fclose(file) == 0;
    if (!written || !closed) {
        std::error_code ec;
        std::filesystem::remove(nativePath, ec);
        error = "Failed writing profile: " + path;
        if (ec) error += "; could not remove partial file: " + ec.message();
        return false;
    }
    return true;
}

bool extract(Zip& zip, const char* name, size_t limit, std::string& data, std::string& error) {
    mz_zip_archive_file_stat stat{};
    const int index = mz_zip_reader_locate_file(&zip.archive, name, nullptr, 0);
    if (index < 0 || !mz_zip_reader_file_stat(&zip.archive, static_cast<mz_uint>(index), &stat) ||
        stat.m_uncomp_size > limit) {
        error = std::string("Missing or oversized profile entry: ") + name;
        return false;
    }
    data.resize(static_cast<size_t>(stat.m_uncomp_size));
    if (!mz_zip_reader_extract_to_mem(&zip.archive, static_cast<mz_uint>(index),
                                      data.data(), data.size(), 0)) {
        error = std::string("Corrupt profile entry: ") + name;
        return false;
    }
    return true;
}
}

bool writeProfile(const std::string& path, const Session& session, std::string& error) {
    if (session.shaderMetadataOverflow) {
        error = "Profile exceeds the 65536 compute code-version metadata limit";
        return false;
    }
    Manifest manifest;
    manifest.complete = session.dropped.load() == 0;
    manifest.recordedEvents = session.eventCount;
    manifest.droppedEvents = session.dropped.load();
    manifest.captureOptions.eventBufferBytes = session.bufferBytes;
    manifest.build = {NACELLE_PROFILE_COMPILER, NACELLE_PROFILE_OS,
                      NACELLE_PROFILE_ARCH, NACELLE_PROFILE_REVISION,
                      session.compilerHash, session.targetTriple, session.cpu, session.cpuFeatures};
    manifest.entries = {{"trace.json.zst", "zstd", 0}, {"shaders.json", "json", 2}};
    std::string manifestJson;
    // Reject oversized metadata before allocating trace objects and compression buffers.
    if (!manifestToJson(manifest, manifestJson, error)) return false;
    std::string shadersJson = "[]";
    if (!session.shaders.codeVersions.empty() && !toJson(session.shaders, shadersJson, error))
        return false;
    if (shadersJson.size() > maxProfileBytes) {
        error = "Profile shader metadata exceeds the 256 MiB format limit";
        return false;
    }

    std::string traceJson;
    {
        std::vector<const Event*> ordered;
        ordered.reserve(static_cast<size_t>(session.eventCount));
        for (size_t i = 0; i < std::min(session.capacity, session.nextBlock.load()); ++i) {
            if (session.events[i].id != 0) ordered.push_back(&session.events[i]);
        }
        std::sort(ordered.begin(), ordered.end(), [](const Event* a, const Event* b) {
            return static_cast<double>(a->startNs) / 1000.0 <
                   static_cast<double>(b->startNs) / 1000.0;
        });
        // Glaze consumes one temporary event at a time instead of a second full object graph.
        const auto events = ordered | std::views::transform([](const Event* event) {
            constexpr const char* stages[] =
                {"unknown", "compute", "graphics", "vertex", "pixel", "mesh", "raytracing"};
            constexpr const char* caches[] = {"unknown", "hit", "miss"};
            TraceEvent item;
            item.name = event->name;
            item.cat = event->attributes.tier && !event->attributes.codeVersionId ? "jit" : "runtime";
            item.ts = static_cast<double>(event->startNs) / 1000.0;
            item.dur = static_cast<double>(event->durationNs) / 1000.0;
            item.tid = event->thread;
            item.args = {event->id, event->parent,
                         formatProfileHash(event->attributes.hashLo, event->attributes.hashHi),
                         stages[static_cast<unsigned>(event->attributes.stage)],
                         event->attributes.simdWidth, event->attributes.tier,
                         caches[static_cast<unsigned>(event->attributes.cache)]};
            if (event->attributes.codeVersionId) {
                item.args.pipelineId = event->attributes.pipelineId;
                item.args.codeVersionId = event->attributes.codeVersionId;
            }
            return item;
        });
        const Trace defaults;
        if (!toJson(glz::obj{"traceEvents", events, "displayTimeUnit", defaults.displayTimeUnit},
                    traceJson, error)) return false;
    }
    if (traceJson.size() > maxProfileBytes - shadersJson.size()) {
        error = "Profile trace exceeds the 256 MiB uncompressed format limit";
        return false;
    }
    std::string compressed(ZSTD_compressBound(traceJson.size()), '\0');
    const size_t result = ZSTD_compress(compressed.data(), compressed.size(),
                                      traceJson.data(), traceJson.size(), 1);
    if (ZSTD_isError(result)) {
        error = std::string("zstd compression failed: ") + ZSTD_getErrorName(result);
        return false;
    }
    compressed.resize(result);

    manifest.entries.front().uncompressedSize = traceJson.size();
    manifest.entries.back().uncompressedSize = shadersJson.size();
    if (!manifestToJson(manifest, manifestJson, error)) return false;

    Zip zip;
    zip.writing = true;
    if (!mz_zip_writer_init_heap(&zip.archive, 0, 0) ||
        !mz_zip_writer_add_mem(&zip.archive, "manifest.json", manifestJson.data(), manifestJson.size(), 0) ||
        !mz_zip_writer_add_mem(&zip.archive, "trace.json.zst", compressed.data(), compressed.size(), 0) ||
        !mz_zip_writer_add_mem(&zip.archive, "shaders.json", shadersJson.data(), shadersJson.size(), 0)) {
        error = "miniz could not construct the profile archive";
        return false;
    }
    void* bytes = nullptr;
    size_t size = 0;
    if (!mz_zip_writer_finalize_heap_archive(&zip.archive, &bytes, &size)) {
        error = "miniz could not finalize the profile archive";
        return false;
    }
    std::unique_ptr<void, decltype(&mz_free)> storage(bytes, &mz_free);
    if (size > maxProfileBytes) {
        error = "Profile archive exceeds the 256 MiB format limit";
        return false;
    }
    return writeExclusive(path, bytes, size, error);
}

bool exportTrace(const std::string& path, const Trace& trace, std::string& error) {
    std::string json;
    return toJson(trace, json, error) && writeExclusive(path, json.data(), json.size(), error);
}

bool readProfile(const std::string& path, Profile& profile, std::string& error) {
    std::ifstream input(std::filesystem::path(std::u8string(path.begin(), path.end())),
                        std::ios::binary | std::ios::ate);
    if (!input || input.tellg() < 0 || static_cast<uint64_t>(input.tellg()) > maxProfileBytes) {
        error = "Cannot read profile or archive exceeds 256 MiB";
        return false;
    }
    std::string archive(static_cast<size_t>(input.tellg()), '\0');
    input.seekg(0);
    if (!input.read(archive.data(), static_cast<std::streamsize>(archive.size()))) {
        error = "Could not read complete profile archive";
        return false;
    }
    Zip zip;
    if (!mz_zip_reader_init_mem(&zip.archive, archive.data(), archive.size(), 0)) {
        error = "Invalid profile ZIP archive";
        return false;
    }
    const auto count = mz_zip_reader_get_num_files(&zip.archive);
    if (count != 3) {
        error = "Profile v1 requires exactly manifest.json, trace.json.zst, and shaders.json";
        return false;
    }
    uint64_t total = 0;
    std::vector<std::string> names;
    for (mz_uint i = 0; i < count; ++i) {
        mz_zip_archive_file_stat stat{};
        if (!mz_zip_reader_file_stat(&zip.archive, i, &stat) || !safePath(stat.m_filename) ||
            stat.m_method != 0 || stat.m_is_encrypted || stat.m_is_directory ||
            stat.m_uncomp_size > maxProfileBytes ||
            total > maxProfileBytes - stat.m_uncomp_size ||
            std::find(names.begin(), names.end(), stat.m_filename) != names.end()) {
            error = "Unsafe, duplicate, or oversized profile ZIP entry";
            return false;
        }
        total += stat.m_uncomp_size;
        names.emplace_back(stat.m_filename);
    }
    std::string json;
    if (!extract(zip, "manifest.json", maxManifestBytes, json, error)) return false;
    Profile parsed;
    if (const auto ec = glz::read<glz::opts{.error_on_missing_keys = true}>(parsed.manifest, json)) {
        error = "Invalid profile manifest: " + glz::format_error(ec, json);
        return false;
    }
    const auto& manifest = parsed.manifest;
    if (manifest.schemaVersion != 1 || manifest.kind != "profile" || manifest.replayable ||
        manifest.clock != "monotonic" || manifest.timestampUnit != "ns" ||
        manifest.entries.size() != 2 || manifest.complete != (manifest.droppedEvents == 0) ||
        manifest.recordedEvents > 1'000'000) {
        error = "Unsupported or inconsistent profile manifest";
        return false;
    }
    const Entry* traceEntry = nullptr;
    const Entry* shadersEntry = nullptr;
    for (const auto& entry : manifest.entries) {
        if (entry.path == "trace.json.zst" && entry.encoding == "zstd" && !traceEntry) traceEntry = &entry;
        else if (entry.path == "shaders.json" && entry.encoding == "json" && !shadersEntry) shadersEntry = &entry;
        else {
            error = "Unsupported or duplicate manifest entry";
            return false;
        }
    }
    if (!traceEntry || !shadersEntry ||
        traceEntry->uncompressedSize > maxProfileBytes ||
        shadersEntry->uncompressedSize > maxProfileBytes - traceEntry->uncompressedSize) {
        error = "Missing or oversized decoded profile entries";
        return false;
    }
    std::string compressed;
    if (!extract(zip, "trace.json.zst", maxProfileBytes, compressed, error)) return false;
    const auto frameSize = ZSTD_getFrameContentSize(compressed.data(), compressed.size());
    if (frameSize != traceEntry->uncompressedSize ||
        ZSTD_findFrameCompressedSize(compressed.data(), compressed.size()) != compressed.size() ||
        ZSTD_getDictID_fromFrame(compressed.data(), compressed.size()) != 0) {
        error = "zstd frame size does not match profile manifest";
        return false;
    }
    json.resize(static_cast<size_t>(frameSize));
    const size_t decoded = ZSTD_decompress(json.data(), json.size(), compressed.data(), compressed.size());
    if (ZSTD_isError(decoded) || decoded != json.size()) {
        error = "Corrupt zstd trace payload";
        return false;
    }
    if (const auto ec = glz::read<glz::opts{.error_on_missing_keys = true}>(parsed.trace, json)) {
        error = "Invalid Chrome trace: " + glz::format_error(ec, json);
        return false;
    }
    if (parsed.trace.traceEvents.size() != manifest.recordedEvents) {
        error = "Trace event count does not match manifest";
        return false;
    }
    for (const auto& event : parsed.trace.traceEvents) {
        if (event.ph != "X" || !std::isfinite(event.ts) || !std::isfinite(event.dur) ||
            event.ts < 0 || event.dur < 0 || !event.tid) {
            error = "Invalid timing span";
            return false;
        }
    }
    if (!extract(zip, "shaders.json", static_cast<size_t>(shadersEntry->uncompressedSize), json, error))
        return false;
    if (json.size() != shadersEntry->uncompressedSize) {
        error = "Shader metadata size does not match manifest";
        return false;
    }
    if (json != "[]") {
        if (const auto ec = glz::read<glz::opts{.error_on_missing_keys = true}>(parsed.shaders, json)) {
            error = "Invalid shader metadata: " + glz::format_error(ec, json);
            return false;
        }
        if (parsed.shaders.schemaVersion != 1 || parsed.shaders.codeVersions.size() > maxCodeVersions) {
            error = "Unsupported shader metadata schema or code-version count";
            return false;
        }
        uint64_t expectedId = 1;
        std::set<std::pair<uint64_t, uint32_t>> identities;
        std::map<uint64_t, std::pair<std::string, uint32_t>> pipelines;
        for (const auto& code : parsed.shaders.codeVersions) {
            if (code.id != expectedId++ || !code.pipelineId || code.pipelineId > maxCodeVersions ||
                code.pso.size() != 32 || code.pso.find_first_not_of("0123456789abcdef") != std::string::npos ||
                code.stage != "compute" || !code.simdWidth || code.version < 1 || code.version > 2 ||
                code.tier < 1 || code.tier > 2 || (code.cache != "hit" && code.cache != "miss") ||
                (code.cache == "hit" && (code.tier != 2 || code.version != 1)) ||
                (code.version == 2 && code.tier != 2)) {
                error = "Invalid compute code-version metadata";
                return false;
            }
            const auto signature = std::make_pair(code.pso, code.simdWidth);
            const auto pipeline = pipelines.try_emplace(code.pipelineId, signature).first;
            if (!identities.emplace(code.pipelineId, code.version).second ||
                pipeline->second != signature) {
                error = "Inconsistent compute pipeline or duplicate code-version identity";
                return false;
            }
        }
    }
    for (const auto& event : parsed.trace.traceEvents) {
        const auto& args = event.args;
        if (!args.pipelineId && !args.codeVersionId) continue;
        if (!args.pipelineId || !args.codeVersionId || *args.codeVersionId == 0 ||
            *args.codeVersionId > parsed.shaders.codeVersions.size()) {
            error = "Invalid trace code-version reference";
            return false;
        }
        const auto& code = parsed.shaders.codeVersions[*args.codeVersionId - 1];
        if (*args.pipelineId != code.pipelineId || args.pso != code.pso || args.stage != code.stage ||
            args.simdWidth != code.simdWidth || args.tier != code.tier || args.cache != code.cache) {
            error = "Trace does not match referenced compute code version";
            return false;
        }
    }
    profile = std::move(parsed);
    return true;
}

} // namespace nacelle::profile
