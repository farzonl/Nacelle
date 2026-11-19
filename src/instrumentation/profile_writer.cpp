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
#include <sstream>

namespace nacelle::profile {
namespace {

struct Zip {
    mz_zip_archive archive{};
    bool writing = false;
    ~Zip() {
        if (writing) mz_zip_writer_end(&archive);
        else mz_zip_reader_end(&archive);
    }
};

std::string hashString(uint64_t lo, uint64_t hi) {
    if (lo == 0 && hi == 0) return {};
    std::ostringstream stream;
    stream << std::hex << std::setfill('0') << std::setw(16) << hi << std::setw(16) << lo;
    return stream.str();
}

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
    Trace trace;
    trace.traceEvents.reserve(static_cast<size_t>(session.eventCount.load()));
    constexpr const char* stages[] = {"unknown", "compute", "graphics", "vertex", "pixel", "mesh", "raytracing"};
    constexpr const char* caches[] = {"unknown", "hit", "miss"};
    for (size_t i = 0; i < std::min(session.capacity, session.nextBlock.load()); ++i) {
        const auto& event = session.events[i];
        if (event.id == 0) continue;
        TraceEvent item;
        item.name = event.name;
        item.cat = event.attributes.tier ? "jit" : "runtime";
        item.ts = static_cast<double>(event.startNs) / 1000.0;
        item.dur = static_cast<double>(event.durationNs) / 1000.0;
        item.tid = event.thread;
        item.args = {event.id, event.parent,
                     hashString(event.attributes.hashLo, event.attributes.hashHi),
                     stages[static_cast<unsigned>(event.attributes.stage)],
                     event.attributes.simdWidth, event.attributes.tier,
                     caches[static_cast<unsigned>(event.attributes.cache)]};
        trace.traceEvents.push_back(std::move(item));
    }
    std::sort(trace.traceEvents.begin(), trace.traceEvents.end(),
              [](const auto& a, const auto& b) { return a.ts < b.ts; });
    std::string traceJson;
    if (!toJson(trace, traceJson, error)) return false;
    if (traceJson.size() > maxProfileBytes - 2) {
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

    Manifest manifest;
    manifest.complete = session.dropped.load() == 0;
    manifest.recordedEvents = session.eventCount.load();
    manifest.droppedEvents = session.dropped.load();
    manifest.captureOptions.eventBufferBytes = session.bufferBytes;
    manifest.build = {NACELLE_PROFILE_COMPILER, NACELLE_PROFILE_OS,
                      NACELLE_PROFILE_ARCH, NACELLE_PROFILE_REVISION,
                      session.compilerHash, session.targetTriple, session.cpu, session.cpuFeatures};
    manifest.entries = {{"trace.json.zst", "zstd", traceJson.size()}, {"shaders.json", "json", 2}};
    std::string manifestJson;
    if (!toJson(manifest, manifestJson, error)) return false;

    Zip zip;
    zip.writing = true;
    if (!mz_zip_writer_init_heap(&zip.archive, 0, 0) ||
        !mz_zip_writer_add_mem(&zip.archive, "manifest.json", manifestJson.data(), manifestJson.size(), 0) ||
        !mz_zip_writer_add_mem(&zip.archive, "trace.json.zst", compressed.data(), compressed.size(), 0) ||
        !mz_zip_writer_add_mem(&zip.archive, "shaders.json", "[]", 2, 0)) {
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
    if (!extract(zip, "manifest.json", 1024 * 1024, json, error)) return false;
    Profile parsed;
    if (const auto ec = glz::read<glz::opts{.error_on_missing_keys = true}>(parsed.manifest, json)) {
        error = "Invalid profile manifest: " + glz::format_error(ec, json);
        return false;
    }
    const auto& manifest = parsed.manifest;
    if (manifest.schemaVersion != 1 || manifest.kind != "profile" || manifest.replayable ||
        manifest.clock != "monotonic" || manifest.timestampUnit != "ns" ||
        manifest.entries.size() != 2 || manifest.complete != (manifest.droppedEvents == 0) ||
        manifest.recordedEvents > maxProfileBytes / sizeof(Event)) {
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
    if (json.size() != shadersEntry->uncompressedSize || json != "[]") {
        error = "Unsupported shader metadata payload";
        return false;
    }
    profile = std::move(parsed);
    return true;
}

} // namespace nacelle::profile
