#include "capture.h"
#include "profile_format.h"
#include "profile_counters.h"
#include "archive.h"
#include "profile_memory.h"

#include <glaze/glaze.hpp>
#include <miniz.h>
#define ZSTD_STATIC_LINKING_ONLY
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
#include <cstring>
#if defined(_WIN32)
#include <windows.h>
#include <sddl.h>
#include <io.h>
#include <fcntl.h>
#include <bcrypt.h>
#endif

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

size_t quotedSize(std::string_view value) {
    size_t bytes = 2;
    for (const unsigned char c : value) {
        if (c < 0x20 && c != '\b' && c != '\f' && c != '\n' && c != '\r' && c != '\t')
            return maxProfileBytes + 1;
        bytes += glz::detail::char_escape_table[c] ? 2 : 1;
    }
    return bytes;
}

template <typename T> size_t jsonSize(const T &value) {
    using V = std::remove_cvref_t<T>;
    if constexpr (std::is_same_v<V, std::string> || std::is_same_v<V, std::string_view>) {
        return quotedSize(value);
    } else if constexpr (std::is_arithmetic_v<V>) {
        char buffer[128];
        char *output = buffer;
        const auto result = glz::write_json(value, output);
        return result ? *result : maxProfileBytes + 1;
    } else if constexpr (requires { value.has_value(); }) {
        return value ? jsonSize(*value) : 4;
    } else if constexpr (std::ranges::range<V>) {
        size_t bytes = 2;
        bool first = true;
        for (const auto &item : value) {
            bytes += jsonSize(item) + (first ? 0 : 1);
            first = false;
            if (bytes > maxProfileBytes)
                break;
        }
        return bytes;
    } else {
        size_t bytes = 2;
        bool first = true;
        const auto members = glz::detail::to_tuple(value);
        glz::for_each<glz::reflect<V>::size>([&](auto index) {
            constexpr size_t I = decltype(index)::value;
            const auto &member = glz::get<I>(members);
            if constexpr (requires { member.has_value(); }) {
                if (!member)
                    return;
            }
            bytes += quotedSize(glz::reflect<V>::keys[I]) + 1 + jsonSize(member) + (first ? 0 : 1);
            first = false;
        });
        return bytes;
    }
}

template <typename T>
bool toJson(const T &value, std::string &json, std::string &error, size_t limit = maxProfileBytes) {
    const size_t size = jsonSize(value);
    if (size > limit) {
        error = "Profile JSON exceeds its format limit or contains unsupported control characters";
        return false;
    }
    // Raw-buffer Glaze serialization cannot grow or allocate inside its noexcept writer.
    std::string serialized(size + glz::write_padding_bytes, '\0');
    char* output = serialized.data();
    const auto result = glz::write_json(value, output);
    if (!result || *result != size) {
        error = "Glaze serialization size or encoding mismatch";
        return false;
    }
    serialized.resize(size);
    json.swap(serialized);
    return true;
}

bool manifestToJson(const Manifest &manifest, std::string &json, std::string &error) {
    if (!toJson(manifest, json, error, maxManifestBytes)) {
        error = "Profile manifest exceeds the 1 MiB format limit or could not be serialized";
        return false;
    }
    if (json.size() > maxManifestBytes) {
        error = "Profile manifest exceeds the 1 MiB format limit";
        return false;
    }
    return true;
}

bool writeExclusive(const std::string &path, const void *bytes, size_t size, std::string &error) {
    const auto nativePath = std::filesystem::path(std::u8string(path.begin(), path.end()));
    FILE *file = nullptr;
#if defined(_WIN32)
    if (_wfopen_s(&file, nativePath.c_str(), L"wbx") != 0)
        file = nullptr;
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
        if (ec)
            error += "; could not remove partial file: " + ec.message();
        return false;
    }
    return true;
}
using File = std::unique_ptr<FILE, decltype(&std::fclose)>;

File temporaryFile(std::string &error) {
#if defined(_WIN32)
    wchar_t directory[MAX_PATH + 1];
    const DWORD directoryLength = GetTempPathW(MAX_PATH + 1, directory);
    if (!directoryLength || directoryLength >= MAX_PATH + 1) {
        error = "Cannot locate private profile staging directory";
        return {nullptr, &std::fclose};
    }
    PSECURITY_DESCRIPTOR descriptor = nullptr;
    if (!ConvertStringSecurityDescriptorToSecurityDescriptorW(L"D:P(A;;GA;;;OW)", SDDL_REVISION_1,
                                                              &descriptor, nullptr)) {
        error = "Cannot configure private profile staging permissions";
        return {nullptr, &std::fclose};
    }
    std::unique_ptr<void, decltype(&LocalFree)> security(descriptor, &LocalFree);
    SECURITY_ATTRIBUTES attributes{sizeof(SECURITY_ATTRIBUTES), descriptor, FALSE};
    for (unsigned attempt = 0; attempt < 16; ++attempt) {
        std::wstring name(directory);
        name += L"nacelle-profile-";
        std::array<ULONG, 4> random{};
        if (BCryptGenRandom(nullptr, reinterpret_cast<PUCHAR>(random.data()), sizeof(random),
                            BCRYPT_USE_SYSTEM_PREFERRED_RNG) < 0) {
            error = "Cannot generate a private profile staging identity";
            return {nullptr, &std::fclose};
        }
        for (const auto value : random) name += std::to_wstring(value) + L"-";
        const HANDLE handle =
            CreateFileW(name.c_str(), GENERIC_READ | GENERIC_WRITE | DELETE, 0, &attributes, CREATE_NEW,
                        FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
        if (handle == INVALID_HANDLE_VALUE) {
            if (GetLastError() == ERROR_FILE_EXISTS)
                continue;
            break;
        }
        const int fd = _open_osfhandle(reinterpret_cast<intptr_t>(handle), _O_BINARY | _O_RDWR);
        if (fd == -1) {
            CloseHandle(handle);
            break;
        }
        FILE *file = _fdopen(fd, "w+b");
        if (!file) {
            _close(fd);
            break;
        }
        File result(file, &std::fclose);
        NACELLE_PROFILE_PRIVATE_FILE(file);
        return result;
    }
#else
    if (FILE *file = std::tmpfile()) {
        File result(file, &std::fclose);
        NACELLE_PROFILE_PRIVATE_FILE(file);
        return result;
    }
#endif
    error = "Cannot create private profile staging file";
    return {nullptr, &std::fclose};
}

bool seekFile(FILE *file, uint64_t offset) {
#if defined(_WIN32)
    return _fseeki64(file, static_cast<__int64>(offset), SEEK_SET) == 0;
#else
    return fseeko(file, static_cast<off_t>(offset), SEEK_SET) == 0;
#endif
}

bool writeExclusiveFile(const std::string &path, FILE *source, size_t size, std::string &error) {
    const auto nativePath = std::filesystem::path(std::u8string(path.begin(), path.end()));
    FILE *raw = nullptr;
#if defined(_WIN32)
    if (_wfopen_s(&raw, nativePath.c_str(), L"wbx") != 0)
        raw = nullptr;
#else
    raw = std::fopen(nativePath.c_str(), "wbx");
#endif
    if (!raw) {
        error = "Cannot create profile (destination must not already exist): " + path;
        return false;
    }
    File output(raw, &std::fclose);
    bool committed = false;
    struct Rollback {
        File &output;
        const std::filesystem::path &path;
        bool &committed;
        ~Rollback() {
            if (committed)
                return;
            output.reset();
            std::error_code ec;
            std::filesystem::remove(path, ec);
            if (ec)
                std::fprintf(stderr, "[NacelleProfile] Could not remove partial profile: %s\n",
                             ec.message().c_str());
        }
    } rollback{output, nativePath, committed};
    std::array<char, 64 * 1024> buffer;
    bool written = seekFile(source, 0);
    for (size_t remaining = size; written && remaining;) {
        const auto amount = std::min(remaining, buffer.size());
        written = std::fread(buffer.data(), 1, amount, source) == amount &&
                  std::fwrite(buffer.data(), 1, amount, output.get()) == amount;
        if (remaining == size)
            NACELLE_PROFILE_MEMORY("write-progress", size);
        remaining -= amount;
    }
    const bool closed = std::fclose(output.release()) == 0;
    if (!written || !closed) {
        std::error_code ec;
        std::filesystem::remove(nativePath, ec);
        error = "Failed writing profile: " + path;
        if (ec)
            error += "; could not remove partial file: " + ec.message();
        return false;
    }
    committed = true;
    return true;
}

TraceEvent traceEvent(const Event &event) {
    constexpr const char *stages[] = {"unknown",    "compute",  "graphics", "vertex", "pixel",        "mesh",
                                      "raytracing", "geometry", "hull",     "domain", "amplification"};
    constexpr const char *caches[] = {"unknown", "hit", "miss"};
    TraceEvent item;
    item.name = event.name;
    item.cat = event.attributes.tier && !event.attributes.codeVersionId ? "jit" : "runtime";
    item.ts = static_cast<double>(event.startNs) / 1000.0;
    item.dur = static_cast<double>(event.durationNs) / 1000.0;
    item.tid = event.thread;
    item.args = {event.id,
                 event.parent,
                 formatProfileHash(event.attributes.hashLo, event.attributes.hashHi),
                 stages[static_cast<unsigned>(event.attributes.stage)],
                 event.attributes.simdWidth,
                 event.attributes.tier,
                 caches[static_cast<unsigned>(event.attributes.cache)]};
    if (event.attributes.codeVersionId || event.attributes.codeSelectionId)
        item.args.pipelineId = event.attributes.pipelineId;
    if (event.attributes.codeVersionId)
        item.args.codeVersionId = event.attributes.codeVersionId;
    if (event.attributes.codeSelectionId)
        item.args.codeSelectionId = event.attributes.codeSelectionId;
    return item;
}

bool compressTrace(const Session &session, FILE *file, size_t limit, size_t &decodedBytes,
                   size_t &compressedBytes, std::string &error) {
    NACELLE_PROFILE_MEMORY("ordering-begin", 0);
    std::vector<const Event *> ordered;
    ordered.reserve(static_cast<size_t>(session.eventCount));
    for (size_t i = 0; i < std::min(session.capacity, session.nextBlock.load()); ++i)
        if (session.events[i].id)
            ordered.push_back(&session.events[i]);
    std::sort(ordered.begin(), ordered.end(), [](const Event *a, const Event *b) {
        return static_cast<double>(a->startNs) / 1000.0 < static_cast<double>(b->startNs) / 1000.0;
    });
    NACELLE_PROFILE_MEMORY("ordering-end", ordered.capacity() * sizeof(const Event *));
    constexpr std::string_view prefix = R"({"traceEvents":[)";
    constexpr std::string_view suffix = R"(],"displayTimeUnit":"ms"})";
    decodedBytes = prefix.size() + suffix.size();
    NACELLE_PROFILE_MEMORY("trace-count-begin", ordered.capacity() * sizeof(const Event *));
    for (size_t i = 0; i < ordered.size(); ++i) {
        const auto bytes = jsonSize(traceEvent(*ordered[i])) + (i ? 1 : 0);
        if (decodedBytes > limit || bytes > limit - decodedBytes) {
            error = "Profile trace exceeds the 256 MiB uncompressed format limit";
            return false;
        }
        decodedBytes += bytes;
    }
    if (decodedBytes > limit) {
        error = "Profile trace exceeds the 256 MiB uncompressed format limit";
        return false;
    }
    NACELLE_PROFILE_MEMORY("trace-count-end", ordered.capacity() * sizeof(const Event *));
    auto context = std::unique_ptr<ZSTD_CCtx, decltype(&ZSTD_freeCCtx)>(ZSTD_createCCtx(), &ZSTD_freeCCtx);
    if (!context)
        throw std::bad_alloc();
    for (const auto result : {ZSTD_CCtx_setParameter(context.get(), ZSTD_c_compressionLevel, 1),
                              ZSTD_CCtx_setParameter(context.get(), ZSTD_c_windowLog, 19),
                              ZSTD_CCtx_setPledgedSrcSize(context.get(), decodedBytes)}) {
        if (ZSTD_isError(result)) {
            error = std::string("zstd setup failed: ") + ZSTD_getErrorName(result);
            return false;
        }
    }
    auto buffer = std::make_unique<char[]>(128 * 1024);
    std::string eventJson;
    compressedBytes = 0;
#if defined(NACELLE_PROFILE_MEMORY_DIAGNOSTICS)
    uint64_t encodingNs = 0, compressionNs = 0;
#endif
    const auto emit = [&](std::string_view bytes, ZSTD_EndDirective directive) {
#if defined(NACELLE_PROFILE_MEMORY_DIAGNOSTICS)
        const auto started = std::chrono::steady_clock::now();
#endif
        ZSTD_inBuffer input{bytes.data(), bytes.size(), 0};
        size_t remaining = 1;
        do {
            ZSTD_outBuffer output{buffer.get(), 128 * 1024, 0};
            remaining = ZSTD_compressStream2(context.get(), &output, &input, directive);
            if (ZSTD_isError(remaining)) {
                error = std::string("zstd compression failed: ") + ZSTD_getErrorName(remaining);
                return false;
            }
            if (output.pos > maxProfileBytes - compressedBytes ||
                std::fwrite(buffer.get(), 1, output.pos, file) != output.pos) {
                error = "Could not write bounded compressed profile staging data";
                return false;
            }
            compressedBytes += output.pos;
        } while (input.pos < input.size || (directive == ZSTD_e_end && remaining));
#if defined(NACELLE_PROFILE_MEMORY_DIAGNOSTICS)
        compressionNs += std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started).count();
#endif
        return true;
    };
    NACELLE_PROFILE_MEMORY("compression-begin", ordered.capacity() * sizeof(const Event *) + 128 * 1024 +
                                                    ZSTD_sizeof_CCtx(context.get()));
    if (!emit(prefix, ZSTD_e_continue))
        return false;
    for (size_t i = 0; i < ordered.size(); ++i) {
#if defined(NACELLE_PROFILE_MEMORY_DIAGNOSTICS)
        const auto started = std::chrono::steady_clock::now();
#endif
        if (!toJson(traceEvent(*ordered[i]), eventJson, error, 4096))
            return false;
#if defined(NACELLE_PROFILE_MEMORY_DIAGNOSTICS)
        encodingNs += std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started).count();
#endif
        if ((i && !emit(",", ZSTD_e_continue)) || !emit(eventJson, ZSTD_e_continue)) return false;
    }
    if (!emit(suffix, ZSTD_e_end) || std::fflush(file) != 0 || !seekFile(file, 0)) {
        if (error.empty())
            error = "Could not finalize compressed profile staging data";
        return false;
    }
    NACELLE_PROFILE_MEMORY("compression-end", ordered.capacity() * sizeof(const Event *) +
                                                  eventJson.capacity() + 128 * 1024 +
                                                  ZSTD_sizeof_CCtx(context.get()));
#if defined(NACELLE_PROFILE_MEMORY_DIAGNOSTICS)
    observeProfileTiming("trace-encoding", encodingNs);
    observeProfileTiming("zstd-and-spool", compressionNs);
#endif
    return true;
}

    size_t writeArchiveFile(void* opaque, mz_uint64 offset, const void* bytes, size_t size) {
        auto* file = static_cast<FILE*>(opaque);
        if (offset > maxProfileBytes || size > maxProfileBytes - offset || !seekFile(file, offset))
            return 0;
        return std::fwrite(bytes, 1, size, file);
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
        error = "Profile metadata admission failed or exceeded its format limits; begin a new capture";
        return false;
    }
    NACELLE_PROFILE_MEMORY("metadata-begin", 0);
    Manifest manifest;
    manifest.complete = session.dropped.load() == 0 && (!session.spatial || session.spatial->complete());
    manifest.recordedEvents = session.eventCount;
    manifest.droppedEvents = session.dropped.load();
    manifest.captureOptions.eventBufferBytes = session.bufferBytes;
    manifest.build = {NACELLE_PROFILE_COMPILER, NACELLE_PROFILE_OS,
                      NACELLE_PROFILE_ARCH, NACELLE_PROFILE_REVISION,
                      session.compilerHash, session.targetTriple, session.cpu, session.cpuFeatures};
    manifest.entries = {{"trace.json.zst", "zstd", 0}, {"shaders.json", "json", 2}};
    std::string artifactsJson, spatialJson, countersJson;
    if (const auto& store = session.retainedArtifacts) {
        ArtifactMetadata artifacts;
        std::set<std::string> admitted;
        {
            std::lock_guard lock(store->mutex);
            if (store->overflow) {
                error = "Artifact retention budget exceeded; reconfigure retention and begin a new capture";
                return false;
            }
            for (const auto& code : session.shaders.codeVersions) {
                ArtifactSelection selection;
                selection.codeVersionId = code.id;
                selection.shaderAvailability = "not-retained";
                selection.nativeAvailability = !code.artifactId && code.tier == 1 ? "tier1-unavailable" : "not-retained";
                for (const auto* identity : {&code.shaderId, &code.artifactId}) {
                    if (!*identity) continue;
                    const auto found = store->groups.find(**identity);
                    if (found == store->groups.end() || found->second.empty()) continue;
                    for (const auto& payload : found->second) {
                        selection.payloadIds.push_back(payload.id);
                        if (admitted.insert(payload.id).second) artifacts.payloads.push_back(payload);
                    }
                    (identity == &code.shaderId ? selection.shaderAvailability : selection.nativeAvailability) = "available";
                }
                artifacts.selections.push_back(std::move(selection));
            }
        }
        manifest.schemaVersion = 2;
        manifest.captureOptions.embeddedSources = store->options.flags & NACELLE_PROFILE_EMBED_SHADER_SOURCES;
        manifest.captureOptions.artifactBytes = store->options.artifact_bytes;
        manifest.captureOptions.sourceBytes = store->options.source_bytes;
        if (!validateArtifacts(artifacts, session.shaders, manifest.captureOptions, error) ||
            !toJson(artifacts, artifactsJson, error)) return false;
        if (artifactsJson.size() > maxProfileBytes) {
            error = "Artifact metadata exceeds 256 MiB";
            return false;
        }
        manifest.entries.push_back({"artifacts.json", "json", artifactsJson.size(), artifactDigest(artifactsJson)});
    }
    if (session.spatial) {
        SpatialMetadata spatial;
        manifest.schemaVersion = 3;
        manifest.captureOptions.spatialBytes = session.spatial->options.record_bytes;
        if (!session.spatial->aggregate(spatial, error) ||
            !validateSpatial(spatial, session.shaders, manifest.captureOptions, error) ||
            !toJson(spatial, spatialJson, error, maxProfileBytes - artifactsJson.size())) return false;
        manifest.entries.push_back({"spatial.json", "json", spatialJson.size(), artifactDigest(spatialJson)});
    }
#if defined(NACELLE_ENABLE_CPU_COUNTERS)
    if (session.counters) {
        const auto counters = session.counters->metadata();
        manifest.schemaVersion = 4;
        manifest.complete = manifest.complete && counters.complete;
        manifest.captureOptions.counterBytes = counters.recordBytes;
        if (!validateCounters(counters, session.shaders, manifest.captureOptions, error) ||
            !toJson(counters, countersJson, error, maxProfileBytes - artifactsJson.size() - spatialJson.size()))
            return false;
        manifest.entries.push_back({"counters.json", "json", countersJson.size(), artifactDigest(countersJson)});
    }
#endif
    std::string manifestJson;
    // Reject oversized metadata before allocating trace objects and compression buffers.
    if (!manifestToJson(manifest, manifestJson, error)) return false;
    std::string shadersJson = "[]";
    if ((!session.shaders.codeVersions.empty() || session.shaders.codeSelections) &&
        !toJson(session.shaders, shadersJson, error, maxProfileBytes - artifactsJson.size() - spatialJson.size() - countersJson.size()))
        return false;
    if (shadersJson.size() > maxProfileBytes) {
        error = "Profile shader metadata exceeds the 256 MiB format limit";
        return false;
    }
    const auto metadataBytes = shadersJson.size() + artifactsJson.size() + spatialJson.size() + countersJson.size();
    if (metadataBytes > maxProfileBytes) {
        error = "Profile metadata exceeds the 256 MiB uncompressed format limit";
        return false;
    }
    NACELLE_PROFILE_MEMORY("metadata-end", manifestJson.capacity() + shadersJson.capacity() +
        artifactsJson.capacity() + spatialJson.capacity() + countersJson.capacity());
    auto compressed = temporaryFile(error);
    if (!compressed) return false;
    size_t traceBytes = 0, compressedBytes = 0;
    if (!compressTrace(session, compressed.get(), maxProfileBytes - metadataBytes,
                       traceBytes, compressedBytes, error)) return false;
    manifest.entries.front().uncompressedSize = traceBytes;
    manifest.entries[1].uncompressedSize = shadersJson.size();
    if (!manifestToJson(manifest, manifestJson, error)) return false;

    auto archiveFile = temporaryFile(error);
    if (!archiveFile) return false;
    Zip zip;
    zip.writing = true;
    zip.archive.m_pWrite = &writeArchiveFile;
    zip.archive.m_pIO_opaque = archiveFile.get();
    NACELLE_PROFILE_MEMORY("archive-begin", metadataBytes);
    if (!mz_zip_writer_init(&zip.archive, 0) ||
        !mz_zip_writer_add_mem(&zip.archive, "manifest.json", manifestJson.data(), manifestJson.size(), 0) ||
        !mz_zip_writer_add_cfile(&zip.archive, "trace.json.zst", compressed.get(), compressedBytes,
            nullptr, nullptr, 0, 0, nullptr, 0, nullptr, 0) ||
        !mz_zip_writer_add_mem(&zip.archive, "shaders.json", shadersJson.data(), shadersJson.size(), 0)) {
        error = "miniz could not construct the profile archive";
        return false;
    }
    if (!artifactsJson.empty() && !mz_zip_writer_add_mem(&zip.archive, "artifacts.json",
            artifactsJson.data(), artifactsJson.size(), 0)) {
        error = "miniz could not add artifact metadata";
        return false;
    }
    if (!spatialJson.empty() && !mz_zip_writer_add_mem(&zip.archive, "spatial.json",
            spatialJson.data(), spatialJson.size(), 0)) {
        error = "miniz could not add spatial metadata";
        return false;
    }
    if (!countersJson.empty() && !mz_zip_writer_add_mem(&zip.archive, "counters.json",
            countersJson.data(), countersJson.size(), 0)) {
        error = "miniz could not add CPU counter metadata";
        return false;
    }
    if (!mz_zip_writer_finalize_archive(&zip.archive) || std::fflush(archiveFile.get()) != 0) {
        error = "miniz could not finalize the profile archive";
        return false;
    }
    const auto size = zip.archive.m_archive_size;
    if (size > maxProfileBytes) {
        error = "Profile archive exceeds the 256 MiB format limit";
        return false;
    }
    NACELLE_PROFILE_MEMORY("archive-end", metadataBytes);
    NACELLE_PROFILE_MEMORY("write-begin", metadataBytes);
    const bool written = writeExclusiveFile(path, archiveFile.get(), static_cast<size_t>(size), error);
    NACELLE_PROFILE_MEMORY("write-end", metadataBytes);
    return written;
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
    return readProfileBytes(archive, profile, error);
}

bool validateStoredArchive(std::span<const char> archive, unsigned minimumEntries,
                           unsigned maximumEntries, std::string& error) {
    if (archive.size() < 22 || archive.size() > maxProfileBytes) {
        error = "Profile archive is truncated or exceeds 256 MiB";
        return false;
    }
    const auto u16 = [&](size_t offset) {
        return static_cast<uint16_t>(static_cast<unsigned char>(archive[offset]) |
            static_cast<unsigned char>(archive[offset + 1]) << 8);
    };
    const auto u32 = [&](size_t offset) {
        return static_cast<uint32_t>(u16(offset)) | static_cast<uint32_t>(u16(offset + 2)) << 16;
    };
    size_t endDirectory = archive.size() - 22;
    const size_t minimum = endDirectory > 65535 ? endDirectory - 65535 : 0;
    while (u32(endDirectory) != 0x06054b50 ||
           endDirectory + 22 + u16(endDirectory + 20) != archive.size()) {
        if (endDirectory == minimum) {
            error = "Missing or corrupt profile ZIP end directory";
            return false;
        }
        --endDirectory;
    }
    if (u16(endDirectory + 4) || u16(endDirectory + 6) ||
        u16(endDirectory + 8) != u16(endDirectory + 10) ||
        static_cast<uint64_t>(u32(endDirectory + 16)) + u32(endDirectory + 12) != endDirectory) {
        error = "Unsupported multi-disk or corrupt profile ZIP directory bounds";
        return false;
    }
    Zip zip;
    if (!mz_zip_reader_init_mem(&zip.archive, archive.data(), archive.size(), 0)) {
        error = "Invalid profile ZIP archive";
        return false;
    }
    const auto count = mz_zip_reader_get_num_files(&zip.archive);
    if ((count < minimumEntries || count > maximumEntries) || mz_zip_is_zip64(&zip.archive)) {
        error = "Profile v1 requires a non-ZIP64 archive with exactly manifest.json, trace.json.zst, and shaders.json";
        return false;
    }
    uint64_t total = 0;
    std::vector<std::string> names;
    std::vector<std::pair<uint64_t, uint64_t>> ranges;
    for (mz_uint i = 0; i < count; ++i) {
        mz_zip_archive_file_stat stat{};
        if (!mz_zip_reader_file_stat(&zip.archive, i, &stat) || !safePath(stat.m_filename) ||
            stat.m_method != 0 || (stat.m_bit_flag & ~0x808) || stat.m_is_encrypted || stat.m_is_directory ||
            stat.m_uncomp_size > maxProfileBytes ||
            total > maxProfileBytes - stat.m_uncomp_size ||
            std::find(names.begin(), names.end(), stat.m_filename) != names.end()) {
            error = "Unsafe, duplicate, or oversized profile ZIP entry";
            return false;
        }
        total += stat.m_uncomp_size;
        names.emplace_back(stat.m_filename);
        if (!mz_zip_validate_file(&zip.archive, i, MZ_ZIP_FLAG_VALIDATE_HEADERS_ONLY)) {
            error = "Profile ZIP directory and local header disagree";
            return false;
        }
        const auto local = stat.m_local_header_ofs;
        const auto directory = zip.archive.m_central_directory_file_ofs;
        if (local > directory || directory - local < 30) {
            error = "Invalid profile ZIP local header bounds";
            return false;
        }
        if (u16(local + 6) != stat.m_bit_flag || u16(local + 8) != stat.m_method) {
            error = "Profile ZIP local flags or method disagree";
            return false;
        }
        const uint64_t end = local + 30 + u16(local + 26) + u16(local + 28) + stat.m_comp_size;
        uint64_t rangeEnd = end;
        if (stat.m_bit_flag & 8) {
            if (end > directory || directory - end < 12) {
                error = "Missing profile ZIP data descriptor";
                return false;
            }
            const auto descriptor = u32(end) == 0x08074b50 ? end + 4 : end;
            if (directory - descriptor < 12 || u32(descriptor) != stat.m_crc32 ||
                u32(descriptor + 4) != stat.m_comp_size || u32(descriptor + 8) != stat.m_uncomp_size) {
                error = "Profile ZIP data descriptor disagrees";
                return false;
            }
            rangeEnd = descriptor + 12;
        }
        if (rangeEnd > directory || std::any_of(ranges.begin(), ranges.end(), [&](const auto& range) {
                return local < range.second && rangeEnd > range.first;
            })) {
            error = "Overlapping or out-of-bounds profile ZIP entry";
            return false;
        }
        ranges.emplace_back(local, rangeEnd);
    }
    return true;
}

bool readStoredEntry(std::span<const char> archive, const char* name, size_t limit,
                     std::string& result, std::string& error) {
    Zip zip;
    if (!mz_zip_reader_init_mem(&zip.archive, archive.data(), archive.size(), 0)) {
        error = "Invalid stored ZIP archive";
        return false;
    }
    return extract(zip, name, limit, result, error);
}

bool writeStoredArchive(const std::string& path,
                        const std::vector<std::pair<std::string, std::string_view>>& entries,
                        std::string& error) {
    Zip zip;
    zip.writing = true;
    if (!mz_zip_writer_init_heap(&zip.archive, 0, 0)) {
        error = "Cannot initialize stored ZIP writer";
        return false;
    }
    uint64_t total = 0;
    for (const auto& [name, data] : entries) {
        if (!safePath(name) || data.size() > maxProfileBytes - total ||
            !mz_zip_writer_add_mem(&zip.archive, name.c_str(), data.data(), data.size(), 0)) {
            error = "Invalid or oversized stored ZIP entry";
            return false;
        }
        total += data.size();
    }
    void* bytes = nullptr;
    size_t size = 0;
    if (!mz_zip_writer_finalize_heap_archive(&zip.archive, &bytes, &size)) {
        error = "Cannot finalize stored ZIP archive";
        return false;
    }
    std::unique_ptr<void, decltype(&mz_free)> storage(bytes, &mz_free);
    if (size > maxProfileBytes) {
        error = "Stored archive exceeds 256 MiB";
        return false;
    }
    return writeExclusive(path, bytes, size, error);
}

bool readProfileBytes(std::span<const char> archive, Profile& profile, std::string& error) {
    if (!validateStoredArchive(archive, 3, 6, error)) return false;
    Zip zip;
    if (!mz_zip_reader_init_mem(&zip.archive, archive.data(), archive.size(), 0)) {
        error = "Invalid profile ZIP archive";
        return false;
    }
    const auto count = mz_zip_reader_get_num_files(&zip.archive);
    std::string json;
    if (!extract(zip, "manifest.json", maxManifestBytes, json, error)) return false;
    Profile parsed;
    if (const auto ec = glz::read<glz::opts{.error_on_missing_keys = true}>(parsed.manifest, json)) {
        error = "Invalid profile manifest: " + glz::format_error(ec, json);
        return false;
    }
    const auto& manifest = parsed.manifest;
    if ((manifest.schemaVersion < 1 || manifest.schemaVersion > 4) || manifest.kind != "profile" || manifest.replayable ||
        manifest.clock != "monotonic" || manifest.timestampUnit != "ns" ||
        (manifest.schemaVersion < 3 && manifest.entries.size() != (manifest.schemaVersion == 1 ? 2 : 3)) ||
        (manifest.schemaVersion == 3 && manifest.entries.size() != 3 && manifest.entries.size() != 4) ||
        (manifest.schemaVersion == 4 && (manifest.entries.size() < 3 || manifest.entries.size() > 5)) ||
        (manifest.schemaVersion < 4 && manifest.captureOptions.counterBytes) ||
        count != manifest.entries.size() + 1 ||
        (manifest.schemaVersion < 3 && manifest.complete != (manifest.droppedEvents == 0)) ||
        manifest.recordedEvents > 1'000'000) {
        error = "Unsupported or inconsistent profile manifest";
        return false;
    }
    const Entry* traceEntry = nullptr;
    const Entry* shadersEntry = nullptr;
    const Entry* artifactsEntry = nullptr;
    const Entry* spatialEntry = nullptr;
    const Entry* countersEntry = nullptr;
    uint64_t decodedTotal = 0;
    for (const auto& entry : manifest.entries) {
        if (entry.uncompressedSize > maxProfileBytes - decodedTotal) {
            error = "Profile decoded payloads exceed 256 MiB";
            return false;
        }
        decodedTotal += entry.uncompressedSize;
        if (entry.path == "trace.json.zst" && entry.encoding == "zstd" && !traceEntry) traceEntry = &entry;
        else if (entry.path == "shaders.json" && entry.encoding == "json" && !shadersEntry) shadersEntry = &entry;
        else if (manifest.schemaVersion >= 2 && entry.path == "artifacts.json" && entry.encoding == "json" && !artifactsEntry &&
                 entry.digest && entry.digest->size() == 16) artifactsEntry = &entry;
        else if (manifest.schemaVersion >= 3 && entry.path == "spatial.json" && entry.encoding == "json" && !spatialEntry &&
                 entry.digest && entry.digest->size() == 16) spatialEntry = &entry;
        else if (manifest.schemaVersion == 4 && entry.path == "counters.json" && entry.encoding == "json" && !countersEntry &&
                 entry.digest && entry.digest->size() == 16) countersEntry = &entry;
        else {
            error = "Unsupported or duplicate manifest entry";
            return false;
        }
    }
    if (!traceEntry || !shadersEntry || (manifest.schemaVersion == 4 && !countersEntry) ||
        traceEntry->uncompressedSize > maxProfileBytes ||
        shadersEntry->uncompressedSize > maxProfileBytes - traceEntry->uncompressedSize) {
        error = "Missing or oversized decoded profile entries";
        return false;
    }
    std::string compressed;
    if (!extract(zip, "trace.json.zst", maxProfileBytes, compressed, error)) return false;
    const auto frameSize = ZSTD_getFrameContentSize(compressed.data(), compressed.size());
    ZSTD_frameHeader header{};
    if (frameSize != traceEntry->uncompressedSize ||
        ZSTD_findFrameCompressedSize(compressed.data(), compressed.size()) != compressed.size() ||
        ZSTD_getDictID_fromFrame(compressed.data(), compressed.size()) != 0 ||
        ZSTD_getFrameHeader(&header, compressed.data(), compressed.size()) != 0 ||
        header.windowSize > maxProfileBytes) {
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
    if (parsed.trace.displayTimeUnit != "ms" || parsed.trace.traceEvents.size() != manifest.recordedEvents) {
        error = "Trace display unit or event count does not match profile format";
        return false;
    }
    for (const auto& event : parsed.trace.traceEvents) {
        if (event.ph != "X" || !std::isfinite(event.ts) || !std::isfinite(event.dur) ||
            event.ts < 0 || event.dur < 0 || !std::isfinite(event.ts + event.dur) || !event.tid) {
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
        if ((parsed.shaders.schemaVersion != 1 && parsed.shaders.schemaVersion != 2) ||
            (parsed.shaders.schemaVersion == 1 && parsed.shaders.codeSelections) ||
            parsed.shaders.codeVersions.size() +
                (parsed.shaders.codeSelections ? parsed.shaders.codeSelections->size() : 0) > maxCodeVersions) {
            error = "Unsupported shader metadata schema or code-version count";
            return false;
        }
        uint64_t expectedId = 1;
        std::set<std::tuple<uint64_t, std::string, uint32_t>> identities;
        std::map<uint64_t, std::pair<std::string, uint32_t>> pipelines;
        for (const auto& code : parsed.shaders.codeVersions) {
            const bool compute = code.stage == "compute";
            const bool graphics = parsed.shaders.schemaVersion == 2 &&
                (((code.stage == "vertex" || code.stage == "pixel" || code.stage == "geometry" ||
                   code.stage == "domain" || code.stage == "mesh" || code.stage == "amplification") &&
                  code.entryRole == code.stage) ||
                 (code.stage == "hull" && (code.entryRole == "hull-control-point" ||
                                         code.entryRole == "hull-patch-constant")));
            if (code.id != expectedId++ || !code.pipelineId || code.pipelineId > maxCodeVersions ||
                code.pso.size() != 32 || code.pso.find_first_not_of("0123456789abcdef") != std::string::npos ||
                (!compute && !graphics) || !code.simdWidth || code.version < 1 ||
                code.tier > 2 || (code.cache != "hit" && code.cache != "miss" && code.cache != "unknown") ||
                (code.shaderId && (code.shaderId->size() != 64 ||
                 code.shaderId->find_first_not_of("0123456789abcdef") != std::string::npos)) ||
                (code.artifactId && (code.artifactId->size() != 64 ||
                 code.artifactId->find_first_not_of("0123456789abcdef") != std::string::npos)) ||
                (parsed.shaders.schemaVersion == 1 && (code.shaderId || code.artifactId)) ||
                (compute && (code.entryRole ||
                 code.version > 2 || code.tier < 1 ||
                 code.cache == "unknown" || (code.cache == "hit" && (code.tier != 2 || code.version != 1)) ||
                 (code.version == 2 && code.tier != 2)))) {
                error = "Invalid code-version metadata";
                return false;
            }
            const auto signature = std::make_pair(code.pso, code.simdWidth);
            const auto pipeline = pipelines.try_emplace(code.pipelineId, signature).first;
            if (!identities.emplace(code.pipelineId, code.entryRole.value_or(code.stage), code.version).second ||
                pipeline->second != signature) {
                error = "Inconsistent pipeline or duplicate code-version identity";
                return false;
            }
        }
        if (parsed.shaders.codeSelections) {
            expectedId = 1;
            for (const auto& selection : *parsed.shaders.codeSelections) {
                if (selection.id != expectedId++ || !selection.pipelineId ||
                    selection.pipelineId > maxCodeVersions || selection.codeVersionIds.size() > maxSelectionEntries) {
                    error = "Invalid code-selection metadata";
                    return false;
                }
                std::set<uint64_t> references;
                for (const auto id : selection.codeVersionIds) {
                    if (!id || id > parsed.shaders.codeVersions.size() || !references.insert(id).second ||
                        parsed.shaders.codeVersions[id - 1].pipelineId != selection.pipelineId ||
                        parsed.shaders.codeVersions[id - 1].stage == "compute") {
                        error = "Invalid code-selection entry reference";
                        return false;
                    }
                }
            }
        }
    }
    for (const auto& event : parsed.trace.traceEvents) {
        const auto& args = event.args;
        const CodeSelection* selection = nullptr;
        if (args.codeSelectionId) {
            if (!parsed.shaders.codeSelections || !*args.codeSelectionId ||
                *args.codeSelectionId > parsed.shaders.codeSelections->size()) {
                error = "Invalid trace code-selection reference";
                return false;
            }
            selection = &parsed.shaders.codeSelections->at(*args.codeSelectionId - 1);
            if (args.pipelineId != selection->pipelineId) {
                error = "Trace does not match referenced code selection";
                return false;
            }
            for (const auto id : selection->codeVersionIds) {
                const auto& code = parsed.shaders.codeVersions[id - 1];
                if (args.pso != code.pso || args.simdWidth != code.simdWidth) {
                    error = "Trace does not match selected pipeline";
                    return false;
                }
            }
        }
        if (!args.codeVersionId && (!args.pipelineId || selection)) continue;
        if (!args.pipelineId || !args.codeVersionId || *args.codeVersionId == 0 ||
            *args.codeVersionId > parsed.shaders.codeVersions.size()) {
            error = "Invalid trace code-version reference";
            return false;
        }
        const auto& code = parsed.shaders.codeVersions[*args.codeVersionId - 1];
        if ((selection && std::find(selection->codeVersionIds.begin(), selection->codeVersionIds.end(),
                                   *args.codeVersionId) == selection->codeVersionIds.end()) ||
            *args.pipelineId != code.pipelineId || args.pso != code.pso || args.stage != code.stage ||
            args.simdWidth != code.simdWidth || args.tier != code.tier || args.cache != code.cache) {
            error = "Trace does not match referenced code version";
            return false;
        }
    }
    if (manifest.schemaVersion == 2 || artifactsEntry) {
        if (!artifactsEntry || !manifest.captureOptions.artifactBytes || !manifest.captureOptions.sourceBytes ||
            !*manifest.captureOptions.artifactBytes || !*manifest.captureOptions.sourceBytes ||
            *manifest.captureOptions.artifactBytes > 64ull * 1024 * 1024 ||
            *manifest.captureOptions.sourceBytes > 16ull * 1024 * 1024 ||
            !extract(zip, "artifacts.json", artifactsEntry->uncompressedSize, json, error)) {
            if (error.empty()) error = "Missing or invalid artifact budget/payload";
            return false;
        }
        if (json.size() != artifactsEntry->uncompressedSize || artifactDigest(json) != artifactsEntry->digest) {
            error = "Artifact size or digest mismatch";
            return false;
        }
        parsed.artifacts.emplace();
        if (const auto ec = glz::read<glz::opts{.error_on_missing_keys = true}>(*parsed.artifacts, json)) {
            error = "Invalid artifact metadata: " + glz::format_error(ec, json);
            return false;
        }
        if (!validateArtifacts(*parsed.artifacts, parsed.shaders, manifest.captureOptions, error)) return false;
    }
    if (manifest.schemaVersion == 3 || spatialEntry) {
        if (!spatialEntry || !manifest.captureOptions.spatialBytes ||
            *manifest.captureOptions.spatialBytes < sizeof(SpatialRecord) ||
            *manifest.captureOptions.spatialBytes > 16ull * 1024 * 1024 ||
            !extract(zip, "spatial.json", spatialEntry->uncompressedSize, json, error)) {
            if (error.empty()) error = "Missing or invalid spatial budget/payload";
            return false;
        }
        if (json.size() != spatialEntry->uncompressedSize || artifactDigest(json) != spatialEntry->digest) {
            error = "Spatial size or digest mismatch";
            return false;
        }
        parsed.spatial.emplace();
        if (const auto ec = glz::read<glz::opts{.error_on_missing_keys = true}>(*parsed.spatial, json)) {
            error = "Invalid spatial metadata: " + glz::format_error(ec, json);
            return false;
        }
        if (!validateSpatial(*parsed.spatial, parsed.shaders, manifest.captureOptions, error) ||
            !validateSpatialReferences(*parsed.spatial, parsed.trace, manifest.droppedEvents == 0, error)) return false;
    }
    if (countersEntry) {
        if (!extract(zip, "counters.json", countersEntry->uncompressedSize, json, error)) return false;
        if (json.size() != countersEntry->uncompressedSize || artifactDigest(json) != countersEntry->digest) {
            error = "Counter size or digest mismatch";
            return false;
        }
        parsed.counters.emplace();
        if (const auto ec = glz::read<glz::opts{.error_on_missing_keys = true}>(*parsed.counters, json)) {
            error = "Invalid counter metadata: " + glz::format_error(ec, json);
            return false;
        }
        if (!validateCounters(*parsed.counters, parsed.shaders, manifest.captureOptions, error) ||
            !validateCounterReferences(*parsed.counters, parsed.trace, manifest.droppedEvents == 0, error))
            return false;
    }
    if (manifest.schemaVersion >= 3 && manifest.complete != (manifest.droppedEvents == 0 &&
        (!parsed.spatial || parsed.spatial->complete) && (!parsed.counters || parsed.counters->complete))) {
        error = "Optional observations/event completeness disagrees with manifest";
        return false;
    }
    profile = std::move(parsed);
    return true;
}

} // namespace nacelle::profile
