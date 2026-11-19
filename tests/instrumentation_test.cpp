#include <nacelle/instrumentation.h>
#include "instrumentation/capture.h"
#include "instrumentation/profile_format.h"

#include <miniz.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <iterator>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct Capture {
    NacelleProfileCapture* value = nullptr;
    ~Capture() { NacelleDestroyProfileCapture(value); }
    void begin(uint64_t bytes = 1024 * 1024) {
        NacelleProfileOptions options;
        options.event_buffer_bytes = bytes;
        require(NacelleBeginProfileCapture(&options, &value) == S_OK, "Begin capture");
    }
};

nacelle::profile::Profile read(const std::filesystem::path& path) {
    nacelle::profile::Profile result;
    std::string error;
    require(nacelle::profile::readProfile(path.string(), result, error), error.c_str());
    return result;
}

uint64_t manifestSize(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    require(static_cast<bool>(input), "Open profile for manifest inspection");
    const std::string bytes((std::istreambuf_iterator<char>(input)),
                            std::istreambuf_iterator<char>());
    require(!input.bad(), "Read profile for manifest inspection");
    mz_zip_archive zip{};
    require(mz_zip_reader_init_mem(&zip, bytes.data(), bytes.size(), 0),
            "Open ZIP for manifest inspection");
    const int index = mz_zip_reader_locate_file(&zip, "manifest.json", nullptr, 0);
    mz_zip_archive_file_stat stat{};
    const bool found = index >= 0 &&
        mz_zip_reader_file_stat(&zip, static_cast<mz_uint>(index), &stat);
    mz_zip_reader_end(&zip);
    require(found, "Find manifest for size inspection");
    return stat.m_uncomp_size;
}
}

int main(int argc, char** argv) {
    const auto directory = std::filesystem::temp_directory_path() /
        ("nacelle-profile-test-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        require(std::filesystem::create_directory(directory), "Create test directory");
        {
            Capture capture;
            capture.begin();
            NacelleProfileCapture* duplicate = nullptr;
            NacelleProfileOptions options;
            require(FAILED(NacelleBeginProfileCapture(&options, &duplicate)), "Reject concurrent capture");
            require(!duplicate, "Clear failed begin output");
            std::thread worker;
            {
                nacelle::profile::Scope root("Compile test shader");
                root.setAttributes({1, 2, 8, nacelle::profile::Stage::Compute, 1,
                                    nacelle::profile::Cache::Miss});
                worker = std::thread([context = root.context()] {
                    nacelle::profile::Scope child(context, "Worker test shader");
                    nacelle::profile::Scope nested("Nested test stage");
                });
            }
            NacelleProfileSummary summary;
            const auto path = directory / "complete.nacelle-profile";
            require(NacelleEndProfileCapture(capture.value, path.string().c_str(), &summary) == S_OK,
                    "End waits for captured async work");
            worker.join();
            auto profile = read(path);
            require(summary.recorded_events == 3 && summary.complete, "All nested events recorded");
            const auto& events = profile.trace.traceEvents;
            const auto root = std::find_if(events.begin(), events.end(), [](const auto& event) {
                return event.name == "Compile test shader";
            });
            require(root != events.end(), "Root event exists");
            require(std::count_if(events.begin(), events.end(), [&](const auto& event) {
                return event.args.parent == root->args.id;
            }) == 1, "Async child retains parent");
            require(events.front().args.pso == "00000000000000020000000000000001",
                    "Persistent PSO identity");
            require(FAILED(NacelleEndProfileCapture(capture.value, path.string().c_str(), &summary)),
                    "Refuse to overwrite existing bundle");
            require(read(path).manifest.complete, "Existing bundle preserved");
            if (argc == 2) {
                const std::string destination = argv[1];
                std::filesystem::copy_file(path,
                    std::filesystem::path(std::u8string(destination.begin(), destination.end())));
            }
        }
        {
            Capture capture;
            capture.begin(sizeof(nacelle::profile::Event));
            for (int i = 0; i < 10; ++i) {
                NACELLE_PROFILE_SCOPE("Overflow test");
            }
            NacelleProfileSummary summary;
            const auto path = directory / "overflow.nacelle-profile";
            require(NacelleEndProfileCapture(capture.value, path.string().c_str(), &summary) == S_FALSE,
                    "Overflow explicitly incomplete");
            auto profile = read(path);
            require(!profile.manifest.complete && summary.recorded_events == 1 &&
                    summary.dropped_events == 9, "Bounded recorder reports exact drops");
        }
        {
            for (int session = 0; session < 2; ++session) {
                Capture capture;
                capture.begin();
                std::vector<std::jthread> workers;
                {
                    nacelle::profile::Scope root("Concurrent root");
                    const auto context = root.context();
                    for (int worker = 0; worker < 4; ++worker) {
                        workers.emplace_back([context] {
                            nacelle::profile::Scope child(context, "Concurrent worker");
                            for (int i = 0; i < 97; ++i) {
                                nacelle::profile::Scope outer("Concurrent outer");
                                nacelle::profile::Scope inner("Concurrent inner");
                            }
                        });
                    }
                }
                NacelleProfileSummary summary;
                const auto path = directory /
                    ("concurrent-" + std::to_string(session) + ".nacelle-profile");
                require(NacelleEndProfileCapture(capture.value, path.string().c_str(), &summary) == S_OK,
                        "Drain concurrent events across repeated sessions");
                for (auto& worker : workers) worker.join();
                const auto profile = read(path);
                constexpr uint64_t expected = 1 + 4 * (1 + 2 * 97);
                require(summary.recorded_events == expected &&
                        profile.manifest.recordedEvents == expected &&
                        profile.trace.traceEvents.size() == expected,
                        "Count every completed concurrent event, including partial blocks");
                std::set<uint64_t> ids;
                for (const auto& event : profile.trace.traceEvents) {
                    require(event.args.id != 0 && ids.insert(event.args.id).second,
                            "Concurrent event IDs are nonzero and unique");
                }
                size_t roots = 0;
                for (const auto& event : profile.trace.traceEvents) {
                    if (event.args.parent == 0) ++roots;
                    else require(ids.contains(event.args.parent), "Concurrent parent ID exists");
                }
                require(roots == 1, "Concurrent scopes retain one root");
            }
        }
        {
            Capture capture;
            capture.begin(65 * sizeof(nacelle::profile::Event));
            std::vector<std::jthread> workers;
            for (int worker = 0; worker < 4; ++worker) {
                workers.emplace_back([] {
                    for (int i = 0; i < 130; ++i) {
                        NACELLE_PROFILE_SCOPE("Concurrent overflow");
                    }
                });
            }
            for (auto& worker : workers) worker.join();
            NacelleProfileSummary summary;
            const auto path = directory / "concurrent-overflow.nacelle-profile";
            require(NacelleEndProfileCapture(capture.value, path.string().c_str(), &summary) == S_FALSE,
                    "Concurrent overflow explicitly incomplete");
            const auto profile = read(path);
            require(summary.recorded_events == 65 && summary.dropped_events == 455 &&
                    profile.manifest.recordedEvents == 65 &&
                    profile.manifest.droppedEvents == 455 &&
                    profile.trace.traceEvents.size() == 65,
                    "Exact concurrent accounting with a partially available final block");
        }
        {
            Capture capture;
            capture.begin();
            NacelleProfileSummary summary;
            const auto path = directory / "empty.nacelle-profile";
            require(NacelleEndProfileCapture(capture.value, path.string().c_str(), &summary) == S_OK,
                    "End empty capture");
            const auto profile = read(path);
            require(summary.recorded_events == 0 && summary.dropped_events == 0 &&
                    profile.trace.traceEvents.empty(), "Empty capture has no stale events");
        }
        {
            Capture cancelled;
            cancelled.begin();
            NACELLE_PROFILE_SCOPE("Cancelled event");
        }
        {
            Capture capture;
            capture.begin();
            {
                NACELLE_PROFILE_SCOPE("Write error test");
            }
            NacelleProfileSummary summary;
            const auto invalid = directory / "missing" / "capture.nacelle-profile";
            require(FAILED(NacelleEndProfileCapture(capture.value, invalid.string().c_str(), &summary)),
                    "Write failures reported");
            const auto retry = directory / "retry.nacelle-profile";
            require(NacelleEndProfileCapture(capture.value, retry.string().c_str(), &summary) == S_OK,
                    "Stopped capture can retry output");
            require(read(retry).manifest.recordedEvents == 1, "Repeated sessions have no stale records");
        }
        {
            constexpr uint64_t manifestLimit = 1024 * 1024;
            nacelle::profile::Session session(sizeof(nacelle::profile::Event));
            session.wait();
            std::string error;
            const auto baseline = directory / "metadata-baseline.nacelle-profile";
            require(nacelle::profile::writeProfile(baseline.string(), session, error),
                    error.c_str());
            const auto overhead = manifestSize(baseline);
            require(overhead < manifestLimit, "Baseline manifest fits documented limit");
            session.compilerHash.assign(static_cast<size_t>(manifestLimit - overhead), 'a');
            const auto boundary = directory / "metadata-boundary.nacelle-profile";
            require(nacelle::profile::writeProfile(boundary.string(), session, error),
                    error.c_str());
            require(manifestSize(boundary) == manifestLimit &&
                    read(boundary).manifest.build.compilerHash == session.compilerHash,
                    "Exactly 1 MiB manifest round-trips");

            session.compilerHash.push_back('a');
            const auto oversized = directory / "metadata-oversized.nacelle-profile";
            error.clear();
            require(!nacelle::profile::writeProfile(oversized.string(), session, error) &&
                    error.find("manifest") != std::string::npos &&
                    !std::filesystem::exists(oversized),
                    "Reject a manifest one byte over the limit without creating a bundle");

            session.compilerHash.assign(static_cast<size_t>(manifestLimit / 2), '"');
            const auto escaped = directory / "metadata-escaped.nacelle-profile";
            error.clear();
            require(!nacelle::profile::writeProfile(escaped.string(), session, error) &&
                    error.find("manifest") != std::string::npos &&
                    !std::filesystem::exists(escaped),
                    "Manifest limit includes JSON escaping expansion");
        }
        {
            Capture capture;
            capture.begin();
            {
                NACELLE_PROFILE_SCOPE("Oversized metadata");
                nacelle::profile::setCompilerConfiguration(std::string(1024 * 1024, 'a'),
                                                           "", "", {});
            }
            NacelleProfileSummary summary;
            const auto path = directory / "metadata-api-error.nacelle-profile";
            require(FAILED(NacelleEndProfileCapture(capture.value, path.string().c_str(), &summary)) &&
                    !std::filesystem::exists(path),
                    "Public API reports oversized metadata instead of saving an unreadable bundle");
        }
        {
            const auto path = directory / "corrupt.nacelle-profile";
            std::ofstream(path, std::ios::binary) << "not a ZIP archive";
            nacelle::profile::Profile profile;
            std::string error;
            require(!nacelle::profile::readProfile(path.string(), profile, error) && !error.empty(),
                    "Reject corrupt archive explicitly");
        }
        NacelleProfileOptions invalidOptions;
        invalidOptions.flags = 1;
        NacelleProfileCapture* invalid = nullptr;
        require(FAILED(NacelleBeginProfileCapture(&invalidOptions, &invalid)), "Reject unsupported options");
        std::filesystem::remove_all(directory);
        std::cout << "Capture lifecycle, async correlation, overflow, write errors, and corrupt input: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "instrumentation_test: " << error.what() << "\nArtifacts: " << directory << '\n';
        return 1;
    }
}
