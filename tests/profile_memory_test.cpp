#include "common/bench_memory.h"
#include "instrumentation/capture.h"
#include "instrumentation/profile_format.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <new>
#include <stdexcept>
#include <thread>
#if defined(_WIN32)
#include <aclapi.h>
#include <sddl.h>
#include <io.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {
thread_local bool trackAllocations = false;
thread_local uint64_t requestedBytes = 0;
std::string failureStage;
std::atomic<unsigned> activeStage{0};
std::atomic<bool> sampling{true};
std::array<std::atomic<double>, 32> observedPeaks{};
const auto started = std::chrono::steady_clock::now();

void check(bool condition, const std::string &error) {
    if (!condition)
        throw std::runtime_error(error);
}

void snapshot(const char *stage, size_t retainedBytes = 0) {
    const auto memory = nacelle_bench::queryProcessMemory();
    check(memory.currentRssMb > 0 && memory.peakRssMb > 0, "RSS measurement is unavailable");
    const auto index = activeStage.fetch_add(1);
    check(index + 1 < observedPeaks.size(), "Diagnostic stage limit exceeded");
    std::cout << "{\"stage\":\"" << stage << "\",\"rssMiB\":" << memory.currentRssMb
              << ",\"processPeakMiB\":" << memory.peakRssMb
              << ",\"previousStageObservedPeakMiB\":" << observedPeaks[index].load()
              << ",\"retainedWorkspaceBytes\":" << retainedBytes << ",\"elapsedMs\":"
              << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count()
              << "}\n";
}
} // namespace

void* operator new(size_t bytes) {
    if (trackAllocations) requestedBytes += bytes;
    if (void* memory = std::malloc(bytes ? bytes : 1)) return memory;
    throw std::bad_alloc();
}
void* operator new[](size_t bytes) { return ::operator new(bytes); }
void operator delete(void* memory) noexcept { std::free(memory); }
void operator delete[](void* memory) noexcept { std::free(memory); }
void operator delete(void* memory, size_t) noexcept { std::free(memory); }
void operator delete[](void* memory, size_t) noexcept { std::free(memory); }

namespace nacelle::profile {
void inspectProfileStaging(FILE* file) {
#if defined(_WIN32)
    const auto handle = reinterpret_cast<HANDLE>(_get_osfhandle(_fileno(file)));
    PSECURITY_DESCRIPTOR descriptor = nullptr;
    PACL dacl = nullptr;
    check(GetSecurityInfo(handle, SE_FILE_OBJECT, DACL_SECURITY_INFORMATION,
                          nullptr, nullptr, &dacl, nullptr, &descriptor) == ERROR_SUCCESS,
          "Cannot inspect staging ACL");
    std::unique_ptr<void, decltype(&LocalFree)> security(descriptor, &LocalFree);
    check(dacl && dacl->AceCount == 1, "Staging ACL must contain only owner rights");
    void* ace = nullptr;
    check(GetAce(dacl, 0, &ace), "Cannot inspect staging owner ACE");
    auto* allowed = static_cast<ACCESS_ALLOWED_ACE*>(ace);
    check(allowed->Header.AceType == ACCESS_ALLOWED_ACE_TYPE, "Unexpected staging permission");
    LPWSTR sid = nullptr;
    check(ConvertSidToStringSidW(&allowed->SidStart, &sid), "Cannot inspect staging SID");
    std::unique_ptr<void, decltype(&LocalFree)> sidStorage(sid, &LocalFree);
    check(std::wstring(sid) == L"S-1-3-4", "Staging access is not restricted to its owner");
#else
    struct stat info{};
    check(fstat(fileno(file), &info) == 0 && (info.st_mode & 0777) == 0600 && info.st_nlink == 0,
          "Staging must be owner-only and unlinked");
#endif
}
void observeProfileTiming(const char* stage, uint64_t nanoseconds) {
    std::cout << "{\"timing\":\"" << stage << "\",\"durationNs\":" << nanoseconds << "}\n";
}
void observeProfileMemory(const char *stage, size_t retainedBytes) {
    snapshot(stage, retainedBytes);
    if (failureStage == stage) {
        failureStage.clear();
        // A test-only exception models transient failure at a phase boundary.
        throw std::bad_alloc();
    }
}
} // namespace nacelle::profile

int main(int argc, char **argv) {
    using namespace nacelle::profile;
    try {
        check(argc == 3 || argc == 4, "Expected budget MiB, workload and optional failure phase");
        const uint64_t budget = std::stoull(argv[1]) * 1024 * 1024;
        check(budget && budget <= 64ull * 1024 * 1024, "Budget must be 1..64 MiB");
        const std::string workload(argv[2]);
        check(workload == "near-full" || workload == "overflow" || workload == "threads" ||
                  workload == "metadata" || workload == "versions",
              "Unknown workload");
        if (argc == 4)
            failureStage = argv[3];
        const auto directory =
            std::filesystem::temp_directory_path() /
            ("nacelle-memory-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        check(std::filesystem::create_directory(directory), "Could not create diagnostic output directory");
        struct Cleanup {
            std::filesystem::path path;
            ~Cleanup() {
                std::error_code error;
                std::filesystem::remove_all(path, error);
            }
        } cleanup{directory};
        std::jthread sampler([] {
            while (sampling.load()) {
                const auto index = activeStage.load();
                const auto memory = nacelle_bench::queryProcessMemory();
                auto previous = observedPeaks[index].load();
                while (previous < memory.currentRssMb &&
                       !observedPeaks[index].compare_exchange_weak(previous, memory.currentRssMb)) {
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        });
        struct StopSampling {
            ~StopSampling() { sampling = false; }
        } stopSampling;
        snapshot("initial");
        auto owned = std::make_shared<Session>(budget);
        auto &session = *owned;
        snapshot("event-allocation", session.capacity * sizeof(Event));
        session.shaders.codeVersions.push_back(
            {1, 1, "00000000000000020000000000000001", "compute", 8, 1, 2, "hit"});
        std::vector<Context> contexts;
        size_t count = session.capacity - 13;
        if (workload == "threads") {
            uint64_t contextAllocationBytes = 0;
            // Real short-lived producers retain escaping contexts and strand their final blocks.
            for (unsigned i = 0; i < 512; ++i) {
                std::thread worker([&] {
                    requestedBytes = 0;
                    trackAllocations = true;
                    Event event{};
                    event.id = owned->allocateEventId();
                    std::strcpy(event.name, "Short-lived worker");
                    owned->record(event);
                    contexts.push_back({std::make_shared<Work>(owned), event.id, {}});
                    trackAllocations = false;
                    contextAllocationBytes += requestedBytes;
                });
                worker.join();
            }
            snapshot("retained-contexts",
                     contexts.capacity() * sizeof(Context) + contexts.size() * sizeof(Work));
            const auto live = nacelle_bench::queryProcessMemory();
            contexts.clear();
            owned->wait();
            count = owned->eventCount;
            const auto output = directory / "threads.nacelle-profile";
            std::string error;
            check(writeProfile(output.string(), *owned, error), error);
            snapshot("saved");
            Profile profile;
            check(readProfile(output.string(), profile, error), error);
            check(profile.manifest.recordedEvents == count &&
                      profile.manifest.droppedEvents == owned->dropped &&
                      profile.manifest.complete == (owned->dropped == 0),
                  "Short-lived-thread accounting changed");
            std::cout << "{\"workload\":\"threads\",\"events\":" << count << ",\"dropped\":" << owned->dropped
                      << ",\"retainedRssMiB\":" << live.currentRssMb
                      << ",\"cumulativeContextAllocationBytes\":" << contextAllocationBytes << "}\n";
            return 0;
        }
        if (workload == "metadata" || workload == "versions") {
            session.shaders.schemaVersion = 2;
            session.shaders.codeVersions.clear();
            session.shaders.codeSelections.emplace();
            requestedBytes = 0;
            trackAllocations = true;
            for (size_t i = 0; i < (workload == "versions" ? maxCodeVersions : maxCodeVersions / 2); ++i) {
                const auto id = i + 1;
                session.pipelineIds.emplace(id, id);
                session.codeVersionIds.emplace(std::make_tuple(id, Stage::Pixel, EntryRole::Default, 2), id);
                session.shaders.codeVersions.push_back({id, id, "00000000000000020000000000000001", "pixel",
                                                        8, 2, 2, "hit", "pixel", std::string(64, 'a'),
                                                        std::string(64, 'b')});
                if (workload == "metadata") {
                    session.shaders.codeSelections->push_back({id, id, {id}});
                    session.codeSelectionParents.push_back(0);
                }
            }
            trackAllocations = false;
            snapshot("version-selection-bookkeeping",
                     session.shaders.codeVersions.capacity() * sizeof(CodeVersion) +
                         session.shaders.codeSelections->capacity() * sizeof(CodeSelection) +
                         session.codeSelectionParents.capacity() * sizeof(uint64_t));
        }
        const auto drops = workload == "overflow" ? 4096 : 0;
        if (drops)
            count = session.capacity;
        for (size_t i = 0; i < count; ++i) {
            Event event{};
            event.id = i + 1;
            event.startNs = (count - i) * 1000 + 500;
            event.durationNs = 1500;
            event.thread = static_cast<uint32_t>(i % 8 + 1);
            event.attributes = {1, 2,          8, (workload == "metadata" || workload == "versions") ? Stage::Pixel : Stage::Compute,
                                2, Cache::Hit, 1, 1};
            std::strcpy(event.name, "Version-tracked worker");
            session.events[i] = event;
        }
        session.nextBlock = session.capacity;
        session.dropped = drops;
        snapshot("recorded");
        session.wait();
        snapshot("drained");
        const auto output = directory / "capture.nacelle-profile";
        std::string error;
        bool failed = false;
        try {
            check(writeProfile(output.string(), session, error), error);
        } catch (const std::bad_alloc &) {
            failed = true;
            check(argc == 4 && !std::filesystem::exists(output), "Failure left a partial output");
        }
        if (argc == 4) {
            check(failed && failureStage.empty(), "Requested failure phase was not reached");
            sampling = false;
            sampler.join();
            activeStage = 0;
            check(writeProfile(output.string(), session, error), "Stopped session could not retry: " + error);
        }
        snapshot("saved");
        sampling = false;
        if (sampler.joinable())
            sampler.join();
        Profile profile;
        check(readProfile(output.string(), profile, error), error);
        check(profile.manifest.recordedEvents == count && profile.manifest.droppedEvents == drops &&
                  profile.manifest.complete == (drops == 0) && profile.trace.traceEvents.size() == count,
              "Serialization changed event/drop/completeness accounting");
        if (workload == "metadata" || workload == "versions")
            check(profile.shaders.codeVersions.size() + profile.shaders.codeSelections->size() ==
                      maxCodeVersions,
                  "Maximum metadata did not round-trip");
        std::cout << "{\"workload\":\"" << workload << "\",\"eventBytes\":" << sizeof(Event)
                  << ",\"sessionBytes\":" << sizeof(Session) << ",\"contextBytes\":" << sizeof(Context)
                  << ",\"workBytes\":" << sizeof(Work) << ",\"events\":" << count << ",\"dropped\":" << drops
                  << ",\"codeVersionBytes\":" << sizeof(CodeVersion)
                  << ",\"codeSelectionBytes\":" << sizeof(CodeSelection)
                  << ",\"cumulativeMetadataAllocationBytes\":" << requestedBytes
                  << ",\"archiveBytes\":" << std::filesystem::file_size(output)
                  << ",\"retryVerified\":" << (failed ? "true" : "false") << "}\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "profile_memory_test: " << error.what() << '\n';
        return 1;
    }
}
