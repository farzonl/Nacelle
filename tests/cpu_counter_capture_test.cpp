#include "instrumentation/capture.h"
#include "instrumentation/profile_counters.h"
#include <nacelle/instrumentation.h>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <thread>

using namespace nacelle::profile;

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
class MockProvider final : public CpuCounterProvider {
public:
    explicit MockProvider(CpuCounterStatus status) : status_(status) {}
    const char* name() const noexcept override {
        return status_ == CpuCounterStatus::UnsupportedProvider ? "unsupported" : "linux-perf-event";
    }
    CpuCounterCapabilities capabilities() const noexcept override {
        return status_ == CpuCounterStatus::UnsupportedProvider ? CpuCounterCapabilities{}
            : CpuCounterCapabilities{true, 3};
    }
    CpuCounterResult begin(CpuCounterGroup&) const noexcept override {
        return status_ == CpuCounterStatus::PermissionDenied || status_ == CpuCounterStatus::UnsupportedProvider
            ? CpuCounterResult{status_, status_ == CpuCounterStatus::PermissionDenied ? 13 : 0}
            : CpuCounterResult{CpuCounterStatus::Available};
    }
    CpuCounterResult end(CpuCounterGroup&) const noexcept override {
        return status_ == CpuCounterStatus::Available ? CpuCounterResult{status_, 0, 123, 456, 120, 60}
            : CpuCounterResult{status_, status_ == CpuCounterStatus::IoFailure ? 5 : 0};
    }
private:
    CpuCounterStatus status_;
};

std::shared_ptr<Session> execute(const CpuCounterProvider* provider, uint64_t bytes, size_t workers) {
    auto session = std::make_shared<Session>(1024 * 1024);
    if (provider) session->counters = std::make_unique<CounterState>(bytes, *provider);
    Context owner{std::make_shared<Work>(session)};
    {
        Scope root(owner, "Compute dispatch");
        root.setAttributes({1, 2, 8, Stage::Compute});
        root.setComputeVersion(1, 1, 1, Cache::Miss);
        const auto context = root.context();
        std::vector<std::thread> threads;
        for (size_t index = 0; index < workers; ++index)
            threads.emplace_back([context] {
                Scope worker(context, "Compute worker");
                worker.startCounters(CpuCounterRegion::ComputeWorker);
            });
        for (auto& thread : threads) thread.join();
    }
    owner = {};
    session->wait();
    return session;
}

void roundTrip(const CpuCounterProvider& provider, size_t workers, uint64_t bytes, bool complete) {
    const auto session = execute(&provider, bytes, workers);
    const auto metadata = session->counters->metadata();
    require(metadata.complete == complete && metadata.attempted == workers, "Counter completion/coverage");
    require(session->eventCount == workers + 1 && session->dropped == 0, "Counter failure changed wall-time execution");
    const auto path = std::filesystem::temp_directory_path() /
        ("nacelle-cpu-counter-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".nacelle-profile");
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() { std::error_code error; std::filesystem::remove(path, error); }
    } cleanup{path};
    std::string error;
    require(writeProfile(path.string(), *session, error), error.c_str());
    Profile parsed;
    require(readProfile(path.string(), parsed, error), error.c_str());
    require(parsed.manifest.schemaVersion == 4 && parsed.counters &&
        parsed.manifest.complete == complete && parsed.counters->complete == complete, "Counter archive/completeness");
    auto corrupted = *parsed.counters;
    corrupted.records[0].codeVersionId = 999;
    require(!validateCounters(corrupted, parsed.shaders, parsed.manifest.captureOptions, error), "Bad counter version accepted");
    corrupted = *parsed.counters;
    corrupted.records[0].cycles = "18446744073709551616";
    require(!validateCounters(corrupted, parsed.shaders, parsed.manifest.captureOptions, error), "Lossy counter integer accepted");
    corrupted = *parsed.counters;
    corrupted.records[0].region = "pixel-worker";
    require(!validateCounters(corrupted, parsed.shaders, parsed.manifest.captureOptions, error), "Wrong counter stage accepted");
    corrupted = *parsed.counters;
    corrupted.records[0].eventId = 900000;
    require(!validateCounterReferences(corrupted, parsed.trace, true, error), "Missing counter span accepted");
    if (complete) {
        require(parsed.counters->records[0].timeEnabledNs == "120" &&
            parsed.counters->records[0].timeRunningNs == "60", "Multiplexing times lost");
        corrupted = *parsed.counters;
        corrupted.records[0].timeRunningNs = "121";
        require(!validateCounters(corrupted, parsed.shaders, parsed.manifest.captureOptions, error), "Impossible counter timing accepted");
    } else {
        require(metadata.dropped || metadata.unavailable, "Unavailable counters looked successful");
    }
}

void measure() {
    constexpr size_t regions = 1024, rounds = 24;
    uint64_t expected = 0;
    for (size_t round = 0; round < rounds; ++round) {
        for (size_t offset = 0; offset < 2; ++offset) {
            const bool enabled = (round + offset) % 2;
            auto session = std::make_shared<Session>(1024 * 1024);
            const auto allocationStart = std::chrono::steady_clock::now();
            if (enabled) session->counters = std::make_unique<CounterState>(1024 * 1024);
            const auto allocationEnd = std::chrono::steady_clock::now();
            Context owner{std::make_shared<Work>(session)};
            uint64_t checksum = 1;
            int64_t setupNs = 0, bodyNs = 0, finishNs = 0;
            const auto started = std::chrono::steady_clock::now();
            {
                Scope root(owner, "Compute dispatch");
                root.setAttributes({1, 2, 8, Stage::Compute});
                root.setComputeVersion(1, 1, 1, Cache::Miss);
                for (size_t region = 0; region < regions; ++region) {
                    std::chrono::steady_clock::time_point completed;
                    {
                        Scope worker("Compute worker");
                        const auto setupStart = std::chrono::steady_clock::now();
                        worker.startCounters(CpuCounterRegion::ComputeWorker);
                        const auto setupEnd = std::chrono::steady_clock::now();
                        for (size_t index = 0; index < 8192; ++index)
                            checksum = checksum * 1664525 + 1013904223;
                        completed = std::chrono::steady_clock::now();
                        setupNs += std::chrono::duration_cast<std::chrono::nanoseconds>(setupEnd - setupStart).count();
                        bodyNs += std::chrono::duration_cast<std::chrono::nanoseconds>(completed - setupEnd).count();
                    }
                    finishNs += std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - completed).count();
                }
            }
            const auto ended = std::chrono::steady_clock::now();
            owner = {};
            session->wait();
            require(session->eventCount == regions + 1 && !session->dropped, "Measured counter execution loss");
            if (!expected) expected = checksum;
            require(checksum == expected, "Measured counter checksum mismatch");
            const auto ns = [](auto duration) {
                return std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
            };
            const auto counter = session->counters ? session->counters->metadata() : CounterMetadata{};
            const auto path = std::filesystem::temp_directory_path() /
                ("nacelle-cpu-cost-" + std::to_string(started.time_since_epoch().count()) + ".nacelle-profile");
            struct Cleanup {
                std::filesystem::path path;
                ~Cleanup() { std::error_code error; std::filesystem::remove(path, error); }
            } cleanup{path};
            std::string error;
            const auto saveStart = std::chrono::steady_clock::now();
            require(writeProfile(path.string(), *session, error), error.c_str());
            const auto saveEnd = std::chrono::steady_clock::now();
            Profile verified;
            require(readProfile(path.string(), verified, error), error.c_str());
            require(verified.manifest.recordedEvents == regions + 1, "Measured archive event loss");
            std::cout << "{\"round\":" << round << ",\"enabled\":" << enabled
                << ",\"allocation_ns\":" << ns(allocationEnd - allocationStart)
                << ",\"setup_ns\":" << setupNs << ",\"body_ns\":" << bodyNs << ",\"finish_ns\":" << finishNs
                << ",\"save_ns\":" << ns(saveEnd - saveStart)
                << ",\"region_ns\":" << ns(ended - started) << ",\"events\":" << session->eventCount
                << ",\"checksum\":\"" << checksum << "\",\"available\":" << counter.complete
                << ",\"unavailable\":" << counter.unavailable << "}\n";
        }
    }
}
}

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--measure") { measure(); return 0; }
        require(argc == 1, "Unexpected counter test arguments");
        for (auto status : {CpuCounterStatus::Available, CpuCounterStatus::PermissionDenied,
                            CpuCounterStatus::UnsupportedProvider, CpuCounterStatus::NotRunning,
                            CpuCounterStatus::IoFailure}) {
            const MockProvider mock(status);
            roundTrip(mock, 4, 4096, status == CpuCounterStatus::Available);
        }
        const MockProvider available(CpuCounterStatus::Available);
        roundTrip(available, 4, sizeof(CounterRecord), false);
        const auto disabled = execute(nullptr, 0, 4);
        require(!disabled->counters && disabled->eventCount == 5, "Counter-disabled behavior changed");
        const CounterState noRegions(64, available);
        require(!noRegions.complete() && noRegions.metadata().availability == "unsupported-scope",
            "No supported regions falsely certified");
        NacelleProfileCounterOptions options;
        options.flags = 2;
        require(NacelleConfigureProfileCounters(&options) == E_INVALIDARG, "Unknown counter flags accepted");
        options.flags = 1;
        options.record_bytes = 63;
        require(NacelleConfigureProfileCounters(&options) == E_INVALIDARG, "Undersized counter pool accepted");
        options.record_bytes = 64;
        require(NacelleConfigureProfileCounters(&options) == S_OK, "Minimum counter pool rejected");
        NacelleProfileOptions captureOptions;
        NacelleProfileCapture* handle = nullptr;
        require(NacelleBeginProfileCapture(&captureOptions, &handle) == S_OK, "Counter Begin failed");
        require(NacelleConfigureProfileCounters(&options) == E_UNEXPECTED, "Active counter reconfiguration allowed");
        NacelleDestroyProfileCapture(handle);
        options.flags = 0;
        require(NacelleConfigureProfileCounters(&options) == S_OK, "Disable counters failed");
        std::cout << "Synthetic counter providers, bounded allocation, versioned native roundtrip, failure preservation: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "cpu_counter_capture_test: " << error.what() << '\n';
        return 1;
    }
}
