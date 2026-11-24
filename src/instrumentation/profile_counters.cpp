#include "profile_counters.h"
#include <algorithm>
#include <charconv>
#include <map>
#include <set>
#include <string_view>

namespace nacelle::profile {
namespace {
constexpr const char* regions[] = {"compute-worker", "pixel-worker", "compute-direct"};
bool unsignedValue(const std::string& text, uint64_t& value) {
    if (text.empty() || (text.size() > 1 && text.front() == '0')) return false;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
}
}

CounterState::CounterState(uint64_t bytes, const CpuCounterProvider& source)
    : provider(source), bytes_(bytes), capacity_(bytes / sizeof(CounterRecord)),
      records_(std::make_unique<CounterRecord[]>(capacity_)) {}

CounterRecord* CounterState::reserve(uint64_t event, uint64_t version, CpuCounterRegion region) {
    const auto index = attempted_.fetch_add(1, std::memory_order_relaxed);
    if (index >= capacity_) return nullptr;
    auto& record = records_[index];
    record.eventId = event;
    record.codeVersionId = version;
    record.region = region;
    return &record;
}

bool CounterState::complete() const {
    const auto attempted = attempted_.load(std::memory_order_relaxed);
    return attempted && attempted <= capacity_ &&
        std::all_of(records_.get(), records_.get() + attempted, [](const auto& record) {
            return record.result.status == CpuCounterStatus::Available;
        });
}

const char* CounterState::failureReason() const {
    const auto attempted = attempted_.load(std::memory_order_relaxed);
    if (!attempted) return "unsupported-scope";
    for (size_t index = 0; index < std::min<uint64_t>(attempted, capacity_); ++index)
        if (records_[index].result.status != CpuCounterStatus::Available)
            return cpuCounterStatusName(records_[index].result.status);
    return attempted > capacity_ ? "record-budget-exhausted" : "available";
}

CounterMetadata CounterState::metadata() const {
    CounterMetadata result;
    result.provider = provider.name();
    const auto capabilities = provider.capabilities();
    result.providerImplemented = capabilities.implemented;
    result.metricMask = capabilities.metricMask;
    result.recordBytes = bytes_;
    result.attempted = attempted_.load(std::memory_order_relaxed);
    result.recorded = std::min<uint64_t>(result.attempted, capacity_);
    result.dropped = result.attempted - result.recorded;
    result.complete = complete();
    result.records.reserve(result.recorded);
    for (size_t index = 0; index < result.recorded; ++index) {
        const auto& record = records_[index];
        const auto& value = record.result;
        result.unavailable += value.status != CpuCounterStatus::Available;
        result.records.push_back({record.eventId, record.codeVersionId,
            regions[static_cast<uint32_t>(record.region)], cpuCounterStatusName(value.status),
            value.nativeError, std::to_string(value.cycles), std::to_string(value.instructions),
            std::to_string(value.timeEnabledNs), std::to_string(value.timeRunningNs)});
    }
    result.availability = result.complete ? "available" :
        !result.attempted ? "unsupported-scope" :
        result.recorded == result.unavailable ? "unavailable" : "partial";
    return result;
}

bool validateCounters(const CounterMetadata& counters, const ShaderMetadata& shaders,
                      const CaptureOptions& options, std::string& error) {
    auto reject = [&] { error = "Invalid counter metadata, availability, budget or code attribution"; return false; };
    if (counters.schemaVersion != 1 ||
        (counters.provider != "linux-perf-event" && counters.provider != "unsupported") ||
        counters.providerImplemented != (counters.provider == "linux-perf-event") ||
        counters.metricMask != (counters.providerImplemented ? 3u : 0u) ||
        counters.cycleUnit != "cycles" || counters.instructionUnit != "instructions" ||
        counters.timeUnit != "ns" || counters.scope != "thread-user-region" ||
        counters.sampling != "none" || counters.scaling != "raw-with-enabled-running" ||
        !counters.kernelExcluded || !counters.hypervisorExcluded || counters.inherited ||
        !options.counterBytes || *options.counterBytes < sizeof(CounterRecord) ||
        *options.counterBytes > maxCounterBytes || counters.recordBytes != *options.counterBytes ||
        counters.recorded != counters.records.size() ||
        counters.recorded > counters.recordBytes / sizeof(CounterRecord) ||
        counters.attempted < counters.recorded ||
        counters.dropped != counters.attempted - counters.recorded ||
        counters.unavailable > counters.recorded ||
        counters.attempted > 9007199254740991ull ||
        counters.complete != (counters.attempted > 0 && counters.dropped == 0 && counters.unavailable == 0) ||
        counters.availability != (counters.complete ? "available" : !counters.attempted ? "unsupported-scope" :
            counters.recorded == counters.unavailable ? "unavailable" : "partial"))
        return reject();
    uint64_t unavailable = 0;
    std::set<uint64_t> events;
    for (const auto& record : counters.records) {
        uint64_t cycles, instructions, enabled, running;
        if (!record.eventId || record.eventId > 9007199254740991ull || !events.insert(record.eventId).second ||
            !record.codeVersionId || record.codeVersionId > shaders.codeVersions.size() ||
            !unsignedValue(record.cycles, cycles) || !unsignedValue(record.instructions, instructions) ||
            !unsignedValue(record.timeEnabledNs, enabled) || !unsignedValue(record.timeRunningNs, running) ||
            record.nativeError < 0) return reject();
        const auto& code = shaders.codeVersions[record.codeVersionId - 1];
        if ((record.region == "pixel-worker" && code.stage != "pixel") ||
            ((record.region == "compute-worker" || record.region == "compute-direct") && code.stage != "compute") ||
            std::find(std::begin(regions), std::end(regions), record.region) == std::end(regions))
            return reject();
        bool known = false;
        for (unsigned status = 0; status <= static_cast<unsigned>(CpuCounterStatus::UnsupportedScope); ++status)
            known |= record.status == cpuCounterStatusName(static_cast<CpuCounterStatus>(status));
        if (!known) return reject();
        if (record.status == "available") {
            if (counters.provider != "linux-perf-event" || record.nativeError ||
                !enabled || !running || running > enabled) return reject();
        } else {
            ++unavailable;
            if (cycles || instructions || enabled || running) return reject();
        }
    }
    return unavailable == counters.unavailable || reject();
}

bool validateCounterReferences(const CounterMetadata& counters, const Trace& trace,
                               bool eventsComplete, std::string& error) {
    std::map<uint64_t, const CounterSample*> references;
    for (const auto& record : counters.records) references.emplace(record.eventId, &record);
    for (const auto& event : trace.traceEvents) {
        const auto found = references.find(event.args.id);
        if (found == references.end()) continue;
        const auto& record = *found->second;
        const char* name = record.region == "compute-worker" ? "Compute worker" :
                           record.region == "pixel-worker" ? "Raster worker" : "Compute direct";
        if (event.ph != "X" || event.name != name || event.args.codeVersionId != record.codeVersionId) {
            error = "Counter record disagrees with its exact recorded region/code version";
            return false;
        }
        references.erase(found);
    }
    if (eventsComplete && !references.empty()) {
        error = "Counter record references a missing trace region";
        return false;
    }
    return true;
}
}
