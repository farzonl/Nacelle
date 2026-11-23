#include "spatial.h"
#include <algorithm>
#include <set>
#include <tuple>
#include <map>

namespace nacelle::profile {
SpatialState::SpatialState(const NacelleProfileSpatialOptions& value)
    : options(value), capacity(std::min<size_t>(maxSpatialRecords, value.record_bytes / sizeof(SpatialRecord))),
      records(std::make_unique<SpatialRecord[]>(capacity)),
      counters(std::make_unique<SpatialCounters[]>(maxSpatialOperations)) {
    operations.reserve(maxSpatialOperations);
}

uint64_t SpatialState::addOperation(SpatialOperation operation) {
    std::lock_guard lock(operationMutex);
    if (operations.size() == maxSpatialOperations) {
        operationOverflow = true;
        return 0;
    }
    operation.id = operations.size() + 1;
    operations.push_back(std::move(operation));
    return operations.size();
}

SpatialRecord* SpatialState::reserve(uint64_t operation, std::array<uint32_t, 3> xyz) {
    auto& counts = counters[operation - 1];
    bool sampled = true;
    for (size_t axis = 0; axis < 3; ++axis)
        sampled &= xyz[axis] >= options.min_xyz[axis] && xyz[axis] < options.max_xyz[axis] &&
                   (xyz[axis] - options.min_xyz[axis]) % options.sample_stride == 0;
    if (!sampled) {
        counts.skipped.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    const auto slot = next.fetch_add(1, std::memory_order_relaxed);
    if (slot >= capacity) {
        counts.dropped.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    auto& record = records[slot];
    record.operation = operation;
    record.xyz = xyz;
    counts.recorded.fetch_add(1, std::memory_order_relaxed);
    return &record;
}

bool SpatialState::complete() const {
    return !operationOverflow.load() && next.load() <= capacity;
}

namespace {
bool add(uint64_t& value, uint64_t increment) {
    // Keep JSON integers exact in both native and JavaScript readers.
    constexpr uint64_t limit = (1ull << 53) - 1;
    if (increment > limit - value) return false;
    value += increment;
    return true;
}
bool product(const std::array<uint32_t, 3>& dimensions, uint64_t& result) {
    result = 1;
    for (const auto value : dimensions) {
        if (!value || value > ((1ull << 53) - 1) / result) return false;
        result *= value;
    }
    return true;
}
}

bool SpatialState::aggregate(SpatialMetadata& result, std::string& error) const {
    if (operationOverflow) {
        error = "Spatial operation limit exceeded (4096)";
        return false;
    }
    result.recordBytes = options.record_bytes;
    result.flags = options.flags;
    result.sampleStride = options.sample_stride;
    std::copy_n(options.min_xyz, 3, result.minXYZ.begin());
    std::copy_n(options.max_xyz, 3, result.maxXYZ.begin());
    result.complete = complete();
    result.operations = operations;
    std::vector<const SpatialRecord*> ordered;
    ordered.reserve(std::min<uint64_t>(capacity, next.load()));
    for (size_t index = 0; index < std::min<uint64_t>(capacity, next.load()); ++index)
        ordered.push_back(&records[index]);
    std::sort(ordered.begin(), ordered.end(), [](const auto* a, const auto* b) {
        return std::tie(a->operation, a->xyz) < std::tie(b->operation, b->xyz);
    });
    for (auto& operation : result.operations) {
        const auto& count = counters[operation.id - 1];
        operation.recorded = count.recorded.load();
        operation.skipped = count.skipped.load();
        operation.dropped = count.dropped.load();
        operation.complete = operation.dropped == 0;
        operation.missingStatus = operation.availability != "available" ? "unavailable" :
                                  operation.dropped ? "dropped-or-unsampled" : "unsampled";
        if (!add(result.recorded, operation.recorded) || !add(result.skipped, operation.skipped) ||
            !add(result.dropped, operation.dropped)) {
            error = "Spatial record counts exceed exact integer range";
            return false;
        }
    }
    for (const auto* record : ordered) {
        auto& operation = result.operations[record->operation - 1];
        if (operation.cells.empty() || operation.cells.back().xyz != record->xyz) {
            SpatialCell cell;
            cell.xyz = record->xyz;
            cell.status = operation.complete ? "observed" : "partial";
            if (operation.stage == "pixel") {
                cell.coveredLanes.resize(operation.simdWidth);
                cell.companionLanes.resize(operation.simdWidth);
                cell.paddingLanes.resize(operation.simdWidth);
            }
            operation.cells.push_back(std::move(cell));
        }
        auto& cell = operation.cells.back();
        bool valid = add(cell.calls, 1) && add(cell.wallNs, record->wallNs) && add(cell.invocations, record->invocations);
        for (size_t lane = 0; lane < cell.coveredLanes.size(); ++lane)
            valid &= add(cell.coveredLanes[lane], record->covered[lane]) &&
                     add(cell.companionLanes[lane], record->companion[lane]) &&
                     add(cell.paddingLanes[lane], record->padding[lane]);
        if (!valid) {
            error = "Spatial aggregation exceeds exact integer range";
            return false;
        }
    }
    return true;
}

bool validateSpatial(const SpatialMetadata& metadata, const ShaderMetadata& shaders,
                     const CaptureOptions& options, std::string& error) {
    const auto fail = [&](const char* message) { error = message; return false; };
    if (metadata.schemaVersion != 1 || !metadata.flags || (metadata.flags & ~3u) || !options.spatialBytes ||
        metadata.recordBytes != *options.spatialBytes || metadata.recordBytes < sizeof(SpatialRecord) ||
        metadata.recordBytes > 16ull * 1024 * 1024 || !metadata.sampleStride ||
        metadata.operations.size() > maxSpatialOperations ||
        metadata.complete != (metadata.dropped == 0))
        return fail("Invalid spatial schema, budget, count or completeness");
    for (size_t axis = 0; axis < 3; ++axis)
        if (metadata.minXYZ[axis] >= metadata.maxXYZ[axis]) return fail("Invalid spatial ROI");
    uint64_t recorded = 0, skipped = 0, dropped = 0;
    size_t cells = 0;
    for (size_t index = 0; index < metadata.operations.size(); ++index) {
        const auto& operation = metadata.operations[index];
        if (operation.id != index + 1 || !operation.eventId || !operation.codeVersionId ||
            operation.codeVersionId > shaders.codeVersions.size() || operation.timeUnit != "ns" ||
            operation.laneSemantics != "entry-submission" ||
            operation.complete != (operation.dropped == 0))
            return fail("Invalid spatial operation identity or metric semantics");
        const auto& code = shaders.codeVersions[operation.codeVersionId - 1];
        uint64_t size = 0, extent = 0, threads = 0;
        const bool pixel = operation.stage == "pixel";
        if (operation.stage != code.stage || operation.pso != code.pso || operation.pipelineId != code.pipelineId ||
            operation.simdWidth != code.simdWidth || (operation.stage != "compute" && !pixel) ||
            !product(operation.dimensions, size) || !product(operation.extent, extent) ||
            (pixel ? operation.simdWidth != 4 && operation.simdWidth != 8 && operation.simdWidth != 16 :
                     !product(operation.threadsPerGroup, threads)) ||
            operation.granularity != (pixel ? "raster-tile" : "threadgroup") ||
            !(metadata.flags & (pixel ? NACELLE_PROFILE_SPATIAL_PIXEL : NACELLE_PROFILE_SPATIAL_COMPUTE)) ||
            operation.laneLayout != (pixel ? "quad-major-xy" : "unavailable") ||
            operation.orientation != (pixel ? "top-left-y-down" : "group-xyz") ||
            (operation.availability != "available" && operation.availability != "dispatch-wide-entry-unavailable") ||
            (operation.availability != "available" && (pixel || operation.recorded || operation.skipped || operation.dropped)) ||
            operation.missingStatus != (operation.availability != "available" ? "unavailable" :
                                        operation.dropped ? "dropped-or-unsampled" : "unsampled"))
            return fail("Invalid spatial dimensions, version attribution or availability");
        for (size_t axis = 0; axis < 3; ++axis)
            if (operation.cellExtent[axis] != (pixel && axis < 2 ? 32u : 1u) ||
                operation.dimensions[axis] != (uint64_t(operation.extent[axis]) + operation.cellExtent[axis] - 1) /
                                             operation.cellExtent[axis])
                return fail("Spatial cell extents disagree with dimensions");
        if (!add(recorded, operation.recorded) || !add(skipped, operation.skipped) || !add(dropped, operation.dropped))
            return fail("Spatial counters exceed exact integer range");
        std::set<std::array<uint32_t, 3>> coordinates;
        uint64_t calls = 0;
        for (const auto& cell : operation.cells) {
            if (++cells > maxSpatialRecords || cell.status != (operation.complete ? "observed" : "partial") ||
                !cell.calls || !add(calls, cell.calls) ||
                cell.wallNs > (1ull << 53) - 1 || cell.invocations > (1ull << 53) - 1 ||
                !coordinates.insert(cell.xyz).second ||
                cell.coveredLanes.size() != (pixel ? operation.simdWidth : 0) ||
                cell.companionLanes.size() != cell.coveredLanes.size() ||
                cell.paddingLanes.size() != cell.coveredLanes.size())
                return fail("Invalid spatial cell counts, coordinates or lane slots");
            for (size_t axis = 0; axis < 3; ++axis)
                if (cell.xyz[axis] >= operation.dimensions[axis] || cell.xyz[axis] < metadata.minXYZ[axis] ||
                    cell.xyz[axis] >= metadata.maxXYZ[axis] ||
                    (cell.xyz[axis] - metadata.minXYZ[axis]) % metadata.sampleStride)
                    return fail("Spatial cell outside dimensions or sampling selection");
            if (!pixel) {
                if (cell.calls > ((1ull << 53) - 1) / threads || cell.invocations != cell.calls * threads)
                    return fail("Spatial compute invocation count disagrees with threadgroup dimensions");
            } else {
                uint64_t invocations = 0;
                for (size_t lane = 0; lane < operation.simdWidth; ++lane) {
                    uint64_t submissions = 0;
                    if (!add(submissions, cell.coveredLanes[lane]) || !add(submissions, cell.companionLanes[lane]) ||
                        !add(submissions, cell.paddingLanes[lane]) || submissions < cell.calls ||
                        cell.invocations % operation.simdWidth || submissions != cell.invocations / operation.simdWidth)
                        return fail("Spatial lane submissions disagree with SIMD batches");
                    if (!add(invocations, cell.coveredLanes[lane]) || !add(invocations, cell.companionLanes[lane]) ||
                        !add(invocations, cell.paddingLanes[lane]))
                        return fail("Spatial lane counts exceed exact integer range");
                }
                if (invocations != cell.invocations) return fail("Spatial submitted lane totals disagree");
            }
        }
        if (calls != operation.recorded) return fail("Spatial observed calls disagree with recorded count");
    }
    if (recorded != metadata.recorded || skipped != metadata.skipped || dropped != metadata.dropped ||
        recorded > metadata.recordBytes / sizeof(SpatialRecord))
        return fail("Spatial totals exceed record pool or disagree with operations");
    return true;
}

bool validateSpatialReferences(const SpatialMetadata& metadata, const Trace& trace,
                               bool eventsComplete, std::string& error) {
    std::map<uint64_t, const SpatialOperation*> references;
    for (const auto& operation : metadata.operations)
        if (!references.emplace(operation.eventId, &operation).second) {
            error = "Duplicate spatial operation event";
            return false;
        }
    for (const auto& event : trace.traceEvents) {
        const auto found = references.find(event.args.id);
        if (found == references.end()) continue;
        const auto& operation = *found->second;
        if (event.args.pipelineId != operation.pipelineId || event.args.codeVersionId != operation.codeVersionId) {
            error = "Spatial operation disagrees with its recorded span";
            return false;
        }
        references.erase(found);
    }
    if (eventsComplete && !references.empty()) {
        error = "Spatial operation references a missing span in complete event data";
        return false;
    }
    return true;
}
}
