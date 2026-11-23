#include "artifacts.h"
#define XXH_INLINE_ALL
#include <xxhash.h>
#include <algorithm>
#include <iomanip>
#include <set>
#include <sstream>

namespace nacelle::profile {
std::string artifactDigest(std::string_view text) {
    std::ostringstream stream;
    stream << std::hex << std::setfill('0') << std::setw(16) << XXH64(text.data(), text.size(), 0);
    return stream.str();
}

bool validateArtifacts(const ArtifactMetadata& metadata, const ShaderMetadata& shaders,
                       const CaptureOptions& options, std::string& error) {
    std::map<std::string, const TextArtifact*> payloads;
    std::map<std::string, size_t> lineCounts;
    uint64_t sources = 0, artifacts = 0;
    if (metadata.schemaVersion != 1 || metadata.payloads.size() > 4096 ||
        metadata.selections.size() != shaders.codeVersions.size()) {
        error = "Unsupported artifact schema or selection/payload count";
        return false;
    }
    for (const auto& payload : metadata.payloads) {
        const bool source = payload.kind == "hlsl";
        if (payload.owner.size() != 64 || payload.owner.find_first_not_of("0123456789abcdef") != std::string::npos ||
            (source && !options.embeddedSources) ||
            (payload.kind != "hlsl" && payload.kind != "dxil" && payload.kind != "mlir" &&
             payload.kind != "llvm" && payload.kind != "assembly") ||
            payload.label.empty() || payload.label.size() > 128 ||
            payload.label.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-") != std::string::npos ||
            payload.id != payload.owner + "-" + payload.kind + "-" + artifactDigest(payload.text) ||
            !payloads.emplace(payload.id, &payload).second || payload.mappings.size() > 65536) {
            error = "Invalid artifact identity, label, kind or source consent";
            return false;
        }
        (source ? sources : artifacts) += payload.text.size();
        lineCounts.emplace(payload.id, 1 + std::count(payload.text.begin(), payload.text.end(), '\n'));
    }
    if (!options.sourceBytes || !options.artifactBytes ||
        sources > *options.sourceBytes || artifacts > *options.artifactBytes) {
        error = "Artifact/source text exceeds declared retention budget";
        return false;
    }
    for (const auto& payload : metadata.payloads) {
        const auto lines = lineCounts.at(payload.id);
        for (const auto& mapping : payload.mappings) {
            const auto source = payloads.find(mapping.source);
            if (!mapping.line || mapping.line > lines ||
                (!mapping.source.empty() && (source == payloads.end() || source->second->kind != "hlsl" ||
                 source->second->owner != payload.owner || !mapping.sourceLine ||
                 mapping.sourceLine > lineCounts.at(mapping.source))) ||
                (mapping.source.empty() && mapping.sourceLine) ||
                (mapping.offset.has_value() != mapping.size.has_value()) ||
                (mapping.size && (!*mapping.size || *mapping.offset > maxProfileBytes ||
                                 *mapping.size > maxProfileBytes - *mapping.offset))) {
                error = "Invalid artifact source/instruction mapping";
                return false;
            }
        }
    }
    std::set<std::string> used;
    for (size_t index = 0; index < metadata.selections.size(); ++index) {
        const auto& selection = metadata.selections[index];
        const auto& code = shaders.codeVersions[index];
        if (selection.codeVersionId != index + 1 || selection.payloadIds.size() > 64 ||
            selection.mappingAvailability != "partial" || selection.laneAvailability != "unavailable" ||
            (selection.shaderAvailability != "available" && selection.shaderAvailability != "not-retained") ||
            (selection.nativeAvailability != "available" && selection.nativeAvailability != "not-retained" &&
             selection.nativeAvailability != "tier1-unavailable")) {
            error = "Invalid artifact availability or code-version selection";
            return false;
        }
        if (selection.nativeAvailability == "tier1-unavailable" && (code.tier != 1 || code.artifactId)) {
            error = "Tier-1 unavailable status disagrees with code version";
            return false;
        }
        std::set<std::string> references;
        bool shader = false, native = false;
        for (const auto& id : selection.payloadIds) {
            const auto found = payloads.find(id);
            const auto nativeKind = found != payloads.end() &&
                (found->second->kind == "assembly" || found->second->kind == "llvm");
            if (found == payloads.end() || !references.insert(id).second ||
                found->second->owner != (nativeKind ? code.artifactId : code.shaderId)) {
                error = "Artifact does not belong to the selected code version";
                return false;
            }
            used.insert(id);
            shader |= !nativeKind;
            native |= nativeKind;
        }
        if (shader != (selection.shaderAvailability == "available") ||
            native != (selection.nativeAvailability == "available")) {
            error = "Artifact availability disagrees with selected payloads";
            return false;
        }
    }
    if (used.size() != payloads.size()) {
        error = "Unreferenced artifact payload";
        return false;
    }
    return true;
}
}
