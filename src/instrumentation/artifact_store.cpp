#include "artifacts.h"
#include <algorithm>
#include <cstdio>
#include <set>

namespace {
std::mutex storeMutex;
std::shared_ptr<nacelle::profile::ArtifactStore> retained;
}

namespace nacelle::profile {
std::shared_ptr<ArtifactStore> artifactStore() {
    std::lock_guard lock(storeMutex);
    return retained;
}

void retainArtifacts(const std::shared_ptr<ArtifactStore>& store, std::string owner,
                     std::vector<TextArtifact> payloads, bool overflow, bool append) {
    if (!store) return;
    std::lock_guard lock(store->mutex);
    store->overflow |= overflow;
    if (store->overflow) return;
    if (store->groups.contains(owner) && !append) return;
    if (append) {
        const auto found = store->groups.find(owner);
        if (found != store->groups.end())
            std::erase_if(payloads, [&](const auto& payload) {
                return std::any_of(found->second.begin(), found->second.end(),
                    [&](const auto& existing) { return existing.kind == payload.kind; });
            });
    }
    uint64_t sources = 0, artifacts = 0;
    for (auto& payload : payloads) {
        payload.owner = owner;
        payload.id = owner + "-" + payload.kind + "-" + artifactDigest(payload.text);
        uint64_t bytes = payload.text.size() + payload.id.size() + payload.owner.size() +
            payload.kind.size() + payload.label.size() + payload.mappings.size() * sizeof(ArtifactMapping);
        for (const auto& mapping : payload.mappings) bytes += mapping.source.size();
        (payload.kind == "hlsl" ? sources : artifacts) += bytes;
    }
    if ((!store->groups.contains(owner) && store->groups.size() >= 4096) ||
        sources > store->options.source_bytes - store->sourceBytes ||
        artifacts > store->options.artifact_bytes - store->artifactBytes) {
        store->overflow = true;
        return;
    }
    store->sourceBytes += sources;
    store->artifactBytes += artifacts;
    std::set<std::string> unique;
    std::erase_if(payloads, [&](const auto& payload) { return !unique.insert(payload.id).second; });
    auto& group = store->groups[owner];
    for (auto& payload : payloads) group.push_back(std::move(payload));
}
}

extern "C" HRESULT NacelleConfigureProfileArtifacts(const NacelleProfileArtifactOptions* options) {
    if (!options || options->struct_size != sizeof(*options) || options->version != 1 ||
        (options->flags & ~3u) ||
        ((options->flags & NACELLE_PROFILE_EMBED_SHADER_SOURCES) && !(options->flags & NACELLE_PROFILE_RETAIN_SHADER_TEXT)) ||
        !options->artifact_bytes || options->artifact_bytes > 64ull * 1024 * 1024 ||
        !options->source_bytes || options->source_bytes > 16ull * 1024 * 1024) {
        std::fprintf(stderr, "[NacelleProfile] Invalid artifact retention options\n");
        return E_INVALIDARG;
    }
    try {
        auto next = options->flags ? std::make_shared<nacelle::profile::ArtifactStore>(*options) : nullptr;
        std::lock_guard lock(storeMutex);
        retained = std::move(next);
        return S_OK;
    } catch (const std::bad_alloc&) {
        std::fprintf(stderr, "[NacelleProfile] Could not allocate artifact store\n");
        return E_OUTOFMEMORY;
    }
}
