#pragma once

#include "profile_format.h"
#include <memory>
#include <mutex>
#include <map>
#include <nacelle/instrumentation.h>

namespace nacelle::profile {

struct ArtifactStore {
    explicit ArtifactStore(NacelleProfileArtifactOptions value) : options(value) {}
    const NacelleProfileArtifactOptions options;
    std::mutex mutex;
    uint64_t artifactBytes = 0, sourceBytes = 0;
    bool overflow = false;
    std::map<std::string, std::vector<TextArtifact>> groups;
};

std::shared_ptr<ArtifactStore> artifactStore();
std::string artifactDigest(std::string_view text);
void retainArtifacts(const std::shared_ptr<ArtifactStore>& store, std::string owner,
                     std::vector<TextArtifact> payloads, bool overflow = false, bool append = false);
bool validateArtifacts(const ArtifactMetadata& metadata, const ShaderMetadata& shaders,
                       const CaptureOptions& options, std::string& error);

} // namespace nacelle::profile
