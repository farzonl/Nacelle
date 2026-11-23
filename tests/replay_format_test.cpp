#include "../src/instrumentation/replay_format.h"
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace nacelle::replay;
namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
uint32_t append(Bundle& bundle, const std::string& bytes, const char* role) {
    const auto id = static_cast<uint32_t>(bundle.metadata.blobs.size() + 1);
    bundle.metadata.blobs.push_back({id, bundle.bytes.size(), bytes.size(), role, contentDigest(bytes)});
    bundle.bytes += bytes;
    return id;
}
}
int main() {
    const auto path = std::filesystem::temp_directory_path() /
        ("nacelle-replay-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        require(contentDigest({}) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "SHA-256 empty vector");
        require(contentDigest(std::string("abc")) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            "SHA-256 abc vector");
        const std::array<std::pair<size_t, const char*>, 5> boundaries{{
            {55, "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318"},
            {56, "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a"},
            {63, "7d3e74a05d7db15bce4ad9ec0658ea98e3f06eeecf16b4c6fff2da457ddc2f34"},
            {64, "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb"},
            {65, "635361c48bb9eab14198e76ea8ab7f1a41685d6ad62aa9146d301d4f17eb0ae0"}}};
        for (const auto& [size, expected] : boundaries)
            require(contentDigest(std::string(size, 'a')) == expected, "SHA-256 padding boundary");
        Bundle bundle;
        auto& environment = bundle.manifest.environment;
        environment.simdWidth = 4;
        environment.pointerBits = 64;
        environment.build.compiler = environment.build.revision = environment.build.cpu =
            environment.build.compilerHash = environment.build.targetTriple =
            environment.build.operatingSystem = environment.build.architecture = "fixture";
        const auto initial = append(bundle, "inpt", "initial");
        const auto expected = append(bundle, "outp", "expected");
        const auto root = append(bundle, "root", "root-signature");
        const auto shader = append(bundle, "dxil", "dxil");
        bundle.metadata.resources.push_back({1, 1, 4, 3, 4, initial, expected});
        bundle.metadata.roots.push_back({1, root});
        bundle.metadata.pipelines.push_back({1, 1, shader, 0, 0, contentDigest(std::string("dxil")),
            contentDigest(std::string("native"))});
        bundle.metadata.commands = {{"submission", {}}, {"pipeline", {1}}, {"root", {1}}, {"dispatch", {1, 1, 1}}};
        std::string error;
        require(validateBundle(bundle, error), error.c_str());
        require(writeBundle(path.string(), bundle, error), error.c_str());
        require(!writeBundle(path.string(), bundle, error), "Exclusive output must not overwrite a replay");
        std::ifstream input(path, std::ios::binary);
        const std::string bytes((std::istreambuf_iterator<char>(input)), {});
        input.close();
        Bundle parsed;
        require(readBundleBytes(bytes, parsed, error), error.c_str());
        require(parsed.bytes == bundle.bytes && parsed.metadata.commands.size() == 4, "Replay round trip");
        for (size_t size : {size_t(0), size_t(1), size_t(29), bytes.size() / 2, bytes.size() - 1})
            require(!readBundleBytes(std::span(bytes.data(), size), parsed, error), "Reject truncated replay archive");
        auto malformed = bundle;
        malformed.bytes[0] ^= 1;
        require(!validateBundle(malformed, error), "Reject corrupt snapshot");
        malformed = bundle;
        malformed.metadata.resources[0].initialBlob = 0;
        require(!validateBundle(malformed, error), "Reject missing snapshot");
        malformed = bundle;
        malformed.metadata.blobs[0].offset = UINT64_MAX;
        require(!validateBundle(malformed, error), "Reject wrapped blob bounds");
        malformed = bundle;
        malformed.manifest.snapshotBytes = 7;
        require(!validateBundle(malformed, error), "Measure both initial and expected snapshot budget");
        malformed = bundle;
        malformed.manifest.environment.build.operatingSystem.clear();
        require(!validateBundle(malformed, error), "Reject incomplete execution environment");
        malformed = bundle;
        malformed.manifest.sourceConsent = true;
        malformed.manifest.shaderConsent = false;
        require(!validateBundle(malformed, error), "Source consent is not shader consent");
        malformed = bundle;
        malformed.metadata.commands.back().data[0] = 65536;
        require(!validateBundle(malformed, error), "Reject dispatch beyond required XYZ limit");
        std::filesystem::remove(path);
        std::cout << "replay_format_test passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::filesystem::remove(path);
        std::cerr << "replay_format_test: " << error.what() << '\n';
        return 1;
    }
}
