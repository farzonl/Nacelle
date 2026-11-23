#include "../src/instrumentation/replay_format.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    nacelle::replay::Bundle bundle;
    std::string error;
    nacelle::replay::readBundleBytes({reinterpret_cast<const char*>(data), size}, bundle, error);
    return 0;
}
