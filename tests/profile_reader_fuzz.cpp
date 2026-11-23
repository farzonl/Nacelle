#include "../src/instrumentation/profile_format.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    nacelle::profile::Profile profile;
    std::string error;
    nacelle::profile::readProfileBytes(
        {reinterpret_cast<const char*>(data), size}, profile, error);
    return 0;
}
