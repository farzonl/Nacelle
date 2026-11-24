#include "instrumentation/profile_format.h"
#include <CLI/CLI.hpp>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <new>

int main(int argc, char** argv) {
    CLI::App app{"Inspect an offline Nacelle profile without executing captured code"};
    std::string input, output;
    app.add_option("profile", input, "Input .nacelle-profile")->required()->check(CLI::ExistingFile);
    app.add_option("--trace", output, "Export Chrome Trace JSON (must not already exist)");
    CLI11_PARSE(app, argc, argv);
    try {
        nacelle::profile::Profile profile;
        std::string error;
        if (!nacelle::profile::readProfile(input, profile, error)) {
            std::cerr << "nacelle-dbg: " << error << '\n';
            return 1;
        }
        if (!output.empty() && !nacelle::profile::exportTrace(output, profile.trace, error)) {
            std::cerr << "nacelle-dbg: " << error << '\n';
            return 1;
        }
        std::cout << (profile.manifest.complete ? "Complete" : "INCOMPLETE")
                  << " profile: " << profile.manifest.recordedEvents
                  << " events, " << profile.manifest.droppedEvents << " dropped\n";
        if (profile.counters) {
            const auto& counters = *profile.counters;
            std::cout << "CPU counters (" << counters.provider << "): " << counters.availability
                      << "; " << counters.recorded << " regions, " << counters.unavailable
                      << " unavailable, " << counters.dropped << " dropped\n";
            for (size_t index = 0; index < std::min<size_t>(8, counters.records.size()); ++index) {
                const auto& record = counters.records[index];
                std::cout << "  event " << record.eventId << ", code " << record.codeVersionId
                          << ": " << record.status << " (native error " << record.nativeError
                          << "), cycles " << record.cycles << ", instructions " << record.instructions
                          << ", enabled/running " << record.timeEnabledNs << "/" << record.timeRunningNs << " ns\n";
            }
        }
        return profile.manifest.complete ? 0 : 2;
    } catch (const std::bad_alloc&) {
        std::cerr << "nacelle-dbg: insufficient memory to inspect profile\n";
    } catch (const std::filesystem::filesystem_error& error) {
        std::cerr << "nacelle-dbg: " << error.what() << '\n';
    }
    return 1;
}
