#include "instrumentation/profile_format.h"
#include <CLI/CLI.hpp>

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
        return profile.manifest.complete ? 0 : 2;
    } catch (const std::bad_alloc&) {
        std::cerr << "nacelle-dbg: insufficient memory to inspect profile\n";
    } catch (const std::filesystem::filesystem_error& error) {
        std::cerr << "nacelle-dbg: " << error.what() << '\n';
    }
    return 1;
}
