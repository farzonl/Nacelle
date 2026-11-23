#include <nacelle/replay.h>
#include <CLI/CLI.hpp>
#include <iostream>

int main(int argc, char** argv) {
    CLI::App app{"Validate a local replay; execution requires explicit trust (this is not a sandbox)"};
    std::string input;
    bool trusted = false;
    uint32_t repetitions = 3;
    app.add_option("bundle", input, "Local .nacelle-replay file")->required()->check(CLI::ExistingFile);
    app.add_flag("--execute-trusted-local", trusted, "Consent to executing trusted local shader binaries");
    app.add_option("--repetitions", repetitions, "Fresh-state output checks")->check(CLI::Range(1, 100));
    CLI11_PARSE(app, argc, argv);
    if (FAILED(NacelleReplayBundle(input.c_str(), trusted ? 1 : 0, repetitions))) return 1;
    std::cout << (trusted ? "Replay outputs matched across " + std::to_string(repetitions) + " fresh-state runs" :
        "Replay validated; no captured code executed") << '\n';
    return 0;
}
