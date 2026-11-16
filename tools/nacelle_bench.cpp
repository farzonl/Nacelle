#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

#include "bench_common.h"
#include "bench_types.h"

namespace {

using namespace nacelle_bench;

struct VkDeviceInfo {
    uint32_t index = 0;
    std::string backendName;
};

std::filesystem::path getExecutableDir(const char* argv0) {
#if defined(_WIN32)
    char buf[MAX_PATH];
    DWORD len = GetModuleFileNameA(nullptr, buf, MAX_PATH);
    if (len > 0 && len < MAX_PATH) {
        return std::filesystem::path(buf).parent_path();
    }
#elif defined(__APPLE__)
    char buf[4096];
    uint32_t sz = sizeof(buf);
    if (_NSGetExecutablePath(buf, &sz) == 0) {
        return std::filesystem::weakly_canonical(buf).parent_path();
    }
#else
    char buf[4096];
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len > 0) {
        buf[len] = '\0';
        return std::filesystem::path(buf).parent_path();
    }
#endif
    std::error_code ec;
    auto p = std::filesystem::weakly_canonical(argv0 ? argv0 : "", ec);
    return p.empty() ? std::filesystem::current_path() : p.parent_path();
}

std::string workerBinaryPath(const std::filesystem::path& binDir, const std::string& name) {
#if defined(_WIN32)
    return (binDir / (name + ".exe")).string();
#else
    return (binDir / name).string();
#endif
}

bool runSubprocess(const std::string& cmdLine, std::string& outStdout, std::string& outStatus, uint32_t timeoutSec = 60) {
    outStdout.clear();
    outStatus.clear();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeoutSec);
#if defined(_WIN32)
    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;
    HANDLE hRead = nullptr;
    HANDLE hWrite = nullptr;
    if (!CreatePipe(&hRead, &hWrite, &sa, 0)) {
        outStatus = "FAILED (CreatePipe)";
        return false;
    }
    SetHandleInformation(hRead, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOA si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = hWrite;
    si.hStdError = hWrite;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);

    PROCESS_INFORMATION pi{};
    std::string mutableCmd = cmdLine;
    BOOL ok = CreateProcessA(
        nullptr,
        mutableCmd.data(),
        nullptr,
        nullptr,
        TRUE,
        0,
        nullptr,
        nullptr,
        &si,
        &pi);
    CloseHandle(hWrite);
    if (!ok) {
        CloseHandle(hRead);
        outStatus = "FAILED (CreateProcess)";
        return false;
    }

    char buf[4096];
    bool timedOut = false;
    for (;;) {
        DWORD avail = 0;
        if (PeekNamedPipe(hRead, nullptr, 0, nullptr, &avail, nullptr) && avail > 0) {
            DWORD bytesRead = 0;
            DWORD toRead = (avail < sizeof(buf)) ? avail : static_cast<DWORD>(sizeof(buf));
            if (ReadFile(hRead, buf, toRead, &bytesRead, nullptr) && bytesRead > 0) {
                outStdout.append(buf, bytesRead);
            }
            continue;
        }
        DWORD waitRc = WaitForSingleObject(pi.hProcess, 25);
        if (waitRc == WAIT_OBJECT_0) {
            while (PeekNamedPipe(hRead, nullptr, 0, nullptr, &avail, nullptr) && avail > 0) {
                DWORD bytesRead = 0;
                DWORD toRead = (avail < sizeof(buf)) ? avail : static_cast<DWORD>(sizeof(buf));
                if (!ReadFile(hRead, buf, toRead, &bytesRead, nullptr) || bytesRead == 0) break;
                outStdout.append(buf, bytesRead);
            }
            break;
        }
        if (timeoutSec > 0 && std::chrono::steady_clock::now() >= deadline) {
            timedOut = true;
            TerminateProcess(pi.hProcess, 1);
            WaitForSingleObject(pi.hProcess, 2000);
            break;
        }
    }
    CloseHandle(hRead);
    DWORD exitCode = 0;
    GetExitCodeProcess(pi.hProcess, &exitCode);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    if (timedOut) {
        outStatus = "TIMED OUT (" + std::to_string(timeoutSec) + "s)";
        return false;
    }
    if (exitCode != 0) {
        std::ostringstream ss;
        ss << "CRASHED (0x" << std::hex << exitCode << ")";
        outStatus = ss.str();
        return false;
    }
    return true;
#else
    int pipefd[2];
    if (pipe(pipefd) != 0) {
        outStatus = "FAILED (pipe)";
        return false;
    }

    pid_t pid = fork();
    if (pid < 0) {
        close(pipefd[0]);
        close(pipefd[1]);
        outStatus = "FAILED (fork)";
        return false;
    }
    if (pid == 0) {
        setpgid(0, 0);
        dup2(pipefd[1], STDOUT_FILENO);
        dup2(pipefd[1], STDERR_FILENO);
        close(pipefd[0]);
        close(pipefd[1]);
        execl("/bin/sh", "sh", "-c", cmdLine.c_str(), static_cast<char*>(nullptr));
        _exit(127);
    }

    close(pipefd[1]);
    int flags = fcntl(pipefd[0], F_GETFL, 0);
    if (flags >= 0) {
        fcntl(pipefd[0], F_SETFL, flags | O_NONBLOCK);
    }

    char buf[4096];
    bool timedOut = false;
    bool eof = false;
    while (!eof) {
        if (timeoutSec > 0 && std::chrono::steady_clock::now() >= deadline) {
            timedOut = true;
            break;
        }
        struct pollfd pfd{};
        pfd.fd = pipefd[0];
        pfd.events = POLLIN;
        int prc = poll(&pfd, 1, 100);
        if (prc > 0) {
            for (;;) {
                ssize_t n = read(pipefd[0], buf, sizeof(buf));
                if (n > 0) {
                    outStdout.append(buf, static_cast<size_t>(n));
                } else if (n == 0) {
                    eof = true;
                    break;
                } else {
                    if (errno == EAGAIN || errno == EWOULDBLOCK) break;
                    if (errno == EINTR) continue;
                    eof = true;
                    break;
                }
            }
        } else if (prc < 0 && errno != EINTR) {
            break;
        }
    }
    close(pipefd[0]);

    int rc = 0;
    if (timedOut) {
        kill(-pid, SIGKILL);
        kill(pid, SIGKILL);
        waitpid(pid, &rc, 0);
        outStatus = "TIMED OUT (" + std::to_string(timeoutSec) + "s)";
        return false;
    }

    for (;;) {
        pid_t w = waitpid(pid, &rc, WNOHANG);
        if (w == pid) break;
        if (w < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (timeoutSec > 0 && std::chrono::steady_clock::now() >= deadline) {
            kill(-pid, SIGKILL);
            kill(pid, SIGKILL);
            waitpid(pid, &rc, 0);
            outStatus = "TIMED OUT (" + std::to_string(timeoutSec) + "s)";
            return false;
        }
        usleep(20000);
    }

    if (rc != 0) {
        if (WIFSIGNALED(rc)) {
            outStatus = "CRASHED (signal " + std::to_string(WTERMSIG(rc)) + ")";
        } else if (WIFEXITED(rc)) {
            outStatus = "CRASHED (exit " + std::to_string(WEXITSTATUS(rc)) + ")";
        } else {
            outStatus = "CRASHED";
        }
        return false;
    }
    return true;
#endif
}

// Lightweight parser for worker JSON lines (we control the exact output format)
bool parseWorkerJsonLine(const std::string& line, BenchRunResult& out) {
    auto getStr = [&](const char* key) -> std::string {
        std::string pat = std::string("\"") + key + "\":\"";
        size_t pos = line.find(pat);
        if (pos == std::string::npos) return "";
        pos += pat.size();
        std::string val;
        while (pos < line.size() && line[pos] != '"') {
            if (line[pos] == '\\' && pos + 1 < line.size()) {
                ++pos;
            }
            val.push_back(line[pos++]);
        }
        return val;
    };
    auto getNum = [&](const char* key) -> double {
        std::string pat = std::string("\"") + key + "\":";
        size_t pos = line.find(pat);
        if (pos == std::string::npos) return 0.0;
        pos += pat.size();
        return std::strtod(line.c_str() + pos, nullptr);
    };
    auto getBool = [&](const char* key) -> bool {
        std::string pat = std::string("\"") + key + "\":";
        size_t pos = line.find(pat);
        if (pos == std::string::npos) return false;
        pos += pat.size();
        return line.compare(pos, 4, "true") == 0;
    };

    out.benchId = getStr("bench_id");
    out.benchName = getStr("bench_name");
    out.category = getStr("category");
    out.backend = getStr("backend");
    out.shaderFormat = getStr("shader_format");
    out.simdWidth = static_cast<uint32_t>(getNum("simd_width"));
    out.ran = getBool("ran");
    out.status = getStr("status");
    out.compileMs = getNum("compile_ms");
    out.firstDispatchMs = getNum("first_dispatch_ms");
    out.coldMs = getNum("cold_ms");
    out.meanMs = getNum("mean_ms");
    out.medianMs = getNum("median_ms");
    out.minMs = getNum("min_ms");
    out.shortJobMs = getNum("short_job_ms");
    out.longJobMs = getNum("long_job_ms");
    out.rssBeforeMb = getNum("rss_before_mb");
    out.rssAfterCompileMb = getNum("rss_after_compile_mb");
    out.rssAfterRunMb = getNum("rss_after_run_mb");
    out.rssDeltaMb = getNum("rss_delta_mb");
    out.peakRssMb = getNum("peak_rss_mb");
    out.checksum = getNum("checksum");
    return !out.benchId.empty();
}

std::vector<BenchRunResult> parseWorkerOutput(const std::string& rawOutput) {
    std::vector<BenchRunResult> out;
    std::istringstream iss(rawOutput);
    std::string line;
    bool inJson = false;
    while (std::getline(iss, line)) {
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
        if (line == "<<<BENCH_JSON_BEGIN>>>") {
            inJson = true;
            continue;
        }
        if (line == "<<<BENCH_JSON_END>>>") {
            inJson = false;
            continue;
        }
        if (inJson && !line.empty()) {
            BenchRunResult r;
            if (parseWorkerJsonLine(line, r)) {
                out.push_back(std::move(r));
            }
        }
    }
    return out;
}

std::vector<VkDeviceInfo> listDevicesViaWorker(const std::filesystem::path& binDir, const std::string& workerName) {
    std::vector<VkDeviceInfo> out;
    std::string workerExe = workerBinaryPath(binDir, workerName);
    if (!std::filesystem::exists(workerExe)) return out;

    std::string rawOut, errStatus;
    if (!runSubprocess("\"" + workerExe + "\" --list-devices", rawOut, errStatus)) {
        return out;
    }

    std::istringstream iss(rawOut);
    std::string line;
    while (std::getline(iss, line)) {
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
        size_t tab = line.find('\t');
        if (tab == std::string::npos) continue;
        uint32_t idx = static_cast<uint32_t>(std::stoul(line.substr(0, tab)));
        std::string name = line.substr(tab + 1);
        out.push_back({idx, std::move(name)});
    }
    return out;
}

std::string resolveWorkerForBackend(const std::filesystem::path& binDir, const std::string& backendToken) {
    if (backendToken.rfind("nacelle-", 0) == 0) {
        return workerBinaryPath(binDir, "bench-worker-nacelle");
    }
    if (backendToken.rfind("vulkan", 0) == 0) {
        return workerBinaryPath(binDir, "bench-worker-vulkan");
    }
    if (backendToken == "metal") {
        return workerBinaryPath(binDir, "bench-worker-metal");
    }
    if (backendToken == "warp" || backendToken.rfind("d3d12", 0) == 0) {
        return workerBinaryPath(binDir, "bench-worker-d3d12");
    }
    return "";
}

std::vector<BenchRunResult> runIsolatedWorker(
    const std::filesystem::path& binDir,
    const std::string& benchId,
    const std::string& backendToken,
    const std::string& shaderDir,
    uint32_t width,
    uint32_t height,
    uint32_t maxSteps,
    uint32_t iters,
    uint32_t timeoutSec,
    std::string& outCrashStatus) {
    outCrashStatus.clear();
    std::string workerExe = resolveWorkerForBackend(binDir, backendToken);
    if (workerExe.empty() || !std::filesystem::exists(workerExe)) {
        outCrashStatus = "FAILED (worker binary missing)";
        return {};
    }

    std::ostringstream cmd;
    cmd << "\"" << workerExe << "\""
        << " --single-bench " << benchId
        << " --single-backend " << backendToken
        << " --shader-dir \"" << shaderDir << "\""
        << " --width " << width
        << " --height " << height
        << " --steps " << maxSteps
        << " --iters " << iters;

    std::string rawOutput;
    runSubprocess(cmd.str(), rawOutput, outCrashStatus, timeoutSec);
    return parseWorkerOutput(rawOutput);
}

struct WinnerInfo {
    std::string category;
    std::string winnerBackend;
    double metricValue = 0.0;
    std::string unit;
};

void writeFullJsonReport(
    const std::string& path,
    const std::vector<BenchRunResult>& allResults,
    const std::vector<WinnerInfo>& categoryWinners,
    const std::vector<WinnerInfo>& regimeWinners,
    const std::map<std::string, double>& overallScores,
    const std::string& overallWinner) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        std::cerr << "Failed to write JSON report to " << path << "\n";
        return;
    }

    ofs << "{\n"
        << "  \"version\": 2,\n"
        << "  \"overall_winner\": \"" << jsonEscape(overallWinner) << "\",\n"
        << "  \"overall_geomean_ms\": {\n";
    {
        size_t idx = 0;
        for (const auto& [b, ms] : overallScores) {
            ofs << "    \"" << jsonEscape(b) << "\": " << std::fixed << std::setprecision(4) << ms;
            if (++idx < overallScores.size()) ofs << ",";
            ofs << "\n";
        }
    }
    ofs << "  },\n"
        << "  \"category_winners\": [\n";
    for (size_t i = 0; i < categoryWinners.size(); ++i) {
        const auto& w = categoryWinners[i];
        ofs << "    {\"category\":\"" << jsonEscape(w.category) << "\","
            << "\"winner\":\"" << jsonEscape(w.winnerBackend) << "\","
            << "\"geomean_ms\":" << std::fixed << std::setprecision(4) << w.metricValue << "}";
        if (i + 1 < categoryWinners.size()) ofs << ",";
        ofs << "\n";
    }
    ofs << "  ],\n"
        << "  \"regime_winners\": [\n";
    for (size_t i = 0; i < regimeWinners.size(); ++i) {
        const auto& w = regimeWinners[i];
        ofs << "    {\"regime\":\"" << jsonEscape(w.category) << "\","
            << "\"winner\":\"" << jsonEscape(w.winnerBackend) << "\","
            << "\"value\":" << std::fixed << std::setprecision(4) << w.metricValue << ","
            << "\"unit\":\"" << jsonEscape(w.unit) << "\"}";
        if (i + 1 < regimeWinners.size()) ofs << ",";
        ofs << "\n";
    }
    ofs << "  ],\n"
        << "  \"suites\": [\n";
    for (size_t i = 0; i < allResults.size(); ++i) {
        ofs << "    " << serializeResultJson(allResults[i]);
        if (i + 1 < allResults.size()) ofs << ",";
        ofs << "\n";
    }
    ofs << "  ],\n"
        << "  \"benchmarks\": [\n";
    std::vector<const BenchRunResult*> compat;
    for (const auto& r : allResults) {
        if (r.benchId == "bench_raymarch") compat.push_back(&r);
    }
    for (size_t i = 0; i < compat.size(); ++i) {
        ofs << "    " << serializeResultJson(*compat[i]);
        if (i + 1 < compat.size()) ofs << ",";
        ofs << "\n";
    }
    ofs << "  ]\n}\n";
}

} // namespace

int main(int argc, char** argv) {
    std::vector<uint32_t> widths = {4, 8, 16};
    uint32_t width = 256;
    uint32_t height = 256;
    uint32_t maxSteps = 48;
    uint32_t iters = 5;
    std::string jsonOut;
    std::string shaderDir = NACELLE_SHADER_BIN_DIR;
    std::string suiteFilter = "all";
    std::string nameFilter;
    uint32_t timeoutSec = 60;
    std::string checkpointPath;
    bool resumeFromCheckpoint = false;
    std::vector<std::string> skipPatterns;
    bool allowPartial = false;
    bool forceWarp = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--quick") {
            width = 128;
            height = 128;
            maxSteps = 24;
            iters = 2;
        } else if (arg.rfind("--simd-width=", 0) == 0) {
            widths.clear();
            std::stringstream ss(arg.substr(13));
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                if (!tok.empty()) widths.push_back(static_cast<uint32_t>(std::stoul(tok)));
            }
        } else if (arg == "--iters" && i + 1 < argc) {
            iters = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--width" && i + 1 < argc) {
            width = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--height" && i + 1 < argc) {
            height = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--steps" && i + 1 < argc) {
            maxSteps = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--json" && i + 1 < argc) {
            jsonOut = argv[++i];
        } else if (arg == "--shader-dir" && i + 1 < argc) {
            shaderDir = argv[++i];
        } else if (arg == "--suite" && i + 1 < argc) {
            suiteFilter = argv[++i];
        } else if (arg == "--filter" && i + 1 < argc) {
            nameFilter = argv[++i];
        } else if (arg == "--timeout" && i + 1 < argc) {
            timeoutSec = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--checkpoint" && i + 1 < argc) {
            checkpointPath = argv[++i];
        } else if (arg == "--resume") {
            resumeFromCheckpoint = true;
        } else if (arg == "--skip" && i + 1 < argc) {
            std::stringstream ss(argv[++i]);
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                if (!tok.empty()) skipPatterns.push_back(tok);
            }
        } else if (arg == "--allow-partial") {
            allowPartial = true;
        } else if (arg == "--warp") {
            forceWarp = true;
        }
    }

    const std::filesystem::path binDir = getExecutableDir(argv[0]);
    auto catalog = buildBenchmarkCatalog();

    std::vector<std::string> backendTokens;
    for (uint32_t w : widths) {
        backendTokens.push_back("nacelle-w" + std::to_string(w));
    }

    std::vector<VkDeviceInfo> d3d12Devs;
#if defined(_WIN32)
    if (forceWarp) {
        backendTokens.push_back("d3d12:warp");
    } else {
        d3d12Devs = listDevicesViaWorker(binDir, "bench-worker-d3d12");
        if (d3d12Devs.empty()) {
            backendTokens.push_back("d3d12:warp");
        } else {
            for (const auto& d : d3d12Devs) {
                backendTokens.push_back("d3d12:" + std::to_string(d.index));
            }
        }
    }
#else
    (void)forceWarp;
    backendTokens.push_back("warp");
#endif

    auto vkDevs = listDevicesViaWorker(binDir, "bench-worker-vulkan");
    if (vkDevs.empty()) {
        backendTokens.push_back("vulkan");
    } else {
        for (const auto& d : vkDevs) {
            backendTokens.push_back("vulkan:" + std::to_string(d.index));
        }
    }
#if defined(__APPLE__)
    backendTokens.push_back("metal");
#endif

    std::vector<BenchRunResult> allResults;
    bool anyNacelleFailed = false;
    bool anyNacellePassed = false;

    auto makeCrashResult = [&](const BenchmarkSpec& spec, const std::string& bt, const std::string& status) {
        BenchRunResult r;
        r.benchId = spec.id;
        r.benchName = spec.name;
        r.category = categoryName(spec.category);
        r.ran = false;
        r.status = status;
        if (bt.rfind("nacelle-w", 0) == 0) {
            uint32_t w = static_cast<uint32_t>(std::stoul(bt.substr(9)));
            r.backend = "Nacelle (W=" + std::to_string(w) + ")";
            r.shaderFormat = "DXIL";
            r.simdWidth = w;
        } else if (bt.rfind("vulkan:", 0) == 0) {
            size_t idx = static_cast<size_t>(std::stoul(bt.substr(7)));
            r.backend = (idx < vkDevs.size()) ? vkDevs[idx].backendName : bt;
            r.shaderFormat = "SPIR-V";
        } else if (bt == "vulkan") {
            r.backend = "Vulkan";
            r.shaderFormat = "SPIR-V";
        } else if (bt == "metal") {
            r.backend = "Metal (IRConverter)";
            r.shaderFormat = "DXIL";
        } else if (bt == "warp" || bt == "d3d12:warp") {
            r.backend = "Microsoft WARP";
            r.shaderFormat = "DXIL";
        } else if (bt.rfind("d3d12:", 0) == 0) {
            size_t idx = static_cast<size_t>(std::stoul(bt.substr(6)));
            r.backend = (idx < d3d12Devs.size()) ? d3d12Devs[idx].backendName : bt;
            r.shaderFormat = "DXIL";
        } else {
            r.backend = bt;
        }
        return r;
    };

    struct BenchTask {
        const BenchmarkSpec* spec;
        std::string backendToken;
    };
    std::vector<BenchTask> tasks;
    for (const auto& spec : catalog) {
        if (suiteFilter != "all" && suiteFilter != categoryKey(spec.category)) continue;
        if (!nameFilter.empty() && spec.id.find(nameFilter) == std::string::npos) continue;

        for (const auto& bt : backendTokens) {
#if !defined(_WIN32)
            if (bt == "warp") {
                if (spec.id == "bench_raymarch") {
                    BenchRunResult r;
                    r.benchId = spec.id;
                    r.benchName = spec.name;
                    r.category = categoryName(spec.category);
                    r.backend = "Microsoft WARP";
                    r.shaderFormat = "DXIL";
                    r.status = "SKIPPED (Windows-only)";
                    allResults.push_back(std::move(r));
                }
                continue;
            }
#endif
            tasks.push_back({&spec, bt});
        }
    }

    auto formatProgress = [](size_t done, size_t total) -> std::string {
        constexpr size_t barWidth = 20;
        size_t pct = (total > 0) ? (done * 100) / total : 100;
        size_t filled = (total > 0) ? (done * barWidth) / total : barWidth;
        std::string bar(barWidth, '-');
        for (size_t i = 0; i < filled && i < barWidth; ++i) {
            bar[i] = '=';
        }
        if (filled < barWidth && done > 0 && done < total) {
            bar[filled] = '>';
        }
        std::ostringstream ss;
        ss << "[" << std::setw(3) << done << "/" << std::setw(3) << total << "] "
           << "[" << bar << "] " << std::setw(3) << pct << "%";
        return ss.str();
    };

    auto summarizeStatus = [](const std::vector<BenchRunResult>& res, const std::string& crashStatus) -> std::string {
        if (res.empty()) {
            return crashStatus.empty() ? "NO OUTPUT" : crashStatus;
        }
        size_t ranCount = 0;
        for (const auto& r : res) {
            if (r.ran) ++ranCount;
        }
        if (ranCount == res.size()) return "OK";
        if (ranCount > 0) return "PARTIAL";
        return res[0].status.empty() ? "FAILED" : res[0].status;
    };

    if (resumeFromCheckpoint && checkpointPath.empty()) {
        checkpointPath = jsonOut.empty() ? "nacelle-bench.checkpoint" : (jsonOut + ".checkpoint");
    }

    auto taskKey = [](const std::string& benchId, const std::string& bt) {
        return benchId + "\t" + bt;
    };

    auto shouldSkipTask = [&](const std::string& benchId, const std::string& bt) {
        std::string combined = benchId + "@" + bt;
        std::string backendOnly = "@" + bt;
        for (const auto& pat : skipPatterns) {
            if (pat == combined || pat == benchId || pat == backendOnly) return true;
        }
        return false;
    };

    std::set<std::string> ckptStarted;
    std::set<std::string> ckptDone;
    if (resumeFromCheckpoint && !checkpointPath.empty() && std::filesystem::exists(checkpointPath)) {
        std::ifstream ifs(checkpointPath);
        std::string line;
        while (std::getline(ifs, line)) {
            while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
            if (line.rfind("START\t", 0) == 0) {
                ckptStarted.insert(line.substr(6));
            } else if (line.rfind("DONE\t", 0) == 0) {
                ckptDone.insert(line.substr(5));
            } else if (line.rfind("RESULT\t", 0) == 0) {
                size_t p1 = line.find('\t', 7);
                if (p1 == std::string::npos) continue;
                size_t p2 = line.find('\t', p1 + 1);
                if (p2 == std::string::npos) continue;
                std::string bt = line.substr(p1 + 1, p2 - (p1 + 1));
                std::string json = line.substr(p2 + 1);
                BenchRunResult r;
                if (parseWorkerJsonLine(json, r)) {
                    if (bt.rfind("nacelle-", 0) == 0) {
                        if (r.ran) anyNacellePassed = true;
                        else anyNacelleFailed = true;
                    }
                    allResults.push_back(std::move(r));
                }
            }
        }
        std::cerr << "Resuming from checkpoint " << checkpointPath
                  << " (" << ckptDone.size() << " completed, "
                  << (ckptStarted.size() > ckptDone.size() ? ckptStarted.size() - ckptDone.size() : 0)
                  << " interrupted)\n" << std::flush;
    }

    std::ofstream ckptOut;
    if (!checkpointPath.empty()) {
        std::ios::openmode mode = resumeFromCheckpoint ? (std::ios::app | std::ios::out) : (std::ios::trunc | std::ios::out);
        ckptOut.open(checkpointPath, mode);
    }

    auto recordResults = [&](const std::string& benchId, const std::string& bt, std::vector<BenchRunResult>& res) {
        if (bt.rfind("nacelle-", 0) == 0 && res.empty()) {
            anyNacelleFailed = true;
        }
        for (auto& r : res) {
            if (bt.rfind("nacelle-", 0) == 0) {
                if (r.ran) anyNacellePassed = true;
                else anyNacelleFailed = true;
            }
            if (ckptOut.is_open()) {
                ckptOut << "RESULT\t" << benchId << "\t" << bt << "\t" << serializeResultJson(r) << "\n";
            }
            allResults.push_back(std::move(r));
        }
        if (ckptOut.is_open()) {
            ckptOut << "DONE\t" << benchId << "\t" << bt << "\n" << std::flush;
        }
    };

    const size_t totalTasks = tasks.size();
    for (size_t ti = 0; ti < totalTasks; ++ti) {
        const auto& spec = *tasks[ti].spec;
        const auto& bt = tasks[ti].backendToken;
        const std::string key = taskKey(spec.id, bt);

        if (ckptDone.count(key)) {
            std::cerr << formatProgress(ti + 1, totalTasks)
                      << " SKIP  " << spec.id << " @ " << bt << " (already in checkpoint)\n" << std::flush;
            continue;
        }
        if (ckptStarted.count(key)) {
            std::cerr << formatProgress(ti + 1, totalTasks)
                      << " SKIP  " << spec.id << " @ " << bt << " (hung/interrupted in prior run)\n" << std::flush;
            std::vector<BenchRunResult> skipped{makeCrashResult(spec, bt, "SKIPPED (hung in prior run)")};
            recordResults(spec.id, bt, skipped);
            continue;
        }
        if (shouldSkipTask(spec.id, bt)) {
            std::cerr << formatProgress(ti + 1, totalTasks)
                      << " SKIP  " << spec.id << " @ " << bt << " (--skip)\n" << std::flush;
            std::vector<BenchRunResult> skipped{makeCrashResult(spec, bt, "SKIPPED (user)")};
            recordResults(spec.id, bt, skipped);
            continue;
        }

        if (ckptOut.is_open()) {
            ckptOut << "START\t" << spec.id << "\t" << bt << "\n" << std::flush;
        }

        std::cerr << formatProgress(ti, totalTasks)
                  << " START " << spec.id << " @ " << bt << "\n" << std::flush;

        const auto tStart = std::chrono::steady_clock::now();
        std::string crashStatus;
        auto res = runIsolatedWorker(binDir, spec.id, bt, shaderDir, width, height, maxSteps, iters, timeoutSec, crashStatus);
        const auto tEnd = std::chrono::steady_clock::now();
        const double elapsedSec = std::chrono::duration<double>(tEnd - tStart).count();

        std::ostringstream elapsedStr;
        elapsedStr << std::fixed << std::setprecision(2) << elapsedSec;
        std::cerr << formatProgress(ti + 1, totalTasks)
                  << " STOP  " << spec.id << " @ " << bt
                  << " (" << elapsedStr.str() << "s) -> "
                  << summarizeStatus(res, crashStatus) << "\n" << std::flush;

        if (res.empty() && !crashStatus.empty()) {
            res.push_back(makeCrashResult(spec, bt, crashStatus));
        }
        recordResults(spec.id, bt, res);
    }

    // Compute Category Winners (lowest geomean dispatch time in ms)
    std::vector<BenchCategory> cats = {
        BenchCategory::Compute, BenchCategory::Vertex, BenchCategory::Pixel,
        BenchCategory::Geometry, BenchCategory::Tessellation, BenchCategory::RayTracing,
        BenchCategory::Mesh
    };

    std::map<std::string, std::vector<double>> backendCategoryMs;
    std::vector<WinnerInfo> categoryWinners;

    for (auto c : cats) {
        std::string cname = categoryName(c);
        std::map<std::string, std::vector<double>> msByBackend;
        for (const auto& r : allResults) {
            if (r.ran && r.category == cname && r.medianMs > 0.0) {
                msByBackend[r.backend].push_back(r.medianMs);
            }
        }
        std::string bestBackend = "N/A";
        double bestMs = 1e18;
        for (const auto& [b, vals] : msByBackend) {
            double gm = computeGeoMean(vals);
            if (gm > 0.0) {
                backendCategoryMs[b].push_back(gm);
                if (gm < bestMs) {
                    bestMs = gm;
                    bestBackend = b;
                }
            }
        }
        categoryWinners.push_back({cname, bestBackend, (bestBackend == "N/A" ? 0.0 : bestMs), "ms"});
    }

    // Overall Geomean Time (geomean of category geomean ms, lower is faster)
    std::map<std::string, double> overallMs;
    std::string overallWinner = "N/A";
    double bestOverallMs = 1e18;
    for (const auto& [b, catMs] : backendCategoryMs) {
        double ov = computeGeoMean(catMs);
        overallMs[b] = ov;
        if (ov > 0.0 && ov < bestOverallMs) {
            bestOverallMs = ov;
            overallWinner = b;
        }
    }

    // Regime Winners: Cold JIT, Hot JIT, Short Job, Long Job, Lowest Peak Memory
    std::map<std::string, std::vector<double>> coldByB, hotByB, shortByB, longByB, memByB;
    for (const auto& r : allResults) {
        if (!r.ran) continue;
        if (r.coldMs > 0.0) coldByB[r.backend].push_back(r.coldMs);
        if (r.medianMs > 0.0) hotByB[r.backend].push_back(r.medianMs);
        if (r.shortJobMs > 0.0) shortByB[r.backend].push_back(r.shortJobMs);
        if (r.longJobMs > 0.0) longByB[r.backend].push_back(r.longJobMs);
        if (r.peakRssMb > 0.0) memByB[r.backend].push_back(r.peakRssMb);
    }

    auto pickLowestGeomean = [](const std::map<std::string, std::vector<double>>& m, const std::string& label, const std::string& unit) -> WinnerInfo {
        std::string best = "N/A";
        double bestVal = 1e18;
        for (const auto& [b, vals] : m) {
            double gm = computeGeoMean(vals);
            if (gm > 0.0 && gm < bestVal) {
                bestVal = gm;
                best = b;
            }
        }
        return {label, best, (best == "N/A" ? 0.0 : bestVal), unit};
    };

    std::vector<WinnerInfo> regimeWinners = {
        pickLowestGeomean(coldByB, "Cold JIT", "ms"),
        pickLowestGeomean(hotByB, "Hot JIT", "ms"),
        pickLowestGeomean(shortByB, "Short-Running Job", "ms"),
        pickLowestGeomean(longByB, "Long-Running Job", "ms"),
        pickLowestGeomean(memByB, "Lowest Peak Memory", "MB"),
    };

    // Print Console Report
    std::cout << "\n========================================================================================================================\n";
    std::cout << "  Nacelle Cross-Backend Shader & Rasterizer Benchmark Suite (" << width << "x" << height << ", steps=" << maxSteps << ", iters=" << iters << ")\n";
    std::cout << "========================================================================================================================\n\n";

    size_t benchColW = 20;
    size_t backendColW = 22;
    for (const auto& r : allResults) {
        benchColW = std::max(benchColW, r.benchName.size() + 2);
        backendColW = std::max(backendColW, r.backend.size() + 2);
    }
    const size_t totalLineW = benchColW + 13 + backendColW + 10 * 5 + 10;

    std::cout << std::left
              << std::setw(static_cast<int>(benchColW)) << "Benchmark"
              << std::setw(13) << "Category"
              << std::setw(static_cast<int>(backendColW)) << "Backend"
              << std::setw(10) << "Cold(ms)"
              << std::setw(10) << "Hot(ms)"
              << std::setw(10) << "Short(ms)"
              << std::setw(10) << "Long(ms)"
              << std::setw(10) << "Peak(MB)"
              << "Status\n";
    std::cout << std::string(totalLineW, '-') << "\n";

    for (const auto& r : allResults) {
        std::cout << std::left
                  << std::setw(static_cast<int>(benchColW)) << r.benchName
                  << std::setw(13) << r.category
                  << std::setw(static_cast<int>(backendColW)) << r.backend;
        if (r.ran) {
            std::cout << std::fixed << std::setprecision(2)
                      << std::setw(10) << r.coldMs
                      << std::setw(10) << r.medianMs
                      << std::setw(10) << r.shortJobMs
                      << std::setw(10) << r.longJobMs
                      << std::setprecision(1) << std::setw(10) << r.peakRssMb
                      << r.status << "\n";
        } else {
            std::cout << std::setw(10) << "-"
                      << std::setw(10) << "-"
                      << std::setw(10) << "-"
                      << std::setw(10) << "-"
                      << std::setw(10) << "-"
                      << r.status << "\n";
        }
    }

    std::cout << "\n--- Category Winners (Geomean Dispatch Time, lower is faster) ---\n";
    for (const auto& w : categoryWinners) {
        std::cout << "  " << std::left << std::setw(16) << w.category
                  << "-> " << std::setw(36) << w.winnerBackend;
        if (w.metricValue > 0.0) {
            std::cout << std::fixed << std::setprecision(2) << w.metricValue << " ms";
        }
        std::cout << "\n";
    }

    std::cout << "\n--- Regime Winners (Geomean, lower is better) ---\n";
    for (const auto& w : regimeWinners) {
        std::cout << "  " << std::left << std::setw(22) << w.category
                  << "-> " << std::setw(30) << w.winnerBackend
                  << std::fixed << std::setprecision(2) << w.metricValue << " " << w.unit << "\n";
    }

    std::cout << "\n--- Overall Standings (Geomean Dispatch Time, lower is faster) ---\n";
    std::vector<std::pair<std::string, double>> sortedOverall(overallMs.begin(), overallMs.end());
    std::sort(sortedOverall.begin(), sortedOverall.end(), [](const auto& a, const auto& b) { return a.second < b.second; });
    for (size_t i = 0; i < sortedOverall.size(); ++i) {
        std::cout << "  #" << (i + 1) << " " << std::left << std::setw(32) << sortedOverall[i].first
                  << std::fixed << std::setprecision(2) << sortedOverall[i].second << " ms\n";
    }
    std::cout << "\nOVERALL WINNER: " << overallWinner << "\n";

    if (!jsonOut.empty()) {
        writeFullJsonReport(jsonOut, allResults, categoryWinners, regimeWinners, overallMs, overallWinner);
        std::cout << "JSON report written to: " << jsonOut << "\n";
    }

    return (anyNacelleFailed && !(allowPartial && anyNacellePassed)) ? 1 : 0;
}
