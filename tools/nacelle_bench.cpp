#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
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
#include <sys/wait.h>
#include <unistd.h>
#define NACELLE_POPEN popen
#define NACELLE_PCLOSE pclose
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

bool runSubprocess(const std::string& cmdLine, std::string& outStdout, std::string& outStatus) {
    outStdout.clear();
    outStatus.clear();
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
    DWORD bytesRead = 0;
    while (ReadFile(hRead, buf, sizeof(buf), &bytesRead, nullptr) && bytesRead > 0) {
        outStdout.append(buf, bytesRead);
    }
    CloseHandle(hRead);
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD exitCode = 0;
    GetExitCodeProcess(pi.hProcess, &exitCode);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    if (exitCode != 0) {
        std::ostringstream ss;
        ss << "CRASHED (0x" << std::hex << exitCode << ")";
        outStatus = ss.str();
        return false;
    }
    return true;
#else
    std::string fullCmd = cmdLine + " 2>&1";
    FILE* pipe = NACELLE_POPEN(fullCmd.c_str(), "r");
    if (!pipe) {
        outStatus = "FAILED (popen)";
        return false;
    }
    char buf[4096];
    while (fgets(buf, sizeof(buf), pipe) != nullptr) {
        outStdout.append(buf);
    }
    int rc = NACELLE_PCLOSE(pipe);
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

std::vector<VkDeviceInfo> listVulkanDevicesViaWorker(const std::filesystem::path& binDir) {
    std::vector<VkDeviceInfo> out;
    std::string vkWorker = workerBinaryPath(binDir, "bench-worker-vulkan");
    if (!std::filesystem::exists(vkWorker)) return out;

    std::string rawOut, errStatus;
    if (!runSubprocess("\"" + vkWorker + "\" --list-devices", rawOut, errStatus)) {
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
    if (backendToken == "warp") {
        return workerBinaryPath(binDir, "bench-worker-warp");
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
    runSubprocess(cmd.str(), rawOutput, outCrashStatus);
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
    bool allowPartial = false;

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
        } else if (arg == "--allow-partial") {
            allowPartial = true;
        }
    }

    const std::filesystem::path binDir = getExecutableDir(argv[0]);
    auto catalog = buildBenchmarkCatalog();

    std::vector<std::string> backendTokens;
    for (uint32_t w : widths) {
        backendTokens.push_back("nacelle-w" + std::to_string(w));
    }
    backendTokens.push_back("warp");

    auto vkDevs = listVulkanDevicesViaWorker(binDir);
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
        } else if (bt == "warp") {
            r.backend = "Microsoft WARP";
            r.shaderFormat = "DXIL";
        } else {
            r.backend = bt;
        }
        return r;
    };

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
            std::string crashStatus;
            auto res = runIsolatedWorker(binDir, spec.id, bt, shaderDir, width, height, maxSteps, iters, crashStatus);
            if (res.empty() && !crashStatus.empty()) {
                res.push_back(makeCrashResult(spec, bt, crashStatus));
            }
            if (bt.rfind("nacelle-", 0) == 0 && res.empty()) {
                anyNacelleFailed = true;
            }
            for (auto& r : res) {
                if (bt.rfind("nacelle-", 0) == 0) {
                    if (r.ran) anyNacellePassed = true;
                    else anyNacelleFailed = true;
                }
                allResults.push_back(std::move(r));
            }
        }
    }

    // Compute Category Winners (lowest geomean dispatch time in ms)
    std::vector<BenchCategory> cats = {
        BenchCategory::Compute, BenchCategory::Vertex, BenchCategory::Pixel,
        BenchCategory::Geometry, BenchCategory::Tessellation, BenchCategory::RayTracing
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
    categoryWinners.push_back({"Mesh", "NOT IMPLEMENTED (DispatchMesh stub)", 0.0, "ms"});

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
