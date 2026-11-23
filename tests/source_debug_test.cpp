#include <nacelle/source_debug.h>
#include "compiler/source_debug.h"
#include "compiler/native_debug.h"
#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <future>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
const std::string shader(64, 'a'), source(64, 'b'), other(64, 'c');
nacelle::source_debug::Site makeSite() {
    nacelle::source_debug::Site site;
    std::strcpy(site.location.shader_id, shader.c_str());
    std::strcpy(site.location.source_id, source.c_str());
    std::strcpy(site.location.label, "compute");
    site.location.stage = NACELLE_SOURCE_COMPUTE;
    site.location.line = 2;
    site.location.frames[0].row = 2;
    std::strcpy(site.location.frames[0].source_id, source.c_str());
    std::strcpy(site.location.frames[0].function, "main");
    site.location.local_count = 1;
    NacelleSourceDebugLocal local;
    local.kind = NACELLE_SOURCE_UNSIGNED;
    local.bits = 32;
    std::strcpy(local.name, "value");
    site.locals.push_back(local);
    return site;
}
NacelleSourceDebugSelection selection() {
    NacelleSourceDebugSelection value;
    std::strcpy(value.shader_id, shader.c_str());
    value.xyz[0] = 7;
    return value;
}
uint64_t begin() {
    auto target = selection();
    uint64_t id = 0;
    require(NacelleBeginSourceDebug(&target, &id) == S_OK && id, "Begin failed");
    return id;
}
void programValidation() {
    llvm::LLVMContext executableContext, debugContext;
    llvm::Module executable("executable", executableContext), debug("embedded-debug", debugContext);
    const auto populate = [](llvm::Module& module, unsigned constant) {
        auto& context = module.getContext();
        llvm::IRBuilder<> builder(context);
        auto* function = llvm::Function::Create(
            llvm::FunctionType::get(builder.getInt32Ty(), false),
            llvm::GlobalValue::ExternalLinkage, "main", module);
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", function));
        builder.CreateRet(builder.getInt32(constant));
        return function;
    };
    populate(executable, 7);
    auto* debugFunction = populate(debug, 7);
    debug.getOrInsertNamedMetadata("dx.source.contents")->addOperand(
        llvm::MDNode::get(debugContext, llvm::MDString::get(debugContext, "debug-only")));
    std::string error;
    require(nacelle::source_debug::sameProgram(executable, debug, error),
            "Debug-only container metadata changed executable identity");
    debugFunction->back().getTerminator()->setOperand(0, llvm::ConstantInt::get(
        llvm::Type::getInt32Ty(debugContext), 8));
    require(!nacelle::source_debug::sameProgram(executable, debug, error) && !error.empty(),
            "Different embedded executable code was admitted");
}
void metadataValidation() {
    using namespace nacelle::source_debug;
    std::string error;
    require(!addSource(std::string(64, 'A'), "text", error) && !error.empty(), "Uppercase identity accepted");
    require(addSource(source, "first\nsecond\n", error), "Source insertion failed");
    require(addSource(source, "first\nsecond\n", error), "Idempotent source insertion failed");
    require(!addSource(source, "replacement", error), "Immutable source replaced");
    auto invalid = makeSite();
    invalid.location.line = 99;
    require(!addSite(invalid, error), "Out of range source line accepted");
    invalid = makeSite();
    std::memset(invalid.location.label, 'x', sizeof(invalid.location.label));
    require(!addSite(invalid, error), "Unterminated label accepted");
    invalid = makeSite();
    invalid.locals[0].name[0] = '\n';
    require(!addSite(invalid, error), "Control character in local name accepted");
    invalid = makeSite();
    invalid.locals.resize(65);
    invalid.location.local_count = 65;
    require(!addSite(invalid, error), "Unbounded locals accepted");
    require(addSite(makeSite(), error) == 1, "Site insertion failed");
    auto target = selection();
    uint64_t unpublished = 0;
    require(FAILED(NacelleBeginSourceDebug(&target, &unpublished)) && !unpublished,
            "Metadata from an unsuccessful/unpublished compilation was admitted");
    publishShader(shader);
    NacelleSourceDebugSite site;
    require(NacelleGetSourceDebugSite(1, &site) == S_OK && site.line == 2, "Site lookup failed");
    require(NacelleGetSourceDebugSite(2, &site) == S_FALSE && !site.id, "Unavailable site not cleared");
    auto otherLine = makeSite();
    otherLine.location.line = 1;
    require(addSite(std::move(otherLine), error) == 2, "Second executable line insertion failed");
    NacelleSourceDebugLocal local;
    require(NacelleGetSourceDebugLocal(1, 0, &local) == S_OK && local.bits == 32, "Local lookup failed");
    require(NacelleGetSourceDebugLocal(1, 1, &local) == S_FALSE && !local.bits, "Unavailable local not cleared");
    uint64_t bytes = 0;
    require(NacelleGetSourceDebugSource(source.c_str(), nullptr, &bytes) == S_OK && bytes == 13,
            "Source size query failed");
    char buffer[16];
    std::memset(buffer, '!', sizeof(buffer));
    bytes = 1;
    require(FAILED(NacelleGetSourceDebugSource(source.c_str(), buffer, &bytes)) &&
            bytes == 13 && buffer[0] == '!', "Short buffer overwritten");
    require(NacelleGetSourceDebugSource(source.c_str(), buffer, &bytes) == S_OK &&
            std::string_view(buffer, bytes) == "first\nsecond\n" && buffer[13] == '!',
            "Source bytes or buffer bound changed");
    require(NacelleGetSourceDebugSource(other.c_str(), nullptr, &bytes) == S_FALSE && !bytes,
            "Missing source not unavailable");
}
}

int main(int argc, char** argv) {
    try {
        using namespace nacelle::source_debug;
        // A disabled submission must not become tracked when configuration is enabled later.
        Work disabledWork;
        const std::string_view mode = argc > 1 ? argv[1] : "";
        NacelleSourceDebugOptions options;
        require(FAILED(NacelleConfigureSourceDebug(nullptr)), "Null configuration accepted");
        options.version = 2;
        require(FAILED(NacelleConfigureSourceDebug(&options)), "Unknown configuration version accepted");
        options.version = 1;
        options.enabled = 2;
        require(FAILED(NacelleConfigureSourceDebug(&options)), "Invalid enabled flag accepted");
        options.enabled = 1;
        for (uint32_t consent = 0; consent < 3; ++consent) {
            options.consent = consent;
            require(FAILED(NacelleConfigureSourceDebug(&options)), "Missing privacy consent accepted");
        }
        options.consent = 3;
        options.source_bytes = 16ull * 1024 * 1024 + 1;
        require(FAILED(NacelleConfigureSourceDebug(&options)), "Excess source budget accepted");
        options.source_bytes = 8ull * 1024 * 1024;
        options.metadata_bytes = 32ull * 1024 * 1024 + 1;
        require(FAILED(NacelleConfigureSourceDebug(&options)), "Excess metadata budget accepted");
        options.metadata_bytes = mode == "metadata-budget" ? 1 : 16ull * 1024 * 1024;
        if (mode == "source-budget") options.source_bytes = 1;
        require(NacelleConfigureSourceDebug(&options) == S_OK && enabled() &&
                nacelle::native_debug::enabled(), "Configuration did not enable native debug");
        require(freeze() && freeze(), "Freeze is not idempotent");
        require(FAILED(NacelleConfigureSourceDebug(&options)), "Late configuration accepted");
        std::string error;
        if (mode == "source-budget") {
            require(!addSource(source, "x", error) && error.find("budget") != std::string::npos,
                    "Source budget not enforced");
            std::puts("SOURCE DEBUG SOURCE BUDGET PASS");
            return 0;
        }
        if (mode == "metadata-budget") {
            require(addSource(source, "first\nsecond\n", error), "Budget source insertion failed");
            require(!addSite(makeSite(), error) && error.find("budget") != std::string::npos,
                    "Metadata record budget not enforced");
            std::puts("SOURCE DEBUG METADATA BUDGET PASS");
            return 0;
        }
        auto target = selection();
        uint64_t id = 0;
        require(FAILED(NacelleBeginSourceDebug(&target, &id)), "Missing shader metadata accepted");
        metadataValidation();
        programValidation();
        {
            Work uncaptured;
            require(FAILED(NacelleBeginSourceDebug(&target, &id)), "Begin accepted non-quiescent work");
        }
        id = begin();
        uint64_t overlap = 0;
        require(FAILED(NacelleBeginSourceDebug(&target, &overlap)) && !overlap, "Overlapping session accepted");
        require(FAILED(NacelleSourceDebugBreakpoint(id, source.c_str(), 3)), "Non-executable breakpoint accepted");
        require(FAILED(NacelleSourceDebugBreakpoint(id, other.c_str(), 2)), "Other source breakpoint accepted");
        require(NacelleSourceDebugBreakpoint(id, source.c_str(), 2) == S_OK, "Executable breakpoint rejected");
        NacelleSourceDebugSnapshot snapshot;
        require(NacelleGetSourceDebugSnapshot(id, &snapshot) == S_FALSE && !snapshot.stop,
                "Unobserved snapshot fabricated");
        require(FAILED(NacelleControlSourceDebug(id, 1, NACELLE_SOURCE_STEP)), "Stale stop accepted");
        {
            Work work;
            Operation operation(shader);
            uint32_t xyz[6]{7, 0, 0, 8, 0, 0};
            uint64_t values[2]{42, 43};
            nacelle_source_debug_observe(1, 2, 2, 3, xyz, values);
            require(NacelleGetSourceDebugSnapshot(id, &snapshot) == S_FALSE, "Inactive target stopped");
            // This host function is deliberately NOT registered JIT code. It must not publish
            // a native identity belonging to an unrelated shader or fabricate a successful stop.
            nacelle_source_debug_observe(1, 2, 3, 3, xyz, values);
        }
        NacelleSourceDebugSummary summary;
        require(NacelleEndSourceDebug(id, &summary) == S_FALSE && summary.cancelled &&
                summary.target_seen && !summary.stops, "Missing native identity looked successful");
        require(FAILED(NacelleEndSourceDebug(id, &summary)), "Stale session ended twice");
        require(FAILED(NacelleGetSourceDebugSnapshot(id, &snapshot)), "Stale snapshot session accepted");
        const auto previous = id;
        id = begin();
        require(id > previous, "Session identity reused");
        require(FAILED(NacelleControlSourceDebug(previous, 0, NACELLE_SOURCE_CANCEL)), "Stale cancellation accepted");
        require(NacelleControlSourceDebug(id, 0, NACELLE_SOURCE_CANCEL) == S_OK, "Cancellation failed");
        require(NacelleEndSourceDebug(id, &summary) == S_FALSE && summary.cancelled,
                "Intentional cancellation looked successful");

        id = begin();
        require(NacelleSourceDebugBreakpoint(id, source.c_str(), 1) == S_OK, "Second executable breakpoint rejected");
        {
            Work work;
            Operation operation(shader);
            uint32_t xyz[3]{7, 0, 0};
            uint64_t value = 42;
            nacelle_source_debug_observe(1, 1, 1, 1, xyz, &value);
        }
        require(SUCCEEDED(NacelleEndSourceDebug(id, &summary)) && summary.target_seen &&
                !summary.cancelled && !summary.stops,
                "Initial stepping ignored the configured breakpoint");

        id = begin();
        require(NacelleSourceDebugBreakpoint(id, source.c_str(), 1) == S_OK, "Operation breakpoint rejected");
        {
            Work work;
            uint32_t xyz[3]{7, 0, 0};
            uint64_t value = 42;
            { Operation first(shader); nacelle_source_debug_observe(1, 1, 1, 1, xyz, &value); }
            { Operation later(shader); nacelle_source_debug_observe(2, 1, 1, 1, xyz, &value); }
        }
        require(NacelleEndSourceDebug(id, &summary) == S_OK && !summary.cancelled && !summary.stops,
                "Observation migrated to a subsequent shader operation");

        auto pixelSite = makeSite();
        std::strcpy(pixelSite.location.shader_id, other.c_str());
        pixelSite.location.stage = NACELLE_SOURCE_PIXEL;
        require(addSite(pixelSite, error) == 3, "Pixel site insertion failed");
        pixelSite.location.line = 1;
        require(addSite(pixelSite, error) == 4, "Pixel continuation site insertion failed");
        publishShader(other);
        auto pixelTarget = selection();
        std::strcpy(pixelTarget.shader_id, other.c_str());
        require(NacelleBeginSourceDebug(&pixelTarget, &id) == S_OK, "Pixel session failed");
        require(NacelleSourceDebugBreakpoint(id, source.c_str(), 1) == S_OK, "Pixel breakpoint failed");
        {
            Work work;
            Operation operation(other);
            uint32_t xyz[3]{7, 0, 0};
            uint64_t value = 42, primitive = 11;
            nacelle_source_debug_observe(3, 1, 1, 1, xyz, &value, &primitive);
            primitive = 22;
            nacelle_source_debug_observe(4, 1, 1, 1, xyz, &value, &primitive);
        }
        require(NacelleEndSourceDebug(id, &summary) == S_OK && !summary.cancelled && !summary.stops,
                "Pixel observation migrated to an overlapping primitive");

        id = begin();
        {
            Work work;
            Operation operation(shader);
            uint32_t xyz[3]{7, 0, 0};
            uint64_t value = 42;
            nacelle_source_debug_observe(1, 1, 0, 1, xyz, &value);
        }
        require(NacelleEndSourceDebug(id, &summary) == S_FALSE && summary.target_seen &&
                !summary.cancelled && !summary.stops,
                "Inactive target was not reported explicitly or looked like a successful step");

        id = begin();
        std::promise<void> entered, release;
        auto released = release.get_future();
        std::atomic<int> result{0};
        std::thread worker([&] {
            Work work;
            entered.set_value();
            released.wait();
            result = 42;
        });
        entered.get_future().wait();
        require(NacelleControlSourceDebug(id, 0, NACELLE_SOURCE_CANCEL) == S_OK, "In-flight cancellation failed");
        auto ending = std::async(std::launch::async, [&] { return NacelleEndSourceDebug(id, &summary); });
        const bool waited = ending.wait_for(std::chrono::milliseconds(25)) == std::future_status::timeout;
        release.set_value();
        worker.join();
        require(waited && ending.get() == S_FALSE && result == 42 && summary.cancelled,
                "End failed to drain original cancelled execution");

        id = begin();
        int originalResult = 1;
        {
            Work first;
            std::thread concurrent([&] { Work rejected; originalResult += 6; });
            concurrent.join();
            originalResult *= 6;
        }
        require(NacelleEndSourceDebug(id, &summary) == S_FALSE && summary.concurrent_work &&
                originalResult == 42, "Concurrent work changed original execution or looked successful");
        id = begin();
        { Work first; }
        { Work subsequent; }
        require(NacelleEndSourceDebug(id, &summary) == S_FALSE && !summary.concurrent_work &&
                !summary.cancelled && !summary.stops,
                "Subsequent readback submission was captured or rejected");
        std::puts("SOURCE DEBUG RUNTIME PASS");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "source_debug_test: %s\n", error.what());
        return 1;
    }
}
