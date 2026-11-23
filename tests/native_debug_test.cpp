#include <nacelle/native_debug.h>
#include "compiler/native_debug.h"
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/TargetSelect.h>
#include <cstdio>
#include <stdexcept>
#include <string_view>
#include <type_traits>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template<typename T> T take(llvm::Expected<T> value) {
    if (!value) throw std::runtime_error(llvm::toString(value.takeError()));
    if constexpr (std::is_reference_v<T>) return *value;
    else return std::move(*value);
}
void check(llvm::Error error) {
    if (error) throw std::runtime_error(llvm::toString(std::move(error)));
}

llvm::orc::ResourceTrackerSP addFunction(llvm::orc::LLJIT& jit, llvm::orc::JITDylib& library) {
    auto context = std::make_unique<llvm::LLVMContext>();
    auto module = std::make_unique<llvm::Module>("native-debug-lifetime", *context);
    module->setDataLayout(jit.getDataLayout());
    module->setTargetTriple(jit.getTargetTriple());
    llvm::IRBuilder<> builder(*module);
    auto* function = llvm::Function::Create(
        llvm::FunctionType::get(builder.getInt32Ty(), {builder.getInt32Ty()}, false),
        llvm::GlobalValue::ExternalLinkage, "nacelle_cs_lifetime", *module);
    builder.SetInsertPoint(llvm::BasicBlock::Create(*context, "entry", function));
    builder.CreateRet(builder.CreateAdd(function->getArg(0), builder.getInt32(7)));
    nacelle::native_debug::prepareModule(*module);
    auto tracker = library.createResourceTracker();
    check(jit.addIRModule(tracker, llvm::orc::ThreadSafeModule(std::move(module), std::move(context))));
    return tracker;
}
}

extern "C" __attribute__((noinline)) void nacelle_native_debug_unloaded() {
    std::puts("NATIVE DEBUG UNLOADED");
}

int main(int argc, char** argv) {
    try {
        const bool budget = argc == 2 && std::string_view(argv[1]) == "budget";
        NacelleNativeDebugOptions options;
        require(FAILED(NacelleConfigureNativeDebug(nullptr)), "Null config accepted");
        options.version = 2;
        require(FAILED(NacelleConfigureNativeDebug(&options)), "Unknown version accepted");
        options.version = 1;
        options.enabled = 2;
        require(FAILED(NacelleConfigureNativeDebug(&options)), "Invalid enabled accepted");
        options.enabled = 1;
        options.object_bytes = 64ull * 1024 * 1024 + 1;
        require(FAILED(NacelleConfigureNativeDebug(&options)), "Excessive budget accepted");
        options.object_bytes = budget ? 1 : 64ull * 1024;
        require(NacelleConfigureNativeDebug(&options) == S_OK, "Configure failed");
        require(nacelle::native_debug::freeze(), "Debug mode not frozen");
        require(FAILED(NacelleConfigureNativeDebug(&options)), "Late config accepted");
        require(FAILED(NacelleGetNativeDebugCode(1, nullptr)), "Null query accepted");
        NacelleNativeDebugCode code;
        require(NacelleGetNativeDebugCode(1, &code) == S_FALSE && !code.begin, "Unknown code not unavailable");
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        std::shared_ptr<llvm::JITEventListener> listener;
        llvm::orc::LLJITBuilder builder;
        std::string error;
        require(nacelle::native_debug::configureBuilder(builder, listener, error), error.c_str());
        auto jit = take(builder.create());
        nacelle::native_debug::installObjectAdmission(*jit);
        std::string materializationError;
        jit->getExecutionSession().setErrorReporter([&](llvm::Error error) {
            materializationError = llvm::toString(std::move(error));
        });
        auto& library = take(jit->createJITDylib("lifetime"));
        auto tracker = addFunction(*jit, library);
        auto symbol = jit->lookup(library, "nacelle_cs_lifetime");
        if (budget) {
            require(!symbol, "Exhausted budget published executable code");
            llvm::consumeError(symbol.takeError());
            require(materializationError.find("budget exhausted") != std::string::npos, "Missing budget diagnostic");
            std::puts("NATIVE DEBUG BUDGET PASS");
            return 0;
        }
        const auto address = take(std::move(symbol)).getValue();
        require(NacelleGetNativeDebugCode(address, &code) == S_OK && code.begin == address &&
            code.end > address && code.code_version && code.object_sha256[0], "Missing registered native range");
        const auto version = code.code_version;
        require(reinterpret_cast<int(*)(int)>(address)(35) == 42, "Debug execution changed result");
        std::string unavailable;
        require(!nacelle::native_debug::verifyEntry(UINT64_MAX, unavailable) &&
                unavailable.find("unavailable") != std::string::npos, "Missing symbols were not explicit");
        check(tracker->remove());
        require(NacelleGetNativeDebugCode(address, &code) == S_FALSE && !code.begin, "Unloaded range remained live");
        nacelle_native_debug_unloaded();
        auto& replacement = take(jit->createJITDylib("replacement"));
        auto replacementTracker = addFunction(*jit, replacement);
        const auto newAddress = take(jit->lookup(replacement, "nacelle_cs_lifetime")).getValue();
        require(NacelleGetNativeDebugCode(newAddress, &code) == S_OK && code.code_version > version,
                "Replacement reused stale code version");
        require(reinterpret_cast<int(*)(int)>(newAddress)(35) == 42, "Replacement changed result");
        check(replacementTracker->remove());
        require(NacelleGetNativeDebugCode(newAddress, &code) == S_FALSE, "Replacement range remained live");
        for (unsigned index = 0; index < 16; ++index) {
            auto& cycle = take(jit->createJITDylib("reclaim-" + std::to_string(index)));
            auto resource = addFunction(*jit, cycle);
            const auto entry = take(jit->lookup(cycle, "nacelle_cs_lifetime")).getValue();
            require(NacelleGetNativeDebugCode(entry, &code) == S_OK, "Reclaimed budget lost registration");
            check(resource->remove());
            require(NacelleGetNativeDebugCode(entry, &code) == S_FALSE, "Reclaimed object remained live");
        }
        std::puts("NATIVE DEBUG LIFETIME PASS");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "native_debug_test: %s\n", error.what());
        return 1;
    }
}
