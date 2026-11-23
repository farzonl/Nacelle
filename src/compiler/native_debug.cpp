#include "native_debug.h"
#include "source_debug.h"
#include "jit_engine.h"
#include <nacelle/native_debug.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <vector>

#include <llvm/ExecutionEngine/JITEventListener.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ObjectTransformLayer.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/Module.h>
#include <llvm/Object/ObjectFile.h>
#include <llvm/Object/SymbolSize.h>

namespace nacelle::native_debug {
namespace {
struct Object {
    std::vector<NacelleNativeDebugCode> code;
};
struct State {
    std::mutex mutex;
    std::atomic<bool> enabled{false};
    bool frozen = false;
    uint64_t limit = 32ull * 1024 * 1024;
    uint64_t used = 0;
    uint64_t sequence = 0;
    std::map<const char*, uint64_t> pending;
    std::map<uint64_t, Object> objects;
};

std::shared_ptr<State> state() {
    static auto value = std::make_shared<State>();
    return value;
}

llvm::Error failure(const char* message) {
    return llvm::make_error<llvm::StringError>(message, llvm::inconvertibleErrorCode());
}

class Listener final : public llvm::JITEventListener {
public:
    explicit Listener(llvm::JITEventListener& debugger) : debugger(debugger), data(state()) {}

    void notifyObjectLoaded(ObjectKey key, const llvm::object::ObjectFile& object,
                            const llvm::RuntimeDyld::LoadedObjectInfo& loaded) override {
        Object record;
        const auto buffer = object.getMemoryBufferRef();
        const auto identity = profileContentIdentity(buffer.getBufferStart(), buffer.getBufferSize());
        for (const auto& [symbol, size] : llvm::object::computeSymbolSizes(object)) {
            // These fields were validated by object admission before linking.
            const auto name = llvm::cantFail(symbol.getName());
            const auto address = llvm::cantFail(symbol.getAddress());
            const auto section = llvm::cantFail(symbol.getSection());
            if (section == object.section_end() || !size) continue;
            if (!(name.starts_with("nacelle_cs_") || name.starts_with("nacelle_vs_") ||
                  name.starts_with("nacelle_ps_"))) continue;
            NacelleNativeDebugCode code;
            code.begin = loaded.getSectionLoadAddress(*section) + address - section->getAddress();
            code.end = code.begin + size;
            std::memcpy(code.symbol, name.data(), name.size());
            std::memcpy(code.object_sha256, identity.data(), identity.size());
            record.code.push_back(code);
        }
        {
            std::lock_guard lock(data->mutex);
            for (auto& code : record.code) code.code_version = ++data->sequence;
            data->objects.emplace(key, std::move(record));
        }
        debugger.notifyObjectLoaded(key, object, loaded);
    }

    void notifyFreeingObject(ObjectKey key) override {
        debugger.notifyFreeingObject(key);
        std::lock_guard lock(data->mutex);
        const auto object = data->objects.find(key);
        if (object != data->objects.end()) {
            data->objects.erase(object);
        }
    }
private:
    llvm::JITEventListener& debugger;
    std::shared_ptr<State> data;
};

class MemoryManager final : public llvm::SectionMemoryManager {
public:
    explicit MemoryManager(const llvm::MemoryBuffer& buffer) : data(state()) {
        std::lock_guard lock(data->mutex);
        const auto pending = data->pending.find(buffer.getBufferStart());
        if (pending != data->pending.end()) {
            bytes = pending->second;
            data->pending.erase(pending);
        }
    }
    ~MemoryManager() override {
        std::lock_guard lock(data->mutex);
        data->used -= bytes;
    }
private:
    std::shared_ptr<State> data;
    uint64_t bytes = 0;
};
}

bool enabled() { return state()->enabled.load(std::memory_order_acquire); }

bool freeze() {
    source_debug::freeze();
    const auto data = state();
    std::lock_guard lock(data->mutex);
    data->frozen = true;
    return data->enabled;
}

void prepareModule(llvm::Module& module) {
    llvm::StripDebugInfo(module);
    module.setSourceFileName("nacelle-native-debug");
    for (auto& function : module) {
        if (function.isDeclaration()) continue;
        function.addFnAttr("frame-pointer", "all");
        function.setUWTableKind(llvm::UWTableKind::Default);
    }
}

bool configureBuilder(llvm::orc::LLJITBuilder& builder,
                      std::shared_ptr<llvm::JITEventListener>& listener, std::string& error) {
    auto* debugger = llvm::JITEventListener::createGDBRegistrationListener();
    if (!debugger) {
        error = "LLVM native JIT debugger registration is unavailable";
        return false;
    }
    listener = std::make_shared<Listener>(*debugger);
    builder.setNumCompileThreads(0);
    builder.setObjectLinkingLayerCreator([listener](llvm::orc::ExecutionSession& session,
#if LLVM_VERSION_MAJOR >= 24
                                                   llvm::jitlink::JITLinkMemoryManager&
#else
                                                   const llvm::Triple&
#endif
                                                   )
        -> llvm::Expected<std::unique_ptr<llvm::orc::ObjectLayer>> {
        auto layer = std::make_unique<llvm::orc::RTDyldObjectLinkingLayer>(session,
            [](const llvm::MemoryBuffer& buffer) { return std::make_unique<MemoryManager>(buffer); });
        layer->registerJITEventListener(*listener);
        return std::unique_ptr<llvm::orc::ObjectLayer>(std::move(layer));
    });
    return true;
}

void installObjectAdmission(llvm::orc::LLJIT& jit) {
    jit.getObjTransformLayer().setTransform([data = state()](std::unique_ptr<llvm::MemoryBuffer> buffer)
        -> llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>> {
        if (buffer->getBufferSize() > data->limit / 2)
            return failure("Native debug object budget exhausted");
        auto object = llvm::object::ObjectFile::createObjectFile(buffer->getMemBufferRef());
        if (!object) return object.takeError();
        if (!(*object)->isELF()) return failure("Native debugging requires ELF objects");
        uint64_t symbols = 0;
        for (const auto& symbol : (*object)->symbols()) {
            auto name = symbol.getName();
            if (!name) return name.takeError();
            auto address = symbol.getAddress();
            if (!address) return address.takeError();
            auto section = symbol.getSection();
            if (!section) return section.takeError();
            if (name->starts_with("nacelle_") && name->size() >= sizeof(NacelleNativeDebugCode::symbol))
                return failure("Native debug symbol name exceeds the bounded record");
            ++symbols;
        }
        const uint64_t bytes = buffer->getBufferSize() * 2ull + symbols * sizeof(NacelleNativeDebugCode) * 2 + 4096;
        std::lock_guard lock(data->mutex);
        if (bytes > data->limit - data->used) return failure("Native debug object budget exhausted");
        if (!data->pending.emplace(buffer->getBufferStart(), bytes).second)
            return failure("Duplicate pending native debug object");
        data->used += bytes;
        return std::move(buffer);
    });
}

std::string objectIdentity(uint64_t address) {
    NacelleNativeDebugCode code;
    return NacelleGetNativeDebugCode(address, &code) == S_OK ? code.object_sha256 : "";
}

bool verifyEntry(uint64_t address, std::string& error) {
    if (!enabled()) return true;
    NacelleNativeDebugCode code;
    if (NacelleGetNativeDebugCode(address, &code) == S_OK && code.begin == address) return true;
    error = "Native debug shader symbol/range registration is unavailable";
    return false;
}
}

extern "C" HRESULT NacelleConfigureNativeDebug(const NacelleNativeDebugOptions* options) {
    const auto fail = [](const char* message) {
        std::fprintf(stderr, "native-debug: %s\n", message);
        return E_INVALIDARG;
    };
    if (!options || options->struct_size != sizeof(*options) || options->version != 1 ||
        options->enabled > 1 || !options->object_bytes || options->object_bytes > 64ull * 1024 * 1024)
        return fail("Invalid configuration or object budget");
#if !defined(__linux__) || !defined(__x86_64__)
    if (options->enabled) return fail("Native debugging currently supports Linux x86_64 with LLDB only");
#endif
    const auto data = nacelle::native_debug::state();
    std::lock_guard lock(data->mutex);
    if (data->frozen) return fail("Configure before any pipeline compilation or cache lookup");
    data->limit = options->object_bytes;
    data->enabled.store(options->enabled != 0, std::memory_order_release);
    return S_OK;
}

extern "C" HRESULT NacelleGetNativeDebugCode(uint64_t address, NacelleNativeDebugCode* code) {
    if (!code || code->struct_size != sizeof(*code) || code->version != 1) {
        std::fprintf(stderr, "native-debug: Invalid code query\n");
        return E_INVALIDARG;
    }
    *code = {};
    const auto data = nacelle::native_debug::state();
    std::lock_guard lock(data->mutex);
    for (const auto& [key, object] : data->objects)
        for (const auto& entry : object.code)
            if (address >= entry.begin && address < entry.end) {
                *code = entry;
                return S_OK;
            }
    return S_FALSE;
}
