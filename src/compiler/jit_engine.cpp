#include "jit_engine.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <unordered_map>

#define XXH_INLINE_ALL
#include <xxhash.h>

#include <BS_thread_pool.hpp>

#include <llvm/ADT/SmallString.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/ExecutionEngine/Orc/Debugging/DebuggerSupportPlugin.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/CodeGen.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/TargetParser/Host.h>
#include <llvm/TargetParser/Triple.h>
#include <llvm/Transforms/IPO/AlwaysInliner.h>

#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h>
#include <mlir/Conversion/MathToLLVM/MathToLLVM.h>
#include <mlir/Conversion/MathToLibm/MathToLibm.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Conversion/UBToLLVM/UBToLLVM.h>
#include <mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Export.h>
#include <mlir/Transforms/Passes.h>

#include "dxil_loader.h"
#include "mlir_pipeline.h"
#include "simt_to_simd_pass.h"
#include "texture_sampler_bc.h"

namespace nacelle {

CompiledComputePSO::~CompiledComputePSO() {
    if (asyncBox) {
        std::lock_guard<std::mutex> lock(asyncBox->mu);
        asyncBox->cs = nullptr;
    }
}

CompiledGraphicsPSO::~CompiledGraphicsPSO() {
    if (asyncBox) {
        std::lock_guard<std::mutex> lock(asyncBox->mu);
        asyncBox->gs = nullptr;
    }
}

static std::atomic<uint32_t> g_simdWidthOverride{0};

void setSimdWidthOverride(uint32_t width) {
    if (width == 0 || width == 4 || width == 8 || width == 16) {
        g_simdWidthOverride.store(width, std::memory_order_relaxed);
    }
}

uint32_t getTargetSimdWidth() {
    uint32_t ov = g_simdWidthOverride.load(std::memory_order_relaxed);
    if (ov == 4 || ov == 8 || ov == 16) return ov;
    if (const char* env = std::getenv("NACELLE_SIMD_WIDTH")) {
        int w = std::atoi(env);
        if (w == 4 || w == 8 || w == 16) return static_cast<uint32_t>(w);
    }
#if defined(__aarch64__) || defined(_M_ARM64)
    return 4;
#else
    auto features = llvm::sys::getHostCPUFeatures();
    if (features.lookup("avx512f")) return 16;
    if (features.lookup("avx2")) return 8;
    return 4;
#endif
}

std::string hashToHex128(uint64_t lo, uint64_t hi) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out(32, '0');
    for (int i = 0; i < 16; ++i) {
        out[i] = kHex[(hi >> ((15 - i) * 4)) & 0xFu];
        out[16 + i] = kHex[(lo >> ((15 - i) * 4)) & 0xFu];
    }
    return out;
}

namespace {

constexpr uint32_t kCompilerMagicVersion = 0x4E434A31; // 'NCJ1'

struct HashFeed {
    std::vector<uint8_t> buf;
    template <typename T>
    void add(const T& v) {
        const auto* p = reinterpret_cast<const uint8_t*>(&v);
        buf.insert(buf.end(), p, p + sizeof(T));
    }
    void addBytes(const void* data, size_t size) {
        if (data && size > 0) {
            const auto* p = static_cast<const uint8_t*>(data);
            buf.insert(buf.end(), p, p + size);
        }
    }
    void addStr(const char* s) {
        uint32_t len = s ? static_cast<uint32_t>(std::strlen(s)) : 0;
        add(len);
        if (len > 0) addBytes(s, len);
    }
};

struct InMemCacheEntry {
    CachedPSOMeta meta{};
    uint64_t addr0 = 0; // cs or vs
    uint64_t addr1 = 0; // ps (graphics)
    std::vector<uint8_t> objBytes;
};

struct JitRuntime {
    std::mutex mu;
    std::unique_ptr<llvm::orc::LLJIT> jit;
    std::unordered_map<std::string, InMemCacheEntry> memCache;
    std::atomic<uint64_t> dylibSeq{1};
    std::atomic<uint64_t> tmpSeq{1};
    BS::thread_pool<> tier2Pool{2};

    ~JitRuntime() {
        tier2Pool.wait();
    }
};

JitRuntime& getJitRuntime() {
    static JitRuntime runtime;
    return runtime;
}

void ensureLLJITLocked(JitRuntime& rt, std::string& outError) {
    if (rt.jit) return;
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();

    auto jtmbOrErr = llvm::orc::JITTargetMachineBuilder::detectHost();
    if (!jtmbOrErr) {
        outError = llvm::toString(jtmbOrErr.takeError());
        return;
    }
    jtmbOrErr->setCodeGenOptLevel(llvm::CodeGenOptLevel::Aggressive);
    llvm::StringMap<bool, llvm::MallocAllocator> hostFeatures = llvm::sys::getHostCPUFeatures();
    for (const auto& kv : hostFeatures) {
        jtmbOrErr->getFeatures().AddFeature(kv.first(), kv.second);
    }
    llvm::StringRef cpu = llvm::sys::getHostCPUName();
    if (!cpu.empty()) {
        jtmbOrErr->setCPU(cpu.str());
    }

    auto jitOrErr = llvm::orc::LLJITBuilder()
                        .setJITTargetMachineBuilder(std::move(*jtmbOrErr))
                        .create();
    if (!jitOrErr) {
        outError = llvm::toString(jitOrErr.takeError());
        return;
    }
    rt.jit = std::move(*jitOrErr);

    if (const char* prof = std::getenv("NACELLE_JIT_PROFILING")) {
        if (prof[0] == '1') {
            if (auto* oll = llvm::dyn_cast<llvm::orc::ObjectLinkingLayer>(&rt.jit->getObjLinkingLayer())) {
                if (rt.jit->getTargetTriple().isOSBinFormatMachO()) {
#if LLVM_VERSION_MAJOR >= 24
                    auto pluginOrErr = llvm::orc::GDBJITDebugInfoRegistrationPlugin::Create(
                        rt.jit->getExecutionSession(),
                        *rt.jit->getProcessSymbolsJITDylib());
#else
                    auto pluginOrErr = llvm::orc::GDBJITDebugInfoRegistrationPlugin::Create(
                        rt.jit->getExecutionSession(),
                        *rt.jit->getProcessSymbolsJITDylib(),
                        rt.jit->getTargetTriple());
#endif
                    if (pluginOrErr) {
                        oll->addPlugin(std::move(*pluginOrErr));
                    } else {
                        llvm::consumeError(pluginOrErr.takeError());
                    }
                }
            }
        }
    }
}

llvm::Expected<llvm::orc::JITDylib&> createIsolatedDylibLocked(JitRuntime& rt) {
    std::string jdName = "nacelle_pso_jd_" + std::to_string(rt.dylibSeq.fetch_add(1, std::memory_order_relaxed));
    return rt.jit->createJITDylib(std::move(jdName));
}

bool emitObjectBytesO3(
    llvm::Module& mod,
    llvm::SmallVectorImpl<char>& outObjBytes,
    std::string& outError) {
    auto jtmbOrErr = llvm::orc::JITTargetMachineBuilder::detectHost();
    if (!jtmbOrErr) {
        outError = llvm::toString(jtmbOrErr.takeError());
        return false;
    }
    jtmbOrErr->setCodeGenOptLevel(llvm::CodeGenOptLevel::Aggressive);
    llvm::StringMap<bool, llvm::MallocAllocator> hostFeatures = llvm::sys::getHostCPUFeatures();
    for (const auto& kv : hostFeatures) {
        jtmbOrErr->getFeatures().AddFeature(kv.first(), kv.second);
    }
    llvm::StringRef cpu = llvm::sys::getHostCPUName();
    if (!cpu.empty()) jtmbOrErr->setCPU(cpu.str());

    auto tmOrErr = jtmbOrErr->createTargetMachine();
    if (!tmOrErr) {
        outError = llvm::toString(tmOrErr.takeError());
        return false;
    }
    std::unique_ptr<llvm::TargetMachine> tm = std::move(*tmOrErr);
    mod.setTargetTriple(tm->getTargetTriple());
    mod.setDataLayout(tm->createDataLayout());

    llvm::PipelineTuningOptions pto;
    pto.LoopVectorization = true;
    pto.SLPVectorization = true;
    pto.LoopInterleaving = true;
    pto.LoopUnrolling = true;

    llvm::LoopAnalysisManager lam;
    llvm::FunctionAnalysisManager fam;
    llvm::CGSCCAnalysisManager cgam;
    llvm::ModuleAnalysisManager mam;

#if LLVM_VERSION_MAJOR >= 24
    llvm::PassBuilder pb(mod.getContext(), tm.get(), pto);
#else
    llvm::PassBuilder pb(tm.get(), pto);
#endif
    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    pb.crossRegisterProxies(lam, fam, cgam, mam);

    llvm::ModulePassManager mpm = pb.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3);
    mpm.run(mod, mam);

    llvm::legacy::PassManager codegenPM;
    llvm::raw_svector_ostream objStream(outObjBytes);
    if (tm->addPassesToEmitFile(codegenPM, objStream, nullptr, llvm::CodeGenFileType::ObjectFile)) {
        outError = "TargetMachine cannot emit object file";
        return false;
    }
    codegenPM.run(mod);
    return true;
}

void runTier1Optimize(llvm::Module& mod) {
    llvm::LoopAnalysisManager lam;
    llvm::FunctionAnalysisManager fam;
    llvm::CGSCCAnalysisManager cgam;
    llvm::ModuleAnalysisManager mam;

#if LLVM_VERSION_MAJOR >= 24
    llvm::PassBuilder pb(mod.getContext());
#else
    llvm::PassBuilder pb;
#endif
    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    pb.crossRegisterProxies(lam, fam, cgam, mam);

    llvm::ModulePassManager mpm = pb.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O1);
    mpm.run(mod, mam);
}

bool writeFileAtomic(const std::filesystem::path& dst, const void* data, size_t size, uint64_t tmpId) {
    std::error_code ec;
    std::filesystem::create_directories(dst.parent_path(), ec);
    if (ec) return false;

    std::filesystem::path tmp = dst;
    tmp += ".tmp." + std::to_string(tmpId);
    {
        std::ofstream ofs(tmp, std::ios::binary | std::ios::trunc);
        if (!ofs) return false;
        ofs.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
        ofs.flush();
        if (!ofs) return false;
    }
    std::filesystem::rename(tmp, dst, ec);
    if (ec) {
        std::filesystem::remove(tmp, ec);
        return false;
    }
    return true;
}

bool readFileBytes(const std::filesystem::path& src, std::vector<uint8_t>& out) {
    std::ifstream ifs(src, std::ios::binary | std::ios::ate);
    if (!ifs) return false;
    auto sz = ifs.tellg();
    if (sz < 0) return false;
    out.resize(static_cast<size_t>(sz));
    ifs.seekg(0, std::ios::beg);
    if (sz > 0 && !ifs.read(reinterpret_cast<char*>(out.data()), sz)) return false;
    return true;
}

std::string cacheKeyFor(const std::filesystem::path& dir, uint64_t lo, uint64_t hi) {
    return (dir / hashToHex128(lo, hi)).string();
}

void applyComputeMeta(const CachedPSOMeta& meta, const std::string& sym, uint64_t addr, CompiledComputePSO& outPSO) {
    outPSO.threadGroupSize[0] = meta.threadGroupSize[0];
    outPSO.threadGroupSize[1] = meta.threadGroupSize[1];
    outPSO.threadGroupSize[2] = meta.threadGroupSize[2];
    outPSO.groupSharedBytes = meta.groupSharedBytes;
    outPSO.simdWidth = meta.simdWidth;
    outPSO.symbolName = sym;
    outPSO.fn.store(reinterpret_cast<CompiledComputeFn>(addr), std::memory_order_release);
}

void applyGraphicsMeta(
    const CachedPSOMeta& meta, const std::string& vsSym, const std::string& psSym,
    uint64_t vsAddr, uint64_t psAddr, CompiledGraphicsPSO& outPSO) {
    outPSO.simdWidth = meta.simdWidth;
    outPSO.numVSOutputRegs = meta.numVSOutputRegs;
    outPSO.posOutputReg = meta.posOutputReg;
    outPSO.numPSInputRegs = meta.numPSInputRegs;
    outPSO.flatInterpMask = meta.flatInterpMask;
    outPSO.psWritesDepth = meta.psWritesDepth != 0;
    outPSO.psUsesDiscard = meta.psUsesDiscard != 0;
    outPSO.psHasSideEffects = meta.psHasSideEffects != 0;
    outPSO.vsSymbolName = vsSym;
    outPSO.psSymbolName = psSym;
    outPSO.vsFn.store(reinterpret_cast<CompiledVertexFn>(vsAddr), std::memory_order_release);
    outPSO.psFn.store(reinterpret_cast<CompiledPixelQuadFn>(psAddr), std::memory_order_release);
}

bool loadObjectIntoJitLocked(
    JitRuntime& rt,
    const uint8_t* objData,
    size_t objSize,
    const std::string& cacheName,
    const std::string& sym0,
    const std::string& sym1,
    uint64_t& outAddr0,
    uint64_t& outAddr1,
    std::string& outError) {
    ensureLLJITLocked(rt, outError);
    if (!rt.jit) return false;

    auto jdOrErr = createIsolatedDylibLocked(rt);
    if (!jdOrErr) {
        outError = llvm::toString(jdOrErr.takeError());
        return false;
    }
    llvm::orc::JITDylib& jd = *jdOrErr;

    auto buf = llvm::MemoryBuffer::getMemBufferCopy(
        llvm::StringRef(reinterpret_cast<const char*>(objData), objSize), cacheName);
    if (auto err = rt.jit->addObjectFile(jd, std::move(buf))) {
        outError = llvm::toString(std::move(err));
        return false;
    }

    auto sym0OrErr = rt.jit->lookup(jd, sym0);
    if (!sym0OrErr) {
        outError = llvm::toString(sym0OrErr.takeError());
        return false;
    }
    outAddr0 = sym0OrErr->getValue();

    if (!sym1.empty()) {
        auto sym1OrErr = rt.jit->lookup(jd, sym1);
        if (!sym1OrErr) {
            outError = llvm::toString(sym1OrErr.takeError());
            return false;
        }
        outAddr1 = sym1OrErr->getValue();
    } else {
        outAddr1 = 0;
    }
    return true;
}

} // namespace

void computeComputePSOHash(
    const D3D12_COMPUTE_PIPELINE_STATE_DESC& desc,
    const RootSignatureLayout& rootLayout,
    uint32_t simdWidth,
    uint64_t& outLo,
    uint64_t& outHi) {
    HashFeed feed;
    feed.add(kCompilerMagicVersion);
    feed.addStr(getHostTargetTripleString().c_str());
    feed.add(simdWidth);
    feed.add(rootLayout.hashLo);
    feed.add(rootLayout.hashHi);
    uint64_t csLen = desc.CS.BytecodeLength;
    feed.add(csLen);
    feed.addBytes(desc.CS.pShaderBytecode, static_cast<size_t>(csLen));
    XXH128_hash_t h = XXH3_128bits(feed.buf.data(), feed.buf.size());
    outLo = h.low64;
    outHi = h.high64;
}

void computeGraphicsPSOHash(
    const D3D12_GRAPHICS_PIPELINE_STATE_DESC& desc,
    const RootSignatureLayout& rootLayout,
    uint32_t simdWidth,
    uint64_t& outLo,
    uint64_t& outHi) {
    HashFeed feed;
    feed.add(kCompilerMagicVersion);
    feed.addStr(getHostTargetTripleString().c_str());
    feed.add(simdWidth);
    feed.add(rootLayout.hashLo);
    feed.add(rootLayout.hashHi);

    auto addShader = [&](const D3D12_SHADER_BYTECODE& sb) {
        uint64_t len = sb.BytecodeLength;
        feed.add(len);
        feed.addBytes(sb.pShaderBytecode, static_cast<size_t>(len));
    };
    addShader(desc.VS);
    addShader(desc.PS);
    addShader(desc.GS);
    addShader(desc.DS);
    addShader(desc.HS);

    feed.add(desc.BlendState);
    feed.add(desc.SampleMask);
    feed.add(desc.RasterizerState);
    feed.add(desc.DepthStencilState);
    feed.add(desc.IBStripCutValue);
    feed.add(desc.PrimitiveTopologyType);
    feed.add(desc.NumRenderTargets);
    feed.addBytes(desc.RTVFormats, sizeof(desc.RTVFormats));
    feed.add(desc.DSVFormat);
    feed.add(desc.SampleDesc);

    feed.add(desc.InputLayout.NumElements);
    for (UINT i = 0; i < desc.InputLayout.NumElements && desc.InputLayout.pInputElementDescs; ++i) {
        const auto& el = desc.InputLayout.pInputElementDescs[i];
        feed.addStr(el.SemanticName);
        feed.add(el.SemanticIndex);
        feed.add(el.Format);
        feed.add(el.InputSlot);
        feed.add(el.AlignedByteOffset);
        feed.add(el.InputSlotClass);
        feed.add(el.InstanceDataStepRate);
    }

    XXH128_hash_t h = XXH3_128bits(feed.buf.data(), feed.buf.size());
    outLo = h.low64;
    outHi = h.high64;
}

std::unique_ptr<llvm::Module> translateMLIRToLinkedLLVMIR(
    mlir::ModuleOp mlirMod,
    llvm::LLVMContext& llvmCtx,
    std::string& outError) {
    mlir::PassManager pm(mlirMod.getContext());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createSCFToControlFlowPass());
    pm.addPass(mlir::createConvertVectorToLLVMPass());
    pm.addPass(mlir::createConvertMathToLLVMPass());
    pm.addPass(mlir::createConvertMathToLibmPass());
    pm.addPass(mlir::createArithToLLVMConversionPass());
    pm.addPass(mlir::createConvertControlFlowToLLVMPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    pm.addPass(mlir::createUBToLLVMConversionPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());

    if (mlir::failed(pm.run(mlirMod))) {
        outError = "MLIR LLVM lowering pipeline failed";
        return nullptr;
    }

    auto llvmMod = mlir::translateModuleToLLVMIR(mlirMod, llvmCtx);
    if (!llvmMod) {
        outError = "translateModuleToLLVMIR failed";
        return nullptr;
    }

    llvmMod->setTargetTriple(llvm::Triple(getHostTargetTripleString()));
    llvmMod->setDataLayout(getHostDataLayoutString());

    bool needsTexRuntime = false;
    for (const llvm::Function& f : *llvmMod) {
        if (f.isDeclaration() && f.getName().starts_with("nacelle_tex_")) {
            needsTexRuntime = true;
            break;
        }
    }

    if (needsTexRuntime) {
        auto bcBuf = llvm::MemoryBuffer::getMemBuffer(
            llvm::StringRef(reinterpret_cast<const char*>(kTextureSamplerBitcode), kTextureSamplerBitcode_size),
            "texture_sampler.bc",
            false);
        auto bcModOrErr = llvm::parseBitcodeFile(bcBuf->getMemBufferRef(), llvmCtx);
        if (!bcModOrErr) {
            outError = "Failed to parse embedded texture_sampler.bc: " + llvm::toString(bcModOrErr.takeError());
            return nullptr;
        }
        std::unique_ptr<llvm::Module> bcMod = std::move(*bcModOrErr);
        bcMod->setTargetTriple(llvmMod->getTargetTriple());
        bcMod->setDataLayout(llvmMod->getDataLayout());

        if (llvm::Linker::linkModules(*llvmMod, std::move(bcMod), llvm::Linker::Flags::LinkOnlyNeeded)) {
            outError = "llvm::Linker failed linking texture_sampler.bc";
            return nullptr;
        }

        for (llvm::Function& f : *llvmMod) {
            if (!f.isDeclaration() && f.getName().starts_with("nacelle_tex_")) {
                f.setLinkage(llvm::GlobalValue::InternalLinkage);
                f.addFnAttr(llvm::Attribute::AlwaysInline);
            }
        }

        llvm::LoopAnalysisManager lam;
        llvm::FunctionAnalysisManager fam;
        llvm::CGSCCAnalysisManager cgam;
        llvm::ModuleAnalysisManager mam;
#if LLVM_VERSION_MAJOR >= 24
        llvm::PassBuilder pb(llvmMod->getContext());
#else
        llvm::PassBuilder pb;
#endif
        pb.registerModuleAnalyses(mam);
        pb.registerCGSCCAnalyses(cgam);
        pb.registerFunctionAnalyses(fam);
        pb.registerLoopAnalyses(lam);
        pb.crossRegisterProxies(lam, fam, cgam, mam);

        llvm::ModulePassManager inlinerPM;
        inlinerPM.addPass(llvm::AlwaysInlinerPass());
        inlinerPM.run(*llvmMod, mam);
    }

    return llvmMod;
}

bool tryLoadComputePSOFromCache(
    uint64_t hashLo, uint64_t hashHi,
    const std::string& symbolName,
    CompiledComputePSO& outPSO) {
    JitRuntime& rt = getJitRuntime();
    std::filesystem::path dir = getCacheDirectory();
    std::string key = cacheKeyFor(dir, hashLo, hashHi);
    uint32_t targetW = getTargetSimdWidth();

    {
        std::lock_guard<std::mutex> lock(rt.mu);
        auto it = rt.memCache.find(key);
        if (it != rt.memCache.end() && it->second.addr0 != 0 && it->second.meta.simdWidth == targetW) {
            applyComputeMeta(it->second.meta, symbolName, it->second.addr0, outPSO);
            return true;
        }
    }

    std::filesystem::path metaPath = dir / (hashToHex128(hashLo, hashHi) + ".meta");
    std::filesystem::path objPath = dir / (hashToHex128(hashLo, hashHi) + ".o");
    std::vector<uint8_t> metaBytes, objBytes;
    if (!readFileBytes(metaPath, metaBytes) || metaBytes.size() != sizeof(CachedPSOMeta)) return false;
    CachedPSOMeta meta{};
    std::memcpy(&meta, metaBytes.data(), sizeof(CachedPSOMeta));
    if (meta.magic != CachedPSOMeta{}.magic || meta.simdWidth != targetW) return false;
    if (!readFileBytes(objPath, objBytes) || objBytes.empty()) return false;

    std::lock_guard<std::mutex> lock(rt.mu);
    auto it = rt.memCache.find(key);
    if (it != rt.memCache.end() && it->second.addr0 != 0 && it->second.meta.simdWidth == targetW) {
        applyComputeMeta(it->second.meta, symbolName, it->second.addr0, outPSO);
        return true;
    }

    uint64_t addr0 = 0, addr1 = 0;
    std::string err;
    if (!loadObjectIntoJitLocked(rt, objBytes.data(), objBytes.size(), key + ".o", symbolName, "", addr0, addr1, err)) {
        return false;
    }

    rt.memCache[key] = InMemCacheEntry{meta, addr0, 0, std::move(objBytes)};
    applyComputeMeta(meta, symbolName, addr0, outPSO);
    return true;
}

bool tryLoadGraphicsPSOFromCache(
    uint64_t hashLo, uint64_t hashHi,
    const std::string& vsSym,
    const std::string& psSym,
    CompiledGraphicsPSO& outPSO) {
    JitRuntime& rt = getJitRuntime();
    std::filesystem::path dir = getCacheDirectory();
    std::string key = cacheKeyFor(dir, hashLo, hashHi);
    uint32_t targetW = getTargetSimdWidth();

    {
        std::lock_guard<std::mutex> lock(rt.mu);
        auto it = rt.memCache.find(key);
        if (it != rt.memCache.end() && it->second.addr0 != 0 && (psSym.empty() || it->second.addr1 != 0) &&
            it->second.meta.simdWidth == targetW) {
            applyGraphicsMeta(it->second.meta, vsSym, psSym, it->second.addr0, it->second.addr1, outPSO);
            return true;
        }
    }

    std::filesystem::path metaPath = dir / (hashToHex128(hashLo, hashHi) + ".meta");
    std::filesystem::path objPath = dir / (hashToHex128(hashLo, hashHi) + ".o");
    std::vector<uint8_t> metaBytes, objBytes;
    if (!readFileBytes(metaPath, metaBytes) || metaBytes.size() != sizeof(CachedPSOMeta)) return false;
    CachedPSOMeta meta{};
    std::memcpy(&meta, metaBytes.data(), sizeof(CachedPSOMeta));
    if (meta.magic != CachedPSOMeta{}.magic || meta.simdWidth != targetW) return false;
    if (!readFileBytes(objPath, objBytes) || objBytes.empty()) return false;

    std::lock_guard<std::mutex> lock(rt.mu);
    auto it = rt.memCache.find(key);
    if (it != rt.memCache.end() && it->second.addr0 != 0 && (psSym.empty() || it->second.addr1 != 0) &&
        it->second.meta.simdWidth == targetW) {
        applyGraphicsMeta(it->second.meta, vsSym, psSym, it->second.addr0, it->second.addr1, outPSO);
        return true;
    }

    uint64_t addr0 = 0, addr1 = 0;
    std::string err;
    if (!loadObjectIntoJitLocked(rt, objBytes.data(), objBytes.size(), key + ".o", vsSym, psSym, addr0, addr1, err)) {
        return false;
    }

    rt.memCache[key] = InMemCacheEntry{meta, addr0, addr1, std::move(objBytes)};
    applyGraphicsMeta(meta, vsSym, psSym, addr0, addr1, outPSO);
    return true;
}

bool jitCompileComputeTier1AndScheduleTier2(
    std::unique_ptr<llvm::Module> llvmMod,
    std::unique_ptr<llvm::LLVMContext> llvmCtx,
    const std::string& symbolName,
    const CachedPSOMeta& meta,
    CompiledComputePSO& outPSO,
    std::string& outError) {
    JitRuntime& rt = getJitRuntime();

    // Snapshot unoptimized IR bitcode before Tier-1 mutates `llvmMod`.
    llvm::SmallString<0> bcBytes;
    {
        llvm::raw_svector_ostream bcOS(bcBytes);
        llvm::WriteBitcodeToFile(*llvmMod, bcOS);
    }

    runTier1Optimize(*llvmMod);

    uint64_t tier1Addr = 0;
    {
        std::lock_guard<std::mutex> lock(rt.mu);
        ensureLLJITLocked(rt, outError);
        if (!rt.jit) return false;
        auto jdOrErr = createIsolatedDylibLocked(rt);
        if (!jdOrErr) {
            outError = llvm::toString(jdOrErr.takeError());
            return false;
        }
        llvm::orc::JITDylib& jd = *jdOrErr;
        if (auto err = rt.jit->addIRModule(jd, llvm::orc::ThreadSafeModule(std::move(llvmMod), std::move(llvmCtx)))) {
            outError = llvm::toString(std::move(err));
            return false;
        }
        auto symOrErr = rt.jit->lookup(jd, symbolName);
        if (!symOrErr) {
            outError = llvm::toString(symOrErr.takeError());
            return false;
        }
        tier1Addr = symOrErr->getValue();
    }

    outPSO.symbolName = symbolName;
    outPSO.fn.store(reinterpret_cast<CompiledComputeFn>(tier1Addr), std::memory_order_release);

    if (const char* noT2 = std::getenv("NACELLE_DISABLE_TIER2"); noT2 && noT2[0] == '1') {
        return true;
    }

    auto box = std::make_shared<PsoUpdateBox>();
    box->cs = &outPSO;
    outPSO.asyncBox = box;

    std::filesystem::path cacheDir = getCacheDirectory();
    std::string hex = hashToHex128(outPSO.cacheHashLo, outPSO.cacheHashHi);
    std::string bcStr(bcBytes.data(), bcBytes.size());

    rt.tier2Pool.detach_task([box, meta, symbolName, cacheDir, hex, bcStr = std::move(bcStr)]() {
        auto ctx2 = std::make_unique<llvm::LLVMContext>();
        auto buf = llvm::MemoryBuffer::getMemBuffer(bcStr, hex + ".bc", false);
        auto modOrErr = llvm::parseBitcodeFile(buf->getMemBufferRef(), *ctx2);
        if (!modOrErr) {
            llvm::consumeError(modOrErr.takeError());
            return;
        }
        std::unique_ptr<llvm::Module> mod2 = std::move(*modOrErr);
        llvm::SmallVector<char, 0> objBytes;
        std::string err;
        if (!emitObjectBytesO3(*mod2, objBytes, err)) return;

        JitRuntime& rt = getJitRuntime();
        uint64_t tId = rt.tmpSeq.fetch_add(1, std::memory_order_relaxed);
        writeFileAtomic(cacheDir / (hex + ".o"), objBytes.data(), objBytes.size(), tId);
        writeFileAtomic(cacheDir / (hex + ".meta"), &meta, sizeof(CachedPSOMeta), tId);

        uint64_t addr0 = 0, addr1 = 0;
        {
            std::lock_guard<std::mutex> lock(rt.mu);
            std::string key = (cacheDir / hex).string();
            if (!loadObjectIntoJitLocked(rt, reinterpret_cast<const uint8_t*>(objBytes.data()),
                                         objBytes.size(), key + ".o", symbolName, "", addr0, addr1, err)) {
                return;
            }
            rt.memCache[key] = InMemCacheEntry{
                meta, addr0, 0,
                std::vector<uint8_t>(reinterpret_cast<const uint8_t*>(objBytes.begin()),
                                     reinterpret_cast<const uint8_t*>(objBytes.end()))};
        }

        std::lock_guard<std::mutex> psoLock(box->mu);
        if (box->cs) {
            box->cs->fn.store(reinterpret_cast<CompiledComputeFn>(addr0), std::memory_order_release);
        }
    });

    return true;
}

bool jitCompileGraphicsTier1AndScheduleTier2(
    std::unique_ptr<llvm::Module> llvmMod,
    std::unique_ptr<llvm::LLVMContext> llvmCtx,
    const std::string& vsSym,
    const std::string& psSym,
    const CachedPSOMeta& meta,
    CompiledGraphicsPSO& outPSO,
    std::string& outError) {
    JitRuntime& rt = getJitRuntime();

    llvm::SmallString<0> bcBytes;
    {
        llvm::raw_svector_ostream bcOS(bcBytes);
        llvm::WriteBitcodeToFile(*llvmMod, bcOS);
    }

    runTier1Optimize(*llvmMod);

    uint64_t vsAddr = 0, psAddr = 0;
    {
        std::lock_guard<std::mutex> lock(rt.mu);
        ensureLLJITLocked(rt, outError);
        if (!rt.jit) return false;
        auto jdOrErr = createIsolatedDylibLocked(rt);
        if (!jdOrErr) {
            outError = llvm::toString(jdOrErr.takeError());
            return false;
        }
        llvm::orc::JITDylib& jd = *jdOrErr;
        if (auto err = rt.jit->addIRModule(jd, llvm::orc::ThreadSafeModule(std::move(llvmMod), std::move(llvmCtx)))) {
            outError = llvm::toString(std::move(err));
            return false;
        }
        auto vsOrErr = rt.jit->lookup(jd, vsSym);
        if (!vsOrErr) {
            outError = llvm::toString(vsOrErr.takeError());
            return false;
        }
        vsAddr = vsOrErr->getValue();

        auto psOrErr = rt.jit->lookup(jd, psSym);
        if (!psOrErr) {
            outError = llvm::toString(psOrErr.takeError());
            return false;
        }
        psAddr = psOrErr->getValue();
    }

    outPSO.vsSymbolName = vsSym;
    outPSO.psSymbolName = psSym;
    outPSO.vsFn.store(reinterpret_cast<CompiledVertexFn>(vsAddr), std::memory_order_release);
    outPSO.psFn.store(reinterpret_cast<CompiledPixelQuadFn>(psAddr), std::memory_order_release);

    if (const char* noT2 = std::getenv("NACELLE_DISABLE_TIER2"); noT2 && noT2[0] == '1') {
        return true;
    }

    auto box = std::make_shared<PsoUpdateBox>();
    box->gs = &outPSO;
    outPSO.asyncBox = box;

    std::filesystem::path cacheDir = getCacheDirectory();
    std::string hex = hashToHex128(outPSO.cacheHashLo, outPSO.cacheHashHi);
    std::string bcStr(bcBytes.data(), bcBytes.size());

    rt.tier2Pool.detach_task([box, meta, vsSym, psSym, cacheDir, hex, bcStr = std::move(bcStr)]() {
        auto ctx2 = std::make_unique<llvm::LLVMContext>();
        auto buf = llvm::MemoryBuffer::getMemBuffer(bcStr, hex + ".bc", false);
        auto modOrErr = llvm::parseBitcodeFile(buf->getMemBufferRef(), *ctx2);
        if (!modOrErr) {
            llvm::consumeError(modOrErr.takeError());
            return;
        }
        std::unique_ptr<llvm::Module> mod2 = std::move(*modOrErr);
        llvm::SmallVector<char, 0> objBytes;
        std::string err;
        if (!emitObjectBytesO3(*mod2, objBytes, err)) return;

        JitRuntime& rt = getJitRuntime();
        uint64_t tId = rt.tmpSeq.fetch_add(1, std::memory_order_relaxed);
        writeFileAtomic(cacheDir / (hex + ".o"), objBytes.data(), objBytes.size(), tId);
        writeFileAtomic(cacheDir / (hex + ".meta"), &meta, sizeof(CachedPSOMeta), tId);

        uint64_t addr0 = 0, addr1 = 0;
        {
            std::lock_guard<std::mutex> lock(rt.mu);
            std::string key = (cacheDir / hex).string();
            if (!loadObjectIntoJitLocked(rt, reinterpret_cast<const uint8_t*>(objBytes.data()),
                                         objBytes.size(), key + ".o", vsSym, psSym, addr0, addr1, err)) {
                return;
            }
            rt.memCache[key] = InMemCacheEntry{
                meta, addr0, addr1,
                std::vector<uint8_t>(reinterpret_cast<const uint8_t*>(objBytes.begin()),
                                     reinterpret_cast<const uint8_t*>(objBytes.end()))};
        }

        std::lock_guard<std::mutex> psoLock(box->mu);
        if (box->gs) {
            box->gs->vsFn.store(reinterpret_cast<CompiledVertexFn>(addr0), std::memory_order_release);
            box->gs->psFn.store(reinterpret_cast<CompiledPixelQuadFn>(addr1), std::memory_order_release);
        }
    });

    return true;
}

bool compileComputePSO(
    const D3D12_COMPUTE_PIPELINE_STATE_DESC& desc,
    const RootSignatureLayout& rootLayout,
    CompiledComputePSO& outPSO,
    std::string& outError) {
    uint32_t simdWidth = getTargetSimdWidth();
    uint64_t lo = 0, hi = 0;
    computeComputePSOHash(desc, rootLayout, simdWidth, lo, hi);
    outPSO.cacheHashLo = lo;
    outPSO.cacheHashHi = hi;

    std::string hex = hashToHex128(lo, hi);
    std::string symbolName = "nacelle_cs_" + hex;

    if (tryLoadComputePSOFromCache(lo, hi, symbolName, outPSO)) {
        return true;
    }

    auto llvmCtx = std::make_unique<llvm::LLVMContext>();
    DxilLoadedShader shader;
    if (!loadAndUpgradeDXIL(desc.CS, *llvmCtx, shader, outError)) {
        return false;
    }

    const RootSignatureLayout& effectiveLayout =
        (rootLayout.params.empty() && rootLayout.staticSamplers.empty() && shader.hasEmbeddedRootSig)
            ? shader.embeddedRootSig
            : rootLayout;

    mlir::MLIRContext mlirCtx;
    auto mlirMod = importAndLiftDXILToSCF(std::move(shader.module), shader.entryName, mlirCtx, outError);
    if (!mlirMod) return false;

    if (!packetizeComputeEntry(*mlirMod, shader, effectiveLayout, simdWidth, symbolName, outError)) {
        return false;
    }

    auto outLlvmCtx = std::make_unique<llvm::LLVMContext>();
    auto linkedMod = translateMLIRToLinkedLLVMIR(*mlirMod, *outLlvmCtx, outError);
    if (!linkedMod) return false;

    CachedPSOMeta meta;
    meta.simdWidth = simdWidth;
    meta.threadGroupSize[0] = shader.threadGroupSize[0];
    meta.threadGroupSize[1] = shader.threadGroupSize[1];
    meta.threadGroupSize[2] = shader.threadGroupSize[2];
    meta.groupSharedBytes = shader.groupSharedBytes;

    outPSO.threadGroupSize[0] = meta.threadGroupSize[0];
    outPSO.threadGroupSize[1] = meta.threadGroupSize[1];
    outPSO.threadGroupSize[2] = meta.threadGroupSize[2];
    outPSO.groupSharedBytes = meta.groupSharedBytes;
    outPSO.simdWidth = simdWidth;

    return jitCompileComputeTier1AndScheduleTier2(
        std::move(linkedMod), std::move(outLlvmCtx), symbolName, meta, outPSO, outError);
}

void waitForAllAsyncJITTasks() {
    getJitRuntime().tier2Pool.wait();
}

void clearInMemoryPSOCacheForTesting() {
    JitRuntime& rt = getJitRuntime();
    std::lock_guard<std::mutex> lock(rt.mu);
    rt.memCache.clear();
}

bool exportCachedPSOBlob(uint64_t hashLo, uint64_t hashHi, std::vector<uint8_t>& outBlob) {
    outBlob.clear();
    std::filesystem::path dir = getCacheDirectory();
    std::string hex = hashToHex128(hashLo, hashHi);
    std::string key = (dir / hex).string();

    CachedPSOMeta meta{};
    std::vector<uint8_t> objBytes;
    bool found = false;
    {
        JitRuntime& rt = getJitRuntime();
        std::lock_guard<std::mutex> lock(rt.mu);
        auto it = rt.memCache.find(key);
        if (it != rt.memCache.end() && !it->second.objBytes.empty()) {
            meta = it->second.meta;
            objBytes = it->second.objBytes;
            found = true;
        }
    }
    if (!found) {
        std::vector<uint8_t> metaBytes;
        if (!readFileBytes(dir / (hex + ".meta"), metaBytes) || metaBytes.size() != sizeof(CachedPSOMeta)) {
            return false;
        }
        std::memcpy(&meta, metaBytes.data(), sizeof(CachedPSOMeta));
        if (!readFileBytes(dir / (hex + ".o"), objBytes) || objBytes.empty()) {
            return false;
        }
    }

    uint32_t objLen = static_cast<uint32_t>(objBytes.size());
    outBlob.resize(sizeof(CachedPSOMeta) + sizeof(uint32_t) + objLen);
    std::memcpy(outBlob.data(), &meta, sizeof(CachedPSOMeta));
    std::memcpy(outBlob.data() + sizeof(CachedPSOMeta), &objLen, sizeof(uint32_t));
    std::memcpy(outBlob.data() + sizeof(CachedPSOMeta) + sizeof(uint32_t), objBytes.data(), objLen);
    return true;
}

bool importCachedPSOBlob(uint64_t hashLo, uint64_t hashHi, const uint8_t* data, size_t size) {
    if (!data || size < sizeof(CachedPSOMeta) + sizeof(uint32_t)) return false;
    CachedPSOMeta meta{};
    std::memcpy(&meta, data, sizeof(CachedPSOMeta));
    if (meta.magic != CachedPSOMeta{}.magic) return false;
    uint32_t objLen = 0;
    std::memcpy(&objLen, data + sizeof(CachedPSOMeta), sizeof(uint32_t));
    if (sizeof(CachedPSOMeta) + sizeof(uint32_t) + objLen > size || objLen == 0) return false;

    std::filesystem::path dir = getCacheDirectory();
    std::string hex = hashToHex128(hashLo, hashHi);
    uint64_t tId = getJitRuntime().tmpSeq.fetch_add(1, std::memory_order_relaxed);
    const uint8_t* objPtr = data + sizeof(CachedPSOMeta) + sizeof(uint32_t);
    bool ok1 = writeFileAtomic(dir / (hex + ".meta"), &meta, sizeof(CachedPSOMeta), tId);
    bool ok2 = writeFileAtomic(dir / (hex + ".o"), objPtr, objLen, tId);
    return ok1 && ok2;
}

} // namespace nacelle
