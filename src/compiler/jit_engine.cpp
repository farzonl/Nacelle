#include "jit_engine.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <unordered_map>

#define XXH_INLINE_ALL
#include <xxhash.h>

#include <BS_thread_pool.hpp>

#include <llvm/ADT/FloatingPointMode.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/ExecutionEngine/Orc/Debugging/DebuggerSupportPlugin.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/FMF.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Operator.h>
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
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar/EarlyCSE.h>
#include <llvm/Transforms/Scalar/SROA.h>
#include <llvm/Transforms/Scalar/SimplifyCFG.h>

#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h>
#include <mlir/Conversion/MathToLLVM/MathToLLVM.h>
#include <mlir/Conversion/MathToLibm/MathToLibm.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Conversion/UBToLLVM/UBToLLVM.h>
#include <mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Export.h>
#include <mlir/Transforms/Passes.h>

#include "dxil_loader.h"
#include "mlir_pipeline.h"
#include "simt_to_simd_pass.h"
#include "compiler_version.h"
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

const char* getCompilerBuildHash() {
    return kCompilerBuildHash;
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
    uint64_t gsAddr = 0; // gs (graphics)
    uint64_t hsAddr = 0;
    uint64_t hsPcAddr = 0;
    uint64_t dsAddr = 0;
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

struct CachedHostTargetInfo {
    std::string cpu;
    std::vector<std::pair<std::string, bool>> features;
};

const CachedHostTargetInfo& getCachedHostTargetInfo() {
    static const CachedHostTargetInfo info = []() {
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        llvm::InitializeNativeTargetAsmParser();

        CachedHostTargetInfo res;
        llvm::StringRef cpu = llvm::sys::getHostCPUName();
        if (!cpu.empty()) {
            res.cpu = cpu.str();
        }
        auto hostFeatures = llvm::sys::getHostCPUFeatures();
        res.features.reserve(hostFeatures.size());
        for (const auto& kv : hostFeatures) {
            res.features.emplace_back(kv.first().str(), kv.second);
        }
        return res;
    }();
    return info;
}

llvm::Expected<llvm::orc::JITTargetMachineBuilder> createHostJTMB(llvm::CodeGenOptLevel optLevel) {
    const CachedHostTargetInfo& info = getCachedHostTargetInfo();
    auto jtmbOrErr = llvm::orc::JITTargetMachineBuilder::detectHost();
    if (!jtmbOrErr) return jtmbOrErr;
    jtmbOrErr->setCodeGenOptLevel(optLevel);
    if (!info.cpu.empty()) {
        jtmbOrErr->setCPU(info.cpu);
    }
    for (const auto& kv : info.features) {
        jtmbOrErr->getFeatures().AddFeature(kv.first, kv.second);
    }
    return jtmbOrErr;
}

bool emitObjectBytesO3(
    llvm::Module& mod,
    llvm::SmallVectorImpl<char>& outObjBytes,
    std::string& outError);

void ensureLLJITLocked(JitRuntime& rt, std::string& outError) {
    if (rt.jit) return;

    auto jtmbOrErr = createHostJTMB(llvm::CodeGenOptLevel::Less);
    if (!jtmbOrErr) {
        outError = llvm::toString(jtmbOrErr.takeError());
        return;
    }

    auto jitOrErr = llvm::orc::LLJITBuilder()
                        .setJITTargetMachineBuilder(std::move(*jtmbOrErr))
                        .create();
    if (!jitOrErr) {
        outError = llvm::toString(jitOrErr.takeError());
        return;
    }
    rt.jit = std::move(*jitOrErr);
    static std::once_flag atexitOnce;
    std::call_once(atexitOnce, []() {
        llvm::LLVMContext initCtx;
        llvm::Module initMod("__nacelle_warmup", initCtx);
        auto* fnTy = llvm::FunctionType::get(llvm::Type::getVoidTy(initCtx), false);
        auto* fn = llvm::Function::Create(fnTy, llvm::GlobalValue::ExternalLinkage, "__nacelle_warmup", initMod);
        auto* bb = llvm::BasicBlock::Create(initCtx, "entry", fn);
        llvm::ReturnInst::Create(initCtx, bb);
        llvm::SmallVector<char, 0> dummyObj;
        std::string dummyErr;
        emitObjectBytesO3(initMod, dummyObj, dummyErr);
        std::atexit([]() { getJitRuntime().tier2Pool.wait(); });
    });

#if defined(__APPLE__)
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
#endif
}

llvm::Expected<llvm::orc::JITDylib&> createIsolatedDylibLocked(JitRuntime& rt) {
    std::string jdName = "nacelle_pso_jd_" + std::to_string(rt.dylibSeq.fetch_add(1, std::memory_order_relaxed));
    auto jdOrErr = rt.jit->createJITDylib(std::move(jdName));
    if (!jdOrErr) return jdOrErr;
    if (auto procJD = rt.jit->getProcessSymbolsJITDylib()) {
        jdOrErr->addToLinkOrder(*procJD);
    }
    jdOrErr->addToLinkOrder(rt.jit->getMainJITDylib());
    return jdOrErr;
}

bool emitObjectBytesO3(
    llvm::Module& mod,
    llvm::SmallVectorImpl<char>& outObjBytes,
    std::string& outError) {
    auto jtmbOrErr = createHostJTMB(llvm::CodeGenOptLevel::Aggressive);
    if (!jtmbOrErr) {
        outError = llvm::toString(jtmbOrErr.takeError());
        return false;
    }

    const char* strictFpEnv = std::getenv("NACELLE_STRICT_FP");
    const bool strictFp = strictFpEnv && strictFpEnv[0] == '1';
    if (!strictFp) {
#if LLVM_VERSION_MAJOR < 24
        jtmbOrErr->getOptions().AllowFPOpFusion = llvm::FPOpFusion::Fast;
#endif
        llvm::FastMathFlags fmf;
        fmf.setAllowContract(true);
        fmf.setAllowReassoc(true);
        fmf.setNoSignedZeros(true);
        fmf.setAllowReciprocal(true);
        fmf.setApproxFunc(true);
        for (llvm::Function& fn : mod) {
            if (fn.isDeclaration()) continue;
            for (llvm::BasicBlock& bb : fn) {
                for (llvm::Instruction& inst : bb) {
                    if (llvm::isa<llvm::FPMathOperator>(&inst)) {
                        inst.setFastMathFlags(fmf);
                    }
                }
            }
        }
    }

    auto tmOrErr = jtmbOrErr->createTargetMachine();
    if (!tmOrErr) {
        outError = llvm::toString(tmOrErr.takeError());
        return false;
    }
    std::unique_ptr<llvm::TargetMachine> tm = std::move(*tmOrErr);
    mod.setTargetTriple(tm->getTargetTriple());
    mod.setDataLayout(tm->createDataLayout());

    llvm::PipelineTuningOptions pto;
    pto.LoopVectorization = false;
    pto.SLPVectorization = false;
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

    llvm::FunctionPassManager fpm;
    fpm.addPass(llvm::SROAPass(llvm::SROAOptions::ModifyCFG));
    fpm.addPass(llvm::EarlyCSEPass(true));
    fpm.addPass(llvm::InstCombinePass());
    fpm.addPass(llvm::SimplifyCFGPass());

    for (llvm::Function& fn : mod) {
        if (!fn.isDeclaration()) {
            fpm.run(fn, fam);
        }
    }
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
    uint64_t vsAddr, uint64_t psAddr, uint64_t gsAddr,
    uint64_t hsAddr, uint64_t hsPcAddr, uint64_t dsAddr,
    CompiledGraphicsPSO& outPSO) {
    outPSO.simdWidth = meta.simdWidth;
    outPSO.numVSOutputRegs = meta.numVSOutputRegs;
    outPSO.posOutputReg = meta.posOutputReg;
    outPSO.numPSInputRegs = meta.numPSInputRegs;
    outPSO.flatInterpMask = meta.flatInterpMask;
    outPSO.clipDistMask = meta.clipDistMask;
    outPSO.cullDistMask = meta.cullDistMask;
    outPSO.clipAttrOffset = meta.clipAttrOffset;
    outPSO.cullAttrOffset = meta.cullAttrOffset;
    outPSO.hasViewportArrayIndex = meta.hasViewportArrayIndex != 0;
    outPSO.vpIndexAttrOffset = meta.vpIndexAttrOffset;
    outPSO.hasPerPrimShadingRate = meta.hasPerPrimShadingRate != 0;
    outPSO.shadingRateAttrOffset = meta.shadingRateAttrOffset;
    outPSO.totalVSScalarAttrs = meta.totalVSScalarAttrs;
    outPSO.hasGS = meta.hasGS != 0;
    outPSO.gsMaxVertexCount = meta.gsMaxVertexCount;
    outPSO.gsOutputTopology = meta.gsOutputTopology;
    outPSO.gsInstanceCount = meta.gsInstanceCount;
    outPSO.numGSInputScalars = meta.numGSInputScalars;
    outPSO.hasTess = meta.hasTess != 0;
    outPSO.hsInputCPCount = meta.hsInputCPCount;
    outPSO.hsOutputCPCount = meta.hsOutputCPCount;
    outPSO.tessDomain = meta.tessDomain;
    outPSO.tessPartitioning = meta.tessPartitioning;
    outPSO.tessOutputPrimitive = meta.tessOutputPrimitive;
    outPSO.hsMaxTessFactor = meta.hsMaxTessFactor;
    outPSO.numHSInputScalars = meta.numHSInputScalars;
    outPSO.numHSOutputScalars = meta.numHSOutputScalars;
    outPSO.numPatchConstScalars = meta.numPatchConstScalars;
    outPSO.psWritesDepth = meta.psWritesDepth != 0;
    outPSO.psUsesDiscard = meta.psUsesDiscard != 0;
    outPSO.psHasSideEffects = meta.psHasSideEffects != 0;
    outPSO.vsSymbolName = vsSym;
    outPSO.psSymbolName = psSym;
    outPSO.gsSymbolName = meta.hasGS ? (vsSym + "_gs") : std::string{};
    outPSO.vsFn.store(reinterpret_cast<CompiledVertexFn>(vsAddr), std::memory_order_release);
    outPSO.psFn.store(reinterpret_cast<CompiledPixelQuadFn>(psAddr), std::memory_order_release);
    outPSO.gsFn.store(reinterpret_cast<CompiledGeometryFn>(gsAddr), std::memory_order_release);
    outPSO.hsFn.store(reinterpret_cast<CompiledHullCPFn>(hsAddr), std::memory_order_release);
    outPSO.hsPcFn.store(reinterpret_cast<CompiledHullPCFn>(hsPcAddr), std::memory_order_release);
    outPSO.dsFn.store(reinterpret_cast<CompiledDomainFn>(dsAddr), std::memory_order_release);
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
    std::string& outError,
    const std::string& sym2 = {},
    uint64_t* outAddr2 = nullptr,
    const std::string& sym3 = {},
    uint64_t* outAddr3 = nullptr,
    const std::string& sym4 = {},
    uint64_t* outAddr4 = nullptr,
    const std::string& sym5 = {},
    uint64_t* outAddr5 = nullptr) {
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
    if (!sym2.empty() && outAddr2) {
        auto sym2OrErr = rt.jit->lookup(jd, sym2);
        if (!sym2OrErr) {
            outError = llvm::toString(sym2OrErr.takeError());
            return false;
        }
        *outAddr2 = sym2OrErr->getValue();
    } else if (outAddr2) {
        *outAddr2 = 0;
    }
    auto lookupOpt = [&](const std::string& s, uint64_t* out) -> bool {
        if (!s.empty() && out) {
            auto r = rt.jit->lookup(jd, s);
            if (!r) {
                outError = llvm::toString(r.takeError());
                return false;
            }
            *out = r->getValue();
        } else if (out) {
            *out = 0;
        }
        return true;
    };
    if (!lookupOpt(sym3, outAddr3) || !lookupOpt(sym4, outAddr4) || !lookupOpt(sym5, outAddr5)) {
        return false;
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
    feed.addStr(kCompilerBuildHash);
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
    feed.addStr(kCompilerBuildHash);
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
    // Upstream MLIR's ConvertMathToLLVMPass lowers math.isnan and math.isfinite
    // to LLVM::IsFPClass, but omits math.isinf and math.isnormal, which would
    // otherwise be rejected as illegal by ConvertMathToLibmPass.
    mlirMod.walk([](mlir::Operation* op) {
        if (auto isInf = llvm::dyn_cast<mlir::math::IsInfOp>(op)) {
            mlir::OpBuilder b(isInf);
            auto repl = mlir::LLVM::IsFPClass::create(
                b, isInf.getLoc(), isInf.getResult().getType(),
                isInf.getOperand(), static_cast<uint32_t>(llvm::fcInf));
            isInf.replaceAllUsesWith(repl.getRes());
            isInf.erase();
        } else if (auto isNorm = llvm::dyn_cast<mlir::math::IsNormalOp>(op)) {
            mlir::OpBuilder b(isNorm);
            auto repl = mlir::LLVM::IsFPClass::create(
                b, isNorm.getLoc(), isNorm.getResult().getType(),
                isNorm.getOperand(), static_cast<uint32_t>(llvm::fcNormal));
            isNorm.replaceAllUsesWith(repl.getRes());
            isNorm.erase();
        }
    });

    mlir::PassManager pm(mlirMod.getContext());
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

#if defined(_WIN32)
    // Win64 guarantees only 16-byte stack alignment; mark JIT functions nounwind
    // with stackrealign so AVX2/AVX-512 spills and allocas receive 32/64-byte alignment.
    for (llvm::Function& f : *llvmMod) {
        if (!f.isDeclaration()) {
            f.setDoesNotThrow();
            f.addFnAttr("stackrealign");
        }
    }
#endif

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

    rt.memCache[key] = InMemCacheEntry{meta, addr0, 0, 0, 0, 0, 0, std::move(objBytes)};
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
            applyGraphicsMeta(it->second.meta, vsSym, psSym, it->second.addr0, it->second.addr1, it->second.gsAddr,
                              it->second.hsAddr, it->second.hsPcAddr, it->second.dsAddr, outPSO);
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
        applyGraphicsMeta(it->second.meta, vsSym, psSym, it->second.addr0, it->second.addr1, it->second.gsAddr,
                          it->second.hsAddr, it->second.hsPcAddr, it->second.dsAddr, outPSO);
        return true;
    }

    uint64_t addr0 = 0, addr1 = 0, gsAddr = 0, hsAddr = 0, hsPcAddr = 0, dsAddr = 0;
    std::string gsSym = meta.hasGS ? (vsSym + "_gs") : std::string{};
    std::string hsSym = meta.hasTess ? (vsSym + "_hs") : std::string{};
    std::string hsPcSym = meta.hasTess ? (vsSym + "_hspc") : std::string{};
    std::string dsSym = meta.hasTess ? (vsSym + "_ds") : std::string{};
    std::string err;
    if (!loadObjectIntoJitLocked(rt, objBytes.data(), objBytes.size(), key + ".o", vsSym, psSym, addr0, addr1, err,
                                 gsSym, &gsAddr, hsSym, &hsAddr, hsPcSym, &hsPcAddr, dsSym, &dsAddr)) {
        return false;
    }

    rt.memCache[key] = InMemCacheEntry{meta, addr0, addr1, gsAddr, hsAddr, hsPcAddr, dsAddr, std::move(objBytes)};
    applyGraphicsMeta(meta, vsSym, psSym, addr0, addr1, gsAddr, hsAddr, hsPcAddr, dsAddr, outPSO);
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
    const char* noT2 = std::getenv("NACELLE_DISABLE_TIER2");
    const bool scheduleTier2 = !(noT2 && noT2[0] == '1');

    // Snapshot unoptimized IR bitcode before Tier-1 mutates `llvmMod`.
    llvm::SmallString<0> bcBytes;
    if (scheduleTier2) {
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

    if (!scheduleTier2) {
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
                meta, addr0, 0, 0, 0, 0, 0,
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

    uint64_t vsAddr = 0, psAddr = 0, gsAddr = 0, hsAddr = 0, hsPcAddr = 0, dsAddr = 0;
    std::string gsSym = meta.hasGS ? (vsSym + "_gs") : std::string{};
    std::string hsSym = meta.hasTess ? (vsSym + "_hs") : std::string{};
    std::string hsPcSym = meta.hasTess ? (vsSym + "_hspc") : std::string{};
    std::string dsSym = meta.hasTess ? (vsSym + "_ds") : std::string{};
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
        auto lookupSym = [&](const std::string& s, uint64_t& outAddr) -> bool {
            auto r = rt.jit->lookup(jd, s);
            if (!r) {
                outError = llvm::toString(r.takeError());
                return false;
            }
            outAddr = r->getValue();
            return true;
        };
        if (!lookupSym(vsSym, vsAddr) || !lookupSym(psSym, psAddr)) return false;
        if (!gsSym.empty() && !lookupSym(gsSym, gsAddr)) return false;
        if (!hsSym.empty() && (!lookupSym(hsSym, hsAddr) || !lookupSym(hsPcSym, hsPcAddr) || !lookupSym(dsSym, dsAddr))) {
            return false;
        }
    }

    outPSO.vsSymbolName = vsSym;
    outPSO.psSymbolName = psSym;
    outPSO.gsSymbolName = gsSym;
    outPSO.vsFn.store(reinterpret_cast<CompiledVertexFn>(vsAddr), std::memory_order_release);
    outPSO.psFn.store(reinterpret_cast<CompiledPixelQuadFn>(psAddr), std::memory_order_release);
    outPSO.gsFn.store(reinterpret_cast<CompiledGeometryFn>(gsAddr), std::memory_order_release);
    outPSO.hsFn.store(reinterpret_cast<CompiledHullCPFn>(hsAddr), std::memory_order_release);
    outPSO.hsPcFn.store(reinterpret_cast<CompiledHullPCFn>(hsPcAddr), std::memory_order_release);
    outPSO.dsFn.store(reinterpret_cast<CompiledDomainFn>(dsAddr), std::memory_order_release);

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

        uint64_t addr0 = 0, addr1 = 0, gsAddr2 = 0, hsAddr2 = 0, hsPcAddr2 = 0, dsAddr2 = 0;
        std::string gsSym2 = meta.hasGS ? (vsSym + "_gs") : std::string{};
        std::string hsSym2 = meta.hasTess ? (vsSym + "_hs") : std::string{};
        std::string hsPcSym2 = meta.hasTess ? (vsSym + "_hspc") : std::string{};
        std::string dsSym2 = meta.hasTess ? (vsSym + "_ds") : std::string{};
        {
            std::lock_guard<std::mutex> lock(rt.mu);
            std::string key = (cacheDir / hex).string();
            if (!loadObjectIntoJitLocked(rt, reinterpret_cast<const uint8_t*>(objBytes.data()),
                                         objBytes.size(), key + ".o", vsSym, psSym, addr0, addr1, err,
                                         gsSym2, &gsAddr2, hsSym2, &hsAddr2, hsPcSym2, &hsPcAddr2, dsSym2, &dsAddr2)) {
                return;
            }
            rt.memCache[key] = InMemCacheEntry{
                meta, addr0, addr1, gsAddr2, hsAddr2, hsPcAddr2, dsAddr2,
                std::vector<uint8_t>(reinterpret_cast<const uint8_t*>(objBytes.begin()),
                                     reinterpret_cast<const uint8_t*>(objBytes.end()))};
        }

        std::lock_guard<std::mutex> psoLock(box->mu);
        if (box->gs) {
            box->gs->vsFn.store(reinterpret_cast<CompiledVertexFn>(addr0), std::memory_order_release);
            box->gs->psFn.store(reinterpret_cast<CompiledPixelQuadFn>(addr1), std::memory_order_release);
            box->gs->gsFn.store(reinterpret_cast<CompiledGeometryFn>(gsAddr2), std::memory_order_release);
            box->gs->hsFn.store(reinterpret_cast<CompiledHullCPFn>(hsAddr2), std::memory_order_release);
            box->gs->hsPcFn.store(reinterpret_cast<CompiledHullPCFn>(hsPcAddr2), std::memory_order_release);
            box->gs->dsFn.store(reinterpret_cast<CompiledDomainFn>(dsAddr2), std::memory_order_release);
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

    const char* profEnv = std::getenv("NACELLE_PROFILE_COMPILE");
    const bool profileCompile = profEnv && profEnv[0] == '1';
    using Clock = std::chrono::steady_clock;
    auto msSince = [](Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    const auto tStart = Clock::now();

    auto llvmCtx = std::make_unique<llvm::LLVMContext>();
    DxilLoadedShader shader;
    if (!loadAndUpgradeDXIL(desc.CS, *llvmCtx, shader, outError)) {
        return false;
    }
    const auto tDxil = Clock::now();

    const RootSignatureLayout& effectiveLayout =
        (rootLayout.params.empty() && rootLayout.staticSamplers.empty() && shader.hasEmbeddedRootSig)
            ? shader.embeddedRootSig
            : rootLayout;

    thread_local mlir::MLIRContext mlirCtx;
    auto mlirMod = importAndLiftDXILToSCF(std::move(shader.module), shader.entryName, mlirCtx, outError);
    if (!mlirMod) return false;
    const auto tLift = Clock::now();

    const uint32_t effectiveWidth = (shader.waveSize > 0) ? shader.waveSize : simdWidth;
    if (!packetizeComputeEntry(
            *mlirMod, shader, effectiveLayout, effectiveWidth, symbolName,
            outError)) {
        return false;
    }
    const auto tPacketize = Clock::now();

    auto outLlvmCtx = std::make_unique<llvm::LLVMContext>();
    auto linkedMod = translateMLIRToLinkedLLVMIR(*mlirMod, *outLlvmCtx, outError);
    if (!linkedMod) return false;
    const auto tTranslate = Clock::now();

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

    bool ok = jitCompileComputeTier1AndScheduleTier2(
        std::move(linkedMod), std::move(outLlvmCtx), symbolName, meta, outPSO, outError);
    const auto tTier1 = Clock::now();

    if (profileCompile) {
        std::fprintf(
            stderr,
            "[NacelleCompile] W=%u total=%.2fms (dxil=%.2fms lift=%.2fms packetize=%.2fms mlir2llvm=%.2fms tier1=%.2fms)\n",
            effectiveWidth,
            msSince(tStart, tTier1),
            msSince(tStart, tDxil),
            msSince(tDxil, tLift),
            msSince(tLift, tPacketize),
            msSince(tPacketize, tTranslate),
            msSince(tTranslate, tTier1));
    }
    return ok;
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
