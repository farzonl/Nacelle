#include "jit_engine.h"
#include "../rasterizer/raytracing_bvh.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <thread>
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
#include <llvm/IR/IRBuilder.h>
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
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Export.h>
#include <mlir/Transforms/Passes.h>

#include "dxil_loader.h"
#include "mlir_pipeline.h"
#include "simt_to_simd_pass.h"
#include "compiler_version.h"
#include "math_simd_bc.h"
#include "rayquery_rt_bc.h"
#include "texture_sampler_bc.h"

#if defined(__APPLE__)
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <dlfcn.h>
#elif defined(NACELLE_HAS_CBLAS)
#include NACELLE_CBLAS_HEADER
#endif

namespace nacelle {

enum class GemmBackend : uint8_t {
    Accelerate,
    OpenBLAS,
    Fallback,
};

static GemmBackend defaultGemmBackend() {
#if defined(__APPLE__)
    return GemmBackend::Accelerate;
#elif defined(NACELLE_HAS_CBLAS)
    return GemmBackend::OpenBLAS;
#else
    return GemmBackend::Fallback;
#endif
}

static GemmBackend getConfiguredGemmBackend() {
    if (const char* env = std::getenv("NACELLE_GEMM_BACKEND")) {
        llvm::StringRef v(env);
        if (v.equals_insensitive("fallback")) return GemmBackend::Fallback;
        if (v.equals_insensitive("openblas")) return GemmBackend::OpenBLAS;
        if (v.equals_insensitive("accelerate") || v.equals_insensitive("amx")) return GemmBackend::Accelerate;
    }
    return defaultGemmBackend();
}

#if defined(__APPLE__)
struct AppleOpenBlasDispatch {
    using SgemmFn = void (*)(
        int, int, int, int, int, int, float, const float*, int, const float*, int, float, float*, int);
    using DgemmFn = void (*)(
        int, int, int, int, int, int, double, const double*, int, const double*, int, double, double*, int);
    void* handle = nullptr;
    SgemmFn sgemm = nullptr;
    DgemmFn dgemm = nullptr;
    bool loaded = false;
};

static const AppleOpenBlasDispatch& getAppleOpenBlasDispatch() {
    static AppleOpenBlasDispatch d = []() {
        AppleOpenBlasDispatch out;
        const char* path = std::getenv("NACELLE_OPENBLAS_LIB");
        if (!path || !path[0]) {
#ifdef NACELLE_OPENBLAS_DYLIB_PATH
            path = NACELLE_OPENBLAS_DYLIB_PATH;
#else
            path = "libopenblas.dylib";
#endif
        }
        if (void* h = dlopen(path, RTLD_LAZY | RTLD_LOCAL)) {
            auto s = reinterpret_cast<AppleOpenBlasDispatch::SgemmFn>(dlsym(h, "cblas_sgemm"));
            auto dg = reinterpret_cast<AppleOpenBlasDispatch::DgemmFn>(dlsym(h, "cblas_dgemm"));
            if (s && dg) {
                out.handle = h;
                out.sgemm = s;
                out.dgemm = dg;
                out.loaded = true;
                return out;
            }
            dlclose(h);
        }
        std::fprintf(
            stderr,
            "[Nacelle] NACELLE_GEMM_BACKEND=openblas requested, but %s could not be loaded; falling back to fallbackGemm\n",
            path);
        return out;
    }();
    return d;
}
#endif

static BS::thread_pool<>& getFallbackGemmPool() {
    static BS::thread_pool<> pool;
    return pool;
}

template <typename T>
static void fallbackGemm(
    const T* A,
    const T* B,
    const T* C,
    T* Out,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t hasC) {
    if (M == 0 || N == 0) return;

    auto runRows = [&](uint32_t iBegin, uint32_t iEnd) {
        const size_t rowBytes = static_cast<size_t>(N) * sizeof(T);
        for (uint32_t i = iBegin; i < iEnd; ++i) {
            T* outRow = Out + static_cast<size_t>(i) * N;
            if (hasC && C) {
                std::memcpy(outRow, C + static_cast<size_t>(i) * N, rowBytes);
            } else {
                std::memset(outRow, 0, rowBytes);
            }
            const T* aRow = A + static_cast<size_t>(i) * K;
            constexpr uint32_t kTileJ = 2048;
            for (uint32_t j0 = 0; j0 < N; j0 += kTileJ) {
                const uint32_t jEnd = std::min(N, j0 + kTileJ);
                for (uint32_t k = 0; k < K; ++k) {
                    const T aVal = aRow[k];
                    const T* bRow = B + static_cast<size_t>(k) * N;
                    for (uint32_t j = j0; j < jEnd; ++j) {
                        outRow[j] += aVal * bRow[j];
                    }
                }
            }
        }
    };

    const uint64_t ops = static_cast<uint64_t>(M) * N * K;
    if (ops >= 65536ull && M >= 16) {
        auto& pool = getFallbackGemmPool();
        if (pool.get_thread_count() > 1) {
            pool.submit_blocks<uint32_t>(0, M, runRows).wait();
            return;
        }
    }
    runRows(0, M);
}

template <typename T>
static void dispatchGemm(
    const T* A,
    const T* B,
    const T* C,
    T* Out,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t hasC) {
    auto copyC = [&]() {
        if (hasC && C) {
            std::memcpy(Out, C, static_cast<size_t>(M) * N * sizeof(T));
        }
    };
    const T beta = (hasC && C) ? T(1) : T(0);

    switch (getConfiguredGemmBackend()) {
    case GemmBackend::Accelerate:
#if defined(__APPLE__)
        copyC();
        if constexpr (std::is_same_v<T, float>) {
            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                1.0f, A, static_cast<int>(K), B, static_cast<int>(N), beta, Out, static_cast<int>(N));
        } else {
            cblas_dgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                1.0, A, static_cast<int>(K), B, static_cast<int>(N), beta, Out, static_cast<int>(N));
        }
        return;
#else
        break;
#endif
    case GemmBackend::OpenBLAS:
#if defined(__APPLE__)
        if (const auto& ob = getAppleOpenBlasDispatch(); ob.loaded) {
            copyC();
            if constexpr (std::is_same_v<T, float>) {
                ob.sgemm(
                    101 /*CblasRowMajor*/, 111 /*CblasNoTrans*/, 111 /*CblasNoTrans*/,
                    static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                    1.0f, A, static_cast<int>(K), B, static_cast<int>(N), beta, Out, static_cast<int>(N));
            } else {
                ob.dgemm(
                    101 /*CblasRowMajor*/, 111 /*CblasNoTrans*/, 111 /*CblasNoTrans*/,
                    static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                    1.0, A, static_cast<int>(K), B, static_cast<int>(N), beta, Out, static_cast<int>(N));
            }
            return;
        }
        break;
#elif defined(NACELLE_HAS_CBLAS)
        copyC();
        if constexpr (std::is_same_v<T, float>) {
            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                1.0f, A, static_cast<int>(K), B, static_cast<int>(N), beta, Out, static_cast<int>(N));
        } else {
            cblas_dgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                1.0, A, static_cast<int>(K), B, static_cast<int>(N), beta, Out, static_cast<int>(N));
        }
        return;
#else
        break;
#endif
    case GemmBackend::Fallback:
        break;
    }

    fallbackGemm<T>(A, B, C, Out, M, N, K, hasC);
}

extern "C" void nacelle_accel_sgemm(
    const float* A,
    const float* B,
    const float* C,
    float* Out,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t hasC) {
    dispatchGemm<float>(A, B, C, Out, M, N, K, hasC);
}

extern "C" void nacelle_accel_dgemm(
    const double* A,
    const double* B,
    const double* C,
    double* Out,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t hasC) {
    dispatchGemm<double>(A, B, C, Out, M, N, K, hasC);
}

bool linkRuntimeBitcodeAndInline(llvm::Module& mod, std::string& outError);

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

struct InMemVSCacheEntry {
    CachedVSMeta meta{};
    uint64_t vsAddr = 0;
    bool tier2Done = false;
    std::vector<uint8_t> objBytes;
    std::vector<ShaderSigElement> vsInputs;
    std::vector<ShaderSigElement> vsOutputs;
    std::vector<DxilSigRecord> vsOutputRecs;
    std::vector<std::shared_ptr<PsoUpdateBox>> waiters;
};

struct InMemCacheEntry {
    CachedPSOMeta meta{};
    uint64_t addr0 = 0; // cs or vs
    uint64_t addr1 = 0; // ps (graphics)
    uint64_t gsAddr = 0; // gs (graphics)
    uint64_t hsAddr = 0;
    uint64_t hsPcAddr = 0;
    uint64_t dsAddr = 0;
    uint64_t msAddr = 0;
    uint64_t asAddr = 0;
    std::vector<uint8_t> objBytes;
};

static std::atomic<uint64_t> g_vertexStageCompileCount{0};

struct JitRuntime {
    std::mutex mu;
    std::unique_ptr<llvm::orc::LLJIT> jit;
    std::unordered_map<std::string, InMemCacheEntry> memCache;
    std::unordered_map<std::string, InMemVSCacheEntry> vsMemCache;
    std::atomic<uint64_t> dylibSeq{1};
    std::atomic<uint64_t> tmpSeq{1};
    bool mathSimdLoaded = false;
    bool rayQueryRtLoaded = false;
    bool texSamplerLoaded = false;
    std::unique_ptr<BS::thread_pool<>> tier2PoolOwned;
    std::atomic<BS::thread_pool<>*> activeTier2Pool{nullptr};

    ~JitRuntime() {
        if (tier2PoolOwned) {
            tier2PoolOwned->wait();
        }
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

BS::thread_pool<>& ensureTier2PoolReady(JitRuntime& rt) {
    static std::once_flag tier2Once;
    std::call_once(tier2Once, [&rt]() {
        unsigned hw = std::thread::hardware_concurrency();
        unsigned poolThreads = std::clamp(hw / 2u, 2u, 8u);
        rt.tier2PoolOwned = std::make_unique<BS::thread_pool<>>(poolThreads);
        rt.activeTier2Pool.store(rt.tier2PoolOwned.get(), std::memory_order_release);

        llvm::LLVMContext initCtx;
        llvm::Module initMod("__nacelle_warmup", initCtx);
        auto* fnTy = llvm::FunctionType::get(llvm::Type::getVoidTy(initCtx), false);
        auto* fn = llvm::Function::Create(fnTy, llvm::GlobalValue::ExternalLinkage, "__nacelle_warmup", initMod);
        auto* bb = llvm::BasicBlock::Create(initCtx, "entry", fn);
        llvm::ReturnInst::Create(initCtx, bb);
        llvm::SmallVector<char, 0> dummyObj;
        std::string dummyErr;
        emitObjectBytesO3(initMod, dummyObj, dummyErr);
        std::atexit([]() {
            if (auto* pool = getJitRuntime().activeTier2Pool.load(std::memory_order_acquire)) {
                pool->wait();
            }
        });
    });
    return *rt.activeTier2Pool.load(std::memory_order_acquire);
}

bool isStrictFloatingPointEnabled() {
    const char* strictFpEnv = std::getenv("NACELLE_STRICT_FP");
    return strictFpEnv && strictFpEnv[0] == '1';
}

void applyShaderFastMathFlags(llvm::Module& mod) {
    if (isStrictFloatingPointEnabled()) return;
    llvm::FastMathFlags fmf;
    fmf.setAllowContract(true);
    fmf.setAllowReassoc(true);
    fmf.setNoSignedZeros(true);
    fmf.setAllowReciprocal(true);
    fmf.setApproxFunc(true);
    for (llvm::Function& fn : mod) {
        if (fn.isDeclaration()) continue;
        // Disable hardware reciprocal/rsqrt estimate + Newton-Raphson expansion
        // (e.g. x86 vrsqrtps + NR), which lacks zero/infinity masking for rsqrt
        // and evaluates 0.0 * inf -> -NaN on ±0.0 / flushed-denorm / +inf inputs
        // instead of D3D12-required ±inf / +0.0. Keep AllowReciprocal on instructions
        // so constant divisors still fold to multiplies.
        fn.addFnAttr("reciprocal-estimates", "none");
        for (llvm::BasicBlock& bb : fn) {
            for (llvm::Instruction& inst : bb) {
                if (llvm::isa<llvm::FPMathOperator>(&inst)) {
                    inst.setFastMathFlags(fmf);
                }
            }
        }
    }
}

void ensureLLJITLocked(JitRuntime& rt, std::string& outError) {
    if (rt.jit) return;

#if defined(__APPLE__)
    // macOS secondary threads (std::thread) default to a 512 KB stack; RegAllocFast at
    // CodeGenOptLevel::None does not reuse stack slots and can overflow 512 KB on large W=16 shaders.
    auto jtmbOrErr = createHostJTMB(llvm::CodeGenOptLevel::Less);
#else
    auto jtmbOrErr = createHostJTMB(llvm::CodeGenOptLevel::None);
#endif
    if (!jtmbOrErr) {
        outError = llvm::toString(jtmbOrErr.takeError());
        return;
    }
#if LLVM_VERSION_MAJOR < 24
    if (!isStrictFloatingPointEnabled()) {
        jtmbOrErr->getOptions().AllowFPOpFusion = llvm::FPOpFusion::Fast;
    }
#endif

    auto jitOrErr = llvm::orc::LLJITBuilder()
                        .setJITTargetMachineBuilder(std::move(*jtmbOrErr))
                        .create();
    if (!jitOrErr) {
        outError = llvm::toString(jitOrErr.takeError());
        return;
    }
    rt.jit = std::move(*jitOrErr);

    {
        auto& es = rt.jit->getExecutionSession();
        llvm::orc::SymbolMap syms;
        auto addSym = [&](const char* name, auto* fn) {
            syms[es.intern(name)] = llvm::orc::ExecutorSymbolDef(
                llvm::orc::ExecutorAddr::fromPtr(fn), llvm::JITSymbolFlags::Exported);
        };
        addSym("nacelle_rayquery_alloc", &nacelle_rayquery_alloc);
        addSym("nacelle_rayquery_trace_inline", &nacelle_rayquery_trace_inline);
        addSym("nacelle_rayquery_proceed", &nacelle_rayquery_proceed);
        addSym("nacelle_rayquery_abort", &nacelle_rayquery_abort);
        addSym("nacelle_rayquery_commit_triangle", &nacelle_rayquery_commit_triangle);
        addSym("nacelle_rayquery_commit_procedural", &nacelle_rayquery_commit_procedural);
        addSym("nacelle_rayquery_state_i32", &nacelle_rayquery_state_i32);
        addSym("nacelle_rayquery_state_f32", &nacelle_rayquery_state_f32);
        addSym("nacelle_rayquery_state_vec", &nacelle_rayquery_state_vec);
        addSym("nacelle_rayquery_state_mat", &nacelle_rayquery_state_mat);
        addSym("nacelle_rayquery_alloc_pkt", &nacelle_rayquery_alloc_pkt);
        addSym("nacelle_rayquery_trace_inline_pkt", &nacelle_rayquery_trace_inline_pkt);
        addSym("nacelle_rayquery_proceed_pkt", &nacelle_rayquery_proceed_pkt);
        addSym("nacelle_rayquery_state_i32_pkt", &nacelle_rayquery_state_i32_pkt);
        addSym("nacelle_rayquery_state_f32_pkt", &nacelle_rayquery_state_f32_pkt);
        addSym("nacelle_rayquery_state_vec_pkt", &nacelle_rayquery_state_vec_pkt);
        addSym("nacelle_dxr_trace_ray", &nacelle_dxr_trace_ray);
        addSym("nacelle_dxr_trace_ray_packet", &nacelle_dxr_trace_ray_packet);
        addSym("nacelle_dxr_state_i32", &nacelle_dxr_state_i32);
        addSym("nacelle_dxr_state_f32", &nacelle_dxr_state_f32);
        addSym("nacelle_dxr_state_vec", &nacelle_dxr_state_vec);
        addSym("nacelle_dxr_state_mat", &nacelle_dxr_state_mat);
        addSym("nacelle_dxr_ignore_hit", &nacelle_dxr_ignore_hit);
        addSym("nacelle_dxr_accept_hit_and_end_search", &nacelle_dxr_accept_hit_and_end_search);
        addSym("nacelle_dxr_report_hit", &nacelle_dxr_report_hit);
        addSym("nacelle_dxr_call_shader", &nacelle_dxr_call_shader);
        addSym("nacelle_accel_sgemm", &nacelle_accel_sgemm);
        addSym("nacelle_accel_dgemm", &nacelle_accel_dgemm);
        if (auto err = rt.jit->getMainJITDylib().define(llvm::orc::absoluteSymbols(std::move(syms)))) {
            outError = llvm::toString(std::move(err));
            rt.jit.reset();
            return;
        }
    }

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
    if (!linkRuntimeBitcodeAndInline(mod, outError)) {
        return false;
    }
    auto jtmbOrErr = createHostJTMB(llvm::CodeGenOptLevel::Aggressive);
    if (!jtmbOrErr) {
        outError = llvm::toString(jtmbOrErr.takeError());
        return false;
    }

#if LLVM_VERSION_MAJOR < 24
    if (!isStrictFloatingPointEnabled()) {
        jtmbOrErr->getOptions().AllowFPOpFusion = llvm::FPOpFusion::Fast;
    }
#endif

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

    if (const char* dumpEnv = std::getenv("NACELLE_DUMP_IR"); dumpEnv && dumpEnv[0] == '1') {
        static std::atomic<uint64_t> postSeq{0};
        uint64_t seq = postSeq.fetch_add(1, std::memory_order_relaxed);
        std::string path = "/tmp/nacelle_ir_" + std::to_string(seq) + "_post_o3.ll";
        std::error_code ec;
        llvm::raw_fd_ostream os(path, ec);
        if (!ec) {
            mod.print(os, nullptr);
            std::fprintf(stderr, "[nacelle] dumped post-O3 LLVM IR: %s\n", path.c_str());
        }
    }

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
    uint64_t msAddr, uint64_t asAddr,
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
    outPSO.hasMS = meta.hasMS != 0;
    outPSO.hasAS = meta.hasAS != 0;
    for (int k = 0; k < 3; ++k) {
        outPSO.msThreadGroupSize[k] = meta.msThreadGroupSize[k];
        outPSO.asThreadGroupSize[k] = meta.asThreadGroupSize[k];
    }
    outPSO.msGroupSharedBytes = meta.msGroupSharedBytes;
    outPSO.msMaxVertexCount = meta.msMaxVertexCount;
    outPSO.msMaxPrimitiveCount = meta.msMaxPrimitiveCount;
    outPSO.msOutputTopology = meta.msOutputTopology;
    outPSO.msPayloadBytes = meta.msPayloadBytes;
    outPSO.msPrimAttrMask = meta.msPrimAttrMask;
    outPSO.asGroupSharedBytes = meta.asGroupSharedBytes;
    outPSO.asPayloadBytes = meta.asPayloadBytes;
    outPSO.psInputFromVSSlot = meta.psInputFromVSSlot;
    outPSO.psUsesPerspBarycentrics = meta.psUsesPerspBarycentrics != 0;
    outPSO.psWritesDepth = meta.psWritesDepth != 0;
    outPSO.psUsesDiscard = meta.psUsesDiscard != 0;
    outPSO.psHasSideEffects = meta.psHasSideEffects != 0;
    outPSO.vsCacheHashLo = meta.vsHashLo;
    outPSO.vsCacheHashHi = meta.vsHashHi;
    outPSO.vsSymbolName = vsSym;
    outPSO.psSymbolName = psSym;
    outPSO.gsSymbolName = meta.hasGS ? (vsSym + "_gs") : std::string{};
    outPSO.vsFn.store(reinterpret_cast<CompiledVertexFn>(vsAddr), std::memory_order_release);
    outPSO.psFn.store(reinterpret_cast<CompiledPixelQuadFn>(psAddr), std::memory_order_release);
    outPSO.gsFn.store(reinterpret_cast<CompiledGeometryFn>(gsAddr), std::memory_order_release);
    outPSO.hsFn.store(reinterpret_cast<CompiledHullCPFn>(hsAddr), std::memory_order_release);
    outPSO.hsPcFn.store(reinterpret_cast<CompiledHullPCFn>(hsPcAddr), std::memory_order_release);
    outPSO.dsFn.store(reinterpret_cast<CompiledDomainFn>(dsAddr), std::memory_order_release);
    outPSO.msFn.store(reinterpret_cast<CompiledMeshFn>(msAddr), std::memory_order_release);
    outPSO.asFn.store(reinterpret_cast<CompiledAmplificationFn>(asAddr), std::memory_order_release);
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
    uint64_t& outHi,
    const MeshShaderStages& meshStages) {
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
    addShader(meshStages.AS);
    addShader(meshStages.MS);

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

void computeVertexStageHash(
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

    uint64_t vsLen = desc.VS.BytecodeLength;
    feed.add(vsLen);
    feed.addBytes(desc.VS.pShaderBytecode, static_cast<size_t>(vsLen));

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

bool linkRuntimeBitcodeAndInline(llvm::Module& mod, std::string& outError) {
    llvm::LLVMContext& llvmCtx = mod.getContext();
    auto linkRuntimeBitcode = [&](const unsigned char* bcData, size_t bcSize,
                                  llvm::StringRef modName, llvm::StringRef fnPrefix) -> bool {
        bool needed = false;
        for (const llvm::Function& f : mod) {
            if (f.isDeclaration() && f.getName().starts_with(fnPrefix)) {
                needed = true;
                break;
            }
        }
        if (!needed) return true;

        auto bcBuf = llvm::MemoryBuffer::getMemBuffer(
            llvm::StringRef(reinterpret_cast<const char*>(bcData), bcSize), modName, false);
        auto bcModOrErr = llvm::parseBitcodeFile(bcBuf->getMemBufferRef(), llvmCtx);
        if (!bcModOrErr) {
            outError = "Failed to parse embedded " + modName.str() + ": " + llvm::toString(bcModOrErr.takeError());
            return false;
        }
        std::unique_ptr<llvm::Module> bcMod = std::move(*bcModOrErr);
        bcMod->setTargetTriple(mod.getTargetTriple());
        bcMod->setDataLayout(mod.getDataLayout());

        if (llvm::Linker::linkModules(mod, std::move(bcMod), llvm::Linker::Flags::LinkOnlyNeeded)) {
            outError = "llvm::Linker failed linking " + modName.str();
            return false;
        }

        uint32_t mathCallCount = 0;
        if (fnPrefix == "nacelle_math_") {
            for (const llvm::Function& fn : mod) {
                for (const llvm::BasicBlock& bb : fn) {
                    for (const llvm::Instruction& inst : bb) {
                        if (const auto* cb = llvm::dyn_cast<llvm::CallBase>(&inst)) {
                            if (const llvm::Function* callee = cb->getCalledFunction();
                                callee && callee->getName().starts_with("nacelle_math_")) {
                                ++mathCallCount;
                            }
                        }
                    }
                }
            }
        }
        const bool forceAlwaysInline = (fnPrefix != "nacelle_math_") || (mathCallCount <= 32);

        for (llvm::Function& f : mod) {
            if (!f.isDeclaration() && f.getName().starts_with(fnPrefix)) {
                f.setLinkage(llvm::GlobalValue::InternalLinkage);
                f.removeFnAttr("target-cpu");
                f.removeFnAttr("target-features");
                f.removeFnAttr("tune-cpu");
                if (!isStrictFloatingPointEnabled()) {
                    f.addFnAttr("reciprocal-estimates", "none");
                }
                if (!f.hasFnAttribute(llvm::Attribute::NoInline)) {
                    if (forceAlwaysInline) {
                        f.addFnAttr(llvm::Attribute::AlwaysInline);
                    } else {
                        f.addFnAttr(llvm::Attribute::InlineHint);
                    }
                }
            }
        }
        return true;
    };

    if (!linkRuntimeBitcode(kTextureSamplerBitcode, kTextureSamplerBitcode_size, "texture_sampler.bc", "nacelle_tex_")) {
        return false;
    }
    if (!linkRuntimeBitcode(kMathSimdBitcode, kMathSimdBitcode_size, "math_simd.bc", "nacelle_math_")) {
        return false;
    }
    if (!linkRuntimeBitcode(kRayQueryRtBitcode, kRayQueryRtBitcode_size, "rayquery_rt.bc", "nacelle_rqbc_")) {
        return false;
    }

    bool needsAlwaysInline = false;
    for (const llvm::Function& f : mod) {
        if (!f.isDeclaration() && f.hasFnAttribute(llvm::Attribute::AlwaysInline)) {
            needsAlwaysInline = true;
            break;
        }
    }
    if (needsAlwaysInline) {
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

        llvm::ModulePassManager inlinerPM;
        inlinerPM.addPass(llvm::AlwaysInlinerPass());
        inlinerPM.run(mod, mam);
    }

    // For non-inlined internal nacelle_math_v<W>_* functions, promote the `out`
    // pointer parameter to a `fastcc` vector return value so O3 SROA eliminates
    // caller/callee stack allocas and stores.
    {
        llvm::SmallVector<llvm::Function*, 16> mathFns;
        for (llvm::Function& f : mod) {
            if (!f.isDeclaration() && f.hasInternalLinkage() &&
                f.getName().starts_with("nacelle_math_v") &&
                f.getReturnType()->isVoidTy() && f.arg_size() == 2 &&
                !f.use_empty()) {
                mathFns.push_back(&f);
            }
        }
        for (llvm::Function* oldFn : mathFns) {
            llvm::Type* vecTy = nullptr;
            bool valid = true;
            llvm::SmallVector<llvm::CallInst*, 32> calls;
            for (llvm::User* u : oldFn->users()) {
                auto* ci = llvm::dyn_cast<llvm::CallInst>(u);
                if (!ci || ci->getCalledFunction() != oldFn) {
                    valid = false;
                    break;
                }
                auto* outAlloca = llvm::dyn_cast<llvm::AllocaInst>(ci->getArgOperand(0));
                if (!outAlloca || !outAlloca->getAllocatedType()->isVectorTy()) {
                    valid = false;
                    break;
                }
                if (!vecTy) {
                    vecTy = outAlloca->getAllocatedType();
                } else if (vecTy != outAlloca->getAllocatedType()) {
                    valid = false;
                    break;
                }
                calls.push_back(ci);
            }
            if (!valid || !vecTy || calls.empty()) continue;

            std::string fnName = oldFn->getName().str();
            oldFn->setName(fnName + ".ptr_out");

            llvm::FunctionType* newTy = llvm::FunctionType::get(vecTy, {oldFn->getArg(1)->getType()}, false);
            llvm::Function* newFn = llvm::Function::Create(
                newTy, llvm::GlobalValue::InternalLinkage, fnName, &mod);
            newFn->setCallingConv(llvm::CallingConv::Fast);
            newFn->copyAttributesFrom(oldFn);
            // Clear parameter-specific attributes from the removed arg0 and remap arg1 -> arg0.
            newFn->setAttributes(llvm::AttributeList::get(
                llvmCtx,
                oldFn->getAttributes().getFnAttrs(),
                llvm::AttributeSet(),
                {oldFn->getAttributes().getParamAttrs(1)}));

            newFn->splice(newFn->begin(), oldFn);
            llvm::IRBuilder<> eb(&newFn->getEntryBlock().front());
            llvm::AllocaInst* localOut = eb.CreateAlloca(vecTy, nullptr, "math.ret.slot");
            oldFn->getArg(0)->replaceAllUsesWith(localOut);
            oldFn->getArg(1)->replaceAllUsesWith(newFn->getArg(0));

            llvm::SmallVector<llvm::ReturnInst*, 4> rets;
            for (llvm::BasicBlock& bb : *newFn) {
                if (auto* ri = llvm::dyn_cast<llvm::ReturnInst>(bb.getTerminator())) {
                    rets.push_back(ri);
                }
            }
            for (llvm::ReturnInst* ri : rets) {
                llvm::IRBuilder<> rb(ri);
                llvm::Value* retVal = rb.CreateLoad(vecTy, localOut, "math.ret");
                rb.CreateRet(retVal);
                ri->eraseFromParent();
            }

            for (llvm::CallInst* ci : calls) {
                llvm::IRBuilder<> cb(ci);
                llvm::CallInst* newCall = cb.CreateCall(newFn, {ci->getArgOperand(1)}, "math.call");
                newCall->setCallingConv(llvm::CallingConv::Fast);
                cb.CreateStore(newCall, ci->getArgOperand(0));
                ci->eraseFromParent();
            }
            oldFn->eraseFromParent();
        }
    }

#if defined(_WIN32)
    for (llvm::Function& f : mod) {
        if (!f.isDeclaration()) {
            f.setDoesNotThrow();
            f.addFnAttr("stackrealign");
        }
    }
#endif
    return true;
}

bool ensureRuntimeBitcodeInMainJDLocked(
    JitRuntime& rt,
    const llvm::Module& shaderMod,
    std::string& outError) {
    auto hasPrefixDecl = [&](llvm::StringRef prefix) {
        for (const llvm::Function& f : shaderMod) {
            if (f.isDeclaration() && f.getName().starts_with(prefix)) {
                return true;
            }
        }
        return false;
    };

    auto loadIntoMainJD = [&](const unsigned char* bcData, size_t bcSize,
                              llvm::StringRef modName, llvm::StringRef fnPrefix) -> bool {
        auto bcCtx = std::make_unique<llvm::LLVMContext>();
        auto bcBuf = llvm::MemoryBuffer::getMemBuffer(
            llvm::StringRef(reinterpret_cast<const char*>(bcData), bcSize), modName, false);
        auto bcModOrErr = llvm::parseBitcodeFile(bcBuf->getMemBufferRef(), *bcCtx);
        if (!bcModOrErr) {
            outError = "Failed to parse embedded " + modName.str() + ": " + llvm::toString(bcModOrErr.takeError());
            return false;
        }
        std::unique_ptr<llvm::Module> bcMod = std::move(*bcModOrErr);
        bcMod->setTargetTriple(rt.jit->getTargetTriple());
        bcMod->setDataLayout(rt.jit->getDataLayout());
        for (llvm::Function& f : *bcMod) {
            if (!f.isDeclaration() && f.getName().starts_with(fnPrefix)) {
                f.removeFnAttr("target-cpu");
                f.removeFnAttr("target-features");
                f.removeFnAttr("tune-cpu");
                if (!isStrictFloatingPointEnabled()) {
                    f.addFnAttr("reciprocal-estimates", "none");
                }
#if defined(_WIN32)
                f.setDoesNotThrow();
                f.addFnAttr("stackrealign");
#endif
            }
        }
        if (auto err = rt.jit->addIRModule(
                rt.jit->getMainJITDylib(),
                llvm::orc::ThreadSafeModule(std::move(bcMod), std::move(bcCtx)))) {
            outError = llvm::toString(std::move(err));
            return false;
        }
        return true;
    };

    if (!rt.texSamplerLoaded && hasPrefixDecl("nacelle_tex_")) {
        if (!loadIntoMainJD(kTextureSamplerBitcode, kTextureSamplerBitcode_size, "texture_sampler.bc", "nacelle_tex_")) {
            return false;
        }
        rt.texSamplerLoaded = true;
    }
    if (!rt.mathSimdLoaded && hasPrefixDecl("nacelle_math_")) {
        if (!loadIntoMainJD(kMathSimdBitcode, kMathSimdBitcode_size, "math_simd.bc", "nacelle_math_")) {
            return false;
        }
        rt.mathSimdLoaded = true;
    }
    if (!rt.rayQueryRtLoaded && hasPrefixDecl("nacelle_rqbc_")) {
        if (!loadIntoMainJD(kRayQueryRtBitcode, kRayQueryRtBitcode_size, "rayquery_rt.bc", "nacelle_rqbc_")) {
            return false;
        }
        rt.rayQueryRtLoaded = true;
    }
    return true;
}

std::unique_ptr<llvm::Module> translateMLIRToLinkedLLVMIR(
    mlir::ModuleOp mlirMod,
    llvm::LLVMContext& llvmCtx,
    std::string& outError,
    bool linkRuntime) {
    mlir::PassManager pm(mlirMod.getContext());
    pm.addPass(mlir::createSCFToControlFlowPass());
    pm.addPass(mlir::createArithToLLVMConversionPass());
    pm.addPass(mlir::createConvertControlFlowToLLVMPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());

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

    // Apply fast-math flags to shader IR before linking freestanding runtime bitcode
    // so range-reduction, cancellation-free polynomials, and NaN/Inf/signed-zero paths
    // in math_simd.bc and texture_sampler.bc are not rewritten by reassoc/nsz/arcp.
    applyShaderFastMathFlags(*llvmMod);

    // Fuse same-argument (sinf, cosf) pairs in the same basic block into `nacelle_math_v<W>_sincosf`.
    {
        llvm::Type* voidTy = llvm::Type::getVoidTy(llvmCtx);
        llvm::Type* ptrTy = llvm::PointerType::getUnqual(llvmCtx);
        llvm::FunctionType* sincosFnTy = llvm::FunctionType::get(voidTy, {ptrTy, ptrTy, ptrTy}, false);
        for (llvm::Function& f : *llvmMod) {
            if (f.isDeclaration()) continue;
            for (llvm::BasicBlock& bb : f) {
                std::unordered_map<llvm::Value*, llvm::CallInst*> sinByArg;
                std::unordered_map<llvm::Value*, llvm::CallInst*> cosByArg;
                llvm::SmallVector<std::pair<llvm::CallInst*, llvm::CallInst*>, 8> pairs;
                for (llvm::Instruction& inst : bb) {
                    auto* ci = llvm::dyn_cast<llvm::CallInst>(&inst);
                    if (!ci || ci->arg_size() != 1) continue;
                    llvm::Function* callee = ci->getCalledFunction();
                    if (!callee) continue;
                    llvm::StringRef cname = callee->getName();
                    if (!cname.starts_with("nacelle_math_v")) continue;
                    llvm::Value* arg = ci->getArgOperand(0);
                    if (cname.ends_with("_sinf")) {
                        if (auto it = cosByArg.find(arg); it != cosByArg.end()) {
                            pairs.push_back({ci, it->second});
                            cosByArg.erase(it);
                        } else {
                            sinByArg.emplace(arg, ci);
                        }
                    } else if (cname.ends_with("_cosf")) {
                        if (auto it = sinByArg.find(arg); it != sinByArg.end()) {
                            pairs.push_back({it->second, ci});
                            sinByArg.erase(it);
                        } else {
                            cosByArg.emplace(arg, ci);
                        }
                    }
                }
                for (auto [sinCall, cosCall] : pairs) {
                    llvm::CallInst* anchor = sinCall->comesBefore(cosCall) ? sinCall : cosCall;
                    llvm::StringRef sinName = sinCall->getCalledFunction()->getName();
                    std::string scName = sinName.drop_back(4).str() + "sincosf";
                    llvm::FunctionCallee scCallee = llvmMod->getOrInsertFunction(scName, sincosFnTy);
                    llvm::IRBuilder<> eb(&f.getEntryBlock().front());
                    llvm::Type* vecTy = sinCall->getType();
                    llvm::AllocaInst* inSlot = eb.CreateAlloca(vecTy, nullptr, "sincos.in");
                    llvm::AllocaInst* outS = eb.CreateAlloca(vecTy, nullptr, "sincos.s");
                    llvm::AllocaInst* outC = eb.CreateAlloca(vecTy, nullptr, "sincos.c");
                    llvm::IRBuilder<> cb(anchor);
                    cb.CreateStore(sinCall->getArgOperand(0), inSlot);
                    cb.CreateCall(scCallee, {outS, outC, inSlot});
                    llvm::Value* sVal = cb.CreateLoad(vecTy, outS, "sincos.s.res");
                    llvm::Value* cVal = cb.CreateLoad(vecTy, outC, "sincos.c.res");
                    sinCall->replaceAllUsesWith(sVal);
                    cosCall->replaceAllUsesWith(cVal);
                    sinCall->eraseFromParent();
                    cosCall->eraseFromParent();
                }
            }
        }
    }

    // Rewrite vector-by-value `nacelle_math_v<W>_*` declarations/calls to ABI-neutral
    // `void(ptr out, ptr in)` so `AlwaysInlinerPass` matches `math_simd.bc` on all targets.
    {
        llvm::SmallVector<llvm::Function*, 16> mathDecls;
        for (llvm::Function& f : *llvmMod) {
            if (f.isDeclaration() && f.getName().starts_with("nacelle_math_") &&
                !f.getReturnType()->isVoidTy()) {
                mathDecls.push_back(&f);
            }
        }
        if (!mathDecls.empty()) {
            llvm::Type* voidTy = llvm::Type::getVoidTy(llvmCtx);
            llvm::Type* ptrTy = llvm::PointerType::getUnqual(llvmCtx);
            llvm::FunctionType* ptrFnTy = llvm::FunctionType::get(voidTy, {ptrTy, ptrTy}, false);
            llvm::DenseMap<std::pair<llvm::Function*, llvm::Type*>,
                           std::pair<llvm::AllocaInst*, llvm::AllocaInst*>> mathScratchSlots;
            for (llvm::Function* oldDecl : mathDecls) {
                std::string name = oldDecl->getName().str();
                oldDecl->setName(name + ".vec_stub");
                llvm::FunctionCallee newCallee = llvmMod->getOrInsertFunction(name, ptrFnTy);
                llvm::SmallVector<llvm::CallInst*, 16> calls;
                for (llvm::User* u : oldDecl->users()) {
                    if (auto* ci = llvm::dyn_cast<llvm::CallInst>(u)) {
                        calls.push_back(ci);
                    }
                }
                for (llvm::CallInst* ci : calls) {
                    llvm::Function* caller = ci->getFunction();
                    llvm::Type* vecTy = ci->getType();
                    auto& slots = mathScratchSlots[{caller, vecTy}];
                    if (!slots.first) {
                        llvm::IRBuilder<> eb(&caller->getEntryBlock().front());
                        slots.first = eb.CreateAlloca(vecTy, nullptr, "math.in");
                        slots.second = eb.CreateAlloca(vecTy, nullptr, "math.out");
                    }
                    llvm::IRBuilder<> cb(ci);
                    cb.CreateStore(ci->getArgOperand(0), slots.first);
                    cb.CreateCall(newCallee, {slots.second, slots.first});
                    llvm::Value* loaded = cb.CreateLoad(vecTy, slots.second, "math.res");
                    ci->replaceAllUsesWith(loaded);
                    ci->eraseFromParent();
                }
                oldDecl->eraseFromParent();
            }
        }
    }

    if (linkRuntime) {
        if (!linkRuntimeBitcodeAndInline(*llvmMod, outError)) {
            return nullptr;
        }
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

    if (const char* dumpEnv = std::getenv("NACELLE_DUMP_IR"); dumpEnv && dumpEnv[0] == '1') {
        static std::atomic<uint64_t> preSeq{0};
        uint64_t seq = preSeq.fetch_add(1, std::memory_order_relaxed);
        std::string path = "/tmp/nacelle_ir_" + std::to_string(seq) + "_pre_o3.ll";
        std::error_code ec;
        llvm::raw_fd_ostream os(path, ec);
        if (!ec) {
            llvmMod->print(os, nullptr);
            std::fprintf(stderr, "[nacelle] dumped pre-O3 LLVM IR: %s\n", path.c_str());
        }
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

    rt.memCache[key] = InMemCacheEntry{meta, addr0, 0, 0, 0, 0, 0, 0, 0, std::move(objBytes)};
    applyComputeMeta(meta, symbolName, addr0, outPSO);
    return true;
}

bool tryLoadVertexStageFromCache(
    uint64_t vsHashLo, uint64_t vsHashHi,
    const std::string& vsSym,
    CachedVSMeta& outVSMeta,
    CompiledVertexFn& outVSFn,
    std::vector<ShaderSigElement>* outVSInputs,
    std::vector<ShaderSigElement>* outVSOutputs,
    std::vector<DxilSigRecord>* outVSOutputRecs) {
    JitRuntime& rt = getJitRuntime();
    std::filesystem::path dir = getCacheDirectory();
    std::string vsHex = hashToHex128(vsHashLo, vsHashHi);
    std::string vsKey = (dir / vsHex).string();
    uint32_t targetW = getTargetSimdWidth();

    {
        std::lock_guard<std::mutex> lock(rt.mu);
        auto it = rt.vsMemCache.find(vsKey);
        if (it != rt.vsMemCache.end() && it->second.vsAddr != 0 && it->second.meta.simdWidth == targetW) {
            outVSMeta = it->second.meta;
            outVSFn = reinterpret_cast<CompiledVertexFn>(it->second.vsAddr);
            if (outVSInputs && !it->second.vsInputs.empty()) *outVSInputs = it->second.vsInputs;
            if (outVSOutputs && !it->second.vsOutputs.empty()) *outVSOutputs = it->second.vsOutputs;
            if (outVSOutputRecs && !it->second.vsOutputRecs.empty()) *outVSOutputRecs = it->second.vsOutputRecs;
            return true;
        }
    }

    std::filesystem::path metaPath = dir / (vsHex + ".vsmeta");
    std::filesystem::path objPath = dir / (vsHex + ".vso");
    std::vector<uint8_t> metaBytes, objBytes;
    if (!readFileBytes(metaPath, metaBytes) || metaBytes.size() != sizeof(CachedVSMeta)) return false;
    CachedVSMeta vsMeta{};
    std::memcpy(&vsMeta, metaBytes.data(), sizeof(CachedVSMeta));
    if (vsMeta.magic != CachedVSMeta{}.magic || vsMeta.simdWidth != targetW) return false;
    if (!readFileBytes(objPath, objBytes) || objBytes.empty()) return false;

    std::lock_guard<std::mutex> lock(rt.mu);
    auto it = rt.vsMemCache.find(vsKey);
    if (it != rt.vsMemCache.end() && it->second.vsAddr != 0 && it->second.meta.simdWidth == targetW) {
        outVSMeta = it->second.meta;
        outVSFn = reinterpret_cast<CompiledVertexFn>(it->second.vsAddr);
        if (outVSInputs && !it->second.vsInputs.empty()) *outVSInputs = it->second.vsInputs;
        if (outVSOutputs && !it->second.vsOutputs.empty()) *outVSOutputs = it->second.vsOutputs;
        if (outVSOutputRecs && !it->second.vsOutputRecs.empty()) *outVSOutputRecs = it->second.vsOutputRecs;
        return true;
    }

    uint64_t vsAddr = 0, dummy = 0;
    std::string err;
    if (!loadObjectIntoJitLocked(rt, objBytes.data(), objBytes.size(), vsKey + ".vso", vsSym, "", vsAddr, dummy, err)) {
        return false;
    }

    InMemVSCacheEntry entry;
    entry.meta = vsMeta;
    entry.vsAddr = vsAddr;
    entry.tier2Done = true;
    entry.objBytes = std::move(objBytes);
    rt.vsMemCache[vsKey] = std::move(entry);
    outVSMeta = vsMeta;
    outVSFn = reinterpret_cast<CompiledVertexFn>(vsAddr);
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
        if (it != rt.memCache.end() &&
            (it->second.meta.hasMS ? (it->second.msAddr != 0) : (it->second.addr0 != 0)) &&
            (psSym.empty() || it->second.addr1 != 0) &&
            it->second.meta.simdWidth == targetW) {
            std::string effVsSym = (it->second.meta.separateVSObject != 0)
                                       ? ("nacelle_vs_" + hashToHex128(it->second.meta.vsHashLo, it->second.meta.vsHashHi))
                                       : vsSym;
            applyGraphicsMeta(it->second.meta, effVsSym, psSym, it->second.addr0, it->second.addr1, it->second.gsAddr,
                              it->second.hsAddr, it->second.hsPcAddr, it->second.dsAddr,
                              it->second.msAddr, it->second.asAddr, outPSO);
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

    if (meta.separateVSObject != 0) {
        std::string sepVsSym = "nacelle_vs_" + hashToHex128(meta.vsHashLo, meta.vsHashHi);
        CachedVSMeta vsMeta{};
        CompiledVertexFn vsFnPtr = nullptr;
        if (!tryLoadVertexStageFromCache(meta.vsHashLo, meta.vsHashHi, sepVsSym, vsMeta, vsFnPtr)) {
            return false;
        }

        std::lock_guard<std::mutex> lock(rt.mu);
        auto it = rt.memCache.find(key);
        if (it != rt.memCache.end() && it->second.addr0 != 0 && (psSym.empty() || it->second.addr1 != 0) &&
            it->second.meta.simdWidth == targetW) {
            applyGraphicsMeta(it->second.meta, sepVsSym, psSym, it->second.addr0, it->second.addr1, 0, 0, 0, 0, 0, 0, outPSO);
            return true;
        }

        uint64_t psAddr = 0, dummy = 0;
        std::string err;
        if (!loadObjectIntoJitLocked(rt, objBytes.data(), objBytes.size(), key + ".o", psSym, "", psAddr, dummy, err)) {
            return false;
        }
        uint64_t vsAddr = reinterpret_cast<uint64_t>(vsFnPtr);
        rt.memCache[key] = InMemCacheEntry{meta, vsAddr, psAddr, 0, 0, 0, 0, 0, 0, std::move(objBytes)};
        applyGraphicsMeta(meta, sepVsSym, psSym, vsAddr, psAddr, 0, 0, 0, 0, 0, 0, outPSO);
        return true;
    }

    std::lock_guard<std::mutex> lock(rt.mu);
    auto it = rt.memCache.find(key);
    if (it != rt.memCache.end() &&
        (it->second.meta.hasMS ? (it->second.msAddr != 0) : (it->second.addr0 != 0)) &&
        (psSym.empty() || it->second.addr1 != 0) &&
        it->second.meta.simdWidth == targetW) {
        applyGraphicsMeta(it->second.meta, vsSym, psSym, it->second.addr0, it->second.addr1, it->second.gsAddr,
                          it->second.hsAddr, it->second.hsPcAddr, it->second.dsAddr,
                          it->second.msAddr, it->second.asAddr, outPSO);
        return true;
    }

    uint64_t primaryAddr = 0, addr1 = 0, gsAddr = 0, hsAddr = 0, hsPcAddr = 0, dsAddr = 0, msAddr = 0, asAddr = 0;
    std::string primarySym = meta.hasMS ? (vsSym + "_ms") : vsSym;
    std::string gsSym = meta.hasGS ? (vsSym + "_gs") : std::string{};
    std::string hsSym = meta.hasTess ? (vsSym + "_hs") : std::string{};
    std::string hsPcSym = meta.hasTess ? (vsSym + "_hspc") : std::string{};
    std::string dsSym = meta.hasTess ? (vsSym + "_ds") : std::string{};
    std::string asSym = meta.hasAS ? (vsSym + "_as") : std::string{};
    std::string err;
    if (meta.hasMS) {
        if (!loadObjectIntoJitLocked(rt, objBytes.data(), objBytes.size(), key + ".o", primarySym, psSym, primaryAddr, addr1, err,
                                     asSym, &asAddr)) {
            return false;
        }
        msAddr = primaryAddr;
    } else {
        if (!loadObjectIntoJitLocked(rt, objBytes.data(), objBytes.size(), key + ".o", primarySym, psSym, primaryAddr, addr1, err,
                                     gsSym, &gsAddr, hsSym, &hsAddr, hsPcSym, &hsPcAddr, dsSym, &dsAddr)) {
            return false;
        }
    }
    uint64_t addr0 = meta.hasMS ? 0 : primaryAddr;

    rt.memCache[key] = InMemCacheEntry{meta, addr0, addr1, gsAddr, hsAddr, hsPcAddr, dsAddr, msAddr, asAddr, std::move(objBytes)};
    applyGraphicsMeta(meta, vsSym, psSym, addr0, addr1, gsAddr, hsAddr, hsPcAddr, dsAddr, msAddr, asAddr, outPSO);
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
        if (!ensureRuntimeBitcodeInMainJDLocked(rt, *llvmMod, outError)) return false;
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

    ensureTier2PoolReady(rt).detach_task([box, meta, symbolName, cacheDir, hex, bcStr = std::move(bcStr)]() {
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
                meta, addr0, 0, 0, 0, 0, 0, 0, 0,
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
    const char* noT2 = std::getenv("NACELLE_DISABLE_TIER2");
    const bool scheduleTier2 = !(noT2 && noT2[0] == '1');

    llvm::SmallString<0> bcBytes;
    if (scheduleTier2) {
        llvm::raw_svector_ostream bcOS(bcBytes);
        llvm::WriteBitcodeToFile(*llvmMod, bcOS);
    }

    runTier1Optimize(*llvmMod);

    uint64_t vsAddr = 0, psAddr = 0, gsAddr = 0, hsAddr = 0, hsPcAddr = 0, dsAddr = 0, msAddr = 0, asAddr = 0;
    std::string gsSym = meta.hasGS ? (vsSym + "_gs") : std::string{};
    std::string hsSym = meta.hasTess ? (vsSym + "_hs") : std::string{};
    std::string hsPcSym = meta.hasTess ? (vsSym + "_hspc") : std::string{};
    std::string dsSym = meta.hasTess ? (vsSym + "_ds") : std::string{};
    std::string msSym = meta.hasMS ? (vsSym + "_ms") : std::string{};
    std::string asSym = meta.hasAS ? (vsSym + "_as") : std::string{};
    {
        std::lock_guard<std::mutex> lock(rt.mu);
        ensureLLJITLocked(rt, outError);
        if (!rt.jit) return false;
        if (!ensureRuntimeBitcodeInMainJDLocked(rt, *llvmMod, outError)) return false;
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
        if (meta.hasMS) {
            if (!lookupSym(msSym, msAddr) || !lookupSym(psSym, psAddr)) return false;
            if (!asSym.empty() && !lookupSym(asSym, asAddr)) return false;
        } else {
            if (!lookupSym(vsSym, vsAddr) || !lookupSym(psSym, psAddr)) return false;
            if (!gsSym.empty() && !lookupSym(gsSym, gsAddr)) return false;
            if (!hsSym.empty() && (!lookupSym(hsSym, hsAddr) || !lookupSym(hsPcSym, hsPcAddr) || !lookupSym(dsSym, dsAddr))) {
                return false;
            }
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
    outPSO.msFn.store(reinterpret_cast<CompiledMeshFn>(msAddr), std::memory_order_release);
    outPSO.asFn.store(reinterpret_cast<CompiledAmplificationFn>(asAddr), std::memory_order_release);

    if (!scheduleTier2) {
        return true;
    }

    auto box = std::make_shared<PsoUpdateBox>();
    box->gs = &outPSO;
    outPSO.asyncBox = box;

    std::filesystem::path cacheDir = getCacheDirectory();
    std::string hex = hashToHex128(outPSO.cacheHashLo, outPSO.cacheHashHi);
    std::string bcStr(bcBytes.data(), bcBytes.size());

    ensureTier2PoolReady(rt).detach_task([box, meta, vsSym, psSym, cacheDir, hex, bcStr = std::move(bcStr)]() {
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

        uint64_t primaryAddr2 = 0, addr1 = 0, gsAddr2 = 0, hsAddr2 = 0, hsPcAddr2 = 0, dsAddr2 = 0, msAddr2 = 0, asAddr2 = 0;
        std::string primarySym2 = meta.hasMS ? (vsSym + "_ms") : vsSym;
        std::string gsSym2 = meta.hasGS ? (vsSym + "_gs") : std::string{};
        std::string hsSym2 = meta.hasTess ? (vsSym + "_hs") : std::string{};
        std::string hsPcSym2 = meta.hasTess ? (vsSym + "_hspc") : std::string{};
        std::string dsSym2 = meta.hasTess ? (vsSym + "_ds") : std::string{};
        std::string asSym2 = meta.hasAS ? (vsSym + "_as") : std::string{};
        {
            std::lock_guard<std::mutex> lock(rt.mu);
            std::string key = (cacheDir / hex).string();
            if (meta.hasMS) {
                if (!loadObjectIntoJitLocked(rt, reinterpret_cast<const uint8_t*>(objBytes.data()),
                                             objBytes.size(), key + ".o", primarySym2, psSym, primaryAddr2, addr1, err,
                                             asSym2, &asAddr2)) {
                    return;
                }
                msAddr2 = primaryAddr2;
            } else {
                if (!loadObjectIntoJitLocked(rt, reinterpret_cast<const uint8_t*>(objBytes.data()),
                                             objBytes.size(), key + ".o", primarySym2, psSym, primaryAddr2, addr1, err,
                                             gsSym2, &gsAddr2, hsSym2, &hsAddr2, hsPcSym2, &hsPcAddr2, dsSym2, &dsAddr2)) {
                    return;
                }
            }
            uint64_t addr0 = meta.hasMS ? 0 : primaryAddr2;
            rt.memCache[key] = InMemCacheEntry{
                meta, addr0, addr1, gsAddr2, hsAddr2, hsPcAddr2, dsAddr2, msAddr2, asAddr2,
                std::vector<uint8_t>(reinterpret_cast<const uint8_t*>(objBytes.begin()),
                                     reinterpret_cast<const uint8_t*>(objBytes.end()))};
        }

        std::lock_guard<std::mutex> psoLock(box->mu);
        if (box->gs) {
            box->gs->vsFn.store(reinterpret_cast<CompiledVertexFn>(meta.hasMS ? 0 : primaryAddr2), std::memory_order_release);
            box->gs->psFn.store(reinterpret_cast<CompiledPixelQuadFn>(addr1), std::memory_order_release);
            box->gs->gsFn.store(reinterpret_cast<CompiledGeometryFn>(gsAddr2), std::memory_order_release);
            box->gs->hsFn.store(reinterpret_cast<CompiledHullCPFn>(hsAddr2), std::memory_order_release);
            box->gs->hsPcFn.store(reinterpret_cast<CompiledHullPCFn>(hsPcAddr2), std::memory_order_release);
            box->gs->dsFn.store(reinterpret_cast<CompiledDomainFn>(dsAddr2), std::memory_order_release);
            box->gs->msFn.store(reinterpret_cast<CompiledMeshFn>(msAddr2), std::memory_order_release);
            box->gs->asFn.store(reinterpret_cast<CompiledAmplificationFn>(asAddr2), std::memory_order_release);
        }
    });

    return true;
}

bool jitCompileVertexStageTier1AndScheduleTier2(
    std::unique_ptr<llvm::Module> llvmMod,
    std::unique_ptr<llvm::LLVMContext> llvmCtx,
    uint64_t vsHashLo, uint64_t vsHashHi,
    const std::string& vsSym,
    const CachedVSMeta& vsMeta,
    const std::vector<ShaderSigElement>& vsInputs,
    const std::vector<ShaderSigElement>& vsOutputs,
    const std::vector<DxilSigRecord>& vsOutputRecs,
    CompiledGraphicsPSO& outPSO,
    std::string& outError) {
    g_vertexStageCompileCount.fetch_add(1, std::memory_order_relaxed);
    JitRuntime& rt = getJitRuntime();
    const char* noT2 = std::getenv("NACELLE_DISABLE_TIER2");
    const bool scheduleTier2 = !(noT2 && noT2[0] == '1');

    llvm::SmallString<0> bcBytes;
    if (scheduleTier2) {
        llvm::raw_svector_ostream bcOS(bcBytes);
        llvm::WriteBitcodeToFile(*llvmMod, bcOS);
    }

    runTier1Optimize(*llvmMod);

    std::filesystem::path cacheDir = getCacheDirectory();
    std::string vsHex = hashToHex128(vsHashLo, vsHashHi);
    std::string vsKey = (cacheDir / vsHex).string();
    uint64_t vsAddr = 0;
    bool launchedTier2 = false;
    {
        std::lock_guard<std::mutex> lock(rt.mu);
        auto existing = rt.vsMemCache.find(vsKey);
        if (existing != rt.vsMemCache.end() && existing->second.vsAddr != 0) {
            vsAddr = existing->second.vsAddr;
        } else {
            ensureLLJITLocked(rt, outError);
            if (!rt.jit) return false;
            if (!ensureRuntimeBitcodeInMainJDLocked(rt, *llvmMod, outError)) return false;
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
            auto symOrErr = rt.jit->lookup(jd, vsSym);
            if (!symOrErr) {
                outError = llvm::toString(symOrErr.takeError());
                return false;
            }
            vsAddr = symOrErr->getValue();

            InMemVSCacheEntry entry;
            entry.meta = vsMeta;
            entry.vsAddr = vsAddr;
            entry.tier2Done = !scheduleTier2;
            entry.vsInputs = vsInputs;
            entry.vsOutputs = vsOutputs;
            entry.vsOutputRecs = vsOutputRecs;
            rt.vsMemCache[vsKey] = std::move(entry);
            launchedTier2 = scheduleTier2;
        }
    }

    outPSO.vsSymbolName = vsSym;
    outPSO.vsFn.store(reinterpret_cast<CompiledVertexFn>(vsAddr), std::memory_order_release);

    if (launchedTier2) {
        std::string bcStr(bcBytes.data(), bcBytes.size());
        ensureTier2PoolReady(rt).detach_task([vsMeta, vsHashLo, vsHashHi, vsSym, cacheDir, vsHex, vsKey, bcStr = std::move(bcStr)]() {
            auto ctx2 = std::make_unique<llvm::LLVMContext>();
            auto buf = llvm::MemoryBuffer::getMemBuffer(bcStr, vsHex + ".bc", false);
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
            writeFileAtomic(cacheDir / (vsHex + ".vso"), objBytes.data(), objBytes.size(), tId);
            writeFileAtomic(cacheDir / (vsHex + ".vsmeta"), &vsMeta, sizeof(CachedVSMeta), tId);

            uint64_t tier2VSAddr = 0, dummy = 0;
            std::vector<std::shared_ptr<PsoUpdateBox>> waiters;
            {
                std::lock_guard<std::mutex> lock(rt.mu);
                if (!loadObjectIntoJitLocked(rt, reinterpret_cast<const uint8_t*>(objBytes.data()),
                                             objBytes.size(), vsKey + ".vso", vsSym, "", tier2VSAddr, dummy, err)) {
                    return;
                }
                auto& entry = rt.vsMemCache[vsKey];
                entry.meta = vsMeta;
                entry.vsAddr = tier2VSAddr;
                entry.tier2Done = true;
                entry.objBytes.assign(reinterpret_cast<const uint8_t*>(objBytes.begin()),
                                      reinterpret_cast<const uint8_t*>(objBytes.end()));
                waiters.swap(entry.waiters);
                for (auto& kv : rt.memCache) {
                    if (kv.second.meta.separateVSObject != 0 &&
                        kv.second.meta.vsHashLo == vsHashLo &&
                        kv.second.meta.vsHashHi == vsHashHi) {
                        kv.second.addr0 = tier2VSAddr;
                    }
                }
            }

            for (const auto& w : waiters) {
                if (!w) continue;
                std::lock_guard<std::mutex> psoLock(w->mu);
                if (w->gs) {
                    w->gs->vsFn.store(reinterpret_cast<CompiledVertexFn>(tier2VSAddr), std::memory_order_release);
                }
            }
        });
    }

    return true;
}

bool jitCompilePixelStageTier1AndScheduleTier2(
    std::unique_ptr<llvm::Module> llvmMod,
    std::unique_ptr<llvm::LLVMContext> llvmCtx,
    uint64_t vsHashLo, uint64_t vsHashHi,
    const std::string& psSym,
    const CachedPSOMeta& meta,
    CompiledGraphicsPSO& outPSO,
    std::string& outError) {
    JitRuntime& rt = getJitRuntime();
    const char* noT2 = std::getenv("NACELLE_DISABLE_TIER2");
    const bool scheduleTier2 = !(noT2 && noT2[0] == '1');

    llvm::SmallString<0> bcBytes;
    if (scheduleTier2) {
        llvm::raw_svector_ostream bcOS(bcBytes);
        llvm::WriteBitcodeToFile(*llvmMod, bcOS);
    }

    runTier1Optimize(*llvmMod);

    std::filesystem::path cacheDir = getCacheDirectory();
    std::string vsHex = hashToHex128(vsHashLo, vsHashHi);
    std::string vsKey = (cacheDir / vsHex).string();

    auto box = std::make_shared<PsoUpdateBox>();
    box->gs = &outPSO;
    outPSO.asyncBox = box;

    uint64_t psAddr = 0;
    uint64_t latestVSAddr = 0;
    {
        std::lock_guard<std::mutex> lock(rt.mu);
        ensureLLJITLocked(rt, outError);
        if (!rt.jit) return false;
        if (!ensureRuntimeBitcodeInMainJDLocked(rt, *llvmMod, outError)) return false;
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
        auto symOrErr = rt.jit->lookup(jd, psSym);
        if (!symOrErr) {
            outError = llvm::toString(symOrErr.takeError());
            return false;
        }
        psAddr = symOrErr->getValue();

        auto vsIt = rt.vsMemCache.find(vsKey);
        if (vsIt != rt.vsMemCache.end()) {
            latestVSAddr = vsIt->second.vsAddr;
            if (!vsIt->second.tier2Done) {
                vsIt->second.waiters.push_back(box);
            }
        }
    }

    if (latestVSAddr != 0) {
        outPSO.vsFn.store(reinterpret_cast<CompiledVertexFn>(latestVSAddr), std::memory_order_release);
    }
    outPSO.psSymbolName = psSym;
    outPSO.psFn.store(reinterpret_cast<CompiledPixelQuadFn>(psAddr), std::memory_order_release);

    if (!scheduleTier2) {
        return true;
    }

    std::string hex = hashToHex128(outPSO.cacheHashLo, outPSO.cacheHashHi);
    std::string bcStr(bcBytes.data(), bcBytes.size());

    ensureTier2PoolReady(rt).detach_task([box, meta, psSym, cacheDir, hex, vsKey, bcStr = std::move(bcStr)]() {
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

        uint64_t tier2PSAddr = 0, dummy = 0;
        uint64_t curVSAddr = 0;
        {
            std::lock_guard<std::mutex> lock(rt.mu);
            std::string key = (cacheDir / hex).string();
            if (!loadObjectIntoJitLocked(rt, reinterpret_cast<const uint8_t*>(objBytes.data()),
                                         objBytes.size(), key + ".o", psSym, "", tier2PSAddr, dummy, err)) {
                return;
            }
            auto vsIt = rt.vsMemCache.find(vsKey);
            if (vsIt != rt.vsMemCache.end()) {
                curVSAddr = vsIt->second.vsAddr;
            }
            rt.memCache[key] = InMemCacheEntry{
                meta, curVSAddr, tier2PSAddr, 0, 0, 0, 0, 0, 0,
                std::vector<uint8_t>(reinterpret_cast<const uint8_t*>(objBytes.begin()),
                                     reinterpret_cast<const uint8_t*>(objBytes.end()))};
        }

        std::lock_guard<std::mutex> psoLock(box->mu);
        if (box->gs) {
            if (curVSAddr != 0) {
                box->gs->vsFn.store(reinterpret_cast<CompiledVertexFn>(curVSAddr), std::memory_order_release);
            }
            box->gs->psFn.store(reinterpret_cast<CompiledPixelQuadFn>(tier2PSAddr), std::memory_order_release);
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

    thread_local mlir::MLIRContext mlirCtx(mlir::MLIRContext::Threading::DISABLED);
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
    auto linkedMod = translateMLIRToLinkedLLVMIR(*mlirMod, *outLlvmCtx, outError, /*linkRuntime=*/false);
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
    if (auto* pool = getJitRuntime().activeTier2Pool.load(std::memory_order_acquire)) {
        pool->wait();
    }
}

void clearInMemoryPSOCacheForTesting() {
    JitRuntime& rt = getJitRuntime();
    std::lock_guard<std::mutex> lock(rt.mu);
    rt.memCache.clear();
    rt.vsMemCache.clear();
    g_vertexStageCompileCount.store(0, std::memory_order_relaxed);
}

uint64_t getVertexStageCompileCount() {
    return g_vertexStageCompileCount.load(std::memory_order_relaxed);
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
    if (meta.separateVSObject == 0) {
        outBlob.resize(sizeof(CachedPSOMeta) + sizeof(uint32_t) + objLen);
        std::memcpy(outBlob.data(), &meta, sizeof(CachedPSOMeta));
        std::memcpy(outBlob.data() + sizeof(CachedPSOMeta), &objLen, sizeof(uint32_t));
        std::memcpy(outBlob.data() + sizeof(CachedPSOMeta) + sizeof(uint32_t), objBytes.data(), objLen);
        return true;
    }

    std::string vsHex = hashToHex128(meta.vsHashLo, meta.vsHashHi);
    std::string vsKey = (dir / vsHex).string();
    CachedVSMeta vsMeta{};
    std::vector<uint8_t> vsObjBytes;
    bool vsFound = false;
    {
        JitRuntime& rt = getJitRuntime();
        std::lock_guard<std::mutex> lock(rt.mu);
        auto vit = rt.vsMemCache.find(vsKey);
        if (vit != rt.vsMemCache.end() && !vit->second.objBytes.empty()) {
            vsMeta = vit->second.meta;
            vsObjBytes = vit->second.objBytes;
            vsFound = true;
        }
    }
    if (!vsFound) {
        std::vector<uint8_t> vsMetaBytes;
        if (!readFileBytes(dir / (vsHex + ".vsmeta"), vsMetaBytes) || vsMetaBytes.size() != sizeof(CachedVSMeta)) {
            return false;
        }
        std::memcpy(&vsMeta, vsMetaBytes.data(), sizeof(CachedVSMeta));
        if (!readFileBytes(dir / (vsHex + ".vso"), vsObjBytes) || vsObjBytes.empty()) {
            return false;
        }
    }

    uint32_t vsObjLen = static_cast<uint32_t>(vsObjBytes.size());
    size_t psoChunk = sizeof(CachedPSOMeta) + sizeof(uint32_t) + objLen;
    size_t vsChunk = sizeof(CachedVSMeta) + sizeof(uint32_t) + vsObjLen;
    outBlob.resize(psoChunk + vsChunk);
    uint8_t* dst = outBlob.data();
    std::memcpy(dst, &meta, sizeof(CachedPSOMeta));
    std::memcpy(dst + sizeof(CachedPSOMeta), &objLen, sizeof(uint32_t));
    std::memcpy(dst + sizeof(CachedPSOMeta) + sizeof(uint32_t), objBytes.data(), objLen);
    dst += psoChunk;
    std::memcpy(dst, &vsMeta, sizeof(CachedVSMeta));
    std::memcpy(dst + sizeof(CachedVSMeta), &vsObjLen, sizeof(uint32_t));
    std::memcpy(dst + sizeof(CachedVSMeta) + sizeof(uint32_t), vsObjBytes.data(), vsObjLen);
    return true;
}

bool importCachedPSOBlob(uint64_t hashLo, uint64_t hashHi, const uint8_t* data, size_t size) {
    if (!data || size < sizeof(CachedPSOMeta) + sizeof(uint32_t)) return false;
    CachedPSOMeta meta{};
    std::memcpy(&meta, data, sizeof(CachedPSOMeta));
    if (meta.magic != CachedPSOMeta{}.magic) return false;
    uint32_t objLen = 0;
    std::memcpy(&objLen, data + sizeof(CachedPSOMeta), sizeof(uint32_t));
    size_t psoChunk = sizeof(CachedPSOMeta) + sizeof(uint32_t) + objLen;
    if (psoChunk > size || objLen == 0) return false;

    std::filesystem::path dir = getCacheDirectory();
    std::string hex = hashToHex128(hashLo, hashHi);
    uint64_t tId = getJitRuntime().tmpSeq.fetch_add(1, std::memory_order_relaxed);
    const uint8_t* objPtr = data + sizeof(CachedPSOMeta) + sizeof(uint32_t);

    if (meta.separateVSObject != 0) {
        if (psoChunk + sizeof(CachedVSMeta) + sizeof(uint32_t) > size) return false;
        const uint8_t* vsBase = data + psoChunk;
        CachedVSMeta vsMeta{};
        std::memcpy(&vsMeta, vsBase, sizeof(CachedVSMeta));
        if (vsMeta.magic != CachedVSMeta{}.magic) return false;
        uint32_t vsObjLen = 0;
        std::memcpy(&vsObjLen, vsBase + sizeof(CachedVSMeta), sizeof(uint32_t));
        if (psoChunk + sizeof(CachedVSMeta) + sizeof(uint32_t) + vsObjLen > size || vsObjLen == 0) return false;

        std::string vsHex = hashToHex128(meta.vsHashLo, meta.vsHashHi);
        const uint8_t* vsObjPtr = vsBase + sizeof(CachedVSMeta) + sizeof(uint32_t);
        if (!writeFileAtomic(dir / (vsHex + ".vsmeta"), &vsMeta, sizeof(CachedVSMeta), tId) ||
            !writeFileAtomic(dir / (vsHex + ".vso"), vsObjPtr, vsObjLen, tId)) {
            return false;
        }
    }

    bool ok1 = writeFileAtomic(dir / (hex + ".meta"), &meta, sizeof(CachedPSOMeta), tId);
    bool ok2 = writeFileAtomic(dir / (hex + ".o"), objPtr, objLen, tId);
    return ok1 && ok2;
}

bool compileDxrShader(
    const D3D12_SHADER_BYTECODE& dxilLib,
    const std::string& exportName,
    const RootSignatureLayout& rootLayout,
    uint32_t simdWidth,
    DxrShaderFn& outFn,
    std::string& outError) {
    outFn = nullptr;
    auto llvmCtx = std::make_unique<llvm::LLVMContext>();
    DxilLoadedShader shader;
    if (!loadAndUpgradeDXILLibraryExport(dxilLib, exportName, *llvmCtx, shader, outError)) {
        return false;
    }

    const RootSignatureLayout& effectiveLayout =
        (rootLayout.params.empty() && rootLayout.staticSamplers.empty() && shader.hasEmbeddedRootSig)
            ? shader.embeddedRootSig
            : rootLayout;

    thread_local mlir::MLIRContext mlirCtx;
    auto mlirMod = importAndLiftDXILToSCF(std::move(shader.module), shader.entryName, mlirCtx, outError);
    if (!mlirMod) return false;

    const std::string symbolName = "nacelle_dxr_" + exportName;
    if (!packetizeDxrEntry(*mlirMod, shader, effectiveLayout, simdWidth, symbolName, outError)) {
        return false;
    }

    auto outLlvmCtx = std::make_unique<llvm::LLVMContext>();
    auto linkedMod = translateMLIRToLinkedLLVMIR(*mlirMod, *outLlvmCtx, outError, /*linkRuntime=*/false);
    if (!linkedMod) return false;

    llvm::SmallVector<char, 0> objBytes;
    if (!emitObjectBytesO3(*linkedMod, objBytes, outError)) return false;

    auto& rt = getJitRuntime();
    std::lock_guard<std::mutex> lock(rt.mu);
    ensureLLJITLocked(rt, outError);
    if (!rt.jit) return false;
    if (!ensureRuntimeBitcodeInMainJDLocked(rt, *linkedMod, outError)) return false;

    uint64_t addr0 = 0, addr1 = 0;
    const std::string objName = "dxr_" + exportName + "_" + std::to_string(rt.dylibSeq.fetch_add(1, std::memory_order_relaxed)) + ".o";
    if (!loadObjectIntoJitLocked(
            rt, reinterpret_cast<const uint8_t*>(objBytes.data()), objBytes.size(),
            objName, symbolName, "", addr0, addr1, outError)) {
        return false;
    }
    outFn = reinterpret_cast<DxrShaderFn>(addr0);
    return outFn != nullptr;
}

} // namespace nacelle
