#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "shader_compiler.h"

namespace llvm {
class LLVMContext;
class Module;
} // namespace llvm

namespace mlir {
class ModuleOp;
}

namespace nacelle {

struct PsoUpdateBox {
    std::mutex mu;
    CompiledComputePSO* cs = nullptr;
    CompiledGraphicsPSO* gs = nullptr;
};

struct CachedPSOMeta {
    uint32_t magic = 0x4E434C31; // 'NCL1'
    uint32_t simdWidth = 4;
    // Compute
    uint32_t threadGroupSize[3]{1, 1, 1};
    uint32_t groupSharedBytes = 0;
    // Graphics
    uint32_t numVSOutputRegs = 0;
    int32_t posOutputReg = 0;
    uint32_t numPSInputRegs = 0;
    uint32_t flatInterpMask = 0;
    uint8_t psWritesDepth = 0;
    uint8_t psUsesDiscard = 0;
    uint8_t psHasSideEffects = 0;
    uint8_t reserved = 0;
};

/// Computes canonical 128-bit XXH3 hash (hex string + hi/lo) for compute or graphics PSO inputs.
std::string hashToHex128(uint64_t lo, uint64_t hi);
void computeComputePSOHash(
    const D3D12_COMPUTE_PIPELINE_STATE_DESC& desc,
    const RootSignatureLayout& rootLayout,
    uint32_t simdWidth,
    uint64_t& outLo,
    uint64_t& outHi);
void computeGraphicsPSOHash(
    const D3D12_GRAPHICS_PIPELINE_STATE_DESC& desc,
    const RootSignatureLayout& rootLayout,
    uint32_t simdWidth,
    uint64_t& outLo,
    uint64_t& outHi);

/// Converts lowered MLIR (SCF/Vector/Arith/Math/Func/LLVM) into LLVM IR and links texture_sampler.bc if referenced.
std::unique_ptr<llvm::Module> translateMLIRToLinkedLLVMIR(
    mlir::ModuleOp mlirMod,
    llvm::LLVMContext& llvmCtx,
    std::string& outError);

/// Checks memory & disk ObjectCache (<cacheDir>/<hex32>.o + .meta). If found, loads into LLJIT and resolves symbol(s).
bool tryLoadComputePSOFromCache(
    uint64_t hashLo, uint64_t hashHi,
    const std::string& symbolName,
    CompiledComputePSO& outPSO);

bool tryLoadGraphicsPSOFromCache(
    uint64_t hashLo, uint64_t hashHi,
    const std::string& vsSym,
    const std::string& psSym,
    CompiledGraphicsPSO& outPSO);

/// Compiles `llvmMod` at Tier-1 (-O1) synchronously for immediate execution and schedules async Tier-2 (-O3)
/// optimization + disk cache `.o` emission on the JIT background thread pool.
bool jitCompileComputeTier1AndScheduleTier2(
    std::unique_ptr<llvm::Module> llvmMod,
    std::unique_ptr<llvm::LLVMContext> llvmCtx,
    const std::string& symbolName,
    const CachedPSOMeta& meta,
    CompiledComputePSO& outPSO,
    std::string& outError);

bool jitCompileGraphicsTier1AndScheduleTier2(
    std::unique_ptr<llvm::Module> llvmMod,
    std::unique_ptr<llvm::LLVMContext> llvmCtx,
    const std::string& vsSym,
    const std::string& psSym,
    const CachedPSOMeta& meta,
    CompiledGraphicsPSO& outPSO,
    std::string& outError);

void clearInMemoryPSOCacheForTesting();
bool exportCachedPSOBlob(uint64_t hashLo, uint64_t hashHi, std::vector<uint8_t>& outBlob);
bool importCachedPSOBlob(uint64_t hashLo, uint64_t hashHi, const uint8_t* data, size_t size);

} // namespace nacelle
