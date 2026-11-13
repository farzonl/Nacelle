#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "../src/common/platform.h"

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Signals.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>

#include "../src/compiler/dxil_loader.h"
#include "../src/compiler/jit_engine.h"
#include "../src/compiler/mlir_pipeline.h"
#include "../src/compiler/pso_synthesizer.h"
#include "../src/compiler/shader_compiler.h"
#include "../src/compiler/simt_to_simd_pass.h"

namespace {

bool readBinaryFile(const std::string& path, std::vector<uint8_t>& out) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) return false;
    auto sz = ifs.tellg();
    if (sz < 0) return false;
    out.resize(static_cast<size_t>(sz));
    ifs.seekg(0, std::ios::beg);
    if (sz > 0 && !ifs.read(reinterpret_cast<char*>(out.data()), sz)) return false;
    return true;
}

void printUsage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " [options] [<compute.dxil>]\n"
              << "  --cs <path>              Compute DXIL shader\n"
              << "  --vs <path>              Vertex DXIL shader (graphics mode)\n"
              << "  --ps <path>              Pixel DXIL shader (graphics mode)\n"
              << "  --simd-width=4|8|16      Target SIMD width (default: host)\n"
              << "  --blend=opaque|alpha     Blend mode for graphics PSO\n"
              << "  --depth=none|less        Depth mode for graphics PSO\n"
              << "  --emit-scf               Print lifted SCF MLIR\n"
              << "  --emit-simd-mlir         Print packetized SIMD MLIR\n"
              << "  --emit-llvm              Print linked LLVM IR\n";
}

} // namespace

int main(int argc, char** argv) {
    llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
    std::string csPath;
    std::string vsPath;
    std::string psPath;
    uint32_t simdWidth = 0;
    bool blendAlpha = false;
    bool depthLess = false;
    bool emitSCF = false;
    bool emitSIMD = false;
    bool emitLLVM = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto nextVal = [&](const std::string& flag) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << flag << "\n";
                std::exit(1);
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--cs") {
            csPath = nextVal(arg);
        } else if (arg == "--vs") {
            vsPath = nextVal(arg);
        } else if (arg == "--ps") {
            psPath = nextVal(arg);
        } else if (arg.rfind("--simd-width=", 0) == 0) {
            simdWidth = static_cast<uint32_t>(std::stoul(arg.substr(13)));
        } else if (arg == "--simd-width") {
            simdWidth = static_cast<uint32_t>(std::stoul(nextVal(arg)));
        } else if (arg.rfind("--blend=", 0) == 0) {
            blendAlpha = (arg.substr(8) == "alpha");
        } else if (arg.rfind("--depth=", 0) == 0) {
            depthLess = (arg.substr(8) == "less");
        } else if (arg == "--emit-scf") {
            emitSCF = true;
        } else if (arg == "--emit-simd-mlir") {
            emitSIMD = true;
        } else if (arg == "--emit-llvm") {
            emitLLVM = true;
        } else if (!arg.empty() && arg[0] != '-' && csPath.empty()) {
            csPath = arg;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    if (!emitSCF && !emitSIMD && !emitLLVM) {
        emitSIMD = true;
    }

    if (simdWidth != 0) {
        nacelle::setSimdWidthOverride(simdWidth);
    }
    uint32_t effectiveW = nacelle::getTargetSimdWidth();

    mlir::MLIRContext mlirCtx;
    nacelle::configureNacelleMLIRContext(mlirCtx);

    if (!vsPath.empty()) {
        std::vector<uint8_t> vsBytes, psBytes;
        if (!readBinaryFile(vsPath, vsBytes)) {
            std::cerr << "Failed to read VS DXIL: " << vsPath << "\n";
            return 1;
        }
        if (!psPath.empty() && !readBinaryFile(psPath, psBytes)) {
            std::cerr << "Failed to read PS DXIL: " << psPath << "\n";
            return 1;
        }

        if (emitSCF) {
            llvm::LLVMContext lctx;
            nacelle::DxilLoadedShader loaded;
            std::string err;
            if (!nacelle::loadAndUpgradeDXIL({vsBytes.data(), vsBytes.size()}, lctx, loaded, err)) {
                std::cerr << "VS DXIL load error: " << err << "\n";
                return 1;
            }
            auto scfMod = nacelle::importAndLiftDXILToSCF(std::move(loaded.module), loaded.entryName, mlirCtx, err);
            if (!scfMod) {
                std::cerr << "VS SCF lift error: " << err << "\n";
                return 1;
            }
            scfMod->print(llvm::outs());
            llvm::outs() << "\n";
        }

        if (emitSIMD || emitLLVM) {
            D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{};
            desc.VS = {vsBytes.data(), vsBytes.size()};
            if (!psBytes.empty()) {
                desc.PS = {psBytes.data(), psBytes.size()};
            }
            desc.NumRenderTargets = 1;
            desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
            desc.SampleMask = UINT32_MAX;
            desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
            desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
            if (blendAlpha) {
                desc.BlendState.RenderTarget[0].BlendEnable = TRUE;
                desc.BlendState.RenderTarget[0].SrcBlend = D3D12_BLEND_SRC_ALPHA;
                desc.BlendState.RenderTarget[0].DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
                desc.BlendState.RenderTarget[0].BlendOp = D3D12_BLEND_OP_ADD;
                desc.BlendState.RenderTarget[0].SrcBlendAlpha = D3D12_BLEND_ONE;
                desc.BlendState.RenderTarget[0].DestBlendAlpha = D3D12_BLEND_ZERO;
                desc.BlendState.RenderTarget[0].BlendOpAlpha = D3D12_BLEND_OP_ADD;
            }
            desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
            if (depthLess) {
                desc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
                desc.DepthStencilState.DepthEnable = TRUE;
                desc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
                desc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
            }

            nacelle::RootSignatureLayout rootLayout{};
            mlir::OwningOpRef<mlir::ModuleOp> combinedMod;
            nacelle::CachedPSOMeta meta{};
            std::vector<nacelle::ShaderSigElement> vsIn, vsOut, psIn, psOut;
            std::string err;
            if (!nacelle::synthesizeGraphicsPSOMLIR(
                    desc, rootLayout, effectiveW,
                    "nacelle_vs_test", "nacelle_ps_test",
                    mlirCtx, combinedMod, meta, vsIn, vsOut, psIn, psOut, err)) {
                std::cerr << "Graphics PSO synthesis error: " << err << "\n";
                return 1;
            }
            if (emitSIMD) {
                combinedMod->print(llvm::outs());
                llvm::outs() << "\n";
            }
            if (emitLLVM) {
                llvm::LLVMContext lctx;
                auto llvmMod = nacelle::translateMLIRToLinkedLLVMIR(*combinedMod, lctx, err);
                if (!llvmMod) {
                    std::cerr << "LLVM IR translation error: " << err << "\n";
                    return 1;
                }
                llvmMod->print(llvm::outs(), nullptr);
            }
        }
        return 0;
    }

    if (csPath.empty()) {
        printUsage(argv[0]);
        return 1;
    }

    std::vector<uint8_t> csBytes;
    if (!readBinaryFile(csPath, csBytes)) {
        std::cerr << "Failed to read CS DXIL: " << csPath << "\n";
        return 1;
    }

    llvm::LLVMContext lctx;
    nacelle::DxilLoadedShader loaded;
    std::string err;
    if (!nacelle::loadAndUpgradeDXIL({csBytes.data(), csBytes.size()}, lctx, loaded, err)) {
        std::cerr << "CS DXIL load error: " << err << "\n";
        return 1;
    }

    auto scfMod = nacelle::importAndLiftDXILToSCF(std::move(loaded.module), loaded.entryName, mlirCtx, err);
    if (!scfMod) {
        std::cerr << "CS SCF lift error: " << err << "\n";
        return 1;
    }

    if (emitSCF) {
        scfMod->print(llvm::outs());
        llvm::outs() << "\n";
    }

    if (emitSIMD || emitLLVM) {
        nacelle::RootSignatureLayout layout = loaded.hasEmbeddedRootSig
                                                 ? loaded.embeddedRootSig
                                                 : nacelle::RootSignatureLayout{};
        if (!nacelle::packetizeComputeEntry(*scfMod, loaded, layout, effectiveW, "nacelle_cs_test", err)) {
            std::cerr << "Compute packetizer error: " << err << "\n";
            return 1;
        }
        if (emitSIMD) {
            scfMod->print(llvm::outs());
            llvm::outs() << "\n";
        }
        if (emitLLVM) {
            llvm::LLVMContext outCtx;
            auto llvmMod = nacelle::translateMLIRToLinkedLLVMIR(*scfMod, outCtx, err);
            if (!llvmMod) {
                std::cerr << "LLVM IR translation error: " << err << "\n";
                return 1;
            }
            llvmMod->print(llvm::outs(), nullptr);
        }
    }

    return 0;
}
