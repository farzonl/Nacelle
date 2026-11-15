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
#include "../src/compiler/dxil_loader.h"
#include "../src/compiler/jit_engine.h"
#include "../src/compiler/pso_synthesizer.h"
#include "../src/compiler/shader_compiler.h"
#include "../src/compiler/simt_to_simd_pass.h"
#include "../src/compiler/structured_cfg.h"

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
              << "  --ms <path>              Mesh DXIL shader (mesh PSO mode)\n"
              << "  --as <path>              Amplification DXIL shader (mesh PSO mode)\n"
              << "  --ps <path>              Pixel DXIL shader (graphics mode)\n"
              << "  --simd-width=4|8|16      Target SIMD width (default: host)\n"
              << "  --blend=opaque|alpha     Blend mode for graphics PSO\n"
              << "  --depth=none|less        Depth mode for graphics PSO\n"
              << "  --auto-input-layout      Synthesize D3D12_INPUT_LAYOUT_DESC from VS input signature\n"
              << "  --emit-scf               Print lifted SCF MLIR\n"
              << "  --emit-simd-mlir         Print packetized SIMD MLIR\n"
              << "  --emit-llvm              Print linked LLVM IR\n";
}

} // namespace

int main(int argc, char** argv) {
    llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
    std::string csPath;
    std::string vsPath;
    std::string msPath;
    std::string asPath;
    std::string psPath;
    uint32_t simdWidth = 0;
    bool blendAlpha = false;
    bool depthLess = false;
    bool autoInputLayout = false;
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
        } else if (arg == "--ms") {
            msPath = nextVal(arg);
        } else if (arg == "--as") {
            asPath = nextVal(arg);
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
        } else if (arg == "--auto-input-layout") {
            autoInputLayout = true;
        } else if (arg == "--emit-scf" || arg == "--emit-structured") {
            emitSCF = true;
        } else if (arg == "--emit-simd-mlir" || arg == "--emit-simd-ir") {
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

    if (!vsPath.empty() || !msPath.empty()) {
        std::vector<uint8_t> vsBytes, msBytes, asBytes, psBytes;
        if (!vsPath.empty() && !readBinaryFile(vsPath, vsBytes)) {
            std::cerr << "Failed to read VS DXIL: " << vsPath << "\n";
            return 1;
        }
        if (!msPath.empty() && !readBinaryFile(msPath, msBytes)) {
            std::cerr << "Failed to read MS DXIL: " << msPath << "\n";
            return 1;
        }
        if (!asPath.empty() && !readBinaryFile(asPath, asBytes)) {
            std::cerr << "Failed to read AS DXIL: " << asPath << "\n";
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
            const auto& primaryBytes = !msBytes.empty() ? msBytes : vsBytes;
            if (!nacelle::loadAndUpgradeDXIL({primaryBytes.data(), primaryBytes.size()}, lctx, loaded, err)) {
                std::cerr << "Graphics DXIL load error: " << err << "\n";
                return 1;
            }
            llvm::Function* entryFn = loaded.module->getFunction(loaded.entryName);
            if (!entryFn) {
                std::cerr << "Graphics entry function not found: " << loaded.entryName << "\n";
                return 1;
            }
            nacelle::StructuredFunction tree;
            if (!nacelle::structurizeAndBuildRegionTree(*entryFn, tree, err)) {
                std::cerr << "Graphics structured CFG error: " << err << "\n";
                return 1;
            }
            tree.print(llvm::outs());
            llvm::outs() << "\n";
        }

        if (emitSIMD || emitLLVM) {
            D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{};
            if (!vsBytes.empty()) {
                desc.VS = {vsBytes.data(), vsBytes.size()};
            }
            nacelle::MeshShaderStages meshStages{};
            if (!msBytes.empty()) {
                meshStages.MS = {msBytes.data(), msBytes.size()};
            }
            if (!asBytes.empty()) {
                meshStages.AS = {asBytes.data(), asBytes.size()};
            }
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

            std::vector<std::string> autoSemNames;
            std::vector<D3D12_INPUT_ELEMENT_DESC> autoElems;
            if (autoInputLayout) {
                llvm::LLVMContext sigCtx;
                nacelle::DxilLoadedShader vsSigLoaded;
                std::string sigErr;
                if (!nacelle::loadAndUpgradeDXIL({vsBytes.data(), vsBytes.size()}, sigCtx, vsSigLoaded, sigErr)) {
                    std::cerr << "VS DXIL load error: " << sigErr << "\n";
                    return 1;
                }
                autoSemNames.reserve(vsSigLoaded.inputSigs.size());
                autoElems.reserve(vsSigLoaded.inputSigs.size());
                for (const auto& sig : vsSigLoaded.inputSigs) {
                    if (sig.systemValueType != 0) continue;
                    autoSemNames.push_back(sig.semanticName);
                    DXGI_FORMAT fmt = DXGI_FORMAT_R32G32B32A32_FLOAT;
                    if (sig.componentType == 1) { // uint32
                        fmt = (sig.cols == 1) ? DXGI_FORMAT_R32_UINT
                              : (sig.cols == 2) ? DXGI_FORMAT_R32G32_UINT
                              : (sig.cols == 3) ? DXGI_FORMAT_R32G32B32_UINT
                                                : DXGI_FORMAT_R32G32B32A32_UINT;
                    } else if (sig.componentType == 2) { // sint32
                        fmt = (sig.cols == 1) ? DXGI_FORMAT_R32_SINT
                              : (sig.cols == 2) ? DXGI_FORMAT_R32G32_SINT
                              : (sig.cols == 3) ? DXGI_FORMAT_R32G32B32_SINT
                                                : DXGI_FORMAT_R32G32B32A32_SINT;
                    } else { // float32
                        fmt = (sig.cols == 1) ? DXGI_FORMAT_R32_FLOAT
                              : (sig.cols == 2) ? DXGI_FORMAT_R32G32_FLOAT
                              : (sig.cols == 3) ? DXGI_FORMAT_R32G32B32_FLOAT
                                                : DXGI_FORMAT_R32G32B32A32_FLOAT;
                    }
                    D3D12_INPUT_ELEMENT_DESC el{};
                    el.SemanticName = autoSemNames.back().c_str();
                    el.SemanticIndex = sig.semanticIndex;
                    el.Format = fmt;
                    el.InputSlot = 0;
                    el.AlignedByteOffset = D3D12_APPEND_ALIGNED_ELEMENT;
                    el.InputSlotClass = D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA;
                    el.InstanceDataStepRate = 0;
                    autoElems.push_back(el);
                }
                desc.InputLayout.pInputElementDescs = autoElems.data();
                desc.InputLayout.NumElements = static_cast<UINT>(autoElems.size());
            }

            nacelle::RootSignatureLayout rootLayout{};
            llvm::LLVMContext lctx;
            std::unique_ptr<llvm::Module> combinedMod;
            nacelle::CachedPSOMeta meta{};
            std::vector<nacelle::ShaderSigElement> vsIn, vsOut, psIn, psOut;
            std::string err;
            if (!nacelle::synthesizeGraphicsPSOLLVM(
                    desc, rootLayout, effectiveW,
                    "nacelle_vs_test", "nacelle_ps_test",
                    lctx, combinedMod, meta, vsIn, vsOut, psIn, psOut, err, meshStages)) {
                std::cerr << "Graphics PSO synthesis error: " << err << "\n";
                return 1;
            }
            if (emitSIMD) {
                combinedMod->print(llvm::outs(), nullptr);
                llvm::outs() << "\n";
            }
            if (emitLLVM) {
                if (!nacelle::linkAndFinalizeLLVMModule(*combinedMod, err)) {
                    std::cerr << "LLVM IR link error: " << err << "\n";
                    return 1;
                }
                combinedMod->print(llvm::outs(), nullptr);
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

    llvm::Function* entryFn = loaded.module->getFunction(loaded.entryName);
    if (!entryFn) {
        std::cerr << "CS entry function not found: " << loaded.entryName << "\n";
        return 1;
    }
    nacelle::StructuredFunction srcTree;
    if (!nacelle::structurizeAndBuildRegionTree(*entryFn, srcTree, err)) {
        std::cerr << "CS structured CFG error: " << err << "\n";
        return 1;
    }

    if (emitSCF) {
        srcTree.print(llvm::outs());
        llvm::outs() << "\n";
    }

    if (emitSIMD || emitLLVM) {
        nacelle::RootSignatureLayout layout = loaded.hasEmbeddedRootSig
                                                 ? loaded.embeddedRootSig
                                                 : nacelle::RootSignatureLayout{};
        if (!nacelle::packetizeComputeEntry(*loaded.module, srcTree, loaded, layout, effectiveW, "nacelle_cs_test", err)) {
            std::cerr << "Compute packetizer error: " << err << "\n";
            return 1;
        }
        entryFn->eraseFromParent();
        if (emitSIMD) {
            loaded.module->print(llvm::outs(), nullptr);
            llvm::outs() << "\n";
        }
        if (emitLLVM) {
            if (!nacelle::linkAndFinalizeLLVMModule(*loaded.module, err)) {
                std::cerr << "LLVM IR link error: " << err << "\n";
                return 1;
            }
            loaded.module->print(llvm::outs(), nullptr);
        }
    }

    return 0;
}
