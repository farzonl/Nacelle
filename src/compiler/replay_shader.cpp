#include "../instrumentation/replay_capture.h"
#include "dxil_loader.h"
#include "jit_engine.h"
#include "native_debug.h"
#include "../common/fp_env.h"
#include <llvm/ADT/SmallVector.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Object/DXContainer.h>
#include <llvm/TargetParser/Host.h>
#include <llvm/Support/MemoryBuffer.h>
#include <algorithm>
#include <bit>
#include <cfenv>
#include <cstdlib>
#include <tuple>
#include <array>
#include <map>
#include "../common/format_utils.h"

namespace nacelle::replay {
Environment currentEnvironment() {
    Environment result;
    result.build.compiler = NACELLE_PROFILE_COMPILER;
    result.build.operatingSystem = NACELLE_PROFILE_OS;
    result.build.architecture = NACELLE_PROFILE_ARCH;
    result.build.revision = NACELLE_PROFILE_REVISION;
    result.build.compilerHash = getCompilerBuildHash();
    result.build.targetTriple = getHostTargetTripleString();
    result.build.cpu = llvm::sys::getHostCPUName().str();
    for (const auto& feature : llvm::sys::getHostCPUFeatures())
        if (feature.second) result.build.cpuFeatures.push_back(feature.first().str());
    std::sort(result.build.cpuFeatures.begin(), result.build.cpuFeatures.end());
    result.simdWidth = getTargetSimdWidth();
    result.pointerBits = sizeof(void*) * 8;
    result.endian = std::endian::native == std::endian::little ? "little" : "unsupported";
    result.rounding = static_cast<uint32_t>(std::fegetround());
    if (result.rounding != FE_TONEAREST && result.rounding != FE_DOWNWARD &&
        result.rounding != FE_UPWARD && result.rounding != FE_TOWARDZERO)
        result.fpPolicy = "unsupported";
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    result.fpControl = _mm_getcsr() & ~0x3fu;
#elif defined(__aarch64__) && (defined(__clang__) || defined(__GNUC__))
    uint64_t control = 0;
    __asm__ volatile("mrs %0, fpcr" : "=r"(control));
    result.fpControl = static_cast<uint32_t>(control);
#else
    result.fpPolicy = "unsupported";
#endif
    for (const auto* name : {"NACELLE_STRICT_FP", "NACELLE_DISABLE_TIER2", "NACELLE_GEMM_BACKEND",
                              "NACELLE_JIT_PROFILING", "NACELLE_DISABLE_GEMM_FASTPATH"}) {
        const char* value = std::getenv(name);
        if (std::string_view(name) == "NACELLE_GEMM_BACKEND") {
            const std::string_view backend = value ? value : "auto";
            result.compilerOptions.push_back(std::string(name) + "=" +
                (backend == "auto" || backend == "fallback" || backend == "accelerate" || backend == "openblas"
                    ? std::string(backend) : "unsupported"));
        } else result.compilerOptions.push_back(std::string(name) + "=" + (value && value[0] == '1' ? "1" : "0"));
    }

    if (native_debug::enabled()) result.compilerOptions.push_back("native-debug-pinned-v1");
    return result;
}

bool sameEnvironment(const Environment& a, const Environment& b) {
    const auto fields = [](const Environment& value) {
        return std::tie(value.build.compiler, value.build.operatingSystem, value.build.architecture,
            value.build.revision, value.build.compilerHash, value.build.targetTriple, value.build.cpu,
            value.build.cpuFeatures, value.simdWidth, value.pointerBits, value.rounding, value.fpControl,
            value.endian, value.fpPolicy, value.compilerOptions);
    };
    return fields(a) == fields(b);
}

bool validateShader(std::span<const char> bytes, bool sourceConsent, std::string& error) {
    llvm::StringRef bitcode(bytes.data(), bytes.size());
    std::optional<llvm::object::DXContainer> owner;
    if (bitcode.starts_with("DXBC")) {
        auto container = llvm::object::DXContainer::create(llvm::MemoryBufferRef(bitcode, "replay-shader"));
        if (!container) { error = llvm::toString(container.takeError()); return false; }
        owner.emplace(std::move(*container));
        auto part = owner->getDXIL(true);
        if (!part) part = owner->getDXIL(false);
        if (!part) { error = "Replay shader has no DXIL"; return false; }
        bitcode = llvm::StringRef(part->second, part->first.Bitcode.Size);
    }
    llvm::LLVMContext context;
    llvm::ParserCallbacks callbacks;
    callbacks.DataLayout = [](llvm::StringRef, llvm::StringRef) -> std::optional<std::string> {
        return getHostDataLayoutString();
    };
    auto original = llvm::parseBitcodeFile(llvm::MemoryBufferRef(bitcode, "replay-shader"), context, callbacks);
    if (!original) { error = llvm::toString(original.takeError()); return false; }
    if (!sourceConsent && ((*original)->getNamedMetadata("dx.source.contents") ||
        (*original)->getNamedMetadata("dx.source.mainFileName") || (*original)->getNamedMetadata("llvm.dbg.cu"))) {
        error = "Replay shader embeds source without source consent";
        return false;
    }
    DxilLoadedShader shader;
    if (!loadAndUpgradeDXIL({bytes.data(), bytes.size()}, context, shader, error)) return false;
    if (shader.stage != "cs") { error = "Replay supports compute shaders only"; return false; }
    if (shader.groupSharedBytes > 32768 ||
        std::any_of(std::begin(shader.threadGroupSize), std::end(shader.threadGroupSize),
            [](uint32_t value) { return !value || value > 1024; }) ||
        uint64_t(shader.threadGroupSize[0]) * shader.threadGroupSize[1] * shader.threadGroupSize[2] > 1024) {
        error = "Replay compute threadgroup/shared-memory dimensions exceed supported bounds";
        return false;
    }
    for (const auto& resource : shader.resources)
        if (resource.rangeSize == UINT32_MAX || resource.resClass == 3 ||
            (resource.resourceKind != 10 && resource.resourceKind != 11 &&
             resource.resourceKind != 12 && resource.resourceKind != 13)) {
            error = "Replay shader uses unsupported resource kind or unbounded binding";
            return false;
        }
    for (const auto& global : shader.module->globals())
        if (global.isDeclaration() && std::none_of(shader.resources.begin(), shader.resources.end(),
            [&](const DxilResourceRecord& resource) { return resource.symbolName == global.getName(); })) {
            error = "Replay unsupported external shader global";
            return false;
        }
    for (const auto& function : *shader.module)
        for (const auto& instruction : llvm::instructions(function)) {
            if (llvm::isa<llvm::AtomicRMWInst>(instruction) || llvm::isa<llvm::AtomicCmpXchgInst>(instruction) ||
                (llvm::isa<llvm::LoadInst>(instruction) && llvm::cast<llvm::LoadInst>(instruction).isAtomic()) ||
                (llvm::isa<llvm::StoreInst>(instruction) && llvm::cast<llvm::StoreInst>(instruction).isAtomic())) {
                error = "Replay shader atomics are unsupported";
                return false;
            }
            const auto* call = llvm::dyn_cast<llvm::CallBase>(&instruction);
            if (!call) continue;
            const auto* target = call->getCalledFunction();
            if (!target) { error = "Replay indirect shader calls are unsupported"; return false; }
            if (target->getName() == "nacelle.groupshared.ptr" &&
                !(*original)->getFunction("nacelle.groupshared.ptr") && call->arg_size() == 1) {
                const auto* offset = llvm::dyn_cast<llvm::ConstantInt>(call->getArgOperand(0));
                if (offset && offset->getZExtValue() < shader.groupSharedBytes) continue;
            }
            if (target->getName().starts_with("dx.op.")) {
                const auto* opcode = call->arg_empty() ? nullptr : llvm::dyn_cast<llvm::ConstantInt>(call->getArgOperand(0));
                if (!opcode) { error = "Replay DXIL opcode is not constant"; return false; }
                const uint64_t op = opcode->getZExtValue();
                if ((op >= 6 && op <= 59) || op == 68 || op == 69 || (op >= 93 && op <= 96) ||
                    (op >= 110 && op <= 123) || op == 139 || op == 140 || op == 146 || op == 147 || op == 216 || op == 217)
                    continue;
                if (op == 80 && call->arg_size() > 1) {
                    const auto* flags = llvm::dyn_cast<llvm::ConstantInt>(call->getArgOperand(1));
                    if (flags && !(flags->getZExtValue() & ~9ull)) continue;
                }
                error = "Replay unsupported or nondeterministic DXIL operation: " + std::to_string(op);
                return false;
            }
            if (target->isDeclaration()) {
                switch (target->getIntrinsicID()) {
                    case llvm::Intrinsic::fabs: case llvm::Intrinsic::fma: case llvm::Intrinsic::fmuladd:
                    case llvm::Intrinsic::sqrt: case llvm::Intrinsic::pow: case llvm::Intrinsic::powi:
                    case llvm::Intrinsic::sin: case llvm::Intrinsic::cos: case llvm::Intrinsic::exp:
                    case llvm::Intrinsic::exp2: case llvm::Intrinsic::log: case llvm::Intrinsic::log2:
                    case llvm::Intrinsic::log10: case llvm::Intrinsic::ceil: case llvm::Intrinsic::floor:
                    case llvm::Intrinsic::trunc: case llvm::Intrinsic::rint: case llvm::Intrinsic::round:
                    case llvm::Intrinsic::roundeven: case llvm::Intrinsic::nearbyint: case llvm::Intrinsic::minnum:
                    case llvm::Intrinsic::maxnum: case llvm::Intrinsic::minimum: case llvm::Intrinsic::maximum:
                    case llvm::Intrinsic::copysign: case llvm::Intrinsic::ctpop: case llvm::Intrinsic::ctlz:
                    case llvm::Intrinsic::cttz: case llvm::Intrinsic::bitreverse: case llvm::Intrinsic::bswap:
                    case llvm::Intrinsic::fshl: case llvm::Intrinsic::fshr:
                    case llvm::Intrinsic::sadd_with_overflow: case llvm::Intrinsic::uadd_with_overflow:
                    case llvm::Intrinsic::ssub_with_overflow: case llvm::Intrinsic::usub_with_overflow:
                    case llvm::Intrinsic::smul_with_overflow: case llvm::Intrinsic::umul_with_overflow:
                    case llvm::Intrinsic::lifetime_start: case llvm::Intrinsic::lifetime_end:
                    case llvm::Intrinsic::assume: case llvm::Intrinsic::memcpy: case llvm::Intrinsic::memmove:
                    case llvm::Intrinsic::memset: break;
                    default:
                        error = "Replay unsupported external or nondeterministic LLVM intrinsic: " +
                            target->getName().str();
                        return false;
                }
            }
        }
    return true;
}

bool validateSemantics(const Bundle& bundle, std::string& error) {
    const auto fail = [&](const char* message) { error = message; return false; };
    if (!validateBundle(bundle, error)) return false;
    const auto& metadata = bundle.metadata;
    std::vector<RootSignatureLayout> roots(metadata.roots.size());
    for (const auto& root : metadata.roots) {
        const auto bytes = blobBytes(bundle, root.blob);
        auto& layout = roots[root.id - 1];
        if (FAILED(NacelleRootSignature::parseBlob(bytes.data(), bytes.size(), layout)) ||
            layout.params.size() > 64 || !layout.staticSamplers.empty())
            return fail("Unsupported replay root signature");
        for (const auto& parameter : layout.params) {
            if (parameter.visibility != D3D12_SHADER_VISIBILITY_ALL ||
                parameter.type > D3D12_ROOT_PARAMETER_TYPE_UAV ||
                parameter.num32BitValues > 256 || parameter.constDwordOffset > 256 - parameter.num32BitValues)
                return fail("Unsupported replay root parameter");
            for (const auto& range : parameter.ranges)
                if (!range.numDescriptors || range.numDescriptors > maxCommands ||
                    range.offsetInTable > maxCommands - range.numDescriptors ||
                    range.type == D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER)
                    return fail("Unsupported replay descriptor range");
        }
    }
    std::vector<std::vector<DxilResourceRecord>> resources(metadata.pipelines.size());
    for (const auto& pipeline : metadata.pipelines) {
        const auto bytes = blobBytes(bundle, pipeline.shaderBlob);
        if (!validateShader(bytes, bundle.manifest.sourceConsent, error)) return false;
        llvm::LLVMContext context;
        DxilLoadedShader shader;
        if (!loadAndUpgradeDXIL({bytes.data(), bytes.size()}, context, shader, error)) return false;
        resources[pipeline.id - 1] = shader.resources;
    }
    std::map<std::pair<uint32_t, uint32_t>, const View*> views;
    for (const auto& view : metadata.views) {
        views.emplace(std::make_pair(view.heap, view.slot), &view);
        const auto& resource = metadata.resources[view.resource - 1];
        if ((view.kind == "srv" || view.kind == "cbv") && !(resource.access & 1))
            return fail("Replay read descriptor lacks declared read access");
        if (view.kind == "uav" && !(resource.flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS))
            return fail("Replay UAV resource lacks UAV creation flags");
        if (view.kind == "cbv") {
            if (view.first % 256 || view.bytes % 256 || view.bytes > 65536)
                return fail("Replay CBV alignment/size is invalid");
        } else if (view.flags) {
            if (view.format != DXGI_FORMAT_R32_TYPELESS || view.stride != 4)
                return fail("Replay raw buffer format/stride is invalid");
        } else if (view.format == DXGI_FORMAT_UNKNOWN) {
            if (view.stride % 4 || view.stride > 2048)
                return fail("Replay structured buffer stride is invalid");
        } else {
            switch (static_cast<DXGI_FORMAT>(view.format)) {
                case DXGI_FORMAT_R32_FLOAT: case DXGI_FORMAT_R32_UINT: case DXGI_FORMAT_R32_SINT:
                case DXGI_FORMAT_R32G32_FLOAT: case DXGI_FORMAT_R32G32_UINT: case DXGI_FORMAT_R32G32_SINT:
                case DXGI_FORMAT_R32G32B32_FLOAT: case DXGI_FORMAT_R32G32B32_UINT: case DXGI_FORMAT_R32G32B32_SINT:
                case DXGI_FORMAT_R32G32B32A32_FLOAT: case DXGI_FORMAT_R32G32B32A32_UINT:
                case DXGI_FORMAT_R32G32B32A32_SINT: break;
                default: return fail("Replay supports only 32-bit-component typed buffer formats");
            }
            const auto format = getFormatInfo(static_cast<DXGI_FORMAT>(view.format));
            if (!format.bytesPerBlock || format.blockWidth != 1 || format.blockHeight != 1 ||
                format.bytesPerBlock != view.stride)
                return fail("Replay typed buffer format/stride is unsupported");
        }
    }
    struct Binding { uint32_t resource = 0, heap = 0, slot = 0; uint64_t offset = 0; bool constants = false; };
    std::array<Binding, 64> bindings{};
    uint32_t root = 0, pipeline = 0;
    for (const auto& command : metadata.commands) {
        const auto& words = command.data;
        if (command.kind == "submission") { root = pipeline = 0; bindings = {}; }
        else if (command.kind == "root") { root = words[0]; bindings = {}; }
        else if (command.kind == "pipeline") pipeline = words[0];
        else if (command.kind == "view" || command.kind == "table" || command.kind == "constants") {
            if (!root || words[0] >= roots[root - 1].params.size())
                return fail("Replay binding root index is invalid");
            const auto& parameter = roots[root - 1].params[words[0]];
            auto& binding = bindings[words[0]];
            if (command.kind == "table") {
                if (parameter.type != D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE)
                    return fail("Replay table uses a non-table root parameter");
                binding = {0, static_cast<uint32_t>(words[1]), static_cast<uint32_t>(words[2])};
                for (const auto& range : parameter.ranges)
                    if (range.offsetInTable + range.numDescriptors >
                        metadata.heaps[binding.heap - 1].count - binding.slot)
                        return fail("Replay table range exceeds heap bounds");
            } else if (command.kind == "constants") {
                if (parameter.type != D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS ||
                    words[1] > parameter.num32BitValues || words.size() - 2 > parameter.num32BitValues - words[1])
                    return fail("Replay constants exceed the root parameter");
                binding.constants = true;
            } else {
                if (parameter.type < D3D12_ROOT_PARAMETER_TYPE_CBV || parameter.type > D3D12_ROOT_PARAMETER_TYPE_UAV)
                    return fail("Replay view uses a non-descriptor root parameter");
                const auto& resource = metadata.resources[words[1] - 1];
                const uint32_t access = parameter.type == D3D12_ROOT_PARAMETER_TYPE_UAV ? 2 : 1;
                if (!(resource.access & access) ||
                    (access == 2 && !(resource.flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)) ||
                    words[2] % (parameter.type == D3D12_ROOT_PARAMETER_TYPE_CBV ? 256 : 4))
                    return fail("Replay root descriptor access/alignment is invalid");
                binding = {static_cast<uint32_t>(words[1]), 0, 0, words[2]};
            }
        } else if (command.kind == "clear") {
            const auto found = views.find({static_cast<uint32_t>(words[0]), static_cast<uint32_t>(words[1])});
            if (found == views.end() || found->second->kind != "uav" || found->second->resource != words[2] ||
                found->second->first || uint64_t(found->second->count) * found->second->stride !=
                    metadata.resources[words[2] - 1].size)
                return fail("Replay clear must match a writable UAV of the same resource");
        } else if (command.kind == "copy" && words[0] == words[2] && words[4] &&
                   words[1] < words[3] + words[4] && words[3] < words[1] + words[4])
            return fail("Replay overlapping same-buffer copies are unsupported");
        else if (command.kind == "dispatch") {
            const auto& layout = roots[root - 1];
            for (const auto& resource : resources[pipeline - 1]) {
                if (!resource.rangeSize || resource.rangeSize > maxCommands ||
                    resource.lowerBound > UINT32_MAX - resource.rangeSize)
                    return fail("Replay shader binding range is invalid");
                for (uint32_t index = 0; index < resource.rangeSize; ++index) {
                    const auto target = layout.resolveBinding(resource.resClass, resource.space, resource.lowerBound + index);
                    if (!target.kind || target.rootParamIndex >= layout.params.size())
                        return fail("Replay shader has an unresolved root binding");
                    const auto& binding = bindings[target.rootParamIndex];
                    if (target.kind == RootBindingTarget::RootConstants) {
                        if (!binding.constants || resource.resClass != 2 ||
                            resource.structuredStride > layout.params[target.rootParamIndex].num32BitValues * 4)
                            return fail("Replay shader constant binding is missing or undersized");
                    } else if (target.kind == RootBindingTarget::RootDescriptor) {
                        if (resource.resourceKind == 10 || !binding.resource ||
                            resource.structuredStride > metadata.resources[binding.resource - 1].size - binding.offset)
                            return fail("Replay shader root descriptor is missing or undersized");
                    } else if (target.kind == RootBindingTarget::DescriptorTableOffset) {
                        if (!binding.heap || target.tableOffsetOrConstStart >=
                            metadata.heaps[binding.heap - 1].count - binding.slot)
                            return fail("Replay shader descriptor table is missing or out of bounds");
                        const auto found = views.find({binding.heap, binding.slot + target.tableOffsetOrConstStart});
                        const auto* kind = resource.resClass == 0 ? "srv" : resource.resClass == 1 ? "uav" : "cbv";
                        if (found == views.end() || found->second->kind != kind ||
                            (resource.resourceKind == 10 && (!found->second->format || found->second->flags)) ||
                            (resource.resourceKind == 11 && !found->second->flags) ||
                            (resource.resourceKind == 12 && (found->second->format || found->second->flags ||
                             found->second->stride != resource.structuredStride)) ||
                            (resource.resClass == 2 && found->second->bytes < resource.structuredStride))
                            return fail("Replay shader/view kind or stride is incompatible");
                    } else return fail("Unsupported replay shader binding");
                }
            }
        }
    }
    return true;
}
}
