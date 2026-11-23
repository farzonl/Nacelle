#include "profile_artifacts.h"
#include "../instrumentation/artifacts.h"
#include "dxil_loader.h"
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Module.h>
#include <llvm/MC/MCAsmInfo.h>
#include <llvm/MC/MCContext.h>
#include <llvm/MC/MCDisassembler/MCDisassembler.h>
#include <llvm/MC/MCInstPrinter.h>
#include <llvm/MC/MCInstrInfo.h>
#include <llvm/MC/MCRegisterInfo.h>
#include <llvm/MC/MCSubtargetInfo.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Object/DXContainer.h>
#include <llvm/Object/ObjectFile.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/ADT/StringExtras.h>
#include <mlir/IR/BuiltinOps.h>
#include <algorithm>
#include <map>

namespace nacelle {
namespace {
class BoundedText : public llvm::raw_ostream {
public:
    explicit BoundedText(uint64_t limit) : limit_(limit) { SetUnbuffered(); }
    std::string text;
    bool overflow = false;
private:
    void write_impl(const char* bytes, size_t size) override {
        if (size > limit_ - text.size()) { overflow = true; return; }
        text.append(bytes, size);
    }
    uint64_t current_pos() const override { return text.size(); }
    uint64_t limit_;
};

void sanitizeModule(llvm::Module& module) {
    llvm::StripDebugInfo(module);
    module.setModuleIdentifier("captured-shader");
    module.setSourceFileName("captured-shader");
    for (auto it = module.named_metadata_begin(); it != module.named_metadata_end();) {
        auto* metadata = &*it++;
        if (metadata->getName().starts_with("dx.source")) module.eraseNamedMetadata(metadata);
    }
}
}

void retainProfileShader(const D3D12_SHADER_BYTECODE& bytecode, const std::string& identity) {
    const auto store = profile::artifactStore();
    if (!store || !bytecode.pShaderBytecode || !bytecode.BytecodeLength) return;
    {
        std::lock_guard lock(store->mutex);
        if (store->groups.contains(identity)) return;
    }
    if (bytecode.BytecodeLength > store->options.artifact_bytes) {
        profile::retainArtifacts(store, identity, {}, true);
        return;
    }
    llvm::StringRef bitcode(static_cast<const char*>(bytecode.pShaderBytecode), bytecode.BytecodeLength);
    if (bitcode.starts_with("DXBC")) {
        auto container = llvm::object::DXContainer::create(llvm::MemoryBufferRef(bitcode, "captured-shader"));
        if (!container) {
            std::fprintf(stderr, "[NacelleProfile] Artifact container: %s\n", llvm::toString(container.takeError()).c_str());
            return;
        }
        auto part = container->getDXIL(true);
        if (!part) part = container->getDXIL(false);
        if (!part) return;
        bitcode = llvm::StringRef(part->second, part->first.Bitcode.Size);
    }
    llvm::LLVMContext context;
    llvm::ParserCallbacks callbacks;
    callbacks.DataLayout = [](llvm::StringRef, llvm::StringRef) -> std::optional<std::string> {
        return getHostDataLayoutString();
    };
    auto module = llvm::parseBitcodeFile(llvm::MemoryBufferRef(bitcode, "captured-shader"), context, callbacks);
    if (!module) {
        std::fprintf(stderr, "[NacelleProfile] Artifact bitcode unavailable: %s\n", llvm::toString(module.takeError()).c_str());
        return;
    }
    std::vector<profile::TextArtifact> payloads;
    std::map<std::string, std::string> sourceIds;
    std::map<std::string, uint32_t> sourceLines;
    if (store->options.flags & NACELLE_PROFILE_EMBED_SHADER_SOURCES) {
        if (const auto* sources = (*module)->getNamedMetadata("dx.source.contents")) {
            uint64_t bytes = 0;
            for (const auto* source : sources->operands()) {
                if (source->getNumOperands() < 2) continue;
                const auto* name = llvm::dyn_cast_or_null<llvm::MDString>(source->getOperand(0));
                const auto* content = llvm::dyn_cast_or_null<llvm::MDString>(source->getOperand(1));
                if (!name || !content) continue;
                bytes += content->getString().size();
                if (bytes > store->options.source_bytes || payloads.size() == 16) {
                    profile::retainArtifacts(store, identity, {}, true);
                    return;
                }
                profile::TextArtifact text;
                text.kind = "hlsl";
                text.label = "source-" + std::to_string(payloads.size() + 1) + ".hlsl";
                text.text = content->getString().str();
                const auto sourceId = identity + "-hlsl-" + profile::artifactDigest(text.text);
                sourceIds.emplace(name->getString().str(), sourceId);
                sourceLines.emplace(sourceId, 1 + std::count(text.text.begin(), text.text.end(), '\n'));
                payloads.push_back(std::move(text));
            }
        }
    }
    std::map<const llvm::Instruction*, std::pair<std::string, unsigned>> locations;
    for (auto& function : **module) {
        for (auto& instruction : llvm::instructions(function)) {
            if (const auto location = instruction.getDebugLoc()) {
                const auto* file = location->getFile();
                auto found = sourceIds.find(file->getFilename().str());
                if (found == sourceIds.end())
                    found = sourceIds.find((file->getDirectory() + "/" + file->getFilename()).str());
                if (found != sourceIds.end() && location.getLine() > 0 &&
                    location.getLine() <= sourceLines.at(found->second))
                    locations.emplace(&instruction, std::make_pair(found->second, location.getLine()));
            }
        }
    }
    sanitizeModule(**module);
    profile::TextArtifact dxil;
    dxil.kind = "dxil";
    dxil.label = "shader.dxil";
    BoundedText output(store->options.artifact_bytes);
    uint32_t line = 1;
    for (auto& function : **module) {
        if (function.isDeclaration()) continue;
        output << "; function ";
        llvm::printEscapedString(function.getName(), output);
        output << "\n";
        ++line;
        for (auto& instruction : llvm::instructions(function)) {
            const auto start = output.text.size();
            instruction.print(output);
            output << "\n";
            if (const auto found = locations.find(&instruction); found != locations.end())
                dxil.mappings.push_back({line, found->second.second, found->second.first});
            line += std::count(output.text.begin() + start, output.text.end(), '\n');
            if (output.overflow || dxil.mappings.size() > 65536) break;
        }
        if (output.overflow) break;
    }
    dxil.text = std::move(output.text);
    payloads.push_back(std::move(dxil));
    profile::retainArtifacts(store, identity, std::move(payloads), output.overflow);
}

void retainProfileLLVM(llvm::Module& module, const std::string& identity) {
    const auto store = profile::artifactStore();
    if (!store) return;
    auto copy = llvm::CloneModule(module);
    sanitizeModule(*copy);
    BoundedText output(store->options.artifact_bytes);
    copy->print(output, nullptr);
    profile::TextArtifact text;
    text.kind = "llvm";
    text.label = "optimized.ll";
    text.text = std::move(output.text);
    profile::retainArtifacts(store, identity, {std::move(text)}, output.overflow, true);
}

void retainProfileMLIR(mlir::ModuleOp module, const std::string& identity) {
    const auto store = profile::artifactStore();
    if (!store || identity.size() != 64) return;
    BoundedText output(store->options.artifact_bytes);
    module.print(output, mlir::OpPrintingFlags());
    profile::TextArtifact text;
    text.kind = "mlir";
    text.label = "shader.mlir";
    text.text = std::move(output.text);
    profile::retainArtifacts(store, identity, {std::move(text)}, output.overflow, true);
}

void retainProfileObject(const void* bytes, size_t size, const std::string& identity) {
    const auto store = profile::artifactStore();
    if (!store) return;
    {
        std::lock_guard lock(store->mutex);
        if (store->groups.contains(identity)) return;
    }
    if (size > store->options.artifact_bytes) {
        profile::retainArtifacts(store, identity, {}, true);
        return;
    }
    auto object = llvm::object::ObjectFile::createObjectFile(llvm::MemoryBufferRef(
        llvm::StringRef(static_cast<const char*>(bytes), size), "captured-object"));
    if (!object) {
        std::fprintf(stderr, "[NacelleProfile] Object unavailable: %s\n", llvm::toString(object.takeError()).c_str());
        return;
    }
    static std::once_flag disassemblerInitialization;
    std::call_once(disassemblerInitialization, [] { llvm::InitializeNativeTargetDisassembler(); });
    const auto triple = (*object)->makeTriple();
    std::string error;
    const auto* target = llvm::TargetRegistry::lookupTarget(triple, error);
    if (!target) { std::fprintf(stderr, "[NacelleProfile] %s\n", error.c_str()); return; }
    std::unique_ptr<llvm::MCRegisterInfo> registers(target->createMCRegInfo(triple));
    std::unique_ptr<llvm::MCInstrInfo> instructions(target->createMCInstrInfo());
    std::unique_ptr<llvm::MCSubtargetInfo> subtarget(target->createMCSubtargetInfo(triple, "", ""));
    llvm::MCTargetOptions options;
    std::unique_ptr<llvm::MCAsmInfo> assembly(target->createMCAsmInfo(*registers, triple, options));
    llvm::MCContext context(triple, *assembly, *registers, *subtarget);
    std::unique_ptr<llvm::MCDisassembler> disassembler(target->createMCDisassembler(*subtarget, context));
    std::unique_ptr<llvm::MCInstPrinter> printer(target->createMCInstPrinter(triple, 0, *assembly, *instructions, *registers));
    if (!disassembler || !printer) {
        std::fprintf(stderr, "[NacelleProfile] Native disassembler unavailable\n");
        return;
    }
    std::vector<profile::TextArtifact> payloads;
    bool overflow = false;
    for (const auto& section : (*object)->sections()) {
        if (!section.isText()) continue;
        if (payloads.size() == 16) { overflow = true; break; }
        auto contents = section.getContents();
        if (!contents) {
            std::fprintf(stderr, "[NacelleProfile] Object section: %s\n", llvm::toString(contents.takeError()).c_str());
            return;
        }
        if (contents->empty()) continue;
        profile::TextArtifact text;
        text.kind = "assembly";
        text.label = "text-" + std::to_string(payloads.size() + 1) + ".asm";
        BoundedText output(store->options.artifact_bytes);
        auto data = llvm::ArrayRef(reinterpret_cast<const uint8_t*>(contents->data()), contents->size());
        uint32_t line = 1;
        for (uint64_t offset = 0; offset < data.size();) {
            llvm::MCInst instruction;
            uint64_t length = 0;
            const auto decoded = disassembler->getInstruction(instruction, length, data.slice(offset), offset, llvm::nulls());
            if (!length) length = 1;
            output << offset << ": ";
            if (decoded == llvm::MCDisassembler::Fail) output << "<unmapped byte>";
            else printer->printInst(&instruction, offset + length, "", *subtarget, output);
            output << "\n";
            text.mappings.push_back({line++, 0, {}, offset, static_cast<uint32_t>(length)});
            offset += length;
            if (output.overflow || text.mappings.size() > 65536) { overflow = true; break; }
        }
        text.text = std::move(output.text);
        payloads.push_back(std::move(text));
        if (overflow) break;
    }
    profile::retainArtifacts(store, identity, std::move(payloads), overflow);
}
}
