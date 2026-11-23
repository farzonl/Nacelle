#include "source_debug.h"
#include "jit_engine.h"
#include <llvm/BinaryFormat/Dwarf.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/CFG.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/DebugProgramInstruction.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/MemoryBufferRef.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <algorithm>
#include <cstring>
#include <map>
#include <set>
#include <tuple>

namespace nacelle::source_debug {
namespace {
struct Binding {
    llvm::Value* value = nullptr;
    bool operator==(const Binding&) const = default;
};
using Values = std::map<llvm::DILocalVariable*, Binding>;

std::string normalized(llvm::StringRef path) {
    auto result = path.str();
    std::replace(result.begin(), result.end(), '\\', '/');
    while (result.starts_with("./")) result.erase(0, 2);
    return result;
}

bool inScope(const llvm::DILocalVariable& variable, const llvm::DILocation& location) {
    for (const llvm::DIScope* scope = location.getScope(); scope; scope = scope->getScope())
        if (scope == variable.getScope()) return true;
    return false;
}

void update(llvm::Instruction& instruction, Values& values) {
    for (auto& record : instruction.getDbgRecordRange()) {
        const auto* variable = llvm::dyn_cast<llvm::DbgVariableRecord>(&record);
        if (!variable) continue;
        auto* value = variable->isDbgValue() && variable->getNumVariableLocationOps() == 1 &&
            variable->getExpression()->getNumElements() == 0 ? variable->getValue() : nullptr;
        if (value && (llvm::isa<llvm::UndefValue, llvm::PoisonValue>(value) ||
                      !(value->getType()->isIntegerTy() || value->getType()->isFloatingPointTy())))
            value = nullptr;
        values[variable->getVariable()] = {value};
    }
}

NacelleSourceDebugLocal describe(llvm::DILocalVariable& variable, llvm::Value* value) {
    NacelleSourceDebugLocal local;
    const auto name = variable.getName();
    if (name.size() >= sizeof(local.name)) {
        std::strcpy(local.name, "<name-unavailable>");
        std::strcpy(local.reason, "local name exceeds bound");
        return local;
    }
    std::memcpy(local.name, name.data(), name.size());
    const llvm::DIType* sourceType = variable.getType();
    for (unsigned depth = 0; depth < 16; ++depth) {
        const auto* derived = llvm::dyn_cast_or_null<llvm::DIDerivedType>(sourceType);
        if (!derived || (derived->getTag() != llvm::dwarf::DW_TAG_typedef &&
                         derived->getTag() != llvm::dwarf::DW_TAG_const_type &&
                         derived->getTag() != llvm::dwarf::DW_TAG_volatile_type)) break;
        sourceType = derived->getBaseType();
    }
    const auto* type = llvm::dyn_cast_or_null<llvm::DIBasicType>(sourceType);
    if (!type || !value || type->getSizeInBits() > 64 ||
        type->getSizeInBits() != value->getType()->getScalarSizeInBits()) {
        std::strcpy(local.reason, "unsupported expression/type or no single reaching value");
        return local;
    }
    local.bits = type->getSizeInBits();
    switch (type->getEncoding()) {
    case llvm::dwarf::DW_ATE_boolean: local.kind = NACELLE_SOURCE_BOOL; break;
    case llvm::dwarf::DW_ATE_signed: local.kind = NACELLE_SOURCE_SIGNED; break;
    case llvm::dwarf::DW_ATE_unsigned: local.kind = NACELLE_SOURCE_UNSIGNED; break;
    case llvm::dwarf::DW_ATE_float: local.kind = NACELLE_SOURCE_FLOAT; break;
    default: std::strcpy(local.reason, "unsupported local encoding"); break;
    }
    return local;
}
}

bool sameProgram(const llvm::Module& executable, const llvm::Module& debug, std::string& error) {
    const auto canonical = [](const llvm::Module& input) -> llvm::Expected<std::string> {
        auto module = llvm::CloneModule(input);
        llvm::StripDebugInfo(*module);
        module->setModuleIdentifier("source-debug-program");
        module->setSourceFileName("source-debug-program");
        for (auto it = module->named_metadata_begin(); it != module->named_metadata_end();) {
            auto* metadata = &*it++;
            module->eraseNamedMetadata(metadata);
        }
        for (auto& function : *module) {
            function.setMetadata("guid", nullptr);
            for (auto& argument : function.args()) argument.setName("");
            for (auto& block : function) {
                block.setName("");
                for (auto& instruction : block) instruction.setName("");
            }
        }
        for (auto& global : module->globals()) global.setMetadata("guid", nullptr);
        llvm::SmallVector<char, 0> bitcode;
        llvm::raw_svector_ostream bitcodeOutput(bitcode);
        llvm::WriteBitcodeToFile(*module, bitcodeOutput);
        llvm::LLVMContext context;
        auto reparsed = llvm::parseBitcodeFile(
            llvm::MemoryBufferRef(llvm::StringRef(bitcode.data(), bitcode.size()), "canonical"), context);
        if (!reparsed) return reparsed.takeError();
        std::string text;
        llvm::raw_string_ostream output(text);
        (*reparsed)->print(output, nullptr);
        return text;
    };
    auto executableText = canonical(executable);
    if (!executableText) { error = llvm::toString(executableText.takeError()); return false; }
    auto debugText = canonical(debug);
    if (!debugText) { error = llvm::toString(debugText.takeError()); return false; }
    if (*executableText == *debugText) return true;
    error = "Embedded debug code does not exactly match executable DXIL; executable metadata remains authoritative";
    return false;
}

bool instrument(llvm::Module& module, const std::string& identity, uint32_t stage,
    const uint32_t* threadsPerGroup, std::string& error) {
    if (!enabled()) return true;
    const auto fail = [&](const char* reason) { error = reason; return false; };
    const auto* embedded = module.getNamedMetadata("dx.source.contents");
    if (!embedded || !module.getNamedMetadata("llvm.dbg.cu"))
        return fail("Source debugging requires unoptimized DXIL with embedded source/debug information");
    for (const auto* operand : module.getNamedMetadata("llvm.dbg.cu")->operands())
        if (const auto* unit = llvm::dyn_cast<llvm::DICompileUnit>(operand); !unit || unit->isOptimized())
            return fail("Optimized or invalid source debug information is unsupported");
    bool unoptimized = false;
    if (const auto* entries = module.getNamedMetadata("dx.entryPoints")) {
        for (const auto* entry : entries->operands()) {
            if (entry->getNumOperands() < 5) continue;
            const auto* properties = llvm::dyn_cast_or_null<llvm::MDNode>(entry->getOperand(4));
            if (!properties) continue;
            for (unsigned i = 0; i + 1 < properties->getNumOperands(); i += 2) {
                const auto* tag = llvm::mdconst::dyn_extract_or_null<llvm::ConstantInt>(properties->getOperand(i));
                const auto* flags = llvm::mdconst::dyn_extract_or_null<llvm::ConstantInt>(properties->getOperand(i + 1));
                if (tag && flags && tag->isZero()) unoptimized = (flags->getZExtValue() & 1) != 0;
            }
        }
    }
    if (!unoptimized) return fail("Source debugging requires DXC -Od; optimized shaders are unsupported");
    std::map<std::string, std::pair<std::string, uint32_t>> sources;
    for (const auto* operand : embedded->operands()) {
        if (operand->getNumOperands() != 2) return fail("Invalid embedded source record");
        const auto* name = llvm::dyn_cast<llvm::MDString>(operand->getOperand(0));
        const auto* text = llvm::dyn_cast<llvm::MDString>(operand->getOperand(1));
        if (!name || !text) return fail("Invalid embedded source strings");
        const auto id = profileContentIdentity(text->getString().data(), text->getString().size());
        if (!addSource(id, text->getString().str(), error)) return false;
        sources.emplace(normalized(name->getString()),
            std::make_pair(id, 1 + std::count(text->getString().begin(), text->getString().end(), '\n')));
    }
    uint32_t sites = 0;
    for (auto& function : module) {
        if (function.isDeclaration()) continue;
        function.convertToNewDbgValues();
        std::set<llvm::DILocalVariable*> variables;
        for (auto& block : function)
            for (auto& instruction : block)
                for (auto& record : instruction.getDbgRecordRange())
                    if (const auto* variable = llvm::dyn_cast<llvm::DbgVariableRecord>(&record))
                        variables.insert(variable->getVariable());
        if (variables.size() > 64) return fail("Source debug local count exceeds 64 per function");
        std::map<llvm::BasicBlock*, Values> exits;
        bool changed = true;
        unsigned iterations = 0;
        while (changed) {
            if (++iterations > function.size() * 4 + 8)
                return fail("Source debug reaching-value analysis did not converge");
            changed = false;
            for (auto& block : function) {
                Values values;
                bool first = true;
                for (auto* predecessor : llvm::predecessors(&block)) {
                    if (first) { values = exits[predecessor]; first = false; }
                    else for (auto& [variable, binding] : values)
                        if (!exits[predecessor].contains(variable) || exits[predecessor].at(variable) != binding)
                            binding = {};
                }
                for (auto& instruction : block) update(instruction, values);
                if (values != exits[&block]) { exits[&block] = std::move(values); changed = true; }
            }
        }
        for (auto& block : function) {
            Values values;
            bool first = true;
            for (auto* predecessor : llvm::predecessors(&block)) {
                if (first) { values = exits[predecessor]; first = false; }
                else for (auto& [variable, binding] : values)
                    if (!exits[predecessor].contains(variable) || exits[predecessor].at(variable) != binding)
                        binding = {};
            }
            std::tuple<const llvm::DIFile*, unsigned, unsigned, const llvm::DILocation*> previous{};
            for (auto it = block.begin(); it != block.end(); ++it) {
                auto& instruction = *it;
                update(instruction, values);
                const auto location = instruction.getDebugLoc();
                if (!location || llvm::isa<llvm::PHINode>(instruction)) continue;
                const auto position = std::make_tuple(location->getFile(), location.getLine(),
                    location->getDiscriminator(), location->getInlinedAt());
                if (!location.getLine() || position == previous) continue;
                previous = position;
                const auto* file = location->getFile();
                if (!file) continue;
                auto source = sources.find(normalized(file->getFilename()));
                if (source == sources.end())
                    source = sources.find(normalized((file->getDirectory() + "/" + file->getFilename()).str()));
                if (source == sources.end() || location.getLine() > source->second.second) continue;
                Site site;
                site.location.stage = stage;
                if (stage == NACELLE_SOURCE_COMPUTE)
                    std::copy_n(threadsPerGroup, 3, site.location.threads_per_group);
                site.location.line = location.getLine();
                site.location.column = location.getCol();
                site.location.discriminator = location->getDiscriminator();
                std::strcpy(site.location.shader_id, identity.c_str());
                std::strcpy(site.location.source_id, source->second.first.c_str());
                std::strcpy(site.location.label, "embedded.hlsl");
                unsigned frameCount = 0;
                for (const auto* frame = location.get(); frame; frame = frame->getInlinedAt()) {
                    if (frameCount == 8) return fail("Source debug inline chain exceeds eight frames");
                    const auto* frameFile = frame->getFile();
                    if (!frameFile) return fail("Source debug inline frame has no source file");
                    auto frameSource = sources.find(normalized(frameFile->getFilename()));
                    if (frameSource == sources.end())
                        frameSource = sources.find(normalized(
                            (frameFile->getDirectory() + "/" + frameFile->getFilename()).str()));
                    if (frameSource == sources.end() || !frame->getLine() ||
                        frame->getLine() > frameSource->second.second)
                        return fail("Source debug inline frame has no embedded source mapping");
                    auto& record = site.location.frames[frameCount++];
                    record.row = frame->getLine();
                    record.column = frame->getColumn();
                    record.discriminator = frame->getDiscriminator();
                    std::strcpy(record.source_id, frameSource->second.first.c_str());
                    const auto* subprogram = frame->getScope()->getSubprogram();
                    const auto name = subprogram ? subprogram->getName() : llvm::StringRef("<unavailable>");
                    if (name.size() >= sizeof(record.function))
                        return fail("Source debug inline function name exceeds bound");
                    std::memcpy(record.function, name.data(), name.size());
                }
                site.location.inline_depth = frameCount - 1;
                llvm::IRBuilder<> builder(&instruction);
                std::vector<llvm::Value*> arguments;
                arguments.push_back(builder.getInt32(0));
                for (auto* variable : variables) {
                    if (!inScope(*variable, *location)) continue;
                    auto* value = values.contains(variable) ? values.at(variable).value : nullptr;
                    auto local = describe(*variable, value);
                    if (location->getInlinedAt()) {
                        local.kind = NACELLE_SOURCE_UNAVAILABLE;
                        std::strcpy(local.reason, "inlined local mapping unavailable");
                    }
                    llvm::Value* bits = builder.getInt64(0);
                    if (local.kind) {
                        if (value->getType()->isFloatingPointTy())
                            value = builder.CreateBitCast(value, builder.getIntNTy(local.bits));
                        bits = builder.CreateZExtOrTrunc(value, builder.getInt64Ty());
                    }
                    site.locals.push_back(local);
                    arguments.push_back(bits);
                }
                const auto id = addSite(std::move(site), error);
                if (!id) return false;
                arguments[0] = builder.getInt32(id);
                auto marker = module.getOrInsertFunction("nacelle.source.site",
                    llvm::FunctionType::get(builder.getVoidTy(), {builder.getInt32Ty()}, true));
                builder.CreateCall(marker, arguments);
                ++sites;
            }
        }
    }
    llvm::StripDebugInfo(module);
    return sites ? true : fail("No executable embedded HLSL source sites are available");
}
}
