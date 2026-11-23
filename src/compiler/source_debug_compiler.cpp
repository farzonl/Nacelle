#include "source_debug.h"
#include "jit_engine.h"
#include "shader_compiler.h"
#include <llvm/BinaryFormat/Dwarf.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/CFG.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/DebugProgramInstruction.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/MemoryBufferRef.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Analysis/InstructionSimplify.h>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <map>
#include <optional>
#include <set>
#include <tuple>

namespace nacelle::source_debug {
namespace {
struct Binding {
    llvm::Value* value = nullptr;
    bool operator==(const Binding&) const = default;
};
using Values = std::map<llvm::DILocalVariable*, Binding>;

const llvm::DIType* unqualified(const llvm::DIType* type) {
    for (unsigned depth = 0; depth < 16; ++depth) {
        const auto* derived = llvm::dyn_cast_or_null<llvm::DIDerivedType>(type);
        if (!derived || (derived->getTag() != llvm::dwarf::DW_TAG_typedef &&
                         derived->getTag() != llvm::dwarf::DW_TAG_const_type &&
                         derived->getTag() != llvm::dwarf::DW_TAG_volatile_type)) break;
        type = derived->getBaseType();
    }
    return type;
}

const llvm::DIBasicType* fragmentType(const llvm::DIType* type, uint64_t offset,
                                     uint64_t size, unsigned depth = 0) {
    type = unqualified(type);
    if (!type || depth == 8 || offset > type->getSizeInBits() ||
        size > type->getSizeInBits() - offset) return nullptr;
    if (const auto* basic = llvm::dyn_cast<llvm::DIBasicType>(type))
        return offset == 0 && size == basic->getSizeInBits() ? basic : nullptr;
    const auto* composite = llvm::dyn_cast<llvm::DICompositeType>(type);
    if (!composite) return nullptr;
    if (composite->getTag() == llvm::dwarf::DW_TAG_array_type) {
        const auto* element = composite->getBaseType();
        if (!element || !element->getSizeInBits()) return nullptr;
        const auto elementSize = element->getSizeInBits();
        if (offset % elementSize + size > elementSize) return nullptr;
        return fragmentType(element, offset % elementSize, size, depth + 1);
    }
    if (composite->getTag() != llvm::dwarf::DW_TAG_structure_type &&
        composite->getTag() != llvm::dwarf::DW_TAG_class_type) return nullptr;
    for (const auto* element : composite->getElements()) {
        const auto* member = llvm::dyn_cast_or_null<llvm::DIDerivedType>(element);
        if (!member || member->getTag() != llvm::dwarf::DW_TAG_member ||
            !member->getBaseType() || member->getSizeInBits() != member->getBaseType()->getSizeInBits())
            continue;
        const auto start = member->getOffsetInBits();
        if (offset >= start && offset - start <= member->getSizeInBits() &&
            size <= member->getSizeInBits() - (offset - start))
            return fragmentType(member->getBaseType(), offset - start, size, depth + 1);
    }
    return nullptr;
}

llvm::Value* packetElement(llvm::Value* value, unsigned lane, const llvm::DataLayout& layout,
                           unsigned depth = 0) {
    if (depth == 32) return nullptr;
    if (auto* constant = llvm::dyn_cast<llvm::Constant>(value))
        return constant->getAggregateElement(lane);
    if (auto* insert = llvm::dyn_cast<llvm::InsertElementInst>(value)) {
        auto* index = llvm::dyn_cast<llvm::ConstantInt>(insert->getOperand(2));
        if (!index) return nullptr;
        if (index->getZExtValue() == lane) return insert->getOperand(1);
        return packetElement(insert->getOperand(0), lane, layout, depth + 1);
    }
    if (auto* shuffle = llvm::dyn_cast<llvm::ShuffleVectorInst>(value)) {
        const auto index = shuffle->getMaskValue(lane);
        if (index < 0) return nullptr;
        const auto width = llvm::cast<llvm::FixedVectorType>(shuffle->getOperand(0)->getType())->getNumElements();
        return packetElement(shuffle->getOperand(index / width), index % width, layout, depth + 1);
    }
    if (auto* instruction = llvm::dyn_cast<llvm::Instruction>(value)) {
        if (auto* simplified = llvm::simplifyInstruction(instruction, llvm::SimplifyQuery(layout));
            simplified && simplified != value)
            return packetElement(simplified, lane, layout, depth + 1);
    }
    return nullptr;
}

struct OptimizedBinding {
    llvm::SmallVector<llvm::Value*, 8> operands;
    llvm::DIExpression* expression = nullptr;
    bool operator==(const OptimizedBinding&) const = default;
};

using ExpressionKey = std::pair<unsigned, std::vector<llvm::Value*>>;

bool expressionLeaves(llvm::Value* value, unsigned opcode, std::vector<llvm::Value*>& leaves,
                      unsigned depth = 0) {
    if (depth == 16 || leaves.size() == 32) return false;
    auto* instruction = llvm::dyn_cast<llvm::BinaryOperator>(value);
    if (instruction && instruction->getOpcode() == opcode)
        return expressionLeaves(instruction->getOperand(0), opcode, leaves, depth + 1) &&
            expressionLeaves(instruction->getOperand(1), opcode, leaves, depth + 1);
    leaves.push_back(value);
    return true;
}

void canonicalizeLeaves(ExpressionKey& key) {
    std::sort(key.second.begin(), key.second.end(), std::less<llvm::Value*>{});
    if (key.first == llvm::Instruction::Or || key.first == llvm::Instruction::And)
        key.second.erase(std::unique(key.second.begin(), key.second.end()), key.second.end());
}

std::optional<ExpressionKey> currentExpression(const OptimizedBinding& binding) {
    llvm::SmallVector<ExpressionKey, 8> stack;
    if (!binding.expression || binding.operands.empty() || binding.expression->getNumElements() > 64)
        return std::nullopt;
    for (auto operation : binding.expression->expr_ops()) {
        const auto opcode = operation.getOp();
        if (opcode == llvm::dwarf::DW_OP_LLVM_arg) {
            if (operation.getArg(0) >= binding.operands.size() || stack.size() == 8) return std::nullopt;
            stack.push_back({0, {binding.operands[operation.getArg(0)]}});
            continue;
        }
        if (opcode == llvm::dwarf::DW_OP_stack_value) continue;
        unsigned binary = 0;
        switch (opcode) {
        case llvm::dwarf::DW_OP_or: binary = llvm::Instruction::Or; break;
        case llvm::dwarf::DW_OP_and: binary = llvm::Instruction::And; break;
        case llvm::dwarf::DW_OP_xor: binary = llvm::Instruction::Xor; break;
        case llvm::dwarf::DW_OP_plus: binary = llvm::Instruction::Add; break;
        case llvm::dwarf::DW_OP_mul: binary = llvm::Instruction::Mul; break;
        default: return std::nullopt;
        }
        if (stack.size() < 2) return std::nullopt;
        auto right = stack.pop_back_val();
        auto& left = stack.back();
        if ((left.first && left.first != binary) || (right.first && right.first != binary) ||
            left.second.size() + right.second.size() > 32) return std::nullopt;
        left.first = binary;
        left.second.insert(left.second.end(), right.second.begin(), right.second.end());
    }
    if (stack.size() != 1 || !stack.front().first) return std::nullopt;
    auto result = stack.pop_back_val();
    std::vector<llvm::Value*> expanded;
    for (auto* operand : result.second)
        if (!expressionLeaves(operand, result.first, expanded)) return std::nullopt;
    result.second = std::move(expanded);
    canonicalizeLeaves(result);
    return result;
}

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
    const llvm::DIType* sourceType = unqualified(variable.getType());
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
        if (const auto* unit = llvm::dyn_cast<llvm::DICompileUnit>(operand); !unit || (!optimized() && unit->isOptimized()))
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
    if (!unoptimized && !optimized()) return fail("Source debugging requires DXC -Od; optimized shaders are unsupported");
    llvm::DIBuilder debugBuilder(module, true);
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
        if (optimized()) {
            using Fragment = std::tuple<llvm::DILocalVariable*, uint64_t, uint64_t>;
            std::map<Fragment, llvm::DILocalVariable*> fragments;
            llvm::SmallVector<llvm::DbgVariableRecord*, 64> records;
            for (auto& block : function)
                for (auto& instruction : block)
                    for (auto& record : instruction.getDbgRecordRange())
                        if (auto* variable = llvm::dyn_cast<llvm::DbgVariableRecord>(&record))
                            records.push_back(variable);
            for (auto* record : records) {
                const auto* expression = record->getExpression();
                const auto fragment = expression->getFragmentInfo();
                if (!fragment || expression->getNumElements() != 3 || !record->isDbgValue() ||
                    record->getNumVariableLocationOps() != 1 || !fragment->SizeInBits ||
                    fragment->SizeInBits > 64) continue;
                auto* variable = record->getVariable();
                const auto* type = fragmentType(variable->getType(), fragment->OffsetInBits, fragment->SizeInBits);
                auto* valueType = record->getValue()->getType();
                if (!type || !(valueType->isIntegerTy() || valueType->isFloatingPointTy()) ||
                    type->getSizeInBits() != valueType->getScalarSizeInBits()) continue;
                const Fragment key{variable, fragment->OffsetInBits, fragment->SizeInBits};
                auto*& component = fragments[key];
                if (!component)
                    component = debugBuilder.createAutoVariable(variable->getScope(),
                        variable->getName().str() + ".bits[" + std::to_string(fragment->OffsetInBits) +
                        ":" + std::to_string(fragment->SizeInBits) + "]", variable->getFile(),
                        variable->getLine(), const_cast<llvm::DIBasicType*>(type));
            }
            // A whole-variable assignment or an overlapping unsupported fragment kills
            // previous component bindings; never retain an old piece across such an update.
            for (auto* record : records) {
                auto* variable = record->getVariable();
                const auto fragment = record->getExpression()->getFragmentInfo();
                llvm::DILocalVariable* component = nullptr;
                if (fragment && record->getExpression()->getNumElements() == 3 &&
                    record->isDbgValue() && record->getNumVariableLocationOps() == 1) {
                    auto found = fragments.find({variable, fragment->OffsetInBits, fragment->SizeInBits});
                    if (found != fragments.end() &&
                        record->getValue()->getType()->getScalarSizeInBits() == fragment->SizeInBits)
                        component = found->second;
                }
                for (const auto& [key, candidate] : fragments) {
                    const auto [owner, offset, size] = key;
                    if (owner != variable || candidate == component) continue;
                    if (fragment && (fragment->OffsetInBits >= offset + size ||
                        (fragment->SizeInBits <= UINT64_MAX - fragment->OffsetInBits &&
                         offset >= fragment->OffsetInBits + fragment->SizeInBits))) continue;
                    auto* killed = new llvm::DbgVariableRecord(
                        llvm::ValueAsMetadata::get(llvm::PoisonValue::get(
                            llvm::Type::getIntNTy(module.getContext(), size))),
                        candidate, debugBuilder.createExpression(), record->getDebugLoc(),
                        llvm::DbgVariableRecord::LocationType::Value);
                    killed->insertBefore(record);
                }
                if (component) {
                    record->setVariable(component);
                    record->setExpression(debugBuilder.createExpression());
                }
            }
        }
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
                    llvm::Value* bits = optimized() ? value : builder.getInt64(0);
                    if (local.kind && !optimized()) {
                        if (value->getType()->isFloatingPointTy())
                            value = builder.CreateBitCast(value, builder.getIntNTy(local.bits));
                        bits = builder.CreateZExtOrTrunc(value, builder.getInt64Ty());
                    }
                    site.locals.push_back(local);
                    arguments.push_back(bits);
                }
                const auto id = addSite(std::move(site), error);
                if (!id) return false;
                if (optimized()) {
                    const auto prefix = "nacelle.site." + std::to_string(id);
                    for (size_t index = 1; index < arguments.size(); ++index) {
                        auto* value = arguments[index];
                        if (!value) continue;
                        auto* variable = debugBuilder.createAutoVariable(location->getScope(),
                            prefix + ".local." + std::to_string(index - 1), location->getFile(),
                            location.getLine(), debugBuilder.createBasicType("bits",
                                value->getType()->getScalarSizeInBits(), llvm::dwarf::DW_ATE_unsigned));
                        debugBuilder.insertDbgValue(value, variable,
                            debugBuilder.createExpression(), location.get(), instruction.getIterator());
                    }
                    auto* label = debugBuilder.createLabel(location->getScope(), prefix,
                        location->getFile(), location.getLine(), location.getCol(), true, std::nullopt);
                    debugBuilder.insertLabel(label, location.get(), instruction.getIterator());
                    ++sites;
                    continue;
                }
                arguments[0] = builder.getInt32(id);
                auto marker = module.getOrInsertFunction("nacelle.source.site",
                    llvm::FunctionType::get(builder.getVoidTy(), {builder.getInt32Ty()}, true));
                builder.CreateCall(marker, arguments);
                ++sites;
            }
        }
    }
    debugBuilder.finalize();
    if (!optimized()) llvm::StripDebugInfo(module);
    return sites ? true : fail("No executable embedded HLSL source sites are available");
}

bool prepareOptimizedModule(llvm::Module& module, std::string& error) {
    if (!optimized()) return true;
    if (module.getFunction("nacelle_source_debug_observe") ||
        module.getFunction("nacelle.source.site")) {
        error = "Optimized provenance contains an eager source observation";
        return false;
    }
    bool hasProvenance = false;
    for (auto& function : module)
        for (auto& block : function)
            for (auto& instruction : block)
                for (auto& record : instruction.getDbgRecordRange())
                    if (const auto* label = llvm::dyn_cast<llvm::DbgLabelRecord>(&record))
                        hasProvenance |= label->getLabel()->getName().starts_with("nacelle.site.");
    if (!hasProvenance) return true;
    module.getOrInsertNamedMetadata("nacelle.source.provenance")->addOperand(
        llvm::MDNode::get(module.getContext(), {}));
    // Packetized functions are new functions, not the original scalar subprograms.
    // Rehome native debug records without changing their bounded runtime identities.
    llvm::DIBuilder builder(module, true);
    auto* file = builder.createFile("embedded.hlsl", "");
    auto* unit = builder.createCompileUnit(llvm::dwarf::DW_LANG_C, file, "Nacelle", true, "", 0);
    (void)unit;
    std::map<llvm::Type*, llvm::DIType*> packetTypes;
    const auto packetType = [&](llvm::Type* type) -> llvm::DIType* {
        auto found = packetTypes.find(type);
        if (found != packetTypes.end()) return found->second;
        auto* scalar = type->getScalarType();
        const auto bits = scalar->isPointerTy() ?
            module.getDataLayout().getPointerTypeSizeInBits(scalar) : scalar->getScalarSizeInBits();
        llvm::DIType* result = nullptr;
        if (scalar->isPointerTy())
            result = builder.createPointerType(nullptr, bits);
        else if (scalar->isIntegerTy() || scalar->isFloatingPointTy())
            result = builder.createBasicType("packet component", bits,
                scalar->isFloatingPointTy() ? llvm::dwarf::DW_ATE_float :
                    (scalar->isIntegerTy(1) ? llvm::dwarf::DW_ATE_boolean : llvm::dwarf::DW_ATE_unsigned));
        if (auto* vector = llvm::dyn_cast<llvm::FixedVectorType>(type); vector && result) {
            llvm::Metadata* count = builder.getOrCreateSubrange(0, vector->getNumElements());
            result = builder.createArrayType(vector->getNumElements() * bits,
                bits, result, builder.getOrCreateArray({count}));
        }
        packetTypes.emplace(type, result);
        return result;
    };
    for (auto& function : module) {
        if (function.isDeclaration()) continue;
        function.convertToNewDbgValues();
        llvm::SmallVector<llvm::DbgVariableRecord*, 64> packetBindings;
        for (auto& block : function)
            for (auto& instruction : block)
                for (auto& record : instruction.getDbgRecordRange())
                    if (auto* variable = llvm::dyn_cast<llvm::DbgVariableRecord>(&record);
                        variable && variable->isDbgValue() &&
                        variable->getNumVariableLocationOps() == 1 &&
                        variable->getExpression()->getNumElements() == 0 &&
                        variable->getVariable()->getName().starts_with("nacelle.site."))
                        packetBindings.push_back(variable);
        const auto insertBinding = [&](llvm::DbgVariableRecord* before, const std::string& name,
                                       llvm::Value* value) {
            auto* original = before->getVariable();
            auto* variable = builder.createAutoVariable(original->getScope(), name,
                original->getFile(), original->getLine(), packetType(value->getType()));
            auto* record = new llvm::DbgVariableRecord(llvm::ValueAsMetadata::get(value),
                variable, builder.createExpression(), before->getDebugLoc());
            record->insertBefore(before);
            return record;
        };
        struct PixelContextBindings {
            llvm::DbgVariableRecord* record = nullptr;
            llvm::Value* batch[2]{};
        };
        std::map<std::string, PixelContextBindings> pixelContexts;
        const auto pixelBatch = [&](llvm::Value* value, unsigned width, unsigned dimension) -> llvm::Value* {
            if (!function.getName().starts_with("nacelle_ps_") || function.arg_size() != 1 ||
                !function.getReturnType()->isVoidTy() || !function.getArg(0)->getType()->isPointerTy())
                return nullptr;
            auto* load = llvm::dyn_cast<llvm::LoadInst>(value);
            auto* address = load ? llvm::dyn_cast<llvm::GetElementPtrInst>(load->getPointerOperand()) : nullptr;
            if (!address || !address->getSourceElementType()->isIntegerTy(8) ||
                address->getNumIndices() != 1) return nullptr;
            auto* scale = llvm::dyn_cast<llvm::BinaryOperator>(address->idx_begin()->get());
            if (!scale || scale->getOpcode() != llvm::Instruction::Mul) return nullptr;
            auto* factor = llvm::dyn_cast<llvm::ConstantInt>(scale->getOperand(1));
            llvm::Value* batch = scale->getOperand(0);
            if (!factor) {
                factor = llvm::dyn_cast<llvm::ConstantInt>(scale->getOperand(0));
                batch = scale->getOperand(1);
            }
            if (!factor || factor->getZExtValue() != width * sizeof(float) ||
                !batch->getType()->isIntegerTy(64)) return nullptr;
            auto* field = llvm::dyn_cast<llvm::LoadInst>(address->getPointerOperand());
            if (!field || !field->getType()->isPointerTy()) return nullptr;
            llvm::APInt offset(module.getDataLayout().getIndexTypeSizeInBits(field->getPointerOperandType()), 0);
            auto* root = field->getPointerOperand()->stripAndAccumulateConstantOffsets(
                module.getDataLayout(), offset, true);
            if (root != function.getArg(0) || offset.getZExtValue() !=
                (dimension == 0 ? offsetof(PixelBatchArgs, pixelX) : offsetof(PixelBatchArgs, pixelY)))
                return nullptr;
            return batch;
        };
        for (size_t index = 0; index < packetBindings.size(); ++index) {
            auto* record = packetBindings[index];
            auto* value = record->getValue();
            auto* type = llvm::dyn_cast<llvm::FixedVectorType>(value->getType());
            if (!type || type->getNumElements() > 16) continue;
            auto name = record->getVariable()->getName();
            if (name.ends_with(".exec"))
                insertBinding(record, name.drop_back(5).str() + ".width",
                    llvm::ConstantInt::get(llvm::Type::getInt32Ty(module.getContext()), type->getNumElements()));
            if (name.contains(".coord.")) {
                llvm::Value* raw = value;
                if (auto* select = llvm::dyn_cast<llvm::SelectInst>(raw)) raw = select->getTrueValue();
                if (auto* conversion = llvm::dyn_cast<llvm::FPToUIInst>(raw)) {
                    raw = conversion->getOperand(0);
                    if (auto* select = llvm::dyn_cast<llvm::SelectInst>(raw)) raw = select->getTrueValue();
                    auto floatName = name.str();
                    floatName.replace(floatName.find(".coord."), 7, ".coordfloat.");
                    packetBindings.push_back(insertBinding(record, floatName, raw));
                    if (name.ends_with(".coord.0") || name.ends_with(".coord.1")) {
                        const unsigned dimension = name.ends_with(".coord.0") ? 0 : 1;
                        if (auto* batch = pixelBatch(raw, type->getNumElements(), dimension)) {
                            auto& descriptor = pixelContexts[name.drop_back(8).str()];
                            descriptor.record = record;
                            descriptor.batch[dimension] = batch;
                        }
                    }
                }
            }
            for (const auto& [prefix, descriptor] : pixelContexts)
                if (descriptor.batch[0] && descriptor.batch[0] == descriptor.batch[1]) {
                    insertBinding(descriptor.record, prefix + ".pixel.args", function.getArg(0));
                    insertBinding(descriptor.record, prefix + ".pixel.batch", descriptor.batch[0]);
                }
            if (name.ends_with(".invocation"))
                if (auto* joined = llvm::dyn_cast<llvm::PHINode>(value))
                    for (unsigned incoming = 0; incoming < joined->getNumIncomingValues(); ++incoming)
                        for (unsigned lane = 0; lane < type->getNumElements(); ++lane) {
                            auto* component = packetElement(joined->getIncomingValue(incoming), lane,
                                module.getDataLayout());
                            auto* cast = llvm::dyn_cast_or_null<llvm::PtrToIntInst>(component);
                            if (!cast) continue;
                            auto* pointer = cast->getPointerOperand();
                            const auto componentName = name.str() + ".incoming." + std::to_string(incoming) +
                                ".pointer.lane." + std::to_string(lane);
                            auto* variable = builder.createAutoVariable(record->getVariable()->getScope(),
                                componentName, record->getVariable()->getFile(), record->getVariable()->getLine(),
                                packetType(pointer->getType()));
                            builder.insertDbgValue(pointer, variable, builder.createExpression(),
                                record->getDebugLoc().get(), joined->getIncomingBlock(incoming)->getTerminator()->getIterator());
                        }
            for (unsigned lane = 0; lane < type->getNumElements(); ++lane) {
                auto* element = packetElement(value, lane, module.getDataLayout());
                if (auto* instruction = llvm::dyn_cast_or_null<llvm::Instruction>(element))
                    if (auto* simplified = llvm::simplifyInstruction(instruction,
                        llvm::SimplifyQuery(module.getDataLayout()))) element = simplified;
                const auto elementName = name.str() + ".lane." + std::to_string(lane);
                if (element) insertBinding(record, elementName, element);
                if (name.ends_with(".invocation"))
                    if (auto* cast = llvm::dyn_cast_or_null<llvm::PtrToIntInst>(element))
                        insertBinding(record, name.str() + ".pointer.lane." + std::to_string(lane),
                            cast->getPointerOperand());
                for (auto* user : value->users()) {
                    auto* extract = llvm::dyn_cast<llvm::ExtractElementInst>(user);
                    auto* selected = extract ? llvm::dyn_cast<llvm::ConstantInt>(extract->getIndexOperand()) : nullptr;
                    if (!selected || selected->getZExtValue() != lane || !extract->getNextNode()) continue;
                    auto* variable = builder.createAutoVariable(record->getVariable()->getScope(),
                        elementName, record->getVariable()->getFile(), record->getVariable()->getLine(),
                        packetType(extract->getType()));
                    builder.insertDbgValue(extract, variable, builder.createExpression(),
                        record->getDebugLoc().get(), extract->getNextNode()->getIterator());
                }
            }
        }
        using SourcePosition = std::tuple<const llvm::DIFile*, unsigned, unsigned, unsigned,
            const llvm::DILocation*>;
        const auto position = [](const llvm::DILocation* location) {
            return SourcePosition{location->getFile(), location->getLine(), location->getColumn(),
                location->getDiscriminator(), location->getInlinedAt()};
        };
        std::map<SourcePosition, llvm::SmallVector<uint32_t, 2>> sourceSites;
        for (auto& block : function)
            for (auto& instruction : block)
                for (auto& record : instruction.getDbgRecordRange())
                    if (const auto* label = llvm::dyn_cast<llvm::DbgLabelRecord>(&record)) {
                        auto name = label->getLabel()->getName();
                        uint32_t identity = 0;
                        if (!name.consume_front("nacelle.site.") || name.getAsInteger(10, identity) ||
                            !label->getDebugLoc()) continue;
                        auto& candidates = sourceSites[position(label->getDebugLoc().get())];
                        if (!llvm::is_contained(candidates, identity)) candidates.push_back(identity);
                        if (candidates.size() > 8) {
                            error = "Optimized source origin set exceeds eight candidates";
                            return false;
                        }
                    }
        auto* subprogram = builder.createFunction(file, function.getName(), function.getName(),
            file, 1, builder.createSubroutineType(builder.getOrCreateTypeArray({})), 1,
            llvm::DINode::FlagArtificial, llvm::DISubprogram::SPFlagDefinition |
            llvm::DISubprogram::SPFlagOptimized);
        function.setSubprogram(subprogram);
        std::map<llvm::DILocalVariable*, llvm::DILocalVariable*> variables;
        std::map<llvm::DILabel*, llvm::DILabel*> labels;
        for (auto& block : function)
            for (auto& instruction : block) {
                const auto location = instruction.getDebugLoc();
                llvm::SmallVector<llvm::Metadata*, 8> anchors;
                if (location && !llvm::isa<llvm::PHINode>(instruction) && !instruction.isTerminator())
                    if (auto found = sourceSites.find(position(location.get())); found != sourceSites.end())
                        for (const auto identity : found->second)
                            anchors.push_back(llvm::ConstantAsMetadata::get(
                                llvm::ConstantInt::get(llvm::Type::getInt32Ty(module.getContext()), identity)));
                for (auto& record : instruction.getDbgRecordRange()) {
                    record.setDebugLoc(llvm::DILocation::get(module.getContext(), 1, 0, subprogram));
                    if (auto* variable = llvm::dyn_cast<llvm::DbgVariableRecord>(&record)) {
                        auto* original = variable->getVariable();
                        auto*& replacement = variables[original];
                        if (!replacement)
                            replacement = builder.createAutoVariable(subprogram, original->getName(),
                                file, 1, variable->getNumVariableLocationOps() == 1 ?
                                    packetType(variable->getVariableLocationOp(0)->getType()) : original->getType());
                        variable->setVariable(replacement);
                    } else if (auto* label = llvm::dyn_cast<llvm::DbgLabelRecord>(&record)) {
                        auto* original = label->getLabel();
                        auto*& replacement = labels[original];
                        if (!replacement)
                            replacement = builder.createLabel(subprogram, original->getName(), file,
                                1, 0, true, std::nullopt);
                        label->setLabel(replacement);
                    }
                }
                if (!anchors.empty())
                    instruction.setMetadata("nacelle.source.anchor",
                        llvm::MDNode::get(module.getContext(), anchors));
                if (location)
                    instruction.setDebugLoc(llvm::DILocation::get(module.getContext(),
                        location.getLine(), location.getCol(), subprogram));
            }
    }
    builder.finalize();
    return true;
}

uint64_t optimizedCompilation(const llvm::Module& module) {
    const auto* metadata = module.getNamedMetadata("nacelle.source.compilation");
    if (!metadata || !metadata->getNumOperands()) return 0;
    const auto* value = llvm::mdconst::dyn_extract_or_null<llvm::ConstantInt>(
        metadata->getOperand(0)->getOperand(0));
    return value ? value->getZExtValue() : 0;
}

bool lowerOptimizedObservations(llvm::Module& module, uint32_t tier, std::string& error) {
    if (!optimized()) return true;
    if (!module.getNamedMetadata("nacelle.source.provenance")) return true;
    if (optimizedCompilation(module)) {
        error = "Optimized source observations have already been lowered";
        return false;
    }
    static std::atomic<uint64_t> nextCompilation{1};
    const auto compilation = nextCompilation.fetch_add(1, std::memory_order_relaxed);
    using Reaching = std::map<std::string, OptimizedBinding>;
    const auto transfer = [](llvm::DbgRecord& record, Reaching& values) {
        auto* variable = llvm::dyn_cast<llvm::DbgVariableRecord>(&record);
        if (!variable || !variable->getVariable()->getName().starts_with("nacelle.site.")) return;
        OptimizedBinding binding;
        if (variable->isDbgValue() && variable->getNumVariableLocationOps() <= 8 &&
            variable->getExpression()->getNumElements() <= 64 && !variable->isKillLocation()) {
            binding.expression = variable->getExpression();
            for (auto* operand : variable->location_ops()) binding.operands.push_back(operand);
        }
        values[variable->getVariable()->getName().str()] = std::move(binding);
    };
    const auto incoming = [](llvm::BasicBlock& block,
                             const std::map<llvm::BasicBlock*, Reaching>& exits) {
        Reaching values;
        bool first = true;
        for (auto* predecessor : llvm::predecessors(&block)) {
            const auto found = exits.find(predecessor);
            const Reaching empty;
            const auto& other = found == exits.end() ? empty : found->second;
            if (first) { values = other; first = false; }
            else for (auto& [name, value] : values) {
                auto binding = other.find(name);
                if (binding == other.end() || binding->second != value) value = {};
            }
        }
        return values;
    };
    uint32_t finalized = 0, labelsSeen = 0, anchorsSeen = 0, masksSeen = 0, coordinatesSeen = 0,
        invocationsSeen = 0;
    for (auto& function : module) {
        if (function.isDeclaration()) continue;
        llvm::DominatorTree dominance(function);
        std::map<ExpressionKey, llvm::SmallVector<llvm::Instruction*, 2>> existingExpressions;
        for (auto& block : function)
            for (auto& instruction : block) {
                const auto opcode = instruction.getOpcode();
                if (opcode != llvm::Instruction::Or && opcode != llvm::Instruction::And &&
                    opcode != llvm::Instruction::Xor && opcode != llvm::Instruction::Add &&
                    opcode != llvm::Instruction::Mul) continue;
                ExpressionKey key{opcode, {}};
                if (!expressionLeaves(&instruction, opcode, key.second)) continue;
                canonicalizeLeaves(key);
                existingExpressions[std::move(key)].push_back(&instruction);
            }
        llvm::AllocaInst* coordinateScratch = nullptr;
        llvm::AllocaInst* valueScratch = nullptr;
        llvm::AllocaInst* invocationScratch = nullptr;
        llvm::AllocaInst* maskScratch = nullptr;
        std::map<llvm::BasicBlock*, Reaching> exits;
        bool changed = true;
        unsigned iterations = 0;
        while (changed) {
            if (++iterations > function.size() * 4 + 8) {
                error = "Optimized source reaching-binding analysis did not converge";
                return false;
            }
            changed = false;
            for (auto& block : function) {
                auto values = incoming(block, exits);
                for (auto& instruction : block)
                    for (auto& record : instruction.getDbgRecordRange()) transfer(record, values);
                if (exits[&block] != values) { exits[&block] = std::move(values); changed = true; }
            }
        }
        for (auto& block : function) {
            auto values = incoming(block, exits);
            for (auto iterator = block.begin(); iterator != block.end(); ++iterator) {
                auto& instruction = *iterator;
                llvm::SmallVector<llvm::DbgRecord*, 16> records;
                for (auto& record : instruction.getDbgRecordRange()) records.push_back(&record);
                for (auto* recordPointer : records) {
                    auto& record = *recordPointer;
                    transfer(record, values);
                    if (auto* label = llvm::dyn_cast<llvm::DbgLabelRecord>(&record);
                        label && label->getLabel()->getName().starts_with("nacelle.site.")) ++labelsSeen;
                }
                const auto* anchor = instruction.getMetadata("nacelle.source.anchor");
                if (!anchor || llvm::isa<llvm::PHINode>(instruction) || instruction.isTerminator()) continue;
                for (const auto& operand : anchor->operands()) {
                    const auto* identity = llvm::mdconst::dyn_extract_or_null<llvm::ConstantInt>(operand);
                    if (!identity) continue;
                    const auto original = static_cast<uint32_t>(identity->getZExtValue());
                    ++anchorsSeen;
                    Site site;
                    if (!readSite(original, site)) {
                        error = "Optimized source label has no admitted immutable metadata";
                        return false;
                    }
                    site.mapping = anchor->getNumOperands() > 1 ?
                        NACELLE_SOURCE_MAPPING_UNAVAILABLE : NACELLE_SOURCE_MAPPING_EXACT;
                    const auto prefix = "nacelle.site." + std::to_string(original);
                    const auto binding = [&](const std::string& suffix) -> llvm::Value* {
                        const auto found = values.find(prefix + suffix);
                        if (found == values.end()) return nullptr;
                        const auto& binding = found->second;
                        if (!binding.expression || binding.operands.empty()) return nullptr;
                        for (auto* value : binding.operands) {
                            if (!value || llvm::isa<llvm::UndefValue, llvm::PoisonValue>(value)) return nullptr;
                            if (auto* definition = llvm::dyn_cast<llvm::Instruction>(value);
                                definition && (definition == &instruction ||
                                    !dominance.dominates(definition, &instruction))) return nullptr;
                            if (auto* constant = llvm::dyn_cast<llvm::Constant>(value);
                                constant && constant->containsUndefOrPoisonElement()) return nullptr;
                        }
                        if (binding.expression->getNumElements() == 0 && binding.operands.size() == 1)
                            return binding.operands.front();
                        // Resolve a current native debug expression only to an already
                        // surviving computation. Never emit its operations or recreate it.
                        auto expression = currentExpression(binding);
                        if (!expression) return nullptr;
                        auto candidates = existingExpressions.find(*expression);
                        if (candidates == existingExpressions.end()) return nullptr;
                        for (auto* candidate : candidates->second)
                            if (candidate != &instruction && dominance.dominates(candidate, &instruction))
                                return candidate;
                        return nullptr;
                    };
                    auto* activeValue = binding(".exec");
                    auto* maskType = activeValue ? llvm::dyn_cast<llvm::FixedVectorType>(activeValue->getType()) : nullptr;
                    auto* widthValue = llvm::dyn_cast_or_null<llvm::ConstantInt>(binding(".width"));
                    const unsigned width = widthValue ? widthValue->getZExtValue() :
                        (maskType ? maskType->getNumElements() : 0);
                    if (width != 4 && width != 8 && width != 16) continue;
                    using Packet = llvm::SmallVector<llvm::Value*, 16>;
                    const auto packet = [&](const std::string& suffix, llvm::Type* element) {
                        Packet result;
                        if (auto* value = binding(suffix);
                            value && value->getType() == llvm::FixedVectorType::get(element, width)) {
                            result.push_back(value);
                            return result;
                        }
                        for (unsigned lane = 0; lane < width; ++lane) {
                            auto* value = binding(suffix + ".lane." + std::to_string(lane));
                            if (!value || value->getType() != element) return Packet{};
                            result.push_back(value);
                        }
                        return result;
                    };
                    auto& context = module.getContext();
                    auto activePacket = packet(".exec", llvm::Type::getInt1Ty(context));
                    auto realPacket = packet(".real", llvm::Type::getInt1Ty(context));
                    const bool masksComplete = !activePacket.empty() && !realPacket.empty();
                    if (masksComplete) ++masksSeen;
                    Packet coordinates[3];
                    bool floatingCoordinates[3]{};
                    bool complete = true;
                    for (unsigned dimension = 0; dimension < 3; ++dimension) {
                        coordinates[dimension] = packet(".coord." + std::to_string(dimension),
                            llvm::Type::getInt32Ty(context));
                        if (coordinates[dimension].empty()) {
                            coordinates[dimension] = packet(".coordfloat." + std::to_string(dimension),
                                llvm::Type::getFloatTy(context));
                            floatingCoordinates[dimension] = !coordinates[dimension].empty();
                        }
                        complete &= !coordinates[dimension].empty();
                    }
                    auto invocationPacket = packet(".invocation", llvm::Type::getInt64Ty(context));
                    bool pointerInvocation = false;
                    if (invocationPacket.empty()) {
                        invocationPacket = packet(".invocation.pointer", llvm::PointerType::getUnqual(context));
                        pointerInvocation = !invocationPacket.empty();
                    }
                    if (complete) ++coordinatesSeen;
                    if (!invocationPacket.empty()) ++invocationsSeen;
                    const bool pixel = site.location.stage == NACELLE_SOURCE_PIXEL;
                    const bool needsPixelContext = pixel &&
                        (!complete || !masksComplete || invocationPacket.empty());
                    llvm::Value* pixelArgs = nullptr;
                    llvm::Value* pixelBatch = nullptr;
                    if (needsPixelContext) {
                        pixelArgs = binding(".pixel.args");
                        pixelBatch = binding(".pixel.batch");
                        const auto entry = [&](const char* suffix) {
                            auto* flag = llvm::dyn_cast_or_null<llvm::ConstantInt>(binding(suffix));
                            return flag && flag->isOne();
                        };
                        if (!pixelArgs || !pixelArgs->getType()->isPointerTy() || !pixelBatch ||
                            !pixelBatch->getType()->isIntegerTy(64) ||
                            (activePacket.empty() && !entry(".exec.entry")) ||
                            (realPacket.empty() && !entry(".real.entry"))) continue;
                    } else if (!complete || !masksComplete) continue;
                    site.compilation = compilation;
                    site.tier = tier;
                    std::vector<Packet> locals;
                    site.availability.clear();
                    for (size_t slot = 0; slot < site.locals.size(); ++slot) {
                        auto& local = site.locals[slot];
                        const bool unsupported = local.kind == NACELLE_SOURCE_UNAVAILABLE;
                        llvm::Type* scalar = nullptr;
                        if (local.kind == NACELLE_SOURCE_FLOAT) {
                            if (local.bits == 16) scalar = llvm::Type::getHalfTy(context);
                            if (local.bits == 32) scalar = llvm::Type::getFloatTy(context);
                            if (local.bits == 64) scalar = llvm::Type::getDoubleTy(context);
                        } else if (local.kind && local.bits && local.bits <= 64)
                            scalar = llvm::Type::getIntNTy(context, local.bits);
                        auto value = scalar ? packet(".local." + std::to_string(slot), scalar) : Packet{};
                        if (value.empty()) {
                            local.kind = NACELLE_SOURCE_UNAVAILABLE;
                            if (!unsupported)
                                std::strcpy(local.reason, "binding killed, unsupported or not reaching optimized site");
                        }
                        locals.push_back(value);
                        site.availability.push_back(!value.empty() ? NACELLE_SOURCE_VALUE_AVAILABLE :
                            (unsupported ? NACELLE_SOURCE_VALUE_UNSUPPORTED : NACELLE_SOURCE_VALUE_OPTIMIZED_OUT));
                    }
                    const auto id = addSite(std::move(site), error);
                    if (!id) return false;
                    llvm::IRBuilder<> builder(&instruction);
                    builder.SetCurrentDebugLocation({});
                    const auto materialize = [&](const Packet& packet) -> llvm::Value* {
                        if (packet.empty()) return nullptr;
                        if (packet.size() == 1) return packet[0];
                        llvm::Value* result = llvm::PoisonValue::get(
                            llvm::FixedVectorType::get(packet[0]->getType(), width));
                        for (unsigned lane = 0; lane < width; ++lane)
                            result = builder.CreateInsertElement(result, packet[lane], lane);
                        return result;
                    };
                    auto* active = materialize(activePacket);
                    auto* real = materialize(realPacket);
                    llvm::Value* coordinateValues[3];
                    for (unsigned dimension = 0; dimension < 3; ++dimension) {
                        auto* value = materialize(coordinates[dimension]);
                        if (value && floatingCoordinates[dimension]) {
                            auto* type = llvm::cast<llvm::VectorType>(value->getType());
                            auto* zero = llvm::Constant::getNullValue(type);
                            auto* limit = llvm::ConstantVector::getSplat(llvm::ElementCount::getFixed(width),
                                llvm::ConstantFP::get(type->getElementType(), 4294967296.0));
                            auto* valid = builder.CreateAnd(builder.CreateFCmpOGE(value, zero),
                                builder.CreateFCmpOLT(value, limit));
                            value = builder.CreateSelect(valid,
                                builder.CreateFPToUI(builder.CreateSelect(valid, value, zero),
                                    llvm::FixedVectorType::get(builder.getInt32Ty(), width)),
                                llvm::Constant::getAllOnesValue(llvm::FixedVectorType::get(builder.getInt32Ty(), width)));
                        }
                        coordinateValues[dimension] = value;
                    }
                    auto* invocation = materialize(invocationPacket);
                    if (pointerInvocation)
                        invocation = builder.CreatePtrToInt(invocation,
                            llvm::FixedVectorType::get(builder.getInt64Ty(), width));
                    std::vector<llvm::Value*> localValues;
                    for (const auto& packet : locals) localValues.push_back(materialize(packet));
                    if (!coordinateScratch) {
                        llvm::IRBuilder<> allocationBuilder(&*function.getEntryBlock().getFirstInsertionPt());
                        allocationBuilder.SetCurrentDebugLocation({});
                        coordinateScratch = allocationBuilder.CreateAlloca(builder.getInt32Ty(), builder.getInt32(3 * 16));
                        valueScratch = allocationBuilder.CreateAlloca(builder.getInt64Ty(), builder.getInt32(64 * 16));
                        invocationScratch = allocationBuilder.CreateAlloca(builder.getInt64Ty(), builder.getInt32(16));
                        maskScratch = allocationBuilder.CreateAlloca(builder.getInt32Ty(), builder.getInt32(2));
                    }
                    auto* coords = coordinateScratch;
                    auto* bits = valueScratch;
                    auto* invocations = invocationScratch;
                    const auto store = [&](llvm::Value* pointer, llvm::Type* type, unsigned offset, llvm::Value* value) {
                        builder.CreateStore(value, builder.CreateGEP(type, pointer, builder.getInt32(offset)));
                    };
                    llvm::Value* contextValid = nullptr;
                    if (needsPixelContext) {
                        auto reader = module.getOrInsertFunction("nacelle_source_debug_read_pixel_context",
                            llvm::FunctionType::get(builder.getInt1Ty(), {builder.getPtrTy(), builder.getInt64Ty(),
                                builder.getInt32Ty(), builder.getPtrTy(), builder.getPtrTy(), builder.getPtrTy()}, false));
                        contextValid = builder.CreateCall(reader, {pixelArgs, pixelBatch, builder.getInt32(width),
                            coords, maskScratch, invocations});
                        if (!active)
                            active = builder.CreateBitCast(builder.CreateTrunc(
                                builder.CreateLoad(builder.getInt32Ty(), maskScratch), builder.getIntNTy(width)),
                                llvm::FixedVectorType::get(builder.getInt1Ty(), width));
                        if (!real)
                            real = builder.CreateBitCast(builder.CreateTrunc(builder.CreateLoad(builder.getInt32Ty(),
                                builder.CreateGEP(builder.getInt32Ty(), maskScratch, builder.getInt32(1))),
                                builder.getIntNTy(width)), llvm::FixedVectorType::get(builder.getInt1Ty(), width));
                        active = builder.CreateSelect(contextValid, active, llvm::Constant::getNullValue(active->getType()));
                        real = builder.CreateSelect(contextValid, real, llvm::Constant::getNullValue(real->getType()));
                    }
                    auto* activeBits = builder.CreateBitCast(active, builder.getIntNTy(width));
                    auto* realBits = builder.CreateBitCast(real, builder.getIntNTy(width));
                    for (unsigned lane = 0; lane < width; ++lane) {
                        for (unsigned dimension = 0; dimension < 3; ++dimension)
                            if (coordinateValues[dimension])
                                store(coords, builder.getInt32Ty(), lane * 3 + dimension,
                                    builder.CreateExtractElement(coordinateValues[dimension], lane));
                        if (invocation || !needsPixelContext)
                            store(invocations, builder.getInt64Ty(), lane, invocation ?
                                builder.CreateExtractElement(invocation, lane) : builder.getInt64(0));
                        for (size_t slot = 0; slot < locals.size(); ++slot) {
                            llvm::Value* value = builder.getInt64(0);
                            if (localValues[slot]) {
                                value = builder.CreateExtractElement(localValues[slot], lane);
                                if (value->getType()->isFloatingPointTy())
                                    value = builder.CreateBitCast(value, builder.getIntNTy(value->getType()->getScalarSizeInBits()));
                                value = builder.CreateZExtOrTrunc(value, builder.getInt64Ty());
                                value = builder.CreateSelect(builder.CreateExtractElement(active, lane),
                                    value, builder.getInt64(0));
                            }
                            store(bits, builder.getInt64Ty(), slot * width + lane, value);
                        }
                    }
                    auto callback = module.getOrInsertFunction("nacelle_source_debug_observe",
                        llvm::FunctionType::get(builder.getVoidTy(), {builder.getInt32Ty(), builder.getInt32Ty(),
                            builder.getInt32Ty(), builder.getInt32Ty(), builder.getPtrTy(),
                            builder.getPtrTy(), builder.getPtrTy()}, false));
                    builder.CreateCall(callback, {builder.getInt32(id), builder.getInt32(width),
                        builder.CreateZExt(activeBits, builder.getInt32Ty()),
                        builder.CreateZExt(realBits, builder.getInt32Ty()), coords, bits, invocations});
                    ++finalized;
                }
            }
        }
    }
    if (!finalized) {
        error = "No optimized source observations retain complete execution/mask/coordinate provenance"
            " (labels=" + std::to_string(labelsSeen) + ", anchors=" + std::to_string(anchorsSeen) +
            ", masks=" + std::to_string(masksSeen) + ", coordinates=" + std::to_string(coordinatesSeen) +
            ", invocations=" + std::to_string(invocationsSeen) + ")";
        return false;
    }
    module.getOrInsertNamedMetadata("nacelle.source.compilation")->addOperand(
        llvm::MDNode::get(module.getContext(), llvm::ConstantAsMetadata::get(
            llvm::ConstantInt::get(llvm::Type::getInt64Ty(module.getContext()), compilation))));
    return true;
}
}
