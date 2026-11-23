#include <nacelle/source_debug.h>
#include "compiler/source_debug.h"
#include "compiler/native_debug.h"
#include "compiler/shader_compiler.h"
#include <llvm/IR/Constants.h>
#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/DebugProgramInstruction.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <future>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <set>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
const std::string shader(64, 'a'), source(64, 'b'), other(64, 'c');
nacelle::source_debug::Site makeSite() {
    nacelle::source_debug::Site site;
    std::strcpy(site.location.shader_id, shader.c_str());
    std::strcpy(site.location.source_id, source.c_str());
    std::strcpy(site.location.label, "compute");
    site.location.stage = NACELLE_SOURCE_COMPUTE;
    site.location.line = 2;
    site.location.frames[0].row = 2;
    std::strcpy(site.location.frames[0].source_id, source.c_str());
    std::strcpy(site.location.frames[0].function, "main");
    site.location.local_count = 1;
    NacelleSourceDebugLocal local;
    local.kind = NACELLE_SOURCE_UNSIGNED;
    local.bits = 32;
    std::strcpy(local.name, "value");
    site.locals.push_back(local);
    return site;
}
NacelleSourceDebugSelection selection() {
    NacelleSourceDebugSelection value;
    std::strcpy(value.shader_id, shader.c_str());
    value.xyz[0] = 7;
    return value;
}
uint64_t begin() {
    auto target = selection();
    uint64_t id = 0;
    require(NacelleBeginSourceDebug(&target, &id) == S_OK && id, "Begin failed");
    return id;
}
void optimizedPolicyValidation() {
    using namespace nacelle::source_debug;
    NacelleOptimizedSourceDebugOptions options;
    options.enabled = 1;
    options.consent = 1;
    require(FAILED(NacelleConfigureOptimizedSourceDebug(&options)),
        "Optimized mode accepted missing value consent");
    options.consent = 3;
    require(NacelleConfigureOptimizedSourceDebug(&options) == S_OK && optimized() &&
        nacelle::native_debug::enabled() && !nacelle::native_debug::pinned(),
        "Optimized source policy incorrectly pins compilation");
    NacelleNativeDebugOptions native;
    require(FAILED(NacelleConfigureNativeDebug(&native)) && nacelle::native_debug::enabled(),
        "Pinned native configuration disabled optimized registration");
    require(freeze() && FAILED(NacelleConfigureOptimizedSourceDebug(&options)),
        "Late optimized reconfiguration accepted");
    std::string error;
    require(addSource(source, "first\nsecond\n", error), "Optimized source admission");
    auto candidate = makeSite();
    candidate.availability = {NACELLE_SOURCE_VALUE_PARTIAL};
    require(!addSite(candidate, error) && !error.empty(),
        "Partial value was admitted as a fully valid per-lane scalar");
    candidate.availability = {UINT32_MAX};
    require(!addSite(candidate, error), "Invalid optimized availability was admitted");
    candidate.availability = {NACELLE_SOURCE_VALUE_AVAILABLE};
    candidate.mapping = NACELLE_SOURCE_MAPPING_MERGED;
    require(!addSite(candidate, error), "Merged mapping hid its source origins");
    const auto origin = primaryOrigin(candidate.location);
    candidate.origins = {origin, origin};
    require(!addSite(candidate, error), "Duplicate origins made a single source look merged");
    candidate.origins.resize(9);
    require(!addSite(candidate, error), "Merged origin cap was silently truncated");
    candidate.origins.clear();
    candidate.mapping = NACELLE_SOURCE_MAPPING_EXACT;
    candidate.compilation = 77;
    candidate.tier = 2;
    const auto site = addSite(std::move(candidate), error);
    require(site && !bindCompilation(77, 1, 2, error) && !error.empty(),
        "Unregistered native code acquired optimized source bindings");
    auto target = selection();
    uint64_t session = 0;
    require(FAILED(NacelleBeginSourceDebug(&target, &session)) && !session,
        "Unmaterialized optimized compilation became executable metadata");
    reportCompilationFailure("Intentional optimized compiler failure for status test");
    NacelleOptimizedSourceDebugStatus status;
    require(NacelleGetOptimizedSourceDebugStatus(&status) == S_FALSE &&
        status.compilation_failed && !status.tier2_compilations,
        "Failed asynchronous compilation was reported as optimized success");
    std::puts("OPTIMIZED SOURCE POLICY PASS");
}

void optimizeSourceModule(llvm::Module& module) {
    llvm::LoopAnalysisManager loops;
    llvm::FunctionAnalysisManager functions;
    llvm::CGSCCAnalysisManager calls;
    llvm::ModuleAnalysisManager modules;
#if LLVM_VERSION_MAJOR >= 24
    llvm::PassBuilder passes(module.getContext());
#else
    llvm::PassBuilder passes;
#endif
    passes.registerModuleAnalyses(modules);
    passes.registerCGSCCAnalyses(calls);
    passes.registerFunctionAnalyses(functions);
    passes.registerLoopAnalyses(loops);
    passes.crossRegisterProxies(loops, functions, calls, modules);
    passes.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3).run(module, modules);
    require(!llvm::verifyModule(module, &llvm::errs()), "O3 produced invalid provenance IR");
}

uint32_t emitOptimizedSite(llvm::DIBuilder& debug, llvm::DISubprogram* scope,
    llvm::DIFile* file, llvm::Instruction& instruction, llvm::Value* active,
    llvm::Value* coordinates, llvm::ArrayRef<llvm::Value*> locals,
    nacelle::source_debug::Site site) {
    std::string error;
    const auto id = nacelle::source_debug::addSite(std::move(site), error);
    require(id != 0, error.c_str());
    const auto prefix = "nacelle.site." + std::to_string(id);
    auto location = instruction.getDebugLoc();
    const auto bind = [&](const std::string& suffix, llvm::Value* value) {
        auto* variable = debug.createAutoVariable(scope, prefix + suffix, file, 2,
            debug.createBasicType("bits", value->getType()->getScalarSizeInBits(), llvm::dwarf::DW_ATE_unsigned));
        debug.insertDbgValue(value, variable, debug.createExpression(), location.get(), instruction.getIterator());
    };
    bind(".exec", active);
    bind(".real", active);
    for (unsigned dimension = 0; dimension < 3; ++dimension)
        bind(".coord." + std::to_string(dimension), coordinates);
    for (size_t index = 0; index < locals.size(); ++index)
        bind(".local." + std::to_string(index), locals[index]);
    debug.insertLabel(debug.createLabel(scope, prefix, file, 2, location.getCol(), true, std::nullopt),
        location.get(), instruction.getIterator());
    return id;
}

std::vector<llvm::CallBase*> optimizedCalls(llvm::Module& module) {
    std::vector<llvm::CallBase*> result;
    for (auto& function : module)
        for (auto& block : function)
            for (auto& instruction : block)
                if (auto* call = llvm::dyn_cast<llvm::CallBase>(&instruction);
                    call && call->getCalledFunction() &&
                    call->getCalledFunction()->getName() == "nacelle_source_debug_observe")
                    result.push_back(call);
    return result;
}

bool dependsOn(llvm::Value* value, llvm::Value* input) {
    llvm::SmallPtrSet<llvm::Value*, 32> seen;
    llvm::SmallVector<llvm::Value*, 32> pending{value};
    while (!pending.empty()) {
        auto* current = pending.pop_back_val();
        if (current == input) return true;
        if (!seen.insert(current).second) continue;
        if (auto* instruction = llvm::dyn_cast<llvm::Instruction>(current))
            for (auto& operand : instruction->operands()) pending.push_back(operand.get());
    }
    return false;
}

void optimizedFragmentValidation(llvm::LLVMContext& context) {
    llvm::Module module("fixed-fragments", context);
    llvm::DIBuilder debug(module);
    auto* file = debug.createFile("fragment.hlsl", "");
    debug.createCompileUnit(llvm::dwarf::DW_LANG_C, file, "source-test", true, "", 0);
    auto* scope = debug.createFunction(file, "fragments", "fragments", file, 2,
        debug.createSubroutineType(debug.getOrCreateTypeArray({})), 2,
        llvm::DINode::FlagZero, llvm::DISubprogram::SPFlagDefinition);
    auto* function = llvm::Function::Create(llvm::FunctionType::get(llvm::Type::getVoidTy(context),
        {llvm::PointerType::getUnqual(context), llvm::Type::getInt32Ty(context)}, false),
        llvm::GlobalValue::ExternalLinkage, "fragments", module);
    function->setSubprogram(scope);
    llvm::IRBuilder<> builder(llvm::BasicBlock::Create(context, "entry", function));
    auto* firstLocation = llvm::DILocation::get(context, 2, 1, scope);
    auto* secondLocation = llvm::DILocation::get(context, 3, 1, scope);
    builder.SetCurrentDebugLocation(firstLocation);
    auto* first = builder.CreateStore(function->getArg(1), function->getArg(0));
    builder.SetCurrentDebugLocation(secondLocation);
    auto* second = builder.CreateStore(builder.getInt32(0), function->getArg(0));
    builder.CreateRetVoid();
    auto* scalar = debug.createBasicType("uint", 32, llvm::dwarf::DW_ATE_unsigned);
    llvm::Metadata* count = debug.getOrCreateSubrange(0, 2);
    auto* array = debug.createArrayType(64, 32, scalar, debug.getOrCreateArray({count}));
    auto* pair = debug.createAutoVariable(scope, "pair", file, 2, array);
    auto* unsupported = debug.createAutoVariable(scope, "straddles", file, 2, array);
    debug.insertDbgValue(function->getArg(1), pair,
        debug.createExpression({llvm::dwarf::DW_OP_LLVM_fragment, 32, 32}),
        firstLocation, first->getIterator());
    debug.insertDbgValue(function->getArg(1), unsupported,
        debug.createExpression({llvm::dwarf::DW_OP_LLVM_fragment, 16, 32}),
        firstLocation, first->getIterator());
    debug.insertDbgValue(llvm::PoisonValue::get(builder.getInt64Ty()), pair,
        debug.createExpression(), secondLocation, second->getIterator());
    debug.finalize();
    module.getOrInsertNamedMetadata("dx.source.contents")->addOperand(llvm::MDNode::get(context,
        {llvm::MDString::get(context, "fragment.hlsl"), llvm::MDString::get(context, "first\nsecond\nthird\n")}));
    std::string error;
    const uint32_t threads[3]{1, 1, 1};
    require(nacelle::source_debug::instrument(module, shader, NACELLE_SOURCE_COMPUTE, threads, error),
        error.c_str());
    bool componentAvailable = false, overlappingUnsupported = false, componentKilled = false;
    for (auto& block : *function) for (auto& instruction : block)
        for (auto& record : instruction.getDbgRecordRange()) {
            auto* label = llvm::dyn_cast<llvm::DbgLabelRecord>(&record);
            if (!label) continue;
            auto name = label->getLabel()->getName();
            uint32_t id = 0;
            if (!name.consume_front("nacelle.site.") || name.getAsInteger(10, id)) continue;
            nacelle::source_debug::Site site;
            require(nacelle::source_debug::readSite(id, site), "Fragment site metadata missing");
            for (const auto& local : site.locals) {
                if (std::string_view(local.name) == "pair.bits[32:32]") {
                    if (site.location.line == 2)
                        componentAvailable |= local.kind == NACELLE_SOURCE_UNSIGNED && local.bits == 32;
                    if (site.location.line == 3)
                        componentKilled |= local.kind == NACELLE_SOURCE_UNAVAILABLE;
                }
                if (std::string_view(local.name) == "straddles")
                    overlappingUnsupported |= local.kind == NACELLE_SOURCE_UNAVAILABLE && local.reason[0];
            }
        }
    require(componentAvailable, "Validated fixed scalar fragment was lost before import");
    require(overlappingUnsupported, "Unknown cross-component fragment was fabricated");
    require(componentKilled, "Whole-variable kill retained an obsolete scalar fragment");
    require(!llvm::verifyModule(module, &llvm::errs()), "Invalid inventoried fragment debug records");
    require(optimizedCalls(module).empty(), "Fragment inventory emitted eager observations");
}

void optimizedCompilerValidation() {
    using namespace nacelle::source_debug;
    NacelleOptimizedSourceDebugOptions options;
    options.enabled = 1;
    options.consent = 3;
    require(NacelleConfigureOptimizedSourceDebug(&options) == S_OK && freeze(),
        "Optimized compiler configuration failed");
    std::string error;
    require(addSource(source, "first\nsecond\nthird\n", error), "Compiler source admission failed");
    llvm::LLVMContext context;
    const auto vector = llvm::FixedVectorType::get(llvm::Type::getInt32Ty(context), 4);
    const auto mask = llvm::FixedVectorType::get(llvm::Type::getInt1Ty(context), 4);
    const auto makeFunction = [&](llvm::Module& module, llvm::DIBuilder& debug, llvm::DIFile* file) {
        auto* scope = debug.createFunction(file, "entry", "entry", file, 2,
            debug.createSubroutineType(debug.getOrCreateTypeArray({})), 2,
            llvm::DINode::FlagZero, llvm::DISubprogram::SPFlagDefinition);
        auto* function = llvm::Function::Create(llvm::FunctionType::get(
            llvm::Type::getVoidTy(context), {llvm::PointerType::getUnqual(context), vector, mask,
                llvm::Type::getInt1Ty(context)}, false), llvm::GlobalValue::ExternalLinkage, "entry", module);
        function->setSubprogram(scope);
        return function;
    };
    const auto prepare = [&](llvm::Module& module) {
        require(prepareOptimizedModule(module, error), error.c_str());
        require(!llvm::verifyModule(module, &llvm::errs()), "Invalid packet debug records");
        require(optimizedCalls(module).empty(), "Eager observation obstructs optimization");
    };
    {
        llvm::Module module("bindings", context);
        llvm::DIBuilder debug(module);
        auto* file = debug.createFile("embedded.hlsl", "");
        debug.createCompileUnit(llvm::dwarf::DW_LANG_C, file, "source-test", true, "", 0);
        auto* function = makeFunction(module, debug, file);
        llvm::IRBuilder<> builder(llvm::BasicBlock::Create(context, "entry", function));
        builder.SetCurrentDebugLocation(llvm::DILocation::get(context, 2, 1, function->getSubprogram()));
        auto* one = llvm::ConstantVector::getSplat(llvm::ElementCount::getFixed(4), builder.getInt32(1));
        auto* live = builder.CreateAdd(function->getArg(1), one, "live");
        auto* dead = builder.CreateMul(function->getArg(1),
            llvm::ConstantVector::getSplat(llvm::ElementCount::getFixed(4), builder.getInt32(37)), "dead");
        auto* store = builder.CreateStore(live, function->getArg(0));
        builder.CreateRetVoid();
        auto site = makeSite();
        site.location.column = 1;
        site.locals.push_back(site.locals[0]);
        std::strcpy(site.locals[0].name, "live");
        std::strcpy(site.locals[1].name, "dead");
        emitOptimizedSite(debug, function->getSubprogram(), file, *store,
            function->getArg(2), function->getArg(1), {live, dead}, site);
        emitOptimizedSite(debug, function->getSubprogram(), file, *store,
            function->getArg(2), function->getArg(1), {live, dead}, site);
        site.locals.clear();
        site.location.local_count = 0;
        site.location.column = 3;
        llvm::cast<llvm::Instruction>(dead)->setDebugLoc(
            llvm::DILocation::get(context, 2, 3, function->getSubprogram()));
        emitOptimizedSite(debug, function->getSubprogram(), file, *llvm::cast<llvm::Instruction>(dead),
            function->getArg(2), function->getArg(1), {}, site);
        debug.finalize();
        prepare(module);
        llvm::ValueToValueMapTy remap;
        auto* clone = llvm::CloneFunction(function, remap);
        clone->setName("cloned");
        auto baseline = llvm::CloneModule(module);
        llvm::StripDebugInfo(*baseline);
        for (auto& fn : *baseline) for (auto& block : fn) for (auto& instruction : block)
            instruction.setMetadata("nacelle.source.anchor", nullptr);
        optimizeSourceModule(*baseline);
        optimizeSourceModule(module);
        for (auto& fn : module) for (auto& block : fn) for (auto& instruction : block)
            require(instruction.getOpcode() != llvm::Instruction::Mul, "Debug binding kept dead multiplication alive");
        auto stripped = llvm::CloneModule(module);
        for (auto& fn : *stripped) for (auto& block : fn) for (auto& instruction : block)
            instruction.setMetadata("nacelle.source.anchor", nullptr);
        require(sameProgram(*baseline, *stripped, error), "Debug-only metadata changed the optimized program");
        require(optimizedCalls(module).empty(), "Observation appeared before complete O3");
        require(lowerOptimizedObservations(module, 2, error), error.c_str());
        require(!llvm::verifyModule(module, &llvm::errs()), "Invalid late-observation IR");
        auto calls = optimizedCalls(module);
        require(calls.size() == 4, "Cloned/merged sites lost identity, or dead anchor survived DCE");
        std::set<uint32_t> identities;
        for (auto* call : calls) {
            const auto id = llvm::cast<llvm::ConstantInt>(call->getArgOperand(0))->getZExtValue();
            Site finalized;
            require(identities.insert(id).second && readSite(id, finalized),
                "Cloned native candidates reused a finalized identity");
            require(finalized.compilation == optimizedCompilation(module) && finalized.tier == 2 &&
                finalized.mapping == NACELLE_SOURCE_MAPPING_UNAVAILABLE,
                "Ambiguous merged candidates were attributed without complete origins");
            require(finalized.locals[0].kind == NACELLE_SOURCE_UNSIGNED &&
                finalized.availability[0] == NACELLE_SOURCE_VALUE_AVAILABLE, "Surviving binding unavailable");
            require(finalized.locals[1].kind == NACELLE_SOURCE_UNAVAILABLE &&
                finalized.availability[1] == NACELLE_SOURCE_VALUE_OPTIMIZED_OUT, "Dead binding recreated");
        }
        const auto compilation = optimizedCompilation(module);
        llvm::StripDebugInfo(module);
        require(compilation && optimizedCompilation(module) == compilation,
            "Native privacy stripping erased compiled-site identity");
        require(!lowerOptimizedObservations(module, 2, error), "Repeated observation lowering accepted");
    }
    {
        llvm::Module module("source-predicate", context);
        llvm::DIBuilder debug(module);
        auto* file = debug.createFile("embedded.hlsl", "");
        debug.createCompileUnit(llvm::dwarf::DW_LANG_C, file, "source-test", true, "", 0);
        auto* function = makeFunction(module, debug, file);
        auto* entry = llvm::BasicBlock::Create(context, "entry", function);
        auto* selected = llvm::BasicBlock::Create(context, "selected", function);
        auto* exit = llvm::BasicBlock::Create(context, "exit", function);
        llvm::IRBuilder<> builder(entry);
        builder.SetCurrentDebugLocation(llvm::DILocation::get(context, 2, 2, function->getSubprogram()));
        auto* predicate = builder.CreateSelect(function->getArg(3), function->getArg(2),
            llvm::Constant::getNullValue(mask));
        builder.CreateCondBr(function->getArg(3), selected, exit);
        builder.SetInsertPoint(selected);
        auto* moved = builder.CreateAdd(function->getArg(1),
            llvm::ConstantVector::getSplat(llvm::ElementCount::getFixed(4), builder.getInt32(1)), "speculated");
        builder.CreateBr(exit);
        builder.SetInsertPoint(exit);
        auto* joined = builder.CreatePHI(vector, 2);
        joined->addIncoming(function->getArg(1), entry);
        joined->addIncoming(moved, selected);
        auto* result = builder.CreateSelect(predicate, joined, function->getArg(1));
        builder.SetCurrentDebugLocation(llvm::DILocation::get(context, 2, 1, function->getSubprogram()));
        auto* store = builder.CreateStore(result, function->getArg(0));
        builder.CreateRetVoid();
        auto site = makeSite();
        site.locals.clear();
        site.location.local_count = 0;
        site.location.column = 2;
        emitOptimizedSite(debug, function->getSubprogram(), file, *llvm::cast<llvm::Instruction>(moved),
            predicate, function->getArg(1), {}, site);
        site.location.column = 1;
        emitOptimizedSite(debug, function->getSubprogram(), file, *store,
            function->getArg(2), function->getArg(1), {}, site);
        debug.finalize();
        prepare(module);
        optimizeSourceModule(module);
        bool hoisted = false;
        for (auto& instruction : function->getEntryBlock())
            hoisted |= instruction.getOpcode() == llvm::Instruction::Add;
        require(hoisted, "O3 did not perform the expected source computation hoist");
        require(lowerOptimizedObservations(module, 2, error), error.c_str());
        for (auto* call : optimizedCalls(module)) {
            Site finalized;
            require(readSite(llvm::cast<llvm::ConstantInt>(call->getArgOperand(0))->getZExtValue(), finalized),
                "Missing hoisted-site metadata");
            if (finalized.location.column != 2) continue;
            require(dependsOn(call->getArgOperand(2), function->getArg(2)) &&
                dependsOn(call->getArgOperand(2), function->getArg(3)),
                "Hoisted host execution replaced the actual source execution predicate");
        }
        require(!llvm::verifyModule(module, &llvm::errs()), "Invalid hoisted-site observations");
    }
    for (const bool missingMask : {false, true}) {
        llvm::Module module(missingMask ? "missing-pixel-execution" : "missing-pixel-invocation", context);
        llvm::DIBuilder debug(module);
        auto* file = debug.createFile("embedded.hlsl", "");
        debug.createCompileUnit(llvm::dwarf::DW_LANG_C, file, "source-test", true, "", 0);
        auto* function = makeFunction(module, debug, file);
        llvm::IRBuilder<> builder(llvm::BasicBlock::Create(context, "entry", function));
        auto* location = llvm::DILocation::get(context, 2, 1, function->getSubprogram());
        builder.SetCurrentDebugLocation(location);
        llvm::Value* pointer = function->getArg(0);
        if (!missingMask) pointer = builder.CreateLoad(builder.getPtrTy(), pointer);
        auto* token = builder.CreatePtrToInt(pointer, builder.getInt64Ty());
        auto* invocation = builder.CreateVectorSplat(4, token);
        llvm::Value* execution = function->getArg(2);
        if (missingMask) {
            llvm::Value* components[4];
            for (int lane = 0; lane < 4; ++lane)
                components[lane] = builder.CreateShuffleVector(execution,
                    llvm::ArrayRef<int>{lane, lane, lane, lane});
            execution = builder.CreateOr(builder.CreateOr(components[0], components[1]),
                builder.CreateOr(components[2], components[3]));
        }
        auto* store = builder.CreateStore(function->getArg(1), function->getArg(0));
        builder.CreateRetVoid();
        auto site = makeSite();
        site.location.stage = NACELLE_SOURCE_PIXEL;
        std::strcpy(site.location.shader_id, other.c_str());
        site.location.column = 1;
        site.locals.clear();
        site.location.local_count = 0;
        const auto id = emitOptimizedSite(debug, function->getSubprogram(), file, *store,
            execution, function->getArg(1), {}, site);
        auto* variable = debug.createAutoVariable(function->getSubprogram(),
            "nacelle.site." + std::to_string(id) + ".invocation", file, 2,
            debug.createBasicType("invocation", 64, llvm::dwarf::DW_ATE_unsigned));
        debug.insertDbgValue(invocation, variable, debug.createExpression(), location, store->getIterator());
        if (missingMask) {
            auto* real = debug.createAutoVariable(function->getSubprogram(),
                "nacelle.site." + std::to_string(id) + ".real", file, 2, nullptr);
            debug.insertDbgValue(function->getArg(2), real, debug.createExpression(), location, store->getIterator());
        }
        debug.finalize();
        prepare(module);
        optimizeSourceModule(module);
        for (auto& block : *function) for (auto& instruction : block)
            require(!llvm::isa<llvm::LoadInst>(instruction), "Debug-only invocation load obstructed O3");
        require(!lowerOptimizedObservations(module, 2, error) &&
            error.find(missingMask ? "masks=0" : "invocations=0") != std::string::npos &&
            error.find(missingMask ? "invocations=1" : "masks=1") != std::string::npos &&
            optimizedCalls(module).empty() && !optimizedCompilation(module),
            "Surviving formal/real-mask input substituted for a killed invocation/execution binding");
    }
    for (unsigned scenario = 0; scenario < 4; ++scenario) {
        llvm::Module module("pixel-entry-context", context);
        llvm::DIBuilder debug(module);
        auto* file = debug.createFile("embedded.hlsl", "");
        debug.createCompileUnit(llvm::dwarf::DW_LANG_C, file, "source-test", true, "", 0);
        auto* scope = debug.createFunction(file, "pixel", "pixel", file, 2,
            debug.createSubroutineType(debug.getOrCreateTypeArray({})), 2,
            llvm::DINode::FlagZero, llvm::DISubprogram::SPFlagDefinition);
        auto* function = llvm::Function::Create(llvm::FunctionType::get(
            llvm::Type::getVoidTy(context), {llvm::PointerType::getUnqual(context)}, false),
            llvm::GlobalValue::ExternalLinkage, "nacelle_ps_context", module);
        function->setSubprogram(scope);
        llvm::IRBuilder<> builder(llvm::BasicBlock::Create(context, "entry", function));
        auto* location = llvm::DILocation::get(context, 2, 1, scope);
        builder.SetCurrentDebugLocation(location);
        const auto field = [&](size_t offset, llvm::Type* type) {
            return builder.CreateLoad(type, builder.CreateGEP(builder.getInt8Ty(),
                function->getArg(0), builder.getInt64(offset)));
        };
        auto* batch = builder.CreateZExt(field(offsetof(nacelle::PixelBatchArgs, batchCount),
            builder.getInt32Ty()), builder.getInt64Ty());
        auto* offset = builder.CreateMul(batch, builder.getInt64(16));
        auto* floatVector = llvm::FixedVectorType::get(builder.getFloatTy(), 4);
        const auto coordinate = [&](size_t offsetInArgs) {
            auto* pointer = field(offsetInArgs, builder.getPtrTy());
            return builder.CreateLoad(floatVector, builder.CreateGEP(builder.getInt8Ty(), pointer, offset));
        };
        auto* x = coordinate(offsetof(nacelle::PixelBatchArgs, pixelX));
        auto* y = coordinate(offsetof(nacelle::PixelBatchArgs, pixelY));
        auto* out = field(offsetof(nacelle::PixelBatchArgs, tileColorSOA), builder.getPtrTy());
        auto* result = scenario == 3 ? llvm::Constant::getNullValue(floatVector) : builder.CreateFAdd(x, y);
        auto* store = builder.CreateStore(result, out);
        builder.CreateRetVoid();
        auto site = makeSite();
        site.location.stage = NACELLE_SOURCE_PIXEL;
        std::strcpy(site.location.shader_id, other.c_str());
        site.location.column = 1;
        site.locals.clear();
        site.location.local_count = 0;
        const auto id = emitOptimizedSite(debug, scope, file, *store,
            llvm::PoisonValue::get(mask), llvm::Constant::getNullValue(vector), {}, site);
        const auto bind = [&](const char* suffix, llvm::Value* value) {
            auto* variable = debug.createAutoVariable(scope,
                "nacelle.site." + std::to_string(id) + suffix, file, 2, nullptr);
            debug.insertDbgValue(value, variable, debug.createExpression(), location, store->getIterator());
        };
        llvm::IRBuilder<> coordinates(store);
        bind(".coord.0", coordinates.CreateFPToUI(x, vector));
        bind(".coord.1", coordinates.CreateFPToUI(y, vector));
        bind(".real", llvm::PoisonValue::get(mask));
        bind(".exec.entry", builder.getInt32(scenario != 1));
        bind(".real.entry", builder.getInt32(scenario != 2));
        debug.finalize();
        prepare(module);
        optimizeSourceModule(module);
        const bool lowered = lowerOptimizedObservations(module, 2, error);
        require(lowered == (scenario == 0),
            "Entry ABI context replaced a branch/discard predicate or a killed semantic index");
        require(optimizedCalls(module).size() == (scenario == 0 ? 1 : 0),
            "Unavailable pixel context acquired an observation");
        require(!llvm::verifyModule(module, &llvm::errs()), "Invalid late pixel-context IR");
    }
    {
        float x[16], y[16];
        for (unsigned lane = 0; lane < 16; ++lane) { x[lane] = lane + 5; y[lane] = lane + 6; }
        float first[3]{}, second[3]{};
        uint32_t coverage[]{0x10, 0x21}, indices[]{0, 0, 1, 0};
        nacelle::PixelTriData triangles[2];
        triangles[0].triInvW = first;
        triangles[1].triInvW = second;
        nacelle::PixelBatchArgs args;
        args.pixelX = x;
        args.pixelY = y;
        args.triInvW = first;
        args.batchCount = 2;
        args.coverageMasks = coverage;
        uint32_t xyz[48]{}, masks[2]{};
        uint64_t invocations[16]{};
        require(nacelle_source_debug_read_pixel_context(&args, 1, 8, xyz, masks, invocations) &&
            masks[0] == 255 && masks[1] == 0x21 && xyz[0] == 13 && xyz[1] == 14 &&
            invocations[0] == reinterpret_cast<uintptr_t>(first),
            "Uniform pixel ABI context lost batch coordinates, entry helpers or primitive identity");
        args.triData = triangles;
        args.quadTriIndices = indices;
        require(nacelle_source_debug_read_pixel_context(&args, 1, 8, xyz, masks, invocations) &&
            invocations[0] == reinterpret_cast<uintptr_t>(second) &&
            invocations[4] == reinterpret_cast<uintptr_t>(first),
            "Packed pixel ABI context conflated primitive identities");
        coverage[1] = 1;
        indices[3] = UINT32_MAX;
        require(nacelle_source_debug_read_pixel_context(&args, 1, 8, xyz, masks, invocations) &&
            masks[0] == 15 && masks[1] == 1 && invocations[4] == 0,
            "Padding quad read a nonexistent primitive descriptor");
        require(!nacelle_source_debug_read_pixel_context(&args, 2, 8, xyz, masks, invocations) &&
            !masks[0] && !masks[1], "Out-of-batch pixel context was available");
        args.sampleCoverageMasks = coverage;
        require(!nacelle_source_debug_read_pixel_context(&args, 0, 8, xyz, masks, invocations),
            "Unsupported multisample context became single-sample provenance");
    }
    optimizedFragmentValidation(context);
    std::puts("OPTIMIZED SOURCE COMPILER PASS");
}
void programValidation() {
    llvm::LLVMContext executableContext, debugContext;
    llvm::Module executable("executable", executableContext), debug("embedded-debug", debugContext);
    const auto populate = [](llvm::Module& module, unsigned constant) {
        auto& context = module.getContext();
        llvm::IRBuilder<> builder(context);
        auto* function = llvm::Function::Create(
            llvm::FunctionType::get(builder.getInt32Ty(), false),
            llvm::GlobalValue::ExternalLinkage, "main", module);
        builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", function));
        builder.CreateRet(builder.getInt32(constant));
        return function;
    };
    populate(executable, 7);
    auto* debugFunction = populate(debug, 7);
    debug.getOrInsertNamedMetadata("dx.source.contents")->addOperand(
        llvm::MDNode::get(debugContext, llvm::MDString::get(debugContext, "debug-only")));
    std::string error;
    require(nacelle::source_debug::sameProgram(executable, debug, error),
            "Debug-only container metadata changed executable identity");
    debugFunction->back().getTerminator()->setOperand(0, llvm::ConstantInt::get(
        llvm::Type::getInt32Ty(debugContext), 8));
    require(!nacelle::source_debug::sameProgram(executable, debug, error) && !error.empty(),
            "Different embedded executable code was admitted");
}
void metadataValidation() {
    using namespace nacelle::source_debug;
    std::string error;
    require(!addSource(std::string(64, 'A'), "text", error) && !error.empty(), "Uppercase identity accepted");
    require(addSource(source, "first\nsecond\n", error), "Source insertion failed");
    require(addSource(source, "first\nsecond\n", error), "Idempotent source insertion failed");
    require(!addSource(source, "replacement", error), "Immutable source replaced");
    auto invalid = makeSite();
    invalid.location.line = 99;
    require(!addSite(invalid, error), "Out of range source line accepted");
    invalid = makeSite();
    std::memset(invalid.location.label, 'x', sizeof(invalid.location.label));
    require(!addSite(invalid, error), "Unterminated label accepted");
    invalid = makeSite();
    invalid.locals[0].name[0] = '\n';
    require(!addSite(invalid, error), "Control character in local name accepted");
    invalid = makeSite();
    invalid.locals.resize(65);
    invalid.location.local_count = 65;
    require(!addSite(invalid, error), "Unbounded locals accepted");
    require(addSite(makeSite(), error) == 1, "Site insertion failed");
    auto target = selection();
    uint64_t unpublished = 0;
    require(FAILED(NacelleBeginSourceDebug(&target, &unpublished)) && !unpublished,
            "Metadata from an unsuccessful/unpublished compilation was admitted");
    publishShader(shader);
    NacelleSourceDebugSite site;
    require(NacelleGetSourceDebugSite(1, &site) == S_OK && site.line == 2, "Site lookup failed");
    require(NacelleGetSourceDebugSite(2, &site) == S_FALSE && !site.id, "Unavailable site not cleared");
    auto otherLine = makeSite();
    otherLine.location.line = 1;
    require(addSite(std::move(otherLine), error) == 2, "Second executable line insertion failed");
    NacelleSourceDebugLocal local;
    require(NacelleGetSourceDebugLocal(1, 0, &local) == S_OK && local.bits == 32, "Local lookup failed");
    require(NacelleGetSourceDebugLocal(1, 1, &local) == S_FALSE && !local.bits, "Unavailable local not cleared");
    uint64_t bytes = 0;
    require(NacelleGetSourceDebugSource(source.c_str(), nullptr, &bytes) == S_OK && bytes == 13,
            "Source size query failed");
    char buffer[16];
    std::memset(buffer, '!', sizeof(buffer));
    bytes = 1;
    require(FAILED(NacelleGetSourceDebugSource(source.c_str(), buffer, &bytes)) &&
            bytes == 13 && buffer[0] == '!', "Short buffer overwritten");
    require(NacelleGetSourceDebugSource(source.c_str(), buffer, &bytes) == S_OK &&
            std::string_view(buffer, bytes) == "first\nsecond\n" && buffer[13] == '!',
            "Source bytes or buffer bound changed");
    require(NacelleGetSourceDebugSource(other.c_str(), nullptr, &bytes) == S_FALSE && !bytes,
            "Missing source not unavailable");
}
}

int main(int argc, char** argv) {
    try {
        using namespace nacelle::source_debug;
        // A disabled submission must not become tracked when configuration is enabled later.
        Work disabledWork;
        const std::string_view mode = argc > 1 ? argv[1] : "";
        if (mode == "optimized") { optimizedPolicyValidation(); return 0; }
        if (mode == "optimized-compiler") { optimizedCompilerValidation(); return 0; }
        NacelleSourceDebugOptions options;
        require(FAILED(NacelleConfigureSourceDebug(nullptr)), "Null configuration accepted");
        options.version = 2;
        require(FAILED(NacelleConfigureSourceDebug(&options)), "Unknown configuration version accepted");
        options.version = 1;
        options.enabled = 2;
        require(FAILED(NacelleConfigureSourceDebug(&options)), "Invalid enabled flag accepted");
        options.enabled = 1;
        for (uint32_t consent = 0; consent < 3; ++consent) {
            options.consent = consent;
            require(FAILED(NacelleConfigureSourceDebug(&options)), "Missing privacy consent accepted");
        }
        options.consent = 3;
        options.source_bytes = 16ull * 1024 * 1024 + 1;
        require(FAILED(NacelleConfigureSourceDebug(&options)), "Excess source budget accepted");
        options.source_bytes = 8ull * 1024 * 1024;
        options.metadata_bytes = 32ull * 1024 * 1024 + 1;
        require(FAILED(NacelleConfigureSourceDebug(&options)), "Excess metadata budget accepted");
        options.metadata_bytes = mode == "metadata-budget" ? 1 : 16ull * 1024 * 1024;
        if (mode == "source-budget") options.source_bytes = 1;
        require(NacelleConfigureSourceDebug(&options) == S_OK && enabled() &&
                nacelle::native_debug::enabled(), "Configuration did not enable native debug");
        require(freeze() && freeze(), "Freeze is not idempotent");
        require(FAILED(NacelleConfigureSourceDebug(&options)), "Late configuration accepted");
        std::string error;
        if (mode == "source-budget") {
            require(!addSource(source, "x", error) && error.find("budget") != std::string::npos,
                    "Source budget not enforced");
            std::puts("SOURCE DEBUG SOURCE BUDGET PASS");
            return 0;
        }
        if (mode == "metadata-budget") {
            require(addSource(source, "first\nsecond\n", error), "Budget source insertion failed");
            require(!addSite(makeSite(), error) && error.find("budget") != std::string::npos,
                    "Metadata record budget not enforced");
            std::puts("SOURCE DEBUG METADATA BUDGET PASS");
            return 0;
        }
        auto target = selection();
        uint64_t id = 0;
        require(FAILED(NacelleBeginSourceDebug(&target, &id)), "Missing shader metadata accepted");
        metadataValidation();
        programValidation();
        {
            Work uncaptured;
            require(FAILED(NacelleBeginSourceDebug(&target, &id)), "Begin accepted non-quiescent work");
        }
        id = begin();
        uint64_t overlap = 0;
        require(FAILED(NacelleBeginSourceDebug(&target, &overlap)) && !overlap, "Overlapping session accepted");
        require(FAILED(NacelleSourceDebugBreakpoint(id, source.c_str(), 3)), "Non-executable breakpoint accepted");
        require(FAILED(NacelleSourceDebugBreakpoint(id, other.c_str(), 2)), "Other source breakpoint accepted");
        require(NacelleSourceDebugBreakpoint(id, source.c_str(), 2) == S_OK, "Executable breakpoint rejected");
        NacelleSourceDebugSnapshot snapshot;
        require(NacelleGetSourceDebugSnapshot(id, &snapshot) == S_FALSE && !snapshot.stop,
                "Unobserved snapshot fabricated");
        require(FAILED(NacelleControlSourceDebug(id, 1, NACELLE_SOURCE_STEP)), "Stale stop accepted");
        {
            Work work;
            Operation operation(shader);
            uint32_t xyz[6]{7, 0, 0, 8, 0, 0};
            uint64_t values[2]{42, 43};
            nacelle_source_debug_observe(1, 2, 2, 3, xyz, values);
            require(NacelleGetSourceDebugSnapshot(id, &snapshot) == S_FALSE, "Inactive target stopped");
            // This host function is deliberately NOT registered JIT code. It must not publish
            // a native identity belonging to an unrelated shader or fabricate a successful stop.
            nacelle_source_debug_observe(1, 2, 3, 3, xyz, values);
        }
        NacelleSourceDebugSummary summary;
        require(NacelleEndSourceDebug(id, &summary) == S_FALSE && summary.cancelled &&
                summary.target_seen && !summary.stops, "Missing native identity looked successful");
        require(FAILED(NacelleEndSourceDebug(id, &summary)), "Stale session ended twice");
        require(FAILED(NacelleGetSourceDebugSnapshot(id, &snapshot)), "Stale snapshot session accepted");
        const auto previous = id;
        id = begin();
        require(id > previous, "Session identity reused");
        require(FAILED(NacelleControlSourceDebug(previous, 0, NACELLE_SOURCE_CANCEL)), "Stale cancellation accepted");
        require(NacelleControlSourceDebug(id, 0, NACELLE_SOURCE_CANCEL) == S_OK, "Cancellation failed");
        require(NacelleEndSourceDebug(id, &summary) == S_FALSE && summary.cancelled,
                "Intentional cancellation looked successful");

        id = begin();
        require(NacelleSourceDebugBreakpoint(id, source.c_str(), 1) == S_OK, "Second executable breakpoint rejected");
        {
            Work work;
            Operation operation(shader);
            uint32_t xyz[3]{7, 0, 0};
            uint64_t value = 42;
            nacelle_source_debug_observe(1, 1, 1, 1, xyz, &value);
        }
        require(SUCCEEDED(NacelleEndSourceDebug(id, &summary)) && summary.target_seen &&
                !summary.cancelled && !summary.stops,
                "Initial stepping ignored the configured breakpoint");

        id = begin();
        require(NacelleSourceDebugBreakpoint(id, source.c_str(), 1) == S_OK, "Operation breakpoint rejected");
        {
            Work work;
            uint32_t xyz[3]{7, 0, 0};
            uint64_t value = 42;
            { Operation first(shader); nacelle_source_debug_observe(1, 1, 1, 1, xyz, &value); }
            { Operation later(shader); nacelle_source_debug_observe(2, 1, 1, 1, xyz, &value); }
        }
        require(NacelleEndSourceDebug(id, &summary) == S_OK && !summary.cancelled && !summary.stops,
                "Observation migrated to a subsequent shader operation");

        auto pixelSite = makeSite();
        std::strcpy(pixelSite.location.shader_id, other.c_str());
        pixelSite.location.stage = NACELLE_SOURCE_PIXEL;
        require(addSite(pixelSite, error) == 3, "Pixel site insertion failed");
        pixelSite.location.line = 1;
        require(addSite(pixelSite, error) == 4, "Pixel continuation site insertion failed");
        publishShader(other);
        auto pixelTarget = selection();
        std::strcpy(pixelTarget.shader_id, other.c_str());
        require(NacelleBeginSourceDebug(&pixelTarget, &id) == S_OK, "Pixel session failed");
        require(NacelleSourceDebugBreakpoint(id, source.c_str(), 1) == S_OK, "Pixel breakpoint failed");
        {
            Work work;
            Operation operation(other);
            uint32_t xyz[3]{7, 0, 0};
            uint64_t value = 42, primitive = 11;
            nacelle_source_debug_observe(3, 1, 1, 1, xyz, &value, &primitive);
            primitive = 22;
            nacelle_source_debug_observe(4, 1, 1, 1, xyz, &value, &primitive);
        }
        require(NacelleEndSourceDebug(id, &summary) == S_OK && !summary.cancelled && !summary.stops,
                "Pixel observation migrated to an overlapping primitive");

        id = begin();
        {
            Work work;
            Operation operation(shader);
            uint32_t xyz[3]{7, 0, 0};
            uint64_t value = 42;
            nacelle_source_debug_observe(1, 1, 0, 1, xyz, &value);
        }
        require(NacelleEndSourceDebug(id, &summary) == S_FALSE && summary.target_seen &&
                !summary.cancelled && !summary.stops,
                "Inactive target was not reported explicitly or looked like a successful step");

        id = begin();
        std::promise<void> entered, release;
        auto released = release.get_future();
        std::atomic<int> result{0};
        std::thread worker([&] {
            Work work;
            entered.set_value();
            released.wait();
            result = 42;
        });
        entered.get_future().wait();
        require(NacelleControlSourceDebug(id, 0, NACELLE_SOURCE_CANCEL) == S_OK, "In-flight cancellation failed");
        auto ending = std::async(std::launch::async, [&] { return NacelleEndSourceDebug(id, &summary); });
        const bool waited = ending.wait_for(std::chrono::milliseconds(25)) == std::future_status::timeout;
        release.set_value();
        worker.join();
        require(waited && ending.get() == S_FALSE && result == 42 && summary.cancelled,
                "End failed to drain original cancelled execution");

        id = begin();
        int originalResult = 1;
        {
            Work first;
            std::thread concurrent([&] { Work rejected; originalResult += 6; });
            concurrent.join();
            originalResult *= 6;
        }
        require(NacelleEndSourceDebug(id, &summary) == S_FALSE && summary.concurrent_work &&
                originalResult == 42, "Concurrent work changed original execution or looked successful");
        id = begin();
        { Work first; }
        { Work subsequent; }
        require(NacelleEndSourceDebug(id, &summary) == S_FALSE && !summary.concurrent_work &&
                !summary.cancelled && !summary.stops,
                "Subsequent readback submission was captured or rejected");
        std::puts("SOURCE DEBUG RUNTIME PASS");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "source_debug_test: %s\n", error.what());
        return 1;
    }
}
