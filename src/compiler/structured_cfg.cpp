#include "structured_cfg.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/DominanceFrontier.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/PostDominators.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/CFG.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Scalar/StructurizeCFG.h>
#include <llvm/Transforms/Utils/LowerSwitch.h>
#include <llvm/Transforms/Utils/UnifyLoopExits.h>

namespace nacelle {

namespace {

void unifyFunctionExits(llvm::Function& F) {
    llvm::SmallVector<llvm::BasicBlock*, 8> exitBlocks;
    for (llvm::BasicBlock& bb : F) {
        auto* term = bb.getTerminator();
        if (llvm::isa<llvm::ReturnInst>(term) || llvm::isa<llvm::UnreachableInst>(term)) {
            exitBlocks.push_back(&bb);
        }
    }

    // Replace any UnreachableInst with a ReturnInst (returning poison if non-void).
    llvm::Type* retTy = F.getReturnType();
    for (llvm::BasicBlock* bb : exitBlocks) {
        if (auto* unreach = llvm::dyn_cast<llvm::UnreachableInst>(bb->getTerminator())) {
            llvm::IRBuilder<> b(unreach);
            if (retTy->isVoidTy()) {
                b.CreateRetVoid();
            } else {
                b.CreateRet(llvm::PoisonValue::get(retTy));
            }
            unreach->eraseFromParent();
        }
    }

    if (exitBlocks.size() <= 1) {
        return;
    }

    llvm::BasicBlock* unifiedExit = llvm::BasicBlock::Create(F.getContext(), "unified.exit", &F);
    llvm::IRBuilder<> exitB(unifiedExit);
    llvm::PHINode* retPhi = nullptr;
    if (!retTy->isVoidTy()) {
        retPhi = exitB.CreatePHI(retTy, static_cast<unsigned>(exitBlocks.size()), "ret.val");
    }

    for (llvm::BasicBlock* bb : exitBlocks) {
        auto* ret = llvm::cast<llvm::ReturnInst>(bb->getTerminator());
        if (retPhi && ret->getReturnValue()) {
            retPhi->addIncoming(ret->getReturnValue(), bb);
        }
        llvm::IRBuilder<> b(ret);
        b.CreateBr(unifiedExit);
        ret->eraseFromParent();
    }

    if (retPhi) {
        exitB.CreateRet(retPhi);
    } else {
        exitB.CreateRetVoid();
    }
}

class RegionTreeBuilder {
public:
    RegionTreeBuilder(
        llvm::DominatorTree& dt,
        llvm::PostDominatorTree& pdt,
        llvm::LoopInfo& li)
        : dt_(dt), pdt_(pdt), li_(li) {}

    std::unique_ptr<SeqNode> buildSeq(
        llvm::BasicBlock* startBB,
        llvm::BasicBlock* stopBB,
        llvm::Loop* curLoop,
        llvm::BasicBlock* loopExitBB) {
        auto seq = std::make_unique<SeqNode>();
        llvm::BasicBlock* cur = startBB;

        while (cur && cur != stopBB) {
            if (curLoop && cur == loopExitBB) {
                break;
            }

            // Check if `cur` is the header of an inner loop not yet entered.
            llvm::Loop* L = li_.getLoopFor(cur);
            if (L && L != curLoop && L->getHeader() == cur) {
                auto wn = std::make_unique<WhileNode>();
                wn->headerBB = cur;
                wn->preheaderBB = L->getLoopPreheader();
                wn->latchBB = L->getLoopLatch();
                wn->exitBB = L->getUniqueExitBlock();
                wn->loop = L;
                wn->body = buildSeq(cur, /*stopBB=*/wn->exitBB, L, wn->exitBB);
                llvm::BasicBlock* nextBB = wn->exitBB;
                seq->children.push_back(std::move(wn));
                cur = nextBB;
                continue;
            }

            seq->children.push_back(std::make_unique<BlockNode>(cur));

            auto* term = cur->getTerminator();
            if (llvm::isa<llvm::ReturnInst>(term) || llvm::isa<llvm::UnreachableInst>(term)) {
                cur = nullptr;
                break;
            }

            auto* br = llvm::dyn_cast<llvm::BranchInst>(term);
            if (!br) {
                cur = nullptr;
                break;
            }

            if (br->isUnconditional()) {
                llvm::BasicBlock* succ = br->getSuccessor(0);
                if (succ == stopBB || (curLoop && (succ == curLoop->getHeader() || succ == loopExitBB))) {
                    cur = nullptr;
                } else {
                    cur = succ;
                }
                continue;
            }

            // Conditional branch
            llvm::BasicBlock* tBB = br->getSuccessor(0);
            llvm::BasicBlock* fBB = br->getSuccessor(1);

            if (curLoop) {
                llvm::BasicBlock* hdr = curLoop->getHeader();
                bool tIsBack = (tBB == hdr);
                bool fIsBack = (fBB == hdr);
                bool tIsExit = (tBB == loopExitBB);
                bool fIsExit = (fBB == loopExitBB);

                // Loop latch: branches between header and exit (or backedge and stop).
                if ((tIsBack && fIsExit) || (fIsBack && tIsExit)) {
                    cur = nullptr;
                    break;
                }
                // Guarded backedge or guarded exit inside loop:
                if (tIsExit && !fIsExit) {
                    auto ifn = std::make_unique<IfNode>();
                    ifn->headerBB = cur;
                    ifn->cond = br->getCondition();
                    ifn->thenExitsLoop = true;
                    ifn->thenRegion = std::make_unique<SeqNode>();
                    ifn->elseRegion = fIsBack ? std::make_unique<SeqNode>()
                                              : buildSeq(fBB, stopBB, curLoop, loopExitBB);
                    ifn->mergeBB = stopBB;
                    seq->children.push_back(std::move(ifn));
                    cur = nullptr;
                    break;
                }
                if (fIsExit && !tIsExit) {
                    auto ifn = std::make_unique<IfNode>();
                    ifn->headerBB = cur;
                    ifn->cond = br->getCondition();
                    ifn->elseExitsLoop = true;
                    ifn->thenRegion = tIsBack ? std::make_unique<SeqNode>()
                                              : buildSeq(tBB, stopBB, curLoop, loopExitBB);
                    ifn->elseRegion = std::make_unique<SeqNode>();
                    ifn->mergeBB = stopBB;
                    seq->children.push_back(std::move(ifn));
                    cur = nullptr;
                    break;
                }
            }

            llvm::BasicBlock* mergeBB = pdt_.findNearestCommonDominator(tBB, fBB);
            if (mergeBB && stopBB && mergeBB != stopBB && pdt_.dominates(stopBB, mergeBB)) {
                mergeBB = stopBB;
            }
            if (!mergeBB && stopBB) {
                mergeBB = stopBB;
            }

            auto ifn = std::make_unique<IfNode>();
            ifn->headerBB = cur;
            ifn->cond = br->getCondition();
            ifn->mergeBB = mergeBB;

            auto buildArm = [&](llvm::BasicBlock* target) -> std::unique_ptr<SeqNode> {
                if (target == mergeBB || target == stopBB) {
                    return std::make_unique<SeqNode>();
                }
                if (curLoop && (target == curLoop->getHeader() || target == loopExitBB)) {
                    return std::make_unique<SeqNode>();
                }
                return buildSeq(target, mergeBB, curLoop, loopExitBB);
            };

            ifn->thenRegion = buildArm(tBB);
            ifn->elseRegion = buildArm(fBB);
            seq->children.push_back(std::move(ifn));

            if (mergeBB == stopBB || (curLoop && (mergeBB == curLoop->getHeader() || mergeBB == loopExitBB))) {
                cur = nullptr;
            } else {
                cur = mergeBB;
            }
        }

        return seq;
    }

private:
    llvm::DominatorTree& dt_;
    llvm::PostDominatorTree& pdt_;
    llvm::LoopInfo& li_;
};

void printIndent(llvm::raw_ostream& os, int indent) {
    for (int i = 0; i < indent; ++i) os << "  ";
}

void printNode(const RegionNode& node, llvm::raw_ostream& os, int indent) {
    switch (node.kind) {
    case RegionNode::Kind::Block: {
        const auto& bn = llvm::cast<BlockNode>(node);
        printIndent(os, indent);
        os << "block %" << bn.bb->getName() << "\n";
        break;
    }
    case RegionNode::Kind::Seq: {
        const auto& sn = llvm::cast<SeqNode>(node);
        for (const auto& child : sn.children) {
            printNode(*child, os, indent);
        }
        break;
    }
    case RegionNode::Kind::If: {
        const auto& in = llvm::cast<IfNode>(node);
        printIndent(os, indent);
        os << "if ";
        in.cond->printAsOperand(os, false);
        os << " (header=%" << in.headerBB->getName();
        if (in.mergeBB) os << ", merge=%" << in.mergeBB->getName();
        os << ") {\n";
        if (in.thenExitsLoop) {
            printIndent(os, indent + 1);
            os << "exit_loop\n";
        } else if (in.thenRegion) {
            printNode(*in.thenRegion, os, indent + 1);
        }
        if (in.elseExitsLoop || (in.elseRegion && !in.elseRegion->children.empty())) {
            printIndent(os, indent);
            os << "} else {\n";
            if (in.elseExitsLoop) {
                printIndent(os, indent + 1);
                os << "exit_loop\n";
            } else {
                printNode(*in.elseRegion, os, indent + 1);
            }
        }
        printIndent(os, indent);
        os << "}\n";
        break;
    }
    case RegionNode::Kind::While: {
        const auto& wn = llvm::cast<WhileNode>(node);
        printIndent(os, indent);
        os << "while (header=%" << wn.headerBB->getName();
        if (wn.exitBB) os << ", exit=%" << wn.exitBB->getName();
        os << ") {\n";
        if (wn.body) {
            printNode(*wn.body, os, indent + 1);
        }
        printIndent(os, indent);
        os << "}\n";
        break;
    }
    }
}

} // namespace

void StructuredFunction::print(llvm::raw_ostream& os) const {
    if (!fn || !root) return;
    os << "structured_fn @" << fn->getName() << " {\n";
    printNode(*root, os, 1);
    os << "}\n";
}

bool structurizeAndBuildRegionTree(
    llvm::Function& F,
    StructuredFunction& outTree,
    std::string& outError) {
    if (F.isDeclaration() || F.empty()) {
        outError = "Cannot structurize empty or declaration function: " + F.getName().str();
        return false;
    }

    unifyFunctionExits(F);

    llvm::LoopAnalysisManager LAM;
    llvm::FunctionAnalysisManager FAM;
    llvm::CGSCCAnalysisManager CGAM;
    llvm::ModuleAnalysisManager MAM;

    llvm::PassBuilder PB;
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

    llvm::FunctionPassManager FPM;
    FPM.addPass(llvm::LowerSwitchPass());
    FPM.addPass(llvm::UnifyLoopExitsPass());
    FPM.addPass(llvm::StructurizeCFGPass(/*SkipUniformRegions=*/false));
    FPM.run(F, FAM);

    // Recompute analyses cleanly after CFG transformation.
    FAM.invalidate(F, llvm::PreservedAnalyses::none());
    auto& dt = FAM.getResult<llvm::DominatorTreeAnalysis>(F);
    auto& pdt = FAM.getResult<llvm::PostDominatorTreeAnalysis>(F);
    auto& li = FAM.getResult<llvm::LoopAnalysis>(F);

    RegionTreeBuilder builder(dt, pdt, li);
    outTree.fn = &F;
    outTree.root = builder.buildSeq(&F.getEntryBlock(), /*stopBB=*/nullptr, /*curLoop=*/nullptr, /*loopExitBB=*/nullptr);
    return true;
}

} // namespace nacelle
