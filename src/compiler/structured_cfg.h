#pragma once

#include <memory>
#include <string>
#include <vector>

namespace llvm {
class BasicBlock;
class Function;
class Loop;
class Value;
class raw_ostream;
} // namespace llvm

namespace nacelle {

struct RegionNode {
    enum class Kind {
        Block,
        Seq,
        If,
        While,
    };

    Kind kind;
    virtual ~RegionNode() = default;

protected:
    explicit RegionNode(Kind k) : kind(k) {}
};

struct BlockNode final : RegionNode {
    llvm::BasicBlock* bb = nullptr;
    explicit BlockNode(llvm::BasicBlock* b) : RegionNode(Kind::Block), bb(b) {}
    static bool classof(const RegionNode* n) { return n->kind == Kind::Block; }
};

struct SeqNode final : RegionNode {
    std::vector<std::unique_ptr<RegionNode>> children;
    SeqNode() : RegionNode(Kind::Seq) {}
    static bool classof(const RegionNode* n) { return n->kind == Kind::Seq; }
};

struct IfNode final : RegionNode {
    llvm::BasicBlock* headerBB = nullptr;
    llvm::Value* cond = nullptr;
    std::unique_ptr<SeqNode> thenRegion;
    std::unique_ptr<SeqNode> elseRegion;
    llvm::BasicBlock* mergeBB = nullptr;
    // Set when one arm branches directly to the enclosing loop's exit block.
    bool thenExitsLoop = false;
    bool elseExitsLoop = false;

    IfNode() : RegionNode(Kind::If) {}
    static bool classof(const RegionNode* n) { return n->kind == Kind::If; }
};

struct WhileNode final : RegionNode {
    llvm::BasicBlock* headerBB = nullptr;
    llvm::BasicBlock* preheaderBB = nullptr;
    llvm::BasicBlock* latchBB = nullptr;
    llvm::BasicBlock* exitBB = nullptr;
    llvm::Loop* loop = nullptr;
    std::unique_ptr<SeqNode> body;

    WhileNode() : RegionNode(Kind::While) {}
    static bool classof(const RegionNode* n) { return n->kind == Kind::While; }
};

struct StructuredFunction {
    llvm::Function* fn = nullptr;
    std::unique_ptr<SeqNode> root;

    void print(llvm::raw_ostream& os) const;
};

/// Unifies function exits, lowers switches, runs LLVM's `StructurizeCFGPass`,
/// and builds a structured region tree (`SeqNode` / `BlockNode` / `IfNode` / `WhileNode`)
/// over `F`'s `BasicBlock`s.
bool structurizeAndBuildRegionTree(
    llvm::Function& F,
    StructuredFunction& outTree,
    std::string& outError);

} // namespace nacelle
