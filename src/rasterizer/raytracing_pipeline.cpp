#include "rasterizer/raytracing_pipeline.h"
#include "rasterizer/raytracing_bvh.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <vector>

namespace nacelle {
namespace {

constexpr uint32_t kRayFlagSkipClosestHitShader = 0x08u;

struct DxrActiveTrace {
    uint32_t rayFlags = 0;
    float tMin = 0.0f;
    float tMax = 0.0f;
    float worldOrigin[3]{};
    float worldDir[3]{};
    PipelineHitResult hit{};

    // AnyHit / Intersection state
    int32_t anyHitAction = 0; // 0=accept, 1=ignore, 2=accept_and_end_search
    void* payloadPtr = nullptr;
    uint32_t rayContrib = 0;
    uint32_t geomMult = 0;

    // Intersection state
    bool reportAccepted = false;
    bool reportEndSearch = false;
    uint32_t reportHitKind = 0;
    float reportT = 0.0f;
    std::array<uint8_t, 64> reportAttr{};
};

struct DxrThreadContext {
    const DriverRootContext* rootCtx = nullptr;
    const CompiledDxrStateObject* stateObj = nullptr;
    D3D12_DISPATCH_RAYS_DESC desc{};
    uint32_t curRayIdx[3]{0, 0, 0};
    DxrActiveTrace* activeTrace = nullptr;
};

thread_local DxrThreadContext* t_dxrCtx = nullptr;

const NacelleShaderIdentifier* readSbtRecord(
    D3D12_GPU_VIRTUAL_ADDRESS startAddr,
    UINT64 sizeBytes,
    UINT64 strideBytes,
    uint64_t index) {
    if (startAddr == 0 || sizeBytes < sizeof(NacelleShaderIdentifier)) return nullptr;
    const UINT64 stride = strideBytes > 0 ? strideBytes : sizeof(NacelleShaderIdentifier);
    const UINT64 offset = index * stride;
    if (offset + sizeof(NacelleShaderIdentifier) > sizeBytes) return nullptr;
    const auto* id = reinterpret_cast<const NacelleShaderIdentifier*>(startAddr + offset);
    if (id->magic != NacelleShaderIdentifier::kMagic) return nullptr;
    return id;
}

const CompiledDxrHitGroup* lookupHitGroup(const CompiledDxrStateObject* stateObj, const NacelleShaderIdentifier* id) {
    if (!stateObj || !id || !id->isHitGroup) return nullptr;
    if (id->entryIndex >= stateObj->hitGroups.size()) return nullptr;
    return &stateObj->hitGroups[id->entryIndex];
}

const CompiledDxrShader* lookupShader(const CompiledDxrStateObject* stateObj, uint32_t idx) {
    if (!stateObj || idx >= stateObj->shaders.size()) return nullptr;
    return &stateObj->shaders[idx];
}

void invokeShaderSingleRay(
    const DxrThreadContext& ctx,
    const CompiledDxrShader* shader,
    void* payloadPtr,
    void* attrPtr) {
    if (!shader || !shader->fn) return;
    uint32_t rx[16]{};
    uint32_t ry[16]{};
    uint32_t rz[16]{};
    rx[0] = ctx.curRayIdx[0];
    ry[0] = ctx.curRayIdx[1];
    rz[0] = ctx.curRayIdx[2];
    shader->fn(
        ctx.rootCtx,
        1u,
        rx,
        ry,
        rz,
        ctx.desc.Width,
        ctx.desc.Height,
        ctx.desc.Depth,
        payloadPtr,
        attrPtr);
}

int32_t anyHitCallback(void* userData, const PipelineHitResult& candidate) {
    auto* ctx = static_cast<DxrThreadContext*>(userData);
    if (!ctx || !ctx->activeTrace || !ctx->stateObj) return 0;

    DxrActiveTrace& tr = *ctx->activeTrace;
    const uint64_t hgIdx = static_cast<uint64_t>(tr.rayContrib) +
                           static_cast<uint64_t>(tr.geomMult) * candidate.geometryIndex +
                           candidate.instanceContributionToHitGroupIndex;
    const auto* id = readSbtRecord(
        ctx->desc.HitGroupTable.StartAddress,
        ctx->desc.HitGroupTable.SizeInBytes,
        ctx->desc.HitGroupTable.StrideInBytes,
        hgIdx);
    const auto* hg = lookupHitGroup(ctx->stateObj, id);
    if (!hg || hg->anyHitIdx == UINT32_MAX) return 0;

    const auto* ahShader = lookupShader(ctx->stateObj, hg->anyHitIdx);
    if (!ahShader) return 0;

    PipelineHitResult savedHit = tr.hit;
    tr.hit = candidate;
    tr.anyHitAction = 0;

    float attr[2] = {candidate.baryU, candidate.baryV};
    invokeShaderSingleRay(*ctx, ahShader, tr.payloadPtr, attr);
    int32_t action = tr.anyHitAction;
    tr.hit = savedHit;
    return action;
}

void intersectionCallback(
    void* userData,
    const PipelineHitResult& candidate,
    float& inOutCommittedT,
    bool& outAccepted,
    bool& outEndSearch,
    uint32_t& outHitKind,
    float& outBaryU,
    float& outBaryV) {
    outAccepted = false;
    outEndSearch = false;
    auto* ctx = static_cast<DxrThreadContext*>(userData);
    if (!ctx || !ctx->activeTrace || !ctx->stateObj) return;

    DxrActiveTrace& tr = *ctx->activeTrace;
    const uint64_t hgIdx = static_cast<uint64_t>(tr.rayContrib) +
                           static_cast<uint64_t>(tr.geomMult) * candidate.geometryIndex +
                           candidate.instanceContributionToHitGroupIndex;
    const auto* id = readSbtRecord(
        ctx->desc.HitGroupTable.StartAddress,
        ctx->desc.HitGroupTable.SizeInBytes,
        ctx->desc.HitGroupTable.StrideInBytes,
        hgIdx);
    const auto* hg = lookupHitGroup(ctx->stateObj, id);
    if (!hg || hg->intersectionIdx == UINT32_MAX) return;

    const auto* isShader = lookupShader(ctx->stateObj, hg->intersectionIdx);
    if (!isShader) return;

    PipelineHitResult savedHit = tr.hit;
    tr.hit = candidate;
    tr.reportAccepted = false;
    tr.reportEndSearch = false;
    tr.reportT = inOutCommittedT;

    invokeShaderSingleRay(*ctx, isShader, nullptr, nullptr);

    if (tr.reportAccepted) {
        outAccepted = true;
        inOutCommittedT = tr.reportT;
        outHitKind = tr.reportHitKind;
        std::memcpy(&outBaryU, tr.reportAttr.data(), sizeof(float));
        std::memcpy(&outBaryV, tr.reportAttr.data() + sizeof(float), sizeof(float));
    }
    outEndSearch = tr.reportEndSearch;
    tr.hit = savedHit;
}

float readMatrixElem(const float m[12], int32_t row, int32_t col) {
    if (row < 0 || row >= 3 || col < 0 || col >= 4) return 0.0f;
    return m[row * 4 + col];
}

} // namespace

void executeDispatchRays(
    BS::thread_pool<>& pool,
    const D3D12_DISPATCH_RAYS_DESC& desc,
    const DriverRootContext& computeCtx,
    const CompiledDxrStateObject& stateObj) {
    if (desc.Width == 0 || desc.Height == 0 || desc.Depth == 0) return;

    const auto* rgId = readSbtRecord(
        desc.RayGenerationShaderRecord.StartAddress,
        desc.RayGenerationShaderRecord.SizeInBytes,
        desc.RayGenerationShaderRecord.SizeInBytes,
        0);
    if (!rgId || rgId->isHitGroup) return;
    const auto* rgShader = lookupShader(&stateObj, rgId->entryIndex);
    if (!rgShader || !rgShader->fn) return;

    const uint32_t W = desc.Width;
    const uint32_t H = desc.Height;
    const uint32_t D = desc.Depth;
    const uint64_t totalRays = static_cast<uint64_t>(W) * H * D;
    const uint32_t simdW = std::max<uint32_t>(1u, std::min<uint32_t>(16u, stateObj.simdWidth));
    const uint64_t numPackets = (totalRays + simdW - 1) / simdW;

    auto runPacketRange = [&](uint64_t pktBegin, uint64_t pktEnd) {
        DxrThreadContext ctx{};
        ctx.rootCtx = &computeCtx;
        ctx.stateObj = &stateObj;
        ctx.desc = desc;
        DxrThreadContext* prevCtx = t_dxrCtx;
        t_dxrCtx = &ctx;

        uint32_t rx[16]{};
        uint32_t ry[16]{};
        uint32_t rz[16]{};

        for (uint64_t pkt = pktBegin; pkt < pktEnd; ++pkt) {
            uint64_t baseRay = pkt * simdW;
            uint32_t count = static_cast<uint32_t>(std::min<uint64_t>(simdW, totalRays - baseRay));
            uint32_t activeMask = (count == 32) ? 0xFFFFFFFFu : ((1u << count) - 1u);
            for (uint32_t l = 0; l < count; ++l) {
                uint64_t linear = baseRay + l;
                uint32_t x = static_cast<uint32_t>(linear % W);
                uint64_t yz = linear / W;
                uint32_t y = static_cast<uint32_t>(yz % H);
                uint32_t z = static_cast<uint32_t>(yz / H);
                rx[l] = x;
                ry[l] = y;
                rz[l] = z;
            }
            ctx.curRayIdx[0] = rx[0];
            ctx.curRayIdx[1] = ry[0];
            ctx.curRayIdx[2] = rz[0];
            rgShader->fn(&computeCtx, activeMask, rx, ry, rz, W, H, D, nullptr, nullptr);
        }

        t_dxrCtx = prevCtx;
    };

    if (numPackets <= 1 || pool.get_thread_count() <= 1) {
        runPacketRange(0, numPackets);
    } else {
        pool.detach_blocks<uint64_t>(0, numPackets, runPacketRange);
        pool.wait();
    }
}

} // namespace nacelle

extern "C" {

void nacelle_dxr_trace_ray(
    const void* tlasPtr,
    int32_t rayFlags,
    int32_t instanceMask,
    int32_t rayContributionToHitGroupIndex,
    int32_t multiplierForGeometryContribution,
    int32_t missShaderIndex,
    float ox,
    float oy,
    float oz,
    float tMin,
    float dx,
    float dy,
    float dz,
    float tMax,
    void* payloadPtr) {
    auto* ctx = nacelle::t_dxrCtx;
    if (!ctx || !ctx->stateObj) return;

    nacelle::DxrActiveTrace trace{};
    trace.rayFlags = static_cast<uint32_t>(rayFlags);
    trace.tMin = tMin;
    trace.tMax = tMax;
    trace.worldOrigin[0] = ox;
    trace.worldOrigin[1] = oy;
    trace.worldOrigin[2] = oz;
    trace.worldDir[0] = dx;
    trace.worldDir[1] = dy;
    trace.worldDir[2] = dz;
    trace.payloadPtr = payloadPtr;
    trace.rayContrib = static_cast<uint32_t>(rayContributionToHitGroupIndex);
    trace.geomMult = static_cast<uint32_t>(multiplierForGeometryContribution);

    nacelle::DxrActiveTrace* savedTrace = ctx->activeTrace;
    ctx->activeTrace = &trace;

    trace.hit = nacelle::traceRayForPipeline(
        tlasPtr,
        static_cast<uint32_t>(rayFlags),
        static_cast<uint32_t>(instanceMask),
        ox, oy, oz, tMin,
        dx, dy, dz, tMax,
        &nacelle::anyHitCallback,
        &nacelle::intersectionCallback,
        ctx);

    if (trace.hit.hit) {
        if (!(trace.rayFlags & nacelle::kRayFlagSkipClosestHitShader)) {
            const uint64_t hgIdx = static_cast<uint64_t>(trace.rayContrib) +
                                   static_cast<uint64_t>(trace.geomMult) * trace.hit.geometryIndex +
                                   trace.hit.instanceContributionToHitGroupIndex;
            const auto* id = nacelle::readSbtRecord(
                ctx->desc.HitGroupTable.StartAddress,
                ctx->desc.HitGroupTable.SizeInBytes,
                ctx->desc.HitGroupTable.StrideInBytes,
                hgIdx);
            const auto* hg = nacelle::lookupHitGroup(ctx->stateObj, id);
            if (hg && hg->closestHitIdx != UINT32_MAX) {
                const auto* chShader = nacelle::lookupShader(ctx->stateObj, hg->closestHitIdx);
                float attr[2] = {trace.hit.baryU, trace.hit.baryV};
                nacelle::invokeShaderSingleRay(*ctx, chShader, payloadPtr, attr);
            }
        }
    } else {
        trace.hit.t = tMax;
        const auto* id = nacelle::readSbtRecord(
            ctx->desc.MissShaderTable.StartAddress,
            ctx->desc.MissShaderTable.SizeInBytes,
            ctx->desc.MissShaderTable.StrideInBytes,
            static_cast<uint64_t>(missShaderIndex));
        if (id && !id->isHitGroup) {
            const auto* missShader = nacelle::lookupShader(ctx->stateObj, id->entryIndex);
            nacelle::invokeShaderSingleRay(*ctx, missShader, payloadPtr, nullptr);
        }
    }

    ctx->activeTrace = savedTrace;
}

int32_t nacelle_dxr_state_i32(int32_t op) {
    auto* ctx = nacelle::t_dxrCtx;
    if (!ctx || !ctx->activeTrace) return 0;
    const auto& tr = *ctx->activeTrace;
    switch (op) {
    case 141: return static_cast<int32_t>(tr.hit.instanceID);
    case 142: return static_cast<int32_t>(tr.hit.instanceIndex);
    case 143: return static_cast<int32_t>(tr.hit.hitKind);
    case 144: return static_cast<int32_t>(tr.rayFlags);
    case 161: return static_cast<int32_t>(tr.hit.primitiveIndex);
    case 213: return static_cast<int32_t>(tr.hit.geometryIndex);
    default: return 0;
    }
}

float nacelle_dxr_state_f32(int32_t op) {
    auto* ctx = nacelle::t_dxrCtx;
    if (!ctx || !ctx->activeTrace) return 0.0f;
    const auto& tr = *ctx->activeTrace;
    switch (op) {
    case 153: return tr.tMin;
    case 154: return tr.hit.hit ? tr.hit.t : tr.tMax;
    default: return 0.0f;
    }
}

float nacelle_dxr_state_vec(int32_t op, int32_t comp) {
    auto* ctx = nacelle::t_dxrCtx;
    if (!ctx || !ctx->activeTrace || comp < 0 || comp >= 3) return 0.0f;
    const auto& tr = *ctx->activeTrace;
    switch (op) {
    case 147: return tr.worldOrigin[comp];
    case 148: return tr.worldDir[comp];
    case 149: return tr.hit.objOrigin[comp];
    case 150: return tr.hit.objDir[comp];
    default: return 0.0f;
    }
}

float nacelle_dxr_state_mat(int32_t op, int32_t row, int32_t col) {
    auto* ctx = nacelle::t_dxrCtx;
    if (!ctx || !ctx->activeTrace) return 0.0f;
    const auto& tr = *ctx->activeTrace;
    switch (op) {
    case 151: return nacelle::readMatrixElem(tr.hit.objToWorld, row, col);
    case 152: return nacelle::readMatrixElem(tr.hit.worldToObj, row, col);
    default: return 0.0f;
    }
}

void nacelle_dxr_ignore_hit() {
    if (nacelle::t_dxrCtx && nacelle::t_dxrCtx->activeTrace) {
        nacelle::t_dxrCtx->activeTrace->anyHitAction = 1;
    }
}

void nacelle_dxr_accept_hit_and_end_search() {
    if (nacelle::t_dxrCtx && nacelle::t_dxrCtx->activeTrace) {
        nacelle::t_dxrCtx->activeTrace->anyHitAction = 2;
    }
}

int32_t nacelle_dxr_report_hit(float tHit, int32_t hitKind, const void* attrPtr, int32_t attrBytes) {
    auto* ctx = nacelle::t_dxrCtx;
    if (!ctx || !ctx->activeTrace) return 0;
    auto& tr = *ctx->activeTrace;
    if (tHit < tr.tMin || tHit > tr.reportT) return 0;
    tr.reportAccepted = true;
    tr.reportT = tHit;
    tr.reportHitKind = static_cast<uint32_t>(hitKind);
    if (attrPtr && attrBytes > 0) {
        const size_t n = std::min<size_t>(static_cast<size_t>(attrBytes), tr.reportAttr.size());
        std::memcpy(tr.reportAttr.data(), attrPtr, n);
    }
    return 1;
}

void nacelle_dxr_call_shader(int32_t shaderIndex, void* paramPtr) {
    auto* ctx = nacelle::t_dxrCtx;
    if (!ctx || !ctx->stateObj) return;
    const auto* id = nacelle::readSbtRecord(
        ctx->desc.CallableShaderTable.StartAddress,
        ctx->desc.CallableShaderTable.SizeInBytes,
        ctx->desc.CallableShaderTable.StrideInBytes,
        static_cast<uint64_t>(shaderIndex));
    if (id && !id->isHitGroup) {
        const auto* shader = nacelle::lookupShader(ctx->stateObj, id->entryIndex);
        nacelle::invokeShaderSingleRay(*ctx, shader, paramPtr, nullptr);
    }
}

} // extern "C"
