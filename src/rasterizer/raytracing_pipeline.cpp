#include "rasterizer/raytracing_pipeline.h"
#include "rasterizer/raytracing_bvh.h"

#include <algorithm>
#include <array>
#include <bit>
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
    std::array<uint8_t, 64> reportAttr;
};

struct DxrPacketView {
    const PipelineHitResult* hits = nullptr;
    const uint32_t* rayFlags = nullptr;
    const float* ox = nullptr;
    const float* oy = nullptr;
    const float* oz = nullptr;
    const float* tMin = nullptr;
    const float* dx = nullptr;
    const float* dy = nullptr;
    const float* dz = nullptr;
    const float* tMax = nullptr;
};

struct DxrThreadContext {
    const DriverRootContext* rootCtx = nullptr;
    const CompiledDxrStateObject* stateObj = nullptr;
    D3D12_DISPATCH_RAYS_DESC desc{};
    bool hasAnyHitOrIntersection = false;
    uint32_t curRayIdx[3]{0, 0, 0};
    const uint32_t* curPacketRx = nullptr;
    const uint32_t* curPacketRy = nullptr;
    const uint32_t* curPacketRz = nullptr;
    DxrActiveTrace* activeTrace = nullptr;
    const DxrPacketView* activePacket = nullptr;
    uint32_t activePacketMask = 0;
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
    DxrThreadContext& ctx,
    const CompiledDxrShader* shader,
    void* payloadPtr,
    void* attrPtr) {
    if (!shader || !shader->fn) return;
    uint32_t rx[16];
    uint32_t ry[16];
    uint32_t rz[16];
    rx[0] = ctx.curRayIdx[0];
    ry[0] = ctx.curRayIdx[1];
    rz[0] = ctx.curRayIdx[2];
    void* pPtrs[16];
    void* aPtrs[16];
    pPtrs[0] = payloadPtr;
    aPtrs[0] = attrPtr;
    const DxrPacketView* savedPkt = ctx.activePacket;
    ctx.activePacket = nullptr;
    shader->fn(
        ctx.rootCtx,
        1u,
        rx,
        ry,
        rz,
        ctx.desc.Width,
        ctx.desc.Height,
        ctx.desc.Depth,
        pPtrs,
        aPtrs);
    ctx.activePacket = savedPkt;
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

    bool hasAnyHitOrIntersection = false;
    for (const auto& hg : stateObj.hitGroups) {
        if (hg.anyHitIdx != UINT32_MAX || hg.intersectionIdx != UINT32_MAX) {
            hasAnyHitOrIntersection = true;
            break;
        }
    }

    auto runPacketRange = [&](uint64_t pktBegin, uint64_t pktEnd) {
        DxrThreadContext ctx{};
        ctx.rootCtx = &computeCtx;
        ctx.stateObj = &stateObj;
        ctx.desc = desc;
        ctx.hasAnyHitOrIntersection = hasAnyHitOrIntersection;
        DxrThreadContext* prevCtx = t_dxrCtx;
        t_dxrCtx = &ctx;

        uint32_t rx[16]{};
        uint32_t ry[16]{};
        uint32_t rz[16]{};
        ctx.curPacketRx = rx;
        ctx.curPacketRy = ry;
        ctx.curPacketRz = rz;

        for (uint64_t pkt = pktBegin; pkt < pktEnd; ++pkt) {
            uint64_t baseRay = pkt * simdW;
            uint32_t count = static_cast<uint32_t>(std::min<uint64_t>(simdW, totalRays - baseRay));
            uint32_t activeMask = (count == 32) ? 0xFFFFFFFFu : ((1u << count) - 1u);
            uint32_t x0 = static_cast<uint32_t>(baseRay % W);
            uint64_t yz0 = baseRay / W;
            uint32_t y0 = static_cast<uint32_t>(yz0 % H);
            uint32_t z0 = static_cast<uint32_t>(yz0 / H);
            if (x0 + count <= W) {
                for (uint32_t l = 0; l < count; ++l) {
                    rx[l] = x0 + l;
                    ry[l] = y0;
                    rz[l] = z0;
                }
            } else {
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

    nacelle::DxrActiveTrace trace;
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
        ctx->hasAnyHitOrIntersection ? &nacelle::anyHitCallback : nullptr,
        ctx->hasAnyHitOrIntersection ? &nacelle::intersectionCallback : nullptr,
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

void nacelle_dxr_trace_ray_packet(
    uint32_t activeMask,
    const void* tlasPtr,
    const int32_t* rayFlags,
    const int32_t* instanceMask,
    const int32_t* rayContributionToHitGroupIndex,
    const int32_t* multiplierForGeometryContribution,
    const int32_t* missShaderIndex,
    const float* ox,
    const float* oy,
    const float* oz,
    const float* tMin,
    const float* dx,
    const float* dy,
    const float* dz,
    const float* tMax,
    void** payloadPtrs) {
    auto* ctx = nacelle::t_dxrCtx;
    if (!ctx || !ctx->stateObj || activeMask == 0) return;

    if (ctx->hasAnyHitOrIntersection) {
        uint32_t m = activeMask;
        while (m) {
            const uint32_t l = static_cast<uint32_t>(std::countr_zero(m));
            m &= m - 1;
            nacelle_dxr_trace_ray(
                tlasPtr, rayFlags[l], instanceMask[l],
                rayContributionToHitGroupIndex[l], multiplierForGeometryContribution[l],
                missShaderIndex[l], ox[l], oy[l], oz[l], tMin[l],
                dx[l], dy[l], dz[l], tMax[l], payloadPtrs[l]);
        }
        return;
    }

    nacelle::PipelineHitResult hits[16];
    nacelle::traceRayPacketForPipeline(
        activeMask,
        tlasPtr,
        reinterpret_cast<const uint32_t*>(rayFlags),
        reinterpret_cast<const uint32_t*>(instanceMask),
        ox, oy, oz, tMin, dx, dy, dz, tMax,
        hits);

    uint32_t targetShaderIdx[16];
    float attrStorage[16][2];
    void* attrPtrs[16];
    uint32_t pendingMask = 0;

    uint32_t m = activeMask;
    while (m) {
        const uint32_t l = static_cast<uint32_t>(std::countr_zero(m));
        m &= m - 1;
        const auto& h = hits[l];
        uint32_t sIdx = UINT32_MAX;
        if (h.hit) {
            if (!(static_cast<uint32_t>(rayFlags[l]) & nacelle::kRayFlagSkipClosestHitShader)) {
                const uint64_t hgIdx = static_cast<uint64_t>(static_cast<uint32_t>(rayContributionToHitGroupIndex[l])) +
                                       static_cast<uint64_t>(static_cast<uint32_t>(multiplierForGeometryContribution[l])) * h.geometryIndex +
                                       h.instanceContributionToHitGroupIndex;
                const auto* id = nacelle::readSbtRecord(
                    ctx->desc.HitGroupTable.StartAddress,
                    ctx->desc.HitGroupTable.SizeInBytes,
                    ctx->desc.HitGroupTable.StrideInBytes,
                    hgIdx);
                const auto* hg = nacelle::lookupHitGroup(ctx->stateObj, id);
                if (hg && hg->closestHitIdx != UINT32_MAX) {
                    sIdx = hg->closestHitIdx;
                    attrStorage[l][0] = h.baryU;
                    attrStorage[l][1] = h.baryV;
                    attrPtrs[l] = attrStorage[l];
                }
            }
        } else {
            const auto* id = nacelle::readSbtRecord(
                ctx->desc.MissShaderTable.StartAddress,
                ctx->desc.MissShaderTable.SizeInBytes,
                ctx->desc.MissShaderTable.StrideInBytes,
                static_cast<uint64_t>(missShaderIndex[l]));
            if (id && !id->isHitGroup) {
                sIdx = id->entryIndex;
                attrPtrs[l] = nullptr;
            }
        }
        if (sIdx != UINT32_MAX) {
            targetShaderIdx[l] = sIdx;
            pendingMask |= (1u << l);
        }
    }

    if (pendingMask != 0) {
        nacelle::DxrPacketView pktView{};
        pktView.hits = hits;
        pktView.rayFlags = reinterpret_cast<const uint32_t*>(rayFlags);
        pktView.ox = ox; pktView.oy = oy; pktView.oz = oz; pktView.tMin = tMin;
        pktView.dx = dx; pktView.dy = dy; pktView.dz = dz; pktView.tMax = tMax;

        const nacelle::DxrPacketView* savedPkt = ctx->activePacket;
        const uint32_t savedPktMask = ctx->activePacketMask;
        ctx->activePacket = &pktView;
        ctx->activePacketMask = activeMask;

        while (pendingMask != 0) {
            const uint32_t first = static_cast<uint32_t>(std::countr_zero(pendingMask));
            const uint32_t sIdx = targetShaderIdx[first];
            uint32_t batchMask = 0;
            uint32_t scan = pendingMask;
            while (scan) {
                const uint32_t l = static_cast<uint32_t>(std::countr_zero(scan));
                scan &= scan - 1;
                if (targetShaderIdx[l] == sIdx) {
                    batchMask |= (1u << l);
                }
            }
            pendingMask &= ~batchMask;

            const auto* shader = nacelle::lookupShader(ctx->stateObj, sIdx);
            if (shader && shader->fn) {
                shader->fn(
                    ctx->rootCtx,
                    batchMask,
                    ctx->curPacketRx,
                    ctx->curPacketRy,
                    ctx->curPacketRz,
                    ctx->desc.Width,
                    ctx->desc.Height,
                    ctx->desc.Depth,
                    payloadPtrs,
                    attrPtrs);
            }
        }

        ctx->activePacket = savedPkt;
        ctx->activePacketMask = savedPktMask;
    }
}

int32_t nacelle_dxr_state_i32(int32_t op, int32_t lane) {
    const auto* ctx = nacelle::t_dxrCtx;
    if (!ctx) return 0;
    if (ctx->activePacket && lane >= 0 && lane < 16) {
        const auto& h = ctx->activePacket->hits[lane];
        switch (op) {
        case 141: return static_cast<int32_t>(h.instanceID);
        case 142: return static_cast<int32_t>(h.instanceIndex);
        case 143: return static_cast<int32_t>(h.hitKind);
        case 144: return static_cast<int32_t>(ctx->activePacket->rayFlags[lane]);
        case 161: return static_cast<int32_t>(h.primitiveIndex);
        case 213: return static_cast<int32_t>(h.geometryIndex);
        default: return 0;
        }
    }
    if (!ctx->activeTrace) return 0;
    const auto* tr = ctx->activeTrace;
    switch (op) {
    case 141: return static_cast<int32_t>(tr->hit.instanceID);
    case 142: return static_cast<int32_t>(tr->hit.instanceIndex);
    case 143: return static_cast<int32_t>(tr->hit.hitKind);
    case 144: return static_cast<int32_t>(tr->rayFlags);
    case 161: return static_cast<int32_t>(tr->hit.primitiveIndex);
    case 213: return static_cast<int32_t>(tr->hit.geometryIndex);
    default: return 0;
    }
}

float nacelle_dxr_state_f32(int32_t op, int32_t lane) {
    const auto* ctx = nacelle::t_dxrCtx;
    if (!ctx) return 0.0f;
    if (ctx->activePacket && lane >= 0 && lane < 16) {
        const auto& p = *ctx->activePacket;
        switch (op) {
        case 153: return p.tMin[lane];
        case 154: return p.hits[lane].hit ? p.hits[lane].t : p.tMax[lane];
        default: return 0.0f;
        }
    }
    if (!ctx->activeTrace) return 0.0f;
    const auto* tr = ctx->activeTrace;
    switch (op) {
    case 153: return tr->tMin;
    case 154: return tr->hit.hit ? tr->hit.t : tr->tMax;
    default: return 0.0f;
    }
}

float nacelle_dxr_state_vec(int32_t op, int32_t comp, int32_t lane) {
    const auto* ctx = nacelle::t_dxrCtx;
    if (!ctx || comp < 0 || comp >= 3) return 0.0f;
    static constexpr float kIdentity[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    if (ctx->activePacket && lane >= 0 && lane < 16) {
        const auto& p = *ctx->activePacket;
        const float ox = p.ox[lane], oy = p.oy[lane], oz = p.oz[lane];
        const float dx = p.dx[lane], dy = p.dy[lane], dz = p.dz[lane];
        switch (op) {
        case 147: return comp == 0 ? ox : (comp == 1 ? oy : oz);
        case 148: return comp == 0 ? dx : (comp == 1 ? dy : dz);
        case 149: {
            const float* m = p.hits[lane].worldToObjPtr ? p.hits[lane].worldToObjPtr : kIdentity;
            const int r = comp * 4;
            return m[r] * ox + m[r + 1] * oy + m[r + 2] * oz + m[r + 3];
        }
        case 150: {
            const float* m = p.hits[lane].worldToObjPtr ? p.hits[lane].worldToObjPtr : kIdentity;
            const int r = comp * 4;
            return m[r] * dx + m[r + 1] * dy + m[r + 2] * dz;
        }
        default: return 0.0f;
        }
    }
    if (!ctx->activeTrace) return 0.0f;
    const auto* tr = ctx->activeTrace;
    switch (op) {
    case 147: return tr->worldOrigin[comp];
    case 148: return tr->worldDir[comp];
    case 149: {
        const float* m = tr->hit.worldToObjPtr ? tr->hit.worldToObjPtr : kIdentity;
        const int r = comp * 4;
        return m[r] * tr->worldOrigin[0] + m[r + 1] * tr->worldOrigin[1] + m[r + 2] * tr->worldOrigin[2] + m[r + 3];
    }
    case 150: {
        const float* m = tr->hit.worldToObjPtr ? tr->hit.worldToObjPtr : kIdentity;
        const int r = comp * 4;
        return m[r] * tr->worldDir[0] + m[r + 1] * tr->worldDir[1] + m[r + 2] * tr->worldDir[2];
    }
    default: return 0.0f;
    }
}

float nacelle_dxr_state_mat(int32_t op, int32_t row, int32_t col, int32_t lane) {
    const auto* ctx = nacelle::t_dxrCtx;
    if (!ctx) return 0.0f;
    static constexpr float kIdentity[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    if (ctx->activePacket && lane >= 0 && lane < 16) {
        const auto& h = ctx->activePacket->hits[lane];
        switch (op) {
        case 151: return nacelle::readMatrixElem(h.objToWorldPtr ? h.objToWorldPtr : kIdentity, row, col);
        case 152: return nacelle::readMatrixElem(h.worldToObjPtr ? h.worldToObjPtr : kIdentity, row, col);
        default: return 0.0f;
        }
    }
    if (!ctx->activeTrace) return 0.0f;
    const auto* tr = ctx->activeTrace;
    switch (op) {
    case 151: return nacelle::readMatrixElem(tr->hit.objToWorldPtr ? tr->hit.objToWorldPtr : kIdentity, row, col);
    case 152: return nacelle::readMatrixElem(tr->hit.worldToObjPtr ? tr->hit.worldToObjPtr : kIdentity, row, col);
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
