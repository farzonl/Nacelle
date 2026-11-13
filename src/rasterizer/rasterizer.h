#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include <BS_thread_pool.hpp>

#include "../api/d3d12_types.h"
#include "../api/pipeline.h"

namespace nacelle {

class NacelleResource;

/// Snapshot of graphics pipeline & OM/IA/RS state consumed by the software rasterizer.
struct GraphicsExecutionState {
    const NacellePipelineState* pso = nullptr;
    DriverRootContext rootCtx{};
    D3D12_PRIMITIVE_TOPOLOGY topology = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    D3D12_INDEX_BUFFER_VIEW indexBuffer{};
    D3D12_VERTEX_BUFFER_VIEW vertexBuffers[16]{};
    D3D12_VIEWPORT viewports[16]{};
    uint32_t numViewports = 0;
    D3D12_RECT scissors[16]{};
    uint32_t numScissors = 0;
    NacelleRenderTargetDescriptor rtvs[8]{};
    uint32_t numRTVs = 0;
    NacelleRenderTargetDescriptor dsv{};
    bool hasDSV = false;
    D3D12_SHADING_RATE shadingRate = D3D12_SHADING_RATE_1X1;
    D3D12_SHADING_RATE_COMBINER shadingRateCombiners[2]{
        D3D12_SHADING_RATE_COMBINER_PASSTHROUGH,
        D3D12_SHADING_RATE_COMBINER_PASSTHROUGH,
    };
};

struct DrawCallParams {
    bool indexed = false;
    uint32_t countPerInstance = 0;
    uint32_t instanceCount = 1;
    uint32_t startIndexOrVertex = 0;
    int32_t baseVertexLocation = 0;
    uint32_t startInstanceLocation = 0;
};

/// Executes a graphics draw call via SIMD vertex shading + sort-middle 64x64 tile rasterization.
void executeRasterDrawCall(
    BS::thread_pool<>& pool,
    const GraphicsExecutionState& state,
    const DrawCallParams& draw);

} // namespace nacelle
