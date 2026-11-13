#pragma once

#include <array>
#include <cstdint>
#include "../common/platform.h"

namespace nacelle {

class NacelleResource;

/// Per-mip layout info shared between NacelleResource and texture_sampler.bc.
struct NacelleMipLayout {
    uint64_t offset = 0;      // Byte offset from resource base
    uint32_t row_pitch = 0;   // Bytes per row (or per block-row if compressed)
    uint32_t slice_pitch = 0; // Bytes per 2D slice
    uint32_t width = 1;
    uint32_t height = 1;
    uint32_t depth = 1;
};

/// Extra metadata attached to texture descriptors via NacelleDescriptor::extra_ptr.
struct NacelleTextureMetadata {
    static constexpr uint32_t kMaxMips = 16;
    uint32_t array_size = 1;
    uint32_t mip_levels = 1;
    uint32_t most_detailed_mip = 0;
    uint32_t dimension = 0; // D3D12_SRV_DIMENSION or D3D12_RESOURCE_DIMENSION
    uint32_t is_cubemap = 0;
    uint32_t is_compressed = 0;
    uint32_t resource_planes = 1;
    uint32_t reserved = 0;
    uint64_t array_slice_stride = 0;
    // If non-null, points to an RGBA32F decoded mip array (used for BC1-BC7 textures on demand).
    const float* decoded_rgba32f = nullptr;
    NacelleMipLayout mips[kMaxMips]{};
    NacelleMipLayout decoded_mips[kMaxMips]{};
};

/// Sampler flags packed into NacelleDescriptor::sampler_flags.
namespace SamplerFlags {
constexpr uint32_t kFilterShift = 0;     // 8 bits (D3D12_FILTER)
constexpr uint32_t kAddrUShift = 8;      // 3 bits (D3D12_TEXTURE_ADDRESS_MODE - 1)
constexpr uint32_t kAddrVShift = 11;     // 3 bits
constexpr uint32_t kAddrWShift = 14;     // 3 bits
constexpr uint32_t kCmpFuncShift = 17;   // 4 bits (D3D12_COMPARISON_FUNC)
constexpr uint32_t kMaxAnisoShift = 21;  // 5 bits (1..16)
constexpr uint32_t kBorderShift = 26;    // 2 bits (0=trans black, 1=opaque black, 2=opaque white, 3=custom)
constexpr uint32_t kReductionShift = 28; // 2 bits (0=standard, 1=comparison, 2=minimum, 3=maximum)
} // namespace SamplerFlags

/// View / resource descriptor (64 bytes, 64-byte aligned) matching SM 6.6 heap layout.
struct alignas(64) NacelleDescriptor {
    void* base_ptr = nullptr;         // 0..7   Host RAM pointer
    uint32_t width = 0;               // 8..11  Elements or texels
    uint32_t height = 1;              // 12..15
    uint32_t depth_or_array = 1;      // 16..19
    uint32_t row_pitch = 0;           // 20..23
    uint32_t format = 0;              // 24..27 DXGI_FORMAT
    uint32_t mip_levels = 1;          // 28..31
    uint32_t sampler_flags = 0;       // 32..35 Sampler filter/address/cmp flags
    uint32_t elem_stride = 0;         // 36..39 Bytes per element / structured stride
    uint64_t byte_size = 0;           // 40..47 Total byte size
    void* extra_ptr = nullptr;        // 48..55 NacelleTextureMetadata* (textures) or counter ptr / border ptr
    float mip_lod_bias = 0.0f;        // 56..59
    uint32_t view_flags = 0;          // 60..63 Dimension / resource kind / min-max LOD packed
};
static_assert(sizeof(NacelleDescriptor) == 64, "NacelleDescriptor must be 64 bytes");

/// Non-shader-visible RTV/DSV CPU descriptor payload stored in RTV/DSV descriptor heaps.
struct NacelleRenderTargetDescriptor {
    NacelleResource* resource = nullptr;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    uint32_t mip_slice = 0;
    uint32_t first_array_slice = 0;
    uint32_t array_size = 1;
    uint32_t flags = 0;
};

/// Context struct passed as the first argument to every JITed compute/vertex/pixel kernel.
struct alignas(64) DriverRootContext {
    static constexpr uint32_t kMaxRootSlots = 64;
    static constexpr uint32_t kMaxStaticSamplers = 16;

    const void* root_slots[kMaxRootSlots]{};
    NacelleDescriptor inline_descriptors[kMaxRootSlots]{};
    uint32_t inline_constants[kMaxRootSlots * 4]{};
    NacelleDescriptor static_samplers[kMaxStaticSamplers]{};

    const NacelleDescriptor* resource_heap_base = nullptr;
    uint32_t resource_heap_count = 0;
    const NacelleDescriptor* sampler_heap_base = nullptr;
    uint32_t sampler_heap_count = 0;

    // Graphics-specific state read by synthesized VS/PS wrappers:
    uint64_t vertex_buffers[16]{};
    uint32_t vertex_strides[16]{};
    uint32_t vertex_sizes[16]{};
    float blend_factor[4]{1.0f, 1.0f, 1.0f, 1.0f};
    uint32_t stencil_ref = 0;
    uint32_t sample_mask = 0xFFFFFFFFu;
};

} // namespace nacelle
