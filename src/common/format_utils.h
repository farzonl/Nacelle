#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#include <directx/dxgiformat.h>
#include <DirectXMath.h>
#include <DirectXPackedVector.h>

namespace nacelle {

struct FormatInfo {
    uint32_t bytesPerBlock = 4; // Bytes per pixel (uncompressed) or per 4x4 block (BCn)
    uint32_t blockWidth = 1;
    uint32_t blockHeight = 1;
    bool isCompressed = false;
    bool isSRGB = false;
    bool isDepth = false;
    bool isInteger = false;
};

inline FormatInfo getFormatInfo(DXGI_FORMAT fmt) {
    switch (fmt) {
    case DXGI_FORMAT_UNKNOWN:
        return {1, 1, 1, false, false, false, false};
    case DXGI_FORMAT_R32G32B32A32_TYPELESS:
    case DXGI_FORMAT_R32G32B32A32_FLOAT:
    case DXGI_FORMAT_R32G32B32A32_UINT:
    case DXGI_FORMAT_R32G32B32A32_SINT:
        return {16, 1, 1, false, false, false, fmt != DXGI_FORMAT_R32G32B32A32_FLOAT};
    case DXGI_FORMAT_R32G32B32_TYPELESS:
    case DXGI_FORMAT_R32G32B32_FLOAT:
    case DXGI_FORMAT_R32G32B32_UINT:
    case DXGI_FORMAT_R32G32B32_SINT:
        return {12, 1, 1, false, false, false, fmt != DXGI_FORMAT_R32G32B32_FLOAT};
    case DXGI_FORMAT_R16G16B16A16_TYPELESS:
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
    case DXGI_FORMAT_R16G16B16A16_UNORM:
    case DXGI_FORMAT_R16G16B16A16_UINT:
    case DXGI_FORMAT_R16G16B16A16_SNORM:
    case DXGI_FORMAT_R16G16B16A16_SINT:
    case DXGI_FORMAT_R32G32_TYPELESS:
    case DXGI_FORMAT_R32G32_FLOAT:
    case DXGI_FORMAT_R32G32_UINT:
    case DXGI_FORMAT_R32G32_SINT:
        return {8, 1, 1, false, false, false,
                (fmt == DXGI_FORMAT_R16G16B16A16_UINT || fmt == DXGI_FORMAT_R16G16B16A16_SINT ||
                 fmt == DXGI_FORMAT_R32G32_UINT || fmt == DXGI_FORMAT_R32G32_SINT)};
    case DXGI_FORMAT_R10G10B10A2_TYPELESS:
    case DXGI_FORMAT_R10G10B10A2_UNORM:
    case DXGI_FORMAT_R10G10B10A2_UINT:
    case DXGI_FORMAT_R11G11B10_FLOAT:
    case DXGI_FORMAT_R8G8B8A8_TYPELESS:
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
    case DXGI_FORMAT_R8G8B8A8_UINT:
    case DXGI_FORMAT_R8G8B8A8_SNORM:
    case DXGI_FORMAT_R8G8B8A8_SINT:
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_TYPELESS:
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
    case DXGI_FORMAT_B8G8R8X8_UNORM:
    case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB:
    case DXGI_FORMAT_R16G16_TYPELESS:
    case DXGI_FORMAT_R16G16_FLOAT:
    case DXGI_FORMAT_R16G16_UNORM:
    case DXGI_FORMAT_R16G16_UINT:
    case DXGI_FORMAT_R16G16_SNORM:
    case DXGI_FORMAT_R16G16_SINT:
    case DXGI_FORMAT_R32_TYPELESS:
    case DXGI_FORMAT_R32_FLOAT:
    case DXGI_FORMAT_R32_UINT:
    case DXGI_FORMAT_R32_SINT:
        return {4, 1, 1, false,
                (fmt == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB || fmt == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB ||
                 fmt == DXGI_FORMAT_B8G8R8X8_UNORM_SRGB),
                false,
                (fmt == DXGI_FORMAT_R8G8B8A8_UINT || fmt == DXGI_FORMAT_R8G8B8A8_SINT ||
                 fmt == DXGI_FORMAT_R32_UINT || fmt == DXGI_FORMAT_R32_SINT ||
                 fmt == DXGI_FORMAT_R10G10B10A2_UINT)};
    case DXGI_FORMAT_D32_FLOAT:
    case DXGI_FORMAT_D24_UNORM_S8_UINT:
    case DXGI_FORMAT_R24G8_TYPELESS:
        return {4, 1, 1, false, false, true, false};
    case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
        return {8, 1, 1, false, false, true, false};
    case DXGI_FORMAT_D16_UNORM:
        return {2, 1, 1, false, false, true, false};
    case DXGI_FORMAT_R8G8_TYPELESS:
    case DXGI_FORMAT_R8G8_UNORM:
    case DXGI_FORMAT_R8G8_UINT:
    case DXGI_FORMAT_R8G8_SNORM:
    case DXGI_FORMAT_R8G8_SINT:
    case DXGI_FORMAT_R16_TYPELESS:
    case DXGI_FORMAT_R16_FLOAT:
    case DXGI_FORMAT_R16_UNORM:
    case DXGI_FORMAT_R16_UINT:
    case DXGI_FORMAT_R16_SNORM:
    case DXGI_FORMAT_R16_SINT:
        return {2, 1, 1, false, false, false, false};
    case DXGI_FORMAT_R8_TYPELESS:
    case DXGI_FORMAT_R8_UNORM:
    case DXGI_FORMAT_R8_UINT:
    case DXGI_FORMAT_R8_SNORM:
    case DXGI_FORMAT_R8_SINT:
    case DXGI_FORMAT_A8_UNORM:
        return {1, 1, 1, false, false, false, false};
    case DXGI_FORMAT_BC1_TYPELESS:
    case DXGI_FORMAT_BC1_UNORM:
    case DXGI_FORMAT_BC1_UNORM_SRGB:
        return {8, 4, 4, true, fmt == DXGI_FORMAT_BC1_UNORM_SRGB, false, false};
    case DXGI_FORMAT_BC4_TYPELESS:
    case DXGI_FORMAT_BC4_UNORM:
    case DXGI_FORMAT_BC4_SNORM:
        return {8, 4, 4, true, false, false, false};
    case DXGI_FORMAT_BC2_TYPELESS:
    case DXGI_FORMAT_BC2_UNORM:
    case DXGI_FORMAT_BC2_UNORM_SRGB:
    case DXGI_FORMAT_BC3_TYPELESS:
    case DXGI_FORMAT_BC3_UNORM:
    case DXGI_FORMAT_BC3_UNORM_SRGB:
    case DXGI_FORMAT_BC5_TYPELESS:
    case DXGI_FORMAT_BC5_UNORM:
    case DXGI_FORMAT_BC5_SNORM:
    case DXGI_FORMAT_BC6H_TYPELESS:
    case DXGI_FORMAT_BC6H_UF16:
    case DXGI_FORMAT_BC6H_SF16:
    case DXGI_FORMAT_BC7_TYPELESS:
    case DXGI_FORMAT_BC7_UNORM:
    case DXGI_FORMAT_BC7_UNORM_SRGB:
        return {16, 4, 4, true,
                (fmt == DXGI_FORMAT_BC2_UNORM_SRGB || fmt == DXGI_FORMAT_BC3_UNORM_SRGB ||
                 fmt == DXGI_FORMAT_BC7_UNORM_SRGB),
                false, false};
    default:
        return {4, 1, 1, false, false, false, false};
    }
}

inline float srgbToLinear(float c) {
    return (c <= 0.04045f) ? (c / 12.92f) : std::pow((c + 0.055f) / 1.055f, 2.4f);
}

inline float linearToSrgb(float c) {
    c = std::clamp(c, 0.0f, 1.0f);
    return (c <= 0.0031308f) ? (c * 12.92f) : (1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f);
}

inline void unpackPixelRGBA32F(DXGI_FORMAT fmt, const uint8_t* src, float out[4]) {
    using namespace DirectX;
    using namespace DirectX::PackedVector;
    switch (fmt) {
    case DXGI_FORMAT_R32G32B32A32_FLOAT:
        std::memcpy(out, src, 16);
        break;
    case DXGI_FORMAT_R32G32B32_FLOAT:
        std::memcpy(out, src, 12);
        out[3] = 1.0f;
        break;
    case DXGI_FORMAT_R32G32_FLOAT:
        std::memcpy(out, src, 8);
        out[2] = 0.0f;
        out[3] = 1.0f;
        break;
    case DXGI_FORMAT_R32_FLOAT:
    case DXGI_FORMAT_D32_FLOAT:
        std::memcpy(out, src, 4);
        out[1] = out[2] = 0.0f;
        out[3] = 1.0f;
        break;
    case DXGI_FORMAT_R16G16B16A16_FLOAT: {
        const auto* h = reinterpret_cast<const HALF*>(src);
        out[0] = XMConvertHalfToFloat(h[0]);
        out[1] = XMConvertHalfToFloat(h[1]);
        out[2] = XMConvertHalfToFloat(h[2]);
        out[3] = XMConvertHalfToFloat(h[3]);
        break;
    }
    case DXGI_FORMAT_R16G16_FLOAT: {
        const auto* h = reinterpret_cast<const HALF*>(src);
        out[0] = XMConvertHalfToFloat(h[0]);
        out[1] = XMConvertHalfToFloat(h[1]);
        out[2] = 0.0f;
        out[3] = 1.0f;
        break;
    }
    case DXGI_FORMAT_R16_FLOAT: {
        const auto* h = reinterpret_cast<const HALF*>(src);
        out[0] = XMConvertHalfToFloat(h[0]);
        out[1] = out[2] = 0.0f;
        out[3] = 1.0f;
        break;
    }
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
        out[0] = src[0] / 255.0f;
        out[1] = src[1] / 255.0f;
        out[2] = src[2] / 255.0f;
        out[3] = src[3] / 255.0f;
        if (fmt == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB) {
            out[0] = srgbToLinear(out[0]);
            out[1] = srgbToLinear(out[1]);
            out[2] = srgbToLinear(out[2]);
        }
        break;
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
        out[0] = src[2] / 255.0f;
        out[1] = src[1] / 255.0f;
        out[2] = src[0] / 255.0f;
        out[3] = src[3] / 255.0f;
        if (fmt == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB) {
            out[0] = srgbToLinear(out[0]);
            out[1] = srgbToLinear(out[1]);
            out[2] = srgbToLinear(out[2]);
        }
        break;
    case DXGI_FORMAT_R10G10B10A2_UNORM: {
        XMUDECN4 v;
        std::memcpy(&v.v, src, 4);
        XMVECTOR vec = XMLoadUDecN4(&v);
        XMStoreFloat4(reinterpret_cast<XMFLOAT4*>(out), vec);
        break;
    }
    case DXGI_FORMAT_R11G11B10_FLOAT: {
        XMFLOAT3PK v;
        std::memcpy(&v.v, src, 4);
        XMVECTOR vec = XMLoadFloat3PK(&v);
        XMStoreFloat3(reinterpret_cast<XMFLOAT3*>(out), vec);
        out[3] = 1.0f;
        break;
    }
    case DXGI_FORMAT_R8G8_UNORM:
        out[0] = src[0] / 255.0f;
        out[1] = src[1] / 255.0f;
        out[2] = 0.0f;
        out[3] = 1.0f;
        break;
    case DXGI_FORMAT_R8_UNORM:
        out[0] = src[0] / 255.0f;
        out[1] = out[2] = 0.0f;
        out[3] = 1.0f;
        break;
    case DXGI_FORMAT_A8_UNORM:
        out[0] = out[1] = out[2] = 0.0f;
        out[3] = src[0] / 255.0f;
        break;
    case DXGI_FORMAT_R16_UNORM:
    case DXGI_FORMAT_D16_UNORM: {
        uint16_t u;
        std::memcpy(&u, src, 2);
        out[0] = u / 65535.0f;
        out[1] = out[2] = 0.0f;
        out[3] = 1.0f;
        break;
    }
    case DXGI_FORMAT_D24_UNORM_S8_UINT: {
        uint32_t u;
        std::memcpy(&u, src, 4);
        out[0] = float(u & 0x00FFFFFFu) / 16777215.0f;
        out[1] = float((u >> 24) & 0xFFu);
        out[2] = 0.0f;
        out[3] = 1.0f;
        break;
    }
    default:
        out[0] = out[1] = out[2] = 0.0f;
        out[3] = 1.0f;
        break;
    }
}

inline void packPixelRGBA32F(DXGI_FORMAT fmt, const float in[4], uint8_t* dst) {
    using namespace DirectX;
    using namespace DirectX::PackedVector;
    switch (fmt) {
    case DXGI_FORMAT_R32G32B32A32_FLOAT:
        std::memcpy(dst, in, 16);
        break;
    case DXGI_FORMAT_R32G32B32_FLOAT:
        std::memcpy(dst, in, 12);
        break;
    case DXGI_FORMAT_R32G32_FLOAT:
        std::memcpy(dst, in, 8);
        break;
    case DXGI_FORMAT_R32_FLOAT:
    case DXGI_FORMAT_D32_FLOAT:
        std::memcpy(dst, in, 4);
        break;
    case DXGI_FORMAT_R16G16B16A16_FLOAT: {
        auto* h = reinterpret_cast<HALF*>(dst);
        h[0] = XMConvertFloatToHalf(in[0]);
        h[1] = XMConvertFloatToHalf(in[1]);
        h[2] = XMConvertFloatToHalf(in[2]);
        h[3] = XMConvertFloatToHalf(in[3]);
        break;
    }
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: {
        float r = fmt == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB ? linearToSrgb(in[0]) : in[0];
        float g = fmt == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB ? linearToSrgb(in[1]) : in[1];
        float b = fmt == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB ? linearToSrgb(in[2]) : in[2];
        dst[0] = static_cast<uint8_t>(std::clamp(r, 0.0f, 1.0f) * 255.0f + 0.5f);
        dst[1] = static_cast<uint8_t>(std::clamp(g, 0.0f, 1.0f) * 255.0f + 0.5f);
        dst[2] = static_cast<uint8_t>(std::clamp(b, 0.0f, 1.0f) * 255.0f + 0.5f);
        dst[3] = static_cast<uint8_t>(std::clamp(in[3], 0.0f, 1.0f) * 255.0f + 0.5f);
        break;
    }
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: {
        float r = fmt == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB ? linearToSrgb(in[0]) : in[0];
        float g = fmt == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB ? linearToSrgb(in[1]) : in[1];
        float b = fmt == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB ? linearToSrgb(in[2]) : in[2];
        dst[0] = static_cast<uint8_t>(std::clamp(b, 0.0f, 1.0f) * 255.0f + 0.5f);
        dst[1] = static_cast<uint8_t>(std::clamp(g, 0.0f, 1.0f) * 255.0f + 0.5f);
        dst[2] = static_cast<uint8_t>(std::clamp(r, 0.0f, 1.0f) * 255.0f + 0.5f);
        dst[3] = static_cast<uint8_t>(std::clamp(in[3], 0.0f, 1.0f) * 255.0f + 0.5f);
        break;
    }
    case DXGI_FORMAT_R10G10B10A2_UNORM: {
        XMUDECN4 v;
        XMStoreUDecN4(&v, XMVectorSet(in[0], in[1], in[2], in[3]));
        std::memcpy(dst, &v.v, 4);
        break;
    }
    case DXGI_FORMAT_R11G11B10_FLOAT: {
        XMFLOAT3PK v;
        XMStoreFloat3PK(&v, XMVectorSet(in[0], in[1], in[2], 1.0f));
        std::memcpy(dst, &v.v, 4);
        break;
    }
    default:
        break;
    }
}

/// Decompresses a single 4x4 BC1-BC7 block into 16 RGBA32F pixels (64 floats).
void decompressBCBlockRGBA32F(DXGI_FORMAT fmt, const uint8_t* block, float out16x4[64]);

} // namespace nacelle
