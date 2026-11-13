#define BCDEC_IMPLEMENTATION
#define BCDEC_BC4BC5_PRECISE
#define BCDEC_STATIC
#include <bcdec.h>

#include "format_utils.h"

namespace nacelle {

void decompressBCBlockRGBA32F(DXGI_FORMAT fmt, const uint8_t* block, float out16x4[64]) {
    uint8_t rgba8[16 * 4];
    bool isSRGB = false;

    switch (fmt) {
    case DXGI_FORMAT_BC1_UNORM_SRGB:
        isSRGB = true;
        [[fallthrough]];
    case DXGI_FORMAT_BC1_UNORM:
    case DXGI_FORMAT_BC1_TYPELESS:
        bcdec_bc1(block, rgba8, 16);
        break;
    case DXGI_FORMAT_BC2_UNORM_SRGB:
        isSRGB = true;
        [[fallthrough]];
    case DXGI_FORMAT_BC2_UNORM:
    case DXGI_FORMAT_BC2_TYPELESS:
        bcdec_bc2(block, rgba8, 16);
        break;
    case DXGI_FORMAT_BC3_UNORM_SRGB:
        isSRGB = true;
        [[fallthrough]];
    case DXGI_FORMAT_BC3_UNORM:
    case DXGI_FORMAT_BC3_TYPELESS:
        bcdec_bc3(block, rgba8, 16);
        break;
    case DXGI_FORMAT_BC7_UNORM_SRGB:
        isSRGB = true;
        [[fallthrough]];
    case DXGI_FORMAT_BC7_UNORM:
    case DXGI_FORMAT_BC7_TYPELESS:
        bcdec_bc7(block, rgba8, 16);
        break;
    case DXGI_FORMAT_BC4_UNORM:
    case DXGI_FORMAT_BC4_TYPELESS:
    case DXGI_FORMAT_BC4_SNORM: {
        float r[16];
        bcdec_bc4_float(block, r, 4 * sizeof(float), fmt == DXGI_FORMAT_BC4_SNORM ? 1 : 0);
        for (int i = 0; i < 16; ++i) {
            out16x4[i * 4 + 0] = r[i];
            out16x4[i * 4 + 1] = 0.0f;
            out16x4[i * 4 + 2] = 0.0f;
            out16x4[i * 4 + 3] = 1.0f;
        }
        return;
    }
    case DXGI_FORMAT_BC5_UNORM:
    case DXGI_FORMAT_BC5_TYPELESS:
    case DXGI_FORMAT_BC5_SNORM: {
        float rg[32];
        bcdec_bc5_float(block, rg, 4 * 2 * sizeof(float), fmt == DXGI_FORMAT_BC5_SNORM ? 1 : 0);
        for (int i = 0; i < 16; ++i) {
            out16x4[i * 4 + 0] = rg[i * 2 + 0];
            out16x4[i * 4 + 1] = rg[i * 2 + 1];
            out16x4[i * 4 + 2] = 0.0f;
            out16x4[i * 4 + 3] = 1.0f;
        }
        return;
    }
    case DXGI_FORMAT_BC6H_UF16:
    case DXGI_FORMAT_BC6H_TYPELESS:
    case DXGI_FORMAT_BC6H_SF16: {
        float rgb[48];
        bcdec_bc6h_float(block, rgb, 4 * 3 * sizeof(float), fmt == DXGI_FORMAT_BC6H_SF16 ? 1 : 0);
        for (int i = 0; i < 16; ++i) {
            out16x4[i * 4 + 0] = rgb[i * 3 + 0];
            out16x4[i * 4 + 1] = rgb[i * 3 + 1];
            out16x4[i * 4 + 2] = rgb[i * 3 + 2];
            out16x4[i * 4 + 3] = 1.0f;
        }
        return;
    }
    default:
        std::memset(out16x4, 0, sizeof(float) * 64);
        return;
    }

    for (int i = 0; i < 16; ++i) {
        float r = rgba8[i * 4 + 0] / 255.0f;
        float g = rgba8[i * 4 + 1] / 255.0f;
        float b = rgba8[i * 4 + 2] / 255.0f;
        float a = rgba8[i * 4 + 3] / 255.0f;
        if (isSRGB) {
            r = srgbToLinear(r);
            g = srgbToLinear(g);
            b = srgbToLinear(b);
        }
        out16x4[i * 4 + 0] = r;
        out16x4[i * 4 + 1] = g;
        out16x4[i * 4 + 2] = b;
        out16x4[i * 4 + 3] = a;
    }
}

} // namespace nacelle
