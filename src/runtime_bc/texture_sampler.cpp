#include <stddef.h>
#include <stdint.h>

extern "C" {

// Must stay binary-compatible with nacelle::NacelleMipLayout / NacelleTextureMetadata / NacelleDescriptor
struct BC_MipLayout {
    uint64_t offset;
    uint32_t row_pitch;
    uint32_t slice_pitch;
    uint32_t width;
    uint32_t height;
    uint32_t depth;
};

struct BC_TextureMetadata {
    uint32_t array_size;
    uint32_t mip_levels;
    uint32_t most_detailed_mip;
    uint32_t dimension;
    uint32_t is_cubemap;
    uint32_t is_compressed;
    uint32_t resource_planes;
    uint32_t reserved;
    uint64_t array_slice_stride;
    const float* decoded_rgba32f;
    BC_MipLayout mips[16];
    BC_MipLayout decoded_mips[16];
};

struct BC_Descriptor {
    void* base_ptr;
    uint32_t width;
    uint32_t height;
    uint32_t depth_or_array;
    uint32_t row_pitch;
    uint32_t format;
    uint32_t mip_levels;
    uint32_t sampler_flags;
    uint32_t elem_stride;
    uint64_t byte_size;
    void* extra_ptr;
    float mip_lod_bias;
    uint32_t view_flags;
};

static inline __attribute__((always_inline)) float f_floor(float x) { return __builtin_floorf(x); }
static inline __attribute__((always_inline)) float f_abs(float x) { return __builtin_fabsf(x); }
static inline __attribute__((always_inline)) float f_sqrt(float x) { return __builtin_sqrtf(x > 0.0f ? x : 0.0f); }
static inline __attribute__((always_inline)) float f_log2(float x) { return __builtin_log2f(x > 1e-20f ? x : 1e-20f); }
static inline __attribute__((always_inline)) float f_pow(float b, float e) {
    return b > 0.0f ? __builtin_exp2f(e * __builtin_log2f(b)) : 0.0f;
}
static inline __attribute__((always_inline)) float f_min(float a, float b) { return a < b ? a : b; }
static inline __attribute__((always_inline)) float f_max(float a, float b) { return a > b ? a : b; }
static inline __attribute__((always_inline)) float f_clamp(float x, float lo, float hi) {
    return f_min(f_max(x, lo), hi);
}
static inline __attribute__((always_inline)) float f_lerp(float a, float b, float t) {
    return __builtin_fmaf(t, b - a, a);
}
static inline __attribute__((always_inline)) float srgb_to_lin(float c) {
    return (c <= 0.04045f) ? (c / 12.92f) : f_pow((c + 0.055f) / 1.055f, 2.4f);
}
static inline __attribute__((always_inline)) float lin_to_srgb(float c) {
    c = f_clamp(c, 0.0f, 1.0f);
    return (c <= 0.0031308f) ? (c * 12.92f) : (1.055f * f_pow(c, 1.0f / 2.4f) - 0.055f);
}

static inline __attribute__((always_inline)) float half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t(h & 0x8000u)) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x3FFu;
            bits = sign | ((exp + 112u) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float res;
    __builtin_memcpy(&res, &bits, 4);
    return res;
}

static inline __attribute__((always_inline)) uint16_t float_to_half(float f) {
    uint32_t x = 0;
    __builtin_memcpy(&x, &f, 4);
    uint16_t sign = static_cast<uint16_t>((x >> 16) & 0x8000u);
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (exp <= 0) return sign;
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
    return static_cast<uint16_t>(sign | (uint32_t(exp) << 10) | (mant >> 13));
}

static inline __attribute__((always_inline)) void read_texel_ptr(
    const uint8_t* ptr, uint32_t format, float out[4]) {
    // DXGI_FORMAT values used in D3D12
    switch (format) {
    case 2: // R32G32B32A32_FLOAT
    case 3: // R32G32B32A32_UINT
    case 4: // R32G32B32A32_SINT
        __builtin_memcpy(out, ptr, 16);
        return;
    case 10: { // R16G16B16A16_FLOAT
        const uint16_t* h = reinterpret_cast<const uint16_t*>(ptr);
        out[0] = half_to_float(h[0]);
        out[1] = half_to_float(h[1]);
        out[2] = half_to_float(h[2]);
        out[3] = half_to_float(h[3]);
        return;
    }
    case 16: // R32G32_FLOAT
    case 17: // R32G32_UINT
    case 18: // R32G32_SINT
        __builtin_memcpy(out, ptr, 8);
        out[2] = 0.0f;
        out[3] = 1.0f;
        return;
    case 24: { // R10G10B10A2_UNORM
        uint32_t v;
        __builtin_memcpy(&v, ptr, 4);
        out[0] = float(v & 0x3FFu) / 1023.0f;
        out[1] = float((v >> 10) & 0x3FFu) / 1023.0f;
        out[2] = float((v >> 20) & 0x3FFu) / 1023.0f;
        out[3] = float((v >> 30) & 0x3u) / 3.0f;
        return;
    }
    case 28: // R8G8B8A8_UNORM
    case 29: // R8G8B8A8_UNORM_SRGB
        out[0] = ptr[0] * (1.0f / 255.0f);
        out[1] = ptr[1] * (1.0f / 255.0f);
        out[2] = ptr[2] * (1.0f / 255.0f);
        out[3] = ptr[3] * (1.0f / 255.0f);
        if (format == 29) {
            out[0] = srgb_to_lin(out[0]);
            out[1] = srgb_to_lin(out[1]);
            out[2] = srgb_to_lin(out[2]);
        }
        return;
    case 87: // B8G8R8A8_UNORM
    case 91: // B8G8R8A8_UNORM_SRGB
        out[0] = ptr[2] * (1.0f / 255.0f);
        out[1] = ptr[1] * (1.0f / 255.0f);
        out[2] = ptr[0] * (1.0f / 255.0f);
        out[3] = ptr[3] * (1.0f / 255.0f);
        if (format == 91) {
            out[0] = srgb_to_lin(out[0]);
            out[1] = srgb_to_lin(out[1]);
            out[2] = srgb_to_lin(out[2]);
        }
        return;
    case 40: // D32_FLOAT
    case 41: // R32_FLOAT
    case 39: // R32_TYPELESS
    case 42: // R32_UINT
    case 43: // R32_SINT
        __builtin_memcpy(out, ptr, 4);
        out[1] = out[2] = 0.0f;
        out[3] = 1.0f;
        return;
    case 45: { // D24_UNORM_S8_UINT
        uint32_t v;
        __builtin_memcpy(&v, ptr, 4);
        out[0] = float(v & 0x00FFFFFFu) / 16777215.0f;
        out[1] = out[2] = 0.0f;
        out[3] = 1.0f;
        return;
    }
    case 49: // R8G8_UNORM
        out[0] = ptr[0] * (1.0f / 255.0f);
        out[1] = ptr[1] * (1.0f / 255.0f);
        out[2] = 0.0f;
        out[3] = 1.0f;
        return;
    case 54: { // R16_FLOAT
        uint16_t h;
        __builtin_memcpy(&h, ptr, 2);
        out[0] = half_to_float(h);
        out[1] = out[2] = 0.0f;
        out[3] = 1.0f;
        return;
    }
    case 55: // D16_UNORM
    case 56: { // R16_UNORM
        uint16_t u;
        __builtin_memcpy(&u, ptr, 2);
        out[0] = float(u) / 65535.0f;
        out[1] = out[2] = 0.0f;
        out[3] = 1.0f;
        return;
    }
    case 61: // R8_UNORM
        out[0] = ptr[0] * (1.0f / 255.0f);
        out[1] = out[2] = 0.0f;
        out[3] = 1.0f;
        return;
    case 65: // A8_UNORM
        out[0] = out[1] = out[2] = 0.0f;
        out[3] = ptr[0] * (1.0f / 255.0f);
        return;
    default:
        out[0] = out[1] = out[2] = 0.0f;
        out[3] = 1.0f;
        return;
    }
}

static inline __attribute__((always_inline)) void write_texel_ptr(
    uint8_t* ptr, uint32_t format, const float in[4]) {
    switch (format) {
    case 2: // R32G32B32A32_FLOAT
    case 3:
    case 4:
        __builtin_memcpy(ptr, in, 16);
        return;
    case 10: { // R16G16B16A16_FLOAT
        uint16_t* h = reinterpret_cast<uint16_t*>(ptr);
        h[0] = float_to_half(in[0]);
        h[1] = float_to_half(in[1]);
        h[2] = float_to_half(in[2]);
        h[3] = float_to_half(in[3]);
        return;
    }
    case 16: // R32G32_FLOAT
    case 17: // R32G32_UINT
    case 18: // R32G32_SINT
        __builtin_memcpy(ptr, in, 8);
        return;
    case 28:
    case 29: {
        float r = (format == 29) ? lin_to_srgb(in[0]) : f_clamp(in[0], 0.0f, 1.0f);
        float g = (format == 29) ? lin_to_srgb(in[1]) : f_clamp(in[1], 0.0f, 1.0f);
        float b = (format == 29) ? lin_to_srgb(in[2]) : f_clamp(in[2], 0.0f, 1.0f);
        float a = f_clamp(in[3], 0.0f, 1.0f);
        ptr[0] = static_cast<uint8_t>(r * 255.0f + 0.5f);
        ptr[1] = static_cast<uint8_t>(g * 255.0f + 0.5f);
        ptr[2] = static_cast<uint8_t>(b * 255.0f + 0.5f);
        ptr[3] = static_cast<uint8_t>(a * 255.0f + 0.5f);
        return;
    }
    case 87:
    case 91: {
        float r = (format == 91) ? lin_to_srgb(in[0]) : f_clamp(in[0], 0.0f, 1.0f);
        float g = (format == 91) ? lin_to_srgb(in[1]) : f_clamp(in[1], 0.0f, 1.0f);
        float b = (format == 91) ? lin_to_srgb(in[2]) : f_clamp(in[2], 0.0f, 1.0f);
        float a = f_clamp(in[3], 0.0f, 1.0f);
        ptr[0] = static_cast<uint8_t>(b * 255.0f + 0.5f);
        ptr[1] = static_cast<uint8_t>(g * 255.0f + 0.5f);
        ptr[2] = static_cast<uint8_t>(r * 255.0f + 0.5f);
        ptr[3] = static_cast<uint8_t>(a * 255.0f + 0.5f);
        return;
    }
    case 39:
    case 40:
    case 41:
    case 42:
    case 43:
        __builtin_memcpy(ptr, in, 4);
        return;
    case 61:
        ptr[0] = static_cast<uint8_t>(f_clamp(in[0], 0.0f, 1.0f) * 255.0f + 0.5f);
        return;
    default:
        return;
    }
}

static inline __attribute__((always_inline)) int32_t apply_address_mode(
    int32_t coord, uint32_t dim, uint32_t mode /*0=wrap,1=mirror,2=clamp,3=border,4=mirror_once*/, bool& outBorder) {
    int32_t idim = static_cast<int32_t>(dim);
    if (idim <= 0) { outBorder = true; return 0; }
    switch (mode) {
    case 0: { // WRAP
        int32_t m = coord % idim;
        return m < 0 ? (m + idim) : m;
    }
    case 1: { // MIRROR
        int32_t period = idim * 2;
        int32_t m = coord % period;
        if (m < 0) m += period;
        return m >= idim ? (period - 1 - m) : m;
    }
    case 2: // CLAMP
        return coord < 0 ? 0 : (coord >= idim ? (idim - 1) : coord);
    case 3: // BORDER
        if (coord < 0 || coord >= idim) { outBorder = true; return 0; }
        return coord;
    case 4: { // MIRROR_ONCE
        int32_t a = coord < 0 ? (-coord - 1) : coord;
        return a >= idim ? (idim - 1) : a;
    }
    default:
        return coord < 0 ? 0 : (coord >= idim ? (idim - 1) : coord);
    }
}

static inline __attribute__((always_inline)) void fetch_texel_mip(
    const BC_Descriptor* tex, uint32_t mip, int32_t x, int32_t y, int32_t z,
    uint32_t addrU, uint32_t addrV, uint32_t addrW, uint32_t borderKind, float out[4]) {
    const auto* meta = reinterpret_cast<const BC_TextureMetadata*>(tex->extra_ptr);
    uint32_t viewMip = ((tex->view_flags >> 8) & 0xFFu) + mip;
    if (meta && viewMip >= meta->mip_levels) viewMip = meta->mip_levels - 1;

    uint32_t w = tex->width >> mip;
    uint32_t h = tex->height >> mip;
    uint32_t d = 1;
    uint32_t rowPitch = tex->row_pitch;
    uint32_t slicePitch = rowPitch * h;
    uint64_t offset = 0;

    if (meta) {
        if (meta->is_compressed && meta->decoded_rgba32f) {
            const auto& dm = meta->decoded_mips[viewMip];
            w = dm.width;
            h = dm.height;
            d = dm.depth;
            bool border = false;
            int32_t sx = apply_address_mode(x, w, addrU, border);
            int32_t sy = apply_address_mode(y, h, addrV, border);
            int32_t sz = (d > 1) ? apply_address_mode(z, d, addrW, border) : (z < 0 ? 0 : (z >= int32_t(meta->array_size) ? int32_t(meta->array_size) - 1 : z));
            if (border) {
                float bv = (borderKind == 2) ? 1.0f : 0.0f;
                float ba = (borderKind >= 1) ? 1.0f : 0.0f;
                out[0] = out[1] = out[2] = bv;
                out[3] = ba;
                return;
            }
            const float* src = meta->decoded_rgba32f + (dm.offset / sizeof(float)) +
                               (size_t(sz) * w * h + size_t(sy) * w + size_t(sx)) * 4;
            out[0] = src[0];
            out[1] = src[1];
            out[2] = src[2];
            out[3] = src[3];
            return;
        }
        const auto& m = meta->mips[viewMip];
        w = m.width;
        h = m.height;
        d = m.depth;
        rowPitch = m.row_pitch;
        slicePitch = m.slice_pitch;
        offset = m.offset;
    }

    if (w == 0) w = 1;
    if (h == 0) h = 1;

    bool border = false;
    int32_t sx = apply_address_mode(x, w, addrU, border);
    int32_t sy = apply_address_mode(y, h, addrV, border);
    int32_t sz = 0;
    if (d > 1) {
        sz = apply_address_mode(z, d, addrW, border);
    } else if (meta && meta->array_size > 1) {
        sz = z < 0 ? 0 : (z >= int32_t(meta->array_size) ? int32_t(meta->array_size) - 1 : z);
    }

    if (border) {
        float bv = (borderKind == 2) ? 1.0f : 0.0f;
        float ba = (borderKind >= 1) ? 1.0f : 0.0f;
        out[0] = out[1] = out[2] = bv;
        out[3] = ba;
        return;
    }

    const uint8_t* base = static_cast<const uint8_t*>(tex->base_ptr) + offset;
    if (sz > 0 && meta && meta->array_size > 1 && d == 1) {
        base += uint64_t(sz) * meta->array_slice_stride;
    } else if (sz > 0 && d > 1) {
        base += uint64_t(sz) * slicePitch;
    }

    const uint8_t* p = base + uint64_t(sy) * rowPitch + uint64_t(sx) * tex->elem_stride;
    read_texel_ptr(p, tex->format, out);
}

static inline __attribute__((always_inline)) void sample_mip_2d(
    const BC_Descriptor* tex, uint32_t mip, float u, float v, int32_t slice,
    bool linearFilter, uint32_t addrU, uint32_t addrV, uint32_t addrW, uint32_t borderKind,
    int32_t offX, int32_t offY, float out[4]) {
    uint32_t w = tex->width >> mip;
    uint32_t h = tex->height >> mip;
    if (w == 0) w = 1;
    if (h == 0) h = 1;

    if (!linearFilter) {
        int32_t ix = static_cast<int32_t>(f_floor(u * float(w))) + offX;
        int32_t iy = static_cast<int32_t>(f_floor(v * float(h))) + offY;
        fetch_texel_mip(tex, mip, ix, iy, slice, addrU, addrV, addrW, borderKind, out);
        return;
    }

    float fx = u * float(w) - 0.5f + float(offX);
    float fy = v * float(h) - 0.5f + float(offY);
    int32_t x0 = static_cast<int32_t>(f_floor(fx));
    int32_t y0 = static_cast<int32_t>(f_floor(fy));
    float tx = fx - float(x0);
    float ty = fy - float(y0);

    float c00[4], c10[4], c01[4], c11[4];
    fetch_texel_mip(tex, mip, x0, y0, slice, addrU, addrV, addrW, borderKind, c00);
    fetch_texel_mip(tex, mip, x0 + 1, y0, slice, addrU, addrV, addrW, borderKind, c10);
    fetch_texel_mip(tex, mip, x0, y0 + 1, slice, addrU, addrV, addrW, borderKind, c01);
    fetch_texel_mip(tex, mip, x0 + 1, y0 + 1, slice, addrU, addrV, addrW, borderKind, c11);

    for (int i = 0; i < 4; ++i) {
        float top = f_lerp(c00[i], c10[i], tx);
        float bot = f_lerp(c01[i], c11[i], tx);
        out[i] = f_lerp(top, bot, ty);
    }
}

static inline __attribute__((always_inline)) void map_cube_coords(
    float x, float y, float z, float& outU, float& outV, int32_t& outFace) {
    float ax = f_abs(x), ay = f_abs(y), az = f_abs(z);
    float ma, sc, tc;
    if (ax >= ay && ax >= az) {
        ma = ax;
        if (x >= 0.0f) { outFace = 0; sc = -z; tc = -y; }
        else           { outFace = 1; sc =  z; tc = -y; }
    } else if (ay >= ax && ay >= az) {
        ma = ay;
        if (y >= 0.0f) { outFace = 2; sc =  x; tc =  z; }
        else           { outFace = 3; sc =  x; tc = -z; }
    } else {
        ma = az;
        if (z >= 0.0f) { outFace = 4; sc =  x; tc = -y; }
        else           { outFace = 5; sc = -x; tc = -y; }
    }
    float invMa = (ma > 1e-12f) ? (0.5f / ma) : 0.0f;
    outU = sc * invMa + 0.5f;
    outV = tc * invMa + 0.5f;
}

static inline __attribute__((always_inline)) void resolve_sample_coords(
    const BC_Descriptor* tex, float u, float v, float w,
    float& su, float& sv, int32_t& slice, uint32_t& addrU, uint32_t& addrV) {
    uint32_t dim = tex->view_flags & 0xFFu;
    uint32_t firstArraySlice = (tex->view_flags >> 16) & 0xFFFFu;
    slice = static_cast<int32_t>(firstArraySlice);
    su = u;
    sv = v;
    if (dim == 2) { // TEXTURE1D
        sv = 0.5f;
    } else if (dim == 3) { // TEXTURE1DARRAY
        sv = 0.5f;
        slice += static_cast<int32_t>(v + 0.5f);
    } else if (dim == 5 || dim == 7) { // TEXTURE2DARRAY / TEXTURE2DMSARRAY
        slice += static_cast<int32_t>(w + 0.5f);
    } else if (dim == 9 || dim == 10) { // TEXTURECUBE / TEXTURECUBEARRAY
        int32_t face = 0;
        map_cube_coords(u, v, w, su, sv, face);
        slice += face;
        addrU = addrV = 2; // CLAMP on cube faces
    }
}

void nacelle_tex_sample(
    const BC_Descriptor* tex,
    const BC_Descriptor* samp,
    float u, float v, float w,
    float ddx_u, float ddx_v,
    float ddy_u, float ddy_v,
    float bias, float explicitLod,
    uint32_t lodMode, // 0=grad(ddx/ddy), 1=bias, 2=explicitLevel
    int32_t offX, int32_t offY, int32_t offZ,
    float clampLod,
    float* outRGBA) {
    (void)offZ;
    if (!tex || !tex->base_ptr) {
        outRGBA[0] = outRGBA[1] = outRGBA[2] = 0.0f;
        outRGBA[3] = 1.0f;
        return;
    }

    uint32_t sflags = samp ? samp->sampler_flags : 0x15u; // default linear wrap
    uint32_t filter = sflags & 0xFFu;
    uint32_t addrU = (sflags >> 8) & 0x7u;
    uint32_t addrV = (sflags >> 11) & 0x7u;
    uint32_t addrW = (sflags >> 14) & 0x7u;
    uint32_t borderKind = (sflags >> 26) & 0x3u;
    float sampBias = samp ? samp->mip_lod_bias : 0.0f;
    float minLod = samp ? float(samp->view_flags & 0xFFFFu) / 256.0f : 0.0f;
    float maxLod = samp ? float((samp->view_flags >> 16) & 0xFFFFu) / 256.0f : 16.0f;

    int32_t slice = 0;
    float su = u, sv = v;
    resolve_sample_coords(tex, u, v, w, su, sv, slice, addrU, addrV);

    bool linearMag = (filter & 0x04u) != 0;
    bool linearMin = (filter & 0x10u) != 0;
    bool linearMip = (filter & 0x01u) != 0;
    bool isAniso = (filter == 0x55u || filter == 0xD5u);

    float lod = 0.0f;
    if (lodMode == 2) {
        lod = explicitLod;
    } else {
        float fw = float(tex->width);
        float fh = float(tex->height);
        float dx2 = ddx_u * ddx_u * fw * fw + ddx_v * ddx_v * fh * fh;
        float dy2 = ddy_u * ddy_u * fw * fw + ddy_v * ddy_v * fh * fh;
        float rho2 = f_max(dx2, dy2);
        if (isAniso) {
            float min2 = f_min(dx2, dy2);
            float maxAniso = samp ? float((sflags >> 21) & 0x1Fu) : 1.0f;
            float ratio = f_sqrt(rho2 / f_max(min2, 1e-12f));
            ratio = f_min(ratio, f_max(1.0f, maxAniso));
            rho2 = rho2 / (ratio * ratio);
        }
        lod = 0.5f * f_log2(f_max(rho2, 1e-12f)) + sampBias + (lodMode == 1 ? bias : 0.0f);
    }

    float maxMipF = float(tex->mip_levels > 0 ? tex->mip_levels - 1 : 0);
    lod = f_clamp(lod, minLod, f_min(maxLod, maxMipF));
    if (clampLod >= 0.0f) lod = f_max(lod, clampLod);

    bool useLinear = (lod <= 0.0f) ? linearMag : linearMin;

    if (!linearMip || tex->mip_levels <= 1) {
        uint32_t m0 = static_cast<uint32_t>(f_clamp(f_floor(lod + 0.5f), 0.0f, maxMipF));
        sample_mip_2d(tex, m0, su, sv, slice, useLinear, addrU, addrV, addrW, borderKind, offX, offY, outRGBA);
        return;
    }

    uint32_t m0 = static_cast<uint32_t>(f_clamp(f_floor(lod), 0.0f, maxMipF));
    uint32_t m1 = m0 + 1 < tex->mip_levels ? m0 + 1 : m0;
    float frac = lod - float(m0);
    float c0[4], c1[4];
    sample_mip_2d(tex, m0, su, sv, slice, useLinear, addrU, addrV, addrW, borderKind, offX, offY, c0);
    if (m1 == m0 || frac <= 0.001f) {
        outRGBA[0] = c0[0]; outRGBA[1] = c0[1]; outRGBA[2] = c0[2]; outRGBA[3] = c0[3];
        return;
    }
    sample_mip_2d(tex, m1, su, sv, slice, useLinear, addrU, addrV, addrW, borderKind, offX, offY, c1);
    for (int i = 0; i < 4; ++i) {
        outRGBA[i] = f_lerp(c0[i], c1[i], frac);
    }
}

static inline __attribute__((always_inline)) float eval_cmp(uint32_t func, float ref, float val) {
    switch (func) {
    case 1: return 0.0f; // NEVER
    case 2: return (ref < val) ? 1.0f : 0.0f; // LESS
    case 3: return (ref == val) ? 1.0f : 0.0f; // EQUAL
    case 4: return (ref <= val) ? 1.0f : 0.0f; // LESS_EQUAL
    case 5: return (ref > val) ? 1.0f : 0.0f; // GREATER
    case 6: return (ref != val) ? 1.0f : 0.0f; // NOT_EQUAL
    case 7: return (ref >= val) ? 1.0f : 0.0f; // GREATER_EQUAL
    case 8: return 1.0f; // ALWAYS
    default: return (ref <= val) ? 1.0f : 0.0f;
    }
}

float nacelle_tex_sample_cmp(
    const BC_Descriptor* tex,
    const BC_Descriptor* samp,
    float u, float v, float w,
    float cmpVal,
    float explicitLod,
    uint32_t isLevelZero,
    int32_t offX, int32_t offY) {
    if (!tex || !tex->base_ptr) return 0.0f;
    uint32_t sflags = samp ? samp->sampler_flags : 0x95u;
    uint32_t filter = sflags & 0xFFu;
    uint32_t addrU = (sflags >> 8) & 0x7u;
    uint32_t addrV = (sflags >> 11) & 0x7u;
    uint32_t addrW = (sflags >> 14) & 0x7u;
    uint32_t cmpFunc = (sflags >> 17) & 0xFu;
    uint32_t borderKind = (sflags >> 26) & 0x3u;
    uint32_t mip = isLevelZero ? 0 : static_cast<uint32_t>(f_clamp(explicitLod, 0.0f, float(tex->mip_levels > 0 ? tex->mip_levels - 1 : 0)));
    (void)w;

    uint32_t width = tex->width >> mip;
    uint32_t height = tex->height >> mip;
    if (width == 0) width = 1;
    if (height == 0) height = 1;

    bool linear = (filter & 0x14u) != 0;
    float t[4];
    if (!linear) {
        int32_t ix = static_cast<int32_t>(f_floor(u * float(width))) + offX;
        int32_t iy = static_cast<int32_t>(f_floor(v * float(height))) + offY;
        fetch_texel_mip(tex, mip, ix, iy, 0, addrU, addrV, addrW, borderKind, t);
        return eval_cmp(cmpFunc, cmpVal, t[0]);
    }

    float fx = u * float(width) - 0.5f + float(offX);
    float fy = v * float(height) - 0.5f + float(offY);
    int32_t x0 = static_cast<int32_t>(f_floor(fx));
    int32_t y0 = static_cast<int32_t>(f_floor(fy));
    float tx = fx - float(x0);
    float ty = fy - float(y0);

    fetch_texel_mip(tex, mip, x0, y0, 0, addrU, addrV, addrW, borderKind, t);
    float c00 = eval_cmp(cmpFunc, cmpVal, t[0]);
    fetch_texel_mip(tex, mip, x0 + 1, y0, 0, addrU, addrV, addrW, borderKind, t);
    float c10 = eval_cmp(cmpFunc, cmpVal, t[0]);
    fetch_texel_mip(tex, mip, x0, y0 + 1, 0, addrU, addrV, addrW, borderKind, t);
    float c01 = eval_cmp(cmpFunc, cmpVal, t[0]);
    fetch_texel_mip(tex, mip, x0 + 1, y0 + 1, 0, addrU, addrV, addrW, borderKind, t);
    float c11 = eval_cmp(cmpFunc, cmpVal, t[0]);

    return f_lerp(f_lerp(c00, c10, tx), f_lerp(c01, c11, tx), ty);
}

void nacelle_tex_load(
    const BC_Descriptor* tex,
    int32_t x, int32_t y, int32_t z, int32_t mip,
    float* outRGBA) {
    if (!tex || !tex->base_ptr || mip < 0 || uint32_t(mip) >= tex->mip_levels) {
        outRGBA[0] = outRGBA[1] = outRGBA[2] = outRGBA[3] = 0.0f;
        return;
    }
    uint32_t w = tex->width >> mip;
    uint32_t h = tex->height >> mip;
    if (w == 0) w = 1;
    if (h == 0) h = 1;
    if (x < 0 || y < 0 || uint32_t(x) >= w || uint32_t(y) >= h || z < 0 || uint32_t(z) >= tex->depth_or_array) {
        outRGBA[0] = outRGBA[1] = outRGBA[2] = outRGBA[3] = 0.0f;
        return;
    }
    fetch_texel_mip(tex, uint32_t(mip), x, y, z, 2, 2, 2, 0, outRGBA);
}

void nacelle_tex_store(
    const BC_Descriptor* tex,
    int32_t x, int32_t y, int32_t z,
    float r, float g, float b, float a) {
    if (!tex || !tex->base_ptr) return;
    if (x < 0 || y < 0 || z < 0 ||
        uint32_t(x) >= tex->width || uint32_t(y) >= tex->height || uint32_t(z) >= tex->depth_or_array) {
        return;
    }
    uint8_t* ptr = static_cast<uint8_t*>(tex->base_ptr) +
                   uint64_t(z) * tex->row_pitch * tex->height +
                   uint64_t(y) * tex->row_pitch +
                   uint64_t(x) * tex->elem_stride;
    float in[4]{r, g, b, a};
    write_texel_ptr(ptr, tex->format, in);
}

static inline __attribute__((always_inline)) void gather_four_texels(
    const BC_Descriptor* tex,
    const BC_Descriptor* samp,
    float u, float v, float w,
    int32_t offX, int32_t offY,
    uint32_t component,
    bool doCmp, float cmpVal,
    float* outRGBA) {
    if (!tex || !tex->base_ptr) {
        outRGBA[0] = outRGBA[1] = outRGBA[2] = outRGBA[3] = 0.0f;
        return;
    }
    uint32_t sflags = samp ? samp->sampler_flags : (doCmp ? 0x95u : 0x15u);
    uint32_t addrU = (sflags >> 8) & 0x7u;
    uint32_t addrV = (sflags >> 11) & 0x7u;
    uint32_t addrW = (sflags >> 14) & 0x7u;
    uint32_t cmpFunc = (sflags >> 17) & 0xFu;
    uint32_t borderKind = (sflags >> 26) & 0x3u;

    int32_t slice = 0;
    float su = u, sv = v;
    resolve_sample_coords(tex, u, v, w, su, sv, slice, addrU, addrV);

    float fx = su * float(tex->width) - 0.5f + float(offX);
    float fy = sv * float(tex->height) - 0.5f + float(offY);
    int32_t x0 = static_cast<int32_t>(f_floor(fx));
    int32_t y0 = static_cast<int32_t>(f_floor(fy));
    uint32_t c = component & 3u;

    const int32_t dx[4] = {0, 1, 1, 0};
    const int32_t dy[4] = {1, 1, 0, 0};
    float t[4];
    // D3D12 Gather ordering: x = (-,+), y = (+,+), z = (+,-), w = (-,-)
    for (int i = 0; i < 4; ++i) {
        fetch_texel_mip(tex, 0, x0 + dx[i], y0 + dy[i], slice, addrU, addrV, addrW, borderKind, t);
        outRGBA[i] = doCmp ? eval_cmp(cmpFunc, cmpVal, t[c]) : t[c];
    }
}

void nacelle_tex_gather(
    const BC_Descriptor* tex,
    const BC_Descriptor* samp,
    float u, float v, float w,
    int32_t offX, int32_t offY,
    uint32_t component,
    float* outRGBA) {
    gather_four_texels(tex, samp, u, v, w, offX, offY, component, false, 0.0f, outRGBA);
}

void nacelle_tex_gather_cmp(
    const BC_Descriptor* tex,
    const BC_Descriptor* samp,
    float u, float v, float w,
    int32_t offX, int32_t offY,
    uint32_t component,
    float cmpVal,
    float* outRGBA) {
    gather_four_texels(tex, samp, u, v, w, offX, offY, component, true, cmpVal, outRGBA);
}

void nacelle_tex_project_lod_coords(
    const BC_Descriptor* tex,
    float u, float v, float w,
    float* outUVW) {
    if (!tex) {
        outUVW[0] = u;
        outUVW[1] = v;
        outUVW[2] = 0.0f;
        return;
    }
    uint32_t dim = tex->view_flags & 0xFFu;
    if (dim == 2 || dim == 3) { // TEXTURE1D / TEXTURE1DARRAY
        outUVW[0] = u;
        outUVW[1] = 0.0f;
        outUVW[2] = 0.0f;
    } else if (dim == 8) { // TEXTURE3D
        outUVW[0] = u;
        outUVW[1] = v;
        outUVW[2] = w;
    } else if (dim == 9 || dim == 10) { // TEXTURECUBE / TEXTURECUBEARRAY
        int32_t face = 0;
        map_cube_coords(u, v, w, outUVW[0], outUVW[1], face);
        outUVW[2] = 0.0f;
    } else { // TEXTURE2D / TEXTURE2DARRAY
        outUVW[0] = u;
        outUVW[1] = v;
        outUVW[2] = 0.0f;
    }
}

float nacelle_tex_calc_lod(
    const BC_Descriptor* tex,
    const BC_Descriptor* samp,
    float ddx_u, float ddx_v, float ddx_w,
    float ddy_u, float ddy_v, float ddy_w,
    uint32_t clamped) {
    if (!tex) return 0.0f;
    uint32_t dim = tex->view_flags & 0xFFu;
    uint32_t sflags = samp ? samp->sampler_flags : 0x15u;
    uint32_t filter = sflags & 0xFFu;
    bool isAniso = (filter == 0x55u || filter == 0xD5u);
    float sampBias = samp ? samp->mip_lod_bias : 0.0f;
    float minLod = samp ? float(samp->view_flags & 0xFFFFu) / 256.0f : 0.0f;
    float maxLod = samp ? float((samp->view_flags >> 16) & 0xFFFFu) / 256.0f : 16.0f;

    float fw = float(tex->width);
    float fh = (dim == 3 || dim == 4) ? 0.0f : float(tex->height);
    float fd = (dim == 8) ? float(tex->depth_or_array) : 0.0f;

    float dx2 = ddx_u * ddx_u * fw * fw + ddx_v * ddx_v * fh * fh + ddx_w * ddx_w * fd * fd;
    float dy2 = ddy_u * ddy_u * fw * fw + ddy_v * ddy_v * fh * fh + ddy_w * ddy_w * fd * fd;
    float rho2 = f_max(dx2, dy2);
    if (isAniso) {
        float min2 = f_min(dx2, dy2);
        float maxAniso = samp ? float((sflags >> 21) & 0x1Fu) : 1.0f;
        float ratio = f_sqrt(rho2 / f_max(min2, 1e-24f));
        ratio = f_min(ratio, f_max(1.0f, maxAniso));
        rho2 = rho2 / (ratio * ratio);
    }

    float lod = 0.5f * f_log2(f_max(rho2, 1e-24f)) + sampBias;
    if (clamped) {
        float maxMipF = float(tex->mip_levels > 0 ? tex->mip_levels - 1 : 0);
        lod = f_clamp(lod, minLod, f_min(maxLod, maxMipF));
    }
    return lod;
}

void nacelle_tex_get_dims(const BC_Descriptor* tex, uint32_t mip, uint32_t* outDims4) {
    if (!tex) {
        outDims4[0] = outDims4[1] = outDims4[2] = outDims4[3] = 0;
        return;
    }
    // Raw buffers (ByteAddressBuffer / RWByteAddressBuffer) report total byte size.
    if ((tex->view_flags & 0xFFu) == 1u && (tex->view_flags & 0x100u) != 0u) {
        outDims4[0] = static_cast<uint32_t>(tex->byte_size);
        outDims4[1] = 0;
        outDims4[2] = 0;
        outDims4[3] = 0;
        return;
    }
    uint32_t w = tex->width >> mip;
    uint32_t h = tex->height >> mip;
    if (w == 0 && tex->width > 0) w = 1;
    if (h == 0 && tex->height > 0) h = 1;
    outDims4[0] = w;
    outDims4[1] = h;
    outDims4[2] = tex->depth_or_array;
    outDims4[3] = tex->mip_levels;
}

} // extern "C"
