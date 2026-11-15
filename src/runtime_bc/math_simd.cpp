#include <stdint.h>

namespace {

template <int W>
using vf32 = float __attribute__((ext_vector_type(W)));

template <int W>
using vi32 = int32_t __attribute__((ext_vector_type(W)));

template <int W>
using vu32 = uint32_t __attribute__((ext_vector_type(W)));

template <typename Dst, typename Src>
inline Dst bit_cast_v(Src s) {
    Dst d;
    __builtin_memcpy(&d, &s, sizeof(d));
    return d;
}

template <int W>
inline vf32<W> splat_f(float v) {
    return (vf32<W>)(v);
}

template <int W>
inline vi32<W> splat_i(int32_t v) {
    return (vi32<W>)(v);
}

template <int W>
inline vu32<W> splat_u(uint32_t v) {
    return (vu32<W>)(v);
}

template <int W>
inline vf32<W> abs_v(vf32<W> x) {
    return bit_cast_v<vf32<W>>(bit_cast_v<vu32<W>>(x) & splat_u<W>(0x7FFFFFFFu));
}

template <int W>
inline vf32<W> copysign_mag_v(vf32<W> mag, vf32<W> sign_src) {
    vu32<W> m = bit_cast_v<vu32<W>>(mag) & splat_u<W>(0x7FFFFFFFu);
    vu32<W> s = bit_cast_v<vu32<W>>(sign_src) & splat_u<W>(0x80000000u);
    return bit_cast_v<vf32<W>>(m | s);
}

template <int W>
inline vf32<W> select_v(vi32<W> mask, vf32<W> t, vf32<W> f) {
    return mask ? t : f;
}

template <int W>
inline vi32<W> select_i(vi32<W> mask, vi32<W> t, vi32<W> f) {
    return mask ? t : f;
}

template <int W>
inline vf32<W> min_v(vf32<W> a, vf32<W> b) {
    return (a < b) ? a : b;
}

// NaN-safe clamp: maps NaN / <lo to lo, >hi to hi, so rint + fptosi never see Inf/NaN/overflow.
template <int W>
inline vf32<W> clamp_finite_v(vf32<W> x, float lo, float hi) {
    vf32<W> a = select_v<W>(x >= splat_f<W>(lo), x, splat_f<W>(lo));
    return select_v<W>(a <= splat_f<W>(hi), a, splat_f<W>(hi));
}

template <int W>
inline vf32<W> pos_inf_v() {
    return bit_cast_v<vf32<W>>(splat_u<W>(0x7F800000u));
}

template <int W>
inline vf32<W> neg_inf_v() {
    return bit_cast_v<vf32<W>>(splat_u<W>(0xFF800000u));
}

template <int W>
inline vf32<W> qnan_v() {
    return bit_cast_v<vf32<W>>(splat_u<W>(0x7FC00000u));
}

template <int W>
inline vf32<W> exp2_v(vf32<W> x) {
    vf32<W> xc = clamp_finite_v<W>(x, -150.0f, 128.0f);
    vf32<W> kf = __builtin_elementwise_rint(xc);
    vi32<W> k = __builtin_convertvector(kf, vi32<W>);
    vf32<W> f = xc - kf;

    // Degree-6 minimax polynomial for 2^f on [-0.5, 0.5]
    vf32<W> p = splat_f<W>(1.53909599e-4f);
    p = p * f + splat_f<W>(1.33991045e-3f);
    p = p * f + splat_f<W>(9.61834379e-3f);
    p = p * f + splat_f<W>(5.55041246e-2f);
    p = p * f + splat_f<W>(2.40226507e-1f);
    p = p * f + splat_f<W>(6.93147182e-1f);
    p = splat_f<W>(1.0f) + f * p;

    // Split k = k1 + k2 so both 2^k1 and 2^k2 are normal floats across [-150, 128].
    vi32<W> k1 = k >> 1;
    vi32<W> k2 = k - k1;
    vf32<W> s1 = bit_cast_v<vf32<W>>((k1 + splat_i<W>(127)) << 23);
    vf32<W> s2 = bit_cast_v<vf32<W>>((k2 + splat_i<W>(127)) << 23);
    vf32<W> res = (p * s1) * s2;

    res = select_v<W>(x >= splat_f<W>(128.0f), pos_inf_v<W>(), res);
    res = select_v<W>(x < splat_f<W>(-150.0f), splat_f<W>(0.0f), res);
    res = select_v<W>(x != x, qnan_v<W>(), res);
    return res;
}

template <int W>
inline vf32<W> log2_v(vf32<W> x) {
    // Normalize positive subnormals when DAZ is disabled; clamp invalid lanes to 1.0f.
    vi32<W> valid_pos = (x > splat_f<W>(0.0f)) & (x < pos_inf_v<W>());
    vf32<W> sx = select_v<W>(valid_pos, x, splat_f<W>(1.0f));

    vi32<W> is_subnorm = sx < bit_cast_v<vf32<W>>(splat_u<W>(0x00800000u));
    sx = select_v<W>(is_subnorm, sx * splat_f<W>(16777216.0f), sx);
    vi32<W> bias = select_i<W>(is_subnorm, splat_i<W>(151), splat_i<W>(127));

    vi32<W> ix = bit_cast_v<vi32<W>>(sx);
    vi32<W> e = (ix >> 23) - bias;
    vf32<W> m = bit_cast_v<vf32<W>>((ix & splat_i<W>(0x007FFFFF)) | splat_i<W>(0x3F800000));

    // Normalize m to [sqrt(0.5), sqrt(2)]
    vi32<W> gt_sqrt2 = (m > splat_f<W>(1.4142135624f));
    m = select_v<W>(gt_sqrt2, m * splat_f<W>(0.5f), m);
    e = select_i<W>(gt_sqrt2, e + splat_i<W>(1), e);

    vf32<W> u = (m - splat_f<W>(1.0f)) / (m + splat_f<W>(1.0f));
    vf32<W> u2 = u * u;

    // Extended-precision 2/ln(2) = c1_hi + c1_lo plus minimax odd polynomial on |u| <= 0.1716
    constexpr float kTwoOverLn2Hi = 2.885390043258667f;
    constexpr float kTwoOverLn2Lo = 3.85192598e-8f;
    vf32<W> p = splat_f<W>(0.32185382f);
    p = p * u2 + splat_f<W>(0.41219705f);
    p = p * u2 + splat_f<W>(0.57707802f);
    p = p * u2 + splat_f<W>(0.96179669f);

    vf32<W> tail = (u * u2 * p) + (u * splat_f<W>(kTwoOverLn2Lo));
    vf32<W> res = (__builtin_convertvector(e, vf32<W>) + u * splat_f<W>(kTwoOverLn2Hi)) + tail;

    res = select_v<W>(x == splat_f<W>(0.0f), neg_inf_v<W>(), res);
    res = select_v<W>(x < splat_f<W>(0.0f), qnan_v<W>(), res);
    // Handles both +Inf (+Inf + +Inf = +Inf) and NaN (NaN + NaN = qNaN)
    res = select_v<W>(!(x < pos_inf_v<W>()), x + x, res);
    return res;
}

template <int W>
inline vf32<W> sin_poly_half_pi_v(vf32<W> r) {
    vf32<W> r2 = r * r;
    vf32<W> p = splat_f<W>(1.60590438e-10f);
    p = p * r2 + splat_f<W>(-2.50521084e-8f);
    p = p * r2 + splat_f<W>(2.75573192e-6f);
    p = p * r2 + splat_f<W>(-1.98412698e-4f);
    p = p * r2 + splat_f<W>(8.33333333e-3f);
    p = p * r2 + splat_f<W>(-1.66666667e-1f);
    return r + r * (r2 * p);
}

template <int W>
inline vf32<W> sin_v(vf32<W> x) {
    constexpr float kOneOverPi = 0.3183098861837907f;
    constexpr float kPi_1 = 3.140625f;              // 0x40490000
    constexpr float kPi_2 = 9.675025939941406e-4f;  // 0x3a7d8000
    constexpr float kPi_3 = 1.5099067241607776e-7f; // 0x34222000
    constexpr float kPi_4 = 5.126688136514179e-12f; // remainder

    vf32<W> ax = abs_v<W>(x);
    vf32<W> sx = select_v<W>(ax < splat_f<W>(3.0e9f), x, splat_f<W>(0.0f));

    vf32<W> qf = __builtin_elementwise_rint(sx * splat_f<W>(kOneOverPi));
    vi32<W> q = __builtin_convertvector(qf, vi32<W>);

    vf32<W> r = sx - qf * splat_f<W>(kPi_1);
    r = r - qf * splat_f<W>(kPi_2);
    r = r - qf * splat_f<W>(kPi_3);
    r = r - qf * splat_f<W>(kPi_4);

    vf32<W> sin_val = sin_poly_half_pi_v<W>(r);
    vu32<W> sin_flip = bit_cast_v<vu32<W>>((q & splat_i<W>(1)) << 31);
    sin_val = bit_cast_v<vf32<W>>(bit_cast_v<vu32<W>>(sin_val) ^ sin_flip);

    vi32<W> not_finite = !(ax < pos_inf_v<W>());
    sin_val = select_v<W>(x == splat_f<W>(0.0f), x, sin_val);
    return select_v<W>(not_finite, qnan_v<W>(), sin_val);
}

template <int W>
inline vf32<W> cos_v(vf32<W> x) {
    constexpr float kOneOverPi = 0.3183098861837907f;
    constexpr float kPi_1 = 3.140625f;              // 0x40490000
    constexpr float kPi_2 = 9.675025939941406e-4f;  // 0x3a7d8000
    constexpr float kPi_3 = 1.5099067241607776e-7f; // 0x34222000
    constexpr float kPi_4 = 5.126688136514179e-12f; // remainder

    vf32<W> ax = abs_v<W>(x);
    vf32<W> sx = select_v<W>(ax < splat_f<W>(1.0e6f), ax, splat_f<W>(0.0f));

    vf32<W> qf = __builtin_elementwise_rint(sx * splat_f<W>(kOneOverPi) - splat_f<W>(0.5f));
    vi32<W> q = __builtin_convertvector(qf, vi32<W>);
    vf32<W> qcf = qf + splat_f<W>(0.5f);

    vf32<W> r = sx - qcf * splat_f<W>(kPi_1);
    r = r - qcf * splat_f<W>(kPi_2);
    r = r - qcf * splat_f<W>(kPi_3);
    r = r - qcf * splat_f<W>(kPi_4);

    vf32<W> cos_val = sin_poly_half_pi_v<W>(r);
    vu32<W> cos_flip = bit_cast_v<vu32<W>>(((q + splat_i<W>(1)) & splat_i<W>(1)) << 31);
    cos_val = bit_cast_v<vf32<W>>(bit_cast_v<vu32<W>>(cos_val) ^ cos_flip);

    vi32<W> not_finite = !(ax < pos_inf_v<W>());
    return select_v<W>(not_finite, qnan_v<W>(), cos_val);
}

template <int W>
inline void sincos_v(vf32<W> x, vf32<W>& out_sin, vf32<W>& out_cos) {
    // 4-part Cody-Waite pi/2 constants with 12 trailing zero mantissa bits on c1..c3
    constexpr float kTwoOverPi = 0.63661977236758134f;
    constexpr float kPiOver2_1 = 1.5703125f;              // 0x3fc90000
    constexpr float kPiOver2_2 = 4.837512969970703e-4f;   // 0x39fda000
    constexpr float kPiOver2_3 = 7.549533620803888e-8f;   // 0x33a22000
    constexpr float kPiOver2_4 = 2.5633440682570896e-12f; // remainder

    vf32<W> ax = abs_v<W>(x);
    vf32<W> sx = select_v<W>(ax < splat_f<W>(3.0e9f), x, splat_f<W>(0.0f));

    vf32<W> qf = __builtin_elementwise_rint(sx * splat_f<W>(kTwoOverPi));
    vi32<W> q = __builtin_convertvector(qf, vi32<W>);

    vf32<W> r = sx - qf * splat_f<W>(kPiOver2_1);
    r = r - qf * splat_f<W>(kPiOver2_2);
    r = r - qf * splat_f<W>(kPiOver2_3);
    r = r - qf * splat_f<W>(kPiOver2_4);

    vf32<W> r2 = r * r;
    vf32<W> s = splat_f<W>(2.75573143e-6f);
    s = s * r2 + splat_f<W>(-1.98412698e-4f);
    s = s * r2 + splat_f<W>(8.33333333e-3f);
    s = s * r2 + splat_f<W>(-1.66666667e-1f);
    s = r + r * (r2 * s);

    vf32<W> c = splat_f<W>(-2.75573192e-7f);
    c = c * r2 + splat_f<W>(2.48015873e-5f);
    c = c * r2 + splat_f<W>(-1.38888889e-3f);
    c = c * r2 + splat_f<W>(4.16666667e-2f);
    c = c * r2 + splat_f<W>(-0.5f);
    c = splat_f<W>(1.0f) + r2 * c;

    vi32<W> swap = (q & splat_i<W>(1)) != splat_i<W>(0);
    vf32<W> sin_val = select_v<W>(swap, c, s);
    vf32<W> cos_val = select_v<W>(swap, s, c);

    vu32<W> sin_flip = bit_cast_v<vu32<W>>((q & splat_i<W>(2)) << 30);
    vu32<W> cos_flip = bit_cast_v<vu32<W>>(((q + splat_i<W>(1)) & splat_i<W>(2)) << 30);

    sin_val = bit_cast_v<vf32<W>>(bit_cast_v<vu32<W>>(sin_val) ^ sin_flip);
    cos_val = bit_cast_v<vf32<W>>(bit_cast_v<vu32<W>>(cos_val) ^ cos_flip);

    vi32<W> not_finite = !(ax < pos_inf_v<W>());
    sin_val = select_v<W>(x == splat_f<W>(0.0f), x, sin_val);
    out_sin = select_v<W>(not_finite, qnan_v<W>(), sin_val);
    out_cos = select_v<W>(not_finite, qnan_v<W>(), cos_val);
}

template <int W>
inline vf32<W> atan_pos_v(vf32<W> x) {
    // x >= 0
    vi32<W> gt1 = (x > splat_f<W>(1.0f));
    vf32<W> z = select_v<W>(gt1, splat_f<W>(1.0f) / x, x);

    // Reduce z in [0, 1] via z > tan(pi/8)
    vi32<W> gt_tan = (z > splat_f<W>(0.4142135624f));
    vf32<W> z_red = (z - splat_f<W>(1.0f)) / (z + splat_f<W>(1.0f));
    z = select_v<W>(gt_tan, z_red, z);

    vf32<W> z2 = z * z;
    vf32<W> p = splat_f<W>(0.05265332f);
    p = p * z2 + splat_f<W>(-0.08897698f);
    p = p * z2 + splat_f<W>(0.11093241f);
    p = p * z2 + splat_f<W>(-0.14283981f);
    p = p * z2 + splat_f<W>(0.19999912f);
    p = p * z2 + splat_f<W>(-0.33333331f);
    vf32<W> res = z + z * (z2 * p);

    constexpr float kPiOver4Hi = 0.7853981852531433f;
    constexpr float kPiOver4Lo = -2.1855694143368963e-8f;
    constexpr float kPiOver2Hi = 1.5707963705062866f;
    constexpr float kPiOver2Lo = -4.371138828673793e-8f;

    res = select_v<W>(gt_tan, splat_f<W>(kPiOver4Hi) + (res + splat_f<W>(kPiOver4Lo)), res);
    res = select_v<W>(gt1, splat_f<W>(kPiOver2Hi) - (res - splat_f<W>(kPiOver2Lo)), res);
    return res;
}

template <int W>
inline vf32<W> atan_v(vf32<W> x) {
    vf32<W> r = atan_pos_v<W>(abs_v<W>(x));
    r = copysign_mag_v<W>(r, x);
    return select_v<W>(x != x, qnan_v<W>(), r);
}

template <int W>
inline void asin_core_v(vf32<W> ax_clamped, vi32<W>& out_hi, vf32<W>& out_poly) {
    vi32<W> hi = (ax_clamped > splat_f<W>(0.5f));
    vf32<W> z = select_v<W>(hi, splat_f<W>(0.5f) * (splat_f<W>(1.0f) - ax_clamped), ax_clamped * ax_clamped);
    vf32<W> s = __builtin_elementwise_sqrt(z);

    vf32<W> p = splat_f<W>(0.042163199f);
    p = p * z + splat_f<W>(0.024181311f);
    p = p * z + splat_f<W>(0.045470026f);
    p = p * z + splat_f<W>(0.074953003f);
    p = p * z + splat_f<W>(0.166667524f);

    vf32<W> u = select_v<W>(hi, s, ax_clamped);
    out_hi = hi;
    out_poly = u + u * (z * p);
}

template <int W>
inline vf32<W> asin_v(vf32<W> x) {
    vf32<W> ax = abs_v<W>(x);
    vi32<W> out_of_domain = !(ax <= splat_f<W>(1.0f));
    vf32<W> ax_safe = select_v<W>(out_of_domain, splat_f<W>(0.0f), ax);

    vi32<W> hi;
    vf32<W> poly;
    asin_core_v<W>(ax_safe, hi, poly);

    constexpr float kPiOver2Hi = 1.5707963705062866f;
    constexpr float kPiOver2Lo = -4.371138828673793e-8f;
    vf32<W> hi_res = splat_f<W>(kPiOver2Hi) - (splat_f<W>(2.0f) * poly - splat_f<W>(kPiOver2Lo));
    vf32<W> res = select_v<W>(hi, hi_res, poly);
    res = copysign_mag_v<W>(res, x);
    return select_v<W>(out_of_domain, qnan_v<W>(), res);
}

template <int W>
inline vf32<W> acos_v(vf32<W> x) {
    vf32<W> ax = abs_v<W>(x);
    vi32<W> out_of_domain = !(ax <= splat_f<W>(1.0f));
    vf32<W> ax_safe = select_v<W>(out_of_domain, splat_f<W>(0.0f), ax);

    vi32<W> hi;
    vf32<W> poly;
    asin_core_v<W>(ax_safe, hi, poly);
    (void)hi;

    constexpr float kPiOver2Hi = 1.5707963705062866f;
    constexpr float kPiOver2Lo = -4.371138828673793e-8f;
    constexpr float kPiHi = 3.1415927410125732f;
    constexpr float kPiLo = -8.742277657347586e-8f;

    vf32<W> spoly = copysign_mag_v<W>(poly, x);
    vf32<W> mid_res = splat_f<W>(kPiOver2Hi) - (spoly - splat_f<W>(kPiOver2Lo));
    vf32<W> pos_hi_res = splat_f<W>(2.0f) * poly;
    vf32<W> neg_hi_res = splat_f<W>(kPiHi) - (splat_f<W>(2.0f) * poly - splat_f<W>(kPiLo));

    vf32<W> res = select_v<W>(x > splat_f<W>(0.5f), pos_hi_res, mid_res);
    res = select_v<W>(x < splat_f<W>(-0.5f), neg_hi_res, res);
    return select_v<W>(out_of_domain, qnan_v<W>(), res);
}

// Computes 2^(k_offset) * exp(ac) for finite ac in [0, 90].
template <int W>
inline vf32<W> exp_scaled_pos_v(vf32<W> ac, int32_t k_offset) {
    constexpr float kLog2E = 1.4426950408889634f;
    constexpr float kLn2Hi = 0.693145751953125f;     // 0x3f317200 (8 trailing zero mantissa bits)
    constexpr float kLn2Lo = 1.428606765330187e-6f;

    vf32<W> kf = __builtin_elementwise_rint(ac * splat_f<W>(kLog2E));
    vi32<W> k = __builtin_convertvector(kf, vi32<W>) + splat_i<W>(k_offset);
    vf32<W> r = (ac - kf * splat_f<W>(kLn2Hi)) - kf * splat_f<W>(kLn2Lo);

    vf32<W> p = splat_f<W>(1.9841270e-4f);
    p = p * r + splat_f<W>(1.3888889e-3f);
    p = p * r + splat_f<W>(8.3333338e-3f);
    p = p * r + splat_f<W>(4.1666668e-2f);
    p = p * r + splat_f<W>(1.6666667e-1f);
    p = p * r + splat_f<W>(5.0e-1f);
    vf32<W> er = splat_f<W>(1.0f) + (r + (r * r) * p);

    vi32<W> k1 = k >> 1;
    vi32<W> k2 = k - k1;
    vf32<W> s1 = bit_cast_v<vf32<W>>((k1 + splat_i<W>(127)) << 23);
    vf32<W> s2 = bit_cast_v<vf32<W>>((k2 + splat_i<W>(127)) << 23);
    return (er * s1) * s2;
}

template <int W>
inline vf32<W> sinh_v(vf32<W> x) {
    vf32<W> ax = abs_v<W>(x);
    vf32<W> ac = clamp_finite_v<W>(ax, 0.0f, 90.0f);

    // Small |x| < 1: cancellation-free odd polynomial
    vf32<W> x2 = ac * ac;
    vf32<W> p = splat_f<W>(2.7557319e-6f);
    p = p * x2 + splat_f<W>(1.9841270e-4f);
    p = p * x2 + splat_f<W>(8.3333338e-3f);
    p = p * x2 + splat_f<W>(1.6666667e-1f);
    vf32<W> small_r = ax + ax * (x2 * p);

    // Medium 1 <= |x| < 20: 0.5 * (e^|x| - e^-|x|)
    vf32<W> w = exp_scaled_pos_v<W>( clamp_finite_v<W>(ax, 0.0f, 20.0f), 0 );
    vf32<W> mid_r = splat_f<W>(0.5f) * (w - splat_f<W>(1.0f) / w);

    // Large 20 <= |x| <= 89.4159851: 0.5 * e^|x| = 2^-1 * e^|x|
    vf32<W> hi_r = exp_scaled_pos_v<W>(ac, -1);

    vf32<W> r = select_v<W>(ax < splat_f<W>(1.0f), small_r, mid_r);
    r = select_v<W>(ax >= splat_f<W>(20.0f), hi_r, r);
    r = select_v<W>(ax > splat_f<W>(89.4159851f), pos_inf_v<W>(), r);
    r = copysign_mag_v<W>(r, x);
    return select_v<W>(x != x, qnan_v<W>(), r);
}

template <int W>
inline vf32<W> cosh_v(vf32<W> x) {
    vf32<W> ax = abs_v<W>(x);
    vf32<W> ac = clamp_finite_v<W>(ax, 0.0f, 90.0f);

    vf32<W> w = exp_scaled_pos_v<W>(clamp_finite_v<W>(ax, 0.0f, 20.0f), 0);
    vf32<W> lo_r = splat_f<W>(0.5f) * (w + splat_f<W>(1.0f) / w);
    vf32<W> hi_r = exp_scaled_pos_v<W>(ac, -1);

    vf32<W> r = select_v<W>(ax >= splat_f<W>(20.0f), hi_r, lo_r);
    r = select_v<W>(ax > splat_f<W>(89.4159851f), pos_inf_v<W>(), r);
    return select_v<W>(x != x, qnan_v<W>(), r);
}

template <int W>
inline vf32<W> tanh_v(vf32<W> x) {
    vf32<W> ax = abs_v<W>(x);

    // Small |x| < 0.25: cancellation-free Taylor polynomial
    vf32<W> as = min_v<W>(ax, splat_f<W>(0.25f));
    vf32<W> x2 = as * as;
    vf32<W> p = splat_f<W>(2.18694885e-2f);
    p = p * x2 + splat_f<W>(-5.39682540e-2f);
    p = p * x2 + splat_f<W>(1.33333333e-1f);
    p = p * x2 + splat_f<W>(-3.33333333e-1f);
    vf32<W> small_r = ax + ax * (x2 * p);

    vf32<W> e2x = exp_scaled_pos_v<W>(clamp_finite_v<W>(splat_f<W>(2.0f) * ax, 0.0f, 20.0f), 0);
    vf32<W> mid_r = (e2x - splat_f<W>(1.0f)) / (e2x + splat_f<W>(1.0f));

    vf32<W> r = select_v<W>(ax < splat_f<W>(0.25f), small_r, mid_r);
    r = select_v<W>(ax >= splat_f<W>(9.5f), splat_f<W>(1.0f), r);
    r = copysign_mag_v<W>(r, x);
    return select_v<W>(x != x, qnan_v<W>(), r);
}

} // namespace

#define NACELLE_DEFINE_MATH_V(W)                                                      \
    extern "C" {                                                                      \
    void nacelle_math_v##W##_exp2f(vf32<W>* out, const vf32<W>* in) {                 \
        *out = exp2_v<W>(*in);                                                        \
    }                                                                                 \
    void nacelle_math_v##W##_log2f(vf32<W>* out, const vf32<W>* in) {                 \
        *out = log2_v<W>(*in);                                                        \
    }                                                                                 \
    void nacelle_math_v##W##_sinf(vf32<W>* out, const vf32<W>* in) {                  \
        *out = sin_v<W>(*in);                                                         \
    }                                                                                 \
    void nacelle_math_v##W##_cosf(vf32<W>* out, const vf32<W>* in) {                  \
        *out = cos_v<W>(*in);                                                         \
    }                                                                                 \
    void nacelle_math_v##W##_tanf(vf32<W>* out, const vf32<W>* in) {                  \
        vf32<W> s, c;                                                                 \
        sincos_v<W>(*in, s, c);                                                       \
        *out = s / c;                                                                 \
    }                                                                                 \
    void nacelle_math_v##W##_asinf(vf32<W>* out, const vf32<W>* in) {                 \
        *out = asin_v<W>(*in);                                                        \
    }                                                                                 \
    void nacelle_math_v##W##_acosf(vf32<W>* out, const vf32<W>* in) {                 \
        *out = acos_v<W>(*in);                                                        \
    }                                                                                 \
    void nacelle_math_v##W##_atanf(vf32<W>* out, const vf32<W>* in) {                 \
        *out = atan_v<W>(*in);                                                        \
    }                                                                                 \
    void nacelle_math_v##W##_sinhf(vf32<W>* out, const vf32<W>* in) {                 \
        *out = sinh_v<W>(*in);                                                        \
    }                                                                                 \
    void nacelle_math_v##W##_coshf(vf32<W>* out, const vf32<W>* in) {                 \
        *out = cosh_v<W>(*in);                                                        \
    }                                                                                 \
    void nacelle_math_v##W##_tanhf(vf32<W>* out, const vf32<W>* in) {                 \
        *out = tanh_v<W>(*in);                                                        \
    }                                                                                 \
    }

NACELLE_DEFINE_MATH_V(4)
NACELLE_DEFINE_MATH_V(8)
NACELLE_DEFINE_MATH_V(16)
