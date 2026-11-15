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
inline vf32<W> abs_v(vf32<W> x) {
    return bit_cast_v<vf32<W>>(bit_cast_v<vu32<W>>(x) & (vu32<W>)(0x7FFFFFFFu));
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

template <int W>
inline vf32<W> max_v(vf32<W> a, vf32<W> b) {
    return (a > b) ? a : b;
}

template <int W>
inline vf32<W> clamp_v(vf32<W> x, float lo, float hi) {
    return min_v<W>(max_v<W>(x, splat_f<W>(lo)), splat_f<W>(hi));
}

template <int W>
inline vf32<W> exp2_v(vf32<W> x) {
    x = clamp_v<W>(x, -126.0f, 127.0f);
    vf32<W> kf = __builtin_elementwise_floor(x);
    vi32<W> k = __builtin_convertvector(kf, vi32<W>);
    vf32<W> f = x - kf;

    // Degree-5 minimax polynomial for 2^f on [0, 1]
    vf32<W> p = splat_f<W>(0.0013333558f);
    p = p * f + splat_f<W>(0.0096181291f);
    p = p * f + splat_f<W>(0.0555041087f);
    p = p * f + splat_f<W>(0.2402265070f);
    p = p * f + splat_f<W>(0.6931471806f);
    p = p * f + splat_f<W>(1.0f);

    vi32<W> exp_bits = (k + splat_i<W>(127)) << 23;
    return p * bit_cast_v<vf32<W>>(exp_bits);
}

template <int W>
inline vf32<W> log2_v(vf32<W> x) {
    // Clamp tiny/non-positive inputs to avoid UB in bit manipulation
    vf32<W> safe_x = max_v<W>(x, splat_f<W>(1.17549435e-38f));
    vi32<W> ix = bit_cast_v<vi32<W>>(safe_x);
    vi32<W> e = (ix >> 23) - splat_i<W>(127);
    vf32<W> m = bit_cast_v<vf32<W>>((ix & splat_i<W>(0x007FFFFF)) | splat_i<W>(0x3F800000));

    // Normalize m to [sqrt(0.5), sqrt(2)]
    vi32<W> gt_sqrt2 = (m > splat_f<W>(1.4142135624f));
    m = select_v<W>(gt_sqrt2, m * splat_f<W>(0.5f), m);
    e = select_i<W>(gt_sqrt2, e + splat_i<W>(1), e);

    vf32<W> u = (m - splat_f<W>(1.0f)) / (m + splat_f<W>(1.0f));
    vf32<W> u2 = u * u;

    // 2 / ln(2) * u * (1 + u2/3 + u2^2/5 + u2^3/7 + u2^4/9)
    vf32<W> p = splat_f<W>(1.0f / 9.0f);
    p = p * u2 + splat_f<W>(1.0f / 7.0f);
    p = p * u2 + splat_f<W>(1.0f / 5.0f);
    p = p * u2 + splat_f<W>(1.0f / 3.0f);
    p = p * u2 + splat_f<W>(1.0f);

    vf32<W> log2_m = splat_f<W>(2.8853900818f) * u * p;
    return __builtin_convertvector(e, vf32<W>) + log2_m;
}

template <int W>
inline void sincos_v(vf32<W> x, vf32<W>& out_sin, vf32<W>& out_cos) {
    // Cody-Waite range reduction by pi/2
    constexpr float kTwoOverPi = 0.63661977236758134f;
    constexpr float kPiOver2_1 = 1.5707962513f;
    constexpr float kPiOver2_2 = 7.5497894159e-8f;
    constexpr float kPiOver2_3 = 5.3903025299e-15f;

    vf32<W> qf = __builtin_elementwise_rint(x * splat_f<W>(kTwoOverPi));
    vi32<W> q = __builtin_convertvector(qf, vi32<W>);

    vf32<W> r = x - qf * splat_f<W>(kPiOver2_1);
    r = r - qf * splat_f<W>(kPiOver2_2);
    r = r - qf * splat_f<W>(kPiOver2_3);

    vf32<W> r2 = r * r;
    // sin(r) ~ r * (1 + r2 * (-1/6 + r2 * (1/120 - r2/5040)))
    vf32<W> s = splat_f<W>(-1.98412698e-4f);
    s = s * r2 + splat_f<W>(8.33333333e-3f);
    s = s * r2 + splat_f<W>(-1.66666667e-1f);
    s = r + r * r2 * s;

    // cos(r) ~ 1 + r2 * (-1/2 + r2 * (1/24 + r2 * (-1/720 + r2/40320)))
    vf32<W> c = splat_f<W>(2.48015873e-5f);
    c = c * r2 + splat_f<W>(-1.38888889e-3f);
    c = c * r2 + splat_f<W>(4.16666667e-2f);
    c = c * r2 + splat_f<W>(-0.5f);
    c = splat_f<W>(1.0f) + r2 * c;

    vi32<W> swap = (q & splat_i<W>(1)) != splat_i<W>(0);
    vf32<W> sin_val = select_v<W>(swap, c, s);
    vf32<W> cos_val = select_v<W>(swap, s, c);

    vi32<W> sin_neg = (q & splat_i<W>(2)) != splat_i<W>(0);
    vi32<W> cos_neg = ((q + splat_i<W>(1)) & splat_i<W>(2)) != splat_i<W>(0);

    out_sin = select_v<W>(sin_neg, -sin_val, sin_val);
    out_cos = select_v<W>(cos_neg, -cos_val, cos_val);
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
    vf32<W> p = splat_f<W>(-0.07528964f);
    p = p * z2 + splat_f<W>(0.10656264f);
    p = p * z2 + splat_f<W>(-0.14208899f);
    p = p * z2 + splat_f<W>(0.19993551f);
    p = p * z2 + splat_f<W>(-0.33333145f);
    vf32<W> res = z + z * z2 * p;

    res = select_v<W>(gt_tan, res + splat_f<W>(0.7853981634f), res);
    res = select_v<W>(gt1, splat_f<W>(1.5707963268f) - res, res);
    return res;
}

template <int W>
inline vf32<W> atan_v(vf32<W> x) {
    vf32<W> r = atan_pos_v<W>(abs_v<W>(x));
    vi32<W> neg = (x < splat_f<W>(0.0f));
    return select_v<W>(neg, -r, r);
}

template <int W>
inline vf32<W> asin_v(vf32<W> x) {
    vf32<W> ax = min_v<W>(abs_v<W>(x), splat_f<W>(1.0f));
    vi32<W> hi = (ax > splat_f<W>(0.5f));
    vf32<W> z = select_v<W>(hi, splat_f<W>(0.5f) * (splat_f<W>(1.0f) - ax), ax * ax);
    vf32<W> s = __builtin_elementwise_sqrt(z);

    // asin(u) = u + u * z * P(z) for u in [0, 0.5]
    vf32<W> p = splat_f<W>(0.042163199f);
    p = p * z + splat_f<W>(0.024181311f);
    p = p * z + splat_f<W>(0.045470026f);
    p = p * z + splat_f<W>(0.074953003f);
    p = p * z + splat_f<W>(0.166667524f);

    vf32<W> u = select_v<W>(hi, s, ax);
    vf32<W> poly = u + u * z * p;
    vf32<W> res = select_v<W>(hi, splat_f<W>(1.5707963268f) - splat_f<W>(2.0f) * poly, poly);

    vi32<W> neg = (x < splat_f<W>(0.0f));
    return select_v<W>(neg, -res, res);
}

template <int W>
inline vf32<W> acos_v(vf32<W> x) {
    return splat_f<W>(1.5707963268f) - asin_v<W>(x);
}

template <int W>
inline vf32<W> sinh_v(vf32<W> x) {
    constexpr float kLog2E = 1.4426950409f;
    vf32<W> t = x * splat_f<W>(kLog2E);
    return splat_f<W>(0.5f) * (exp2_v<W>(t) - exp2_v<W>(-t));
}

template <int W>
inline vf32<W> cosh_v(vf32<W> x) {
    constexpr float kLog2E = 1.4426950409f;
    vf32<W> t = x * splat_f<W>(kLog2E);
    return splat_f<W>(0.5f) * (exp2_v<W>(t) + exp2_v<W>(-t));
}

template <int W>
inline vf32<W> tanh_v(vf32<W> x) {
    constexpr float kTwoLog2E = 2.8853900818f;
    vf32<W> ax = abs_v<W>(x);
    vf32<W> e2x = exp2_v<W>(ax * splat_f<W>(kTwoLog2E));
    vf32<W> r = (e2x - splat_f<W>(1.0f)) / (e2x + splat_f<W>(1.0f));
    r = select_v<W>(ax > splat_f<W>(10.0f), splat_f<W>(1.0f), r);
    vi32<W> neg = (x < splat_f<W>(0.0f));
    return select_v<W>(neg, -r, r);
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
        vf32<W> s, c;                                                                 \
        sincos_v<W>(*in, s, c);                                                       \
        *out = s;                                                                     \
    }                                                                                 \
    void nacelle_math_v##W##_cosf(vf32<W>* out, const vf32<W>* in) {                  \
        vf32<W> s, c;                                                                 \
        sincos_v<W>(*in, s, c);                                                       \
        *out = c;                                                                     \
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
