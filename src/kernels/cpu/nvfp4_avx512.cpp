// src/kernels/cpu/nvfp4_avx512.cpp - NVFP4 expert rows against Q8_0 activations, several tokens at once.
//
// The ggml dot (ggml_vec_dot_nvfp4_q8_0, AVX-512 branch) is single-token: every token re-decodes the codes and
// rebuilds the scales.  Here a block's codes, sign mask and UE4M3 scales are decoded once and every token applies
// them with a masked subtract and one VNNI dot.  Per token and row the arithmetic is the ggml dot's: the same
// int32 sums per 4 bytes (dpbusd equals maddubs + madd at these magnitudes), the same per-lane fp32 products
// d_y * scale, the same fused accumulation across blocks and the same final reduction, so results are bitwise equal.
#include "basalt/kernels/cpu/nvfp4_avx512.hpp"
#include "basalt/kernels/cpu/swiglu.hpp"
#include "basalt/platform/compiler.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "basalt/ggml/ggml_common.h"

#include <immintrin.h>

#include <cmath>
#include <bit>
#include <cstring>

namespace basalt::kernels::cpu {
namespace {

// UE4M3 scale as ggml's dot decodes it (nvfp4_scale in ggml_cpu_dots.c), one entry per 7-bit code.  Built at compile
// time: this file is compiled for AVX-512, and a static constructor here would run before the CPU is checked.
struct ScaleLut {
    float v[128];
    constexpr ScaleLut() : v{} {
        for (uint32_t x = 0; x < 128; ++x) {
            const uint32_t e = x >> 3, m = x & 7;
            if (x == 0 || x == 0x7F) v[x] = 0.0f;
            else if (e == 0) v[x] = (float) m * (1.0f / 1024.0f);
            else v[x] = std::bit_cast<float>((uint32_t) (((e + 119) << 23) | (m << 20)));
        }
    }
};
constexpr ScaleLut kLut{};

// Largest row the per-token packed buffers cover (values), and the tokens one pass takes.
constexpr int kMaxN = 16384;
constexpr int kMaxT = 8;

// Packed per-token Q8_0 scales (fp32, 64B aligned) and contiguous 64-byte Q8_0 quants, one workspace per worker.
struct ActWorkspace {
    alignas(64) float dv[kMaxT * kMaxN / 4];
    alignas(64) int8_t aq[kMaxT * kMaxN];
};

ActWorkspace& act_workspace() {
    static thread_local ActWorkspace ws;
    return ws;
}

inline float h2f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h))); }

// One decoded weight block: |w| bytes, the sign mask, the 16-lane fp32 scale (sub-block s on lanes 4s..4s+3).
struct Blk {
    __m512i aw;
    __mmask64 m;
    __m512 ws;
};

inline Blk decode(const block_nvfp4* x, const __m512i values_abs, const __m512i m4b, const __m512i m8,
                  const __m512i dup_idx, const __m512i lane) {
    const __m512i q = _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*) x->qs));
    const __m512i dup = _mm512_permutexvar_epi64(dup_idx, q);
    const __m512i shifted = _mm512_mask_srli_epi16(dup, 0xF0F0F0F0u, dup, 4);
    const __m512i codes = _mm512_and_si512(shifted, m4b);
    Blk b;
    b.aw = _mm512_shuffle_epi8(values_abs, codes);
    b.m = _mm512_test_epi8_mask(codes, m8);
    const __m128 s = _mm_setr_ps(kLut.v[x->d[0] & 0x7F], kLut.v[x->d[1] & 0x7F], kLut.v[x->d[2] & 0x7F],
                                 kLut.v[x->d[3] & 0x7F]);
    b.ws = _mm512_permutexvar_ps(lane, _mm512_castps128_ps512(s));
    return b;
}

// Pack each token's Q8_0 scales and 64-byte quant blocks once into 64B-aligned L1D workspace.
// Layout: block-major [ib][t] so inner-loop loads for all NT tokens at block ib are sequential in L1D.
template <int NT>
inline void pack_tokens(const block_q8_0* const* y, int nb, float* __restrict__ dv, int8_t* __restrict__ aq) {
    for (int ib = 0; ib < nb; ++ib) {
        for (int t = 0; t < NT; ++t) {
            const block_q8_0& b0 = y[t][2 * ib];
            const block_q8_0& b1 = y[t][2 * ib + 1];
            const __m256 d0 = _mm256_set1_ps(h2f(b0.d));
            const __m256 d1 = _mm256_set1_ps(h2f(b1.d));
            const size_t idx = (size_t) ib * NT + (size_t) t;
            _mm512_store_ps(dv + 16 * idx, _mm512_insertf32x8(_mm512_castps256_ps512(d0), d1, 1));
            const __m512i a = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*) b0.qs)),
                                                 _mm256_loadu_si256((const __m256i*) b1.qs), 1);
            _mm512_store_si512((__m512i*) (aq + 64 * idx), a);
        }
    }
}

// R rows (each its own block pointer) against NT tokens: res[k][t].
template <int R, int NT>
inline void rows_dot(const block_nvfp4* const* w, int nb, const float* __restrict__ dv, const int8_t* __restrict__ aq,
                     float (*res)[kMaxT]) {
    __m512 acc[R][NT];
    for (int k = 0; k < R; ++k)
        for (int t = 0; t < NT; ++t) acc[k][t] = _mm512_setzero_ps();
    const __m512i zero = _mm512_setzero_si512();
    const __m512i values_abs = _mm512_abs_epi8(_mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*) kvalues_fp4)));
    const __m512i m4b = _mm512_set1_epi8(0x0f);
    const __m512i m8 = _mm512_set1_epi8(0x08);
    const __m512i dup_idx = _mm512_setr_epi64(0, 0, 1, 1, 2, 2, 3, 3);
    const __m512i lane = _mm512_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3);
    for (int ib = 0; ib < nb; ++ib) {
        Blk b[R];
        for (int k = 0; k < R; ++k) {
            if constexpr (platform::kPreferMemberwiseVectorAssignment) {
                const Blk decoded = decode(w[k] + ib, values_abs, m4b, m8, dup_idx, lane);
                b[k].aw = decoded.aw;
                b[k].m = decoded.m;
                b[k].ws = decoded.ws;
            } else {
                b[k] = decode(w[k] + ib, values_abs, m4b, m8, dup_idx, lane);
            }
        }
        const float* dv_ib = dv + 16 * ((size_t) ib * NT);
        const int8_t* aq_ib = aq + 64 * ((size_t) ib * NT);
        for (int t = 0; t < NT; ++t) {
            const __m512i a = _mm512_load_si512((const __m512i*) (aq_ib + 64 * (size_t) t));
            const __m512 d = _mm512_load_ps(dv_ib + 16 * (size_t) t);
            for (int k = 0; k < R; ++k) {
                const __m512i sa = _mm512_mask_sub_epi8(a, b[k].m, zero, a);
                const __m512i p = _mm512_dpbusd_epi32(zero, b[k].aw, sa);
                acc[k][t] = _mm512_fmadd_ps(_mm512_mul_ps(d, b[k].ws), _mm512_cvtepi32_ps(p), acc[k][t]);
            }
        }
    }
    for (int k = 0; k < R; ++k)
        for (int t = 0; t < NT; ++t) res[k][t] = _mm512_reduce_add_ps(acc[k][t]);
}

template <int NT>
void gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, float sg, float su, const SwiGlu& glu,
             const void* const* act, float* const* ff, int r0, int r1) {
    const block_q8_0* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_0*) act[t];
    const int nb = n / QK_NVFP4;
    ActWorkspace& ws = act_workspace();
    pack_tokens<NT>(y, nb, ws.dv, ws.aq);
    int r = r0;
    if constexpr (NT <= 4) {
        float res4[4][kMaxT];
        for (; r + 2 <= r1; r += 2) {
            const block_nvfp4* w[4] = {
                (const block_nvfp4*) (blob + (size_t) r * gu_row),
                (const block_nvfp4*) (blob + up_off + (size_t) r * gu_row),
                (const block_nvfp4*) (blob + (size_t) (r + 1) * gu_row),
                (const block_nvfp4*) (blob + up_off + (size_t) (r + 1) * gu_row),
            };
            rows_dot<4, NT>(w, nb, ws.dv, ws.aq, res4);
            for (int t = 0; t < NT; ++t) {
                ff[t][r] = swiglu(glu, res4[0][t] * sg, res4[1][t] * su);
                ff[t][r + 1] = swiglu(glu, res4[2][t] * sg, res4[3][t] * su);
            }
        }
    }
    float res[2][kMaxT];
    for (; r < r1; ++r) {
        const block_nvfp4* w[2] = {(const block_nvfp4*) (blob + (size_t) r * gu_row),
                                   (const block_nvfp4*) (blob + up_off + (size_t) r * gu_row)};
        rows_dot<2, NT>(w, nb, ws.dv, ws.aq, res);
        for (int t = 0; t < NT; ++t) {
            ff[t][r] = swiglu(glu, res[0][t] * sg, res[1][t] * su);
        }
    }
}

template <int NT>
void down_rows(const uint8_t* w0, size_t row_bytes, int n, float sd, const void* const* act, float* const* out, int r0,
               int r1) {
    const block_q8_0* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_0*) act[t];
    const int nb = n / QK_NVFP4;
    ActWorkspace& ws = act_workspace();
    pack_tokens<NT>(y, nb, ws.dv, ws.aq);
    int r = r0;
    if constexpr (NT <= 4) {
        float res4[4][kMaxT];
        for (; r + 4 <= r1; r += 4) {
            const block_nvfp4* w[4] = {
                (const block_nvfp4*) (w0 + (size_t) r * row_bytes),
                (const block_nvfp4*) (w0 + (size_t) (r + 1) * row_bytes),
                (const block_nvfp4*) (w0 + (size_t) (r + 2) * row_bytes),
                (const block_nvfp4*) (w0 + (size_t) (r + 3) * row_bytes),
            };
            rows_dot<4, NT>(w, nb, ws.dv, ws.aq, res4);
            for (int t = 0; t < NT; ++t) {
                out[t][r] = res4[0][t] * sd;
                out[t][r + 1] = res4[1][t] * sd;
                out[t][r + 2] = res4[2][t] * sd;
                out[t][r + 3] = res4[3][t] * sd;
            }
        }
    }
    float res[2][kMaxT];
    for (; r + 2 <= r1; r += 2) {
        const block_nvfp4* w[2] = {(const block_nvfp4*) (w0 + (size_t) r * row_bytes),
                                   (const block_nvfp4*) (w0 + (size_t) (r + 1) * row_bytes)};
        rows_dot<2, NT>(w, nb, ws.dv, ws.aq, res);
        for (int t = 0; t < NT; ++t) {
            out[t][r] = res[0][t] * sd;
            out[t][r + 1] = res[1][t] * sd;
        }
    }
    if (r < r1) {
        const block_nvfp4* w[1] = {(const block_nvfp4*) (w0 + (size_t) r * row_bytes)};
        rows_dot<1, NT>(w, nb, ws.dv, ws.aq, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[0][t] * sd;
    }
}

}  // namespace

bool nvfp4_512_fits(int n_embd, int n_ff) noexcept {
    return n_embd % QK_NVFP4 == 0 && n_ff % QK_NVFP4 == 0 && n_embd <= kMaxN && n_ff <= kMaxN;
}

void nvfp4_512_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, float sg, float su, const SwiGlu& glu,
                       const void* const* act, int nt, float* const* ff, int r0, int r1) {
    for (int t0 = 0; t0 < nt; t0 += kMaxT) {
        const int k = nt - t0 < kMaxT ? nt - t0 : kMaxT;
        const void* const* a = act + t0;
        float* const* o = ff + t0;
        switch (k) {
            case 1: gu_rows<1>(blob, gu_row, up_off, n, sg, su, glu, a, o, r0, r1); break;
            case 2: gu_rows<2>(blob, gu_row, up_off, n, sg, su, glu, a, o, r0, r1); break;
            case 3: gu_rows<3>(blob, gu_row, up_off, n, sg, su, glu, a, o, r0, r1); break;
            case 4: gu_rows<4>(blob, gu_row, up_off, n, sg, su, glu, a, o, r0, r1); break;
            case 5: gu_rows<5>(blob, gu_row, up_off, n, sg, su, glu, a, o, r0, r1); break;
            case 6: gu_rows<6>(blob, gu_row, up_off, n, sg, su, glu, a, o, r0, r1); break;
            case 7: gu_rows<7>(blob, gu_row, up_off, n, sg, su, glu, a, o, r0, r1); break;
            default: gu_rows<8>(blob, gu_row, up_off, n, sg, su, glu, a, o, r0, r1); break;
        }
    }
}

void nvfp4_512_down_rows(const uint8_t* w, size_t row_bytes, int n, float sd, const void* const* act, int nt,
                         float* const* out, int r0, int r1) {
    for (int t0 = 0; t0 < nt; t0 += kMaxT) {
        const int k = nt - t0 < kMaxT ? nt - t0 : kMaxT;
        const void* const* a = act + t0;
        float* const* o = out + t0;
        switch (k) {
            case 1: down_rows<1>(w, row_bytes, n, sd, a, o, r0, r1); break;
            case 2: down_rows<2>(w, row_bytes, n, sd, a, o, r0, r1); break;
            case 3: down_rows<3>(w, row_bytes, n, sd, a, o, r0, r1); break;
            case 4: down_rows<4>(w, row_bytes, n, sd, a, o, r0, r1); break;
            case 5: down_rows<5>(w, row_bytes, n, sd, a, o, r0, r1); break;
            case 6: down_rows<6>(w, row_bytes, n, sd, a, o, r0, r1); break;
            case 7: down_rows<7>(w, row_bytes, n, sd, a, o, r0, r1); break;
            default: down_rows<8>(w, row_bytes, n, sd, a, o, r0, r1); break;
        }
    }
}

}  // namespace basalt::kernels::cpu
