#include "cpu_warm.hpp"
#include "basalt/models/qwen4exp/decode/verify.hpp"
#include "basalt/models/qwen4exp/decode/native_head.hpp"
#include "basalt/models/qwen4exp/decode/gdn_head_split.hpp"
#include "basalt/platform/compiler.hpp"
#include "basalt/kernels/pdl.hpp"
#include "basalt/core/progress.hpp"
#include "basalt/models/qwen4exp/kernels/verify_kernels.hpp"
#include "basalt/models/qwen4exp/kernels/dec_moe.hpp"
#include "basalt/log.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <vector>
#include <immintrin.h>

namespace basalt::core::verify_detail {

// layer `c.layer`'s router rows [e0, e1) on the job's tokens
BASALT_TARGET_AVX2_FMA_F16C void warm_logits(CpuWarm& c, int e0, int e1, const std::atomic<bool>& stop) {
    const int NB = c.n_embd / 32, pt = c.n_tok;
    const uint8_t* R = c.rq.data() + (size_t) c.layer * c.n_expert * (c.n_embd / 2);
    const float* S = c.rs.data() + (size_t) c.layer * c.n_expert * NB;
    const __m128i lo4 = _mm_set1_epi8(0x0f), off = _mm_set1_epi8(8);
    for (int e = e0; e < e1; ++e) {
        if (stop.load(std::memory_order_relaxed)) return;
        __m256 acc[basalt::kernels::kVerifyMaxT];
        for (int t = 0; t < pt; ++t) acc[t] = _mm256_setzero_ps();
        const uint8_t* row = R + (size_t) e * (c.n_embd / 2);
        for (int b = 0; b < NB; ++b) {
            const __m128i q = _mm_loadu_si128((const __m128i*) (row + b * 16));
            const __m128i ql = _mm_sub_epi8(_mm_and_si128(q, lo4), off);
            const __m128i qh = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(q, 4), lo4), off);
            const __m256i wl = _mm256_cvtepi8_epi16(ql);
            const __m256i wh = _mm256_cvtepi8_epi16(qh);
            const float sw = S[(size_t) e * NB + b];
            for (int t = 0; t < pt; ++t) {
                const uint8_t* blk = c.xq + ((size_t) t * NB + b) * 36;
                uint16_t dh;
                std::memcpy(&dh, blk, 2);
                const __m256i x = _mm256_loadu_si256((const __m256i*) (blk + 4));
                const __m256i p = _mm256_add_epi32(_mm256_madd_epi16(wl, _mm256_cvtepi8_epi16(_mm256_castsi256_si128(x))),
                                                   _mm256_madd_epi16(wh, _mm256_cvtepi8_epi16(_mm256_extracti128_si256(x, 1))));
                acc[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(sw * _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(dh)))), acc[t]);
            }
        }
        for (int t = 0; t < pt; ++t) {
            __m128 v = _mm_add_ps(_mm256_castps256_ps128(acc[t]), _mm256_extractf128_ps(acc[t], 1));
            v = _mm_add_ps(v, _mm_movehl_ps(v, v));
            v = _mm_add_ss(v, _mm_shuffle_ps(v, v, 1));
            c.logits[(size_t) t * c.n_expert + e] = _mm_cvtss_f32(v);
        }
    }
}

void warm_select(CpuWarm& c) {
    const int NE = c.n_expert, K = std::min(c.k, 16);
    const int want = std::clamp(c.max_warm, 0, CpuWarm::kMaxBlobs);
    c.n_blobs = 0;
    if (want <= 0) return;
    float score[512] = {};
    for (int t = 0; t < c.n_tok; ++t) {
        const float* lg = c.logits.data() + (size_t) t * NE;
        int top[16];
        float val[16];
        int m = 0;
        for (int e = 0; e < NE; ++e) {
            const float v = lg[e];
            if (m == K && v <= val[K - 1]) continue;
            int j = m < K ? m++ : K - 1;
            while (j > 0 && val[j - 1] < v) { val[j] = val[j - 1]; top[j] = top[j - 1]; --j; }
            val[j] = v;
            top[j] = e;
        }
        float w[16], sum = 0.0f;
        for (int j = 0; j < m; ++j) sum += (w[j] = std::exp(val[j] - val[0]));
        for (int j = 0; j < m; ++j) {
            const int e = top[j];
            if (c.res0[e] >= 0 || (c.res1 != nullptr && c.res1[e] >= 0)) continue;
            score[e] += w[j] / sum;
        }
    }
    int best_id[CpuWarm::kMaxBlobs];
    float best_sc[CpuWarm::kMaxBlobs];
    int nb = 0;
    for (int e = 0; e < NE; ++e) {
        const float sc = score[e];
        if (sc <= 0.0f) continue;
        if (nb == want && sc <= best_sc[want - 1]) continue;
        int j = nb < want ? nb++ : want - 1;
        while (j > 0 && best_sc[j - 1] < sc) { best_sc[j] = best_sc[j - 1]; best_id[j] = best_id[j - 1]; --j; }
        best_sc[j] = sc;
        best_id[j] = e;
    }
    int out = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t* b = c.src->resident_blob(c.layer, best_id[i]);
        if (b != nullptr) {
            c.blobs[out] = b;
            c.blob_ids[out] = best_id[i];
            ++out;
        }
    }
    c.n_blobs = out;
    if (c.stats_on) {
        c.stat_layer = c.layer;
        c.stat_n = out;
        for (int i = 0; i < out; ++i) c.stat_ids[i] = c.blob_ids[i];
    }
}

void warm_record_actual(CpuWarm& c, int layer, const int32_t* ids, const int32_t* kind, int n_entries) {
    if (!c.stats_on || c.stat_layer != layer) return;
    int seen[128];
    int ns = 0;
    for (int i = 0; i < n_entries && ns < 128; ++i) {
        if (kind[i] >= 0 || ids[i] < 0) continue;
        bool dup = false;
        for (int j = 0; j < ns; ++j) if (seen[j] == ids[i]) { dup = true; break; }
        if (!dup) seen[ns++] = ids[i];
    }
    ++c.stat_layers;
    if (ns > 0) ++c.stat_layers_with_miss;
    c.stat_miss_experts += (uint64_t) ns;
    for (int j = 0; j < ns; ++j) {
        if (c.stat_n > 0 && seen[j] == c.stat_ids[0]) ++c.stat_top1_hits;
        for (int k = 0; k < c.stat_n; ++k) {
            if (seen[j] == c.stat_ids[k]) { ++c.stat_topk_hits; break; }
        }
    }
    if ((c.stat_layers % 9600ull) == 0) {
        const double miss = (double) std::max<uint64_t>(1, c.stat_miss_experts);
        const double touched_mb = (double) c.stat_bytes_touched.load(std::memory_order_relaxed) / 1e6;
        basalt::log::info("basalt warm", "layers=%llu (with_miss=%llu) actual_cpu_experts=%llu top1_hit=%.1f%% top%d_hit=%.1f%% avg_warmed=%.2f MB/layer",
                          (unsigned long long) c.stat_layers, (unsigned long long) c.stat_layers_with_miss,
                          (unsigned long long) c.stat_miss_experts,
                          100.0 * (double) c.stat_top1_hits / miss, c.max_warm,
                          100.0 * (double) c.stat_topk_hits / miss, touched_mb / (double) c.stat_layers);
    }
}

void warm_job(void* ctx, int w, int W, const std::atomic<bool>& stop) {
    CpuWarm& c = *(CpuWarm*) ctx;
    if (c.max_warm <= 0 || W <= 0) return;
    warm_logits(c, c.n_expert * w / W, c.n_expert * (w + 1) / W, stop);
    if (c.arrive.fetch_add(1, std::memory_order_acq_rel) + 1 == W) {
        if (!stop.load(std::memory_order_relaxed)) warm_select(c);
        c.ready.store(1, std::memory_order_release);
    } else {
        while (c.ready.load(std::memory_order_acquire) == 0) {
            if (stop.load(std::memory_order_relaxed)) return;
            _mm_pause();
        }
    }
    const int nb = c.n_blobs;
    if (nb <= 0) return;

    int ccd_w = W;
    int rank = w;
    int ccd = 0;
    const bool dual = c.pool != nullptr && c.pool->dual_ccd() && c.gu_row > 0 && c.d_row > 0 && c.n_ff >= 32;
    if (dual) {
        rank = c.pool->ccd_worker_rank(w, W, ccd_w);
        ccd = c.pool->worker_ccd(w);
        if (ccd_w <= 0) return;
        if (c.warm_w_per_ccd > 0 && c.warm_w_per_ccd < ccd_w) {
            if (rank >= c.warm_w_per_ccd) return;
            ccd_w = c.warm_w_per_ccd;
        }
    } else if (c.warm_w_per_ccd > 0 && 2 * c.warm_w_per_ccd < ccd_w) {
        if (rank >= 2 * c.warm_w_per_ccd) return;
        ccd_w = 2 * c.warm_w_per_ccd;
    }

    const int gu_mid = basalt::kernels::cpu::ExpertPool::ccd_split_row(c.n_ff);
    const int d_mid = basalt::kernels::cpu::ExpertPool::ccd_split_row(c.n_embd);
    const size_t g0 = dual ? (ccd == 0 ? 0 : (size_t) gu_mid * c.gu_row) : 0;
    const size_t g1 = dual ? (ccd == 0 ? (size_t) gu_mid * c.gu_row : (size_t) c.n_ff * c.gu_row) : c.up_off;
    const size_t u0 = c.up_off + g0;
    const size_t u1 = c.up_off + g1;
    const size_t d0 = c.down_off + (dual ? (ccd == 0 ? 0 : (size_t) d_mid * c.d_row) : 0);
    const size_t d1 = dual ? (ccd == 0 ? (c.down_off + (size_t) d_mid * c.d_row) : c.bytes) : c.bytes;

    uint64_t x = 0;
    uint64_t touched = 0;
    auto touch_span = [&](const uint8_t* blob, size_t s0, size_t s1) -> bool {
        if (s1 <= s0) return true;
        const size_t per = ((s1 - s0 + (size_t) ccd_w - 1) / (size_t) ccd_w + 63) & ~(size_t) 63;
        const size_t b0 = std::min(s1, s0 + per * (size_t) rank), b1 = std::min(s1, b0 + per);
        for (size_t o = b0; o < b1; o += 512) {
            if (stop.load(std::memory_order_relaxed)) return false;
            const uint8_t* p = blob + o;
            const size_t n = std::min<size_t>(512, b1 - o);
            for (size_t q = 0; q + 8 <= n; q += 64) x += *(const volatile uint64_t*) (p + q);
            touched += n;
        }
        return true;
    };

    for (int bi = 0; bi < nb; ++bi) {
        const uint8_t* blob = c.blobs[bi];
        if (blob == nullptr) continue;
        if (!touch_span(blob, g0, g1)) break;
        if (c.up_off > 0 && !touch_span(blob, u0, u1)) break;
        if (!touch_span(blob, d0, d1)) break;
    }
    if (c.stats_on && touched > 0) c.stat_bytes_touched.fetch_add(touched, std::memory_order_relaxed);
    if (x == 0x9e3779b97f4a7c15ull) c.sink.fetch_add(x, std::memory_order_relaxed);
}

bool router_codes(const WeightTable& wt, const ModelGeometry& g, std::vector<int8_t>* q8, std::vector<float>* s8,
                  std::vector<uint8_t>* q4, std::vector<float>* s4) {
    const int64_t N = g.n_embd, NE = g.n_expert, L = g.n_layers, NB = N / 32;
    if (NE > basalt::kernels::kDmPredExperts || N % 32 != 0) return false;
    std::vector<uint16_t> src((size_t) (NE * N));
    if (q8 != nullptr) {
        q8->assign((size_t) (L * NE * N), 0);
        s8->assign((size_t) (L * NE * NB), 0.0f);
    }
    if (q4 != nullptr) {
        q4->assign((size_t) (L * NE * N / 2), 0);
        s4->assign((size_t) (L * NE * NB), 0.0f);
    }
    for (int64_t l = 0; l < L; ++l) {
        const WeightRef* w = LayerView(wt, l).get("ffn_gate_inp.weight");
        if (w == nullptr || w->bytes != (uint64_t) (NE * N * 2)) return false;
        if (cudaMemcpy(src.data(), w->data, (size_t) w->bytes, cudaMemcpyDefault) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        for (int64_t r = 0; r < NE; ++r)
            for (int64_t b = 0; b < NB; ++b) {
                float v[32], amax = 0.0f;
                for (int i = 0; i < 32; ++i) {
                    const uint32_t u = (uint32_t) src[(size_t) (r * N + b * 32 + i)] << 16;
                    std::memcpy(&v[i], &u, 4);
                    amax = std::max(amax, std::fabs(v[i]));
                }
                const size_t blk = (size_t) ((l * NE + r) * NB + b);
                if (q8 != nullptr) {
                    const float d = amax / 127.0f;
                    (*s8)[blk] = d;
                    for (int i = 0; i < 32; ++i)
                        (*q8)[blk * 32 + (size_t) i] = (int8_t) (d > 0.0f ? std::lrintf(v[i] / d) : 0);
                }
                if (q4 != nullptr) {
                    const float d = amax / 7.0f;
                    (*s4)[blk] = d;
                    uint8_t* out = q4->data() + blk * 16;
                    for (int i = 0; i < 16; ++i) {
                        const long c0 = d > 0.0f ? std::lrintf(v[i] / d) : 0, c1 = d > 0.0f ? std::lrintf(v[i + 16] / d) : 0;
                        out[i] = (uint8_t) ((std::clamp<long>(c0, -7, 7) + 8) | ((std::clamp<long>(c1, -7, 7) + 8) << 4));
                    }
                }
            }
    }
    return true;
}

bool warm_router(const std::vector<int8_t>& q, const std::vector<float>& sc, int8_t** rq, float** rs) {
    if (cudaMalloc((void**) rq, q.size()) != cudaSuccess || cudaMalloc((void**) rs, sc.size() * 4) != cudaSuccess ||
        cudaMemcpy(*rq, q.data(), q.size(), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(*rs, sc.data(), sc.size() * 4, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaGetLastError();
        if (*rq) cudaFree(*rq);
        if (*rs) cudaFree(*rs);
        *rq = nullptr;
        *rs = nullptr;
        return false;
    }
    return true;
}

}  // namespace basalt::core::verify_detail
