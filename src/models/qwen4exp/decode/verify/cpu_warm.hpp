#pragma once

#include "common.hpp"
#include "basalt/models/qwen4exp/decode/verify.hpp"
#include "basalt/models/qwen4exp/decode/native_head.hpp"
#include "basalt/kernels/cpu/expert.hpp"
#include "basalt/models/qwen4exp/geometry.hpp"
#include "basalt/kernels/cpu/pool.hpp"
#include <atomic>
#include <vector>

namespace basalt::core {

// The CPU tier's cache warm-up.  After layer l's doorbell the parked pool workers score layer l + 1's router on layer
// l's FFN input (the ring's q8_1 rows; the router in 4-bit with one scale per 32 values, a prediction only), take
// every token's top k, rank the predicted experts only the CPU holds by their summed weight, and read up to max_warm
// experts into their own L3 domains (aligned to gate/up/down half-rows on dual-CCD CPUs).
struct CpuWarm {
    static constexpr int kMaxBlobs = 4;
    std::vector<uint8_t> rq;  ///< [n_layers][n_expert][n_embd / 2]
    std::vector<float> rs;    ///< [n_layers][n_expert][n_embd / 32]
    int n_expert = 0, n_embd = 0, n_ff = 0, k = 0;
    size_t bytes = 0, up_off = 0, down_off = 0, gu_row = 0, d_row = 0;
    ExpertSource* src = nullptr;
    const basalt::kernels::cpu::ExpertPool* pool = nullptr;
    int max_warm = 2;
    int warm_w_per_ccd = 3;   ///< 0 = all active workers in each L3 domain; >0 caps memory-touching workers per CCD
    bool stats_on = false;
    // the job in flight
    int layer = 0, n_tok = 0;
    const uint8_t* xq = nullptr;                      ///< the ring's q8_1 rows, tokens back to back
    const int32_t* res0 = nullptr, *res1 = nullptr;   ///< the layer's CUDA0 / peer residency (res1 may be null)
    std::vector<float> logits;                        ///< [kVerifyMaxT][n_expert]
    alignas(64) std::atomic<int> arrive{0};
    alignas(64) std::atomic<int> ready{0};
    const uint8_t* blobs[kMaxBlobs] = {};
    int blob_ids[kMaxBlobs] = {-1, -1, -1, -1};
    int n_blobs = 0;
    std::atomic<uint64_t> sink{0};
    // optional diagnostic counters (enabled when BASALT_WARM_STATS=1)
    int stat_layer = -1;
    int stat_n = 0;
    int stat_ids[kMaxBlobs] = {-1, -1, -1, -1};
    uint64_t stat_layers = 0, stat_miss_experts = 0, stat_top1_hits = 0, stat_topk_hits = 0, stat_layers_with_miss = 0;
    std::atomic<uint64_t> stat_bytes_touched{0};
};

namespace verify_detail {
void warm_job(void* ctx, int w, int W, const std::atomic<bool>& stop);
void warm_record_actual(CpuWarm& c, int layer, const int32_t* ids, const int32_t* kind, int n_entries);
bool router_codes(const WeightTable& wt, const ModelGeometry& g, std::vector<int8_t>* q8, std::vector<float>* s8,
                  std::vector<uint8_t>* q4, std::vector<float>* s4);
bool warm_router(const std::vector<int8_t>& q, const std::vector<float>& sc, int8_t** rq, float** rs);
}

}  // namespace basalt::core
