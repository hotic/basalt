// src/kernels/cpu/pool.cpp - the CPU expert pool.  Read pool.hpp first; it explains the protocol.
#include "basalt/kernels/cpu/pool.hpp"
#include "basalt/core/progress.hpp"
#include "basalt/kernels/cpu/cpu_quota.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <immintrin.h>

#include <cstdio>
#include <cstdlib>
#include "basalt/log.hpp"

namespace basalt::kernels::cpu {

namespace {
constexpr uint64_t pack_head(uint32_t epoch, uint32_t n, uint32_t i) {
    return ((uint64_t) epoch << 32) | ((uint64_t) n << 16) | (uint64_t) i;
}

bool env_disabled(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] == '1' && v[1] == '\0';
}
}  // namespace

std::vector<int> physical_cores(bool skip_first, PoolAffinity affinity) {
    return detect_cpu_topology(skip_first, affinity).worker_cores;
}

int auto_pool_workers(int worker_cores, int p_cores, int e_cores, bool hybrid, double cpu_quota, PoolAffinity affinity) {
    int n = worker_cores;
    if (hybrid && affinity != PoolAffinity::All) {
        n = std::max(1, p_cores - 1);
        if (affinity == PoolAffinity::Auto) n += e_cores / 2;
    }
    if (cpu_quota > 0) {
        const int fit = std::max(1, (int) cpu_quota - 1);
        if (fit < n) n = fit;
    }
    return std::max(1, n);
}

int start_pool_workers(int given, bool affinity_given, PoolAffinity affinity) {
    if (given > 0) return given;
    const CpuTopology topo = detect_cpu_topology(true);
    const PoolAffinity pick = !affinity_given || affinity == PoolAffinity::Auto ? PoolAffinity::Auto
                              : affinity == PoolAffinity::PCores               ? PoolAffinity::PCores
                                                                               : PoolAffinity::All;
    return auto_pool_workers((int) topo.worker_cores.size(), topo.p_cores, topo.e_cores, topo.is_hybrid, cgroup_cpu_quota(), pick);
}

namespace {
class StallClock {
public:
    bool expired(uint32_t progress) {
        if ((++spins_ & 1023u) != 0) return false;
        const auto now = std::chrono::steady_clock::now();
        if (!armed_ || progress != seen_) {
            armed_ = true;
            seen_ = progress;
            t0_ = now;
            return false;
        }
        return now - t0_ > ExpertPool::kStall;
    }

private:
    uint32_t spins_ = 0, seen_ = 0;
    bool armed_ = false;
    std::chrono::steady_clock::time_point t0_{};
};

std::atomic<const ExpertPool*> g_diag_pool{nullptr};
void diag_active_pool(std::FILE* f) {
    if (const ExpertPool* p = g_diag_pool.load()) p->diag(f);
}
int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
}  // namespace

void ExpertPool::diag(std::FILE* f) const {
    const uint64_t h = head_.load();
    std::fprintf(f, "  expert pool: epoch %u, batch epoch %u: %u of %u jobs claimed, %u done; %u of %d workers parked, "
                    "%u sleeping; mode %d, dual_ccd %d\n", epoch_.load(), (uint32_t) (h >> 32), (uint32_t) h & 0xffffu,
                 (uint32_t) (h >> 16) & 0xffffu, done_.load(), parked_.load(), n_, sleepers_.load(), mode_,
                 use_dual_ ? 1 : 0);
    std::fprintf(f, "  expert pool active workers: %d\n", active_.load());
    std::fprintf(f, "  expert pool threads:");
    for (int i = 0; i < n_; ++i) {
        const int32_t s = wstate_[(size_t) i].load();
        if (s == kParked) std::fprintf(f, " w%d=parked", i);
        else if (s == kSleeping) std::fprintf(f, " w%d=sleeping", i);
        else if (s == kBetween) std::fprintf(f, " w%d=draining", i);
        else std::fprintf(f, " w%d=job%d", i, s);
    }
    const int32_t hs = hstate_.load();
    const char* hn = hs == kIdle ? "idle" : hs == kWaitParked ? "waiting for the workers to park"
                   : hs == kWaitDone ? "waiting for the jobs to finish" : "running a job";
    std::fprintf(f, "; host %s", hn);
    if (hs >= 0) std::fprintf(f, " %d", hs);
    std::fprintf(f, " for %lld ms\n", (long long) (now_ms() - hstate_ms_.load()));
}

int ExpertPool::worker_ccd(int id) const {
    if (!dual_ccd_) return 0;
    if (id < 0) return host_ccd_;
    return id < (int) worker_ccd_.size() ? worker_ccd_[(size_t) id] : 0;
}

int ExpertPool::ccd_worker_rank(int id, int active, int& ccd_workers) const {
    if (!dual_ccd_) {
        ccd_workers = active;
        return id;
    }
    const int my = worker_ccd(id);
    int rank = 0, total = 0;
    const int lim = std::min(active, (int) worker_ccd_.size());
    for (int i = 0; i < lim; ++i) {
        if (worker_ccd_[(size_t) i] != my) continue;
        if (i == id) rank = total;
        ++total;
    }
    ccd_workers = total;
    return rank;
}

void ExpertPool::pin_host_once() {
    if (!pin_threads_ || topo_.host_core < 0) return;
    if (!host_pinned_.load(std::memory_order_relaxed) && !host_pinned_.exchange(true, std::memory_order_acq_rel)) {
        if (!env_disabled("BASALT_NO_HOST_PIN")) (void) platform::pin_current_thread(topo_.host_core);
    }
}

ExpertPool::ExpertPool(int n_workers, PoolAffinity affinity)
    : affinity_(affinity), topo_(detect_cpu_topology(true, affinity)) {
    n_ = n_workers;
    if (n_ < 1) n_ = 1;
    active_.store(n_);
    wstate_.reset(new std::atomic<int32_t>[(size_t) n_]);
    for (int i = 0; i < n_; ++i) wstate_[(size_t) i].store(kParked);
    hstate_ms_.store(now_ms());
    g_diag_pool.store(this);
    basalt::core::diag_pool_fn().store(&diag_active_pool);
    split_multi_.resize((size_t) kMaxSplitMulti);
    fused_.reset(new FusedExpert[(size_t) kMaxSplitMulti]);

    // Detect L3 cache domains (e.g. dual-CCD Ryzen X3D) and assign stable physical cores.
    worker_cpu_.assign((size_t) n_, -1);
    worker_ccd_.assign((size_t) n_, 0);
    const bool allow_pin = !topo_.is_hybrid && !env_disabled("BASALT_NO_PIN") &&
                           (int) topo_.worker_cores.size() >= n_ && topo_.host_core >= 0;
    if (allow_pin) {
        pin_threads_ = true;
        for (int i = 0; i < n_; ++i) worker_cpu_[(size_t) i] = topo_.worker_cores[(size_t) i];
        const auto l3 = platform::cpu_l3_groups();
        if (l3.size() == 2 && !env_disabled("BASALT_NO_CCD_PIN")) {
            auto group_of = [&](int cpu) -> int {
                for (size_t g = 0; g < l3.size(); ++g)
                    if (std::find(l3[g].begin(), l3[g].end(), cpu) != l3[g].end()) return (int) g;
                return -1;
            };
            const int hc = group_of(topo_.host_core);
            int cnt[2] = {0, 0};
            bool ok = hc >= 0;
            if (ok) ++cnt[hc];
            for (int i = 0; ok && i < n_; ++i) {
                const int gc = group_of(worker_cpu_[(size_t) i]);
                if (gc < 0) ok = false;
                else {
                    worker_ccd_[(size_t) i] = gc;
                    ++cnt[gc];
                }
            }
            if (ok && cnt[0] >= 4 && cnt[1] >= 4 && std::abs(cnt[0] - cnt[1]) <= 2) {
                dual_ccd_ = true;
                host_ccd_ = hc;
            }
        }
    }

    threads_.reserve((size_t) n_);
    for (int i = 0; i < n_; ++i) {
        threads_.emplace_back([this, i] { worker(i); });
    }
}

ExpertPool::~ExpertPool() {
    const ExpertPool* self = this;
    g_diag_pool.compare_exchange_strong(self, nullptr);
    idle_stop();
    stop_.store(true, std::memory_order_release);
    publish();
    for (auto& t : threads_) t.join();
}

void ExpertPool::publish() {
    epoch_.fetch_add(1, std::memory_order_seq_cst);
    if (sleepers_.load(std::memory_order_seq_cst) != 0) {
        std::lock_guard<std::mutex> lk(sleep_mu_);
        sleep_cv_.notify_all();
    }
}

void ExpertPool::worker(int id) {
    if (pin_threads_ && id >= 0 && id < (int) worker_cpu_.size() && worker_cpu_[(size_t) id] >= 0) {
        (void) platform::pin_current_thread(worker_cpu_[(size_t) id]);
    }
    uint32_t seen = 0;
    parked_.fetch_add(1, std::memory_order_acq_rel);
    for (;;) {
        const auto parked_at = std::chrono::steady_clock::now();
        auto yield_at = parked_at + kYieldEvery;
        uint32_t spins = 0;
        while (epoch_.load(std::memory_order_acquire) == seen) {
            if (stop_.load(std::memory_order_relaxed)) return;
            _mm_pause();
            if ((++spins & 63u) != 0) continue;
            const auto now = std::chrono::steady_clock::now();
            if (now - parked_at < spin_before_sleep_) {
                if (now >= yield_at) {
                    std::this_thread::yield();
                    yield_at = now + kYieldEvery;
                }
                continue;
            }
            std::unique_lock<std::mutex> lk(sleep_mu_);
            wstate_[(size_t) id].store(kSleeping, std::memory_order_relaxed);
            sleepers_.fetch_add(1, std::memory_order_seq_cst);
            sleep_cv_.wait(lk, [&] {
                return epoch_.load(std::memory_order_seq_cst) != seen || stop_.load(std::memory_order_relaxed);
            });
            sleepers_.fetch_sub(1, std::memory_order_relaxed);
            wstate_[(size_t) id].store(kParked, std::memory_order_relaxed);
        }
        if (stop_.load(std::memory_order_acquire)) return;
        seen = epoch_.load(std::memory_order_acquire);
        if (id >= active_.load(std::memory_order_acquire)) continue;
        parked_.fetch_sub(1, std::memory_order_acq_rel);
        wstate_[(size_t) id].store(kBetween, std::memory_order_relaxed);
        drain(id, seen);
        wstate_[(size_t) id].store(kParked, std::memory_order_relaxed);
        parked_.fetch_add(1, std::memory_order_acq_rel);
    }
}

int ExpertPool::claim_from(std::atomic<uint64_t>& head, uint32_t epoch, int limit) {
    uint64_t h = head.load(std::memory_order_acquire);
    for (;;) {
        if ((uint32_t) (h >> 32) != epoch) return -1;
        const uint32_t n = (uint32_t) (h >> 16) & 0xffffu, i = (uint32_t) h & 0xffffu;
        const uint32_t cap = limit >= 0 ? (std::min)(n, (uint32_t) limit) : n;
        if (i >= cap) return -1;
        if (head.compare_exchange_weak(h, h + 1, std::memory_order_acq_rel, std::memory_order_acquire))
            return (int) i;
    }
}

int ExpertPool::claim(uint32_t epoch) {
    return claim_from(head_, epoch);
}

uint32_t ExpertPool::begin_batch(int n) {
    if (n < 0 || n > 0xffff) {
        basalt::log::error("basalt", "expert pool batch of %d jobs is out of range", n);
        std::abort();
    }
    done_.store(0, std::memory_order_relaxed);
    const uint32_t e = epoch_.load(std::memory_order_relaxed) + 1;
    if (mode_ == 8 && use_dual_) {
        head_ccd_[0].store(pack_head(e, (uint32_t) dual_ccd_tasks_, 0), std::memory_order_release);
        head_ccd_[1].store(pack_head(e, (uint32_t) dual_ccd_tasks_, 0), std::memory_order_release);
        head_.store(pack_head(e, (uint32_t) n, (uint32_t) n), std::memory_order_release);
    } else {
        head_.store(pack_head(e, (uint32_t) n, 0), std::memory_order_release);
    }
    publish();
    return e;
}

void ExpertPool::stalled() {
    basalt::core::release_gpu_waits(basalt::log::stream());
    std::abort();
}

void ExpertPool::wait_parked(const char* what) {
    hstate_.store(kWaitParked, std::memory_order_relaxed);
    hstate_ms_.store(now_ms(), std::memory_order_relaxed);
    StallClock clock;
    while (parked_.load(std::memory_order_acquire) != (uint32_t) n_) {
        _mm_pause();
        if (clock.expired(parked_.load(std::memory_order_relaxed))) {
            basalt::log::error("basalt", "the CPU expert pool stalled %s (%u of %d workers parked) - stopping the engine "
                                 "so the server can start it again",
                         what, parked_.load(), n_);
            stalled();
        }
    }
}

void ExpertPool::wait_done(int n) {
    hstate_.store(kWaitDone, std::memory_order_relaxed);
    hstate_ms_.store(now_ms(), std::memory_order_relaxed);
    StallClock clock;
    for (;;) {
        const uint32_t d = done_.load(std::memory_order_acquire);
        if (d >= (uint32_t) n) return;
        _mm_pause();
        if (clock.expired(d)) {
            basalt::log::error("basalt", "the CPU expert pool stalled: %u of %d jobs done, %u of %d workers parked - "
                                 "stopping the engine so the server can start it again",
                         d, n, parked_.load(), n_);
            stalled();
        }
    }
}

void ExpertPool::wait_ready(int e) {
    StallClock clock;
    while (fused_[(size_t) e].ready.load(std::memory_order_acquire) == 0) {
        _mm_pause();
        if (clock.expired(done_.load(std::memory_order_relaxed))) {
            basalt::log::error("basalt", "the CPU expert pool stalled: the gate/up rows of expert %d of the batch did not finish "
                                 "(%u of %d tasks done, %u of %d workers parked) - stopping the engine so the server can "
                                 "start it again",
                         e, done_.load(), njobs_, parked_.load(), n_);
            stalled();
        }
    }
}

void ExpertPool::run_dual_task(int ccd, int ci, uint32_t epoch) {
    const int n = fused_n_, pg = fused_gu_parts_, pd = fused_d_parts_;
    if (ci < n * pg) {
        const int e = ci / pg, part = ci % pg;
        const int base = ccd == 0 ? 0 : dual_gu_mid_;
        const int len = ccd == 0 ? dual_gu_mid_ : ((int) nfmt_->n_ff - dual_gu_mid_);
        const int r0 = part_row_range(part, pg, base, len);
        const int r1 = part_row_range(part + 1, pg, base, len);
        if (r0 < r1) native_gu_range(e, r0, r1);
        FusedExpert& fe = fused_[(size_t) e];
        if (fe.gu_left.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            native_quantize(e);
            fe.ready.store(1, std::memory_order_release);
        }
        return;
    }
    const int k = ci - n * pg, e = k / pd, part = k % pd;
    const int base = ccd == 0 ? 0 : dual_d_mid_;
    const int len = ccd == 0 ? dual_d_mid_ : ((int) nfmt_->n_embd - dual_d_mid_);
    const int r0 = part_row_range(part, pd, base, len);
    const int r1 = part_row_range(part + 1, pd, base, len);
    if (fused_[(size_t) e].ready.load(std::memory_order_acquire) == 0) {
        // Help finish any unclaimed gate/up work on the peer CCD first (prevents any possibility of starvation).
        const int peer = 1 - ccd;
        while (fused_[(size_t) e].ready.load(std::memory_order_acquire) == 0) {
            const int steal = claim_from(head_ccd_[peer], epoch, n * pg);
            if (steal < 0) break;
            run_dual_task(peer, steal, epoch);
            done_.fetch_add(1, std::memory_order_release);
        }
        if (fused_[(size_t) e].ready.load(std::memory_order_acquire) == 0) {
            if (r0 < r1) {
                prefetch_until_ready(e, mjobs_[e].blob + nfmt_->down_off + (size_t) r0 * nfmt_->d_row,
                                     (size_t) (r1 - r0) * nfmt_->d_row);
            }
            if (fused_[(size_t) e].ready.load(std::memory_order_acquire) == 0) wait_ready(e);
        }
    }
    if (r0 < r1) native_down_range(e, r0, r1);
}

void ExpertPool::drain(int id, uint32_t epoch) {
    if (mode_ == 7) {
        // Stable worker-aligned idle work: each active worker claims at most one ticket so idle_stop() accounting
        // stays exact, and receives its pinned worker index `id` rather than the dynamic ticket number.
        const int ci = claim(epoch);
        if (ci < 0) return;
        if (id >= 0) wstate_[(size_t) id].store(ci, std::memory_order_relaxed);
        idle_fn_(idle_ctx_, id >= 0 ? id : ci, active_.load(std::memory_order_relaxed), idle_halt_);
        done_.fetch_add(1, std::memory_order_release);
        return;
    }
    if (mode_ == 8 && use_dual_) {
        const int my = worker_ccd(id);
        for (int pass = 0; pass < 2; ++pass) {
            const int ccd = pass == 0 ? my : (1 - my);
            for (;;) {
                const int ci = claim_from(head_ccd_[ccd], epoch);
                if (ci < 0) break;
                const int tag = ccd * dual_ccd_tasks_ + ci;
                if (id >= 0) wstate_[(size_t) id].store(tag, std::memory_order_relaxed);
                else { hstate_.store(tag, std::memory_order_relaxed); hstate_ms_.store(now_ms(), std::memory_order_relaxed); }
                run_dual_task(ccd, ci, epoch);
                done_.fetch_add(1, std::memory_order_release);
            }
        }
        return;
    }
    for (;;) {
        const int ci = claim(epoch);
        if (ci < 0) break;
        const uint32_t i = (uint32_t) ci;
        if (id >= 0) wstate_[(size_t) id].store(ci, std::memory_order_relaxed);
        else { hstate_.store(ci, std::memory_order_relaxed); hstate_ms_.store(now_ms(), std::memory_order_relaxed); }
        const int n = fused_n_, pg = fused_gu_parts_, pd = fused_d_parts_;
        if ((int) i < n * pg) {
            const int e = (int) i / pg, part = (int) i % pg;
            const int r0 = part_row_range(part, pg, 0, (int) nfmt_->n_ff);
            const int r1 = part_row_range(part + 1, pg, 0, (int) nfmt_->n_ff);
            if (r0 < r1) native_gu_range(e, r0, r1);
            FusedExpert& fe = fused_[(size_t) e];
            if (fe.gu_left.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                native_quantize(e);
                fe.ready.store(1, std::memory_order_release);
            }
        } else {
            const int k = (int) i - n * pg, e = k / pd, part = k % pd;
            const int r0 = part_row_range(part, pd, 0, (int) nfmt_->n_embd);
            const int r1 = part_row_range(part + 1, pd, 0, (int) nfmt_->n_embd);
            if (fused_[(size_t) e].ready.load(std::memory_order_acquire) == 0) {
                if (r0 < r1) {
                    prefetch_until_ready(e, mjobs_[e].blob + nfmt_->down_off + (size_t) r0 * nfmt_->d_row,
                                         (size_t) (r1 - r0) * nfmt_->d_row);
                }
                if (fused_[(size_t) e].ready.load(std::memory_order_acquire) == 0) wait_ready(e);
            }
            if (r0 < r1) native_down_range(e, r0, r1);
        }
        done_.fetch_add(1, std::memory_order_release);
    }
}

int ExpertPool::part_row(int part, int parts, int rows) const {
    const int small = (std::min)(fused_tail_, parts / 2), big = parts - small;
    const int64_t units = 4 * (int64_t) (std::min)(part, big) + (std::max)(0, part - big);
    return (int) ((int64_t) rows * units / (4 * (int64_t) big + small));
}

int ExpertPool::part_row_range(int part, int parts, int r_base, int r_len) const {
    if (part <= 0) return r_base;
    if (part >= parts) return r_base + r_len;
    // Align inner cuts to 16 floats (64 bytes = 1 cache line) when possible to prevent false sharing in SplitBufMulti::ff.
    if (r_len >= 32 && (r_len & 15) == 0 && parts <= r_len / 16) {
        return r_base + 16 * part_row(part, parts, r_len / 16);
    }
    return r_base + part_row(part, parts, r_len);
}

void ExpertPool::prefetch_until_ready(int e, const uint8_t* p, size_t bytes) const {
    for (size_t o = 0; o < bytes; o += 64) {
        _mm_prefetch((const char*) p + o, _MM_HINT_T1);
        if ((o & 4095) == 4032 && fused_[(size_t) e].ready.load(std::memory_order_relaxed) != 0) return;
    }
}

void ExpertPool::native_gu_range(int e, int r0, int r1) {
    SplitBufMulti& sb = split_multi_[(size_t) e];
    const ExpertJobMulti& j = mjobs_[e];
    if (q2_native_kernels(nfmt_->gu_type)) {
        thread_local float gbuf[MAXT][kNativeMaxFF], ubuf[MAXT][kNativeMaxFF];
        float* gp[MAXT];
        float* up[MAXT];
        for (int t = 0; t < j.nt; ++t) { gp[t] = gbuf[t]; up[t] = ubuf[t]; }
        const int nbk = (int) (nfmt_->n_embd / 64);
        q2_rows_any(j.blob, nfmt_->gu_row, nbk, j.act, j.nt, gp, r0, r1);
        q2_rows_any(j.blob + nfmt_->up_off, nfmt_->gu_row, nbk, j.act, j.nt, up, r0, r1);
        for (int t = 0; t < j.nt; ++t)
            for (int r = r0; r < r1; ++r)
                sb.ff[t][r] = (gbuf[t][r] / (1.f + std::exp(-gbuf[t][r]))) * ubuf[t][r];
        return;
    }
    float* ff[MAXT];
    for (int t = 0; t < j.nt; ++t) ff[t] = sb.ff[t];
    native_gu_rows(*nfmt_, j.blob, j.nact, j.nt, ff, r0, r1);
}

void ExpertPool::native_down_range(int e, int r0, int r1) {
    SplitBufMulti& sb = split_multi_[(size_t) e];
    const ExpertJobMulti& j = mjobs_[e];
    if (q2_native_kernels(nfmt_->d_type)) {
        const ActQ* a2[MAXT];
        for (int t = 0; t < j.nt; ++t) a2[t] = &sb.a2[t];
        q2_rows_any(j.blob + nfmt_->down_off, nfmt_->d_row, (int) (nfmt_->n_ff / 64), a2, j.nt, j.out, r0, r1);
        return;
    }
    const void* hq[MAXT];
    for (int t = 0; t < j.nt; ++t) hq[t] = sb.hq[t];
    native_down_rows(*nfmt_, j.blob, hq, j.nt, j.out, r0, r1);
}

void ExpertPool::native_quantize(int e) {
    SplitBufMulti& sb = split_multi_[(size_t) e];
    for (int t = 0; t < mjobs_[e].nt; ++t)
        if (q2_native_kernels(nfmt_->d_type)) act_quant_any(sb.ff[t], (int) nfmt_->n_ff, sb.a2[t]);
        else native_quant_h(*nfmt_, sb.ff[t], sb.hq[t]);
}

int ExpertPool::set_active_workers(int k) {
    k = (std::min)((std::max)(k, 1), n_);
    idle_stop();
    wait_parked("before changing the active workers");
    active_.store(k, std::memory_order_release);
    return k;
}

void ExpertPool::idle_start(IdleFn fn, void* ctx) {
    if (n_ == 1 || fn == nullptr) return;
    idle_stop();
    idle_fn_ = fn;
    idle_ctx_ = ctx;
    idle_halt_.store(false, std::memory_order_relaxed);
    mode_ = 7;
    const int act = active_.load(std::memory_order_relaxed);
    njobs_ = act;
    begin_batch(act);
    idle_live_ = true;
}

void ExpertPool::idle_signal_stop() {
    if (!idle_live_) return;
    idle_halt_.store(true, std::memory_order_release);
}

void ExpertPool::idle_stop() {
    if (!idle_live_) return;
    idle_halt_.store(true, std::memory_order_release);
    uint64_t h = head_.load(std::memory_order_acquire);
    uint32_t claimed = 0;
    for (;;) {
        const uint32_t n = (uint32_t) (h >> 16) & 0xffffu, i = (uint32_t) h & 0xffffu;
        claimed = i < n ? i : n;
        if (i >= n || head_.compare_exchange_weak(h, (h & ~(uint64_t) 0xffffu) | n, std::memory_order_acq_rel,
                                                  std::memory_order_acquire))
            break;
    }
    wait_done((int) claimed);
    idle_live_ = false;
    mode_ = 0;
}

void ExpertPool::run_phase(int mode, int n_tasks) {
    pin_host_once();
    idle_stop();
    mode_ = mode;
    njobs_ = n_tasks;
    const uint32_t e = begin_batch(n_tasks);
    drain(-1, e);
    wait_done(n_tasks);
    hstate_.store(kIdle, std::memory_order_relaxed);
    hstate_ms_.store(now_ms(), std::memory_order_relaxed);
}

int ExpertPool::split_multi_batch(const NativeFmt& f, ExpertJobMulti* jobs, int nb, double& gu_share) {
    mjobs_ = jobs;
    nfmt_ = &f;
    const int act = active_.load(std::memory_order_relaxed);
    const int threads = act + 1;
    const int tpt = cpu_avx512_ok() ? kTasksPerThread : kTasksPerThreadAvx2;
    gu_share = (double) f.down_off / (double) (f.down_off + f.d_row * (size_t) f.n_embd);
    fused_n_ = nb;

    // Enable dual-CCD half-row partitioning only when both L3 domains have active threads and row dims allow 16-row alignment.
    int ccnt[2] = {0, 0};
    if (dual_ccd_) {
        ++ccnt[host_ccd_];
        for (int i = 0; i < act && i < (int) worker_ccd_.size(); ++i) ++ccnt[worker_ccd_[(size_t) i]];
    }
    use_dual_ = dual_ccd_ && ccnt[0] > 0 && ccnt[1] > 0 && f.n_ff >= 32 && f.n_embd >= 32;

    if (use_dual_) {
        dual_gu_mid_ = ccd_split_row((int) f.n_ff);
        dual_d_mid_ = ccd_split_row((int) f.n_embd);
        const int half_threads = (std::max)(1, (std::max)(ccnt[0], ccnt[1]));
        const int gu_min_half = (std::min)(dual_gu_mid_, (int) f.n_ff - dual_gu_mid_);
        const int d_min_half = (std::min)(dual_d_mid_, (int) f.n_embd - dual_d_mid_);
        const int gu_cap = (gu_min_half >= 32 && (gu_min_half & 15) == 0) ? (gu_min_half / 16) : gu_min_half;
        const int d_cap = (d_min_half >= 32 && (d_min_half & 15) == 0) ? (d_min_half / 16) : d_min_half;
        fused_tail_ = half_threads;
        // 4 tasks/thread per half keeps cache-line aligned chunks (~16-32 rows) without atomics churn.
        const int half_tpt = (std::min)(tpt, 6);
        fused_gu_parts_ = (std::min)(gu_cap, (std::max)(1, (int) std::lround(half_tpt * half_threads * gu_share / nb)));
        fused_d_parts_ = (std::min)(d_cap, (std::max)(1, (int) std::lround(half_tpt * half_threads * (1.0 - gu_share) / nb)));
        for (int e = 0; e < nb; ++e) {
            fused_[(size_t) e].gu_left.store(2 * fused_gu_parts_, std::memory_order_relaxed);
            fused_[(size_t) e].ready.store(0, std::memory_order_relaxed);
        }
        dual_ccd_tasks_ = nb * (fused_gu_parts_ + fused_d_parts_);
        return 2 * dual_ccd_tasks_;
    }

    fused_tail_ = threads;
    const int gu_cap = (f.n_ff >= 32 && (f.n_ff & 15) == 0) ? ((int) f.n_ff / 16) : (int) f.n_ff;
    const int d_cap = (f.n_embd >= 32 && (f.n_embd & 15) == 0) ? ((int) f.n_embd / 16) : (int) f.n_embd;
    fused_gu_parts_ = (std::min)(gu_cap, (std::max)(1, (int) std::lround(tpt * threads * gu_share / nb)));
    fused_d_parts_ = (std::min)(d_cap, (std::max)(1, (int) std::lround(tpt * threads * (1.0 - gu_share) / nb)));
    for (int e = 0; e < nb; ++e) {
        fused_[(size_t) e].gu_left.store(fused_gu_parts_, std::memory_order_relaxed);
        fused_[(size_t) e].ready.store(0, std::memory_order_relaxed);
    }
    return nb * (fused_gu_parts_ + fused_d_parts_);
}

void ExpertPool::start_split_multi_native(const NativeFmt& f, ExpertJobMulti* jobs, int n) {
    pin_host_once();
    pend_tasks_ = 0;
    if (n <= 0) return;
    if (n > kMaxSplitMulti) {
        run_split_multi_native(f, jobs, n);
        return;
    }
    pend_t0_ = std::chrono::steady_clock::now();
    const int tasks = split_multi_batch(f, jobs, n, pend_gu_share_);
    idle_stop();
    mode_ = 8;
    njobs_ = tasks;
    pend_epoch_ = begin_batch(tasks);
    pend_tasks_ = tasks;
    pend_n_ = n;
}

void ExpertPool::finish_split_multi_native() {
    if (pend_tasks_ <= 0) return;
    drain(-1, pend_epoch_);
    wait_done(pend_tasks_);
    hstate_.store(kIdle, std::memory_order_relaxed);
    hstate_ms_.store(now_ms(), std::memory_order_relaxed);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - pend_t0_).count();
    ms_multi_gu += ms * pend_gu_share_;
    ms_multi_down += ms * (1.0 - pend_gu_share_);
    multi_bytes += (int64_t) pend_n_ * (int64_t) nfmt_->bytes;
    mode_ = 0;
    pend_tasks_ = 0;
}

void ExpertPool::run_split_multi_native(const NativeFmt& f, ExpertJobMulti* jobs, int n) {
    if (n <= 0) return;
    for (int b0 = 0; b0 < n; b0 += kMaxSplitMulti) {
        const int nb = (std::min)(kMaxSplitMulti, n - b0);
        const auto a = std::chrono::steady_clock::now();
        double gu_share = 0;
        run_phase(8, split_multi_batch(f, jobs + b0, nb, gu_share));
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
        ms_multi_gu += ms * gu_share;
        ms_multi_down += ms * (1.0 - gu_share);
    }
    multi_bytes += (int64_t) n * (int64_t) f.bytes;
    mode_ = 0;
}

}  // namespace basalt::kernels::cpu
