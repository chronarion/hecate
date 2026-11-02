// Grace CPU engine for the cold expert band (hecate, tier_split EXL3_TIER_CPU): a persistent pool of pinned threads
// spins on the RTX handoff's ISSUE flag (pinned host memory, written by the GB300 with st.release.sys), reads the step's
// MoE input / selection / routing weights from the shared slots, computes the picked cold experts (uniform int4 per row,
// row-major, 2 codes per byte, low nibble first; fp32 row scales) for <= 8 rows, writes the fp32 partial into a pinned
// slot and releases CPU_DONE = gen, which the GB300 stream waits on before its fused combine.
// Activations are quantized to int8 with a scale per 32 values (q8_0-style) and multiplied with SDOT.
// meta[0] = layer, meta[1] = rows (written by the handoff kernel before ISSUE). rows > 8 (prefill): acknowledge only.
#include <torch/extension.h>
#include <arm_neon.h>
#include <atomic>
#include <cmath>
#include <cstring>
#include <pthread.h>
#include <sched.h>
#include <thread>
#include <time.h>
#include <vector>

namespace {

constexpr int H = 6144, I = 2048, MAXR = 8, TOPK = 8, MAXP = 8;

struct Layer { int lo = 0, n = 0, slot = -1; std::vector<const uint8_t*> g, u, d; std::vector<const float*> gs, us, ds; };

struct Pool
{
    std::vector<Layer> layers;                  // indexed by layer id
    const volatile uint32_t* issue = nullptr;   // RTX ISSUE flag (host)
    const volatile uint32_t* meta = nullptr;    // [layer, rows]
    uint32_t* done = nullptr;                   // CPU_DONE flag (host)
    const uint16_t* x = nullptr;                // [CAP][H] fp16 slot
    const int64_t* sel = nullptr;               // [CAP][TOPK]
    const uint16_t* rw = nullptr;               // [CAP][TOPK] fp16
    float* out = nullptr;                       // [MAXR][H] fp32 partial
    uintptr_t slot_base = 0; size_t slot_stride = 0, oy = 0, os = 0, ow = 0;   // slot mode (per-layer graph slots)
    int nt = 0, core0 = 0;
    bool skip = false;                          // EXL3_TIER_CPU_SKIP=1: timing diagnostic, jobs acknowledged with a zero partial
    std::vector<std::thread> th;
    std::atomic<bool> stop{false};
    // per job (written by thread 0 between barriers)
    // hecate: counter and sense on separate cache lines (adjacent, every fetch_add fought the spinners for the line)
    alignas(128) std::atomic<uint32_t> bar_count{0};
    alignas(128) std::atomic<uint32_t> bar_sense{0};
    int R = 0, P = 0, L = -1; int pick[MAXP]; float wgt[MAXP][MAXR]; int nr[MAXP]; int rows[MAXP][MAXR];
    alignas(64) int8_t xqp[MAXP][MAXR * H]; float xgp[MAXP][MAXR * H / 32];   // x rows of each pick's rows, compacted
    alignas(64) int8_t xq[MAXR * H]; float xg[MAXR * H / 32];
    alignas(64) float gy[MAXP][MAXR][I], uy[MAXP][MAXR][I];
    alignas(64) int8_t aq[MAXP][MAXR][I]; float ag[MAXP][MAXR][I / 32];
    uint64_t jobs = 0, busy = 0;
};
Pool* g_pool = nullptr;

inline float h2f(uint16_t h) { __fp16 v; memcpy(&v, &h, 2); return (float) v; }

void barrier(Pool& P, uint32_t& local_sense)
{
    local_sense ^= 1u;
    if (P.bar_count.fetch_add(1, std::memory_order_acq_rel) + 1 == (uint32_t) P.nt)
    {
        P.bar_count.store(0, std::memory_order_relaxed);
        P.bar_sense.store(local_sense, std::memory_order_release);
    }
    else
        while (P.bar_sense.load(std::memory_order_acquire) != local_sense) asm volatile("yield");
}

inline void split(int n, int t, int nt, int& a, int& b) { a = (int) ((int64_t) n * t / nt); b = (int) ((int64_t) n * (t + 1) / nt); }

// rows [a, b) of y[r][n] = s[n] * sum_k (q[n][k] - 8) * x[r][k], x as int8 groups of 32 (xq, xg), R rows
inline void gemv_rows(const uint8_t* q, const float* s, const int8_t* xq, const float* xg, int K, int R, int a, int b,
                      float* y, size_t ystride)
{
    for (int n = a; n < b; ++n)
    {
        const uint8_t* qr = q + (size_t) n * K / 2;
        float32x4_t acc[MAXR];
        for (int r = 0; r < R; ++r) acc[r] = vdupq_n_f32(0.f);
        for (int k = 0; k < K; k += 32)
        {
            const uint8x16_t bb = vld1q_u8(qr + k / 2);
            const int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(bb, vdupq_n_u8(15))), vdupq_n_s8(8));
            const int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(bb, 4)), vdupq_n_s8(8));
            const int8x16x2_t z = vzipq_s8(lo, hi);
            for (int r = 0; r < R; ++r)
            {
                const int8_t* xr = xq + (size_t) r * K + k;
                int32x4_t d = vdotq_s32(vdupq_n_s32(0), z.val[0], vld1q_s8(xr));
                d = vdotq_s32(d, z.val[1], vld1q_s8(xr + 16));
                acc[r] = vfmaq_n_f32(acc[r], vcvtq_f32_s32(d), xg[(size_t) r * (K / 32) + k / 32]);
            }
        }
        for (int r = 0; r < R; ++r) y[(size_t) r * ystride + n] = s[n] * vaddvq_f32(acc[r]);
    }
}

inline void quant_group(const float* v, int8_t* q, float* sc)
{
    float m = 0.f; for (int k = 0; k < 32; ++k) m = fmaxf(m, fabsf(v[k]));
    const float s = m > 0 ? m / 127.f : 1.f; *sc = s; const float inv = 1.f / s;
    for (int k = 0; k < 32; ++k) q[k] = (int8_t) lrintf(v[k] * inv);
}

void worker(int t)
{
    Pool& P = *g_pool;
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(P.core0 + t, &cs); pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
    uint32_t last = __atomic_load_n((const uint32_t*) P.issue, __ATOMIC_ACQUIRE), sense = 0;
    auto now_ns = [] { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint64_t) t.tv_sec * 1000000000ull + t.tv_nsec; };
    uint64_t last_job = now_ns(), spins = 0;
    while (!P.stop.load(std::memory_order_relaxed))
    {
        uint32_t g = __atomic_load_n((const uint32_t*) P.issue, __ATOMIC_ACQUIRE);
        if (g == last)
        {
            // spin through decode incl. step boundaries (jobs ~0.5 ms apart, boundaries ~6 ms); idle server: back off by
            // time since the last job: spin < 20 ms, then 20 us sleeps, then 1 ms sleeps after 1 s
            if ((++spins & 1023) == 0 || spins > (1u << 20))
            {
                const uint64_t idle = now_ns() - last_job;
                if (idle > 1000000000ull) { struct timespec ts = {0, 1000000}; nanosleep(&ts, nullptr); continue; }
                if (idle > 20000000ull) { struct timespec ts = {0, 20000}; nanosleep(&ts, nullptr); continue; }
            }
            asm volatile("yield");
            continue;
        }
        spins = 0; last = g; last_job = now_ns();
        // every thread reads the same job header; thread 0 derives the pick list
        if (t == 0)
        {
            P.L = (int) P.meta[0]; P.R = (int) P.meta[1]; P.P = 0;
            const bool ok = P.R >= 1 && P.R <= MAXR && P.L >= 0 && P.L < (int) P.layers.size() && P.layers[P.L].n > 0;
            if (ok)
            {
                const Layer& Ly = P.layers[P.L];
                if (P.slot_base)                     // per-layer slots (rtx_graph): this layer's x / sel / rw
                {
                    const uintptr_t b = P.slot_base + (size_t) Ly.slot * P.slot_stride;
                    P.x = (const uint16_t*) (b + P.oy); P.sel = (const int64_t*) (b + P.os); P.rw = (const uint16_t*) (b + P.ow);
                }
                for (int i = 0; i < P.R * TOPK; ++i)
                {
                    const int64_t e = P.sel[i] - Ly.lo;
                    if (e < 0 || e >= Ly.n) continue;
                    int j = 0; while (j < P.P && P.pick[j] != (int) e) ++j;
                    if (j == P.P) { if (P.P == MAXP) continue; P.pick[P.P] = (int) e; for (int r = 0; r < MAXR; ++r) P.wgt[P.P][r] = 0.f; ++P.P; }
                    P.wgt[j][i / TOPK] += h2f(P.rw[i]);
                }
            }
            else P.R = (P.R >= 1 && P.R <= MAXR) ? P.R : 0;
            for (int p = 0; p < P.P; ++p)
            {
                P.nr[p] = 0;
                for (int r = 0; r < P.R; ++r) if (P.wgt[p][r] != 0.f) P.rows[p][P.nr[p]++] = r;
            }
        }
        barrier(P, sense);
        const int R = P.R, NP = P.skip ? 0 : P.P;
        if (R > 0 && NP == 0)
        {
            int a, b; split(R * H, t, P.nt, a, b); memset(P.out + a, 0, (size_t) (b - a) * 4);
        }
        else if (R > 0)
        {
            const Layer& Ly = P.layers[P.L];
            // x -> int8 groups
            { int a, b; split(R * H / 32, t, P.nt, a, b);
              for (int gi = a; gi < b; ++gi) { float v[32]; for (int k = 0; k < 32; ++k) v[k] = h2f(P.x[(size_t) gi * 32 + k]); quant_group(v, P.xq + gi * 32, P.xg + gi); } }
            barrier(P, sense);
            { int a, b; split(NP * MAXR, t, P.nt, a, b);
              for (int w = a; w < b; ++w) { const int p = w / MAXR, j = w % MAXR; if (j >= P.nr[p]) continue; const int r = P.rows[p][j];
                  memcpy(P.xqp[p] + (size_t) j * H, P.xq + (size_t) r * H, H); memcpy(P.xgp[p] + (size_t) j * (H / 32), P.xg + (size_t) r * (H / 32), H / 32 * 4); } }
            barrier(P, sense);
            // gate + up rows of every pick
            { int a, b; split(NP * 2 * I, t, P.nt, a, b);
              for (int w = a; w < b; ) {
                  const int p = w / (2 * I), m = (w / I) % 2, n0 = w % I, n1 = std::min(I, n0 + (b - w));
                  const int e = P.pick[p];
                  gemv_rows(m ? Ly.u[e] : Ly.g[e], m ? Ly.us[e] : Ly.gs[e], P.xqp[p], P.xgp[p], H, P.nr[p], n0, n1, m ? &P.uy[p][0][0] : &P.gy[p][0][0], I);
                  w += n1 - n0; } }
            barrier(P, sense);
            // act = silu(g) * u * routing weight -> int8 groups
            { int a, b; split(NP * MAXR * I / 32, t, P.nt, a, b);
              for (int gi = a; gi < b; ++gi) {
                  const int p = gi / (MAXR * I / 32), r = (gi / (I / 32)) % MAXR, k0 = (gi % (I / 32)) * 32;
                  if (r >= P.nr[p]) continue;
                  float v[32]; const float wt = P.wgt[p][P.rows[p][r]];
                  for (int k = 0; k < 32; ++k) { const float gg = P.gy[p][r][k0 + k]; v[k] = gg / (1.f + expf(-gg)) * P.uy[p][r][k0 + k] * wt; }
                  quant_group(v, &P.aq[p][r][k0], &P.ag[p][r][k0 / 32]); } }
            barrier(P, sense);
            // down rows, summed over picks in registers
            { int a, b; split(H, t, P.nt, a, b);
              float tmp[MAXR];
              for (int n = a; n < b; ++n) {
                  for (int r = 0; r < R; ++r) tmp[r] = 0.f;
                  for (int p = 0; p < NP; ++p) {
                      float y1[MAXR]; const int e = P.pick[p];
                      gemv_rows(Ly.d[e], Ly.ds[e], &P.aq[p][0][0], &P.ag[p][0][0], I, P.nr[p], n, n + 1, y1 - n, 1);
                      for (int j = 0; j < P.nr[p]; ++j) tmp[P.rows[p][j]] += y1[j]; }
                  for (int r = 0; r < R; ++r) P.out[(size_t) r * H + n] = tmp[r]; } }
        }
        barrier(P, sense);
        if (t == 0) { ++P.jobs; P.busy += NP; __atomic_store_n(P.done, g, __ATOMIC_RELEASE); }
    }
}

}  // namespace

// one layer's cold band: local ids [lo, lo + n); per expert row-major int4 codes + fp32 scales (CPU tensors, kept alive)
void cb_add_layer(int64_t layer, int64_t lo, int64_t slot, std::vector<torch::Tensor> g, std::vector<torch::Tensor> u, std::vector<torch::Tensor> d,
                  std::vector<torch::Tensor> gs, std::vector<torch::Tensor> us, std::vector<torch::Tensor> ds)
{
    if (!g_pool) g_pool = new Pool();
    if ((int) g_pool->layers.size() <= layer) g_pool->layers.resize(layer + 1);
    Layer& L = g_pool->layers[layer]; L.lo = (int) lo; L.n = (int) g.size(); L.slot = (int) slot;
    for (size_t i = 0; i < g.size(); ++i)
    {
        L.g.push_back(g[i].data_ptr<uint8_t>()); L.u.push_back(u[i].data_ptr<uint8_t>()); L.d.push_back(d[i].data_ptr<uint8_t>());
        L.gs.push_back(gs[i].data_ptr<float>()); L.us.push_back(us[i].data_ptr<float>()); L.ds.push_back(ds[i].data_ptr<float>());
    }
}

void cb_start(int64_t issue, int64_t meta, int64_t done, int64_t x, int64_t sel, int64_t rw, int64_t out, int64_t threads, int64_t core0,
              int64_t slot_base, int64_t slot_stride, int64_t oy, int64_t os, int64_t ow)
{
    TORCH_CHECK(g_pool && g_pool->th.empty(), "cpuband: add layers first, start once");
    Pool& P = *g_pool;
    P.issue = (const volatile uint32_t*) issue; P.meta = (const volatile uint32_t*) meta; P.done = (uint32_t*) done;
    P.x = (const uint16_t*) x; P.sel = (const int64_t*) sel; P.rw = (const uint16_t*) rw; P.out = (float*) out;
    P.nt = (int) threads; P.core0 = (int) core0;
    { const char* e = getenv("EXL3_TIER_CPU_SKIP"); P.skip = e && atoi(e) != 0; }
    P.slot_base = (uintptr_t) slot_base; P.slot_stride = (size_t) slot_stride; P.oy = (size_t) oy; P.os = (size_t) os; P.ow = (size_t) ow;
    for (int t = 0; t < P.nt; ++t) P.th.emplace_back(worker, t);
}

std::vector<int64_t> cb_stats() { return g_pool ? std::vector<int64_t>{(int64_t) g_pool->jobs, (int64_t) g_pool->busy} : std::vector<int64_t>{0, 0}; }

void cb_stop()
{
    if (!g_pool) return;
    g_pool->stop = true; for (auto& t : g_pool->th) t.join(); g_pool->th.clear(); g_pool->stop = false;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("add_layer", &cb_add_layer);
    m.def("start", &cb_start);
    m.def("stats", &cb_stats);
    m.def("stop", &cb_stop);
}
