// hecate: multi-block radix select for the DSA indexer top-k of few-row selections over long rows (see the comment in
// dsa_topk.cu's dsa_topk_gr / dev-topk/topk_radix.cu). Needs topk_key_u and topk_find_bucket from the including file.
#pragma once
namespace hecate_radix {
constexpr int RT = 256, RS = 4096, RSLOT = RS / 8, RMAXG = 256;  // threads per CTA, keys per slice (<= 1M keys per row)
constexpr uint16_t RKEY_NEG_INF = 0x03ff;                        // topk_key(0xfc00): -inf

struct Ws
{
    int* h1;      // [R][G][256] high-byte histogram per slice
    int* h2;      // [R][G][256] low-byte histogram of the keys in bucket b1, per slice
    int* s2;      // [R][G][256] suffix sums of h2: count strictly above each low byte
    int* a1;      // [R][G] keys above bucket b1 per slice (b1 < 0: every finite key)
    int* rb;      // [R][2] b1, count above b1
};

__device__ __forceinline__ int row_T(int T, const int* t_ptr, int t_seq, int row)
{
    return t_ptr ? (t_seq > 0 ? t_ptr[row / t_seq] : *t_ptr) : T;
}

// the slice's keys (8 per uint4 slot, two slots per thread), -inf / past-T entries as RKEY_NEG_INF
__device__ __forceinline__ void load_slice(const half* row_s, int T, int g, uint16_t (&key)[2][8])
{
    const uint4* s8 = (const uint4*) row_s;
    #pragma unroll
    for (int u = 0; u < 2; ++u)
    {
        const int slot = g * RSLOT + u * RT + threadIdx.x, i0 = slot * 8;
        uint4 v = make_uint4(0xfc00fc00u, 0xfc00fc00u, 0xfc00fc00u, 0xfc00fc00u);
        if (i0 + 8 <= T) v = s8[slot];
        else if (i0 < T)
        {
            uint16_t tmp[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) tmp[j] = i0 + j < T ? __half_as_ushort(row_s[i0 + j]) : (uint16_t) 0xfc00;
            v = make_uint4(tmp[0] | (tmp[1] << 16), tmp[2] | (tmp[3] << 16), tmp[4] | (tmp[5] << 16), tmp[6] | (tmp[7] << 16));
        }
        const uint32_t w[4] = {v.x, v.y, v.z, v.w};
        #pragma unroll
        for (int j = 0; j < 4; ++j) { key[u][2 * j] = topk_key_u((uint16_t) w[j]); key[u][2 * j + 1] = topk_key_u((uint16_t) (w[j] >> 16)); }
    }
}

// warp-private histograms (8 x 256 in shared memory), summed per bucket
template <typename F>
__device__ __forceinline__ void slice_hist(const uint16_t (&key)[2][8], int (*wh)[256], int* out, F bucket)
{
    const int warp = threadIdx.x / 32;
    for (int i = threadIdx.x; i < 8 * 256; i += RT) (&wh[0][0])[i] = 0;
    __syncthreads();
    #pragma unroll
    for (int u = 0; u < 2; ++u)
        #pragma unroll
        for (int j = 0; j < 8; ++j) { const int b = bucket(key[u][j]); if (b >= 0) atomicAdd(&wh[warp][b], 1); }
    __syncthreads();
    int s = 0;
    #pragma unroll
    for (int w = 0; w < 8; ++w) s += wh[w][threadIdx.x];
    out[threadIdx.x] = s;
}


// rows whose live width fits SMALLG slices: the whole selection in one CTA (three kernels cost more than the work there);
// the same passes, order and output as below
constexpr int SMALLG = 3;
__device__ __forceinline__ void small_select(const half* __restrict__ row_s, int Tr, int k, int k_pad, int* __restrict__ row_o)
{
    __shared__ int wh[8][256];
    __shared__ int tot[256], res[2], wcnt[8];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5, GL = (Tr + RS - 1) / RS;
    uint16_t key[SMALLG][2][8];
    #pragma unroll
    for (int g = 0; g < SMALLG; ++g)
    {
        if (g < GL) load_slice(row_s, Tr, g, key[g]);
        else
            #pragma unroll
            for (int u = 0; u < 2; ++u)
                #pragma unroll
                for (int j = 0; j < 8; ++j) key[g][u][j] = RKEY_NEG_INF;
    }
    for (int i = t; i < 8 * 256; i += RT) (&wh[0][0])[i] = 0;
    __syncthreads();
    #pragma unroll
    for (int g = 0; g < SMALLG; ++g)
        #pragma unroll
        for (int u = 0; u < 2; ++u)
            #pragma unroll
            for (int j = 0; j < 8; ++j) if (key[g][u][j] > RKEY_NEG_INF) atomicAdd(&wh[warp][key[g][u][j] >> 8], 1);
    __syncthreads();
    { int x = 0;
      #pragma unroll
      for (int w = 0; w < 8; ++w) x += wh[w][t];
      tot[t] = x; }
    __syncthreads();
    if (warp == 0) topk_find_bucket(tot, k, lane, res);
    __syncthreads();
    const int b1 = res[0], cnt_hi = res[1];
    uint16_t thresh = RKEY_NEG_INF; int n_eq = 0, all_gt = cnt_hi;
    if (b1 >= 0)
    {
        for (int i = t; i < 8 * 256; i += RT) (&wh[0][0])[i] = 0;
        __syncthreads();                                             // (res / tot reads above done)
        #pragma unroll
        for (int g = 0; g < SMALLG; ++g)
            #pragma unroll
            for (int u = 0; u < 2; ++u)
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                {
                    const uint16_t kk = key[g][u][j];
                    if (kk > RKEY_NEG_INF && (kk >> 8) == b1) atomicAdd(&wh[warp][kk & 0xff], 1);
                }
        __syncthreads();
        { int x = 0;
          #pragma unroll
          for (int w = 0; w < 8; ++w) x += wh[w][t];
          tot[t] = x; }
        __syncthreads();
        if (warp == 0) topk_find_bucket(tot, k - cnt_hi, lane, res);
        __syncthreads();
        const int b0 = max(res[0], 0);
        thresh = (uint16_t) ((b1 << 8) | b0);
        n_eq = (k - cnt_hi) - res[1];
        all_gt = cnt_hi + res[1];
    }
    const int cap = all_gt + n_eq;
    int base_gt = 0, base_eq = all_gt;
    #pragma unroll
    for (int g = 0; g < SMALLG; ++g)
    {
        if (g >= GL) break;
        #pragma unroll
        for (int u = 0; u < 2; ++u)
        {
            unsigned mg = 0, me = 0;
            #pragma unroll
            for (int j = 0; j < 8; ++j)
            {
                const uint16_t kk = key[g][u][j];
                if (kk > RKEY_NEG_INF && kk > thresh) mg |= 1u << j;
                if (b1 >= 0 && kk == thresh) me |= 1u << j;
            }
            const int c = __popc(mg) | (__popc(me) << 16);
            int inc = c;
            #pragma unroll
            for (int o = 1; o < 32; o <<= 1) { const int n = __shfl_up_sync(0xffffffffu, inc, o); if (lane >= o) inc += n; }
            if (lane == 31) wcnt[warp] = inc;
            __syncthreads();
            int wb = 0, tt = 0;
            #pragma unroll
            for (int w = 0; w < 8; ++w) { const int x = wcnt[w]; if (w < warp) wb += x; tt += x; }
            __syncthreads();
            const int ex = wb + inc - c;
            int og = base_gt + (ex & 0xffff), oe = base_eq + (ex >> 16);
            const int i0 = (g * RSLOT + u * RT + t) * 8;
            while (mg) { const int j = __ffs((int) mg) - 1; mg &= mg - 1; row_o[og++] = i0 + j; }
            while (me) { const int j = __ffs((int) me) - 1; me &= me - 1; if (oe < cap) row_o[oe] = i0 + j; ++oe; }
            base_gt += tt & 0xffff; base_eq += tt >> 16;
        }
    }
    for (int j = cap + t; j < k_pad; j += RT) row_o[j] = -1;
}

__global__ void __launch_bounds__(RT) radix_a(const half* __restrict__ scores, int T, int s_stride, const int* __restrict__ t_ptr,
                                               int t_seq, Ws ws, int G, int* __restrict__ out, int k, int k_pad)
{
    const int row = blockIdx.y, g = blockIdx.x;
    const int Tr = row_T(T, t_ptr, t_seq, row);
    if ((Tr + RS - 1) / RS <= SMALLG)                                // short live row: one CTA does all of it
    {
        if (g == 0) small_select(scores + (size_t) row * s_stride, Tr, k, k_pad, out + (size_t) row * k_pad);
        return;
    }
    if (g * RS >= Tr) return;                                        // only the live slices are ever summed
    __shared__ int wh[8][256];
    uint16_t key[2][8];
    load_slice(scores + (size_t) row * s_stride, Tr, g, key);
    slice_hist(key, wh, ws.h1 + ((size_t) row * G + g) * 256, [](uint16_t k) { return k > RKEY_NEG_INF ? (int) (k >> 8) : -1; });
}

__global__ void __launch_bounds__(RT) radix_c(const half* __restrict__ scores, int T, int s_stride, const int* __restrict__ t_ptr,
                                               int t_seq, Ws ws, int G, int k)
{
    const int row = blockIdx.y, g = blockIdx.x, t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int Tr = row_T(T, t_ptr, t_seq, row), GL = (Tr + RS - 1) / RS;
    if (g >= GL || GL <= SMALLG) return;                             // (slice 0 is always live: it writes rb)
    __shared__ int tot[256], mine[256], res[2], wh[8][256], red[8];
    const int* h1 = ws.h1 + (size_t) row * G * 256;
    int s = 0;
    for (int gg = 0; gg < GL; ++gg) s += h1[gg * 256 + t];         // fixed order: every CTA gets the same sums
    tot[t] = s; mine[t] = h1[g * 256 + t];
    __syncthreads();
    if (warp == 0) topk_find_bucket(tot, k, lane, res);
    __syncthreads();
    const int b1 = res[0];
    // keys above bucket b1 in this slice (b1 = -1: every finite key)
    int a = t > b1 ? mine[t] : 0;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
    if (lane == 0) red[warp] = a;
    __syncthreads();
    if (t == 0)
    {
        int x = 0;
        #pragma unroll
        for (int w = 0; w < 8; ++w) x += red[w];
        ws.a1[(size_t) row * G + g] = x;
        if (g == 0) { ws.rb[row * 2] = b1; ws.rb[row * 2 + 1] = res[1]; }
    }
    int* h2 = ws.h2 + ((size_t) row * G + g) * 256;
    int* s2 = ws.s2 + ((size_t) row * G + g) * 256;
    if (b1 < 0) { h2[t] = 0; s2[t] = 0; return; }
    uint16_t key[2][8];
    load_slice(scores + (size_t) row * s_stride, Tr, g, key);
    __syncthreads();                                                 // red / tot reads done (wh reuse is separate)
    slice_hist(key, wh, h2, [b1](uint16_t kk) { return kk > RKEY_NEG_INF && (kk >> 8) == b1 ? (int) (kk & 0xff) : -1; });
    // suffix sums: s2[b] = sum of h2[b'] for b' > b (exclusive, descending scan)
    __syncthreads();
    mine[t] = h2[t];
    __syncthreads();
    // thread t owns bucket 255 - t: inclusive scan over t of mine[255 - t], minus its own
    int v = mine[255 - t], inc = v;
    #pragma unroll
    for (int o = 1; o < 32; o <<= 1) { const int n = __shfl_up_sync(0xffffffffu, inc, o); if (lane >= o) inc += n; }
    if (lane == 31) red[warp] = inc;
    __syncthreads();
    int wbase = 0;
    for (int w = 0; w < warp; ++w) wbase += red[w];
    s2[255 - t] = wbase + inc - v;
}

__global__ void __launch_bounds__(RT) radix_e(const half* __restrict__ scores, int* __restrict__ out, int T, int s_stride,
                                               int k, int k_pad, const int* __restrict__ t_ptr, int t_seq, Ws ws, int G)
{
    const int row = blockIdx.y, g = blockIdx.x, t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int Tr = row_T(T, t_ptr, t_seq, row), GL = (Tr + RS - 1) / RS;
    if (g >= GL || GL <= SMALLG) return;                             // (slice 0 is always live: it writes the padding)
    __shared__ int tot[256], res[2], wcnt[8], sh_pre[3];
    const int b1 = ws.rb[row * 2], cnt_hi = ws.rb[row * 2 + 1];
    uint16_t thresh; int n_eq = 0, b0 = 0;
    if (b1 < 0) thresh = RKEY_NEG_INF;
    else
    {
        const int* h2 = ws.h2 + (size_t) row * G * 256;
        int s = 0;
        for (int gg = 0; gg < GL; ++gg) s += h2[gg * 256 + t];
        tot[t] = s;
        __syncthreads();
        if (warp == 0) topk_find_bucket(tot, k - cnt_hi, lane, res);
        __syncthreads();
        b0 = max(res[0], 0);
        thresh = (uint16_t) ((b1 << 8) | b0);
        n_eq = (k - cnt_hi) - res[1];
    }
    // offsets: keys > v* of the slices before this one, ties of the slices before this one, all keys > v* of the row
    if (warp == 0)
    {
        int pg = 0, pe = 0, ag = 0;
        for (int gg = lane; gg < GL; gg += 32)
        {
            const size_t o = (size_t) row * G + gg;
            const int gt = ws.a1[o] + (b1 < 0 ? 0 : ws.s2[o * 256 + b0]);
            const int eq = b1 < 0 ? 0 : ws.h2[o * 256 + b0];
            if (gg < g) { pg += gt; pe += eq; }
            ag += gt;
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
        {
            pg += __shfl_xor_sync(0xffffffffu, pg, o); pe += __shfl_xor_sync(0xffffffffu, pe, o); ag += __shfl_xor_sync(0xffffffffu, ag, o);
        }
        if (lane == 0) { sh_pre[0] = pg; sh_pre[1] = pe; sh_pre[2] = ag; }
    }
    __syncthreads();
    const int pre_gt = sh_pre[0], pre_eq = sh_pre[1], all_gt = sh_pre[2];
    const int cap = all_gt + n_eq;                                   // = k when the row has >= k finite keys
    int* row_o = out + (size_t) row * k_pad;
    if (g == 0)
        for (int j = cap + t; j < k_pad; j += RT) row_o[j] = -1;
    uint16_t key[2][8];
    load_slice(scores + (size_t) row * s_stride, Tr, g, key);
    int base_gt = pre_gt, base_eq = all_gt + pre_eq;
    #pragma unroll
    for (int u = 0; u < 2; ++u)
    {
        unsigned mg = 0, me = 0;
        #pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            const uint16_t kk = key[u][j];
            if (kk > RKEY_NEG_INF && kk > thresh) mg |= 1u << j;
            if (b1 >= 0 && kk == thresh) me |= 1u << j;
        }
        // packed block scan: > v* counts in the low 16 bits, ties in the high 16 bits (<= 4096 each)
        const int c = __popc(mg) | (__popc(me) << 16);
        int inc = c;
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1) { const int n = __shfl_up_sync(0xffffffffu, inc, o); if (lane >= o) inc += n; }
        if (lane == 31) wcnt[warp] = inc;
        __syncthreads();
        int wb = 0, tt = 0;
        #pragma unroll
        for (int w = 0; w < 8; ++w) { const int x = wcnt[w]; if (w < warp) wb += x; tt += x; }
        __syncthreads();                                             // wcnt reads done before the next round's writes
        const int ex = wb + inc - c;
        int og = base_gt + (ex & 0xffff), oe = base_eq + (ex >> 16);
        const int i0 = (g * RSLOT + u * RT + t) * 8;
        while (mg) { const int j = __ffs((int) mg) - 1; mg &= mg - 1; row_o[og++] = i0 + j; }
        while (me) { const int j = __ffs((int) me) - 1; me &= me - 1; if (oe < cap) row_o[oe] = i0 + j; ++oe; }
        base_gt += tt & 0xffff; base_eq += tt >> 16;
    }
}
}  // namespace hecate_radix
