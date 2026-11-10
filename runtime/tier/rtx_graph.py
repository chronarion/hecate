"""EXL3_RTX_GRAPH=1 (tier_split): the RTX band of every tier layer runs from ONE CUDA graph per verify step instead of
being enqueued layer by layer from Python (the per-layer rtx_submit cost ~150 us of host time: cuda-python stream memops,
the coop launch, device/stream context switches).

Per tier layer L: its own pinned slots (x, sel, rw, out) and two flag words (ISSUE_L, DONE_L, one cache line each).
RTX graph for R rows (captured once per R): for every layer, a 1-thread kernel spins until ISSUE_L >= 1 and resets it,
the ExLlamaV3 coop launches read the slot and write out_L, a 1-thread kernel releases DONE_L = 1.
GB300 per decode layer: the handoff kernel writes slot L and releases ISSUE_L = 1; the combine spins until DONE_L >= 1,
resets it, then adds out_L. The graph is launched at the first tier layer of each decode forward (rows <= 8).
Prefill (> 8 rows) keeps rtx_split's per-layer path (its own slots and generation flags)."""
import os
import torch
from torch.utils.cpp_extension import load_inline
import rtx_split as rs
from cuda.bindings import driver as cu

_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
__global__ void wait_reset_k(unsigned* f, unsigned want, unsigned reset)
{
    while (*(volatile unsigned*) f < want) __nanosleep(200);
    *(volatile unsigned*) f = reset;
    __threadfence_system();
}
__global__ void release_k(unsigned* f, unsigned v)
{
    __threadfence_system();
    asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(f), "r"(v) : "memory");
}
// hecate (EXL3_RTX_STAGE1): wait for the layer's ISSUE and copy its slot (rows y, selection, weights; pinned host memory
// written by the GB300) into the RTX VRAM staging buffers in one kernel: the three copy-engine nodes it replaces cost
// ~14 us from ISSUE to the first RTX kernel (node transitions), this kernel ~one PCIe round trip. ISSUE is reset after
// the copy (the next release of this layer's ISSUE is a step later)
__global__ void __launch_bounds__(1024) wait_stage_k(unsigned* f, const uint4* ys, const uint4* ss, const uint4* ws,
                                                     uint4* yd, uint4* sd, uint4* wd, int ny, int ns, int nw)
{
    if (threadIdx.x == 0)
    {
        unsigned v;
        do { asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(f) : "memory"); if (v < 1u) __nanosleep(200); } while (v < 1u);
    }
    __syncthreads();
    uint4 a = {}, b = {};
    if (threadIdx.x < ns) a = __ldcv(ss + threadIdx.x);
    if (threadIdx.x < nw) b = __ldcv(ws + threadIdx.x);
    for (int base = 0; base < ny; base += 4 * 1024)          // all loads of a 64 KB batch in flight before its stores
    {
        uint4 t[4];
        #pragma unroll
        for (int k = 0; k < 4; ++k) { const int i = base + threadIdx.x + k * 1024; if (i < ny) t[k] = __ldcv(ys + i); }
        #pragma unroll
        for (int k = 0; k < 4; ++k) { const int i = base + threadIdx.x + k * 1024; if (i < ny) yd[i] = t[k]; }
    }
    if (threadIdx.x < ns) sd[threadIdx.x] = a;
    if (threadIdx.x < nw) wd[threadIdx.x] = b;
    if (threadIdx.x == 0) { *(volatile unsigned*) f = 0u; __threadfence_system(); }
}
void wait_reset(int64_t f, int64_t want, int64_t reset, int64_t dev)
{
    const at::cuda::CUDAGuard g((c10::DeviceIndex) dev);
    wait_reset_k<<<1, 1, 0, at::cuda::getCurrentCUDAStream().stream()>>>((unsigned*) f, (unsigned) want, (unsigned) reset);
}
void release(int64_t f, int64_t v, int64_t dev)
{
    const at::cuda::CUDAGuard g((c10::DeviceIndex) dev);
    release_k<<<1, 1, 0, at::cuda::getCurrentCUDAStream().stream()>>>((unsigned*) f, (unsigned) v);
}
// byte counts must be multiples of 16 (the selection and weights at most 16 KB each)
void wait_stage(int64_t f, int64_t ys, int64_t ss, int64_t ws, int64_t yd, int64_t sd, int64_t wd, int64_t ny, int64_t ns, int64_t nw, int64_t dev)
{
    TORCH_CHECK(ny % 16 == 0 && ns % 16 == 0 && nw % 16 == 0 && ns <= 1024 * 16 && nw <= 1024 * 16, "wait_stage: sizes");
    const at::cuda::CUDAGuard g((c10::DeviceIndex) dev);
    wait_stage_k<<<1, 1024, 0, at::cuda::getCurrentCUDAStream().stream()>>>((unsigned*) f, (const uint4*) ys, (const uint4*) ss,
        (const uint4*) ws, (uint4*) yd, (uint4*) sd, (uint4*) wd, (int) (ny / 16), (int) (ns / 16), (int) (nw / 16));
}
"""
_BD = os.path.join(os.environ.get("EXL3_TIER_BUILD", os.environ.get("EXL3_MOE4_SRC", "/opt/moe4")), "build_rtxflags"); os.makedirs(_BD, exist_ok=True)
_fk = load_inline(name="rtxflags2", cpp_sources="void wait_reset(int64_t f, int64_t want, int64_t reset, int64_t dev);\n"
                  "void release(int64_t f, int64_t v, int64_t dev);\n"
                  "void wait_stage(int64_t f, int64_t ys, int64_t ss, int64_t ws, int64_t yd, int64_t sd, int64_t wd, int64_t ny, "
                  "int64_t ns, int64_t nw, int64_t dev);", cuda_sources=_SRC, functions=["wait_reset", "release", "wait_stage"],
                  build_directory=_BD,
                  extra_cuda_cflags=["-O3", "-gencode=arch=compute_103a,code=sm_103a", "-gencode=arch=compute_120a,code=sm_120a"])

R8 = 8
# EXL3_RTX_MOE3=1 (default): the band runs on the Hecate trellis MoE kernels (tier/moe3/moe3.cu); 0 = ExLlamaV3's coop pair.
# The int4 far experts on the RTX (EXL3_TIER_RTX_I4, larger-KV profiles) feed the coop kernel's shared input, so those
# profiles keep the coop pair
_MOE3 = os.environ.get("EXL3_RTX_MOE3", "1") == "1" and int(os.environ.get("EXL3_TIER_RTX_I4", "0") or 0) == 0
M3 = None
if not _MOE3 and os.environ.get("EXL3_RTX_MOE3", "1") == "1":
    print("rtx_graph: EXL3_TIER_RTX_I4 is set, the RTX band runs on ExLlamaV3's coop kernels", flush=True)
if _MOE3:
    _BD3 = os.path.join(os.environ.get("EXL3_TIER_BUILD", "/tmp"), "build_moe3"); os.makedirs(_BD3, exist_ok=True)
    from torch.utils.cpp_extension import load as _load
    M3 = _load(name="moe3", sources=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "moe3", "moe3.cu")],
               build_directory=_BD3, verbose=False, extra_cuda_cflags=["-O3", "-gencode=arch=compute_120a,code=sm_120a", "--use_fast_math"])
MODS = []          # tier modules in load (= forward) order; tier_split appends
_G = None
MOE4R = None       # moe4 built for sm_120a (tier_split, EXL3_TIER_RTX_I4): int4 far experts of a layer (m._rtx4) on the RTX


class _Slots:
    def __init__(self, n, H, topk):
        S = rs._S
        self.n, self.H = n, H
        self.ny, self.ns, self.nw, self.no = R8 * H * 2, R8 * topk * 8, R8 * topk * 2, R8 * H * 4
        self.per = (self.ny + self.ns + self.nw + self.no + 128 + 255) // 256 * 256      # + ISSUE / DONE lines
        tot = self.per * n
        with torch.cuda.device(S.g):
            self.hbuf = int(rs._ck(cu.cuMemHostAlloc(tot, cu.CU_MEMHOSTALLOC_PORTABLE | cu.CU_MEMHOSTALLOC_DEVICEMAP)))
        import ctypes; ctypes.memset(self.hbuf, 0, tot)
        self.dbase = {}
        for d in (S.g, S.r):
            with torch.cuda.device(d):
                self.dbase[d.index] = int(rs._ck(cu.cuMemHostGetDevicePointer(self.hbuf, 0)))

        class _CAI:
            def __init__(s, ptr, shape, ts): s.__cuda_array_interface__ = {"shape": shape, "typestr": ts, "data": (ptr, False), "version": 3}
        self.views = {}
        for d in (S.g, S.r):
            vs = []
            with torch.cuda.device(d):
                for i in range(n):
                    b = self.dbase[d.index] + i * self.per
                    vs.append(dict(y=torch.as_tensor(_CAI(b, (R8, H), "<f2"), device=d),
                                   s=torch.as_tensor(_CAI(b + self.ny, (R8, topk), "<i8"), device=d),
                                   w=torch.as_tensor(_CAI(b + self.ny + self.ns, (R8, topk), "<f2"), device=d),
                                   o=torch.as_tensor(_CAI(b + self.ny + self.ns + self.nw, (R8, H), "<f4"), device=d),
                                   issue=b + self.ny + self.ns + self.nw + self.no, done=b + self.ny + self.ns + self.nw + self.no + 64))
            self.views[d.index] = vs
        print(f"rtx_graph: {n} layer slots, {tot / 2**20:.0f} MiB pinned", flush=True)


_RTX_STAGE = os.environ.get("EXL3_RTX_STAGE", "1") == "1"   # stage the slot (x / sel / rw) into RTX VRAM per layer
# EXL3_RTX_STAGE1=1: the ISSUE wait and the slot copy as one kernel (wait_stage_k) instead of wait_reset_k + 3 copy nodes
_RTX_STAGE1 = _RTX_STAGE and os.environ.get("EXL3_RTX_STAGE1", "0") == "1"


class _Graph:
    def __init__(self):
        S = rs._S
        self.slots = _Slots(len(MODS), S.H, S.topk)
        for i, m in enumerate(MODS): m._gslot = i
        self.graphs = {}

    def graph(self, R):
        g = self.graphs.get(R)
        if g is not None:
            return g
        S = rs._S; sv = self.slots.views[S.r.index]; ri = S.r.index
        if _RTX_STAGE and getattr(self, "stage_y", None) is None:
            v0 = sv[0]
            self.stage_y = torch.empty_like(v0["y"], device=S.r); self.stage_s = torch.empty_like(v0["s"], device=S.r)
            self.stage_w = torch.empty_like(v0["w"], device=S.r)
        r4m = [m for m in MODS if getattr(m, "_rtx4", None) is not None]
        if r4m and getattr(self, "r4y", None) is None:
            assert _RTX_STAGE, "EXL3_TIER_RTX_I4 needs EXL3_RTX_STAGE=1 (moe4 reads x / sel / rw from VRAM)"
            with torch.cuda.device(S.r):
                self.r4y = torch.zeros(R8, S.H, dtype=torch.float, device=S.r)
                self.r4scr = MOE4R.scratch(S.r)
                sms = torch.cuda.get_device_properties(S.r).multi_processor_count
                self.r4b = (sms * 2, sms * 3)
                # one eager call outside the capture (no picks: sets the kernel attributes, checks the build)
                x0 = torch.zeros(1, S.H, dtype=torch.half, device=S.r); s0 = torch.full((1, S.topk), -1, dtype=torch.long, device=S.r)
                t4 = r4m[0]._rtx4.tab
                MOE4R.forward(x0, s0, torch.zeros(1, S.topk, dtype=torch.half, device=S.r), r4m[0]._rtx4.k, t4["gate_w"], t4["up_w"],
                              t4["down_w"], t4["gate_s"], t4["up_s"], t4["down_s"], *self.r4scr, self.r4y[:1], self.r4b[0],
                              self.r4b[1], r4m[0]._rtx4.lo)
                torch.cuda.synchronize(S.r)
            print(f"rtx_graph: int4 far experts on the RTX in {len(r4m)} layers (moe4 grid {self.r4b})", flush=True)
        if _MOE3 and getattr(self, "m3scr", None) is None:
            assert not r4m and _RTX_STAGE, "EXL3_RTX_MOE3 needs EXL3_RTX_STAGE=1"
            with torch.cuda.device(S.r):
                self.m3scr = M3.scratch(S.r, S.H, 2048)
                m0 = MODS[0]; T0 = m0._rtx.tabs        # one eager call with no picks (kernel attributes, build check)
                x0 = torch.zeros(1, S.H, dtype=torch.half, device=S.r); s0 = torch.full((1, S.topk), -1, dtype=torch.long, device=S.r)
                M3.forward(x0, s0, torch.zeros(1, S.topk, dtype=torch.half, device=S.r), m0._rtx.first, m0._rtx.E - m0._rtx.first,
                           *T0, *self.m3scr, torch.zeros(1, S.H, dtype=torch.float, device=S.r))
                torch.cuda.synchronize(S.r)
            print("rtx_graph: RTX band on the moe3 kernels", flush=True)
        g = torch.cuda.CUDAGraph()
        torch.cuda.synchronize(S.r)
        with torch.cuda.device(S.r), torch.cuda.graph(g, stream=S.stream):
            for i, m in enumerate(MODS):
                v, Rt = sv[i], m._rtx; T = Rt.tabs
                yin, sin, win = v["y"], v["s"], v["w"]
                if _RTX_STAGE1:
                    yin, sin, win = self.stage_y[:R], self.stage_s[:R], self.stage_w[:R]
                    _fk.wait_stage(v["issue"], v["y"].data_ptr(), v["s"].data_ptr(), v["w"].data_ptr(), yin.data_ptr(),
                                   sin.data_ptr(), win.data_ptr(), yin.numel() * 2, sin.numel() * 8, win.numel() * 2, ri)
                else:
                    _fk.wait_reset(v["issue"], 1, 0, ri)
                if _RTX_STAGE and not _RTX_STAGE1:   # one bulk PCIe read of the slot instead of per-warp reads in the coop kernels
                    yin, sin, win = self.stage_y[:R], self.stage_s[:R], self.stage_w[:R]
                    yin.copy_(v["y"][:R]); sin.copy_(v["s"][:R]); win.copy_(v["w"][:R])
                r4, sh = getattr(m, "_rtx4", None), None
                if r4 is not None:
                    # int4 far experts first, into a zeroed VRAM partial that the coop kernel adds into the slot output
                    # (its shared-expert input, written for every row incl. rows without a band pick)
                    sh = self.r4y[:R]
                    sh.zero_()
                    t4 = r4.tab
                    MOE4R.forward(yin, sin, win, r4.k, t4["gate_w"], t4["up_w"], t4["down_w"], t4["gate_s"], t4["up_s"],
                                  t4["down_s"], *self.r4scr, sh, self.r4b[0], self.r4b[1], r4.lo)
                if _MOE3:
                    M3.forward(yin[:R], sin[:R], win[:R], Rt.first, Rt.E - Rt.first, *T, *self.m3scr, v["o"][:R])
                for c in (range(0, R, rs.CHUNK) if not _MOE3 else ()):
                    n = min(rs.CHUNK, R - c)
                    rs.ext.exl3_moe_coop(yin[c:c + n], sin[c:c + n], win[c:c + n], Rt.first, Rt.E, Rt.H,
                                         T[0], T[1], T[2], T[3], T[4], T[5], T[6], T[7], T[8], None, None, None,
                                         Rt.K[0], Rt.K[1], Rt.K[2], False, True, 0, 0.0, True,
                                         S.had_g, S.had_u, S.gu_g, S.gu_u, S.act, S.dout, S.ctr, v["o"][c:c + n],
                                         sh[c:c + n] if sh is not None else None, None)
                _fk.release(v["done"], 1, ri)
        torch.cuda.synchronize(S.r)
        self.graphs[R] = g
        print(f"rtx_graph: captured the {len(MODS)}-layer RTX graph for {R} rows", flush=True)
        return g


def get():
    global _G
    if _G is None:
        _G = _Graph()
    return _G


def launch(R):
    """first tier layer of a decode forward: replay the whole step's RTX work (it waits on each layer's ISSUE)"""
    G = get(); S = rs._S
    g = G.graph(R)
    with torch.cuda.device(S.r), torch.cuda.stream(S.stream):
        g.replay()


def handoff(m, y, sel, rw, bsz, moe4, cpu=None, zero2=0):
    """GB300: write layer m's slot and release its ISSUE (one kernel); cpu = (meta_ptr, layer, flag_ptr, gen) also posts
    the job to the Grace band pool (its own generation flag: the RTX graph resets ISSUE_L as soon as it sees it)"""
    G = get(); S = rs._S; v = G.slots.views[S.g.index][m._gslot]
    meta, lay, f2, g2 = cpu if cpu is not None else (0, 0, 0, 0)
    if zero2:      # moe4 v8: also zero the Grace band's CPU_DONE before its ISSUE (fused combine, tier_split CF)
        moe4.handoff2(y.view(bsz, -1), sel.view(bsz, -1), rw.view(bsz, -1), v["y"].data_ptr(), v["s"].data_ptr(),
                      v["w"].data_ptr(), v["issue"], 1, bsz, meta, lay, f2, g2, v["done"], zero2)
        return
    moe4.handoff(y.view(bsz, -1), sel.view(bsz, -1), rw.view(bsz, -1), v["y"].data_ptr(), v["s"].data_ptr(), v["w"].data_ptr(),
                 v["issue"], 1, bsz, meta, lay, f2, g2, v["done"])


_RLOG = {}


def log_routing(R):
    """EXL3_ROUTE_LOG=path: append this decode step's expert selection of every tier layer, read from the pinned slots
    after the verify forward has finished (call before the next step can be launched). Record: u16 0xA55A, u16 layers,
    u16 rows, then int16 [layers][rows][top-k] placement-local ids (0 .. 255 in the layer's placement order); the first
    record of a file is a header: u16 0xA55B, u16 layers, then int16 layer index per slot. tools/placement/route_stats.py
    reads it."""
    path = os.environ.get("EXL3_ROUTE_LOG")
    if not path or R < 1 or R > R8 or not MODS:
        return
    import numpy as np, struct
    if not _RLOG:
        import ctypes
        sl = get().slots
        _RLOG["f"] = open(path, "ab", buffering=1 << 20)
        _RLOG["a"] = np.ctypeslib.as_array((ctypes.c_uint8 * (sl.per * sl.n)).from_address(sl.hbuf)).reshape(sl.n, sl.per)
        _RLOG["sl"] = sl
        lay = []
        for m in MODS:
            k = (getattr(m, "key", "") or "").split(".")
            lay.append(int(k[k.index("layers") + 1]) if "layers" in k else -1)
        _RLOG["f"].write(struct.pack("<HH", 0xA55B, len(lay)) + np.asarray(lay, np.int16).tobytes())
    sl, topk = _RLOG["sl"], rs._S.topk
    s = _RLOG["a"][:, sl.ny:sl.ny + R * topk * 8].copy().view(np.int64).reshape(sl.n, R, topk)
    _RLOG["f"].write(struct.pack("<HHH", 0xA55A, sl.n, R) + s.astype(np.int16).tobytes())


def slot_view(m):
    """layer m's GB300-side slot views (y / s / w host-mapped tensors, issue / done flag pointers)"""
    return get().slots.views[rs._S.g.index][m._gslot]


def slot_layout():
    """(host base, per-slot stride, y / sel / rw offsets) of the per-layer slots, for the CPU band pool"""
    sl = get().slots
    return sl.hbuf, sl.per, 0, sl.ny, sl.ny + sl.ns


def wait_done(m, dev):
    """(out slot, DONE flag) of layer m on the GB300: the fused combine kernel spins on DONE itself; the layer's next
    handoff resets it before releasing ISSUE"""
    v = get().slots.views[rs._S.g.index][m._gslot]
    return v["o"].data_ptr(), v["done"]
