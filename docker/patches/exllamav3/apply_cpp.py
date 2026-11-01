"""C++ patch for ExLlamaV3 v1.6.5 (hecate): EXL3_INT8_SQ_MAX_M (default 2, max 4) lets the per-slice int8
GEMV take 3-4 activation rows (MTP verification width) instead of falling through to the fp16 GEMM. The
kernel is generic in M; only the M = 3, 4 instances (plain int8, K = 4..6) are added."""
import pathlib, sys
root = pathlib.Path(sys.argv[1])

def sub(rel, old, new, count=1):
    p = root / rel; s = p.read_text()
    assert s.count(old) == count, (rel, old[:70], s.count(old))
    p.write_text(s.replace(old, new))

sub("quant/exl3_gemv_int8.cu", "static int _exl3_gemv_int8_mode = 0;",
"""static int exl3_gemv_int8_sq_max_m()
{
    static const int v = [] { const char* e = getenv("EXL3_INT8_SQ_MAX_M"); int m = e ? atoi(e) : 2; return m < 1 ? 1 : (m > 4 ? 4 : m); }();
    return v;
}

static int _exl3_gemv_int8_mode = 0;""")
sub("quant/exl3_gemv_int8.cu", "    if (size_m > 2) return false;\n    int M = size_m;",
                               "    if (size_m > exl3_gemv_int8_sq_max_m()) return false;\n    int M = size_m;")
sub("quant/exl3_gemv_int8.cu", "if (size_m <= (residual ? 1 : 2) && exl3_gemv_int8_sq(",
                               "if (size_m <= (residual ? 1 : exl3_gemv_int8_sq_max_m()) && exl3_gemv_int8_sq(")
for k in (4, 5, 6):
    sub(f"quant/comp_units/exl3_gemv_int8_inst_sq_k{k}.cu",
        "        case 2: { SELM_(2) }\n",
        "        case 2: { SELM_(2) }\n"
        f"        case 3: if (residual) return nullptr; return c_fp32 ? (void*) exl3_gemv_int8_sq_kernel<{k}, 3, true, false> : (void*) exl3_gemv_int8_sq_kernel<{k}, 3, false, false>;\n"
        f"        case 4: if (residual) return nullptr; return c_fp32 ? (void*) exl3_gemv_int8_sq_kernel<{k}, 4, true, false> : (void*) exl3_gemv_int8_sq_kernel<{k}, 4, false, false>;\n")
# MoE coop kernels A/B: minimum resident blocks per SM (build arg MOE_COOP_MINB, default 1 = stock). Stock
# ptxas uses 84-87 registers x 512 threads, i.e. one block (16 warps) per SM; 2 caps registers at 64 so two
# blocks fit. Register allocation only: same instructions and arithmetic, bit-identical output.
import os
minb = int(os.environ.get("MOE_COOP_MINB", "1"))
if minb > 1:
    for kern in ("void exl3_moe_coop_a_kernel(const MoeCoopParams p)", "void exl3_moe_coop_b_kernel(const MoeCoopParams p)"):
        sub("quant/exl3_moe_coop_kernel.cuh", f"__global__ __launch_bounds__(THREADS)\n{kern}",
            f"__global__ __launch_bounds__(THREADS, {minb})\n{kern}")
# MoE coop main loop (build arg MOE_COOP_LOOP2=1): prefetch ring 4 -> 2 with the fp16->fp32 fold keyed on the
# global slice index (same fold points as stock), and a guard-free copy of the unrolled loop when the warp's
# slice count is a multiple of the ring depth (true at GLM-5.3 shapes). Same accumulation order: bit-identical.
# Standalone RTX harness: 425.8 -> 404.1 us per MoE layer call.
loop2 = os.environ.get("MOE_COOP_LOOP2", "0") == "1"
if loop2:
    kf = "quant/exl3_moe_coop_kernel.cuh"
    sub(kf, "constexpr int PF = 4;                           // prefetch ring depth",
            "constexpr int PF = 2;                           // prefetch ring depth (hecate: 4 -> 2)")
    sub(kf, "            if ((d + 1) % FOLD == 0 || i + 1 == myn)", "            if ((i + 1) % FOLD == 0 || i + 1 == myn)")
    src = (root / kf).read_text()
    a = src.index("    for (int ib = 0; ib < myn; ib += PF)\n")
    b = src.index("    // Cross-warp reduction over the k splits.")
    loop = src[a:b]
    guard = "            if (i >= myn) break;\n"
    assert loop.count(guard) == 1
    fast = loop.replace(guard, "")
    indent = lambda t: "".join(("    " + l if l.strip() else l) for l in t.splitlines(True))
    new = ("    if (myn % PF == 0)\n    {\n" + indent(fast) + "    }\n    else\n    {\n" + indent(loop) + "    }\n\n")
    (root / kf).write_text(src[:a] + new + src[b:])
# MoE coop affine fold (build arg MOE_COOP_AFFINE=1, needs loop2): 3-bit mul1 tiles feed h = 1024 + byte-sum (an
# exact fp16 pattern) to fp32-accumulating MMAs and apply sum(w x) = k_inv * sum(h x) + k_bias * sum(x) once per
# output: no per-weight HFMA2, no fp16 partial sums / fold. Not bit-identical but closer to an fp32 reference
# (rel 5.9e-4 vs stock 7.1e-4). RTX harness: 425.8 (stock) / 403.7 (loop2) -> 377.9 us per MoE layer call.
affine = os.environ.get("MOE_COOP_AFFINE", "0") == "1"
if affine:
    assert loop2, "MOE_COOP_AFFINE builds on MOE_COOP_LOOP2"
    import pathlib as _pl
    kf = root / "quant/exl3_moe_coop_kernel.cuh"
    _gen = _pl.Path(__file__).with_name("affine_kernel.py")
    exec(compile(_gen.read_text(), str(_gen), "exec"), {"KERNEL_PATH": str(kf)})
print("patched", f"(moe coop min blocks {minb}, loop2 {loop2}, affine {affine})")
