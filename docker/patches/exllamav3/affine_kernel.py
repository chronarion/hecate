"""MoE coop affine fold (called by apply_cpp.py with KERNEL_PATH set; the kernel must already have loop2).
3-bit mul1 tiles: h = 1024 + byte-sum to fp32-accumulating MMAs, sum(w x) = k_inv*sum(h x) + k_bias*sum(x)."""
import re
src = open(KERNEL_PATH).read()


def sub(old, new, n=1):
    global src
    assert src.count(old) == n, (old[:70], src.count(old))
    src = src.replace(old, new)


def rsub(pat, fn, n):
    global src
    src, k = re.subn(pat, fn, src)
    assert k == n, (pat[:60], k)


HELPERS = """// ---------------------------------------------------------------------------------------------
// hecate AFFINE: mul1 decode without the per-weight affine. h = 1024 + byte-sum is an exact fp16 pattern;
// sum(w x) = k_inv * sum(h x) + k_bias * sum(x) is applied once per output, with fp32 MMA accumulation
#ifndef MOE_COOP_AFFINE
#define MOE_COOP_AFFINE 1
#endif
__device__ __forceinline__ half2 decode_pair_mul1_raw(uint32_t x0, uint32_t x1)
{
    x0 *= 0x83DCD12Du;
    x1 *= 0x83DCD12Du;
    const uint32_t s0 = __dp4a(x0, 0x01010101u, 0x6400u);
    const uint32_t s1 = __dp4a(x1, 0x01010101u, 0x6400u);
    return __halves2half2(__ushort_as_half((unsigned short) s0), __ushort_as_half((unsigned short) s1));
}
__device__ __forceinline__ void decode8_raw(uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3,
    uint32_t w4, uint32_t w5, uint32_t w6, uint32_t w7, FragB& f0, FragB& f1)
{
    f0[0] = decode_pair_mul1_raw(w0, w1); f0[1] = decode_pair_mul1_raw(w2, w3);
    f1[0] = decode_pair_mul1_raw(w4, w5); f1[1] = decode_pair_mul1_raw(w6, w7);
}
__device__ __forceinline__ void dq8_regs_3bits_raw(uint32_t a, uint32_t b, int s2, FragB& f0, FragB& f1)
{
    uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
    w7 = fshift(b, a, s2); w6 = w7 >> 3; w5 = w6 >> 3; w4 = w5 >> 3;
    w3 = fshift(b, a, s2 + 12); w2 = w3 >> 3; w1 = w2 >> 3; w0 = w1 >> 3;
    decode8_raw(w0 & 0xffff, w1 & 0xffff, w2 & 0xffff, w3 & 0xffff, w4 & 0xffff, w5 & 0xffff, w6 & 0xffff, w7 & 0xffff, f0, f1);
}

"""
sub("// ---------------------------------------------------------------------------------------------\n// Elementwise helpers",
    HELPERS + "// ---------------------------------------------------------------------------------------------\n// Elementwise helpers")
i = src.index("    constexpr int TCOLS = tile_cols<WIDE>();\n")
src = (src[:i] + "    constexpr int TCOLS = tile_cols<WIDE>();\n    constexpr bool AFFINE = MOE_COOP_AFFINE && REG && bits == 3 && cb == 2;\n"
       + src[i + len("    constexpr int TCOLS = tile_cols<WIDE>();\n"):])
sub("    FragC_h ch[WNT][2] = {};\n    float2 acc0[WNT][2] = {};\n",
    "    FragC_h ch[WNT][2] = {};\n    float2 acc0[WNT][2] = {};\n    [[maybe_unused]] FragC cf[WNT][2] = {};\n    [[maybe_unused]] float xs = 0.0f;\n")
rsub(r"\n( +)a01\[1\] = hzero;\n +a23\[1\] = hzero;\n",
     lambda m: (f"\n{m.group(1)}a01[1] = hzero;\n{m.group(1)}a23[1] = hzero;\n{m.group(1)}if constexpr (AFFINE)\n{m.group(1)}{{\n"
                f"{m.group(1)}    const float2 s0 = __half22float2(a01[0]), s1 = __half22float2(a23[0]);\n"
                f"{m.group(1)}    xs += (s0.x + s0.y) + (s1.x + s1.y);\n{m.group(1)}}}\n"), 2)
rsub(r"\n( +)exl3_gemv_ns::dq8_regs_3bits<cb>\(awv, bwv, x_s2, f0, f1\);\n",
     lambda m: (f"\n{m.group(1)}if constexpr (AFFINE) dq8_regs_3bits_raw(awv, bwv, x_s2, f0, f1);\n"
                f"{m.group(1)}else exl3_gemv_ns::dq8_regs_3bits<cb>(awv, bwv, x_s2, f0, f1);\n"), 2)
rsub(r"\n( +)exl3_gemv_ns::mma_ab_h\(a01, a23, f0, ch\[t\]\[0\]\);\n +exl3_gemv_ns::mma_ab_h\(a01, a23, f1, ch\[t\]\[1\]\);\n",
     lambda m: (f"\n{m.group(1)}if constexpr (AFFINE)\n{m.group(1)}{{\n{m.group(1)}    FragA af;\n"
                f"{m.group(1)}    af[0] = a01[0]; af[1] = a01[1]; af[2] = a23[0]; af[3] = a23[1];\n"
                f"{m.group(1)}    ptx_mma_m16n8k16(af, f0, cf[t][0]);\n{m.group(1)}    ptx_mma_m16n8k16(af, f1, cf[t][1]);\n{m.group(1)}}}\n"
                f"{m.group(1)}else\n{m.group(1)}{{\n{m.group(1)}    exl3_gemv_ns::mma_ab_h(a01, a23, f0, ch[t][0]);\n"
                f"{m.group(1)}    exl3_gemv_ns::mma_ab_h(a01, a23, f1, ch[t][1]);\n{m.group(1)}}}\n"), 2)
rsub(r"if \(\(i \+ 1\) % FOLD == 0 \|\| i \+ 1 == myn\)", lambda m: "if (!AFFINE && ((i + 1) % FOLD == 0 || i + 1 == myn))", 2)
sub("    // Cross-warp reduction over the k splits.",
"""    if constexpr (AFFINE)
    {
        // Full row sum of x over this warp's k range (the 4 lanes of a row hold disjoint k pairs)
        xs += __shfl_xor_sync(0xffffffffu, xs, 1);
        xs += __shfl_xor_sync(0xffffffffu, xs, 2);
        const float k_inv = __half2float(__ushort_as_half((unsigned short) 0x1eee));
        const float k_bias = __half2float(__ushort_as_half((unsigned short) 0xc931));
        #pragma unroll
        for (int t = 0; t < WNT; ++t)
            #pragma unroll
            for (int f = 0; f < 2; ++f)
            {
                acc0[t][f].x = fmaf(k_inv, cf[t][f][0], k_bias * xs);
                acc0[t][f].y = fmaf(k_inv, cf[t][f][1], k_bias * xs);
            }
    }

    // Cross-warp reduction over the k splits.""")
open(KERNEL_PATH, "w").write(src)
