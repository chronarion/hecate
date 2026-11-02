"""hecate (norm2): RMSNorm rows of up to 2 x blockDim float4 columns (dim 6144 at 1024 threads: the input and post-attention
norms, both with the fused residual add) keep their values in registers instead of re-reading the residual after the
reduction, and every input / residual / weight load is issued before the first store and the reduction. Same column
ownership (columns t and t + blockDim), same summation order and arithmetic as the loop it replaces: bit-identical.
RES_POST keeps the original path (it reads y before writing it). EXL3_NORM2=0 at load time selects the original paths
(same-image A/B).
   python3 apply_norm2.py EXT_DIR"""
import pathlib, sys
ext = pathlib.Path(sys.argv[1])


def sub(path, old, new, count=1):
    s = path.read_text()
    assert s.count(old) == count, (str(path), old[:70], s.count(old))
    path.write_text(s.replace(old, new))


cu = ext / "norm.cu"
sub(cu, """    if (single)
    {
        // One float4 per thread: keep the value in a register between the two phases
        float4 x4 = {};""", """    // hecate (norm2): up to two float4 columns per thread (t, t + blockDim) in registers, all loads first
    if (norm2 && res_mode != RES_POST && columns <= 2 * (int) blockDim.x)
    {
        const int ca = t, cb = t + blockDim.x;
        const bool ha = ca < columns, hb = cb < columns;
        float4 xa = {}, xb = {}, ra = {}, rb = {}, wa = {}, wb = {};
        auto load_w = [&] (float4& w4, int column)
        {
            if constexpr (weight_bf16) read_bfloat164   (w4, ((const bfloat164*) w) + column);
            else                       read_half4<false>(w4, ((const half4*)     w) + column);
        };
        auto load_r = [&] (float4& r4, int column)
        {
            if constexpr (residual_fp16) read_half4<false>(r4, ((const half4*) (r + row_off)) + column);
            else                         read_float4(r4, ((const float4*) (r + row_off)) + column);
        };
        // r += x, rounded to the residual dtype (as add_resid_in)
        auto add_store_r = [&] (float4& x4, const float4& r4, int column)
        {
            x4.x += r4.x;
            x4.y += r4.y;
            x4.z += r4.z;
            x4.w += r4.w;
            if constexpr (residual_fp16)
            {
                half4 h4
                (
                    __halves2half2(__float2half_rn(x4.x), __float2half_rn(x4.y)),
                    __halves2half2(__float2half_rn(x4.z), __float2half_rn(x4.w))
                );
                WRITE64(((half4*) (r + row_off)) + column, h4);
                x4.x = LOW_TO_FLOAT(h4.x);
                x4.y = HIGH_TO_FLOAT(h4.x);
                x4.z = LOW_TO_FLOAT(h4.y);
                x4.w = HIGH_TO_FLOAT(h4.y);
            }
            else
                write_float4(x4, ((float4*) (r + row_off)) + column);
        };
        auto store_out = [&] (float4& x4, const float4& w4c, int column, float rmf)
        {
            if (w)
            {
                float4 w4 = w4c;
                if (constant_bias != 0.0f)
                {
                    w4.x += constant_bias;
                    w4.y += constant_bias;
                    w4.z += constant_bias;
                    w4.w += constant_bias;
                }
                apply4(x4, w4, rmf);
            }
            else
                apply4_nw(x4, rmf);
            if constexpr (output_fp16) write_half4(x4, ((half4*) (y + row_off)) + column);
            if constexpr (output_fp32) write_float4(x4, ((float4*) (y + row_off)) + column);
        };
        if (ha) read_in(xa, x + row_off + 4 * ca);
        if (hb) read_in(xb, x + row_off + 4 * cb);
        if constexpr (res_mode == RES_IN)
        {
            if (ha) load_r(ra, ca);
            if (hb) load_r(rb, cb);
        }
        if (w)
        {
            if (ha) load_w(wa, ca);
            if (hb) load_w(wb, cb);
        }
        float sum = 0.0f;
        if (ha)
        {
            if constexpr (res_mode == RES_IN) add_store_r(xa, ra, ca);
            sum = sum_sq4(sum, xa);
        }
        if (hb)
        {
            if constexpr (res_mode == RES_IN) add_store_r(xb, rb, cb);
            sum = sum_sq4(sum, xb);
        }
        sum = reduce_dyn(sum, warp_id, lane_id);
        float rmf = rsqrtf(sum / (float) dim + epsilon) * constant_scale;
        if (ha) store_out(xa, wa, ca, rmf);
        if (hb) store_out(xb, wb, cb, rmf);
    }
    else if (single)
    {
        // One float4 per thread: keep the value in a register between the two phases
        float4 x4 = {};""")
sub(cu, """    const float constant_scale,
    const int w_groups          // weight spans w_groups rows, cycled by row index (grouped norm)
)
{
    constexpr bool input_fp32 = std::is_same_v<input_t, float>;""", """    const float constant_scale,
    const int w_groups,         // weight spans w_groups rows, cycled by row index (grouped norm)
    const int norm2             // hecate: register path for <= 2 x blockDim float4 columns (EXL3_NORM2, default on)
)
{
    constexpr bool input_fp32 = std::is_same_v<input_t, float>;""")
sub(cu, """            constant_scale,                                                         \\
            w_groups                                                                \\
        );                                                                          \\""", """            constant_scale,                                                         \\
            w_groups,                                                               \\
            norm2_on                                                                \\
        );                                                                          \\""")
sub(cu, """    auto tr = res_mode == RES_IN ? r.value().scalar_type() : tx;
""", """    auto tr = res_mode == RES_IN ? r.value().scalar_type() : tx;
    static const int norm2_on = [] { const char* e = getenv("EXL3_NORM2"); return e && e[0] == '0' ? 0 : 1; }();
""")
print("apply_norm2: ok")
