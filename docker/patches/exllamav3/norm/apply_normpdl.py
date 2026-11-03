"""hecate (EXL3_NORM_PDL): rms_norm_kernel as a programmatic dependent. The kernel waits for its predecessor
(griddepcontrol.wait, a no-op without the launch attribute) before its first load and immediately lets its own dependent
launch (griddepcontrol.launch_dependents), so the int8 GEMV after a norm (the q_a norm before q_b) can become resident and
fill its weight ring while the 4-row norm runs. The launch attribute is set only with EXL3_NORM_PDL=1 (read once at load
time): same-image A/B. Apply after apply_norm2.py.   python3 apply_normpdl.py EXT_DIR"""
import pathlib, sys
ext = pathlib.Path(sys.argv[1])


def sub(path, old, new, count=1):
    s = path.read_text()
    assert s.count(old) == count, (str(path), old[:70], s.count(old))
    path.write_text(s.replace(old, new))


cu = ext / "norm.cu"
sub(cu, """    constexpr bool weight_bf16 = std::is_same_v<weight_t, bfloat16>;
    constexpr bool residual_fp16 = std::is_same_v<residual_t, half>;

    int t = threadIdx.x;""", """    constexpr bool weight_bf16 = std::is_same_v<weight_t, bfloat16>;
    constexpr bool residual_fp16 = std::is_same_v<residual_t, half>;

    // hecate (EXL3_NORM_PDL): wait for the producer of x / r, then let the next kernel start its prologue
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;");

    int t = threadIdx.x;""")
# launch helper before rms_norm_impl
sub(cu, "void rms_norm_impl\n", """// hecate (EXL3_NORM_PDL): launch as a programmatic dependent of the previous kernel in the stream
template <typename Kern, typename... Args>
static void norm_launch(Kern kern, dim3 grid, dim3 block, cudaStream_t stream, Args... args)
{
    static const bool pdl = [] { const char* e = getenv("EXL3_NORM_PDL"); return e && atoi(e) != 0; }();
    if (!pdl) { kern<<<grid, block, 0, stream>>>(args...); return; }
    cudaLaunchConfig_t cfg = {}; cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.gridDim = grid; cfg.blockDim = block; cfg.stream = stream; cfg.attrs = at; cfg.numAttrs = 1;
    cuda_check(cudaLaunchKernelEx(&cfg, kern, args...));
}

void rms_norm_impl
""")
sub(cu, """        if (rows) rms_norm_kernel<_res, __tx, __ty, __tw, __tr><<<gridDim, blockDim, 0, stream>>>  \\
        (                                                                                          \\
            (const __tx*) x.data_ptr(),                                                            \\""",
"""        if (rows) norm_launch(rms_norm_kernel<_res, __tx, __ty, __tw, __tr>, gridDim, blockDim, stream, \\
            (const __tx*) x.data_ptr(),                                                            \\""")
s = cu.read_text()
old = """            w_groups,                                                               \\
            norm2_on                                                                \\
        );                                                                          \\"""
assert s.count(old) == 1, s.count(old)
cu.write_text(s.replace(old, """            w_groups,                                                               \\
            norm2_on                                                                \\
        );                                                                          \\""".replace("        );", "        );")))
print("patched: EXL3_NORM_PDL (rms_norm_kernel)")
