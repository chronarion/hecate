"""hecate: multi-block radix select for the DSA indexer top-k (exllamav3_ext/dsa_topk.cu). Few-row selections over long
rows (decode verify rows at long context: R <= 16, row width >= EXL3_DSA_TOPK_RADIX, default 16384; 0 = off) run as three
kernels over (4096-key slice, row) CTAs (topk_radix_kernels.cuh) instead of one 1024-thread block per row; the output is
the single-block kernel's exactly (keys > v* in ascending index order, then ties, then -1 padding). Per-row live widths
from the device bound (t_ptr / t_seq) or the graph-patched T, as dsa_topk_kernel.
   python3 apply_topk_radix.py EXT_DIR SRC_DIR"""
import pathlib, shutil, sys
ext, srcdir = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])


def sub(path, old, new, count=1):
    s = path.read_text()
    assert s.count(old) == count, (str(path), old[:70], s.count(old))
    path.write_text(s.replace(old, new))


shutil.copy(srcdir / "topk_radix_kernels.cuh", ext / "topk_radix_kernels.cuh")
f = ext / "dsa_topk.cu"
sub(f, "template <bool VEC>\n__global__ __launch_bounds__(TOPK_THREADS)\nvoid dsa_topk_kernel\n",
    '#include "topk_radix_kernels.cuh"   // hecate\n\ntemplate <bool VEC>\n__global__ __launch_bounds__(TOPK_THREADS)\nvoid dsa_topk_kernel\n')
sub(f, "void dsa_topk_prepare(const at::Tensor& scores_static, int k_pad)\n{\n", '''// hecate: the radix select's workspace (per-slice histograms, suffix sums and counts, per-row threshold bucket) for up to
// RADIX_MAX_R rows x RMAXG slices, allocated once per device like the split workspace
#define RADIX_MAX_R 16
static int dsa_topk_radix_min()
{
    static const int v = [](){ const char* e = getenv("EXL3_DSA_TOPK_RADIX"); return e ? atoi(e) : 16384; }();
    return v;
}
static at::Tensor& dsa_topk_radix_ws(int device)
{
    static std::mutex mtx;
    static auto* ws = new std::map<int, at::Tensor>();
    std::lock_guard<std::mutex> lock(mtx);
    auto it = ws->find(device);
    if (it == ws->end())
    {
        at::Tensor t = at::empty
        (
            {(int64_t) RADIX_MAX_R * (3 * hecate_radix::RMAXG * 256 + hecate_radix::RMAXG + 2)},
            at::TensorOptions().dtype(at::kInt).device(at::Device(at::kCUDA, device))
        );
        it = ws->emplace(device, std::move(t)).first;
    }
    return it->second;
}

void dsa_topk_prepare(const at::Tensor& scores_static, int k_pad)
{
    if (dsa_topk_radix_min() > 0 && scores_static.dim() == 2 && scores_static.size(1) >= dsa_topk_radix_min())
        (void) dsa_topk_radix_ws(scores_static.device().index());
''')
sub(f, '''    // Few-row, long-scan selections are parallelism-starved in the single-block kernel;
''', '''    // hecate: few-row selections over long rows as a multi-block radix select (same output as the single-block kernel)
    // (prefill slabs of up to 256 rows run eagerly: their workspace is a per-call allocation)
    if (dsa_topk_radix_min() > 0 && vec && (R <= RADIX_MAX_R || (!graph && R <= 256)) && T >= dsa_topk_radix_min() &&
        T <= hecate_radix::RS * hecate_radix::RMAXG)
    {
        const int G = (T + hecate_radix::RS - 1) / hecate_radix::RS;
        at::Tensor ws_big;
        if (R > RADIX_MAX_R)
            ws_big = at::empty({(int64_t) R * (3 * G * 256 + G + 2)}, at::TensorOptions().dtype(at::kInt).device(scores.device()));
        int* w = R > RADIX_MAX_R ? (int*) ws_big.data_ptr() : (int*) dsa_topk_radix_ws(scores.device().index()).data_ptr();
        const size_t rg = (size_t) R * G;
        hecate_radix::Ws ws{w, w + rg * 256, w + 2 * rg * 256, w + 3 * rg * 256, w + 3 * rg * 256 + rg};
        const dim3 grid_r(G, R);
        const half* sp = (const half*) scores.data_ptr();
        const int ss = (int) scores.stride(0);
        hecate_radix::radix_a<<<grid_r, hecate_radix::RT, 0, stream>>>(sp, T, ss, t_ptr_, t_seq, ws, G, (int*) indices.data_ptr(), k, k_pad);
        cuda_check(cudaPeekAtLastError());
        hecate_radix::radix_c<<<grid_r, hecate_radix::RT, 0, stream>>>(sp, T, ss, t_ptr_, t_seq, ws, G, k);
        cuda_check(cudaPeekAtLastError());
        hecate_radix::radix_e<<<grid_r, hecate_radix::RT, 0, stream>>>(sp, (int*) indices.data_ptr(), T, ss, k, k_pad, t_ptr_, t_seq, ws, G);
        cuda_check(cudaPeekAtLastError());
        if (graph && !t_ptr_)
        {
            graph->record_param((void*) hecate_radix::radix_a, GP_dsa_T, 1, 4);
            graph->record_param((void*) hecate_radix::radix_a, GP_end, 0);
            graph->record_param((void*) hecate_radix::radix_c, GP_dsa_T, 1, 4);
            graph->record_param((void*) hecate_radix::radix_c, GP_end, 0);
            graph->record_param((void*) hecate_radix::radix_e, GP_dsa_T, 2, 4);
            graph->record_param((void*) hecate_radix::radix_e, GP_end, 0);
        }
        return;
    }

    // Few-row, long-scan selections are parallelism-starved in the single-block kernel;
''')
print("hecate: radix DSA top-k applied")
