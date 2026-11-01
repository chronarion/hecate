"""hecate: tcgen05 sparse attention for the MLA prefill (hmla.cu). Adds `hmla_attn` to the extension and routes
MLAAttention._attend_sparse (the DSA gather attention of prompt chunks past index_topk tokens) through it when
EXL3_HMLA_PREFILL=1 (default): one sequence, the Q6 latent cache, GLM-5.3's MLA shapes; anything else keeps the Triton
kernel. Decode keeps the Triton kernels.
   python3 apply_hmla.py EXT_DIR SP_DIR SRC_DIR"""
import pathlib, shutil, sys
ext, sp, srcdir = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])


def sub(path, old, new, count=1):
    s = path.read_text()
    assert s.count(old) == count, (str(path), old[:70], s.count(old))
    path.write_text(s.replace(old, new))


shutil.copy(srcdir / "hmla.cu", ext / "hmla.cu")
(ext / "hmla.cuh").write_text('''#pragma once

#include <ATen/Tensor.h>

// hecate: tcgen05 sparse attention for the MLA prefill, see hmla.cu
void hmla_attn(at::Tensor q_lat, at::Tensor q_pe, at::Tensor ckv, at::Tensor ckv_s, at::Tensor kpe, at::Tensor block_table,
               at::Tensor indices, double scale, at::Tensor out);
''')
sub(ext / "bindings.cpp", '#include "hgemm.cuh"\n', '#include "hgemm.cuh"\n#include "hmla.cuh"\n')
sub(ext / "bindings.cpp", '    m.def("exl3_gemm", &exl3_gemm, "exl3_gemm");\n',
    '    m.def("exl3_gemm", &exl3_gemm, "exl3_gemm");\n    m.def("hmla_attn", &hmla_attn, "hmla_attn");\n')

sub(sp / "modules/mla_attn.py", '''        from .attention_fn.dsa_triton import dsa_attn
''', '''        if _HMLA_PREFILL and qc is not None and qc[1] == 6 and bsz == 1 and block_table.shape[0] == 1 and \\
                self.num_q_heads == 64 and self.kv_lora_rank == 512 and self.qk_rope_head_dim == 64:
            # hecate: tcgen05 kernel, one CTA per query row (all 64 heads), the Q6 rows dequantized in shared memory
            from ..ext import exllamav3_ext as _hx
            R_ = bsz * seqlen
            o_lat = torch.empty((self.num_q_heads, R_, self.kv_lora_rank), dtype = torch.half, device = q_lat.device)
            _hx.hmla_attn(q_lat.contiguous(), q_pe.reshape(R_, self.num_q_heads, self.qk_rope_head_dim).contiguous(),
                          ckv_cache, qc[0], kpe_cache, block_table, indices.contiguous(), float(self.sm_scale), o_lat)
            return o_lat
        from .attention_fn.dsa_triton import dsa_attn
''')
sub(sp / "modules/mla_attn.py", "import os\n", 'import os\n_HMLA_PREFILL = os.environ.get("EXL3_HMLA_PREFILL", "1") == "1"   # hecate\n')
print("apply_hmla: ok")
