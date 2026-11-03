"""hecate: the prefill DSA indexer scoring on the tcgen05 kernel (runtime/core/hidx.cu, JIT-built into EXL3_TIER_BUILD):
MLAAttention._indexer_topk's tile_scores calls hidx_scores instead of the Triton dsa_indexer_scores when the shapes match
(32 heads x 128 dims, a score backing, contiguous keys or a paged plane with 128-multiple pages). Bit-exact scores.
EXL3_HIDX=0 keeps the Triton kernel.   python3 apply_hidx.py MLA_ATTN_PY [HIDX_CU=/opt/hecate/core/hidx.cu]"""
import sys
p = sys.argv[1]; cu = sys.argv[2] if len(sys.argv) > 2 else "/opt/hecate/core/hidx.cu"
s = open(p).read()


def rep(a, b, cnt=1):
    global s
    assert s.count(a) == cnt, (a[:80], s.count(a)); s = s.replace(a, b)


rep("_score_tile = int(os.environ.get(\"EXL3_DSA_SCORE_TILE\", 32768))\n", f'''_score_tile = int(os.environ.get("EXL3_DSA_SCORE_TILE", 32768))
# hecate: tcgen05 prefill indexer scoring (hidx.cu), bit-exact vs dsa_indexer_scores
_HIDX = os.environ.get("EXL3_HIDX", "1") == "1"
_hidx_mod = None


def _hidx():
    global _hidx_mod
    if _hidx_mod is None:
        from torch.utils.cpp_extension import load as _load
        bd = os.path.join(os.environ.get("EXL3_TIER_BUILD", "/tmp"), "build_hidx"); os.makedirs(bd, exist_ok = True)
        _hidx_mod = _load(name = "hidx", sources = ["{cu}"], build_directory = bd, verbose = False,
                          extra_cuda_cflags = ["-O3", "-gencode=arch=compute_103a,code=sm_103a"])
    return _hidx_mod
''')
rep("""                def tile_scores(t0, t1, scores_out = None):
                    if k_idx_chunk is not None:
                        return dsa_indexer_scores(
                            q_idx[b, r0:r1], w[b, r0:r1], k_idx_chunk[b][t0:t1],
                            pos0 - t0, 1, t1 - t0, scores = scores_out,
                        )
""", """                def tile_scores(t0, t1, scores_out = None):
                    if _HIDX and scores_out is not None and H_i == 32 and D_i == 128 and \\
                            (k_idx_chunk is not None or idx_pool.shape[1] % 128 == 0):
                        # hecate: tcgen05 scoring, same scores as dsa_indexer_scores (bit-exact)
                        if k_idx_chunk is not None:
                            kk, bt_, epp_ = k_idx_chunk[b][t0:t1].contiguous(), None, 0
                        else:
                            epp_ = idx_pool.shape[1]; bt_ = block_table[b]
                            if t0:
                                bt_ = bt_[t0 // epp_ : -(-t1 // epp_)]
                            kk, bt_ = idx_pool.view(-1, D_i), bt_.contiguous()
                        _hidx().hidx_scores(q_idx[b, r0:r1].contiguous(), w[b, r0:r1].contiguous(), kk, pos0 - t0, t1 - t0,
                                            D_i ** -0.5 * H_i ** -0.5, scores_out, 64, bt_, epp_, t1 - t0)
                        return scores_out[:, :t1 - t0]
                    if k_idx_chunk is not None:
                        return dsa_indexer_scores(
                            q_idx[b, r0:r1], w[b, r0:r1], k_idx_chunk[b][t0:t1],
                            pos0 - t0, 1, t1 - t0, scores = scores_out,
                        )
""")
open(p, "w").write(s)
print("hecate: hidx prefill indexer scoring applied")
