"""hecate: the decode MoE combine fused into the next block's input norm (EXL3_TIER_COMBINE_NORM, runtime/tier/cn.cu).
The MoE layer (tier_split) may hand its final combine to the next TransformerBlock as params["resid_pending_combine"] =
(can_fuse(norm, x), run_combine(), run_fused(x, norm) -> normed input); this patch lets the block take it, and tells the MLP
when deferring is allowed (its output goes to resid_pending).   python3 apply_cn.py SP (the exllamav3 package directory)"""
import sys
from pathlib import Path
p = Path(sys.argv[1]) / "modules/transformer.py"           # SP = the exllamav3 package directory
s = p.read_text()


def sub(a, b):
    global s
    assert s.count(a) == 1, (a[:70], s.count(a)); s = s.replace(a, b)


sub("""        pending = params.pop("resid_pending", None)
""", """        pending = params.pop("resid_pending", None)
        # hecate: the previous MoE layer may leave its combine (partials -> pending) to this block's input norm,
        # as (can_fuse(norm, x), run_combine(), run_fused(x, norm) -> normed input)
        pend_comb = params.pop("resid_pending_combine", None)
""")
sub("""        if pending is not None and not fuse_in:
            x += pending
""", """        if pend_comb is not None and not (fuse_in and pend_comb[0](self.attn_norm, x)):
            pend_comb[1]()
            pend_comb = None
        if pending is not None and not fuse_in:
            x += pending
""")
sub("""            elif fuse_in:
                y = self.attn_norm.forward(pending, params, out_dtype = torch.half, residual_in = x)""",
"""            elif fuse_in and pend_comb is not None:
                y = pend_comb[2](x, self.attn_norm)
            elif fuse_in:
                y = self.attn_norm.forward(pending, params, out_dtype = torch.half, residual_in = x)""")
sub("""                params["residual"] = x
""", """                params["residual"] = x
                # hecate: the MLP may defer its final combine to the next block's input norm only when its output goes
                # to resid_pending below
                params["mlp_defer_ok"] = (
                    not self.mlp_post_norm and self.mlp_resid_scalar is None and bool(params.get("resid_defer")) and
                    not export_state and self.layer_scalar_f is None and (out_dtype or self.out_dtype) in (None, x.dtype)
                )
""")
sub("""            y = self.mlp.forward(y, params)
            if self.mlp_resid_scalar is not None:""", """            y = self.mlp.forward(y, params)
            params.pop("mlp_defer_ok", None)
            if self.mlp_resid_scalar is not None:""")
p.write_text(s)
print("apply_cn: ok")
