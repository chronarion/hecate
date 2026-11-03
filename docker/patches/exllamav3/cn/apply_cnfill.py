"""hecate (EXL3_CN_PDL): the tier decode combine (runtime/tier/cn.cu, combine_norm_pdl) reads no routed sum from the MoE
layer whose experts all live in the bands, so the empty-slice branch need not zero-fill one; the layer opts in per call
through self.tier_nofill(cpu_pending), and the tier's other combine paths zero it before use.
   python3 apply_cnfill.py SP (the exllamav3 package directory)"""
import sys
from pathlib import Path
p = Path(sys.argv[1]) / "modules/block_sparse_mlp.py"
s = p.read_text()
a = """        elif self.intermediate_size == 0 or self.num_local_experts == 0:
            final_hidden_states = torch.zeros(eshape, dtype = torch.float, device = y.device)
"""
b = """        elif self.intermediate_size == 0 or self.num_local_experts == 0:
            nofill = getattr(self, "tier_nofill", None)
            final_hidden_states = (torch.empty if nofill is not None and nofill(cpu_pending) else torch.zeros)(
                eshape, dtype = torch.float, device = y.device)
"""
assert s.count(a) == 1, s.count(a)
p.write_text(s.replace(a, b))
print("apply_cnfill: ok")
