"""Fused MLA cache append (ext.mla_kvq) + absorb-rope kernel vs the unfused chain the BC graph runs: staging kernel ->
rope (q_pe heads + key) -> quant_cache_cont -> quant scatter, and the plain absorb kernel. GLM-5.3 shapes (64 heads,
latent 512, rope 64, nope 192, Q6 cache, page 256). Checks the cache words / scales / rope keys at the appended slots,
that no other slot is touched, q_pe and q_lat; prints mismatch rates and the dequantized latent error, then times the
fused kernel against the four it replaces. In the image, on the GB300 (the server stopped):
   docker run --rm --device nvidia.com/gpu=<GB300 UUID> -v $PWD/tools/dev:/w --entrypoint python hecate:latest -I /w/test_kvq.py"""
import torch, triton
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.attention_fn.mla_triton import (_mla_stage_kernel, _mla_absorb_kernel, _mla_absorb_rope_kernel,
                                                       mla_kv_quant_append)
dev = torch.device("cuda:0")
torch.manual_seed(0)
H, D_nope, D_r, D_c = 64, 192, 64, 512
QK = D_nope + D_r
PAGE, BITS = 256, 6
NG = D_c // 32
W = NG * BITS
CKV_W = D_c + D_r
EPS = 1e-5
ok = True
for bsz, q_len, use_positions, style, attn_factor in ((1, 4, True, 1, 1.0), (1, 1, False, 1, 1.0), (2, 4, True, 1, 0.8),
                                                     (1, 8, True, 1, 1.0), (1, 4, True, 2, 1.0)):
    R = bsz * q_len
    pages_per_seq, pages = 6, 16
    block_table = torch.stack([torch.randperm(pages)[:pages_per_seq] for _ in range(bsz)]).int().to(dev)
    seqlens = torch.randint(0, pages_per_seq * PAGE - q_len, (bsz,), dtype=torch.int32)
    seqlens[0] = PAGE - 2 if q_len >= 4 else seqlens[0]          # make one chunk straddle a page boundary
    cache_seqlens = seqlens.to(dev)
    position = int(seqlens[0]) if not use_positions else 0
    if not use_positions and bsz > 1: continue
    positions = cache_seqlens.clone() if use_positions else None
    ckv_kpe = (torch.randn(R, CKV_W, device=dev) * 1.5).half()
    kv_norm_w = (torch.rand(D_c, device=dev) + 0.5).half()
    inv_freq = (1.0 / (10000.0 ** (torch.arange(0, D_r, 2, device=dev).float() / D_r))).contiguous()
    q_full = (torch.randn(R, H * QK, device=dev) * 0.5).half()
    w_uk = (torch.randn(D_c, H * D_nope, device=dev) * 0.05).half()

    # reference chain
    q_pe = torch.empty(R, H, D_r, dtype=torch.half, device=dev)
    ckv = torch.empty(R, D_c, dtype=torch.half, device=dev)
    kpe = torch.empty(R, D_r, dtype=torch.half, device=dev)
    _mla_stage_kernel[(R,)](q_full, ckv_kpe, kv_norm_w, q_pe, ckv, kpe, eps=EPS, n_q_heads=H, QK_DIM=QK, Q_STRIDE=H * QK,
                            CKV_STRIDE=CKV_W, D_nope=D_nope, D_c=D_c, D_r=D_r, num_warps=4, num_stages=2)
    q_pe4 = q_pe.view(bsz, q_len, H, D_r)
    kpe4 = kpe.view(bsz, q_len, 1, D_r)
    ext.rope(q_pe4, q_pe4, kpe4, kpe4, inv_freq, position, positions, None, style, attn_factor, None, None, EPS, 0.0, 0.0,
             0, 1, 0)
    qk_r = torch.zeros(pages, PAGE, W, dtype=torch.int32, device=dev)
    sk_r = torch.zeros(pages, PAGE, NG, dtype=torch.half, device=dev)
    kc_r = torch.zeros(pages, PAGE, 1, D_r, dtype=torch.half, device=dev)
    mla_kv_quant_append(ckv.view(bsz, q_len, D_c), kpe.view(bsz, q_len, D_r), qk_r, sk_r, kc_r, block_table, cache_seqlens,
                        BITS)
    q_lat_r = torch.empty(H, R, D_c, dtype=torch.half, device=dev)
    BM = min(64, triton.next_power_of_2(max(R, 16)))
    _mla_absorb_kernel[(triton.cdiv(R, BM), D_c // 128, H)](q_full, w_uk, q_lat_r, R, n_q_heads=H, QK_DIM=QK,
        Q_STRIDE=H * QK, D_nope=D_nope, NOPE_PAD=256, D_c=D_c, BLOCK_M=BM, BLOCK_N=128, num_warps=4, num_stages=2)

    # fused
    qk_f = torch.zeros_like(qk_r); sk_f = torch.zeros_like(sk_r); kc_f = torch.zeros_like(kc_r)
    sincos = torch.empty(R, D_r, dtype=torch.float, device=dev)
    ext.mla_kvq(ckv_kpe, kv_norm_w, inv_freq, position, positions, style, attn_factor, EPS, block_table, cache_seqlens,
                qk_f, sk_f, kc_f, sincos, q_len)
    q_lat_f = torch.empty_like(q_lat_r)
    q_pe_f = torch.empty_like(q_pe)
    _mla_absorb_rope_kernel[(triton.cdiv(R, BM), D_c // 128, H)](q_full, w_uk, q_lat_f, R, q_pe_f, sincos, n_q_heads=H,
        QK_DIM=QK, Q_STRIDE=H * QK, D_nope=D_nope, NOPE_PAD=256, D_c=D_c, BLOCK_M=BM, BLOCK_N=128, D_r=D_r,
        ROPE_GPTJ=int(style == 1), num_warps=4, num_stages=2)
    torch.cuda.synchronize()

    # appended slots
    toks = []
    for b in range(bsz):
        for p in range(q_len):
            a = int(seqlens[b]) + p
            toks.append(int(block_table[b, a // PAGE]) * PAGE + a % PAGE)
    toks = torch.tensor(toks, device=dev)
    mask = torch.zeros(pages * PAGE, dtype=torch.bool, device=dev); mask[toks] = True
    stray = (qk_f.view(-1, W)[~mask] != 0).any().item() or (sk_f.view(-1, NG)[~mask] != 0).any().item() or \
            (kc_f.view(-1, D_r)[~mask] != 0).any().item()
    wq_r, wq_f = qk_r.view(-1, W)[toks], qk_f.view(-1, W)[toks]
    sc_r, sc_f = sk_r.view(-1, NG)[toks], sk_f.view(-1, NG)[toks]
    kp_r, kp_f = kc_r.view(-1, D_r)[toks], kc_f.view(-1, D_r)[toks]
    dq_r = torch.empty(R, D_c, dtype=torch.half, device=dev); dq_f = torch.empty_like(dq_r)
    ext.dequant_cache_cont(wq_r.contiguous(), sc_r.contiguous(), dq_r, 0.0)
    ext.dequant_cache_cont(wq_f.contiguous(), sc_f.contiguous(), dq_f, 0.0)
    word_mis = (wq_r != wq_f).float().mean().item()
    scale_mis = (sc_r != sc_f).float().mean().item()
    kpe_d = (kp_r.float() - kp_f.float()).abs().max().item()
    qpe_d = (q_pe.float() - q_pe_f.float()).abs().max().item()
    qpe_mis = (q_pe != q_pe_f).float().mean().item()
    qlat_eq = torch.equal(q_lat_r, q_lat_f)
    dq_err = (dq_r.float() - dq_f.float()).abs().max().item()
    ref_err = (dq_r.float() - ckv.float()).abs().max().item()     # the quantizer's own error, for scale
    good = (not stray) and word_mis < 0.01 and scale_mis < 0.01 and kpe_d <= 2e-3 and qpe_d <= 2e-3 and qlat_eq and \
           dq_err <= ref_err and torch.isfinite(q_pe_f).all().item()
    ok &= good
    print(f"bsz {bsz} q_len {q_len} positions {use_positions} style {style} af {attn_factor}: words differ {word_mis:.4f}, "
          f"scales {scale_mis:.4f}, kpe max |d| {kpe_d:.1e}, q_pe max |d| {qpe_d:.1e} (differ {qpe_mis:.4f}), q_lat equal "
          f"{qlat_eq}, dequant max |d| {dq_err:.3e} (quantizer error {ref_err:.3e}), stray writes {stray} -> "
          f"{'PASS' if good else 'FAIL'}", flush=True)

# timing (CUDA graph of 20 launches): fused vs stage + rope + quant + scatter at R = 4
R, bsz, q_len = 4, 1, 4
g1, g2 = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
s = torch.cuda.Stream()
with torch.cuda.stream(s):
    for _ in range(2):
        ext.mla_kvq(ckv_kpe[:R], kv_norm_w, inv_freq, 0, cache_seqlens[:1].clone(), 1, 1.0, EPS, block_table[:1], cache_seqlens[:1],
                    qk_f, sk_f, kc_f, sincos, q_len)
    torch.cuda.synchronize()
    pos1 = cache_seqlens[:1].clone()
    with torch.cuda.graph(g1, stream=s):
        for _ in range(20):
            ext.mla_kvq(ckv_kpe[:R], kv_norm_w, inv_freq, 0, pos1, 1, 1.0, EPS, block_table[:1], cache_seqlens[:1],
                        qk_f, sk_f, kc_f, sincos, q_len)
    with torch.cuda.graph(g2, stream=s):
        for _ in range(20):
            _mla_stage_kernel[(R,)](q_full[:R], ckv_kpe[:R], kv_norm_w, q_pe[:R], ckv[:R], kpe[:R], eps=EPS, n_q_heads=H,
                                    QK_DIM=QK, Q_STRIDE=H * QK, CKV_STRIDE=CKV_W, D_nope=D_nope, D_c=D_c, D_r=D_r,
                                    num_warps=4, num_stages=2)
            ext.rope(q_pe[:R].view(1, q_len, H, D_r), q_pe[:R].view(1, q_len, H, D_r), kpe[:R].view(1, q_len, 1, D_r),
                     kpe[:R].view(1, q_len, 1, D_r), inv_freq, 0, pos1, None, 1, 1.0, None, None, EPS, 0.0, 0.0, 0, 1, 0)
            mla_kv_quant_append(ckv[:R].view(1, q_len, D_c), kpe[:R].view(1, q_len, D_r), qk_r, sk_r, kc_r, block_table[:1],
                                cache_seqlens[:1], BITS)
for g, name in ((g1, "fused"), (g2, "unfused chain")):
    g.replay(); torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(10): g.replay()
    e1.record(); torch.cuda.synchronize()
    print(f"{name}: {e0.elapsed_time(e1) / 200 * 1e3:.2f} us per call", flush=True)
print("ALL PASS" if ok else "SOME FAILED")
