"""EXL3_VERIFY_GRAPH=1: the target model's decode-verify forward (one sequence, 2..8 rows, dense attention regime) replayed
from one CUDA graph per (rows, block-table width) instead of ~1900 host launches (host enqueue ~19 ms
per step, the co-limiter with the GPU's ~20 ms). A module graph launch cannot nest inside the capture: with an image that
has exllamav3_ext.graphs_force_eager (docker/patches/exllamav3/vg_eager.py) the BC modules run their C++ launch sequences eagerly only while capturing
and keep their graphs everywhere else (MTP drafts, prefill, eager steps); older images need EXL3_GRAPHS=0.

What may change from step to step and how the captured graph sees it:
  - input ids: copied into a static device buffer before each replay (the forward runs on that buffer)
  - block table / cache seqlens / positions: the generator uploads them into its per-shape device arena before the
    forward; the graph bakes the arena's addresses, checked before every replay (a new arena = recapture)
  - MLA split count / length and page count: functions of the block-table width only, hence one graph per width (the
    width moves every 256 tokens); sparse (DSA) regime steps, where host scalars change per step, run eagerly
  - RTX band: its own graph, launched here before each replay (tier_split skips the launch while capturing)
  - Grace band: per-layer generations from a reserved range while capturing (the pool only needs consecutive jobs to
    differ); the GB300's CPU_DONE wait is >= g and resets the flag to 0 (tier_split._cpu_partial), so a stale larger
    generation can never satisfy it
  - outputs: the logits tensor and the MTP export states are the captured static tensors (states cloned per step)
The first step of a new key runs eagerly (warms workspaces and caches for that shape), the second captures and replays.
Pipelined MTP step (EXL3_PRELAUNCH, tools/dev/prelaunch_patch.py): verify_prelaunch replays an existing verify graph on
drafted ids that are still on the device, and draft_graph(defer=True) replays an existing draft graph with an async
readback, so the generator can launch a step before it finishes the previous step's bookkeeping."""
import collections
import ctypes
import os
import sys
import threading
import time
import torch

ON = os.environ.get("EXL3_VERIFY_GRAPH", "0") == "1"
MAXCTX = int(os.environ.get("EXL3_VERIFY_GRAPH_MAXCTX", "0") or 0)   # dense-regime bound (0: the model's index_topk)
# EXL3_VERIFY_GRAPH_SPARSE=1 (needs EXL3_DSA_MULTIROW=1): sparse (DSA) regime steps are captured too, keyed by regime
SPARSE = os.environ.get("EXL3_VERIFY_GRAPH_SPARSE", "0") == "1"
EVERY = int(os.environ.get("EXL3_VERIFY_GRAPH_EVERY", "2000"))
# EXL3_VERIFY_GRAPH_CHECK=N: every Nth replay, rerun the step eagerly (it rewrites the same KV positions with the same
# values) and compare its logits with the replay's (KL of the softmaxes, top-1 agreement); diagnostic, costs a forward
CHECK = int(os.environ.get("EXL3_VERIFY_GRAPH_CHECK", "0") or 0)
# EXL3_DRAFT_GRAPH=1: the MTP draft passes of a decode step (dense regime, one sequence) replay as ONE CUDA graph per (passes,
# gap rows, block-table width): ids / hidden state / block table / base seqlen copied into static device buffers before the
# replay, per-pass seqlens derived on the device inside the graph (pass 0 = base - gap rows, pass k = base + k; the eager
# loop's in-place `cache_seqlens += 1` on a pinned buffer would race its async uploads once the host runs ahead). Hooked
# from the patched generator's iterate_draftmodel_mtp_gen; returns None to fall back to the eager loop
DRAFT = os.environ.get("EXL3_DRAFT_GRAPH", "0") == "1"
_D = {"graphs": {}, "warm": set(), "buf": {}, "pool": None, "cs": None, "dead": False,
      "n": {"draft_replay": 0, "draft_capture": 0, "draft_eager": 0}}


def _draft_body(gen, B, n_gap, window, sl_host):
    dm = gen.draft_model
    lm_head = gen.model.modules[gen.model.logit_layer_idx]
    cols = []
    ids, hid = B["ids0"], B["hid0"]
    for idx in range(window):
        if idx == 0:
            torch.sub(B["slb"], n_gap, out=B["sl"][0])
        else:
            torch.add(B["slb"], idx, out=B["sl"][idx])
        params = {"target_hidden": hid, "attn_mode": "flash_attn", "block_table": B["bt"], "cache": gen.draft_cache,
                  "cache_seqlens": B["sl"][idx], "draft_step": idx, "sparse_head": gen._sparse_flag,
                  "_mla_host_seqlens": [sl_host - n_gap if idx == 0 else sl_host + idx]}
        state = dm.forward(ids, params)
        if idx == 0 and n_gap:
            state = state[:, -1:, :]
        state = lm_head.prepare_for_device(state, params)
        new_ids = dm.sample_from_state(state, params)
        cols.append(new_ids)
        ids = new_ids
        hid = params["next_target_hidden"] if "next_target_hidden" in params else state
    return torch.cat(cols, dim=1)


def draft_graph(gen, bsz, block_index, cache_seqlens, batch_ids, temp_hidden, gap, window, defer=False):
    if _D["dead"] or _S["dead"] or _S["fe"] is None or bsz != 1:
        return None
    ts = sys.modules.get("tier_split")
    if ts is None:
        return None
    sl = int(cache_seqlens[0])
    n_gap = int(gap["ids"].shape[-1]) if gap is not None else 0
    lim = _S.get("maxctx") or MAXCTX or int(getattr(gen.model.config, "index_topk", 0) or 0) or (1 << 30)
    regime = int(sl + window + 1 > lim)
    if regime and (not SPARSE or sl + 1 <= lim):     # (a step whose passes cross into the sparse regime stays eager)
        _D["n"]["draft_eager"] += 1
        return None
    W = int(block_index.shape[1])
    key = (window, n_gap, W, regime) if SPARSE else (window, n_gap, W)
    if defer and key not in _D["graphs"]:
        return None
    if key not in _D["warm"]:
        _D["warm"].add(key)
        _D["n"]["draft_eager"] += 1
        return None
    dev = temp_hidden.device
    B = _D["buf"].get(key)
    if B is None:
        B = _D["buf"][key] = dict(
            ids0=torch.zeros((1, 1 + n_gap), dtype=torch.long, device=dev),
            hid0=torch.zeros((1, 1 + n_gap, temp_hidden.shape[-1]), dtype=temp_hidden.dtype, device=dev),
            bt=torch.zeros((1, W), dtype=block_index.dtype, device=dev),
            slb=torch.zeros((1,), dtype=cache_seqlens.dtype, device=dev),
            sl=[torch.zeros((1,), dtype=cache_seqlens.dtype, device=dev) for _ in range(window)])
    if n_gap:
        B["ids0"][:, :n_gap].copy_(gap["ids"].view(1, -1), non_blocking=True)
        B["ids0"][:, n_gap:].copy_(batch_ids.view(1, -1), non_blocking=True)
        B["hid0"][:, :n_gap].copy_(gap["target_hidden"].view(1, n_gap, -1), non_blocking=True)
        B["hid0"][:, n_gap:].copy_(temp_hidden.view(1, 1, -1))
    else:
        B["ids0"].copy_(batch_ids.view(1, -1), non_blocking=True)
        B["hid0"].copy_(temp_hidden.view(1, 1, -1))
    B["bt"].copy_(block_index.view(1, -1), non_blocking=True)
    B["slb"].copy_(cache_seqlens.view(-1)[:1], non_blocking=True)
    ent = _D["graphs"].get(key)
    if ent is None:
        if _D["pool"] is None:
            _D["pool"] = torch.cuda.graph_pool_handle()
        if _D["cs"] is None:
            _D["cs"] = torch.cuda.Stream()
        cs, main = _D["cs"], torch.cuda.current_stream()
        cs.wait_stream(main)
        g = torch.cuda.CUDAGraph()
        ts._VG["cap"] = True
        _S["fe"](True)
        try:
            with torch.cuda.stream(cs):
                g.capture_begin(pool=_D["pool"], capture_error_mode="thread_local")
                try:
                    out = _draft_body(gen, B, n_gap, window, sl)
                finally:
                    g.capture_end()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f" -- vgraph: draft capture of {key} failed ({type(e).__name__}: {e}); draft graphs off", flush=True)
            _D["dead"] = True
            torch.cuda.synchronize()
            return None
        finally:
            ts._VG["cap"] = False
            _S["fe"](False)
        main.wait_stream(cs)
        ent = _D["graphs"][key] = {"g": g, "out": out}
        _D["n"]["draft_capture"] += 1
        print(f" -- vgraph: captured draft graph {key}", flush=True)
    ent["g"].replay()
    _D["n"]["draft_replay"] += 1
    if defer:   # hecate (prelaunch): async readback, the caller launches the verify on the device ids first
        gen.draft_ids_pinned[:bsz, :window].copy_(ent["out"], non_blocking=True)
        ev = torch.cuda.Event(); ev.record()
        return {"dev": ent["out"], "pinned": gen.draft_ids_pinned[:, :window], "event": ev}
    gen.draft_ids_pinned[:bsz, :window].copy_(ent["out"])
    return gen.draft_ids_pinned[:, :window]
_PARAM_KEYS = {"attn_mode", "block_table", "cache", "cache_seqlens", "recurrent_states", "indexed_embeddings", "positions",
               "recurrent_history", "pinned_staging", "dev_cache", "export_state_norm_keys"}
_S = {"fe": None, "gen": None, "graphs": {}, "warm": set(), "ids": {}, "pool": None, "dead": False,
      "n": {"replay": 0, "capture": 0, "eager": 0}, "why": {}, "calls": 0, "cap_s": 0.0, "chk": [],
      "ring": collections.deque(maxlen=48), "last_ev": None, "dumped": False}


def _watchdog():
    """a replay whose completion event stays pending > 20 s: dump the Grace pool's ISSUE / DONE / meta words, every RTX
    layer slot's ISSUE / DONE flags (host-pinned, readable while the GPU spins) and the last vgraph actions, once"""
    while True:
        time.sleep(5)
        e = _S["last_ev"]
        if e is None or _S["dumped"]:
            continue
        ev, t = e
        try:
            if ev.query() or time.time() - t < 20:
                continue
        except Exception:
            continue
        _S["dumped"] = True
        ts = sys.modules.get("tier_split")
        u32 = lambda a: ctypes.c_uint32.from_address(a).value
        try:
            c = ts._CPU.ctl
            print(f" -- vgraph WATCHDOG: replay pending {time.time() - t:.0f} s; CPU ISSUE {u32(c):#x} DONE {u32(c + 64):#x} "
                  f"meta {u32(c + 128)} {u32(c + 132)}", flush=True)
            sl = ts.rg.get().slots; off = sl.ny + sl.ns + sl.nw + sl.no
            fl = [(i, u32(sl.hbuf + i * sl.per + off), u32(sl.hbuf + i * sl.per + off + 64)) for i in range(sl.n)]
            print(" -- vgraph WATCHDOG: RTX slots (issue, done) != (0, 0): "
                  + " ".join(f"{i}:{a},{b}" for i, a, b in fl if a or b), flush=True)
        except Exception as ex:
            print(f" -- vgraph WATCHDOG: flag read failed: {ex}", flush=True)
        print(" -- vgraph WATCHDOG: last actions: " + " | ".join(_S["ring"]), flush=True)


def verify_prelaunch(model, last_tok, draft_dev, params):
    """hecate (prelaunch): replay an already captured verify graph whose input ids are the pinned last token plus the drafted
    ids still on the device; None when no graph matches (the caller then runs the regular forward)"""
    ts = sys.modules.get("tier_split")
    if _S["dead"] or ts is None or not ON:
        return None
    sl_h, bt_h = params.get("cache_seqlens"), params.get("block_table")
    if sl_h is None or bt_h is None or sl_h.is_cuda or bt_h.is_cuda or set(params) - _PARAM_KEYS - {"export_states"}:
        return None
    R = 1 + int(draft_dev.shape[-1])
    sl = sl_h.tolist()
    lim = _S.get("maxctx")
    if lim is None or (sl[0] + R > lim and not SPARSE):
        return None
    dev = torch.device("cuda", torch.cuda.current_device())
    ptrs = _dev_ptrs(params, dev)
    key = (R, bt_h.shape[1], (int(sl[0] + R > lim),) if SPARSE else ())
    ent = _S["graphs"].get(key)
    ids = _S["ids"].get(R)
    if ptrs is None or ent is None or ent["ptrs"] != ptrs or ids is None:
        return None
    ids[:, :1].copy_(last_tok, non_blocking=True)
    ids[:, 1:].copy_(draft_dev.view(1, -1))
    if ts.rg.MODS:
        ts.rg.launch(R)
    ent["g"].replay()
    ev = torch.cuda.Event(); ev.record(); _S["last_ev"] = (ev, time.time())
    _S["ring"].append(f"prelaunch {key[0]}x{key[1]} sl {sl[0]}")
    for k, v_ in ent["new"].items():
        params[k] = [t.clone() for t in v_] if k == "export_states" else v_
    _S["n"]["replay"] += 1
    _S["n"]["pre"] = _S["n"].get("pre", 0) + 1
    return ent["out"]


def _eager(reason):
    _S["ring"].append(f"eager {reason}")
    _S["n"]["eager"] += 1
    _S["why"][reason] = _S["why"].get(reason, 0) + 1


def _report():
    _S["calls"] += 1
    if _S["calls"] % EVERY == 0:
        n = _S["n"]
        why = ", ".join(f"{k} {v}" for k, v in sorted(_S["why"].items(), key=lambda kv: -kv[1]))
        print(f" -- vgraph: {n['replay']} replays, {n['capture']} captures ({_S['cap_s']:.1f} s), {n['eager']} eager"
              + (f" ({why})" if why else "") + f", {len(_S['graphs'])} graphs"
              + (f"; drafts {_D['n']}" if DRAFT else ""), flush=True)
        if _S["chk"]:
            c = _S["chk"]; _S["chk"] = []
            mn = lambda i: sum(x[i] for x in c) / len(c)
            print(f" -- vgraph check: {len(c)} steps, KL(eager || graph) mean {mn(0):.2e} max {max(x[0] for x in c):.2e}, "
                  f"top-1 agree {mn(1):.4f} | noise floor KL(eager || eager) mean {mn(2):.2e} max {max(x[2] for x in c):.2e}, "
                  f"top-1 agree {mn(3):.4f}", flush=True)


def _dev_ptrs(params, dev):
    dc = params.get("dev_cache")
    if not dc:
        return None
    out = []
    for k in ("block_table", "cache_seqlens", "positions"):
        hv = params.get(k)
        if hv is None:
            out.append(0); continue
        e = dc.get((id(hv), dev))
        if e is None:
            return None
        out.append(e[1].data_ptr())
    return tuple(out)


def _verify(model, fw0, input_ids, params):
    ts = sys.modules.get("tier_split")
    if _S["dead"] or ts is None:
        _eager("off"); return fw0(model, input_ids, params)
    if input_ids.dim() != 2 or input_ids.shape[0] != 1 or not (2 <= input_ids.shape[1] <= 8) or input_ids.is_cuda:
        _eager("shape"); return fw0(model, input_ids, params)
    extra = set(params) - _PARAM_KEYS
    if extra or params.get("recurrent_states") or params.get("indexed_embeddings"):
        _eager("params " + ",".join(sorted(extra))); return fw0(model, input_ids, params)
    sl_h, bt_h = params.get("cache_seqlens"), params.get("block_table")
    if sl_h is None or bt_h is None or sl_h.is_cuda or bt_h.is_cuda:
        _eager("host"); return fw0(model, input_ids, params)
    R = input_ids.shape[1]
    sl = sl_h.tolist()
    lim = _S.get("maxctx")
    if lim is None:
        lim = _S["maxctx"] = MAXCTX or int(getattr(model.config, "index_topk", 0) or 0) or (1 << 30)
        print(f" -- vgraph: dense-regime bound {lim} tokens", flush=True)
    regime = int(sl[0] + R > lim)
    if regime and not SPARSE:
        _eager("sparse")                  # DSA regime: eager (per-module graphs) unless EXL3_VERIFY_GRAPH_SPARSE
        return fw0(model, input_ids, params)
    dev = torch.device("cuda", torch.cuda.current_device())
    ptrs = _dev_ptrs(params, dev)
    if ptrs is None:
        _eager("arena"); return fw0(model, input_ids, params)
    flags = (regime,) if SPARSE else ()
    key = (R, bt_h.shape[1], flags)
    ent = _S["graphs"].get(key)
    if ent is not None and ent["ptrs"] != ptrs:
        del _S["graphs"][key]; ent = None
        _S["why"]["recapture"] = _S["why"].get("recapture", 0) + 1
    ids = _S["ids"].get(R)
    if ids is None:
        ids = _S["ids"][R] = torch.zeros((1, R), dtype=torch.long, device=dev)
    if ent is None:
        if key not in _S["warm"]:
            _S["warm"].add(key)
            _eager("warm"); return fw0(model, input_ids, params)
        t0 = time.perf_counter()
        ids.copy_(input_ids, non_blocking=True)
        p = dict(params); p["_mla_host_seqlens"] = sl
        if _S["pool"] is None:
            _S["pool"] = torch.cuda.graph_pool_handle()
        g = torch.cuda.CUDAGraph()
        ts._VG["cap"] = True
        if _S["fe"] is not None:
            _S["fe"](True)
        try:
            with torch.cuda.graph(g, pool=_S["pool"], capture_error_mode="thread_local"):
                out = fw0(model, ids, p)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f" -- vgraph: capture of {key} failed ({type(e).__name__}: {e}); verify graphs off", flush=True)
            _S["dead"] = True
            torch.cuda.synchronize()
            _eager("failed"); return fw0(model, input_ids, params)
        finally:
            ts._VG["cap"] = False
            if _S["fe"] is not None:
                _S["fe"](False)
        new = {k: v for k, v in p.items() if k not in params and k not in ("_mla_host_seqlens", "input_ids",
                                                                           "layer_instance", "resid_defer")}
        ent = _S["graphs"][key] = {"g": g, "out": out, "new": new, "ptrs": ptrs}
        _S["n"]["capture"] += 1; _S["cap_s"] += time.perf_counter() - t0
        print(f" -- vgraph: captured {key} in {time.perf_counter() - t0:.2f} s, outputs {sorted(new)}", flush=True)
    else:
        ids.copy_(input_ids, non_blocking=True)
    if ts.rg.MODS:
        ts.rg.launch(R)                     # the step's RTX band (waits on each layer's ISSUE, as in tier_submit)
    ent["g"].replay()
    ev = torch.cuda.Event(); ev.record(); _S["last_ev"] = (ev, time.time())
    _S["ring"].append(f"replay {key[0]}x{key[1]} {int(key[2][0]) if key[2] else '-'} sl {sl[0]}")
    for k, v in ent["new"].items():
        params[k] = [t.clone() for t in v] if k == "export_states" else v
    _S["n"]["replay"] += 1
    if CHECK and _S["n"]["replay"] % CHECK == 0:
        # two eager reruns: their mutual KL is the noise floor (fp32 atomics in the router / moe4 reorder sums, near-tie
        # expert picks can flip), graph vs eager has to match it
        p2 = {k: v for k, v in params.items() if k not in ent["new"]}
        a = ent["out"].float().clone()
        b = fw0(model, input_ids, dict(p2)).float().clone()
        c = fw0(model, input_ids, dict(p2)).float()
        la, lb, lc = (torch.log_softmax(t.view(-1, t.shape[-1]), -1) for t in (a, b, c))
        kl = lambda p_, q_: (p_.exp() * (p_ - q_)).sum(-1).max().item()
        _S["chk"].append((kl(lb, la), (la.argmax(-1) == lb.argmax(-1)).float().mean().item(), kl(lb, lc),
                          (lb.argmax(-1) == lc.argmax(-1)).float().mean().item()))
    _report()
    return ent["out"]


def install():
    if not ON:
        return
    from exllamav3.ext import exllamav3_ext as _ext
    _S["fe"] = getattr(_ext, "graphs_force_eager", None)
    if _S["fe"] is None and os.environ.get("EXL3_GRAPHS", "1") != "0":
        print(" -- vgraph: EXL3_VERIFY_GRAPH=1 needs EXL3_GRAPHS=0 or an image with graphs_force_eager (module graphs "
              "cannot nest in the capture); off", flush=True)
        return
    from exllamav3.generator import generator as _g
    G = _g.Generator
    ig = G.iterate_gen

    def iterate_gen(self, results, draft_tokens=None, pre=None):
        if draft_tokens is None:
            return ig(self, results, draft_tokens, pre=pre)
        m = self.model
        prev = m.__dict__.get("forward")
        fw = m.forward                      # bound (class or an inner wrapper's) forward

        def forward(input_ids=None, params=None, _fw=fw):
            if params is None:
                return _fw(input_ids=input_ids, params=params)
            return _verify(m, lambda _m, i, p: _fw(input_ids=i, params=p), input_ids, params)
        m.forward = forward
        try:
            return ig(self, results, draft_tokens, pre=pre)
        finally:
            if prev is None:
                m.__dict__.pop("forward", None)
            else:
                m.forward = prev
    G.iterate_gen = iterate_gen
    threading.Thread(target=_watchdog, name="vgraph-watchdog", daemon=True).start()
    print(f" -- hecate: verify-forward CUDA graphs (EXL3_VERIFY_GRAPH, dense attention regime" + (f" <= {MAXCTX} tokens)" if MAXCTX else ", up to the model index_topk)"), flush=True)
