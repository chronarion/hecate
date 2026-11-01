"""Regenerate runtime/patched/{generator,async_generator}.py from ExLlamaV3's own files (rerun after an ExLlamaV3 update;
every replacement is asserted, so a changed upstream file fails loudly instead of silently mis-patching):
  generator.py        decode share while a prompt is ingested (EXL3_PREFILL_DECODE_SHARE), MTP gap merge
                      (EXL3_MTP_MERGE_GAP), hook for the draft-loop CUDA graph (EXL3_DRAFT_GRAPH), and the pipelined
                      MTP decode step (EXL3_PRELAUNCH, prelaunch_patch.py)
  async_generator.py  yield to the event loop after long steps, so requests arriving mid-prefill are admitted
   python3 make_patches.py IMAGE_FILES_DIR OUT_DIR   (IMAGE_FILES_DIR: exllamav3/generator/ of the built image)"""
import sys, os
src, out = sys.argv[1:3]
os.makedirs(out, exist_ok=True)
def patch(name, reps):
    s = open(f"{src}/{name}").read()
    for old, new in reps:
        assert s.count(old) == 1, (name, old[:80], s.count(old))
        s = s.replace(old, new)
    open(f"{out}/{name}", "w").write(s)
    print("patched", name)


patch("generator.py", [
    # time-based decode share while a prompt is ingested beside generating jobs (EXL3_PREFILL_DECODE_SHARE = r in
    # (0, 1)): after each shared prefill chunk (timed to GPU completion) the generating jobs decode alone for
    # chunk time * r / (1 - r), instead of the fixed prefill_cooldown iteration count
    ("            if self.prefill_cooldown_left > 0:\n"
     "                self.prefill_cooldown_left -= 1\n"
     "                deferred = prefilling\n",
     "            if _PF_SHARE > 0:\n"
     "                if time.perf_counter() < getattr(self, \"_pf_next\", 0.0):\n"
     "                    deferred = prefilling\n"
     "                else:\n"
     "                    selected = prefilling[self.prefill_rotation % len(prefilling)]\n"
     "                    self.prefill_rotation += 1\n"
     "                    deferred = [job for job in prefilling if job is not selected]\n"
     "                    pf_timed = time.perf_counter()\n"
     "            elif self.prefill_cooldown_left > 0:\n"
     "                self.prefill_cooldown_left -= 1\n"
     "                deferred = prefilling\n"),
    ("        prefilling = [job for job in self.active_jobs if not job.is_prefill_done()]\n"
     "        chunk_size = None\n",
     "        prefilling = [job for job in self.active_jobs if not job.is_prefill_done()]\n"
     "        chunk_size = None\n"
     "        pf_timed = None\n"),
    ("        # Recurrent checkpoints\n"
     "        if self.recurrent_cache is not None:\n",
     "        if pf_timed is not None:\n"
     "            torch.cuda.synchronize()\n"
     "            t_end = time.perf_counter()\n"
     "            self._pf_next = t_end + (t_end - pf_timed) * _PF_SHARE / (1.0 - _PF_SHARE)\n"
     "\n"
     "        # Recurrent checkpoints\n"
     "        if self.recurrent_cache is not None:\n"),
    ("def _pad_pages(", "_PF_SHARE = min(0.9, max(0.0, float(os.environ.get(\"EXL3_PREFILL_DECODE_SHARE\", \"0\") or 0)))\n"
     "_MTP_MERGE = os.environ.get(\"EXL3_MTP_MERGE_GAP\", \"0\") == \"1\"\n\n\ndef _pad_pages("),
    # EXL3_MTP_MERGE_GAP=1 (one drafting job): the MTP gap prefill of the accepted positions K+1..K+A-1 is not run after
    # the verify; it is stashed and becomes the first A-1 rows of the next round's first draft pass (one A-row MTP
    # forward instead of a gap forward plus a 1-row one; causal, so the same K/V and the same last-row state). A stash
    # the next round does not consume (several jobs, no drafting) is flushed as the original prefill first.
    ("                if accepted_length > 1:\n"
     "                    self.draft_model.prefill(\n"
     "                        batch_ids[a_idx:b_idx, 1:accepted_length],\n",
     "                if accepted_length > 1 and _MTP_MERGE and len(accepted_lengths) == 1 and b_idx - a_idx == 1:\n"
     "                    job.mtp_gap = {\n"
     "                        \"ids\": batch_ids[a_idx:b_idx, 1:accepted_length].clone(),\n"
     "                        \"block_table\": block_index[a_idx:b_idx].clone(),\n"
     "                        \"cache_seqlens\": p_cache_seqlens[a_idx:b_idx] + 1,\n"
     "                        \"target_hidden\": target_hidden[a_idx:b_idx, :accepted_length - 1, :].clone(),\n"
     "                    }\n"
     "                elif accepted_length > 1:\n"
     "                    self.draft_model.prefill(\n"
     "                        batch_ids[a_idx:b_idx, 1:accepted_length],\n"),
    ("                job.mtp_last_hidden = target_hidden[\n"
     "                    a_idx:b_idx, accepted_length - 1:accepted_length, :\n"
     "                ].clone()\n",
     "                job.mtp_last_hidden = target_hidden[\n"
     "                    a_idx:b_idx, accepted_length - 1:accepted_length, :\n"
     "                ].clone()\n"
     "                if getattr(job, \"mtp_gap\", None) is not None:\n"
     "                    job.mtp_gap[\"owner\"] = job.mtp_last_hidden   # stale once a rewind / replay replaces the carry\n"),
    ("    def iterate_gen(self, results: list, draft_tokens: torch.Tensor | None = None):\n",
     "    def _mtp_flush_gaps(self):\n"
     "        # write stashed MTP gaps (EXL3_MTP_MERGE_GAP) the way the verify round would have\n"
     "        for job in self.active_jobs:\n"
     "            g = getattr(job, \"mtp_gap\", None)\n"
     "            if g is None:\n"
     "                continue\n"
     "            job.mtp_gap = None\n"
     "            if g.get(\"owner\") is not job.mtp_last_hidden or job.mtp_last_hidden is None:\n"
     "                continue\n"
     "            self.draft_model.prefill(g[\"ids\"], {\"attn_mode\": \"flash_attn\", \"block_table\": g[\"block_table\"],\n"
     "                \"cache\": self.draft_cache, \"cache_seqlens\": g[\"cache_seqlens\"], \"target_hidden\": g[\"target_hidden\"]})\n"
     "\n"
     "    def iterate_gen(self, results: list, draft_tokens: torch.Tensor | None = None):\n"
     "        if _MTP_MERGE:\n"
     "            self._mtp_flush_gaps()\n"),
    ("        batch_ids = self.draft_input_ids_pinned[:batch_size, :]\n"
     "        batch_ids.copy_(torch.cat(input_ids_list, dim = 0))\n"
     "        temp_hidden = torch.cat(mtp_hidden_list, dim = 0)\n",
     "        batch_ids = self.draft_input_ids_pinned[:batch_size, :]\n"
     "        batch_ids.copy_(torch.cat(input_ids_list, dim = 0))\n"
     "        temp_hidden = torch.cat(mtp_hidden_list, dim = 0)\n"
     "        gap = None\n"
     "        if _MTP_MERGE:\n"
     "            jobs_d = [job for job in self.active_jobs if job.is_prefill_done()]\n"
     "            g = getattr(jobs_d[0], \"mtp_gap\", None) if len(jobs_d) == 1 else None\n"
     "            if g is not None and g.get(\"owner\") is jobs_d[0].mtp_last_hidden and self.draft_calibrator is None \\\n"
     "                    and not self.draft_model.caps.get(\"draft_uses_target_cache\", False) and self.sparse_head is None:\n"
     "                gap = g; jobs_d[0].mtp_gap = None\n"
     "            else:\n"
     "                self._mtp_flush_gaps()\n"),
    ("            batch_state = self.draft_model.forward(batch_ids, params)\n",
     "            if gap is not None and idx == 0:\n"
     "                # first draft pass carries the stashed gap rows: positions kv - (A - 1) .. kv, last row drafts\n"
     "                n_gap = gap[\"ids\"].shape[-1]\n"
     "                ids0 = torch.cat((gap[\"ids\"].to(batch_ids.device), batch_ids), dim = -1)\n"
     "                params[\"target_hidden\"] = torch.cat((gap[\"target_hidden\"].to(temp_hidden.device), temp_hidden), dim = 1)\n"
     "                params[\"cache_seqlens\"] = cache_seqlens - n_gap\n"
     "                batch_state = self.draft_model.forward(ids0, params)[:, -1:, :]\n"
     "            else:\n"
     "                batch_state = self.draft_model.forward(batch_ids, params)\n"),
    # decode-draft CUDA graph (runtime/core/exl3_vgraph.py, EXL3_DRAFT_GRAPH): every MTP pass of the step replayed at once
    ("                if self.target_recurrent else None\n        for idx in range(window):\n            params = {\n                \"target_hidden\": temp_hidden,\n",
     "                if self.target_recurrent else None\n"
     "        _vg = __import__(\"sys\").modules.get(\"exl3_vgraph\")\n"
     "        if _vg is not None and _vg.DRAFT and not target_kv and cal is None and self.sparse_head is None:\n"
     "            _r = _vg.draft_graph(self, batch_size, block_index, cache_seqlens, batch_ids, temp_hidden, gap, window)\n"
     "            if _r is not None:\n"
     "                return _r\n"
     "        for idx in range(window):\n            params = {\n                \"target_hidden\": temp_hidden,\n"),
])

patch("async_generator.py", [
    ("import asyncio\n", "import asyncio\nimport os as _os\nimport time as _time\n"),
    ("                results = self.generator.iterate()\n"
     "                self.deliver_results(results)\n",
     "                _t_it = _time.perf_counter()\n"
     "                results = self.generator.iterate()\n"
     "                self.deliver_results(results)\n"
     "                # hecate: iterate() runs on the event loop, so a multi-second prefill chunk stalls every incoming\n"
     "                # request (one loop turn per chunk; a short request arriving mid-prefill waited for the whole prompt).\n"
     "                # After a long step, give the loop a short real sleep so queued requests reach enqueue()\n"
     "                if _time.perf_counter() - _t_it > _LONG_STEP:\n"
     "                    await asyncio.sleep(_LONG_YIELD)\n"),
    ("class AsyncGenerator", "_LONG_STEP = float(_os.environ.get(\"EXL3_LONG_STEP_S\", \"0.2\"))\n"
     "_LONG_YIELD = float(_os.environ.get(\"EXL3_LONG_STEP_YIELD_S\", \"0.02\"))\n\n\nclass AsyncGenerator"),
])

# pipelined MTP decode step (EXL3_PRELAUNCH), on top of the generator patches above
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prelaunch_patch  # noqa: E402
prelaunch_patch.apply(f"{out}/generator.py")
