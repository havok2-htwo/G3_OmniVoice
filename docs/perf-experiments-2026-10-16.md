# OmniVoice performance experiments (2026-10-16)

Handoff for the next round of latency / throughput work. Everything here was measured in-process on the
RTX 5090 (32 GB, Windows 11 WDDM), with the **24GB VRAM** tier values (`max_batch_size` 4, `max_parallel_requests` 4,
`num_step` 20, `vram_budget_mb` 5000, bf16, `torch.compile` dynamic, sdpa) unless stated otherwise. The GPU was
otherwise idle during these runs.

Raw result files and scripts live in the scratch folder of the session (see "Scripts" below); the numbers that matter
are copied here.

## Units and what is reliable

- **Reliable:** wall time of a whole wave (`wall_s`), audio seconds produced (`audio_s`, from the job metrics), and the
  realtime factor derived from them (`rtf_x` = audio s / wall s). Medians of 3 waves after untimed warm-up.
- **Reliable for single requests:** time to first audio (TTFA) = `first_audio_at - created_at` for streamed requests.
- **Not reliable:** the batch counter in `bench_mix.py` (`batches`, `batch_sizes`). The per-wave numbers disagreed with
  the trace of the batch hook; do not draw batch-level conclusions from it. Re-check with a direct trace before relying on it.

## Baseline numbers

| Measurement | Result | Source |
|---|---|---|
| Single request, one ~7 s sentence, idle worker: decoder step cost | B=1: 0.209 s per `generate` (20 steps, ~10 ms per step) | `cfg/graphs_test.py` |
| Batch of 4 sentences in one `generate` | 0.624 s (~31 ms per step) | `cfg/graphs_test.py` |
| Four separate solo generates of the same 4 sentences | 0.84–0.87 s total | `seedrow/row_seeds.py` |
| TTFA, streamed single request, idle worker (12 samples) | median 0.23 s | `mixvoice/bench_mix.py` (`BENCH_TTFA=1`) |
| 16 concurrent requests, one voice (SonjaV2) | wall 1.27 s, RTF 39.9 | `mixvoice/bench_mix.py` |
| 16 concurrent requests, voices Sonja + SonjaV2 alternating | wall 1.71–1.76 s, RTF 30.5 | same |
| 16 concurrent requests, Sonja only (14.4 s reference) | wall 1.77 s, RTF 30.5 | same |
| 16 concurrent requests, 16 different seeds, one voice | wall 1.93 s, RTF 26.0, every batch of size 1 | same, `BENCH_DISTINCT_SEEDS=1` |
| 16 requests, short (62 chars) and long (305 chars) alternating | wall 2.57 s, RTF 37 | same, `len_mix` |

Under the full combined load (all five G3 services on the same card, `docs/VRAM_BUDGET.md` and the combined
run), OmniVoice ran at roughly 25–50 % of its solo throughput (16 sentences: 2.6 s solo vs 10–12 s under load).

## Experiments that worked

1. **Clone requests of different voices share a batch** (committed, `01baf21`). The group key no longer includes the
   voice for `TaskType.base`; the batch runner already passes one clone prompt per item. Mixed voices went from 4
   partial batches (1.76 s) to 2 full batches (1.71 s). Same-voice load unchanged. Gain is small; the benefit is
   batch fill under mixed traffic.
2. **SonjaV2 (8.2 s reference) instead of Sonja (14.4 s):** 1.27 s vs 1.77 s for the same 16 requests, RTF 40 vs 30.
   This is already the default voice.

## Experiments that were tested and rejected

3. **Length-grouped batch planning** (sort the candidate sentences by length against the anchor, oldest request
   first). Batches did become length-homogeneous ([62, 62, 305, 305] instead of [62, 305, 62, 305]) but wall time was
   identical (2.57 s vs 2.57 s). Reverted. Conclusion: padding to the longest item is not the bottleneck at batch 4.
4. **CFG unconditional rows at their own length** (split the single 2B-row forward into a conditional forward at the
   full length and an unconditional forward cut to the longest unconditional target; `cfg/cfg_split.py`):
   - output bit-identical to the stock forward (max abs difference 0.0 for all items);
   - peak allocated 2561 → 2342 MiB (SonjaV2) and 2815 → 2510 MiB (Sonja);
   - **slower**: generate 0.627 → 0.719 s (×0.87) for SonjaV2, 0.870 → 0.924 s (×0.94) for Sonja. Two smaller forwards pay
     the per-forward overhead twice, and the saved work is small. Not adopted.
   - An earlier version read the bound back to the host on every step; caching it per generate did not change the result.
5. **KV cache off** (`model.llm.config.use_cache = False`): peak 2561 → 2239 MiB at batch 4, time unchanged (0.627 vs
   0.625 s), **but the audio changes** (max abs difference 1.06–1.36 per item). Not adopted. Not yet explained; the
   transformers mask path with and without a cache is the first thing to check.
6. **CUDA graphs for the decoder** (`torch.compile(mode="reduce-overhead", dynamic=False)` on the forward, fixed shapes):
   B=1 0.2086 → 0.2048 s, B=4 0.624 → 0.607 s. 2–3 % only. Not worth the shape-bucketing and recompile cost.
   Conclusion: the step is compute-bound at these sizes, not launch-bound.
7. **Codec decode** (fp32 as configured, random codes of 175 frames = 7 s): 5.5 ms for one item, 18.7 ms for a batch of
   four. About 2 % of a request. Not a target.

## Seeds: the open question

- **Today:** the seed is applied once per batch (`runtime_v2.py`, `_apply_seed(items[0].request.seed)`) and is part of
  the batch group key, so requests with different seeds never share a batch. Measured: 16 distinct seeds on one voice
  give 1.93 s and every batch has size 1 (vs 1.27 s with one seed). Loss about 35 %.
- **Only random seeds** (`seed` = None) are safe to batch today: they share the group key `random`. Most production
  traffic is probably like that; check the real request mix before optimising.
- **Prototype of per-row seeds** (`seedrow/row_seeds.py`): the only random draw in the library is `_gumbel_sample`
  (`torch.rand_like`, twice per step, `models/omnivoice.py` around line 1502). Giving each row its own
  `torch.Generator` seeded from its request makes distinct seeds batchable at **the same speed** (batched distinct seeds
  0.655 s vs stock same-seed batch 0.658 s).
- **Caveats of the prototype, not solved yet:**
  - it changes the noise stream, so the audio for a given seed differs from stock; quality equivalence needs a WER run
    (`Seed-Lauf` / WER benchmark) and a listening check;
  - the noise for a row is drawn with the shape of the padded batch, so the same seed gives different audio in a batch and
    alone (max abs difference 1.1–1.4 per item in the test). Per-item reproducibility needs noise drawn at the item's own
    length, e.g. `torch.rand` of the item's `[C, target_len]` and padded afterwards;
  - the patch is a monkey patch of the `omnivoice` package (pinned upstream version). A proper change needs an upstream
    patch or a vendored copy.
- **If that works:** the best seed would no longer depend on batch size (today it does), which also fixes the seed benchmark.

## Ideas not measured yet

- `num_step` 18 or 16: linear cost, about 10 % per two steps removed; needs WER and a listening check. Quality decision.
- `max_parallel_requests` 4 vs 8 on the 24 GB tier: how much queueing it adds under 16 concurrent requests (TTFA under load).
- Larger batches on the 24 GB tier (`max_batch_size` 6–8) if VRAM allows: throughput flat from 4 in the solo runs, so likely no gain.
- `torch.compile` mode `max-autotune-no-cudagraphs` for the decoder (about 1–5 % in the earlier 4090 notes, not checked here).
- Flex attention for OmniVoice on the 5090 (the 4090 note says a tie with sdpa).
- FlashInfer and other Linux-only kernels: not testable on this PC (no WSL, no Docker).

## Scripts (scratch folder of the session)

- `mixvoice/bench_mix.py OUT.json WAVES` — same/mixed/Sonja-only/length-mix waves, TTFA for streamed requests,
  `BENCH_TTFA=1`, `BENCH_DISTINCT_SEEDS=1`, `BENCH_FIFO=1` (disables the planner ordering that no longer exists).
- `cfg/cfg_split.py OUT.json` — CFG split against stock, output equality, timing, peak memory.
- `cfg/graphs_test.py OUT.json` — stock dynamic compile vs `reduce-overhead` on the decoder, B=1 and B=4.
- `cfg/cache_decode.py OUT.json` — KV cache on/off, codec decode timing.
- `seedrow/row_seeds.py OUT.json` — per-row generators prototype.

Environment for all of them: `OMNIVOICE_TTS_DATA_DIR` pointing at a copy of `data/` (the scripts refuse the real one),
`LOCALAPPDATA` and `APPDATA` as in `start_server.bat`, `OMNIVOICE_TTS_MODELS_ROOT_DIR=X:\dev\G3_OmniVoice\models`,
`CUDA_VISIBLE_DEVICES=0`, `PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256`,
`OPENBLAS_NUM_THREADS=4`, and the `.conda-env` python.

## Code state

- Committed: clone requests of different voices share a batch (`01baf21`), with a test in `tests/test_runtime_v2.py`.
- Not committed: nothing. The experiments above are in the scratch folder only.
- Live OmniVoice on port 8091 runs the current committed code with the `32GB VRAM` tier values.
