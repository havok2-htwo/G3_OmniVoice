# TTS engine comparison: OmniVoice vs Ming-3.0-tts vs Gradium Phonon (2026-10-29)

This page compares the server's three engines on one machine, with one method. The Ming and OmniVoice numbers
come from [ming-3.0-tts-2026-10.md](ming-3.0-tts-2026-10.md) and were re-run here to add CPU and RAM. Phonon was
tuned first (section 3) and then measured with the same method.

## Hardware and software

- Windows 11, AMD Ryzen 7 9800X3D (8 cores / 16 threads), 63.6 GB RAM, one RTX 5090 32 GB (WDDM, about 1.06–1.09 GB
  in use with nothing running).
- `.conda-env`: Python 3.12, torch 2.10+cu130, `omnivoice` 0.1.5, `ptts` 0.5.0. Repo at `c4f4794`.
- **OmniVoice**: `32GB VRAM` preset (batch 8, `max_parallel_requests` 8, `batch_wait_ms` 30, compile on, num_step 20).
- **Ming 32 / Ming 64**: `ming_max_slots` = `max_parallel_requests` = 32 or 64, `batch_wait_ms` 0, CUDA graphs, bf16.
- **Phonon** (best config from section 3): `phonon_quant` q4k, `phonon_workers` 8, `phonon_threads` 0
  (= 1 Rust thread per engine on 16 logical CPUs), `max_batch_size` = `max_parallel_requests` = 16, `batch_wait_ms` 30.
  The three `phonon_*` values are not runtime settings; they were set with `OMNIVOICE_TTS_PHONON_QUANT`,
  `OMNIVOICE_TTS_PHONON_WORKERS` and `OMNIVOICE_TTS_PHONON_THREADS`.
- All engines: `max_queue_size` 256, language `de`. Voice `SonjaV2` for OmniVoice and Ming, `default` for Phonon
  (the only voice Phonon mode has).

## 1. Main table

Throughput and per-request speed are given as "N× realtime" (seconds of audio per wall second); RTF = wall time /
audio time, lower is better. TTFA is p50 / p95 in ms on the NDJSON stream; a burst of N = N requests sent at the
same instant. Where the re-run differed from the reference run by more than 10 %, both are shown as
"reference (re-run)". CPU and RAM come only from the re-run. Memory "GB" here is 1000 MiB as `nvidia-smi` and
psutil report it (about 1.05 GB SI, 0.98 GiB), the same convention as the Ming page.

| | OmniVoice `32GB VRAM` | Ming 32 slots | Ming 64 slots | Phonon q4k, 8 workers, batch 16 |
| --- | --- | --- | --- | --- |
| **Aggregate throughput at saturation** (45 s closed loop) | 54.5× (32 clients) | 154.8× (32) | 220.7× (64) | 33.2× (32) |
| Per-request speed, 1 stream | 21.8× (RTF 0.046) | 7.7× (0.129) | 6.0× (0.166) | 8.3× (0.121) |
| Per-request speed, burst of 32 (p50) | 2.5× (0.402) | 4.9× (0.206) | 4.2× (0.239) | 1.3× (0.762) |
| Per-request speed at saturation (p50) | 1.7× (0.582) | 4.9× (0.202) | 3.6× (0.281) | 1.03× (0.971) |
| TTFA, 1 stream | 213 / 333 (224 / 243) | 30 / 38 | 36 / 48 | 77 / 85 |
| TTFA, burst of 4 | 409 / 426 | 58 / 59 | 69 / 70 | 102 / 118 |
| TTFA, burst of 16 | 1074 / 1442 | 115 / 131 (126 / 143) | 125 / 143 | 1089 / 1277 ¹ |
| TTFA, burst of 32 | 1856 / 2986 | 197 / 229 (197 / 264) | 201 / 235 (215 / 278) | 1932 / 3220 |
| TTFA at saturation | 2815 / 3014 ² | 61 / 87 (65 / 98) | 85 / 136 (94 / 187) | 2399 / 3302 ² |
| VRAM idle / peak (GB) | 2.56 / 4.09 | 4.38 / 4.39 | 4.65 / 4.67 | 0.00 / 0.01 (CPU only) |
| CPU at saturation, server process (logical cores of 16) | 0.5 | 1.2 | 1.2 | 10.7 |
| Whole-machine CPU at saturation | 9 % | 14 % | 15 % | 76 % |
| RAM, server process: idle / at saturation (working set, GB) | 2.54 / 2.69 | 2.42 / 2.73 | 2.43 / 2.76 | 1.43 / 2.19 |
| RAM, private bytes (commit) at saturation (GB) ³ | 7.5 | 7.9 | 8.2 | 2.7 |
| Server start to warm (s) | 48 (re-run 50) | 12 (12.7) | 13 (13.6) | 6.6 |
| German WER, `text_optimization` off / deterministic | 4.69 % / 3.18 % | 4.36 % / 3.52 % ⁴ | (not measured) ⁴ | 4.02 % / 3.02 % ⁵ |

¹ Phonon renders at most 8 sentences at once (8 workers). In a burst of 16 the first 8 get audio after about 0.1 s,
the other 8 wait about one sentence. The p50 sits on that edge, so it is unstable: the raw PCM run of the same
burst gave 541 / 1573 ms.
² More clients than the engine admits at once (OmniVoice 8, Phonon 16): this is mostly queueing time.
³ Private bytes include CUDA's pinned and mapped host memory for the GPU engines; the working set is the physical
RAM in use.
⁴ Ming WER was measured with 32 slots. Ming and OmniVoice: 3 voices (SonjaV2, SonjaV3, Tom), 60 clips per mode.
⁵ Phonon: voice `default` only, 20 clips per mode. Measured with q4k; q8 gave 5.03 % / 4.02 % (section 4).

Raw PCM TTFA (`/v1/audio/speech`, `stream: true`), p50 / p95 in ms:

| Engine | 1 stream | Burst 16 |
| --- | --- | --- |
| OmniVoice `32GB VRAM` | 198 / 231 | 1100 / 1644 |
| Ming 32 | 30 / 31 | 116 / 138 |
| Ming 64 | 36 / 37 | 126 / 141 |
| Phonon q4k, 8 workers | 82 / 86 | 541 / 1573 ¹ |

### Re-run vs reference

The CPU re-runs (same settings as the reference runs of 2026-10-28) matched the reference throughput within 3 %
(Ming 32 155.3×, Ming 64 216.5×, OmniVoice 53.5×) and the start times within 5 %.

TTFA differences above 10 % appear only in the tails under load:

- Ming 32: burst-32 p95 +15 %, saturation p95 +13 %; the burst-16 p50 was +10 %.
- Ming 64: burst-32 p95 +18 %, saturation p50 / p95 +11 % / +37 %. The burst of 64 was 459 / 725 ms against
  376 / 568 ms.
- OmniVoice: single-stream p95 −27 %. The reference p95 contained one slow request.

Apart from the Ming 32 burst-16 p50 (+10.0 %), the p50 values at 1–16 streams are within 10 %.

## 2. Features

| | OmniVoice | Ming-3.0-tts 0.5B | Gradium Phonon 40M multilingual |
| --- | --- | --- | --- |
| Cloning from voice profiles | Yes (reference audio + transcript) | Yes (through the `...-Base` alias) | No. `ptts` reports no cloning; Gradium offers it only in its studio |
| Streaming | Per sentence: a whole sentence is rendered before its first chunk leaves | Native inside a sentence: first chunk after 1–2 latents (80–160 ms of audio) | Native inside a sentence: `engine.stream()` chunks are forwarded as produced |
| Output sample rate | 24 kHz | 24 kHz (the engine runs at 44.1 kHz and resamples, streaming) | 24 kHz (native) |
| Hardware | CUDA GPU; 2.6 GB idle / 4.1 GB peak in the 32GB tier | CUDA GPU, bf16 + CUDA graphs; 4.4–4.7 GB, flat in load | CPU only (Rust, GGUF q8 / q4k); 0 VRAM, about 11 of 16 logical cores at saturation |
| License | Apache-2.0 (model and `omnivoice` package) | MIT | Weights, tokenizer and voices CC-BY-NC-4.0 (non-commercial); runtime `ptts` MIT OR Apache-2.0 |
| Languages | 646 language codes in the model card | zh, en, de, es, fr, ko, ja | en, fr, de, es, pt |
| German voices available here | Every voice profile: 55 in this install (clones; the language follows the request) | The same 55 voice profiles | 1 (`default`). Upstream has about 100 catalog voices (`gradium/phonon-voices`), but this runtime never loads the catalog |
| Model load to warm | 48 s (compile) | 12–13 s | 6.6 s (3 s to health) |

## 3. Phonon tuning

Method: q8 or q4k, 32 closed-loop clients for 30 s, voice `default`, `batch_wait_ms` 30, queue 256, one
single-stream pass of 20 sentences first. "Threads" is `phonon_threads`: 0 = 16 logical CPUs / (2 × workers), so
1 thread at 8 workers, 2 at 4 and 4 at 2. Batch = `max_batch_size` = `max_parallel_requests`. CPU is the server
process's mean over the saturation window, in logical cores.

| Quant | Workers | Threads | Batch | Saturation | Sat. TTFA p50 / p95 (ms) | Sat. RTF p50 | CPU (cores) | × RT per core | RSS peak (GB) | 1 stream TTFA p50 / RTF |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q8 | 2 | 0 (4) | 8 | 25.0× | 4547 / 5256 | 1.25 | 5.1 | 4.9 | 1.20 | 62 / 0.063 |
| q8 | 2 | 8 | 8 | 24.9× | 4560 / 5288 | 1.25 | 7.7 | 3.2 | 1.23 | 63 / 0.059 |
| q8 | 2 | 16 | 8 | 17.0× | 6668 / 7765 | 1.82 | 11.9 | 1.4 | 1.24 | 65 / 0.071 |
| q8 | 4 | 0 (2) | 8 | 29.8× ⁶ | 3601 / 4181 | 1.04 | 7.1 | 4.2 | 1.53 | 63 / 0.081 |
| q8 | 4 | 0 (2) | 8 | 25.3× / 25.2× | 4201 / 4901 | 1.22 | 6.6 | 3.8 | 1.53 | 78 / 0.107 |
| q8 | 4 | 8 | 8 | 28.9× ⁶ | 3670 / 4329 | 1.08 | 11.4 | 2.5 | 1.55 | 61 / 0.058 |
| q8 | 4 | 16 | 8 | 22.0× ⁶ | 4818 / 5635 | 1.42 | 13.4 | 1.6 | 1.54 | 68 / 0.074 |
| q8 | 4 | 0 (2) | 16 | 25.2× | 3574 / 4881 | 1.22 | 6.5 | 3.9 | 1.53 | 80 / 0.106 |
| **q4k** | **4** | **0 (2)** | 8 | 26.2× | 4066 / 4737 | 1.19 | 5.7 | 4.6 | 1.50 | 86 / 0.106 |
| **q4k** | **4** | **0 (2)** | **16** | **26.5×** | 3427 / 4653 | 1.18 | **5.7** | 4.7 | 1.48 | 80 / 0.105 |
| q8 | 8 | 0 (1) | 8 | 31.9× ⁶ / 30.7× | 3315 / 3373 | 0.98 | 12.6 / 12.4 | 2.5 | 2.12 | 81 / 0.114 |
| q8 | 8 | 8 | 8 | 31.0× ⁶ | 3407 / 3485 | 1.01 | 14.3 | 2.2 | 2.13 | 50 / 0.058 |
| q8 | 8 | 16 | 8 | 27.3× ⁶ | 3884 / 3941 | 1.15 | 14.5 | 1.9 | 2.12 | 63 / 0.070 |
| q8 | 8 | 0 (1) | 16 | 31.5× | 2577 / 3506 | 1.00 | 12.6 | 2.5 | 2.14 | 82 / 0.126 |
| q8 | 8 | 0 (1) | 32 | 30.1× | 2885 / 3896 | 1.04 | 11.3 | 2.7 | 2.15 | 82 / 0.126 |
| q4k | 8 | 0 (1) | 8 | 31.5× | 3345 / 3404 | 1.02 | 10.5 | 3.0 | 2.09 | 80 / 0.123 |
| **q4k** | **8** | **0 (1)** | **16** | **32.8× / 32.8×** (45 s run: 33.2×) | 2539 / 3378 | 0.97 | **10.7** | 3.1 | 2.06 | 80 / 0.121 |

⁶ First sweep (2026-10-28 late evening). In that sweep the same configs ran faster than in the second sweep
(4–18 %: q8 / 4 workers / threads 0 gave 29.8× vs 25.3× and 25.2×, single-stream RTF 0.081 vs 0.107;
q8 / 8 workers gave 31.9× vs 30.7×). The machine was idle in
both, and the cause is unknown. Compare configs only within one sweep: the rows without ⁶ ran back to back
on 2026-10-29.

What the sweep shows:

- **More threads per engine never helped throughput.** Extra threads only shorten a single request (RTF 0.058 with
  8 threads vs 0.11 with 1, at 8 workers), at the cost of 2–7 more cores and the same or lower throughput. Workers scale; threads don't.
- **q4k vs q8:** about the same throughput (31.5× vs 30.7–31.9× at batch 8), but about 2 cores less CPU
  (10.5 vs 12.4 at 8 workers; 5.7 vs 6.6 at 4), about 15 % less. That is about 20 % more audio per core. WER was not worse (section 4).
- **Batch 16 vs 8:** the serial batch path finishes a whole batch before the next one starts.
  - With batch 8 and 8 workers, each batch ends on its slowest sentence and the other workers sit idle.
  - With batch 16 the free workers pick up the second half, so throughput rose about 4 % (q4k: 31.5× → 32.8×,
    repeated) and saturation TTFA p50 fell by about 25 % (3.3 s → 2.5 s).
  - At 4 workers the throughput did not change, only the TTFA p50 (−15 %).
- **Batch 32 is worse** (30.1×): with only 8 workers, a 32-sentence batch waits for 4 waves before anything new is
  admitted.

**Best overall (dedicated machine):** q4k, 8 workers, threads 0, batch / parallel 16: 32.8× realtime, 10.7 logical
cores (67 % of the machine; 76 % whole-machine load with the bench client), 2.1–2.2 GB RAM. 77 ms TTFA alone, about 100 ms for 4 concurrent streams, 8.3× realtime
per stream.

**Shared machine (about 4 workers):** q4k, 4 workers, threads 0, batch / parallel 16: 26.5× realtime on 5.7
logical cores (about 35 % of the machine), 1.5 GB RAM, 80 ms TTFA alone. Per core it is the most efficient config
of the second sweep (4.7× per core; only q8 / 2 workers in the faster first sweep was higher, 4.9×), and it leaves the GPU services and the desktop about 10 threads.

## 4. German WER (G3_Whisper round trip)

Same 20 sentences and scorer as in [ming-3.0-tts-2026-10.md](ming-3.0-tts-2026-10.md#4-german-wer-g3_whisper-round-trip)
(13 with numbers, dates, units or abbreviations; 7 plain), remote G3_Whisper at 192.168.20.146:7861
(Cohere Transcribe). Phonon with 8 workers and batch 16; voice `default`.

| Engine | Mode | WER all | Numeric | Plain | Exact clips |
| --- | --- | --- | --- | --- | --- |
| Phonon q4k | off | 4.02 % | 6.11 % | 0.00 % | 15 / 20 |
| Phonon q4k | deterministic | 3.02 % | 4.58 % | 0.00 % | 16 / 20 |
| Phonon q8 | off | 5.03 % | 6.11 % | 2.94 % | 14 / 20 |
| Phonon q8 | deterministic | 4.02 % | 4.58 % | 2.94 % | 15 / 20 |
| Ming 32 (reference, 3 voices) | off / deterministic | 4.36 % / 3.52 % | 6.62 % / 5.34 % | 0 / 0 % | 45 / 47 of 60 |
| OmniVoice (reference, 3 voices) | off / deterministic | 4.69 % / 3.18 % | 6.87 % / 4.83 % | 0.49 / 0 % | 41 / 48 of 60 |

For one voice only (SonjaV2), Ming scored 3.5 % / 4.0 % and OmniVoice 5.0 % / 3.5 % (off / deterministic).

Inspecting every Phonon clip with errors:

- **Formatting, not misreadings, in both quants** (6 of the 199 words in mode deterministic, 4 in mode off):
  - `1.250` vs `1250`;
  - `z. B.` vs `zum Beispiel` (mode deterministic only);
  - Whisper wrote `achtundzwanzig` and `dreihundertfünfzig` as words for the Phonon audio. The scorer does not fold
    these compounds back to digits.
- **Real misreadings, mode off:**
  - `Am 3. Oktober 1990` came out as "Am 4. Oktober": wrong day, year dropped (both quants);
  - `z. B.` was read as letters ("Zenz B", "Zenzbegier", both quants).
- **Real misreadings, mode deterministic:** none with q4k.
- **q8 only, both modes:** "Unsere Kinder spielen" came out as "An so Kinderspielen". It is the only difference
  between the quants, a single sentence, so q4k vs q8 is within noise here.

Phonon's German intelligibility is on par with Ming and OmniVoice on these sentences. `ptts` normalizes numbers
itself (`lang=de`), so mode off is not raw text for Phonon. The server's `deterministic` pass still removed the
date and `z. B.` misreadings; the q8-only misreading stays in both modes.

## 5. Which engine for which use

- **Ming: the default for live and streaming use on a GPU.**
  - 30–70 ms TTFA up to 4 streams and about 200 ms for 32 at once.
  - 155× (32 slots) to 221× (64 slots) realtime, on about 1 CPU core and 4.4–4.7 GB VRAM.
  - Clones the voice profiles. Pick the slot count by the concurrency you expect.
- **OmniVoice: fastest single job, widest language coverage.**
  - Turns one request around fastest (21.8× realtime, RTF 0.046), so it suits offline rendering of one long text,
    and it covers 646 languages, plus the Auto-Voice and Voice-Design models.
  - Its TTFA (≥ 200 ms, about 1–3 s under load) and its 54× throughput in the 32GB tier make it the weakest choice
    for many concurrent live streams.
- **Phonon: when no GPU is free, or on a CPU-only host.**
  - About 33× realtime on this 8-core CPU with 0 VRAM: 16 parallel streams still play 2× faster than realtime,
    with about 80–100 ms TTFA up to 4 streams.
  - Limits:
    - one voice (`default`), no cloning;
    - five languages;
    - a non-commercial license (CC-BY-NC-4.0);
    - at saturation it takes about 2/3 of the CPU (10.7 of 16 logical cores).
  - On a machine that also runs the GPU services, use the 4-worker config (26× on about 6 cores).

## 6. Proposals (not applied; defaults, presets and runtime code are unchanged)

- **Phonon defaults:**
  - `phonon_quant` q8 → q4k: same throughput, about 15 % less CPU (about 20 % more audio per core), WER not worse.
  - `phonon_workers` 2 → 4 ("shared machine"). Offer 8 as a "Phonon dedicated CPU" choice.
  - Keep `phonon_threads` 0.
- **Batch settings in phonon mode:** `max_batch_size` / `max_parallel_requests` of about 2 × workers (16 at
  8 workers), not the OmniVoice tier's 8. With the serial batch path this keeps idle workers fed; 4 × workers was
  worse.
- **Runtime (bigger gain):** replace the serial batch path for Phonon with a free-running worker pool. A new
  sentence would start as soon as any engine is free, as Ming's slot queue does.
  - This would remove the wait for a batch's slowest sentence, the bimodal burst-16 TTFA (footnote ¹) and most of the
    2.4 s saturation TTFA.
  - Expected: throughput a little above the batch-16 result; TTFA close to one sentence's render time under load.
  - Not measured.
- **Voices:** load the `gradium/phonon-voices` catalog (about 100 voices, check how many are German) so Phonon
  offers more than one German voice. Cloning is not possible with `ptts`.

## Method

Scripts: `tools/benchmarks/bench_engine_streaming.py` and `tools/benchmarks/bench_wer_roundtrip.py`, described in
[ming-3.0-tts-2026-10.md](ming-3.0-tts-2026-10.md#method).

- **The spawned server:** `--spawn ENGINE` starts it in a throwaway data dir (copied voice profiles, no secrets) on
  port 8191.
- **Load patterns:**
  - NDJSON bursts of 1 / 4 / 16 / 32 (20 / 6 / 3 / 2 rounds; Ming 64 also 64);
  - raw PCM bursts of 1 / 16;
  - then 32 closed-loop clients (Ming 64: 64 clients) for 45 s.
- **Sentences:** single German sentences of 3–6 s of speech from a fixed list of 16.
- **CPU and RAM sampling:** a wrapper around the streaming bench (session scratchpad `bench_c/cpu_wrap.py`, not
  committed) adds psutil sampling every 0.5 s.
  - It covers the server process tree.
  - CPU % is converted to logical cores; the saturation window skips its first 3 s.
  - The wrapper also records working set (RSS) and private bytes.
  - It does not change the bench itself.
- **Order:** one benchmark at a time. Before each run, `nvidia-smi` showed the idle baseline (about 1.09 GB, 0 %)
  and port 8191 had no listener. Every spawned server was stopped by the script.
- **WER:** server spawned with the same Phonon settings, warmed, then `bench_wer_roundtrip.py --voices default
  --modes off,deterministic` against the remote Whisper.
- **Start to warm:** spawn until four warm requests and one burst of 8 have succeeded (`ready_s`). For OmniVoice
  this includes `torch.compile`.

## Caveats

- **Phonon's voice and clip count:** it was measured with its own `default` voice, the other engines with `SonjaV2`.
  - Throughput counts audio seconds, so a different speaking rate does not bias it, but the per-sentence lengths
    differ slightly.
  - Phonon WER rests on 20 clips per mode, against 60 for the other engines; one clip is worth about 0.5–1 %.
- **Phonon run-to-run variation between sessions is larger than on the GPU engines.** The two sweeps differed by
  4–18 % for the same config, for an unknown reason (footnote ⁶). Within one session, repeats agreed within about
  1 % (0.04–0.5 % for the 30 s repeats, 1.3 % for the 45 s run).
- **Sentences:** all speed numbers use short single sentences. For long texts, TTFA depends on the first sentence,
  and the per-second cost stays the same.
- **CPU numbers are means over 0.5 s samples.**
  - The GPU engines' CPU (0.5–1.2 cores) is mostly the Python launch and HTTP threads.
  - OmniVoice showed less CPU at saturation (0.5) than in the 32-burst (1.2), because at saturation it waits on
    full GPU batches.
- **RAM:** private bytes of the GPU engines (5.8–8.2 GB) are commit charge that includes CUDA host mappings. The
  working set (2.4–2.8 GB) is the physical RAM.
- **Phonon at saturation was CPU-bound** (server 10.7 of 16 logical cores, 76 % whole-machine load with the bench
  client). Anything else on the CPU lowers its numbers; nothing
  else ran during these tests.
- **Not measured:**
  - Phonon on other CPUs;
  - Phonon running alongside a GPU engine on the same machine;
  - voice similarity and prosody (nobody listened systematically).

Raw JSON results (not committed), in the session scratchpad:

- **`bench_d/`:**
  - full runs: `full_phonon_q4k_w8_b16.json`, `full_ming_s32.json`, `full_ming_s64.json`, `full_omni32.json`;
  - Phonon sweep: `t_q4k_w8_t0_b8.json`, `t_q4k_w8_t0_b16.json`, `t_q4k_w8_t0_b16_r2.json`, `t_q8_w8_t0_b16.json`,
    `t_q8_w8_t0_b32.json`, `t_q8_w8_t0_b8_r2.json`, `t_q8_w4_t0_b8.json`, `t_q8_w4_t0_b8_r2.json`,
    `t_q8_w4_t0_b16.json`, `t_q4k_w4_t0_b8_a.json`, `t_q4k_w4_t0_b16_a.json`;
  - WER: `wer_phonon_q4k_remote.json`, `wer_phonon_q8_remote.json`;
  - logs: `sweep2.log`, `sweep3.log`, `full.log`.
- **`bench_c/`:** the first sweep, `t_q8_w{2,4,8}_t{0,8,16}.json`.
- **`bench_b/`:** the reference Ming and OmniVoice runs.
