# Ming-3.0-tts 0.5B vs OmniVoice: streaming, throughput, VRAM, German WER (2026-10-28)

All numbers were measured by a client over HTTP against the real server (`omnivoice_tts_server.main`),
using the public API that clients use. Nothing was timed in-process, except the VRAM-cap test in section 5.

All three engines (OmniVoice, Ming and the CPU-only Gradium Phonon) are compared side by side, with CPU and RAM,
in [tts-engine-comparison-2026-10.md](tts-engine-comparison-2026-10.md).

## Hardware and software

- One RTX 5090 32 GB (cuda:0), Windows 11 (WDDM), driver baseline about 1.06 GB in use with nothing running.
- `.conda-env`: Python 3.12, torch 2.10+cu130, transformers 5.9.0.
- Repo at `5c259be` (Ming engine with CUDA graphs and batch-invariant slots, Ming queue adapter).
- **Ming** runs bf16 with CUDA graphs, flow_steps 10, cfg 2.0 (both branches every step), sigma 0.25 and
  temperature 0. Output is resampled to 24 kHz (the client contract).
- **OmniVoice** runs with its `32GB VRAM` preset: batch 8, `max_parallel_requests` 8, `batch_wait_ms` 30,
  compile on, packed CFG, cuDNN SDPA, num_step 20.
- Both engines: `max_queue_size` 256, so the benchmark never hits the queue limit.
- Voice for all speed tests: `SonjaV2` (voice profile clone, 24 kHz reference), language `de`.

## Method

Scripts are in `tools/benchmarks/`:

| Script | What it does |
| --- | --- |
| `bench_engine_streaming.py` | Starts a server with the engine and settings given, in a throwaway data dir that holds a copy of the voice profiles and no secrets. Runs the load patterns below and samples `nvidia-smi` every 100 ms. Stops the server at the end. |
| `bench_wer_roundtrip.py` | 20 fixed German sentences × 3 voices × 2 text modes. Synthesizes each via `/v1/audio/speech`, transcribes it with G3_Whisper (`/transcribe/`, Cohere Transcribe) and scores it with the server's own `WerBenchmarkService._wer`. |
| `bench_ming_vram_cap.py` | Runs the Ming engine in-process under `torch.cuda.set_per_process_memory_fraction`, to simulate a 3090 or a shared card. Also scans the Ming code for anything that would not run on sm_86. |

Terms used in the tables:

- **TTFA**: from sending the request to receiving the first non-empty audio bytes, measured on the client.
  - On the NDJSON stream (`/api/v1/synthesize/stream`) that is the first `chunk` event.
  - On the raw stream (`/v1/audio/speech`, `stream: true`, 24 kHz PCM16) it is the first body bytes.
- **Burst of N**: N requests sent at the same instant, repeated 20 / 6 / 3 / 2 times for N = 1 / 4 / 16 / 32. Every request is a single German sentence of about 3–6 s of speech, from a fixed list of 16 sentences.
  - N = 1 runs sequentially.
  - Bursts are the worst case for TTFA. Requests that arrive spread out, joining a busy engine, get first audio after about 40 ms (measured in the I-agent e2e test).
- **Per-request RTF**: wall time of the request divided by the seconds of audio it returned (lower is better).
- **Saturation**: C clients in a closed loop (each sends its next request when the previous one ends) for 45 s. Throughput = audio seconds of all completed requests / wall seconds.
- **VRAM**: whole card `memory.used` from `nvidia-smi`, minus the baseline taken before the server started. WDDM cannot report per-process memory, so nothing else ran during these runs (the WER phase used the remote G3_Whisper). The figures include the CUDA context.

## 1. Headline: Ming vs OmniVoice (RTX 5090, same texts and voice)

| | OmniVoice `32GB VRAM` | Ming 16 slots | Ming 32 slots | Ming 64 slots |
| --- | --- | --- | --- | --- |
| TTFA, 1 stream, p50 / p95 (ms) | 213 / 333 | 28 / 33 | 30 / 38 | 36 / 48 |
| TTFA, burst of 4, p50 / p95 | 409 / 426 | 55 / 56 | 58 / 59 | 69 / 70 |
| TTFA, burst of 16, p50 / p95 | 1074 / 1442 | 110 / 126 | 115 / 131 | 125 / 143 |
| TTFA, burst of 32, p50 / p95 | 1856 / 2986 | 372 / 894 ¹ | 197 / 229 | 201 / 235 |
| Per-request RTF, 1 stream | 0.046 | 0.114 | 0.129 | 0.166 |
| Per-request RTF, burst of 32 (p50) | 0.402 | 0.215 | 0.206 | 0.239 |
| Saturation throughput, audio s / wall s | 54.5x (32 clients) | 93.5x (32) | 154.8x (32) | 220.7x (64) |
| TTFA at saturation, p50 / p95 | 2815 / 3014 ² | 818 / 921 ² | 61 / 87 | 85 / 136 |
| VRAM idle (GB) | 2.56 | 4.25 | 4.38 | 4.65 |
| VRAM peak in the run (GB) | 4.09 | 4.25 | 4.39 | 4.67 |
| Server start to warm (s) | 48 | 12 | 12 | 13 |

¹ More streams than slots: the extra requests wait about one sentence for a free slot.
² More clients than the engine admits at once (OmniVoice: 8 parallel; Ming: 16 slots): this is mostly queueing time.

Reading the table:

- **TTFA:** Ming is 6–9x lower at every load level.
  - Ming streams every chunk as it is generated. The first chunk is 1–2 latents (80–160 ms of audio).
  - OmniVoice renders a whole sentence before its first chunk leaves.
- **Single stream:** OmniVoice is about 2.5x faster at turning around one request (RTF 0.046 vs 0.11–0.17).
  - For streaming clients this does not matter: Ming already plays 6–9x faster than realtime.
- **Under load:** Ming wins. At 32 concurrent streams it has half the per-request RTF and a 15x lower p95 TTFA.
  - Saturation throughput is 2.8x (32 slots) to 4x (64 slots) that of OmniVoice's 32GB preset.

## 2. Ming slot sweep (`ming_max_slots`, `max_parallel_requests` = slots)

NDJSON TTFA p50 / p95 in ms; per-request RTF p50 in brackets:

| Slots | 1 stream | Burst 4 | Burst 16 | Burst 32 | Burst = slots |
| --- | --- | --- | --- | --- | --- |
| 8 | 26 / 35 (0.107) | 53 / 54 (0.120) | 293 / 765 (0.182) ¹ | 908 / 2009 (0.335) ¹ | — |
| 16 | 28 / 33 (0.114) | 55 / 56 (0.127) | 110 / 126 (0.153) | 372 / 894 (0.215) ¹ | — |
| 32 | 30 / 38 (0.129) | 58 / 59 (0.143) | 115 / 131 (0.170) | 197 / 229 (0.206) | — |
| 48 | 32 / 41 (0.144) | 62 / 64 (0.158) | 118 / 134 (0.184) | 201 / 233 (0.213) | 48: 263 / 361 (0.251) |
| 64 | 36 / 48 (0.166) | 69 / 70 (0.182) | 125 / 143 (0.211) | 201 / 235 (0.239) | 64: 376 / 568 (0.312) |
| 64, chunks ≤ 16 latents | 38 / 50 (0.167) | 73 / 75 (0.185) | 132 / 154 (0.216) | 213 / 249 (0.247) | 64: 399 / 584 (0.337) |

Raw 24 kHz PCM (`/v1/audio/speech`), TTFA p50 / p95 in ms:

| Engine | 1 stream | Burst 16 |
| --- | --- | --- |
| Ming 8 | 26 / 27 | 299 / 764 ¹ |
| Ming 16 | 27 / 28 | 109 / 126 |
| Ming 32 | 30 / 31 | 116 / 138 |
| Ming 48 | 32 / 33 | 118 / 132 |
| Ming 64 | 36 / 37 | 126 / 141 |
| OmniVoice 32GB | 198 / 231 | 1100 / 1644 |

The raw PCM path has the same TTFA as NDJSON.

Saturation and VRAM:

| Slots | Clients | Audio s / wall s | Requests in 45 s | TTFA p50 / p95 (ms) | RTF p50 | VRAM idle / peak (GB) |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 32 | 55.9x | 584 | 1965 / 2138 ² | 0.566 | 4.15 / 4.16 |
| 16 | 32 | 93.5x | 958 | 818 / 921 ² | 0.333 | 4.25 / 4.25 |
| 32 | 32 | 154.8x | 1574 | 61 / 87 | 0.202 | 4.38 / 4.39 |
| 48 | 48 | 199.3x | 2029 | 71 / 118 | 0.234 | 4.46 / 4.46 |
| 64 | 64 | 220.7x | 2257 | 85 / 136 | 0.281 | 4.65 / 4.67 |
| 64, chunks ≤ 16 | 64 | 200.1x | 2050 | 93 / 199 | 0.310 | 4.72 / 4.72 |

What the sweep shows:

- **VRAM is flat in the slot count:** +0.5 GB from 8 to 64 slots.
  - Ming allocates its slot KV cache, CUDA graphs and VAE tiles at load, so idle equals peak.
  - The load transient is no higher than idle.
- **Per-row cost grows with slots, even at low load.** The engine runs batch-invariant: every step computes all slots, so a row's audio never depends on its neighbours.
  - Choose the slot count by the concurrency you expect, not "as many as possible".
- **Chunk size:** later chunks of 16 latents (`ming_max_chunk_latents` 16) were slower than 8 on every metric, so every preset keeps 2 / 8.

## 3. Presets set from these numbers (`presets.py`)

| Preset | Slots / parallel | Ming VRAM | Why |
| --- | --- | --- | --- |
| `Ming 32GB VRAM` | 32 / 32 | 4.4 GB | 155x RT, 30 ms TTFA alone, about 200 ms p50 for a burst of 32. Shares the card with the other G3 services. |
| `Ming 24GB VRAM` | 16 / 16 | 4.25 GB | 94x RT, 110 ms TTFA for a burst of 16, lowest per-row RTF that still covers 16 streams. Runs under a 4 GiB torch cap (section 5). |
| `Ming 24+8 VRAM` | 16 / 16 | 4.25 GB | Same as the 24 GB tier on the 3090 (card matching picks the 24 GB card). Fewer slots keep the per-step time short on the slower card. |
| `Ming best performance` | 64 / 64 | 4.7 GB | Highest throughput (221x RT) and the best TTFA under heavy load. The queue limit must be at least 64; it is not a preset key. |

All tiers: CUDA graphs on, 24 kHz output, chunks of 2 then up to 8 latents, `batch_wait_ms` 0.

### VRAM tier budgets: Ming instead of OmniVoice

The other services' peaks come from the 2026-10-16 per-service measurements (Whisper, EMBED, RERANK, DIA). OmniVoice's peaks there were measured with the 14.4 s "Sonja" prompt.

| Tier | OmniVoice peak in the sum | Ming peak | Old sum | Sum with Ming | Target |
| --- | --- | --- | --- | --- | --- |
| 32GB (RTX 5090) | 5.0 GB (batch 8) | 4.4 GB (32 slots) | 27.1 GB | ~26.5 GB | ≤ 30 GB |
| 24GB (RTX 4090 / 3090, one card) | 3.8 GB (batch 4) | 4.25 GB (16 slots) | 21.8 GB | ~22.3 GB | ≤ 23 GB |
| 24+8, the 3090 | 3.8 GB | 4.25 GB (16 slots) | 20.7 GB | ~21.2 GB | ≤ 23 GB |
| Best performance (Ming alone) | — | 4.7 GB (64 slots) | — | 4.7 GB + desktop | — |

- **How Ming compares with OmniVoice:**
  - **Idle:** Ming costs about 1.7–2.1 GB more than OmniVoice (4.25–4.65 vs 2.56 GB) because it allocates everything up front.
  - **Peak:** Ming's peak is about the same as OmniVoice's (4.1 GB with SonjaV2 in this run, 5.0 GB with Sonja at batch 8) and never grows with load or text length.
- **What fits:**
  - The 32GB tier gets about 0.6 GB lower.
  - The 24GB tier gets 0.45 GB higher but stays 0.7 GB under its target.
  - A 32-slot Ming on the 24GB tier would also fit (~22.4 GB). 16 slots were chosen for per-row speed on the slower cards.

## 4. German WER (G3_Whisper round trip)

Setup:

- **Sentences:** 20 sentences. 13 contain numbers, dates, times, units or abbreviations (1990, 1.250 Euro, 14:30 Uhr, Dr., z. B., ICE 578, B 27, 2,5 Kilogramm, 7,99 Euro, GmbH, EU, ...); 7 are plain.
- **Voices:** SonjaV2, SonjaV3, Tom (male). 60 clips per engine and mode.
- **Text modes:** `text_optimization` off (raw text to the engine) and `deterministic` (the server's rule-based numbers-to-words pass first).
- **ASR and scoring:** the G3_Whisper server at 192.168.20.146:7861 (Cohere Transcribe), scored with the admin WER scorer (German folding, number words to digits, 2-letter tolerance).
- **Server settings:** Ming with 32 slots; OmniVoice with its `32GB VRAM` preset.

Corpus WER (sum of word errors / sum of reference words):

| Engine | Mode | WER all | Numeric sentences | Plain sentences | Exact clips | SonjaV2 / SonjaV3 / Tom |
| --- | --- | --- | --- | --- | --- | --- |
| Ming | off | 4.36 % | 6.62 % | 0.00 % | 45 / 60 | 3.5 / 4.0 / 5.5 % |
| Ming | deterministic | 3.52 % | 5.34 % | 0.00 % | 47 / 60 | 4.0 / 4.0 / 2.5 % |
| OmniVoice | off | 4.69 % | 6.87 % | 0.49 % | 41 / 60 | 5.0 / 4.5 / 4.5 % |
| OmniVoice | deterministic | 3.18 % | 4.84 % | 0.00 % | 48 / 60 | 3.5 / 3.5 / 2.5 % |

**Run-to-run variation:**

- **Ming** is deterministic per seed. A first pass with a local G3_Whisper (same Cohere model) gave identical transcripts and numbers.
- **OmniVoice** renders differ from run to run. In the local-Whisper pass it scored 4.36 % (off) and 3.69 % (deterministic), so its ±0.5 % spread is larger than the gap between the engines.

Most of the remaining "errors" are formatting differences between the reference and Whisper's output, the same for both engines:

- `1.250` vs `1250`
- `z. B.` vs `zum Beispiel`
- `Kilogramm` vs `kg`
- `B 27` vs `B27`
- phone digit grouping

Real misreadings found by inspecting every clip with errors:

- **Ming, mode off (4 of 60 clips):**
  - `ICE 578` was read wrong for all three voices (e.g. "7588", an inserted nonsense word, "7 und 5 an 8").
  - `7,99` came out as "7,9" once (Tom).
  - With `deterministic` there were no real misreadings.
  - All 21 plain clips were perfect in both modes.
- **OmniVoice, mode off (about 6 of 60):**
  - garbled phone numbers ("03-Sucht", "03912345567");
  - `ICE 78` and `ICE 85`;
  - `15.` dropped;
  - "Euro Porto" became "Port";
  - a plain sentence with an inserted word ("Kinderspielen seien", SonjaV3).
- **OmniVoice, mode deterministic (2 of 60):** "Poteau" for "Porto", and "IECE".

Ming's German intelligibility is on par with OmniVoice: equal or better with raw text, and within run-to-run noise with `deterministic`.

- **Digit strings:** both engines are weak on raw digit strings such as train and phone numbers.
- **Clients that send digits:** they should keep the server's `deterministic` text optimization on; `text_optimization_default` is an admin setting.
- **What the WER does not measure:** voice similarity and prosody. Nobody listened to the samples systematically.

### 4a. Seed search (2026-10-29): no gain for Ming

Same method as the OmniVoice seed search: the server's own WER benchmark (`POST /api/admin/wer-benchmarks/runs`),
vLLM sentences, G3_Whisper on 192.168.20.146:7861, tolerance 2, text optimization `off` (always off for WER jobs),
preset "Ming 32GB VRAM", voices SonjaV2 and SonjaV3. Ming's default seed is 0.

| Run | Sentences | Seeds | SonjaV2 exact (best / seed 0 / worst) | SonjaV3 exact (best / seed 0 / worst) |
| --- | --- | --- | --- | --- |
| Sweep | 200 everyday sentences | 0–9 | 199 (seeds 1, 3, 6, 9) / 198 / 197 | 200 (seeds 2, 5, 9) / 199 / 198 |
| Check | 200 fresh, with prices, times, dates | 5 per voice | 171 (seed 1) / 167 / 167 (seed 9) | 175 (seed 7) / 174 / 170 (seed 9) |

- **The ranking does not carry over:** seeds 2, 5 and 9 were the best SonjaV3 seeds in the sweep but tied with or
  fell below seed 0 on fresh sentences, and seed 7 went from last place to first. The spread is 1–5 sentences out of
  200 in every run, which is noise.
- **Most errors happen with every seed:** 28 of the 38 (SonjaV2) and 23 of the 35 (SonjaV3) deviating check
  sentences deviate for all five seeds. They are number formats ("14 Uhr 35" transcribed as "14:35 Uhr",
  "24 Euro und 50 Cent" as "24,50 €"; scoring artifacts that hit every engine) and raw dates and large numbers
  (`30.04.2023`, `1250 Euro`), which the `deterministic` text optimization handles in production.
- **Why Ming differs from OmniVoice:** Ming's LLM picks each next latent greedily (temperature 0); the seed only
  drives the flow head's noise (sigma 0.25) per 80 ms latent. Content and pronunciation are decided by the
  deterministic part, so the seed changes the sound slightly but rarely the words. OmniVoice diffuses the whole
  utterance from seeded noise, so a seed can decide whether a word comes out right.
- **Decision:** no Ming seeds in `presets.json`; Ming keeps seed 0. Per-engine voice seeds (`engine_seeds`) stay
  available for voices where a sweep does show a stable winner.
- **First sweep discarded:** the first sweep (25 seeds) and its check ran on sentence pools that partly came from the
  ASCII-folded template pool and prompts (`fuer`, `naechsten`), which Ming reads literally. That was fixed
  (CHANGELOG "WER test sentences with real umlauts") before the runs above. It showed the same picture: 21 of 25
  SonjaV2 seeds tied at 198/200.

## 5. RTX 3090 / shared-card simulation (`bench_ming_vram_cap.py`, in-process)

Under each cap, every slot rendered a 5-sentence German text at once (about 29 s of audio per row, several fragments per row).

| Cap (torch allocator) | Slots | Result | Torch peak allocated / reserved (GB) | Aggregate (5090) |
| --- | --- | --- | --- | --- |
| 24 GiB card (fraction 0.754) | 16 / 32 / 64 | all OK | 3.04 / 3.62, 3.17 / 3.79, 3.44 / 4.07 | 106x / 166x / 253x |
| 4.0 GiB | 16 / 32 / 64 | all OK, identical numbers | same | 103x / 166x / 253x |
| 3.5 GiB | 16 / 32 / 64 | OOM at load (weight fusion in `fast.py` briefly holds both copies) | — | — |

**Architecture check:** a static scan of `omnivoice_tts_server/ming/*.py` found no fp8 dtypes and no imports of flash-attn, triton or FlashInfer. There is no `torch.compile` and no compute-capability gate; the only hit is a comment about cuBLAS behaviour on sm_120.

The engine uses only bf16 GEMMs, torch SDPA (default backend choice), CUDA graphs, cuFFT (ISTFT) and onnxruntime on the CPU (CAM++). All of these run on Ampere (sm_86).

**Not measured: speed on a real 3090.** Expect roughly 2–3x the step time of the 5090, which is why the 3090 tiers use 16 slots.

## Caveats

- **Short sentences only.** All speed numbers use single sentences of 3–6 s. Long texts are split into sentences by the queue (400 characters per sentence for Ming) and then into 24-word fragments by the engine; per-second costs stay the same, but TTFA depends only on the first fragment.
- **Burst TTFA grows with burst size.** Ming prefills new rows one at a time (2.5–5 ms each) to stay batch-invariant. Spread-out arrivals see 30–40 ms.
- **First request after idle.** After about 10 s of idle the GPU drops into a low power state, and the next request takes 100–250 ms to first audio (the max column in the raw JSON). Warm requests are as in the tables.
- **VRAM numbers are whole-card deltas** on WDDM, measured with nothing else running. The desktop baseline (about 1.06 GB) was stable within ±40 MB.
- **The OmniVoice comparison uses its 32GB tier (batch 8) as shipped.** Larger OmniVoice batches give more throughput for more VRAM and latency (earlier tuning saw about 95x RT at batch 16–32); that is not a tier configuration and was not re-measured here.
- **Run-to-run noise:** about ±5 % on throughput. The chunk-16 vs chunk-8 difference (−9 %) is above that, but is a single run.

Raw JSON results (not committed) were in the session scratchpad under `bench_b/`: `ming_s{8,16,32,48,64}.json`, `ming_s64_c16.json`, `omni32.json`, `wer_ming_remote.json`, `wer_omni_remote.json` (local-Whisper pass: `wer_ming.json`, `wer_omni.json`), `cap24card.json`, `cap4.0.json`, `cap35.json`.
