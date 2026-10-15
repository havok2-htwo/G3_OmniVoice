# Changelog

All notable changes to G3_OmniVoice are recorded here.

## [Unreleased]

### Changed - omnivoice 0.1.5 -> upstream master 08be0b4 (0.2.1 core), flex + FlashInfer

- `omnivoice` is pinned to upstream commit `08be0b4` (2026-08-24) through its GitHub
  archive URL. Its inference core is byte-identical to the 0.2.1 release, and it adds
  the opt-in FlashInfer path. With the same seed, 0.1.5 and the new pin rendered
  bit-identical audio for all 30 benchmark utterances (SDPA, bf16, compile, Sonja), so
  default output does not change.
- `attention_implementation: flex_attention` (alias `flex`) now renders correctly.
  - Transformers' flex integration treated OmniVoice's dense bool mask as a +1/+0
    score bias, so padded keys were never masked; this is true in 0.1.5 too. The server
    now converts the mask to a `BlockMask`.
  - On RTX 30/40/50 cards, Inductor's default flex template does not fit in shared
    memory, the compile fails and dynamo silently runs flex eager (~10x slower). The
    server now passes smaller tile sizes there.
  - On the 4090, patched flex matched SDPA on WER and speaker similarity. It was
    roughly as fast (batch 6: 1.59-1.67 s vs 1.76-1.80 s), so SDPA stays the default.
- New setting `enable_flashinfer` (default off; env `OMNIVOICE_TTS_ENABLE_FLASHINFER`).
  - On Linux + CUDA with `flashinfer` installed, the model loads in fp16, gets
    upstream's `apply_flashinfer`, skips `torch.compile`, and must pass a 2-step test
    generation.
  - Anywhere else, or on any failure, the server logs the reason and keeps
    `attention_implementation`.
  - The Dockerfile installs `flashinfer-python[cu13]==0.6.15.post1` +
    `flashinfer-jit-cache==0.6.15.post1+cu130` and enables the setting
    (`--build-arg ENABLE_FLASHINFER=false` to opt out). This is untested on real Linux
    hardware so far.
- `GET /api/admin/settings` reports `attention_backend` / `attention_backend_note`
  (shown in the admin runtime card). `PUT` rejects unknown attention implementations.
- Request `metadata` accepts `pad_duration`, `fade_duration` and `normalize_text`. These
  are new generate() options in omnivoice >= 0.2 and are only forwarded when set.
  `normalize_text` is dropped unless the request has an explicit language and the
  normaliser it needs is installed.
- `requirements-lock.txt` re-synced with the live env: transformers 5.10.2,
  huggingface_hub 1.18.0, gradio 6.17.3 and 13 more drifted pins.
- Dockerfile: fixed a stray `\r` that ended the runtime `ENV` block early, so
  `OPENBLAS_NUM_THREADS` was parsed as an instruction.

### Fixed — review findings on the 0.2.1 upgrade and the text optimizer

- `metadata.normalize_text` is gated on the language omnivoice itself resolves, so
  `de-DE`, `en-US` or unknown tags no longer fail the batch with an ImportError
  (WeTextProcessing missing); the flag is ignored instead. `num2words` is now a
  dependency, so the flag works for de/fr/... (bare integers).
- `attention_implementation: eager` converts OmniVoice's bool mask into an additive
  mask; before, padded keys were attended to (+1/+0 bias) in every CFG batch.
- `metadata.pad_duration` (0-2 s) and `fade_duration` (0-1 s) are validated: `Infinity`,
  huge or non-numeric values return `422` instead of failing or allocating hours of
  audio. An invalid `metadata.text_optimization` (including `"llm"`) is `422` too, like
  the top-level field (was `400`).
- Text optimizer:
  - `Auto` / no language no longer binds Dutch, Polish, Italian, French, Spanish,
    Portuguese, Turkish or Russian text to German because of a `24.09.2026` date or a
    `1.234,56` amount; German/English function words now bind `Auto` text without
    notation (`binding_origin` `text_words` / `other_language`).
  - Only OmniVoice control spans (non-verbal tags, ARPAbet pronunciations) are held
    out; citations, checkboxes and editorial brackets are optimized as in the TS code.
    Placeholders avoid private-use characters already in the text, so one U+F8FF no
    longer disables tag protection.
  - CRLF / CR / U+2028 / U+2029 line breaks are normalised, so a heading followed by a
    Windows blank line stays its own sentence.
  - JS `/i` semantics: Turkish `İ`/`ı`, `ſ`, `K` and `ẞ` no longer fold like Python's
    IGNORECASE; parity with the TS golden still holds for all 6120 records.
  - A 5 s budget bounds the rule passes (number-dense input was quadratic, 12 s for
    6000 characters); on timeout the raw text is spoken (`binding_origin: timeout`).
    Matching releases the GIL.
  - The dry run shares the submit path: empty input and input that optimizes to
    nothing return `400`, rule errors fall back to the raw text. The admin preview
    shows the raw text when the effective mode is `off`.
  - `normalize_text` is dropped only when the optimizer actually rewrote the text.
- `GET /v1/jobs/{job_id}/audio` sends `X-Text-Optimization`, and CORS exposes it.

### Added — spoken-text optimization for German and English

New request field `text_optimization` on `/v1/audio/speech`, `/api/v1/synthesize`,
`/api/v1/synthesize/stream` and `/v1/jobs`. With `"deterministic"` (or `true`) the
server rewrites the input into its spoken form before sentence splitting. Dates,
times, durations, ranges, numbers, currency, units, phone numbers, IPs, versions
and list markers become words; full URLs become their domain; Markdown, emoji and
stack-trace frames are removed; abbreviations are expanded and known acronyms are
spelled.

- The rules are a Python port of the Genesis3 deterministic TTS pipeline
  (`buildDeterministicTtsFallback`, about 140 passes) in
  `backend/src/omnivoice_tts_server/text_optimizer/`. Rule names
  (`applied_rules`) are the same as in Genesis. A small JS-regex layer on the
  `regex` package keeps JavaScript semantics, for example ASCII-only `\d`, `\w`
  and `\b`.
- Parity: the port reproduces the TS outputs for all 6120 golden records (1530
  inputs from the Genesis sweep corpus, the TS unit tests and OmniVoice extras, in
  de/en and floor/final) and all 15347 per-stage records. It also passes the 517
  replayed assertions of the ported TS unit tests.
- Language: an explicit de/en language uses that rule set. `Auto` binds the
  language from language-specific notation in the text. Other languages, and text
  with no such notation, stay unchanged.
- OmniVoice inline tags (`[laughter]`, `[B EY1 S]`) are protected and come back
  verbatim. Blank lines become single newlines, so a heading stays its own
  sentence. `input`, `ref_text` and `instructions` are never rewritten.
- The result is in the synthesize response, the admin job detail and the NDJSON
  `start` event (`text_optimization` with `spoken_text`). `/v1/audio/speech`
  sends an `X-Text-Optimization` header.
- New admin setting `text_optimization_default` (`off` | `deterministic`,
  default `off`). It applies only to public requests without the field.
  Benchmark, WER and finetune jobs stay off, so WER still compares against the
  digit reference.
- Dry run: `POST /api/v1/text/optimize` (API-key gate) and
  `POST /api/admin/text/optimize`. Admin Quick Synthesis has a mode switch and a
  "Text-Vorschau" button.
- Clients that do not send the field get the same sentences as before.
- When the optimizer bound German or English, a `metadata.normalize_text` flag is
  dropped before batching, so omnivoice does not normalise the already spoken text a
  second time.
- `regex` is now a declared dependency (it was already installed via
  `transformers`).

### Fixed — retained job audio and idle RAM

A finished job kept its audio two or three times in RAM for the whole retention
(up to 500 jobs / 7 days). The store was bounded by job count, not by size, so
many long jobs could grow it without limit.

- The PCM parts are freed once `final_audio` is built. Failed and cancelled jobs
  free them too.
- Only the stream queue an endpoint reads is filled: NDJSON events for
  `/api/v1/synthesize/stream`, raw PCM for streaming `/v1/audio/speech`. Before,
  every streaming job also filled the other queue, and nobody drained it. Jobs
  with no stream reader (`/v1/jobs` with `stream: true`, benchmarks) buffer
  nothing. When a stream ends or the client disconnects, the queue is emptied.
- New setting `max_retained_audio_mb` (default 256,
  `OMNIVOICE_TTS_MAX_RETAINED_AUDIO_MB`, `0` = unlimited) caps the audio kept for
  finished jobs. Above the cap, the least recently used jobs (completion or last
  download) lose their audio. The job stays in the job list, and its download
  returns `410 Gone`. The newest job and jobs finished in the last 60 s always
  keep their audio, so waiting callers and poll-then-download clients still get
  theirs.
- The startup warmup runs under `torch.inference_mode()`, like real requests.
  Before, it built autograd state.
- `start_server.bat` and the Dockerfile set `OPENBLAS_NUM_THREADS=4`. numpy/scipy
  OpenBLAS started ~24 threads at import, and each committed 32 MiB it never
  used (~1.5 GB commit, no RAM). Runtime does not change.

### Changed — VRAM budget 7000 MB

The RTX 4090 is shared with Whisper, EMBED and DIA, so the default
`vram_budget_mb` drops from 10000 to 7000. Values saved in
`data/runtime_settings.json` still take precedence.

- At 7000 MB the scheduler caps one batch at ~95 s of estimated audio and one
  sequence at 1587 chars. Before, the caps were ~167 s and 2777 chars.
- Measured on the 4090 (Base + `Sonja`, bf16, compile on, batch 6):
  - Six ~400-char sentences now run as batches of 4 + 2 instead of one batch
    of 6. Peak VRAM on the OmniVoice process fell from 5.9 GB to 4.7 GB, and
    wall time stayed the same (4.0 s for 168 s of audio).
  - The highest peak left is a batch of six ~265-char sentences (111 s of
    audio) at ~5.1 GB.
  - Short texts and mixed concurrent loads did not change.
- The capacity constants (3000 MB + 42 MB per audio second, 0.06 s per char)
  stay conservative. On the 4090 the model is resident at ~2.4 GB and a batch
  adds ~21-24 MB per second of audio, so real peaks stay well below the budget.

### Fixed — per-batch latency and recompile stalls

Measured on the RTX 4090 with Base + `Sonja`, bf16, compile on, trim on:

- The per-batch VRAM trim (`cuda_memory_trim_after_batch`) no longer runs a full
  `gc.collect()`. It walked ~700k tracked objects (~215 ms) on every batch and
  freed no VRAM, because batch tensors are released by refcount. `empty_cache()`
  still runs and returns the reserved pool to the same 2.1 GB as before.
  One short sentence went from ~600 ms to ~290 ms, a batch of 4 from ~1310 ms to
  ~840 ms. Audio output is unchanged. Unload and the admin "free memory" action
  still collect garbage.
- A failed request (unknown voice, missing `ref_text`, ...) no longer resets the
  torch.compile/dynamo caches. The reset made the next request re-trace and
  recompile the LLM (~35 s). Only a CUDA out-of-memory error still resets them.

### Changed — runtime defaults

Fresh installs now start with the tuned production settings. Values already saved
from the admin panel (`data/runtime_settings.json`) still win. Installs that never
saved settings (no such file, e.g. env-only Docker volumes) switch to these
defaults on the next start and need a `Sonja` voice profile or another default voice.

- Default model `k2-fsa/OmniVoice-Base` with default voice `Sonja`.
- `bfloat16`, `compile_model` on, `cuda_memory_trim_after_batch` on,
  `allow_model_downloads` on (also in `start_server.bat`).
- `max_parallel_requests` 8, `max_batch_size` 8, `batch_wait_ms` 0,
  `vram_budget_mb` 10000, `max_input_chars` 25000.
- Generation: `num_step` 18, `guidance_scale` 2.0, `t_shift` 0.1,
  `position_temperature` 5.0, `class_temperature` 0.0, `audio_chunk_duration` 30.1,
  `audio_chunk_threshold` 30.0, `denoise`/`preprocess_prompt`/`postprocess_output` on.
- WER `vllm_model` defaults to `Qwen/Qwen3.6-35B-A3B-FP8`.

### Added — shipped voice preset "Sonja"

- The `Sonja` clone profile (24 kHz reference WAV + ref_text) ships in
  `omnivoice_tts_server/voice_presets/` and is seeded into the voice store on
  startup, once per data dir. A profile already named `Sonja` is left alone, and a
  deleted preset stays deleted (`data/voice_presets_seeded.json`).

### Fixed — default voice for Base

- Base requests without a voice (e.g. OpenAI stock voices like `alloy`) now clone
  the configured default voice profile instead of failing with HTTP 400.
- The built-in `Auto Voice` sent without an explicit model is rendered as AutoVoice
  on the shared checkpoint while Base is active, instead of failing with HTTP 400.
- The admin VRAM estimate used the stale 11.5 GB model base; it now mirrors
  `capacity.py` (3 GB), so the shown peak/limits match what the server enforces.

### Finetune panel — self-generated training data + trainer (admin-only)

Two new admin-gated tabs ("Data Generation" and "Training") for nudging the
OmniVoice model on cases it stumbles on (e.g. German abbreviations like "GmbH").

- **Data Generation:** a persisted domain list (topics), manual or LLM-generated
  ("N more, exclude existing"); per-domain batched sentence generation with dedup;
  a self-generation run that, per (voice, sentence), synthesizes with a seed,
  transcribes via the configured Whisper/ASR server, scores WER, and retries with a
  new seed until it passes or hits `max_attempts` (default 10). Accepted clips land
  under `data/finetune/train/<voice>/` as `clip_NNNN.wav` + matching `.txt`
  (+ `voice_sample_<name>.wav`). Voices per run are selectable: cloned saved voices
  (Base), AutoVoice, or both. A human-eval browser plays clips and deletes wrong
  ones (removing audio + transcript). Reuses the existing vLLM/Whisper clients and
  WER scorer from the WER benchmark.
- **Training:** a CPU/GPU preprocess step (audio → Higgs codec-token cache via the
  bundled `omnivoice` `extract_audio_tokens`), then a full finetune via the bundled
  `omnivoice` trainer launched as a subprocess (`init_from_checkpoint` from the base,
  `sdpa` attention by default, bf16, low LR), with live loss/step/ETA progress.
  Resulting checkpoints can be promoted to `models/Custom-<name>/` and become
  selectable models (`local/Custom-<name>`) loadable via the existing model-ops.
- New admin endpoints under `POST/GET/DELETE /api/admin/finetune/*` (domains,
  sentence/domain generation, generate run + SSE/status, clip browser, preprocess +
  train + status/stream/cancel, checkpoints + promote). New setting
  `OMNIVOICE_TTS_FINETUNE_AUDIO_TOKENIZER` (default `eustlb/higgs-audio-v2-tokenizer`).
- Conservative path adds no new dependencies (`omnivoice`, `accelerate`,
  `webdataset`, `transformers` already present).

### Fixed

- `apiFetch` (frontend) now types its `body` as `unknown`, clearing the
  pre-existing TypeScript errors at JSON-body call sites; `tsc --noEmit` is clean.

## [0.2.0] - 2026-06-10

### Open WebUI / OpenAI compatibility

- `POST /v1/audio/speech` now tolerates stock OpenAI client configs: alias
  models (`tts-1`, `tts-1-hd`, `gpt-4o-mini-tts`, ...) and unknown ids fall
  back to the active model instead of failing at load time; supported model
  ids match case-insensitively.
- Voices resolve case-insensitively against built-in voices and saved voice
  profiles (by id or name). Standard OpenAI voice names (`alloy`, `echo`,
  `fable`, ...) map to the default voice; unknown voices return 404.
- A resolved custom voice profile automatically routes the request to the
  `k2-fsa/OmniVoice-Base` voice-clone alias, so Open WebUI can use saved
  voices without knowing OmniVoice model aliases.
- `GET /v1/audio/models` (new) returns `{ "models": [...] }` and
  `GET /v1/audio/voices` returns `{ "voices": [...] }` — the exact shapes
  Open WebUI reads from custom OpenAI-compatible TTS endpoints.
- `GET /v1/models` now returns the OpenAI list shape
  `{ "object": "list", "data": [...] }`; the previous plain
  `ModelInfo[]` list moved to `GET /api/v1/models` (demo UI updated).
- `GET /v1/voices` returns an OpenAI-style list object with both `data` and
  `voices` keys.

### API

- Added `GET /v1/audio/languages` (+ alias `GET /api/v1/languages`) returning the
  supported synthesis languages (`Auto` + the curated language set), parallel to
  `GET /v1/audio/voices`. Stops clients that probe this endpoint from getting 404s.

### Synthesis language

- Demo and Admin Quick Synthesis now expose a language dropdown (default `Auto`)
  instead of always sending `language: "Auto"`. Options come from the shared
  `SYNTH_LANGUAGE_OPTIONS` set that mirrors `GET /api/v1/languages`.
- Backend now maps the curated UI labels (`Deutsch`, `Français`, …) to the ISO
  codes OmniVoice's `generate(language=...)` resolves (`de`, `fr`, …). Previously
  only `Auto` and `English` took effect; the other localized labels were unknown
  to the model and silently fell back to language-agnostic mode.

### Hardening / Stability

- Alias-reload fix: loaded models are cached on the resolved checkpoint instead
  of the alias, so switching between AutoVoice / VoiceDesign / Base no longer
  reloads identical weights.
- `torch.compile` now uses `dynamic=True` so varying batch sizes no longer
  trigger recompiles.
- Job-retention reaper with bounded eviction (`retention_days` +
  `max_retained_jobs`).
- Streaming endpoints detect client disconnect and cancel the job.
- `wait_for_completion` uses `job_condition` instead of busy-polling.
- CORS `allow_credentials` disabled when the origin is a wildcard.
- `submit()` validation errors map to HTTP 400; only a full queue returns 429.
- Constant-time admin-key comparison.
- `benchmark_runs` capped; background tasks tracked and cancelled on shutdown.
- Favicon requests return 204; uvicorn "Invalid HTTP request received" log
  noise filtered.
- Admin benchmark polling only runs while a run is active and honors
  `poll_interval_ms` (previously a 1s flood).

### Audiobook chunking

- `audio_chunk_threshold` / `audio_chunk_duration` validation caps raised to
  3600s so internal chunking can be effectively disabled for continuous
  prosody.
- `capacity.py` `MODEL_RESIDENT_MB` corrected from 11500 to 3000 (the old value
  was reserved-cache and over-chunked long texts).

### Settings presets

- Save / load / delete named parameter sets via
  `GET/PUT/DELETE /api/admin/settings/presets`, stored in
  `data/settings_presets.json`, with matching admin UI controls.

### VRAM

- `cuda_memory_trim_after_batch` releases the reserved CUDA pool after batches;
  the trim also runs right after the startup warmup so idle VRAM is low
  immediately.
- Startup warmup uses a representative ~20-word sentence.
- `start_server.bat` sets
  `PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256`.
- New capacity module: `vram_budget_mb` + `max_input_chars` derive
  `max_chars_per_chunk` and a per-batch audio-second budget (`capacity.py`).
- Net effect: idle GPU footprint dropped from ~13GB to ~3-4GB.

### fp8

- Experimental fp8 dtype option (`FineGrainedFP8Config`) with a
  kernels-availability gate and automatic bf16 fallback, selectable in the
  admin Dtype dropdown.

### Tooling

- `start_all.bat` launcher starts the backend (:8091) and the frontend dev
  server (:5181) in named windows, with clean window titles.
- Diagnostic benchmark scripts under `tools/benchmarks/`.

### Audio export

- Demo and Admin generated-audio downloads now request MP3 output for easier
  sharing. Backend export supports `response_format="mp3"` and
  `?format=mp3`, using system FFmpeg or `imageio-ffmpeg`.
