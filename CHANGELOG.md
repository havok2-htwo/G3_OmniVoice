# Changelog

All notable changes to G3_OmniVoice are recorded here.

## [Unreleased]

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
