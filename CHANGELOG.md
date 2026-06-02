# Changelog

All notable changes to G3_OmniVoice are recorded here.

## [Unreleased] - 2026-06-02

### API

- Added `GET /v1/audio/languages` (+ alias `GET /api/v1/languages`) returning the
  supported synthesis languages (`Auto` + the curated language set), parallel to
  `GET /v1/audio/voices`. Stops clients that probe this endpoint from getting 404s.

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
- Diagnostic benchmark scripts (`bench_*.py`).
