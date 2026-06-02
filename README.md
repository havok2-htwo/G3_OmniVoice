# G3_OmniVoice

Windows-first OmniVoice TTS server using the active G3 control-room shape:

- FastAPI backend for synthesis, queueing, admin auth, stats, settings, model ops, and benchmarks
- React/Vite landing page, public demo, and private admin panel
- Same local `X-Admin-Key` browser workflow as the existing G3 stack
- OmniVoice runtime adapter for `k2-fsa/OmniVoice`
- Traffic benchmark mode for random user-like arrivals, sentence-count variation, TTFA percentiles, best/worst, and queue timing
- Streaming requests use a first-audio fast path: the first sentence of each active stream is scheduled before larger follow-up batches

## Project Layout

- `OMNIVOICE_TTS_SERVER_SPEC.md` - local product/API notes
- `docs/API.md` - endpoint reference with request/response examples
- `backend/` - Python service
- `frontend/` - React/Vite UI
- `models/` - default local model directory, ignored by git
- `data/` - runtime settings, admin key hashes, voice profiles, ignored by git
- `.conda-env/` - project-local Conda environment, ignored by git
- `.conda/` - project-local Conda package/cache state, ignored by git

## Quick Install

Run:

```powershell
.\install.bat
```

That workflow creates or reuses:

- local Conda env `.conda-env`
- local Conda package/cache state `.conda`
- CUDA PyTorch
- editable backend package
- frontend dependencies and `frontend/dist`

The script uses `conda create -p .\.conda-env ...` by default. Override paths
with `OMNIVOICE_TTS_CONDA_ENV_DIR`, `OMNIVOICE_TTS_LOCAL_CONDA_HOME`,
`OMNIVOICE_TTS_CONDA_EXE`, or provide a ready interpreter with
`OMNIVOICE_TTS_PYTHON`.

## Start

```powershell
.\start_server.bat
```

`start_server.bat` also sets `PYTORCH_CUDA_ALLOC_CONF`
(`garbage_collection_threshold:0.8,max_split_size_mb:256`) to keep the CUDA
caching-allocator pool small, lowering idle VRAM and leaving room for other
local models.

To start the backend and the frontend dev server together in their own named
windows:

```powershell
.\start_all.bat
```

Open:

- `http://127.0.0.1:8091/`
- `http://127.0.0.1:8091/demo`
- `http://127.0.0.1:8091/admin`

The server prints a temporary startup admin key for emergency browser access.
On a fresh `data/` directory this startup key also becomes the initial persisted
admin key. Rotation stores only a hash under `data/`.

## API

- Human reference: `docs/API.md`
- Interactive docs: `http://127.0.0.1:8091/docs`
- OpenAPI JSON: `http://127.0.0.1:8091/openapi.json`

## Frontend Dev

```powershell
cd frontend
npm install
npm run dev -- --port 5181
```

The dev server proxies `/api` and `/v1` to `http://127.0.0.1:8091`.

## Backend Dev

```powershell
& .\.conda-env\python.exe -m pip install -e .\backend[dev]
& .\.conda-env\python.exe -m pytest backend\tests -q -p no:cacheprovider
```

For first real model download, either place the checkpoint under:

```text
models\OmniVoice
```

or set:

```powershell
$env:OMNIVOICE_TTS_ALLOW_MODEL_DOWNLOADS = 'true'
```

## Operator Features

- **Settings presets** - save, load, and delete named parameter sets from the
  admin panel (`GET/PUT/DELETE /api/admin/settings/presets`, persisted in
  `data/settings_presets.json`).
- **Audiobook chunking controls** - `audio_chunk_threshold` and
  `audio_chunk_duration` accept values up to 3600s, so internal chunking can be
  effectively disabled for continuous prosody on long, single-pass narration.
- **VRAM trim / low idle footprint** - `cuda_memory_trim_after_batch` releases
  the reserved CUDA pool after batches; the trim also runs right after the
  startup warmup, so idle GPU usage stays low (~3-4GB) immediately. A
  `vram_budget_mb` / `max_input_chars` capacity budget derives chunk sizes and a
  per-batch audio-second limit.
- **Experimental fp8 dtype** - selectable in the admin Dtype dropdown
  (`FineGrainedFP8Config`), gated on kernel availability with automatic bf16
  fallback.

## Notes

- Public demo and synthesis endpoints intentionally stay open for local performance parity.
- Admin endpoints use `X-Admin-Key`.
- Streaming is simulated PCM chunking after sentence/batch generation because OmniVoice does not expose an official native streaming hook.
- Non-streaming output is WAV-first; MP3 is not implemented.
