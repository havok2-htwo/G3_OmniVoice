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

## Notes

- Public demo and synthesis endpoints intentionally stay open for local performance parity.
- Admin endpoints use `X-Admin-Key`.
- Streaming is simulated PCM chunking after sentence/batch generation because OmniVoice does not expose an official native streaming hook.
- Non-streaming output is WAV-first; MP3 is not implemented.
