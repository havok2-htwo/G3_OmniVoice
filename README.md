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
- `backend/` - Python service
- `frontend/` - React/Vite UI
- `models/` - default local model directory, ignored by git
- `data/` - runtime settings, admin key hashes, voice profiles, ignored by git

## Quick Install

Run:

```powershell
.\install.bat
```

That workflow creates or reuses:

- Conda env `X:\KI\anaconda3\envs\omnivoice-tts-gui`
- CUDA PyTorch
- editable backend package
- frontend dependencies and `frontend/dist`

## Start

```powershell
.\start_server.bat
```

Open:

- `http://127.0.0.1:8091/`
- `http://127.0.0.1:8091/demo`
- `http://127.0.0.1:8091/admin`

The server prints a temporary startup admin key for emergency browser access.
The default configured admin key in `start_server.bat` is
`mein-geheimer-key-1234`; rotation stores only a hash under `data/`.

## Frontend Dev

```powershell
cd frontend
npm install
npm run dev -- --port 5181
```

The dev server proxies `/api` and `/v1` to `http://127.0.0.1:8091`.

## Backend Dev

```powershell
& X:\KI\anaconda3\envs\omnivoice-tts-gui\python.exe -m pip install -e .\backend[dev]
& X:\KI\anaconda3\envs\omnivoice-tts-gui\python.exe -m pytest backend\tests -q -p no:cacheprovider
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
