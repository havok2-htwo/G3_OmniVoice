# G3 OmniVoice Backend

FastAPI backend for the local OmniVoice stack.

## Runtime Modes

- `OMNIVOICE_TTS_RUNTIME_BACKEND=mock`
- `OMNIVOICE_TTS_RUNTIME_BACKEND=omnivoice`

## Important Environment Variables

- `OMNIVOICE_TTS_PORT`
- `OMNIVOICE_TTS_MODELS_ROOT_DIR`
- `OMNIVOICE_TTS_ALLOW_MODEL_DOWNLOADS`
- `OMNIVOICE_TTS_ACTIVE_MODEL`
- `OMNIVOICE_TTS_DEFAULT_VOICE`
- `OMNIVOICE_TTS_WHISPER_BASE_URL`
- `OMNIVOICE_TTS_WHISPER_PATH`
- `OMNIVOICE_TTS_PREFERRED_DEVICE`
- `OMNIVOICE_TTS_TORCH_DTYPE`

Default model directory:

- `x:\dev\G3_OmniVoice\models`

## Run Tests

Build the frontend first because the static-serving test expects
`frontend/dist`:

```powershell
cd ..\frontend
npm install
npm run build
cd ..\backend
& ..\.conda-env\python.exe -m pytest tests -q -p no:cacheprovider
```

## Current Caveats

- The real runtime needs `omnivoice==0.1.5` plus CUDA PyTorch.
- Saved clone voices require `ref_text`.
- Streaming is simulated chunked PCM, not native OmniVoice token streaming.
- Non-streaming MP3 is not implemented.
