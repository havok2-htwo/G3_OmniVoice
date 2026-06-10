# G3_OmniVoice API

Base URL for the local server:

```text
http://127.0.0.1:8091
```

Interactive FastAPI docs are available at:

```text
http://127.0.0.1:8091/docs
http://127.0.0.1:8091/openapi.json
```

Public synthesis routes are intentionally open for local demo and performance
testing. Admin routes require either:

```http
X-Admin-Key: <admin-key>
```

or:

```http
Authorization: Bearer <admin-key>
```

The startup batch file prints a temporary startup admin key. Use it before it
expires to rotate/create the persistent admin key. Rotated keys are returned
once and only their hash is stored under `data/`.

## Models

All public model aliases route to the same OmniVoice checkpoint, but they select
different request behavior:

| Model | Task type | Use case |
| --- | --- | --- |
| `k2-fsa/OmniVoice-AutoVoice` | `CustomVoice` | plain automatic voice generation |
| `k2-fsa/OmniVoice-VoiceDesign` | `VoiceDesign` | voice design from `instructions` |
| `k2-fsa/OmniVoice-Base` | `Base` | voice cloning with a saved voice profile or `ref_audio` plus `ref_text` |

## Speech Request

Most synthesis endpoints accept this JSON body:

```json
{
  "input": "Hallo, das ist ein Test.",
  "model": "k2-fsa/OmniVoice-AutoVoice",
  "voice": null,
  "task_type": "CustomVoice",
  "language": "Auto",
  "instructions": null,
  "response_format": "wav",
  "speed": 1.0,
  "stream": false,
  "ref_audio": null,
  "ref_text": null,
  "seed": 1234,
  "metadata": {
    "num_step": 8,
    "guidance_scale": 1.0,
    "duration": null,
    "t_shift": null,
    "denoise": true,
    "preprocess_prompt": true,
    "postprocess_output": true,
    "audio_chunk_duration": null,
    "audio_chunk_threshold": null,
    "position_temperature": null,
    "class_temperature": null
  }
}
```

Useful notes:

- `input` is required.
- `model` defaults to the active runtime model.
- `task_type` is inferred from `model` when omitted.
- `voice` is a saved custom voice id for `Base` clone requests.
- `instructions` is used by `VoiceDesign`; unsupported instruction words are rejected by OmniVoice.
- `ref_audio` is reserved for direct clone integrations; saved voice profiles are the normal UI/API path.
- `response_format` accepts `wav` and `mp3` for non-streaming audio responses.
- Streaming uses PCM chunks for low-latency playback; completed stream jobs can be downloaded as MP3.
- `metadata` overrides runtime generation settings for that request.

## Public Routes

### Health

```http
GET /health
GET /api/health
```

Response:

```json
{ "ok": true }
```

### Voices

```http
GET /api/v1/voices
GET /v1/voices
GET /v1/audio/voices
```

`/api/v1/voices` returns an object:

```json
{
  "voices": [
    {
      "voice_id": "voice_abc123",
      "name": "Demo Voice",
      "source": "custom",
      "created_at": "2026-05-21T00:00:00Z",
      "ref_text": null,
      "filename": null,
      "has_audio": false
    }
  ]
}
```

The `/v1/...` aliases return OpenAI/Open-WebUI-compatible objects:

```http
GET /v1/voices        -> { "object": "list", "data": [...], "voices": [...] }
GET /v1/audio/voices  -> { "voices": [ { "id": "...", "object": "voice", "name": "...", "source": "..." } ] }
```

### Models

```http
GET /api/v1/models
```

Response:

```json
[
  {
    "model_id": "k2-fsa/OmniVoice-AutoVoice",
    "loaded": true,
    "active": true,
    "task_types": ["CustomVoice"]
  }
]
```

OpenAI/Open-WebUI-compatible aliases:

```http
GET /v1/models        -> { "object": "list", "data": [ { "id": "...", "object": "model", ... } ] }
GET /v1/audio/models  -> { "models": [ { "id": "...", "object": "model", ... } ] }
```

### Synthesize Metadata

```http
POST /api/v1/synthesize
Content-Type: application/json
```

This submits a non-streaming job and waits for completion. It returns metadata,
not audio bytes.

```json
{
  "input": "Hallo Welt.",
  "model": "k2-fsa/OmniVoice-AutoVoice",
  "response_format": "wav"
}
```

Response:

```json
{
  "job_id": "job_1770000000_0000",
  "status": "completed",
  "model": "k2-fsa/OmniVoice-AutoVoice",
  "task_type": "CustomVoice",
  "voice": null,
  "sample_rate": 24000,
  "metrics": {
    "queue_wait_ms": 1,
    "model_warm_ms": 0,
    "ttfa_ms": 420,
    "job_wall_ms": 900,
    "audio_duration_ms": 2200,
    "realtime_x": 2.44,
    "output_bytes": 105644,
    "batch_count": 1,
    "max_batch_size_seen": 1,
    "last_batch_size": 1,
    "sentences_total": 1,
    "sentences_rendered": 1
  },
  "audio_url": null,
  "error_message": null
}
```

### Synthesize Event Stream

```http
POST /api/v1/synthesize/stream
Content-Type: application/json
```

Returns newline-delimited JSON with media type `application/x-ndjson`.

Request:

```json
{
  "input": "Erster Satz. Zweiter Satz.",
  "model": "k2-fsa/OmniVoice-AutoVoice",
  "stream": true
}
```

Event types:

```json
{ "type": "start", "job_id": "job_...", "sentence_count": 2, "queue_position": 0 }
{ "type": "batch", "job_id": "job_...", "batch_id": "batch_...", "batch_size": 1, "sentence_index": 0 }
{ "type": "chunk", "job_id": "job_...", "pcm16_b64": "...", "sample_rate": 24000, "duration_ms": 120, "sentence_index": 0 }
{ "type": "done", "job_id": "job_...", "metrics": { "ttfa_ms": 420, "realtime_x": 2.5 } }
{ "type": "error", "message": "..." }
```

### OpenAI-Compatible Audio

```http
POST /v1/audio/speech
Content-Type: application/json
```

Non-streaming requests return audio bytes. Use `response_format="wav"` for
`audio/wav` or `response_format="mp3"` for `audio/mpeg`.

OpenAI clients (e.g. Open WebUI with API base `http://<host>:8091/v1`) work out
of the box: alias models (`tts-1`, `tts-1-hd`, `gpt-4o-mini-tts`, ...) fall back
to the active model, voices resolve case-insensitively by built-in/profile id or
name (standard OpenAI voices like `alloy` map to the default voice), and a
resolved custom voice profile automatically routes to the
`k2-fsa/OmniVoice-Base` voice-clone alias.

```powershell
Invoke-WebRequest `
  -Uri http://127.0.0.1:8091/v1/audio/speech `
  -Method POST `
  -ContentType 'application/json' `
  -Body '{"input":"Hallo Welt.","model":"k2-fsa/OmniVoice-AutoVoice","response_format":"wav"}' `
  -OutFile out.wav
```

When `stream=true`, the endpoint returns raw `audio/pcm` chunks. The sample rate
is 24 kHz, mono, signed 16-bit little-endian PCM.

### Async Jobs

```http
POST /v1/jobs
Content-Type: application/json
```

Submits a job and returns immediately:

```json
{
  "job_id": "job_1770000000_0000",
  "status": "queued",
  "queue_position": 0,
  "eta_ms": 0
}
```

Use admin job routes to inspect or download completed job audio.
Completed public jobs can also be downloaded with:

```http
GET /v1/jobs/{job_id}/audio?format=mp3
```

## Admin Routes

All routes below require `X-Admin-Key` or a bearer token.

### Admin Key

```http
GET /api/admin/keys
POST /api/admin/keys
```

`GET` returns key metadata. `POST` rotates the master admin key and returns the
new token once:

```json
{
  "admin_key": {
    "key_id": "key_abc123",
    "label": "Master Admin Key",
    "created_at": "2026-05-21T00:00:00Z",
    "last_used_at": null
  },
  "token": "omnivoice_tts_..."
}
```

### Settings

```http
GET /api/admin/settings
PATCH /api/admin/settings
```

Patch only the fields you want to change. Persisted runtime settings are stored
under `data/runtime_settings.json`.

Common fields:

```json
{
  "default_model": "k2-fsa/OmniVoice-AutoVoice",
  "model_directory": "X:\\dev\\G3_OmniVoice\\models",
  "allow_model_downloads": true,
  "preferred_device": "cuda:0",
  "attention_implementation": "sdpa",
  "torch_dtype": "float16",
  "compile_model": false,
  "cudagraph_skip_dynamic_graphs": true,
  "max_parallel_requests": 4,
  "max_batch_size": 8,
  "batch_wait_ms": 20,
  "sentence_chunking": true,
  "stream_prebuffer_ms": 0,
  "num_step": 8,
  "guidance_scale": 1.0,
  "vllm_base_url": "http://192.168.20.126:8000",
  "vllm_model": "",
  "whisper_base_url": "http://192.168.0.200:7861",
  "wer_concurrency": 4,
  "wer_transcription_concurrency": 16
}
```

### Dashboard Snapshot And SSE

```http
GET /api/admin/snapshot
GET /api/admin/dashboard/stream
```

`/snapshot` returns current queue, settings, GPU, jobs, voices, models, current
batch and recent batches.

`/dashboard/stream` returns Server-Sent Events:

```text
event: dashboard.snapshot
data: { ...snapshot... }
```

The stream refreshes at `poll_interval_ms` and also on relevant backend events.

### Model Ops

```http
GET  /api/admin/models
GET  /api/admin/vllm/models?base_url=http://192.168.20.126:8000
POST /api/admin/models/download
POST /api/admin/models/delete
POST /api/admin/models/preload
POST /api/admin/models/warmup
POST /api/admin/models/unload
POST /api/admin/models/reload
POST /api/admin/runtime/free-memory
```

Download/delete body:

```json
{
  "model": "k2-fsa/OmniVoice-AutoVoice",
  "storage_path": "X:\\dev\\G3_OmniVoice\\models"
}
```

Preload/warmup/reload body:

```json
{
  "model": "k2-fsa/OmniVoice-VoiceDesign",
  "task_type": "VoiceDesign",
  "language": "Auto",
  "instructions": "female, young adult, american accent"
}
```

`/runtime/free-memory` unloads stale references where possible, runs Python GC,
and trims CUDA cache/reserved memory when the runtime supports it.

### Jobs

```http
GET    /api/admin/jobs
GET    /api/admin/jobs/{job_id}
GET    /api/admin/jobs/{job_id}/audio
DELETE /api/admin/jobs/{job_id}
```

`DELETE` cancels active/queued jobs or removes completed jobs from memory.
`/audio` returns completed WAV bytes by default, or MP3 with `?format=mp3`.

### Voice Library

```http
GET    /api/admin/voices
POST   /api/admin/voices
GET    /api/admin/voices/{voice_id}/audio
DELETE /api/admin/voices/{voice_id}
POST   /api/admin/voices/transcribe
```

Create a voice profile with `multipart/form-data`:

| Field | Required | Description |
| --- | --- | --- |
| `audio_sample` | yes | source audio file |
| `name` | yes | display name |
| `consent` | yes | must be `true` |
| `ref_text` | yes | exact spoken text for OmniVoice clone prompting |

Uploaded samples are normalized to 24 kHz WAV and persisted under `data/voices`.

Transcription accepts `multipart/form-data` with field `file` and forwards it to
the configured Whisper server. It returns:

```json
{ "transcription": "..." }
```

### Traffic Benchmark

```http
POST /api/admin/benchmarks/runs
GET  /api/admin/benchmarks/runs
```

Traffic benchmark request:

```json
{
  "name": "traffic smoke",
  "text": "Erster Satz. Zweiter Satz. Dritter Satz. Vierter Satz.",
  "mode": "traffic",
  "duration_seconds": 60,
  "requests_per_minute": 120,
  "min_sentences_per_request": 1,
  "max_sentences_per_request": 5,
  "completion_timeout_seconds": 180,
  "random_seed": 123,
  "exclusive": true,
  "cases": [
    {
      "label": "auto fp16 step8",
      "request": {
        "model": "k2-fsa/OmniVoice-AutoVoice",
        "response_format": "wav",
        "metadata": { "num_step": 8 }
      }
    }
  ]
}
```

`mode="iterations"` uses `iterations`, `warmup_iterations` and
`parallel_requests` instead of random traffic arrivals. Results include TTFA min,
p50, p95, p99, max, queue wait and realtime metrics.

### WER Benchmark

```http
POST /api/admin/wer-benchmarks/runs
GET  /api/admin/wer-benchmarks/runs
```

WER benchmark request:

```json
{
  "name": "WER seed scan",
  "count": 100,
  "concurrency": 4,
  "transcription_concurrency": 16,
  "vllm_base_url": "http://192.168.20.126:8000",
  "vllm_model": "",
  "whisper_base_url": "http://192.168.0.200:7861",
  "language": "Deutsch",
  "min_words": 5,
  "max_words": 16,
  "tolerance_letters_per_word": 2,
  "completion_timeout_seconds": 180,
  "random_seed": 8882,
  "seed_range": 0,
  "exclusive": true,
  "request": {
    "model": "k2-fsa/OmniVoice-AutoVoice",
    "response_format": "wav",
    "metadata": { "num_step": 8 }
  }
}
```

Behavior:

- vLLM generates sentence pools in chunks of 4 sentences.
- For identical `count`, `language`, word limits and prompt, the sentence pool is cached.
- Changing TTS seed or generation settings does not change cached sentences.
- `seed_range > 0` runs `random_seed` through `random_seed + seed_range` and returns a sorted seed leaderboard.
- Whisper transcriptions are batched by `transcription_concurrency`.
- WER normalization tolerates common German spelling variants, number words vs digits, and joined/split compounds, while still counting real wrong words.

Response summary includes:

```json
{
  "summary": {
    "total": 100,
    "completed": 100,
    "success_count": 100,
    "failure_count": 0,
    "wer_avg": 0.012,
    "wer_p50": 0.0,
    "wer_p95": 0.08,
    "wer_max": 0.18,
    "exact_count": 87,
    "exact_rate": 0.87
  },
  "seed_leaderboard": [
    {
      "seed": 8882,
      "wer_avg": 0.012,
      "wer_p50": 0.0,
      "wer_max": 0.18
    }
  ]
}
```

## Status Codes

Common responses:

| Code | Meaning |
| --- | --- |
| `200` | request accepted or completed |
| `400` | invalid request, unsupported model, missing consent/ref text |
| `401` | missing or invalid admin key |
| `404` | job, voice, audio or API route not found |
| `409` | synthesis job was cancelled |
| `429` | queue saturated |
| `500` | synthesis/runtime failure |

