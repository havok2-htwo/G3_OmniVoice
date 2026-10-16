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
testing until at least one client API key exists. After that, public synthesis
callers send:

```http
X-API-Key: <client-api-key>
```

Admin routes use username/password login and an httpOnly session cookie:

```http
POST /api/admin/auth/login
```

The default first-run login is `admin` / `admin`, and changing that password is
required before protected admin actions are available.

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
  Besides the settings keys, three omnivoice >= 0.2 options are accepted here and only
  forwarded when set: `pad_duration` and `fade_duration` (seconds of silence padding and
  fade per side, default 0.1 each, as before; `0` gives gapless joins) and
  `normalize_text: true` (upstream text normalisation of the target text).
  `pad_duration` must be a finite number from 0 to 2 and `fade_duration` from 0 to 1;
  anything else (including `Infinity` or a non-numeric string) is rejected with `422`.
  The server forwards `normalize_text` only when the request language is one OmniVoice
  recognises (a code such as `de` / `en`, an English name such as `German`, or a UI
  label such as `Deutsch`). `Auto`, no language and tags OmniVoice does not know
  (`de-DE`, `en-US`) would make upstream guess Chinese/English, so the flag is ignored
  for them. For `en`/`zh` it also needs WeTextProcessing, which has no Windows wheels.
  Other languages use num2words (installed with the server) for bare integers only, so
  German decimals, dates and units stay unchanged. Whenever a requirement is missing the
  flag is ignored instead of failing the request.
  When `text_optimization` rewrote a German or English text (at least one rule
  applied), `normalize_text` is skipped, because the text is in spoken form at that
  point.
- `text_optimization`: `"deterministic"` (or `true`) rewrites the text into its spoken
  form for German/English before sentence splitting: dates, times, numbers, currency,
  units, phone numbers, URLs (domain only), Markdown, emoji, abbreviations, acronyms.
  `"off"` (or `false`) keeps today's behaviour. Omitted means the admin default
  `text_optimization_default` (default `off`) for public requests and always `off`
  for benchmark, WER and finetune jobs. `"llm"` is reserved (`422`, also as
  `metadata.text_optimization`). `Auto` / no language binds de/en from the text's
  notation or its German/English function words; text in other languages (also with
  `24.09.2026`-style dates) and explicit other languages are left unchanged.
  `input`, `ref_text`, `instructions` and inline tags like `[laughter]` are never
  rewritten. OpenAI SDKs pass it via `extra_body` or `metadata.text_optimization`.
  Details: `API_DOCUMENTATION.md`, section "Text optimization".

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
  "error_message": null,
  "text_optimization": null
}
```

`text_optimization` is `{mode, language, binding_origin, applied_rules, spoken_text}`
when the request ran with text optimization.

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

With text optimization the `start` event also carries `text_optimization`
(including `spoken_text`).

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

When text optimization ran, the response has a header like
`X-Text-Optimization: deterministic; language=de; rules=date_de_dmy,time_de_hhmm`.

### Text Optimization Preview

```http
POST /api/v1/text/optimize
Content-Type: application/json
```

```json
{ "input": "Am 24.09.2026 um 14:30 Uhr kostet es 29,90 €.", "language": "Deutsch" }
```

Returns the spoken text, the Genesis3 rule names that fired, the bound language and
its origin, and the sentences a synthesis request would render. Same API-key gate as
synthesis; `POST /api/admin/text/optimize` is the admin-session twin.

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

Finished audio stays in memory up to `max_retained_audio_mb` (default 256 MB,
`OMNIVOICE_TTS_MAX_RETAINED_AUDIO_MB`). Above it the least recently used jobs
lose their audio and the download returns `410 Gone`; the job itself stays listed.

## Admin Routes

Protected routes below require the admin session cookie.

### Admin Auth

```http
POST /api/admin/auth/login
POST /api/admin/auth/logout
GET /api/admin/auth/whoami
POST /api/admin/auth/change-password
```

### Client API Keys

```http
GET /api/admin/api-keys
POST /api/admin/api-keys
DELETE /api/admin/api-keys/{key_id}
```

`POST` creates a client API key and returns the new token once:

```json
{
  "id": "key_abc123",
  "alias": "local-client",
  "created_at": "2026-05-21T00:00:00Z",
  "token": "omnivoice_tts_..."
}
```

### Settings

```http
GET /api/admin/settings
PUT /api/admin/settings
```

Patch only the fields you want to change. Persisted runtime settings are stored
under `data/runtime_settings.json`.

Common fields:

```json
{
  "default_model": "k2-fsa/OmniVoice-Base",
  "default_voice": "Sonja",
  "model_directory": "X:\\dev\\G3_OmniVoice\\models",
  "allow_model_downloads": true,
  "preferred_device": "cuda:0",
  "attention_implementation": "sdpa",
  "enable_flashinfer": false,
  "torch_dtype": "bfloat16",
  "compile_model": true,
  "cudagraph_skip_dynamic_graphs": true,
  "cuda_memory_trim_after_batch": true,
  "max_parallel_requests": 8,
  "max_batch_size": 8,
  "batch_wait_ms": 0,
  "vram_budget_mb": 7000,
  "max_input_chars": 25000,
  "sentence_chunking": true,
  "stream_prebuffer_ms": 0,
  "num_step": 18,
  "guidance_scale": 2.0,
  "vllm_base_url": "http://192.168.20.126:8000",
  "vllm_model": "Qwen/Qwen3.6-35B-A3B-FP8",
  "whisper_base_url": "http://127.0.0.1:7861",
  "wer_concurrency": 4,
  "wer_transcription_concurrency": 16,
  "text_optimization_default": "off"
}
```

`text_optimization_default` (`off` | `deterministic`) is used by public synthesis
requests without their own `text_optimization` field.

### Settings Presets

```http
GET    /api/admin/settings/presets
PUT    /api/admin/settings/presets/{name}
DELETE /api/admin/settings/presets/{name}
```

A preset is a named set of runtime/VRAM settings. Loading one in the admin panel
only fills the settings draft; it takes effect through the normal
`PUT /api/admin/settings`, so presets never bypass validation.

`GET` lists the built-in GPU tiers first, in this order, then saved presets by name:

| Name | Machine | Batch / parallel | `vram_budget_mb` | Process VRAM idle / peak |
| --- | --- | --- | --- | --- |
| `32GB VRAM` | RTX 5090, all G3 services on one card (whole-card peak <= 30 GB) | 8 / 8 | 7000 | ~2.6 / ~5.0 GB |
| `24GB VRAM` | RTX 4090, all G3 services on one card (whole-card peak <= 23 GB) | 4 / 4 | 5000 | ~2.6 / ~3.8 GB |
| `24+8 VRAM` | RTX 3090 (24 GB) + RTX 2070S (8 GB), OmniVoice on the 24 GB card | 4 / 4 | 5000 | ~2.6 / ~3.8 GB |

All tiers: `batch_wait_ms` 30, `cuda_memory_trim_after_batch` on, `num_step` 20,
`compile_model` on, `torch_dtype` `bfloat16`, FlashInfer off; the other preset keys
are the code defaults. The VRAM figures were measured on the RTX 5090 with the 14.4 s
`Sonja` prompt at batch 8 / batch 4 (the shorter `SonjaV2` default needs less).
Cloning throughput is ~35x realtime from batch 4 on and flat above it, so the tiers
stop at 8 and 4.

The number in front names the card OmniVoice runs on; OmniVoice has no `8+24 VRAM`
tier. `24+8 VRAM` sets `preferred_device` to the CUDA card whose total memory is
within ~2 GiB of 24 GB, because machines number their cards differently. If no card
matches, `preferred_device` is left out and `note` says why. Single-card tiers never
set `preferred_device`. The server sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` (unless the
environment sets it), so `cuda:N` is `nvidia-smi` GPU N.

```json
{
  "presets": [
    {
      "name": "24+8 VRAM",
      "built_in": true,
      "description": "3090 (24 GB) + 2070S (8 GB), OmniVoice auf der 24-GB-Karte. Batch 4, VRAM-Budget 5000 MB: ...",
      "target_vram_gb": 24,
      "note": null,
      "created_at": null,
      "values": { "preferred_device": "cuda:0", "torch_dtype": "bfloat16", "max_batch_size": 4, "vram_budget_mb": 5000 }
    }
  ]
}
```

`PUT` body: `{ "values": { ... } }`. The response is the stored preset. Only these keys
are kept, each validated and normalised exactly like `PUT /api/admin/settings`
(`bf16` -> `bfloat16`, `CUDA:1` -> `cuda:1`); everything else (URLs, paths,
voices, theme, unknown keys) is dropped, and `null` values are skipped:

`preferred_device`, `attention_implementation`, `enable_flashinfer`, `torch_dtype`,
`compile_model`, `compile_cudagraphs`, `cudagraph_skip_dynamic_graphs`,
`cuda_memory_trim_after_batch`, `warmup_on_startup`, `max_parallel_requests`,
`max_batch_size`, `batch_wait_ms`, `vram_budget_mb`, `max_input_chars`, `num_step`,
`audio_chunk_duration`, `audio_chunk_threshold`.

Names are trimmed (inner whitespace collapsed), at most 64 characters. Saved presets
are written atomically to `data/settings_presets.json`.

Errors: `400` for a built-in name (save or delete), an empty or too long name, an
invalid value, or a preset without any preset key; `404` when deleting an unknown
preset.

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
  "whisper_base_url": "http://127.0.0.1:7861",
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
| `401` | missing or invalid admin session, or missing/invalid client API key |
| `404` | job, voice, audio or API route not found |
| `409` | synthesis job was cancelled |
| `429` | queue saturated |
| `500` | synthesis/runtime failure |

