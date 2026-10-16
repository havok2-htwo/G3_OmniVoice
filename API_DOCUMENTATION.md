# API Documentation

This document describes the API surface that is currently implemented in the
repository. If this file and the code diverge, the code wins until the
documentation is updated in the same work step.

Primary server entry point:

- [backend/src/omnivoice_tts_server/main.py](backend/src/omnivoice_tts_server/main.py)

Relevant route and storage implementations:

- [backend/src/omnivoice_tts_server/app.py](backend/src/omnivoice_tts_server/app.py)
- [backend/src/omnivoice_tts_server/api/router_v2.py](backend/src/omnivoice_tts_server/api/router_v2.py)
- [backend/src/omnivoice_tts_server/security.py](backend/src/omnivoice_tts_server/security.py)
- [backend/src/omnivoice_tts_server/domain/state.py](backend/src/omnivoice_tts_server/domain/state.py)
- [backend/src/omnivoice_tts_server/config.py](backend/src/omnivoice_tts_server/config.py)
- [backend/src/omnivoice_tts_server/presets.py](backend/src/omnivoice_tts_server/presets.py)

## Base URL

Default local server URL:

- `http://127.0.0.1:8091`

OpenAPI / interactive docs:

- `GET /docs`
- `GET /openapi.json`

Frontend routes served by the backend when `frontend/dist` exists:

- `GET /`
- `GET /demo`
- `GET /admin`

## Authentication

Open/public routes:

- no API key required until at least one client API key exists
- once client API keys exist, public synthesis callers send `X-API-Key`
- intended for local demo, compatibility clients, and performance testing

Protected admin routes:

- use username/password login via `/api/admin/auth/login`
- store the browser session in an httpOnly cookie
- the default first-run login is `admin` / `admin`
- changing the default password is required before protected admin actions are available

Client API key header for public synthesis routes after keys have been created:

```http
X-API-Key: omnivoice_tts_xxxxxxxxxxxxxxxxxxxxxxxx
```

## Local Storage And Git Safety

Default writable runtime storage is `data/` under the repository root. The path
comes from `Settings.data_dir` and can be overridden with
`OMNIVOICE_TTS_DATA_DIR`.

Persisted local files:

- `data/server_secrets.json`: admin users, sessions, and client API key hashes
- `data/runtime_settings.json`: settings saved via `PUT /api/admin/settings`
- `data/settings_presets.json`: named settings presets
- `data/voice_profiles.json`: custom voice metadata, consent flag, file name, and `ref_text`
- `data/voices/<voice_id>.wav`: normalized 24 kHz WAV samples for custom voices

Presets store only settings values. They do not embed voice audio or
transcriptions.

Currently in-memory only:

- synthesis job history
- completed job WAV bytes
- admin job/audio listings
- `/api/admin/voices/transcribe` transcription responses
- benchmark and WER benchmark runs
- WER generated sentence cache and per-sample transcripts
- OmniVoice prompt cache

Model data is separate:

- default model/cache root is `models/`
- it can be changed with `OMNIVOICE_TTS_MODELS_ROOT_DIR` or the admin settings

GitHub safety:

- `.gitignore` ignores `data/`, `models/`, `uploads/`, `artifacts/`, `.conda/`, and `.conda-env/`
- voice samples, local secrets, presets, runtime settings, and model weights are not synced to GitHub by default
- they would only be committed if someone deliberately force-adds ignored files

## Models

All public model aliases route to the same OmniVoice checkpoint. The alias or
explicit `task_type` selects request behavior:

| Model | Task type | Use case |
| --- | --- | --- |
| `k2-fsa/OmniVoice-AutoVoice` | `CustomVoice` | plain automatic voice generation |
| `k2-fsa/OmniVoice-VoiceDesign` | `VoiceDesign` | voice design from structured `instructions` |
| `k2-fsa/OmniVoice-Base` | `Base` | voice cloning with a saved voice profile or reference audio/text |

Runtime backends:

- `mock`: fast test backend for UI and API tests
- `omnivoice`: real `k2-fsa/OmniVoice` backend

## Speech Request Shape

Most synthesis routes accept this JSON body:

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

Notes:

- `input` is required for synthesis.
- `model` defaults to the active runtime model.
- `task_type` is inferred from the selected model when omitted.
- `voice` is a saved custom voice id for `Base` clone requests.
- `language` is an optional synthesis hint. `Auto` (the default) and any unrecognized value let OmniVoice auto-detect the language from the text. The curated UI labels (`Deutsch`, `English`, `Français`, …) are mapped to OmniVoice language codes (`de`, `en`, `fr`, …) before generation; specifying the language can slightly improve pronunciation/quality. The label list is served by `GET /api/v1/languages` and surfaced as a dropdown in both the Demo and Admin Quick Synthesis UIs.
- `instructions` is used by `VoiceDesign`; invalid instruction terms are rejected.
- `response_format` accepts `wav` and `mp3` for non-streaming audio responses.
- Streaming still uses PCM chunks for low-latency playback; completed stream jobs can be downloaded as MP3.
- `metadata` overrides runtime generation settings for one request.
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
- `text_optimization` turns on spoken-text optimization for German and English (see below).

### Text optimization (`text_optimization`)

OmniVoice reads digits, dates, units and symbols poorly. With
`"text_optimization": "deterministic"` the server rewrites the text into its spoken
form before sentence splitting. The rules are a port of the Genesis3 deterministic
TTS pipeline; no LLM is involved:

- dates and times (`24.09.2026`, `14:30 Uhr`, `3:30 PM`, `12/24/2026`, ISO stamps),
  durations, ranges (`9-17 Uhr` -> `neun bis siebzehn Uhr`)
- numbers, decimals, thousands, ordinals, fractions, percent, currency
  (`29,90 €` -> `neunundzwanzig Euro und neunzig Cent`), units (`5 km`, `2,4 GHz`),
  temperatures, phone numbers, IPs, versions, ports, hex colours, IBANs, list markers
- full URLs are reduced to their domain (`https://heise.de/x` -> `heise punkt de`),
  stack-trace frames are dropped
- Markdown (fences, links, bold, bullets, tables) and emoji are removed
- abbreviations (`z.B.`, `inkl.`, `Tel.`, `e.g.`, ...) are expanded and known
  acronyms are spelled (`API` -> `A P I`)
- `!` at a sentence end becomes `.`, quotation marks are dropped, en/em dashes
  become a spaced hyphen pause. These punctuation changes only happen when at least
  one rule fired; a text no rule touches is spoken exactly as sent.

Values:

| Value | Meaning |
| --- | --- |
| omitted / `null` | admin default `text_optimization_default` (default `off`) for public requests; always `off` for benchmark, WER and finetune jobs |
| `"off"` or `false` | no optimization; the sentences are exactly the same as without the field |
| `"deterministic"` or `true` | deterministic optimization |
| `"llm"` | reserved for a later LLM rewrite mode; rejected with `422` (also as `metadata.text_optimization`) |

Language: an explicit `de*`/`Deutsch`/`German` or `en*`/`English` uses that rule
set; any other explicit language (`fr`, `Français`, ...) leaves the text unchanged.
`Auto` or no language (the OpenAI SDK default) decides from the text:

1. Language-specific notation (`14:30 Uhr`, `24.09.2026`, `29,90 €`, `3:30 PM`, month
   names, ...) binds German or English (`binding_origin: chunk_notation`), unless the
   text's frequent function words belong to another language or the text is mostly in
   a non-Latin script. Dutch, Polish, Italian, French, Spanish, Portuguese, Turkish or
   Russian text with a `24.09.2026` date or a `1.234,56` amount is therefore left
   unchanged (`other_language`).
2. Without such notation, German or English function words (`der`, `und`, `ist`,
   `the`, `and`, `is`, ...) bind the language (`text_words`), so `Das kostet 5 Euro.`
   is optimized with `Auto` too.
3. Otherwise nothing is bound and the text is left unchanged (`none`). Pass an
   explicit `language` when the text is very short or mixed.

Bracketed text is ordinary text: only OmniVoice's own control spans (the non-verbal
tags such as `[laughter]` / `[sigh]` and ARPAbet pronunciations such as `[B EY1 S]`)
are protected, so `[ca. 5 €]` or a citation `[1]` is optimized like the rest.

Limits: the rule passes have a 5-second budget per request. Number-dense input (long
digit lists, `1.1.1...` runs) can exceed it; the raw text is then spoken
(`binding_origin: timeout`). Known gaps inherited from the Genesis rules (kept for
parity with the TS reference): `25:30 min` is read as a ratio (`fünfundzwanzig zu
dreißig`), a lone attributive `1` (`1 Person`, `Ich kaufe 1 Butter`) stays a digit in
German, a keycap `1️⃣` becomes `1`, and a sentence-initial `Am 3.4.` is lower-cased.

What is never changed: `request.input` stays the raw text (job lists show it),
`ref_text` and `instructions` are not touched, and OmniVoice inline control spans
such as `[laughter]` or CMU pronunciations like `[B EY1 S]` pass through verbatim.
`max_input_chars` is checked on the raw input; the spoken text is usually longer, so
it can produce more chunks.

Where the result shows up:

- `POST /api/v1/synthesize` response and `GET /api/admin/jobs/{job_id}`:
  `text_optimization` = `{mode, language, binding_origin, applied_rules, spoken_text}`
  (`null` when the mode was `off`)
- `POST /api/v1/synthesize/stream`: the `start` event carries the same object
- `POST /v1/audio/speech` and `GET /v1/jobs/{job_id}/audio` (for jobs created with
  `POST /v1/jobs`): header
  `X-Text-Optimization: deterministic; language=de; rules=date_de_dmy,time_de_hhmm`.
  CORS exposes it, so browser clients on other origins can read it.

OpenAI SDK clients send the field with `extra_body`, or as
`metadata.text_optimization`, which is read when the top-level field is missing:

```python
client.audio.speech.create(model="tts-1", voice="Sonja", input="Am 24.09. um 14:30 Uhr.",
                           extra_body={"text_optimization": "deterministic", "language": "de"})
```

`POST /api/v1/text/optimize` shows the spoken text without synthesizing.

## Public API

### `GET /health`

Purpose:

- lightweight server health check

Response:

```json
{ "ok": true }
```

### `GET /api/health`

Purpose:

- API-prefixed alias for the same health check

Response:

```json
{ "ok": true }
```

### `GET /api/v1/voices`

Purpose:

- returns built-in and saved voice profiles

Response:

```json
{
  "voices": [
    {
      "voice_id": "voice_abc123",
      "name": "Demo Voice",
      "source": "custom",
      "created_at": "2026-05-21T00:00:00Z",
      "ref_text": "Das ist die Referenz.",
      "filename": "sample.wav.normalized.wav",
      "has_audio": true
    }
  ]
}
```

### `GET /v1/voices`

Purpose:

- OpenAI-style voice list alias

Response shape:

```json
{
  "object": "list",
  "data": [
    { "id": "auto voice", "object": "voice", "name": "Auto Voice", "source": "built-in" }
  ],
  "voices": [
    { "id": "auto voice", "object": "voice", "name": "Auto Voice", "source": "built-in" }
  ]
}
```

### `GET /v1/audio/voices`

Purpose:

- Open WebUI compatibility: voice list for OpenAI-compatible TTS endpoints

Behavior:

- Open WebUI reads the `voices` key from `{base}/audio/voices`

Response shape:

```json
{
  "voices": [
    { "id": "auto voice", "object": "voice", "name": "Auto Voice", "source": "built-in" },
    { "id": "voice_abc123", "object": "voice", "name": "Demo Voice", "source": "custom" }
  ]
}
```

### `GET /api/v1/languages`

Purpose:

- returns supported language labels for synthesis UI/client selection

### `GET /v1/audio/languages`

Purpose:

- compatibility alias for the supported language list

### `GET /api/v1/models`

Purpose:

- lists supported OmniVoice model aliases and current loaded/active state (used by the demo UI)

Response shape:

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

### `GET /v1/models`

Purpose:

- OpenAI-compatible model list

Response shape:

```json
{
  "object": "list",
  "data": [
    {
      "id": "k2-fsa/OmniVoice-AutoVoice",
      "object": "model",
      "created": 0,
      "owned_by": "omnivoice-local",
      "name": "k2-fsa/OmniVoice-AutoVoice",
      "active": true
    }
  ]
}
```

### `GET /v1/audio/models`

Purpose:

- Open WebUI compatibility: model list for OpenAI-compatible TTS endpoints

Behavior:

- Open WebUI reads the `models` key from `{base}/audio/models`

Response shape:

```json
{
  "models": [
    { "id": "k2-fsa/OmniVoice-AutoVoice", "object": "model", "created": 0, "owned_by": "omnivoice-local", "name": "k2-fsa/OmniVoice-AutoVoice", "active": true }
  ]
}
```

### `POST /api/v1/synthesize`

Purpose:

- submits a non-streaming synthesis job
- waits for completion
- returns metadata, not audio bytes

Content type:

- `application/json`

Request:

```json
{
  "input": "Hallo Welt.",
  "model": "k2-fsa/OmniVoice-AutoVoice",
  "response_format": "wav"
}
```

Response shape:

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

With `text_optimization` on, the last field holds
`{"mode": "deterministic", "language": "de", "binding_origin": "request", "applied_rules": ["time_de_hour_only"], "spoken_text": "Um neun Uhr."}`.

### `POST /api/v1/synthesize/stream`

Purpose:

- submits a streaming synthesis job
- returns newline-delimited JSON events
- emits chunks after sentence/batch generation because OmniVoice does not expose native partial audio streaming

Content type:

- request: `application/json`
- response: `application/x-ndjson`

Event examples:

```json
{ "type": "start", "job_id": "job_...", "sentence_count": 2, "queue_position": 0 }
```

With `text_optimization` on, `start` also has
`"text_optimization": {"mode": "deterministic", "language": "de", "binding_origin": "request", "applied_rules": [...], "spoken_text": "..."}`.

```json
{ "type": "batch", "job_id": "job_...", "batch_id": "batch_...", "batch_size": 1, "sentence_index": 0 }
```

```json
{ "type": "chunk", "job_id": "job_...", "pcm16_b64": "...", "sample_rate": 24000, "duration_ms": 120, "sentence_index": 0 }
```

```json
{ "type": "done", "job_id": "job_...", "metrics": { "ttfa_ms": 420, "realtime_x": 2.5 } }
```

```json
{ "type": "error", "message": "..." }
```

### `POST /v1/audio/speech`

Purpose:

- OpenAI-compatible speech endpoint for TTS clients (Open WebUI, OpenAI SDKs)

Content type:

- request: `application/json`

Behavior:

- `stream=false` with `response_format="wav"`: returns `audio/wav`
- `stream=false` with `response_format="mp3"`: returns `audio/mpeg`
- `stream=true`: returns raw `audio/pcm` chunks
- streaming PCM is 24 kHz, mono, signed 16-bit little-endian

OpenAI-client tolerance:

- `model`: exact supported model ids (case-insensitive) are used as-is; OpenAI aliases (`tts-1`, `tts-1-hd`, `gpt-4o-mini-tts`, ...) and unknown ids fall back to the active model
- `voice`: resolved case-insensitively against built-in voices and saved voice profiles (by id or name); standard OpenAI voice names (`alloy`, `echo`, `fable`, ...) fall back to the default voice; anything else returns `404`
- a resolved custom voice profile automatically routes the request to the `k2-fsa/OmniVoice-Base` (voice clone) alias
- the `Authorization: Bearer ...` header sent by OpenAI clients is accepted and ignored (the endpoint stays public, like all synthesis endpoints)
- `text_optimization` (top level via `extra_body`, or `metadata.text_optimization`) is honoured; when it ran, the response has an `X-Text-Optimization` header

Example:

```powershell
Invoke-WebRequest `
  -Uri http://127.0.0.1:8091/v1/audio/speech `
  -Method POST `
  -ContentType 'application/json' `
  -Body '{"input":"Hallo Welt.","model":"k2-fsa/OmniVoice-AutoVoice","response_format":"wav"}' `
  -OutFile out.wav
```

### `POST /v1/jobs`

Purpose:

- submits a synthesis job and returns immediately
- admin job routes can inspect the job and download completed audio

Content type:

- `application/json`

Response:

```json
{
  "job_id": "job_1770000000_0000",
  "status": "queued",
  "queue_position": 0,
  "eta_ms": 0
}
```

### `POST /api/v1/text/optimize`

Purpose:

- dry run of `text_optimization="deterministic"`: returns the spoken text and the
  sentences a synthesis request would render, without synthesizing
- same API-key gate as the synthesis routes; `max_input_chars` applies (`413`)

Request:

```json
{ "input": "Am 24.09.2026 um 14:30 Uhr kostet es 29,90 €.", "language": "Deutsch", "stage": "final" }
```

`stage` is `final` (default, what synthesis uses) or `floor`. `floor` keeps bare
German four-digit numbers and determiner-bound ordinals in digits; it exists for
an LLM rewrite step and is not used for synthesis.

Response:

```json
{
  "text": "Am vierundzwanzigsten September zweitausendsechsundzwanzig um vierzehn Uhr dreißig kostet es neunundzwanzig Euro und neunzig Cent.",
  "applied_rules": ["date_de_dmy", "time_de_hhmm", "currency_suffix"],
  "language": "de",
  "binding_origin": "request",
  "sentences": ["Am vierundzwanzigsten September ... neunzig Cent."]
}
```

`binding_origin` is one of:

- `request`: explicit de/en request language
- `chunk_notation`: bound from German/English notation (`Auto` / no language)
- `text_words`: bound from German/English function words (`Auto` / no language)
- `none`: nothing bound, text unchanged
- `other_language`: `Auto` text written in another language or script, unchanged
- `unsupported_language`: explicit non-de/en language, unchanged
- `timeout`: the rule passes exceeded their 5 s budget, the raw text is spoken
- `error`: a rule raised, the raw text is spoken (the job's `text_optimization`
  also carries `error`)
- `tags_unprotected`: no free placeholder for the inline tags, unchanged

`applied_rules` uses the Genesis3 rule names. The dry run applies the same checks as
synthesis: empty input, or input that optimizes to nothing, returns `400`.

## Admin API

All protected endpoints in this section require a valid admin session cookie.

### `POST /api/admin/auth/login`

Purpose:

- creates an admin browser session

Request:

```json
{
  "username": "admin",
  "password": "admin"
}
```

Response:

```json
{
  "username": "admin",
  "must_change_password": true
}
```

### `POST /api/admin/auth/change-password`

Purpose:

- changes the current admin password and refreshes the session

Request:

```json
{
  "current_password": "admin",
  "new_password": "new-password"
}
```

### `GET /api/admin/auth/whoami`

Purpose:

- returns the current admin username and password-change state

### `POST /api/admin/auth/logout`

Purpose:

- clears the current browser session

### `GET /api/admin/api-keys`

Purpose:

- returns metadata for admin-managed client API keys

### `POST /api/admin/api-keys`

Purpose:

- creates a client API key for public synthesis callers
- returns the plaintext token exactly once

Request:

```json
{
  "alias": "local-client"
}
```

Response:

```json
{
  "id": "key_abc123",
  "alias": "local-client",
  "created_at": "2026-05-21T00:10:00Z",
  "token": "omnivoice_tts_xxxxxxxxxxxxxxxxxxxxxxxx"
}
```

### `DELETE /api/admin/api-keys/{key_id}`

Purpose:

- deletes a client API key

### `GET /api/admin/settings`

Purpose:

- returns current runtime settings

Response fields include:

- model directory, active/default model, default voice
- Whisper/vLLM URLs used by transcription and WER benchmarks
- queue and batching settings
- runtime backend, device, dtype, attention implementation, compile flags
- `enable_flashinfer` and the read-only `attention_backend` / `attention_backend_note`: the
  attention the resident model really runs (`sdpa`, `flex_attention`, `flashinfer`) and why it
  differs from the settings (for example FlashInfer requested on Windows)
- generation defaults such as `num_step`, `guidance_scale`, temperatures, and chunking controls
- `text_optimization_default` (`off` | `deterministic`)

### `PUT /api/admin/settings`

Purpose:

- validates and persists runtime settings
- updates in-memory settings used by the running server

Persistence:

- writes `data/runtime_settings.json`

Common request fields:

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
  "compile_cudagraphs": false,
  "cudagraph_skip_dynamic_graphs": true,
  "cuda_memory_trim_after_batch": true,
  "warmup_on_startup": true,
  "max_parallel_requests": 8,
  "max_batch_size": 8,
  "batch_wait_ms": 0,
  "vram_budget_mb": 7000,
  "max_input_chars": 25000,
  "sentence_chunking": true,
  "stream_chunk_ms": 140,
  "stream_prebuffer_ms": 0,
  "num_step": 18,
  "guidance_scale": 2.0,
  "t_shift": 0.1,
  "position_temperature": 5.0,
  "class_temperature": 0.0,
  "audio_chunk_duration": 30.1,
  "audio_chunk_threshold": 30.0,
  "denoise": true,
  "preprocess_prompt": true,
  "postprocess_output": true,
  "whisper_base_url": "http://127.0.0.1:7861",
  "vllm_base_url": "http://192.168.20.126:8000",
  "vllm_model": "Qwen/Qwen3.6-35B-A3B-FP8",
  "wer_concurrency": 4,
  "wer_transcription_concurrency": 16,
  "text_optimization_default": "off"
}
```

`text_optimization_default` applies to public synthesis requests that do not send
`text_optimization` themselves (env: `OMNIVOICE_TTS_TEXT_OPTIMIZATION_DEFAULT`).
Benchmark, WER and finetune jobs ignore it.

### `POST /api/admin/text/optimize`

Admin-session twin of `POST /api/v1/text/optimize` (same body and response), used by
the "Text-Vorschau" button in Quick Synthesis.

### `GET /api/admin/runtime/devices`

Purpose:

- lists selectable runtime devices
- includes CPU plus available CUDA devices
- marks the configured device as selected

### `GET /api/admin/settings/presets`

Purpose:

- lists saved settings presets

Response:

```json
{
  "presets": [
    {
      "name": "Fast fp8",
      "values": { "torch_dtype": "fp8", "num_step": 8 },
      "created_at": "2026-05-21T00:00:00Z"
    }
  ]
}
```

### `PUT /api/admin/settings/presets/{name}`

Purpose:

- saves a named settings preset

Persistence:

- writes `data/settings_presets.json`

Request:

```json
{
  "values": {
    "torch_dtype": "bfloat16",
    "num_step": 16,
    "guidance_scale": 1.25
  }
}
```

### `DELETE /api/admin/settings/presets/{name}`

Purpose:

- deletes a saved settings preset

Response:

```json
{ "ok": true }
```

### `GET /api/admin/snapshot`

Purpose:

- returns current dashboard state

Response content includes:

- overview metrics
- runtime settings
- model list
- voice list
- recent jobs
- current/recent batches
- runtime and queue status

### `GET /api/admin/dashboard/stream`

Purpose:

- streams dashboard snapshots as Server-Sent Events

Response media type:

- `text/event-stream`

Event example:

```text
event: dashboard.snapshot
data: { ...snapshot... }
```

### `GET /api/admin/models`

Purpose:

- returns local model/cache status for the selected storage root

Query params:

- `storage_path` (optional)

Status values:

- `missing`
- `partial`
- `downloading`
- `ready`
- `error`

### `GET /api/admin/vllm/models`

Purpose:

- asks a vLLM-compatible server for available model ids

Query params:

- `base_url` (optional): defaults to the saved `vllm_base_url`

### `POST /api/admin/models/download`

Purpose:

- enables model downloads
- starts or resumes an OmniVoice model download job

Request:

```json
{
  "model": "k2-fsa/OmniVoice-AutoVoice",
  "storage_path": "X:\\dev\\G3_OmniVoice\\models"
}
```

### `POST /api/admin/models/delete`

Purpose:

- removes cached model files for a managed model from the selected storage root

Request:

```json
{
  "model": "k2-fsa/OmniVoice-AutoVoice",
  "storage_path": "X:\\dev\\G3_OmniVoice\\models"
}
```

### `POST /api/admin/models/preload`

Purpose:

- loads the selected model into memory

Request:

```json
{
  "model": "k2-fsa/OmniVoice-AutoVoice"
}
```

### `POST /api/admin/models/warmup`

Purpose:

- loads the selected model if needed
- runs a representative warmup generation

Request:

```json
{
  "model": "k2-fsa/OmniVoice-VoiceDesign",
  "task_type": "VoiceDesign",
  "language": "Auto",
  "instructions": "female, young adult, american accent"
}
```

### `POST /api/admin/models/unload`

Purpose:

- releases the resident model and trims runtime caches where possible

### `POST /api/admin/models/reload`

Purpose:

- unloads and loads the selected model again

### `POST /api/admin/runtime/free-memory`

Purpose:

- clears prompt cache
- runs Python GC
- resets compile cache where supported
- trims CUDA cache/reserved memory

Response fields:

- `ok`
- `message`
- `memory_before_mb`
- `memory_after_mb`
- `memory_total_mb`
- `released_mb`

### `GET /api/admin/jobs`

Purpose:

- lists current in-memory jobs

Notes:

- jobs are not persisted across restart
- terminal jobs are pruned by retention limits

### `GET /api/admin/jobs/{job_id}`

Purpose:

- returns in-memory job detail
- `input_preview` is the raw input; `text_optimization` (with `spoken_text`) is set
  when the job ran with text optimization

### `GET /api/admin/jobs/{job_id}/audio`

Purpose:

- returns completed job audio

Query params:

- `format=mp3`: returns `audio/mpeg`
- omitted or any other value: returns `audio/wav`

Notes:

- completed audio is kept in memory only
- it is not written under `data/`
- retained audio is capped at `max_retained_audio_mb` (default 256 MB, env
  `OMNIVOICE_TTS_MAX_RETAINED_AUDIO_MB`, `0` = unlimited). Above the cap the least
  recently used finished jobs (completion or last download) lose their audio. The
  job record stays in the job list. The newest job and jobs finished less than
  60 s ago always keep their audio.
- `404 Audio not ready`: the job has no audio yet (queued, running, failed)
- `410 Gone`: the audio was dropped by the byte cap; synthesize again

### `GET /v1/jobs/{job_id}/audio`

Purpose:

- public compatibility route for downloading completed public job audio
- supports the same `format=mp3` query param as the admin job audio route
- same retention rules and `404` / `410` responses as the admin job audio route

### `DELETE /api/admin/jobs/{job_id}`

Purpose:

- cancels queued/active jobs
- removes completed jobs from memory

### `GET /api/admin/voices`

Purpose:

- lists built-in and saved voice profiles with admin details

### `POST /api/admin/voices`

Purpose:

- creates a custom OmniVoice clone profile
- requires explicit consent and exact reference text

Content type:

- `multipart/form-data`

Form fields:

| Field | Required | Description |
| --- | --- | --- |
| `audio_sample` | yes | source audio file |
| `name` | yes | display name |
| `consent` | yes | must be `true` |
| `ref_text` | yes | exact spoken text for OmniVoice clone prompting |

Persistence:

- normalized audio is stored as `data/voices/<voice_id>.wav`
- metadata is stored in `data/voice_profiles.json`

### `DELETE /api/admin/voices/{voice_id}`

Purpose:

- deletes a saved custom voice profile

Persistence:

- rewrites `data/voice_profiles.json`
- removes the profile from the in-memory voice map

### `GET /api/admin/voices/{voice_id}/audio`

Purpose:

- returns the saved normalized WAV sample for a voice profile

### `POST /api/admin/voices/transcribe`

Purpose:

- forwards an uploaded sample to the configured Whisper server
- helps the operator fill `ref_text` for voice cloning

Content type:

- `multipart/form-data`

Form fields:

- `file` (required)

Response:

```json
{
  "transcription": "Das ist die Referenz."
}
```

Persistence:

- response is not stored by this endpoint

### `POST /api/admin/benchmarks/runs`

Purpose:

- creates a traffic or iteration benchmark run

Request:

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

Behavior:

- `mode=traffic` creates random user-like arrivals over the time window
- `mode=iterations` uses `iterations`, `warmup_iterations`, and `parallel_requests`
- results include TTFA, queue wait, wall time, realtime, and success/failure metrics
- benchmark runs are stored in memory only

### `GET /api/admin/benchmarks/runs`

Purpose:

- lists in-memory benchmark runs

### `POST /api/admin/wer-benchmarks/runs`

Purpose:

- generates reference sentences with vLLM
- synthesizes them with OmniVoice
- transcribes the audio with the configured Whisper server
- computes word error rate and optional seed leaderboard

Request:

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

- vLLM generates sentence pools in chunks of 4 sentences
- identical sentence-generation inputs reuse the in-memory sentence cache
- Whisper transcriptions are batched by `transcription_concurrency`
- WER normalization tolerates common German spelling variants, number words vs digits, and joined/split compounds
- WER runs, generated source text, transcripts, and sentence cache are in-memory only

Response summary shape:

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

### `GET /api/admin/wer-benchmarks/runs`

Purpose:

- lists in-memory WER benchmark runs

## Runtime Notes

- OmniVoice aliases share one underlying `k2-fsa/OmniVoice` checkpoint.
- `CustomVoice` calls `model.generate(text=[...], language=[...])`.
- `VoiceDesign` calls `model.generate(..., instruct=[...])`.
- `Base` clone mode calls `model.generate(..., voice_clone_prompt=[...])`.
- Saved clone voices require explicit consent and `ref_text`.
- Non-streaming output can be returned as WAV or MP3.
- MP3 export uses FFmpeg. The app first checks `ffmpeg` on `PATH`, then falls back to the bundled `imageio-ffmpeg` package.
- Streaming is simulated by emitting PCM chunks after sentence/batch completion.
- A first-audio fast path schedules the first sentence of streaming requests before larger follow-up batches.
- `fp8` uses `transformers.FineGrainedFP8Config` when the compatible `kernels` package is installed.
- On Windows, the unsupported DeepGEMM Hub lookup is disabled so fp8 uses the finegrained-fp8 Triton fallback without repeated Hub retries.

## Status Codes

Common responses:

| Code | Meaning |
| --- | --- |
| `200` | request accepted or completed |
| `400` | invalid request, unsupported model, missing consent/ref text |
| `401` | missing or invalid admin session, or missing/invalid client API key |
| `404` | job, voice, audio, model cache, preset, or route not found |
| `409` | synthesis job was cancelled |
| `429` | queue saturated |
| `500` | synthesis/runtime failure |
