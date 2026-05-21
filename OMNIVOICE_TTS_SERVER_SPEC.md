# G3_OmniVoice Server Spec

This project is a near 1:1 G3 control-room server with the runtime adapter
replaced by OmniVoice.

## Compatibility Surface

- `GET /health`
- `GET /api/health`
- `GET /api/v1/voices`
- `GET /v1/voices`
- `GET /v1/audio/voices`
- `GET /v1/models`
- `POST /api/v1/synthesize`
- `POST /api/v1/synthesize/stream`
- `POST /v1/audio/speech`
- `POST /v1/jobs`

The public demo and synthesis endpoints are intentionally open for local
performance testing. Admin operations use `X-Admin-Key` or
`Authorization: Bearer <key>`.

## Admin Surface

- `GET /api/admin/keys`
- `POST /api/admin/keys`
- `GET /api/admin/settings`
- `PUT /api/admin/settings`
- `GET /api/admin/snapshot`
- `GET /api/admin/dashboard/stream`
- `GET /api/admin/jobs`
- `GET /api/admin/jobs/{job_id}`
- `GET /api/admin/jobs/{job_id}/audio`
- `DELETE /api/admin/jobs/{job_id}`
- `GET /api/admin/voices`
- `POST /api/admin/voices`
- `DELETE /api/admin/voices/{voice_id}`
- `POST /api/admin/voices/transcribe`
- `POST /api/admin/models/download`
- `POST /api/admin/models/preload`
- `POST /api/admin/models/warmup`
- `POST /api/admin/models/unload`
- `POST /api/admin/models/reload`
- `POST /api/admin/benchmarks/runs`
- `GET /api/admin/benchmarks/runs`

## Runtime Modes

- `mock`: fast sine-wave backend for UI and API tests.
- `omnivoice`: real `k2-fsa/OmniVoice` backend.

Model aliases exposed to the UI:

- `k2-fsa/OmniVoice-AutoVoice`
- `k2-fsa/OmniVoice-VoiceDesign`
- `k2-fsa/OmniVoice-Base`

All aliases load the same underlying `k2-fsa/OmniVoice` checkpoint. The suffix
selects the request translation mode.

## OmniVoice Mapping

- `CustomVoice` / AutoVoice: `model.generate(text=[...], language=[...])`
- `VoiceDesign`: `model.generate(..., instruct=[instructions])`
- `Base`: `model.generate(..., voice_clone_prompt=[...])`

Saved clone voices require explicit consent and `ref_text`. This avoids
unexpected Whisper downloads or extra memory use during synthesis.

## Streaming

OmniVoice does not currently expose an official token/audio streaming generator.
The server preserves the existing stream event contract by emitting PCM chunks
after each sentence/batch completes. Stream events remain:

- `start`
- `batch`
- `chunk`
- `done`
- `error`

For user-facing streams the scheduler uses a first-audio fast path: the first
pending sentence of each compatible streaming request is reserved before larger
follow-up sentence batches. That lowers TTFA without pretending OmniVoice can
emit partial audio from inside a single `generate(...)` call.

## Benchmarks

`POST /api/admin/benchmarks/runs` supports:

- `mode="traffic"`: random user-like arrivals over `duration_seconds`, with
  `requests_per_minute`, random sentence counts, TTFA p50/p95/p99, best/worst,
  queue wait, wall time, and success/failure counts.
- `mode="iterations"`: fixed legacy loop with `iterations` and
  `parallel_requests`.

## Output Format

Non-streaming output is WAV-first. `mp3` is not implemented unless an encoder is
added deliberately.
