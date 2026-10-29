# Changelog

All notable changes to G3_OmniVoice are recorded here.

## [Unreleased]

### Added - Ming delivery controls for cloned voices

- Synthesis requests take `emotion` (`happy`, `sad`, `angry`, `fearful`, `disgusted`, `surprised`, `neutral`),
  `speaking_rate` (`slow`, `medium`, `fast`), `pitch` and `volume` (`low`, `medium`, `high`). German aliases work
  too (`fröhlich`, `traurig`, `langsam`, `tief`, `laut`, ...), case-insensitive; unknown values are `422`. OpenAI
  SDK clients can send them as `metadata.<field>`. On Ming they become one structured caption (upstream's EMOTION /
  BASIC tasks); given speed / pitch / volume fields are filled with `medium`, emotion alone stays emotion-only.
- On Ming, `instructions` is now the free-form instruction (upstream FREEFORM). It cannot be combined with the
  structured controls (`422`). OmniVoice keeps `instructions` for VoiceDesign.
- `reference_mode`: `full` clones with the reference latents, transcript and speaker embedding (as before),
  `speaker` with the speaker embedding only. `auto` (default) uses `speaker` when a structured control is set,
  else `full` (`AUTO_REFERENCE_MODE_WITH_CONTROLS` in `ming_runtime.py`). Rows with different controls share one
  engine batch and render bit-identically to solo renders.
- Inline tags work across engines: `[Laughter]`, `[Cough]`, `[Breathing]`, `[Uhm]`, ... match case-insensitively,
  are protected from text optimization and are spelled for the active engine (`[Laughter]` on Ming, `[laughter]` on
  OmniVoice). A tag the active engine does not know is removed instead of read aloud. This also runs with
  `text_optimization` off. Glued tags (`[Uhm][Cough]`) are tags too. OmniVoice and Phonon ignore the delivery
  controls.
- Engine switch: a queued job moved to Ming keeps its structured controls and drops a free-form `instructions`
  next to them. A queued sentence left with only tags the new engine does not know is skipped, not read aloud.
- Admin quick synthesis on Ming: a "Ming-Ausdruck" box with emotion, tempo, pitch, volume and reference mode,
  the free instruction (disabled while a structured control is set, so the panel never sends a `422` combination)
  and buttons that insert sound tags at the cursor (German labels, rare ones behind "mehr Laute"). Language and
  seed moved next to model and voice. WER and benchmark runs on Ming do not send the quick-synthesis free
  instruction.
- Measured on SonjaV2, JarvisV2 and Genesis3 (6 German sentences per control, pitch/loudness/duration, CAM++ voice
  similarity, Whisper WER). With the full reference, emotion, pitch, volume and free instructions barely change the
  output; only `fast` shortens it (9-14 %). With `speaker`, emotion clearly changes the delivery (Genesis3 `happy`
  +49 % F0, JarvisV2 `surprised` +40 %), at a voice-similarity cost of about 0.03-0.06. Tempo works in both modes;
  pitch and volume do not reliably. A free-form "whisper" instruction had no measurable effect. WER stayed ~0, but
  one of 216 `speaker` renders ran on to 24.7 s instead of 3 s.

### Fixed - WER scorer counts spoken number formats as equal

- Whisper writes `14:35 Uhr` for a spoken `14 Uhr 35` and `24,50 Euro` for `24 Euro und 50 Cent`, so the scorer
  counted 1-3 errors on sentences every engine read correctly, the same for every seed. Times, prices, `€`, unit
  abbreviations (`kg`, `km`, `cm`, `mm`, `ml`, `km/h`, `%`), ordinals 1-31 (`ersten`, `zwoelften`) and thousands
  dots (`12.500`) now get one form before scoring. In the Ming seed sweeps this cleared 223 of 318 deviating samples;
  the rest are real misreadings (`899` read as `809`). Only the score changes, never the text the engine reads.

### Fixed - WER test sentences with real umlauts

- The template pool that fills a WER sentence pool when vLLM delivers too few unique sentences was written in ASCII
  transliteration (`fuer`, `Verkaeufer`, `naechsten`), and so were the vLLM prompts. The engines read those words
  literally (Ming says "vor" for `fuer`), which inflated the WER of every engine and made seed rankings depend on
  how a seed happens to pronounce `fuer`. Pool and prompts now use real umlauts and ß, and the prompt asks the LLM
  for them explicitly. Sentence pools cached before this change are not reused (cache key version 6).
- A vLLM chunk that adds no new sentence is retried up to 3 times in a row instead of falling back to the template
  pool at once. Runs report `fallback_sentence_count` (admin panel: "Vorlagen-Saetze"), also on a cache hit.

### Added - best seed per engine

- A voice profile stores one best seed per engine: `seed` stays the OmniVoice seed (older profiles, presets and PATCH
  clients unchanged), `engine_seeds` holds the Ming and Phonon seeds. A request or WER run without a seed uses the
  seed of the engine that is active at that moment, so Ming no longer gets the OmniVoice-benchmarked seed. A queued
  job held through an engine switch is given the new engine's seed (a seed the client sent stays).
- `PATCH /api/admin/voices/{id}` takes an optional `engine` (`omnivoice` default, `ming`, `phonon`; unknown is 400);
  voice lists return `engine_seeds` next to `seed`.
- `voice_presets/presets.json` may carry `engine_seeds`; existing installs get each engine's value once in a preset
  voice without a seed for that engine (`data/voice_presets_engine_seeds.json` remembers it, operator values and
  cleared slots win).
- WER runs report the `engine` they render on and the voice profile they render with (`resolved_voice`, fixed at the
  start); an engine switch while a run still renders stops it with an error message instead of mixing two engines
  in one leaderboard.
- Admin panel: one seed field per engine on every voice, the WER voice list shows the active engine's seed, each run
  shows its engine, and every seed leaderboard row has a button that saves that seed for the run's voice and engine.
- Tests: a Ming GPU test renders two voice prompts, German and English and four seeds in one batch and compares
  every row bit-exactly with its solo render; a queue test shows jobs with different voices and seeds share one
  concurrent dispatch.

### Added - Ming-3.0-tts engine, live engine switch

- `runtime_backend=ming` serves inclusionAI Ming-3.0-tts 0.5B next to OmniVoice and Phonon, through the same
  queue and endpoints. It clones the saved voice profiles via the alias `inclusionAI/Ming-3.0-tts-0.5B-Base`,
  streams every chunk the moment it is generated and outputs 24 kHz like the other engines.
- The engine switches without a restart: `POST /api/admin/runtime/backend`, `runtime_backend` in the settings PUT,
  or the new "Engine" box in the admin panel (status of every engine, switch progress). Running jobs finish on the
  old engine (cancelled after `drain_timeout_s`), queued and new requests go to the new one.
- `GET /api/admin/runtime/engines` lists the engines with install/model status.
- `runtime_backend` is saved in `runtime_settings.json`; `OMNIVOICE_TTS_RUNTIME_BACKEND` still wins when set.
  `start_server.bat` and the Dockerfile no longer force `omnivoice`.
- Ming settings `ming_max_slots`, `ming_cuda_graphs`, `ming_output_sample_rate`, `ming_first_chunk_latents`,
  `ming_max_chunk_latents`, `ming_model_dir`, `ming_model_id`, and presets `Ming 32GB VRAM`, `Ming 24GB VRAM`,
  `Ming 24+8 VRAM`, `Ming best performance` (32 / 16 / 16 / 64 slots, measured: docs/benchmarks/ming-3.0-tts-2026-10.md). The OmniVoice tiers are
  unchanged and do not change the engine.
- `install.ps1` installs the backend extra `[ming]` and downloads the Ming weights with `-DownloadMing`.
- Requests naming a model id the active engine does not serve (e.g. an OmniVoice id while Ming runs) are served by
  the active engine's default model instead of failing.

### Fixed - engine side effects

- A Phonon server wrote an empty `voice_profiles.json` at shutdown (the profiles are not loaded in Phonon mode).
- Saving the settings while Phonon ran stored its model id as `active_model`, so the next OmniVoice start ran the
  wrong task type. Engine aliases are never saved now, and an unknown saved `active_model` falls back to
  `k2-fsa/OmniVoice-Base` at startup.
- Finetuning (data generation, training, promotion) returns 409 while another engine than OmniVoice is active, and
  promoted finetune ids no longer leak into another engine's model list.

### Added - WER benchmark per voice, best seed per voice, cancel

- The WER benchmark has a voice selector (default: the voice from the settings). An unknown voice is rejected with 400
  instead of failing every sentence.
- Each custom voice has a "bester Seed" field (,  may be null). A request without
  a seed uses the voice's seed (already the case before); a WER run without a seed now uses it too, and its results
  carry the seed that actually ran.
-  stops a running benchmark: its task ends, its queued and
  running synthesis jobs are cancelled, and the run finishes with status  and the results so far.
- The run card shows the voice and an "abbrechen" button while the run is active.

### Fixed - large WER runs froze the admin dashboard

- The run list carried every per-sample result, so each dashboard poll during a run serialised a 16 x 1000 run
  (16 000 items with transcripts) and the UI rendered a node per deviating sample. The list now omits the per-sample
  results (`results_count` gives the number) and the deviations come from
  `GET /api/admin/wer-benchmarks/runs/{id}/deviations`, worst first, 100 per page, with a "Weitere laden" button.

### Changed - the dashboard's TTFA line is live too

- The TTFA line showed the average over all completed jobs and stayed flat. It now shows the mean time to first audio
  of the jobs that finished in the last 30 s ("TTFA 30 s"), sampled on every dashboard update, and 0 when none
  finished. The status line keeps the overall average.

### Changed - the dashboard's realtime line is a live throughput

- The line under "Realtime" was the average over the last completed jobs, so it stayed flat between jobs. It now
  shows the audio delivered per wall second over the last 30 s ("Durchsatz 30 s"), sampled on every dashboard update
  like the GPU line, and falls to 0 when idle. The status pill keeps the job average.

### Added - WER benchmark per voice, best seed per voice, cancel

- The WER benchmark has a voice selector (default: the voice from the settings). An unknown voice is rejected with 400
  instead of failing every sentence.
- Each custom voice has a "bester Seed" field (`PATCH /api/admin/voices/{id}`, `seed` may be null). A request without
  a seed already uses the voice's seed; a WER run without a seed now uses it too, and its results carry the seed that
  actually ran.
- `POST /api/admin/wer-benchmarks/runs/{run_id}/cancel` stops a running benchmark: its task ends, its queued and
  running synthesis jobs are cancelled, and the run finishes with status `cancelled` and the results so far.
- The run card shows the voice and a cancel button while the run is active.

### Changed - packed CFG and cuDNN attention on by default

- The listening test passed (SonjaV2, Ero2, SonjaV3; all variants clean), so `packed_cfg` and `sdpa_prefer_cudnn`
  are now on in the code defaults and in every VRAM tier. Batch 4 on the RTX 5090: SonjaV2 0.634 -> 0.385 s,
  Ero2 0.904 -> 0.609 s. Existing installs without these keys in `data/runtime_settings.json` get them on the next
  start; both can be switched off in the admin UI.

### Added - SonjaV3 voice preset (3.9 s reference, seed 9301)

- A shorter Sonja clip ("Franz jagt im komplett verwahrlosten Taxi quer durch Bayern.", 3.88 s, 24 kHz mono) ships
  as preset `SonjaV3` with its benchmarked seed 9301. Existing installs get it on the next start (presets are
  seeded once per data dir); the default voice is unchanged.
- Seed search without vLLM (its backend was unreachable): 25 seeds 9300-9324 on the 200 sentences of the SonjaV2
  runs, batch 4, local Whisper, the server's WER with tolerance 2, in both the stock configuration and with
  `packed_cfg` + `sdpa_prefer_cudnn`. 9301 was first in both (199/200 exact each, 0.05 % WER; 9305 had 195/194).
  Confirmed on 100 fresh sentences: 99/100 exact in both configurations.

### Added - a stream's first sentence jumps into the running batch (`stream_first_preempt`, on)

- A streaming request that arrives while a batch renders no longer waits for the batch. Its first sentence is queued
  for the synthesizer, and a forward pre-hook runs it solo at the running batch's next diffusion step boundary (or
  right after the batch if no boundary is left). The running batch's random state and per-item seeds are saved and
  restored around it, so its audio is unchanged; it pauses for the solo generate.
- Measured on the RTX 5090 (32GB tier, batch of 8 running, new sentence after 200 ms): the new sentence was ready
  after 234 ms instead of ~1.1 s (rest of the batch + solo), 189 ms with `packed_cfg` + `sdpa_prefer_cudnn`. The
  running batch took 1142 -> 1314 ms. Both bit-identical to their undisturbed runs.
- Batch results are now matched against the sentences of that batch only, so a job's preempted sentence in flight
  does not fail the job as "Batch result was incomplete".

### Added - optional packed CFG decoding (`packed_cfg`, off by default)

- Stock OmniVoice runs 2B rows per diffusion step, B conditional and B unconditional, all padded to the longest
  conditional row; with an 8 s voice prompt ~31 % of the computed tokens are padding. `packed_cfg` runs one row per
  item, `[conditional | unconditional]`, with a block-diagonal mask and positions restarting at 0 for the
  unconditional part (`omnivoice_tts_server/packed_decode.py`). Same inputs, schedule and sampler calls; a unit test
  checks the layout reproduces the stock loop exactly on a per-row model.
- Measured on the RTX 5090 (24GB tier, other services running), batch 1/4/8: SonjaV2 0.222/0.634/1.223 s ->
  0.146/0.431/0.840 s (-31 to -34 %), Ero2 0.283/0.904/1.800 s -> 0.255/0.731/1.436 s (-10 to -20 %). Together with
  `sdpa_prefer_cudnn`: SonjaV2 0.154/0.385/0.745 s, Ero2 0.215/0.609/1.185 s (-24 to -39 %).
- Not bit-identical, but reproducible across restarts (16/16 identical). Whisper: SonjaV2 0 % WER in every
  variant; Ero2 0.96 % packed vs 4.0 % stock (the difference is the voice's occasional "Oh," onset).

### Changed - the CUDA cache trim runs when the queue runs empty, not after every batch

- `cuda_memory_trim_after_batch` released the CUDA cache after every batch, and the next batch of a busy period had
  to allocate it again: 8 ms per batch at batch 4, 11 ms at batch 8 (eight back-to-back batches, RTX 5090). The trim
  now runs once the queue has no pending sentence left, so VRAM still goes back to the other services after a
  busy period. Audio unchanged.

### Added - optional cuDNN attention (`sdpa_prefer_cudnn`, off by default)

- With OmniVoice's dense bool mask, PyTorch's default SDPA order picks the mem-efficient kernel (this Windows build
  has no flash kernel). The new setting puts cuDNN first at model load. Measured on the RTX 5090 (24GB tier, seed
  9305, other services running): SonjaV2 batch 1/4/8 0.222/0.634/1.223 s -> 0.190/0.528/1.021 s; Ero2
  0.283/0.904/1.800 s -> 0.218/0.708/1.424 s (-15 to -23 %).
- Not bit-identical (bf16 rounding of a different kernel, max 0.004 per attention output). Whisper transcribes all 16
  A/B sentences correctly in both variants.
- torch.compile bakes the kernel into the graph and Inductor's on-disk cache key does not include the SDPA order, so
  the setting also tags the compile cache key (`torch.compiler.config.cache_key_tag`); without that a restart reused
  the cached mem-efficient graph and the setting had no effect. Admin UI: "cuDNN-Attention bevorzugen" (reload needed).

### Changed - silence trimming in numpy, same samples as pydub

- OmniVoice trims every generated item (and every voice-clone reference) with pydub, which measures one RMS window
  per 10 ms step in a Python loop: ~7 ms of CPU per 7 s item, holding the GIL. `omnivoice_tts_server/fast_silence.py`
  is a line-by-line numpy port that keeps pydub's rounding, zero padding and integer RMS, and returns the same samples
  (0 differences over 916 real-voice and edge cases; parity tests in `tests/test_fast_silence.py`). It checks itself
  against pydub when the model loads and stays off if they differ.
- Together with the decode overlap, measured on the RTX 5090 with the other services running, audio bit-identical:
  batch 1 0.228 -> 0.222 s, batch 4 0.653 -> 0.632 s, batch 8 1.259 -> 1.218 s (-3.3 %).

### Changed - codec decode overlaps the post-processing of the previous item

- After the diffusion loop, `generate()` decoded each item, waited for its copy to the host and only then removed
  silences on the CPU, so GPU and CPU took turns. The next item's decode is now launched (with an async copy into
  pinned memory) before the CPU works on the current one (`omnivoice_tts_server/decode_overlap.py`).
- Bit-identical audio (checked for batch 1, 4 and 8, SonjaV2, fixed seed). Measured on the RTX 5090 with the other
  services running: batch 8 1.237 -> 1.207 s (-30 ms, 2.4 %), batch 4 -2.5 ms, batch 1 unchanged.

### Changed - batch_wait_ms is a real coalescing window; GPU stats off the event loop

- `batch_wait_ms` now counts from the arrival of the oldest request that has not had a batch yet, and the worker keeps
  waiting until that deadline or a full plan. Before, the wait restarted when the worker looked and ended at the first
  new arrival, so a burst arriving over a few ms split into two batches (test: three requests 20 ms apart gave batches
  2 + 1, now 3), while a request that had already waited through a running batch waited another 30 ms on top.
- The dashboard snapshot no longer runs nvidia-smi (30-50 ms) on the event loop for every SSE event. The last reading is
  served and refreshed in a background thread when it is older than 1 s.
- Audio is unchanged: both only change when a batch starts, not what it computes.

### Changed - requests with different seeds share a batch

- Seeds no longer split batch groups. A batch whose items ask for one seed (or none) runs the stock sampler exactly as
  before. A batch with mixed seeds draws each item's diffusion noise from that item's own generator
  (`omnivoice_tts_server/row_seeds.py`), so one item's audio depends only on its own seed, not on its batch. Measured
  in process on the 24GB tier, batch of 4: mixed seeds 0.663 s vs same seed 0.665 s, and every item identical to its
  solo run. Through the queue, 16 requests with 16 seeds: 1.27 s (was 1.93 s, every batch of 1); same seed unchanged.
- Note: a seed's audio can differ between a batch with one seed (stock noise stream) and a batch with mixed seeds
  (per-item stream). See the performance notes before using seed benchmarks across batch compositions.

### Changed - clone requests of different voices share a batch

- Base (clone) requests no longer key their batch group on the voice. The batch runner already passes one clone prompt
  per item, so requests for SonjaV2 and Sonja fill the same batch. The seed, language and generation settings still
  separate groups, because one seed is applied per batch. Measured in process on the 24GB VRAM tier (16 concurrent
  requests, batch 4): mixed voices went from 4 batches (1.76 s) to 2 full batches (1.71 s); single-voice load is
  unchanged (1.27 s). Single-request time to first audio, streamed, idle worker: median 0.23 s.

### Changed - fresh installs start with the 24GB VRAM tier

- Without `data/runtime_settings.json` the server now starts with the defaults plus the
  `24GB VRAM` tier (batch / parallel 4, `vram_budget_mb` 5000, `batch_wait_ms` 30) and
  saves those tier keys, so the dashboard shows what runs; the log names the preset.
  Explicit `OMNIVOICE_TTS_<KEY>` env vars beat the tier for their key, and URLs/paths
  are not written, so Docker env values keep applying. Existing installs keep their
  settings. Tests and tools that pass their own `Settings` are not affected.

### Changed - local G3_WHISPER by default

- `whisper_base_url` defaults to `http://127.0.0.1:7861` (was `http://192.168.0.200:7861`), also for
  WER benchmark runs: the G3 suite runs on one box. The Docker stack keeps setting
  `OMNIVOICE_TTS_WHISPER_BASE_URL=http://whisper:7861`; saved settings are unchanged. vLLM stays
  external (`http://192.168.20.126:8000`).

### Fixed - VRAM budget counts the clone prompt

- `vram_budget_mb` is now a real upper bound for the process. The batch planner
  modelled VRAM from generated audio alone and missed that every batch item carries
  the voice's reference prompt. With the 14.4 s `Sonja` prompt the RTX 5090 measured
  5.3 GB under budget 5000, 7.7 GB under 7000 and 11.6 GB under 10000.
  - Each item now costs its estimated speech plus its prompt length. The prompt
    length is the voice profile's audio length, read from the WAV header; silence
    trimming only shortens it by a few ms. AutoVoice and VoiceDesign count 0 s. A
    `ref_audio` that cannot be read counts 15 s.
  - The planner also counts the first item of a batch. Before, it was free.
  - The first-audio batch of streaming requests (their first sentence goes ahead of
    the rest) keeps the same budget. Before, it took up to `max_batch_size` items
    regardless of `vram_budget_mb`; the remaining streams get their first sentence
    in the next batch.
  - New constants, the upper envelope of all measured points (batch 1-32 of 7 s and
    22 s sentences; 0-8 % above the measured peak, never below): 2700 MB resident
    + 15.5 MB per second of generated + prompt audio, 0.07 s of speech per char
    (measured 0.069-0.071). Before: 3000 MB + 42 MB/s, 0.06 s per char.
  - Effect: budget 7000 still fits a batch of 8 normal sentences with `Sonja`
    (~5.4 GB estimated) and with `SonjaV2`. A batch of 32 short sentences is split
    into batches of up to 12 (`Sonja`) or 18 (`SonjaV2`); before, it was 17. 22 s
    sentences go 7 per batch; before, it was 6.
  - `max_chars_per_chunk` (one sequence) keeps the old 42 MB per second, because a
    raised `audio_chunk_threshold` turns off the 30 s internal chunking and long
    unchunked sequences were only measured at that rate. It also reserves 15 s for
    the prompt. At 7000 MB it is now 1248 chars (before: 1587), at 5000 MB 568 (793).
  - The admin VRAM estimate mirrors the new model. It also shows about how many
    ~110-char `SonjaV2` sentences fit in one batch.
- The Base startup warmup clones the configured `default_voice` when that profile
  exists. Before, it used the first stored profile with audio, so it warmed a
  different prompt length than the default voice uses. Without a usable default
  voice it still falls back to the first profile that can clone.

### Added - SonjaV2 default voice and per-voice seeds

- New shipped clone preset `SonjaV2` (`voice_presets/sonjav2.wav`, 8.2 s, transcript =
  the first two sentences of `Sonja`), the default voice of fresh installs
  (`default_voice = "SonjaV2"`). Upstream recommends 3-10 s of reference audio; every
  batch item carries the reference, so the shorter prompt cuts compute and about half of
  the per-item VRAM (~210 MB per item with the 14.4 s `Sonja`, measured on the RTX 5090).
  `Sonja` stays shipped. Existing installs get `SonjaV2` added on their next start; their
  saved `default_voice` is not changed.
- Voice profiles have an optional `seed` (stored in `voice_profiles.json`, listed by the
  voice catalog with details). A request for that voice without its own seed uses it; an
  explicit request seed always wins. Presets in `voice_presets/presets.json` can ship it.
  `SonjaV2` ships `seed: null` until the WER seed benchmark has picked one. Note: the seed
  is applied once per batch, so the best seed depends on `max_batch_size` and `num_step`.
- `num_step` default 18 -> 20.

### Added - built-in GPU tier presets

- `GET /api/admin/settings/presets` lists three read-only built-in tiers first:
  `32GB VRAM` (RTX 5090), `24GB VRAM` (RTX 4090) and `24+8 VRAM` (3090 + 2070S box,
  OmniVoice on the 3090). The names are the same in every G3 service.
  - `24+8 VRAM` sets `preferred_device` to the card whose total memory matches 24 GB
    (within ~2 GiB), wherever that card sits. Without such a card the GPU is left
    alone and the preset carries a `note`.
  - Tier values, measured on the RTX 5090 (`presets.BUILTIN_PRESETS`):
    - `32GB VRAM`: `max_batch_size` / `max_parallel_requests` 8, `vram_budget_mb`
      7000. About 2.6 GB idle and 5 GB peak for the process (batch 8 with `Sonja`).
    - `24GB VRAM` and `24+8 VRAM`: batch / parallel 4, `vram_budget_mb` 5000. About
      2.6 GB idle and 3.8 GB peak (batch 4 with `Sonja`).
    - All tiers: `batch_wait_ms` 30, `cuda_memory_trim_after_batch` on, `num_step`
      20, compile on, `bfloat16`, FlashInfer off. The other keys are the code
      defaults. With `Sonja`, throughput is ~35x realtime from batch 4 on and flat
      above it, so larger batches would only cost VRAM and latency.
  - Saving or deleting under a built-in name returns 400.
- Saved presets now keep only an allowlist of runtime/VRAM keys (`PRESET_KEYS`:
  device, dtype, attention, compile, batching, VRAM budget, `num_step`, audio
  chunking). Each value is validated and normalised like `PUT /api/admin/settings`;
  invalid values return 400. Before, any values were stored unchecked, including
  URLs and paths from the settings draft. Stored presets are filtered the same way on
  load, and the file is written atomically.
- Admin panel: the preset menu groups "GPU-Stufen (eingebaut)" and "Gespeichert".
  "In Entwurf laden" fills the settings draft, and save/delete are disabled for
  built-in tiers. Description and card-match notes are shown below the menu.
- The server sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` when the package is imported (an
  explicit environment value wins), so `cuda:N` matches `nvidia-smi` GPU N.

### Changed - omnivoice 0.1.5 -> upstream master 08be0b4 (0.2.1 core), flex + FlashInfer

- `omnivoice` is pinned to upstream commit `08be0b4` (2026-08-24) through its GitHub
  archive URL. Its inference core is byte-identical to the 0.2.1 release, and it adds
  the opt-in FlashInfer path. With the same seed, 0.1.5 and the new pin rendered
  bit-identical audio for all 30 benchmark utterances (SDPA, bf16, compile, Sonja), so
  default output does not change.
- `attention_implementation: flex_attention` (alias `flex`) now renders correctly.
  - Transformers' flex integration treated OmniVoice's dense bool mask as a +1/+0
    score bias, so padded keys were never masked; this is true in 0.1.5 too. The server
    now converts the mask to a `BlockMask`.
  - On RTX 30/40/50 cards, Inductor's default flex template does not fit in shared
    memory, the compile fails and dynamo silently runs flex eager (~10x slower). The
    server now passes smaller tile sizes there.
  - On the 4090, patched flex matched SDPA on WER and speaker similarity. It was
    roughly as fast (batch 6: 1.59-1.67 s vs 1.76-1.80 s), so SDPA stays the default.
- New setting `enable_flashinfer` (default off; env `OMNIVOICE_TTS_ENABLE_FLASHINFER`).
  - On Linux + CUDA with `flashinfer` installed, the model loads in fp16, gets
    upstream's `apply_flashinfer`, skips `torch.compile`, and must pass a 2-step test
    generation.
  - Anywhere else, or on any failure, the server logs the reason and keeps
    `attention_implementation`.
  - The Dockerfile installs `flashinfer-python[cu13]==0.6.15.post1` +
    `flashinfer-jit-cache==0.6.15.post1+cu130` and enables the setting
    (`--build-arg ENABLE_FLASHINFER=false` to opt out). This is untested on real Linux
    hardware so far.
- `GET /api/admin/settings` reports `attention_backend` / `attention_backend_note`
  (shown in the admin runtime card). `PUT` rejects unknown attention implementations.
- Request `metadata` accepts `pad_duration`, `fade_duration` and `normalize_text`. These
  are new generate() options in omnivoice >= 0.2 and are only forwarded when set.
  `normalize_text` is dropped unless the request has an explicit language and the
  normaliser it needs is installed.
- `requirements-lock.txt` re-synced with the live env: transformers 5.10.2,
  huggingface_hub 1.18.0, gradio 6.17.3 and 13 more drifted pins.
- Dockerfile: fixed a stray `\r` that ended the runtime `ENV` block early, so
  `OPENBLAS_NUM_THREADS` was parsed as an instruction.

### Fixed — review findings on the 0.2.1 upgrade and the text optimizer

- `metadata.normalize_text` is gated on the language omnivoice itself resolves, so
  `de-DE`, `en-US` or unknown tags no longer fail the batch with an ImportError
  (WeTextProcessing missing); the flag is ignored instead. `num2words` is now a
  dependency, so the flag works for de/fr/... (bare integers).
- `attention_implementation: eager` converts OmniVoice's bool mask into an additive
  mask; before, padded keys were attended to (+1/+0 bias) in every CFG batch.
- `metadata.pad_duration` (0-2 s) and `fade_duration` (0-1 s) are validated: `Infinity`,
  huge or non-numeric values return `422` instead of failing or allocating hours of
  audio. An invalid `metadata.text_optimization` (including `"llm"`) is `422` too, like
  the top-level field (was `400`).
- Text optimizer:
  - `Auto` / no language no longer binds Dutch, Polish, Italian, French, Spanish,
    Portuguese, Turkish or Russian text to German because of a `24.09.2026` date or a
    `1.234,56` amount; German/English function words now bind `Auto` text without
    notation (`binding_origin` `text_words` / `other_language`).
  - Only OmniVoice control spans (non-verbal tags, ARPAbet pronunciations) are held
    out; citations, checkboxes and editorial brackets are optimized as in the TS code.
    Placeholders avoid private-use characters already in the text, so one U+F8FF no
    longer disables tag protection.
  - CRLF / CR / U+2028 / U+2029 line breaks are normalised, so a heading followed by a
    Windows blank line stays its own sentence.
  - JS `/i` semantics: Turkish `İ`/`ı`, `ſ`, `K` and `ẞ` no longer fold like Python's
    IGNORECASE; parity with the TS golden still holds for all 6120 records.
  - A 5 s budget bounds the rule passes (number-dense input was quadratic, 12 s for
    6000 characters); on timeout the raw text is spoken (`binding_origin: timeout`).
    Matching releases the GIL.
  - The dry run shares the submit path: empty input and input that optimizes to
    nothing return `400`, rule errors fall back to the raw text. The admin preview
    shows the raw text when the effective mode is `off`.
  - `normalize_text` is dropped only when the optimizer actually rewrote the text.
- `GET /v1/jobs/{job_id}/audio` sends `X-Text-Optimization`, and CORS exposes it.

### Added — spoken-text optimization for German and English

New request field `text_optimization` on `/v1/audio/speech`, `/api/v1/synthesize`,
`/api/v1/synthesize/stream` and `/v1/jobs`. With `"deterministic"` (or `true`) the
server rewrites the input into its spoken form before sentence splitting. Dates,
times, durations, ranges, numbers, currency, units, phone numbers, IPs, versions
and list markers become words; full URLs become their domain; Markdown, emoji and
stack-trace frames are removed; abbreviations are expanded and known acronyms are
spelled.

- The rules are a Python port of the Genesis3 deterministic TTS pipeline
  (`buildDeterministicTtsFallback`, about 140 passes) in
  `backend/src/omnivoice_tts_server/text_optimizer/`. Rule names
  (`applied_rules`) are the same as in Genesis. A small JS-regex layer on the
  `regex` package keeps JavaScript semantics, for example ASCII-only `\d`, `\w`
  and `\b`.
- Parity: the port reproduces the TS outputs for all 6120 golden records (1530
  inputs from the Genesis sweep corpus, the TS unit tests and OmniVoice extras, in
  de/en and floor/final) and all 15347 per-stage records. It also passes the 517
  replayed assertions of the ported TS unit tests.
- Language: an explicit de/en language uses that rule set. `Auto` binds the
  language from language-specific notation in the text. Other languages, and text
  with no such notation, stay unchanged.
- OmniVoice inline tags (`[laughter]`, `[B EY1 S]`) are protected and come back
  verbatim. Blank lines become single newlines, so a heading stays its own
  sentence. `input`, `ref_text` and `instructions` are never rewritten.
- The result is in the synthesize response, the admin job detail and the NDJSON
  `start` event (`text_optimization` with `spoken_text`). `/v1/audio/speech`
  sends an `X-Text-Optimization` header.
- New admin setting `text_optimization_default` (`off` | `deterministic`,
  default `off`). It applies only to public requests without the field.
  Benchmark, WER and finetune jobs stay off, so WER still compares against the
  digit reference.
- Dry run: `POST /api/v1/text/optimize` (API-key gate) and
  `POST /api/admin/text/optimize`. Admin Quick Synthesis has a mode switch and a
  "Text-Vorschau" button.
- Clients that do not send the field get the same sentences as before.
- When the optimizer bound German or English, a `metadata.normalize_text` flag is
  dropped before batching, so omnivoice does not normalise the already spoken text a
  second time.
- `regex` is now a declared dependency (it was already installed via
  `transformers`).

### Fixed — retained job audio and idle RAM

A finished job kept its audio two or three times in RAM for the whole retention
(up to 500 jobs / 7 days). The store was bounded by job count, not by size, so
many long jobs could grow it without limit.

- The PCM parts are freed once `final_audio` is built. Failed and cancelled jobs
  free them too.
- Only the stream queue an endpoint reads is filled: NDJSON events for
  `/api/v1/synthesize/stream`, raw PCM for streaming `/v1/audio/speech`. Before,
  every streaming job also filled the other queue, and nobody drained it. Jobs
  with no stream reader (`/v1/jobs` with `stream: true`, benchmarks) buffer
  nothing. When a stream ends or the client disconnects, the queue is emptied.
- New setting `max_retained_audio_mb` (default 256,
  `OMNIVOICE_TTS_MAX_RETAINED_AUDIO_MB`, `0` = unlimited) caps the audio kept for
  finished jobs. Above the cap, the least recently used jobs (completion or last
  download) lose their audio. The job stays in the job list, and its download
  returns `410 Gone`. The newest job and jobs finished in the last 60 s always
  keep their audio, so waiting callers and poll-then-download clients still get
  theirs.
- The startup warmup runs under `torch.inference_mode()`, like real requests.
  Before, it built autograd state.
- `start_server.bat` and the Dockerfile set `OPENBLAS_NUM_THREADS=4`. numpy/scipy
  OpenBLAS started ~24 threads at import, and each committed 32 MiB it never
  used (~1.5 GB commit, no RAM). Runtime does not change.

### Changed — VRAM budget 7000 MB

The RTX 4090 is shared with Whisper, EMBED and DIA, so the default
`vram_budget_mb` drops from 10000 to 7000. Values saved in
`data/runtime_settings.json` still take precedence.

- At 7000 MB the scheduler caps one batch at ~95 s of estimated audio and one
  sequence at 1587 chars. Before, the caps were ~167 s and 2777 chars.
- Measured on the 4090 (Base + `Sonja`, bf16, compile on, batch 6):
  - Six ~400-char sentences now run as batches of 4 + 2 instead of one batch
    of 6. Peak VRAM on the OmniVoice process fell from 5.9 GB to 4.7 GB, and
    wall time stayed the same (4.0 s for 168 s of audio).
  - The highest peak left is a batch of six ~265-char sentences (111 s of
    audio) at ~5.1 GB.
  - Short texts and mixed concurrent loads did not change.
- The capacity constants (3000 MB + 42 MB per audio second, 0.06 s per char)
  stay conservative. On the 4090 the model is resident at ~2.4 GB and a batch
  adds ~21-24 MB per second of audio, so real peaks stay well below the budget.

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
