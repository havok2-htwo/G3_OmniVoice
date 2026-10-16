from __future__ import annotations

import asyncio
import io
import wave
from datetime import datetime, timezone

import pytest

from omnivoice_tts_server.capacity import (
    DEFAULT_PROMPT_SECONDS,
    MODEL_RESIDENT_MB,
    batch_audio_budget_seconds,
    capacity_summary,
    estimate_audio_seconds,
    estimate_item_seconds,
    estimate_vram_mb,
    max_chars_per_chunk,
    prompt_audio_seconds,
)
from omnivoice_tts_server.config import Settings
from omnivoice_tts_server.domain.models import SpeechRequest, TaskType
from omnivoice_tts_server.domain.state import InMemoryStore, VoiceProfileRecord
from omnivoice_tts_server.prompt_batch import split_sentences
from omnivoice_tts_server.services_v2 import EventHub, QueueService
from omnivoice_tts_server.voice_presets import PRESETS_DIR

SONJA_PROMPT_S = 14.45
SONJA_V2_PROMPT_S = 8.21
# Measured on the RTX 5090 (2026-10-16, Base + the 14.4 s Sonja prompt, bf16, compile,
# trim on): (batch items, total input chars, process peak MiB). Short = ~99-char German
# sentences (~7 s audio each), long = ~314-char sentences (~22 s each).
IDLE_MIB = 2615
SHORT_CHARS = 3181 / 32
LONG_CHARS = 10057 / 32
MEASURED_POINTS = [
    (1, SHORT_CHARS, 2991),
    (2, 2 * SHORT_CHARS, 3217),
    (4, 4 * SHORT_CHARS, 3807),
    (8, 8 * SHORT_CHARS, 4990),
    (9, 9 * SHORT_CHARS, 5310),
    (12, 12 * SHORT_CHARS, 6200),
    (16, 16 * SHORT_CHARS, 7718),
    (17, 17 * SHORT_CHARS, 7718),
    (24, 24 * SHORT_CHARS, 10276),
    (29, 29 * SHORT_CHARS, 11608),
    (32, 32 * SHORT_CHARS, 12842),
    (9, 9 * LONG_CHARS, 7294),
    (16, 16 * LONG_CHARS, 10698),
    (17, 17 * LONG_CHARS, 11846),
    (32, 32 * LONG_CHARS, 20216),
]


def _estimate_for(items: int, chars: float, prompt_s: float = SONJA_PROMPT_S) -> float:
    return estimate_vram_mb(estimate_audio_seconds('x' * int(round(chars))) + items * prompt_s)


def make_wav_bytes(seconds: float, sample_rate: int = 24_000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b'\x00\x00' * int(seconds * sample_rate))
    return buffer.getvalue()


def test_budget_disabled_when_zero_or_negative() -> None:
    assert batch_audio_budget_seconds(0) == 0.0
    assert batch_audio_budget_seconds(-100) == 0.0
    assert max_chars_per_chunk(0) == 0


def test_budget_scales_with_vram() -> None:
    small = batch_audio_budget_seconds(16000)
    large = batch_audio_budget_seconds(28000)
    assert 0 < small < large
    # chars-per-chunk tracks the audio budget
    assert max_chars_per_chunk(16000) < max_chars_per_chunk(28000)


def test_estimated_peak_stays_within_budget() -> None:
    summary = capacity_summary(24000)
    # The derived audio budget must not imply a peak above the requested budget.
    assert summary['estimated_peak_vram_mb'] <= 24000 + 1  # rounding slack
    assert summary['estimated_peak_vram_mb'] > MODEL_RESIDENT_MB


@pytest.mark.parametrize(('items', 'chars', 'measured_mib'), MEASURED_POINTS)
def test_estimate_is_an_upper_bound_of_every_measured_point(items, chars, measured_mib) -> None:
    estimate = _estimate_for(items, chars)
    # Never below the measured peak (so the budget is a real bound), and not needlessly
    # conservative: within 10 % for short and long sentences alike.
    assert measured_mib <= estimate <= measured_mib * 1.10


@pytest.mark.parametrize('budget', [5000, 7000, 10000, 12000])
@pytest.mark.parametrize('sentence_chars', [SHORT_CHARS, LONG_CHARS])
def test_budget_caps_measured_workloads_below_the_budget(budget, sentence_chars) -> None:
    # The planner's batch size for a budget, then the measured peak of that batch size
    # (linear between measured points, idle = 0 items) must stay within the budget.
    budget_s = batch_audio_budget_seconds(budget)
    item_s = estimate_item_seconds('x' * int(round(sentence_chars)), SONJA_PROMPT_S)
    items = max(1, int(budget_s // item_s))
    measured = [(0, IDLE_MIB)] + sorted((n, mib) for n, chars, mib in MEASURED_POINTS if abs(chars / n - sentence_chars) < 1)
    below = max((p for p in measured if p[0] <= items), default=measured[0])
    above = min((p for p in measured if p[0] >= items), default=measured[-1])
    if above[0] == below[0]:
        peak = below[1]
    else:
        peak = below[1] + (above[1] - below[1]) * (items - below[0]) / (above[0] - below[0])
    assert peak <= budget


def test_tier_budgets_fit_their_batch_sizes_with_the_default_voice() -> None:
    sentence = 'x' * 110
    # 32GB tier: batch 8 of normal sentences fits budget 7000 with SonjaV2 and with Sonja.
    for prompt_s in (SONJA_V2_PROMPT_S, SONJA_PROMPT_S):
        assert 8 * estimate_item_seconds(sentence, prompt_s) <= batch_audio_budget_seconds(7000)
    # 24GB tiers: batch 4 fits budget 5000 with Sonja, also for ~300-char sentences with SonjaV2.
    assert 4 * estimate_item_seconds(sentence, SONJA_PROMPT_S) <= batch_audio_budget_seconds(5000)
    assert 4 * estimate_item_seconds('x' * 300, SONJA_V2_PROMPT_S) <= batch_audio_budget_seconds(5000)


def test_estimate_audio_seconds_monotonic() -> None:
    assert estimate_audio_seconds('') > 0
    assert estimate_audio_seconds('a' * 1000) > estimate_audio_seconds('a' * 100)


def test_item_seconds_add_the_prompt() -> None:
    assert estimate_item_seconds('a' * 100) == estimate_audio_seconds('a' * 100)
    assert estimate_item_seconds('a' * 100, 8.0) == pytest.approx(estimate_audio_seconds('a' * 100) + 8.0)
    assert estimate_item_seconds('a' * 100, -3.0) == estimate_audio_seconds('a' * 100)


def test_prompt_audio_seconds_reads_wav_bytes_and_paths(tmp_path) -> None:
    assert prompt_audio_seconds(make_wav_bytes(2.5)) == pytest.approx(2.5)
    assert prompt_audio_seconds(make_wav_bytes(3.0, sample_rate=16_000)) == pytest.approx(3.0)
    path = tmp_path / 'ref.wav'
    path.write_bytes(make_wav_bytes(1.5))
    assert prompt_audio_seconds(str(path)) == pytest.approx(1.5)
    assert prompt_audio_seconds((PRESETS_DIR / 'sonjav2.wav').read_bytes()) == pytest.approx(SONJA_V2_PROMPT_S, abs=0.05)
    assert prompt_audio_seconds((PRESETS_DIR / 'sonja.wav').read_bytes()) == pytest.approx(SONJA_PROMPT_S, abs=0.05)


def test_prompt_audio_seconds_fallbacks(tmp_path) -> None:
    assert prompt_audio_seconds(None) == 0.0
    assert prompt_audio_seconds(b'') == 0.0
    assert prompt_audio_seconds(b'RIFF-not-really') == DEFAULT_PROMPT_SECONDS
    assert prompt_audio_seconds('https://example.invalid/ref.wav') == DEFAULT_PROMPT_SECONDS
    assert prompt_audio_seconds(str(tmp_path / 'missing.wav')) == DEFAULT_PROMPT_SECONDS


def test_max_chars_per_chunk_reserves_the_prompt() -> None:
    # 7000 MB: (7000 - 2700) / 42 MB per sequence second - 15 s prompt allowance, at 0.07 s/char.
    assert max_chars_per_chunk(7000) == 1248
    assert max_chars_per_chunk(3000) == 240  # floor


def test_split_sentences_hard_split_when_disabled() -> None:
    # Chunking off: a long text must still be split to <= max_chars pieces.
    text = ('Wort ' * 400).strip()  # ~2000 chars, no sentence punctuation
    parts = split_sentences(text, enabled=False, max_chars=300)
    assert len(parts) > 1
    assert all(len(p) <= 300 for p in parts)


def test_split_sentences_hard_split_caps_long_sentence() -> None:
    # Chunking on, but one giant sentence must be hard-split to the budget.
    text = ('lang ' * 500).strip() + '.'
    parts = split_sentences(text, enabled=True, max_chars=400)
    assert all(len(p) <= 400 for p in parts)


def test_split_sentences_no_max_chars_is_noop() -> None:
    text = 'Erster Satz. Zweiter Satz.'
    assert split_sentences(text, enabled=True) == split_sentences(text, enabled=True, max_chars=0)


# --- batch planner -----------------------------------------------------------------

SENTENCE = 'Der Zug nach Hamburg faehrt heute ausnahmsweise von Gleis sieben ab, bitte beachten Sie die Wagenreihung.'


def _profile(voice_id: str, name: str, seconds: float) -> VoiceProfileRecord:
    return VoiceProfileRecord(
        voice_id=voice_id,
        name=name,
        source='upload',
        created_at=datetime(2026, 10, 1, tzinfo=timezone.utc),
        audio_bytes=make_wav_bytes(seconds),
        ref_text='Hallo.',
        consent=True,
    )


def _queue(tmp_path, **overrides) -> QueueService:
    overrides.setdefault('max_batch_size', 32)
    overrides.setdefault('max_parallel_requests', 32)
    settings = Settings(data_dir=tmp_path, sentence_chunking=False, **overrides)
    store = InMemoryStore(max_queue_size=64)
    for profile in (_profile('voice_long', 'Long', SONJA_PROMPT_S), _profile('voice_short', 'Short', SONJA_V2_PROMPT_S)):
        store.voice_profiles[profile.voice_id] = profile
    return QueueService(store, object(), EventHub(store), settings)


def _plan_size(queue: QueueService, count: int, **request_fields) -> int:
    async def run() -> int:
        for _ in range(count):
            await queue.submit(SpeechRequest(input=SENTENCE, **request_fields))
        queue._promote_waiting_locked()
        return len(queue._build_batch_plan_locked())

    return asyncio.run(run())


def test_submit_records_the_prompt_seconds_per_request(tmp_path) -> None:
    queue = _queue(tmp_path)

    async def run() -> list[float]:
        long_job = await queue.submit(SpeechRequest(input=SENTENCE, voice='voice_long'))
        short_job = await queue.submit(SpeechRequest(input=SENTENCE, voice='Short'))
        auto_job = await queue.submit(
            SpeechRequest(input=SENTENCE, model='k2-fsa/OmniVoice-AutoVoice', task_type=TaskType.custom_voice)
        )
        url_job = await queue.submit(SpeechRequest(input=SENTENCE, ref_audio='https://example.invalid/a.wav', ref_text='Hallo.'))
        return [queue.store.request_states[job.job_id].prompt_seconds for job in (long_job, short_job, auto_job, url_job)]

    long_s, short_s, auto_s, url_s = asyncio.run(run())
    assert long_s == pytest.approx(SONJA_PROMPT_S)
    assert short_s == pytest.approx(SONJA_V2_PROMPT_S)
    assert auto_s == 0.0
    assert url_s == DEFAULT_PROMPT_SECONDS


def test_planner_counts_each_items_prompt(tmp_path) -> None:
    item_long = estimate_item_seconds(SENTENCE, SONJA_PROMPT_S)
    item_short = estimate_item_seconds(SENTENCE, SONJA_V2_PROMPT_S)
    budget_s = batch_audio_budget_seconds(5000)
    # A longer prompt means fewer items per batch under the same budget.
    assert _plan_size(_queue(tmp_path / 'a', vram_budget_mb=5000), 32, voice='voice_long') == int(budget_s // item_long)
    assert _plan_size(_queue(tmp_path / 'b', vram_budget_mb=5000), 32, voice='voice_short') == int(budget_s // item_short)
    assert int(budget_s // item_short) > int(budget_s // item_long)


def test_planned_batch_estimate_stays_within_budget(tmp_path) -> None:
    for budget in (5000, 7000, 10000):
        items = _plan_size(_queue(tmp_path / str(budget), vram_budget_mb=budget), 32, voice='voice_long')
        assert estimate_vram_mb(items * estimate_item_seconds(SENTENCE, SONJA_PROMPT_S)) <= budget


def test_planner_without_budget_or_prompt(tmp_path) -> None:
    assert _plan_size(_queue(tmp_path / 'a', vram_budget_mb=0), 32, voice='voice_long') == 32
    # AutoVoice carries no prompt, so it packs more items than a clone under the same budget.
    auto = _plan_size(
        _queue(tmp_path / 'b', vram_budget_mb=5000),
        32,
        model='k2-fsa/OmniVoice-AutoVoice',
        task_type=TaskType.custom_voice,
    )
    assert auto == int(batch_audio_budget_seconds(5000) // estimate_item_seconds(SENTENCE))


def test_planner_always_takes_one_item_over_budget(tmp_path) -> None:
    queue = _queue(tmp_path, vram_budget_mb=3000)
    assert _plan_size(queue, 4, voice='voice_long') == 1
