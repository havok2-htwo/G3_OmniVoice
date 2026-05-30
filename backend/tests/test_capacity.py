from __future__ import annotations

from omnivoice_tts_server.capacity import (
    MODEL_RESIDENT_MB,
    batch_audio_budget_seconds,
    capacity_summary,
    estimate_audio_seconds,
    max_chars_per_chunk,
)
from omnivoice_tts_server.prompt_batch import split_sentences


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


def test_estimate_audio_seconds_monotonic() -> None:
    assert estimate_audio_seconds('') > 0
    assert estimate_audio_seconds('a' * 1000) > estimate_audio_seconds('a' * 100)


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
