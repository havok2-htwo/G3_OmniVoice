"""VRAM-aware capacity limits, derived from measurements on an RTX 5090 (32 GB).

Measured 2026-10-16 (k2-fsa/OmniVoice Base, bf16, sdpa, torch.compile dynamic, num_step
18/22, trim after batch on; process VRAM from the per-process counter). Peak VRAM of one
generate() batch grows ~linearly with the audio the batch carries, and every batch item
carries its own voice-clone reference prompt besides the audio it generates:

    VRAM_MB  ~=  MODEL_RESIDENT_MB  +  MB_PER_AUDIO_SECOND * sum over items (generated s + prompt s)

The prompt term is large: the 14.4 s `Sonja` reference costs ~225 MB per item, more than
a 7 s sentence itself. A model on generated audio alone undershot (budget 7000 -> 7.7 GB
measured). The constants below are the upper envelope of every measured point (batch
1-32 of ~7 s sentences, batch 9-32 of ~22 s sentences, the budget runs): the estimate is
0-8 % above the measured peak, never below it. AutoVoice / VoiceDesign carry no prompt.

Throughput (RTX 5090, num_step 18): Base cloning with `Sonja` runs ~30x realtime at batch
1 and ~35x from batch 4 on, flat up to batch 32 (the GPU is busy from ~batch 16), so a
bigger batch only adds VRAM and latency. num_step 22 costs ~18 % (~29x). Longer items
amortise the prompt better: ~22 s sentences reach ~53x. AutoVoice (no prompt) goes from
39x (batch 1) to 69x (batch 4) and 84x (batch 16); the older ~95-100x note was AutoVoice.

From one `vram_budget_mb` knob (process VRAM upper bound) we derive two enforced limits:

  * max audio seconds per batch (generated + prompt, per item) -> the batch planner
  * max chars per generated sequence -> guards a single long text (even with chunking off)

Set `vram_budget_mb <= 0` to disable budgeting (unbounded, old behaviour).
"""
from __future__ import annotations

import io
import os
import wave
from typing import Any

# Measured constants (upper envelope of the RTX 5090 points, see the module docstring).
MODEL_RESIDENT_MB = 2700           # idle process 2.6 GB (weights, tokenizer, CUDA context)
MB_PER_AUDIO_SECOND = 15.5         # per second of generated + prompt audio across a batch
EST_AUDIO_SECONDS_PER_CHAR = 0.07  # measured 0.069-0.071 s of German speech per input char

# One generated sequence: only measured as a single unchunked ~9 min text that hit the
# 32 GB ceiling (~2.3-2.5 GB per audio minute). Kept for the per-sequence char limit, so a
# raised audio_chunk_threshold (no internal 30 s chunking) cannot outgrow the budget.
MB_PER_SEQUENCE_SECOND = 42.0
# Prompt length reserved when sizing one sequence (the voice is not known while the text
# is split, e.g. in the text preview); the 14.4 s Sonja reference fits.
CHUNK_PROMPT_ALLOWANCE_SECONDS = 15.0
# Prompt length counted when a reference cannot be measured (ref_audio URL, broken file).
DEFAULT_PROMPT_SECONDS = 15.0

MIN_CHUNK_CHARS = 240              # never split shorter than this
MIN_BATCH_AUDIO_SECONDS = 15.0     # always allow at least this much per batch


def estimate_audio_seconds(text: str) -> float:
    return max(0.5, len(text or '') * EST_AUDIO_SECONDS_PER_CHAR)


def estimate_item_seconds(text: str, prompt_seconds: float = 0.0) -> float:
    """Batch cost of one item in audio seconds: the speech it generates plus its prompt."""
    return estimate_audio_seconds(text) + max(0.0, float(prompt_seconds or 0.0))


def estimate_vram_mb(audio_seconds: float) -> float:
    """Process VRAM for a batch carrying `audio_seconds` (generated + prompt, summed over items)."""
    return MODEL_RESIDENT_MB + max(0.0, audio_seconds) * MB_PER_AUDIO_SECOND


def batch_audio_budget_seconds(vram_budget_mb: int | float | None) -> float:
    """Max audio seconds (generated + prompt per item) to pack into one generate() batch
    for the given VRAM budget. Returns 0.0 (= unlimited) when budgeting is disabled."""
    if not vram_budget_mb or vram_budget_mb <= 0:
        return 0.0
    return max(MIN_BATCH_AUDIO_SECONDS, (float(vram_budget_mb) - MODEL_RESIDENT_MB) / MB_PER_AUDIO_SECOND)


def max_chars_per_chunk(vram_budget_mb: int | float | None) -> int:
    """Max characters in a single generated sequence for the given VRAM budget.
    Returns 0 (= unlimited) when budgeting is disabled."""
    if not vram_budget_mb or vram_budget_mb <= 0:
        return 0
    sequence_s = (float(vram_budget_mb) - MODEL_RESIDENT_MB) / MB_PER_SEQUENCE_SECOND - CHUNK_PROMPT_ALLOWANCE_SECONDS
    return max(MIN_CHUNK_CHARS, int(sequence_s / EST_AUDIO_SECONDS_PER_CHAR))


def prompt_audio_seconds(audio: bytes | str | None) -> float:
    """Length of a voice-clone reference in seconds: WAV/FLAC/... bytes or a local path.

    The engine's prompt preprocessing only removes silence (with ref_text given it never
    cuts long audio), so the file length is an upper bound of what each item carries
    (shipped Sonja: 14.45 s file, 14.41 s after trimming). Unreadable input counts as
    DEFAULT_PROMPT_SECONDS; no input counts as 0.
    """
    if not audio:
        return 0.0
    if isinstance(audio, str) and not os.path.isfile(audio):
        return DEFAULT_PROMPT_SECONDS
    source: Any = io.BytesIO(audio) if isinstance(audio, (bytes, bytearray)) else audio
    try:
        import soundfile

        info = soundfile.info(source)
        if info.samplerate > 0 and info.frames > 0:
            return info.frames / float(info.samplerate)
    except Exception:
        pass
    try:
        if isinstance(source, io.BytesIO):
            source.seek(0)
        with wave.open(source, 'rb') as wav_file:
            if wav_file.getframerate() > 0:
                return wav_file.getnframes() / float(wav_file.getframerate())
    except Exception:
        pass
    return DEFAULT_PROMPT_SECONDS


def capacity_summary(vram_budget_mb: int | float | None) -> dict:
    """Human-facing derived limits + the VRAM the budget corresponds to."""
    budget_s = batch_audio_budget_seconds(vram_budget_mb)
    return {
        'vram_budget_mb': int(vram_budget_mb or 0),
        'max_batch_audio_seconds': round(budget_s, 1),
        'max_chars_per_chunk': max_chars_per_chunk(vram_budget_mb),
        'estimated_peak_vram_mb': int(estimate_vram_mb(budget_s)) if budget_s > 0 else 0,
        'model_resident_mb': MODEL_RESIDENT_MB,
        'mb_per_audio_second': MB_PER_AUDIO_SECOND,
    }


# --- Ming-3.0-tts 0.5B (runtime_backend='ming') --------------------------------------------------------------------
# Measured 2026-10 on the RTX 5090 (ming/engine.py, CUDA graphs, bf16, batch_invariant): the engine allocates its KV
# cache for every slot at load, so VRAM does not grow with the audio a batch carries. Peak per process (nvidia-smi,
# CUDA context included): 4.25 GB at 16 slots, 4.38 GB at 32, 4.64 GB at 64 -> ~4.1 GB + ~8.5 MB per slot. Every slot
# count up to 64 fits the 24 GB 3090 tier.
MING_RESIDENT_BASE_MB = 4114
MING_MB_PER_SLOT = 8.5
# Queue-level sentence cap: the engine splits long text into ~24-word fragments itself and streams them seamlessly in
# one row, so this only bounds a row's length (and lets one long paragraph spread over several slots).
MING_MAX_CHARS_PER_SENTENCE = 400
# Slots (a quarter) that only a job with nothing rendering may take: one job's further sentences never fill the
# engine, so a request arriving during a busy period still starts within about one step.
MING_NEWCOMER_HEADROOM_SHARE = 0.25


def ming_resident_vram_mb(max_slots: int) -> int:
    return int(MING_RESIDENT_BASE_MB + MING_MB_PER_SLOT * max(1, int(max_slots)))


def ming_capacity_profile(max_slots: int) -> dict[str, Any]:
    """Queue limits for the Ming engine (see QueueService._build_concurrent_plan_locked).

    max_chars_per_chunk   queue sentence split length (replaces the OmniVoice VRAM-derived limit)
    max_batch_items       sentences per dispatched stream batch
    max_inflight_items    sentences rendering at once over all running batches (= engine slots)
    newcomer_headroom     slots only the next sentence of a job with nothing rendering may take
    """
    slots = max(1, int(max_slots))
    return {
        'engine': 'ming',
        'max_chars_per_chunk': MING_MAX_CHARS_PER_SENTENCE,
        'max_batch_items': slots,
        'max_inflight_items': slots,
        'newcomer_headroom': int(slots * MING_NEWCOMER_HEADROOM_SHARE) if slots > 1 else 0,
        'estimated_peak_vram_mb': ming_resident_vram_mb(slots),
    }
