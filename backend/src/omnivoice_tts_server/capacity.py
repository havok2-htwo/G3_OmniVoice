"""VRAM-aware capacity limits, derived from measurements on an RTX 5090 (32 GB).

Empirically (k2-fsa/OmniVoice, bf16, num_step~=18): peak VRAM grows ~linearly with the
total audio in flight (a single generate sequence, or the sum across a batch):

    VRAM_MB  ~=  MODEL_RESIDENT_MB  +  audio_seconds * MB_PER_AUDIO_SECOND

A single unchunked ~9 min text hit the 32 GB ceiling (~2.3 GB/audio-min). From one
`vram_budget_mb` knob we derive two enforced limits so peak VRAM stays within budget
regardless of how the work arrives:

  * max chars per generated sequence  -> guards a single long text (even with chunking off)
  * max audio seconds packed per batch -> guards many sentences batched together

Set `vram_budget_mb <= 0` to disable budgeting (unbounded, old behaviour).
"""
from __future__ import annotations

# Measured constants (intentionally a little conservative so estimates over- not
# under-shoot real VRAM, i.e. we batch slightly less rather than risk OOM).
MODEL_RESIDENT_MB = 3000           # model params+tokenizer+CUDA ctx (~2.4-2.9 GB measured; the
                                   # old 11500 was reserved-cache, not resident, and over-chunked long texts)
MB_PER_AUDIO_SECOND = 42.0         # ~2.5 GB/min (measured ~2.3 GB/min)
EST_AUDIO_SECONDS_PER_CHAR = 0.06  # measured ~0.05-0.06 s of speech per input char

MIN_CHUNK_CHARS = 240              # never split shorter than this
MIN_BATCH_AUDIO_SECONDS = 15.0     # always allow at least this much per batch


def estimate_audio_seconds(text: str) -> float:
    return max(0.5, len(text or '') * EST_AUDIO_SECONDS_PER_CHAR)


def estimate_vram_mb(audio_seconds: float) -> float:
    return MODEL_RESIDENT_MB + max(0.0, audio_seconds) * MB_PER_AUDIO_SECOND


def batch_audio_budget_seconds(vram_budget_mb: int | float | None) -> float:
    """Max audio seconds to pack into one generate() batch for the given VRAM budget.
    Returns 0.0 (= unlimited) when budgeting is disabled."""
    if not vram_budget_mb or vram_budget_mb <= 0:
        return 0.0
    return max(MIN_BATCH_AUDIO_SECONDS, (float(vram_budget_mb) - MODEL_RESIDENT_MB) / MB_PER_AUDIO_SECOND)


def max_chars_per_chunk(vram_budget_mb: int | float | None) -> int:
    """Max characters in a single generated sequence for the given VRAM budget.
    Returns 0 (= unlimited) when budgeting is disabled."""
    budget_s = batch_audio_budget_seconds(vram_budget_mb)
    if budget_s <= 0:
        return 0
    return max(MIN_CHUNK_CHARS, int(budget_s / EST_AUDIO_SECONDS_PER_CHAR))


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
