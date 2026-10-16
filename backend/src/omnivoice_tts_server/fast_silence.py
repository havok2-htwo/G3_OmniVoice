"""A numpy port of ``omnivoice.utils.audio.remove_silence`` that returns exactly the same samples.

OmniVoice trims silences from every generated item with pydub (``split_on_silence`` with a 10 ms seek step plus
edge trimming). pydub measures the RMS of one window per seek step in a Python loop: ~650 windows of 500 ms for a
7 s item, ~7 ms of CPU per item while holding the GIL. This port computes all window RMS values at once from a
cumulative sum and keeps pydub's logic line by line, including its quirks:

* lengths are whole milliseconds, ``round(1000 * frames / rate)``, so a slice can end up to ~1 ms past the data and
  pydub pads the missing frames with zeros (they count in the RMS and in the output);
* positions convert with ``int(ms * (rate / 1000.0))``;
* RMS is ``audioop.rms``: a float64 sum of squares (exact for these window sizes), divided by the sample count,
  square-rooted and truncated to an integer;
* an all-silent input becomes ``AudioSegment.silent(0)`` (11025 Hz, empty).

Only mono float input takes the fast path; anything else calls the stock function. ``install`` checks the port
against pydub on a synthetic signal first and leaves the stock function in place if they differ.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Callable

import numpy as np

logger = logging.getLogger('omnivoice_tts_server.fast_silence')

_MAX_AMPLITUDE = 32768.0  # pydub max_possible_amplitude for 16-bit samples


def _db_to_float(db: float) -> float:
    return 10 ** (float(db) / 20)


def _ratio_to_db(ratio: float) -> float:
    ratio = float(ratio)
    if ratio == 0:
        return -float('inf')
    return 20 * math.log(ratio, 10)


class _Segment:
    """The parts of pydub.AudioSegment the trimming uses: mono int16 samples at one frame rate."""

    __slots__ = ('data', 'rate')

    def __init__(self, data: np.ndarray, rate: int) -> None:
        self.data = data
        self.rate = rate

    def __len__(self) -> int:
        return round(1000 * (float(len(self.data)) / self.rate))

    def _frames(self, ms: float) -> int:
        if ms < 0:
            ms = len(self) - abs(ms)
        return int(ms * (self.rate / 1000.0))

    def __getitem__(self, item: slice) -> _Segment:
        length = len(self)
        start = item.start if item.start is not None else 0
        end = item.stop if item.stop is not None else length
        start = self._frames(min(start, length))
        end = self._frames(min(end, length))
        data = self.data[start:end]
        missing = (end - start) - len(data)
        if missing > 0:
            if missing > 2 * (self.rate / 1000.0):
                raise ValueError(f'pydub would refuse to pad {missing} frames')
            data = np.concatenate([data, np.zeros(missing, dtype=np.int16)])
        return _Segment(data, self.rate)

    def rms(self) -> int:
        n = len(self.data)
        if n == 0:
            return 0
        values = self.data.astype(np.int64)
        return int(math.sqrt(float(int(np.dot(values, values))) / float(n)))

    def dbfs(self) -> float:
        rms = self.rms()
        if not rms:
            return -float('inf')
        return _ratio_to_db(rms / _MAX_AMPLITUDE)

    def reverse(self) -> _Segment:
        return _Segment(self.data[::-1].copy(), self.rate)


def _window_rms(seg: _Segment, starts_ms: list[int], length_ms: int) -> np.ndarray:
    """audioop.rms of seg[s:s + length_ms] for every s, with pydub's zero padding at the end."""
    factor = seg.rate / 1000.0
    starts = np.array([int(s * factor) for s in starts_ms], dtype=np.int64)
    ends = np.array([int((s + length_ms) * factor) for s in starts_ms], dtype=np.int64)
    needed = int(ends.max()) if len(ends) else 0
    data = seg.data
    if needed > len(data):
        if needed - len(data) > 2 * factor:
            raise ValueError('pydub would refuse to pad this window')
        data = np.concatenate([data, np.zeros(needed - len(data), dtype=np.int16)])
    squares = data.astype(np.int64) ** 2
    cumulative = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(squares)])
    sums = (cumulative[ends] - cumulative[starts]).astype(np.float64)
    counts = (ends - starts).astype(np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        rms = np.floor(np.sqrt(sums / counts))
    rms[counts <= 0] = 0
    return rms


def _detect_silence(seg: _Segment, min_silence_len: int, silence_thresh: float, seek_step: int) -> list[list[int]]:
    seg_len = len(seg)
    if seg_len < min_silence_len:
        return []
    threshold = _db_to_float(silence_thresh) * _MAX_AMPLITUDE
    last_slice_start = seg_len - min_silence_len
    slice_starts = list(range(0, last_slice_start + 1, seek_step))
    if last_slice_start % seek_step:
        slice_starts.append(last_slice_start)
    rms = _window_rms(seg, slice_starts, min_silence_len)
    silence_starts = [start for start, value in zip(slice_starts, rms.tolist()) if value <= threshold]
    if not silence_starts:
        return []
    silent_ranges = []
    prev_i = silence_starts.pop(0)
    current_range_start = prev_i
    for silence_start_i in silence_starts:
        continuous = silence_start_i == prev_i + seek_step
        silence_has_gap = silence_start_i > (prev_i + min_silence_len)
        if not continuous and silence_has_gap:
            silent_ranges.append([current_range_start, prev_i + min_silence_len])
            current_range_start = silence_start_i
        prev_i = silence_start_i
    silent_ranges.append([current_range_start, prev_i + min_silence_len])
    return silent_ranges


def _detect_nonsilent(seg: _Segment, min_silence_len: int, silence_thresh: float, seek_step: int) -> list[list[int]]:
    silent_ranges = _detect_silence(seg, min_silence_len, silence_thresh, seek_step)
    len_seg = len(seg)
    if not silent_ranges:
        return [[0, len_seg]]
    if silent_ranges[0][0] == 0 and silent_ranges[0][1] == len_seg:
        return []
    prev_end_i = 0
    nonsilent_ranges = []
    end_i = 0
    for start_i, end_i in silent_ranges:
        nonsilent_ranges.append([prev_end_i, start_i])
        prev_end_i = end_i
    if end_i != len_seg:
        nonsilent_ranges.append([prev_end_i, len_seg])
    if nonsilent_ranges[0] == [0, 0]:
        nonsilent_ranges.pop(0)
    return nonsilent_ranges


def _split_on_silence(
    seg: _Segment, min_silence_len: int, silence_thresh: float, keep_silence: int, seek_step: int
) -> list[_Segment]:
    output_ranges = [
        [start - keep_silence, end + keep_silence]
        for start, end in _detect_nonsilent(seg, min_silence_len, silence_thresh, seek_step)
    ]
    for range_i, range_ii in zip(output_ranges, output_ranges[1:]):
        last_end = range_i[1]
        next_start = range_ii[0]
        if next_start < last_end:
            range_i[1] = (last_end + next_start) // 2
            range_ii[0] = range_i[1]
    return [seg[max(start, 0):min(end, len(seg))] for start, end in output_ranges]


def _detect_leading_silence(seg: _Segment, silence_threshold: float, chunk_size: int = 10) -> int:
    trim_ms = 0
    length = len(seg)
    while seg[trim_ms:trim_ms + chunk_size].dbfs() < silence_threshold and trim_ms < length:
        trim_ms += chunk_size
    return min(trim_ms, length)


def _remove_silence_edges(seg: _Segment, lead_sil: int, trail_sil: int, silence_threshold: float) -> _Segment:
    start_idx = _detect_leading_silence(seg, silence_threshold)
    start_idx = max(0, start_idx - lead_sil)
    seg = seg[start_idx:]
    seg = seg.reverse()
    start_idx = _detect_leading_silence(seg, silence_threshold)
    start_idx = max(0, start_idx - trail_sil)
    seg = seg[start_idx:]
    return seg.reverse()


def remove_silence_fast(
    audio: np.ndarray,
    sampling_rate: int,
    mid_sil: int = 300,
    lead_sil: int = 100,
    trail_sil: int = 300,
) -> np.ndarray:
    """Same arguments and output as ``omnivoice.utils.audio.remove_silence``, for mono (1, T) float audio."""
    if not isinstance(audio, np.ndarray) or audio.ndim != 2 or audio.shape[0] != 1 or audio.dtype.kind != 'f':
        raise TypeError('fast path needs mono (1, T) float audio')
    samples = (audio * 32768.0).clip(-32768, 32767).astype(np.int16)[0]
    seg = _Segment(samples, int(sampling_rate))
    if mid_sil > 0:
        pieces = _split_on_silence(seg, min_silence_len=mid_sil, silence_thresh=-50, keep_silence=mid_sil, seek_step=10)
        if pieces:
            seg = _Segment(np.concatenate([piece.data for piece in pieces]), int(sampling_rate))
        else:
            seg = _Segment(np.zeros(0, dtype=np.int16), 11025)  # AudioSegment.silent(duration=0)
    seg = _remove_silence_edges(seg, lead_sil, trail_sil, -50)
    return (seg.data.astype(np.float32) / 32768.0)[np.newaxis, :]


def _self_check(stock: Callable[..., np.ndarray]) -> bool:
    rate = 24000
    t = np.arange(int(rate * 3.3) + 13) / rate
    tone = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    tone[: int(0.4 * rate)] = 0.0
    tone[int(1.2 * rate): int(2.0 * rate)] *= 0.0005
    tone[-int(0.35 * rate):] = 0.0
    audio = tone[np.newaxis, :]
    for mid_sil, lead_sil, trail_sil in ((500, 100, 100), (200, 100, 200)):
        expected = stock(audio, rate, mid_sil=mid_sil, lead_sil=lead_sil, trail_sil=trail_sil)
        got = remove_silence_fast(audio, rate, mid_sil=mid_sil, lead_sil=lead_sil, trail_sil=trail_sil)
        if expected.shape != got.shape or expected.dtype != got.dtype or not np.array_equal(expected, got):
            return False
    return True


def install() -> bool:
    """Route OmniVoice's silence trimming through the numpy port if it matches pydub here. Returns True if active."""
    try:
        import omnivoice.models.omnivoice as model_module
    except Exception as exc:  # pragma: no cover - omnivoice missing
        logger.debug('fast silence trimming not installed: %s', exc)
        return False
    current = getattr(model_module, 'remove_silence', None)
    if not callable(current):
        return False
    if getattr(current, '_g3_fast_silence', False):
        return True
    stock = current
    try:
        matches = _self_check(stock)
    except Exception as exc:
        logger.warning('fast silence trimming disabled: self-check failed with %s', exc)
        return False
    if not matches:
        logger.warning('fast silence trimming disabled: output differs from pydub on the self-check')
        return False

    def remove_silence(audio: Any, sampling_rate: int, mid_sil: int = 300, lead_sil: int = 100, trail_sil: int = 300):
        try:
            return remove_silence_fast(audio, sampling_rate, mid_sil, lead_sil, trail_sil)
        except (TypeError, ValueError):
            return stock(audio, sampling_rate, mid_sil=mid_sil, lead_sil=lead_sil, trail_sil=trail_sil)

    remove_silence._g3_fast_silence = True  # type: ignore[attr-defined]
    remove_silence._g3_stock = stock  # type: ignore[attr-defined]
    model_module.remove_silence = remove_silence
    logger.info('fast silence trimming active (numpy port, matches pydub)')
    return True
