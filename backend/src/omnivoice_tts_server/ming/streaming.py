"""Pure streaming helpers for the Ming engine (no torch, CPU-testable).

- ``ChunkSchedule``: how many latents to decode per streaming VAE call (small first, then growing).
- ``SilenceTrimmer``: a streaming, sample-exact version of upstream ``ming_tts.trim_trailing_silence``.
- ``StreamResampler``: stateful soxr resampling per row (bit-identical to one-shot ``soxr.resample``).
- ``MingStream``: the blocking chunk iterator a row hands to its consumer (the Audio8 engines use it too).
"""
from __future__ import annotations

import queue
import time
from typing import Any, Iterator

import numpy as np

# Upstream ming_tts.py defaults.
SILENCE_FRAME_SECONDS = 0.1
SILENCE_THRESHOLD = 1e-3
SILENCE_TAIL_SECONDS = 0.3

_EMPTY = np.zeros(0, dtype=np.float32)
_END = object()


class ChunkSchedule:
    """Latents per streaming VAE decode call for one row.

    The decoder's upsampler needs one latent of lookahead, so with ``g`` latents generated (fragment not finished) at
    most ``g - 1`` can be decoded. The first call happens once ``first`` latents exist and decodes ``first - 1``
    (at least 1); later calls double up to ``maximum``. The final call of a fragment decodes everything that is
    left, which is at most ``maximum + 1`` latents because the schedule is checked after every LM step. The schedule
    is not reset between fragments of a row: audio is already flowing by then, so later fragments use big chunks.
    """

    def __init__(self, first: int = 2, maximum: int = 8) -> None:
        self.first = max(1, int(first))
        self.initial = max(1, self.first - 1)
        self.maximum = max(self.initial, int(maximum))
        self.target = self.initial

    def take(self, generated: int, decoded: int, final: bool) -> int:
        """Number of latents to decode now (0 = not yet)."""
        if final:
            return max(0, generated - decoded)
        if generated - 1 - decoded >= self.target:
            n = self.target
            self.target = min(self.maximum, self.target * 2)
            return n
        return 0

    def reset(self) -> None:
        self.target = self.initial


class SilenceTrimmer:
    """Streaming ``trim_trailing_silence`` for one fragment.

    Upstream scans the finished fragment in 0.1 s frames, finds the last frame whose mean |x| is above the threshold
    and keeps 0.3 s after it, dropping any sub-frame remainder at the end. Streaming equivalent: everything up to the
    end of the last loud frame plus the tail is kept no matter what follows, so it is released at once; audio beyond
    that is held until a loud frame proves it is not trailing silence. ``finish`` applies upstream's final cut.

    The result equals upstream exactly as long as the audio passed to ``finish`` is at least one frame long (the
    engine's final decode call always delivers more than that), because then no sample of the dropped remainder was
    released early.
    """

    def __init__(self, sample_rate: int, threshold: float = SILENCE_THRESHOLD,
                 frame_seconds: float = SILENCE_FRAME_SECONDS, tail_seconds: float = SILENCE_TAIL_SECONDS) -> None:
        self.frame = int(sample_rate * frame_seconds)
        self.tail = int(sample_rate * tail_seconds)
        self.threshold = float(threshold)
        self.reset()

    def reset(self) -> None:
        self._buf = _EMPTY
        self._buf_start = 0
        self.total = 0
        self.released = 0
        self._frames_done = 0
        self.last_loud_end = 0

    def _append(self, audio: np.ndarray) -> None:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size:
            self._buf = np.concatenate([self._buf, audio]) if self._buf.size else audio.copy()
            self.total += audio.size
        frame = self.frame
        if frame <= 0:
            return
        while (self._frames_done + 1) * frame <= self.total:
            start = self._frames_done * frame - self._buf_start
            score = float(np.mean(np.abs(self._buf[start:start + frame]), dtype=np.float64))
            self._frames_done += 1
            if score > self.threshold:
                self.last_loud_end = self._frames_done * frame

    def _emit(self, end: int) -> np.ndarray:
        if end <= self.released:
            return _EMPTY
        out = self._buf[self.released - self._buf_start:end - self._buf_start]
        self.released = end
        keep_from = min(self.released, self._frames_done * self.frame)
        if keep_from > self._buf_start:
            self._buf = self._buf[keep_from - self._buf_start:]
            self._buf_start = keep_from
        return out

    def push(self, audio: np.ndarray) -> np.ndarray:
        self._append(audio)
        return self._emit(min(self.total, self.last_loud_end + self.tail))

    def finish(self, audio: np.ndarray | None = None) -> np.ndarray:
        if audio is not None:
            self._append(audio)
        if self.frame <= 0 or self.total < self.frame:
            final = min(self.total, self.tail)
        else:
            complete = (self.total // self.frame) * self.frame
            final = min(complete, self.last_loud_end + self.tail)
        out = self._emit(final)
        self.reset()
        return out

    @property
    def held(self) -> int:
        return self.total - self.released


def trim_trailing_silence(audio: np.ndarray, sample_rate: int, threshold: float = SILENCE_THRESHOLD,
                          tail_seconds: float = SILENCE_TAIL_SECONDS) -> np.ndarray:
    """Non-streaming reference (upstream ming_tts.trim_trailing_silence on a 1-D array)."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return audio
    frame = int(sample_rate * SILENCE_FRAME_SECONDS)
    tail = int(sample_rate * tail_seconds)
    if frame <= 0 or audio.size < frame:
        return audio[:tail]
    num_frames = (audio.size - frame) // frame + 1
    audio = audio[:num_frames * frame]
    scores = np.abs(audio.reshape(num_frames, frame)).mean(axis=1, dtype=np.float64)
    index = num_frames - 1
    while index >= 0 and scores[index] <= threshold:
        index -= 1
    if index < 0:
        return audio[:tail]
    return audio[:index * frame + frame + tail]


class StreamResampler:
    """Stateful mono resampler; a pass-through when the rates match."""

    def __init__(self, in_rate: int, out_rate: int, quality: str = 'HQ') -> None:
        self.in_rate = int(in_rate)
        self.out_rate = int(out_rate)
        self._stream = None
        if self.in_rate != self.out_rate:
            import soxr

            self._stream = soxr.ResampleStream(self.in_rate, self.out_rate, 1, dtype='float32', quality=quality)

    def push(self, audio: np.ndarray, last: bool = False) -> np.ndarray:
        audio = np.ascontiguousarray(np.asarray(audio, dtype=np.float32).reshape(-1))
        if self._stream is None:
            return audio
        return np.asarray(self._stream.resample_chunk(audio, last=last), dtype=np.float32).reshape(-1)

    def flush(self) -> np.ndarray:
        return self.push(_EMPTY, last=True)


class MingStream:
    """Blocking iterator over np.float32 audio chunks at the engine's output sample rate."""

    def __init__(self, row_id: str, engine: Any) -> None:
        self.row_id = row_id
        self._engine = engine
        self._queue: queue.Queue = queue.Queue()
        self._row: Any = None
        self.submitted_at = time.perf_counter()
        self.started_at: float | None = None
        self.first_chunk_at: float | None = None
        self.finished_at: float | None = None
        self.metrics: dict[str, Any] = {}
        self.sample_rate = engine.output_sample_rate
        self._done = False

    # -- producer side (GPU thread) ---------------------------------------------------------------------------
    def _put(self, chunk: np.ndarray) -> None:
        if chunk.size == 0 or self._done:
            return
        if self.first_chunk_at is None:
            self.first_chunk_at = time.perf_counter()
            self.metrics['ttfa_ms'] = (self.first_chunk_at - self.submitted_at) * 1000.0
        self.metrics['samples'] = self.metrics.get('samples', 0) + int(chunk.size)
        self._queue.put(chunk)

    def _finish(self, error: BaseException | None = None) -> None:
        if self._done:
            return
        self._done = True
        self.finished_at = time.perf_counter()
        self.metrics['total_ms'] = (self.finished_at - self.submitted_at) * 1000.0
        self.metrics['audio_s'] = self.metrics.get('samples', 0) / float(self.sample_rate)
        self._queue.put(error if error is not None else _END)

    # -- consumer side ----------------------------------------------------------------------------------------
    def next_chunk(self, timeout: float | None = None) -> np.ndarray | None:
        """Next chunk, or None when the row has finished. Raises the engine error, or queue.Empty on timeout."""
        item = self._queue.get(timeout=timeout)
        if item is _END:
            self._queue.put(_END)
            return None
        if isinstance(item, BaseException):
            self._queue.put(item)
            raise item
        return item

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            chunk = self.next_chunk()
            if chunk is None:
                return
            yield chunk

    def cancel(self) -> None:
        self._engine._cancel(self)

    @property
    def done(self) -> bool:
        return self._done
