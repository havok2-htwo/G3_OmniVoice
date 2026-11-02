"""Audio8 0.1B INT8 on the CPU with onnxruntime, no torch. Same interface as the slot engine (``engine.py``).

Port of ``onnx_runtime_0_1b_int8/arktts_runtime`` of https://github.com/Audio8-AI/Audio8_TTS (Apache-2.0, Audio8 team):
the graph contract, the prompt layout, sampling with repetition-aware sampling, the reference encoder. Different here:

- Rows run in parallel on ``workers`` pairs of threads. The LM thread renders frames, its decoder thread turns them into
  audio, so the codec does not hold up the LM. The LM sessions run on one thread each (more threads gain under 10 %) and
  are shared by all workers; every decoder has its own sessions and ``threads`` threads.
- The prompt head of a voice runs once: its state is cached and copied per fragment, only the text is prefilled.
- Streaming cuts the codec in two (``onnx_codec.py``), which is what makes it realtime.
- Text is fragmented and frames are capped like in the slot engine (``text.py``).

The first frame of a fragment cannot be EOS (it would produce no audio), as upstream. A row's audio is bit-identical for
its text, voice and seed, whatever else runs: each fragment has its own generator, the LM runs on one thread whatever
the worker count, and chunks sit on a fixed frame grid, because the convolutions' short context makes the rounding of a
sample depend on where its chunk starts.
"""
from __future__ import annotations

import collections
import gc
import hashlib
import itertools
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..ming.streaming import MingStream as Audio8OnnxStream
from ..ming.streaming import SilenceTrimmer, StreamResampler
from .onnx_codec import StreamDecoder, cpu_session
from .text import (DEFAULT_FRAGMENT_MAX_CHARS, EXPECTED_FRAMES_PER_CHAR, MIN_FRAGMENT_CHARS, frame_cap, prefix_ids,
                   split_fragments, suffix_ids, user_turn_ids)
from .voice_cache import voice_file

logger = logging.getLogger(__name__)

NATIVE_SAMPLE_RATE = 44100
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 50
# Repetition-aware sampling (config.json ras_*): a semantic token seen in the last frames is drawn again like this.
RAS_TEMPERATURE = 1.0
RAS_TOP_P = 0.9
RAS_WINDOW = 10
# Room a voice must leave for the frames of a fragment, on top of its prompt.
PROMPT_RESERVE = 64
# Upstream's limits for a reference clip.
MIN_REFERENCE_SECONDS = 0.5
MAX_REFERENCE_SECONDS = 30.0
# Voice states held in memory, about 19 MB each for a 14 s reference.
MAX_CACHED_VOICES = 32
_SEED_MASK = 2 ** 64 - 1


def default_threads(workers: int) -> int:
    """Decoder threads per worker: the physical cores (half the logical ones) left after one LM thread per worker."""
    cores = max(1, (os.cpu_count() or 2) // 2)
    return max(1, cores // max(1, int(workers)) - 1)


# -- sampling --------------------------------------------------------------------------------------------------------


def sample_token(logits: np.ndarray, temperature: float, top_p: float, top_k: int, rng: np.random.Generator) -> int:
    """Upstream's ``_sample``: top-k and top-p decide on the logits as they are, temperature applies to what is left,
    Gumbel-max draws from it. Only the k best tokens are sorted, which is what a frame's 11 draws need."""
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).any():
        raise ValueError('model returned empty or non-finite logits')
    temperature = max(float(temperature), 1e-5)
    top_p = min(max(float(top_p), 1e-5), 1.0)
    top_k = min(max(int(top_k), 1), values.size)
    best = np.argpartition(values, values.size - top_k)[values.size - top_k:]
    best = best[np.argsort(values[best])[::-1]]
    peak = values[best[0]]
    keep = np.cumsum(np.exp(values[best] - peak)) / np.exp(values - peak).sum() <= top_p
    keep[0] = True
    best = best[keep]
    uniform = np.clip(rng.random(best.size), 1e-12, 1.0 - 1e-12)
    return int(best[np.argmax(values[best] / temperature - np.log(-np.log(uniform)))])


@dataclass(frozen=True)
class Sampling:
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    top_k: int = DEFAULT_TOP_K


# -- language model ----------------------------------------------------------------------------------------------------


@dataclass
class SlowState:
    keys: np.ndarray             # (layers, 1, heads, max_seq_len, head_dim)
    values: np.ndarray
    conv: np.ndarray
    ssm: np.ndarray


class OnnxLM:
    """The slow and fast graphs. They are per-token and their state lives here, so one instance serves all threads."""

    _SLOW_INPUTS = {'codes', 'position', 'cache_keys', 'cache_values', 'conv_states', 'ssm_states'}
    _FAST_INPUTS = {'slow_hidden', 'token_id', 'use_slow_hidden', 'input_pos'} | {
        f'cache_{kind}_{index}' for kind in ('key', 'value') for index in range(4)}

    def __init__(self, slow: Any, fast: Any, manifest: dict[str, Any]) -> None:
        self.slow = slow
        self.fast = fast
        self.books = int(manifest['num_codebooks'])
        self.codebook_size = int(manifest['codebook_size'])
        self.begin = int(manifest['semantic_begin_id'])
        self.max_seq_len = int(manifest['max_seq_len'])
        self.fast_layers = int(manifest['num_fast_layers'])
        slow_inputs = {item.name: item for item in slow.get_inputs()}
        fast_inputs = {item.name: item for item in fast.get_inputs()}
        missing = (self._SLOW_INPUTS - set(slow_inputs)) | (self._FAST_INPUTS - set(fast_inputs))
        if missing:
            raise ValueError(f'not the Audio8 0.1B graphs: inputs {sorted(missing)} are missing')
        if manifest.get('slow_logits_layout') != 'relative_semantic_then_eos':
            raise ValueError(f'unsupported slow logits layout {manifest.get("slow_logits_layout")!r}')
        self._state_shapes = {name: tuple(slow_inputs[name].shape)
                              for name in ('cache_keys', 'cache_values', 'conv_states', 'ssm_states')}
        self._fast_cache_shape = tuple(fast_inputs['cache_key_0'].shape)
        if any(not isinstance(size, int) for shape in self._state_shapes.values() for size in shape):
            raise ValueError(f'unexpected dynamic state shapes: {self._state_shapes}')
        slow_outputs = [item.name for item in slow.get_outputs()]
        self._slow_order = [slow_outputs.index(name) for name in ('logits', 'hidden', 'key_delta', 'value_delta',
                                                                  'next_conv_states', 'next_ssm_states')]
        fast_outputs = [item.name for item in fast.get_outputs()]
        self._fast_logits = fast_outputs.index('logits')
        self._fast_deltas = [(f'cache_{kind}_{index}', fast_outputs.index(f'{kind}_delta_{index}'))
                             for index in range(self.fast_layers) for kind in ('key', 'value')]

    def fresh_state(self, head: SlowState | None = None) -> SlowState:
        """Zero state, with the positions of a cached prompt head copied in."""
        state = SlowState(np.zeros(self._state_shapes['cache_keys'], dtype=np.float32),
                          np.zeros(self._state_shapes['cache_values'], dtype=np.float32),
                          np.zeros(self._state_shapes['conv_states'], dtype=np.float32),
                          np.zeros(self._state_shapes['ssm_states'], dtype=np.float32))
        if head is not None:
            length = head.keys.shape[3]
            state.keys[:, :, :, :length] = head.keys
            state.values[:, :, :, :length] = head.values
            state.conv[...] = head.conv
            state.ssm[...] = head.ssm
        return state

    def step(self, column: np.ndarray, position: int, state: SlowState) -> tuple[np.ndarray, np.ndarray]:
        """One column of the prompt or of the frames: (logits of the next semantic token, hidden for the fast graph)."""
        if not 0 <= int(position) < state.keys.shape[3]:
            raise ValueError(f'slow position {position} exceeds the cache length {state.keys.shape[3]}')
        outputs = self.slow.run(None, {
            'codes': column, 'position': np.asarray([int(position)], dtype=np.int64), 'cache_keys': state.keys,
            'cache_values': state.values, 'conv_states': state.conv, 'ssm_states': state.ssm})
        logits, hidden, key, value, conv, ssm = (outputs[index] for index in self._slow_order)
        state.keys[:, :, :, int(position)] = key
        state.values[:, :, :, int(position)] = value
        state.conv = conv
        state.ssm = ssm
        return logits[0, -1], hidden[:, -1:, :]

    def _fast_step(self, hidden: np.ndarray, token: int, use_hidden: bool, position: int,
                   caches: dict[str, np.ndarray]) -> np.ndarray:
        outputs = self.fast.run(None, {
            'slow_hidden': hidden, 'token_id': np.asarray([[int(token)]], dtype=np.int64),
            'use_slow_hidden': np.asarray([bool(use_hidden)]), 'input_pos': np.asarray([int(position)], dtype=np.int64),
            **caches})
        for name, index in self._fast_deltas:
            caches[name][:, :, int(position):int(position) + 1] = outputs[index]
        return outputs[self._fast_logits][0, -1]

    def sample_semantic(self, logits: np.ndarray, previous: collections.deque[int], sampling: Sampling,
                        rng: np.random.Generator) -> int | None:
        """A code of the first codebook, or None for EOS."""
        values = np.asarray(logits, dtype=np.float32).reshape(-1)
        eos = self.codebook_size
        if values.size != eos + 1:
            raise ValueError(f'unexpected slow logits size {values.size}, expected {eos + 1}')
        if not previous:
            values = values.copy()
            values[eos] = -np.inf
        normal = sample_token(values, sampling.temperature, sampling.top_p, sampling.top_k, rng)
        repeat = sample_token(values, RAS_TEMPERATURE, RAS_TOP_P, sampling.top_k, rng)
        code = repeat if normal != eos and normal in previous else normal
        return None if code == eos else code

    def sample_frame(self, first: int, hidden: np.ndarray, sampling: Sampling, rng: np.random.Generator) -> np.ndarray:
        """All codebooks of a frame whose first code is known."""
        caches = {f'cache_{kind}_{index}': np.zeros(self._fast_cache_shape, dtype=np.float32)
                  for kind in ('key', 'value') for index in range(self.fast_layers)}
        self._fast_step(hidden, 0, True, 0, caches)
        frame = np.empty(self.books, dtype=np.int64)
        frame[0] = token = first
        for book in range(1, self.books):
            logits = self._fast_step(hidden, token, False, book, caches)
            frame[book] = token = sample_token(logits, sampling.temperature, sampling.top_p, sampling.top_k, rng)
        return frame

    def column(self, frame: np.ndarray) -> np.ndarray:
        out = np.empty((1, self.books + 1, 1), dtype=np.int64)
        out[0, 0, 0] = int(frame[0]) + self.begin
        out[0, 1:, 0] = frame
        return out

    def text_column(self, token: int) -> np.ndarray:
        out = np.zeros((1, self.books + 1, 1), dtype=np.int64)
        out[0, 0, 0] = int(token)
        return out


# -- engine ------------------------------------------------------------------------------------------------------------


class _Tokenizer:
    """``encode`` in the shape text.py expects, from the tokenizers package alone."""

    def __init__(self, path: Path) -> None:
        from tokenizers import Tokenizer

        self._tokenizer = Tokenizer.from_file(str(path))

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=add_special_tokens).ids)


@dataclass
class Audio8OnnxVoice:
    key: str
    ref_text: str | None
    codes: np.ndarray | None     # (books, T)
    state: SlowState             # after the prompt head; keys / values hold only its positions
    duration_s: float

    @property
    def length(self) -> int:
        return int(self.state.keys.shape[3])


@dataclass
class _Fragment:
    text: str
    ids: list[int]
    cap: int


@dataclass(eq=False)
class _Row:
    row_id: str
    fragments: list[_Fragment]
    prompt: Audio8OnnxVoice
    seed: int
    stream: Audio8OnnxStream
    resampler: StreamResampler
    cancelled: bool = False
    first_sent: bool = False
    total_frames: int = 0


@dataclass(eq=False)
class _Job:
    """The frames of one fragment on their way from the LM thread to the decoder thread."""
    row: _Row
    frames: np.ndarray
    count: int = 0
    closed: bool = False
    last: bool = False           # no fragment of the row follows
    error: BaseException | None = None
    changed: threading.Condition = field(default_factory=threading.Condition)

    def publish(self, frame: np.ndarray) -> None:
        self.frames[self.count] = frame
        with self.changed:
            self.count += 1
            self.changed.notify_all()

    def close(self, *, last: bool, error: BaseException | None = None) -> None:
        with self.changed:
            self.closed = True
            self.last = last
            self.error = error
            self.changed.notify_all()

    def wait(self, ready: int) -> tuple[int, bool]:
        """Blocks until ``ready`` frames exist or the LM is done: (frames, LM done)."""
        with self.changed:
            while self.count < ready and not self.closed:
                self.changed.wait()
            return self.count, self.closed


class _Worker:
    def __init__(self, index: int, decoder: Any) -> None:
        self.index = index
        self.decoder = decoder
        self.jobs: collections.deque[_Job | None] = collections.deque()
        self.available = threading.Condition()

    def put(self, job: _Job | None) -> None:
        with self.available:
            self.jobs.append(job)
            self.available.notify_all()

    def take(self) -> _Job | None:
        with self.available:
            while not self.jobs:
                self.available.wait()
            return self.jobs.popleft()


class Audio8OnnxEngine:
    def __init__(self, model_dir: Path | str, *, workers: int = 2, threads: int = 0,
                 output_sample_rate: int = NATIVE_SAMPLE_RATE, first_chunk_frames: int = 6, chunk_frames: int = 8,
                 fragment_max_chars: int = DEFAULT_FRAGMENT_MAX_CHARS, temperature: float = DEFAULT_TEMPERATURE,
                 top_p: float = DEFAULT_TOP_P, top_k: int = DEFAULT_TOP_K, voice_cache_dir: Path | str | None = None,
                 decoder_factory: Callable[..., Any] | None = None) -> None:
        self.model_dir = Path(model_dir)
        self.workers = max(1, int(workers))
        self.threads = int(threads) or default_threads(self.workers)
        self.output_sample_rate = int(output_sample_rate)
        self.first_chunk_frames = max(1, int(first_chunk_frames))
        self.chunk_frames = max(1, int(chunk_frames))
        self.fragment_max_chars = max(MIN_FRAGMENT_CHARS, int(fragment_max_chars))
        self.sampling = Sampling(float(temperature), float(top_p), int(top_k))
        self.voice_cache_dir = Path(voice_cache_dir) if voice_cache_dir else None
        self._decoder_factory = decoder_factory or StreamDecoder
        self.manifest: dict[str, Any] = {}
        self.lm: OnnxLM | None = None
        self.tokenizer: Any = None
        self.hop = 0
        self._workers: list[_Worker] = []
        self._threads: list[threading.Thread] = []
        self._cv = threading.Condition()
        self._pending: collections.deque[_Row] = collections.deque()
        self._active: set[_Row] = set()
        self._stop = False
        self._loaded = False
        self._voices: collections.OrderedDict[str, Audio8OnnxVoice] = collections.OrderedDict()
        self._voice_lock = threading.Lock()
        self._encode_lock = threading.Lock()
        self._plain_voice: Audio8OnnxVoice | None = None
        self._ids = itertools.count()
        self._frame_ms_ema = 0.0
        self._decode_ms_ema = 0.0
        self.load_s = 0.0

    # -- lifecycle -----------------------------------------------------------------------------------------------------

    def _read_manifest(self) -> dict[str, Any]:
        path = self.model_dir / 'runtime_manifest.json'
        if not path.is_file():
            raise FileNotFoundError(f'Audio8 runtime manifest not found: {path}')
        return json.loads(path.read_text(encoding='utf-8'))

    def _open_lm(self, manifest: dict[str, Any]) -> OnnxLM:
        precision = manifest['default_precision']
        return OnnxLM(cpu_session(self.model_dir / manifest['slow_decode_models'][precision], 1),
                      cpu_session(self.model_dir / manifest['fast_models'][precision], 1), manifest)

    def _open_tokenizer(self) -> Any:
        return _Tokenizer(self.model_dir / 'tokenizer' / 'tokenizer.json')

    def load(self) -> None:
        if self._loaded:
            return
        started = time.perf_counter()
        self.manifest = manifest = self._read_manifest()
        if int(manifest['sample_rate']) != NATIVE_SAMPLE_RATE:
            raise ValueError(f'unexpected sample rate {manifest["sample_rate"]}')
        self.hop = int(manifest['codec_hop_length'])
        self.tokenizer = self._open_tokenizer()
        self.lm = self._open_lm(manifest)
        codec = self.model_dir / manifest['codec_models'][manifest['default_codec_precision']]
        self._workers = [_Worker(index, self._decoder_factory(codec, self.threads, self.hop))
                         for index in range(self.workers)]
        self._plain_voice = self._build_voice('', None, None, 0.0)
        warm = np.zeros((self.first_chunk_frames + self.chunk_frames, self.lm.books), dtype=np.int64)
        for worker in self._workers:  # the first run of a session allocates its buffers
            worker.decoder.decode(warm, 0, warm.shape[0])
        self._stop = False
        self._threads = []
        for worker in self._workers:
            for name, target in (('lm', self._lm_loop), ('codec', self._decode_loop)):
                thread = threading.Thread(target=target, args=(worker,), name=f'audio8-onnx-{name}-{worker.index}',
                                          daemon=True)
                thread.start()
                self._threads.append(thread)
        self._loaded = True
        self.load_s = time.perf_counter() - started

    def close(self) -> None:
        with self._cv:
            self._stop = True
            pending, self._pending = self._pending, collections.deque()
            self._cv.notify_all()
        error = RuntimeError('Audio8 engine closed')
        for row in pending:
            row.stream._finish(error)
        for thread in self._threads:
            thread.join(timeout=30)
        self._threads = []
        self._workers = []
        self.lm = None
        self._plain_voice = None
        with self._voice_lock:
            self._voices.clear()
        self._loaded = False
        gc.collect()

    @property
    def loaded(self) -> bool:
        return self._loaded

    # -- voices --------------------------------------------------------------------------------------------------------

    def _cache_path(self, voice_id: str | None, wave: np.ndarray, sample_rate: int) -> Path | None:
        """The codes depend on the model and the audio alone, so a transcript sent with a request adds no file."""
        if self.voice_cache_dir is None or voice_id is None:
            return None
        audio = hashlib.sha256(wave.tobytes()).hexdigest()
        return voice_file(self.voice_cache_dir, voice_id, f'{self.manifest["model_fingerprint"]}|{sample_rate}|{audio}')

    def _read_codes(self, path: Path | None) -> np.ndarray | None:
        if path is None or not path.is_file():
            return None
        try:
            codes = np.load(path, allow_pickle=False).astype(np.int64)
            self._check_codes(codes)
            return codes
        except (OSError, ValueError) as exc:
            logger.warning('Audio8: ignoring the unreadable voice cache %s: %s', path.name, exc)
            return None

    def _write_codes(self, path: Path | None, codes: np.ndarray) -> None:
        if path is None:
            return
        temp = path.with_suffix(f'.{os.getpid()}.{threading.get_ident()}.tmp')
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(temp, 'wb') as handle:
                np.save(handle, codes.astype(np.uint16), allow_pickle=False)
            os.replace(temp, path)
        except OSError as exc:  # only a head start for the next process
            logger.warning('Audio8: could not save the voice cache %s: %s', path.name, exc)
            temp.unlink(missing_ok=True)

    def _check_codes(self, codes: np.ndarray) -> None:
        size = int(self.manifest['codebook_size'])
        if codes.ndim != 2 or codes.shape[0] != int(self.manifest['num_codebooks']) or codes.shape[1] == 0:
            raise ValueError(f'invalid reference codes of shape {codes.shape}')
        if int(codes.min()) < 0 or int(codes.max()) >= size:
            raise ValueError(f'reference codes outside [0, {size})')

    def _encode_reference(self, wave: np.ndarray) -> np.ndarray:
        """The encoder is only needed here and is the largest graph, so it is loaded per call and released, as upstream."""
        path = self.model_dir / 'registration' / 'codec_encoder_fp16.onnx'
        padding = -wave.size % self.hop
        audio = np.pad(wave, (0, padding)) if padding else wave
        session = cpu_session(path, self.threads * self.workers)
        try:
            info = session.get_inputs()[0]
            kind = np.float16 if info.type == 'tensor(float16)' else np.float32
            codes = np.asarray(session.run(None, {info.name: audio.reshape(1, 1, -1).astype(kind)})[0])
        finally:
            del session
            gc.collect()
        codes = (codes[0] if codes.ndim == 3 else codes).astype(np.int64)
        self._check_codes(codes)
        return codes

    def _head_columns(self, ids: list[int], codes: np.ndarray | None, tail: list[int]) -> np.ndarray:
        """Columns of the prompt head, all that comes before the target text: (books + 1, P)."""
        books = int(self.manifest['num_codebooks'])
        length = 0 if codes is None else codes.shape[1]
        head = np.zeros((books + 1, len(ids) + length + len(tail)), dtype=np.int64)
        head[0, :len(ids)] = ids
        if codes is not None:
            head[0, len(ids):len(ids) + length] = codes[0] + int(self.manifest['semantic_begin_id'])
            head[1:, len(ids):len(ids) + length] = codes
        head[0, len(ids) + length:] = tail
        return head

    def _build_voice(self, key: str, ref_text: str | None, codes: np.ndarray | None, duration_s: float) -> Audio8OnnxVoice:
        lm = self.lm
        head = self._head_columns(prefix_ids(self.tokenizer, ref_text), codes, user_turn_ids(self.tokenizer))
        length = head.shape[1]
        if length >= lm.max_seq_len - PROMPT_RESERVE:
            raise ValueError(f'Audio8 voice prompt too long: {length} of {lm.max_seq_len} positions; use a shorter '
                             'reference clip')
        state = lm.fresh_state()
        for position in range(length):
            lm.step(head[np.newaxis, :, position:position + 1], position, state)
        state.keys = state.keys[:, :, :, :length].copy()
        state.values = state.values[:, :, :, :length].copy()
        return Audio8OnnxVoice(key=key, ref_text=ref_text, codes=codes, state=state, duration_s=duration_s)

    def voice_prompt(self, key: str, audio: np.ndarray, sample_rate: int, ref_text: str, *,
                     voice_id: str | None = None) -> Audio8OnnxVoice:
        """Encode (once per key) a reference voice and run its prompt head. The codes of a saved voice (``voice_id``)
        stay on disk across restarts (``voice_cache.py``); other references stay in memory."""
        with self._voice_lock:
            cached = self._voices.get(key)
            if cached is not None:
                self._voices.move_to_end(key)
                return cached
        if not self._loaded:
            raise RuntimeError('Audio8 engine is not loaded')
        ref_text = ' '.join(str(ref_text or '').split())
        if not ref_text:
            raise ValueError('Audio8 voice cloning needs the reference transcript (ref_text)')
        wave = np.asarray(audio, dtype=np.float32)
        if wave.ndim == 2:  # (samples, channels) or (channels, samples)
            wave = wave.mean(axis=1 if wave.shape[0] > wave.shape[1] else 0)
        wave = wave.reshape(-1)
        duration = wave.size / float(sample_rate)
        if not MIN_REFERENCE_SECONDS <= duration <= MAX_REFERENCE_SECONDS:
            raise ValueError(f'reference audio must be {MIN_REFERENCE_SECONDS:g} to {MAX_REFERENCE_SECONDS:g} seconds '
                             f'long, got {duration:.1f}')
        if not np.isfinite(wave).all():
            raise ValueError('reference audio contains non-finite samples')
        with self._encode_lock:  # one encoder at a time, and two requests for a new voice encode it once
            with self._voice_lock:
                cached = self._voices.get(key)
            if cached is not None:
                return cached
            path = self._cache_path(voice_id, wave, int(sample_rate))
            codes = self._read_codes(path)
            if codes is None:
                if int(sample_rate) != NATIVE_SAMPLE_RATE:
                    wave = StreamResampler(int(sample_rate), NATIVE_SAMPLE_RATE).push(wave, last=True)
                codes = self._encode_reference(wave)
                self._write_codes(path, codes)
            prompt = self._build_voice(key, ref_text, codes, duration)
            with self._voice_lock:
                self._voices[key] = prompt
                while len(self._voices) > MAX_CACHED_VOICES:
                    self._voices.popitem(last=False)
            return prompt

    def has_voice(self, key: str) -> bool:
        with self._voice_lock:
            return key in self._voices

    def drop_voice(self, key: str) -> None:
        """Called when voices change. The files go with the voice's profile, see ``voice_cache.py``."""
        with self._voice_lock:
            self._voices.pop(key, None)

    # -- rows ----------------------------------------------------------------------------------------------------------

    def _build_fragments(self, text: str, voice: Audio8OnnxVoice) -> list[_Fragment]:
        limit = self.lm.max_seq_len
        room = limit - voice.length - PROMPT_RESERVE
        max_chars = min(self.fragment_max_chars, room // EXPECTED_FRAMES_PER_CHAR)
        if max_chars < MIN_FRAGMENT_CHARS:
            raise ValueError(f'Audio8 voice prompt too long: {voice.length} of {limit} positions; use a shorter '
                             'reference clip')
        out = []
        cached = len(user_turn_ids(self.tokenizer))  # part of the voice's prompt head
        for piece in split_fragments(text, max_chars):
            ids = suffix_ids(self.tokenizer, piece)[cached:]
            budget = limit - voice.length - len(ids) - 1
            if budget <= 8:
                raise ValueError(f'Audio8 prompt too long: {voice.length + len(ids)} of {limit} positions used by voice '
                                 f'prompt and text fragment {piece!r}')
            out.append(_Fragment(piece, ids, min(budget, frame_cap(piece))))
        return out

    def submit(self, text: str, *, prompt: Audio8OnnxVoice | None, seed: int,
               row_id: str | None = None) -> Audio8OnnxStream:
        """Queue one row. ``prompt`` None is plain synthesis without a reference voice."""
        if not self._loaded:
            raise RuntimeError('Audio8 engine is not loaded')
        row_id = row_id or f'audio8-onnx-{next(self._ids)}'
        voice = prompt if prompt is not None else self._plain_voice
        fragments = self._build_fragments(text, voice)
        stream = Audio8OnnxStream(row_id, self)
        row = _Row(row_id=row_id, fragments=fragments, prompt=voice, seed=int(seed), stream=stream,
                   resampler=StreamResampler(NATIVE_SAMPLE_RATE, self.output_sample_rate))
        stream._row = row
        stream.metrics['fragments'] = len(fragments)
        with self._cv:
            if self._stop:
                raise RuntimeError('Audio8 engine is not running')
            self._pending.append(row)
            self._cv.notify_all()
        return stream

    def render(self, text: str, *, prompt: Audio8OnnxVoice | None = None, seed: int = 0,
               row_id: str | None = None) -> np.ndarray:
        chunks = list(self.submit(text, prompt=prompt, seed=seed, row_id=row_id))
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    def stats(self) -> dict[str, Any]:
        with self._cv:
            active, pending = len(self._active), len(self._pending)
        return {
            'active_rows': active, 'pending_rows': pending, 'workers': self.workers, 'threads': self.threads,
            'voices': len(self._voices), 'loaded': self._loaded, 'frame_ms': round(self._frame_ms_ema, 3),
            'decode_ms': round(self._decode_ms_ema, 3), 'first_chunk_frames': self.first_chunk_frames,
            'chunk_frames': self.chunk_frames, 'load_s': round(self.load_s, 3),
        }

    def _cancel(self, stream: Audio8OnnxStream) -> None:
        row = stream._row
        if row is None:
            return
        row.cancelled = True
        with self._cv:
            waiting = row in self._pending
            if waiting:
                self._pending.remove(row)
        if waiting:
            row.stream._finish()

    # -- LM thread -----------------------------------------------------------------------------------------------------

    def _next_row(self) -> _Row | None:
        with self._cv:
            while not self._pending and not self._stop:
                self._cv.wait()
            if self._stop:
                return None
            row = self._pending.popleft()
            self._active.add(row)
            row.stream.started_at = time.perf_counter()
            return row

    def _lm_loop(self, worker: _Worker) -> None:
        while True:
            row = self._next_row()
            if row is None:
                worker.put(None)
                return
            self._render_row(row, worker)

    def _render_row(self, row: _Row, worker: _Worker) -> None:
        for index, fragment in enumerate(row.fragments):
            job = _Job(row, np.empty((fragment.cap, self.lm.books), dtype=np.int64))
            worker.put(job)
            try:
                self._generate(row, fragment, index, job)
            except Exception as exc:  # noqa: BLE001 - fails this row, the worker keeps serving
                logger.exception('Audio8 row %s failed', row.row_id)
                job.close(last=True, error=exc)
                return
            stopping = row.cancelled or self._stop
            job.close(last=stopping or index == len(row.fragments) - 1,
                      error=RuntimeError('Audio8 engine closed') if self._stop else None)
            if stopping:
                return

    def _generate(self, row: _Row, fragment: _Fragment, index: int, job: _Job) -> None:
        lm = self.lm
        rng = np.random.default_rng([row.seed & _SEED_MASK, index])
        state = lm.fresh_state(row.prompt.state)
        position = row.prompt.length
        logits: np.ndarray | None = None
        hidden: np.ndarray | None = None
        for token in fragment.ids:
            if row.cancelled or self._stop:
                return
            logits, hidden = lm.step(lm.text_column(token), position, state)
            position += 1
        previous: collections.deque[int] = collections.deque(maxlen=RAS_WINDOW)
        for count in range(fragment.cap):
            if row.cancelled or self._stop:
                return
            started = time.perf_counter()
            first = lm.sample_semantic(logits, previous, self.sampling, rng)
            if first is None:
                return
            previous.append(first)
            frame = lm.sample_frame(first, hidden, self.sampling, rng)
            job.publish(frame)
            row.total_frames += 1
            if count + 1 == fragment.cap:
                row.stream.metrics['frame_cap_hits'] = row.stream.metrics.get('frame_cap_hits', 0) + 1
                return
            logits, hidden = lm.step(lm.column(frame), position, state)
            position += 1
            elapsed = (time.perf_counter() - started) * 1000.0
            self._frame_ms_ema = elapsed if self._frame_ms_ema == 0.0 else 0.98 * self._frame_ms_ema + 0.02 * elapsed

    # -- decoder thread ------------------------------------------------------------------------------------------------

    def _decode_loop(self, worker: _Worker) -> None:
        while True:
            job = worker.take()
            if job is None:
                return
            try:
                self._decode_job(worker, job)
            except Exception as exc:  # noqa: BLE001 - fails this row, the worker keeps serving
                logger.exception('Audio8 decode of row %s failed', job.row.row_id)
                job.row.cancelled = True  # its LM thread stops at the next step
                self._finish(job.row, exc)

    def _decode_job(self, worker: _Worker, job: _Job) -> None:
        row = job.row
        trimmer = SilenceTrimmer(NATIVE_SAMPLE_RATE)
        decoded = 0
        while not row.stream.done:  # done: an earlier fragment failed
            ready = decoded + (self.chunk_frames if row.first_sent else self.first_chunk_frames)
            count, closed = job.wait(ready)
            if row.cancelled or job.error is not None:
                job.wait(job.frames.shape[0] + 1)  # the LM thread closes the job at its next step
                if job.last:
                    self._finish(row, job.error)
                return
            end = min(count, ready)
            final = closed and end == count
            audio = None
            if end > decoded:
                started = time.perf_counter()
                audio = worker.decoder.decode(job.frames, decoded, end)
                elapsed = (time.perf_counter() - started) * 1000.0
                self._decode_ms_ema = elapsed if self._decode_ms_ema == 0.0 else 0.9 * self._decode_ms_ema + 0.1 * elapsed
                decoded = end
            if final:
                out = trimmer.finish(audio)
            elif audio is not None:
                out = trimmer.push(audio)
            else:
                out = np.zeros(0, dtype=np.float32)
            if out.size:
                row.first_sent = True
                row.stream._put(row.resampler.push(out))
            if final:
                if job.last:
                    row.stream._put(row.resampler.flush())
                    row.stream.metrics['frames'] = row.total_frames
                    self._finish(row, None)
                return

    def _finish(self, row: _Row, error: BaseException | None) -> None:
        with self._cv:
            self._active.discard(row)
        row.stream._finish(error)
