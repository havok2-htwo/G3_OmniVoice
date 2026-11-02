"""Audio8 slot engine: continuous batching, per-row determinism, true chunk streaming.

One dedicated GPU thread owns every CUDA operation. Each loop iteration ("step") does, for all active rows at once:

1. prefill of rows that join (or start their next text fragment): the voice KV is copied into the slot and only the
   text is run, in fixed tiles of rows,
2. ``sample``: one code frame per row,
3. ONE device->host copy (the EOS flags); ``slow``, the next transformer step of every row, is enqueued at once, so the
   GPU never waits for the host bookkeeping,
4. while it runs, the CPU hands out finished audio (trailing-silence trim, resampling, streams),
5. fixed tiles of rows whose chunk is due go to the codec on a second CUDA stream, so they overlap the next LM steps.
   A chunk that nobody waits for is held for a few steps until a tile is full; a row's first audio and a fragment's end
   go out at once.

A finished slot is free for the next row at the step boundary. With ``cuda_graphs`` every GPU part replays a graph
captured at load; without, the same functions run eagerly.

``batch_invariant=True`` (default) runs every GPU call at a fixed shape, so a row's audio is bit-identical whatever
else runs: a row's numbers depend on its text, voice, seed and on nothing else. ``batch_invariant=False`` uses the smallest
slot bucket per LM call, which is faster at low load, but rows then differ at bf16-noise level from their neighbours (and
sampling turns a last-bit difference into another, equally valid, rendition).
"""
from __future__ import annotations

import collections
import contextlib
import heapq
import itertools
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from ..capacity import AUDIO8_VOICE_CACHE_MB as MAX_VOICE_CACHE_MB
from ..ming.engine import MingStream as Audio8Stream
from ..ming.engine import _engine_stream, bucket_for
from ..ming.streaming import SilenceTrimmer, StreamResampler
from .codec_stream import SlotCodec
from .lm import SlotLM
from .noise import row_key
from .text import (DEFAULT_FRAGMENT_MAX_CHARS, EXPECTED_FRAMES_PER_CHAR, MIN_FRAGMENT_CHARS, frame_cap, prefix_ids,
                   split_fragments, suffix_ids)

logger = logging.getLogger(__name__)

NATIVE_SAMPLE_RATE = 44100
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 50
# Positions per row: reference + text + audio. Upstream allows 2048; attention cost grows with this, and 1024 leaves
# room for a 20 s reference plus a fragment of 200 characters.
DEFAULT_CONTEXT_LENGTH = 1024
# Prefill call shapes: text tokens per call. Longer text takes several calls, the prefill is incremental.
PREFILL_LENGTHS = (32, 64, 96, 128)
# Room a voice must leave for the frames of a fragment, on top of its prompt (the longest fixed suffix and some slack).
PROMPT_RESERVE = 64


def _language_model(model: Any) -> Callable[..., Any]:
    if getattr(model.config, 'slow_backbone', None) == 'falcon_h1':
        from .falcon_lm import FalconSlotLM

        return FalconSlotLM
    return SlotLM


def prefill_length(tokens: int) -> int:
    for length in PREFILL_LENGTHS:
        if tokens <= length:
            return length
    return PREFILL_LENGTHS[-1]


@dataclass
class Audio8VoicePrompt:
    key: str
    ref_text: str | None
    codes: torch.Tensor | None   # (books, T) long, CPU
    keys: torch.Tensor           # (layers, heads, P, head_dim) on device
    values: torch.Tensor
    duration_s: float
    state: Any = None            # what else the language model keeps per voice (FalconSlotLM: the Mamba state)

    @property
    def length(self) -> int:
        return int(self.keys.shape[2])

    @property
    def size_bytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in (self.keys, self.values, *(self.state or ())))


@dataclass
class _Fragment:
    text: str
    ids: list[int]
    cap: int                     # most frames the fragment may take


@dataclass
class _Row:
    row_id: str
    fragments: list[_Fragment]
    prompt: Audio8VoicePrompt
    seed: int
    stream: Audio8Stream
    trimmer: SilenceTrimmer
    resampler: StreamResampler
    slot: int = -1
    frag_index: int = 0
    state: str = 'prefill'       # prefill | run
    step: int = 0                # sample calls in the current fragment
    frames: int = 0              # code frames of the current fragment
    decoded: int = 0             # of those, handed to the codec
    pos: int = 0                 # KV positions used in the slot
    fed: int = 0                 # text tokens of the fragment already prefilled
    key: int = 0                 # noise key of the current fragment
    first_sent: bool = False     # the row's first chunk is out
    cancelled: bool = False
    total_steps: int = 0

    @property
    def fragment(self) -> _Fragment:
        return self.fragments[self.frag_index]


@dataclass
class _Item:
    row: _Row
    slot: int
    start: int                   # first frame of this chunk within the fragment
    n: int                       # frames in this chunk
    final: bool                  # last chunk of the fragment
    last: bool                   # last chunk of the row
    urgent: bool                 # first audio of the row, or the slot is about to be reused
    born: int                    # engine step the item became due


@dataclass
class _Tile:
    """One decode call in flight, with its own staging: several can be on the way at once."""
    chunk: int = 0
    items: list[_Item] = field(default_factory=list)
    stage: Any = None            # (rows, chunk_max, books) long on device: frames, gathered on the main stream
    index_host: Any = None
    slot_host: Any = None
    index_dev: Any = None
    slot_dev: Any = None
    meta_host: Any = None
    out_host: Any = None
    event: Any = None
    launched_at: float = 0.0


class Audio8Engine:
    def __init__(self, model_dir: Path | str, device: str = 'cuda:0', *, max_slots: int = 32, cuda_graphs: bool = True,
                 output_sample_rate: int = NATIVE_SAMPLE_RATE, first_chunk_frames: int = 4, chunk_frames: int = 12,
                 fragment_max_chars: int = DEFAULT_FRAGMENT_MAX_CHARS, temperature: float = DEFAULT_TEMPERATURE,
                 top_p: float = DEFAULT_TOP_P, top_k: int = DEFAULT_TOP_K, context_length: int = DEFAULT_CONTEXT_LENGTH,
                 dtype: torch.dtype = torch.bfloat16, codec_dtype: torch.dtype | None = None,
                 batch_invariant: bool = True, decode_tile_rows: int | None = None, prefill_tile_rows: int = 8,
                 decode_hold_steps: int = 4, loader: Callable[..., tuple[Any, Any, Any]] | None = None,
                 lm_factory: Callable[..., Any] | None = None, codec_factory: Callable[..., Any] | None = None) -> None:
        self.model_dir = Path(model_dir)
        self.device = torch.device(device)
        self.max_slots = max(1, min(64, int(max_slots)))
        self.cuda_graphs = bool(cuda_graphs) and self.device.type == 'cuda'
        self.output_sample_rate = int(output_sample_rate)
        self.chunk_frames = max(1, int(chunk_frames))
        self.first_chunk_frames = max(1, min(int(first_chunk_frames), self.chunk_frames))
        self.fragment_max_chars = max(MIN_FRAGMENT_CHARS, int(fragment_max_chars))
        self.temperature, self.top_p, self.top_k = float(temperature), float(top_p), int(top_k)
        self.context_length = int(context_length)
        self.dtype = dtype
        self.codec_dtype = codec_dtype or dtype
        self.batch_invariant = bool(batch_invariant)
        tile = max(8, self.max_slots // 4) if decode_tile_rows is None else int(decode_tile_rows)
        self.decode_tile_rows = max(1, min(self.max_slots, tile))
        self.prefill_tile_rows = max(1, min(self.max_slots, int(prefill_tile_rows)))
        self.decode_hold_steps = max(0, int(decode_hold_steps))
        self._loader = loader
        self._lm_factory = lm_factory
        self._codec_factory = codec_factory or SlotCodec
        self.model: Any = None
        self.codec: Any = None
        self.tokenizer: Any = None
        self.lm: Any = None
        self.slot_codec: Any = None
        self._voices: collections.OrderedDict[str, Audio8VoicePrompt] = collections.OrderedDict()
        self._voice_lock = threading.Lock()
        self._plain_voice: Audio8VoicePrompt | None = None
        self._cv = threading.Condition()
        self._pending: collections.deque[_Row] = collections.deque()
        self._tasks: collections.deque = collections.deque()
        self._rows: dict[int, _Row] = {}
        self._free_slots: list[int] = []
        self._thread: threading.Thread | None = None
        self._stop = False
        self._ids = itertools.count(1)
        self._loaded = False
        self._graphs: dict[tuple, torch.cuda.CUDAGraph] = {}
        self._graph_pools: Any = None
        self._ahead_bk: int | None = None
        self._due: collections.deque[_Item] = collections.deque()
        self._inflight: collections.deque[_Tile] = collections.deque()
        self._free_tiles: list[_Tile] = []
        self._step_ms_ema = 0.0
        self._decode_ms_ema = 0.0
        self._prefill_ms_ema = 0.0
        self._steps_total = 0
        self._last_bucket = 0
        self.capture_s = 0.0
        self.warmup_s = 0.0

    # -- lifecycle ---------------------------------------------------------------------------------------------------

    def load(self) -> None:
        if self._loaded:
            return
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
        if self._loader is not None:
            self.model, self.codec, self.tokenizer = self._loader(self.model_dir, self.device, self.dtype,
                                                                  self.codec_dtype)
        else:
            from .model import load_audio8_model

            self.model, self.codec, self.tokenizer = load_audio8_model(self.model_dir, self.device, self.dtype,
                                                                       self.codec_dtype)
        if self.top_k > self.model.config.codebook_size:
            self.top_k = self.model.config.codebook_size
        self._stop = False
        self._free_slots = list(range(self.max_slots))
        heapq.heapify(self._free_slots)
        ready: dict[str, Any] = {'event': threading.Event(), 'error': None}
        self._thread = threading.Thread(target=self._loop, args=(ready,), name='audio8-gpu', daemon=True)
        self._thread.start()
        ready['event'].wait()
        if ready['error'] is not None:
            self._thread.join(timeout=30)
            self._thread = None
            self._free_all()
            raise ready['error']
        self._loaded = True

    def _allocate(self) -> None:
        S, dev = self.max_slots, self.device
        pin = dev.type == 'cuda'
        factory = self._lm_factory or _language_model(self.model)
        self.lm = factory(self.model, slots=S, length=self.context_length, top_k=self.top_k, top_p=self.top_p,
                                   temperature=self.temperature, device=dev, dtype=self.dtype)
        self.slot_codec = self._codec_factory(self.codec, slots=S, max_chunk=self.chunk_frames,
                                              max_frames=self.context_length, device=dev, dtype=self.codec_dtype)
        self.books = self.lm.books
        self.pos_host = [torch.zeros(S, dtype=torch.long, pin_memory=pin) for _ in range(2)]
        self.pos_np = [t.numpy() for t in self.pos_host]
        self._pos_flip = 0
        self.step_host = torch.zeros((2, S), dtype=torch.long, pin_memory=pin)
        self.step_np = self.step_host.numpy()
        self.eos_host = torch.zeros(S, dtype=torch.bool, pin_memory=pin)
        self.eos_np = self.eos_host.numpy()
        # Prefill staging, rewritten every step: ids, then (slot, start, last) per row of a call.
        self.prefill_host = torch.zeros((4 * S + 16) * (PREFILL_LENGTHS[-1] + 3), dtype=torch.long, pin_memory=pin)
        self.prefill_np = self.prefill_host.numpy()
        self._prefill_shapes: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
        self._decode_shapes: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
        self.chunks = tuple(sorted({self.first_chunk_frames, self.chunk_frames}))
        self._all_tiles: list[_Tile] = []
        rows = self.decode_tile_rows
        cmax = self.chunk_frames
        out_len = self.slot_codec.out_len(cmax)
        for _ in range(S // rows + 3):
            tile = _Tile(stage=torch.zeros(rows, cmax, self.books, device=dev, dtype=torch.long),
                         index_host=torch.zeros(rows, cmax, dtype=torch.long, pin_memory=pin),
                         slot_host=torch.zeros(rows, 1, dtype=torch.long, pin_memory=pin),
                         index_dev=torch.zeros(rows, cmax, dtype=torch.long, device=dev),
                         slot_dev=torch.zeros(rows, 1, dtype=torch.long, device=dev),
                         meta_host=torch.zeros(rows, 4, dtype=torch.long, pin_memory=pin),
                         out_host=torch.zeros(rows, out_len, dtype=torch.float32, pin_memory=pin))
            self._all_tiles.append(tile)
        self._free_tiles = list(self._all_tiles)
        if dev.type == 'cuda':
            self.decode_stream = _engine_stream(dev, 'audio8_decode', priority=-1)
            self.eos_event = torch.cuda.Event()
        else:  # CPU runs are for tests of the host logic
            self.decode_stream = None
            self.eos_event = None

    def _prefill_shape(self, rows: int, length: int) -> dict[str, torch.Tensor]:
        buf = self._prefill_shapes.get((rows, length))
        if buf is None:
            dev = self.device
            buf = {'ids': torch.zeros(rows, length, device=dev, dtype=torch.long),
                   'meta': torch.zeros(rows, 3, device=dev, dtype=torch.long)}
            buf['meta'][:, 0] = self.max_slots
            self._prefill_shapes[(rows, length)] = buf
        return buf

    def _decode_shape(self, chunk: int) -> dict[str, torch.Tensor]:
        rows = self.decode_tile_rows
        buf = self._decode_shapes.get((rows, chunk))
        if buf is None:
            dev = self.device
            buf = {'codes': torch.zeros(rows, chunk, self.books, device=dev, dtype=torch.long),
                   'meta': torch.zeros(rows, 4, device=dev, dtype=torch.long),
                   'out': torch.zeros(rows, self.slot_codec.out_len(chunk), device=dev, dtype=torch.float32)}
            buf['meta'][:, 0] = self.slot_codec.trash
            self._decode_shapes[(rows, chunk)] = buf
        return buf

    def lm_buckets(self) -> list[int]:
        if self.batch_invariant:
            return [self.max_slots]
        return sorted({bucket_for(n, self.max_slots) for n in range(1, self.max_slots + 1)})

    def prefill_shapes(self) -> list[tuple[int, int]]:
        return [(self.prefill_tile_rows, length) for length in PREFILL_LENGTHS]

    def _prepare(self) -> None:
        """Warm up every shape (cuBLAS, cuDNN, FFT plans), then capture the CUDA graphs."""
        t0 = time.perf_counter()
        cuda = self.device.type == 'cuda'
        lm, codec = self.lm, self.slot_codec

        def run_prefill(buf):
            lm.prefill(buf['ids'], buf['meta'][:, 0], buf['meta'][:, 1], buf['meta'][:, 2])

        for bk in self.lm_buckets():
            for _ in range(2):
                lm.sample(bk)
                lm.slow(bk)
        for rows, length in self.prefill_shapes():
            run_prefill(self._prefill_shape(rows, length))
        for chunk in self.chunks:
            buf = self._decode_shape(chunk)
            for _ in range(2):
                codec.run(buf['codes'], buf['meta'], buf['out'])
        if cuda:
            torch.cuda.synchronize(self.device)
        self.warmup_s = time.perf_counter() - t0
        if self.cuda_graphs:
            t0 = time.perf_counter()
            lm_pool = torch.cuda.graph_pool_handle()
            decode_pool = torch.cuda.graph_pool_handle()
            # One capture stream per graph family: cuBLAS keys its workspace by (handle, stream), so graphs captured on
            # one stream would share a workspace, and the LM graphs replay concurrently with the codec graphs.
            lm_capture = _engine_stream(self.device, 'audio8_lm_capture')
            decode_capture = _engine_stream(self.device, 'audio8_decode_capture')
            graphs: dict[tuple, torch.cuda.CUDAGraph] = {}

            def capture(key: tuple, fn: Callable[[], None], pool: Any, stream: Any) -> None:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool, stream=stream, capture_error_mode='thread_local'):
                    fn()
                graphs[key] = graph

            for bk in sorted(self.lm_buckets(), reverse=True):
                capture(('sample', bk), lambda bk=bk: lm.sample(bk), lm_pool, lm_capture)
                capture(('slow', bk), lambda bk=bk: lm.slow(bk), lm_pool, lm_capture)
            for rows, length in sorted(self.prefill_shapes(), reverse=True):
                buf = self._prefill_shape(rows, length)
                capture(('prefill', rows, length), lambda b=buf: run_prefill(b), lm_pool, lm_capture)
            for chunk in sorted(self.chunks, reverse=True):
                buf = self._decode_shape(chunk)
                capture(('decode', chunk), lambda b=buf: codec.run(b['codes'], b['meta'], b['out']), decode_pool,
                        decode_capture)
            torch.cuda.synchronize(self.device)
            self._graphs = graphs
            self._graph_pools = (lm_pool, decode_pool)
            self.capture_s = time.perf_counter() - t0
        # Warm-up wrote into slot state: start clean.
        codec.reset()
        lm.reset()
        if cuda:
            torch.cuda.synchronize(self.device)

    def _run(self, key: tuple, fn: Callable[[], None]) -> None:
        graph = self._graphs.get(key)
        if graph is not None:
            graph.replay()
        else:
            fn()

    def close(self) -> None:
        thread = self._thread
        if thread is not None:
            with self._cv:
                self._stop = True
                self._cv.notify_all()
            thread.join(timeout=30)
        self._thread = None
        err = RuntimeError('Audio8 engine closed')
        with self._cv:
            for row in list(self._pending) + list(self._rows.values()):
                row.stream._finish(err)
            for item in self._due:
                item.row.stream._finish(err)
            for tile in self._inflight:
                for item in tile.items:
                    item.row.stream._finish(err)
            self._pending.clear()
            self._rows.clear()
            self._due.clear()
            self._inflight.clear()
            while self._tasks:
                _, box = self._tasks.popleft()
                box['error'] = err
                box['event'].set()
        self._free_all()

    def _free_all(self) -> None:
        self._graphs = {}
        self._graph_pools = None
        self._prefill_shapes = {}
        self._decode_shapes = {}
        self._all_tiles = []
        self._free_tiles = []
        for name in ('pos_host', 'pos_np', 'step_host', 'step_np', 'eos_host', 'eos_np', 'prefill_host', 'prefill_np',
                     'decode_stream', 'eos_event'):
            if hasattr(self, name):
                delattr(self, name)
        self.lm = None
        self.slot_codec = None
        self.model = None
        self.codec = None
        self.tokenizer = None
        self._plain_voice = None
        with self._voice_lock:
            self._voices.clear()
        self._loaded = False
        import gc

        gc.collect()
        if self.device.type == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()

    @property
    def loaded(self) -> bool:
        return self._loaded

    # -- public API --------------------------------------------------------------------------------------------------

    def voice_prompt(self, key: str, audio: np.ndarray, sample_rate: int, ref_text: str) -> Audio8VoicePrompt:
        """Encode (once per key) a reference voice: reference codes, and the KV of the prompt head they sit in."""
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
        if wave.size == 0:
            raise ValueError('reference audio is empty')
        waveform = torch.from_numpy(np.ascontiguousarray(wave))
        if int(sample_rate) != NATIVE_SAMPLE_RATE:
            from torchaudio.functional import resample

            waveform = resample(waveform, int(sample_rate), NATIVE_SAMPLE_RATE)
        prompt = self._run_on_gpu_thread(lambda: self._encode_prompt(key, ref_text, waveform,
                                                                     wave.size / float(sample_rate)))
        with self._voice_lock:
            prompt = self._voices.setdefault(key, prompt)
            self._voices.move_to_end(key)
            self._trim_voices()
            return prompt

    def _trim_voices(self) -> None:
        """Least recently used out above the budget, never the newest. A row holds its own prompt, so a voice that is
        rendering can go; its memory is freed with the row."""
        total = sum(voice.size_bytes for voice in self._voices.values())
        while len(self._voices) > 1 and total > MAX_VOICE_CACHE_MB * 2 ** 20:
            total -= self._voices.popitem(last=False)[1].size_bytes

    def has_voice(self, key: str) -> bool:
        with self._voice_lock:
            return key in self._voices

    def drop_voice(self, key: str) -> None:
        with self._voice_lock:
            self._voices.pop(key, None)

    def submit(self, text: str, *, prompt: Audio8VoicePrompt | None, seed: int, row_id: str | None = None) -> Audio8Stream:
        """Queue one row. ``prompt`` None is plain synthesis without a reference voice."""
        if not self._loaded:
            raise RuntimeError('Audio8 engine is not loaded')
        row_id = row_id or f'audio8-{next(self._ids)}'
        voice = prompt if prompt is not None else self._plain_voice
        fragments = self._build_fragments(text, voice)
        stream = Audio8Stream(row_id, self)
        row = _Row(row_id=row_id, fragments=fragments, prompt=voice, seed=int(seed), stream=stream,
                   trimmer=SilenceTrimmer(NATIVE_SAMPLE_RATE),
                   resampler=StreamResampler(NATIVE_SAMPLE_RATE, self.output_sample_rate))
        stream._row = row
        stream.metrics['fragments'] = len(fragments)
        with self._cv:
            if self._stop or self._thread is None:
                raise RuntimeError('Audio8 engine is not running')
            self._pending.append(row)
            self._cv.notify_all()
        return stream

    def render(self, text: str, *, prompt: Audio8VoicePrompt | None = None, seed: int = 0,
               row_id: str | None = None) -> np.ndarray:
        chunks = list(self.submit(text, prompt=prompt, seed=seed, row_id=row_id))
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    def stats(self) -> dict[str, Any]:
        with self._cv:
            active = len(self._rows)
            pending = len(self._pending)
        out: dict[str, Any] = {
            'active_rows': active, 'pending_rows': pending, 'max_slots': self.max_slots,
            'free_slots': len(self._free_slots), 'step_ms': round(self._step_ms_ema, 3),
            'decode_ms': round(self._decode_ms_ema, 3), 'prefill_ms': round(self._prefill_ms_ema, 3),
            'steps': self._steps_total, 'bucket': self._last_bucket, 'voices': len(self._voices),
            'loaded': self._loaded, 'cuda_graphs': bool(self._graphs), 'graphs': len(self._graphs),
            'batch_invariant': self.batch_invariant, 'capture_s': round(self.capture_s, 3),
            'warmup_s': round(self.warmup_s, 3), 'first_chunk_frames': self.first_chunk_frames,
            'chunk_frames': self.chunk_frames, 'decode_tile_rows': self.decode_tile_rows,
            'prefill_tile_rows': self.prefill_tile_rows, 'context_length': self.context_length,
        }
        if self._loaded and self.device.type == 'cuda':
            out['vram_allocated_mb'] = round(torch.cuda.memory_allocated(self.device) / 2 ** 20, 1)
            out['vram_reserved_mb'] = round(torch.cuda.memory_reserved(self.device) / 2 ** 20, 1)
            out['vram_peak_mb'] = round(torch.cuda.max_memory_allocated(self.device) / 2 ** 20, 1)
        return out

    # -- text / prompt preparation (caller thread) ---------------------------------------------------------------------

    def _build_fragments(self, text: str, voice: Audio8VoicePrompt) -> list[_Fragment]:
        room = self.context_length - voice.length - PROMPT_RESERVE
        max_chars = min(self.fragment_max_chars, room // EXPECTED_FRAMES_PER_CHAR)
        if max_chars < MIN_FRAGMENT_CHARS:
            raise ValueError(f'Audio8 voice prompt too long: {voice.length} of {self.context_length} positions; use a '
                             'shorter reference clip')
        out = []
        for piece in split_fragments(text, max_chars):
            ids = suffix_ids(self.tokenizer, piece)
            budget = self.context_length - voice.length - len(ids) - 1
            if budget <= 8:
                raise ValueError(f'Audio8 prompt too long: {voice.length + len(ids)} of {self.context_length} positions '
                                 f'used by voice prompt and text fragment {piece!r}')
            out.append(_Fragment(piece, ids, min(budget, frame_cap(piece))))
        return out

    def _prompt_head(self, ids: list[int], codes: torch.Tensor | None) -> torch.Tensor:
        """The first P columns of the prompt: (1, books + 1, P), like ArkttsModel._prepare_prompt."""
        config = self.model.config
        length = 0 if codes is None else codes.shape[1]
        prompt = torch.zeros(1, self.books + 1, len(ids) + length, dtype=torch.long, device=self.device)
        prompt[0, 0, :len(ids)] = torch.tensor(ids, device=self.device)
        if codes is not None:
            prompt[0, 0, len(ids):] = codes[0].to(self.device) + config.semantic_begin_id
            prompt[0, 1:, len(ids):] = codes.to(self.device)
        return prompt

    def _encode_prompt(self, key: str, ref_text: str, waveform: torch.Tensor, duration_s: float) -> Audio8VoicePrompt:
        codec = self.codec
        dtype = next(codec.parameters()).dtype
        audio = waveform[None, None].to(device=self.device, dtype=dtype)
        codes, lengths = codec.encode(audio, torch.tensor([waveform.shape[-1]], device=self.device))
        codes = codes[0, :, :int(lengths[0])].long()
        if codes.numel() == 0 or int(codes.min()) < 0:
            raise ValueError('reference audio produced no codes')
        head = self._prompt_head(prefix_ids(self.tokenizer, ref_text), codes)
        if head.shape[-1] >= self.context_length - PROMPT_RESERVE:
            raise ValueError(f'Audio8 voice prompt too long: {head.shape[-1]} of {self.context_length} positions; use a '
                             'shorter reference clip')
        keys, values, state = self.lm.voice_kv(head)
        return Audio8VoicePrompt(key=key, ref_text=ref_text, codes=codes.cpu(), keys=keys, values=values,
                                 duration_s=duration_s, state=state)

    def _encode_plain(self) -> None:
        keys, values, state = self.lm.voice_kv(self._prompt_head(prefix_ids(self.tokenizer, None), None))
        self._plain_voice = Audio8VoicePrompt(key='', ref_text=None, codes=None, keys=keys, values=values,
                                              duration_s=0.0, state=state)

    def _run_on_gpu_thread(self, fn: Callable[[], Any]) -> Any:
        box: dict[str, Any] = {'event': threading.Event(), 'result': None, 'error': None}
        with self._cv:
            if self._thread is None or self._stop:
                raise RuntimeError('Audio8 engine is not running')
            self._tasks.append((fn, box))
            self._cv.notify_all()
        box['event'].wait()
        if box['error'] is not None:
            raise box['error']
        return box['result']

    def _cancel(self, stream: Audio8Stream) -> None:
        row = stream._row
        if row is None:
            return
        row.cancelled = True
        with self._cv:
            self._cv.notify_all()

    # -- GPU thread ----------------------------------------------------------------------------------------------------

    def _loop(self, ready: dict[str, Any]) -> None:
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
        with torch.inference_mode():
            try:
                self._allocate()
                self._prepare()
                self._encode_plain()
            except BaseException as exc:  # noqa: BLE001 - handed to load()
                logger.exception('Audio8 engine preparation failed')
                ready['error'] = exc
                ready['event'].set()
                return
            ready['event'].set()
            while True:
                with self._cv:
                    while (not self._stop and not self._tasks and not self._pending and not self._rows
                           and not self._inflight and not self._due):
                        self._cv.wait()
                    if self._stop:
                        return
                    tasks = list(self._tasks)
                    self._tasks.clear()
                    self._admit_locked()
                for fn, box in tasks:
                    try:
                        box['result'] = fn()
                    except BaseException as exc:  # noqa: BLE001 - handed to the caller
                        box['error'] = exc
                    box['event'].set()
                self._drop_cancelled()
                rows = list(self._rows.values())
                if not rows:
                    self._idle()
                    continue
                try:
                    self._iteration(rows)
                except Exception as exc:  # noqa: BLE001 - fail the rows, keep serving
                    logger.exception('Audio8 step failed')
                    self._fail_all(rows, exc)

    def _idle(self) -> None:
        try:
            self._launch_due(everything=True)
            self._deliver(block=True)
        except Exception as exc:  # noqa: BLE001
            logger.exception('Audio8 audio delivery failed')
            self._fail_all(list(self._rows.values()), exc)

    def _fail_all(self, rows: list[_Row], exc: BaseException) -> None:
        self._ahead_bk = None
        items = list(self._due)
        self._due.clear()
        for tile in self._inflight:
            items += tile.items
            self._recycle(tile)
        self._inflight.clear()
        for item in items:
            item.row.stream._finish(exc)
        for row in rows:
            self._release(row, exc)
        if self.device.type == 'cuda':
            try:
                torch.cuda.synchronize(self.device)
            except Exception:  # noqa: BLE001
                logger.exception('CUDA synchronize after a failed Audio8 step')

    def _admit_locked(self) -> None:
        while self._pending and self._free_slots:
            row = self._pending.popleft()
            if row.cancelled:
                row.stream._finish()
                continue
            row.slot = heapq.heappop(self._free_slots)
            self._rows[row.slot] = row
            row.stream.started_at = time.perf_counter()

    def _drop_cancelled(self) -> None:
        cancelled = [r for r in self._rows.values() if r.cancelled]
        for row in cancelled:
            self._release(row, None)
        if cancelled:
            # Their chunks would read frames of a slot the next row is about to overwrite.
            self._due = collections.deque(item for item in self._due if not item.row.cancelled)
        with self._cv:
            if any(r.cancelled for r in self._pending):
                keep: collections.deque[_Row] = collections.deque()
                for r in self._pending:
                    if r.cancelled:
                        r.stream._finish()
                    else:
                        keep.append(r)
                self._pending = keep

    def _free_slot(self, row: _Row) -> None:
        with self._cv:
            if self._rows.get(row.slot) is row:
                del self._rows[row.slot]
                heapq.heappush(self._free_slots, row.slot)

    def _release(self, row: _Row, error: BaseException | None) -> None:
        self._free_slot(row)
        row.stream.metrics['steps'] = row.total_steps
        row.stream._finish(error)

    def _bucket(self, rows: list[_Row]) -> int:
        if self.batch_invariant:
            return self.max_slots
        return bucket_for(max(r.slot for r in rows) + 1, self.max_slots)

    def _launch_slow(self, rows: list[_Row]) -> None:
        """Enqueue the slow step of ``rows`` (their last frame at r.pos) and advance their positions."""
        if not rows:
            self._ahead_bk = None
            return
        bk = self._bucket(rows)
        flip = self._pos_flip
        self._pos_flip ^= 1
        pos = self.pos_np[flip]
        pos[:bk] = 0
        for r in rows:
            pos[r.slot] = r.pos
            r.pos += 1
        lm = self.lm
        lm.pos[:bk].copy_(self.pos_host[flip][:bk], non_blocking=True)
        self._run(('slow', bk), lambda: lm.slow(bk))
        self._ahead_bk = bk

    def _iteration(self, rows: list[_Row]) -> None:
        t0 = time.perf_counter()
        lm = self.lm
        running = [r for r in rows if r.state == 'run']
        if running and self._ahead_bk is None:
            self._launch_slow(running)  # normally already launched at the end of the previous step
        self._ahead_bk = None
        joining = [r for r in rows if r.state == 'prefill']
        if joining:
            self._prefill(joining)
        bk = self._bucket(rows)
        self._last_bucket = bk
        host = self.step_np
        host[:, :bk] = 0
        for r in rows:
            host[0, r.slot] = r.step
            host[1, r.slot] = r.key
        lm.step[:bk].copy_(self.step_host[0, :bk], non_blocking=True)
        lm.key[:bk].copy_(self.step_host[1, :bk], non_blocking=True)
        self._run(('sample', bk), lambda: lm.sample(bk))
        self.eos_host[:bk].copy_(lm.eos[:bk], non_blocking=True)
        if self.eos_event is not None:
            self.eos_event.record()
        # The next step of every row is enqueued before the EOS flags are read. For a row that turns out to be done it
        # computes a throw-away column: it writes KV at a position that is rewritten by the next prefill, or overwritten
        # before anything attends to it.
        self._launch_slow(rows)
        # While the GPU works: hand out the audio of decode calls that are done.
        self._deliver(block=False)
        if self.eos_event is not None:
            self.eos_event.synchronize()  # the one sync per step
        self._steps_total += 1
        self._book(rows)
        self._launch_due(everything=False)
        dt = (time.perf_counter() - t0) * 1000.0
        self._step_ms_ema = dt if self._step_ms_ema == 0.0 else 0.9 * self._step_ms_ema + 0.1 * dt

    def _book(self, rows: list[_Row]) -> None:
        """What the step's EOS flags mean for each row: frames, chunks due, fragments and rows that ended."""
        eos = self.eos_np
        now = self._steps_total
        for r in rows:
            ended = bool(eos[r.slot])
            r.step += 1
            r.total_steps += 1
            r.state = 'run'
            if not ended:
                r.frames += 1
            done = ended or r.frames >= r.fragment.cap
            if done and not ended:
                r.stream.metrics['capped'] = r.stream.metrics.get('capped', 0) + 1
                logger.warning('audio8: row %s fragment %s hit its cap of %s frames', r.row_id, r.frag_index, r.frames)
            target = self.chunk_frames if r.first_sent else self.first_chunk_frames
            n = r.frames - r.decoded
            if not done and n < target:
                continue
            last = done and r.frag_index + 1 >= len(r.fragments)
            first = not r.first_sent and n > 0
            r.first_sent = r.first_sent or n > 0
            self._due.append(_Item(row=r, slot=r.slot, start=r.decoded, n=n, final=done, last=last,
                                   urgent=first or done, born=now))
            r.decoded = r.frames
            if last:
                # The slot is free for a new row now (its frames are gathered this step, before anything can write
                # there); the stream finishes when this chunk's audio is delivered.
                r.stream.metrics['steps'] = r.total_steps
                self._free_slot(r)
            elif done:
                r.frag_index += 1
                r.state = 'prefill'

    # -- prefill -------------------------------------------------------------------------------------------------------

    def _prefill(self, rows: list[_Row]) -> None:
        t0 = time.perf_counter()
        lm = self.lm
        for r in rows:
            voice = r.prompt
            P = voice.length
            lm.load_voice(r.slot, voice.keys, voice.values, voice.state)
            r.fed = 0
            r.key = row_key(r.seed, r.frag_index)
            r.step = 0
            r.frames = 0
            r.decoded = 0
            r.pos = P + len(r.fragment.ids)
        waiting = list(rows)
        offset = 0
        tile = self.prefill_tile_rows
        while waiting:
            groups: dict[int, list[_Row]] = {}
            for r in waiting:
                groups.setdefault(prefill_length(len(r.fragment.ids) - r.fed), []).append(r)
            for length, group in sorted(groups.items()):
                for i in range(0, len(group), tile):
                    offset = self._prefill_tile(group[i:i + tile], length, offset)
            waiting = [r for r in waiting if r.fed < len(r.fragment.ids)]
        dt = (time.perf_counter() - t0) * 1000.0
        self._prefill_ms_ema = dt if self._prefill_ms_ema == 0.0 else 0.9 * self._prefill_ms_ema + 0.1 * dt

    def _prefill_tile(self, rows: list[_Row], length: int, offset: int) -> int:
        lm = self.lm
        R = self.prefill_tile_rows
        buf = self._prefill_shape(R, length)
        need = R * (length + 3)
        if offset + need > self.prefill_np.size:
            # Everything staged so far has to reach the GPU before its memory is written again.
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            offset = 0
        ids = self.prefill_np[offset:offset + R * length].reshape(R, length)
        meta = self.prefill_np[offset + R * length:offset + need].reshape(R, 3)
        ids[:] = 0
        meta[:] = (self.max_slots, 0, 0)
        for j, r in enumerate(rows):
            tokens = r.fragment.ids[r.fed:r.fed + length]
            ids[j, :len(tokens)] = tokens
            meta[j] = (r.slot, r.prompt.length + r.fed, len(tokens) - 1)
            r.fed += len(tokens)
        buf['ids'].copy_(self.prefill_host[offset:offset + R * length].view(R, length), non_blocking=True)
        buf['meta'].copy_(self.prefill_host[offset + R * length:offset + need].view(R, 3), non_blocking=True)
        self._run(('prefill', R, length),
                  lambda: lm.prefill(buf['ids'], buf['meta'][:, 0], buf['meta'][:, 1], buf['meta'][:, 2]))
        return offset + need

    # -- audio ---------------------------------------------------------------------------------------------------------

    def _launch_due(self, *, everything: bool) -> None:
        """Send due chunks to the codec. A call costs its whole tile, so chunks nobody waits for wait for company."""
        now = self._steps_total
        hold = self.decode_hold_steps
        while self._due:
            ready = (everything or len(self._due) >= self.decode_tile_rows
                     or any(item.urgent for item in self._due) or now - self._due[0].born >= hold)
            if not ready:
                return
            self._launch_tile()

    def _launch_tile(self) -> None:
        """One tile: rows in order of arrival, one chunk size, one chunk per row (a row's next chunk needs the state
        this one leaves). Zero-frame items (a fragment that ended exactly on a chunk edge) only go along for delivery."""
        tile = self._take_tile()  # may deliver, and fail: nothing has left _due yet
        rows = self.decode_tile_rows
        chunk: int | None = None
        taken: list[_Item] = []
        rest: collections.deque[_Item] = collections.deque()
        seen: set[int] = set()
        for item in self._due:
            size = self.first_chunk_frames if item.n <= self.first_chunk_frames else self.chunk_frames
            if (len(taken) < rows and id(item.row) not in seen and (chunk is None or size == chunk)):
                chunk = size
                taken.append(item)
                seen.add(id(item.row))
            else:
                seen.add(id(item.row))
                rest.append(item)
        self._due = rest
        tile.chunk = chunk
        tile.items = taken
        self._inflight.append(tile)  # first: when the launch fails, _fail_all ends these streams
        self._gather(tile)

    def _take_tile(self) -> _Tile:
        while not self._free_tiles:
            self._deliver(block=True, one=True)
        return self._free_tiles.pop()

    def _recycle(self, tile: _Tile) -> None:
        tile.items = []
        tile.event = None
        self._free_tiles.append(tile)

    def _gather(self, tile: _Tile) -> None:
        """Frames of the tile on the main stream (it is ordered with the steps writing them), decode on the codec's."""
        lm, codec = self.lm, self.slot_codec
        rows, cmax = self.decode_tile_rows, self.chunk_frames
        meta = tile.meta_host.numpy()
        meta[:] = 0
        meta[:, 0] = codec.trash
        index = tile.index_host.numpy()
        for j, item in enumerate(tile.items):
            index[j] = np.minimum(np.arange(item.start, item.start + cmax), self.context_length - 1)
            tile.slot_host[j, 0] = item.slot
            meta[j] = (item.slot, item.start, item.n, 1 if item.start == 0 else 0)
        for j, item in enumerate(tile.items):
            if item.n == 0:  # delivery only: the codec must not touch the row's state
                meta[j] = (codec.trash, 0, 0, 0)
        count = len(tile.items)
        tile.index_dev[:count].copy_(tile.index_host[:count], non_blocking=True)
        tile.slot_dev[:count].copy_(tile.slot_host[:count], non_blocking=True)
        tile.stage[:count] = lm.frames[tile.slot_dev[:count], tile.index_dev[:count]]
        main = torch.cuda.current_stream(self.device) if self.decode_stream is not None else None
        tile.launched_at = time.perf_counter()
        buf = self._decode_shape(tile.chunk)
        with torch.cuda.stream(self.decode_stream) if self.decode_stream is not None else contextlib.nullcontext():
            if main is not None:
                self.decode_stream.wait_stream(main)
            buf['meta'].copy_(tile.meta_host, non_blocking=True)
            buf['codes'].copy_(tile.stage[:, :tile.chunk])
            self._run(('decode', tile.chunk), lambda: codec.run(buf['codes'], buf['meta'], buf['out']))
            tile.out_host[:, :buf['out'].shape[1]].copy_(buf['out'], non_blocking=True)
            if self.decode_stream is not None:
                tile.event = torch.cuda.Event()
                tile.event.record(self.decode_stream)

    def _deliver(self, *, block: bool, one: bool = False) -> None:
        while self._inflight:
            tile = self._inflight[0]
            if not block and tile.event is not None and not tile.event.query():
                return
            self._inflight.popleft()
            try:
                self._deliver_tile(tile)
            except BaseException as exc:
                # The tile is no longer reachable from _fail_all, and rows whose last fragment ended earlier have
                # already left _rows: finish every stream of this tile here so no consumer blocks forever.
                for item in tile.items:
                    item.row.stream._finish(exc)
                raise
            finally:
                self._recycle(tile)
            if one:
                return

    def _deliver_tile(self, tile: _Tile) -> None:
        if tile.event is not None:
            tile.event.synchronize()
        dt = (time.perf_counter() - tile.launched_at) * 1000.0
        self._decode_ms_ema = dt if self._decode_ms_ema == 0.0 else 0.9 * self._decode_ms_ema + 0.1 * dt
        samples = self.slot_codec.out_len(1)
        for j, item in enumerate(tile.items):
            r = item.row
            if r.stream.done:
                continue
            audio = tile.out_host[j, :item.n * samples].numpy().copy()
            out = r.trimmer.finish(audio) if item.final else r.trimmer.push(audio)
            if out.size:
                r.stream._put(r.resampler.push(out))
            if item.last:
                r.stream._put(r.resampler.flush())
                r.stream.metrics['steps'] = r.total_steps
                r.stream._finish()
