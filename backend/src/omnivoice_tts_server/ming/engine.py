"""Ming-3.0-tts 0.5B engine: continuous batching, per-row determinism, true chunk streaming.

One dedicated GPU thread owns every CUDA operation. Each loop iteration ("step") does, for all active rows at once:

1. batched LLM decode of rows that already have an input embedding (slot KV cache, per-row positions, padded mask),
2. prefill of rows that join (or start their next text fragment) at this step boundary (one row per call, padded to
   a 64-token bucket),
3. batched CFG flow sampling (10 Euler steps, both branches) with per-row starting noise drawn from the row's own
   seeded generator, the stop head, and the aggregator projection for the next input,
4. ONE device->host copy (the stop flags) -- while the GPU runs this step, the CPU post-processes the audio of the
   previous step's VAE job (trailing-silence trim, resampling, hand-off to the streams),
5. one batched streaming VAE decode job for every row whose chunk schedule is due, on a second CUDA stream, so it
   overlaps the next LM step; its audio is copied to pinned host memory and delivered during the next step.

With ``cuda_graphs`` every GPU part (decode, flow, prefill per length bucket, VAE per shape) is replayed from CUDA
graphs captured at load time; without, the very same functions run eagerly (bit-identical results).

The math per row is the upstream ``BailingMMNativeForConditionalGeneration.sample`` loop (cfg 2.0, sigma 0.25,
temperature 0, flow steps 10, stop when p(stop) > 0.5 and step > 3, 512-position budget including the prompt) and
``ming_tts.MingTTS.synthesize`` (fragmenting, per-fragment trailing-silence trim, same prompt for every fragment).
The fast paths (``fast.py``, ``vae_stream.py``) restate the reference modules with fused/hoisted kernels; they agree
with them within bf16 rounding.

``batch_invariant=True`` (default) runs every GPU call at a fixed shape (all slots, fixed VAE chunk), so a row's audio
is bit-identical whatever else runs. ``batch_invariant=False`` uses the smallest bucket per call: faster at low load,
but rows then differ at bf16-noise level depending on their neighbours.
"""
from __future__ import annotations

import collections
import heapq
import itertools
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import torch
import torch.nn.functional as F

from .fast import FastFlow, FastQwen, linear_plans, load_plans, save_plans
from .flow import flow_timesteps
from .model import MAX_CACHE_LEN, MingModel, load_ming_model
from .prompt import build_input_ids, build_tts_prompt, normalize_language_id, validate_tokenizer
from .streaming import ChunkSchedule, SilenceTrimmer, StreamResampler
from .text import prepare_fragments
from .vae_stream import SlotDecoder

logger = logging.getLogger(__name__)

# Quality knobs (fixed; see upstream ming_tts.synthesize defaults).
CFG_SCALE = 2.0
SIGMA = 0.25
TEMPERATURE = 0.0
FLOW_STEPS = 10
STOP_MIN_STEP = 3  # upstream: stop only when step > 3
MAX_DECODE_STEPS = 1000
BUCKETS = (1, 2, 4, 8, 16, 32, 64)
PREFILL_BUCKET = 64  # prefill lengths are padded up to a multiple of this (one graph per bucket)
_END = object()


def bucket_for(n: int, max_slots: int) -> int:
    """Smallest bucket holding ``n`` slots (capped at ``max_slots``)."""
    for b in BUCKETS:
        if b >= n:
            return min(b, max_slots)
    return max_slots


def chunk_buckets(max_chunk: int) -> tuple[int, ...]:
    """VAE chunk sizes a decode call is padded to: powers of two below the maximum, the maximum, maximum + 1."""
    out = {max_chunk, max_chunk + 1}
    b = 1
    while b < max_chunk:
        out.add(b)
        b *= 2
    return tuple(sorted(out))


@dataclass
class MingVoicePrompt:
    key: str
    ref_text: str
    seed: int
    latent: torch.Tensor            # (1, T, 64) on device, model dtype (VAE posterior sample)
    spk_emb: torch.Tensor           # (1, 192) float32 CPU (CAM++)
    projected_latent: torch.Tensor  # (T, hidden) on device (aggregator projection, spliced into the prompt)
    projected_spk: torch.Tensor     # (hidden,) on device (spk_head projection)
    duration_s: float

    @property
    def num_latents(self) -> int:
        return int(self.latent.shape[1])


@dataclass
class _Fragment:
    text: str
    input_ids: list[int]
    spk_positions: list[int]
    latent_position: int
    budget: int
    ids_host: torch.Tensor | None = None    # pinned (L,) long
    noise_host: torch.Tensor | None = None  # pinned (budget, latent) model dtype, from the row's generator


@dataclass
class _Row:
    row_id: str
    fragments: list[_Fragment]
    prompt: MingVoicePrompt | None
    seed: int
    stream: 'MingStream'
    generator: torch.Generator
    trimmer: SilenceTrimmer
    resampler: StreamResampler
    schedule: ChunkSchedule
    slot: int = -1
    frag_index: int = 0
    state: str = 'prefill'  # prefill | decode
    step: int = 0           # LM steps done in the current fragment (= latents generated)
    pos: int = 0            # KV positions used in the slot
    decoded: int = 0        # latents of the current fragment already sent to the VAE decoder
    cancelled: bool = False
    total_steps: int = 0

    @property
    def fragment(self) -> _Fragment:
        return self.fragments[self.frag_index]


@dataclass
class _VaeItem:
    row: _Row
    slot: int
    start: int         # first latent of this chunk within the fragment
    n: int             # latents in this chunk
    generated: int     # latents generated in the fragment so far
    final: bool        # last chunk of the fragment
    last: bool         # last chunk of the row
    chunk: int = 0     # padded chunk size (graph shape)
    index: int = 0     # row in the host output buffer


@dataclass
class _VaeJob:
    items: list[_VaeItem] = field(default_factory=list)
    event: Any = None
    launched_at: float = 0.0


class MingStream:
    """Blocking iterator over np.float32 audio chunks at the engine's output sample rate."""

    def __init__(self, row_id: str, engine: 'MingEngine') -> None:
        self.row_id = row_id
        self._engine = engine
        self._queue: queue.Queue = queue.Queue()
        self._row: _Row | None = None
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


class MingEngine:
    def __init__(self, model_dir: Path | str, device: str = 'cuda:0', *, max_slots: int = 16, cuda_graphs: bool = True,
                 output_sample_rate: int = 24000, first_chunk_latents: int = 2, max_chunk_latents: int = 8,
                 dtype: torch.dtype = torch.bfloat16, batch_invariant: bool = True, vae_tile_rows: int | None = None,
                 plan_cache: Path | str | None = None) -> None:
        self.model_dir = Path(model_dir)
        self.device = torch.device(device)
        self.max_slots = max(1, min(64, int(max_slots)))
        self.cuda_graphs = bool(cuda_graphs) and self.device.type == 'cuda'
        self.output_sample_rate = int(output_sample_rate)
        self.first_chunk_latents = max(1, int(first_chunk_latents))
        self.max_chunk_latents = max(1, self.first_chunk_latents - 1, int(max_chunk_latents))
        self.dtype = dtype
        # batch_invariant: every GPU call has the same shape whatever else is running (all slots; prefill one row
        # per call; VAE at all slots x the largest chunk), so a row's numbers do not depend on its neighbours (bf16
        # kernels pick tiles by shape; with varying shapes the autoregressive loop amplifies last-bit differences
        # into a different, equally valid, rendition).
        self.batch_invariant = bool(batch_invariant)
        # VAE decode in batch-invariant mode runs in fixed tiles of this many rows x the largest chunk (several tiles
        # per step when more rows are due): same shape for every call, cost proportional to the rows that are due.
        # Default max(8, slots/4): measured best on the 5090 for 16/32/64 slots (a tile of 8 costs ~1.1x a tile of 4).
        tile = max(8, self.max_slots // 4) if vae_tile_rows is None else int(vae_tile_rows)
        self.vae_tile_rows = max(1, min(self.max_slots, tile))
        # Persisted GEMM kernel plans (see fast.py): same machine + same software = same kernels = same audio.
        self.plan_cache = Path(plan_cache) if plan_cache is not None else self.model_dir / '.g3_linear_plans.json'
        self.native_sample_rate = 44100
        self.model: MingModel | None = None
        self.tokenizer = None
        self.speaker = None
        self._voices: dict[str, MingVoicePrompt] = {}
        self._voice_lock = threading.Lock()
        self._cv = threading.Condition()
        self._pending: collections.deque[_Row] = collections.deque()
        self._tasks: collections.deque = collections.deque()
        self._rows: dict[int, _Row] = {}
        self._free_slots: list[int] = []
        self._thread: threading.Thread | None = None
        self._stop = False
        self._ids = itertools.count(1)
        self._step_ms_ema = 0.0
        self._vae_ms_ema = 0.0
        self._steps_total = 0
        self._last_bucket = 0
        self._loaded = False
        self._vae_job: _VaeJob | None = None
        self.graph_runner = None  # kept for API compatibility (True-ish when graphs are active)
        self._graphs: dict[tuple, torch.cuda.CUDAGraph] = {}
        self.capture_s = 0.0
        self.warmup_s = 0.0

    # -- lifecycle --------------------------------------------------------------------------------------------------

    def load(self) -> None:
        if self._loaded:
            return
        from transformers import Qwen2Tokenizer

        from .speaker import SpeakerEmbedder

        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
        model = load_ming_model(self.model_dir, self.device, self.dtype)
        # tokenizer_config.json names Qwen2Tokenizer; loading it directly skips AutoConfig on the 'dense' config.json.
        tokenizer = Qwen2Tokenizer.from_pretrained(str(self.model_dir))
        validate_tokenizer(tokenizer)
        self.native_sample_rate = model.sample_rate
        self.model = model
        self.tokenizer = tokenizer
        self.speaker = SpeakerEmbedder(self.model_dir / 'campplus.onnx')
        self._stop = False
        self._free_slots = list(range(self.max_slots))
        heapq.heapify(self._free_slots)
        ready = {'event': threading.Event(), 'error': None}
        self._thread = threading.Thread(target=self._loop, args=(ready,), name='ming-gpu', daemon=True)
        self._thread.start()
        ready['event'].wait()
        if ready['error'] is not None:
            self._thread.join(timeout=30)
            self._thread = None
            self._free_all()
            raise ready['error']
        self._loaded = True

    def _allocate(self) -> None:
        m = self.model
        cfg = m.llm.config
        S, dev, dt = self.max_slots, self.device, self.dtype
        pin = dev.type == 'cuda'
        self.n_layers = cfg.num_hidden_layers
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = getattr(cfg, 'head_dim', None) or cfg.hidden_size // cfg.num_attention_heads
        self.hidden = cfg.hidden_size
        self.latent_dim = m.latent_dim
        shape = (S, self.n_kv, MAX_CACHE_LEN, self.head_dim)
        self.k_cache = [torch.zeros(shape, device=dev, dtype=dt) for _ in range(self.n_layers)]
        self.v_cache = [torch.zeros(shape, device=dev, dtype=dt) for _ in range(self.n_layers)]
        self.pos_gpu = torch.zeros(S, device=dev, dtype=torch.long)
        self.step_gpu = torch.zeros(S, device=dev, dtype=torch.long)
        # Pinned host staging (numpy views for cheap element writes). Positions are double-buffered because the next
        # step's decode is launched before this step's stop flags are read.
        self.pos_host = [torch.zeros(S, dtype=torch.long, pin_memory=pin) for _ in range(2)]
        self.pos_np = [t.numpy() for t in self.pos_host]
        self.step_host = torch.zeros(S, dtype=torch.long, pin_memory=pin)
        self.step_np = self.step_host.numpy()
        self.stop_host = torch.zeros(S, dtype=torch.bool, pin_memory=pin)
        self.stop_np = self.stop_host.numpy()
        self._pos_flip = 0
        self._ahead_bk: int | None = None
        self.slot_index = torch.arange(S, device=dev)
        self.kv_arange = torch.arange(MAX_CACHE_LEN, device=dev)
        self.next_embed = torch.zeros(S, 1, self.hidden, device=dev, dtype=dt)
        self.z_buf = torch.zeros(S, 1, self.hidden, device=dev, dtype=dt)
        self.history = torch.zeros(S, 1, self.latent_dim, device=dev, dtype=dt)
        self.noise = torch.zeros(S, MAX_CACHE_LEN, self.latent_dim, device=dev, dtype=dt)
        self.lat_buf = torch.zeros(S, MAX_CACHE_LEN, self.latent_dim, device=dev, dtype=dt)
        self.stop_buf = torch.zeros(S, device=dev, dtype=torch.bool)
        # Upstream builds the schedule inside every sample() call in the noise dtype; it is a constant.
        self.timesteps = flow_timesteps(FLOW_STEPS, dev, dt)
        # Prefill (one row per call, padded length).
        self.prefill_h = torch.zeros(1, MAX_CACHE_LEN, self.hidden, device=dev, dtype=dt)
        self.prefill_ids = torch.zeros(MAX_CACHE_LEN, device=dev, dtype=torch.long)
        self.prefill_slot = torch.zeros(1, device=dev, dtype=torch.long)
        self.prefill_last = torch.zeros(1, device=dev, dtype=torch.long)
        # Fast paths (fused weights are views; the reference modules keep working).
        self.fq = FastQwen(m.llm, rope_positions=MAX_CACHE_LEN, dtype=dt)
        self.ff = FastFlow(m, self.timesteps, FLOW_STEPS, CFG_SCALE, m.patch_size)
        self.vae = SlotDecoder(m.audio.decoder, slots=S, max_chunk_latents=self.max_chunk_latents,
                               max_fragment_latents=MAX_CACHE_LEN, device=dev, dtype=dt)
        # VAE job buffers: one gather of every due row's latent window on the main stream, per-shape graph inputs.
        self.vae_chunks = chunk_buckets(self.max_chunk_latents)
        cmax = self.vae_chunks[-1]
        self.vae_stage = torch.zeros(S, cmax + 2, self.latent_dim, device=dev, dtype=dt)
        self.vae_idx_host = torch.zeros(S, cmax + 2, dtype=torch.long, pin_memory=pin)
        self.vae_slot_host = torch.zeros(S, dtype=torch.long, pin_memory=pin)
        self.vae_idx_dev = torch.zeros(S, cmax + 2, dtype=torch.long, device=dev)
        self.vae_slot_dev = torch.zeros(S, dtype=torch.long, device=dev)
        self.vae_out_host = torch.zeros(S, self.vae.out_len(cmax), dtype=torch.float32, pin_memory=pin)
        # Per-job meta staging: every tile/group gets its own region (the H2D copies are asynchronous).
        self.vae_meta_host = torch.zeros(2 * S + S * len(self.vae_chunks), 4, dtype=torch.long, pin_memory=pin)
        self.vae_meta_np = self.vae_meta_host.numpy()
        self._vae_shapes: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
        if dev.type == 'cuda':
            self.vae_stream = torch.cuda.Stream(device=dev, priority=-1)
            self.stop_event = torch.cuda.Event()
        else:  # pragma: no cover - CPU runs are for tests of the host logic only
            self.vae_stream = None
            self.stop_event = None

    def _vae_shape(self, rows: int, chunk: int) -> dict[str, torch.Tensor]:
        buf = self._vae_shapes.get((rows, chunk))
        if buf is None:
            dev = self.device
            buf = {'lat': torch.zeros(rows, chunk + 2, self.latent_dim, device=dev, dtype=self.dtype),
                   'meta': torch.zeros(rows, 4, device=dev, dtype=torch.long),
                   'out': torch.zeros(rows, self.vae.out_len(chunk), device=dev, dtype=torch.float32)}
            buf['meta'][:, 0] = self.vae.trash
            self._vae_shapes[(rows, chunk)] = buf
        return buf

    def lm_buckets(self) -> list[int]:
        if self.batch_invariant:
            return [self.max_slots]
        return sorted({bucket_for(n, self.max_slots) for n in range(1, self.max_slots + 1)})

    def vae_shapes(self) -> list[tuple[int, int]]:
        if self.batch_invariant:
            return [(self.vae_tile_rows, self.vae_chunks[-1])]
        return [(r, c) for r in self.lm_buckets() for c in self.vae_chunks]

    def prefill_buckets(self) -> list[int]:
        return list(range(PREFILL_BUCKET, MAX_CACHE_LEN + 1, PREFILL_BUCKET))

    def _prepare(self) -> None:
        """Warm up every shape (linear plans, cuFFT plans, timestep embeddings), then capture the CUDA graphs."""
        t0 = time.perf_counter()
        cuda = self.device.type == 'cuda'
        if cuda:
            load_plans(self.plan_cache, self.device)
        side = torch.cuda.Stream(device=self.device) if cuda else None
        if cuda:
            side.wait_stream(torch.cuda.current_stream())
        ctx = torch.cuda.stream(side) if cuda else _nullcontext()
        with ctx:
            for bk in self.lm_buckets():
                self.ff.prepare(bk)
                for _ in range(2):
                    self._decode_core(bk)
                    self._flow_core(bk)
            for lb in self.prefill_buckets():
                self._prefill_core(lb)
            for rows, chunk in self.vae_shapes():
                buf = self._vae_shape(rows, chunk)
                for _ in range(2):
                    self.vae.run(buf['lat'], buf['meta'], buf['out'])
        if cuda:
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize(self.device)
            save_plans(self.plan_cache, self.device)
        self.warmup_s = time.perf_counter() - t0
        if self.cuda_graphs:
            t0 = time.perf_counter()
            lm_pool = torch.cuda.graph_pool_handle()
            vae_pool = torch.cuda.graph_pool_handle()
            graphs = {}

            def capture(key, fn, pool):
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=pool, capture_error_mode='thread_local'):
                    fn()
                graphs[key] = g

            for bk in sorted(self.lm_buckets(), reverse=True):
                capture(('decode', bk), lambda bk=bk: self._decode_core(bk), lm_pool)
                capture(('flow', bk), lambda bk=bk: self._flow_core(bk), lm_pool)
            for lb in sorted(self.prefill_buckets(), reverse=True):
                capture(('prefill', lb), lambda lb=lb: self._prefill_core(lb), lm_pool)
            for rows, chunk in sorted(self.vae_shapes(), reverse=True):
                buf = self._vae_shape(rows, chunk)
                capture(('vae', rows, chunk), lambda b=buf: self.vae.run(b['lat'], b['meta'], b['out']), vae_pool)
            torch.cuda.synchronize(self.device)
            self._graphs = graphs
            self._graph_pools = (lm_pool, vae_pool)
            self.graph_runner = True
            self.capture_s = time.perf_counter() - t0
        # Warm-up wrote into slot state: start clean.
        self.vae.reset()
        for t in self.k_cache + self.v_cache:
            t.zero_()
        for t in (self.next_embed, self.z_buf, self.history, self.lat_buf, self.noise):
            t.zero_()
        if cuda:
            torch.cuda.synchronize(self.device)

    def _run(self, key: tuple, fn: Callable[[], None]) -> None:
        g = self._graphs.get(key)
        if g is not None:
            g.replay()
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
        err = RuntimeError('Ming engine closed')
        with self._cv:
            for row in list(self._pending) + list(self._rows.values()):
                row.stream._finish(err)
            if self._vae_job is not None:
                for item in self._vae_job.items:
                    item.row.stream._finish(err)
            self._pending.clear()
            self._rows.clear()
            while self._tasks:
                _, box = self._tasks.popleft()
                box['error'] = err
                box['event'].set()
        self._free_all()

    def _free_all(self) -> None:
        self._vae_job = None
        self.graph_runner = None
        self._graphs = {}
        self._graph_pools = None
        self._vae_shapes = {}
        for name in ('k_cache', 'v_cache', 'pos_gpu', 'step_gpu', 'pos_host', 'step_host', 'stop_host', 'pos_np',
                     'step_np', 'stop_np', 'slot_index',
                     'kv_arange', 'next_embed', 'z_buf', 'history', 'noise', 'lat_buf', 'stop_buf', 'timesteps',
                     'prefill_h', 'prefill_ids', 'prefill_slot', 'prefill_last', 'fq', 'ff', 'vae', 'vae_stage',
                     'vae_idx_host', 'vae_slot_host', 'vae_idx_dev', 'vae_meta_host', 'vae_meta_np', 'vae_slot_dev', 'vae_out_host',
                     'vae_stream', 'stop_event'):
            if hasattr(self, name):
                delattr(self, name)
        self.model = None
        self.speaker = None
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

    # -- public API ---------------------------------------------------------------------------------------------------

    def voice_prompt(self, key: str, audio: np.ndarray, sample_rate: int, ref_text: str, seed: int = 0) -> MingVoicePrompt:
        """Encode (once per key) a reference voice: VAE posterior sample with a fixed seed, CAM++ embedding."""
        with self._voice_lock:
            cached = self._voices.get(key)
        if cached is not None:
            return cached
        if not self._loaded:
            raise RuntimeError('Ming engine is not loaded')
        ref_text = str(ref_text or '').strip()
        if not ref_text:
            raise ValueError('Ming voice cloning needs the reference transcript (ref_text)')
        import torchaudio

        wave = np.asarray(audio, dtype=np.float32)
        if wave.ndim == 2:  # (samples, channels) or (channels, samples)
            wave = wave.mean(axis=1 if wave.shape[0] > wave.shape[1] else 0)
        wave = wave.reshape(-1)
        if wave.size == 0:
            raise ValueError('reference audio is empty')
        waveform = torch.from_numpy(np.ascontiguousarray(wave)).unsqueeze(0)
        sr = int(sample_rate)
        spk_wave = waveform if sr == 16000 else torchaudio.transforms.Resample(sr, 16000)(waveform)
        spk_emb = self.speaker(spk_wave)
        vae_sr = self.native_sample_rate
        vae_wave = waveform if sr == vae_sr else torchaudio.transforms.Resample(sr, vae_sr)(waveform)
        align = self.model.prompt_align
        remainder = vae_wave.shape[-1] % align
        if remainder:
            padded = torch.zeros((1, vae_wave.shape[-1] + align - remainder), dtype=vae_wave.dtype)
            padded[:, -vae_wave.shape[-1]:] = vae_wave
            vae_wave = padded
        prompt = self._run_on_gpu_thread(lambda: self._encode_prompt(key, ref_text, seed, vae_wave, spk_emb,
                                                                     wave.size / float(sr)))
        with self._voice_lock:
            return self._voices.setdefault(key, prompt)

    def has_voice(self, key: str) -> bool:
        with self._voice_lock:
            return key in self._voices

    def drop_voice(self, key: str) -> None:
        with self._voice_lock:
            self._voices.pop(key, None)

    def submit(self, text: str, *, prompt: MingVoicePrompt | None, language: str | None, seed: int,
               row_id: str | None = None) -> MingStream:
        if not self._loaded:
            raise RuntimeError('Ming engine is not loaded')
        row_id = row_id or f'ming-{next(self._ids)}'
        fragments = self._build_fragments(text, prompt, language)
        stream = MingStream(row_id, self)
        gen = torch.Generator(device='cpu')
        gen.manual_seed(int(seed) % (2 ** 63))
        pin = self.device.type == 'cuda'
        for frag in fragments:
            # Starting noise for every LM step of each fragment, drawn from the row's own generator in fragment order
            # (exactly the draws the step loop used to make at each fragment start), staged in pinned memory.
            noise = torch.randn((frag.budget, self.latent_dim), generator=gen, dtype=torch.float32).to(self.dtype)
            frag.noise_host = noise.pin_memory() if pin else noise
            ids = torch.tensor(frag.input_ids, dtype=torch.long)
            frag.ids_host = ids.pin_memory() if pin else ids
        row = _Row(row_id=row_id, fragments=fragments, prompt=prompt, seed=int(seed), stream=stream, generator=gen,
                   trimmer=SilenceTrimmer(self.native_sample_rate),
                   resampler=StreamResampler(self.native_sample_rate, self.output_sample_rate),
                   schedule=ChunkSchedule(self.first_chunk_latents, self.max_chunk_latents))
        stream._row = row
        stream.metrics['fragments'] = len(fragments)
        with self._cv:
            if self._stop or self._thread is None:
                raise RuntimeError('Ming engine is not running')
            self._pending.append(row)
            self._cv.notify_all()
        return stream

    def render(self, text: str, *, prompt: MingVoicePrompt | None = None, language: str | None = None, seed: int = 0,
               row_id: str | None = None) -> np.ndarray:
        stream = self.submit(text, prompt=prompt, language=language, seed=seed, row_id=row_id)
        chunks = list(stream)
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    def stats(self) -> dict[str, Any]:
        with self._cv:
            active = len(self._rows)
            pending = len(self._pending)
        out = {
            'active_rows': active, 'pending_rows': pending, 'max_slots': self.max_slots,
            'free_slots': len(self._free_slots), 'step_ms': round(self._step_ms_ema, 3),
            'vae_ms': round(self._vae_ms_ema, 3), 'steps': self._steps_total,
            'bucket': self._last_bucket, 'voices': len(self._voices), 'loaded': self._loaded,
            'cuda_graphs': bool(self._graphs), 'graphs': len(self._graphs), 'batch_invariant': self.batch_invariant,
            'capture_s': round(self.capture_s, 3), 'warmup_s': round(self.warmup_s, 3),
            'first_chunk_latents': self.first_chunk_latents, 'max_chunk_latents': self.max_chunk_latents,
            'vae_tile_rows': self.vae_tile_rows,
            'linear_plans': linear_plans(),
        }
        if self._loaded and self.device.type == 'cuda':
            out['vram_allocated_mb'] = round(torch.cuda.memory_allocated(self.device) / 2 ** 20, 1)
            out['vram_reserved_mb'] = round(torch.cuda.memory_reserved(self.device) / 2 ** 20, 1)
            out['vram_peak_mb'] = round(torch.cuda.max_memory_allocated(self.device) / 2 ** 20, 1)
        return out

    # -- text / prompt preparation (caller thread) --------------------------------------------------------------------

    def _build_fragments(self, text: str, prompt: MingVoicePrompt | None, language: str | None) -> list[_Fragment]:
        lang = normalize_language_id(language)
        tts_prompt = build_tts_prompt(lang)
        out = []
        for piece in prepare_fragments(text, lang):
            ids, spk_pos, lat_pos = build_input_ids(
                self.tokenizer, tts_prompt, piece, num_speakers=1 if prompt is not None else 0,
                prompt_text=prompt.ref_text if prompt is not None else None,
                prompt_latent_steps=prompt.num_latents if prompt is not None else 0)
            budget = min(MAX_DECODE_STEPS, MAX_CACHE_LEN - len(ids))
            if budget <= STOP_MIN_STEP + 1:
                raise ValueError(f'Ming prompt too long: {len(ids)} of {MAX_CACHE_LEN} positions used by voice prompt '
                                 f'and text fragment {piece!r}; use a shorter reference clip')
            out.append(_Fragment(piece, ids, spk_pos, lat_pos, budget))
        return out

    def _encode_prompt(self, key, ref_text, seed, vae_wave, spk_emb, duration_s) -> MingVoicePrompt:
        m = self.model
        gen = torch.Generator(device='cpu')
        gen.manual_seed(int(seed) % (2 ** 63))
        # Upstream encodes the prompt outside autocast, in bf16.
        with torch.autocast(device_type=self.device.type, enabled=False):
            latent = m.audio.encode_latent(vae_wave.to(device=self.device, dtype=self.dtype), generator=gen)
        latent = latent.to(self.dtype)
        projected_latent = m.linear_proj_audio(latent.reshape(-1, m.patch_size, m.latent_dim))
        projected_latent = projected_latent.reshape(-1, projected_latent.size(-1))
        projected_spk = m.spk_head(spk_emb.to(self.device))[0]
        return MingVoicePrompt(key=key, ref_text=ref_text, seed=int(seed), latent=latent, spk_emb=spk_emb,
                               projected_latent=projected_latent, projected_spk=projected_spk, duration_s=duration_s)

    def _run_on_gpu_thread(self, fn: Callable[[], Any]) -> Any:
        box = {'event': threading.Event(), 'result': None, 'error': None}
        with self._cv:
            if self._thread is None or self._stop:
                raise RuntimeError('Ming engine is not running')
            self._tasks.append((fn, box))
            self._cv.notify_all()
        box['event'].wait()
        if box['error'] is not None:
            raise box['error']
        return box['result']

    def _cancel(self, stream: MingStream) -> None:
        row = stream._row
        if row is None:
            return
        row.cancelled = True
        with self._cv:
            self._cv.notify_all()

    # -- GPU thread ---------------------------------------------------------------------------------------------------

    def _loop(self, ready: dict) -> None:
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, dtype=self.dtype):
            try:
                self._allocate()
                self._prepare()
            except BaseException as exc:  # noqa: BLE001 - handed to load()
                logger.exception('Ming engine preparation failed')
                ready['error'] = exc
                ready['event'].set()
                return
            ready['event'].set()
            while True:
                with self._cv:
                    while (not self._stop and not self._tasks and not self._pending and not self._rows
                           and self._vae_job is None):
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
                    self._deliver_safe()
                    continue
                try:
                    self._iteration(rows)
                except Exception as exc:  # noqa: BLE001 - fail the rows, keep serving
                    logger.exception('Ming step failed')
                    self._fail_all(rows, exc)

    def _fail_all(self, rows: list[_Row], exc: BaseException) -> None:
        self._ahead_bk = None
        job, self._vae_job = self._vae_job, None
        if job is not None:
            for item in job.items:
                item.row.stream._finish(exc)
        for row in rows:
            self._release(row, exc)
        if self.device.type == 'cuda':
            try:
                torch.cuda.synchronize(self.device)
            except Exception:  # noqa: BLE001
                logger.exception('CUDA synchronize after a failed Ming step')

    def _deliver_safe(self) -> None:
        try:
            self._deliver()
        except Exception as exc:  # noqa: BLE001
            logger.exception('Ming audio delivery failed')
            self._fail_all(list(self._rows.values()), exc)

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
        for row in [r for r in self._rows.values() if r.cancelled]:
            self._release(row, None)
        with self._cv:
            if any(r.cancelled for r in self._pending):
                keep = collections.deque()
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

    def _launch_decode(self, rows: list[_Row]) -> None:
        """Enqueue the LLM decode for ``rows`` (their next token at r.pos) and advance their positions."""
        if not rows:
            self._ahead_bk = None
            return
        top = max(r.slot for r in rows) + 1
        bk = self.max_slots if self.batch_invariant else bucket_for(top, self.max_slots)
        buf = self._pos_flip
        self._pos_flip ^= 1
        pos = self.pos_np[buf]
        pos[:bk] = 0
        for r in rows:
            pos[r.slot] = r.pos
            r.pos += 1
        self.pos_gpu[:bk].copy_(self.pos_host[buf][:bk], non_blocking=True)
        self._run(('decode', bk), lambda: self._decode_core(bk))
        self._ahead_bk = bk

    def _iteration(self, rows: list[_Row]) -> None:
        t0 = time.perf_counter()
        decode_rows = [r for r in rows if r.state == 'decode']
        prefill_rows = [r for r in rows if r.state == 'prefill']
        if decode_rows and self._ahead_bk is None:
            self._launch_decode(decode_rows)  # normally already launched at the end of the previous step
        self._ahead_bk = None
        top = max(r.slot for r in rows) + 1
        bk = self.max_slots if self.batch_invariant else bucket_for(top, self.max_slots)
        self._last_bucket = bk
        for r in prefill_rows:
            self._prefill(r)
        steps = self.step_np
        steps[:bk] = 0
        for r in rows:
            steps[r.slot] = r.step
        self.step_gpu[:bk].copy_(self.step_host[:bk], non_blocking=True)
        self._run(('flow', bk), lambda: self._flow_core(bk))
        self.stop_host[:bk].copy_(self.stop_buf[:bk], non_blocking=True)
        if self.stop_event is not None:
            self.stop_event.record()
        # Speculatively enqueue the next step's decode for every row now (its input embedding is what the flow just
        # wrote), so the GPU never waits for the host bookkeeping below. For rows that turn out to have finished
        # their fragment it computes a throw-away token: it only writes KV at a position that is either rewritten
        # by the next prefill or overwritten before it is ever attended to (decode writes a position, then attends
        # to positions <= it).
        self._launch_decode(rows)
        # While the GPU works: hand out the audio of the previous VAE job.
        self._deliver()
        if self.stop_event is not None:
            self.stop_event.synchronize()  # the one sync per step
        stop = self.stop_np
        self._steps_total += 1
        job = _VaeJob()
        for r in rows:
            step = r.step
            r.step += 1
            r.total_steps += 1
            frag_done = bool(stop[r.slot] and step > STOP_MIN_STEP) or r.step >= r.fragment.budget
            r.state = 'decode'
            n = r.schedule.take(r.step, r.decoded, frag_done)
            last = frag_done and r.frag_index + 1 >= len(r.fragments)
            if n:
                job.items.append(_VaeItem(row=r, slot=r.slot, start=r.decoded, n=n, generated=r.step,
                                          final=frag_done, last=last))
                r.decoded += n
            if frag_done:
                if not last:
                    r.frag_index += 1
                    r.state = 'prefill'
                    r.decoded = 0
                else:
                    # The slot is free for a new row now (its latents are gathered below, in stream order); the
                    # stream finishes when this job's audio is delivered.
                    r.stream.metrics['steps'] = r.total_steps
                    self._free_slot(r)
        if job.items:
            self._launch_vae(job)
        dt = (time.perf_counter() - t0) * 1000.0
        self._step_ms_ema = dt if self._step_ms_ema == 0.0 else 0.9 * self._step_ms_ema + 0.1 * dt

    # -- LLM ----------------------------------------------------------------------------------------------------------

    def _decode_core(self, bk: int) -> None:
        """One LLM token for slots [0, bk): reads next_embed/pos_gpu, writes z_buf and the KV cache. Graph-safe."""
        q = self.fq
        h = self.next_embed[:bk]
        pos = self.pos_gpu[:bk]
        cos, sin = q.rope(pos.unsqueeze(1))
        # Additive fp32 mask over all cache positions: 0 up to the row's position, -inf beyond.
        mask = torch.zeros(bk, 1, 1, MAX_CACHE_LEN, device=h.device, dtype=torch.float32).masked_fill(
            (self.kv_arange.unsqueeze(0) > pos.unsqueeze(1)).view(bk, 1, 1, MAX_CACHE_LEN), float('-inf'))
        bidx = self.slot_index[:bk]
        n_rep = self.n_heads // self.n_kv
        g = bk * self.n_kv
        for i in range(self.n_layers):
            qh, kh, vh = q.layer_qkv(i, h, cos, sin)
            kc, vc = self.k_cache[i], self.v_cache[i]
            kc[bidx, :, pos] = kh[:, :, 0, :]
            vc[bidx, :, pos] = vh[:, :, 0, :]
            # GQA without repeating K/V: the n_rep query heads of a KV group become the query axis. One query per
            # row: explicit attention (fp32 scores and softmax, like the fused kernels) is ~2x faster than SDPA's
            # memory-efficient kernel at this shape, which parallelises poorly over 512 keys.
            qg = qh.reshape(g, n_rep, self.head_dim)
            scores = torch.bmm(qg, kc[:bk].view(g, MAX_CACHE_LEN, self.head_dim).transpose(1, 2),
                               out_dtype=torch.float32).view(bk, self.n_kv, n_rep, MAX_CACHE_LEN)
            probs = torch.softmax(scores * q.scaling + mask, dim=-1).to(vc.dtype)
            attn = torch.bmm(probs.view(g, n_rep, MAX_CACHE_LEN), vc[:bk].view(g, MAX_CACHE_LEN, self.head_dim))
            h = q.layer_out(i, h, attn.reshape(bk, 1, self.n_heads * self.head_dim))
        self.z_buf[:bk].copy_(q.final_norm(h))

    def _prefill_core(self, lb: int) -> None:
        """Causal forward of prefill_h[:, :lb] for slot prefill_slot: KV cache [0, lb) and z_buf. Graph-safe.

        Right padding is harmless: a query never sees later (padded) keys, and the decode mask only admits KV
        positions below the row's position counter, which the padding positions are not."""
        q = self.fq
        h = self.prefill_h[:, :lb]
        cos, sin = q.cos[:lb].unsqueeze(0), q.sin[:lb].unsqueeze(0)
        slot = self.prefill_slot
        for i in range(self.n_layers):
            qh, kh, vh = q.layer_qkv(i, h, cos, sin)
            self.k_cache[i][slot, :, :lb] = kh
            self.v_cache[i][slot, :, :lb] = vh
            attn = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True, enable_gqa=True, scale=q.scaling)
            h = q.layer_out(i, h, attn.transpose(1, 2).reshape(1, lb, self.n_heads * self.head_dim))
        z = q.final_norm(h.index_select(1, self.prefill_last))
        self.z_buf[slot] = z

    def _embed(self, r: _Row, length: int) -> None:
        """Token embeddings of the row's fragment (prompt latents and speaker embedding spliced in) -> prefill_h."""
        frag = r.fragment
        ids = self.prefill_ids[:length]
        ids.copy_(frag.ids_host, non_blocking=True)
        emb = self.prefill_h[0, :length]
        torch.index_select(self.model.llm.embed_tokens.weight, 0, ids, out=emb)
        if r.prompt is not None:
            for p in frag.spk_positions:
                emb[p] = r.prompt.projected_spk.to(emb.dtype)
            if frag.latent_position >= 0:
                n = r.prompt.num_latents
                emb[frag.latent_position:frag.latent_position + n] = r.prompt.projected_latent.to(emb.dtype)

    def _prefill(self, r: _Row) -> None:
        """Prompt forward for a row starting a fragment: fills its KV slot, z_buf, history and noise."""
        m = self.model
        frag = r.fragment
        length = len(frag.input_ids)
        lb = -(-length // PREFILL_BUCKET) * PREFILL_BUCKET
        self._embed(r, length)
        self.prefill_slot.fill_(r.slot)
        self.prefill_last.fill_(length - 1)
        self._run(('prefill', lb), lambda: self._prefill_core(lb))
        self.noise[r.slot, :frag.budget].copy_(frag.noise_host, non_blocking=True)
        if r.prompt is not None:
            self.history[r.slot].copy_(r.prompt.latent[0, -m.history_patch_size:].to(self.dtype))
        else:
            self.history[r.slot].zero_()
        r.pos = length
        r.step = 0
        r.decoded = 0

    # -- flow head, stop head, aggregator -------------------------------------------------------------------------

    def _flow_core(self, bk: int) -> None:
        """Sample one latent per slot in [0, bk) and prepare the next LM input. Graph-safe."""
        bidx = self.slot_index[:bk]
        step = self.step_gpu[:bk]
        z = self.z_buf[:bk]
        x0 = self.noise[bidx, step].unsqueeze(1)
        latent = self.ff.sample(x0, z, self.history[:bk])
        self.stop_buf[:bk].copy_(self.ff.stop(z))
        self.lat_buf[bidx, step] = latent[:, 0, :]
        self.history[:bk].copy_(latent)
        self.next_embed[:bk].copy_(self.ff.aggregate(latent))

    # -- audio --------------------------------------------------------------------------------------------------------

    def _launch_vae(self, job: _VaeJob) -> None:
        """Gather every due row's latent window (main stream), then decode on the VAE stream (overlaps the next
        LM step) and copy the audio to pinned host memory."""
        vae = self.vae
        cmax = self.vae_chunks[-1]
        items = job.items
        for item in items:
            item.chunk = cmax if self.batch_invariant else next(c for c in self.vae_chunks if c >= item.n)
        items.sort(key=lambda it: it.chunk)
        idx = self.vae_idx_host
        for k, it in enumerate(items):
            it.index = k
            # Window [start-1, start+n] (+ padding), clamped to the generated latents: left edge repeats latent 0
            # (upsampler edge clamp), the right end is the lookahead latent or, for the final chunk, a repeat.
            lo, hi = it.start - 1, it.generated - 1
            window = np.clip(np.arange(lo, lo + cmax + 2), 0, hi)
            idx[k].copy_(torch.from_numpy(window))
            self.vae_slot_host[k] = it.slot
        n_items = len(items)
        self.vae_idx_dev[:n_items].copy_(idx[:n_items], non_blocking=True)
        self.vae_slot_dev[:n_items].copy_(self.vae_slot_host[:n_items], non_blocking=True)
        self.vae_stage[:n_items] = self.lat_buf[self.vae_slot_dev[:n_items, None], self.vae_idx_dev[:n_items]]
        main = torch.cuda.current_stream(self.device) if self.vae_stream is not None else None
        job.launched_at = time.perf_counter()
        ctx = torch.cuda.stream(self.vae_stream) if self.vae_stream is not None else _nullcontext()
        with ctx:
            if main is not None:
                self.vae_stream.wait_stream(main)
            k = 0
            meta_off = 0
            while k < n_items:
                chunk = items[k].chunk
                same = sum(1 for it in items[k:] if it.chunk == chunk)
                rows = self.vae_tile_rows if self.batch_invariant else bucket_for(same, self.max_slots)
                count = min(same, rows)
                buf = self._vae_shape(rows, chunk)
                # Own pinned region per tile: the previous tile's asynchronous copy may still be pending. (Jobs are
                # serialised -- delivered before the next launch -- so regions are free again at the next job.)
                meta = self.vae_meta_np[meta_off:meta_off + rows]
                meta[:] = 0
                meta[:, 0] = vae.trash  # padding rows: trash slot, no valid frames
                for j, it in enumerate(items[k:k + count]):
                    meta[j] = (it.slot, vae.scale * it.start, vae.scale * it.n, 1 if it.start == 0 else 0)
                buf['meta'].copy_(self.vae_meta_host[meta_off:meta_off + rows], non_blocking=True)
                meta_off += rows
                buf['lat'][:count].copy_(self.vae_stage[k:k + count, :chunk + 2])
                self._run(('vae', rows, chunk), lambda b=buf: vae.run(b['lat'], b['meta'], b['out']))
                self.vae_out_host[k:k + count, :buf['out'].shape[1]].copy_(buf['out'][:count], non_blocking=True)
                k += count
            if self.vae_stream is not None:
                job.event = torch.cuda.Event()
                job.event.record(self.vae_stream)
        self._vae_job = job

    def _deliver(self) -> None:
        job, self._vae_job = self._vae_job, None
        if job is None:
            return
        if job.event is not None:
            job.event.synchronize()
        dt = (time.perf_counter() - job.launched_at) * 1000.0
        self._vae_ms_ema = dt if self._vae_ms_ema == 0.0 else 0.9 * self._vae_ms_ema + 0.1 * dt
        vae = self.vae
        frames_hop = vae.scale * vae.hop
        for it in job.items:
            r = it.row
            if r.stream.done:
                continue
            host = self.vae_out_host[it.index]
            audio = host[:it.n * frames_hop].numpy().copy()
            if it.start == 0:
                audio = audio[vae.pad:]  # "same" padding: the first pad samples of a fragment are dropped
            if not it.final:
                out = r.trimmer.push(audio)
            else:
                tail_at = it.chunk * frames_hop
                tail = host[tail_at:tail_at + vae.pad].numpy()
                out = r.trimmer.finish(np.concatenate([audio, tail]))
            if out.size:
                r.stream._put(r.resampler.push(out))
            if it.last:
                r.stream._put(r.resampler.flush())
                r.stream.metrics['steps'] = r.total_steps
                r.stream._finish()


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
