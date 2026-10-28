"""Ming-3.0-tts 0.5B engine: continuous batching, per-row determinism, true chunk streaming.

One dedicated GPU thread owns every CUDA operation. Each loop iteration ("step") does, for all active rows at once:

1. batched LLM decode of rows that already have an input embedding (slot KV cache, per-row positions, padded mask),
2. batched prefill of rows that join (or start their next text fragment) at this step boundary,
3. batched CFG flow sampling (10 Euler steps, both branches) with per-row starting noise drawn from the row's own
   seeded generator, the stop head, and the aggregator projection for the next input,
4. ONE device->host copy (the stop flags),
5. per-row streaming VAE decode on a growing chunk schedule, streaming trailing-silence trim, stateful resampling.

The math per row is the upstream ``BailingMMNativeForConditionalGeneration.sample`` loop (cfg 2.0, sigma 0.25,
temperature 0, flow steps 10, stop when p(stop) > 0.5 and step > 3, 512-position budget including the prompt) and
``ming_tts.MingTTS.synthesize`` (fragmenting, per-fragment trailing-silence trim, same prompt for every fragment).

``_decode_core`` and ``_flow_core`` only read/write preallocated per-slot buffers with static shapes for a given
bucket size (1/2/4/8/16/32/64 active slots), so they can be captured as CUDA graphs.
"""
from __future__ import annotations

import collections
import heapq
import itertools
import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import torch
import torch.nn.functional as F

from .audio_vae import DecodeStreamState
from .flow import flow_timesteps, sample_flow
from .model import MAX_CACHE_LEN, MingModel, load_ming_model
from .prompt import build_input_ids, build_tts_prompt, normalize_language_id, validate_tokenizer
from .streaming import ChunkSchedule, SilenceTrimmer, StreamResampler
from .text import prepare_fragments

logger = logging.getLogger(__name__)

# Quality knobs (fixed; see upstream ming_tts.synthesize defaults).
CFG_SCALE = 2.0
SIGMA = 0.25
TEMPERATURE = 0.0
FLOW_STEPS = 10
STOP_MIN_STEP = 3  # upstream: stop only when step > 3
MAX_DECODE_STEPS = 1000
BUCKETS = (1, 2, 4, 8, 16, 32, 64)
PREFILL_TOKEN_BUDGET = 8192  # rows per prefill call are limited to about this many (padded) tokens
_END = object()


def bucket_for(n: int, max_slots: int) -> int:
    """Smallest bucket holding ``n`` slots (capped at ``max_slots``)."""
    for b in BUCKETS:
        if b >= n:
            return min(b, max_slots)
    return max_slots


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q, k, cos, sin):
    # transformers' Qwen2 apply_rotary_pos_emb (unsqueeze_dim=1), inlined so no hub kernel can replace it.
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


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
    decoded: int = 0        # latents of the current fragment already fed to the VAE decoder
    decode_state: DecodeStreamState | None = None
    cancelled: bool = False
    total_steps: int = 0

    @property
    def fragment(self) -> _Fragment:
        return self.fragments[self.frag_index]


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
        if chunk.size == 0:
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
                 dtype: torch.dtype = torch.bfloat16, batch_invariant: bool = True) -> None:
        self.model_dir = Path(model_dir)
        self.device = torch.device(device)
        self.max_slots = max(1, min(64, int(max_slots)))
        self.cuda_graphs = bool(cuda_graphs)
        self.output_sample_rate = int(output_sample_rate)
        self.first_chunk_latents = max(1, int(first_chunk_latents))
        self.max_chunk_latents = max(self.first_chunk_latents, int(max_chunk_latents))
        self.dtype = dtype
        # batch_invariant: every step runs at the full slot count and every prefill is a single row, so each GEMM /
        # attention call has the same shape whatever else is running and a row's numbers do not depend on its
        # neighbours (bf16 kernels pick tiles by shape; with varying shapes the autoregressive loop amplifies the
        # last-bit differences into a different, equally valid, rendition).
        self.batch_invariant = bool(batch_invariant)
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
        self._steps_total = 0
        self._last_bucket = 0
        self._loaded = False
        self.graph_runner = None  # optional CUDA-graph replayer installed by an optimisation layer

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
        self._allocate()
        self._stop = False
        self._free_slots = list(range(self.max_slots))
        heapq.heapify(self._free_slots)
        self._thread = threading.Thread(target=self._loop, name='ming-gpu', daemon=True)
        self._thread.start()
        self._loaded = True

    def _allocate(self) -> None:
        m = self.model
        cfg = m.llm.config
        S, dev, dt = self.max_slots, self.device, self.dtype
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
        self.pos_host = torch.zeros(S, dtype=torch.long, pin_memory=dev.type == 'cuda')
        self.step_host = torch.zeros(S, dtype=torch.long, pin_memory=dev.type == 'cuda')
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
            self._pending.clear()
            self._rows.clear()
            while self._tasks:
                _, box = self._tasks.popleft()
                box['error'] = err
                box['event'].set()
        self.graph_runner = None
        for name in ('k_cache', 'v_cache', 'pos_gpu', 'step_gpu', 'slot_index', 'kv_arange', 'next_embed', 'z_buf',
                     'history', 'noise', 'lat_buf', 'stop_buf', 'timesteps'):
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
            'free_slots': len(self._free_slots), 'step_ms': round(self._step_ms_ema, 3), 'steps': self._steps_total,
            'bucket': self._last_bucket, 'voices': len(self._voices), 'loaded': self._loaded,
            'cuda_graphs': self.cuda_graphs and self.graph_runner is not None,
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

    def _loop(self) -> None:
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, dtype=self.dtype):
            while True:
                with self._cv:
                    while not self._stop and not self._tasks and not self._pending and not self._rows:
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
                if not self._rows:
                    continue
                rows = list(self._rows.values())
                try:
                    self._iteration(rows)
                except Exception as exc:  # noqa: BLE001 - fail the rows, keep serving
                    logger.exception('Ming step failed')
                    for row in rows:
                        self._release(row, exc)

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

    def _release(self, row: _Row, error: BaseException | None) -> None:
        with self._cv:
            if self._rows.get(row.slot) is row:
                del self._rows[row.slot]
                heapq.heappush(self._free_slots, row.slot)
        row.decode_state = None
        row.stream.metrics['steps'] = row.total_steps
        row.stream._finish(error)

    def _iteration(self, rows: list[_Row]) -> None:
        t0 = time.perf_counter()
        decode_rows = [r for r in rows if r.state == 'decode']
        prefill_rows = [r for r in rows if r.state == 'prefill']
        top = max(r.slot for r in rows) + 1
        bk = self.max_slots if self.batch_invariant else bucket_for(top, self.max_slots)
        self._last_bucket = bk
        # Host -> device: positions and per-fragment step counters of the slots in the bucket.
        self.pos_host[:bk].zero_()
        self.step_host[:bk].zero_()
        for r in decode_rows:
            self.pos_host[r.slot] = r.pos
            self.step_host[r.slot] = r.step
        self.pos_gpu[:bk].copy_(self.pos_host[:bk], non_blocking=True)
        self.step_gpu[:bk].copy_(self.step_host[:bk], non_blocking=True)
        if decode_rows:
            if self.graph_runner is not None:
                self.graph_runner.decode(bk)
            else:
                self._decode_core(bk)
            for r in decode_rows:
                r.pos += 1
        if prefill_rows:
            self._prefill(prefill_rows)
        if self.graph_runner is not None:
            self.graph_runner.flow(bk)
        else:
            self._flow_core(bk)
        stop = self.stop_buf[:bk].cpu().tolist()  # the one sync per step
        self._steps_total += 1
        for r in rows:
            step = r.step
            r.step += 1
            r.total_steps += 1
            frag_done = (stop[r.slot] and step > STOP_MIN_STEP) or r.step >= r.fragment.budget
            r.state = 'decode'
            self._emit_audio(r, frag_done)
        dt = (time.perf_counter() - t0) * 1000.0
        self._step_ms_ema = dt if self._step_ms_ema == 0.0 else 0.9 * self._step_ms_ema + 0.1 * dt

    # -- LLM ----------------------------------------------------------------------------------------------------------

    def _layer_qkv(self, layer, x, cos, sin):
        attn = layer.self_attn
        B, L, _ = x.shape
        q = attn.q_proj(x).view(B, L, -1, self.head_dim).transpose(1, 2)
        k = attn.k_proj(x).view(B, L, -1, self.head_dim).transpose(1, 2)
        v = attn.v_proj(x).view(B, L, -1, self.head_dim).transpose(1, 2)
        q, k = _apply_rope(q, k, cos, sin)
        return q, k, v

    def _decode_core(self, bk: int) -> None:
        """One LLM token for slots [0, bk): reads next_embed/pos_gpu, writes z_buf and the KV cache. Graph-safe."""
        llm = self.model.llm
        h = self.next_embed[:bk]
        pos = self.pos_gpu[:bk]
        cos, sin = llm.rotary_emb(h, pos.unsqueeze(1))
        mask = (self.kv_arange.unsqueeze(0) <= pos.unsqueeze(1)).view(bk, 1, 1, MAX_CACHE_LEN)
        bidx = self.slot_index[:bk]
        n_rep = self.n_heads // self.n_kv
        for i, layer in enumerate(llm.layers):
            residual = h
            x = layer.input_layernorm(h)
            q, k, v = self._layer_qkv(layer, x, cos, sin)
            kc, vc = self.k_cache[i], self.v_cache[i]
            kc[bidx, :, pos] = k[:, :, 0, :]
            vc[bidx, :, pos] = v[:, :, 0, :]
            # GQA without repeating K/V: the n_rep query heads of a KV group become the query axis.
            qg = q.reshape(bk, self.n_kv, n_rep, self.head_dim)
            attn = F.scaled_dot_product_attention(qg, kc[:bk], vc[:bk], attn_mask=mask,
                                                  scale=layer.self_attn.scaling)
            attn = attn.reshape(bk, 1, self.n_heads * self.head_dim)
            h = residual + layer.self_attn.o_proj(attn)
            residual = h
            h = residual + layer.mlp(layer.post_attention_layernorm(h))
        self.z_buf[:bk].copy_(llm.norm(h))

    def _prefill(self, rows: list[_Row]) -> None:
        """Prompt forward for rows starting a fragment: fills their KV slots, z_buf, history and noise."""
        m = self.model
        groups: list[list[_Row]] = [[]]
        longest = 0
        for r in sorted(rows, key=lambda r: len(r.fragment.input_ids)):
            n = len(r.fragment.input_ids)
            if groups[-1] and (self.batch_invariant
                               or max(longest, n) * (len(groups[-1]) + 1) > PREFILL_TOKEN_BUDGET):
                groups.append([])
                longest = 0
            groups[-1].append(r)
            longest = max(longest, n)
        for group in groups:
            self._prefill_group(group)
        for r in rows:
            frag = r.fragment
            # Starting noise for every LM step of this fragment, from the row's own generator.
            noise = torch.randn((frag.budget, self.latent_dim), generator=r.generator, dtype=torch.float32)
            self.noise[r.slot, :frag.budget].copy_(noise.to(self.dtype))
            if r.prompt is not None:
                self.history[r.slot].copy_(r.prompt.latent[0, -m.history_patch_size:].to(self.dtype))
            else:
                self.history[r.slot].zero_()
            r.pos = len(frag.input_ids)
            r.step = 0
            r.decoded = 0
            r.decode_state = DecodeStreamState()

    def _embed(self, r: _Row) -> torch.Tensor:
        frag = r.fragment
        ids = torch.tensor(frag.input_ids, dtype=torch.long, device=self.device)
        emb = self.model.llm.embed_tokens(ids)
        if r.prompt is not None:
            for p in frag.spk_positions:
                emb[p] = r.prompt.projected_spk.to(emb.dtype)
            if frag.latent_position >= 0:
                n = r.prompt.num_latents
                emb[frag.latent_position:frag.latent_position + n] = r.prompt.projected_latent.to(emb.dtype)
        return emb

    def _prefill_group(self, rows: list[_Row]) -> None:
        llm = self.model.llm
        lengths = [len(r.fragment.input_ids) for r in rows]
        lmax = max(lengths)
        n = len(rows)
        h = torch.zeros(n, lmax, self.hidden, device=self.device, dtype=self.dtype)
        for i, r in enumerate(rows):
            h[i, :lengths[i]] = self._embed(r)
        position_ids = torch.arange(lmax, device=self.device).unsqueeze(0).expand(n, -1)
        cos, sin = llm.rotary_emb(h, position_ids)
        slots = torch.tensor([r.slot for r in rows], device=self.device, dtype=torch.long)
        for i, layer in enumerate(llm.layers):
            residual = h
            x = layer.input_layernorm(h)
            q, k, v = self._layer_qkv(layer, x, cos, sin)
            self.k_cache[i][slots, :, :lmax] = k
            self.v_cache[i][slots, :, :lmax] = v
            # Right padding + causal attention: a valid query never sees a padded key.
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True,
                                                  scale=layer.self_attn.scaling)
            attn = attn.transpose(1, 2).reshape(n, lmax, self.n_heads * self.head_dim)
            h = residual + layer.self_attn.o_proj(attn)
            residual = h
            h = residual + layer.mlp(layer.post_attention_layernorm(h))
        last = torch.tensor([l - 1 for l in lengths], device=self.device, dtype=torch.long)
        z = llm.norm(h[torch.arange(n, device=self.device), last]).unsqueeze(1)
        self.z_buf[slots] = z

    # -- flow head, stop head, aggregator -------------------------------------------------------------------------

    def _flow_core(self, bk: int) -> None:
        """Sample one latent per slot in [0, bk) and prepare the next LM input. Graph-safe."""
        m = self.model
        bidx = self.slot_index[:bk]
        step = self.step_gpu[:bk]
        z = self.z_buf[:bk]
        x0 = self.noise[bidx, step].unsqueeze(1)
        latent = sample_flow(m.dit, x0, z, self.history[:bk], cfg_scale=CFG_SCALE, steps=FLOW_STEPS,
                             patch_size=m.patch_size, sigma=SIGMA, temperature=TEMPERATURE, timesteps=self.timesteps)
        self.stop_buf[:bk].copy_(m.stop_head(z).softmax(dim=-1)[:, 0, 1] > 0.5)
        self.lat_buf[bidx, step] = latent[:, 0, :]
        self.history[:bk].copy_(latent)
        self.next_embed[:bk].copy_(m.linear_proj_audio(latent))

    # -- audio --------------------------------------------------------------------------------------------------------

    def _emit_audio(self, r: _Row, frag_done: bool) -> None:
        pending = r.step - r.decoded
        if not frag_done and not r.schedule.ready(pending):
            return
        dec = self.model.audio.decoder
        latents = self.lat_buf[r.slot:r.slot + 1, r.decoded:r.step]
        r.decoded = r.step
        if r.decode_state.latents_in == 0 and not frag_done and latents.shape[1] > 1:
            # The upsampler only buffers its first input (one-latent lookahead). Feed the first latent alone so this
            # call already yields audio for latents [0, n-1) instead of waiting for the next chunk.
            dec.stream_decode(latents[:, :1], r.decode_state, last_chunk=False)
            latents = latents[:, 1:]
        audio = dec.stream_decode(latents, r.decode_state, last_chunk=frag_done)
        audio = audio[0, 0].float().cpu().numpy()
        if not frag_done:
            r.schedule.advance()
            out = r.trimmer.push(audio)
            if out.size:
                r.stream._put(r.resampler.push(out))
            return
        out = r.trimmer.finish(audio)
        if out.size:
            r.stream._put(r.resampler.push(out))
        r.decode_state = None
        if r.frag_index + 1 < len(r.fragments):
            r.frag_index += 1
            r.state = 'prefill'
            return
        r.stream._put(r.resampler.flush())
        self._release(r, None)
