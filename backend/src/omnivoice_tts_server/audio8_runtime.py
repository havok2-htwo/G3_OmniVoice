"""The Audio8 models behind the synthesizer interface the queue uses, one class for all of them.

``audio8/variants.py`` lists the checkpoints; each is an engine of its own in ``engines.py``. Everything the queue
needs beyond the engine (voices resolved like OmniVoice and cached per voice in the engine, every sentence an engine
row, chunks forwarded as they are decoded, one ``sentence_finished`` marker per sentence, concurrent batches joining the
running engine batch) is the same as for Ming, so this class only says what differs.

Language: Audio8 has no language parameter, the text alone decides. Requests may send ``language``; it is accepted and
not used. Delivery controls (emotion, rate, instructions, ...) do not exist for this model and are not read.

Output is PCM16 mono at ``audio8_output_sample_rate`` (44.1 kHz by default, the native rate).

``audio8_onnx`` runs on the CPU (``audio8/onnx_engine.py``): it has workers instead of slots and needs no VRAM. Its
engine has the same interface, so only the settings it reads differ here.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable

from .audio8.variants import Audio8Variant, audio8_variant
from .audio8.voice_cache import prune_voices, voice_cache_folder
from .capacity import audio8_capacity_profile, audio8_onnx_capacity_profile
from .config import Settings
from .domain.state import InMemoryStore
from .ming_runtime import _CACHE_MARKER, MING_DEFAULT_SEED, WARMUP_TEXT, MingSynthesizer, _decode_audio
from .runtime_v2 import BatchSynthesisItem, normalize_runtime_device


def _default_engine_factory(**kwargs: Any) -> Any:
    from .audio8.engine import Audio8Engine

    return Audio8Engine(**kwargs)


def _onnx_engine_factory(**kwargs: Any) -> Any:
    from .audio8.onnx_engine import Audio8OnnxEngine

    return Audio8OnnxEngine(**kwargs)


class Audio8Synthesizer(MingSynthesizer):
    def __init__(self, settings: Settings, store: InMemoryStore | None, *, variant: str = 'audio8_realistic',
                 engine_factory: Callable[..., Any] | None = None) -> None:
        self.variant: Audio8Variant = audio8_variant(variant)
        self.on_cpu = self.variant.runtime == 'onnx'
        self.engine_id = self.variant.engine_id
        self.engine_label = self.variant.label
        self.base_alias = self.variant.base_alias
        default_factory = _onnx_engine_factory if self.on_cpu else _default_engine_factory
        super().__init__(settings, store, engine_factory=engine_factory or default_factory)

    def _capacity(self, slots: int) -> dict[str, Any]:
        if self.on_cpu:
            return audio8_onnx_capacity_profile(slots, self.engine_id)
        return audio8_capacity_profile(slots, self.engine_id)

    def _configured_slots(self) -> int:
        return int(self.settings.audio8_onnx_workers if self.on_cpu else self.settings.audio8_max_slots)

    def _read_settings(self) -> None:
        self.model_id = self.variant.repo_id
        self.max_slots = self._configured_slots()
        self.sample_rate = int(self.settings.audio8_output_sample_rate)

    def _model_dir(self) -> Path:
        return Path(self.settings.models_root_dir) / self.variant.folder

    def _engine_kwargs(self) -> dict[str, Any]:
        settings = self.settings
        if self.on_cpu:
            return {
                'model_dir': self._model_dir(),
                'workers': self.max_slots,
                'threads': int(settings.audio8_onnx_threads),
                'output_sample_rate': self.sample_rate,
                'first_chunk_frames': int(settings.audio8_onnx_first_chunk_frames),
                'chunk_frames': int(settings.audio8_onnx_chunk_frames),
                'fragment_max_chars': int(settings.audio8_fragment_max_chars),
                'temperature': float(settings.audio8_temperature),
                'top_p': float(settings.audio8_top_p),
                'top_k': int(settings.audio8_top_k),
                'voice_cache_dir': voice_cache_folder(settings.data_dir),
            }
        return {
            'model_dir': self._model_dir(),
            'device': normalize_runtime_device(getattr(settings, 'preferred_device', None)),
            'max_slots': self.max_slots,
            'cuda_graphs': bool(settings.audio8_cuda_graphs),
            'output_sample_rate': self.sample_rate,
            'first_chunk_frames': int(settings.audio8_first_chunk_frames),
            'chunk_frames': int(settings.audio8_chunk_frames),
            'fragment_max_chars': int(settings.audio8_fragment_max_chars),
            'temperature': float(settings.audio8_temperature),
            'top_p': float(settings.audio8_top_p),
            'top_k': int(settings.audio8_top_k),
        }

    def _load_sync(self) -> int:
        self._prune_voice_files()  # a voice may be gone since the files were written, with this engine not loaded
        return super()._load_sync()

    def _prune_voice_files(self) -> None:
        if self.on_cpu and self.store is not None:
            prune_voices(voice_cache_folder(self.settings.data_dir), [p.voice_id for p in self._profiles()])

    def _sync_voice_cache(self, engine: Any) -> None:
        changed = _CACHE_MARKER not in getattr(self.store, 'prompt_cache', {_CACHE_MARKER: True})
        super()._sync_voice_cache(engine)
        if changed:  # a request that was encoding while its voice was deleted wrote the file afterwards
            self._prune_voice_files()

    def _warmup_render(self, engine: Any, prompt: Any, language: str) -> None:
        engine.render(WARMUP_TEXT, prompt=prompt, seed=MING_DEFAULT_SEED)

    def runtime_info(self) -> dict[str, Any]:
        engine = self._engine
        stats: dict[str, Any] = {}
        if engine is not None:
            try:
                stats = dict(engine.stats())
            except Exception as exc:  # noqa: BLE001
                stats = {'error': str(exc)}
        info = {
            'engine': self.engine_id,
            'model_id': self.model_id,
            'model_dir': str(self._model_dir()),
            'revision': self.variant.revision,
            'license': self.variant.license,
            'loaded': engine is not None,
            'output_sample_rate': self.sample_rate,
            'engine_stats': stats,
        }
        if self.on_cpu:
            return {
                **info,
                'workers': self.max_slots,
                'threads': stats.get('threads', int(self.settings.audio8_onnx_threads)),
                'first_chunk_frames': int(self.settings.audio8_onnx_first_chunk_frames),
                'chunk_frames': int(self.settings.audio8_onnx_chunk_frames),
                'attention_backend': None,
                'attention_backend_note': 'Audio8 CPU engine: onnxruntime INT8',
            }
        return {
            **info,
            'max_slots': self.max_slots,
            'cuda_graphs': bool(self.settings.audio8_cuda_graphs),
            'first_chunk_frames': int(self.settings.audio8_first_chunk_frames),
            'chunk_frames': int(self.settings.audio8_chunk_frames),
            'attention_backend': 'sdpa' if engine is not None else None,
            'attention_backend_note': 'Audio8 engine: slot KV with CUDA graphs' if stats.get('cuda_graphs') else None,
        }

    async def free_memory(self) -> dict[str, int | str | bool | None]:
        if not self.on_cpu:
            return await super().free_memory()
        closed = await asyncio.to_thread(self._close_sync, only_if_idle=True)
        return {
            'ok': True,
            'message': (f'{self.engine_label} engine closed; it reloads with the next request.' if closed
                        else f'{self.engine_label} engine is rendering and stays loaded (retry when idle).'),
            'memory_before_mb': None,
            'memory_after_mb': None,
            'memory_total_mb': None,
            'released_mb': None,
        }

    def _voice_prompt(self, engine: Any, key: str, raw: bytes, ref_text: str, voice_id: str | None = None) -> Any:
        # Decoding a reference WAV takes about a millisecond; the engine returns its cached prompt for a known key.
        audio, sample_rate = _decode_audio(raw)
        on_disk = {'voice_id': voice_id} if self.on_cpu else {}
        prompt = engine.voice_prompt(key, audio, sample_rate, ref_text, **on_disk)
        with self._voice_lock:
            self._voice_keys.add(key)
        return prompt

    def _submit_row(self, engine: Any, item: BatchSynthesisItem) -> Any:
        return engine.submit(item.text, prompt=self._prompt_for(engine, item.request), seed=self._seed(item.request),
                             row_id=f'{item.job_id}:{item.sentence_index}')
