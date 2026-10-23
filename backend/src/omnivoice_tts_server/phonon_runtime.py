"""Gradium Phonon engine behind the same synthesizer interface the queue uses for OmniVoice.

Phonon (`ptts` package) has no batch API: one text per synthesis call. Batching here means
several sentences render at once on a pool of engine instances, and streaming means each
sentence's chunks are forwarded as the engine produces them. The queue puts them back in
sentence order (`_flush_native_streaming_sentences_locked`).
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import io
import logging
import os
import queue
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncIterator

import numpy as np

from .config import Settings
from .domain.state import InMemoryStore
from .runtime_v2 import BatchSynthesisItem, BatchSynthesisResult

logger = logging.getLogger(__name__)


def _float_to_pcm16(audio: np.ndarray) -> bytes:
    clipped = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0)
    return (clipped * 32767.0).astype('<i2').tobytes()


class PhononSynthesizer:
    # Tells the queue that `stream_batch` exists and may run for requests of any task type.
    supports_native_streaming = True

    def __init__(self, settings: Settings, store: InMemoryStore) -> None:
        self.settings = settings
        self.store = store
        self.model_id = settings.phonon_model_id
        self.sample_rate = 24_000
        self._loaded_model_id: str | None = None
        self._voice_names: list[str] = []
        self._free: queue.Queue[Any] = queue.Queue()
        self._executor: ThreadPoolExecutor | None = None
        self._ensure_lock: asyncio.Lock | None = None
        self._warm_ms = 0

    # -- model lifecycle -------------------------------------------------------------------------------------------

    def _thread_budget(self) -> int:
        if self.settings.phonon_threads:
            return self.settings.phonon_threads
        return max(1, (os.cpu_count() or 4) // (2 * self.settings.phonon_workers))

    def _load_sync(self) -> None:
        # Read by the Rust runtime when its pool is first used, so it must be set before the engines run.
        os.environ.setdefault('RAYON_NUM_THREADS', str(self._thread_budget()))
        import ptts

        model_dir = self.settings.phonon_model_dir
        if not (model_dir / 'config.json').exists():
            raise RuntimeError(f'Phonon checkpoint not found at {model_dir}. Download {self.model_id} there first.')
        started = time.perf_counter()
        engines = [
            ptts.TTS(str(model_dir), lang=self.settings.phonon_lang, quant=self.settings.phonon_quant)
            for _ in range(self.settings.phonon_workers)
        ]
        self.sample_rate = engines[0].sample_rate
        self._voice_names = list(engines[0].voices)
        for engine in engines:
            self._free.put(engine)
        self._executor = ThreadPoolExecutor(max_workers=self.settings.phonon_workers, thread_name_prefix='phonon')
        self._warm_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            'phonon loaded dir=%s workers=%s lang=%s quant=%s voices=%s load_ms=%s',
            model_dir, len(engines), self.settings.phonon_lang, self.settings.phonon_quant,
            len(self._voice_names), self._warm_ms,
        )

    def load_now(self) -> None:
        self._load_sync()
        self._loaded_model_id = self.model_id

    def voice_names(self) -> list[str]:
        return list(self._voice_names)

    async def ensure_model(self, requested_model: str | None) -> tuple[str, int]:
        if self._loaded_model_id is not None:
            return self.model_id, 0
        if self._ensure_lock is None:
            self._ensure_lock = asyncio.Lock()
        async with self._ensure_lock:
            if self._loaded_model_id is None:
                await asyncio.to_thread(self._load_sync)
                self._loaded_model_id = self.model_id
                return self.model_id, self._warm_ms
        return self.model_id, 0

    def resident_models(self) -> set[str]:
        return {self.model_id} if self._loaded_model_id else set()

    async def preload(self, requested_model: str | None = None) -> tuple[str, int]:
        return await self.ensure_model(requested_model)

    async def warmup(self, requested_model: str | None = None, request: Any = None) -> tuple[str, int]:
        return await self.ensure_model(requested_model or (request.model if request else None))

    async def unload(self) -> tuple[str, int]:
        self._loaded_model_id = None
        self._free = queue.Queue()
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        gc.collect()
        return self.model_id, 0

    async def reload(self, requested_model: str | None = None) -> tuple[str, int]:
        await self.unload()
        return await self.ensure_model(requested_model)

    async def free_memory(self) -> dict[str, int | str | bool | None]:
        gc.collect()
        return {
            'ok': True,
            'message': 'Phonon runtime memory cleanup completed.',
            'memory_before_mb': None,
            'memory_after_mb': None,
            'memory_total_mb': None,
            'released_mb': None,
        }

    def runtime_info(self) -> dict[str, Any]:
        return {
            'engine': 'phonon',
            'phonon_lang': self.settings.phonon_lang,
            'phonon_quant': self.settings.phonon_quant,
            'phonon_workers': self.settings.phonon_workers,
            'attention_backend': None,
            'attention_backend_note': None,
        }

    def duration_ms(self, text: str) -> int:
        # Rough speaking-rate estimate for queue budgeting; Phonon speaks about 15 characters per second.
        return max(500, 200 + len(text) * 65)

    def pcm_to_wav(self, pcm: bytes, *, sample_rate: int | None = None) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate or self.sample_rate)
            wav_file.writeframes(pcm)
        return buffer.getvalue()

    # -- rendering -------------------------------------------------------------------------------------------------

    def _voice_for(self, item: BatchSynthesisItem) -> str | None:
        voice = (item.request.voice or '').strip()
        return voice if voice in self._voice_names else None

    @contextlib.contextmanager
    def _engine(self):
        engine = self._free.get()
        try:
            yield engine
        finally:
            self._free.put(engine)

    def _synth_sentence(self, item: BatchSynthesisItem) -> bytes:
        kwargs: dict[str, Any] = {'voice': self._voice_for(item)}
        if item.request.seed is not None:
            kwargs['seed'] = item.request.seed
        with self._engine() as engine:
            return _float_to_pcm16(engine.synth(item.text, **kwargs))

    async def render_batch(self, items: list[BatchSynthesisItem], trim_after: bool = True) -> list[BatchSynthesisResult]:
        if not items:
            return []
        if self._executor is None:
            await self.ensure_model(None)
        loop = asyncio.get_running_loop()
        pcms = await asyncio.gather(*(loop.run_in_executor(self._executor, self._synth_sentence, item) for item in items))
        return [
            BatchSynthesisResult(
                job_id=item.job_id,
                sentence_index=item.sentence_index,
                sample_rate=self.sample_rate,
                pcm=pcm,
                duration_ms=int(len(pcm) / 2 / self.sample_rate * 1000),
            )
            for item, pcm in zip(items, pcms)
        ]

    async def stream_batch(
        self,
        items: list[BatchSynthesisItem],
        chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> AsyncIterator[list[BatchSynthesisResult]]:
        """Stream every sentence of the batch as the engines produce chunks.

        Sentences render in parallel, up to `phonon_workers` at once, and their chunks arrive interleaved.
        Each yielded list holds the chunks available since the previous yield. A sentence ends with one
        empty result carrying `sentence_finished=True`. The queue re-orders by sentence index.
        """
        if not items:
            return
        if self._executor is None:
            await self.ensure_model(None)
        loop = asyncio.get_running_loop()
        out: asyncio.Queue[tuple[BatchSynthesisItem, Any]] = asyncio.Queue()
        done = object()
        cancelled = threading.Event()

        def produce(item: BatchSynthesisItem) -> None:
            if cancelled.is_set():
                loop.call_soon_threadsafe(out.put_nowait, (item, done))
                return
            kwargs: dict[str, Any] = {'voice': self._voice_for(item)}
            if item.request.seed is not None:
                kwargs['seed'] = item.request.seed
            try:
                with self._engine() as engine:
                    for chunk in engine.stream(item.text, **kwargs):
                        if cancelled.is_set():
                            break
                        loop.call_soon_threadsafe(out.put_nowait, (item, _float_to_pcm16(chunk)))
            except Exception as exc:  # surfaced to the consumer below
                loop.call_soon_threadsafe(out.put_nowait, (item, exc))
            finally:
                loop.call_soon_threadsafe(out.put_nowait, (item, done))

        for item in items:
            self._executor.submit(produce, item)

        finished = 0
        try:
            while finished < len(items):
                batch = [await out.get()]
                while not out.empty():
                    batch.append(out.get_nowait())
                results: list[BatchSynthesisResult] = []
                for item, payload in batch:
                    if isinstance(payload, Exception):
                        raise payload
                    if payload is done:
                        finished += 1
                        results.append(
                            BatchSynthesisResult(
                                job_id=item.job_id,
                                sentence_index=item.sentence_index,
                                sample_rate=self.sample_rate,
                                pcm=b'',
                                duration_ms=0,
                                sentence_finished=True,
                            )
                        )
                        continue
                    results.append(
                        BatchSynthesisResult(
                            job_id=item.job_id,
                            sentence_index=item.sentence_index,
                            sample_rate=self.sample_rate,
                            pcm=payload,
                            duration_ms=int(len(payload) / 2 / self.sample_rate * 1000),
                        )
                    )
                yield results
        finally:
            cancelled.set()
