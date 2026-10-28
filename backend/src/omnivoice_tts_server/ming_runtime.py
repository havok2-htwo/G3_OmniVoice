"""inclusionAI Ming-3.0-tts 0.5B behind the synthesizer interface the queue uses for OmniVoice.

The engine (``ming/engine.py``) runs one continuous batch on its own GPU thread: rows join at the next LM step and
stream audio chunks the moment they are decoded. This adapter

* resolves the request voice like OmniVoice (saved voice profile or inline ref_audio + ref_text) and caches the
  encoded voice prompt in the engine, keyed by model, voice, audio hash and transcript,
* submits every sentence of a batch as its own engine row and forwards each chunk to the queue at once
  (``stream_batch``), with one ``sentence_finished`` marker per sentence after its last chunk,
* lets the queue run several stream batches at once (``supports_concurrent_batches``), so a request that arrives
  while others stream joins the running engine batch instead of waiting for it.

Output is PCM16 mono at ``ming_output_sample_rate`` (24 kHz by default: the client contract; the engine resamples
44.1 kHz with a stateful streaming resampler per row).
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import gc
import hashlib
import io
import logging
import threading
import time
import wave
from pathlib import Path
from typing import Any, AsyncIterator, Callable

import numpy as np

from .capacity import ming_capacity_profile
from .config import Settings
from .domain.state import InMemoryStore, VoiceProfileRecord
from .runtime_v2 import (
    LANGUAGE_LABEL_TO_CODE,
    BatchSynthesisItem,
    BatchSynthesisResult,
    normalize_runtime_device,
    query_nvidia_smi,
)

logger = logging.getLogger('omnivoice_tts_server.ming_runtime')

# The clone alias the queue serves. It ends in 'Base' so task-type inference, the default voice profile and the
# OpenAI voice routing treat it like OmniVoice-Base.
MING_BASE_ALIAS = 'inclusionAI/Ming-3.0-tts-0.5B-Base'
# Seed of a request that sends none (and of the voice prompt's VAE posterior sample): output is reproducible.
MING_DEFAULT_SEED = 0
MING_PROMPT_SEED = 0
# 'Auto' / unknown languages: the language the text's function words point to, else German.
MING_DEFAULT_LANGUAGE = 'de'
WARMUP_TEXT = 'Dies ist ein kurzer Aufwärmsatz für die Sprachausgabe.'
# store.prompt_cache entry whose absence tells us the router cleared the cache (voice created / deleted, switch).
_CACHE_MARKER = 'ming|voice-prompt-generation'
_DONE = object()


def _float_to_pcm16(audio: np.ndarray) -> bytes:
    clipped = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0)
    return (clipped * 32767.0).astype('<i2').tobytes()


def ming_language(language: str | None, text: str = '') -> str:
    """Ming language id for a request language: explicit ids and UI labels map directly ("Deutsch" -> de);
    'Auto', empty and languages Ming has no name for use the language the text's function words point to when Ming
    knows it, else German."""
    from .ming.prompt import normalize_language_id

    value = (language or '').strip()
    if value and value.lower() != 'auto':
        resolved = normalize_language_id(LANGUAGE_LABEL_TO_CODE.get(value.lower(), value))
        if resolved is not None:
            return resolved
    try:
        from .text_optimizer.text_language import detect_word_language

        detected = detect_word_language(text or '')
    except Exception:  # noqa: BLE001 - detection is a hint only
        detected = None
    resolved = normalize_language_id(detected) if detected else None
    return resolved or MING_DEFAULT_LANGUAGE


def _decode_audio(raw: bytes) -> tuple[np.ndarray, int]:
    import soundfile

    audio, sample_rate = soundfile.read(io.BytesIO(raw), dtype='float32', always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, int(sample_rate)


def _ref_audio_bytes(ref_audio: str) -> bytes:
    """Bytes of an inline reference: a local file path or a (data URI) base64 string."""
    value = ref_audio.strip()
    if value.startswith('data:'):
        value = value.split(',', 1)[1] if ',' in value else ''
    else:
        path = Path(value)
        try:
            if len(value) < 4096 and path.is_file():
                return path.read_bytes()
        except OSError:
            pass
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RuntimeError('ref_audio must be a readable local file path or base64 audio for the Ming engine.') from exc


def _default_engine_factory(**kwargs: Any) -> Any:
    from .ming.engine import MingEngine

    return MingEngine(**kwargs)


class MingSynthesizer:
    # stream_batch exists and serves every task type; the queue streams Ming batches natively.
    supports_native_streaming = True
    # The queue may dispatch further batches while a native stream batch runs: they join the engine's batch.
    supports_concurrent_batches = True
    uses_voice_profiles = True

    def __init__(self, settings: Settings, store: InMemoryStore | None, *,
                 engine_factory: Callable[..., Any] | None = None) -> None:
        self.settings = settings
        self.store = store
        self.model_id = settings.ming_model_id
        self.sample_rate = int(settings.ming_output_sample_rate)
        self.max_slots = int(settings.ming_max_slots)
        self._engine_factory = engine_factory or _default_engine_factory
        self._engine: Any = None
        self._loaded_model_id: str | None = None
        self._load_lock = threading.Lock()
        self._ensure_lock: asyncio.Lock | None = None
        self._ensure_loop: Any = None
        self._warm_ms = 0
        self._voice_keys: set[str] = set()
        self._voice_lock = threading.Lock()
        self._streams_lock = threading.Lock()
        self._job_streams: dict[str, set[Any]] = {}

    # -- capabilities ----------------------------------------------------------------------------------------------

    @property
    def capacity_profile(self) -> dict[str, Any]:
        return ming_capacity_profile(self.max_slots)

    # -- model lifecycle -------------------------------------------------------------------------------------------

    def _engine_kwargs(self) -> dict[str, Any]:
        settings = self.settings
        data_dir = getattr(settings, 'data_dir', None)
        kwargs: dict[str, Any] = {
            'model_dir': Path(settings.ming_model_dir),
            'device': normalize_runtime_device(getattr(settings, 'preferred_device', None)),
            'max_slots': self.max_slots,
            'cuda_graphs': bool(settings.ming_cuda_graphs),
            'output_sample_rate': self.sample_rate,
            'first_chunk_latents': int(settings.ming_first_chunk_latents),
            'max_chunk_latents': int(settings.ming_max_chunk_latents),
        }
        if data_dir:
            # GEMM kernel plans (ming/fast.py) persist next to the server data, not inside the model folder.
            kwargs['plan_cache'] = Path(data_dir) / 'ming_linear_plans.json'
        return kwargs

    def _load_sync(self) -> int:
        with self._load_lock:
            if self._engine is not None:
                return 0
            model_dir = Path(self.settings.ming_model_dir)
            if not (model_dir / 'config.json').exists():
                raise RuntimeError(f'Ming checkpoint not found at {model_dir}. Download {self.model_id} there first.')
            started = time.perf_counter()
            self._reset_process_kernel_choices()
            engine = self._engine_factory(**self._engine_kwargs())
            engine.load()
            self._engine = engine
            self._loaded_model_id = MING_BASE_ALIAS
            with self._voice_lock:
                self._voice_keys.clear()
            # The default voice's prompt is what most requests need first: encode it now, not in the first request.
            # One short render runs the first-call paths (tokenizer, pinned staging, resampler) here too.
            profile = self._warmup_voice_profile()
            try:
                prompt = self._prompt_from_profile(engine, profile, profile.ref_text) if profile is not None else None
                engine.render(WARMUP_TEXT, prompt=prompt, language=MING_DEFAULT_LANGUAGE, seed=MING_DEFAULT_SEED)
            except Exception as exc:  # noqa: BLE001 - only a head start
                logger.warning('ming: load warmup failed (non-critical): %s', exc)
            self._warm_ms = int((time.perf_counter() - started) * 1000)
            stats = engine.stats() if hasattr(engine, 'stats') else {}
            logger.info(
                'ming loaded dir=%s device=%s slots=%s graphs=%s rate=%s load_ms=%s vram_reserved_mb=%s',
                model_dir, self._engine_kwargs()['device'], self.max_slots, stats.get('cuda_graphs'),
                self.sample_rate, self._warm_ms, stats.get('vram_reserved_mb'),
            )
            return self._warm_ms

    @staticmethod
    def _reset_process_kernel_choices() -> None:
        """Undo process-wide kernel choices another engine made, before the CUDA graphs are captured.

        OmniVoice puts cuDNN first in the SDPA backend order when it loads. After a live switch OmniVoice -> Ming the
        graphs would capture cuDNN attention and render (equally valid, but) different audio than a Ming-only
        process. The default order keeps Ming's output the same however the process got here; OmniVoice sets its
        order again on its next load."""
        import sys

        torch = sys.modules.get('torch')
        if torch is None:
            return
        try:
            from .runtime_v2 import apply_sdpa_backend_preference

            apply_sdpa_backend_preference(torch, False)
        except Exception as exc:  # noqa: BLE001
            logger.warning('ming: could not restore the default SDPA backend order: %s', exc)

    def _lock_for_loop(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._ensure_lock is None or self._ensure_loop is not loop:
            self._ensure_lock = asyncio.Lock()
            self._ensure_loop = loop
        return self._ensure_lock

    async def ensure_model(self, requested_model: str | None) -> tuple[str, int]:
        # Any model id is served by the one checkpoint (benchmark / WER services submit OmniVoice ids directly).
        if self._engine is not None:
            return MING_BASE_ALIAS, 0
        async with self._lock_for_loop():
            if self._engine is not None:
                return MING_BASE_ALIAS, 0
            warm_ms = await asyncio.to_thread(self._load_sync)
            return MING_BASE_ALIAS, warm_ms

    def resident_models(self) -> set[str]:
        return {MING_BASE_ALIAS, self.model_id} if self._engine is not None else set()

    async def preload(self, requested_model: str | None = None) -> tuple[str, int]:
        return await self.ensure_model(requested_model)

    async def warmup(self, requested_model: str | None = None, request: Any = None) -> tuple[str, int]:
        """Load (CUDA graphs are captured at load) and render one short clone sentence with the default voice."""
        _, load_ms = await self.ensure_model(requested_model)
        started = time.perf_counter()
        voice = getattr(request, 'voice', None) if request is not None else None
        language = getattr(request, 'language', None) if request is not None else None

        def run() -> None:
            engine = self._engine
            if engine is None:
                return
            profile = self._resolve_voice_profile(voice) or self._warmup_voice_profile()
            if profile is None:
                logger.info('ming warmup without a voice prompt (no saved voice profile with ref_text)')
            prompt = self._prompt_from_profile(engine, profile, profile.ref_text) if profile is not None else None
            engine.render(WARMUP_TEXT, prompt=prompt, language=ming_language(language, WARMUP_TEXT),
                          seed=MING_DEFAULT_SEED)

        try:
            await asyncio.to_thread(run)
        except Exception as exc:  # noqa: BLE001
            logger.warning('ming warmup failed (non-critical): %s', exc)
        return MING_BASE_ALIAS, load_ms + int((time.perf_counter() - started) * 1000)

    def _close_sync(self) -> None:
        with self._load_lock:
            engine, self._engine = self._engine, None
            self._loaded_model_id = None
            with self._voice_lock:
                self._voice_keys.clear()
            if engine is not None:
                try:
                    engine.close()
                finally:
                    del engine
            gc.collect()
            try:
                import sys

                torch = sys.modules.get('torch')
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as exc:  # noqa: BLE001
                logger.warning('ming: CUDA cache release failed: %s', exc)

    async def unload(self) -> tuple[str, int]:
        await asyncio.to_thread(self._close_sync)
        return MING_BASE_ALIAS, 0

    async def reload(self, requested_model: str | None = None) -> tuple[str, int]:
        await self.unload()
        return await self.ensure_model(requested_model)

    def _busy(self) -> bool:
        engine = self._engine
        if engine is None:
            return False
        try:
            stats = engine.stats()
        except Exception:  # noqa: BLE001
            return True
        return bool(stats.get('active_rows') or stats.get('pending_rows'))

    async def free_memory(self) -> dict[str, int | str | bool | None]:
        """Release the engine's CUDA memory (weights, KV slots, graphs) when it is idle; it reloads on demand."""
        device = getattr(self.settings, 'preferred_device', None)
        before = await asyncio.to_thread(query_nvidia_smi, device)
        if self._busy():
            message = 'Ming engine is rendering; CUDA memory stays allocated (retry when idle).'
        else:
            await asyncio.to_thread(self._close_sync)
            message = 'Ming engine closed and its CUDA memory released; it reloads with the next request.'
        after = await asyncio.to_thread(query_nvidia_smi, device)
        before_used = int(before.get('memory_used_mb') or 0)
        after_used = int(after.get('memory_used_mb') or 0)
        return {
            'ok': True,
            'message': message,
            'memory_before_mb': before_used,
            'memory_after_mb': after_used,
            'memory_total_mb': int(after.get('memory_total_mb') or before.get('memory_total_mb') or 0),
            'released_mb': max(0, before_used - after_used),
        }

    def runtime_info(self) -> dict[str, Any]:
        engine = self._engine
        stats: dict[str, Any] = {}
        if engine is not None:
            try:
                stats = dict(engine.stats())
            except Exception as exc:  # noqa: BLE001
                stats = {'error': str(exc)}
        return {
            'engine': 'ming',
            'model_id': self.model_id,
            'model_dir': str(self.settings.ming_model_dir),
            'loaded': engine is not None,
            'max_slots': self.max_slots,
            'cuda_graphs': bool(self.settings.ming_cuda_graphs),
            'output_sample_rate': self.sample_rate,
            'first_chunk_latents': int(self.settings.ming_first_chunk_latents),
            'max_chunk_latents': int(self.settings.ming_max_chunk_latents),
            'engine_stats': stats,
            'attention_backend': 'sdpa' if engine is not None else None,
            'attention_backend_note': 'Ming engine: torch SDPA with CUDA graphs' if stats.get('cuda_graphs') else None,
        }

    def duration_ms(self, text: str) -> int:
        # Ming speaks German at roughly 14 characters per second (same envelope as the OmniVoice estimate).
        return max(500, int(len(text or '') * 70))

    def pcm_to_wav(self, pcm: bytes, *, sample_rate: int | None = None) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate or self.sample_rate)
            wav_file.writeframes(pcm)
        return buffer.getvalue()

    # -- voices ----------------------------------------------------------------------------------------------------

    def _profiles(self) -> list[VoiceProfileRecord]:
        return list(getattr(self.store, 'voice_profiles', {}).values()) if self.store is not None else []

    def _resolve_voice_profile(self, voice_name: str | None) -> VoiceProfileRecord | None:
        if not voice_name:
            return None
        for profile in self._profiles():
            if profile.voice_id == voice_name or profile.name == voice_name:
                return profile
        return None

    def _warmup_voice_profile(self) -> VoiceProfileRecord | None:
        """The configured default voice when it can clone, else the first profile that can (OmniVoice's rule)."""
        usable = [p for p in self._profiles() if p.audio_bytes and (p.ref_text or '').strip()]
        default_voice = (getattr(self.settings, 'default_voice', '') or '').strip()
        if default_voice:
            lowered = default_voice.casefold()
            profile = next((p for p in usable if default_voice in (p.voice_id, p.name)), None) or next(
                (p for p in usable if lowered in (p.voice_id.casefold(), str(p.name or '').casefold())), None
            )
            if profile is not None:
                return profile
        return usable[0] if usable else None

    def _sync_voice_cache(self, engine: Any) -> None:
        """Drop our encoded prompts once the router cleared store.prompt_cache (voice created / deleted)."""
        cache = getattr(self.store, 'prompt_cache', None)
        if cache is None or _CACHE_MARKER in cache:
            return
        with self._voice_lock:
            stale, self._voice_keys = self._voice_keys, set()
        for key in stale:
            try:
                engine.drop_voice(key)
            except Exception:  # noqa: BLE001
                pass
        cache[_CACHE_MARKER] = True

    def _voice_prompt(self, engine: Any, key: str, raw: bytes, ref_text: str) -> Any:
        # Decoding a reference WAV takes about a millisecond; the engine returns its cached prompt for a known key.
        audio, sample_rate = _decode_audio(raw)
        prompt = engine.voice_prompt(key, audio, sample_rate, ref_text, seed=MING_PROMPT_SEED)
        with self._voice_lock:
            self._voice_keys.add(key)
        return prompt

    def _prompt_from_profile(self, engine: Any, profile: VoiceProfileRecord, ref_text: str) -> Any:
        self._sync_voice_cache(engine)
        raw = profile.audio_bytes or b''
        key = '|'.join(['ming', self.model_id, profile.voice_id, hashlib.sha1(raw).hexdigest(), ref_text])
        return self._voice_prompt(engine, key, raw, ref_text)

    def _prompt_for(self, engine: Any, request: Any) -> Any:
        """The engine voice prompt of a request, or None for plain (uncloned) synthesis."""
        profile = self._resolve_voice_profile(getattr(request, 'voice', None))
        if profile is not None and profile.audio_bytes:
            ref_text = (getattr(request, 'ref_text', None) or profile.ref_text or '').strip()
            if not ref_text:
                raise RuntimeError('Ming voice cloning requires ref_text for saved voice profiles.')
            return self._prompt_from_profile(engine, profile, ref_text)
        ref_audio = getattr(request, 'ref_audio', None)
        if ref_audio:
            ref_text = (getattr(request, 'ref_text', None) or '').strip()
            if not ref_text:
                raise RuntimeError('Ming voice cloning requires ref_text with ref_audio.')
            self._sync_voice_cache(engine)
            raw = _ref_audio_bytes(ref_audio)
            key = '|'.join(['ming', self.model_id, 'inline', hashlib.sha1(raw).hexdigest(), ref_text])
            return self._voice_prompt(engine, key, raw, ref_text)
        return None

    # -- rendering -------------------------------------------------------------------------------------------------

    @staticmethod
    def _seed(request: Any) -> int:
        seed = getattr(request, 'seed', None)
        return MING_DEFAULT_SEED if seed is None else int(seed)

    def _submit_items(self, engine: Any, items: list[BatchSynthesisItem]) -> list[Any]:
        """Resolve every voice first (one bad voice fails the batch before any row starts), then submit all rows."""
        prompts = [self._prompt_for(engine, item.request) for item in items]
        streams: list[Any] = []
        try:
            for item, prompt in zip(items, prompts):
                stream = engine.submit(
                    item.text,
                    prompt=prompt,
                    language=ming_language(getattr(item.request, 'language', None), item.text),
                    seed=self._seed(item.request),
                    row_id=f'{item.job_id}:{item.sentence_index}',
                )
                streams.append(stream)
                with self._streams_lock:
                    self._job_streams.setdefault(item.job_id, set()).add(stream)
        except BaseException:
            self._cancel_streams(items[: len(streams)], streams)
            raise
        return streams

    def _cancel_streams(self, items: list[BatchSynthesisItem], streams: list[Any]) -> None:
        for item, stream in zip(items, streams):
            try:
                if not getattr(stream, 'done', False):
                    stream.cancel()
            except Exception:  # noqa: BLE001
                logger.exception('ming: cancelling a row failed')
            with self._streams_lock:
                rows = self._job_streams.get(item.job_id)
                if rows is not None:
                    rows.discard(stream)
                    if not rows:
                        self._job_streams.pop(item.job_id, None)

    def cancel_job(self, job_id: str) -> int:
        """Stop every engine row of a job (cancelled or failed in the queue); their streams end at the next step."""
        with self._streams_lock:
            streams = list(self._job_streams.get(job_id, ()))
        for stream in streams:
            try:
                stream.cancel()
            except Exception:  # noqa: BLE001
                logger.exception('ming: cancelling job %s failed', job_id)
        return len(streams)

    async def render_batch(self, items: list[BatchSynthesisItem], trim_after: bool = True) -> list[BatchSynthesisResult]:
        """All sentences render together in the engine's batch; each comes back whole (silence already trimmed)."""
        if not items:
            return []
        parts: dict[tuple[str, int], list[bytes]] = {(item.job_id, item.sentence_index): [] for item in items}
        sample_rate = self.sample_rate
        async for results in self.stream_batch(items):
            for result in results:
                sample_rate = result.sample_rate
                if result.pcm:
                    parts[(result.job_id, result.sentence_index)].append(result.pcm)
        out = []
        for item in items:
            pcm = b''.join(parts[(item.job_id, item.sentence_index)])
            out.append(
                BatchSynthesisResult(
                    job_id=item.job_id,
                    sentence_index=item.sentence_index,
                    sample_rate=sample_rate,
                    pcm=pcm,
                    duration_ms=int(len(pcm) / 2 / sample_rate * 1000),
                )
            )
        return out

    async def stream_batch(
        self,
        items: list[BatchSynthesisItem],
        chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> AsyncIterator[list[BatchSynthesisResult]]:
        """Stream every sentence of the batch: each engine chunk is forwarded as soon as it is decoded.

        Chunks of different sentences arrive interleaved; each yielded list holds what arrived since the previous
        yield. A sentence ends with one empty result carrying ``sentence_finished=True``; the queue re-orders by
        sentence index. Closing the generator cancels the rows still rendering. ``chunk_size`` / ``overlap`` are
        OmniVoice knobs; the engine's chunk schedule (ming_first_chunk_latents / ming_max_chunk_latents) applies.
        """
        if not items:
            return
        if self._engine is None:
            await self.ensure_model(None)
        engine = self._engine
        if engine is None:
            raise RuntimeError('Ming engine is not loaded')
        loop = asyncio.get_running_loop()
        submit = loop.run_in_executor(None, self._submit_items, engine, items)
        try:
            streams = await asyncio.shield(submit)
        except asyncio.CancelledError:
            # The rows may still get submitted after we stop waiting: cancel them once they exist.
            def cleanup(future: 'asyncio.Future[list[Any]]') -> None:
                if not future.cancelled() and future.exception() is None:
                    self._cancel_streams(items, future.result())

            submit.add_done_callback(cleanup)
            raise
        sample_rate = int(getattr(engine, 'output_sample_rate', self.sample_rate) or self.sample_rate)
        out: asyncio.Queue[tuple[int, Any]] = asyncio.Queue()

        def post(index: int, payload: Any) -> None:
            try:
                loop.call_soon_threadsafe(out.put_nowait, (index, payload))
            except RuntimeError:  # event loop closed (shutdown)
                pass

        def pump(index: int, stream: Any) -> None:
            try:
                for chunk in stream:
                    post(index, _float_to_pcm16(chunk))
            except BaseException as exc:  # noqa: BLE001 - surfaced to the consumer
                post(index, exc)
            finally:
                post(index, _DONE)

        for index, stream in enumerate(streams):
            threading.Thread(target=pump, args=(index, stream), name=f'ming-pump-{index}', daemon=True).start()

        finished = 0
        try:
            while finished < len(items):
                batch = [await out.get()]
                while not out.empty():
                    batch.append(out.get_nowait())
                results: list[BatchSynthesisResult] = []
                error: BaseException | None = None
                for index, payload in batch:
                    item = items[index]
                    if isinstance(payload, BaseException):
                        error = error or payload
                        continue
                    if payload is _DONE:
                        finished += 1
                        results.append(
                            BatchSynthesisResult(
                                job_id=item.job_id,
                                sentence_index=item.sentence_index,
                                sample_rate=sample_rate,
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
                            sample_rate=sample_rate,
                            pcm=payload,
                            duration_ms=int(len(payload) / 2 / sample_rate * 1000),
                        )
                    )
                if error is not None:
                    raise error
                yield results
        finally:
            self._cancel_streams(items, streams)
