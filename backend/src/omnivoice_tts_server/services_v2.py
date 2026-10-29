from __future__ import annotations

import asyncio
import ast
import base64
import hashlib
import json
import logging
import math
import random
import re
import statistics
import threading
import time
import unicodedata
import uuid
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from .config import Settings
from .domain.models import (
    BenchmarkCaseRequest,
    BenchmarkCaseSummary,
    BenchmarkIterationResult,
    BenchmarkRunCreateRequest,
    BenchmarkRunResponse,
    GpuStatsResponse,
    JobMetrics,
    JobStatus,
    SpeechRequest,
    StatGlobal,
    StatRolling,
    StatsResponse,
    TaskType,
    TranscriptionResponse,
    WerBenchmarkCreateRequest,
    WerBenchmarkItemResult,
    WerBenchmarkRunResponse,
    WerBenchmarkSeedSummary,
    WerBenchmarkSummary,
    delivery_controls_conflict,
)
from .domain.state import InMemoryStore, JobRecord, RequestState, utcnow
from .prompt_batch import chunk_pcm16le, split_sentences
from .text_optimizer.omnivoice_tags import render_inline_tags
from .runtime_v2 import BatchSynthesisItem, BatchSynthesisResult, query_nvidia_smi
from .capacity import (
    DEFAULT_PROMPT_SECONDS,
    batch_audio_budget_seconds,
    estimate_item_seconds,
    max_chars_per_chunk,
    prompt_audio_seconds,
)
from .voice_design import normalize_voice_design_instruct

logger = logging.getLogger('omnivoice_tts_server.services')


def spawn_tracked_task(store: InMemoryStore, coro: Any, *, label: str) -> 'asyncio.Task[Any]':
    """Create a background task that is referenced (so it is not GC'd mid-run) and
    whose exceptions are logged instead of silently swallowed. Tasks are tracked on
    the store so the lifespan can cancel them at shutdown."""
    task = asyncio.create_task(coro)
    store.background_tasks.add(task)

    def _on_done(done: 'asyncio.Task[Any]') -> None:
        store.background_tasks.discard(done)
        if done.cancelled():
            return
        exc = done.exception()
        if exc is not None:
            logger.error('background task %s failed: %s', label, exc, exc_info=exc)

    task.add_done_callback(_on_done)
    return task


class QueueSaturatedError(RuntimeError):
    """Raised by submit() only when the queue is full -> maps to HTTP 429.

    Distinguished from validation RuntimeErrors (empty input, bad voice/model)
    so callers don't translate a permanent 400-class failure into a retryable 429.
    """


class RequestTooLargeError(RuntimeError):
    """Raised by submit() when the input text exceeds max_input_chars -> HTTP 413."""


class InvalidRequestFieldsError(RuntimeError):
    """Raised by submit() for request fields that cannot be combined on the active engine -> HTTP 422."""


class EventHub:
    def __init__(self, store: InMemoryStore) -> None:
        self.store = store

    async def publish(self, event: str, payload: dict[str, Any]) -> None:
        message = {'event': event, 'data': payload}
        stale: list[asyncio.Queue[dict[str, Any] | None]] = []
        for queue in list(self.store.event_subscribers):
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                stale.append(queue)
        for queue in stale:
            self.store.event_subscribers.discard(queue)

    async def subscribe(self) -> asyncio.Queue[dict[str, Any] | None]:
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=256)
        self.store.event_subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any] | None]) -> None:
        self.store.event_subscribers.discard(queue)

    @staticmethod
    def encode_sse(event: str, payload: dict[str, Any]) -> str:
        return f"event: {event}\ndata: {json.dumps(payload, default=str)}\n\n"


class QueueService:
    def __init__(self, store: InMemoryStore, synthesizer: Any, events: EventHub, settings: Settings) -> None:
        self.store = store
        self.synthesizer = synthesizer
        self.events = events
        self.settings = settings
        self._background_tasks: set[asyncio.Task] = set()
        # Concurrent native streaming (supports_concurrent_batches): sentences rendering in a running stream batch
        # (until their sentence_finished marker) and the running batches by id.
        self._native_inflight: set[tuple[str, int]] = set()
        self._running_batches: dict[str, dict[str, Any]] = {}

    async def start_worker(self) -> None:
        if self.store.worker_task and not self.store.worker_task.done():
            return
        self.store.worker_stop.clear()
        self.store.worker_task = asyncio.create_task(self._worker_loop())

    def running_batch_count(self) -> int:
        """Stream batches still running as their own tasks (concurrent engines only; 0 otherwise)."""
        return len(self._running_batches)

    async def stop_worker(self) -> None:
        self.store.worker_stop.set()
        async with self.store.job_condition:
            self.store.job_condition.notify_all()
        if self.store.worker_task:
            try:
                await self.store.worker_task
            except Exception:
                pass

    async def submit(
        self,
        request: SpeechRequest,
        *,
        owner_scope: str = 'public',
        api_key_id: str | None = None,
        stream_consumer: str | None = None,
    ) -> JobRecord:
        """Queue a request. stream_consumer names the job queue an HTTP handler will read
        (STREAM_EVENTS / STREAM_CHUNKS); None means nobody streams it, so no stream copies
        of the audio are buffered on the job."""
        text = (request.input or '').strip()
        if not text:
            raise RuntimeError('Missing input text')
        max_input = int(getattr(self.settings, 'max_input_chars', 0) or 0)
        if max_input > 0 and len(text) > max_input:
            raise RequestTooLargeError(
                f'Input text is {len(text)} characters; the limit is {max_input}. '
                'Split it into smaller requests.'
            )
        self._apply_default_voice(request)
        seed_from_profile = request.seed is None
        self._apply_voice_seed(request)
        self._validate_request_voice_model(request)
        self._validate_delivery_controls(request)
        spoken_text, text_optimization = await self._optimize_spoken_text(request, text, owner_scope)
        request = self._drop_upstream_normalize_after_optimization(request, text_optimization)
        # Inline tags ([laughter], [Cough], ...) in the active engine's spelling; tags it does not know are dropped.
        spoken_text = render_inline_tags(spoken_text, self.settings.runtime_backend).strip()
        if not spoken_text:
            raise RuntimeError('Missing input text')

        sentences = split_sentences(
            spoken_text,
            enabled=self.settings.sentence_chunking,
            short_sentence_merge_max_chars=self.settings.short_sentence_merge_max_chars,
            following_sentence_merge_min_chars=self.settings.following_sentence_merge_min_chars,
            max_chars=self.max_chars_per_sentence(),
        )
        if not sentences:
            raise RuntimeError('Missing input text')

        now = utcnow()
        job = JobRecord(
            job_id=f'job_{uuid.uuid4().hex}',
            request=request,
            created_at=now,
            updated_at=now,
            owner_scope=owner_scope,
            api_key_id=api_key_id,
            sentences_total=len(sentences),
            stream_consumer=stream_consumer,
            text_optimization=text_optimization,
            spoken_text=spoken_text if text_optimization is not None else None,
            seed_from_profile=seed_from_profile,
        )
        job.metrics['sentences_total'] = len(sentences)
        state = RequestState(
            job_id=job.job_id,
            group_key=self._group_key_for_request(request),
            sentences=sentences,
            prompt_seconds=self._prompt_seconds_for_request(request),
        )

        async with self.store.job_condition:
            total_outstanding = len(self.store.waiting_requests) + len(self.store.active_request_ids)
            if total_outstanding >= self.store.max_queue_size:
                raise QueueSaturatedError('Queue saturated')
            self.store.jobs[job.job_id] = job
            self.store.request_states[job.job_id] = state
            self.store.waiting_requests.append(job.job_id)
            self._recompute_positions_locked()
            start_event = {
                'type': 'start',
                'job_id': job.job_id,
                'sentence_count': len(sentences),
                'queue_position': job.queue_position,
            }
            if text_optimization is not None:
                start_event['text_optimization'] = {**text_optimization, 'spoken_text': spoken_text}
            job.push_stream_event(start_event)
            self.store.job_condition.notify_all()
            preempt = self._preemption_possible(request)

        await self._publish_state()
        if preempt:
            task = asyncio.create_task(self._preempt_first_sentence(job.job_id))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
        return job

    @staticmethod
    def _drop_upstream_normalize_after_optimization(
        request: SpeechRequest, text_optimization: dict[str, Any] | None
    ) -> SpeechRequest:
        """Skip omnivoice's own normalize_text once our optimizer rewrote a de/en text.

        The spoken text is already verbalized, so a second (and for German weaker,
        integer-only) normalisation pass can only hurt. Text the optimizer bound but left
        unchanged (no rule applied) keeps the flag. Done before the group key is
        computed, so such requests never share a batch with ones that still normalize.
        """
        metadata = request.metadata or {}
        if not metadata.get('normalize_text') or not text_optimization:
            return request
        if text_optimization.get('language') not in {'de', 'en'} or not text_optimization.get('applied_rules'):
            return request
        logger.info(
            'normalize_text ignored: text_optimization already verbalized the %s text',
            text_optimization.get('language'),
        )
        return request.model_copy(
            update={'metadata': {key: value for key, value in metadata.items() if key != 'normalize_text'}}
        )

    async def _optimize_spoken_text(
        self, request: SpeechRequest, text: str, owner_scope: str
    ) -> tuple[str, dict[str, Any] | None]:
        """Apply the request's text optimization mode (text_optimizer package).

        Returns (text to split and speak, info for the job) - info is None when the mode is
        'off', so such requests keep today's sentences byte for byte. Runs in a thread: the
        rule passes are CPU-bound regex work (about 1 s for a 25k-character input).
        """
        from .text_optimizer.modes import resolve_text_optimization_mode

        try:
            mode = resolve_text_optimization_mode(
                request.text_optimization,
                request.metadata,
                getattr(self.settings, 'text_optimization_default', 'off'),
                owner_scope,
            )
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        if mode != 'deterministic':
            return text, None
        from .text_optimizer.service import optimize_request_text

        # Shared with the dry-run preview: raw-text fallback on a rule error or timeout.
        outcome = await asyncio.to_thread(optimize_request_text, text, request.language)
        if not outcome.text:
            raise RuntimeError('Input text is empty after text optimization')
        return outcome.text, outcome.as_dict()

    async def cancel(self, job_id: str) -> JobRecord:
        async with self.store.job_condition:
            job = self.store.jobs[job_id]
            if job.status in {JobStatus.completed, JobStatus.failed, JobStatus.cancelled}:
                raise RuntimeError('Job already finished')
            if job.status == JobStatus.cancelling:
                return job

            job.cancel_requested = True
            job.updated_at = utcnow()

            if job_id in self.store.waiting_requests:
                try:
                    self.store.waiting_requests.remove(job_id)
                except ValueError:
                    pass
                self._mark_cancelled_locked(job, 'Cancelled before execution.')
                self._recompute_positions_locked()
                self.store.job_condition.notify_all()
            else:
                job.status = JobStatus.cancelling
                job.error_message = 'Cancellation requested.'
                if self._uses_concurrent_batches():
                    self._concurrent_cancel_locked(job)
                self.store.job_condition.notify_all()

        await self._publish_state()
        return job

    def max_chars_per_sentence(self) -> int:
        """Sentence split length: the synthesizer's own capacity profile when it has one (Ming), else the limit the
        OmniVoice VRAM budget implies."""
        profile = self._capacity_profile()
        if profile is not None and 'max_chars_per_chunk' in profile:
            return max(0, int(profile.get('max_chars_per_chunk') or 0))
        return max_chars_per_chunk(getattr(self.settings, 'vram_budget_mb', 0))

    async def delete(self, job_id: str) -> None:
        async with self.store.job_condition:
            job = self.store.jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.status not in {JobStatus.completed, JobStatus.failed, JobStatus.cancelled}:
                job.cancel_requested = True
                job.status = JobStatus.cancelling
                job.error_message = 'Cancellation requested.'
                if self._uses_concurrent_batches() and job_id not in self.store.waiting_requests:
                    self._concurrent_cancel_locked(job)
            else:
                self.store.jobs.pop(job_id, None)
                self.store.request_states.pop(job_id, None)
                if job_id in self.store.waiting_requests:
                    self.store.waiting_requests.remove(job_id)
                if job_id in self.store.active_request_ids:
                    self.store.active_request_ids.remove(job_id)
                self._recompute_positions_locked()
            self.store.job_condition.notify_all()

        await self._publish_state()

    async def wait_for_completion(self, job_id: str) -> JobRecord:
        terminal = {JobStatus.completed, JobStatus.failed, JobStatus.cancelled}
        async with self.store.job_condition:
            await self.store.job_condition.wait_for(
                lambda: job_id not in self.store.jobs or self.store.jobs[job_id].status in terminal
            )
            job = self.store.jobs.get(job_id)
        if job is None:
            raise RuntimeError(f'Job {job_id} was removed before it completed.')
        return job

    def get(self, job_id: str) -> JobRecord:
        return self.store.jobs[job_id]

    def queue_snapshot(self) -> dict[str, Any]:
        return {
            'queue_depth': self.store.queue_depth(),
            'active_requests': self.store.active_requests(),
            'active_model': self.store.active_model,
            'worker_state': self.store.worker_state,
        }

    def _mark_loaded_model_locked(self, model_id: str) -> None:
        self.store.active_model = model_id
        self.store.models_loaded.clear()
        self.store.models_loaded.update(self._resident_models() or {model_id})
        self.store.models_loaded.add(model_id)

    def _resident_models(self) -> set[str]:
        """Aliases backed by the currently resident checkpoint (all share one)."""
        resident_fn = getattr(self.synthesizer, 'resident_models', None)
        if callable(resident_fn):
            try:
                return set(resident_fn())
            except Exception:
                return set()
        return set()

    async def _worker_loop(self) -> None:
        if self._uses_concurrent_batches():
            await self._concurrent_worker_loop()
            return
        self.store.worker_state = 'idle'
        await self._publish_state()

        while not self.store.worker_stop.is_set():
            batch_items: list[BatchSynthesisItem] = []
            current_batch: dict[str, Any] | None = None
            involved_job_ids: list[str] = []
            anchor_request: SpeechRequest | None = None

            async with self.store.job_condition:
                await self.store.job_condition.wait_for(
                    lambda: self.store.worker_stop.is_set()
                    or bool(self.store.waiting_requests)
                    or any(
                        self.store.request_states.get(job_id) and self.store.request_states[job_id].has_pending_sentences()
                        for job_id in self.store.active_request_ids
                    )
                )
                if self.store.worker_stop.is_set():
                    break

                self._promote_waiting_locked()
                # Coalescing window: until the batch is full or batch_wait_ms after the oldest new request arrived.
                # A request that already waited through a running batch is past its window and starts at once.
                remaining_s = self._batch_wait_remaining_s_locked()
                while remaining_s > 0:
                    try:
                        await asyncio.wait_for(self.store.job_condition.wait(), timeout=remaining_s)
                    except asyncio.TimeoutError:
                        pass
                    if self.store.worker_stop.is_set():
                        break
                    self._promote_waiting_locked()
                    remaining_s = self._batch_wait_remaining_s_locked()
                if self.store.worker_stop.is_set():
                    break

                batch_plan = self._build_batch_plan_locked()
                if not batch_plan:
                    continue

                batch_items, current_batch, involved_job_ids = self._reserve_batch_locked(batch_plan)
                if not batch_items or current_batch is None:
                    continue

                anchor_request = batch_items[0].request
                self.store.current_batch = current_batch
                self.store.worker_state = 'warming'
                for job_id in involved_job_ids:
                    job = self.store.jobs[job_id]
                    if job.started_at is None:
                        job.started_at = utcnow()
                        job.metrics['queue_wait_ms'] = int((job.started_at - job.created_at).total_seconds() * 1000)
                    if job.status not in {JobStatus.cancelling, JobStatus.cancelled}:
                        target_model = job.request.model or self.store.active_model
                        resident = self._resident_models()
                        needs_warm = bool(target_model) and target_model not in resident
                        job.status = JobStatus.warming if needs_warm else JobStatus.running
                    job.updated_at = utcnow()

            await self._publish_state()

            try:
                model_used, warm_ms = await self.synthesizer.ensure_model(anchor_request.model if anchor_request else None)
                self.store.worker_state = 'running'
                if self._can_use_native_streaming(batch_items):
                    await self._process_native_streaming_batch(
                        batch_items=batch_items,
                        involved_job_ids=involved_job_ids,
                        current_batch=current_batch,
                        model_used=model_used,
                        warm_ms=warm_ms,
                    )
                    await self._publish_state()
                    continue
                trim_idle = getattr(self.synthesizer, 'trim_idle', None)
                if callable(trim_idle):
                    # Trim once the queue runs empty instead of after every batch (8-11 ms per batch measured);
                    # the next batch of a busy period reuses the cached blocks.
                    results = await self.synthesizer.render_batch(batch_items, trim_after=False)
                else:
                    results = await self.synthesizer.render_batch(batch_items)
            except Exception as exc:
                async with self.store.job_condition:
                    self.store.current_batch = None
                    self.store.worker_state = 'idle'
                    for job_id in involved_job_ids:
                        job = self.store.jobs.get(job_id)
                        if job is None:
                            continue
                        if job.cancel_requested:
                            self._mark_cancelled_locked(job, 'Cancelled during synthesis.')
                        else:
                            self._fail_job_locked(job, str(exc))
                    self._recompute_positions_locked()
                    self.store.job_condition.notify_all()
                await self._publish_state()
                continue

            async with self.store.job_condition:
                self.store.current_batch = None
                self.store.worker_state = 'idle'
                self._mark_loaded_model_locked(model_used)
                self.store.recent_batches.append(current_batch)

                self._apply_batch_results_locked(involved_job_ids, batch_items, results, model_used, warm_ms)

                self._recompute_positions_locked()
                self.store.prune_terminal_jobs(
                    retention_days=self.settings.retention_days,
                    max_retained_jobs=self.settings.max_retained_jobs,
                    max_retained_audio_bytes=int(getattr(self.settings, 'max_retained_audio_mb', 0) or 0) * 1024 * 1024,
                )
                self.store.job_condition.notify_all()
                queue_idle = not self.store.waiting_requests and not any(
                    self.store.request_states.get(job_id) and self.store.request_states[job_id].has_pending_sentences()
                    for job_id in self.store.active_request_ids
                )

            await self._publish_state()
            trim_idle = getattr(self.synthesizer, 'trim_idle', None)
            if queue_idle and callable(trim_idle):
                await trim_idle()

        self.store.worker_state = 'stopped'
        await self._publish_state()

    def _apply_batch_results_locked(
        self,
        involved_job_ids: list[str],
        batch_items: list[BatchSynthesisItem],
        results: list[Any],
        model_used: str,
        warm_ms: int,
    ) -> None:
        """Store a finished batch's audio on its jobs, emit what is in order, complete finished jobs.

        Only the sentences of this batch are checked: a job can have another sentence in flight elsewhere (a stream's
        first sentence rendered at a step boundary, see _preempt_first_sentence).
        """
        results_by_key = {(item.job_id, item.sentence_index): item for item in results}
        for job_id in involved_job_ids:
            job = self.store.jobs.get(job_id)
            state = self.store.request_states.get(job_id)
            if job is None or state is None:
                continue

            job.model_used = model_used
            if job.metrics.get('model_warm_ms') is None:
                job.metrics['model_warm_ms'] = warm_ms
            job.metrics['batch_count'] = state.batch_count

            if job.cancel_requested:
                self._mark_cancelled_locked(job, 'Cancelled during synthesis.')
                continue

            for sentence_index in [item.sentence_index for item in batch_items if item.job_id == job_id]:
                result = results_by_key.get((job_id, sentence_index))
                if result is None:
                    self._fail_job_locked(job, 'Batch result was incomplete.')
                    break
                state.ready_sentence_pcm[sentence_index] = result.pcm
                state.sentence_duration_ms[sentence_index] = result.duration_ms
                state.sample_rate = result.sample_rate
                state.inflight_sentence_indices.discard(sentence_index)

            if job.status == JobStatus.failed:
                continue

            self._flush_ready_sentences_locked(job, state)
            if state.is_complete():
                self._complete_job_locked(job, state)

    def _preemption_possible(self, request: SpeechRequest) -> bool:
        return (
            bool(getattr(self.settings, 'stream_first_preempt', False))
            and bool(request.stream)
            and self.store.current_batch is not None
            and callable(getattr(self.synthesizer, 'render_urgent', None))
        )

    async def _preempt_first_sentence(self, job_id: str) -> None:
        """Render a new stream's first sentence at the running batch's next step boundary instead of after it.

        The running batch pauses for one solo generate (~0.15-0.25 s) and continues unchanged (its random state is
        restored). If the batch ends before the next step boundary, the sentence goes back to the queue, where the
        stream-first plan picks it up first.
        """
        async with self.store.job_condition:
            self._promote_waiting_locked()
            job = self.store.jobs.get(job_id)
            state = self.store.request_states.get(job_id)
            if (
                job is None
                or state is None
                or job.cancel_requested
                or job_id not in self.store.active_request_ids
                or state.batch_count != 0
                or not state.pending_sentence_indices
                or state.pending_sentence_indices[0] != 0
                or self.store.current_batch is None
            ):
                return
            target_model = job.request.model or self.store.active_model
            if not target_model or target_model not in self._resident_models():
                return
            items, _, _ = self._reserve_batch_locked([(job_id, 0)])
            if job.started_at is None:
                job.started_at = utcnow()
                job.metrics['queue_wait_ms'] = int((job.started_at - job.created_at).total_seconds() * 1000)
            job.status = JobStatus.running
            job.metrics['first_sentence_preempted'] = True
            job.updated_at = utcnow()

        try:
            results = await self.synthesizer.render_urgent(items)
        except Exception as exc:
            async with self.store.job_condition:
                if job.cancel_requested:
                    self._mark_cancelled_locked(job, 'Cancelled during synthesis.')
                else:
                    self._fail_job_locked(job, str(exc))
                self._recompute_positions_locked()
                self.store.job_condition.notify_all()
            await self._publish_state()
            return

        async with self.store.job_condition:
            if results is None:
                # No step boundary left in the running batch: hand the sentence back to the queue.
                state.inflight_sentence_indices.discard(0)
                state.pending_sentence_indices.appendleft(0)
                state.batch_count = max(0, state.batch_count - 1)
                job.metrics.pop('first_sentence_preempted', None)
            else:
                model_used = getattr(self.synthesizer, '_loaded_model_id', None) or self.store.active_model or ''
                self._apply_batch_results_locked([job_id], items, results, model_used, 0)
            self._recompute_positions_locked()
            self.store.job_condition.notify_all()
        await self._publish_state()

    def _can_use_native_streaming(self, batch_items: list[BatchSynthesisItem]) -> bool:
        if not batch_items or not hasattr(self.synthesizer, 'stream_batch'):
            return False
        if getattr(self.synthesizer, 'supports_native_streaming', False):
            # Engines that stream any task type (Phonon) do not need OmniVoice's Base task.
            return all(item.request.stream for item in batch_items)
        return all(item.request.stream and self._task_type_for_request(item.request) == TaskType.base for item in batch_items)

    async def _process_native_streaming_batch(
        self,
        *,
        batch_items: list[BatchSynthesisItem],
        involved_job_ids: list[str],
        current_batch: dict[str, Any],
        model_used: str,
        warm_ms: int,
    ) -> None:
        durations_by_key: dict[tuple[str, int], int] = {}
        batch_id = str(current_batch.get('batch_id') or '')

        async for results in self.synthesizer.stream_batch(
            batch_items,
            chunk_size=max(2, int(self.settings.stream_chunk_ms / 10)),
            overlap=4,
        ):
            async with self.store.job_condition:
                self._mark_loaded_model_locked(model_used)
                for result in results:
                    job = self.store.jobs.get(result.job_id)
                    state = self.store.request_states.get(result.job_id)
                    if job is None or state is None:
                        continue
                    if job.cancel_requested:
                        self._mark_cancelled_locked(job, 'Cancelled during native streaming.')
                        continue
                    self._apply_native_result_locked(
                        job, state, result, durations_by_key, batch_id=batch_id, model_used=model_used, warm_ms=warm_ms
                    )
                self.store.job_condition.notify_all()

        async with self.store.job_condition:
            self.store.current_batch = None
            self.store.worker_state = 'idle'
            self._mark_loaded_model_locked(model_used)
            self.store.recent_batches.append(current_batch)

            for item in batch_items:
                job = self.store.jobs.get(item.job_id)
                state = self.store.request_states.get(item.job_id)
                if job is None or state is None:
                    continue
                if job.cancel_requested:
                    self._mark_cancelled_locked(job, 'Cancelled during native streaming.')
                    continue
                self._finish_native_item_locked(job, state, item, durations_by_key, batch_id=batch_id)

            self._recompute_positions_locked()
            self.store.job_condition.notify_all()

    def _apply_native_result_locked(
        self,
        job: JobRecord,
        state: RequestState,
        result: BatchSynthesisResult,
        durations_by_key: dict[tuple[str, int], int],
        *,
        batch_id: str,
        model_used: str,
        warm_ms: int,
    ) -> None:
        """One natively streamed chunk (or sentence-finished marker): emit it in sentence order or buffer it."""
        job.model_used = model_used
        job.sample_rate = result.sample_rate
        state.sample_rate = result.sample_rate
        # The serial native path streams stream requests only; concurrent engines also collect non-stream requests
        # this way, and those stay 'running' like a rendered batch.
        job.status = JobStatus.streaming if job.request.stream else JobStatus.running
        job.updated_at = utcnow()
        if job.metrics.get('model_warm_ms') is None:
            job.metrics['model_warm_ms'] = warm_ms
        job.metrics['batch_count'] = state.batch_count

        if result.pcm:
            key = (result.job_id, result.sentence_index)
            durations_by_key[key] = durations_by_key.get(key, 0) + result.duration_ms

        if result.pcm and result.sentence_index == state.next_emit_sentence_index:
            prebuffer_ms = max(0, int(getattr(self.settings, 'stream_prebuffer_ms', 0)))
            already_started = state.chunk_index_by_sentence.get(result.sentence_index, 0) > 0
            if prebuffer_ms > 0 and not already_started:
                state.pending_preview_pcm.setdefault(result.sentence_index, []).append(result.pcm)
                state.pending_preview_duration_ms.setdefault(result.sentence_index, []).append(result.duration_ms)
                buffered_ms = sum(state.pending_preview_duration_ms.get(result.sentence_index, []))
                if buffered_ms >= prebuffer_ms:
                    pending_chunks = state.pending_preview_pcm.pop(result.sentence_index, [])
                    state.pending_preview_duration_ms.pop(result.sentence_index, None)
                    for pending_index, chunk in enumerate(pending_chunks):
                        self._emit_native_chunk_locked(
                            job,
                            state,
                            sentence_index=result.sentence_index,
                            pcm=chunk,
                            sample_rate=result.sample_rate,
                            batch_id=batch_id,
                            final_chunk_of_sentence=result.sentence_finished
                            and pending_index == len(pending_chunks) - 1,
                        )
            else:
                self._emit_native_chunk_locked(
                    job,
                    state,
                    sentence_index=result.sentence_index,
                    pcm=result.pcm,
                    sample_rate=result.sample_rate,
                    batch_id=batch_id,
                    final_chunk_of_sentence=result.sentence_finished,
                )
        elif result.pcm:
            state.pending_preview_pcm.setdefault(result.sentence_index, []).append(result.pcm)
            state.pending_preview_duration_ms.setdefault(result.sentence_index, []).append(result.duration_ms)
        if result.sentence_finished:
            state.completed_streaming_sentence_indices.add(result.sentence_index)
            self._flush_native_streaming_sentences_locked(job, state, batch_id=batch_id)

    def _finish_native_item_locked(
        self,
        job: JobRecord,
        state: RequestState,
        item: BatchSynthesisItem,
        durations_by_key: dict[tuple[str, int], int],
        *,
        batch_id: str,
        concurrent: bool = False,
    ) -> None:
        """A natively streamed sentence is done: flush what is in order, complete the job when nothing is left."""
        key = (item.job_id, item.sentence_index)
        duration_ms = durations_by_key.get(key, 0)
        state.sentence_duration_ms.pop(item.sentence_index, None)
        state.ready_sentence_pcm.pop(item.sentence_index, None)
        state.inflight_sentence_indices.discard(item.sentence_index)
        if duration_ms <= 0:
            self._fail_job_locked(job, 'Native stream produced no audio for a sentence.')
            if concurrent:
                self._cancel_engine_rows(job.job_id)
            return
        state.completed_streaming_sentence_indices.add(item.sentence_index)
        self._flush_native_streaming_sentences_locked(job, state, batch_id=batch_id)

        job.updated_at = utcnow()
        job.metrics['batch_count'] = state.batch_count
        if state.is_complete():
            self._complete_job_locked(job, state)
        elif concurrent:
            # Other sentences of the job may still render in another batch: keep its status.
            return
        elif job.status not in {JobStatus.failed, JobStatus.cancelled, JobStatus.cancelling}:
            job.status = JobStatus.running

    # -- concurrent native streaming (engines with supports_concurrent_batches, e.g. Ming) -------------------------
    #
    # The engine runs one continuous batch: a stream batch dispatched while others are still streaming simply joins
    # it at the next step. So the worker does not await a stream batch here; each runs as its own task and the worker
    # keeps planning, bounded by the synthesizer's capacity_profile (sentences in flight <= engine slots). All job
    # state is still only touched under store.job_condition.

    def _uses_concurrent_batches(self) -> bool:
        synthesizer = self.synthesizer
        return (
            bool(getattr(synthesizer, 'supports_concurrent_batches', False))
            and bool(getattr(synthesizer, 'supports_native_streaming', False))
            and hasattr(synthesizer, 'stream_batch')
        )

    def _capacity_profile(self) -> dict[str, Any] | None:
        profile = getattr(self.synthesizer, 'capacity_profile', None)
        if callable(profile):
            try:
                profile = profile()
            except Exception:
                return None
        return profile if isinstance(profile, dict) else None

    def _concurrent_limits(self) -> tuple[int, int, int]:
        """(sentences in flight, slots kept free for jobs with nothing rendering, items per batch)."""
        profile = self._capacity_profile() or {}
        total = max(1, int(profile.get('max_inflight_items') or self.settings.max_batch_size))
        headroom = max(0, min(total - 1, int(profile.get('newcomer_headroom') or 0)))
        batch_items = max(1, int(profile.get('max_batch_items') or total))
        return total, headroom, batch_items

    def _build_concurrent_plan_locked(self) -> list[tuple[str, int]]:
        """Sentences to dispatch now.

        1. The next sentence of every job with nothing rendering (a new job's first sentence: lowest time to first
           audio). These may use any free slot.
        2. Further sentences, always to the job with the fewest sentences rendering (fair share), but only while
           newcomer_headroom slots stay free: a request arriving during a busy period still starts at the next step.

        Groups do not matter: every engine row carries its own voice, language and seed.
        """
        total, headroom, max_items = self._concurrent_limits()
        inflight = len(self._native_inflight)
        room = min(total - inflight, max_items)
        if room <= 0:
            return []
        eligible = [
            job_id
            for job_id in self.store.active_request_ids
            if job_id in self.store.request_states
            and job_id in self.store.jobs
            and self.store.request_states[job_id].has_pending_sentences()
            and not self.store.jobs[job_id].cancel_requested
        ]
        if not eligible:
            return []
        load: dict[str, int] = {job_id: 0 for job_id in eligible}
        for job_id, _ in self._native_inflight:
            if job_id in load:
                load[job_id] += 1
        plan: list[tuple[str, int]] = []
        planned: dict[str, int] = {}

        def take(job_id: str) -> None:
            offset = planned.get(job_id, 0)
            plan.append((job_id, self.store.request_states[job_id].pending_sentence_indices[offset]))
            planned[job_id] = offset + 1
            load[job_id] += 1

        for job_id in eligible:
            if len(plan) >= room:
                return plan
            if load[job_id] == 0:
                take(job_id)
        extra_room = min(room, total - headroom - inflight)
        while len(plan) < extra_room:
            candidates = [
                job_id
                for job_id in eligible
                if planned.get(job_id, 0) < len(self.store.request_states[job_id].pending_sentence_indices)
            ]
            if not candidates:
                break
            take(min(candidates, key=lambda job_id: load[job_id]))
        return plan

    def _concurrent_dispatch_ready_locked(self) -> bool:
        if self.store.waiting_requests and len(self.store.active_request_ids) < self.settings.max_parallel_requests:
            return True
        return bool(self._build_concurrent_plan_locked())

    def _cancel_engine_rows(self, job_id: str) -> None:
        cancel_job = getattr(self.synthesizer, 'cancel_job', None)
        if callable(cancel_job):
            try:
                cancel_job(job_id)
            except Exception as exc:
                logger.warning('cancelling engine rows of %s failed: %s', job_id, exc)

    def _concurrent_cancel_locked(self, job: JobRecord) -> None:
        """A cancel in concurrent mode: stop the job's engine rows; with none rendering, the job ends now (no batch
        end would ever look at it again)."""
        state = self.store.request_states.get(job.job_id)
        if state is None or not state.inflight_sentence_indices:
            self._mark_cancelled_locked(job, 'Cancelled during synthesis.')
        self._cancel_engine_rows(job.job_id)

    def _refresh_running_batches_locked(self) -> None:
        running = list(self._running_batches.values())
        self.store.current_batch = running[-1] if running else None
        self.store.worker_state = 'running' if running else 'idle'

    async def _concurrent_worker_loop(self) -> None:
        self.store.worker_state = 'idle'
        await self._publish_state()

        while not self.store.worker_stop.is_set():
            async with self.store.job_condition:
                await self.store.job_condition.wait_for(
                    lambda: self.store.worker_stop.is_set() or self._concurrent_dispatch_ready_locked()
                )
                if self.store.worker_stop.is_set():
                    break
                self._promote_waiting_locked()
                remaining_s = self._batch_wait_remaining_s_locked()
                while remaining_s > 0:
                    try:
                        await asyncio.wait_for(self.store.job_condition.wait(), timeout=remaining_s)
                    except asyncio.TimeoutError:
                        pass
                    if self.store.worker_stop.is_set():
                        break
                    self._promote_waiting_locked()
                    remaining_s = self._batch_wait_remaining_s_locked()
                if self.store.worker_stop.is_set():
                    break

                batch_plan = self._build_concurrent_plan_locked()
                if not batch_plan:
                    continue
                batch_items, current_batch, involved_job_ids = self._reserve_batch_locked(batch_plan)
                if not batch_items or current_batch is None:
                    continue
                for item in batch_items:
                    self._native_inflight.add((item.job_id, item.sentence_index))
                self._running_batches[current_batch['batch_id']] = current_batch
                self.store.current_batch = current_batch
                resident = self._resident_models()
                for job_id in involved_job_ids:
                    job = self.store.jobs[job_id]
                    if job.started_at is None:
                        job.started_at = utcnow()
                        job.metrics['queue_wait_ms'] = int((job.started_at - job.created_at).total_seconds() * 1000)
                    if job.status not in {JobStatus.cancelling, JobStatus.cancelled, JobStatus.streaming}:
                        target_model = job.request.model or self.store.active_model
                        needs_warm = bool(target_model) and target_model not in resident
                        job.status = JobStatus.warming if needs_warm else JobStatus.running
                    job.updated_at = utcnow()
                self.store.worker_state = 'running' if resident else 'warming'
                anchor_model = batch_items[0].request.model

            await self._publish_state()
            try:
                model_used, warm_ms = await self.synthesizer.ensure_model(anchor_model)
            except Exception as exc:
                await self._end_concurrent_batch(batch_items, involved_job_ids, current_batch, '', {}, set(), exc)
                continue
            spawn_tracked_task(
                self.store,
                self._run_concurrent_batch(
                    batch_items=batch_items,
                    involved_job_ids=involved_job_ids,
                    current_batch=current_batch,
                    model_used=model_used,
                    warm_ms=warm_ms,
                ),
                label=f'stream batch {current_batch["batch_id"]}',
            )

        self.store.worker_state = 'stopped'
        await self._publish_state()

    async def _run_concurrent_batch(
        self,
        *,
        batch_items: list[BatchSynthesisItem],
        involved_job_ids: list[str],
        current_batch: dict[str, Any],
        model_used: str,
        warm_ms: int,
    ) -> None:
        durations_by_key: dict[tuple[str, int], int] = {}
        finished: set[tuple[str, int]] = set()
        settled: set[tuple[str, int]] = set()
        items_by_key = {(item.job_id, item.sentence_index): item for item in batch_items}
        batch_id = str(current_batch.get('batch_id') or '')
        error: BaseException | None = None
        try:
            async for results in self.synthesizer.stream_batch(
                batch_items,
                chunk_size=max(2, int(self.settings.stream_chunk_ms / 10)),
                overlap=4,
            ):
                async with self.store.job_condition:
                    self._mark_loaded_model_locked(model_used)
                    settled_now = False
                    for result in results:
                        key = (result.job_id, result.sentence_index)
                        if result.sentence_finished:
                            # The engine slot is free again: the worker may dispatch the next sentence.
                            finished.add(key)
                            self._native_inflight.discard(key)
                        job = self.store.jobs.get(result.job_id)
                        state = self.store.request_states.get(result.job_id)
                        if job is None or state is None:
                            continue
                        if job.cancel_requested:
                            self._mark_cancelled_locked(job, 'Cancelled during native streaming.')
                            self._cancel_engine_rows(job.job_id)
                            continue
                        item = items_by_key.get(key)
                        item_error = getattr(result, 'error', None)
                        if item_error:
                            # This sentence failed on its own (e.g. an undecodable voice): only its job fails.
                            settled.add(key)
                            self._fail_job_locked(job, str(item_error))
                            self._cancel_engine_rows(job.job_id)
                            settled_now = True
                            continue
                        self._apply_native_result_locked(
                            job, state, result, durations_by_key, batch_id=batch_id, model_used=model_used,
                            warm_ms=warm_ms,
                        )
                        if result.sentence_finished and item is not None and key not in settled:
                            # Settle the sentence now (complete its job when nothing else is left) instead of when
                            # the slowest row of this batch -- possibly another job's -- has finished.
                            settled.add(key)
                            self._finish_native_item_locked(
                                job, state, item, durations_by_key, batch_id=batch_id, concurrent=True
                            )
                            settled_now = True
                    if settled_now:
                        self._recompute_positions_locked()
                    self.store.job_condition.notify_all()
        except asyncio.CancelledError:
            error = RuntimeError('Synthesis was interrupted (server shutdown).')
            raise
        except Exception as exc:
            logger.warning('native stream batch %s failed: %s', batch_id, exc)
            error = exc
        finally:
            await self._end_concurrent_batch(
                batch_items, involved_job_ids, current_batch, model_used, durations_by_key, finished, error,
                settled=settled,
            )

    async def _end_concurrent_batch(
        self,
        batch_items: list[BatchSynthesisItem],
        involved_job_ids: list[str],
        current_batch: dict[str, Any],
        model_used: str,
        durations_by_key: dict[tuple[str, int], int],
        finished: set[tuple[str, int]],
        error: BaseException | None,
        *,
        settled: set[tuple[str, int]] | None = None,
    ) -> None:
        """Release the batch's slots and settle its remaining sentences (``settled`` ones were already handled when
        they finished): finished ones complete their jobs when nothing else is left; with an error, every job with an
        unfinished sentence in this batch fails (cancelled ones cancel)."""
        batch_id = str(current_batch.get('batch_id') or '')
        async with self.store.job_condition:
            for item in batch_items:
                self._native_inflight.discard((item.job_id, item.sentence_index))
            self._running_batches.pop(batch_id, None)
            self._refresh_running_batches_locked()
            if model_used:
                self._mark_loaded_model_locked(model_used)
            self.store.recent_batches.append(current_batch)

            for item in batch_items:
                if settled and (item.job_id, item.sentence_index) in settled:
                    continue
                job = self.store.jobs.get(item.job_id)
                state = self.store.request_states.get(item.job_id)
                if job is None or state is None:
                    continue
                if job.cancel_requested:
                    self._mark_cancelled_locked(job, 'Cancelled during native streaming.')
                    self._cancel_engine_rows(job.job_id)
                    continue
                if error is not None and (item.job_id, item.sentence_index) not in finished:
                    self._fail_job_locked(job, str(error) or type(error).__name__)
                    self._cancel_engine_rows(job.job_id)
                    continue
                self._finish_native_item_locked(job, state, item, durations_by_key, batch_id=batch_id, concurrent=True)

            self._recompute_positions_locked()
            self.store.prune_terminal_jobs(
                retention_days=self.settings.retention_days,
                max_retained_jobs=self.settings.max_retained_jobs,
                max_retained_audio_bytes=int(getattr(self.settings, 'max_retained_audio_mb', 0) or 0) * 1024 * 1024,
            )
            self.store.job_condition.notify_all()
        try:
            await self._publish_state()
        except Exception:
            pass

    def _batch_wait_remaining_s_locked(self) -> float:
        """Seconds the worker should still wait for more requests before planning a batch (0 = plan now).

        The window opens when the oldest new request (no batch yet) arrived, not when the worker looks, so a request
        that arrived while a batch was running does not wait batch_wait_ms a second time. It closes early once the
        plan is full.
        """
        wait_ms = self.settings.batch_wait_ms
        if wait_ms <= 0:
            return 0.0
        if self._uses_concurrent_batches():
            # Rows join the engine's running continuous batch at the next step: waiting only adds latency.
            return 0.0
        if len(self.store.active_request_ids) >= self.settings.max_parallel_requests and not self.store.waiting_requests:
            return 0.0
        oldest_new = None
        for job_id in list(self.store.active_request_ids) + list(self.store.waiting_requests):
            state = self.store.request_states.get(job_id)
            job = self.store.jobs.get(job_id)
            if state is None or job is None or job.cancel_requested or state.batch_count != 0:
                continue
            if oldest_new is None or job.created_at < oldest_new:
                oldest_new = job.created_at
        if oldest_new is None:
            return 0.0
        remaining_s = wait_ms / 1000.0 - (utcnow() - oldest_new).total_seconds()
        if remaining_s <= 0:
            return 0.0
        if not 0 < len(self._build_batch_plan_locked()) < self.settings.max_batch_size:
            return 0.0
        return remaining_s

    def _emit_native_chunk_locked(
        self,
        job: JobRecord,
        state: RequestState,
        *,
        sentence_index: int,
        pcm: bytes,
        sample_rate: int,
        batch_id: str,
        final_chunk_of_sentence: bool = False,
    ) -> None:
        if not pcm:
            return
        job.sample_rate = sample_rate
        state.sample_rate = sample_rate
        job.status = JobStatus.streaming if job.request.stream else JobStatus.running
        job.updated_at = utcnow()
        if job.first_audio_at is None:
            job.first_audio_at = utcnow()
            if job.started_at:
                job.metrics['ttfa_ms'] = int((job.first_audio_at - job.started_at).total_seconds() * 1000)

        job.pcm_parts.append(pcm)
        chunk_index = state.chunk_index_by_sentence.get(sentence_index, 0)
        state.chunk_index_by_sentence[sentence_index] = chunk_index + 1
        emitted_samples = state.emitted_samples_by_sentence.get(sentence_index, 0) + len(pcm) // 2
        state.emitted_samples_by_sentence[sentence_index] = emitted_samples
        state.emitted_audio_ms += int(len(pcm) / 2 / max(sample_rate, 1) * 1000)
        progress_step = min(sentence_index + 1, len(state.sentences))
        job.push_stream_chunk(pcm)
        if not job.wants_stream_events:
            return
        job.stream_events.put_nowait(
            {
                'type': 'chunk',
                'job_id': job.job_id,
                'sentence_index': sentence_index,
                'chunk_index': chunk_index,
                'sample_rate': sample_rate,
                'pcm16_b64': base64.b64encode(pcm).decode('ascii'),
                'emitted_audio_ms': state.emitted_audio_ms,
                'preview': True,
                'final_chunk_of_sentence': final_chunk_of_sentence,
                'progress_step': progress_step,
                'native_stream': True,
                'batch_id': batch_id,
            }
        )

    def _flush_native_streaming_sentences_locked(self, job: JobRecord, state: RequestState, *, batch_id: str) -> None:
        # Drain the next sentence's buffered chunks even while it is still streaming. Otherwise its later chunks
        # take the direct path and reach the client ahead of the buffered ones.
        while True:
            sentence_index = state.next_emit_sentence_index
            pending_chunks = state.pending_preview_pcm.pop(sentence_index, [])
            state.pending_preview_duration_ms.pop(sentence_index, None)
            finished = sentence_index in state.completed_streaming_sentence_indices
            for chunk_index, chunk in enumerate(pending_chunks):
                self._emit_native_chunk_locked(
                    job,
                    state,
                    sentence_index=sentence_index,
                    pcm=chunk,
                    sample_rate=state.sample_rate,
                    batch_id=batch_id,
                    final_chunk_of_sentence=finished and chunk_index == len(pending_chunks) - 1,
                )
            if not finished:
                break
            state.completed_streaming_sentence_indices.discard(sentence_index)
            state.next_emit_sentence_index += 1
            job.metrics['sentences_rendered'] = state.next_emit_sentence_index

    def _promote_waiting_locked(self) -> None:
        while self.store.waiting_requests and len(self.store.active_request_ids) < self.settings.max_parallel_requests:
            job_id = self.store.waiting_requests.popleft()
            if job_id not in self.store.jobs or job_id not in self.store.request_states:
                continue
            job = self.store.jobs[job_id]
            if job.cancel_requested:
                self._mark_cancelled_locked(job, 'Cancelled before execution.')
                continue
            self.store.active_request_ids.append(job_id)
            job.queue_position = 0
            job.eta_ms = 0

    def _build_batch_plan_locked(self) -> list[tuple[str, int]]:
        anchor_job_id = next(
            (
                job_id
                for job_id in self.store.active_request_ids
                if job_id in self.store.request_states
                and self.store.request_states[job_id].has_pending_sentences()
                and not self.store.jobs[job_id].cancel_requested
            ),
            None,
        )
        if anchor_job_id is None:
            return []

        anchor_group = self.store.request_states[anchor_job_id].group_key
        eligible = [
            job_id
            for job_id in self.store.active_request_ids
            if job_id in self.store.request_states
            and self.store.request_states[job_id].group_key == anchor_group
            and self.store.request_states[job_id].has_pending_sentences()
            and not self.store.jobs[job_id].cancel_requested
        ]
        if not eligible:
            return []

        # VRAM-aware budget: cap the batch by estimated audio seconds (not just count),
        # so a batch of long sentences uses no more VRAM than one of short ones. Each item
        # counts its generated speech plus the clone prompt it carries. 0 = off.
        audio_budget = batch_audio_budget_seconds(getattr(self.settings, 'vram_budget_mb', 0))
        first_audio_plan = self._stream_first_audio_plan_locked(eligible, audio_budget)
        if first_audio_plan:
            return first_audio_plan

        plan: list[tuple[str, int]] = []
        planned_audio_s = 0.0
        per_request_offsets = {job_id: 0 for job_id in eligible}
        made_progress = True
        while len(plan) < self.settings.max_batch_size and made_progress:
            made_progress = False
            for job_id in eligible:
                state = self.store.request_states[job_id]
                offset = per_request_offsets[job_id]
                if offset >= len(state.pending_sentence_indices):
                    continue
                sentence_index = state.pending_sentence_indices[offset]
                if audio_budget > 0:
                    # Always include at least one sentence; stop before exceeding budget.
                    item_s = estimate_item_seconds(state.sentences[sentence_index], state.prompt_seconds)
                    if plan and planned_audio_s + item_s > audio_budget:
                        return plan
                    planned_audio_s += item_s
                plan.append((job_id, sentence_index))
                per_request_offsets[job_id] = offset + 1
                made_progress = True
                if len(plan) >= self.settings.max_batch_size:
                    break
        return plan

    def _stream_first_audio_plan_locked(
        self, eligible_job_ids: list[str], audio_budget: float = 0.0
    ) -> list[tuple[str, int]]:
        plan: list[tuple[str, int]] = []
        planned_audio_s = 0.0
        for job_id in eligible_job_ids:
            job = self.store.jobs[job_id]
            state = self.store.request_states[job_id]
            if not job.request.stream or job.first_audio_at is not None or state.next_emit_sentence_index != 0:
                continue
            if not state.pending_sentence_indices:
                continue
            sentence_index = state.pending_sentence_indices[0]
            if audio_budget > 0:
                # Same VRAM budget as the main plan; the rest get their first audio next batch.
                item_s = estimate_item_seconds(state.sentences[sentence_index], state.prompt_seconds)
                if plan and planned_audio_s + item_s > audio_budget:
                    break
                planned_audio_s += item_s
            plan.append((job_id, sentence_index))
            if len(plan) >= self.settings.max_batch_size:
                break
        return plan

    def _reserve_batch_locked(
        self,
        batch_plan: list[tuple[str, int]],
    ) -> tuple[list[BatchSynthesisItem], dict[str, Any] | None, list[str]]:
        if not batch_plan:
            return [], None, []

        batch_id = uuid.uuid4().hex[:8]
        items: list[BatchSynthesisItem] = []
        involved_job_ids: list[str] = []
        sentence_indices: list[int] = []
        request_ids: list[str] = []
        task_type = None
        voice = None
        language = None
        model_id = None

        incremented_batch_count: set[str] = set()
        for job_id, expected_sentence_index in batch_plan:
            state = self.store.request_states[job_id]
            actual = state.pending_sentence_indices.popleft()
            if actual != expected_sentence_index:
                raise RuntimeError(
                    f'Sentence queue order drifted for request {job_id}: expected {expected_sentence_index}, got {actual}.'
                )
            state.inflight_sentence_indices.add(actual)
            if job_id not in incremented_batch_count:
                state.batch_count += 1
                incremented_batch_count.add(job_id)
            job = self.store.jobs[job_id]
            items.append(
                BatchSynthesisItem(
                    job_id=job_id,
                    sentence_index=actual,
                    request=job.request.model_copy(update={'input': state.sentences[actual]}),
                    text=state.sentences[actual],
                )
            )
            if job_id not in involved_job_ids:
                involved_job_ids.append(job_id)
            request_ids.append(job_id)
            sentence_indices.append(actual)
            planned_batch_size = len(batch_plan)
            job.metrics['last_batch_size'] = planned_batch_size
            job.metrics['max_batch_size_seen'] = max(int(job.metrics.get('max_batch_size_seen') or 0), planned_batch_size)
            model_id = job.request.model or self.store.active_model or self.settings.active_model
            task_type = job.request.task_type or self._task_type_for_request(job.request)
            voice = job.request.voice
            language = job.request.language or 'Auto'
            job.push_stream_event(
                {
                    'type': 'batch',
                    'job_id': job_id,
                    'batch_id': batch_id,
                    'sentence_index': actual,
                    'batch_size': planned_batch_size,
                }
            )

        for job_id in involved_job_ids:
            if job_id in self.store.active_request_ids:
                self.store.active_request_ids.remove(job_id)
                self.store.active_request_ids.append(job_id)

        current_batch = {
            'batch_id': batch_id,
            'model_id': model_id or '',
            'task_type': task_type.value if isinstance(task_type, TaskType) else str(task_type),
            'voice': voice,
            'language': language,
            'size': len(items),
            'started_at': utcnow(),
            'request_ids': request_ids,
            'unique_request_count': len(set(request_ids)),
            'sentence_indices': sentence_indices,
        }
        return items, current_batch, involved_job_ids

    def _flush_ready_sentences_locked(self, job: JobRecord, state: RequestState) -> None:
        while state.next_emit_sentence_index in state.ready_sentence_pcm:
            sentence_index = state.next_emit_sentence_index
            pcm = state.ready_sentence_pcm.pop(sentence_index)
            duration_ms = state.sentence_duration_ms.pop(sentence_index, 0)
            job.sample_rate = state.sample_rate
            job.pcm_parts.append(pcm)
            previous_audio_ms = state.emitted_audio_ms
            state.next_emit_sentence_index += 1
            state.emitted_audio_ms += duration_ms
            job.updated_at = utcnow()
            job.metrics['sentences_rendered'] = state.next_emit_sentence_index

            if job.first_audio_at is None:
                job.first_audio_at = utcnow()
                if job.started_at:
                    job.metrics['ttfa_ms'] = int((job.first_audio_at - job.started_at).total_seconds() * 1000)

            if job.request.stream:
                job.status = JobStatus.streaming
                if job.stream_consumer is None:
                    continue
                emitted_samples = 0
                for chunk_index, chunk in enumerate(
                    chunks := list(chunk_pcm16le(pcm, sample_rate=job.sample_rate, chunk_ms=self.settings.stream_chunk_ms))
                ):
                    emitted_samples += len(chunk) // 2
                    job.push_stream_chunk(chunk)
                    if not job.wants_stream_events:
                        continue
                    job.stream_events.put_nowait(
                        {
                            'type': 'chunk',
                            'job_id': job.job_id,
                            'sentence_index': sentence_index,
                            'chunk_index': chunk_index,
                            'sample_rate': job.sample_rate,
                            'pcm16_b64': base64.b64encode(chunk).decode('ascii'),
                            'emitted_audio_ms': previous_audio_ms + int(emitted_samples / max(job.sample_rate, 1) * 1000),
                            'preview': False,
                            'final_chunk_of_sentence': chunk_index == len(chunks) - 1,
                            'progress_step': sentence_index + 1,
                            'native_stream': False,
                        }
                    )
            else:
                job.status = JobStatus.running

    def _complete_job_locked(self, job: JobRecord, state: RequestState) -> None:
        combined_pcm = b''.join(job.pcm_parts)
        # final_audio is the only copy the job keeps; the parts and the joined buffer
        # would otherwise stay referenced (2-3x the audio) for the whole retention.
        job.pcm_parts = []
        job.final_audio = self.synthesizer.pcm_to_wav(combined_pcm, sample_rate=job.sample_rate)
        del combined_pcm
        job.content_type = 'audio/wav'
        job.completed_at = utcnow()
        job.updated_at = job.completed_at
        job.status = JobStatus.completed
        job.error_message = None
        job.metrics['audio_duration_ms'] = state.emitted_audio_ms
        if job.started_at:
            job.metrics['job_wall_ms'] = int((job.completed_at - job.started_at).total_seconds() * 1000)
        job.metrics['output_bytes'] = len(job.final_audio or b'')
        job.metrics['batch_count'] = state.batch_count
        wall_ms = max(int(job.metrics.get('job_wall_ms') or 1), 1)
        duration_ms = int(job.metrics.get('audio_duration_ms') or 0)
        job.metrics['realtime_x'] = round(duration_ms / wall_ms, 3) if duration_ms else 0.0

        self.store.total_jobs_completed += 1
        self.store.total_audio_seconds += duration_ms / 1000.0
        if job.api_key_id:
            self.store.record_api_key_usage(job.api_key_id, duration_ms / 1000.0)
        self.store.completed_job_metrics.append(job.metrics.copy())
        self.store.audio_output_events.append((time.monotonic(), duration_ms / 1000.0))
        if job.metrics.get('ttfa_ms') is not None:
            self.store.ttfa_events.append((time.monotonic(), float(job.metrics['ttfa_ms'])))

        self.store.request_states.pop(job.job_id, None)
        if job.job_id in self.store.active_request_ids:
            self.store.active_request_ids.remove(job.job_id)

        if job.request.stream:
            job.push_stream_event(
                {
                    'type': 'done',
                    'result': {
                        'job_id': job.job_id,
                        'status': job.status.value,
                        'sample_rate': job.sample_rate,
                        'metrics': job.metrics,
                    },
                }
            )
            job.end_stream()

    def _fail_job_locked(self, job: JobRecord, message: str) -> None:
        now = utcnow()
        job.status = JobStatus.failed
        job.error_message = message
        job.completed_at = now
        job.updated_at = now
        job.final_audio = None
        job.content_type = None
        job.pcm_parts = []
        self.store.request_states.pop(job.job_id, None)
        if job.job_id in self.store.active_request_ids:
            self.store.active_request_ids.remove(job.job_id)
        if job.job_id in self.store.waiting_requests:
            self.store.waiting_requests.remove(job.job_id)
        if job.request.stream:
            job.push_stream_event({'type': 'error', 'message': message})
            job.end_stream()

    def _mark_cancelled_locked(self, job: JobRecord, message: str) -> None:
        now = utcnow()
        job.cancel_requested = True
        job.status = JobStatus.cancelled
        job.completed_at = now
        job.updated_at = now
        job.queue_position = 0
        job.eta_ms = 0
        job.error_message = message
        job.pcm_parts = []
        self.store.request_states.pop(job.job_id, None)
        if job.job_id in self.store.active_request_ids:
            self.store.active_request_ids.remove(job.job_id)
        if job.job_id in self.store.waiting_requests:
            self.store.waiting_requests.remove(job.job_id)
        if job.request.stream:
            job.push_stream_event({'type': 'error', 'message': message})
            job.end_stream()

    def _recompute_positions_locked(self) -> None:
        for index, job_id in enumerate(self.store.waiting_requests, start=1):
            job = self.store.jobs[job_id]
            job.queue_position = index
            job.eta_ms = self.store.estimate_eta_ms(index, len(job.request.input or ''))
            job.updated_at = utcnow()

    def _task_type_for_request(self, request: SpeechRequest) -> TaskType:
        if request.task_type is not None:
            return request.task_type
        model_id = request.model or self.store.active_model or self.settings.active_model
        if model_id.endswith('VoiceDesign'):
            return TaskType.voice_design
        if model_id.endswith('Base'):
            return TaskType.base
        return TaskType.custom_voice

    def _voice_profile_for_request(self, request: SpeechRequest):
        voice = request.voice
        if not voice:
            return None
        for profile in self.store.voice_profiles.values():
            if profile.voice_id == voice or profile.name == voice:
                return profile
        return None

    def _prompt_seconds_for_request(self, request: SpeechRequest) -> float:
        """Reference audio each batch item of this request carries (Base cloning only)."""
        if self._task_type_for_request(request) != TaskType.base:
            return 0.0
        profile = self._voice_profile_for_request(request)
        if profile is not None and profile.audio_bytes:
            return prompt_audio_seconds(profile.audio_bytes)
        if request.ref_audio:
            return prompt_audio_seconds(request.ref_audio)
        return DEFAULT_PROMPT_SECONDS

    def _apply_default_voice(self, request: SpeechRequest) -> None:
        # Base clones a voice, so a request without one (e.g. an OpenAI stock voice that
        # resolved to None) uses the configured default voice profile when it exists.
        if request.ref_audio or self._task_type_for_request(request) != TaskType.base:
            return
        if request.voice:
            # A built-in voice has nothing to clone: render it as AutoVoice on the shared
            # checkpoint. Pinning task_type (not model) keeps the active alias unchanged.
            built_in = {name.casefold() for name in self.settings.built_in_voices}
            if (
                request.task_type is None
                and not request.model
                and request.voice.casefold() in built_in
                and self._voice_profile_for_request(request) is None
            ):
                request.task_type = TaskType.custom_voice
            return
        default_voice = (self.settings.default_voice or '').strip()
        if not default_voice:
            return
        profiles = list(self.store.voice_profiles.values())
        lowered = default_voice.casefold()
        profile = next((p for p in profiles if default_voice in (p.voice_id, p.name)), None) or next(
            (p for p in profiles if lowered in (p.voice_id.casefold(), str(p.name or '').casefold())), None
        )
        if profile is not None:
            request.voice = profile.voice_id

    def _apply_voice_seed(self, request: SpeechRequest) -> None:
        # A voice profile may carry its benchmarked seed; it applies only when the request sends no seed. Seeds
        # are per engine: the active one (read live, a switch rewrites settings.runtime_backend) picks the seed.
        if request.seed is not None:
            return
        profile = self._voice_profile_for_request(request)
        seed = profile.seed_for_engine(self.settings.runtime_backend) if profile is not None else None
        if seed is not None:
            request.seed = int(seed)

    def reapply_voice_seed(self, job: JobRecord) -> None:
        """Resolve the profile seed again for the engine active now (a held job requeued after an engine switch).
        A seed the client sent stays as it is."""
        if not job.seed_from_profile:
            return
        job.request.seed = None
        self._apply_voice_seed(job.request)

    def _validate_delivery_controls(self, request: SpeechRequest) -> None:
        # On Ming, instructions is the free-form instruction: it cannot be sent together with structured controls.
        # Other engines ignore the delivery controls.
        if self.settings.runtime_backend != 'ming':
            return
        conflict = delivery_controls_conflict(request)
        if conflict:
            raise InvalidRequestFieldsError(conflict)

    def _validate_request_voice_model(self, request: SpeechRequest) -> None:
        task_type = self._task_type_for_request(request)
        voice_profile = self._voice_profile_for_request(request)
        if task_type == TaskType.custom_voice and voice_profile is not None:
            raise RuntimeError(
                f'Voice profile "{voice_profile.name}" requires the OmniVoice Base alias.'
            )
        if task_type == TaskType.base and voice_profile is None and not request.ref_audio:
            raise RuntimeError('Base voice cloning requires a saved custom voice profile or ref_audio + ref_text.')
        if task_type == TaskType.base and voice_profile is not None and not (request.ref_text or voice_profile.ref_text or '').strip():
            raise RuntimeError('OmniVoice Base voice cloning requires ref_text for saved voice profiles.')
        if task_type == TaskType.base and request.ref_audio and not (request.ref_text or '').strip():
            raise RuntimeError('OmniVoice Base voice cloning requires ref_text with ref_audio.')
        if task_type == TaskType.voice_design:
            request.instructions = normalize_voice_design_instruct(request.instructions)

    def _group_key_for_request(self, request: SpeechRequest) -> str:
        task_type = self._task_type_for_request(request)
        model_id = request.model or self.store.active_model or self.settings.active_model
        language = (request.language or 'Auto').strip()
        instructions_hash = hashlib.sha1((request.instructions or '').encode('utf-8')).hexdigest()[:12]
        if task_type == TaskType.base:
            # Clone requests of different voices share a batch: the batch runner passes one clone prompt per item
            # (runtime_v2 _generate_batch_sync), so the voice does not have to match.
            voice_key = 'clone'
        elif task_type == TaskType.voice_design:
            voice_key = instructions_hash
        else:
            voice_key = request.voice or self.settings.default_voice
        generation_payload = {
            'speed': request.speed,
            'metadata': request.metadata,
            'num_step': self.settings.num_step,
            'guidance_scale': self.settings.guidance_scale,
            'duration': self.settings.duration,
            't_shift': self.settings.t_shift,
            'denoise': self.settings.denoise,
            'preprocess_prompt': self.settings.preprocess_prompt,
            'postprocess_output': self.settings.postprocess_output,
            'audio_chunk_duration': self.settings.audio_chunk_duration,
            'audio_chunk_threshold': self.settings.audio_chunk_threshold,
            'position_temperature': self.settings.position_temperature,
            'class_temperature': self.settings.class_temperature,
        }
        delivery = request.delivery_controls()
        if delivery:
            # Only when set, so the keys of requests without delivery controls stay as they were.
            generation_payload['delivery'] = delivery
        generation_hash = hashlib.sha1(json.dumps(generation_payload, sort_keys=True, default=str).encode('utf-8')).hexdigest()[:12]
        # The seed is not part of the group: items with different seeds share a batch (see runtime_v2 / row_seeds).
        return '|'.join([model_id, task_type.value, language, voice_key, instructions_hash, generation_hash])

    async def _publish_state(self) -> None:
        await self.events.publish('dashboard.snapshot', self.queue_snapshot())


class StatsService:
    # nvidia-smi takes 30-50 ms. The dashboard stream builds a snapshot per event, so a direct call would block the
    # event loop (and the worker's hand-offs) that often. Serve the last reading and refresh it in a thread instead.
    GPU_STATS_MAX_AGE_S = 1.0

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._gpu_lock = threading.Lock()
        self._gpu_stats: dict[str, Any] | None = None
        self._gpu_stats_key: str | None = None
        self._gpu_stats_at = 0.0
        self._gpu_refreshing = False

    @staticmethod
    def _avg(values: list[float]) -> float | None:
        return statistics.mean(values) if values else None

    def build_stats(self, store: InMemoryStore) -> StatsResponse:
        rolling = StatRolling()
        metrics = list(store.completed_job_metrics)
        if metrics:
            ttfa = [metric['ttfa_ms'] for metric in metrics if metric.get('ttfa_ms') is not None]
            queue_wait = [metric['queue_wait_ms'] for metric in metrics if metric.get('queue_wait_ms') is not None]
            job_wall = [metric['job_wall_ms'] for metric in metrics if metric.get('job_wall_ms') is not None]
            realtime = [metric['realtime_x'] for metric in metrics if metric.get('realtime_x') is not None]
            rolling.ttfa_ms_avg = self._avg(ttfa)
            rolling.queue_wait_ms_avg = self._avg(queue_wait)
            rolling.job_wall_ms_avg = self._avg(job_wall)
            rolling.realtime_x_avg = self._avg(realtime)

        global_stats = StatGlobal(
            jobs_total=store.total_jobs_completed,
            audio_seconds_total=round(store.total_audio_seconds, 3),
            realtime_x_avg=rolling.realtime_x_avg,
        )
        return StatsResponse(
            active_model=store.active_model or '',
            queue_depth=store.queue_depth(),
            worker_state=store.worker_state,
            rolling=rolling,
            global_=global_stats,
        )

    def build_gpu_stats(self) -> GpuStatsResponse:
        device = self.settings.preferred_device
        with self._gpu_lock:
            cached = self._gpu_stats if self._gpu_stats_key == device else None
            stale = cached is None or time.monotonic() - self._gpu_stats_at >= self.GPU_STATS_MAX_AGE_S
            start_refresh = stale and cached is not None and not self._gpu_refreshing
            if start_refresh:
                self._gpu_refreshing = True
        if cached is None:
            # First reading (or the device changed): read once synchronously.
            cached = self._read_gpu_stats(device)
        elif start_refresh:
            threading.Thread(target=self._read_gpu_stats, args=(device,), name='gpu-stats', daemon=True).start()
        return GpuStatsResponse(**cached)

    def _read_gpu_stats(self, device: str | None) -> dict[str, Any]:
        try:
            stats = query_nvidia_smi(device)
        except Exception:
            stats = None
        with self._gpu_lock:
            self._gpu_refreshing = False
            if stats is None:
                return self._gpu_stats or {}
            self._gpu_stats = stats
            self._gpu_stats_key = device
            self._gpu_stats_at = time.monotonic()
        return stats


class TranscriptionService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def transcribe(self, filename: str, content_type: str, data: bytes) -> TranscriptionResponse:
        return await self.transcribe_with_base_url(
            filename,
            content_type,
            data,
            base_url=self.settings.whisper_base_url,
            whisper_path=None,
        )

    async def transcribe_with_base_url(
        self,
        filename: str,
        content_type: str,
        data: bytes,
        *,
        base_url: str | None,
        whisper_path: str | None = None,
        timeout_seconds: float | None = None,
    ) -> TranscriptionResponse:
        if base_url:
            final_error: Exception | None = None
            async with httpx.AsyncClient(timeout=timeout_seconds or 30, trust_env=False) as client:
                for endpoint, form_data in self._whisper_request_candidates(
                    base_url,
                    whisper_path,
                ):
                    try:
                        response = await client.post(
                            endpoint,
                            files={'file': (filename, data, content_type or 'audio/wav')},
                            data=form_data,
                        )
                        response.raise_for_status()
                    except httpx.HTTPStatusError as exc:
                        final_error = exc
                        if exc.response.status_code in {404, 405}:
                            continue
                        raise
                    except httpx.HTTPError as exc:
                        final_error = exc
                        continue
                    payload = response.json()
                    transcription = payload.get('transcription') or payload.get('text') or ''
                    if transcription:
                        return TranscriptionResponse(transcription=transcription, voice_vector=payload.get('voice_vector'))
                    final_error = RuntimeError(f'Whisper endpoint {endpoint} returned no transcript text.')
            raise RuntimeError(f'Whisper transcription failed: {final_error}') from final_error

        return TranscriptionResponse(
            transcription=f'Mock transcription for {filename or "audio"} ({len(data)} bytes).',
            voice_vector=[0.1, -0.2, 0.3],
        )

    @staticmethod
    def _whisper_request_candidates(base_url: str, whisper_path: str | None) -> list[tuple[str, dict[str, str]]]:
        normalized = str(base_url or '').strip().rstrip('/')
        if normalized and '://' not in normalized:
            normalized = f'http://{normalized}'
        parsed = urlsplit(normalized)
        root = urlunsplit((parsed.scheme, parsed.netloc, '', '', '')).rstrip('/') if parsed.scheme and parsed.netloc else normalized
        path = parsed.path.rstrip('/')
        endpoint_like = bool(path and path != '/v1')
        base_without_v1 = normalized[:-3].rstrip('/') if normalized.endswith('/v1') else root if endpoint_like else normalized
        candidates: list[tuple[str, dict[str, str]]] = []
        seen: set[str] = set()

        def add(endpoint: str, fields: dict[str, str]) -> None:
            if endpoint and endpoint not in seen:
                candidates.append((endpoint, fields))
                seen.add(endpoint)

        openai_fields = {'model': 'whisper-1'}
        genesis_fields = {'engine': 'local', 'voice_ident': 'false'}

        if endpoint_like:
            add(
                normalized,
                openai_fields if 'audio/transcriptions' in path else genesis_fields,
            )

        custom_path = str(whisper_path or '').strip()
        if custom_path:
            if not custom_path.startswith('/'):
                custom_path = f'/{custom_path}'
            add(f'{base_without_v1}{custom_path}', {'engine': 'local', 'voice_ident': 'false'})

        add(f'{normalized}/audio/transcriptions', openai_fields)
        add(f'{base_without_v1}/audio/transcriptions', openai_fields)
        add(f'{base_without_v1}/v1/audio/transcriptions', openai_fields)
        add(f'{normalized}/transcribe/', genesis_fields)
        add(f'{base_without_v1}/transcribe/', genesis_fields)
        return candidates


class BenchmarkService:
    def __init__(self, store: InMemoryStore, queue_service: QueueService, events: EventHub, settings: Settings) -> None:
        self.store = store
        self.queue_service = queue_service
        self.events = events
        self.settings = settings

    async def create_run(self, payload: BenchmarkRunCreateRequest) -> BenchmarkRunResponse:
        run_id = f'bench_{uuid.uuid4().hex[:10]}'
        created_at = utcnow()
        mode = (payload.mode or 'traffic').lower()
        case_count = max(1, len(payload.cases))
        total_requests = (
            self._planned_traffic_request_count(payload) * case_count
            if mode == 'traffic'
            else payload.iterations * payload.parallel_requests * case_count
        )
        run: dict[str, Any] = {
            'run_id': run_id,
            'name': payload.name,
            'status': 'running',
            'created_at': created_at,
            'completed_at': None,
            'mode': mode,
            'iterations': payload.iterations,
            'warmup_iterations': payload.warmup_iterations,
            'parallel_requests': payload.parallel_requests,
            'duration_seconds': payload.duration_seconds,
            'requests_per_minute': payload.requests_per_minute,
            'completion_timeout_seconds': payload.completion_timeout_seconds,
            'total_requests': total_requests,
            'exclusive': payload.exclusive,
            'results': [],
            'error_message': None,
        }
        self.store.benchmark_runs[run_id] = run
        self._prune_runs()
        await self.events.publish('dashboard.snapshot', {'reason': 'benchmark.started', 'run_id': run_id})
        spawn_tracked_task(self.store, self._execute_run(run, payload), label=f'benchmark:{run_id}')
        return self._to_response(run)

    def _prune_runs(self) -> None:
        # Each run carries a full per-iteration results[] list (potentially thousands
        # of entries); keep only the newest N so the dict (and every /runs poll) stays
        # bounded. Mirrors the WER service, which already clears between runs.
        cap = max(1, int(getattr(self.settings, 'max_retained_benchmark_runs', 25)))
        runs = self.store.benchmark_runs
        if len(runs) <= cap:
            return
        ordered = sorted(runs.items(), key=lambda kv: kv[1]['created_at'])
        for run_id, _ in ordered[: len(runs) - cap]:
            runs.pop(run_id, None)

    async def list_runs(self) -> list[BenchmarkRunResponse]:
        cap = max(1, int(getattr(self.settings, 'max_retained_benchmark_runs', 25)))
        ordered = sorted(self.store.benchmark_runs.values(), key=lambda item: item['created_at'], reverse=True)
        return [self._to_response(run) for run in ordered[:cap]]

    async def _execute_run(self, run: dict[str, Any], payload: BenchmarkRunCreateRequest) -> None:
        try:
            if payload.exclusive:
                async with self.store.exclusive_lock:
                    await self._execute_payload(run, payload)
            else:
                await self._execute_payload(run, payload)
            run['status'] = 'completed'
        except Exception as exc:
            run['status'] = 'failed'
            run['error_message'] = str(exc)
        finally:
            run['completed_at'] = utcnow()
            await self.events.publish('dashboard.snapshot', {'reason': 'benchmark.completed', 'run_id': run['run_id']})

    async def _execute_payload(self, run: dict[str, Any], payload: BenchmarkRunCreateRequest) -> None:
        if (payload.mode or 'traffic').lower() == 'traffic':
            await self._execute_traffic(run, payload)
            return
        await self._execute_cases(run, payload)

    async def _execute_cases(self, run: dict[str, Any], payload: BenchmarkRunCreateRequest) -> None:
        cases = self._benchmark_cases(payload)
        total_rounds = payload.warmup_iterations + payload.iterations
        for case in cases:
            for round_index in range(total_rounds):
                warmup = round_index < payload.warmup_iterations
                jobs = []
                submit_errors: list[Exception] = []
                for _ in range(payload.parallel_requests):
                    request = self._case_request(case, case.request.input or payload.text)
                    try:
                        jobs.append(await self.queue_service.submit(request, owner_scope='benchmark'))
                    except Exception as exc:
                        submit_errors.append(exc)

                for submit_error in submit_errors:
                    run['results'].append(
                        BenchmarkIterationResult(
                            iteration=round_index + 1,
                            warmup=warmup,
                            label=case.label,
                            success=False,
                            text_preview=(request.input or '')[:160],
                            error_message=str(submit_error),
                        )
                    )

                finished_jobs = await asyncio.gather(
                    *(self.queue_service.wait_for_completion(job.job_id) for job in jobs),
                    return_exceptions=True,
                )
                for finished in finished_jobs:
                    if isinstance(finished, Exception):
                        result = BenchmarkIterationResult(
                            iteration=round_index + 1,
                            warmup=warmup,
                            label=case.label,
                            success=False,
                            error_message=str(finished),
                        )
                    else:
                        result = self._result_from_finished(
                            finished,
                            iteration=round_index + 1,
                            warmup=warmup,
                            label=case.label,
                        )
                    run['results'].append(result)
                await self.events.publish('dashboard.snapshot', {'reason': 'benchmark.iteration', 'run_id': run['run_id']})

    async def _execute_traffic(self, run: dict[str, Any], payload: BenchmarkRunCreateRequest) -> None:
        cases = self._benchmark_cases(payload)
        run['total_requests'] = self._planned_traffic_request_count(payload) * len(cases)
        for case in cases:
            await self._execute_warmups(run, payload, case)
            await self._execute_traffic_case(run, payload, case)

    async def _execute_warmups(self, run: dict[str, Any], payload: BenchmarkRunCreateRequest, case: BenchmarkCaseRequest) -> None:
        for iteration in range(payload.warmup_iterations):
            request = self._case_request(case, case.request.input or payload.text)
            try:
                job = await self.queue_service.submit(request, owner_scope='benchmark')
                finished = await self.queue_service.wait_for_completion(job.job_id)
                result = self._result_from_finished(
                    finished,
                    iteration=iteration + 1,
                    warmup=True,
                    label=case.label,
                    sentence_count=len(self._sentence_pool(request.input or '')),
                )
            except Exception as exc:
                result = BenchmarkIterationResult(
                    iteration=iteration + 1,
                    warmup=True,
                    label=case.label,
                    success=False,
                    text_preview=(request.input or '')[:160],
                    error_message=str(exc),
                )
            run['results'].append(result)
            await self.events.publish('dashboard.snapshot', {'reason': 'benchmark.warmup', 'run_id': run['run_id']})

    async def _execute_traffic_case(self, run: dict[str, Any], payload: BenchmarkRunCreateRequest, case: BenchmarkCaseRequest) -> None:
        request_count = self._planned_traffic_request_count(payload)
        rng = random.Random(payload.random_seed)
        pool = self._sentence_pool(case.request.input or payload.text)
        offsets = sorted(rng.random() * payload.duration_seconds for _ in range(request_count))
        planned_requests = [
            self._random_request_text(pool, payload, rng)
            for _ in range(request_count)
        ]
        start_time = asyncio.get_running_loop().time()
        tasks = [
            asyncio.create_task(
                self._execute_traffic_request(
                    run,
                    case,
                    iteration=index + 1,
                    start_time=start_time,
                    scheduled_offset_seconds=offset,
                    text=text,
                    sentence_count=sentence_count,
                    completion_timeout_seconds=payload.completion_timeout_seconds,
                )
            )
            for index, (offset, (text, sentence_count)) in enumerate(zip(offsets, planned_requests, strict=False))
        ]
        if tasks:
            await asyncio.gather(*tasks)

    async def _execute_traffic_request(
        self,
        run: dict[str, Any],
        case: BenchmarkCaseRequest,
        *,
        iteration: int,
        start_time: float,
        scheduled_offset_seconds: float,
        text: str,
        sentence_count: int,
        completion_timeout_seconds: int,
    ) -> None:
        loop = asyncio.get_running_loop()
        await asyncio.sleep(max(0.0, start_time + scheduled_offset_seconds - loop.time()))
        submitted_at_ms = int(max(0.0, loop.time() - start_time) * 1000)
        scheduled_at_ms = int(scheduled_offset_seconds * 1000)
        request = self._case_request(case, text)
        job_id: str | None = None
        try:
            job = await self.queue_service.submit(request, owner_scope='benchmark')
            job_id = job.job_id
            finished = await asyncio.wait_for(
                self.queue_service.wait_for_completion(job.job_id),
                timeout=max(1, int(completion_timeout_seconds)),
            )
            result = self._result_from_finished(
                finished,
                iteration=iteration,
                warmup=False,
                label=case.label,
                request_id=job_id,
                scheduled_at_ms=scheduled_at_ms,
                submitted_at_ms=submitted_at_ms,
                completed_at_ms=int(max(0.0, loop.time() - start_time) * 1000),
                sentence_count=sentence_count,
            )
        except asyncio.TimeoutError:
            if job_id:
                try:
                    await self.queue_service.cancel(job_id)
                except Exception:
                    pass
            result = BenchmarkIterationResult(
                iteration=iteration,
                warmup=False,
                label=case.label,
                success=False,
                request_id=job_id,
                scheduled_at_ms=scheduled_at_ms,
                submitted_at_ms=submitted_at_ms,
                completed_at_ms=int(max(0.0, loop.time() - start_time) * 1000),
                sentence_count=sentence_count,
                text_preview=text[:160],
                error_message=f'Request exceeded benchmark completion timeout of {completion_timeout_seconds}s.',
            )
        except Exception as exc:
            result = BenchmarkIterationResult(
                iteration=iteration,
                warmup=False,
                label=case.label,
                success=False,
                request_id=job_id,
                scheduled_at_ms=scheduled_at_ms,
                submitted_at_ms=submitted_at_ms,
                completed_at_ms=int(max(0.0, loop.time() - start_time) * 1000),
                sentence_count=sentence_count,
                text_preview=text[:160],
                error_message=str(exc),
            )
        run['results'].append(result)
        await self.events.publish('dashboard.snapshot', {'reason': 'benchmark.request', 'run_id': run['run_id']})

    def _to_response(self, run: dict[str, Any]) -> BenchmarkRunResponse:
        results: list[BenchmarkIterationResult] = list(run.get('results', []))
        labels = sorted({item.label for item in results})
        cases: list[BenchmarkCaseSummary] = []
        for label in labels:
            cases.append(self._summary_for_label(label, results))
        return BenchmarkRunResponse(
            run_id=run['run_id'],
            name=run['name'],
            status=run['status'],
            created_at=run['created_at'],
            completed_at=run.get('completed_at'),
            mode=run.get('mode', 'traffic'),
            iterations=run['iterations'],
            warmup_iterations=run['warmup_iterations'],
            parallel_requests=run['parallel_requests'],
            duration_seconds=run.get('duration_seconds', 60),
            requests_per_minute=run.get('requests_per_minute', 30.0),
            completion_timeout_seconds=run.get('completion_timeout_seconds', 180),
            total_requests=run.get('total_requests', 0),
            exclusive=run['exclusive'],
            cases=cases,
            results=results,
            error_message=run.get('error_message'),
        )

    def _summary_for_label(self, label: str, results: list[BenchmarkIterationResult]) -> BenchmarkCaseSummary:
        measured = [item for item in results if item.label == label and not item.warmup]
        successful = [item for item in measured if item.success]
        ttfa = [item.metrics.ttfa_ms for item in successful]
        queue_wait = [item.metrics.queue_wait_ms for item in successful]
        job_wall = [item.metrics.job_wall_ms for item in successful]
        return BenchmarkCaseSummary(
            label=label,
            iterations=len(successful),
            success_count=len(successful),
            failure_count=len(measured) - len(successful),
            ttfa_ms_avg=self._avg(ttfa),
            ttfa_ms_min=self._min(ttfa),
            ttfa_ms_p50=self._percentile(ttfa, 50),
            ttfa_ms_p95=self._percentile(ttfa, 95),
            ttfa_ms_p99=self._percentile(ttfa, 99),
            ttfa_ms_max=self._max(ttfa),
            queue_wait_ms_avg=self._avg(queue_wait),
            queue_wait_ms_p95=self._percentile(queue_wait, 95),
            queue_wait_ms_p99=self._percentile(queue_wait, 99),
            job_wall_ms_avg=self._avg(job_wall),
            job_wall_ms_min=self._min(job_wall),
            job_wall_ms_p50=self._percentile(job_wall, 50),
            job_wall_ms_p95=self._percentile(job_wall, 95),
            job_wall_ms_p99=self._percentile(job_wall, 99),
            job_wall_ms_max=self._max(job_wall),
            realtime_x_avg=self._avg([item.metrics.realtime_x for item in successful]),
            audio_duration_ms_avg=self._avg([item.metrics.audio_duration_ms for item in successful]),
        )

    @staticmethod
    def _benchmark_cases(payload: BenchmarkRunCreateRequest) -> list[BenchmarkCaseRequest]:
        return payload.cases or [BenchmarkCaseRequest(label='default', request=SpeechRequest())]

    @staticmethod
    def _case_request(case: BenchmarkCaseRequest, text: str) -> SpeechRequest:
        return case.request.model_copy(
            update={
                'input': text,
                'stream': False,
                'response_format': 'wav',
            }
        )

    @staticmethod
    def _planned_traffic_request_count(payload: BenchmarkRunCreateRequest) -> int:
        count = int(round(float(payload.requests_per_minute) * int(payload.duration_seconds) / 60.0))
        return max(1, min(count, 20_000))

    @staticmethod
    def _sentence_pool(text: str) -> list[str]:
        pool = split_sentences(
            text,
            enabled=True,
            short_sentence_merge_max_chars=0,
            following_sentence_merge_min_chars=0,
        )
        return pool or [text.strip()] if text.strip() else ['Benchmark request.']

    @staticmethod
    def _random_request_text(
        pool: list[str],
        payload: BenchmarkRunCreateRequest,
        rng: random.Random,
    ) -> tuple[str, int]:
        min_count = max(1, min(payload.min_sentences_per_request, len(pool)))
        max_count = max(min_count, min(payload.max_sentences_per_request, len(pool)))
        sentence_count = rng.randint(min_count, max_count)
        if sentence_count >= len(pool):
            return ' '.join(pool), len(pool)
        selected_indices = sorted(rng.sample(range(len(pool)), sentence_count))
        return ' '.join(pool[index] for index in selected_indices), sentence_count

    @staticmethod
    def _result_from_finished(
        finished: JobRecord,
        *,
        iteration: int,
        warmup: bool,
        label: str,
        request_id: str | None = None,
        scheduled_at_ms: int | None = None,
        submitted_at_ms: int | None = None,
        completed_at_ms: int | None = None,
        sentence_count: int | None = None,
    ) -> BenchmarkIterationResult:
        success = finished.status == JobStatus.completed
        return BenchmarkIterationResult(
            iteration=iteration,
            warmup=warmup,
            label=label,
            success=success,
            metrics=JobMetrics(**finished.metrics),
            request_id=request_id or finished.job_id,
            scheduled_at_ms=scheduled_at_ms,
            submitted_at_ms=submitted_at_ms,
            completed_at_ms=completed_at_ms,
            sentence_count=sentence_count,
            text_preview=(finished.request.input or '')[:160],
            error_message=finished.error_message,
        )

    @staticmethod
    def _avg(values: list[float | int | None]) -> float | None:
        clean = [float(value) for value in values if value is not None]
        return statistics.mean(clean) if clean else None

    @staticmethod
    def _min(values: list[float | int | None]) -> float | None:
        clean = [float(value) for value in values if value is not None]
        return min(clean) if clean else None

    @staticmethod
    def _max(values: list[float | int | None]) -> float | None:
        clean = [float(value) for value in values if value is not None]
        return max(clean) if clean else None

    @staticmethod
    def _percentile(values: list[float | int | None], percentile: float) -> float | None:
        clean = sorted(float(value) for value in values if value is not None)
        if not clean:
            return None
        if len(clean) == 1:
            return clean[0]
        rank = (max(0.0, min(100.0, percentile)) / 100.0) * (len(clean) - 1)
        lower = math.floor(rank)
        upper = math.ceil(rank)
        if lower == upper:
            return clean[lower]
        weight = rank - lower
        return clean[lower] * (1.0 - weight) + clean[upper] * weight


# The TTS engines read ASCII transliterations literally ("fuer" is not "für"), so test sentences use real letters.
WER_ORTHOGRAPHY_RULE = 'Schreibe Umlaute und ß als echte Zeichen (ä, ö, ü, ß), nie als ae, oe, ue oder ss.'
# Chunks in a row that add no new sentence before the template pool fills the rest.
WER_MAX_STALLED_CHUNKS = 3
# Unit abbreviations the scorer spells out ("1,8 kg" = "1,8 Kilogramm").
WER_UNIT_WORDS = {
    'km/h': 'kilometer pro stunde',
    'kg': 'kilogramm',
    'km': 'kilometer',
    'cm': 'zentimeter',
    'mm': 'millimeter',
    'ml': 'milliliter',
}
# Folded German ordinal stems 1-31 ("ersten", "zwoelfte") -> digits, so "am 12. Mai" = "am zwoelften Mai".
WER_ORDINAL_STEMS = {
    stem: str(number)
    for number, stem in enumerate(
        [
            'erst', 'zweit', 'dritt', 'viert', 'fuenft', 'sechst', 'siebt', 'acht', 'neunt', 'zehnt',
            'elft', 'zwoelft', 'dreizehnt', 'vierzehnt', 'fuenfzehnt', 'sechzehnt', 'siebzehnt', 'achtzehnt',
            'neunzehnt', 'zwanzigst', 'einundzwanzigst', 'zweiundzwanzigst', 'dreiundzwanzigst',
            'vierundzwanzigst', 'fuenfundzwanzigst', 'sechsundzwanzigst', 'siebenundzwanzigst',
            'achtundzwanzigst', 'neunundzwanzigst', 'dreissigst', 'einunddreissigst',
        ],
        start=1,
    )
}
WER_ORDINAL_STEMS['siebent'] = '7'


class WerBenchmarkService:
    def __init__(
        self,
        store: InMemoryStore,
        queue_service: QueueService,
        transcription_service: TranscriptionService,
        events: EventHub,
        settings: Settings,
    ) -> None:
        self.store = store
        self.queue_service = queue_service
        self.transcription_service = transcription_service
        self.events = events
        self.settings = settings
        self._tasks: dict[str, asyncio.Task[Any]] = {}

    def _voice_profile(self, voice_name: str | None) -> Any:
        """The voice profile a run renders with: its own voice, else the default voice; None when there is none."""
        name = (voice_name or self.settings.default_voice or '').strip()
        if not name:
            return None
        lowered = name.casefold()
        return next(
            (p for p in self.store.voice_profiles.values() if name in (p.voice_id, p.name)),
            None,
        ) or next(
            (p for p in self.store.voice_profiles.values() if lowered in (p.voice_id.casefold(), str(p.name or '').casefold())),
            None,
        )

    def best_seed_for_voice(self, voice_name: str | None) -> int | None:
        """The seed saved on the voice for the active engine (its benchmarked best), or None. An unknown or empty
        voice has none."""
        profile = self._voice_profile(voice_name)
        return profile.seed_for_engine(self.settings.runtime_backend) if profile is not None else None

    def _check_engine(self, run: dict[str, Any]) -> None:
        """Stop a run whose engine was switched away: one leaderboard must not mix two engines' results."""
        engine = run.get('engine')
        active = self.settings.runtime_backend
        if engine is not None and active != engine:
            raise RuntimeError(
                f'The engine was switched from {engine} to {active} during the run; '
                'the run was stopped so results of two engines do not mix. Start a new run.'
            )

    async def cancel_run(self, run_id: str) -> WerBenchmarkRunResponse:
        """Stop a running benchmark: end its task and cancel its queued and running synthesis jobs."""
        run = self.store.wer_benchmark_runs.get(run_id)
        if run is None:
            raise KeyError(run_id)
        if run['status'] != 'running':
            return self._to_response(run)
        task = self._tasks.get(run_id)
        if task is not None and not task.done():
            task.cancel()
        for job in list(self.store.jobs.values()):
            if job.owner_scope != 'wer-benchmark' or job.status in {JobStatus.completed, JobStatus.failed, JobStatus.cancelled}:
                continue
            try:
                await self.queue_service.cancel(job.job_id)
            except Exception:
                pass
        if task is not None:
            # A task cancelled before it started never runs its own cleanup, so wait briefly and finish here.
            await asyncio.wait({task}, timeout=15)
        if run['status'] == 'running':
            run['status'] = 'cancelled'
            run['completed_at'] = utcnow()
        await self.events.publish('dashboard.snapshot', {'reason': 'wer_benchmark.cancelled', 'run_id': run_id})
        return self._to_response(run)

    async def create_run(self, payload: WerBenchmarkCreateRequest) -> WerBenchmarkRunResponse:
        run_id = f'wer_{uuid.uuid4().hex[:10]}'
        seed_values = self._seed_values(payload)
        resolved_profile = self._voice_profile(payload.request.voice)
        run: dict[str, Any] = {
            'run_id': run_id,
            'name': payload.name,
            'status': 'running',
            'created_at': utcnow(),
            'completed_at': None,
            'count': payload.count,
            'concurrency': payload.concurrency,
            'seed_start': seed_values[0],
            'seed_range': 0 if payload.seed_values else payload.seed_range,
            'seed_values': seed_values,
            'vllm_base_url': payload.vllm_base_url,
            'vllm_model': payload.vllm_model,
            'whisper_base_url': payload.whisper_base_url,
            'language': payload.language,
            'min_words': payload.min_words,
            'max_words': payload.max_words,
            'tolerance_letters_per_word': payload.tolerance_letters_per_word,
            'completion_timeout_seconds': payload.completion_timeout_seconds,
            'sentence_cache_hit': False,
            'sentence_cache_key': None,
            'exclusive': payload.exclusive,
            'voice': payload.request.voice,
            # The profile the run renders with, fixed at the start (a later default-voice change must not move a
            # leaderboard seed to another voice).
            'resolved_voice': resolved_profile.voice_id if resolved_profile is not None else None,
            'engine': self.settings.runtime_backend,
            'results': [],
            'error_message': None,
        }
        payload.transcription_concurrency = max(
            1,
            payload.transcription_concurrency or self.settings.wer_transcription_concurrency,
        )
        run['transcription_concurrency'] = payload.transcription_concurrency
        self.store.wer_benchmark_runs.clear()
        self.store.wer_benchmark_runs[run_id] = run
        await self.events.publish('dashboard.snapshot', {'reason': 'wer_benchmark.started', 'run_id': run_id})
        self._tasks[run_id] = spawn_tracked_task(self.store, self._execute_run(run, payload), label=f'wer:{run_id}')
        return self._to_response(run)

    async def list_runs(self) -> list[WerBenchmarkRunResponse]:
        return [
            self._to_response(run, include_results=False)
            for run in sorted(self.store.wer_benchmark_runs.values(), key=lambda item: item['created_at'], reverse=True)[:1]
        ]

    def deviations(self, run_id: str, *, offset: int = 0, limit: int = 100) -> tuple[int, list[WerBenchmarkItemResult]]:
        """Samples that deviated (WER > 0), worst first, one page. Raises KeyError for an unknown run."""
        run = self.store.wer_benchmark_runs.get(run_id)
        if run is None:
            raise KeyError(run_id)
        items = [item for item in run.get('results', []) if item.success and (item.wer or 0) > 0]
        items.sort(key=lambda item: (-(item.wer or 0), item.seed if item.seed is not None else -1, item.index))
        return len(items), items[offset : offset + limit]

    async def _execute_run(self, run: dict[str, Any], payload: WerBenchmarkCreateRequest) -> None:
        try:
            if payload.exclusive:
                async with self.store.exclusive_lock:
                    await self._execute_payload(run, payload)
            else:
                await self._execute_payload(run, payload)
            run['status'] = 'completed'
        except asyncio.CancelledError:
            run['status'] = 'cancelled'
        except Exception as exc:
            run['status'] = 'failed'
            run['error_message'] = str(exc)
        finally:
            run['completed_at'] = utcnow()
            self._tasks.pop(run['run_id'], None)
            await self.events.publish('dashboard.snapshot', {'reason': 'wer_benchmark.completed', 'run_id': run['run_id']})

    async def _execute_payload(self, run: dict[str, Any], payload: WerBenchmarkCreateRequest) -> None:
        sentences, cache_info = await self._generate_sentences(payload)
        run['sentence_cache_hit'] = cache_info['hit']
        run['sentence_cache_key'] = cache_info['key']
        run['fallback_sentence_count'] = int(cache_info.get('fallback_count') or 0)
        run['count'] = len(sentences)
        seed_values = list(run.get('seed_values') or self._seed_values(payload))
        run['seed_values'] = seed_values
        run['seed_start'] = seed_values[0] if seed_values else None
        wave_size = max(1, int(payload.concurrency))
        transcription_concurrency = max(1, int(payload.transcription_concurrency))

        for requested_seed in seed_values:
            # No seed in the benchmark means the voice's saved best seed, the same seed a request without one gets.
            seed = (
                requested_seed
                if requested_seed is not None
                else self.best_seed_for_voice(run.get('resolved_voice') or payload.request.voice)
            )
            syntheses: list[dict[str, Any]] = []

            # Keep TTS batching honest: submit a whole wave together, wait for that wave's audio,
            # then start the next wave. Whisper runs only after all audio is rendered, so ASR
            # cannot stagger later TTS jobs into tiny batches or starve the GPU.
            indexed_sentences = list(enumerate(sentences, start=1))
            for start in range(0, len(indexed_sentences), wave_size):
                wave = indexed_sentences[start : start + wave_size]
                self._check_engine(run)
                wave_results = await asyncio.gather(
                    *(self._synthesize_wer_item(index, sentence, payload, seed=seed) for index, sentence in wave)
                )
                # A switch while the wave rendered may have moved its held jobs to the new engine: drop the wave.
                self._check_engine(run)
                for item in wave_results:
                    if item.get('success'):
                        syntheses.append(item)
                    else:
                        run['results'].append(item['result'])
                await self.events.publish(
                    'dashboard.snapshot',
                    {'reason': 'wer_benchmark.tts_wave', 'run_id': run['run_id'], 'seed': seed},
                )

            for start in range(0, len(syntheses), transcription_concurrency):
                wave = syntheses[start : start + transcription_concurrency]
                transcription_results = await asyncio.gather(*(self._transcribe_wer_item(item, payload) for item in wave))
                for result in transcription_results:
                    run['results'].append(result)
                await self.events.publish(
                    'dashboard.snapshot',
                    {
                        'reason': 'wer_benchmark.asr_wave',
                        'run_id': run['run_id'],
                        'seed': seed,
                        'batch_size': len(wave),
                    },
                )

    async def _generate_sentences(self, payload: WerBenchmarkCreateRequest) -> tuple[list[str], dict[str, Any]]:
        excluded_keys = self._sentence_keys(payload.exclude_sentences)
        cache_key = self._sentence_cache_key(
            count=payload.count,
            language=payload.language,
            min_words=payload.min_words,
            max_words=payload.max_words,
            prompt=payload.prompt or '',
            exclude_sentences=payload.exclude_sentences,
        )
        cached = self._cached_sentences(
            cache_key,
            payload.count,
            min_words=payload.min_words,
            max_words=payload.max_words,
            excluded_keys=excluded_keys,
        )
        if cached is not None:
            fallback_count = int(self.store.wer_sentence_cache[cache_key].get('fallback_count') or 0)
            return cached, {'hit': True, 'key': cache_key, 'fallback_count': fallback_count}

        if self.settings.runtime_backend.lower() == 'mock' or payload.vllm_base_url.strip().lower() == 'mock':
            sentences: list[str] = []
            seen: set[str] = set()
            candidate_count = payload.count + len(excluded_keys)
            for index in range(1, candidate_count + 1):
                sentence = f'Dies ist ein kurzer WER Testsatz Nummer {index} mit klaren deutschen Wörtern.'
                if not self._is_valid_wer_sentence(
                    sentence,
                    min_words=payload.min_words,
                    max_words=payload.max_words,
                ):
                    continue
                sentence_key = self._sentence_identity(sentence)
                if not sentence_key or sentence_key in excluded_keys or sentence_key in seen:
                    continue
                sentences.append(sentence)
                seen.add(sentence_key)
                if len(sentences) >= payload.count:
                    break
            fallback_count = max(0, payload.count - len(sentences))
            if fallback_count:
                sentences.extend(
                    self._fallback_sentences(
                        payload,
                        fallback_count,
                        offset=len(sentences),
                        existing_sentences=sentences,
                    )
                )
            self._store_sentence_cache(cache_key, sentences, fallback_count=fallback_count)
            return list(sentences), {'hit': False, 'key': cache_key, 'fallback_count': fallback_count}

        base_url = self._normalize_base_url(payload.vllm_base_url)
        model = (payload.vllm_model or '').strip() or await self._resolve_vllm_model(base_url)

        parsed: list[str] = []
        seen: set[str] = set()
        async with httpx.AsyncClient(timeout=90, trust_env=False) as client:
            chunk_index = 1
            # A chunk with nothing new (repeats, out-of-bounds lengths, unparsable reply) is retried a few times
            # before the template pool fills the rest: those sentences are far less varied.
            stalled_chunks = 0
            while len(parsed) < payload.count:
                previous_len = len(parsed)
                missing = payload.count - len(parsed)
                chunk_count = min(4, missing)
                candidates = await self._generate_sentence_chunk(
                    client=client,
                    base_url=base_url,
                    model=model,
                    payload=payload,
                    chunk_count=chunk_count,
                    chunk_index=chunk_index,
                    existing_count=len(parsed),
                )
                for sentence in candidates:
                    if not self._is_valid_wer_sentence(
                        sentence,
                        min_words=payload.min_words,
                        max_words=payload.max_words,
                    ):
                        continue
                    normalized_key = self._sentence_identity(sentence)
                    if not normalized_key or normalized_key in excluded_keys or normalized_key in seen:
                        continue
                    parsed.append(sentence)
                    seen.add(normalized_key)
                    if len(parsed) >= payload.count:
                        break
                chunk_index += 1
                if len(parsed) > previous_len:
                    stalled_chunks = 0
                    continue
                stalled_chunks += 1
                if stalled_chunks >= WER_MAX_STALLED_CHUNKS:
                    break
        fallback_count = max(0, payload.count - len(parsed))
        if fallback_count:
            parsed.extend(
                self._fallback_sentences(
                    payload,
                    fallback_count,
                    offset=len(parsed),
                    existing_sentences=parsed,
                )
            )
        run_count = payload.count
        sentences = parsed[:run_count]
        self._store_sentence_cache(cache_key, sentences, fallback_count=fallback_count)
        return list(sentences), {'hit': False, 'key': cache_key, 'fallback_count': fallback_count}

    @staticmethod
    def _is_valid_wer_sentence(sentence: str, *, min_words: int, max_words: int) -> bool:
        normalized = re.sub(r'\s+', ' ', str(sentence or '')).strip()
        if not normalized:
            return False
        if re.search(r'\b(?:cache[_ -]?key|chunk[_ -]?id|benchmark[_ -]?id)\b', normalized, re.IGNORECASE):
            return False
        if re.search(r'\b[0-9a-f]{16,}\b', normalized, re.IGNORECASE):
            return False
        word_count = len(re.findall(r'\b[^\W_]+\b', normalized, flags=re.UNICODE))
        return min_words <= word_count <= max_words

    @staticmethod
    def _sentence_identity(sentence: str) -> str:
        """Return the WER-normalized identity used for deduplication and exclusions."""
        return ' '.join(WerBenchmarkService._normalize_words(str(sentence or '')))

    @staticmethod
    def _sentence_keys(sentences: list[str]) -> set[str]:
        keys: set[str] = set()
        for sentence in sentences:
            key = WerBenchmarkService._sentence_identity(sentence)
            if key:
                keys.add(key)
        return keys

    @staticmethod
    def _seed_values(payload: WerBenchmarkCreateRequest) -> list[int | None]:
        if payload.seed_values:
            unique: list[int] = []
            seen: set[int] = set()
            for seed in payload.seed_values:
                if seed in seen:
                    continue
                unique.append(seed)
                seen.add(seed)
            return unique
        if payload.seed_range <= 0:
            return [payload.random_seed]
        base_seed = payload.random_seed if payload.random_seed is not None else 0
        end_seed = base_seed + payload.seed_range
        if end_seed > 2_147_483_647:
            raise RuntimeError('WER seed range exceeds the maximum supported seed value.')
        return list(range(base_seed, end_seed + 1))

    async def _generate_sentence_chunk(
        self,
        *,
        client: httpx.AsyncClient,
        base_url: str,
        model: str,
        payload: WerBenchmarkCreateRequest,
        chunk_count: int,
        chunk_index: int,
        existing_count: int,
    ) -> list[str]:
        custom_prompt = (payload.prompt or '').strip()
        if custom_prompt:
            prompt = (
                f'Nutze diese Anforderungen für einen WER-Test: {custom_prompt}\n\n'
                f'Erzeuge jetzt genau {chunk_count} neue, voneinander unabhängige Testsätze auf {payload.language}. '
                f'Jeder Satz soll {payload.min_words} bis {payload.max_words} Wörter haben. '
                'Keine Nummerierung, keine Listenmarker, keine Abkürzungen, keine chinesischen Zeichen und keine Mischsprache. '
                'Zahlen immer als Ziffern schreiben, zum Beispiel 3 Uhr statt drei Uhr. '
                f'{WER_ORTHOGRAPHY_RULE} '
                'Antworte ausschließlich mit einem JSON-Array von Strings, ohne Objekt, ohne chunk_id und ohne Wrapper.'
            )
        else:
            prompt = (
                f'Erzeuge genau {chunk_count} voneinander unabhängige Testsätze auf {payload.language}. '
                f'Jeder Satz soll {payload.min_words} bis {payload.max_words} Wörter haben, natürlich klingen '
                'und keine Nummerierung enthalten. Verwende gemischte Alltagsthemen, aber keine Zitate, Listenmarker '
                'oder Abkürzungen. Keine chinesischen Zeichen und keine Mischsprache. '
                'Zahlen immer als Ziffern schreiben, zum Beispiel 3 Uhr statt drei Uhr. '
                f'{WER_ORTHOGRAPHY_RULE} '
                'Gib ausschließlich ein JSON-Array von Strings zurück, ohne Objekt, ohne chunk_id und ohne Wrapper.'
            )
        prompt += f' Chunk: {chunk_index}; bisherige Sätze: {existing_count}.'

        response = await client.post(
            f'{base_url}/v1/chat/completions',
            json={
                'model': model,
                'messages': [
                    {
                        'role': 'system',
                        'content': 'Du erzeugst saubere TTS-Testsätze und antwortest nur mit JSON.',
                    },
                    {'role': 'user', 'content': prompt},
                ],
                'temperature': 0.9,
                'top_p': 0.95,
                'max_tokens': min(2048, max(256, chunk_count * max(payload.max_words, 4) * 8)),
                'chat_template_kwargs': {'enable_thinking': False},
            },
        )
        response.raise_for_status()
        data = response.json()
        content = str(data.get('choices', [{}])[0].get('message', {}).get('content') or '')
        return self._parse_generated_sentences(content, chunk_count)

    @staticmethod
    def _fallback_sentences(
        payload: WerBenchmarkCreateRequest,
        count: int,
        *,
        offset: int = 0,
        existing_sentences: list[str] | None = None,
    ) -> list[str]:
        if count <= 0:
            return []
        subjects = [
            'Die Nachbarin',
            'Ein ruhiger Besucher',
            'Meine Kollegin',
            'Der junge Fahrer',
            'Eine freundliche Stimme',
            'Das kleine Team',
            'Ein alter Freund',
            'Die Lehrerin',
            'Mein Nachbar',
            'Eine junge Familie',
            'Der freundliche Verkäufer',
            'Unsere kleine Gruppe',
        ]
        verbs = [
            'beschreibt',
            'erklärt',
            'sortiert',
            'plant',
            'findet',
            'vergleicht',
            'beobachtet',
            'notiert',
            'bespricht',
            'prüft',
            'erinnert',
            'ordnet',
        ]
        objects = [
            'den hellen Morgen im Park',
            'eine kurze Nachricht für die Gruppe',
            'das neue Rezept aus der Küche',
            'mehrere ruhige Wege durch die Stadt',
            'einen einfachen Gedanken zum Feierabend',
            'die warmen Farben am Fenster',
            'ein klares Ergebnis nach dem Test',
            'den leisen Klang im leeren Raum',
            'einen wichtigen Termin für morgen',
            'die Einkaufsliste auf dem Tisch',
            'eine kleine Änderung im Tagesplan',
            'den nächsten Ausflug am Wochenende',
        ]
        endings = [
            'mit sehr deutlicher Aussprache',
            'ohne Hast und ohne fremde Namen',
            'in einem natürlichen deutschen Satz',
            'für einen fairen Vergleich',
            'mit einfachen und bekannten Wörtern',
            'während draußen der Abend beginnt',
            'bei einer Tasse Tee',
            'vor dem gemeinsamen Mittagessen',
            'nach einem kurzen Telefonat',
            'mit einem freundlichen Lächeln',
        ]
        total_combinations = len(subjects) * len(verbs) * len(objects) * len(endings)
        seed_material = json.dumps(
            {
                'language': payload.language,
                'prompt': payload.prompt or '',
                'min_words': payload.min_words,
                'max_words': payload.max_words,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        digest = hashlib.sha1(seed_material.encode('utf-8')).digest()
        start = int.from_bytes(digest[:4], 'big') % total_combinations
        step = (int.from_bytes(digest[4:8], 'big') % total_combinations) or 1
        while math.gcd(step, total_combinations) != 1:
            step = (step + 1) % total_combinations or 1
        blocked_keys = WerBenchmarkService._sentence_keys(payload.exclude_sentences)
        blocked_keys.update(WerBenchmarkService._sentence_keys(existing_sentences or []))
        sentences: list[str] = []
        for candidate_index in range(total_combinations):
            combination = (start + (offset + candidate_index) * step) % total_combinations
            ending = endings[combination % len(endings)]
            combination //= len(endings)
            obj = objects[combination % len(objects)]
            combination //= len(objects)
            verb = verbs[combination % len(verbs)]
            combination //= len(verbs)
            subject = subjects[combination % len(subjects)]
            sentence = f'{subject} {verb} {obj} {ending}.'
            if not WerBenchmarkService._is_valid_wer_sentence(
                sentence,
                min_words=payload.min_words,
                max_words=payload.max_words,
            ):
                continue
            sentence_key = WerBenchmarkService._sentence_identity(sentence)
            if not sentence_key or sentence_key in blocked_keys:
                continue
            sentences.append(sentence)
            blocked_keys.add(sentence_key)
            if len(sentences) >= count:
                return sentences
        raise RuntimeError(
            'WER fallback sentence pool cannot satisfy the request: '
            f'needed {count} unique sentences with {payload.min_words}-{payload.max_words} words '
            f'after exclusions, but only {len(sentences)} were available.'
        )

    def _cached_sentences(
        self,
        cache_key: str,
        count: int,
        *,
        min_words: int,
        max_words: int,
        excluded_keys: set[str],
    ) -> list[str] | None:
        cached = self.store.wer_sentence_cache.get(cache_key)
        if not cached:
            return None
        sentences = cached.get('sentences') or []
        if len(sentences) != count:
            self.store.wer_sentence_cache.pop(cache_key, None)
            return None
        seen: set[str] = set()
        for sentence in sentences:
            sentence_key = self._sentence_identity(sentence)
            if (
                not self._is_valid_wer_sentence(sentence, min_words=min_words, max_words=max_words)
                or not sentence_key
                or sentence_key in excluded_keys
                or sentence_key in seen
            ):
                self.store.wer_sentence_cache.pop(cache_key, None)
                return None
            seen.add(sentence_key)
        cached['last_used_at'] = utcnow()
        cached['hits'] = int(cached.get('hits') or 0) + 1
        return list(sentences)

    def _store_sentence_cache(self, cache_key: str, sentences: list[str], *, fallback_count: int = 0) -> None:
        self.store.wer_sentence_cache[cache_key] = {
            'sentences': list(sentences),
            'fallback_count': int(fallback_count),
            'created_at': utcnow(),
            'last_used_at': utcnow(),
            'hits': 0,
        }
        if len(self.store.wer_sentence_cache) <= 16:
            return
        stale_keys = sorted(
            self.store.wer_sentence_cache,
            key=lambda key: self.store.wer_sentence_cache[key].get('last_used_at') or self.store.wer_sentence_cache[key].get('created_at'),
        )
        for key in stale_keys[: max(0, len(stale_keys) - 16)]:
            self.store.wer_sentence_cache.pop(key, None)

    @staticmethod
    def _sentence_cache_key(
        *,
        count: int,
        language: str,
        min_words: int,
        max_words: int,
        prompt: str,
        exclude_sentences: list[str] | None = None,
    ) -> str:
        payload = {
            'generator_version': 6,
            'count': count,
            'language': (language or '').strip().lower(),
            'min_words': min_words,
            'max_words': max_words,
            'prompt': (prompt or '').strip(),
            'exclude_sentence_keys': sorted(WerBenchmarkService._sentence_keys(exclude_sentences or [])),
        }
        digest = hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()[:16]
        return f'wer_sentences_{count}_{digest}'

    async def _resolve_vllm_model(self, base_url: str) -> str:
        async with httpx.AsyncClient(timeout=15, trust_env=False) as client:
            response = await client.get(f'{base_url}/v1/models')
            response.raise_for_status()
            data = response.json()
        models = data.get('data') or []
        if not models:
            raise RuntimeError('vLLM returned no models from /v1/models.')
        return str(models[0].get('id') or models[0].get('model') or '').strip()

    @staticmethod
    def _normalize_base_url(value: str) -> str:
        normalized = str(value or '').strip().rstrip('/')
        if normalized and '://' not in normalized:
            normalized = f'http://{normalized}'
        for suffix in ('/v1/models', '/v1/chat/completions', '/v1/completions', '/v1'):
            if normalized.lower().endswith(suffix):
                return normalized[: -len(suffix)].rstrip('/')
        return normalized

    async def _synthesize_wer_item(
        self,
        index: int,
        sentence: str,
        payload: WerBenchmarkCreateRequest,
        *,
        seed: int | None,
    ) -> dict[str, Any]:
        request_update: dict[str, Any] = {
            'input': sentence,
            'stream': False,
            'response_format': 'wav',
        }
        if seed is not None:
            request_update['seed'] = seed
        request = payload.request.model_copy(update=request_update)
        job_id: str | None = None
        start = asyncio.get_running_loop().time()
        try:
            job = await self.queue_service.submit(request, owner_scope='wer-benchmark')
            job_id = job.job_id
            finished = await asyncio.wait_for(
                self.queue_service.wait_for_completion(job.job_id),
                timeout=max(1, int(payload.completion_timeout_seconds)),
            )
            if finished.status != JobStatus.completed or not finished.final_audio:
                raise RuntimeError(finished.error_message or 'Synthesis failed.')
            synthesis_ms = int((asyncio.get_running_loop().time() - start) * 1000)
            return {
                'success': True,
                'index': index,
                'seed': seed,
                'sentence': sentence,
                'job_id': job_id,
                'audio': finished.final_audio,
                'content_type': finished.content_type or 'audio/wav',
                'metrics': JobMetrics(**finished.metrics),
                'synthesis_ms': synthesis_ms,
                'started_at': start,
            }
        except asyncio.TimeoutError:
            if job_id:
                try:
                    await self.queue_service.cancel(job_id)
                except Exception:
                    pass
            synthesis_ms = int((asyncio.get_running_loop().time() - start) * 1000)
            return {
                'success': False,
                'result': WerBenchmarkItemResult(
                    index=index,
                    seed=seed,
                    success=False,
                    source_text=sentence,
                    job_id=job_id,
                    synthesis_ms=synthesis_ms,
                    total_ms=synthesis_ms,
                    error_message=f'Request exceeded WER benchmark timeout of {payload.completion_timeout_seconds}s.',
                ),
            }
        except Exception as exc:
            synthesis_ms = int((asyncio.get_running_loop().time() - start) * 1000)
            return {
                'success': False,
                'result': WerBenchmarkItemResult(
                    index=index,
                    seed=seed,
                    success=False,
                    source_text=sentence,
                    job_id=job_id,
                    synthesis_ms=synthesis_ms,
                    total_ms=synthesis_ms,
                    error_message=str(exc),
                ),
            }

    async def _transcribe_wer_item(
        self,
        item: dict[str, Any],
        payload: WerBenchmarkCreateRequest,
    ) -> WerBenchmarkItemResult:
        index = int(item['index'])
        seed = item.get('seed')
        sentence = str(item['sentence'])
        job_id = item.get('job_id')
        synthesis_ms = int(item.get('synthesis_ms') or 0)
        start = asyncio.get_running_loop().time()
        try:
            if self.settings.runtime_backend.lower() == 'mock' or payload.whisper_base_url.strip().lower() == 'mock':
                transcript = sentence
            else:
                transcription = await asyncio.wait_for(
                    self.transcription_service.transcribe_with_base_url(
                        f'wer_{index:04d}.wav',
                        str(item.get('content_type') or 'audio/wav'),
                        item['audio'],
                        base_url=payload.whisper_base_url,
                        whisper_path=None,
                        timeout_seconds=max(1, int(payload.completion_timeout_seconds)),
                    ),
                    timeout=max(1, int(payload.completion_timeout_seconds)),
                )
                transcript = transcription.transcription

            transcription_ms = int((asyncio.get_running_loop().time() - start) * 1000)
            wer = self._wer(sentence, transcript, payload.tolerance_letters_per_word)
            return WerBenchmarkItemResult(
                index=index,
                seed=seed,
                success=True,
                source_text=sentence,
                transcript=transcript,
                normalized_source=' '.join(wer['reference_words']),
                normalized_transcript=' '.join(wer['hypothesis_words']),
                wer=wer['wer'],
                word_count=wer['word_count'],
                word_errors=wer['word_errors'],
                substitutions=wer['substitutions'],
                insertions=wer['insertions'],
                deletions=wer['deletions'],
                job_id=job_id,
                metrics=item.get('metrics') or JobMetrics(),
                synthesis_ms=synthesis_ms,
                transcription_ms=transcription_ms,
                total_ms=synthesis_ms + transcription_ms,
            )
        except asyncio.TimeoutError:
            transcription_ms = int((asyncio.get_running_loop().time() - start) * 1000)
            return WerBenchmarkItemResult(
                index=index,
                seed=seed,
                success=False,
                source_text=sentence,
                job_id=job_id,
                metrics=item.get('metrics') or JobMetrics(),
                synthesis_ms=synthesis_ms,
                transcription_ms=transcription_ms,
                total_ms=synthesis_ms + transcription_ms,
                error_message=f'Whisper exceeded WER benchmark timeout of {payload.completion_timeout_seconds}s.',
            )
        except Exception as exc:
            transcription_ms = int((asyncio.get_running_loop().time() - start) * 1000)
            return WerBenchmarkItemResult(
                index=index,
                seed=seed,
                success=False,
                source_text=sentence,
                job_id=job_id,
                metrics=item.get('metrics') or JobMetrics(),
                synthesis_ms=synthesis_ms,
                transcription_ms=transcription_ms,
                total_ms=synthesis_ms + transcription_ms,
                error_message=str(exc),
            )

    def _to_response(self, run: dict[str, Any], *, include_results: bool = True) -> WerBenchmarkRunResponse:
        results = sorted(
            list(run.get('results', [])),
            key=lambda item: (item.seed if item.seed is not None else -1, item.index),
        )
        total_per_seed = int(run.get('count', 0))
        seed_count = max(1, len(run.get('seed_values') or [run.get('seed_start')]))
        return WerBenchmarkRunResponse(
            run_id=run['run_id'],
            name=run['name'],
            status=run['status'],
            created_at=run['created_at'],
            completed_at=run.get('completed_at'),
            count=run.get('count', 0),
            concurrency=run.get('concurrency', 1),
            transcription_concurrency=run.get('transcription_concurrency', 1),
            seed_start=run.get('seed_start'),
            seed_range=run.get('seed_range', 0),
            seed_values=list(run.get('seed_values') or []),
            vllm_base_url=run.get('vllm_base_url', ''),
            vllm_model=run.get('vllm_model'),
            whisper_base_url=run.get('whisper_base_url', ''),
            language=run.get('language', ''),
            min_words=run.get('min_words', 0),
            max_words=run.get('max_words', 0),
            tolerance_letters_per_word=run.get('tolerance_letters_per_word', 2),
            completion_timeout_seconds=run.get('completion_timeout_seconds', 180),
            sentence_cache_hit=bool(run.get('sentence_cache_hit', False)),
            fallback_sentence_count=int(run.get('fallback_sentence_count') or 0),
            sentence_cache_key=run.get('sentence_cache_key'),
            exclusive=run.get('exclusive', True),
            voice=run.get('voice'),
            resolved_voice=run.get('resolved_voice'),
            engine=run.get('engine') or 'omnivoice',
            summary=self._summary(results, total_per_seed * seed_count),
            seed_leaderboard=self._seed_leaderboard(results, total_per_seed),
            results=results if include_results else [],
            results_count=len(results),
            error_message=run.get('error_message'),
        )

    @classmethod
    def _summary(cls, results: list[WerBenchmarkItemResult], total: int) -> WerBenchmarkSummary:
        successes = [item for item in results if item.success]
        wers = [item.wer for item in successes if item.wer is not None]
        exact_count = sum(1 for item in successes if item.word_errors == 0)
        success_count = len(successes)
        return WerBenchmarkSummary(
            total=total,
            completed=len(results),
            success_count=success_count,
            failure_count=len(results) - success_count,
            wer_avg=cls._avg(wers),
            wer_p50=cls._percentile(wers, 50),
            wer_p95=cls._percentile(wers, 95),
            wer_max=cls._max(wers),
            exact_count=exact_count,
            exact_rate=(exact_count / success_count) if success_count else None,
        )

    @classmethod
    def _seed_leaderboard(cls, results: list[WerBenchmarkItemResult], total_per_seed: int) -> list[WerBenchmarkSeedSummary]:
        grouped: dict[int | None, list[WerBenchmarkItemResult]] = {}
        for item in results:
            grouped.setdefault(item.seed, []).append(item)
        leaderboard: list[WerBenchmarkSeedSummary] = []
        for seed, seed_results in grouped.items():
            summary = cls._summary(seed_results, total_per_seed)
            leaderboard.append(
                WerBenchmarkSeedSummary(
                    seed=seed,
                    total=summary.total,
                    completed=summary.completed,
                    success_count=summary.success_count,
                    failure_count=summary.failure_count,
                    wer_avg=summary.wer_avg,
                    wer_p50=summary.wer_p50,
                    wer_p95=summary.wer_p95,
                    wer_max=summary.wer_max,
                    exact_count=summary.exact_count,
                    exact_rate=summary.exact_rate,
                )
            )
        return sorted(
            leaderboard,
            key=lambda item: (
                item.wer_avg is None,
                item.wer_avg if item.wer_avg is not None else math.inf,
                item.wer_p50 if item.wer_p50 is not None else math.inf,
                item.wer_max if item.wer_max is not None else math.inf,
                item.failure_count,
            ),
        )

    @staticmethod
    def _parse_generated_sentences(content: str, count: int) -> list[str]:
        cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.IGNORECASE | re.DOTALL).strip()
        cleaned = cleaned.strip().strip('`')
        if cleaned.lower().startswith('json'):
            cleaned = cleaned[4:].strip()
        candidates: list[str] = []
        parsed_payload = WerBenchmarkService._parse_llm_container(cleaned)
        if parsed_payload is None:
            start = cleaned.find('[')
            end = cleaned.rfind(']')
            if start >= 0 and end > start:
                parsed_payload = WerBenchmarkService._parse_llm_container(cleaned[start : end + 1])
        if parsed_payload is not None:
            candidates = WerBenchmarkService._extract_sentence_candidates(parsed_payload)
        if not candidates:
            for line in cleaned.splitlines():
                line = re.sub(r'^\s*[\-\*\d\.\)\:]+\s*', '', line).strip().strip('"')
                if line:
                    candidates.append(line)
        normalized: list[str] = []
        seen: set[str] = set()
        for sentence in candidates:
            sentence = WerBenchmarkService._sanitize_generated_sentence(sentence)
            if not sentence:
                continue
            key = sentence.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(sentence)
            if len(normalized) >= count:
                break
        return normalized

    @staticmethod
    def _parse_llm_container(text: str) -> Any | None:
        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(text)
            except Exception:
                continue
        return None

    @staticmethod
    def _extract_sentence_candidates(payload: Any) -> list[str]:
        if isinstance(payload, str):
            return [payload]
        if isinstance(payload, dict):
            preferred_keys = ('text_samples', 'sentences', 'texts', 'samples', 'items', 'data')
            for key in preferred_keys:
                if key in payload:
                    return WerBenchmarkService._extract_sentence_candidates(payload[key])
            extracted: list[str] = []
            for value in payload.values():
                extracted.extend(WerBenchmarkService._extract_sentence_candidates(value))
            return extracted
        if isinstance(payload, (list, tuple)):
            extracted = []
            for item in payload:
                extracted.extend(WerBenchmarkService._extract_sentence_candidates(item))
            return extracted
        return []

    @staticmethod
    def _sanitize_generated_sentence(sentence: str) -> str:
        sentence = str(sentence or '').strip().strip('"').strip("'")
        nested = WerBenchmarkService._parse_llm_container(sentence)
        if nested is not None and not isinstance(nested, str):
            extracted = WerBenchmarkService._extract_sentence_candidates(nested)
            sentence = extracted[0] if extracted else ''
        sentence = re.sub(r'^\s*[\-\*\d\.\)\:]+\s*', '', sentence)
        sentence = re.sub(r'\s+', ' ', sentence).strip()
        if not sentence or re.search(r'[\u3400-\u9fff\uf900-\ufaff]|[\u00e5\u00e7]', sentence):
            return ''
        if any(char in sentence for char in '{}[]'):
            return ''
        sentence = WerBenchmarkService._replace_number_words(sentence)
        return sentence

    @staticmethod
    def _replace_number_words(text: str) -> str:
        number_words = {
            'null': '0',
            'eins': '1',
            'zwei': '2',
            'drei': '3',
            'vier': '4',
            'fuenf': '5',
            'funf': '5',
            'sechs': '6',
            'sieben': '7',
            'acht': '8',
            'neun': '9',
            'zehn': '10',
            'elf': '11',
            'zwoelf': '12',
            'zwolf': '12',
            'dreizehn': '13',
            'vierzehn': '14',
            'fuenfzehn': '15',
            'funfzehn': '15',
            'sechzehn': '16',
            'siebzehn': '17',
            'achtzehn': '18',
            'neunzehn': '19',
            'zwanzig': '20',
            'dreissig': '30',
            'vierzig': '40',
            'fuenfzig': '50',
            'funfzig': '50',
            'sechzig': '60',
            'siebzig': '70',
            'achtzig': '80',
            'neunzig': '90',
            'hundert': '100',
        }

        def normalize_key(value: str) -> str:
            return WerBenchmarkService._fold_german_orthography(value)

        def replace_word(match: re.Match[str]) -> str:
            raw = match.group(0)
            return number_words.get(normalize_key(raw), raw)

        return re.sub(r'\b[^\W\d_]+\b', replace_word, text, flags=re.IGNORECASE)

    @classmethod
    def _wer(cls, reference: str, hypothesis: str, tolerance_letters: int) -> dict[str, Any]:
        reference_words = cls._normalize_words(reference)
        hypothesis_words = cls._normalize_words(hypothesis)
        ref_len = len(reference_words)
        hyp_len = len(hypothesis_words)
        cells: list[list[tuple[int, int, int, int]]] = [
            [(0, 0, 0, 0) for _ in range(hyp_len + 1)]
            for _ in range(ref_len + 1)
        ]
        for row in range(1, ref_len + 1):
            cost, sub, ins, delete = cells[row - 1][0]
            cells[row][0] = (cost + 1, sub, ins, delete + 1)
        for col in range(1, hyp_len + 1):
            cost, sub, ins, delete = cells[0][col - 1]
            cells[0][col] = (cost + 1, sub, ins + 1, delete)

        for row in range(1, ref_len + 1):
            for col in range(1, hyp_len + 1):
                ref_word = reference_words[row - 1]
                hyp_word = hypothesis_words[col - 1]
                tolerant_match = cls._words_match(ref_word, hyp_word, tolerance_letters)
                sub_cost, sub_count, ins_count, del_count = cells[row - 1][col - 1]
                substitution = (
                    sub_cost,
                    sub_count,
                    ins_count,
                    del_count,
                ) if tolerant_match else (
                    sub_cost + 1,
                    sub_count + 1,
                    ins_count,
                    del_count,
                )
                del_cost, del_sub, del_ins, del_del = cells[row - 1][col]
                deletion = (del_cost + 1, del_sub, del_ins, del_del + 1)
                ins_cost, ins_sub, ins_ins, ins_del = cells[row][col - 1]
                insertion = (ins_cost + 1, ins_sub, ins_ins + 1, ins_del)
                candidates = [substitution, deletion, insertion]

                if row >= 2:
                    joined_ref = reference_words[row - 2] + ref_word
                    if cls._joined_word_matches(joined_ref, hyp_word, tolerance_letters):
                        prev_cost, prev_sub, prev_ins, prev_del = cells[row - 2][col - 1]
                        candidates.append((prev_cost, prev_sub, prev_ins, prev_del))

                if col >= 2:
                    joined_hyp = hypothesis_words[col - 2] + hyp_word
                    if cls._joined_word_matches(ref_word, joined_hyp, tolerance_letters):
                        prev_cost, prev_sub, prev_ins, prev_del = cells[row - 1][col - 2]
                        candidates.append((prev_cost, prev_sub, prev_ins, prev_del))

                cells[row][col] = min(candidates, key=lambda item: (item[0], item[1] + item[2] + item[3]))

        word_errors, substitutions, insertions, deletions = cells[ref_len][hyp_len]
        return {
            'reference_words': reference_words,
            'hypothesis_words': hypothesis_words,
            'word_count': ref_len,
            'word_errors': word_errors,
            'substitutions': substitutions,
            'insertions': insertions,
            'deletions': deletions,
            'wer': (word_errors / max(ref_len, 1)),
        }

    @staticmethod
    def _normalize_words(text: str) -> list[str]:
        normalized = WerBenchmarkService._replace_number_words(text)
        normalized = WerBenchmarkService._fold_german_orthography(normalized)
        normalized = WerBenchmarkService._canonicalize_spoken_formats(normalized)
        normalized = re.sub(r'[^a-z0-9\s]+', ' ', normalized)
        return [WerBenchmarkService._normalize_wer_token(word) for word in normalized.split() if word]

    @staticmethod
    def _fold_german_orthography(text: str) -> str:
        normalized = unicodedata.normalize('NFC', text.lower())
        normalized = (
            normalized
            .replace('\u00e4', 'ae')
            .replace('\u00f6', 'oe')
            .replace('\u00fc', 'ue')
            .replace('\u00df', 'ss')
        )
        normalized = unicodedata.normalize('NFKD', normalized)
        return ''.join(char for char in normalized if not unicodedata.combining(char))

    @staticmethod
    def _canonicalize_spoken_formats(text: str) -> str:
        """Give times, prices, units and ordinals one written form, so the scorer counts what was spoken.

        Whisper writes "14:35 Uhr" for a spoken "14 Uhr 35" and "24,50 Euro" for "24 Euro und 50 Cent". Only
        the score uses this; the sentence the engine reads is never rewritten. Expects folded lowercase text.
        """
        text = re.sub(r'(?<![\d.,])(\d{1,3}(?:\.\d{3})+)(?!\d|\.\d)', lambda m: m.group(1).replace('.', ''), text)

        def time(match: re.Match[str]) -> str:
            hours, minutes = int(match.group(1)), int(match.group(2))
            if hours > 24 or minutes > 59:
                return match.group(0)
            return f'{hours} uhr' if minutes == 0 else f'{hours} uhr {minutes}'

        text = re.sub(r'\b(\d{1,2})[:.](\d{2})\s*uhr\b', time, text)
        text = re.sub(r'\b(\d{1,2}):(\d{2})\b', time, text)
        text = re.sub(r'\b(\d{1,2})\s+uhr\s+(\d{1,2})(?:\s+minuten)?\b', time, text)

        def price(match: re.Match[str]) -> str:
            euros, cents = match.group(1), int(match.group(2) or match.group(3))
            return f'{euros} euro' if cents == 0 else f'{euros} euro {cents}'

        text = re.sub(r'\b(\d+),(\d{2})()[\s-]*(?:euro|€)', price, text)
        # "und 50" only counts as cents with "Cent" after it ("5 Euro und 3 Kinder" stays as it is).
        text = re.sub(
            r'\b(\d+)[\s-]*(?:euro|€)\s+(?:und\s+(\d{1,2})\s+cent|(\d{1,2})(?:\s+cent)?)\b(?![.,]\d)', price, text
        )
        text = re.sub(r'(\d)\s*€', r'\1 euro', text)

        for short, word in WER_UNIT_WORDS.items():
            text = re.sub(rf'(?<=\d)\s*{re.escape(short)}\b|\b{re.escape(short)}\b', f' {word}', text)
        text = re.sub(r'(\d)\s*%', r'\1 prozent', text)

        def ordinal(match: re.Match[str]) -> str:
            return WER_ORDINAL_STEMS.get(match.group(1), match.group(0))

        return re.sub(r'\b([a-z]+?)(?:e|en|er|es|em)\b', ordinal, text)

    @staticmethod
    def _normalize_wer_token(word: str) -> str:
        word = word.replace('sauce', 'sosse')
        return word

    @classmethod
    def _joined_word_matches(cls, joined: str, other: str, tolerance_letters: int) -> bool:
        if len(joined) < 6 or len(other) < 6:
            return False
        return cls._words_match(joined, other, tolerance_letters)

    @classmethod
    def _words_match(cls, left: str, right: str, tolerance_letters: int) -> bool:
        if left == right:
            return True
        if left.isdigit() or right.isdigit() or tolerance_letters <= 0:
            return False
        min_len = min(len(left), len(right))
        if min_len <= 3:
            return False
        max_distance = min(tolerance_letters, 1 if min_len <= 5 else 2)
        return cls._word_distance(left, right) <= max_distance

    @staticmethod
    def _word_distance(left: str, right: str) -> int:
        if left == right:
            return 0
        if not left:
            return len(right)
        if not right:
            return len(left)
        previous = list(range(len(right) + 1))
        for row, left_char in enumerate(left, start=1):
            current = [row]
            for col, right_char in enumerate(right, start=1):
                current.append(
                    min(
                        previous[col] + 1,
                        current[col - 1] + 1,
                        previous[col - 1] + (0 if left_char == right_char else 1),
                    )
                )
            previous = current
        return previous[-1]

    @staticmethod
    def _avg(values: list[float | int | None]) -> float | None:
        clean = [float(value) for value in values if value is not None]
        return statistics.mean(clean) if clean else None

    @staticmethod
    def _max(values: list[float | int | None]) -> float | None:
        clean = [float(value) for value in values if value is not None]
        return max(clean) if clean else None

    @staticmethod
    def _percentile(values: list[float | int | None], percentile: float) -> float | None:
        clean = sorted(float(value) for value in values if value is not None)
        if not clean:
            return None
        if len(clean) == 1:
            return clean[0]
        rank = (max(0.0, min(100.0, percentile)) / 100.0) * (len(clean) - 1)
        lower = math.floor(rank)
        upper = math.ceil(rank)
        if lower == upper:
            return clean[lower]
        weight = rank - lower
        return clean[lower] * (1.0 - weight) + clean[upper] * weight
