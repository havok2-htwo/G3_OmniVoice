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
)
from .domain.state import InMemoryStore, JobRecord, RequestState, utcnow
from .prompt_batch import chunk_pcm16le, split_sentences
from .runtime_v2 import BatchSynthesisItem, query_nvidia_smi
from .capacity import batch_audio_budget_seconds, estimate_audio_seconds, max_chars_per_chunk
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

    async def start_worker(self) -> None:
        if self.store.worker_task and not self.store.worker_task.done():
            return
        self.store.worker_stop.clear()
        self.store.worker_task = asyncio.create_task(self._worker_loop())

    async def stop_worker(self) -> None:
        self.store.worker_stop.set()
        async with self.store.job_condition:
            self.store.job_condition.notify_all()
        if self.store.worker_task:
            try:
                await self.store.worker_task
            except Exception:
                pass

    async def submit(self, request: SpeechRequest, *, owner_scope: str = 'public') -> JobRecord:
        text = (request.input or '').strip()
        if not text:
            raise RuntimeError('Missing input text')
        max_input = int(getattr(self.settings, 'max_input_chars', 0) or 0)
        if max_input > 0 and len(text) > max_input:
            raise RequestTooLargeError(
                f'Input text is {len(text)} characters; the limit is {max_input}. '
                'Split it into smaller requests.'
            )
        self._validate_request_voice_model(request)

        sentences = split_sentences(
            text,
            enabled=self.settings.sentence_chunking,
            short_sentence_merge_max_chars=self.settings.short_sentence_merge_max_chars,
            following_sentence_merge_min_chars=self.settings.following_sentence_merge_min_chars,
            max_chars=max_chars_per_chunk(getattr(self.settings, 'vram_budget_mb', 0)),
        )
        if not sentences:
            raise RuntimeError('Missing input text')

        now = utcnow()
        job = JobRecord(
            job_id=f'job_{now.timestamp():.0f}_{len(self.store.jobs):04d}',
            request=request,
            created_at=now,
            updated_at=now,
            owner_scope=owner_scope,
            sentences_total=len(sentences),
        )
        job.metrics['sentences_total'] = len(sentences)
        state = RequestState(job_id=job.job_id, group_key=self._group_key_for_request(request), sentences=sentences)

        async with self.store.job_condition:
            total_outstanding = len(self.store.waiting_requests) + len(self.store.active_request_ids)
            if total_outstanding >= self.store.max_queue_size:
                raise QueueSaturatedError('Queue saturated')
            self.store.jobs[job.job_id] = job
            self.store.request_states[job.job_id] = state
            self.store.waiting_requests.append(job.job_id)
            self._recompute_positions_locked()
            job.stream_events.put_nowait(
                {
                    'type': 'start',
                    'job_id': job.job_id,
                    'sentence_count': len(sentences),
                    'queue_position': job.queue_position,
                }
            )
            self.store.job_condition.notify_all()

        await self._publish_state()
        return job

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
                self.store.job_condition.notify_all()

        await self._publish_state()
        return job

    async def delete(self, job_id: str) -> None:
        async with self.store.job_condition:
            job = self.store.jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.status not in {JobStatus.completed, JobStatus.failed, JobStatus.cancelled}:
                job.cancel_requested = True
                job.status = JobStatus.cancelling
                job.error_message = 'Cancellation requested.'
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
                if self._should_batch_wait_locked():
                    try:
                        await asyncio.wait_for(
                            self.store.job_condition.wait(),
                            timeout=self.settings.batch_wait_ms / 1000.0,
                        )
                    except asyncio.TimeoutError:
                        pass
                    if self.store.worker_stop.is_set():
                        break
                    self._promote_waiting_locked()

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

                    for sentence_index in list(state.inflight_sentence_indices):
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

                self._recompute_positions_locked()
                self.store.prune_terminal_jobs(
                    retention_days=self.settings.retention_days,
                    max_retained_jobs=self.settings.max_retained_jobs,
                )
                self.store.job_condition.notify_all()

            await self._publish_state()

        self.store.worker_state = 'stopped'
        await self._publish_state()

    def _can_use_native_streaming(self, batch_items: list[BatchSynthesisItem]) -> bool:
        if not batch_items or not hasattr(self.synthesizer, 'stream_batch'):
            return False
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

                    job.model_used = model_used
                    job.sample_rate = result.sample_rate
                    state.sample_rate = result.sample_rate
                    job.status = JobStatus.streaming
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

                key = (item.job_id, item.sentence_index)
                duration_ms = durations_by_key.get(key, 0)
                state.sentence_duration_ms.pop(item.sentence_index, None)
                state.ready_sentence_pcm.pop(item.sentence_index, None)
                state.inflight_sentence_indices.discard(item.sentence_index)
                if duration_ms <= 0:
                    self._fail_job_locked(job, 'Native stream produced no audio for a sentence.')
                    continue
                state.completed_streaming_sentence_indices.add(item.sentence_index)
                self._flush_native_streaming_sentences_locked(job, state, batch_id=batch_id)

                job.updated_at = utcnow()
                job.metrics['batch_count'] = state.batch_count
                if state.is_complete():
                    self._complete_job_locked(job, state)
                elif job.status not in {JobStatus.failed, JobStatus.cancelled, JobStatus.cancelling}:
                    job.status = JobStatus.running

            self._recompute_positions_locked()
            self.store.job_condition.notify_all()

    def _should_batch_wait_locked(self) -> bool:
        if self.settings.batch_wait_ms <= 0:
            return False
        if len(self.store.active_request_ids) >= self.settings.max_parallel_requests and not self.store.waiting_requests:
            return False
        has_new_request = False
        for job_id in self.store.active_request_ids:
            state = self.store.request_states.get(job_id)
            job = self.store.jobs.get(job_id)
            if state is not None and job is not None and not job.cancel_requested and state.batch_count == 0:
                has_new_request = True
                break
        if not has_new_request and not self.store.waiting_requests:
            return False
        return 0 < len(self._build_batch_plan_locked()) < self.settings.max_batch_size

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
        job.status = JobStatus.streaming
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
        job.stream_chunks.put_nowait(pcm)
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
        while state.next_emit_sentence_index in state.completed_streaming_sentence_indices:
            sentence_index = state.next_emit_sentence_index
            pending_chunks = state.pending_preview_pcm.pop(sentence_index, [])
            state.pending_preview_duration_ms.pop(sentence_index, None)
            for chunk_index, chunk in enumerate(pending_chunks):
                self._emit_native_chunk_locked(
                    job,
                    state,
                    sentence_index=sentence_index,
                    pcm=chunk,
                    sample_rate=state.sample_rate,
                    batch_id=batch_id,
                    final_chunk_of_sentence=chunk_index == len(pending_chunks) - 1,
                )
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

        first_audio_plan = self._stream_first_audio_plan_locked(eligible)
        if first_audio_plan:
            return first_audio_plan

        # VRAM-aware budget: cap the batch by estimated audio seconds (not just count),
        # so a batch of long sentences uses no more VRAM than one of short ones. 0 = off.
        audio_budget = batch_audio_budget_seconds(getattr(self.settings, 'vram_budget_mb', 0))
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
                if audio_budget > 0 and plan:
                    # Always include at least one sentence; stop before exceeding budget.
                    sentence_s = estimate_audio_seconds(state.sentences[sentence_index])
                    if planned_audio_s + sentence_s > audio_budget:
                        return plan
                    planned_audio_s += sentence_s
                plan.append((job_id, sentence_index))
                per_request_offsets[job_id] = offset + 1
                made_progress = True
                if len(plan) >= self.settings.max_batch_size:
                    break
        return plan

    def _stream_first_audio_plan_locked(self, eligible_job_ids: list[str]) -> list[tuple[str, int]]:
        plan: list[tuple[str, int]] = []
        for job_id in eligible_job_ids:
            job = self.store.jobs[job_id]
            state = self.store.request_states[job_id]
            if not job.request.stream or job.first_audio_at is not None or state.next_emit_sentence_index != 0:
                continue
            if not state.pending_sentence_indices:
                continue
            plan.append((job_id, state.pending_sentence_indices[0]))
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
            job.stream_events.put_nowait(
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
                emitted_samples = 0
                for chunk_index, chunk in enumerate(
                    chunks := list(chunk_pcm16le(pcm, sample_rate=job.sample_rate, chunk_ms=self.settings.stream_chunk_ms))
                ):
                    emitted_samples += len(chunk) // 2
                    job.stream_chunks.put_nowait(chunk)
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
        job.final_audio = self.synthesizer.pcm_to_wav(combined_pcm, sample_rate=job.sample_rate)
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
        self.store.completed_job_metrics.append(job.metrics.copy())

        self.store.request_states.pop(job.job_id, None)
        if job.job_id in self.store.active_request_ids:
            self.store.active_request_ids.remove(job.job_id)

        if job.request.stream:
            job.stream_events.put_nowait(
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
            job.stream_events.put_nowait(None)
            job.stream_chunks.put_nowait(None)

    def _fail_job_locked(self, job: JobRecord, message: str) -> None:
        now = utcnow()
        job.status = JobStatus.failed
        job.error_message = message
        job.completed_at = now
        job.updated_at = now
        job.final_audio = None
        job.content_type = None
        self.store.request_states.pop(job.job_id, None)
        if job.job_id in self.store.active_request_ids:
            self.store.active_request_ids.remove(job.job_id)
        if job.job_id in self.store.waiting_requests:
            self.store.waiting_requests.remove(job.job_id)
        if job.request.stream:
            job.stream_events.put_nowait({'type': 'error', 'message': message})
            job.stream_events.put_nowait(None)
            job.stream_chunks.put_nowait(None)

    def _mark_cancelled_locked(self, job: JobRecord, message: str) -> None:
        now = utcnow()
        job.cancel_requested = True
        job.status = JobStatus.cancelled
        job.completed_at = now
        job.updated_at = now
        job.queue_position = 0
        job.eta_ms = 0
        job.error_message = message
        self.store.request_states.pop(job.job_id, None)
        if job.job_id in self.store.active_request_ids:
            self.store.active_request_ids.remove(job.job_id)
        if job.job_id in self.store.waiting_requests:
            self.store.waiting_requests.remove(job.job_id)
        if job.request.stream:
            job.stream_events.put_nowait({'type': 'error', 'message': message})
            job.stream_events.put_nowait(None)
            job.stream_chunks.put_nowait(None)

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
        seed_key = str(request.seed) if request.seed is not None else 'random'
        if task_type == TaskType.base:
            voice_key = request.voice or hashlib.sha1(
                f"{request.ref_audio or ''}|{request.ref_text or ''}|{request.x_vector_only_mode}".encode('utf-8')
            ).hexdigest()[:12]
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
        generation_hash = hashlib.sha1(json.dumps(generation_payload, sort_keys=True, default=str).encode('utf-8')).hexdigest()[:12]
        return '|'.join([model_id, task_type.value, language, voice_key, instructions_hash, seed_key, generation_hash])

    async def _publish_state(self) -> None:
        await self.events.publish('dashboard.snapshot', self.queue_snapshot())


class StatsService:
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

    @staticmethod
    def build_gpu_stats() -> GpuStatsResponse:
        return GpuStatsResponse(**query_nvidia_smi())


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

    async def create_run(self, payload: WerBenchmarkCreateRequest) -> WerBenchmarkRunResponse:
        run_id = f'wer_{uuid.uuid4().hex[:10]}'
        seed_values = self._seed_values(payload)
        run: dict[str, Any] = {
            'run_id': run_id,
            'name': payload.name,
            'status': 'running',
            'created_at': utcnow(),
            'completed_at': None,
            'count': payload.count,
            'concurrency': payload.concurrency,
            'seed_start': seed_values[0],
            'seed_range': payload.seed_range,
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
        spawn_tracked_task(self.store, self._execute_run(run, payload), label=f'wer:{run_id}')
        return self._to_response(run)

    async def list_runs(self) -> list[WerBenchmarkRunResponse]:
        return [
            self._to_response(run)
            for run in sorted(self.store.wer_benchmark_runs.values(), key=lambda item: item['created_at'], reverse=True)[:1]
        ]

    async def _execute_run(self, run: dict[str, Any], payload: WerBenchmarkCreateRequest) -> None:
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
            await self.events.publish('dashboard.snapshot', {'reason': 'wer_benchmark.completed', 'run_id': run['run_id']})

    async def _execute_payload(self, run: dict[str, Any], payload: WerBenchmarkCreateRequest) -> None:
        sentences, cache_info = await self._generate_sentences(payload)
        run['sentence_cache_hit'] = cache_info['hit']
        run['sentence_cache_key'] = cache_info['key']
        run['count'] = len(sentences)
        seed_values = list(run.get('seed_values') or self._seed_values(payload))
        run['seed_values'] = seed_values
        run['seed_start'] = seed_values[0] if seed_values else None
        wave_size = max(1, int(payload.concurrency))
        transcription_concurrency = max(1, int(payload.transcription_concurrency))

        for seed in seed_values:
            syntheses: list[dict[str, Any]] = []

            # Keep TTS batching honest: submit a whole wave together, wait for that wave's audio,
            # then start the next wave. Whisper runs only after all audio is rendered, so ASR
            # cannot stagger later TTS jobs into tiny batches or starve the GPU.
            indexed_sentences = list(enumerate(sentences, start=1))
            for start in range(0, len(indexed_sentences), wave_size):
                wave = indexed_sentences[start : start + wave_size]
                wave_results = await asyncio.gather(
                    *(self._synthesize_wer_item(index, sentence, payload, seed=seed) for index, sentence in wave)
                )
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
        if self.settings.runtime_backend.lower() == 'mock' or payload.vllm_base_url.strip().lower() == 'mock':
            cache_key = self._sentence_cache_key(
                count=payload.count,
                language=payload.language,
                min_words=payload.min_words,
                max_words=payload.max_words,
                prompt=payload.prompt or '',
            )
            cached = self._cached_sentences(cache_key, payload.count)
            if cached is not None:
                return cached, {'hit': True, 'key': cache_key}
            sentences = [
                f'Dies ist ein kurzer WER Testsatz Nummer {index} mit klaren deutschen Woertern.'
                for index in range(1, payload.count + 1)
            ]
            self._store_sentence_cache(cache_key, sentences)
            return list(sentences), {'hit': False, 'key': cache_key}

        base_url = self._normalize_base_url(payload.vllm_base_url)
        model = (payload.vllm_model or '').strip() or await self._resolve_vllm_model(base_url)
        cache_key = self._sentence_cache_key(
            count=payload.count,
            language=payload.language,
            min_words=payload.min_words,
            max_words=payload.max_words,
            prompt=payload.prompt or '',
        )
        cached = self._cached_sentences(cache_key, payload.count)
        if cached is not None:
            return cached, {'hit': True, 'key': cache_key}

        parsed: list[str] = []
        seen: set[str] = set()
        async with httpx.AsyncClient(timeout=90, trust_env=False) as client:
            chunk_index = 1
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
                    normalized_key = re.sub(r'\s+', ' ', sentence).strip().lower()
                    if not normalized_key or normalized_key in seen:
                        continue
                    parsed.append(sentence)
                    seen.add(normalized_key)
                    if len(parsed) >= payload.count:
                        break
                if not candidates:
                    break
                if len(parsed) == previous_len:
                    break
                chunk_index += 1
        if len(parsed) < payload.count:
            parsed.extend(self._fallback_sentences(payload, payload.count - len(parsed), offset=len(parsed)))
        run_count = payload.count
        sentences = parsed[:run_count]
        self._store_sentence_cache(cache_key, sentences)
        return list(sentences), {'hit': False, 'key': cache_key}

    @staticmethod
    def _seed_values(payload: WerBenchmarkCreateRequest) -> list[int | None]:
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
                f'Nutze diese Anforderungen fuer einen WER-Test: {custom_prompt}\n\n'
                f'Erzeuge jetzt genau {chunk_count} neue, voneinander unabhaengige Testsaetze auf {payload.language}. '
                f'Jeder Satz soll {payload.min_words} bis {payload.max_words} Woerter haben. '
                'Keine Nummerierung, keine Listenmarker, keine Abkuerzungen, keine chinesischen Zeichen und keine Mischsprache. '
                'Zahlen immer als Ziffern schreiben, zum Beispiel 3 Uhr statt drei Uhr. '
                'Antworte ausschliesslich mit einem JSON-Array von Strings, ohne Objekt, ohne chunk_id und ohne Wrapper.'
            )
        else:
            prompt = (
                f'Erzeuge genau {chunk_count} voneinander unabhaengige Testsaetze auf {payload.language}. '
                f'Jeder Satz soll {payload.min_words} bis {payload.max_words} Woerter haben, natuerlich klingen '
                'und keine Nummerierung enthalten. Verwende gemischte Alltagsthemen, aber keine Zitate, Listenmarker '
                'oder Abkuerzungen. Keine chinesischen Zeichen und keine Mischsprache. '
                'Zahlen immer als Ziffern schreiben, zum Beispiel 3 Uhr statt drei Uhr. '
                'Gib ausschliesslich ein JSON-Array von Strings zurueck, ohne Objekt, ohne chunk_id und ohne Wrapper.'
            )
        prompt += f' Chunk: {chunk_index}; bisherige Saetze: {existing_count}.'

        response = await client.post(
            f'{base_url}/v1/chat/completions',
            json={
                'model': model,
                'messages': [
                    {
                        'role': 'system',
                        'content': 'Du erzeugst saubere TTS-Testsaetze und antwortest nur mit JSON.',
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
    def _fallback_sentences(payload: WerBenchmarkCreateRequest, count: int, *, offset: int = 0) -> list[str]:
        subjects = [
            'Die Nachbarin',
            'Ein ruhiger Besucher',
            'Meine Kollegin',
            'Der junge Fahrer',
            'Eine freundliche Stimme',
            'Das kleine Team',
            'Ein alter Freund',
            'Die Lehrerin',
        ]
        verbs = [
            'beschreibt',
            'erklaert',
            'sortiert',
            'plant',
            'findet',
            'vergleicht',
            'beobachtet',
            'notiert',
        ]
        objects = [
            'den hellen Morgen im Park',
            'eine kurze Nachricht fuer die Gruppe',
            'das neue Rezept aus der Kueche',
            'mehrere ruhige Wege durch die Stadt',
            'einen einfachen Gedanken zum Feierabend',
            'die warmen Farben am Fenster',
            'ein klares Ergebnis nach dem Test',
            'den leisen Klang im leeren Raum',
        ]
        endings = [
            'mit sehr deutlicher Aussprache',
            'ohne Hast und ohne fremde Namen',
            'in einem natuerlichen deutschen Satz',
            'fuer einen fairen Vergleich',
            'mit einfachen und bekannten Woertern',
            'waehrend draussen der Abend beginnt',
        ]
        rng = random.Random(17 + offset)
        sentences: list[str] = []
        for index in range(count):
            subject = subjects[(offset + index) % len(subjects)]
            verb = rng.choice(verbs)
            obj = objects[(offset * 3 + index) % len(objects)]
            ending = rng.choice(endings)
            sentences.append(f'{subject} {verb} {obj} {ending}.')
        return sentences

    def _cached_sentences(self, cache_key: str, count: int) -> list[str] | None:
        cached = self.store.wer_sentence_cache.get(cache_key)
        if not cached:
            return None
        sentences = cached.get('sentences') or []
        if len(sentences) != count:
            return None
        cached['last_used_at'] = utcnow()
        cached['hits'] = int(cached.get('hits') or 0) + 1
        return list(sentences)

    def _store_sentence_cache(self, cache_key: str, sentences: list[str]) -> None:
        self.store.wer_sentence_cache[cache_key] = {
            'sentences': list(sentences),
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
    ) -> str:
        payload = {
            'generator_version': 4,
            'count': count,
            'language': (language or '').strip().lower(),
            'min_words': min_words,
            'max_words': max_words,
            'prompt': (prompt or '').strip(),
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

    def _to_response(self, run: dict[str, Any]) -> WerBenchmarkRunResponse:
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
            vllm_base_url=run.get('vllm_base_url', ''),
            vllm_model=run.get('vllm_model'),
            whisper_base_url=run.get('whisper_base_url', ''),
            language=run.get('language', ''),
            min_words=run.get('min_words', 0),
            max_words=run.get('max_words', 0),
            tolerance_letters_per_word=run.get('tolerance_letters_per_word', 2),
            completion_timeout_seconds=run.get('completion_timeout_seconds', 180),
            sentence_cache_hit=bool(run.get('sentence_cache_hit', False)),
            sentence_cache_key=run.get('sentence_cache_key'),
            exclusive=run.get('exclusive', True),
            summary=self._summary(results, total_per_seed * seed_count),
            seed_leaderboard=self._seed_leaderboard(results, total_per_seed),
            results=results,
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
            normalized = value.lower().replace('\u00df', 'ss')
            normalized = unicodedata.normalize('NFKD', normalized)
            return ''.join(char for char in normalized if not unicodedata.combining(char))

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
        normalized = normalized.lower().replace('\u00df', 'ss')
        normalized = unicodedata.normalize('NFKD', normalized)
        normalized = ''.join(char for char in normalized if not unicodedata.combining(char))
        normalized = re.sub(r'[^a-z0-9\s]+', ' ', normalized)
        return [WerBenchmarkService._normalize_wer_token(word) for word in normalized.split() if word]

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
