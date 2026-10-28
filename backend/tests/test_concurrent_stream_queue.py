"""QueueService with a native-streaming synthesizer that supports concurrent batches (the Ming contract).

The fake renders every sentence as a row of a few chunks; rows of different batches run at the same time, like the
engine's continuous batch. Gates let a test hold a sentence open after its first chunk.
"""
from __future__ import annotations

import asyncio
import io
import wave

import numpy as np
import pytest

from omnivoice_tts_server.capacity import ming_capacity_profile
from omnivoice_tts_server.config import Settings
from omnivoice_tts_server.domain.models import JobStatus, SpeechRequest
from omnivoice_tts_server.domain.state import STREAM_EVENTS, InMemoryStore
from omnivoice_tts_server.runtime_v2 import BatchSynthesisResult, MockSynthesizer
from omnivoice_tts_server.services_v2 import EventHub, QueueService

AUTO = 'k2-fsa/OmniVoice-AutoVoice'
ALIAS = 'inclusionAI/Ming-3.0-tts-0.5B-Base'
S1 = 'Der Zug nach Hamburg fährt heute von Gleis sieben ab.'
S2 = 'Die Abfahrt verzögert sich leider um etwa zehn Minuten.'
S3 = 'Wir bitten alle Reisenden um Verständnis für diese Lage.'


def _chunk_value(sentence_index: int, chunk_index: int) -> int:
    return sentence_index * 100 + chunk_index + 1


class FakeConcurrentSynth(MockSynthesizer):
    supports_native_streaming = True
    supports_concurrent_batches = True
    uses_voice_profiles = True

    def __init__(self, slots: int = 4, chunks: int = 3, delay: float = 0.005) -> None:
        super().__init__(sample_rate=24_000)
        self.capacity_profile = ming_capacity_profile(slots)
        self.chunks = chunks
        self.delay = delay
        self.gates: dict[str, asyncio.Event] = {}      # sentence text -> hold after the first chunk
        self.delays: dict[str, float] = {}             # sentence text -> per-chunk delay
        self.fail_texts: set[str] = set()
        self.error_texts: set[str] = set()             # sentence text -> per-item error result (bad voice)
        self.cancel_calls: list[str] = []
        self._cancelled: set[str] = set()
        self.active = 0
        self.max_active = 0
        self.batches: list[list[tuple[str, int]]] = []
        self.render_calls = 0

    async def ensure_model(self, requested_model):
        return ALIAS, 0

    def resident_models(self):
        return {ALIAS}

    def cancel_job(self, job_id: str) -> int:
        self.cancel_calls.append(job_id)
        self._cancelled.add(job_id)
        for gate in self.gates.values():
            gate.set()
        return 1

    async def render_batch(self, items, trim_after: bool = True):
        self.render_calls += 1
        return await super().render_batch(items)

    async def stream_batch(self, items, chunk_size=None, overlap=None):
        self.batches.append([(item.job_id, item.sentence_index) for item in items])
        out: asyncio.Queue = asyncio.Queue()
        self._errored: set[tuple[str, int]] = set()

        async def row(item):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                if item.text in self.error_texts:
                    await out.put(('error', item))
                    self._errored.add((item.job_id, item.sentence_index))
                    return
                for chunk_index in range(self.chunks):
                    if item.job_id in self._cancelled:
                        break
                    await asyncio.sleep(self.delays.get(item.text, self.delay))
                    if item.text in self.fail_texts and chunk_index == 1:
                        raise RuntimeError('engine step failed')
                    pcm = np.full(480, _chunk_value(item.sentence_index, chunk_index), dtype='<i2').tobytes()
                    await out.put(BatchSynthesisResult(item.job_id, item.sentence_index, 24_000, pcm, 20))
                    gate = self.gates.get(item.text)
                    if gate is not None and chunk_index == 0:
                        await gate.wait()
            except Exception as exc:  # noqa: BLE001
                await out.put(exc)
            finally:
                self.active -= 1
                await out.put(('done', item))

        tasks = [asyncio.create_task(row(item)) for item in items]
        finished = 0
        try:
            while finished < len(items):
                payload = await out.get()
                if isinstance(payload, Exception):
                    raise payload
                if isinstance(payload, tuple):
                    kind, item = payload
                    if kind == 'error':
                        yield [BatchSynthesisResult(item.job_id, item.sentence_index, 24_000, b'', 0,
                                                    sentence_finished=True, error='bad reference audio')]
                        continue
                    finished += 1
                    if (item.job_id, item.sentence_index) in self._errored:
                        continue  # its sentence_finished (with the error) was already sent
                    yield [BatchSynthesisResult(item.job_id, item.sentence_index, 24_000, b'', 0, sentence_finished=True)]
                    continue
                yield [payload]
        finally:
            for task in tasks:
                task.cancel()


def _queue(tmp_path, synth, **overrides) -> QueueService:
    overrides.setdefault('max_parallel_requests', 8)
    overrides.setdefault('batch_wait_ms', 0)
    overrides.setdefault('sentence_chunking', True)
    overrides.setdefault('vram_budget_mb', 0)
    settings = Settings(data_dir=tmp_path, **overrides)
    store = InMemoryStore(max_queue_size=64)
    return QueueService(store, synth, EventHub(store), settings)


async def _submit(queue: QueueService, text: str, *, stream: bool = True):
    return await queue.submit(SpeechRequest(input=text, model=AUTO, stream=stream),
                              stream_consumer=STREAM_EVENTS if stream else None)


async def _wait(predicate, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError('timed out')
        await asyncio.sleep(0.005)


def _events(job) -> list[dict]:
    events = []
    while not job.stream_events.empty():
        event = job.stream_events.get_nowait()
        if event is not None:
            events.append(event)
    return events


def _wav_values(wav: bytes) -> list[int]:
    with wave.open(io.BytesIO(wav), 'rb') as handle:
        assert handle.getframerate() == 24_000
        samples = np.frombuffer(handle.readframes(handle.getnframes()), dtype='<i2')
    values: list[int] = []
    for value in samples.tolist():
        if not values or values[-1] != value:
            values.append(value)
    return values


def test_sentences_stream_in_order_while_later_ones_finish_first(tmp_path) -> None:
    synth = FakeConcurrentSynth(slots=8)
    synth.delays = {S1: 0.03, S2: 0.002, S3: 0.002}  # sentence 0 is the slowest
    queue = _queue(tmp_path, synth)

    async def run():
        await queue.start_worker()
        job = await _submit(queue, f'{S1} {S2} {S3}')
        await _wait(lambda: job.status == JobStatus.completed)
        queue.store.worker_stop.set()
        return job

    job = asyncio.run(run())
    events = _events(job)
    chunks = [e for e in events if e['type'] == 'chunk']
    assert len(chunks) == 9 and all(e['native_stream'] and e['sample_rate'] == 24_000 for e in chunks)
    order = [(e['sentence_index'], e['chunk_index']) for e in chunks]
    assert order == sorted(order), 'chunks reach the client in sentence order'
    assert events[-1]['type'] == 'done'
    assert synth.max_active == 3, 'all three sentences rendered at the same time'
    assert _wav_values(job.final_audio) == [_chunk_value(s, c) for s in range(3) for c in range(3)]
    assert job.metrics['ttfa_ms'] >= 0 and job.metrics['queue_wait_ms'] >= 0


def test_a_request_arriving_during_a_stream_gets_audio_before_that_stream_ends(tmp_path) -> None:
    synth = FakeConcurrentSynth(slots=4)
    gate = asyncio.Event
    queue = _queue(tmp_path, synth)

    async def run():
        synth.gates[S1] = gate()
        await queue.start_worker()
        first = await _submit(queue, S1)
        await _wait(lambda: first.first_audio_at is not None)
        second = await _submit(queue, S2)
        await _wait(lambda: second.first_audio_at is not None)
        assert first.status == JobStatus.streaming, 'the first stream is still running'
        assert len(queue._running_batches) == 2
        await _wait(lambda: second.status == JobStatus.completed)
        assert first.status == JobStatus.streaming
        synth.gates[S1].set()
        await _wait(lambda: first.status == JobStatus.completed)
        assert queue.store.current_batch is None and queue.store.worker_state == 'idle'
        assert not queue._native_inflight
        queue.store.worker_stop.set()

    asyncio.run(run())


def test_cancel_stops_the_engine_rows_and_leaves_the_others(tmp_path) -> None:
    synth = FakeConcurrentSynth(slots=4)
    queue = _queue(tmp_path, synth)

    async def run():
        synth.gates[S1] = asyncio.Event()
        await queue.start_worker()
        doomed = await _submit(queue, S1)
        other = await _submit(queue, S2)
        await _wait(lambda: doomed.first_audio_at is not None)
        await queue.cancel(doomed.job_id)
        await _wait(lambda: doomed.status == JobStatus.cancelled and other.status == JobStatus.completed)
        assert doomed.job_id in synth.cancel_calls
        await _wait(lambda: not queue._running_batches)
        queue.store.worker_stop.set()
        return doomed, other

    doomed, other = asyncio.run(run())
    assert _events(doomed)[-1]['type'] == 'error'
    assert other.final_audio is not None


def test_cancel_between_sentences_ends_the_job_at_once(tmp_path) -> None:
    synth = FakeConcurrentSynth(slots=1)  # one sentence at a time
    queue = _queue(tmp_path, synth)

    async def run():
        await queue.start_worker()
        job = await _submit(queue, f'{S1} {S2}')
        await queue.cancel(job.job_id)
        await _wait(lambda: job.status == JobStatus.cancelled)
        await _wait(lambda: not queue._running_batches)
        assert queue.store.active_request_ids == []
        queue.store.worker_stop.set()

    asyncio.run(run())


def test_an_engine_error_fails_only_the_jobs_of_that_batch(tmp_path) -> None:
    synth = FakeConcurrentSynth(slots=4)
    queue = _queue(tmp_path, synth)

    async def run():
        synth.gates[S2] = asyncio.Event()
        synth.fail_texts.add(S1)
        await queue.start_worker()
        healthy = await _submit(queue, S2)
        await _wait(lambda: healthy.first_audio_at is not None)
        broken = await _submit(queue, S1)
        await _wait(lambda: broken.status == JobStatus.failed)
        assert 'engine step failed' in (broken.error_message or '')
        assert healthy.status == JobStatus.streaming
        synth.gates[S2].set()
        await _wait(lambda: healthy.status == JobStatus.completed)
        queue.store.worker_stop.set()
        return broken

    broken = asyncio.run(run())
    events = _events(broken)
    assert events[-1] == {'type': 'error', 'message': 'engine step failed'}


def test_mixed_batch_streams_stream_requests_and_collects_the_rest(tmp_path) -> None:
    synth = FakeConcurrentSynth(slots=4)
    queue = _queue(tmp_path, synth)

    async def run():
        await queue.start_worker()
        streamed = await _submit(queue, S1)
        collected = await _submit(queue, S2, stream=False)
        statuses: list[JobStatus] = []
        while collected.status != JobStatus.completed:
            statuses.append(collected.status)
            await asyncio.sleep(0.001)
        await _wait(lambda: streamed.status == JobStatus.completed)
        queue.store.worker_stop.set()
        return streamed, collected, statuses

    streamed, collected, statuses = asyncio.run(run())
    assert synth.render_calls == 0, 'every item went through stream_batch'
    assert JobStatus.streaming not in statuses, 'a non-stream job is never reported as streaming'
    assert any(e['type'] == 'chunk' for e in _events(streamed))
    assert _wav_values(collected.final_audio) == [_chunk_value(0, c) for c in range(3)]


def test_sentences_in_flight_never_exceed_the_engine_slots(tmp_path) -> None:
    synth = FakeConcurrentSynth(slots=4)
    queue = _queue(tmp_path, synth)
    text = ' '.join([S1, S2, S3] * 3)

    async def run():
        await queue.start_worker()
        jobs = [await _submit(queue, text, stream=index % 2 == 0) for index in range(4)]
        await _wait(lambda: all(job.status == JobStatus.completed for job in jobs), timeout=10)
        queue.store.worker_stop.set()
        return jobs

    jobs = asyncio.run(run())
    assert synth.max_active <= 4
    for job in jobs:
        expected = [_chunk_value(s, c) for s in range(9) for c in range(3)]
        assert _wav_values(job.final_audio) == expected


def test_first_sentences_of_new_jobs_go_first(tmp_path) -> None:
    synth = FakeConcurrentSynth(slots=4)
    queue = _queue(tmp_path, synth)

    async def run():
        long_job = await _submit(queue, ' '.join([S1, S2, S3] * 2))
        short_job = await _submit(queue, S2)
        async with queue.store.job_condition:
            queue._promote_waiting_locked()
            plan = queue._build_concurrent_plan_locked()
        return long_job, short_job, plan

    long_job, short_job, plan = asyncio.run(run())
    # slots 4, 2 per job, 1 slot kept free for newcomers: both first sentences, then the long job's second; a third
    # would fill the last slot.
    assert plan == [(long_job.job_id, 0), (short_job.job_id, 0), (long_job.job_id, 1)]


def test_capacity_profile_sets_the_sentence_split_length(tmp_path) -> None:
    synth = FakeConcurrentSynth(slots=4)
    synth.capacity_profile = {**synth.capacity_profile, 'max_chars_per_chunk': 0}
    queue = _queue(tmp_path, synth, sentence_chunking=False, vram_budget_mb=7000)
    long_text = ' '.join(['wort'] * 600)

    async def run():
        return await _submit(queue, long_text, stream=False)

    job = asyncio.run(run())
    assert job.sentences_total == 1, 'OmniVoice VRAM split length does not apply'
    plain = _queue(tmp_path, MockSynthesizer(), sentence_chunking=False, vram_budget_mb=7000)
    assert plain.max_chars_per_sentence() > 0
    assert not plain._uses_concurrent_batches()


@pytest.mark.parametrize('flags', [(True, False), (False, True)])
def test_concurrency_needs_both_capability_flags(tmp_path, flags) -> None:
    synth = FakeConcurrentSynth()
    synth.supports_native_streaming, synth.supports_concurrent_batches = flags
    assert not _queue(tmp_path, synth)._uses_concurrent_batches()


def test_one_job_never_takes_the_slots_kept_for_newcomers(tmp_path) -> None:
    synth = FakeConcurrentSynth(slots=8)  # headroom 2
    queue = _queue(tmp_path, synth)

    async def run():
        long_job = await _submit(queue, ' '.join([S1, S2, S3] * 4))
        async with queue.store.job_condition:
            queue._promote_waiting_locked()
            first = queue._build_concurrent_plan_locked()
            queue._reserve_batch_locked(first)
            queue._native_inflight.update(first)
        newcomers = [await _submit(queue, S2) for _ in range(3)]
        async with queue.store.job_condition:
            queue._promote_waiting_locked()
            second = queue._build_concurrent_plan_locked()
        return long_job, newcomers, first, second

    long_job, newcomers, first, second = asyncio.run(run())
    assert first == [(long_job.job_id, index) for index in range(6)], 'one job fills all but the headroom'
    # The newcomers take the two free slots first; the third waits for a finished row.
    assert second == [(newcomers[0].job_id, 0), (newcomers[1].job_id, 0)]


def test_further_sentences_go_to_the_least_loaded_job(tmp_path) -> None:
    synth = FakeConcurrentSynth(slots=8)
    queue = _queue(tmp_path, synth)

    async def run():
        jobs = [await _submit(queue, ' '.join([S1, S2, S3] * 2)) for _ in range(2)]
        async with queue.store.job_condition:
            queue._promote_waiting_locked()
            queue._native_inflight.update({(jobs[0].job_id, 90), (jobs[0].job_id, 91)})  # job 0 renders two already
            queue._native_inflight.add((jobs[1].job_id, 90))
            plan = queue._build_concurrent_plan_locked()
        return jobs, plan

    jobs, plan = asyncio.run(run())
    # 3 in flight, 8 slots, 2 kept free -> 3 more: job 1 (load 1) first, then alternating from equal load.
    assert [job_id for job_id, _ in plan] == [jobs[1].job_id, jobs[0].job_id, jobs[1].job_id]


def test_an_abandoned_stream_cancels_its_job_in_its_own_task() -> None:
    from types import SimpleNamespace

    from omnivoice_tts_server.api import router_v2

    class Queue:
        def __init__(self) -> None:
            self.cancelled: list[str] = []

        async def cancel(self, job_id: str) -> None:
            self.cancelled.append(job_id)

    async def run():
        queue = Queue()
        router_v2._cancel_abandoned_stream(queue, SimpleNamespace(job_id='live', status=JobStatus.streaming))
        router_v2._cancel_abandoned_stream(queue, SimpleNamespace(job_id='done', status=JobStatus.completed))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        return queue.cancelled

    assert asyncio.run(run()) == ['live']


def test_a_job_completes_when_its_own_sentence_ends_not_the_whole_batch(tmp_path) -> None:
    """Two jobs' first sentences share one stream batch: the short one completes while the long one still renders."""
    synth = FakeConcurrentSynth(slots=4)
    queue = _queue(tmp_path, synth)

    async def run():
        synth.gates[S1] = asyncio.Event()
        slow = await _submit(queue, S1, stream=False)
        fast = await _submit(queue, S2, stream=False)
        await queue.start_worker()  # both are waiting: one batch
        await _wait(lambda: fast.status == JobStatus.completed)
        assert synth.batches[0] == [(slow.job_id, 0), (fast.job_id, 0)]
        assert slow.status == JobStatus.running and len(queue._running_batches) == 1
        assert fast.job_id not in queue.store.active_request_ids
        synth.gates[S1].set()
        await _wait(lambda: slow.status == JobStatus.completed)
        await _wait(lambda: not queue._running_batches)
        assert not queue._native_inflight and queue.store.active_request_ids == []
        queue.store.worker_stop.set()
        return slow, fast

    slow, fast = asyncio.run(run())
    assert _wav_values(fast.final_audio) == [_chunk_value(0, c) for c in range(3)]
    assert _wav_values(slow.final_audio) == [_chunk_value(0, c) for c in range(3)]


def test_a_per_item_error_fails_only_that_job_of_the_batch(tmp_path) -> None:
    synth = FakeConcurrentSynth(slots=4)
    synth.error_texts.add(S1)
    queue = _queue(tmp_path, synth)

    async def run():
        broken = await _submit(queue, S1)
        healthy = await _submit(queue, S2)
        await queue.start_worker()
        await _wait(lambda: broken.status == JobStatus.failed and healthy.status == JobStatus.completed)
        assert len(synth.batches[0]) == 2, 'both were in one batch'
        await _wait(lambda: not queue._running_batches)
        assert not queue._native_inflight
        queue.store.worker_stop.set()
        return broken, healthy

    broken, healthy = asyncio.run(run())
    assert broken.error_message == 'bad reference audio'
    assert _events(broken)[-1] == {'type': 'error', 'message': 'bad reference audio'}
    assert broken.job_id in synth.cancel_calls and healthy.job_id not in synth.cancel_calls
    assert _wav_values(healthy.final_audio) == [_chunk_value(0, c) for c in range(3)]


def test_concurrent_engines_skip_the_batch_wait_window(tmp_path) -> None:
    synth = FakeConcurrentSynth(slots=4)
    queue = _queue(tmp_path, synth, batch_wait_ms=5000)

    async def run():
        await queue.start_worker()
        job = await _submit(queue, S1)
        await _wait(lambda: job.status == JobStatus.completed, timeout=2.0)
        queue.store.worker_stop.set()

    asyncio.run(run())
