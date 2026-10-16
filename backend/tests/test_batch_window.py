from __future__ import annotations

import asyncio
from datetime import timedelta

from omnivoice_tts_server.config import Settings
from omnivoice_tts_server.domain.models import SpeechRequest
from omnivoice_tts_server.domain.state import InMemoryStore, utcnow
from omnivoice_tts_server.runtime_v2 import MockSynthesizer
from omnivoice_tts_server.services_v2 import EventHub, QueueService

SENTENCE = 'Der Zug nach Hamburg fährt heute von Gleis sieben ab.'
AUTO = 'k2-fsa/OmniVoice-AutoVoice'


class RecordingSynthesizer(MockSynthesizer):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

    async def render_batch(self, items):
        self.batch_sizes.append(len(items))
        return await super().render_batch(items)


def _queue(tmp_path, synthesizer=None, **overrides) -> QueueService:
    overrides.setdefault('max_batch_size', 4)
    overrides.setdefault('max_parallel_requests', 4)
    overrides.setdefault('batch_wait_ms', 30)
    settings = Settings(data_dir=tmp_path, sentence_chunking=False, vram_budget_mb=0, **overrides)
    store = InMemoryStore(max_queue_size=64)
    return QueueService(store, synthesizer or object(), EventHub(store), settings)


def _remaining(queue: QueueService, count: int, *, age_ms: int = 0) -> float:
    async def run() -> float:
        for _ in range(count):
            job = await queue.submit(SpeechRequest(input=SENTENCE, model=AUTO))
            job.created_at = utcnow() - timedelta(milliseconds=age_ms)
        queue._promote_waiting_locked()
        return queue._batch_wait_remaining_s_locked()

    return asyncio.run(run())


def test_window_opens_for_a_fresh_partial_batch(tmp_path) -> None:
    remaining = _remaining(_queue(tmp_path), 1)
    assert 0 < remaining <= 0.030


def test_window_counts_from_arrival_not_from_when_the_worker_looks(tmp_path) -> None:
    # A request that already waited through a running batch is past its window and starts at once.
    assert _remaining(_queue(tmp_path), 1, age_ms=100) == 0.0
    assert 0 < _remaining(_queue(tmp_path), 1, age_ms=10) <= 0.020


def test_window_closes_when_the_plan_is_full(tmp_path) -> None:
    assert _remaining(_queue(tmp_path), 4) == 0.0


def test_window_is_off_when_batch_wait_is_zero(tmp_path) -> None:
    assert _remaining(_queue(tmp_path, batch_wait_ms=0), 1) == 0.0


def test_window_is_not_ended_by_the_first_arrival(tmp_path) -> None:
    # Stock behaviour ended the wait at the first notify, so a burst arriving over a few ms split into two batches.
    synth = RecordingSynthesizer()
    queue = _queue(tmp_path, synth, batch_wait_ms=200)

    async def run() -> None:
        await queue.start_worker()
        jobs = [await queue.submit(SpeechRequest(input=SENTENCE, model=AUTO))]
        for _ in range(2):
            await asyncio.sleep(0.02)
            jobs.append(await queue.submit(SpeechRequest(input=SENTENCE, model=AUTO)))
        for _ in range(200):
            if all(job.completed_at for job in jobs):
                break
            await asyncio.sleep(0.01)
        queue.store.worker_stop.set()
        async with queue.store.job_condition:
            queue.store.job_condition.notify_all()
        await queue.store.worker_task

    asyncio.run(run())
    assert synth.batch_sizes == [3]


def test_gpu_stats_are_served_from_cache_and_refreshed_in_the_background(tmp_path, monkeypatch) -> None:
    import threading

    from omnivoice_tts_server import services_v2

    calls: list[threading.Thread] = []
    release = threading.Event()

    def fake_query(device):
        calls.append(threading.current_thread())
        if len(calls) > 1:
            release.wait(2)
        return {'name': 'GPU', 'memory_used_mb': len(calls), 'memory_total_mb': 100, 'utilization_percent': 0, 'temperature_c': None}

    monkeypatch.setattr(services_v2, 'query_nvidia_smi', fake_query)
    stats = services_v2.StatsService(Settings(data_dir=tmp_path))
    main = threading.current_thread()

    assert stats.build_gpu_stats().memory_used_mb == 1  # first reading is synchronous
    assert stats.build_gpu_stats().memory_used_mb == 1  # fresh: no new call
    stats._gpu_stats_at -= 5
    assert stats.build_gpu_stats().memory_used_mb == 1  # stale: old value now, refresh in a thread
    assert stats.build_gpu_stats().memory_used_mb == 1  # refresh in flight: no second thread
    release.set()
    for _ in range(100):
        if stats._gpu_stats['memory_used_mb'] == 2:
            break
        threading.Event().wait(0.01)
    assert stats.build_gpu_stats().memory_used_mb == 2
    assert len(calls) == 2 and calls[0] is main and calls[1] is not main


class TrimRecordingSynthesizer(MockSynthesizer):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[str] = []

    async def render_batch(self, items, *, trim_after: bool = True):
        self.events.append(f'batch{len(items)}:trim_after={trim_after}')
        return await super().render_batch(items)

    async def trim_idle(self) -> None:
        self.events.append('trim_idle')


def test_cuda_trim_runs_when_the_queue_runs_empty_not_after_every_batch(tmp_path) -> None:
    synth = TrimRecordingSynthesizer()
    queue = _queue(tmp_path, synth, max_batch_size=2, batch_wait_ms=0)

    async def run() -> None:
        jobs = [await queue.submit(SpeechRequest(input=SENTENCE, model=AUTO)) for _ in range(5)]
        await queue.start_worker()
        for _ in range(300):
            if all(job.completed_at for job in jobs):
                break
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        queue.store.worker_stop.set()
        async with queue.store.job_condition:
            queue.store.job_condition.notify_all()
        await queue.store.worker_task

    asyncio.run(run())
    assert synth.events == ['batch2:trim_after=False', 'batch2:trim_after=False', 'batch1:trim_after=False', 'trim_idle']
