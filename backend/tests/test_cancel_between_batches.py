"""An active job cancelled while none of its sentences is in a batch must end, not spin the worker.

Regression: cancel()/delete() only flag an active job; the batch plan skips flagged jobs while the worker's wake-up
predicate still counted their pending sentences. The worker then looped without yielding and the whole server
(uvicorn included) stopped answering.
"""
from __future__ import annotations

import asyncio
import threading
import time

import pytest

from omnivoice_tts_server.config import Settings
from omnivoice_tts_server.domain.models import JobStatus, SpeechRequest
from omnivoice_tts_server.domain.state import InMemoryStore
from omnivoice_tts_server.runtime_v2 import MockSynthesizer
from omnivoice_tts_server.services_v2 import EventHub, QueueService

MODEL = 'k2-fsa/OmniVoice-AutoVoice'
S1 = 'Der Zug nach Hamburg fährt heute von Gleis sieben ab.'
S2 = 'Die Abfahrt verzögert sich leider um etwa zehn Minuten.'


def _queue(tmp_path, **overrides) -> QueueService:
    values = dict(data_dir=tmp_path, max_parallel_requests=4, batch_wait_ms=0, sentence_chunking=True, vram_budget_mb=0)
    values.update(overrides)
    store = InMemoryStore(max_queue_size=16)
    return QueueService(store, MockSynthesizer(), EventHub(store), Settings(**values))


def _run_with_deadline(queue: QueueService, coro_factory, seconds: float = 10.0):
    """Run in a thread: a starved event loop would never reach its own timeout."""
    box: dict = {}

    def target() -> None:
        try:
            box['result'] = asyncio.run(coro_factory())
        except BaseException as exc:  # noqa: BLE001
            box['error'] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(seconds)
    if thread.is_alive():
        # A spinning loop checks worker_stop on every pass: stop it so it does not burn a core for later tests.
        queue.store.worker_stop.set()
        thread.join(5)
        pytest.fail('worker loop starved the event loop')
    if 'error' in box:
        raise box['error']
    return box.get('result')


async def _until(predicate, seconds: float = 5.0) -> bool:
    deadline = time.monotonic() + seconds
    while not predicate():
        if time.monotonic() > deadline:
            return False
        await asyncio.sleep(0.005)
    return True


async def _stop(queue: QueueService) -> None:
    queue.store.worker_stop.set()
    async with queue.store.job_condition:
        queue.store.job_condition.notify_all()


@pytest.mark.parametrize('how', ['cancel', 'delete'])
def test_cancelled_active_job_without_a_batch_is_settled(tmp_path, how) -> None:
    queue = _queue(tmp_path)

    async def run():
        doomed = await queue.submit(SpeechRequest(input=f'{S1} {S2}', model=MODEL))
        async with queue.store.job_condition:
            queue._promote_waiting_locked()
        assert doomed.job_id in queue.store.active_request_ids
        await getattr(queue, how)(doomed.job_id)
        assert doomed.status == JobStatus.cancelling
        await queue.start_worker()
        healthy = await queue.submit(SpeechRequest(input=S1, model=MODEL))
        settled = await _until(lambda: doomed.status == JobStatus.cancelled and healthy.status == JobStatus.completed)
        await _stop(queue)
        return settled, doomed, healthy, list(queue.store.active_request_ids)

    settled, doomed, healthy, active = _run_with_deadline(queue, run)
    assert settled, f'loop stayed responsive but the job was not settled: {doomed.status}, {healthy.status}'
    assert doomed.status == JobStatus.cancelled
    assert healthy.status == JobStatus.completed
    assert active == []


def test_delete_of_a_waiting_job_ends_it_at_once(tmp_path) -> None:
    queue = _queue(tmp_path, max_parallel_requests=1)

    async def run():
        first = await queue.submit(SpeechRequest(input=S1, model=MODEL))
        queued = await queue.submit(SpeechRequest(input=S2, model=MODEL))
        async with queue.store.job_condition:
            queue._promote_waiting_locked()
        assert queued.job_id in queue.store.waiting_requests
        await queue.delete(queued.job_id)
        return first, queued, list(queue.store.waiting_requests)

    first, queued, waiting = _run_with_deadline(queue, run)
    assert queued.status == JobStatus.cancelled
    assert waiting == []
    assert first.status != JobStatus.cancelled
