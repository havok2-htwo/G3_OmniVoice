from __future__ import annotations

from datetime import datetime, timedelta, timezone

from omnivoice_tts_server.config import Settings
from omnivoice_tts_server.domain.models import JobStatus, SpeechRequest
from omnivoice_tts_server.domain.state import (
    AUDIO_EVICTION_GRACE_SECONDS,
    STREAM_CHUNKS,
    STREAM_EVENTS,
    InMemoryStore,
    JobRecord,
)

NOW = datetime.now(timezone.utc)  # prune_terminal_jobs uses the real clock
MB = 1024 * 1024


def _finished_job(store: InMemoryStore, job_id: str, *, minutes_ago: float, audio_mb: float, status=JobStatus.completed) -> JobRecord:
    stamp = NOW - timedelta(minutes=minutes_ago)
    job = JobRecord(
        job_id=job_id,
        request=SpeechRequest(input='x'),
        created_at=stamp,
        updated_at=stamp,
        status=status,
        completed_at=stamp,
        final_audio=b'\0' * int(audio_mb * MB),
    )
    store.jobs[job_id] = job
    return job


def test_default_byte_cap_is_256_mb() -> None:
    assert Settings().max_retained_audio_mb == 256


def test_evicts_least_recently_used_audio_until_under_cap() -> None:
    store = InMemoryStore(max_queue_size=4)
    oldest = _finished_job(store, 'a', minutes_ago=30, audio_mb=4)
    middle = _finished_job(store, 'b', minutes_ago=20, audio_mb=4)
    newest = _finished_job(store, 'c', minutes_ago=10, audio_mb=4)

    evicted = store.evict_retained_audio(9 * MB, now=NOW)

    assert evicted == 1
    assert oldest.final_audio is None and oldest.audio_evicted
    assert middle.final_audio and newest.final_audio
    assert set(store.jobs) == {'a', 'b', 'c'}  # records stay listed
    assert store.retained_audio_bytes() == 8 * MB


def test_recent_download_counts_as_use() -> None:
    store = InMemoryStore(max_queue_size=4)
    oldest = _finished_job(store, 'a', minutes_ago=30, audio_mb=4)
    middle = _finished_job(store, 'b', minutes_ago=20, audio_mb=4)
    _finished_job(store, 'c', minutes_ago=10, audio_mb=4)
    oldest.audio_accessed_at = NOW - timedelta(minutes=1)

    store.evict_retained_audio(9 * MB, now=NOW)

    assert oldest.final_audio is not None
    assert middle.final_audio is None and middle.audio_evicted


def test_most_recent_and_grace_period_jobs_are_never_evicted() -> None:
    store = InMemoryStore(max_queue_size=4)
    old = _finished_job(store, 'old', minutes_ago=30, audio_mb=4)
    fresh = _finished_job(store, 'fresh', minutes_ago=AUDIO_EVICTION_GRACE_SECONDS / 120, audio_mb=4)
    newest = _finished_job(store, 'newest', minutes_ago=0, audio_mb=4)

    store.evict_retained_audio(1, now=NOW)

    assert old.final_audio is None
    assert fresh.final_audio is not None  # inside the grace window
    assert newest.final_audio is not None  # always keep the latest result


def test_single_oversized_result_stays_downloadable() -> None:
    store = InMemoryStore(max_queue_size=4)
    big = _finished_job(store, 'big', minutes_ago=30, audio_mb=8)
    assert store.evict_retained_audio(1 * MB, now=NOW) == 0
    assert big.final_audio is not None


def test_active_jobs_are_not_counted_or_touched() -> None:
    store = InMemoryStore(max_queue_size=4)
    old = _finished_job(store, 'old', minutes_ago=30, audio_mb=4)
    running = _finished_job(store, 'running', minutes_ago=40, audio_mb=4, status=JobStatus.running)
    _finished_job(store, 'new', minutes_ago=10, audio_mb=4)

    store.evict_retained_audio(5 * MB, now=NOW)

    assert running.final_audio is not None
    assert old.final_audio is None


def test_prune_applies_byte_cap_and_zero_disables_it() -> None:
    store = InMemoryStore(max_queue_size=4)
    a = _finished_job(store, 'a', minutes_ago=30, audio_mb=4)
    _finished_job(store, 'b', minutes_ago=20, audio_mb=4)

    store.prune_terminal_jobs(retention_days=7, max_retained_jobs=500, max_retained_audio_bytes=0)
    assert a.final_audio is not None

    store.prune_terminal_jobs(retention_days=7, max_retained_jobs=500, max_retained_audio_bytes=5 * MB)
    assert a.final_audio is None
    assert set(store.jobs) == {'a', 'b'}


def test_stream_queues_only_fill_for_their_consumer() -> None:
    job = JobRecord(job_id='j', request=SpeechRequest(input='x'), created_at=NOW, updated_at=NOW)
    job.push_stream_event({'type': 'start'})
    job.push_stream_chunk(b'pcm')
    job.end_stream()
    assert job.stream_events.qsize() == 0 and job.stream_chunks.qsize() == 0

    job.stream_consumer = STREAM_EVENTS
    job.push_stream_event({'type': 'start'})
    job.push_stream_chunk(b'pcm')
    job.end_stream()
    assert job.stream_events.qsize() == 2 and job.stream_chunks.qsize() == 0

    job.detach_stream()
    assert job.stream_consumer is None and job.stream_events.qsize() == 0

    job.stream_consumer = STREAM_CHUNKS
    job.push_stream_chunk(b'pcm')
    job.end_stream()
    assert job.stream_chunks.qsize() == 2 and job.stream_events.qsize() == 0
