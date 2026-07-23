from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from omnivoice_tts_server import services_v2
from omnivoice_tts_server.config import Settings
from omnivoice_tts_server.domain.models import JobStatus, SpeechRequest, WerBenchmarkCreateRequest
from omnivoice_tts_server.domain.state import InMemoryStore, JobRecord
from omnivoice_tts_server.services_v2 import EventHub, QueueService, WerBenchmarkService


def test_queue_job_ids_stay_unique_at_500_job_retention_limit(tmp_path, monkeypatch) -> None:
    fixed_now = datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(services_v2, 'utcnow', lambda: fixed_now)

    settings = Settings(
        data_dir=tmp_path,
        max_queue_size=2,
        max_retained_jobs=500,
        sentence_chunking=False,
    )
    store = InMemoryStore(max_queue_size=2)
    queue = QueueService(store, object(), EventHub(store), settings)
    old_timestamp = fixed_now - timedelta(days=1)

    for index in range(settings.max_retained_jobs):
        job_id = f'retained_{index:04d}'
        store.jobs[job_id] = JobRecord(
            job_id=job_id,
            request=SpeechRequest(input=f'Alter Auftrag {index}'),
            created_at=old_timestamp,
            updated_at=old_timestamp,
            completed_at=old_timestamp,
            status=JobStatus.completed,
        )

    async def submit_across_prune_boundary() -> tuple[JobRecord, JobRecord]:
        first = await queue.submit(SpeechRequest(input='Erster neuer Auftrag'))
        store.waiting_requests.remove(first.job_id)
        first.status = JobStatus.completed
        first.completed_at = fixed_now
        first.updated_at = fixed_now

        removed = store.prune_terminal_jobs(max_retained_jobs=settings.max_retained_jobs)
        assert removed == 1
        assert len(store.jobs) == settings.max_retained_jobs
        assert first.job_id in store.jobs

        second = await queue.submit(SpeechRequest(input='Zweiter neuer Auftrag'))
        return first, second

    first, second = asyncio.run(submit_across_prune_boundary())

    assert first.job_id != second.job_id
    assert first.job_id in store.jobs
    assert second.job_id in store.jobs
    assert len(store.jobs) == settings.max_retained_jobs + 1


@pytest.mark.parametrize(
    ('umlaut_form', 'ascii_form'),
    [
        ('Äpfel', 'Aepfel'),
        ('schön', 'schoen'),
        ('für', 'fuer'),
        ('Straße', 'Strasse'),
        ('fu\u0308r', 'fuer'),
    ],
)
def test_wer_treats_german_transliterations_as_equal_without_tolerance(
    umlaut_form: str,
    ascii_form: str,
) -> None:
    forward = WerBenchmarkService._wer(umlaut_form, ascii_form, tolerance_letters=0)
    reverse = WerBenchmarkService._wer(ascii_form, umlaut_form, tolerance_letters=0)

    assert forward['word_errors'] == 0
    assert forward['wer'] == 0
    assert reverse['word_errors'] == 0
    assert reverse['wer'] == 0


def test_wer_normalizes_fuer_in_a_complete_sentence_without_tolerance() -> None:
    reference = 'Das Wetter ist heute wunderbar fuer einen langen Spaziergang im Park.'
    hypothesis = 'Das Wetter ist heute wunderbar für einen langen Spaziergang im Park.'

    result = WerBenchmarkService._wer(reference, hypothesis, tolerance_letters=0)

    assert result['reference_words'] == result['hypothesis_words']
    assert result['word_errors'] == 0
    assert result['wer'] == 0


@pytest.mark.parametrize(
    ('sentence', 'expected'),
    [
        ('Wir treffen uns morgen im kleinen Cafe.', True),
        ('Zu kurz.', False),
        ('Dieser Satz enthaelt deutlich mehr als die erlaubten acht einzelnen Woerter.', False),
        ('cache_key": "59ef22dab1884888aa257b2d2faa6e7c"', False),
        ('59ef22dab1884888aa257b2d2faa6e7c', False),
    ],
)
def test_wer_sentence_validation_enforces_bounds_and_rejects_technical_fragments(
    sentence: str,
    expected: bool,
) -> None:
    assert WerBenchmarkService._is_valid_wer_sentence(sentence, min_words=4, max_words=8) is expected


def test_wer_fallback_produces_1000_unique_bounded_sentences_per_prompt() -> None:
    first_payload = WerBenchmarkCreateRequest(
        count=1000,
        min_words=4,
        max_words=16,
        prompt='Abwechslungsreiche Alltagsthemen fuer die Vorrunde.',
    )
    second_payload = first_payload.model_copy(
        update={'prompt': 'Andere Alltagsthemen fuer die Finalrunde.'},
    )

    first = WerBenchmarkService._fallback_sentences(first_payload, 1000)
    second = WerBenchmarkService._fallback_sentences(second_payload, 1000)

    assert len(first) == len(set(first)) == 1000
    assert len(second) == len(set(second)) == 1000
    assert all(
        WerBenchmarkService._is_valid_wer_sentence(
            sentence,
            min_words=first_payload.min_words,
            max_words=first_payload.max_words,
        )
        for sentence in first + second
    )
    assert set(first) != set(second)
