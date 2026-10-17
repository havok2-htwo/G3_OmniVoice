from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from omnivoice_tts_server.config import Settings
from omnivoice_tts_server.domain.models import SpeechRequest, WerBenchmarkCreateRequest, WerBenchmarkItemResult
from omnivoice_tts_server.domain.state import InMemoryStore, VoiceProfileRecord
from omnivoice_tts_server.services_v2 import EventHub, WerBenchmarkService


class FakeQueue:
    def __init__(self) -> None:
        self.cancelled: list[str] = []

    async def cancel(self, job_id: str):
        self.cancelled.append(job_id)


def _service(tmp_path, store=None) -> tuple[WerBenchmarkService, InMemoryStore]:
    store = store or InMemoryStore(max_queue_size=8)
    settings = Settings(data_dir=tmp_path, default_voice='voice_b')
    svc = WerBenchmarkService(store, FakeQueue(), object(), EventHub(store), settings)
    return svc, store


def _voice(voice_id: str, name: str, seed: int | None) -> VoiceProfileRecord:
    return VoiceProfileRecord(
        voice_id=voice_id,
        name=name,
        source='custom',
        created_at=datetime(2026, 10, 17, tzinfo=timezone.utc),
        audio_bytes=b'RIFF',
        ref_text='Hallo.',
        consent=True,
        seed=seed,
    )


def test_best_seed_resolves_by_voice_name_id_and_default(tmp_path) -> None:
    svc, store = _service(tmp_path)
    store.voice_profiles['voice_a'] = _voice('voice_a', 'SonjaV3', 9301)
    store.voice_profiles['voice_b'] = _voice('voice_b', 'Ero2', None)
    assert svc.best_seed_for_voice('SonjaV3') == 9301
    assert svc.best_seed_for_voice('voice_a') == 9301
    assert svc.best_seed_for_voice('sonjav3') == 9301
    assert svc.best_seed_for_voice('Ero2') is None
    assert svc.best_seed_for_voice('Unbekannt') is None
    assert svc.best_seed_for_voice(None) is None  # default voice has no seed here


def test_benchmark_without_seed_uses_the_voices_best_seed(tmp_path, monkeypatch) -> None:
    svc, store = _service(tmp_path)
    store.voice_profiles['voice_a'] = _voice('voice_a', 'SonjaV3', 9301)
    seen: list[int | None] = []

    async def fake_sentences(payload):
        return ['Ein Satz.'], {'hit': False, 'key': None}

    async def fake_synth(index, sentence, payload, *, seed):
        seen.append(seed)
        return {'success': True, 'index': index, 'seed': seed, 'sentence': sentence, 'audio': b'', 'content_type': 'audio/wav', 'metrics': {}, 'synthesis_ms': 1, 'started_at': 0.0}

    async def fake_transcribe(item, payload):
        return WerBenchmarkItemResult(index=1, seed=item['seed'], success=True, source_text=item['sentence'], transcript=item['sentence'], wer=0.0)

    monkeypatch.setattr(svc, '_generate_sentences', fake_sentences)
    monkeypatch.setattr(svc, '_synthesize_wer_item', fake_synth)
    monkeypatch.setattr(svc, '_transcribe_wer_item', fake_transcribe)

    async def run() -> None:
        payload = WerBenchmarkCreateRequest(count=1, exclusive=False, request=SpeechRequest(input='x', voice='voice_a'))
        response = await svc.create_run(payload)
        assert response.voice == 'voice_a'
        for _ in range(200):
            if store.wer_benchmark_runs[response.run_id]['status'] != 'running':
                break
            await asyncio.sleep(0.01)

    asyncio.run(run())
    assert seen == [9301]


def test_cancel_stops_a_running_benchmark_and_keeps_the_status(tmp_path, monkeypatch) -> None:
    svc, store = _service(tmp_path)

    async def slow_sentences(payload):
        await asyncio.sleep(60)
        return ['x'], {'hit': False, 'key': None}

    monkeypatch.setattr(svc, '_generate_sentences', slow_sentences)

    async def run() -> None:
        response = await svc.create_run(WerBenchmarkCreateRequest(count=1, exclusive=False, request=SpeechRequest(input='x')))
        await asyncio.sleep(0.05)
        cancelled = await svc.cancel_run(response.run_id)
        assert cancelled.status == 'cancelled'
        assert cancelled.completed_at is not None
        assert response.run_id not in svc._tasks
        again = await svc.cancel_run(response.run_id)  # cancelling a finished run changes nothing
        assert again.status == 'cancelled'

    asyncio.run(run())


def test_cancel_unknown_run_raises_key_error(tmp_path) -> None:
    svc, _ = _service(tmp_path)

    async def run() -> None:
        try:
            await svc.cancel_run('wer_missing')
        except KeyError:
            return
        raise AssertionError('expected KeyError')

    asyncio.run(run())
