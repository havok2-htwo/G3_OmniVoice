from __future__ import annotations

import asyncio
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from omnivoice_tts_server.app import create_app
from omnivoice_tts_server.config import Settings
from omnivoice_tts_server.domain.state import VoiceProfileRecord

TEST_MODELS_DIR = Path(__file__).resolve().parent / 'fixtures' / 'models'
PASSWORD = 'test-admin-password'


def make_client() -> TestClient:
    temp = tempfile.TemporaryDirectory()
    app = create_app(
        Settings(runtime_backend='mock', models_root_dir=TEST_MODELS_DIR, data_dir=Path(temp.name), active_model='k2-fsa/OmniVoice-AutoVoice', default_voice='Auto Voice')
    )
    client = TestClient(app)
    client.__enter__()
    login = client.post('/api/admin/auth/login', json={'username': 'admin', 'password': 'admin'})
    assert login.status_code == 200, login.text
    changed = client.post('/api/admin/auth/change-password', json={'current_password': 'admin', 'new_password': PASSWORD})
    assert changed.status_code == 200, changed.text
    client._temp = temp
    return client


def _add_voice(client: TestClient, voice_id: str, name: str, seed: int | None = None) -> None:
    client.app.state.store.voice_profiles[voice_id] = VoiceProfileRecord(
        voice_id=voice_id,
        name=name,
        source='custom',
        created_at=datetime(2026, 10, 17, tzinfo=timezone.utc),
        audio_bytes=b'RIFF',
        ref_text='Hallo.',
        consent=True,
        seed=seed,
    )


def test_voice_seed_is_saved_listed_and_cleared() -> None:
    client = make_client()
    _add_voice(client, 'voice_x', 'SonjaV3')

    saved = client.patch('/api/admin/voices/voice_x', json={'seed': 9301})
    assert saved.status_code == 200, saved.text
    assert saved.json()['seed'] == 9301

    listed = {item['voice_id']: item for item in client.get('/api/admin/voices').json()}
    assert listed['voice_x']['seed'] == 9301

    cleared = client.patch('/api/admin/voices/voice_x', json={'seed': None})
    assert cleared.json()['seed'] is None

    assert client.patch('/api/admin/voices/voice_x', json={'seed': -1}).status_code == 422
    assert client.patch('/api/admin/voices/missing', json={'seed': 1}).status_code == 404


def test_voice_rename_keeps_the_seed() -> None:
    client = make_client()
    _add_voice(client, 'voice_y', 'Alt', seed=42)
    renamed = client.patch('/api/admin/voices/voice_y', json={'name': 'Neu'})
    assert renamed.status_code == 200
    assert renamed.json()['name'] == 'Neu'
    assert renamed.json()['seed'] == 42


def test_wer_run_rejects_an_unknown_voice() -> None:
    client = make_client()
    response = client.post('/api/admin/wer-benchmarks/runs', json={'count': 1, 'request': {'input': 'x', 'voice': 'Gibt es nicht'}})
    assert response.status_code == 400
    assert 'not found' in response.json()['detail']


def test_wer_run_can_be_cancelled_over_http(monkeypatch) -> None:
    client = make_client()
    _add_voice(client, 'voice_z', 'Ero2')
    service = client.app.state.wer_benchmark_service

    async def slow_sentences(payload):
        await asyncio.sleep(60)
        return ['x'], {'hit': False, 'key': None}

    monkeypatch.setattr(service, '_generate_sentences', slow_sentences)
    created = client.post('/api/admin/wer-benchmarks/runs', json={'count': 1, 'exclusive': False, 'request': {'input': 'x', 'voice': 'voice_z'}})
    assert created.status_code == 200, created.text
    run_id = created.json()['run_id']
    assert created.json()['voice'] == 'voice_z'

    cancelled = client.post(f'/api/admin/wer-benchmarks/runs/{run_id}/cancel')
    assert cancelled.status_code == 200, cancelled.text
    assert cancelled.json()['status'] == 'cancelled'
    assert client.post('/api/admin/wer-benchmarks/runs/wer_missing/cancel').status_code == 404
