"""The real Audio8 models through the real server routes: start, live switch, clone a voice profile, audio out.

Skipped without the model files; the GPU engines also need CUDA. Slots are small, so this says nothing about speed.
"""
from __future__ import annotations

import io
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from omnivoice_tts_server.app import create_app
from omnivoice_tts_server.audio8.variants import AUDIO8_VARIANTS
from omnivoice_tts_server.config import Settings

from test_app import login_admin

torch = pytest.importorskip('torch')
sf = pytest.importorskip('soundfile')

REPO = Path(__file__).resolve().parents[2]
VOICE = REPO / 'data' / 'voices' / 'voice_preset_sonjav2.wav'
REF_TEXT = ('Franz jagt im komplett verwahrlosten Taxi quer durch Bayern. Vorbei an dichten Wäldern, kleinen Dörfern '
            'und alten Bahnhöfen.')


def _usable(engine: str) -> bool:
    variant = AUDIO8_VARIANTS[engine]
    folder = REPO / 'models' / variant.folder
    if not VOICE.is_file() or not all((folder / name).is_file() for name in variant.weight_files):
        return False
    return variant.runtime == 'onnx' or torch.cuda.is_available()


ENGINES = [pytest.param(engine, marks=pytest.mark.skipif(not _usable(engine), reason=f'needs the {engine} model'))
           for engine in AUDIO8_VARIANTS]


@pytest.fixture(scope='module')
def client(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp('data')
    # A profile per engine, or the first switch would start from the default tier's slots.
    saved = {
        'audio8_max_slots': 2, 'audio8_onnx_workers': 1,
        'engine_profiles': {engine: {'max_parallel_requests': 2} for engine in AUDIO8_VARIANTS},
    }
    (data_dir / 'runtime_settings.json').write_text(json.dumps(saved), encoding='utf-8')
    app = create_app(Settings(
        runtime_backend='mock', models_root_dir=REPO / 'models', data_dir=data_dir, default_voice='Anna',
        active_model='k2-fsa/OmniVoice-AutoVoice',
    ))
    with TestClient(app) as test_client:
        login_admin(test_client)
        created = test_client.post(
            '/api/admin/voices',
            files={'audio_sample': ('anna.wav', VOICE.read_bytes(), 'audio/wav')},
            data={'name': 'Anna', 'ref_text': REF_TEXT, 'consent': 'true'},
        )
        assert created.status_code in (200, 201), created.text
        test_client.voice_id = created.json()['voice_id']
        yield test_client
        shutil.rmtree(data_dir, ignore_errors=True)


def _wav(response) -> tuple[np.ndarray, int]:
    assert response.status_code == 200, response.text
    return sf.read(io.BytesIO(response.content), dtype='float32')


@pytest.mark.parametrize('engine', ENGINES)
def test_switch_clone_and_stream_on_the_real_model(client, engine) -> None:
    switched = client.post('/api/admin/runtime/backend', json={'backend': engine, 'wait': True})
    assert switched.status_code == 200, switched.text
    assert switched.json()['switch']['error'] is None, switched.json()['switch']
    settings = client.app.state.settings
    alias = AUDIO8_VARIANTS[engine].base_alias
    assert settings.active_model == alias

    capabilities = client.get('/api/v1/capabilities').json()
    assert (capabilities['engine'], capabilities['loaded'], capabilities['sample_rate']) == (engine, True, 44100)
    assert client.get('/api/admin/settings').json()['output_sample_rate'] == 44100

    text = 'Guten Morgen, das ist ein kurzer Test.'
    payload = {'model': 'tts-1', 'input': text, 'voice': 'Anna', 'seed': 1234, 'response_format': 'wav'}
    audio, rate = _wav(client.post('/v1/audio/speech', json=payload))
    assert rate == 44100
    assert 0.8 < len(audio) / rate < 8.0
    assert 0.003 < float(np.sqrt(np.mean(audio ** 2))) < 0.5

    again, _ = _wav(client.post('/v1/audio/speech', json=payload))
    assert np.array_equal(audio, again), 'same text, voice and seed through the queue give the same samples'

    pcm = client.post('/v1/audio/speech', json={**payload, 'response_format': 'pcm', 'stream': True})
    assert pcm.status_code == 200, pcm.text
    streamed = np.frombuffer(pcm.content, dtype=np.int16).astype(np.float32) / 32768.0
    assert len(streamed) == len(audio) or abs(len(streamed) - len(audio)) < rate * 0.05
    length = min(len(streamed), len(audio))
    assert np.max(np.abs(streamed[:length] - audio[:length])) < 2 / 32768.0, 'streamed and file audio are one render'


@pytest.mark.parametrize('engine', ENGINES)
def test_a_voice_seed_and_the_lower_rate_reach_the_real_engine(client, engine) -> None:
    assert client.post('/api/admin/runtime/backend', json={'backend': engine, 'wait': True}).status_code == 200
    patched = client.patch(f'/api/admin/voices/{client.voice_id}', json={'seed': 4321, 'engine': engine})
    assert patched.status_code == 200, patched.text
    payload = {'model': 'tts-1', 'input': 'Das ist der zweite Test.', 'voice': 'Anna', 'response_format': 'wav'}
    by_profile, _ = _wav(client.post('/v1/audio/speech', json=payload))
    by_request, _ = _wav(client.post('/v1/audio/speech', json={**payload, 'seed': 4321}))
    assert np.array_equal(by_profile, by_request)

    assert client.put('/api/admin/settings', json={'audio8_output_sample_rate': 24000}).status_code == 200
    try:
        assert client.post('/api/admin/models/reload', json={}).status_code == 200
        low, rate = _wav(client.post('/v1/audio/speech', json=payload))
        assert rate == 24000 and client.get('/api/v1/capabilities').json()['sample_rate'] == 24000
        assert abs(len(low) / rate - len(by_profile) / 44100) < 0.1
    finally:
        client.put('/api/admin/settings', json={'audio8_output_sample_rate': 44100})
        client.post('/api/admin/models/reload', json={})


def test_back_to_mock_frees_the_engine(client) -> None:
    assert client.post('/api/admin/runtime/backend', json={'backend': 'mock', 'wait': True}).status_code == 200
    assert client.app.state.settings.active_model == 'k2-fsa/OmniVoice-AutoVoice'
