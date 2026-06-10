from __future__ import annotations

import asyncio
import io
import tempfile
import time
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from omnivoice_tts_server.api import router_v2
from omnivoice_tts_server.api.router_v2 import _normalize_vllm_base_url
from omnivoice_tts_server.app import create_app
from omnivoice_tts_server.config import Settings
from omnivoice_tts_server.services_v2 import WerBenchmarkService


TEST_MODELS_DIR = Path(__file__).resolve().parents[2] / 'test-models'
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def make_client(**overrides) -> TestClient:
    temp_data_dir = tempfile.TemporaryDirectory()
    app = create_app(
        Settings(
            admin_api_key='test-admin-key',
            runtime_backend='mock',
            models_root_dir=TEST_MODELS_DIR,
            data_dir=Path(temp_data_dir.name),
            **overrides,
        )
    )
    client = TestClient(app)
    client.__enter__()
    client._temp_data_dir = temp_data_dir
    return client


def auth_headers() -> dict[str, str]:
    return {'X-Admin-Key': 'test-admin-key'}


def test_vllm_url_normalization_accepts_base_or_endpoint() -> None:
    expected = 'http://192.168.20.126:8000'
    for value in (
        expected,
        f'{expected}/',
        f'{expected}/v1',
        f'{expected}/v1/models',
        f'{expected}/v1/chat/completions',
        '192.168.20.126:8000/v1/models',
    ):
        assert _normalize_vllm_base_url(value) == expected
        assert WerBenchmarkService._normalize_base_url(value) == expected


def test_wer_sentence_parser_extracts_clean_text_samples() -> None:
    python_dict_payload = (
        "{'chunk_id': 11, 'text_samples': ['Der alte Hund schlief friedlich in der warmen Sonne des Gartens.', "
        "'Wir treffen uns um drei Uhr im Cafe.', 'Im Garten bluehen Tulpen, 尽管 der Morgen kuehl ist.']}"
    )
    parsed = WerBenchmarkService._parse_generated_sentences(python_dict_payload, 4)
    assert parsed == [
        'Der alte Hund schlief friedlich in der warmen Sonne des Gartens.',
        'Wir treffen uns um 3 Uhr im Cafe.',
    ]

    python_list_payload = "['Das Auto braucht dringend eine Inspektion.', 'Sie hat 2 neue Buecher gekauft.']"
    assert WerBenchmarkService._parse_generated_sentences(python_list_payload, 2) == [
        'Das Auto braucht dringend eine Inspektion.',
        'Sie hat 2 neue Buecher gekauft.',
    ]

    wer = WerBenchmarkService._wer('Wir treffen uns um drei Uhr im Cafe.', 'Wir treffen uns um 3 Uhr im Cafe.', 0)
    assert wer['word_errors'] == 0


@pytest.mark.parametrize(
    ('reference', 'hypothesis'),
    [
        (
            'Die Sonne scheint hell, sodass wir heute einen langen Spaziergang machen wollen.',
            'Die Sonne scheint hell, so dass wir heute einen langen Spaziergang machen wollen.',
        ),
        (
            'Der Zug war leider 3 Minuten zu spaet, sodass wir etwas warten mussten.',
            'Der Zug war leider drei Minuten zu spaet, so dass wir etwas warten mussten.',
        ),
        (
            'Wir haben heute Nachmittag ein neues Sofa gekauft, das perfekt in die Wohnzimmer Ecke passt.',
            'Wir haben heute Nachmittag ein neues Sofa gekauft, das perfekt in die Wohnzimmerecke passt.',
        ),
        (
            'Regen und Wind machen heute das Fahrrad fahren ziemlich ungemuetlich und gefaehrlich auf den Strassen.',
            'Regen und Wind machen heute das Fahrradfahren ziemlich ungemuetlich und gefaehrlich auf den Strassen.',
        ),
        (
            'Wir haben gestern Abend gemeinsam leckere Pasta mit Tomatensauce gekocht.',
            'Wir haben gestern Abend gemeinsam leckere Pasta mit Tomatensosse gekocht.',
        ),
    ],
)
def test_wer_tolerates_orthographic_variants(reference: str, hypothesis: str) -> None:
    wer = WerBenchmarkService._wer(reference, hypothesis, 2)
    assert wer['word_errors'] == 0


def make_wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24_000)
        wav_file.writeframes(b'\x00\x00' * 2400)
    return buffer.getvalue()


def wait_for_admin_job_status(client: TestClient, job_id: str, allowed: set[str], timeout: float = 2.0) -> dict:
    deadline = time.time() + timeout
    last_payload: dict | None = None
    while time.time() < deadline:
        response = client.get(f'/api/admin/jobs/{job_id}', headers=auth_headers())
        assert response.status_code == 200
        last_payload = response.json()
        if last_payload['status'] in allowed:
            return last_payload
        time.sleep(0.02)
    raise AssertionError(f"Timed out waiting for {job_id} in {allowed}; last payload: {last_payload}")


def test_health_is_open() -> None:
    client = make_client()
    assert client.get('/health').json() == {'ok': True}
    assert client.get('/api/health').json() == {'ok': True}


def test_frontend_dist_is_served() -> None:
    frontend_dist = PROJECT_ROOT / 'frontend' / 'dist'
    if not (frontend_dist / 'index.html').exists():
        pytest.skip('frontend/dist missing; run npm install && npm run build to test static assets')
    asset_dir = frontend_dist / 'assets'
    asset_file = next(asset_dir.glob('*.*'))

    client = make_client(frontend_dist_dir=frontend_dist)
    assert client.get('/').status_code == 200
    assert client.get('/admin').status_code == 200
    assert client.get('/demo').status_code == 200
    assert client.get(f'/assets/{asset_file.name}').status_code == 200
    assert client.get('/v1/unknown').status_code == 404
    assert client.get('/api/admin/unknown', headers=auth_headers()).status_code == 404


def test_admin_snapshot_and_settings_roundtrip() -> None:
    client = make_client()
    snapshot = client.get('/api/admin/snapshot', headers=auth_headers())
    assert snapshot.status_code == 200
    assert snapshot.json()['settings']['default_model'] == 'k2-fsa/OmniVoice-AutoVoice'

    updated = client.put(
        '/api/admin/settings',
        headers=auth_headers(),
        json={
            'model_directory': str(TEST_MODELS_DIR / 'raid-cache'),
            'default_model': 'k2-fsa/OmniVoice-Base',
            'default_voice': 'voice-test',
            'queue_limit': 12,
            'whisper_base_url': 'http://192.168.0.200:7861',
            'vllm_base_url': 'http://192.168.20.126:8000',
            'vllm_model': 'qwen3-35b',
            'wer_concurrency': 9,
            'wer_transcription_concurrency': 2,
            'num_step': 8,
            'guidance_scale': 1.5,
            'preferred_device': 'cpu',
            'torch_dtype': 'bf16',
            'compile_cudagraphs': True,
            'cudagraph_skip_dynamic_graphs': True,
            'cuda_memory_trim_after_batch': True,
        },
    )
    assert updated.status_code == 200
    payload = updated.json()
    assert payload['default_model'] == 'k2-fsa/OmniVoice-Base'
    assert payload['queue_limit'] == 12
    assert payload['whisper_base_url'] == 'http://192.168.0.200:7861'
    assert payload['whisper_path'] == ''
    assert payload['vllm_base_url'] == 'http://192.168.20.126:8000'
    assert payload['vllm_model'] == 'qwen3-35b'
    assert payload['wer_concurrency'] == 9
    assert payload['wer_transcription_concurrency'] == 2
    assert payload['num_step'] == 8
    assert payload['preferred_device'] == 'cpu'
    assert payload['torch_dtype'] == 'bfloat16'
    assert payload['compile_cudagraphs'] is True
    assert payload['cudagraph_skip_dynamic_graphs'] is True
    assert payload['cuda_memory_trim_after_batch'] is True
    assert payload['model_directory'].endswith('raid-cache')

    devices = client.get('/api/admin/runtime/devices', headers=auth_headers())
    assert devices.status_code == 200
    device_payload = devices.json()
    assert device_payload['preferred_device'] == 'cpu'
    assert any(device['id'] == 'cpu' for device in device_payload['devices'])

    invalid_device = client.put('/api/admin/settings', headers=auth_headers(), json={'preferred_device': 'intel'})
    assert invalid_device.status_code == 400

    vllm_models = client.get('/api/admin/vllm/models?base_url=mock', headers=auth_headers())
    assert vllm_models.status_code == 200
    assert vllm_models.json()['models'] == ['mock-qwen3-35b']


def test_admin_key_metadata_and_rotation() -> None:
    client = make_client()
    metadata = client.get('/api/admin/keys', headers=auth_headers())
    assert metadata.status_code == 200

    rotated = client.post('/api/admin/keys', headers=auth_headers())
    assert rotated.status_code == 200
    payload = rotated.json()
    assert payload['token'].startswith('omnivoice_tts_')
    assert client.get('/api/admin/keys', headers={'X-Admin-Key': 'test-admin-key'}).status_code == 401
    assert client.get('/api/admin/keys', headers={'X-Admin-Key': payload['token']}).status_code == 200


def test_startup_admin_key_expires_without_becoming_persistent() -> None:
    with tempfile.TemporaryDirectory() as data_dir:
        app = create_app(
            Settings(
                admin_api_key='dev-admin-key',
                startup_admin_key='startup-temp-key',
                startup_admin_key_ttl_seconds=1,
                runtime_backend='mock',
                models_root_dir=TEST_MODELS_DIR,
                data_dir=Path(data_dir),
            )
        )
        with TestClient(app) as client:
            assert client.get('/api/admin/keys', headers={'X-Admin-Key': 'startup-temp-key'}).status_code == 200
            assert client.get('/api/admin/keys', headers={'X-Admin-Key': 'dev-admin-key'}).status_code == 401
            time.sleep(1.2)
            assert client.get('/api/admin/keys', headers={'X-Admin-Key': 'startup-temp-key'}).status_code == 401


def test_rotated_admin_key_persists_across_restart() -> None:
    def make_app(data_dir: Path, startup_key: str):
        return create_app(
            Settings(
                admin_api_key='dev-admin-key',
                startup_admin_key=startup_key,
                startup_admin_key_ttl_seconds=60,
                runtime_backend='mock',
                models_root_dir=TEST_MODELS_DIR,
                data_dir=data_dir,
            )
        )

    with tempfile.TemporaryDirectory() as raw_data_dir:
        data_dir = Path(raw_data_dir)
        with TestClient(make_app(data_dir, 'startup-one')) as client:
            rotated = client.post('/api/admin/keys', headers={'X-Admin-Key': 'startup-one'})
            assert rotated.status_code == 200
            token = rotated.json()['token']
            assert token.startswith('omnivoice_tts_')

        with TestClient(make_app(data_dir, 'startup-two')) as client:
            assert client.get('/api/admin/keys', headers={'X-Admin-Key': token}).status_code == 200
            assert client.get('/api/admin/keys', headers={'X-Admin-Key': 'startup-one'}).status_code == 401
            assert client.get('/api/admin/keys', headers={'X-Admin-Key': 'dev-admin-key'}).status_code == 401
            assert client.get('/api/admin/keys', headers={'X-Admin-Key': 'startup-two'}).status_code == 200


def test_speech_returns_audio() -> None:
    client = make_client()
    response = client.post('/v1/audio/speech', json={'input': 'Hallo Welt', 'response_format': 'wav'})
    assert response.status_code == 200
    assert response.headers['content-type'].startswith('audio/wav')
    assert len(response.content) > 44


def test_speech_can_return_mp3(monkeypatch) -> None:
    monkeypatch.setattr(router_v2, 'wav_to_mp3', lambda wav_bytes: b'ID3mock-mp3')
    client = make_client()
    response = client.post('/v1/audio/speech', json={'input': 'Hallo Welt', 'response_format': 'mp3'})
    assert response.status_code == 200
    assert response.headers['content-type'].startswith('audio/mpeg')
    assert response.headers['content-disposition'].endswith('.mp3"')
    assert response.content == b'ID3mock-mp3'


def test_openai_compat_listings_match_open_webui_shapes() -> None:
    client = make_client()

    models = client.get('/v1/models').json()
    assert models['object'] == 'list'
    model_ids = [entry['id'] for entry in models['data']]
    assert 'k2-fsa/OmniVoice-AutoVoice' in model_ids

    audio_models = client.get('/v1/audio/models').json()
    assert [entry['id'] for entry in audio_models['models']] == model_ids

    audio_voices = client.get('/v1/audio/voices').json()
    assert {'id': 'auto voice', 'object': 'voice', 'name': 'Auto Voice', 'source': 'built-in'} in audio_voices['voices']

    voices = client.get('/v1/voices').json()
    assert voices['object'] == 'list'
    assert voices['data'] == voices['voices'] == audio_voices['voices']

    plain_models = client.get('/api/v1/models').json()
    assert [entry['model_id'] for entry in plain_models] == model_ids


def test_openai_compat_speech_accepts_alias_model_and_standard_voice() -> None:
    client = make_client()
    response = client.post('/v1/audio/speech', json={'model': 'tts-1', 'input': 'Hallo Welt', 'voice': 'alloy'})
    assert response.status_code == 200
    assert response.headers['content-type'].startswith('audio/wav')
    assert len(response.content) > 44


def test_openai_compat_speech_rejects_unknown_voice() -> None:
    client = make_client()
    response = client.post('/v1/audio/speech', json={'input': 'Hallo Welt', 'voice': 'does-not-exist'})
    assert response.status_code == 404


def test_openai_compat_speech_routes_custom_profile_to_base_alias() -> None:
    client = make_client()
    created = client.post(
        '/api/admin/voices',
        headers=auth_headers(),
        files={'audio_sample': ('sample.wav', make_wav_bytes(), 'audio/wav')},
        data={'name': 'Open WebUI Voice', 'consent': 'true', 'ref_text': 'Hallo, das ist eine Teststimme.'},
    )
    assert created.status_code == 200
    voice_id = created.json()['voice_id']

    listed = client.get('/v1/audio/voices').json()['voices']
    assert any(entry['id'] == voice_id for entry in listed)

    # Case-insensitive profile name resolves and auto-routes to the Base alias.
    response = client.post('/v1/audio/speech', json={'model': 'tts-1', 'input': 'Hallo Welt', 'voice': 'open webui voice'})
    assert response.status_code == 200
    assert response.headers['content-type'].startswith('audio/wav')
    assert len(response.content) > 44


def test_completed_job_audio_can_be_downloaded_as_mp3(monkeypatch) -> None:
    monkeypatch.setattr(router_v2, 'wav_to_mp3', lambda wav_bytes: b'ID3job-mp3')
    client = make_client()
    create = client.post('/v1/jobs', json={'input': 'Download me'})
    assert create.status_code == 200
    job_id = create.json()['job_id']
    wait_for_admin_job_status(client, job_id, {'completed'})

    admin_audio = client.get(f'/api/admin/jobs/{job_id}/audio?format=mp3', headers=auth_headers())
    assert admin_audio.status_code == 200
    assert admin_audio.headers['content-type'].startswith('audio/mpeg')
    assert admin_audio.content == b'ID3job-mp3'

    public_audio = client.get(f'/v1/jobs/{job_id}/audio?format=mp3')
    assert public_audio.status_code == 200
    assert public_audio.headers['content-type'].startswith('audio/mpeg')
    assert public_audio.content == b'ID3job-mp3'


def test_sentence_chunks_are_reserved_as_real_batches() -> None:
    client = make_client(
        max_batch_size=3,
        sentence_chunking=True,
        short_sentence_merge_max_chars=0,
        following_sentence_merge_min_chars=0,
    )
    response = client.post(
        '/api/v1/synthesize',
        json={
            'input': 'Satz eins ist lang genug. Satz zwei ist lang genug. Satz drei ist lang genug. Satz vier ist lang genug.',
            'response_format': 'wav',
        },
    )
    assert response.status_code == 200
    metrics = response.json()['metrics']
    assert metrics['sentences_total'] == 4
    assert metrics['max_batch_size_seen'] == 3
    assert metrics['batch_count'] == 2


def test_public_stream_returns_ndjson() -> None:
    client = make_client()
    with client.stream('POST', '/api/v1/synthesize/stream', json={'input': 'Stream test'}) as response:
        assert response.status_code == 200
        body = ''.join(response.iter_text())
    assert '"type": "start"' in body
    assert '"type": "chunk"' in body
    assert '"type": "done"' in body


def test_streaming_request_gets_first_sentence_fast_path() -> None:
    client = make_client(
        max_batch_size=4,
        sentence_chunking=True,
        short_sentence_merge_max_chars=0,
        following_sentence_merge_min_chars=0,
    )
    with client.stream(
        'POST',
        '/api/v1/synthesize/stream',
        json={
            'input': 'Erster Satz ist lang genug. Zweiter Satz ist lang genug. Dritter Satz ist lang genug.',
        },
    ) as response:
        assert response.status_code == 200
        body = ''.join(response.iter_text())
    assert '"type": "batch"' in body
    assert '"batch_size": 1' in body


def test_jobs_queue_and_admin_lookup() -> None:
    client = make_client()
    create = client.post('/v1/jobs', json={'input': 'Test job'})
    assert create.status_code == 200
    job_id = create.json()['job_id']
    finished = wait_for_admin_job_status(client, job_id, {'completed'})
    assert finished['metrics']['job_wall_ms'] > 0


def test_delete_job_cancels_active_job() -> None:
    client = make_client()
    original_render_batch = client.app.state.synthesizer.render_batch

    async def slow_render_batch(items):
        await asyncio.sleep(0.2)
        return await original_render_batch(items)

    client.app.state.synthesizer.render_batch = slow_render_batch
    create = client.post('/v1/jobs', json={'input': 'Cancel me while running'})
    assert create.status_code == 200
    job_id = create.json()['job_id']

    time.sleep(0.05)
    deleted = client.delete(f'/api/admin/jobs/{job_id}', headers=auth_headers())
    assert deleted.status_code == 200
    finished = wait_for_admin_job_status(client, job_id, {'cancelled'})
    assert 'Cancelled' in finished['error_message']


def test_voice_crud_requires_consent_and_ref_text() -> None:
    client = make_client()
    missing = client.post(
        '/api/admin/voices',
        headers=auth_headers(),
        files={'audio_sample': ('sample.wav', make_wav_bytes(), 'audio/wav')},
        data={'name': 'No consent', 'consent': 'false', 'ref_text': 'Hallo'},
    )
    assert missing.status_code == 400

    created = client.post(
        '/api/admin/voices',
        headers=auth_headers(),
        files={'audio_sample': ('sample.wav', make_wav_bytes(), 'audio/wav')},
        data={'name': 'Test Voice', 'consent': 'true', 'ref_text': 'Hallo, das ist eine Teststimme.'},
    )
    assert created.status_code == 200
    voice_id = created.json()['voice_id']
    voices = client.get('/api/admin/voices', headers=auth_headers())
    voice_item = next(item for item in voices.json() if item['voice_id'] == voice_id)
    assert voice_item['ref_text'] == 'Hallo, das ist eine Teststimme.'
    assert voice_item['has_audio'] is True
    audio = client.get(f'/api/admin/voices/{voice_id}/audio', headers=auth_headers())
    assert audio.status_code == 200
    assert audio.headers['content-type'].startswith('audio/wav')
    assert len(audio.content) > 44
    assert client.delete(f'/api/admin/voices/{voice_id}', headers=auth_headers()).status_code == 200


def test_model_ops_and_benchmark_smoke() -> None:
    client = make_client()
    models = client.get('/api/admin/models', headers=auth_headers())
    assert models.status_code == 200
    assert models.json()['models'][0]['id'] == 'k2-fsa/OmniVoice-AutoVoice'
    download = client.post('/api/admin/models/download', headers=auth_headers(), json={'model': 'k2-fsa/OmniVoice-AutoVoice'})
    assert download.status_code == 200
    assert download.json()['models'][0]['status'] == 'ready'
    delete_cache = client.post('/api/admin/models/delete', headers=auth_headers(), json={'model': 'k2-fsa/OmniVoice-AutoVoice'})
    assert delete_cache.status_code == 200
    assert delete_cache.json()['ok'] is True
    preload = client.post('/api/admin/models/preload', headers=auth_headers(), json={'model': 'k2-fsa/OmniVoice-AutoVoice'})
    assert preload.status_code == 200
    warmup = client.post('/api/admin/models/warmup', headers=auth_headers(), json={'model': 'k2-fsa/OmniVoice-VoiceDesign', 'task_type': 'VoiceDesign'})
    assert warmup.status_code == 200
    unload = client.post('/api/admin/models/unload', headers=auth_headers(), json={'model': 'k2-fsa/OmniVoice-AutoVoice'})
    assert unload.status_code == 200
    reload = client.post('/api/admin/models/reload', headers=auth_headers(), json={'model': 'k2-fsa/OmniVoice-AutoVoice'})
    assert reload.status_code == 200
    cleanup = client.post('/api/admin/runtime/free-memory', headers=auth_headers())
    assert cleanup.status_code == 200
    assert cleanup.json()['ok'] is True

    created = client.post(
        '/api/admin/benchmarks/runs',
        headers=auth_headers(),
        json={
            'name': 'smoke',
            'text': 'Benchmark smoke',
            'mode': 'iterations',
            'iterations': 1,
            'warmup_iterations': 0,
            'parallel_requests': 4,
            'exclusive': True,
            'cases': [{'label': 'default', 'request': {'response_format': 'wav'}}],
        },
    )
    assert created.status_code == 200
    time.sleep(0.15)
    listed = client.get('/api/admin/benchmarks/runs', headers=auth_headers())
    assert listed.status_code == 200
    assert listed.json()[0]['name'] == 'smoke'
    assert listed.json()[0]['parallel_requests'] == 4

    traffic = client.post(
        '/api/admin/benchmarks/runs',
        headers=auth_headers(),
        json={
            'name': 'traffic smoke',
            'text': 'Erster Satz. Zweiter Satz. Dritter Satz. Vierter Satz.',
            'mode': 'traffic',
            'duration_seconds': 1,
            'requests_per_minute': 120,
            'min_sentences_per_request': 1,
            'max_sentences_per_request': 2,
            'warmup_iterations': 0,
            'exclusive': True,
            'random_seed': 123,
            'cases': [{'label': 'traffic', 'request': {'response_format': 'wav'}}],
        },
    )
    assert traffic.status_code == 200
    assert traffic.json()['total_requests'] == 2
    time.sleep(1.4)
    listed = client.get('/api/admin/benchmarks/runs', headers=auth_headers())
    traffic_run = next(item for item in listed.json() if item['name'] == 'traffic smoke')
    assert traffic_run['status'] == 'completed'
    assert traffic_run['cases'][0]['success_count'] == 2
    assert traffic_run['cases'][0]['ttfa_ms_p99'] is not None

    wer = client.post(
        '/api/admin/wer-benchmarks/runs',
        headers=auth_headers(),
        json={
            'name': 'wer smoke',
            'count': 3,
            'concurrency': 2,
            'transcription_concurrency': 1,
            'vllm_base_url': 'mock',
            'whisper_base_url': 'mock',
            'completion_timeout_seconds': 10,
            'request': {'response_format': 'wav'},
        },
    )
    assert wer.status_code == 200
    deadline = time.time() + 2.0
    wer_run = None
    while time.time() < deadline:
        listed = client.get('/api/admin/wer-benchmarks/runs', headers=auth_headers())
        assert listed.status_code == 200
        wer_run = next(item for item in listed.json() if item['name'] == 'wer smoke')
        if wer_run['status'] == 'completed':
            break
        time.sleep(0.05)
    assert wer_run is not None
    assert wer_run['status'] == 'completed'
    assert wer_run['summary']['success_count'] == 3
    assert wer_run['summary']['wer_avg'] == 0
    assert len(wer_run['results']) == 3
    assert wer_run['transcription_concurrency'] == 1
    assert wer_run['results'][0]['synthesis_ms'] is not None
    assert wer_run['results'][0]['transcription_ms'] is not None
    assert wer_run['sentence_cache_hit'] is False

    wer_again = client.post(
        '/api/admin/wer-benchmarks/runs',
        headers=auth_headers(),
        json={
            'name': 'wer smoke cached',
            'count': 3,
            'concurrency': 2,
            'vllm_base_url': 'mock',
            'whisper_base_url': 'mock',
            'completion_timeout_seconds': 10,
            'request': {'response_format': 'wav', 'metadata': {'num_step': 8}},
        },
    )
    assert wer_again.status_code == 200
    deadline = time.time() + 2.0
    cached_run = None
    while time.time() < deadline:
        listed = client.get('/api/admin/wer-benchmarks/runs', headers=auth_headers())
        assert listed.status_code == 200
        cached_run = next(item for item in listed.json() if item['name'] == 'wer smoke cached')
        if cached_run['status'] == 'completed':
            break
        time.sleep(0.05)
    assert cached_run is not None
    assert cached_run['status'] == 'completed'
    assert cached_run['sentence_cache_hit'] is True
    assert [item['source_text'] for item in cached_run['results']] == [item['source_text'] for item in wer_run['results']]

    wer_range = client.post(
        '/api/admin/wer-benchmarks/runs',
        headers=auth_headers(),
        json={
            'name': 'wer seed range',
            'count': 2,
            'concurrency': 2,
            'transcription_concurrency': 2,
            'vllm_base_url': 'mock',
            'whisper_base_url': 'mock',
            'completion_timeout_seconds': 10,
            'random_seed': 5,
            'seed_range': 2,
            'request': {'response_format': 'wav'},
        },
    )
    assert wer_range.status_code == 200
    deadline = time.time() + 2.0
    range_run = None
    while time.time() < deadline:
        listed = client.get('/api/admin/wer-benchmarks/runs', headers=auth_headers())
        assert listed.status_code == 200
        range_run = next(item for item in listed.json() if item['name'] == 'wer seed range')
        if range_run['status'] == 'completed':
            break
        time.sleep(0.05)
    assert range_run is not None
    assert range_run['status'] == 'completed'
    assert range_run['summary']['total'] == 6
    assert range_run['summary']['success_count'] == 6
    assert [item['seed'] for item in range_run['seed_leaderboard']] == [5, 6, 7]
    assert len(range_run['results']) == 6
