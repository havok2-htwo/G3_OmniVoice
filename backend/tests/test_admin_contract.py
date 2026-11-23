"""Shared G3 admin contract (STANDARD.md sections 4 and 5): optional API keys, admin auth hardening, /api/admin/perf
and the GPU identity (cuda index -> nvidia-smi card by UUID)."""
from __future__ import annotations

import json
import tempfile
import time
from collections import deque
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from omnivoice_tts_server import g3_gpu, runtime_v2
from omnivoice_tts_server.api import router_v2
from omnivoice_tts_server.app import create_app
from omnivoice_tts_server.config import Settings
from omnivoice_tts_server.domain import state as state_module
from omnivoice_tts_server.domain.state import InMemoryStore, api_key_hint

from test_app import TEST_MODELS_DIR, make_client

SPEECH = {'model': 'k2-fsa/OmniVoice-AutoVoice', 'input': 'Hallo Welt', 'response_format': 'wav'}
READ_ROUTES = ['/v1/models', '/api/v1/models', '/v1/audio/models', '/v1/voices', '/api/v1/voices',
               '/v1/audio/voices', '/v1/capabilities', '/api/v1/capabilities']


class Anonymous:
    """The same TestClient (same event loop) without the admin session cookie."""

    def __init__(self, client: TestClient) -> None:
        self.client = client

    def _call(self, method: str, *args, **kwargs):
        saved = dict(self.client.cookies)
        self.client.cookies.clear()
        try:
            return getattr(self.client, method)(*args, **kwargs)
        finally:
            for name, value in saved.items():
                self.client.cookies.set(name, value)

    def get(self, *args, **kwargs):
        return self._call('get', *args, **kwargs)

    def post(self, *args, **kwargs):
        return self._call('post', *args, **kwargs)


def _with_key(client: TestClient) -> tuple[Anonymous, dict]:
    created = client.post('/api/admin/api-keys', json={'alias': 'contract'})
    assert created.status_code == 200, created.text
    return Anonymous(client), created.json()


# ---------- API keys ----------

def test_bearer_and_x_api_key_are_both_accepted() -> None:
    client = make_client()
    anon, key = _with_key(client)
    assert anon.post('/v1/audio/speech', json=SPEECH, headers={'Authorization': f"Bearer {key['token']}"}).status_code == 200
    assert anon.post('/v1/audio/speech', json=SPEECH, headers={'X-API-Key': key['token']}).status_code == 200
    denied = anon.post('/v1/audio/speech', json=SPEECH)
    assert denied.status_code == 401
    assert denied.headers.get('www-authenticate') == 'Bearer'
    assert 'Authorization: Bearer' in denied.json()['detail']
    assert anon.post('/v1/audio/speech', json=SPEECH, headers={'Authorization': 'Bearer wrong'}).status_code == 401


def test_admin_session_may_use_the_public_api_but_an_explicit_bad_key_loses() -> None:
    client = make_client()
    _with_key(client)
    # Quick Synthesis in the admin UI calls the public stream route with the session cookie only.
    assert client.post('/v1/audio/speech', json=SPEECH).status_code == 200
    stream = client.post('/api/v1/synthesize/stream', json={**SPEECH, 'stream': True})
    assert stream.status_code == 200
    assert client.post('/v1/audio/speech', json=SPEECH, headers={'X-API-Key': 'wrong'}).status_code == 401


def test_session_usage_is_not_counted_on_any_key() -> None:
    client = make_client()
    _, key = _with_key(client)
    assert client.post('/v1/audio/speech', json=SPEECH).status_code == 200
    row = next(r for r in client.get('/api/admin/api-keys').json()['keys'] if r['id'] == key['id'])
    assert row['usage']['request_count'] == 0


@pytest.mark.parametrize('path', READ_ROUTES)
def test_read_routes_are_open_without_keys_and_protected_with_keys(path: str) -> None:
    client = make_client()
    anon = Anonymous(client)
    assert anon.get(path).status_code == 200
    _, key = _with_key(client)
    assert anon.get(path).status_code == 401
    assert anon.get(path, headers={'Authorization': f"Bearer {key['token']}"}).status_code == 200
    assert client.get(path).status_code == 200  # admin session


def test_job_audio_route_is_protected_once_a_key_exists() -> None:
    client = make_client()
    anon, key = _with_key(client)
    headers = {'X-API-Key': key['token']}
    job = anon.post('/v1/jobs', json=SPEECH, headers=headers)
    assert job.status_code == 200, job.text
    job_id = job.json()['job_id']
    for _ in range(100):
        if anon.get(f'/v1/jobs/{job_id}/audio', headers=headers).status_code == 200:
            break
        time.sleep(0.05)
    assert anon.get(f'/v1/jobs/{job_id}/audio', headers=headers).status_code == 200
    assert anon.get(f'/v1/jobs/{job_id}/audio').status_code == 401


def test_health_and_auth_setup_stay_public() -> None:
    client = make_client()
    anon, _ = _with_key(client)
    assert anon.get('/health').status_code == 200
    assert anon.get('/api/admin/auth/setup').json() == {'setup_pending': False}


def test_key_hint_is_stored_listed_and_never_the_hash() -> None:
    client = make_client()
    _, key = _with_key(client)
    token = key['token']
    assert key['hint'] == api_key_hint(token)
    assert key['hint'].startswith('omnivoice_tts_') and '…' in key['hint']
    assert key['hint'].endswith(token[-4:])
    rows = client.get('/api/admin/api-keys').json()['keys']
    assert rows[0]['hint'] == key['hint']
    assert all('hash' not in name for row in rows for name in row)
    assert all('token' not in row for row in rows)


def test_alias_is_trimmed_and_capped() -> None:
    client = make_client()
    created = client.post('/api/admin/api-keys', json={'alias': '  ' + 'x' * 80 + '  '}).json()
    assert created['alias'] == 'x' * 64


def test_secrets_file_is_written_atomically_with_private_mode(tmp_path: Path) -> None:
    store = InMemoryStore(max_queue_size=4)
    store.load_secrets(tmp_path)
    store.create_api_key('a')
    path = tmp_path / 'server_secrets.json'
    payload = json.loads(path.read_text(encoding='utf-8'))
    assert payload['version'] == 2
    assert list(tmp_path.glob('*.tmp')) == []
    if hasattr(path, 'stat') and __import__('os').name == 'posix':
        assert path.stat().st_mode & 0o777 == 0o600


def test_legacy_key_without_hint_loads_and_lists_none(tmp_path: Path) -> None:
    store = InMemoryStore(max_queue_size=4)
    store.load_secrets(tmp_path)
    created = store.create_api_key('old')
    path = tmp_path / 'server_secrets.json'
    payload = json.loads(path.read_text(encoding='utf-8'))
    payload['api_keys'][created['id']].pop('hint')
    payload['api_keys'][created['id']]['unknown_future_field'] = 1
    path.write_text(json.dumps(payload), encoding='utf-8')
    again = InMemoryStore(max_queue_size=4)
    again.load_secrets(tmp_path)
    assert again.list_api_keys()[0]['hint'] is None
    assert again.match_api_key(created['token']) == created['id']


def test_unreadable_secrets_fail_closed_and_are_not_overwritten() -> None:
    with tempfile.TemporaryDirectory() as raw:
        data_dir = Path(raw)
        broken = '{"version": 2, "api_keys": {'  # truncated JSON
        (data_dir / 'server_secrets.json').write_text(broken, encoding='utf-8')
        app = create_app(Settings(runtime_backend='mock', models_root_dir=TEST_MODELS_DIR, data_dir=data_dir,
                                  active_model='k2-fsa/OmniVoice-AutoVoice', default_voice='Auto Voice'))
        with TestClient(app) as client:
            assert client.post('/v1/audio/speech', json=SPEECH).status_code == 503
            assert client.get('/v1/models').status_code == 503
            assert client.get('/health').status_code == 200
        assert (data_dir / 'server_secrets.json').read_text(encoding='utf-8') == broken


def test_usage_is_flushed_debounced_not_per_request(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(state_module, 'API_KEY_USAGE_FLUSH_SECONDS', 0.2)
    store = InMemoryStore(max_queue_size=4)
    store.load_secrets(tmp_path)
    created = store.create_api_key('u')
    writes: list[int] = []
    original = store.save_secrets
    monkeypatch.setattr(store, 'save_secrets', lambda d: (original(d), writes.append(1)))
    for _ in range(5):
        store.record_api_key_usage(created['id'], 1.5)
    assert writes == []
    for _ in range(50):
        if writes:
            break
        time.sleep(0.05)
    assert len(writes) == 1
    usage = json.loads((tmp_path / 'server_secrets.json').read_text(encoding='utf-8'))['api_keys'][created['id']]['usage']
    assert usage['request_count'] == 5
    assert usage['total_seconds_processed'] == 7.5


# ---------- admin auth hardening ----------

def test_session_cookie_is_strict_and_httponly() -> None:
    with tempfile.TemporaryDirectory() as raw:
        app = create_app(Settings(runtime_backend='mock', models_root_dir=TEST_MODELS_DIR, data_dir=Path(raw)))
        with TestClient(app) as client:
            login = client.post('/api/admin/auth/login', json={'username': 'admin', 'password': 'admin'})
            cookie = login.headers['set-cookie'].lower()
            assert 'samesite=strict' in cookie and 'httponly' in cookie
            assert client.get('/api/admin/auth/setup').json() == {'setup_pending': True}


def test_new_password_needs_eight_characters() -> None:
    with tempfile.TemporaryDirectory() as raw:
        app = create_app(Settings(runtime_backend='mock', models_root_dir=TEST_MODELS_DIR, data_dir=Path(raw)))
        with TestClient(app) as client:
            client.post('/api/admin/auth/login', json={'username': 'admin', 'password': 'admin'})
            short = client.post('/api/admin/auth/change-password', json={'current_password': 'admin', 'new_password': 'abcdefg'})
            assert short.status_code == 400
            assert '8 characters' in short.json()['detail']
            ok = client.post('/api/admin/auth/change-password', json={'current_password': 'admin', 'new_password': 'abcdefgh'})
            assert ok.status_code == 200
            assert client.get('/api/admin/auth/setup').json() == {'setup_pending': False}


def test_login_is_rate_limited_after_five_failures() -> None:
    with tempfile.TemporaryDirectory() as raw:
        app = create_app(Settings(runtime_backend='mock', models_root_dir=TEST_MODELS_DIR, data_dir=Path(raw)))
        with TestClient(app) as client:
            for _ in range(5):
                assert client.post('/api/admin/auth/login', json={'username': 'admin', 'password': 'nope'}).status_code == 401
            blocked = client.post('/api/admin/auth/login', json={'username': 'admin', 'password': 'admin'})
            assert blocked.status_code == 429


def test_cross_origin_admin_writes_are_refused() -> None:
    client = make_client()
    evil = client.post('/api/admin/api-keys', json={'alias': 'x'}, headers={'Origin': 'http://evil.example'})
    assert evil.status_code == 403
    same = client.post('/api/admin/api-keys', json={'alias': 'x'}, headers={'Origin': 'http://testserver'})
    assert same.status_code == 200
    # Reads are not origin-checked.
    assert client.get('/api/admin/api-keys', headers={'Origin': 'http://evil.example'}).status_code == 200


# ---------- performance contract ----------

def test_perf_metrics_units_and_idle_values() -> None:
    store = InMemoryStore(max_queue_size=4)
    now = 1000.0
    metrics = {m['key']: m for m in router_v2.perf_metrics(store, now)}
    assert metrics['throughput']['unit'] == 'x' and metrics['throughput']['value'] == 0.0
    assert metrics['throughput']['window_s'] == 30
    assert metrics['queue']['integer'] is True and metrics['queue']['value'] == 0
    assert metrics['latency']['unit'] == 'ms'
    assert metrics['latency']['value'] is None  # no job in the window: unknown, not 0 ms


def test_perf_metrics_are_sliding_window_rates() -> None:
    store = InMemoryStore(max_queue_size=4)
    store.audio_output_events = deque([(990.0, 6.0), (995.0, 9.0), (900.0, 300.0)])
    store.ttfa_events = deque([(990.0, 400.0), (995.0, 200.0)])
    metrics = {m['key']: m for m in router_v2.perf_metrics(store, 1000.0)}
    assert metrics['throughput']['value'] == pytest.approx(15.0 / 30.0, abs=0.01)  # the old 300 s is outside 30 s
    assert metrics['latency']['value'] == pytest.approx(300.0)
    later = {m['key']: m for m in router_v2.perf_metrics(store, 1100.0)}
    assert later['throughput']['value'] == 0.0
    assert later['latency']['value'] is None


FAKE_GPUS = [
    {'nvml_index': 0, 'uuid': 'GPU-aaaa', 'pci_bus_id': '0:1', 'name': 'NVIDIA GeForce RTX 2070 SUPER', 'util_pct': 3.0,
     'mem_total_mib': 8192, 'mem_used_mib': 341, 'process_mib': 0.0, 'temp_c': 40, 'power_w': 20.0,
     'power_limit_w': 215.0, 'cuda_index': None, 'used_by_service': False},
    {'nvml_index': 1, 'uuid': 'GPU-bbbb', 'pci_bus_id': '0:2', 'name': 'NVIDIA GeForce RTX 3090', 'util_pct': 37.0,
     'mem_total_mib': 24576, 'mem_used_mib': 17978, 'process_mib': 2426.0, 'temp_c': 54, 'power_w': 112.4,
     'power_limit_w': 350.0, 'cuda_index': 0, 'used_by_service': True},
]


def test_perf_endpoint_returns_schema_1_and_needs_admin(monkeypatch) -> None:
    monkeypatch.setattr(g3_gpu, 'gpu_snapshot', lambda: [dict(g) for g in FAKE_GPUS])
    client = make_client()
    body = client.get('/api/admin/perf').json()
    assert body['schema'] == 1 and body['service'] == 'G3 OmniVoice'
    assert [g['nvml_index'] for g in body['gpus'] if g['used_by_service']] == [1]
    assert [m['key'] for m in body['metrics']] == ['throughput', 'queue', 'latency']
    assert Anonymous(client).get('/api/admin/perf').status_code == 401


def test_cuda0_maps_to_the_card_with_the_same_uuid(monkeypatch) -> None:
    # CUDA_VISIBLE_DEVICES=<3090 uuid>: cuda:0 is nvidia-smi GPU 1, not GPU 0.
    monkeypatch.setattr(g3_gpu, 'gpu_snapshot', lambda: [dict(g) for g in FAKE_GPUS])
    stats = runtime_v2.query_nvidia_smi('cuda:0')
    assert stats['name'] == 'NVIDIA GeForce RTX 3090'
    assert stats['memory_total_mb'] == 24576
    assert stats['memory_used_mb'] == 17978
    assert stats['process_memory_mb'] == 2426
    assert stats['nvml_index'] == 1


def test_device_label_uses_the_nvidia_smi_number(monkeypatch) -> None:
    monkeypatch.setattr(g3_gpu, 'gpu_snapshot', lambda: [dict(g) for g in FAKE_GPUS])
    assert runtime_v2._gpu_device_label(0, 'NVIDIA GeForce RTX 3090', 24576) == 'GPU 1 · GeForce RTX 3090 · 24 GB (cuda:0)'
