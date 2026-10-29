"""Engine registry, persisted runtime_backend and the live engine switch (with a fake Ming synthesizer)."""
from __future__ import annotations

import json
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from omnivoice_tts_server import engines
from omnivoice_tts_server.config import (
    RUNTIME_SETTINGS_FIELDS,
    Settings,
    load_runtime_settings,
    save_runtime_settings,
    set_persisted_overrides,
)
from omnivoice_tts_server.domain.state import InMemoryStore
from omnivoice_tts_server.runtime_v2 import MockSynthesizer, build_synthesizer

from test_app import make_client

ALIAS = 'inclusionAI/Ming-3.0-tts-0.5B-Base'


class FakeMingSynthesizer(MockSynthesizer):
    """Stands in for ming_runtime.MingSynthesizer: tone output, records the lifecycle calls."""

    supports_native_streaming = False
    supports_concurrent_batches = True
    uses_voice_profiles = True
    instances: list['FakeMingSynthesizer'] = []
    fail_build = False

    def __init__(self, settings, store) -> None:
        if FakeMingSynthesizer.fail_build:
            raise RuntimeError('ming deps broken')
        super().__init__(sample_rate=24_000)
        self.settings = settings
        self.device = settings.preferred_device
        self.slots = settings.ming_max_slots
        self.ensure_calls: list[str | None] = []
        self.rendered: list[str] = []
        self.unloaded = False
        FakeMingSynthesizer.instances.append(self)

    async def ensure_model(self, requested_model):
        self.ensure_calls.append(requested_model)
        return ALIAS, 0

    async def render_batch(self, items, trim_after: bool = True):
        self.rendered.extend(item.text for item in items)
        return await super().render_batch(items)

    async def unload(self):
        self.unloaded = True
        return ALIAS, 0


@pytest.fixture
def fake_ming(monkeypatch, tmp_path):
    module = types.ModuleType('omnivoice_tts_server.ming_runtime')
    module.MING_BASE_ALIAS = ALIAS
    module.MingSynthesizer = FakeMingSynthesizer
    monkeypatch.setitem(sys.modules, 'omnivoice_tts_server.ming_runtime', module)
    monkeypatch.setattr(engines, '_module_installed', lambda name: True)
    FakeMingSynthesizer.instances = []
    FakeMingSynthesizer.fail_build = False
    model_dir = tmp_path / 'Ming-3.0-tts-0.5b'
    model_dir.mkdir()
    (model_dir / 'config.json').write_text('{}', encoding='utf-8')
    (model_dir / 'model.safetensors').write_bytes(b'')
    monkeypatch.delenv('OMNIVOICE_TTS_RUNTIME_BACKEND', raising=False)
    return model_dir


def _client(model_dir: Path):
    return make_client(ming_model_dir=model_dir)


def _close(client) -> None:
    client.__exit__(None, None, None)
    client._temp_data_dir.cleanup()


def _switch(client, backend: str, **extra) -> dict:
    response = client.post('/api/admin/runtime/backend', json={'backend': backend, 'wait': True, **extra})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body['active'] == backend, body
    return body


def _saved(client) -> dict:
    data_dir = Path(client._temp_data_dir.name)
    return json.loads((data_dir / 'runtime_settings.json').read_text(encoding='utf-8'))


# --- settings persistence -----------------------------------------------------------------------------------------


def test_ming_settings_are_persisted_runtime_settings() -> None:
    for name in ('runtime_backend', 'ming_model_id', 'ming_model_dir', 'ming_max_slots', 'ming_cuda_graphs',
                 'ming_output_sample_rate', 'ming_first_chunk_latents', 'ming_max_chunk_latents'):
        assert name in RUNTIME_SETTINGS_FIELDS
    defaults = Settings()
    assert (defaults.ming_max_slots, defaults.ming_cuda_graphs, defaults.ming_output_sample_rate) == (16, True, 24000)
    assert (defaults.ming_first_chunk_latents, defaults.ming_max_chunk_latents) == (2, 8)
    assert defaults.ming_model_dir.name == 'Ming-3.0-tts-0.5b'
    assert defaults.ming_model_id == 'inclusionAI/Ming-3.0-tts-0.5B'


@pytest.mark.parametrize(
    ('environ', 'saved', 'expected'),
    [
        ({}, 'ming', 'ming'),
        ({'OMNIVOICE_TTS_RUNTIME_BACKEND': 'omnivoice'}, 'ming', 'omnivoice'),
        ({}, 'tortoise', 'omnivoice'),
        ({}, None, 'omnivoice'),
    ],
)
def test_runtime_backend_precedence_env_then_file_then_default(tmp_path, environ, saved, expected) -> None:
    payload = {'max_batch_size': 3}
    if saved is not None:
        payload['runtime_backend'] = saved
    (tmp_path / 'runtime_settings.json').write_text(json.dumps(payload), encoding='utf-8')
    settings = Settings(data_dir=tmp_path, runtime_backend=environ.get('OMNIVOICE_TTS_RUNTIME_BACKEND', 'omnivoice'))
    load_runtime_settings(settings, environ=environ)
    assert settings.runtime_backend == expected
    assert settings.max_batch_size == 3


def test_ming_model_dir_round_trips_as_path(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path, ming_model_dir=tmp_path / 'm')
    save_runtime_settings(settings)
    reloaded = Settings(data_dir=tmp_path)
    load_runtime_settings(reloaded, environ={})
    assert reloaded.ming_model_dir == tmp_path / 'm'


def test_persisted_overrides_keep_a_foreign_active_model_off_disk(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path, active_model=ALIAS)
    set_persisted_overrides(settings, {'active_model': 'k2-fsa/OmniVoice-Base'})
    save_runtime_settings(settings)
    saved = json.loads((tmp_path / 'runtime_settings.json').read_text(encoding='utf-8'))
    assert saved['active_model'] == 'k2-fsa/OmniVoice-Base'
    assert settings.active_model == ALIAS, 'memory keeps the engine alias'


def test_fix_active_model_resets_an_unknown_alias() -> None:
    settings = Settings(active_model='gradium/phonon-40m-multilingual')
    assert engines.fix_active_model(settings) is True
    assert settings.active_model == 'k2-fsa/OmniVoice-Base'
    assert engines.fix_active_model(settings) is False


# --- dispatch and registry ----------------------------------------------------------------------------------------


def test_build_synthesizer_ming_branch_exposes_the_base_alias(fake_ming) -> None:
    settings = Settings(runtime_backend='ming', ming_model_dir=fake_ming, preferred_device='cuda:1', ming_max_slots=7)
    store = InMemoryStore(max_queue_size=4)
    synthesizer = build_synthesizer(settings, store)
    assert isinstance(synthesizer, FakeMingSynthesizer)
    assert settings.supported_models == [ALIAS] and ALIAS.endswith('Base')
    assert settings.active_model == ALIAS and store.active_model == ALIAS
    assert (synthesizer.device, synthesizer.slots) == ('cuda:1', 7), 'Ming reads preferred_device and its settings'


def test_engine_registry_reports_install_state(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(engines, '_module_installed', lambda name: name not in {'ptts'})
    settings = Settings(runtime_backend='mock', ming_model_dir=tmp_path / 'missing', models_root_dir=tmp_path,
                        allow_model_downloads=False)
    by_id = {info.id: info for info in engines.list_engines(settings)}
    assert list(by_id) == ['omnivoice', 'ming', 'phonon', 'mock']
    assert by_id['mock'].active and by_id['mock'].test_only and by_id['mock'].status == 'active'
    assert by_id['phonon'].status == 'missing_package' and not by_id['phonon'].available
    assert by_id['ming'].status == 'missing_model' and 'hf download inclusionAI/Ming-3.0-tts-0.5B' in by_id['ming'].detail
    assert by_id['omnivoice'].status == 'missing_model'
    settings.allow_model_downloads = True
    assert engines.engine_info(engines.ENGINES['omnivoice'], settings).status == 'downloadable'


def test_foreign_model_ids_map_to_the_active_engine() -> None:
    settings = Settings(runtime_backend='ming', supported_models=[ALIAS], active_model=ALIAS)
    assert engines.map_request_model(settings, 'k2-fsa/OmniVoice-Base') is None
    assert engines.map_request_model(settings, 'tts-1') is None
    assert engines.map_request_model(settings, ALIAS.lower()) == ALIAS
    assert engines.map_request_model(settings, None) is None
    omni = Settings(runtime_backend='omnivoice')
    assert engines.map_request_model(omni, 'something/else') == 'something/else', 'OmniVoice keeps every id'


# --- live switch through the admin API ----------------------------------------------------------------------------


def test_engines_endpoint_lists_every_engine(fake_ming) -> None:
    client = _client(fake_ming)
    try:
        body = client.get('/api/admin/runtime/engines').json()
        assert body['active'] == 'mock'
        assert [engine['id'] for engine in body['engines']] == ['omnivoice', 'ming', 'phonon', 'mock']
        assert body['switch']['state'] == 'idle'
        ming = next(engine for engine in body['engines'] if engine['id'] == 'ming')
        assert ming['available'] and ming['native_streaming'] and ming['uses_voice_profiles']
    finally:
        _close(client)


def test_live_switch_to_ming_and_back(fake_ming) -> None:
    client = _client(fake_ming)
    try:
        before = client.get('/api/admin/settings').json()
        body = _switch(client, 'ming')
        assert body['switch']['state'] == 'ready', body['switch']
        assert body['switch']['from_backend'] == 'mock' and body['switch']['to_backend'] == 'ming'
        fake = FakeMingSynthesizer.instances[-1]
        assert fake.ensure_calls == [ALIAS], 'the new engine preloads its model'

        settings = client.get('/api/admin/settings').json()
        assert settings['runtime_backend'] == 'ming' and settings['default_model'] == ALIAS
        assert [model['model_id'] for model in client.get('/api/v1/models').json()] == [ALIAS]
        assert [entry['id'] for entry in client.get('/v1/models').json()['data']] == [ALIAS]

        # A client that still names an OmniVoice model is served by Ming without a change.
        response = client.post(
            '/api/v1/synthesize', json={'input': 'Hallo Ming.', 'model': 'k2-fsa/OmniVoice-Base', 'voice': 'Auto Voice'}
        )
        assert response.status_code == 200, response.text
        assert fake.rendered == ['Hallo Ming.']
        assert response.json()['sample_rate'] == 24_000

        saved = _saved(client)
        assert saved['runtime_backend'] == 'ming'
        assert saved['active_model'] == before['default_model'], 'the Ming alias never reaches the saved settings'

        # The settings PUT in Ming mode keeps the leak closed too.
        assert client.put('/api/admin/settings', json={'ming_max_slots': 9}).status_code == 200
        saved = _saved(client)
        assert saved['ming_max_slots'] == 9 and saved['active_model'] == before['default_model']

        # Finetuning trains OmniVoice: refused cleanly while Ming runs.
        assert client.post('/api/admin/finetune/train', json={}).status_code == 409
        assert client.post('/api/admin/finetune/generate', json={}).status_code == 409

        _switch(client, 'mock')
        assert fake.unloaded is True
        restored = client.get('/api/admin/settings').json()
        assert restored['runtime_backend'] == 'mock'
        assert restored['default_model'] == before['default_model']
        models = [model['model_id'] for model in client.get('/api/v1/models').json()]
        assert ALIAS not in models and before['default_model'] in models
        assert _saved(client)['runtime_backend'] == 'mock'
    finally:
        _close(client)


def test_settings_put_switches_the_engine(fake_ming) -> None:
    client = _client(fake_ming)
    try:
        response = client.put('/api/admin/settings', json={'runtime_backend': 'ming', 'ming_max_slots': 12})
        assert response.status_code == 200, response.text
        deadline = time.time() + 10
        while time.time() < deadline:
            body = client.get('/api/admin/runtime/engines').json()
            if body['switch']['state'] in {'ready', 'failed'}:
                break
            time.sleep(0.05)
        assert body['active'] == 'ming' and body['switch']['state'] == 'ready'
        assert FakeMingSynthesizer.instances[-1].slots == 12, 'the Ming knobs of the same PUT apply to the new engine'
    finally:
        _close(client)


def test_unavailable_engine_is_rejected_before_anything_changes(fake_ming, monkeypatch) -> None:
    client = _client(fake_ming)
    try:
        monkeypatch.setattr(engines, '_module_installed', lambda name: False)
        response = client.post('/api/admin/runtime/backend', json={'backend': 'phonon'})
        assert response.status_code == 400 and 'not available' in response.json()['detail']
        response = client.put('/api/admin/settings', json={'runtime_backend': 'ming', 'max_batch_size': 5})
        assert response.status_code == 400
        assert client.get('/api/admin/settings').json()['max_batch_size'] != 5
        assert client.post('/api/admin/runtime/backend', json={'backend': 'tortoise'}).status_code == 422
    finally:
        _close(client)


def test_failed_build_keeps_the_old_engine(fake_ming) -> None:
    client = _client(fake_ming)
    try:
        FakeMingSynthesizer.fail_build = True
        body = _switch_expect(client, 'ming')
        assert body['active'] == 'mock'
        assert body['switch']['state'] == 'failed' and 'ming deps broken' in body['switch']['error']
        response = client.post('/api/v1/synthesize', json={'input': 'Noch da.', 'voice': 'Auto Voice'})
        assert response.status_code == 200, response.text
    finally:
        _close(client)


def _switch_expect(client, backend: str) -> dict:
    response = client.post('/api/admin/runtime/backend', json={'backend': backend, 'wait': True})
    assert response.status_code == 200, response.text
    return response.json()


class GatedSynthesizer(MockSynthesizer):
    """The old engine: render_batch blocks until the test opens the gate."""

    def __init__(self) -> None:
        super().__init__(sample_rate=24_000)
        self.gate = threading.Event()
        self.started = threading.Event()
        self.rendered: list[str] = []

    async def ensure_model(self, requested_model):
        return requested_model or 'k2-fsa/OmniVoice-AutoVoice', 0

    async def render_batch(self, items):
        import asyncio

        self.started.set()
        await asyncio.to_thread(self.gate.wait, 10)
        self.rendered.extend(item.text for item in items)
        return await super().render_batch(items)

    async def unload(self):
        return 'k2-fsa/OmniVoice-AutoVoice', 0


def test_running_job_finishes_on_the_old_engine_and_queued_job_moves(fake_ming) -> None:
    client = _client(fake_ming)
    try:
        old = GatedSynthesizer()
        state = client.app.state
        state.synthesizer = old
        state.queue_service.synthesizer = old
        first = client.post('/v1/jobs', json={'input': 'Erster Satz.', 'voice': 'Auto Voice'}).json()['job_id']
        assert old.started.wait(5)
        second = client.post('/v1/jobs', json={'input': 'Zweiter Satz.', 'voice': 'Auto Voice'}).json()['job_id']

        response = client.post('/api/admin/runtime/backend', json={'backend': 'ming'})
        assert response.status_code == 200
        assert response.json()['switch']['state'] == 'draining'
        # A second switch while one runs is refused, and so is a settings save.
        assert client.post('/api/admin/runtime/backend', json={'backend': 'mock'}).status_code == 409
        assert client.put('/api/admin/settings', json={'max_batch_size': 2}).status_code == 409

        old.gate.set()
        deadline = time.time() + 10
        while time.time() < deadline:
            body = client.get('/api/admin/runtime/engines').json()
            if body['switch']['state'] in {'ready', 'failed'}:
                break
            time.sleep(0.05)
        assert body['switch']['state'] == 'ready', body['switch']
        assert body['switch']['requeued_jobs'] == 1 and body['switch']['cancelled_jobs'] == 0

        for job_id in (first, second):
            deadline = time.time() + 10
            while time.time() < deadline and client.get(f'/api/admin/jobs/{job_id}').json()['status'] != 'completed':
                time.sleep(0.05)
        assert old.rendered == ['Erster Satz.']
        assert FakeMingSynthesizer.instances[-1].rendered == ['Zweiter Satz.']
    finally:
        _close(client)


def test_drain_timeout_cancels_the_running_job(fake_ming, monkeypatch) -> None:
    monkeypatch.setattr(engines, 'CANCEL_GRACE_S', 0.2)
    client = _client(fake_ming)
    try:
        old = GatedSynthesizer()
        state = client.app.state
        state.synthesizer = old
        state.queue_service.synthesizer = old
        job_id = client.post('/v1/jobs', json={'input': 'Haengt.', 'voice': 'Auto Voice'}).json()['job_id']
        assert old.started.wait(5)
        response = client.post('/api/admin/runtime/backend', json={'backend': 'ming', 'drain_timeout_s': 0.1})
        assert response.status_code == 200
        time.sleep(0.6)
        old.gate.set()
        deadline = time.time() + 10
        while time.time() < deadline:
            body = client.get('/api/admin/runtime/engines').json()
            if body['switch']['state'] in {'ready', 'failed'}:
                break
            time.sleep(0.05)
        assert body['switch']['state'] == 'ready' and body['switch']['cancelled_jobs'] == 1
        assert client.get(f'/api/admin/jobs/{job_id}').json()['status'] == 'cancelled'
    finally:
        _close(client)


def test_phonon_mode_never_wipes_the_saved_voice_profiles(tmp_path, monkeypatch) -> None:
    """Regression: shutdown saved the (unloaded, empty) profile set in Phonon mode."""
    from fastapi.testclient import TestClient

    from omnivoice_tts_server.app import create_app

    (tmp_path / 'voice_profiles.json').write_text(
        json.dumps({'v1': {'voice_id': 'v1', 'name': 'Keep', 'source': 'upload', 'created_at': '2026-10-01T00:00:00+00:00'}}),
        encoding='utf-8',
    )

    class FakePhonon(MockSynthesizer):
        def __init__(self, settings, store) -> None:
            super().__init__(sample_rate=24_000)

        def load_now(self) -> None:
            pass

        def voice_names(self) -> list[str]:
            return ['Catalog']

    module = types.ModuleType('omnivoice_tts_server.phonon_runtime')
    module.PhononSynthesizer = FakePhonon
    monkeypatch.setitem(sys.modules, 'omnivoice_tts_server.phonon_runtime', module)
    app = create_app(Settings(runtime_backend='phonon', data_dir=tmp_path))
    with TestClient(app) as client:
        assert app.state.store.voice_profiles == {}
        assert 'Catalog' in app.state.settings.built_in_voices
        login = client.post('/api/admin/auth/login', json={'username': 'admin', 'password': 'admin'})
        assert login.status_code == 200
        client.post('/api/admin/auth/change-password', json={'current_password': 'admin', 'new_password': 'pw-for-tests-123'})
        assert client.delete('/api/admin/voices/v1').status_code == 409, 'profiles are not editable in Phonon mode'
    saved = json.loads((tmp_path / 'voice_profiles.json').read_text(encoding='utf-8'))
    assert 'v1' in saved


def test_empty_backend_env_does_not_pin_the_engine(tmp_path) -> None:
    """`docker run -e OMNIVOICE_TTS_RUNTIME_BACKEND=`: empty means unpinned, as env_backend_override reports it."""
    (tmp_path / 'runtime_settings.json').write_text(json.dumps({'runtime_backend': 'ming'}), encoding='utf-8')
    settings = Settings(data_dir=tmp_path, runtime_backend='')
    environ = {'OMNIVOICE_TTS_RUNTIME_BACKEND': '  '}
    load_runtime_settings(settings, environ=environ)
    assert settings.runtime_backend == 'ming'
    assert engines.env_backend_override(environ) is None


def test_default_voice_chosen_under_ming_survives_ming_phonon_back(fake_ming, monkeypatch) -> None:
    class FakePhonon(MockSynthesizer):
        def __init__(self, settings, store) -> None:
            super().__init__(sample_rate=24_000)

        def load_now(self) -> None:
            pass

        def voice_names(self) -> list[str]:
            return ['Catalog']

    module = types.ModuleType('omnivoice_tts_server.phonon_runtime')
    module.PhononSynthesizer = FakePhonon
    monkeypatch.setitem(sys.modules, 'omnivoice_tts_server.phonon_runtime', module)
    phonon_dir = fake_ming.parent / 'phonon'
    phonon_dir.mkdir()
    (phonon_dir / 'config.json').write_text('{}', encoding='utf-8')
    client = make_client(ming_model_dir=fake_ming, phonon_model_dir=phonon_dir)
    try:
        _switch(client, 'ming')
        before = client.get('/api/admin/settings').json()['default_voice']
        chosen = 'Ming Favourite'
        assert before != chosen
        response = client.put('/api/admin/settings', json={'default_voice': chosen})
        assert response.status_code == 200, response.text
        _switch(client, 'phonon')
        assert _saved(client)['default_voice'] == chosen, 'Phonon mode persists the OmniVoice-side default voice'
        _switch(client, 'mock')
        assert client.get('/api/admin/settings').json()['default_voice'] == chosen
    finally:
        _close(client)


def test_switch_waits_for_a_stream_batch_dispatched_at_the_last_moment(fake_ming) -> None:
    """A concurrent engine's stream batches run as their own tasks: unload must wait for them."""
    import asyncio

    from omnivoice_tts_server.engines import EngineSwitcher

    order: list[str] = []

    class Queue:
        def __init__(self) -> None:
            self.batches = 1

        def running_batch_count(self) -> int:
            return self.batches

        async def cancel(self, job_id):  # pragma: no cover - not reached
            order.append('cancel')

    async def run():
        queue = Queue()
        switcher = EngineSwitcher.__new__(EngineSwitcher)
        switcher.app_state = types.SimpleNamespace(queue_service=queue, store=InMemoryStore(max_queue_size=4))

        async def finish_batch():
            await asyncio.sleep(0.2)
            order.append('batch done')
            queue.batches = 0

        task = asyncio.create_task(finish_batch())
        await switcher._await_running_batches(5.0)
        order.append('unload')
        await task

    asyncio.run(run())
    assert order == ['batch done', 'unload']


def test_held_job_gets_the_voice_seed_of_the_new_engine(fake_ming) -> None:
    """A job without its own seed that waits through a switch renders with the new engine's voice seed; a client
    seed stays."""
    from datetime import datetime, timezone

    from omnivoice_tts_server.domain.state import VoiceProfileRecord

    client = _client(fake_ming)
    try:
        state = client.app.state
        state.store.voice_profiles['voice_seeded'] = VoiceProfileRecord(
            voice_id='voice_seeded', name='Seeded', source='custom', created_at=datetime(2026, 10, 1, tzinfo=timezone.utc),
            audio_bytes=b'RIFF', ref_text='Hallo.', consent=True, seed=9305, engine_seeds={'ming': 41},
        )
        old = GatedSynthesizer()
        state.synthesizer = old
        state.queue_service.synthesizer = old
        first = client.post('/v1/jobs', json={'input': 'Erster Satz.', 'voice': 'Auto Voice'}).json()['job_id']
        assert old.started.wait(5)
        held = []
        for extra in ({}, {'seed': 5}):
            response = client.post(
                '/v1/jobs',
                json={'input': 'Wartet.', 'voice': 'voice_seeded', 'model': 'k2-fsa/OmniVoice-Base', **extra},
            )
            assert response.status_code in (200, 202), response.text
            held.append(response.json()['job_id'])
        assert [state.store.jobs[job_id].request.seed for job_id in held] == [9305, 5]

        assert client.post('/api/admin/runtime/backend', json={'backend': 'ming'}).status_code == 200
        old.gate.set()
        deadline = time.time() + 10
        while time.time() < deadline:
            body = client.get('/api/admin/runtime/engines').json()
            if body['switch']['state'] in {'ready', 'failed'}:
                break
            time.sleep(0.05)
        assert body['switch']['state'] == 'ready', body['switch']
        assert body['switch']['requeued_jobs'] == 2
        assert [state.store.jobs[job_id].request.seed for job_id in held] == [41, 5]
        for job_id in (first, *held):
            deadline = time.time() + 10
            while time.time() < deadline and client.get(f'/api/admin/jobs/{job_id}').json()['status'] != 'completed':
                time.sleep(0.05)
    finally:
        _close(client)
