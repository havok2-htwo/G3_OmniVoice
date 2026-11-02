"""The admin picks one model: clients cannot move that selection, each engine has its own settings, values and presets,
and clients can read what the selection offers."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from omnivoice_tts_server import engines, presets
from omnivoice_tts_server.app import create_app
from omnivoice_tts_server.config import (
    ENGINE_TUNED_FIELDS,
    RUNTIME_SETTINGS_FIELDS,
    Settings,
    clean_engine_profiles,
    load_runtime_settings,
    save_runtime_settings,
)
from omnivoice_tts_server.domain.models import ServerSettingsUpdateRequest, SpeechRequest, TaskType
from omnivoice_tts_server.domain.state import InMemoryStore, VoiceProfileRecord
from omnivoice_tts_server.finetune.registry import register_checkpoint
from omnivoice_tts_server.runtime_v2 import (
    OMNIVOICE_AUTO_ALIAS,
    OMNIVOICE_BASE_ALIAS,
    OMNIVOICE_DESIGN_ALIAS,
    BatchSynthesisItem,
    MockSynthesizer,
    OmniVoiceSynthesizer,
)

from test_app import make_client
from test_engine_switch import (  # noqa: F401 - fake_ming is a fixture
    ALIAS,
    FakeMingSynthesizer,
    GatedSynthesizer,
    _close,
    _saved,
    _switch,
    fake_ming,
)
from test_finetune import _create_domain_with_sentences, _wait_run
from test_finetune import make_client as make_finetune_client
from test_runtime_v2 import FakeOmniVoice, FakeOmniVoiceModel, install_fake_omnivoice, make_settings, make_wav_bytes

LOCAL = 'local/Custom-My-FT'
ALL_ALIASES = [OMNIVOICE_AUTO_ALIAS, OMNIVOICE_DESIGN_ALIAS, OMNIVOICE_BASE_ALIAS]
MING_TIER = presets.BUILTIN_PRESETS['Ming 24GB VRAM']['values']


def _add_voice(client) -> str:
    client.app.state.store.voice_profiles['voice_a'] = VoiceProfileRecord(
        voice_id='voice_a', name='Anna', source='custom', created_at=datetime(2026, 10, 1, tzinfo=timezone.utc),
        audio_bytes=make_wav_bytes(), content_type='audio/wav', filename='a.wav', ref_text='Hallo.', consent=True,
    )
    return 'voice_a'


class RecordingSynthesizer(MockSynthesizer):
    def __init__(self) -> None:
        super().__init__(sample_rate=24_000)
        self.ensure_calls: list[str | None] = []

    async def ensure_model(self, requested_model):
        self.ensure_calls.append(requested_model)
        return await super().ensure_model(requested_model)


def _record(client) -> RecordingSynthesizer:
    recorder = RecordingSynthesizer()
    client.app.state.synthesizer = recorder
    client.app.state.queue_service.synthesizer = recorder
    return recorder


def _base_client(**overrides):
    overrides.setdefault('active_model', OMNIVOICE_BASE_ALIAS)
    return make_client(**overrides)


def _selection(client) -> tuple[str, str, str]:
    state = client.app.state
    return (
        client.get('/api/admin/settings').json()['default_model'],
        state.settings.active_model,
        state.store.active_model,
    )


def _active_listed(client) -> list[str]:
    return [entry['id'] for entry in client.get('/v1/models').json()['data'] if entry['active']]


# --- B1: client requests never move the selection -------------------------------------------------------------------


@pytest.mark.parametrize(
    ('route', 'payload'),
    [
        ('/api/v1/synthesize', {'input': 'Hallo.', 'model': OMNIVOICE_AUTO_ALIAS}),
        ('/api/v1/synthesize', {'input': 'Hallo.', 'model': OMNIVOICE_DESIGN_ALIAS, 'instructions': 'female'}),
        ('/api/v1/synthesize', {'input': 'Hallo.', 'model': 'tts-1'}),
        ('/v1/audio/speech', {'input': 'Hallo.', 'model': OMNIVOICE_AUTO_ALIAS}),
        ('/v1/audio/speech', {'input': 'Hallo.', 'model': 'tts-1', 'voice': 'Auto Voice'}),
        ('/v1/jobs', {'input': 'Hallo.', 'model': 'tts-1'}),
    ],
)
def test_a_client_model_id_leaves_the_selection_alone(route, payload) -> None:
    client = _base_client(default_voice='Anna')
    try:
        _add_voice(client)
        response = client.post(route, json=payload)
        assert response.status_code in (200, 202), response.text
        if route == '/v1/jobs':
            job_id = response.json()['job_id']
            deadline = time.time() + 5
            while time.time() < deadline and client.get(f'/api/admin/jobs/{job_id}').json()['status'] != 'completed':
                time.sleep(0.02)
        assert _selection(client) == (OMNIVOICE_BASE_ALIAS,) * 3
        assert _active_listed(client) == [OMNIVOICE_BASE_ALIAS]
        # The no-model requests after it still clone: the default voice, and a profile without a 400.
        after = client.post('/api/v1/synthesize', json={'input': 'Danach.'})
        assert after.status_code == 200, after.text
        assert after.json()['voice'] == 'voice_a'
        assert client.post('/api/v1/synthesize', json={'input': 'Danach.', 'voice': 'Anna'}).status_code == 200
        assert client.put('/api/admin/settings', json=client.get('/api/admin/settings').json()).status_code == 200
        assert _saved(client)['active_model'] == OMNIVOICE_BASE_ALIAS
    finally:
        _close(client)


def test_an_alias_keeps_its_task_type_for_that_request() -> None:
    client = _base_client(default_voice='Anna')
    try:
        _add_voice(client)
        auto = client.post('/api/v1/synthesize', json={'input': 'Hallo.', 'model': OMNIVOICE_AUTO_ALIAS})
        assert auto.json()['task_type'] == 'CustomVoice' and auto.json()['model'] == OMNIVOICE_AUTO_ALIAS
        design = client.post('/api/v1/synthesize', json={'input': 'Hallo.', 'model': OMNIVOICE_DESIGN_ALIAS, 'instructions': 'female'})
        assert design.json()['task_type'] == 'VoiceDesign'
        pinned = client.post(
            '/api/v1/synthesize', json={'input': 'Hallo.', 'model': OMNIVOICE_AUTO_ALIAS, 'task_type': 'Base', 'voice': 'Anna'}
        )
        assert pinned.status_code == 200, pinned.text
        assert pinned.json()['task_type'] == 'Base', 'an explicit task_type wins'
    finally:
        _close(client)


def test_every_id_but_local_runs_on_the_selected_checkpoint() -> None:
    client = _base_client(active_model=LOCAL, supported_models=[*ALL_ALIASES, LOCAL], default_voice='Anna')
    try:
        _add_voice(client)
        recorder = _record(client)
        # What Atrium sends, literally.
        atrium = client.post('/api/v1/synthesize', json={'input': 'Hallo.', 'model': OMNIVOICE_BASE_ALIAS, 'voice': 'voice_a'})
        assert atrium.status_code == 200, atrium.text
        assert atrium.json()['model'] == LOCAL, 'the response names what really rendered'
        assert atrium.json()['task_type'] == 'Base', 'and stays a clone'
        assert recorder.ensure_calls == [LOCAL]

        # Thunderbird and DiscordScan: a profile voice on the OpenAI route is forced to the Base alias.
        assert client.post('/v1/audio/speech', json={'input': 'Hallo.', 'voice': 'Anna'}).status_code == 200
        assert client.post('/api/v1/synthesize', json={'input': 'Hallo.', 'model': 'tts-1'}).status_code == 200
        assert recorder.ensure_calls == [LOCAL, LOCAL, LOCAL]

        other = 'local/Other'
        assert client.post('/api/v1/synthesize', json={'input': 'Hallo.', 'model': other}).status_code == 200
        assert recorder.ensure_calls[-1] == other, 'an explicit local/... is loaded as named'
        assert _selection(client) == (LOCAL,) * 3
    finally:
        _close(client)


def test_a_base_clone_renders_as_a_clone_on_a_finetuned_checkpoint(tmp_path, monkeypatch) -> None:
    install_fake_omnivoice(monkeypatch)
    settings = make_settings(tmp_path, active_model=LOCAL, supported_models=[*ALL_ALIASES, LOCAL])
    checkpoint = settings.models_root_dir / 'Custom-My-FT'
    checkpoint.mkdir(parents=True)
    (checkpoint / 'config.json').write_text('{}', encoding='utf-8')
    store = InMemoryStore(max_queue_size=8)
    store.voice_profiles['voice_a'] = VoiceProfileRecord(
        voice_id='voice_a', name='Anna', source='custom', created_at=datetime(2026, 10, 1, tzinfo=timezone.utc),
        audio_bytes=make_wav_bytes(), content_type='audio/wav', filename='a.wav', ref_text='Hallo.', consent=True,
    )
    synthesizer = OmniVoiceSynthesizer(settings=settings, store=store)
    asyncio.run(synthesizer.ensure_model(engines.served_model(settings, OMNIVOICE_BASE_ALIAS)))
    assert len(FakeOmniVoice.load_calls) == 1 and FakeOmniVoice.load_calls[0]['model_source'] == str(checkpoint)
    request = SpeechRequest(input='Klon.', model=OMNIVOICE_BASE_ALIAS, voice='voice_a')
    asyncio.run(synthesizer.render_batch([BatchSynthesisItem(job_id='j', sentence_index=0, text='Klon.', request=request)]))
    assert 'voice_clone_prompt' in FakeOmniVoiceModel.instances[-1].calls[-1]


def test_the_synthesizer_never_writes_the_selection(tmp_path, monkeypatch) -> None:
    install_fake_omnivoice(monkeypatch)
    settings = make_settings(tmp_path, active_model=OMNIVOICE_BASE_ALIAS, supported_models=[*ALL_ALIASES, LOCAL])
    checkpoint = settings.models_root_dir / 'Custom-My-FT'
    checkpoint.mkdir(parents=True)
    (checkpoint / 'config.json').write_text('{}', encoding='utf-8')
    synthesizer = OmniVoiceSynthesizer(settings=settings, store=InMemoryStore(max_queue_size=8))

    asyncio.run(synthesizer.ensure_model(OMNIVOICE_AUTO_ALIAS))
    assert settings.active_model == OMNIVOICE_BASE_ALIAS, 'first load'
    asyncio.run(synthesizer.ensure_model(OMNIVOICE_DESIGN_ALIAS))
    assert settings.active_model == OMNIVOICE_BASE_ALIAS, 're-pointed alias'
    assert asyncio.run(synthesizer.ensure_model(LOCAL)) == (LOCAL, pytest.approx(0, abs=10_000))
    assert settings.active_model == OMNIVOICE_BASE_ALIAS, 'explicit local checkpoint'
    assert len(FakeOmniVoice.load_calls) == 2


def test_finetune_datagen_in_auto_mode_keeps_the_selection() -> None:
    client = make_finetune_client()
    try:
        domain_id = _create_domain_with_sentences(client, 'Auto Domain', 2)
        start = client.post('/api/admin/finetune/generate', json={
            'domain_ids': [domain_id], 'voice_mode': 'auto', 'language': 'Deutsch',
            'wer_threshold': 0.0, 'max_attempts': 1, 'whisper_base_url': 'mock',
        })
        assert start.status_code == 200, start.text
        assert _wait_run(client, start.json()['run_id'])['status'] == 'completed'
        assert client.app.state.settings.active_model == OMNIVOICE_BASE_ALIAS
        assert client.app.state.store.active_model == OMNIVOICE_BASE_ALIAS
    finally:
        client.__exit__(None, None, None)
        client._temp_data_dir.cleanup()
        client._temp_models_dir.cleanup()


def _promoted(client, *, select: bool = True) -> tuple[str, Path]:
    settings = client.app.state.settings
    entry = register_checkpoint(
        settings.data_dir, name='My FT', dirname='Custom-My-FT', base_model=OMNIVOICE_BASE_ALIAS, steps=1, run_id=None,
        now_iso='2026-10-01T00:00:00+00:00',
    )
    settings.supported_models.append(entry['model_id'])
    folder = settings.models_root_dir / 'Custom-My-FT'
    folder.mkdir()
    (folder / 'config.json').write_text('{}', encoding='utf-8')
    if select:
        assert client.put('/api/admin/settings', json={'default_model': LOCAL}).status_code == 200
    return entry['checkpoint_id'], folder


def _close_finetune(client) -> None:
    client.__exit__(None, None, None)
    client._temp_data_dir.cleanup()
    client._temp_models_dir.cleanup()


def test_the_selected_checkpoint_cannot_be_deleted() -> None:
    client = make_finetune_client(active_model=OMNIVOICE_BASE_ALIAS)
    try:
        checkpoint_id, folder = _promoted(client)
        route = f'/api/admin/finetune/checkpoints/{checkpoint_id}'
        refused = client.delete(route)
        assert refused.status_code == 409, refused.text
        assert LOCAL in refused.json()['detail']
        assert folder.exists() and LOCAL in client.app.state.settings.supported_models
        assert client.get('/api/admin/finetune/checkpoints').json()['checkpoints']

        # Off the selection it goes, and no selection is left pointing at it.
        assert client.put('/api/admin/settings', json={'default_model': OMNIVOICE_BASE_ALIAS}).status_code == 200
        assert client.delete(route).status_code == 200
        assert not folder.exists() and LOCAL not in client.app.state.settings.supported_models
        assert _selection(client) == (OMNIVOICE_BASE_ALIAS,) * 3
        assert client.delete(route).status_code == 404
    finally:
        _close_finetune(client)


def test_the_selected_checkpoint_stays_while_another_engine_runs(fake_ming) -> None:  # noqa: F811
    client = make_finetune_client(active_model=OMNIVOICE_BASE_ALIAS, ming_model_dir=fake_ming)
    try:
        checkpoint_id, folder = _promoted(client)
        _switch(client, 'ming')
        assert client.delete(f'/api/admin/finetune/checkpoints/{checkpoint_id}').status_code == 409
        assert folder.exists()
        _switch(client, 'mock')
        assert _selection(client) == (LOCAL,) * 3
    finally:
        _close_finetune(client)


def test_delete_cache_never_takes_the_base_weights_for_a_checkpoint() -> None:
    client = make_finetune_client(active_model=OMNIVOICE_BASE_ALIAS)
    try:
        _promoted(client)
        weights = client.app.state.settings.models_root_dir / 'OmniVoice'
        weights.mkdir()
        for payload in ({'model_id': LOCAL}, {'model': LOCAL}, {}):
            assert client.post('/api/admin/models/delete', json=payload).status_code == 400, payload
            assert weights.exists(), payload
        named = client.post('/api/admin/models/delete', json={'model_id': OMNIVOICE_BASE_ALIAS})
        assert named.status_code == 200 and named.json()['removed'] is True
        assert not weights.exists()
    finally:
        _close_finetune(client)


def test_admin_actions_still_move_the_selection() -> None:
    client = _base_client(default_voice='Auto Voice')
    try:
        for route in ('preload', 'warmup', 'reload', 'unload'):
            assert client.post(f'/api/admin/models/{route}', json={'model': OMNIVOICE_AUTO_ALIAS}).status_code == 200
            assert _selection(client) == (OMNIVOICE_AUTO_ALIAS,) * 3, route
            assert client.post(f'/api/admin/models/{route}', json={'model': OMNIVOICE_BASE_ALIAS}).status_code == 200
            assert _selection(client) == (OMNIVOICE_BASE_ALIAS,) * 3, route
        assert client.post('/api/admin/models/preload', json={'model': 'tts-1'}).status_code == 200
        assert _selection(client) == (OMNIVOICE_BASE_ALIAS,) * 3, 'an id nobody lists is no selection'
    finally:
        _close(client)


def test_the_settings_put_moves_the_selection_at_once() -> None:
    client = make_client()
    try:
        assert _active_listed(client) == [OMNIVOICE_AUTO_ALIAS]
        response = client.put('/api/admin/settings', json={'default_model': OMNIVOICE_BASE_ALIAS})
        assert response.status_code == 200, response.text
        assert _active_listed(client) == [OMNIVOICE_BASE_ALIAS]
        assert _selection(client) == (OMNIVOICE_BASE_ALIAS,) * 3
    finally:
        _close(client)


# --- B2: what the admin can select ------------------------------------------------------------------------------------


def test_selectable_models_hide_the_client_aliases() -> None:
    assert engines.selectable_models(Settings()) == [OMNIVOICE_BASE_ALIAS]
    with_local = Settings(supported_models=[*ALL_ALIASES, LOCAL])
    assert engines.selectable_models(with_local) == [OMNIVOICE_BASE_ALIAS, LOCAL]
    ming = Settings(runtime_backend='ming', supported_models=[ALIAS], active_model=ALIAS)
    assert engines.selectable_models(ming) == [ALIAS]


@pytest.mark.parametrize('saved', [OMNIVOICE_AUTO_ALIAS, OMNIVOICE_DESIGN_ALIAS])
def test_a_saved_alias_stays_selectable_while_it_is_active(saved) -> None:
    settings = Settings(active_model=saved)
    assert engines.selectable_models(settings) == [saved, OMNIVOICE_BASE_ALIAS]
    assert settings.active_model == saved, 'no migration'


def test_the_active_model_is_listed_even_outside_supported_models() -> None:
    assert engines.selectable_models(Settings(active_model='k2-fsa/Other')) == [OMNIVOICE_BASE_ALIAS, 'k2-fsa/Other']


def test_the_admin_lists_only_selectable_models_and_clients_keep_every_alias() -> None:
    client = _base_client()
    try:
        assert [row['id'] for row in client.get('/api/admin/models').json()['models']] == [OMNIVOICE_BASE_ALIAS]
        snapshot = client.get('/api/admin/snapshot').json()
        assert snapshot['selectable_models'] == [OMNIVOICE_BASE_ALIAS]
        assert [model['model_id'] for model in snapshot['models']] == ALL_ALIASES
        assert [model['model_id'] for model in client.get('/api/v1/models').json()] == ALL_ALIASES
        assert [entry['id'] for entry in client.get('/v1/models').json()['data']] == ALL_ALIASES
        assert [entry['id'] for entry in client.get('/v1/audio/models').json()['models']] == ALL_ALIASES
    finally:
        _close(client)


def test_engine_loaded_is_read_from_the_synthesizer(tmp_path, monkeypatch) -> None:
    install_fake_omnivoice(monkeypatch)
    settings = make_settings(tmp_path, active_model=OMNIVOICE_BASE_ALIAS, supported_models=[*ALL_ALIASES, LOCAL])
    checkpoint = settings.models_root_dir / 'Custom-My-FT'
    checkpoint.mkdir(parents=True)
    (checkpoint / 'config.json').write_text('{}', encoding='utf-8')
    synthesizer = OmniVoiceSynthesizer(settings=settings, store=InMemoryStore(max_queue_size=8))
    spec = engines.ENGINES['omnivoice']

    assert engines.engine_info(spec, settings, synthesizer).loaded is False, 'nothing loaded yet'
    asyncio.run(synthesizer.ensure_model(OMNIVOICE_AUTO_ALIAS))
    assert engines.engine_info(spec, settings, synthesizer).loaded is True, 'the aliases share the checkpoint'
    asyncio.run(synthesizer.ensure_model(LOCAL))
    assert engines.engine_info(spec, settings, synthesizer).loaded is False, 'a client replaced the selection'
    settings.active_model = LOCAL
    assert engines.engine_info(spec, settings, synthesizer).loaded is True
    asyncio.run(synthesizer.unload())
    assert engines.engine_info(spec, settings, synthesizer).loaded is False


def test_engine_list_reports_loaded_for_the_active_engine_only(fake_ming) -> None:  # noqa: F811
    client = make_client(ming_model_dir=fake_ming)
    try:
        def by_id() -> dict:
            return {item['id']: item for item in client.get('/api/admin/runtime/engines').json()['engines']}

        assert by_id()['mock']['loaded'] is True and by_id()['ming']['loaded'] is False
        client.app.state.synthesizer.resident_models = lambda: set()
        assert by_id()['mock']['loaded'] is False, 'store.models_loaded plays no part'
    finally:
        _close(client)


# --- B3: settings per engine ----------------------------------------------------------------------------------------


def test_engine_settings_fields_are_put_fields_of_the_right_engine() -> None:
    fields = engines.ENGINE_SETTINGS_FIELDS
    assert set(fields) == set(engines.ENGINES)
    assert fields['mock'] == fields['omnivoice']
    update_fields = set(ServerSettingsUpdateRequest.model_fields)
    for names in fields.values():
        assert set(names) <= update_fields and len(set(names)) == len(names)
    assert set(engines.SERVER_SETTINGS_FIELDS) <= update_fields
    for engine, names in fields.items():
        assert not set(names) & set(engines.SERVER_SETTINGS_FIELDS), engine
    assert not any(name.startswith(('ming_', 'phonon_')) for name in fields['omnivoice'])
    assert not any(name.startswith('phonon_') for name in fields['ming'])
    assert not any(name.startswith('ming_') for name in fields['phonon'])
    # Read by the OmniVoice path only (Ming and Phonon stream through stream_prebuffer_ms instead).
    assert 'stream_chunk_ms' in fields['omnivoice'] and 'stream_prebuffer_ms' not in fields['omnivoice']
    assert 'preferred_device' not in fields['phonon'], 'Phonon runs on the CPU'
    assert 'default_voice' not in fields['phonon'], 'Phonon speaks its catalog voices'
    assert 'max_batch_size' not in fields['ming'], 'the engine slots size a Ming batch'


def test_engine_list_exposes_the_fields() -> None:
    client = make_client()
    try:
        body = client.get('/api/admin/runtime/engines').json()
        assert body['server_settings_fields'] == list(engines.SERVER_SETTINGS_FIELDS)
        for item in body['engines']:
            assert item['settings_fields'] == list(engines.ENGINE_SETTINGS_FIELDS[item['id']])
    finally:
        _close(client)


def test_phonon_settings_are_persisted_and_validated() -> None:
    for name in ('phonon_lang', 'phonon_quant', 'phonon_workers', 'phonon_threads'):
        assert name in RUNTIME_SETTINGS_FIELDS
    client = make_client()
    try:
        before = client.get('/api/admin/settings').json()
        assert (before['phonon_lang'], before['phonon_quant'], before['phonon_workers'], before['phonon_threads']) == ('de', 'q8', 2, 0)
        response = client.put(
            '/api/admin/settings',
            json={'phonon_lang': ' en ', 'phonon_quant': 'q4k', 'phonon_workers': 8, 'phonon_threads': 3},
        )
        assert response.status_code == 200, response.text
        assert (response.json()['phonon_lang'], response.json()['phonon_quant']) == ('en', 'q4k')
        saved = _saved(client)
        assert (saved['phonon_workers'], saved['phonon_threads'], saved['phonon_lang']) == (8, 3, 'en')

        for bad in ({'phonon_workers': 0}, {'phonon_workers': 17}, {'phonon_threads': -1}, {'phonon_threads': 65}):
            assert client.put('/api/admin/settings', json=bad).status_code == 422, bad
        assert client.put('/api/admin/settings', json={'phonon_lang': '  '}).status_code == 200
        assert client.get('/api/admin/settings').json()['phonon_lang'] == 'en', 'blank keeps the value'
    finally:
        _close(client)


def test_phonon_settings_load_back(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path, phonon_lang='en', phonon_quant='q4k', phonon_workers=4, phonon_threads=2)
    save_runtime_settings(settings)
    reloaded = Settings(data_dir=tmp_path)
    load_runtime_settings(reloaded, environ={})
    assert (reloaded.phonon_lang, reloaded.phonon_quant, reloaded.phonon_workers, reloaded.phonon_threads) == ('en', 'q4k', 4, 2)


def test_phonon_keys_are_preset_keys(tmp_path) -> None:
    saved = presets.save_preset(
        tmp_path, 'cpu', {'runtime_backend': 'phonon', 'phonon_quant': 'q4k', 'phonon_workers': 4}, now_iso='2026-10-31T00:00:00+00:00'
    )
    assert saved['values'] == {'runtime_backend': 'phonon', 'phonon_quant': 'q4k', 'phonon_workers': 4}
    assert saved['engine'] == 'phonon'
    with pytest.raises(ValueError, match='phonon_workers'):
        presets.save_preset(tmp_path, 'bad', {'phonon_workers': 0}, now_iso='2026-10-31T00:00:00+00:00')


# --- B4: tuned values stay with their engine --------------------------------------------------------------------------


def _wait_switch(client) -> dict:
    deadline = time.time() + 10
    body = {}
    while time.time() < deadline:
        body = client.get('/api/admin/runtime/engines').json()
        if body['switch']['state'] in {'ready', 'failed'}:
            return body
        time.sleep(0.02)
    raise AssertionError(body)


def test_engine_profiles_are_persisted_but_not_public() -> None:
    assert 'engine_profiles' in RUNTIME_SETTINGS_FIELDS
    assert 'engine_profiles' not in ServerSettingsUpdateRequest.model_fields
    client = make_client()
    try:
        assert 'engine_profiles' not in client.get('/api/admin/settings').json()
        assert client.put('/api/admin/settings', json={'batch_wait_ms': 25}).status_code == 200
        assert _saved(client)['engine_profiles']['mock']['batch_wait_ms'] == 25
    finally:
        _close(client)


def test_a_ming_tier_does_not_leak_into_the_engine_it_came_from(fake_ming) -> None:  # noqa: F811
    client = make_client(ming_model_dir=fake_ming, max_parallel_requests=4, batch_wait_ms=30, max_batch_size=4)
    try:
        _switch(client, 'ming')
        ming = client.get('/api/admin/settings').json()
        assert (ming['max_parallel_requests'], ming['batch_wait_ms']) == (16, 0), 'no profile yet: the built-in tier'
        assert ming['ming_max_slots'] == 16
        assert ming['max_batch_size'] == 4, 'a field the tier does not set keeps its value'

        assert client.put('/api/admin/settings', json={'max_parallel_requests': 32}).status_code == 200
        _switch(client, 'mock')
        back = client.get('/api/admin/settings').json()
        assert (back['max_parallel_requests'], back['batch_wait_ms']) == (4, 30)
        assert _saved(client)['max_parallel_requests'] == 4

        _switch(client, 'ming')
        assert client.get('/api/admin/settings').json()['max_parallel_requests'] == 32, 'the profile, not the tier'
        assert _saved(client)['engine_profiles']['mock']['batch_wait_ms'] == 30
    finally:
        _close(client)


def test_nothing_but_the_tuned_fields_moves_between_engines(fake_ming) -> None:  # noqa: F811
    client = make_client(ming_model_dir=fake_ming)
    try:
        assert client.put('/api/admin/settings', json={'num_step': 12, 'max_input_chars': 5000}).status_code == 200
        _switch(client, 'ming')
        assert client.put('/api/admin/settings', json={'num_step': 13, 'max_input_chars': 6000}).status_code == 200
        _switch(client, 'mock')
        after = client.get('/api/admin/settings').json()
        assert (after['num_step'], after['max_input_chars']) == (13, 6000)
    finally:
        _close(client)


def test_same_put_values_go_to_the_target_not_to_the_running_engine(fake_ming) -> None:  # noqa: F811
    client = make_client(ming_model_dir=fake_ming, max_parallel_requests=4, batch_wait_ms=30)
    try:
        response = client.put(
            '/api/admin/settings',
            json={'runtime_backend': 'ming', 'max_parallel_requests': 24, 'ming_max_slots': 24, 'preferred_device': 'cuda:1'},
        )
        assert response.status_code == 200, response.text
        assert _wait_switch(client)['switch']['state'] == 'ready'
        ming = client.get('/api/admin/settings').json()
        assert (ming['max_parallel_requests'], ming['ming_max_slots'], ming['preferred_device']) == (24, 24, 'cuda:1')
        assert ming['batch_wait_ms'] == 0, 'the tier still fills what the PUT left out'
        assert FakeMingSynthesizer.instances[-1].device == 'cuda:1'
        assert FakeMingSynthesizer.instances[-1].slots == 24

        _switch(client, 'mock')
        mock = client.get('/api/admin/settings').json()
        assert (mock['max_parallel_requests'], mock['batch_wait_ms']) == (4, 30)
        assert mock['preferred_device'] != 'cuda:1'
    finally:
        _close(client)


def test_a_put_that_only_sends_a_ming_knob_keeps_the_rest_of_the_tier(fake_ming) -> None:  # noqa: F811
    client = make_client(ming_model_dir=fake_ming, ming_first_chunk_latents=5, batch_wait_ms=30)
    try:
        assert client.put('/api/admin/settings', json={'runtime_backend': 'ming', 'ming_max_slots': 12}).status_code == 200
        assert _wait_switch(client)['switch']['state'] == 'ready'
        ming = client.get('/api/admin/settings').json()
        assert ming['ming_max_slots'] == 12
        assert ming['ming_first_chunk_latents'] == MING_TIER['ming_first_chunk_latents']
        assert ming['max_parallel_requests'] == MING_TIER['max_parallel_requests']
    finally:
        _close(client)


def test_a_failed_switch_puts_every_value_back(fake_ming) -> None:  # noqa: F811
    client = make_client(ming_model_dir=fake_ming, max_parallel_requests=4, batch_wait_ms=30, ming_max_slots=7)
    try:
        FakeMingSynthesizer.fail_build = True
        response = client.post('/api/admin/runtime/backend', json={'backend': 'ming', 'wait': True})
        assert response.json()['switch']['state'] == 'failed'
        after = client.get('/api/admin/settings').json()
        assert (after['max_parallel_requests'], after['batch_wait_ms'], after['ming_max_slots']) == (4, 30, 7)
        assert _saved(client)['max_parallel_requests'] == 4
        assert 'ming' not in _saved(client)['engine_profiles'], 'still no profile: the next try gets the tier'
    finally:
        _close(client)


def test_an_engine_without_a_tier_keeps_the_current_values() -> None:
    settings = Settings(max_parallel_requests=5, runtime_backend='omnivoice')
    replaced = engines.swap_engine_values(settings, 'omnivoice', 'phonon')
    assert settings.max_parallel_requests == 5
    assert set(replaced) == set(ENGINE_TUNED_FIELDS)
    assert settings.engine_profiles['omnivoice']['max_parallel_requests'] == 5


def _write_saved(path: Path, **values) -> None:
    (path / 'runtime_settings.json').write_text(json.dumps(values), encoding='utf-8')


def _started(path: Path, environ: dict | None = None, *, backend: str = 'omnivoice') -> Settings:
    environ = environ or {}
    settings = Settings(data_dir=path, runtime_backend=environ.get('OMNIVOICE_TTS_RUNTIME_BACKEND', backend))
    load_runtime_settings(settings, environ=environ)
    engines.apply_startup_engine_values(settings, environ=environ)
    return settings


def test_start_with_a_saved_ming_and_no_profile_gives_a_tuned_ming_server(tmp_path) -> None:
    _write_saved(tmp_path, runtime_backend='ming', max_parallel_requests=8, batch_wait_ms=30, ming_max_slots=8)
    settings = _started(tmp_path)
    for key, value in MING_TIER.items():
        if key != 'runtime_backend':
            assert getattr(settings, key) == value, key
    assert settings.runtime_backend == 'ming'
    assert settings.engine_profiles['ming']['max_parallel_requests'] == 16


@pytest.mark.parametrize('environ', [None, {'OMNIVOICE_TTS_RUNTIME_BACKEND': 'ming'}])
def test_start_without_a_profile_keeps_the_saved_ming_sample_rate(tmp_path, environ) -> None:
    _write_saved(tmp_path, runtime_backend='ming', ming_output_sample_rate=44100, ming_max_slots=8)
    settings = _started(tmp_path, environ)
    assert settings.ming_output_sample_rate == 44100, 'clients of this host keep their rate'
    assert (settings.ming_max_slots, settings.max_parallel_requests) == (16, 16), 'the rest of the tier applies'


def test_start_without_a_saved_ming_sample_rate_gives_the_tiers_rate(tmp_path) -> None:
    _write_saved(tmp_path, runtime_backend='ming')
    assert _started(tmp_path).ming_output_sample_rate == MING_TIER['ming_output_sample_rate']


class RateFollowingMing(FakeMingSynthesizer):
    """The real engine takes its rate from the settings (ming_runtime.py); FakeMingSynthesizer stays at 24000."""

    def __init__(self, settings, store) -> None:
        super().__init__(settings, store)
        self.sample_rate = int(settings.ming_output_sample_rate)


def _assert_ming_rate(client, rate: int) -> None:
    assert client.app.state.settings.ming_output_sample_rate == rate
    assert _saved(client)['ming_output_sample_rate'] == rate
    assert client.get('/api/v1/capabilities').json()['sample_rate'] == rate


@pytest.mark.parametrize('rate', [24000, 44100])
def test_a_live_switch_keeps_the_ming_sample_rate_of_the_host(fake_ming, monkeypatch, rate) -> None:
    monkeypatch.setattr(sys.modules['omnivoice_tts_server.ming_runtime'], 'MingSynthesizer', RateFollowingMing)
    client = make_client(ming_model_dir=fake_ming, ming_output_sample_rate=rate)
    try:
        _switch(client, 'ming')
        assert client.app.state.settings.ming_max_slots == MING_TIER['ming_max_slots'], 'the rest of the tier applies'
        _assert_ming_rate(client, rate)
    finally:
        _close(client)


def test_a_settings_put_that_switches_keeps_the_ming_sample_rate(fake_ming, monkeypatch) -> None:
    monkeypatch.setattr(sys.modules['omnivoice_tts_server.ming_runtime'], 'MingSynthesizer', RateFollowingMing)
    client = make_client(ming_model_dir=fake_ming, ming_output_sample_rate=44100)
    try:
        assert client.put('/api/admin/settings', json={'runtime_backend': 'ming', 'ming_max_slots': 12}).status_code == 200
        assert _wait_switch(client)['switch']['state'] == 'ready'
        _assert_ming_rate(client, 44100)
    finally:
        _close(client)


def test_a_sent_ming_sample_rate_still_wins_a_switch(fake_ming, monkeypatch) -> None:
    monkeypatch.setattr(sys.modules['omnivoice_tts_server.ming_runtime'], 'MingSynthesizer', RateFollowingMing)
    client = make_client(ming_model_dir=fake_ming, ming_output_sample_rate=44100)
    try:
        response = client.put('/api/admin/settings', json={'runtime_backend': 'ming', 'ming_output_sample_rate': 24000})
        assert response.status_code == 200
        assert _wait_switch(client)['switch']['state'] == 'ready'
        _assert_ming_rate(client, 24000)
    finally:
        _close(client)


def test_an_env_pin_of_ming_starts_the_same_way(tmp_path) -> None:
    _write_saved(tmp_path, runtime_backend='omnivoice', max_parallel_requests=4, batch_wait_ms=30)
    settings = _started(tmp_path, {'OMNIVOICE_TTS_RUNTIME_BACKEND': 'ming'})
    assert (settings.runtime_backend, settings.max_parallel_requests, settings.batch_wait_ms) == ('ming', 16, 0)


def test_an_env_pin_applies_the_pinned_engines_profile(tmp_path) -> None:
    _write_saved(
        tmp_path, runtime_backend='omnivoice', max_parallel_requests=4, ming_max_slots=32,
        engine_profiles={'ming': {'max_parallel_requests': 32, 'batch_wait_ms': 0}},
    )
    settings = _started(tmp_path, {'OMNIVOICE_TTS_RUNTIME_BACKEND': 'ming'})
    assert (settings.max_parallel_requests, settings.ming_max_slots) == (32, 32)


def test_the_environment_wins_over_the_startup_tier(tmp_path) -> None:
    environ = {'OMNIVOICE_TTS_RUNTIME_BACKEND': 'ming', 'omnivoice_tts_ming_max_slots': '8'}
    settings = Settings(data_dir=tmp_path, runtime_backend='ming', ming_max_slots=8)
    engines.apply_startup_engine_values(settings, environ=environ)
    assert (settings.ming_max_slots, settings.max_parallel_requests) == (8, 16)


def test_start_with_a_profile_applies_it(tmp_path) -> None:
    _write_saved(tmp_path, runtime_backend='ming', max_parallel_requests=8, engine_profiles={'ming': {'max_parallel_requests': 32}})
    assert _started(tmp_path).max_parallel_requests == 32


@pytest.mark.parametrize('backend', ['omnivoice', 'mock'])
def test_omnivoice_and_mock_seed_their_profile_from_the_current_values(tmp_path, backend) -> None:
    _write_saved(tmp_path, runtime_backend=backend, max_parallel_requests=6, batch_wait_ms=30)
    settings = _started(tmp_path)
    assert (settings.max_parallel_requests, settings.batch_wait_ms) == (6, 30)
    assert settings.engine_profiles[backend]['max_parallel_requests'] == 6
    assert set(settings.engine_profiles[backend]) == set(ENGINE_TUNED_FIELDS)


SAVED_OMNIVOICE = {
    'runtime_backend': 'omnivoice', 'max_parallel_requests': 4, 'max_batch_size': 4, 'batch_wait_ms': 30,
    'vram_budget_mb': 5000,
}
OMNIVOICE_PROFILE = {
    'engine_profiles': {'omnivoice': {'max_parallel_requests': 4, 'max_batch_size': 4, 'vram_budget_mb': 5000}},
}


def _spawned(tmp_path, monkeypatch, backend: str, saved: dict | None, overrides: dict, preset: str | None = None) -> Settings:
    """What a benchmark's server runs with: prepare_data_dir, then the server's start."""
    path = Path(__file__).resolve().parents[2] / 'tools' / 'benchmarks' / 'bench_engine_streaming.py'
    spec = importlib.util.spec_from_file_location('bench_engine_streaming', path)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    repo = tmp_path / 'repo'
    (repo / 'data' / 'voices').mkdir(parents=True)
    if saved is not None:
        _write_saved(repo / 'data', **saved)
    monkeypatch.setattr(bench, 'REPO', repo)
    monkeypatch.setattr(sys, 'path', list(sys.path))
    data = bench.prepare_data_dir(tmp_path / 'work', backend, overrides, preset)
    return _started(data, {'OMNIVOICE_TTS_RUNTIME_BACKEND': backend})


@pytest.mark.parametrize(
    ('backend', 'saved', 'overrides', 'preset', 'expected'),
    [
        ('ming', SAVED_OMNIVOICE, {'ming_max_slots': 32, 'max_parallel_requests': 32}, None,
         {'ming_max_slots': 32, 'max_parallel_requests': 32, 'batch_wait_ms': 30}),
        ('ming', SAVED_OMNIVOICE, {}, 'Ming best performance', {'ming_max_slots': 64, 'max_parallel_requests': 64}),
        ('omnivoice', {**SAVED_OMNIVOICE, **OMNIVOICE_PROFILE}, {}, '32GB VRAM',
         {'max_parallel_requests': 8, 'max_batch_size': 8, 'vram_budget_mb': 7000}),
        ('omnivoice', {**SAVED_OMNIVOICE, **OMNIVOICE_PROFILE}, {'max_batch_size': 16}, None,
         {'max_batch_size': 16, 'vram_budget_mb': 5000}),
        # A key outside the profile, with nothing tuned in the file: the Ming tier must not come back for it.
        ('ming', None, {'ming_max_slots': 32}, None, {'ming_max_slots': 32}),
        # The repo's saved profile of the spawned engine stays the base of the override.
        ('ming', {**SAVED_OMNIVOICE, 'engine_profiles': {'ming': {'max_parallel_requests': 32}}}, {'ming_max_slots': 48}, None,
         {'ming_max_slots': 48, 'max_parallel_requests': 32}),
    ],
)
def test_a_benchmark_server_runs_with_what_was_asked_for(tmp_path, monkeypatch, backend, saved, overrides, preset, expected) -> None:
    settings = _spawned(tmp_path, monkeypatch, backend, saved, overrides, preset)
    assert settings.runtime_backend == backend
    assert {name: getattr(settings, name) for name in expected} == expected
    assert set(settings.engine_profiles) == {backend}, 'one engine runs here'


def test_a_bare_benchmark_spawn_starts_like_any_server_with_that_file(tmp_path, monkeypatch) -> None:
    settings = _spawned(tmp_path, monkeypatch, 'ming', SAVED_OMNIVOICE, {})
    assert (settings.ming_max_slots, settings.max_parallel_requests, settings.batch_wait_ms) == (16, 16, 0)


def test_a_file_without_profiles_loads(tmp_path) -> None:
    _write_saved(tmp_path, max_batch_size=3)
    settings = Settings(data_dir=tmp_path)
    load_runtime_settings(settings, environ={})
    assert settings.engine_profiles == {} and settings.max_batch_size == 3


def test_the_start_writes_nothing(tmp_path) -> None:
    _write_saved(tmp_path, runtime_backend='mock', max_batch_size=3)
    before = (tmp_path / 'runtime_settings.json').read_bytes()
    app = create_app(Settings(data_dir=tmp_path, runtime_backend='mock'))
    with TestClient(app):
        pass
    assert (tmp_path / 'runtime_settings.json').read_bytes() == before


def test_a_hand_edited_profile_block_is_cleaned(tmp_path) -> None:
    assert clean_engine_profiles('nonsense') == {}
    cleaned = clean_engine_profiles(
        {'ming': {'max_parallel_requests': 8, 'nonsense': 1, 'batch_wait_ms': None}, 'tortoise': {'max_batch_size': 2}, 'mock': 'x'}
    )
    assert cleaned == {'ming': {'max_parallel_requests': 8}}


def test_provisioning_a_copied_file_keeps_the_new_gpu(tmp_path) -> None:
    """LinuxHeadless provision.sh rewrites the flat preferred_device of a file exported from another machine."""
    _write_saved(
        tmp_path, runtime_backend='omnivoice', preferred_device='cuda:0', max_parallel_requests=4,
        engine_profiles={'omnivoice': {'preferred_device': 'cuda:1', 'max_parallel_requests': 4}},
    )
    settings = _started(tmp_path)
    assert settings.preferred_device == 'cuda:0'
    assert settings.engine_profiles['omnivoice']['preferred_device'] == 'cuda:0'


def test_provisioning_also_moves_the_other_engines_off_the_old_gpu(tmp_path) -> None:
    _write_saved(
        tmp_path, runtime_backend='omnivoice', preferred_device='cuda:0', max_parallel_requests=4,
        engine_profiles={
            'omnivoice': {'preferred_device': 'cuda:1', 'max_parallel_requests': 4},
            'ming': {'preferred_device': 'cuda:1', 'max_parallel_requests': 32},
            'phonon': {'max_parallel_requests': 2},
        },
    )
    settings = _started(tmp_path)
    assert settings.engine_profiles['ming'] == {'preferred_device': 'cuda:0', 'max_parallel_requests': 32}
    assert settings.engine_profiles['phonon'] == {'max_parallel_requests': 2}, 'no device, nothing old to replace'

    pinned = _started(tmp_path, {'OMNIVOICE_TTS_RUNTIME_BACKEND': 'ming'})
    assert (pinned.runtime_backend, pinned.preferred_device, pinned.max_parallel_requests) == ('ming', 'cuda:0', 32)


def test_a_file_the_server_saved_keeps_each_engines_gpu(tmp_path) -> None:
    profiles = {'omnivoice': {'preferred_device': 'cuda:1'}, 'ming': {'preferred_device': 'cuda:0'}}
    _write_saved(tmp_path, runtime_backend='omnivoice', preferred_device='cuda:1', engine_profiles=profiles)
    settings = _started(tmp_path)
    assert settings.engine_profiles['ming']['preferred_device'] == 'cuda:0'


def test_a_saved_profile_without_a_device_says_nothing_about_a_rewrite(tmp_path) -> None:
    _write_saved(
        tmp_path, runtime_backend='omnivoice', preferred_device='cuda:1',
        engine_profiles={'omnivoice': {'max_parallel_requests': 4}, 'ming': {'preferred_device': 'cuda:0'}},
    )
    settings = _started(tmp_path)
    assert settings.engine_profiles['omnivoice']['preferred_device'] == 'cuda:1'
    assert settings.engine_profiles['ming']['preferred_device'] == 'cuda:0'


# --- B5: presets per engine -------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('values', 'expected'),
    [
        ({'max_batch_size': 4}, 'omnivoice'),
        ({'ming_max_slots': 8}, 'ming'),
        ({'runtime_backend': 'omnivoice', 'ming_max_slots': 8}, 'omnivoice'),
        ({'runtime_backend': 'ming'}, 'ming'),
        ({'runtime_backend': 'phonon'}, 'phonon'),
    ],
)
def test_preset_engine_is_derived(values, expected) -> None:
    assert presets.preset_engine(values) == expected


def test_preset_list_carries_the_engine_and_filters() -> None:
    client = make_client()
    try:
        data_dir = Path(client._temp_data_dir.name)
        presets.save_preset(data_dir, 'only ming keys', {'ming_max_slots': 8}, now_iso='2026-10-31T00:00:00+00:00')
        listed = client.get('/api/admin/settings/presets').json()['presets']
        assert {item['name']: item['engine'] for item in listed} == {
            '32GB VRAM': 'omnivoice', '24GB VRAM': 'omnivoice', '24+8 VRAM': 'omnivoice',
            'Ming 32GB VRAM': 'ming', 'Ming 24GB VRAM': 'ming', 'Ming 24+8 VRAM': 'ming', 'Ming best performance': 'ming',
            **{name: tier['values']['runtime_backend'] for name, tier in presets.AUDIO8_PRESETS.items()},
            'only ming keys': 'ming',
        }
        names = [item['name'] for item in client.get('/api/admin/settings/presets', params={'engine': 'ming'}).json()['presets']]
        assert names == ['Ming 32GB VRAM', 'Ming 24GB VRAM', 'Ming 24+8 VRAM', 'Ming best performance', 'only ming keys']
        assert [item['name'] for item in client.get('/api/admin/settings/presets', params={'engine': ' OmniVoice '}).json()['presets']] == [
            '32GB VRAM', '24GB VRAM', '24+8 VRAM',
        ]
        assert client.get('/api/admin/settings/presets', params={'engine': 'phonon'}).json()['presets'] == []

        saved = json.loads((data_dir / presets.PRESETS_FILE).read_text(encoding='utf-8'))
        assert saved['only ming keys']['values'] == {'ming_max_slots': 8} and 'engine' not in saved['only ming keys']
        response = client.put('/api/admin/settings/presets/lab', json={'values': {'runtime_backend': 'ming', 'ming_max_slots': 4}})
        assert response.json()['engine'] == 'ming'
    finally:
        _close(client)


def test_builtin_preset_values_stay_as_they_were() -> None:
    assert 'engine' not in presets.BUILTIN_PRESETS['24GB VRAM']['values']
    assert 'runtime_backend' not in presets.BUILTIN_PRESETS['24GB VRAM']['values']


# --- B6: downloading the model of an engine that is not active ----------------------------------------------------------


def test_downloadable_and_available_keep_their_meanings(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(engines, '_module_installed', lambda name: True)
    settings = Settings(runtime_backend='mock', ming_model_dir=tmp_path / 'missing', models_root_dir=tmp_path,
                        phonon_model_dir=tmp_path / 'phonon', allow_model_downloads=False)
    by_id = {info.id: info for info in engines.list_engines(settings)}
    assert by_id['ming'].downloadable is True and by_id['ming'].available is False and by_id['ming'].status == 'missing_model'
    assert by_id['phonon'].downloadable is False and by_id['phonon'].available is False
    assert 'CC-BY-NC-4.0' in by_id['phonon'].detail and 'gated' in by_id['phonon'].detail
    assert by_id['omnivoice'].downloadable is True, 'the download turns the flag on itself'
    assert by_id['omnivoice'].available is False and by_id['omnivoice'].status == 'missing_model'
    assert 'schaltet allow_model_downloads ein' in by_id['omnivoice'].detail
    settings.allow_model_downloads = True
    info = engines.engine_info(engines.ENGINES['omnivoice'], settings)
    assert info.downloadable is True and info.available is True

    (tmp_path / 'missing').mkdir()
    (tmp_path / 'missing' / 'config.json').write_text('{}', encoding='utf-8')
    (tmp_path / 'missing' / 'model.safetensors').write_bytes(b'')
    assert engines.engine_info(engines.ENGINES['ming'], settings).downloadable is False


def test_omnivoice_present_with_the_flag_off_offers_no_download(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(engines, '_module_installed', lambda name: True)
    (tmp_path / 'OmniVoice').mkdir()
    (tmp_path / 'OmniVoice' / 'config.json').write_text('{}', encoding='utf-8')
    settings = Settings(runtime_backend='mock', models_root_dir=tmp_path, allow_model_downloads=False)
    info = engines.engine_info(engines.ENGINES['omnivoice'], settings)
    assert info.downloadable is False and info.available is True


def test_omnivoice_weights_can_be_fetched_with_the_flag_off_from_another_engine(monkeypatch, tmp_path, fake_ming) -> None:  # noqa: F811
    calls = _fake_hub(monkeypatch, lambda: None)
    client = make_client(ming_model_dir=fake_ming, allow_model_downloads=False)
    try:
        client.app.state.settings.models_root_dir = tmp_path / 'no-weights-here'
        _switch(client, 'ming')

        def omnivoice() -> dict:
            return next(item for item in client.get('/api/admin/runtime/engines').json()['engines'] if item['id'] == 'omnivoice')

        assert (omnivoice()['downloadable'], omnivoice()['available']) == (True, False)
        response = client.post('/api/admin/models/download', json={'engine': 'omnivoice'})
        assert response.status_code == 200, response.text
        deadline = time.time() + 5
        while time.time() < deadline and not calls:
            time.sleep(0.01)
        assert calls and calls[0][0] == 'k2-fsa/OmniVoice'
        assert client.get('/api/admin/settings').json()['allow_model_downloads'] is True
    finally:
        _close(client)


def _fake_hub(monkeypatch, handler) -> list[tuple]:
    calls: list[tuple] = []

    def snapshot_download(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        handler()

    module = types.ModuleType('huggingface_hub')
    module.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, 'huggingface_hub', module)
    return calls


def test_ming_downloads_before_it_is_selected(monkeypatch, tmp_path) -> None:
    calls = _fake_hub(monkeypatch, lambda: None)
    target = tmp_path / 'ming'
    client = make_client(ming_model_dir=target, allow_model_downloads=False)
    try:
        response = client.post('/api/admin/models/download', json={'engine': 'ming'})
        assert response.status_code == 200, response.text
        assert response.json()['job']['repo_id'] == engines.MING_REPO_ID
        assert response.json()['job']['model_id'] == engines.MING_BASE_ALIAS
        deadline = time.time() + 5
        while time.time() < deadline and not calls:
            time.sleep(0.01)
        assert calls == [(engines.MING_REPO_ID, {'local_dir': str(target)})]
        assert client.get('/api/admin/settings').json()['allow_model_downloads'] is False, "OmniVoice's flag stays"
        assert client.get('/api/admin/settings').json()['runtime_backend'] == 'mock'
    finally:
        _close(client)


def test_a_running_and_a_failed_ming_download_show_on_the_engine_list(monkeypatch, tmp_path) -> None:
    import threading

    release = threading.Event()

    def blocked() -> None:
        release.wait(5)
        raise RuntimeError('no route to hub')

    _fake_hub(monkeypatch, blocked)
    monkeypatch.setattr(engines, '_module_installed', lambda name: True)
    client = make_client(ming_model_dir=tmp_path / 'ming')
    try:
        def ming() -> dict:
            return next(item for item in client.get('/api/admin/runtime/engines').json()['engines'] if item['id'] == 'ming')

        assert client.post('/api/admin/models/download', json={'engine': 'ming'}).status_code == 200
        running = ming()
        assert (running['status'], running['downloadable'], running['available']) == ('downloading', False, False)
        release.set()
        deadline = time.time() + 5
        while time.time() < deadline and 'fehlgeschlagen' not in (ming()['detail'] or ''):
            time.sleep(0.01)
        failed = ming()
        assert 'no route to hub' in failed['detail'] and failed['downloadable'] is True
        assert failed['status'] == 'missing_model'
    finally:
        _close(client)


def test_phonon_stays_a_manual_download() -> None:
    client = make_client()
    try:
        response = client.post('/api/admin/models/download', json={'engine': 'phonon'})
        assert response.status_code == 400
        assert 'by hand' in response.json()['detail'] and 'CC-BY-NC-4.0' in response.json()['detail']
    finally:
        _close(client)


@pytest.mark.parametrize('engine', ['mock', 'tortoise'])
def test_other_engines_cannot_be_downloaded(engine) -> None:
    client = make_client()
    try:
        assert client.post('/api/admin/models/download', json={'engine': engine}).status_code == 400
    finally:
        _close(client)


def test_a_download_without_engine_still_needs_a_known_model() -> None:
    client = make_client()
    try:
        assert client.post('/api/admin/models/download', json={'model_id': 'nope'}).status_code == 400
        assert client.post('/api/admin/models/download', json={'model_id': OMNIVOICE_AUTO_ALIAS}).status_code == 200
    finally:
        _close(client)


# --- B7: client discovery ---------------------------------------------------------------------------------------------

CAPABILITY_KEYS = {
    'engine', 'engine_label', 'model', 'loaded', 'sample_rate', 'voice_cloning', 'voice_profiles', 'streaming',
    'languages', 'response_formats', 'options', 'option_notes',
}
OPTION_KEYS = {
    'speed', 'seed', 'language', 'instructions', 'emotion', 'speaking_rate', 'pitch', 'volume', 'reference_mode',
}
DELIVERY_OPTIONS = {'emotion', 'speaking_rate', 'pitch', 'volume', 'reference_mode'}


def _capabilities_of(client) -> dict:
    response = client.get('/api/v1/capabilities')
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == CAPABILITY_KEYS and set(body['options']) == OPTION_KEYS
    assert body['option_notes'] == {}, 'only the Audio8 engines carry a note so far'
    assert client.get('/v1/capabilities').json() == body
    return body


def test_capabilities_are_open_like_the_other_get_routes() -> None:
    client = _base_client()
    try:
        client.post('/api/admin/auth/logout')
        client.cookies.clear()
        assert client.get('/api/admin/settings').status_code == 401
        assert client.get('/api/v1/capabilities').status_code == 200
        assert client.get('/v1/capabilities').status_code == 200
    finally:
        _close(client)


def test_capabilities_of_omnivoice_base() -> None:
    client = _base_client()
    try:
        client.app.state.settings.runtime_backend = 'omnivoice'
        client.app.state.synthesizer._loaded_model_id = OMNIVOICE_BASE_ALIAS
        body = _capabilities_of(client)
        assert (body['engine'], body['engine_label'], body['model']) == ('omnivoice', 'OmniVoice', OMNIVOICE_BASE_ALIAS)
        assert body['loaded'] is True and body['sample_rate'] == 24_000
        assert (body['voice_cloning'], body['voice_profiles'], body['streaming']) == (True, True, 'sentence')
        assert 'Deutsch' in body['languages'] and 'Polski' in body['languages']
        enabled = {name for name, value in body['options'].items() if value}
        assert enabled == {'speed', 'seed', 'language'}, 'a Base clone takes no instructions'
        client.app.state.settings.runtime_backend = 'mock'
    finally:
        _close(client)


def test_capabilities_of_a_finetuned_selection_promise_clones_no_instructions() -> None:
    client = _base_client(active_model=LOCAL, supported_models=[*ALL_ALIASES, LOCAL])
    try:
        client.app.state.settings.runtime_backend = 'omnivoice'
        client.app.state.synthesizer._loaded_model_id = None
        body = _capabilities_of(client)
        assert body['model'] == LOCAL and body['loaded'] is False
        assert body['options']['instructions'] is False, "a Base request clones on it, each request's task type decides"
        client.app.state.settings.runtime_backend = 'mock'
    finally:
        _close(client)


@pytest.mark.parametrize('alias', [OMNIVOICE_AUTO_ALIAS, OMNIVOICE_DESIGN_ALIAS])
def test_capabilities_of_a_saved_alias_selection_read_instructions(alias) -> None:
    client = _base_client(active_model=alias)
    try:
        client.app.state.settings.runtime_backend = 'omnivoice'
        client.app.state.synthesizer._loaded_model_id = None
        assert _capabilities_of(client)['options']['instructions'] is True, 'nothing migrates a saved alias'
        client.app.state.settings.runtime_backend = 'mock'
    finally:
        _close(client)


def test_capabilities_of_ming(fake_ming) -> None:  # noqa: F811
    client = make_client(ming_model_dir=fake_ming)
    try:
        _switch(client, 'ming')
        body = _capabilities_of(client)
        assert (body['engine'], body['engine_label'], body['model']) == ('ming', 'Ming-3.0-tts 0.5B', ALIAS)
        assert (body['voice_cloning'], body['voice_profiles'], body['streaming']) == (True, True, 'native')
        assert body['sample_rate'] == 24_000
        assert body['languages'] == ['Auto', 'Deutsch', 'English', 'Français', 'Español', '中文', '日本語', '한국어']
        options = body['options']
        assert options['speed'] is False, 'speaking_rate is the Ming control, speed is not read'
        assert all(options[name] for name in ('seed', 'language', 'instructions', *DELIVERY_OPTIONS))
    finally:
        _close(client)


def test_capabilities_of_phonon(monkeypatch, tmp_path) -> None:
    class FakePhonon(MockSynthesizer):
        def __init__(self, settings, store) -> None:
            super().__init__(sample_rate=24_000)
            self.model_id = settings.phonon_model_id

        def load_now(self) -> None:
            pass

        def voice_names(self) -> list[str]:
            return ['Catalog']

    module = types.ModuleType('omnivoice_tts_server.phonon_runtime')
    module.PhononSynthesizer = FakePhonon
    monkeypatch.setitem(sys.modules, 'omnivoice_tts_server.phonon_runtime', module)
    app = create_app(Settings(runtime_backend='phonon', data_dir=tmp_path, phonon_lang='en'))
    with TestClient(app) as client:
        body = client.get('/v1/capabilities').json()
    assert (body['engine'], body['model']) == ('phonon', 'gradium/phonon-40m-multilingual')
    assert (body['voice_cloning'], body['voice_profiles'], body['streaming']) == (False, False, 'native')
    assert body['languages'] == ['English']
    assert {name for name, value in body['options'].items() if value} == {'seed'}, 'language and speed are not read'


def test_capabilities_of_the_mock_promise_nothing() -> None:
    client = make_client()
    try:
        body = _capabilities_of(client)
        assert body['engine'] == 'mock' and body['voice_cloning'] is False
        assert not any(body['options'].values())
    finally:
        _close(client)


def test_response_formats_follow_the_mp3_encoder(monkeypatch) -> None:
    from omnivoice_tts_server.api import router_v2

    client = make_client()
    try:
        monkeypatch.setattr(router_v2, 'mp3_export_available', lambda: True)
        assert _capabilities_of(client)['response_formats'] == ['wav', 'mp3', 'pcm']
        monkeypatch.setattr(router_v2, 'mp3_export_available', lambda: False)
        assert _capabilities_of(client)['response_formats'] == ['wav', 'pcm']
    finally:
        _close(client)


def test_capabilities_answer_during_a_switch_without_waiting(fake_ming) -> None:  # noqa: F811
    client = make_client(ming_model_dir=fake_ming)
    try:
        old = GatedSynthesizer()
        state = client.app.state
        state.synthesizer = old
        state.queue_service.synthesizer = old
        client.post('/v1/jobs', json={'input': 'Haengt.', 'voice': 'Auto Voice'})
        assert old.started.wait(5)
        assert client.post('/api/admin/runtime/backend', json={'backend': 'ming'}).status_code == 200

        started = time.perf_counter()
        body = _capabilities_of(client)
        assert time.perf_counter() - started < 2, 'a client must not wait for the drain'
        assert (body['engine'], body['model'], body['loaded']) == ('ming', ALIAS, False)

        old.gate.set()
        assert _wait_switch(client)['switch']['state'] == 'ready'
        assert _capabilities_of(client)['engine'] == 'ming'
    finally:
        _close(client)


def test_no_existing_response_gains_the_engine() -> None:
    client = make_client()
    try:
        assert 'engine' not in client.get('/api/v1/models').json()[0]
        assert 'engine' not in client.get('/v1/models').json()['data'][0]
        assert 'engine' not in client.post('/api/v1/synthesize', json={'input': 'Hallo.'}).json()
    finally:
        _close(client)
