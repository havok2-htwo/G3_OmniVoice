"""The four Audio8 engines through the server: registry, switch, profiles, settings, presets, seeds, downloads and
capabilities, with a fake synthesizer. test_audio8_wiring_gpu.py runs the same routes on the real models."""
from __future__ import annotations

import json
import re
import sys
import time
import types
from pathlib import Path
from typing import get_args

import pytest
from fastapi.testclient import TestClient

from omnivoice_tts_server import engines, presets
from omnivoice_tts_server.app import create_app
from omnivoice_tts_server.audio8.variants import AUDIO8_VARIANTS
from omnivoice_tts_server.audio8.voice_cache import voice_cache_folder, voice_tag
from omnivoice_tts_server.config import (
    AUDIO8_SETTINGS_FIELDS,
    ENGINE_TUNED_FIELDS,
    RUNTIME_BACKENDS,
    RUNTIME_SETTINGS_FIELDS,
    RUNTIME_SETTINGS_FILE,
    Settings,
    clean_engine_profiles,
    load_runtime_settings,
    save_runtime_settings,
)
from omnivoice_tts_server.domain.models import RuntimeBackendName, ServerSettingsUpdateRequest, SpeechRequest
from omnivoice_tts_server.domain.state import InMemoryStore, seed_engine_key
from omnivoice_tts_server.runtime_v2 import MockSynthesizer, build_synthesizer
from omnivoice_tts_server.text_optimizer.omnivoice_tags import render_inline_tags

from test_app import make_client
from test_engine_switch import _close, _saved, _switch
from test_engine_voice_seeds import _voice
from test_model_selection import _add_voice, _fake_hub

ENGINE_IDS = list(AUDIO8_VARIANTS)
GPU_IDS = [engine for engine in ENGINE_IDS if engine != 'audio8_onnx']


class FakeAudio8Synthesizer(MockSynthesizer):
    """Stands in for audio8_runtime.Audio8Synthesizer: tone output at the configured rate."""

    supports_native_streaming = False
    supports_concurrent_batches = True
    uses_voice_profiles = True
    instances: list['FakeAudio8Synthesizer'] = []

    def __init__(self, settings, store, variant) -> None:
        super().__init__(sample_rate=settings.audio8_output_sample_rate)
        self.base_alias = AUDIO8_VARIANTS[variant].base_alias
        self.variant = variant
        self.slots = settings.audio8_max_slots
        self.seeds: list[int | None] = []
        self.loaded = False
        FakeAudio8Synthesizer.instances.append(self)

    async def ensure_model(self, requested_model):
        self.loaded = True
        return self.base_alias, 0

    def resident_models(self) -> set[str]:
        return {self.base_alias} if self.loaded else set()

    async def render_batch(self, items, trim_after: bool = True):
        self.seeds.extend(item.request.seed for item in items)
        return await super().render_batch(items)

    async def unload(self):
        self.loaded = False
        return self.base_alias, 0


def _write_weights(models_root: Path, engine: str, *, skip: str | None = None) -> Path:
    variant = AUDIO8_VARIANTS[engine]
    folder = models_root / variant.folder
    for name in variant.weight_files:
        if name != skip:
            path = folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'{}')
    return folder


@pytest.fixture
def models_root(monkeypatch, tmp_path) -> Path:
    module = types.ModuleType('omnivoice_tts_server.audio8_runtime')
    module.Audio8Synthesizer = FakeAudio8Synthesizer
    monkeypatch.setitem(sys.modules, 'omnivoice_tts_server.audio8_runtime', module)
    monkeypatch.setattr(engines, '_module_installed', lambda name: True)
    monkeypatch.delenv('OMNIVOICE_TTS_RUNTIME_BACKEND', raising=False)
    FakeAudio8Synthesizer.instances = []
    root = tmp_path / 'models'
    for engine in ENGINE_IDS:
        _write_weights(root, engine)
    return root


def _client(models_root: Path, **overrides):
    client = make_client(**overrides)
    client.app.state.settings.models_root_dir = models_root
    return client


def _engine(client, engine: str) -> dict:
    return next(item for item in client.get('/api/admin/runtime/engines').json()['engines'] if item['id'] == engine)


def _wait_for(calls: list) -> None:
    deadline = time.time() + 5
    while time.time() < deadline and not calls:
        time.sleep(0.01)


# --- registry ------------------------------------------------------------------------------------------------------


def test_every_name_list_knows_the_four_engines() -> None:
    assert set(ENGINE_IDS) == {'audio8_realistic', 'audio8_preview', 'audio8_mini', 'audio8_onnx'}
    for names in (RUNTIME_BACKENDS, engines.ENGINES, engines.ENGINE_SETTINGS_FIELDS, get_args(RuntimeBackendName)):
        assert set(ENGINE_IDS) <= set(names)
    assert set(engines.ENGINE_DEFAULT_TIERS) >= set(ENGINE_IDS)
    assert all(seed_engine_key(engine) == engine for engine in ENGINE_IDS)


@pytest.mark.parametrize('engine', ENGINE_IDS)
def test_an_engine_is_one_spec_of_its_variant(engine) -> None:
    spec, variant = engines.ENGINES[engine], AUDIO8_VARIANTS[engine]
    assert (spec.label, spec.license, spec.approx_size_gb) == (variant.label, variant.license, variant.approx_size_gb)
    assert spec.uses_voice_profiles and spec.native_streaming and spec.preload and spec.voice_cloning
    assert not spec.native_model_ids
    assert spec.client_options == ('seed', 'language')
    assert spec.cpu_only is (engine == 'audio8_onnx')
    assert spec.license_restricted is (engine == 'audio8_mini')
    assert ('onnxruntime' in spec.packages) is (engine == 'audio8_onnx')
    assert variant.base_alias.endswith('-Base')


def test_the_cpu_engine_offers_no_gpu_settings_and_the_gpu_engines_no_cpu_settings() -> None:
    for engine in GPU_IDS:
        fields = engines.ENGINE_SETTINGS_FIELDS[engine]
        assert 'preferred_device' in fields and 'audio8_max_slots' in fields
        assert not [name for name in fields if name.startswith('audio8_onnx_')]
    fields = engines.ENGINE_SETTINGS_FIELDS['audio8_onnx']
    assert 'preferred_device' not in fields and 'audio8_max_slots' not in fields and 'audio8_cuda_graphs' not in fields
    assert {'audio8_onnx_workers', 'audio8_onnx_threads', 'audio8_output_sample_rate', 'audio8_top_k'} <= set(fields)
    assert set(AUDIO8_SETTINGS_FIELDS) == {
        name for engine in ENGINE_IDS for name in engines.ENGINE_SETTINGS_FIELDS[engine] if name.startswith('audio8_')
    }


@pytest.mark.parametrize('engine', ENGINE_IDS)
def test_engine_list_shows_folder_licence_and_ready(models_root, engine) -> None:
    client = _client(models_root)
    try:
        info = _engine(client, engine)
        variant = AUDIO8_VARIANTS[engine]
        assert info['model_id'] == variant.repo_id
        assert Path(info['model_dir']) == models_root / variant.folder
        assert (info['status'], info['available'], info['downloadable']) == ('ready', True, False)
        assert (info['license'], info['license_restricted']) == (variant.license, variant.license_restricted)
        assert info['cpu_only'] is (engine == 'audio8_onnx')
        assert info['approx_size_gb'] == variant.approx_size_gb
    finally:
        _close(client)


def test_licences_of_the_other_engines_show_too(models_root) -> None:
    client = _client(models_root)
    try:
        assert _engine(client, 'ming')['license'] == 'MIT'
        phonon = _engine(client, 'phonon')
        assert (phonon['license_restricted'], phonon['cpu_only']) == (True, True)
        assert _engine(client, 'omnivoice')['cpu_only'] is False
    finally:
        _close(client)


@pytest.mark.parametrize('engine', ENGINE_IDS)
def test_one_missing_weights_file_means_the_model_is_missing(monkeypatch, tmp_path, engine) -> None:
    variant = AUDIO8_VARIANTS[engine]
    root = tmp_path / 'models'
    _write_weights(root, engine, skip=variant.weight_files[-1])
    monkeypatch.setattr(engines, '_module_installed', lambda name: True)
    info = engines.engine_info(engines.ENGINES[engine], Settings(models_root_dir=root, data_dir=tmp_path))
    assert (info.model_present, info.available, info.downloadable, info.status) == (False, False, True, 'missing_model')
    assert f'--revision {variant.revision}' in info.detail
    assert variant.folder in info.detail


def test_a_missing_package_names_the_extra(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(engines, '_module_installed', lambda name: name != 'onnx')
    info = engines.engine_info(engines.ENGINES['audio8_onnx'], Settings(models_root_dir=tmp_path, data_dir=tmp_path))
    assert info.status == 'missing_package'
    assert 'onnx' in info.detail and 'backend[audio8]' in info.detail


def test_a_switch_to_a_model_that_is_not_there_is_refused(models_root) -> None:
    client = _client(models_root)
    try:
        (models_root / AUDIO8_VARIANTS['audio8_mini'].folder / 'codec.pth').unlink()
        response = client.post('/api/admin/runtime/backend', json={'backend': 'audio8_mini'})
        assert response.status_code == 400
        assert 'not available' in response.json()['detail']
        assert client.app.state.settings.runtime_backend == 'mock'
    finally:
        _close(client)


# --- the factory, with the real synthesizer class -----------------------------------------------------------------


@pytest.mark.parametrize('engine', ENGINE_IDS)
def test_build_synthesizer_serves_the_variant_through_its_alias(tmp_path, engine) -> None:
    pytest.importorskip('torch')
    settings = Settings(runtime_backend=engine, models_root_dir=tmp_path, data_dir=tmp_path)
    store = InMemoryStore(max_queue_size=8)
    synthesizer = build_synthesizer(settings, store)
    variant = AUDIO8_VARIANTS[engine]
    assert type(synthesizer).__name__ == 'Audio8Synthesizer' and synthesizer.variant is variant
    assert settings.supported_models == [variant.base_alias]
    assert settings.active_model == variant.base_alias == store.active_model
    assert synthesizer.capacity_profile['engine'] == engine
    assert synthesizer.sample_rate == 44100
    assert engines.engine_model(settings, engine) == variant.base_alias


# --- live switching and profiles --------------------------------------------------------------------------------------


@pytest.mark.parametrize('engine', ENGINE_IDS)
def test_switch_there_and_back(models_root, engine) -> None:
    client = _client(models_root, default_voice='Anna')
    settings = client.app.state.settings
    alias = AUDIO8_VARIANTS[engine].base_alias
    try:
        _switch(client, engine)
        synthesizer = client.app.state.synthesizer
        assert isinstance(synthesizer, FakeAudio8Synthesizer) and synthesizer.variant == engine
        assert (settings.active_model, settings.supported_models) == (alias, [alias])
        assert client.app.state.store.active_model == alias
        assert _engine(client, engine)['loaded'] is True, 'preloaded'
        saved = _saved(client)
        assert saved['runtime_backend'] == engine
        assert saved['active_model'] == 'k2-fsa/OmniVoice-AutoVoice', "the engine's alias is never saved"

        _switch(client, 'mock')
        assert settings.active_model == 'k2-fsa/OmniVoice-AutoVoice'
        assert alias not in settings.supported_models
        assert synthesizer.loaded is False
    finally:
        _close(client)


def test_a_switch_between_two_audio8_engines_builds_the_new_one(models_root) -> None:
    client = _client(models_root)
    try:
        _switch(client, 'audio8_realistic')
        first = client.app.state.synthesizer
        _switch(client, 'audio8_mini')
        assert first.loaded is False
        assert client.app.state.synthesizer.variant == 'audio8_mini'
        assert client.app.state.settings.supported_models == [AUDIO8_VARIANTS['audio8_mini'].base_alias]
    finally:
        _close(client)


@pytest.mark.parametrize(('engine', 'tier'), [
    ('audio8_realistic', 'Audio8 24GB VRAM'),
    ('audio8_preview', 'Audio8 Preview 24GB VRAM'),
    ('audio8_mini', 'Audio8 Mini 24GB VRAM'),
])
def test_a_first_switch_starts_from_the_default_tier(models_root, engine, tier) -> None:
    client = _client(models_root)
    settings = client.app.state.settings
    try:
        settings.audio8_output_sample_rate = 24000
        _switch(client, engine)
        assert (settings.audio8_max_slots, settings.max_parallel_requests, settings.batch_wait_ms) == (16, 16, 0)
        assert client.app.state.synthesizer.slots == 16
        assert engines.ENGINE_DEFAULT_TIERS[engine] == tier
        assert settings.audio8_output_sample_rate == 24000, 'the rate clients get is not tuning, a saved one stays'
    finally:
        _close(client)


def test_the_cpu_engine_starts_with_one_request_per_worker(models_root) -> None:
    client = _client(models_root)
    settings = client.app.state.settings
    try:
        _switch(client, 'audio8_onnx')
        assert (settings.audio8_onnx_workers, settings.max_parallel_requests) == (2, 2)
        assert settings.audio8_max_slots == 32, 'GPU slots stay untouched'
    finally:
        _close(client)


def test_each_engine_keeps_its_own_profile(models_root) -> None:
    client = _client(models_root)
    settings = client.app.state.settings
    try:
        _switch(client, 'audio8_realistic')
        assert client.put('/api/admin/settings', json={'max_parallel_requests': 7, 'stream_prebuffer_ms': 150}).status_code == 200
        _switch(client, 'audio8_preview')
        assert settings.max_parallel_requests == 16
        client.put('/api/admin/settings', json={'stream_prebuffer_ms': 40})
        _switch(client, 'audio8_onnx')
        assert settings.max_parallel_requests == 2
        _switch(client, 'audio8_realistic')
        assert (settings.max_parallel_requests, settings.stream_prebuffer_ms) == (7, 150)
        assert {'audio8_realistic', 'audio8_preview', 'audio8_onnx'} <= set(settings.engine_profiles)
        assert set(settings.engine_profiles['audio8_onnx']) == set(ENGINE_TUNED_FIELDS)
    finally:
        _close(client)


def _audio8_values(settings) -> tuple:
    return (settings.audio8_max_slots, settings.max_parallel_requests, settings.audio8_first_chunk_frames,
            settings.audio8_chunk_frames, settings.audio8_cuda_graphs)


def test_slots_and_chunks_stay_with_their_engine_across_a_first_switch(models_root) -> None:
    """Regression: the profile restored the parallel requests but not the slots of the tier, so 32/32 came back as 16/32."""
    client = _client(models_root)
    settings = client.app.state.settings
    try:
        _switch(client, 'audio8_realistic')
        response = client.put('/api/admin/settings', json={
            'audio8_max_slots': 32, 'max_parallel_requests': 32, 'audio8_first_chunk_frames': 6,
            'audio8_chunk_frames': 16, 'audio8_cuda_graphs': False})
        assert response.status_code == 200, response.text
        chosen = (32, 32, 6, 16, False)
        assert _audio8_values(settings) == chosen
        _switch(client, 'audio8_mini')
        assert _audio8_values(settings) == (16, 16, 4, 12, True), 'its own tier, nothing of the other engine'
        _switch(client, 'audio8_realistic')
        assert _audio8_values(settings) == chosen
        assert client.app.state.synthesizer.slots == 32
        _switch(client, 'audio8_mini')
        assert _audio8_values(settings) == (16, 16, 4, 12, True)
    finally:
        _close(client)


def test_saved_profiles_of_the_audio8_engines_survive_a_restart(tmp_path) -> None:
    profiles = clean_engine_profiles({engine: {'max_parallel_requests': 3} for engine in (*ENGINE_IDS, 'tortoise')})
    assert set(profiles) == set(ENGINE_IDS)
    kept = clean_engine_profiles({'audio8_mini': {'audio8_max_slots': 8, 'audio8_cuda_graphs': False}})
    assert kept == {'audio8_mini': {'audio8_max_slots': 8, 'audio8_cuda_graphs': False}}


def test_a_profile_saved_without_slots_keeps_the_current_value() -> None:
    settings = Settings(audio8_max_slots=24, engine_profiles={'audio8_mini': {'max_parallel_requests': 8}})
    assert engines.engine_start_values(settings, 'audio8_mini') == {'max_parallel_requests': 8}


def test_the_saved_engine_comes_back_after_a_restart(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path, runtime_backend='audio8_mini')
    save_runtime_settings(settings)
    reloaded = Settings(data_dir=tmp_path)
    load_runtime_settings(reloaded, environ={})
    assert reloaded.runtime_backend == 'audio8_mini'


@pytest.mark.parametrize('engine', ENGINE_IDS)
def test_a_server_can_start_on_an_audio8_engine(models_root, tmp_path, engine) -> None:
    data_dir = tmp_path / 'data'
    app = create_app(Settings(runtime_backend=engine, models_root_dir=models_root, data_dir=data_dir))
    with TestClient(app) as client:
        settings = client.app.state.settings
        alias = AUDIO8_VARIANTS[engine].base_alias
        assert client.app.state.synthesizer.variant == engine
        assert (settings.active_model, settings.supported_models) == (alias, [alias])
        expected = engines.default_tier_values(engine)
        assert {name: getattr(settings, name) for name in expected} == expected
        assert set(ENGINE_TUNED_FIELDS) == set(settings.engine_profiles[engine])
        assert client.get('/api/v1/capabilities').json()['engine'] == engine
        save_runtime_settings(settings)
    saved = json.loads((data_dir / RUNTIME_SETTINGS_FILE).read_text(encoding='utf-8'))
    assert saved['runtime_backend'] == engine
    assert saved['active_model'] == 'k2-fsa/OmniVoice-Base', "the engine's alias is never saved"


def test_a_foreign_model_id_is_served_by_the_active_engine(models_root) -> None:
    client = _client(models_root)
    settings = client.app.state.settings
    try:
        _switch(client, 'audio8_preview')
        alias = AUDIO8_VARIANTS['audio8_preview'].base_alias
        assert engines.map_request_model(settings, 'k2-fsa/OmniVoice-Base') is None
        assert engines.map_request_model(settings, alias.upper()) == alias
        assert engines.finetune_supported(settings) is False
    finally:
        _close(client)


# --- settings ------------------------------------------------------------------------------------------------------


AUDIO8_VALUES = {
    'audio8_max_slots': 8, 'audio8_cuda_graphs': False, 'audio8_output_sample_rate': 24000, 'audio8_first_chunk_frames': 6,
    'audio8_chunk_frames': 16, 'audio8_fragment_max_chars': 240, 'audio8_temperature': 0.7, 'audio8_top_p': 0.9,
    'audio8_top_k': 40, 'audio8_onnx_workers': 3, 'audio8_onnx_threads': 4, 'audio8_onnx_first_chunk_frames': 8,
    'audio8_onnx_chunk_frames': 10,
}


def test_the_value_table_covers_every_setting() -> None:
    assert set(AUDIO8_VALUES) == set(AUDIO8_SETTINGS_FIELDS)
    assert set(AUDIO8_SETTINGS_FIELDS) <= RUNTIME_SETTINGS_FIELDS
    assert set(AUDIO8_SETTINGS_FIELDS) <= set(ServerSettingsUpdateRequest.model_fields)


def test_settings_put_get_and_persist(models_root) -> None:
    client = _client(models_root)
    try:
        response = client.put('/api/admin/settings', json=AUDIO8_VALUES)
        assert response.status_code == 200, response.text
        body = client.get('/api/admin/settings').json()
        assert {name: body[name] for name in AUDIO8_VALUES} == AUDIO8_VALUES
        saved = _saved(client)
        assert {name: saved[name] for name in AUDIO8_VALUES} == AUDIO8_VALUES
        reloaded = Settings(data_dir=Path(client._temp_data_dir.name))
        load_runtime_settings(reloaded, environ={})
        assert {name: getattr(reloaded, name) for name in AUDIO8_VALUES} == AUDIO8_VALUES
    finally:
        _close(client)


def test_zero_threads_is_a_value_not_a_missing_field(models_root) -> None:
    client = _client(models_root)
    try:
        client.app.state.settings.audio8_onnx_threads = 5
        assert client.put('/api/admin/settings', json={'audio8_onnx_threads': 0}).status_code == 200
        assert client.app.state.settings.audio8_onnx_threads == 0
    finally:
        _close(client)


@pytest.mark.parametrize(('name', 'value'), [
    ('audio8_max_slots', 0), ('audio8_max_slots', 65), ('audio8_output_sample_rate', 48000),
    ('audio8_first_chunk_frames', 0), ('audio8_first_chunk_frames', 13), ('audio8_chunk_frames', 33),
    ('audio8_fragment_max_chars', 23), ('audio8_fragment_max_chars', 401), ('audio8_temperature', 0),
    ('audio8_temperature', 2.5), ('audio8_top_p', 0), ('audio8_top_p', 1.5), ('audio8_top_k', 0), ('audio8_top_k', 2000),
    ('audio8_onnx_workers', 0), ('audio8_onnx_workers', 17), ('audio8_onnx_threads', -1), ('audio8_onnx_threads', 65),
    ('audio8_onnx_first_chunk_frames', 33), ('audio8_onnx_chunk_frames', 65),
])
def test_settings_put_refuses_values_the_settings_model_refuses(models_root, name, value) -> None:
    client = _client(models_root)
    try:
        before = getattr(client.app.state.settings, name)
        assert client.put('/api/admin/settings', json={name: value}).status_code == 422
        assert getattr(client.app.state.settings, name) == before
        with pytest.raises(ValueError):
            Settings(**{name: value})
    finally:
        _close(client)


def test_a_put_can_switch_to_an_audio8_engine_and_tune_it(models_root) -> None:
    client = _client(models_root)
    try:
        response = client.put('/api/admin/settings', json={'runtime_backend': 'audio8_mini', 'audio8_max_slots': 4})
        assert response.status_code == 200, response.text
        deadline = time.time() + 5
        while time.time() < deadline and client.app.state.settings.runtime_backend != 'audio8_mini':
            time.sleep(0.01)
        while time.time() < deadline and client.app.state.engine_switcher.switching:
            time.sleep(0.01)
        assert client.app.state.settings.runtime_backend == 'audio8_mini'
        assert client.app.state.settings.audio8_max_slots == 4, 'a value sent with the switch beats the default tier'
    finally:
        _close(client)


# --- output sample rate --------------------------------------------------------------------------------------------


def test_settings_tell_the_frontend_the_rate_of_the_active_engine(models_root) -> None:
    client = _client(models_root)
    try:
        assert client.get('/api/admin/settings').json()['output_sample_rate'] == 24000
        _switch(client, 'audio8_realistic')
        assert client.get('/api/admin/settings').json()['output_sample_rate'] == 44100
        client.put('/api/admin/settings', json={'audio8_output_sample_rate': 24000})
        _switch(client, 'audio8_preview')
        assert client.get('/api/admin/settings').json()['output_sample_rate'] == 24000
    finally:
        _close(client)


def test_a_pending_switch_reports_the_rate_of_the_target() -> None:
    settings = Settings(audio8_output_sample_rate=24000, ming_output_sample_rate=44100)
    assert engines.engine_output_sample_rate(settings, 'audio8_onnx') == 24000
    assert engines.engine_output_sample_rate(settings, 'ming') == 44100
    assert engines.engine_output_sample_rate(settings, 'omnivoice') == settings.sample_rate


# --- capabilities -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize('engine', ENGINE_IDS)
def test_capabilities_of_an_audio8_engine(models_root, engine) -> None:
    client = _client(models_root)
    try:
        _switch(client, engine)
        body = client.get('/api/v1/capabilities').json()
        assert body == client.get('/v1/capabilities').json()
        assert body['engine'] == engine
        assert body['engine_label'] == AUDIO8_VARIANTS[engine].label
        assert body['model'] == AUDIO8_VARIANTS[engine].base_alias
        assert (body['sample_rate'], body['streaming'], body['loaded']) == (44100, 'native', True)
        assert (body['voice_cloning'], body['voice_profiles']) == (True, True)
        assert {name for name, honored in body['options'].items() if honored} == {'seed', 'language'}
        assert body['languages'] == ['Auto'], 'the text decides, no client language is read'
        assert 'no language parameter' in body['option_notes']['language']
    finally:
        _close(client)


def test_other_engines_carry_no_option_notes(models_root) -> None:
    client = _client(models_root)
    try:
        assert client.get('/api/v1/capabilities').json()['option_notes'] == {}
    finally:
        _close(client)


# --- per-engine voice seeds ------------------------------------------------------------------------------------------


def test_a_voice_has_one_seed_per_audio8_engine() -> None:
    profile = _voice(9305, audio8_realistic=11, audio8_onnx=22)
    assert profile.seed_for_engine('audio8_realistic') == 11
    assert profile.seed_for_engine('audio8_onnx') == 22
    assert profile.seed_for_engine('audio8_mini') is None
    assert profile.all_engine_seeds() == {'omnivoice': 9305, 'audio8_realistic': 11, 'audio8_onnx': 22}


@pytest.mark.parametrize('active', ['omnivoice', 'audio8_onnx', 'audio8_realistic'])
def test_deleting_a_voice_removes_its_codes_whichever_engine_runs(models_root, active) -> None:
    """Regression: the file outlived the voice when the engine had been switched or restarted since it was written."""
    client = _client(models_root, default_voice='Anna')
    try:
        voice_id = _add_voice(client)
        if active != 'omnivoice':
            _switch(client, active)
        folder = voice_cache_folder(client.app.state.settings.data_dir)
        folder.mkdir(parents=True, exist_ok=True)
        codes = folder / f'{voice_tag(voice_id)}.abc.npy'
        other = folder / f'{voice_tag("voice_other")}.abc.npy'
        codes.write_bytes(b'x')
        other.write_bytes(b'x')
        assert client.delete(f'/api/admin/voices/{voice_id}').status_code == 200
        assert not codes.exists() and other.exists()
    finally:
        _close(client)


def test_a_file_that_cannot_be_removed_does_not_fail_the_delete(models_root, monkeypatch) -> None:
    client = _client(models_root, default_voice='Anna')
    try:
        voice_id = _add_voice(client)
        folder = voice_cache_folder(client.app.state.settings.data_dir)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f'{voice_tag(voice_id)}.abc.npy').write_bytes(b'x')

        def locked(self, missing_ok=False):
            raise PermissionError('in use')

        with monkeypatch.context() as patch:
            patch.setattr(Path, 'unlink', locked)
            assert client.delete(f'/api/admin/voices/{voice_id}').status_code == 200
        assert voice_id not in client.app.state.store.voice_profiles
    finally:
        _close(client)


def test_the_seed_of_the_active_audio8_engine_reaches_the_renderer(models_root) -> None:
    client = _client(models_root, default_voice='Anna')
    try:
        voice_id = _add_voice(client)
        for engine, seed in (('audio8_realistic', 101), ('audio8_mini', 202)):
            response = client.patch(f'/api/admin/voices/{voice_id}', json={'seed': seed, 'engine': engine})
            assert response.status_code == 200, response.text
        client.app.state.store.save_voices(client.app.state.settings.data_dir)
        for engine, seed in (('audio8_realistic', 101), ('audio8_mini', 202)):
            _switch(client, engine)
            response = client.post('/api/v1/synthesize', json={'input': 'Hallo Welt.', 'voice': 'Anna'})
            assert response.status_code == 200, response.text
            assert client.app.state.synthesizer.seeds == [seed]
    finally:
        _close(client)


# --- downloads -------------------------------------------------------------------------------------------------------


def test_the_installer_downloads_the_pinned_revisions_and_checks_the_same_files_as_the_server() -> None:
    """Regression: it only looked for config.json, so a download cut off before the weights counted as complete."""
    script = (Path(__file__).resolve().parents[2] / 'install.ps1').read_text(encoding='utf-8')
    found = re.findall(r"@\{ Repo = '([^']+)'; Folder = '([^']+)'; Revision = '([^']+)'; Weights = @\(([^)]*)\) \}", script)
    assert len(found) == len(AUDIO8_VARIANTS)
    table = {repo: (folder, revision, tuple(re.findall(r"'([^']+)'", weights))) for repo, folder, revision, weights in found}
    for variant in AUDIO8_VARIANTS.values():
        assert table[variant.repo_id] == (variant.folder, variant.revision, variant.weight_files)
    audio8_part = script[script.index('$Audio8Models'):script.index('$PhononDir')]
    ming_hint = "Test-Path (Join-Path $MingDir 'config.json')"
    assert "'config.json'))" not in audio8_part.replace(ming_hint, '')


@pytest.mark.parametrize('engine', ENGINE_IDS)
def test_a_download_fetches_exactly_the_pinned_revision(monkeypatch, tmp_path, engine) -> None:
    calls = _fake_hub(monkeypatch, lambda: None)
    variant = AUDIO8_VARIANTS[engine]
    client = make_client()
    root = tmp_path / 'models'
    client.app.state.settings.models_root_dir = root
    try:
        response = client.post('/api/admin/models/download', json={'engine': engine})
        assert response.status_code == 200, response.text
        assert response.json()['job']['repo_id'] == variant.repo_id
        assert response.json()['job']['model_id'] == variant.base_alias
        _wait_for(calls)
        assert calls == [(variant.repo_id, {'local_dir': str(root / variant.folder), 'revision': variant.revision})]
        assert len(variant.revision) == 40
    finally:
        _close(client)


def test_a_model_id_in_the_request_cannot_move_the_repo_of_a_download(monkeypatch, tmp_path) -> None:
    calls = _fake_hub(monkeypatch, lambda: None)
    client = make_client()
    client.app.state.settings.models_root_dir = tmp_path
    try:
        response = client.post(
            '/api/admin/models/download', json={'engine': 'audio8_mini', 'model_id': 'attacker/evil-model'},
        )
        assert response.status_code == 200, response.text
        _wait_for(calls)
        variant = AUDIO8_VARIANTS['audio8_mini']
        assert [call[0] for call in calls] == [variant.repo_id]
        assert calls[0][1]['revision'] == variant.revision
    finally:
        _close(client)


def test_the_download_list_stays_closed(monkeypatch) -> None:
    calls = _fake_hub(monkeypatch, lambda: None)
    client = make_client()
    try:
        assert client.post('/api/admin/models/download', json={'engine': 'audio8_nope'}).status_code == 400
        phonon = client.post('/api/admin/models/download', json={'engine': 'phonon'})
        assert phonon.status_code == 400 and 'by hand' in phonon.json()['detail']
        assert calls == []
    finally:
        _close(client)


def test_a_running_download_shows_on_the_engine_list(monkeypatch, tmp_path) -> None:
    import threading

    release = threading.Event()
    calls = _fake_hub(monkeypatch, release.wait)
    monkeypatch.setattr(engines, '_module_installed', lambda name: True)
    client = make_client()
    client.app.state.settings.models_root_dir = tmp_path
    try:
        assert client.post('/api/admin/models/download', json={'engine': 'audio8_onnx'}).status_code == 200
        _wait_for(calls)
        info = _engine(client, 'audio8_onnx')
        assert (info['status'], info['detail'], info['available']) == ('downloading', 'Download laeuft.', False)
        assert _engine(client, 'audio8_mini')['status'] == 'missing_model', 'another variant is not affected'
    finally:
        release.set()
        _close(client)


def test_the_active_engine_lists_its_model_with_size(monkeypatch, models_root) -> None:
    client = _client(models_root)
    try:
        _switch(client, 'audio8_mini')
        rows = client.get('/api/admin/models').json()
        rows = rows['models'] if isinstance(rows, dict) else rows
        statuses = [row for row in rows if 'approx_size_gb' in row]
        assert statuses and statuses[0]['status'] == 'ready'
        assert statuses[0]['approx_size_gb'] == AUDIO8_VARIANTS['audio8_mini'].approx_size_gb
    finally:
        _close(client)


def test_the_server_does_not_delete_an_audio8_model(models_root) -> None:
    client = _client(models_root)
    try:
        _switch(client, 'audio8_realistic')
        response = client.post('/api/admin/models/delete', json={})
        assert response.status_code == 400 and 'by hand' in response.json()['detail']
    finally:
        _close(client)


# --- presets ---------------------------------------------------------------------------------------------------------


AUDIO8_TIER_NAMES = [name for name in presets.BUILTIN_PRESETS if name.startswith('Audio8')]


def test_every_gpu_engine_has_the_three_tiers_and_the_cpu_engine_its_own() -> None:
    assert len(AUDIO8_TIER_NAMES) == 10
    for prefix, engine in (('Audio8', 'audio8_realistic'), ('Audio8 Preview', 'audio8_preview'), ('Audio8 Mini', 'audio8_mini')):
        tiers = {name: presets.BUILTIN_PRESETS[f'{prefix} {name}'] for name in ('32GB VRAM', '24GB VRAM', '24+8 VRAM')}
        assert all(tier['values']['runtime_backend'] == engine for tier in tiers.values())
        assert [tier['values']['audio8_max_slots'] for tier in tiers.values()] == [32, 16, 16]
        assert [tier['target_vram_gb'] for tier in tiers.values()] == [None, None, 24]
    assert presets.BUILTIN_PRESETS['Audio8 ONNX CPU']['values']['runtime_backend'] == 'audio8_onnx'


@pytest.mark.parametrize('name', AUDIO8_TIER_NAMES)
def test_a_tier_survives_the_validation_of_the_settings_put(name) -> None:
    values = presets.BUILTIN_PRESETS[name]['values']
    assert presets.preset_values(values) == values, 'every key is a preset key, every value valid'
    assert values['max_parallel_requests'] == values.get('audio8_max_slots', values.get('audio8_onnx_workers'))
    assert presets.BUILTIN_PRESETS[name]['description']
    assert presets.preset_engine(values) == values['runtime_backend']


def test_the_default_tiers_exist_and_leave_the_output_rate_alone() -> None:
    for engine in ENGINE_IDS:
        values = engines.default_tier_values(engine)
        assert values and 'runtime_backend' not in values and 'audio8_output_sample_rate' not in values


def test_the_admin_lists_a_tier_under_its_engine(tmp_path) -> None:
    listed = {item['name']: item['engine'] for item in presets.list_presets(tmp_path)}
    assert listed['Audio8 32GB VRAM'] == 'audio8_realistic'
    assert listed['Audio8 Preview 24+8 VRAM'] == 'audio8_preview'
    assert listed['Audio8 Mini 24GB VRAM'] == 'audio8_mini'
    assert listed['Audio8 ONNX CPU'] == 'audio8_onnx'


def test_a_saved_preset_without_engine_key_lands_on_an_audio8_engine() -> None:
    assert presets.preset_engine({'audio8_onnx_workers': 2}) == 'audio8_onnx'
    assert presets.preset_engine({'audio8_max_slots': 4}) == 'audio8_realistic'
    assert presets.preset_engine({'ming_max_slots': 4}) == 'ming'


def test_a_tier_loaded_as_a_preset_does_not_switch_the_engine(models_root) -> None:
    client = _client(models_root)
    try:
        values = dict(presets.BUILTIN_PRESETS['Audio8 32GB VRAM']['values'])
        values.pop('runtime_backend')
        assert client.put('/api/admin/settings', json=values).status_code == 200
        assert client.app.state.settings.runtime_backend == 'mock'
        assert client.app.state.settings.audio8_max_slots == 32
    finally:
        _close(client)


# --- request handling ----------------------------------------------------------------------------------------------


def test_inline_tags_are_never_read_aloud() -> None:
    for engine in ENGINE_IDS:
        assert render_inline_tags('[laughter] Hallo [Cough] Welt.', engine).split() == ['Hallo', 'Welt.']


def test_a_held_job_with_ming_controls_keeps_its_instruction_on_audio8() -> None:
    request = SpeechRequest(input='Hallo.', instructions='froehlich', emotion='happy')
    job = types.SimpleNamespace(request=request, job_id='j')
    engines._resolve_delivery_conflict(job, 'audio8_realistic')
    assert job.request is request, 'only Ming refuses that combination, Audio8 reads neither'
