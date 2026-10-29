"""Per-engine best seeds on voice profiles: storage, the active engine picking the seed, the admin PATCH, preset
backfill and WER runs bound to one engine."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import pytest

from omnivoice_tts_server import voice_presets
from omnivoice_tts_server.config import Settings
from omnivoice_tts_server.domain.models import SpeechRequest, WerBenchmarkCreateRequest, WerBenchmarkItemResult
from omnivoice_tts_server.domain.state import InMemoryStore, VoiceProfileRecord, clean_engine_seeds, seed_engine_key
from omnivoice_tts_server.services_v2 import EventHub, QueueService, WerBenchmarkService
from omnivoice_tts_server.voice_presets import ENGINE_SEEDS_MARKER_FILE, seed_voice_presets

from test_wer_admin_http import _add_voice, make_client


def _voice(seed: int | None = None, **engine_seeds: int) -> VoiceProfileRecord:
    return VoiceProfileRecord(
        voice_id='voice_sonja',
        name='SonjaV2',
        source='custom',
        created_at=datetime(2026, 10, 1, tzinfo=timezone.utc),
        audio_bytes=b'RIFF',
        ref_text='Hallo, ich bin Sonja.',
        consent=True,
        seed=seed,
        engine_seeds=dict(engine_seeds),
    )


# --- the record --------------------------------------------------------------------------------------------------


def test_seed_for_engine_reads_seed_for_omnivoice_and_engine_seeds_otherwise() -> None:
    profile = _voice(9305, ming=77)
    assert profile.seed_for_engine('omnivoice') == 9305
    assert profile.seed_for_engine('mock') == 9305, 'mock renders like OmniVoice'
    assert profile.seed_for_engine('MING') == 77
    assert profile.seed_for_engine('phonon') is None
    assert profile.seed_for_engine('tortoise') is None
    assert profile.all_engine_seeds() == {'omnivoice': 9305, 'ming': 77}


def test_set_seed_for_engine_keeps_the_other_engines() -> None:
    profile = _voice(9305)
    profile.set_seed_for_engine('ming', 12)
    profile.set_seed_for_engine('phonon', 0)
    assert (profile.seed, profile.engine_seeds) == (9305, {'ming': 12, 'phonon': 0})
    profile.set_seed_for_engine('ming', None)
    profile.set_seed_for_engine('omnivoice', None)
    assert (profile.seed, profile.engine_seeds) == (None, {'phonon': 0})
    with pytest.raises(ValueError):
        profile.set_seed_for_engine('tortoise', 1)


def test_engine_key_and_file_cleanup() -> None:
    assert [seed_engine_key(name) for name in ('omnivoice', 'mock', ' Ming ', 'phonon', '', None, 'x')] == [
        'omnivoice', 'omnivoice', 'ming', 'phonon', None, None, None,
    ]
    # OmniVoice lives in seed; unknown engines, bools, out-of-range and non-int values are dropped.
    assert clean_engine_seeds({'ming': 5, 'omnivoice': 1, 'x': 2, 'phonon': True}) == {'ming': 5}
    assert clean_engine_seeds({'ming': -1, 'phonon': 2_147_483_648}) == {}
    assert clean_engine_seeds({'ming': '5'}) == {}
    assert clean_engine_seeds(None) == {}


def test_engine_seeds_round_trip_and_old_files_still_load(tmp_path) -> None:
    store = InMemoryStore(max_queue_size=4)
    store.voice_profiles['voice_sonja'] = _voice(9305, ming=77, phonon=3)
    store.save_voices(tmp_path)
    reloaded = InMemoryStore(max_queue_size=4)
    reloaded.load_voices(tmp_path)
    profile = reloaded.voice_profiles['voice_sonja']
    assert (profile.seed, profile.engine_seeds) == (9305, {'ming': 77, 'phonon': 3})

    # A file written before per-engine seeds has no engine_seeds key.
    path = tmp_path / 'voice_profiles.json'
    payload = json.loads(path.read_text(encoding='utf-8'))
    del payload['voice_sonja']['engine_seeds']
    path.write_text(json.dumps(payload), encoding='utf-8')
    old = InMemoryStore(max_queue_size=4)
    old.load_voices(tmp_path)
    assert (old.voice_profiles['voice_sonja'].seed, old.voice_profiles['voice_sonja'].engine_seeds) == (9305, {})


# --- the active engine picks the seed -----------------------------------------------------------------------------


def _queue(tmp_path, profile: VoiceProfileRecord, backend: str) -> tuple[QueueService, Settings]:
    settings = Settings(data_dir=tmp_path, sentence_chunking=False, runtime_backend=backend)
    store = InMemoryStore(max_queue_size=settings.max_queue_size)
    store.voice_profiles[profile.voice_id] = profile
    return QueueService(store, object(), EventHub(store), settings), settings


@pytest.mark.parametrize(('backend', 'expected'), [('omnivoice', 9305), ('mock', 9305), ('ming', 77), ('phonon', None)])
def test_request_without_seed_gets_the_active_engines_seed(tmp_path, backend, expected) -> None:
    queue, _ = _queue(tmp_path, _voice(9305, ming=77), backend)
    job = asyncio.run(queue.submit(SpeechRequest(input='Guten Tag.')))
    assert job.request.seed == expected


def test_voice_seed_follows_a_live_engine_switch(tmp_path) -> None:
    queue, settings = _queue(tmp_path, _voice(9305, ming=77), 'omnivoice')
    assert asyncio.run(queue.submit(SpeechRequest(input='Erster Satz.'))).request.seed == 9305
    settings.runtime_backend = 'ming'  # what EngineSwitcher._run does to the shared settings
    assert asyncio.run(queue.submit(SpeechRequest(input='Zweiter Satz.'))).request.seed == 77
    assert asyncio.run(queue.submit(SpeechRequest(input='Dritter Satz.', seed=5))).request.seed == 5


class _FakeQueue:
    async def cancel(self, job_id: str):
        return None


def _wer(tmp_path, backend: str = 'mock') -> tuple[WerBenchmarkService, InMemoryStore, Settings]:
    store = InMemoryStore(max_queue_size=8)
    settings = Settings(data_dir=tmp_path, default_voice='voice_sonja', runtime_backend=backend)
    return WerBenchmarkService(store, _FakeQueue(), object(), EventHub(store), settings), store, settings


def test_best_seed_for_voice_reads_the_active_engine(tmp_path) -> None:
    svc, store, settings = _wer(tmp_path)
    store.voice_profiles['voice_sonja'] = _voice(9305, ming=77)
    assert svc.best_seed_for_voice('SonjaV2') == 9305
    settings.runtime_backend = 'ming'
    assert svc.best_seed_for_voice('SonjaV2') == 77
    settings.runtime_backend = 'phonon'
    assert svc.best_seed_for_voice('SonjaV2') is None


# --- admin PATCH -------------------------------------------------------------------------------------------------


def test_patch_sets_and_clears_one_engines_seed() -> None:
    client = make_client()
    _add_voice(client, 'voice_x', 'SonjaV3', seed=9301)

    saved = client.patch('/api/admin/voices/voice_x', json={'seed': 41, 'engine': 'ming'})
    assert saved.status_code == 200, saved.text
    assert saved.json()['seed'] == 9301, 'the OmniVoice seed is untouched'
    assert saved.json()['engine_seeds'] == {'omnivoice': 9301, 'ming': 41}

    listed = {item['voice_id']: item for item in client.get('/api/admin/voices').json()}
    assert listed['voice_x']['engine_seeds'] == {'omnivoice': 9301, 'ming': 41}
    public = {item['voice_id']: item for item in client.get('/api/v1/voices').json()['voices']}
    assert public['voice_x']['engine_seeds'] == {} and public['voice_x']['seed'] is None, 'details stay admin-only'

    # No engine is the OmniVoice seed (older clients); null clears only the named engine.
    assert client.patch('/api/admin/voices/voice_x', json={'seed': 7}).json()['engine_seeds'] == {'omnivoice': 7, 'ming': 41}
    cleared = client.patch('/api/admin/voices/voice_x', json={'seed': None, 'engine': 'ming'}).json()
    assert (cleared['seed'], cleared['engine_seeds']) == (7, {'omnivoice': 7})

    profile = client.app.state.store.voice_profiles['voice_x']
    assert (profile.seed, profile.engine_seeds) == (7, {})


def test_patch_rejects_an_unknown_engine_and_changes_nothing() -> None:
    client = make_client()
    _add_voice(client, 'voice_x', 'SonjaV3', seed=9301)
    response = client.patch('/api/admin/voices/voice_x', json={'seed': 1, 'engine': 'tortoise', 'name': 'Neu'})
    assert response.status_code == 400
    assert 'Unknown engine' in response.json()['detail']
    profile = client.app.state.store.voice_profiles['voice_x']
    assert (profile.name, profile.seed, profile.engine_seeds) == ('SonjaV3', 9301, {})
    assert client.patch('/api/admin/voices/voice_x', json={'seed': -1, 'engine': 'ming'}).status_code == 422


def test_patched_engine_seed_is_persisted() -> None:
    client = make_client()
    _add_voice(client, 'voice_x', 'SonjaV3')
    assert client.patch('/api/admin/voices/voice_x', json={'seed': 12, 'engine': 'phonon'}).status_code == 200
    reloaded = InMemoryStore(max_queue_size=4)
    reloaded.load_voices(client.app.state.settings.data_dir)
    assert reloaded.voice_profiles['voice_x'].engine_seeds == {'phonon': 12}


# --- presets -----------------------------------------------------------------------------------------------------


def _patch_presets(tmp_path, monkeypatch, **engine_seeds_by_voice: dict) -> None:
    presets = json.loads(voice_presets.PRESETS_FILE.read_text(encoding='utf-8'))
    for preset in presets:
        if preset['voice_id'] in engine_seeds_by_voice:
            preset['engine_seeds'] = engine_seeds_by_voice[preset['voice_id']]
    patched = tmp_path / 'presets.json'
    patched.write_text(json.dumps(presets), encoding='utf-8')
    monkeypatch.setattr(voice_presets, 'PRESETS_FILE', patched)


def _restart(data_dir) -> InMemoryStore:
    store = InMemoryStore(max_queue_size=4)
    store.load_voices(data_dir)
    seed_voice_presets(store, data_dir)
    return store


def test_preset_engine_seeds_reach_an_existing_install_once(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / 'data'
    # An install seeded before the presets carried engine seeds.
    store = InMemoryStore(max_queue_size=4)
    seed_voice_presets(store, data_dir)
    assert store.voice_profiles['voice_preset_sonjav2'].engine_seeds == {}

    _patch_presets(
        tmp_path, monkeypatch,
        voice_preset_sonjav2={'ming': 11, 'omnivoice': 1},
        voice_preset_sonjav3={'ming': 22, 'phonon': 33},
    )
    # The operator already chose a Ming seed for SonjaV3.
    store.voice_profiles['voice_preset_sonjav3'].engine_seeds['ming'] = 99
    store.save_voices(data_dir)

    store = _restart(data_dir)
    v2, v3 = store.voice_profiles['voice_preset_sonjav2'], store.voice_profiles['voice_preset_sonjav3']
    assert (v2.seed, v2.engine_seeds) == (9305, {'ming': 11}), 'seed stays; an omnivoice entry is ignored'
    assert (v3.seed, v3.engine_seeds) == (9301, {'ming': 99, 'phonon': 33}), 'the operator value wins'

    # A seed the operator cleared stays cleared on the next start.
    v2.engine_seeds.pop('ming')
    store.save_voices(data_dir)
    store = _restart(data_dir)
    assert store.voice_profiles['voice_preset_sonjav2'].engine_seeds == {}

    # A new value shipped later does not refill it either: once handled, the slot is the operator's.
    _patch_presets(tmp_path, monkeypatch, voice_preset_sonjav2={'ming': 12})
    store = _restart(data_dir)
    assert store.voice_profiles['voice_preset_sonjav2'].engine_seeds == {}
    marker = json.loads((data_dir / ENGINE_SEEDS_MARKER_FILE).read_text(encoding='utf-8'))
    assert marker['voice_preset_sonjav2'] == {'ming': 11}

    # An engine the presets did not ship before is still filled.
    _patch_presets(tmp_path, monkeypatch, voice_preset_sonjav2={'ming': 12, 'phonon': 4})
    store = _restart(data_dir)
    assert store.voice_profiles['voice_preset_sonjav2'].engine_seeds == {'phonon': 4}


def test_fresh_install_gets_preset_engine_seeds(tmp_path, monkeypatch) -> None:
    _patch_presets(tmp_path, monkeypatch, voice_preset_sonjav3={'ming': 5})
    data_dir = tmp_path / 'data'
    store = InMemoryStore(max_queue_size=4)
    assert 'voice_preset_sonjav3' in seed_voice_presets(store, data_dir)
    assert store.voice_profiles['voice_preset_sonjav3'].engine_seeds == {'ming': 5}
    reloaded = InMemoryStore(max_queue_size=4)
    reloaded.load_voices(data_dir)
    assert reloaded.voice_profiles['voice_preset_sonjav3'].engine_seeds == {'ming': 5}


def test_shipped_presets_load_without_engine_seeds(tmp_path) -> None:
    store = InMemoryStore(max_queue_size=4)
    seed_voice_presets(store, tmp_path)
    assert store.voice_profiles['voice_preset_sonjav2'].all_engine_seeds() == {'omnivoice': 9305}
    assert store.voice_profiles['voice_preset_sonjav3'].all_engine_seeds() == {'omnivoice': 9301}


# --- WER runs are bound to one engine ------------------------------------------------------------------------------


def _fake_wer_steps(svc, monkeypatch, *, on_synth=None) -> list[int | None]:
    seen: list[int | None] = []

    async def fake_sentences(payload):
        return ['Ein Satz.', 'Zweiter Satz.'], {'hit': False, 'key': None}

    async def fake_synth(index, sentence, payload, *, seed):
        seen.append(seed)
        if on_synth is not None:
            on_synth(len(seen))
        return {'success': True, 'index': index, 'seed': seed, 'sentence': sentence, 'audio': b'', 'content_type': 'audio/wav', 'metrics': {}, 'synthesis_ms': 1, 'started_at': 0.0}

    async def fake_transcribe(item, payload):
        return WerBenchmarkItemResult(index=item['index'], seed=item['seed'], success=True, source_text=item['sentence'], transcript=item['sentence'], wer=0.0)

    monkeypatch.setattr(svc, '_generate_sentences', fake_sentences)
    monkeypatch.setattr(svc, '_synthesize_wer_item', fake_synth)
    monkeypatch.setattr(svc, '_transcribe_wer_item', fake_transcribe)
    return seen


def _finish(svc, store, payload) -> dict:
    async def run() -> str:
        response = await svc.create_run(payload)
        for _ in range(300):
            if store.wer_benchmark_runs[response.run_id]['status'] != 'running':
                break
            await asyncio.sleep(0.01)
        return response.run_id

    run_id = asyncio.run(run())
    return svc._to_response(store.wer_benchmark_runs[run_id]).model_dump()


def test_wer_run_records_its_engine_and_uses_that_engines_seed(tmp_path, monkeypatch) -> None:
    svc, store, _ = _wer(tmp_path, backend='ming')
    store.voice_profiles['voice_sonja'] = _voice(9305, ming=77)
    seen = _fake_wer_steps(svc, monkeypatch)
    response = _finish(svc, store, WerBenchmarkCreateRequest(count=2, concurrency=2, exclusive=False, request=SpeechRequest(input='x', voice='voice_sonja')))
    assert response['status'] == 'completed', response['error_message']
    assert response['engine'] == 'ming'
    assert seen == [77, 77]
    assert [row['seed'] for row in response['seed_leaderboard']] == [77]


def test_engine_switch_during_a_run_stops_it_without_mixing_results(tmp_path, monkeypatch) -> None:
    svc, store, settings = _wer(tmp_path, backend='omnivoice')

    def switch_after_first_wave(rendered: int) -> None:
        if rendered == 3:  # the first item of the second seed's wave
            settings.runtime_backend = 'ming'

    _fake_wer_steps(svc, monkeypatch, on_synth=switch_after_first_wave)
    response = _finish(
        svc, store,
        WerBenchmarkCreateRequest(count=2, concurrency=2, exclusive=False, seed_values=[1, 2, 3], request=SpeechRequest(input='x')),
    )
    assert response['status'] == 'failed'
    assert response['engine'] == 'omnivoice'
    assert 'switched from omnivoice to ming' in response['error_message']
    # Seed 1 finished on OmniVoice; the wave that saw the switch is dropped, seed 3 never starts.
    assert [row['seed'] for row in response['seed_leaderboard']] == [1]
    assert response['results_count'] == 2


def test_engine_switch_in_the_last_asr_wave_keeps_a_finished_run(tmp_path, monkeypatch) -> None:
    """Whisper does not depend on the TTS engine: a switch after the last TTS wave does not fail the run."""
    svc, store, settings = _wer(tmp_path, backend='omnivoice')
    _fake_wer_steps(svc, monkeypatch)

    async def transcribe_then_switch(item, payload):
        settings.runtime_backend = 'ming'
        return WerBenchmarkItemResult(index=item['index'], seed=item['seed'], success=True, source_text=item['sentence'], transcript=item['sentence'], wer=0.0)

    monkeypatch.setattr(svc, '_transcribe_wer_item', transcribe_then_switch)
    response = _finish(
        svc, store,
        WerBenchmarkCreateRequest(count=2, concurrency=2, exclusive=False, seed_values=[1], request=SpeechRequest(input='x')),
    )
    assert response['status'] == 'completed', response['error_message']
    assert (response['engine'], response['results_count']) == ('omnivoice', 2)


def test_wer_run_keeps_the_voice_it_resolved_at_the_start(tmp_path, monkeypatch) -> None:
    svc, store, settings = _wer(tmp_path, backend='ming')
    store.voice_profiles['voice_sonja'] = _voice(9305, ming=77)
    store.voice_profiles['voice_other'] = VoiceProfileRecord(
        voice_id='voice_other', name='SonjaV3', source='custom', created_at=datetime(2026, 10, 1, tzinfo=timezone.utc),
        audio_bytes=b'RIFF', ref_text='Hallo.', consent=True, seed=9301, engine_seeds={'ming': 5},
    )
    seen = _fake_wer_steps(svc, monkeypatch, on_synth=lambda rendered: setattr(settings, 'default_voice', 'voice_other'))
    # No voice of its own: the run renders the default voice of its start, also for the seed of a later seed pass.
    response = _finish(
        svc, store,
        WerBenchmarkCreateRequest(count=2, concurrency=1, exclusive=False, request=SpeechRequest(input='x')),
    )
    assert response['status'] == 'completed', response['error_message']
    assert (response['voice'], response['resolved_voice']) == (None, 'voice_sonja')
    assert seen == [77, 77]

    named = _finish(svc, store, WerBenchmarkCreateRequest(count=2, exclusive=False, request=SpeechRequest(input='x', voice='sonjav3')))
    assert named['resolved_voice'] == 'voice_other'
    settings.default_voice = 'Auto Voice'
    assert _finish(svc, store, WerBenchmarkCreateRequest(count=2, exclusive=False, request=SpeechRequest(input='x')))['resolved_voice'] is None


def test_requeue_resolves_a_profile_seed_for_the_new_engine(tmp_path) -> None:
    """A job held during an engine switch keeps no seed of the old engine (EngineSwitcher._requeue)."""
    queue, settings = _queue(tmp_path, _voice(9305, ming=77), 'omnivoice')
    from_profile = asyncio.run(queue.submit(SpeechRequest(input='Erster Satz.')))
    from_client = asyncio.run(queue.submit(SpeechRequest(input='Zweiter Satz.', seed=5)))
    assert (from_profile.request.seed, from_profile.seed_from_profile) == (9305, True)
    assert (from_client.request.seed, from_client.seed_from_profile) == (5, False)

    settings.runtime_backend = 'ming'
    for job in (from_profile, from_client):
        queue.reapply_voice_seed(job)
    assert from_profile.request.seed == 77
    assert from_client.request.seed == 5, 'a client seed is never replaced'

    # Phonon has no seed for the voice: the profile seed is dropped, not carried over; back on OmniVoice it returns.
    settings.runtime_backend = 'phonon'
    queue.reapply_voice_seed(from_profile)
    assert from_profile.request.seed is None
    settings.runtime_backend = 'omnivoice'
    queue.reapply_voice_seed(from_profile)
    assert from_profile.request.seed == 9305
