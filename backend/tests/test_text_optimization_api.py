"""text_optimization request flag, admin default and endpoints."""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from omnivoice_tts_server.app import create_app
from omnivoice_tts_server.config import Settings, load_runtime_settings
from pydantic import ValidationError

from omnivoice_tts_server.domain.models import SpeechRequest
from omnivoice_tts_server.domain.state import InMemoryStore, VoiceProfileRecord
from omnivoice_tts_server.prompt_batch import split_sentences
from omnivoice_tts_server.services_v2 import EventHub, QueueService

GERMAN_TEXT = 'Am 24.09.2026 um 14:30 Uhr kostet es 29,90 €. Tel. 0341 1234567, ca. 5 km entfernt.'
SPOKEN_GERMAN = (
    'Am vierundzwanzigsten September zweitausendsechsundzwanzig um vierzehn Uhr dreißig kostet es '
    'neunundzwanzig Euro und neunzig Cent. Telefon null drei vier eins eins zwei drei vier fünf sechs sieben, '
    'circa fünf Kilometer entfernt.'
)


def sonja_profile() -> VoiceProfileRecord:
    return VoiceProfileRecord(
        voice_id='voice_sonja',
        name='Sonja',
        source='upload',
        created_at=datetime(2026, 10, 1, tzinfo=timezone.utc),
        audio_bytes=b'RIFF',
        ref_text='Hallo, ich bin Sonja. Heute ist der 3.10.',
        consent=True,
    )


def make_queue(tmp_path, **overrides) -> tuple[QueueService, InMemoryStore]:
    settings = Settings(data_dir=tmp_path, **overrides)
    store = InMemoryStore(max_queue_size=settings.max_queue_size)
    profile = sonja_profile()
    store.voice_profiles[profile.voice_id] = profile
    return QueueService(store, object(), EventHub(store), settings), store


def submit(queue: QueueService, request: SpeechRequest, owner_scope: str = 'public'):
    return asyncio.run(queue.submit(request, owner_scope=owner_scope))


def expected_sentences(settings: Settings, text: str) -> list[str]:
    from omnivoice_tts_server.capacity import max_chars_per_chunk

    return split_sentences(
        text,
        enabled=settings.sentence_chunking,
        short_sentence_merge_max_chars=settings.short_sentence_merge_max_chars,
        following_sentence_merge_min_chars=settings.following_sentence_merge_min_chars,
        max_chars=max_chars_per_chunk(settings.vram_budget_mb),
    )


# --- QueueService --------------------------------------------------------------------------------

def test_default_is_off_and_sentences_are_byte_identical(tmp_path) -> None:
    queue, store = make_queue(tmp_path)
    assert queue.settings.text_optimization_default == 'off'
    job = submit(queue, SpeechRequest(input=GERMAN_TEXT, language='Deutsch'))
    assert store.request_states[job.job_id].sentences == expected_sentences(queue.settings, GERMAN_TEXT)
    assert job.text_optimization is None and job.spoken_text is None


def test_explicit_flag_optimizes_the_spoken_sentences_only(tmp_path) -> None:
    queue, store = make_queue(tmp_path)
    request = SpeechRequest(input=GERMAN_TEXT, language='Deutsch', text_optimization='deterministic')
    job = submit(queue, request)
    assert ' '.join(store.request_states[job.job_id].sentences) == SPOKEN_GERMAN
    assert job.spoken_text == SPOKEN_GERMAN
    assert job.request.input == GERMAN_TEXT
    assert job.text_optimization['mode'] == 'deterministic'
    assert job.text_optimization['language'] == 'de'
    assert job.text_optimization['binding_origin'] == 'request'
    assert {'date_de_dmy', 'time_de_hhmm', 'currency_suffix', 'abbr_tel', 'abbr_ca'} <= set(job.text_optimization['applied_rules'])
    # The clone prompt is never rewritten.
    assert store.voice_profiles['voice_sonja'].ref_text == 'Hallo, ich bin Sonja. Heute ist der 3.10.'
    assert job.request.ref_text is None


def test_upstream_normalize_text_is_dropped_once_the_optimizer_bound_de_or_en(tmp_path) -> None:
    queue, store = make_queue(tmp_path)
    metadata = {'normalize_text': True, 'pad_duration': 0.0}
    optimized = submit(
        queue,
        SpeechRequest(input=GERMAN_TEXT, language='Deutsch', text_optimization='deterministic', metadata=metadata),
    )
    assert optimized.request.metadata == {'pad_duration': 0.0}
    assert metadata == {'normalize_text': True, 'pad_duration': 0.0}  # caller's dict untouched
    raw = submit(queue, SpeechRequest(input=GERMAN_TEXT, language='Deutsch', metadata=dict(metadata)))
    assert raw.request.metadata == metadata
    assert store.request_states[optimized.job_id].group_key != store.request_states[raw.job_id].group_key
    # A language the optimizer leaves alone keeps upstream normalisation.
    french = submit(
        queue,
        SpeechRequest(input='Il est 14 h.', language='fr', text_optimization='deterministic', metadata=dict(metadata)),
    )
    assert french.request.metadata == metadata
    # Bound to en but no rule changed the text: upstream normalisation stays on.
    untouched = submit(
        queue,
        SpeechRequest(input='Hello there, friend.', language='en', text_optimization=True, metadata=dict(metadata)),
    )
    assert untouched.text_optimization['language'] == 'en' and untouched.text_optimization['applied_rules'] == []
    assert untouched.request.metadata == metadata


def test_ref_text_and_instructions_are_never_touched(tmp_path) -> None:
    queue, _ = make_queue(tmp_path)
    request = SpeechRequest(
        input='Um 9 Uhr.',
        ref_audio='data:audio/wav;base64,UklGRg==',
        ref_text='Referenz am 3.10. um 9 Uhr.',
        instructions='female, low pitch',
        language='de',
        text_optimization=True,
    )
    job = submit(queue, request)
    assert job.spoken_text == 'Um neun Uhr.'
    assert job.request.ref_text == 'Referenz am 3.10. um 9 Uhr.'
    assert job.request.instructions == 'female, low pitch'


def test_admin_default_applies_to_public_requests_only(tmp_path) -> None:
    queue, store = make_queue(tmp_path, text_optimization_default='deterministic')
    public = submit(queue, SpeechRequest(input=GERMAN_TEXT, language='de'))
    assert public.spoken_text == SPOKEN_GERMAN
    for scope in ('benchmark', 'wer-benchmark', 'finetune'):
        job = submit(queue, SpeechRequest(input=GERMAN_TEXT, language='de'), owner_scope=scope)
        assert job.text_optimization is None
        assert store.request_states[job.job_id].sentences == expected_sentences(queue.settings, GERMAN_TEXT)
    opted_out = submit(queue, SpeechRequest(input=GERMAN_TEXT, language='de', text_optimization='off'))
    assert opted_out.text_optimization is None


def test_metadata_fallback_for_openai_style_clients(tmp_path) -> None:
    queue, _ = make_queue(tmp_path)
    job = submit(queue, SpeechRequest(input='Um 9 Uhr.', language='de', metadata={'text_optimization': 'deterministic'}))
    assert job.spoken_text == 'Um neun Uhr.'
    # Validated on the model like the top-level field (HTTP 422, see the HTTP tests).
    with pytest.raises(ValidationError, match='reserved'):
        SpeechRequest(input='Um 9 Uhr.', metadata={'text_optimization': 'llm'})
    with pytest.raises(ValidationError):
        SpeechRequest(input='Um 9 Uhr.', metadata={'text_optimization': 'banana'})


def test_auto_language_binds_from_notation_or_stays_raw(tmp_path) -> None:
    queue, _ = make_queue(tmp_path)
    english = submit(queue, SpeechRequest(input='The call is on March 3, 2026 at 4:15 PM.', language='Auto', text_optimization=True))
    assert english.text_optimization['language'] == 'en'
    # The post-processor silences the hyphen between letters ("twenty-six" -> "twenty six").
    assert english.spoken_text == 'The call is on March third two thousand twenty six at four fifteen P M.'
    # No notation, but German function words: bound from the words.
    words = submit(queue, SpeechRequest(input='Ich habe 3 Katzen.', text_optimization=True))
    assert words.text_optimization['language'] == 'de'
    assert words.text_optimization['binding_origin'] == 'text_words'
    assert words.spoken_text == 'Ich habe drei Katzen.'
    plain = submit(queue, SpeechRequest(input='Kaffee 3 Mal.', text_optimization=True))
    assert plain.text_optimization['language'] is None
    assert plain.spoken_text == 'Kaffee 3 Mal.'
    french = submit(queue, SpeechRequest(input='Le 3 mars à 14h30.', language='fr', text_optimization=True))
    assert french.spoken_text == 'Le 3 mars à 14h30.'
    assert french.text_optimization['binding_origin'] == 'unsupported_language'


@pytest.mark.parametrize(
    'text',
    [
        'Op 24.09.2026 kost het 1.234,56 euro.',
        'Spotkanie odbędzie się 24.09.2026 o godzinie 14:30.',
        'La riunione è il 24.09.2026 e costa 29,90 €.',
        'La réunion aura lieu le 24.09.2026 et coûte 29,90 €.',
        'Toplantı 24.09.2026 tarihinde ve fiyatı 29,90 €.',
        'Встреча состоится 24.09.2026, цена 1.234,56 руб.',
    ],
)
@pytest.mark.parametrize('language', [None, 'Auto'])
def test_auto_language_leaves_other_languages_with_german_notation_alone(tmp_path, text, language) -> None:
    queue, _ = make_queue(tmp_path, text_optimization_default='deterministic')
    job = submit(queue, SpeechRequest(input=text, language=language))
    assert job.spoken_text == text
    assert job.text_optimization['language'] is None
    assert job.text_optimization['binding_origin'] == 'other_language'


def test_max_input_chars_is_checked_on_the_raw_input(tmp_path) -> None:
    from omnivoice_tts_server.services_v2 import RequestTooLargeError

    queue, _ = make_queue(tmp_path, max_input_chars=40)
    # 39 raw characters pass even though the spoken text is much longer.
    ok = submit(queue, SpeechRequest(input='Am 24.09.2026 um 14:30 Uhr, 29,90 €.', language='de', text_optimization=True))
    assert len(ok.spoken_text) > 40
    with pytest.raises(RequestTooLargeError):
        submit(queue, SpeechRequest(input='x' * 41, text_optimization=True))


def test_text_that_optimizes_to_nothing_is_rejected(tmp_path) -> None:
    queue, _ = make_queue(tmp_path)
    with pytest.raises(RuntimeError, match='empty after text optimization'):
        submit(queue, SpeechRequest(input='```\ncode only\n```', language='de', text_optimization=True))


def test_flag_validation_on_the_model() -> None:
    assert SpeechRequest(input='x', text_optimization=True).text_optimization == 'deterministic'
    assert SpeechRequest(input='x', text_optimization=False).text_optimization == 'off'
    assert SpeechRequest(input='x').text_optimization is None
    with pytest.raises(ValueError):
        SpeechRequest(input='x', text_optimization='llm')


@pytest.mark.parametrize(
    'metadata',
    [
        {'pad_duration': float('inf')},
        {'pad_duration': float('nan')},
        {'pad_duration': 3600},
        {'pad_duration': -0.1},
        {'fade_duration': 1.5},
        {'fade_duration': 'abc'},
        {'pad_duration': True},
    ],
)
def test_pad_and_fade_metadata_are_bounded(metadata) -> None:
    with pytest.raises(ValidationError):
        SpeechRequest(input='x', metadata=metadata)


def test_pad_and_fade_metadata_in_range_pass() -> None:
    request = SpeechRequest(input='x', metadata={'pad_duration': 0, 'fade_duration': '0.05'})
    assert request.metadata == {'pad_duration': 0, 'fade_duration': '0.05'}
    assert SpeechRequest(input='x', metadata={'pad_duration': 2.0, 'fade_duration': 1.0}).metadata['pad_duration'] == 2.0


def test_runtime_settings_persist_the_default(tmp_path) -> None:
    (tmp_path / 'runtime_settings.json').write_text(json.dumps({'text_optimization_default': 'deterministic'}), encoding='utf-8')
    settings = Settings(data_dir=tmp_path)
    load_runtime_settings(settings)
    assert settings.text_optimization_default == 'deterministic'


# --- HTTP ------------------------------------------------------------------------------------------

def make_client(**overrides) -> TestClient:
    overrides.setdefault('active_model', 'k2-fsa/OmniVoice-AutoVoice')
    overrides.setdefault('default_voice', 'Auto Voice')
    temp_data_dir = tempfile.TemporaryDirectory()
    app = create_app(
        Settings(
            runtime_backend='mock',
            models_root_dir=Path(__file__).resolve().parents[2] / 'test-models',
            data_dir=Path(temp_data_dir.name),
            **overrides,
        )
    )
    client = TestClient(app)
    client.__enter__()
    client._temp_data_dir = temp_data_dir
    login = client.post('/api/admin/auth/login', json={'username': 'admin', 'password': 'admin'})
    assert login.status_code == 200, login.text
    changed = client.post(
        '/api/admin/auth/change-password', json={'current_password': 'admin', 'new_password': 'test-admin-password'}
    )
    assert changed.status_code == 200, changed.text
    return client


def recorded_texts(client: TestClient) -> list[str]:
    synthesizer = client.app.state.queue_service.synthesizer
    seen: list[str] = []
    original = synthesizer.render_batch

    async def render_batch(items):
        seen.extend(item.text for item in items)
        return await original(items)

    synthesizer.render_batch = render_batch
    return seen


def test_speech_endpoint_flag_header_and_rendered_text() -> None:
    client = make_client(sentence_chunking=False)
    seen = recorded_texts(client)
    plain = client.post('/v1/audio/speech', json={'input': 'Um 9 Uhr.', 'language': 'de'})
    assert plain.status_code == 200
    assert 'x-text-optimization' not in plain.headers
    optimized = client.post('/v1/audio/speech', json={'input': 'Um 9 Uhr.', 'language': 'de', 'text_optimization': 'deterministic'})
    assert optimized.status_code == 200
    assert optimized.headers['x-text-optimization'] == 'deterministic; language=de; rules=time_de_hour_only'
    assert seen == ['Um 9 Uhr.', 'Um neun Uhr.']
    rejected = client.post('/v1/audio/speech', json={'input': 'Um 9 Uhr.', 'text_optimization': 'llm'})
    assert rejected.status_code == 422


def test_metadata_flag_over_http_and_its_validation() -> None:
    client = make_client(sentence_chunking=False)
    seen = recorded_texts(client)
    optimized = client.post(
        '/v1/audio/speech', json={'input': 'Um 9 Uhr.', 'language': 'de', 'metadata': {'text_optimization': 'deterministic'}}
    )
    assert optimized.status_code == 200
    assert optimized.headers['x-text-optimization'] == 'deterministic; language=de; rules=time_de_hour_only'
    assert seen == ['Um neun Uhr.']
    for route in ('/v1/audio/speech', '/api/v1/synthesize', '/v1/jobs'):
        for value in ('llm', 'banana'):
            rejected = client.post(route, json={'input': 'Um 9 Uhr.', 'metadata': {'text_optimization': value}})
            assert rejected.status_code == 422, (route, value, rejected.text)


def test_pad_duration_infinity_is_rejected_over_http() -> None:
    client = make_client()
    body = '{"input": "Hallo", "metadata": {"pad_duration": Infinity}}'
    response = client.post('/v1/audio/speech', content=body, headers={'content-type': 'application/json'})
    assert response.status_code == 422
    huge = client.post('/api/v1/synthesize', json={'input': 'Hallo', 'metadata': {'pad_duration': 3600}})
    assert huge.status_code == 422
    ok = client.post('/api/v1/synthesize', json={'input': 'Hallo', 'metadata': {'pad_duration': 0.5}})
    assert ok.status_code == 200


def test_streaming_speech_sends_the_header() -> None:
    client = make_client()
    with client.stream(
        'POST',
        '/v1/audio/speech',
        json={'input': 'Um 9 Uhr.', 'language': 'de', 'stream': True, 'text_optimization': 'deterministic'},
    ) as response:
        assert response.status_code == 200
        assert response.headers['x-text-optimization'] == 'deterministic; language=de; rules=time_de_hour_only'
        assert b''.join(response.iter_bytes())
    with client.stream('POST', '/v1/audio/speech', json={'input': 'Um 9 Uhr.', 'stream': True}) as response:
        assert 'x-text-optimization' not in response.headers
        b''.join(response.iter_bytes())


def _wait_completed(client: TestClient, job_id: str, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if client.get(f'/api/admin/jobs/{job_id}').json()['status'] == 'completed':
            return
        time.sleep(0.02)
    raise AssertionError(f'job {job_id} did not complete')


def test_jobs_route_applies_the_flag_and_reports_it_on_the_audio() -> None:
    client = make_client(sentence_chunking=False)
    seen = recorded_texts(client)
    created = client.post('/v1/jobs', json={'input': 'Um 9 Uhr.', 'language': 'de', 'text_optimization': True})
    assert created.status_code == 200
    job_id = created.json()['job_id']
    _wait_completed(client, job_id)
    audio = client.get(f'/v1/jobs/{job_id}/audio')
    assert audio.status_code == 200
    assert audio.headers['x-text-optimization'] == 'deterministic; language=de; rules=time_de_hour_only'
    assert seen == ['Um neun Uhr.']
    plain = client.post('/v1/jobs', json={'input': 'Um 9 Uhr.'}).json()['job_id']
    _wait_completed(client, plain)
    assert 'x-text-optimization' not in client.get(f'/v1/jobs/{plain}/audio').headers


def test_cors_exposes_the_text_optimization_header() -> None:
    client = make_client()
    response = client.post(
        '/v1/audio/speech',
        json={'input': 'Um 9 Uhr.', 'language': 'de', 'text_optimization': True},
        headers={'Origin': 'https://example.org'},
    )
    assert response.status_code == 200
    assert 'x-text-optimization' in response.headers.get('access-control-expose-headers', '').lower()


def test_synthesize_and_job_detail_report_the_spoken_text() -> None:
    client = make_client()
    response = client.post('/api/v1/synthesize', json={'input': 'Um 9 Uhr.', 'language': 'de', 'text_optimization': True})
    assert response.status_code == 200
    payload = response.json()
    info = payload['text_optimization']
    assert info['spoken_text'] == 'Um neun Uhr.' and info['applied_rules'] == ['time_de_hour_only']
    detail = client.get(f"/api/admin/jobs/{payload['job_id']}").json()
    assert detail['text_optimization']['spoken_text'] == 'Um neun Uhr.'
    assert detail['input_preview'] == 'Um 9 Uhr.'
    plain = client.post('/api/v1/synthesize', json={'input': 'Um 9 Uhr.'}).json()
    assert plain['text_optimization'] is None


def test_ndjson_start_event_carries_the_optimization() -> None:
    client = make_client()
    with client.stream(
        'POST', '/api/v1/synthesize/stream', json={'input': 'Um 9 Uhr.', 'language': 'de', 'text_optimization': 'deterministic'}
    ) as response:
        assert response.status_code == 200
        events = [json.loads(line) for line in response.iter_lines() if line]
    start = events[0]
    assert start['type'] == 'start'
    assert start['text_optimization']['spoken_text'] == 'Um neun Uhr.'
    with client.stream('POST', '/api/v1/synthesize/stream', json={'input': 'Um 9 Uhr.'}) as response:
        plain_start = json.loads(next(line for line in response.iter_lines() if line))
    assert 'text_optimization' not in plain_start


def test_dry_run_endpoints() -> None:
    client = make_client()
    response = client.post('/api/v1/text/optimize', json={'input': GERMAN_TEXT, 'language': 'Deutsch'})
    assert response.status_code == 200
    payload = response.json()
    assert payload['text'] == SPOKEN_GERMAN
    assert payload['language'] == 'de' and payload['binding_origin'] == 'request'
    assert ' '.join(payload['sentences']) == SPOKEN_GERMAN
    floor = client.post('/api/v1/text/optimize', json={'input': 'Pokémon kam 1996 raus.', 'language': 'de', 'stage': 'floor'})
    assert floor.json()['text'] == 'Pokémon kam 1996 raus.'
    admin = client.post('/api/admin/text/optimize', json={'input': 'See you at 7pm.', 'language': 'en'})
    assert admin.status_code == 200 and admin.json()['text'] == 'See you at seven P M.'


def test_dry_run_rejects_what_synthesis_rejects_and_falls_back_on_errors(monkeypatch) -> None:
    client = make_client()
    empty = client.post('/api/v1/text/optimize', json={'input': '   '})
    assert empty.status_code == 400 and empty.json()['detail'] == 'Missing input text'
    code_only = client.post('/api/v1/text/optimize', json={'input': '```\ncode\n```', 'language': 'de'})
    assert code_only.status_code == 400
    assert code_only.json()['detail'] == 'Input text is empty after text optimization'
    synth = client.post('/api/v1/synthesize', json={'input': '```\ncode\n```', 'language': 'de', 'text_optimization': True})
    assert synth.status_code == 400

    def broken(*args, **kwargs):
        raise RuntimeError('rule bug')

    monkeypatch.setattr('omnivoice_tts_server.text_optimizer.service.optimize_text', broken)
    fallback = client.post('/api/v1/text/optimize', json={'input': 'Um 9 Uhr.', 'language': 'de'})
    assert fallback.status_code == 200
    assert fallback.json()['text'] == 'Um 9 Uhr.' and fallback.json()['binding_origin'] == 'error'
    spoken = client.post('/api/v1/synthesize', json={'input': 'Um 9 Uhr.', 'language': 'de', 'text_optimization': True})
    assert spoken.status_code == 200
    assert spoken.json()['text_optimization']['binding_origin'] == 'error'


def test_admin_settings_roundtrip_for_the_default() -> None:
    client = make_client()
    assert client.get('/api/admin/settings').json()['text_optimization_default'] == 'off'
    updated = client.put('/api/admin/settings', json={'text_optimization_default': 'deterministic'})
    assert updated.status_code == 200
    assert updated.json()['text_optimization_default'] == 'deterministic'
    saved = json.loads((client.app.state.settings.data_dir / 'runtime_settings.json').read_text(encoding='utf-8'))
    assert saved['text_optimization_default'] == 'deterministic'
    seen = recorded_texts(client)
    assert client.post('/v1/audio/speech', json={'input': 'Um 9 Uhr.', 'language': 'de'}).status_code == 200
    assert seen == ['Um neun Uhr.']
    # An explicit 'off' (or false) over HTTP still wins over the admin default.
    off = client.post('/v1/audio/speech', json={'input': 'Um 9 Uhr.', 'language': 'de', 'text_optimization': 'off'})
    assert off.status_code == 200 and 'x-text-optimization' not in off.headers
    meta_off = client.post('/v1/audio/speech', json={'input': 'Um 9 Uhr.', 'language': 'de', 'metadata': {'text_optimization': False}})
    assert meta_off.status_code == 200
    assert seen == ['Um neun Uhr.', 'Um 9 Uhr.', 'Um 9 Uhr.']
    invalid = client.put('/api/admin/settings', json={'text_optimization_default': 'llm'})
    assert invalid.status_code == 422
