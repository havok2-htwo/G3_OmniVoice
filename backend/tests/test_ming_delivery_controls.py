"""Ming delivery controls: request fields, caption building, reference mode, batching key and inline tags."""
from __future__ import annotations

import asyncio
import json
import unicodedata
from collections import deque
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from omnivoice_tts_server import ming_runtime
from omnivoice_tts_server.domain.models import SpeechRequest
from omnivoice_tts_server.domain.state import RequestState
from omnivoice_tts_server.engines import EngineSwitcher, _render_pending_tags
from omnivoice_tts_server.ming.prompt import build_structured_instruction
from omnivoice_tts_server.ming_runtime import ming_instruction, resolve_reference_mode
from omnivoice_tts_server.services_v2 import InvalidRequestFieldsError
from omnivoice_tts_server.text_optimizer.omnivoice_tags import MING_TAGS, NONVERBAL_TAGS, render_inline_tags
from omnivoice_tts_server.text_optimizer.service import optimize_request_text

from test_app import make_client
from test_text_optimization_api import make_queue, submit

# --- request fields ------------------------------------------------------------------------------


def test_delivery_controls_accept_english_values_and_german_aliases() -> None:
    request = SpeechRequest(input='x', emotion='Fröhlich', speaking_rate='LANGSAM', pitch='tief', volume='laut',
                            reference_mode='Sprecher')
    assert (request.emotion, request.speaking_rate, request.pitch, request.volume, request.reference_mode) == (
        'happy', 'slow', 'low', 'high', 'speaker')
    for alias, value in [('froehlich', 'happy'), ('traurig', 'sad'), ('wuetend', 'angry'), ('verärgert', 'angry'),
                         ('ängstlich', 'fearful'), ('angeekelt', 'disgusted'), ('ueberrascht', 'surprised'),
                         ('Neutral', 'neutral'), ('disgusted', 'disgusted')]:
        assert SpeechRequest(input='x', emotion=alias).emotion == value
    assert SpeechRequest(input='x', speaking_rate='normal').speaking_rate == 'medium'
    assert SpeechRequest(input='x', volume='mittel').volume == 'medium'
    assert SpeechRequest(input='x', emotion='  ').emotion is None
    assert SpeechRequest(input='x').delivery_controls() == {}


@pytest.mark.parametrize('field,value', [('emotion', 'furious'), ('speaking_rate', 'very fast'), ('pitch', 3),
                                         ('volume', 'loud'), ('reference_mode', 'latents')])
def test_unknown_delivery_values_are_rejected(field, value) -> None:
    with pytest.raises(ValidationError):
        SpeechRequest(input='x', **{field: value})
    with pytest.raises(ValidationError):
        SpeechRequest(input='x', metadata={field: value})


def test_metadata_delivery_controls_fill_unset_fields() -> None:
    request = SpeechRequest(input='x', emotion='sad', metadata={'emotion': 'happy', 'pitch': 'hoch',
                                                               'reference_mode': 'full'})
    assert (request.emotion, request.pitch, request.reference_mode) == ('sad', 'high', 'full')
    assert request.delivery_controls() == {'emotion': 'sad', 'pitch': 'high', 'reference_mode': 'full'}
    # The metadata value of a field sent at the top level is ignored, not validated.
    assert SpeechRequest(input='x', emotion='happy', metadata={'emotion': 'joy'}).emotion == 'happy'
    # A blank top-level field counts as not sent.
    assert SpeechRequest(input='x', emotion=' ', metadata={'emotion': 'traurig'}).emotion == 'sad'


def test_decomposed_umlaut_aliases_match() -> None:
    for alias, value in [('fröhlich', 'happy'), ('wütend', 'angry'), ('ängstlich', 'fearful'),
                         ('überrascht', 'surprised')]:
        decomposed = unicodedata.normalize('NFD', alias)
        assert decomposed != alias
        assert SpeechRequest(input='x', emotion=decomposed).emotion == value
        assert SpeechRequest(input='x', metadata={'emotion': decomposed}).emotion == value


def test_http_validation_and_the_ming_only_freeform_conflict() -> None:
    client = make_client()
    body = {'input': 'Hallo Welt', 'response_format': 'wav'}
    for path in ('/v1/audio/speech', '/api/v1/synthesize', '/api/v1/synthesize/stream', '/v1/jobs'):
        assert client.post(path, json={**body, 'emotion': 'furious'}).status_code == 422
        assert client.post(path, json={**body, 'metadata': {'volume': 'loud'}}).status_code == 422
    # OmniVoice (mock) ignores the controls, also next to instructions.
    mixed = {**body, 'instructions': 'Sprich leise.', 'emotion': 'traurig'}
    assert client.post('/v1/audio/speech', json=mixed).status_code == 200
    client.app.state.settings.runtime_backend = 'ming'
    for path in ('/v1/audio/speech', '/api/v1/synthesize', '/api/v1/synthesize/stream', '/v1/jobs'):
        response = client.post(path, json=mixed)
        assert response.status_code == 422
        assert 'emotion' in response.json()['detail']
    response = client.post('/v1/audio/speech', json={**body, 'instructions': 'x', 'metadata': {'speaking_rate': 'fast'}})
    assert response.status_code == 422


# --- caption and reference mode ------------------------------------------------------------------


def _request(**fields) -> SimpleNamespace:
    base = {'instructions': None, 'emotion': None, 'speaking_rate': None, 'pitch': None, 'volume': None,
            'reference_mode': None}
    return SimpleNamespace(**{**base, **fields})


def test_ming_instruction_builds_one_structured_caption() -> None:
    assert ming_instruction(_request()) is None
    assert ming_instruction(_request(instructions='  ')) is None
    assert ming_instruction(_request(instructions=' Sprich wie ein Nachrichtensprecher. ')) == (
        'Sprich wie ein Nachrichtensprecher.')
    # Emotion alone: an emotion-only caption.
    assert ming_instruction(_request(emotion='happy')) == build_structured_instruction(emotion='高兴')
    # One of speed / pitch / volume: BASIC with the others at medium.
    assert ming_instruction(_request(speaking_rate='slow')) == build_structured_instruction(
        speed='慢速', pitch='中', volume='中')
    assert ming_instruction(_request(volume='low', pitch='high')) == build_structured_instruction(
        speed='中速', pitch='高', volume='低')
    # Emotion and attributes together: one caption.
    combined = ming_instruction(_request(emotion='angry', speaking_rate='fast'))
    assert combined == build_structured_instruction(emotion='愤怒', speed='快速', pitch='中', volume='中')
    item = json.loads(combined)['audio_sequence'][0]
    assert (item['情感'], item['语速'], item['基频'], item['音量']) == ('愤怒', '快速', '中', '中')
    with pytest.raises(ValueError, match='cannot be combined'):
        ming_instruction(_request(instructions='leise', pitch='low'))


def test_reference_mode_resolution(monkeypatch) -> None:
    assert resolve_reference_mode(_request()) == 'full'
    assert resolve_reference_mode(_request(instructions='leise')) == 'full'
    assert resolve_reference_mode(_request(reference_mode='auto', emotion='sad')) == 'speaker'
    assert resolve_reference_mode(_request(volume='high')) == 'speaker'
    assert resolve_reference_mode(_request(reference_mode='full', emotion='sad')) == 'full'
    assert resolve_reference_mode(_request(reference_mode='speaker')) == 'speaker'
    # The auto rule is one constant.
    monkeypatch.setattr(ming_runtime, 'AUTO_REFERENCE_MODE_WITH_CONTROLS', 'full')
    assert resolve_reference_mode(_request(emotion='sad')) == 'full'


# --- queue -----------------------------------------------------------------------------------------


def test_batch_group_key_differs_per_delivery_control(tmp_path) -> None:
    queue, _ = make_queue(tmp_path)
    base = {'input': 'Hallo.', 'voice': 'SonjaV2', 'model': 'k2-fsa/OmniVoice-Base'}
    keys = [
        queue._group_key_for_request(SpeechRequest(**base)),
        queue._group_key_for_request(SpeechRequest(**base, emotion='happy')),
        queue._group_key_for_request(SpeechRequest(**base, emotion='sad')),
        queue._group_key_for_request(SpeechRequest(**base, emotion='sad', pitch='low')),
        queue._group_key_for_request(SpeechRequest(**base, emotion='sad', reference_mode='full')),
        queue._group_key_for_request(SpeechRequest(**base, metadata={'speaking_rate': 'fast'})),
        queue._group_key_for_request(SpeechRequest(**base, speaking_rate='schnell')),
    ]
    assert len(set(keys[:6])) == 6
    assert keys[5] != keys[6]  # metadata itself is part of the key too
    assert queue._group_key_for_request(SpeechRequest(**base, emotion='froehlich')) == keys[1]


def test_queue_rejects_freeform_with_controls_on_ming_only(tmp_path) -> None:
    queue, _ = make_queue(tmp_path, runtime_backend='ming')
    with pytest.raises(InvalidRequestFieldsError):
        submit(queue, SpeechRequest(input='Hallo.', instructions='leise', emotion='sad'))
    job = submit(queue, SpeechRequest(input='Hallo.', instructions='leise'))
    assert job.request.instructions == 'leise'
    omni, _ = make_queue(tmp_path / 'omni')
    assert submit(omni, SpeechRequest(input='Hallo.', instructions='leise', emotion='sad')).request.emotion == 'sad'


# --- inline tags -----------------------------------------------------------------------------------


def test_tags_are_spelled_for_the_engine_and_unknown_ones_dropped() -> None:
    text = 'Na ja [LAUGHTER] gut [Breathing], dann [question-yi] los [Question-huh].'
    assert render_inline_tags(text, 'ming') == 'Na ja [Laughter] gut [Breathing], dann los [Question-huh].'
    assert render_inline_tags(text, 'omnivoice') == 'Na ja [laughter] gut, dann [question-yi] los.'
    assert render_inline_tags(text, 'phonon') == 'Na ja gut, dann los.'
    assert render_inline_tags('[Cough] Hallo [Uhm] [sigh] Welt', 'omnivoice') == 'Hallo [sigh] Welt'
    # OmniVoice's own lowercase tags, ARPAbet spans and Markdown links stay as they are.
    for tag in NONVERBAL_TAGS:
        assert render_inline_tags(f'a [{tag}] b', 'omnivoice') == f'a [{tag}] b'
    for tag in MING_TAGS:
        assert render_inline_tags(f'a [{tag.lower()}] b', 'ming') == f'a [{tag}] b'
    kept = 'Sag [B EY1 S] und [Laughter](https://example.org).'
    assert render_inline_tags(kept, 'ming') is kept
    plain = 'Kein Tag [1] hier.'
    assert render_inline_tags(plain, 'omnivoice') is plain


@pytest.mark.parametrize('optimization', ['off', 'deterministic'])
def test_queue_renders_tags_for_the_active_engine(tmp_path, optimization) -> None:
    text = 'Am 3.10. um 14:30 Uhr [laughter] kommt er [Cough] an.'
    ming, ming_store = make_queue(tmp_path / 'ming', runtime_backend='ming')
    job = submit(ming, SpeechRequest(input=text, language='Deutsch', text_optimization=optimization))
    spoken = ' '.join(ming_store.request_states[job.job_id].sentences)
    assert '[Laughter]' in spoken and '[Cough]' in spoken
    omni, omni_store = make_queue(tmp_path / 'omni')
    job = submit(omni, SpeechRequest(input=text, language='Deutsch', text_optimization=optimization))
    spoken = ' '.join(omni_store.request_states[job.job_id].sentences)
    assert '[laughter]' in spoken and 'Cough' not in spoken and 'kommt er an.' in spoken
    if optimization == 'deterministic':
        assert 'vierzehn Uhr dreißig' in spoken and 'laughter]' in job.spoken_text
    else:
        assert '14:30' in spoken


def test_glued_tags_are_tags() -> None:
    assert render_inline_tags('Na ja [laughter][Breathing] gut.', 'ming') == 'Na ja [Laughter][Breathing] gut.'
    assert render_inline_tags('Na ja [Cough][sigh] gut.', 'omnivoice') == 'Na ja [sigh] gut.'
    assert render_inline_tags('[Uhm][Cough][laughter] gut.', 'phonon') == 'gut.'
    # With text optimization the Markdown stripper must not read the first one as a reference link.
    assert optimize_request_text('Na ja [Uhm][Cough] gut.', 'de').text == 'Na ja [Uhm][Cough] gut.'
    # Real Markdown links and references are still stripped.
    assert optimize_request_text('Siehe [Doku][1] und [laughter](https://x.de).', 'de').text == 'Siehe Doku und laughter.'


def test_a_dropped_tag_between_two_words_leaves_a_space() -> None:
    assert render_inline_tags('Hallo[Cough]Welt', 'omnivoice') == 'Hallo Welt'
    assert render_inline_tags('gut[laughter]danke', 'phonon') == 'gut danke'
    assert render_inline_tags('Hallo[Cough], Welt', 'omnivoice') == 'Hallo, Welt'
    assert render_inline_tags('你好[Cough]呀', 'omnivoice') == '你好呀'


@pytest.mark.parametrize('optimization', ['off', 'deterministic'])
def test_queue_keeps_glued_tags_as_tags(tmp_path, optimization) -> None:
    ming, ming_store = make_queue(tmp_path, runtime_backend='ming')
    job = submit(ming, SpeechRequest(input='Na ja [uhm][laughter] gut.', language='Deutsch',
                                     text_optimization=optimization))
    assert ' '.join(ming_store.request_states[job.job_id].sentences) == 'Na ja [Uhm][Laughter] gut.'


def test_a_text_of_only_unknown_tags_is_rejected(tmp_path) -> None:
    omni, _ = make_queue(tmp_path)
    with pytest.raises(RuntimeError, match='Missing input text'):
        submit(omni, SpeechRequest(input='[Breathing] [Cough]'))


def _state(sentences: list[str], pending: list[int], next_emit: int = 0) -> RequestState:
    state = RequestState(job_id='j', group_key='g', sentences=list(sentences))
    state.pending_sentence_indices = deque(pending)
    state.next_emit_sentence_index = next_emit
    return state


def test_engine_switch_respells_the_pending_sentences() -> None:
    state = _state(['[laughter] Eins.', 'Zwei [sigh].', '[question-yi]', '[surprise-wa].', 'Drei.'], [1, 2, 3, 4], 1)
    # Sentence 0 is already rendered. Sentences 2 and 3 have nothing left to say on Ming: dropped, not spoken.
    assert _render_pending_tags(state, 'ming') == [2, 3]
    assert state.sentences[:2] == ['[laughter] Eins.', 'Zwei [Sigh].']
    assert list(state.pending_sentence_indices) == [1, 4]
    assert state.next_emit_sentence_index == 1
    assert state.ready_sentence_pcm == {2: b'', 3: b''}
    assert state.completed_streaming_sentence_indices == {2, 3}

    # A dropped sentence at the front: the next one is the first to emit.
    leading = _state(['[question-yi]', 'Eins.'], [0, 1])
    assert _render_pending_tags(leading, 'ming') == [0]
    assert (list(leading.pending_sentence_indices), leading.next_emit_sentence_index) == ([1], 1)
    assert leading.ready_sentence_pcm == {}
    # Phonon knows no tags: "[Laughter]." would leave a lone "." to synthesize.
    phonon = _state(['Gut.', '[Laughter].', '[Laughter] Ja.'], [0, 1, 2])
    assert _render_pending_tags(phonon, 'phonon') == [1]
    assert phonon.sentences == ['Gut.', '[Laughter].', 'Ja.']
    assert list(phonon.pending_sentence_indices) == [0, 2]


def test_dropped_sentences_are_stepped_over_on_both_emit_paths(tmp_path) -> None:
    queue, store = make_queue(tmp_path)
    job = submit(queue, SpeechRequest(input='Eins. Zwei.'))
    pcm = b'\x01\x00' * 240
    # Rendered batch path.
    state = _state(['Eins.', '[Cough]', 'Zwei.'], [0, 1, 2])
    _render_pending_tags(state, 'omnivoice')
    state.pending_sentence_indices.clear()
    state.ready_sentence_pcm.update({0: pcm, 2: pcm})
    queue._flush_ready_sentences_locked(job, state)
    assert state.next_emit_sentence_index == 3 and state.is_complete()
    assert b''.join(job.pcm_parts) == pcm + pcm
    # Native stream path.
    job.pcm_parts = []
    native = _state(['Eins.', '[Cough]', 'Zwei.'], [0, 1, 2])
    _render_pending_tags(native, 'omnivoice')
    native.pending_sentence_indices.clear()
    native.pending_preview_pcm.update({0: [pcm], 2: [pcm]})
    native.completed_streaming_sentence_indices.update({0, 2})
    queue._flush_native_streaming_sentences_locked(job, native, batch_id='b')
    assert native.next_emit_sentence_index == 3
    assert b''.join(job.pcm_parts) == pcm + pcm


def _switcher(queue, store) -> EngineSwitcher:
    return EngineSwitcher(SimpleNamespace(store=store, queue_service=queue, settings=queue.settings), {})


def test_requeue_finishes_a_held_job_left_without_speakable_sentences(tmp_path) -> None:
    queue, store = make_queue(tmp_path, runtime_backend='ming')
    queue.synthesizer = SimpleNamespace(pcm_to_wav=lambda pcm, sample_rate: b'RIFF' + pcm)
    only_tags = submit(queue, SpeechRequest(input='[Breathing] [Uhm]'))
    partly = submit(queue, SpeechRequest(input='Eins. Zwei.'))
    kept = submit(queue, SpeechRequest(input='Hallo [Breathing] Welt.'))
    for job in (only_tags, partly, kept):
        store.waiting_requests.remove(job.job_id)  # held by the switch
    # 'partly': sentence 0 already emitted, only a Ming-only tag is left.
    state = store.request_states[partly.job_id]
    state.sentences = ['Eins.', '[Cough]']
    state.pending_sentence_indices = deque([1])
    state.next_emit_sentence_index = 1
    partly.pcm_parts = [b'\x01\x00' * 10]

    queue.settings.runtime_backend = 'omnivoice'
    held = [only_tags.job_id, partly.job_id, kept.job_id]
    assert asyncio.run(_switcher(queue, store)._requeue(held)) == 1
    assert list(store.waiting_requests) == [kept.job_id]
    assert only_tags.status.value == 'failed' and 'empty on the omnivoice engine' in only_tags.error_message
    assert partly.status.value == 'completed' and partly.final_audio == b'RIFF' + b'\x01\x00' * 10
    assert store.request_states[kept.job_id].sentences == ['Hallo Welt.']


def test_requeue_onto_ming_drops_freeform_instructions_next_to_structured_controls(tmp_path) -> None:
    queue, store = make_queue(tmp_path)
    mixed = submit(queue, SpeechRequest(input='Hallo.', instructions='Sprich leise.', emotion='traurig'))
    freeform = submit(queue, SpeechRequest(input='Hallo.', instructions='Sprich leise.'))
    for job in (mixed, freeform):
        store.waiting_requests.remove(job.job_id)
    queue.settings.runtime_backend = 'ming'
    assert asyncio.run(_switcher(queue, store)._requeue([mixed.job_id, freeform.job_id])) == 2
    # The structured controls win; the job does not fail on Ming for it.
    assert (mixed.request.instructions, mixed.request.emotion) == (None, 'sad')
    assert ming_instruction(mixed.request) == build_structured_instruction(emotion='悲伤')
    assert freeform.request.instructions == 'Sprich leise.'
    assert store.request_states[mixed.job_id].group_key == queue._group_key_for_request(mixed.request)

