from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from omnivoice_tts_server.config import Settings
from omnivoice_tts_server.domain.models import WerBenchmarkCreateRequest
from omnivoice_tts_server.domain.state import InMemoryStore
from omnivoice_tts_server.services_v2 import EventHub, WerBenchmarkService


def _service(tmp_path, *, runtime_backend: str = 'mock') -> tuple[WerBenchmarkService, InMemoryStore]:
    store = InMemoryStore(max_queue_size=2)
    service = WerBenchmarkService(
        store,
        object(),
        object(),
        EventHub(store),
        Settings(data_dir=tmp_path, runtime_backend=runtime_backend),
    )
    return service, store


def test_sentence_identity_uses_wer_normalization_for_exclusions() -> None:
    umlaut = '  Das ist FÜR Gäste! '
    transliterated = 'das ist fuer Gaeste.'

    assert WerBenchmarkService._sentence_identity(umlaut) == WerBenchmarkService._sentence_identity(transliterated)


def test_sentence_cache_key_is_order_insensitive_but_includes_exclusion_set() -> None:
    common = {
        'count': 10,
        'language': 'Deutsch',
        'min_words': 4,
        'max_words': 16,
        'prompt': 'Alltagssaetze',
    }
    first = WerBenchmarkService._sentence_cache_key(
        **common,
        exclude_sentences=['Das ist für Gäste.', 'Wir fahren morgen los.'],
    )
    equivalent = WerBenchmarkService._sentence_cache_key(
        **common,
        exclude_sentences=[' wir fahren morgen los! ', 'Das ist fuer Gaeste.', 'Das ist für Gäste.'],
    )
    different = WerBenchmarkService._sentence_cache_key(
        **common,
        exclude_sentences=['Nur dieser andere Satz wird ausgeschlossen.'],
    )

    assert first == equivalent
    assert first != different


def test_mock_generation_excludes_previous_pool_and_caches_new_pool(tmp_path) -> None:
    service, _ = _service(tmp_path)
    first_payload = WerBenchmarkCreateRequest(
        count=20,
        vllm_base_url='mock',
        min_words=4,
        max_words=16,
    )
    first, first_cache = asyncio.run(service._generate_sentences(first_payload))
    second_payload = first_payload.model_copy(update={'exclude_sentences': first})

    second, second_cache = asyncio.run(service._generate_sentences(second_payload))
    repeated, repeated_cache = asyncio.run(service._generate_sentences(second_payload))

    first_keys = service._sentence_keys(first)
    second_keys = service._sentence_keys(second)
    assert len(first_keys) == len(second_keys) == first_payload.count
    assert first_keys.isdisjoint(second_keys)
    assert first_cache['key'] != second_cache['key']
    assert second_cache['hit'] is False
    assert repeated == second
    assert repeated_cache == {'hit': True, 'key': second_cache['key']}


def test_cached_pool_is_rejected_when_it_contains_excluded_or_duplicate_sentences(tmp_path) -> None:
    service, store = _service(tmp_path)
    excluded = 'Dieser klare Beispielsatz wird sicher ausgeschlossen.'
    payload = WerBenchmarkCreateRequest(
        count=2,
        vllm_base_url='mock',
        min_words=4,
        max_words=16,
        exclude_sentences=[excluded],
    )
    cache_key = service._sentence_cache_key(
        count=payload.count,
        language=payload.language,
        min_words=payload.min_words,
        max_words=payload.max_words,
        prompt=payload.prompt or '',
        exclude_sentences=payload.exclude_sentences,
    )
    store.wer_sentence_cache[cache_key] = {'sentences': [excluded, excluded]}

    sentences, cache_info = asyncio.run(service._generate_sentences(payload))

    assert cache_info == {'hit': False, 'key': cache_key}
    assert len(service._sentence_keys(sentences)) == payload.count
    assert service._sentence_identity(excluded) not in service._sentence_keys(sentences)


def test_llm_candidates_are_filtered_against_exclusions_and_local_duplicates(tmp_path, monkeypatch) -> None:
    service, _ = _service(tmp_path, runtime_backend='omnivoice')
    excluded = 'Das Wetter ist heute wunderbar fuer einen Spaziergang.'
    accepted = 'Am Nachmittag besucht meine Schwester den ruhigen Wochenmarkt.'
    payload = WerBenchmarkCreateRequest(
        count=4,
        vllm_base_url='http://vllm.test',
        vllm_model='test-model',
        min_words=4,
        max_words=16,
        exclude_sentences=[excluded],
    )
    generate_chunk = AsyncMock(
        side_effect=[
            [
                'Das Wetter ist heute wunderbar für einen Spaziergang!',
                accepted,
                accepted.upper(),
            ],
            [],
        ]
    )
    monkeypatch.setattr(service, '_generate_sentence_chunk', generate_chunk)

    sentences, cache_info = asyncio.run(service._generate_sentences(payload))

    sentence_keys = service._sentence_keys(sentences)
    assert cache_info['hit'] is False
    assert len(sentences) == len(sentence_keys) == payload.count
    assert service._sentence_identity(excluded) not in sentence_keys
    assert service._sentence_identity(accepted) in sentence_keys
    assert generate_chunk.await_count == 2


def test_fallback_scans_beyond_exclusions_for_a_disjoint_1000_sentence_pool() -> None:
    first_payload = WerBenchmarkCreateRequest(
        count=1000,
        min_words=4,
        max_words=16,
        prompt='Gleiche Vorgaben fuer beide Runden.',
    )
    first = WerBenchmarkService._fallback_sentences(first_payload, first_payload.count)
    second_payload_data = first_payload.model_dump()
    second_payload_data['exclude_sentences'] = first
    second_payload = WerBenchmarkCreateRequest.model_validate(second_payload_data)

    second = WerBenchmarkService._fallback_sentences(second_payload, second_payload.count)

    first_keys = WerBenchmarkService._sentence_keys(first)
    second_keys = WerBenchmarkService._sentence_keys(second)
    assert len(first_keys) == len(second_keys) == 1000
    assert first_keys.isdisjoint(second_keys)


def test_fallback_fails_clearly_when_word_bounds_make_request_impossible() -> None:
    payload = WerBenchmarkCreateRequest(count=1, min_words=1, max_words=1)

    with pytest.raises(RuntimeError, match='cannot satisfy the request.*only 0 were available'):
        WerBenchmarkService._fallback_sentences(payload, 1)
