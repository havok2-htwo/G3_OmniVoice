from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from omnivoice_tts_server.config import Settings
from omnivoice_tts_server.domain.models import SpeechRequest, TaskType
from omnivoice_tts_server.domain.state import InMemoryStore, VoiceProfileRecord
from omnivoice_tts_server.services_v2 import EventHub, QueueService


def make_queue(tmp_path, *profiles: VoiceProfileRecord) -> tuple[QueueService, InMemoryStore]:
    settings = Settings(data_dir=tmp_path, sentence_chunking=False)
    store = InMemoryStore(max_queue_size=settings.max_queue_size)
    for profile in profiles:
        store.voice_profiles[profile.voice_id] = profile
    return QueueService(store, object(), EventHub(store), settings), store


def sonja_profile(ref_text: str = 'Hallo, ich bin Sonja.') -> VoiceProfileRecord:
    return VoiceProfileRecord(
        voice_id='voice_sonja',
        name='Sonja',
        source='upload',
        created_at=datetime(2026, 10, 1, tzinfo=timezone.utc),
        audio_bytes=b'RIFF',
        ref_text=ref_text,
        consent=True,
    )


def test_defaults_are_base_with_sonja(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path)
    assert settings.active_model == 'k2-fsa/OmniVoice-Base'
    assert settings.default_voice == 'Sonja'
    assert settings.torch_dtype == 'bfloat16'
    assert (settings.max_parallel_requests, settings.max_batch_size) == (8, 8)


def test_base_request_without_voice_uses_default_voice_profile(tmp_path) -> None:
    queue, _ = make_queue(tmp_path, sonja_profile())
    job = asyncio.run(queue.submit(SpeechRequest(input='Guten Tag.')))
    assert job.request.voice == 'voice_sonja'


def test_explicit_voice_is_not_replaced_by_default(tmp_path) -> None:
    other = VoiceProfileRecord(
        voice_id='voice_other',
        name='Other',
        source='upload',
        created_at=datetime(2026, 10, 1, tzinfo=timezone.utc),
        audio_bytes=b'RIFF',
        ref_text='Andere Stimme.',
        consent=True,
    )
    queue, _ = make_queue(tmp_path, sonja_profile(), other)
    job = asyncio.run(queue.submit(SpeechRequest(input='Guten Tag.', voice='Other')))
    assert job.request.voice == 'Other'


def test_base_request_without_voice_fails_when_default_profile_missing(tmp_path) -> None:
    queue, _ = make_queue(tmp_path)
    with pytest.raises(RuntimeError, match='Base voice cloning requires'):
        asyncio.run(queue.submit(SpeechRequest(input='Guten Tag.')))


def test_built_in_voice_under_base_renders_as_auto_voice(tmp_path) -> None:
    queue, store = make_queue(tmp_path, sonja_profile())
    store.active_model = 'k2-fsa/OmniVoice-Base'
    job = asyncio.run(queue.submit(SpeechRequest(input='Guten Tag.', voice='Auto Voice')))
    assert job.request.voice == 'Auto Voice'
    assert job.request.task_type == TaskType.custom_voice
    assert job.request.model is None
    assert store.active_model == 'k2-fsa/OmniVoice-Base'


def test_built_in_voice_with_explicit_base_model_is_still_rejected(tmp_path) -> None:
    queue, _ = make_queue(tmp_path, sonja_profile())
    with pytest.raises(RuntimeError, match='Base voice cloning requires'):
        asyncio.run(queue.submit(SpeechRequest(input='Guten Tag.', voice='Auto Voice', model='k2-fsa/OmniVoice-Base')))
