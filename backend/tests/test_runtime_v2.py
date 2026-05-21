from __future__ import annotations

import asyncio
import io
import sys
import types
import wave
from pathlib import Path

import numpy as np
import pytest

from omnivoice_tts_server.config import Settings
from omnivoice_tts_server.domain.models import SpeechRequest, TaskType
from omnivoice_tts_server.domain.state import InMemoryStore, VoiceProfileRecord, utcnow
from omnivoice_tts_server.runtime_v2 import (
    OMNIVOICE_AUTO_ALIAS,
    OMNIVOICE_BASE_ALIAS,
    OMNIVOICE_DESIGN_ALIAS,
    BatchSynthesisItem,
    OmniVoiceSynthesizer,
)
from omnivoice_tts_server.voice_design import (
    DEFAULT_VOICE_DESIGN_INSTRUCT,
    normalize_voice_design_instruct,
)


def make_settings(tmp_path: Path, **overrides) -> Settings:
    return Settings(
        runtime_backend='omnivoice',
        preferred_device='cpu',
        models_root_dir=tmp_path / 'models',
        data_dir=tmp_path / 'data',
        allow_model_downloads=True,
        warmup_on_startup=False,
        **overrides,
    )


def make_wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24_000)
        wav_file.writeframes(b'\x00\x00' * 2400)
    return buffer.getvalue()


class FakeOmniVoiceModel:
    instances: list['FakeOmniVoiceModel'] = []

    def __init__(self) -> None:
        self.sampling_rate = 24_000
        self.calls: list[dict] = []
        self.clone_prompt_calls = 0
        self.llm = object()
        FakeOmniVoiceModel.instances.append(self)

    def generate(self, text, **kwargs):
        texts = text if isinstance(text, list) else [text]
        self.calls.append({'text': list(texts), **kwargs})
        return [np.linspace(-0.2, 0.2, 2400, dtype=np.float32) for _ in texts]

    def create_voice_clone_prompt(self, ref_audio, ref_text, preprocess_prompt=True):
        self.clone_prompt_calls += 1
        return {
            'ref_audio': ref_audio,
            'ref_text': ref_text,
            'preprocess_prompt': preprocess_prompt,
            'call': self.clone_prompt_calls,
        }


class FakeOmniVoice:
    load_calls: list[dict] = []

    @classmethod
    def from_pretrained(cls, model_source, **kwargs):
        cls.load_calls.append({'model_source': model_source, **kwargs})
        return FakeOmniVoiceModel()


def install_fake_omnivoice(monkeypatch) -> None:
    FakeOmniVoice.load_calls.clear()
    FakeOmniVoiceModel.instances.clear()
    module = types.ModuleType('omnivoice')
    module.OmniVoice = FakeOmniVoice
    monkeypatch.setitem(sys.modules, 'omnivoice', module)


def run_render(synthesizer: OmniVoiceSynthesizer, items: list[BatchSynthesisItem]):
    return asyncio.run(synthesizer.render_batch(items))


def test_voice_design_instruct_normalization_and_errors() -> None:
    assert normalize_voice_design_instruct(None) == DEFAULT_VOICE_DESIGN_INSTRUCT
    assert normalize_voice_design_instruct('Female, Young Adult, low pitch') == 'female, young adult, low pitch'
    assert normalize_voice_design_instruct('female, young adult, 河南话') == 'female, young adult, 河南话'

    with pytest.raises(RuntimeError, match='keine freien Beschreibungen'):
        normalize_voice_design_instruct('A calm neutral voice.')

    with pytest.raises(RuntimeError, match='pro Kategorie nur einen Wert'):
        normalize_voice_design_instruct('female, male')

    with pytest.raises(RuntimeError, match='Accent und Chinese Dialect'):
        normalize_voice_design_instruct('british accent, 河南话')


def test_auto_voice_batches_generation_params_and_pcm(tmp_path, monkeypatch) -> None:
    install_fake_omnivoice(monkeypatch)
    settings = make_settings(
        tmp_path,
        num_step=16,
        guidance_scale=1.25,
        duration=2.5,
        denoise=True,
    )
    store = InMemoryStore(max_queue_size=8)
    synthesizer = OmniVoiceSynthesizer(settings=settings, store=store)

    model_id, warm_ms = asyncio.run(synthesizer.ensure_model(OMNIVOICE_AUTO_ALIAS))
    assert model_id == OMNIVOICE_AUTO_ALIAS
    assert warm_ms >= 0
    assert FakeOmniVoice.load_calls[-1]['load_asr'] is False

    results = run_render(
        synthesizer,
        [
            BatchSynthesisItem(
                job_id='job_1',
                sentence_index=0,
                text='Hallo OmniVoice.',
                request=SpeechRequest(
                    input='Hallo OmniVoice.',
                    model=OMNIVOICE_AUTO_ALIAS,
                    task_type=TaskType.custom_voice,
                    language='Auto',
                    instructions='warm, direct',
                    seed=42,
                ),
            )
        ],
    )

    call = FakeOmniVoiceModel.instances[-1].calls[-1]
    assert call['text'] == ['Hallo OmniVoice.']
    assert call['language'] == [None]
    assert call['instruct'] == ['warm, direct']
    assert call['num_step'] == 16
    assert call['guidance_scale'] == 1.25
    assert call['duration'] == 2.5
    assert call['denoise'] is True
    assert results[0].sample_rate == 24_000
    assert len(results[0].pcm) == 4800

    wav_bytes = synthesizer.pcm_to_wav(results[0].pcm)
    assert wav_bytes.startswith(b'RIFF')


def test_voice_design_uses_instruct_batch(tmp_path, monkeypatch) -> None:
    install_fake_omnivoice(monkeypatch)
    settings = make_settings(tmp_path)
    store = InMemoryStore(max_queue_size=8)
    synthesizer = OmniVoiceSynthesizer(settings=settings, store=store)
    asyncio.run(synthesizer.ensure_model(OMNIVOICE_DESIGN_ALIAS))

    run_render(
        synthesizer,
        [
            BatchSynthesisItem(
                job_id='job_1',
                sentence_index=0,
                text='Voice design sample.',
                request=SpeechRequest(
                    input='Voice design sample.',
                    model=OMNIVOICE_DESIGN_ALIAS,
                    task_type=TaskType.voice_design,
                    instructions='female, young adult, low pitch',
                    language='English',
                ),
            )
        ],
    )

    call = FakeOmniVoiceModel.instances[-1].calls[-1]
    assert call['instruct'] == ['female, young adult, low pitch']
    assert call['language'] == ['English']


def test_base_clone_uses_saved_profile_and_prompt_cache(tmp_path, monkeypatch) -> None:
    install_fake_omnivoice(monkeypatch)
    settings = make_settings(tmp_path, preprocess_prompt=False)
    store = InMemoryStore(max_queue_size=8)
    store.voice_profiles['voice_test'] = VoiceProfileRecord(
        voice_id='voice_test',
        name='Test Voice',
        source='custom',
        created_at=utcnow(),
        audio_bytes=make_wav_bytes(),
        content_type='audio/wav',
        filename='sample.wav',
        ref_text='Das ist die Referenz.',
        consent=True,
    )
    synthesizer = OmniVoiceSynthesizer(settings=settings, store=store)
    asyncio.run(synthesizer.ensure_model(OMNIVOICE_BASE_ALIAS))

    request = SpeechRequest(
        input='Bitte klone diese Stimme.',
        model=OMNIVOICE_BASE_ALIAS,
        voice='voice_test',
        task_type=TaskType.base,
    )
    item = BatchSynthesisItem(job_id='job_1', sentence_index=0, text=request.input or '', request=request)
    run_render(synthesizer, [item])
    run_render(synthesizer, [item])

    model = FakeOmniVoiceModel.instances[-1]
    assert model.clone_prompt_calls == 1
    first_call = model.calls[0]
    assert first_call['voice_clone_prompt'][0]['ref_text'] == 'Das ist die Referenz.'
    assert first_call['voice_clone_prompt'][0]['preprocess_prompt'] is False


def test_batch_group_keys_separate_clone_design_and_auto(tmp_path) -> None:
    settings = make_settings(tmp_path)
    store = InMemoryStore(max_queue_size=8)
    from omnivoice_tts_server.services_v2 import EventHub, QueueService
    from omnivoice_tts_server.runtime_v2 import MockSynthesizer

    queue = QueueService(store, MockSynthesizer(), EventHub(store), settings)
    auto_key = queue._group_key_for_request(SpeechRequest(input='A', model=OMNIVOICE_AUTO_ALIAS, task_type=TaskType.custom_voice))
    design_key = queue._group_key_for_request(
        SpeechRequest(input='A', model=OMNIVOICE_DESIGN_ALIAS, task_type=TaskType.voice_design, instructions='female')
    )
    clone_key = queue._group_key_for_request(
        SpeechRequest(input='A', model=OMNIVOICE_BASE_ALIAS, task_type=TaskType.base, ref_audio='sample.wav', ref_text='Hallo')
    )

    assert len({auto_key, design_key, clone_key}) == 3
