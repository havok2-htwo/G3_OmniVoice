"""Audio8Synthesizer (audio8_runtime.py) against a fake engine: what differs from MingSynthesizer, plus the shared
paths the queue relies on. The lifecycle itself is tested with Ming in test_ming_runtime.py."""
from __future__ import annotations

import asyncio
import base64
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from omnivoice_tts_server import ming_runtime
from omnivoice_tts_server.audio8.variants import AUDIO8_VARIANTS
from omnivoice_tts_server.audio8.voice_cache import voice_cache_folder, voice_tag
from omnivoice_tts_server.audio8_runtime import Audio8Synthesizer
from omnivoice_tts_server.capacity import audio8_capacity_profile, audio8_onnx_capacity_profile
from omnivoice_tts_server.config import Settings
from omnivoice_tts_server.domain.state import InMemoryStore, VoiceProfileRecord, utcnow
from omnivoice_tts_server.runtime_v2 import BatchSynthesisItem
from test_ming_runtime import FakeStream, _wav_bytes


class FakeEngine:
    output_sample_rate = 44_100
    instances: list['FakeEngine'] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.encoded: list[str] = []
        self.voices: dict[str, SimpleNamespace] = {}
        self.voice_kwargs: list[dict] = []
        self.submitted: list[dict] = []
        self.rendered: list[dict] = []
        self.loaded = False
        self.closed = False
        FakeEngine.instances.append(self)

    def load(self) -> None:
        self.loaded = True

    def close(self) -> None:
        self.closed = True

    def voice_prompt(self, key, audio, sample_rate, ref_text, **kwargs):
        self.voice_kwargs.append(kwargs)
        if key not in self.voices:
            self.encoded.append(key)
            self.voices[key] = SimpleNamespace(key=key, ref_text=ref_text)
        return self.voices[key]

    def drop_voice(self, key) -> None:
        self.voices.pop(key, None)

    def submit(self, text, *, prompt, seed, row_id=None):
        self.submitted.append({'text': text, 'prompt': prompt, 'seed': seed, 'row_id': row_id})
        stream = FakeStream(row_id)
        stream.put(np.full(len(text), 0.5, dtype=np.float32))
        stream.finish()
        return stream

    def render(self, text, *, prompt=None, seed=0, row_id=None):
        self.rendered.append({'text': text, 'prompt': prompt, 'seed': seed})
        return np.zeros(4, dtype=np.float32)

    def stats(self) -> dict:
        return {'active_rows': 0, 'pending_rows': 0, 'cuda_graphs': True}


def _profile() -> VoiceProfileRecord:
    return VoiceProfileRecord(voice_id='voice_sonja', name='SonjaV2', source='custom', created_at=utcnow(),
                              audio_bytes=_wav_bytes(), content_type='audio/wav', filename='x.wav',
                              ref_text='Hallo Welt, das ist Sonja.')


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(ming_runtime.MingSynthesizer, '_reset_process_kernel_choices', staticmethod(lambda: None))
    root = tmp_path / 'models'
    for variant in AUDIO8_VARIANTS.values():
        (root / variant.folder).mkdir(parents=True)
        (root / variant.folder / 'config.json').write_text('{}', encoding='utf-8')
    settings = Settings(data_dir=tmp_path / 'data', models_root_dir=root, audio8_max_slots=8, preferred_device='cuda',
                        default_voice='SonjaV2')
    store = InMemoryStore(max_queue_size=8)
    profile = _profile()
    store.voice_profiles[profile.voice_id] = profile
    FakeEngine.instances = []

    def make(variant: str = 'audio8_realistic') -> Audio8Synthesizer:
        return Audio8Synthesizer(settings, store, variant=variant, engine_factory=lambda **kw: FakeEngine(**kw))

    return SimpleNamespace(settings=settings, store=store, profile=profile, make=make, root=root)


def _item(index, text, **request) -> BatchSynthesisItem:
    fields = {'voice': 'SonjaV2', 'seed': None, 'language': 'de', 'stream': True, 'ref_audio': None, 'ref_text': None}
    fields.update(request)
    return BatchSynthesisItem(job_id='job', sentence_index=index, request=SimpleNamespace(**fields), text=text)


async def _collect(synth, items):
    out = []
    async for batch in synth.stream_batch(items):
        out.extend(batch)
    return out


@pytest.mark.parametrize('variant', ['audio8_realistic', 'audio8_preview', 'audio8_mini'])
def test_each_variant_loads_its_own_folder_from_the_settings(env, variant) -> None:
    synth = env.make(variant)
    assert synth.resident_models() == set()
    model, warm_ms = asyncio.run(synth.ensure_model('k2-fsa/OmniVoice-Base'))
    spec = AUDIO8_VARIANTS[variant]
    assert model == spec.base_alias and model.endswith('-Base') and warm_ms >= 0
    engine = FakeEngine.instances[-1]
    assert engine.loaded
    assert engine.kwargs == {
        'model_dir': env.root / spec.folder, 'device': 'cuda:0', 'max_slots': 8, 'cuda_graphs': True,
        'output_sample_rate': 44_100, 'first_chunk_frames': 4, 'chunk_frames': 12, 'fragment_max_chars': 200,
        'temperature': 0.8, 'top_p': 0.95, 'top_k': 50,
    }
    assert synth.resident_models() == {spec.base_alias, spec.repo_id}
    assert synth.capacity_profile == audio8_capacity_profile(8, variant)
    assert synth.supports_native_streaming and synth.supports_concurrent_batches and synth.uses_voice_profiles
    info = synth.runtime_info()
    assert info['engine'] == variant and info['loaded'] and info['revision'] == spec.revision


def test_the_cpu_variant_loads_with_its_own_settings(env) -> None:
    env.settings.audio8_onnx_workers = 3
    env.settings.audio8_onnx_threads = 2
    env.settings.audio8_output_sample_rate = 24_000
    synth = env.make('audio8_onnx')
    spec = AUDIO8_VARIANTS['audio8_onnx']
    assert synth.capacity_profile == audio8_onnx_capacity_profile(3, 'audio8_onnx')
    model, _ = asyncio.run(synth.ensure_model(None))
    assert model == spec.base_alias
    engine = FakeEngine.instances[-1]
    assert engine.kwargs == {
        'model_dir': env.root / spec.folder, 'workers': 3, 'threads': 2, 'output_sample_rate': 24_000,
        'first_chunk_frames': 6, 'chunk_frames': 8, 'fragment_max_chars': 200, 'temperature': 0.8, 'top_p': 0.95,
        'top_k': 50, 'voice_cache_dir': env.settings.data_dir / 'audio8_onnx_voices',
    }
    assert engine.rendered and engine.encoded
    assert synth.supports_native_streaming and synth.supports_concurrent_batches and synth.uses_voice_profiles
    info = synth.runtime_info()
    assert info['engine'] == 'audio8_onnx' and info['workers'] == 3 and info['license'] == 'Apache-2.0'
    assert 'cuda_graphs' not in info and 'max_slots' not in info


def test_the_default_factory_of_the_cpu_variant_is_the_onnx_engine(env, monkeypatch) -> None:
    from omnivoice_tts_server import audio8_runtime

    synth = audio8_runtime.Audio8Synthesizer(env.settings, env.store, variant='audio8_onnx')
    assert synth._engine_factory is audio8_runtime._onnx_engine_factory
    slots = audio8_runtime.Audio8Synthesizer(env.settings, env.store, variant='audio8_preview')
    assert slots._engine_factory is audio8_runtime._default_engine_factory


def test_the_cpu_variant_frees_memory_without_touching_the_gpu(env, monkeypatch) -> None:
    def no_gpu(device=None):
        raise AssertionError('nvidia-smi was queried')

    monkeypatch.setattr(ming_runtime, 'query_nvidia_smi', no_gpu)
    synth = env.make('audio8_onnx')
    asyncio.run(synth.ensure_model(None))
    result = asyncio.run(synth.free_memory())
    assert result['ok'] and 'CUDA' not in result['message'] and FakeEngine.instances[-1].closed
    assert result['released_mb'] is None


def test_the_cpu_variant_stays_loaded_while_it_renders(env, monkeypatch) -> None:
    synth = env.make('audio8_onnx')
    asyncio.run(synth.ensure_model(None))
    monkeypatch.setattr(FakeEngine, 'stats', lambda self: {'active_rows': 1, 'pending_rows': 0})
    assert 'stays loaded' in asyncio.run(synth.free_memory())['message']
    assert not FakeEngine.instances[-1].closed


def test_missing_checkpoint_names_the_model(env) -> None:
    (env.root / AUDIO8_VARIANTS['audio8_preview'].folder / 'config.json').unlink()
    with pytest.raises(RuntimeError, match='Audio8 Preview 0.6B checkpoint not found'):
        asyncio.run(env.make('audio8_preview').ensure_model(None))


def test_load_encodes_the_default_voice_and_warms_up_with_it(env) -> None:
    synth = env.make()
    asyncio.run(synth.ensure_model(None))
    engine = FakeEngine.instances[-1]
    key = '|'.join(['audio8_realistic', 'Edge0/Audio8-TTS-Realistic-0.6b', 'voice_sonja',
                    hashlib.sha1(env.profile.audio_bytes).hexdigest(), env.profile.ref_text])
    assert engine.encoded == [key]
    assert engine.rendered[0]['prompt'].key == key and engine.rendered[0]['seed'] == 0


def test_stream_batch_seeds_voices_and_markers(env) -> None:
    synth = env.make()
    results = asyncio.run(_collect(synth, [_item(0, 'Eins.', seed=7, language='Auto'), _item(1, 'Zwei.')]))
    engine = FakeEngine.instances[-1]
    assert [s['seed'] for s in engine.submitted] == [7, 0]
    assert [s['row_id'] for s in engine.submitted] == ['job:0', 'job:1']
    assert engine.submitted[0]['prompt'] is engine.submitted[1]['prompt']
    assert sum(1 for r in results if r.sentence_finished) == 2
    assert all(r.sample_rate == 44_100 for r in results)


def test_a_bad_reference_fails_only_its_item_with_a_neutral_message(env) -> None:
    results = asyncio.run(_collect(env.make(), [
        _item(0, 'Gut.'),
        _item(1, 'Kaputt.', voice=None, ref_audio='!!not base64!!', ref_text='Referenz.'),
    ]))
    finished = {r.sentence_index: r for r in results if r.sentence_finished}
    assert finished[0].error is None
    assert 'ref_audio' in finished[1].error and 'Ming' not in finished[1].error


def test_ref_text_is_required_and_named_for_the_right_engine(env) -> None:
    results = asyncio.run(_collect(env.make(), [_item(0, 'x', voice=None, ref_audio='AAAA')]))
    assert 'Audio8 Realistic 0.6B voice cloning requires ref_text' in results[0].error


def test_reload_applies_changed_settings(env) -> None:
    synth = env.make()
    asyncio.run(synth.ensure_model(None))
    env.settings.audio8_max_slots = 16
    env.settings.audio8_output_sample_rate = 24_000
    env.settings.audio8_top_k = 30
    assert synth.capacity_profile == audio8_capacity_profile(8, 'audio8_realistic')
    asyncio.run(synth.reload())
    kwargs = FakeEngine.instances[-1].kwargs
    assert (kwargs['max_slots'], kwargs['output_sample_rate'], kwargs['top_k']) == (16, 24_000, 30)
    assert synth.capacity_profile == audio8_capacity_profile(16, 'audio8_realistic')


def _codes_on_disk(env, *voice_ids) -> list:
    folder = voice_cache_folder(env.settings.data_dir)
    folder.mkdir(parents=True, exist_ok=True)
    return [folder / f'{voice_tag(voice_id)}.abc.npy' for voice_id in voice_ids]


def test_the_cpu_engine_gets_the_saved_voice_and_a_reference_of_a_request_stays_off_disk(env) -> None:
    synth = env.make('audio8_onnx')
    asyncio.run(synth.ensure_model(None))
    engine = FakeEngine.instances[-1]
    assert engine.voice_kwargs[-1] == {'voice_id': 'voice_sonja'}
    inline = 'data:audio/wav;base64,' + base64.b64encode(_wav_bytes()).decode('ascii')
    synth._prompt_for(engine, SimpleNamespace(voice='nobody', ref_audio=inline, ref_text='Inline.'))
    assert engine.voice_kwargs[-1] == {'voice_id': None}


def test_the_gpu_engines_keep_no_voice_on_disk(env) -> None:
    synth = env.make('audio8_realistic')
    asyncio.run(synth.ensure_model(None))
    assert FakeEngine.instances[-1].voice_kwargs[-1] == {}


def test_a_load_clears_codes_of_voices_deleted_while_the_engine_was_gone(env) -> None:
    kept, deleted = _codes_on_disk(env, 'voice_sonja', 'voice_deleted')
    kept.write_bytes(b'x')
    deleted.write_bytes(b'x')
    asyncio.run(env.make('audio8_onnx').ensure_model(None))
    assert kept.exists() and not deleted.exists()


def test_a_voice_change_clears_codes_written_after_the_delete_route_ran(env) -> None:
    synth = env.make('audio8_onnx')
    asyncio.run(synth.ensure_model(None))
    late = _codes_on_disk(env, 'voice_deleted')[0]
    late.write_bytes(b'x')
    synth._prompt_from_profile(FakeEngine.instances[-1], env.profile, env.profile.ref_text)
    assert late.exists(), 'nothing changed yet'
    env.store.prompt_cache.clear()  # what voice create / delete do
    synth._prompt_from_profile(FakeEngine.instances[-1], env.profile, env.profile.ref_text)
    assert not late.exists()


def test_the_gpu_engines_leave_the_cpu_engines_files_alone(env) -> None:
    other = _codes_on_disk(env, 'voice_deleted')[0]
    other.write_bytes(b'x')
    asyncio.run(env.make('audio8_mini').ensure_model(None))
    assert other.exists()


def test_free_memory_message_names_the_model(env, monkeypatch) -> None:
    monkeypatch.setattr(ming_runtime, 'query_nvidia_smi', lambda device=None: {})
    synth = env.make()
    asyncio.run(synth.ensure_model(None))
    message = asyncio.run(synth.free_memory())['message']
    assert message.startswith('Audio8 Realistic 0.6B engine closed') and FakeEngine.instances[-1].closed
