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
    _disable_windows_deepgemm_hub_kernel,
    normalize_runtime_device,
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


def test_runtime_device_normalization() -> None:
    assert normalize_runtime_device('cuda') == 'cuda:0'
    assert normalize_runtime_device('CUDA:1') == 'cuda:1'
    assert normalize_runtime_device(' cpu ') == 'cpu'
    with pytest.raises(ValueError):
        normalize_runtime_device('intel')


def test_windows_fp8_disables_deepgemm_hub_lookup(monkeypatch) -> None:
    hub_kernels = types.SimpleNamespace(
        _HUB_KERNEL_MAPPING={
            'deep-gemm': {'repo_id': 'kernels-community/deep-gemm', 'version': 1},
            'finegrained-fp8': {'repo_id': 'kernels-community/finegrained-fp8', 'version': 1},
        },
        _KERNEL_MODULE_MAPPING={'deep-gemm': object()},
    )
    finegrained_fp8 = types.SimpleNamespace(_load_deepgemm_kernel=lambda: None)

    monkeypatch.setattr('omnivoice_tts_server.runtime_v2.os.name', 'nt')

    assert _disable_windows_deepgemm_hub_kernel(hub_kernels, finegrained_fp8) is True
    assert 'deep-gemm' not in hub_kernels._HUB_KERNEL_MAPPING
    assert 'finegrained-fp8' in hub_kernels._HUB_KERNEL_MAPPING
    assert hub_kernels._KERNEL_MODULE_MAPPING['deep-gemm'] is None

    with pytest.raises(ImportError, match='disabled on Windows'):
        finegrained_fp8._load_deepgemm_kernel()


def test_deepgemm_hook_is_noop_off_windows(monkeypatch) -> None:
    hub_kernels = types.SimpleNamespace(
        _HUB_KERNEL_MAPPING={'deep-gemm': {'repo_id': 'kernels-community/deep-gemm', 'version': 1}},
        _KERNEL_MODULE_MAPPING={},
    )
    finegrained_fp8 = types.SimpleNamespace(_load_deepgemm_kernel=lambda: None)

    monkeypatch.setattr('omnivoice_tts_server.runtime_v2.os.name', 'posix')

    assert _disable_windows_deepgemm_hub_kernel(hub_kernels, finegrained_fp8) is False
    assert 'deep-gemm' in hub_kernels._HUB_KERNEL_MAPPING
    assert finegrained_fp8._load_deepgemm_kernel() is None


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
    # UI label 'English' is now normalized to the ISO code 'en' that generate() resolves.
    assert call['language'] == ['en']


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


def test_alias_switch_reuses_resident_checkpoint(tmp_path, monkeypatch) -> None:
    # AutoVoice/VoiceDesign/Base all map to ONE checkpoint; switching alias must
    # re-point the task type WITHOUT evicting and reloading identical weights.
    install_fake_omnivoice(monkeypatch)
    settings = make_settings(tmp_path)
    store = InMemoryStore(max_queue_size=8)
    synthesizer = OmniVoiceSynthesizer(settings=settings, store=store)

    asyncio.run(synthesizer.ensure_model(OMNIVOICE_AUTO_ALIAS))
    assert len(FakeOmniVoice.load_calls) == 1
    assert OMNIVOICE_BASE_ALIAS in synthesizer.resident_models()

    model_id, warm_ms = asyncio.run(synthesizer.ensure_model(OMNIVOICE_BASE_ALIAS))
    assert model_id == OMNIVOICE_BASE_ALIAS
    assert warm_ms == 0
    assert len(FakeOmniVoice.load_calls) == 1  # no second from_pretrained
    assert synthesizer._loaded_model_id == OMNIVOICE_BASE_ALIAS

    asyncio.run(synthesizer.ensure_model(OMNIVOICE_DESIGN_ALIAS))
    assert len(FakeOmniVoice.load_calls) == 1  # still no reload

    # An explicit reload() DOES re-load the weights (used after dtype/compile changes).
    asyncio.run(synthesizer.reload(OMNIVOICE_AUTO_ALIAS))
    assert len(FakeOmniVoice.load_calls) == 2


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


def _auto_item(text: str = 'Hallo.') -> BatchSynthesisItem:
    return BatchSynthesisItem(
        job_id='job_trim',
        sentence_index=0,
        text=text,
        request=SpeechRequest(input=text, model=OMNIVOICE_AUTO_ALIAS, task_type=TaskType.custom_voice),
    )


def _record_trims(synthesizer: OmniVoiceSynthesizer, monkeypatch) -> list[dict]:
    calls: list[dict] = []
    monkeypatch.setattr(synthesizer, '_trim_cuda_cache_sync', lambda **kwargs: calls.append(kwargs))
    return calls


def test_per_batch_trim_skips_gc_collect(tmp_path, monkeypatch) -> None:
    install_fake_omnivoice(monkeypatch)
    synthesizer = OmniVoiceSynthesizer(settings=make_settings(tmp_path, cuda_memory_trim_after_batch=True), store=InMemoryStore(max_queue_size=8))
    asyncio.run(synthesizer.ensure_model(OMNIVOICE_AUTO_ALIAS))
    calls = _record_trims(synthesizer, monkeypatch)

    run_render(synthesizer, [_auto_item()])

    # The hot path only releases the CUDA pool; a full gc.collect() cost ~215 ms per batch.
    assert calls == [{'collect_garbage': False}]


def test_trim_collect_garbage_flag_controls_gc(tmp_path, monkeypatch) -> None:
    import omnivoice_tts_server.runtime_v2 as runtime_module

    synthesizer = OmniVoiceSynthesizer(settings=make_settings(tmp_path), store=InMemoryStore(max_queue_size=8))
    collected: list[int] = []
    monkeypatch.setattr(runtime_module.gc, 'collect', lambda *args: collected.append(1) or 0)

    synthesizer._trim_cuda_cache_sync(collect_garbage=False)
    assert collected == []
    synthesizer._trim_cuda_cache_sync()
    assert collected == [1]


def test_failed_generate_keeps_compiled_graphs_unless_oom(tmp_path, monkeypatch) -> None:
    import torch

    install_fake_omnivoice(monkeypatch)
    synthesizer = OmniVoiceSynthesizer(settings=make_settings(tmp_path), store=InMemoryStore(max_queue_size=8))
    asyncio.run(synthesizer.ensure_model(OMNIVOICE_AUTO_ALIAS))
    calls = _record_trims(synthesizer, monkeypatch)
    model = FakeOmniVoiceModel.instances[-1]

    def fail(exc):
        def generate(text, **kwargs):
            raise exc
        return generate

    # An ordinary failure (bad input, missing voice) must not reset dynamo: the next
    # request would re-trace and recompile the LLM (~35 s measured on the RTX 4090).
    monkeypatch.setattr(model, 'generate', fail(RuntimeError('bad request')))
    with pytest.raises(RuntimeError):
        run_render(synthesizer, [_auto_item()])
    assert calls[-1] == {'reset_compile_cache': False}

    monkeypatch.setattr(model, 'generate', fail(torch.cuda.OutOfMemoryError('CUDA out of memory.')))
    with pytest.raises(torch.cuda.OutOfMemoryError):
        run_render(synthesizer, [_auto_item()])
    assert calls[-1] == {'reset_compile_cache': True}


def _omnivoice_style_mask(c_lens: list[int], u_lens: list[int]):
    """Dense (2B, 1, S, S) bool mask exactly as OmniVoice._generate_iterative builds it."""
    import torch

    batch = len(c_lens)
    size = max(c_lens)
    mask = torch.zeros((2 * batch, 1, size, size), dtype=torch.bool)
    for i, (c_len, u_len) in enumerate(zip(c_lens, u_lens)):
        mask[i, :, :c_len, :c_len] = True
        mask[batch + i, :, :u_len, :u_len] = True
        if size > u_len:
            pad_diag = torch.arange(u_len, size)
            mask[batch + i, :, pad_diag, pad_diag] = True
    return mask


def test_flex_block_mask_matches_dense_sdpa_mask() -> None:
    import torch
    from torch.nn.attention.flex_attention import BlockMask, flex_attention

    from omnivoice_tts_server.runtime_v2 import install_flex_block_mask_forward

    mask = _omnivoice_style_mask(c_lens=[150, 90], u_lens=[60, 40])

    class TinyModel:
        def forward(self, input_ids=None, attention_mask=None):
            return attention_mask

    model = TinyModel()
    assert install_flex_block_mask_forward(model) is True
    block_mask = model.forward(input_ids=None, attention_mask=mask)
    assert isinstance(block_mask, BlockMask)
    # The same mask object (one generate(), num_step forwards) reuses the BlockMask.
    assert model.forward(input_ids=None, attention_mask=mask) is block_mask
    # A new mask (next generate()) is converted again, never served from the cache.
    other = _omnivoice_style_mask(c_lens=[150, 90], u_lens=[70, 40])
    assert model.forward(input_ids=None, attention_mask=other) is not block_mask
    # Absent masks pass through untouched.
    assert model.forward(input_ids=None, attention_mask=None) is None

    torch.manual_seed(0)
    rows, heads, size, dim = mask.shape[0], 2, mask.shape[-1], 16
    q, k, v = (torch.randn(rows, heads, size, dim) for _ in range(3))
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    out = flex_attention(q, k, v, block_mask=block_mask)
    valid = mask[:, 0].any(dim=-1)  # rows SDPA defines (fully masked pad rows are unused)
    assert torch.allclose(out.permute(0, 2, 1, 3)[valid], ref.permute(0, 2, 1, 3)[valid], atol=1e-5)


def test_flex_kernel_options_wrap_the_llm_attention() -> None:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, AttentionInterface

    from omnivoice_tts_server.runtime_v2 import flex_kernel_options_for_device, install_flex_kernel_options

    seen: list[dict] = []
    AttentionInterface.register('test_flex_base', lambda module, q, k, v, *args, **kwargs: seen.append(kwargs) or 'out')
    switched: list[str] = []
    llm = types.SimpleNamespace(
        config=types.SimpleNamespace(_attn_implementation='test_flex_base'),
        set_attn_implementation=switched.append,
    )
    options = {'BLOCK_M': 64, 'BLOCK_N': 64, 'num_stages': 2, 'num_warps': 4}
    assert install_flex_kernel_options(types.SimpleNamespace(llm=llm), options) is True
    assert switched == ['g3_flex_attention']
    assert ALL_ATTENTION_FUNCTIONS['g3_flex_attention'](None, 1, 2, 3, None, scaling=0.1) == 'out'
    assert seen[-1]['kernel_options'] == options

    # sdpa models are left alone.
    llm.config._attn_implementation = 'sdpa'
    assert install_flex_kernel_options(types.SimpleNamespace(llm=llm), options) is False

    def props(smem):
        return types.SimpleNamespace(cuda=types.SimpleNamespace(get_device_properties=lambda device: types.SimpleNamespace(shared_memory_per_block_optin=smem)))

    assert flex_kernel_options_for_device(props(101376), 'cuda:0') == options  # RTX 4090
    assert flex_kernel_options_for_device(props(232448), 'cuda:0') is None  # H100: inductor default fits


def test_flex_attention_load_installs_mask_patch(tmp_path, monkeypatch) -> None:
    install_fake_omnivoice(monkeypatch)
    monkeypatch.setattr(FakeOmniVoiceModel, 'forward', lambda self, **kwargs: kwargs, raising=False)
    synthesizer = OmniVoiceSynthesizer(
        settings=make_settings(tmp_path, attention_implementation='flex', compile_model=False),
        store=InMemoryStore(max_queue_size=8),
    )
    asyncio.run(synthesizer.ensure_model(OMNIVOICE_AUTO_ALIAS))

    assert FakeOmniVoice.load_calls[-1]['attn_implementation'] == 'flex_attention'
    assert synthesizer.attention_backend == 'flex_attention'
    assert hasattr(FakeOmniVoiceModel.instances[-1], '_g3_flex_block_mask_cache')


def test_unknown_attention_falls_back_to_sdpa(tmp_path, monkeypatch) -> None:
    install_fake_omnivoice(monkeypatch)
    synthesizer = OmniVoiceSynthesizer(
        settings=make_settings(tmp_path, attention_implementation='flash_attention_9', compile_model=False),
        store=InMemoryStore(max_queue_size=8),
    )
    asyncio.run(synthesizer.ensure_model(OMNIVOICE_AUTO_ALIAS))

    assert FakeOmniVoice.load_calls[-1]['attn_implementation'] == 'sdpa'
    assert synthesizer.attention_backend == 'sdpa'
    assert 'flash_attention_9' in (synthesizer.attention_backend_note or '')


def test_flashinfer_request_is_skipped_off_linux(tmp_path, monkeypatch) -> None:
    import torch

    install_fake_omnivoice(monkeypatch)
    monkeypatch.setattr('omnivoice_tts_server.runtime_v2.sys.platform', 'win32')
    synthesizer = OmniVoiceSynthesizer(
        settings=make_settings(tmp_path, enable_flashinfer=True, torch_dtype='bfloat16', compile_model=False),
        store=InMemoryStore(max_queue_size=8),
    )
    asyncio.run(synthesizer.ensure_model(OMNIVOICE_AUTO_ALIAS))

    load = FakeOmniVoice.load_calls[-1]
    assert load['dtype'] == torch.bfloat16
    assert load['attn_implementation'] == 'sdpa'
    assert synthesizer.attention_backend == 'sdpa'
    assert 'Linux' in (synthesizer.attention_backend_note or '')


def _install_fake_flashinfer(monkeypatch, apply_flashinfer):
    import omnivoice_tts_server.runtime_v2 as runtime_module

    fi_module = types.ModuleType('omnivoice.models.omnivoice_flashinfer')
    fi_module._CTX = {'wrapper': object()}
    fi_module.apply_flashinfer = apply_flashinfer
    monkeypatch.setitem(sys.modules, 'omnivoice.models.omnivoice_flashinfer', fi_module)
    monkeypatch.setattr(runtime_module, 'flashinfer_unavailable_reason', lambda settings: None)
    return fi_module


def test_flashinfer_loads_fp16_patches_and_skips_compile(tmp_path, monkeypatch) -> None:
    import torch

    install_fake_omnivoice(monkeypatch)
    patched: list = []
    fi_module = _install_fake_flashinfer(monkeypatch, lambda model, **kwargs: patched.append(model) or model)
    synthesizer = OmniVoiceSynthesizer(
        settings=make_settings(tmp_path, enable_flashinfer=True, torch_dtype='bfloat16', compile_model=True),
        store=InMemoryStore(max_queue_size=8),
    )
    compiled: list = []
    monkeypatch.setattr(synthesizer, '_compile_model', lambda model: compiled.append(model))
    asyncio.run(synthesizer.ensure_model(OMNIVOICE_AUTO_ALIAS))

    model = FakeOmniVoiceModel.instances[-1]
    assert FakeOmniVoice.load_calls[-1]['dtype'] == torch.float16
    assert patched == [model]
    assert model.calls[-1]['num_step'] == 2  # smoke-test generate before going live
    assert compiled == []
    assert synthesizer.attention_backend == 'flashinfer'

    asyncio.run(synthesizer.unload())
    assert fi_module._CTX == {'wrapper': None}


def test_flashinfer_failure_reloads_with_configured_backend(tmp_path, monkeypatch) -> None:
    import torch

    install_fake_omnivoice(monkeypatch)

    def broken(model, **kwargs):
        raise RuntimeError('missing JIT kernel')

    _install_fake_flashinfer(monkeypatch, broken)
    synthesizer = OmniVoiceSynthesizer(
        settings=make_settings(tmp_path, enable_flashinfer=True, torch_dtype='bfloat16', compile_model=False),
        store=InMemoryStore(max_queue_size=8),
    )
    asyncio.run(synthesizer.ensure_model(OMNIVOICE_AUTO_ALIAS))

    assert [call['dtype'] for call in FakeOmniVoice.load_calls] == [torch.float16, torch.bfloat16]
    assert synthesizer.attention_backend == 'sdpa'
    assert 'missing JIT kernel' in (synthesizer.attention_backend_note or '')


def test_new_generation_kwargs_pass_through_metadata(tmp_path, monkeypatch) -> None:
    install_fake_omnivoice(monkeypatch)
    synthesizer = OmniVoiceSynthesizer(settings=make_settings(tmp_path), store=InMemoryStore(max_queue_size=8))
    asyncio.run(synthesizer.ensure_model(OMNIVOICE_AUTO_ALIAS))
    monkeypatch.setattr(
        'omnivoice_tts_server.runtime_v2.importlib.util.find_spec',
        lambda name, *args: None if name == 'tn' else object(),
    )

    def item(language, **metadata):
        text = 'Es kostet 12 Euro.'
        request = SpeechRequest(
            input=text, model=OMNIVOICE_AUTO_ALIAS, task_type=TaskType.custom_voice, language=language, metadata=metadata
        )
        return BatchSynthesisItem(job_id='job_kw', sentence_index=0, text=text, request=request)

    default_kwargs = synthesizer._generation_kwargs([item('Deutsch')])
    assert not {'pad_duration', 'fade_duration', 'normalize_text'} & set(default_kwargs)

    kwargs = synthesizer._generation_kwargs([item('Deutsch', pad_duration=0, fade_duration='0.05', normalize_text=True)])
    assert kwargs['pad_duration'] == 0.0
    assert kwargs['fade_duration'] == 0.05
    assert kwargs['normalize_text'] is True
    # Auto language would send German through upstream's English normaliser, and en/zh
    # need WeTextProcessing (missing here): the flag is dropped instead of failing.
    assert 'normalize_text' not in synthesizer._generation_kwargs([item('Auto', normalize_text=True)])
    assert 'normalize_text' not in synthesizer._generation_kwargs([item('English', normalize_text=True)])


def _kw_item(language, **metadata):
    text = 'Es kostet 12 Euro.'
    # model_construct: internal requests skip the SpeechRequest validators.
    request = SpeechRequest.model_construct(
        input=text, model=OMNIVOICE_AUTO_ALIAS, task_type=TaskType.custom_voice, language=language, metadata=metadata,
        speed=1.0,
    )
    return BatchSynthesisItem(job_id='job_kw', sentence_index=0, text=text, request=request)


@pytest.mark.parametrize('language', ['de-DE', 'en-US', 'xx', 'Auto', None, 'Klingonisch'])
def test_normalize_text_is_dropped_for_languages_omnivoice_does_not_resolve(tmp_path, monkeypatch, language) -> None:
    # The live env: tn (WeTextProcessing) missing. Upstream resolves these to None and
    # would auto-detect English -> ImportError for the whole batch.
    install_fake_omnivoice(monkeypatch)
    synthesizer = OmniVoiceSynthesizer(settings=make_settings(tmp_path), store=InMemoryStore(max_queue_size=8))
    monkeypatch.setattr(
        'omnivoice_tts_server.runtime_v2.importlib.util.find_spec',
        lambda name, *args: None if name == 'tn' else object(),
    )
    assert 'normalize_text' not in synthesizer._generation_kwargs([_kw_item(language, normalize_text=True)])


def test_normalize_text_gate_matches_upstream_language_resolution(tmp_path, monkeypatch) -> None:
    pytest.importorskip('omnivoice.utils.lang_map')
    from omnivoice_tts_server.runtime_v2 import resolve_omnivoice_language

    assert resolve_omnivoice_language('de') == 'de'
    assert resolve_omnivoice_language('German') == 'de'
    assert resolve_omnivoice_language('de-DE') is None
    assert resolve_omnivoice_language('en-US') is None
    assert resolve_omnivoice_language('xx') is None
    assert resolve_omnivoice_language(None) is None

    synthesizer = OmniVoiceSynthesizer(settings=make_settings(tmp_path), store=InMemoryStore(max_queue_size=8))
    available = {'num2words'}
    monkeypatch.setattr(
        'omnivoice_tts_server.runtime_v2.importlib.util.find_spec',
        lambda name, *args: object() if name in available else None,
    )
    assert synthesizer._generation_kwargs([_kw_item('German', normalize_text=True)])['normalize_text'] is True
    assert synthesizer._generation_kwargs([_kw_item('Deutsch', normalize_text=True)])['normalize_text'] is True
    assert 'normalize_text' not in synthesizer._generation_kwargs([_kw_item('en', normalize_text=True)])
    available.add('tn')
    assert synthesizer._generation_kwargs([_kw_item('en', normalize_text=True)])['normalize_text'] is True
    # Without num2words the non-zh/en path would be a silent no-op: dropped and logged.
    available.discard('num2words')
    assert 'normalize_text' not in synthesizer._generation_kwargs([_kw_item('de', normalize_text=True)])


@pytest.mark.parametrize(
    ('metadata', 'expected'),
    [
        ({'pad_duration': float('inf')}, {}),
        ({'pad_duration': 3600, 'fade_duration': 0.05}, {'fade_duration': 0.05}),
        ({'pad_duration': 'abc'}, {}),
        ({'pad_duration': '0.5', 'fade_duration': 1}, {'pad_duration': 0.5, 'fade_duration': 1.0}),
    ],
)
def test_pad_and_fade_are_bounded_in_the_runtime_too(tmp_path, monkeypatch, metadata, expected) -> None:
    install_fake_omnivoice(monkeypatch)
    synthesizer = OmniVoiceSynthesizer(settings=make_settings(tmp_path), store=InMemoryStore(max_queue_size=8))
    kwargs = synthesizer._generation_kwargs([_kw_item('de', **metadata)])
    assert {key: kwargs[key] for key in ('pad_duration', 'fade_duration') if key in kwargs} == expected


def test_eager_mask_patch_turns_the_bool_mask_into_an_additive_mask() -> None:
    import torch

    from omnivoice_tts_server.runtime_v2 import install_eager_additive_mask_forward

    mask = _omnivoice_style_mask(c_lens=[40, 24], u_lens=[16, 10])

    class TinyModel:
        llm = types.SimpleNamespace(dtype=torch.bfloat16)

        def forward(self, input_ids=None, attention_mask=None):
            return attention_mask

    model = TinyModel()
    assert install_eager_additive_mask_forward(model) is True
    additive = model.forward(input_ids=None, attention_mask=mask)
    assert additive.dtype == torch.bfloat16 and additive.shape == mask.shape
    assert torch.all(additive[mask] == 0)
    assert torch.all(additive[~mask] == torch.finfo(torch.bfloat16).min)
    assert model.forward(input_ids=None, attention_mask=mask) is additive  # cached per mask
    assert model.forward(input_ids=None, attention_mask=None) is None

    # Eager attention (scores + mask, softmax) now equals SDPA with the bool mask.
    torch.manual_seed(0)
    rows, heads, size, dim = mask.shape[0], 2, mask.shape[-1], 16
    q, k, v = (torch.randn(rows, heads, size, dim) for _ in range(3))
    float_mask = torch.zeros(mask.shape).masked_fill(~mask, torch.finfo(torch.float32).min)
    scores = q @ k.transpose(-1, -2) / dim**0.5 + float_mask
    eager = torch.softmax(scores, dim=-1) @ v
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    valid = mask[:, 0].any(dim=-1)
    assert torch.allclose(eager.permute(0, 2, 1, 3)[valid], ref.permute(0, 2, 1, 3)[valid], atol=1e-5)
    # Unpatched (bool added as +1/+0) it does not.
    broken = torch.softmax(q @ k.transpose(-1, -2) / dim**0.5 + mask.float(), dim=-1) @ v
    assert not torch.allclose(broken.permute(0, 2, 1, 3)[valid], ref.permute(0, 2, 1, 3)[valid], atol=1e-3)


def test_eager_attention_load_installs_the_mask_patch(tmp_path, monkeypatch) -> None:
    install_fake_omnivoice(monkeypatch)
    monkeypatch.setattr(FakeOmniVoiceModel, 'forward', lambda self, **kwargs: kwargs, raising=False)
    synthesizer = OmniVoiceSynthesizer(
        settings=make_settings(tmp_path, attention_implementation='eager', compile_model=False),
        store=InMemoryStore(max_queue_size=8),
    )
    asyncio.run(synthesizer.ensure_model(OMNIVOICE_AUTO_ALIAS))

    assert FakeOmniVoice.load_calls[-1]['attn_implementation'] == 'eager'
    assert synthesizer.attention_backend == 'eager'
    assert hasattr(FakeOmniVoiceModel.instances[-1], '_g3_eager_additive_mask_cache')


def _clone_profile(voice_id: str, name: str, ref_text: str) -> VoiceProfileRecord:
    return VoiceProfileRecord(
        voice_id=voice_id,
        name=name,
        source='custom',
        created_at=utcnow(),
        audio_bytes=make_wav_bytes(),
        content_type='audio/wav',
        ref_text=ref_text,
        consent=True,
    )


@pytest.mark.parametrize(('default_voice', 'expected_ref_text'), [('SonjaV2', 'Zweite.'), ('sonjav2', 'Zweite.'), ('voice_b', 'Zweite.'), ('Missing', 'Erste.'), ('', 'Erste.')])
def test_base_warmup_uses_the_default_voice(tmp_path, monkeypatch, default_voice, expected_ref_text) -> None:
    install_fake_omnivoice(monkeypatch)
    store = InMemoryStore(max_queue_size=8)
    store.voice_profiles['voice_none'] = VoiceProfileRecord(voice_id='voice_none', name='NoAudio', source='custom', created_at=utcnow())
    store.voice_profiles['voice_a'] = _clone_profile('voice_a', 'Sonja', 'Erste.')
    store.voice_profiles['voice_b'] = _clone_profile('voice_b', 'SonjaV2', 'Zweite.')
    synthesizer = OmniVoiceSynthesizer(settings=make_settings(tmp_path, default_voice=default_voice), store=store)
    asyncio.run(synthesizer.ensure_model(OMNIVOICE_BASE_ALIAS))

    synthesizer._run_warmup_inference(OMNIVOICE_BASE_ALIAS, None)

    # Without a usable default voice the first profile that can clone is warmed, as before.
    call = FakeOmniVoiceModel.instances[-1].calls[-1]
    assert call['voice_clone_prompt'][0]['ref_text'] == expected_ref_text


def test_clone_requests_of_different_voices_share_a_batch(tmp_path) -> None:
    settings = make_settings(tmp_path)
    store = InMemoryStore(max_queue_size=8)
    from omnivoice_tts_server.services_v2 import EventHub, QueueService
    from omnivoice_tts_server.runtime_v2 import MockSynthesizer

    queue = QueueService(store, MockSynthesizer(), EventHub(store), settings)
    base = dict(input='A', model=OMNIVOICE_BASE_ALIAS, task_type=TaskType.base, language='de', seed=9305)
    voice_a = queue._group_key_for_request(SpeechRequest(**base, voice='SonjaV2'))
    voice_b = queue._group_key_for_request(SpeechRequest(**base, voice='Sonja'))
    other_seed = queue._group_key_for_request(SpeechRequest(**{**base, 'seed': 12}, voice='SonjaV2'))
    other_language = queue._group_key_for_request(SpeechRequest(**{**base, 'language': 'en'}, voice='SonjaV2'))

    assert voice_a == voice_b
    # Seeds no longer split groups: a batch with mixed seeds draws per item (row_seeds).
    assert voice_a == other_seed
    assert voice_a != other_language

