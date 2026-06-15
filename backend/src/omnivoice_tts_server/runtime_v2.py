from __future__ import annotations

import asyncio
import functools
import gc
import hashlib
import importlib.util
import io
import logging
import os
import subprocess
import time
import wave
from dataclasses import dataclass
from typing import Any

from .config import Settings
from .domain.models import SpeechRequest, TaskType
from .domain.state import InMemoryStore, VoiceProfileRecord
from .voice_design import DEFAULT_VOICE_DESIGN_INSTRUCT, normalize_voice_design_instruct

OMNIVOICE_MODEL_ID = 'k2-fsa/OmniVoice'
OMNIVOICE_AUTO_ALIAS = 'k2-fsa/OmniVoice-AutoVoice'
OMNIVOICE_DESIGN_ALIAS = 'k2-fsa/OmniVoice-VoiceDesign'
OMNIVOICE_BASE_ALIAS = 'k2-fsa/OmniVoice-Base'

# Maps the curated UI language labels to ISO codes that OmniVoice.generate()
# resolves. OmniVoice only recognizes English names ("German") or ISO codes
# ("de"); the localized labels shown in the UI ("Deutsch") are otherwise
# unknown to the model and silently fall back to language-agnostic mode.
# 'Auto', empty, and unknown values map to None so the model auto-detects.
LANGUAGE_LABEL_TO_CODE: dict[str, str] = {
    'deutsch': 'de',
    'english': 'en',
    'français': 'fr',
    'español': 'es',
    'italiano': 'it',
    'nederlands': 'nl',
    'polski': 'pl',
    'português': 'pt',
    'türkçe': 'tr',
    'русский': 'ru',
    'українська': 'uk',
    '中文': 'zh',
    '日本語': 'ja',
    '한국어': 'ko',
}

logger = logging.getLogger('omnivoice_tts_server.runtime')


def _disable_windows_deepgemm_hub_kernel(
    hub_kernels_module: Any | None = None,
    finegrained_fp8_module: Any | None = None,
) -> bool:
    """Avoid Transformers' DeepGEMM Hub lookup on Windows while keeping fp8 fallback active."""
    if os.name != 'nt':
        return False

    changed = False
    try:
        if hub_kernels_module is None:
            from transformers.integrations import hub_kernels as hub_kernels_module

        hub_mapping = getattr(hub_kernels_module, '_HUB_KERNEL_MAPPING', None)
        if isinstance(hub_mapping, dict) and 'deep-gemm' in hub_mapping:
            hub_mapping.pop('deep-gemm', None)
            changed = True

        module_mapping = getattr(hub_kernels_module, '_KERNEL_MODULE_MAPPING', None)
        if isinstance(module_mapping, dict) and module_mapping.get('deep-gemm') is not None:
            module_mapping['deep-gemm'] = None
            changed = True
    except Exception as exc:
        logger.debug('could not disable Transformers DeepGEMM hub mapping: %s', exc)

    try:
        if finegrained_fp8_module is None:
            from transformers.integrations import finegrained_fp8 as finegrained_fp8_module

        loader = getattr(finegrained_fp8_module, '_load_deepgemm_kernel', None)
        if getattr(loader, '_omnivoice_windows_disabled', False):
            return changed

        @functools.cache
        def _disabled_deepgemm_kernel() -> Any:
            raise ImportError(
                'DeepGEMM hub kernel is disabled on Windows; using Triton finegrained-fp8 fallback.'
            )

        setattr(_disabled_deepgemm_kernel, '_omnivoice_windows_disabled', True)
        finegrained_fp8_module._load_deepgemm_kernel = _disabled_deepgemm_kernel
        changed = True
    except Exception as exc:
        logger.debug('could not disable Transformers DeepGEMM loader: %s', exc)

    return changed


def normalize_runtime_device(value: str | None) -> str:
    device = (value or 'cuda:0').strip().lower()
    if device == 'cuda':
        return 'cuda:0'
    if device == 'cpu':
        return 'cpu'
    if device.startswith('cuda:'):
        raw_index = device.split(':', maxsplit=1)[1]
        if not raw_index.isdigit():
            raise ValueError('CUDA device must look like cuda:0, cuda:1, ...')
        index = int(raw_index)
        if index < 0:
            raise ValueError('CUDA device index must be >= 0')
        return f'cuda:{index}'
    raise ValueError('Unsupported runtime device. Use cpu or cuda:<index>.')


def _cuda_index_for_device(device: str | None) -> int | None:
    normalized = normalize_runtime_device(device)
    if not normalized.startswith('cuda:'):
        return None
    return int(normalized.split(':', maxsplit=1)[1])


def _parse_nvidia_smi_int(value: str) -> int:
    value = value.strip()
    if not value or value.upper() in {'N/A', '[N/A]'}:
        return 0
    digits = ''.join(char for char in value if char.isdigit() or char == '-')
    if not digits or digits == '-':
        return 0
    return int(digits)


def _unavailable_gpu_stats() -> dict[str, int | str | None]:
    return {
        'name': 'Unavailable',
        'memory_used_mb': 0,
        'memory_total_mb': 0,
        'utilization_percent': 0,
        'temperature_c': None,
    }


@dataclass
class BatchSynthesisItem:
    job_id: str
    sentence_index: int
    request: SpeechRequest
    text: str


@dataclass
class BatchSynthesisResult:
    job_id: str
    sentence_index: int
    sample_rate: int
    pcm: bytes
    duration_ms: int
    sentence_finished: bool = False


class OmniVoiceSynthesizer:
    def __init__(self, settings: Settings, store: InMemoryStore) -> None:
        self.settings = settings
        self.store = store
        self.sample_rate = settings.sample_rate
        self._loaded_model_id: str | None = None
        self._loaded_model_source: str | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._soundfile: Any | None = None
        self._numpy: Any | None = None

    def duration_ms(self, text: str) -> int:
        return max(500, 240 + len(text) * 38)

    def pcm_to_wav(self, pcm: bytes, *, sample_rate: int | None = None) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate or self.sample_rate)
            wav_file.writeframes(pcm)
        return buffer.getvalue()

    async def ensure_model(self, requested_model: str | None) -> tuple[str, int]:
        target_model = requested_model or self.settings.active_model or OMNIVOICE_AUTO_ALIAS
        if self._loaded_model_id == target_model and self._model is not None:
            return target_model, 0
        # The OmniVoice aliases (AutoVoice/VoiceDesign/Base) all resolve to ONE
        # checkpoint; only the task type differs at generate() time. If the requested
        # alias maps to the already-resident checkpoint, just re-point the alias
        # instead of evicting and reloading identical weights.
        if self._model is not None and self._loaded_model_source is not None:
            try:
                resolved_source, _ = self._resolve_model_source(target_model)
            except Exception:
                resolved_source = None
            if resolved_source is not None and resolved_source == self._loaded_model_source:
                self._loaded_model_id = target_model
                self.settings.active_model = target_model
                return target_model, 0
        warm_ms = await asyncio.to_thread(self._load_model_sync, target_model)
        return target_model, warm_ms

    async def preload(self, requested_model: str | None = None) -> tuple[str, int]:
        return await self.ensure_model(requested_model)

    async def warmup(self, requested_model: str | None = None, request: SpeechRequest | None = None) -> tuple[str, int]:
        model_id, warm_ms = await self.ensure_model(requested_model)
        await asyncio.to_thread(self._run_warmup_inference, model_id, request)
        return model_id, warm_ms

    async def unload(self) -> tuple[str, int]:
        model_id = self._loaded_model_id or self.settings.active_model
        await asyncio.to_thread(self._release_model)
        return model_id, 0

    async def reload(self, requested_model: str | None = None) -> tuple[str, int]:
        await self.unload()
        return await self.ensure_model(requested_model)

    async def free_memory(self) -> dict[str, int | str | bool | None]:
        return await asyncio.to_thread(self._free_memory_sync)

    async def render_batch(self, items: list[BatchSynthesisItem]) -> list[BatchSynthesisResult]:
        if not items:
            return []
        return await asyncio.to_thread(self._generate_batch_sync, items)

    def _load_model_sync(self, model_id: str) -> int:
        start = time.perf_counter()
        torch, omnivoice_cls = self._load_runtime_dependencies()
        model_source, extra_kwargs = self._resolve_model_source(model_id)
        if self._model is not None and self._loaded_model_id != model_id:
            self._release_model()

        kwargs = {
            'device_map': 'auto' if self.settings.enable_cpu_offload else self.settings.preferred_device,
            'dtype': self._torch_dtype(torch),
            'attn_implementation': self.settings.attention_implementation,
            'load_asr': False,
        }
        kwargs.update(extra_kwargs)

        model = None
        quant_config = self._fp8_quant_config_if_requested()
        if quant_config is not None:
            try:
                logger.info('loading model=%s with EXPERIMENTAL fp8 quantization', model_id)
                model = omnivoice_cls.from_pretrained(model_source, quantization_config=quant_config, **kwargs)
                logger.info('fp8 quantized load succeeded for model=%s', model_id)
            except Exception as exc:
                # Never let an experimental fp8 load brick startup: fall back to the dtype below.
                logger.warning('fp8 quantized load failed (%s); falling back to dtype=%s', exc, self.settings.torch_dtype)
                self._trim_cuda_cache_sync()
                model = None

        if model is None:
            try:
                model = omnivoice_cls.from_pretrained(model_source, **kwargs)
            except TypeError:
                relaxed = dict(kwargs)
                relaxed.pop('attn_implementation', None)
                model = omnivoice_cls.from_pretrained(model_source, **relaxed)
            except Exception as exc:
                if self.settings.attention_implementation == 'sdpa':
                    raise RuntimeError(f'Failed to load OmniVoice model {model_id}: {exc}') from exc
                fallback = dict(kwargs)
                fallback['attn_implementation'] = 'sdpa'
                try:
                    model = omnivoice_cls.from_pretrained(model_source, **fallback)
                except Exception:
                    raise RuntimeError(f'Failed to load OmniVoice model {model_id}: {exc}') from exc

        if getattr(self.settings, 'compile_model', False):
            self._compile_model(model)

        self._model = model
        self._loaded_model_id = model_id
        self._loaded_model_source = model_source
        self.settings.active_model = model_id
        self.sample_rate = int(getattr(model, 'sampling_rate', self.settings.sample_rate) or self.settings.sample_rate)

        if getattr(self.settings, 'warmup_on_startup', True):
            self._run_warmup_inference(model_id, None)
            # Warmup bypasses the per-batch trim, so its reserved-pool spike would stay
            # pinned right after startup (the ~9GB-at-load the operator sees). Release it
            # now so idle VRAM is low immediately, not only after the first real job.
            self._trim_cuda_cache_sync()

        return int((time.perf_counter() - start) * 1000)

    def _compile_model(self, model: Any) -> None:
        if self._torch is None:
            return
        if importlib.util.find_spec('triton') is None:
            logger.warning('torch.compile skipped because Triton is not installed in this environment.')
            self.settings.compile_model = False
            return
        try:
            import torch._dynamo
            import torch._inductor.config as inductor_config

            torch._dynamo.config.suppress_errors = True
            use_cudagraphs = bool(getattr(self.settings, 'compile_cudagraphs', False))
            triton_config = getattr(inductor_config, 'triton', None)
            if triton_config is not None:
                if hasattr(triton_config, 'cudagraphs'):
                    triton_config.cudagraphs = use_cudagraphs
                if hasattr(triton_config, 'cudagraph_skip_dynamic_graphs'):
                    triton_config.cudagraph_skip_dynamic_graphs = bool(self.settings.cudagraph_skip_dynamic_graphs)
                if bool(self.settings.cudagraph_skip_dynamic_graphs) and hasattr(
                    triton_config, 'cudagraph_dynamic_shape_warn_limit'
                ):
                    triton_config.cudagraph_dynamic_shape_warn_limit = None

            def compile_target(target: Any) -> Any:
                if use_cudagraphs:
                    return self._torch.compile(target, mode='reduce-overhead')
                # dynamic=True compiles a single shape-generic graph so that varying
                # batch sizes (1..max_batch_size sentences) do NOT each trigger a fresh
                # 10-60s Inductor recompilation. Without this, every new batch shape
                # stalls the single worker and starves the GPU (observed: cold batch
                # sizes took 10-60s at <10% GPU; warm sizes ran at 95% GPU in ~250ms).
                return self._torch.compile(target, dynamic=True)

            if hasattr(model, 'llm'):
                model.llm = compile_target(model.llm)
            elif hasattr(model, 'model'):
                model.model = compile_target(model.model)
            logger.info('torch.compile enabled for OmniVoice model (cudagraphs=%s).', 'on' if use_cudagraphs else 'off')
        except Exception as exc:
            logger.warning('torch.compile failed and was disabled: %s', exc)
            self.settings.compile_model = False

    def _run_warmup_inference(self, model_id: str, request: SpeechRequest | None) -> None:
        if self._model is None:
            return
        logger.info('warmup model_id=%s starting...', model_id)
        try:
            task_type = request.task_type if request and request.task_type else self._task_type_from_model(model_id)
            # A representative ~20-word sentence warms the compiled kernels for a realistic
            # input shape (better first-request latency than a 1-word 'Warmup.').
            text = (
                request.input
                if request and request.input
                else 'Dies ist ein Aufwaermsatz fuer das Sprachmodell mit etwa zwanzig Woertern, damit die Kernel fuer typische Eingaben vorbereitet sind.'
            )
            language = self._normalize_language(request.language if request and request.language else None)
            if task_type == TaskType.voice_design:
                self._model.generate(
                    text=[text],
                    language=[language],
                    instruct=[
                        normalize_voice_design_instruct(request.instructions if request else None)
                    ],
                )
            elif task_type == TaskType.base:
                profile = next((item for item in self.store.voice_profiles.values() if item.audio_bytes and item.ref_text), None)
                if profile is None:
                    logger.info('warmup skipped for Base model (requires saved voice profile with ref_text)')
                    return
                self._model.generate(
                    text=[text],
                    language=[language],
                    voice_clone_prompt=[self._clone_prompt_from_profile(profile, ref_text=profile.ref_text)],
                )
            else:
                self._model.generate(text=[text], language=[language])
            logger.info('warmup model_id=%s done', model_id)
        except Exception as exc:
            logger.warning('warmup failed (non-critical): %s', exc)

    def _load_runtime_dependencies(self) -> tuple[Any, Any]:
        try:
            import numpy
            import soundfile
            import torch
            from omnivoice import OmniVoice
        except Exception as exc:
            raise RuntimeError(
                'OmniVoice runtime dependencies are missing or incompatible. '
                f'Import failed with {type(exc).__name__}: {exc}. '
                'Install the pinned Windows runtime versions from docs/RUNTIME_VERSIONS.md.'
            ) from exc

        preferred_device = normalize_runtime_device(self.settings.preferred_device)
        self.settings.preferred_device = preferred_device
        cuda_index = _cuda_index_for_device(preferred_device)

        if cuda_index is not None and not torch.cuda.is_available():
            raise RuntimeError('No CUDA-capable NVIDIA GPU is available for the configured runtime.')

        if cuda_index is not None:
            device_count = torch.cuda.device_count()
            if cuda_index >= device_count:
                raise RuntimeError(
                    f'Configured CUDA device {preferred_device} does not exist. '
                    f'Available CUDA device count: {device_count}.'
                )
            torch.cuda.set_device(cuda_index)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision('high')

        self._torch = torch
        self._soundfile = soundfile
        self._numpy = numpy
        return torch, OmniVoice

    def _resolve_model_source(self, model_id: str) -> tuple[str, dict[str, Any]]:
        self.settings.models_root_dir.mkdir(parents=True, exist_ok=True)
        # Custom finetuned checkpoints are addressed as 'local/<dirname>' and live under
        # models_root_dir/<dirname> (a full HF-format dir written by the omnivoice trainer's
        # save_pretrained). Resolving to that dir forces ensure_model() to reload real weights
        # instead of reusing the resident base checkpoint.
        if model_id.startswith('local/'):
            custom_dir = self.settings.models_root_dir / model_id[len('local/'):]
            if (custom_dir / 'config.json').exists():
                return str(custom_dir), {}
            raise RuntimeError(
                f'Custom model {model_id} not found at {custom_dir} (missing config.json).'
            )
        local_dir = self.settings.models_root_dir / 'OmniVoice'
        if local_dir.exists():
            return str(local_dir), {}
        if not self.settings.allow_model_downloads:
            raise RuntimeError(
                f'OmniVoice model not found under {local_dir}. Enable downloads or place the model there first.'
            )
        os.environ.setdefault('HF_HOME', str(self.settings.models_root_dir / '.hf'))
        return OMNIVOICE_MODEL_ID, {'cache_dir': str(self.settings.models_root_dir)}

    def _torch_dtype(self, torch: Any) -> Any:
        bf16 = getattr(torch, 'bfloat16', torch.float16)
        mapping = {
            'float16': torch.float16,
            'bfloat16': bf16,
            'float32': torch.float32,
            'fp8': bf16,  # fp8 keeps bf16 as the compute dtype; weights quantized via quant config
        }
        return mapping.get(self.settings.torch_dtype.lower(), torch.float16)

    def _fp8_quant_config_if_requested(self) -> Any | None:
        """Experimental: when torch_dtype == 'fp8', quantize the model at load time via
        transformers FineGrainedFP8Config (RTX 50xx has native fp8 tensor cores); compute
        dtype stays bf16. Returns None unless fp8 is selected / the config class is missing.
        Audio quality MUST be verified -- fp8 on the diffusion head can degrade output."""
        if (self.settings.torch_dtype or '').lower() != 'fp8':
            return None
        # finegrained-fp8 inference needs HF `kernels` at generate() time. Without it the
        # model loads but every synthesis raises ImportError -- so only engage fp8 when
        # kernels is importable, otherwise fall back to the bf16 compute dtype.
        try:
            import kernels  # noqa: F401
        except Exception:
            logger.warning(
                'fp8 requested but the `kernels` package is missing -> using bf16. '
                'Install it (pip install -U kernels) to enable fp8; verify it does not '
                'break model loading on this torch/transformers/GPU stack first.'
            )
            return None
        if _disable_windows_deepgemm_hub_kernel():
            logger.info('DeepGEMM hub kernel disabled on Windows; using Triton finegrained-fp8 fallback.')
        try:
            from transformers import FineGrainedFP8Config

            return FineGrainedFP8Config()
        except Exception as exc:
            logger.warning('fp8 requested but FineGrainedFP8Config is unavailable: %s', exc)
            return None

    def _release_model(self) -> None:
        self._model = None
        self._loaded_model_id = None
        self._loaded_model_source = None
        self.store.prompt_cache.clear()
        self._trim_cuda_cache_sync()

    def resident_models(self) -> set[str]:
        """Alias model_ids served by the currently resident checkpoint.

        All OmniVoice aliases (AutoVoice/VoiceDesign/Base) share a single checkpoint,
        so when any one is loaded they are all effectively resident -- only the task
        type differs per request. Used so a task-type switch is not reported as a
        cold 'warming' model reload."""
        if self._model is None:
            return set()
        return set(self.settings.supported_models)

    def _free_memory_sync(self) -> dict[str, int | str | bool | None]:
        before = query_nvidia_smi(self.settings.preferred_device)
        before_used = int(before.get('memory_used_mb') or 0)
        self.store.prompt_cache.clear()
        self._trim_cuda_cache_sync(reset_compile_cache=True)
        after = query_nvidia_smi(self.settings.preferred_device)
        after_used = int(after.get('memory_used_mb') or 0)
        return {
            'ok': True,
            'message': 'CUDA and compile caches trimmed; loaded model remains in memory.',
            'memory_before_mb': before_used,
            'memory_after_mb': after_used,
            'memory_total_mb': int(after.get('memory_total_mb') or before.get('memory_total_mb') or 0),
            'released_mb': max(0, before_used - after_used),
        }

    def _trim_cuda_cache_sync(self, *, reset_compile_cache: bool = False) -> None:
        gc.collect()
        if self._torch is None:
            return
        if reset_compile_cache:
            compiler = getattr(self._torch, 'compiler', None)
            compiler_reset = getattr(compiler, 'reset', None)
            if callable(compiler_reset):
                try:
                    compiler_reset()
                except Exception:
                    pass
            else:
                try:
                    import torch._dynamo

                    torch._dynamo.reset()
                except Exception:
                    pass
        try:
            cuda_index = _cuda_index_for_device(self.settings.preferred_device)
        except ValueError:
            return
        if cuda_index is None:
            return
        if not self._torch.cuda.is_available():
            return
        try:
            if cuda_index < self._torch.cuda.device_count():
                self._torch.cuda.set_device(cuda_index)
        except Exception:
            pass
        for cuda_call in (
            self._torch.cuda.synchronize,
            self._torch.cuda.empty_cache,
            self._torch.cuda.ipc_collect,
            self._torch.cuda.reset_peak_memory_stats,
        ):
            try:
                cuda_call()
            except Exception:
                pass

    def _generate_batch_sync(self, items: list[BatchSynthesisItem]) -> list[BatchSynthesisResult]:
        if self._model is None:
            raise RuntimeError('No model loaded. Call ensure_model before generation.')

        texts = [item.text.strip() for item in items]
        if any(not text for text in texts):
            raise RuntimeError('Missing input text')

        task_type = items[0].request.task_type or self._task_type_from_model(self._loaded_model_id or self.settings.active_model)
        languages = [self._normalize_language(item.request.language) for item in items]
        kwargs = self._generation_kwargs(items)

        try:
            with self._torch.inference_mode():
                self._apply_seed(items[0].request.seed)
                if task_type == TaskType.voice_design:
                    wavs = self._model.generate(
                        text=texts,
                        language=languages,
                        instruct=[normalize_voice_design_instruct(item.request.instructions) for item in items],
                        **kwargs,
                    )
                elif task_type == TaskType.base:
                    prompts = [self._clone_prompt(item.request) for item in items]
                    wavs = self._model.generate(
                        text=texts,
                        language=languages,
                        voice_clone_prompt=prompts,
                        **kwargs,
                    )
                else:
                    instructs = [item.request.instructions or '' for item in items]
                    generate_kwargs = dict(kwargs)
                    if any(instructs):
                        generate_kwargs['instruct'] = instructs
                    wavs = self._model.generate(text=texts, language=languages, **generate_kwargs)
        except Exception:
            self._trim_cuda_cache_sync(reset_compile_cache=True)
            raise

        sample_rate = int(getattr(self._model, 'sampling_rate', self.sample_rate) or self.sample_rate)
        self.sample_rate = sample_rate
        results: list[BatchSynthesisResult] = []
        for item, wav in zip(items, list(wavs), strict=False):
            pcm = self._audio_array_to_pcm_bytes(wav)
            results.append(
                BatchSynthesisResult(
                    job_id=item.job_id,
                    sentence_index=item.sentence_index,
                    sample_rate=sample_rate,
                    pcm=pcm,
                    duration_ms=int(round(len(pcm) / 2 / max(sample_rate, 1) * 1000)),
                )
            )
        del wavs
        if self.settings.cuda_memory_trim_after_batch:
            self._trim_cuda_cache_sync()
        return results

    def _generation_kwargs(self, items: list[BatchSynthesisItem]) -> dict[str, Any]:
        request = items[0].request
        metadata = request.metadata or {}
        candidates = {
            'num_step': self.settings.num_step,
            'guidance_scale': self.settings.guidance_scale,
            'duration': self.settings.duration,
            't_shift': self.settings.t_shift,
            'denoise': self.settings.denoise,
            'preprocess_prompt': self.settings.preprocess_prompt,
            'postprocess_output': self.settings.postprocess_output,
            'audio_chunk_duration': self.settings.audio_chunk_duration,
            'audio_chunk_threshold': self.settings.audio_chunk_threshold,
            'position_temperature': self.settings.position_temperature,
            'class_temperature': self.settings.class_temperature,
        }
        result = {key: metadata.get(key, value) for key, value in candidates.items() if metadata.get(key, value) is not None}
        speed = metadata.get('speed', request.speed)
        if speed and float(speed) != 1.0:
            result['speed'] = speed
        return result

    def _apply_seed(self, seed: int | None) -> None:
        if seed is None or self._torch is None:
            return
        normalized_seed = int(seed) % (2**31)
        self._torch.manual_seed(normalized_seed)
        if self._torch.cuda.is_available():
            self._torch.cuda.manual_seed_all(normalized_seed)

    def _task_type_from_model(self, model_id: str | None) -> TaskType:
        if not model_id:
            return TaskType.custom_voice
        if model_id.endswith('VoiceDesign'):
            return TaskType.voice_design
        if model_id.endswith('Base'):
            return TaskType.base
        return TaskType.custom_voice

    @staticmethod
    def _normalize_language(language: str | None) -> str | None:
        value = (language or '').strip()
        if not value or value.lower() == 'auto':
            return None
        # Translate curated UI labels (e.g. "Deutsch") to codes the model
        # resolves; pass anything else through for OmniVoice to resolve itself.
        return LANGUAGE_LABEL_TO_CODE.get(value.lower(), value)

    def _clone_prompt(self, request: SpeechRequest) -> Any:
        voice_profile = self._resolve_voice_profile(request.voice)
        if voice_profile and voice_profile.audio_bytes:
            ref_text = (request.ref_text or voice_profile.ref_text or '').strip()
            if not ref_text:
                raise RuntimeError('OmniVoice cloning requires ref_text for saved voice profiles.')
            return self._clone_prompt_from_profile(voice_profile, ref_text=ref_text)

        if request.ref_audio:
            ref_text = (request.ref_text or '').strip()
            if not ref_text:
                raise RuntimeError('OmniVoice cloning requires ref_text with ref_audio.')
            key = hashlib.sha1(
                '|'.join(
                    [self._loaded_model_id or '', request.ref_audio[:256], ref_text, str(self._preprocess_prompt_enabled())]
                ).encode('utf-8')
            ).hexdigest()
            cached = self.store.prompt_cache.get(key)
            if cached is not None:
                return cached
            if hasattr(self._model, 'create_voice_clone_prompt'):
                prompt = self._model.create_voice_clone_prompt(
                    ref_audio=request.ref_audio,
                    ref_text=ref_text,
                    preprocess_prompt=self._preprocess_prompt_enabled(),
                )
                if isinstance(prompt, list):
                    prompt = prompt[0]
                self.store.prompt_cache[key] = prompt
                return prompt
            return {'ref_audio': request.ref_audio, 'ref_text': ref_text}

        raise RuntimeError('Base voice cloning requires a saved custom voice profile or ref_audio + ref_text.')

    def _clone_prompt_from_profile(self, profile: VoiceProfileRecord, *, ref_text: str) -> Any:
        key = self._voice_prompt_cache_key(profile=profile, model_id=self._loaded_model_id or '', ref_text=ref_text)
        cached = self.store.prompt_cache.get(key)
        if cached is not None:
            return cached
        audio = self._audio_bytes_to_prompt_audio(profile)
        if hasattr(self._model, 'create_voice_clone_prompt'):
            prompt = self._model.create_voice_clone_prompt(
                ref_audio=audio,
                ref_text=ref_text,
                preprocess_prompt=self._preprocess_prompt_enabled(),
            )
            if isinstance(prompt, list):
                prompt = prompt[0]
        else:
            prompt = {'ref_audio': audio, 'ref_text': ref_text}
        self.store.prompt_cache[key] = prompt
        return prompt

    def _voice_prompt_cache_key(self, *, profile: VoiceProfileRecord, model_id: str, ref_text: str) -> str:
        fingerprint = hashlib.sha1(profile.audio_bytes or b'').hexdigest()
        return '|'.join([model_id, profile.voice_id, profile.name, fingerprint, ref_text, str(self._preprocess_prompt_enabled())])

    def _preprocess_prompt_enabled(self) -> bool:
        return True if self.settings.preprocess_prompt is None else bool(self.settings.preprocess_prompt)

    def _resolve_voice_profile(self, voice_name: str | None) -> VoiceProfileRecord | None:
        if not voice_name:
            return None
        for profile in self.store.voice_profiles.values():
            if profile.voice_id == voice_name or profile.name == voice_name:
                return profile
        return None

    def _audio_bytes_to_prompt_audio(self, profile: VoiceProfileRecord) -> tuple[Any, int]:
        if self._soundfile is None:
            self._load_runtime_dependencies()
        buffer = io.BytesIO(profile.audio_bytes or b'')
        audio, sample_rate = self._soundfile.read(buffer, dtype='float32', always_2d=False)
        if self._torch is not None:
            audio = self._torch.as_tensor(audio, dtype=self._torch.float32)
        return audio, int(sample_rate)

    def _audio_array_to_pcm_bytes(self, audio: Any) -> bytes:
        if self._numpy is None:
            self._load_runtime_dependencies()
        array = audio
        if hasattr(array, 'detach'):
            array = array.detach().float().cpu().numpy()
        else:
            array = self._numpy.asarray(array)
        if array.ndim > 1:
            array = array.reshape(-1)
        array = self._numpy.clip(array.astype('float32'), -1.0, 1.0)
        return (array * 32767.0).astype(self._numpy.int16).tobytes()


class MockSynthesizer:
    def __init__(self, sample_rate: int = 24_000) -> None:
        self.sample_rate = sample_rate

    async def ensure_model(self, requested_model: str | None) -> tuple[str, int]:
        return requested_model or OMNIVOICE_AUTO_ALIAS, 0

    def resident_models(self) -> set[str]:
        return {OMNIVOICE_AUTO_ALIAS, OMNIVOICE_DESIGN_ALIAS, OMNIVOICE_BASE_ALIAS}

    async def preload(self, requested_model: str | None = None) -> tuple[str, int]:
        return await self.ensure_model(requested_model)

    async def warmup(self, requested_model: str | None = None, request: SpeechRequest | None = None) -> tuple[str, int]:
        return await self.ensure_model(requested_model or (request.model if request else None))

    async def unload(self) -> tuple[str, int]:
        return OMNIVOICE_AUTO_ALIAS, 0

    async def reload(self, requested_model: str | None = None) -> tuple[str, int]:
        return await self.ensure_model(requested_model)

    async def free_memory(self) -> dict[str, int | str | bool | None]:
        return {
            'ok': True,
            'message': 'Mock runtime memory cleanup completed.',
            'memory_before_mb': 0,
            'memory_after_mb': 0,
            'memory_total_mb': 0,
            'released_mb': 0,
        }

    def duration_ms(self, text: str) -> int:
        return max(650, min(4500, 260 + len(text) * 34))

    def frequency_for(self, text: str) -> int:
        digest = hashlib.sha256(text.encode('utf-8')).digest()
        return 170 + digest[0] % 220

    def pcm_to_wav(self, pcm: bytes, *, sample_rate: int | None = None) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate or self.sample_rate)
            wav_file.writeframes(pcm)
        return buffer.getvalue()

    async def render_batch(self, items: list[BatchSynthesisItem]) -> list[BatchSynthesisResult]:
        results: list[BatchSynthesisResult] = []
        for item in items:
            text = item.text or ''
            duration_ms = self.duration_ms(text)
            total_samples = int(self.sample_rate * duration_ms / 1000)
            freq = self.frequency_for(f'{item.request.task_type}|{item.request.voice}|{text}')
            amplitude = 10_000
            frames = bytearray()
            for sample_index in range(total_samples):
                value = int(amplitude * __import__('math').sin(2 * __import__('math').pi * freq * sample_index / self.sample_rate))
                frames.extend(int(value).to_bytes(2, byteorder='little', signed=True))
            results.append(
                BatchSynthesisResult(
                    job_id=item.job_id,
                    sentence_index=item.sentence_index,
                    sample_rate=self.sample_rate,
                    pcm=bytes(frames),
                    duration_ms=duration_ms,
                )
            )
        return results


def build_synthesizer(settings: Settings, store: InMemoryStore) -> Any:
    if settings.runtime_backend.lower() == 'mock':
        return MockSynthesizer(sample_rate=settings.sample_rate)
    if settings.runtime_backend.lower() != 'omnivoice':
        raise RuntimeError(f'Unsupported runtime backend: {settings.runtime_backend}')
    return OmniVoiceSynthesizer(settings=settings, store=store)


def query_nvidia_smi_all() -> list[dict[str, int | str | None]]:
    try:
        result = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu',
                '--format=csv,noheader,nounits',
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=3,
        )
    except Exception:
        return []

    gpus: list[dict[str, int | str | None]] = []
    for line in (entry.strip() for entry in result.stdout.splitlines() if entry.strip()):
        parts = [part.strip() for part in line.split(',', maxsplit=5)]
        if len(parts) != 6:
            continue
        index, name, memory_used, memory_total, utilization, temperature = parts
        gpus.append(
            {
                'index': _parse_nvidia_smi_int(index),
                'name': name,
                'memory_used_mb': _parse_nvidia_smi_int(memory_used),
                'memory_total_mb': _parse_nvidia_smi_int(memory_total),
                'utilization_percent': _parse_nvidia_smi_int(utilization),
                'temperature_c': _parse_nvidia_smi_int(temperature) if temperature else None,
            }
        )
    return gpus


def query_nvidia_smi(preferred_device: str | None = None) -> dict[str, int | str | None]:
    gpus = query_nvidia_smi_all()
    if not gpus:
        return _unavailable_gpu_stats()

    try:
        cuda_index = _cuda_index_for_device(preferred_device)
    except ValueError:
        cuda_index = None

    selected = None
    if cuda_index is not None:
        selected = next((gpu for gpu in gpus if gpu.get('index') == cuda_index), None)
        if selected is None and 0 <= cuda_index < len(gpus):
            selected = gpus[cuda_index]
    selected = selected or gpus[0]
    return {
        'name': str(selected.get('name') or 'Unavailable'),
        'memory_used_mb': int(selected.get('memory_used_mb') or 0),
        'memory_total_mb': int(selected.get('memory_total_mb') or 0),
        'utilization_percent': int(selected.get('utilization_percent') or 0),
        'temperature_c': selected.get('temperature_c') if selected.get('temperature_c') is not None else None,
    }


def query_runtime_devices(preferred_device: str | None = None) -> list[dict[str, Any]]:
    selected_device = normalize_runtime_device(preferred_device)
    devices: list[dict[str, Any]] = [
        {
            'id': 'cpu',
            'label': 'CPU',
            'kind': 'cpu',
            'name': 'CPU',
            'index': None,
            'memory_total_mb': None,
            'available': True,
            'selected': selected_device == 'cpu',
        }
    ]

    cuda_devices: list[dict[str, Any]] = []
    try:
        import torch

        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(index)
                properties = torch.cuda.get_device_properties(index)
                memory_total_mb = int(getattr(properties, 'total_memory', 0) // (1024 * 1024))
                label = f'cuda:{index} - {name}'
                if memory_total_mb:
                    label += f' ({memory_total_mb // 1024} GB)'
                cuda_devices.append(
                    {
                        'id': f'cuda:{index}',
                        'label': label,
                        'kind': 'cuda',
                        'name': name,
                        'index': index,
                        'memory_total_mb': memory_total_mb,
                        'available': True,
                        'selected': selected_device == f'cuda:{index}',
                    }
                )
    except Exception:
        cuda_devices = []

    if not cuda_devices:
        for gpu in query_nvidia_smi_all():
            index = int(gpu.get('index') or 0)
            name = str(gpu.get('name') or f'NVIDIA GPU {index}')
            memory_total_mb = int(gpu.get('memory_total_mb') or 0)
            label = f'cuda:{index} - {name}'
            if memory_total_mb:
                label += f' ({memory_total_mb // 1024} GB)'
            cuda_devices.append(
                {
                    'id': f'cuda:{index}',
                    'label': label,
                    'kind': 'cuda',
                    'name': name,
                    'index': index,
                    'memory_total_mb': memory_total_mb,
                    'available': True,
                    'selected': selected_device == f'cuda:{index}',
                }
            )

    devices.extend(cuda_devices)
    if selected_device not in {device['id'] for device in devices}:
        devices.append(
            {
                'id': selected_device,
                'label': f'{selected_device} - configured, unavailable',
                'kind': 'cuda' if selected_device.startswith('cuda:') else 'other',
                'name': 'Configured device',
                'index': _cuda_index_for_device(selected_device) if selected_device.startswith('cuda:') else None,
                'memory_total_mb': None,
                'available': False,
                'selected': True,
            }
        )
    return devices
