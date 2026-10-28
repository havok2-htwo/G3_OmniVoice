from __future__ import annotations

import asyncio
import contextlib
import functools
import gc
import hashlib
import importlib.util
import io
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import Future
import wave
from dataclasses import dataclass
from types import MethodType
from typing import Any

from .config import Settings
from .domain.models import METADATA_SECONDS_LIMITS, SpeechRequest, TaskType, bounded_metadata_seconds
from . import decode_overlap, fast_silence, packed_decode, row_seeds
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


_TORCH_DTYPE_ALIASES = {'fp16': 'float16', 'bf16': 'bfloat16', 'fp32': 'float32', 'float8': 'fp8'}
SUPPORTED_TORCH_DTYPES = ('float16', 'bfloat16', 'float32', 'fp8')


def normalize_torch_dtype(value: str | None) -> str:
    """Map a dtype setting (aliases like bf16 accepted) to one of SUPPORTED_TORCH_DTYPES."""
    raw = (value or '').strip().lower()
    dtype = _TORCH_DTYPE_ALIASES.get(raw, raw)
    if dtype not in SUPPORTED_TORCH_DTYPES:
        raise ValueError('Unsupported torch dtype')
    return dtype


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


SUPPORTED_ATTENTION_IMPLEMENTATIONS = ('sdpa', 'flex_attention', 'eager')
_ATTENTION_ALIASES = {'flex': 'flex_attention', 'flexattention': 'flex_attention'}


def resolve_omnivoice_language(language: str | None) -> str | None:
    """The language id OmniVoice.generate() resolves ``language`` to (None = agnostic).

    Mirrors omnivoice.models.omnivoice._resolve_language over omnivoice.utils.lang_map
    (exact id, else lower-cased full name, else None) without importing the model module.
    """
    if language is None or not str(language).strip() or str(language).lower() == 'none':
        return None
    value = str(language)
    try:
        from omnivoice.utils.lang_map import LANG_IDS, LANG_NAME_TO_ID
    except Exception:
        known = set(LANGUAGE_LABEL_TO_CODE.values())
        return value if value in known else None
    if value in LANG_IDS:
        return value
    return LANG_NAME_TO_ID.get(value.lower())


def normalize_attention_implementation(value: str | None) -> str:
    """Map an attention setting to a Transformers attn_implementation we support."""
    raw = (value or 'sdpa').strip().lower()
    name = _ATTENTION_ALIASES.get(raw, raw)
    if name not in SUPPORTED_ATTENTION_IMPLEMENTATIONS:
        raise ValueError(
            f'Unsupported attention implementation {value!r}. '
            f'Use one of: {", ".join(SUPPORTED_ATTENTION_IMPLEMENTATIONS)}.'
        )
    return name


def _dense_mask_to_block_mask(mask: Any, create_block_mask: Any) -> Any:
    # Drop the singleton head dim first: mask_mod runs under vmap, where mixing a python
    # int index with the index tensors falls back to .item() and fails.
    dense = mask[:, 0]

    def mask_mod(batch_idx: Any, head_idx: Any, q_idx: Any, kv_idx: Any) -> Any:
        return dense[batch_idx, q_idx, kv_idx]

    return create_block_mask(
        mask_mod,
        B=dense.shape[0],
        H=None,
        Q_LEN=dense.shape[-2],
        KV_LEN=dense.shape[-1],
        device=dense.device,
    )


def install_flex_block_mask_forward(model: Any) -> bool:
    """Make OmniVoice's dense bool attention mask work under flex_attention.

    OmniVoice's iterative decoder passes a dense (2B, 1, S, S) bool mask. Transformers'
    flex_attention integration treats any non-BlockMask tensor as an additive score
    bias, so the bool mask becomes +1/+0 and padded keys are NOT masked (CFG rows always
    carry padding). Converting the mask to a BlockMask on the model's forward restores
    exact SDPA semantics and lets flex skip fully padded blocks. The conversion sits
    outside the compiled LLM and is cached per mask tensor: one generate() reuses the
    same mask object for all num_step forwards.
    """
    try:
        import torch
        from torch.nn.attention.flex_attention import create_block_mask
    except Exception as exc:
        logger.warning('flex_attention BlockMask support unavailable: %s', exc)
        return False
    original_forward = getattr(model, 'forward', None)
    if not callable(original_forward):
        return False
    # Holding the mask keeps identity checks safe (no address reuse) until the next
    # generate() replaces it; inference tensors have no version counter to compare.
    cache: dict[str, Any] = {'mask': None, 'block_mask': None}

    def forward(self: Any, *args: Any, attention_mask: Any = None, **kwargs: Any) -> Any:
        if isinstance(attention_mask, torch.Tensor) and attention_mask.dtype == torch.bool and attention_mask.dim() == 4:
            if cache['mask'] is not attention_mask:
                cache['block_mask'] = _dense_mask_to_block_mask(attention_mask, create_block_mask)
                cache['mask'] = attention_mask
            attention_mask = cache['block_mask']
        return original_forward(*args, attention_mask=attention_mask, **kwargs)

    model.forward = MethodType(forward, model)
    model._g3_flex_block_mask_cache = cache
    return True


def install_eager_additive_mask_forward(model: Any) -> bool:
    """Make OmniVoice's dense bool attention mask work under the eager attention.

    Transformers' eager attention adds a 4D attention_mask to the scores as-is
    (``attn_weights + attention_mask``), so OmniVoice's (2B, 1, S, S) bool mask becomes
    a +1/+0 bias and padded keys are NOT masked (same bug as unpatched flex). The mask is
    converted once per mask tensor into an additive mask in the model dtype: 0 where
    attention is allowed, finfo(dtype).min where it is not.
    """
    try:
        import torch
    except Exception as exc:
        logger.warning('eager attention mask conversion unavailable: %s', exc)
        return False
    original_forward = getattr(model, 'forward', None)
    if not callable(original_forward):
        return False
    cache: dict[str, Any] = {'mask': None, 'additive': None}

    def _mask_dtype() -> Any:
        llm = getattr(model, 'llm', None)
        dtype = getattr(llm, 'dtype', None) or getattr(model, 'dtype', None)
        return dtype if isinstance(dtype, torch.dtype) and dtype.is_floating_point else torch.float32

    def forward(self: Any, *args: Any, attention_mask: Any = None, **kwargs: Any) -> Any:
        if isinstance(attention_mask, torch.Tensor) and attention_mask.dtype == torch.bool and attention_mask.dim() == 4:
            if cache['mask'] is not attention_mask:
                dtype = _mask_dtype()
                additive = torch.zeros(attention_mask.shape, dtype=dtype, device=attention_mask.device)
                cache['additive'] = additive.masked_fill(~attention_mask, torch.finfo(dtype).min)
                cache['mask'] = attention_mask
            attention_mask = cache['additive']
        return original_forward(*args, attention_mask=attention_mask, **kwargs)

    model.forward = MethodType(forward, model)
    model._g3_eager_additive_mask_cache = cache
    return True


# Inductor picks the A100 flex config (164 KB shared memory) for every sm_8x part. On
# GPUs with ~100 KB per block (RTX 30/40/50, sm_86/89/120) that template does not fit
# for head_dim=128 ("out of resource: Required 114688, Hardware limit 101376"), the
# compile fails and dynamo silently runs the unfused eager flex (~10x slower).
_FLEX_SMALL_SMEM_KERNEL_OPTIONS = {'BLOCK_M': 64, 'BLOCK_N': 64, 'num_stages': 2, 'num_warps': 4}
_FLEX_SMALL_SMEM_LIMIT = 160 * 1024


def flex_kernel_options_for_device(torch_module: Any, device: Any) -> dict[str, int] | None:
    try:
        props = torch_module.cuda.get_device_properties(device)
    except Exception:
        return None
    smem = int(getattr(props, 'shared_memory_per_block_optin', 0) or 0)
    if smem and smem < _FLEX_SMALL_SMEM_LIMIT:
        return dict(_FLEX_SMALL_SMEM_KERNEL_OPTIONS)
    return None


def install_flex_kernel_options(model: Any, kernel_options: dict[str, Any]) -> bool:
    """Route the LLM's flex attention through a wrapper that adds kernel_options."""
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, AttentionInterface
    except Exception as exc:
        logger.warning('cannot set flex kernel options: %s', exc)
        return False
    llm = getattr(model, 'llm', None)
    config = getattr(llm, 'config', None)
    # omnivoice >= 0.2 switches the llm to its own 'omnivoice_flex_attention' wrapper.
    base_name = getattr(config, '_attn_implementation', None) or 'flex_attention'
    if 'flex' not in str(base_name):
        return False
    base_fn = ALL_ATTENTION_FUNCTIONS[base_name]

    def flex_with_kernel_options(module: Any, query: Any, key: Any, value: Any, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get('kernel_options') is None:
            kwargs['kernel_options'] = kernel_options
        return base_fn(module, query, key, value, *args, **kwargs)

    AttentionInterface.register('g3_flex_attention', flex_with_kernel_options)
    llm.set_attn_implementation('g3_flex_attention')
    return True


def flashinfer_unavailable_reason(settings: Settings) -> str | None:
    """Why FlashInfer cannot be used on this host, or None when it can be tried."""
    if not sys.platform.startswith('linux'):
        return f'FlashInfer kernels need Linux (platform is {sys.platform})'
    try:
        device = normalize_runtime_device(settings.preferred_device)
    except ValueError as exc:
        return str(exc)
    if not device.startswith('cuda:'):
        return 'FlashInfer needs a CUDA device'
    if settings.enable_cpu_offload:
        return 'FlashInfer is incompatible with CPU offload'
    if (settings.torch_dtype or '').lower() == 'fp8':
        return 'FlashInfer is incompatible with fp8 weights'
    if importlib.util.find_spec('flashinfer') is None:
        return 'flashinfer is not installed'
    try:
        has_patch = importlib.util.find_spec('omnivoice.models.omnivoice_flashinfer') is not None
    except Exception:
        has_patch = False
    if not has_patch:
        return 'installed omnivoice has no omnivoice_flashinfer module (needs upstream >= 08be0b4)'
    return None


def _reset_flashinfer_context() -> None:
    # omnivoice_flashinfer keeps the planned wrapper (128 MB workspace) in a module dict;
    # drop it with the model so a release really frees the VRAM.
    module = sys.modules.get('omnivoice.models.omnivoice_flashinfer')
    ctx = getattr(module, '_CTX', None)
    if isinstance(ctx, dict):
        ctx.clear()
        ctx['wrapper'] = None


_DEFAULT_SDP_PRIORITY: list[int] | None = None
_SDPA_CACHE_TAG = 'g3-sdpa-cudnn-first'


def apply_sdpa_backend_preference(torch_module: Any, prefer_cudnn: bool) -> bool:
    """Put cuDNN first in PyTorch's SDPA backend order, or restore the default order.

    With OmniVoice's dense bool mask the default order picks the mem-efficient kernel (this Windows build has no
    flash kernel); cuDNN is 2-2.7x faster per attention call on the RTX 5090 (~15-22 % per generate), not
    bit-identical. torch.compile bakes the chosen kernel into the graph, and Inductor's on-disk cache key does not
    include the order, so the order is also written into torch.compiler.config.cache_key_tag: otherwise a cached
    graph compiled under the other order would be reused. Takes effect for graphs compiled after the call (model
    load). Returns True if cuDNN is first.
    """
    global _DEFAULT_SDP_PRIORITY
    c_api = getattr(torch_module, '_C', None)
    get_order = getattr(c_api, '_get_sdp_priority_order', None)
    set_order = getattr(c_api, '_set_sdp_priority_order', None)
    if not callable(get_order) or not callable(set_order):
        return False
    try:
        from torch.nn.attention import SDPBackend

        cudnn = int(SDPBackend.CUDNN_ATTENTION)
        current = [int(value) for value in get_order()]
        if _DEFAULT_SDP_PRIORITY is None:
            _DEFAULT_SDP_PRIORITY = list(current)
        order = [cudnn] + [value for value in _DEFAULT_SDP_PRIORITY if value != cudnn] if prefer_cudnn else list(_DEFAULT_SDP_PRIORITY)
        if order != current:
            set_order(order)
        compiler_config = getattr(getattr(torch_module, 'compiler', None), 'config', None)
        if compiler_config is not None and hasattr(compiler_config, 'cache_key_tag'):
            tags = [tag for tag in str(compiler_config.cache_key_tag or '').split(',') if tag and tag != _SDPA_CACHE_TAG]
            if prefer_cudnn:
                tags.append(_SDPA_CACHE_TAG)
            compiler_config.cache_key_tag = ','.join(tags)
        return prefer_cudnn
    except Exception as exc:
        logger.warning('could not set the SDPA backend order: %s', exc)
        return False


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
        # Attention backend the resident model really runs ('sdpa', 'flex_attention',
        # 'flashinfer', ...) and why it differs from the setting, if it does.
        self.attention_backend: str | None = None
        self.attention_backend_note: str | None = None
        # Step-boundary preemption: while a batch renders, urgent items (a stream's first sentence) queued here run
        # solo at the next diffusion step boundary (forward pre-hook), see render_urgent.
        self._urgent: queue.SimpleQueue[tuple[list[BatchSynthesisItem], Future]] = queue.SimpleQueue()
        self._urgent_lock = threading.Lock()
        self._rendering = False
        self._in_urgent = False

    def runtime_info(self) -> dict[str, Any]:
        return {
            'attention_backend': self.attention_backend,
            'attention_backend_note': self.attention_backend_note,
        }

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

    async def render_batch(self, items: list[BatchSynthesisItem], *, trim_after: bool = True) -> list[BatchSynthesisResult]:
        if not items:
            return []
        return await asyncio.to_thread(self._render_batch_thread, items, trim_after)

    def _render_batch_thread(self, items: list[BatchSynthesisItem], trim_after: bool) -> list[BatchSynthesisResult]:
        with self._urgent_lock:
            self._rendering = True
        try:
            return self._generate_batch_sync(items, trim_after=trim_after)
        finally:
            with self._urgent_lock:
                self._rendering = False
            # Urgent items that arrived after the last step boundary run now, before the worker plans again.
            self._run_urgent_sync()

    async def render_urgent(self, items: list[BatchSynthesisItem]) -> list[BatchSynthesisResult] | None:
        """Render ``items`` inside the running batch, at its next diffusion step boundary.

        Returns None when no batch is rendering (the caller then leaves the items to the normal queue). The running
        batch's random state is saved and restored around the urgent items, so its audio does not change; it only
        pauses for the urgent generate.
        """
        if not items or self._model is None:
            return None
        future: Future = Future()
        with self._urgent_lock:
            if not self._rendering:
                return None
            self._urgent.put((items, future))
        return await asyncio.wrap_future(future)

    def _run_urgent_sync(self) -> None:
        """Run queued urgent items now (worker thread, between two diffusion steps or right after the batch)."""
        if self._in_urgent:
            return
        while True:
            try:
                items, future = self._urgent.get_nowait()
            except queue.Empty:
                return
            if not future.set_running_or_notify_cancel():
                continue
            torch = self._torch
            cuda = torch is not None and torch.cuda.is_available()
            cpu_state = torch.get_rng_state() if torch is not None else None
            cuda_state = torch.cuda.get_rng_state() if cuda else None
            active_seeds = row_seeds._ACTIVE
            row_seeds._ACTIVE = None
            self._in_urgent = True
            try:
                result = self._generate_batch_sync(items, trim_after=False)
            except BaseException as exc:  # handed to the waiting request
                future.set_exception(exc)
            else:
                future.set_result(result)
            finally:
                self._in_urgent = False
                row_seeds._ACTIVE = active_seeds
                if cpu_state is not None:
                    torch.set_rng_state(cpu_state)
                if cuda_state is not None:
                    torch.cuda.set_rng_state(cuda_state)

    def _install_urgent_hook(self, model: Any) -> None:
        register = getattr(model, 'register_forward_pre_hook', None)
        if not callable(register):
            return

        def hook(module: Any, args: Any) -> None:
            if not self._in_urgent and not self._urgent.empty():
                self._run_urgent_sync()

        register(hook)

    async def trim_idle(self) -> None:
        """Release the CUDA cache once the queue has run empty (the queue calls this instead of a per-batch trim)."""
        if self._model is None or not self.settings.cuda_memory_trim_after_batch:
            return
        await asyncio.to_thread(self._trim_cuda_cache_sync, collect_garbage=False)

    def _load_model_sync(self, model_id: str) -> int:
        start = time.perf_counter()
        torch, omnivoice_cls = self._load_runtime_dependencies()
        # Same samples as OmniVoice's pydub silence trimming at a fraction of the CPU time (self-checked on install).
        fast_silence.install()
        model_source, extra_kwargs = self._resolve_model_source(model_id)
        if self._model is not None and self._loaded_model_id != model_id:
            self._release_model()

        use_flashinfer = False
        note: str | None = None
        if getattr(self.settings, 'enable_flashinfer', False):
            reason = flashinfer_unavailable_reason(self.settings)
            if reason is None:
                use_flashinfer = True
            else:
                note = f'FlashInfer requested but not used: {reason}'
                logger.warning('%s; using %s', note, self.settings.attention_implementation)

        try:
            configured_attn = normalize_attention_implementation(self.settings.attention_implementation)
        except ValueError as exc:
            note = f'{exc} Falling back to sdpa.'
            logger.warning(note)
            configured_attn = 'sdpa'
        # apply_flashinfer swaps the attention itself; it patches the stock SDPA model.
        attn_implementation = 'sdpa' if use_flashinfer else configured_attn

        model, attn_implementation = self._from_pretrained(
            omnivoice_cls,
            model_id,
            model_source,
            extra_kwargs,
            # FlashInfer's packed attention is planned for fp16 only (bf16 raises in run()).
            dtype=torch.float16 if use_flashinfer else self._torch_dtype(torch),
            attn_implementation=attn_implementation,
            allow_fp8=not use_flashinfer,
        )

        backend = attn_implementation
        if use_flashinfer:
            failure: str | None = None
            try:
                self._apply_flashinfer(model)
                backend = 'flashinfer'
            except Exception as exc:
                failure = f'{type(exc).__name__}: {exc}'
            if failure is not None:
                # Outside the except block, so the traceback no longer pins the patched model.
                note = f'FlashInfer failed ({failure}); reloaded without it'
                logger.warning(note)
                del model
                _reset_flashinfer_context()
                self._trim_cuda_cache_sync()
                model, attn_implementation = self._from_pretrained(
                    omnivoice_cls,
                    model_id,
                    model_source,
                    extra_kwargs,
                    dtype=self._torch_dtype(torch),
                    attn_implementation=configured_attn,
                    allow_fp8=True,
                )
                backend = attn_implementation
                use_flashinfer = False

        if backend == 'flex_attention' and not install_flex_block_mask_forward(model):
            # Unpatched flex ignores the padding mask (bool mask -> +1 bias): never run it.
            note = 'flex_attention mask patch unavailable; reloaded with sdpa'
            logger.warning(note)
            del model
            self._trim_cuda_cache_sync()
            model, backend = self._from_pretrained(
                omnivoice_cls,
                model_id,
                model_source,
                extra_kwargs,
                dtype=self._torch_dtype(torch),
                attn_implementation='sdpa',
                allow_fp8=True,
            )

        if backend == 'eager' and not install_eager_additive_mask_forward(model):
            # Eager adds the bool mask as a +1/+0 bias, so padding would be attended to.
            note = 'eager attention mask patch unavailable; reloaded with sdpa'
            logger.warning(note)
            del model
            self._trim_cuda_cache_sync()
            model, backend = self._from_pretrained(
                omnivoice_cls,
                model_id,
                model_source,
                extra_kwargs,
                dtype=self._torch_dtype(torch),
                attn_implementation='sdpa',
                allow_fp8=True,
            )

        if backend == 'flex_attention' and str(self.settings.preferred_device).startswith('cuda'):
            kernel_options = flex_kernel_options_for_device(torch, self.settings.preferred_device)
            if kernel_options and install_flex_kernel_options(model, kernel_options):
                logger.info('flex_attention kernel options for this GPU: %s', kernel_options)

        if note is None and not use_flashinfer and backend != configured_attn:
            note = f'{configured_attn} failed to load; using {backend}'
        self.attention_backend = backend
        self.attention_backend_note = note
        logger.info('model=%s attention backend=%s', model_id, backend)

        if getattr(self.settings, 'packed_cfg', True) and not use_flashinfer:
            # One row per item [cond | uncond] instead of 2B padded rows (packed_decode). Installed before the
            # compile and the warm-up, so the warm-up compiles the shapes real requests use.
            if packed_decode.install(model):
                logger.info('packed CFG decoding active')
        # torch.compile would trace around FlashInfer's patched forwards and planned
        # wrapper; upstream runs that path eager.
        if getattr(self.settings, 'compile_model', False) and not use_flashinfer:
            self._compile_model(model)

        self._install_urgent_hook(model)
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

    def _from_pretrained(
        self,
        omnivoice_cls: Any,
        model_id: str,
        model_source: str,
        extra_kwargs: dict[str, Any],
        *,
        dtype: Any,
        attn_implementation: str,
        allow_fp8: bool,
    ) -> tuple[Any, str]:
        """Load the checkpoint; returns (model, attn_implementation actually used)."""
        kwargs = {
            'device_map': 'auto' if self.settings.enable_cpu_offload else self.settings.preferred_device,
            'dtype': dtype,
            'attn_implementation': attn_implementation,
            'load_asr': False,
        }
        kwargs.update(extra_kwargs)

        model = None
        quant_config = self._fp8_quant_config_if_requested() if allow_fp8 else None
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
                attn_implementation = 'default'
            except Exception as exc:
                if attn_implementation == 'sdpa':
                    raise RuntimeError(f'Failed to load OmniVoice model {model_id}: {exc}') from exc
                logger.warning('load with attn_implementation=%s failed (%s); retrying with sdpa', attn_implementation, exc)
                fallback = dict(kwargs)
                fallback['attn_implementation'] = 'sdpa'
                try:
                    model = omnivoice_cls.from_pretrained(model_source, **fallback)
                except Exception:
                    raise RuntimeError(f'Failed to load OmniVoice model {model_id}: {exc}') from exc
                attn_implementation = 'sdpa'
        return model, attn_implementation

    def _apply_flashinfer(self, model: Any) -> None:
        """Patch the model with upstream's FlashInfer packed attention and smoke-test it.

        Plain apply_flashinfer (eager, no CUDA graphs): the graph cache keys on the exact
        tuple of sequence lengths and never evicts, so per-request text would capture a
        new graph almost every time. A missing JIT kernel only shows up at the first
        generate(), hence the tiny test generation before the model goes live.
        """
        from omnivoice.models.omnivoice_flashinfer import apply_flashinfer

        apply_flashinfer(model)
        grad_ctx = self._torch.inference_mode() if self._torch is not None else contextlib.nullcontext()
        with grad_ctx:
            model.generate(text=['Test.'], language=['en'], num_step=2)

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
            # Same grad mode as the request path (_generate_batch_sync): without it the
            # warmup builds autograd state and its compiled graphs are guarded on grad mode,
            # so the first real request would recompile instead of reusing them.
            grad_ctx = self._torch.inference_mode() if self._torch is not None else contextlib.nullcontext()
            with grad_ctx:
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
                    profile = self._warmup_voice_profile()
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

    def _warmup_voice_profile(self) -> VoiceProfileRecord | None:
        """The configured default voice when it can clone, else the first profile that can.

        Warming the default voice compiles the prompt length real default-voice requests use
        and caches its clone prompt. The lookup matches QueueService._apply_default_voice
        (voice_id or name, exact first, then case-insensitive).
        """
        usable = [item for item in self.store.voice_profiles.values() if item.audio_bytes and item.ref_text]
        default_voice = (self.settings.default_voice or '').strip()
        if default_voice:
            lowered = default_voice.casefold()
            profile = next((p for p in usable if default_voice in (p.voice_id, p.name)), None) or next(
                (p for p in usable if lowered in (p.voice_id.casefold(), str(p.name or '').casefold())), None
            )
            if profile is not None:
                return profile
        return usable[0] if usable else None

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
            apply_sdpa_backend_preference(torch, bool(getattr(self.settings, 'sdpa_prefer_cudnn', True)))

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
        self.attention_backend = None
        self.attention_backend_note = None
        _reset_flashinfer_context()
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

    def _trim_cuda_cache_sync(self, *, reset_compile_cache: bool = False, collect_garbage: bool = True) -> None:
        if collect_garbage:
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

    def _generate_batch_sync(self, items: list[BatchSynthesisItem], trim_after: bool = True) -> list[BatchSynthesisResult]:
        if self._model is None:
            raise RuntimeError('No model loaded. Call ensure_model before generation.')

        texts = [item.text.strip() for item in items]
        if any(not text for text in texts):
            raise RuntimeError('Missing input text')

        task_type = items[0].request.task_type or self._task_type_from_model(self._loaded_model_id or self.settings.active_model)
        languages = [self._normalize_language(item.request.language) for item in items]
        kwargs = self._generation_kwargs(items)

        # One seed per batch keeps the stock sampler (bit-identical to before). Items that ask for different seeds get
        # per-item generators instead, so they can share a batch (row_seeds).
        seeds = [item.request.seed for item in items]
        mixed_seeds = len({seed for seed in seeds}) > 1
        seed_scope = (
            row_seeds.per_item_seeds(seeds, self._model.device) if mixed_seeds else contextlib.nullcontext()
        )
        try:
            with self._torch.inference_mode(), seed_scope, decode_overlap.overlapped_decode(self._model):
                if not mixed_seeds:
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
        except Exception as exc:
            # Only an OOM justifies dropping the compiled graphs; any other failure (bad
            # voice, missing ref_text, ...) would otherwise force a full dynamo re-trace
            # and recompile on the next request.
            self._trim_cuda_cache_sync(reset_compile_cache=self._is_cuda_oom(exc))
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
        if trim_after and self.settings.cuda_memory_trim_after_batch:
            # Hot path: release the reserved pool only. A full gc.collect() walks the
            # ~700k tracked objects of the loaded stack (~215 ms on a 5900X) on every
            # batch while freeing no VRAM -- batch tensors die by refcount.
            self._trim_cuda_cache_sync(collect_garbage=False)
        return results

    def _is_cuda_oom(self, exc: BaseException) -> bool:
        oom_type = getattr(getattr(self._torch, 'cuda', None), 'OutOfMemoryError', None)
        if isinstance(oom_type, type) and isinstance(exc, oom_type):
            return True
        return 'out of memory' in str(exc).lower()

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
        # omnivoice >= 0.2: per-request silence padding / fade per side (upstream default
        # 0.1 s each, identical to the old hard-coded values). Only forwarded when a
        # client sets them, so the default output stays bit-identical.
        # Bounded (pad <= 2 s, fade <= 1 s) and finite: SpeechRequest rejects anything else
        # with 422; this second check only guards internally built requests.
        for key in METADATA_SECONDS_LIMITS:
            try:
                value = bounded_metadata_seconds(key, metadata.get(key))
            except ValueError as exc:
                logger.warning('%s ignored: %s', key, exc)
                continue
            if value is not None:
                result[key] = value
        if metadata.get('normalize_text') and self._upstream_normalize_text_usable(items):
            result['normalize_text'] = True
        return result

    def _upstream_normalize_text_usable(self, items: list[BatchSynthesisItem]) -> bool:
        """Gate omnivoice's own normalize_text (zh/en via WeTextProcessing, else num2words).

        The decision uses the language generate() itself resolves (lang_map): anything it
        does not recognise ('de-DE', 'en-US', 'xx', Auto) becomes None there, and
        normalize_text(None) guesses zh/en, so German text would get the English
        normaliser. A missing WeTextProcessing (no pynini wheels on Windows) raises
        ImportError out of generate() and fails the whole batch, and without num2words
        the other languages are a silent no-op. Only forward the flag when every item
        has a recognised language whose normaliser is importable.
        """
        resolved = [resolve_omnivoice_language(self._normalize_language(item.request.language)) for item in items]
        if any(code is None for code in resolved):
            logger.info(
                'normalize_text ignored: needs a language OmniVoice recognises per request (got %s)',
                sorted({str(item.request.language) for item in items}),
            )
            return False
        if any(code in ('zh', 'en') for code in resolved) and importlib.util.find_spec('tn') is None:
            logger.info('normalize_text ignored: WeTextProcessing (tn) is not installed')
            return False
        if any(code not in ('zh', 'en') for code in resolved) and importlib.util.find_spec('num2words') is None:
            logger.info('normalize_text ignored: num2words is not installed')
            return False
        return True

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
    if settings.runtime_backend.lower() == 'phonon':
        from .phonon_runtime import PhononSynthesizer

        synthesizer = PhononSynthesizer(settings=settings, store=store)
        # Voices are known only after the engines load, and request voice resolution needs them at once.
        synthesizer.load_now()
        # One checkpoint serves here: expose only it, with its catalog voices as built-in voices.
        settings.supported_models = [settings.phonon_model_id]
        settings.active_model = settings.phonon_model_id
        settings.built_in_voices = [*settings.built_in_voices, *synthesizer.voice_names()]
        store.active_model = settings.phonon_model_id
        return synthesizer
    if settings.runtime_backend.lower() == 'ming':
        from .ming_runtime import MING_BASE_ALIAS, MingSynthesizer

        # Weights load lazily (ensure_model); the queue's first batch or the startup preload pulls them in.
        synthesizer = MingSynthesizer(settings=settings, store=store)
        # One clone alias whose id ends in 'Base': task-type inference, the default voice profile and the
        # OpenAI voice routing treat it exactly like OmniVoice-Base. Voice profiles stay loaded.
        settings.supported_models = [MING_BASE_ALIAS]
        settings.active_model = MING_BASE_ALIAS
        store.active_model = MING_BASE_ALIAS
        return synthesizer
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
