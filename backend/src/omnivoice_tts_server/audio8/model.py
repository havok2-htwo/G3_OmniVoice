"""Load an Audio8 checkpoint from its pinned local folder.

The modeling code lives next to the weights and is loaded with ``trust_remote_code`` from that folder only
(``local_files_only``): nothing is fetched at load, and the code stays with the checkpoint instead of being vendored
here (the 0.1B code is under its own license).
"""
from __future__ import annotations

import contextlib
import json
import logging
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

# Non-persistent buffers of ArkttsModel and how to rebuild them. Transformers 5 builds the model on the meta device, so
# they come out of from_pretrained as garbage (for the 0.6B: zeros); cloning then stops after 2 frames without an error,
# because a zero rotation wipes out every query and key.
_ROPE_BUFFERS = {'freqs_cis': ('max_seq_len', 'head_dim'), 'fast_freqs_cis': ('num_codebooks', 'fast_head_dim')}
# The same inside the 0.1B's Falcon-H1 backbone: mup_vector comes out as garbage too, and everything Mamba does goes
# through it.
_FALCON_BUFFERS = {'inv_freq', 'original_inv_freq', 'mup_vector'}


def restore_rope_tables(model: Any) -> None:
    """Recompute the RoPE tables and refuse to continue when the model holds another non-persistent buffer we do not
    know how to rebuild. The second part is what catches a future checkpoint with the same trap."""
    module = sys.modules[type(model).__module__]
    config = model.config
    tables = {name: spec for name, spec in _ROPE_BUFFERS.items() if name in model._non_persistent_buffers_set}
    unknown = sorted(set(model._non_persistent_buffers_set) - set(tables))
    nested = sorted(f'{name}.{buffer}' for name, sub in model.named_modules() if name
                    for buffer in sub._non_persistent_buffers_set - _FALCON_BUFFERS)
    if unknown or nested:
        raise RuntimeError(f'Audio8 model has non-persistent buffers that from_pretrained does not restore: '
                           f'{unknown + nested}')
    for name, (length, head_dim) in tables.items():
        table = getattr(model, name)
        fresh = module._precompute_rope(getattr(config, length), getattr(config, head_dim), config.rope_base)
        table.copy_(fresh.to(device=table.device, dtype=table.dtype))
    if hasattr(model, 'slow'):
        _restore_falcon_buffers(model.slow)


def _restore_falcon_buffers(slow: Any) -> None:
    module = sys.modules[type(slow).__module__]
    rotary = slow.rotary_emb
    frequencies, _ = rotary.compute_default_rope_parameters(slow.config, device=rotary.inv_freq.device)
    rotary.inv_freq.copy_(frequencies)
    rotary.original_inv_freq.copy_(frequencies)
    multipliers = module.compute_mup_vector(slow.config)
    for layer in slow.layers:
        layer.mamba.mup_vector.copy_(multipliers.to(layer.mamba.mup_vector))


def _hybrid_cache(config: Any, batch_size: int, dtype: torch.dtype, devices: Any = None) -> Any:
    from transformers import DynamicCache

    return DynamicCache(config=config)


@contextlib.contextmanager
def _falcon_remote_code() -> Iterator[None]:
    """What the 0.1B's code needs around its import and construction under transformers 5.

    - It imports FalconHybridMambaAttentionDynamicCache, which transformers 5 replaced by DynamicCache. The name only
      serves upstream's generate(), which the engine does not use.
    - FalconH1Mixer.__init__ fetches compiled Mamba kernels from the Hub. Nothing may be fetched at load, and the kernels
      only serve Falcon's own forward: without them it takes its plain torch path, same math."""
    from transformers.models.falcon_h1 import modeling_falcon_h1 as falcon

    if not hasattr(falcon, 'FalconHybridMambaAttentionDynamicCache'):
        falcon.FalconHybridMambaAttentionDynamicCache = _hybrid_cache
    fetch = falcon.lazy_load_kernel
    falcon.lazy_load_kernel = lambda _name: None
    try:
        yield
    finally:
        falcon.lazy_load_kernel = fetch


def _uses_falcon(path: str) -> bool:
    config = json.loads((Path(path) / 'config.json').read_text(encoding='utf-8'))
    return config.get('slow_backbone') == 'falcon_h1'


def _plain_snake(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    return x.reshape(shape)


def _replace_scripted_snake(codec: Any) -> None:
    """The codec's activation is a torch.jit.script function. The JIT rounds differently from its second call on, so a
    reference clip encoded first in a process got other codes (13 % of them) than the same clip encoded later. The
    plain function has one behaviour, from the first call on."""
    module = sys.modules[type(codec).__module__]
    module._arktts_snake = _plain_snake


def load_audio8_model(model_dir: Path | str, device: torch.device | str, dtype: torch.dtype,
                      codec_dtype: torch.dtype | None = None) -> tuple[Any, Any, Any]:
    """(model, codec, tokenizer) on ``device``. Raises on missing or unexpected weights instead of leaving them random."""
    from transformers import AutoModel, AutoTokenizer

    path = str(Path(model_dir))
    with _falcon_remote_code() if _uses_falcon(path) else contextlib.nullcontext():
        model, info = AutoModel.from_pretrained(path, trust_remote_code=True, local_files_only=True, dtype=dtype,
                                                output_loading_info=True)
    tied = set(getattr(model, 'all_tied_weights_keys', None) or ())
    missing = [key for key in info.get('missing_keys', ()) if key not in tied]
    unexpected = list(info.get('unexpected_keys', ()))
    if missing or unexpected or info.get('mismatched_keys'):
        raise RuntimeError(f'Audio8 checkpoint {path} does not fit the model: missing={missing[:8]} '
                           f'unexpected={unexpected[:8]} mismatched={info.get("mismatched_keys")}')
    model = model.eval().to(device)
    model.requires_grad_(False)
    restore_rope_tables(model)
    codec = model.load_codec(dtype=codec_dtype or dtype)
    codec.requires_grad_(False)
    _replace_scripted_snake(codec)
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, local_files_only=True, trust_remote_code=False,
                                              fix_mistral_regex=False)
    return model, codec, tokenizer
