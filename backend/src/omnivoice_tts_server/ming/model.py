# Adapted from inclusionAI/Ming-3.0-tts @ 9332fb9 (modeling_bailingmm.py, configuration_bailingmm.py), MIT license,
# see LICENSE in this directory. Only the module tree is kept (so model.safetensors loads strictly); the generation
# loop lives in engine.py. Changes against upstream:
# - The LLM is a bare Qwen2Model under ``model.model`` (no lm_head: the tied 151936-vocab head is never needed, the
#   loop only uses the final normed hidden state, which is what upstream reads from ``hidden_states[-1]``).
# - Weights load from the local directory with safetensors (no trust_remote_code, no network), built on the meta
#   device and assigned, then cast to bf16 as a whole exactly like upstream ``.eval().to(torch.bfloat16)``.
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from torch import nn
from transformers import Qwen2Config, Qwen2Model

from .audio_vae import AudioVAE
from .flow import Aggregator, FlowLoss

logger = logging.getLogger(__name__)

MAX_CACHE_LEN = 512
SPEAKER_EMBEDDING_DIM = 192
# The tied LM head is not part of this tree; nothing else may be missing or unexpected.
_IGNORED_KEYS = ('model.lm_head.weight',)


class _LLM(nn.Module):
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.model = Qwen2Model(config)


class MingModel(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        if str(config.get('model_type')) != 'dense':
            raise NotImplementedError(f"only the dense Ming-3.0-tts-0.5B is supported, got {config.get('model_type')!r}")
        self.config = config
        llm_args = dict(config['llm_config'])
        llm_args['_attn_implementation'] = 'sdpa'
        llm_config = Qwen2Config.from_dict(llm_args)
        llm_config._attn_implementation = 'sdpa'
        self.model = _LLM(llm_config)
        self.llm_config = llm_config
        self.audio = AudioVAE(config['audio_tokenizer_config'])
        self.latent_dim = int(config['audio_tokenizer_config']['enc_kwargs']['latent_dim'])
        hidden = llm_config.hidden_size
        self.linear_proj_audio = Aggregator(in_channels=self.latent_dim, llm_input_dim=hidden,
                                            **config['aggregator_config'])
        self.patch_size = int(config['ditar_config']['patch_size'])
        self.history_patch_size = int(config['ditar_config'].get('history_patch_size', self.patch_size))
        self.flowloss = FlowLoss(z_channels=self.latent_dim, llm_cond_dim=hidden, **config['ditar_config'])
        self.stop_head = nn.Linear(hidden, 2, bias=True)
        self.spk_head = nn.Linear(SPEAKER_EMBEDDING_DIM, hidden, bias=True)

    @property
    def llm(self) -> Qwen2Model:
        return self.model.model

    @property
    def dit(self):
        return self.flowloss.cfm.model

    @property
    def sample_rate(self) -> int:
        return self.audio.sample_rate

    @property
    def prompt_align(self) -> int:
        """Prompt waveforms are front-padded to a multiple of this many samples (upstream _pad_align)."""
        enc = self.audio.encoder
        return enc.hop_size * max(1, enc.patch_size) * self.patch_size


def _materialize_rotary(module: nn.Module, device: torch.device) -> None:
    """Non-persistent rotary buffers are not in the checkpoint; recompute them after a meta-device build."""
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

    for sub in module.modules():
        if isinstance(sub, Qwen2RotaryEmbedding):
            if sub.rope_type == 'default':
                inv_freq, scaling = sub.compute_default_rope_parameters(sub.config, device)
            else:
                inv_freq, scaling = ROPE_INIT_FUNCTIONS[sub.rope_type](sub.config, device)
            sub.register_buffer('inv_freq', inv_freq, persistent=False)
            sub.register_buffer('original_inv_freq', inv_freq.clone(), persistent=False)
            sub.attention_scaling = scaling


def load_ming_model(model_dir: str | Path, device: str | torch.device = 'cuda:0',
                    dtype: torch.dtype = torch.bfloat16) -> MingModel:
    from safetensors.torch import load_file

    model_dir = Path(model_dir)
    config = json.loads((model_dir / 'config.json').read_text(encoding='utf-8'))
    with torch.device('meta'):
        model = MingModel(config)
    state = load_file(str(model_dir / 'model.safetensors'), device='cpu')
    for key in _IGNORED_KEYS:
        state.pop(key, None)
    result = model.load_state_dict(state, strict=False, assign=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f'Ming checkpoint mismatch: missing={result.missing_keys[:10]} '
                           f'unexpected={result.unexpected_keys[:10]}')
    del state
    _materialize_rotary(model, torch.device('cpu'))
    leftovers = [name for name, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    if leftovers:
        raise RuntimeError(f'Ming model has uninitialised tensors after loading: {leftovers[:10]}')
    # Upstream: from_pretrained(...).eval().to(torch.bfloat16).to(device) -- buffers (rotary inv_freq, ISTFT window)
    # are cast with everything else.
    model = model.eval().to(dtype).to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model
