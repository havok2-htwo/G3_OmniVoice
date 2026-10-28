# Vendored from inclusionAI/Ming-3.0-tts @ 9332fb9 (fm/modules.py, fm/dit.py, fm/CFM.py, fm/flowloss.py), MIT license,
# see LICENSE in this directory. Trimmed to inference. Changes against upstream:
# - RotaryEmbedding / apply_rotary_pos_emb are a minimal copy of x-transformers 2.9.2 (same ops, no einops).
# - The RMSNorm "native" check uses packaging.Version: upstream's float(torch.__version__[:3]) reads "2.10" as 2.1.
# - Training-only code (CFM.forward, cfg dropout) is dropped. Module and parameter names are unchanged so the
#   checkpoint loads strictly.
# - ``sample_flow`` is the CFM sampler with explicit starting noise; the temperature-0 SDE term (an exact +0.0) is
#   skipped. The timestep schedule is computed exactly like upstream (in the noise dtype, on the noise device).
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from packaging.version import Version
from torch import nn

_TORCH_HAS_RMS_NORM = Version(torch.__version__.split('+')[0]) >= Version('2.4')


# -- x-transformers rotary (use_xpos=False path only) -------------------------------------------------------------


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.interpolation_factor = 1.0

    def forward_from_seq_len(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        return self.forward(t)

    @torch.autocast('cuda', enabled=False)
    def forward(self, t: torch.Tensor):
        if t.ndim == 1:
            t = t.unsqueeze(0)
        freqs = torch.einsum('b i , j -> b i j', t.type_as(self.inv_freq), self.inv_freq) / self.interpolation_factor
        freqs = torch.stack((freqs, freqs), dim=-1)
        freqs = freqs.reshape(*freqs.shape[:-2], freqs.shape[-2] * 2)
        return freqs, 1.0


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return x.reshape(*x.shape[:-2], x.shape[-2] * 2)


@torch.autocast('cuda', enabled=False)
def apply_rotary_pos_emb(t: torch.Tensor, freqs: torch.Tensor, scale=1):
    rot_dim, seq_len, orig_dtype = freqs.shape[-1], t.shape[-2], t.dtype
    freqs = freqs[:, -seq_len:, :]
    if t.ndim == 4 and freqs.ndim == 3:
        freqs = freqs.unsqueeze(1)
    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos() * scale) + (_rotate_half(t) * freqs.sin() * scale)
    out = torch.cat((t, t_unrotated), dim=-1)
    return out.type(orig_dtype)


# -- fm/modules.py --------------------------------------------------------------------------------------------------


class RMSNorm(nn.Module):
    # Class-level switch so a parity test can force upstream's (accidental) manual path on torch >= 2.10.
    native_default = _TORCH_HAS_RMS_NORM

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.native_rms_norm = None  # None -> class default

    def forward(self, x):
        native = self.native_default if self.native_rms_norm is None else self.native_rms_norm
        if native:
            if self.weight.dtype in (torch.float16, torch.bfloat16):
                x = x.to(self.weight.dtype)
            x = F.rms_norm(x, normalized_shape=(x.shape[-1],), weight=self.weight, eps=self.eps)
        else:
            variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            if self.weight.dtype in (torch.float16, torch.bfloat16):
                x = x.to(self.weight.dtype)
            x = x * self.weight
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, dropout=0.0, approximate: str = 'none'):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU(approximate=approximate))
        self.ff = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.ff(x)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)
        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, dim), nn.Dropout(dropout)])

    def forward(self, x, mask=None, rope=None):
        batch_size = x.shape[0]
        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)
        head_dim = key.shape[-1] // self.heads
        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        if rope is not None:
            freqs, xpos_scale = rope
            q_scale, k_scale = (xpos_scale, xpos_scale ** -1.0) if xpos_scale is not None else (1.0, 1.0)
            query = apply_rotary_pos_emb(query, freqs, q_scale)
            key = apply_rotary_pos_emb(key, freqs, k_scale)
        x = F.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        x = x.to(query.dtype)
        x = self.to_out[0](x)
        x = self.to_out[1](x)
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
        return x


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.1, **kwargs):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(dim=hidden_size, heads=num_heads, dim_head=hidden_size // num_heads, dropout=dropout)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = FeedForward(dim=hidden_size, mult=mlp_ratio, dropout=dropout, approximate='tanh')

    def forward(self, x, mask, rope):
        x = x + self.attn(self.norm1(x), mask=mask, rope=rope)
        x = x + self.mlp(self.norm2(x))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, x):
        return self.linear(self.norm_final(x))


# -- fm/dit.py ------------------------------------------------------------------------------------------------------


class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class TimestepEmbedder(nn.Module):
    def __init__(self, dim, freq_embed_dim=256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, timestep):
        time_hidden = self.time_embed(timestep).to(timestep.dtype)
        return self.time_mlp(time_hidden)


class CondEmbedder(nn.Module):
    def __init__(self, input_feature_size, hidden_size, dropout_prob):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.cond_embedder = nn.Linear(input_feature_size, hidden_size)

    def forward(self, llm_cond, train=False):
        return self.cond_embedder(llm_cond)


class DiT(nn.Module):
    def __init__(self, in_channels=4, hidden_size=1024, depth=28, num_heads=16, mlp_ratio=4.0, llm_cond_dim=896,
                 cfg_dropout_prob=0.1, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.x_embedder = nn.Linear(in_channels, hidden_size)
        self.c_embedder = CondEmbedder(llm_cond_dim, hidden_size, cfg_dropout_prob)
        self.hidden_size = hidden_size
        self.rotary_embed = RotaryEmbedding(hidden_size // num_heads)
        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **kwargs) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, self.out_channels)

    def forward(self, x, t, c, latent_history, mask=None):
        t = self.t_embedder(t).unsqueeze(1)
        x_now = self.x_embedder(x)
        x_history = self.x_embedder(latent_history)
        x = torch.cat([x_history, x_now], dim=1)
        c = self.c_embedder(c)
        y = t + c
        x = torch.cat([y, x], dim=1)
        rope = self.rotary_embed.forward_from_seq_len(x.shape[1])
        for block in self.blocks:
            x = block(x, None, rope)
        return self.final_layer(x)

    def forward_with_cfg(self, x, t, c, latent_history, patch_size):
        """Conditional and unconditional branch as one doubled batch (both always computed)."""
        x = torch.cat([x, x], dim=0)
        latent_history = torch.cat([latent_history, latent_history], dim=0)
        c = torch.cat([c, torch.zeros_like(c)], dim=0)
        if t.ndim == 0:
            t = t.repeat(x.shape[0])
        model_out = self.forward(x, t, c, latent_history)
        return model_out[:, -patch_size:, :]


class Aggregator(nn.Module):
    def __init__(self, in_channels=4, hidden_size=1152, depth=28, num_heads=16, mlp_ratio=4.0, llm_input_dim=896,
                 **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads
        self.word_embedder = nn.Embedding(1, hidden_size)
        self.x_embedder = nn.Linear(in_channels, hidden_size)
        self.hidden_size = hidden_size
        self.rotary_embed = RotaryEmbedding(hidden_size // num_heads)
        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **kwargs) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, llm_input_dim)

    def forward(self, x, mask=None):
        x = self.x_embedder(x)
        cls_embed = self.word_embedder(torch.zeros((x.shape[0], 1), dtype=torch.long, device=x.device))
        x = torch.cat([cls_embed, x], dim=1)
        rope = self.rotary_embed.forward_from_seq_len(x.shape[1])
        for block in self.blocks:
            x = block(x, None, rope)
        x = self.final_layer(x)
        return x[:, :1, :]


# -- fm/CFM.py + fm/flowloss.py -------------------------------------------------------------------------------------

_EPSS = {
    5: [0, 2, 4, 8, 16, 32],
    6: [0, 2, 4, 6, 8, 16, 32],
    7: [0, 2, 4, 6, 8, 16, 24, 32],
    10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32],
    12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
    16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32],
}


def get_epss_timesteps(n, device, dtype):
    t = _EPSS.get(n, [])
    if not t:
        return torch.linspace(0, 1, n + 1, device=device, dtype=dtype)
    return (1 / 32) * torch.tensor(t, device=device, dtype=dtype)


def flow_timesteps(steps: int, device, dtype, sway_sampling_coef: float | None = -1.0, use_epss: bool = True):
    """The sampler's time grid, computed with the same ops and dtype as upstream CFM.sample."""
    if use_epss:
        t = get_epss_timesteps(steps, device=device, dtype=dtype)
    else:
        t = torch.linspace(0, 1, steps + 1, device=device, dtype=dtype)
    if sway_sampling_coef is not None:
        t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)
    return t


class CFM(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model


class FlowLoss(nn.Module):
    def __init__(self, z_channels, llm_cond_dim, **kwargs):
        super().__init__()
        self.z_channels = z_channels
        self.cfm = CFM(model=DiT(in_channels=z_channels, llm_cond_dim=llm_cond_dim, **kwargs))


def sample_flow(dit: DiT, noise: torch.Tensor, c: torch.Tensor, latent_history: torch.Tensor, *, cfg_scale: float = 2.0,
                steps: int = 10, patch_size: int = 1, sigma: float = 0.25, temperature: float = 0.0,
                sde_noise: torch.Tensor | None = None, timesteps: torch.Tensor | None = None) -> torch.Tensor:
    """Euler integration of the CFG flow from ``noise`` (B, patch, latent) to a latent patch.

    Identical to upstream CFM.sample. At temperature 0 the stochastic term is ``0 * sde_noise`` and is skipped.
    """
    t = timesteps if timesteps is not None else flow_timesteps(steps, noise.device, noise.dtype)
    y = noise
    for step in range(steps):
        dt = t[step + 1] - t[step]
        pred_cfg = dit.forward_with_cfg(x=y, t=t[step], c=c, latent_history=latent_history, patch_size=patch_size)
        pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
        y = y + (pred + (pred - null_pred) * cfg_scale) * dt
        if temperature != 0.0:
            if sde_noise is None:
                raise ValueError('temperature > 0 needs sde_noise')
            y = y + sigma * (temperature ** 0.5) * (dt.abs() ** 0.5) * sde_noise[step]
    return y
