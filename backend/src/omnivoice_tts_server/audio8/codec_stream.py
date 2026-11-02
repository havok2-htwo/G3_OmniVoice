"""Batched, slot-based streaming decoder (graph-safe) for the Audio8 codec.

The codec decode is causal from end to end, so it streams exactly: same math as ``ArkttsCodec.decode`` run over the
whole fragment, restated with static shapes and per-slot state. One call decodes up to ``chunk`` code frames per row
and needs no overlap and no re-run of old frames:

- Codes: one fused table holds ``out_proj(codebook)`` of all ten quantizers, so a frame is ten lookups and a sum (one
  rounding, in fp32, instead of ten embeddings plus ten 1x1 convolutions).
- ``post_module``: eight causal transformer layers with a 128-frame window. Stacked, they reach about 1000 frames back,
  which is why re-running a window of frames only approximates a full decode. Here every layer keeps its KV of the last
  128 frames in a ring per slot (each key carries its absolute position), so a new frame sees exactly what it sees in
  a full decode, for the price of its own queries.
- Upsampler and waveform decoder: stride-1 causal convolutions keep their left context per slot; transposed
  convolutions keep their last input frame. The audio of a frame therefore does not depend on the chunk it was
  decoded in.

Rows and frames in a fixed-shape call can be padding. ``meta = (slot, first_frame, valid_frames, start)`` per row: frames
beyond ``valid_frames`` run through the same kernels but, everything being causal, cannot touch the valid audio; their
KV goes to a trash slot and the context saved for the next call is taken at the end of the valid frames. ``start``
clears the row's state (new fragment). Padding rows use the trash slot and ``valid_frames = 0``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

_NEG = -(1 << 40)
WINDOW = 128  # ArkttsDownsampleQuantizer.post_module window_size


@dataclass
class _Conv:
    weight: torch.Tensor
    bias: torch.Tensor | None
    dilation: int
    groups: int
    pad: int                     # left context in samples
    rate: int                    # samples at this stage per code frame
    state: torch.Tensor | None   # (slots + 1, channels, pad)


@dataclass
class _Transpose:
    weight: torch.Tensor
    bias: torch.Tensor | None
    stride: int
    rate: int                    # input samples per code frame
    state: torch.Tensor | None   # (slots + 1, channels, 1); None when the kernel has no overlap (kernel == stride)


class SlotCodec:
    def __init__(self, codec: Any, slots: int, max_chunk: int, max_frames: int, device: torch.device,
                 dtype: torch.dtype) -> None:
        self.dtype = dtype
        self.device = device
        self.slots = int(slots)
        self.trash = self.slots
        self.frame_samples = int(codec.frame_length)
        self.max_chunk = int(max_chunk)
        self.max_frames = int(max_frames)
        self._n_states = self.slots + 1
        q = codec.quantizer
        tables, sizes = [], []
        for vq in [*q.semantic_quantizer.quantizers, *q.quantizer.quantizers]:
            sizes.append(vq.codebook_size)
            tables.append(vq.out_proj(vq.codebook.weight.t()[None]).squeeze(0).t().float())
        self.table = torch.cat(tables)
        self.table_offset = torch.tensor([sum(sizes[:i]) for i in range(len(sizes))], device=device)
        self.code_max = torch.tensor(sizes, device=device) - 1
        self.n_codebooks = len(sizes)
        self._init_transformer(q.post_module)
        self._init_convs(q, codec.decoder)
        self.frame_index = torch.arange(self.max_chunk, device=device)
        self.sample_index = torch.arange(self.max_chunk * self.frame_samples, device=device)

    # -- construction ------------------------------------------------------------------------------------------------

    def _init_transformer(self, module: Any) -> None:
        self.tf_layers = module.layers
        self.tf_norm = module.norm
        first = module.layers[0].attention
        self.n_head, self.n_kv, self.head_dim = first.n_head, first.n_local_heads, first.head_dim
        self.hidden = self.n_head * self.head_dim
        self.ring = -(-(WINDOW + self.max_chunk) // 8) * 8
        shape = (self._n_states, self.n_kv, self.ring, self.head_dim)
        self.k = [torch.zeros(shape, device=self.device, dtype=self.dtype) for _ in module.layers]
        self.v = [torch.zeros(shape, device=self.device, dtype=self.dtype) for _ in module.layers]
        self.kpos = torch.full((self._n_states, self.ring), _NEG, device=self.device, dtype=torch.long)
        # Same table a full decode builds per call; the last entry serves padding frames beyond the budget.
        frequencies = 1.0 / (module.rope_base ** (torch.arange(0, self.head_dim, 2, device=self.device).float()
                                                  / self.head_dim))
        length = self.max_frames + self.max_chunk
        phases = torch.outer(torch.arange(length, device=self.device).float(), frequencies)
        values = torch.polar(torch.ones_like(phases), phases)
        table = torch.stack((values.real, values.imag), dim=-1).to(torch.bfloat16).float()
        self.rope_cos, self.rope_sin = table[..., 0], table[..., 1]
        self.rope_last = length - 1
        self.gamma = [(layer.attention_layer_scale.gamma, layer.ffn_layer_scale.gamma) for layer in module.layers]

    def _conv(self, conv: Any, rate: int) -> _Conv:
        inner = conv.conv
        pad = conv.padding
        state = None
        if pad:
            state = torch.zeros(self._n_states, inner.in_channels, pad, device=self.device, dtype=self.dtype)
        return _Conv(inner.weight.detach().clone(), None if inner.bias is None else inner.bias.detach().clone(),
                     inner.dilation[0], inner.groups, pad, rate, state)

    def _transpose(self, conv: Any, rate: int) -> _Transpose:
        inner = conv.conv
        overlap = conv.kernel_size - conv.stride
        if overlap not in (0, conv.stride):
            raise NotImplementedError(f'unexpected transposed convolution {conv.kernel_size}/{conv.stride}')
        state = None
        if overlap:
            state = torch.zeros(self._n_states, inner.in_channels, 1, device=self.device, dtype=self.dtype)
        return _Transpose(inner.weight.detach().clone(), None if inner.bias is None else inner.bias.detach().clone(),
                          conv.stride, rate, state)

    def _init_convs(self, quantizer: Any, decoder: Any) -> None:
        self.states: list[torch.Tensor] = []
        self.up = []
        rate = 1
        for stage in quantizer.upsample:
            transpose, block = stage[0], stage[1]
            step = self._transpose(transpose, rate)
            rate *= transpose.stride
            self.up.append((step, self._conv(block.dwconv, rate), block))
        layers = list(decoder.model)
        self.first = self._conv(layers[0], rate)
        self.blocks = []
        for block in layers[1:-3]:
            parts = list(block.block)
            snake, transpose, units = parts[0], parts[1], parts[2:]
            step = self._transpose(transpose, rate)
            rate *= transpose.stride
            residual = []
            for unit in units:
                seq = list(unit.block)
                residual.append(((seq[0].alpha, self._conv(seq[1], rate)), (seq[2].alpha, self._conv(seq[3], rate))))
            self.blocks.append((snake.alpha, step, residual))
        self.last_alpha = layers[-3].alpha
        self.last = self._conv(layers[-2], rate)
        self.rate = rate
        if rate != self.frame_samples:
            raise NotImplementedError(f'codec frame {self.frame_samples} != decoder rate {rate}')
        for conv in self._walk():
            if conv.state is not None:
                self.states.append(conv.state)
        # The last convolution runs in fp32: the audio is not worth rounding to bf16's 8 bits of mantissa.
        self.last.weight = self.last.weight.float()
        self.last.bias = None if self.last.bias is None else self.last.bias.float()
        self.last.state = self.last.state.float() if self.last.state is not None else None
        self.states = [state for state in self.states if state.dtype == self.dtype] + (
            [self.last.state] if self.last.state is not None else [])

    def _walk(self):
        for step, conv, _ in self.up:
            yield step
            yield conv
        yield self.first
        for _, step, residual in self.blocks:
            yield step
            for (_, a), (_, b) in residual:
                yield a
                yield b
        yield self.last

    # -- state -------------------------------------------------------------------------------------------------------

    def reset(self) -> None:
        self.kpos.fill_(_NEG)
        for state in self.states:
            state.zero_()

    def out_len(self, chunk: int) -> int:
        return chunk * self.frame_samples

    # -- one call ----------------------------------------------------------------------------------------------------

    def _context(self, x: torch.Tensor, state: torch.Tensor, pad: int, rate: int, read: torch.Tensor,
                 write: torch.Tensor, keep: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """x with its left context prepended; stores the context of the next call (the end of the valid samples)."""
        past = state.index_select(0, read) * keep
        full = torch.cat((past, x), dim=-1)
        index = (valid * rate)[:, None, None] + torch.arange(pad, device=x.device)[None, None, :]
        state.index_copy_(0, write, full.gather(2, index.expand(-1, x.shape[1], -1)))
        return full

    def _apply_conv(self, x: torch.Tensor, conv: _Conv, read, write, keep, valid) -> torch.Tensor:
        if conv.pad:
            x = self._context(x, conv.state, conv.pad, conv.rate, read, write, keep, valid)
        return F.conv1d(x, conv.weight, conv.bias, dilation=conv.dilation, groups=conv.groups)

    def _apply_transpose(self, x: torch.Tensor, step: _Transpose, read, write, keep, valid) -> torch.Tensor:
        if step.state is None:
            return F.conv_transpose1d(x, step.weight, step.bias, stride=step.stride)
        x = self._context(x, step.state, 1, step.rate, read, write, keep, valid)
        return F.conv_transpose1d(x, step.weight, step.bias, stride=step.stride)[..., step.stride:-step.stride]

    @staticmethod
    def _snake(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        return x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)

    def _rotate(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        shaped = x.float().reshape(*x.shape[:-1], -1, 2)
        a, b = shaped[..., 0], shaped[..., 1]
        return torch.stack((a * cos - b * sin, b * cos + a * sin), dim=-1).flatten(3).to(x.dtype)

    def _transformer(self, x: torch.Tensor, pos: torch.Tensor, slot: torch.Tensor, write: torch.Tensor,
                     keep: torch.Tensor) -> torch.Tensor:
        R, T, _ = x.shape
        ring_index = pos % self.ring
        kept = torch.where(keep[:, 0] > 0, self.kpos[slot], self.kpos[slot].new_full((), _NEG))
        self.kpos[slot] = kept
        self.kpos[write, ring_index] = pos
        kp = self.kpos[slot]
        mask = (kp[:, None, :] <= pos[:, :, None]) & (kp[:, None, :] > pos[:, :, None] - WINDOW)
        n_rep = self.n_head // self.n_kv
        mask = mask[:, None, None].expand(R, 1, n_rep, T, self.ring).reshape(R, 1, n_rep * T, self.ring)
        at = pos.clamp(max=self.rope_last)
        cos, sin = self.rope_cos[at][:, :, None], self.rope_sin[at][:, :, None]
        for i, layer in enumerate(self.tf_layers):
            attention = layer.attention
            y = attention.wqkv(layer.attention_norm(x))
            qh, kh, vh = y.split((self.hidden, self.n_kv * self.head_dim, self.n_kv * self.head_dim), dim=-1)
            qh = self._rotate(qh.view(R, T, self.n_head, self.head_dim), cos, sin)
            kh = self._rotate(kh.view(R, T, self.n_kv, self.head_dim), cos, sin)
            vh = vh.view(R, T, self.n_kv, self.head_dim)
            self.k[i][write, :, ring_index] = kh
            self.v[i][write, :, ring_index] = vh
            # GQA without repeating K/V: the n_rep query heads of one KV group become its query axis.
            grouped = qh.transpose(1, 2).reshape(R, self.n_kv, n_rep * T, self.head_dim)
            out = F.scaled_dot_product_attention(grouped, self.k[i][slot], self.v[i][slot], attn_mask=mask)
            out = out.reshape(R, self.n_head, T, self.head_dim).transpose(1, 2).reshape(R, T, self.hidden)
            hidden = x + attention.wo(out) * self.gamma[i][0]
            x = hidden + layer.feed_forward(layer.ffn_norm(hidden)) * self.gamma[i][1]
        return self.tf_norm(x)

    def run(self, codes: torch.Tensor, meta: torch.Tensor, out: torch.Tensor) -> None:
        """codes (R, T, books) long, meta (R, 4) long, out (R, T * frame_samples) fp32. Graph-safe."""
        R, T = codes.shape[0], codes.shape[1]
        slot, first, valid, start = meta[:, 0], meta[:, 1], meta[:, 2], meta[:, 3]
        keep = (1 - start).to(self.dtype)[:, None, None]
        is_valid = self.frame_index[:T][None, :] < valid[:, None]
        pos = first[:, None] + self.frame_index[:T][None, :]
        write = torch.where(valid > 0, slot, torch.full_like(slot, self.trash))
        write_frames = torch.where(is_valid, slot[:, None], torch.full_like(pos, self.trash))
        indices = torch.minimum(codes, self.code_max) + self.table_offset
        x = self.table[indices].sum(dim=2).to(self.dtype)
        x = self._transformer(x, pos, slot, write_frames, keep).transpose(1, 2)
        ctx = (slot, write, keep, valid)
        for step, conv, block in self.up:
            x = self._apply_transpose(x, step, *ctx)
            residual = x
            y = self._apply_conv(x, conv, *ctx).permute(0, 2, 1)
            y = block.pwconv2(block.act(block.pwconv1(block.norm(y))))
            x = residual + (block.gamma * y).permute(0, 2, 1)
        x = self._apply_conv(x, self.first, *ctx)
        for alpha, step, residual_units in self.blocks:
            x = self._apply_transpose(self._snake(x, alpha), step, *ctx)
            for (alpha_a, conv_a), (alpha_b, conv_b) in residual_units:
                y = self._apply_conv(self._snake(x, alpha_a), conv_a, *ctx)
                x = x + self._apply_conv(self._snake(y, alpha_b), conv_b, *ctx)
        x = self._snake(x.float(), self.last_alpha.float())
        full = torch.cat((self.last.state.index_select(0, slot) * keep.float(), x), dim=-1)
        index = (valid * self.last.rate)[:, None, None] + torch.arange(self.last.pad, device=x.device)[None, None, :]
        self.last.state.index_copy_(0, write, full.gather(2, index.expand(-1, x.shape[1], -1)))
        out.copy_(torch.tanh(F.conv1d(full, self.last.weight, self.last.bias))[:, 0])


def snr_db(reference: Any, other: Any) -> float:
    """Signal-to-difference ratio of two equally long signals, in dB."""
    error = float(((reference - other) ** 2).sum())
    power = float((reference ** 2).sum())
    return math.inf if error == 0 else 10.0 * math.log10(max(power, 1e-30) / error)
