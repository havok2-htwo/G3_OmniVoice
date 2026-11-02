"""The Audio8 0.1B language model at static shapes: Falcon-H1 slow branch, slot state for attention and Mamba.

Everything else about this checkpoint is the 0.6B's (the fast branch, the codec, the prompt), so this only replaces the
slow branch of ``SlotLM``. A Falcon-H1 layer runs a Mamba2 mixer next to the attention, on the same normalized input, and
adds both to the residual. So a slot holds, per layer, besides the KV columns:

- ``conv``: the last d_conv - 1 inputs of the causal convolution that makes the mixer's x, B and C;
- ``ssm``: the state space, h[t] = exp(A dt[t]) h[t-1] + dt[t] B[t] x[t], read out as y[t] = C[t] h[t] + D x[t], per head.

Both are plain tensors, so a step stays one graph-safe call at a fixed shape; there is no per-token Python as in the
model's own cache. Same recurrence as Falcon's cached forward. The state and its read-out run in fp32, like Falcon's
prompt pass does (its cached step rounds the read-out to bf16).

``prefill`` advances many positions in one call, with the state-space recurrence written out as a matrix over the
positions of the call: a Python loop over them would be 24 layers times up to 128 small steps, per call. The sum is the
one Falcon evaluates per chunk of 128, here with the previous state as a start value and without the chunks.

Sampling is ``SlotLM.sample`` with two differences of this checkpoint's ``generate()``: its scores come from a head of
their own (``semantic_output``), and its repetition window holds score indices, not token ids. Upstream starts that
window with zeros, so index 0 counts as repeated from the second frame on; that stays. The fast branch always gets the
normalized hidden state, whatever ``norm_fastlayer_input`` says.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from .lm import SlotLM

_NEUTRAL_MULTIPLIERS = ('attention_in_multiplier', 'attention_out_multiplier', 'key_multiplier', 'ssm_in_multiplier',
                        'ssm_out_multiplier')


class FalconSlotLM(SlotLM):
    def __init__(self, model: Any, **kwargs: Any) -> None:
        c = model.config
        scaled = [name for name in _NEUTRAL_MULTIPLIERS if getattr(c, name) != 1.0]
        if scaled or set(c.ssm_multipliers) != {1.0} or set(c.mlp_multipliers) != {1.0}:
            raise NotImplementedError(f'Falcon-H1 multipliers are not implemented here: {scaled or "ssm/mlp"}')
        if c.mamba_rms_norm or c.mamba_n_groups != 1 or model.slow.layers[0].mamba.time_step_limit != (0.0, math.inf):
            raise NotImplementedError('Audio8 Mamba setup differs from the 0.1B checkpoint')
        super().__init__(model, **kwargs)
        S, dtype, device = self.S, self.dtype, self.device
        first = self.layers[0].mamba
        self.heads, self.state_size = first.num_heads, first.ssm_state_size
        self.x_size, self.head_size = first.intermediate_size, first.head_dim
        self.conv_size, self.past = first.conv_dim, first.conv_kernel_size - 1
        self.split = (self.x_size, self.conv_size, self.heads)
        self.mixer = (self.x_size, self.state_size, self.state_size)
        count = len(self.layers)
        self.conv = torch.zeros(count, S + 1, self.conv_size, self.past, device=device, dtype=dtype)
        self.ssm = torch.zeros(count, S + 1, self.heads, self.head_size, self.state_size, device=device,
                               dtype=torch.float32)
        self.embedding_scale = c.embedding_multiplier
        self.window_base = 0
        # fp32 copies of the parameters of the state space, one per layer.
        self.A = [(-torch.exp(layer.mamba.A_log.float())).detach() for layer in self.layers]
        self.D = [layer.mamba.D.float().detach()[:, None] for layer in self.layers]
        self.dt_bias = [layer.mamba.dt_bias.float().detach() for layer in self.layers]
        self.conv_weight = [layer.mamba.conv1d.weight.detach() for layer in self.layers]
        self.conv_bias = [layer.mamba.conv1d.bias.detach() for layer in self.layers]
        self.past_index = torch.arange(1, self.past + 1, device=device)

    # -- what the slow branch of this checkpoint decides ----------------------------------------------------------------

    def _slow_layers(self) -> list[Any]:
        return list(self.model.slow.layers)

    def _score_weight(self) -> torch.Tensor:
        return self.model.semantic_output.weight.detach().t()

    def _rope_table(self) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(self.L, device=self.device, dtype=torch.float32)
        phases = torch.outer(positions, self.model.slow.rotary_emb.inv_freq.float().to(self.device))
        return phases.cos(), phases.sin()

    @staticmethod
    def _rotate_half(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        a, b = x.float().chunk(2, dim=-1)  # Falcon's rotate_half: the two halves, not pairs as in the 0.6B
        return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1).to(x.dtype)

    def reset(self) -> None:
        super().reset()
        self.conv.zero_()
        self.ssm.zero_()

    def load_voice(self, slot: int, keys: torch.Tensor, values: torch.Tensor, state: Any) -> None:
        super().load_voice(slot, keys, values, state)
        self.conv[:, slot] = state[0]
        self.ssm[:, slot] = state[1]

    def _finish(self, x: torch.Tensor, index: torch.Tensor) -> None:
        normed = self.model.slow.final_layernorm(x)
        self.logits[index] = torch.bmm(normed.float().view(x.shape[0], 1, -1),
                                       self.score_weight.float().expand(x.shape[0], -1, -1))[:, 0]
        self.hidden[index] = normed

    # -- one position ----------------------------------------------------------------------------------------------------

    def _mamba_step(self, n: int, mamba: Any, h: torch.Tensor) -> torch.Tensor:
        """One position per slot of the layer's mixer: h (B, 1, dim). Reads and writes the slots' conv and ssm state."""
        bk = h.shape[0]
        gate, inputs, dt = mamba.in_proj(h)[:, 0].split(self.split, dim=-1)
        window = torch.cat((self.conv[n, :bk], inputs[:, :, None]), dim=-1)
        self.conv[n, :bk].copy_(window[:, :, 1:])
        mixed = F.silu((window * self.conv_weight[n][:, 0]).sum(-1) + self.conv_bias[n]).float()
        x, B, C = mixed.split(self.mixer, dim=-1)
        x = x.view(bk, self.heads, self.head_size)
        step = F.softplus(dt.float() + self.dt_bias[n]).view(bk, self.heads, 1, 1)
        state = self.ssm[n, :bk] * torch.exp(step * self.A[n][:, None, None]) + step * x[..., None] * B[:, None, None]
        self.ssm[n, :bk].copy_(state)
        y = (state * C[:, None, None]).sum(-1) + x * self.D[n]
        return mamba.out_proj((y.view(bk, self.x_size) * F.silu(gate.float())).to(h.dtype))[:, None]

    def slow(self, bk: int) -> None:
        """One slow step per slot in [0, bk): reads next_x and pos, writes KV at pos, conv and ssm state, logits and
        hidden. Graph-safe."""
        c = self.config
        pos = self.pos[:bk]
        index = self.slot_index[:bk]
        cos, sin = self.cos[pos][:, None, None], self.sin[pos][:, None, None]
        mask = self._step_mask(pos)
        nl, head = c.n_local_heads, c.head_dim
        x = self.next_x[:bk] * self.embedding_scale
        for n, layer in enumerate(self.layers):
            attention = layer.self_attn
            h = layer.input_layernorm(x)
            mixed = self._mamba_step(n, layer.mamba, h)
            q = self._rotate_half(attention.q_proj(h).view(bk, 1, c.n_head, head), cos, sin)
            k = self._rotate_half(attention.k_proj(h).view(bk, 1, nl, head), cos, sin)
            attended = attention.o_proj(self._attend(n, q, k, attention.v_proj(h).view(bk, nl, head), index, pos, mask))
            x = x + mixed + attended
            x = x + layer.feed_forward(layer.pre_ff_layernorm(x))
        self._finish(x, index)

    # -- many positions --------------------------------------------------------------------------------------------------

    def _mamba_run(self, n: int, mamba: Any, h: torch.Tensor, conv: torch.Tensor, ssm: torch.Tensor,
                   last: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """T positions per row, h (R, T, dim), from the rows' conv and ssm state. Returns the output and the state after
        position ``last`` of each row. Positions behind ``last`` are padding; being causal they change nothing before it."""
        R, T = h.shape[:2]
        H = self.heads
        gate, inputs, dt = mamba.in_proj(h).split(self.split, dim=-1)
        window = torch.cat((conv, inputs.transpose(1, 2)), dim=-1)
        ends = (last[:, None, None] + self.past_index[None, None]).expand(R, self.conv_size, self.past)
        next_conv = window.gather(2, ends)
        mixed = F.silu(F.conv1d(window, self.conv_weight[n], self.conv_bias[n], groups=self.conv_size)).float()
        x, B, C = mixed.transpose(1, 2).split(self.mixer, dim=-1)
        x = x.view(R, T, H, self.head_size).transpose(1, 2)                             # (R, H, T, head_size)
        step = F.softplus(dt.float() + self.dt_bias[n]).transpose(1, 2)                 # (R, H, T)
        total = (step * self.A[n][None, :, None]).cumsum(-1)
        # decay[t, s] = exp(A (dt[s + 1] + ... + dt[t])): what state s has left at position t.
        decay = torch.exp(total[:, :, :, None] - total[:, :, None, :]).tril()
        weighted = x * step[..., None]
        from_start = torch.exp(total)                                                   # what the start state has left
        y = (decay * (C @ B.transpose(1, 2))[:, None]) @ weighted
        y = y + from_start[..., None] * (ssm @ C.transpose(1, 2)[:, None]).transpose(2, 3) + x * self.D[n][:, None]
        out = (y.transpose(1, 2).reshape(R, T, self.x_size) * F.silu(gate.float())).to(h.dtype)
        row = last.view(R, 1, 1, 1).expand(R, H, 1, T)
        kept = decay.gather(2, row)                                                     # (R, H, 1, T)
        state = (weighted.transpose(2, 3) * kept) @ B[:, None]
        state = state + from_start.gather(2, last.view(R, 1, 1).expand(R, H, 1))[..., None] * ssm
        return mamba.out_proj(out), next_conv, state

    def _layer_run(self, n: int, layer: Any, x: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                   conv: torch.Tensor, ssm: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, index: torch.Tensor,
                   mask: torch.Tensor, last: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """One layer over T positions: x (R, T, dim), keys and values (R, heads, L, head_dim) of the rows."""
        c = self.config
        R, T = x.shape[:2]
        nl, head = c.n_local_heads, c.head_dim
        attention = layer.self_attn
        h = layer.input_layernorm(x)
        mixed, conv, ssm = self._mamba_run(n, layer.mamba, h, conv, ssm, last)
        q = self._rotate_half(attention.q_proj(h).view(R, T, c.n_head, head), cos, sin).transpose(1, 2)
        k = self._rotate_half(attention.k_proj(h).view(R, T, nl, head), cos, sin).transpose(1, 2)
        keys = keys.scatter(2, index, k)
        values = values.scatter(2, index, attention.v_proj(h).view(R, T, nl, head).transpose(1, 2))
        out = F.scaled_dot_product_attention(q.float(), keys.float(), values.float(), attn_mask=mask, enable_gqa=True)
        x = x + mixed + attention.o_proj(out.to(x.dtype).transpose(1, 2).reshape(R, T, c.n_head * head))
        return x + layer.feed_forward(layer.pre_ff_layernorm(x)), keys, values, conv, ssm

    def _positions(self, start: torch.Tensor, T: int, L: int) -> tuple[torch.Tensor, ...]:
        R = start.shape[0]
        c = self.config
        pos = (start[:, None] + torch.arange(T, device=start.device)[None]).clamp(max=L - 1)
        mask = (self.key_index[:L][None, None] <= pos[:, :, None])[:, None]
        cos, sin = self.cos[pos][:, :, None], self.sin[pos][:, :, None]
        return cos, sin, pos[:, None, :, None].expand(R, c.n_local_heads, T, c.head_dim), mask

    def prefill(self, ids: torch.Tensor, slots: torch.Tensor, start: torch.Tensor, last: torch.Tensor) -> None:
        """Text tokens ids (R, T) at positions start + [0, T) of slots, on top of the state of the slots. Graph-safe.
        Same contract as ``SlotLM.prefill``; padding rows use the extra row of the state."""
        R, T = ids.shape
        cos, sin, index, mask = self._positions(start, T, self.L)
        x = self.model.embeddings(ids) * self.embedding_scale
        for n, layer in enumerate(self.layers):
            x, keys, values, conv, ssm = self._layer_run(
                n, layer, x, self.K[n].index_select(0, slots), self.V[n].index_select(0, slots),
                self.conv[n].index_select(0, slots), self.ssm[n].index_select(0, slots), cos, sin, index, mask, last)
            self.K[n].index_copy_(0, slots, keys)
            self.V[n].index_copy_(0, slots, values)
            self.conv[n].index_copy_(0, slots, conv)
            self.ssm[n].index_copy_(0, slots, ssm)
        self._finish(x.gather(1, last.view(R, 1, 1).expand(R, 1, x.shape[2])), slots)

    def voice_kv(self, prompt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, Any]:
        """The state of the slots after a prompt head (1, books + 1, P): KV (layers, heads, P, head_dim), and the Mamba
        state as a pair (conv, ssm), layers first."""
        c = self.config
        P = prompt.shape[-1]
        device, dtype = self.device, self.dtype
        x = self.model._embed(prompt) * self.embedding_scale
        cos, sin, index, mask = self._positions(torch.zeros(1, device=device, dtype=torch.long), P, P)
        last = torch.tensor([P - 1], device=device)
        keys, values, convs, states = [], [], [], []
        for n, layer in enumerate(self.layers):
            empty = torch.zeros(1, c.n_local_heads, P, c.head_dim, device=device, dtype=dtype)
            x, k, v, conv, ssm = self._layer_run(
                n, layer, x, empty, empty, torch.zeros(1, self.conv_size, self.past, device=device, dtype=dtype),
                torch.zeros(1, self.heads, self.head_size, self.state_size, device=device), cos, sin, index, mask, last)
            keys.append(k[0])
            values.append(v[0])
            convs.append(conv[0])
            states.append(ssm[0])
        return torch.stack(keys), torch.stack(values), (torch.stack(convs), torch.stack(states))
