"""The Audio8 language model at static shapes: one slot per row, every step graph-safe.

Same math as ``ArkttsModel.generate()``, split the way the engine loop needs it:

- ``sample``: from the logits of the slow transformer to one frame per row. Semantic token with repetition-aware
  sampling, then the fast transformer for the other codebooks. The frame goes to ``frames`` and is embedded for the
  next ``slow`` call.
- ``slow``: one slow transformer step per row over a fixed key length, with a per-row position instead of the row
  masks and left padding of the batched ``generate()``.
- ``prefill``: the text part of the prompt for up to ``rows`` rows. The head of the prompt (system turn, reference
  transcript, reference codes) is the same for every request of a voice, so its KV is computed once per voice
  (``voice_kv``) and only copied into the slot; the attention then starts at the text.

Sampling rules, as in ``generate()``: top-k and top-p on the softmax of the scores, temperature, then a draw, with a
second draw at (ras_top_p, ras_temperature) that replaces the first when it repeats a token of the last
``ras_window_size`` frames. The draw is argmax(log p + gumbel) over the top-k, with the row's own noise (``noise.py``)
instead of one generator shared by all rows.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from .noise import gumbel


class SlotLM:
    def __init__(self, model: Any, *, slots: int, length: int, top_k: int, top_p: float, temperature: float,
                 device: torch.device, dtype: torch.dtype) -> None:
        c = model.config
        self.model, self.config = model, c
        self.S, self.L = int(slots), int(length)
        self.device, self.dtype = device, dtype
        self.top_k, self.top_p, self.temperature = int(top_k), float(top_p), max(float(temperature), 1e-5)
        self.books = c.num_codebooks
        self.layers, self.fast_layers = self._slow_layers(), list(model.fast_layers)
        self.eos_index = c.codebook_size  # position of EOS in the (codebook_size + 1) scores of a frame
        self.scale = 1.0 / math.sqrt(c.head_dim)
        self.fast_scale = 1.0 / math.sqrt(c.fast_head_dim)
        S, L, d = self.S, self.L, device
        # One row more than slots: padding rows of a prefill call write there.
        self.K = torch.zeros(len(self.layers), S + 1, c.n_local_heads, L, c.head_dim, device=d, dtype=dtype)
        self.V = torch.zeros_like(self.K)
        self.fk = [torch.zeros(S, c.fast_n_local_heads, c.num_codebooks, c.fast_head_dim, device=d, dtype=dtype)
                   for _ in self.fast_layers]
        self.fv = [torch.zeros_like(t) for t in self.fk]
        self.logits = torch.zeros(S + 1, self.eos_index + 1, device=d, dtype=torch.float32)
        self.hidden = torch.zeros(S + 1, 1, c.dim, device=d, dtype=dtype)
        self.next_x = torch.zeros(S, 1, c.dim, device=d, dtype=dtype)
        self.frames = torch.zeros(S, L, self.books, device=d, dtype=torch.long)
        self.prev = torch.zeros(S, c.ras_window_size, device=d, dtype=torch.long)
        self.eos = torch.zeros(S, device=d, dtype=torch.bool)
        self.pos = torch.zeros(S, device=d, dtype=torch.long)
        self.step = torch.zeros(S, device=d, dtype=torch.long)
        self.key = torch.zeros(S, device=d, dtype=torch.long)
        self.window_base = c.semantic_begin_id  # the repetition window holds token ids
        self.score_weight = self._score_weight()
        self.eos_token = torch.full((S,), c.eos_token_id, device=d, dtype=torch.long)
        self.codebook_offsets = torch.arange(self.books, device=d) * c.codebook_size
        self.slot_index = torch.arange(S, device=d)
        self.key_index = torch.arange(L, device=d)
        self.first_keep = torch.zeros(S, self.top_k, device=d, dtype=torch.bool)
        self.first_keep[:, 0] = True
        self.draws = (self.books + 1) * self.top_k
        self.draw_index = torch.arange(self.draws, device=d).view(1, self.books + 1, self.top_k)
        self.cos, self.sin = self._rope_table()
        fast = model.fast_freqs_cis.float()
        self.fast_cos, self.fast_sin = fast[..., 0], fast[..., 1]

    # -- what the slow branch of a checkpoint decides --------------------------------------------------------------------

    def _slow_layers(self) -> list[Any]:
        return list(self.model.layers)

    def _score_weight(self) -> torch.Tensor:
        """(dim, scores): the slow branch's output head, one column per score of a frame."""
        c = self.config
        w = self.model.embeddings.weight
        return torch.cat((w[c.semantic_begin_id:c.semantic_end_id + 1], w[c.eos_token_id:c.eos_token_id + 1])).detach().t()

    def _rope_table(self) -> tuple[torch.Tensor, torch.Tensor]:
        rope = self.model.freqs_cis.float()
        return rope[..., 0], rope[..., 1]

    # -- slot state ----------------------------------------------------------------------------------------------------

    def reset(self) -> None:
        self.K.zero_()
        self.V.zero_()
        self.prev.zero_()

    def load_voice(self, slot: int, keys: torch.Tensor, values: torch.Tensor, state: Any) -> None:
        """Put a voice's cached prompt head (what ``voice_kv`` returned) into a slot."""
        P = keys.shape[2]
        self.K[:, slot, :, :P] = keys
        self.V[:, slot, :, :P] = values

    # -- shared pieces -----------------------------------------------------------------------------------------------

    @staticmethod
    def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        shaped = x.float().reshape(*x.shape[:-1], -1, 2)
        a, b = shaped[..., 0], shaped[..., 1]
        return torch.stack((a * cos - b * sin, b * cos + a * sin), dim=-1).flatten(3).to(x.dtype)

    @staticmethod
    def _qkv(attention: Any, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        kv_size = attention.n_local_heads * attention.head_dim
        return attention.wqkv(x).split((attention.n_head * attention.head_dim, kv_size, kv_size), dim=-1)

    def _pick(self, scores: torch.Tensor, noise: torch.Tensor, top_p: float, temperature: float) -> torch.Tensor:
        """Index of one draw per row: scores (B, N), noise (B, top_k) gumbel. Scores are sorted by topk, which is all
        of ArkttsLegacyTopKTopPLogitsProcessor; only the cumulative mass needs the softmax over all N."""
        values, index = scores.topk(self.top_k, dim=-1)
        mass = torch.softmax(scores, dim=-1).gather(1, index)
        keep = (mass.cumsum(-1) <= top_p) | self.first_keep[:scores.shape[0]]
        filtered = values.masked_fill(~keep, -math.inf) / temperature
        return index.gather(1, (filtered + noise).argmax(-1, keepdim=True))[:, 0]

    # -- sample ------------------------------------------------------------------------------------------------------

    def _fast_step(self, h: torch.Tensor, i: int, bk: int) -> torch.Tensor:
        cos, sin = self.fast_cos[i], self.fast_sin[i]
        for n, layer in enumerate(self.fast_layers):
            attention = layer.attention
            q, k, v = self._qkv(attention, layer.attention_norm(h))
            q = self._rotate(q.view(bk, 1, attention.n_head, attention.head_dim), cos, sin)
            k = self._rotate(k.view(bk, 1, attention.n_local_heads, attention.head_dim), cos, sin)
            ck, cv = self.fk[n], self.fv[n]
            ck[:bk, :, i].copy_(k[:, 0])
            cv[:bk, :, i].copy_(v.view(bk, attention.n_local_heads, attention.head_dim))
            groups = attention.n_head // attention.n_local_heads
            grouped = q.view(bk, attention.n_local_heads, groups, attention.head_dim)
            scores = grouped @ ck[:bk, :, :i + 1].transpose(-2, -1) * self.fast_scale
            out = torch.softmax(scores, dim=-1) @ cv[:bk, :, :i + 1]
            h = h + attention.wo(out.view(bk, 1, attention.n_head * attention.head_dim))
            h = h + layer.feed_forward(layer.ffn_norm(h))
        return h

    def sample(self, bk: int) -> None:
        """One frame per slot in [0, bk) from ``logits`` / ``hidden``. Graph-safe."""
        m, c = self.model, self.config
        step = self.step[:bk]
        started = step > 0
        noise = gumbel((self.key[:bk] + step * self.draws).view(bk, 1, 1) + self.draw_index)
        logits = self.logits[:bk]
        normal = self._pick(logits, noise[:, 0], self.top_p, self.temperature)
        high = self._pick(logits, noise[:, 1], c.ras_top_p, max(c.ras_temperature, 1e-5))
        repeated = (self.prev[:bk] == (normal + self.window_base)[:, None]).any(1) & started
        chosen = torch.where(repeated & (normal != self.eos_index), high, normal)
        ended = chosen == self.eos_index
        semantic = torch.where(ended, self.eos_token[:bk], chosen + c.semantic_begin_id)
        self._fast_step(m.fast_project_in(self.hidden[:bk]), 0, bk)
        current = chosen.clamp_max(c.codebook_size - 1)
        books = [current]
        h = m.fast_embeddings(current)[:, None]
        for i in range(1, self.books):
            scores = m.fast_output(m.fast_norm(self._fast_step(h, i, bk)))[:, -1].float()
            current = self._pick(scores, noise[:, i + 1], self.top_p, self.temperature)
            books.append(current)
            h = m.fast_embeddings(current)[:, None]
        codes = torch.stack(books, dim=1)
        self.frames[self.slot_index[:bk], step] = codes
        self.eos[:bk].copy_(ended)
        x = m.embeddings(semantic) + m.codebook_embeddings(codes + self.codebook_offsets).sum(1)
        self.next_x[:bk].copy_(x[:, None])
        # Upstream's window stays empty after a row's first frame. (A row ends on EOS, so what the window would hold for
        # EOS is never read.)
        rolled = torch.cat((self.prev[:bk, 1:], (chosen + self.window_base)[:, None]), dim=1)
        self.prev[:bk].copy_(torch.where(started[:, None], rolled, torch.zeros_like(rolled)))

    # -- slow step ---------------------------------------------------------------------------------------------------

    def _step_mask(self, pos: torch.Tensor) -> torch.Tensor:
        bk = pos.shape[0]
        return torch.zeros(bk, 1, 1, self.L, device=self.device, dtype=torch.float32).masked_fill(
            (self.key_index[None] > pos[:, None]).view(bk, 1, 1, self.L), -math.inf)

    def _attend(self, n: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, index: torch.Tensor, pos: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """One query per row over the row's keys, after writing its column at pos: q and k (B, 1, heads, head_dim)
        rotated, v (B, kv heads, head_dim). Scores and softmax run in fp32: bf16 cannot hold a distribution over 1024
        keys."""
        c = self.config
        bk, nl, head = q.shape[0], c.n_local_heads, c.head_dim
        groups = c.n_head // nl
        g = bk * nl
        ck, cv = self.K[n], self.V[n]
        ck[index, :, pos] = k[:, 0]
        cv[index, :, pos] = v
        scores = torch.bmm(q.reshape(g, groups, head), ck[:bk].view(g, self.L, head).transpose(1, 2),
                           out_dtype=torch.float32).view(bk, nl, groups, self.L)
        probs = torch.softmax(scores * self.scale + mask, dim=-1).to(cv.dtype)
        out = torch.bmm(probs.view(g, groups, self.L), cv[:bk].view(g, self.L, head))
        return out.reshape(bk, 1, c.n_head * head)

    def slow(self, bk: int) -> None:
        """One slow transformer step per slot in [0, bk): reads next_x and pos, writes the KV column at pos, logits and
        hidden. Graph-safe."""
        c = self.config
        pos = self.pos[:bk]
        index = self.slot_index[:bk]
        cos, sin = self.cos[pos][:, None, None], self.sin[pos][:, None, None]
        mask = self._step_mask(pos)
        nl, head = c.n_local_heads, c.head_dim
        x = self.next_x[:bk]
        for n, layer in enumerate(self.layers):
            attention = layer.attention
            q, k, v = self._qkv(attention, layer.attention_norm(x))
            q = self._rotate(q.view(bk, 1, c.n_head, head), cos, sin)
            k = self._rotate(k.view(bk, 1, nl, head), cos, sin)
            x = x + attention.wo(self._attend(n, q, k, v.view(bk, nl, head), index, pos, mask))
            x = x + layer.feed_forward(layer.ffn_norm(x))
        self._finish(x, index)

    def _finish(self, x: torch.Tensor, index: torch.Tensor) -> None:
        normed = self.model.norm(x)
        self.logits[index] = torch.bmm(normed.float().view(x.shape[0], 1, -1),
                                       self.score_weight.float().expand(x.shape[0], -1, -1))[:, 0]
        self.hidden[index] = normed if self.config.norm_fastlayer_input else x

    # -- prefill -----------------------------------------------------------------------------------------------------

    def prefill(self, ids: torch.Tensor, slots: torch.Tensor, start: torch.Tensor, last: torch.Tensor) -> None:
        """Text tokens ids (R, T) at positions start + [0, T) of slots. Graph-safe. Rows are right-padded: a query never
        sees later keys, and keys of padding positions stay above the row's position until a step overwrites them.
        Padding rows use the extra row of K / V. ``last`` picks each row's last real token for logits and hidden."""
        m, c = self.model, self.config
        R, T = ids.shape
        nl, head = c.n_local_heads, c.head_dim
        pos = (start[:, None] + torch.arange(T, device=ids.device)[None]).clamp(max=self.L - 1)
        mask = (self.key_index[None, None] <= pos[:, :, None])[:, None]
        cos, sin = self.cos[pos][:, :, None], self.sin[pos][:, :, None]
        index = pos[:, None, :, None].expand(R, nl, T, head)
        x = m.embeddings(ids)
        for n, layer in enumerate(self.layers):
            attention = layer.attention
            q, k, v = self._qkv(attention, layer.attention_norm(x))
            q = self._rotate(q.view(R, T, c.n_head, head), cos, sin).transpose(1, 2)
            k = self._rotate(k.view(R, T, nl, head), cos, sin).transpose(1, 2)
            v = v.view(R, T, nl, head).transpose(1, 2)
            keys = self.K[n].index_select(0, slots).scatter_(2, index, k)
            values = self.V[n].index_select(0, slots).scatter_(2, index, v)
            self.K[n].index_copy_(0, slots, keys)
            self.V[n].index_copy_(0, slots, values)
            out = F.scaled_dot_product_attention(q, keys, values, attn_mask=mask, enable_gqa=True)
            x = x + attention.wo(out.transpose(1, 2).reshape(R, T, c.n_head * head))
            x = x + layer.feed_forward(layer.ffn_norm(x))
        self._finish(x.gather(1, last.view(R, 1, 1).expand(R, 1, c.dim)), slots)

    # -- voice prefix ------------------------------------------------------------------------------------------------

    def voice_kv(self, prompt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, Any]:
        """KV of a prompt head (1, books + 1, P), layers first: (layers, heads, P, head_dim) each, and what else a slot
        needs to go on from there (nothing here). The model's own forward; the head has no target text, so this is what
        every request of the voice shares."""
        m = self.model
        P = prompt.shape[-1]
        m._setup_generation_caches(1, self.config.max_seq_len, self.dtype)
        try:
            ids = torch.arange(P, device=self.device)
            m._slow_step(prompt, ids, ids[None], torch.ones(1, P, device=self.device, dtype=torch.long))
            keys = torch.stack([layer.attention.kv_cache.keys[0, :, :P] for layer in m.layers])
            values = torch.stack([layer.attention.kv_cache.values[0, :, :P] for layer in m.layers])
        finally:
            for layer in [*m.layers, *m.fast_layers]:
                layer.attention.kv_cache = None
        return keys, values, None
