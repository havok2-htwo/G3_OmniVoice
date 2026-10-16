from __future__ import annotations

import types

import pytest
import torch

from omnivoice_tts_server import packed_decode

ov = pytest.importorskip('omnivoice.models.omnivoice')


class TinyOmni:
    """Stand-in exposing what _generate_iterative uses. Its forward is per-row and attention-free except for a
    masked mean over the row's visible keys, so a packed row must reproduce the stock rows exactly if the layout,
    the masks and the positions are right."""

    def __init__(self) -> None:
        self.config = types.SimpleNamespace(num_audio_codebook=2, audio_mask_id=7, audio_vocab_size=8)
        self.device = torch.device('cpu')
        torch.manual_seed(0)
        self.emb = torch.randn(64, 4, dtype=torch.float64)
        self.pos = torch.randn(512, 4, dtype=torch.float64)
        self.head = torch.randn(4, 2 * 8, dtype=torch.float64)
        self.inputs: dict[str, torch.Tensor] = {}

    def _prepare_inference_inputs(self, text, target, ref_text, ref_audio, lang, instruct, denoise):
        cond_len = 3 + len(text)
        ids = torch.tensor([[[(hash(text) + j) % 7 for j in range(cond_len - target)] + [7] * target] * 2])
        mask = torch.zeros(1, cond_len, dtype=torch.bool)
        mask[0, cond_len - target:] = True
        return {'input_ids': ids, 'audio_mask': mask}

    def __call__(self, input_ids, audio_mask, attention_mask, position_ids=None):
        B, C, S = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(S).repeat(B, 1)
        x = self.emb[input_ids.sum(1) % 64] + self.pos[position_ids]          # [B, S, 4]
        m = attention_mask[:, 0].to(torch.float64)                             # [B, S, S]
        ctx = (m @ x) / m.sum(-1, keepdim=True).clamp(min=1)                   # masked mean over visible keys
        logits = (ctx @ self.head).view(B, S, 2, 8).permute(0, 2, 1, 3)
        return types.SimpleNamespace(logits=logits)

    _predict_tokens_with_scoring = ov.OmniVoice._predict_tokens_with_scoring


def _task(texts, targets):
    return types.SimpleNamespace(
        batch_size=len(texts), texts=texts, target_lens=targets, ref_texts=[None] * len(texts),
        ref_audio_tokens=[None] * len(texts), langs=[None] * len(texts), instructs=[None] * len(texts),
    )


def test_packed_loop_matches_stock_layout_exactly_on_a_per_row_model() -> None:
    config = ov.OmniVoiceGenerationConfig(num_step=6, guidance_scale=2.0, position_temperature=0.0)
    task = _task(['abcdefgh', 'xyz', 'hello world!'], [5, 3, 6])
    stock_model, packed_model = TinyOmni(), TinyOmni()
    stock = ov.OmniVoice._generate_iterative(stock_model, task, config)
    assert packed_decode.install(packed_model)
    packed = packed_model._generate_iterative(task, config)
    assert [t.shape for t in stock] == [t.shape for t in packed]
    assert all(torch.equal(a, b) for a, b in zip(stock, packed))
    packed_decode.uninstall(packed_model)
    assert not packed_decode.is_installed(packed_model)
