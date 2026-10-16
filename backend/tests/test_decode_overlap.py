from __future__ import annotations

import types

import pytest
import torch

from omnivoice_tts_server.decode_overlap import overlapped_decode


class FakeTokenizer(torch.nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.calls: list[int] = []

    def decode(self, codes: torch.Tensor):
        self.calls.append(int(codes[0, 0, 0]))
        values = codes.float().repeat_interleave(4, dim=-1)[:, :1, :] * 0.001
        return types.SimpleNamespace(audio_values=values)


class FakeModel:
    """Mimics OmniVoice.generate: tokens from _generate_iterative, then decode + host copy item by item."""

    def __init__(self, device: torch.device) -> None:
        self.audio_tokenizer = FakeTokenizer(device)
        self.device = device

    def _generate_iterative(self, count: int) -> list[torch.Tensor]:
        tokens = torch.arange(count * 8 * 6, device=self.device).view(count, 8, 6)
        return [tokens[i, :, : 6 - i % 2] for i in range(count)]

    def generate(self, count: int) -> list:
        results = self._generate_iterative(count)
        return [
            self.audio_tokenizer.decode(t.to(self.audio_tokenizer.device).unsqueeze(0)).audio_values[0].cpu().numpy()
            for t in results
        ]


def test_noop_on_cpu_and_restores_methods() -> None:
    model = FakeModel(torch.device('cpu'))
    with overlapped_decode(model):
        assert 'decode' not in vars(model.audio_tokenizer)
        out = model.generate(3)
    assert len(out) == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_overlap_decodes_one_ahead_with_identical_output() -> None:
    device = torch.device('cuda:0')
    stock = FakeModel(device).generate(4)
    model = FakeModel(device)
    with overlapped_decode(model):
        got = model.generate(4)
    assert all((a == b).all() for a, b in zip(stock, got))
    # Every item decoded exactly once, in order, the next one launched before the current one is returned.
    assert model.audio_tokenizer.calls == [0, 48, 96, 144]
    assert 'decode' not in vars(model.audio_tokenizer) and '_generate_iterative' not in vars(model)
