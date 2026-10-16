"""Per-item seeds inside one batch.

OmniVoice draws its random numbers in ``_gumbel_sample`` (position scores and token sampling) from one global
generator, which is seeded once per batch. Items that ask for different seeds therefore cannot share a batch. This
module lets a batch with mixed seeds draw each item's noise from that item's own generator instead.

Only batches whose items ask for more than one distinct seed take this path (see ``runtime_v2``); a batch with one
seed, or with none, runs the stock sampler exactly as before.

How an item is recognised: the model samples item by item inside each diffusion step, and calls
``_predict_tokens_with_scoring`` once per item, in item order, before the sampler. Counting those calls (modulo the
batch size) gives the current item. Each item's noise then depends only on its own seed and its own calls, so one
item gives the same audio whatever else is in its batch.

The hooks patch the ``omnivoice`` package at runtime; they are installed once and are inactive outside ``per_item_seeds``.
"""
from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from typing import Any

import torch

_ACTIVE: dict[str, Any] | None = None
_INSTALLED = False


def _normalize(seed: int) -> int:
    return int(seed) % (2**31)


def install() -> None:
    """Patch the omnivoice sampler and the per-item hook once. Safe to call repeatedly."""
    global _INSTALLED
    if _INSTALLED:
        return
    import omnivoice.models.omnivoice as module

    stock_gumbel = module._gumbel_sample
    stock_predict = module.OmniVoice._predict_tokens_with_scoring

    def gumbel_sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
        state = _ACTIVE
        generator = None if state is None else state['generators'][state['item']]
        if generator is None:
            return stock_gumbel(logits, temperature)
        scaled_logits = logits / temperature
        u = torch.rand(scaled_logits.shape, generator=generator, device=scaled_logits.device, dtype=scaled_logits.dtype)
        gumbel_noise = -torch.log(-torch.log(u + 1e-10) + 1e-10)
        return scaled_logits + gumbel_noise

    def predict_tokens_with_scoring(self, c_logits, u_logits, gen_config):
        state = _ACTIVE
        if state is not None:
            state['item'] = state['calls'] % state['size']
            state['calls'] += 1
        return stock_predict(self, c_logits, u_logits, gen_config)

    module._gumbel_sample = gumbel_sample
    module.OmniVoice._predict_tokens_with_scoring = predict_tokens_with_scoring
    _INSTALLED = True


@contextlib.contextmanager
def per_item_seeds(seeds: Sequence[int | None], device: torch.device) -> Iterator[None]:
    """Draw item i's noise from a generator seeded with seeds[i]; ``None`` keeps the global random stream."""
    global _ACTIVE
    install()
    generators = []
    for seed in seeds:
        if seed is None:
            generators.append(None)
            continue
        generator = torch.Generator(device=device)
        generator.manual_seed(_normalize(seed))
        generators.append(generator)
    previous = _ACTIVE
    _ACTIVE = {'generators': generators, 'item': 0, 'calls': 0, 'size': len(seeds)}
    try:
        yield
    finally:
        _ACTIVE = previous
