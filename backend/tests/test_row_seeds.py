"""Per-item seeds inside a batch (row_seeds): noise per item comes from that item's own seed, in its own order."""
from __future__ import annotations

import torch

from omnivoice_tts_server import row_seeds


def _sampler_draws(seeds, item_order, shape=(1, 8, 5)):
    """Run the patched sampler for the items in item_order (as the model does, one item after another)."""
    import omnivoice.models.omnivoice as module

    row_seeds.install()
    logits = torch.zeros(shape)
    out = {}
    with row_seeds.per_item_seeds(seeds, torch.device('cpu')):
        state = row_seeds._ACTIVE
        for item in item_order:
            state['item'] = item
            out.setdefault(item, []).append(module._gumbel_sample(logits.clone(), 1.0))
    return out


def test_item_noise_depends_only_on_its_own_seed() -> None:
    together = _sampler_draws([11, 22, 33], item_order=[0, 1, 2, 0, 1, 2])
    alone = _sampler_draws([22], item_order=[0, 0])
    assert torch.equal(together[1][0], alone[0][0])
    assert torch.equal(together[1][1], alone[0][1])


def test_different_seeds_give_different_noise() -> None:
    draws = _sampler_draws([11, 22], item_order=[0, 1])
    assert not torch.equal(draws[0][0], draws[1][0])


def test_no_seed_item_keeps_the_stock_sampler_path() -> None:
    import omnivoice.models.omnivoice as module

    row_seeds.install()
    logits = torch.zeros(1, 8, 5)
    with row_seeds.per_item_seeds([None], torch.device('cpu')):
        row_seeds._ACTIVE['item'] = 0
        torch.manual_seed(7)
        got = module._gumbel_sample(logits.clone(), 1.0)
    torch.manual_seed(7)
    u = torch.rand_like(logits)
    expected = logits + (-torch.log(-torch.log(u + 1e-10) + 1e-10))
    assert torch.equal(got, expected)


def test_context_is_cleared_after_use() -> None:
    with row_seeds.per_item_seeds([1, 2], torch.device('cpu')):
        assert row_seeds._ACTIVE is not None
    assert row_seeds._ACTIVE is None
