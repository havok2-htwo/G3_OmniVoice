"""Per-row random numbers from a counter, inside the CUDA graph.

Sampling needs ``draws`` Gumbel numbers per row and step. A row's numbers must not depend on what else runs, and a
shared generator cannot give that. So each number is a hash of (row key, step, draw): splitmix64's finalizer over a
counter, which is a good generator on its own. It needs no state, no staging memory and no host copy, it is exact
integer math, and being pure tensor ops it can be captured in a graph.
"""
from __future__ import annotations

import torch

_GOLDEN = -0x61c8864680b583eb  # 0x9e3779b97f4a7c15 as a two's complement int64
_MUL1 = -0x40a7b892e31b1a47   # 0xbf58476d1ce4e5b9
_MUL2 = -0x6b2fb644ecceee15   # 0x94d049bb133111eb
_MASK30, _MASK27, _MASK31 = (1 << 34) - 1, (1 << 37) - 1, (1 << 33) - 1
_SCALE = 2.0 ** -24


def row_key(seed: int, fragment: int) -> int:
    """Row key of a request seed and fragment index, as a signed int64: fragments of a row draw different numbers."""
    z = (int(seed) * 0x9e3779b97f4a7c15 + int(fragment) * 0xd1342543de82ef95 + 0x2545f4914f6cdd1d) & (2 ** 64 - 1)
    z = ((z ^ (z >> 30)) * 0xbf58476d1ce4e5b9) & (2 ** 64 - 1)
    z = ((z ^ (z >> 27)) * 0x94d049bb133111eb) & (2 ** 64 - 1)
    z ^= z >> 31
    return z - 2 ** 64 if z >= 2 ** 63 else z


def uniform(counter: torch.Tensor) -> torch.Tensor:
    """fp32 numbers in (0, 1) for an int64 counter tensor. int64 multiplication wraps, so this is splitmix64 exactly;
    the shifts mask off the sign extension of ``>>`` on signed integers."""
    z = counter * _GOLDEN
    z = (z ^ ((z >> 30) & _MASK30)) * _MUL1
    z = (z ^ ((z >> 27) & _MASK27)) * _MUL2
    z = z ^ ((z >> 31) & _MASK31)
    return (((z >> 40) & 0xffffff).float() + 0.5) * _SCALE


def gumbel(counter: torch.Tensor) -> torch.Tensor:
    """argmax(log p + gumbel) draws from p, like the exponential race of ArkttsModel._sample."""
    return -torch.log(-torch.log(uniform(counter)))
