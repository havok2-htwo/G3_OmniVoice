"""Overlap the codec decode of the next item with the CPU post-processing of the current one.

After the diffusion loop, ``OmniVoice.generate`` handles the items one by one: decode the audio tokens on the GPU,
copy the waveform to the host (``.cpu()`` waits for the GPU), then remove silences and fade on the CPU (~7 ms per
~7 s item). The GPU sits idle during that CPU work, and the CPU waits during the next decode.

Inside ``overlapped_decode`` the tokenizer's ``decode`` launches item i+1's decode and an async copy into pinned host
memory before it waits for item i's copy, so the GPU decodes i+1 while the CPU post-processes i. Every decode runs
exactly as before on the same stream with the same input, so the audio is bit-identical; only the waiting moves.

Which item comes next is taken from the token lists the decoding loop returned (``_generate_iterative``), matched
by tensor identity (storage pointer, shape, strides). Tokens that do not match (e.g. chunks of long texts) are
decoded the stock way. Model or tokenizer without the expected attributes, or a non-CUDA device: no-op.
"""
from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

import torch


def _key(tensor: torch.Tensor) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    return (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()))


class _Overlap:
    def __init__(self, decode: Any, device: torch.device) -> None:
        self.decode = decode
        self.device = device
        self.sequences: list[list[torch.Tensor]] = []
        self.position: dict[tuple, tuple[int, int]] = {}
        self.pending: dict[tuple, tuple[Any, torch.cuda.Event]] = {}

    def capture(self, tokens: list[Any]) -> None:
        tensors = [t for t in tokens if isinstance(t, torch.Tensor)]
        if len(tensors) != len(tokens):
            return
        index = len(self.sequences)
        self.sequences.append(tensors)
        for j, tensor in enumerate(tensors):
            self.position.setdefault(_key(tensor), (index, j))

    def _launch(self, codes: torch.Tensor, args: tuple, kwargs: dict) -> tuple[Any, torch.cuda.Event]:
        output = self.decode(codes, *args, **kwargs)
        values = output if isinstance(output, torch.Tensor) else output.audio_values
        host = torch.empty(values.shape, dtype=values.dtype, device='cpu', pin_memory=True)
        host.copy_(values, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        if isinstance(output, torch.Tensor):
            return host, event
        output.audio_values = host
        return output, event

    def __call__(self, codes: Any, *args: Any, **kwargs: Any) -> Any:
        if not isinstance(codes, torch.Tensor) or codes.dim() != 3 or codes.shape[0] != 1:
            return self.decode(codes, *args, **kwargs)
        key = _key(codes[0])
        where = self.position.get(key)
        if where is None:
            return self.decode(codes, *args, **kwargs)
        current = self.pending.pop(key, None) or self._launch(codes, args, kwargs)
        sequence, j = where
        if j + 1 < len(self.sequences[sequence]):
            following = self.sequences[sequence][j + 1]
            following_key = _key(following)
            if following_key not in self.pending:
                # Same call the stock code makes for that item: tokens.to(device).unsqueeze(0).
                self.pending[following_key] = self._launch(following.to(self.device).unsqueeze(0), args, kwargs)
        output, event = current
        event.synchronize()
        return output


@contextlib.contextmanager
def overlapped_decode(model: Any) -> Iterator[None]:
    """Within this block, ``model.generate`` overlaps item decodes with post-processing (see module doc)."""
    tokenizer = getattr(model, 'audio_tokenizer', None)
    decode = getattr(tokenizer, 'decode', None)
    iterative = getattr(model, '_generate_iterative', None)
    device = getattr(tokenizer, 'device', None)
    if (
        not callable(decode)
        or not callable(iterative)
        or not isinstance(device, torch.device)
        or device.type != 'cuda'
        or not torch.cuda.is_available()
    ):
        yield
        return

    overlap = _Overlap(decode, device)
    # Restore instance attributes that were there before (e.g. the packed decoding loop), drop the rest.
    previous = {
        (owner, name): vars(owner)[name]
        for owner, name in ((model, '_generate_iterative'), (tokenizer, 'decode'))
        if name in vars(owner)
    }

    def generate_iterative(*args: Any, **kwargs: Any) -> Any:
        tokens = iterative(*args, **kwargs)
        overlap.capture(tokens)
        return tokens

    model._generate_iterative = generate_iterative
    tokenizer.decode = overlap
    try:
        yield
    finally:
        # Drop the instance attributes so the class methods show through again.
        for owner, name in ((model, '_generate_iterative'), (tokenizer, 'decode')):
            if (owner, name) in previous:
                setattr(owner, name, previous[(owner, name)])
                continue
            try:
                delattr(owner, name)
            except AttributeError:
                pass
        overlap.pending.clear()
