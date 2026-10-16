"""Classifier-free guidance without the padding: one row per item holding [conditional | unconditional].

Stock ``OmniVoice._generate_iterative`` runs 2B rows per diffusion step: B conditional rows (style + text + voice
prompt + target, length c) and B unconditional rows (target only, length t), all padded to the longest conditional
row. The unconditional rows are mostly padding: with a ~8 s voice prompt and a ~7 s sentence, c ~ 450 and t ~ 175,
so ~31 % of the tokens every step computes are padding.

Here item i is one row ``[cond_i | uncond_i]`` of length c_i + t_i with a block-diagonal mask (each part only sees
itself, as in its own stock row) and positions that restart at 0 for the unconditional part (as in its stock row).
Rows are padded to the longest c + t. Same model, same inputs per segment, same schedule, same sampler calls in the
same order (so seeds and ``row_seeds`` behave as before); only the forward's arithmetic layout changes, so results
are equal up to floating-point rounding, not bit-identical.

``install(model)`` binds the packed loop as the model's ``_generate_iterative``; ``uninstall`` restores the stock one.
The loop below follows omnivoice 0.1.5 ``_generate_iterative`` line by line except for the batch layout.
"""
from __future__ import annotations

import math
from types import MethodType
from typing import Any

import torch

_ATTR = '_g3_packed_cfg'


def _packed_generate_iterative(self: Any, task: Any, gen_config: Any) -> list[torch.Tensor]:
    from omnivoice.models import omnivoice as ov

    B = task.batch_size
    C = self.config.num_audio_codebook
    inputs_list = [
        self._prepare_inference_inputs(
            task.texts[i],
            task.target_lens[i],
            task.ref_texts[i],
            task.ref_audio_tokens[i],
            task.langs[i],
            task.instructs[i],
            gen_config.denoise,
        )
        for i in range(B)
    ]

    c_lens = [inp['input_ids'].size(2) for inp in inputs_list]
    t_lens = list(task.target_lens)
    row_lens = [c + t for c, t in zip(c_lens, t_lens)]
    L = max(row_lens)
    pad_id = self.config.audio_mask_id

    batch_input_ids = torch.full((B, C, L), pad_id, dtype=torch.long, device=self.device)
    batch_audio_mask = torch.zeros((B, L), dtype=torch.bool, device=self.device)
    batch_attention_mask = torch.zeros((B, 1, L, L), dtype=torch.bool, device=self.device)
    position_ids = torch.arange(L, device=self.device).repeat(B, 1)

    for i, inp in enumerate(inputs_list):
        c, t = c_lens[i], t_lens[i]
        # Conditional part: positions 0..c-1, attends to itself.
        batch_input_ids[i, :, :c] = inp['input_ids'][0]
        batch_audio_mask[i, :c] = inp['audio_mask'][0]
        batch_attention_mask[i, :, :c, :c] = True
        # Unconditional part: the target tokens only, positions 0..t-1, attends to itself.
        batch_input_ids[i, :, c:c + t] = inp['input_ids'][0, :, -t:]
        batch_audio_mask[i, c:c + t] = inp['audio_mask'][0, -t:]
        batch_attention_mask[i, :, c:c + t, c:c + t] = True
        position_ids[i, c:c + t] = torch.arange(t, device=self.device)
        if L > c + t:
            # Padding attends to itself only, so no attention row is fully masked.
            pad_diag = torch.arange(c + t, L, device=self.device)
            batch_attention_mask[i, :, pad_diag, pad_diag] = True

    tokens = torch.full((B, C, max(t_lens)), self.config.audio_mask_id, dtype=torch.long, device=self.device)

    timesteps = ov._get_time_steps(t_start=0.0, t_end=1.0, num_step=gen_config.num_step, t_shift=gen_config.t_shift).tolist()
    schedules = []
    for t_len in t_lens:
        total_mask = t_len * C
        rem = total_mask
        sched = []
        for step in range(gen_config.num_step):
            num = (
                rem
                if step == gen_config.num_step - 1
                else min(math.ceil(total_mask * (timesteps[step + 1] - timesteps[step])), rem)
            )
            sched.append(int(num))
            rem -= int(num)
        schedules.append(sched)

    layer_ids = torch.arange(C, device=self.device).view(1, -1, 1)

    for step in range(gen_config.num_step):
        batch_logits = self(
            input_ids=batch_input_ids,
            audio_mask=batch_audio_mask,
            attention_mask=batch_attention_mask,
            position_ids=position_ids,
        ).logits.to(torch.float32)

        for i in range(B):
            k = schedules[i][step]
            if k <= 0:
                continue
            c_len, t_len = c_lens[i], t_lens[i]
            c_logits = batch_logits[i:i + 1, :, c_len - t_len:c_len, :]
            u_logits = batch_logits[i:i + 1, :, c_len:c_len + t_len, :]

            pred_tokens, scores = self._predict_tokens_with_scoring(c_logits, u_logits, gen_config)
            scores = scores - (layer_ids * gen_config.layer_penalty_factor)
            if gen_config.position_temperature > 0.0:
                # Looked up on the module at call time, like the stock loop, so row_seeds' patch applies.
                scores = ov._gumbel_sample(scores, gen_config.position_temperature)

            sample_tokens = tokens[i:i + 1, :, :t_len]
            scores.masked_fill_(sample_tokens != self.config.audio_mask_id, -float('inf'))
            _, topk_idx = torch.topk(scores.flatten(), k)
            flat_tokens = sample_tokens.flatten()
            flat_tokens[topk_idx] = pred_tokens.flatten()[topk_idx]
            sample_tokens.copy_(flat_tokens.view_as(sample_tokens))

            tokens[i:i + 1, :, :t_len] = sample_tokens
            batch_input_ids[i:i + 1, :, c_len - t_len:c_len] = sample_tokens
            batch_input_ids[i:i + 1, :, c_len:c_len + t_len] = sample_tokens

    return [tokens[i, :, :t_lens[i]] for i in range(B)]


def install(model: Any) -> bool:
    """Make ``model._generate_iterative`` the packed loop (an instance attribute; the class keeps the stock one)."""
    if not callable(getattr(model, '_prepare_inference_inputs', None)) or not hasattr(model, 'config'):
        return False
    model._generate_iterative = MethodType(_packed_generate_iterative, model)
    setattr(model, _ATTR, True)
    return True


def uninstall(model: Any) -> None:
    for name in ('_generate_iterative', _ATTR):
        try:
            delattr(model, name)
        except AttributeError:
            pass


def is_installed(model: Any) -> bool:
    return bool(getattr(model, _ATTR, False))
