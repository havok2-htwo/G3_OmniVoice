"""Batched, slot-based streaming VAE decoder (graph-safe) for the Ming engine.

Same math as ``audio_vae.Decoder.stream_decode`` (upstream ``low_level_reconstruct(use_cache=True)``), restated with
static shapes so many rows can be decoded in one call and the call can be captured as a CUDA graph:

- Upsampling: ``nn.Upsample(scale_factor=4, mode='linear', align_corners=False)`` evaluated per output frame. Frame
  ``4k + r`` blends latents k-1/k (r = 0, 1) or k/k+1 (r = 2, 3) with the kernel's exact weights (0.375/0.625,
  0.125/0.875, ...) in fp32 and one rounding, clamping at the fragment edges -- bit-identical to the kernel, and
  identical to upstream's streaming upsampler with its one-latent lookahead.
- Backbone: the causal sliding-window (64 keys) Qwen2 decoder with a per-slot KV ring buffer; each key slot carries
  its absolute position, so the mask is ``key_pos <= q_pos and key_pos > q_pos - 64``.
- ISTFT head ("same" padding): per-frame inverse FFT + window, overlap-add with a per-slot carry of the last
  ``win - hop`` samples (fp32), divided by the window envelope. The envelope is the one upstream's one-pass decode
  computes (window^2 folded in the model dtype); it is periodic after the first three hops, so a prefix, one period
  and a tail table describe every position.

One call ("job group") decodes, for every row, ``n`` latents starting at latent ``d`` of the row's current fragment.
Inputs per row: the latent window ``[d-1, d+n]`` (clamped to the fragment, the last entry is the lookahead or, for the
final chunk, a clamped repeat) and ``meta = (slot, first_frame, valid_frames, start)``. Rows beyond the valid ones
and frames beyond ``valid_frames`` are padding: their KV/overlap writes go to a trash slot and their audio is ignored.
Output per row: ``F*hop`` samples of final audio (only the first ``valid_frames*hop`` are meaningful) followed by
``pad`` samples of the tail, which is only meaningful for the fragment's final chunk.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .fast import FastQwen, linear

_NEG = -(1 << 40)


class SlotDecoder:
    def __init__(self, decoder, slots: int, max_chunk_latents: int, max_fragment_latents: int,
                 device: torch.device, dtype: torch.dtype) -> None:
        cfg = decoder.decoder.config
        self.decoder = decoder
        self.dtype = dtype
        self.device = device
        self.slots = int(slots)
        self.trash = self.slots  # extra slot index for padding writes
        self.scale = int(decoder.patch_size)
        self.window = int(cfg.sliding_window)
        istft = decoder.head.istft
        self.hop = int(istft.hop_length)
        self.win = int(istft.win_length)
        self.n_fft = int(istft.n_fft)
        self.pad = (self.win - self.hop) // 2
        self.buf_len = self.win - self.hop
        if self.buf_len % self.hop or self.buf_len >= self.scale * self.hop:
            raise NotImplementedError('unexpected ISTFT geometry')
        self.max_chunk = int(max_chunk_latents)
        self.max_frames = self.scale * (self.max_chunk + 1)
        self.ring = -(-(self.window + self.max_frames) // 8) * 8
        self.qwen = FastQwen(decoder.decoder, rope_positions=self.scale * max_fragment_latents + self.max_frames + 8,
                             dtype=dtype)
        self.rope_last = self.qwen.cos.shape[0] - 1
        nkv, hd = self.qwen.n_kv, self.qwen.head_dim
        n_slots = self.slots + 1
        self.k = [torch.zeros(n_slots, nkv, self.ring, hd, device=device, dtype=dtype) for _ in self.qwen.layers]
        self.v = [torch.zeros(n_slots, nkv, self.ring, hd, device=device, dtype=dtype) for _ in self.qwen.layers]
        self.kpos = torch.full((n_slots, self.ring), _NEG, device=device, dtype=torch.long)
        self.ola = torch.zeros(n_slots, self.buf_len, device=device, dtype=torch.float32)
        self._envelope_tables()
        self.arange_buf = torch.arange(self.buf_len, device=device)
        self.lerp = torch.tensor([[0.375, 0.625, 0.0], [0.125, 0.875, 0.0], [0.0, 0.875, 0.125], [0.0, 0.625, 0.375]],
                                 device=device, dtype=torch.float32)

    def _envelope_tables(self) -> None:
        window = self.decoder.head.istft.window  # model dtype, like upstream after .to(bfloat16)
        frames = 16
        env = F.fold(window.square().expand(1, frames, -1).transpose(1, 2),
                     output_size=(1, (frames - 1) * self.hop + self.win), kernel_size=(1, self.win),
                     stride=(1, self.hop)).reshape(-1)
        self.env_prefix = env[:self.buf_len].clone()
        self.env_period = env[self.buf_len:self.buf_len + self.hop].clone()
        self.env_tail = env[frames * self.hop:frames * self.hop + self.buf_len].clone()

    def reset(self) -> None:
        self.kpos.fill_(_NEG)
        self.ola.zero_()

    def out_len(self, chunk: int) -> int:
        return self.scale * chunk * self.hop + self.pad

    def run(self, lat: torch.Tensor, meta: torch.Tensor, out: torch.Tensor) -> None:
        """lat (R, C+2, latent) model dtype, meta (R, 4) long, out (R, 4*C*hop + pad) fp32. Graph-safe."""
        R, C = lat.shape[0], lat.shape[1] - 2
        nF = self.scale * C
        dec, q = self.decoder, self.qwen
        dev = lat.device
        slot, p0, nf, start = meta[:, 0], meta[:, 1], meta[:, 2], meta[:, 3].bool()
        j = torch.arange(nF, device=dev)
        pos = p0[:, None] + j[None, :]
        valid = j[None, :] < nf[:, None]
        # fc1 + streaming linear upsample (exact kernel weights, fp32, one rounding).
        x = linear(lat, dec.fc1.weight, dec.fc1.bias).float()
        ctx = torch.stack((x[:, :-2], x[:, 1:-1], x[:, 2:]), dim=2)          # (R, C, 3, H)
        h = torch.matmul(self.lerp, ctx)                                    # (R, C, 4, H) == w0*a + w1*b (+0*c)
        h = h.reshape(R, nF, -1).to(self.dtype)
        # Ring positions; padding frames/rows write to the trash slot.
        ri = pos % self.ring
        slot_w = torch.where(valid, slot[:, None], torch.full_like(pos, self.trash))
        kp = torch.where(start[:, None], torch.full_like(self.kpos[slot], _NEG), self.kpos[slot])
        self.kpos[slot] = kp
        self.kpos[slot_w, ri] = pos
        kp = self.kpos[slot]
        mask = (kp[:, None, :] <= pos[:, :, None]) & (kp[:, None, :] > pos[:, :, None] - self.window)
        n_rep = q.n_heads // q.n_kv
        mask = mask[:, None, None].expand(R, 1, n_rep, nF, self.ring).reshape(R, 1, n_rep * nF, self.ring)
        cos, sin = q.rope(pos.clamp(max=self.rope_last))
        for i in range(len(q.layers)):
            qh, kh, vh = q.layer_qkv(i, h, cos, sin)
            kc, vc = self.k[i], self.v[i]
            kc[slot_w, :, ri] = kh.permute(0, 2, 1, 3)
            vc[slot_w, :, ri] = vh.permute(0, 2, 1, 3)
            qg = qh.reshape(R, q.n_kv, n_rep * nF, q.head_dim)
            a = F.scaled_dot_product_attention(qg, kc[slot], vc[slot], attn_mask=mask, scale=q.scaling)
            a = a.reshape(R, q.n_heads, nF, q.head_dim).transpose(1, 2).reshape(R, nF, q.hidden)
            h = q.layer_out(i, h, a)
        h = q.final_norm(h)
        # ISTFT head (ISTFTHead.forward + ISTFT.forward with streaming buffers).
        head = dec.head
        x_pred = linear(h, head.out.weight, head.out.bias).transpose(1, 2)
        mag, p = x_pred.chunk(2, dim=1)
        mag = torch.clip(torch.exp(mag), max=1e2)
        spec = mag * (torch.cos(p) + 1j * torch.sin(p))
        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm='backward')
        ifft = ifft * head.istft.window[None, :, None]
        ifft = torch.where(valid[:, None, :], ifft, torch.zeros((), device=dev, dtype=ifft.dtype))
        y = F.fold(ifft, output_size=(1, (nF - 1) * self.hop + self.win), kernel_size=(1, self.win),
                   stride=(1, self.hop))[:, 0, 0, :]
        carry = torch.where(start[:, None], torch.zeros_like(self.ola[slot]), self.ola[slot])
        y[:, :self.buf_len] += carry
        tail = y.gather(1, nf[:, None] * self.hop + self.arange_buf[None, :])
        self.ola[slot] = tail
        n = p0[:, None] * self.hop + torch.arange(nF * self.hop, device=dev)[None, :]
        env = torch.where(n < self.buf_len, self.env_prefix[n.clamp(max=self.buf_len - 1)],
                          self.env_period[n % self.hop])
        out[:, :nF * self.hop] = y[:, :nF * self.hop] / env
        out[:, nF * self.hop:] = tail[:, :self.pad] / self.env_tail[:self.pad]
