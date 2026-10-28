# Vendored from inclusionAI/Ming-3.0-tts @ 9332fb9 (audio_tokenizer/modeling_audio_vae.py, vae_modules.py, istft.py),
# MIT license, see LICENSE in this directory. Trimmed to the 0.5B inference path. Changes against upstream:
# - No semantic (Whisper) module: the 0.5B checkpoint has none (semantic_module_kwargs is null).
# - AudioVAE is a plain nn.Module (the checkpoint keys are unchanged: audio.encoder.*, audio.decoder.*).
# - Encoder: the 4-layer aggregator gets its OWN copy of the backbone config. Upstream mutates the shared config
#   (num_hidden_layers = 4) after building the 24-layer encoder; transformers 4.46 ignored that at run time, but
#   transformers 5.x slices ``layers[:config.num_hidden_layers]`` and would silently run only 4 of the 24 encoder layers.
# - The posterior sample takes an explicit torch.Generator (diffusers' OobleckDiagonalGaussianDistribution inlined).
# - Decoder.stream_decode: the streaming path of low_level_reconstruct with an explicit per-stream state object.
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2Config, Qwen2Model


def backbone_config(args: dict) -> Qwen2Config:
    args = dict(args)
    # The checkpoint pins flash_attention_2; SDPA computes the same attention (upstream relaxes the pin the same way
    # when flash-attn is absent). Sliding-window masks are honoured by both.
    args['_attn_implementation'] = 'sdpa'
    args['attn_implementation'] = None
    # vocab_size is 1 (inputs_embeds only); the copied Qwen token ids would only trigger config warnings.
    args.pop('bos_token_id', None)
    args.pop('eos_token_id', None)
    config = Qwen2Config.from_dict(config_dict=args)
    config._attn_implementation = 'sdpa'
    return config


class StreamingLinearUpsample(nn.Module):
    def __init__(self, scale_factor=4):
        super().__init__()
        self.scale_factor = scale_factor
        self.upsampler = nn.Upsample(scale_factor=scale_factor, mode='linear', align_corners=False)

    def forward(self, x, state=None, is_last=False):
        if state is None:
            state = {'prev_chunk': None, 'history_last': None, 'is_first': True}
        if x is None and not is_last:
            return None, state
        if state['is_first'] and is_last:
            out = self.upsampler(x.transpose(1, 2)).transpose(1, 2)
            return out, None
        output_chunks = []
        if state['is_first']:
            state['prev_chunk'] = x
            state['is_first'] = False
            if not is_last:
                return None, state
        if state['prev_chunk'] is not None:
            p = state['prev_chunk'].transpose(1, 2)
            lookahead = x[:, :1, :].transpose(1, 2)
            if state['history_last'] is None:
                up = self.upsampler(torch.cat([p, lookahead], dim=2))
                out_prev = up[:, :, :p.size(2) * self.scale_factor]
            else:
                up = self.upsampler(torch.cat([state['history_last'], p, lookahead], dim=2))
                start = self.scale_factor
                out_prev = up[:, :, start:start + p.size(2) * self.scale_factor]
            output_chunks.append(out_prev.transpose(1, 2))
            state['history_last'] = p[:, :, -1:]
            state['prev_chunk'] = x
        if is_last:
            p = state['prev_chunk'].transpose(1, 2)
            up = self.upsampler(torch.cat([state['history_last'], p], dim=2))
            output_chunks.append(up[:, :, self.scale_factor:].transpose(1, 2))
            state = None
        final_out = torch.cat(output_chunks, dim=1) if output_chunks else None
        return final_out, state


class ISTFT(nn.Module):
    """ISTFT with "same" padding and streaming overlap buffers (vocos-style)."""

    def __init__(self, n_fft: int, hop_length: int, win_length: int, padding: str = 'same'):
        super().__init__()
        if padding not in ('center', 'same'):
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer('window', torch.hann_window(win_length))
        self.buffer_len = self.win_length - self.hop_length

    def _buffer_process(self, x, buffer, pad, last_chunk=False, streaming=False):
        if streaming:
            if buffer is None:
                x = x[:, pad:]
            if buffer is not None:
                x[:, :self.buffer_len] += buffer
            buffer = x[:, -self.buffer_len:]
            if not last_chunk:
                x = x[:, :-self.buffer_len]
            else:
                x = x[:, :-pad]
        else:
            x = x[:, pad:-pad]
        return x, buffer

    def forward(self, spec, audio_buffer=None, window_buffer=None, streaming=False, last_chunk=False):
        if self.padding == 'center':
            return torch.istft(spec, self.n_fft, self.hop_length, self.win_length, self.window, center=True)
        pad = (self.win_length - self.hop_length) // 2
        _, _, T = spec.shape
        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm='backward')
        ifft = ifft * self.window[None, :, None]
        output_size = (T - 1) * self.hop_length + self.win_length
        y = F.fold(ifft, output_size=(1, output_size), kernel_size=(1, self.win_length),
                   stride=(1, self.hop_length))[:, 0, 0, :]
        y, audio_buffer = self._buffer_process(y, audio_buffer, pad, last_chunk=last_chunk, streaming=streaming)
        window_sq = self.window.square().expand(1, T, -1).transpose(1, 2)
        window_envelope = F.fold(window_sq, output_size=(1, output_size), kernel_size=(1, self.win_length),
                                 stride=(1, self.hop_length)).squeeze(0).squeeze(0)
        window_envelope, window_buffer = self._buffer_process(window_envelope, window_buffer, pad,
                                                              last_chunk=last_chunk, streaming=streaming)
        window_envelope = window_envelope.squeeze()
        y = y / window_envelope
        return y, audio_buffer, window_buffer


class ISTFTHead(nn.Module):
    def __init__(self, dim: int, n_fft: int, hop_length: int, padding: str = 'same'):
        super().__init__()
        self.out = nn.Linear(dim, n_fft + 2)
        self.istft = ISTFT(n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding)

    def forward(self, x, audio_buffer=None, window_buffer=None, streaming=False, last_chunk=False):
        x_pred = self.out(x).transpose(1, 2)
        mag, p = x_pred.chunk(2, dim=1)
        mag = torch.clip(torch.exp(mag), max=1e2)
        S = mag * (torch.cos(p) + 1j * torch.sin(p))
        audio, audio_buffer, window_buffer = self.istft(S, audio_buffer=audio_buffer, window_buffer=window_buffer,
                                                        streaming=streaming, last_chunk=last_chunk)
        return audio.unsqueeze(1), x_pred, audio_buffer, window_buffer


class Encoder(nn.Module):
    def __init__(self, encoder_args, input_dim=320, hop_size=320, latent_dim=64, patch_size=-1):
        super().__init__()
        config = backbone_config(encoder_args)
        self.encoder = Qwen2Model(config)
        self.input_dim = input_dim
        self.hop_size = hop_size
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(input_dim, config.hidden_size, bias=False)
        self.fc2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.fc3 = nn.Linear(config.hidden_size, latent_dim * 2)
        self.norm = nn.LayerNorm(config.hidden_size)
        self.patch_size = patch_size
        if patch_size != -1:
            agg_config = copy.deepcopy(config)
            agg_config.num_hidden_layers = 4
            self.aggregator = Qwen2Model(agg_config)
            self.cls_embed = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

    def get_frames(self, x):
        num_frames_total = (x.size(-1) + self.hop_size - 1) // self.hop_size
        expected_len = (num_frames_total - 1) * self.hop_size + self.input_dim
        waveform = F.pad(x, (0, expected_len - x.size(-1)), value=0.0)
        return waveform.unfold(dimension=-1, size=self.input_dim, step=self.hop_size)

    def pad_patch_insert_cls(self, x):
        bsz, num_frame, dim = x.size()
        r = num_frame % self.patch_size
        pad_num = self.patch_size - r if r else 0
        x = F.pad(x, (0, 0, 0, pad_num), value=0.0)
        x = x.reshape(-1, self.patch_size, dim)
        x = torch.cat((x, self.cls_embed.expand(x.size(0), -1, -1)), dim=1)
        return x.reshape(bsz, -1, dim)

    def forward(self, waveform):
        x = self.get_frames(waveform)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.encoder(inputs_embeds=x).last_hidden_state
        if self.patch_size != -1:
            x = self.pad_patch_insert_cls(x)
            x = self.aggregator(inputs_embeds=x).last_hidden_state
            bsz, _, dim = x.size()
            x = x.reshape(-1, self.patch_size + 1, dim)
            x = x[:, -1:, :].reshape(bsz, -1, dim)
        return self.fc3(x)


@dataclass
class DecodeStreamState:
    """Per-stream decoder state: upsampler lookahead, sliding-window KV cache, ISTFT overlap buffers."""

    upsample: Any = None
    audio_buffer: torch.Tensor | None = None
    window_buffer: torch.Tensor | None = None
    past_key_values: Any = None
    latents_in: int = 0
    samples_out: int = 0
    extra: dict = field(default_factory=dict)


class Decoder(nn.Module):
    def __init__(self, decoder_args, output_dim=320, latent_dim=64, patch_size=-1):
        super().__init__()
        config = backbone_config(decoder_args)
        self.decoder = Qwen2Model(config)
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(latent_dim, config.hidden_size)
        self.hop_length = output_dim
        self.head = ISTFTHead(dim=config.hidden_size, n_fft=self.hop_length * 4, hop_length=self.hop_length,
                              padding='same')
        self.patch_size = patch_size
        if self.patch_size != -1:
            self.upsampling = StreamingLinearUpsample(scale_factor=patch_size)

    def decode_full(self, latent: torch.Tensor) -> torch.Tensor:
        """Upstream ``AudioVAE.decode(latent, use_cache=False, last_chunk=True)``: one pass, (B, 1, samples)."""
        x = self.fc1(latent)
        if self.patch_size != -1:
            x = self.upsampling.upsampler(x.transpose(1, 2)).transpose(1, 2)
        x = self.decoder(inputs_embeds=x, use_cache=False).last_hidden_state
        audio, _, _, _ = self.head(x, streaming=False)
        return audio

    def stream_decode(self, latent: torch.Tensor | None, state: DecodeStreamState, last_chunk: bool) -> torch.Tensor:
        """Upstream low_level_reconstruct(use_cache=True): feed latents (B, n, 64), get the audio that is final.

        Returns (B, 1, samples); samples may be 0 while the upsampler waits for its one-latent lookahead.
        """
        if latent is not None:
            state.latents_in += latent.shape[1]
            x = self.fc1(latent)
            bsz, device, dtype = x.size(0), x.device, x.dtype
        else:
            x = None
            bsz, device, dtype = 1, self.fc1.weight.device, self.fc1.weight.dtype
        x, state.upsample = self.upsampling(x, state=state.upsample, is_last=last_chunk)
        if x is None:
            return torch.empty(bsz, 1, 0, device=device, dtype=dtype)
        outputs = self.decoder(inputs_embeds=x, past_key_values=state.past_key_values, use_cache=True)
        state.past_key_values = outputs.past_key_values
        audio, _, state.audio_buffer, state.window_buffer = self.head(
            outputs.last_hidden_state, streaming=True, audio_buffer=state.audio_buffer,
            window_buffer=state.window_buffer, last_chunk=last_chunk)
        state.samples_out += audio.shape[-1]
        return audio


class AudioVAE(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        if config.get('semantic_module_kwargs') is not None:
            raise NotImplementedError('Ming audio tokenizers with a semantic module are not supported')
        self.config = config
        enc = config['enc_kwargs']
        dec = config['dec_kwargs']
        self.patch_size = config.get('patch_size', -1)
        self.sample_rate = int(config.get('sample_rate', 44100))
        self.encoder = Encoder(encoder_args=enc['backbone'], input_dim=enc['input_dim'],
                               hop_size=enc.get('hop_size', 320), latent_dim=enc['latent_dim'],
                               patch_size=self.patch_size)
        self.decoder = Decoder(decoder_args=dec['backbone'], output_dim=dec['output_dim'], latent_dim=dec['latent_dim'],
                               patch_size=self.patch_size)

    @torch.inference_mode()
    def encode_latent(self, waveform: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        """(B, samples) waveform -> (B, frames, latent_dim) posterior sample (Oobleck diagonal Gaussian)."""
        h = self.encoder(waveform).transpose(1, 2)
        mean, scale = h.chunk(2, dim=1)
        std = F.softplus(scale) + 1e-4
        gen_device = generator.device if generator is not None else h.device
        noise = torch.randn(mean.shape, generator=generator, device=gen_device, dtype=torch.float32)
        latent = mean + std * noise.to(device=h.device, dtype=h.dtype)
        return latent.transpose(1, 2)
