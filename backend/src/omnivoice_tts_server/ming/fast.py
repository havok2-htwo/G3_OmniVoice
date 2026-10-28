"""Graph-friendly fast paths for the Ming hot loop (LLM decode/prefill, flow head, aggregator, VAE decoder stack).

Same math as the vendored reference modules, with fewer and better-shaped kernels. Every change is either
bit-identical or a reordering inside one bf16 rounding step:

- ``linear``: cuBLASLt picks a slow kernel for the fused bias epilogue at some small M (2-4x slower on sm_120). Where
  that is measurably the case we write the bias into the output and accumulate the GEMM on top (``addmm_`` with
  beta=1): still one fp32 accumulation and one rounding, just a different GEMM kernel.
- q/k/v (and Qwen2 gate/up) projections run as one GEMM on concatenated weights. The original parameters are
  re-pointed to views of the fused tensors, so no weight memory is duplicated and the reference modules still work.
- RoPE: cos/sin tables are precomputed once with exact fp32 products (the reference computes ``inv_freq @ positions``
  with a K=1 matmul, which is the same exact product unless TF32 is enabled process-wide -- the server enables TF32,
  so the tables are now also independent of that switch). The rotation itself uses identical elementwise ops;
  ``rotate_half(x) * sin`` is written as ``roll/flip(x) * signed_sin`` (exactly equal in IEEE arithmetic), fused
  into one ``addcmul`` (one rounding instead of two).
- Residual + bias + projection in the flow blocks is one GEMM accumulating into ``x + b`` (``residual_linear``).
- Flow head: everything that is constant over the 10 Euler steps (condition and history embeddings, the timestep
  embeddings, the rotary tables, ``dt``) is computed once per LM step / once per bucket instead of every step.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

# Bias linears: "fused" = F.linear (cuBLASLt bias epilogue), "split" = bias copy + GEMM accumulate (addmm_, beta=1).
# Both do one fp32 accumulation and one rounding. cuBLASLt's heuristic for the fused epilogue is erratic at small M
# (e.g. 29 us instead of 7 us for 24x1024 @ 1024x3072 on sm_120), so each shape is timed once per process and the
# split form is used only when it is clearly faster (> SPLIT_MARGIN). Shapes are planned during the eager warm-up,
# i.e. before any graph capture, and the plan never changes afterwards (graphs on/off and every replay agree).
# Because the choice changes the last bits of the result, plans are persisted (``load_plans`` / ``save_plans``,
# keyed by GPU, torch and CUDA version) so a restart on the same machine reuses them and renders identically.
SPLIT_MARGIN = 1.3
_LINEAR_MODE = 'auto'  # 'auto' | 'fused' | 'split'
_PLANS: dict[tuple, bool] = {}
_TIMINGS: dict[tuple, tuple[float, float]] = {}
_DIRTY = False


def _device_tag(index: int | None) -> str:
    props = torch.cuda.get_device_properties(index if index is not None else torch.cuda.current_device())
    return f'{props.name}|sm{props.major}{props.minor}|torch{torch.__version__}|cuda{torch.version.cuda}'


def load_plans(path, device: torch.device) -> int:
    """Load persisted plans for this GPU/software stack; returns how many were loaded."""
    import json
    from pathlib import Path

    path = Path(path)
    if device.type != 'cuda' or not path.is_file():
        return 0
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return 0
    entries = data.get(_device_tag(device.index), {}) if isinstance(data, dict) else {}
    dtypes = {str(t): t for t in (torch.bfloat16, torch.float16, torch.float32)}
    n = 0
    for key, split in entries.items():
        try:
            m, k, nn, dt = key.split(',')
            _PLANS.setdefault((int(m), int(k), int(nn), dtypes[dt], device.index), bool(split))
            n += 1
        except (ValueError, KeyError):
            continue
    return n


def save_plans(path, device: torch.device) -> bool:
    """Persist the plans of this device (merged into an existing file). Never raises."""
    global _DIRTY
    import json
    from pathlib import Path

    if device.type != 'cuda' or not _DIRTY:
        return False
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
        if not isinstance(data, dict):
            data = {}
        tag = _device_tag(device.index)
        entries = data.setdefault(tag, {})
        for (m, k, n, dt, idx), split in _PLANS.items():
            if idx == device.index:
                entries[f'{m},{k},{n},{dt}'] = bool(split)
        tmp = path.with_suffix(path.suffix + '.tmp')
        tmp.write_text(json.dumps(data, indent=0, sort_keys=True), encoding='utf-8')
        tmp.replace(path)
        _DIRTY = False
        return True
    except OSError:
        return False


def set_linear_mode(mode: str) -> None:
    global _LINEAR_MODE
    if mode not in ('auto', 'fused', 'split'):
        raise ValueError(mode)
    _LINEAR_MODE = mode


def linear_plans() -> dict[str, int]:
    split = sum(1 for v in _PLANS.values() if v)
    return {'shapes': len(_PLANS), 'split': split}


def _split(x2: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # clone, not contiguous(): for one row, expand() is already "contiguous" and addmm_ would write into the bias.
    out = b.expand(x2.shape[0], b.shape[0]).clone(memory_format=torch.contiguous_format)
    out.addmm_(x2, w.t())
    return out


def _time_us(fn, reps: int = 10, replays: int = 6) -> float:
    for _ in range(3):
        fn()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, capture_error_mode='thread_local'):
        for _ in range(reps):
            fn()
    g.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        g.replay()
    end.record()
    end.synchronize()
    del g
    return start.elapsed_time(end) * 1000.0 / (reps * replays)


def _use_split(m: int, k: int, n: int, x: torch.Tensor) -> bool:
    global _DIRTY
    if _LINEAR_MODE != 'auto':
        return _LINEAR_MODE == 'split'
    if x.device.type != 'cuda':
        return False
    key = (m, k, n, x.dtype, x.device.index)
    plan = _PLANS.get(key)
    if plan is not None:
        return plan
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(f'linear shape {key} was not planned before CUDA graph capture')
    xs = torch.randn(m, k, device=x.device, dtype=x.dtype)
    ws = torch.randn(n, k, device=x.device, dtype=x.dtype) * k ** -0.5
    bs = torch.randn(n, device=x.device, dtype=x.dtype)
    try:
        with torch.inference_mode():
            fused = min(_time_us(lambda: F.linear(xs, ws, bs)) for _ in range(2))
            split = min(_time_us(lambda: _split(xs, ws, bs)) for _ in range(2))
        plan = fused > SPLIT_MARGIN * split
        _TIMINGS[key] = (fused, split)
    except Exception:  # noqa: BLE001 - no graphs available: keep the reference kernel
        plan = False
    _PLANS[key] = plan
    _DIRTY = True
    return plan


def linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None = None) -> torch.Tensor:
    if b is None:
        return F.linear(x, w)
    k = x.shape[-1]
    m = x.numel() // k
    if not _use_split(m, k, w.shape[0], x):
        return F.linear(x, w, b)
    return _split(x.reshape(m, k), w, b).view(*x.shape[:-1], w.shape[0])


def residual_linear(x: torch.Tensor, a: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``x + linear(a, w, b)`` as (x + b) accumulated by one GEMM (beta=1): 2 kernels instead of 3, no bias epilogue."""
    out = x + b
    out.view(-1, out.shape[-1]).addmm_(a.reshape(-1, a.shape[-1]), w.t())
    return out


def qwen_rms(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen2RMSNorm: normalise in fp32, round to the input dtype, multiply by the weight in that dtype."""
    h = F.rms_norm(x.float(), (x.shape[-1],), eps=eps)
    return weight * h.to(x.dtype)


def _fuse(params: list[torch.nn.Parameter]) -> tuple[torch.Tensor, list[torch.Tensor]]:
    fused = torch.cat([p.detach() for p in params], dim=0).contiguous()
    views, start = [], 0
    for p in params:
        n = p.shape[0]
        views.append(fused[start:start + n])
        start += n
    return fused, views


def _repoint(module: torch.nn.Module, name: str, view: torch.Tensor) -> None:
    setattr(module, name, torch.nn.Parameter(view, requires_grad=False))


def rope_table(inv_freq: torch.Tensor, positions: int, dtype: torch.dtype, scaling: float = 1.0):
    """Qwen2RotaryEmbedding cos/sin for positions [0, n): (n, head_dim) in ``dtype``; sin is pre-signed for roll."""
    pos = torch.arange(positions, device=inv_freq.device, dtype=torch.float32)
    freqs = pos[:, None] * inv_freq.float()[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = (emb.cos() * scaling).to(dtype)
    sin = (emb.sin() * scaling).to(dtype)
    half = sin.shape[-1] // 2
    sin_signed = torch.cat((-sin[:, :half], sin[:, half:]), dim=-1)
    return cos.contiguous(), sin_signed.contiguous()


def apply_qwen_rope(x: torch.Tensor, cos: torch.Tensor, sin_signed: torch.Tensor) -> torch.Tensor:
    """x (..., D) * cos + rotate_half(x) * sin with rotate_half(x) = cat(-x2, x1) == roll(x, D/2) * sign."""
    return torch.addcmul(x * cos, x.roll(x.shape[-1] // 2, dims=-1), sin_signed)


@dataclass
class QwenLayer:
    ln1: torch.Tensor
    ln2: torch.Tensor
    eps1: float
    eps2: float
    w_qkv: torch.Tensor
    b_qkv: torch.Tensor | None
    w_o: torch.Tensor
    w_gu: torch.Tensor
    w_down: torch.Tensor


class FastQwen:
    """A Qwen2Model stack (the Ming LLM or the VAE decoder backbone) with fused projections.

    ``layer_qkv`` returns rotated q (B, Hq, L, D), k (B, Hkv, L, D), v (B, Hkv, L, D); ``layer_out`` finishes the
    layer given the attention output (B, L, Hq*D). The attention itself is up to the caller (slot KV cache, ring
    buffer, causal prefill).
    """

    def __init__(self, model, rope_positions: int, dtype: torch.dtype) -> None:
        cfg = model.config
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = getattr(cfg, 'head_dim', None) or cfg.hidden_size // cfg.num_attention_heads
        self.hidden = cfg.hidden_size
        self.layers: list[QwenLayer] = []
        for layer in model.layers:
            a, mlp = layer.self_attn, layer.mlp
            w_qkv, (wq, wk, wv) = _fuse([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight])
            for mod, view in ((a.q_proj, wq), (a.k_proj, wk), (a.v_proj, wv)):
                _repoint(mod, 'weight', view)
            b_qkv = None
            if a.q_proj.bias is not None:
                b_qkv, (bq, bk, bv) = _fuse([a.q_proj.bias, a.k_proj.bias, a.v_proj.bias])
                for mod, view in ((a.q_proj, bq), (a.k_proj, bk), (a.v_proj, bv)):
                    _repoint(mod, 'bias', view)
            w_gu, (wg, wu) = _fuse([mlp.gate_proj.weight, mlp.up_proj.weight])
            _repoint(mlp.gate_proj, 'weight', wg)
            _repoint(mlp.up_proj, 'weight', wu)
            if a.o_proj.bias is not None or mlp.down_proj.bias is not None or mlp.gate_proj.bias is not None:
                raise NotImplementedError('unexpected Qwen2 bias layout')
            self.layers.append(QwenLayer(
                ln1=layer.input_layernorm.weight, ln2=layer.post_attention_layernorm.weight,
                eps1=float(layer.input_layernorm.variance_epsilon),
                eps2=float(layer.post_attention_layernorm.variance_epsilon),
                w_qkv=w_qkv, b_qkv=b_qkv, w_o=a.o_proj.weight, w_gu=w_gu, w_down=mlp.down_proj.weight))
        self.scaling = float(model.layers[0].self_attn.scaling)
        self.norm_w = model.norm.weight
        self.norm_eps = float(model.norm.variance_epsilon)
        rot = model.rotary_emb
        self.cos, self.sin = rope_table(rot.inv_freq, rope_positions, dtype, float(rot.attention_scaling))
        self.n_inter = self.layers[0].w_gu.shape[0] // 2

    def rope(self, positions: torch.Tensor):
        """cos/sin (…, D) for integer positions of any shape."""
        return self.cos[positions], self.sin[positions]

    def layer_qkv(self, i: int, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        lp = self.layers[i]
        B, L, _ = h.shape
        x = qwen_rms(h, lp.ln1, lp.eps1)
        qkv = linear(x, lp.w_qkv, lp.b_qkv).view(B, L, self.n_heads + 2 * self.n_kv, self.head_dim)
        nqk = self.n_heads + self.n_kv
        qk = apply_qwen_rope(qkv[:, :, :nqk], cos.unsqueeze(2), sin.unsqueeze(2))
        q = qk[:, :, :self.n_heads].transpose(1, 2)
        k = qk[:, :, self.n_heads:].transpose(1, 2)
        v = qkv[:, :, nqk:].transpose(1, 2)
        return q, k, v

    def layer_out(self, i: int, h: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        lp = self.layers[i]
        h = h + linear(attn, lp.w_o)
        x = qwen_rms(h, lp.ln2, lp.eps2)
        g, u = linear(x, lp.w_gu).split(self.n_inter, dim=-1)
        return h + linear(F.silu(g) * u, lp.w_down)

    def final_norm(self, h: torch.Tensor) -> torch.Tensor:
        return qwen_rms(h, self.norm_w, self.norm_eps)


# -- flow head (DiT) + aggregator ---------------------------------------------------------------------------------


@dataclass
class DiTLayer:
    n1: torch.Tensor
    n2: torch.Tensor
    eps1: float
    eps2: float
    w_qkv: torch.Tensor
    b_qkv: torch.Tensor
    w_o: torch.Tensor
    b_o: torch.Tensor
    w_up: torch.Tensor
    b_up: torch.Tensor
    w_down: torch.Tensor
    b_down: torch.Tensor


def _dit_layers(blocks) -> list[DiTLayer]:
    out = []
    for blk in blocks:
        a = blk.attn
        w_qkv, (wq, wk, wv) = _fuse([a.to_q.weight, a.to_k.weight, a.to_v.weight])
        b_qkv, (bq, bk, bv) = _fuse([a.to_q.bias, a.to_k.bias, a.to_v.bias])
        for mod, wv_, bv_ in ((a.to_q, wq, bq), (a.to_k, wk, bk), (a.to_v, wv, bv)):
            _repoint(mod, 'weight', wv_)
            _repoint(mod, 'bias', bv_)
        up, down = blk.mlp.ff[0][0], blk.mlp.ff[2]
        if blk.mlp.ff[0][1].approximate != 'tanh':
            raise NotImplementedError('DiT MLP must use tanh GELU')
        if not (blk.norm1.native_default if blk.norm1.native_rms_norm is None else blk.norm1.native_rms_norm):
            raise NotImplementedError('fast flow path needs the native RMSNorm (torch >= 2.4)')
        out.append(DiTLayer(n1=blk.norm1.weight, n2=blk.norm2.weight, eps1=blk.norm1.eps, eps2=blk.norm2.eps,
                            w_qkv=w_qkv, b_qkv=b_qkv, w_o=a.to_out[0].weight, b_o=a.to_out[0].bias,
                            w_up=up.weight, b_up=up.bias, w_down=down.weight, b_down=down.bias))
    return out


class FastFlow:
    """The CFG flow sampler (fm/CFM.sample + DiT.forward_with_cfg), the stop head and the aggregator."""

    def __init__(self, model, timesteps: torch.Tensor, steps: int, cfg_scale: float, patch_size: int) -> None:
        dit = model.dit
        agg = model.linear_proj_audio
        if patch_size != 1:
            raise NotImplementedError('fast flow path supports patch_size 1 (Ming-3.0-tts 0.5B)')
        self.dit = dit
        self.agg = agg
        self.stop_head = model.stop_head
        self.steps = int(steps)
        self.cfg = float(cfg_scale)
        self.heads = dit.num_heads
        self.dit_layers = _dit_layers(dit.blocks)
        self.agg_layers = _dit_layers(agg.blocks)
        self.agg_heads = agg.num_heads
        self.dit_cos, self.dit_sin = self._rope(dit.rotary_embed, 3)
        self.agg_cos, self.agg_sin = self._rope(agg.rotary_embed, 2)
        t = timesteps
        self.t = [t[i] for i in range(self.steps)]
        self.dt = [t[i + 1] - t[i] for i in range(self.steps)]
        with torch.autocast(device_type=t.device.type, dtype=t.dtype):
            cls = agg.word_embedder(torch.zeros((1, 1), dtype=torch.long, device=t.device))
        self.agg_cls = cls
        self._t_emb: dict[int, list[torch.Tensor]] = {}

    @staticmethod
    def _rope(rotary, seq_len: int):
        # Reference: freqs = einsum('b i, j -> b i j', t.type_as(inv_freq), inv_freq) under autocast(enabled=False),
        # i.e. in inv_freq's dtype; then apply_rotary_pos_emb takes freqs.cos()/sin() in that dtype and multiplies
        # the (bf16) activations, promoting to the wider of the two.
        inv = rotary.inv_freq
        tt = torch.arange(seq_len, device=inv.device).type_as(inv)
        freqs = (tt[:, None] * inv[None, :]) / rotary.interpolation_factor
        freqs = torch.stack((freqs, freqs), dim=-1).reshape(seq_len, -1)
        cos = freqs.cos()
        sin = freqs.sin().view(seq_len, -1, 2)
        sin = torch.stack((-sin[..., 0], sin[..., 1]), dim=-1).reshape(seq_len, -1)
        return cos.view(1, seq_len, 1, -1).contiguous(), sin.view(1, seq_len, 1, -1).contiguous()

    def t_embeddings(self, rows: int) -> list[torch.Tensor]:
        """Timestep embeddings for every Euler step at batch ``rows`` (= 2 x slots), computed like the reference.

        Computed eagerly once per batch size (call ``prepare`` before any CUDA-graph capture of that size)."""
        cached = self._t_emb.get(rows)
        if cached is None:
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(f'FastFlow.prepare({rows // 2}) must run before graph capture')
            cached = []
            for step in range(self.steps):
                cached.append(self.dit.t_embedder(self.t[step].repeat(rows)).unsqueeze(1))
            self._t_emb[rows] = cached
        return cached

    def prepare(self, batch: int) -> None:
        self.t_embeddings(2 * batch)

    # The block as in fm/modules.py DiTBlock with the reference ops; q/k/v fused, rotary tables precomputed.
    @staticmethod
    def _block(lp: DiTLayer, x: torch.Tensor, heads: int, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, H = x.shape
        D = H // heads
        h = F.rms_norm(x, (H,), weight=lp.n1, eps=lp.eps1)
        qkv = linear(h, lp.w_qkv, lp.b_qkv).view(B, T, 3 * heads, D)
        qk = qkv[:, :, :2 * heads]
        # apply_rotary_pos_emb: (t * cos) + (rotate_half(t) * sin) in the promoted dtype, then back to bf16.
        flipped = qk.view(B, T, 2 * heads, D // 2, 2).flip(-1).view(B, T, 2 * heads, D)
        qk = torch.addcmul(qk * cos, flipped, sin).to(x.dtype)
        q = qk[:, :, :heads].transpose(1, 2)
        k = qk[:, :, heads:].transpose(1, 2)
        v = qkv[:, :, 2 * heads:].transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        a = a.transpose(1, 2).reshape(B, T, H)
        x = residual_linear(x, a, lp.w_o, lp.b_o)
        h = F.rms_norm(x, (H,), weight=lp.n2, eps=lp.eps2)
        h = F.gelu(linear(h, lp.w_up, lp.b_up), approximate='tanh')
        return residual_linear(x, h, lp.w_down, lp.b_down)

    def sample(self, noise: torch.Tensor, z: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        """noise (B, 1, C), z (B, 1, hidden), history (B, 1, C) -> latent (B, 1, C). Same math as sample_flow."""
        dit = self.dit
        B = noise.shape[0]
        t_emb = self.t_embeddings(2 * B)
        ce, xe, fl = dit.c_embedder.cond_embedder, dit.x_embedder, dit.final_layer
        c = linear(torch.cat([z, torch.zeros_like(z)], dim=0), ce.weight, ce.bias)
        x_hist = linear(torch.cat([history, history], dim=0), xe.weight, xe.bias)
        y = noise
        for step in range(self.steps):
            x_now = linear(torch.cat([y, y], dim=0), xe.weight, xe.bias)
            x = torch.cat([t_emb[step] + c, x_hist, x_now], dim=1)
            for lp in self.dit_layers:
                x = self._block(lp, x, self.heads, self.dit_cos, self.dit_sin)
            # FinalLayer is per token; only the last (patch) token is used.
            last = F.rms_norm(x[:, -1:, :], (x.shape[-1],), weight=fl.norm_final.weight, eps=fl.norm_final.eps)
            out = linear(last, fl.linear.weight, fl.linear.bias)
            pred, null_pred = torch.chunk(out, 2, dim=0)
            y = y + (pred + (pred - null_pred) * self.cfg) * self.dt[step]
        return y

    def aggregate(self, latent: torch.Tensor) -> torch.Tensor:
        """Aggregator.forward: latent (B, 1, C) -> next LLM input embedding (B, 1, hidden)."""
        agg = self.agg
        x = linear(latent, agg.x_embedder.weight, agg.x_embedder.bias)
        x = torch.cat([self.agg_cls.expand(x.shape[0], -1, -1), x], dim=1)
        for lp in self.agg_layers:
            x = self._block(lp, x, self.agg_heads, self.agg_cos, self.agg_sin)
        fl = agg.final_layer
        first = F.rms_norm(x[:, :1, :], (x.shape[-1],), weight=fl.norm_final.weight, eps=fl.norm_final.eps)
        return linear(first, fl.linear.weight, fl.linear.bias)

    def stop(self, z: torch.Tensor) -> torch.Tensor:
        return self.stop_head(z).softmax(dim=-1)[:, 0, 1] > 0.5
