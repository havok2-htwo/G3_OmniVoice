"""GPU tests for Audio8 0.1B on the slot engine (skipped without CUDA or the model files).

The yardstick is the checkpoint's own code: the engine must reach upstream's numbers in fp32, and with upstream's sampler
fed the engine's noise, ``generate()`` must pick the very same frames.
"""
from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('soxr')
pytest.importorskip('transformers')

REPO = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO / 'models' / 'Audio8-TTS-Preview-0.1b'
VOICE = REPO / 'data' / 'voices' / 'voice_preset_sonjav2.wav'
REF_TEXT = ('Franz jagt im komplett verwahrlosten Taxi quer durch Bayern. Vorbei an dichten Wäldern, kleinen Dörfern '
            'und alten Bahnhöfen.')

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not (MODEL_DIR / 'model.safetensors').is_file() or not VOICE.is_file(),
    reason='needs CUDA, models/Audio8-TTS-Preview-0.1b and data/voices/voice_preset_sonjav2.wav')

ENGINE_KW = dict(max_slots=4, decode_tile_rows=2, prefill_tile_rows=2)
TEXTS = ['Good morning everyone.', 'The weather is lovely today, would you not agree?', 'See you later!', 'Sure.']
LONGER = 'The train to the airport leaves in ten minutes from platform four, so please take your bags now.'


def _voice_audio():
    import soundfile as sf

    return sf.read(str(VOICE), dtype='float32')


def _engine(**kwargs):
    from omnivoice_tts_server.audio8.engine import Audio8Engine

    settings = {**ENGINE_KW, 'cuda_graphs': False, **kwargs}
    eng = Audio8Engine(MODEL_DIR, 'cuda:0', **settings)
    eng.load()
    return eng


@pytest.fixture(scope='module')
def engine():
    eng = _engine()
    yield eng
    eng.close()


@pytest.fixture(scope='module')
def voice(engine):
    audio, sr = _voice_audio()
    return engine.voice_prompt('sonjav2', audio, sr, REF_TEXT)


@pytest.fixture(scope='module')
def exact():
    """The whole model in fp32: rounding out of the way, upstream and engine must agree to fp32's own noise."""
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False  # the server allows it; it costs 3 decimal digits here
    eng = _engine(dtype=torch.float32, codec_dtype=torch.bfloat16)
    yield eng
    eng.close()
    torch.backends.cuda.matmul.allow_tf32 = tf32


@pytest.fixture(scope='module')
def exact_voice(exact):
    audio, sr = _voice_audio()
    return exact.voice_prompt('sonjav2', audio, sr, REF_TEXT)


def _prompt(engine, voice, text):
    from omnivoice_tts_server.audio8.text import prefix_ids, suffix_ids

    ids = suffix_ids(engine.tokenizer, text)
    head = engine._prompt_head(prefix_ids(engine.tokenizer, voice.ref_text), voice.codes)
    tail = torch.zeros(1, engine.books + 1, len(ids), dtype=torch.long, device=engine.device)
    tail[0, 0] = torch.tensor(ids, device=engine.device)
    return torch.cat((head, tail), dim=2), head.shape[2], ids


def _upstream_forward(model, full):
    """Upstream's whole forward, no cache: scores and hidden at every position."""
    mask = torch.ones(1, full.shape[2], device=full.device, dtype=torch.long)
    normalized = model._slow_backbone(model._embed(full), attention_mask=mask)
    return model.semantic_output(normalized)[0].float(), normalized[0].float()


# -- loading ---------------------------------------------------------------------------------------------------------------

def test_non_persistent_buffers_are_real_after_loading(engine):
    """Regression (transformers 5): mup_vector and the fast RoPE table come out of from_pretrained as garbage. mup_vector
    near zero switches Mamba off without any error."""
    import sys

    model = engine.model
    rope = sys.modules[type(model).__module__]._precompute_rope
    c = model.config
    torch.testing.assert_close(model.fast_freqs_cis.cpu(), rope(c.num_codebooks, c.fast_head_dim, c.rope_base))
    for layer in model.slow.layers:
        assert (layer.mamba.mup_vector == 1).all()
    expected = 1.0 / (c.rope_base ** (torch.arange(0, c.head_dim, 2, dtype=torch.float32) / c.head_dim))
    torch.testing.assert_close(model.slow.rotary_emb.inv_freq.cpu(), expected)


def test_prompt_tokens_match_the_processor(engine):
    from transformers import AutoProcessor

    from omnivoice_tts_server.audio8.text import prefix_ids, suffix_ids

    processor = AutoProcessor.from_pretrained(str(MODEL_DIR), trust_remote_code=True, local_files_only=True)
    text = 'Die Brücke über den Fluss.'
    prefix, suffix = processor._prompt_segments(text, REF_TEXT, True)
    assert prefix.tolist() == prefix_ids(engine.tokenizer, REF_TEXT)
    assert suffix.tolist() == suffix_ids(engine.tokenizer, text)
    full, _ = processor._prompt_segments(text, None, False)
    assert full.tolist() == prefix_ids(engine.tokenizer, None) + suffix_ids(engine.tokenizer, text)


# -- the language model ----------------------------------------------------------------------------------------------------

def _trace(engine, voice, prompt, P, ids, frames, tile):
    """Voice head, prefill in calls of ``tile`` tokens next to a padding row, then one slow step per frame."""
    lm, model = engine.lm, engine.model
    c = model.config
    slot = 1
    T = len(ids)
    lm.load_voice(slot, voice.keys, voice.values, voice.state)
    fed = 0
    while fed < T:
        tokens = torch.zeros(2, tile, device=engine.device, dtype=torch.long)
        chunk = ids[fed:fed + tile]
        tokens[0, :len(chunk)] = torch.tensor(chunk, device=engine.device)
        tokens[1, :3] = 7
        lm.prefill(tokens, torch.tensor([slot, 4], device=engine.device), torch.tensor([P + fed, 0], device=engine.device),
                   torch.tensor([len(chunk) - 1, 2], device=engine.device))
        fed += len(chunk)
    logits, hidden = [lm.logits[slot].clone()], [lm.hidden[slot, 0].float().clone()]
    for i in range(frames.shape[1]):
        column = torch.cat((frames[:1, i] + c.semantic_begin_id, frames[:, i])).view(1, c.num_codebooks + 1, 1)
        lm.next_x[slot, 0] = model._embed(column)[0, 0]
        lm.pos[:] = 0
        lm.pos[slot] = P + T + i
        lm.slow(4)
        logits.append(lm.logits[slot].clone())
        hidden.append(lm.hidden[slot, 0].float().clone())
    return torch.stack(logits), torch.stack(hidden)


@pytest.mark.parametrize('tile', [64, 32, 7])
def test_prefill_and_steps_match_upstream_in_fp32(exact, exact_voice, tile):
    """Whole path against upstream's uncached forward: voice head, text prefill in one or several calls, Mamba and
    attention state of every step. Same numbers up to fp32 noise (scores run around 10)."""
    text = 'Die Abfahrt verzögert sich leider um etwa zehn Minuten.'
    prompt, P, ids = _prompt(exact, exact_voice, text)
    c = exact.model.config

    def work():
        frames = exact_voice.codes[:, 20:50].to(exact.device)
        columns = torch.cat((frames[:1] + c.semantic_begin_id, frames))
        reference, reference_hidden = _upstream_forward(exact.model, torch.cat((prompt, columns[None]), dim=2))
        logits, hidden = _trace(exact, exact_voice, prompt, P, ids, frames, tile)
        at = prompt.shape[2] - 1
        return logits, hidden, reference[at:], reference_hidden[at:]

    logits, hidden, reference, reference_hidden = exact._run_on_gpu_thread(work)
    assert logits.shape == reference.shape
    assert (logits - reference).abs().max().item() < 2e-3
    assert (hidden - reference_hidden).abs().max().item() < 2e-3


def test_plain_voice_matches_upstream_in_fp32(exact):
    voice = exact._plain_voice
    prompt, P, ids = _prompt(exact, voice, 'Good morning everyone, the weather is lovely today.')

    def work():
        frames = torch.zeros(10, 0, dtype=torch.long, device=exact.device)
        reference, _ = _upstream_forward(exact.model, prompt)
        return _trace(exact, voice, prompt, P, ids, frames, 64)[0], reference[-1:]

    logits, reference = exact._run_on_gpu_thread(work)
    assert (logits - reference).abs().max().item() < 2e-3


def test_bf16_stays_at_bf16_rounding(engine, voice):
    """The default engine runs in bf16: the same draw distribution up to what bf16 rounding does to upstream too."""
    text = 'Die Abfahrt verzögert sich leider um etwa zehn Minuten.'
    prompt, P, ids = _prompt(engine, voice, text)
    model32 = copy.deepcopy(engine.model).float()
    try:

        def work():
            c = engine.model.config
            frames = voice.codes[:, 20:50].to(engine.device)
            columns = torch.cat((frames[:1] + c.semantic_begin_id, frames))
            full = torch.cat((prompt, columns[None]), dim=2)
            at = prompt.shape[2] - 1
            exact, _ = _upstream_forward(model32, full)
            rounded, _ = _upstream_forward(engine.model, full)
            logits, _ = _trace(engine, voice, prompt, P, ids, frames, 64)
            return logits, exact[at:], rounded[at:]

        logits, exact, rounded = engine._run_on_gpu_thread(work)
    finally:
        del model32
        torch.cuda.empty_cache()

    def tv(a, b):
        return 0.5 * (torch.softmax(a / 0.8, -1) - torch.softmax(b / 0.8, -1)).abs().sum(-1).mean().item()

    assert tv(logits, exact) < 1.5 * tv(rounded, exact) + 1e-3
    assert tv(logits, exact) < 0.1


def test_mamba_state_matters(engine, voice):
    """Rows must really carry the Mamba state: dropping it after the prompt changes the scores."""
    prompt, P, ids = _prompt(engine, voice, 'Die Brücke über den Fluss.')

    def work():
        frames = torch.zeros(10, 0, dtype=torch.long, device=engine.device)
        blank = replace(voice, state=(torch.zeros_like(voice.state[0]), torch.zeros_like(voice.state[1])))
        return (_trace(engine, voice, prompt, P, ids, frames, 32)[0],
                _trace(engine, blank, prompt, P, ids, frames, 32)[0])

    kept, dropped = engine._run_on_gpu_thread(work)
    assert (kept - dropped).abs().mean().item() > 0.05


# -- same noise, same frames -------------------------------------------------------------------------------------------------

class SharedNoise:
    """ArkttsModel._sample with the engine's numbers: draw n of frame t of a row, by rank within the top-k."""

    def __init__(self, key, top_k, draws_per_frame, device):
        self.key, self.top_k, self.per_frame, self.device = key, top_k, draws_per_frame, device
        self.call = 0

    def __call__(self, scores, generator=None):
        from omnivoice_tts_server.audio8.noise import gumbel

        frame, draw = divmod(self.call, self.per_frame)
        self.call += 1
        values, index = scores.topk(self.top_k, dim=-1)
        counter = self.key + frame * (self.per_frame * self.top_k) + draw * self.top_k + torch.arange(self.top_k,
                                                                                                      device=self.device)
        picked = (values + gumbel(counter)[None]).argmax(-1, keepdim=True)
        return index.gather(1, picked)[:, 0]


@pytest.mark.parametrize(('text', 'seed', 'clone'), [(LONGER, 3, True), (TEXTS[1], 4, True), (TEXTS[1], 5, False)])
def test_engine_frames_equal_upstream_generate_with_the_same_noise(exact, exact_voice, text, seed, clone):
    """Upstream's generate() with the engine's noise picks the engine's frames, EOS included: sampling, repetition window,
    fast branch and stopping are the same, not just similar. Then the audio: stream against upstream's decode."""
    from omnivoice_tts_server.audio8.codec_stream import snr_db
    from omnivoice_tts_server.audio8.noise import row_key

    voice = exact_voice if clone else exact._plain_voice
    stream = exact.submit(text, prompt=exact_voice if clone else None, seed=seed)
    audio = np.concatenate(list(stream))
    prompt, _, _ = _prompt(exact, voice, text)

    def work():
        model = exact.model
        ours = exact.lm.frames[0, :stream.metrics['steps'] - 1].clone()
        noise = SharedNoise(row_key(seed, 0), exact.top_k, model.config.num_codebooks + 1, exact.device)
        model._sample = noise
        try:
            out = model.generate(prompt, torch.ones(1, prompt.shape[2], device=exact.device, dtype=torch.long),
                                 max_new_tokens=500, temperature=exact.temperature, top_p=exact.top_p, top_k=exact.top_k,
                                 do_sample=True, return_dict_in_generate=True)
        finally:
            del model._sample
        # Not model.decode_audio: that would move the engine's codec to the model's dtype.
        length = int(out.code_lengths[0])
        return ours, out, exact.codec.decode(out.codes[:, :, :length])[0, 0].float().cpu().numpy()

    ours, out, upstream_audio = exact._run_on_gpu_thread(work)
    assert bool(out.finished[0])
    codes = out.codes[0, :, :int(out.code_lengths[0])].T
    assert torch.equal(codes, ours)
    n = min(audio.size, upstream_audio.size)  # trailing silence is trimmed
    assert n > 0.9 * upstream_audio.size and snr_db(upstream_audio[:n], audio[:n]) > 30


# -- rows ------------------------------------------------------------------------------------------------------------------

def test_streams_a_clone_with_a_first_short_chunk(engine, voice):
    stream = engine.submit(TEXTS[1], prompt=voice, seed=1)
    chunks = list(stream)
    audio = np.concatenate(chunks)
    assert chunks[0].size == 4 * 2048 and len(chunks) >= 2
    assert stream.metrics['steps'] > 30 and 1.0 < audio.size / 44100 < 8.0
    assert np.isfinite(audio).all() and 0.01 < np.abs(audio).max() <= 1.0


def test_rows_are_independent_of_batch_composition(engine, voice):
    alone = [engine.render(t, prompt=voice, seed=10 + i) for i, t in enumerate(TEXTS)]
    np.testing.assert_array_equal(alone[1], engine.render(TEXTS[1], prompt=voice, seed=11))
    streams = [engine.submit(t, prompt=voice, seed=10 + i) for i, t in enumerate(TEXTS)]
    for a, s in zip(alone, streams):
        np.testing.assert_array_equal(a, np.concatenate(list(s)))
    first = engine.submit(TEXTS[1], prompt=voice, seed=11)  # a row joining a running one changes neither
    got = [first.next_chunk(timeout=60)]
    late = engine.submit(TEXTS[0], prompt=voice, seed=10)
    got += list(first)
    np.testing.assert_array_equal(np.concatenate(got), alone[1])
    np.testing.assert_array_equal(np.concatenate(list(late)), alone[0])


def test_a_slot_keeps_nothing_of_the_row_before(engine, voice):
    """Mamba state is the one part of a slot a mask does not hide: the next row must start clean."""
    clean = engine.render(TEXTS[0], prompt=voice, seed=21)
    for i in range(3):
        engine.render(LONGER, prompt=voice, seed=70 + i)
    plain = engine.render(TEXTS[2], prompt=None, seed=8)
    np.testing.assert_array_equal(clean, engine.render(TEXTS[0], prompt=voice, seed=21))
    np.testing.assert_array_equal(plain, engine.render(TEXTS[2], prompt=None, seed=8))


def test_long_text_is_fragmented_and_every_fragment_is_spoken(engine, voice):
    text = ('The Entrepreneurs opened the notebook, loaded the document from the cloud and worked on the presentation in '
            'the cafe next to the terminal. The driver waited patiently, while the ensemble rehearsed in the foyer, and '
            'nobody noticed how late it had become until the lights went out.')
    stream = engine.submit(text, prompt=voice, seed=3)
    audio = np.concatenate(list(stream))
    assert stream.metrics['fragments'] >= 2
    assert audio.size / 44100 > 0.03 * len(text)
    assert 'capped' not in stream.metrics
    streams = [engine.submit(text, prompt=voice, seed=3) for _ in range(2)]
    for s in streams:
        np.testing.assert_array_equal(audio, np.concatenate(list(s)))


def test_cuda_graphs_match_eager_bit_exactly(engine, voice):
    eager = [engine.render(t, prompt=voice, seed=20 + i) for i, t in enumerate(TEXTS)]
    graphed = _engine(cuda_graphs=True)
    try:
        assert graphed.stats()['cuda_graphs'] and graphed.stats()['graphs'] >= 7
        audio, sr = _voice_audio()
        vp = graphed.voice_prompt('sonjav2', audio, sr, REF_TEXT)
        assert torch.equal(vp.codes, voice.codes) and torch.equal(vp.state[1], voice.state[1])
        streams = [graphed.submit(t, prompt=vp, seed=20 + i) for i, t in enumerate(TEXTS)]
        together = [np.concatenate(list(s)) for s in streams]
    finally:
        graphed.close()
    del vp
    for a, b in zip(eager, together):
        np.testing.assert_array_equal(a, b)
