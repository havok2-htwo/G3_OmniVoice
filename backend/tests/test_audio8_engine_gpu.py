"""GPU tests for the Audio8 slot engine (skipped without CUDA or the model files).

They pin what the fast paths must keep: the math of the reference model, exact streaming, per-row determinism.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('soxr')
pytest.importorskip('transformers')

REPO = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO / 'models' / 'Audio8-TTS-Realistic-0.6b'
PREVIEW_DIR = REPO / 'models' / 'Audio8-TTS-Preview-0.6b'
VOICE = REPO / 'data' / 'voices' / 'voice_preset_sonjav2.wav'
REF_TEXT = ('Franz jagt im komplett verwahrlosten Taxi quer durch Bayern. Vorbei an dichten Wäldern, kleinen Dörfern '
            'und alten Bahnhöfen.')

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not (MODEL_DIR / 'model.safetensors').is_file() or not VOICE.is_file(),
    reason='needs CUDA, models/Audio8-TTS-Realistic-0.6b and data/voices/voice_preset_sonjav2.wav')

# Tiles smaller than the rows in flight, so calls with several tiles are exercised too.
ENGINE_KW = dict(max_slots=4, decode_tile_rows=2, prefill_tile_rows=2)
TEXTS = ['Guten Morgen zusammen.', 'Heute ist ein schöner Tag, findest du nicht?', 'Bis später!', 'Na klar.']


def _voice_audio():
    import soundfile as sf

    return sf.read(str(VOICE), dtype='float32')


@pytest.fixture(scope='module')
def engine():
    from omnivoice_tts_server.audio8.engine import Audio8Engine

    eng = Audio8Engine(MODEL_DIR, 'cuda:0', cuda_graphs=False, **ENGINE_KW)
    eng.load()
    yield eng
    eng.close()


@pytest.fixture(scope='module')
def voice(engine):
    audio, sr = _voice_audio()
    return engine.voice_prompt('sonjav2', audio, sr, REF_TEXT)


def _on_gpu_thread(engine, fn):
    return engine._run_on_gpu_thread(fn)


def _full_prompt(engine, voice, text):
    """(prompt, P, ids): the prompt the way ArkttsModel._prepare_prompt lays it out for one row."""
    from omnivoice_tts_server.audio8.text import prefix_ids, suffix_ids

    ids = suffix_ids(engine.tokenizer, text)
    head = engine._prompt_head(prefix_ids(engine.tokenizer, voice.ref_text), voice.codes)
    tail = torch.zeros(1, engine.books + 1, len(ids), dtype=torch.long, device=engine.device)
    tail[0, 0] = torch.tensor(ids, device=engine.device)
    return torch.cat((head, tail), dim=2), head.shape[2], ids


# -- loading ---------------------------------------------------------------------------------------------------------------

def test_rope_tables_are_real_after_loading(engine):
    """Regression (transformers 5): from_pretrained leaves the non-persistent tables empty and cloning ends after 2 frames."""
    import sys

    model = engine.model
    rope = sys.modules[type(model).__module__]._precompute_rope
    c = model.config
    torch.testing.assert_close(model.freqs_cis.cpu(), rope(c.max_seq_len, c.head_dim, c.rope_base))
    torch.testing.assert_close(model.fast_freqs_cis.cpu(), rope(c.num_codebooks, c.fast_head_dim, c.rope_base))
    assert float(model.freqs_cis[:, 0, 0].abs().mean()) > 0.5  # cos of a slowly growing angle: never empty
    names = {name for name, buffer in model.named_buffers() if not buffer.isfinite().all()}
    assert not names


def test_a_clone_does_not_stop_after_two_frames(engine, voice):
    stream = engine.submit(TEXTS[1], prompt=voice, seed=4)
    audio = np.concatenate(list(stream))
    assert stream.metrics['steps'] > 30 and audio.size > 30 * 2048


def test_prompt_tokens_match_the_processor(engine):
    """Our prefix / suffix split reproduces ArkttsProcessor, with and without a reference."""
    from transformers import AutoProcessor

    from omnivoice_tts_server.audio8.text import prefix_ids, suffix_ids

    processor = AutoProcessor.from_pretrained(str(MODEL_DIR), trust_remote_code=True, local_files_only=True)
    text = 'Die Brücke über den Fluss.'
    prefix, suffix = processor._prompt_segments(text, REF_TEXT, True)
    assert prefix.tolist() == prefix_ids(engine.tokenizer, REF_TEXT)
    assert suffix.tolist() == suffix_ids(engine.tokenizer, text)
    full, _ = processor._prompt_segments(text, None, False)
    assert full.tolist() == prefix_ids(engine.tokenizer, None) + suffix_ids(engine.tokenizer, text)


def test_voice_prompt_is_cached(engine, voice):
    audio, sr = _voice_audio()
    assert engine.voice_prompt('sonjav2', audio, sr, REF_TEXT) is voice
    assert engine.has_voice('sonjav2')
    copy_ = engine.voice_prompt('copy', audio, sr, REF_TEXT)
    assert torch.equal(copy_.codes, voice.codes) and torch.equal(copy_.keys, voice.keys)
    engine.drop_voice('copy')
    assert not engine.has_voice('copy')
    assert voice.codes.shape == (10, pytest.approx(voice.duration_s * 44100 / 2048, abs=2))


def test_a_voice_evicted_while_it_renders_keeps_rendering(engine, voice, monkeypatch):
    """A row holds its own prompt, so the cache may let go of a voice in use. Same audio as with the voice cached."""
    from omnivoice_tts_server.audio8 import engine as engine_module

    audio, sr = _voice_audio()
    victim = engine.voice_prompt('victim', audio, sr, REF_TEXT)
    expected = engine.render(TEXTS[1], prompt=victim, seed=7)
    monkeypatch.setattr(engine_module, 'MAX_VOICE_CACHE_MB', 0)
    stream = engine.submit(TEXTS[1], prompt=victim, seed=7)
    engine.voice_prompt('newcomer', audio, sr, REF_TEXT)
    assert not engine.has_voice('victim') and engine.has_voice('newcomer')
    del victim
    got = np.concatenate(list(stream))
    engine.drop_voice('newcomer')
    with engine._voice_lock:  # the module fixture's voice comes back for the tests after this one
        engine._voices['sonjav2'] = voice
    np.testing.assert_array_equal(got, expected)


# -- the language model ----------------------------------------------------------------------------------------------------

def _reference(model, dtype, prompt, column=None):
    """(logits over the frame's scores, hidden, KV of layer 5) of the model's own forward over ``prompt``, then, with
    ``column``, one cached step on top."""
    c = model.config
    L = prompt.shape[2]
    index = torch.arange(L, device=prompt.device)
    model._setup_generation_caches(1, 2048, dtype)
    try:
        logits, hidden = model._slow_step(prompt, index, index[None], torch.ones(1, L, device=prompt.device,
                                                                                 dtype=torch.long))
        kv = [(layer.attention.kv_cache.keys[0, :, :L].clone(), layer.attention.kv_cache.values[0, :, :L].clone())
              for layer in model.layers]
        if column is not None:
            position = torch.tensor([L], device=prompt.device)
            logits, hidden = model._slow_step(column, position, position[None],
                                              torch.ones(1, L + 1, device=prompt.device, dtype=torch.long))
    finally:
        for layer in [*model.layers, *model.fast_layers]:
            layer.attention.kv_cache = None
    scores = torch.cat((logits[0, c.semantic_begin_id:c.semantic_end_id + 1], logits[0, c.eos_token_id:c.eos_token_id + 1]))
    return scores.float(), hidden[0, 0].float(), kv


@pytest.fixture(scope='module')
def model32(engine):
    model = copy.deepcopy(engine.model).float()
    yield model
    del model
    torch.cuda.empty_cache()


def _err(a, b) -> float:
    return (a - b).abs().mean().item()


def _assert_same_up_to_bf16(ours, reference, exact):
    """Both bf16 paths are about equally far from the fp32 result: what is left between them is rounding."""
    assert _err(ours, exact) < 1.5 * _err(reference, exact) + 1e-3
    assert _err(ours, reference) < 3 * _err(reference, exact) + 1e-3


def test_prefill_on_cached_voice_kv_matches_the_reference_forward(engine, voice, model32):
    """Voice KV + text prefill vs the model's own forward over the whole prompt, next to a padded neighbour."""
    text = 'Die Abfahrt verzögert sich leider um etwa zehn Minuten.'
    model, lm = engine.model, engine.lm
    prompt, P, ids = _full_prompt(engine, voice, text)
    T = len(ids)

    def work():
        reference = _reference(model, engine.dtype, prompt)
        exact = _reference(model32, torch.float32, prompt)
        lm.K[:, 2, :, :P] = voice.keys
        lm.V[:, 2, :, :P] = voice.values
        tokens = torch.zeros(2, 64, device=engine.device, dtype=torch.long)
        tokens[0, :T] = torch.tensor(ids, device=engine.device)
        tokens[1, :5] = tokens[0, :5]
        slots = torch.tensor([2, 4], device=engine.device)  # 4: the padding row
        lm.prefill(tokens, slots, torch.tensor([P, 0], device=engine.device), torch.tensor([T - 1, 4], device=engine.device))
        return reference, exact, lm.logits[2].clone(), lm.hidden[2, 0].float(), lm.K[5, 2, :, :P + T].float().clone()

    reference, exact, logits, hidden, keys = _on_gpu_thread(engine, work)
    _assert_same_up_to_bf16(logits, reference[0], exact[0])
    _assert_same_up_to_bf16(hidden, reference[1], exact[1])
    _assert_same_up_to_bf16(keys, reference[2][5][0].float(), exact[2][5][0].float())
    first_draw = lambda scores: torch.softmax(scores / 0.8, -1)  # noqa: E731
    assert (first_draw(logits) - first_draw(exact[0])).abs().sum().item() < 0.1


def test_slow_step_matches_the_reference_step(engine, voice, model32):
    """One slow step of the engine vs the model's cached forward, at the same position over the same keys."""
    text = 'Die Abfahrt verzögert sich leider um etwa zehn Minuten.'
    model, lm = engine.model, engine.lm
    prompt, P, ids = _full_prompt(engine, voice, text)
    L = prompt.shape[2]

    def work():
        c = model.config
        codes = voice.codes[:, 3].to(engine.device)
        column = torch.cat((codes[:1] + c.semantic_begin_id, codes)).view(1, c.num_codebooks + 1, 1)
        reference = _reference(model, engine.dtype, prompt, column)
        exact = _reference(model32, torch.float32, prompt, column)
        for n, (keys, values) in enumerate(reference[2]):
            lm.K[n, 1, :, :L] = keys
            lm.V[n, 1, :, :L] = values
        lm.next_x[1, 0] = model._embed(column)[0, 0]
        lm.pos[:] = 0
        lm.pos[1] = L
        lm.slow(4)
        return reference, exact, lm.logits[1].clone(), lm.hidden[1, 0].float()

    reference, exact, logits, hidden = _on_gpu_thread(engine, work)
    _assert_same_up_to_bf16(logits, reference[0], exact[0])
    _assert_same_up_to_bf16(hidden, reference[1], exact[1])


@pytest.mark.parametrize(('top_p', 'temperature'), [(0.95, 0.8), (0.9, 1.0), (0.5, 0.7)])
def test_pick_keeps_exactly_what_the_reference_processor_keeps(engine, top_p, temperature):
    """Top-k + top-p set and draw probabilities of SlotLM._pick vs ArkttsLegacyTopKTopPLogitsProcessor + ArkttsModel._sample."""
    import sys

    module = sys.modules[type(engine.model).__module__]
    lm = engine.lm
    gen = torch.Generator(device='cuda').manual_seed(int(top_p * 100))
    scores = (torch.randn(4, 4097, device='cuda', generator=gen) * 3).float()

    def work():
        processed = module.ArkttsLegacyTopKTopPLogitsProcessor(lm.top_k, top_p)(None, scores)
        expected = torch.softmax(processed / temperature, dim=-1)
        draws = 20000
        counts = torch.zeros_like(expected)
        for row in range(4):
            noise = -torch.log(-torch.log(torch.rand(draws, lm.top_k, device='cuda', generator=gen).clamp_min(1e-9)))
            lm_scores = scores[row:row + 1].expand(4, -1)
            picks = []
            for i in range(0, draws, 4):
                picks.append(lm._pick(lm_scores, noise[i:i + 4], top_p, temperature))
            counts[row].scatter_add_(0, torch.cat(picks), torch.ones(draws, device='cuda'))
        return processed, expected, counts / draws

    processed, expected, measured = _on_gpu_thread(engine, work)
    assert ((measured > 0) & torch.isinf(processed)).sum().item() == 0  # nothing outside the kept set is ever drawn
    assert (measured - expected).abs().sum(dim=-1).max().item() < 0.06


# -- streaming decode ------------------------------------------------------------------------------------------------------

def _stream_decode(sc, codes, schedule, dtype=None):
    """Drive a SlotCodec for one row through the given chunk sizes; codes (books, T)."""
    T = codes.shape[1]
    parts, done, k = [], 0, 0
    chunk = sc.max_chunk
    while done < T:
        n = min(schedule[k % len(schedule)], T - done)
        k += 1
        frames = torch.zeros(1, chunk, codes.shape[0], device=codes.device, dtype=torch.long)
        frames[0, :n] = codes[:, done:done + n].T
        meta = torch.tensor([[1, done, n, int(done == 0)]], device=codes.device)
        out = torch.zeros(1, sc.out_len(chunk), device=codes.device)
        sc.run(frames, meta, out)
        parts.append(out[0, :n * sc.frame_samples].clone())
        done += n
    return torch.cat(parts)


@pytest.fixture(scope='module')
def real_codes(engine, voice):
    return voice.codes.to(engine.device)


def test_streamed_decode_is_exact_up_to_fp32_rounding(engine, real_codes):
    """The whole point of the ring KV and the per-slot convolution state: same audio as a full decode, for any chunking.
    The yardstick is the codec itself: being causal, it must give the same audio for T frames and for 2T frames, and
    fp32 rounding in the deep convolution stack lets the two differ at about 65 dB."""
    from omnivoice_tts_server.audio8.codec_stream import SlotCodec, snr_db

    codec = copy.deepcopy(engine.codec).float()

    def work():
        full = codec.decode(real_codes[None])[0, 0]
        longer = codec.decode(torch.cat((real_codes, real_codes), dim=1)[None])[0, 0, :full.shape[0]]
        sc = SlotCodec(codec, slots=2, max_chunk=12, max_frames=1024, device=engine.device, dtype=torch.float32)
        return full, longer, [_stream_decode(sc, real_codes, schedule) for schedule in ([12], [4, 12], [5, 7, 12, 3, 1])]

    full, longer, streams = _on_gpu_thread(engine, work)
    del codec
    torch.cuda.empty_cache()
    floor = snr_db(full.cpu().numpy(), longer.cpu().numpy())
    assert floor > 55
    for stream in streams:
        assert stream.shape == full.shape
        assert snr_db(full.cpu().numpy(), stream.cpu().numpy()) > floor - 3


def test_streamed_decode_in_bf16_is_as_close_as_bf16_gets(engine, real_codes):
    """bf16 rounding is the floor: a full bf16 decode is 39 dB from the fp32 one. The stream sits within 2 dB of it, and
    its chunking does not change a bit."""
    from omnivoice_tts_server.audio8.codec_stream import snr_db

    codec32 = copy.deepcopy(engine.codec).float()
    sc = engine.slot_codec

    def work():
        exact = codec32.decode(real_codes[None])[0, 0]
        full = engine.codec.decode(real_codes[None])[0, 0].float()
        return exact, full, _stream_decode(sc, real_codes, [12]), _stream_decode(sc, real_codes, [4, 12, 5])

    exact, full, a, b = _on_gpu_thread(engine, work)
    del codec32
    torch.cuda.empty_cache()
    assert torch.equal(a, b)
    exact, full, a = (t.cpu().numpy() for t in (exact, full, a))
    assert snr_db(full, a) > 35
    assert snr_db(exact, a) > snr_db(exact, full) - 2


def test_decoding_does_not_depend_on_the_neighbour_in_the_tile(engine, real_codes):
    sc = engine.slot_codec
    C = sc.max_chunk
    window = real_codes[:, :C].T

    def work():
        rows = sc.slots
        codes = torch.zeros(2, C, sc.n_codebooks, device=engine.device, dtype=torch.long)
        codes[0] = window
        codes[1] = real_codes[:, C:2 * C].T
        alone = torch.zeros(2, sc.out_len(C), device=engine.device)
        meta = torch.tensor([[0, 0, C, 1], [rows, 0, 0, 0]], device=engine.device)
        sc.run(codes, meta, alone)
        pair = torch.zeros_like(alone)
        meta = torch.tensor([[0, 0, C, 1], [1, 0, C, 1]], device=engine.device)
        sc.run(codes, meta, pair)
        swapped = torch.zeros_like(alone)
        meta = torch.tensor([[1, 0, C, 1], [0, 0, C, 1]], device=engine.device)
        sc.run(codes.flip(0), meta, swapped)
        return alone, pair, swapped

    alone, pair, swapped = _on_gpu_thread(engine, work)
    assert torch.equal(alone[0], pair[0])
    assert torch.equal(pair[0], swapped[1])


def test_no_seams_between_chunks(engine, voice):
    """Sample steps across chunk edges look like the steps everywhere else."""
    stream = engine.submit('Die Bibliothek bleibt wegen Renovierungsarbeiten bis Ende des Monats geschlossen.',
                           prompt=voice, seed=8)
    chunks = list(stream)
    audio = np.concatenate(chunks)
    assert len(chunks) > 3
    steps = np.abs(np.diff(audio))
    edges = np.cumsum([c.size for c in chunks[:-1]]) - 1
    # Chunks in the trimmed tail can be cut anywhere, but never on a step larger than speech makes itself.
    assert steps[edges].max() <= np.percentile(steps, 99.99)


# -- rows ------------------------------------------------------------------------------------------------------------------

def test_streams_german_clone(engine, voice):
    stream = engine.submit(TEXTS[0], prompt=voice, seed=1)
    chunks = list(stream)
    audio = np.concatenate(chunks)
    assert engine.output_sample_rate == 44100
    assert chunks[0].size == 4 * 2048 and len(chunks) >= 2
    assert 0.8 < audio.size / 44100 < 6.0
    assert np.isfinite(audio).all() and 0.01 < np.abs(audio).max() <= 1.0
    assert stream.metrics['ttfa_ms'] < stream.metrics['total_ms']


def test_same_seed_same_audio_other_seed_other_audio(engine, voice):
    a = engine.render(TEXTS[1], prompt=voice, seed=5)
    np.testing.assert_array_equal(a, engine.render(TEXTS[1], prompt=voice, seed=5))
    other = engine.render(TEXTS[1], prompt=voice, seed=6)
    assert a.size != other.size or not np.array_equal(a, other)


def test_rows_are_independent_of_batch_composition(engine, voice):
    alone = [engine.render(t, prompt=voice, seed=10 + i) for i, t in enumerate(TEXTS)]
    streams = [engine.submit(t, prompt=voice, seed=10 + i) for i, t in enumerate(TEXTS)]
    together = [np.concatenate(list(s)) for s in streams]
    for a, b in zip(alone, together):
        np.testing.assert_array_equal(a, b)
    # A row joining while another one is running changes neither of them.
    first = engine.submit(TEXTS[1], prompt=voice, seed=11)
    got = [first.next_chunk(timeout=60)]
    late = engine.submit(TEXTS[0], prompt=voice, seed=10)
    got += list(first)
    np.testing.assert_array_equal(np.concatenate(got), alone[1])
    np.testing.assert_array_equal(np.concatenate(list(late)), alone[0])


def test_rows_with_other_voices_and_plain_rows_are_independent(engine, voice):
    import json

    import soundfile as sf

    presets_dir = REPO / 'backend' / 'src' / 'omnivoice_tts_server' / 'voice_presets'
    preset = next(p for p in json.loads((presets_dir / 'presets.json').read_text(encoding='utf-8'))
                  if p['voice_id'] == 'voice_preset_sonjav3')
    audio, sr = sf.read(str(presets_dir / preset['audio_file']), dtype='float32')
    other = engine.voice_prompt('sonjav3', audio, sr, preset['ref_text'])
    assert other.length != voice.length
    rows = [(TEXTS[0], voice, 31), ('The weather is lovely today.', other, 32), ('See you later!', None, 33),
            (TEXTS[1], other, 34)]
    alone = [engine.render(t, prompt=vp, seed=seed) for t, vp, seed in rows]
    streams = [engine.submit(t, prompt=vp, seed=seed) for t, vp, seed in rows]
    for a, s in zip(alone, streams):
        np.testing.assert_array_equal(a, np.concatenate(list(s)))
    assert not np.array_equal(alone[0], engine.render(rows[0][0], prompt=other, seed=31))


def test_long_text_is_fragmented_and_every_fragment_is_spoken(engine, voice):
    text = ('Also improvisierten wir: Die Entrepreneurin aus unserem Start-up öffnete ihr Notebook, lud das Exposé aus '
            'der Cloud und überarbeitete die Präsentation im Café neben dem Terminal. Der Chauffeur wartete geduldig, '
            'während das Ensemble im Foyer probte.')
    stream = engine.submit(text, prompt=voice, seed=3)
    audio = np.concatenate(list(stream))
    assert stream.metrics['fragments'] >= 2
    assert audio.size / 44100 > 0.04 * len(text)
    assert 'capped' not in stream.metrics
    streams = [engine.submit(text, prompt=voice, seed=3) for _ in range(2)]
    for s in streams:
        np.testing.assert_array_equal(audio, np.concatenate(list(s)))


def test_more_rows_than_slots_all_finish(engine, voice):
    streams = [engine.submit(TEXTS[i % 4], prompt=voice, seed=40 + i) for i in range(11)]
    for s in streams:
        assert np.concatenate(list(s)).size > 2048
    deadline = time.time() + 5
    while engine.stats()['free_slots'] != engine.max_slots and time.time() < deadline:
        time.sleep(0.01)
    assert engine.stats()['free_slots'] == engine.max_slots


def test_cancel_stops_at_the_next_step(engine, voice):
    stream = engine.submit('Dies ist ein viel längerer Satz, den wir gleich nach dem ersten Stück abbrechen werden, '
                           'damit er sicher noch läuft, wenn das passiert.', prompt=voice, seed=0)
    assert stream.next_chunk(timeout=60) is not None
    stream.cancel()
    list(stream)
    deadline = time.time() + 10
    while engine.stats()['active_rows'] and time.time() < deadline:
        time.sleep(0.02)
    assert engine.stats()['active_rows'] == 0
    assert engine.stats()['free_slots'] == engine.max_slots


def test_24khz_output(engine, voice):
    from omnivoice_tts_server.audio8.engine import Audio8Engine

    native = engine.render(TEXTS[1], prompt=voice, seed=7)
    eng = Audio8Engine(MODEL_DIR, 'cuda:0', cuda_graphs=False, output_sample_rate=24000, **ENGINE_KW)
    eng.load()
    try:
        audio, sr = _voice_audio()
        resampled = eng.render(TEXTS[1], prompt=eng.voice_prompt('v', audio, sr, REF_TEXT), seed=7)
    finally:
        eng.close()
    assert resampled.size == pytest.approx(native.size * 24000 / 44100, abs=32)


# -- graphs -----------------------------------------------------------------------------------------------------------------

def test_cuda_graphs_match_eager_bit_exactly(engine, voice):
    """Graph replay runs the very same kernels: graphs on == graphs off, alone and batched."""
    from omnivoice_tts_server.audio8.engine import Audio8Engine

    eager = [engine.render(t, prompt=voice, seed=20 + i) for i, t in enumerate(TEXTS)]
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    graphed = Audio8Engine(MODEL_DIR, 'cuda:0', cuda_graphs=True, **ENGINE_KW)
    graphed.load()
    try:
        assert graphed.stats()['cuda_graphs'] and graphed.stats()['graphs'] >= 7
        audio, sr = _voice_audio()
        vp = graphed.voice_prompt('sonjav2', audio, sr, REF_TEXT)
        assert torch.equal(vp.codes, voice.codes)
        streams = [graphed.submit(t, prompt=vp, seed=20 + i) for i, t in enumerate(TEXTS)]
        together = [np.concatenate(list(s)) for s in streams]
        alone = graphed.render(TEXTS[1], prompt=vp, seed=21)
        ttfa = streams[0].metrics['ttfa_ms']
    finally:
        graphed.close()
    del vp
    assert torch.cuda.memory_allocated() < before + 64 * 2 ** 20  # close() frees weights, caches, graph pools
    for a, b in zip(eager, together):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(alone, eager[1])
    assert ttfa < 1000


def test_bucketed_mode_renders_sanely(engine, voice):
    """batch_invariant=False: smallest slot bucket per LM call. Same kind of audio, not the same bits."""
    from omnivoice_tts_server.audio8.engine import Audio8Engine

    reference = engine.render(TEXTS[1], prompt=voice, seed=5)
    eng = Audio8Engine(MODEL_DIR, 'cuda:0', cuda_graphs=True, batch_invariant=False, **ENGINE_KW)
    eng.load()
    try:
        assert eng.stats()['graphs'] > 7
        audio, sr = _voice_audio()
        vp = eng.voice_prompt('sonjav2', audio, sr, REF_TEXT)
        outs = [eng.render(TEXTS[1], prompt=vp, seed=5)]
        outs += [np.concatenate(list(s)) for s in [eng.submit(TEXTS[1], prompt=vp, seed=5 + i) for i in range(3)]]
    finally:
        eng.close()
    for out in outs:
        assert 0.6 < out.size / reference.size < 1.5
        assert np.isfinite(out).all() and 0.01 < np.abs(out).max() <= 1.0


@pytest.mark.skipif(not (PREVIEW_DIR / 'model.safetensors').is_file(), reason='needs models/Audio8-TTS-Preview-0.6b')
def test_preview_checkpoint_runs_on_the_same_engine():
    from omnivoice_tts_server.audio8.engine import Audio8Engine

    eng = Audio8Engine(PREVIEW_DIR, 'cuda:0', cuda_graphs=False, max_slots=2, decode_tile_rows=2, prefill_tile_rows=2)
    eng.load()
    try:
        audio, sr = _voice_audio()
        vp = eng.voice_prompt('sonjav2', audio, sr, REF_TEXT)
        stream = eng.submit(TEXTS[1], prompt=vp, seed=1)
        out = np.concatenate(list(stream))
    finally:
        eng.close()
    assert stream.metrics['steps'] > 30 and np.isfinite(out).all() and np.abs(out).max() > 0.01
