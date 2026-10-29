"""GPU tests for the Ming engine (skipped without CUDA, the model files or onnxruntime).

Upstream-parity numbers (against the inclusionAI reference implementation) are produced by a separate scratch script
because the reference code is not part of this repository; these tests pin the engine's own invariants.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('onnxruntime')
pytest.importorskip('soxr')

REPO = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO / 'models' / 'Ming-3.0-tts-0.5b'
VOICE = REPO / 'data' / 'voices' / 'voice_preset_sonjav2.wav'
REF_TEXT = ('Franz jagt im komplett verwahrlosten Taxi quer durch Bayern. Vorbei an dichten Wäldern, kleinen Dörfern '
            'und alten Bahnhöfen.')

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not (MODEL_DIR / 'model.safetensors').is_file() or not VOICE.is_file(),
    reason='needs CUDA, models/Ming-3.0-tts-0.5b and data/voices/voice_preset_sonjav2.wav')


# vae_tile_rows=2 < rows in flight, so VAE jobs with several tiles are exercised too.
ENGINE_KW = dict(max_slots=4, vae_tile_rows=2)


@pytest.fixture(scope='module')
def engine():
    from omnivoice_tts_server.ming.engine import MingEngine

    eng = MingEngine(MODEL_DIR, 'cuda:0', cuda_graphs=False, **ENGINE_KW)
    eng.load()
    yield eng
    eng.close()


@pytest.fixture(scope='module')
def voice(engine):
    import soundfile as sf

    audio, sr = sf.read(str(VOICE), dtype='float32')
    return engine.voice_prompt('sonjav2', audio, sr, REF_TEXT, seed=0)


def test_voice_prompt_is_cached_and_seeded(engine, voice):
    import soundfile as sf

    audio, sr = sf.read(str(VOICE), dtype='float32')
    assert engine.voice_prompt('sonjav2', audio, sr, REF_TEXT) is voice
    again = engine.voice_prompt('sonjav2-copy', audio, sr, REF_TEXT, seed=0)
    assert torch.equal(again.latent, voice.latent)
    assert voice.num_latents == pytest.approx(voice.duration_s * 12.5, abs=2)
    assert voice.spk_emb.shape == (1, 192)
    engine.drop_voice('sonjav2-copy')


def test_streams_german_clone(engine, voice):
    stream = engine.submit('Hallo, das ist ein kurzer Test.', prompt=voice, language='de', seed=1)
    chunks = list(stream)
    audio = np.concatenate(chunks)
    assert engine.output_sample_rate == 24000
    assert len(chunks) > 2 and chunks[0].size < audio.size / 2  # true chunk streaming
    assert 0.8 < audio.size / 24000 < 6.0
    assert np.isfinite(audio).all() and 0.01 < np.abs(audio).max() <= 1.5
    assert stream.metrics['ttfa_ms'] < stream.metrics['total_ms']


def test_rows_are_independent_of_batch_composition(engine, voice):
    texts = ['Guten Morgen zusammen.', 'Heute ist ein schöner Tag.', 'Bis später!']
    alone = [engine.render(t, prompt=voice, language='de', seed=10 + i) for i, t in enumerate(texts)]
    streams = [engine.submit(t, prompt=voice, language='de', seed=10 + i) for i, t in enumerate(texts)]
    together = [np.concatenate(list(s)) for s in streams]
    for a, b in zip(alone, together):
        np.testing.assert_array_equal(a, b)
    # A row joining while another one is running does not change either of them.
    first = engine.submit(texts[1], prompt=voice, language='de', seed=11)
    got = [first.next_chunk(timeout=60)]
    late = engine.submit(texts[0], prompt=voice, language='de', seed=10)
    got += list(first)
    np.testing.assert_array_equal(np.concatenate(got), alone[1])
    np.testing.assert_array_equal(np.concatenate(list(late)), alone[0])


@pytest.fixture(scope='module')
def voice_v3(engine):
    import json

    import soundfile as sf

    presets_dir = REPO / 'backend' / 'src' / 'omnivoice_tts_server' / 'voice_presets'
    preset = next(p for p in json.loads((presets_dir / 'presets.json').read_text(encoding='utf-8'))
                  if p['voice_id'] == 'voice_preset_sonjav3')
    audio, sr = sf.read(str(presets_dir / preset['audio_file']), dtype='float32')
    return engine.voice_prompt('sonjav3', audio, sr, preset['ref_text'], seed=0)


def test_rows_with_different_voices_languages_and_seeds_are_independent(engine, voice, voice_v3):
    """A mixed continuous batch (two clone prompts, de + en, one seed per row) renders every row like it renders
    alone, bit for bit."""
    rows = [
        ('Guten Morgen zusammen.', voice, 'de', 31),
        ('The weather is lovely today.', voice_v3, 'en', 32),
        ('See you later!', voice, 'en', 33),
        ('Heute ist ein schöner Tag.', voice_v3, 'de', 34),
    ]
    assert len(rows) <= ENGINE_KW['max_slots']
    alone = [engine.render(t, prompt=vp, language=lang, seed=seed) for t, vp, lang, seed in rows]
    streams = [engine.submit(t, prompt=vp, language=lang, seed=seed) for t, vp, lang, seed in rows]
    together = [np.concatenate(list(s)) for s in streams]
    for a, b in zip(alone, together):
        np.testing.assert_array_equal(a, b)
    # Same text and seed, other voice: the prompt really reaches its own row.
    assert not np.array_equal(alone[0], engine.render(rows[0][0], prompt=voice_v3, language='de', seed=31))



def test_rows_with_different_instructions_and_reference_modes_are_independent(engine, voice):
    """Delivery controls are per row: captions (structured and free-form) and speaker-only cloning mixed in one
    batch render every row like it renders alone, bit for bit; and a caption really changes the audio."""
    from omnivoice_tts_server.ming.prompt import build_structured_instruction

    happy = build_structured_instruction(emotion='高兴')
    basic = build_structured_instruction(speed='快速', pitch='中', volume='中')
    rows = [
        ('Das ist ja großartig, wir haben es geschafft!', happy, False, 41),
        ('Das ist ja großartig, wir haben es geschafft!', None, True, 41),
        ('Wir sehen uns morgen früh am Bahnhof.', basic, False, 42),
        ('Wir sehen uns morgen früh am Bahnhof.', 'Speak slowly, in a calm and warm voice.', True, 43),
    ]
    assert len(rows) <= ENGINE_KW['max_slots']
    alone = [engine.render(t, prompt=voice, language='de', seed=seed, instruction=ins, acoustic_prompt=acoustic)
             for t, ins, acoustic, seed in rows]
    streams = [engine.submit(t, prompt=voice, language='de', seed=seed, instruction=ins, acoustic_prompt=acoustic)
               for t, ins, acoustic, seed in rows]
    together = [np.concatenate(list(s)) for s in streams]
    for a, b in zip(alone, together):
        np.testing.assert_array_equal(a, b)
    for audio in alone:
        assert np.isfinite(audio).all() and 0.01 < np.abs(audio).max() <= 1.5
    text = rows[0][0]
    speaker_only = engine.render(text, prompt=voice, language='de', seed=41, acoustic_prompt=False)
    # The caption changes the audio (speaker-only, same seed), and so does dropping the acoustic prompt.
    assert not np.array_equal(alone[0], speaker_only)
    assert not np.array_equal(alone[1], speaker_only)
    # A free-form caption changes the full clone too.
    assert not np.array_equal(alone[3], engine.render(rows[3][0], prompt=voice, language='de', seed=43))
    # Blank instruction means none.
    np.testing.assert_array_equal(alone[1], engine.render(text, prompt=voice, language='de', seed=41,
                                                          instruction='  '))

def test_long_text_is_fragmented_and_streamed(engine, voice):
    text = ('Das ist ein sehr langer Text, der in mehrere Fragmente zerlegt werden muss, weil er mehr als '
            'vierundzwanzig Wörter hat. Und hier kommt noch ein zweiter Satz, damit wir sicher über die Grenze kommen.')
    stream = engine.submit(text, prompt=voice, language='de', seed=3)
    audio = np.concatenate(list(stream))
    assert stream.metrics['fragments'] == 2
    assert audio.size / 24000 > 4.0


def test_no_voice_prompt_and_cancel(engine):
    audio = engine.render('Hello there.', prompt=None, language='en', seed=0)
    assert audio.size > 0
    stream = engine.submit('This is a much longer sentence that we are going to cancel right after the first chunk.',
                           prompt=None, language='en', seed=0)
    assert stream.next_chunk(timeout=60) is not None
    stream.cancel()
    rest = list(stream)  # ends normally
    assert len(rest) < 20
    deadline = time.time() + 10
    while engine.stats()['active_rows'] and time.time() < deadline:
        time.sleep(0.02)
    assert engine.stats()['active_rows'] == 0
    assert engine.stats()['free_slots'] == engine.max_slots


def test_streamed_decode_equals_full_decode_in_fp32(engine):
    """The chunked VAE decode is exact math (bf16 runs differ only by rounding)."""
    from omnivoice_tts_server.ming.audio_vae import DecodeStreamState

    dec = copy.deepcopy(engine.model.audio.decoder).float()
    gen = torch.Generator(device='cpu').manual_seed(0)
    latents = torch.randn(1, 37, 64, generator=gen).cuda()
    with torch.inference_mode():
        full = dec.decode_full(latents)[0, 0]
        state, parts, i = DecodeStreamState(), [], 0
        for n in (1, 1, 2, 4, 8, 8, 8, 8):
            if i >= latents.shape[1]:
                break
            last = i + n >= latents.shape[1]
            parts.append(dec.stream_decode(latents[:, i:i + n], state, last_chunk=last)[0, 0])
            i += n
        streamed = torch.cat(parts)
    del dec
    torch.cuda.empty_cache()
    assert streamed.shape == full.shape == (37 * 4 * 882,)
    rel = ((streamed - full).pow(2).mean().sqrt() / full.pow(2).mean().sqrt()).item()
    assert rel < 1e-4


def test_cuda_graphs_match_eager_bit_exactly(engine, voice):
    """Graph replay runs the very same kernels: graphs on == graphs off, alone and batched (multi-tile VAE jobs)."""
    import soundfile as sf

    from omnivoice_tts_server.ming.engine import MingEngine

    texts = ['Guten Morgen zusammen.', 'Heute ist ein schöner Tag, findest du nicht?', 'Bis später!', 'Na klar.']
    eager = [engine.render(t, prompt=voice, language='de', seed=20 + i) for i, t in enumerate(texts)]
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated()
    graphed = MingEngine(MODEL_DIR, 'cuda:0', cuda_graphs=True, **ENGINE_KW)
    graphed.load()
    try:
        assert graphed.stats()['cuda_graphs'] and graphed.stats()['graphs'] > 0
        audio, sr = sf.read(str(VOICE), dtype='float32')
        vp = graphed.voice_prompt('sonjav2', audio, sr, REF_TEXT, seed=0)
        assert torch.equal(vp.latent, voice.latent)
        streams = [graphed.submit(t, prompt=vp, language='de', seed=20 + i) for i, t in enumerate(texts)]
        together = [np.concatenate(list(s)) for s in streams]
        alone = graphed.render(texts[1], prompt=vp, language='de', seed=21)
        ttfa = streams[0].metrics['ttfa_ms']
    finally:
        graphed.close()
    del vp
    assert torch.cuda.memory_allocated() < mem_before + 64 * 2 ** 20  # close() frees weights, caches, graph pools
    for a, b in zip(eager, together):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(alone, eager[1])
    assert ttfa < 1000


def _slot_stream(sd, latents, schedule, dtype):
    """Drive a SlotDecoder like the engine does (lookahead, growing chunks, final remainder) for one row."""
    total = latents.shape[1]
    parts, decoded, generated = [], 0, 0
    from omnivoice_tts_server.ming.streaming import ChunkSchedule

    sched = ChunkSchedule(*schedule)
    while decoded < total:
        generated = min(total, generated + 1)
        final = generated == total
        n = sched.take(generated, decoded, final)
        if not n:
            continue
        chunk = n
        window = [min(max(t, 0), generated - 1) for t in range(decoded - 1, decoded + chunk + 1)]
        lat = latents[:, window].to(dtype)
        meta = torch.tensor([[0, sd.scale * decoded, sd.scale * n, int(decoded == 0)]], device=lat.device)
        out = torch.zeros(1, sd.out_len(chunk), device=lat.device)
        sd.run(lat, meta, out)
        audio = out[0, :sd.scale * n * sd.hop]
        parts.append(audio[sd.pad:] if decoded == 0 else audio)
        if final:
            parts.append(out[0, sd.scale * chunk * sd.hop:])
        decoded += n
    return torch.cat(parts)


def test_slot_decoder_is_exact_in_fp32(engine):
    """The batched slot decoder (ring KV, closed-form upsampling, OLA carry) equals the one-pass decode."""
    from omnivoice_tts_server.ming.vae_stream import SlotDecoder

    dec = copy.deepcopy(engine.model.audio.decoder).float()
    gen = torch.Generator(device='cpu').manual_seed(1)
    with torch.inference_mode():
        sd = SlotDecoder(dec, slots=1, max_chunk_latents=8, max_fragment_latents=512, device=torch.device('cuda'),
                         dtype=torch.float32)
        for length in (1, 5, 23, 70):
            latents = torch.randn(1, length, 64, generator=gen).cuda()
            full = dec.decode_full(latents)[0, 0]
            got = _slot_stream(sd, latents, (2, 8), torch.float32)
            assert got.shape == full.shape == (length * 4 * 882,)
            rel = ((got - full).pow(2).mean().sqrt() / full.pow(2).mean().sqrt()).item()
            assert rel < 1e-4, (length, rel)
    del dec, sd
    torch.cuda.empty_cache()


def test_fast_flow_matches_reference_sampler(engine):
    """FastFlow (fused/hoisted) vs the vendored sampler: same distance to the fp32 result, i.e. bf16 noise only."""
    from omnivoice_tts_server.ming.engine import CFG_SCALE, FLOW_STEPS, SIGMA, TEMPERATURE
    from omnivoice_tts_server.ming.flow import sample_flow

    m = engine.model
    gen = torch.Generator(device='cpu').manual_seed(2)
    B = 8
    z = (torch.randn(B, 1, 896, generator=gen) * 2).cuda().bfloat16()
    hist = torch.randn(B, 1, 64, generator=gen).cuda().bfloat16()
    noise = torch.randn(B, 1, 64, generator=gen).cuda().bfloat16()
    kw = dict(cfg_scale=CFG_SCALE, steps=FLOW_STEPS, patch_size=1, sigma=SIGMA, temperature=TEMPERATURE)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        ref = sample_flow(m.dit, noise, z, hist, timesteps=engine.timesteps, **kw).float()
        engine.ff.prepare(B)
        fast = engine.ff.sample(noise, z, hist).float()
        ref_agg = m.linear_proj_audio(ref.bfloat16()).float()
        fast_agg = engine.ff.aggregate(ref.bfloat16()).float()
        dit32 = copy.deepcopy(m.dit).float()
        with torch.autocast('cuda', enabled=False):
            exact = sample_flow(dit32, noise.float(), z.float(), hist.float(), timesteps=engine.timesteps.float(), **kw)
    del dit32
    torch.cuda.empty_cache()
    ref_err = (ref - exact).abs().mean().item()
    fast_err = (fast - exact).abs().mean().item()
    assert fast_err < 1.5 * ref_err + 1e-3, (fast_err, ref_err)
    assert (fast - ref).abs().mean().item() < 0.05
    assert (fast_agg - ref_agg).abs().mean().item() < 0.02 * ref_agg.abs().mean().item() + 1e-3


def test_fast_llm_matches_reference_forward(engine):
    """Our prefill + slot-cache decode (fused projections) vs transformers' Qwen2Model forward: both are equally far
    from the fp32 result (bf16 noise), whatever kernels they use."""
    llm = engine.model.llm
    gen = torch.Generator(device='cpu').manual_seed(3)
    L = 40
    ids = torch.randint(0, 150000, (1, L + 1), generator=gen).cuda()
    llm32 = copy.deepcopy(llm).float()

    def work():
        x = llm.embed_tokens(ids)
        with torch.autocast('cuda', enabled=False):
            exact = llm32(inputs_embeds=x.float(), use_cache=False).last_hidden_state.float()
        ref = llm(inputs_embeds=x, use_cache=False).last_hidden_state.float()
        # Slot 3 of the engine buffers (no rows are active between tests; this runs on the engine's GPU thread).
        engine.prefill_h[0, :L] = x[0, :L]
        engine.prefill_slot.fill_(3)
        engine.prefill_last.fill_(L - 1)
        engine._prefill_core(64)
        z_prefill = engine.z_buf[3, 0].float().clone()
        engine.next_embed[3] = x[0, L:L + 1]
        engine.pos_gpu.zero_()
        engine.pos_gpu[3] = L
        engine._decode_core(4)
        return exact, ref, z_prefill, engine.z_buf[3, 0].float().clone()

    exact, ref, z_prefill, z_decode = engine._run_on_gpu_thread(work)
    del llm32
    torch.cuda.empty_cache()

    def err(a, b):
        return (a - b).abs().mean().item()

    for ours, pos in ((z_prefill, L - 1), (z_decode, L)):
        assert err(ours, exact[0, pos]) < 1.3 * err(ref[0, pos], exact[0, pos]), pos
        assert err(ours, ref[0, pos]) < 0.05 * exact.abs().mean().item(), pos


def test_bucketed_mode_renders_sanely(engine, voice):
    """batch_invariant=False (smallest bucket per call, incl. one-row GEMMs) renders the same kind of audio."""
    from omnivoice_tts_server.ming.engine import MingEngine

    text = 'Heute ist ein schöner Tag, findest du nicht?'
    reference = engine.render(text, prompt=voice, language='de', seed=5)
    eng = MingEngine(MODEL_DIR, 'cuda:0', cuda_graphs=True, batch_invariant=False, max_slots=4)
    eng.load()
    try:
        assert eng.stats()['graphs'] > 11  # one graph per bucket and shape
        vp = eng.voice_prompt('sonjav2', voice_audio(), 24000, REF_TEXT, seed=0)
        alone = eng.render(text, prompt=vp, language='de', seed=5)
        streams = [eng.submit(text, prompt=vp, language='de', seed=5 + i) for i in range(3)]
        outs = [np.concatenate(list(s)) for s in streams]
    finally:
        eng.close()
    for audio in [alone] + outs:
        assert 0.7 < audio.size / reference.size < 1.4
        assert np.isfinite(audio).all() and 0.01 < np.abs(audio).max() <= 1.5


def voice_audio():
    import soundfile as sf

    audio, sr = sf.read(str(VOICE), dtype='float32')
    assert sr == 24000
    return audio
