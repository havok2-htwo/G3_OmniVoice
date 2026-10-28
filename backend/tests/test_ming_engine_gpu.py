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


@pytest.fixture(scope='module')
def engine():
    from omnivoice_tts_server.ming.engine import MingEngine

    eng = MingEngine(MODEL_DIR, 'cuda:0', max_slots=4, cuda_graphs=False)
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
