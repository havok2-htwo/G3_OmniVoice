"""CPU-only tests for the Ming engine's pure logic (no GPU, no model files)."""
from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from omnivoice_tts_server.ming.prompt import build_input_ids, build_tts_prompt, normalize_language_id
from omnivoice_tts_server.ming.streaming import ChunkSchedule, SilenceTrimmer, StreamResampler, trim_trailing_silence
from omnivoice_tts_server.ming.text import cut_text_by_semantic_length, get_semantic_length, prepare_fragments

# Expected values produced by upstream Ming-3.0-tts @ 9332fb9 (frontend.text_segment_cut + MingTTS.prepare_text with
# text_normalization=False).
UPSTREAM_FRAGMENTS = [
    ('de', 'Das ist ein sehr langer Text, der in mehrere Fragmente zerlegt werden muss, weil er mehr als vierundzwanzig '
           'Wörter hat. Und hier kommt noch ein zweiter Satz mit vielen weiteren Wörtern, damit wir sicher über die '
           'Grenze kommen.',
     ['Das ist ein sehr langer Text， der in mehrere Fragmente zerlegt werden muss， weil er mehr als vierundzwanzig '
      'Wörter hat。',
      'Und hier kommt noch ein zweiter Satz mit vielen weiteren Wörtern， damit wir sicher über die Grenze kommen。']),
    ('en', 'Hello world', ['Hello world.']),
    ('en', 'The U.S.A. has 3.14 reasons, and we list them all here today because this sentence is deliberately long '
           'enough to be split into two parts; see?',
     ['The U.S.A。', 'has 3.14 reasons，and we list them all here today because this sentence is deliberately long '
                    'enough to be split into two parts; see？']),
    ('de', 'Kurz.', ['Kurz.']),
    ('de', 'Erstens, zweitens, drittens. Viertens!', ['Erstens, zweitens, drittens. Viertens!']),
    ('en', 'Use `code` here,', ['Use code here.']),
]


@pytest.mark.parametrize(('language', 'text', 'expected'), UPSTREAM_FRAGMENTS)
def test_fragments_match_upstream(language, text, expected):
    assert prepare_fragments(text, language) == expected


def test_fragment_helpers():
    assert get_semantic_length("Don't stop, mañana 3 你好") == 5
    assert cut_text_by_semantic_length('...') == []
    with pytest.raises(ValueError):
        prepare_fragments('   ', 'de')


def test_language_and_prompt():
    assert normalize_language_id('de-DE') == 'de'
    assert normalize_language_id('German') == 'german'
    assert normalize_language_id('auto') is None
    assert normalize_language_id('xx') is None
    assert build_tts_prompt(None) == 'Please generate speech based on the following description.\n'
    assert build_tts_prompt('de').endswith('Target language: German.\n')


class _CharTokenizer:
    """One id per special token or character; enough to check the layout bookkeeping."""

    SPECIAL = ['<|im_start|>', '<|im_end|>', '<|vision_start|>', '<|vision_pad|>', '<|vision_end|>', '<audioPatch>',
               '<audio>']

    def encode(self, text, add_special_tokens=False):
        ids, i = [], 0
        while i < len(text):
            for j, tok in enumerate(self.SPECIAL):
                if text.startswith(tok, i):
                    ids.append(1000 + j)
                    i += len(tok)
                    break
            else:
                ids.append(ord(text[i]))
                i += 1
        return ids

    def convert_tokens_to_ids(self, token):
        return 1000 + self.SPECIAL.index(token)


def test_input_layout_positions():
    tok = _CharTokenizer()
    ids, spk, lat = build_input_ids(tok, 'P', 'Hallo', num_speakers=1, prompt_text='Ref', prompt_latent_steps=3)
    assert len(spk) == 1 and ids[spk[0]] == tok.convert_tokens_to_ids('<|vision_pad|>')
    assert ids[lat:] == [tok.convert_tokens_to_ids('<audioPatch>')] * 3
    assert ids[lat - 1] == tok.convert_tokens_to_ids('<audio>')
    ids2, spk2, lat2 = build_input_ids(tok, 'P', 'Hallo')
    assert spk2 == [] and lat2 == -1 and ids2[-1] == tok.convert_tokens_to_ids('<audio>')


def test_chunk_schedule_grows_to_max():
    s = ChunkSchedule(2, 8)
    sizes, decoded = [], 0
    for generated in range(1, 40):
        n = s.take(generated, decoded, final=False)
        assert decoded + n <= generated - 1  # one latent of lookahead stays undecoded
        if n:
            sizes.append(n)
            decoded += n
    assert sizes[:5] == [1, 2, 4, 8, 8]
    assert s.take(40, decoded, final=True) == 40 - decoded <= 9
    s.reset()
    assert s.target == 1
    assert ChunkSchedule(4, 2).maximum == 3 and ChunkSchedule(4, 2).initial == 3
    assert ChunkSchedule(1, 8).initial == 1  # first audio after 2 latents either way


def test_chunk_schedule_final_chunk_is_bounded():
    for first in range(1, 9):
        for maximum in range(1, 33):
            for length in range(1, 120):
                s = ChunkSchedule(first, maximum)
                decoded = 0
                for generated in range(1, length):
                    decoded += s.take(generated, decoded, final=False)
                assert 1 <= s.take(length, decoded, final=True) <= s.maximum + 1


def _signal(rng, sr, pattern):
    """pattern: list of (seconds, loud?)"""
    parts = []
    for seconds, loud in pattern:
        n = int(sr * seconds)
        parts.append(rng.normal(0, 0.2 if loud else 1e-4, n).astype(np.float32))
    return np.concatenate(parts)


def _stream_trim(audio, sr, chunks):
    trimmer = SilenceTrimmer(sr)
    out, i = [], 0
    for n in chunks[:-1]:
        out.append(trimmer.push(audio[i:i + n]))
        i += n
    out.append(trimmer.finish(audio[i:]))
    return np.concatenate(out)


@pytest.mark.parametrize('pattern', [
    [(0.5, True), (1.0, False)],
    [(0.2, False), (0.6, True), (0.45, False), (0.3, True), (0.73, False)],
    [(0.05, False)],
    [(1.3, False)],
    [(0.8, True)],
    [(0.3, True), (0.25, False), (0.31, True), (0.02, False)],
])
def test_silence_trimmer_matches_offline_trim(pattern):
    sr = 44100
    rng = np.random.default_rng(1)
    audio = _signal(rng, sr, pattern)
    expected = trim_trailing_silence(audio, sr)
    for chunk in (2205, 3528, 7056, 28224, 999):
        sizes = [chunk] * (len(audio) // chunk)
        rest = len(audio) - sum(sizes)
        # The engine's final call always carries >= one 0.1 s frame; mimic that.
        while sizes and rest < sr // 10:
            rest += sizes.pop()
        sizes.append(rest)
        got = _stream_trim(audio, sr, sizes)
        np.testing.assert_array_equal(got, expected)


def test_silence_trimmer_releases_speech_early():
    sr = 44100
    trimmer = SilenceTrimmer(sr)
    loud = np.full(4410, 0.5, dtype=np.float32)
    assert trimmer.push(loud).size == 4410
    quiet = np.zeros(sr, dtype=np.float32)
    released = trimmer.push(quiet)
    assert released.size == int(sr * 0.3)  # the guaranteed tail goes out at once
    assert trimmer.held == sr - int(sr * 0.3)
    assert trimmer.push(loud).size == trimmer.total - (4410 + int(sr * 0.3))  # speech resumes: held audio released
    assert trimmer.held == 0


def test_resampler_streaming_is_continuous():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.1, 44100).astype(np.float32)
    whole = StreamResampler(44100, 24000)
    ref = np.concatenate([whole.push(x), whole.flush()])
    r = StreamResampler(44100, 24000)
    parts = [r.push(x[i:i + 2205]) for i in range(0, len(x), 2205)] + [r.flush()]
    got = np.concatenate(parts)
    assert len(got) == len(ref) == 24000
    np.testing.assert_array_equal(got, ref)
    same = StreamResampler(44100, 44100)
    np.testing.assert_array_equal(same.push(x[:10]), x[:10])
    assert same.flush().size == 0


def test_bucket_for():
    from omnivoice_tts_server.ming.engine import bucket_for

    assert [bucket_for(n, 16) for n in (1, 2, 3, 5, 9, 16)] == [1, 2, 4, 8, 16, 16]
    assert bucket_for(11, 12) == 12
    assert bucket_for(40, 64) == 64


def test_stream_queue_semantics():
    from omnivoice_tts_server.ming.engine import MingStream

    cancelled = []
    engine = SimpleNamespace(output_sample_rate=24000, _cancel=lambda s: cancelled.append(s))
    s = MingStream('r1', engine)
    s._put(np.zeros(0, dtype=np.float32))  # empty chunks are dropped
    s._put(np.ones(240, dtype=np.float32))
    threading.Timer(0.01, lambda: (s._put(np.ones(480, dtype=np.float32)), s._finish())).start()
    chunks = list(s)
    assert [c.size for c in chunks] == [240, 480]
    assert s.metrics['samples'] == 720 and s.metrics['ttfa_ms'] >= 0 and s.done
    assert s.next_chunk() is None  # stays finished
    s.cancel()
    assert cancelled == [s]
    failing = MingStream('r2', engine)
    failing._finish(RuntimeError('boom'))
    with pytest.raises(RuntimeError, match='boom'):
        list(failing)


@pytest.mark.parametrize('rows', [1, 2, 7])
def test_split_bias_linear_never_writes_into_the_bias(rows):
    """Regression: for one row, b.expand(1, n) is already 'contiguous', so a contiguous() copy aliased the bias."""
    import torch

    from omnivoice_tts_server.ming import fast

    torch.manual_seed(0)
    x = torch.randn(rows, 8)
    w = torch.randn(5, 8)
    b = torch.randn(5)
    before = b.clone()
    out = fast._split(x, w, b)
    torch.testing.assert_close(out, torch.nn.functional.linear(x, w, b))
    assert torch.equal(b, before)
    fast.set_linear_mode('split')
    try:
        fast.linear(x.unsqueeze(0), w, b)
        fast.linear(x.unsqueeze(0), w, b)
    finally:
        fast.set_linear_mode('auto')
    assert torch.equal(b, before)
