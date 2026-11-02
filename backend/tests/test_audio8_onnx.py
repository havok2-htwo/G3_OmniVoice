"""Audio8 CPU engine (audio8/onnx_engine.py, onnx_codec.py).

The first part runs without the model, on fake graphs that follow the real contract. The second part runs the real
INT8 graphs on the CPU and is skipped when the model folder is missing.
"""
from __future__ import annotations

import asyncio
import io
import json
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from omnivoice_tts_server.audio8 import onnx_codec, onnx_engine
from omnivoice_tts_server.audio8.onnx_engine import (Audio8OnnxEngine, OnnxLM, Sampling, SlowState, default_threads,
                                                     sample_token)
from omnivoice_tts_server.audio8.text import suffix_ids, user_turn_ids
from omnivoice_tts_server.audio8.variants import AUDIO8_VARIANTS
from omnivoice_tts_server.audio8.voice_cache import prune_voices, remove_voice, voice_tag
from omnivoice_tts_server.audio8_runtime import Audio8Synthesizer
from omnivoice_tts_server.capacity import audio8_onnx_capacity_profile
from omnivoice_tts_server.config import Settings
from omnivoice_tts_server.domain.state import InMemoryStore, VoiceProfileRecord, utcnow
from omnivoice_tts_server.runtime_v2 import BatchSynthesisItem

BOOKS = 3
SIZE = 16
BEGIN = 100
BOOM = 9999
HOP = 2048
MANIFEST = {
    'num_codebooks': BOOKS, 'codebook_size': SIZE, 'semantic_begin_id': BEGIN, 'max_seq_len': 400, 'num_fast_layers': 4,
    'slow_logits_layout': 'relative_semantic_then_eos', 'sample_rate': 44100, 'codec_hop_length': HOP,
    'model_fingerprint': 'test', 'default_precision': 'int8', 'default_codec_precision': 'fp16', 'codec_models': {'fp16': 'x'},
}


def _names(*names, shape=None):
    return [SimpleNamespace(name=name, shape=shape) for name in names]


class FakeSlow:
    """The slow graph's contract. conv counts all columns, ssm the audio columns: the state must travel through."""

    def __init__(self, eos_after: int | None = 12, delay: float = 0.0) -> None:
        self.eos_after = eos_after
        self.delay = delay
        self.positions: list[int] = []

    def get_inputs(self):
        return [SimpleNamespace(name='codes', shape=[1, BOOKS + 1, 1]), SimpleNamespace(name='position', shape=[1]),
                SimpleNamespace(name='cache_keys', shape=[1, 1, 2, 400, 4]),
                SimpleNamespace(name='cache_values', shape=[1, 1, 2, 400, 4]),
                SimpleNamespace(name='conv_states', shape=[1, 1, 2, 1]),
                SimpleNamespace(name='ssm_states', shape=[1, 1, 2, 1, 1])]

    def get_outputs(self):
        return _names('logits', 'hidden', 'key_delta', 'value_delta', 'next_conv_states', 'next_ssm_states')

    def run(self, _, feeds):
        if self.delay:
            time.sleep(self.delay)
        column = feeds['codes']
        if int(column[0, 0, 0]) == BOOM:
            raise RuntimeError('boom')
        self.positions.append(int(feeds['position'][0]))
        conv = feeds['conv_states'] + 1
        ssm = feeds['ssm_states'] + (1 if int(column[0, 0, 0]) >= BEGIN else 0)
        rng = np.random.default_rng(int(conv.flat[0]))
        logits = rng.normal(size=(1, 1, SIZE + 1)).astype(np.float32)
        logits[0, 0, SIZE] = -100.0
        if self.eos_after is not None and int(ssm.flat[0]) >= self.eos_after:
            logits[0, 0, SIZE] = 100.0
        hidden = np.full((1, 1, 8), float(conv.flat[0]), dtype=np.float32)
        delta = np.full((1, 1, 2, 4), float(feeds['position'][0]), dtype=np.float32)
        return [logits, hidden, delta, delta, conv, ssm]


class FakeFast:
    def get_inputs(self):
        names = ['slow_hidden', 'token_id', 'use_slow_hidden', 'input_pos']
        names += [f'cache_{kind}_{index}' for kind in ('key', 'value') for index in range(4)]
        return [SimpleNamespace(name=name, shape=[1, 2, BOOKS, 4]) for name in names]

    def get_outputs(self):
        return _names('logits', *[f'{kind}_delta_{index}' for index in range(4) for kind in ('key', 'value')])

    def run(self, _, feeds):
        rng = np.random.default_rng(int(feeds['token_id'][0, 0]) + 1000 * int(feeds['input_pos'][0]))
        delta = np.ones((1, 2, 1, 4), dtype=np.float32)
        return [rng.normal(size=(1, 1, SIZE)).astype(np.float32)] + [delta] * 8


class FakeDecoder:
    """Loud audio derived from the frames, like the real one a function of the frames only."""

    def __init__(self, path=None, threads=1, hop=HOP) -> None:
        self.hop = hop
        self.calls: list[tuple[int, int]] = []
        self.fail = False

    def decode(self, frames, start, end):
        if self.fail:
            self.fail = False
            raise RuntimeError('codec failed')
        self.calls.append((start, end))
        return np.repeat(0.2 + frames[start:end, 0].astype(np.float32) / 100.0, self.hop)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [BOOM if text == 'BOOM' else 1 + ord(char) % 50 for char in text]


class FakeEngine(Audio8OnnxEngine):
    def __init__(self, slow=None, **kwargs) -> None:
        kwargs.setdefault('workers', 1)
        kwargs.setdefault('threads', 1)
        super().__init__('unused', decoder_factory=lambda *a: self.decoders.append(FakeDecoder(*a)) or self.decoders[-1],
                         **kwargs)
        self.decoders: list[FakeDecoder] = []
        self.slow = slow or FakeSlow()

    def _read_manifest(self):
        return dict(MANIFEST)

    def _open_tokenizer(self):
        return FakeTokenizer()

    def _open_lm(self, manifest):
        return OnnxLM(self.slow, FakeFast(), manifest)


@pytest.fixture
def make():
    engines = []

    def build(**kwargs):
        engine = FakeEngine(**kwargs)
        engine.load()
        engines.append(engine)
        return engine

    yield build
    for engine in engines:
        engine.close()


def _voice(engine, key='v', voice_id='voice_v', ref_text='Referenz.'):
    return engine.voice_prompt(key, np.zeros(44100, dtype=np.float32), 44100, ref_text, voice_id=voice_id)


def _codes(frames=20):
    return np.arange(BOOKS * frames, dtype=np.int64).reshape(BOOKS, frames) % SIZE


@pytest.fixture
def encoded(monkeypatch):
    calls = []

    def encode(self, wave):
        calls.append(wave.size)
        return _codes()

    monkeypatch.setattr(Audio8OnnxEngine, '_encode_reference', encode)
    return calls


# -- sampling -----------------------------------------------------------------------------------------------------------


def test_sample_token_keeps_only_top_k() -> None:
    logits = np.arange(100, dtype=np.float32)
    rng = np.random.default_rng(0)
    drawn = {sample_token(logits, 5.0, 1.0, 3, rng) for _ in range(300)}
    assert drawn == {97, 98, 99}


def test_sample_token_top_p_uses_the_unscaled_distribution_and_keeps_the_best() -> None:
    # Probabilities 0.5 / 0.3 / 0.2: cumulative 0.5, 0.8, 1.0. Upstream drops what exceeds top_p.
    logits = np.log(np.array([0.3, 0.5, 0.2]))
    rng = np.random.default_rng(0)
    assert {sample_token(logits, 100.0, 0.85, 3, rng) for _ in range(300)} == {0, 1}
    assert {sample_token(logits, 100.0, 0.1, 3, rng) for _ in range(50)} == {1}


def test_sample_token_counts_tokens_outside_top_k_in_top_p() -> None:
    # The two best hold 0.2 of the mass: a top_p that only sees the top k would cut the second one away.
    logits = np.log(np.array([0.1, 0.1] + [0.8 / 80] * 80))
    rng = np.random.default_rng(0)
    assert {sample_token(logits, 1.0, 0.25, 2, rng) for _ in range(200)} == {0, 1}


def test_sample_token_follows_the_probabilities() -> None:
    rng = np.random.default_rng(1)
    draws = [sample_token(np.array([0.0, 1.0]), 1.0, 1.0, 2, rng) for _ in range(4000)]
    assert abs(np.mean(draws) - np.e / (1 + np.e)) < 0.02


def test_sample_token_refuses_broken_logits() -> None:
    with pytest.raises(ValueError):
        sample_token(np.array([np.nan, np.inf * 0]), 1.0, 1.0, 1, np.random.default_rng(0))


def _lm(eos_after=12):
    return OnnxLM(FakeSlow(eos_after), FakeFast(), dict(MANIFEST))


def test_the_first_frame_cannot_be_eos_later_ones_can() -> None:
    logits = np.zeros(SIZE + 1, dtype=np.float32)
    logits[SIZE] = 100.0
    logits[3] = 1.0
    lm = _lm()
    sampling = Sampling(1.0, 1.0, 1)
    assert lm.sample_semantic(logits, onnx_engine.collections.deque(), sampling, np.random.default_rng(0)) == 3
    assert lm.sample_semantic(logits, onnx_engine.collections.deque([3]), sampling, np.random.default_rng(0)) is None


def test_a_repeated_token_is_drawn_again_with_the_ras_settings(monkeypatch) -> None:
    draws = iter([5, 9])
    seen = []

    def fake(values, temperature, top_p, top_k, rng):
        seen.append((temperature, top_p))
        return next(draws)

    monkeypatch.setattr(onnx_engine, 'sample_token', fake)
    previous = onnx_engine.collections.deque([5], maxlen=10)
    code = _lm().sample_semantic(np.zeros(SIZE + 1), previous, Sampling(0.8, 0.95, 50), np.random.default_rng(0))
    assert code == 9 and seen == [(0.8, 0.95), (1.0, 0.9)]


def test_a_new_token_is_kept(monkeypatch) -> None:
    draws = iter([5, 9])
    monkeypatch.setattr(onnx_engine, 'sample_token', lambda *a: next(draws))
    previous = onnx_engine.collections.deque([6], maxlen=10)
    assert _lm().sample_semantic(np.zeros(SIZE + 1), previous, Sampling(), np.random.default_rng(0)) == 5


def test_lm_refuses_other_graphs() -> None:
    class Other(FakeSlow):
        def get_inputs(self):
            return super().get_inputs()[:3]

    with pytest.raises(ValueError, match='conv_states'):
        OnnxLM(Other(), FakeFast(), dict(MANIFEST))
    with pytest.raises(ValueError, match='layout'):
        OnnxLM(FakeSlow(), FakeFast(), {**MANIFEST, 'slow_logits_layout': 'other'})


def test_lm_state_is_written_at_its_position_and_a_cached_head_restores_it() -> None:
    lm = _lm()
    state = lm.fresh_state()
    for position in range(3):
        lm.step(lm.text_column(5), position, state)
    assert state.keys[0, 0, 0, 2, 0] == 2 and state.conv.flat[0] == 3
    head = SlowState(state.keys[:, :, :, :3].copy(), state.values[:, :, :, :3].copy(), state.conv, state.ssm)
    fresh = lm.fresh_state(head)
    assert fresh.keys[0, 0, 0, 2, 0] == 2 and fresh.keys[0, 0, 0, 3, 0] == 0 and fresh.conv.flat[0] == 3
    with pytest.raises(ValueError, match='exceeds'):
        lm.step(lm.text_column(5), 400, state)


# -- rows -----------------------------------------------------------------------------------------------------------------


def test_a_row_streams_chunks_on_the_schedule(make) -> None:
    engine = make(first_chunk_frames=4, chunk_frames=4, slow=FakeSlow(eos_after=13))
    chunks = list(engine.submit('Hallo.', prompt=None, seed=1))
    decoder = engine.decoders[0]
    warm = decoder.calls[0]
    assert decoder.calls[1:] == [(0, 4), (4, 8), (8, 12), (12, 13)]
    assert warm[0] == 0 and len(chunks) == 4


def test_seed_decides_and_rows_are_the_same_on_any_worker(make) -> None:
    single = make(workers=1)
    many = make(workers=3)
    alone = single.render('Satz eins.', seed=4)
    assert np.array_equal(alone, single.render('Satz eins.', seed=4))
    assert not np.array_equal(alone, single.render('Satz eins.', seed=5))
    streams = [many.submit(text, prompt=None, seed=seed) for text, seed in
               (('Satz zwei.', 1), ('Satz eins.', 4), ('Satz drei.', 2))]
    assert np.array_equal(np.concatenate(list(streams[1])), alone)


def test_negative_seeds_work(make) -> None:
    assert make().render('Satz.', seed=-1).size


def test_a_row_without_eos_ends_at_the_frame_cap(make) -> None:
    engine = make(slow=FakeSlow(eos_after=None))
    stream = engine.submit('Kurz.', prompt=None, seed=0)
    list(stream)
    fragment_cap = onnx_engine.frame_cap('Kurz.')
    assert stream.metrics['frames'] == fragment_cap and stream.metrics['frame_cap_hits'] == 1


def test_long_text_renders_in_fragments_each_with_its_own_generator(make) -> None:
    engine = make(fragment_max_chars=40, slow=FakeSlow(eos_after=6))
    text = 'Das ist der erste Satz des Textes. Und hier folgt der zweite Satz. Ein dritter Satz kommt auch noch.'
    voice = engine._plain_voice
    fragments = engine._build_fragments(text, voice)
    assert len(fragments) >= 2 and ' '.join(fragment.text for fragment in fragments) == text
    stream = engine.submit(text, prompt=None, seed=2)
    assert list(stream)
    assert stream.metrics['fragments'] == len(fragments) and stream.metrics['frames'] == 6 * len(fragments)
    assert stream.metrics.get('frame_cap_hits') is None


def test_only_the_text_is_prefilled_per_fragment(make) -> None:
    slow = FakeSlow(eos_after=2)
    engine = make(slow=slow)
    voice = engine._plain_voice
    slow.positions.clear()
    engine.render('Abc', prompt=None, seed=0)
    assert slow.positions[0] == voice.length
    assert len(slow.positions) == len(engine._build_fragments('Abc', voice)[0].ids) + 2


def test_the_prompt_has_the_layout_of_the_full_prompt(make) -> None:
    engine = make()
    codes = _codes(5)
    tokenizer = engine.tokenizer
    ids = [1, 2, 3]
    tail = user_turn_ids(tokenizer)
    head = engine._head_columns(ids, codes, tail)
    assert head[0].tolist() == ids + (codes[0] + BEGIN).tolist() + tail
    assert np.array_equal(head[1:, 3:8], codes) and not head[1:, :3].any() and not head[1:, 8:].any()
    assert suffix_ids(tokenizer, 'x')[:len(tail)] == tail


def test_a_prompt_too_long_for_the_context_is_refused(make, encoded, monkeypatch) -> None:
    engine = make()
    monkeypatch.setattr(Audio8OnnxEngine, '_encode_reference', lambda self, wave: _codes(390))
    with pytest.raises(ValueError, match='too long'):
        _voice(engine)


def test_cancel_stops_a_running_row(make) -> None:
    engine = make(slow=FakeSlow(eos_after=None, delay=0.002), first_chunk_frames=1, chunk_frames=1)
    stream = engine.submit('Ein langer Satz, der nicht von selbst endet.', prompt=None, seed=0)
    assert stream.next_chunk(timeout=10) is not None
    stream.cancel()
    assert list(stream) is not None and stream.done
    assert engine.stats()['active_rows'] == 0
    assert engine.render('Weiter.', seed=0).size


def test_cancel_of_a_waiting_row_ends_it_at_once(make) -> None:
    engine = make(slow=FakeSlow(eos_after=None, delay=0.002))
    running = engine.submit('Lang und ohne Ende, wirklich.', prompt=None, seed=0)
    assert running.next_chunk(timeout=30) is not None
    waiting = engine.submit('Wartet.', prompt=None, seed=0)
    waiting.cancel()
    assert waiting.done and engine.stats()['pending_rows'] == 0
    running.cancel()
    list(running)


def test_a_failing_row_fails_alone(make) -> None:
    engine = make(slow=FakeSlow(eos_after=4), workers=1)
    engine.tokenizer.encode = lambda text, add_special_tokens=False: [BOOM] if 'BOOM' in text else [1, 2]
    bad = engine.submit('BOOM', prompt=None, seed=0)
    good = engine.submit('Gut.', prompt=None, seed=0)
    with pytest.raises(RuntimeError, match='boom'):
        list(bad)
    assert list(good)
    assert engine.stats()['active_rows'] == 0


def test_a_decoder_failure_fails_the_row_and_the_worker_goes_on(make) -> None:
    engine = make(slow=FakeSlow(eos_after=4))
    engine.decoders[0].fail = True
    with pytest.raises(RuntimeError, match='codec failed'):
        list(engine.submit('Eins.', prompt=None, seed=0))
    assert engine.render('Zwei.', seed=0).size


def test_close_ends_every_row_with_an_error() -> None:
    engine = FakeEngine(slow=FakeSlow(eos_after=None, delay=0.002))
    engine.load()
    running = engine.submit('Lang und ohne Ende, wirklich.', prompt=None, seed=0)
    waiting = engine.submit('Wartet.', prompt=None, seed=0)
    engine.close()
    for stream in (running, waiting):
        with pytest.raises(RuntimeError, match='closed'):
            list(stream)
    assert not engine.loaded
    with pytest.raises(RuntimeError, match='not loaded'):
        engine.submit('x', prompt=None, seed=0)


def test_output_is_resampled_when_asked() -> None:
    pytest.importorskip('soxr')
    engine = FakeEngine(output_sample_rate=24000, slow=FakeSlow(eos_after=30))
    engine.load()
    try:
        audio = engine.render('Hallo.', seed=0)
    finally:
        engine.close()
    assert abs(audio.size - int(30 * HOP / 4410) * 4410 * 24000 / 44100) < 100


# -- voices ---------------------------------------------------------------------------------------------------------------


def test_a_voice_is_encoded_once_and_cached_on_disk(make, encoded, tmp_path) -> None:
    first = make(voice_cache_dir=tmp_path)
    prompt = _voice(first)
    assert _voice(first) is prompt and len(encoded) == 1
    assert prompt.codes.shape == (BOOKS, 20) and first.has_voice('v')
    files = list(tmp_path.iterdir())
    assert len(files) == 1 and files[0].suffix == '.npy'
    second = make(voice_cache_dir=tmp_path)
    again = _voice(second)
    assert len(encoded) == 1 and np.array_equal(again.codes, prompt.codes)
    assert np.array_equal(again.state.keys, prompt.state.keys)


def test_a_broken_cache_file_is_encoded_again(make, encoded, tmp_path) -> None:
    engine = make(voice_cache_dir=tmp_path)
    _voice(engine)
    next(tmp_path.iterdir()).write_bytes(b'not numpy')
    engine._voices.clear()
    _voice(engine)
    assert len(encoded) == 2
    other = make(voice_cache_dir=tmp_path)
    _voice(other)
    assert len(encoded) == 2


def test_the_cache_is_per_model(make, encoded, tmp_path) -> None:
    engine = make(voice_cache_dir=tmp_path)
    _voice(engine)
    engine._voices.clear()
    engine.manifest['model_fingerprint'] = 'another'
    _voice(engine)
    assert len(encoded) == 2 and len(list(tmp_path.iterdir())) == 2


def test_drop_voice_forgets_the_voice_in_memory(make, encoded, tmp_path) -> None:
    engine = make(voice_cache_dir=tmp_path)
    _voice(engine)
    engine.drop_voice('v')
    assert not engine.has_voice('v')
    _voice(engine)
    assert len(encoded) == 1, 'a voice change must not wipe the codes of the voices that stay'
    engine.drop_voice('never seen')


def test_a_reference_without_a_saved_voice_is_never_written(make, encoded, tmp_path) -> None:
    engine = make(voice_cache_dir=tmp_path)
    _voice(engine, voice_id=None)
    assert engine.has_voice('v') and not list(tmp_path.iterdir())


def test_a_saved_voice_has_one_file_whatever_transcript_a_request_sends(make, encoded, tmp_path) -> None:
    engine = make(voice_cache_dir=tmp_path)
    for index in range(4):
        _voice(engine, key=f'v{index}', ref_text=f'Transkript {index}.')
    assert len(encoded) == 1 and len(list(tmp_path.iterdir())) == 1


def test_deleting_a_voice_finds_its_codes_with_no_engine_loaded(make, encoded, tmp_path) -> None:
    engine = make(voice_cache_dir=tmp_path)
    _voice(engine, 'a', 'voice_a')
    _voice(engine, 'b', 'voice_b')
    engine.close()
    assert len(list(tmp_path.iterdir())) == 2
    remove_voice(tmp_path, 'voice_a')
    assert [path.name.split('.')[0] for path in tmp_path.iterdir()] == [voice_tag('voice_b')]
    remove_voice(tmp_path, 'voice_a')
    remove_voice(tmp_path / 'not there', 'voice_a')


def test_pruning_keeps_only_files_of_current_voices(make, encoded, tmp_path) -> None:
    engine = make(voice_cache_dir=tmp_path)
    _voice(engine, 'a', 'voice_a')
    _voice(engine, 'b', 'voice_b')
    (tmp_path / ('f' * 64 + '.npy')).write_bytes(b'file of the old layout')
    (tmp_path / f'{voice_tag("voice_gone")}.abc.123.456.tmp').write_bytes(b'left by a crash')
    prune_voices(tmp_path, ['voice_a'])
    assert [path.name.split('.')[0] for path in tmp_path.iterdir()] == [voice_tag('voice_a')]
    prune_voices(tmp_path / 'not there', [])


def test_voices_in_memory_are_bounded(make, encoded, monkeypatch) -> None:
    monkeypatch.setattr(onnx_engine, 'MAX_CACHED_VOICES', 2)
    engine = make()
    for key in 'abc':
        _voice(engine, key)
    assert not engine.has_voice('a') and engine.has_voice('b') and engine.has_voice('c')


def test_concurrent_requests_encode_a_new_voice_once(make, encoded) -> None:
    engine = make()
    results = []
    threads = [threading.Thread(target=lambda: results.append(_voice(engine))) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(encoded) == 1 and all(result is results[0] for result in results)


@pytest.mark.parametrize('seconds', [0.1, 31.0])
def test_reference_length_is_checked(make, encoded, seconds) -> None:
    with pytest.raises(ValueError, match='seconds'):
        make().voice_prompt('v', np.zeros(int(44100 * seconds), dtype=np.float32), 44100, 'Text.')


def test_reference_needs_a_transcript_and_clean_samples(make, encoded) -> None:
    engine = make()
    with pytest.raises(ValueError, match='ref_text'):
        engine.voice_prompt('v', np.zeros(44100, dtype=np.float32), 44100, '  ')
    with pytest.raises(ValueError, match='non-finite'):
        engine.voice_prompt('v', np.full(44100, np.nan, dtype=np.float32), 44100, 'Text.')


def test_reference_is_resampled_to_the_native_rate(make, encoded) -> None:
    pytest.importorskip('soxr')
    make().voice_prompt('v', np.zeros(24000, dtype=np.float32), 24000, 'Text.')
    assert abs(encoded[0] - 44100) < 16


def test_invalid_reference_codes_are_refused(make) -> None:
    engine = make()
    with pytest.raises(ValueError, match='outside'):
        engine._check_codes(_codes() + SIZE)
    with pytest.raises(ValueError, match='shape'):
        engine._check_codes(_codes()[:2])


# -- the rest -------------------------------------------------------------------------------------------------------------


def test_the_thread_default_leaves_room_for_the_lm(monkeypatch) -> None:
    monkeypatch.setattr(onnx_engine.os, 'cpu_count', lambda: 16)
    assert [default_threads(n) for n in (1, 2, 3, 4, 8, 16)] == [7, 3, 1, 1, 1, 1]
    monkeypatch.setattr(onnx_engine.os, 'cpu_count', lambda: None)
    assert default_threads(1) == 1


def test_the_capacity_is_one_sentence_per_worker_without_vram() -> None:
    profile = audio8_onnx_capacity_profile(3)
    assert profile['engine'] == 'audio8_onnx' and profile['max_inflight_items'] == 3 and profile['max_batch_items'] == 3
    assert profile['estimated_peak_vram_mb'] == 0


def test_no_torch_is_imported() -> None:
    # The package __init__ builds the server, which imports torch for the other engines: leave it out.
    package = Path(onnx_engine.__file__).resolve().parents[1]
    code = ('import sys, types; package = types.ModuleType("omnivoice_tts_server"); '
            f'package.__path__ = [{str(package)!r}]; sys.modules["omnivoice_tts_server"] = package; '
            'from omnivoice_tts_server.audio8 import onnx_engine, onnx_codec; '
            'sys.exit(1 if "torch" in sys.modules else 0)')
    assert subprocess.run([sys.executable, '-c', code], check=False).returncode == 0


def test_split_decoder_cuts_a_small_graph(tmp_path) -> None:
    onnx = pytest.importorskip('onnx')
    pytest.importorskip('onnxruntime')
    from onnx import TensorProto, helper

    cut = onnx_codec.CUT_TENSOR
    graph = helper.make_graph(
        [helper.make_node('Cast', ['codes'], ['as_float'], to=TensorProto.FLOAT16),
         helper.make_node('Mul', ['as_float', 'two'], [cut]),
         helper.make_node('Cast', [cut], ['wide'], to=TensorProto.FLOAT),
         helper.make_node('Add', ['wide', 'one'], ['audio'])],
        'g',
        [helper.make_tensor_value_info('codes', TensorProto.INT64, [None, None, None])],
        [helper.make_tensor_value_info('audio', TensorProto.FLOAT, [None, None, None])],
        [helper.make_tensor('two', TensorProto.FLOAT16, [], [2.0]), helper.make_tensor('one', TensorProto.FLOAT, [], [1.0])])
    path = tmp_path / 'codec.onnx'
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=8), str(path))
    front_bytes, back_bytes = onnx_codec.split_decoder(path)
    front = onnx.load_from_string(front_bytes)
    back = onnx.load_from_string(back_bytes)
    assert [node.op_type for node in front.graph.node] == ['Cast', 'Mul']
    assert [node.op_type for node in back.graph.node] == ['Cast', 'Add']
    assert [weight.name for weight in front.graph.initializer] == ['two']
    assert [weight.name for weight in back.graph.initializer] == ['one']
    latents = onnx_codec.cpu_session(front_bytes, 1).run(None, {'codes': np.array([[[3]]], dtype=np.int64)})[0]
    assert latents.dtype == np.float16 and latents.tolist() == [[[6.0]]]
    audio = onnx_codec.cpu_session(back_bytes, 1).run(None, {cut: latents})[0]
    assert audio.tolist() == [[[7.0]]]


def test_split_decoder_refuses_another_graph(tmp_path) -> None:
    onnx = pytest.importorskip('onnx')
    from onnx import TensorProto, helper

    graph = helper.make_graph([helper.make_node('Identity', ['a'], ['b'])], 'g',
                              [helper.make_tensor_value_info('a', TensorProto.FLOAT, [None])],
                              [helper.make_tensor_value_info('b', TensorProto.FLOAT, [None])])
    path = tmp_path / 'other.onnx'
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=8), str(path))
    with pytest.raises(RuntimeError, match='not the pinned'):
        onnx_codec.split_decoder(path)


def test_stream_decoder_gives_the_transformer_context_and_the_convolutions_little(monkeypatch) -> None:
    class Session:
        def __init__(self, name):
            self.name = name
            self.seen = None

        def get_inputs(self):
            return [SimpleNamespace(name=self.name)]

        def run(self, _, feeds):
            self.seen = feeds[self.name]
            if self.name == 'codes':
                return [np.zeros((1, 8, self.seen.shape[-1] * onnx_codec.LATENTS_PER_FRAME), dtype=np.float16)]
            return [np.arange(self.seen.shape[-1] * HOP // onnx_codec.LATENTS_PER_FRAME, dtype=np.float32)[None, None]]

    decoder = onnx_codec.StreamDecoder.__new__(onnx_codec.StreamDecoder)
    decoder.front, decoder.back = Session('codes'), Session('latents')
    decoder.hop, decoder._codes, decoder._latents = HOP, 'codes', 'latents'
    frames = np.zeros((500, 10), dtype=np.int64)
    audio = decoder.decode(frames, 400, 412)
    assert audio.size == 12 * HOP
    assert decoder.front.seen.shape == (1, 10, onnx_codec.TRANSFORMER_CONTEXT_FRAMES)
    assert decoder.back.seen.shape[-1] == (12 + onnx_codec.CONV_CONTEXT_FRAMES) * onnx_codec.LATENTS_PER_FRAME
    assert decoder.decode(frames, 0, 3).size == 3 * HOP
    assert decoder.front.seen.shape[-1] == 3


# -- the real graphs (CPU) ------------------------------------------------------------------------------------------------

MODEL = Path(__file__).resolve().parents[2] / 'models' / AUDIO8_VARIANTS['audio8_onnx'].folder
real = pytest.mark.skipif(not (MODEL / 'runtime_manifest.json').exists(), reason='Audio8 ONNX model not downloaded')
TEXT = 'Guten Morgen, das ist ein kurzer Test der Sprachausgabe.'


@pytest.fixture(scope='module')
def engine():
    instance = Audio8OnnxEngine(MODEL, workers=2, threads=2)
    instance.load()
    yield instance
    instance.close()


@pytest.fixture(scope='module')
def reference(engine):
    manifest = engine.manifest
    codes = np.load(MODEL / manifest['reference_codes']).astype(np.int64)
    return engine._build_voice('packaged', manifest['reference_text'], codes, 0.0)


def _snr(signal: np.ndarray, reference_signal: np.ndarray) -> float:
    return float(10 * np.log10((reference_signal ** 2).sum() / max(((signal - reference_signal) ** 2).sum(), 1e-20)))


def _rendered_frames(engine, voice, text=TEXT, seed=1) -> np.ndarray:
    fragment = engine._build_fragments(text, voice)[0]
    captured = []

    class Capture:
        def publish(self, frame):
            captured.append(frame.copy())

    row = onnx_engine._Row('t', [fragment], voice, seed, None, None)
    engine._generate(row, fragment, 0, Capture())
    return np.stack(captured)


@real
def test_real_sessions_are_cpu_only(engine) -> None:
    assert engine.lm.slow.get_providers() == ['CPUExecutionProvider']
    assert engine._workers[0].decoder.back.get_providers() == ['CPUExecutionProvider']


@real
def test_real_render_is_speech_sized_ends_by_eos_and_repeats(engine, reference) -> None:
    stream = engine.submit(TEXT, prompt=reference, seed=3)
    chunks = list(stream)
    audio = np.concatenate(chunks)
    seconds = audio.size / 44100
    assert 1.5 < seconds < 8.0 and len(chunks) >= 2
    assert 'frame_cap_hits' not in stream.metrics
    assert 0.01 < float(np.sqrt(np.mean(audio ** 2))) < 0.5 and float(np.abs(audio).max()) <= 1.0
    assert np.array_equal(audio, engine.render(TEXT, prompt=reference, seed=3))


@real
def test_real_rows_side_by_side_equal_rows_alone(engine, reference) -> None:
    alone = engine.render(TEXT, prompt=reference, seed=11)
    streams = [engine.submit(TEXT, prompt=reference, seed=seed) for seed in (11, 12)]
    together = [np.concatenate(list(stream)) for stream in streams]
    assert np.array_equal(together[0], alone)


@real
def test_real_plain_synthesis_works(engine) -> None:
    assert engine.render(TEXT, seed=2).size > 44100


@real
def test_real_chunks_equal_the_whole_decode(engine, reference) -> None:
    frames = _rendered_frames(engine, reference)
    whole = onnx_codec.cpu_session(MODEL / 'codec_decoder_fp16.onnx', 2).run(None, {'codes': frames.T[None]})[0].reshape(-1)
    decoder = engine._workers[0].decoder
    assert _snr(decoder.decode(frames, 0, frames.shape[0]), whole) > 70
    parts = [decoder.decode(frames, start, min(start + 8, frames.shape[0])) for start in range(0, frames.shape[0], 8)]
    streamed = np.concatenate(parts)
    assert streamed.size == whole.size
    assert _snr(streamed, whole) > 50


@real
def test_real_one_frame_chunks_decode(engine, reference) -> None:
    frames = _rendered_frames(engine, reference)
    decoder = engine._workers[0].decoder
    assert decoder.decode(frames[:1], 0, 1).size == engine.hop
    assert decoder.decode(frames, 3, 4).size == engine.hop


@real
def test_real_long_context_stays_close_to_the_whole_decode(engine) -> None:
    codes = np.tile(np.load(MODEL / 'reference_codes.npy').astype(np.int64), (1, 5))[:, :500].T
    decoder = engine._workers[0].decoder
    streamed = np.concatenate([decoder.decode(codes, start, start + 12) for start in range(0, 492, 12)])
    whole = onnx_codec.cpu_session(MODEL / 'codec_decoder_fp16.onnx', 2).run(None, {'codes': codes[:492].T[None]})[0]
    assert _snr(streamed, whole.reshape(-1)) > 45


@real
def test_real_cloning_encodes_and_caches_a_reference(engine, tmp_path) -> None:
    clip = engine.render(TEXT, seed=5)
    first = Audio8OnnxEngine(MODEL, workers=1, threads=2, voice_cache_dir=tmp_path)
    first.load()
    try:
        started = time.perf_counter()
        voice = first.voice_prompt('clip', clip, 44100, TEXT, voice_id='voice_clip')
        encode_s = time.perf_counter() - started
        assert voice.codes.shape[0] == 10
        assert abs(voice.codes.shape[1] - clip.size / engine.hop) <= 2
        first._voices.clear()
        started = time.perf_counter()
        again = first.voice_prompt('clip', clip, 44100, TEXT, voice_id='voice_clip')
        assert time.perf_counter() - started < encode_s
        assert np.array_equal(again.codes, voice.codes)
        assert np.array_equal(again.state.ssm, voice.state.ssm)
        audio = first.render('Das ist ein geklonter Satz.', prompt=voice, seed=1)
        assert audio.size > 22050
    finally:
        first.close()
    assert len(list(tmp_path.iterdir())) == 1
    assert json.dumps(first.stats())


@real
def test_real_synthesizer_streams_a_saved_voice_through_the_queue_interface(engine, tmp_path) -> None:
    clip = (np.clip(engine.render(TEXT, seed=5), -1, 1) * 32767).astype('<i2').tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(44100)
        handle.writeframes(clip)
    profile = VoiceProfileRecord(voice_id='voice_clip', name='Clip', source='custom', created_at=utcnow(),
                                 audio_bytes=buffer.getvalue(), content_type='audio/wav', filename='x.wav', ref_text=TEXT)
    store = InMemoryStore(max_queue_size=8)
    store.voice_profiles[profile.voice_id] = profile
    settings = Settings(data_dir=tmp_path, models_root_dir=MODEL.parent, default_voice='Clip', audio8_onnx_workers=2,
                        audio8_onnx_threads=2, audio8_output_sample_rate=24000)
    synth = Audio8Synthesizer(settings, store, variant='audio8_onnx')

    def item(index, text, seed):
        request = SimpleNamespace(voice='Clip', seed=seed, language='de', stream=True, ref_audio=None, ref_text=None)
        return BatchSynthesisItem(job_id='job', sentence_index=index, request=request, text=text)

    async def run():
        results = []
        async for batch in synth.stream_batch([item(0, 'Erster Satz zum Testen.', 1), item(1, 'Noch ein zweiter Satz.', 2)]):
            results.extend(batch)
        await synth.unload()
        return results

    results = asyncio.run(run())
    assert sum(1 for result in results if result.sentence_finished) == 2
    assert all(result.error is None and result.sample_rate == 24000 for result in results)
    for index in (0, 1):
        assert sum(len(result.pcm) for result in results if result.sentence_index == index) > 2 * 24000 // 2
    assert len(list((tmp_path / 'audio8_onnx_voices').iterdir())) == 1
    assert synth.resident_models() == set()


@real
def test_real_variant_lists_files_that_exist() -> None:
    for name in AUDIO8_VARIANTS['audio8_onnx'].weight_files:
        assert (MODEL / name).is_file(), name
