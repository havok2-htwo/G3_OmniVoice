"""MingSynthesizer (ming_runtime.py) against a fake engine: lifecycle, voices, language, streaming, cancellation."""
from __future__ import annotations

import asyncio
import hashlib
import io
import queue
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile

from omnivoice_tts_server import ming_runtime
from omnivoice_tts_server.config import Settings
from omnivoice_tts_server.domain.state import InMemoryStore, VoiceProfileRecord, utcnow
from omnivoice_tts_server.ming_runtime import MING_BASE_ALIAS, MingSynthesizer, ming_language
from omnivoice_tts_server.runtime_v2 import BatchSynthesisItem

_END = object()


class FakeStream:
    """MingStream stand-in: the test (or the engine thread) pushes chunks; iteration blocks like the real one."""

    def __init__(self, row_id: str) -> None:
        self.row_id = row_id
        self._queue: queue.Queue = queue.Queue()
        self.done = False
        self.cancelled = False

    def put(self, chunk: np.ndarray) -> None:
        self._queue.put(chunk)

    def finish(self, error: BaseException | None = None) -> None:
        if not self.done:
            self.done = True
            self._queue.put(error if error is not None else _END)

    def __iter__(self):
        while True:
            item = self._queue.get()
            if item is _END:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    def cancel(self) -> None:
        self.cancelled = True
        self.finish()


class FakeEngine:
    output_sample_rate = 24_000
    instances: list['FakeEngine'] = []

    def __init__(self, auto: bool = True, **kwargs) -> None:
        self.kwargs = kwargs
        self.auto = auto  # True: every row renders 'chunks' chunks at once and ends
        self.loaded = False
        self.closed = False
        self.voices: dict[str, SimpleNamespace] = {}
        self.encoded: list[str] = []
        self.dropped: list[str] = []
        self.submitted: list[dict] = []
        self.streams: list[FakeStream] = []
        self.rendered: list[str] = []
        FakeEngine.instances.append(self)

    def load(self) -> None:
        self.loaded = True

    def close(self) -> None:
        self.closed = True
        self.loaded = False

    def voice_prompt(self, key, audio, sample_rate, ref_text, seed=0):
        if key not in self.voices:
            self.encoded.append(key)
            self.voices[key] = SimpleNamespace(key=key, samples=len(audio), sample_rate=sample_rate, ref_text=ref_text,
                                               seed=seed)
        return self.voices[key]

    def drop_voice(self, key) -> None:
        self.dropped.append(key)
        self.voices.pop(key, None)

    def submit(self, text, *, prompt, language, seed, row_id=None):
        stream = FakeStream(row_id)
        self.submitted.append({'text': text, 'prompt': prompt, 'language': language, 'seed': seed, 'row_id': row_id})
        self.streams.append(stream)
        if self.auto:
            for value in (0.25, 0.5):
                stream.put(np.full(len(text), value, dtype=np.float32))
            stream.finish()
        return stream

    def render(self, text, *, prompt=None, language=None, seed=0, row_id=None):
        self.rendered.append(text)
        return np.zeros(10, dtype=np.float32)

    def stats(self) -> dict:
        active = sum(1 for s in self.streams if not s.done)
        return {'active_rows': active, 'pending_rows': 0, 'cuda_graphs': True, 'max_slots': self.kwargs.get('max_slots')}


def _wav_bytes(seconds: float = 0.5, sample_rate: int = 24_000) -> bytes:
    buffer = io.BytesIO()
    soundfile.write(buffer, np.zeros(int(seconds * sample_rate), dtype=np.float32), sample_rate, format='WAV',
                    subtype='PCM_16')
    return buffer.getvalue()


def _profile(voice_id='voice_sonja', name='SonjaV2', ref_text='Hallo Welt, das ist Sonja.') -> VoiceProfileRecord:
    return VoiceProfileRecord(voice_id=voice_id, name=name, source='custom', created_at=utcnow(),
                              audio_bytes=_wav_bytes(), content_type='audio/wav', filename='x.wav', ref_text=ref_text)


@pytest.fixture
def synth_env(tmp_path, monkeypatch):
    # The real hook changes the process-wide SDPA order; test_load_restores_the_default_sdpa_order covers it.
    monkeypatch.setattr(MingSynthesizer, '_reset_process_kernel_choices', staticmethod(lambda: None))
    model_dir = tmp_path / 'Ming-3.0-tts-0.5b'
    model_dir.mkdir()
    (model_dir / 'config.json').write_text('{}', encoding='utf-8')
    settings = Settings(data_dir=tmp_path / 'data', ming_model_dir=model_dir, ming_max_slots=8, preferred_device='cuda',
                        default_voice='SonjaV2')
    store = InMemoryStore(max_queue_size=8)
    profile = _profile()
    store.voice_profiles[profile.voice_id] = profile
    FakeEngine.instances = []

    def make(auto: bool = True) -> MingSynthesizer:
        return MingSynthesizer(settings, store, engine_factory=lambda **kw: FakeEngine(auto=auto, **kw))

    return SimpleNamespace(settings=settings, store=store, profile=profile, make=make, model_dir=model_dir)


def _item(index, text, job_id='job', **request) -> BatchSynthesisItem:
    fields = {'voice': 'SonjaV2', 'seed': None, 'language': 'de', 'stream': True, 'ref_audio': None, 'ref_text': None}
    fields.update(request)
    return BatchSynthesisItem(job_id=job_id, sentence_index=index, request=SimpleNamespace(**fields), text=text)


async def _collect(synth, items):
    out = []
    async for batch in synth.stream_batch(items):
        out.extend(batch)
    return out


def test_ensure_model_builds_the_engine_from_settings(synth_env) -> None:
    synth = synth_env.make()
    assert synth.resident_models() == set()
    model, warm_ms = asyncio.run(synth.ensure_model('k2-fsa/OmniVoice-Base'))  # any id: one checkpoint serves all
    assert model == MING_BASE_ALIAS and warm_ms >= 0
    engine = FakeEngine.instances[-1]
    assert engine.loaded
    assert engine.kwargs['device'] == 'cuda:0'
    assert engine.kwargs['max_slots'] == 8
    assert engine.kwargs['output_sample_rate'] == 24_000
    assert engine.kwargs['first_chunk_latents'] == 2 and engine.kwargs['max_chunk_latents'] == 8
    assert engine.kwargs['plan_cache'] == synth_env.settings.data_dir / 'ming_linear_plans.json'
    # The default voice prompt is encoded at load, not in the first request.
    assert len(engine.encoded) == 1 and engine.encoded[0].startswith('ming|')
    assert synth.resident_models() == {MING_BASE_ALIAS, 'inclusionAI/Ming-3.0-tts-0.5B'}
    assert asyncio.run(synth.ensure_model(None)) == (MING_BASE_ALIAS, 0)
    assert len(FakeEngine.instances) == 1
    assert synth.capacity_profile['max_inflight_items'] == 8
    assert synth.supports_native_streaming and synth.supports_concurrent_batches and synth.uses_voice_profiles


def test_missing_checkpoint_fails_clearly(synth_env) -> None:
    (synth_env.model_dir / 'config.json').unlink()
    synth = synth_env.make()
    with pytest.raises(RuntimeError, match='Ming checkpoint not found'):
        asyncio.run(synth.ensure_model(None))
    assert synth.resident_models() == set()


def test_stream_batch_forwards_chunks_then_one_marker_per_sentence(synth_env) -> None:
    synth = synth_env.make()
    items = [_item(0, 'abcdef'), _item(1, 'xyz')]
    results = asyncio.run(_collect(synth, items))
    for index, text in ((0, 'abcdef'), (1, 'xyz')):
        mine = [r for r in results if r.sentence_index == index]
        assert [bool(r.sentence_finished) for r in mine] == [False, False, True]
        assert all(r.sample_rate == 24_000 for r in mine)
        assert [len(r.pcm) // 2 for r in mine] == [len(text), len(text), 0]
        first = np.frombuffer(mine[0].pcm, dtype='<i2')
        assert int(first[0]) == int(0.25 * 32767)
    engine = FakeEngine.instances[-1]
    assert [s['row_id'] for s in engine.submitted] == ['job:0', 'job:1']


def test_first_chunk_arrives_while_the_row_still_renders(synth_env) -> None:
    synth = synth_env.make(auto=False)
    asyncio.run(synth.ensure_model(None))
    engine = FakeEngine.instances[-1]

    async def run():
        gen = synth.stream_batch([_item(0, 'Ein Satz.')])
        task = asyncio.ensure_future(gen.__anext__())
        for _ in range(200):
            if engine.streams:
                break
            await asyncio.sleep(0.005)
        stream = engine.streams[0]
        stream.put(np.full(4, 0.1, dtype=np.float32))
        first = await asyncio.wait_for(task, timeout=2)
        assert not stream.done, 'the chunk was forwarded before the row ended'
        stream.put(np.full(4, 0.2, dtype=np.float32))
        stream.finish()
        rest = []
        async for batch in gen:
            rest.extend(batch)
        return first, rest

    first, rest = asyncio.run(run())
    assert len(first) == 1 and len(first[0].pcm) == 8 and not first[0].sentence_finished
    assert [r.sentence_finished for r in rest][-1] is True
    assert sum(1 for r in rest if r.sentence_finished) == 1


def test_voice_profile_prompt_is_cached_and_keyed_by_audio_and_transcript(synth_env) -> None:
    synth = synth_env.make()
    asyncio.run(synth.ensure_model(None))
    engine = FakeEngine.instances[-1]
    profile = synth_env.profile
    expected = '|'.join(['ming', 'inclusionAI/Ming-3.0-tts-0.5B', profile.voice_id,
                         hashlib.sha1(profile.audio_bytes).hexdigest(), profile.ref_text])
    asyncio.run(_collect(synth, [_item(0, 'Eins.'), _item(1, 'Zwei.', voice=profile.voice_id)]))
    assert engine.encoded == [expected]
    prompts = [s['prompt'] for s in engine.submitted]
    assert prompts[0] is prompts[1] and prompts[0].key == expected
    assert prompts[0].samples == 12_000 and prompts[0].sample_rate == 24_000
    # A request transcript overrides the stored one (new key).
    asyncio.run(_collect(synth, [_item(0, 'Drei.', ref_text='Anderer Text.')]))
    assert engine.encoded[-1].endswith('|Anderer Text.')


def test_router_cache_clear_drops_the_encoded_prompts(synth_env) -> None:
    synth = synth_env.make()
    asyncio.run(synth.ensure_model(None))
    engine = FakeEngine.instances[-1]
    key = engine.encoded[0]
    synth_env.store.prompt_cache.clear()  # what voice create / delete do
    asyncio.run(_collect(synth, [_item(0, 'Eins.')]))
    assert engine.dropped == [key]
    assert engine.encoded == [key, key], 're-encoded after the clear'
    asyncio.run(_collect(synth, [_item(0, 'Zwei.')]))
    assert engine.dropped == [key], 'no further drops while the cache marker stays'


def test_inline_reference_audio_and_no_voice(synth_env, tmp_path) -> None:
    synth = synth_env.make()
    asyncio.run(synth.ensure_model(None))
    engine = FakeEngine.instances[-1]
    ref = tmp_path / 'ref.wav'
    ref.write_bytes(_wav_bytes(0.25, 16_000))
    asyncio.run(_collect(synth, [
        _item(0, 'Inline.', voice=None, ref_audio=str(ref), ref_text='Referenz.'),
        _item(1, 'Ohne Stimme.', voice='Auto Voice'),
    ]))
    inline, plain = engine.submitted
    assert inline['prompt'].key.split('|')[2] == 'inline' and inline['prompt'].sample_rate == 16_000
    assert plain['prompt'] is None
    with pytest.raises(RuntimeError, match='ref_text'):
        asyncio.run(_collect(synth, [_item(0, 'x', voice=None, ref_audio=str(ref))]))


def test_seed_and_language_mapping(synth_env) -> None:
    synth = synth_env.make()
    asyncio.run(_collect(synth, [
        _item(0, 'Eins.', seed=None, language='Deutsch'),
        _item(1, 'Hello, this is the English text of the request.', seed=42, language='Auto'),
        _item(2, 'Zwei.', language=None),
        _item(3, 'Trois.', language='fr'),
    ]))
    submitted = FakeEngine.instances[-1].submitted
    assert [s['seed'] for s in submitted] == [ming_runtime.MING_DEFAULT_SEED, 42, 0, 0]
    assert [s['language'] for s in submitted] == ['de', 'en', 'de', 'fr']


def test_ming_language() -> None:
    assert ming_language('de-DE') == 'de'
    assert ming_language('English') == 'en'
    assert ming_language('auto', 'Der Hund und die Katze sind nicht im Haus.') == 'de'
    assert ming_language('auto', 'The dog and the cat are not in the house.') == 'en'
    assert ming_language('auto', '12345') == 'de'
    assert ming_language('pl', 'Tekst.') == 'de', 'a language Ming has no name for falls back'


def test_engine_error_propagates_and_cancels_the_other_rows(synth_env) -> None:
    synth = synth_env.make(auto=False)
    asyncio.run(synth.ensure_model(None))
    engine = FakeEngine.instances[-1]

    async def run():
        gen = synth.stream_batch([_item(0, 'a'), _item(1, 'b')])
        task = asyncio.ensure_future(_drain(gen))
        for _ in range(200):
            if len(engine.streams) == 2:
                break
            await asyncio.sleep(0.005)
        engine.streams[0].finish(RuntimeError('CUDA step failed'))
        await asyncio.wait_for(task, timeout=2)

    async def _drain(gen):
        async for _ in gen:
            pass

    with pytest.raises(RuntimeError, match='CUDA step failed'):
        asyncio.run(run())
    assert engine.streams[1].cancelled


def test_closing_the_generator_and_cancel_job_stop_the_rows(synth_env) -> None:
    synth = synth_env.make(auto=False)
    asyncio.run(synth.ensure_model(None))
    engine = FakeEngine.instances[-1]

    async def run():
        gen = synth.stream_batch([_item(0, 'a', job_id='j1'), _item(1, 'b', job_id='j1'), _item(0, 'c', job_id='j2')])
        task = asyncio.ensure_future(gen.__anext__())
        for _ in range(200):
            if len(engine.streams) == 3:
                break
            await asyncio.sleep(0.005)
        assert synth.cancel_job('j1') == 2
        first = await asyncio.wait_for(task, timeout=2)
        assert {r.job_id for r in first} <= {'j1'} and all(r.sentence_finished for r in first)
        await gen.aclose()

    asyncio.run(run())
    assert [s.cancelled for s in engine.streams] == [True, True, True]
    assert synth._job_streams == {}


def test_render_batch_returns_whole_sentences(synth_env) -> None:
    synth = synth_env.make()
    results = asyncio.run(synth.render_batch([_item(0, 'abcd', stream=False), _item(1, 'xy', stream=False)]))
    assert [(r.sentence_index, len(r.pcm) // 2, r.sample_rate) for r in results] == [(0, 8, 24_000), (1, 4, 24_000)]
    assert all(not r.sentence_finished for r in results)


def test_unload_free_memory_and_warmup(synth_env, monkeypatch) -> None:
    monkeypatch.setattr(ming_runtime, 'query_nvidia_smi', lambda device=None: {'memory_used_mb': 5000,
                                                                                'memory_total_mb': 32000})
    synth = synth_env.make()
    model, _ = asyncio.run(synth.warmup())
    engine = FakeEngine.instances[-1]
    # One render at load (first-call paths) and one for the warmup itself.
    assert model == MING_BASE_ALIAS and engine.rendered == [ming_runtime.WARMUP_TEXT] * 2
    info = synth.runtime_info()
    assert info['engine'] == 'ming' and info['loaded'] and info['engine_stats']['max_slots'] == 8
    cleanup = asyncio.run(synth.free_memory())
    assert cleanup['ok'] and set(cleanup) == {'ok', 'message', 'memory_before_mb', 'memory_after_mb',
                                              'memory_total_mb', 'released_mb'}
    assert engine.closed and synth.resident_models() == set()
    asyncio.run(synth.ensure_model(None))
    assert asyncio.run(synth.unload()) == (MING_BASE_ALIAS, 0)
    assert FakeEngine.instances[-1].closed
    wav = synth.pcm_to_wav(b'\x00\x00' * 24)
    assert wav[:4] == b'RIFF'


def test_free_memory_keeps_a_busy_engine(synth_env, monkeypatch) -> None:
    monkeypatch.setattr(ming_runtime, 'query_nvidia_smi', lambda device=None: {})
    synth = synth_env.make(auto=False)
    asyncio.run(synth.ensure_model(None))
    engine = FakeEngine.instances[-1]
    engine.streams.append(FakeStream('busy'))
    cleanup = asyncio.run(synth.free_memory())
    assert not engine.closed and 'rendering' in cleanup['message']


def test_submit_runs_off_the_event_loop(synth_env) -> None:
    synth = synth_env.make()
    asyncio.run(synth.ensure_model(None))
    engine = FakeEngine.instances[-1]
    threads: list[str] = []
    original = engine.submit

    def submit(*args, **kwargs):
        threads.append(threading.current_thread().name)
        return original(*args, **kwargs)

    engine.submit = submit
    asyncio.run(_collect(synth, [_item(0, 'a')]))
    assert threads and threads[0] != threading.main_thread().name


def test_load_restores_the_default_sdpa_order(synth_env, monkeypatch) -> None:
    import omnivoice_tts_server.runtime_v2 as runtime_v2

    pytest.importorskip('torch')  # the hook only acts once torch is imported
    monkeypatch.undo()
    calls = []
    monkeypatch.setattr(runtime_v2, 'apply_sdpa_backend_preference', lambda torch, prefer: calls.append(prefer))
    synth = synth_env.make()
    asyncio.run(synth.ensure_model(None))
    assert calls == [False], 'graphs are captured with the default SDPA order, whatever OmniVoice set before'
