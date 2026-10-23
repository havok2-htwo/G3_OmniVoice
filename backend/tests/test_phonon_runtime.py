import asyncio
import queue
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np

from omnivoice_tts_server.phonon_runtime import PhononSynthesizer
from omnivoice_tts_server.runtime_v2 import BatchSynthesisItem


class FakeEngine:
    """Stands in for ptts.TTS: one constant sample per character, streamed in two halves."""

    def synth(self, text, voice=None, seed=None):
        return np.full(len(text), 0.5, dtype=np.float32)

    def stream(self, text, voice=None, seed=None):
        audio = self.synth(text)
        half = len(audio) // 2
        yield audio[:half]
        yield audio[half:]


def _synthesizer(workers: int = 2) -> PhononSynthesizer:
    settings = SimpleNamespace(phonon_workers=workers, phonon_threads=0, phonon_model_id='gradium/phonon-40m-multilingual')
    synth = PhononSynthesizer(settings=settings, store=None)
    for _ in range(workers):
        synth._free.put(FakeEngine())
    synth._executor = ThreadPoolExecutor(max_workers=workers)
    synth._voice_names = ['default']
    synth._loaded_model_id = synth.model_id
    return synth


def _items(texts):
    request = SimpleNamespace(voice=None, seed=None, stream=True)
    return [BatchSynthesisItem(job_id='job', sentence_index=i, request=request, text=t) for i, t in enumerate(texts)]


def test_render_batch_keeps_sentence_order_and_lengths():
    synth = _synthesizer()
    items = _items(['a' * 10, 'b' * 4, 'c' * 7])
    results = asyncio.run(synth.render_batch(items))
    assert [r.sentence_index for r in results] == [0, 1, 2]
    assert [len(r.pcm) // 2 for r in results] == [10, 4, 7]
    assert all(r.sample_rate == 24_000 for r in results)


def test_stream_batch_delivers_every_chunk_then_one_finish_marker_per_sentence():
    synth = _synthesizer()
    items = _items(['x' * 6, 'y' * 8, 'z' * 2])

    async def collect():
        out = []
        async for batch in synth.stream_batch(items):
            out.extend(batch)
        return out

    results = asyncio.run(collect())
    for index, item in enumerate(items):
        mine = [r for r in results if r.sentence_index == index]
        finished = [r for r in mine if r.sentence_finished]
        audio = [r for r in mine if not r.sentence_finished]
        assert len(finished) == 1 and mine[-1].sentence_finished
        assert sum(len(r.pcm) // 2 for r in audio) == len(item.text)
        assert len(audio) == 2  # FakeEngine yields two chunks per sentence


def test_stream_batch_raises_when_an_engine_fails():
    synth = _synthesizer(workers=1)

    class Broken(FakeEngine):
        def stream(self, text, voice=None, seed=None):
            raise RuntimeError('engine down')
            yield  # pragma: no cover

    synth._free = queue.Queue()
    synth._free.put(Broken())

    async def collect():
        async for _ in synth.stream_batch(_items(['boom'])):
            pass

    try:
        asyncio.run(collect())
    except RuntimeError as exc:
        assert 'engine down' in str(exc)
    else:
        raise AssertionError('stream_batch swallowed the engine error')
