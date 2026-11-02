"""In-process benchmark of the Audio8 CPU engine (no server, no HTTP, no GPU).

  * load time and process memory;
  * voice: first encode (encoder + prompt head), and the same with the codes coming from the disk cache;
  * TTFA and RTF of one stream, and of N streams at once;
  * saturation: one closed-loop client per worker for --seconds, audio seconds per wall second;
  * playback gaps: a player that starts with the first chunk and plays at realtime, how long it runs dry;
  * stream quality: a fragment decoded in chunks against all of its frames at once.

    python tools/benchmarks/bench_audio8_onnx.py --layouts 1x7 2x3 3x2 4x1 --out audio8_onnx.json

The texts are the ones bench_engine_streaming.py uses for every engine.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(REPO / 'backend' / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_engine_streaming import TEXTS  # noqa: E402

REF_TEXT = ('Franz jagt im komplett verwahrlosten Taxi quer durch Bayern. Vorbei an dichten Wäldern, kleinen Dörfern '
            'und alten Bahnhöfen.')
MODEL = REPO / 'models' / 'Audio8-TTS-Preview-0.1B-ONNX-INT8'
FAILED_THREADS: list[BaseException] = []


def rss_mb() -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    return round(psutil.Process(os.getpid()).memory_info().rss / 2 ** 20, 1)


def play(stream, rate: int) -> dict:
    """Consume a stream like a player that starts at the first chunk: returns TTFA, RTF and the time it runs dry."""
    t0 = stream.submitted_at
    playhead = 0.0   # wall time at which the audio received so far runs out
    gap = 0.0
    gaps = 0
    samples = 0
    first = None
    for chunk in stream:
        now = time.perf_counter() - t0
        if first is None:
            first = now
            playhead = now
        elif now > playhead:
            gap += now - playhead
            gaps += 1
            playhead = now
        playhead += chunk.size / rate
        samples += chunk.size
    wall = time.perf_counter() - t0
    audio = samples / rate
    return {'ttfa_ms': first * 1000.0, 'rtf': wall / audio, 'audio_s': audio, 'gap_ms': gap * 1000.0, 'gaps': gaps,
            'after_ttfa_rtf': (wall - first) / audio}


def together(engine, voice, rows: int, round_: int) -> list[dict]:
    streams = [engine.submit(TEXTS[(round_ * rows + i) % len(TEXTS)], prompt=voice, seed=1000 * round_ + i)
               for i in range(rows)]
    results: list[dict | None] = [None] * rows

    def consume(index: int) -> None:
        try:
            results[index] = play(streams[index], engine.output_sample_rate)
        except BaseException as exc:  # noqa: BLE001 - reported after the run
            FAILED_THREADS.append(exc)

    threads = [threading.Thread(target=consume, args=(i,)) for i in range(rows)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return [result for result in results if result is not None]


def summary(results: list[dict]) -> dict:
    def stat(name: str) -> tuple[float, float]:
        values = [row[name] for row in results]
        return round(statistics.median(values), 3), round(float(np.percentile(values, 95)), 3)

    return {'n': len(results), 'ttfa_p50_ms': stat('ttfa_ms')[0], 'ttfa_p95_ms': stat('ttfa_ms')[1],
            'rtf_p50': stat('rtf')[0], 'rtf_p95': stat('rtf')[1], 'gap_ms_p50': stat('gap_ms')[0],
            'gap_ms_p95': stat('gap_ms')[1], 'rows_with_gap': sum(1 for row in results if row['gaps'])}


def saturation(engine, voice, clients: int, seconds: float) -> dict:
    stop = time.perf_counter() + seconds
    lock = threading.Lock()
    audio = [0.0]
    rows: list[dict] = []

    def client(index: int) -> None:
        j = index
        try:
            while time.perf_counter() < stop:
                result = play(engine.submit(TEXTS[j % len(TEXTS)], prompt=voice, seed=j), engine.output_sample_rate)
                j += clients
                with lock:
                    audio[0] += result['audio_s']
                    rows.append(result)
        except BaseException as exc:  # noqa: BLE001 - reported after the run
            FAILED_THREADS.append(exc)

    threads = [threading.Thread(target=client, args=(i,)) for i in range(clients)]
    t0 = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall = time.perf_counter() - t0
    return {'x_realtime': round(audio[0] / wall, 2), 'audio_s': round(audio[0], 1), **summary(rows)}


def quality(engine, voice) -> dict:
    """One real row's frames, decoded in engine chunks and whole. 'Whole' is the original graph, not the two halves."""
    from omnivoice_tts_server.audio8 import onnx_codec, onnx_engine

    lm = engine.lm
    fragment = engine._build_fragments(TEXTS[3], voice)[0]
    captured: list[np.ndarray] = []

    class Capture:
        def publish(self, frame: np.ndarray) -> None:
            captured.append(frame.copy())

    row = onnx_engine._Row('quality', [fragment], voice, 3, None, None)
    engine._generate(row, fragment, 0, Capture())
    frames = np.stack(captured)
    full = onnx_codec.cpu_session(engine.model_dir / 'codec_decoder_fp16.onnx', engine.threads)
    reference = full.run(None, {'codes': frames.T[None]})[0].reshape(-1)
    decoder = engine._workers[0].decoder
    parts, start = [], 0
    while start < frames.shape[0]:
        end = min(frames.shape[0], start + engine.chunk_frames)
        parts.append(decoder.decode(frames, start, end))
        start = end
    stream = np.concatenate(parts)
    noise = float(((stream - reference) ** 2).sum())
    seams = np.abs(np.diff(stream))
    edges = np.cumsum([part.size for part in parts[:-1]]) - 1
    return {'frames': int(frames.shape[0]), 'lm_ms_per_frame': engine.stats()['frame_ms'],
            'stream_vs_full_decode_snr_db': round(10 * np.log10(float((reference ** 2).sum()) / max(noise, 1e-20)), 1),
            'seam_step_max': round(float(seams[edges].max()), 4),
            'step_p99_9_elsewhere': round(float(np.percentile(seams, 99.9)), 4)}


def run(args, workers: int, threads: int) -> dict:
    import soundfile as sf

    from omnivoice_tts_server.audio8.onnx_engine import Audio8OnnxEngine

    before = rss_mb()
    with tempfile.TemporaryDirectory() as cache:
        engine = Audio8OnnxEngine(MODEL, workers=workers, threads=threads, output_sample_rate=args.rate,
                                  first_chunk_frames=args.first, chunk_frames=args.chunk, voice_cache_dir=cache)
        engine.load()
        out: dict = {'workers': workers, 'threads': engine.threads, 'first_chunk_frames': args.first,
                     'chunk_frames': args.chunk, 'load_s': round(engine.load_s, 2), 'rss_before_mb': before,
                     'rss_loaded_mb': rss_mb()}
        audio, rate = sf.read(str(REPO / 'data' / 'voices' / f'{args.voice}.wav'), dtype='float32')
        t0 = time.perf_counter()
        voice = engine.voice_prompt(args.voice, audio, rate, REF_TEXT)
        out['voice_first_s'] = round(time.perf_counter() - t0, 2)
        out['voice_positions'] = voice.length
        out['voice_seconds'] = round(voice.duration_s, 1)
        engine._voices.clear()
        t0 = time.perf_counter()
        voice = engine.voice_prompt(args.voice, audio, rate, REF_TEXT)
        out['voice_from_disk_s'] = round(time.perf_counter() - t0, 2)
        out['rss_voice_mb'] = rss_mb()
        for label, rows in (('1', 1), *((str(n), n) for n in args.bursts)):
            if rows <= 1 and label != '1':
                continue
            results: list[dict] = []
            for round_ in range(args.rounds):
                results += together(engine, voice, rows, round_)
            out[f'streams_{label}'] = summary(results)
        out['saturation'] = saturation(engine, voice, workers, args.seconds)
        out['rss_peak_like_mb'] = rss_mb()
        out['quality'] = quality(engine, voice)
        out['failed'] = [repr(exc) for exc in FAILED_THREADS]
        engine.close()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--layouts', nargs='+', default=['2x3'], help='WORKERSxTHREADS of the decoder')
    parser.add_argument('--voice', default='voice_preset_sonjav2', help='file below data/voices/, no extension')
    parser.add_argument('--rate', type=int, default=44100)
    parser.add_argument('--first', type=int, default=6)
    parser.add_argument('--chunk', type=int, default=8)
    parser.add_argument('--bursts', type=int, nargs='*', default=[2, 4])
    parser.add_argument('--rounds', type=int, default=4)
    parser.add_argument('--seconds', type=float, default=30.0)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    results = []
    for layout in args.layouts:
        workers, threads = (int(part) for part in layout.split('x'))
        result = run(args, workers, threads)
        results.append(result)
        print(json.dumps(result, indent=1), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
