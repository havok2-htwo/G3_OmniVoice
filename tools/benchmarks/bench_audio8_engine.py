"""In-process benchmark of the Audio8 slot engine (no server, no HTTP): the same measurements as the Audio8 prototype.

  * TTFA of bursts of N rows submitted together, for every N in --bursts up to the slot count;
  * saturation: one closed-loop client per slot for --seconds, audio seconds per wall second, per-request RTF;
  * stream quality: engine audio against a full decode of the same frames (SNR, step at chunk seams);
  * VRAM reserved by torch.

    python tools/benchmarks/bench_audio8_engine.py --slots 32 64 --out audio8.json
    python tools/benchmarks/bench_audio8_engine.py --model Audio8-TTS-Preview-0.6b --codec-fp32
    python tools/benchmarks/bench_audio8_engine.py --model Audio8-TTS-Preview-0.1b

The texts are the ones bench_engine_streaming.py uses for every engine. Run it with nothing else busy on the GPU.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(REPO / 'backend' / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_engine_streaming import TEXTS  # noqa: E402

# Prototype numbers from the 2026-11-01 probe (same texts, voice Genesis3, one slot count per burst size).
PROTOTYPE = {'ttfa_p50_ms': {'1': 58, '4': 99, '16': 215, '32': 380}, 'ttfa_p95_ms': {'1': 59, '4': 139, '16': 339, '32': 636},
             'saturation_x': {'32': 30.8}}
REF_TEXT = ('Franz jagt im komplett verwahrlosten Taxi quer durch Bayern. Vorbei an dichten Wäldern, kleinen Dörfern '
            'und alten Bahnhöfen.')
FAILED_THREADS: list[BaseException] = []


def load_voice(engine, name: str):
    import soundfile as sf

    audio, sr = sf.read(str(REPO / 'data' / 'voices' / f'{name}.wav'), dtype='float32')
    return engine.voice_prompt(name, audio, sr, REF_TEXT)


def burst(engine, voice, rows: int, round_: int) -> list[dict]:
    streams = [engine.submit(TEXTS[i % len(TEXTS)], prompt=voice, seed=1000 * round_ + i) for i in range(rows)]
    for stream in streams:
        for _ in stream:
            pass
    return [stream.metrics for stream in streams]


def saturation(engine, voice, clients: int, seconds: float) -> dict:
    stop = time.perf_counter() + seconds
    lock = threading.Lock()
    audio, rtfs, ttfas = [0.0], [], []

    def client(index: int) -> None:
        j = index
        try:
            while time.perf_counter() < stop:
                t0 = time.perf_counter()
                stream = engine.submit(TEXTS[j % len(TEXTS)], prompt=voice, seed=j)
                j += clients
                for _ in stream:
                    pass
                with lock:
                    audio[0] += stream.metrics['audio_s']
                    rtfs.append((time.perf_counter() - t0) / stream.metrics['audio_s'])
                    ttfas.append(stream.metrics['ttfa_ms'])
        except BaseException as exc:  # noqa: BLE001 - reported after the run
            FAILED_THREADS.append(exc)

    threads = [threading.Thread(target=client, args=(i,)) for i in range(clients)]
    t0 = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall = time.perf_counter() - t0
    return {'x_realtime': round(audio[0] / wall, 1), 'audio_s': round(audio[0], 1), 'requests': len(rtfs),
            'rtf_p50': round(statistics.median(rtfs), 3), 'rtf_p95': round(float(np.percentile(rtfs, 95)), 3),
            'ttfa_p50_ms': round(statistics.median(ttfas), 1), 'ttfa_p95_ms': round(float(np.percentile(ttfas, 95)), 1)}


def quality(engine, voice) -> dict:
    """Decode the frames of real rows both ways: chunked through the engine's SlotCodec state, and whole."""
    import torch

    from omnivoice_tts_server.audio8.codec_stream import snr_db

    results, seams, elsewhere = [], [], []
    for index, text in enumerate(TEXTS[:8]):
        stream = engine.submit(text, prompt=voice, seed=index)
        chunks = list(stream)
        audio = np.concatenate(chunks)
        steps = np.abs(np.diff(audio))
        edges = np.cumsum([chunk.size for chunk in chunks[:-1]]) - 1
        seams.append(float(steps[edges].max()))
        elsewhere.append(float(np.percentile(steps, 99.9)))

    def full_vs_stream():
        codes = voice.codes.to(engine.device)
        reference = engine.codec.decode(codes[None])[0, 0].float().cpu().numpy()
        sc = engine.slot_codec
        parts = []
        for start in range(0, codes.shape[1], sc.max_chunk):
            n = min(sc.max_chunk, codes.shape[1] - start)
            frames = torch.zeros(sc.max_chunk, codes.shape[0], device=codes.device, dtype=torch.long)
            frames[:n] = codes[:, start:start + n].T
            batch = frames[None].expand(engine.decode_tile_rows, -1, -1)
            meta = torch.zeros(engine.decode_tile_rows, 4, device=codes.device, dtype=torch.long)
            meta[:, 0] = sc.trash
            meta[0] = torch.tensor([0, start, n, int(start == 0)], device=codes.device)
            out = torch.zeros(engine.decode_tile_rows, sc.out_len(sc.max_chunk), device=codes.device)
            sc.run(batch, meta, out)
            parts.append(out[0, :n * sc.frame_samples].cpu().numpy())
        return reference, np.concatenate(parts)

    reference, streamed = engine._run_on_gpu_thread(full_vs_stream)
    results = {'stream_vs_full_decode_snr_db': round(snr_db(reference, streamed), 1),
               'seam_step_max': round(max(seams), 4), 'step_p99_9_elsewhere': round(statistics.median(elsewhere), 4)}
    return results


def run(args, slots: int) -> dict:
    import torch

    from omnivoice_tts_server.audio8.engine import Audio8Engine

    torch.backends.cuda.matmul.allow_tf32 = True  # as in the server
    t0 = time.perf_counter()
    engine = Audio8Engine(REPO / 'models' / args.model, 'cuda:0', max_slots=slots, cuda_graphs=not args.eager,
                          batch_invariant=not args.bucketed, codec_dtype=torch.float32 if args.codec_fp32 else None,
                          output_sample_rate=args.rate, decode_tile_rows=args.decode_tile or None,
                          prefill_tile_rows=args.prefill_tile, decode_hold_steps=args.hold)
    engine.load()
    load_s = time.perf_counter() - t0
    voice = load_voice(engine, args.voice)
    for i in range(2):
        burst(engine, voice, 1, i)
    out: dict = {'slots': slots, 'load_s': round(load_s, 2), 'prompt_positions': voice.length, 'ttfa': {}}
    for rows in args.bursts:
        if rows > slots:
            continue
        ttfas: list[float] = []
        for round_ in range(10 if rows == 1 else 4):
            ttfas += [m['ttfa_ms'] for m in burst(engine, voice, rows, round_)]
        out['ttfa'][str(rows)] = {'p50_ms': round(statistics.median(ttfas), 1),
                                  'p95_ms': round(float(np.percentile(ttfas, 95)), 1),
                                  'max_ms': round(max(ttfas), 1)}
    out['saturation'] = saturation(engine, voice, slots, args.seconds)
    if args.quality:
        out['quality'] = quality(engine, voice)
    stats = engine.stats()
    out['engine'] = {key: stats[key] for key in ('graphs', 'capture_s', 'warmup_s', 'step_ms', 'decode_ms', 'prefill_ms',
                                                 'decode_tile_rows', 'prefill_tile_rows', 'vram_reserved_mb',
                                                 'vram_peak_mb') if key in stats}
    engine.close()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', default='Audio8-TTS-Realistic-0.6b', help='folder below models/')
    parser.add_argument('--voice', default='voice_preset_sonjav2', help='file below data/voices/, no extension')
    parser.add_argument('--slots', type=int, nargs='+', default=[32, 64])
    parser.add_argument('--bursts', type=int, nargs='+', default=[1, 4, 16, 32, 64])
    parser.add_argument('--seconds', type=float, default=30.0, help='length of the saturation run')
    parser.add_argument('--rate', type=int, default=44100, choices=(24000, 44100))
    parser.add_argument('--eager', action='store_true', help='no CUDA graphs')
    parser.add_argument('--bucketed', action='store_true', help='batch_invariant=False')
    parser.add_argument('--codec-fp32', action='store_true')
    parser.add_argument('--decode-tile', type=int, default=0)
    parser.add_argument('--prefill-tile', type=int, default=8)
    parser.add_argument('--hold', type=int, default=4, help='steps a chunk may wait for a full decode tile')
    parser.add_argument('--no-quality', dest='quality', action='store_false')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    results = {'prototype': PROTOTYPE, 'runs': []}
    for slots in args.slots:
        result = run(args, slots)
        results['runs'].append(result)
        print(json.dumps(result), flush=True)
    if FAILED_THREADS:
        raise SystemExit(f'{len(FAILED_THREADS)} client threads failed: {FAILED_THREADS[0]!r}')
    if args.out:
        args.out.write_text(json.dumps(results, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
