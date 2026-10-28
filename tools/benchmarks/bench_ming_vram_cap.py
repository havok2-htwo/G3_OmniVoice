"""Ming engine under a hard VRAM cap (RTX 3090 / shared-card simulation), in-process.

torch.cuda.set_per_process_memory_fraction caps the caching allocator of this process: allocations beyond the
cap raise OOM exactly as on a smaller card. For each max_slots value the engine is loaded under the cap, every
slot renders a long multi-fragment German text at once (the worst case for the KV cache and the VAE tiles),
and the run must finish without OOM. Reports torch peak allocated / reserved, the card delta seen by
nvidia-smi (CUDA context included) and whether the rows completed.

Also prints a static scan of the Ming package for code that would not run on Ampere (sm_86): fp8 dtypes,
flash-attn / triton / FlashInfer imports, torch.compile, compute-capability gates.

  python bench_ming_vram_cap.py --slots 16,32,64 --cap-gb 3.7 --out cap.json
  python bench_ming_vram_cap.py --slots 64 --card-gb 24 --out cap24.json   # whole 3090, as E2 did
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'backend' / 'src'))

LONG_TEXT = (
    'Morgen früh wird es in weiten Teilen des Landes neblig und ziemlich kühl. Am Nachmittag lockert die '
    'Bewölkung auf und die Sonne kommt immer öfter hervor. Im Norden weht ein frischer Wind von der See, im Süden '
    'bleibt es dagegen meist windstill und spürbar wärmer. Zum Wochenende hin steigen die Temperaturen weiter an, '
    'und in den Bergen kann es am Abend vereinzelt Gewitter geben. Wer einen Ausflug plant, sollte deshalb einen '
    'Regenschirm einpacken und die aktuellen Warnungen im Blick behalten.'
)
ARCH_PATTERNS = {
    'fp8 dtype': r'float8|fp8',
    'flash-attn package': r'import flash_attn|from flash_attn',
    'triton': r'import triton|from triton',
    'flashinfer': r'flashinfer',
    'torch.compile': r'torch\.compile',
    'capability gate': r'get_device_capability|sm_9|sm_10|sm_12',
}


def arch_scan() -> dict[str, list[str]]:
    hits: dict[str, list[str]] = {}
    for path in sorted((REPO / 'backend' / 'src' / 'omnivoice_tts_server' / 'ming').glob('*.py')):
        for n, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            code = line.split('#', 1)[0]
            for label, pattern in ARCH_PATTERNS.items():
                if re.search(pattern, code):
                    hits.setdefault(label, []).append(f'{path.name}:{n}: {line.strip()[:120]}')
    return hits


def card_used_mb() -> int:
    out = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                  text=True)
    return int(out.strip().splitlines()[0])


def load_voice(name: str) -> tuple[np.ndarray, int, str]:
    import soundfile

    profiles = json.loads((REPO / 'data' / 'voice_profiles.json').read_text(encoding='utf-8'))
    items = profiles if isinstance(profiles, list) else profiles.get('profiles', list(profiles.values()))
    for item in items:
        if item.get('name') == name or item.get('voice_id') == name:
            wav = REPO / 'data' / 'voices' / f'{item["voice_id"]}.wav'
            if not wav.exists():
                wav = REPO / 'data' / 'voices' / item['filename']
            audio, sr = soundfile.read(str(wav), dtype='float32', always_2d=False)
            return audio, sr, item['ref_text']
    raise SystemExit(f'voice {name} not found')


def run_one(slots: int, cap_bytes: int | None, voice: tuple[np.ndarray, int, str], rows: int) -> dict:
    import torch
    from omnivoice_tts_server.ming.engine import MingEngine

    dev = torch.device('cuda:0')
    total = torch.cuda.get_device_properties(dev).total_memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)
    if cap_bytes:
        torch.cuda.set_per_process_memory_fraction(min(1.0, cap_bytes / total), dev)
    base_mb = card_used_mb()
    result: dict = {'slots': slots, 'rows': rows, 'cap_mb': round(cap_bytes / 2 ** 20) if cap_bytes else None}
    engine = MingEngine(REPO / 'models' / 'Ming-3.0-tts-0.5b', 'cuda:0', max_slots=slots)
    try:
        t0 = time.perf_counter()
        engine.load()
        result['load_s'] = time.perf_counter() - t0
        audio, sr, ref_text = voice
        prompt = engine.voice_prompt('bench', audio, sr, ref_text, seed=0)
        result['idle_torch_reserved_mb'] = torch.cuda.memory_reserved(dev) / 2 ** 20
        result['idle_card_delta_mb'] = card_used_mb() - base_mb
        streams = [engine.submit(LONG_TEXT, prompt=prompt, language='de', seed=i) for i in range(rows)]
        audio_s = [0.0] * rows
        errors: list[str] = []
        peak_card = [0]
        stop = threading.Event()

        def sample() -> None:
            while not stop.is_set():
                peak_card[0] = max(peak_card[0], card_used_mb())
                time.sleep(0.1)

        def consume(i: int) -> None:
            try:
                for chunk in streams[i]:
                    audio_s[i] += len(chunk) / engine.output_sample_rate
            except Exception as exc:  # noqa: BLE001
                errors.append(f'{type(exc).__name__}: {exc}')

        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        t0 = time.perf_counter()
        threads = [threading.Thread(target=consume, args=(i,)) for i in range(rows)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.perf_counter() - t0
        stop.set()
        sampler.join()
        result.update({
            'ok': not errors, 'errors': errors[:3], 'wall_s': wall, 'audio_s_per_row_min': min(audio_s),
            'audio_s_sum': sum(audio_s), 'audio_s_per_wall_s': sum(audio_s) / wall,
            'torch_peak_allocated_mb': torch.cuda.max_memory_allocated(dev) / 2 ** 20,
            'torch_peak_reserved_mb': torch.cuda.max_memory_reserved(dev) / 2 ** 20,
            'card_peak_delta_mb': peak_card[0] - base_mb,
            'step_ms': engine.stats().get('step_ms'),
        })
    except Exception as exc:  # noqa: BLE001 - an OOM under the cap is the result
        result.update({'ok': False, 'errors': [f'{type(exc).__name__}: {exc}'[:300]]})
    finally:
        engine.close()
        torch.cuda.set_per_process_memory_fraction(1.0, dev)
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--slots', default='16,32,64')
    parser.add_argument('--cap-gb', type=float, default=None, help='torch allocator cap for this process (GiB)')
    parser.add_argument('--card-gb', type=float, default=None, help='simulate a card of this size (GiB, whole card)')
    parser.add_argument('--rows', type=int, default=0, help='rows per run (0 = all slots)')
    parser.add_argument('--voice', default='SonjaV2')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    import torch

    cap = None
    if args.cap_gb:
        cap = int(args.cap_gb * 2 ** 30)
    elif args.card_gb:
        cap = int(args.card_gb * 2 ** 30)
    voice = load_voice(args.voice)
    report = {
        'device': torch.cuda.get_device_name(0), 'capability': list(torch.cuda.get_device_capability(0)),
        'torch': torch.__version__, 'cuda': torch.version.cuda, 'cap_gb': args.cap_gb, 'card_gb': args.card_gb,
        'arch_scan': arch_scan(), 'runs': [],
    }
    for slots in [int(s) for s in args.slots.split(',') if s.strip()]:
        res = run_one(slots, cap, voice, args.rows or slots)
        report['runs'].append(res)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)
    print('arch scan:', json.dumps(report['arch_scan'], indent=1))
    args.out.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
