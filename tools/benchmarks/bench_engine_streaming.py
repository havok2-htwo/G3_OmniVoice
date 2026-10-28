"""Client-side streaming benchmark for the G3 TTS server, for any engine (OmniVoice, Ming, Phonon).

Measures what a client sees over HTTP, nothing in-process:
  * TTFA (request sent -> first audio bytes) on the NDJSON stream (/api/v1/synthesize/stream) and on the
    raw 24 kHz PCM stream (/v1/audio/speech, stream=true), for bursts of N simultaneous requests;
  * per-request RTF (wall time of the request / seconds of audio it returned);
  * saturation throughput: C closed-loop clients for D seconds, audio seconds per wall second;
  * VRAM via nvidia-smi (whole card, sampled every 100 ms): baseline before the server starts, idle after
    warmup, peak during the run. On Windows (WDDM) nvidia-smi cannot report per-process memory, so the
    numbers are card totals and the server's share is the delta to the baseline: run with nothing else
    busy on the card.

Two ways to run it:
  1. Against a server that is already running:
       python bench_engine_streaming.py --base-url http://127.0.0.1:8191 --out result.json
  2. Spawn a server with a given engine and settings in a throwaway data dir (voice profiles are copied from
     the repo's data/, secrets are not), benchmark it and stop it again:
       python bench_engine_streaming.py --spawn ming --set ming_max_slots=32 --set max_parallel_requests=32 \
           --work-dir C:/tmp/bench --out ming32.json
     --spawn omnivoice --preset "32GB VRAM" applies an OmniVoice tier's values from presets.py.

Requires httpx + numpy (both in the backend env). The WER round trip lives in bench_wer_roundtrip.py.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_PYTHON = REPO / '.conda-env' / 'python.exe'

# Single German sentences, about 3-6 s of speech each; the same list for every engine.
TEXTS = [
    'Der Zug nach Hamburg fährt heute ausnahmsweise von Gleis sieben ab.',
    'Bitte beachten Sie die geänderte Wagenreihung am Bahnsteig.',
    'Die Abfahrt verzögert sich leider um etwa zehn Minuten.',
    'Im Bordbistro erhalten Sie warme und kalte Getränke sowie kleine Snacks.',
    'Nächster Halt ist Hannover Hauptbahnhof, der Ausstieg ist in Fahrtrichtung rechts.',
    'Wir danken Ihnen für Ihre Geduld und wünschen Ihnen eine gute Reise.',
    'Reisende nach Berlin steigen bitte in den bereitstehenden Zug auf dem Nachbargleis um.',
    'Das Rauchen ist im gesamten Zug und auf den Bahnsteigen nicht gestattet.',
    'Morgen früh wird es in weiten Teilen des Landes neblig und ziemlich kühl.',
    'Am Nachmittag lockert die Bewölkung auf und die Sonne kommt immer öfter hervor.',
    'Die Bibliothek bleibt wegen Renovierungsarbeiten bis Ende des Monats geschlossen.',
    'Unser Kundenservice ist montags bis freitags von acht bis achtzehn Uhr erreichbar.',
    'Bitte halten Sie Ihre Kundennummer bereit, damit wir Ihnen schneller helfen können.',
    'Die neue Brücke über den Fluss soll im kommenden Frühjahr eröffnet werden.',
    'In der Altstadt findet am Wochenende wieder der beliebte Bauernmarkt statt.',
    'Für die Wanderung empfehlen wir festes Schuhwerk und ausreichend Wasser.',
]


# ----------------------------------------------------------------------------------------------------------------
# VRAM sampling (nvidia-smi, whole card)
# ----------------------------------------------------------------------------------------------------------------

class VramSampler:
    """Samples the card's memory.used (MiB) every interval_ms with one long-running nvidia-smi process."""

    def __init__(self, gpu_index: int = 0, interval_ms: int = 100) -> None:
        self.gpu_index = gpu_index
        self.interval_ms = interval_ms
        self.samples: list[tuple[float, int]] = []
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> 'VramSampler':
        self._proc = subprocess.Popen(
            ['nvidia-smi', '-i', str(self.gpu_index), '--query-gpu=memory.used', '--format=csv,noheader,nounits',
             f'-lms', str(self.interval_ms)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        return self

    def _read(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            line = line.strip()
            if line.isdigit():
                self.samples.append((time.perf_counter(), int(line)))

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def now(self) -> float:
        return time.perf_counter()

    def window(self, start: float, end: float | None = None) -> list[int]:
        end = end if end is not None else time.perf_counter()
        return [mb for (t, mb) in self.samples if start <= t <= end]

    def peak(self, start: float, end: float | None = None) -> int | None:
        values = self.window(start, end)
        return max(values) if values else None

    def median(self, start: float, end: float | None = None) -> float | None:
        values = self.window(start, end)
        return float(statistics.median(values)) if values else None


def vram_once(gpu_index: int = 0) -> int:
    out = subprocess.check_output(['nvidia-smi', '-i', str(gpu_index), '--query-gpu=memory.used',
                                   '--format=csv,noheader,nounits'], text=True)
    return int(out.strip().splitlines()[0])


def settle_vram(gpu_index: int = 0, seconds: float = 4.0) -> float:
    """Median card memory over a few seconds (for baselines)."""
    values = []
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        values.append(vram_once(gpu_index))
        time.sleep(0.25)
    return float(statistics.median(values))


# ----------------------------------------------------------------------------------------------------------------
# Single requests
# ----------------------------------------------------------------------------------------------------------------

async def ndjson_stream(client: httpx.AsyncClient, base: str, text: str, *, voice: str, language: str,
                        seed: int | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {'input': text, 'voice': voice, 'language': language}
    if seed is not None:
        payload['seed'] = seed
    started = time.perf_counter()
    first = None
    pcm_bytes = 0
    rate = None
    chunks = 0
    done: dict[str, Any] = {}
    async with client.stream('POST', f'{base}/api/v1/synthesize/stream', json=payload) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.strip():
                continue
            event = json.loads(line)
            kind = event.get('type')
            if kind == 'chunk':
                data = base64.b64decode(event.get('pcm16_b64') or '')
                if data and first is None:
                    first = time.perf_counter()
                pcm_bytes += len(data)
                rate = event.get('sample_rate') or rate
                chunks += 1
            elif kind == 'done':
                done = event.get('result') or {}
            elif kind == 'error':
                raise RuntimeError(f'stream error: {event}')
    ended = time.perf_counter()
    rate = int(rate or 24000)
    audio_s = pcm_bytes / 2 / rate
    total_ms = (ended - started) * 1000
    metrics = done.get('metrics') or {}
    return {
        'ttfa_ms': (first - started) * 1000 if first else None,
        'total_ms': total_ms,
        'audio_s': audio_s,
        'rtf': (total_ms / 1000) / audio_s if audio_s > 0 else None,
        'chunks': chunks,
        'sample_rate': rate,
        'server_ttfa_ms': metrics.get('ttfa_ms'),
        'server_queue_wait_ms': metrics.get('queue_wait_ms'),
        'status': done.get('status'),
    }


async def raw_pcm_stream(client: httpx.AsyncClient, base: str, text: str, *, voice: str, language: str,
                         seed: int | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {'input': text, 'voice': voice, 'language': language, 'stream': True}
    if seed is not None:
        payload['seed'] = seed
    started = time.perf_counter()
    first = None
    nbytes = 0
    async with client.stream('POST', f'{base}/v1/audio/speech', json=payload) as response:
        response.raise_for_status()
        ctype = response.headers.get('content-type', '')
        async for chunk in response.aiter_bytes():
            if chunk and first is None:
                first = time.perf_counter()
            nbytes += len(chunk)
    ended = time.perf_counter()
    audio_s = nbytes / 2 / 24000  # the raw stream contract is 24 kHz mono PCM16
    total_ms = (ended - started) * 1000
    return {
        'ttfa_ms': (first - started) * 1000 if first else None,
        'total_ms': total_ms,
        'audio_s': audio_s,
        'rtf': (total_ms / 1000) / audio_s if audio_s > 0 else None,
        'content_type': ctype,
    }


# ----------------------------------------------------------------------------------------------------------------
# Load patterns
# ----------------------------------------------------------------------------------------------------------------

def pct(values: list[float], q: float) -> float | None:
    clean = [v for v in values if v is not None]
    return float(np.percentile(clean, q)) if clean else None


def summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in results if not r.get('error')]
    ttfa = [r['ttfa_ms'] for r in ok if r.get('ttfa_ms') is not None]
    rtf = [r['rtf'] for r in ok if r.get('rtf') is not None]
    out = {
        'requests': len(results),
        'failed': len(results) - len(ok),
        'ttfa_ms_p50': pct(ttfa, 50), 'ttfa_ms_p95': pct(ttfa, 95),
        'ttfa_ms_min': min(ttfa) if ttfa else None, 'ttfa_ms_max': max(ttfa) if ttfa else None,
        'rtf_p50': pct(rtf, 50), 'rtf_p95': pct(rtf, 95),
        'audio_s_sum': sum(r.get('audio_s') or 0 for r in ok),
    }
    server = [r['server_ttfa_ms'] for r in ok if r.get('server_ttfa_ms') is not None]
    if server:
        out['server_ttfa_ms_p50'] = pct(server, 50)
        out['server_ttfa_ms_p95'] = pct(server, 95)
    errors = sorted({str(r['error'])[:200] for r in results if r.get('error')})
    if errors:
        out['errors'] = errors[:5]
    return out


async def _guard(coro: Awaitable[dict[str, Any]]) -> dict[str, Any]:
    try:
        return await coro
    except Exception as exc:  # noqa: BLE001 - a failed request is a data point, not a crash
        return {'error': f'{type(exc).__name__}: {exc}'}


async def bursts(fn: Callable[..., Awaitable[dict[str, Any]]], client: httpx.AsyncClient, base: str, *, n: int,
                 rounds: int, voice: str, language: str, gap_s: float = 0.3) -> dict[str, Any]:
    """`rounds` bursts of n requests sent at the same instant (worst case for TTFA); sequential when n == 1."""
    results: list[dict[str, Any]] = []
    offset = 0
    wall = 0.0
    for _ in range(rounds):
        texts = [TEXTS[(offset + i) % len(TEXTS)] for i in range(n)]
        offset += n
        t0 = time.perf_counter()
        batch = await asyncio.gather(*(_guard(fn(client, base, t, voice=voice, language=language)) for t in texts))
        wall += time.perf_counter() - t0
        results.extend(batch)
        if gap_s:
            await asyncio.sleep(gap_s)
    out = summarise(results)
    out['concurrency'] = n
    out['rounds'] = rounds
    out['burst_audio_s_per_wall_s'] = out['audio_s_sum'] / wall if wall else None
    return out


async def saturation(client: httpx.AsyncClient, base: str, *, clients: int, seconds: float, voice: str,
                     language: str) -> dict[str, Any]:
    """`clients` closed-loop NDJSON streams for `seconds`; throughput counts audio of completed requests."""
    deadline = time.perf_counter() + seconds
    results: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    last_end = [t0]

    async def worker(idx: int) -> None:
        k = idx
        while time.perf_counter() < deadline:
            r = await _guard(ndjson_stream(client, base, TEXTS[k % len(TEXTS)], voice=voice, language=language))
            results.append(r)
            last_end[0] = max(last_end[0], time.perf_counter())
            k += clients

    await asyncio.gather(*(worker(i) for i in range(clients)))
    wall = last_end[0] - t0
    out = summarise(results)
    out.update({'clients': clients, 'seconds': seconds, 'wall_s': wall,
                'audio_s_per_wall_s': out['audio_s_sum'] / wall if wall else None})
    return out


# ----------------------------------------------------------------------------------------------------------------
# Server spawning
# ----------------------------------------------------------------------------------------------------------------

def _parse_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def prepare_data_dir(work_dir: Path, backend: str, overrides: dict[str, Any], preset: str | None) -> Path:
    """A throwaway data dir: voice profiles + voices copied from the repo, runtime settings = repo settings
    + preset + overrides. Secrets, jobs and caches are never copied (a fresh admin account is created)."""
    data = work_dir / 'data'
    src = REPO / 'data'
    data.mkdir(parents=True, exist_ok=True)
    for name in ('voice_profiles.json', 'voice_presets_seeded.json'):
        if (src / name).exists():
            shutil.copy2(src / name, data / name)
    if not (data / 'voices').exists():
        shutil.copytree(src / 'voices', data / 'voices')
    settings: dict[str, Any] = {}
    if (src / 'runtime_settings.json').exists():
        settings = json.loads((src / 'runtime_settings.json').read_text(encoding='utf-8'))
    if preset:
        sys.path.insert(0, str(REPO / 'backend' / 'src'))
        from omnivoice_tts_server.presets import BUILTIN_PRESETS  # noqa: PLC0415

        settings.update(BUILTIN_PRESETS[preset]['values'])
    settings.update(overrides)
    settings['runtime_backend'] = backend
    (data / 'runtime_settings.json').write_text(json.dumps(settings, indent=2, sort_keys=True), encoding='utf-8')
    return data


def spawn_server(data_dir: Path, backend: str, port: int, log_path: Path, python: Path) -> subprocess.Popen:
    env = dict(os.environ)
    env.update({
        'OMNIVOICE_TTS_DATA_DIR': str(data_dir),
        'OMNIVOICE_TTS_PORT': str(port),
        'OMNIVOICE_TTS_HOST': '127.0.0.1',
        'OMNIVOICE_TTS_RUNTIME_BACKEND': backend,
        'OMNIVOICE_TTS_MODELS_ROOT_DIR': str(REPO / 'models'),
        'OMNIVOICE_TTS_ALLOW_MODEL_DOWNLOADS': 'false',
        # Same allocator / thread settings as start_server.bat.
        'PYTORCH_CUDA_ALLOC_CONF': 'garbage_collection_threshold:0.8,max_split_size_mb:256',
        'OPENBLAS_NUM_THREADS': '4',
        'PYTHONUNBUFFERED': '1',
        'PYTHONUTF8': '1',
    })
    log = open(log_path, 'w', encoding='utf-8')  # noqa: SIM115 - closed by the OS when the server exits
    return subprocess.Popen([str(python), '-u', '-m', 'omnivoice_tts_server.main'], cwd=str(REPO), env=env,
                            stdout=log, stderr=subprocess.STDOUT)


async def wait_ready(base: str, *, voice: str, language: str, timeout_s: float = 900.0) -> dict[str, Any]:
    """Health first, then streamed warmup requests until two succeed (loads + compiles the model)."""
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
        while True:
            try:
                if (await client.get(f'{base}/health')).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.perf_counter() - t0 > timeout_s:
                raise TimeoutError('server did not come up')
            await asyncio.sleep(1.0)
        health_s = time.perf_counter() - t0
        warm = []
        for i in range(4):
            warm.append(await ndjson_stream(client, base, TEXTS[i], voice=voice, language=language))
        # One small burst so batch shapes > 1 are compiled/captured too.
        await asyncio.gather(*(ndjson_stream(client, base, t, voice=voice, language=language) for t in TEXTS[:8]))
    return {'health_s': health_s, 'ready_s': time.perf_counter() - t0, 'warmup_ttfa_ms': [w['ttfa_ms'] for w in warm]}


# ----------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------

DEFAULT_ROUNDS = {1: 20, 4: 6, 8: 4, 16: 3, 32: 2, 48: 2, 64: 2}


async def run_suite(base: str, args: argparse.Namespace, sampler: VramSampler | None) -> dict[str, Any]:
    out: dict[str, Any] = {'ndjson': [], 'raw_pcm': []}
    async with httpx.AsyncClient(timeout=httpx.Timeout(900.0), limits=httpx.Limits(max_connections=256)) as client:
        for n in args.concurrency:
            t = sampler.now() if sampler else 0
            res = await bursts(ndjson_stream, client, base, n=n, rounds=args.rounds or DEFAULT_ROUNDS.get(n, 2),
                               voice=args.voice, language=args.language)
            if sampler:
                res['vram_peak_mb'] = sampler.peak(t)
            out['ndjson'].append(res)
            print(f'[ndjson] n={n}: ttfa p50 {res["ttfa_ms_p50"]:.0f} p95 {res["ttfa_ms_p95"]:.0f} ms, '
                  f'rtf p50 {res["rtf_p50"]:.3f}, failed {res["failed"]}', flush=True)
        for n in args.raw_concurrency:
            res = await bursts(raw_pcm_stream, client, base, n=n, rounds=args.rounds or DEFAULT_ROUNDS.get(n, 2),
                               voice=args.voice, language=args.language)
            out['raw_pcm'].append(res)
            print(f'[raw pcm] n={n}: ttfa p50 {res["ttfa_ms_p50"]:.0f} p95 {res["ttfa_ms_p95"]:.0f} ms, '
                  f'failed {res["failed"]}', flush=True)
        out['saturation'] = []
        for clients in args.saturation_clients:
            t = sampler.now() if sampler else 0
            res = await saturation(client, base, clients=clients, seconds=args.saturation_seconds, voice=args.voice,
                                   language=args.language)
            if sampler:
                res['vram_peak_mb'] = sampler.peak(t)
            out['saturation'].append(res)
            print(f'[saturation] clients={clients}: {res["audio_s_per_wall_s"]:.1f} audio s / wall s, '
                  f'ttfa p50 {res["ttfa_ms_p50"]:.0f} p95 {res["ttfa_ms_p95"]:.0f} ms, rtf p50 {res["rtf_p50"]:.3f}, '
                  f'failed {res["failed"]}', flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base-url', default=None, help='benchmark a running server (default: spawn one)')
    parser.add_argument('--spawn', choices=['omnivoice', 'ming', 'phonon'], help='engine of the spawned server')
    parser.add_argument('--preset', default=None, help='built-in preset whose values the spawned server starts with')
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                        help='runtime setting override for the spawned server (JSON value), repeatable')
    parser.add_argument('--work-dir', type=Path, default=None, help='throwaway dir for the spawned server')
    parser.add_argument('--port', type=int, default=8191)
    parser.add_argument('--python', type=Path, default=DEFAULT_PYTHON)
    parser.add_argument('--gpu-index', type=int, default=0)
    parser.add_argument('--voice', default='SonjaV2')
    parser.add_argument('--language', default='de')
    parser.add_argument('--concurrency', default='1,4,16,32', help='burst sizes for the NDJSON TTFA test')
    parser.add_argument('--raw-concurrency', default='1,16', help='burst sizes for the raw PCM TTFA test')
    parser.add_argument('--rounds', type=int, default=0, help='bursts per size (0 = 20/6/3/2 by size)')
    parser.add_argument('--saturation-clients', default='32', help='closed-loop client counts, comma separated')
    parser.add_argument('--saturation-seconds', type=float, default=45.0)
    parser.add_argument('--label', default='')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.concurrency = [int(x) for x in args.concurrency.split(',') if x.strip()]
    args.raw_concurrency = [int(x) for x in args.raw_concurrency.split(',') if x.strip()]
    args.saturation_clients = [int(x) for x in args.saturation_clients.split(',') if x.strip()]

    report: dict[str, Any] = {'label': args.label, 'voice': args.voice, 'language': args.language,
                              'started_at': time.strftime('%Y-%m-%d %H:%M:%S')}
    proc = None
    sampler = VramSampler(args.gpu_index).start()
    try:
        if args.base_url:
            base = args.base_url.rstrip('/')
        else:
            if not args.spawn or not args.work_dir:
                parser.error('--spawn and --work-dir are required without --base-url')
            overrides = {}
            for item in args.set:
                key, _, raw = item.partition('=')
                overrides[key.strip()] = _parse_value(raw.strip())
            report.update({'spawn': args.spawn, 'preset': args.preset, 'overrides': overrides})
            args.work_dir.mkdir(parents=True, exist_ok=True)
            data = prepare_data_dir(args.work_dir, args.spawn, overrides, args.preset)
            report['vram_baseline_mb'] = settle_vram(args.gpu_index)
            base = f'http://127.0.0.1:{args.port}'
            log_path = args.work_dir / f'server_{args.label or args.spawn}.log'
            t_spawn = time.perf_counter()
            proc = spawn_server(data, args.spawn, args.port, log_path, args.python)
            report['ready'] = asyncio.run(wait_ready(base, voice=args.voice, language=args.language))
            report['vram_load_peak_mb'] = sampler.peak(t_spawn)
            time.sleep(8.0)  # let trim-after-batch / allocator settle
            report['vram_idle_mb'] = settle_vram(args.gpu_index)
            report['vram_idle_delta_mb'] = report['vram_idle_mb'] - report['vram_baseline_mb']
        t_run = sampler.now()
        report['results'] = asyncio.run(run_suite(base, args, sampler))
        report['vram_run_peak_mb'] = sampler.peak(t_run)
        if 'vram_baseline_mb' in report and report['vram_run_peak_mb'] is not None:
            report['vram_run_peak_delta_mb'] = report['vram_run_peak_mb'] - report['vram_baseline_mb']
            for section in ('ndjson', 'saturation'):
                for row in report['results'].get(section, []):
                    if row.get('vram_peak_mb') is not None:
                        row['vram_peak_delta_mb'] = row['vram_peak_mb'] - report['vram_baseline_mb']
    finally:
        sampler.stop()
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
