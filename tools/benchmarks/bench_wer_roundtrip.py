"""German WER round trip against a running G3 TTS server (any engine) and G3_Whisper.

Each fixed German sentence (half of them with numbers, dates, times, units or abbreviations) is synthesized
through the public API (POST /v1/audio/speech, non-streaming WAV) for every voice, transcribed by G3_Whisper
(POST /transcribe/, engine=local) and scored with the server's own WER function
(WerBenchmarkService._wer: German orthography folding, number words -> digits, 2-letter tolerance per word),
so the numbers match the admin WER benchmark. The sentence list is fixed (no LLM), so runs on different
engines see exactly the same input.

  python bench_wer_roundtrip.py --base-url http://127.0.0.1:8191 --voices SonjaV2,SonjaV3,Tom \
      --modes off,deterministic --out wer_ming.json --wav-dir wavs_ming

--sentences-file reads a corpus instead of the built-in 20 sentences, for example wer_sentences_de.tsv (131 fixed
sentences in the groups plain, numeric, long). --seed sends a fixed seed; without it the voice's own seed applies.

--modes picks the request's text_optimization: 'off' = the raw sentence goes to the engine, 'deterministic'
= the server's rule-based spoken-text optimization (numbers, dates, units -> words) runs first. Run with the
backend env python (it imports omnivoice_tts_server for the scorer).
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import sys
import time
import wave
from pathlib import Path
from typing import Any

import httpx

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'backend' / 'src'))
from omnivoice_tts_server.services_v2 import WerBenchmarkService  # noqa: E402

# (sentence, has numbers/abbreviations)
SENTENCES: list[tuple[str, bool]] = [
    ('Am 3. Oktober 1990 wurde Deutschland wiedervereinigt.', True),
    ('Die Rechnung über 1.250 Euro ist bis zum 15. November fällig.', True),
    ('Der Termin bei Dr. Müller ist am Dienstag um 14:30 Uhr.', True),
    ('Bitte rufen Sie uns unter der Nummer 030 1234567 an.', True),
    ('Die Temperatur steigt heute auf 28 Grad Celsius.', True),
    ('Der ICE 578 nach München hat etwa 20 Minuten Verspätung.', True),
    ('Die GmbH beschäftigt rund 350 Mitarbeiterinnen und Mitarbeiter.', True),
    ('Wir treffen uns z. B. am Hauptbahnhof oder in der Nähe vom Rathaus.', True),
    ('Das Paket wiegt 2,5 Kilogramm und kostet 7,99 Euro Porto.', True),
    ('Die Bundesstraße B 27 ist zwischen Stuttgart und Tübingen gesperrt.', True),
    ('Laut Statistik nutzen 87 Prozent der Haushalte täglich das Internet.', True),
    ('Herr Schmidt wohnt in der Goethestraße 12 im dritten Stock.', True),
    ('Die Sitzung beginnt pünktlich um neun Uhr im großen Konferenzraum.', False),
    ('Unsere Kinder spielen am liebsten draußen im Garten Fußball.', False),
    ('Nach dem langen Winter freuen sich alle auf den Frühling.', False),
    ('Der Kühlschrank ist leer, wir müssen dringend einkaufen gehen.', False),
    ('Die Polizei bittet die Bevölkerung um Hinweise zu dem Vorfall.', False),
    ('In Hamburg regnet es heute, in Berlin bleibt es überwiegend trocken.', False),
    ('Die EU hat neue Regeln für den Datenschutz beschlossen.', True),
    ('Bitte schalten Sie Ihre Mobiltelefone während der Vorstellung aus.', False),
]


def wav_seconds(data: bytes) -> float:
    with wave.open(io.BytesIO(data)) as handle:
        return handle.getnframes() / handle.getframerate()


def load_sentences(path: Path | None) -> list[tuple[str, str]]:
    """(group, sentence). Built-in list: group numeric or plain. File: one group<TAB>sentence per line, # comments."""
    if path is None:
        return [('numeric' if numeric else 'plain', text) for text, numeric in SENTENCES]
    lines = path.read_text(encoding='utf-8').splitlines()
    rows = [line.split('	', 1) for line in lines if line.strip() and not line.startswith('#')]
    return [(group.strip(), text.strip()) for group, text in rows]


async def synthesize(client: httpx.AsyncClient, base: str, text: str, voice: str, mode: str,
                     seed: int | None) -> tuple[bytes, float]:
    t0 = time.perf_counter()
    body: dict[str, Any] = {'input': text, 'voice': voice, 'language': 'de', 'response_format': 'wav',
                            'text_optimization': mode}
    if seed is not None:
        body['seed'] = seed
    response = await client.post(f'{base}/v1/audio/speech', json=body)
    response.raise_for_status()
    return response.content, (time.perf_counter() - t0) * 1000


async def transcribe(client: httpx.AsyncClient, whisper: str, wav: bytes) -> str:
    response = await client.post(f'{whisper}/transcribe/', files={'file': ('tts.wav', wav, 'audio/wav')},
                                 data={'engine': 'local', 'voice_ident': 'false'})
    response.raise_for_status()
    body = response.json()
    return str(body.get('transcription') or body.get('text') or '')


def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [i for i in items if i.get('wer') is not None]
    words = sum(i['word_count'] for i in ok)
    errors = sum(i['word_errors'] for i in ok)
    return {
        'items': len(items), 'scored': len(ok),
        'wer_corpus': errors / words if words else None,  # sum of errors / sum of reference words
        'wer_mean': sum(i['wer'] for i in ok) / len(ok) if ok else None,
        'exact': sum(1 for i in ok if i['word_errors'] == 0),
        'deletions': sum(i['deletions'] for i in ok),
        'audio_s_sum': sum(i.get('audio_s') or 0 for i in ok),
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    base = args.base_url.rstrip('/')
    whisper = args.whisper_url.rstrip('/')
    sem_tts = asyncio.Semaphore(args.concurrency)
    sem_asr = asyncio.Semaphore(args.asr_concurrency)
    items: list[dict[str, Any]] = []
    sentences = load_sentences(args.sentences_file)
    groups = sorted({group for group, _ in sentences})
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:

        async def one(voice: str, mode: str, idx: int, text: str, group: str) -> None:
            item: dict[str, Any] = {'voice': voice, 'mode': mode, 'index': idx, 'text': text, 'group': group}
            try:
                async with sem_tts:
                    wav, synth_ms = await synthesize(client, base, text, voice, mode, args.seed)
                item['synthesis_ms'] = synth_ms
                item['audio_s'] = wav_seconds(wav)
                if args.wav_dir:
                    (args.wav_dir / f'{voice}_{mode}_{idx:02d}.wav').write_bytes(wav)
                async with sem_asr:
                    transcript = await transcribe(client, whisper, wav)
                score = WerBenchmarkService._wer(text, transcript, args.tolerance)
                item.update({'transcript': transcript, 'wer': score['wer'], 'word_count': score['word_count'],
                             'word_errors': score['word_errors'], 'substitutions': score['substitutions'],
                             'insertions': score['insertions'], 'deletions': score['deletions']})
            except Exception as exc:  # noqa: BLE001
                item['error'] = f'{type(exc).__name__}: {exc}'
            items.append(item)

        await asyncio.gather(*(one(v, m, i, t, g) for v in args.voices for m in args.modes
                               for i, (g, t) in enumerate(sentences)))
    items.sort(key=lambda i: (i['voice'], i['mode'], i['index']))
    summary: dict[str, Any] = {}
    for mode in args.modes:
        rows = [i for i in items if i['mode'] == mode]
        summary[mode] = {
            'all': aggregate(rows),
            'groups': {g: aggregate([i for i in rows if i['group'] == g]) for g in groups},
            'per_voice': {v: aggregate([i for i in rows if i['voice'] == v]) for v in args.voices},
        }
    return {'base_url': base, 'whisper_url': whisper, 'voices': args.voices, 'modes': args.modes,
            'tolerance_letters_per_word': args.tolerance, 'summary': summary, 'items': items}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base-url', default='http://127.0.0.1:8191')
    parser.add_argument('--whisper-url', default='http://127.0.0.1:7861')
    parser.add_argument('--voices', default='SonjaV2,SonjaV3,Tom')
    parser.add_argument('--modes', default='off,deterministic')
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--asr-concurrency', type=int, default=4)
    parser.add_argument('--tolerance', type=int, default=2, help='letters per word (admin WER default 2)')
    parser.add_argument('--sentences-file', type=Path, default=None, help='group<TAB>sentence per line')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--wav-dir', type=Path, default=None)
    parser.add_argument('--label', default='')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.voices = [v.strip() for v in args.voices.split(',') if v.strip()]
    args.modes = [m.strip() for m in args.modes.split(',') if m.strip()]
    if args.wav_dir:
        args.wav_dir.mkdir(parents=True, exist_ok=True)
    report = asyncio.run(run(args))
    report['label'] = args.label
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    for mode, block in report['summary'].items():
        a = block['all']
        print(f'[{args.label} {mode}] WER corpus {a["wer_corpus"]:.3%}, mean {a["wer_mean"]:.3%}, exact '
              f'{a["exact"]}/{a["scored"]}; '
              + ', '.join(f'{g} {s["wer_corpus"]:.3%}' for g, s in block['groups'].items()) + '; per voice '
              + ', '.join(f'{v} {s["wer_corpus"]:.3%}' for v, s in block['per_voice'].items()))
    failed = [i for i in report['items'] if i.get('error')]
    if failed:
        print(f'{len(failed)} failed, first: {failed[0]["error"]}')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
