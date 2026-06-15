"""Turn data/finetune/train/<voice>/ into the raw JSONL manifests the omnivoice
preprocess step (extract_audio_tokens) consumes.

Output per run under data/finetune/datasets/<run_id>/:
    train/raw.jsonl   dev/raw.jsonl      # {"id","audio_path","text","language_id","instruct"?}
After token extraction these become train/data.lst + dev/data.lst, referenced by
data_config.json (written by write_data_config once the .lst files exist).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import storage


def dataset_dir(data_dir: Path, run_id: str) -> Path:
    return data_dir / 'finetune' / 'datasets' / run_id


def _clip_records(data_dir: Path, voices: list[str] | None) -> list[dict[str, Any]]:
    root = storage.train_root(data_dir)
    records: list[dict[str, Any]] = []
    voice_dirs = (
        [root / storage.sanitize_voice_dir(v) for v in voices]
        if voices else storage.dataset_voice_dirs(data_dir)
    )
    for vdir in voice_dirs:
        if not vdir.is_dir():
            continue
        for wav_path in sorted(vdir.glob('clip_*.wav')):
            txt_path = wav_path.with_suffix('.txt')
            if not txt_path.exists():
                continue
            text = txt_path.read_text(encoding='utf-8').strip()
            if not text:
                continue
            language_id = 'de'
            instruct = None
            json_path = wav_path.with_suffix('.json')
            if json_path.exists():
                try:
                    meta = json.loads(json_path.read_text(encoding='utf-8'))
                    language_id = meta.get('language_id') or language_id
                    instruct = meta.get('instruct')
                except Exception:
                    pass
            record = {
                'id': f'{vdir.name}__{wav_path.stem}',
                'audio_path': str(wav_path.resolve()),
                'text': text,
                'language_id': language_id,
            }
            if instruct:
                record['instruct'] = instruct
            records.append(record)
    return records


def build_manifests(data_dir: Path, run_id: str, *, voices: list[str] | None, dev_fraction: float) -> dict[str, Any]:
    records = _clip_records(data_dir, voices)
    if not records:
        raise ValueError('No clips found under the training folder for the selected voices.')

    ds_dir = dataset_dir(data_dir, run_id)
    train_dir = ds_dir / 'train'
    dev_dir = ds_dir / 'dev'
    train_dir.mkdir(parents=True, exist_ok=True)
    dev_dir.mkdir(parents=True, exist_ok=True)

    # Deterministic dev hold-out: every Nth record. Keep at least 1 dev sample if a
    # positive fraction was requested and there are >=2 records, but never empty the train set.
    dev_every = 0
    if dev_fraction > 0 and len(records) >= 2:
        dev_every = max(2, round(1.0 / min(max(dev_fraction, 1e-6), 0.5)))

    train_records: list[dict[str, Any]] = []
    dev_records: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if dev_every and index % dev_every == 0:
            dev_records.append(record)
        else:
            train_records.append(record)
    if not train_records:  # safety: never starve training
        train_records = dev_records
        dev_records = []

    _write_jsonl(train_dir / 'raw.jsonl', train_records)
    if dev_records:
        _write_jsonl(dev_dir / 'raw.jsonl', dev_records)

    return {
        'dataset_dir': str(ds_dir),
        'train_jsonl': str(train_dir / 'raw.jsonl'),
        'dev_jsonl': str(dev_dir / 'raw.jsonl') if dev_records else None,
        'train_count': len(train_records),
        'dev_count': len(dev_records),
        'voices': sorted({rec['id'].split('__', 1)[0] for rec in records}),
    }


def write_data_config(data_dir: Path, run_id: str) -> str:
    """Write data_config.json referencing the data.lst files produced by token extraction."""
    ds_dir = dataset_dir(data_dir, run_id)
    train_lst = ds_dir / 'train' / 'data.lst'
    dev_lst = ds_dir / 'dev' / 'data.lst'
    if not train_lst.exists():
        raise FileNotFoundError(f'Missing {train_lst}; run the preprocess (token extraction) step first.')
    config: dict[str, Any] = {'train': [{'language_id': 'de', 'manifest_path': [str(train_lst.resolve())]}]}
    if dev_lst.exists():
        config['dev'] = [{'language_id': 'de', 'manifest_path': [str(dev_lst.resolve())]}]
    path = ds_dir / 'data_config.json'
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
    return str(path)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
