"""Named settings presets so the operator can quickly switch between parameter sets.

A preset is just a stored bag of settings values (the same keys the PUT /settings
endpoint accepts). Applying one means loading its values into the settings draft and
saving through the normal validated PUT path -- so presets never bypass validation.

Stored as data/settings_presets.json: { name: {name, values, created_at} }.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

PRESETS_FILE = 'settings_presets.json'
_LOCK = threading.RLock()


def _path(data_dir: Path) -> Path:
    return data_dir / PRESETS_FILE


def load_presets(data_dir: Path) -> dict[str, dict[str, Any]]:
    with _LOCK:
        path = _path(data_dir)
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            return {}
    if not isinstance(data, dict):
        return {}
    presets: dict[str, dict[str, Any]] = {}
    for name, entry in data.items():
        if isinstance(entry, dict) and isinstance(entry.get('values'), dict):
            presets[str(name)] = {
                'name': str(entry.get('name') or name),
                'values': entry['values'],
                'created_at': entry.get('created_at'),
            }
    return presets


def list_presets(data_dir: Path) -> list[dict[str, Any]]:
    return sorted(load_presets(data_dir).values(), key=lambda item: (item.get('name') or '').lower())


def save_preset(data_dir: Path, name: str, values: dict[str, Any], *, now_iso: str) -> dict[str, Any]:
    clean = (name or '').strip()
    if not clean:
        raise ValueError('Preset name is required.')
    if not isinstance(values, dict):
        raise ValueError('Preset values must be an object.')
    with _LOCK:
        presets = load_presets(data_dir)
        existing = presets.get(clean) or {}
        entry = {
            'name': clean,
            'values': values,
            'created_at': existing.get('created_at') or now_iso,
        }
        presets[clean] = entry
        _write(data_dir, presets)
    return entry


def delete_preset(data_dir: Path, name: str) -> bool:
    with _LOCK:
        presets = load_presets(data_dir)
        if name not in presets:
            return False
        presets.pop(name, None)
        _write(data_dir, presets)
    return True


def _write(data_dir: Path, presets: dict[str, dict[str, Any]]) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    _path(data_dir).write_text(json.dumps(presets, indent=2, ensure_ascii=False), encoding='utf-8')
