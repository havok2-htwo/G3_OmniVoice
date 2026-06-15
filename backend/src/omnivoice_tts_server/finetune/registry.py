"""Registry of promoted custom finetune checkpoints.

A promoted checkpoint is a full HF-format OmniVoice dir under models_root_dir/<dirname>
(written by the trainer's save_pretrained, then copied on promote). The runtime addresses
it as model_id 'local/<dirname>' (see runtime_v2._resolve_model_source). This registry is
the source of truth that re-seeds settings.supported_models at startup so the model appears
in the model list / selector and the existing model-ops endpoints can load it.

Stored as data/finetune/checkpoints.json: { checkpoint_id: {...} }.
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any
from uuid import uuid4

_LOCK = threading.RLock()
_SLUG = re.compile(r'[^A-Za-z0-9_-]+')


def _path(data_dir: Path) -> Path:
    return data_dir / 'finetune' / 'checkpoints.json'


def slug(name: str) -> str:
    cleaned = _SLUG.sub('-', (name or '').strip()).strip('-')
    return cleaned or 'model'


def model_id_for(dirname: str) -> str:
    return f'local/{dirname}'


def load_checkpoints(data_dir: Path) -> dict[str, dict[str, Any]]:
    with _LOCK:
        path = _path(data_dir)
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            return {}
    return data if isinstance(data, dict) else {}


def list_checkpoints(data_dir: Path) -> list[dict[str, Any]]:
    return sorted(load_checkpoints(data_dir).values(), key=lambda item: item.get('created_at') or '', reverse=True)


def get_checkpoint(data_dir: Path, checkpoint_id: str) -> dict[str, Any] | None:
    return load_checkpoints(data_dir).get(checkpoint_id)


def _write(data_dir: Path, checkpoints: dict[str, dict[str, Any]]) -> None:
    path = _path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(checkpoints, indent=2, ensure_ascii=False), encoding='utf-8')


def register_checkpoint(
    data_dir: Path, *, name: str, dirname: str, base_model: str, steps: int, run_id: str | None, now_iso: str,
) -> dict[str, Any]:
    with _LOCK:
        checkpoints = load_checkpoints(data_dir)
        checkpoint_id = f'ckpt_{uuid4().hex[:12]}'
        entry = {
            'checkpoint_id': checkpoint_id,
            'name': name,
            'dirname': dirname,
            'model_id': model_id_for(dirname),
            'base_model': base_model,
            'steps': steps,
            'run_id': run_id,
            'created_at': now_iso,
        }
        checkpoints[checkpoint_id] = entry
        _write(data_dir, checkpoints)
    return entry


def delete_checkpoint(data_dir: Path, checkpoint_id: str) -> dict[str, Any] | None:
    with _LOCK:
        checkpoints = load_checkpoints(data_dir)
        entry = checkpoints.pop(checkpoint_id, None)
        if entry is not None:
            _write(data_dir, checkpoints)
    return entry


def seed_supported_models(settings: Any) -> None:
    """Extend settings.supported_models with every registered custom model id (idempotent)."""
    try:
        registered = load_checkpoints(settings.data_dir)
    except Exception:
        return
    for entry in registered.values():
        model_id = entry.get('model_id')
        if model_id and model_id not in settings.supported_models:
            settings.supported_models.append(model_id)
