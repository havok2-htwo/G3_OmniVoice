"""Named settings presets: read-only built-in GPU tiers plus presets the operator saves.

A preset is a bag of settings values (keys of PUT /api/admin/settings). Applying one
loads its values into the settings draft; the operator reviews them and saves through
the normal validated PUT, so a preset never bypasses validation.

Built-in tiers (one name per target machine, identical across the G3 services):
- "32GB VRAM"  RTX 5090, every G3 service on one card, whole-card peak <= 30 GB.
- "24GB VRAM"  RTX 4090, every G3 service on one card, whole-card peak <= 23 GB.
- "24+8 VRAM"  3090 (24 GB) + 2070S (8 GB) box, OmniVoice on the 24 GB card.
The number in front names the card the service runs on. OmniVoice has no "8+24 VRAM"
tier: it belongs on the 24 GB card of the two-card box. A tier with a target card size
picks preferred_device by matching the detected CUDA cards' total memory (machines
number their cards differently), see resolve_presets. Single-card tiers never set it.

Saved presets live in <data_dir>/settings_presets.json ({name: {name, values,
created_at}}); they cannot reuse a built-in name, and their values are filtered to
PRESET_KEYS and normalised exactly like PUT /api/admin/settings does.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .config import RUNTIME_SETTINGS_FIELDS
from .domain.models import ServerSettingsUpdateRequest
from .runtime_v2 import (
    normalize_attention_implementation,
    normalize_runtime_device,
    normalize_torch_dtype,
    query_runtime_devices,
)

logger = logging.getLogger('omnivoice_tts_server.presets')

PRESETS_FILE = 'settings_presets.json'
PRESET_NAME_MAX_LENGTH = 64
_LOCK = threading.RLock()

# Keys a preset may set: device, precision, compile, batching and VRAM behaviour.
# Secrets, URLs, model/data paths, the default voice and the UI theme stay out, so
# loading a preset never touches anything machine- or account-specific.
PRESET_KEYS = (
    'preferred_device',
    'attention_implementation',
    'enable_flashinfer',
    'torch_dtype',
    'compile_model',
    'compile_cudagraphs',
    'cudagraph_skip_dynamic_graphs',
    'cuda_memory_trim_after_batch',
    'warmup_on_startup',
    'max_parallel_requests',
    'max_batch_size',
    'batch_wait_ms',
    'vram_budget_mb',
    'max_input_chars',
    'num_step',
    'audio_chunk_duration',
    'audio_chunk_threshold',
)
# Every preset key is a persisted runtime setting and accepted by the settings PUT
# under the same name (checked here so a rename cannot silently break presets).
assert set(PRESET_KEYS) <= RUNTIME_SETTINGS_FIELDS
assert set(PRESET_KEYS) <= set(ServerSettingsUpdateRequest.model_fields)

# Tolerance when matching a tier's card size to a detected card (a "24 GB" card
# reports ~23.6-24.0 GiB of total memory).
CARD_MATCH_TOLERANCE_MB = 2048

# ---------------------------------------------------------------------------
# BUILT-IN TIER VALUES
# PLACEHOLDER: every tier currently carries the service's code defaults (config.Settings)
# for the preset keys. The real per-tier values come from VRAM/throughput measurements
# on each target machine in a later step and replace these. Do not tune them by guess.
# preferred_device is never listed here: two-card tiers get it from card matching.
# ---------------------------------------------------------------------------
_CURRENT_DEFAULTS: dict[str, Any] = {
    'attention_implementation': 'sdpa',
    'enable_flashinfer': False,
    'torch_dtype': 'bfloat16',
    'compile_model': True,
    'compile_cudagraphs': False,
    'cudagraph_skip_dynamic_graphs': True,
    'cuda_memory_trim_after_batch': True,
    'warmup_on_startup': True,
    'max_parallel_requests': 8,
    'max_batch_size': 8,
    'batch_wait_ms': 0,
    'vram_budget_mb': 7000,
    'max_input_chars': 25000,
    'num_step': 20,
    'audio_chunk_duration': 30.1,
    'audio_chunk_threshold': 30.0,
}
BUILTIN_PRESETS: dict[str, dict[str, Any]] = {
    '32GB VRAM': {
        'description': 'RTX 5090, alle G3-Dienste auf einer Karte (Spitze der ganzen Karte <= 30 GB).',
        'target_vram_gb': None,
        'values': dict(_CURRENT_DEFAULTS),
    },
    '24GB VRAM': {
        'description': 'RTX 4090, alle G3-Dienste auf einer Karte (Spitze der ganzen Karte <= 23 GB).',
        'target_vram_gb': None,
        'values': dict(_CURRENT_DEFAULTS),
    },
    '24+8 VRAM': {
        'description': '3090 (24 GB) + 2070S (8 GB), OmniVoice auf der 24-GB-Karte.',
        'target_vram_gb': 24,
        'values': dict(_CURRENT_DEFAULTS),
    },
}


def _path(data_dir: Path) -> Path:
    return data_dir / PRESETS_FILE


def clean_preset_name(name: str) -> str:
    clean = ' '.join(str(name or '').split())
    if not clean:
        raise ValueError('Preset name is required.')
    if len(clean) > PRESET_NAME_MAX_LENGTH:
        raise ValueError(f'Preset names have at most {PRESET_NAME_MAX_LENGTH} characters.')
    return clean


def _normalize_value(key: str, value: Any) -> Any:
    """One preset value, validated and normalised the way PUT /api/admin/settings does."""
    try:
        parsed = getattr(ServerSettingsUpdateRequest.model_validate({key: value}), key)
    except ValidationError as exc:
        first = exc.errors()[0] if exc.errors() else {}
        raise ValueError(f'{key}: {first.get("msg") or "invalid value"}') from None
    if key == 'preferred_device':
        return normalize_runtime_device(parsed)
    if key == 'attention_implementation':
        return normalize_attention_implementation(parsed)
    if key == 'torch_dtype':
        return normalize_torch_dtype(parsed)
    return parsed


def preset_values(values: dict[str, Any]) -> dict[str, Any]:
    """Only PRESET_KEYS (unknown keys and nulls are dropped), each normalised.

    Raises ValueError naming the key when a value would be rejected by the settings PUT.
    """
    if not isinstance(values, dict):
        raise ValueError('Preset values must be an object.')
    cleaned: dict[str, Any] = {}
    for key in PRESET_KEYS:
        if key in values and values[key] is not None:
            try:
                cleaned[key] = _normalize_value(key, values[key])
            except ValueError as exc:
                message = str(exc)
                raise ValueError(message if message.startswith(f'{key}:') else f'{key}: {message}') from None
    return cleaned


def _lenient_preset_values(values: dict[str, Any]) -> dict[str, Any]:
    """Like preset_values, but drops invalid values instead of failing (stored file)."""
    cleaned: dict[str, Any] = {}
    for key in PRESET_KEYS:
        if key in values and values[key] is not None:
            try:
                cleaned[key] = _normalize_value(key, values[key])
            except ValueError:
                continue
    return cleaned


def load_presets(data_dir: Path) -> dict[str, dict[str, Any]]:
    """Saved (non built-in) presets from the data dir; built-in names in the file are ignored."""
    with _LOCK:
        path = _path(data_dir)
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            logger.warning('could not load settings presets from %s: %s', path, exc)
            return {}
    if not isinstance(data, dict):
        return {}
    presets: dict[str, dict[str, Any]] = {}
    for name, entry in data.items():
        if not (isinstance(entry, dict) and isinstance(entry.get('values'), dict)):
            continue
        clean = ' '.join(str(name).split())
        if not clean or clean in BUILTIN_PRESETS:
            continue
        presets[clean] = {
            'name': clean,
            'values': _lenient_preset_values(entry['values']),
            'created_at': entry.get('created_at'),
        }
    return presets


def _detected_cards_mb() -> list[tuple[int, int]]:
    """(cuda index, total memory MiB) of every detected CUDA card; empty without CUDA."""
    cards: list[tuple[int, int]] = []
    try:
        devices = query_runtime_devices('cpu')
    except Exception:
        return cards
    for device in devices:
        if device.get('kind') == 'cuda' and device.get('available') and device.get('index') is not None:
            cards.append((int(device['index']), int(device.get('memory_total_mb') or 0)))
    return sorted(cards)


def _match_card(target_gb: int | None, cards: list[tuple[int, int]]) -> tuple[str | None, str | None]:
    """preferred_device for a tier's card size on this machine, plus a note when no card matches."""
    if not target_gb:
        return None, None
    target_mb = target_gb * 1024
    for index, total_mb in cards:
        if total_mb and abs(total_mb - target_mb) <= CARD_MATCH_TOLERANCE_MB:
            return f'cuda:{index}', None
    detected = ', '.join(f'cuda:{index} {total_mb / 1024:.0f} GB' for index, total_mb in cards) or 'keine'
    return None, f'Keine {target_gb}-GB-Karte gefunden (erkannt: {detected}); die GPU-Auswahl bleibt unveraendert.'


def list_presets(data_dir: Path) -> list[dict[str, Any]]:
    """Built-in tiers first (tier order), then saved presets by name."""
    cards = _detected_cards_mb()
    result: list[dict[str, Any]] = []
    for name, preset in BUILTIN_PRESETS.items():
        values = dict(preset['values'])
        device, note = _match_card(preset.get('target_vram_gb'), cards)
        if device:
            values['preferred_device'] = device
        result.append(
            {
                'name': name,
                'built_in': True,
                'description': preset.get('description', ''),
                'target_vram_gb': preset.get('target_vram_gb'),
                'note': note,
                'values': values,
                'created_at': None,
            }
        )
    saved = load_presets(data_dir)
    for name in sorted(saved, key=str.lower):
        result.append({**saved[name], 'built_in': False, 'description': '', 'target_vram_gb': None, 'note': None})
    return result


def save_preset(data_dir: Path, name: str, values: dict[str, Any], *, now_iso: str) -> dict[str, Any]:
    clean = clean_preset_name(name)
    if clean in BUILTIN_PRESETS:
        raise ValueError(f'"{clean}" is a built-in preset; save under another name.')
    cleaned = preset_values(values)
    if not cleaned:
        raise ValueError('The preset contains no preset settings.')
    with _LOCK:
        presets = load_presets(data_dir)
        entry = {
            'name': clean,
            'values': cleaned,
            'created_at': (presets.get(clean) or {}).get('created_at') or now_iso,
        }
        presets[clean] = entry
        _write(data_dir, presets)
    return {**entry, 'built_in': False, 'description': '', 'target_vram_gb': None, 'note': None}


def delete_preset(data_dir: Path, name: str) -> bool:
    clean = ' '.join(str(name or '').split())
    if clean in BUILTIN_PRESETS:
        raise ValueError(f'"{clean}" is a built-in preset and cannot be deleted.')
    with _LOCK:
        presets = load_presets(data_dir)
        if clean not in presets:
            return False
        presets.pop(clean)
        _write(data_dir, presets)
    return True


def _write(data_dir: Path, presets: dict[str, dict[str, Any]]) -> None:
    """Atomic write: a crash mid-write never leaves a truncated presets file behind."""
    data_dir.mkdir(parents=True, exist_ok=True)
    target = _path(data_dir)
    fd, tmp_name = tempfile.mkstemp(prefix=f'.{PRESETS_FILE}.', suffix='.tmp', dir=data_dir)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            handle.write(json.dumps(presets, indent=2, ensure_ascii=False))
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
