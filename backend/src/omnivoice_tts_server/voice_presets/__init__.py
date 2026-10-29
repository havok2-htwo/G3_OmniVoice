"""Voice profiles shipped with the server (e.g. the default voice "Sonja").

Presets are copied into the voice store once per data dir, so they behave like any
uploaded clone profile afterwards. A marker file remembers what was seeded: a preset
the operator deleted stays deleted, and one whose name is already taken is skipped.

A preset may carry engine_seeds (best seeds for engines other than OmniVoice, whose seed
is `seed`). New entries reach existing installs too: each (voice, engine) is filled in
once into a preset voice that has no seed for that engine, and a second marker file
remembers it, so a seed the operator changed or cleared is never overwritten or refilled
(a later shipped value for the same engine is not applied either).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from ..domain.state import InMemoryStore, VoiceProfileRecord, clean_engine_seeds, utcnow

logger = logging.getLogger(__name__)

PRESETS_DIR = Path(__file__).resolve().parent
PRESETS_FILE = PRESETS_DIR / 'presets.json'
SEEDED_MARKER_FILE = 'voice_presets_seeded.json'
ENGINE_SEEDS_MARKER_FILE = 'voice_presets_engine_seeds.json'


def _load_seeded_ids(marker_path: Path) -> set[str]:
    try:
        payload = json.loads(marker_path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return set()
    except Exception:
        logger.warning('voice presets: unreadable marker %s, treating as empty', marker_path)
        return set()
    return {str(item) for item in payload} if isinstance(payload, list) else set()


def _load_applied_engine_seeds(marker_path: Path) -> dict[str, dict[str, int]]:
    try:
        payload = json.loads(marker_path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return {}
    except Exception:
        logger.warning('voice presets: unreadable marker %s, treating as empty', marker_path)
        return {}
    if not isinstance(payload, dict):
        return {}
    return {str(voice_id): clean_engine_seeds(seeds) for voice_id, seeds in payload.items()}


def _fill_engine_seeds(store: InMemoryStore, presets: list[dict], data_dir: Path) -> list[str]:
    """Fill each preset voice's missing engine seeds from presets.json, each engine once; return the changed ids."""
    marker_path = data_dir / ENGINE_SEEDS_MARKER_FILE
    applied = _load_applied_engine_seeds(marker_path)
    changed: list[str] = []
    marker_changed = False
    for preset in presets:
        voice_id = preset['voice_id']
        profile = store.voice_profiles.get(voice_id)
        if profile is None:
            continue
        done = applied.setdefault(voice_id, {})
        for engine, seed in clean_engine_seeds(preset.get('engine_seeds')).items():
            # Handled before: the slot is the operator's now, even when they cleared it.
            if engine in done:
                continue
            done[engine] = seed
            marker_changed = True
            if engine not in profile.engine_seeds:
                profile.engine_seeds[engine] = seed
                if voice_id not in changed:
                    changed.append(voice_id)
    if changed:
        store.save_voices(data_dir)
        logger.info('voice presets: engine seeds filled for %s', ', '.join(changed))
    if marker_changed:
        data_dir.mkdir(parents=True, exist_ok=True)
        marker_path.write_text(json.dumps({k: v for k, v in sorted(applied.items()) if v}, indent=2), encoding='utf-8')
    return changed


def seed_voice_presets(store: InMemoryStore, data_dir: Path) -> list[str]:
    """Add shipped voice presets missing from the store; return the seeded voice ids."""
    presets = json.loads(PRESETS_FILE.read_text(encoding='utf-8'))
    marker_path = data_dir / SEEDED_MARKER_FILE
    seeded_ids = _load_seeded_ids(marker_path)
    taken_names = {str(profile.name or '').casefold() for profile in store.voice_profiles.values()}
    added: list[str] = []
    marker_changed = False
    for preset in presets:
        voice_id = preset['voice_id']
        if voice_id in seeded_ids:
            continue
        seeded_ids.add(voice_id)
        marker_changed = True
        if voice_id in store.voice_profiles or preset['name'].casefold() in taken_names:
            continue
        store.voice_profiles[voice_id] = VoiceProfileRecord(
            voice_id=voice_id,
            name=preset['name'],
            # 'custom' marks a clonable Base profile throughout the API and admin UI.
            source='custom',
            created_at=utcnow(),
            audio_bytes=(PRESETS_DIR / preset['audio_file']).read_bytes(),
            content_type='audio/wav',
            filename=preset['audio_file'],
            ref_text=preset['ref_text'],
            consent=True,
            seed=preset.get('seed'),
        )
        taken_names.add(preset['name'].casefold())
        added.append(voice_id)
    if added:
        store.save_voices(data_dir)
        logger.info('voice presets: seeded %s', ', '.join(added))
    if marker_changed:
        data_dir.mkdir(parents=True, exist_ok=True)
        marker_path.write_text(json.dumps(sorted(seeded_ids), indent=2), encoding='utf-8')
    _fill_engine_seeds(store, presets, data_dir)
    return added
