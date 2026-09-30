"""Voice profiles shipped with the server (e.g. the default voice "Sonja").

Presets are copied into the voice store once per data dir, so they behave like any
uploaded clone profile afterwards. A marker file remembers what was seeded: a preset
the operator deleted stays deleted, and one whose name is already taken is skipped.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from ..domain.state import InMemoryStore, VoiceProfileRecord, utcnow

logger = logging.getLogger(__name__)

PRESETS_DIR = Path(__file__).resolve().parent
PRESETS_FILE = PRESETS_DIR / 'presets.json'
SEEDED_MARKER_FILE = 'voice_presets_seeded.json'


def _load_seeded_ids(marker_path: Path) -> set[str]:
    try:
        payload = json.loads(marker_path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return set()
    except Exception:
        logger.warning('voice presets: unreadable marker %s, treating as empty', marker_path)
        return set()
    return {str(item) for item in payload} if isinstance(payload, list) else set()


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
        )
        taken_names.add(preset['name'].casefold())
        added.append(voice_id)
    if added:
        store.save_voices(data_dir)
        logger.info('voice presets: seeded %s', ', '.join(added))
    if marker_changed:
        data_dir.mkdir(parents=True, exist_ok=True)
        marker_path.write_text(json.dumps(sorted(seeded_ids), indent=2), encoding='utf-8')
    return added
