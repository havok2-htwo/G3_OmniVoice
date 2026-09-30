from __future__ import annotations

import io
import json
from datetime import datetime, timezone

import soundfile as sf

from omnivoice_tts_server.domain.state import InMemoryStore, VoiceProfileRecord
from omnivoice_tts_server.voice_presets import SEEDED_MARKER_FILE, seed_voice_presets


def test_sonja_preset_is_seeded_and_persisted(tmp_path) -> None:
    store = InMemoryStore(max_queue_size=4)
    assert seed_voice_presets(store, tmp_path) == ['voice_preset_sonja']

    profile = store.voice_profiles['voice_preset_sonja']
    assert profile.name == 'Sonja'
    assert profile.source == 'custom'
    assert profile.consent is True
    assert profile.ref_text.startswith('Franz jagt im komplett verwahrlosten Taxi')
    info = sf.info(io.BytesIO(profile.audio_bytes))
    assert (info.samplerate, info.channels) == (24_000, 1)

    reloaded = InMemoryStore(max_queue_size=4)
    reloaded.load_voices(tmp_path)
    assert reloaded.voice_profiles['voice_preset_sonja'].audio_bytes == profile.audio_bytes
    assert json.loads((tmp_path / SEEDED_MARKER_FILE).read_text(encoding='utf-8')) == ['voice_preset_sonja']


def test_deleted_preset_is_not_seeded_again(tmp_path) -> None:
    store = InMemoryStore(max_queue_size=4)
    seed_voice_presets(store, tmp_path)
    store.voice_profiles.pop('voice_preset_sonja')
    store.save_voices(tmp_path)

    restarted = InMemoryStore(max_queue_size=4)
    restarted.load_voices(tmp_path)
    assert seed_voice_presets(restarted, tmp_path) == []
    assert 'voice_preset_sonja' not in restarted.voice_profiles


def test_existing_profile_named_sonja_is_kept(tmp_path) -> None:
    store = InMemoryStore(max_queue_size=4)
    store.voice_profiles['voice_own'] = VoiceProfileRecord(
        voice_id='voice_own',
        name='sonja',
        source='custom',
        created_at=datetime(2026, 10, 1, tzinfo=timezone.utc),
        audio_bytes=b'RIFF',
        ref_text='Eigene Aufnahme.',
        consent=True,
    )
    assert seed_voice_presets(store, tmp_path) == []
    assert set(store.voice_profiles) == {'voice_own'}
