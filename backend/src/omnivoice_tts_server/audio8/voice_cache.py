"""Reference codes of the CPU engine on disk: encoding a reference takes seconds there, a restart should not repeat it.

Codec codes are enough to clone a speaker, so they must not outlive the voice profile that holds the speaker's consent.
A file's name therefore starts with a tag of its voice_id: deleting a voice finds its files with no engine loaded, and a
load removes what belongs to no current voice. A reference sent with a single request has no profile and is never written.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

VOICE_CACHE_FOLDER = 'audio8_onnx_voices'


def voice_cache_folder(data_dir: Path | str) -> Path:
    return Path(data_dir) / VOICE_CACHE_FOLDER


def voice_tag(voice_id: str) -> str:
    return hashlib.sha256(voice_id.encode('utf-8')).hexdigest()[:32]


def voice_file(folder: Path, voice_id: str, content: str) -> Path:
    """The file of a voice's codes. ``content`` names what they were made from (the model and the audio)."""
    return folder / f'{voice_tag(voice_id)}.{hashlib.sha256(content.encode("utf-8")).hexdigest()}.npy'


def _remove(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:  # the next load prunes it
        logger.warning('Audio8: could not remove the voice cache %s: %s', path.name, exc)


def remove_voice(folder: Path, voice_id: str) -> None:
    """Every file of a voice, whichever engine runs."""
    if folder.is_dir():
        for path in folder.glob(f'{voice_tag(voice_id)}.*'):
            _remove(path)


def prune_voices(folder: Path, voice_ids: Iterable[str]) -> None:
    """Everything in the folder that belongs to none of these voices."""
    if not folder.is_dir():
        return
    tags = {voice_tag(voice_id) for voice_id in voice_ids}
    for path in folder.iterdir():
        if path.is_file() and path.name.split('.')[0] not in tags:
            _remove(path)
