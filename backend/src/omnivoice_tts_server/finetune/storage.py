"""On-disk layout for finetune training material.

    data/finetune/train/<voice>/
        voice_sample_<voice>.wav        # reference sample for the voice
        clip_0000.wav  clip_0000.txt    # accepted clip + its intended sentence (same basename)
        clip_0000.json                  # optional sidecar {language_id, instruct, wer, seed}

A "clip_id" exposed to the API is the base64url of the clip's path relative to the
train root; every access re-validates it stays inside the train root (traversal guard).
"""
from __future__ import annotations

import base64
import json
import re
import threading
import wave
from pathlib import Path
from typing import Any

_LOCK = threading.RLock()

_SAFE = re.compile(r'[^A-Za-z0-9_-]+')


def train_root(data_dir: Path) -> Path:
    return data_dir / 'finetune' / 'train'


def sanitize_voice_dir(name: str) -> str:
    cleaned = _SAFE.sub('_', (name or '').strip()).strip('_')
    return cleaned or 'voice'


def voice_dir(data_dir: Path, voice_label: str) -> Path:
    return train_root(data_dir) / sanitize_voice_dir(voice_label)


def encode_clip_id(rel_path: str) -> str:
    return base64.urlsafe_b64encode(rel_path.encode('utf-8')).decode('ascii').rstrip('=')


def _decode_clip_id(clip_id: str) -> str:
    padding = '=' * (-len(clip_id) % 4)
    return base64.urlsafe_b64decode((clip_id + padding).encode('ascii')).decode('utf-8')


def resolve_clip_wav(data_dir: Path, clip_id: str) -> Path:
    """Decode a clip_id to an absolute .wav path, guarding against path traversal."""
    root = train_root(data_dir).resolve()
    rel = _decode_clip_id(clip_id)
    candidate = (root / rel).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError('clip_id escapes the training directory.')
    if candidate.suffix.lower() != '.wav':
        raise ValueError('clip_id does not point at a wav clip.')
    return candidate


def _wav_seconds(path: Path) -> float:
    try:
        with wave.open(str(path), 'rb') as handle:
            frames = handle.getnframes()
            rate = handle.getframerate() or 1
            return frames / float(rate)
    except Exception:
        return 0.0


def ensure_voice_sample(data_dir: Path, voice_label: str, wav_bytes: bytes) -> Path:
    """Write voice_sample_<voice>.wav once (used as the reference sample for the folder)."""
    target_dir = voice_dir(data_dir, voice_label)
    with _LOCK:
        target_dir.mkdir(parents=True, exist_ok=True)
        sample_path = target_dir / f'voice_sample_{sanitize_voice_dir(voice_label)}.wav'
        if not sample_path.exists() and wav_bytes:
            sample_path.write_bytes(wav_bytes)
    return sample_path


def _next_clip_index(target_dir: Path) -> int:
    highest = -1
    for entry in target_dir.glob('clip_*.wav'):
        match = re.match(r'clip_(\d+)\.wav$', entry.name)
        if match:
            highest = max(highest, int(match.group(1)))
    return highest + 1


def write_clip(
    data_dir: Path,
    voice_label: str,
    *,
    wav_bytes: bytes,
    text: str,
    meta: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Write clip_NNNN.wav + clip_NNNN.txt (+ optional .json). Returns (clip_id, basename)."""
    target_dir = voice_dir(data_dir, voice_label)
    with _LOCK:
        target_dir.mkdir(parents=True, exist_ok=True)
        index = _next_clip_index(target_dir)
        basename = f'clip_{index:04d}'
        wav_path = target_dir / f'{basename}.wav'
        txt_path = target_dir / f'{basename}.txt'
        wav_path.write_bytes(wav_bytes)
        txt_path.write_text(text.strip() + '\n', encoding='utf-8')
        if meta:
            (target_dir / f'{basename}.json').write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8'
            )
    rel = wav_path.resolve().relative_to(train_root(data_dir).resolve()).as_posix()
    return encode_clip_id(rel), basename


def list_clips(data_dir: Path, *, voice: str | None = None) -> list[dict[str, Any]]:
    root = train_root(data_dir)
    if not root.exists():
        return []
    clips: list[dict[str, Any]] = []
    voice_dirs = [root / sanitize_voice_dir(voice)] if voice else sorted(p for p in root.iterdir() if p.is_dir())
    for vdir in voice_dirs:
        if not vdir.is_dir():
            continue
        for wav_path in sorted(vdir.glob('clip_*.wav')):
            basename = wav_path.stem
            txt_path = vdir / f'{basename}.txt'
            json_path = vdir / f'{basename}.json'
            text = txt_path.read_text(encoding='utf-8').strip() if txt_path.exists() else ''
            wer: float | None = None
            if json_path.exists():
                try:
                    meta = json.loads(json_path.read_text(encoding='utf-8'))
                    wer = meta.get('wer')
                except Exception:
                    wer = None
            stat = wav_path.stat()
            rel = wav_path.resolve().relative_to(root.resolve()).as_posix()
            clips.append(
                {
                    'clip_id': encode_clip_id(rel),
                    'voice': vdir.name,
                    'text': text,
                    'wer': wer,
                    'filename': wav_path.name,
                    'size_bytes': stat.st_size,
                    'created_at': _iso_from_mtime(stat.st_mtime),
                }
            )
    return clips


def _iso_from_mtime(mtime: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()


def delete_clip(data_dir: Path, clip_id: str) -> list[str]:
    """Delete a clip's .wav plus its sibling .txt/.json. Returns removed filenames."""
    wav_path = resolve_clip_wav(data_dir, clip_id)
    removed: list[str] = []
    with _LOCK:
        for suffix in ('.wav', '.txt', '.json'):
            sibling = wav_path.with_suffix(suffix)
            if sibling.exists():
                sibling.unlink()
                removed.append(sibling.name)
    return removed


def dataset_voice_dirs(data_dir: Path) -> list[Path]:
    root = train_root(data_dir)
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir())


def dataset_summary(data_dir: Path) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for vdir in dataset_voice_dirs(data_dir):
        clips = sorted(vdir.glob('clip_*.wav'))
        seconds = sum(_wav_seconds(p) for p in clips)
        summary.append({'voice': vdir.name, 'clips': len(clips), 'seconds': round(seconds, 2)})
    return summary
