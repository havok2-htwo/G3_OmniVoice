from __future__ import annotations

import shutil
import subprocess


def _ffmpeg_executable() -> str | None:
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def wav_to_mp3(wav_bytes: bytes, *, bitrate: str = '192k', timeout_seconds: int = 120) -> bytes:
    if not wav_bytes:
        raise RuntimeError('Cannot export empty audio as MP3.')
    ffmpeg = _ffmpeg_executable()
    if not ffmpeg:
        raise RuntimeError('MP3 export requires ffmpeg or imageio-ffmpeg in the runtime environment.')

    completed = subprocess.run(
        [
            ffmpeg,
            '-hide_banner',
            '-loglevel',
            'error',
            '-i',
            'pipe:0',
            '-vn',
            '-codec:a',
            'libmp3lame',
            '-b:a',
            bitrate,
            '-f',
            'mp3',
            'pipe:1',
        ],
        input=wav_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0 or not completed.stdout:
        stderr = completed.stderr.decode('utf-8', errors='replace').strip()
        detail = f': {stderr}' if stderr else ''
        raise RuntimeError(f'FFmpeg MP3 export failed{detail}')
    return completed.stdout
