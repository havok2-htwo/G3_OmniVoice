from __future__ import annotations

import numpy as np
import pytest

from omnivoice_tts_server.fast_silence import remove_silence_fast

audio_utils = pytest.importorskip('omnivoice.utils.audio')

RATE = 24000
ARGS = [(500, 100, 100), (200, 100, 200), (300, 100, 300), (0, 100, 100)]


def _cases() -> list[np.ndarray]:
    rng = np.random.default_rng(7)
    t = np.arange(RATE * 4) / RATE
    speechlike = (0.3 * np.sin(2 * np.pi * 180 * t) * (np.sin(2 * np.pi * 1.3 * t) > -0.2)).astype(np.float32)
    speechlike[: RATE // 3] = 0
    speechlike[RATE * 2 : RATE * 2 + RATE * 7 // 10] *= 1e-4
    cases = [speechlike, np.full(RATE * 2 + 13, 1.5, np.float32)]
    for n in (0, 1, 23, 240, 11999, 12000, 12001, 12013, RATE * 3 + 7, RATE * 3 + 13):
        cases.append(np.zeros(n, np.float32))
        cases.append((0.2 * rng.standard_normal(n)).astype(np.float32))
        x = (0.2 * rng.standard_normal(n)).astype(np.float32)
        x[: n // 3] = 0
        x[n - n // 4 :] *= 1e-4
        cases.append(x)
    return cases


@pytest.mark.parametrize('args', ARGS)
def test_port_matches_pydub_sample_for_sample(args) -> None:
    mid_sil, lead_sil, trail_sil = args
    for x in _cases():
        audio = x[np.newaxis, :]
        expected = audio_utils.remove_silence(audio, RATE, mid_sil=mid_sil, lead_sil=lead_sil, trail_sil=trail_sil)
        got = remove_silence_fast(audio, RATE, mid_sil, lead_sil, trail_sil)
        assert got.dtype == expected.dtype
        assert got.shape == expected.shape, (len(x), args)
        assert np.array_equal(got, expected), (len(x), args)


def test_non_mono_input_is_rejected_so_the_wrapper_falls_back() -> None:
    with pytest.raises(TypeError):
        remove_silence_fast(np.zeros((2, 100), np.float32), RATE)
