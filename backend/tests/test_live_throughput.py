from __future__ import annotations

from collections import deque

from omnivoice_tts_server.domain.state import LIVE_THROUGHPUT_WINDOW_S, live_throughput_x


def test_live_throughput_is_audio_per_wall_second_over_the_window() -> None:
    events = deque([(100.0, 4.0), (110.0, 6.0)])
    assert live_throughput_x(events, 115.0) == round(10.0 / LIVE_THROUGHPUT_WINDOW_S, 2)


def test_live_throughput_falls_back_to_zero_once_the_window_has_passed() -> None:
    events = deque([(100.0, 4.0)])
    assert live_throughput_x(events, 100.0 + LIVE_THROUGHPUT_WINDOW_S + 1) == 0.0
    assert live_throughput_x(deque(), 1.0) == 0.0
