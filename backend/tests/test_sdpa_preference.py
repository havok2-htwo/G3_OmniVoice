from __future__ import annotations

import torch
from torch.nn.attention import SDPBackend

from omnivoice_tts_server import runtime_v2
from omnivoice_tts_server.runtime_v2 import apply_sdpa_backend_preference


def test_cudnn_first_and_back_with_cache_tag() -> None:
    original = list(torch._C._get_sdp_priority_order())
    original_tag = torch.compiler.config.cache_key_tag
    try:
        assert apply_sdpa_backend_preference(torch, True) is True
        order = list(torch._C._get_sdp_priority_order())
        assert order[0] == int(SDPBackend.CUDNN_ATTENTION)
        assert sorted(order) == sorted(original)
        assert runtime_v2._SDPA_CACHE_TAG in torch.compiler.config.cache_key_tag.split(',')
        # Applying twice does not duplicate the tag.
        apply_sdpa_backend_preference(torch, True)
        assert torch.compiler.config.cache_key_tag.split(',').count(runtime_v2._SDPA_CACHE_TAG) == 1

        assert apply_sdpa_backend_preference(torch, False) is False
        assert list(torch._C._get_sdp_priority_order()) == original
        assert runtime_v2._SDPA_CACHE_TAG not in torch.compiler.config.cache_key_tag.split(',')
    finally:
        torch._C._set_sdp_priority_order(original)
        torch.compiler.config.cache_key_tag = original_tag


def test_setting_round_trips_through_presets() -> None:
    from omnivoice_tts_server.presets import PRESET_KEYS

    assert 'sdpa_prefer_cudnn' in PRESET_KEYS
