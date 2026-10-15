"""Deterministic TTS text optimizer (Python port of the Genesis3 TypeScript pipeline).

The port keeps the TS rule names (``applied_rules``) and is checked against golden
outputs of the original TS code (tests/test_text_optimizer_parity.py).

Public entry points:

* :func:`optimize_text` - the server wrapper: language binding for request languages
  incl. ``Auto``, OmniVoice inline-tag protection, paragraph handling.
* :func:`optimize_tts_text` - the full deterministic pipeline (stack frames, URL domains,
  Markdown, de/en language passes, post-processor) for an already bound language.
* :func:`optimize_tts_text_for_language` - the language passes alone (``index.ts``).
* :func:`resolve_text_optimization_mode` - request flag / metadata / admin default.

Importing the package is cheap; the ~250 rule patterns are compiled on first use of a
pipeline function (about one second), not at server import time.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Optional

from .modes import TEXT_OPTIMIZATION_MODES, parse_text_optimization_mode, resolve_text_optimization_mode

_LAZY = {
    "TextOptimizationOutcome": ".service",
    "optimize_text": ".service",
    "resolve_optimizer_language": ".service",
    "DeterministicTtsResult": ".pipeline",
    "build_deterministic_tts_text": ".pipeline",
    "optimize_tts_text_for_language": ".pipeline",
    "detect_hint_language_from_text": ".language_binding",
    "resolve_tts_language_binding": ".language_binding",
}


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module, __name__), name)


def optimize_tts_text(text: str, language: Optional[str], *, stage: str = "final"):
    """Full deterministic pipeline for an already bound language ('de', 'en', ...)."""
    from .pipeline import build_deterministic_tts_text

    return build_deterministic_tts_text(text, language or "", stage=stage)


__all__ = [
    "TEXT_OPTIMIZATION_MODES",
    "parse_text_optimization_mode",
    "resolve_text_optimization_mode",
    "optimize_tts_text",
    *_LAZY,
]
