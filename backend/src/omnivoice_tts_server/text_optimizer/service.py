"""Server-facing glue: request flag resolution, language binding, OmniVoice tag protection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import regex

from .language_binding import resolve_tts_language_binding
from .modes import (  # noqa: F401  (re-exported)
    TEXT_OPTIMIZATION_MODES,
    parse_text_optimization_mode,
    resolve_text_optimization_mode,
)
from .omnivoice_tags import protect_inline_tags, restore_inline_tags
from .pipeline import build_deterministic_tts_text

_LANG_DE = regex.compile(r"^(?:de|deu|ger)(?:[-_].*)?$")
_LANG_EN = regex.compile(r"^(?:en|eng)(?:[-_].*)?$")
_LANGUAGE_NAMES = {
    "deutsch": "de",
    "german": "de",
    "englisch": "en",
    "english": "en",
}


def resolve_optimizer_language(request_language: Optional[str], text: str) -> tuple[Optional[str], str]:
    """Bind the optimizer language: (language 'de'/'en' or None, origin).

    An explicit de/en request language is used as given ('request'). 'Auto' or no language
    binds from the text's notation ('chunk_notation'), else nothing ('none'). Any other
    explicit language is a no-op ('unsupported_language'): the German/English rule sets
    would otherwise rewrite e.g. French text.
    """
    value = (request_language or "").strip().lower()
    if not value or value == "auto":
        language, origin = resolve_tts_language_binding(text, None)
        return (language or None), origin
    value = _LANGUAGE_NAMES.get(value, value)
    if _LANG_DE.match(value):
        return "de", "request"
    if _LANG_EN.match(value):
        return "en", "request"
    return None, "unsupported_language"


@dataclass
class TextOptimizationOutcome:
    mode: str
    text: str
    language: Optional[str] = None
    binding_origin: str = "none"
    applied_rules: list[str] = field(default_factory=list)
    protected_tags: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.applied_rules)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "language": self.language,
            "binding_origin": self.binding_origin,
            "applied_rules": list(self.applied_rules),
        }


_PARAGRAPH_BREAK = regex.compile(r"[ \t]*\n(?:[ \t]*\n)+[ \t]*")


def optimize_text(text: str, language: Optional[str], *, stage: str = "final") -> TextOptimizationOutcome:
    """Run the deterministic pipeline on ``text`` for a request language (de/en/Auto).

    Two OmniVoice-specific wrappers around the Genesis pipeline: inline control spans
    (``[laughter]``, ``[B EY1 S]``) are protected, and paragraph breaks are reduced to a
    single newline first, because the pipeline folds a blank line into a space and the
    sentence splitter would then glue a heading onto the next paragraph. When no rule
    applies the original text is returned unchanged, so sentence splitting stays identical.
    """
    bound, origin = resolve_optimizer_language(language, text)
    if bound is None:
        return TextOptimizationOutcome(mode="deterministic", text=text, language=None, binding_origin=origin)
    protected, spans = protect_inline_tags(_PARAGRAPH_BREAK.sub("\n", text))
    result = build_deterministic_tts_text(protected, bound, stage=stage)
    if not result.applied_rules:
        return TextOptimizationOutcome(
            mode="deterministic", text=text, language=bound, binding_origin=origin, protected_tags=len(spans)
        )
    spoken = restore_inline_tags(result.text, spans)
    return TextOptimizationOutcome(
        mode="deterministic",
        text=spoken,
        language=bound,
        binding_origin=origin,
        applied_rules=list(result.applied_rules),
        protected_tags=len(spans),
    )
