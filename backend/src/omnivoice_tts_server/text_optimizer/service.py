"""Server-facing glue: request flag resolution, language binding, OmniVoice tag protection."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import regex

from .jsre import time_budget
from .language_binding import resolve_tts_language_binding
from .modes import (  # noqa: F401  (re-exported)
    TEXT_OPTIMIZATION_MODES,
    parse_text_optimization_mode,
    resolve_text_optimization_mode,
)
from .omnivoice_tags import protect_inline_tags, restore_inline_tags
from .pipeline import build_deterministic_tts_text
from .text_language import detect_word_language, words_contradict

logger = logging.getLogger("omnivoice_tts_server.text_optimizer")

_LANG_DE = regex.compile(r"^(?:de|deu|ger)(?:[-_].*)?$")
_LANG_EN = regex.compile(r"^(?:en|eng)(?:[-_].*)?$")
_LANGUAGE_NAMES = {
    "deutsch": "de",
    "german": "de",
    "englisch": "en",
    "english": "en",
}

# Wall-clock budget for the rule passes of one text. Prose runs ~1 s per 25k characters;
# number-dense input (long "1.1.1..." or CSV digit runs) is quadratic in the Genesis
# patterns and would otherwise hold a worker thread for minutes. On timeout the raw
# text is spoken (binding_origin 'timeout').
OPTIMIZER_TIME_BUDGET_SECONDS = 5.0


def resolve_optimizer_language(request_language: Optional[str], text: str) -> tuple[Optional[str], str]:
    """Bind the optimizer language: (language 'de'/'en' or None, origin).

    An explicit de/en request language is used as given ('request'). Any other explicit
    language is a no-op ('unsupported_language'): the German/English rule sets would
    otherwise rewrite e.g. French text.

    'Auto' or no language:

    * the Genesis notation vote (dd.mm.yyyy, 1.234,56, 29,90 €, 3:30 PM, ...) binds
      ('chunk_notation') unless the text's function words say another language or the
      text is mostly non-Latin script ('other_language') - Dutch, Polish, Italian,
      French, Turkish or Russian write the same dates and amounts;
    * without a notation anchor, German or English function words bind ('text_words');
    * otherwise nothing is bound ('none' when undecided, 'other_language' when the
      words name another language).
    """
    value = (request_language or "").strip().lower()
    if not value or value == "auto":
        language, origin = resolve_tts_language_binding(text, None)
        if language:
            if words_contradict(text, language):
                return None, "other_language"
            return language, origin
        detected = detect_word_language(text)
        if detected in ("de", "en"):
            return detected, "text_words"
        return None, ("other_language" if detected is not None else "none")
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
    error: Optional[str] = None

    @property
    def changed(self) -> bool:
        return bool(self.applied_rules)

    def as_dict(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "mode": self.mode,
            "language": self.language,
            "binding_origin": self.binding_origin,
            "applied_rules": list(self.applied_rules),
        }
        if self.error:
            info["error"] = self.error
        return info


_PARAGRAPH_BREAK = regex.compile(r"[ \t]*\n(?:[ \t]*\n)+[ \t]*")
_LINE_BREAKS = regex.compile(r"\r\n?| | ")


def _normalize_line_breaks(text: str) -> str:
    """CRLF / CR / U+2028 -> '\\n', U+2029 (paragraph separator) -> blank line."""
    return _LINE_BREAKS.sub(lambda m: "\n\n" if m.group(0) == " " else "\n", text)


def optimize_text(
    text: str,
    language: Optional[str],
    *,
    stage: str = "final",
    time_budget_seconds: Optional[float] = OPTIMIZER_TIME_BUDGET_SECONDS,
) -> TextOptimizationOutcome:
    """Run the deterministic pipeline on ``text`` for a request language (de/en/Auto).

    OmniVoice-specific wrappers around the Genesis pipeline: line endings are
    normalised (CRLF, CR, U+2028/U+2029), paragraph breaks are reduced to a single
    newline first, because the pipeline folds a blank line into a space and the sentence
    splitter would then glue a heading onto the next paragraph; and OmniVoice control
    spans (``[laughter]``, ``[B EY1 S]``) are protected. When no rule applies the
    original text is returned unchanged, so sentence splitting stays identical. The rule
    passes run under ``time_budget_seconds``; when it is spent the raw text comes back
    with binding_origin 'timeout'.
    """
    bound, origin = resolve_optimizer_language(language, text)
    if bound is None:
        return TextOptimizationOutcome(mode="deterministic", text=text, language=None, binding_origin=origin)
    prepared = _PARAGRAPH_BREAK.sub("\n", _normalize_line_breaks(text))
    protection = protect_inline_tags(prepared)
    if protection is None:
        # No free placeholder for every control span: skip rather than speak the tags.
        return TextOptimizationOutcome(mode="deterministic", text=text, language=bound, binding_origin="tags_unprotected")
    protected, spans = protection
    try:
        with time_budget(time_budget_seconds):
            result = build_deterministic_tts_text(protected, bound, stage=stage)
    except TimeoutError:
        logger.warning(
            "text optimization exceeded %.1f s for %d characters; speaking the raw text",
            time_budget_seconds or 0.0,
            len(text),
        )
        return TextOptimizationOutcome(mode="deterministic", text=text, language=bound, binding_origin="timeout")
    if not result.applied_rules:
        return TextOptimizationOutcome(
            mode="deterministic", text=text, language=bound, binding_origin=origin, protected_tags=len(spans)
        )
    return TextOptimizationOutcome(
        mode="deterministic",
        text=restore_inline_tags(result.text, spans),
        language=bound,
        binding_origin=origin,
        applied_rules=list(result.applied_rules),
        protected_tags=len(spans),
    )


def optimize_request_text(text: str, language: Optional[str], *, stage: str = "final") -> TextOptimizationOutcome:
    """What a synthesis request (and its dry-run preview) speaks for ``text``.

    Like :func:`optimize_text`, but a rule bug never costs the request its audio: any
    exception falls back to the raw text (binding_origin 'error'). The returned text is
    stripped; the caller rejects an empty result.
    """
    try:
        outcome = optimize_text(text, language, stage=stage)
    except Exception as exc:
        logger.exception("text optimization failed; speaking the raw input")
        outcome = TextOptimizationOutcome(mode="deterministic", text=text, binding_origin="error", error=str(exc))
    outcome.text = outcome.text.strip()
    return outcome
