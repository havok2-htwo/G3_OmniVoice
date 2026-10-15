"""Language binding from Genesis3 ``hintInjector.ts`` (detection only, no hint injection).

``detect_hint_language_from_text`` votes with notation that only one of German
or English writes; ``resolve_tts_language_binding`` falls back to a UI/default
language unless the text carries the other convention's digit grouping.
"""

from __future__ import annotations

from typing import Optional

from .jsre import js, js_trim

GERMAN_LANGUAGE_ANCHORS = (
    js(r"\b(?:januar|februar|märz|maerz|mai|juni|juli|oktober|dezember)\b", "gi"),
    js(r"\b\d{1,2}\.\s*(?:januar|februar|m[aä]e?rz|april|mai|juni|juli|august|september|oktober|november|dezember)\s+\d{2,4}\b", "gi"),
    js(r"\b\d{1,2}(?::\d{2})?\s*Uhr\b", "gi"),
    js(r"\b\d{1,2}\.\d{1,2}\.(?:19|20)\d{2}\b", "g"),
    js(r"\b\d{1,3}(?:\.\d{3})+,\d{1,2}\b(?![.,]?\d)", "g"),
    js(r"\b\d+,\d{1,2}\s?(?:€|EUR\b)", "g"),
    js(r"\b\d[\d.]*,\d+\s?(?:Mio|Mrd|Tsd)\b\.?", "g"),
    js(r"\b(?:am|vom|bis|zum|ab)\s+\d{1,2}\.\d{1,2}\.", "g"),
    js(r"\b\d{1,2}\.\s?(?:Jan|Feb|Mär|Apr|Jun|Jul|Aug|Sept?|Okt|Nov|Dez)\.?(?![\p{L}])", "gu"),
)
ENGLISH_LANGUAGE_ANCHORS = (
    js(r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},\s*\d{4}\b", "gi"),
    js(r"\b(?:january|february|march|june|july|october|december)\b", "gi"),
    js(r"\b\d{1,2}(?::\d{2})?\s*(?:[ap]\.m\.?|[AP]M)\b", "g"),
    js(r"\b\d{1,2}(?::\d{2})?[ap]m\b", "g"),
    js(r"\b\d{1,3}(?:,\d{3})+\.\d{1,2}\b(?![.,]?\d)", "g"),
    js(r"\$\s?\d[\d,]*\.\d{2}\b", "g"),
    js(r"\b\d+\.\d{2}\s?(?:USD|GBP)\b", "g"),
    js(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+\d{1,2}(?:st|nd|rd|th)?\b", "g"),
    js(r"\b\d{1,3}(?:st|nd|rd|th)\b", "g"),
)

FOREIGN_GROUPING_BY_LANGUAGE = {
    "de": js(r"(?<![\d.,])\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b(?![.,]\d)", ""),
    "en": js(r"(?<![\d.,])\b\d{1,3}(?:\.\d{3})+(?:,\d+)?\b(?![.,]\d)", ""),
}


def _count_language_anchors(text: str, patterns) -> int:
    return sum(sum(1 for _ in pattern.finditer(text)) for pattern in patterns)


def detect_hint_language_from_text(text: str) -> Optional[str]:
    """``detectHintLanguageFromText``: 'de', 'en' or None (ties go to German)."""
    source = text or ""
    german = _count_language_anchors(source, GERMAN_LANGUAGE_ANCHORS)
    english = _count_language_anchors(source, ENGLISH_LANGUAGE_ANCHORS)
    if german == 0 and english == 0:
        return None
    return "de" if german >= english else "en"


def resolve_tts_language_binding(text: str, ui_language: Optional[str]) -> tuple[str, str]:
    """``resolveTtsLanguageBinding`` -> (language '' | 'de' | 'en', origin).

    origin is 'chunk_notation', 'ui_locale_fallback' or 'none'.
    """
    detected = detect_hint_language_from_text(text)
    if detected:
        return detected, "chunk_notation"
    ui = js_trim(ui_language or "").lower()
    fallback = "de" if ui.startswith("de") else "en" if ui.startswith("en") else ""
    if not fallback:
        return "", "none"
    foreign = FOREIGN_GROUPING_BY_LANGUAGE.get(fallback)
    if foreign is not None and foreign.test(text or ""):
        return "", "none"
    return fallback, "ui_locale_fallback"
