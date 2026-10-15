"""Protect OmniVoice inline control spans from the text optimizer.

OmniVoice reads bracketed spans as control syntax: non-verbal tags such as
``[laughter]`` / ``[sigh]`` and CMU pronunciation overrides such as
``[B EY1 S]``. The Genesis pipeline would turn them into spoken words ("laughter",
"B EY 1 S"), so they are swapped for private-use placeholders before the
pipeline runs and restored verbatim afterwards (the same spans upstream
``omnivoice.utils.text.normalize_text`` holds out).

A bracket directly followed by ``(`` or ``[`` is a Markdown link / reference
and is left to the Markdown stripper.
"""

from __future__ import annotations

import regex

# One BMP private-use character per span. The post-processor's own URL markers use
# U+E000/U+E001, so this range is disjoint. No language pass matches these characters:
# they are neither letters, digits, marks, whitespace nor emoji.
_PLACEHOLDER_BASE = 0xF000
_PLACEHOLDER_LIMIT = 0xF8FF
_TAG_RE = regex.compile(r"\[[^\[\]\n]+\](?![(\[])")
_PLACEHOLDER_RE = regex.compile("[-]")


def protect_inline_tags(text: str) -> tuple[str, list[str]]:
    """Replace inline control spans with placeholders. Returns (text, spans)."""
    spans: list[str] = []

    def replace(match: "regex.Match[str]") -> str:
        if _PLACEHOLDER_BASE + len(spans) > _PLACEHOLDER_LIMIT:
            return match.group(0)
        spans.append(match.group(0))
        return chr(_PLACEHOLDER_BASE + len(spans) - 1)

    if "[" not in text or _PLACEHOLDER_RE.search(text):
        # Text that already carries characters of the placeholder range is left alone,
        # so a restore can never swap a user's own character for a tag.
        return text, spans
    return _TAG_RE.sub(replace, text), spans


def restore_inline_tags(text: str, spans: list[str]) -> str:
    if not spans:
        return text

    def replace(match: "regex.Match[str]") -> str:
        index = ord(match.group(0)) - _PLACEHOLDER_BASE
        return spans[index] if 0 <= index < len(spans) else match.group(0)

    return _PLACEHOLDER_RE.sub(replace, text)


def count_unrestored_placeholders(text: str) -> int:
    return len(_PLACEHOLDER_RE.findall(text))
