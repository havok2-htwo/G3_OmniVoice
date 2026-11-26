"""Port of Genesis3 ``src/utils/ttsLanguageOptimizer/markdownStripper.ts``.

Steps: emoji, fenced code, definition lines and HTML blocks over the whole text, then the line
pass (markdown_line_structure.py), which reads every line's block marker, runs the inline rules
and speaks the structure of the answer as sentence boundaries (Bug 824). Footnotes and link
reference definitions are not read at all.
"""

from __future__ import annotations

from typing import Optional

from .jsre import js, js_replace, js_trim
from .markdown_html import strip_html_blocks_for_tts
from .markdown_line_structure import MarkdownSpokenWords, apply_markdown_line_structure
from .reference_links import is_link_reference_definition_line
from .shared import remove_zero_width_joiners_for_tts, strip_emoji_lines_for_tts

_FOOTNOTE_DEFINITION_LINE_RE = js(r"^ {0,3}\[\^[^\]]+\]:", "")
_FENCED_CODE_RE = js(r"```[^\n`]*\n[\s\S]*?```", "g")
_FENCE = js(r"```", "g")
_THREE_LINE_BREAKS = js(r"\n{3,}", "g")
_BLANK_LINE_SPACES = js(r"^[ \t]+\n", "gm")


def _is_definition_line(line: str) -> bool:
    return _FOOTNOTE_DEFINITION_LINE_RE.test(line) or is_link_reference_definition_line(line)


def _drop_definition_lines(text: str) -> str:
    lines = text.split("\n")
    kept = [line for line in lines if not _is_definition_line(line)]
    if len(kept) == len(lines) or not any(js_trim(line) for line in kept):
        return text
    return "\n".join(kept)


def is_unspoken_markdown_only(text: str) -> bool:
    """``isUnspokenMarkdownOnly``: nothing but fenced code and definition lines."""
    source = text or ""
    if not js_trim(source):
        return False
    return all(not js_trim(line) or _is_definition_line(line) for line in js_replace(_FENCED_CODE_RE, source, "").split("\n"))


def _markdown_words_for(language: Optional[str]) -> Optional[MarkdownSpokenWords]:
    from .de import GERMAN_MARKDOWN_WORDS
    from .en import ENGLISH_MARKDOWN_WORDS

    normalized = js_trim(language or "").lower()
    if normalized.startswith("de"):
        return GERMAN_MARKDOWN_WORDS
    if normalized.startswith("en"):
        return ENGLISH_MARKDOWN_WORDS
    return None


def strip_markdown_for_tts(text: str, language: Optional[str] = None, *, mixed_status_run: bool = False) -> str:
    """Remove Markdown decorations (and emoji) while keeping the inner content.

    ``language`` only decides the words a task item's state and a status mark are spoken with.
    """
    result = text or ""
    if not result:
        return result
    result = remove_zero_width_joiners_for_tts(strip_emoji_lines_for_tts(result))
    result = js_replace(_FENCED_CODE_RE, result, "")
    result = js_replace(_FENCE, result, "")
    result = _drop_definition_lines(result)
    result = strip_html_blocks_for_tts(result)
    result = apply_markdown_line_structure(result, _markdown_words_for(language), mixed_status_run=mixed_status_run)
    result = js_replace(_THREE_LINE_BREAKS, result, "\n\n")
    return js_replace(_BLANK_LINE_SPACES, result, "\n")
