"""Port of Genesis3 ``src/utils/ttsLanguageOptimizer/markdownInline.ts``.

The Markdown rules that work inside one line, run by the line pass (markdown_line_structure.py):
footnote references go, inline code, images and links become their text, inline math and HTML are
unwrapped, entities decoded, a backslash escape keeps its character, emphasis is unwrapped,
struck-through text is dropped as retracted, and a table pipe becomes a comma.
"""

from __future__ import annotations

from .jsre import js, js_replace
from .latex_math import read_inline_math_for_tts
from .markdown_html import decode_html_entities_for_tts, strip_inline_html_for_tts

_FOOTNOTE_REFERENCE = js(r"\[\^[^\]\s]+\]", "g")
_INLINE_CODE = js(r"`([^`\n]+?)`", "g")
_IMAGE = js(r"!\[([^\]]*)\]\([^)]*\)", "g")
_LINK = js(r"\[([^\]]+)\]\([^)]*\)", "g")
_REFERENCE_LINK = js(r"(?<![\p{L}\p{N}_\]])\[([^\]]+)\]\[[^\]]*\]", "gu")
_TRIPLE_STAR = js(r"\*\*\*([^*\n]+?)\*\*\*", "g")
_TRIPLE_UNDERSCORE = js(r"___([^_\n]+?)___", "g")
_DOUBLE_STAR = js(r"\*\*([^*\n]+?)\*\*", "g")
_DOUBLE_UNDERSCORE = js(r"__([^_\n]+?)__", "g")
_STRIKETHROUGH = js(r"~~([^~\n]+?)~~", "g")
_SINGLE_STAR = js(r"(?<![\p{L}\p{N}_*])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![\p{L}\p{N}_*])", "gu")
_SINGLE_UNDERSCORE = js(r"(?<![\p{L}\p{N}_])_(?!\s)([^_\n]+?)(?<!\s)_(?![\p{L}\p{N}_])", "gu")
_TABLE_PIPE = js(r"[ \t]*\|[ \t]*", "g")
_SINGLE_PIPE = js(r"[ \t]*(?<!\|)\|(?!\|)[ \t]*", "g")


def strip_inline_markdown_for_tts(text: str, table_row: bool = True) -> str:
    """``stripInlineMarkdownForTts``. ``table_row``: every pipe becomes a comma (else a "||" stays)."""
    result = text or ""
    if not result:
        return result
    result = js_replace(_FOOTNOTE_REFERENCE, result, "")
    result = js_replace(_INLINE_CODE, result, "$1")
    result = js_replace(_IMAGE, result, "$1")
    result = js_replace(_LINK, result, "$1")
    result = js_replace(_REFERENCE_LINK, result, "$1")
    result = read_inline_math_for_tts(result)
    result = decode_html_entities_for_tts(strip_inline_html_for_tts(result))
    result = resolve_backslash_escapes_for_tts(result)
    result = js_replace(_TRIPLE_STAR, result, "$1")
    result = js_replace(_TRIPLE_UNDERSCORE, result, "$1")
    result = js_replace(_DOUBLE_STAR, result, "$1")
    result = js_replace(_DOUBLE_UNDERSCORE, result, "$1")
    result = js_replace(_STRIKETHROUGH, result, "")
    result = js_replace(_SINGLE_STAR, result, "$1")
    result = js_replace(_SINGLE_UNDERSCORE, result, "$1")
    return js_replace(_TABLE_PIPE if table_row else _SINGLE_PIPE, result, ", ")


_BACKSLASH_ESCAPE_RE = js(r"(?<!\\)\\([!-\/:-@\[-`{-~])", "g")
_PATH_TOKEN_RE = js(r"(?:^|[^\p{L}\p{N}])[A-Za-z]:\\|[\p{L}\p{N}_]\\[\p{L}\p{N}]", "u")
_TOKEN = js(r"\S+", "g")


def resolve_backslash_escapes_for_tts(text: str) -> str:
    """Resolves the backslash escapes of a line, except inside a path token."""
    if "\\" not in text:
        return text

    def token(match, *_rest):
        if "\\" in match and not _PATH_TOKEN_RE.test(match):
            return js_replace(_BACKSLASH_ESCAPE_RE, match, "$1")
        return match

    return js_replace(_TOKEN, text, token)
