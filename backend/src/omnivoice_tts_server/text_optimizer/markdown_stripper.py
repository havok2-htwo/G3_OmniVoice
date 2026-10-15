"""Port of Genesis3 ``src/utils/ttsLanguageOptimizer/markdownStripper.ts``."""

from __future__ import annotations

from .jsre import js, js_replace
from .shared import strip_emoji_for_tts

_STEPS = (
    (js(r"```[^\n`]*\n[\s\S]*?```", "g"), ""),
    (js(r"```", "g"), ""),
    (js(r"`([^`\n]+?)`", "g"), "$1"),
    (js(r"!\[([^\]]*)\]\([^)]*\)", "g"), "$1"),
    (js(r"\[([^\]]+)\]\([^)]*\)", "g"), "$1"),
    (js(r"\[([^\]]+)\]\[[^\]]*\]", "g"), "$1"),
    (js(r"^[ \t]{0,3}#{1,6}[ \t]+", "gm"), ""),
    (js(r"^[ \t]*>+[ \t]?", "gm"), ""),
    (js(r"^[ \t]*([*_\-])\1{2,}[ \t]*$", "gm"), ""),
    (js(r"^[ \t]*\|?[ \t]*:?-{3,}:?[ \t]*(?:\|[ \t]*:?-{3,}:?[ \t]*)*\|?[ \t]*$", "gm"), ""),
    (js(r"^[ \t]*[-+*•·][ \t]+(?=\S)", "gm"), ""),
    (js(r"\*\*\*([^*\n]+?)\*\*\*", "g"), "$1"),
    (js(r"___([^_\n]+?)___", "g"), "$1"),
    (js(r"\*\*([^*\n]+?)\*\*", "g"), "$1"),
    (js(r"__([^_\n]+?)__", "g"), "$1"),
    (js(r"~~([^~\n]+?)~~", "g"), ""),
    (js(r"(?<![\w*])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![\w*])", "g"), "$1"),
    (js(r"(?<![\w_])_(?!\s)([^_\n]+?)(?<!\s)_(?![\w_])", "g"), "$1"),
    (js(r"[ \t]*\|[ \t]*", "g"), ", "),
    (js(r"\n{3,}", "g"), "\n\n"),
    (js(r"^[ \t]+\n", "gm"), "\n"),
)


def strip_markdown_for_tts(text: str) -> str:
    """Remove Markdown decorations (and emoji) while keeping the inner content."""
    result = text or ""
    if not result:
        return result
    result = strip_emoji_for_tts(result)
    for pattern, replacement in _STEPS:
        result = js_replace(pattern, result, replacement)
    return result
