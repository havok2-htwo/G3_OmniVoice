"""Port of Genesis3 ``src/utils/ttsLanguageOptimizer/markdownHtml.ts``.

The HTML a language model writes into Markdown, removed before the speech rules see it. Only
element names of a fixed list count as tags, so a generic type ("List<string>") and a comparison
("Wert < 5") stay as written.
"""

from __future__ import annotations

from .jsre import js, js_replace, js_trim
from .shared import remove_invisible_characters_for_tts, remove_zero_width_joiners_for_tts

_COMMENT_RE = js(r"<!--[\s\S]*?-->", "g")
_SUMMARY_RE = js(r"<summary(?=[\s>])[^<>]*>([\s\S]*?)<\/summary\s*>", "gi")
_LINE_BREAK_TAG_RE = js(r"<br\s*\/?>", "gi")
_BLOCK_TAG_RE = js(r"<\/?(?:p|div|li|ul|ol|tr|table|h[1-6]|blockquote|details)(?=[\s/>])[^<>]*>", "gi")
_EMAIL_AUTOLINK_RE = js(r"<(?:mailto:)?([^\s<>@]+@[^\s<>]+)>", "gi")
_STRUCK_ELEMENT_RE = js(r"<(s|del|strike)(?=[\s>])[^<>]*>[\s\S]*?<\/\1\s*>", "gi")
_INLINE_TAG_RE = js(r"<\/?(?:b|strong|i|em|u|code|kbd|span|sup|sub|mark|small|abbr|a|ins|font)(?=[\s/>])[^<>]*>", "gi")
_OTHER_TAG_RE = js(
    r"<\/?(?:img|hr|br|section|article|center|figure|figcaption|thead|tbody|tfoot|td|th|caption|header|footer|nav|main"
    r"|aside|dl|dt|dd|pre|q|cite|time|var|samp|big|tt|wbr|label|col|colgroup|summary|details|s|del|strike|p|div|li|ul"
    r"|ol|tr|table|h[1-6]|blockquote)(?=[\s/>])[^<>]*>",
    "gi",
)
_ENTITY_RE = js(r"&(?:#(\d+)|#x([0-9a-f]+)|([a-z][a-z0-9]{1,7}));", "gi")

_NAMED_ENTITIES = {
    "amp": "&", "lt": "<", "gt": ">", "quot": '"', "apos": "'", "nbsp": " ", "ensp": " ", "emsp": " ", "thinsp": " ",
    "shy": "", "euro": "€", "cent": "¢", "pound": "£", "yen": "¥", "copy": "©", "reg": "®", "trade": "™", "deg": "°",
    "sect": "§", "para": "¶", "middot": "·", "bull": "•", "hellip": "…", "ndash": "–", "mdash": "—", "minus": "−",
    "times": "×", "divide": "÷", "plusmn": "±", "frac12": "½", "frac14": "¼", "frac34": "¾", "laquo": "«", "raquo": "»",
    "lsquo": "‘", "rsquo": "’", "sbquo": "‚", "ldquo": "“", "rdquo": "”", "bdquo": "„", "auml": "ä", "ouml": "ö",
    "uuml": "ü", "Auml": "Ä", "Ouml": "Ö", "Uuml": "Ü", "szlig": "ß", "eacute": "é", "egrave": "è", "agrave": "à",
    "rarr": "→", "larr": "←", "hearts": "♥",
}

_LABEL_END = js(r"[.!?:]$", "")


def strip_html_blocks_for_tts(text: str) -> str:
    """Comments, summaries, line breaks and block tags of the whole text."""
    source = text or ""
    if "<" not in source:
        return source

    def summary(_match, content, *_rest):
        label = js_trim(content)
        return f"{label}\n" if not label or _LABEL_END.test(label) else f"{label}:\n"

    source = js_replace(_COMMENT_RE, source, "")
    source = js_replace(_SUMMARY_RE, source, summary)
    source = js_replace(_LINE_BREAK_TAG_RE, source, "\n")
    return js_replace(_BLOCK_TAG_RE, source, "\n")


_FOOTNOTE_SUPERSCRIPT_RE = js(r"(?<=(?<!\p{L})\p{L}{4,})<sup(?=[\s>])[^<>]*>[ \t]*\d+[ \t]*<\/sup\s*>", "giu")
_SUPERSCRIPT_NUMBER_RE = js(r"(?<=[^\s<>])<sup(?=[\s>])[^<>]*>[ \t]*([-+−]?\d+)[ \t]*<\/sup\s*>", "gi")
_SUBSCRIPT_ELEMENT_RE = js(r"(?<=([\p{L}\d])|[^\p{L}\d]|^)<sub(?=[\s>])[^<>]*>([^<>]*)<\/sub\s*>(?=([\p{L}\d])?)", "giu")
_SUPERSCRIPT_GLYPHS = "⁰¹²³⁴⁵⁶⁷⁸⁹"


def _superscript(_match, value, *_rest):
    return "".join(
        _SUPERSCRIPT_GLYPHS[int(c)] if "0" <= c <= "9" else ("⁺" if c == "+" else "⁻") for c in value
    )


def _subscript(_match, before, content, after, *_rest):
    return f"{' ' if before is not None else ''}{content}{' ' if after is not None else ''}"


def strip_inline_html_for_tts(text: str) -> str:
    """Autolinked addresses, struck-through elements, inline and other known tags inside a line."""
    source = text or ""
    if "<" not in source:
        return source
    source = js_replace(_EMAIL_AUTOLINK_RE, source, "$1")
    source = js_replace(_STRUCK_ELEMENT_RE, source, "")
    source = js_replace(_FOOTNOTE_SUPERSCRIPT_RE, source, "")
    source = js_replace(_SUPERSCRIPT_NUMBER_RE, source, _superscript)
    source = js_replace(_SUBSCRIPT_ELEMENT_RE, source, _subscript)
    source = js_replace(_INLINE_TAG_RE, source, "")
    return js_replace(_OTHER_TAG_RE, source, " ")


def decode_html_entities_for_tts(text: str) -> str:
    """"Tom &amp; Jerry" -> "Tom & Jerry", "&#8364;" -> "€"; unknown names stay."""
    source = text or ""
    if "&" not in source:
        return source

    def entity(match, decimal, hex_digits, name, *_rest):
        if decimal or hex_digits:
            code_point = int(decimal) if decimal else int(hex_digits, 16)
            if code_point <= 0 or code_point > 0x10FFFF:
                return match
            return " " if code_point == 0xA0 else chr(code_point)
        named = _NAMED_ENTITIES.get(str(name))
        if named is None:
            named = _NAMED_ENTITIES.get(str(name).lower())
        return match if named is None else named

    decoded = js_replace(_ENTITY_RE, source, entity)
    return source if decoded == source else remove_zero_width_joiners_for_tts(remove_invisible_characters_for_tts(decoded))
