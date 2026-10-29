"""Protect the engines' inline control spans from the text optimizer, and spell them for the active engine.

OmniVoice reads exactly two bracketed forms as control syntax: the 13 non-verbal
tags of ``omnivoice.models.omnivoice._NONVERBAL_PATTERN`` (``[laughter]``,
``[sigh]``, ...) and CMU/ARPAbet pronunciation overrides such as ``[B EY1 S]``.
Ming-3.0-tts reads its 14 paralinguistic tags (``[Laughter]``, ``[Breathing]``, ...,
case-sensitive). The Genesis pipeline would turn them into spoken words
("laughter", "B EY 1 S"), so they are swapped for private-use placeholders before
the pipeline runs and restored verbatim afterwards. Tags of either engine match
case-insensitively. Every other bracket span (citations ``[1]``, editorial notes
``[ca. 5 €]``, Markdown checkboxes ``- [x]``) is ordinary text and goes through the
pipeline exactly like in the TS reference.

``render_inline_tags`` then spells every tag the way the active engine expects
(``[Laughter]`` on Ming, ``[laughter]`` on OmniVoice) and removes the ones it does
not know, so a tag is never read aloud. It runs for every request, with or without
text optimization.

A bracket directly followed by ``(``, or by a ``[`` that does not start another tag,
is a Markdown link / reference and is left to the Markdown stripper. Glued tags
(``[Uhm][Cough]``) are tags.

Placeholders are private-use code points that do not occur in the input (the
BMP block U+F000-U+F8FF first, then plane 15), so a text that already carries
e.g. the Apple logo U+F8FF keeps its tag protection.
"""

from __future__ import annotations

from typing import Optional

import regex

# Kept in sync with omnivoice.models.omnivoice._NONVERBAL_PATTERN (omnivoice 0.2.x).
NONVERBAL_TAGS = (
    "laughter",
    "sigh",
    "confirmation-en",
    "question-en",
    "question-ah",
    "question-oh",
    "question-ei",
    "question-yi",
    "surprise-ah",
    "surprise-oh",
    "surprise-wa",
    "surprise-yo",
    "dissatisfaction-hnn",
)
# Ming-3.0-tts paralinguistic tags (upstream frontend/toolkit.py PARALINGUISTIC_TAGS), in the model's spelling.
MING_TAGS = (
    "Breathing",
    "Cough",
    "Sigh",
    "Laughter",
    "Surprise-ah",
    "Surprise-oh",
    "Dissatisfaction-hnn",
    "Uhm",
    "Confirmation-en",
    "Question-ei",
    "Question-ah",
    "Question-en",
    "Question-oh",
    "Question-huh",
)
# The tag spelling of each engine (casefolded name -> spelling); an engine missing here knows no tags.
ENGINE_TAG_SPELLINGS: dict[str, dict[str, str]] = {
    "omnivoice": {tag.casefold(): tag for tag in NONVERBAL_TAGS},
    "mock": {tag.casefold(): tag for tag in NONVERBAL_TAGS},
    "ming": {tag.casefold(): tag for tag in MING_TAGS},
}
# ARPAbet phones as used by the CMU dictionary; vowels may carry a stress digit 0-2.
_ARPABET_VOWEL = r"(?:AA|AE|AH|AO|AW|AXR|AX|AY|EH|ER|EY|IH|IX|IY|OW|OY|UH|UW|UX)[0-2]?"
_ARPABET_CONSONANT = r"(?:B|CH|DH|DX|D|EL|EM|EN|F|G|HH|JH|K|L|M|NG|NX|N|P|Q|R|SH|S|TH|T|V|WH|W|Y|ZH|Z)"
_PHONE = rf"(?:{_ARPABET_VOWEL}|{_ARPABET_CONSONANT})"
# Both engines' tags, matched case-insensitively.
_ALL_TAGS = sorted({tag.casefold() for tag in (*NONVERBAL_TAGS, *MING_TAGS)})
_NONVERBAL = "|".join(regex.escape(tag) for tag in _ALL_TAGS)
_TAG_SPAN = rf"\[(?:(?i:{_NONVERBAL})|{_PHONE}(?: {_PHONE})*)\]"
# Not followed by "(" (a link) nor by a "[" that does not start another tag (a reference link).
_TAG_RE = regex.compile(rf"{_TAG_SPAN}(?!\()(?!(?!{_TAG_SPAN})\[)")
# Only the non-verbal tags (not ARPAbet). render_inline_tags runs after the Markdown stripper, so only "(" counts.
_NONVERBAL_RE = regex.compile(rf"\[((?i:{_NONVERBAL}))\](?!\()")
# Letters / digits of scripts that separate words with spaces (not Han, kana, Thai, ...).
_SPACED_WORD_CHAR = regex.compile(
    r"(?![\p{Han}\p{Hiragana}\p{Katakana}\p{Thai}\p{Lao}\p{Khmer}\p{Myanmar}])[\p{L}\p{N}]"
)
# A dropped tag before one of these (or at the start of a line) takes its spaces with it.
_CLOSING_PUNCT = frozenset(".,;:!?…)]}\"'»“”’\n")

# Candidate placeholder code points, in order. The post-processor's own URL markers
# are U+E000/U+E001, so these ranges are disjoint. No language pass matches private-use
# characters: they are neither letters, digits, marks, whitespace nor emoji.
_PLACEHOLDER_RANGES = ((0xF000, 0xF8FF), (0xF0000, 0xFFFFD))
_PRIVATE_USE_RE = regex.compile(r"[-\U000F0000-\U000FFFFD]")


def _free_placeholders(text: str, count: int) -> Optional[list[str]]:
    used = set(_PRIVATE_USE_RE.findall(text))
    free: list[str] = []
    for start, end in _PLACEHOLDER_RANGES:
        for cp in range(start, end + 1):
            ch = chr(cp)
            if ch not in used:
                free.append(ch)
                if len(free) == count:
                    return free
    return None


class InlineTagProtection:
    """The spans held out of one text and the placeholders that stand for them."""

    __slots__ = ("spans", "placeholders")

    def __init__(self, spans: list[str], placeholders: list[str]) -> None:
        self.spans = spans
        self.placeholders = placeholders

    def __len__(self) -> int:
        return len(self.spans)

    def restore(self, text: str) -> str:
        if not self.spans:
            return text
        lookup = dict(zip(self.placeholders, self.spans))
        pattern = regex.compile("[" + "".join(regex.escape(ch) for ch in self.placeholders) + "]")
        return pattern.sub(lambda m: lookup[m.group(0)], text)

    def count_unrestored(self, text: str) -> int:
        return sum(text.count(ch) for ch in self.placeholders)


def protect_inline_tags(text: str) -> Optional[tuple[str, InlineTagProtection]]:
    """Replace OmniVoice / Ming control spans with placeholders.

    Returns (protected text, protection), or None when there are not enough free
    placeholders for every span; the caller then skips optimization instead of
    running the pipeline on unprotected tags.
    """
    matches = list(_TAG_RE.finditer(text)) if "[" in text else []
    if not matches:
        return text, InlineTagProtection([], [])
    placeholders = _free_placeholders(text, len(matches))
    if placeholders is None:
        return None
    out: list[str] = []
    last = 0
    for m, placeholder in zip(matches, placeholders):
        out.append(text[last : m.start()])
        out.append(placeholder)
        last = m.end()
    out.append(text[last:])
    return "".join(out), InlineTagProtection([m.group(0) for m in matches], placeholders)


def restore_inline_tags(text: str, protection: InlineTagProtection) -> str:
    return protection.restore(text)


def render_inline_tags(text: str, engine: Optional[str]) -> str:
    """Spell every non-verbal tag the way ``engine`` reads it; drop the tags it does not know.

    Matching is case-insensitive (``[LAUGHTER]`` is ``[laughter]`` on OmniVoice and
    ``[Laughter]`` on Ming). Dropped tags take their surrounding spaces with them, leaving one
    space between two words (also when the tag had none, in scripts that use spaces). ARPAbet spans stay as they are. Text without tags comes back
    unchanged (the same string).
    """
    if "[" not in text:
        return text
    spellings = ENGINE_TAG_SPELLINGS.get((engine or "").strip().lower(), {})
    free = _free_placeholders(text, 1)
    if free is None:
        return text
    marker = free[0]
    dropped = False

    def spell(m: "regex.Match[str]") -> str:
        nonlocal dropped
        spelled = spellings.get(m.group(1).casefold())
        if spelled is None:
            dropped = True
            return marker
        return f"[{spelled}]"

    rendered = _NONVERBAL_RE.sub(spell, text)
    if dropped:
        spelled_text = rendered

        def close(m: "regex.Match[str]") -> str:
            before = spelled_text[m.start() - 1] if m.start() > 0 else "\n"
            after = spelled_text[m.end()] if m.end() < len(spelled_text) else "\n"
            if before == "\n" or after in _CLOSING_PUNCT:
                return ""
            if m.group(0).strip(marker):
                return " "
            # "Hallo[Cough]Welt" keeps two words; "你好[Cough]呀" needs no space.
            return " " if _SPACED_WORD_CHAR.match(before) and _SPACED_WORD_CHAR.match(after) else ""

        rendered = regex.sub(rf"[ \t]*(?:{regex.escape(marker)}[ \t]*)+", close, spelled_text)
    return text if rendered == text else rendered
