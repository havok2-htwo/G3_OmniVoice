"""Protect OmniVoice inline control spans from the text optimizer.

OmniVoice reads exactly two bracketed forms as control syntax: the 13 non-verbal
tags of ``omnivoice.models.omnivoice._NONVERBAL_PATTERN`` (``[laughter]``,
``[sigh]``, ...) and CMU/ARPAbet pronunciation overrides such as ``[B EY1 S]``.
The Genesis pipeline would turn them into spoken words ("laughter", "B EY 1 S"),
so they are swapped for private-use placeholders before the pipeline runs and
restored verbatim afterwards. Every other bracket span (citations ``[1]``,
editorial notes ``[ca. 5 €]``, Markdown checkboxes ``- [x]``) is ordinary text
and goes through the pipeline exactly like in the TS reference.

A bracket directly followed by ``(`` or ``[`` is a Markdown link / reference
and is left to the Markdown stripper.

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
# ARPAbet phones as used by the CMU dictionary; vowels may carry a stress digit 0-2.
_ARPABET_VOWEL = r"(?:AA|AE|AH|AO|AW|AXR|AX|AY|EH|ER|EY|IH|IX|IY|OW|OY|UH|UW|UX)[0-2]?"
_ARPABET_CONSONANT = r"(?:B|CH|DH|DX|D|EL|EM|EN|F|G|HH|JH|K|L|M|NG|NX|N|P|Q|R|SH|S|TH|T|V|WH|W|Y|ZH|Z)"
_PHONE = rf"(?:{_ARPABET_VOWEL}|{_ARPABET_CONSONANT})"
_NONVERBAL = "|".join(regex.escape(tag) for tag in NONVERBAL_TAGS)
_TAG_RE = regex.compile(rf"\[(?:(?:{_NONVERBAL})|{_PHONE}(?: {_PHONE})*)\](?![(\[])")

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
    """Replace OmniVoice control spans with placeholders.

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
