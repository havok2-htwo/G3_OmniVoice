"""Port of Genesis3 ``src/utils/ttsLanguageOptimizer/textOrthography.ts`` (Bugs 182, 424).

Does the text's own WRITING allow the fallback language? Reads orthography, never vocabulary:
the script, the alphabet (letters the language does not have in more than the odd loanword), and
for German one positive sign (ä/ö/ü/ß, a decimal comma, or German noun capitalisation).
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

from .jsre import js, js_replace

_WORD_PATTERN = js(r"[\p{L}\p{M}]+(?:['’][\p{L}\p{M}]+)*", "gu")
_LATIN_LETTER = js(r"\p{Script=Latin}", "u")
_LETTER = js(r"\p{L}", "u")
_ALPHABET_LETTER = {"de": js(r"[A-Za-zÄÖÜäöüß]", ""), "en": js(r"[A-Za-z]", "")}
_GERMAN_LETTER = js(r"[ÄÖÜäöüß]", "")
_DECIMAL_COMMA = js(r"(?<![\d.,])\d+,\d{1,2}(?![\d,])", "")
_SKIPPABLE_BEFORE_WORD = js(r"[\s\"'„“”‚‘’»«()[\]{}*_#>•·`–—-]", "u")
_SENTENCE_END = js(r"[.!?…]", "u")
_TECHNICAL_TOKEN = js(r"\S*(?:[@/\\_]|\p{L}\.\p{L})\S*", "gu")
_DIGIT_OR_UNDERSCORE = js(r"[\d_]", "")

_MIN_FOREIGN_ALPHABET_WORDS = 2
_MIN_FOREIGN_ALPHABET_SHARE = 1 / 8
_MIN_INNER_WORDS_FOR_CAPITALISATION = 6
_MIN_GERMAN_CAPITAL_SHARE = 0.12


def _starts_sentence(text: str, index: int) -> bool:
    i = index - 1
    while i >= 0 and _SKIPPABLE_BEFORE_WORD.test(text[i]):
        if text[i] == "\n":
            return True
        i -= 1
    if i < 0:
        return True
    return _SENTENCE_END.test(text[i])


def _is_all_upper_case(word: str) -> bool:
    return word == word.upper() and word != word.lower()


def _is_capitalised(word: str) -> bool:
    first = word[:1]
    return first != first.lower() and first == first.upper()


def _has_latin_letter_outside(word: str, alphabet) -> bool:
    return any(_LATIN_LETTER.test(char) and not alphabet.test(char) for char in word)


@dataclass
class TextOrthographyProfile:
    words: int = 0
    latin_letters: int = 0
    other_script_letters: int = 0
    words_outside_alphabet: dict = field(default_factory=lambda: {"de": 0, "en": 0})
    inner_words: int = 0
    capitalised_inner_words: int = 0


def profile_text_orthography(text: str) -> TextOrthographyProfile:
    source = js_replace(_TECHNICAL_TOKEN, unicodedata.normalize("NFC", text or ""), lambda token, *_r: " " * len(token))
    profile = TextOrthographyProfile()
    for char in source:
        if not _LETTER.test(char):
            continue
        if _LATIN_LETTER.test(char):
            profile.latin_letters += 1
        else:
            profile.other_script_letters += 1
    for match in _WORD_PATTERN.finditer(source):
        word = match.group(0)
        index = match.start()
        before = source[index - 1] if index > 0 else ""
        after = source[index + len(word)] if index + len(word) < len(source) else ""
        if _DIGIT_OR_UNDERSCORE.test(before) or _DIGIT_OR_UNDERSCORE.test(after):
            continue
        profile.words += 1
        if _has_latin_letter_outside(word, _ALPHABET_LETTER["de"]):
            profile.words_outside_alphabet["de"] += 1
        if _has_latin_letter_outside(word, _ALPHABET_LETTER["en"]):
            profile.words_outside_alphabet["en"] += 1
        if len(word) < 2 or _is_all_upper_case(word):
            continue
        if _starts_sentence(source, index):
            continue
        profile.inner_words += 1
        if _is_capitalised(word):
            profile.capitalised_inner_words += 1
    return profile


def text_orthography_allows_language(text: str, language: str) -> bool:
    """``textOrthographyAllowsLanguage``: the writing allows German/English as the fallback reading."""
    lang = "de" if language.startswith("de") else "en" if language.startswith("en") else None
    if not lang:
        return False
    source = unicodedata.normalize("NFC", text or "")
    profile = profile_text_orthography(source)
    if profile.other_script_letters > profile.latin_letters:
        return False
    outside = profile.words_outside_alphabet[lang]
    if outside >= _MIN_FOREIGN_ALPHABET_WORDS and outside >= profile.words * _MIN_FOREIGN_ALPHABET_SHARE:
        return False
    if lang == "en":
        return True
    if _GERMAN_LETTER.test(source) or _DECIMAL_COMMA.test(source):
        return True
    if profile.inner_words < _MIN_INNER_WORDS_FOR_CAPITALISATION:
        return True
    return profile.capitalised_inner_words >= profile.inner_words * _MIN_GERMAN_CAPITAL_SHARE
