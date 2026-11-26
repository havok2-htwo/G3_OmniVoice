"""Port of Genesis3 ``src/utils/ttsLanguageOptimizer/shared.ts``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

from .jsre import JS_WHITESPACE, JsRegExp, js, js_replace, js_trim

TtsOptimizeStage = Literal["floor", "final"]


@dataclass
class TtsAlgorithmOptimizationResult:
    text: str
    applied_rules: list[str]
    language: Optional[str]
    # Bug 1024: the day.month pairs a floor-stage pass held (keep_held_day_months), by placeholder slot.
    held_day_months: Optional[list[str]] = None


@dataclass(frozen=True)
class TtsOptimizeOptions:
    """``TtsOptimizeOptions`` of shared.ts (see there for each flag)."""

    stage: Optional[str] = None
    continues_numbered_list: bool = False
    starts_inside_line: bool = False
    answer_text: Optional[str] = None
    hold_list_opener: bool = False
    day_months_settled: bool = False
    keep_held_day_months: bool = False


# Bug 1024: the placeholder a floor-stage pass puts where it holds a day.month for the model, and its
# reversal. One private-use slot per pair, so at most HELD_DAY_MONTH_LIMIT per text.
HELD_DAY_MONTH_LIMIT = 256


def held_day_month_placeholder(slot: int) -> str:
    return f"\uE000{chr(0xE100 + slot)}\uE001"


_HELD_DAY_MONTH_PLACEHOLDER = js(r"\uE000([\uE100-\uE1FF])\uE001", "g")


def restore_held_day_months(text: str, held: Optional[list[str]]) -> str:
    if not held:
        return text

    def put_back(_placeholder, slot, *_rest):
        index = ord(slot) - 0xE100
        return held[index] if index < len(held) and held[index] is not None else ""

    return js_replace(_HELD_DAY_MONTH_PLACEHOLDER, text, put_back)


@dataclass(frozen=True)
class SpeakPass:
    """One deterministic replace rule (``SpeakPass`` in shared.ts).

    ``speak`` receives ``(match, *groups)`` with ``None`` for groups that did not
    participate and returns the spoken form, or ``None`` to keep the match.
    ``defer_in_floor`` marks occurrences the floor stage leaves in digits. The
    TS ``hint`` callbacks only feed the (not ported) hint injector.
    """

    label: str
    pattern: JsRegExp
    speak: Callable[..., Optional[str]]
    defer_in_floor: Optional[Callable[..., bool]] = field(default=None)


def replace_with_tracking(
    text: str,
    pattern: JsRegExp,
    replacement: str | Callable[..., str],
    label: str,
    applied_rules: list[str],
) -> str:
    """``replaceWithTracking``: replace and record ``label`` once if anything changed."""
    did_change = False

    def callback(*args):
        nonlocal did_change
        original = args[0] if args[0] is not None else ""
        replaced = replacement(*args) if callable(replacement) else replacement
        if original != replaced:
            did_change = True
        return replaced

    result = js_replace(pattern, text, callback)
    if did_change and label not in applied_rules:
        applied_rules.append(label)
    return result


_EMOJI_KEYCAP = js(r"([0-9#*])️?⃣", "gu")
_EMOJI_SEQUENCE = js(
    r"(?:\p{Extended_Pictographic}|[\u{1F3FB}-\u{1F3FF}]|[\u{1F1E6}-\u{1F1FF}])"
    r"(?:[️︎\u200D]|\p{Extended_Pictographic}|[\u{1F3FB}-\u{1F3FF}]|[\u{1F1E6}-\u{1F1FF}])*",
    "gu",
)


def strip_emoji_for_tts(text: str) -> str:
    source = text or ""
    if not source:
        return source
    return js_replace(_EMOJI_SEQUENCE, js_replace(_EMOJI_KEYCAP, source, "$1"), " ")


# "Pst" in its spellings (Psst, Pscht, Pschhh) has no vowel to voice (Bug 857). Lower-case letters
# only after the P, so the time zone "PST" stays.
SHUSHING_SOURCE = "[Pp]s+(?:t+|ch+t?)"
_SHUSHING = js(
    rf"(?<![\p{{L}}\p{{N}}-])(?:{SHUSHING_SOURCE}(?![\p{{L}}\p{{N}}]|[-_./\\][\p{{L}}\p{{N}}])[.…!?,:;]*(?:[ \t]+[–—-])?[ \t]*)+(\p{{Ll}})?",
    "gu",
)
_ONLY_SHUSHING = js(rf"^(?:[\s.…!?,:;]*{SHUSHING_SOURCE})+[\s.…!?,:;]*$", "u")
_TRAILING_BLANKS = js(r"[ \t]+$", "")
_ENDS_SENTENCE = js(r"[.!?…:\n]$", "")
_LETTER_OR_NUMBER = js(r"[\p{L}\p{N}]", "u")


def is_only_shushing(text: str) -> bool:
    return _ONLY_SHUSHING.test(text)


def skip_shushing_for_tts(text: str) -> str:
    """``skipShushingForTts``: "Pst… hörst du das?" as "Hörst du das?"."""

    def callback(_match, next_char, offset, source):
        if not next_char:
            return ""
        before = js_replace(_TRAILING_BLANKS, source[:offset], "")
        return next_char.upper() if not before or _ENDS_SENTENCE.test(before) else next_char

    skipped = js_replace(_SHUSHING, text, callback)
    return skipped if _LETTER_OR_NUMBER.test(skipped) else text


# What a status mark that opens a list item says (Bug 826): 'positive' | 'negative' | 'warning'.
_STATUS_MARKS = {
    "✅": "positive", "✔": "positive", "☑": "positive", "✓": "positive",
    "❌": "negative", "✖": "negative", "✗": "negative", "⛔": "negative", "🚫": "negative",
    "⚠": "warning",
}
_NON_PICTOGRAPHIC_STATUS_MARK = js(r"[✓✗]", "")
_VARIATION_SELECTOR_EMOJI = "️"
_EMOJI_SEQUENCE_AT = js(_EMOJI_SEQUENCE.source, "uy")
_EMOJI_PRESENTATION = js(r"\p{Emoji_Presentation}", "u")
_LINE_PREFIX = js(r"^([ \t]*)([-+*•·][ \t]+)?", "")
_LIST_ITEM_LINE = js(r"^([ \t]*)(?:[-+*•·]|\d{1,2}[.)])[ \t]+\S", "")


@dataclass(frozen=True)
class StatusMark:
    status: str
    glyph: str
    length: int


def status_mark_at(text: str, start: int) -> Optional[StatusMark]:
    """``statusMarkAt``: the status mark at ``start`` with its optional U+FE0F, or None."""
    source = text or ""
    if start < 0 or start >= len(source):
        return None
    glyph = source[start]
    status = _STATUS_MARKS.get(glyph)
    if not status:
        return None
    length = 1 + (1 if source[start + 1 : start + 2] == _VARIATION_SELECTOR_EMOJI else 0)
    return StatusMark(status, glyph, length)


def _emoji_sequence_end(text: str, start: int) -> int:
    m = _EMOJI_SEQUENCE_AT.match_at(text, start)
    return start + len(m.group(0)) if m else start


def _is_shown_as_emoji(sequence: str) -> bool:
    for character in sequence:
        if character == _VARIATION_SELECTOR_EMOJI or ord(character) >= 0x1F000 or _EMOJI_PRESENTATION.test(character):
            return True
    return False


def _introduces_list(lines: list[str], index: int) -> bool:
    prefix = _LINE_PREFIX.search(lines[index])
    own = len(prefix.group(1)) if prefix else 0
    for line in lines[index + 1 :]:
        if not js_trim(line):
            continue
        item = _LIST_ITEM_LINE.search(line)
        return bool(item and len(item.group(1)) >= own)
    return False


_LEADING_BLANKS = js(r"^[ \t]+", "")


def _strip_emoji_line(line: str, opens_list: bool) -> str:
    prefix = _LINE_PREFIX.search(line)
    indentation = (prefix.group(1) if prefix else "") or ""
    bullet = (prefix.group(2) if prefix else "") or ""
    start = len(prefix.group(0)) if prefix else 0
    mark = status_mark_at(line, start)
    end = _emoji_sequence_end(line, start + (mark.length if mark else 0))
    if end == start:
        return strip_emoji_for_tts(line)
    rest = js_replace(_LEADING_BLANKS, strip_emoji_for_tts(line[end:]), "")
    if not js_trim(rest):
        return strip_emoji_for_tts(line)
    if not mark and not _is_shown_as_emoji(line[start:end]):
        return f"{indentation}{bullet}{rest}"
    if not mark and not bullet and opens_list:
        return f"{indentation}{rest}"
    if not bullet:
        bullet = "- "
    return f"{indentation}{bullet}{mark.glyph} {rest}" if mark else f"{indentation}{bullet}{rest}"


def strip_emoji_lines_for_tts(text: str) -> str:
    """``stripEmojiLinesForTts``: strip_emoji_for_tts, keeping the lines of the text (Bug 826)."""
    source = text or ""
    if not source:
        return source
    if strip_emoji_for_tts(source) == source and not _NON_PICTOGRAPHIC_STATUS_MARK.test(source):
        return source
    lines = source.split("\n")
    return "\n".join(_strip_emoji_line(line, _introduces_list(lines, index)) for index, line in enumerate(lines))


_QUOTE_BETWEEN_WORD_CHARS = js(r"(?<=[\p{L}\p{N}])[\"„“”«»‚‹›](?=[\p{L}\p{N}])", "gu")
_QUOTE_MARK = js(r"[\"„“”«»‚‹›]", "g")


def strip_quotation_marks_for_tts(text: str) -> str:
    source = text or ""
    if not source:
        return source
    return js_replace(_QUOTE_MARK, js_replace(_QUOTE_BETWEEN_WORD_CHARS, source, " "), "")


# The last step of the rule sets and of the post-processor: a run of whitespace that holds a line
# break becomes ONE line break, not a space (line structure is what the line-anchored rules read).
WHITESPACE_RUN_WITH_LINE_BREAK = js(r"(?<!\s)\s*\n\s*", "g")

# Every kind of line break as "\n" ("\r\n", a lone "\r", U+2028, U+2029).
ANY_LINE_BREAK = js(r"\r\n?|[\u2028\u2029]", "g")


def unify_line_breaks_for_tts(text: str) -> str:
    return js_replace(ANY_LINE_BREAK, text or "", "\n")


# Invisible characters (Bug 836): terminal escapes, controls, format characters except the zero
# width joiner, and the horizontal Unicode spaces as a plain space.
_TERMINAL_ESCAPE = js(r"\x1B(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1B]*(?:\x07|\x1B\\))", "g")
_LINE_ENDING_CONTROL = js(r"[\x0B\x0C\x85]", "g")
_INVISIBLE_CHARACTER = js(r"[\x00-\x08\x0E-\x1F\x7F-\x84\x86-\x9F]|(?!\u200D)\p{Cf}", "gu")
_UNICODE_SPACE = js(r"[\u00A0\u1680\u2000-\u200A\u202F\u205F\u3000]", "g")
_ANY_INVISIBLE = js(r"[\x00-\x08\x0B-\x1F\x7F-\x9F\u00A0\u1680\u2000-\u200A\u202F\u205F\u3000]|(?!\u200D)\p{Cf}", "u")


def remove_invisible_characters_for_tts(text: str) -> str:
    source = text or ""
    if not _ANY_INVISIBLE.test(source):
        return source
    source = js_replace(_TERMINAL_ESCAPE, source, "")
    source = js_replace(_LINE_ENDING_CONTROL, source, "\n")
    source = js_replace(_INVISIBLE_CHARACTER, source, "")
    return js_replace(_UNICODE_SPACE, source, " ")


def remove_zero_width_joiners_for_tts(text: str) -> str:
    source = text or ""
    return source.replace("\u200d", "") if "\u200d" in source else source


_INDENTED_LIST_ITEM = js(r"^[ \t]+(?:[-+*•·]|\d{1,2}[.)])[ \t]+\S", "")


def js_trim_start(text: str) -> str:
    return text.lstrip(JS_WHITESPACE)


def js_trim_end(text: str) -> str:
    return text.rstrip(JS_WHITESPACE)


def trim_keeping_list_item_depth_for_tts(text: str) -> str:
    """``trimKeepingListItemDepthForTts``: a list item that opens the text keeps its indentation (Bug 825)."""
    trimmed = js_trim_end(text or "")
    leading = trimmed[: len(trimmed) - len(js_trim_start(trimmed))]
    rest = trimmed[leading.rfind("\n") + 1 :]
    return rest if _INDENTED_LIST_ITEM.test(rest) else js_trim_start(rest)


# A space before a punctuation mark goes, except before a dot that opens a name (".env"), a relative
# path ("./deploy.sh"), a colon before a digit (":8081") and "!" before a letter, digit, "=" or "("
# (Bugs 832, 836, 838).
_SPACE_BEFORE_PUNCTUATION = js(r"[^\S\n]+([,;?]|!(?![\p{L}\p{N}=(])|:(?!\d)|\.(?![\p{L}\p{N}]|\.?\/))", "gu")
_TRAILING_EXTENSION = js(r"\s*\.[A-Za-z0-9]{2,8}$", "")
_DOT_NAME = js(r"(?<!\S)\.[\p{L}\p{N}]", "u")


def holds_dot_name_before_extension(filename: str) -> bool:
    """``holdsDotNameBeforeExtension`` (Bug 832)."""
    name = js_replace(_TRAILING_EXTENSION, filename or "", "")
    return _DOT_NAME.test(name)


_MULTI_BLANKS = js(r"[^\S\n]{2,}", "g")


def finalize_optimized_text(text: str) -> str:
    result = js_replace(WHITESPACE_RUN_WITH_LINE_BREAK, text or "", "\n")
    result = js_replace(_MULTI_BLANKS, result, " ")
    return js_trim(js_replace(_SPACE_BEFORE_PUNCTUATION, result, "$1"))


# The rule the floor reports when it spoke a single "1." that opens a list (Bug 846).
TTS_LIST_OPENER_RULE = "list_opener"
# Bug 1024: the floor stage kept a day.month in digits for the optimizer model to decide.
TTS_DAY_MONTH_HELD_RULE = "day_month_held_de"

# A list item behind its marker, on the same line (Bug 846).
_LIST_ITEM_START = js(r'^\d{1,2}\.[ \t]+["„“(]?[\p{L}\p{N}]', "u")

# A "1." that opens a line and runs on to a second dotted number: a range, a coordination or a date
# written with spaces; none of them opens a list (Bug 846).
LIST_OPENER_PAIR = js(r"^[ \t]*1\.[ \t]*(?:[^\p{L}\p{N}\s]{1,2}[ \t]*)?(?:\p{Ll}+[ \t]+)*\d{1,2}\.(?!\d)", "mu")


_ORDERED_MARKER = js(r"(\d{1,2})\.\s+(?=\S)", "g")
_WORDISH_CHAR = js(r"[\p{L}\p{N}_]", "u")


def _ordered_sequence_boundary_before(source: str, marker_start: int) -> Optional[str]:
    cursor = marker_start - 1
    crossed_line_break = False
    while cursor >= 0 and source[cursor] in JS_WHITESPACE:
        if source[cursor] in ("\n", "\r"):
            crossed_line_break = True
        cursor -= 1
    if crossed_line_break:
        return "line"
    if cursor < 0:
        return "start"
    prior = source[cursor]
    if prior == ":":
        return "colon"
    if prior in (".", "!", "?"):
        return "sentence"
    if prior == ";":
        return "clause"
    return None


def replace_ordered_sequence_markers(
    text: str,
    render_ordinal: Callable[[int], str],
    label: str,
    applied_rules: list[str],
    *,
    keep_lone_marker: bool = False,
    starts_inside_line: bool = False,
    hold_list_opener: bool = False,
) -> str:
    """``replaceOrderedSequenceMarkers``: digit-dot list markers to discourse ordinals."""
    source = text or ""
    candidates: list[dict] = []
    for m in _ORDERED_MARKER.finditer(source):
        start = m.start()
        preceding = source[start - 1] if start > 0 else ""
        if preceding and _WORDISH_CHAR.test(preceding):
            continue
        boundary = _ordered_sequence_boundary_before(source, start)
        value = int(m.group(1))
        if not boundary or value < 1 or value > 99:
            continue
        candidates.append({"start": start, "end": m.end(), "value": value, "boundary": boundary})

    selected: set[int] = set()
    for index, candidate in enumerate(candidates):
        if candidate["boundary"] in ("start", "line"):
            selected.add(index)

    run_start = 0
    while run_start < len(candidates):
        run_end = run_start + 1
        while run_end < len(candidates) and candidates[run_end]["value"] == candidates[run_end - 1]["value"] + 1:
            run_end += 1
        first = candidates[run_start]
        anchored = first["boundary"] in ("start", "line", "colon") or (
            first["boundary"] == "sentence" and first["value"] == 1
        )
        if run_end - run_start >= 2 and anchored:
            selected.update(range(run_start, run_end))
        run_start = run_end

    if not selected:
        return source
    lone = len(selected) == 1 and keep_lone_marker
    list_opener = (
        lone
        and not hold_list_opener
        and any(
            index in selected
            and candidate["value"] == 1
            and _LIST_ITEM_START.test(source[candidate["start"] :])
            and (candidate["boundary"] == "line" or (candidate["boundary"] == "start" and not starts_inside_line))
            for index, candidate in enumerate(candidates)
        )
    )
    if lone and not list_opener:
        return source
    result = source
    for index in sorted(selected, reverse=True):
        candidate = candidates[index]
        ordinal = js_trim(render_ordinal(candidate["value"]) or "")
        if not ordinal:
            continue
        result = f"{result[:candidate['start']]}{ordinal}: {result[candidate['end']:]}"
    rule = TTS_LIST_OPENER_RULE if list_opener else label
    if result != source and rule not in applied_rules:
        applied_rules.append(rule)
    return result


# Bug 840: the last part of a version that stands for any number ("18.x", "v3.*") and the tags of a
# pre-release suffix ("-beta.2", "-rc1", "-SNAPSHOT").
VERSION_WILDCARD_SOURCE = r"[x*](?![\w*]|\.\d)"
VERSION_PRE_RELEASE_TAG_SOURCE = "(?:alpha|beta|rc|preview|pre|dev|snapshot)"

# An e-mail address (Bug 969). Groups: 1 local part, 2 host.
EMAIL_ADDRESS_PATTERN_SOURCE = (
    r"(?<![\p{L}\p{N}_.])(?:mailto:)?([\p{L}\p{N}_+-]+(?:\.[\p{L}\p{N}_+-]+)*)@((?:[\p{L}\p{N}_-]+\.)+\p{L}{2,})"
    r"(?![\p{L}\p{N}_])(?:\?[\p{L}\p{N}_%.-]+=(?:[^\s<>()[\]]*[^\s<>()[\].,;:!?])?)?"
)

PHONE_NUMBER_PATTERN_SOURCE = (
    r"(?<![\p{L}\d.,/@:-])(?:\+\d{1,4}(?:[ \t]*\((?:0|\d{2,5})\))?(?:[ \t]*\d{1,12})?|\(0\)[ \t]*\d{1,12}"
    r"|\(0\d{1,5}\)|\(\d{3,5}\)|0\d{1,5})(?:(?:[ \t]*[/-][ \t]*|[ \t]+)\d{1,12})+(?![\p{L}\d_])"
)

_PHONE_STRIP = js(r"[+()]", "g")
_PHONE_SPLIT = js(r"[\s/-]+", "")
_ALL_DIGITS = js(r"^\d+$", "")


def parse_phone_number_match(match: str) -> Optional[tuple[bool, str]]:
    """``parsePhoneNumberMatch`` -> (plus, digits) or None."""
    raw = js_trim(match or "")
    plus = raw.startswith("+")
    groups = [part for part in _PHONE_SPLIT.split(js_trim(js_replace(_PHONE_STRIP, raw, " "))) if part]
    if len(groups) < 2 or any(not _ALL_DIGITS.test(part) for part in groups):
        return None
    digits = "".join(groups)
    if len(digits) < 7 or len(digits) > 16:
        return None
    return plus, digits
