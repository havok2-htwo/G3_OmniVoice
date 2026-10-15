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
    r"(?:[️︎‍]|\p{Extended_Pictographic}|[\u{1F3FB}-\u{1F3FF}]|[\u{1F1E6}-\u{1F1FF}])*",
    "gu",
)


def strip_emoji_for_tts(text: str) -> str:
    source = text or ""
    if not source:
        return source
    return js_replace(_EMOJI_SEQUENCE, js_replace(_EMOJI_KEYCAP, source, "$1"), " ")


_QUOTE_BETWEEN_WORD_CHARS = js(r"(?<=[\p{L}\p{N}])[\"„“”«»‚‹›](?=[\p{L}\p{N}])", "gu")
_QUOTE_MARK = js(r"[\"„“”«»‚‹›]", "g")


def strip_quotation_marks_for_tts(text: str) -> str:
    source = text or ""
    if not source:
        return source
    return js_replace(_QUOTE_MARK, js_replace(_QUOTE_BETWEEN_WORD_CHARS, source, " "), "")


_MULTI_WS = js(r"\s{2,}", "g")
_SPACE_BEFORE_PUNCT = js(r"\s+([,.;:!?])", "g")


def finalize_optimized_text(text: str) -> str:
    return js_trim(js_replace(_SPACE_BEFORE_PUNCT, js_replace(_MULTI_WS, text or "", " "), "$1"))


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
    if keep_lone_marker and len(selected) == 1:
        return source
    result = source
    for index in sorted(selected, reverse=True):
        candidate = candidates[index]
        ordinal = js_trim(render_ordinal(candidate["value"]) or "")
        if not ordinal:
            continue
        result = f"{result[:candidate['start']]}{ordinal}: {result[candidate['end']:]}"
    if result != source and label not in applied_rules:
        applied_rules.append(label)
    return result


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
