"""Port of Genesis3 ``src/utils/ttsLanguageOptimizer/notationPrepass.ts``.

Rewrites notation that carries a spoken word (range dashes, "per" slashes, signs, symbols, glued
codes, footnote marks, coordinates, e-mail addresses, weekday rows) into that word BEFORE the
value passes of de/en turn digits into number words. Rule labels match the TS source.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

from .jsre import is_js_integer, js, js_number, js_replace, js_replace_first_literal, js_trim
from .pair_context import GERMAN_PAIR_CONTEXT
from .shared import EMAIL_ADDRESS_PATTERN_SOURCE, replace_with_tracking

NotationLanguage = Literal["de", "en"]


@dataclass(frozen=True)
class NotationPrepassConfig:
    lang: NotationLanguage
    integer_words: Callable[[int], str]
    digit_words: Callable[[str], str]
    year_words: Callable[[int], str]
    # The unit symbols of the language as a regex alternation (Bug 838).
    unit_symbol_source: Optional[str] = None
    # The unit symbols (in their order) and unit words: a power or footnote mark after one stays (Bug 839).
    unit_symbols: tuple = field(default_factory=tuple)
    unit_words: frozenset = field(default_factory=frozenset)
    # The number labels ("Kap.", "Art.", "S.") as a regex alternation (Bug 839).
    number_label_source: Optional[str] = None
    # How one label of an e-mail address or a domain is written for the engine (Bug 969).
    address_label: Optional[Callable[[str, bool], str]] = None


def _fractions(pairs):
    return {glyph: {"alone": alone, "afterInteger": after} for glyph, alone, after in pairs}


WORDS: dict[str, dict] = {
    "de": {
        "rangeTo": "bis",
        "and": "und",
        "per": "pro",
        "dot": "punkt",
        "at": "at",
        "plus": "plus",
        "minus": "minus",
        "plusMinus": "plus minus",
        "minusPlus": "minus plus",
        "about": "etwa",
        "atLeast": "mindestens",
        "atMost": "höchstens",
        "greaterThan": "mehr als",
        "lessThan": "weniger als",
        "greaterThanRelation": "größer als",
        "lessThanRelation": "kleiner als",
        "notEqual": "ungleich",
        "or": "oder",
        "modulo": "modulo",
        "resultWordSource": "(?:gleich|ergibt|ist)",
        "symbolNounSource": "(?:[Pp]feil|[Oo]perator|[Zz]eichen|[Ss]ymbol)",
        "operatorWordSource": "(?:plus|minus|gleich|mal|durch|hoch)",
        "digitWordSource": "(?:null|eins|zwei|drei|vier|fünf|sechs|sieben|acht|neun)",
        "number": "Nummer",
        "section": "Paragraf",
        "sections": "Paragrafen",
        "star": "Stern",
        "percent": "Prozent",
        "degree": "Grad",
        "degrees": "Grad",
        "degreeOne": "ein",
        "oneCount": "eine",
        "minute": {"singular": "Minute", "plural": "Minuten"},
        "second": {"singular": "Sekunde", "plural": "Sekunden"},
        "compassPoints": {"N": "Nord", "S": "Süd", "E": "Ost", "O": "Ost", "W": "West"},
        "mathSymbols": {
            "√": "Wurzel aus", "∛": "Kubikwurzel aus", "∜": "vierte Wurzel aus", "∞": "unendlich", "∑": "Summe",
            "∏": "Produkt", "∫": "Integral", "∆": "Delta", "Δ": "Delta", "∈": "Element von",
        },
        "cents": "Cent",
        "yen": "Yen",
        "arrowTo": "nach",
        "ratio": "zu",
        "outOf": "von",
        "spanUnitSource": "(?:[Ss]tunden?|[Ss]td|h|[Mm]inuten?|[Mm]in|[Ss]ekunden?|[Ss]ek|sec|s)",
        "clockHour": "Uhr",
        "port": "Port",
        "equals": "gleich",
        "dividedBy": "durch",
        "times": "mal",
        "atSign": "bei",
        "careOf": "per Adresse",
        "power": "hoch",
        "hashKey": "Raute",
        "periods": {
            "s": "Sekunde", "sek": "Sekunde", "sek.": "Sekunde", "sec": "Sekunde",
            "min": "Minute", "min.": "Minute",
            "h": "Stunde", "std": "Stunde", "std.": "Stunde", "stunde": "Stunde",
            "tag": "Tag", "woche": "Woche", "monat": "Monat", "jahr": "Jahr", "a": "Jahr",
            "s²": "Quadratsekunde",
            "l": "Liter", "ml": "Milliliter", "dl": "Deziliter",
            "kg": "Kilogramm", "g": "Gramm", "km": "Kilometer", "m": "Meter",
            "m²": "Quadratmeter", "m2": "Quadratmeter", "m³": "Kubikmeter", "m3": "Kubikmeter",
            "kwh": "Kilowattstunde", "stk": "Stück", "stk.": "Stück", "person": "Person", "nacht": "Nacht",
        },
        "weekdays": {"mo": "Montag", "di": "Dienstag", "mi": "Mittwoch", "do": "Donnerstag", "fr": "Freitag", "sa": "Samstag", "so": "Sonntag"},
        "ambiguousWeekdays": ["so"],
        "colonSentenceOpeners": ["so"],
        "unicodeFractions": _fractions((
            ("½", "ein halb", "einhalb"), ("¼", "ein Viertel", "einviertel"), ("¾", "drei Viertel", "dreiviertel"),
            ("⅓", "ein Drittel", "eindrittel"), ("⅔", "zwei Drittel", "zweidrittel"), ("⅛", "ein Achtel", "einachtel"),
            ("⅜", "drei Achtel", "dreiachtel"), ("⅝", "fünf Achtel", "fünfachtel"), ("⅞", "sieben Achtel", "siebenachtel"),
            ("⅕", "ein Fünftel", "einfünftel"), ("⅖", "zwei Fünftel", "zweifünftel"), ("⅗", "drei Fünftel", "dreifünftel"),
            ("⅘", "vier Fünftel", "vierfünftel"), ("⅙", "ein Sechstel", "einsechstel"), ("⅚", "fünf Sechstel", "fünfsechstel"),
            ("⅐", "ein Siebtel", "einsiebtel"), ("⅑", "ein Neuntel", "einneuntel"), ("⅒", "ein Zehntel", "einzehntel"),
        )),
    },
    "en": {
        "rangeTo": "to",
        "and": "and",
        "per": "per",
        "dot": "dot",
        "at": "at",
        "plus": "plus",
        "minus": "minus",
        "plusMinus": "plus or minus",
        "minusPlus": "minus or plus",
        "about": "about",
        "atLeast": "at least",
        "atMost": "at most",
        "greaterThan": "more than",
        "lessThan": "less than",
        "greaterThanRelation": "greater than",
        "lessThanRelation": "less than",
        "notEqual": "not equal to",
        "or": "or",
        "modulo": "mod",
        "resultWordSource": "(?:equals|is)",
        "symbolNounSource": "(?:[Aa]rrow|[Oo]perator|[Ss]ign|[Ss]ymbol)",
        "operatorWordSource": "(?:plus|minus|equals|times|divided by|to the power of)",
        "digitWordSource": "(?:zero|one|two|three|four|five|six|seven|eight|nine)",
        "number": "number",
        "section": "section",
        "sections": "sections",
        "star": "star",
        "percent": "percent",
        "degree": "degree",
        "degrees": "degrees",
        "degreeOne": "one",
        "oneCount": "one",
        "minute": {"singular": "minute", "plural": "minutes"},
        "second": {"singular": "second", "plural": "seconds"},
        "compassPoints": {"N": "north", "S": "south", "E": "east", "W": "west"},
        "mathSymbols": {
            "√": "square root of", "∛": "cube root of", "∜": "fourth root of", "∞": "infinity", "∑": "sum",
            "∏": "product", "∫": "integral", "∆": "delta", "Δ": "delta", "∈": "in", "∂": "partial",
        },
        "cents": "cents",
        "yen": "yen",
        "arrowTo": "to",
        "ratio": "to",
        "outOf": "out of",
        "spanUnitSource": "(?:[Mm]in(?:utes?|s)?|[Hh](?:ou)?rs?|[Hh]ours?|h)",
        "clockHour": "o'clock",
        "port": "port",
        "equals": "equals",
        "dividedBy": "divided by",
        "times": "times",
        "atSign": "at",
        "careOf": "care of",
        "power": "to the power of",
        "hashKey": "hash",
        "periods": {
            "s": "second", "sec": "second", "sec.": "second",
            "min": "minute", "min.": "minute",
            "h": "hour", "hr": "hour", "hr.": "hour", "hour": "hour",
            "day": "day", "week": "week", "wk": "week", "mo": "month", "month": "month", "yr": "year", "yr.": "year",
            "year": "year", "a": "year",
            "l": "liter", "ml": "milliliter", "dl": "deciliter", "gal": "gallon",
            "kg": "kilogram", "g": "gram", "km": "kilometer", "m": "meter", "mi": "mile", "mile": "mile",
            "m²": "square meter", "m2": "square meter", "m³": "cubic meter", "m3": "cubic meter",
            "kwh": "kilowatt-hour", "person": "person", "night": "night", "unit": "unit", "pc": "piece", "pcs": "piece",
        },
        "weekdays": {
            "mon": "Monday", "tue": "Tuesday", "tues": "Tuesday", "wed": "Wednesday", "thu": "Thursday", "thur": "Thursday",
            "thurs": "Thursday", "fri": "Friday", "sat": "Saturday", "sun": "Sunday",
        },
        "ambiguousWeekdays": ["sun", "sat", "wed", "mon"],
        "colonSentenceOpeners": [],
        "unicodeFractions": _fractions((
            ("½", "one half", "and a half"), ("¼", "one quarter", "and a quarter"),
            ("¾", "three quarters", "and three quarters"), ("⅓", "one third", "and a third"),
            ("⅔", "two thirds", "and two thirds"), ("⅛", "one eighth", "and an eighth"),
            ("⅜", "three eighths", "and three eighths"), ("⅝", "five eighths", "and five eighths"),
            ("⅞", "seven eighths", "and seven eighths"), ("⅕", "one fifth", "and a fifth"),
            ("⅖", "two fifths", "and two fifths"), ("⅗", "three fifths", "and three fifths"),
            ("⅘", "four fifths", "and four fifths"), ("⅙", "one sixth", "and a sixth"),
            ("⅚", "five sixths", "and five sixths"), ("⅐", "one seventh", "and a seventh"),
            ("⅑", "one ninth", "and a ninth"), ("⅒", "one tenth", "and a tenth"),
        )),
    },
}

ENGLISH_DECADES = {
    "10": "tens", "20": "twenties", "30": "thirties", "40": "forties", "50": "fifties",
    "60": "sixties", "70": "seventies", "80": "eighties", "90": "nineties",
}

_DOTS = js(r"\.", "g")
_COMMAS = js(r",", "g")
_NON_DIGITS = js(r"\D", "g")
_REGEX_SPECIALS = js(r"[.*+?^${}()|[\]\\]", "g")
_ALL_DIGITS = js(r"^\d+$", "")
NEVER_SOURCE = "(?!)"


def escape_for_regexp(value: str) -> str:
    return js_replace(_REGEX_SPECIALS, value, "\\$&")


def _line_index(source: str, offset: int) -> int:
    """``source.slice(0, offset).split('\\n').length - 1``."""
    return source.count("\n", 0, offset)


def _count_words(token: str, one_word: str, singular: str, plural: str, cfg: NotationPrepassConfig) -> str:
    """A count of a coordinate part (Bug 839)."""
    if not _ALL_DIGITS.test(token):
        return f"{token} {plural}"
    value = int(token)
    return f"{one_word} {singular}" if value == 1 else f"{cfg.integer_words(value)} {plural}"


def _parse_operand_value(token: str, lang: str):
    raw = token or ""
    if ":" in raw:
        return None
    if lang == "de":
        normalized = js_replace_first_literal(js_replace(_DOTS, raw, ""), ",", ".")
    else:
        normalized = js_replace(_COMMAS, raw, "")
    value = js_number(normalized)
    return value if value == value and value not in (float("inf"), float("-inf")) else None


# --- ranges -----------------------------------------------------------------------------------

RANGE_PATTERN = js(
    r"(?<![\p{L}\d.,:/–—+)-])(?:(?<=((?:\((?:0\d{2,5}|\d{3})\)|\+\d{1,4}(?:[ \t]*\(0\)[ \t]*\d{1,5})?|(?<![\p{L}\d.,])0\d{2,5})"
    r"(?:(?:[ \t]*\/[ \t]*|[ \t]+)\d{1,12}){0,4})[ \t]+)|)([€$£]?)(\d(?:[\d.,:]*\d)?)(\.?)([ \t]?(?:%|°[CF]?|€|\$|£|Uhr|[ap]\.?m\.?))?"
    r"([ \t]*)([–—-])([ \t]*)([€$£]?)(\d(?:[\d.,:]*\d)?)(?![\d.,:]*[–—-]\d)(?![ \t]+\d)(?=([ \t]?(?:Uhr|[ap]\.?m\.?)\b)?)",
    "giu",
)
_CLOCK_TIME = js(r"^\d{1,2}:\d{2}(?::\d{2})?$", "")
_CLOCK_UNIT = js(r"Uhr|m\.?$", "i")


def _is_clock_time(token: str) -> bool:
    return _CLOCK_TIME.test(token)


def _digits_of(token: str) -> str:
    return js_replace(_NON_DIGITS, token or "", "")


def _is_range_date(day, month, year) -> bool:
    """A possible day and month of a date range; a short year only after two-digit day and month (Bug 842)."""
    day_value = js_number(day)
    month_value = js_number(month)
    if not (1 <= day_value <= 31 and 1 <= month_value <= 12):
        return False
    return year is None or len(year) != 2 or (len(str(day)) == 2 and len(str(month)) == 2)


_PHONE_MIN_DIGITS = 7
_PHONE_MAX_DIGITS = 16


def _is_phone_number_end(phone_prefix, left: str, right: str) -> bool:
    """Whether a pair ends a phone number (Bug 840)."""
    if phone_prefix is None:
        return False
    digits = len(_digits_of(phone_prefix + left + right))
    return _PHONE_MIN_DIGITS <= digits <= _PHONE_MAX_DIGITS


_ELLIPSIS_RANGE_PATTERN = {
    "de": js(
        r"(?<![\w.,:+\-−]|(?:\.\.|…)[ \t]*)([-−]?)(\d{1,2}:\d{2}|\d+(?:,\d+)?)[ \t]*(?:\.{2,3}|…)[ \t]*"
        r"(?=([-−]?)(\d{1,2}:\d{2}|\d+(?:,\d+)?)(?![\d,:]|\.\d|[ \t]*(?:\.{2,3}|…)[ \t]*[-−]?\d))",
        "g",
    ),
    "en": js(
        r"(?<![\w.,:+\-−]|(?:\.\.|…)[ \t]*)([-−]?)(\d{1,2}:\d{2}|\d+(?:\.\d+)?)[ \t]*(?:\.{2,3}|…)[ \t]*"
        r"(?=([-−]?)(\d{1,2}:\d{2}|\d+(?:\.\d+)?)(?![\d,:]|\.\d|[ \t]*(?:\.{2,3}|…)[ \t]*[-−]?\d))",
        "g",
    ),
}


def _is_ellipsis_range(left_sign: str, left: str, right_sign: str, right: str, lang: str) -> bool:
    left_time = _is_clock_time(left)
    right_time = _is_clock_time(right)
    if left_time or right_time:
        return left_time and right_time and not left_sign and not right_sign
    a = _parse_operand_value(left, lang)
    b = _parse_operand_value(right, lang)
    if a is None or b is None:
        return False
    return (-a if left_sign else a) < (-b if right_sign else b)


_DIGIT_DASH_DIGIT_SPACED = js(r"\d[ \t]*[–—-][ \t]*\d", "")
_DIGIT_DASH_DIGIT = js(r"\d[–—-]\d", "")
_DATE_RANGE = js(
    r"(?=\d)(?<![\w.,/-])(?:((\d{1,2})\.(\d{1,2})\.(\d{4}|\d{2})?)[ \t]*[–—-][ \t]*(?=(\d{1,2})\.(\d{1,2})\.(\d{4}|\d{2})?(?!\d))"
    r"|((\d{1,2})\/(\d{1,2})\/(\d{4}|\d{2}))[ \t]*[–—-][ \t]*(?=(\d{1,2})\/(\d{1,2})\/(\d{4}|\d{2})(?![\d/]))"
    r"|(\d{4}-(\d{2})-(\d{2}))(?:[ \t]*[–—][ \t]*|[ \t]+-[ \t]+)(?=\d{4}-(\d{2})-(\d{2})(?!\d)))",
    "g",
)


def _rewrite_ranges(text: str, cfg: NotationPrepassConfig, applied_rules: list[str]) -> str:
    lang = cfg.lang
    to = WORDS[lang]["rangeTo"]
    result = text
    if lang == "de" and _DIGIT_DASH_DIGIT_SPACED.test(result):

        def date_range(match, *groups):
            (dotted, d1, m1, y1, d2, m2, y2, slashed, sd1, sm1, sy1, sd2, sm2, sy2, iso, im1, id1, im2, id2) = groups[:19]
            first = dotted if dotted is not None else (slashed if slashed is not None else iso)
            if dotted is not None:
                ok = _is_range_date(d1, m1, y1) and _is_range_date(d2, m2, y2)
            elif slashed is not None:
                ok = _is_range_date(sd1, sm1, sy1) and _is_range_date(sd2, sm2, sy2)
            else:
                ok = _is_range_date(id1, im1, None) and _is_range_date(id2, im2, None)
            return f"{first} {to} " if ok and first is not None else match

        result = replace_with_tracking(result, _DATE_RANGE, date_range, "range_connector", applied_rules)
    if lang == "de" and _DIGIT_DASH_DIGIT.test(result):
        ctx = GERMAN_PAIR_CONTEXT
        measures = "|".join(escape_for_regexp(symbol) for symbol in cfg.unit_symbols)
        follower = r"[ \t]*(?:\p{Lu}|[%°=+*/×÷€$£]|(?:" + (measures or NEVER_SOURCE) + r"|Mio|Mrd|Tsd|EUR|USD|CHF)(?![\p{L}\d]))"
        pattern = js(
            r"(?<![\p{L}\d.,:/–—+-])(?=\d{1,2}[–—-]\d)(?:(?<=" + ctx["scoreBefore"] + r")()|)(\d{1,2})[–—-](\d{1,2})(?!\d|[.,:–—-]\d)"
            r"(?:(?=" + ctx["scoreAfter"] + r")()|)(?:(?=" + follower + r")()|)",
            "gu",
        )

        def score(match, label, home, away, verb, followed, *_rest):
            if (label is None and verb is None) or followed is not None:
                return match
            return f"{home} {WORDS[lang]['ratio']} {away}"

        result = replace_with_tracking(result, pattern, score, "score_hyphen", applied_rules)
    if ".." in result or "…" in result:

        def ellipsis(match, left_sign, left, right_sign, right, *_rest):
            return f"{left_sign}{left} {to} " if _is_ellipsis_range(left_sign, left, right_sign, right, lang) else match

        result = replace_with_tracking(result, _ELLIPSIS_RANGE_PATTERN[lang], ellipsis, "range_connector", applied_rules)

    def callback(match, phone_prefix, left_currency, left, left_dot, left_unit, space_before, dash, space_after,
                 right_currency, right, right_unit, *_rest):
        if _is_phone_number_end(phone_prefix, left, right):
            return match
        both_times = _is_clock_time(left) and _is_clock_time(right)
        clock_context = both_times or bool(right_unit) or _CLOCK_UNIT.test(left_unit or "")
        spaced = len(space_before) > 0 or len(space_after) > 0
        if dash == "-":
            if spaced and not both_times:
                return match
            if not clock_context:
                left_digits = _digits_of(left)
                right_digits = _digits_of(right)
                if len(left_digits) > 4 or len(right_digits) > 4:
                    return match
                if (len(left_digits) > 1 and left_digits.startswith("0")) or (
                    len(right_digits) > 1 and right_digits.startswith("0")
                ):
                    return match
                a = _parse_operand_value(left, lang)
                b = _parse_operand_value(right, lang)
                if a is None or b is None or not (a < b):
                    return match
        elif not clock_context and not left_dot:
            a = _parse_operand_value(left, lang)
            b = _parse_operand_value(right, lang)
            if a is not None and b is not None and a > b:
                return match
        moved_currency = left_currency if (left_currency and not right_currency) else ""
        kept_currency = "" if moved_currency else left_currency
        return f"{kept_currency}{left}{left_dot}{left_unit or ''} {to} {right_currency or moved_currency}{right}"

    return replace_with_tracking(result, RANGE_PATTERN, callback, "range_connector", applied_rules)


# --- per-period slashes -------------------------------------------------------------------------

_PER_HUNDRED_KM = js(r"(?<=\d[ \t]?)l[ \t]?\/[ \t]?100[ \t]?km\b", "gi")
_RPM = js(r"\bU[ \t]?\/[ \t]?min\b", "g")


def _period_pattern(lang: str):
    keys = sorted(WORDS[lang]["periods"].keys(), key=lambda key: -len(key))
    period_keys = "|".join(escape_for_regexp(key) for key in keys)
    return js(r"(?<=[\p{L}\d%€$£²³])[ \t]?\/[ \t]?(" + period_keys + r")(?![\p{L}/])(?:(?=[ \t]?\d)()|)", "giu")


def _rewrite_per_period(text: str, lang: str, applied_rules: list[str]) -> str:
    words = WORDS[lang]
    result = replace_with_tracking(
        text,
        _PER_HUNDRED_KM,
        "Liter auf hundert Kilometer" if lang == "de" else "liters per hundred kilometers",
        "per_hundred_km",
        applied_rules,
    )
    result = replace_with_tracking(
        result,
        _RPM,
        "Umdrehungen pro Minute" if lang == "de" else "revolutions per minute",
        "per_period",
        applied_rules,
    )

    def callback(match, period, before_number, *_rest):
        if len(period) == 1 and period != period.lower() and (period != "L" or before_number is not None):
            return match
        label = words["periods"].get(period.lower())
        return f" {words['per']} {label}" if label else match

    return replace_with_tracking(result, _period_pattern(lang), callback, "per_period", applied_rules)


# --- signs and symbols --------------------------------------------------------------------------

TIME_ZONE_SOURCE = "(?:UTC|GMT|CET|CEST|MEZ|MESZ|EST|EDT|PST|PDT)"
_EQUATION_OPERAND_SOURCE = r"(?:\d[\d.,]*(?:[ \t]?[%€$£])?|(?:\d+[ \t]?)?(?<![\p{L}\p{Nd}_])\p{L}(?![\p{L}\p{Nd}_]))"
_EQUATION_OPERATOR_SOURCE = r"[ \t]*[-+*/×÷−][ \t]*"
_EQUATION_MINUS_PATTERN = js(
    r"[ \t]+-[ \t]+(?:(?<=(?:[\d%€$£)]|(?<![\p{L}\p{Nd}_])\p{L})[ \t]+-[ \t]+)(?=" + _EQUATION_OPERAND_SOURCE
    + r"(?:" + _EQUATION_OPERATOR_SOURCE + _EQUATION_OPERAND_SOURCE + r"){0,9}\)?[ \t]*=)|"
    + r"(?<==[ \t]*(?:" + _EQUATION_OPERAND_SOURCE + _EQUATION_OPERATOR_SOURCE + r"){0,9}" + _EQUATION_OPERAND_SOURCE
    + r"[ \t]+-[ \t]+)(?=\d|(?<![\p{L}\p{Nd}_])\p{L}(?![\p{L}\p{Nd}_])))",
    "gu",
)

_SUPERSCRIPTS = "⁰¹²³⁴⁵⁶⁷⁸⁹"
_SUBSCRIPTS = "₀₁₂₃₄₅₆₇₈₉"
_HAS_SUPERSCRIPT = js(r"[⁰¹²³⁴⁵⁶⁷⁸⁹]", "")
_FORMULA_LINE = js(r"[=×÷≠]", "")
_EQUATION_LINE = js(r"[=÷≠]", "")
_FOOTNOTE_AFTER_SIGN = js(
    r"(?=[⁰¹²³⁴⁵⁶⁷⁸⁹])(?:(?<=§§?[ \t]?\d+(?:[ \t]?[a-z])?)|(?<=[€$£¥%])|(?<=(?<![\d,])(?:19|20)\d{2}))[⁰¹²³⁴⁵⁶⁷⁸⁹]+(?![⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺])",
    "g",
)
_SUBSCRIPT_DIGITS = js(r"(?<=([\p{L}\d])|[^\p{L}\d⁄]|^)([₀₁₂₃₄₅₆₇₈₉]+)(?=([\p{L}\d])?)", "gu")
_SUBSCRIPT_GLYPH = js(r"[₀₁₂₃₄₅₆₇₈₉]", "g")
_SPACED_MINUS_BEFORE_SIGNED = js(r"(?<=(?:[\d%€$£)]|(?<![\p{L}\p{Nd}_])\p{L})[ \t]+)-[ \t]+(?=-\d)", "gu")
_SPACED_UNICODE_MINUS = js(r"(?<=[\d%€$£)\]]|(?<![\p{L}\p{Nd}_])\p{L})[ \t]+−[ \t]+(?=[\d(€$£]|\p{L}(?![\p{L}\p{Nd}_]))", "gu")
_DASH_SIGN = js(r"(?<![^\s:(\[])(?<!\d[ \t]*)[–—](?=\d)", "g")
_UNICODE_MINUS = js(r"−(?=\s?\d)", "g")
_ZONE_SIGN = js(r"\b(" + TIME_ZONE_SOURCE + r")(?:(-)|\+(?=\d{2}:?\d{2}\b))(?=\d)", "g")
_TIMES_SIGN = js(r"(\d[\d.,]*)[ \t]?×[ \t]?(?=(\d+(?:[.,]\d+)*))", "g")
_DIVISION_SIGN = js(r"[ \t]*÷[ \t]*", "g")
_GLUED_PLUS_MINUS = js(r"(?<=[\p{L}\d])(?=[±∓][ \t]?\d)", "gu")
_PLUS_MINUS = js(r"±\s?(?=\d)", "g")
_MINUS_PLUS = js(r"∓\s?(?=\d)", "g")
_LEADING_PLUS = js(r"(?<![\w+])\+(?=\d[\d.,]*(?![\d.,]*[ \t(]+\d)(?:[ \t]*(?:%|°|€|\$|£|\p{L})|[ \t]*$|[.,;:!?)]))", "gu")
_INFIX_PLUS_GLUED = js(r"(?<=[\p{L}\d])\+(?=[\p{L}\d])(?!\d{2}:?\d{2}\b)", "gu")
_TILDE_RANGE = js(r"(?<=\d)[ \t]*~[ \t]*(?=\d)", "g")
_APPROX = js(r"[≈~]\s?(?=[\d€$£])", "g")
_GE = js(r"≥\s?(?=\d)", "g")
_LE = js(r"≤\s?(?=\d)", "g")
_GE_ASCII = js(r"(?<=\s)>=\s?(?=\d)", "g")
_LE_ASCII = js(r"(?<=\s)<=\s?(?=\d)", "g")
_GT_ASCII = js(r"(?<=\s)>\s?(?=\d)", "g")
_LT_ASCII = js(r"(?<=\s)<\s?(?=\d)", "g")
_NOT_EQUAL = js(r"[ \t]*(?:!==?|=\/=|≠)[ \t]*", "g")
_DOUBLE_EQUALS = js(
    r"(?:(?<=[\d)\]]|(?<![\p{L}\p{Nd}_])\p{L})[ \t]*(?<![=<>!])={2,3}(?![=>])[ \t]*(?=[\p{L}\d(-])"
    r"|(?<=[\p{L}\d)\]])[ \t]*(?<![=<>!])={2,3}(?![=>])[ \t]*(?=[\d(-]|\p{L}(?!\p{L})))",
    "gu",
)
_LESS_THAN_RELATION = js(r"(?<=[\p{L}\d])[ \t]+<[ \t]+(?=\p{L})", "gu")
_GREATER_THAN_RELATION = js(r"(?<=(?<!\p{L})\p{L}|\d)[ \t]+>[ \t]+(?=\p{L}(?!\p{L}))", "gu")
_DIAL_CODE = js(r"\*(#)?(\d{1,6})#(?=([\p{L}\d])?)", "gu")
_SECTIONS_SIGN = js(r"§§\s?(?=\d)", "g")
_SECTION_SIGN = js(r"§\s?(?=\d)", "g")
_CODE_OR = js(r"(?<=\S)[ \t]+\|\|[ \t]+(?=\S)", "g")
_CODE_AND = js(r"(?<=\S)[ \t]+&&[ \t]+(?=\S)", "g")
_AMPERSAND = js(r"(?<=\S)\s&\s(?=\S)", "g")
_DEGREE_MARK = js(r"[°º]", "")
_BARE_DEGREE = js(r"(?<=\d)\s?[°º](?![ \t]?[CFK]\b)", "g")
_TIMES_BETWEEN_OPERANDS = js(
    r"(?<=[€$£%°)\]])[ \t]*×[ \t]*(?=[\p{L}\d(€$£-])|(?<=\p{L})[ \t]*×[ \t]*(?=[\d(€$£-])|(?<=\d)[ \t]*×[ \t]*(?=\()"
    r"|(?<=(?<!\p{L})\p{L})[ \t]*×[ \t]*(?=\p{L})|(?<=\p{L})[ \t]*×[ \t]*(?=\p{L}(?!\p{L}))",
    "gu",
)
_TIMES_BETWEEN_WORDS = js(r"(?<=\p{L})[ \t]*×[ \t]*(?=\p{L})", "gu")
_ASTERISK_TIMES = js(
    r"(?<=[\d%€$£°⁰¹²³⁴⁵⁶⁷⁸⁹)\]]|(?<![\p{L}\p{Nd}_])\p{L})[ \t]+\*[ \t]+(?=[\d(€$£-]|\p{L}(?![\p{L}\p{Nd}_]))", "gu"
)
_TRAILING_PLUS = js(r"(?<=[\p{L}\d])\+(?=[ \t\n]|$|[,.;:!?)\"”])", "gu")
_TRAILING_MINUS = js(r"(?:(?<=\d)|(?<=\b[A-F])(?<!\p{Lu}[- \t][A-F]))[-−](?=[ \t\n]|$|[,.;:!?)\"”])", "gu")
_INFIX_PLUS_SPACED = js(r"(?<=[\p{L}\d€$£%°⁰¹²³⁴⁵⁶⁷⁸⁹)\]])[ \t]+\+(?:[ \t]+(?=[\p{L}\d(€$£]|-\d)|(?=-\d))", "gu")
_FORMULA_SLASH = js(r"(?<=[\p{L})\]]|\b(?:[1-9]\d*|0)(?:[.,]\d+)?)(?<![+\d][ \t]+\d+(?:[.,]\d+)?)[ \t]+\/[ \t]+(?=[\d(])", "gu")
_EQUALS = js(r"(?<=[\p{L}\d°%€$£)⁰¹²³⁴⁵⁶⁷⁸⁹])[ \t]*(?<![=<>!])=(?![=>])[ \t]*(?=[\p{L}\d(?-])", "gu")
_MENU_PATH = js(r"(?<=\p{L})[ \t]*>[ \t]*(?=\p{L})", "gu")
_AT_SIGN = js(r"(?<=\s)@(?=\s)", "g")
_CARE_OF = js(r"\bc\/o\b", "g")
_SUPERSCRIPT_2 = js(r"(?<=[a-z])\^2\b", "g")
_SUPERSCRIPT_3 = js(r"(?<=[a-z])\^3\b", "g")
_POWER_CARET = js(r"(?<=\d)\^(-?\d+)", "g")
_POWER_CARET_LETTER = js(r"(?<=[\p{L}\d)\]])\^(?=[(\p{L}])", "gu")
_SUPERSCRIPT_MINUS = js(r"(?<=[⁰¹²³⁴⁵⁶⁷⁸⁹])-(?=\d)", "g")
_POWER_SUPERSCRIPT = js(r"(?<=\d)([⁻⁺]?)([⁰¹²³⁴⁵⁶⁷⁸⁹]+)", "g")
_LETTER_POWER = js(r"(?<![\p{L}\d])(?<!\d[ \t])([a-z])([⁻⁺]?)([⁰¹²³⁴⁵⁶⁷⁸⁹]+)", "gu")
_MIDDLE_DOT = js(r"(?<=\d)[ \t]?·[ \t]?(?=\d)", "g")
_HASH_KEY = js(r"(?<=\d)#(?=[ \t]|$|[,.;:!?)])", "g")
_CENT_SIGN = js(r"(?<=\d)\s?¢", "g")
_YEN_SIGN = js(r"¥\s?(\d[\d.,]*\d|\d)", "g")
_DASH_CENTS = js(
    r"(?<=\d)[.,][-–—]{1,2}(?=[ \t]?(?:[€$£]|EUR|USD|GBP|CHF|JPY|YEN|Fr\.))"
    r"|(?<=(?:[€$£]|\b(?:EUR|USD|GBP|CHF|JPY|YEN))[ \t]?\d[\d.']*)[.,][-–—]{1,2}(?![\d-])",
    "g",
)
_MIXED_FRACTION = js(r"(?<![\w.,/\-−])([1-9]\d*)[ \t]+([1-9])\/([1-9]\d?)(?![\d/]|[.,]\d)", "g")


def _superscript_exponent(sign: str, digits: str) -> str:
    """The exponent of a superscript power: its digits, with a minus where it has one."""
    plain = "".join(str(_SUPERSCRIPTS.index(glyph)) for glyph in digits)
    return f"-{plain}" if sign == "⁻" else plain


def _rewrite_symbols(text: str, cfg: NotationPrepassConfig, applied_rules: list[str]) -> str:
    lang = cfg.lang
    w = WORDS[lang]
    formula_lines: set[int] = set()
    equation_lines: set[int] = set()
    for index, line in enumerate(text.split("\n")):
        if _FORMULA_LINE.test(line):
            formula_lines.add(index)
        if _EQUATION_LINE.test(line):
            equation_lines.add(index)
    result = text
    if lang == "de" and _HAS_SUPERSCRIPT.test(result):
        result = replace_with_tracking(result, _FOOTNOTE_AFTER_SIGN, "", "footnote_mark", applied_rules)
        if cfg.number_label_source:
            result = replace_with_tracking(
                result,
                js(
                    r"(?=[⁰¹²³⁴⁵⁶⁷⁸⁹])(?<=(?<!\p{L})" + cfg.number_label_source
                    + r"[ \t]*\d+(?:[ \t]?[a-z])?)[⁰¹²³⁴⁵⁶⁷⁸⁹]+(?![⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺])",
                    "gu",
                ),
                "",
                "footnote_mark",
                applied_rules,
            )

        def footnote_word(match, word, digits, *_rest):
            if word in cfg.unit_symbols or f"{word}{digits}" in cfg.unit_symbols or word in cfg.unit_words:
                return match
            return word

        result = replace_with_tracking(
            result,
            js(
                r"(?<![\p{L}\d])(?<!\d[ \t])(?<!\b" + w["per"] + r"[ \t]+)(\p{L}{3,})([⁰¹²³⁴⁵⁶⁷⁸⁹]+)(?![⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺])",
                "gu",
            ),
            footnote_word,
            "footnote_mark",
            applied_rules,
        )

    def subscript(_match, before, digits, after, *_rest):
        plain = js_replace(_SUBSCRIPT_GLYPH, digits, lambda glyph, *_a: str(_SUBSCRIPTS.index(glyph)))
        return f"{' ' if before is not None else ''}{plain}{' ' if after is not None else ''}"

    result = replace_with_tracking(result, _SUBSCRIPT_DIGITS, subscript, "subscript", applied_rules)
    math_glyphs = "".join(w["mathSymbols"].keys())
    if js("[" + math_glyphs + "]", "u").test(result):
        result = replace_with_tracking(
            result,
            js(r"[ \t]*([" + math_glyphs + r"])[ \t]*(?=([^\s,.;:!?)\]])?)", "gu"),
            lambda _m, glyph, operand, *_r: f" {w['mathSymbols'][glyph]}{' ' if operand is not None else ''}",
            "formula_symbol",
            applied_rules,
        )
    result = replace_with_tracking(result, _EQUATION_MINUS_PATTERN, f" {w['minus']} ", "binary_minus", applied_rules)
    result = replace_with_tracking(result, _SPACED_MINUS_BEFORE_SIGNED, f"{w['minus']} ", "binary_minus", applied_rules)
    result = replace_with_tracking(result, _SPACED_UNICODE_MINUS, f" {w['minus']} ", "binary_minus", applied_rules)
    result = replace_with_tracking(result, _DASH_SIGN, "-", "dash_sign", applied_rules)
    result = replace_with_tracking(result, _UNICODE_MINUS, "-", "unicode_minus", applied_rules)
    result = replace_with_tracking(
        result,
        _ZONE_SIGN,
        lambda _m, zone, minus, *_r: f"{zone} {w['minus'] if minus else w['plus']} ",
        "zone_sign",
        applied_rules,
    )
    result = replace_with_tracking(
        result,
        js(
            r"(?<!(?:\d|\b" + w["digitWordSource"] + r")[ \t]+\()(?:(?<=[(\[=+*×÷][ \t]*)([a-z])-(?=\d)"
            r"|(?<![\w.,-])([a-z])-(?=\d+(?:,\d+)?[ \t]*[)\]=]))",
            "g",
        ),
        lambda _m, first, second, *_r: f"{first if first is not None else second} {w['minus']} ",
        "variable_minus",
        applied_rules,
    )

    def times(_match, left, right, *_rest):
        dimension = _ALL_DIGITS.test(left) and _ALL_DIGITS.test(right)
        word = "mal" if lang == "de" else ("by" if dimension else "times")
        return f"{left} {word} "

    result = replace_with_tracking(result, _TIMES_SIGN, times, "times_sign", applied_rules)
    result = replace_with_tracking(result, _DIVISION_SIGN, f" {w['dividedBy']} ", "division_sign", applied_rules)
    result = replace_with_tracking(result, _GLUED_PLUS_MINUS, " ", "plus_minus", applied_rules)
    result = replace_with_tracking(result, _PLUS_MINUS, f"{w['plusMinus']} ", "plus_minus", applied_rules)
    result = replace_with_tracking(result, _MINUS_PLUS, f"{w['minusPlus']} ", "minus_plus", applied_rules)
    result = replace_with_tracking(result, _LEADING_PLUS, f"{w['plus']} ", "leading_plus", applied_rules)
    result = replace_with_tracking(result, _INFIX_PLUS_GLUED, f" {w['plus']} ", "infix_plus", applied_rules)
    result = replace_with_tracking(result, _TILDE_RANGE, f" {w['rangeTo']} ", "tilde_range", applied_rules)
    result = replace_with_tracking(result, _APPROX, f"{w['about']} ", "approx_sign", applied_rules)
    result = replace_with_tracking(result, _GE, f"{w['atLeast']} ", "comparison_sign", applied_rules)
    result = replace_with_tracking(result, _LE, f"{w['atMost']} ", "comparison_sign", applied_rules)
    result = replace_with_tracking(result, _GE_ASCII, f"{w['atLeast']} ", "comparison_sign", applied_rules)
    result = replace_with_tracking(result, _LE_ASCII, f"{w['atMost']} ", "comparison_sign", applied_rules)
    result = replace_with_tracking(result, _GT_ASCII, f"{w['greaterThan']} ", "comparison_sign", applied_rules)
    result = replace_with_tracking(result, _LT_ASCII, f"{w['lessThan']} ", "comparison_sign", applied_rules)
    result = replace_with_tracking(result, _NOT_EQUAL, f" {w['notEqual']} ", "not_equal", applied_rules)
    result = replace_with_tracking(result, _DOUBLE_EQUALS, f" {w['equals']} ", "equals_sign", applied_rules)
    result = replace_with_tracking(
        result, _LESS_THAN_RELATION, f" {w['lessThanRelation']} ", "comparison_relation", applied_rules
    )
    result = replace_with_tracking(
        result, _GREATER_THAN_RELATION, f" {w['greaterThanRelation']} ", "comparison_relation", applied_rules
    )
    if "*" in result:

        def dial(_match, hash_sign, digits, after, *_rest):
            parts = [w["star"], w["hashKey"] if hash_sign else "", cfg.digit_words(digits), w["hashKey"]]
            return " ".join(part for part in parts if part) + (" " if after is not None else "")

        result = replace_with_tracking(result, _DIAL_CODE, dial, "dial_code", applied_rules)
    number_word = f"[{w['number'][0].upper()}{w['number'][0].lower()}]{w['number'][1:]}"
    number_namers = (
        f"(?:{number_word}|(?<!\\p{{L}}){cfg.number_label_source})" if cfg.number_label_source else number_word
    )
    result = replace_with_tracking(
        result,
        js(
            r"(?<![\w&])#(?:(?<=" + number_namers + r"[ \t]*:?[ \t]*#)()|)(?![0-9a-fA-F]{6}\b|[0-9a-fA-F]{8}\b)(?=\d|[A-Z]{1,4}\d)",
            "gu",
        ),
        lambda _m, labelled, *_r: "" if labelled is not None else f"{w['number']} ",
        "number_sign",
        applied_rules,
    )
    result = replace_with_tracking(result, _SECTIONS_SIGN, f"{w['sections']} ", "section_sign", applied_rules)
    result = replace_with_tracking(result, _SECTION_SIGN, f"{w['section']} ", "section_sign", applied_rules)
    result = replace_with_tracking(result, _CODE_OR, f" {w['or']} ", "code_operator", applied_rules)
    result = replace_with_tracking(result, _CODE_AND, f" {w['and']} ", "code_operator", applied_rules)
    result = replace_with_tracking(result, _AMPERSAND, f" {w['and']} ", "ampersand", applied_rules)
    result = replace_with_tracking(
        result,
        js(
            r"\s?(?:[→➔➜⇒]|(?<!" + w["symbolNounSource"] + r"[ \t]*)(?:(?<![<-])-+>|(?<![<=])=+>))\s?",
            "g",
        ),
        f" {w['arrowTo']} ",
        "arrow",
        applied_rules,
    )
    if _DEGREE_MARK.test(result):
        compass = "|".join(w["compassPoints"].keys())

        def coordinates(_match, degrees, minutes, seconds, point, *_rest):
            parts = [
                _count_words(degrees, w["degreeOne"], w["degree"], w["degrees"], cfg),
                _count_words(minutes, w["oneCount"], w["minute"]["singular"], w["minute"]["plural"], cfg),
                _count_words(seconds, w["oneCount"], w["second"]["singular"], w["second"]["plural"], cfg)
                if seconds is not None
                else "",
                w["compassPoints"][point] if point is not None else "",
            ]
            return " ".join(part for part in parts if part)

        result = replace_with_tracking(
            result,
            js(
                r"(?<![\d.,])(\d{1,3})[ \t]?°[ \t]?(\d{1,2}(?:[.,]\d+)?)[ \t]?['′’](?:[ \t]?(\d{1,2}(?:[.,]\d+)?)[ \t]?(?:''|[\"″”]))?"
                r"(?:[ \t]?(" + compass + r")\b)?",
                "g",
            ),
            coordinates,
            "coordinates",
            applied_rules,
        )
        result = replace_with_tracking(
            result,
            js(r"(?<=\d)[ \t]?[°º][ \t]?(" + compass + r")(?![\p{L}\d-])", "gu"),
            lambda _m, point, *_r: f" {w['degrees']} {w['compassPoints'][point]}",
            "coordinates",
            applied_rules,
        )
    result = replace_with_tracking(result, _BARE_DEGREE, f" {w['degrees']}", "bare_degree", applied_rules)
    unit_dimension = r"(?<=[°%])[ \t]?[×x][ \t]?(?=\d)|(?<=\p{L})[ \t]?×[ \t]?(?=\d)"
    if cfg.unit_symbol_source:
        unit_dimension += r"|[ \t]+x[ \t]+(?<=\d[ \t]?(?:" + cfg.unit_symbol_source + r")[ \t]+x[ \t]+)(?=\d)"
    result = replace_with_tracking(
        result, js(unit_dimension, "gu"), " mal " if lang == "de" else " by ", "unit_dimension", applied_rules
    )
    result = replace_with_tracking(result, _TIMES_BETWEEN_OPERANDS, f" {w['times']} ", "times_sign", applied_rules)
    result = replace_with_tracking(
        result,
        _TIMES_BETWEEN_WORDS,
        lambda match, offset, source: f" {w['times']} " if _line_index(source, offset) in equation_lines else match,
        "times_sign",
        applied_rules,
    )
    result = replace_with_tracking(result, _ASTERISK_TIMES, f" {w['times']} ", "asterisk_times", applied_rules)
    result = replace_with_tracking(result, _TRAILING_PLUS, f" {w['plus']}", "trailing_plus", applied_rules)
    result = replace_with_tracking(result, _TRAILING_MINUS, f" {w['minus']}", "trailing_minus", applied_rules)
    result = replace_with_tracking(result, _INFIX_PLUS_SPACED, f" {w['plus']} ", "infix_plus", applied_rules)
    result = replace_with_tracking(
        result,
        _FORMULA_SLASH,
        lambda match, offset, source: f" {w['dividedBy']} " if _line_index(source, offset) in formula_lines else match,
        "division_sign",
        applied_rules,
    )
    result = replace_with_tracking(
        result,
        js(
            r"(?<=(?<![\w.,])\p{L})[ \t]+%[ \t]+(?=\d+(?!\d|[.,]\d))|(?<=(?<![\w.,-])\d+)[ \t]+%[ \t]+(?=\d+[ \t]*(?:==?|"
            + w["resultWordSource"] + r"\b))",
            "gu",
        ),
        f" {w['modulo']} ",
        "modulo",
        applied_rules,
    )
    result = replace_with_tracking(result, _EQUALS, f" {w['equals']} ", "equals_sign", applied_rules)
    result = replace_with_tracking(result, _MENU_PATH, ", ", "menu_path", applied_rules)
    result = replace_with_tracking(result, _AT_SIGN, w["atSign"], "at_sign", applied_rules)
    result = replace_with_tracking(result, _CARE_OF, w["careOf"], "care_of", applied_rules)
    result = replace_with_tracking(result, _SUPERSCRIPT_2, "²", "superscript_unit", applied_rules)
    result = replace_with_tracking(result, _SUPERSCRIPT_3, "³", "superscript_unit", applied_rules)
    result = replace_with_tracking(
        result, _POWER_CARET, lambda _m, exponent, *_r: f" {w['power']} {exponent}", "power", applied_rules
    )
    result = replace_with_tracking(result, _POWER_CARET_LETTER, f" {w['power']} ", "power", applied_rules)
    if _HAS_SUPERSCRIPT.test(result):
        result = replace_with_tracking(result, _SUPERSCRIPT_MINUS, f" {w['minus']} ", "binary_minus", applied_rules)
        result = replace_with_tracking(
            result,
            _POWER_SUPERSCRIPT,
            lambda _m, sign, digits, *_r: f" {w['power']} {_superscript_exponent(sign, digits)}",
            "power",
            applied_rules,
        )
        result = replace_with_tracking(
            result,
            _LETTER_POWER,
            lambda match, letter, sign, digits, *_r: (
                match if letter in cfg.unit_symbols else f"{letter} {w['power']} {_superscript_exponent(sign, digits)}"
            ),
            "power",
            applied_rules,
        )
    result = replace_with_tracking(
        result, _MIDDLE_DOT, " mal " if lang == "de" else " times ", "middle_dot_times", applied_rules
    )
    result = replace_with_tracking(result, _HASH_KEY, f" {w['hashKey']}", "hash_key", applied_rules)
    result = replace_with_tracking(result, _CENT_SIGN, f" {w['cents']}", "cent_sign", applied_rules)
    result = replace_with_tracking(
        result, _YEN_SIGN, lambda _m, amount, *_r: f"{amount} {w['yen']}", "yen_sign", applied_rules
    )
    result = replace_with_tracking(result, _DASH_CENTS, "", "dash_cents", applied_rules)
    fractions = w["unicodeFractions"]
    fraction_glyphs = "".join(fractions.keys())
    if "/" in result:

        def mixed(match, integer, numerator, denominator, *_rest):
            for candidate in fractions:
                if unicodedata.normalize("NFKD", candidate).replace("⁄", "/", 1) == f"{numerator}/{denominator}":
                    return f"{integer}{candidate}"
            return match

        result = replace_with_tracking(result, _MIXED_FRACTION, mixed, "unicode_fraction", applied_rules)

    def fraction(match, integer_token, glyph, percent, *_rest):
        entry = fractions.get(glyph)
        if not entry:
            return match
        words = entry["alone"]
        if integer_token:
            value = int(integer_token)
            integer_words = "ein" if lang == "de" and value == 1 else cfg.integer_words(value)
            words = f"{integer_words}{entry['afterInteger']}" if lang == "de" else f"{integer_words} {entry['afterInteger']}"
        return f"{words} {w['percent']}" if percent else words

    return replace_with_tracking(
        result,
        js(r"(?<!\d)(?:(\d+)[ \t]?)?([" + fraction_glyphs + r"])(?:[ \t]?(%))?", "g"),
        fraction,
        "unicode_fraction",
        applied_rules,
    )


# --- ratios, scores, counts -----------------------------------------------------------------------

_ONE_OR_TWO_DIGITS = js(r"^\d{1,2}$", "")
_SECONDS_WITH_FRACTION = js(r"^[0-5]\d\.(?!000)\d{3}$", "")
_LEADING_ZERO_NUMBER = js(r"^0\d+$", "")


def is_slash_day_month(left: str, right: str) -> bool:
    """A slash pair that a dateless prefix makes a day and a month: the day above the month (Bug 842)."""
    if not _ONE_OR_TWO_DIGITS.test(left) or not _ONE_OR_TWO_DIGITS.test(right):
        return False
    day = int(left)
    month = int(right)
    return 1 <= month <= 12 and day <= 31 and day > month


def _is_colon_pair_for_another_rule(left: str, right: str, lang: str, power, span) -> bool:
    if "." in right:
        return _SECONDS_WITH_FRACTION.test(right)
    if len(right) != 2 or not _ALL_DIGITS.test(left):
        return False
    hours = int(left)
    minutes = int(right)
    if minutes > 59:
        return False
    if span is not None:
        return True
    if power is not None:
        return False
    return hours <= 23 or (lang == "de" and hours == 24 and minutes == 0)


_TWENTY_FOUR_SEVEN = js(r"\b24\/7\b", "g")


def _rewrite_ratios_and_counts(text: str, cfg: NotationPrepassConfig, applied_rules: list[str]) -> str:
    lang = cfg.lang
    w = WORDS[lang]
    ctx = GERMAN_PAIR_CONTEXT if lang == "de" else None

    def ratio(match, labelled, left, right, power, span, hour, *_rest):
        if not (labelled is not None and hour is None) and _is_colon_pair_for_another_rule(left, right, lang, power, span):
            return match
        spoken_left = cfg.integer_words(int(left)) if _ALL_DIGITS.test(left) else left
        return f"{spoken_left} {w['ratio']} {cfg.integer_words(int(right.replace('.', '')))}"

    result = replace_with_tracking(
        text,
        js(
            r"(?<![\d:.,])(?:(?<=" + (ctx["ratioBefore"] if ctx else NEVER_SOURCE) + r")()|)(\d{1,3}(?:[.,]\d+)?):(\d{1,3}(?:\.\d{3})+|\d{1,6})"
            r"(?!\d|[.,]\d|:)(?:(?=([ \t]+" + w["power"] + r"\b))|(?=([ \t]*" + w["spanUnitSource"] + r"(?![\p{L}\d])))"
            r"|(?=([ \t]*" + w["clockHour"] + r"\b))|)",
            "gu",
        ),
        ratio,
        "ratio_colon",
        applied_rules,
    )
    result = replace_with_tracking(
        result,
        _TWENTY_FOUR_SEVEN,
        "24 7" if lang == "de" else "twenty-four seven",
        "twenty_four_seven",
        applied_rules,
    )
    result = replace_with_tracking(
        result,
        js(
            r"(?<![\d./,])(?=\d{2,3}\/\d)(?:(?<=" + (ctx["bloodPressureBefore"] if ctx else NEVER_SOURCE) + r")"
            r"|(?=\d{2,3}\/\d{2,3}[ \t]?mmHg\b))(\d{2,3})\/(\d{2,3})(?![\d/]|[.,]\d)",
            "gu",
        ),
        lambda _m, systolic, diastolic, *_r: f"{systolic} {'zu' if lang == 'de' else 'over'} {diastolic}",
        "blood_pressure",
        applied_rules,
    )

    def count_slash(match, code, count_label, date_prefix, left, right, count_noun, compound, mark, *_rest):
        left = left or ""
        right = right or ""
        if code is not None or _LEADING_ZERO_NUMBER.test(left):
            return match
        counted = count_label is not None or count_noun is not None
        if not counted and date_prefix is not None and is_slash_day_month(left, right) and compound is None:
            return match
        numerator = js_number(js_replace_first_literal(left, ",", "."))
        denominator = js_number(right)
        has_fraction_word = (denominator <= 12 or denominator == 100) and denominator >= 2
        if _ALL_DIGITS.test(left) and has_fraction_word and (compound is not None or (not counted and numerator < denominator)):
            return match
        if mark is not None and not counted:
            return match
        if lang == "en" and _ALL_DIGITS.test(left) and 1 <= numerator <= 12 and 1 <= denominator <= 31:
            return match
        return f"{left} {w['outOf']} {right}"

    def ctx_source(key: str) -> str:
        return ctx[key] if ctx else NEVER_SOURCE

    return replace_with_tracking(
        result,
        js(
            r"(?<![\d./,\p{L}])(?=\d{1,3}(?:[.,]\d+)?\/\d)(?:(?<=" + ctx_source("codeBefore") + r")()|(?<=" + ctx_source("countBefore")
            + r")()|(?<=" + ctx_source("datePrefixBefore") + r")()|)"
            r"(\d{1,3}(?:[.,]\d+)?)\/(\d{1,3})(?![\d/%]|[.,]\d|\s?\/)"
            r"(?:(?=" + ctx_source("countAfter") + r")()|(?=" + ctx_source("compoundAfter") + r")()|(?=[.,])()|)",
            "gu",
        ),
        count_slash,
        "count_slash",
        applied_rules,
    )


# --- identifiers --------------------------------------------------------------------------------

_DIGIT_CHAR = js(r"\d", "")
_LATIN_CHAR = js(r"[A-Za-z]", "")
_HEX_LITERAL = js(r"\b0x([0-9a-fA-F]{2,16})\b", "g")
_SCIENTIFIC = js(r"(?<![\w#.,])(\d{1,3}(?:[.,]\d+)?)[eE](?:\+|([-−]))?(\d{1,3})(?!\w|[.,]\d)", "g")
_IBAN = js(r"\b([A-Z]{2}\d{2})((?:[ \t]?[A-Z0-9]{4}){2,7}(?:[ \t]?[A-Z0-9]{1,4})?)\b", "g")
_WS_RUN = js(r"\s+", "g")
_GROUPS_OF_FOUR = js(r".{1,4}", "g")
_DIGIT_GROUPS_SPACED = js(r"\b([1-9]\d{3}(?:[ \t]\d{4}){2,}(?:[ \t]\d{1,4})?)\b", "g")
_SPACE_OR_TAB = js(r"[ \t]", "")
_DIGIT_GROUPS_DASHED = js(r"(?<![\w.+])(\d{1,}(?:-\d{2,}){2,})(?![\w-])", "g")
_ISO_DATE_SHAPE = js(r"^\d{4}-\d{2}-\d{2}$", "")
_EMAIL = js(EMAIL_ADDRESS_PATTERN_SOURCE, "giu")
_CAPITALISED_WORD = js(r"^\p{Lu}\p{Ll}+$", "u")
_ANY_UPPER = js(r"\p{Lu}", "u")
_BANK_CODE_LABEL_SOURCE = "(?:BIC|SWIFT)"
_BANK_CODE = js(r"(?<=\b" + _BANK_CODE_LABEL_SOURCE + r"[ \t]*:?[ \t]+)[A-Z]{6}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b", "g")
_ALNUM_CODE = js(
    r"(?<![#\w:])(?<!\d{4}-\d{2}-)(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])([A-Z0-9]{2,24})\b(?:(?<=\d)-(\d{1,6})(?!\w|[.,]\d|-\d))?", "g"
)
_ALNUM_BOUNDARY = js(r"(\d)(?=[A-Z])|([A-Z])(?=\d)", "g")
_TRAILING_DIGITS = js(r"\d+$", "")
_SINGLE_LOWER = js(r"^[a-z]$", "")
_DOTTED_NUMBER_LETTER = js(r"(?<=\s)(\d{1,3}\.\d{1,3})(?![hk])([a-z])\b", "g")
_VERSION_TAG = js(r"(?<![\w.])v(\d{1,3})\b(?!\.(?:\d|[xX*](?![\w*]|\.\d))|\d)", "g")
_WORD_GLUED_NUMBER = js(r"\b([a-z]{3,})(\d{1,4})\b(?!\.\d|[\d-])", "g")


def _spell_identifier(token: str, cfg: NotationPrepassConfig) -> str:
    parts = []
    for ch in token or "":
        if _DIGIT_CHAR.test(ch):
            parts.append(cfg.digit_words(ch))
        elif _LATIN_CHAR.test(ch):
            parts.append(ch.upper())
        else:
            parts.append(ch)
    return " ".join(parts)


def _is_code_range_end(code: str, number: str) -> bool:
    """Whether the number after a code's hyphen ends a range from the code's last number (Bug 840)."""
    found = _TRAILING_DIGITS.search(code)
    start = found.group(0) if found else ""
    if not start or len(number) > len(start) + 1:
        return False
    if (len(start) > 1 and start.startswith("0")) or (len(number) > 1 and number.startswith("0")):
        return False
    return int(start) < int(number)


def _rewrite_identifiers(text: str, cfg: NotationPrepassConfig, applied_rules: list[str]) -> str:
    w = WORDS[cfg.lang]
    result = replace_with_tracking(
        text,
        _HEX_LITERAL,
        lambda _m, hex_digits, *_r: f"{cfg.digit_words('0')} x {_spell_identifier(hex_digits, cfg)}",
        "hex_literal",
        applied_rules,
    )

    def scientific(_match, mantissa, minus, exponent, *_rest):
        spoken = mantissa.replace(".", ",", 1) if cfg.lang == "de" else mantissa.replace(",", ".", 1)
        return f"{spoken} {w['times']} 10 {w['power']} {'-' if minus else ''}{int(exponent)}"

    result = replace_with_tracking(result, _SCIENTIFIC, scientific, "scientific_notation", applied_rules)

    def iban(_match, head, body, *_rest):
        groups = _GROUPS_OF_FOUR.match_all(js_replace(_WS_RUN, body, ""))
        return ", ".join([_spell_identifier(head, cfg), *(_spell_identifier(group, cfg) for group in groups)])

    result = replace_with_tracking(result, _IBAN, iban, "iban", applied_rules)
    result = replace_with_tracking(
        result,
        _DIGIT_GROUPS_SPACED,
        lambda _m, chain, *_r: ", ".join(cfg.digit_words(group) for group in _SPACE_OR_TAB.split(chain)),
        "digit_groups",
        applied_rules,
    )

    def dashed(match, chain, *_rest):
        if _ISO_DATE_SHAPE.test(chain):
            return match
        return ", ".join(cfg.digit_words(group) for group in chain.split("-"))

    result = replace_with_tracking(result, _DIGIT_GROUPS_DASHED, dashed, "digit_groups", applied_rules)

    def label(part: str, is_ending: bool) -> str:
        return cfg.address_label(part, is_ending) if cfg.address_label else part

    def email(_match, local, domain, *_rest):
        host_labels = domain.split(".")
        next_sentence = None
        if (
            len(host_labels) >= 3
            and _CAPITALISED_WORD.test(host_labels[-1])
            and not _ANY_UPPER.test(host_labels[-2])
        ):
            next_sentence = host_labels.pop()
        spoken_local = f" {w['dot']} ".join(label(part, False) for part in local.split("."))
        spoken_host = f" {w['dot']} ".join(
            label(part, index == len(host_labels) - 1) for index, part in enumerate(host_labels)
        )
        return f"{spoken_local} {w['at']} {spoken_host}{'.' + next_sentence if next_sentence else ''}"

    result = replace_with_tracking(result, _EMAIL, email, "email_address", applied_rules)
    result = replace_with_tracking(
        result, _BANK_CODE, lambda code, *_r: _spell_identifier(code, cfg), "labelled_code", applied_rules
    )

    def alnum(_match, code, number, *_rest):
        spoken = _spell_identifier(code, cfg) if len(code) >= 8 else js_replace(_ALNUM_BOUNDARY, code, "$1$2 ")
        if number is None:
            return spoken
        return f"{spoken}-{number}" if _is_code_range_end(code, number) else f"{spoken} {number}"

    result = replace_with_tracking(result, _ALNUM_CODE, alnum, "alnum_code_split", applied_rules)
    unit_letters = "".join(symbol for symbol in cfg.unit_symbols if _SINGLE_LOWER.test(symbol))
    labelled_unit_letter = (
        r"(?!(?<=(?:(?<!\p{L})" + cfg.number_label_source + r"[ \t]*|§[ \t]?)\d{1,3})[" + unit_letters + r"]\b)"
        if cfg.number_label_source and unit_letters
        else ""
    )
    result = replace_with_tracking(
        result,
        js(r"(?<=[\s§])(\d{1,3})" + labelled_unit_letter + r"([a-rt-z])\b", "gu"),
        lambda _m, digits, letter, *_r: f"{digits} {letter}",
        "alnum_code_split",
        applied_rules,
    )
    result = replace_with_tracking(
        result,
        _DOTTED_NUMBER_LETTER,
        lambda _m, digits, letter, *_r: f"{digits} {letter}",
        "alnum_code_split",
        applied_rules,
    )
    result = replace_with_tracking(
        result, _VERSION_TAG, lambda _m, digits, *_r: f"v {digits}", "alnum_code_split", applied_rules
    )
    return replace_with_tracking(
        result,
        _WORD_GLUED_NUMBER,
        lambda _m, word, digits, *_r: f"{word} {digits}",
        "alnum_code_split",
        applied_rules,
    )


# --- multipliers --------------------------------------------------------------------------------

_MULTIPLIER_PRICED = js(
    r"(?<![\w.,])(\d{1,4})[ \t]*[x×][ \t]*(?=[€$£]?\d[\d.,]*[ \t]?(?:[€$£]|EUR|USD|mg|g|kg|ml|l|min|Min\.|Std\.|h|s)(?![\p{L}\d]))",
    "gu",
)
_MULTIPLIER_BEFORE_EQUALS = js(r"(?<![\w.,/^-])(\d{1,3})[ \t]?×[ \t]*=[ \t]*(?=[\p{L}\d(€$£-])", "gu")
_CODE_COUNT = js(r"(?<=-\d{1,3})(?=[x×](?![\p{L}\d]))", "gu")


def _rewrite_multipliers(text: str, cfg: NotationPrepassConfig, applied_rules: list[str]) -> str:
    de = cfg.lang == "de"
    result = replace_with_tracking(
        text,
        _MULTIPLIER_PRICED,
        lambda _m, count, *_r: f"{count} {'mal' if de else 'times'} ",
        "multiplier",
        applied_rules,
    )

    def spoken_count(count: str) -> str:
        value = int(count)
        if de:
            return f"{'ein' if value == 1 else cfg.integer_words(value)}mal"
        if value == 1:
            return "once"
        if value == 2:
            return "twice"
        return f"{cfg.integer_words(value)} times"

    result = replace_with_tracking(
        result,
        _MULTIPLIER_BEFORE_EQUALS,
        lambda _m, count, *_r: f"{spoken_count(count)} {WORDS[cfg.lang]['equals']} ",
        "multiplier",
        applied_rules,
    )
    result = replace_with_tracking(
        result,
        js(
            r"(?<![\w.,/^-])(\d{1,3})[ \t]?[x×](?=[ \t]+(?![\d€$£]|[+−=<>*/×÷]|-[ \t]*[\d(]|"
            + WORDS[cfg.lang]["operatorWordSource"] + r"\b)|$|[,.;:!?)])",
            "g",
        ),
        lambda _m, count, *_r: spoken_count(count),
        "multiplier",
        applied_rules,
    )
    return replace_with_tracking(result, _CODE_COUNT, " ", "code_count", applied_rules)


# --- suffixes and decades -----------------------------------------------------------------------

_APOSTROPHE_DECADE = js(r"(?<![\w'’])['’]([1-9]0)s\b", "g")
_SHORT_YEAR = js(r"(?<![\w'’])['’](\d{2})(?![\d\p{L}'’]|[.,]\d)(?![^'’\n]*['’](?!\d{2}(?!\d)|\p{L}))", "gu")
_GERMAN_NUMBER_SUFFIX = js(
    r"(?<![\w.,])(\d{1,4})(?:(er|ern|ers|fach|mal|stel|tel)|-?([Ff]ach(?:e[nrs]?)?|FACH(?:E[NRS]?)?))\b", "g"
)
_ENGLISH_DECADE = js(r"(?<![\w.,])'?(\d|1\d{2}|20\d)0s\b", "g")


def _rewrite_suffixes(text: str, cfg: NotationPrepassConfig, applied_rules: list[str]) -> str:
    if cfg.lang == "de":
        result = replace_with_tracking(text, _APOSTROPHE_DECADE, lambda _m, tens, *_r: f"{tens}er", "decade", applied_rules)
        result = replace_with_tracking(result, _SHORT_YEAR, lambda _m, year, *_r: year, "short_year", applied_rules)

        def german(_match, number_token, bound_suffix, multiple, *_rest):
            value = int(number_token)
            suffix = bound_suffix if bound_suffix is not None else str(multiple).lower()
            if suffix.startswith("er"):
                if value == 1:
                    base = "eins"
                elif len(number_token) == 4:
                    base = cfg.year_words(value)
                else:
                    base = cfg.integer_words(value)
                return f"{base}{suffix}"
            base = "ein" if value == 1 else cfg.integer_words(value)
            return f"{base}{suffix}"

        return replace_with_tracking(result, _GERMAN_NUMBER_SUFFIX, german, "number_suffix_de", applied_rules)

    def english(match, stem, *_rest):
        tens = f"{stem[-1:]}0"
        decade = ENGLISH_DECADES.get(tens)
        if len(stem) == 1:
            return decade if decade is not None else match
        year = int(f"{stem}0")
        if year % 100 == 0:
            return f"{cfg.year_words(year)}s"
        if not decade:
            return match
        if 2000 < year < 2100:
            return f"twenty {decade}"
        spoken_year = cfg.year_words(year)
        last_space = spoken_year.rfind(" ")
        return f"{spoken_year[:last_space + 1]}{decade}"

    return replace_with_tracking(text, _ENGLISH_DECADE, english, "decade_en", applied_rules)


# --- weekdays -----------------------------------------------------------------------------------

_WEEKDAY_MONTH_WORD = (
    r"(?:Jan(?:uary|uar)?|Feb(?:ruary|ruar)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e|i)?|Jul(?:y|i)?|Aug(?:ust)?"
    r"|Sep(?:t|tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?(?![\p{L}])"
)

_WEEKDAY_DATE_OR_TIME_AFTER = {
    "de": js(
        r"^[ \t]+(?:(?:den|dem)[ \t]+\d{1,2}\.|\d{1,2}\.(?:[ \t]*\p{L}|\d)|\d{1,2}:\d{2}|\d{1,2}(?:[ \t]*(?:[–—-]|bis)[ \t]*\d{1,2})"
        r"|\d{1,2}[ \t]*Uhr\b)",
        "u",
    ),
    "en": js(
        r"^[ \t]+(?:\d{1,2}(?:st|nd|rd|th)\b|\d{1,2}[./]\d{1,2}\b|\d{1,2}:\d{2}|\d{1,2}[ \t]*(?:[–—-]|to)[ \t]*\d{1,2}\b"
        r"|\d{1,2}[ \t]*[ap]\.?m\b|" + _WEEKDAY_MONTH_WORD + r"[ \t]+(?:the[ \t]+)?\d{1,2}\b)",
        "u",
    ),
}
_DASHES = js(r"[–—-]", "")
_M_DASH_F = js(r"\bM\s?[–—-]\s?F\b", "g")


def _rewrite_weekdays(text: str, lang: str, applied_rules: list[str], row_source: Optional[str] = None) -> str:
    days = WORDS[lang]["weekdays"]
    keys = sorted(days.keys(), key=lambda key: -len(key))
    abbr = "|".join(key[:1].upper() + key[1:] for key in keys)
    to = WORDS[lang]["rangeTo"]
    and_word = WORDS[lang]["and"]
    ambiguous = set(WORDS[lang]["ambiguousWeekdays"])
    date_or_time_after = _WEEKDAY_DATE_OR_TIME_AFTER[lang]

    def expand(token: str) -> str:
        return days.get(token.lower()) or token

    def is_ambiguous(token: str) -> bool:
        return token.lower() in ambiguous

    joined = js(
        r"\b((?:(?:" + abbr + r")\.?[ \t]?(?:[–—-]|\/|\+|&|,)[ \t]?)+)(" + abbr + r")(?:\.(?![ \t]*$|[ \t]+\p{Lu}))?(?![\p{L}])",
        "gu",
    )
    token_global = js(r"(" + abbr + r")\.?[ \t]?([–—/+&,-])", "g")
    token_single = js(r"(" + abbr + r")\.?[ \t]?([–—/+&,-])", "")

    def joined_callback(match, head, last, *_rest):
        parts = []
        for piece in token_global.match_all(head):
            m = token_single.first(piece)
            if m:
                parts.append((expand(m.group(1)), m.group(2)))
        if not parts:
            return match
        out = ""
        for index, (day, glue_token) in enumerate(parts):
            is_last = index == len(parts) - 1
            if _DASHES.test(glue_token):
                glue = f" {to} "
            elif is_last:
                glue = f" {and_word} "
            else:
                glue = ", "
            out += f"{day}{glue}"
        return f"{out}{expand(last)}"

    result = replace_with_tracking(text, joined, joined_callback, "weekday_range", applied_rules)
    if lang == "en":
        result = replace_with_tracking(result, _M_DASH_F, f"Monday {to} Friday", "weekday_range", applied_rules)
    day_names = "|".join(dict.fromkeys(days.values()))
    result = replace_with_tracking(
        result,
        js(r"\b(" + day_names + r")[ \t]?[–—-][ \t]?(" + day_names + r")(?![\p{L}])", "gu"),
        lambda _m, first, second, *_r: f"{first} {to} {second}",
        "weekday_range",
        applied_rules,
    )
    single = js(
        r"(?<![\p{L}])(" + abbr + r")(\.?,?)(?=([ \t]+(?:\d|den\b|dem\b|" + _WEEKDAY_MONTH_WORD + r")[^\n]{0,16}))",
        "gu",
    )

    def single_callback(match, token, tail, after, *_rest):
        if is_ambiguous(token) and not date_or_time_after.test(after):
            return match
        return f"{expand(token)}{js_replace_first_literal(tail, '.', '')}"

    result = replace_with_tracking(result, single, single_callback, "weekday_abbreviation", applied_rules)
    full_names = "|".join(days.values())
    colon_openers = set(WORDS[lang]["colonSentenceOpeners"])
    row_end = r"\.?(?=:([^\n]{0,16})|,?[ \t]+\d)"
    row_days: set[str] = set()

    def collect_rows(source: str, row_start: str, row_tail: str) -> None:
        rows = js(r"(?:^|\n)" + row_start + r"(" + abbr + r"|" + full_names + r")" + row_tail, "gu")
        for row in rows.finditer(source):
            row_days.add(expand(row.group(1)))

    collect_rows(result, r"[ \t]*", row_end)
    if row_source:
        collect_rows(
            row_source,
            r"[ \t]*(?:\|[ \t]*)?(?:(?:[-*+•]|\d{1,3}[.)])[ \t]+)?(?:\*\*|__|[*_])?",
            r"\.?(?:\*\*|__|[*_])?(?=[ \t]*[:|]|,?[ \t]+\d)",
        )

    def heading(match, start, indent, token, after_colon, *_rest):
        if (
            is_ambiguous(token)
            and len(row_days) < 2
            and not (
                after_colon is not None
                and (token.lower() not in colon_openers or date_or_time_after.test(after_colon))
            )
        ):
            return match
        return f"{start}{indent}{expand(token)}"

    return replace_with_tracking(
        result,
        js(r"(^|\n)([ \t]*)(" + abbr + r")" + row_end, "gu"),
        heading,
        "weekday_abbreviation",
        applied_rules,
    )


# --- plain form of notation ---------------------------------------------------------------------

_DECIMAL_WITHOUT_ZERO_PATTERN = js(r"(?<=^|[\s(\[{\"'„“‚‘»«])\.(\d+)(?![\d\p{L}]|[.,]\d)", "gmu")
_FRACTION_SLASH = js(r"(?<=(?<!\d)\d{1,2})⁄(?=\d{1,3}(?!\d))", "g")
_PLUS_MINUS_WRITTEN = js(
    r"\+[ \t]?\/[ \t]?-(?=[ \t]?\d)|(?<![\d%)\]€$£°²³][ \t]*)(?<!(?<![\p{L}\p{Nd}_])\p{L}[ \t]*)\+-(?=[ \t]?\d)", "gu"
)
_MINUS_PLUS_WRITTEN = js(r"-[ \t]?\/[ \t]?\+(?=[ \t]?\d)", "g")


def apply_notation_normalization(text: str, cfg: NotationPrepassConfig, applied_rules: list[str]) -> str:
    """``applyNotationNormalization`` (Bug 836): notation both languages read alike in its plain form."""
    result = text or ""
    if "." in result:
        result = replace_with_tracking(
            result,
            _DECIMAL_WITHOUT_ZERO_PATTERN,
            lambda _m, digits, *_r: f"0{',' if cfg.lang == 'de' else '.'}{digits}",
            "decimal_zero",
            applied_rules,
        )
    if "⁄" in result:
        result = replace_with_tracking(result, _FRACTION_SLASH, "/", "fraction_slash", applied_rules)
    if "+" in result:
        result = replace_with_tracking(result, _PLUS_MINUS_WRITTEN, "±", "plus_minus_written", applied_rules)
        result = replace_with_tracking(result, _MINUS_PLUS_WRITTEN, "∓", "plus_minus_written", applied_rules)
    return result


# --- magnitude suffixes (Bug 843) ---------------------------------------------------------------

_MAGNITUDE_FUNCTION_WORD_SOURCE = (
    "(?:aus|als|bis|des|das|es|uns|eines|keines|alles|etwas|nichts|ins|ans|aufs|fürs|ums|vors|durchs|übers|is|was|has|this|its|his)"
)
_MAGNITUDE_CURRENCY_SOURCE = r"(?:[€$£]|\b(?:EUR|USD|GBP|CHF)\b)"
_HAS_MAGNITUDE = js(r"\d[ \t]?(?:[kKMB]|bn)(?![\p{L}\d])", "u")
_CURRENCY_IN = js(r"[€$£]|EUR|USD|GBP|CHF", "")
_THOUSANDS_DOTS = js(r"\B(?=(\d{3})+(?!\d))", "g")


def _rewrite_magnitude_suffixes(text: str, cfg: NotationPrepassConfig, applied_rules: list[str]) -> str:
    if cfg.lang != "de" or not _HAS_MAGNITUDE.test(text):
        return text
    pattern = js(
        r"(?<![\p{L}\d.,:§]|\p{L}-)(?<!(?:(?<![\p{L}])(?:" + (cfg.number_label_source or NEVER_SOURCE) + r"|\p{Lu})|§)[ \t]+)"
        r"(?:(?<=" + _MAGNITUDE_CURRENCY_SOURCE + r"[ \t]?)()|)"
        r"(\d{1,3})(?:[.,](\d{1,2}))?(?:([kKMB])|[ \t]?(bn))(?![\p{L}\d]|-(?!\d))"
        r"(?:(?=([ \t]?(?:" + _MAGNITUDE_CURRENCY_SOURCE + r"|\+)|[ \t]+\p{Lu}|[ \t]+(?!" + _MAGNITUDE_FUNCTION_WORD_SOURCE
        + r"\b)\p{Ll}{2,}s\b))|)",
        "gu",
    )

    def callback(match, currency_before, whole, fraction, suffix, spaced_bn, after, *_rest):
        whole = whole or ""
        near_currency = currency_before is not None or (after is not None and _CURRENCY_IN.test(after))
        unit = spaced_bn if spaced_bn is not None else suffix
        if unit in ("K", "B") and not near_currency:
            return match
        if unit == "k" and len(whole) == 1 and fraction is None and not near_currency:
            return match
        if unit == "M" and len(whole) == 1 and fraction is None and after is None:
            return match
        if unit in ("k", "K"):
            value = int(whole) * 1000 + (int(fraction.ljust(3, "0")) if fraction else 0)
            return js_replace(_THOUSANDS_DOTS, str(value), ".")
        return f"{f'{whole},{fraction}' if fraction is not None else whole} {'Mio' if unit == 'M' else 'Mrd'}"

    return replace_with_tracking(text, pattern, callback, "magnitude_suffix", applied_rules)


def apply_notation_prepass(
    text: str, cfg: NotationPrepassConfig, applied_rules: list[str], answer_text: Optional[str] = None
) -> str:
    """``applyNotationPrepass``: magnitudes, identifiers, ranges, per-periods, symbols, ratios,
    multipliers, suffixes, weekdays (in this order). ``answer_text``: the whole answer so far when
    ``text`` is one speech piece of it."""
    result = text or ""
    if not js_trim(result):
        return result
    result = _rewrite_magnitude_suffixes(result, cfg, applied_rules)
    result = _rewrite_identifiers(result, cfg, applied_rules)
    result = _rewrite_ranges(result, cfg, applied_rules)
    result = _rewrite_per_period(result, cfg.lang, applied_rules)
    result = _rewrite_symbols(result, cfg, applied_rules)
    result = _rewrite_ratios_and_counts(result, cfg, applied_rules)
    result = _rewrite_multipliers(result, cfg, applied_rules)
    result = _rewrite_suffixes(result, cfg, applied_rules)
    return _rewrite_weekdays(result, cfg.lang, applied_rules, answer_text)


__all__ = [
    "NotationPrepassConfig",
    "WORDS",
    "apply_notation_normalization",
    "apply_notation_prepass",
    "escape_for_regexp",
    "is_js_integer",
    "is_slash_day_month",
]
