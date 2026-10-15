"""Port of Genesis3 ``src/utils/ttsLanguageOptimizer/notationPrepass.ts``.

Rewrites notation that carries a spoken word (range dashes, "per" slashes,
signs, symbols, glued codes) into that word BEFORE the value passes of de/en
turn digits into number words. Rule labels match the TS source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from .jsre import is_js_integer, js, js_number, js_replace, js_replace_first_literal, js_trim
from .shared import replace_with_tracking

NotationLanguage = Literal["de", "en"]


@dataclass(frozen=True)
class NotationPrepassConfig:
    lang: NotationLanguage
    integer_words: Callable[[int], str]
    digit_words: Callable[[str], str]
    year_words: Callable[[int], str]


WORDS: dict[str, dict] = {
    "de": {
        "rangeTo": "bis",
        "and": "und",
        "per": "pro",
        "dot": "Punkt",
        "at": "at",
        "plus": "plus",
        "minus": "minus",
        "plusMinus": "plus minus",
        "about": "etwa",
        "atLeast": "mindestens",
        "atMost": "höchstens",
        "greaterThan": "mehr als",
        "lessThan": "weniger als",
        "number": "Nummer",
        "section": "Paragraf",
        "degrees": "Grad",
        "cents": "Cent",
        "yen": "Yen",
        "arrowTo": "nach",
        "ratio": "zu",
        "outOf": "von",
        "port": "Port",
        "equals": "gleich",
        "atSign": "bei",
        "power": "hoch",
        "hashKey": "Raute",
        "periods": {
            "s": "Sekunde", "sek": "Sekunde", "sek.": "Sekunde", "sec": "Sekunde",
            "min": "Minute", "min.": "Minute",
            "h": "Stunde", "std": "Stunde", "std.": "Stunde", "stunde": "Stunde",
            "tag": "Tag", "woche": "Woche", "monat": "Monat", "jahr": "Jahr", "a": "Jahr",
            "l": "Liter", "ml": "Milliliter", "dl": "Deziliter",
            "kg": "Kilogramm", "g": "Gramm", "km": "Kilometer", "m": "Meter",
            "m²": "Quadratmeter", "m2": "Quadratmeter", "m³": "Kubikmeter", "m3": "Kubikmeter",
            "kwh": "Kilowattstunde", "stk": "Stück", "stk.": "Stück", "person": "Person", "nacht": "Nacht",
        },
        "weekdays": {
            "mo": "Montag", "di": "Dienstag", "mi": "Mittwoch", "do": "Donnerstag",
            "fr": "Freitag", "sa": "Samstag", "so": "Sonntag",
        },
        "unicodeFractions": {
            "½": ("ein halb", "einhalb"),
            "¼": ("ein Viertel", "einviertel"),
            "¾": ("drei Viertel", "dreiviertel"),
            "⅓": ("ein Drittel", "eindrittel"),
            "⅔": ("zwei Drittel", "zweidrittel"),
            "⅛": ("ein Achtel", "einachtel"),
        },
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
        "about": "about",
        "atLeast": "at least",
        "atMost": "at most",
        "greaterThan": "more than",
        "lessThan": "less than",
        "number": "number",
        "section": "section",
        "degrees": "degrees",
        "cents": "cents",
        "yen": "yen",
        "arrowTo": "to",
        "ratio": "to",
        "outOf": "out of",
        "port": "port",
        "equals": "equals",
        "atSign": "at",
        "power": "to the power of",
        "hashKey": "hash",
        "periods": {
            "s": "second", "sec": "second", "sec.": "second",
            "min": "minute", "min.": "minute",
            "h": "hour", "hr": "hour", "hr.": "hour", "hour": "hour",
            "day": "day", "week": "week", "wk": "week", "mo": "month", "month": "month",
            "yr": "year", "yr.": "year", "year": "year", "a": "year",
            "l": "liter", "ml": "milliliter", "dl": "deciliter", "gal": "gallon",
            "kg": "kilogram", "g": "gram", "km": "kilometer", "m": "meter", "mi": "mile", "mile": "mile",
            "m²": "square meter", "m2": "square meter", "m³": "cubic meter", "m3": "cubic meter",
            "kwh": "kilowatt-hour", "person": "person", "night": "night", "unit": "unit", "pc": "piece", "pcs": "piece",
        },
        "weekdays": {
            "mon": "Monday", "tue": "Tuesday", "tues": "Tuesday", "wed": "Wednesday", "thu": "Thursday",
            "thur": "Thursday", "thurs": "Thursday", "fri": "Friday", "sat": "Saturday", "sun": "Sunday",
        },
        "unicodeFractions": {
            "½": ("one half", "and a half"),
            "¼": ("one quarter", "and a quarter"),
            "¾": ("three quarters", "and three quarters"),
            "⅓": ("one third", "and a third"),
            "⅔": ("two thirds", "and two thirds"),
            "⅛": ("one eighth", "and an eighth"),
        },
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


def escape_for_regexp(value: str) -> str:
    return js_replace(_REGEX_SPECIALS, value, "\\$&")


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
    r"(?<![\p{L}\d.,:/–—+)-])([€$£]?)(\d(?:[\d.,:]*\d)?)(\.?)([ \t]?(?:%|°[CF]?|€|\$|£|Uhr|[ap]\.?m\.?))?([ \t]*)([–—-])([ \t]*)([€$£]?)(\d(?:[\d.,:]*\d)?)(?![\d.,:]*[–—-]\d)(?![ \t]+\d)(?=([ \t]?(?:Uhr|[ap]\.?m\.?)\b)?)",
    "giu",
)
_CLOCK_TIME = js(r"^\d{1,2}:\d{2}(?::\d{2})?$", "")
_CLOCK_UNIT = js(r"Uhr|m\.?$", "i")


def _is_clock_time(token: str) -> bool:
    return _CLOCK_TIME.test(token)


def _digits_of(token: str) -> str:
    return js_replace(_NON_DIGITS, token or "", "")


def _rewrite_ranges(text: str, lang: str, applied_rules: list[str]) -> str:
    to = WORDS[lang]["rangeTo"]

    def callback(match, left_currency, left, left_dot, left_unit, space_before, dash, space_after,
                 right_currency, right, right_unit, *_rest):
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

    return replace_with_tracking(text, RANGE_PATTERN, callback, "range_connector", applied_rules)


# --- per-period slashes -------------------------------------------------------------------------

_PER_HUNDRED_KM = js(r"(?<=\d[ \t]?)l[ \t]?\/[ \t]?100[ \t]?km\b", "gi")
_RPM = js(r"\bU[ \t]?\/[ \t]?min\b", "g")


def _period_pattern(lang: str):
    keys = sorted(WORDS[lang]["periods"].keys(), key=lambda key: -len(key))
    period_keys = "|".join(escape_for_regexp(key) for key in keys)
    return js(f"(?<=[\\p{{L}}\\d%€$£²³])[ \\t]?\\/[ \\t]?({period_keys})(?![\\p{{L}}/])", "giu")


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

    def callback(match, period, *_rest):
        if len(period) == 1 and period != period.lower() and period != "L":
            return match
        label = words["periods"].get(period.lower())
        return f" {words['per']} {label}" if label else match

    return replace_with_tracking(result, _period_pattern(lang), callback, "per_period", applied_rules)


# --- signs and symbols --------------------------------------------------------------------------

_UNICODE_MINUS = js(r"\u2212(?=\s?\d)", "g")
_TIMES_SIGN = js(r"(\d[\d.,]*)[ \t]?\u00d7[ \t]?(?=(\d+(?:[.,]\d+)*))", "g")
_PLUS_MINUS = js(r"±\s?(?=\d)", "g")
_LEADING_PLUS = js(r"(?<![\w+])\+(?=\d[\d.,]*(?![\d.,]*[ \t(]+\d)(?:[ \t]*(?:%|°|€|\$|£|\p{L})|[ \t]*$|[.,;:!?)]))", "gu")
_INFIX_PLUS_GLUED = js(r"(?<=[\p{L}\d])\+(?=[\p{L}\d])(?!\d{2}:?\d{2}\b)", "gu")
_APPROX = js(r"[≈~]\s?(?=[\d€$£])", "g")
_GE = js(r"≥\s?(?=\d)", "g")
_LE = js(r"≤\s?(?=\d)", "g")
_GE_ASCII = js(r"(?<=\s)>=\s?(?=\d)", "g")
_LE_ASCII = js(r"(?<=\s)<=\s?(?=\d)", "g")
_GT_ASCII = js(r"(?<=\s)>\s?(?=\d)", "g")
_LT_ASCII = js(r"(?<=\s)<\s?(?=\d)", "g")
_NUMBER_SIGN = js(r"(?<![\w&])#(?![0-9a-fA-F]{6}\b|[0-9a-fA-F]{8}\b)(?=\d|[A-Z]{1,4}\d)", "g")
_SECTION_SIGN = js(r"§§?\s?(?=\d)", "g")
_AMPERSAND = js(r"(?<=\S)\s&\s(?=\S)", "g")
_ARROW = js(r"\s?[→➔➜⇒]\s?", "g")
_BARE_DEGREE = js(r"(?<=\d)\s?[°º](?![ \t]?[CFK]\b)", "g")
_UNIT_DIMENSION = js(r"(?<=[°%])[ \t]?[×x][ \t]?(?=\d)|(?<=\p{L})[ \t]?×[ \t]?(?=\d)", "gu")
_TRAILING_PLUS = js(r"(?<=[\p{L}\d])\+(?=[ \t]|$|[,.;:!?)\"”])", "gu")
_TRAILING_MINUS = js(r"(?<=\d|\b[A-F])[-−](?=[ \t]|$|[,.;:!?)\"”])", "g")
_INFIX_PLUS_SPACED = js(r"(?<=[\p{L}\d])[ \t]+\+[ \t]+(?=[\p{L}\d])", "gu")
_EQUALS = js(r"(?<=[\p{L}\d°%€$£)])[ \t]*(?<![=<>!])=(?![=>])[ \t]*(?=[\p{L}\d(-])", "gu")
_MENU_PATH = js(r"(?<=\p{L})[ \t]*>[ \t]*(?=\p{L})", "gu")
_AT_SIGN = js(r"(?<=\s)@(?=\s)", "g")
_SUPERSCRIPT_2 = js(r"(?<=[a-z])\^2\b", "g")
_SUPERSCRIPT_3 = js(r"(?<=[a-z])\^3\b", "g")
_POWER_CARET = js(r"(?<=\d)\^(-?\d+)", "g")
_POWER_SUPERSCRIPT = js(r"(?<=\d)([⁰¹²³⁴⁵⁶⁷⁸⁹]+)", "g")
_SUPERSCRIPT_DIGIT = js(r"[⁰¹²³⁴⁵⁶⁷⁸⁹]", "g")
_MIDDLE_DOT = js(r"(?<=\d)[ \t]?·[ \t]?(?=\d)", "g")
_HASH_KEY = js(r"(?<=\d)#(?=[ \t]|$|[,.;:!?)])", "g")
_CENT_SIGN = js(r"(?<=\d)\s?¢", "g")
_YEN_SIGN = js(r"¥\s?(\d[\d.,]*\d|\d)", "g")
_DASH_CENTS = js(r"(?<=\d)[.,]-(?=[ \t]?(?:€|EUR|CHF|Fr\.))", "g")
_UNICODE_FRACTION = js(r"(?:(\d+)[ \t]?)?([½¼¾⅓⅔⅛])", "g")
_ALL_DIGITS = js(r"^\d+$", "")
_SUPERSCRIPTS = "⁰¹²³⁴⁵⁶⁷⁸⁹"


def _rewrite_symbols(text: str, cfg: NotationPrepassConfig, applied_rules: list[str]) -> str:
    lang = cfg.lang
    w = WORDS[lang]
    result = text
    result = replace_with_tracking(result, _UNICODE_MINUS, "-", "unicode_minus", applied_rules)

    def times(match, left, right, *_rest):
        dimension = _ALL_DIGITS.test(left) and _ALL_DIGITS.test(right)
        word = "mal" if lang == "de" else ("by" if dimension else "times")
        return f"{left} {word} "

    result = replace_with_tracking(result, _TIMES_SIGN, times, "times_sign", applied_rules)
    result = replace_with_tracking(result, _PLUS_MINUS, f"{w['plusMinus']} ", "plus_minus", applied_rules)
    result = replace_with_tracking(result, _LEADING_PLUS, f"{w['plus']} ", "leading_plus", applied_rules)
    result = replace_with_tracking(result, _INFIX_PLUS_GLUED, f" {w['plus']} ", "infix_plus", applied_rules)
    result = replace_with_tracking(result, _APPROX, f"{w['about']} ", "approx_sign", applied_rules)
    result = replace_with_tracking(result, _GE, f"{w['atLeast']} ", "comparison_sign", applied_rules)
    result = replace_with_tracking(result, _LE, f"{w['atMost']} ", "comparison_sign", applied_rules)
    result = replace_with_tracking(result, _GE_ASCII, f"{w['atLeast']} ", "comparison_sign", applied_rules)
    result = replace_with_tracking(result, _LE_ASCII, f"{w['atMost']} ", "comparison_sign", applied_rules)
    result = replace_with_tracking(result, _GT_ASCII, f"{w['greaterThan']} ", "comparison_sign", applied_rules)
    result = replace_with_tracking(result, _LT_ASCII, f"{w['lessThan']} ", "comparison_sign", applied_rules)
    result = replace_with_tracking(result, _NUMBER_SIGN, f"{w['number']} ", "number_sign", applied_rules)
    result = replace_with_tracking(result, _SECTION_SIGN, f"{w['section']} ", "section_sign", applied_rules)
    result = replace_with_tracking(result, _AMPERSAND, f" {w['and']} ", "ampersand", applied_rules)
    result = replace_with_tracking(result, _ARROW, f" {w['arrowTo']} ", "arrow", applied_rules)
    result = replace_with_tracking(result, _BARE_DEGREE, f" {w['degrees']}", "bare_degree", applied_rules)
    result = replace_with_tracking(
        result, _UNIT_DIMENSION, " mal " if lang == "de" else " by ", "unit_dimension", applied_rules
    )
    result = replace_with_tracking(result, _TRAILING_PLUS, f" {w['plus']}", "trailing_plus", applied_rules)
    result = replace_with_tracking(result, _TRAILING_MINUS, f" {w['minus']}", "trailing_minus", applied_rules)
    result = replace_with_tracking(result, _INFIX_PLUS_SPACED, f" {w['plus']} ", "infix_plus", applied_rules)
    result = replace_with_tracking(result, _EQUALS, f" {w['equals']} ", "equals_sign", applied_rules)
    result = replace_with_tracking(result, _MENU_PATH, ", ", "menu_path", applied_rules)
    result = replace_with_tracking(result, _AT_SIGN, w["atSign"], "at_sign", applied_rules)
    result = replace_with_tracking(result, _SUPERSCRIPT_2, "²", "superscript_unit", applied_rules)
    result = replace_with_tracking(result, _SUPERSCRIPT_3, "³", "superscript_unit", applied_rules)
    result = replace_with_tracking(
        result, _POWER_CARET, lambda match, exponent, *_r: f" {w['power']} {exponent}", "power", applied_rules
    )

    def superscript_power(match, digits, *_rest):
        plain = js_replace(_SUPERSCRIPT_DIGIT, digits, lambda glyph, *_a: str(_SUPERSCRIPTS.index(glyph)))
        return f" {w['power']} {plain}"

    result = replace_with_tracking(result, _POWER_SUPERSCRIPT, superscript_power, "power", applied_rules)
    result = replace_with_tracking(
        result, _MIDDLE_DOT, " mal " if lang == "de" else " times ", "middle_dot_times", applied_rules
    )
    result = replace_with_tracking(result, _HASH_KEY, f" {w['hashKey']}", "hash_key", applied_rules)
    result = replace_with_tracking(result, _CENT_SIGN, f" {w['cents']}", "cent_sign", applied_rules)
    result = replace_with_tracking(
        result, _YEN_SIGN, lambda match, amount, *_r: f"{amount} {w['yen']}", "yen_sign", applied_rules
    )
    result = replace_with_tracking(result, _DASH_CENTS, "", "dash_cents", applied_rules)
    fractions = w["unicodeFractions"]

    def fraction(match, integer_token, glyph, *_rest):
        entry = fractions.get(glyph)
        if not entry:
            return match
        alone, after_integer = entry
        if not integer_token:
            return alone
        value = int(integer_token)
        words = "ein" if (lang == "de" and value == 1) else cfg.integer_words(value)
        return f"{words}{after_integer}" if lang == "de" else f"{words} {after_integer}"

    return replace_with_tracking(result, _UNICODE_FRACTION, fraction, "unicode_fraction", applied_rules)


# --- ratios, scores, counts -----------------------------------------------------------------------

_RATIO_COLON = js(r"(?<![\d:.,])(\d{1,3}(?:[.,]\d+)?):(\d{1,6})(?!\d|[.,]\d|:)", "g")
_TWENTY_FOUR_SEVEN = js(r"\b24\/7\b", "g")
_BLOOD_PRESSURE = js(r"(?<![\d./,])(\d{2,3})\/(\d{2,3})(?=[ \t]?mmHg\b)", "g")
_COUNT_SLASH = js(r"(?<![\d./,\p{L}])(\d{1,3}(?:[.,]\d+)?)\/(\d{1,3})(?![\d/.,%]|\s?\/)", "gu")


def _rewrite_ratios_and_counts(text: str, cfg: NotationPrepassConfig, applied_rules: list[str]) -> str:
    lang = cfg.lang
    w = WORDS[lang]

    def ratio(match, left, right, *_rest):
        if _ALL_DIGITS.test(left) and len(right) == 2 and int(left) <= 23 and int(right) <= 59:
            return match
        spoken_left = cfg.integer_words(int(left)) if _ALL_DIGITS.test(left) else left
        return f"{spoken_left} {w['ratio']} {cfg.integer_words(int(right))}"

    result = replace_with_tracking(text, _RATIO_COLON, ratio, "ratio_colon", applied_rules)
    result = replace_with_tracking(
        result,
        _TWENTY_FOUR_SEVEN,
        "24 7" if lang == "de" else "twenty-four seven",
        "twenty_four_seven",
        applied_rules,
    )
    result = replace_with_tracking(
        result,
        _BLOOD_PRESSURE,
        lambda match, systolic, diastolic, *_r: f"{systolic} {'zu' if lang == 'de' else 'over'} {diastolic}",
        "blood_pressure",
        applied_rules,
    )

    def count_slash(match, left, right, *_rest):
        numerator = js_number(js_replace_first_literal(left, ",", "."))
        denominator = js_number(right)
        is_fraction = (
            _ALL_DIGITS.test(left)
            and numerator < denominator
            and (denominator <= 12 or denominator == 100)
            and denominator >= 2
        )
        if is_fraction:
            return match
        if (
            lang == "en"
            and _ALL_DIGITS.test(left)
            and 1 <= numerator <= 12
            and 1 <= denominator <= 31
        ):
            return match
        return f"{left} {w['outOf']} {right}"

    return replace_with_tracking(result, _COUNT_SLASH, count_slash, "count_slash", applied_rules)


# --- identifiers --------------------------------------------------------------------------------

_DIGIT_CHAR = js(r"\d", "")
_LATIN_CHAR = js(r"[A-Za-z]", "")
_HEX_LITERAL = js(r"\b0x([0-9a-fA-F]{2,16})\b", "g")
_IBAN = js(r"\b([A-Z]{2}\d{2})((?:[ \t]?[A-Z0-9]{4}){2,7}(?:[ \t]?[A-Z0-9]{1,4})?)\b", "g")
_WS_RUN = js(r"\s+", "g")
_GROUPS_OF_FOUR = js(r".{1,4}", "g")
_DIGIT_GROUPS_SPACED = js(r"\b([1-9]\d{3}(?:[ \t]\d{4}){2,}(?:[ \t]\d{1,4})?)\b", "g")
_SPACE_OR_TAB = js(r"[ \t]", "")
_DIGIT_GROUPS_DASHED = js(r"(?<![\w.+])(\d{1,}(?:-\d{2,}){2,})(?![\w-])", "g")
_ISO_DATE_SHAPE = js(r"^\d{4}-\d{2}-\d{2}$", "")
_EMAIL = js(r"(?<![\w.])([\w+-]+(?:\.[\w+-]+)*)@((?:[\w-]+\.)+[a-z]{2,})\b", "gi")
_ALNUM_CODE = js(r"(?<![#\w:])(?<!\d{4}-\d{2}-)(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{2,24}\b", "g")
_ALNUM_BOUNDARY = js(r"(\d)(?=[A-Z])|([A-Z])(?=\d)", "g")
_HOUSE_NUMBER_LETTER = js(r"(?<=\s)(\d{1,3})([a-rt-z])\b", "g")
_VERSION_TAG = js(r"(?<![\w.])v(\d{1,3})\b(?!\.\d|\d)", "g")
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


def _rewrite_identifiers(text: str, cfg: NotationPrepassConfig, applied_rules: list[str]) -> str:
    result = text
    result = replace_with_tracking(
        result,
        _HEX_LITERAL,
        lambda match, hex_digits, *_r: f"{cfg.digit_words('0')} x {_spell_identifier(hex_digits, cfg)}",
        "hex_literal",
        applied_rules,
    )

    def iban(match, head, body, *_rest):
        groups = _GROUPS_OF_FOUR.match_all(js_replace(_WS_RUN, body, ""))
        return ", ".join([_spell_identifier(head, cfg), *(_spell_identifier(group, cfg) for group in groups)])

    result = replace_with_tracking(result, _IBAN, iban, "iban", applied_rules)
    result = replace_with_tracking(
        result,
        _DIGIT_GROUPS_SPACED,
        lambda match, chain, *_r: ", ".join(cfg.digit_words(group) for group in _SPACE_OR_TAB.split(chain)),
        "digit_groups",
        applied_rules,
    )

    def dashed(match, chain, *_rest):
        if _ISO_DATE_SHAPE.test(chain):
            return match
        return ", ".join(cfg.digit_words(group) for group in chain.split("-"))

    result = replace_with_tracking(result, _DIGIT_GROUPS_DASHED, dashed, "digit_groups", applied_rules)
    w = WORDS[cfg.lang]
    result = replace_with_tracking(
        result,
        _EMAIL,
        lambda match, local, domain, *_r: (
            f"{local.replace('.', ' ' + w['dot'] + ' ')} {w['at']} {domain.replace('.', ' ' + w['dot'] + ' ')}"
        ),
        "email_address",
        applied_rules,
    )

    def alnum(token, *_rest):
        if len(token) >= 8:
            return _spell_identifier(token, cfg)
        return js_replace(_ALNUM_BOUNDARY, token, "$1$2 ")

    result = replace_with_tracking(result, _ALNUM_CODE, alnum, "alnum_code_split", applied_rules)
    result = replace_with_tracking(
        result,
        _HOUSE_NUMBER_LETTER,
        lambda match, digits, letter, *_r: f"{digits} {letter}",
        "alnum_code_split",
        applied_rules,
    )
    result = replace_with_tracking(
        result, _VERSION_TAG, lambda match, digits, *_r: f"v {digits}", "alnum_code_split", applied_rules
    )
    result = replace_with_tracking(
        result,
        _WORD_GLUED_NUMBER,
        lambda match, word, digits, *_r: f"{word} {digits}",
        "alnum_code_split",
        applied_rules,
    )
    return result


# --- multipliers --------------------------------------------------------------------------------

_MULTIPLIER_PRICED = js(
    r"(?<![\w.,])(\d{1,4})[ \t]*[x×][ \t]*(?=[€$£]?\d[\d.,]*[ \t]?(?:[€$£]|EUR|USD|mg|g|kg|ml|l|min|Min\.|Std\.|h|s)(?![\p{L}\d]))",
    "gu",
)
_MULTIPLIER_BARE = js(r"(?<![\w.,])(\d{1,3})[ \t]?[x×](?=[ \t]+(?![\d€$£])|$|[,.;:!?)])", "g")


def _rewrite_multipliers(text: str, cfg: NotationPrepassConfig, applied_rules: list[str]) -> str:
    de = cfg.lang == "de"
    result = replace_with_tracking(
        text,
        _MULTIPLIER_PRICED,
        lambda match, count, *_r: f"{count} {'mal' if de else 'times'} ",
        "multiplier",
        applied_rules,
    )

    def bare(match, count, *_rest):
        value = int(count)
        if de:
            base = "ein" if value == 1 else cfg.integer_words(value)
            return f"{base}mal"
        if value == 1:
            return "once"
        if value == 2:
            return "twice"
        return f"{cfg.integer_words(value)} times"

    return replace_with_tracking(result, _MULTIPLIER_BARE, bare, "multiplier", applied_rules)


# --- suffixes and decades -----------------------------------------------------------------------

_GERMAN_NUMBER_SUFFIX = js(r"(?<![\w.,])(\d{1,4})(er|ern|ers|fach|mal|stel|tel)\b", "g")
_ENGLISH_DECADE = js(r"(?<![\w.,])'?(\d|1\d{2}|20\d)0s\b", "g")


def _rewrite_suffixes(text: str, cfg: NotationPrepassConfig, applied_rules: list[str]) -> str:
    if cfg.lang == "de":

        def german(match, number_token, suffix, *_rest):
            value = int(number_token)
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

        return replace_with_tracking(text, _GERMAN_NUMBER_SUFFIX, german, "number_suffix_de", applied_rules)

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
        spoken_year = cfg.year_words(year)
        last_space = spoken_year.rfind(" ")
        return f"{spoken_year[:last_space + 1]}{decade}"

    return replace_with_tracking(text, _ENGLISH_DECADE, english, "decade_en", applied_rules)


# --- weekdays -----------------------------------------------------------------------------------

_DASHES = js(r"[–—-]", "")
_M_DASH_F = js(r"\bM\s?[–—-]\s?F\b", "g")


def _weekday_patterns(lang: str):
    days = WORDS[lang]["weekdays"]
    keys = sorted(days.keys(), key=lambda key: -len(key))
    abbr = "|".join(key[:1].upper() + key[1:] for key in keys)
    joined = js(
        f"\\b((?:(?:{abbr})\\.?[ \\t]?(?:[–—-]|\\/|\\+|&|,)[ \\t]?)+)({abbr})(?:\\.(?![ \\t]*$|[ \\t]+\\p{{Lu}}))?(?![\\p{{L}}])",
        "gu",
    )
    token_global = js(f"({abbr})\\.?[ \\t]?([–—/+&,-])", "g")
    token_single = js(f"({abbr})\\.?[ \\t]?([–—/+&,-])", "")
    single = js(
        f"(?<![\\p{{L}}])({abbr})(\\.?,?)(?=[ \\t]+(?:\\d|den\\b|dem\\b|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec))",
        "gu",
    )
    heading = js(f"(^|\\n)([ \\t]*)({abbr})\\.?(?=:)", "gu")
    return joined, token_global, token_single, single, heading


def _rewrite_weekdays(text: str, lang: str, applied_rules: list[str]) -> str:
    days = WORDS[lang]["weekdays"]
    to = WORDS[lang]["rangeTo"]
    and_word = WORDS[lang]["and"]
    joined, token_global, token_single, single, heading = _weekday_patterns(lang)

    def expand(token: str) -> str:
        return days.get(token.lower()) or token

    def joined_callback(match, head, last, *_rest):
        tokens = token_global.match_all(head)
        parts = []
        for piece in tokens:
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
    result = replace_with_tracking(
        result,
        single,
        lambda match, token, tail, *_r: f"{expand(token)}{js_replace_first_literal(tail, '.', '')}",
        "weekday_abbreviation",
        applied_rules,
    )
    result = replace_with_tracking(
        result,
        heading,
        lambda match, start, indent, token, *_r: f"{start}{indent}{expand(token)}",
        "weekday_abbreviation",
        applied_rules,
    )
    return result


def apply_notation_prepass(text: str, cfg: NotationPrepassConfig, applied_rules: list[str]) -> str:
    """``applyNotationPrepass``: identifiers, ranges, per-periods, symbols, ratios,
    multipliers, suffixes, weekdays (in this order)."""
    result = text or ""
    if not js_trim(result):
        return result
    result = _rewrite_identifiers(result, cfg, applied_rules)
    result = _rewrite_ranges(result, cfg.lang, applied_rules)
    result = _rewrite_per_period(result, cfg.lang, applied_rules)
    result = _rewrite_symbols(result, cfg, applied_rules)
    result = _rewrite_ratios_and_counts(result, cfg, applied_rules)
    result = _rewrite_multipliers(result, cfg, applied_rules)
    result = _rewrite_suffixes(result, cfg, applied_rules)
    result = _rewrite_weekdays(result, cfg.lang, applied_rules)
    return result


__all__ = ["NotationPrepassConfig", "apply_notation_prepass", "escape_for_regexp", "is_js_integer"]
