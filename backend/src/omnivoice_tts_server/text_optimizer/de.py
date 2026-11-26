"""Port of Genesis3 ``src/utils/ttsLanguageOptimizer/de.ts`` (German replace passes).

Pass labels, order and spoken forms follow the TS source one to one so the
golden outputs of the TS implementation can be compared directly. The ``hint``
callbacks of the TS passes are not ported (they only feed the hint injector).
"""

from __future__ import annotations

from typing import Optional

from .jsre import is_js_integer, js, js_number, js_replace, js_trim
from .markdown_line_structure import MarkdownSpokenWords
from .notation_prepass import (
    NotationPrepassConfig,
    apply_notation_normalization,
    apply_notation_prepass,
    escape_for_regexp,
    is_slash_day_month,
)
from .pair_context import GERMAN_PAIR_CONTEXT
from .shared import (
    EMAIL_ADDRESS_PATTERN_SOURCE,
    HELD_DAY_MONTH_LIMIT,
    LIST_OPENER_PAIR,
    PHONE_NUMBER_PATTERN_SOURCE,
    TTS_DAY_MONTH_HELD_RULE,
    VERSION_PRE_RELEASE_TAG_SOURCE,
    VERSION_WILDCARD_SOURCE,
    SpeakPass,
    TtsAlgorithmOptimizationResult,
    TtsOptimizeOptions,
    finalize_optimized_text,
    held_day_month_placeholder,
    holds_dot_name_before_extension,
    parse_phone_number_match,
    replace_ordered_sequence_markers,
    replace_with_tracking,
    restore_held_day_months,
)

# The words of a task item's state and of a status mark (Bug 826).
GERMAN_MARKDOWN_WORDS = MarkdownSpokenWords(task_open='Offen', task_done='Erledigt', status_yes='Ja', status_no='Nein', status_warning='Achtung')


GERMAN_DIGITS = ["null", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun"]
GERMAN_COMPOUND_DIGITS = ["", "ein", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun"]
GERMAN_TEENS = ["zehn", "elf", "zwölf", "dreizehn", "vierzehn", "fünfzehn", "sechzehn", "siebzehn", "achtzehn", "neunzehn"]
GERMAN_TENS = ["", "zehn", "zwanzig", "dreißig", "vierzig", "fünfzig", "sechzig", "siebzig", "achtzig", "neunzig"]
GERMAN_MONTHS = [
    "Januar", "Februar", "März", "April", "Mai", "Juni",
    "Juli", "August", "September", "Oktober", "November", "Dezember",
]

GERMAN_FRACTION_DENOMINATORS = {
    2: "halb", 3: "Drittel", 4: "Viertel", 5: "Fünftel", 6: "Sechstel", 7: "Siebtel",
    8: "Achtel", 9: "Neuntel", 10: "Zehntel", 11: "Elftel", 12: "Zwölftel", 100: "Hundertstel",
}

GERMAN_MONTH_NAME_TO_INDEX = {
    "januar": 1, "februar": 2, "maerz": 3, "april": 4, "mai": 5, "juni": 6,
    "juli": 7, "august": 8, "september": 9, "oktober": 10, "november": 11, "dezember": 12,
}

_EURO = {"unitSingular": "Euro", "unitPlural": "Euro", "subSingular": "Cent", "subPlural": "Cent"}
_DOLLAR = {"unitSingular": "Dollar", "unitPlural": "Dollar", "subSingular": "Cent", "subPlural": "Cent"}
_POUND = {"unitSingular": "Pfund", "unitPlural": "Pfund", "subSingular": "Pence", "subPlural": "Pence"}
# Bug 843: the Swiss franc and the yen; the yen names no subunit in use.
_FRANC = {"unitSingular": "Franken", "unitPlural": "Franken", "subSingular": "Rappen", "subPlural": "Rappen"}
_YEN = {"unitSingular": "Yen", "unitPlural": "Yen", "subSingular": "", "subPlural": ""}
GERMAN_CURRENCY_MAP = {
    "EUR": _EURO, "EURO": _EURO, "€": _EURO,
    "USD": _DOLLAR, "DOLLAR": _DOLLAR, "$": _DOLLAR,
    "GBP": _POUND, "PFUND": _POUND, "£": _POUND,
    "CHF": _FRANC, "JPY": _YEN, "YEN": _YEN,
}

# Bug 843: the currency glyphs and codes of the money passes, and the dative prepositions after
# which exactly one takes "einem"/"einer".
GERMAN_CURRENCY_SOURCE = r"(?:€|\$|£|(?:EURO?|USD|DOLLAR|GBP|PFUND|CHF|JPY|YEN)\b)"
# Bug 435: the names of the currencies of an exchange pair ("EUR/USD" is "Euro zu US-Dollar").
GERMAN_CURRENCY_PAIR_NAMES = {"EUR": "Euro", "USD": "US-Dollar", "GBP": "Pfund", "CHF": "Franken", "JPY": "Yen"}
GERMAN_CURRENCY_PAIR_CODE_SOURCE = "|".join(GERMAN_CURRENCY_PAIR_NAMES)
GERMAN_DATIVE_BEFORE_SOURCE = r"(?<![\p{L}])(?:[Nn]ach|[Mm]it|[Ss]eit|[Vv]on|[Vv]or|[Uu]nter|[Ii]n|[Aa]us|[Bb]ei|[Zz]u)[ \t]+"


def _month_label(month) -> Optional[str]:
    """``GERMAN_MONTHS[month - 1]`` without Python's negative indexing."""
    if isinstance(month, (int, float)) and month == month and float(month).is_integer() and 1 <= month <= 12:
        return GERMAN_MONTHS[int(month) - 1]
    return None


def _js_str(value) -> str:
    """Template-literal interpolation of a possibly undefined value."""
    return "undefined" if value is None else str(value)


# --- number words ---------------------------------------------------------------------------------

def german_under_hundred(value: int, standalone: bool = True) -> str:
    if value < 10:
        if value == 1:
            return "eins" if standalone else "ein"
        return GERMAN_DIGITS[value] if standalone else GERMAN_COMPOUND_DIGITS[value]
    if value < 20:
        return GERMAN_TEENS[value - 10]
    tens, ones = divmod(value, 10)
    if ones == 0:
        return GERMAN_TENS[tens]
    return f"{GERMAN_COMPOUND_DIGITS[ones]}und{GERMAN_TENS[tens]}"


def german_under_thousand(value: int) -> str:
    if value < 100:
        return german_under_hundred(value, True)
    hundreds, remainder = divmod(value, 100)
    hundred_part = f"{'ein' if hundreds == 1 else GERMAN_COMPOUND_DIGITS[hundreds]}hundert"
    if remainder == 0:
        return hundred_part
    return f"{hundred_part}{german_under_hundred(remainder, True)}"


def german_under_million(value: int) -> str:
    if value < 1000:
        return german_under_thousand(value)
    thousands, remainder = divmod(value, 1000)
    thousand_part = f"{'ein' if thousands == 1 else german_under_thousand(thousands)}tausend"
    if remainder == 0:
        return thousand_part
    return f"{thousand_part}{german_under_thousand(remainder)}"


_GERMAN_SCALES = (
    (1000000000000000, "Billiarde", "Billiarden"),
    (1000000000000, "Billion", "Billionen"),
    (1000000000, "Milliarde", "Milliarden"),
    (1000000, "Million", "Millionen"),
)


def integer_to_german_words(value: int) -> str:
    value = int(value)
    if value == 0:
        return "null"
    if value < 0:
        return f"minus {integer_to_german_words(-value)}"
    if value < 1000000:
        return german_under_million(value)
    remaining = value
    parts: list[str] = []
    for scale, singular, plural in _GERMAN_SCALES:
        if remaining < scale:
            continue
        count, remaining = divmod(remaining, scale)
        spoken_count = "eine" if count == 1 else integer_to_german_words(count)
        unit = singular if count == 1 else plural
        parts.append(f"{spoken_count} {unit}")
    if remaining > 0:
        parts.append(integer_to_german_words(remaining))
    return " ".join(parts)


def year_to_german_words(year: int) -> str:
    year = int(year)
    if 1100 <= year < 2000:
        first_part, second_part = divmod(year, 100)
        prefix = german_under_hundred(first_part, True)
        if second_part == 0:
            return f"{prefix}hundert"
        return f"{prefix}hundert{german_under_hundred(second_part, True)}"
    return integer_to_german_words(year)


_YEAR_HINT_LEFTOVER = js(r"(?<![\w.])(\d{4})\s*\(\s*([\p{L}]+)(?:\s*\/\s*([\p{L}]+))?\s*\)", "gu")


def resolve_german_year_hint_leftovers(text: str, applied_rules: list[str]) -> str:
    def callback(match, number_token, first_variant, second_variant, *_rest):
        value = int(number_token)
        readings = {integer_to_german_words(value).lower(), year_to_german_words(value).lower()}
        if second_variant is not None:
            if first_variant.lower() in readings and second_variant.lower() in readings:
                return year_to_german_words(value)
            return match
        return first_variant if first_variant.lower() in readings else match

    return replace_with_tracking(text, _YEAR_HINT_LEFTOVER, callback, "year_hint_leftover_de", applied_rules)


_IRREGULAR_ORDINALS = {1: "erste", 3: "dritte", 7: "siebte", 8: "achte"}


def base_german_ordinal(day: int) -> str:
    irregular_value = _IRREGULAR_ORDINALS.get(day)
    if irregular_value:
        return irregular_value
    remainder = day % 100
    stem = integer_to_german_words(day)
    if day > 20 and 1 <= remainder <= 19:
        small = _IRREGULAR_ORDINALS.get(remainder)
        if small:
            small_stem = integer_to_german_words(remainder)
            return f"{stem[:len(stem) - len(small_stem)]}{small}"
        return f"{stem}te"
    if stem.endswith("eins"):
        stem = f"{stem[:-4]}ein"
    return f"{stem}te" if day < 20 else f"{stem}ste"


GERMAN_MONTH_SOURCE = (
    "(?:Januar|Februar|M(?:aerz|ärz)|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember"
    "|Jan|Feb|Mär|Mrz|Apr|Jun|Jul|Aug|Sept|Sep|Okt|Nov|Dez)"
)

_GERMAN_MONTH_ABBREVIATIONS = {
    "jan": 1, "feb": 2, "mae": 3, "mrz": 3, "apr": 4, "mai": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "okt": 10, "nov": 11, "dez": 12,
}
_TRAILING_DOT = js(r"\.$", "")
_UMLAUT_A = js(r"ä", "g")


def normalize_german_month_token(token: str) -> str:
    return js_replace(_UMLAUT_A, js_replace(_TRAILING_DOT, js_trim(token or ""), "").lower(), "ae")


def german_month_index(token: str) -> Optional[int]:
    key = normalize_german_month_token(token)
    full = GERMAN_MONTH_NAME_TO_INDEX.get(key)
    if full:
        return full
    return _GERMAN_MONTH_ABBREVIATIONS.get(key[:3])


def german_ordinal_day(day: int, inflection: str) -> str:
    """``inflection``: 'nominative' | 'oblique' | 'weak'."""
    base = base_german_ordinal(day)
    if inflection == "weak":
        return base
    return f"{base}n" if inflection == "oblique" else f"{base}r"


_WHITESPACE_SPLIT = js(r"\s+", "")
_DER_WORD = js(r"^der$", "i")


def german_day_inflection_after(prefix) -> str:
    """Bug 682: the first word in front of the day decides the ending of its ordinal."""
    raw_prefix = js_trim(prefix or "")
    if not raw_prefix:
        return "nominative"
    return "weak" if _DER_WORD.test(_WHITESPACE_SPLIT.split(raw_prefix)[0]) else "oblique"


# The prefixes of the date rules, with an optional apposition after the preposition (Bugs 682, 842).
GERMAN_PREFIXED_DATE_WORD_SOURCE = "(?:am|dem|den|vom|zum|von|bis|ab|seit|nach|vor|für)"
GERMAN_CONTRACTING_BIS_SOURCE = (
    r"(?<!(?<!" + GERMAN_PREFIXED_DATE_WORD_SOURCE + r"\s+)(?:(?<!\d)\d{1,2}(?:\.\d{1,2}\.|\/\d{1,2}\/)(?:\d{4}|\d{2})?"
    r"|(?<![\d-])\d{4}-\d{2}-\d{2})\s*|(?:ter|ster)\s+(?:Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober"
    r"|November|Dezember)(?:\s+[a-zäöüß]+)?\s+)bis"
)
GERMAN_DATE_PREFIX_SOURCE = (
    r"((?:am|dem|den|vom|zum|von|ab|seit|nach|vor|für|" + GERMAN_CONTRACTING_BIS_SOURCE + r")\s+(?:[A-ZÄÖÜ][a-zäöüß]{2,},\s*)?|der\s+)"
)
GERMAN_DATELESS_DATE_PREFIX_SOURCE = GERMAN_DATE_PREFIX_SOURCE

# The article each contracting preposition gives a date (Bug 842).
GERMAN_DATE_CONTRACTIONS = {
    "von": "vom", "bis": "bis zum", "ab": "ab dem", "seit": "seit dem", "nach": "nach dem", "vor": "vor dem", "für": "für den",
}
_STARTS_UPPER = js(r"^\p{Lu}", "u")


def speak_german_date_prefix(raw_prefix: str) -> str:
    contracted = GERMAN_DATE_CONTRACTIONS.get(raw_prefix.lower())
    if not contracted:
        return raw_prefix
    return f"{contracted[0].upper()}{contracted[1:]}" if _STARTS_UPPER.test(raw_prefix) else contracted


# Bug 682: the rules capture the dot behind a month and the follower, and keep the dot where it
# ends the sentence.
GERMAN_FOLLOWER_SOURCE = r"(?=([ \t]*(?:\n[ \t]*)?\S?))"
_BLANKS_ONLY = js(r"^[ \t]*$", "")
_LOWER_LINE_START = js(r"\n[ \t]*\p{Ll}", "u")
_BLANKS_THEN_UPPER = js(r"^[ \t]+\p{Lu}", "u")


def german_dot_ends_sentence(dot, follower) -> bool:
    if not dot:
        return False
    rest = follower or ""
    if _BLANKS_ONLY.test(rest):
        return True
    if "\n" in rest:
        return not _LOWER_LINE_START.test(rest)
    return _BLANKS_THEN_UPPER.test(rest)


def german_dot_ends_line(dot, follower) -> bool:
    """Only the line-end case of german_dot_ends_sentence (Bug 825)."""
    rest = follower or ""
    return bool(dot) and "\n" in rest and not _LOWER_LINE_START.test(rest)


def german_list_ordinal(value: int) -> str:
    base = base_german_ordinal(value)
    return f"{base[:-1]}ens" if base.endswith("e") else f"{base}ens"


def capitalize_german_token(value: str) -> str:
    if not value:
        return value
    return f"{value[:1].upper()}{value[1:]}"


_YEAR_TOKEN = js(r"^\d{2,4}$", "")


def parse_spoken_german_year_token(token) -> Optional[int]:
    if not _YEAR_TOKEN.test(token if token is not None else ""):
        return None
    year = int(token)
    if len(token) == 2:
        return 1900 + year if year >= 50 else 2000 + year
    return year


_ALPHA_DIGIT = js(r"([ÄÖÜäöüA-Za-z])(\d)", "g")
_DIGIT_ALPHA = js(r"(\d)([ÄÖÜäöüA-Za-z])", "g")
_CAMEL = js(r"([a-zäöüß])([A-ZÄÖÜ])", "g")


def split_german_alpha_numeric_boundaries(value: str) -> str:
    return js_replace(_DIGIT_ALPHA, js_replace(_ALPHA_DIGIT, value or "", "$1 $2"), "$1 $2")


def split_german_camel_case_boundaries(value: str) -> str:
    return js_replace(_CAMEL, value or "", "$1 $2")


_SPACE_BEFORE_EXTENSION = js(r"\s+\.(?=[A-Za-z0-9]{2,8}\b)", "g")
_EXTENSION = js(r"^(.*?)(?:\.([A-Za-z0-9]{2,8}))$", "")
_PATH_SEPARATORS = js(r"[\\/]+", "g")
_BRACKETS = js(r"[()[\]{}]+", "g")
_UNDERSCORES_DASHES = js(r"[_-]+", "g")


def speak_german_filename_phrase(value: str) -> str:
    result = js_trim(value or "")
    if not result:
        return result
    result = js_replace(_SPACE_BEFORE_EXTENSION, result, ".")
    extension_match = _EXTENSION.first(result)
    extension = ""
    if extension_match:
        result = extension_match.group(1)
        extension = extension_match.group(2).upper()
    result = js_replace(_PATH_SEPARATORS, result, " ")
    result = js_replace(_BRACKETS, result, " ")
    result = js_replace(_UNDERSCORES_DASHES, result, " ")
    result = split_german_camel_case_boundaries(result)
    result = split_german_alpha_numeric_boundaries(result)
    result = finalize_optimized_text(result)
    return f"{result} {' '.join(extension)}" if extension else result


def speak_german_temperature(token: str, unit: str) -> str:
    spoken_value = speak_german_attributive_numeric_token(token)
    if unit == "F":
        return f"{spoken_value} Grad Fahrenheit"
    if unit == "K":
        return f"{spoken_value} Kelvin"
    return f"{spoken_value} Grad Celsius"


def spell_german_digits(token: str) -> str:
    return " ".join(GERMAN_DIGITS[int(ch)] for ch in token if "0" <= ch <= "9")


# No grouping starts with a zero, and no group has four digits (Bug 837).
_GROUPED_THOUSANDS = js(r"^[1-9]\d{0,2}(?:\.\d{3})+$", "")
_ALL_DIGITS = js(r"^\d+$", "")
_LEADING_ZEROS = js(r"^0+(?=\d)", "")


def parse_german_number_token(token):
    raw = js_trim(token or "")
    if not raw:
        return None
    negative = raw.startswith("-")
    normalized = raw[1:] if negative else raw
    if "," in normalized:
        parts = normalized.split(",")
        if len(parts) != 2:
            return None
        return negative, parts[0].replace(".", ""), parts[1]
    if "." in normalized:
        if _GROUPED_THOUSANDS.test(normalized):
            return negative, normalized.replace(".", ""), None
        parts = normalized.split(".")
        if len(parts) >= 2 and (
            normalized.startswith("0.") or len(parts[-1]) != 3 or (len(parts) == 2 and len(parts[0]) > 3)
        ):
            return negative, parts[0], "".join(parts[1:])
        return negative, normalized.replace(".", ""), None
    return negative, normalized, None


def _strip_leading_zeros(value: str) -> str:
    return js_replace(_LEADING_ZEROS, value, "") or "0"


def _valid_number(parsed) -> bool:
    return bool(parsed) and _ALL_DIGITS.test(parsed[1] or "0") and (parsed[2] is None or _ALL_DIGITS.test(parsed[2]))


def speak_german_numeric_token(token: str) -> str:
    parsed = parse_german_number_token(token)
    if not _valid_number(parsed):
        return token
    negative, integer_part, fraction_part = parsed
    integer_part = _strip_leading_zeros(integer_part)
    prefix = "minus " if negative else ""
    integer_words = integer_to_german_words(int(integer_part))
    if fraction_part:
        return js_trim(f"{prefix}{integer_words} komma {spell_german_digits(fraction_part)}")
    return js_trim(f"{prefix}{integer_words}")


def speak_german_attributive_numeric_token(token: str) -> str:
    parsed = parse_german_number_token(token)
    is_exactly_one = bool(parsed) and not parsed[2] and _strip_leading_zeros(parsed[1]) == "1"
    if not is_exactly_one:
        return speak_german_numeric_token(token)
    return "minus ein" if parsed[0] else "ein"


_NONZERO_DIGIT = js(r"[1-9]", "")


def speak_german_currency(amount_token: str, currency_token: str) -> str:
    parsed = parse_german_number_token(amount_token)
    currency = GERMAN_CURRENCY_MAP.get(js_trim(currency_token or "").upper())
    if not parsed or not currency or not _ALL_DIGITS.test(parsed[1] or "0"):
        return js_trim(f"{amount_token} {currency_token}")
    negative, integer_part, fraction_part = parsed
    integer_part = _strip_leading_zeros(integer_part)
    integer_value = int(integer_part)
    integer_words = "ein" if integer_value == 1 else integer_to_german_words(integer_value)
    unit_label = currency["unitSingular"] if integer_value == 1 else currency["unitPlural"]
    if fraction_part and len(fraction_part) > 2:
        return f"{speak_german_numeric_token(amount_token)} {currency['unitPlural']}"
    # Bug 843: a currency without a subunit in use reads its decimals as a decimal ("108,45 JPY").
    if fraction_part and not currency["subPlural"] and _NONZERO_DIGIT.test(fraction_part):
        return f"{speak_german_numeric_token(amount_token)} {currency['unitPlural']}"
    normalized_fraction = fraction_part.ljust(2, "0") if fraction_part else ""
    fraction_value = js_number(normalized_fraction) if normalized_fraction else 0
    prefix = "minus " if negative else ""
    if fraction_value > 0:
        sub_label = currency["subSingular"] if fraction_value == 1 else currency["subPlural"]
        sub_words = "ein" if fraction_value == 1 else integer_to_german_words(int(fraction_value))
        # Bug 843: an amount below one unit is its subunits alone.
        if integer_value == 0:
            return f"{prefix}{sub_words} {sub_label}"
        return f"{prefix}{integer_words} {unit_label} und {sub_words} {sub_label}"
    return js_trim(f"{prefix}{integer_words} {unit_label}")


def speak_german_percent(token: str) -> str:
    return f"{speak_german_attributive_numeric_token(token)} Prozent"


GERMAN_SCALE_WORDS = {
    "tsd": ("Tausend", "Tausend"),
    "mio": ("Million", "Millionen"),
    "mrd": ("Milliarde", "Milliarden"),
    # Bug 843: only with its dot, since "Bio" before a noun is organic food.
    "bio": ("Billion", "Billionen"),
}


def _unit_word(singular: str, plural: Optional[str] = None, feminine: bool = False):
    return (singular, singular if plural is None else plural, feminine)


GERMAN_UNIT_WORDS = {
    "nm": _unit_word("Nanometer"),
    "µm": _unit_word("Mikrometer"),
    "mm": _unit_word("Millimeter"),
    "cm": _unit_word("Zentimeter"),
    "m": _unit_word("Meter"),
    "km": _unit_word("Kilometer"),
    "m²": _unit_word("Quadratmeter"),
    "km²": _unit_word("Quadratkilometer"),
    "m³": _unit_word("Kubikmeter"),
    "ha": _unit_word("Hektar"),
    "µg": _unit_word("Mikrogramm"),
    "mg": _unit_word("Milligramm"),
    "g": _unit_word("Gramm"),
    "kg": _unit_word("Kilogramm"),
    "ml": _unit_word("Milliliter"),
    "cl": _unit_word("Zentiliter"),
    "dl": _unit_word("Deziliter"),
    "l": _unit_word("Liter"),
    "ms": _unit_word("Millisekunde", "Millisekunden", True),
    "s": _unit_word("Sekunde", "Sekunden", True),
    "Sek": _unit_word("Sekunde", "Sekunden", True),
    "min": _unit_word("Minute", "Minuten", True),
    "Min": _unit_word("Minute", "Minuten", True),
    "h": _unit_word("Stunde", "Stunden", True),
    "Std": _unit_word("Stunde", "Stunden", True),
    "W": _unit_word("Watt"),
    "kW": _unit_word("Kilowatt"),
    "MW": _unit_word("Megawatt"),
    "Wh": _unit_word("Wattstunde", "Wattstunden", True),
    "kWh": _unit_word("Kilowattstunde", "Kilowattstunden", True),
    "V": _unit_word("Volt"),
    "mAh": _unit_word("Milliamperestunden"),
    # Bug 843: units the fuzz test found raw; "Nm" is the newton-metre.
    "L": _unit_word("Liter"),
    "ft": _unit_word("Fuß"),
    "lb": _unit_word("Pfund"),
    "lbs": _unit_word("Pfund"),
    "oz": _unit_word("Unze", "Unzen", True),
    "Nm": _unit_word("Newtonmeter"),
    "mmol": _unit_word("Millimol"),
    "µs": _unit_word("Mikrosekunde", "Mikrosekunden", True),
    "kbps": _unit_word("Kilobit pro Sekunde"),
    "Mbps": _unit_word("Megabit pro Sekunde"),
    "Gbps": _unit_word("Gigabit pro Sekunde"),
    "rpm": _unit_word("Umdrehung pro Minute", "Umdrehungen pro Minute", True),
    "cm²": _unit_word("Quadratzentimeter"),
    "mm²": _unit_word("Quadratmillimeter"),
    "dm²": _unit_word("Quadratdezimeter"),
    "cm³": _unit_word("Kubikzentimeter"),
    "mm³": _unit_word("Kubikmillimeter"),
    "km³": _unit_word("Kubikkilometer"),
    "dm³": _unit_word("Kubikdezimeter"),
    "dB(A)": _unit_word("Dezibel A"),
    "dB": _unit_word("Dezibel"),
    "Hz": _unit_word("Hertz"),
    "kHz": _unit_word("Kilohertz"),
    "MHz": _unit_word("Megahertz"),
    "GHz": _unit_word("Gigahertz"),
    "hPa": _unit_word("Hektopascal"),
    "mmHg": _unit_word("Millimeter Quecksilbersäule"),
    "mbar": _unit_word("Millibar"),
    "bar": _unit_word("Bar"),
    "kcal": _unit_word("Kilokalorie", "Kilokalorien", True),
    "kJ": _unit_word("Kilojoule"),
    "kB": _unit_word("Kilobyte"),
    "KB": _unit_word("Kilobyte"),
    "MB": _unit_word("Megabyte"),
    "GB": _unit_word("Gigabyte"),
    "TB": _unit_word("Terabyte"),
    "KiB": _unit_word("Kibibyte"),
    "MiB": _unit_word("Mebibyte"),
    "GiB": _unit_word("Gibibyte"),
    "TiB": _unit_word("Tebibyte"),
    "kbit": _unit_word("Kilobit"),
    "Mbit": _unit_word("Megabit"),
    "Gbit": _unit_word("Gigabit"),
    "fps": _unit_word("F P S"),
    "bpm": _unit_word("Schlag pro Minute", "Schläge pro Minute"),
    "mph": _unit_word("Meile pro Stunde", "Meilen pro Stunde", True),
    "kn": _unit_word("Knoten"),
    "Bft": _unit_word("Beaufort"),
    "ct": _unit_word("Cent"),
    "px": _unit_word("Pixel"),
}

GERMAN_UNIT_KEY_SOURCE = "|".join(
    escape_for_regexp(key) for key in sorted(GERMAN_UNIT_WORDS.keys(), key=lambda key: -len(key))
)
_UPPER_ASCII = js(r"[A-Z]", "")
_SHORT_NUMBER = js(r"^\d{1,3}$", "")


def is_german_unit_reading_plausible(number_token: str, spacing: str, unit_key: str) -> bool:
    if len(unit_key) > 1 or _UPPER_ASCII.test(unit_key):
        return True
    # A hyphen ("64-MB-Grenze") counts as glued, not as a space.
    if _BLANK_SPACING.test(spacing):
        return True
    return spacing == "" and unit_key in ("h", "s") and _SHORT_NUMBER.test(number_token)


_BLANK_SPACING = js(r"^[ \t]+$", "")


def speak_german_amount_scale(number_token, scale_token, prefix_currency=None, suffix_currency=None, dative=False):
    parsed = parse_german_number_token(number_token)
    if not _valid_number(parsed):
        return None
    scale = GERMAN_SCALE_WORDS.get((scale_token or "").lower())
    if not scale:
        return None
    negative, integer_part, fraction_part = parsed
    integer_part = _strip_leading_zeros(integer_part)
    is_exactly_one = not negative and not fraction_part and integer_part == "1"
    currency_token = js_trim(prefix_currency or suffix_currency or "")
    currency = GERMAN_CURRENCY_MAP.get(currency_token.upper()) if currency_token else None
    if (scale_token or "").lower() == "tsd" and not fraction_part:
        folded = integer_to_german_words(int(integer_part) * 1000)
        parts = [f"minus {folded}" if negative else folded, currency["unitPlural"] if currency else ""]
        return " ".join(part for part in parts if part)
    # Bug 843: exactly one after a dative preposition ("bei einer Million").
    number_words = ("einer" if dative else "eine") if is_exactly_one else speak_german_numeric_token(number_token)
    parts = [number_words, scale[0] if is_exactly_one else scale[1], currency["unitPlural"] if currency else ""]
    return " ".join(part for part in parts if part)


_EIN_AT_END = js(r"(^|\s)ein$", "")
_EIN_SUFFIX = js(r"ein$", "")


def speak_german_unit(number_token: str, spacing: str, unit_token: str, dative: bool = False) -> Optional[str]:
    key = js_replace(_TRAILING_DOT, unit_token or "", "")
    unit = GERMAN_UNIT_WORDS.get(key)
    if not unit or not is_german_unit_reading_plausible(number_token, spacing, key):
        return None
    singular, plural, feminine = unit
    words = speak_german_attributive_numeric_token(number_token)
    if _EIN_AT_END.test(words):
        # Bug 843: exactly one after a dative preposition ("mit einem Meter").
        if dative and words == "ein":
            return f"{'einer' if feminine else 'einem'} {singular}"
        return f"{js_replace(_EIN_SUFFIX, words, 'eine') if feminine else words} {singular}"
    return f"{words} {plural}"


def speak_german_version(prefix, parts) -> str:
    spoken = " Punkt ".join(speak_german_version_part(part) for part in parts if isinstance(part, str) and part != "")
    return f"Version {spoken}" if prefix else spoken


_VERSION_WILDCARD_PART = js(r"^[x*]$", "i")


def speak_german_version_part(part: str) -> str:
    """Bug 840: a part with a leading zero keeps its digits, a wildcard part is "x"."""
    if _VERSION_WILDCARD_PART.test(part):
        return "x"
    if len(part) > 1 and part.startswith("0"):
        return spell_german_digits(part)
    return integer_to_german_words(int(part))


_GERMAN_SCALE_ABBREVIATION = js(r"^(?:Tsd|Mio|Mrd)\.?$", "")
_TRAILING_CLOSERS = js(r"[,;:!?)\]\"”]+$", "")


def is_german_measure_after_version(follower) -> bool:
    """Bug 840: a unit symbol, a scale abbreviation or a percent sign after "Version 2.5" keeps the measure."""
    if follower is None:
        return False
    if follower.startswith("%"):
        return True
    token = js_replace(_TRAILING_CLOSERS, follower, "")
    return js_replace(_TRAILING_DOT, token, "") in GERMAN_UNIT_WORDS or _GERMAN_SCALE_ABBREVIATION.test(token)


def looks_like_german_dotted_date(first: str, second: str, third: str, version_marked: bool = False) -> bool:
    day = int(first)
    month = int(second)
    return 1 <= day <= 31 and 1 <= month <= 12 and (len(third) == 4 or (len(third) == 2 and not version_marked))


def speak_german_ipv4(octets, port=None) -> Optional[str]:
    if any(int(octet) > 255 for octet in octets):
        return None
    # Bug 840: a round number in the German grouping shares the shape of an address.
    if not port and len(octets[1]) == 3 and octets[2] == "000" and octets[3] == "000":
        return None
    spoken = " Punkt ".join(spell_german_digits(octet) for octet in octets)
    return f"{spoken}, Port {spell_german_digits(port)}" if port else spoken


_UND = js(r"und", "i")
_ODER = js(r"oder", "i")


def speak_german_day_range(prefix, day1_token, month1_token, joiner_token, day2_token,
                           month_numeric, year_numeric, month_name, year_named, second_prefix=None) -> Optional[str]:
    day1 = int(day1_token)
    day2 = int(day2_token)
    month2 = int(month_numeric) if month_numeric else german_month_index(month_name or "")
    month1 = int(month1_token) if month1_token else month2
    if not month1 or not month2 or month1 < 1 or month1 > 12 or month2 < 1 or month2 > 12:
        return None
    if day1 < 1 or day1 > 31 or day2 < 1 or day2 > 31:
        return None
    raw_prefix = js_trim(prefix or "")
    inflection = "oblique" if raw_prefix else "nominative"
    joiner = "und" if _UND.test(joiner_token) else "oder" if _ODER.test(joiner_token) else "bis"
    if year_numeric:
        year = parse_spoken_german_year_token(year_numeric)
    elif year_named:
        year = int(year_named)
    else:
        year = None
    tail = f"{GERMAN_MONTHS[month2 - 1]}{' ' + year_to_german_words(year) if year else ''}"
    if month1 == month2:
        first = german_ordinal_day(day1, inflection)
    else:
        first = f"{german_ordinal_day(day1, inflection)} {GERMAN_MONTHS[month1 - 1]}"
    if second_prefix:
        second = f"{js_trim(second_prefix).lower()} {german_ordinal_day(day2, 'oblique')}"
    else:
        second = german_ordinal_day(day2, inflection)
    days = f"{first} {joiner} {second} {tail}"
    return f"{speak_german_date_prefix(raw_prefix)} {days}" if raw_prefix else days


GERMAN_SENTENCE_OPENER_SOURCE = (
    "(?:Ich|Du|Er|Sie|Es|Wir|Ihr|Das|Der|Die|Dann|Danach|Dabei|Dazu|Damit|Deshalb|Also|Aber|Und|Oder|Bitte|Danke|"
    "Jetzt|Heute|Hier|Dort|So|Wenn|Falls|Was|Wer|Wie|Wo|Wann|Warum|Kein|Keine|Mehr|Noch|Schon|Alles|Nichts|Ja|Nein|"
    "Okay|OK|Los|Gut|Klar|Passt|Soll|Kann|Willst|Möchtest|Brauchst|Hast|Bist|Ist|Sind|Habe|Gibt|Lass|Viel|Alle|"
    "Diese|Dieser|Dieses|Am|Im|Um|Auf|In|Mit|Von|Zu|Nach|Vor|Bei|Für|Über|Unter|Seit|Bis|Ab|Aus|Als|Da|Doch|Nur|"
    "Außerdem|Zudem|Übrigens|Allerdings|Trotzdem|Eigentlich|Vielleicht|Sonst|Erst|Später|Gleich|Sobald|Solange|Sag|"
    "Melde|Schau|Denk|Vergiss|Mach|Geh|Komm|Ruf|Schreib|Sollte|Könnte|Würde|Wäre|Weitere|Weiter|Nimm|Gib|Zeig|"
    "Bring|Hol|Öffne|Schick|Prüf|Warte|Erinnere|Stell|Leg|Setz|Frag|Antworte|Dein|Deine|Mein|Meine|Sein|Seine|Unser|"
    "Unsere|Euer|Eure|Man|Jemand|Niemand|Etwas|Denn|Weil|Obwohl|Sowohl|Weder|Entweder|Zum|Zur|Vom|Beim|Ins|Ans|"
    "Ein|Eine|Einen|Einem|Einer|Eines)"
)
# An interjection opens a sentence only before a mark, a lower-case word or the end (Bug 837).
GERMAN_INTERJECTION_OPENER_SOURCE = "(?:Super)"
GERMAN_ORDINAL_PREPOSITION_SOURCE = (
    "(?:[Ii]n|[Aa]n|[Aa]uf|[Bb]ei|[Mm]it|[Nn]ach|[Ss]eit|[Vv]on|[Zz]u|[Aa]us|[Vv]or|[Uu]nter|[Üü]ber|[Zz]wischen|"
    "[Gg]egenüber|[Ee]ntgegen|[Ll]aut|[Ww]egen|[Tt]rotz|[Ww]ährend|[Dd]ank|[Gg]emäß)"
)
GERMAN_ORDINAL_DETERMINER_SOURCE = (
    "(?:[Aa]m|[Bb]eim|[Ii]m|[Dd]em|[Dd]en|[Vv]om|[Zz]um|[Zz]ur|[Bb]is|[Dd]er|[Dd]es|[Dd]ie|[Dd]as|"
    "[Jj]ede[nmrs]?|[Ii]hre?[nmrs]?|[Ss]eine?[nmrs]?|[Mm]eine?[nmrs]?|[Dd]eine?[nmrs]?|[Uu]nsere?[nmrs]?)"
)
GERMAN_ORDINAL_PREFIX_SOURCE = f"(?:{GERMAN_ORDINAL_PREPOSITION_SOURCE}[ \\t]+)?{GERMAN_ORDINAL_DETERMINER_SOURCE}"
GERMAN_CONTRACTED_DETERMINER_SOURCE = "(?:[Aa]m|[Bb]eim|[Ii]m|[Vv]om|[Zz]um|[Zz]ur)"
# The words after which a number names something rather than counting it (Bug 837).
GERMAN_NUMBER_LABEL_SOURCE = (
    "(?:Abschnitt|Kapitel|Kap\\.|Absatz|Abs\\.|Artikel|Art\\.|Paragraf|Paragraph|Ziffer|Ziff\\.|Nummer|Nr\\.|Hausnummer|"
    "Seite|S\\.|Haus|Zimmer|Raum|Halle|Gleis|Bahnsteig|Terminal|Tor|Gate|Platz|Sitzplatz|Rang|Sitz|Reihe|Block|Etage|"
    "Stock|Woche|Kalenderwoche|KW|Quartal|Heft|Ausgabe|Band|Jahrgang|Folge|Staffel|Episode|Teil|Schritt|Anhang|Anlage|"
    "Tabelle|Abbildung|Abb\\.|Level|Stufe|Klasse|Az\\.|Aktenzeichen)"
)
GERMAN_VERSUS_SOURCE = "(?:vs\\.|versus|gegen)"
GERMAN_CENTURY_SOURCE = "(?:Jh\\.|Jhd\\.|Jahrhundert|Jahrhunderts|Jahrtausend|Jahrtausends)"
GERMAN_OPERATOR_WORD_SOURCE = "(?:plus|minus|mal|durch|gleich|modulo)"
GERMAN_BARE_NUMBER_SOURCE = r"(?<!\b" + GERMAN_ORDINAL_DETERMINER_SOURCE + r"[ \t]+)\b\d+[ \t]+"
GERMAN_OPERATION_SOURCE = (
    r"(?:(?<!\b" + GERMAN_ORDINAL_DETERMINER_SOURCE + r"[ \t]+\d{0,3})\d|[)\]]|(?<![\p{L}\p{Nd}_])\p{L})[ \t]+"
    + GERMAN_OPERATOR_WORD_SOURCE
)
GERMAN_SENTENCE_FINAL_PREFIX_SOURCE = (
    f"(?:{GERMAN_ORDINAL_PREPOSITION_SOURCE}[ \\t]+{GERMAN_ORDINAL_DETERMINER_SOURCE}|{GERMAN_CONTRACTED_DETERMINER_SOURCE}(?![\\p{{L}}\\p{{N}}_])|"
    f"(?<!{GERMAN_BARE_NUMBER_SOURCE}){GERMAN_ORDINAL_DETERMINER_SOURCE})"
)
# Groups: 1 the prefix, 2 an adjective before a century, 3 (empty) after a label/versus/range/operation,
# 4 a minus, 5 the number, 6 and 7 a chained second ordinal, 8 the follower (up to two words).
GERMAN_SENTENCE_FINAL_INTEGER_PATTERN = js(
    r"(?:(?<=\S[ \t]+)|(?<![^\n])(?=" + GERMAN_ORDINAL_PREFIX_SOURCE + r"[ \t]+(?:\p{Ll}+[ \t]+)?\d))(?:("
    + GERMAN_SENTENCE_FINAL_PREFIX_SOURCE + r"[ \t]+)(\p{Ll}+[ \t]+(?=\d{1,3}\.[ \t]+" + GERMAN_CENTURY_SOURCE
    + r"(?!\p{L})))?)?(?=-?\d)(?:(?<=(?:\b(?:" + GERMAN_NUMBER_LABEL_SOURCE + r"|" + GERMAN_VERSUS_SOURCE + r")|"
    + GERMAN_BARE_NUMBER_SOURCE + r"bis|" + GERMAN_OPERATION_SOURCE + r")[ \t]+)()|)(-)?(\d{1,3})\.(?:[ \t]*(und|bis)[ \t]*(\d{1,3})\.)?"
    r"(?=([ \t]*(?:$|\n|[\"”“)\]])|[ \t]+\S+(?:[ \t]+\S+)?))",
    "gu",
)
_SENTENCE_OPENER_FOLLOWER = js(
    r"^[ \t]+(?:" + GERMAN_SENTENCE_OPENER_SOURCE + r"\b|" + GERMAN_INTERJECTION_OPENER_SOURCE
    + r"(?:[^\p{L}\s]|[ \t]*$|[ \t]+\p{Ll}))",
    "u",
)
_TERMINAL_FOLLOWER = js(r"^[ \t]*(?:$|\n|[\"”“)\]])", "")
_CAPITALISED_FOLLOWER = js(r"^[ \t]+\p{Lu}", "u")
_NOMINATIVE_ARTICLE = js(r"(?:^|\s)(?:der|die|das)$", "i")
_PREPOSITION_PREFIX = js(f"^{GERMAN_ORDINAL_PREPOSITION_SOURCE}\\s", "")


def is_german_sentence_final_follower(follower: str) -> bool:
    return _TERMINAL_FOLLOWER.test(follower) or _SENTENCE_OPENER_FOLLOWER.test(follower)


def is_german_sentence_final_number(follower: str, after_label) -> bool:
    """A number after a label, a versus word, a range connector or an operator ends before a capital (Bug 837)."""
    return is_german_sentence_final_follower(follower) or (after_label is not None and _CAPITALISED_FOLLOWER.test(follower))


def is_deferred_german_ordinal(prefix, after_label, sign, follower) -> bool:
    return bool(js_trim(prefix or "")) and not sign and not is_german_sentence_final_number(follower, after_label)


# Bug 682: after a possessive the ordinal's ending follows the possessive's own ending.
_GERMAN_POSSESSIVE = js(r"^(?:mein|dein|sein|ihr|unser|euer)(e|em|en|er|es)?$", "i")


def german_ordinal_after_possessive(determiner: str):
    possessive = _GERMAN_POSSESSIVE.search(determiner)
    if not possessive:
        return None
    ending = (possessive.group(1) or "").lower()
    if not ending:
        return lambda day: f"Nummer {integer_to_german_words(day)}"
    if ending == "e":
        return base_german_ordinal
    return lambda day: german_ordinal_day(day, "oblique")


_NUMMER_PREFIX = js(r"^Nummer ", "")


def speak_german_sentence_final_integer(prefix, adjective, after_label, sign, number_token, chain_joiner, second_token,
                                        follower, for_hint=False) -> Optional[str]:
    value = int(number_token)
    terminal = is_german_sentence_final_number(follower, after_label)
    capitalised_follower = _CAPITALISED_FOLLOWER.test(follower)
    raw_prefix = js_trim(prefix or "")
    if sign:
        if second_token is not None or not terminal:
            return None
        cardinal = integer_to_german_words(-value)
        return f"{raw_prefix} {cardinal}." if raw_prefix else f"{cardinal}."
    if raw_prefix:
        nominative = _NOMINATIVE_ARTICLE.test(raw_prefix) and not _PREPOSITION_PREFIX.test(raw_prefix)
        after_possessive = german_ordinal_after_possessive((_WHITESPACE_SPLIT.split(raw_prefix) or [""])[-1] or "")

        def inflect(day: int) -> str:
            if after_possessive:
                return after_possessive(day)
            return base_german_ordinal(day) if nominative else german_ordinal_day(day, "oblique")

        head = f"{raw_prefix} {js_trim(adjective)}" if adjective else raw_prefix
        gendered_alternative = for_hint and after_possessive and _NUMMER_PREFIX.test(after_possessive(value))
        if second_token is not None:
            second = int(second_token)
            if value < 1 or second < 1 or value > 999 or second > 999:
                return None
            spoken = f"{head} {inflect(value)} {_js_str(chain_joiner)} {inflect(second)}{'.' if terminal else ''}"
            if not gendered_alternative:
                return spoken
            return (
                f"{spoken}/{head} {base_german_ordinal(value)} {_js_str(chain_joiner)} {base_german_ordinal(second)}"
                f"{'.' if terminal else ''}"
            )
        if value >= 1 and (value <= 31 or (not terminal and capitalised_follower)):
            spoken = f"{head} {inflect(value)}{'.' if terminal else ''}"
            if gendered_alternative:
                return f"{spoken}/{head} {base_german_ordinal(value)}{'.' if terminal else ''}"
            return spoken
        if terminal:
            return f"{head} {integer_to_german_words(value)}."
        return None
    if second_token is not None or not terminal:
        return None
    return f"{integer_to_german_words(value)}."


def spell_german_hex(token: str) -> str:
    return " ".join(GERMAN_DIGITS[int(ch)] if "0" <= ch <= "9" else ch.upper() for ch in token or "")


def speak_german_phone_number(match: str) -> Optional[str]:
    parsed = parse_phone_number_match(match)
    if not parsed:
        return None
    plus, digits = parsed
    spoken_digits = spell_german_digits(digits)
    return f"plus {spoken_digits}" if plus else spoken_digits


def speak_german_network_port(label: str, number_token: str) -> Optional[str]:
    port = int(number_token)
    if port < 0 or port > 65535:
        return None
    return f"{js_trim(label or '')} {spell_german_digits(number_token)}"


# "fünf" was stored as the mojibake "fÃ¼nf" before Bug 811; fixed in Genesis3 since.
GERMAN_DIGIT_WORD_SOURCE = "(?:null|eins|zwei|drei|vier|fünf|sechs|sieben|acht|neun)"
GERMAN_URL_PORT_PATTERN = js(
    f"\\b((?:https?:\\/\\/|www\\.)[^\\s/:?#]+):(\\d{{1,5}})\\b(?:\\s+\\(?((?:{GERMAN_DIGIT_WORD_SOURCE})(?:\\s+{GERMAN_DIGIT_WORD_SOURCE}){{0,4}})\\)?)?",
    "gi",
)
_WS_RUN = js(r"\s+", "g")


def normalize_german_url_ports(text: str, applied_rules: list[str]) -> str:
    def callback(match, url_prefix, number_token, existing_spoken, *_rest):
        port = int(number_token)
        if port < 0 or port > 65535:
            return match
        spoken = spell_german_digits(number_token)
        existing = js_replace(_WS_RUN, js_trim(existing_spoken or ""), " ")
        suffix = f" {existing}" if existing and existing.lower() != spoken.lower() else ""
        return f"{url_prefix}: {spoken}{suffix}"

    return replace_with_tracking(text, GERMAN_URL_PORT_PATTERN, callback, "url_port_de", applied_rules)


# The vowels of the German alphabet: a label without one cannot be said as a word.
_GERMAN_ADDRESS_VOWEL = js(r"[aeiouyäöü]", "i")
_GERMAN_LETTERS_ONLY = js(r"^[a-zäöüß]+$", "i")
_LETTERS_ONLY = js(r"^\p{L}+$", "u")
_HAS_DIGIT = js(r"\d", "")
_DIGIT_OR_OTHER_RUNS = js(r"\d+|[^\d]+", "g")
_UNDERSCORE_OR_PLUS_SPLIT = js(r"([_+])", "")
_UNDERSCORE_OR_PLUS = js(r"[_+]", "")


def speak_german_address_label(label: str, is_ending: bool) -> str:
    """Bug 969: how one label of a domain or an e-mail address is written for the engine."""

    def speak_letters(letters: str, spell_pair: bool) -> str:
        if not _GERMAN_LETTERS_ONLY.test(letters):
            return letters
        spelled = (spell_pair and len(letters) == 2) or (len(letters) >= 2 and not _GERMAN_ADDRESS_VOWEL.test(letters))
        return "-".join(letters.upper()) if spelled else letters

    def speak_part(part: str, spell_pair: bool) -> str:
        if not _HAS_DIGIT.test(part):
            return speak_letters(part, spell_pair) if _LETTERS_ONLY.test(part) else part
        runs = []
        for run in _DIGIT_OR_OTHER_RUNS.match_all(part):
            if not _ALL_DIGITS.test(run):
                runs.append(speak_letters(run, False) if _LETTERS_ONLY.test(run) else run)
            elif len(run) >= 3 or (len(run) > 1 and run.startswith("0")):
                runs.append(spell_german_digits(run))
            else:
                runs.append(integer_to_german_words(int(run)))
        return " ".join(runs)

    parts = (label or "").split("-")
    if any(not part for part in parts):
        return label

    def speak_pieces(part: str, spell_pair: bool) -> str:
        pair = spell_pair and not _UNDERSCORE_OR_PLUS.test(part)
        return "".join(
            piece if piece in ("_", "+") or not piece else speak_part(piece, pair)
            for piece in _UNDERSCORE_OR_PLUS_SPLIT.split(part)
        )

    return " minus ".join(speak_pieces(part, is_ending and len(parts) == 1) for part in parts)


_CAPITALISED_WORD = js(r"^\p{Lu}\p{Ll}+$", "u")


def speak_german_domain(label_part: str, tld: str) -> Optional[str]:
    labels = [part for part in (js_trim(p) for p in (label_part or "").split(".")) if part]
    if not labels:
        return None
    # A capital with lower-case letters behind a dot is the next sentence after a missing blank.
    if _CAPITALISED_WORD.test(tld or ""):
        return None
    host = [speak_german_address_label(part, False) for part in labels]
    return f"{' punkt '.join(host)} punkt {speak_german_address_label((tld or '').lower(), True)}"


def speak_german_time(hours: int, minutes: int, seconds: Optional[int] = None) -> str:
    hour_words = "ein" if hours == 1 else integer_to_german_words(hours)
    if seconds is not None:
        return f"{hour_words} Uhr {integer_to_german_words(minutes)} und {integer_to_german_words(seconds)} Sekunden"
    if minutes == 0:
        return f"{hour_words} Uhr"
    return f"{hour_words} Uhr {integer_to_german_words(minutes)}"


def speak_german_duration(hours, minutes, seconds, zero_form: str, dative: bool = False) -> str:
    def count(value: int, singular: str, plural: str) -> str:
        return f"{'einer' if dative else 'eine'} {singular}" if value == 1 else f"{integer_to_german_words(value)} {plural}"

    parts: list[str] = []
    if hours is not None and hours > 0:
        parts.append(count(hours, "Stunde", "Stunden"))
    if minutes > 0:
        parts.append(count(minutes, "Minute", "Minuten"))
    if seconds is not None and seconds > 0:
        parts.append(count(seconds, "Sekunde", "Sekunden"))
    if not parts:
        return zero_form
    if len(parts) == 1:
        return parts[0]
    return f"{', '.join(parts[:-1])} und {parts[-1]}"


def speak_german_date(day: int, month: int, year: int, prefix_token=None) -> Optional[str]:
    month_label = _month_label(month)
    if not month_label:
        return None
    raw_prefix = js_trim(prefix_token or "")
    day_label = german_ordinal_day(day, german_day_inflection_after(raw_prefix))
    if raw_prefix:
        return f"{speak_german_date_prefix(raw_prefix)} {day_label} {month_label} {year_to_german_words(year)}"
    return f"{day_label} {month_label} {year_to_german_words(year)}"


def speak_german_dateless_named_date(prefix, day_token: str, month_token: str) -> Optional[str]:
    """A day with a named month and no year ("am 6. September", "der 1. Mai")."""
    day = js_number(day_token)
    month = german_month_index(month_token)
    if not is_js_integer(day) or not month or day < 1 or day > 31:
        return None
    month_label = _month_label(month)
    if not month_label:
        return None
    raw_prefix = js_trim(prefix or "")
    day_label = german_ordinal_day(int(day), german_day_inflection_after(raw_prefix))
    return f"{speak_german_date_prefix(raw_prefix)} {day_label} {month_label}" if raw_prefix else f"{day_label} {month_label}"


GERMAN_BARE_YEAR_SOURCE = r"(?<![\w.\-])(?<!\bmal[ \t]+)(1[1-9]\d{2})(?![\p{L}\p{N}_]|[.,]\d|[ \t]+mal[ \t]+\d)"
GERMAN_BARE_YEAR_PATTERN = js(GERMAN_BARE_YEAR_SOURCE, "gu")
GERMAN_BARE_YEAR_AT = js(GERMAN_BARE_YEAR_SOURCE, "uy")

# Bug 837: a number grouped with spaces (also no-break, narrow no-break or thin space) or with
# apostrophes (the Swiss grouping) is joined into the dot grouping before any rule runs.
_GERMAN_GROUP_SPACE_SOURCE = "[ \u00A0\u202F\u2009]"
_GERMAN_GLUED_SYMBOL_SOURCE = (
    f"(?:x|{GERMAN_UNIT_KEY_SOURCE}|Tsd|Mio|Mrd|Bio\\.|€|\\$|£|EURO?|USD|DOLLAR|GBP|PFUND|CHF|JPY|YEN)"
)
_S = _GERMAN_GROUP_SPACE_SOURCE
GERMAN_SPACED_THOUSANDS_PATTERN = js(
    r"(?<![\d+.,]|[\d+]" + _S + r")(?=[1-9])"
    + r"(?:(?=[1-9]\d{0,2}(?:" + _S + r"000(?!\d))+(?!" + _S + r"\d|\.\d))|"
    + r"(?<!\b" + GERMAN_NUMBER_LABEL_SOURCE + _S + r"+(?:[1-9]\d?" + _S + r"\d{3}(?:[ \t]*,|" + _S + r"+(?:und|oder|bis))[ \t]+){0,5}))"
    + r"\b(?:[1-9]\d?(?:" + _S + r"\d{3}(?!\d))+|[1-9]\d{2}(?:" + _S + r"000(?!\d))+|[1-9]\d{2}(?:" + _S + r"\d{3}(?!\d))+(?=,\d|"
    + _S + r"?[€$£%]))"
    + r"(?!" + _S + r"\d|\.\d)(?:(?<=000)|(?!-\p{L}|(?!" + _GERMAN_GLUED_SYMBOL_SOURCE + r"(?!\p{L}))\p{L}))",
    "gu",
)
GERMAN_APOSTROPHE_THOUSANDS_PATTERN = js(
    r"(?<![\d+.,'‘’])\b([1-9]\d{0,2}(?:['‘’]\d{3}(?!\d))+(?:\.\d{3}(?!\d))*)(?:[.,](\d{1,2})(?!\d))?(?!\d|['‘’]\d|\.\d)", "g"
)
_APOSTROPHE_DIGIT = js(r"['‘’]\d", "")
_APOSTROPHES = js(r"['‘’]", "g")
_GROUP_SPACES = js("[ \u00A0\u202F\u2009]", "g")


def join_german_spaced_thousands(text: str, applied_rules: Optional[list[str]] = None) -> str:
    """``joinGermanSpacedThousands``: one dot per space, so offsets into the text stay valid."""
    rules = applied_rules if applied_rules is not None else []
    source = text or ""
    if _APOSTROPHE_DIGIT.test(source):
        source = replace_with_tracking(
            source,
            GERMAN_APOSTROPHE_THOUSANDS_PATTERN,
            lambda _m, integer_part, fraction, *_r: (
                f"{js_replace(_APOSTROPHES, integer_part, '.')}{',' + fraction if fraction is not None else ''}"
            ),
            "thousands_apostrophe_de",
            rules,
        )
    return replace_with_tracking(
        source,
        GERMAN_SPACED_THOUSANDS_PATTERN,
        lambda match, *_r: js_replace(_GROUP_SPACES, match, "."),
        "thousands_space_de",
        rules,
    )


def is_bare_german_year_at(text: str, offset: int, match: str) -> bool:
    m = GERMAN_BARE_YEAR_AT.match_at(text, offset)
    return bool(m and m.start() == offset and m.group(0) == match)


# Bug 422: a number that names something ("RTX 4090", "ProRes 422", "Wittenberg 4.0") stays in digits
# in the floor stage; the final stage reads it as before. Groups: 1 the integer, 2 the version.
_GERMAN_NAME_TOKEN_SOURCE = r"(?:\p{Lu}{2,}|\p{L}*\p{Ll}\p{Lu}\p{L}*)"
GERMAN_NAME_NUMBER_PATTERN = js(
    r"(?:(?<=(?<![\p{L}\p{N}_])" + _GERMAN_NAME_TOKEN_SOURCE + r"[ \t])(\d{3,4})|(?<=(?<![\p{L}\p{N}_])(?:"
    + _GERMAN_NAME_TOKEN_SOURCE + r"|(?!" + GERMAN_NUMBER_LABEL_SOURCE + r"[ \t])\p{Lu}\p{Ll}+)[ \t])(\d+\.(?!\d{3}(?!\d))\d+)"
    r"(?!(?<=(?<![\p{L}\p{N}_])\p{Lu}\p{Ll}+[ \t]\d+\.\d+)[ \t]+\p{Lu}\p{Ll}))(?=$|[\s,;:!?)\]\"'“”»–—-]|\.(?!\d))",
    "gu",
)

# Memo of the last text's name-number spans (as in de.ts). The optimizer runs on several threads at
# once (asyncio.to_thread per request), so text and spans live in one immutable tuple that is read
# once and replaced in a single assignment; a thread never pairs one text with another's spans.
_name_number_cache: tuple = (None, ())


def is_german_name_number_at(text: str, offset: int, match: str) -> bool:
    global _name_number_cache
    cached_text, spans = _name_number_cache
    if cached_text != text:
        spans = tuple((m.start(), m.end()) for m in GERMAN_NAME_NUMBER_PATTERN.finditer(text))
        _name_number_cache = (text, spans)
    end = offset + len(match)
    return any(offset < stop and end > start for start, stop in spans)


# Bug 1024: a day.month without a year right after a word is a date or a number that continues a
# name; the floor holds the unclear ones for the model. Groups of date_de_dm: prefix, day, month,
# glued letter, follower, the token before the word before, the word before, the side after the
# dot, a paired date after it.
_GERMAN_WEEKDAY_TOKEN = js(r"^(?:Mo|Di|Mi|Do|Fr|Sa|So|Montag|Dienstag|Mittwoch|Donnerstag|Freitag|Samstag|Sonntag)$", "")
_GERMAN_DAY_MONTH_WORD_BEFORE = js(r"^\p{L}[\p{L}\p{N}_+-]*$", "u")
_STARTS_ZERO = js(r"^0", "")
_VERSION_WORD_END = js(r"(?:^|[^A-Za-z0-9_])version$", "i")
_PAIR_JOINER_WORD = js(r"^(?:bis|und|oder|sowie)$", "i")
_DATE_DOT_END = js(r"\d\.,?$", "")


def is_ambiguous_german_day_month(groups) -> bool:
    prefix, day_token, month_token, glued, _follower, before_word, word, sentence_side, paired_after = (list(groups) + [None] * 9)[:9]
    if js_trim(prefix or "") or glued is not None or sentence_side is None or paired_after is not None:
        return False
    day = js_number(day_token)
    month = js_number(month_token)
    if not (1 <= day <= 31 and 1 <= month <= 12) or _STARTS_ZERO.test(_js_str(day_token)) or _STARTS_ZERO.test(_js_str(month_token)):
        return False
    if not word or not _GERMAN_DAY_MONTH_WORD_BEFORE.test(word) or _GERMAN_WEEKDAY_TOKEN.test(word) or _VERSION_WORD_END.test(word):
        return False
    return not (_PAIR_JOINER_WORD.test(word) and _DATE_DOT_END.test(before_word or ""))


def is_settled_german_day_month(groups) -> bool:
    prefix, day_token, month_token, glued = (list(groups) + [None] * 4)[:4]
    day = js_number(day_token)
    month = js_number(month_token)
    return not js_trim(prefix or "") and glued is None and 1 <= day <= 31 and 1 <= month <= 12


_ENDS_SENTENCE_SIDE = js(r"^[ \t]*(?:\n|$)|^[ \t\u00A0]+\p{Lu}", "u")


def speak_german_day_month_as_number(day_token: str, month_token: str, sentence_side) -> str:
    ends_sentence = _ENDS_SENTENCE_SIDE.test(sentence_side if sentence_side is not None else "")
    return f"{speak_german_version(None, [day_token, month_token])}{'.' if ends_sentence else ''}"


_DAY_MONTH_HINT_LEFTOVER = js(
    r"(?<![\w.])(\d{1,2})\.(\d{1,2})\.[ \t]*\([ \t]*([^()/\n]+?)[ \t]*(?:\/[ \t]*([^()/\n]+?)[ \t]*)?\)", "gu"
)


def resolve_german_day_month_hint_leftovers(text: str, applied_rules: list[str]) -> str:
    def callback(match, day_token, month_token, first, second, *_rest):
        day = js_number(day_token)
        month = js_number(month_token)
        if not (1 <= day <= 31 and 1 <= month <= 12):
            return match

        def fold(arm: str) -> str:
            return js_trim(js_replace(_TRAILING_DOT, arm, "")).lower()

        readings = {
            f"{german_ordinal_day(int(day), german_day_inflection_after(''))} {GERMAN_MONTHS[int(month) - 1]}".lower(),
            speak_german_version(None, [day_token, month_token]).lower(),
        }
        if second is not None:
            return first if fold(first) in readings and fold(second) in readings else match
        return first if fold(first) in readings else match

    return replace_with_tracking(text, _DAY_MONTH_HINT_LEFTOVER, callback, "day_month_hint_leftover_de", applied_rules)


_ANY_LOWER = js(r"\p{Ll}", "u")
_NEXT_LINE_NOT_LOWER = js(r"^[ \t]*\n[ \t]*[^a-zäöüß\s]", "")


def speak_german_dateless_day_month(_match, prefix, day_token, month_token, glued, follower, *_rest) -> Optional[str]:
    """The date reading of date_de_dm (Bug 399), shared by its speak and its hint (Bug 1024)."""
    if glued is not None and _ANY_LOWER.test(glued):
        return None
    opener = follower is not None and (_SENTENCE_OPENER_FOLLOWER.test(follower) or _NEXT_LINE_NOT_LOWER.test(follower))
    day = js_number(day_token)
    month = js_number(month_token)
    if not is_js_integer(day) or not is_js_integer(month) or day < 1 or day > 31 or month < 1 or month > 12:
        return None
    month_label = GERMAN_MONTHS[int(month) - 1]
    raw_prefix = js_trim(prefix or "")
    day_label = german_ordinal_day(int(day), german_day_inflection_after(raw_prefix))
    spoken = f"{speak_german_date_prefix(raw_prefix)} {day_label} {month_label}" if raw_prefix else f"{day_label} {month_label}"
    return f"{spoken}." if opener else spoken


GERMAN_SPAN_UNIT_AFTER_SOURCE = (
    r"(?:[ \t]*(?:[Mm]inuten?|[Ss]ekunden?|sec|s|(?:[Mm]in|[Ss]ek)(?:\.(?![ \t]*$|[ \t]+\p{Lu}|[ \t]*\n(?![ \t]*\p{Ll})))?)(?![\p{L}\d]))?"
)


def speak_german_span_with_fraction(minutes_token: str, seconds_token: str, fraction, dative: bool = False) -> str:
    minutes = int(minutes_token)
    seconds_value = int(seconds_token)
    if fraction is not None:
        seconds = f"{integer_to_german_words(seconds_value)} komma {spell_german_digits(fraction)} Sekunden"
    else:
        seconds = speak_german_duration(None, 0, seconds_value, "null Sekunden", dative)
    if minutes == 0:
        return seconds
    if fraction is None:
        return speak_german_duration(None, minutes, seconds_value, "null Sekunden", dative)
    minute_words = f"{'einer' if dative else 'eine'} Minute" if minutes == 1 else f"{integer_to_german_words(minutes)} Minuten"
    return f"{minute_words} und {seconds}"


def is_german_clock_time(hours: int, minutes: int) -> bool:
    return (hours <= 23 and minutes <= 59) or (hours == 24 and minutes == 0)


GERMAN_DOTTED_TIME_JOIN_SOURCE = r"(?:[ \t]?h)?[ \t]*(?:bis|[–—-])[ \t]*"
GERMAN_NO_MEASURE_AFTER_SOURCE = (
    r"(?![ \t]*(?:[%°€$£]|(?:" + "|".join(escape_for_regexp(symbol) for symbol in GERMAN_UNIT_WORDS)
    + r"|Mio|Mrd|Tsd|EUR|USD|CHF)(?![\p{L}\d])))"
)
GERMAN_CODE_LETTER_NUMBER_SOURCE = r"(?:(?<![\p{L}])(?:" + GERMAN_NUMBER_LABEL_SOURCE + r"|Paragrafen|§§?|\p{Lu})[ \t]*)"
GERMAN_NO_DATE_BEFORE_SOURCE = r"(?:(?<![\p{L}\d])(?:" + GERMAN_NUMBER_LABEL_SOURCE + r"|minus|plus|gleich)[ \t]+|[=±−][ \t]*)"
# --- speak callbacks -------------------------------------------------------------------------------

_SIMPLE_DIGITS = js(r"\d{1,2}", "g")


def _speak_version(_match, labelled, short_prefix, short_first, short_second, short_wildcard, prefix, first, second, third,
                   fourth, third_wildcard, second_wildcard, labeled_first, labeled_second, labeled_follower, bare_first,
                   bare_wildcard, tag, tag_number, *_rest):
    marked = labelled is not None
    version_prefix = None
    if short_first is not None:
        version_prefix = short_prefix
        spoken = speak_german_version(None, [short_first, short_second if short_second is not None else short_wildcard])
    elif labeled_first is not None:
        if is_german_measure_after_version(labeled_follower):
            return None
        spoken = speak_german_version(None, [labeled_first, labeled_second])
    elif bare_first is not None:
        if bare_wildcard == "*" and not marked:
            return None
        spoken = speak_german_version(None, [bare_first, bare_wildcard])
    else:
        version_prefix = prefix
        wildcard = third_wildcard if third_wildcard is not None else second_wildcard
        if wildcard == "*" and not prefix and not marked:
            return None
        if third is not None:
            if not fourth and not wildcard and looks_like_german_dotted_date(first, second, third, bool(prefix) or marked):
                return None
            if all(len(part) == 3 for part in (second, third, fourth) if part):
                return None
        spoken = speak_german_version(
            None, [first, second, third, fourth if fourth is not None else third_wildcard, second_wildcard]
        )
    if tag is not None:
        spoken = f"{spoken} {f'{tag} {speak_german_version_part(tag_number)}' if tag_number is not None else tag}"
    return f"Version {spoken}" if version_prefix and not marked else spoken


def _speak_slash_full(_match, prefix, day_token, month_token, year_token, *_rest):
    day = js_number(day_token)
    month = js_number(month_token)
    year = parse_spoken_german_year_token(year_token) if len(year_token) == 2 else int(year_token)
    if year is None:
        return None
    if not is_js_integer(day) or not is_js_integer(month) or day < 1 or day > 31 or month < 1 or month > 12:
        return None
    return speak_german_date(int(day), int(month), year, prefix) or None


def _speak_slash_day_month(_match, prefix, day_token, month_token, *_rest):
    if not is_slash_day_month(day_token, month_token):
        return None
    raw_prefix = js_trim(prefix or "")
    return (
        f"{speak_german_date_prefix(raw_prefix)} {german_ordinal_day(int(day_token), german_day_inflection_after(raw_prefix))}"
        f" {GERMAN_MONTHS[int(month_token) - 1]}"
    )


def _speak_month_year(_match, code, month_token, year_token, *_rest):
    if code is not None:
        return None
    month = js_number(month_token)
    year = js_number(year_token)
    if not is_js_integer(month) or month < 1 or month > 12 or not is_js_integer(year):
        return None
    return f"{GERMAN_MONTHS[int(month) - 1]} {year_to_german_words(int(year))}"


def _speak_fraction(_match, code, numerator_token, denominator_token, inch, compound, *_rest):
    if code is not None and inch is None:
        return None
    numerator = js_number(numerator_token)
    denominator = js_number(denominator_token)
    label = GERMAN_FRACTION_DENOMINATORS.get(int(denominator)) if is_js_integer(denominator) else None
    if not label or not is_js_integer(numerator) or (numerator >= denominator and compound is None):
        return None
    numerator_words = "ein" if numerator == 1 else integer_to_german_words(int(numerator))
    return f"{numerator_words} {label}"


_ODER_WORD = js(r"oder", "i")


def _speak_day_list(_match, prefix, first_day, middle, joiner_token, last_day, month_name, month_dot, month_number,
                    follower, *_rest):
    days = [int(first_day), *(int(day) for day in _SIMPLE_DIGITS.match_all(str(middle))), *([int(last_day)] if last_day else [])]
    month = german_month_index(month_name) if month_name else js_number(month_number)
    if not month or not (1 <= month <= 12) or any(day < 1 or day > 31 for day in days):
        return None
    month = int(month)
    raw_prefix = js_trim(prefix or "")
    inflection = german_day_inflection_after(raw_prefix)
    spoken_days = [german_ordinal_day(day, inflection) for day in days]
    joiner = "oder" if _ODER_WORD.test(joiner_token or "") else "und"
    listed = f"{', '.join(spoken_days[:-1])} {joiner} {spoken_days[-1]}"
    spoken = f"{speak_german_date_prefix(raw_prefix) + ' ' if raw_prefix else ''}{listed} {GERMAN_MONTHS[month - 1]}"
    return f"{spoken}." if month_name and german_dot_ends_sentence(month_dot, follower) else spoken


def _speak_day_range(_match, prefix, day1, month1, joiner, second_prefix, day2, month_numeric, year_numeric, month_name,
                     month_dot, year_named, follower, *_rest):
    spoken = speak_german_day_range(
        prefix, day1, month1, joiner, day2, month_numeric, year_numeric, month_name, year_named, second_prefix
    )
    return f"{spoken}." if spoken and not year_named and german_dot_ends_sentence(month_dot, follower) else spoken


def _speak_named_month(_match, prefix, day_token, month_token, year_token, *_rest):
    day = js_number(day_token)
    month = german_month_index(month_token)
    year = parse_spoken_german_year_token(year_token)
    if not is_js_integer(day) or not month or year is None or day < 1 or day > 31:
        return None
    return speak_german_date(int(day), month, year, prefix) or None


def _speak_named_month_no_year(_match, prefix, day_token, month_token, month_dot, follower, *_rest):
    spoken = speak_german_dateless_named_date(prefix, day_token, month_token)
    return f"{spoken}." if spoken and german_dot_ends_sentence(month_dot, follower) else spoken


def _speak_dmy(_match, prefix, day_token, month_token, year_token, *_rest):
    day = js_number(day_token)
    month = js_number(month_token)
    year = js_number(year_token)
    if not is_js_integer(day) or not is_js_integer(month) or not is_js_integer(year) or day < 1 or day > 31 or month < 1 or month > 12:
        return None
    return speak_german_date(int(day), int(month), int(year), prefix) or None


def _speak_dmy_short_year(_match, prefix, day_token, month_token, year_token, *_rest):
    if not prefix and (len(day_token) < 2 or len(month_token) < 2):
        return None
    day = js_number(day_token)
    month = js_number(month_token)
    year = parse_spoken_german_year_token(year_token)
    if not is_js_integer(day) or not is_js_integer(month) or year is None or day < 1 or day > 31 or month < 1 or month > 12:
        return None
    return speak_german_date(int(day), int(month), year, prefix) or None


def _speak_dm_padded(_match, prefix, day_token, month_token, *_rest):
    day = int(day_token)
    month = int(month_token)
    if day < 1 or day > 31 or month < 1 or month > 12:
        return None
    return f"{js_trim(str(prefix))} {german_ordinal_day(day, 'oblique')} {GERMAN_MONTHS[month - 1]}"


def _speak_iso(_match, prefix, year_token, month_token, day_token, hours_token, minutes_token, *_rest):
    day = js_number(day_token)
    month = js_number(month_token)
    year = js_number(year_token)
    if not is_js_integer(day) or not is_js_integer(month) or not is_js_integer(year) or day < 1 or day > 31 or month < 1 or month > 12:
        return None
    spoken_date = speak_german_date(int(day), int(month), int(year), prefix)
    if not spoken_date:
        return None
    if hours_token is not None:
        hours = int(hours_token)
        minutes = int(minutes_token)
        if hours <= 23 and minutes <= 59:
            return f"{spoken_date} um {speak_german_time(hours, minutes)}"
    return spoken_date


def _speak_year_month(_match, year_token, month_token, *_rest):
    return f"{GERMAN_MONTHS[int(month_token) - 1]} {year_to_german_words(int(year_token))}"


def _speak_duration_compact(_match, dative_mark, days_token, hours_token, minutes_token, seconds_token, only_minutes,
                            only_seconds, *_rest):
    dative = dative_mark is not None
    if only_minutes is not None:
        return speak_german_duration(None, int(only_minutes), int(only_seconds), "null Sekunden", dative)
    minutes = int(minutes_token)
    seconds = None if seconds_token is None else int(seconds_token)
    if minutes > 59 or (seconds is not None and seconds > 59):
        return None
    span = speak_german_duration(int(hours_token), minutes, seconds, "null Minuten", dative)
    if days_token is None:
        return span
    days = int(days_token)
    days_words = ("einem Tag" if dative else "ein Tag") if days == 1 else f"{integer_to_german_words(days)} Tage"
    return f"{days_words}, {span}"


_HOUR_UNIT = js(r"^(?:st|h)", "i")


def _speak_duration_unit(_match, dative_mark, first_token, second_token, third_token, unit_token, *_rest):
    dative = dative_mark is not None
    first = int(first_token)
    second = int(second_token)
    third = None if third_token is None else int(third_token)
    if second > 59 or (third is not None and third > 59):
        return None
    if third is not None:
        return speak_german_duration(first, second, third, "null Sekunden", dative)
    if _HOUR_UNIT.test(unit_token or ""):
        return speak_german_duration(first, second, None, "null Stunden", dative)
    return speak_german_duration(None, first, second, "null Minuten", dative)


def _speak_duration_context(_match, dative_mark, frac_minutes, frac_seconds, fraction, per_minutes, per_seconds,
                            label_hours, label_minutes, label_seconds, verb_hours, verb_minutes, verb_seconds,
                            minus_minutes, minus_seconds, minus_fraction, plus_minutes, plus_seconds, plus_fraction, *_rest):
    dative = dative_mark is not None
    if frac_minutes is not None:
        return speak_german_span_with_fraction(frac_minutes, frac_seconds, fraction, dative)
    if minus_minutes is not None:
        return f"minus {speak_german_span_with_fraction(minus_minutes, minus_seconds, minus_fraction)}"
    if plus_minutes is not None:
        return speak_german_span_with_fraction(plus_minutes, plus_seconds, plus_fraction)
    if per_minutes is not None:
        return speak_german_duration(None, int(per_minutes), int(per_seconds), "null Minuten", dative)
    return speak_german_duration(
        int(label_hours if label_hours is not None else verb_hours),
        int(label_minutes if label_minutes is not None else verb_minutes),
        int(label_seconds if label_seconds is not None else verb_seconds),
        "null Sekunden",
        dative,
    )


def _speak_hhmm_dot(_match, hours_token, minutes_token, *_rest):
    hours = int(hours_token)
    minutes = int(minutes_token)
    if not is_german_clock_time(hours, minutes):
        return None
    return speak_german_time(hours, minutes)


def _day_and_month(day: int, month: int) -> bool:
    return 1 <= day <= 31 and 1 <= month <= 12


def _speak_hhmm_dot_context(_match, h_hours, h_minutes, first_hours, first_minutes, next_hours, next_minutes,
                            previous_hours, previous_minutes, second_hours, second_minutes, *_rest):
    if h_hours is not None:
        hours = int(h_hours)
        minutes = int(h_minutes)
        return speak_german_time(hours, minutes) if is_german_clock_time(hours, minutes) else None
    if first_hours is not None:
        hours_token, minutes_token, other_hours_token, other_minutes_token = first_hours, first_minutes, next_hours, next_minutes
    else:
        hours_token, minutes_token, other_hours_token, other_minutes_token = (
            second_hours, second_minutes, previous_hours, previous_minutes
        )
    hours = int(hours_token)
    minutes = int(minutes_token)
    other_hours = int(other_hours_token)
    other_minutes = int(other_minutes_token)
    two_digit_hours = len(hours_token) == 2 or len(other_hours_token) == 2
    if not is_german_clock_time(hours, minutes) or not is_german_clock_time(other_hours, other_minutes) or not two_digit_hours:
        return None
    if _day_and_month(hours, minutes) and _day_and_month(other_hours, other_minutes):
        return None
    return speak_german_time(hours, minutes)


def _speak_hhmm(_match, hours_token, minutes_token, seconds_token, *_rest):
    hours = int(hours_token)
    minutes = int(minutes_token)
    seconds = None if seconds_token is None else int(seconds_token)
    if hours < 0 or not is_german_clock_time(hours, minutes) or (
        seconds is not None and (seconds < 0 or seconds > 59 or (hours == 24 and seconds > 0))
    ):
        return None
    if hours == 24:
        return speak_german_time(24, 0)
    return speak_german_time(hours, minutes, seconds)


def _speak_hour_only(_match, hours_token, *_rest):
    hours = int(hours_token)
    if hours < 0 or hours > 24:
        return None
    return speak_german_time(hours, 0)


def _speak_amount_scale(_match, dative, prefix_currency, number_token, scale_token, scale_dot, suffix_currency, follower,
                        next_word, *_rest):
    spoken = speak_german_amount_scale(number_token, scale_token, prefix_currency, suffix_currency, dative is not None)
    before_opener = bool(scale_dot) and next_word is not None and _SENTENCE_OPENER_FOLLOWER.test(next_word)
    at_text_end = bool(scale_dot) and _BLANKS_ONLY.test(follower or "")
    if spoken and not suffix_currency and (german_dot_ends_line(scale_dot, follower) or before_opener or at_text_end):
        return f"{spoken}."
    return spoken


_SINGLE_LETTER = js(r"^\p{L}$", "u")
_SINGLE_UPPER = js(r"^\p{Lu}$", "u")


def _speak_unit_suffix(_match, dative, code_number, number_token, spacing, unit_token, *_rest):
    if code_number is not None and _SINGLE_LETTER.test(unit_token):
        if _SINGLE_UPPER.test(unit_token) or spacing == "":
            return f"{speak_german_numeric_token(number_token)} {unit_token}"
    return speak_german_unit(number_token, spacing, unit_token, dative is not None)


_DIMENSION_X = js(r"[ \t]*x[ \t]*", "")


def _speak_dimension_unit(_match, chain, spacing, unit_token, *_rest):
    operands = _DIMENSION_X.split(chain or "")
    unit = GERMAN_UNIT_WORDS.get(unit_token)
    if not unit or not is_german_unit_reading_plausible(operands[-1], spacing, unit_token):
        return None
    return f"{' mal '.join(speak_german_numeric_token(operand) for operand in operands)} {unit[1]}"


def _speak_currency_pair(_match, base_code, quote_code, *_rest):
    base = GERMAN_CURRENCY_PAIR_NAMES.get(base_code)
    quote = GERMAN_CURRENCY_PAIR_NAMES.get(quote_code)
    return f"{base} zu {quote}" if base and quote and base_code != quote_code else None


def _speak_currency_prefix(_match, sign, currency_token, amount_token, *_rest):
    negative = "-" if (sign or amount_token.startswith("-")) else ""
    unsigned = amount_token[1:] if amount_token.startswith("-") else amount_token
    return speak_german_currency(f"{negative}{unsigned}", currency_token)


def _speak_per_mille(_match, number_token, sign, kelvin_token, *_rest):
    if kelvin_token is not None:
        return speak_german_temperature(kelvin_token, "K")
    if sign == "‰":
        return f"{speak_german_attributive_numeric_token(number_token)} Promille"
    return f"{speak_german_numeric_token(number_token)} pro zehntausend"


_X_SPLIT = js(r"\s*[x×]\s*", "")


def _speak_dimension(_match, chain, *_rest):
    return " mal ".join(speak_german_numeric_token(token) for token in _X_SPLIT.split(chain or ""))


GERMAN_SPEAK_PASSES: list[SpeakPass] = [
    SpeakPass("email_address_span_de", js(EMAIL_ADDRESS_PATTERN_SOURCE, "giu"), lambda match, *_r: match),
    SpeakPass(
        "domain_de",
        js(
            r"(?<![\p{L}\p{N}_@/.-])((?:[\p{L}\p{N}](?:[\p{L}\p{N}-]*[\p{L}\p{N}])?\.)+)(de|com|org|net|eu|at|ch|io|dev|info|news)"
            r"(?![\p{L}\p{N}_-])",
            "giu",
        ),
        lambda _m, label_part, tld, *_r: speak_german_domain(label_part, tld),
    ),
    SpeakPass(
        "network_port_de",
        js(r"\b(port(?:nummer)?)(?:\s*[:=]\s*|\s+)(\d{1,5})\b|(?<=\s)(?<!\d[ \t]+):(\d{2,5})\b(?![.,:]?\d)", "gi"),
        lambda _m, label, number_token, bare_token, *_r: (
            speak_german_network_port("Port", bare_token) if bare_token is not None else speak_german_network_port(label, number_token)
        ),
    ),
    SpeakPass(
        "ipv4_de",
        js(r"(?<![\w.])(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})(?::(\d{1,5}))?(?!\w|\.\d)", "g"),
        lambda _m, a, b, c, d, port, *_r: speak_german_ipv4([a, b, c, d], port),
    ),
    SpeakPass(
        "version_de",
        js(
            r"(?<![\w.,-])(?:(?<=\bversion[ \t]*:?[ \t]*)()|)"
            r"(?:(v)(\d{1,3})\.(?:(\d{1,3})(?!\d|\.(?:\d|" + VERSION_WILDCARD_SOURCE + r"))|(" + VERSION_WILDCARD_SOURCE + r"))"
            r"|(v)?(\d{1,3})\.(\d{1,3})\.(?:(\d{1,4})(?:\.(?:(\d{1,4})|(" + VERSION_WILDCARD_SOURCE + r")))?(?!\d|[.,]\d)|("
            + VERSION_WILDCARD_SOURCE + r"))"
            r"|(?<=\bversion[ \t]+)(\d{1,3})\.(\d{1,2})(?!\d|[.,]\d)(?:(?=[ \t]*(%|[^\s\d]+))|)"
            r"|(\d{1,3})\.(" + VERSION_WILDCARD_SOURCE + r"))"
            r"(?:-(" + VERSION_PRE_RELEASE_TAG_SOURCE + r")(?:\.?(\d{1,3}))?(?![\w-]|[.,]\d))?",
            "gi",
        ),
        _speak_version,
    ),
    SpeakPass(
        "date_de_slash_full",
        js(
            r"(?<![\d.,/])(?:(?<![A-Za-zÀ-ÖØ-öø-ÿ])" + GERMAN_DATE_PREFIX_SOURCE
            + r")?(\d{1,2})\/(\d{1,2})\/((?:19|20)\d{2}|(?<=\d{2}\/\d{2}\/)\d{2})(?![\d/])",
            "gi",
        ),
        _speak_slash_full,
    ),
    SpeakPass(
        "date_de_slash_day_month",
        js(
            r"\b" + GERMAN_DATELESS_DATE_PREFIX_SOURCE + r"(\d{1,2})\/(\d{1,2})(?![\d/]|[.,]\d)" + GERMAN_NO_MEASURE_AFTER_SOURCE
            + r"(?![ \t]*[\"″]|" + GERMAN_PAIR_CONTEXT["compoundAfter"] + r"|" + GERMAN_PAIR_CONTEXT["countAfter"] + r")",
            "giu",
        ),
        _speak_slash_day_month,
    ),
    SpeakPass(
        "date_de_month_year",
        js(r"(?<![\d.,/])(?:(?<=" + GERMAN_PAIR_CONTEXT["codeBefore"] + r")()|)(0?[1-9]|1[0-2])\/((?:19|20)\d{2})(?![\d/])", "gu"),
        _speak_month_year,
    ),
    SpeakPass(
        "fraction_de",
        js(
            r"(?<![\d.,/])(?:(?<=" + GERMAN_PAIR_CONTEXT["codeBefore"] + r")()|)([1-9]\d?)\/(100|1[0-2]|[2-9])(?![\d/])(?![.,]\d)"
            r"(?:(?=" + GERMAN_PAIR_CONTEXT["inchAfter"] + r")()|(?=" + GERMAN_PAIR_CONTEXT["compoundAfter"] + r")()|)",
            "gu",
        ),
        _speak_fraction,
    ),
    SpeakPass(
        "phone_extension_de",
        js(
            r"\b((?:[Dd]urchwahl|DW|[Tt]el|[Tt]elefon|[Ff]ax|[Zz]entrale|[Mm]obil|[Hh]andy|[Nn]ebenstelle|[Aa]pparat|[Ee]xt))"
            r"(\.?:?[ \t]+(?:(?:ist|lautet)[ \t]+)?)-[ \t]?(\d{1,6})(?!\d|[.,]\d)",
            "g",
        ),
        lambda _m, label, link, digits, *_r: f"{label}{link}{spell_german_digits(digits)}",
    ),
    SpeakPass("phone_number_de", js(PHONE_NUMBER_PATTERN_SOURCE, "gu"), lambda match, *_r: speak_german_phone_number(match)),
    SpeakPass(
        "date_de_day_list",
        js(
            r"(?<![\d.])\b((?:am|dem|den|vom|zum|bis|von|ab|seit|nach|vor|für|der)\s+)?(\d{1,2})\.((?:[ \t]*[/,][ \t]*\d{1,2}\.)+)"
            r"(?:[ \t]*(und|oder)[ \t]*(\d{1,2})\.)?[ \t]*(?:(" + GERMAN_MONTH_SOURCE + r")(\.?)(?![\p{L}])|(\d{1,2})\.(?!\d))"
            + GERMAN_FOLLOWER_SOURCE,
            "giu",
        ),
        _speak_day_list,
    ),
    SpeakPass(
        "date_de_day_range",
        js(
            r"(?<![\d.])\b((?:am|dem|den|vom|zum|bis|zwischen|von|ab|seit)\s+(?:dem\s+)?)?(\d{1,2})\.(?:\s?(\d{1,2})\.)?\s*(bis|und|oder|[–—-])"
            r"\s*(zum\s+|dem\s+)?(\d{1,2})\.\s*(?:(\d{1,2})\.(?:(\d{4}|\d{2})\b|(?!\d))|(" + GERMAN_MONTH_SOURCE
            + r")(\.?)(?![\p{L}])(?:\s+((?:19|20)\d{2})\b)?)" + GERMAN_FOLLOWER_SOURCE,
            "giu",
        ),
        _speak_day_range,
    ),
    SpeakPass(
        "date_de_named_month",
        js(r"\b" + GERMAN_DATE_PREFIX_SOURCE + r"?(\d{1,2})\.\s*(" + GERMAN_MONTH_SOURCE + r")\.?\s+(\d{2,4})\b(?!:)", "gi"),
        _speak_named_month,
    ),
    SpeakPass(
        "date_de_named_month_no_year",
        js(
            r"\b" + GERMAN_DATELESS_DATE_PREFIX_SOURCE + r"?(\d{1,2})\.\s*(" + GERMAN_MONTH_SOURCE
            + r")(\.?)(?![\p{L}])(?!\s*,?\s*\d{2,4}\b(?!:))" + GERMAN_FOLLOWER_SOURCE,
            "giu",
        ),
        _speak_named_month_no_year,
    ),
    SpeakPass("date_de_dmy", js(r"\b" + GERMAN_DATE_PREFIX_SOURCE + r"?(\d{1,2})\.(\d{1,2})\.(\d{4})\b", "gi"), _speak_dmy),
    SpeakPass(
        "date_de_dmy_short_year",
        js(r"\b" + GERMAN_DATE_PREFIX_SOURCE + r"?(\d{1,2})\.(\d{1,2})\.(\d{2})\b", "gi"),
        _speak_dmy_short_year,
    ),
    SpeakPass(
        "date_de_dm_padded",
        js(r"\b((?:am|dem|den|vom|zum)\s+)(\d{2})\.(\d{2})(?![\d.,\p{L}]|[ \t]*(?:-\d|Uhr\b|%|[€$£]))", "giu"),
        _speak_dm_padded,
    ),
    SpeakPass(
        "date_de_dm_number_period",
        js(r"(?<=" + GERMAN_NO_DATE_BEFORE_SOURCE + r")(\d{1,2}\.\d{1,2})\.(?=[ \t]+\p{Ll})", "gu"),
        lambda _m, pair, *_r: pair,
    ),
    SpeakPass(
        "date_de_dm",
        js(
            r"(?:\b" + GERMAN_DATELESS_DATE_PREFIX_SOURCE + r"|(?<![\w|/\-]|\d[-–]\d+[,;]?\s*)(?<!" + GERMAN_NO_DATE_BEFORE_SOURCE
            + r"))(\d{1,2})\.(\d{1,2})\.(?!\d)(?:(?=(\p{L}))|)(?:(?=([ \t]+\S+|[ \t]*\n[ \t]*\S))|)"
            r"(?:(?<=(?:(\S+)[ \t\u00A0]+)?(\S+)[ \t\u00A0]+\d{1,2}\.\d{1,2}\.)|)(?:(?=([ \t]*(?:\n|$)|[ \t\u00A0]+\p{L}))|)"
            r"(?:(?=([ \t]*(?:bis|und|oder|–|—|-|,)[ \t]*(?:zum[ \t]+)?\d))|)",
            "giu",
        ),
        speak_german_dateless_day_month,
    ),
    SpeakPass(
        "date_iso",
        js(
            r"(?:\b" + GERMAN_DATE_PREFIX_SOURCE + r")?(?<![\p{L}\d])(\d{4})-(\d{2})-(\d{2})(?:T(\d{2}):(\d{2})(?::(\d{2}))?(?:\.\d+)?"
            r"(?:Z|[+-]\d{2}:?\d{2})?)?(?!\d)",
            "giu",
        ),
        _speak_iso,
    ),
    SpeakPass("date_de_year_month", js(r"(?<![\w.-])((?:19|20)\d{2})-(0[1-9]|1[0-2])(?![\w-])", "g"), _speak_year_month),
    SpeakPass(
        "duration_de_compact",
        js(
            r"(?:(?<=" + GERMAN_DATIVE_BEFORE_SOURCE + r")()|)\b(?:(\d{1,3})[ \t]?(?:d|Tage?)[ \t]?)?(\d{1,3})[ \t]?(?:h|Std(?:\.(?![ \t]*$|[ \t]+"
            + GERMAN_SENTENCE_OPENER_SOURCE + r"(?![\p{L}])|[ \t]*\n(?![ \t]*\p{Ll})))?|Stunden?)[ \t]?(?:und[ \t])?(\d{1,2})[ \t]?"
            r"(?:m|min|Min(?:\.(?![ \t]*$|[ \t]+" + GERMAN_SENTENCE_OPENER_SOURCE + r"(?![\p{L}])|[ \t]*\n(?![ \t]*\p{Ll})))?|Minuten?)"
            r"(?![\p{L}])(?:[ \t]?(?:und[ \t])?(\d{1,2})[ \t]?(?:s|sec|Sek(?:\.(?![ \t]*$|[ \t]+" + GERMAN_SENTENCE_OPENER_SOURCE
            + r"(?![\p{L}])|[ \t]*\n(?![ \t]*\p{Ll})))?|Sekunden?)(?![\p{L}]))?|\b(\d{1,3})[ \t]?m[ \t]?(\d{1,2})[ \t]?s(?![\p{L}])",
            "gu",
        ),
        _speak_duration_compact,
    ),
    SpeakPass(
        "duration_de_unit",
        js(
            r"(?:(?<=(?<![\p{L}])(?:[Nn]ach|[Mm]it|[Ss]eit|[Vv]on|[Vv]or|[Uu]nter|[Ii]n|[Aa]us|[Bb]ei|[Zz]u)[ \t]+)()|)\b(\d{1,3}):(\d{2})"
            r"(?::(\d{2}))?\s*([Mm]inuten?|[Mm]in(?:\.(?!\s*$|\s+\p{Lu}|[ \t]*\n(?![ \t]*\p{Ll})))?|MIN|[Ss]tunden?"
            r"|[Ss]td(?:\.(?!\s*$|\s+\p{Lu}|[ \t]*\n(?![ \t]*\p{Ll})))?|STD|h|[Ss]ekunden?|[Ss]ek(?:\.(?!\s*$|\s+\p{Lu}|[ \t]*\n(?![ \t]*\p{Ll})))?"
            r"|sec|s)(?![\p{L}])",
            "gu",
        ),
        _speak_duration_unit,
    ),
    SpeakPass(
        "duration_de_context",
        js(
            r"(?:(?<=" + GERMAN_DATIVE_BEFORE_SOURCE + r")()|)(?<![\d.,:])(?:(?<![-+−])(?<!(?<![\p{L}\d])plus[ \t]+)(\d{1,2}):([0-5]\d)[.,]"
            r"(?!000(?!\d))(\d{1,3})(?![\d:]|[.,]\d)" + GERMAN_SPAN_UNIT_AFTER_SOURCE
            + r"|(\d{1,2}):([0-5]\d)(?![\d:]|[.,]\d)(?=[ \t]*pro\b)"
            + r"|(?<!" + GERMAN_PAIR_CONTEXT["clockPrepositionBefore"] + r")(?<=" + GERMAN_PAIR_CONTEXT["durationBefore"]
            + r")(\d{1,2}):([0-5]\d):([0-5]\d)(?![\d:]|[.,]\d)(?![ \t]*(?:Uhr\b|bis[ \t]+\d{1,2}:\d{2}))"
            + r"|(?<!" + GERMAN_PAIR_CONTEXT["clockPrepositionBefore"] + r")(\d{1,2}):([0-5]\d):([0-5]\d)(?![\d:]|[.,]\d)(?="
            + GERMAN_PAIR_CONTEXT["durationAfter"] + r")"
            + r"|(?<=(?:^|[\s(\[]))-(\d):([0-5]\d)(?:[.,](\d{1,3}))?(?![\d:]|[.,]\d)" + GERMAN_SPAN_UNIT_AFTER_SOURCE
            + r"|(?<=(?<!(?:[\d)\]]|\p{Lu}{2,}))[ \t]+plus[ \t]+|(?:^|[(\[])plus[ \t]+)(\d):([0-5]\d)(?:[.,](\d{1,3}))?(?![\d:]|[.,]\d)"
            + GERMAN_SPAN_UNIT_AFTER_SOURCE + r")",
            "gmu",
        ),
        _speak_duration_context,
    ),
    SpeakPass(
        "time_de_hhmm_dot",
        js(r"\b(\d{1,2})[.,](\d{2})(?:[ \t]*Uhr\b|(?=[ \t]*(?:bis|und|[–—-])[ \t]*\d{1,2}[.,]\d{2}[ \t]*Uhr\b))", "gi"),
        _speak_hhmm_dot,
    ),
    SpeakPass(
        "time_de_hhmm_dot_context",
        js(
            r"(?<![\d.,])(?=\d{1,2}\.\d{2})(?:(?<!" + GERMAN_PAIR_CONTEXT["durationBefore"] + r")(\d{1,2})\.(\d{2})[ \t]?h\b"
            + r"|(\d{1,2})\.(\d{2})(?=" + GERMAN_DOTTED_TIME_JOIN_SOURCE + r"(\d{1,2})\.(\d{2})(?![\d]|[.,]\d)(?:[ \t]?h\b|"
            + GERMAN_NO_MEASURE_AFTER_SOURCE + r"))"
            + r"|(?<=(?<![\d.,])(\d{1,2})\.(\d{2})" + GERMAN_DOTTED_TIME_JOIN_SOURCE + r")(\d{1,2})\.(\d{2})(?![\d]|[.,]\d)"
            + GERMAN_NO_MEASURE_AFTER_SOURCE + r")",
            "gu",
        ),
        _speak_hhmm_dot_context,
    ),
    SpeakPass("time_de_hhmm", js(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?(?:\s*(uhr))?\b", "gi"), _speak_hhmm),
    SpeakPass("time_de_hour_only", js(r"(?<!\d[.,:])\b(\d{1,2})\s*uhr\b", "gi"), _speak_hour_only),
    SpeakPass(
        "amount_scale_de",
        js(
            r"(?:(?<=" + GERMAN_DATIVE_BEFORE_SOURCE + r")()|)(?<!\w)(?:(€|\$|£|(?:EURO?|USD|DOLLAR|GBP|PFUND|CHF|JPY|YEN)\b)\s*)?"
            r"(-?\d[\d.,]*\d|-?\d)\s*(Mio|Mrd|Tsd|Bio(?=\.))\b(\.?)(?:\s*(€|\$|£|(?:EURO?|USD|DOLLAR|GBP|PFUND|CHF|JPY|YEN)\b))?"
            + GERMAN_FOLLOWER_SOURCE + r"(?:(?=([ \t]+[A-ZÄÖÜ][a-zäöüß]*))|)",
            "giu",
        ),
        _speak_amount_scale,
    ),
    SpeakPass(
        "unit_suffix_de",
        js(
            r"(?<!\w)(?<!\d[.,])(?:(?<=" + GERMAN_DATIVE_BEFORE_SOURCE + r")()|)(?:(?<=" + GERMAN_CODE_LETTER_NUMBER_SOURCE
            + r")()|)(-?\d(?:[\d.,]*\d)?)([ \t]*|-)(" + GERMAN_UNIT_KEY_SOURCE + r")(?:(?<=Min|Sek|Std)\.(?![ \t]*$|[ \t]+"
            + GERMAN_SENTENCE_OPENER_SOURCE + r"(?![\p{L}])|[ \t]*\n(?![ \t]*\p{Ll}))|(?<!Min|Sek|Std)\.(?![ \t]*$|[ \t]+\p{Lu}"
            r"|[ \t]*\n(?![ \t]*\p{Ll})))?(?![\p{L}\d])",
            "gu",
        ),
        _speak_unit_suffix,
    ),
    SpeakPass(
        "dimension_unit_de",
        js(
            r"(?<![\w.,])((?:[1-9](?:[\d.]*\d)?|0(?=,))(?:,\d+)?(?:[ \t]*x[ \t]*\d(?:[\d.]*\d)?(?:,\d+)?)+)([ \t]*)("
            + GERMAN_UNIT_KEY_SOURCE + r")(?:\.(?![ \t]*$|[ \t]+\p{Lu}|[ \t]*\n(?![ \t]*\p{Ll})))?(?![\p{L}\d])",
            "gu",
        ),
        _speak_dimension_unit,
    ),
    SpeakPass("hex_color_de", js(r"#([0-9a-fA-F]{6}|[0-9a-fA-F]{8})\b", "g"), lambda _m, hex_token, *_r: spell_german_hex(hex_token)),
    SpeakPass(
        "currency_pair_de",
        js(
            r"(?<![\p{L}\p{N}/])(" + GERMAN_CURRENCY_PAIR_CODE_SOURCE + r")\/(" + GERMAN_CURRENCY_PAIR_CODE_SOURCE + r")(?![\p{L}\p{N}/])",
            "gu",
        ),
        _speak_currency_pair,
    ),
    SpeakPass(
        "currency_prefix",
        js(
            r"(?<!\w)(?<!\/(?=[A-Za-z]))(?<!\d[ \t]*)(-?)(" + GERMAN_CURRENCY_SOURCE + r")(?:(?<=[€$£])\s*|\s+)(-?\d(?:[\d.,]*\d)?)"
            r"(?![\d.,]*[ \t]*%)",
            "gi",
        ),
        _speak_currency_prefix,
    ),
    SpeakPass(
        "currency_standard_de",
        js(r"(?<![\p{L}\d])(Euro)[- ](\d)([a-z])?(?![\p{L}\d]|[.,]\d)", "gu"),
        lambda _m, name, digit, letter, *_r: f"{name} {integer_to_german_words(int(digit))}{' ' + letter if letter else ''}",
    ),
    SpeakPass(
        "currency_suffix",
        js(
            r"(?<!\w)(-?\d(?:[\d.,]*\d)?)\s*(" + GERMAN_CURRENCY_SOURCE + r")(?=$|[\s,.;:!?)\]“”\"»«’']|\/(?!(?:EUR|USD|GBP|CHF|JPY)\b))",
            "gi",
        ),
        lambda _m, amount_token, currency_token, *_r: speak_german_currency(amount_token, currency_token),
    ),
    SpeakPass(
        "per_mille_de",
        js(r"(?<!\w)(-?\d(?:[\d.,]*\d)?)\s*(‰|‱)|(?<!\w)(-?\d+,\d+)[ \t]?K\b(?![ \t]+\p{Lu})", "gu"),
        _speak_per_mille,
    ),
    SpeakPass(
        "percent_de",
        js(r"(?<!\w)(-?\d[\d.,]*)\s*%|(?<=\))[ \t]?%", "g"),
        lambda _m, number_token, *_r: " Prozent" if number_token is None else speak_german_percent(number_token),
    ),
    SpeakPass(
        "temperature_de",
        js(r"(?<!\w)(-?\d[\d.,]*)\s*(?:°|º)\s*([CFK])\b", "gi"),
        lambda _m, number_token, unit_token, *_r: speak_german_temperature(number_token, (unit_token or "").upper()),
    ),
    SpeakPass(
        "dimension_de",
        js(
            r"(?<![\w.,])((?:[1-9]\d{0,4}|0(?=,))(?:,\d+)?(?:\s*[x×]\s*\d{1,5}(?:,\d+)?)+)(?!\w|[.,]\d)(?![ \t]*(?:[€$£]|(?:EUR|eur|Euro"
            r"|euro|USD|Dollar|GBP|Pfund|CHF|JPY|Yen|" + GERMAN_UNIT_KEY_SOURCE + r")(?![\p{L}\d])))",
            "gu",
        ),
        _speak_dimension,
    ),
    SpeakPass(
        "name_number_de",
        GERMAN_NAME_NUMBER_PATTERN,
        lambda _m, integer_token, version_token, *_r: speak_german_numeric_token(integer_token or version_token),
        lambda *_args: True,
    ),
    SpeakPass(
        "decimal_number_de",
        js(r"(?<![\w.,])(-?\d{1,3}(?:\.\d{3})+,\d+|-?\d+,\d+|-?\d+\.\d+)(?![.,]?\d)", "g"),
        lambda _m, number_token, *_r: speak_german_numeric_token(number_token),
    ),
    SpeakPass(
        "thousands_number_de",
        js(r"(?<![\w.,])(-?\d{1,3}(?:\.\d{3})+|-?\d{1,3}(?:,\d{3}){2,})(?![.,]?\d)", "g"),
        lambda _m, number_token, *_r: speak_german_numeric_token(
            number_token.replace(",", ".") if "," in number_token else number_token
        ),
    ),
    SpeakPass(
        "digit_sequence_de",
        js(
            r"\b(0\d{2,}|(?<=\b(?:\d+|\p{Lu})[ \t]+|(?:\.\.\.|…)[ \t]*)0\d(?![.,:]?\d)|(?<!\d[.,:])0\d(?=[ \t]+\d)|\d{5,})\b",
            "gu",
        ),
        lambda _m, number_token, *_r: spell_german_digits(number_token),
    ),
    SpeakPass(
        "integer_number_de_sentence_final",
        GERMAN_SENTENCE_FINAL_INTEGER_PATTERN,
        lambda _m, prefix, adjective, after_label, sign, number_token, chain_joiner, second_token, follower, *_r: (
            speak_german_sentence_final_integer(
                prefix, adjective, after_label, sign, number_token, chain_joiner, second_token, follower
            )
        ),
        lambda _m, prefix, _adjective, after_label, sign, _number, _joiner, _second, follower, *_r: is_deferred_german_ordinal(
            prefix, after_label, sign, follower
        ),
    ),
    SpeakPass(
        "year_number_de",
        GERMAN_BARE_YEAR_PATTERN,
        lambda _m, number_token, *_r: year_to_german_words(int(number_token)),
    ),
    SpeakPass(
        "integer_number_de",
        js(
            r"(?<!\w)(?<!(?<!\.\.)\.)(?!(?=1[^\S\r\n])(?<!\b\p{Lu}+[ \t]+)(?!(?<=\b" + GERMAN_NUMBER_LABEL_SOURCE + r"[ \t]+)1[^\S\r\n]+(?:"
            + GERMAN_NUMBER_LABEL_SOURCE + r"(?!\p{L})|\p{Lu}(?:\p{Ll}?\p{Lu})*\b))1[^\S\r\n]+(?:\p{Lu}|(?!(?:von|bis|zu|in|mit|am|um"
            r"|bei|nach|vor|aus|gegen|oder|und|für|auf|an|über|unter|ohne|pro|je|als|wie|im|ins|ans|beim|zum|zur|vom|des|der|die|das"
            r"|den|dem|ist|war|hat|wird|sind)\b)\p{Ll}+[^\S\r\n]+\p{Lu}))(-?\d+)\b(?:(?!\.)|(?=\.\.\.)|(?<=\d{4})(?!\.\d))",
            "gu",
        ),
        lambda _m, number_token, *_r: speak_german_numeric_token(number_token),
    ),
]

GERMAN_SPEAK_PASS_BY_LABEL = {speak_pass.label: speak_pass for speak_pass in GERMAN_SPEAK_PASSES}
# The number passes after name_number_de that would otherwise read a name's number the floor keeps.
_GERMAN_NAME_NUMBER_GUARDED_PASSES = {
    "decimal_number_de", "thousands_number_de", "digit_sequence_de", "integer_number_de_sentence_final", "year_number_de",
    "integer_number_de",
}

GERMAN_NOTATION_PREPASS = NotationPrepassConfig(
    lang="de",
    integer_words=integer_to_german_words,
    digit_words=spell_german_digits,
    year_words=year_to_german_words,
    unit_symbol_source=GERMAN_UNIT_KEY_SOURCE,
    unit_symbols=tuple(GERMAN_UNIT_WORDS),
    unit_words=frozenset(word for unit in GERMAN_UNIT_WORDS.values() for word in (unit[0], unit[1])),
    number_label_source=GERMAN_NUMBER_LABEL_SOURCE,
    address_label=speak_german_address_label,
)

_FILENAME_LIST_ITEM = js(r"^(\s*)(\d{1,2})\.\s+(.+?\.[A-Za-z0-9]{2,8})([ \t]+[,.:])?[ \t]*$", "gm")


def optimize_german_tts_text(text: str, options: Optional[TtsOptimizeOptions] = None) -> TtsAlgorithmOptimizationResult:
    """``optimizeGermanTtsText``. ``stage='floor'`` defers bare years, inflected ordinals, names'
    numbers and unclear day.month pairs to the optimizer model."""
    opts = options or TtsOptimizeOptions()
    result = text or ""
    applied_rules: list[str] = []
    defer_to_model = opts.stage == "floor"
    if not js_trim(result):
        return TtsAlgorithmOptimizationResult(result, applied_rules, "de")

    def run_speak_pass(label: str) -> None:
        nonlocal result
        speak_pass = GERMAN_SPEAK_PASS_BY_LABEL.get(label)
        if speak_pass is None:
            return

        def callback(*args):
            match = args[0]
            groups = args[1:-2]
            if defer_to_model and (
                label == "year_number_de"
                or (label == "integer_number_de" and is_bare_german_year_at(args[-1], args[-2], match))
            ):
                return match
            if defer_to_model and speak_pass.defer_in_floor is not None and speak_pass.defer_in_floor(match, *groups):
                return match
            if (
                defer_to_model
                and label in _GERMAN_NAME_NUMBER_GUARDED_PASSES
                and is_german_name_number_at(args[-1], args[-2], match)
            ):
                return match
            spoken = speak_pass.speak(match, *groups)
            return match if spoken is None else spoken

        result = replace_with_tracking(result, speak_pass.pattern, callback, speak_pass.label, applied_rules)

    held_day_months: list[str] = []
    if defer_to_model and not opts.day_months_settled:
        day_month_pass = GERMAN_SPEAK_PASS_BY_LABEL["date_de_dm"]

        def hold(*args):
            match = args[0]
            if not is_ambiguous_german_day_month(args[1:-2]) or len(held_day_months) >= HELD_DAY_MONTH_LIMIT:
                return match
            held_day_months.append(match)
            return held_day_month_placeholder(len(held_day_months) - 1)

        result = js_replace(day_month_pass.pattern, result, hold)

    hold_list_opener = opts.hold_list_opener or LIST_OPENER_PAIR.test(result)

    result = apply_notation_normalization(result, GERMAN_NOTATION_PREPASS, applied_rules)
    result = join_german_spaced_thousands(result, applied_rules)

    def filename_item(match, leading_whitespace, index_token, filename, item_mark, *_rest):
        index = int(index_token)
        if index < 1 or index > 99 or holds_dot_name_before_extension(filename):
            return match
        return (
            f"{leading_whitespace}{capitalize_german_token(german_list_ordinal(index))}: "
            f"{speak_german_filename_phrase(filename)}{item_mark or ''}"
        )

    result = replace_with_tracking(result, _FILENAME_LIST_ITEM, filename_item, "filename_list_item_de", applied_rules)
    result = normalize_german_url_ports(result, applied_rules)
    result = resolve_german_year_hint_leftovers(result, applied_rules)
    result = resolve_german_day_month_hint_leftovers(result, applied_rules)
    run_speak_pass("domain_de")
    result = apply_notation_prepass(result, GERMAN_NOTATION_PREPASS, applied_rules, opts.answer_text)
    for label in (
        "network_port_de", "ipv4_de", "version_de", "date_de_slash_full", "date_de_slash_day_month", "date_de_month_year",
        "fraction_de", "phone_extension_de", "phone_number_de", "date_de_day_list", "date_de_day_range",
        "date_de_named_month", "date_de_named_month_no_year", "date_de_dmy", "date_de_dmy_short_year", "date_de_dm_padded",
        "date_de_dm_number_period", "date_de_dm", "date_iso", "date_de_year_month", "duration_de_compact",
        "duration_de_unit", "duration_de_context", "time_de_hhmm_dot", "time_de_hhmm_dot_context", "time_de_hhmm",
        "time_de_hour_only",
    ):
        run_speak_pass(label)

    result = replace_ordered_sequence_markers(
        result,
        lambda index: capitalize_german_token(german_list_ordinal(index)),
        "list_marker_de",
        applied_rules,
        keep_lone_marker=defer_to_model and not opts.continues_numbered_list,
        starts_inside_line=opts.starts_inside_line,
        hold_list_opener=hold_list_opener,
    )

    for label in (
        "amount_scale_de", "dimension_unit_de", "unit_suffix_de", "hex_color_de", "currency_standard_de",
        "currency_prefix", "currency_suffix", "currency_pair_de", "per_mille_de", "percent_de", "temperature_de",
        "dimension_de", "name_number_de", "decimal_number_de", "thousands_number_de", "digit_sequence_de",
        "integer_number_de_sentence_final", "year_number_de", "integer_number_de",
    ):
        run_speak_pass(label)

    if held_day_months:
        applied_rules.append(TTS_DAY_MONTH_HELD_RULE)
    keep_held = opts.keep_held_day_months
    if not keep_held:
        result = restore_held_day_months(result, held_day_months)
    return TtsAlgorithmOptimizationResult(
        finalize_optimized_text(result),
        applied_rules,
        "de",
        held_day_months if keep_held and held_day_months else None,
    )
