"""Port of Genesis3 ``src/utils/ttsLanguageOptimizer/de.ts`` (German replace passes).

Pass labels, order and spoken forms follow the TS source one to one so the
golden outputs of the TS implementation can be compared directly. The ``hint``
callbacks of the TS passes are not ported (they only feed the hint injector).
"""

from __future__ import annotations

from typing import Optional

from .jsre import js, js_number, js_replace, js_trim
from .notation_prepass import NotationPrepassConfig, apply_notation_prepass, escape_for_regexp
from .shared import (
    PHONE_NUMBER_PATTERN_SOURCE,
    SpeakPass,
    TtsAlgorithmOptimizationResult,
    finalize_optimized_text,
    parse_phone_number_match,
    replace_ordered_sequence_markers,
    replace_with_tracking,
)

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
GERMAN_CURRENCY_MAP = {
    "EUR": _EURO, "EURO": _EURO, "€": _EURO,
    "USD": _DOLLAR, "DOLLAR": _DOLLAR, "$": _DOLLAR,
    "GBP": _POUND, "PFUND": _POUND, "£": _POUND,
}


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
    base = base_german_ordinal(day)
    return f"{base}n" if inflection == "oblique" else f"{base}r"


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


_GROUPED_THOUSANDS = js(r"^\d{1,3}(?:\.\d{3})+$", "")
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
        if len(parts) >= 2 and (normalized.startswith("0.") or len(parts[-1]) != 3):
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
    normalized_fraction = fraction_part.ljust(2, "0") if fraction_part else ""
    fraction_value = js_number(normalized_fraction) if normalized_fraction else 0
    prefix = "minus " if negative else ""
    if fraction_value > 0:
        sub_label = currency["subSingular"] if fraction_value == 1 else currency["subPlural"]
        sub_words = "ein" if fraction_value == 1 else integer_to_german_words(int(fraction_value))
        return f"{prefix}{integer_words} {unit_label} und {sub_words} {sub_label}"
    return js_trim(f"{prefix}{integer_words} {unit_label}")


def speak_german_percent(token: str) -> str:
    return f"{speak_german_attributive_numeric_token(token)} Prozent"


GERMAN_SCALE_WORDS = {
    "tsd": ("Tausend", "Tausend"),
    "mio": ("Million", "Millionen"),
    "mrd": ("Milliarde", "Milliarden"),
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
    if len(spacing) > 0:
        return True
    return unit_key in ("h", "s") and _SHORT_NUMBER.test(number_token)


def speak_german_amount_scale(number_token, scale_token, prefix_currency=None, suffix_currency=None):
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
    number_words = "eine" if is_exactly_one else speak_german_numeric_token(number_token)
    parts = [number_words, scale[0] if is_exactly_one else scale[1], currency["unitPlural"] if currency else ""]
    return " ".join(part for part in parts if part)


_EIN_AT_END = js(r"(^|\s)ein$", "")
_EIN_SUFFIX = js(r"ein$", "")


def speak_german_unit(number_token: str, spacing: str, unit_token: str) -> Optional[str]:
    key = js_replace(_TRAILING_DOT, unit_token or "", "")
    unit = GERMAN_UNIT_WORDS.get(key)
    if not unit or not is_german_unit_reading_plausible(number_token, spacing, key):
        return None
    singular, plural, feminine = unit
    words = speak_german_attributive_numeric_token(number_token)
    if _EIN_AT_END.test(words):
        return f"{js_replace(_EIN_SUFFIX, words, 'eine') if feminine else words} {singular}"
    return f"{words} {plural}"


def speak_german_version(prefix, parts) -> str:
    spoken = " Punkt ".join(integer_to_german_words(int(part)) for part in parts if isinstance(part, str) and part != "")
    return f"Version {spoken}" if prefix else spoken


def looks_like_german_dotted_date(first: str, second: str, third: str) -> bool:
    day = int(first)
    month = int(second)
    return 1 <= day <= 31 and 1 <= month <= 12 and len(third) in (2, 4)


def speak_german_ipv4(octets, port=None) -> Optional[str]:
    if any(int(octet) > 255 for octet in octets):
        return None
    spoken = " Punkt ".join(spell_german_digits(octet) for octet in octets)
    return f"{spoken}, Port {spell_german_digits(port)}" if port else spoken


_UND = js(r"und", "i")


def speak_german_day_range(prefix, day1_token, month1_token, joiner_token, day2_token,
                           month_numeric, year_numeric, month_name, year_named) -> Optional[str]:
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
    joiner = "und" if _UND.test(joiner_token) else "bis"
    year = year_numeric or year_named
    tail = f"{GERMAN_MONTHS[month2 - 1]}{' ' + year_to_german_words(int(year)) if year else ''}"
    if month1 == month2:
        first = german_ordinal_day(day1, inflection)
    else:
        first = f"{german_ordinal_day(day1, inflection)} {GERMAN_MONTHS[month1 - 1]}"
    days = f"{first} {joiner} {german_ordinal_day(day2, inflection)} {tail}"
    return f"{raw_prefix} {days}" if raw_prefix else days


GERMAN_SENTENCE_OPENER_SOURCE = (
    "(?:Ich|Du|Er|Sie|Es|Wir|Ihr|Das|Der|Die|Dann|Danach|Dabei|Dazu|Damit|Deshalb|Also|Aber|Und|Oder|Bitte|Danke|"
    "Jetzt|Heute|Hier|Dort|So|Wenn|Falls|Was|Wer|Wie|Wo|Wann|Warum|Kein|Keine|Mehr|Noch|Schon|Alles|Nichts|Ja|Nein|"
    "Okay|OK|Los|Gut|Super|Klar|Passt|Soll|Kann|Willst|Möchtest|Brauchst|Hast|Bist|Ist|Sind|Habe|Gibt|Lass|Viel|Alle|"
    "Diese|Dieser|Dieses|Am|Im|Um|Auf|In|Mit|Von|Zu|Nach|Vor|Bei|Für|Über|Unter|Seit|Bis|Ab|Aus|Als|Da|Doch|Nur|"
    "Außerdem|Zudem|Übrigens|Allerdings|Trotzdem|Eigentlich|Vielleicht|Sonst|Erst|Später|Gleich|Sobald|Solange|Sag|"
    "Melde|Schau|Denk|Vergiss|Mach|Geh|Komm|Ruf|Schreib|Sollte|Könnte|Würde|Wäre|Weitere|Weiter|Nimm|Gib|Zeig|"
    "Bring|Hol|Öffne|Schick|Prüf|Warte|Erinnere|Stell|Leg|Setz|Frag|Antworte|Dein|Deine|Mein|Meine|Sein|Seine|Unser|"
    "Unsere|Euer|Eure|Man|Jemand|Niemand|Etwas|Denn|Weil|Obwohl|Sowohl|Weder|Entweder|Zum|Zur|Vom|Beim|Ins|Ans)"
)
GERMAN_ORDINAL_PREPOSITION_SOURCE = (
    "(?:[Ii]n|[Aa]n|[Aa]uf|[Bb]ei|[Mm]it|[Nn]ach|[Ss]eit|[Vv]on|[Zz]u|[Aa]us|[Vv]or|[Uu]nter|[Üü]ber|[Zz]wischen)"
)
GERMAN_ORDINAL_PREFIX_SOURCE = (
    f"(?:{GERMAN_ORDINAL_PREPOSITION_SOURCE}[ \\t]+)?(?:[Aa]m|[Bb]eim|[Ii]m|[Dd]em|[Dd]en|[Vv]om|[Zz]um|[Zz]ur|[Bb]is|[Dd]er|[Dd]es|[Dd]ie|[Dd]as|"
    "[Jj]ede[nmrs]?|[Ii]hre?[nms]?|[Ss]eine?[nms]?|[Mm]eine?[nms]?|[Dd]eine?[nms]?|[Uu]nsere?[nms]?)"
)
GERMAN_SENTENCE_FINAL_INTEGER_PATTERN = js(
    rf"(?:(?<=\S[ \t]+)|(?<![^\n])(?={GERMAN_ORDINAL_PREFIX_SOURCE}[ \t]+\d))({GERMAN_ORDINAL_PREFIX_SOURCE}[ \t]+)?(\d{{1,3}})\.(?:[ \t]*(und|bis)[ \t]*(\d{{1,3}})\.)?(?=([ \t]*(?:$|\n|[\"”“)\]])|[ \t]+\S+))",
    "gu",
)
_TERMINAL_FOLLOWER = js(r"^[ \t]*(?:$|\n|[\"”“)\]])", "")
_OPENER_FOLLOWER = js(f"^[ \\t]+{GERMAN_SENTENCE_OPENER_SOURCE}\\b", "u")
_CAPITALISED_FOLLOWER = js(r"^[ \t]+\p{Lu}", "u")
_NOMINATIVE_ARTICLE = js(r"(?:^|\s)(?:der|die|das)$", "i")
_PREPOSITION_PREFIX = js(f"^{GERMAN_ORDINAL_PREPOSITION_SOURCE}\\s", "")


def is_german_sentence_final_follower(follower: str) -> bool:
    return _TERMINAL_FOLLOWER.test(follower) or _OPENER_FOLLOWER.test(follower)


def is_deferred_german_ordinal(prefix, follower) -> bool:
    return bool(js_trim(prefix or "")) and not is_german_sentence_final_follower(follower)


def speak_german_sentence_final_integer(prefix, number_token, chain_joiner, second_token, follower) -> Optional[str]:
    value = int(number_token)
    terminal = is_german_sentence_final_follower(follower)
    capitalised_follower = _CAPITALISED_FOLLOWER.test(follower)
    raw_prefix = js_trim(prefix or "")
    if raw_prefix:
        nominative = _NOMINATIVE_ARTICLE.test(raw_prefix) and not _PREPOSITION_PREFIX.test(raw_prefix)

        def inflect(day: int) -> str:
            return base_german_ordinal(day) if nominative else german_ordinal_day(day, "oblique")

        if second_token is not None:
            second = int(second_token)
            if value < 1 or second < 1 or value > 999 or second > 999:
                return None
            return f"{raw_prefix} {inflect(value)} {_js_str(chain_joiner)} {inflect(second)}{'.' if terminal else ''}"
        if value >= 1 and (value <= 31 or (not terminal and capitalised_follower)):
            return f"{raw_prefix} {inflect(value)}{'.' if terminal else ''}"
        if terminal:
            return f"{raw_prefix} {integer_to_german_words(value)}."
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


# Mojibake 'fÃ¼nf' is verbatim from de.ts (GERMAN_DIGIT_WORD_SOURCE) and kept for parity: the
# dedupe of an already spoken URL port therefore misses ports containing a 5.
GERMAN_DIGIT_WORD_SOURCE = "(?:null|eins|zwei|drei|vier|fÃ¼nf|sechs|sieben|acht|neun)"
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


def speak_german_domain(label_part: str, tld: str) -> Optional[str]:
    labels = [part for part in (js_trim(p) for p in (label_part or "").split(".")) if part]
    if not labels:
        return None
    return f"{' punkt '.join(labels)} punkt {(tld or '').lower()}"


def speak_german_time(hours: int, minutes: int, seconds: Optional[int] = None) -> str:
    hour_words = "ein" if hours == 1 else integer_to_german_words(hours)
    if seconds is not None:
        return f"{hour_words} Uhr {integer_to_german_words(minutes)} und {integer_to_german_words(seconds)} Sekunden"
    if minutes == 0:
        return f"{hour_words} Uhr"
    return f"{hour_words} Uhr {integer_to_german_words(minutes)}"


def speak_german_duration(hours, minutes, seconds, zero_form: str) -> str:
    def count(value: int, singular: str, plural: str) -> str:
        return f"eine {singular}" if value == 1 else f"{integer_to_german_words(value)} {plural}"

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
    inflection = "oblique" if raw_prefix else "nominative"
    day_label = german_ordinal_day(day, inflection)
    if raw_prefix:
        return f"{raw_prefix} {day_label} {month_label} {year_to_german_words(year)}"
    return f"{day_label} {month_label} {year_to_german_words(year)}"


GERMAN_BARE_YEAR_SOURCE = r"(?<![\w.\-])(?<!\bmal[ \t]+)(1[1-9]\d{2})(?![\p{L}\p{N}_]|[.,]\d|[ \t]+mal[ \t]+\d)"
GERMAN_BARE_YEAR_PATTERN = js(GERMAN_BARE_YEAR_SOURCE, "gu")
GERMAN_BARE_YEAR_AT = js(GERMAN_BARE_YEAR_SOURCE, "uy")


def is_bare_german_year_at(text: str, offset: int, match: str) -> bool:
    m = GERMAN_BARE_YEAR_AT.match_at(text, offset)
    return bool(m and m.start() == offset and m.group(0) == match)


# --- speak callbacks -------------------------------------------------------------------------------

def _speak_version(_match, short_prefix, short_first, short_second, prefix, first, second, third, fourth):
    if short_first is not None:
        return speak_german_version(short_prefix, [short_first, short_second])
    if not fourth and looks_like_german_dotted_date(first, second, third):
        return None
    if all(len(part) == 3 for part in (second, third, fourth) if part):
        return None
    return speak_german_version(prefix, [first, second, third, fourth])


def _speak_slash_full(_match, prefix, day_token, month_token, year_token):
    day, month, year = int(day_token), int(month_token), int(year_token)
    if day < 1 or day > 31 or month < 1 or month > 12:
        return None
    return speak_german_date(day, month, year, prefix) or None


def _speak_month_year(_match, month_token, year_token):
    month, year = int(month_token), int(year_token)
    if month < 1 or month > 12:
        return None
    return f"{GERMAN_MONTHS[month - 1]} {year_to_german_words(year)}"


def _speak_fraction(_match, numerator_token, denominator_token):
    numerator, denominator = int(numerator_token), int(denominator_token)
    label = GERMAN_FRACTION_DENOMINATORS.get(denominator)
    if not label or numerator >= denominator:
        return None
    numerator_words = "ein" if numerator == 1 else integer_to_german_words(numerator)
    return f"{numerator_words} {label}"


def _speak_named_month(_match, prefix, day_token, month_token, year_token):
    day = int(day_token)
    month = german_month_index(month_token)
    year = parse_spoken_german_year_token(year_token)
    if not month or year is None or day < 1 or day > 31:
        return None
    return speak_german_date(day, month, year, prefix) or None


def _speak_named_month_no_year(_match, prefix, day_token, month_token):
    day = int(day_token)
    month = german_month_index(month_token)
    if not month or day < 1 or day > 31:
        return None
    month_label = _month_label(month)
    if not month_label:
        return None
    raw_prefix = js_trim(prefix or "")
    day_label = german_ordinal_day(day, "oblique" if raw_prefix else "nominative")
    return f"{raw_prefix} {day_label} {month_label}" if raw_prefix else f"{day_label} {month_label}"


def _speak_dmy(_match, prefix, day_token, month_token, year_token):
    day, month, year = int(day_token), int(month_token), int(year_token)
    if day < 1 or day > 31 or month < 1 or month > 12:
        return None
    return speak_german_date(day, month, year, prefix) or None


def _speak_dmy_short_year(_match, prefix, day_token, month_token, year_token):
    day, month = int(day_token), int(month_token)
    year = parse_spoken_german_year_token(year_token)
    if year is None or day < 1 or day > 31 or month < 1 or month > 12:
        return None
    return speak_german_date(day, month, year, prefix) or None


def _speak_dm(_match, prefix, day_token, month_token):
    day, month = int(day_token), int(month_token)
    if day < 1 or day > 31 or month < 1 or month > 12:
        return None
    month_label = _month_label(month)
    if not month_label:
        return None
    normalized_prefix = js_trim(prefix or "").lower()
    day_label = german_ordinal_day(day, "oblique" if normalized_prefix else "nominative")
    return f"{normalized_prefix} {day_label} {month_label}" if normalized_prefix else f"{day_label} {month_label}"


def _speak_iso(_match, year_token, month_token, day_token, hours_token, minutes_token, *_seconds):
    day, month, year = int(day_token), int(month_token), int(year_token)
    if day < 1 or day > 31 or month < 1 or month > 12:
        return None
    spoken_date = speak_german_date(day, month, year, None)
    if not spoken_date:
        return None
    if hours_token is not None:
        hours, minutes = int(hours_token), int(minutes_token)
        if hours <= 23 and minutes <= 59:
            return f"{spoken_date} um {speak_german_time(hours, minutes)}"
    return spoken_date


def _speak_year_month(_match, year_token, month_token):
    return f"{GERMAN_MONTHS[int(month_token) - 1]} {year_to_german_words(int(year_token))}"


def _speak_duration_compact(_match, days_token, hours_token, minutes_token, seconds_token, only_minutes, only_seconds):
    if only_minutes is not None:
        return speak_german_duration(None, int(only_minutes), int(only_seconds), "null Sekunden")
    minutes = int(minutes_token)
    seconds = None if seconds_token is None else int(seconds_token)
    if minutes > 59 or (seconds is not None and seconds > 59):
        return None
    span = speak_german_duration(int(hours_token), minutes, seconds, "null Minuten")
    if days_token is None:
        return span
    days = int(days_token)
    return f"{'ein Tag' if days == 1 else integer_to_german_words(days) + ' Tage'}, {span}"


_HOUR_UNIT = js(r"^(?:st|h)", "i")


def _speak_duration_unit(_match, first_token, second_token, third_token, unit_token):
    first, second = int(first_token), int(second_token)
    third = None if third_token is None else int(third_token)
    if second > 59 or (third is not None and third > 59):
        return None
    if third is not None:
        return speak_german_duration(first, second, third, "null Sekunden")
    if _HOUR_UNIT.test(unit_token or ""):
        return speak_german_duration(first, second, None, "null Stunden")
    return speak_german_duration(None, first, second, "null Minuten")


def _speak_hhmm_dot(_match, hours_token, minutes_token):
    hours, minutes = int(hours_token), int(minutes_token)
    if hours > 23 or minutes > 59:
        return None
    return speak_german_time(hours, minutes)


def _speak_hhmm(_match, hours_token, minutes_token, seconds_token, *_uhr):
    hours, minutes = int(hours_token), int(minutes_token)
    seconds = None if seconds_token is None else int(seconds_token)
    if hours < 0 or hours > 23 or minutes < 0 or minutes > 59 or (seconds is not None and (seconds < 0 or seconds > 59)):
        return None
    return speak_german_time(hours, minutes, seconds)


def _speak_hour_only(_match, hours_token):
    hours = int(hours_token)
    if hours < 0 or hours > 23:
        return None
    return speak_german_time(hours, 0)


def _speak_currency_prefix(_match, sign, currency_token, amount_token):
    negative = "-" if (sign or amount_token.startswith("-")) else ""
    unsigned = amount_token[1:] if amount_token.startswith("-") else amount_token
    return speak_german_currency(f"{negative}{unsigned}", currency_token)


_X_SPLIT = js(r"\s*[x×]\s*", "i")


def _speak_dimension(_match, chain):
    return " mal ".join(integer_to_german_words(int(token)) for token in _X_SPLIT.split(chain or ""))


GERMAN_SPEAK_PASSES: list[SpeakPass] = [
    SpeakPass(
        "domain_de",
        js(r"(?<![\w@/.-])((?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+)(de|com|org|net|eu|at|ch|io|dev|info|news)(?![\w-])", "gi"),
        lambda _m, label_part, tld: speak_german_domain(label_part, tld),
    ),
    SpeakPass(
        "network_port_de",
        js(r"\b(port(?:nummer)?)(?:\s*[:=]\s*|\s+)(\d{1,5})\b", "gi"),
        lambda _m, label, number_token: speak_german_network_port(label, number_token),
    ),
    SpeakPass(
        "ipv4_de",
        js(r"(?<![\w.])(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})(?::(\d{1,5}))?(?!\w|\.\d)", "g"),
        lambda _m, a, b, c, d, port: speak_german_ipv4([a, b, c, d], port),
    ),
    SpeakPass(
        "version_de",
        js(r"(?<![\w.,-])(?:(v)(\d{1,3})\.(\d{1,3})(?!\d|\.\d)|(v)?(\d{1,3})\.(\d{1,3})\.(\d{1,4})(?:\.(\d{1,4}))?(?!\d|[.,]\d))", "gi"),
        _speak_version,
    ),
    SpeakPass(
        "date_de_slash_full",
        js(r"(?<![\d.,/])((?:am|dem|den|vom|zum)\s+(?:[A-ZÄÖÜ][a-zäöüß]{2,},\s*)?)?(\d{1,2})\/(\d{1,2})\/((?:19|20)\d{2})(?![\d/])", "gi"),
        _speak_slash_full,
    ),
    SpeakPass(
        "date_de_month_year",
        js(r"(?<![\d.,/])(0?[1-9]|1[0-2])\/((?:19|20)\d{2})(?![\d/])", "g"),
        _speak_month_year,
    ),
    SpeakPass(
        "fraction_de",
        js(r"(?<![\d.,/])([1-9]\d?)\/(100|1[0-2]|[2-9])(?![\d/])(?![.,]\d)", "g"),
        _speak_fraction,
    ),
    SpeakPass(
        "phone_number_de",
        js(PHONE_NUMBER_PATTERN_SOURCE, "gu"),
        lambda match: speak_german_phone_number(match),
    ),
    SpeakPass(
        "date_de_day_range",
        js(
            rf"(?<![\d.])\b((?:am|dem|den|vom|zum|bis|zwischen)\s+(?:dem\s+)?)?(\d{{1,2}})\.(?:\s?(\d{{1,2}})\.)?\s*(bis|und|[–—-])\s*(?:zum\s+|dem\s+)?(\d{{1,2}})\.\s*(?:(\d{{1,2}})\.(?:(\d{{4}})\b|(?!\d))|({GERMAN_MONTH_SOURCE})\.?(?![\p{{L}}])(?:\s+((?:19|20)\d{{2}})\b)?)",
            "giu",
        ),
        lambda _m, *groups: speak_german_day_range(*groups),
    ),
    SpeakPass(
        "date_de_named_month",
        js(
            rf"\b((?:am|dem|den|vom|zum)\s+(?:[A-ZÄÖÜ][a-zäöüß]{{2,}},\s*)?)?(\d{{1,2}})\.\s*({GERMAN_MONTH_SOURCE})\.?\s+(\d{{2,4}})\b(?!:)",
            "gi",
        ),
        _speak_named_month,
    ),
    SpeakPass(
        "date_de_named_month_no_year",
        js(
            rf"\b((?:am|dem|den|vom|zum|bis)\s+(?:[A-ZÄÖÜ][a-zäöüß]{{2,}},\s*)?)?(\d{{1,2}})\.\s*({GERMAN_MONTH_SOURCE})\.?(?![\p{{L}}])(?!\s*,?\s*\d{{2,4}}\b(?!:))",
            "giu",
        ),
        _speak_named_month_no_year,
    ),
    SpeakPass(
        "date_de_dmy",
        js(r"\b((?:am|dem|den|vom|zum)\s+(?:[A-ZÄÖÜ][a-zäöüß]{2,},\s*)?)?(\d{1,2})\.(\d{1,2})\.(\d{4})\b", "gi"),
        _speak_dmy,
    ),
    SpeakPass(
        "date_de_dmy_short_year",
        js(r"\b((?:am|dem|den|vom|zum)\s+(?:[A-ZÄÖÜ][a-zäöüß]{2,},\s*)?)?(\d{2})\.(\d{2})\.(\d{2})\b", "gi"),
        _speak_dmy_short_year,
    ),
    SpeakPass(
        "date_de_dm",
        js(r"\b((?:am|dem|den|vom|zum|bis)\s+(?:[A-ZÄÖÜ][a-zäöüß]{2,},\s*)?)?(\d{1,2})\.(\d{1,2})\.(?!\d)", "gi"),
        _speak_dm,
    ),
    SpeakPass(
        "date_iso",
        js(r"(?<![\p{L}\d])(\d{4})-(\d{2})-(\d{2})(?:T(\d{2}):(\d{2})(?::(\d{2}))?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?(?!\d)", "gu"),
        _speak_iso,
    ),
    SpeakPass(
        "date_de_year_month",
        js(r"(?<![\w.-])((?:19|20)\d{2})-(0[1-9]|1[0-2])(?![\w-])", "g"),
        _speak_year_month,
    ),
    SpeakPass(
        "duration_de_compact",
        js(
            r"\b(?:(\d{1,3})[ \t]?(?:d|Tage?)[ \t]?)?(\d{1,3})[ \t]?(?:h|Std(?:\.(?![ \t]*$|[ \t]+\p{Lu}))?|Stunden?)[ \t]?(?:und[ \t])?(\d{1,2})[ \t]?(?:m|min|Min(?:\.(?![ \t]*$|[ \t]+\p{Lu}))?|Minuten?)(?![\p{L}])(?:[ \t]?(?:und[ \t])?(\d{1,2})[ \t]?(?:s|sec|Sek(?:\.(?![ \t]*$|[ \t]+\p{Lu}))?|Sekunden?)(?![\p{L}]))?|\b(\d{1,3})[ \t]?m[ \t]?(\d{1,2})[ \t]?s(?![\p{L}])",
            "gu",
        ),
        _speak_duration_compact,
    ),
    SpeakPass(
        "duration_de_unit",
        js(
            r"\b(\d{1,3}):(\d{2})(?::(\d{2}))?\s*([Mm]inuten?|[Mm]in(?:\.(?!\s*$|\s+\p{Lu}))?|MIN|[Ss]tunden?|[Ss]td(?:\.(?!\s*$|\s+\p{Lu}))?|STD|h)(?![\p{L}])",
            "gu",
        ),
        _speak_duration_unit,
    ),
    SpeakPass(
        "time_de_hhmm_dot",
        js(r"\b(\d{1,2})\.(\d{2})(?:[ \t]*Uhr\b|(?=[ \t]*(?:bis|und|[–—-])[ \t]*\d{1,2}\.\d{2}[ \t]*Uhr\b))", "gi"),
        _speak_hhmm_dot,
    ),
    SpeakPass(
        "time_de_hhmm",
        js(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?(?:\s*(uhr))?\b", "gi"),
        _speak_hhmm,
    ),
    SpeakPass(
        "time_de_hour_only",
        js(r"\b(\d{1,2})\s*uhr\b", "gi"),
        _speak_hour_only,
    ),
    SpeakPass(
        "amount_scale_de",
        js(
            r"(?<!\w)(?:(€|\$|£|(?:EURO?|USD|DOLLAR|GBP|PFUND)\b)\s*)?(-?\d[\d.,]*\d|-?\d)\s*(Mio|Mrd|Tsd)\b\.?(?:\s*(€|\$|£|(?:EURO?|USD|DOLLAR|GBP|PFUND)\b))?",
            "gi",
        ),
        lambda _m, prefix_currency, number_token, scale_token, suffix_currency: speak_german_amount_scale(
            number_token, scale_token, prefix_currency, suffix_currency
        ),
    ),
    SpeakPass(
        "unit_suffix_de",
        js(rf"(?<!\w)(-?\d[\d.,]*)([ \t]*)({GERMAN_UNIT_KEY_SOURCE})(?:\.(?![ \t]*$|[ \t]+\p{{Lu}}))?(?![\p{{L}}\d])", "gu"),
        lambda _m, number_token, spacing, unit_token: speak_german_unit(number_token, spacing, unit_token),
    ),
    SpeakPass(
        "hex_color_de",
        js(r"#([0-9a-fA-F]{6}|[0-9a-fA-F]{8})\b", "g"),
        lambda _m, hex_token: spell_german_hex(hex_token),
    ),
    SpeakPass(
        "currency_prefix",
        js(r"(?<!\w)(-?)(€|\$|£|(?:EURO?|USD|DOLLAR|GBP|PFUND)\b)\s*(-?\d(?:[\d.,]*\d)?)", "gi"),
        _speak_currency_prefix,
    ),
    SpeakPass(
        "currency_suffix",
        js(
            r"(?<!\w)(-?\d(?:[\d.,]*\d)?)\s*(€|\$|£|(?:EURO?|USD|DOLLAR|GBP|PFUND)\b)(?=$|[\s,.;:!?)]|\/(?!(?:EUR|USD|GBP|CHF|JPY)\b))",
            "gi",
        ),
        lambda _m, amount_token, currency_token: speak_german_currency(amount_token, currency_token),
    ),
    SpeakPass(
        "percent_de",
        js(r"(?<!\w)(-?\d[\d.,]*)\s*%", "g"),
        lambda _m, number_token: speak_german_percent(number_token),
    ),
    SpeakPass(
        "temperature_de",
        js(r"(?<!\w)(-?\d[\d.,]*)\s*(?:°|º)\s*([CFK])\b", "gi"),
        lambda _m, number_token, unit_token: speak_german_temperature(number_token, (unit_token or "").upper()),
    ),
    SpeakPass(
        "dimension_de",
        js(r"(?<![\w.,])([1-9]\d{0,4}(?:\s*[x×]\s*\d{1,5})+)(?!\w|[.,]\d)", "gi"),
        _speak_dimension,
    ),
    SpeakPass(
        "decimal_number_de",
        js(r"(?<![\w.,])(-?\d{1,3}(?:\.\d{3})+,\d+|-?\d+,\d+|-?\d+\.\d+)(?![.,]?\d)", "g"),
        lambda _m, number_token: speak_german_numeric_token(number_token),
    ),
    SpeakPass(
        "thousands_number_de",
        js(r"(?<![\w.,])(-?\d{1,3}(?:\.\d{3})+|-?\d{1,3}(?:,\d{3}){2,})(?![.,]?\d)", "g"),
        lambda _m, number_token: speak_german_numeric_token(
            number_token.replace(",", ".") if "," in number_token else number_token
        ),
    ),
    SpeakPass(
        "digit_sequence_de",
        js(r"\b(0\d{2,}|\d{5,})\b", "g"),
        lambda _m, number_token: spell_german_digits(number_token),
    ),
    SpeakPass(
        "integer_number_de_sentence_final",
        GERMAN_SENTENCE_FINAL_INTEGER_PATTERN,
        lambda _m, prefix, number_token, chain_joiner, second_token, follower: speak_german_sentence_final_integer(
            prefix, number_token, chain_joiner, second_token, follower
        ),
        lambda _m, prefix, _number, _joiner, _second, follower: is_deferred_german_ordinal(prefix, follower),
    ),
    SpeakPass(
        "year_number_de",
        GERMAN_BARE_YEAR_PATTERN,
        lambda _m, number_token: year_to_german_words(int(number_token)),
    ),
    SpeakPass(
        "integer_number_de",
        js(
            r"(?<![\w.])(?!1[^\S\r\n]+(?:\p{Lu}|(?!(?:von|bis|zu|in|mit|am|um|bei|nach|vor|aus|gegen|oder|und|für|auf|an|über|unter|ohne|pro|je|als|wie|im|ins|ans|beim|zum|zur|vom|des|der|die|das|den|dem|ist|war|hat|wird|sind)\b)\p{Ll}+[^\S\r\n]+\p{Lu}))(-?\d+)\b(?:(?!\.)|(?<=\d{4})(?!\.\d))",
            "gu",
        ),
        lambda _m, number_token: speak_german_numeric_token(number_token),
    ),
]

GERMAN_SPEAK_PASS_BY_LABEL = {speak_pass.label: speak_pass for speak_pass in GERMAN_SPEAK_PASSES}

_FILENAME_LIST_ITEM = js(r"^(\s*)(\d{1,2})\.\s+(.+?\.[A-Za-z0-9]{2,8})\s*$", "gm")

GERMAN_NOTATION_PREPASS = NotationPrepassConfig(
    lang="de",
    integer_words=integer_to_german_words,
    digit_words=spell_german_digits,
    year_words=year_to_german_words,
)


def optimize_german_tts_text(text: str, stage: str = "final") -> TtsAlgorithmOptimizationResult:
    """``optimizeGermanTtsText``. ``stage='floor'`` defers bare years and inflected ordinals."""
    result = text or ""
    applied_rules: list[str] = []
    defer_to_model = stage == "floor"
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
            spoken = speak_pass.speak(match, *groups)
            return match if spoken is None else spoken

        result = replace_with_tracking(result, speak_pass.pattern, callback, speak_pass.label, applied_rules)

    def filename_item(match, leading_whitespace, index_token, filename, *_rest):
        index = int(index_token)
        if index < 1 or index > 99:
            return match
        return f"{leading_whitespace}{capitalize_german_token(german_list_ordinal(index))}: {speak_german_filename_phrase(filename)}"

    result = replace_with_tracking(result, _FILENAME_LIST_ITEM, filename_item, "filename_list_item_de", applied_rules)
    result = normalize_german_url_ports(result, applied_rules)
    result = resolve_german_year_hint_leftovers(result, applied_rules)
    result = apply_notation_prepass(result, GERMAN_NOTATION_PREPASS, applied_rules)
    for label in (
        "domain_de", "network_port_de", "ipv4_de", "version_de", "date_de_slash_full", "date_de_month_year",
        "fraction_de", "phone_number_de", "date_de_day_range", "date_de_named_month", "date_de_named_month_no_year",
        "date_de_dmy", "date_de_dmy_short_year", "date_de_dm", "date_iso", "date_de_year_month",
        "duration_de_compact", "duration_de_unit", "time_de_hhmm_dot", "time_de_hhmm", "time_de_hour_only",
    ):
        run_speak_pass(label)

    result = replace_ordered_sequence_markers(
        result,
        lambda index: capitalize_german_token(german_list_ordinal(index)),
        "list_marker_de",
        applied_rules,
        keep_lone_marker=defer_to_model,
    )

    for label in (
        "amount_scale_de", "unit_suffix_de", "hex_color_de", "currency_prefix", "currency_suffix", "percent_de",
        "temperature_de", "dimension_de", "decimal_number_de", "thousands_number_de", "digit_sequence_de",
        "integer_number_de_sentence_final", "year_number_de", "integer_number_de",
    ):
        run_speak_pass(label)

    return TtsAlgorithmOptimizationResult(finalize_optimized_text(result), applied_rules, "de")
