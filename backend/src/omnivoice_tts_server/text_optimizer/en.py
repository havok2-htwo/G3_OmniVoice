"""Port of Genesis3 ``src/utils/ttsLanguageOptimizer/en.ts`` (English replace passes).

English has no floor/final distinction (index.ts calls it without options).
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

ENGLISH_DIGITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
ENGLISH_SMALL = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
]
ENGLISH_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
ENGLISH_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}
ENGLISH_FRACTION_DENOMINATORS = {
    2: ("half", "halves"), 3: ("third", "thirds"), 4: ("quarter", "quarters"), 5: ("fifth", "fifths"),
    6: ("sixth", "sixths"), 7: ("seventh", "sevenths"), 8: ("eighth", "eighths"), 9: ("ninth", "ninths"),
    10: ("tenth", "tenths"), 11: ("eleventh", "elevenths"), 12: ("twelfth", "twelfths"),
    100: ("hundredth", "hundredths"),
}
ENGLISH_MONTH_SOURCE = (
    "(?:January|February|March|April|May|June|July|August|September|October|November|December"
    "|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec)"
)
ENGLISH_MONTH_LABELS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

_USD = {"unitSingular": "dollar", "unitPlural": "dollars", "subSingular": "cent", "subPlural": "cents"}
_EUR = {"unitSingular": "euro", "unitPlural": "euros", "subSingular": "cent", "subPlural": "cents"}
_GBP = {"unitSingular": "pound", "unitPlural": "pounds", "subSingular": "penny", "subPlural": "pence"}
ENGLISH_CURRENCY_MAP = {
    "USD": _USD, "$": _USD,
    "EUR": _EUR, "EURO": _EUR, "EUROS": _EUR, "€": _EUR,
    "GBP": _GBP, "£": _GBP,
}

_TRAILING_DOT = js(r"\.$", "")


def _month_label(month) -> Optional[str]:
    if isinstance(month, (int, float)) and month == month and float(month).is_integer() and 1 <= month <= 12:
        return ENGLISH_MONTH_LABELS[int(month) - 1]
    return None


def english_month_index(token: str) -> Optional[int]:
    key = js_replace(_TRAILING_DOT, (token or "").lower(), "")
    full = ENGLISH_MONTHS.get(key)
    if full:
        return full
    if len(key) < 3:
        return None
    for name, index in ENGLISH_MONTHS.items():
        if name.startswith(key[:3]):
            return index
    return None


def english_under_hundred(value: int) -> str:
    if value < 20:
        return ENGLISH_SMALL[value]
    tens, ones = divmod(value, 10)
    if ones == 0:
        return ENGLISH_TENS[tens]
    return f"{ENGLISH_TENS[tens]}-{ENGLISH_SMALL[ones]}"


def english_under_thousand(value: int) -> str:
    if value < 100:
        return english_under_hundred(value)
    hundreds, remainder = divmod(value, 100)
    hundred_part = f"{ENGLISH_SMALL[hundreds]} hundred"
    if remainder == 0:
        return hundred_part
    return f"{hundred_part} {english_under_hundred(remainder)}"


_ENGLISH_SCALES = (
    (1000000000000000, "quadrillion"),
    (1000000000000, "trillion"),
    (1000000000, "billion"),
    (1000000, "million"),
    (1000, "thousand"),
)


def integer_to_english_words(value: int) -> str:
    value = int(value)
    if value == 0:
        return "zero"
    if value < 0:
        return f"minus {integer_to_english_words(-value)}"
    remaining = value
    parts: list[str] = []
    for scale, label in _ENGLISH_SCALES:
        if remaining < scale:
            continue
        count, remaining = divmod(remaining, scale)
        parts.append(f"{integer_to_english_words(count)} {label}")
    if remaining > 0:
        if remaining < 1000:
            parts.append(english_under_thousand(remaining))
        else:
            parts.append(integer_to_english_words(remaining))
    return " ".join(parts)


def year_to_english_words(year: int) -> str:
    year = int(year)
    if 2000 <= year <= 2099:
        if year == 2000:
            return "two thousand"
        return f"two thousand {integer_to_english_words(year - 2000)}"
    if 1000 < year < 2000:
        first_part, second_part = divmod(year, 100)
        prefix = english_under_hundred(first_part)
        if second_part == 0:
            return f"{prefix} hundred"
        if second_part < 10:
            return f"{prefix} oh {english_under_hundred(second_part)}"
        return f"{prefix} {english_under_hundred(second_part)}"
    return integer_to_english_words(year)


_SPOKEN_VARIANT_SEPARATORS = js(r"[\s-]+", "g")


def normalize_english_spoken_variant(value: str) -> str:
    return js_replace(_SPOKEN_VARIANT_SEPARATORS, js_trim(value or ""), " ").lower()


_YEAR_HINT_LEFTOVER = js(
    r"(?<![\w.])(\d{4})\s*\(\s*([\p{L}][\p{L}\s-]*?)(?:\s*\/\s*([\p{L}][\p{L}\s-]*?))?\s*\)", "gu"
)


def resolve_english_year_hint_leftovers(text: str, applied_rules: list[str]) -> str:
    def callback(match, number_token, first_variant, second_variant, *_rest):
        value = int(number_token)
        readings = {
            normalize_english_spoken_variant(integer_to_english_words(value)),
            normalize_english_spoken_variant(year_to_english_words(value)),
        }
        if second_variant is not None:
            if (
                normalize_english_spoken_variant(first_variant) in readings
                and normalize_english_spoken_variant(second_variant) in readings
            ):
                return year_to_english_words(value)
            return match
        return js_trim(first_variant) if normalize_english_spoken_variant(first_variant) in readings else match

    return replace_with_tracking(text, _YEAR_HINT_LEFTOVER, callback, "year_hint_leftover_en", applied_rules)


_IRREGULAR_ORDINALS = {
    1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth",
    9: "ninth", 10: "tenth", 11: "eleventh", 12: "twelfth", 13: "thirteenth", 14: "fourteenth",
    15: "fifteenth", 16: "sixteenth", 17: "seventeenth", 18: "eighteenth", 19: "nineteenth",
}
_TRAILING_Y = js(r"y$", "")


def ordinal_english(value: int) -> str:
    irregular_value = _IRREGULAR_ORDINALS.get(value)
    if irregular_value:
        return irregular_value
    if value < 100:
        tens, ones = divmod(value, 10)
        if ones == 0:
            return f"{js_replace(_TRAILING_Y, ENGLISH_TENS[tens], 'i')}eth"
        return f"{ENGLISH_TENS[tens]}-{ordinal_english(ones)}"
    remainder = value % 100
    head = integer_to_english_words(value - remainder)
    if remainder == 0:
        return f"{head}th"
    return f"{head} {ordinal_english(remainder)}"


def capitalize_english_token(value: str) -> str:
    if not value:
        return value
    return f"{value[:1].upper()}{value[1:]}"


_YEAR_TOKEN = js(r"^\d{2,4}$", "")


def parse_english_year_token(token) -> Optional[int]:
    if not _YEAR_TOKEN.test(token if token is not None else ""):
        return None
    year = int(token)
    if len(token) == 2:
        return 1900 + year if year >= 50 else 2000 + year
    return year


_ALPHA_DIGIT = js(r"([A-Za-z])(\d)", "g")
_DIGIT_ALPHA = js(r"(\d)([A-Za-z])", "g")
_CAMEL = js(r"([a-z])([A-Z])", "g")


def split_english_alpha_numeric_boundaries(value: str) -> str:
    return js_replace(_DIGIT_ALPHA, js_replace(_ALPHA_DIGIT, value or "", "$1 $2"), "$1 $2")


def split_english_camel_case_boundaries(value: str) -> str:
    return js_replace(_CAMEL, value or "", "$1 $2")


_SPACE_BEFORE_EXTENSION = js(r"\s+\.(?=[A-Za-z0-9]{2,8}\b)", "g")
_EXTENSION = js(r"^(.*?)(?:\.([A-Za-z0-9]{2,8}))$", "")
_PATH_SEPARATORS = js(r"[\\/]+", "g")
_BRACKETS = js(r"[()[\]{}]+", "g")
_UNDERSCORES_DASHES = js(r"[_-]+", "g")
_DIGIT_DOT_DIGIT = js(r"(\d)\.(?=\d)", "g")


def speak_english_filename_phrase(value: str) -> str:
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
    result = js_replace(_DIGIT_DOT_DIGIT, result, "$1 ")
    result = split_english_camel_case_boundaries(result)
    result = split_english_alpha_numeric_boundaries(result)
    result = finalize_optimized_text(result)
    return f"{result} {' '.join(extension)}" if extension else result


def speak_english_temperature(token: str, unit: str) -> str:
    spoken_value = speak_english_numeric_token(token)
    if unit == "F":
        return f"{spoken_value} degrees Fahrenheit"
    if unit == "K":
        return f"{spoken_value} kelvin"
    return f"{spoken_value} degrees Celsius"


def spell_english_digits(token: str) -> str:
    return " ".join(ENGLISH_DIGITS[int(ch)] for ch in token if "0" <= ch <= "9")


_ALL_DIGITS = js(r"^\d+$", "")
_LEADING_ZEROS = js(r"^0+(?=\d)", "")


def parse_english_number_token(token):
    raw = js_trim(token or "")
    if not raw:
        return None
    negative = raw.startswith("-")
    normalized = raw[1:] if negative else raw
    if "." in normalized:
        parts = normalized.split(".")
        if len(parts) != 2:
            return None
        return negative, parts[0].replace(",", ""), parts[1]
    return negative, normalized.replace(",", ""), None


def _strip_leading_zeros(value: str) -> str:
    return js_replace(_LEADING_ZEROS, value, "") or "0"


def _valid_number(parsed) -> bool:
    return bool(parsed) and _ALL_DIGITS.test(parsed[1] or "0") and (parsed[2] is None or _ALL_DIGITS.test(parsed[2]))


def speak_english_numeric_token(token: str) -> str:
    parsed = parse_english_number_token(token)
    if not _valid_number(parsed):
        return token
    negative, integer_part, fraction_part = parsed
    integer_words = integer_to_english_words(int(_strip_leading_zeros(integer_part)))
    prefix = "minus " if negative else ""
    if fraction_part:
        return js_trim(f"{prefix}{integer_words} point {spell_english_digits(fraction_part)}")
    return js_trim(f"{prefix}{integer_words}")


def speak_english_currency(amount_token: str, currency_token: str) -> str:
    parsed = parse_english_number_token(amount_token)
    currency = ENGLISH_CURRENCY_MAP.get(js_trim(currency_token or "").upper())
    if not parsed or not currency or not _ALL_DIGITS.test(parsed[1] or "0"):
        return js_trim(f"{amount_token} {currency_token}")
    negative, integer_part, fraction_part = parsed
    integer_value = int(_strip_leading_zeros(integer_part))
    integer_words = integer_to_english_words(integer_value)
    unit_label = currency["unitSingular"] if integer_value == 1 else currency["unitPlural"]
    prefix = "minus " if negative else ""
    if fraction_part and len(fraction_part) > 2:
        return f"{speak_english_numeric_token(amount_token)} {currency['unitPlural']}"
    normalized_fraction = fraction_part.ljust(2, "0") if fraction_part else ""
    fraction_value = js_number(normalized_fraction) if normalized_fraction else 0
    if fraction_value > 0:
        sub_label = currency["subSingular"] if fraction_value == 1 else currency["subPlural"]
        return f"{prefix}{integer_words} {unit_label} and {integer_to_english_words(int(fraction_value))} {sub_label}"
    return js_trim(f"{prefix}{integer_words} {unit_label}")


ENGLISH_SCALE_WORDS = {"k": "thousand", "m": "million", "mn": "million", "b": "billion", "bn": "billion", "tn": "trillion"}


def _unit_word(singular: str, plural: Optional[str] = None):
    return (singular, f"{singular}s" if plural is None else plural)


ENGLISH_UNIT_WORDS = {
    "nm": _unit_word("nanometer"),
    "µm": _unit_word("micrometer"),
    "mm": _unit_word("millimeter"),
    "cm": _unit_word("centimeter"),
    "m": _unit_word("meter"),
    "km": _unit_word("kilometer"),
    "mi": _unit_word("mile"),
    "ft": _unit_word("foot", "feet"),
    "yd": _unit_word("yard"),
    "sq ft": _unit_word("square foot", "square feet"),
    "sq mi": _unit_word("square mile"),
    "m²": _unit_word("square meter"),
    "km²": _unit_word("square kilometer"),
    "m³": _unit_word("cubic meter"),
    "ha": _unit_word("hectare"),
    "µg": _unit_word("microgram"),
    "mg": _unit_word("milligram"),
    "g": _unit_word("gram"),
    "kg": _unit_word("kilogram"),
    "oz": _unit_word("ounce"),
    "lb": _unit_word("pound"),
    "lbs": _unit_word("pound"),
    "ml": _unit_word("milliliter"),
    "cl": _unit_word("centiliter"),
    "dl": _unit_word("deciliter"),
    "l": _unit_word("liter"),
    "gal": _unit_word("gallon"),
    "ms": _unit_word("millisecond"),
    "s": _unit_word("second"),
    "sec": _unit_word("second"),
    "secs": _unit_word("second"),
    "min": _unit_word("minute"),
    "mins": _unit_word("minute"),
    "h": _unit_word("hour"),
    "hr": _unit_word("hour"),
    "hrs": _unit_word("hour"),
    "W": _unit_word("watt"),
    "kW": _unit_word("kilowatt"),
    "MW": _unit_word("megawatt"),
    "Wh": _unit_word("watt-hour"),
    "kWh": _unit_word("kilowatt-hour"),
    "V": _unit_word("volt"),
    "mAh": _unit_word("milliampere-hour"),
    "dB": _unit_word("decibel"),
    "Hz": _unit_word("hertz", "hertz"),
    "kHz": _unit_word("kilohertz", "kilohertz"),
    "MHz": _unit_word("megahertz", "megahertz"),
    "GHz": _unit_word("gigahertz", "gigahertz"),
    "hPa": _unit_word("hectopascal"),
    "mmHg": _unit_word("millimeters of mercury", "millimeters of mercury"),
    "mbar": _unit_word("millibar"),
    "bar": _unit_word("bar", "bar"),
    "psi": _unit_word("P S I", "P S I"),
    "kcal": _unit_word("kilocalorie"),
    "kJ": _unit_word("kilojoule"),
    "kB": _unit_word("kilobyte"),
    "KB": _unit_word("kilobyte"),
    "MB": _unit_word("megabyte"),
    "GB": _unit_word("gigabyte"),
    "TB": _unit_word("terabyte"),
    "KiB": _unit_word("kibibyte"),
    "MiB": _unit_word("mebibyte"),
    "GiB": _unit_word("gibibyte"),
    "TiB": _unit_word("tebibyte"),
    "kbit": _unit_word("kilobit"),
    "Mbit": _unit_word("megabit"),
    "Gbit": _unit_word("gigabit"),
    "kbps": _unit_word("kilobit per second", "kilobits per second"),
    "Mbps": _unit_word("megabit per second", "megabits per second"),
    "Gbps": _unit_word("gigabit per second", "gigabits per second"),
    "fps": _unit_word("F P S", "F P S"),
    "bpm": _unit_word("beat per minute", "beats per minute"),
    "mph": _unit_word("mile per hour", "miles per hour"),
    "kn": _unit_word("knot"),
    "px": _unit_word("pixel"),
}
ENGLISH_UNIT_KEY_SOURCE = "|".join(
    escape_for_regexp(key) for key in sorted(ENGLISH_UNIT_WORDS.keys(), key=lambda key: -len(key))
)
_UPPER_ASCII = js(r"[A-Z]", "")
_SHORT_NUMBER = js(r"^\d{1,3}$", "")


def is_english_unit_reading_plausible(number_token: str, spacing: str, unit_key: str) -> bool:
    if len(unit_key) > 1 or _UPPER_ASCII.test(unit_key):
        return True
    if len(spacing) > 0:
        return True
    return unit_key in ("h", "s") and _SHORT_NUMBER.test(number_token)


def speak_english_duration(hours, minutes, seconds, zero_form: str) -> str:
    def count(value: int, singular: str, plural: str) -> str:
        return f"{integer_to_english_words(value)} {singular if value == 1 else plural}"

    parts: list[str] = []
    if hours is not None and hours > 0:
        parts.append(count(hours, "hour", "hours"))
    if minutes > 0:
        parts.append(count(minutes, "minute", "minutes"))
    if seconds is not None and seconds > 0:
        parts.append(count(seconds, "second", "seconds"))
    if not parts:
        return zero_form
    if len(parts) == 1:
        return parts[0]
    return f"{', '.join(parts[:-1])} and {parts[-1]}"


def speak_english_version(prefix, parts) -> str:
    spoken = " point ".join(integer_to_english_words(int(part)) for part in parts if isinstance(part, str) and part != "")
    return f"version {spoken}" if prefix else spoken


def speak_english_ipv4(octets, port=None) -> Optional[str]:
    if any(int(octet) > 255 for octet in octets):
        return None
    spoken = " dot ".join(spell_english_digits(octet) for octet in octets)
    return f"{spoken}, port {spell_english_digits(port)}" if port else spoken


def speak_english_amount_scale(number_token, scale_token, currency_token=None) -> Optional[str]:
    parsed = parse_english_number_token(number_token)
    if not _valid_number(parsed):
        return None
    scale_word = ENGLISH_SCALE_WORDS.get((scale_token or "").lower())
    if not scale_word:
        return None
    currency = ENGLISH_CURRENCY_MAP.get(js_trim(currency_token).upper()) if currency_token else None
    parts = [speak_english_numeric_token(number_token), scale_word, currency["unitPlural"] if currency else ""]
    return " ".join(part for part in parts if part)


def speak_english_unit(number_token: str, spacing: str, unit_token: str) -> Optional[str]:
    key = js_replace(_TRAILING_DOT, unit_token or "", "")
    unit = ENGLISH_UNIT_WORDS.get(key)
    if not unit or not is_english_unit_reading_plausible(number_token, spacing, key):
        return None
    parsed = parse_english_number_token(number_token)
    is_exactly_one = bool(parsed) and not parsed[0] and not parsed[2] and _strip_leading_zeros(parsed[1]) == "1"
    return f"{speak_english_numeric_token(number_token)} {unit[0] if is_exactly_one else unit[1]}"


def spell_english_hex(token: str) -> str:
    return " ".join(ENGLISH_DIGITS[int(ch)] if "0" <= ch <= "9" else ch.upper() for ch in token or "")


def speak_english_time(hours: int, minutes: int, seconds: Optional[int] = None, raw_meridiem: Optional[str] = None) -> str:
    if raw_meridiem:
        meridiem = raw_meridiem
    elif hours >= 13:
        meridiem = "PM"
    elif hours == 0:
        meridiem = "AM"
    else:
        meridiem = None
    suffix = f" {meridiem[0]} M" if meridiem else ""

    def minute_words() -> str:
        return f"oh {integer_to_english_words(minutes)}" if minutes < 10 else integer_to_english_words(minutes)

    if meridiem:
        hour12 = hours % 12 or 12
        hour_words = integer_to_english_words(hour12)
        if seconds is not None:
            return f"{hour_words} {minute_words()} and {integer_to_english_words(seconds)} seconds{suffix}"
        if minutes == 0:
            return f"{hour_words}{suffix}"
        return f"{hour_words} {minute_words()}{suffix}"
    hour_words = integer_to_english_words(hours)
    if seconds is not None:
        return f"{hour_words} {minute_words()} and {integer_to_english_words(seconds)} seconds"
    if minutes == 0:
        return f"{hour_words} o'clock"
    return f"{hour_words} {minute_words()}"


def speak_english_phone_number(match: str) -> Optional[str]:
    parsed = parse_phone_number_match(match)
    if not parsed:
        return None
    plus, digits = parsed
    spoken_digits = spell_english_digits(digits)
    return f"plus {spoken_digits}" if plus else spoken_digits


def speak_english_network_port(label: str, number_token: str) -> Optional[str]:
    port = int(number_token)
    if port < 0 or port > 65535:
        return None
    return f"{js_trim(label or '')} {spell_english_digits(number_token)}"


ENGLISH_DIGIT_WORD_SOURCE = "(?:zero|one|two|three|four|five|six|seven|eight|nine)"
ENGLISH_URL_PORT_PATTERN = js(
    f"\\b((?:https?:\\/\\/|www\\.)[^\\s/:?#]+):(\\d{{1,5}})\\b(?:\\s+\\(?((?:{ENGLISH_DIGIT_WORD_SOURCE})(?:\\s+{ENGLISH_DIGIT_WORD_SOURCE}){{0,4}})\\)?)?",
    "gi",
)
_WS_RUN = js(r"\s+", "g")


def normalize_english_url_ports(text: str, applied_rules: list[str]) -> str:
    def callback(match, url_prefix, number_token, existing_spoken, *_rest):
        port = int(number_token)
        if port < 0 or port > 65535:
            return match
        spoken = spell_english_digits(number_token)
        existing = js_replace(_WS_RUN, js_trim(existing_spoken or ""), " ")
        suffix = f" {existing}" if existing and existing.lower() != spoken.lower() else ""
        return f"{url_prefix}: {spoken}{suffix}"

    return replace_with_tracking(text, ENGLISH_URL_PORT_PATTERN, callback, "url_port_en", applied_rules)


def speak_english_date(month: int, day: int, year: int) -> Optional[str]:
    month_label = _month_label(month)
    if not month_label:
        return None
    return f"{month_label} {ordinal_english(day)} {year_to_english_words(year)}"


# --- speak callbacks -------------------------------------------------------------------------------

def _speak_domain(_match, label_part, tld):
    labels = [part for part in (js_trim(p) for p in (label_part or "").split(".")) if part]
    if not labels:
        return None
    return f"{' dot '.join(labels)} dot {(tld or '').lower().replace('.', ' dot ', 1)}"


def _speak_version(_match, short_prefix, short_first, short_second, prefix, first, second, third, fourth):
    if short_first is not None:
        return speak_english_version(short_prefix, [short_first, short_second])
    if all(len(part) == 3 for part in (second, third, fourth) if part):
        return None
    return speak_english_version(prefix, [first, second, third, fourth])


def _speak_slash_full(_match, first_token, second_token, year_token):
    first, second, year = int(first_token), int(second_token), int(year_token)
    if first > 12 and 1 <= second <= 12:
        day, month = first, second
    elif 1 <= first <= 12:
        month, day = first, second
    else:
        return None
    if day < 1 or day > 31:
        return None
    return speak_english_date(month, day, year) or None


def _speak_month_year(_match, month_token, year_token):
    month, year = int(month_token), int(year_token)
    if month < 1 or month > 12:
        return None
    return f"{ENGLISH_MONTH_LABELS[month - 1]} {year_to_english_words(year)}"


def _speak_fraction(_match, numerator_token, denominator_token):
    numerator, denominator = int(numerator_token), int(denominator_token)
    forms = ENGLISH_FRACTION_DENOMINATORS.get(denominator)
    if not forms or numerator >= denominator:
        return None
    return f"one {forms[0]}" if numerator == 1 else f"{integer_to_english_words(numerator)} {forms[1]}"


def _speak_month_day_slash(_match, month_token, day_token, year_token):
    month, day = int(month_token), int(day_token)
    year = None if year_token is None else parse_english_year_token(year_token)
    if year_token is not None and year is None:
        return None
    label = _month_label(month)
    if not label:
        return None
    return f"{label} {ordinal_english(day)}" if year is None else speak_english_date(month, day, year)


def _speak_iso(_match, year_token, month_token, day_token, hours_token, minutes_token, *_seconds):
    year, month, day = int(year_token), int(month_token), int(day_token)
    if month < 1 or month > 12 or day < 1 or day > 31:
        return None
    spoken_date = speak_english_date(month, day, year)
    if not spoken_date:
        return None
    if hours_token is not None:
        hours, minutes = int(hours_token), int(minutes_token)
        if hours <= 23 and minutes <= 59:
            return f"{spoken_date} at {speak_english_time(hours, minutes)}"
    return spoken_date


def _speak_year_month(_match, year_token, month_token):
    return f"{ENGLISH_MONTH_LABELS[int(month_token) - 1]} {year_to_english_words(int(year_token))}"


def _speak_named_month(_match, month_label, day_token, year_token):
    month = english_month_index(month_label)
    day = int(day_token)
    year = parse_english_year_token(year_token)
    if not month or year is None or day < 1 or day > 31:
        return None
    return speak_english_date(month, day, year) or None


def _speak_day_month(_match, prefix, day_token, month_label, year_token):
    month = english_month_index(month_label)
    day = int(day_token)
    year = parse_english_year_token(year_token)
    if not month or year is None or day < 1 or day > 31:
        return None
    spoken_date = speak_english_date(month, day, year)
    if not spoken_date:
        return None
    return f"{prefix}{spoken_date}" if prefix else spoken_date


def _speak_month_day_range(_match, month_label, first_token, second_token):
    month = english_month_index(month_label)
    first, second = int(first_token), int(second_token)
    if not month or first < 1 or first > 31 or second < 1 or second > 31:
        return None
    return f"{ENGLISH_MONTH_LABELS[month - 1]} {ordinal_english(first)} to {ordinal_english(second)}"


def _speak_month_day(_match, month_label, day_token):
    month = english_month_index(month_label)
    day = int(day_token)
    if not month or day < 1 or day > 31:
        return None
    return f"{ENGLISH_MONTH_LABELS[month - 1]} {ordinal_english(day)}"


def _speak_day_month_no_year(_match, prefix, day_token, month_label):
    month = english_month_index(month_label)
    day = int(day_token)
    if not month or day < 1 or day > 31:
        return None
    spoken = f"{ENGLISH_MONTH_LABELS[month - 1]} {ordinal_english(day)}"
    return f"{prefix}{spoken}" if prefix else spoken


_STARTS_WITH_H = js(r"^h", "i")
_STARTS_WITH_P = js(r"^p", "i")


def _speak_duration_unit(_match, first_token, second_token, third_token, unit_token):
    first, second = int(first_token), int(second_token)
    third = None if third_token is None else int(third_token)
    if second > 59 or (third is not None and third > 59):
        return None
    if third is not None:
        return speak_english_duration(first, second, third, "zero seconds")
    if _STARTS_WITH_H.test(unit_token or ""):
        return speak_english_duration(first, second, None, "zero hours")
    return speak_english_duration(None, first, second, "zero minutes")


def _speak_duration_compact(_match, days_token, hours_token, minutes_token, seconds_token, only_minutes, only_seconds):
    if only_minutes is not None:
        return speak_english_duration(None, int(only_minutes), int(only_seconds), "zero seconds")
    minutes = int(minutes_token)
    seconds = None if seconds_token is None else int(seconds_token)
    if minutes > 59 or (seconds is not None and seconds > 59):
        return None
    span = speak_english_duration(int(hours_token), minutes, seconds, "zero minutes")
    if days_token is None:
        return span
    days = int(days_token)
    return f"{integer_to_english_words(days)} {'day' if days == 1 else 'days'}, {span}"


def _speak_hmm_dot(_match, hours_token, minutes_token, meridiem_token):
    hours, minutes = int(hours_token), int(minutes_token)
    if hours < 1 or hours > 12 or minutes > 59:
        return None
    meridiem = "PM" if _STARTS_WITH_P.test(meridiem_token) else "AM"
    return speak_english_time(hours, minutes, None, meridiem)


_MERIDIEM_NOISE = js(r"[\s.]", "g")


def _speak_hhmm(_match, hours_token, minutes_token, seconds_token, meridiem_token):
    hours, minutes = int(hours_token), int(minutes_token)
    seconds = None if seconds_token is None else int(seconds_token)
    meridiem_normalized = js_replace(_MERIDIEM_NOISE, meridiem_token or "", "").upper()
    meridiem = meridiem_normalized if meridiem_normalized in ("AM", "PM") else None
    if hours < 0 or hours > 23 or minutes < 0 or minutes > 59 or (seconds is not None and (seconds < 0 or seconds > 59)):
        return None
    return speak_english_time(hours, minutes, seconds, meridiem)


def _speak_hour_meridiem(_match, hours_token, meridiem_token):
    hours = int(hours_token)
    if hours < 1 or hours > 12:
        return None
    meridiem = "PM" if _STARTS_WITH_P.test(meridiem_token) else "AM"
    return speak_english_time(hours, 0, None, meridiem)


def _speak_feet_inches(_match, feet_token, inches_token):
    feet, inches = int(feet_token), int(inches_token)
    return (
        f"{integer_to_english_words(feet)} {'foot' if feet == 1 else 'feet'} "
        f"{integer_to_english_words(inches)} {'inch' if inches == 1 else 'inches'}"
    )


def _speak_currency_prefix(_match, sign, currency_token, amount_token):
    negative = "-" if (sign or amount_token.startswith("-")) else ""
    unsigned = amount_token[1:] if amount_token.startswith("-") else amount_token
    return speak_english_currency(f"{negative}{unsigned}", currency_token)


_X_SPLIT = js(r"\s*[x×]\s*", "i")


def _speak_dimension(_match, chain):
    return " by ".join(integer_to_english_words(int(token)) for token in _X_SPLIT.split(chain or ""))


ENGLISH_SPEAK_PASSES: list[SpeakPass] = [
    SpeakPass(
        "domain_en",
        js(r"(?<![\w@/.-])((?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+)(de|com|org|net|eu|at|ch|io|dev|info|news|co\.uk|us|ai)(?![\w-])", "gi"),
        _speak_domain,
    ),
    SpeakPass(
        "network_port_en",
        js(r"\b(port(?:\s+number)?)(?:\s*[:=]\s*|\s+)(\d{1,5})\b", "gi"),
        lambda _m, label, number_token: speak_english_network_port(label, number_token),
    ),
    SpeakPass(
        "ipv4_en",
        js(r"(?<![\w.])(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})(?::(\d{1,5}))?(?!\w|\.\d)", "g"),
        lambda _m, a, b, c, d, port: speak_english_ipv4([a, b, c, d], port),
    ),
    SpeakPass(
        "version_en",
        js(r"(?<![\w.,-])(?:(v)(\d{1,3})\.(\d{1,3})(?!\d|\.\d)|(v)?(\d{1,3})\.(\d{1,3})\.(\d{1,4})(?:\.(\d{1,4}))?(?!\d|[.,]\d))", "gi"),
        _speak_version,
    ),
    SpeakPass(
        "date_en_slash_full",
        js(r"(?<![\d.,/])(\d{1,2})\/(\d{1,2})\/((?:19|20)\d{2})(?![\d/])", "g"),
        _speak_slash_full,
    ),
    SpeakPass(
        "date_en_month_year",
        js(r"(?<![\d.,/])(0?[1-9]|1[0-2])\/((?:19|20)\d{2})(?![\d/])", "g"),
        _speak_month_year,
    ),
    SpeakPass(
        "fraction_en",
        js(r"(?<![\d.,/])([1-9]\d?)\/(100|1[0-2]|[2-9])(?![\d/])(?![.,]\d)", "g"),
        _speak_fraction,
    ),
    SpeakPass(
        "phone_number_en",
        js(PHONE_NUMBER_PATTERN_SOURCE, "gu"),
        lambda match: speak_english_phone_number(match),
    ),
    SpeakPass(
        "date_en_month_day_slash",
        js(r"(?<![\d/.,])(1[0-2]|0?[1-9])\/(3[01]|[12]\d|0?[1-9])(?:\/((?:19|20)?\d{2}))?(?![\d/])", "g"),
        _speak_month_day_slash,
    ),
    SpeakPass(
        "date_iso",
        js(r"(?<![\p{L}\d])(\d{4})-(\d{2})-(\d{2})(?:T(\d{2}):(\d{2})(?::(\d{2}))?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?(?!\d)", "gu"),
        _speak_iso,
    ),
    SpeakPass(
        "date_en_year_month",
        js(r"(?<![\w.-])((?:19|20)\d{2})-(0[1-9]|1[0-2])(?![\w-])", "g"),
        _speak_year_month,
    ),
    SpeakPass(
        "date_en_named_month",
        js(rf"\b({ENGLISH_MONTH_SOURCE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,\s*|\s+)(\d{{2,4}})\b(?!:)", "gi"),
        _speak_named_month,
    ),
    SpeakPass(
        "date_en_day_month",
        js(rf"\b(on\s+)?(\d{{1,2}})(?:st|nd|rd|th)?\s+({ENGLISH_MONTH_SOURCE})\.?\s+(\d{{2,4}})\b(?!:)", "gi"),
        _speak_day_month,
    ),
    SpeakPass(
        "date_en_month_day_range",
        js(
            rf"\b({ENGLISH_MONTH_SOURCE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\s*(?:to|[–—-])\s*(\d{{1,2}})(?:st|nd|rd|th)?\b(?!\s*,?\s*\d{{4}})(?![.:]\d)",
            "gi",
        ),
        _speak_month_day_range,
    ),
    SpeakPass(
        "date_en_month_day",
        js(rf"\b({ENGLISH_MONTH_SOURCE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?!\s*,?\s*\d{{2,4}}\b(?!:))(?![.:]\d)", "gi"),
        _speak_month_day,
    ),
    SpeakPass(
        "date_en_day_month_no_year",
        js(
            rf"\b(on\s+|the\s+)?(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({ENGLISH_MONTH_SOURCE})\.?(?![\p{{L}}])(?!\s*,?\s*\d{{2,4}}\b(?!:))",
            "giu",
        ),
        _speak_day_month_no_year,
    ),
    SpeakPass(
        "duration_en_unit",
        js(r"\b(\d{1,3}):(\d{2})(?::(\d{2}))?\s*(min(?:utes?|s)?|h(?:ou)?rs?|hours?|h)(?![\p{L}])", "giu"),
        _speak_duration_unit,
    ),
    SpeakPass(
        "duration_en_compact",
        js(
            r"\b(?:(\d{1,3})[ \t]?(?:d|days?)[ \t]?)?(\d{1,3})[ \t]?(?:h|hrs?|hours?)[ \t]?(?:and[ \t])?(\d{1,2})[ \t]?(?:m|mins?|minutes?)(?![\p{L}])(?:[ \t]?(?:and[ \t])?(\d{1,2})[ \t]?(?:s|secs?|seconds?)(?![\p{L}]))?|\b(\d{1,3})[ \t]?m[ \t]?(\d{1,2})[ \t]?s(?![\p{L}])",
            "gu",
        ),
        _speak_duration_compact,
    ),
    SpeakPass(
        "time_en_hmm_dot",
        js(
            r"\b(\d{1,2})\.(\d{2})\s?(a\.?m(?:\.(?![ \t]*$|[ \t]+\p{Lu}))?|p\.?m(?:\.(?![ \t]*$|[ \t]+\p{Lu}))?)(?![\p{L}])",
            "giu",
        ),
        _speak_hmm_dot,
    ),
    SpeakPass(
        "time_en_hhmm",
        js(
            r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?(?:\s*(a\.?\s*m(?:\.(?![ \t]*$|[ \t]+\p{Lu}))?|p\.?\s*m(?:\.(?![ \t]*$|[ \t]+\p{Lu}))?))?(?![\p{L}\d])",
            "giu",
        ),
        _speak_hhmm,
    ),
    SpeakPass(
        "time_en_hour_meridiem",
        js(
            r"(?<![\w.:])(\d{1,2})\s*(a\.?\s*m(?:\.(?![ \t]*$|[ \t]+\p{Lu}))?|p\.?\s*m(?:\.(?![ \t]*$|[ \t]+\p{Lu}))?)(?![\p{L}])",
            "giu",
        ),
        _speak_hour_meridiem,
    ),
    SpeakPass(
        "ordinal_en",
        js(r"\b(\d{1,4})(?:st|nd|rd|th)\b", "g"),
        lambda _m, number_token: ordinal_english(int(number_token)),
    ),
    SpeakPass(
        "feet_inches_en",
        js(r"\b(\d{1,2})\s?(?:ft|')\s?(\d{1,2})\s?(?:in|\")(?![\p{L}])", "gu"),
        _speak_feet_inches,
    ),
    SpeakPass(
        "inch_mark_en",
        js(r"(?<=\s)(\d+(?:\.\d+)?)\"(?=\s|$|[,.;:)])", "g"),
        lambda _m, number_token: f"{speak_english_numeric_token(number_token)} {'inch' if number_token == '1' else 'inches'}",
    ),
    SpeakPass(
        "amount_scale_en",
        js(
            r"(?<!\w)(?:(€|\$|£|(?:EUROS?|EUR|USD|GBP)\b)\s*)?(-?\d[\d.,]*\d|-?\d)\s*(bn|mn|tn)\b\.?(?:\s*(€|\$|£|(?:EUROS?|EUR|USD|GBP)\b))?",
            "gi",
        ),
        lambda _m, prefix_currency, number_token, scale_token, suffix_currency: speak_english_amount_scale(
            number_token, scale_token, prefix_currency or suffix_currency
        ),
    ),
    SpeakPass(
        "amount_scale_symbol_en",
        js(r"(?<!\w)(€|\$|£)\s*(-?\d[\d.,]*)\s*([kmbKMB])\b", "g"),
        lambda _m, currency_token, number_token, scale_token: speak_english_amount_scale(
            number_token, scale_token, currency_token
        ),
    ),
    SpeakPass(
        "unit_suffix_en",
        js(rf"(?<!\w)(-?\d[\d.,]*)([ \t]*)({ENGLISH_UNIT_KEY_SOURCE})(?:\.(?![ \t]*$|[ \t]+\p{{Lu}}))?(?![\p{{L}}\d])", "gu"),
        lambda _m, number_token, spacing, unit_token: speak_english_unit(number_token, spacing, unit_token),
    ),
    SpeakPass(
        "hex_color_en",
        js(r"#([0-9a-fA-F]{6}|[0-9a-fA-F]{8})\b", "g"),
        lambda _m, hex_token: spell_english_hex(hex_token),
    ),
    SpeakPass(
        "currency_prefix",
        js(r"(?<!\w)(-?)(€|\$|£|(?:EUROS?|EUR|USD|GBP)\b)\s*(-?\d(?:[\d.,]*\d)?)", "gi"),
        _speak_currency_prefix,
    ),
    SpeakPass(
        "currency_suffix",
        js(
            r"(?<!\w)(-?\d(?:[\d.,]*\d)?)\s*(€|\$|£|(?:EUROS?|EUR|USD|GBP)\b)(?=$|[\s,.;:!?)]|\/(?!(?:EUR|USD|GBP|CHF|JPY)\b))",
            "gi",
        ),
        lambda _m, amount_token, currency_token: speak_english_currency(amount_token, currency_token),
    ),
    SpeakPass(
        "percent_en",
        js(r"(?<!\w)(-?\d[\d.,]*)\s*%", "g"),
        lambda _m, number_token: f"{speak_english_numeric_token(number_token)} percent",
    ),
    SpeakPass(
        "temperature_en",
        js(r"(?<!\w)(-?\d[\d.,]*)\s*(?:°|º)\s*([CFK])\b", "gi"),
        lambda _m, number_token, unit_token: speak_english_temperature(number_token, (unit_token or "").upper()),
    ),
    SpeakPass(
        "dimension_en",
        js(r"(?<![\w.,])([1-9]\d{0,4}(?:\s*[x×]\s*\d{1,5})+)(?!\w|[.,]\d)", "gi"),
        _speak_dimension,
    ),
    SpeakPass(
        "decimal_number_en",
        js(r"(?<![\w.,])(-?\d{1,3}(?:,\d{3})+\.\d+|-?\d+\.\d+)(?![.,]?\d)", "g"),
        lambda _m, number_token: speak_english_numeric_token(number_token),
    ),
    SpeakPass(
        "thousands_number_en",
        js(r"(?<![\w.,])(-?\d{1,3}(?:,\d{3})+|-?\d{1,3}(?:\.\d{3}){2,})(?![.,]?\d)", "g"),
        lambda _m, number_token: speak_english_numeric_token(
            number_token.replace(".", ",") if "." in number_token else number_token
        ),
    ),
    SpeakPass(
        "digit_sequence_en",
        js(r"\b(0\d{2,}|\d{5,})\b", "g"),
        lambda _m, number_token: spell_english_digits(number_token),
    ),
    SpeakPass(
        "year_number_en",
        js(
            "(?:(?<=\\b(?:in|since|from|until|till|through|(?<![\\d.,][ \\t]*)by|of|around|circa|after|before|between|year|born|died|founded|established|late|early|mid|spring|summer|autumn|fall|winter|"
            + ENGLISH_MONTH_SOURCE
            + ")[.:]?\\s+(?:1\\d{3}\\s*(?:,|and|or|to|through|until|till)\\s+)*)|(?<=\\()|(?<=(?:^|[.!?])\\s*))(1\\d{3})(?![\\p{L}\\p{N}_]|[.,]\\d|[ \\t]+by[ \\t]+\\d)",
            "gimu",
        ),
        lambda _m, number_token: year_to_english_words(int(number_token)),
    ),
    SpeakPass(
        "integer_number_en",
        js(r"(?<![\w.])(-?\d+)\b(?!\.\d)", "g"),
        lambda _m, number_token: speak_english_numeric_token(number_token),
    ),
]

ENGLISH_SPEAK_PASS_BY_LABEL = {speak_pass.label: speak_pass for speak_pass in ENGLISH_SPEAK_PASSES}

_FILENAME_LIST_ITEM = js(r"^(\s*)(\d{1,2})\.\s+(.+?\.[A-Za-z0-9]{2,8})\s*$", "gm")

ENGLISH_NOTATION_PREPASS = NotationPrepassConfig(
    lang="en",
    integer_words=integer_to_english_words,
    digit_words=spell_english_digits,
    year_words=year_to_english_words,
)


def optimize_english_tts_text(text: str) -> TtsAlgorithmOptimizationResult:
    """``optimizeEnglishTtsText``."""
    result = text or ""
    applied_rules: list[str] = []
    if not js_trim(result):
        return TtsAlgorithmOptimizationResult(result, applied_rules, "en")

    def run_speak_pass(label: str) -> None:
        nonlocal result
        speak_pass = ENGLISH_SPEAK_PASS_BY_LABEL.get(label)
        if speak_pass is None:
            return

        def callback(*args):
            match = args[0]
            spoken = speak_pass.speak(match, *args[1:-2])
            return match if spoken is None else spoken

        result = replace_with_tracking(result, speak_pass.pattern, callback, speak_pass.label, applied_rules)

    def filename_item(match, leading_whitespace, index_token, filename, *_rest):
        index = int(index_token)
        if index < 1 or index > 99:
            return match
        return f"{leading_whitespace}{capitalize_english_token(ordinal_english(index))}: {speak_english_filename_phrase(filename)}"

    result = replace_with_tracking(result, _FILENAME_LIST_ITEM, filename_item, "filename_list_item_en", applied_rules)
    result = normalize_english_url_ports(result, applied_rules)
    result = resolve_english_year_hint_leftovers(result, applied_rules)
    result = apply_notation_prepass(result, ENGLISH_NOTATION_PREPASS, applied_rules)
    for label in (
        "domain_en", "network_port_en", "ipv4_en", "version_en", "date_en_slash_full", "date_en_month_year",
        "fraction_en", "phone_number_en", "date_en_month_day_slash", "date_iso", "date_en_year_month",
        "date_en_named_month", "date_en_day_month", "date_en_month_day_range", "date_en_month_day",
        "date_en_day_month_no_year", "duration_en_unit", "duration_en_compact", "time_en_hmm_dot",
        "time_en_hhmm", "time_en_hour_meridiem", "ordinal_en", "feet_inches_en", "inch_mark_en",
    ):
        run_speak_pass(label)

    result = replace_ordered_sequence_markers(
        result, lambda index: capitalize_english_token(ordinal_english(index)), "list_marker_en", applied_rules
    )

    for label in (
        "amount_scale_en", "amount_scale_symbol_en", "unit_suffix_en", "hex_color_en", "currency_prefix",
        "currency_suffix", "percent_en", "temperature_en", "dimension_en", "decimal_number_en",
        "thousands_number_en", "digit_sequence_en", "year_number_en", "integer_number_en",
    ):
        run_speak_pass(label)

    return TtsAlgorithmOptimizationResult(finalize_optimized_text(result), applied_rules, "en")
