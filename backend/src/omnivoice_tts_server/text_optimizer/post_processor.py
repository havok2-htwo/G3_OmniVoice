"""Port of Genesis3 ``src/utils/ttsTextPostProcessor.ts`` (``postProcessTtsText``).

Runs after the language layer: emoji/quote strip, dash pauses, technical
separators, Markdown/LaTeX leftovers, dotted abbreviations (filtered to the
bound language), Genesis-specific token rules, acronym spelling and the final
whitespace/punctuation cleanup. Direct URLs are protected throughout.
"""

from __future__ import annotations

from typing import Optional

from .jsre import js, js_replace, js_trim
from .shared import strip_emoji_for_tts, strip_quotation_marks_for_tts

TRANSIT_STOP_WITH_LABEL_PATTERN = js(r"\b[A-Z\u00c4\u00d6\u00dc]{2,5}-\d{2}\s*\|\s*([^\n,.;:!?()]+)", "g")

# (label, pattern, replacement, lang, closes_sentence)
DOTTED_ABBREVIATION_REPLACEMENTS = [
    ("abbr_idr", js(r"\bi\.\s*d\.\s*r\.?(?=\W|$)", "gi"), "in der Regel", "de", False),
    ("abbr_zb", js(r"\bz\.\s*b\.?(?=\W|$)", "gi"), "zum Beispiel", "de", False),
    ("abbr_dh", js(r"\bd\.\s*h\.?(?=\W|$)", "gi"), "das heißt", "de", False),
    ("abbr_ua", js(r"\bu\.\s*a\.?(?=\W|$)", "gi"), "unter anderem", "de", False),
    ("abbr_uu", js(r"\bu\.\s*u\.?(?=\W|$)", "gi"), "unter Umständen", "de", False),
    ("abbr_bzw", js(r"\bbzw\.?(?=\W|$)", "gi"), "beziehungsweise", "de", False),
    ("abbr_usw", js(r"\busw\.?(?=\W|$)", "gi"), "und so weiter", "de", False),
    ("abbr_ggf", js(r"\bggfs?\.?(?=\W|$)", "gi"), "gegebenenfalls", "de", False),
    ("abbr_uvm", js(r"\bu\.\s?v\.\s?m\.(?=\W|$)", "gi"), "und vieles mehr", "de", True),
    ("abbr_uvm", js(r"\buvm\.?(?=\W|$)", "gi"), "und vieles mehr", "de", True),
    ("abbr_bspw", js(r"\bbspw\.(?=\W|$)", "gi"), "beispielsweise", "de", False),
    ("abbr_sog", js(r"\bsog\.(?=\s)", "gi"), "sogenannte", "de", False),
    ("abbr_mind", js(r"\bmind\.(?=\s)", "gi"), "mindestens", "de", False),
    ("abbr_ca", js(r"\bca\.(?=\s)", "gi"), "circa", "de", False),
    ("abbr_ie_de", js(r"\bI\.\s?E\.(?=\W|$)", "g"), "Internationale Einheiten", "de", False),
    ("abbr_hbf", js(r"\bHbf\.?(?=\W|$)", "g"), "Hauptbahnhof", "de", True),
    ("abbr_mwst", js(r"\bMwSt\.?(?=\W|$)", "g"), "Mehrwertsteuer", "de", True),
    ("abbr_zzgl", js(r"\bzzgl\.(?=\W|$)", "gi"), "zuzüglich", "de", False),
    ("abbr_inkl", js(r"\binkl\.(?=\W|$)", "gi"), "inklusive", "de", False),
    ("abbr_exkl", js(r"\bexkl\.(?=\W|$)", "gi"), "exklusive", "de", False),
    ("abbr_evtl", js(r"\bevtl\.(?=\W|$)", "gi"), "eventuell", "de", False),
    ("abbr_vgl", js(r"\bvgl\.(?=\W|$)", "gi"), "vergleiche", "de", False),
    ("abbr_nr", js(r"\bNr\.(?=\s)", "g"), "Nummer", "de", False),
    ("abbr_pa", js(r"\bp\.\s?a\.(?=\W|$)", "g"), "pro Jahr", "de", False),
    ("abbr_abs", js(r"\bAbs\.(?=\s)", "g"), "Absatz", "de", False),
    ("abbr_art", js(r"\bArt\.(?=\s)", "g"), "Artikel", "de", False),
    ("abbr_max", js(r"\bmax\.(?=\s)", "gi"), "maximal", "de", False),
    ("abbr_min", js(r"\bmin\.(?=\s)", "gi"), "mindestens", "de", False),
    ("abbr_tel", js(r"\bTel\.(?=\s)", "g"), "Telefon", "de", False),
    ("abbr_str", js(r"(\p{L})str\.(?=\W|$)", "gu"), lambda match, letter, *_r: f"{letter}straße", "de", False),
    ("abbr_str", js(r"\bStr\.(?=\W|$)", "g"), "Straße", "de", False),
    ("abbr_jh", js(r"\bJh\.(?=\W|$)", "g"), "Jahrhundert", "de", True),
    ("abbr_pkt", js(r"\bPkt\.?(?=\W|$)", "g"), "Punkte", "de", True),
    ("abbr_gleis", js(r"\bGl\.(?=\s)", "g"), "Gleis", "de", False),
    ("abbr_wagen", js(r"\bWg\.(?=\s)", "g"), "Wagen", "de", False),
    ("abbr_platz", js(r"\bPl\.(?=\s)", "g"), "Platz", "de", False),
    ("abbr_abf", js(r"\bAbf\.(?=\s)", "g"), "Abfahrt", "de", False),
    ("abbr_ank", js(r"\bAnk\.(?=\s)", "g"), "Ankunft", "de", False),
    ("abbr_eg", js(r"\be\.\s*g\.?(?=\W|$)", "gi"), "for example", "en", False),
    ("abbr_ie", js(r"\bi\.\s*e\.?(?=\W|$)", "gi"), "that is", "en", False),
    ("abbr_approx", js(r"\bapprox\.(?=\W|$)", "gi"), "approximately", "en", False),
    ("abbr_vs", js(r"\bvs\.(?=\W|$)", "gi"), "versus", "en", False),
    ("abbr_no", js(r"\bNo\.(?=\s?\d)", "g"), "number", "en", False),
    ("abbr_wo", js(r"\bw\/o(?=\s)", "g"), "without", "en", False),
    ("abbr_w", js(r"\bw\/(?=\s)", "g"), "with", "en", False),
]

EXACT_TOKEN_SPELLINGS = {
    "GMBH": "G M B H", "GBR": "G B R", "OHG": "O H G", "KG": "K G", "UG": "U G", "AG": "A G", "SE": "S E",
    "EV": "E V", "EK": "E K", "KI": "K I", "AI": "A I", "IT": "I T", "LLM": "L L M", "TTS": "T T S",
    "GPT": "G P T", "API": "A P I", "SDK": "S D K", "URL": "U R L", "USB": "U S B", "CPU": "C P U",
    "GPU": "G P U", "JSON": "J S O N", "XML": "X M L", "HTML": "H T M L", "CSS": "C S S", "HTTP": "H T T P",
    "HTTPS": "H T T P S", "DNS": "D N S", "IP": "I P", "PDF": "P D F", "DOCX": "D O C X", "XLSX": "X L S X",
    "BMW": "B M W", "VW": "V W", "DHL": "D H L", "ADAC": "A D A C", "RKI": "R K I", "ARD": "A R D",
    "ZDF": "Z D F", "TÜV": "T Ü V", "NASA": "N A S A", "EU": "E U", "USA": "U S A",
    "ADM": "A D M", "NFC": "N F C", "BLE": "B L E", "VPN": "V P N", "OBD": "O B D", "ABS": "A B S",
    "ESP": "E S P", "SSD": "S S D", "HDD": "H D D", "NAS": "N A S", "FTP": "F T P", "SSH": "S S H",
    "SQL": "S Q L", "CRM": "C R M", "ERP": "E R P", "MVP": "M V P", "SLA": "S L A",
    "HR": "H R", "PR": "P R", "FAQ": "F A Q", "PIN": "P I N", "TAN": "T A N", "IBAN": "I B A N", "BIC": "B I C",
    "SWR": "S W R", "MDR": "M D R", "NDR": "N D R", "WDR": "W D R", "BBC": "B B C", "CNN": "C N N",
}

PRESERVE_UPPERCASE_TOKENS = {"OK", "RAM", "HAVAG", "MDV", "PVP", "HOS", "SONS", "INSREICH"}

SPELLED_FILE_EXTENSIONS = {
    "7Z", "AVI", "CSV", "DB", "DOC", "DOCX", "GIF", "HTM", "HTML", "JPEG", "JPG", "JSON", "JS", "JSX", "MD",
    "MKV", "MOV", "MP3", "MP4", "ODP", "ODS", "ODT", "PDF", "PNG", "PPT", "PPTX", "PY", "RAR", "SQL", "SVG",
    "TAR", "TS", "TSV", "TSX", "TXT", "WAV", "WEBP", "XLS", "XLSX", "XML", "YAML", "YML", "ZIP",
}

_DIRECT_URL = js(r"\b(?:https?:\/\/|www\.)[^\s<>()]+", "gi")
_URL_PLACEHOLDER = js(r"\uE000TTSURL(\d+)\uE001", "g")


def _protect_direct_urls(text: str) -> tuple[str, list[str]]:
    urls: list[str] = []

    def callback(url, *_rest):
        urls.append(url)
        return f"\ue000TTSURL{len(urls) - 1}\ue001"

    return js_replace(_DIRECT_URL, text, callback), urls


def _restore_direct_urls(text: str, urls: list[str]) -> str:
    def callback(placeholder, index_token, *_rest):
        index = int(index_token)
        return urls[index] if index < len(urls) else placeholder

    return js_replace(_URL_PLACEHOLDER, text, callback)


def _replace_with_tracking(text, pattern, replacement, label, applied_rules) -> str:
    """The post-processor's own replaceWithTracking (dedupe happens at the end)."""
    did_change = False

    def callback(*args):
        nonlocal did_change
        original = args[0] if args[0] is not None else ""
        replaced = replacement(*args) if callable(replacement) else replacement
        if original != replaced:
            did_change = True
        return replaced

    result = js_replace(pattern, text, callback)
    if did_change:
        applied_rules.append(label)
    return result


def _spell_out_letters(value: str) -> str:
    return " ".join(value)


_WS_RUN = js(r"\s+", "g")
_FILENAME_LITERAL = js(r"\b([\p{L}\p{N}][\p{L}\p{N}_-]*)\s*\.\s*([A-Za-z0-9]{1,8})\b", "giu")
_FILENAME_SPOKEN_DOT = js(r"\b([\p{L}\p{N}][\p{L}\p{N}_-]*)\s+(?:dot|punkt)\s+([A-Za-z0-9]{1,8})\b", "giu")
_FILENAME_SPACED_EXTENSION = js(
    r"\b([\p{L}\p{N}][\p{L}\p{N}_-]*)\s+(?:dot|punkt)\s+((?:[A-Za-z0-9]\s+){1,7}[A-Za-z0-9])\b", "giu"
)


def _normalize_spoken_file_extensions(text: str, applied_rules: list[str]) -> str:
    def replace_extension(match, basename, extension, *_rest):
        compact = js_replace(_WS_RUN, extension or "", "").upper()
        if compact not in SPELLED_FILE_EXTENSIONS:
            return match
        return f"{basename} {_spell_out_letters(compact)}"

    result = _replace_with_tracking(text, _FILENAME_LITERAL, replace_extension, "filename_extension_spelling", applied_rules)
    result = _replace_with_tracking(result, _FILENAME_SPOKEN_DOT, replace_extension, "filename_extension_spelling", applied_rules)
    result = _replace_with_tracking(
        result, _FILENAME_SPACED_EXTENSION, replace_extension, "filename_extension_spelling", applied_rules
    )
    return result


_TECHNICAL_SEPARATORS = js(r"[_/\\-]+", "g")
_DIGIT = js(r"\d", "")
_ONLY_HYPHENS = js(r"^-+$", "")
_LETTER_OR_NUMBER = js(r"[\p{L}\p{N}]", "u")


def _normalize_technical_separators(text: str, applied_rules: list[str]) -> str:
    def callback(separator, offset, source):
        previous = source[offset - 1] if offset > 0 else ""
        end = offset + len(separator)
        following = source[end] if end < len(source) else ""
        if separator == "/" and _DIGIT.test(previous) and _DIGIT.test(following):
            return separator
        if _ONLY_HYPHENS.test(separator):
            if _DIGIT.test(previous) and _DIGIT.test(following):
                return separator
            if not previous and _DIGIT.test(following):
                return separator
        if "_" in separator or "/" in separator or "\\" in separator:
            return " "
        if _LETTER_OR_NUMBER.test(previous) and _LETTER_OR_NUMBER.test(following):
            return " "
        return separator

    return _replace_with_tracking(text, _TECHNICAL_SEPARATORS, callback, "technical_identifier_separators", applied_rules)


_MARKDOWN_FORMATTING_STEPS = (
    (js(r"!\[[^\]]*\](?:\([^)]*\))?", "g"), ""),
    (js(r"\[([^\]]+)\]\([^)]*\)", "g"), "$1"),
    (js(r"\[([^\]]+)\]", "g"), "$1"),
    (js(r"\*{3}([^*]+)\*{3}", "g"), "$1"),
    (js(r"_{3}([^_]+)_{3}", "g"), "$1"),
    (js(r"\*{2}([^*]+)\*{2}", "g"), "$1"),
    (js(r"_{2}([^_]+)_{2}", "g"), "$1"),
    (js(r"(^|\s)\*([^*\n]+)\*(?=\s|$|[,.;:!?])", "gm"), "$1$2"),
    (js(r"(^|\s)_([^_\n]+)_(?=\s|$|[,.;:!?])", "gm"), "$1$2"),
    (js(r"`([^`]+)`", "g"), "$1"),
    (js(r"```[^\n]*\n[\s\S]*?```", "g"), ""),
    (js(r"~~([^~\n]+?)~~", "g"), ""),
    (js(r"^\s*(?:```|~~~)[^\n]*$", "gm"), ""),
    (js(r"^#{1,6}\s+", "gm"), ""),
    (js(r"^>+\s?", "gm"), ""),
    (js(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$", "gm"), ""),
    (js(r"\$\$[\s\S]*?\$\$", "g"), ""),
    (js(r"\$([^$\n]+)\$", "g"), "$1"),
    (js(r"\\(?:text|mathrm|mathbf|mathit|mathcal|operatorname)\{([^}]*)\}", "g"), "$1"),
    (js(r"\\frac\{([^}]*)\}\{([^}]*)\}", "g"), "$1/$2"),
    (js(r"\\(?:infty|cdot|cdots|dots|ldots|times|div|pm|mp)\b", "g"), "..."),
    (js(r"\\(?:sum|prod|int|lim|max|min|sup|inf)\b", "g"), ""),
    (js(r"\\[a-zA-Z]+\{[^}]*\}", "g"), ""),
    (js(r"\\[a-zA-Z]+", "g"), ""),
    (js(r"[{}]", "g"), ""),
    (js(r"(?<![\w])\*+(?![\w])", "g"), ""),
)


def _strip_markdown_formatting(text: str, applied_rules: list[str]) -> str:
    result = text
    for pattern, replacement in _MARKDOWN_FORMATTING_STEPS:
        result = js_replace(pattern, result, replacement)
    if result != text:
        applied_rules.append("strip_markdown")
    return result


def _replace_transit_stop_codes_with_labels(text: str, applied_rules: list[str]) -> str:
    return _replace_with_tracking(
        text,
        TRANSIT_STOP_WITH_LABEL_PATTERN,
        lambda match, stop_label, *_r: js_trim(stop_label or ""),
        "transit_stop_label",
        applied_rules,
    )


_MIXED_CASE_PATTERNS = (
    (js(r"\bGmbH\b", "g"), "GMBH"),
    (js(r"\bGbR\b", "g"), "GBR"),
    (js(r"\bOHG\b", "g"), "OHG"),
    (js(r"\bKG\b", "g"), "KG"),
    (js(r"\bUG\b", "g"), "UG"),
    (js(r"\bAG\b", "g"), "AG"),
    (js(r"\bSE\b", "g"), "SE"),
    (js(r"\be\.\s*V\.\b", "gi"), "EV"),
    (js(r"\be\.\s*K\.\b", "gi"), "EK"),
)


def _replace_known_mixed_case_tokens(text: str, applied_rules: list[str]) -> str:
    result = text
    for pattern, token in _MIXED_CASE_PATTERNS:
        replacement = EXACT_TOKEN_SPELLINGS.get(token)
        if not replacement:
            continue
        result = _replace_with_tracking(result, pattern, replacement, f"token_{token.lower()}", applied_rules)
    return result


_ACRONYM_HYPHEN = js(r"\b([A-Z\u00c4\u00d6\u00dc]{2,8})-(?=\w)", "g")
_ACRONYM = js(r"\b([A-Z\u00c4\u00d6\u00dc]{2,8})\b", "g")


def _replace_known_acronyms(text: str, applied_rules: list[str]) -> str:
    def hyphen(match, token, *_rest):
        exact = EXACT_TOKEN_SPELLINGS.get((token or "").upper())
        return f"{exact} - " if exact else match

    hyphen_pass = _replace_with_tracking(text, _ACRONYM_HYPHEN, hyphen, "known_acronym_hyphen_compound", applied_rules)

    def standalone(match, token, *_rest):
        normalized = (token or "").upper()
        if normalized in PRESERVE_UPPERCASE_TOKENS:
            return match
        exact = EXACT_TOKEN_SPELLINGS.get(normalized)
        return exact if exact is not None else match

    return _replace_with_tracking(hyphen_pass, _ACRONYM, standalone, "generic_acronyms", applied_rules)


_DASH_PAUSE = js(r"[ \t]*[–—][ \t]*", "g")
# The TS callback tests source.slice(offset + match.length) against /^[ \t]*(?:$|\n)/ and
# /^[ \t]+["“„]?\p{Lu}/u. Sticky matching at that position gives the same answer without
# copying the rest of the text for every abbreviation (quadratic on long inputs).
_AT_END = js(r"[ \t]*(?:$|\n)", "y")
_BEFORE_CAPITAL = js(r"[ \t]+[\"“„]?\p{Lu}", "uy")
_SONS = js(r"\bS\s+O\s+N\s+S\b", "g")
_INSREICH = js(r"\bI\s+N\s+S\s+R\s+E\s+I\s+C\s+H\b", "g")
_UNDERSCORE_WORD_BETWEEN = js(r"(\w)\s+(?:Unterstrich|underscore)\s+(\w)", "gi")
_UNDERSCORE_WORD = js(r"\b(?:Unterstrich|underscore)\b", "gi")
_HYPHEN_WORD = js(r"([\p{L}\d])\s+(?:Bindestrich|hyphen)\s+([\p{L}\d])", "giu")
_SLASH_WORD = js(r"(\w)\s+(?:Schr\u00e4gstrich|slash|backslash)\s+(\w)", "giu")
_PHONE_SLASH = js(r"(\d)(?:\s+\/\s*|\s*\/\s+)(\d)", "g")
_WORD_ECHO = js(r"(\b[\p{L}\p{M}]+)\s*\(\s*\1\s*\)", "giu")
_PHRASE_ECHO = js(r"((?:\b[\p{L}\p{M}]+\b[ \t]+){1,7}\b[\p{L}\p{M}]+\b)\s*\(\s*\1\s*\)", "giu")
_EXCLAMATION_END = js(r"!+(?=\s|$)", "g")
_EXCLAMATION = js(r"!", "g")
_MULTI_WS = js(r"\s{2,}", "g")
_SPACE_BEFORE_PUNCT = js(r"\s+([,.;:!?])", "g")


def post_process_tts_text(text: str, language: Optional[str] = None) -> tuple[str, list[str]]:
    """``postProcessTtsText``. Returns (text, applied_rules)."""
    result = text or ""
    applied_rules: list[str] = []
    if not js_trim(result):
        return result, applied_rules

    normalized_language = js_trim(language or "").lower()[:2]
    bound_language = normalized_language if normalized_language in ("de", "en") else None

    result, urls = _protect_direct_urls(result)

    emoji_stripped = strip_emoji_for_tts(result)
    if emoji_stripped != result:
        result = emoji_stripped
        applied_rules.append("emoji_strip")

    quotes_stripped = strip_quotation_marks_for_tts(result)
    if quotes_stripped != result:
        result = quotes_stripped
        applied_rules.append("quote_strip")

    result = _replace_with_tracking(result, _DASH_PAUSE, " - ", "dash_pause", applied_rules)
    result = _normalize_technical_separators(result, applied_rules)
    result = _strip_markdown_formatting(result, applied_rules)

    for label, pattern, replacement, lang, closes_sentence in DOTTED_ABBREVIATION_REPLACEMENTS:
        if bound_language and lang != bound_language:
            continue

        def callback(*args, _replacement=replacement, _closes=closes_sentence):
            match = args[0] if args[0] is not None else ""
            expanded = _replacement(*args) if callable(_replacement) else _replacement
            offset = args[-2]
            source = args[-1] if args[-1] is not None else ""
            rest_start = offset + len(match)
            at_end = _AT_END.match_at(source, rest_start) is not None
            before_capital = _closes is True and _BEFORE_CAPITAL.match_at(source, rest_start) is not None
            if match.endswith(".") and not expanded.endswith(".") and (at_end or before_capital):
                return f"{expanded}."
            return expanded

        result = _replace_with_tracking(result, pattern, callback, label, applied_rules)

    result = _replace_transit_stop_codes_with_labels(result, applied_rules)
    result = _replace_known_mixed_case_tokens(result, applied_rules)
    result = _normalize_spoken_file_extensions(result, applied_rules)

    result = js_replace(_SONS, result, "SONS")
    result = js_replace(_INSREICH, result, "INSREICH")

    result = _replace_known_acronyms(result, applied_rules)

    result = js_replace(_UNDERSCORE_WORD_BETWEEN, result, "$1 $2")
    result = js_replace(_UNDERSCORE_WORD, result, "")
    result = js_replace(_HYPHEN_WORD, result, "$1 $2")
    result = js_replace(_SLASH_WORD, result, "$1 $2")
    result = js_replace(_PHONE_SLASH, result, "$1 $2")
    result = js_replace(_WORD_ECHO, result, "$1")
    result = js_replace(_PHRASE_ECHO, result, "$1")

    result = js_replace(_EXCLAMATION_END, result, ".")
    result = js_replace(_EXCLAMATION, result, "")
    result = js_replace(_MULTI_WS, result, " ")
    result = js_replace(_SPACE_BEFORE_PUNCT, result, "$1")
    result = js_trim(result)

    result = _restore_direct_urls(result, urls)
    return result, list(dict.fromkeys(applied_rules))
