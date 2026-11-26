"""Port of Genesis3 ``src/utils/ttsTextPostProcessor.ts`` (``postProcessTtsText``).

Runs after the language layer: emoji/quote strip, "Pst" skip, dash pauses, technical separators,
Markdown/LaTeX leftovers, dotted abbreviations (filtered to the bound language), dotted capitals,
file extensions and dot names, acronym spelling with hyphens (Bug 964) and the final
whitespace/punctuation cleanup. Direct URLs are protected throughout.

The plugin word tokens of Genesis3 (``pluginTtsWordTokens``, bundled plugins) are empty here:
G3_OmniVoice has no plugins, and the golden fixtures are recorded with no plugin registered.
"""

from __future__ import annotations

from typing import Optional

from .jsre import js, js_replace, js_trim
from .latex_math import reduce_latex_for_tts, unwrap_inline_math_for_tts
from .shared import finalize_optimized_text, skip_shushing_for_tts, strip_emoji_for_tts, strip_quotation_marks_for_tts
from .spoken_abbreviations import (
    ENGLISH_NUMBER_WORD_SOURCE,
    GERMAN_NUMBER_WORD_SOURCE,
    SPOKEN_ABBREVIATIONS_DE,
    SPOKEN_ABBREVIATIONS_EN,
    speak_english_day,
    speak_german_day,
)

# Upper-case words a bundled Genesis plugin keeps as words (tts.wordTokens); none in OmniVoice.
PLUGIN_TTS_WORD_TOKENS: frozenset[str] = frozenset()

TRANSIT_STOP_WITH_LABEL_PATTERN = js(r"\b[A-ZÄÖÜ]{2,5}-\d{2}\s*\|\s*([^\n,.;:!?()]+)", "g")


class _Rule:
    """``DottedAbbreviationRule``."""

    __slots__ = ("label", "pattern", "replacement", "lang", "closes_sentence", "leads_next_word", "bound_only")

    def __init__(self, label, pattern, replacement, lang, *, closes_sentence=False, leads_next_word=False, bound_only=False):
        self.label = label
        self.pattern = pattern
        self.replacement = replacement
        self.lang = lang
        self.closes_sentence = closes_sentence
        self.leads_next_word = leads_next_word
        self.bound_only = bound_only


_NUMBER_SOURCE = {"de": GERMAN_NUMBER_WORD_SOURCE, "en": ENGLISH_NUMBER_WORD_SOURCE}
_WHITESPACE_CHAR = js(r"\s", "")


def _listed_abbreviation_rules(entries, lang: str) -> list[_Rule]:
    """Bug 870: the spellings of spoken_abbreviations.py, exact in case."""
    rules = []
    for spelling, word, kind in entries:
        spelled = spelling.replace(".", "\\.").replace(" ", r"\s+")
        number = _NUMBER_SOURCE[lang]
        if kind == "number":
            source = rf"(?<!(?:\d|\b{number})[ \t]+){spelled}(?=[ \t]*(?:\d|{number}))"
        else:
            source = rf"{spelled}(?=\s|[,;:)\]]|$)"

        def replacement(match, offset, text, _word=word, _kind=kind):
            following = text[offset + len(match)] if offset + len(match) < len(text) else ""
            return f"{_word} " if _kind == "number" and not _WHITESPACE_CHAR.test(following) else _word

        rules.append(
            _Rule(
                f"abbr_{spelling.lower()}",
                js(rf"(?<![\p{{L}}\p{{N}}./]){source}", "gu"),
                replacement,
                lang,
                closes_sentence=kind == "closes",
                leads_next_word=kind == "title",
                bound_only=True,
            )
        )
    return rules


_NW = r"(?:null|eins|ein|zwei|drei|vier|fünf|sechs|sech|sieben|sieb|acht|neun|zehn|elf|zwölf|zwanzig|ßig|zig|hundert|tausend|und)+(?![\p{L}])"

DOTTED_ABBREVIATION_REPLACEMENTS: list[_Rule] = [
    _Rule("abbr_art_nr", js(r"\bArt\.-Nr\.(?=\W|$)", "g"), "Artikelnummer", "de"),
    _Rule(
        "abbr_satz",
        js(rf"(?<=\b(?:Absatz|Abs\.|Artikel|Art\.|Paragraf|Paragraph|§)\s+(?:\d|{_NW})\s+)S\.(?=\s+(?:\d|{_NW}))", "gu"),
        "Satz",
        "de",
    ),
    _Rule(
        "abbr_seite",
        js(rf"(?<!\b(?:Seite|Satz)\s+)\bS\.(\s*)(?=(?:\d|{_NW}))", "gu"),
        lambda _match, gap, *_r: f"Seite{gap or ' '}",
        "de",
    ),
    _Rule("abbr_folgende", js(rf"(?<=(?:\d|{_NW})\s+)ff?\.(?=\W|$)", "gu"), "folgende", "de"),
    _Rule("abbr_kap", js(r"(?<!\bKapitel\s+)\bKap\.(?=\s)", "g"), "Kapitel", "de", leads_next_word=True),
    _Rule("abbr_vchr", js(r"\bv\.\s?Chr\.(?=\W|$)", "g"), "vor Christus", "de", closes_sentence=True),
    _Rule("abbr_nchr", js(r"\bn\.\s?Chr\.(?=\W|$)", "g"), "nach Christus", "de", closes_sentence=True),
    _Rule("abbr_eff", js(r"\beff\.(?=\s)", "g"), "effektiv", "de", leads_next_word=True),
    _Rule("abbr_pro_person", js(r"\bp\.\s?P\.(?=\W|$)", "g"), "pro Person", "de", closes_sentence=True),
    _Rule("abbr_taeglich", js(r"(?<![\p{L}])tägl\.(?=\W|$)", "gu"), "täglich", "de"),
    _Rule("abbr_whg", js(r"\bWhg\.(?=\W|$)", "g"), "Wohnung", "de"),
    _Rule("abbr_postf", js(r"\bPostf\.(?=\W|$)", "g"), "Postfach", "de"),
    _Rule("abbr_abt", js(r"\bAbt\.(?=\W|$)", "g"), "Abteilung", "de"),
    _Rule("abbr_dok", js(r"\bDok\.(?=\W|$)", "g"), "Dokument", "de"),
    _Rule("abbr_idr", js(r"\bi\.\s*d\.\s*r\.?(?=\W|$)", "gi"), "in der Regel", "de"),
    _Rule("abbr_zb", js(r"\bz\.\s*b\.?(?=\W|$)", "gi"), "zum Beispiel", "de"),
    _Rule("abbr_dh", js(r"\bd\.\s*h\.?(?=\W|$)", "gi"), "das heißt", "de"),
    _Rule("abbr_ua", js(r"\bu\.\s*a\.?(?=\W|$)", "gi"), "unter anderem", "de"),
    _Rule("abbr_uu", js(r"\bu\.\s*u\.?(?=\W|$)", "gi"), "unter Umständen", "de"),
    _Rule("abbr_bzw", js(r"\bbzw\.?(?=\W|$)", "gi"), "beziehungsweise", "de"),
    _Rule("abbr_usw", js(r"\busw\.?(?=\W|$)", "gi"), "und so weiter", "de"),
    _Rule("abbr_ggf", js(r"\bggfs?\.?(?=\W|$)", "gi"), "gegebenenfalls", "de"),
    _Rule("abbr_uvm", js(r"\bu\.\s?v\.\s?m\.(?=\W|$)", "gi"), "und vieles mehr", "de", closes_sentence=True),
    _Rule("abbr_uvm", js(r"\buvm\.?(?=\W|$)", "gi"), "und vieles mehr", "de", closes_sentence=True),
    _Rule("abbr_bspw", js(r"\bbspw\.(?=\W|$)", "gi"), "beispielsweise", "de", leads_next_word=True),
    _Rule("abbr_sog", js(r"\bsog\.(?=\s)", "gi"), "sogenannte", "de", leads_next_word=True),
    _Rule("abbr_mind", js(r"\bmind\.(?=\s)", "gi"), "mindestens", "de", leads_next_word=True),
    _Rule("abbr_ca", js(r"\bca\.(?=\s)", "gi"), "circa", "de", leads_next_word=True),
    _Rule("abbr_ie_de", js(r"\bI\.\s?E\.(?=\W|$)", "g"), "Internationale Einheiten", "de"),
    _Rule("abbr_hbf", js(r"\bHbf\.?(?=\W|$)", "g"), "Hauptbahnhof", "de", closes_sentence=True),
    _Rule("abbr_mwst", js(r"\bMwSt\.?(?=\W|$)", "gi"), "Mehrwertsteuer", "de", closes_sentence=True),
    _Rule("abbr_zzgl", js(r"\bzzgl\.(?=\W|$)", "gi"), "zuzüglich", "de", leads_next_word=True),
    _Rule("abbr_inkl", js(r"\binkl\.(?=\W|$)", "gi"), "inklusive", "de", leads_next_word=True),
    _Rule("abbr_exkl", js(r"\bexkl\.(?=\W|$)", "gi"), "exklusive", "de", leads_next_word=True),
    _Rule("abbr_evtl", js(r"\bevtl\.(?=\W|$)", "gi"), "eventuell", "de", leads_next_word=True),
    _Rule("abbr_vgl", js(r"\bvgl\.(?=\W|$)", "gi"), "vergleiche", "de", leads_next_word=True),
    _Rule("abbr_nr", js(r"\bNr\.(?=\s|:)", "g"), "Nummer", "de", leads_next_word=True),
    _Rule("abbr_pa", js(r"\bp\.\s?a\.(?=\W|$)", "g"), "pro Jahr", "de"),
    _Rule("abbr_abs", js(r"\bAbs\.(?=\s)", "g"), "Absatz", "de", leads_next_word=True),
    _Rule("abbr_art", js(r"\bArt\.(?=\s)", "g"), "Artikel", "de", leads_next_word=True),
    _Rule("abbr_max", js(r"\bmax\.(?=\s)", "gi"), "maximal", "de", leads_next_word=True),
    _Rule("abbr_min", js(r"\b(?:min\.(?=\s)|Min\.(?=[ \t]+(?:\d|\p{Ll})|\n))", "gu"), "mindestens", "de", leads_next_word=True),
    _Rule("abbr_tel", js(r"\bTel\.(?=\s)", "g"), "Telefon", "de", leads_next_word=True),
    _Rule("abbr_str", js(r"(\p{L})str\.(?=\W|$)", "gu"), lambda _match, letter, *_r: f"{letter}straße", "de"),
    _Rule("abbr_str", js(r"\bStr\.(?=\W|$)", "g"), "Straße", "de"),
    _Rule(
        "abbr_sankt",
        js(
            r"(?<!(?:\d|\b(?:eins|zwei|drei|vier|fünf|sechs|sieben|acht|neun|zehn|elf|zwölf|\p{Ll}*(?:zehn|zig|ßig|hundert|tausend)))[ \t]+)"
            r"\bSt\.(?:[ \t]*|-)(?=\p{Lu})",
            "gu",
        ),
        "Sankt ",
        "de",
        bound_only=True,
    ),
    _Rule("abbr_saint", js(r"(?<!(?:\d|\p{Lu}\p{L}*)[ \t]+)\bSt\.(?:[ \t]*|-)(?=\p{Lu})", "gu"), "Saint ", "en", bound_only=True),
    _Rule("abbr_jh", js(r"\bJh\.(?=\W|$)", "g"), "Jahrhundert", "de", closes_sentence=True),
    _Rule("abbr_pkt", js(r"\bPkt\.?(?=\W|$)", "g"), "Punkte", "de", closes_sentence=True),
    _Rule("abbr_stk", js(r"\bStk\.?(?=\W|$)", "g"), "Stück", "de"),
    _Rule("abbr_gleis", js(r"\bGl\.(?=\s)", "g"), "Gleis", "de", leads_next_word=True),
    _Rule("abbr_wagen", js(r"\bWg\.(?=\s)", "g"), "Wagen", "de", leads_next_word=True),
    _Rule("abbr_platz", js(r"\bPl\.(?=\s)", "g"), "Platz", "de", leads_next_word=True),
    _Rule("abbr_abf", js(r"\bAbf\.(?=\s)", "g"), "Abfahrt", "de", leads_next_word=True),
    _Rule("abbr_ank", js(r"\bAnk\.(?=\s)", "g"), "Ankunft", "de", leads_next_word=True),
    _Rule("abbr_eg", js(r"\be\.\s*g\.?(?=\W|$)", "gi"), "for example", "en"),
    _Rule("abbr_ie", js(r"\b[iI]\.\s*e\.?(?=\W|$)", "g"), "that is", "en"),
    _Rule("abbr_approx", js(r"\bapprox\.(?=\W|$)", "gi"), "approximately", "en"),
    _Rule("abbr_vs", js(r"\bvs\.(?=\W|$)", "gi"), "versus", "en"),
    _Rule(
        "abbr_no",
        js(
            r"(?<![?!][ \t]*)(?<!\b(?:is|was|are|were|be|say|says|said|Is|Was|Are|Were|Be|Say|Says|Said)[ \t]+)"
            r"\b[Nn]o\.(?=[ \t]*(?:\d|\p{Ll}))(?![ \t]*number\b)",
            "gu",
        ),
        "number",
        "en",
    ),
    _Rule("abbr_wo", js(r"\bw\/o(?=\s)", "g"), "without", "en"),
    _Rule("abbr_w", js(r"\bw\/(?=\s)", "g"), "with", "en"),
    *_listed_abbreviation_rules(SPOKEN_ABBREVIATIONS_DE, "de"),
    *_listed_abbreviation_rules(SPOKEN_ABBREVIATIONS_EN, "en"),
    _Rule(
        "abbr_weekday_de",
        js(r"(?<![\p{L}\p{N}.])(Mo|Di|Mi|Do|Fr|Sa|So)\.(?=\s|[,;:)]|$)", "gu"),
        lambda match, day, offset, source: speak_german_day(match, day, offset, source),
        "de",
        bound_only=True,
    ),
    _Rule(
        "abbr_gemaess",
        js(
            r"(?<![\p{L}\p{N}.])gem\.(?=[ \t]+(?:Paragraf|Paragraph|Artikel|Absatz|Ziffer|Nummer|de[rnms]|ihre[rnm]?|seine[rnm]?"
            r"|unsere[rnm]?)(?![\p{L}]))",
            "gu",
        ),
        "gemäß",
        "de",
        bound_only=True,
    ),
    _Rule(
        "abbr_weekday_en",
        js(r"(?<![\p{L}\p{N}.])(Mon|Tue|Wed|Thu|Fri|Sat)\.(?=\s|[,;:)]|$)", "gu"),
        lambda match, day, offset, source: speak_english_day(match, day, offset, source),
        "en",
        bound_only=True,
    ),
    _Rule("abbr_doctor", js(r"(?<!(?:\d|\p{Lu}\p{L}*)[ \t]+)\bDr\.[ \t]*(?=\p{Lu})", "gu"), "Doctor ", "en", bound_only=True),
]

# The letters are written apart by blanks here and joined by hyphens on use (Bug 964).
_EXACT_TOKEN_SPELLING_ENTRIES = (
    ("GMBH", "G M B H"), ("GBR", "G B R"), ("OHG", "O H G"), ("KG", "K G"), ("UG", "U G"), ("AG", "A G"),
    ("SE", "S E"), ("EV", "E V"), ("EK", "E K"), ("KI", "K I"), ("AI", "A I"), ("IT", "I T"), ("LLM", "L L M"),
    ("TTS", "T T S"), ("GPT", "G P T"), ("API", "A P I"), ("SDK", "S D K"), ("URL", "U R L"), ("USB", "U S B"),
    ("CPU", "C P U"), ("GPU", "G P U"), ("JSON", "J S O N"), ("XML", "X M L"), ("HTML", "H T M L"), ("CSS", "C S S"),
    ("HTTP", "H T T P"), ("HTTPS", "H T T P S"), ("DNS", "D N S"), ("IP", "I P"), ("PDF", "P D F"),
    ("DOCX", "D O C X"), ("XLSX", "X L S X"), ("CSV", "C S V"), ("BMW", "B M W"), ("VW", "V W"), ("DHL", "D H L"),
    ("ADAC", "A D A C"), ("RKI", "R K I"), ("ARD", "A R D"), ("ZDF", "Z D F"), ("TÜV", "T Ü V"), ("NASA", "N A S A"),
    ("EU", "E U"), ("USA", "U S A"), ("ADM", "A D M"), ("NFC", "N F C"), ("BLE", "B L E"), ("VPN", "V P N"),
    ("OBD", "O B D"), ("ABS", "A B S"), ("ESP", "E S P"), ("SSD", "S S D"), ("HDD", "H D D"), ("NAS", "N A S"),
    ("FTP", "F T P"), ("SSH", "S S H"), ("SQL", "S Q L"), ("DB", "D B"), ("CRM", "C R M"), ("ERP", "E R P"),
    ("MVP", "M V P"), ("SLA", "S L A"), ("HR", "H R"), ("PR", "P R"), ("FAQ", "F A Q"), ("PIN", "P I N"),
    ("TAN", "T A N"), ("IBAN", "I B A N"), ("BIC", "B I C"), ("SWR", "S W R"), ("MDR", "M D R"), ("NDR", "N D R"),
    ("WDR", "W D R"), ("BBC", "B B C"), ("CNN", "C N N"),
)

EXACT_TOKEN_SPELLINGS = {token: spelling.replace(" ", "-") for token, spelling in _EXACT_TOKEN_SPELLING_ENTRIES}

PRESERVE_UPPERCASE_TOKENS = {"OK", "RAM", "MDV", "PVP", "HOS", "INSREICH"}

SPELLED_FILE_EXTENSIONS = {
    "7Z", "AVI", "CSV", "DB", "DOC", "DOCX", "GIF", "HTM", "HTML", "JPEG", "JPG", "JSON", "JS", "JSX", "MD",
    "MKV", "MOV", "MP3", "MP4", "ODP", "ODS", "ODT", "PDF", "PNG", "PPT", "PPTX", "PY", "RAR", "SQL", "SVG",
    "TAR", "TS", "TSV", "TSX", "TXT", "WAV", "WEBP", "XLS", "XLSX", "XML", "YAML", "YML", "ZIP",
    "SH", "BAT", "PS1", "EXE", "DLL", "ISO", "APK", "CS", "CPP", "RS", "KT", "RB", "PHP", "CFG", "SCSS",
}

# Extensions said as a word after the spoken dot of the bound language (Bug 836).
SPOKEN_FILE_EXTENSIONS = {"CONF", "LOG", "INI", "ENV", "LOCK", "TOML", "BAK", "TMP", "JAVA", "PROPERTIES", "GRADLE"}

_DIRECT_URL = js(r"\b(?:https?:\/\/|www\.)[^\s<>()]+", "gi")
_URL_PLACEHOLDER = js(r"\uE000TTSURL(\d+)\uE001", "g")


def _protect_direct_urls(text: str) -> tuple[str, list[str]]:
    urls: list[str] = []

    def callback(url, *_rest):
        urls.append(url)
        return f"\uE000TTSURL{len(urls) - 1}\uE001"

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
    """Joined by hyphens, as the table's spellings (Bug 964)."""
    return "-".join(value)


_SPOKEN_NAME_DOT = {"de": "punkt", "en": "dot"}
_SPACES_AND_HYPHENS = js(r"[\s-]+", "g")
_FILENAME_LITERAL = js(r"\b([\p{L}\p{N}][\p{L}\p{N}_-]*)([ \t]?)\.([A-Za-z0-9]{1,10})\b", "giu")
_FILENAME_SPOKEN_DOT = js(r"\b([\p{L}\p{N}][\p{L}\p{N}_-]*)\s+(?:[Dd][Oo][Tt]|punkt)\s+()([A-Za-z0-9]{1,10})\b", "gu")
_FILENAME_SPACED_EXTENSION = js(
    r"\b([\p{L}\p{N}][\p{L}\p{N}_-]*)\s+(?:[Dd][Oo][Tt]|punkt)\s+()((?:[A-Za-z0-9](?:\s+|-)){1,7}[A-Za-z0-9])\b", "gu"
)
_AT_AFTER = js(r"\s+at(?![\p{L}\p{N}])", "uy")
_AT_BEFORE_DE = js(r"(?<![\p{L}\p{N}])at(?:\s+[\p{L}\p{N}-]+){0,8}\s+$", "u")
_AT_BEFORE_EN = js(r"(?<![\p{L}\p{N}])(?:at|dot|punkt)\s+$", "u")


def _normalize_spoken_file_extensions(text: str, language: Optional[str], applied_rules: list[str]) -> str:
    def replace_extension(match, basename, stray_space, extension, *_rest):
        compact = js_replace(_SPACES_AND_HYPHENS, extension or "", "").upper()
        if compact in SPELLED_FILE_EXTENSIONS:
            return f"{basename} {_spell_out_letters(compact)}"
        if language and compact in SPOKEN_FILE_EXTENSIONS and not stray_space:
            return f"{basename} {_SPOKEN_NAME_DOT[language]} {compact.lower()}"
        return match

    def inside_spoken_address(offset: int, match_length: int, source: str) -> bool:
        if _AT_AFTER.match_at(source, offset + match_length) is not None:
            return True
        before = source[max(0, offset - 160):offset]
        return (_AT_BEFORE_DE if language == "de" else _AT_BEFORE_EN).test(before)

    def replace_spoken_dot_extension(*args):
        match = args[0] if args[0] is not None else ""
        offset = args[-2]
        source = args[-1] if args[-1] is not None else ""
        if inside_spoken_address(offset, len(match), source):
            return match
        return replace_extension(match, args[1], args[2], args[3])

    result = _replace_with_tracking(text, _FILENAME_LITERAL, replace_extension, "filename_extension_spelling", applied_rules)
    result = _replace_with_tracking(
        result, _FILENAME_SPOKEN_DOT, replace_spoken_dot_extension, "filename_extension_spelling", applied_rules
    )
    return _replace_with_tracking(
        result, _FILENAME_SPACED_EXTENSION, replace_spoken_dot_extension, "filename_extension_spelling", applied_rules
    )


# A name that opens with a dot and a lower-case letter (".env", ".gitignore", ".env.example").
_DOT_NAME_PATTERN = js(r"(?<=^|[\s(\[{\"'„“‚‘»«])\.(\p{Ll}[\p{L}\p{N}_-]*(?:\.[\p{L}\p{N}_-]+)*)", "gmu")

# Articles and determiners after which a spoken separator word is a noun of the sentence (Bug 836).
_NOT_AFTER_DETERMINER_SOURCE = (
    "(?<!\\b(?:der|die|das|den|dem|des|ein|eine|einen|einem|einer|eines|kein|keine|keinen|dieser|diese|dieses|jeder|jede|jedes|"
    "the|a|an|this|that|each|every|no|any)\\s+)"
)

_HOME_TILDE_PATTERN = js(r"(?<=^|\s)~(?=[ \t]+\.\p{Ll})", "gmu")
_SPOKEN_HOME_TILDE = {"de": "Tilde", "en": "tilde"}
_DOT = js(r"\.", "g")


def _speak_dot_names(text: str, language: str, applied_rules: list[str]) -> str:
    """Bug 832: a name that opens with a dot is spoken with the dot word of the bound language."""
    if "." not in text:
        return text
    dot = _SPOKEN_NAME_DOT[language]
    with_tilde = (
        _replace_with_tracking(text, _HOME_TILDE_PATTERN, _SPOKEN_HOME_TILDE[language], "dot_name", applied_rules)
        if "~" in text
        else text
    )

    def callback(_match, name, *_rest):
        extension = name.upper()
        if "." not in name and extension in SPELLED_FILE_EXTENSIONS:
            return _spell_out_letters(extension)
        return f"{dot} {js_replace(_DOT, name, f' {dot} ')}"

    return _replace_with_tracking(with_tilde, _DOT_NAME_PATTERN, callback, "dot_name", applied_rules)


_PATH_PREFIX = js(r"(?<![\p{L}\p{N}_.])(?:\.{1,2}\/)+(?=[\p{L}\p{N}_~])", "gu")
_TECHNICAL_SEPARATORS = js(r"[_/\\-]+", "g")
_DIGIT = js(r"\d", "")
_ONLY_HYPHENS = js(r"^-+$", "")
_EXPONENT_LETTER = js(r"[eE]", "")
_UPPER = js(r"\p{Lu}", "u")
_LETTER_OR_NUMBER = js(r"[\p{L}\p{N}]", "u")


def _char_at(source: str, index: int) -> str:
    return source[index] if 0 <= index < len(source) else ""


def _normalize_technical_separators(text: str, applied_rules: list[str]) -> str:
    without_path_prefix = (
        _replace_with_tracking(text, _PATH_PREFIX, "", "technical_identifier_separators", applied_rules)
        if "./" in text
        else text
    )

    def callback(separator, offset, source):
        previous = _char_at(source, offset - 1)
        following = _char_at(source, offset + len(separator))
        if separator == "/" and _DIGIT.test(previous) and _DIGIT.test(following):
            return separator
        if _ONLY_HYPHENS.test(separator):
            if _DIGIT.test(previous) and _DIGIT.test(following):
                return separator
            if not previous and _DIGIT.test(following):
                return separator
            if (
                separator == "-"
                and _EXPONENT_LETTER.test(previous)
                and _DIGIT.test(following)
                and _DIGIT.test(_char_at(source, offset - 2))
            ):
                return separator
        if "_" in separator or "/" in separator or "\\" in separator:
            return " "
        if (
            separator == "-"
            and _UPPER.test(previous)
            and _UPPER.test(following)
            and not _LETTER_OR_NUMBER.test(_char_at(source, offset - 2))
            and not _LETTER_OR_NUMBER.test(_char_at(source, offset + 2))
        ):
            return separator
        if _LETTER_OR_NUMBER.test(previous) and _LETTER_OR_NUMBER.test(following):
            return " "
        return separator

    return _replace_with_tracking(
        without_path_prefix, _TECHNICAL_SEPARATORS, callback, "technical_identifier_separators", applied_rules
    )


_MD_IMAGE = js(r"!\[[^\]]*\](?:\([^)]*\))?", "g")
_MD_LINK = js(r"\[([^\]]+)\]\([^)]*\)", "g")
_MD_INDEX = js(r"(?<=[\p{L}\p{N}_\]])\[([^\]]+)\](?=([\p{L}\p{N}_])?)", "gu")
_MD_STEPS_BEFORE_LATEX = (
    (js(r"\[([^\]]+)\]", "g"), "$1"),
    (js(r"\*{3}([^*]+)\*{3}", "g"), "$1"),
    (js(r"_{3}([^_]+)_{3}", "g"), "$1"),
    (js(r"\*{2}([^*]+)\*{2}", "g"), "$1"),
    (js(r"_{2}([^_]+)_{2}", "g"), "$1"),
    (js(r"(^|\s)\*(?!\s)([^*\n]+)(?<!\s)\*(?=\s|$|[,.;:!?])", "gm"), "$1$2"),
    (js(r"(^|\s)_([^_\n]+)_(?=\s|$|[,.;:!?])", "gm"), "$1$2"),
    (js(r"`([^`]+)`", "g"), "$1"),
    (js(r"```[^\n]*\n[\s\S]*?```", "g"), ""),
    (js(r"~~([^~\n]+?)~~", "g"), ""),
    (js(r"^\s*(?:```|~~~)[^\n]*$", "gm"), ""),
    (js(r"^#{1,6}\s+", "gm"), ""),
    (js(r"^>+\s?", "gm"), ""),
    (js(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$", "gm"), ""),
    (js(r"\$\$[\s\S]*?\$\$", "g"), ""),
)
_MD_LONE_STARS = js(
    r"(?<![\w])\*+(?![\w])(?<!(?<=(?:[\d%€$£°²³)\]]|(?<![\p{L}\p{Nd}_])\p{L})[ \t]+)\*(?=[ \t]+(?:[\d(€$£-]|\p{L}(?![\p{L}\p{Nd}_]))))",
    "gu",
)


def _strip_markdown_formatting(text: str, applied_rules: list[str]) -> str:
    result = js_replace(_MD_IMAGE, text, "")
    result = js_replace(_MD_LINK, result, "$1")
    result = js_replace(
        _MD_INDEX, result, lambda _m, content, word_after, *_r: f" {content} " if word_after is not None else f" {content}"
    )
    for pattern, replacement in _MD_STEPS_BEFORE_LATEX:
        result = js_replace(pattern, result, replacement)
    result = reduce_latex_for_tts(unwrap_inline_math_for_tts(result))
    result = js_replace(_MD_LONE_STARS, result, "")
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


# Single capitals, each with its dot, where the last dot cannot end the sentence (Bug 966).
_DOTTED_CAPITALS_PATTERN = js(r"(?<![\p{L}\p{N}.-])((?:\p{Lu}\.){2,})(?=[ \t]+\p{Ll}|[,;:)\]]|([ \t]*(?:\n|$)))", "gu")


def _spell_dotted_capitals(text: str, applied_rules: list[str]) -> str:
    return _replace_with_tracking(
        text,
        _DOTTED_CAPITALS_PATTERN,
        lambda _m, letters, line_end, *_r: f"{_spell_out_letters(letters.replace('.', ''))}{'.' if line_end is not None else ''}",
        "dotted_capitals_spelling",
        applied_rules,
    )


_ACRONYM_HYPHEN = js(r"\b([A-ZÄÖÜ]{2,8})-(?=\w)", "g")
_ACRONYM = js(r"\b([A-ZÄÖÜ]{2,8})\b", "g")


def _replace_known_acronyms(text: str, applied_rules: list[str]) -> str:
    def hyphen(match, token, *_rest):
        exact = EXACT_TOKEN_SPELLINGS.get((token or "").upper())
        return f"{exact} - " if exact else match

    hyphen_pass = _replace_with_tracking(text, _ACRONYM_HYPHEN, hyphen, "known_acronym_hyphen_compound", applied_rules)

    def standalone(match, token, *_rest):
        normalized = (token or "").upper()
        if normalized in PRESERVE_UPPERCASE_TOKENS or normalized in PLUGIN_TTS_WORD_TOKENS:
            return match
        exact = EXACT_TOKEN_SPELLINGS.get(normalized)
        return exact if exact is not None else match

    return _replace_with_tracking(hyphen_pass, _ACRONYM, standalone, "generic_acronyms", applied_rules)


# Two or more single capitals in a row, apart only by blanks (Bug 964).
_SPELLED_LETTER_RUN_PATTERN = js(r"(?<![\p{L}\p{N}-])\p{Lu}(?:[ \t]+\p{Lu}(?![\p{L}\p{N}]))+(?![\p{L}\p{N}]|-[\p{L}\p{N}])", "gu")
_BLANK_RUN = js(r"[ \t]+", "g")


def _join_spelled_letter_runs(text: str, applied_rules: list[str]) -> str:
    return _replace_with_tracking(
        text,
        _SPELLED_LETTER_RUN_PATTERN,
        lambda run, *_r: js_replace(_BLANK_RUN, run, "-"),
        "spelled_letter_hyphens",
        applied_rules,
    )


_DASH_PAUSE = js(r"[ \t]*[–—][ \t]*", "g")
# The TS callback tests slices of the source around the match; sticky matching at that position
# gives the same answer without copying the rest of the text for every abbreviation.
_SPELLED_LETTER_BEFORE = js(r"\p{Lu}-$", "u")
_SPELLED_LETTER_AFTER = js(r"-\p{Lu}", "uy")
_AT_END = js(r"[ \t]*(?:$|\n)", "y")
_STARTS_UPPER = js(r"^\p{Lu}", "u")
_BEFORE_CAPITAL = js(r"[ \t]+[\"“„]?\p{Lu}", "uy")
_INSREICH = js(r"\bI(?:\s+|-)N(?:\s+|-)S(?:\s+|-)R(?:\s+|-)E(?:\s+|-)I(?:\s+|-)C(?:\s+|-)H\b", "g")
_UNDERSCORE_WORD_BETWEEN = js(rf"(\w)\s+{_NOT_AFTER_DETERMINER_SOURCE}(?:Unterstrich|underscore)\s+(\w)", "gi")
_UNDERSCORE_WORD = js(rf"{_NOT_AFTER_DETERMINER_SOURCE}\b(?:Unterstrich|underscore)\b", "gi")
_HYPHEN_WORD = js(rf"([\p{{L}}\d])\s+{_NOT_AFTER_DETERMINER_SOURCE}(?:Bindestrich|hyphen)\s+([\p{{L}}\d])", "giu")
_SLASH_WORD = js(rf"(\w)\s+{_NOT_AFTER_DETERMINER_SOURCE}(?:Schrägstrich|slash|backslash)\s+(\w)", "giu")
_PHONE_SLASH = js(r"(\d)(?:\s+\/\s*|\s*\/\s+)(\d)", "g")
_WORD_ECHO = js(r"(\b[\p{L}\p{M}]+)\s*\(\s*\1\s*\)", "giu")
_PHRASE_ECHO = js(r"((?:\b[\p{L}\p{M}]+\b[ \t]+){1,7}\b[\p{L}\p{M}]+\b)\s*\(\s*\1\s*\)", "giu")
_EXCLAMATION_END = js(r"!+(?=\s|$)", "g")
_EXCLAMATION = js(r"!(?!=)", "g")


def _expand_abbreviations(text: str, bound_language: Optional[str], applied_rules: list[str]) -> str:
    result = text
    for rule in DOTTED_ABBREVIATION_REPLACEMENTS:
        if (rule.lang != bound_language) if bound_language else rule.bound_only:
            continue

        def callback(*args, _rule=rule):
            match = args[0] if args[0] is not None else ""
            expanded = _rule.replacement(*args) if callable(_rule.replacement) else _rule.replacement
            offset = args[-2]
            source = args[-1] if args[-1] is not None else ""
            rest_start = offset + len(match)
            # Bug 964: a letter of a spelled word is no abbreviation.
            if _SPELLED_LETTER_BEFORE.test(source[max(0, offset - 2):offset]) or (
                _SPELLED_LETTER_AFTER.match_at(source, rest_start) is not None
            ):
                return match
            at_end = _AT_END.match_at(source, rest_start) is not None
            if _rule.leads_next_word and at_end and _STARTS_UPPER.test(match):
                return match
            before_capital = _rule.closes_sentence and _BEFORE_CAPITAL.match_at(source, rest_start) is not None
            if match.endswith(".") and not expanded.endswith(".") and (at_end or before_capital):
                return f"{expanded}."
            return expanded

        result = _replace_with_tracking(result, rule.pattern, callback, rule.label, applied_rules)
    return result


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

    shushing_skipped = skip_shushing_for_tts(result)
    if shushing_skipped != result:
        result = shushing_skipped
        applied_rules.append("shushing_skip")

    result = _replace_with_tracking(result, _DASH_PAUSE, " - ", "dash_pause", applied_rules)
    result = _normalize_technical_separators(result, applied_rules)
    result = _strip_markdown_formatting(result, applied_rules)
    result = _expand_abbreviations(result, bound_language, applied_rules)

    result = _replace_transit_stop_codes_with_labels(result, applied_rules)
    result = _replace_known_mixed_case_tokens(result, applied_rules)
    result = _spell_dotted_capitals(result, applied_rules)

    result = _normalize_spoken_file_extensions(result, bound_language, applied_rules)
    if bound_language:
        result = _speak_dot_names(result, bound_language, applied_rules)

    result = js_replace(_INSREICH, result, "INSREICH")
    for token in PLUGIN_TTS_WORD_TOKENS:
        result = js_replace(js(r"\b" + "(?:\\s+|-)".join(token) + r"\b", "g"), result, token)

    result = _replace_known_acronyms(result, applied_rules)

    result = js_replace(_UNDERSCORE_WORD_BETWEEN, result, "$1 $2")
    result = js_replace(_UNDERSCORE_WORD, result, "")
    result = js_replace(_HYPHEN_WORD, result, "$1 $2")
    result = js_replace(_SLASH_WORD, result, "$1 $2")
    result = js_replace(_PHONE_SLASH, result, "$1 $2")
    result = js_replace(_WORD_ECHO, result, "$1")
    result = js_replace(_PHRASE_ECHO, result, "$1")

    result = _join_spelled_letter_runs(result, applied_rules)
    result = finalize_optimized_text(js_replace(_EXCLAMATION, js_replace(_EXCLAMATION_END, result, "."), ""))

    result = _restore_direct_urls(result, urls)
    return result, list(dict.fromkeys(applied_rules))
