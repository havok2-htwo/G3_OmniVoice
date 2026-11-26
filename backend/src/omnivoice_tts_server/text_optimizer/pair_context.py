"""Port of Genesis3 ``src/utils/ttsLanguageOptimizer/pairContext.ts`` (Bug 841).

The words around a colon, slash or hyphen pair that decide its reading: German data for the count,
ratio, blood pressure and score rules of the notation prepass and for the month, fraction, clock
and span passes of de.py. Each source is the body of a lookaround for a ``u`` pattern.
"""

from __future__ import annotations

from .jsre import js, js_replace

_REGEX_SPECIAL = js(r"[.*+?^${}()|[\]\\]", "g")


def _escape_word(word: str) -> str:
    return js_replace(_REGEX_SPECIAL, word, "\\$&")


def _alternation(words) -> str:
    return f"(?:{'|'.join(_escape_word(word) for word in words)})"


def _either_case_initial(words) -> str:
    """A label written at the start of a sentence or inside one ("Maßstab", "maßstab")."""
    return "(?:" + "|".join(f"[{word[0].upper()}{word[0].lower()}]{_escape_word(word[1:])}" for word in words) + ")"


# The words between a label and its pair: at most two, none of them the end of a sentence.
_WORDS_BETWEEN = r"(?:[ \t]+[^\s.!?]+){0,2}"


def _word_within_before(source: str) -> str:
    return rf"(?<![\p{{L}}\d]){source}(?![\p{{L}}\d])[^\s.!?]*{_WORDS_BETWEEN}[ \t]+"


def _word_within_after(source: str) -> str:
    return rf"{_WORDS_BETWEEN}[ \t]+{source}(?![\p{{L}}\d])"


_SLASH_CODE_LABELS = [
    "HJ", "Halbjahr", "KW", "Kalenderwoche", "FJ", "GJ", "FY", "Geschäftsjahr", "Wirtschaftsjahr", "WS", "WiSe", "SoSe",
    "Wintersemester", "Sommersemester", "Nr.", "Nummer", "Heft", "Ausgabe", "Band", "Jahrgang", "Quartal", "Az.",
    "Aktenzeichen", "Saison", "Schuljahr",
]
_OUT_OF_LABELS = [
    "Bewertung", "Note", "Wertung", "Stufe", "Level", "Track", "Titel", "Folge", "Episode", "Teil", "Kapitel", "Seite", "Frage",
    "Aufgabe", "Runde", "Schritt", "Spiel",
]
_OUT_OF_NOUNS = ["Punkte", "Punkten", "Sterne", "Sternen", "Stimmen"]
_ORDINAL_PREPOSITIONS = [
    "in", "an", "auf", "bei", "mit", "nach", "seit", "von", "zu", "aus", "vor", "unter", "über", "zwischen", "gegenüber",
    "entgegen", "laut", "wegen", "trotz", "während", "dank", "gemäß",
]
_BLOOD_PRESSURE_LABELS = ["Blutdruck", "RR"]
_FRACTION_COMPOUNDS = ["Takt"]
_RATIO_LABELS = ["Maßstab", "Verhältnis", "Faktor", "Quote", "Übersetzung", "Untersetzung", "Skala", "Proportion"]
_SCORE_LABELS = [
    "Spielstand", "Endstand", "Zwischenstand", "Halbzeitstand", "Halbzeit", "Torverhältnis", "Ergebnis", "Endergebnis", "endete",
    "endet",
]
_SCORE_VERBS = ["gewonnen", "verloren", "geschlagen", "gesiegt", "unterlegen", "endete", "endet"]
_DURATION_LABELS = [
    "Pace", "Runde", "Rundenzeit", "Bestzeit", "Laufzeit", "Fahrzeit", "Spielzeit", "Gesamtzeit", "Zielzeit", "Marathon",
    "Halbmarathon", "Split", "Rekord", "Dauer", "dauert", "dauerte",
]
_DURATION_VERBS = ["gedauert", "gebraucht", "benötigt", "gelaufen", "gefahren", "geschwommen"]
_CLOCK_PREPOSITIONS = ["um", "gegen", "ab", "bis", "von", "zwischen"]

_SLASH_CODE_LABEL_SOURCE = _alternation(_SLASH_CODE_LABELS)

GERMAN_PAIR_CONTEXT = {
    "codeBefore": (
        rf"(?<![\p{{L}}\d])(?:{_SLASH_CODE_LABEL_SOURCE}|\p{{Lu}}|(?<=(?<![\p{{L}}\d]){_SLASH_CODE_LABEL_SOURCE}"
        r"(?:[ \t]*:)?[ \t]*(?:\d{1,2}|[IVXL]{1,5})[ \t]+)\p{Lu}\p{L}{1,3})(?:[ \t]*:)?[ \t]*"
    ),
    "inchAfter": '["″]',
    "countBefore": (
        rf"(?<![\p{{L}}\d]){_alternation(_OUT_OF_LABELS)}(?:[ \t]*:)?[ \t]+(?:{_alternation(_ORDINAL_PREPOSITIONS)}[ \t]+)?"
    ),
    "countAfter": rf"[ \t]+{_alternation(_OUT_OF_NOUNS)}(?![\p{{L}}\d])",
    "compoundAfter": rf"(?:-|[ \t]+){_alternation(_FRACTION_COMPOUNDS)}(?![\p{{L}}\d])",
    "bloodPressureBefore": _word_within_before(f"{_alternation(_BLOOD_PRESSURE_LABELS)}\\p{{L}}*"),
    "ratioBefore": rf"{_either_case_initial(_RATIO_LABELS)}(?![\p{{L}}\d])(?:[ \t]*:)?[ \t]+(?:\p{{Ll}}+[ \t]+)?",
    "scoreBefore": _word_within_before(_either_case_initial(_SCORE_LABELS)),
    "scoreAfter": _word_within_after(_alternation(_SCORE_VERBS)),
    "durationBefore": _word_within_before(_either_case_initial(_DURATION_LABELS)),
    "durationAfter": _word_within_after(_alternation(_DURATION_VERBS)),
    "datePrefixBefore": (
        r"(?<![\p{L}\d])(?:[Aa]m|[Dd]em|[Dd]en|[Vv]om|[Zz]um|[Vv]on|[Bb]is|[Aa]b|[Ss]eit|[Nn]ach|[Vv]or|[Ff]ür)[ \t]+"
    ),
    "clockPrepositionBefore": rf"(?<![\p{{L}}\d]){_either_case_initial(_CLOCK_PREPOSITIONS)}[ \t]+",
}
