"""Port of Genesis3 ``src/utils/ttsLanguageOptimizer/spokenAbbreviations.ts`` (Bug 870).

The abbreviations the floor writes out, per language, each with the kind that says what the
post-processor does with it, and the weekdays whose word depends on what stands around them.
Spelling is exact, case included. The chunker-only lists of the TS module are not ported
(G3_OmniVoice keeps its own sentence splitter).

Kinds: ``title`` opens a name ("Dr."); ``word`` runs on ("bzgl."); ``closes`` may end a sentence
("etc."), the dot comes back before a capital; ``number`` only before a number ("Tab. 2").
"""

from __future__ import annotations

from .jsre import js

SPOKEN_ABBREVIATIONS_DE: tuple[tuple[str, str, str], ...] = (
    ("Dipl.-Ing.", "Diplomingenieur", "title"),
    ("Dr.-Ing.", "Doktor Ingenieur", "title"),
    ("Dr. med.", "Doktor med", "title"),
    ("Dr.", "Doktor", "title"),
    ("Prof.", "Professor", "title"),
    ("Hrn.", "Herrn", "title"),
    ("Hr.", "Herr", "title"),
    ("Jhdt.", "Jahrhundert", "closes"),
    ("Jhd.", "Jahrhundert", "closes"),
    ("Feb.", "Februar", "closes"),
    ("Apr.", "April", "closes"),
    ("Aug.", "August", "closes"),
    ("Sept.", "September", "closes"),
    ("Sep.", "September", "closes"),
    ("Okt.", "Oktober", "closes"),
    ("Nov.", "November", "closes"),
    ("Dez.", "Dezember", "closes"),
    ("Bd.", "Band", "number"),
    ("Abb.", "Abbildung", "number"),
    ("Aufl.", "Auflage", "closes"),
    ("Ausg.", "Ausgabe", "closes"),
    ("Hrsg.", "Herausgeber", "word"),
    ("i. V. m.", "in Verbindung mit", "word"),
    ("i.V.m.", "in Verbindung mit", "word"),
    ("bzgl.", "bezüglich", "word"),
    ("Bzgl.", "Bezüglich", "word"),
    ("Betr.", "Betreff", "word"),
    ("ggü.", "gegenüber", "word"),
    ("lt.", "laut", "word"),
    ("eigtl.", "eigentlich", "word"),
    ("einschl.", "einschließlich", "word"),
    ("zzt.", "zurzeit", "word"),
    ("z.Zt.", "zurzeit", "word"),
    ("z. Zt.", "zurzeit", "word"),
    ("z. Z.", "zurzeit", "word"),
    ("z. T.", "zum Teil", "word"),
    ("z.T.", "zum Teil", "word"),
    ("gez.", "gezeichnet", "word"),
    ("s. o.", "siehe oben", "closes"),
    ("s.o.", "siehe oben", "closes"),
    ("s. u.", "siehe unten", "closes"),
    ("s.u.", "siehe unten", "closes"),
    ("v. a.", "vor allem", "word"),
    ("v.a.", "vor allem", "word"),
    ("a. D.", "außer Dienst", "word"),
    ("a.D.", "außer Dienst", "word"),
    ("etc.", "et cetera", "closes"),
    ("Pos.", "Position", "number"),
    ("Kto.", "Konto", "number"),
    ("Ziff.", "Ziffer", "number"),
    ("Stck.", "Stück", "word"),
    ("Pck.", "Päckchen", "word"),
    ("Bhf.", "Bahnhof", "word"),
)

SPOKEN_ABBREVIATIONS_EN: tuple[tuple[str, str, str], ...] = (
    ("Mrs.", "Missus", "title"),
    ("Mr.", "Mister", "title"),
    ("Ms.", "Miz", "title"),
    ("Prof.", "Professor", "title"),
    ("Blvd.", "Boulevard", "closes"),
    ("Rd.", "Road", "closes"),
    ("Apt.", "Apartment", "number"),
    ("etc.", "et cetera", "closes"),
    ("cf.", "compare", "word"),
    ("Fig.", "Figure", "number"),
    ("fig.", "figure", "number"),
    ("ext.", "extension", "number"),
)

# The number behind an abbreviation is a word by now, the language layer ran first.
GERMAN_NUMBER_WORD_SOURCE = (
    r"(?!(?:und|ein)(?![\p{L}]))(?:null|eins|ein|zwei|drei|vier|fünf|sechs|sech|sieben|sieb|acht|neun|zehn|elf|zwölf"
    r"|zwanzig|ßig|zig|hundert|tausend|und)+(?![\p{L}])"
)
ENGLISH_NUMBER_WORD_SOURCE = (
    r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|(?:thir|four|fif|six|seven|eigh|nine)teen"
    r"|(?:twen|thir|for|fif|six|seven|eigh|nine)ty|hundred|thousand)(?![\p{L}])"
)

_GERMAN_WEEKDAYS = {
    "Mo": "Montag", "Di": "Dienstag", "Mi": "Mittwoch", "Do": "Donnerstag", "Fr": "Freitag", "Sa": "Samstag", "So": "Sonntag",
}
_ENGLISH_WEEKDAYS = {
    "Mon": "Monday", "Tue": "Tuesday", "Wed": "Wednesday", "Thu": "Thursday", "Fri": "Friday", "Sat": "Saturday",
}

_GERMAN_DAY_FOR_CERTAIN = js(r"(?:^|[\s(])(?:[Aa]m|[Jj]ede[nm])[ \t]+$", "")
_GERMAN_DAY_RANGE_BEFORE = js(r"(?:\p{L}tag|\b(?:Mo|Di|Mi|Do|Sa|So)\.)[ \t]+bis[ \t]+$", "u")
_GERMAN_DAY_RANGE_AFTER = js(r"^[ \t]+bis[ \t]+(?:Mo|Di|Mi|Do|Fr|Sa|So)\.", "")
_GERMAN_DAY_BEFORE = js(
    r"(?:^|[\s(])(?:[Aa]b|[Bb]is|[Vv]om|[Vv]on)[ \t]+$|(?:tag|\b(?:Mo|Di|Mi|Do|Fr|Sa|So)\.)(?:,| und| oder| bis)[ \t]+$", ""
)
_ENGLISH_DAY_BEFORE = js(
    r"(?:^|[\s(])(?:[Oo]n|[Bb]y|[Ff]rom|[Uu]ntil|[Tt]ill|[Tt]o|[Ee]very)[ \t]+$"
    r"|(?:day|\b(?:Mon|Tue|Wed|Thu|Fri|Sat)\.)(?:,| and| or| to)[ \t]+$",
    "",
)
_FOLLOWED_BY_CAPITAL = js(r"^[ \t]+\p{Lu}", "u")
_NAME_PARTICLE_AFTER = js(r"^[ \t]+(?:von|vom|van|de|zu|zur|ter|ten)[ \t]+\p{Lu}", "u")
_GERMAN_NUMBER_BEFORE = js(rf"(?:\d|\b{GERMAN_NUMBER_WORD_SOURCE})[ \t]+$", "u")


def speak_german_day(match: str, day: str, offset: int, source: str) -> str:
    """Only what the surroundings settle is a day: "am Mo.", "Mo. bis Fr.", "bis Fr. im Urlaub"."""
    before = source[:offset]
    after = source[offset + len(match):]
    capital = _FOLLOWED_BY_CAPITAL.test(after)
    as_day = f"{_GERMAN_WEEKDAYS[day]}{'.' if capital else ''}"
    if _GERMAN_NUMBER_BEFORE.test(before) or _NAME_PARTICLE_AFTER.test(after):
        return match
    if _GERMAN_DAY_FOR_CERTAIN.test(before) or _GERMAN_DAY_RANGE_BEFORE.test(before) or _GERMAN_DAY_RANGE_AFTER.test(after):
        return as_day
    return as_day if not capital and _GERMAN_DAY_BEFORE.test(before) else match


def speak_english_day(match: str, day: str, offset: int, source: str) -> str:
    after = source[offset + len(match):]
    if not _ENGLISH_DAY_BEFORE.test(source[:offset]):
        return match
    return f"{_ENGLISH_WEEKDAYS[day]}{'.' if _FOLLOWED_BY_CAPITAL.test(after) else ''}"
