"""Cheap lexical language check for requests without a de/en language (``Auto`` / none).

The Genesis binding (``language_binding.py``) votes on notation only: dd.mm.yyyy,
``1.234,56``, ``29,90 €``, ``14:30 Uhr``. Dutch, Polish, Italian, French, Turkish,
Spanish, Portuguese and Russian write most of these as well, so notation alone would
bind their text to German and splice German words in. This module counts frequent
function words of the Latin-script languages OmniVoice speaks and reports which one a
text is written in, so the server can

* refuse a notation binding when the words say another language, and
* bind de/en from the words when there is no notation anchor at all.

It is deliberately conservative: no decision without evidence, ties are undecided,
and text that is mostly in a non-Latin script is never German or English.
"""

from __future__ import annotations

from typing import Optional

import regex

# Frequent, mostly language-specific function words (lower case). Overlaps between
# languages are fine: a word counts for every list it is in, the strict maximum wins.
_STOPWORDS: dict[str, frozenset[str]] = {
    "de": frozenset(
        """
        der die das dem den des ein eine einen einem einer eines und oder aber ist sind
        war waren wird werden wurde nicht kein keine mit von zu zum zur auf für fuer über
        ueber unter bei nach aus auch noch schon nur sehr ich du er sie wir ihr mein dein
        sein unser euer dass daß wenn weil wie wo hier heute morgen gestern uhr am im um
        vom bis ab es sich haben hat habe kostet kosten siehe bitte danke gibt kann können
        koennen muss müssen soll sollen will wollen jetzt dann denn doch also sowie etwa
        ca ungefähr
        """.split()
    ),
    "en": frozenset(
        """
        the a an and or but is are was were be been being not no with of to for on at by
        from in into about this that these those it its he she we they you your our their
        i my me his her him them there here what which who when where why how will would
        can could should must have has had do does did costs cost weighs see please thanks
        today tomorrow yesterday than then also only very just
        """.split()
    ),
    "nl": frozenset(
        """
        de het een en of maar is zijn was waren wordt worden niet geen met van voor op aan
        bij naar uit ook nog al alleen zeer ik jij je hij zij wij jullie mijn uw onze dat
        als omdat hoe waar hier vandaag morgen gisteren kost kosten om er dan deze die dit
        zal zou kan moet heeft hebben wel
        """.split()
    ),
    "fr": frozenset(
        """
        le la les un une des du de et ou mais est sont était être pas ne avec pour sur dans
        par au aux ce cette ces il elle nous vous ils elles je tu mon ton son notre votre
        que qui quand où comment ici aujourd'hui demain hier coûte coute lieu aura sera
        très tres aussi encore déjà deja
        """.split()
    ),
    "es": frozenset(
        """
        el la los las un una unos unas y o pero es son era está están estar no con de del
        para por en al este esta estos estas él ella nosotros vosotros ellos yo tú mi tu su
        nuestro que quien cuando donde cómo aquí hoy mañana ayer cuesta será muy también
        ya reunión
        """.split()
    ),
    "it": frozenset(
        """
        il lo la gli le un uno una e o ma è sono era essere non con di del della dei delle
        per su nel nella al alla questo questa quello che chi quando dove come qui oggi
        domani ieri costa sarà molto anche già riunione
        """.split()
    ),
    "pt": frozenset(
        """
        o a os as um uma uns umas e ou mas é são era está estão não com de do da dos das
        para por em no na nos nas ao este esta isso que quem quando onde como aqui hoje
        amanhã ontem custa será muito também já reunião
        """.split()
    ),
    "pl": frozenset(
        """
        i w z na do że nie jest są był była się to o od za po jak ale lub czy ten ta te
        tego jego jej my wy oni ja ty mój twój nasz kiedy gdzie tutaj dzisiaj jutro
        wczoraj kosztuje będzie bardzo także już godzinie odbędzie spotkanie
        """.split()
    ),
    "tr": frozenset(
        """
        ve veya ama bir bu şu o için ile da de ki değil var yok çok daha gibi kadar olarak
        olan ben sen biz siz onlar benim senin nerede ne zaman nasıl burada bugün yarın dün
        saat tarihinde fiyatı toplantı olacak
        """.split()
    ),
}

_URL_RE = regex.compile(r"(?:https?://|www\.)\S+|\S+@\S+\.\S+")
_WORD_RE = regex.compile(r"[\p{L}\p{M}]+(?:['’][\p{L}]+)?")
_LETTER_RE = regex.compile(r"\p{L}")
_LATIN_RE = regex.compile(r"\p{Script=Latin}")


def latin_share(text: str) -> float:
    letters = len(_LETTER_RE.findall(text))
    if not letters:
        return 1.0
    return len(_LATIN_RE.findall(text)) / letters


def word_language_scores(text: str) -> dict[str, int]:
    words = [w.lower() for w in _WORD_RE.findall(_URL_RE.sub(" ", text or ""))]
    return {lang: sum(1 for w in words if w in vocab) for lang, vocab in _STOPWORDS.items()}


def detect_word_language(text: str) -> Optional[str]:
    """The language whose function words strictly dominate, '' for a non-Latin script
    text, None when the words do not decide."""
    if latin_share(text or "") < 0.5:
        return ""
    scores = word_language_scores(text)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    (best, best_score), (_, second_score) = ranked[0], ranked[1]
    if best_score == 0 or best_score == second_score:
        return None
    return best


def words_contradict(text: str, language: str) -> bool:
    """True when the text is written in a script / language other than ``language``.

    A notation binding is kept when the words agree, when they do not decide, or when
    ``language`` ties for the highest score; it is dropped only on positive evidence
    for another language.
    """
    if latin_share(text or "") < 0.5:
        return True
    scores = word_language_scores(text)
    own = scores.get(language, 0)
    return any(score > own for lang, score in scores.items() if lang != language)
