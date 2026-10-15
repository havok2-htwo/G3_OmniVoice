"""JavaScript RegExp compatibility layer on top of the ``regex`` module.

The text optimizer is a port of TypeScript code whose behaviour depends on
ECMAScript regex semantics. The patterns are kept in their original JS source
form and translated here, so the port can be diffed against the TS code line by
line. Differences that are handled:

* ``\\d``, ``\\w`` and ``\\b`` are ASCII-only in JS, even with the ``u`` flag.
* ``\\s`` is the ECMAScript WhiteSpace + LineTerminator set (it contains
  U+FEFF and not U+0085, unlike Python's).
* ``.`` excludes the four JS line terminators, ``^``/``$`` are only line
  anchors with the ``m`` flag, and ``$`` never matches before a final newline.
* ``\\p{..}`` and ``\\u{..}`` are only escapes with the ``u`` flag.
* Sticky (``y``) matching uses ``pattern.match(text, pos)``; lookbehind can see
  text before ``pos`` exactly like JS ``lastIndex``.
* ``String.prototype.replace`` is emulated by :func:`js_replace`, including
  the callback argument layout ``(match, *groups, offset, source)`` with
  ``None`` for groups that did not participate, and ``$1``/``$&``/``$$``
  replacement templates.
* ``/i`` follows ECMAScript case-insensitivity instead of Python's: without
  ``u`` two characters are equal when their single-character upper-case forms
  are equal and no non-ASCII character maps onto ASCII (so ``İ``, ``ı``, ``ſ``,
  ``K`` and ``ẞ`` never match ``i``/``s``/``k``/``ß``); with ``u`` simple case
  folding is used (``ſ``~``s``, ``K``~``k``, ``ẞ``~``ß``, never ``İ``/``ı``).
  Patterns are compiled without ``regex.IGNORECASE``: literals and classes are
  expanded to their JS case-equivalents, backreferences compare case-folded.

Known, accepted gaps (not exercised by the optimizer): captures inside a
repeated group are not reset per iteration as in JS; offsets are code points
instead of UTF-16 code units, so a callback that looks at ``source[offset-1]``
sees a whole astral letter where JS sees a lone surrogate (the JS result is the
buggier one there); ``\\w``/``\\b`` under ``/iu`` stay ASCII (JS adds ``ſ`` and
``K``).

Every match runs with ``concurrent=True`` (the GIL is released while the regex
engine scans) and honours :func:`time_budget`: once the budget is spent the next
match raises :class:`TimeoutError`, so a pathological input cannot hold a worker
thread for minutes.
"""

from __future__ import annotations

import contextvars
import math
import time
from contextlib import contextmanager
from functools import lru_cache
from typing import Callable, Iterator

import regex

__all__ = [
    "JS_WHITESPACE",
    "time_budget",
    "JsRegExp",
    "js",
    "js_number",
    "js_replace",
    "js_trim",
    "js_replace_first_literal",
]

# ECMAScript WhiteSpace + LineTerminator (what \s matches and String.trim strips).
JS_WHITESPACE = (
    "\t\n\x0b\x0c\r \xa0         "
    "       　﻿"
)
_WS_CLASS_BODY = "\\t\\n\\x0b\\x0c\\r \\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000\\ufeff"
_LINE_TERMINATORS_BODY = "\\n\\r\\u2028\\u2029"
_WORD_BODY = "A-Za-z0-9_"
_DIGIT_BODY = "0-9"

_WORD_BOUNDARY = "(?:(?<=[A-Za-z0-9_])(?![A-Za-z0-9_])|(?<![A-Za-z0-9_])(?=[A-Za-z0-9_]))"
_NOT_WORD_BOUNDARY = "(?:(?<=[A-Za-z0-9_])(?=[A-Za-z0-9_])|(?<![A-Za-z0-9_])(?![A-Za-z0-9_]))"

_QUANTIFIER_RE = regex.compile(r"\{[0-9]+(?:,[0-9]*)?\}")
_HEX4_RE = regex.compile(r"[0-9A-Fa-f]{4}")
_HEX2_RE = regex.compile(r"[0-9A-Fa-f]{2}")


class JsRegexTranslationError(ValueError):
    pass


# --- time budget ---------------------------------------------------------------------------

_DEADLINE: contextvars.ContextVar[float | None] = contextvars.ContextVar("jsre_deadline", default=None)


@contextmanager
def time_budget(seconds: float | None):
    """Bound the regex work done inside the block (per thread / context).

    ``None`` or a non-positive value disables the budget. When it is spent, the next
    match raises :class:`TimeoutError` (the ``regex`` module's own timeout error).
    """
    if seconds is None or seconds <= 0:
        yield
        return
    token = _DEADLINE.set(time.monotonic() + float(seconds))
    try:
        yield
    finally:
        _DEADLINE.reset(token)


def _timeout() -> float | None:
    deadline = _DEADLINE.get()
    if deadline is None:
        return None
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("regex time budget exhausted")
    return remaining


# --- ECMAScript case-insensitivity ---------------------------------------------------------


def _js_canonicalize_non_unicode(ch: str) -> str:
    """ECMAScript Canonicalize(ch) for /i without u (toUpperCase, single unit, no ->ASCII)."""
    upper = ch.upper()
    if len(upper) != 1:
        return ch
    if ord(ch) >= 128 and ord(upper) < 128:
        return ch
    return upper


def _simple_case_fold(ch: str) -> str:
    """Unicode simple case folding (CaseFolding.txt C+S), what /iu canonicalizes with."""
    folded = ch.casefold()
    if len(folded) == 1:
        return folded
    lower = ch.lower()
    if len(lower) == 1:
        return lower
    return ch


@lru_cache(maxsize=2)
def _case_groups(unicode_mode: bool) -> tuple[dict[str, str], str]:
    """char -> all chars JS /i treats as equal to it (only chars with a partner), plus
    the concatenation of those chars for one-pass class membership scans."""
    canonicalize = _simple_case_fold if unicode_mode else _js_canonicalize_non_unicode
    # Without u, JS compares UTF-16 code units: astral letters never fold.
    limit = 0x1FFFF if unicode_mode else 0xFFFF
    groups: dict[str, list[str]] = {}
    for cp in range(limit + 1):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        ch = chr(cp)
        groups.setdefault(canonicalize(ch), []).append(ch)
    mapping = {ch: "".join(group) for group in groups.values() if len(group) > 1 for ch in group}
    return mapping, "".join(mapping)


def _class_char(ch: str) -> str:
    cp = ord(ch)
    return f"\\u{cp:04x}" if cp <= 0xFFFF else f"\\U{cp:08x}"


def _case_variants_class(ch: str, unicode_mode: bool) -> str | None:
    group = _case_groups(unicode_mode)[0].get(ch)
    if group is None:
        return None
    return "[" + "".join(_class_char(c) for c in group) + "]"


def _class_case_closure(body_text: str, unicode_mode: bool) -> str:
    """Extra class body that adds the JS case-equivalents of every member of ``[body]``."""
    mapping, cased = _case_groups(unicode_mode)
    member = regex.compile(f"[{body_text}]", regex.V0)
    matched = set(member.findall(cased))
    extras = {other for ch in matched for other in mapping[ch] if other not in matched}
    return "".join(_class_char(c) for c in sorted(extras))


_SINGLE_CHAR_FRAGMENT = regex.compile(r"\\u([0-9A-Fa-f]{4})|\\U([0-9A-Fa-f]{8})|\\x([0-9A-Fa-f]{2})|([^\\])")
_BACKREF_FRAGMENT = regex.compile(r"\\[1-9][0-9]*|\(\?P=[^)]+\)")


def _fragment_char(fragment: str) -> str | None:
    m = _SINGLE_CHAR_FRAGMENT.fullmatch(fragment)
    if m is None:
        return None
    if m.group(4) is not None:
        return m.group(4)
    return chr(int(m.group(1) or m.group(2) or m.group(3), 16))


def _escape_literal(ch: str) -> str:
    if ch.isalnum() or ch == "_":
        return ch
    return "\\" + ch


def _read_braced(source: str, start: int) -> tuple[str, int]:
    """Return the text inside ``{...}`` starting at ``source[start] == '{'``."""
    end = source.index("}", start)
    return source[start + 1 : end], end + 1


def _translate_escape(source: str, i: int, unicode_mode: bool, in_class: bool) -> tuple[str, int, str | None]:
    """Translate the escape at ``source[i] == '\\'``.

    Returns (python_fragment, next_index, negated_class_body). For a negated
    shorthand inside a class (\\S, \\W, \\D) the fragment is empty and the
    complement body is returned separately.
    """
    if i + 1 >= len(source):
        raise JsRegexTranslationError("dangling backslash")
    c = source[i + 1]
    j = i + 2
    if c == "d":
        return (_DIGIT_BODY if in_class else f"[{_DIGIT_BODY}]"), j, None
    if c == "D":
        return ("", j, _DIGIT_BODY) if in_class else (f"[^{_DIGIT_BODY}]", j, None)
    if c == "w":
        return (_WORD_BODY if in_class else f"[{_WORD_BODY}]"), j, None
    if c == "W":
        return ("", j, _WORD_BODY) if in_class else (f"[^{_WORD_BODY}]", j, None)
    if c == "s":
        return (_WS_CLASS_BODY if in_class else f"[{_WS_CLASS_BODY}]"), j, None
    if c == "S":
        return ("", j, _WS_CLASS_BODY) if in_class else (f"[^{_WS_CLASS_BODY}]", j, None)
    if c == "b":
        return ("\\x08" if in_class else _WORD_BOUNDARY), j, None
    if c == "B":
        if in_class:
            return "B", j, None
        return _NOT_WORD_BOUNDARY, j, None
    if c in ("p", "P"):
        if unicode_mode and j < len(source) and source[j] == "{":
            name, j2 = _read_braced(source, j)
            return f"\\{c}{{{name}}}", j2, None
        return c, j, None
    if c == "u":
        if unicode_mode and j < len(source) and source[j] == "{":
            hexdigits, j2 = _read_braced(source, j)
            return f"\\U{int(hexdigits, 16):08x}", j2, None
        if _HEX4_RE.match(source, j):
            return f"\\u{source[j:j + 4]}", j + 4, None
        return "u", j, None
    if c == "x":
        if _HEX2_RE.match(source, j):
            return f"\\x{source[j:j + 2]}", j + 2, None
        return "x", j, None
    if c in "nrtfv":
        return "\\" + c, j, None
    if c == "0" and not (j < len(source) and source[j].isdigit()):
        return "\\x00", j, None
    if c.isdigit():
        if in_class:
            # Octal-ish escapes inside classes are not used by the optimizer.
            raise JsRegexTranslationError("numeric escape in class")
        k = j
        while k < len(source) and source[k].isdigit():
            k += 1
        return "\\" + source[i + 1 : k], k, None
    if c == "k" and j < len(source) and source[j] == "<":
        end = source.index(">", j)
        return f"(?P={source[j + 1:end]})", end + 1, None
    if c == "c" and j < len(source) and source[j].isalpha():
        return f"\\x{ord(source[j]) % 32:02x}", j + 1, None
    # Identity escape.
    return _escape_literal(c), j, None


def _translate_class(source: str, i: int, unicode_mode: bool, ignore_case: bool = False) -> tuple[str, int]:
    """Translate the character class starting at ``source[i] == '['``."""
    j = i + 1
    negated = False
    if j < len(source) and source[j] == "^":
        negated = True
        j += 1
    body: list[str] = []
    complements: list[str] = []
    while True:
        if j >= len(source):
            raise JsRegexTranslationError("unterminated character class")
        ch = source[j]
        if ch == "]":
            j += 1
            break
        if ch == "\\":
            fragment, j, complement = _translate_escape(source, j, unicode_mode, in_class=True)
            if complement is not None:
                complements.append(complement)
            else:
                body.append(fragment)
            continue
        if ch == "[":
            body.append("\\[")
        elif ch == "^":
            body.append("\\^")
        elif unicode_mode and ord(ch) > 0xFFFF:
            body.append(ch)
        else:
            body.append(ch)
        j += 1

    body_text = "".join(body)
    if ignore_case and body_text:
        body_text += _class_case_closure(body_text, unicode_mode)
    if not complements:
        if not body_text:
            # JS: [] never matches, [^] matches any character.
            return ("(?s:.)" if negated else "(?!)"), j
        return f"[{'^' if negated else ''}{body_text}]", j

    if not negated:
        alternatives = [f"[^{comp}]" for comp in complements]
        if body_text:
            alternatives.insert(0, f"[{body_text}]")
        return "(?:" + "|".join(alternatives) + ")", j
    # Complement of (body ∪ ¬C1 ∪ ¬C2 …) = ¬body ∩ C1 ∩ C2 …
    parts = []
    if body_text:
        parts.append(f"(?![{body_text}])")
    for comp in complements[:-1]:
        parts.append(f"(?=[{comp}])")
    parts.append(f"[{complements[-1]}]")
    return "(?:" + "".join(parts) + ")", j


@lru_cache(maxsize=None)
def translate(source: str, flags: str) -> str:
    unicode_mode = "u" in flags or "v" in flags
    ignore_case = "i" in flags
    multiline = "m" in flags
    dotall = "s" in flags
    out: list[str] = []
    i = 0
    n = len(source)
    while i < n:
        ch = source[i]
        if ch == "\\":
            fragment, i, _ = _translate_escape(source, i, unicode_mode, in_class=False)
            if ignore_case:
                if _BACKREF_FRAGMENT.fullmatch(fragment):
                    # JS /i compares a backreference case-insensitively.
                    fragment = f"(?i:{fragment})"
                elif fragment.startswith(("\\p{", "\\P{")):
                    # A property escape is a set too: /iu matches its case-equivalents
                    # (\\p{Lu} matches lower-case letters under /iu).
                    fragment = f"[{fragment}{_class_case_closure(fragment, unicode_mode)}]"
                else:
                    char = _fragment_char(fragment)
                    variants = _case_variants_class(char, unicode_mode) if char is not None else None
                    if variants is not None:
                        fragment = variants
            out.append(fragment)
            continue
        if ch == "[":
            fragment, i = _translate_class(source, i, unicode_mode, ignore_case)
            out.append(fragment)
            continue
        if ch == ".":
            out.append("(?s:.)" if dotall else f"[^{_LINE_TERMINATORS_BODY}]")
        elif ch == "^":
            out.append(f"(?<![^{_LINE_TERMINATORS_BODY}])" if multiline else "(?<!(?s:.))")
        elif ch == "$":
            out.append(f"(?![^{_LINE_TERMINATORS_BODY}])" if multiline else "(?!(?s:.))")
        elif ch == "{":
            m = _QUANTIFIER_RE.match(source, i)
            if m:
                out.append(m.group(0))
                i = m.end()
                continue
            out.append("\\{")
        elif ch == "}":
            out.append("\\}")
        elif ch == "(":
            if source.startswith("(?<", i) and not source.startswith("(?<=", i) and not source.startswith("(?<!", i):
                end = source.index(">", i)
                out.append(f"(?P<{source[i + 3:end]}>")
                i = end + 1
                continue
            out.append("(")
        elif ch == "#":
            out.append("\\#")
        elif ch.isspace():
            out.append(ch)
        else:
            variants = _case_variants_class(ch, unicode_mode) if ignore_case else None
            out.append(variants if variants is not None else ch)
        i += 1
    return "".join(out)


class JsRegExp:
    """A compiled JS regular expression (``new RegExp(source, flags)``)."""

    __slots__ = ("source", "flags", "global_", "sticky", "compiled")

    def __init__(self, source: str, flags: str = "") -> None:
        self.source = source
        self.flags = flags
        self.global_ = "g" in flags
        self.sticky = "y" in flags
        # No regex.IGNORECASE: translate() expands /i to the JS case-equivalents.
        self.compiled = _compile(translate(source, flags), regex.V0)

    def with_flags(self, flags: str) -> "JsRegExp":
        return JsRegExp(self.source, flags)

    # RegExp.prototype.test for a non-global pattern.
    def test(self, text: str) -> bool:
        return self.compiled.search(text, concurrent=True, timeout=_timeout()) is not None

    def search(self, text: str, pos: int = 0):
        return self.compiled.search(text, pos, concurrent=True, timeout=_timeout())

    def match_at(self, text: str, pos: int):
        """Sticky match: only at ``pos`` (lookbehind may see text before it)."""
        return self.compiled.match(text, pos, concurrent=True, timeout=_timeout())

    def finditer(self, text: str) -> Iterator["regex.Match[str]"]:
        return self.compiled.finditer(text, concurrent=True, timeout=_timeout())

    def match_all(self, text: str) -> list[str]:
        """``String.prototype.match`` with a global pattern (``[]`` instead of null)."""
        return [m.group(0) for m in self.finditer(text)]

    def first(self, text: str):
        """``String.prototype.match`` with a non-global pattern."""
        return self.search(text)

    def split(self, text: str, limit: int | None = None) -> list[str]:
        parts = self.compiled.split(text, concurrent=True, timeout=_timeout())
        # regex.split includes captured groups like JS; JS gives undefined -> keep None.
        if limit is not None:
            parts = parts[:limit]
        return parts

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"/{self.source}/{self.flags}"


@lru_cache(maxsize=None)
def _compile(pattern: str, pyflags: int):
    return regex.compile(pattern, pyflags)


@lru_cache(maxsize=None)
def js(source: str, flags: str = "") -> JsRegExp:
    """Cached ``new RegExp(source, flags)``."""
    return JsRegExp(source, flags)


ReplaceCallback = Callable[..., str]


def _expand_template(template: str, m: "regex.Match[str]", text: str) -> str:
    if "$" not in template:
        return template
    group_count = len(m.groups())
    out: list[str] = []
    i = 0
    n = len(template)
    while i < n:
        ch = template[i]
        if ch != "$" or i + 1 >= n:
            out.append(ch)
            i += 1
            continue
        nxt = template[i + 1]
        if nxt == "$":
            out.append("$")
            i += 2
        elif nxt == "&":
            out.append(m.group(0))
            i += 2
        elif nxt == "`":
            out.append(text[: m.start()])
            i += 2
        elif nxt == "'":
            out.append(text[m.end():])
            i += 2
        elif nxt.isdigit():
            two = template[i + 1 : i + 3]
            if len(two) == 2 and two.isdigit() and 1 <= int(two) <= group_count:
                value = m.group(int(two))
                out.append(value or "")
                i += 3
            elif 1 <= int(nxt) <= group_count:
                value = m.group(int(nxt))
                out.append(value or "")
                i += 2
            else:
                out.append("$")
                i += 1
        else:
            out.append("$")
            i += 1
    return "".join(out)


def js_replace(pattern: JsRegExp, text: str, replacement: str | ReplaceCallback) -> str:
    """``text.replace(pattern, replacement)`` with JS semantics."""
    if pattern.global_:
        matches = list(pattern.finditer(text))
    else:
        first = pattern.search(text)
        matches = [first] if first is not None else []
    if not matches:
        return text
    out: list[str] = []
    last = 0
    for m in matches:
        out.append(text[last : m.start()])
        if callable(replacement):
            value = replacement(m.group(0), *m.groups(), m.start(), text)
            out.append(_js_to_string(value))
        else:
            out.append(_expand_template(replacement, m, text))
        last = m.end()
    out.append(text[last:])
    return "".join(out)


def js_replace_first_literal(text: str, needle: str, replacement: str) -> str:
    """``text.replace('needle', 'replacement')`` (string pattern: first occurrence only)."""
    index = text.find(needle)
    if index < 0:
        return text
    # A string pattern still expands $& / $$ in the replacement; the port never relies on it.
    return text[:index] + replacement + text[index + len(needle) :]


def _js_to_string(value) -> str:
    if value is None:
        return "undefined"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return js_number_to_string(value)
    return str(value)


def js_number_to_string(value: float) -> str:
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    if value == int(value) and abs(value) < 1e21:
        return str(int(value))
    return repr(value)


def js_trim(text: str) -> str:
    return text.strip(JS_WHITESPACE)


_JS_DECIMAL_RE = regex.compile(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")


def js_number(value) -> float:
    """``Number(value)`` for strings (and None -> NaN, like ``Number(undefined)``)."""
    if value is None:
        return math.nan
    if isinstance(value, (int, float)):
        return float(value)
    text = js_trim(str(value))
    if text == "":
        return 0.0
    if _JS_DECIMAL_RE.fullmatch(text):
        return float(text)
    lowered = text
    if lowered in ("Infinity", "+Infinity"):
        return math.inf
    if lowered == "-Infinity":
        return -math.inf
    for prefix, base in (("0x", 16), ("0X", 16), ("0o", 8), ("0O", 8), ("0b", 2), ("0B", 2)):
        if text.startswith(prefix):
            try:
                return float(int(text[2:], base))
            except ValueError:
                return math.nan
    return math.nan


def is_js_integer(value: float) -> bool:
    """``Number.isInteger``."""
    return isinstance(value, (int, float)) and math.isfinite(value) and float(value).is_integer()
