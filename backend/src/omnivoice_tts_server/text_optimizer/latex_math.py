"""Port of Genesis3 ``src/utils/ttsLanguageOptimizer/latexMath.ts`` (Bug 839).

The math of an answer, read before the speech rules: inline math needs dollar signs that hug their
content; a glyph command becomes the glyph, a Greek letter or function keeps its name, a text,
accent or typeface command keeps what it wraps, a fraction of two numbers stays "2/3", an integer
power of a number becomes superscript glyphs.
"""

from __future__ import annotations

from .jsre import js, js_replace, js_trim

_VARIANT_PREFIX = "var"
_SQUARE_ROOT = "√"
_DIVISION_SIGN = "÷"
_SUPERSCRIPT_DIGITS = "⁰¹²³⁴⁵⁶⁷⁸⁹"
_SUPERSCRIPT_MINUS = "⁻"

_GLYPHS = {
    "cdot": "×", "times": "×", "div": "÷", "pm": "±", "mp": "∓", "infty": "∞", "leq": "≤", "le": "≤",
    "geq": "≥", "ge": "≥", "neq": "≠", "ne": "≠", "approx": "≈", "to": "→", "rightarrow": "→",
    "Rightarrow": "⇒", "sum": "∑", "prod": "∏", "int": "∫", "partial": "∂", "in": "∈", "dots": "...",
    "ldots": "...", "cdots": "...",
}

_ROOT_GLYPHS = {"3": "∛", "4": "∜"}

_NAMES = {
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu", "nu", "xi",
    "pi", "rho", "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega", "Gamma", "Delta", "Theta", "Lambda", "Xi",
    "Pi", "Sigma", "Upsilon", "Phi", "Psi", "Omega", "lim", "max", "min", "sup", "inf", "log", "ln", "exp", "sin",
    "cos", "tan", "det", "gcd",
}

INLINE_FORMULA_RE = js(r"(?<![$0-9])\$(?=[^$\s])([^$\n]*?[^$\s])\$(?![$0-9])", "g")
_SUBSCRIPT_RE = js(r"_\{([^{}]*)\}|_(\\[a-zA-Z]+|[^{}\s\\])", "g")
_FORMULA_SLASH_RE = js(
    r"(?<=(?<![\p{L}0-9.,])[0-9]+)[ \t]*\/[ \t]*(?=\p{L}(?![\p{L}0-9]))|(?<=(?<![\p{L}0-9])\p{L})[ \t]*\/[ \t]*(?=[0-9]+(?![0-9.,]))",
    "gu",
)
_TEXT_COMMAND_RE = js(
    r"\\(?:text|textrm|textbf|textit|emph|mathrm|mathbf|mathit|mathcal|mathbb|mathfrak|mathsf|mathtt|mathscr|operatorname"
    r"|boldsymbol|bm|hat|widehat|check|tilde|widetilde|acute|grave|dot|ddot|breve|bar|vec|overline|underline"
    r"|overrightarrow|overleftarrow)\{([^}]*)\}",
    "g",
)
_FRACTION_RE = js(r"\\[dt]?frac\{([^}]*)\}\{([^}]*)\}(?=([\p{L}0-9\\])?)", "gu")
_ROOT_RE = js(r"\\sqrt(?:\[([^\]]*)\])?\{([^}]*)\}", "g")
_BRACED_POWER_RE = js(r"(?<=([0-9])|[\p{L})\]])\^\{([^{}\\]+)\}", "gu")
_COMMAND_RE = js(r"\\([a-zA-Z]+)(\{[^}]*\})?", "g")
_SPACING_COMMAND_RE = js(r"\\[,;:! \\]", "g")
_DIGITS_ONLY = js(r"^[0-9]+$", "")
_SIGNED_INTEGER = js(r"^-?[0-9]+$", "")
_SLASH = js(r"\/", "g")
_LATEX_MARK = js(r"[\\{}]", "")
_BRACES = js(r"[{}]", "g")


def _speak_fraction(_match, numerator, denominator, following, *_rest):
    if _DIGITS_ONLY.test(numerator) and _DIGITS_ONLY.test(denominator):
        fraction = f"{numerator}/{denominator}"
    else:
        fraction = f"{numerator} {_DIVISION_SIGN} {denominator}"
    return f"{fraction} " if following is not None else fraction


def _speak_root(_match, index, radicand, *_rest):
    return f"{_ROOT_GLYPHS.get(js_trim(index or ''), _SQUARE_ROOT)}{radicand}"


def _speak_power(_match, number_base, exponent, *_rest):
    if number_base is None or not _SIGNED_INTEGER.test(exponent):
        if len(exponent) == 1:
            return f"^{exponent}"
        return f"^({js_replace(_SLASH, exponent, f' {_DIVISION_SIGN} ')})"
    return "".join(_SUPERSCRIPT_MINUS if character == "-" else _SUPERSCRIPT_DIGITS[int(character)] for character in exponent)


def _speak_command(_match, command_name, argument, *_rest):
    glyph = _GLYPHS.get(command_name)
    if glyph is not None:
        return f"{glyph}{argument or ''}"
    name = command_name
    if name.startswith(_VARIANT_PREFIX) and name[len(_VARIANT_PREFIX):] in _NAMES:
        name = name[len(_VARIANT_PREFIX):]
    return f" {name} {argument or ''}" if name in _NAMES else ""


def reduce_latex_for_tts(text: str) -> str:
    """``reduceLatexForTts``: text/accent/typeface commands, fractions, roots, powers, commands, spacing, braces."""
    source = text or ""
    if not _LATEX_MARK.test(source):
        return source
    source = js_replace(_TEXT_COMMAND_RE, source, "$1")
    source = js_replace(_FRACTION_RE, source, _speak_fraction)
    source = js_replace(_ROOT_RE, source, _speak_root)
    source = js_replace(_BRACED_POWER_RE, source, _speak_power)
    source = js_replace(_COMMAND_RE, source, _speak_command)
    source = js_replace(_SPACING_COMMAND_RE, source, " ")
    return js_replace(_BRACES, source, "")


def read_inline_math_for_tts(text: str) -> str:
    """``readInlineMathForTts``: unwrap inline math of a line and reduce its LaTeX."""
    source = text or ""
    if "$" not in source:
        return source

    def callback(_match, content, *_rest):
        content = js_replace(_SUBSCRIPT_RE, content, " $1$2 ")
        return reduce_latex_for_tts(js_replace(_FORMULA_SLASH_RE, content, f" {_DIVISION_SIGN} "))

    return js_replace(INLINE_FORMULA_RE, source, callback)


def unwrap_inline_math_for_tts(text: str) -> str:
    return js_replace(INLINE_FORMULA_RE, text or "", "$1")
