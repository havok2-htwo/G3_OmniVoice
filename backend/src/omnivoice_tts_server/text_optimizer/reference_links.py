"""Port of Genesis3 ``src/utils/ttsLanguageOptimizer/referenceLinkStripper.ts``.

A full URL is never read aloud: it is reduced to its bare domain. Stack-trace
frames are removed from the spoken rendering.
"""

from __future__ import annotations

from typing import Optional

from .jsre import js, js_replace, js_trim

URL_ANYWHERE_RE = js(r"<((?:https?:\/\/|www\.)[^\s<>]+)>|(?:https?:\/\/|www\.)[^\s<>]+", "gi")
TRAILING_PUNCTUATION_RE = js(r"[.,;:!?)\"'\]]+$", "")
SPEAKABLE_HOST_RE = js(r"^[a-z0-9-]+(?:\.[a-z0-9-]+)+$", "i")
_SCHEME = js(r"^https?:\/\/", "i")
_USERINFO = js(r"^[^/?#@]*@", "")
_HOST_SPLIT = js(r"[/?#:]", "")
_WWW = js(r"^www\.", "i")


def _spoken_domain_for_url(raw_url: str) -> Optional[str]:
    suffix_match = TRAILING_PUNCTUATION_RE.first(raw_url)
    suffix = suffix_match.group(0) if suffix_match else ""
    core = raw_url[: -len(suffix)] if suffix else raw_url
    host = js_replace(_USERINFO, js_replace(_SCHEME, core, ""), "")
    host = _HOST_SPLIT.split(host)[0]
    host = js_replace(_WWW, host, "")
    if not SPEAKABLE_HOST_RE.test(host):
        return None
    return f"{host}{suffix}"


STRONG_LOCATION_RE = js(r"[a-z]+:\/\/|\/|:\d", "i")
PAREN_STACK_FRAME_RE = js(r"(?:^|[ \t])at[ \t]+(?:async[ \t]+|new[ \t]+)?[\w$.<>\[\]#]+[ \t]*\(([^()\n]*)\)", "g")
BARE_STACK_FRAME_RE = js(r"(?:^|[ \t])at[ \t]+(?:async[ \t]+)?[^\s()]*\/[^\s()]*:\d+:\d+(?=[\s.,;]|$)", "g")
PYTHON_STACK_FRAME_RE = js(r"(?:^|[ \t])File[ \t]+\"[^\"\n]+\",[ \t]*line[ \t]+\d+(?:,[ \t]*in[ \t]+[\w<>.]+)?", "g")
_MULTI_BLANK = js(r"[ \t]{2,}", "g")
_TRAILING_BLANK = js(r"[ \t]+(\n|$)", "g")


def strip_stack_trace_frames_for_tts(text: str) -> tuple[str, int]:
    """Returns (text, stripped_frames)."""
    raw = text or ""
    if not js_trim(raw):
        return raw, 0
    stripped = 0

    def paren(match, location, *_rest):
        nonlocal stripped
        if not STRONG_LOCATION_RE.test(location):
            return match
        stripped += 1
        return ""

    def drop(*_args):
        nonlocal stripped
        stripped += 1
        return ""

    result = js_replace(PAREN_STACK_FRAME_RE, raw, paren)
    result = js_replace(BARE_STACK_FRAME_RE, result, drop)
    result = js_replace(PYTHON_STACK_FRAME_RE, result, drop)
    if stripped == 0:
        return raw, 0
    result = js_replace(_TRAILING_BLANK, js_replace(_MULTI_BLANK, result, " "), "$1")
    return result, stripped


def reduce_reference_urls_to_spoken_domain_for_tts(text: str) -> tuple[str, int]:
    """Returns (text, reduced_urls)."""
    raw = text or ""
    if not js_trim(raw):
        return raw, 0
    reduced = 0

    def callback(match, autolinked, *_rest):
        nonlocal reduced
        candidate = autolinked if isinstance(autolinked, str) and autolinked else match
        domain = _spoken_domain_for_url(candidate)
        if domain is None:
            return match
        if domain != match:
            reduced += 1
        return domain

    return js_replace(URL_ANYWHERE_RE, raw, callback), reduced
