"""Port of Genesis3 ``src/utils/ttsLanguageOptimizer/markdownLineStructure.ts`` (Bugs 824-826).

The line pass of the Markdown strip: every line is classified on its source, loses its block
marker, gets the inline rules (markdown_inline.py) and, where Markdown shows a boundary that the
text does not write, a boundary mark at its end (" ,", " .", " :").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .jsre import js, js_replace, js_trim
from .markdown_inline import strip_inline_markdown_for_tts
from .shared import js_trim_start, status_mark_at


@dataclass(frozen=True)
class MarkdownSpokenWords:
    """The words of the bound language for task states and status marks (Bug 826)."""

    task_open: str
    task_done: str
    status_yes: str
    status_no: str
    status_warning: str


@dataclass(eq=False)
class _Line:
    role: str = "plain"  # 'blank' | 'heading' | 'item' | 'plain'
    kind: str = "none"  # 'none' | 'bullet' | 'task' | 'status' | 'ordered' | 'label' | 'pipeRow'
    task_state: str = "none"
    status: Optional[str] = None
    depth: int = 0
    in_quote: bool = False
    source: str = ""
    raw: str = ""
    marker: str = ""
    text: str = ""
    content: str = ""
    continued_item: Optional["_Line"] = None
    run: int = 0
    heavy_run: bool = False
    mark: str = ""


_HEADING_INDENTATION = 3
_HASH_TITLE_MAX_WORDS = 8
_HEAVY_ITEM_WORDS = 8
_TAB_WIDTH = 4
_CONTENT_INDENTATION = 2
_COMMA = " ,"
_PERIOD = " ."
_COLON = " :"

_HARD_BREAK_BACKSLASH_RE = js(r"(?<!\\)\\[ \t]*$", "")
_QUOTE_MARKER_RE = js(r"^[ \t]*(?:>[ \t]?)+", "")
_ATX_HEADING_RE = js(r"^#{1,6}[ \t]+(?=\S)", "")
_HASH_TITLE_RE = js(r"^#(?=\p{Lu})", "u")
_BOLD_ONLY_LINE_RE = js(r"^(?:\*\*[^*\n]+\*\*|__[^_\n]+__)$", "")
_TABLE_SEPARATOR_RE = js(r"^\|?[ \t]*:?-{3,}:?[ \t]*(?:\|[ \t]*:?-{3,}:?[ \t]*)*\|?$", "")
_SHORT_TABLE_SEPARATOR_RE = js(
    r"^(?:\|[ \t]*:?-+:?[ \t]*(?:\|[ \t]*:?-+:?[ \t]*)*\||\|?[ \t]*:?-+:?[ \t]*(?:\|[ \t]*:?-+:?[ \t]*)+\|?)$", ""
)
_LONG_TABLE_SEPARATOR = "|---|"
_HORIZONTAL_RULE_RE = js(r"^([*_\-])\1{2,}$", "")
_SETEXT_UNDERLINE_RE = js(r"^(?:={3,}|-{3,})$", "")
_BULLET_RE = js(r"^[-+*•·][ \t]+(?=\S)", "")
_ORDERED_MARKER_RE = js(r"^(\d{1,2})[.)][ \t]+(?=\S)", "")
_LABEL_LINE_RE = js(r"^\*\*[^*\n]+(?::\*\*|\*\*:)", "")
_TASK_ITEM_RE = js(r"^(?:[-+*]|(\d{1,2})[.)])[ \t]+\[([ xX])\][ \t]+(?=\S)", "")
_EDGE_PIPE_RE = js(r"^\||(?<!\\)\|$", "")
_EDGE_PIPES_RE = js(r"^\|[ \t]*|[ \t]*(?<!\\)\|$", "g")
_UNESCAPED_PIPE_RE = js(r"(?<!\\)\|", "")
_ESCAPED_BLOCK_MARKER_RE = js(r"^\\[#>+\-](?=[ \t])", "")
_ESCAPED_ORDERED_MARKER_RE = js(r"^(\d{1,2})\\\.(?=[ \t])", "")
_ENDS_WITH_PUNCTUATION_RE = js(r"[.!?:;,…][\"'“”„‘’»«)\]]*$", "")
_ENDS_WITH_MARK_IN_BRACKET_RE = js(r"[.!][)\]]$", "")
_EDGE_LINE_BREAKS = js(r"^\n+|\n+$", "g")
_BLANKS = js(r"[ \t]+", "")
_STARTS_LOWER = js(r"^\p{Ll}", "u")
_STARTS_WORDISH = js(r"^[\p{L}\p{N}]", "u")


def apply_markdown_line_structure(
    text: str, words: Optional[MarkdownSpokenWords] = None, *, mixed_status_run: bool = False
) -> str:
    """``applyMarkdownLineStructure``: speaks the structure of the text as boundary marks."""
    source = text or ""
    if not js_trim(source):
        return source
    lines = _read_structure(source)
    if words:
        _speak_states(lines, words, mixed_status_run)
    _mark_boundaries(lines)
    return js_replace(_EDGE_LINE_BREAKS, "\n".join(line.content + line.mark for line in lines), "")


def _read_structure(source: str) -> list[_Line]:
    lines = _classify(source.split("\n"))
    _mark_continuations(lines)
    _mark_runs(lines)
    return lines


def mixed_status_run_reader(text: str) -> Callable[[str], bool]:
    """``mixedStatusRunReader``: whether a piece holds a status item of a mixed run of the whole text."""
    lines = _read_structure(text or "")
    mixed = _mixed_status_runs(lines)
    items = {line.source for line in lines if line.status and line.run in mixed}
    return lambda piece: bool(items) and any(_read_line(line).source in items for line in piece.split("\n"))


def _classify(sources: list[str]) -> list[_Line]:
    lines = [_read_line(source) for source in sources]
    _read_short_separators(lines)
    _resolve_neighbours(lines)
    kept: list[_Line] = []
    for line in lines:
        if line.role == "blank":
            kept.append(line)
            continue
        _read_content(line)
        if line.content:
            kept.append(line)
    return kept


def _read_line(source_line: str) -> _Line:
    without_break = js_replace(_HARD_BREAK_BACKSLASH_RE, source_line, "")
    quote_match = _QUOTE_MARKER_RE.search(without_break)
    quote = quote_match.group(0) if quote_match else ""
    rest = without_break[len(quote):]
    trimmed = js_trim(rest)
    line = _Line(depth=_indentation(rest), in_quote=len(quote) > 0, source=trimmed, raw=trimmed)
    if not trimmed:
        line.role = "blank"
        return line
    _read_block(line)
    return line


def _read_block(line: _Line) -> None:
    text = line.source
    heading = _ATX_HEADING_RE.search(text)
    if heading and line.depth <= _HEADING_INDENTATION:
        line.role = "heading"
        line.raw = text[len(heading.group(0)):]
        return
    if _is_hash_title(text) and line.depth <= _HEADING_INDENTATION:
        line.role = "heading"
        line.raw = text[1:]
        return
    if _BOLD_ONLY_LINE_RE.test(text):
        line.role = "heading"
        return
    if _TABLE_SEPARATOR_RE.test(text) or _HORIZONTAL_RULE_RE.test(text):
        return
    _read_item(line)


def _read_item(line: _Line) -> None:
    text = line.source
    task = _TASK_ITEM_RE.search(text)
    if task:
        _set_item(line, "task", text[len(task.group(0)):])
        line.task_state = "open" if task.group(2) == " " else "done"
        if task.group(1):
            line.kind = "ordered"
            line.marker = f"{task.group(1)}. "
        return
    bullet = _BULLET_RE.search(text)
    content = text[len(bullet.group(0)):] if bullet else text
    mark = status_mark_at(content, 0)
    if mark:
        _set_item(line, "status", js_trim_start(content[mark.length:]))
        line.status = mark.status
        return
    if bullet:
        _set_item(line, "bullet", content)
        return
    ordered = _ORDERED_MARKER_RE.search(text)
    if ordered:
        _set_item(line, "ordered", text[len(ordered.group(0)):])
        line.marker = f"{ordered.group(1)}. "
        return
    if _LABEL_LINE_RE.test(text):
        _set_item(line, "label", text)
        return
    if _EDGE_PIPE_RE.test(text):
        _set_item(line, "pipeRow", js_replace(_EDGE_PIPES_RE, text, ""))


def _set_item(line: _Line, kind: str, raw: str) -> None:
    line.role = "item"
    line.kind = kind
    line.raw = raw


def _read_short_separators(lines: list[_Line]) -> None:
    for index in range(1, len(lines)):
        if (
            _SHORT_TABLE_SEPARATOR_RE.test(lines[index].source)
            and _UNESCAPED_PIPE_RE.test(lines[index - 1].source)
            and (index == 1 or not _UNESCAPED_PIPE_RE.test(lines[index - 2].source))
        ):
            lines[index] = _read_line(_LONG_TABLE_SEPARATOR)


def _resolve_neighbours(lines: list[_Line]) -> None:
    for index, line in enumerate(lines):
        if line.role != "plain" or _TABLE_SEPARATOR_RE.test(line.source):
            continue
        if _UNESCAPED_PIPE_RE.test(line.source) and (
            _has_pipe_neighbour(lines, index - 1) or _has_pipe_neighbour(lines, index + 1)
        ):
            _set_item(line, "pipeRow", js_replace(_EDGE_PIPES_RE, line.source, ""))
    removed: set[int] = set()
    for index, line in enumerate(lines):
        if line.role != "plain":
            continue
        previous = lines[index - 1] if index > 0 else None
        if (
            _SETEXT_UNDERLINE_RE.test(line.source)
            and line.depth <= _HEADING_INDENTATION
            and previous is not None
            and previous.role == "plain"
            and id(previous) not in removed
        ):
            previous.role = "heading"
            removed.add(id(line))
        elif _TABLE_SEPARATOR_RE.test(line.source) and (
            "|" in line.source
            or not _HORIZONTAL_RULE_RE.test(line.source)
            or (previous is not None and previous.kind == "pipeRow")
        ):
            removed.add(id(line))
        elif _HORIZONTAL_RULE_RE.test(line.source):
            line.role = "blank"
    lines[:] = [line for line in lines if id(line) not in removed]
    for index, line in enumerate(lines):
        if _interrupts_paragraph(lines, index):
            line.role = "plain"
            line.kind = "none"
            line.raw = line.source
            line.marker = ""


def _interrupts_paragraph(lines: list[_Line], index: int) -> bool:
    line = lines[index]
    if line.kind != "ordered" or line.task_state != "none":
        return False
    ordered = _ORDERED_MARKER_RE.search(line.source)
    if ordered and int(ordered.group(1)) == 1:
        return False
    cursor = index - 1
    while cursor >= 0 and lines[cursor].role == "plain":
        cursor -= 1
    return cursor < index - 1 and (cursor < 0 or lines[cursor].role != "item")


def _has_pipe_neighbour(lines: list[_Line], index: int) -> bool:
    if index < 0 or index >= len(lines) or lines[index].role == "blank":
        return False
    source = lines[index].source
    return _TABLE_SEPARATOR_RE.test(source) or lines[index].kind == "pipeRow" or _UNESCAPED_PIPE_RE.test(source)


def _read_content(line: _Line) -> None:
    raw = js_replace(_ESCAPED_ORDERED_MARKER_RE, js_replace(_ESCAPED_BLOCK_MARKER_RE, line.raw, ""), "$1")
    line.text = js_trim(strip_inline_markdown_for_tts(raw, line.kind == "pipeRow"))
    line.content = line.marker + line.text if line.text else ""


def _mark_continuations(lines: list[_Line]) -> None:
    for index in range(1, len(lines)):
        line = lines[index]
        if line.role != "plain":
            continue
        item = _item_above(lines, index)
        if item and _continues_item(line, item, lines[index - 1].role == "blank"):
            line.role = "item"
            line.kind = item.kind
            line.depth = item.depth
            line.continued_item = item


def _item_above(lines: list[_Line], index: int) -> Optional[_Line]:
    for above in range(index - 1, -1, -1):
        line = lines[above]
        if line.role == "blank":
            continue
        if line.role != "item" or line.kind in ("label", "pipeRow"):
            return None
        return line.continued_item or line
    return None


def _continues_item(line: _Line, item: _Line, after_blank_line: bool) -> bool:
    if line.in_quote != item.in_quote:
        return False
    if after_blank_line:
        return line.depth >= item.depth + _CONTENT_INDENTATION
    return line.depth > item.depth or _STARTS_LOWER.test(line.text)


def _mark_runs(lines: list[_Line]) -> None:
    run = 0
    start = 0
    while start < len(lines):
        if lines[start].role != "item":
            start += 1
            continue
        run += 1
        end = _run_end(lines, start)
        heavy = False
        for index in range(start, end):
            heavy = heavy or _is_heavy_item(lines, index)
        for index in range(start, end):
            if lines[index].role == "item":
                lines[index].run = run
                lines[index].heavy_run = heavy
        start = end


def _run_end(lines: list[_Line], start: int) -> int:
    is_row = lines[start].kind == "pipeRow"
    last = start + 1
    for index in range(start + 1, len(lines)):
        line = lines[index]
        if line.role == "blank":
            continue
        if line.role != "item" or (line.kind == "pipeRow") != is_row:
            break
        last = index + 1
    return last


def _is_heavy_item(lines: list[_Line], index: int) -> bool:
    line = lines[index]
    if line.role != "item" or line.continued_item:
        return False
    if line.kind in ("label", "pipeRow"):
        return True
    text = line.text
    for following in lines[index + 1:]:
        if following.continued_item is line:
            text += f" {following.text}"
        elif following.role != "blank":
            break
    words = len([part for part in _BLANKS.split(text) if part])
    return words > _HEAVY_ITEM_WORDS or ", " in text or "; " in text or ": " in text


def _speak_states(lines: list[_Line], words: MarkdownSpokenWords, in_mixed_run: bool) -> None:
    mixed_runs = _mixed_status_runs(lines)
    for line in lines:
        word = _state_word(line, in_mixed_run or line.run in mixed_runs, words)
        if not word or (line.status == "warning" and _starts_with_word(line.text, word)):
            continue
        line.content = f"{line.marker}{word}: {line.text}"


def _mixed_status_runs(lines: list[_Line]) -> set[int]:
    positive_runs = {line.run for line in lines if line.status == "positive"}
    return {line.run for line in lines if line.status == "negative" and line.run in positive_runs}


def _state_word(line: _Line, mixed_run: bool, words: MarkdownSpokenWords) -> str:
    if line.task_state == "open":
        return words.task_open
    if line.task_state == "done":
        return words.task_done
    if line.status == "warning":
        return words.status_warning
    if line.status == "positive":
        return words.status_yes if mixed_run else ""
    if line.status == "negative":
        return words.status_no if mixed_run else ""
    return ""


def _starts_with_word(text: str, word: str) -> bool:
    return text.lower().startswith(word.lower()) and not _STARTS_WORDISH.test(text[len(word):])


def _mark_boundaries(lines: list[_Line]) -> None:
    for index, line in enumerate(lines):
        if line.role == "blank":
            continue
        following = _next_non_blank(lines, index)
        if following is None:
            continue
        if _ENDS_WITH_PUNCTUATION_RE.test(line.content) and not (
            line.role == "heading" and _ENDS_WITH_MARK_IN_BRACKET_RE.test(line.content)
        ):
            continue
        line.mark = _mark_for(line, following, lines[index + 1])


def _mark_for(line: _Line, following: _Line, directly_next: _Line) -> str:
    if line.role == "heading":
        return _PERIOD
    if line.role == "item":
        return _item_mark(line, following, directly_next)
    if directly_next.role == "item" and directly_next.in_quote == line.in_quote:
        return _COLON
    if directly_next.role in ("blank", "heading") or directly_next.in_quote != line.in_quote:
        return _PERIOD
    return ""


def _item_mark(line: _Line, following: _Line, directly_next: _Line) -> str:
    item = line.continued_item or line
    if following.continued_item is item:
        return "" if following is directly_next else _PERIOD
    if following.role != "item" or following.run != line.run:
        return _PERIOD
    if following.depth > line.depth:
        return _COLON
    if following.depth < line.depth or line.kind == "ordered" or line.heavy_run:
        return _PERIOD
    return _COMMA


def _next_non_blank(lines: list[_Line], index: int) -> Optional[_Line]:
    for following in lines[index + 1:]:
        if following.role != "blank":
            return following
    return None


def _is_hash_title(text: str) -> bool:
    if not _HASH_TITLE_RE.test(text) or text.find("#", 1) >= 0:
        return False
    return len([part for part in _BLANKS.split(text) if part]) <= _HASH_TITLE_MAX_WORDS


def _indentation(text: str) -> int:
    width = 0
    for character in text:
        if character == " ":
            width += 1
        elif character == "\t":
            width += _TAB_WIDTH
        else:
            break
    return width
