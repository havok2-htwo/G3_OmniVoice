"""Language dispatch (``index.ts``) and the deterministic pipeline
(``runtime/ttsDeterministicFallback.ts`` ``buildDeterministicTtsFallback``)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

from .de import optimize_german_tts_text
from .en import optimize_english_tts_text
from .jsre import js_trim
from .markdown_stripper import strip_markdown_for_tts
from .post_processor import post_process_tts_text
from .reference_links import (
    reduce_reference_urls_to_spoken_domain_for_tts,
    strip_link_reference_definitions_for_tts,
    strip_stack_trace_frames_for_tts,
)
from .shared import (
    TtsAlgorithmOptimizationResult,
    TtsOptimizeOptions,
    finalize_optimized_text,
    remove_invisible_characters_for_tts,
    restore_held_day_months,
    trim_keeping_list_item_depth_for_tts,
    unify_line_breaks_for_tts,
)


def optimize_tts_text_for_language(
    text: str, language: Optional[str], stage: str = "final", *, options: Optional[TtsOptimizeOptions] = None
) -> TtsAlgorithmOptimizationResult:
    """``optimizeTtsTextForLanguage``: de* -> German passes, en* -> English passes, anything else
    -> whitespace finalize only. ``options`` carries the TS ``TtsOptimizeOptions``; ``stage`` sets
    its stage."""
    opts = replace(options or TtsOptimizeOptions(), stage=stage)
    normalized_language = js_trim(language or "").lower()
    if not js_trim(text or ""):
        return TtsAlgorithmOptimizationResult(text or "", [], normalized_language or None)
    if normalized_language.startswith("de"):
        return optimize_german_tts_text(text, opts)
    if normalized_language.startswith("en"):
        return optimize_english_tts_text(text, opts)
    return TtsAlgorithmOptimizationResult(finalize_optimized_text(text), [], normalized_language or None)


@dataclass
class DeterministicTtsResult:
    text: str
    applied_rules: list[str]
    language: Optional[str]
    # 'ordered_sequence_markers' when the only semantic change was the list-marker transform.
    authorization: Optional[str]


def build_deterministic_tts_text(
    source_text: str,
    source_language: str,
    *,
    enable_algorithmic_optimization: bool = True,
    stage: str = "final",
    continues_numbered_list: bool = False,
    starts_inside_line: bool = False,
    mixed_status_run: bool = False,
    answer_text: Optional[str] = None,
) -> DeterministicTtsResult:
    """Port of ``buildDeterministicTtsFallback``: line breaks, invisible characters, frames, link
    reference definitions, URL domains, Markdown, language passes, post-processor and (if the
    post-processor changed anything) a second language pass; held day.month pairs go back last.

    The options continues_numbered_list, starts_inside_line, mixed_status_run and answer_text mirror
    the TS chunk-context options. service.py never passes them (one whole text per request); they
    are kept for parity and only checked by the TS unit-test replay, not by the golden fixtures."""
    source = trim_keeping_list_item_depth_for_tts(
        remove_invisible_characters_for_tts(unify_line_breaks_for_tts(source_text or ""))
    )
    if not source:
        return DeterministicTtsResult("", [], None, None)

    applied_rules: list[str] = []
    frame_text, stripped_frames = strip_stack_trace_frames_for_tts(source)
    if stripped_frames > 0:
        applied_rules.append("stack_trace_frames")

    definition_text, stripped_definitions = strip_link_reference_definitions_for_tts(frame_text)
    if stripped_definitions > 0:
        applied_rules.append("link_reference_definitions")

    url_text, reduced_urls = reduce_reference_urls_to_spoken_domain_for_tts(definition_text)
    if reduced_urls > 0:
        applied_rules.append("reference_url_domains")

    markdown_free = strip_markdown_for_tts(url_text, source_language, mixed_status_run=mixed_status_run)
    if markdown_free != url_text:
        applied_rules.append("markdown_strip")

    base_options = TtsOptimizeOptions(
        stage=stage,
        continues_numbered_list=continues_numbered_list,
        starts_inside_line=starts_inside_line,
        answer_text=answer_text,
        keep_held_day_months=True,
    )
    text = markdown_free
    language: Optional[str] = None
    semantic_rules: list[str] = []
    held_day_months: Optional[list[str]] = None
    if enable_algorithmic_optimization and source_language:
        optimized = optimize_tts_text_for_language(markdown_free, source_language, stage, options=base_options)
        held_day_months = optimized.held_day_months
        semantic_rules = optimized.applied_rules
        applied_rules.extend(semantic_rules)
        if optimized.text:
            text = optimized.text
        language = optimized.language

    authorization = (
        "ordered_sequence_markers"
        if reduced_urls == 0
        and markdown_free == source
        and len(semantic_rules) == 1
        and semantic_rules[0] in ("list_marker_de", "list_marker_en")
        else None
    )

    post_text, post_rules = post_process_tts_text(text, source_language)
    if post_text:
        text = post_text
    applied_rules.extend(post_rules)

    if enable_algorithmic_optimization and source_language and post_rules:
        second = optimize_tts_text_for_language(
            text,
            source_language,
            stage,
            options=replace(base_options, hold_list_opener=True, day_months_settled=True),
        )
        if second.text:
            text = second.text
        for rule in second.applied_rules:
            if rule not in applied_rules:
                applied_rules.append(rule)

    text = restore_held_day_months(text, held_day_months)
    return DeterministicTtsResult(text, applied_rules, language, authorization)
