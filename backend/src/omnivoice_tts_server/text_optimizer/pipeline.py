"""Language dispatch (``index.ts``) and the deterministic pipeline
(``runtime/ttsDeterministicFallback.ts`` ``buildDeterministicTtsFallback``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .de import optimize_german_tts_text
from .en import optimize_english_tts_text
from .jsre import js_trim
from .markdown_stripper import strip_markdown_for_tts
from .post_processor import post_process_tts_text
from .reference_links import reduce_reference_urls_to_spoken_domain_for_tts, strip_stack_trace_frames_for_tts
from .shared import TtsAlgorithmOptimizationResult, finalize_optimized_text


def optimize_tts_text_for_language(
    text: str, language: Optional[str], stage: str = "final"
) -> TtsAlgorithmOptimizationResult:
    """``optimizeTtsTextForLanguage``: de* -> German passes (stage-aware), en* -> English
    passes (stage is ignored, as in TS), anything else -> whitespace finalize only."""
    normalized_language = js_trim(language or "").lower()
    if not js_trim(text or ""):
        return TtsAlgorithmOptimizationResult(text or "", [], normalized_language or None)
    if normalized_language.startswith("de"):
        return optimize_german_tts_text(text, stage)
    if normalized_language.startswith("en"):
        return optimize_english_tts_text(text)
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
) -> DeterministicTtsResult:
    """Port of ``buildDeterministicTtsFallback``: frames, URL domains, Markdown, language
    passes, post-processor and (if the post-processor changed anything) a second language pass."""
    source = js_trim(source_text or "")
    if not source:
        return DeterministicTtsResult("", [], None, None)

    applied_rules: list[str] = []
    frame_text, stripped_frames = strip_stack_trace_frames_for_tts(source)
    if stripped_frames > 0:
        applied_rules.append("stack_trace_frames")

    url_text, reduced_urls = reduce_reference_urls_to_spoken_domain_for_tts(frame_text)
    if reduced_urls > 0:
        applied_rules.append("reference_url_domains")

    markdown_free = strip_markdown_for_tts(url_text)
    if markdown_free != url_text:
        applied_rules.append("markdown_strip")

    text = markdown_free
    language: Optional[str] = None
    semantic_rules: list[str] = []
    if enable_algorithmic_optimization and source_language:
        optimized = optimize_tts_text_for_language(markdown_free, source_language, stage)
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
        second = optimize_tts_text_for_language(text, source_language, stage)
        if second.text:
            text = second.text
        for rule in second.applied_rules:
            if rule not in applied_rules:
                applied_rules.append(rule)

    return DeterministicTtsResult(text, applied_rules, language, authorization)
