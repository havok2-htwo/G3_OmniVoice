"""Parity of the Python text optimizer with the Genesis3 TypeScript implementation.

Fixtures (tests/fixtures/text_optimizer/, generated, never hand-edited):

* golden.json.gz - 1530 unique inputs (the Genesis TTS sweep corpus, every input the TS
  unit tests pass to the pipeline, and OmniVoice extra cases) x {de, en} x {floor, final}:
  optimizeTtsTextForLanguage output + appliedRules, and the full deterministic pipeline
  (buildDeterministicTtsFallback) output + appliedRules + authorization.
* golden_components.json.gz - per-stage records: Markdown strip, stack-frame strip, URL
  reduction, postProcessTtsText for de/en/None, language detection and binding.

They come from the unmodified TS sources run under node + tsx. To regenerate after a
Genesis update, replace the copied TS sources in the golden harness, rerun its two node
commands and gzip the outputs (see the commit that added this file).

A failure prints a per-rule mismatch histogram plus examples, so a port fix can go pass
by pass (the component goldens isolate the stages).
"""

from __future__ import annotations

import collections
import gzip
import json
from pathlib import Path

import pytest

from omnivoice_tts_server.text_optimizer import language_binding, markdown_stripper, pipeline, post_processor
from omnivoice_tts_server.text_optimizer import reference_links

FIXTURES = Path(__file__).resolve().parent / 'fixtures' / 'text_optimizer'


def _load(name: str):
    with gzip.open(FIXTURES / name, 'rt', encoding='utf-8') as handle:
        return json.load(handle)


@pytest.fixture(scope='module')
def golden() -> list[dict]:
    return _load('golden.json.gz')


@pytest.fixture(scope='module')
def components() -> list[dict]:
    return _load('golden_components.json.gz')


def _report(failures: list[tuple[str, dict, object, object]], total: int) -> str:
    histogram = collections.Counter(key for key, *_ in failures)
    lines = [f'{len(failures)} of {total} records differ']
    lines += [f'  {count:5d}  {key}' for key, count in histogram.most_common(30)]
    for key, record, expected, got in failures[:12]:
        lines.append(f'--- {key}\n  in : {record.get("input")!r} ({record.get("language")}, {record.get("stage")})')
        lines.append(f'  exp: {expected!r}\n  got: {got!r}')
    return '\n'.join(lines)


def _rule_key(prefix: str, language, expected_rules, got_rules) -> str:
    diff = sorted(set(expected_rules) ^ set(got_rules))
    return f'{prefix}:{language}:' + (','.join(diff) if diff else 'text')


def test_golden_record_count(golden) -> None:
    assert len(golden) == 6120
    assert {(r['language'], r['stage']) for r in golden} == {('de', 'floor'), ('de', 'final'), ('en', 'floor'), ('en', 'final')}


def test_language_optimizer_matches_ts_golden(golden) -> None:
    failures = []
    for record in golden:
        result = pipeline.optimize_tts_text_for_language(record['input'], record['language'], record['stage'])
        got = (result.text, result.applied_rules)
        expected = (record['output'], record['appliedRules'])
        if got != expected:
            failures.append((_rule_key('optimizer', record['language'], expected[1], got[1]), record, expected, got))
    assert not failures, _report(failures, len(golden))


def test_deterministic_pipeline_matches_ts_golden(golden) -> None:
    failures = []
    for record in golden:
        result = pipeline.build_deterministic_tts_text(record['input'], record['language'], stage=record['stage'])
        got = (result.text, result.applied_rules, result.authorization)
        exp = record['pipeline']
        expected = (exp['output'], exp['appliedRules'], exp['authorization'])
        if got != expected:
            failures.append((_rule_key('pipeline', record['language'], expected[1], got[1]), record, expected, got))
    assert not failures, _report(failures, len(golden))


def _component_result(record: dict):
    fn = record['fn']
    text = record['input']
    if fn == 'stripMarkdownForTts':
        return markdown_stripper.strip_markdown_for_tts(text)
    if fn == 'stripStackTraceFramesForTts':
        stripped, count = reference_links.strip_stack_trace_frames_for_tts(text)
        return {'text': stripped, 'strippedFrames': count}
    if fn == 'reduceReferenceUrlsToSpokenDomainForTts':
        reduced, count = reference_links.reduce_reference_urls_to_spoken_domain_for_tts(text)
        return {'text': reduced, 'reducedUrls': count}
    if fn == 'detectHintLanguageFromText':
        return language_binding.detect_hint_language_from_text(text)
    if fn == 'resolveTtsLanguageBinding':
        language, origin = language_binding.resolve_tts_language_binding(text, record['language'])
        return {'language': language, 'origin': origin}
    if fn == 'postProcessTtsText':
        processed, rules = post_processor.post_process_tts_text(text, record['language'])
        return {'text': processed, 'appliedRules': rules}
    raise AssertionError(f'unknown component {fn}')


def test_pipeline_components_match_ts_golden(components) -> None:
    assert len(components) == 15347
    failures = []
    for record in components:
        got = _component_result(record)
        if got != record['output']:
            failures.append((f"{record['fn']}:{record.get('language')}", record, record['output'], got))
    assert not failures, _report(failures, len(components))


def test_floor_and_final_differ_only_in_german(golden) -> None:
    """The stage is a German-only switch (index.ts ignores it for English)."""
    by_key: dict[tuple[str, str], dict[str, str]] = collections.defaultdict(dict)
    for record in golden:
        by_key[(record['input'], record['language'])][record['stage']] = record['output']
    english_diffs = sum(1 for (text, lang), outs in by_key.items() if lang == 'en' and outs['floor'] != outs['final'])
    german_diffs = sum(1 for (text, lang), outs in by_key.items() if lang == 'de' and outs['floor'] != outs['final'])
    assert english_diffs == 0
    assert german_diffs > 0
