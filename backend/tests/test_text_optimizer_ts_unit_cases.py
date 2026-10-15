"""The Genesis3 TS unit tests of the ported functions, replayed against the Python port.

tests/fixtures/text_optimizer/ts_unit_cases.json.gz was recorded by running the original
TS test files (tts-language-optimizer, tts-language-binding, tts-hint-language-detection;
303 passing tests) under an assertion-recording runner:

* ``assertions``: every ``expect(actual).<matcher>(expected)`` whose ``actual`` came from a
  ported function, with the call (function + arguments) and the accessor that produced
  ``actual`` (``.text``, ``.appliedRules.join(',')``, ...). Replayed one by one below.
* ``calls``: every call of a ported function made by those tests with its full TS result;
  the Python result must be identical. This also covers assertions on derived values
  (``/\\d/.test(result.text)``, counts) that cannot be linked to a single accessor.

Not covered: assertions on the hint injector (injectTtsHintsForLanguage), the spoken
number fidelity gate, SentenceChunker and the Markdown table parser, none of which are
ported (G3_OmniVoice keeps its own sentence splitter).
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from omnivoice_tts_server.text_optimizer import language_binding, markdown_stripper, pipeline, post_processor
from omnivoice_tts_server.text_optimizer import reference_links, shared
from omnivoice_tts_server.text_optimizer.jsre import js

FIXTURE = Path(__file__).resolve().parent / 'fixtures' / 'text_optimizer' / 'ts_unit_cases.json.gz'

with gzip.open(FIXTURE, 'rt', encoding='utf-8') as _handle:
    CASES = json.load(_handle)


def run_ts_function(fn: str, args: list):
    """Call the Python port the way the TS test called the TS function; return a JS-shaped value."""
    text = args[0] if args else None
    if fn == 'optimizeTtsTextForLanguage':
        options = args[2] if len(args) > 2 and isinstance(args[2], dict) else {}
        result = pipeline.optimize_tts_text_for_language(text or '', args[1] if len(args) > 1 else None, options.get('stage') or 'final')
        return {'text': result.text, 'appliedRules': result.applied_rules, 'language': result.language}
    if fn == 'buildDeterministicTtsFallback':
        params = args[0]
        result = pipeline.build_deterministic_tts_text(
            params.get('sourceText') or '',
            params.get('sourceLanguage') or '',
            enable_algorithmic_optimization=bool(params.get('enableAlgorithmicOptimization')),
            stage=params.get('stage') or 'final',
        )
        return {
            'text': result.text,
            'appliedRules': result.applied_rules,
            'language': result.language,
            'authorization': result.authorization,
        }
    if fn == 'postProcessTtsText':
        processed, rules = post_processor.post_process_tts_text(text or '', args[1] if len(args) > 1 else None)
        return {'text': processed, 'appliedRules': rules}
    if fn == 'stripMarkdownForTts':
        return markdown_stripper.strip_markdown_for_tts(text)
    if fn == 'reduceReferenceUrlsToSpokenDomainForTts':
        reduced, count = reference_links.reduce_reference_urls_to_spoken_domain_for_tts(text)
        return {'text': reduced, 'reducedUrls': count}
    if fn == 'stripStackTraceFramesForTts':
        stripped, count = reference_links.strip_stack_trace_frames_for_tts(text)
        return {'text': stripped, 'strippedFrames': count}
    if fn == 'detectHintLanguageFromText':
        return language_binding.detect_hint_language_from_text(text)
    if fn == 'resolveTtsLanguageBinding':
        language, origin = language_binding.resolve_tts_language_binding(text, args[1] if len(args) > 1 else None)
        return {'language': language, 'origin': origin}
    if fn == 'parsePhoneNumberMatch':
        parsed = shared.parse_phone_number_match(text)
        return None if parsed is None else {'plus': parsed[0], 'digits': parsed[1]}
    if fn == 'stripEmojiForTts':
        return shared.strip_emoji_for_tts(text)
    if fn == 'stripQuotationMarksForTts':
        return shared.strip_quotation_marks_for_tts(text)
    if fn == 'finalizeOptimizedText':
        return shared.finalize_optimized_text(text)
    raise AssertionError(f'unported function {fn}')


def _access(result, accessor: str):
    if accessor == 'self':
        return result
    if accessor == 'appliedRules.join':
        return ','.join(result['appliedRules'])
    return result[accessor]


def _js_string(value) -> str:
    if isinstance(value, list):
        return ','.join(_js_string(item) for item in value)
    if value is None:
        return 'null'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    return str(value)


def _matches(matcher: str, actual, expected) -> bool:
    if matcher == 'toBe':  # Object.is for JSON scalars
        if actual is None or expected is None or isinstance(actual, bool) or isinstance(expected, bool):
            return actual is expected
        return actual == expected
    if matcher == 'toEqual':
        return json.loads(json.dumps(actual)) == expected
    if matcher == 'toContain':
        return actual is not None and expected in actual
    if matcher == 'toMatch':
        if isinstance(expected, dict) and '__regex' in expected:
            return js(expected['__regex'], expected['flags'].replace('g', '').replace('y', '')).test(_js_string(actual))
        return expected in _js_string(actual)
    if matcher == 'toBeNull':
        return actual is None
    if matcher == 'toBeTruthy':
        return bool(actual)
    if matcher == 'toBeFalsy':
        return not actual
    if matcher == 'toBeGreaterThan':
        return actual > expected
    if matcher == 'toBeLessThan':
        return actual < expected
    raise AssertionError(f'unknown matcher {matcher}')


def _case_id(case: dict, index: int) -> str:
    return f"{case['file'].split('/')[-1].removesuffix('.test.ts')}-{index}"


@pytest.mark.parametrize('case', CASES['assertions'], ids=[_case_id(c, i) for i, c in enumerate(CASES['assertions'])])
def test_ts_assertion(case: dict) -> None:
    actual = _access(run_ts_function(case['fn'], case['args']), case['accessor'])
    ok = _matches(case['matcher'], actual, case['expected'])
    assert ok != case['negate'], (
        f"{case['file']} > {case['test']}\n  expect({case['fn']}(...).{case['accessor']})"
        f"{'.not' if case['negate'] else ''}.{case['matcher']}({case['expected']!r})\n  actual: {actual!r}"
    )


def test_ts_unit_call_results_are_identical() -> None:
    mismatches = []
    for call in CASES['calls']:
        got = json.loads(json.dumps(run_ts_function(call['fn'], call['args'])))
        if got != call['result']:
            mismatches.append(f"{call['file']} > {call['test']}\n  {call['fn']}{tuple(call['args'])!r}\n  ts : {call['result']!r}\n  py : {got!r}")
    assert not mismatches, f'{len(mismatches)} of {len(CASES["calls"])} calls differ\n' + '\n'.join(mismatches[:15])


def test_fixture_covers_the_ported_suites() -> None:
    files = {case['file'] for case in CASES['assertions']}
    assert len(CASES['assertions']) >= 500
    assert len(CASES['calls']) >= 450
    assert 'tts-language-optimizer/notation-prepass.test.ts' in files
    assert 'tts-language-binding/ui-locale-fallback.test.ts' in files
