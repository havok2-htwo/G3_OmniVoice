"""The Genesis3 TS unit tests of the ported functions, replayed against the Python port.

tests/fixtures/text_optimizer/ts_unit_cases.json.gz was recorded by running the TS test files
(tts-language-optimizer, tts-language-binding, tts-hint-language-detection,
tts-text-post-processor) under an assertion-recording runner that wraps the ported functions:

* ``assertions``: every ``expect(actual).<matcher>(expected)`` of a passing test whose ``actual``
  came from a ported function, with the call (function + arguments) and the accessor that
  produced ``actual`` (``.text``, ``.appliedRules.join(',')``, ...). Replayed one by one below.
* ``calls``: every top-level call of a ported function made by those tests (deduplicated) with
  its full TS result; the Python result must be identical. This also covers assertions on
  derived values that cannot be linked to a single accessor. The calls of the day.month
  generator test and of the sweep ratchet are left out here: their inputs are in the golden
  corpus (golden.json.gz).

Not covered: assertions on the hint injector (injectTtsHintsForLanguage), the spoken number
fidelity gate, the language detector, the opaque identifier reduction, SentenceChunker and the
Markdown table parser, none of which are ported (G3_OmniVoice keeps its own sentence splitter).
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from omnivoice_tts_server.text_optimizer import language_binding, latex_math, markdown_html, markdown_inline
from omnivoice_tts_server.text_optimizer import markdown_stripper, pipeline, post_processor, reference_links, shared
from omnivoice_tts_server.text_optimizer import text_orthography
from omnivoice_tts_server.text_optimizer.de import join_german_spaced_thousands
from omnivoice_tts_server.text_optimizer.jsre import js
from omnivoice_tts_server.text_optimizer.notation_prepass import is_slash_day_month

FIXTURE = Path(__file__).resolve().parent / 'fixtures' / 'text_optimizer' / 'ts_unit_cases.json.gz'

with gzip.open(FIXTURE, 'rt', encoding='utf-8') as _handle:
    CASES = json.load(_handle)

_OPTION_KEYS = {
    'continuesNumberedList': 'continues_numbered_list',
    'startsInsideLine': 'starts_inside_line',
    'answerText': 'answer_text',
    'holdListOpener': 'hold_list_opener',
    'dayMonthsSettled': 'day_months_settled',
    'keepHeldDayMonths': 'keep_held_day_months',
}


def _optimizer_result(result) -> dict:
    out = {'text': result.text, 'appliedRules': result.applied_rules, 'language': result.language}
    if result.held_day_months:
        out['heldDayMonths'] = result.held_day_months
    return out


def _string_arg(args: list, index: int):
    return args[index] if len(args) > index and isinstance(args[index], str) else None


def run_ts_function(fn: str, args: list):
    """Call the Python port the way the TS test called the TS function; return a JS-shaped value."""
    text = args[0] if args else None
    if fn == 'optimizeTtsTextForLanguage':
        options = args[2] if len(args) > 2 and isinstance(args[2], dict) else {}
        flags = {_OPTION_KEYS[key]: value for key, value in options.items() if key in _OPTION_KEYS}
        result = pipeline.optimize_tts_text_for_language(
            text or '', _string_arg(args, 1), options.get('stage') or 'final', options=shared.TtsOptimizeOptions(**flags)
        )
        return _optimizer_result(result)
    if fn == 'buildDeterministicTtsFallback':
        params = args[0]
        source = params.get('sourceText')
        result = pipeline.build_deterministic_tts_text(
            source if isinstance(source, str) else '',
            params.get('sourceLanguage') or '',
            enable_algorithmic_optimization=bool(params.get('enableAlgorithmicOptimization')),
            stage=params.get('stage') or 'final',
            continues_numbered_list=bool(params.get('continuesNumberedList')),
            starts_inside_line=bool(params.get('startsInsideLine')),
            mixed_status_run=bool(params.get('mixedStatusRun')),
            answer_text=params.get('answerText'),
        )
        return {
            'text': result.text,
            'appliedRules': result.applied_rules,
            'language': result.language,
            'authorization': result.authorization,
        }
    if fn == 'postProcessTtsText':
        processed, rules = post_processor.post_process_tts_text(text or '', _string_arg(args, 1))
        return {'text': processed, 'appliedRules': rules}
    if fn == 'stripMarkdownForTts':
        options = args[2] if len(args) > 2 and isinstance(args[2], dict) else {}
        return markdown_stripper.strip_markdown_for_tts(
            text, _string_arg(args, 1), mixed_status_run=bool(options.get('mixedStatusRun'))
        )
    if fn == 'isUnspokenMarkdownOnly':
        return markdown_stripper.is_unspoken_markdown_only(text)
    if fn == 'reduceReferenceUrlsToSpokenDomainForTts':
        reduced, count = reference_links.reduce_reference_urls_to_spoken_domain_for_tts(text)
        return {'text': reduced, 'reducedUrls': count}
    if fn == 'stripStackTraceFramesForTts':
        stripped, count = reference_links.strip_stack_trace_frames_for_tts(text)
        return {'text': stripped, 'strippedFrames': count}
    if fn == 'stripLinkReferenceDefinitionsForTts':
        stripped, count = reference_links.strip_link_reference_definitions_for_tts(text)
        return {'text': stripped, 'strippedDefinitions': count}
    if fn == 'isLinkReferenceDefinitionLine':
        return reference_links.is_link_reference_definition_line(text)
    if fn == 'detectHintLanguageFromText':
        return language_binding.detect_hint_language_from_text(text)
    if fn == 'resolveTtsLanguageBinding':
        verdict = args[2] if len(args) > 2 and isinstance(args[2], dict) else None
        language, origin = language_binding.resolve_tts_language_binding(text, _string_arg(args, 1), verdict)
        return {'language': language, 'origin': origin}
    if fn == 'parsePhoneNumberMatch':
        parsed = shared.parse_phone_number_match(text)
        return None if parsed is None else {'plus': parsed[0], 'digits': parsed[1]}
    if fn == 'joinGermanSpacedThousands':
        return join_german_spaced_thousands(text or '')
    if fn == 'textOrthographyAllowsLanguage':
        return text_orthography.text_orthography_allows_language(text or '', args[1])
    if fn == 'isSlashDayMonth':
        return is_slash_day_month(args[0], args[1])
    if fn == 'stripInlineMarkdownForTts':
        return markdown_inline.strip_inline_markdown_for_tts(text, args[1] if len(args) > 1 else True)
    if fn == 'resolveBackslashEscapesForTts':
        return markdown_inline.resolve_backslash_escapes_for_tts(text)
    simple = {
        'stripEmojiForTts': shared.strip_emoji_for_tts,
        'stripEmojiLinesForTts': shared.strip_emoji_lines_for_tts,
        'stripQuotationMarksForTts': shared.strip_quotation_marks_for_tts,
        'finalizeOptimizedText': shared.finalize_optimized_text,
        'isOnlyShushing': shared.is_only_shushing,
        'skipShushingForTts': shared.skip_shushing_for_tts,
        'unifyLineBreaksForTts': shared.unify_line_breaks_for_tts,
        'removeInvisibleCharactersForTts': shared.remove_invisible_characters_for_tts,
        'removeZeroWidthJoinersForTts': shared.remove_zero_width_joiners_for_tts,
        'trimKeepingListItemDepthForTts': shared.trim_keeping_list_item_depth_for_tts,
        'holdsDotNameBeforeExtension': shared.holds_dot_name_before_extension,
        'stripHtmlBlocksForTts': markdown_html.strip_html_blocks_for_tts,
        'stripInlineHtmlForTts': markdown_html.strip_inline_html_for_tts,
        'decodeHtmlEntitiesForTts': markdown_html.decode_html_entities_for_tts,
        'reduceLatexForTts': latex_math.reduce_latex_for_tts,
        'readInlineMathForTts': latex_math.read_inline_math_for_tts,
        'unwrapInlineMathForTts': latex_math.unwrap_inline_math_for_tts,
    }
    if fn in simple:
        return simple[fn](text)
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
