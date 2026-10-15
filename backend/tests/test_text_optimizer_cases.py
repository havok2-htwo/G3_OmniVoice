"""OmniVoice-specific behaviour of the text optimizer (not covered by the TS golden)."""

from __future__ import annotations

import time

import pytest

from omnivoice_tts_server.text_optimizer import (
    optimize_text,
    optimize_tts_text,
    optimize_tts_text_for_language,
    parse_text_optimization_mode,
    resolve_optimizer_language,
    resolve_text_optimization_mode,
)
from omnivoice_tts_server.text_optimizer.jsre import js, js_number, js_replace, js_trim
from omnivoice_tts_server.text_optimizer.omnivoice_tags import protect_inline_tags, restore_inline_tags


# --- OmniVoice inline control spans ----------------------------------------------------------

def test_inline_tags_survive_german_pipeline() -> None:
    outcome = optimize_text('Wir lachen [laughter] über Version 2.4.1 und [B EY1 S].', 'de')
    assert outcome.text == 'Wir lachen [laughter] über Version zwei Punkt vier Punkt eins und [B EY1 S].'
    assert outcome.protected_tags == 2
    assert 'version_de' in outcome.applied_rules


def test_inline_tags_survive_english_pipeline_with_auto_binding() -> None:
    outcome = optimize_text('That was funny [laughter], see [B EY1 S] at 3:30 PM on 12/24/2026.', 'Auto')
    assert outcome.language == 'en'
    assert outcome.binding_origin == 'chunk_notation'
    assert '[laughter]' in outcome.text and '[B EY1 S]' in outcome.text
    assert 'three thirty P M' in outcome.text
    assert 'December' in outcome.text


def test_unprotected_genesis_pipeline_would_destroy_the_tags() -> None:
    # Documents why the wrapper exists: the TS pipeline speaks the tag text.
    raw = optimize_tts_text('Wir lachen [laughter] und [B EY1 S].', 'de')
    assert '[laughter]' not in raw.text
    assert 'B EY 1 S' in raw.text


def test_markdown_links_are_not_treated_as_tags() -> None:
    outcome = optimize_text('Siehe [die Doku](https://example.com/docs) ab 3 Uhr.', 'de')
    assert outcome.text == 'Siehe die Doku ab drei Uhr.'
    assert 'markdown_strip' in outcome.applied_rules


def test_genesis_quirks_kept_for_parity() -> None:
    # date_de_dm lower-cases its prefix and takes the sentence period as the date's dot.
    outcome = optimize_text('Am 3.10. geht es los.', 'de')
    assert outcome.text == 'am dritten Oktober geht es los.'
    assert optimize_text('Wir starten am 3.10.', 'de').text == 'Wir starten am dritten Oktober'
    # Exclamation marks become periods and quotes are dropped (post-processor).
    assert optimize_text('Super! Er sagt „Hallo“ um 9 Uhr.', 'de').text == 'Super. Er sagt Hallo um neun Uhr.'


def test_tag_placeholders_roundtrip_and_respect_existing_private_use_text() -> None:
    protected, spans = protect_inline_tags('a [sigh] b [B EY1 S] c')
    assert spans == ['[sigh]', '[B EY1 S]']
    assert '[' not in protected
    assert restore_inline_tags(protected, spans) == 'a [sigh] b [B EY1 S] c'
    untouched, none = protect_inline_tags('x  [sigh]')
    assert none == [] and untouched == 'x  [sigh]'


# --- language binding --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ('language', 'expected'),
    [
        ('de', 'de'), ('DE', 'de'), ('de-DE', 'de'), ('de_AT', 'de'), ('Deutsch', 'de'), ('German', 'de'),
        ('en', 'en'), ('en-US', 'en'), ('English', 'en'), ('Englisch', 'en'),
        ('fr', None), ('Français', None), ('zh', None), ('Dene', None),
    ],
)
def test_explicit_request_language_mapping(language: str, expected: str | None) -> None:
    bound, origin = resolve_optimizer_language(language, 'Am 24.09.2026 um 14:30 Uhr.')
    assert bound == expected
    assert origin == ('request' if expected else 'unsupported_language')


def test_auto_binding_uses_notation_and_passes_through_without_it() -> None:
    assert resolve_optimizer_language('Auto', 'Treffen am 24.09.2026 um 14:30 Uhr.') == ('de', 'chunk_notation')
    assert resolve_optimizer_language(None, 'Meet me on March 3, 2026 at 4:15 PM.') == ('en', 'chunk_notation')
    text = 'Ich habe 3 Katzen.'
    outcome = optimize_text(text, 'Auto')
    assert outcome.language is None and outcome.binding_origin == 'none'
    assert outcome.text == text and outcome.applied_rules == []


def test_other_languages_are_a_no_op() -> None:
    text = 'Rendez-vous le 3 mars à 14h30, z.B. demain!'
    outcome = optimize_text(text, 'fr')
    assert outcome.text == text
    assert outcome.applied_rules == []
    # The bare language dispatch also leaves non-de/en languages alone (finalize only).
    assert optimize_tts_text_for_language('a  b', 'fr').text == 'a b'


def test_explicit_language_wins_over_notation() -> None:
    # English notation, but the request says German: the German rule set runs.
    outcome = optimize_text('Am 3. Oktober um 14:30 Uhr, 3rd floor.', 'de')
    assert outcome.language == 'de'
    assert 'dritten Oktober' in outcome.text


# --- wrapper behaviour ---------------------------------------------------------------------------

def test_text_without_any_rule_is_returned_unchanged() -> None:
    text = 'Hallo  Welt,\n\nwie geht es dir ?'
    outcome = optimize_text(text, 'de')
    assert outcome.applied_rules == []
    assert outcome.text == text


def test_paragraph_breaks_become_single_newlines() -> None:
    outcome = optimize_text('Überschrift\n\n  \nUm 9 Uhr gibt es 2 Sorten.', 'de')
    assert outcome.text == 'Überschrift\nUm neun Uhr gibt es zwei Sorten.'


def test_list_lines_keep_their_newlines() -> None:
    outcome = optimize_text('Einkauf:\n1. Butter\n2. Milch\n3. Eier', 'de')
    assert outcome.text == 'Einkauf:\nErstens: Butter\nZweitens: Milch\nDrittens: Eier'


def test_url_port_mojibake_is_rescued_by_the_post_processor() -> None:
    # de.ts writes 'fÃ¼nf' in GERMAN_DIGIT_WORD_SOURCE, so the optimizer alone keeps a spoken port
    # containing a 5 twice (kept for parity, see the parity notes); the post-processor's echo
    # collapse removes the duplicate in the full pipeline.
    text = 'Server: http://localhost:5000 (fünf null null null) läuft.'
    optimizer_only = optimize_tts_text_for_language(text, 'de').text
    assert optimizer_only.count('fünf null null null') == 2
    assert optimize_tts_text(text, 'de').text == 'Server: http://localhost: fünf null null null läuft.'
    ok = optimize_tts_text_for_language('Server: http://localhost:8080 (acht null acht null) läuft.', 'de').text
    assert ok.count('acht null acht null') == 1


def test_sonja_example_end_to_end() -> None:
    outcome = optimize_text(
        'Hallo, ich bin Sonja. Am 24.09.2026 um 14:30 Uhr treffen wir uns in Raum 3.12, Tel. +49 341 1234567.', 'de'
    )
    assert outcome.text == (
        'Hallo, ich bin Sonja. Am vierundzwanzigsten September zweitausendsechsundzwanzig um vierzehn Uhr dreißig '
        'treffen wir uns in Raum drei komma eins zwei, Telefon plus vier neun drei vier eins eins zwei drei vier '
        'fünf sechs sieben.'
    )


def test_max_size_input_stays_fast() -> None:
    chunk = 'Am 24.09.2026 um 14:30 Uhr kostet es 29,90 € (inkl. 19 % MwSt.), Tel. +49 341 1234567, siehe https://example.com/x. '
    text = (chunk * 250)[:25000]
    started = time.perf_counter()
    outcome = optimize_text(text, 'de')
    elapsed = time.perf_counter() - started
    assert 'neunundzwanzig Euro und neunzig Cent' in outcome.text
    assert elapsed < 15.0, elapsed


# --- request flag ---------------------------------------------------------------------------------

@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        (None, None), ('', None), ('default', None), (True, 'deterministic'), (False, 'off'),
        ('deterministic', 'deterministic'), ('DETERMINISTIC', 'deterministic'), ('off', 'off'),
        ('on', 'deterministic'), ('true', 'deterministic'), ('false', 'off'),
    ],
)
def test_parse_text_optimization_mode(value, expected) -> None:
    assert parse_text_optimization_mode(value) == expected


@pytest.mark.parametrize('value', ['llm', 'fancy', 3, ['deterministic']])
def test_parse_text_optimization_mode_rejects(value) -> None:
    with pytest.raises(ValueError):
        parse_text_optimization_mode(value)


def test_resolve_mode_precedence_and_scopes() -> None:
    assert resolve_text_optimization_mode(None, {}, 'off', 'public') == 'off'
    assert resolve_text_optimization_mode(None, {}, 'deterministic', 'public') == 'deterministic'
    assert resolve_text_optimization_mode('off', {'text_optimization': 'deterministic'}, 'deterministic', 'public') == 'off'
    assert resolve_text_optimization_mode(None, {'text_optimization': True}, 'off', 'public') == 'deterministic'
    for scope in ('benchmark', 'wer-benchmark', 'finetune'):
        assert resolve_text_optimization_mode(None, {}, 'deterministic', scope) == 'off'
    assert resolve_text_optimization_mode('deterministic', {}, 'off', 'benchmark') == 'deterministic'
    # A broken persisted default never breaks requests.
    assert resolve_text_optimization_mode(None, {}, 'garbage', 'public') == 'off'


# --- JS regex compatibility layer -------------------------------------------------------------------

def test_jsre_shorthands_are_ascii_like_javascript() -> None:
    assert not js(r'\d', 'u').test('٣')
    assert not js(r'\w', 'u').test('ü')
    assert not js(r'\büber', 'u').test('über')
    assert js(r'\s', '').test('﻿')
    assert not js(r'\s', '').test('\x85')
    assert js(r'[^\S\r\n]', '').test('\t') and not js(r'[^\S\r\n]', '').test('\n')


def test_jsre_anchors_and_dot() -> None:
    assert not js(r'a$', '').test('a\n')
    assert js(r'a$', 'm').test('a\nb')
    assert not js(r'a.b', '').test('a b')
    assert js(r'a.b', 's').test('a\nb')
    assert js(r'p{L}', '').test('p{L}')  # without the u flag \p is a plain "p"
    assert js(r'\p{L}', 'u').test('ä')


def test_jsre_replace_semantics() -> None:
    seen = []

    def callback(match, first, second, offset, source):
        seen.append((match, first, second, offset))
        return f'<{first or ""}>'

    assert js_replace(js(r'(a)|(b)', 'g'), 'xab', callback) == 'x<a><>'
    assert seen == [('a', 'a', None, 1), ('b', None, 'b', 2)]
    assert js_replace(js(r'(\d)(x)?', 'g'), '1x2', '[$1$2|$&|$$]') == '[1x|1x|$][2|2|$]'
    assert js_replace(js(r'o', ''), 'foo', '0') == 'f0o'
    assert js_replace(js(r'x*', 'g'), 'abxd', '-') == '-a-b--d-'


def test_jsre_sticky_and_number() -> None:
    sticky = js(r'(?<![a-z])\d+', 'y')
    assert sticky.match_at('ab12', 2) is None
    assert sticky.match_at('ab 12', 3).group(0) == '12'
    assert js_number('') == 0 and js_number(' 08 ') == 8 and js_number('1e3') == 1000
    assert js_number('1.2.3') != js_number('1.2.3')  # NaN
    assert js_trim('﻿ a 　') == 'a'
