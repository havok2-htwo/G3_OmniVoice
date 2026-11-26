"""Build the corpus for gen_golden.mts, and the fixtures from the node outputs.

Step 1 (before gen_golden.mts):
    python build_fixtures.py corpus <work dir> <genesis worktree>
  reads <work dir>/ts_unit_cases_raw.json (record_units.mts) and the current fixtures, writes
  <work dir>/inputs.json and <work dir>/components.json.
Step 2 (after gen_golden.mts):
    python build_fixtures.py fixtures <work dir>
  writes the three gzipped fixtures into backend/tests/fixtures/text_optimizer/.

Corpus: every input of the previous golden (so no case is ever dropped), the Genesis TTS sweep
corpus, and the inputs the ported TS unit tests pass to the pipeline. The day.month generator test
crosses ~5000 contexts; every 25th of them (sorted) keeps the shapes at a fraction of the size.
"""

from __future__ import annotations

import glob
import gzip
import json
import sys
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[2] / 'backend' / 'tests' / 'fixtures' / 'text_optimizer'
GENERATOR_FILE = 'tts-language-optimizer/dateless-day-month-or-number.test.ts'
# Their inputs reach the golden corpus; their calls would only repeat it in the unit-case replay.
CALLS_LEFT_TO_THE_GOLDEN = {GENERATOR_FILE, 'tts-language-optimizer/sweep-corpus-ratchet.test.ts'}


def _load_gz(name: str):
    with gzip.open(FIXTURES / name, 'rt', encoding='utf-8') as handle:
        return json.load(handle)


def build_corpus(work: Path, genesis: Path) -> None:
    raw = json.loads((work / 'ts_unit_cases_raw.json').read_text(encoding='utf-8'))
    inputs: dict[str, list] = {}
    for record in _load_gz('golden.json.gz'):
        # The generator's sample is drawn anew below, so the corpus does not grow on every rebuild.
        if all(GENERATOR_FILE in str(source.get('ref', '')) for source in record['sources']):
            continue
        inputs.setdefault(record['input'], list(record['sources']))

    def add(text, source):
        if isinstance(text, str):
            sources = inputs.setdefault(text, [])
            if source not in sources:
                sources.append(source)

    for path in sorted(glob.glob(str(genesis / 'tests' / 'fixtures' / 'tts-sweep-cases' / '*.json'))):
        for case in json.loads(Path(path).read_text(encoding='utf-8')):
            add(case['text'], {'origin': 'sweep', 'ref': case['id'], 'lang': case['lang']})

    generated = {}
    for call in raw['calls']:
        fn, args = call['fn'], call['args']
        if fn == 'buildDeterministicTtsFallback' and isinstance(args[0], dict):
            text, lang = args[0].get('sourceText'), args[0].get('sourceLanguage') or ''
        elif fn in ('optimizeTtsTextForLanguage', 'postProcessTtsText', 'stripMarkdownForTts') and args:
            text, lang = args[0], args[1] if len(args) > 1 and isinstance(args[1], str) else ''
        else:
            continue
        source = {'origin': 'unit_test', 'ref': f"{call['file']}#{fn}", 'lang': lang}
        if call['file'] == GENERATOR_FILE:
            generated.setdefault(text, source)
        else:
            add(text, source)
    for text in sorted(t for t in generated if isinstance(t, str) and t not in inputs)[::25]:
        add(text, generated[text])

    corpus = [{'input': text, 'sources': sources} for text, sources in inputs.items()]
    (work / 'inputs.json').write_text(json.dumps(corpus, ensure_ascii=False), encoding='utf-8')

    corpus_texts = set(inputs)
    components = [
        {key: value for key, value in record.items() if key != 'output'}
        for record in _load_gz('golden_components.json.gz')
        if record['ref'] != 'all_inputs' or record['input'] in corpus_texts
    ]
    have = {(r['fn'], r['input'], r.get('language', '__none__')) for r in components}

    def comp(fn, text, language='__none__'):
        if (fn, text, language) not in have:
            have.add((fn, text, language))
            record = {'fn': fn, 'input': text, 'ref': 'all_inputs'}
            if language != '__none__':
                record['language'] = language
            components.append(record)

    for entry in corpus:
        text = entry['input']
        for fn in ('stripMarkdownForTts', 'stripStackTraceFramesForTts', 'stripLinkReferenceDefinitionsForTts',
                   'reduceReferenceUrlsToSpokenDomainForTts', 'detectHintLanguageFromText'):
            comp(fn, text)
        for language in ('de', 'en'):
            comp('stripMarkdownForTts', text, language)
        for language in ('de', 'en', ''):
            comp('resolveTtsLanguageBinding', text, language)
        for language in ('de', 'en', None):
            comp('postProcessTtsText', text, language)
    (work / 'components.json').write_text(json.dumps(components, ensure_ascii=False), encoding='utf-8')
    print(f'inputs {len(corpus)}, components {len(components)}')


def _dedupe(items, key):
    seen, out = set(), []
    for item in items:
        k = json.dumps(key(item), ensure_ascii=False)
        if k not in seen:
            seen.add(k)
            out.append(item)
    return out


def write_fixtures(work: Path) -> None:
    raw = json.loads((work / 'ts_unit_cases_raw.json').read_text(encoding='utf-8'))
    calls = _dedupe(
        (c for c in raw['calls'] if c['file'] not in CALLS_LEFT_TO_THE_GOLDEN), lambda c: [c['fn'], c['args']]
    )
    assertions = _dedupe(
        raw['assertions'], lambda a: [a['file'], a['fn'], a['args'], a['accessor'], a['matcher'], a['negate'], a['expected']]
    )
    outputs = {
        'golden.json.gz': json.loads((work / 'golden.json').read_text(encoding='utf-8')),
        'golden_components.json.gz': json.loads((work / 'golden_components.json').read_text(encoding='utf-8')),
        'ts_unit_cases.json.gz': {'assertions': assertions, 'calls': calls},
    }
    for name, data in outputs.items():
        with gzip.GzipFile(FIXTURES / name, 'wb', mtime=0) as handle:
            handle.write(json.dumps(data, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))
    print(f"golden {len(outputs['golden.json.gz'])}, components {len(outputs['golden_components.json.gz'])}, "
          f'assertions {len(assertions)}, calls {len(calls)}')


if __name__ == '__main__':
    if sys.argv[1] == 'corpus':
        build_corpus(Path(sys.argv[2]), Path(sys.argv[3]))
    elif sys.argv[1] == 'fixtures':
        write_fixtures(Path(sys.argv[2]))
    else:
        raise SystemExit(__doc__)
