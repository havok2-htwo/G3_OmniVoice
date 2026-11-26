"""Append recording wrappers to the ported TS functions in the golden worktree (revert with git checkout)."""
import sys
from pathlib import Path

root = Path(sys.argv[1])
FUNCS = {
    'src/utils/ttsLanguageOptimizer/shared.ts': [
        'stripEmojiForTts', 'isOnlyShushing', 'skipShushingForTts', 'stripEmojiLinesForTts', 'stripQuotationMarksForTts',
        'unifyLineBreaksForTts', 'removeInvisibleCharactersForTts', 'removeZeroWidthJoinersForTts',
        'trimKeepingListItemDepthForTts', 'holdsDotNameBeforeExtension', 'finalizeOptimizedText', 'parsePhoneNumberMatch',
        'restoreHeldDayMonths',
    ],
    'src/utils/ttsLanguageOptimizer/markdownStripper.ts': ['isUnspokenMarkdownOnly', 'stripMarkdownForTts'],
    'src/utils/ttsLanguageOptimizer/markdownLineStructure.ts': ['applyMarkdownLineStructure'],
    'src/utils/ttsLanguageOptimizer/markdownInline.ts': ['stripInlineMarkdownForTts', 'resolveBackslashEscapesForTts'],
    'src/utils/ttsLanguageOptimizer/markdownHtml.ts': ['stripHtmlBlocksForTts', 'stripInlineHtmlForTts', 'decodeHtmlEntitiesForTts'],
    'src/utils/ttsLanguageOptimizer/latexMath.ts': ['reduceLatexForTts', 'readInlineMathForTts', 'unwrapInlineMathForTts'],
    'src/utils/ttsLanguageOptimizer/referenceLinkStripper.ts': [
        'stripStackTraceFramesForTts', 'isLinkReferenceDefinitionLine', 'stripLinkReferenceDefinitionsForTts',
        'reduceReferenceUrlsToSpokenDomainForTts',
    ],
    'src/utils/ttsLanguageOptimizer/index.ts': ['optimizeTtsTextForLanguage'],
    'src/utils/ttsLanguageOptimizer/hintInjector.ts': ['detectHintLanguageFromText', 'resolveTtsLanguageBinding'],
    'src/utils/ttsLanguageOptimizer/textOrthography.ts': ['textOrthographyAllowsLanguage'],
    'src/utils/ttsLanguageOptimizer/de.ts': ['joinGermanSpacedThousands', 'optimizeGermanTtsText'],
    'src/utils/ttsLanguageOptimizer/en.ts': ['optimizeEnglishTtsText'],
    'src/utils/ttsLanguageOptimizer/notationPrepass.ts': ['isSlashDayMonth'],
    'src/utils/ttsTextPostProcessor.ts': ['postProcessTtsText'],
    'src/runtime/ttsDeterministicFallback.ts': ['buildDeterministicTtsFallback'],
    'tests/lib/runner.ts': [],
}
for rel, names in FUNCS.items():
    path = root / rel
    text = path.read_text(encoding='utf-8')
    lines = ['', '// --- golden recorder (temporary) ---']
    for name in names:
        lines.append(f"{name} = (globalThis as any).__ttsWrap('{name}', {name});")
    if rel.endswith('runner.ts'):
        lines.append('expect = (globalThis as any).__ttsWrapExpect(expect);')
    path.write_text(text + '\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
print('instrumented', len(FUNCS))
