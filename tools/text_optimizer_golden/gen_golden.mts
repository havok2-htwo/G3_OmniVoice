// Golden generator for the G3_OmniVoice text optimizer port.
// Usage: node <tsx cli> gen_golden.ts <genesis root with src/> <inputs.json> <components.json> <out dir>
// Runs the unmodified TS sources; writes golden.json and golden_components.json.
import { readFileSync, writeFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import path from 'node:path';

const [root, inputsPath, componentsPath, outDir] = process.argv.slice(2);
const mod = async (rel: string) => import(pathToFileURL(path.join(root, rel)).href);

const index = await mod('src/utils/ttsLanguageOptimizer/index.ts');
const fallback = await mod('src/runtime/ttsDeterministicFallback.ts');
const post = await mod('src/utils/ttsTextPostProcessor.ts');

const inputs: Array<{ input: string; sources: unknown[] }> = JSON.parse(readFileSync(inputsPath, 'utf8'));
const components: Array<{ fn: string; input: string; language?: string | null; ref: string }> = JSON.parse(
  readFileSync(componentsPath, 'utf8')
);

const golden: unknown[] = [];
for (const { input, sources } of inputs) {
  for (const language of ['de', 'en']) {
    for (const stage of ['floor', 'final']) {
      const opt = index.optimizeTtsTextForLanguage(input, language, { stage });
      const pipe = fallback.buildDeterministicTtsFallback({
        sourceText: input,
        sourceLanguage: language,
        enableAlgorithmicOptimization: true,
        stage,
      });
      golden.push({
        input,
        language,
        stage,
        sources,
        output: opt.text,
        appliedRules: opt.appliedRules,
        pipeline: { output: pipe.text, appliedRules: pipe.appliedRules, authorization: pipe.authorization },
      });
    }
  }
}

function component(fn: string, input: string, language: string | null | undefined): unknown {
  switch (fn) {
    case 'stripMarkdownForTts':
      return language == null ? index.stripMarkdownForTts(input) : index.stripMarkdownForTts(input, language);
    case 'stripStackTraceFramesForTts':
      return index.stripStackTraceFramesForTts(input);
    case 'stripLinkReferenceDefinitionsForTts':
      return index.stripLinkReferenceDefinitionsForTts(input);
    case 'reduceReferenceUrlsToSpokenDomainForTts':
      return index.reduceReferenceUrlsToSpokenDomainForTts(input);
    case 'detectHintLanguageFromText':
      return index.detectHintLanguageFromText(input);
    case 'resolveTtsLanguageBinding':
      return index.resolveTtsLanguageBinding(input, language);
    case 'postProcessTtsText':
      return language == null ? post.postProcessTtsText(input) : post.postProcessTtsText(input, language);
    default:
      throw new Error(`unknown component ${fn}`);
  }
}

const componentRecords = components
  .filter((c) => c.fn !== 'stripLinkReferenceDefinitionsForTts' || typeof index.stripLinkReferenceDefinitionsForTts === 'function')
  .map((c) => ({ ...c, output: component(c.fn, c.input, c.language) }));

writeFileSync(path.join(outDir, 'golden.json'), JSON.stringify(golden));
writeFileSync(path.join(outDir, 'golden_components.json'), JSON.stringify(componentRecords));
console.log(`golden ${golden.length}, components ${componentRecords.length}`);
