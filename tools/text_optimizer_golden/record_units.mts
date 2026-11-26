// Assertion-recording runner for the TS unit tests of the ported functions.
// Usage: node <tsx cli> record_units.mts <genesis root (instrumented)> <out json> <test dir>...
import { readdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const [root, outPath, ...testDirs] = process.argv.slice(2);

type Call = { fn: string; args: unknown; result: unknown; raw: unknown; seq: number };
type Assertion = { actual: unknown; matcher: string; negate: boolean; expected: unknown; seq: number };

let depth = 0;
let seq = 0;
let calls: Call[] = [];
let assertions: Assertion[] = [];

function plain(value: unknown): unknown {
  // JSON-clone; undefined inside objects drops (as JSON.stringify does), a top-level undefined is null.
  if (value === undefined) return null;
  return JSON.parse(JSON.stringify(value));
}

function serializable(value: unknown): boolean {
  try {
    const seen = JSON.stringify(value, (_k, v) => {
      if (typeof v === 'function' || v instanceof RegExp || typeof v === 'bigint' || v instanceof Set || v instanceof Map) throw new Error('x');
      return v;
    });
    return seen !== undefined || value === undefined;
  } catch {
    return false;
  }
}

(globalThis as any).__ttsWrap = (name: string, fn: (...args: any[]) => any) =>
  function wrapped(this: unknown, ...args: any[]) {
    if (depth > 0) return fn.apply(this, args);
    const argsOk = serializable(args);
    const argsCopy = argsOk ? plain(args) : null;
    depth += 1;
    try {
      const result = fn.apply(this, args);
      if (argsOk && serializable(result)) calls.push({ fn: name, args: argsCopy, result: plain(result), raw: result, seq: seq++ });
      return result;
    } finally {
      depth -= 1;
    }
  };

function expectedValue(value: unknown): unknown {
  if (value instanceof RegExp) return { __regex: value.source, flags: value.flags };
  return value === undefined ? null : plain(value);
}

(globalThis as any).__ttsWrapExpect = (expect: (actual: any) => any) =>
  function wrappedExpect(actual: any) {
    const inner = expect(actual);
    const record = (matcher: string, negate: boolean, expected: unknown) =>
      assertions.push({ actual, matcher, negate, expected: expectedValue(expected), seq: seq++ });
    const wrap = (target: any, negate: boolean) => {
      const out: any = {};
      for (const key of Object.keys(target)) {
        if (key === 'not') continue;
        out[key] = (...a: any[]) => {
          record(key, negate, a[0]);
          return target[key](...a);
        };
      }
      return out;
    };
    const out = wrap(inner, false);
    out.not = wrap(inner.not, true);
    return out;
  };

const ACCESSORS = [
  'self', 'text', 'appliedRules', 'appliedRules.join', 'language', 'authorization', 'reducedUrls', 'strippedFrames',
  'strippedDefinitions', 'origin', 'plus', 'digits', 'heldDayMonths',
];

function access(result: any, accessor: string): unknown {
  if (accessor === 'self') return result;
  if (result === null || typeof result !== 'object') return Symbol('none');
  if (accessor === 'appliedRules.join') return Array.isArray(result.appliedRules) ? result.appliedRules.join(',') : Symbol('none');
  return accessor in result ? result[accessor] : Symbol('none');
}

function same(a: unknown, b: unknown): boolean {
  if (typeof a === 'object' && a !== null) return a === b;
  return Object.is(a, b);
}

const runner = await import(pathToFileURL(path.join(root, 'tests/lib/runner.ts')).href);
const outAssertions: any[] = [];
const outCalls: any[] = [];
let passedTests = 0;
let failedTests = 0;
let linked = 0;
let unlinked = 0;

for (const dir of testDirs) {
  const files = readdirSync(path.join(root, 'tests', dir)).filter((f) => f.endsWith('.test.ts')).sort();
  for (const file of files) {
    const rel = `${dir}/${file}`;
    const before = runner.suiteCount();
    await import(pathToFileURL(path.join(root, 'tests', dir, file)).href);
    const after = runner.suiteCount();
    let suiteNames: string[] = [];
    const perTest: Array<{ name: string; calls: Call[]; assertions: Assertion[] }> = [];
    const { suiteNames: names, results } = await runner.runSuites(before, after, () => true, {
      beforeTest: () => {
        calls = [];
        assertions = [];
      },
      afterTest: (result: any) => {
        perTest.push({ name: `${result.suite}|${result.name}|${result.outcome.status}`, calls, assertions });
      },
    });
    suiteNames = names;
    perTest.forEach((t, i) => {
      const r = results[i];
      const status = r.outcome.status;
      if (status !== 'pass') {
        if (status === 'fail') {
          failedTests += 1;
          console.error(`FAIL ${rel} > ${r.name}: ${r.outcome.message}`);
        }
        return;
      }
      passedTests += 1;
      const testName = `${suiteNames[r.suite]} > ${r.name}`;
      for (const c of t.calls) outCalls.push({ file: rel, test: testName, fn: c.fn, args: c.args, result: c.result });
      for (const a of t.assertions) {
        let found: { call: Call; accessor: string } | null = null;
        for (const c of [...t.calls].reverse()) {
          if (c.seq > a.seq) continue;
          const accessor = ACCESSORS.find((acc) => same(access(c.raw, acc), a.actual));
          if (accessor) {
            found = { call: c, accessor };
            break;
          }
        }
        if (!found) {
          unlinked += 1;
          continue;
        }
        linked += 1;
        outAssertions.push({
          file: rel,
          test: testName,
          fn: found.call.fn,
          args: found.call.args,
          accessor: found.accessor,
          matcher: a.matcher,
          negate: a.negate,
          expected: a.expected,
        });
      }
    });
  }
}

writeFileSync(outPath, JSON.stringify({ assertions: outAssertions, calls: outCalls }));
console.log(`tests passed ${passedTests}, failed ${failedTests}; assertions linked ${linked}, unlinked ${unlinked}; calls ${outCalls.length}`);
