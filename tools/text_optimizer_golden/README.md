# Text optimizer golden harness

Regenerates the parity fixtures of `backend/src/omnivoice_tts_server/text_optimizer/` (the Python
port of the Genesis3 deterministic TTS pipeline, `buildDeterministicTtsFallback`) from the
unmodified Genesis3 TypeScript sources.

Reference: Genesis3 `main` at `0e1ded39e` (2026-11-26). The first port was made against
`ac7863b35` (2026-10-15).

Needs a Genesis3 checkout of the reference commit with its `node_modules` (a separate worktree;
`record_units.mts` needs one whose sources it may temporarily instrument). Run from this folder,
`WORK` is any scratch directory, `G3` the Genesis3 worktree:

```bash
python instrument.py "$G3"                                   # wraps the ported functions (temporary)
node "$G3/node_modules/tsx/dist/cli.mjs" record_units.mts "$G3" "$WORK/ts_unit_cases_raw.json" \
  tts-language-optimizer tts-language-binding tts-hint-language-detection tts-text-post-processor
git -C "$G3" checkout -- src tests                           # removes the wrappers again
python build_fixtures.py corpus "$WORK" "$G3"                # inputs.json, components.json
node "$G3/node_modules/tsx/dist/cli.mjs" gen_golden.mts "$G3" "$WORK/inputs.json" "$WORK/components.json" "$WORK"
python build_fixtures.py fixtures "$WORK"                    # writes the three .json.gz fixtures
```

- `record_units.mts` runs the TS unit tests of the ported functions and records every top-level
  call of a ported function (arguments and result) and every assertion on such a result. Tests
  that do not pass are left out; a few source guards fail when run outside the Genesis3 cwd and
  touch no ported function.
- `gen_golden.mts` writes `optimizeTtsTextForLanguage` and `buildDeterministicTtsFallback` for
  every input x de/en x floor/final, and the per-stage component records.
- The corpus keeps every input of the previous fixtures, so an update only ever adds cases.

Then `pytest tests/test_text_optimizer_parity.py tests/test_text_optimizer_ts_unit_cases.py`
in `backend/` shows what the Python port has to follow (a per-rule mismatch histogram).
