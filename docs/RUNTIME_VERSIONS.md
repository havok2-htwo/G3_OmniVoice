# Runtime Versions

Known-good Windows/CUDA runtime matrix for the local `.conda-env`.

These versions were verified together on Windows with CUDA PyTorch:

| Package | Version |
| --- | --- |
| Python | 3.12.13 |
| torch | 2.10.0+cu130 |
| torchvision | 0.25.0+cu130 |
| torchaudio | 2.10.0+cu130 |
| triton-windows | 3.6.0.post26 |
| omnivoice | 0.2.1 (upstream master `08be0b4`, see below) |
| transformers | 5.10.2 |
| huggingface_hub | 1.18.0 |
| kernels | 0.14.1 |
| kernels-data | 0.15.2 |
| gradio | 6.17.3 |
| tomlkit | 0.14.0 |
| numpy | 2.4.4 |
| soundfile | 0.14.0 |

`requirements-lock.txt` is the full snapshot of the live `.conda-env`.

## omnivoice pin

`backend/pyproject.toml` installs upstream `k2-fsa/OmniVoice` master at commit
`08be0b4ccbac3e13e374e86fbfead4b4cac343e2` (2026-08-24) from its GitHub archive
URL. PyPI's newest release is still 0.2.1, and that commit also reports
`0.2.1`, so `importlib.metadata.version('omnivoice')` cannot tell them apart.

- `omnivoice/models/omnivoice.py` is byte-identical between tag 0.2.1 and
  `08be0b4`. Everything after 0.2.1 is opt-in: the FlashInfer path
  (`omnivoice_flashinfer.py`), LoRA finetuning through PEFT (`use_lora`, off by
  default) and a fix to the denoise script.
- 0.1.5 -> 0.2.1 changes nothing our server uses by default. With the same seed,
  0.1.5 and the pinned commit produced bit-identical PCM for all 30 benchmark
  utterances (SDPA, bf16, compile on, Sonja voice).
- New generate() options the server passes through from request `metadata`:
  `pad_duration`, `fade_duration` (default 0.1 s each, as before) and
  `normalize_text` (see `API_DOCUMENTATION.md`).
- Install into an existing env without touching torch:

```powershell
& .\.conda-env\python.exe -m pip install --no-deps "omnivoice @ https://github.com/k2-fsa/OmniVoice/archive/08be0b4ccbac3e13e374e86fbfead4b4cac343e2.zip"
```

## Attention backends

`attention_implementation` accepts `sdpa` (default), `flex_attention` (alias
`flex`) and `eager`. Unknown values are rejected by `PUT /api/admin/settings`
and fall back to `sdpa` at load. `GET /api/admin/settings` reports the backend
the resident model really runs as `attention_backend`, and the reason for any
fallback as `attention_backend_note`.

- **flex_attention** works correctly only with the server's patch. OmniVoice
  passes a dense bool mask, and Transformers' flex integration adds it as a
  +1/+0 score bias, so padded keys are not masked (true for 0.1.5, 0.2.1 and
  master). The server converts the mask to a `BlockMask` before the LLM.
  Inductor's default flex template also needs more shared memory than RTX
  30/40/50 GPUs have (114688 > 101376 bytes for head_dim 128). The compile then
  fails, and dynamo silently runs unfused eager flex, about 10x slower. On GPUs
  with less than 160 KB of shared memory per block, the server uses
  `BLOCK_M=64, BLOCK_N=64, num_stages=2`.
- **FlashInfer** (`enable_flashinfer`, env `OMNIVOICE_TTS_ENABLE_FLASHINFER`)
  uses upstream's `apply_flashinfer` on Linux + CUDA when `flashinfer` is
  installed.
  - The model then loads in fp16, because FlashInfer's packed attention is
    planned for fp16 and bf16 raises.
  - `torch.compile` is skipped, and fp8 is incompatible.
  - CUDA graphs stay off: upstream's graph cache keys on the exact tuple of
    sequence lengths and never evicts.
  - A 2-step test generation runs before the model goes live. If the import,
    the patch or that generation fails, the server reloads with
    `attention_implementation`.
  - It needs roughly 0.7 GB more VRAM (fused QKV and gate/up weight copies, plus
    a 128 MB workspace), so raise `vram_budget_mb` to match.
  - On Windows the setting only logs why it is unused.

Benchmark on the RTX 4090, shared with the live Whisper/DIA/EMBED services, so
expect noise of about ±10-30 %. Settings: bf16, compile on, `num_step` 18,
guidance 2.0, Sonja voice, seed 1234, latency in ms as min / median of 3-5 warm
runs:

| Case | 0.1.5 SDPA | 08be0b4 SDPA | 08be0b4 flex (patched) |
| --- | --- | --- | --- |
| first call, cold Inductor cache | 108.6 s | 103.1 s | 103.4 s |
| B=1 short (2.7 s audio) | 283 / 357 | 284 / 386 | 298 / 364 |
| B=1 medium (8.9 s audio) | 393 / 396 | 391 / 545 | 296 / 374 |
| batch 6, mixed lengths (33 s audio) | 1784 / 3041 | 1755 / 1795 | 1586 / 1670 |
| B=1 long (28 s audio) | 627 / 631 | 654 / 671 | 614 / 648 |
| peak allocated / reserved MB | 3353 / 3868 | 3353 / 3868 | 3326 / 3932 |
| WER, 30 utterances / 470 words | 0.85 % | identical PCM | 1.28 % |
| ReDimNet cosine to Sonja, mean | 0.869 | identical PCM | 0.869 |

The WER and similarity differences for flex are within seed-to-seed noise: on
0.1.5, seed 1234 gave 0.43 % WER and seed 777 gave 1.28 %, with mean
similarity 0.869 vs 0.870. The 0.1.5 run used 3 warm repeats, the others 5.
Without the shared-memory kernel options, flex fell back to eager on this GPU
and took 3.9-6.4 s for the same cases. SDPA stays the default; flex is an opt-in
with no clear gain on the 4090.

## FlashInfer on Linux

The Docker image installs it by default (`--build-arg ENABLE_FLASHINFER=false`
leaves it out) and sets `OMNIVOICE_TTS_ENABLE_FLASHINFER` to the same value.
FlashInfer 0.6.15.post1 is the version upstream validated. For a manual Linux
venv with the torch 2.10 cu130 stack:

```bash
pip install "flashinfer-python[cu13]==0.6.15.post1" "flashinfer-jit-cache==0.6.15.post1+cu130" \
    --extra-index-url https://flashinfer.ai/whl/cu130/
export OMNIVOICE_TTS_ENABLE_FLASHINFER=true
```

This path has not been run on this Windows box, because FlashInfer kernels are
Linux-only. Before you rely on it, compare SDPA bf16 with FlashInfer fp16 on the
target GPU: WER and speaker similarity against the same seed. A value saved in
`data/runtime_settings.json` (`enable_flashinfer`) overrides the environment
variable.

Important fp8 note:

- `kernels 0.14.1` works with `transformers 5.9.0` and exposes `FineGrainedFP8Config`.
- `kernels 0.15.1` is not compatible with this Transformers build. It raises
  `ValueError: Either a revision or a version must be specified` while importing
  `omnivoice`, because `transformers.integrations.hub_kernels` constructs
  `LayerRepository` without a revision/version.
- Keep `kernels>=0.14.1,<0.15` and `tomlkit>=0.13.3,<0.15` pinned.
- On Windows, `transformers` tries `kernels-community/deep-gemm` before the
  Triton `kernels-community/finegrained-fp8` fallback. DeepGEMM currently has no
  Windows build for this stack, so the server disables only that Hub lookup at
  runtime to avoid repeated Hugging Face requests/rate limits.

Validation commands:

```powershell
& .\.conda-env\python.exe -m pip check
& .\.conda-env\python.exe -c "import numpy, soundfile, torch, omnivoice, kernels, transformers, gradio, tomlkit; from transformers import FineGrainedFP8Config; print(torch.__version__, kernels.__version__, transformers.__version__, FineGrainedFP8Config().__class__.__name__)"
& .\.conda-env\python.exe -m pytest backend\tests -q -p no:cacheprovider --basetemp .tmp\pytest
```
