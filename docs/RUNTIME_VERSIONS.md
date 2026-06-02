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
| omnivoice | 0.1.5 |
| transformers | 5.9.0 |
| kernels | 0.14.1 |
| kernels-data | 0.15.1 |
| gradio | 6.15.2 |
| tomlkit | 0.14.0 |
| numpy | 2.4.4 |
| soundfile | 0.13.1 |

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
