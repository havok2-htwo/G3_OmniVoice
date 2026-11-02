"""The Audio8 models the server can run. One EngineSpec per entry; everything else about them is shared.

``runtime`` picks the implementation behind ``Audio8Synthesizer``:

- ``slots``: the slot engine in ``engine.py`` with the transformer language model of ``lm.py`` (0.6B checkpoints).
- ``falcon``: the same engine with the Falcon-H1 language model of ``falcon_lm.py``, attention and Mamba state per slot
  (0.1B checkpoint).
- ``onnx``: the CPU engine in ``onnx_engine.py``, onnxruntime INT8 language model with an FP16 codec, no torch.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Audio8Variant:
    engine_id: str
    label: str
    repo_id: str
    folder: str         # below models_root_dir
    revision: str       # pinned: remote code loads from this folder, so an unreviewed revision must not get there
    runtime: str        # slots | falcon | onnx
    license: str
    approx_size_gb: float
    # Commercial use is limited, the admin shows the licence as a warning.
    license_restricted: bool = False
    # Marks the weights files that must exist before the engine can load.
    weight_files: tuple[str, ...] = ('config.json', 'model.safetensors', 'codec.pth')

    @property
    def base_alias(self) -> str:
        """The clone alias the queue serves. It ends in 'Base' like MING_BASE_ALIAS: task-type inference, the default
        voice profile and the OpenAI voice routing then treat it like OmniVoice-Base."""
        return f'{self.repo_id}-Base'


AUDIO8_VARIANTS: dict[str, Audio8Variant] = {
    'audio8_realistic': Audio8Variant(
        engine_id='audio8_realistic',
        label='Audio8 Realistic 0.6B',
        repo_id='Edge0/Audio8-TTS-Realistic-0.6b',
        folder='Audio8-TTS-Realistic-0.6b',
        revision='925bbd4c2d8360af85f2594bf0205173fda853ba',
        runtime='slots',
        license='Apache-2.0',
        approx_size_gb=2.4,
    ),
    'audio8_preview': Audio8Variant(
        engine_id='audio8_preview',
        label='Audio8 Preview 0.6B',
        repo_id='Edge0/Audio8-TTS-Preview-0.6b',
        folder='Audio8-TTS-Preview-0.6b',
        revision='20e519d6e2c485ea1544f95f1fcaa6de6712bfc2',
        runtime='slots',
        license='Apache-2.0',
        approx_size_gb=2.4,
    ),
    'audio8_mini': Audio8Variant(
        engine_id='audio8_mini',
        label='Audio8 Preview 0.1B',
        repo_id='Edge0/Audio8-TTS-Preview-0.1b',
        folder='Audio8-TTS-Preview-0.1b',
        revision='8c837020ab800498834038363dea06757872afda',
        runtime='falcon',
        license='Audio8 Community License v1.0: kommerzielle Nutzung nur unter 2 Mio. US$ Jahresumsatz',
        approx_size_gb=1.6,
        license_restricted=True,
    ),
    'audio8_onnx': Audio8Variant(
        engine_id='audio8_onnx',
        label='Audio8 0.1B ONNX INT8',
        repo_id='Edge0/audio8-TTS-0.1B-ONNX-INT8',
        folder='Audio8-TTS-Preview-0.1B-ONNX-INT8',
        revision='317c12d4e0da83847b594fcf8bd74bf2c76615ec',
        runtime='onnx',
        license='Apache-2.0',
        approx_size_gb=0.8,
        weight_files=('config.json', 'runtime_manifest.json', 'slow_ar_int8.onnx.data', 'fast_ar_int8.onnx.data',
                      'codec_decoder_fp16.onnx.data', 'registration/codec_encoder_fp16.onnx.data'),
    ),
}


def audio8_variant(engine_id: str) -> Audio8Variant:
    try:
        return AUDIO8_VARIANTS[engine_id]
    except KeyError:
        raise ValueError(f'unknown Audio8 engine {engine_id!r}; known: {sorted(AUDIO8_VARIANTS)}') from None
