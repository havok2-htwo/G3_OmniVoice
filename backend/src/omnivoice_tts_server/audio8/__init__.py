"""Audio8 TTS engines (DualAR semantic + codebook transformers, 44.1 kHz codec), see ``engine.py``.

The torch-free parts (``variants``, ``text``, the CPU engine ``onnx_engine``) import cheaply; ``engine`` and
``codec_stream`` need torch.
"""
from .variants import AUDIO8_VARIANTS, Audio8Variant, audio8_variant

__all__ = ['AUDIO8_VARIANTS', 'Audio8Variant', 'audio8_variant']
