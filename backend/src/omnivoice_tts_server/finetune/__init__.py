"""Finetune capability: self-generate WER-filtered training data and (phase 2) train
a custom OmniVoice checkpoint, all driven from the admin panel.

Phase 1 (this milestone): domain list management, LLM sentence generation, batched
TTS+ASR+WER generation loop that lands accepted clips under data/finetune/train/<voice>/,
and a human-eval clip browser (play + delete).

Phase 2: audio->codec-token cache (extract_audio_tokens), the omnivoice trainer, and
loading the resulting custom checkpoint via the existing model-ops surface.
"""

from .service import FinetuneService
from .trainer import FinetuneTrainer

__all__ = ['FinetuneService', 'FinetuneTrainer']
