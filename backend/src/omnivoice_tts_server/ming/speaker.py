# Vendored from inclusionAI/Ming-3.0-tts @ 9332fb9 (spkemb_extractor.py), MIT license, see LICENSE in this directory.
# CAM++ speaker embedding (campplus.onnx) on the CPU through onnxruntime, 80-bin Kaldi fbank at 16 kHz.
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import torch


class SpeakerEmbedder:
    def __init__(self, onnx_path: str | Path, threads: int = 2) -> None:
        import onnxruntime

        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = threads
        self._session = onnxruntime.InferenceSession(str(onnx_path), sess_options=options,
                                                     providers=['CPUExecutionProvider'])
        self._input_name = self._session.get_inputs()[0].name
        self._lock = threading.Lock()

    def __call__(self, waveform_16k: torch.Tensor) -> torch.Tensor:
        """(1, samples) float32 waveform at 16 kHz -> (1, 192) float32 embedding."""
        import torchaudio.compliance.kaldi as kaldi

        feat = kaldi.fbank(waveform_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        with self._lock:
            out = self._session.run(None, {self._input_name: feat.unsqueeze(0).cpu().numpy()})[0]
        return torch.tensor([np.asarray(out, dtype=np.float32).flatten().tolist()])
