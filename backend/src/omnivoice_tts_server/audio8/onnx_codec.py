"""Streaming decode of the ONNX codec on the CPU (no torch).

Upstream streams by re-running ``codec_decoder_fp16.onnx`` over ``stream_context_frames`` (128) + the new frames. Two
things are wrong with that here, both measured on a Ryzen 9800X3D:

- It is too slow: the whole decode costs about 12 ms per frame on 8 threads, so 140 frames for a chunk of 12 take 1.8 s
  for 0.56 s of audio. Nearly all of that is the waveform convolutions (10.5 ms per frame), the transformer is cheap
  (37 ms for those 140 frames).
- It is not exact: the codec's eight transformer layers have a 128-frame window each and reach about 1000 frames when
  stacked (see ``codec_stream.py``), so 128 frames of context end 13.8 dB from a full decode (SNR on real codes).

So the graph is cut once at load, in memory, into the two halves it is made of: ``front`` (codes to latents: quantizer,
transformer, upsampler) and ``back`` (latents to waveform). ``front`` gets 300 frames of context, ``back`` only the new
frames plus 4. That gives 53 dB from a full decode at 228 ms per chunk of 12 frames (8 threads). The codec is causal, so
upstream's ``stream_guard_frames`` is not needed either: the audio of a frame does not change with later frames.

The cut needs the ``onnx`` package, nothing else does. Weights stay in the model folder: the halves reference the
``.onnx.data`` file and onnxruntime reads it from there.

The graphs are by the Audio8 team (Apache-2.0), https://github.com/Audio8-AI/Audio8_TTS, onnx_runtime_0_1b_int8.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# Output of the last upsampler layer: the only tensor the waveform decoder reads from the rest of the graph, besides
# weights. Part of the pinned model revision; a graph without it is refused.
CUT_TENSOR = '/upsample/upsample.1/upsample.1.1/Add_output_0'
LATENTS_PER_FRAME = 4
TRANSFORMER_CONTEXT_FRAMES = 300
CONV_CONTEXT_FRAMES = 4


def session_options(threads: int, *, external_data_dir: Path | None = None) -> Any:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 3
    options.intra_op_num_threads = max(1, int(threads))
    options.inter_op_num_threads = 1
    # Several sessions share the cores; a pool that spins between ops takes them from the others.
    options.add_session_config_entry('session.intra_op.allow_spinning', '0')
    if external_data_dir is not None:
        options.add_session_config_entry('session.model_external_initializers_file_folder_path', str(external_data_dir))
    return options


def cpu_session(model: Path | bytes, threads: int, *, external_data_dir: Path | None = None) -> Any:
    import onnxruntime as ort

    source = model if isinstance(model, bytes) else str(model)
    return ort.InferenceSession(source, sess_options=session_options(threads, external_data_dir=external_data_dir),
                                providers=['CPUExecutionProvider'])


def _subgraph(model: Any, name: str, inputs: list[Any], outputs: list[Any]) -> bytes:
    """The nodes and weights the ``outputs`` need, starting from ``inputs``."""
    from onnx import helper

    graph = model.graph
    producer = {output: node for node in graph.node for output in node.output}
    stop = {value.name for value in inputs}
    needed: set[int] = set()
    pending = [value.name for value in outputs]
    seen: set[str] = set()
    while pending:
        tensor = pending.pop()
        if tensor in seen or tensor in stop:
            continue
        seen.add(tensor)
        node = producer.get(tensor)
        if node is not None and id(node) not in needed:
            needed.add(id(node))
            pending.extend(item for item in node.input if item)
    nodes = [node for node in graph.node if id(node) in needed]  # keeps the topological order
    used = {item for node in nodes for item in node.input}
    weights = [weight for weight in graph.initializer if weight.name in used]
    sub = helper.make_graph(nodes, name, inputs=inputs, outputs=outputs, initializer=weights)
    return helper.make_model(sub, opset_imports=model.opset_import, ir_version=model.ir_version).SerializeToString()


def split_decoder(path: Path) -> tuple[bytes, bytes]:
    """(front, back) serialized; their weights stay external, next to ``path``."""
    import onnx
    from onnx import TensorProto, helper

    model = onnx.load(str(path), load_external_data=False)
    graph = model.graph
    if not any(CUT_TENSOR in node.output for node in graph.node):
        raise RuntimeError(f'{path.name}: tensor {CUT_TENSOR} not found; this is not the pinned Audio8 codec graph')
    latent = helper.make_tensor_value_info(CUT_TENSOR, TensorProto.FLOAT16, ['batch', 'dim', 'latents'])
    return (_subgraph(model, 'audio8_codec_front', [graph.input[0]], [latent]),
            _subgraph(model, 'audio8_codec_back', [latent], [graph.output[0]]))


class StreamDecoder:
    """One decoder instance with its own thread pool. ``decode`` is called by one thread at a time."""

    def __init__(self, model_path: Path, threads: int, hop: int) -> None:
        front, back = split_decoder(model_path)
        self.front = cpu_session(front, threads, external_data_dir=model_path.parent)
        self.back = cpu_session(back, threads, external_data_dir=model_path.parent)
        self.hop = int(hop)
        self._codes = self.front.get_inputs()[0].name
        self._latents = self.back.get_inputs()[0].name

    def decode(self, frames: np.ndarray, start: int, end: int) -> np.ndarray:
        """Audio of ``frames[start:end]``, given all frames of the fragment so far: (end - start) * hop samples."""
        count = int(end) - int(start)
        window = frames[max(0, int(end) - TRANSFORMER_CONTEXT_FRAMES):int(end)]
        codes = np.ascontiguousarray(window.T[np.newaxis])
        latents = self.front.run(None, {self._codes: codes})[0]
        keep = (count + CONV_CONTEXT_FRAMES) * LATENTS_PER_FRAME
        audio = self.back.run(None, {self._latents: latents[..., -keep:]})[0]
        return np.asarray(audio, dtype=np.float32).reshape(-1)[-count * self.hop:]
