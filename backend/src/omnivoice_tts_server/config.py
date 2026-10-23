from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODELS_ROOT = PROJECT_ROOT / 'models'
DEFAULT_FRONTEND_DIST = PROJECT_ROOT / 'frontend' / 'dist'
DEFAULT_DATA_DIR = PROJECT_ROOT / 'data'

RUNTIME_SETTINGS_FILE = 'runtime_settings.json'
RUNTIME_SETTINGS_FIELDS = {
    'active_model',
    'default_voice',
    'models_root_dir',
    'whisper_base_url',
    'whisper_path',
    'vllm_base_url',
    'vllm_model',
    'wer_concurrency',
    'wer_transcription_concurrency',
    'retention_days',
    'max_queue_size',
    'allow_model_downloads',
    'preferred_device',
    'attention_implementation',
    'enable_flashinfer',
    'torch_dtype',
    'compile_model',
    'compile_cudagraphs',
    'sdpa_prefer_cudnn',
    'packed_cfg',
    'stream_first_preempt',
    'cudagraph_skip_dynamic_graphs',
    'cuda_memory_trim_after_batch',
    'warmup_on_startup',
    'frontend_poll_interval_ms',
    'frontend_theme',
    'sentence_chunking',
    'short_sentence_merge_max_chars',
    'following_sentence_merge_min_chars',
    'max_parallel_requests',
    'max_batch_size',
    'batch_wait_ms',
    'vram_budget_mb',
    'max_input_chars',
    'stream_chunk_ms',
    'stream_prebuffer_ms',
    'num_step',
    'guidance_scale',
    'duration',
    't_shift',
    'denoise',
    'preprocess_prompt',
    'postprocess_output',
    'audio_chunk_duration',
    'audio_chunk_threshold',
    'position_temperature',
    'class_temperature',
    'text_optimization_default',
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='OMNIVOICE_TTS_', case_sensitive=False)

    app_name: str = 'G3_OmniVoice Server'
    host: str = '0.0.0.0'
    port: int = 8091
    # The suite runs on one box, so G3_WHISPER is local by default; Docker sets OMNIVOICE_TTS_WHISPER_BASE_URL.
    whisper_base_url: str | None = 'http://127.0.0.1:7861'
    whisper_path: str = ''
    vllm_base_url: str = 'http://192.168.20.126:8000'
    vllm_model: str = 'Qwen/Qwen3.6-35B-A3B-FP8'
    wer_concurrency: int = Field(default=4, ge=1, le=64)
    wer_transcription_concurrency: int = Field(default=16, ge=1, le=64)
    # Audio codec tokenizer used by the finetune preprocess step (extract_audio_tokens).
    # Defaults to the same Higgs-Audio-V2 tokenizer the runtime uses for voice cloning;
    # point at a local snapshot dir to stay fully offline.
    finetune_audio_tokenizer: str = 'eustlb/higgs-audio-v2-tokenizer'
    active_model: str = 'k2-fsa/OmniVoice-Base'
    # Base clones a saved voice profile; this one must exist (matched by name or voice_id).
    default_voice: str = 'SonjaV2'
    supported_models: list[str] = Field(
        default_factory=lambda: [
            'k2-fsa/OmniVoice-AutoVoice',
            'k2-fsa/OmniVoice-VoiceDesign',
            'k2-fsa/OmniVoice-Base',
        ]
    )
    built_in_voices: list[str] = Field(
        default_factory=lambda: ['Auto Voice']
    )
    models_root_dir: Path = Field(default=DEFAULT_MODELS_ROOT)
    frontend_dist_dir: Path = Field(default=DEFAULT_FRONTEND_DIST)
    data_dir: Path = Field(default=DEFAULT_DATA_DIR)
    runtime_backend: str = 'omnivoice'
    # runtime_backend='phonon' serves Gradium Phonon (ptts package) from this checkpoint folder instead of OmniVoice.
    # Gated repos need the checkpoint downloaded beforehand; the server never reads HF_TOKEN itself.
    phonon_model_id: str = 'gradium/phonon-40m-multilingual'
    phonon_model_dir: Path = Field(default=DEFAULT_MODELS_ROOT / 'phonon-40m-multilingual')
    phonon_lang: str = 'de'
    phonon_quant: str = 'q8'
    # Engine instances, one sentence per engine at a time. Each holds its own weights and thread pool.
    phonon_workers: int = Field(default=2, ge=1, le=16)
    # Rust thread-pool size per engine (RAYON_NUM_THREADS). 0 = cpu_count // (2 * phonon_workers). Left at the
    # runtime default (one pool per core per engine), the engines oversubscribe the CPU and run ~3x slower.
    phonon_threads: int = Field(default=0, ge=0, le=64)
    allow_model_downloads: bool = True
    enable_cpu_offload: bool = False
    compile_model: bool = True
    compile_cudagraphs: bool = False
    # Put cuDNN first in PyTorch's SDPA backend order (default order picks the mem-efficient kernel). Not
    # bit-identical (listening test passed 2026-10-16); on the RTX 5090 ~15-22 % faster per generate. Applied at model load.
    sdpa_prefer_cudnn: bool = True
    # One row per item [conditional | unconditional] instead of 2B padded rows (packed_decode.py). Not
    # bit-identical (listening test passed 2026-10-16); ~20-33 % faster per generate on the RTX 5090. Applied at model load.
    packed_cfg: bool = True
    # A stream's first sentence that arrives while a batch renders runs solo at the batch's next diffusion step
    # boundary instead of after the batch. The running batch's audio is unchanged (random state restored).
    stream_first_preempt: bool = True
    cudagraph_skip_dynamic_graphs: bool = True
    cuda_memory_trim_after_batch: bool = True
    warmup_on_startup: bool = True
    preferred_device: str = 'cuda:0'
    # sdpa | flex_attention | eager. flex_attention gets its padding mask converted to a
    # BlockMask at load (runtime_v2.install_flex_block_mask_forward).
    attention_implementation: str = 'sdpa'
    # Upstream FlashInfer packed attention (omnivoice_flashinfer, Linux + CUDA + the
    # flashinfer wheels). When it is active the model loads in fp16 and torch.compile is
    # skipped; anywhere else the server logs why and keeps attention_implementation.
    enable_flashinfer: bool = False
    torch_dtype: str = 'bfloat16'
    max_queue_size: int = 32
    sample_rate: int = 24_000
    channels: int = 1
    retention_days: int = 7
    max_retained_jobs: int = 500
    # Byte cap on audio kept for finished jobs (download via /v1/jobs/{id}/audio). Above
    # it the least recently used jobs lose their audio (record stays). 0 = unlimited.
    max_retained_audio_mb: int = Field(default=256, ge=0)
    max_retained_benchmark_runs: int = 25
    benchmark_dataset_default: str = 'de_standard_v1'
    frontend_poll_interval_ms: int = 500
    frontend_theme: str = 'onyx'
    sentence_chunking: bool = True
    short_sentence_merge_max_chars: int = 30
    following_sentence_merge_min_chars: int = 20
    max_parallel_requests: int = 8
    max_batch_size: int = 8
    batch_wait_ms: int = 0
    vram_budget_mb: int = 7000         # 0 = disable VRAM budgeting (unbounded)
    max_input_chars: int = 25000       # hard reject single requests above this (abuse guard)
    stream_chunk_ms: int = 140
    stream_prebuffer_ms: int = 0
    num_step: int | None = Field(default=20, ge=1, le=256)
    guidance_scale: float | None = Field(default=2.0, ge=0.0, le=20.0)
    duration: float | None = Field(default=None, ge=0.1, le=120.0)
    t_shift: float | None = Field(default=0.1, ge=0.0, le=10.0)
    denoise: bool | None = True
    preprocess_prompt: bool | None = True
    postprocess_output: bool | None = True
    # Caps raised so internal OmniVoice audio chunking can be pushed high enough to be
    # effectively disabled (audiobooks need one continuous sequence for prosody). Set the
    # threshold above your longest single generation; the real ceiling is VRAM (~7-9 min).
    audio_chunk_duration: float | None = Field(default=30.1, ge=0.1, le=3600.0)
    audio_chunk_threshold: float | None = Field(default=30.0, ge=0.1, le=3600.0)
    position_temperature: float | None = Field(default=5.0, ge=0.0, le=5.0)
    class_temperature: float | None = Field(default=0.0, ge=0.0, le=5.0)
    # Spoken-text optimization for public requests that do not set text_optimization
    # themselves ('off' | 'deterministic'). Benchmarks and finetune jobs stay 'off'.
    text_optimization_default: Literal['off', 'deterministic'] = 'off'

    @field_validator('models_root_dir', mode='before')
    @classmethod
    def validate_models_root_dir(cls, value: str | Path) -> Path:
        if isinstance(value, Path):
            return value.expanduser()
        return Path(value).expanduser()

    @field_validator('frontend_dist_dir', mode='before')
    @classmethod
    def validate_frontend_dist_dir(cls, value: str | Path) -> Path:
        if isinstance(value, Path):
            return value.expanduser()
        return Path(value).expanduser()

    @field_validator('data_dir', mode='before')
    @classmethod
    def validate_data_dir(cls, value: str | Path) -> Path:
        if isinstance(value, Path):
            return value.expanduser()
        return Path(value).expanduser()


def load_runtime_settings(settings: Settings) -> None:
    path = settings.data_dir / RUNTIME_SETTINGS_FILE
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return
    if not isinstance(payload, dict):
        return
    for field_name in RUNTIME_SETTINGS_FIELDS:
        if field_name not in payload:
            continue
        value = payload[field_name]
        if field_name == 'models_root_dir':
            value = Path(value).expanduser()
        setattr(settings, field_name, value)


def save_runtime_settings(settings: Settings, fields: Iterable[str] | None = None) -> None:
    """Persist the runtime settings (or only `fields` of them; keys left out of the file
    keep coming from the environment / code defaults on the next start)."""
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    payload = {}
    names = RUNTIME_SETTINGS_FIELDS if fields is None else RUNTIME_SETTINGS_FIELDS & set(fields)
    for field_name in sorted(names):
        value = getattr(settings, field_name)
        payload[field_name] = str(value) if isinstance(value, Path) else value
    path = settings.data_dir / RUNTIME_SETTINGS_FILE
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    load_runtime_settings(settings)
    settings.models_root_dir.mkdir(parents=True, exist_ok=True)
    return settings
