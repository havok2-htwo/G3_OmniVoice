from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

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
    'torch_dtype',
    'compile_model',
    'compile_cudagraphs',
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
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='OMNIVOICE_TTS_', case_sensitive=False)

    app_name: str = 'G3_OmniVoice Server'
    host: str = '0.0.0.0'
    port: int = 8091
    admin_api_key: str = Field(default='dev-admin-key')
    startup_admin_key: str = ''
    startup_admin_key_ttl_seconds: int = Field(default=300, ge=1, le=3600)
    whisper_base_url: str | None = 'http://192.168.0.200:7861'
    whisper_path: str = ''
    vllm_base_url: str = 'http://192.168.20.126:8000'
    vllm_model: str = ''
    wer_concurrency: int = Field(default=4, ge=1, le=64)
    wer_transcription_concurrency: int = Field(default=16, ge=1, le=64)
    active_model: str = 'k2-fsa/OmniVoice-AutoVoice'
    default_voice: str = 'Auto Voice'
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
    allow_model_downloads: bool = False
    enable_cpu_offload: bool = False
    compile_model: bool = False
    compile_cudagraphs: bool = False
    cudagraph_skip_dynamic_graphs: bool = True
    cuda_memory_trim_after_batch: bool = False
    warmup_on_startup: bool = True
    preferred_device: str = 'cuda:0'
    attention_implementation: str = 'sdpa'
    torch_dtype: str = 'float16'
    max_queue_size: int = 32
    sample_rate: int = 24_000
    channels: int = 1
    retention_days: int = 7
    max_retained_jobs: int = 500
    max_retained_benchmark_runs: int = 25
    benchmark_dataset_default: str = 'de_standard_v1'
    frontend_poll_interval_ms: int = 500
    frontend_theme: str = 'onyx'
    sentence_chunking: bool = True
    short_sentence_merge_max_chars: int = 30
    following_sentence_merge_min_chars: int = 20
    max_parallel_requests: int = 6
    max_batch_size: int = 8
    batch_wait_ms: int = 35
    vram_budget_mb: int = 24000        # 0 = disable VRAM budgeting (unbounded)
    max_input_chars: int = 100000      # hard reject single requests above this (abuse guard)
    stream_chunk_ms: int = 140
    stream_prebuffer_ms: int = 0
    num_step: int | None = Field(default=None, ge=1, le=256)
    guidance_scale: float | None = Field(default=None, ge=0.0, le=20.0)
    duration: float | None = Field(default=None, ge=0.1, le=120.0)
    t_shift: float | None = Field(default=None, ge=0.0, le=10.0)
    denoise: bool | None = None
    preprocess_prompt: bool | None = None
    postprocess_output: bool | None = None
    # Caps raised so internal OmniVoice audio chunking can be pushed high enough to be
    # effectively disabled (audiobooks need one continuous sequence for prosody). Set the
    # threshold above your longest single generation; the real ceiling is VRAM (~7-9 min).
    audio_chunk_duration: float | None = Field(default=None, ge=0.1, le=3600.0)
    audio_chunk_threshold: float | None = Field(default=None, ge=0.1, le=3600.0)
    position_temperature: float | None = Field(default=None, ge=0.0, le=5.0)
    class_temperature: float | None = Field(default=None, ge=0.0, le=5.0)

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


def save_runtime_settings(settings: Settings) -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    payload = {}
    for field_name in sorted(RUNTIME_SETTINGS_FIELDS):
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
