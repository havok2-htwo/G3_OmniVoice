from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field


class JobStatus(str, Enum):
    queued = 'queued'
    warming = 'warming'
    running = 'running'
    streaming = 'streaming'
    cancelling = 'cancelling'
    completed = 'completed'
    failed = 'failed'
    cancelled = 'cancelled'


class TaskType(str, Enum):
    custom_voice = 'CustomVoice'
    voice_design = 'VoiceDesign'
    base = 'Base'


class SpeechRequest(BaseModel):
    input: str | None = None
    model: str | None = None
    voice: str | None = None
    task_type: TaskType | None = None
    language: str | None = None
    instructions: str | None = None
    response_format: str = 'wav'
    speed: float = 1.0
    stream: bool = False
    ref_audio: str | None = None
    ref_text: str | None = None
    x_vector_only_mode: bool = False
    seed: int | None = Field(default=None, ge=0, le=2_147_483_647)
    metadata: dict[str, Any] = Field(default_factory=dict)


class JobMetrics(BaseModel):
    queue_wait_ms: int | None = None
    model_warm_ms: int | None = None
    ttfa_ms: int | None = None
    job_wall_ms: int | None = None
    audio_duration_ms: int | None = None
    realtime_x: float | None = None
    output_bytes: int | None = None
    batch_count: int | None = None
    max_batch_size_seen: int | None = None
    last_batch_size: int | None = None
    sentences_total: int | None = None
    sentences_rendered: int | None = None


class SpeechJobCreateResponse(BaseModel):
    job_id: str
    status: JobStatus
    queue_position: int
    eta_ms: int


class SynthesisResultResponse(BaseModel):
    job_id: str
    status: JobStatus
    model: str | None = None
    task_type: TaskType | None = None
    voice: str | None = None
    sample_rate: int
    metrics: JobMetrics = Field(default_factory=JobMetrics)
    audio_url: str | None = None
    error_message: str | None = None


class JobListItem(BaseModel):
    job_id: str
    status: JobStatus
    model: str | None = None
    task_type: TaskType | None = None
    voice: str | None = None
    input_preview: str
    queue_position: int
    eta_ms: int
    created_at: datetime
    updated_at: datetime
    metrics: JobMetrics = Field(default_factory=JobMetrics)
    error_message: str | None = None


class JobDetailResponse(JobListItem):
    started_at: datetime | None = None
    first_audio_at: datetime | None = None
    completed_at: datetime | None = None
    sentences_total: int | None = None
    batch_count: int | None = None


class VoiceProfileCreateResponse(BaseModel):
    voice_id: str
    name: str
    source: str
    created_at: datetime


class VoiceProfileListItem(BaseModel):
    voice_id: str
    name: str
    source: str
    created_at: datetime | None = None
    ref_text: str | None = None
    filename: str | None = None
    has_audio: bool = False


class VoiceCatalogResponse(BaseModel):
    voices: list[VoiceProfileListItem] = Field(default_factory=list)


class ModelInfo(BaseModel):
    model_id: str
    loaded: bool
    active: bool
    task_types: list[TaskType]


class StatRolling(BaseModel):
    ttfa_ms_avg: float | None = None
    queue_wait_ms_avg: float | None = None
    job_wall_ms_avg: float | None = None
    realtime_x_avg: float | None = None


class StatGlobal(BaseModel):
    jobs_total: int
    audio_seconds_total: float
    realtime_x_avg: float | None = None


class StatsResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    active_model: str
    queue_depth: int
    worker_state: str
    rolling: StatRolling
    global_: StatGlobal = Field(alias='global')


class GpuStatsResponse(BaseModel):
    name: str
    memory_used_mb: int
    memory_total_mb: int
    utilization_percent: int
    temperature_c: int | None = None


class RuntimeDeviceInfo(BaseModel):
    id: str
    label: str
    kind: str
    name: str
    index: int | None = None
    memory_total_mb: int | None = None
    available: bool = True
    selected: bool = False


class RuntimeDeviceListResponse(BaseModel):
    preferred_device: str
    devices: list[RuntimeDeviceInfo] = Field(default_factory=list)


class TranscriptionResponse(BaseModel):
    transcription: str
    voice_vector: list[float] | None = None


class ServerSettingsResponse(BaseModel):
    model_directory: str
    default_model: str
    default_voice: str
    whisper_base_url: str | None = None
    whisper_path: str
    vllm_base_url: str
    vllm_model: str = ''
    wer_concurrency: int
    wer_transcription_concurrency: int
    retention_days: int
    queue_limit: int
    runtime_backend: str
    allow_model_downloads: bool
    preferred_device: str
    attention_implementation: str
    torch_dtype: str
    compile_model: bool
    compile_cudagraphs: bool
    cudagraph_skip_dynamic_graphs: bool
    cuda_memory_trim_after_batch: bool
    warmup_on_startup: bool
    sample_rate: int
    poll_interval_ms: int
    theme: str
    built_in_voices: list[str] = Field(default_factory=list)
    sentence_chunking: bool
    short_sentence_merge_max_chars: int
    following_sentence_merge_min_chars: int
    max_parallel_requests: int
    max_batch_size: int
    batch_wait_ms: int
    vram_budget_mb: int = 0
    max_input_chars: int = 0
    # Derived from vram_budget_mb (read-only, ignored on update):
    max_batch_audio_seconds: float = 0.0
    max_chars_per_chunk: int = 0
    estimated_peak_vram_mb: int = 0
    stream_chunk_ms: int
    stream_prebuffer_ms: int
    num_step: int | None = None
    guidance_scale: float | None = None
    duration: float | None = None
    t_shift: float | None = None
    denoise: bool | None = None
    preprocess_prompt: bool | None = None
    postprocess_output: bool | None = None
    audio_chunk_duration: float | None = None
    audio_chunk_threshold: float | None = None
    position_temperature: float | None = None
    class_temperature: float | None = None


class ServerSettingsUpdateRequest(BaseModel):
    model_directory: str | None = None
    default_model: str | None = None
    default_voice: str | None = None
    whisper_base_url: str | None = None
    whisper_path: str | None = None
    vllm_base_url: str | None = None
    vllm_model: str | None = None
    wer_concurrency: int | None = Field(default=None, ge=1, le=64)
    wer_transcription_concurrency: int | None = Field(default=None, ge=1, le=64)
    retention_days: int | None = Field(default=None, ge=1, le=365)
    queue_limit: int | None = Field(default=None, ge=1, le=512)
    allow_model_downloads: bool | None = None
    preferred_device: str | None = None
    attention_implementation: str | None = None
    torch_dtype: str | None = None
    compile_model: bool | None = None
    compile_cudagraphs: bool | None = None
    cudagraph_skip_dynamic_graphs: bool | None = None
    cuda_memory_trim_after_batch: bool | None = None
    warmup_on_startup: bool | None = None
    poll_interval_ms: int | None = Field(default=None, ge=250, le=5000)
    theme: str | None = None
    sentence_chunking: bool | None = None
    short_sentence_merge_max_chars: int | None = Field(default=None, ge=0, le=512)
    following_sentence_merge_min_chars: int | None = Field(default=None, ge=0, le=1024)
    max_parallel_requests: int | None = Field(default=None, ge=1, le=128)
    max_batch_size: int | None = Field(default=None, ge=1, le=128)
    batch_wait_ms: int | None = Field(default=None, ge=0, le=1000)
    vram_budget_mb: int | None = Field(default=None, ge=0, le=80000)
    max_input_chars: int | None = Field(default=None, ge=0, le=1_000_000)
    stream_chunk_ms: int | None = Field(default=None, ge=20, le=1000)
    stream_prebuffer_ms: int | None = Field(default=None, ge=0, le=5000)
    num_step: int | None = Field(default=None, ge=1, le=256)
    guidance_scale: float | None = Field(default=None, ge=0.0, le=20.0)
    duration: float | None = Field(default=None, ge=0.1, le=120.0)
    t_shift: float | None = Field(default=None, ge=0.0, le=10.0)
    denoise: bool | None = None
    preprocess_prompt: bool | None = None
    postprocess_output: bool | None = None
    # Raised caps: set the threshold above your longest text to effectively disable
    # internal audio chunking (needed for audiobook prosody). Real ceiling is VRAM.
    audio_chunk_duration: float | None = Field(default=None, ge=0.1, le=3600.0)
    audio_chunk_threshold: float | None = Field(default=None, ge=0.1, le=3600.0)
    position_temperature: float | None = Field(default=None, ge=0.0, le=5.0)
    class_temperature: float | None = Field(default=None, ge=0.0, le=5.0)


class SettingsPresetItem(BaseModel):
    name: str
    values: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None


class SettingsPresetListResponse(BaseModel):
    presets: list[SettingsPresetItem] = Field(default_factory=list)


class SettingsPresetSaveRequest(BaseModel):
    values: dict[str, Any] = Field(default_factory=dict)


class ModelOperationRequest(BaseModel):
    model: str | None = None
    task_type: TaskType | None = None
    voice: str | None = None
    instructions: str | None = None
    language: str | None = None


class ModelOperationResponse(BaseModel):
    ok: bool
    model: str
    warm_ms: int
    message: str


class ModelDownloadRequest(BaseModel):
    model_id: str | None = None
    model: str | None = None
    storage_path: str | None = None


class ModelDownloadStatus(BaseModel):
    id: str
    label: str
    kind: str
    status: str
    local_path: str | None = None
    cache_path: str | None = None
    error: str | None = None
    updated_at: str | None = None
    storage_root: str
    approx_size_gb: float | None = None
    size_on_disk_gb: float | None = None


class ModelDownloadListResponse(BaseModel):
    models: list[ModelDownloadStatus] = Field(default_factory=list)


class ModelDownloadActionResponse(ModelDownloadListResponse):
    ok: bool = True
    job: dict[str, Any] | None = None
    removed: bool | None = None
    removed_path: str | None = None


class VllmModelsResponse(BaseModel):
    ok: bool = True
    base_url: str
    models: list[str] = Field(default_factory=list)
    selected_model: str | None = None
    error: str | None = None


class MemoryCleanupResponse(BaseModel):
    ok: bool
    message: str
    memory_before_mb: int | None = None
    memory_after_mb: int | None = None
    memory_total_mb: int | None = None
    released_mb: int | None = None


class BenchmarkCaseRequest(BaseModel):
    label: str
    request: SpeechRequest = Field(default_factory=SpeechRequest)


class BenchmarkRunCreateRequest(BaseModel):
    name: str = 'OmniVoice benchmark'
    text: str = 'Hallo! Das ist ein kurzer OmniVoice Benchmark.'
    mode: str = Field(default='traffic', pattern='^(iterations|traffic)$')
    iterations: int = Field(default=3, ge=1, le=50)
    warmup_iterations: int = Field(default=1, ge=0, le=20)
    parallel_requests: int = Field(default=1, ge=1, le=64)
    duration_seconds: int = Field(default=60, ge=1, le=3600)
    requests_per_minute: float = Field(default=30.0, ge=0.1, le=6000.0)
    min_sentences_per_request: int = Field(default=1, ge=1, le=100)
    max_sentences_per_request: int = Field(default=5, ge=1, le=100)
    completion_timeout_seconds: int = Field(default=180, ge=1, le=3600)
    random_seed: int | None = Field(default=None, ge=0, le=2_147_483_647)
    exclusive: bool = True
    cases: list[BenchmarkCaseRequest] = Field(default_factory=list)


class BenchmarkIterationResult(BaseModel):
    iteration: int
    warmup: bool
    label: str
    success: bool
    metrics: JobMetrics = Field(default_factory=JobMetrics)
    request_id: str | None = None
    scheduled_at_ms: int | None = None
    submitted_at_ms: int | None = None
    completed_at_ms: int | None = None
    sentence_count: int | None = None
    text_preview: str | None = None
    error_message: str | None = None


class BenchmarkCaseSummary(BaseModel):
    label: str
    iterations: int
    success_count: int = 0
    failure_count: int = 0
    ttfa_ms_avg: float | None = None
    ttfa_ms_min: float | None = None
    ttfa_ms_p50: float | None = None
    ttfa_ms_p95: float | None = None
    ttfa_ms_p99: float | None = None
    ttfa_ms_max: float | None = None
    queue_wait_ms_avg: float | None = None
    queue_wait_ms_p95: float | None = None
    queue_wait_ms_p99: float | None = None
    job_wall_ms_avg: float | None = None
    job_wall_ms_min: float | None = None
    job_wall_ms_p50: float | None = None
    job_wall_ms_p95: float | None = None
    job_wall_ms_p99: float | None = None
    job_wall_ms_max: float | None = None
    realtime_x_avg: float | None = None
    audio_duration_ms_avg: float | None = None


class BenchmarkRunResponse(BaseModel):
    run_id: str
    name: str
    status: str
    created_at: datetime
    completed_at: datetime | None = None
    mode: str = 'traffic'
    iterations: int
    warmup_iterations: int
    parallel_requests: int
    duration_seconds: int = 60
    requests_per_minute: float = 30.0
    completion_timeout_seconds: int = 180
    total_requests: int = 0
    exclusive: bool
    cases: list[BenchmarkCaseSummary] = Field(default_factory=list)
    results: list[BenchmarkIterationResult] = Field(default_factory=list)
    error_message: str | None = None


class WerBenchmarkCreateRequest(BaseModel):
    name: str = 'G3_OmniVoice WER benchmark'
    count: int = Field(default=100, ge=1, le=1000)
    concurrency: int = Field(default=4, ge=1, le=64)
    transcription_concurrency: int = Field(default=16, ge=1, le=64)
    vllm_base_url: str = 'http://192.168.20.126:8000'
    vllm_model: str | None = None
    whisper_base_url: str = 'http://192.168.0.200:7861'
    whisper_path: str | None = None
    language: str = 'Deutsch'
    prompt: str | None = None
    exclude_sentences: list[str] = Field(default_factory=list, max_length=10_000)
    min_words: int = Field(default=5, ge=1, le=80)
    max_words: int = Field(default=16, ge=1, le=120)
    tolerance_letters_per_word: int = Field(default=2, ge=0, le=8)
    completion_timeout_seconds: int = Field(default=180, ge=1, le=3600)
    random_seed: int | None = Field(default=None, ge=0, le=2_147_483_647)
    seed_range: int = Field(default=0, ge=0, le=1024)
    seed_values: list[Annotated[int, Field(ge=0, le=2_147_483_647)]] | None = Field(default=None, max_length=1025)
    exclusive: bool = True
    request: SpeechRequest = Field(default_factory=SpeechRequest)


class WerBenchmarkItemResult(BaseModel):
    index: int
    seed: int | None = None
    success: bool
    source_text: str
    transcript: str | None = None
    normalized_source: str | None = None
    normalized_transcript: str | None = None
    wer: float | None = None
    word_count: int | None = None
    word_errors: int | None = None
    substitutions: int | None = None
    insertions: int | None = None
    deletions: int | None = None
    job_id: str | None = None
    metrics: JobMetrics = Field(default_factory=JobMetrics)
    synthesis_ms: int | None = None
    transcription_ms: int | None = None
    total_ms: int | None = None
    error_message: str | None = None


class WerBenchmarkSummary(BaseModel):
    total: int
    completed: int
    success_count: int
    failure_count: int
    wer_avg: float | None = None
    wer_p50: float | None = None
    wer_p95: float | None = None
    wer_max: float | None = None
    exact_count: int = 0
    exact_rate: float | None = None


class WerBenchmarkSeedSummary(WerBenchmarkSummary):
    seed: int | None = None


class WerBenchmarkRunResponse(BaseModel):
    run_id: str
    name: str
    status: str
    created_at: datetime
    completed_at: datetime | None = None
    count: int
    concurrency: int
    transcription_concurrency: int
    seed_start: int | None = None
    seed_range: int = 0
    seed_values: list[int | None] = Field(default_factory=list)
    vllm_base_url: str
    vllm_model: str | None = None
    whisper_base_url: str
    language: str
    min_words: int
    max_words: int
    tolerance_letters_per_word: int
    completion_timeout_seconds: int
    sentence_cache_hit: bool = False
    sentence_cache_key: str | None = None
    exclusive: bool
    summary: WerBenchmarkSummary
    seed_leaderboard: list[WerBenchmarkSeedSummary] = Field(default_factory=list)
    results: list[WerBenchmarkItemResult] = Field(default_factory=list)
    error_message: str | None = None


class DashboardOverview(BaseModel):
    active_model: str
    queue_depth: int
    active_requests: int
    worker_state: str
    ttfa_ms_avg: float | None = None
    queue_wait_ms_avg: float | None = None
    job_wall_ms_avg: float | None = None
    realtime_x_avg: float | None = None
    jobs_total: int
    audio_seconds_total: float
    gpu_name: str
    gpu_memory_used_mb: int
    gpu_memory_total_mb: int
    gpu_utilization_pct: int
    gpu_temperature_c: int | None = None


class BatchSnapshot(BaseModel):
    batch_id: str
    model_id: str
    task_type: TaskType
    voice: str | None = None
    language: str | None = None
    size: int
    started_at: datetime
    request_ids: list[str] = Field(default_factory=list)
    sentence_indices: list[int] = Field(default_factory=list)


class DashboardSnapshot(BaseModel):
    overview: DashboardOverview
    settings: ServerSettingsResponse
    models: list[ModelInfo] = Field(default_factory=list)
    voices: list[VoiceProfileListItem] = Field(default_factory=list)
    jobs: list[JobListItem] = Field(default_factory=list)
    current_batch: BatchSnapshot | None = None
    recent_batches: list[BatchSnapshot] = Field(default_factory=list)
