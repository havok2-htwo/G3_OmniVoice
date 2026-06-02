from __future__ import annotations

import asyncio
import io
import json
import os
import shutil
import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, Response, UploadFile, status
from fastapi.responses import StreamingResponse

from ..config import save_runtime_settings
from ..domain.models import (
    AdminKeyMetadata,
    AdminKeyResponse,
    AdminKeyRotateResponse,
    BatchSnapshot,
    BenchmarkRunCreateRequest,
    BenchmarkRunResponse,
    DashboardOverview,
    DashboardSnapshot,
    GpuStatsResponse,
    JobDetailResponse,
    JobListItem,
    JobStatus,
    MemoryCleanupResponse,
    ModelDownloadActionResponse,
    ModelDownloadListResponse,
    ModelDownloadRequest,
    ModelDownloadStatus,
    ModelOperationRequest,
    ModelOperationResponse,
    ModelInfo,
    ServerSettingsResponse,
    ServerSettingsUpdateRequest,
    SettingsPresetItem,
    SettingsPresetListResponse,
    SettingsPresetSaveRequest,
    SpeechJobCreateResponse,
    SpeechRequest,
    SynthesisResultResponse,
    TaskType,
    TranscriptionResponse,
    VoiceCatalogResponse,
    VoiceProfileCreateResponse,
    VoiceProfileListItem,
    VllmModelsResponse,
    WerBenchmarkCreateRequest,
    WerBenchmarkRunResponse,
)
from ..domain.state import VoiceProfileRecord, new_id, utcnow
from ..runtime_v2 import DEFAULT_VOICE_DESIGN_INSTRUCT, OMNIVOICE_MODEL_ID
from ..security import get_admin_record, require_admin_key, rotate_admin_key
from ..capacity import capacity_summary as _capacity_summary
from ..presets import delete_preset as _delete_preset
from ..presets import list_presets as _list_presets
from ..presets import save_preset as _save_preset
from ..services_v2 import (
    BenchmarkService,
    EventHub,
    QueueSaturatedError,
    QueueService,
    RequestTooLargeError,
    TranscriptionService,
    WerBenchmarkService,
)

router = APIRouter()
health = APIRouter()
admin = APIRouter(prefix='/api/admin', dependencies=[Depends(require_admin_key)])

MODEL_DOWNLOAD_META = {
    'k2-fsa/OmniVoice-AutoVoice': {
        'label': 'OmniVoice AutoVoice',
        'kind': 'autovoice',
        'approx_size_gb': None,
    },
    'k2-fsa/OmniVoice-VoiceDesign': {
        'label': 'OmniVoice VoiceDesign',
        'kind': 'voice-design',
        'approx_size_gb': None,
    },
    'k2-fsa/OmniVoice-Base': {
        'label': 'OmniVoice Base',
        'kind': 'voice-clone',
        'approx_size_gb': None,
    },
}


def _supported_task_types(model_id: str) -> list[TaskType]:
    if model_id.endswith('VoiceDesign'):
        return [TaskType.voice_design]
    if model_id.endswith('Base'):
        return [TaskType.base]
    return [TaskType.custom_voice]


def _effective_task_type(model_id: str | None, request_task_type: TaskType | None) -> TaskType:
    if request_task_type is not None:
        return request_task_type
    target_model = model_id or ''
    if target_model.endswith('VoiceDesign'):
        return TaskType.voice_design
    if target_model.endswith('Base'):
        return TaskType.base
    return TaskType.custom_voice


def _settings_response(settings: Any) -> ServerSettingsResponse:
    return ServerSettingsResponse(
        model_directory=str(settings.models_root_dir),
        default_model=settings.active_model,
        default_voice=settings.default_voice,
        whisper_base_url=settings.whisper_base_url,
        whisper_path='',
        vllm_base_url=settings.vllm_base_url,
        vllm_model=settings.vllm_model,
        wer_concurrency=settings.wer_concurrency,
        wer_transcription_concurrency=settings.wer_transcription_concurrency,
        retention_days=settings.retention_days,
        queue_limit=settings.max_queue_size,
        runtime_backend=settings.runtime_backend,
        allow_model_downloads=settings.allow_model_downloads,
        preferred_device=settings.preferred_device,
        attention_implementation=settings.attention_implementation,
        torch_dtype=settings.torch_dtype,
        compile_model=settings.compile_model,
        compile_cudagraphs=settings.compile_cudagraphs,
        cudagraph_skip_dynamic_graphs=settings.cudagraph_skip_dynamic_graphs,
        cuda_memory_trim_after_batch=settings.cuda_memory_trim_after_batch,
        warmup_on_startup=settings.warmup_on_startup,
        sample_rate=settings.sample_rate,
        poll_interval_ms=settings.frontend_poll_interval_ms,
        theme=settings.frontend_theme,
        built_in_voices=settings.built_in_voices,
        sentence_chunking=settings.sentence_chunking,
        short_sentence_merge_max_chars=settings.short_sentence_merge_max_chars,
        following_sentence_merge_min_chars=settings.following_sentence_merge_min_chars,
        max_parallel_requests=settings.max_parallel_requests,
        max_batch_size=settings.max_batch_size,
        batch_wait_ms=settings.batch_wait_ms,
        vram_budget_mb=getattr(settings, 'vram_budget_mb', 0),
        max_input_chars=getattr(settings, 'max_input_chars', 0),
        max_batch_audio_seconds=_capacity_summary(getattr(settings, 'vram_budget_mb', 0))['max_batch_audio_seconds'],
        max_chars_per_chunk=_capacity_summary(getattr(settings, 'vram_budget_mb', 0))['max_chars_per_chunk'],
        estimated_peak_vram_mb=_capacity_summary(getattr(settings, 'vram_budget_mb', 0))['estimated_peak_vram_mb'],
        stream_chunk_ms=settings.stream_chunk_ms,
        stream_prebuffer_ms=settings.stream_prebuffer_ms,
        num_step=settings.num_step,
        guidance_scale=settings.guidance_scale,
        duration=settings.duration,
        t_shift=settings.t_shift,
        denoise=settings.denoise,
        preprocess_prompt=settings.preprocess_prompt,
        postprocess_output=settings.postprocess_output,
        audio_chunk_duration=settings.audio_chunk_duration,
        audio_chunk_threshold=settings.audio_chunk_threshold,
        position_temperature=settings.position_temperature,
        class_temperature=settings.class_temperature,
    )


def _voice_items(request: Request, *, include_details: bool = False) -> list[VoiceProfileListItem]:
    store = request.app.state.store
    settings = request.app.state.settings
    built_in = [
        VoiceProfileListItem(voice_id=name.lower(), name=name, source='built-in', created_at=None)
        for name in settings.built_in_voices
    ]
    custom = [
        VoiceProfileListItem(
            voice_id=voice.voice_id,
            name=voice.name,
            source=voice.source,
            created_at=voice.created_at,
            ref_text=voice.ref_text if include_details else None,
            filename=voice.filename if include_details else None,
            has_audio=bool(voice.audio_bytes) if include_details else False,
        )
        for voice in sorted(store.voice_profiles.values(), key=lambda item: item.created_at, reverse=True)
    ]
    return built_in + custom


def _model_items(request: Request) -> list[ModelInfo]:
    settings = request.app.state.settings
    store = request.app.state.store
    active = store.active_model or settings.active_model
    return [
        ModelInfo(
            model_id=model,
            loaded=model in store.models_loaded or model == active,
            active=model == active,
            task_types=_supported_task_types(model),
        )
        for model in settings.supported_models
    ]


def _model_download_storage_root(settings: Any, storage_path: str | None = None) -> Path:
    raw = (storage_path or '').strip()
    if not raw:
        return Path(settings.models_root_dir).expanduser()
    return Path(raw).expanduser()


def _model_download_key(storage_root: Path) -> tuple[str, str]:
    return (OMNIVOICE_MODEL_ID, os.path.normcase(str(storage_root.resolve(strict=False))))


def _normalize_http_base_url(value: str) -> str:
    normalized = str(value or '').strip().rstrip('/')
    if normalized and '://' not in normalized:
        normalized = f'http://{normalized}'
    parsed = urlsplit(normalized)
    if parsed.scheme and parsed.netloc:
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip('/'), '', '')).rstrip('/')
    return normalized


def _normalize_vllm_base_url(value: str) -> str:
    normalized = _normalize_http_base_url(value)
    for suffix in ('/v1/models', '/v1/chat/completions', '/v1/completions', '/v1'):
        if normalized.lower().endswith(suffix):
            return normalized[: -len(suffix)].rstrip('/')
    return normalized


def _repo_cache_root(storage_root: Path) -> Path:
    return storage_root / f"models--{OMNIVOICE_MODEL_ID.replace('/', '--')}"


def _local_omnivoice_dir(storage_root: Path) -> Path:
    return storage_root / 'OmniVoice'


def _latest_hf_snapshot(storage_root: Path) -> Path | None:
    snapshots_root = _repo_cache_root(storage_root) / 'snapshots'
    if not snapshots_root.exists():
        return None
    snapshots = [item for item in snapshots_root.iterdir() if item.is_dir()]
    if not snapshots:
        return None
    return max(snapshots, key=lambda item: item.stat().st_mtime)


def _directory_size_gb(path: Path | None) -> float | None:
    if path is None or not path.exists():
        return None
    total = 0
    try:
        for item in path.rglob('*'):
            if item.is_file():
                try:
                    total += item.stat().st_size
                except OSError:
                    pass
    except OSError:
        return None
    return round(total / (1024 ** 3), 3)


def _model_download_statuses(request: Request, storage_path: str | None = None) -> list[ModelDownloadStatus]:
    settings = request.app.state.settings
    store = request.app.state.store
    storage_root = _model_download_storage_root(settings, storage_path)
    storage_root.mkdir(parents=True, exist_ok=True)
    key = _model_download_key(storage_root)
    with store.model_download_lock:
        job = dict(store.model_download_jobs.get(key) or {})

    local_dir = _local_omnivoice_dir(storage_root)
    repo_root = _repo_cache_root(storage_root)
    snapshot = _latest_hf_snapshot(storage_root)
    ready_path = local_dir if local_dir.exists() else snapshot
    cache_path = local_dir if local_dir.exists() else repo_root if repo_root.exists() else None
    size_source = ready_path or cache_path

    if job.get('status') == 'downloading':
        status = 'downloading'
    elif ready_path or job.get('status') == 'ready':
        status = 'ready'
    elif job.get('status') == 'error':
        status = 'error'
    elif repo_root.exists():
        status = 'partial'
    else:
        status = 'missing'

    statuses: list[ModelDownloadStatus] = []
    for model_id in settings.supported_models:
        metadata = MODEL_DOWNLOAD_META.get(model_id, {})
        statuses.append(
            ModelDownloadStatus(
                id=model_id,
                label=str(metadata.get('label') or model_id),
                kind=str(metadata.get('kind') or 'model'),
                status=status,
                local_path=str(ready_path) if ready_path else None,
                cache_path=str(cache_path) if cache_path else None,
                error=str(job.get('error')) if job.get('status') == 'error' and job.get('error') else None,
                updated_at=str(job.get('updated_at')) if job.get('updated_at') else None,
                storage_root=str(storage_root),
                approx_size_gb=metadata.get('approx_size_gb'),
                size_on_disk_gb=_directory_size_gb(size_source),
            )
        )
    return statuses


def _queue_model_download(request: Request, model_id: str, storage_path: str | None = None) -> dict[str, Any]:
    settings = request.app.state.settings
    store = request.app.state.store
    if model_id not in settings.supported_models:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Unsupported model id')
    storage_root = _model_download_storage_root(settings, storage_path)
    storage_root.mkdir(parents=True, exist_ok=True)
    key = _model_download_key(storage_root)

    with store.model_download_lock:
        existing = store.model_download_jobs.get(key)
        if existing and existing.get('status') == 'downloading':
            return dict(existing)
        job = {
            'model_id': model_id,
            'repo_id': OMNIVOICE_MODEL_ID,
            'storage_root': str(storage_root),
            'status': 'downloading',
            'error': None,
            'updated_at': utcnow().isoformat(),
        }
        store.model_download_jobs[key] = job

    if settings.runtime_backend.lower() == 'mock':
        with store.model_download_lock:
            store.model_download_jobs[key] = {**job, 'status': 'ready', 'updated_at': utcnow().isoformat()}
            return dict(store.model_download_jobs[key])

    def worker() -> None:
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(OMNIVOICE_MODEL_ID, cache_dir=str(storage_root), resume_download=True)
            next_job = {**job, 'status': 'ready', 'error': None, 'updated_at': utcnow().isoformat()}
        except Exception as exc:
            next_job = {**job, 'status': 'error', 'error': str(exc), 'updated_at': utcnow().isoformat()}
        with store.model_download_lock:
            store.model_download_jobs[key] = next_job

    threading.Thread(target=worker, daemon=True).start()
    return job


def _delete_model_cache(request: Request, model_id: str, storage_path: str | None = None) -> dict[str, Any]:
    settings = request.app.state.settings
    store = request.app.state.store
    if model_id not in settings.supported_models:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Unsupported model id')
    storage_root = _model_download_storage_root(settings, storage_path)
    key = _model_download_key(storage_root)
    with store.model_download_lock:
        existing = store.model_download_jobs.get(key)
        if existing and existing.get('status') == 'downloading':
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Model is still downloading')

    local_dir = _local_omnivoice_dir(storage_root)
    repo_root = _repo_cache_root(storage_root)
    target = local_dir if local_dir.exists() else repo_root if repo_root.exists() else None
    removed = False
    removed_path = None
    if target is not None:
        shutil.rmtree(target, ignore_errors=False)
        removed = True
        removed_path = str(target)

    with store.model_download_lock:
        store.model_download_jobs.pop(key, None)
    return {'ok': True, 'removed': removed, 'removed_path': removed_path}


def _job_item(request: Request, job_id: str) -> JobListItem:
    job = request.app.state.store.jobs[job_id]
    return JobListItem(
        job_id=job.job_id,
        status=job.status,
        model=job.model_used or job.request.model,
        task_type=_effective_task_type(job.model_used or job.request.model, job.request.task_type),
        voice=job.request.voice,
        input_preview=job.preview(),
        queue_position=job.queue_position,
        eta_ms=job.eta_ms,
        created_at=job.created_at,
        updated_at=job.updated_at,
        metrics=job.metrics,
        error_message=job.error_message,
    )


def _job_detail(request: Request, job_id: str) -> JobDetailResponse:
    job = request.app.state.store.jobs[job_id]
    return JobDetailResponse(
        job_id=job.job_id,
        status=job.status,
        model=job.model_used or job.request.model,
        task_type=_effective_task_type(job.model_used or job.request.model, job.request.task_type),
        voice=job.request.voice,
        input_preview=job.preview(),
        queue_position=job.queue_position,
        eta_ms=job.eta_ms,
        created_at=job.created_at,
        updated_at=job.updated_at,
        metrics=job.metrics,
        error_message=job.error_message,
        started_at=job.started_at,
        first_audio_at=job.first_audio_at,
        completed_at=job.completed_at,
        sentences_total=job.sentences_total,
        batch_count=int(job.metrics.get('batch_count') or 0),
    )


def _admin_key_metadata(request: Request) -> AdminKeyMetadata:
    record = get_admin_record(request.app.state.store)
    return AdminKeyMetadata(
        key_id=record.key_id,
        label='Master Admin Key',
        created_at=record.created_at,
        last_used_at=record.last_used_at,
    )


def _dashboard_snapshot(request: Request) -> DashboardSnapshot:
    store = request.app.state.store
    stats = request.app.state.stats_service.build_stats(store)
    gpu: GpuStatsResponse = request.app.state.stats_service.build_gpu_stats()
    current_batch = None
    if store.current_batch:
        current_batch = BatchSnapshot(
            batch_id=store.current_batch['batch_id'],
            model_id=store.current_batch['model_id'],
            task_type=TaskType(store.current_batch['task_type']),
            voice=store.current_batch.get('voice'),
            language=store.current_batch.get('language'),
            size=store.current_batch['size'],
            started_at=store.current_batch['started_at'],
            request_ids=list(store.current_batch.get('request_ids', [])),
            sentence_indices=list(store.current_batch.get('sentence_indices', [])),
        )

    recent_batches = [
        BatchSnapshot(
            batch_id=item['batch_id'],
            model_id=item['model_id'],
            task_type=TaskType(item['task_type']),
            voice=item.get('voice'),
            language=item.get('language'),
            size=item['size'],
            started_at=item['started_at'],
            request_ids=list(item.get('request_ids', [])),
            sentence_indices=list(item.get('sentence_indices', [])),
        )
        for item in list(store.recent_batches)[-10:]
    ]

    overview = DashboardOverview(
        active_model=stats.active_model,
        queue_depth=stats.queue_depth,
        active_requests=store.active_requests(),
        worker_state=stats.worker_state,
        ttfa_ms_avg=stats.rolling.ttfa_ms_avg,
        queue_wait_ms_avg=stats.rolling.queue_wait_ms_avg,
        job_wall_ms_avg=stats.rolling.job_wall_ms_avg,
        realtime_x_avg=stats.rolling.realtime_x_avg,
        jobs_total=stats.global_.jobs_total,
        audio_seconds_total=stats.global_.audio_seconds_total,
        gpu_name=gpu.name,
        gpu_memory_used_mb=gpu.memory_used_mb,
        gpu_memory_total_mb=gpu.memory_total_mb,
        gpu_utilization_pct=gpu.utilization_percent,
        gpu_temperature_c=gpu.temperature_c,
    )
    jobs = [_job_item(request, job_id) for job_id in sorted(store.jobs, key=lambda value: store.jobs[value].created_at, reverse=True)[:24]]
    return DashboardSnapshot(
        overview=overview,
        settings=_settings_response(request.app.state.settings),
        models=_model_items(request),
        voices=_voice_items(request, include_details=True),
        jobs=jobs,
        admin_key=_admin_key_metadata(request),
        current_batch=current_batch,
        recent_batches=recent_batches,
    )


async def _submit_or_429(queue: QueueService, payload: SpeechRequest, *, owner_scope: str = 'public'):
    try:
        return await queue.submit(payload, owner_scope=owner_scope)
    except QueueSaturatedError as exc:
        # Genuine rate limit -> retryable.
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)) from exc
    except RequestTooLargeError as exc:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)) from exc
    except RuntimeError as exc:
        # Validation failures (empty input, bad voice/model) are permanent -> 400, not 429.
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@health.get('/health')
async def healthcheck() -> dict[str, bool]:
    return {'ok': True}


@router.get('/api/health')
async def healthcheck_api() -> dict[str, bool]:
    return await healthcheck()


@admin.get('/keys', response_model=AdminKeyResponse)
async def get_admin_keys(request: Request) -> AdminKeyResponse:
    return AdminKeyResponse(admin_key=_admin_key_metadata(request))


@admin.post('/keys', response_model=AdminKeyRotateResponse)
async def rotate_keys(request: Request) -> AdminKeyRotateResponse:
    record, token = rotate_admin_key(request.app.state.store, request.app.state.settings)
    return AdminKeyRotateResponse(
        admin_key=AdminKeyMetadata(
            key_id=record.key_id,
            label='Master Admin Key',
            created_at=record.created_at,
            last_used_at=record.last_used_at,
        ),
        token=token,
    )


@admin.get('/settings', response_model=ServerSettingsResponse)
async def get_admin_settings(request: Request) -> ServerSettingsResponse:
    return _settings_response(request.app.state.settings)


@admin.put('/settings', response_model=ServerSettingsResponse)
async def update_admin_settings(request: Request, payload: ServerSettingsUpdateRequest) -> ServerSettingsResponse:
    settings = request.app.state.settings
    store = request.app.state.store

    if payload.default_model is not None:
        if payload.default_model not in settings.supported_models:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Unsupported default model')
        settings.active_model = payload.default_model
    if payload.default_voice is not None:
        settings.default_voice = payload.default_voice
    if payload.model_directory is not None:
        settings.models_root_dir = Path(payload.model_directory).expanduser()
        settings.models_root_dir.mkdir(parents=True, exist_ok=True)
    if payload.whisper_base_url is not None:
        settings.whisper_base_url = payload.whisper_base_url or None
    if payload.whisper_path is not None:
        settings.whisper_path = ''
    if payload.vllm_base_url is not None:
        settings.vllm_base_url = payload.vllm_base_url
    if payload.vllm_model is not None:
        settings.vllm_model = payload.vllm_model
    if payload.wer_concurrency is not None:
        settings.wer_concurrency = payload.wer_concurrency
    if payload.wer_transcription_concurrency is not None:
        settings.wer_transcription_concurrency = payload.wer_transcription_concurrency
    if payload.retention_days is not None:
        settings.retention_days = payload.retention_days
    if payload.queue_limit is not None:
        settings.max_queue_size = payload.queue_limit
        store.max_queue_size = payload.queue_limit
    if payload.allow_model_downloads is not None:
        settings.allow_model_downloads = payload.allow_model_downloads
    if payload.preferred_device is not None:
        settings.preferred_device = payload.preferred_device
    if payload.attention_implementation is not None:
        settings.attention_implementation = payload.attention_implementation
    if payload.torch_dtype is not None:
        dtype_aliases = {'fp16': 'float16', 'bf16': 'bfloat16', 'fp32': 'float32', 'float8': 'fp8'}
        requested_dtype = dtype_aliases.get(payload.torch_dtype.lower(), payload.torch_dtype.lower())
        if requested_dtype not in {'float16', 'bfloat16', 'float32', 'fp8'}:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Unsupported torch dtype')
        settings.torch_dtype = requested_dtype
    if payload.compile_model is not None:
        settings.compile_model = payload.compile_model
    if payload.compile_cudagraphs is not None:
        settings.compile_cudagraphs = payload.compile_cudagraphs
    if payload.cudagraph_skip_dynamic_graphs is not None:
        settings.cudagraph_skip_dynamic_graphs = payload.cudagraph_skip_dynamic_graphs
    if payload.cuda_memory_trim_after_batch is not None:
        settings.cuda_memory_trim_after_batch = payload.cuda_memory_trim_after_batch
    if payload.warmup_on_startup is not None:
        settings.warmup_on_startup = payload.warmup_on_startup
    if payload.poll_interval_ms is not None:
        settings.frontend_poll_interval_ms = payload.poll_interval_ms
    if payload.theme is not None:
        settings.frontend_theme = payload.theme
    if payload.sentence_chunking is not None:
        settings.sentence_chunking = payload.sentence_chunking
    if payload.short_sentence_merge_max_chars is not None:
        settings.short_sentence_merge_max_chars = payload.short_sentence_merge_max_chars
    if payload.following_sentence_merge_min_chars is not None:
        settings.following_sentence_merge_min_chars = payload.following_sentence_merge_min_chars
    if payload.max_parallel_requests is not None:
        settings.max_parallel_requests = payload.max_parallel_requests
    if payload.max_batch_size is not None:
        settings.max_batch_size = payload.max_batch_size
    if payload.batch_wait_ms is not None:
        settings.batch_wait_ms = payload.batch_wait_ms
    if payload.vram_budget_mb is not None:
        settings.vram_budget_mb = payload.vram_budget_mb
    if payload.max_input_chars is not None:
        settings.max_input_chars = payload.max_input_chars
    if payload.stream_chunk_ms is not None:
        settings.stream_chunk_ms = payload.stream_chunk_ms
    if payload.stream_prebuffer_ms is not None:
        settings.stream_prebuffer_ms = payload.stream_prebuffer_ms
    if payload.num_step is not None:
        settings.num_step = payload.num_step
    if payload.guidance_scale is not None:
        settings.guidance_scale = payload.guidance_scale
    if payload.duration is not None:
        settings.duration = payload.duration
    if payload.t_shift is not None:
        settings.t_shift = payload.t_shift
    if payload.denoise is not None:
        settings.denoise = payload.denoise
    if payload.preprocess_prompt is not None:
        settings.preprocess_prompt = payload.preprocess_prompt
    if payload.postprocess_output is not None:
        settings.postprocess_output = payload.postprocess_output
    if payload.audio_chunk_duration is not None:
        settings.audio_chunk_duration = payload.audio_chunk_duration
    if payload.audio_chunk_threshold is not None:
        settings.audio_chunk_threshold = payload.audio_chunk_threshold
    if payload.position_temperature is not None:
        settings.position_temperature = payload.position_temperature
    if payload.class_temperature is not None:
        settings.class_temperature = payload.class_temperature

    save_runtime_settings(settings)
    return _settings_response(settings)


@admin.get('/settings/presets', response_model=SettingsPresetListResponse)
async def list_settings_presets(request: Request) -> SettingsPresetListResponse:
    data_dir = request.app.state.settings.data_dir
    return SettingsPresetListResponse(presets=[SettingsPresetItem(**item) for item in _list_presets(data_dir)])


@admin.put('/settings/presets/{name}', response_model=SettingsPresetItem)
async def save_settings_preset(request: Request, name: str, payload: SettingsPresetSaveRequest) -> SettingsPresetItem:
    data_dir = request.app.state.settings.data_dir
    try:
        entry = _save_preset(data_dir, name, payload.values, now_iso=utcnow().isoformat())
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return SettingsPresetItem(**entry)


@admin.delete('/settings/presets/{name}')
async def delete_settings_preset(request: Request, name: str) -> dict[str, bool]:
    if not _delete_preset(request.app.state.settings.data_dir, name):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Preset not found')
    return {'ok': True}


@admin.get('/snapshot', response_model=DashboardSnapshot)
async def admin_snapshot(request: Request) -> DashboardSnapshot:
    return _dashboard_snapshot(request)


@admin.get('/models', response_model=ModelDownloadListResponse)
async def admin_models(request: Request, storage_path: str | None = None) -> ModelDownloadListResponse:
    return ModelDownloadListResponse(models=_model_download_statuses(request, storage_path=storage_path))


@admin.get('/vllm/models', response_model=VllmModelsResponse)
async def admin_vllm_models(request: Request, base_url: str | None = None) -> VllmModelsResponse:
    settings = request.app.state.settings
    target_base_url = _normalize_vllm_base_url(base_url or settings.vllm_base_url)
    selected = settings.vllm_model or None
    if not target_base_url:
        return VllmModelsResponse(ok=False, base_url='', selected_model=selected, error='Missing vLLM Base URL.')
    if target_base_url.lower() == 'mock' or settings.runtime_backend.lower() == 'mock':
        return VllmModelsResponse(ok=True, base_url=target_base_url, models=['mock-qwen3-35b'], selected_model=selected)
    try:
        models_url = f'{target_base_url}/v1/models'
        async with httpx.AsyncClient(timeout=10, trust_env=False) as client:
            response = await client.get(models_url)
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:
        return VllmModelsResponse(ok=False, base_url=target_base_url, selected_model=selected, error=str(exc))
    models = []
    seen: set[str] = set()
    for item in payload.get('data') or []:
        model_id = str(item.get('id') or item.get('model') or '').strip()
        if model_id and model_id not in seen:
            models.append(model_id)
            seen.add(model_id)
    return VllmModelsResponse(ok=bool(models), base_url=target_base_url, models=models, selected_model=selected, error=None if models else 'vLLM returned no models.')


@admin.post('/models/preload', response_model=ModelOperationResponse)
async def admin_preload_model(request: Request, payload: ModelOperationRequest) -> ModelOperationResponse:
    model_id, warm_ms = await request.app.state.synthesizer.preload(payload.model or request.app.state.settings.active_model)
    request.app.state.store.active_model = model_id
    request.app.state.store.models_loaded.clear()
    request.app.state.store.models_loaded.add(model_id)
    await request.app.state.events.publish('dashboard.snapshot', {'reason': 'model.preloaded', 'model': model_id})
    return ModelOperationResponse(ok=True, model=model_id, warm_ms=warm_ms, message='Model preloaded.')


@admin.post('/models/download', response_model=ModelDownloadActionResponse)
async def admin_download_model(request: Request, payload: ModelDownloadRequest) -> ModelDownloadActionResponse:
    settings = request.app.state.settings
    settings.allow_model_downloads = True
    save_runtime_settings(settings)
    model_id = payload.model_id or payload.model or settings.active_model
    job = _queue_model_download(request, model_id, storage_path=payload.storage_path)
    await request.app.state.events.publish('dashboard.snapshot', {'reason': 'model.downloaded', 'model': model_id})
    return ModelDownloadActionResponse(
        ok=True,
        job=job,
        models=_model_download_statuses(request, storage_path=payload.storage_path),
    )


@admin.post('/models/delete', response_model=ModelDownloadActionResponse)
async def admin_delete_model_cache(request: Request, payload: ModelDownloadRequest) -> ModelDownloadActionResponse:
    model_id = payload.model_id or payload.model or request.app.state.settings.active_model
    result = _delete_model_cache(request, model_id, storage_path=payload.storage_path)
    await request.app.state.events.publish('dashboard.snapshot', {'reason': 'model.cache_deleted', 'model': model_id})
    return ModelDownloadActionResponse(
        ok=True,
        removed=result.get('removed'),
        removed_path=result.get('removed_path'),
        models=_model_download_statuses(request, storage_path=payload.storage_path),
    )


@admin.post('/models/warmup', response_model=ModelOperationResponse)
async def admin_warmup_model(request: Request, payload: ModelOperationRequest) -> ModelOperationResponse:
    warmup_request = SpeechRequest(
        input='Warmup.',
        model=payload.model or request.app.state.settings.active_model,
        task_type=payload.task_type,
        voice=payload.voice,
        instructions=payload.instructions or (
            DEFAULT_VOICE_DESIGN_INSTRUCT if payload.task_type == TaskType.voice_design else None
        ),
        language=payload.language or 'Auto',
    )
    model_id, warm_ms = await request.app.state.synthesizer.warmup(warmup_request.model, warmup_request)
    request.app.state.store.active_model = model_id
    request.app.state.store.models_loaded.clear()
    request.app.state.store.models_loaded.add(model_id)
    await request.app.state.events.publish('dashboard.snapshot', {'reason': 'model.warmed', 'model': model_id})
    return ModelOperationResponse(ok=True, model=model_id, warm_ms=warm_ms, message='Warmup completed.')


@admin.post('/models/unload', response_model=ModelOperationResponse)
async def admin_unload_model(request: Request, payload: ModelOperationRequest) -> ModelOperationResponse:
    model_id, warm_ms = await request.app.state.synthesizer.unload()
    request.app.state.store.models_loaded.clear()
    request.app.state.store.active_model = payload.model or request.app.state.settings.active_model
    await request.app.state.events.publish('dashboard.snapshot', {'reason': 'model.unloaded', 'model': model_id})
    return ModelOperationResponse(ok=True, model=model_id, warm_ms=warm_ms, message='Model unloaded.')


@admin.post('/models/reload', response_model=ModelOperationResponse)
async def admin_reload_model(request: Request, payload: ModelOperationRequest) -> ModelOperationResponse:
    model_id, warm_ms = await request.app.state.synthesizer.reload(payload.model or request.app.state.settings.active_model)
    request.app.state.store.active_model = model_id
    request.app.state.store.models_loaded.clear()
    request.app.state.store.models_loaded.add(model_id)
    await request.app.state.events.publish('dashboard.snapshot', {'reason': 'model.reloaded', 'model': model_id})
    return ModelOperationResponse(ok=True, model=model_id, warm_ms=warm_ms, message='Model reloaded.')


@admin.post('/runtime/free-memory', response_model=MemoryCleanupResponse)
async def admin_free_memory(request: Request) -> MemoryCleanupResponse:
    cleanup = await request.app.state.synthesizer.free_memory()
    await request.app.state.events.publish('dashboard.snapshot', {'reason': 'runtime.memory_freed'})
    return MemoryCleanupResponse(
        ok=bool(cleanup.get('ok', True)),
        message=str(cleanup.get('message') or 'CUDA cache trimmed.'),
        memory_before_mb=cleanup.get('memory_before_mb'),
        memory_after_mb=cleanup.get('memory_after_mb'),
        memory_total_mb=cleanup.get('memory_total_mb'),
        released_mb=cleanup.get('released_mb'),
    )


@admin.get('/dashboard/stream')
async def admin_dashboard_stream(request: Request) -> StreamingResponse:
    events: EventHub = request.app.state.events
    queue = await events.subscribe()

    async def iterator() -> Any:
        try:
            yield EventHub.encode_sse('dashboard.snapshot', _dashboard_snapshot(request).model_dump(mode='json'))
            while True:
                if await request.is_disconnected():
                    break
                interval_seconds = max(0.25, min(float(request.app.state.settings.frontend_poll_interval_ms) / 1000.0, 5.0))
                try:
                    await asyncio.wait_for(queue.get(), timeout=interval_seconds)
                except asyncio.TimeoutError:
                    pass
                yield EventHub.encode_sse('dashboard.snapshot', _dashboard_snapshot(request).model_dump(mode='json'))
        finally:
            events.unsubscribe(queue)

    return StreamingResponse(iterator(), media_type='text/event-stream')


@admin.get('/jobs', response_model=list[JobListItem])
async def admin_jobs(request: Request) -> list[JobListItem]:
    return [
        _job_item(request, job_id)
        for job_id in sorted(request.app.state.store.jobs, key=lambda value: request.app.state.store.jobs[value].created_at, reverse=True)
    ]


@admin.get('/jobs/{job_id}', response_model=JobDetailResponse)
async def admin_job_detail(request: Request, job_id: str) -> JobDetailResponse:
    if job_id not in request.app.state.store.jobs:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Job not found')
    return _job_detail(request, job_id)


@admin.get('/jobs/{job_id}/audio')
async def admin_job_audio(request: Request, job_id: str) -> Response:
    job = request.app.state.store.jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Job not found')
    if not job.final_audio:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Audio not ready')
    return Response(content=job.final_audio, media_type=job.content_type or 'audio/wav')


@admin.delete('/jobs/{job_id}')
async def admin_delete_job(request: Request, job_id: str) -> dict[str, bool]:
    queue: QueueService = request.app.state.queue_service
    if job_id not in request.app.state.store.jobs:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Job not found')
    await queue.delete(job_id)
    return {'ok': True}


@admin.post('/benchmarks/runs', response_model=BenchmarkRunResponse)
async def admin_create_benchmark(request: Request, payload: BenchmarkRunCreateRequest) -> BenchmarkRunResponse:
    service: BenchmarkService = request.app.state.benchmark_service
    return await service.create_run(payload)


@admin.get('/benchmarks/runs', response_model=list[BenchmarkRunResponse])
async def admin_list_benchmarks(request: Request) -> list[BenchmarkRunResponse]:
    service: BenchmarkService = request.app.state.benchmark_service
    return await service.list_runs()


@admin.post('/wer-benchmarks/runs', response_model=WerBenchmarkRunResponse)
async def admin_create_wer_benchmark(request: Request, payload: WerBenchmarkCreateRequest) -> WerBenchmarkRunResponse:
    service: WerBenchmarkService = request.app.state.wer_benchmark_service
    return await service.create_run(payload)


@admin.get('/wer-benchmarks/runs', response_model=list[WerBenchmarkRunResponse])
async def admin_list_wer_benchmarks(request: Request) -> list[WerBenchmarkRunResponse]:
    service: WerBenchmarkService = request.app.state.wer_benchmark_service
    return await service.list_runs()


@admin.get('/voices', response_model=list[VoiceProfileListItem])
async def admin_list_voices(request: Request) -> list[VoiceProfileListItem]:
    return _voice_items(request, include_details=True)


@admin.post('/voices', response_model=VoiceProfileCreateResponse)
async def create_voice_profile(
    request: Request,
    audio_sample: UploadFile = File(...),
    name: str = Form(...),
    consent: bool = Form(False),
    ref_text: str | None = Form(default=None),
) -> VoiceProfileCreateResponse:
    if not consent:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Consent is required to save an OmniVoice clone profile.')
    if not (ref_text or '').strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='ref_text is required for OmniVoice clone profiles.')
    raw = await audio_sample.read()
    wav_bytes = await asyncio.to_thread(_transcode_to_wav, raw)
    store = request.app.state.store
    voice = VoiceProfileRecord(
        voice_id=new_id('voice'),
        name=name,
        source='custom',
        created_at=utcnow(),
        audio_bytes=wav_bytes,
        content_type='audio/wav',
        filename=(audio_sample.filename or 'sample') + '.normalized.wav',
        ref_text=ref_text,
        consent=consent,
    )
    store.voice_profiles[voice.voice_id] = voice
    store.prompt_cache.clear()
    store.save_voices(request.app.state.settings.data_dir)
    await request.app.state.events.publish('dashboard.snapshot', {'reason': 'voice.created'})
    return VoiceProfileCreateResponse(voice_id=voice.voice_id, name=voice.name, source=voice.source, created_at=voice.created_at)


def _transcode_to_wav(raw: bytes) -> bytes:
    import numpy as np
    import soundfile as sf

    buf = io.BytesIO(raw)
    try:
        audio, sr = sf.read(buf, dtype='float32', always_2d=False)
    except Exception:
        try:
            import librosa

            buf.seek(0)
            audio, sr = librosa.load(buf, sr=None, mono=True)
        except Exception as exc:
            raise RuntimeError(f'Could not decode the uploaded audio: {exc}') from exc

    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)

    target_sr = 24_000
    if sr != target_sr:
        try:
            import librosa

            audio = librosa.resample(audio.astype(np.float32), orig_sr=int(sr), target_sr=target_sr)
            sr = target_sr
        except Exception as exc:
            raise RuntimeError(f'Could not resample the uploaded audio: {exc}') from exc

    out = io.BytesIO()
    sf.write(out, audio.astype(np.float32), sr, format='WAV', subtype='PCM_16')
    return out.getvalue()


@admin.delete('/voices/{voice_id}')
async def delete_voice_profile(request: Request, voice_id: str) -> dict[str, bool]:
    store = request.app.state.store
    if voice_id not in store.voice_profiles:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Voice not found')
    store.voice_profiles.pop(voice_id, None)
    store.prompt_cache.clear()
    store.save_voices(request.app.state.settings.data_dir)
    await request.app.state.events.publish('dashboard.snapshot', {'reason': 'voice.deleted'})
    return {'ok': True}


@admin.get('/voices/{voice_id}/audio')
async def get_voice_profile_audio(request: Request, voice_id: str) -> Response:
    voice = request.app.state.store.voice_profiles.get(voice_id)
    if not voice:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Voice not found')
    if not voice.audio_bytes:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Voice sample not found')
    filename = voice.filename or f'{voice.voice_id}.wav'
    return Response(
        content=voice.audio_bytes,
        media_type=voice.content_type or 'audio/wav',
        headers={'Content-Disposition': f'inline; filename="{filename}"'},
    )


@admin.post('/voices/transcribe', response_model=TranscriptionResponse)
async def transcribe(request: Request, file: UploadFile = File(...)) -> TranscriptionResponse:
    data = await file.read()
    service: TranscriptionService = request.app.state.transcription_service
    return await service.transcribe(file.filename or 'audio.wav', file.content_type or 'audio/wav', data)


@router.get('/api/v1/voices', response_model=VoiceCatalogResponse)
async def list_public_voices(request: Request) -> VoiceCatalogResponse:
    return VoiceCatalogResponse(voices=_voice_items(request))


@router.get('/v1/voices', response_model=list[VoiceProfileListItem])
async def list_voices_alias(request: Request) -> list[VoiceProfileListItem]:
    return _voice_items(request)


@router.get('/v1/audio/voices', response_model=list[VoiceProfileListItem])
async def list_audio_voices(request: Request) -> list[VoiceProfileListItem]:
    return _voice_items(request)


@router.get('/v1/models', response_model=list[ModelInfo])
async def list_models(request: Request) -> list[ModelInfo]:
    return _model_items(request)


@router.post('/api/v1/synthesize', response_model=SynthesisResultResponse)
async def synthesize(request: Request, payload: SpeechRequest) -> SynthesisResultResponse:
    queue: QueueService = request.app.state.queue_service
    job = await _submit_or_429(queue, payload.model_copy(update={'stream': False}), owner_scope='public')
    finished = await queue.wait_for_completion(job.job_id)
    if finished.status == JobStatus.cancelled:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=finished.error_message or 'Synthesis cancelled')
    if finished.status != JobStatus.completed:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=finished.error_message or 'Synthesis failed')
    return SynthesisResultResponse(
        job_id=finished.job_id,
        status=finished.status,
        model=finished.model_used or finished.request.model,
        task_type=_effective_task_type(finished.model_used or finished.request.model, finished.request.task_type),
        voice=finished.request.voice,
        sample_rate=finished.sample_rate,
        metrics=finished.metrics,
    )


@router.post('/api/v1/synthesize/stream')
async def synthesize_stream(request: Request, payload: SpeechRequest) -> StreamingResponse:
    queue: QueueService = request.app.state.queue_service
    job = await _submit_or_429(queue, payload.model_copy(update={'stream': True, 'response_format': 'pcm'}), owner_scope='public')

    async def iterator() -> Any:
        disconnected = False
        try:
            while True:
                if await request.is_disconnected():
                    disconnected = True
                    break
                try:
                    event = await asyncio.wait_for(job.stream_events.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                if event is None:
                    break
                yield json.dumps(event) + '\n'
        finally:
            # Client gone mid-stream: cancel so the single worker stops rendering
            # audio nobody consumes (otherwise the job runs to completion and its
            # PCM piles up in the unbounded stream queue).
            if disconnected:
                try:
                    await queue.cancel(job.job_id)
                except Exception:
                    pass

    return StreamingResponse(iterator(), media_type='application/x-ndjson')


@router.post('/v1/audio/speech', response_model=None)
async def speech(request: Request, payload: SpeechRequest) -> Response | StreamingResponse:
    queue: QueueService = request.app.state.queue_service
    if payload.stream:
        job = await _submit_or_429(queue, payload.model_copy(update={'stream': True, 'response_format': 'pcm'}), owner_scope='public')

        async def stream_job() -> Any:
            disconnected = False
            try:
                while True:
                    if await request.is_disconnected():
                        disconnected = True
                        break
                    try:
                        chunk = await asyncio.wait_for(job.stream_chunks.get(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue
                    if chunk is None:
                        break
                    yield chunk
            finally:
                if disconnected:
                    try:
                        await queue.cancel(job.job_id)
                    except Exception:
                        pass

        return StreamingResponse(stream_job(), media_type='audio/pcm')

    job = await _submit_or_429(queue, payload.model_copy(update={'stream': False}), owner_scope='public')
    finished = await queue.wait_for_completion(job.job_id)
    if finished.status == JobStatus.cancelled:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=finished.error_message or 'Synthesis cancelled')
    if finished.status != JobStatus.completed or not finished.final_audio:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=finished.error_message or 'Synthesis failed')
    return Response(content=finished.final_audio, media_type=finished.content_type or 'audio/wav')


@router.post('/v1/jobs', response_model=SpeechJobCreateResponse)
async def create_job(request: Request, payload: SpeechRequest) -> SpeechJobCreateResponse:
    queue: QueueService = request.app.state.queue_service
    job = await _submit_or_429(queue, payload, owner_scope='public')
    return SpeechJobCreateResponse(job_id=job.job_id, status=job.status, queue_position=job.queue_position, eta_ms=job.eta_ms)


router.include_router(health)
router.include_router(admin)
