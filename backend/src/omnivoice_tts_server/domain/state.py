from __future__ import annotations

import asyncio
import json
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .models import JobStatus, SpeechRequest

def _json_load(path: Path) -> dict[str, Any] | list[Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

def utcnow() -> datetime:
    return datetime.now(timezone.utc)

def new_id(prefix: str) -> str:
    return f'{prefix}_{uuid4().hex[:12]}'


@dataclass
class ApiKeyRecord:
    key_id: str
    name: str
    key_hash: str
    created_at: datetime
    last_used_at: datetime | None = None
    disabled: bool = False


@dataclass
class VoiceProfileRecord:
    voice_id: str
    name: str
    source: str
    created_at: datetime
    audio_bytes: bytes | None = None
    content_type: str | None = None
    filename: str | None = None
    ref_text: str | None = None
    consent: bool = False


@dataclass
class RequestState:
    job_id: str
    group_key: str
    sentences: list[str]
    pending_sentence_indices: deque[int] = field(init=False)
    inflight_sentence_indices: set[int] = field(default_factory=set)
    ready_sentence_pcm: dict[int, bytes] = field(default_factory=dict)
    sentence_duration_ms: dict[int, int] = field(default_factory=dict)
    pending_preview_pcm: dict[int, list[bytes]] = field(default_factory=dict)
    pending_preview_duration_ms: dict[int, list[int]] = field(default_factory=dict)
    completed_streaming_sentence_indices: set[int] = field(default_factory=set)
    emitted_samples_by_sentence: dict[int, int] = field(default_factory=dict)
    chunk_index_by_sentence: dict[int, int] = field(default_factory=dict)
    next_emit_sentence_index: int = 0
    emitted_audio_ms: int = 0
    batch_count: int = 0
    sample_rate: int = 24_000

    def __post_init__(self) -> None:
        self.pending_sentence_indices = deque(range(len(self.sentences)))

    def has_pending_sentences(self) -> bool:
        return bool(self.pending_sentence_indices)

    def is_complete(self) -> bool:
        return (
            self.next_emit_sentence_index >= len(self.sentences)
            and not self.pending_sentence_indices
            and not self.inflight_sentence_indices
        )


@dataclass
class JobRecord:
    job_id: str
    request: SpeechRequest
    created_at: datetime
    updated_at: datetime
    status: JobStatus = JobStatus.queued
    queue_position: int = 0
    eta_ms: int = 0
    started_at: datetime | None = None
    first_audio_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    model_used: str | None = None
    final_audio: bytes | None = None
    content_type: str | None = None
    cancel_requested: bool = False
    stream_chunks: asyncio.Queue[bytes | None] = field(default_factory=asyncio.Queue)
    stream_events: asyncio.Queue[dict[str, Any] | None] = field(default_factory=asyncio.Queue)
    metrics: dict[str, Any] = field(default_factory=dict)
    pcm_parts: list[bytes] = field(default_factory=list)
    sample_rate: int = 24_000
    sentences_total: int = 0
    owner_scope: str = 'public'

    def preview(self, length: int = 80) -> str:
        text = (self.request.input or '').strip()
        return text if len(text) <= length else f'{text[: length - 1]}...'


class InMemoryStore:
    def __init__(self, max_queue_size: int) -> None:
        self.max_queue_size = max_queue_size
        self.jobs: dict[str, JobRecord] = {}
        self.waiting_requests: deque[str] = deque()
        self.active_request_ids: list[str] = []
        self.request_states: dict[str, RequestState] = {}
        self.job_condition = asyncio.Condition()
        self.worker_stop = asyncio.Event()
        self.worker_task: asyncio.Task[None] | None = None
        # Long-running side tasks (benchmark/WER runs, reaper) kept referenced so the
        # event loop does not GC them mid-run, and so they can be cancelled at shutdown.
        self.background_tasks: set[asyncio.Task[Any]] = set()
        self.worker_state = 'idle'
        self.models_loaded: set[str] = set()
        self.active_model: str | None = None
        self.api_keys: dict[str, ApiKeyRecord] = {}
        self.voice_profiles: dict[str, VoiceProfileRecord] = {}
        self.event_subscribers: set[asyncio.Queue[dict[str, Any] | None]] = set()
        self.completed_job_metrics: deque[dict[str, float]] = deque(maxlen=128)
        self.total_jobs_completed = 0
        self.total_audio_seconds = 0.0
        self.exclusive_lock = asyncio.Lock()
        self.current_batch: dict[str, Any] | None = None
        self.recent_batches: deque[dict[str, Any]] = deque(maxlen=64)
        self.prompt_cache: dict[str, Any] = {}
        self.benchmark_runs: dict[str, Any] = {}
        self.wer_benchmark_runs: dict[str, Any] = {}
        self.wer_sentence_cache: dict[str, dict[str, Any]] = {}
        self.model_download_jobs: dict[tuple[str, str], dict[str, Any]] = {}
        self.model_download_lock = threading.RLock()
        self.file_lock = threading.RLock()

    def load_secrets(self, data_dir: Path) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        secrets_path = data_dir / "server_secrets.json"
        with self.file_lock:
            payload = _json_load(secrets_path)
            if isinstance(payload, dict):
                for key, item in payload.items():
                    if isinstance(item, dict) and "key_id" in item:
                        last_used = item.get("last_used_at")
                        self.api_keys[item["key_id"]] = ApiKeyRecord(
                            key_id=item["key_id"],
                            name=item["name"],
                            key_hash=item["key_hash"],
                            created_at=datetime.fromisoformat(item["created_at"]),
                            last_used_at=datetime.fromisoformat(last_used) if last_used else None,
                            disabled=item.get("disabled", False)
                        )

    def save_secrets(self, data_dir: Path) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        secrets_path = data_dir / "server_secrets.json"
        with self.file_lock:
            payload = {}
            for record in self.api_keys.values():
                payload[record.key_id] = {
                    "key_id": record.key_id,
                    "name": record.name,
                    "key_hash": record.key_hash,
                    "created_at": record.created_at.isoformat(),
                    "last_used_at": record.last_used_at.isoformat() if record.last_used_at else None,
                    "disabled": record.disabled
                }
            _json_dump(secrets_path, payload)

    def load_voices(self, data_dir: Path) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        voices_path = data_dir / "voice_profiles.json"
        voices_audio_dir = data_dir / "voices"
        with self.file_lock:
            payload = _json_load(voices_path)
            if isinstance(payload, dict):
                for key, item in payload.items():
                    if isinstance(item, dict) and "voice_id" in item:
                        audio_path = voices_audio_dir / f"{item['voice_id']}.wav"
                        audio_bytes = audio_path.read_bytes() if audio_path.exists() else None
                        self.voice_profiles[item["voice_id"]] = VoiceProfileRecord(
                            voice_id=item["voice_id"],
                            name=item["name"],
                            source=item["source"],
                            created_at=datetime.fromisoformat(item["created_at"]),
                            audio_bytes=audio_bytes,
                            content_type=item.get("content_type", "audio/wav"),
                            filename=item.get("filename"),
                            ref_text=item.get("ref_text"),
                            consent=item.get("consent", False)
                        )

    def save_voices(self, data_dir: Path) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        voices_path = data_dir / "voice_profiles.json"
        voices_audio_dir = data_dir / "voices"
        voices_audio_dir.mkdir(exist_ok=True)
        with self.file_lock:
            payload = {}
            for record in self.voice_profiles.values():
                if record.audio_bytes:
                    audio_path = voices_audio_dir / f"{record.voice_id}.wav"
                    audio_path.write_bytes(record.audio_bytes)
                payload[record.voice_id] = {
                    "voice_id": record.voice_id,
                    "name": record.name,
                    "source": record.source,
                    "created_at": record.created_at.isoformat(),
                    "content_type": record.content_type,
                    "filename": record.filename,
                    "ref_text": record.ref_text,
                    "consent": record.consent
                }
            _json_dump(voices_path, payload)

    def queue_depth(self) -> int:
        return len(self.waiting_requests)

    def active_requests(self) -> int:
        return len(self.active_request_ids)

    def queue_position(self, job_id: str) -> int:
        try:
            return list(self.waiting_requests).index(job_id) + 1
        except ValueError:
            return 0

    def estimate_eta_ms(self, position: int, text_length: int) -> int:
        if position <= 0:
            return 0
        return position * (900 + min(text_length * 10, 2800))

    def prune_terminal_jobs(self, *, retention_days: int = 0, max_retained_jobs: int = 500) -> int:
        """Evict finished jobs so the in-memory store stays bounded.

        Removes terminal jobs (completed/failed/cancelled) older than retention_days
        and, beyond max_retained_jobs, the oldest terminal jobs regardless of age.
        Frees the heavy fields (final_audio WAV bytes, pcm_parts) and drops the
        request_states entry. Never touches queued/active jobs. Returns count removed.

        Callers must hold job_condition (or run before the worker starts); this only
        mutates plain dicts and does no awaiting.
        """
        terminal = {JobStatus.completed, JobStatus.failed, JobStatus.cancelled}
        removable = [(job_id, job) for job_id, job in self.jobs.items() if job.status in terminal]
        to_remove: set[str] = set()

        if retention_days and retention_days > 0:
            cutoff = utcnow() - timedelta(days=retention_days)
            for job_id, job in removable:
                stamp = job.completed_at or job.updated_at or job.created_at
                if stamp and stamp < cutoff:
                    to_remove.add(job_id)

        remaining = [(job_id, job) for job_id, job in removable if job_id not in to_remove]
        if max_retained_jobs >= 0 and len(remaining) > max_retained_jobs:
            remaining.sort(key=lambda kv: kv[1].completed_at or kv[1].updated_at or kv[1].created_at)
            overflow = len(remaining) - max_retained_jobs
            for job_id, _ in remaining[:overflow]:
                to_remove.add(job_id)

        for job_id in to_remove:
            job = self.jobs.pop(job_id, None)
            if job is not None:
                job.final_audio = None
                job.pcm_parts = []
            self.request_states.pop(job_id, None)
        return len(to_remove)
