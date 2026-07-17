from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..passwords import build_pbkdf2_hash, verify_pbkdf2_hash
from .models import JobStatus, SpeechRequest

# --- auth configuration ---
SESSION_TTL_SECONDS = 7 * 24 * 3600  # 7 days
DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "admin"
API_KEY_PREFIX = "omnivoice_tts"
SECRETS_VERSION = 2


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

def hash_token(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()

def generate_api_key() -> str:
    return f'{API_KEY_PREFIX}_{secrets.token_urlsafe(24)}'

def _parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except (ValueError, TypeError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


@dataclass
class ApiKeyRecord:
    key_id: str
    name: str
    key_hash: str
    created_at: datetime
    last_used_at: datetime | None = None
    disabled: bool = False
    alias: str = ""
    total_seconds_processed: float = 0.0
    request_count: int = 0


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
    api_key_id: str | None = None

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
        self.users: dict[str, dict[str, Any]] = {}
        self.sessions: dict[str, dict[str, Any]] = {}
        self.data_dir: Path | None = None
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
        # Finetune data-generation + training runs (kept in-memory like benchmark_runs;
        # the heavy artifacts live on disk under data/finetune/, not here).
        self.finetune_runs: dict[str, Any] = {}
        self.finetune_train_runs: dict[str, Any] = {}
        self.model_download_jobs: dict[tuple[str, str], dict[str, Any]] = {}
        self.model_download_lock = threading.RLock()
        self.file_lock = threading.RLock()

    def _load_api_key(self, item: dict[str, Any]) -> None:
        usage = item.get("usage") if isinstance(item.get("usage"), dict) else {}
        self.api_keys[item["key_id"]] = ApiKeyRecord(
            key_id=item["key_id"],
            name=item.get("name", "client"),
            key_hash=item["key_hash"],
            created_at=_parse_iso(item.get("created_at")) or utcnow(),
            last_used_at=_parse_iso(usage.get("last_used_at")) or _parse_iso(item.get("last_used_at")),
            disabled=item.get("disabled", False),
            alias=item.get("alias") or item.get("name") or "",
            total_seconds_processed=float(usage.get("total_seconds_processed") or 0.0),
            request_count=int(usage.get("request_count") or 0),
        )

    @staticmethod
    def _load_user(username: str, item: dict[str, Any]) -> dict[str, Any]:
        return {
            "username": item.get("username") or username,
            "password_hash": item.get("password_hash", ""),
            "must_change_password": bool(item.get("must_change_password", False)),
            "created_at": item.get("created_at"),
            "updated_at": item.get("updated_at"),
            "last_login_at": item.get("last_login_at"),
        }

    def load_secrets(self, data_dir: Path) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = data_dir
        secrets_path = data_dir / "server_secrets.json"
        with self.file_lock:
            payload = _json_load(secrets_path)
            if not isinstance(payload, dict):
                payload = {}
            self.users = {}
            self.sessions = {}
            self.api_keys = {}

            if "version" not in payload:
                # Legacy flat {key_id: ApiKeyRecord}: drop the implicit admin record (its
                # token is obsolete; admin is now a user) and keep the rest as client keys.
                for item in payload.values():
                    if isinstance(item, dict) and "key_id" in item and item.get("name") != "admin":
                        self._load_api_key(item)
            else:
                for username, item in (payload.get("users") or {}).items():
                    if isinstance(item, dict):
                        self.users[username] = self._load_user(username, item)
                for token_hash, item in (payload.get("sessions") or {}).items():
                    if isinstance(item, dict):
                        self.sessions[token_hash] = {
                            "username": item.get("username"),
                            "created_at": item.get("created_at"),
                            "expires_at": item.get("expires_at"),
                            "last_seen_at": item.get("last_seen_at"),
                        }
                for item in (payload.get("api_keys") or {}).values():
                    if isinstance(item, dict) and "key_id" in item:
                        self._load_api_key(item)

            if not self.users:
                self._seed_default_admin()
            self._prune_sessions()
            self.save_secrets(data_dir)

    def save_secrets(self, data_dir: Path) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        secrets_path = data_dir / "server_secrets.json"
        with self.file_lock:
            payload = {
                "version": SECRETS_VERSION,
                "users": dict(self.users),
                "sessions": dict(self.sessions),
                "api_keys": {
                    rec.key_id: {
                        "key_id": rec.key_id,
                        "name": rec.name,
                        "key_hash": rec.key_hash,
                        "alias": rec.alias,
                        "created_at": rec.created_at.isoformat(),
                        "usage": {
                            "total_seconds_processed": round(float(rec.total_seconds_processed), 3),
                            "request_count": int(rec.request_count),
                            "last_used_at": rec.last_used_at.isoformat() if rec.last_used_at else None,
                        },
                    }
                    for rec in self.api_keys.values()
                },
            }
            _json_dump(secrets_path, payload)

    # ---- auth: users, sessions, client api keys ----
    def _persist(self) -> None:
        if self.data_dir is not None:
            self.save_secrets(self.data_dir)

    @staticmethod
    def _make_user(username: str, password: str, *, must_change: bool) -> dict[str, Any]:
        stamp = utcnow().isoformat()
        return {
            "username": username,
            "password_hash": build_pbkdf2_hash(password),
            "must_change_password": must_change,
            "created_at": stamp,
            "updated_at": stamp,
            "last_login_at": None,
        }

    def _seed_default_admin(self) -> None:
        self.users[DEFAULT_ADMIN_USERNAME] = self._make_user(
            DEFAULT_ADMIN_USERNAME, DEFAULT_ADMIN_PASSWORD, must_change=True
        )

    def _prune_sessions(self) -> None:
        now = utcnow()
        expired = [
            th for th, s in self.sessions.items()
            if (_parse_iso(s.get("expires_at")) is not None and now > _parse_iso(s.get("expires_at")))
        ]
        for th in expired:
            self.sessions.pop(th, None)

    def verify_user(self, username: str, password: str) -> dict[str, Any] | None:
        username = (username or "").strip().lower()
        if not username or not password:
            return None
        with self.file_lock:
            user = self.users.get(username)
            if not user or not verify_pbkdf2_hash(password, user.get("password_hash", "")):
                return None
            return dict(user)

    def touch_login(self, username: str) -> None:
        with self.file_lock:
            user = self.users.get(username)
            if user:
                user["last_login_at"] = utcnow().isoformat()
                self._persist()

    def set_password(self, username: str, new_password: str) -> bool:
        username = (username or "").strip().lower()
        with self.file_lock:
            user = self.users.get(username)
            if not user:
                return False
            user["password_hash"] = build_pbkdf2_hash(new_password)
            user["must_change_password"] = False
            user["updated_at"] = utcnow().isoformat()
            for th in [t for t, s in self.sessions.items() if s.get("username") == username]:
                self.sessions.pop(th, None)
            self._persist()
            return True

    def create_session(self, username: str) -> str:
        raw = secrets.token_urlsafe(32)
        with self.file_lock:
            self.sessions[hash_token(raw)] = {
                "username": username,
                "created_at": utcnow().isoformat(),
                "expires_at": (utcnow() + timedelta(seconds=SESSION_TTL_SECONDS)).isoformat(),
                "last_seen_at": utcnow().isoformat(),
            }
            self._persist()
        return raw

    def get_session(self, raw_token: str | None) -> dict[str, Any] | None:
        if not raw_token:
            return None
        token_hash = hash_token(raw_token)
        with self.file_lock:
            session = self.sessions.get(token_hash)
            if not session:
                return None
            expires = _parse_iso(session.get("expires_at"))
            if expires is not None and utcnow() > expires:
                self.sessions.pop(token_hash, None)
                self._persist()
                return None
            user = self.users.get(session.get("username"))
            if not user:
                return None
            return {"username": user["username"], "must_change_password": bool(user.get("must_change_password"))}

    def delete_session(self, raw_token: str | None) -> None:
        if not raw_token:
            return
        with self.file_lock:
            if self.sessions.pop(hash_token(raw_token), None) is not None:
                self._persist()

    def list_api_keys(self) -> list[dict[str, Any]]:
        with self.file_lock:
            out = [
                {
                    "id": rec.key_id,
                    "alias": rec.alias or rec.name,
                    "created_at": rec.created_at.isoformat(),
                    "usage": {
                        "total_seconds_processed": round(float(rec.total_seconds_processed), 3),
                        "request_count": int(rec.request_count),
                        "last_used_at": rec.last_used_at.isoformat() if rec.last_used_at else None,
                    },
                }
                for rec in self.api_keys.values()
            ]
        out.sort(key=lambda r: r["created_at"])
        return out

    def has_api_keys(self) -> bool:
        return len(self.api_keys) > 0

    def create_api_key(self, alias: str) -> dict[str, Any]:
        alias = (alias or "").strip() or "Unnamed key"
        raw = generate_api_key()
        key_id = new_id("key")
        created = utcnow()
        with self.file_lock:
            self.api_keys[key_id] = ApiKeyRecord(
                key_id=key_id,
                name="client",
                key_hash=hash_token(raw),
                created_at=created,
                alias=alias,
            )
            self._persist()
        return {"id": key_id, "alias": alias, "created_at": created.isoformat(), "token": raw}

    def delete_api_key(self, key_id: str) -> bool:
        with self.file_lock:
            removed = self.api_keys.pop(key_id, None) is not None
            if removed:
                self._persist()
            return removed

    def match_api_key(self, raw_key: str | None) -> str | None:
        if not raw_key:
            return None
        provided = hash_token(raw_key.strip())
        with self.file_lock:
            for rec in self.api_keys.values():
                if not rec.disabled and secrets.compare_digest(rec.key_hash or "", provided):
                    return rec.key_id
        return None

    def record_api_key_usage(self, key_id: str | None, audio_seconds: float) -> None:
        if not key_id:
            return
        with self.file_lock:
            rec = self.api_keys.get(key_id)
            if not rec:
                return
            rec.total_seconds_processed = round(float(rec.total_seconds_processed) + float(audio_seconds or 0.0), 3)
            rec.request_count = int(rec.request_count) + 1
            rec.last_used_at = utcnow()
            self._persist()

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
