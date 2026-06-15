"""Pydantic request/response models for the finetune admin endpoints.

Kept in this subpackage (not domain/models.py) so the finetune feature stays
self-contained and the already-large shared models module does not grow further.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# --- Domains ---------------------------------------------------------------

class DomainItem(BaseModel):
    domain_id: str
    name: str
    description: str = ''
    created_at: str | None = None
    sentence_count: int = 0
    sentences: list[str] | None = None  # only populated on detail requests


class DomainListResponse(BaseModel):
    domains: list[DomainItem] = Field(default_factory=list)


class DomainCreateRequest(BaseModel):
    name: str
    description: str = ''


class DomainUpdateRequest(BaseModel):
    name: str | None = None
    description: str | None = None


class DomainGenerateRequest(BaseModel):
    """Ask the text LLM to invent N new domain topics, excluding the ones we already have."""
    count: int = Field(default=20, ge=1, le=200)
    language: str = 'Deutsch'
    vllm_base_url: str | None = None
    vllm_model: str | None = None


class DomainGenerateResponse(BaseModel):
    created: list[DomainItem] = Field(default_factory=list)
    skipped_duplicates: int = 0


class SentenceGenerateRequest(BaseModel):
    """Generate N new sentences for one domain, deduped against that domain's existing set."""
    count: int = Field(default=50, ge=1, le=2000)
    language: str = 'Deutsch'
    min_words: int = Field(default=5, ge=1, le=80)
    max_words: int = Field(default=16, ge=1, le=120)
    vllm_base_url: str | None = None
    vllm_model: str | None = None


class SentenceGenerateResponse(BaseModel):
    domain_id: str
    added: int = 0
    skipped_duplicates: int = 0
    sentence_count: int = 0
    sample: list[str] = Field(default_factory=list)


# --- Data generation runs --------------------------------------------------

class DatagenStartRequest(BaseModel):
    # What to say
    domain_ids: list[str] = Field(default_factory=list)
    max_sentences_per_domain: int = Field(default=0, ge=0, le=5000)  # 0 = all stored
    language: str = 'Deutsch'

    # Who says it
    voice_mode: str = Field(default='clone', pattern='^(clone|auto|both)$')
    voice_ids: list[str] = Field(default_factory=list)  # required when mode includes clone

    # Quality gate
    wer_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    max_attempts: int = Field(default=10, ge=1, le=50)

    # Throughput
    tts_concurrency: int = Field(default=4, ge=1, le=64)
    transcription_concurrency: int = Field(default=8, ge=1, le=64)

    # Endpoints / determinism (fall back to server settings when omitted)
    vllm_base_url: str | None = None
    vllm_model: str | None = None
    whisper_base_url: str | None = None
    tolerance_letters_per_word: int = Field(default=0, ge=0, le=8)
    base_seed: int | None = Field(default=None, ge=0, le=2_147_483_647)
    completion_timeout_seconds: int = Field(default=180, ge=1, le=3600)
    exclusive: bool = False


class DatagenVoiceProgress(BaseModel):
    voice: str
    planned: int = 0
    accepted: int = 0
    rejected: int = 0
    attempts: int = 0


class DatagenRunResponse(BaseModel):
    run_id: str
    status: str  # running | completed | failed | cancelled
    phase: str = 'generating'
    created_at: datetime
    completed_at: datetime | None = None
    planned: int = 0
    accepted: int = 0
    rejected: int = 0
    attempts: int = 0
    pct: float = 0.0
    current: str | None = None
    voices: list[DatagenVoiceProgress] = Field(default_factory=list)
    wer_threshold: float = 0.0
    max_attempts: int = 10
    error_message: str | None = None


# --- Human-eval clip browser ----------------------------------------------

class ClipItem(BaseModel):
    clip_id: str
    voice: str
    text: str
    wer: float | None = None
    filename: str
    size_bytes: int = 0
    created_at: str | None = None


class ClipListResponse(BaseModel):
    voices: list[str] = Field(default_factory=list)
    total: int = 0
    clips: list[ClipItem] = Field(default_factory=list)


class ClipDeleteResponse(BaseModel):
    ok: bool = True
    removed: list[str] = Field(default_factory=list)


class DatasetVoiceSummary(BaseModel):
    voice: str
    clips: int = 0
    seconds: float = 0.0


class DatasetSummaryResponse(BaseModel):
    voices: list[DatasetVoiceSummary] = Field(default_factory=list)
    total_clips: int = 0


# --- Training (phase 2) ----------------------------------------------------

class TrainStartRequest(BaseModel):
    name: str = Field(default='omnivoice-finetune')
    voices: list[str] = Field(default_factory=list)  # folder names; empty = all under train/
    dev_fraction: float = Field(default=0.05, ge=0.0, le=0.5)
    epochs: int = Field(default=3, ge=1, le=100)
    steps_override: int = Field(default=0, ge=0, le=1_000_000)  # >0 wins over epochs estimate
    learning_rate: float = Field(default=3e-5, gt=0.0, le=1e-2)
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=0.5)
    weight_decay: float = Field(default=0.01, ge=0.0, le=1.0)
    batch_tokens: int = Field(default=8192, ge=512, le=131072)
    max_batch_size: int = Field(default=16, ge=1, le=256)
    gradient_accumulation_steps: int = Field(default=4, ge=1, le=256)
    attn_implementation: str = Field(default='sdpa', pattern='^(sdpa|flex_attention)$')
    mixed_precision: str = Field(default='bf16', pattern='^(bf16|fp16|no)$')
    keep_last_n_checkpoints: int = Field(default=2, ge=1, le=20)
    seed: int = Field(default=42, ge=0, le=2_147_483_647)


class TrainRunResponse(BaseModel):
    run_id: str
    status: str  # queued | preprocessing | training | completed | failed | cancelled
    phase: str = ''
    created_at: datetime
    completed_at: datetime | None = None
    base_model: str | None = None
    dataset_dir: str | None = None
    output_dir: str | None = None
    train_count: int = 0
    dev_count: int = 0
    total_steps: int = 0
    current_step: int = 0
    loss: float | None = None
    eval_loss: float | None = None
    lr: float | None = None
    steps_per_sec: float | None = None
    eta_ms: int | None = None
    pct: float = 0.0
    loss_curve: list[float] = Field(default_factory=list)
    checkpoint_dir: str | None = None
    log_tail: list[str] = Field(default_factory=list)
    error_message: str | None = None


class CheckpointItem(BaseModel):
    checkpoint_id: str
    name: str
    model_id: str
    dirname: str
    base_model: str | None = None
    steps: int = 0
    run_id: str | None = None
    created_at: str | None = None
    exists: bool = True


class CheckpointListResponse(BaseModel):
    checkpoints: list[CheckpointItem] = Field(default_factory=list)


class PromoteRequest(BaseModel):
    name: str  # human name -> models/Custom-<slug>/


# --- shared error envelope (matches existing FastAPI {detail} shape on raise) ---

class OkResponse(BaseModel):
    ok: bool = True
    detail: str | None = None
    extra: dict[str, Any] | None = None
