"""FinetuneService: LLM sentence/domain generation + the self-generation loop.

Reuses the existing data-gen primitives instead of reinventing them:
- text LLM (vLLM) call shape + robust JSON-array parsing from WerBenchmarkService
- TTS via QueueService.submit (so real OmniVoice batching/capacity applies)
- ASR via TranscriptionService.transcribe_with_base_url
- WER scoring via WerBenchmarkService._wer (custom Levenshtein with tolerance)

Accepted clips land on disk under data/finetune/train/<voice>/ (see storage.py).
"""
from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

import httpx

from ..config import Settings
from ..domain.models import JobStatus, SpeechRequest, TaskType
from ..domain.state import InMemoryStore, utcnow
from ..runtime_v2 import (
    LANGUAGE_LABEL_TO_CODE,
    OMNIVOICE_AUTO_ALIAS,
    OMNIVOICE_BASE_ALIAS,
)
from ..services_v2 import EventHub, QueueService, TranscriptionService, WerBenchmarkService, spawn_tracked_task
from . import domains as domains_store
from . import storage
from .schemas import (
    ClipListResponse,
    DatagenRunResponse,
    DatagenStartRequest,
    DatagenVoiceProgress,
    DatasetSummaryResponse,
    DatasetVoiceSummary,
    DomainGenerateRequest,
    DomainGenerateResponse,
    DomainItem,
    SentenceGenerateRequest,
    SentenceGenerateResponse,
)

logger = logging.getLogger('omnivoice_tts_server.finetune')

_SEED_MOD = 2_147_483_647


def _stable_seed(*parts: str) -> int:
    import hashlib

    digest = hashlib.sha1('|'.join(parts).encode('utf-8')).hexdigest()
    return int(digest[:8], 16) % _SEED_MOD


class _Target:
    __slots__ = ('voice_label', 'mode', 'voice_id', 'ref_text', 'model', 'task_type', 'sample_bytes',
                 'domain_id', 'sentence', 'language_id', 'attempts')

    def __init__(self, *, voice_label, mode, voice_id, ref_text, model, task_type, sample_bytes,
                 domain_id, sentence, language_id):
        self.voice_label = voice_label
        self.mode = mode
        self.voice_id = voice_id
        self.ref_text = ref_text
        self.model = model
        self.task_type = task_type
        self.sample_bytes = sample_bytes
        self.domain_id = domain_id
        self.sentence = sentence
        self.language_id = language_id
        self.attempts = 0

    def seed_for(self, base_seed: int, attempt: int) -> int:
        return (base_seed + _stable_seed(self.voice_label, self.sentence) + attempt) % _SEED_MOD


class FinetuneService:
    def __init__(
        self,
        store: InMemoryStore,
        queue_service: QueueService,
        transcription_service: TranscriptionService,
        wer_service: WerBenchmarkService,
        events: EventHub,
        settings: Settings,
    ) -> None:
        self.store = store
        self.queue_service = queue_service
        self.transcription_service = transcription_service
        self.wer_service = wer_service
        self.events = events
        self.settings = settings

    # --- vLLM helpers ------------------------------------------------------

    def _resolve_vllm_base(self, override: str | None) -> str:
        return WerBenchmarkService._normalize_base_url(override or self.settings.vllm_base_url)

    async def _resolve_vllm_model(self, base_url: str, override: str | None) -> str:
        model = (override or self.settings.vllm_model or '').strip()
        if model:
            return model
        return await self.wer_service._resolve_vllm_model(base_url)

    async def _chat(self, client: httpx.AsyncClient, base_url: str, model: str, system: str, user: str, *, max_tokens: int) -> str:
        response = await client.post(
            f'{base_url}/v1/chat/completions',
            json={
                'model': model,
                'messages': [
                    {'role': 'system', 'content': system},
                    {'role': 'user', 'content': user},
                ],
                'temperature': 0.9,
                'top_p': 0.95,
                'max_tokens': max_tokens,
                'chat_template_kwargs': {'enable_thinking': False},
            },
        )
        response.raise_for_status()
        data = response.json()
        return str(data.get('choices', [{}])[0].get('message', {}).get('content') or '')

    @property
    def _is_mock(self) -> bool:
        return self.settings.runtime_backend.lower() == 'mock'

    # --- Domain CRUD -------------------------------------------------------

    def list_domains(self) -> list[DomainItem]:
        return [_domain_item(entry, include_sentences=False) for entry in domains_store.list_domains(self.settings.data_dir)]

    def get_domain_detail(self, domain_id: str) -> DomainItem:
        entry = domains_store.get_domain(self.settings.data_dir, domain_id)
        if entry is None:
            raise KeyError(domain_id)
        return _domain_item(entry, include_sentences=True)

    def create_domain(self, name: str, description: str) -> DomainItem:
        entry = domains_store.create_domain(self.settings.data_dir, name, description, now_iso=utcnow().isoformat())
        return _domain_item(entry, include_sentences=False)

    def update_domain(self, domain_id: str, name: str | None, description: str | None) -> DomainItem:
        entry = domains_store.update_domain(self.settings.data_dir, domain_id, name=name, description=description)
        return _domain_item(entry, include_sentences=False)

    def delete_domain(self, domain_id: str) -> bool:
        return domains_store.delete_domain(self.settings.data_dir, domain_id)

    # --- Domain generation -------------------------------------------------

    async def generate_domains(self, req: DomainGenerateRequest) -> DomainGenerateResponse:
        data_dir = self.settings.data_dir
        existing = [entry['name'] for entry in domains_store.list_domains(data_dir)]
        existing_lower = {name.strip().lower() for name in existing}

        if self._is_mock or (req.vllm_base_url or self.settings.vllm_base_url).strip().lower() == 'mock':
            candidates = [
                {'name': f'Testdomäne {i}', 'description': f'Automatisch erzeugte Beispieldomäne {i}.'}
                for i in range(1, req.count + 1)
            ]
        else:
            base_url = self._resolve_vllm_base(req.vllm_base_url)
            model = await self._resolve_vllm_model(base_url, req.vllm_model)
            candidates = await self._generate_domain_objects(base_url, model, req, existing)

        created: list[DomainItem] = []
        skipped = 0
        now = utcnow().isoformat()
        for cand in candidates:
            name = str(cand.get('name') or '').strip()
            if not name or name.lower() in existing_lower:
                skipped += 1
                continue
            try:
                entry = domains_store.create_domain(
                    data_dir, name, str(cand.get('description') or ''), now_iso=now
                )
            except ValueError:
                skipped += 1
                continue
            existing_lower.add(name.lower())
            created.append(_domain_item(entry, include_sentences=False))
            if len(created) >= req.count:
                break
        return DomainGenerateResponse(created=created, skipped_duplicates=skipped)

    async def _generate_domain_objects(
        self, base_url: str, model: str, req: DomainGenerateRequest, existing: list[str]
    ) -> list[dict[str, Any]]:
        exclude = ', '.join(existing[-60:]) if existing else '(noch keine)'
        system = 'Du entwirfst kurze Themen/Domänen für TTS-Trainingsdaten und antwortest nur mit JSON.'
        user = (
            f'Erzeuge genau {req.count} neue, voneinander verschiedene Themen/Domänen für '
            f'{req.language}-TTS-Trainingsdaten. Eine Domäne beschreibt, worum die Sätze handeln '
            'sollen — besonders Fälle, an denen ein TTS-Modell oft scheitert: Abkürzungen '
            '(z. B. GmbH, AG, z. B.), Zahlen/Daten/Uhrzeiten, Eigennamen, Fremdwörter, Fachbegriffe. '
            f'Vermeide diese bereits vorhandenen Domänen: {exclude}. '
            'Antworte ausschließlich mit einem JSON-Array von Objekten der Form '
            '{"name": "kurzer Titel", "description": "1 Satz, worum es geht"}.'
        )
        async with httpx.AsyncClient(timeout=90, trust_env=False) as client:
            content = await self._chat(client, base_url, model, system, user, max_tokens=min(4096, 64 + req.count * 48))
        return _parse_domain_objects(content)

    # --- Sentence generation ----------------------------------------------

    async def generate_sentences(self, domain_id: str, req: SentenceGenerateRequest) -> SentenceGenerateResponse:
        data_dir = self.settings.data_dir
        domain = domains_store.get_domain(data_dir, domain_id)
        if domain is None:
            raise KeyError(domain_id)
        existing = list(domain.get('sentences') or [])
        seen = {domains_store.normalize_sentence_key(s) for s in existing}

        collected: list[str] = []
        if self._is_mock or (req.vllm_base_url or self.settings.vllm_base_url).strip().lower() == 'mock':
            for i in range(req.count):
                collected.append(f'{domain["name"]}: Beispielsatz Nummer {len(existing) + i + 1} mit klaren Woertern.')
        else:
            base_url = self._resolve_vllm_base(req.vllm_base_url)
            model = await self._resolve_vllm_model(base_url, req.vllm_model)
            async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
                chunk_index = 1
                while len(collected) < req.count:
                    missing = req.count - len(collected)
                    chunk = min(20, missing)
                    candidates = await self._generate_sentence_chunk(
                        client, base_url, model, domain, req, chunk, chunk_index, existing + collected
                    )
                    progressed = False
                    for sentence in candidates:
                        key = domains_store.normalize_sentence_key(sentence)
                        if not key or key in seen:
                            continue
                        seen.add(key)
                        collected.append(sentence)
                        progressed = True
                        if len(collected) >= req.count:
                            break
                    chunk_index += 1
                    if not candidates or not progressed:
                        break

        added, skipped, total = domains_store.add_sentences(data_dir, domain_id, collected)
        refreshed = domains_store.get_domain(data_dir, domain_id) or domain
        return SentenceGenerateResponse(
            domain_id=domain_id,
            added=added,
            skipped_duplicates=skipped,
            sentence_count=total,
            sample=list(refreshed.get('sentences') or [])[-min(10, total):],
        )

    async def _generate_sentence_chunk(
        self, client, base_url, model, domain, req: SentenceGenerateRequest,
        chunk_count: int, chunk_index: int, existing: list[str],
    ) -> list[str]:
        exclude = ' | '.join(existing[-40:]) if existing else ''
        system = 'Du erzeugst saubere TTS-Testsätze und antwortest nur mit JSON.'
        user = (
            f'Thema/Domäne: "{domain["name"]}". Kontext: {domain.get("description") or "keiner"}. '
            f'Erzeuge genau {chunk_count} neue, natürliche {req.language}-Sätze zu diesem Thema. '
            f'Jeder Satz {req.min_words}-{req.max_words} Wörter, ohne Nummerierung, ohne Listenmarker, '
            'ohne Abkürzungspunkte am Satzende, keine chinesischen Zeichen, keine Mischsprache. '
            'Zahlen als Ziffern schreiben (z. B. 3 Uhr statt drei Uhr). '
            + (f'Vermeide diese bereits vorhandenen Sätze: {exclude}. ' if exclude else '')
            + f'Chunk {chunk_index}. Antworte ausschließlich mit einem JSON-Array von Strings.'
        )
        content = await self._chat(
            client, base_url, model, system, user,
            max_tokens=min(2048, max(256, chunk_count * max(req.max_words, 4) * 8)),
        )
        return WerBenchmarkService._parse_generated_sentences(content, chunk_count)

    # --- Data generation run ----------------------------------------------

    def _build_targets(self, req: DatagenStartRequest) -> list[_Target]:
        data_dir = self.settings.data_dir
        # Collect sentences per selected domain.
        sentence_items: list[tuple[str, str]] = []  # (domain_id, sentence)
        for domain_id in req.domain_ids:
            domain = domains_store.get_domain(data_dir, domain_id)
            if domain is None:
                continue
            sentences = list(domain.get('sentences') or [])
            if req.max_sentences_per_domain > 0:
                sentences = sentences[: req.max_sentences_per_domain]
            for sentence in sentences:
                sentence_items.append((domain_id, sentence))

        language_id = LANGUAGE_LABEL_TO_CODE.get((req.language or '').strip().lower())

        # Build the voice specs for this run.
        voice_specs: list[dict[str, Any]] = []
        modes = ['clone', 'auto'] if req.voice_mode == 'both' else [req.voice_mode]
        if 'clone' in modes:
            for voice_id in req.voice_ids:
                profile = self.store.voice_profiles.get(voice_id)
                if profile is None:
                    continue
                voice_specs.append({
                    'label': profile.name,
                    'mode': 'clone',
                    'voice_id': profile.voice_id,
                    'ref_text': profile.ref_text,
                    'model': OMNIVOICE_BASE_ALIAS,
                    'task_type': TaskType.base,
                    'sample_bytes': profile.audio_bytes,
                })
        if 'auto' in modes:
            voice_specs.append({
                'label': 'auto',
                'mode': 'auto',
                'voice_id': None,
                'ref_text': None,
                'model': OMNIVOICE_AUTO_ALIAS,
                'task_type': TaskType.custom_voice,
                'sample_bytes': None,
            })

        targets: list[_Target] = []
        for spec in voice_specs:
            for domain_id, sentence in sentence_items:
                targets.append(_Target(
                    voice_label=spec['label'], mode=spec['mode'], voice_id=spec['voice_id'],
                    ref_text=spec['ref_text'], model=spec['model'], task_type=spec['task_type'],
                    sample_bytes=spec['sample_bytes'], domain_id=domain_id, sentence=sentence,
                    language_id=language_id,
                ))
        return targets

    async def start_datagen(self, req: DatagenStartRequest) -> DatagenRunResponse:
        from ..domain.state import new_id

        run_id = new_id('ftgen')
        targets = self._build_targets(req)
        if not targets:
            raise ValueError('No sentences to generate. Select domains with stored sentences and at least one voice.')

        voices: dict[str, dict[str, int]] = {}
        for target in targets:
            voices.setdefault(target.voice_label, {'planned': 0, 'accepted': 0, 'rejected': 0, 'attempts': 0})
            voices[target.voice_label]['planned'] += 1

        run: dict[str, Any] = {
            'run_id': run_id,
            'status': 'running',
            'phase': 'generating',
            'created_at': utcnow(),
            'completed_at': None,
            'planned': len(targets),
            'accepted': 0,
            'rejected': 0,
            'attempts': 0,
            'current': None,
            'cancelled': False,
            'error_message': None,
            'wer_threshold': req.wer_threshold,
            'max_attempts': req.max_attempts,
            'voices': voices,
        }
        # Only keep the latest run to bound memory (mirrors the WER service).
        self.store.finetune_runs.clear()
        self.store.finetune_runs[run_id] = run
        base_seed = req.base_seed if req.base_seed is not None else random.Random().randint(0, _SEED_MOD)
        spawn_tracked_task(self.store, self._execute_datagen(run, req, targets, base_seed), label=f'finetune-gen:{run_id}')
        await self.events.publish('dashboard.snapshot', {'reason': 'finetune.started', 'run_id': run_id})
        return _datagen_response(run)

    async def _execute_datagen(self, run: dict[str, Any], req: DatagenStartRequest, targets: list[_Target], base_seed: int) -> None:
        tts_sem = asyncio.Semaphore(max(1, req.tts_concurrency))
        asr_sem = asyncio.Semaphore(max(1, req.transcription_concurrency))
        whisper_base = (req.whisper_base_url or self.settings.whisper_base_url or '').strip()
        data_dir = self.settings.data_dir
        sample_written: set[str] = set()

        async def run_body() -> None:
            pending = list(targets)
            for attempt in range(req.max_attempts):
                if run['cancelled'] or not pending:
                    break
                run['phase'] = f'attempt {attempt + 1}/{req.max_attempts}'
                results = await asyncio.gather(
                    *(self._attempt_target(t, attempt, base_seed, req, whisper_base, tts_sem, asr_sem) for t in pending)
                )
                next_pending: list[_Target] = []
                for target, verdict in zip(pending, results):
                    run['attempts'] += 1
                    run['voices'][target.voice_label]['attempts'] += 1
                    if run['cancelled']:
                        break
                    if verdict['accepted']:
                        self._save_accepted(data_dir, target, verdict, sample_written)
                        run['accepted'] += 1
                        run['voices'][target.voice_label]['accepted'] += 1
                    else:
                        next_pending.append(target)
                    run['current'] = f'{target.voice_label}: {target.sentence[:60]}'
                pending = next_pending
                await self._publish(run)
            # Whatever never passed within max_attempts counts as rejected (skip on cancel —
            # those targets were simply never finished, not quality-rejected).
            if not run['cancelled']:
                for target in pending:
                    run['rejected'] += 1
                    run['voices'][target.voice_label]['rejected'] += 1

        try:
            if req.exclusive:
                async with self.store.exclusive_lock:
                    await run_body()
            else:
                await run_body()
            run['status'] = 'cancelled' if run['cancelled'] else 'completed'
        except Exception as exc:  # noqa: BLE001 - surface to UI, log for ops
            run['status'] = 'failed'
            run['error_message'] = str(exc)
            logger.error('finetune datagen run %s failed: %s', run['run_id'], exc, exc_info=exc)
        finally:
            run['phase'] = 'done'
            run['current'] = None
            run['completed_at'] = utcnow()
            await self._publish(run)

    async def _attempt_target(
        self, target: _Target, attempt: int, base_seed: int, req: DatagenStartRequest,
        whisper_base: str, tts_sem: asyncio.Semaphore, asr_sem: asyncio.Semaphore,
    ) -> dict[str, Any]:
        seed = target.seed_for(base_seed, attempt)
        try:
            async with tts_sem:
                audio, content_type = await self._synthesize(target, seed, req.completion_timeout_seconds)
            async with asr_sem:
                transcript = await self._transcribe(audio, content_type, whisper_base, req.completion_timeout_seconds, target.sentence)
            wer = WerBenchmarkService._wer(target.sentence, transcript, req.tolerance_letters_per_word)
            accepted = wer['wer'] <= req.wer_threshold
            return {'accepted': accepted, 'audio': audio, 'wer': wer['wer'], 'transcript': transcript, 'seed': seed}
        except Exception as exc:  # noqa: BLE001 - a single failed attempt should not kill the run
            logger.debug('attempt failed (%s / seed %s): %s', target.voice_label, seed, exc)
            return {'accepted': False, 'audio': None, 'wer': None, 'transcript': None, 'seed': seed, 'error': str(exc)}

    async def _synthesize(self, target: _Target, seed: int, timeout_s: int) -> tuple[bytes, str]:
        request = SpeechRequest(
            input=target.sentence,
            model=target.model,
            voice=target.voice_id,
            task_type=target.task_type,
            language=target.language_id,
            ref_text=target.ref_text if target.task_type == TaskType.base else None,
            stream=False,
            response_format='wav',
            seed=seed,
        )
        job = await self.queue_service.submit(request, owner_scope='finetune')
        finished = await asyncio.wait_for(
            self.queue_service.wait_for_completion(job.job_id),
            timeout=max(1, int(timeout_s)),
        )
        if finished.status != JobStatus.completed or not finished.final_audio:
            raise RuntimeError(finished.error_message or 'Synthesis failed.')
        return finished.final_audio, finished.content_type or 'audio/wav'

    async def _transcribe(self, audio: bytes, content_type: str, whisper_base: str, timeout_s: int, fallback: str) -> str:
        if self._is_mock or not whisper_base or whisper_base.lower() == 'mock':
            return fallback
        transcription = await asyncio.wait_for(
            self.transcription_service.transcribe_with_base_url(
                'finetune.wav', content_type, audio,
                base_url=whisper_base, whisper_path=None, timeout_seconds=max(1, int(timeout_s)),
            ),
            timeout=max(1, int(timeout_s)),
        )
        return transcription.transcription

    def _save_accepted(self, data_dir, target: _Target, verdict: dict[str, Any], sample_written: set[str]) -> None:
        # Write the per-voice reference sample once.
        if target.voice_label not in sample_written:
            sample_bytes = target.sample_bytes or verdict.get('audio')
            if sample_bytes:
                storage.ensure_voice_sample(data_dir, target.voice_label, sample_bytes)
            sample_written.add(target.voice_label)
        meta = {'wer': verdict.get('wer'), 'seed': verdict.get('seed'), 'domain_id': target.domain_id}
        if target.language_id:
            meta['language_id'] = target.language_id
        storage.write_clip(data_dir, target.voice_label, wav_bytes=verdict['audio'], text=target.sentence, meta=meta)

    async def _publish(self, run: dict[str, Any]) -> None:
        await self.events.publish('dashboard.snapshot', {'reason': 'finetune.progress', 'run_id': run['run_id']})

    def get_run(self, run_id: str | None = None) -> DatagenRunResponse | None:
        if run_id:
            run = self.store.finetune_runs.get(run_id)
        else:
            run = next(iter(sorted(self.store.finetune_runs.values(), key=lambda r: r['created_at'], reverse=True)), None)
        return _datagen_response(run) if run else None

    def cancel_run(self, run_id: str) -> bool:
        run = self.store.finetune_runs.get(run_id)
        if not run:
            return False
        run['cancelled'] = True
        return True

    # --- Human-eval browser ------------------------------------------------

    def list_clips(self, voice: str | None = None) -> ClipListResponse:
        clips = storage.list_clips(self.settings.data_dir, voice=voice)
        voices = sorted({clip['voice'] for clip in storage.list_clips(self.settings.data_dir)})
        return ClipListResponse(voices=voices, total=len(clips), clips=clips)

    def delete_clip(self, clip_id: str) -> list[str]:
        return storage.delete_clip(self.settings.data_dir, clip_id)

    def clip_wav_path(self, clip_id: str):
        return storage.resolve_clip_wav(self.settings.data_dir, clip_id)

    def dataset_summary(self) -> DatasetSummaryResponse:
        rows = storage.dataset_summary(self.settings.data_dir)
        return DatasetSummaryResponse(
            voices=[DatasetVoiceSummary(**row) for row in rows],
            total_clips=sum(int(row['clips']) for row in rows),
        )


# --- module helpers --------------------------------------------------------

def _domain_item(entry: dict[str, Any], *, include_sentences: bool) -> DomainItem:
    sentences = list(entry.get('sentences') or [])
    return DomainItem(
        domain_id=entry['domain_id'],
        name=entry['name'],
        description=entry.get('description') or '',
        created_at=entry.get('created_at'),
        sentence_count=len(sentences),
        sentences=sentences if include_sentences else None,
    )


def _parse_domain_objects(content: str) -> list[dict[str, Any]]:
    import re

    cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.IGNORECASE | re.DOTALL).strip().strip('`')
    if cleaned.lower().startswith('json'):
        cleaned = cleaned[4:].strip()
    payload = WerBenchmarkService._parse_llm_container(cleaned)
    if payload is None:
        start, end = cleaned.find('['), cleaned.rfind(']')
        if start >= 0 and end > start:
            payload = WerBenchmarkService._parse_llm_container(cleaned[start:end + 1])
    objects: list[dict[str, Any]] = []
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict) and item.get('name'):
                objects.append({'name': str(item['name']), 'description': str(item.get('description') or '')})
            elif isinstance(item, str) and item.strip():
                objects.append({'name': item.strip(), 'description': ''})
    return objects


def _datagen_response(run: dict[str, Any]) -> DatagenRunResponse:
    planned = int(run.get('planned') or 0)
    done = int(run.get('accepted') or 0) + int(run.get('rejected') or 0)
    pct = (done / planned * 100.0) if planned else 0.0
    return DatagenRunResponse(
        run_id=run['run_id'],
        status=run['status'],
        phase=run.get('phase') or '',
        created_at=run['created_at'],
        completed_at=run.get('completed_at'),
        planned=planned,
        accepted=int(run.get('accepted') or 0),
        rejected=int(run.get('rejected') or 0),
        attempts=int(run.get('attempts') or 0),
        pct=round(pct, 1),
        current=run.get('current'),
        voices=[
            DatagenVoiceProgress(voice=label, **counts)
            for label, counts in run.get('voices', {}).items()
        ],
        wer_threshold=float(run.get('wer_threshold') or 0.0),
        max_attempts=int(run.get('max_attempts') or 0),
        error_message=run.get('error_message'),
    )
