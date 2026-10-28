"""TTS engine registry and the live engine switch.

Engines: 'omnivoice' (default), 'ming' (inclusionAI Ming-3.0-tts 0.5B), 'phonon' (Gradium Phonon) and
'mock' (test tone). `runtime_v2.build_synthesizer` builds the synthesizer for `settings.runtime_backend`;
this module lists the engines with their install state and swaps the running one without a restart:

1. new public requests wait (`EngineSwitcher.wait_ready`), queued jobs are held back from the old engine;
2. jobs already running finish on the old engine (up to `drain_timeout_s`, then they are cancelled);
3. the queue worker stops, the old synthesizer unloads and CUDA memory is released;
4. the settings the old engine rewrote (model list, active model, ...) get their OmniVoice values back,
   the new synthesizer is built and bound to the app, the queue and the finetune trainer;
5. the held jobs go back to the queue front, the worker restarts, the new engine preloads its model.

A failed build rolls back to the old synthesizer object (it reloads lazily on the next request).
"""
from __future__ import annotations

import asyncio
import gc
import importlib.util
import logging
import os
import sys
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import (
    RUNTIME_BACKEND_ENV,
    RUNTIME_BACKENDS,
    Settings,
    restore_engine_fields,
    save_runtime_settings,
    set_persisted_overrides,
    snapshot_engine_fields,
)
from .domain.models import EngineInfo, EngineSwitchStatus, JobStatus, SpeechRequest
from .domain.state import utcnow

logger = logging.getLogger('omnivoice_tts_server.engines')

DEFAULT_ACTIVE_MODEL = 'k2-fsa/OmniVoice-Base'
MING_REPO_ID = 'inclusionAI/Ming-3.0-tts-0.5B'
_TERMINAL_JOB = {JobStatus.completed, JobStatus.failed, JobStatus.cancelled}
_TERMINAL_RUN = {'completed', 'failed', 'cancelled', 'error', 'done'}
# Running jobs that ignore a cancel get this long before the engine is swapped anyway.
CANCEL_GRACE_S = 15.0


@dataclass(frozen=True)
class EngineSpec:
    id: str
    label: str
    description: str
    # Modules that must import for the engine to run (checked with importlib, nothing is imported).
    packages: tuple[str, ...]
    # Saved voice profiles (data/voice_profiles.json) are clone prompts for this engine.
    uses_voice_profiles: bool
    # Serves the OmniVoice model ids itself. Other engines rewrite supported_models/active_model, get
    # requests for foreign model ids mapped to their own default and never persist their own alias.
    native_model_ids: bool
    native_streaming: bool
    # Load the model right after start/switch so the first request is not cold.
    preload: bool
    test_only: bool = False


ENGINES: dict[str, EngineSpec] = {
    'omnivoice': EngineSpec(
        id='omnivoice',
        label='OmniVoice',
        description='k2-fsa OmniVoice (Standard). Klonen ueber Stimmprofile, AutoVoice, VoiceDesign, Finetuning.',
        packages=('omnivoice',),
        uses_voice_profiles=True,
        native_model_ids=True,
        native_streaming=False,
        preload=True,
    ),
    'ming': EngineSpec(
        id='ming',
        label='Ming-3.0-tts 0.5B',
        description='inclusionAI Ming-3.0-tts 0.5B (MIT). Klonen ueber Stimmprofile, echtes Chunk-Streaming, '
        'kontinuierliches Batching; Ausgabe 24 kHz.',
        packages=('omnivoice_tts_server.ming', 'omnivoice_tts_server.ming_runtime', 'einops', 'onnxruntime', 'soxr', 'soundfile'),
        uses_voice_profiles=True,
        native_model_ids=False,
        native_streaming=True,
        preload=True,
    ),
    'phonon': EngineSpec(
        id='phonon',
        label='Gradium Phonon',
        description='Gradium Phonon 40M (CPU, ptts). Eigene Katalogstimmen, keine Stimmprofile.',
        packages=('ptts',),
        uses_voice_profiles=False,
        native_model_ids=False,
        native_streaming=True,
        preload=False,
    ),
    'mock': EngineSpec(
        id='mock',
        label='Mock (Testton)',
        description='Sinuston ohne Modell, fuer Tests.',
        packages=(),
        uses_voice_profiles=True,
        native_model_ids=True,
        native_streaming=False,
        preload=False,
        test_only=True,
    ),
}
assert set(ENGINES) == set(RUNTIME_BACKENDS)


def engine_spec(backend: str | None) -> EngineSpec | None:
    return ENGINES.get(str(backend or '').strip().lower())


def env_backend_override(environ: Any = None) -> str | None:
    env = os.environ if environ is None else environ
    for name, value in env.items():
        if name.upper() == RUNTIME_BACKEND_ENV and str(value).strip():
            return str(value).strip().lower()
    return None


def _module_installed(name: str) -> bool:
    if name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _omnivoice_model_present(settings: Settings) -> bool:
    root = Path(settings.models_root_dir)
    if (root / 'OmniVoice' / 'config.json').exists():
        return True
    for cache in (root, root / '.hf' / 'hub'):
        snapshots = cache / 'models--k2-fsa--OmniVoice' / 'snapshots'
        if snapshots.is_dir() and any(item.is_dir() for item in snapshots.iterdir()):
            return True
    return False


def _ming_model_present(settings: Settings) -> bool:
    model_dir = Path(settings.ming_model_dir)
    return (model_dir / 'config.json').exists() and any(model_dir.glob('*.safetensors'))


def engine_info(spec: EngineSpec, settings: Settings) -> EngineInfo:
    missing = [name for name in spec.packages if not _module_installed(name)]
    installed = not missing
    model_id: str | None = None
    model_dir: str | None = None
    downloadable = False
    if spec.id == 'omnivoice':
        model_id = 'k2-fsa/OmniVoice'
        model_dir = str(Path(settings.models_root_dir) / 'OmniVoice')
        model_present = _omnivoice_model_present(settings)
        downloadable = bool(settings.allow_model_downloads)
        hint = 'Download beim ersten Laden (allow_model_downloads).'
    elif spec.id == 'ming':
        model_id = settings.ming_model_id
        model_dir = str(settings.ming_model_dir)
        model_present = _ming_model_present(settings)
        hint = f'hf download {MING_REPO_ID} --local-dir "{model_dir}"'
    elif spec.id == 'phonon':
        model_id = settings.phonon_model_id
        model_dir = str(settings.phonon_model_dir)
        model_present = (Path(settings.phonon_model_dir) / 'config.json').exists()
        hint = f'Checkpoint {model_id} (gated) nach {model_dir} laden.'
    else:
        model_present = True
        hint = ''
    if not installed:
        status, detail = 'missing_package', 'Fehlende Python-Pakete: ' + ', '.join(missing)
        if spec.id == 'ming':
            detail += ' (install: pip install -e "backend[ming]")'
    elif not model_present and downloadable:
        status, detail = 'downloadable', hint
    elif not model_present:
        status, detail = 'missing_model', 'Modell fehlt. ' + hint
    else:
        status, detail = 'ready', None
    active = settings.runtime_backend == spec.id
    return EngineInfo(
        id=spec.id,
        label=spec.label,
        description=spec.description,
        active=active,
        installed=installed,
        model_present=model_present,
        available=installed and (model_present or downloadable),
        status='active' if active else status,
        detail=detail,
        model_id=model_id,
        model_dir=model_dir,
        uses_voice_profiles=spec.uses_voice_profiles,
        native_streaming=spec.native_streaming,
        test_only=spec.test_only,
    )


def list_engines(settings: Settings) -> list[EngineInfo]:
    return [engine_info(spec, settings) for spec in ENGINES.values()]


def uses_voice_profiles(backend: str | None, synthesizer: Any = None) -> bool:
    flag = getattr(synthesizer, 'uses_voice_profiles', None) if synthesizer is not None else None
    if isinstance(flag, bool):
        return flag
    spec = engine_spec(backend)
    return spec.uses_voice_profiles if spec else True


def finetune_supported(settings: Settings) -> bool:
    """Data generation, training and promotion target OmniVoice checkpoints."""
    return settings.runtime_backend in ('omnivoice', 'mock')


def map_request_model(settings: Settings, model: str | None) -> str | None:
    """A request naming a model the active engine does not serve (e.g. an OmniVoice id while Ming or Phonon
    runs) is served by the active engine's default model (None = the active alias), so clients need no change.
    OmniVoice and mock keep every id as given."""
    spec = engine_spec(settings.runtime_backend)
    if spec is None or spec.native_model_ids or not model:
        return model
    if model in settings.supported_models:
        return model
    lowered = model.casefold()
    for candidate in settings.supported_models:
        if candidate.casefold() == lowered:
            return candidate
    return None


def map_speech_request(settings: Settings, request: SpeechRequest) -> SpeechRequest:
    mapped = map_request_model(settings, request.model)
    if mapped == request.model:
        return request
    return request.model_copy(update={'model': mapped})


def fix_active_model(settings: Settings) -> bool:
    """Reset an active_model the model list does not contain (e.g. a Phonon id saved by an older version).
    Returns True when it changed."""
    if settings.active_model in settings.supported_models or not settings.supported_models:
        return False
    previous = settings.active_model
    settings.active_model = (
        DEFAULT_ACTIVE_MODEL if DEFAULT_ACTIVE_MODEL in settings.supported_models else settings.supported_models[0]
    )
    logger.warning('active_model %r is not a supported model; reset to %r', previous, settings.active_model)
    return True


def persist_overrides_for(backend: str, baseline: dict[str, Any]) -> dict[str, Any]:
    """Fields save_runtime_settings must write from the OmniVoice baseline while `backend` runs."""
    spec = engine_spec(backend)
    if spec is None or spec.native_model_ids:
        return {}
    overrides = {'active_model': baseline.get('active_model', DEFAULT_ACTIVE_MODEL)}
    if not spec.uses_voice_profiles:
        overrides['default_voice'] = baseline.get('default_voice')
    return overrides


def restored_fields_for(backend: str) -> tuple[str, ...]:
    """Fields to put back to the baseline when leaving `backend`. A profile-using engine shares the default
    voice with OmniVoice, so a default voice chosen there is kept."""
    spec = engine_spec(backend)
    fields = ('supported_models', 'active_model', 'built_in_voices')
    if spec is not None and not spec.uses_voice_profiles:
        fields = (*fields, 'default_voice')
    return fields


def release_cuda_memory() -> None:
    gc.collect()
    torch = sys.modules.get('torch')
    if torch is None:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception as exc:  # noqa: BLE001
        logger.warning('CUDA cache release failed: %s', exc)


_SWITCHERS: 'weakref.WeakKeyDictionary[Any, EngineSwitcher]' = weakref.WeakKeyDictionary()


def switcher_for(store: Any) -> 'EngineSwitcher | None':
    try:
        return _SWITCHERS.get(store)
    except TypeError:
        return None


class EngineSwitchError(RuntimeError):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


class EngineSwitcher:
    """Owns the live switch for one app (app.state.engine_switcher)."""

    def __init__(self, app_state: Any, baseline: dict[str, Any]) -> None:
        self.app_state = app_state
        self.baseline = baseline
        self.status = EngineSwitchStatus()
        self.voice_profiles_active = False
        self._lock = asyncio.Lock()
        self._ready = asyncio.Event()
        self._ready.set()
        self._task: asyncio.Task | None = None
        _SWITCHERS[app_state.store] = self

    # -- state ---------------------------------------------------------------------------------------------------

    @property
    def switching(self) -> bool:
        return self._lock.locked()

    @property
    def pending_backend(self) -> str | None:
        return self.status.to_backend if self.switching else None

    async def wait_ready(self) -> None:
        """New public requests wait here while a switch runs, then go to the new engine."""
        await self._ready.wait()

    def apply_persist_overrides(self) -> None:
        settings = self.app_state.settings
        set_persisted_overrides(settings, persist_overrides_for(settings.runtime_backend, self.baseline))

    # -- voice profiles ------------------------------------------------------------------------------------------

    def sync_voice_profiles(self, backend: str, synthesizer: Any = None) -> None:
        """Load the saved voice profiles for engines that clone from them; hide them (after saving) otherwise,
        exactly like a fresh start in that mode."""
        settings = self.app_state.settings
        store = self.app_state.store
        wanted = uses_voice_profiles(backend, synthesizer)
        if wanted and not self.voice_profiles_active:
            from .voice_presets import seed_voice_presets

            store.load_voices(settings.data_dir)
            try:
                seed_voice_presets(store, settings.data_dir)
            except (OSError, ValueError) as exc:
                logger.warning('voice presets not seeded: %s', exc)
            self.voice_profiles_active = True
        elif not wanted and self.voice_profiles_active:
            store.save_voices(settings.data_dir)
            store.voice_profiles.clear()
            store.prompt_cache.clear()
            self.voice_profiles_active = False

    # -- switch --------------------------------------------------------------------------------------------------

    def check_target(self, target: str) -> EngineSpec:
        spec = engine_spec(target)
        if spec is None:
            raise EngineSwitchError(f'Unknown runtime backend: {target}', 400)
        info = engine_info(spec, self.app_state.settings)
        if not info.available:
            raise EngineSwitchError(f'{spec.label} is not available: {info.detail}', 400)
        store = self.app_state.store
        for run in list(getattr(store, 'finetune_train_runs', {}).values()) + list(getattr(store, 'finetune_runs', {}).values()):
            if isinstance(run, dict) and str(run.get('status') or '') not in _TERMINAL_RUN:
                raise EngineSwitchError('A finetune run is active; finish or cancel it before switching the engine.', 409)
        return spec

    async def request_switch(self, target: str, *, wait: bool = False, drain_timeout_s: float = 120.0) -> EngineSwitchStatus:
        """Start a switch to `target` (no-op when it already runs). Raises EngineSwitchError when the engine is
        unknown or not installed, or when another switch or a finetune run is active."""
        target = str(target or '').strip().lower()
        settings = self.app_state.settings
        if self.switching:
            raise EngineSwitchError(f'An engine switch to {self.status.to_backend} is already running.', 409)
        if target == settings.runtime_backend:
            return self.status
        self.check_target(target)
        await self._lock.acquire()
        self.status = EngineSwitchStatus(
            state='draining', from_backend=settings.runtime_backend, to_backend=target, started_at=utcnow(),
            message='Laufende Jobs werden fertiggestellt.',
        )
        self._ready.clear()
        task = asyncio.create_task(self._run(target, drain_timeout_s))
        self._task = task
        store = self.app_state.store
        tracked = getattr(store, 'background_tasks', None)
        if isinstance(tracked, set):
            tracked.add(task)
            task.add_done_callback(tracked.discard)
        if wait:
            await asyncio.shield(task)
        return self.status

    async def _publish(self, reason: str) -> None:
        events = getattr(self.app_state, 'events', None)
        if events is None:
            return
        try:
            await events.publish('dashboard.snapshot', {'reason': reason, 'engine_switch': self.status.model_dump(mode='json')})
        except Exception:  # noqa: BLE001
            pass

    async def _run(self, target: str, drain_timeout_s: float) -> None:
        state = self.app_state
        settings: Settings = state.settings
        store = state.store
        queue = state.queue_service
        old_backend = settings.runtime_backend
        old_spec = engine_spec(old_backend)
        old_synth = state.synthesizer
        held: list[str] = []
        success = False
        try:
            await self._publish('engine.switch.draining')
            drained, cancelled = await self._drain(drain_timeout_s, held)
            self.status.drained_jobs = drained
            self.status.cancelled_jobs = cancelled

            self.status.state = 'unloading'
            self.status.message = 'Alte Engine wird entladen.'
            await self._publish('engine.switch.unloading')
            await queue.stop_worker()
            try:
                await old_synth.unload()
            except Exception as exc:  # noqa: BLE001
                logger.warning('engine switch: unload of %s failed: %s', old_backend, exc)
            close = getattr(old_synth, 'close', None)
            if callable(close):
                try:
                    result = close()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as exc:  # noqa: BLE001
                    logger.warning('engine switch: close of %s failed: %s', old_backend, exc)
            await asyncio.to_thread(release_cuda_memory)

            # Back to the OmniVoice values of the fields the old engine rewrote (an OmniVoice-like engine has
            # the current values as its baseline).
            old_fields = snapshot_engine_fields(settings)
            if old_spec is None or old_spec.native_model_ids:
                self.baseline = snapshot_engine_fields(settings)
            else:
                restore_engine_fields(settings, self.baseline, restored_fields_for(old_backend))
            store.prompt_cache.clear()
            store.models_loaded.clear()

            self.status.state = 'loading'
            self.status.message = 'Neue Engine wird gestartet.'
            await self._publish('engine.switch.loading')
            settings.runtime_backend = target
            new_synth: Any = None
            try:
                self.sync_voice_profiles(target)
                from .runtime_v2 import build_synthesizer

                new_synth = await asyncio.to_thread(build_synthesizer, settings, store)
            except Exception as exc:
                # Keep the old synthesizer object: it reloads its model on the next request.
                logger.exception('engine switch to %s failed; keeping %s', target, old_backend)
                settings.runtime_backend = old_backend
                restore_engine_fields(settings, old_fields)
                self.sync_voice_profiles(old_backend, old_synth)
                store.active_model = settings.active_model
                self.status.state = 'failed'
                self.status.error = f'{type(exc).__name__}: {exc}'
                self.status.message = f'Wechsel fehlgeschlagen, {old_backend} bleibt aktiv.'

            if new_synth is not None:
                spec = engine_spec(target)
                if spec is not None and spec.native_model_ids:
                    from .finetune.registry import seed_supported_models

                    seed_supported_models(settings)
                    fix_active_model(settings)
                store.active_model = settings.active_model
                self.sync_voice_profiles(target, new_synth)
                self._bind(new_synth)
                success = True
        except Exception as exc:  # noqa: BLE001
            logger.exception('engine switch to %s failed', target)
            self.status.state = 'failed'
            self.status.error = f'{type(exc).__name__}: {exc}'
        finally:
            self.apply_persist_overrides()
            self.status.requeued_jobs = await self._requeue(held)
            try:
                await queue.start_worker()
            except Exception as exc:  # noqa: BLE001
                logger.exception('engine switch: worker restart failed: %s', exc)
            try:
                save_runtime_settings(settings)
            except Exception as exc:  # noqa: BLE001
                logger.warning('engine switch: saving runtime settings failed: %s', exc)
            self._ready.set()
            self._lock.release()
        if success:
            await self._preload(target)
        self.status.finished_at = utcnow()
        await self._publish('engine.switch.done')

    def _bind(self, synthesizer: Any) -> None:
        state = self.app_state
        state.synthesizer = synthesizer
        state.queue_service.synthesizer = synthesizer
        trainer = getattr(state, 'finetune_trainer', None)
        if trainer is not None:
            trainer.synthesizer = synthesizer

    async def _preload(self, target: str) -> None:
        spec = engine_spec(target)
        settings = self.app_state.settings
        synthesizer = self.app_state.synthesizer
        if spec is not None and spec.preload and settings.active_model:
            self.status.message = 'Modell wird geladen.'
            await self._publish('engine.switch.preload')
            try:
                await synthesizer.ensure_model(settings.active_model)
                self.app_state.store.models_loaded.add(settings.active_model)
            except Exception as exc:  # noqa: BLE001
                logger.warning('engine switch: model preload for %s failed: %s', target, exc)
                self.status.state = 'ready'
                self.status.error = f'Engine aktiv, aber das Modell konnte nicht geladen werden: {exc}'
                self.status.message = None
                return
        self.status.state = 'ready'
        self.status.error = None
        self.status.message = f'{spec.label if spec else target} aktiv.'

    async def _drain(self, timeout_s: float, held: list[str]) -> tuple[int, int]:
        """Hold queued jobs back and let running ones finish (cancel them after timeout_s)."""
        store = self.app_state.store
        queue = self.app_state.queue_service
        loop = asyncio.get_running_loop()

        def stash_locked() -> None:
            while store.waiting_requests:
                held.append(store.waiting_requests.popleft())

        async def busy() -> bool:
            async with store.job_condition:
                stash_locked()
                return bool(store.active_request_ids) or store.current_batch is not None

        async with store.job_condition:
            stash_locked()
            drained = len(store.active_request_ids)
        deadline = loop.time() + timeout_s
        while await busy() and loop.time() < deadline:
            await asyncio.sleep(0.05)
        cancelled = 0
        if await busy():
            for job_id in list(store.active_request_ids):
                try:
                    await queue.cancel(job_id)
                    cancelled += 1
                except Exception:  # noqa: BLE001
                    pass
            grace = loop.time() + CANCEL_GRACE_S
            while await busy() and loop.time() < grace:
                await asyncio.sleep(0.05)
        return max(0, drained - cancelled), cancelled

    async def _requeue(self, held: list[str]) -> int:
        if not held:
            return 0
        store = self.app_state.store
        queue = self.app_state.queue_service
        settings = self.app_state.settings
        count = 0
        async with store.job_condition:
            for job_id in reversed(held):
                job = store.jobs.get(job_id)
                if job is None or job.status in _TERMINAL_JOB:
                    continue
                job.request = map_speech_request(settings, job.request)
                request_state = store.request_states.get(job_id)
                group_key = getattr(queue, '_group_key_for_request', None)
                if request_state is not None and callable(group_key):
                    try:
                        request_state.group_key = group_key(job.request)
                    except Exception:  # noqa: BLE001
                        pass
                store.waiting_requests.appendleft(job_id)
                count += 1
            recompute = getattr(queue, '_recompute_positions_locked', None)
            if callable(recompute):
                recompute()
            store.job_condition.notify_all()
        return count
