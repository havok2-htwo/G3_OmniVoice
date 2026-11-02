"""TTS engine registry and the live engine switch.

Engines: 'omnivoice' (default), 'ming' (inclusionAI Ming-3.0-tts 0.5B), the four Audio8 checkpoints of
audio8/variants.py, 'phonon' (Gradium Phonon) and 'mock' (test tone). `runtime_v2.build_synthesizer` builds the synthesizer for `settings.runtime_backend`;
this module lists the engines with their install state and swaps the running one without a restart:

1. new public requests wait (`EngineSwitcher.wait_ready`), queued jobs are held back from the old engine;
2. jobs already running finish on the old engine (up to `drain_timeout_s`, then they are cancelled);
3. the queue worker stops, the old synthesizer unloads and CUDA memory is released;
4. the settings the old engine rewrote (model list, active model, ...) get their OmniVoice values back, the old
   engine's tuned values go to its profile and the new engine's are applied (ENGINE_TUNED_FIELDS), the new
   synthesizer is built and bound to the app, the queue and the finetune trainer;
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, get_args

from .audio8.variants import AUDIO8_VARIANTS
from .config import (
    ENGINE_TUNED_FIELDS,
    RUNTIME_BACKEND_ENV,
    RUNTIME_BACKENDS,
    Settings,
    restore_engine_fields,
    save_runtime_settings,
    set_persisted_overrides,
    snapshot_engine_fields,
)
from .domain.models import (
    EngineInfo,
    EngineSwitchStatus,
    JobStatus,
    RuntimeBackendName,
    ServerSettingsUpdateRequest,
    SpeechRequest,
)
from .domain.state import utcnow

logger = logging.getLogger('omnivoice_tts_server.engines')

DEFAULT_ACTIVE_MODEL = 'k2-fsa/OmniVoice-Base'
MING_REPO_ID = 'inclusionAI/Ming-3.0-tts-0.5B'
# The clone alias the queue serves on Ming. It ends in 'Base' so task-type inference, the default voice profile and
# the OpenAI voice routing treat it like OmniVoice-Base.
MING_BASE_ALIAS = 'inclusionAI/Ming-3.0-tts-0.5B-Base'
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
    # GET /v1/capabilities: the engine clones a reference voice, and the request fields it really reads.
    voice_cloning: bool = True
    client_options: tuple[str, ...] = ()
    # CapabilityOptions field -> note for clients, e.g. an option that is accepted but changes nothing.
    option_notes: Mapping[str, str] = field(default_factory=dict)
    # Runs without a GPU: no device selection, no VRAM.
    cpu_only: bool = False
    # The checkpoint's licence as the admin shows it; restricted when commercial use is limited.
    license: str | None = None
    license_restricted: bool = False
    approx_size_gb: float | None = None


_AUDIO8_LANGUAGE_NOTE = 'Accepted, but Audio8 has no language parameter: the text alone decides.'


def _audio8_spec(engine_id: str, description: str) -> EngineSpec:
    variant = AUDIO8_VARIANTS[engine_id]
    runtime = ('onnxruntime', 'onnx', 'tokenizers') if variant.runtime == 'onnx' else ('transformers',)
    return EngineSpec(
        id=engine_id,
        label=variant.label,
        description=description,
        packages=('omnivoice_tts_server.audio8', 'omnivoice_tts_server.audio8_runtime', *runtime, 'soxr'),
        uses_voice_profiles=True,
        native_model_ids=False,
        native_streaming=True,
        preload=True,
        client_options=('seed', 'language'),
        option_notes={'language': _AUDIO8_LANGUAGE_NOTE},
        cpu_only=variant.runtime == 'onnx',
        license=variant.license,
        license_restricted=variant.license_restricted,
        approx_size_gb=variant.approx_size_gb,
    )


ENGINES: dict[str, EngineSpec] = {
    'omnivoice': EngineSpec(
        id='omnivoice',
        label='OmniVoice',
        description='k2-fsa OmniVoice Base (Standard). Klonen ueber Stimmprofile, Finetuning.',
        packages=('omnivoice',),
        uses_voice_profiles=True,
        native_model_ids=True,
        native_streaming=False,
        preload=True,
        client_options=('speed', 'seed', 'language', 'instructions'),
    ),
    'ming': EngineSpec(
        id='ming',
        label='Ming-3.0-tts 0.5B',
        description='inclusionAI Ming-3.0-tts 0.5B (MIT). Klonen ueber Stimmprofile, echtes Chunk-Streaming, '
        'kontinuierliches Batching; Ausgabe 24 kHz.',
        packages=('omnivoice_tts_server.ming', 'omnivoice_tts_server.ming_runtime', 'onnxruntime', 'soxr'),
        uses_voice_profiles=True,
        native_model_ids=False,
        native_streaming=True,
        preload=True,
        client_options=(
            'seed', 'language', 'instructions', 'emotion', 'speaking_rate', 'pitch', 'volume', 'reference_mode',
        ),
        license='MIT',
        approx_size_gb=3.0,
    ),
    'audio8_realistic': _audio8_spec(
        'audio8_realistic',
        'Audio8 Realistic 0.6B (Apache-2.0). Klonen ueber Stimmprofile, echtes Chunk-Streaming, kontinuierliches '
        'Batching; Ausgabe 44,1 kHz.',
    ),
    'audio8_preview': _audio8_spec(
        'audio8_preview',
        'Audio8 Preview 0.6B (Apache-2.0). Klonen ueber Stimmprofile, echtes Chunk-Streaming, kontinuierliches '
        'Batching; Ausgabe 44,1 kHz.',
    ),
    'audio8_mini': _audio8_spec(
        'audio8_mini',
        'Audio8 Preview 0.1B (Falcon-H1, weniger VRAM). Klonen ueber Stimmprofile, echtes Chunk-Streaming, '
        'kontinuierliches Batching; Ausgabe 44,1 kHz.',
    ),
    'audio8_onnx': _audio8_spec(
        'audio8_onnx',
        'Audio8 0.1B ONNX INT8 (Apache-2.0) auf der CPU, keine GPU noetig. Klonen ueber Stimmprofile, '
        'Chunk-Streaming; Ausgabe 44,1 kHz.',
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
        voice_cloning=False,
        client_options=('seed',),
        cpu_only=True,
        license='CC-BY-NC-4.0: keine kommerzielle Nutzung',
        license_restricted=True,
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
        voice_cloning=False,
    ),
}
assert set(ENGINES) == set(RUNTIME_BACKENDS) == set(get_args(RuntimeBackendName))

# PUT /api/admin/settings fields each engine reads. The admin shows these and nothing of another engine.
_OMNIVOICE_SETTINGS_FIELDS = (
    'default_voice', 'preferred_device', 'model_directory', 'allow_model_downloads', 'max_parallel_requests',
    'max_batch_size', 'batch_wait_ms', 'vram_budget_mb', 'stream_chunk_ms', 'torch_dtype', 'num_step',
    'guidance_scale', 'duration', 't_shift', 'position_temperature', 'class_temperature', 'audio_chunk_duration',
    'audio_chunk_threshold', 'compile_model', 'sdpa_prefer_cudnn', 'packed_cfg', 'stream_first_preempt',
    'cudagraph_skip_dynamic_graphs', 'cuda_memory_trim_after_batch', 'warmup_on_startup', 'denoise',
    'preprocess_prompt', 'postprocess_output',
)
_AUDIO8_SHARED_SETTINGS_FIELDS = (
    'default_voice', 'max_parallel_requests', 'stream_prebuffer_ms', 'audio8_output_sample_rate',
    'audio8_fragment_max_chars', 'audio8_temperature', 'audio8_top_p', 'audio8_top_k',
)
_AUDIO8_GPU_SETTINGS_FIELDS = (
    *_AUDIO8_SHARED_SETTINGS_FIELDS, 'preferred_device', 'audio8_max_slots', 'audio8_first_chunk_frames',
    'audio8_chunk_frames', 'audio8_cuda_graphs',
)
ENGINE_SETTINGS_FIELDS: dict[str, tuple[str, ...]] = {
    'omnivoice': _OMNIVOICE_SETTINGS_FIELDS,
    'ming': (
        'default_voice', 'preferred_device', 'max_parallel_requests', 'stream_prebuffer_ms', 'ming_max_slots',
        'ming_first_chunk_latents', 'ming_max_chunk_latents', 'ming_output_sample_rate', 'ming_model_dir',
        'ming_cuda_graphs',
    ),
    'audio8_realistic': _AUDIO8_GPU_SETTINGS_FIELDS,
    'audio8_preview': _AUDIO8_GPU_SETTINGS_FIELDS,
    'audio8_mini': _AUDIO8_GPU_SETTINGS_FIELDS,
    'audio8_onnx': (
        *_AUDIO8_SHARED_SETTINGS_FIELDS, 'audio8_onnx_workers', 'audio8_onnx_threads',
        'audio8_onnx_first_chunk_frames', 'audio8_onnx_chunk_frames',
    ),
    'phonon': (
        'max_parallel_requests', 'max_batch_size', 'batch_wait_ms', 'vram_budget_mb', 'stream_prebuffer_ms',
        'phonon_lang', 'phonon_quant', 'phonon_workers', 'phonon_threads',
    ),
    'mock': _OMNIVOICE_SETTINGS_FIELDS,
}
# PUT /api/admin/settings fields of the whole service, whichever engine runs.
SERVER_SETTINGS_FIELDS: tuple[str, ...] = (
    'whisper_base_url', 'vllm_base_url', 'vllm_model', 'wer_concurrency', 'wer_transcription_concurrency',
    'queue_limit', 'max_input_chars', 'text_optimization_default', 'sentence_chunking',
)
assert set(ENGINE_SETTINGS_FIELDS) == set(ENGINES)
assert {name for names in ENGINE_SETTINGS_FIELDS.values() for name in names} <= set(ServerSettingsUpdateRequest.model_fields)
assert set(SERVER_SETTINGS_FIELDS) <= set(ServerSettingsUpdateRequest.model_fields)
assert set(ENGINE_TUNED_FIELDS) <= set(ServerSettingsUpdateRequest.model_fields)

# Aliases clients keep using; the admin selects neither of them.
CLIENT_ONLY_ALIAS_SUFFIXES = ('-AutoVoice', '-VoiceDesign')
# The built-in tier an engine starts from while it has no profile yet.
ENGINE_DEFAULT_TIERS = {
    'ming': 'Ming 24GB VRAM',
    'audio8_realistic': 'Audio8 24GB VRAM',
    'audio8_preview': 'Audio8 Preview 24GB VRAM',
    'audio8_mini': 'Audio8 Mini 24GB VRAM',
    'audio8_onnx': 'Audio8 ONNX CPU',
}
# Tier keys a start or switch without a profile leaves alone. The sample rate is what clients of this host get, not
# tuning: a saved 44100 stays, and nothing saved gives the code default, which equals the tier's value. Applying a tier
# as a preset still sets it, because the PUT sends it.
TIER_KEYS_NOT_STARTED_WITH = ('runtime_backend', 'ming_output_sample_rate', 'audio8_output_sample_rate')


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


def audio8_model_dir(settings: Settings, engine_id: str) -> Path:
    return Path(settings.models_root_dir) / AUDIO8_VARIANTS[engine_id].folder


def engine_loaded(settings: Settings, synthesizer: Any) -> bool:
    """The active engine really holds the admin's selection in memory, read from the synthesizer itself
    (store.models_loaded and resident_models list every alias)."""
    if settings.runtime_backend == 'omnivoice':
        # One checkpoint at a time, and a client's explicit local/... replaces the selection until the next request.
        loaded = getattr(synthesizer, '_loaded_model_id', None)
        if loaded is None:
            return False
        selected = settings.active_model or ''
        return loaded == selected or not (loaded.startswith('local/') or selected.startswith('local/'))
    resident = getattr(synthesizer, 'resident_models', None)
    return bool(resident()) if callable(resident) else False


def engine_info(spec: EngineSpec, settings: Settings, synthesizer: Any = None) -> EngineInfo:
    missing = [name for name in spec.packages if not _module_installed(name)]
    installed = not missing
    model_id: str | None = None
    model_dir: str | None = None
    # OmniVoice fetches missing weights itself when it loads; Ming needs them on disk before a switch.
    fetched_at_load = False
    downloadable = False
    if spec.id == 'omnivoice':
        model_id = 'k2-fsa/OmniVoice'
        model_dir = str(Path(settings.models_root_dir) / 'OmniVoice')
        model_present = _omnivoice_model_present(settings)
        fetched_at_load = bool(settings.allow_model_downloads)
        # The download route turns allow_model_downloads on itself: with the flag off and no weights it is the only way
        # back from another engine.
        downloadable = fetched_at_load or not model_present
        hint = (
            'Download beim ersten Laden (allow_model_downloads).' if fetched_at_load
            else 'Download hier im Admin (schaltet allow_model_downloads ein).'
        )
    elif spec.id == 'ming':
        model_id = settings.ming_model_id
        model_dir = str(settings.ming_model_dir)
        model_present = _ming_model_present(settings)
        downloadable = not model_present
        hint = f'Download hier im Admin oder: hf download {MING_REPO_ID} --local-dir "{model_dir}"'
    elif spec.id in AUDIO8_VARIANTS:
        variant = AUDIO8_VARIANTS[spec.id]
        model_id = variant.repo_id
        model_dir = str(audio8_model_dir(settings, spec.id))
        model_present = all((Path(model_dir) / name).exists() for name in variant.weight_files)
        downloadable = not model_present
        hint = (
            f'Download hier im Admin oder: hf download {variant.repo_id} --revision {variant.revision} '
            f'--local-dir "{model_dir}"'
        )
    elif spec.id == 'phonon':
        model_id = settings.phonon_model_id
        model_dir = str(settings.phonon_model_dir)
        model_present = (Path(settings.phonon_model_dir) / 'config.json').exists()
        hint = (
            f'Nur manueller Download: Checkpoint {model_id} ist gated (CC-BY-NC-4.0, Zugang bei Hugging Face '
            f'beantragen) und gehoert nach {model_dir}.'
        )
    else:
        model_present = True
        hint = ''
    if not installed:
        status, detail = 'missing_package', 'Fehlende Python-Pakete: ' + ', '.join(missing)
        if spec.id == 'ming':
            detail += ' (install: pip install -e "backend[ming]")'
        elif spec.id in AUDIO8_VARIANTS:
            detail += ' (install: pip install -e "backend[audio8]")'
    elif not model_present and fetched_at_load:
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
        available=installed and (model_present or fetched_at_load),
        downloadable=downloadable,
        loaded=active and synthesizer is not None and engine_loaded(settings, synthesizer),
        status='active' if active else status,
        detail=detail,
        model_id=model_id,
        model_dir=model_dir,
        uses_voice_profiles=spec.uses_voice_profiles,
        native_streaming=spec.native_streaming,
        test_only=spec.test_only,
        cpu_only=spec.cpu_only,
        license=spec.license,
        license_restricted=spec.license_restricted,
        approx_size_gb=spec.approx_size_gb,
        settings_fields=list(ENGINE_SETTINGS_FIELDS[spec.id]),
    )


def list_engines(settings: Settings, synthesizer: Any = None) -> list[EngineInfo]:
    return [engine_info(spec, settings, synthesizer) for spec in ENGINES.values()]


def selectable_models(settings: Settings) -> list[str]:
    """What the admin can pick: supported_models without the client-only aliases, in that order. A saved alias stays
    listed while it is the active model, nothing migrates it. Clients keep every id of supported_models."""
    models = [
        model for model in settings.supported_models
        if model == settings.active_model or not model.endswith(CLIENT_ONLY_ALIAS_SUFFIXES)
    ]
    return models if settings.active_model in models else [*models, settings.active_model]


def served_model(settings: Settings, model: str | None) -> str | None:
    """The id to load for a request. Every id but local/... names the checkpoint the admin selected: with a promoted
    local/x active, a client's literal OmniVoice-Base runs on it and evicts nothing. An explicit local/... is loaded."""
    selected = settings.active_model or ''
    if selected.startswith('local/') and not (model or '').startswith('local/'):
        return selected
    return model


def engine_model(settings: Settings, backend: str, baseline: dict[str, Any] | None = None) -> str:
    """The model id `backend` serves, also while another engine's alias is still in settings (a running switch or a
    download for an engine that is not active). `baseline` holds the OmniVoice selection while such an engine runs."""
    if backend == 'ming':
        return MING_BASE_ALIAS
    if backend in AUDIO8_VARIANTS:
        return AUDIO8_VARIANTS[backend].base_alias
    if backend == 'phonon':
        return settings.phonon_model_id
    current = engine_spec(settings.runtime_backend)
    if baseline and current is not None and not current.native_model_ids:
        return str(baseline.get('active_model') or DEFAULT_ACTIVE_MODEL)
    return settings.active_model


def engine_output_sample_rate(settings: Settings, backend: str) -> int:
    """The rate `backend` delivers once loaded with the current settings. Its synthesizer knows it while it runs."""
    if backend == 'ming':
        return settings.ming_output_sample_rate
    if backend in AUDIO8_VARIANTS:
        return settings.audio8_output_sample_rate
    return settings.sample_rate


# -- engine profiles (ENGINE_TUNED_FIELDS per engine, settings.engine_profiles) ------------------------------------------


def tuned_values(settings: Settings) -> dict[str, Any]:
    return {name: getattr(settings, name) for name in ENGINE_TUNED_FIELDS}


def default_tier_values(backend: str) -> dict[str, Any]:
    name = ENGINE_DEFAULT_TIERS.get(backend)
    if name is None:
        return {}
    from .presets import BUILTIN_PRESETS

    return {key: value for key, value in BUILTIN_PRESETS[name]['values'].items() if key not in TIER_KEYS_NOT_STARTED_WITH}


def engine_start_values(settings: Settings, backend: str) -> dict[str, Any]:
    """Settings values `backend` starts with: its profile, else its default tier, else nothing (current values stay)."""
    profile = settings.engine_profiles.get(backend)
    if profile:
        return {name: profile[name] for name in ENGINE_TUNED_FIELDS if name in profile}
    return default_tier_values(backend)


def swap_engine_values(settings: Settings, old_backend: str, target: str) -> dict[str, Any]:
    """Engine switch: the old engine's tuned values go to its profile, the target's are applied. Returns the replaced
    values for a rollback."""
    incoming = engine_start_values(settings, target)
    replaced = {name: getattr(settings, name) for name in (*ENGINE_TUNED_FIELDS, *incoming)}
    settings.engine_profiles[old_backend] = tuned_values(settings)
    for name, value in incoming.items():
        setattr(settings, name, value)
    return replaced


def stage_switch_values(settings: Settings, target: str, previous: dict[str, Any], sent: set[str]) -> None:
    """A settings PUT that also switches the engine: its tuned values belong to the target, so the running engine gets
    `previous` back (the switch then stores those under it) and the sent ones go to the target's profile. Other fields
    of the PUT stay applied and win over the target's default tier."""
    sent_tuned = {name: getattr(settings, name) for name in ENGINE_TUNED_FIELDS if name in sent}
    for name, value in previous.items():
        setattr(settings, name, value)
    start = engine_start_values(settings, target)
    for name, value in start.items():
        if name not in ENGINE_TUNED_FIELDS and name not in sent:
            setattr(settings, name, value)
    tuned_start = {name: value for name, value in start.items() if name in ENGINE_TUNED_FIELDS}
    settings.engine_profiles[target] = {**tuned_values(settings), **tuned_start, **sent_tuned}


def apply_startup_engine_values(settings: Settings, environ: Any = None) -> None:
    """Before the synthesizer is built: the active engine (saved or pinned by env) runs with its own values. Without a
    profile its default tier applies, except keys the environment sets explicitly (like the first-start tier), and the
    profile is seeded in memory. Nothing is written: the next full save stores it."""
    backend = settings.runtime_backend
    values = engine_start_values(settings, backend)
    seed = not settings.engine_profiles.get(backend)
    if seed:
        env_prefix = str(Settings.model_config.get('env_prefix') or '').upper()
        env_names = {name.upper() for name in (os.environ if environ is None else environ)}
        values = {name: value for name, value in values.items() if f'{env_prefix}{name}'.upper() not in env_names}
    for name, value in values.items():
        setattr(settings, name, value)
    if seed:
        settings.engine_profiles[backend] = tuned_values(settings)


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


def _render_pending_tags(request_state: Any, backend: str) -> list[int]:
    """Respell the inline tags of a held job's pending sentences for the engine it now runs on. A tag the old engine
    already dropped stays dropped.

    A sentence left with nothing to say (only tags the new engine does not know, maybe some punctuation) is dropped,
    never sent as is: it would be read aloud. It counts as rendered with no audio. Returns the dropped indices."""
    import regex

    from .text_optimizer.omnivoice_tags import render_inline_tags

    dropped: list[int] = []
    for index in request_state.pending_sentence_indices:
        text = request_state.sentences[index]
        rendered = render_inline_tags(text, backend)
        if rendered == text:
            continue
        rendered = rendered.strip()
        if regex.search(r'[\p{L}\p{N}]', rendered):
            request_state.sentences[index] = rendered
        else:
            dropped.append(index)
    if dropped:
        _drop_pending_sentences(request_state, dropped)
    return dropped


def _drop_pending_sentences(request_state: Any, indices: list[int]) -> None:
    """Take sentences out of the render plan; they emit no audio. Both emit paths (a rendered batch's
    ready_sentence_pcm, a native stream's completed_streaming_sentence_indices) step over them."""
    drop = set(indices)
    request_state.pending_sentence_indices = type(request_state.pending_sentence_indices)(
        index for index in request_state.pending_sentence_indices if index not in drop
    )
    # Leading dropped sentences: nothing to wait for, the next sentence is the first to emit.
    while request_state.next_emit_sentence_index in drop:
        drop.discard(request_state.next_emit_sentence_index)
        request_state.next_emit_sentence_index += 1
    for index in drop:
        request_state.ready_sentence_pcm[index] = b''
        request_state.sentence_duration_ms[index] = 0
        request_state.completed_streaming_sentence_indices.add(index)


def _resolve_delivery_conflict(job: Any, backend: str) -> None:
    """A held job moving to Ming with a free-form instruction and structured controls (accepted on another engine,
    which ignores the controls): the structured controls win, the free-form instruction is dropped. Submitting such
    a request on Ming is a 422; a job already accepted is not failed for it."""
    from .domain.models import delivery_controls_conflict

    if backend != 'ming' or not delivery_controls_conflict(job.request):
        return
    logger.warning('engine switch: job %s runs on ming with emotion/speaking_rate/pitch/volume; its free-form '
                   'instructions are dropped', job.job_id)
    job.request = job.request.model_copy(update={'instructions': None})


def _finish_without_rendering(queue: Any, job: Any, request_state: Any, backend: str) -> bool:
    """A held job whose remaining sentences were all dropped (_render_pending_tags) has nothing left to render:
    complete it with the audio it already has, or fail it when there is none. Returns True when the job is done.
    Called with the store lock held."""
    if request_state.pending_sentence_indices or request_state.inflight_sentence_indices:
        return False
    complete = getattr(queue, '_complete_job_locked', None)
    fail = getattr(queue, '_fail_job_locked', None)
    flush = getattr(queue, '_flush_ready_sentences_locked', None)
    if not (callable(complete) and callable(fail) and callable(flush)):
        return False
    if not job.pcm_parts and not any(request_state.ready_sentence_pcm.values()):
        fail(job, f'Input text is empty on the {backend} engine: it reads none of the inline tags left.')
        return True
    flush(job, request_state)
    if request_state.is_complete():
        complete(job, request_state)
        return True
    return False


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
        shutting_down = False
        try:
            await self._publish('engine.switch.draining')
            drained, cancelled = await self._drain(drain_timeout_s, held)
            self.status.drained_jobs = drained
            self.status.cancelled_jobs = cancelled

            self.status.state = 'unloading'
            self.status.message = 'Alte Engine wird entladen.'
            await self._publish('engine.switch.unloading')
            await queue.stop_worker()
            await self._await_running_batches(drain_timeout_s)
            try:
                await old_synth.unload()
            except Exception as exc:  # noqa: BLE001
                logger.warning('engine switch: unload of %s failed: %s', old_backend, exc)
            # unload() is the whole contract: the old object must stay reloadable for a rollback.
            await asyncio.to_thread(release_cuda_memory)

            # Back to the OmniVoice values of the fields the old engine rewrote (an OmniVoice-like engine has
            # the current values as its baseline).
            old_fields = snapshot_engine_fields(settings)
            if old_spec is None or old_spec.native_model_ids:
                self.baseline = snapshot_engine_fields(settings)
            else:
                if old_spec.uses_voice_profiles:
                    # The default voice is shared with OmniVoice (restored_fields_for keeps it): one chosen under
                    # this engine must survive a later switch through an engine that restores the baseline.
                    self.baseline['default_voice'] = settings.default_voice
                restore_engine_fields(settings, self.baseline, restored_fields_for(old_backend))
            store.prompt_cache.clear()
            store.models_loaded.clear()

            self.status.state = 'loading'
            self.status.message = 'Neue Engine wird gestartet.'
            await self._publish('engine.switch.loading')
            replaced_values = swap_engine_values(settings, old_backend, target)
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
                restore_engine_fields(settings, replaced_values, replaced_values)
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
                settings.engine_profiles[target] = tuned_values(settings)
                success = True
        except asyncio.CancelledError:
            # Server shutdown: leave the worker stopped.
            shutting_down = True
            self.status.state = 'failed'
            self.status.error = 'Cancelled (server shutdown).'
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception('engine switch to %s failed', target)
            self.status.state = 'failed'
            self.status.error = f'{type(exc).__name__}: {exc}'
        finally:
            self.apply_persist_overrides()
            self.status.requeued_jobs = await self._requeue(held)
            if not shutting_down:
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

    async def _await_running_batches(self, timeout_s: float) -> None:
        """Wait for stream batches that run as their own tasks (concurrent engines): stop_worker only awaits the
        worker, and a batch dispatched right before it stopped would otherwise lose its engine in unload()."""
        queue = self.app_state.queue_service
        running = getattr(queue, 'running_batch_count', None)
        if not callable(running):
            return
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.0, timeout_s)
        while running() and loop.time() < deadline:
            await asyncio.sleep(0.05)
        if not running():
            return
        store = self.app_state.store
        for job_id in list(store.active_request_ids):
            try:
                await queue.cancel(job_id)
            except Exception:  # noqa: BLE001
                pass
        grace = loop.time() + CANCEL_GRACE_S
        while running() and loop.time() < grace:
            await asyncio.sleep(0.05)

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
                _resolve_delivery_conflict(job, settings.runtime_backend)
                # A seed taken from the voice profile belongs to the old engine: resolve it for the new one.
                reapply_seed = getattr(queue, 'reapply_voice_seed', None)
                if callable(reapply_seed):
                    reapply_seed(job)
                request_state = store.request_states.get(job_id)
                if request_state is not None and _render_pending_tags(request_state, settings.runtime_backend):
                    if _finish_without_rendering(queue, job, request_state, settings.runtime_backend):
                        continue
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
