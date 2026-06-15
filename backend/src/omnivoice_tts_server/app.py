from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .api.router_v2 import router as api_router
from .capacity import capacity_summary
from .config import Settings, get_settings
from .domain.state import InMemoryStore
from .finetune import FinetuneService, FinetuneTrainer
from .finetune.registry import seed_supported_models
from .runtime_v2 import build_synthesizer
from .security import bootstrap_admin_key, setup_startup_admin_key
from .services_v2 import (
    BenchmarkService,
    EventHub,
    QueueService,
    StatsService,
    TranscriptionService,
    WerBenchmarkService,
    spawn_tracked_task,
)

logger = logging.getLogger('omnivoice_tts_server')


class _DropInvalidHTTPRequestWarning(logging.Filter):
    """uvicorn/h11 logs 'Invalid HTTP request received' for non-HTTP bytes hitting the
    port (e.g. a TLS handshake to the plaintext port, or LAN scanners). Verified benign:
    our response framing and HTTP/1.1 keep-alive are correct (a standard client reusing a
    connection succeeds), so this is external noise, not an app bug. Drop just this record
    so it stops flooding the log while keeping every other uvicorn warning."""

    def filter(self, record: logging.LogRecord) -> bool:
        return 'Invalid HTTP request received' not in record.getMessage()


_LOG_FILTER_INSTALLED = False


def _install_uvicorn_log_filter() -> None:
    global _LOG_FILTER_INSTALLED
    if _LOG_FILTER_INSTALLED:
        return
    logging.getLogger('uvicorn.error').addFilter(_DropInvalidHTTPRequestWarning())
    _LOG_FILTER_INSTALLED = True


def configure_frontend(app: FastAPI, settings: Settings) -> None:
    frontend_root = settings.frontend_dist_dir.resolve()
    index_file = frontend_root / 'index.html'
    admin_file = frontend_root / 'admin.html'
    demo_file = frontend_root / 'demo.html'
    assets_dir = frontend_root / 'assets'

    if not index_file.exists():
        logger.info('frontend dist not found path=%s', frontend_root)
        return

    if assets_dir.is_dir():
        app.mount('/assets', StaticFiles(directory=assets_dir), name='frontend-assets')

    logger.info('frontend dist enabled path=%s', frontend_root)

    @app.get('/favicon.ico', include_in_schema=False)
    async def frontend_favicon() -> Response:
        icon = frontend_root / 'favicon.ico'
        if icon.is_file():
            return FileResponse(icon)
        # No icon shipped: 204 so browsers stop logging a 404 on every page load.
        return Response(status_code=204)

    @app.get('/', include_in_schema=False)
    async def frontend_index() -> FileResponse:
        return FileResponse(index_file)

    @app.get('/admin', include_in_schema=False)
    async def frontend_admin() -> FileResponse:
        return FileResponse(admin_file if admin_file.exists() else index_file)

    @app.get('/demo', include_in_schema=False)
    async def frontend_demo() -> FileResponse:
        return FileResponse(demo_file if demo_file.exists() else index_file)

    @app.get('/{full_path:path}', include_in_schema=False)
    async def frontend_spa(full_path: str) -> FileResponse:
        if (
            full_path == 'v1'
            or full_path.startswith('v1/')
            or full_path == 'api'
            or full_path.startswith('api/')
        ):
            raise HTTPException(status_code=404, detail='Not found')

        requested_path = (frontend_root / full_path).resolve()
        if requested_path.is_file() and requested_path.is_relative_to(frontend_root):
            return FileResponse(requested_path)

        raise HTTPException(status_code=404, detail='Not found')


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _install_uvicorn_log_filter()
        settings.models_root_dir.mkdir(parents=True, exist_ok=True)
        store = InMemoryStore(max_queue_size=settings.max_queue_size)
        store.load_secrets(settings.data_dir)
        store.load_voices(settings.data_dir)
        store.active_model = settings.active_model
        events = EventHub(store)
        synthesizer = build_synthesizer(settings, store)
        queue_service = QueueService(store, synthesizer, events, settings)
        stats_service = StatsService(settings)
        transcription_service = TranscriptionService(settings)
        benchmark_service = BenchmarkService(store, queue_service, events, settings)
        wer_benchmark_service = WerBenchmarkService(store, queue_service, transcription_service, events, settings)
        finetune_service = FinetuneService(store, queue_service, transcription_service, wer_benchmark_service, events, settings)
        # Re-seed the model list with any promoted custom finetune checkpoints so they are
        # selectable and the existing model-ops endpoints can load them.
        seed_supported_models(settings)
        finetune_trainer = FinetuneTrainer(store, settings, events, synthesizer)
        app.state.settings = settings
        app.state.store = store
        app.state.events = events
        app.state.synthesizer = synthesizer
        app.state.queue_service = queue_service
        app.state.stats_service = stats_service
        app.state.transcription_service = transcription_service
        app.state.benchmark_service = benchmark_service
        app.state.wer_benchmark_service = wer_benchmark_service
        app.state.finetune_service = finetune_service
        app.state.finetune_trainer = finetune_trainer
        bootstrap_admin_key(store, settings)
        store.save_secrets(settings.data_dir)
        setup_startup_admin_key(app, settings)
        logger.info(
            'startup runtime_backend=%s host=%s port=%s models_root=%s active_model=%s allow_downloads=%s default_voice=%s supported_models=%s',
            settings.runtime_backend,
            settings.host,
            settings.port,
            settings.models_root_dir,
            settings.active_model,
            settings.allow_model_downloads,
            settings.default_voice,
            ', '.join(settings.supported_models),
        )
        logger.info('startup frontend_dist=%s', settings.frontend_dist_dir)
        await queue_service.start_worker()
        logger.info('startup worker_state=%s queue_limit=%s', store.worker_state, settings.max_queue_size)
        cap = capacity_summary(getattr(settings, 'vram_budget_mb', 0))
        if cap['vram_budget_mb'] > 0:
            logger.info(
                'startup vram_budget_mb=%s -> max_batch_audio_s=%.0f max_chars_per_chunk=%s est_peak_vram_mb=%s max_input_chars=%s',
                cap['vram_budget_mb'], cap['max_batch_audio_seconds'], cap['max_chars_per_chunk'],
                cap['estimated_peak_vram_mb'], getattr(settings, 'max_input_chars', 0),
            )
        else:
            logger.info('startup vram budgeting disabled (vram_budget_mb=0)')

        async def _job_reaper() -> None:
            # Backstop for the per-batch prune: evicts terminal jobs even while idle so
            # store.jobs / final_audio never grow unbounded (retention_days is enforced here).
            while True:
                await asyncio.sleep(120)
                async with store.job_condition:
                    removed = store.prune_terminal_jobs(
                        retention_days=settings.retention_days,
                        max_retained_jobs=settings.max_retained_jobs,
                    )
                if removed:
                    logger.info('job reaper evicted %s terminal jobs', removed)

        spawn_tracked_task(store, _job_reaper(), label='job-reaper')

        # Pre-load the active model at startup so the first real request is never cold.
        # Also primes torch.compile / Triton kernel cache via a warmup inference pass.
        if settings.runtime_backend.lower() == 'omnivoice' and settings.active_model:
            try:
                logger.info('startup pre-loading model=%s', settings.active_model)
                await synthesizer.ensure_model(settings.active_model)
                logger.info('startup model ready model=%s', settings.active_model)
            except Exception as exc:
                logger.warning('startup model pre-load failed (non-critical): %s', exc)

        try:
            yield
        finally:
            logger.info('shutdown requested')
            await queue_service.stop_worker()
            pending = list(store.background_tasks)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            store.save_secrets(settings.data_dir)
            store.save_voices(settings.data_dir)
            logger.info('shutdown complete')

    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    # allow_origins='*' together with allow_credentials=True is unsafe: Starlette then
    # reflects any caller's Origin and lets it send credentialed requests. This API
    # authenticates via the X-Admin-Key / Authorization headers (not cookies), so we
    # keep the wildcard origin for easy LAN access but disable credentials.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=False,
        allow_methods=['*'],
        allow_headers=['*'],
    )

    @app.middleware('http')
    async def log_requests(request: Request, call_next):
        started = time.perf_counter()
        client = request.client.host if request.client else '-'
        origin = request.headers.get('origin', '-')
        auth = 'yes' if (
            request.headers.get('authorization')
            or request.headers.get('x-admin-key')
        ) else 'no'
        logger.info(
            'request start method=%s path=%s client=%s origin=%s auth=%s',
            request.method,
            request.url.path,
            client,
            origin,
            auth,
        )
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = int((time.perf_counter() - started) * 1000)
            logger.exception(
                'request error method=%s path=%s client=%s duration_ms=%s',
                request.method,
                request.url.path,
                client,
                duration_ms,
            )
            raise
        duration_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            'request end method=%s path=%s status=%s duration_ms=%s',
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response

    app.include_router(api_router)
    configure_frontend(app, settings)

    @app.exception_handler(KeyError)
    async def key_error_handler(_: Request, exc: KeyError):
        return JSONResponse(status_code=404, content={'detail': str(exc) or 'Not found'})

    return app
