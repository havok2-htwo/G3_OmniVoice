from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .api.router_v2 import router as api_router
from .config import Settings, get_settings
from .domain.state import InMemoryStore
from .runtime_v2 import build_synthesizer
from .security import bootstrap_admin_key, setup_startup_admin_key
from .services_v2 import BenchmarkService, EventHub, QueueService, StatsService, TranscriptionService, WerBenchmarkService

logger = logging.getLogger('omnivoice_tts_server')


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
        settings.models_root_dir.mkdir(parents=True, exist_ok=True)
        store = InMemoryStore(max_queue_size=settings.max_queue_size)
        store.load_secrets(settings.data_dir)
        store.load_voices(settings.data_dir)
        store.active_model = settings.active_model
        events = EventHub(store)
        synthesizer = build_synthesizer(settings, store)
        queue_service = QueueService(store, synthesizer, events, settings)
        stats_service = StatsService()
        transcription_service = TranscriptionService(settings)
        benchmark_service = BenchmarkService(store, queue_service, events, settings)
        wer_benchmark_service = WerBenchmarkService(store, queue_service, transcription_service, events, settings)
        app.state.settings = settings
        app.state.store = store
        app.state.events = events
        app.state.synthesizer = synthesizer
        app.state.queue_service = queue_service
        app.state.stats_service = stats_service
        app.state.transcription_service = transcription_service
        app.state.benchmark_service = benchmark_service
        app.state.wer_benchmark_service = wer_benchmark_service
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
            store.save_secrets(settings.data_dir)
            store.save_voices(settings.data_dir)
            logger.info('shutdown complete')

    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
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
