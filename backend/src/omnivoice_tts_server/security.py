from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import Depends, Header, HTTPException, Request, status

from .config import Settings, get_settings
from .domain.state import ApiKeyRecord, InMemoryStore, new_id, utcnow


def hash_key(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def generate_api_key() -> str:
    return f'omnivoice_tts_{secrets.token_urlsafe(24)}'


def get_store(request: Request) -> InMemoryStore:
    return request.app.state.store


def _get_startup_key_context(request: Request) -> dict:
    return getattr(request.app.state, '_startup_key_context', {})


async def require_admin_key(
    request: Request,
    authorization: str | None = Header(default=None),
    x_admin_key: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> ApiKeyRecord:
    provided: str | None = None
    if x_admin_key:
        provided = x_admin_key.strip()
    elif authorization and authorization.lower().startswith('bearer '):
        provided = authorization.split(' ', 1)[1].strip()

    if not provided:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Admin key required')

    store = get_store(request)
    settings = getattr(request.app.state, 'settings', settings)
    provided_hash = hash_key(provided)

    # Check persisted admin key records
    for record in store.api_keys.values():
        if (
            record.name == 'admin'
            and not record.disabled
            and secrets.compare_digest(record.key_hash, provided_hash)
        ):
            record.last_used_at = datetime.now(timezone.utc)
            store.save_secrets(settings.data_dir)
            return record

    # Check temporary startup admin key (TTL-based, shown at server start)
    ctx = _get_startup_key_context(request)
    startup_key = ctx.get('key', '')
    expires_at = ctx.get('expires_at')
    if startup_key and secrets.compare_digest(startup_key, provided):
        if expires_at is None or datetime.now(timezone.utc) <= expires_at:
            # Return the admin record so callers always get a valid record
            admin = next((r for r in store.api_keys.values() if r.name == 'admin' and not r.disabled), None)
            if admin:
                admin.last_used_at = datetime.now(timezone.utc)
                store.save_secrets(settings.data_dir)
                return admin

    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid admin key')


def bootstrap_admin_key(store: InMemoryStore, settings: Settings) -> ApiKeyRecord:
    existing = next((record for record in store.api_keys.values() if record.name == 'admin'), None)
    if existing:
        return existing
    record = ApiKeyRecord(
        key_id=new_id('key'),
        name='admin',
        key_hash=hash_key(settings.admin_api_key),
        created_at=utcnow(),
    )
    store.api_keys[record.key_id] = record
    return record


def setup_startup_admin_key(app, settings: Settings) -> None:
    """Register the temporary startup admin key on app.state so require_admin_key can accept it."""
    raw = (settings.startup_admin_key or '').strip()
    if not raw:
        return
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=settings.startup_admin_key_ttl_seconds)
    app.state._startup_key_context = {
        'key': raw,
        'expires_at': expires_at,
    }


def get_admin_record(store: InMemoryStore) -> ApiKeyRecord:
    record = next((entry for entry in store.api_keys.values() if entry.name == 'admin' and not entry.disabled), None)
    if record is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='Admin key is not initialized')
    return record


def rotate_admin_key(store: InMemoryStore, settings: Settings) -> tuple[ApiKeyRecord, str]:
    raw_key = generate_api_key()
    record = get_admin_record(store)
    record.key_hash = hash_key(raw_key)
    record.created_at = utcnow()
    record.last_used_at = None
    settings.admin_api_key = raw_key
    store.save_secrets(settings.data_dir)
    return record, raw_key
