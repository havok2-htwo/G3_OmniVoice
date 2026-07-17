from __future__ import annotations

from fastapi import HTTPException, Request, Response, status

from .domain.state import SESSION_TTL_SECONDS, InMemoryStore

SESSION_COOKIE_NAME = "g3_omnivoice_session"


def get_store(request: Request) -> InMemoryStore:
    return request.app.state.store


def _cookie_secure(request: Request) -> bool:
    forwarded = request.headers.get("x-forwarded-proto")
    if forwarded:
        return forwarded.split(",")[0].strip().lower() == "https"
    return request.url.scheme == "https"


def set_session_cookie(response: Response, request: Request, raw_token: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=raw_token,
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        samesite="lax",
        secure=_cookie_secure(request),
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(key=SESSION_COOKIE_NAME, path="/")


def require_session(request: Request) -> dict:
    """Resolve the admin session cookie to a user context, else 401."""
    ctx = get_store(request).get_session(request.cookies.get(SESSION_COOKIE_NAME))
    if not ctx:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required.")
    return ctx


def require_admin(request: Request) -> dict:
    """Session required; blocks all admin actions until the forced password change is done."""
    ctx = require_session(request)
    if ctx.get("must_change_password"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="password_change_required")
    return ctx


def authorize_api_key(request: Request) -> str | None:
    """Public-endpoint gate. No client keys configured -> open (None). Otherwise a valid
    X-API-Key header is required; returns the matched key id for usage accounting."""
    store = get_store(request)
    if not store.has_api_keys():
        return None
    key_id = store.match_api_key(request.headers.get("x-api-key"))
    if not key_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="A valid X-API-Key header is required.")
    return key_id
