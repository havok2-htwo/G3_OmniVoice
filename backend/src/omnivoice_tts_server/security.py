from __future__ import annotations

import threading
import time
from collections import deque
from urllib.parse import urlsplit

from fastapi import HTTPException, Request, Response, status

from .domain.state import SESSION_TTL_SECONDS, InMemoryStore

SESSION_COOKIE_NAME = "g3_omnivoice_session"
API_KEY_REQUIRED_DETAIL = "A valid API key is required (Authorization: Bearer <key> or X-API-Key: <key>)."
API_KEY_STORE_UNAVAILABLE_DETAIL = "The API key store could not be read. Ask the administrator to check the server log."

# Login rate limit: at most this many failed logins per client IP within the window (per app instance).
LOGIN_MAX_FAILURES = 5
LOGIN_WINDOW_SECONDS = 60.0
_login_lock = threading.Lock()


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
        samesite="strict",
        secure=_cookie_secure(request),
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(key=SESSION_COOKIE_NAME, path="/")


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _login_failures(request: Request) -> dict[str, deque[float]]:
    failures = getattr(request.app.state, 'login_failures', None)
    if failures is None:
        failures = {}
        request.app.state.login_failures = failures
    return failures


def login_rate_limited(request: Request) -> bool:
    """True when this IP already failed LOGIN_MAX_FAILURES times within the window."""
    now = time.monotonic()
    with _login_lock:
        failures = _login_failures(request).get(_client_ip(request))
        if not failures:
            return False
        while failures and failures[0] < now - LOGIN_WINDOW_SECONDS:
            failures.popleft()
        return len(failures) >= LOGIN_MAX_FAILURES


def record_login_failure(request: Request) -> None:
    with _login_lock:
        store = _login_failures(request)
        store.setdefault(_client_ip(request), deque(maxlen=LOGIN_MAX_FAILURES * 4)).append(time.monotonic())


def clear_login_failures(request: Request) -> None:
    with _login_lock:
        _login_failures(request).pop(_client_ip(request), None)


def require_same_origin(request: Request) -> None:
    """Mutating admin calls must come from this server's own pages (Origin, else Referer, must match Host).
    Requests without either header (curl, scripts) are allowed: the session cookie is SameSite=Strict anyway."""
    if request.method in {"GET", "HEAD", "OPTIONS"}:
        return
    source = request.headers.get("origin") or request.headers.get("referer")
    if not source or source == "null":
        if source == "null":
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cross-origin admin request refused.")
        return
    host = (request.headers.get("x-forwarded-host") or request.headers.get("host") or "").split(",")[0].strip().lower()
    if urlsplit(source).netloc.lower() != host:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cross-origin admin request refused.")


def require_session(request: Request) -> dict:
    """Resolve the admin session cookie to a user context, else 401."""
    ctx = get_store(request).get_session(request.cookies.get(SESSION_COOKIE_NAME))
    if not ctx:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required.")
    require_same_origin(request)
    return ctx


def require_admin(request: Request) -> dict:
    """Session required; blocks all admin actions until the forced password change is done."""
    ctx = require_session(request)
    if ctx.get("must_change_password"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="password_change_required")
    return ctx


def extract_api_key(request: Request) -> str | None:
    """The key a client sent: Authorization: Bearer <key> first, then X-API-Key. None when neither is present."""
    authorization = request.headers.get("authorization") or ""
    scheme, _, value = authorization.partition(" ")
    if scheme.lower() == "bearer" and value.strip():
        return value.strip()
    header = (request.headers.get("x-api-key") or "").strip()
    return header or None


def _api_key_required() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=API_KEY_REQUIRED_DETAIL,
        headers={"WWW-Authenticate": "Bearer"},
    )


def authorize_api_key(request: Request) -> str | None:
    """Gate of the public API. No client key configured -> open (None).

    Once a key exists one of these is required, in this order: Authorization: Bearer <key>, X-API-Key: <key>,
    or a valid admin session (the admin's own Playground; no usage is counted for it). A key that is sent but
    invalid is refused even with a session. Returns the matched key id for usage accounting."""
    store = get_store(request)
    if store.api_keys_unavailable():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=API_KEY_STORE_UNAVAILABLE_DETAIL)
    if not store.has_api_keys():
        return None
    provided = extract_api_key(request)
    if provided is not None:
        key_id = store.match_api_key(provided)
        if not key_id:
            raise _api_key_required()
        return key_id
    session = store.get_session(request.cookies.get(SESSION_COOKIE_NAME))
    if session and not session.get("must_change_password"):
        return None
    raise _api_key_required()
