#!/bin/sh
set -e

# On a fresh deploy with no admin key configured, generate a temporary startup admin key
# (like start_server.bat did) and print it to the container log so you can reach the admin
# UI. It is accepted as the X-Admin-Key for OMNIVOICE_TTS_STARTUP_ADMIN_KEY_TTL_SECONDS
# after startup; use it to log in and rotate a persistent key in the dashboard.
# Set OMNIVOICE_TTS_ADMIN_API_KEY for a fixed key instead (which disables this).
if [ -z "${OMNIVOICE_TTS_ADMIN_API_KEY:-}" ] && [ -z "${OMNIVOICE_TTS_STARTUP_ADMIN_KEY:-}" ]; then
  key="$(python /app/tools/generate_startup_admin_key.py 2>/dev/null || true)"
  if [ -n "$key" ]; then
    OMNIVOICE_TTS_STARTUP_ADMIN_KEY="$key"
    export OMNIVOICE_TTS_STARTUP_ADMIN_KEY
    : "${OMNIVOICE_TTS_STARTUP_ADMIN_KEY_TTL_SECONDS:=1800}"
    export OMNIVOICE_TTS_STARTUP_ADMIN_KEY_TTL_SECONDS
    echo "============================================================" >&2
    echo "OmniVoice - temporary startup admin key:" >&2
    echo "  ${OMNIVOICE_TTS_STARTUP_ADMIN_KEY}" >&2
    echo "Valid ~${OMNIVOICE_TTS_STARTUP_ADMIN_KEY_TTL_SECONDS}s after startup. Use it as the" >&2
    echo "X-Admin-Key in the admin UI, then rotate a persistent key there." >&2
    echo "(If the first-boot model download outlasts it, just restart the container to get" >&2
    echo " a fresh key. Set OMNIVOICE_TTS_ADMIN_API_KEY for a fixed key and disable this.)" >&2
    echo "============================================================" >&2
  fi
fi

exec "$@"
