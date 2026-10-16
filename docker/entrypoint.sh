#!/bin/sh
set -e

# Admin access is username/password (default admin/admin, forced change on first login),
# so no temporary startup admin key is generated here anymore. All OmniVoice runtime
# configuration comes from the OMNIVOICE_TTS_* environment variables.
# Deliberately writes no data/runtime_settings.json: on a fresh volume the server sees a
# first start and applies the "24GB VRAM" tier (presets.FIRST_START_PRESET). Tier keys the
# environment sets explicitly (e.g. OMNIVOICE_TTS_ENABLE_FLASHINFER) keep the env value,
# and URLs/paths set here via env are never written to that file.
exec "$@"
