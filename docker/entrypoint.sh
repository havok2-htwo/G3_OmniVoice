#!/bin/sh
set -e

# Admin access is username/password (default admin/admin, forced change on first login),
# so no temporary startup admin key is generated here anymore. All OmniVoice runtime
# configuration comes from the OMNIVOICE_TTS_* environment variables.
exec "$@"
