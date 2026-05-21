from __future__ import annotations

import secrets


def main() -> int:
    print(f"omnivoice_tts_startup_admin_{secrets.token_urlsafe(24)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
