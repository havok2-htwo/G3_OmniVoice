from __future__ import annotations

import hashlib
import os
import secrets


def build_pbkdf2_hash(password: str, iterations: int = 390000) -> str:
    salt = os.urandom(16).hex()
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), iterations).hex()
    return f"pbkdf2_sha256${iterations}${salt}${digest}"


def verify_pbkdf2_hash(password: str, encoded: str) -> bool:
    """Constant-time verify of a password against a pbkdf2_sha256$iter$salt$digest string."""
    try:
        algorithm, iterations, salt, digest = (encoded or "").split("$", 3)
    except ValueError:
        return False
    if algorithm != "pbkdf2_sha256":
        return False
    try:
        expected = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt.encode("utf-8"), int(iterations)
        ).hex()
    except (ValueError, TypeError):
        return False
    return secrets.compare_digest(expected, digest)
