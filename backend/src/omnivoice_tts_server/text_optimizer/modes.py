"""Request flag handling for text optimization (import-light: no pipeline regexes)."""

from __future__ import annotations

from typing import Any, Optional

TEXT_OPTIMIZATION_MODES = ("off", "deterministic")
# Reserved for a later LLM rewrite mode over vllm_base_url; rejected for now.
RESERVED_TEXT_OPTIMIZATION_MODES = ("llm",)
# Only interactive traffic follows the admin default. Benchmarks / WER runs compare the
# audio against the raw text and finetune data generation needs audio that matches its
# transcript, so for them an unset flag always means "off".
DEFAULT_FOLLOWING_SCOPES = ("public",)


def parse_text_optimization_mode(value: Any) -> Optional[str]:
    """Normalise a flag value: None stays None, booleans map to deterministic/off.

    Raises ValueError for unknown or reserved values.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return "deterministic" if value else "off"
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("", "default"):
            return None
        if lowered in ("true", "on", "1"):
            return "deterministic"
        if lowered in ("false", "none", "0"):
            return "off"
        if lowered in TEXT_OPTIMIZATION_MODES:
            return lowered
        if lowered in RESERVED_TEXT_OPTIMIZATION_MODES:
            raise ValueError(f"text_optimization '{lowered}' is reserved and not available yet")
    raise ValueError("text_optimization must be 'off', 'deterministic', true or false")


def resolve_text_optimization_mode(
    requested: Any, metadata: Optional[dict], default_mode: Optional[str], owner_scope: str
) -> str:
    """Effective mode for one request: explicit field, then metadata.text_optimization,
    then the admin default (public requests only), else 'off'."""
    mode = parse_text_optimization_mode(requested)
    if mode is None and isinstance(metadata, dict) and "text_optimization" in metadata:
        mode = parse_text_optimization_mode(metadata.get("text_optimization"))
    if mode is None and owner_scope in DEFAULT_FOLLOWING_SCOPES:
        try:
            mode = parse_text_optimization_mode(default_mode)
        except ValueError:
            mode = None
    return mode or "off"
