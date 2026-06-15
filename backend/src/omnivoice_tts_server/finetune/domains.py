"""Persisted finetune domain list (topics) with their generated sentences.

Mirrors presets.py: a single JSON file under data/finetune/, guarded by an RLock.
Shape: { domain_id: {domain_id, name, description, created_at, sentences: [...] } }.

A "domain" is a steering topic for sentence generation, e.g. name="Saetze mit GmbH",
description="Deutsche Saetze, in denen die Abkuerzung GmbH vorkommt." Sentences are
stored per domain and deduplicated on insert.
"""
from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any
from uuid import uuid4

_LOCK = threading.RLock()


def _path(data_dir: Path) -> Path:
    return data_dir / 'finetune' / 'domains.json'


def _read(data_dir: Path) -> dict[str, dict[str, Any]]:
    path = _path(data_dir)
    if not path.exists():
        return {}
    try:
        import json

        data = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    domains: dict[str, dict[str, Any]] = {}
    for domain_id, entry in data.items():
        if not isinstance(entry, dict):
            continue
        sentences = entry.get('sentences')
        domains[str(domain_id)] = {
            'domain_id': str(entry.get('domain_id') or domain_id),
            'name': str(entry.get('name') or domain_id),
            'description': str(entry.get('description') or ''),
            'created_at': entry.get('created_at'),
            'sentences': [str(s) for s in sentences] if isinstance(sentences, list) else [],
        }
    return domains


def _write(data_dir: Path, domains: dict[str, dict[str, Any]]) -> None:
    import json

    path = _path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(domains, indent=2, ensure_ascii=False), encoding='utf-8')


def normalize_sentence_key(sentence: str) -> str:
    """Same normalization the WER sentence generator uses for dedup (services_v2.py)."""
    return re.sub(r'\s+', ' ', sentence or '').strip().lower()


def load_domains(data_dir: Path) -> dict[str, dict[str, Any]]:
    with _LOCK:
        return _read(data_dir)


def list_domains(data_dir: Path) -> list[dict[str, Any]]:
    domains = load_domains(data_dir)
    return sorted(domains.values(), key=lambda item: (item.get('created_at') or '', item.get('name') or ''))


def get_domain(data_dir: Path, domain_id: str) -> dict[str, Any] | None:
    return load_domains(data_dir).get(domain_id)


def create_domain(data_dir: Path, name: str, description: str, *, now_iso: str) -> dict[str, Any]:
    clean = (name or '').strip()
    if not clean:
        raise ValueError('Domain name is required.')
    with _LOCK:
        domains = _read(data_dir)
        # Reject case-insensitive name duplicates so the LLM "exclude existing" loop and
        # manual entry converge on a unique set.
        lowered = {entry['name'].strip().lower() for entry in domains.values()}
        if clean.lower() in lowered:
            raise ValueError(f'A domain named "{clean}" already exists.')
        domain_id = f'dom_{uuid4().hex[:12]}'
        entry = {
            'domain_id': domain_id,
            'name': clean,
            'description': (description or '').strip(),
            'created_at': now_iso,
            'sentences': [],
        }
        domains[domain_id] = entry
        _write(data_dir, domains)
    return entry


def update_domain(data_dir: Path, domain_id: str, *, name: str | None, description: str | None) -> dict[str, Any]:
    with _LOCK:
        domains = _read(data_dir)
        entry = domains.get(domain_id)
        if entry is None:
            raise KeyError(domain_id)
        if name is not None and name.strip():
            entry['name'] = name.strip()
        if description is not None:
            entry['description'] = description.strip()
        domains[domain_id] = entry
        _write(data_dir, domains)
    return entry


def delete_domain(data_dir: Path, domain_id: str) -> bool:
    with _LOCK:
        domains = _read(data_dir)
        if domain_id not in domains:
            return False
        domains.pop(domain_id, None)
        _write(data_dir, domains)
    return True


def add_sentences(data_dir: Path, domain_id: str, sentences: list[str]) -> tuple[int, int, int]:
    """Append new, non-duplicate sentences to a domain. Returns (added, skipped, total)."""
    with _LOCK:
        domains = _read(data_dir)
        entry = domains.get(domain_id)
        if entry is None:
            raise KeyError(domain_id)
        existing = list(entry.get('sentences') or [])
        seen = {normalize_sentence_key(s) for s in existing}
        added = 0
        skipped = 0
        for sentence in sentences:
            clean = re.sub(r'\s+', ' ', str(sentence or '')).strip()
            key = normalize_sentence_key(clean)
            if not key or key in seen:
                skipped += 1
                continue
            existing.append(clean)
            seen.add(key)
            added += 1
        entry['sentences'] = existing
        domains[domain_id] = entry
        _write(data_dir, domains)
    return added, skipped, len(existing)
