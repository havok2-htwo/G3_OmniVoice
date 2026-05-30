from __future__ import annotations

import re


SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?\u3002\uff01\uff1f\u2026])\s+|\n+')
SHORT_SENTENCE_MERGE_ENDINGS = ('!', '?', '.', '\uff01', '\uff1f', '\u3002')


def _hard_split(text: str, max_chars: int) -> list[str]:
    """Split text into pieces of at most max_chars, preferring whitespace boundaries.
    Guards a single over-long sequence (e.g. chunking off, or one giant sentence) so
    no generate() call exceeds the VRAM budget. max_chars <= 0 is a no-op."""
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    current = ''
    for word in text.split():
        if current and len(current) + 1 + len(word) > max_chars:
            chunks.append(current)
            current = word
        else:
            current = f'{current} {word}' if current else word
    if current:
        chunks.append(current)
    out: list[str] = []
    for chunk in chunks:
        while len(chunk) > max_chars:  # a single word longer than the limit
            out.append(chunk[:max_chars])
            chunk = chunk[max_chars:]
        if chunk:
            out.append(chunk)
    return out or [text[:max_chars]]


def split_sentences(
    text: str,
    *,
    enabled: bool,
    short_sentence_merge_max_chars: int = 30,
    following_sentence_merge_min_chars: int = 20,
    max_chars: int = 0,
) -> list[str]:
    cleaned = (text or '').strip()
    if not cleaned:
        return []
    if not enabled:
        return _hard_split(cleaned, max_chars)

    merge_limit = max(0, int(short_sentence_merge_max_chars))
    min_following_chars = max(0, int(following_sentence_merge_min_chars))

    def is_short_prompt(part: str) -> bool:
        return len(part) <= merge_limit and part.endswith(SHORT_SENTENCE_MERGE_ENDINGS)

    merged_parts: list[str] = []
    for line in [segment.strip() for segment in cleaned.splitlines() if segment.strip()]:
        parts = [part.strip() for part in SENTENCE_SPLIT_RE.split(line) if part.strip()]
        if not parts:
            continue

        line_parts: list[str] = []
        index = 0
        while index < len(parts):
            current = parts[index]
            if not is_short_prompt(current):
                line_parts.append(current)
                index += 1
                continue

            short_group = [current]
            index += 1
            while index < len(parts) and is_short_prompt(parts[index]):
                short_group.append(parts[index])
                index += 1

            if index < len(parts):
                next_part = parts[index]
                next_is_short = is_short_prompt(next_part)
                if (not next_is_short) or len(next_part) >= min_following_chars:
                    line_parts.append(' '.join(short_group + [next_part]))
                    index += 1
                    continue

            if len(short_group) >= 2:
                merged_short_group = ' '.join(short_group)
                if line_parts:
                    line_parts[-1] = f'{line_parts[-1]} {merged_short_group}'
                else:
                    line_parts.append(merged_short_group)
            else:
                line_parts.extend(short_group)

        merged_parts.extend(line_parts)

    result = merged_parts or [cleaned]
    if max_chars > 0:
        result = [piece for part in result for piece in _hard_split(part, max_chars)]
    return result


def chunk_pcm16le(pcm_bytes: bytes, *, sample_rate: int, chunk_ms: int) -> list[bytes]:
    chunk_size = max(2, int(sample_rate * max(chunk_ms, 20) / 1000) * 2)
    return [pcm_bytes[index : index + chunk_size] for index in range(0, len(pcm_bytes), chunk_size)]
