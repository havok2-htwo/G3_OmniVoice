from __future__ import annotations

import difflib
import re

DEFAULT_VOICE_DESIGN_INSTRUCT = 'female, young adult, moderate pitch'

VOICE_DESIGN_CATEGORIES: dict[str, tuple[str, ...]] = {
    'gender': ('male', 'female', '男', '女'),
    'age': ('child', 'teenager', 'young adult', 'middle-aged', 'elderly', '儿童', '少年', '青年', '中年', '老年'),
    'pitch': (
        'very low pitch',
        'low pitch',
        'moderate pitch',
        'high pitch',
        'very high pitch',
        '极低音调',
        '低音调',
        '中音调',
        '高音调',
        '极高音调',
    ),
    'style': ('whisper', '耳语'),
    'accent': (
        'american accent',
        'australian accent',
        'british accent',
        'canadian accent',
        'chinese accent',
        'indian accent',
        'japanese accent',
        'korean accent',
        'portuguese accent',
        'russian accent',
    ),
    'dialect': (
        '河南话',
        '陕西话',
        '四川话',
        '贵州话',
        '云南话',
        '桂林话',
        '济南话',
        '石家庄话',
        '甘肃话',
        '宁夏话',
        '青岛话',
        '东北话',
    ),
}

VOICE_DESIGN_VALID_ENGLISH = tuple(
    item for category in VOICE_DESIGN_CATEGORIES.values() for item in category
)
_VOICE_DESIGN_VALID_SET = set(VOICE_DESIGN_VALID_ENGLISH)


def normalize_voice_design_instruct(instruct: str | None) -> str:
    value = (instruct or '').strip()
    if not value:
        return DEFAULT_VOICE_DESIGN_INSTRUCT

    raw_items = [item.strip() for item in re.split(r'\s*[,\uFF0C]\s*', value) if item.strip()]
    normalized: list[str] = []
    unknown: list[tuple[str, str | None]] = []

    for item in raw_items:
        lowered = item.lower()
        if lowered in _VOICE_DESIGN_VALID_SET:
            if lowered not in normalized:
                normalized.append(lowered)
            continue
        suggestion = difflib.get_close_matches(lowered, VOICE_DESIGN_VALID_ENGLISH, n=1, cutoff=0.6)
        unknown.append((item, suggestion[0] if suggestion else None))

    if unknown:
        details = '; '.join(
            f'{item} -> maybe {suggestion}' if suggestion else item
            for item, suggestion in unknown
        )
        raise RuntimeError(
            'VoiceDesign akzeptiert nur feste Tags, keine freien Beschreibungen. '
            f'Ungueltig: {details}. Erlaubt: {", ".join(VOICE_DESIGN_VALID_ENGLISH)}.'
        )

    has_accent = any(item in VOICE_DESIGN_CATEGORIES['accent'] for item in normalized)
    has_dialect = any(item in VOICE_DESIGN_CATEGORIES['dialect'] for item in normalized)
    if has_accent and has_dialect:
        raise RuntimeError('VoiceDesign kann English Accent und Chinese Dialect nicht gleichzeitig nutzen.')

    conflicts: list[str] = []
    for category, items in VOICE_DESIGN_CATEGORIES.items():
        hits = [item for item in normalized if item in items]
        if len(hits) > 1:
            conflicts.append(f'{category}: {" vs ".join(hits)}')
    if conflicts:
        raise RuntimeError(
            'VoiceDesign erlaubt pro Kategorie nur einen Wert. '
            f'Konflikte: {"; ".join(conflicts)}.'
        )

    return ', '.join(normalized) if normalized else DEFAULT_VOICE_DESIGN_INSTRUCT
