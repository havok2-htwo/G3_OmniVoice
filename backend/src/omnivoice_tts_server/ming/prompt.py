# Vendored from inclusionAI/Ming-3.0-tts @ 9332fb9 (prompt_contract.py, tts_tasks.py), MIT license, see LICENSE in
# this directory. Trimmed to the dense (0.5B) layout and the plain TTS / zero-shot cloning task. The strings are part
# of the model's input protocol and must stay byte-identical to upstream.
# Also from tts_tasks.py: the structured control caption (EMOTION / BASIC) and the instruction normalization.
from __future__ import annotations

import json
from collections import OrderedDict

DEFAULT_TTS_PROMPT = 'Please generate speech based on the following description.\n'

DENSE_SYSTEM_PREFIX = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n'
DENSE_USER_PREFIX = '<|im_start|>user\n'
DENSE_USER_SUFFIX = '<|im_end|>\n'
DENSE_ASSISTANT_PREFIX = '<|im_start|>assistant\n'
DENSE_SPEAKER_START = '<|vision_start|>'
DENSE_SPEAKER_PAD = '<|vision_pad|>'
DENSE_SPEAKER_END = '<|vision_end|>\n'
DENSE_INSTRUCTION_EOS = '<|im_end|>'
TEXT_INPUT_PREFIX = ' Text input:\n'
AUDIO_START = '<audio>'
AUDIO_PATCH = '<audioPatch>'

DENSE_TOKEN_IDS = {
    '<|endoftext|>': 151643,
    '<|im_start|>': 151644,
    '<|im_end|>': 151645,
    '<|vision_start|>': 151652,
    '<|vision_end|>': 151653,
    '<|vision_pad|>': 151654,
    '<audioPatch>': 151705,
    '<audio>': 151706,
}

_BGM_MARKERS = ('Genre: ', 'Mood: ', 'Instrument: ', 'Theme: ', 'Duration: ')

TARGET_LANGUAGE_NAMES = {
    'zh': 'Chinese', 'zh-cn': 'Chinese', 'chinese': 'Chinese',
    'en': 'English', 'english': 'English',
    'ja': 'Japanese', 'japanese': 'Japanese',
    'ko': 'Korean', 'korean': 'Korean',
    'de': 'German', 'german': 'German',
    'fr': 'French', 'french': 'French',
    'es': 'Spanish', 'spanish': 'Spanish',
    'yue': 'Chinese', 'zh-yue': 'Chinese', 'cantonese': 'Chinese',
}


def normalize_language_id(language: str | None) -> str | None:
    """Map a request language ("de", "de-DE", "German", "auto", None) to a Ming language id or None."""
    if language is None:
        return None
    value = str(language).strip().lower().replace('_', '-')
    if not value or value in {'auto', 'none', 'unknown'}:
        return None
    if value in TARGET_LANGUAGE_NAMES:
        return value
    base = value.split('-', 1)[0]
    if base in TARGET_LANGUAGE_NAMES:
        return base
    return None


def build_tts_prompt(target_language: str | None = None) -> str:
    if target_language is None or not str(target_language).strip():
        return DEFAULT_TTS_PROMPT
    language = str(target_language).strip()
    name = TARGET_LANGUAGE_NAMES.get(language.lower(), language)
    return DEFAULT_TTS_PROMPT + f'Target language: {name}.\n'


BASIC_ATTRIBUTE_VALUES = {
    'speed': ('慢速', '中速', '快速'),
    'pitch': ('低', '中', '高'),
    'volume': ('低', '中', '高'),
}
EMOTION_VALUES = ('高兴', '悲伤', '愤怒', '恐惧', '厌恶', '惊讶', '中性')

_ZH_FIELDS = ('序号', '说话人', '方言', '风格', '语速', '基频', '音量', '情感')
_EN_FIELDS = ('id', 'speaker', 'dialect', 'style', 'speed', 'pitch', 'volume', 'emotion')
_BGM_FIELDS = ('Genre', 'Mood', 'Instrument', 'Theme', 'ENV', 'SNR')


def build_structured_instruction(*, dialect=None, style=None, speed=None, pitch=None, volume=None, emotion=None,
                                 ip=None, schema='zh') -> str:
    """The ordered assistant-side caption the checkpoint was fine-tuned on (EMOTION / BASIC / ... tasks)."""
    schema = str(schema).strip().lower()
    values = (1, 'speaker_1', dialect, style, speed, pitch, volume, emotion)
    if schema in {'zh', 'chinese'}:
        fields, ip_key = _ZH_FIELDS, 'IP'
    elif schema in {'en', 'english'}:
        fields, ip_key = _EN_FIELDS, 'ip'
    else:
        raise ValueError(f'Unsupported structured instruction schema: {schema!r}')
    item = OrderedDict(zip(fields, values))
    item['BGM'] = OrderedDict((key, None) for key in _BGM_FIELDS)
    item[ip_key] = ip
    return json.dumps(OrderedDict((('audio_sequence', [item]),)), ensure_ascii=False)


def normalize_instruction(instruction: str | None) -> str | None:
    """Blank means no instruction, not an empty assistant EOS prefix."""
    if instruction is None or not str(instruction).strip():
        return None
    return str(instruction).strip()


def validate_tokenizer(tokenizer) -> None:
    """Fail loudly when the tokenizer cannot express the trained layout (names AND ids must match)."""
    broken = {}
    for token in (AUDIO_START, AUDIO_PATCH, DENSE_SPEAKER_START, DENSE_SPEAKER_PAD, DENSE_INSTRUCTION_EOS):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == getattr(tokenizer, 'unk_token_id', None):
            broken[token] = token_id
            continue
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if encoded != [token_id]:
            broken[token] = encoded
    mismatched = {token: (want, tokenizer.convert_tokens_to_ids(token))
                  for token, want in DENSE_TOKEN_IDS.items() if tokenizer.convert_tokens_to_ids(token) != want}
    if broken or mismatched:
        raise ValueError(f'Ming tokenizer does not match the trained layout: broken={broken} mismatched={mismatched}')


def build_input_ids(tokenizer, prompt: str, text: str, *, num_speakers: int = 0, instruction: str | None = None,
                    prompt_text: str | None = None, prompt_latent_steps: int = 0) -> tuple[list[int], list[int], int]:
    """Token ids, the index of each speaker-embedding row, and the first prompt-latent row (-1 if none)."""
    def encode(value):
        return tokenizer.encode(value, add_special_tokens=False)

    speaker_block: list[int] = []
    for index in range(num_speakers):
        speaker_block.extend(encode(f'  speaker_{index + 1}:') + encode(DENSE_SPEAKER_START)
                             + encode(DENSE_SPEAKER_PAD) + encode(DENSE_SPEAKER_END))
    instruction_block: list[int] = []
    if instruction is not None:
        instruction_block = encode(instruction) + encode(DENSE_INSTRUCTION_EOS)
    prompt_text_block: list[int] = []
    latent_block: list[int] = []
    if prompt_latent_steps > 0 and prompt_text is not None:
        prompt_text_block = encode(prompt_text)
        latent_block = encode(AUDIO_PATCH) * prompt_latent_steps
    text_prefix = [] if all(m in text for m in _BGM_MARKERS) else encode(TEXT_INPUT_PREFIX)
    head = encode(DENSE_SYSTEM_PREFIX) + encode(DENSE_USER_PREFIX) + encode(prompt)
    turn_end = encode(DENSE_USER_SUFFIX) + encode(DENSE_ASSISTANT_PREFIX)
    input_ids = (head + speaker_block + text_prefix + prompt_text_block + encode(text) + turn_end
                 + instruction_block + encode(AUDIO_START))
    speaker_pad_id = tokenizer.convert_tokens_to_ids(DENSE_SPEAKER_PAD)
    speaker_start_id = tokenizer.convert_tokens_to_ids(DENSE_SPEAKER_START)
    speaker_positions = [pos + 1 for pos, tok in enumerate(input_ids) if tok == speaker_start_id]
    if len(speaker_positions) != num_speakers:
        raise RuntimeError(f'expected {num_speakers} speaker slots, found {len(speaker_positions)}')
    for pos in speaker_positions:
        if input_ids[pos] != speaker_pad_id:
            raise RuntimeError('speaker slot is not followed by its pad token')
    latent_position = len(input_ids) if latent_block else -1
    input_ids.extend(latent_block)
    return input_ids, speaker_positions, latent_position
