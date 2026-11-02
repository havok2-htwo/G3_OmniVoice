"""Text side of the Audio8 engines (no torch): fragments, frame cap, prompt token layout.

Audio8 samples its way through the text and now and then leaves a clause out of a long sentence (seen on a sentence of
178 characters). Fragments therefore stay bounded, and they are cut where a reader pauses anyway: sentence ends first,
then clause punctuation, then plain spaces.

The bound is a trade-off, measured with the engine on German text (Genesis3, 12 seeds per sentence). Whole sentences of
178 and 232 characters: nothing dropped. At 140 characters the 178-character sentence is cut mid-clause, its second
fragment opens with a bare verb ("lud das Expose ...") and is misread in most runs ("Luthers Expose"), which costs more
than the drop it guards against. So a sentence stays whole up to 200 characters and only longer ones are cut.
"""
from __future__ import annotations

import re
from typing import Any

DEFAULT_FRAGMENT_MAX_CHARS = 200
MIN_FRAGMENT_CHARS = 24

# Code frames per character. 2048 samples at 44.1 kHz are 46 ms; German speech runs about 1.5 frames per character.
# The cap only has to end a row that never reaches EOS, so it leaves room for slow delivery and for what a character
# can expand to when read out (a digit, one CJK syllable).
FRAMES_PER_CHAR = 4
FRAMES_PER_DIGIT = 16
FRAMES_PER_CJK = 10
MIN_FRAME_CAP = 64
EXPECTED_FRAMES_PER_CHAR = 3

_SENTENCE = re.compile(r'(?<=[.!?…。！？])\s+|(?<=[.!?…。！？]["\'”’)\]])\s+')
# Latin marks need a space behind them: '3,5' and '12:30' are not clauses. CJK writes none.
_CLAUSE = re.compile(r'(?<=[,;:])\s+|(?<=[，；：、])\s*|\s+(?=[–—-]\s)')
_CJK_MARKS = '，；：、'
_CJK = re.compile(r'[぀-ヿ㐀-鿿가-힣]')
_DIGIT = re.compile(r'\d')
_CONTROL = re.compile(r'<\|[^|<>]*\|>')


def clean_text(text: str) -> str:
    """Whitespace collapsed like the processor does, and chat control tokens out of the user's text: encoded with
    special tokens on, '<|im_end|>' in a request would end the prompt's user turn."""
    return ' '.join(_CONTROL.sub(' ', str(text)).split())


def _merge(pieces: list[str], max_chars: int) -> list[str]:
    out: list[str] = []
    for piece in pieces:
        joint = '' if out and out[-1][-1] in _CJK_MARKS else ' '
        if out and len(out[-1]) + len(joint) + len(piece) <= max_chars:
            out[-1] = f'{out[-1]}{joint}{piece}'
        else:
            out.append(piece)
    return out


def _split_words(text: str, max_chars: int) -> list[str]:
    pieces, line = [], ''
    for word in text.split(' '):
        while len(word) > max_chars:  # no space to cut at
            if line:
                pieces.append(line)
                line = ''
            pieces.append(word[:max_chars])
            word = word[max_chars:]
        if line and len(line) + 1 + len(word) > max_chars:
            pieces.append(line)
            line = word
        else:
            line = f'{line} {word}' if line else word
    if line:
        pieces.append(line)
    return pieces


def _split_sentence(sentence: str, max_chars: int) -> list[str]:
    if len(sentence) <= max_chars:
        return [sentence]
    clauses = [clause.strip() for clause in _CLAUSE.split(sentence) if clause.strip()]
    pieces: list[str] = []
    for clause in clauses:
        pieces.extend([clause] if len(clause) <= max_chars else _split_words(clause, max_chars))
    return _merge(pieces, max_chars)


def split_fragments(text: str, max_chars: int = DEFAULT_FRAGMENT_MAX_CHARS) -> list[str]:
    """Text pieces of at most ``max_chars`` characters, in reading order, nothing dropped."""
    max_chars = max(MIN_FRAGMENT_CHARS, int(max_chars))
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError('text must not be empty')
    if len(cleaned) <= max_chars:
        return [cleaned]
    pieces: list[str] = []
    for sentence in _SENTENCE.split(cleaned):
        if sentence and sentence.strip():
            pieces.extend(_split_sentence(sentence.strip(), max_chars))
    return _merge(pieces, max_chars)


def frame_cap(text: str) -> int:
    """Most code frames a text can take: ends a row that fails to reach EOS."""
    digits = len(_DIGIT.findall(text))
    cjk = len(_CJK.findall(text))
    other = len(text) - digits - cjk
    return MIN_FRAME_CAP + FRAMES_PER_CHAR * other + FRAMES_PER_DIGIT * digits + FRAMES_PER_CJK * cjk


def _encode(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def prefix_ids(tokenizer: Any, reference_text: str | None) -> list[int]:
    """Tokens before the reference codes (ArkttsProcessor._prompt_segments). Without a reference the same system turn
    is closed in the suffix, so the fixed head of the prompt can be cached either way."""
    if reference_text is None:
        return _encode(tokenizer, '<|im_start|>system\n') + _encode(tokenizer, 'convert the provided text to speech')
    cleaned = ' '.join(str(reference_text).split())
    if not re.search(r'<\|speaker:\d+\|>', cleaned):
        cleaned = f'<|speaker:0|>{cleaned}'
    ids: list[int] = []
    for part in ('<|im_start|>system\n', 'convert the provided text to speech reference to the following:\n\nText:\n',
                 cleaned, '\n\nSpeech:\n'):
        ids += _encode(tokenizer, part)
    return ids


def user_turn_ids(tokenizer: Any) -> list[int]:
    """The start of the suffix, the same for every text: it can be cached with the voice."""
    return _encode(tokenizer, '<|im_end|>\n') + _encode(tokenizer, '<|im_start|>user\n')


def suffix_ids(tokenizer: Any, text: str) -> list[int]:
    """Tokens after the reference codes: the target text and the turn that starts the voice."""
    ids = user_turn_ids(tokenizer)
    for part in (text, '<|im_end|>\n', '<|im_start|>assistant\n<|voice|>'):
        ids += _encode(tokenizer, part)
    return ids
