# Vendored from inclusionAI/Ming-3.0-tts @ 9332fb9 (frontend/text_segment_cut.py, frontend/toolkit.py,
# ming_tts.MingTTS.prepare_text), MIT license, see LICENSE in this directory. Trimmed to what fragmenting needs:
# position bookkeeping (calibrate_positions) is dropped because only the fragments are used, and the FST text
# normalization for zh/en (pynini) is not included -- the engine behaves like upstream ``text_normalization=False``.
from __future__ import annotations

import re
import string
import unicodedata

DEFAULT_MAX_SEMANTIC_LENGTH = 24


def is_chinese(text: str) -> bool:
    return bool(re.search(r'[一-鿿]', text))


def get_semantic_length(text: str) -> int:
    """One per CJK character / kana / hangul syllable plus one per Latin word."""
    text = unicodedata.normalize('NFC', text)
    cjk_char_count = len(re.findall(
        r'[一-鿿ぁ-ゖゝ-ゟ'
        r'ァ-ヺー-ヿ가-힣]', text))
    latin_letters = r'A-Za-zÀ-ÖØ-öø-ɏḀ-ỿ'
    latin_word_count = len(re.findall(rf'[{latin_letters}]+(?:[\'’][{latin_letters}]+)*', text))
    return cjk_char_count + latin_word_count


def has_valid_content(text: str) -> bool:
    punctuation_and_whitespace = string.punctuation + string.whitespace
    return any(char not in punctuation_and_whitespace for char in text)


def append_text_fragment(fragments, new_text, max_len, min_tail_length):
    new_text = new_text.lstrip('，,:;' + string.whitespace)
    if not has_valid_content(new_text):
        return fragments
    if not fragments:
        fragments.append(new_text)
        return fragments
    last_fragment = fragments[-1]
    if get_semantic_length(last_fragment) + get_semantic_length(new_text) <= max_len:
        if last_fragment.endswith(('。', '！', '？')) and get_semantic_length(new_text) < min_tail_length:
            fragments.append(new_text)
        else:
            separator = ''
            if not last_fragment.endswith(' ') and re.match(r'^[a-zA-Z0-9]', new_text):
                separator = ' '
            fragments[-1] += separator + new_text
    else:
        fragments.append(new_text)
    return fragments


def split_long_fragment(text_fragment, max_len):
    """Split along punctuation; a punctuation-free clause longer than ``max_len`` is kept whole."""
    if get_semantic_length(text_fragment) <= max_len:
        return [text_fragment]
    clause_pattern = r'([，、；。！？,;!?]|\s+[-•·])'
    merged_clauses = []
    for part in re.split(clause_pattern, text_fragment):
        if part == '':
            continue
        if merged_clauses and re.fullmatch(clause_pattern, part):
            merged_clauses[-1] += part
        else:
            merged_clauses.append(part)
    fragments = []
    current_fragment = ''
    for clause in merged_clauses:
        clause_len = get_semantic_length(clause)
        if clause_len > max_len:
            if current_fragment:
                fragments.append(current_fragment)
                current_fragment = ''
            fragments.append(clause)
            continue
        if get_semantic_length(current_fragment) + clause_len <= max_len:
            current_fragment += clause
        else:
            if current_fragment:
                fragments.append(current_fragment)
            current_fragment = clause
    if current_fragment:
        fragments.append(current_fragment)
    return fragments


def cut_text_by_semantic_length(text, max_semantic_length=DEFAULT_MAX_SEMANTIC_LENGTH, min_tail_length=5):
    """Upstream fragmenting: sentences, then clauses, merged up to ``max_semantic_length``.

    Note (upstream behaviour, kept): text longer than the limit gets ASCII .!?, rewritten to full-width 。！？，
    in the fragments; the model was trained on that.
    """
    if not has_valid_content(text):
        return []
    placeholder = '##DOT##'
    processed = re.sub(r'(\d)\.(\d)', r'\1' + placeholder + r'\2', text)
    for _ in range(3):
        processed = re.sub(r'([A-Z])\.([A-Z])', r'\1' + placeholder + r'\2', processed)
    processed = processed.replace('\n', ' ').replace('。，', '。')
    if get_semantic_length(processed) <= max_semantic_length:
        return [processed.replace(placeholder, '.')]

    normalized = processed.replace('.', '。').replace('!', '！').replace('?', '？').replace(',', '，')
    sentences = []
    current = ''
    for char in normalized:
        current += char
        if char in '。！？':
            sentence = current.strip()
            if sentence:
                sentences.append(sentence)
            current = ''
    if current:
        sentences.append(current.strip())
        if not sentences[-1].endswith(('。', '！', '？')):
            sentences[-1] += '。'

    result = []
    for sentence in sentences:
        clauses = []
        current_clause = ''
        for char in sentence:
            current_clause += char
            if char in '，;；':
                clause = current_clause.strip()
                if clause and has_valid_content(clause):
                    clauses.append(clause)
                elif clause and clauses:
                    clauses[-1] += clause
                current_clause = ''
        if current_clause:
            clause = current_clause.strip()
            if clause and has_valid_content(clause):
                clauses.append(clause)
            elif clause and clauses:
                clauses[-1] += clause
        i = 0
        while i < len(clauses):
            clause = clauses[i]
            clause_len = get_semantic_length(clause)
            if clause_len < min_tail_length and i + 1 < len(clauses):
                combined = clause + clauses[i + 1]
                if get_semantic_length(combined) <= max_semantic_length:
                    result = append_text_fragment(result, combined, max_semantic_length, min_tail_length)
                    i += 2
                    continue
            if clause_len > max_semantic_length:
                for frag in split_long_fragment(clause, max_semantic_length):
                    result = append_text_fragment(result, frag, max_semantic_length, min_tail_length)
            else:
                result = append_text_fragment(result, clause, max_semantic_length, min_tail_length)
            i += 1
    return [frag.replace(placeholder, '.') for frag in result]


# -- toolkit ----------------------------------------------------------------------------------------------------------

WEAK_TAIL_PUNCT = '，,、;；'


def strip_markdown_code(text: str) -> str:
    text = re.sub(r'``(.+?)``', r'\1', text)
    text = re.sub(r'`(.+?)`', r'\1', text)
    return text.replace('`', '')


def preprocess_text(text: str) -> str:
    return strip_markdown_code(text)


def add_period(text: str) -> str:
    """Terminate text that has no closing punctuation; a weak tail (comma etc.) becomes a full stop."""
    stripped = text.rstrip()
    if not stripped:
        return '.'
    last_char = stripped[-1]
    if last_char in WEAK_TAIL_PUNCT:
        has_chinese = bool(re.search(r'[一-鿿]', stripped))
        return stripped[:-1] + ('。' if has_chinese else '.')
    if not last_char.isalnum():
        return stripped
    is_cjk = '一' <= last_char <= '鿿' or '㐀' <= last_char <= '䶿'
    return stripped + ('。' if is_cjk else '.')


def preserve_multilingual_text(language_id) -> bool:
    """True for explicit non-zh/en languages: upstream passes their text through without add_period / TN."""
    if language_id is None or not str(language_id).strip():
        return False
    language = str(language_id).strip().lower().replace('_', '-')
    return language not in {'zh', 'zh-cn', 'chinese', 'en', 'english'}


def prepare_fragments(text: str, language_id: str | None,
                      max_length: int = DEFAULT_MAX_SEMANTIC_LENGTH) -> list[str]:
    """``MingTTS.prepare_text`` with ``text_normalization=False``."""
    text = str(text).strip()
    if not text:
        raise ValueError('text must not be empty')
    if preserve_multilingual_text(language_id):
        fragments = cut_text_by_semantic_length(preprocess_text(text).strip(), max_length)
    else:
        fragments = cut_text_by_semantic_length(add_period(preprocess_text(text)), max_length)
    prepared = [fragment for fragment in fragments if fragment.strip()]
    if not prepared:
        raise ValueError(f'text produced no synthesizable fragments: {text!r}')
    return prepared
