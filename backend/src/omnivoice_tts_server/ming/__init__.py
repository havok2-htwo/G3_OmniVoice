"""Ming-3.0-tts 0.5B (inclusionAI, MIT) as an in-process engine: vendored model code plus our batched loop.

Upstream: https://github.com/inclusionAI/Ming-3.0-tts @ 9332fb9 (see LICENSE in this directory).
Heavy imports (torch, transformers) happen when ``engine`` is imported, not here.
"""

MING_HF_ID = 'inclusionAI/Ming-3.0-tts-0.5B'

__all__ = ['MING_HF_ID', 'MingEngine', 'MingStream', 'MingVoicePrompt']


def __getattr__(name):
    if name in ('MingEngine', 'MingStream', 'MingVoicePrompt'):
        from . import engine

        return getattr(engine, name)
    raise AttributeError(name)
