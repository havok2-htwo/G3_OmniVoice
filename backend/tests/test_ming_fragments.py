from omnivoice_tts_server.config import Settings
from omnivoice_tts_server.ming.text import prepare_fragments

SENTENCE = ('Also, ich sage dir das jetzt ganz ehrlich, ich habe es satt, dass du immer zu spät kommst, '
            'und ich will, dass das aufhört, sofort, verstanden?')


def test_the_default_keeps_upstreams_fragments():
    assert Settings.model_fields['ming_fragment_max_chars'].default == 24
    assert len(prepare_fragments(SENTENCE, 'de', 24)) == 2


def test_a_higher_limit_keeps_a_sentence_whole():
    assert prepare_fragments(SENTENCE, 'de', 200) == [SENTENCE]


def test_text_over_the_limit_is_still_split():
    assert len(prepare_fragments('Das ist der erste Satz. ' * 60, 'de', 200)) > 1
