from omnivoice_tts_server.domain.models import WerBenchmarkCreateRequest
from omnivoice_tts_server.services_v2 import WerBenchmarkService


def test_explicit_wer_seed_values_override_seed_range() -> None:
    payload = WerBenchmarkCreateRequest(
        random_seed=5,
        seed_range=100,
        seed_values=[8888, 8890, 2912],
    )

    assert WerBenchmarkService._seed_values(payload) == [8888, 8890, 2912]


def test_explicit_wer_seed_values_are_deduped_in_order() -> None:
    payload = WerBenchmarkCreateRequest(
        seed_values=[8888, 8890, 8888, 2912],
    )

    assert WerBenchmarkService._seed_values(payload) == [8888, 8890, 2912]


def test_wer_seed_range_still_works_without_explicit_values() -> None:
    payload = WerBenchmarkCreateRequest(random_seed=5, seed_range=2)

    assert WerBenchmarkService._seed_values(payload) == [5, 6, 7]
