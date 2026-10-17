from __future__ import annotations

from omnivoice_tts_server.config import Settings
from omnivoice_tts_server.domain.models import WerBenchmarkItemResult
from omnivoice_tts_server.domain.state import InMemoryStore, utcnow
from omnivoice_tts_server.services_v2 import EventHub, WerBenchmarkService


def _item(index: int, seed: int, wer: float | None, success: bool = True) -> WerBenchmarkItemResult:
    return WerBenchmarkItemResult(index=index, seed=seed, success=success, source_text='x', transcript='x', wer=wer)


def _service(tmp_path) -> tuple[WerBenchmarkService, InMemoryStore]:
    store = InMemoryStore(max_queue_size=8)
    svc = WerBenchmarkService(store, object(), object(), EventHub(store), Settings(data_dir=tmp_path))
    return svc, store


def test_deviations_are_only_the_failing_samples_worst_first_and_paged(tmp_path) -> None:
    svc, store = _service(tmp_path)
    results = [_item(1, 1, 0.0), _item(2, 1, 0.2), _item(3, 2, 0.5), _item(4, 2, 0.1), _item(5, 2, None, success=False)]
    store.wer_benchmark_runs['wer_t'] = {'run_id': 'wer_t', 'name': 'n', 'status': 'completed', 'created_at': utcnow(), 'results': results, 'count': 5, 'seed_values': [1, 2]}

    total, first = svc.deviations('wer_t', offset=0, limit=2)
    assert total == 3
    assert [item.wer for item in first] == [0.5, 0.2]

    total, rest = svc.deviations('wer_t', offset=2, limit=2)
    assert total == 3
    assert [item.wer for item in rest] == [0.1]


def test_run_list_leaves_out_the_per_sample_results(tmp_path) -> None:
    svc, store = _service(tmp_path)
    results = [_item(i, 1, 0.1) for i in range(1, 50)]
    store.wer_benchmark_runs['wer_t'] = {'run_id': 'wer_t', 'name': 'n', 'status': 'running', 'created_at': utcnow(), 'results': results, 'count': 49, 'seed_values': [1]}

    listed = __import__('asyncio').run(svc.list_runs())[0]
    assert listed.results == []
    assert listed.results_count == 49
    assert listed.summary.completed == 49
