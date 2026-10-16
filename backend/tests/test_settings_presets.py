"""Settings presets: built-in GPU tiers, card matching for the two-card tier, saved presets and the admin routes."""
from __future__ import annotations

import json
import os

import pytest

from omnivoice_tts_server import presets
from omnivoice_tts_server.config import RUNTIME_SETTINGS_FIELDS, Settings
from omnivoice_tts_server.domain.models import ServerSettingsUpdateRequest

from test_app import make_client

TIER_NAMES = ['32GB VRAM', '24GB VRAM', '24+8 VRAM']
NOW = '2026-10-16T12:00:00+00:00'


def _cards(monkeypatch, *cards_mb: int) -> None:
    """Pretend these CUDA cards (total MiB, cuda index = position) are installed; never touches the GPU."""
    monkeypatch.setattr(presets, '_detected_cards_mb', lambda: list(enumerate(cards_mb)))


def _tier(data_dir, name: str) -> dict:
    return next(preset for preset in presets.list_presets(data_dir) if preset['name'] == name)


def test_cuda_device_order_is_pci_bus_id() -> None:
    assert os.environ.get('CUDA_DEVICE_ORDER') == 'PCI_BUS_ID'


def test_preset_keys_are_runtime_settings() -> None:
    assert set(presets.PRESET_KEYS) <= RUNTIME_SETTINGS_FIELDS
    assert set(presets.PRESET_KEYS) <= set(ServerSettingsUpdateRequest.model_fields)
    for secret_or_path in ('models_root_dir', 'whisper_base_url', 'vllm_base_url', 'default_voice', 'data_dir'):
        assert secret_or_path not in presets.PRESET_KEYS


def test_builtin_tiers_come_first_in_tier_order(monkeypatch, tmp_path) -> None:
    _cards(monkeypatch, 32607)
    presets.save_preset(tmp_path, 'aaa first by name', {'max_batch_size': 4}, now_iso=NOW)
    listed = presets.list_presets(tmp_path)
    assert [preset['name'] for preset in listed] == [*TIER_NAMES, 'aaa first by name']
    assert [preset['built_in'] for preset in listed] == [True, True, True, False]
    assert '8+24 VRAM' not in presets.BUILTIN_PRESETS


@pytest.mark.parametrize('name', TIER_NAMES)
def test_builtin_values_are_valid_settings(name) -> None:
    values = presets.BUILTIN_PRESETS[name]['values']
    assert set(values) <= set(presets.PRESET_KEYS)
    assert 'preferred_device' not in values, 'built-ins get the GPU only from card matching'
    assert presets.preset_values(values) == values, 'a built-in value would be changed or rejected by the settings PUT'


@pytest.mark.parametrize(
    ('name', 'batch', 'budget'), [('32GB VRAM', 8, 7000), ('24GB VRAM', 4, 5000), ('24+8 VRAM', 4, 5000)]
)
def test_builtin_tiers_carry_the_measured_values(name, batch, budget) -> None:
    values = presets.BUILTIN_PRESETS[name]['values']
    assert (values['max_batch_size'], values['max_parallel_requests'], values['vram_budget_mb']) == (batch, batch, budget)
    assert values['batch_wait_ms'] == 30
    assert values['cuda_memory_trim_after_batch'] is True
    assert (values['num_step'], values['compile_model'], values['torch_dtype']) == (20, True, 'bfloat16')
    assert values['enable_flashinfer'] is False
    assert 'RTX 5090' in presets.BUILTIN_PRESETS[name]['description'], 'the description names where the VRAM was measured'


def test_untuned_tier_keys_stay_at_the_code_defaults(tmp_path) -> None:
    defaults = Settings(data_dir=tmp_path)
    tuned = {'max_batch_size', 'max_parallel_requests', 'vram_budget_mb', 'batch_wait_ms'}
    for name in TIER_NAMES:
        for key, value in presets.BUILTIN_PRESETS[name]['values'].items():
            if key not in tuned:
                assert value == getattr(defaults, key), f'{name}: {key}'


@pytest.mark.parametrize(('cards', 'expected'), [((24576, 8192), 'cuda:0'), ((8192, 24576), 'cuda:1'), ((8192, 24268), 'cuda:1')])
def test_two_card_tier_picks_the_24gb_card_wherever_it_sits(monkeypatch, tmp_path, cards, expected) -> None:
    _cards(monkeypatch, *cards)
    tier = _tier(tmp_path, '24+8 VRAM')
    assert tier['values']['preferred_device'] == expected
    assert tier['target_vram_gb'] == 24
    assert tier['note'] is None


@pytest.mark.parametrize('cards', [(32607,), ()])
def test_two_card_tier_without_a_matching_card_leaves_the_gpu_alone(monkeypatch, tmp_path, cards) -> None:
    _cards(monkeypatch, *cards)
    tier = _tier(tmp_path, '24+8 VRAM')
    assert 'preferred_device' not in tier['values']
    assert 'Keine 24-GB-Karte' in tier['note']


def test_single_card_tiers_never_set_the_gpu(monkeypatch, tmp_path) -> None:
    _cards(monkeypatch, 24576, 8192)
    for name in ('32GB VRAM', '24GB VRAM'):
        tier = _tier(tmp_path, name)
        assert 'preferred_device' not in tier['values']
        assert tier['note'] is None


def test_builtin_values_are_not_mutated_by_listing(monkeypatch, tmp_path) -> None:
    _cards(monkeypatch, 24576)
    presets.list_presets(tmp_path)
    assert 'preferred_device' not in presets.BUILTIN_PRESETS['24+8 VRAM']['values']


def test_saved_preset_keeps_only_preset_keys_normalised(monkeypatch, tmp_path) -> None:
    _cards(monkeypatch)
    saved = presets.save_preset(
        tmp_path,
        '  my   tier ',
        {
            'max_batch_size': 4,
            'torch_dtype': 'BF16',
            'preferred_device': 'CUDA:1',
            'attention_implementation': 'SDPA',
            'num_step': None,
            'models_root_dir': 'D:\\models',
            'vllm_base_url': 'http://secret-host:8000',
            'default_voice': 'Sonja',
            'nonsense': 1,
        },
        now_iso=NOW,
    )
    expected = {'preferred_device': 'cuda:1', 'attention_implementation': 'sdpa', 'torch_dtype': 'bfloat16', 'max_batch_size': 4}
    assert saved['name'] == 'my tier'
    assert saved['values'] == expected
    listed = [preset for preset in presets.list_presets(tmp_path) if not preset['built_in']]
    assert [(preset['name'], preset['values']) for preset in listed] == [('my tier', expected)]
    raw = (tmp_path / presets.PRESETS_FILE).read_text(encoding='utf-8')
    assert 'secret-host' not in raw and 'models' not in json.loads(raw)['my tier']['values']


def test_saving_again_keeps_created_at(tmp_path) -> None:
    presets.save_preset(tmp_path, 'lab', {'max_batch_size': 4}, now_iso=NOW)
    again = presets.save_preset(tmp_path, 'lab', {'max_batch_size': 6}, now_iso='2030-01-01T00:00:00+00:00')
    assert again['created_at'] == NOW
    assert again['values'] == {'max_batch_size': 6}


@pytest.mark.parametrize(
    'values',
    [{'max_batch_size': 0}, {'vram_budget_mb': -1}, {'torch_dtype': 'int4'}, {'preferred_device': 'gpu1'}, {'attention_implementation': 'magic'}],
)
def test_invalid_values_are_rejected_like_the_settings_put(tmp_path, values) -> None:
    key = next(iter(values))
    with pytest.raises(ValueError, match=key):
        presets.save_preset(tmp_path, 'bad', values, now_iso=NOW)
    assert not (tmp_path / presets.PRESETS_FILE).exists()


def test_stored_file_with_bad_entries_is_filtered_on_load(monkeypatch, tmp_path) -> None:
    _cards(monkeypatch)
    (tmp_path / presets.PRESETS_FILE).write_text(
        json.dumps(
            {
                'old': {'name': 'old', 'values': {'max_batch_size': 0, 'batch_wait_ms': 5, 'vllm_model': 'x'}, 'created_at': NOW},
                '32GB VRAM': {'name': '32GB VRAM', 'values': {'max_batch_size': 2}},
                'broken': 'nope',
            }
        ),
        encoding='utf-8',
    )
    saved = [preset for preset in presets.list_presets(tmp_path) if not preset['built_in']]
    assert [(preset['name'], preset['values']) for preset in saved] == [('old', {'batch_wait_ms': 5})]
    assert _tier(tmp_path, '32GB VRAM')['values'] == presets.BUILTIN_PRESETS['32GB VRAM']['values']


def test_builtin_names_are_protected(tmp_path) -> None:
    with pytest.raises(ValueError):
        presets.save_preset(tmp_path, '32GB VRAM', {'max_batch_size': 4}, now_iso=NOW)
    with pytest.raises(ValueError):
        presets.save_preset(tmp_path, ' 24+8   VRAM ', {'max_batch_size': 4}, now_iso=NOW)
    with pytest.raises(ValueError):
        presets.delete_preset(tmp_path, '24GB VRAM')


def test_delete_saved_preset(monkeypatch, tmp_path) -> None:
    _cards(monkeypatch)
    presets.save_preset(tmp_path, 'tmp', {'max_batch_size': 4}, now_iso=NOW)
    assert presets.delete_preset(tmp_path, 'tmp') is True
    assert presets.delete_preset(tmp_path, 'tmp') is False
    assert all(preset['built_in'] for preset in presets.list_presets(tmp_path))
    assert not list(tmp_path.glob('*.tmp')), 'atomic write left a temp file behind'


def test_empty_long_or_keyless_presets_are_rejected(tmp_path) -> None:
    with pytest.raises(ValueError):
        presets.save_preset(tmp_path, '   ', {'max_batch_size': 4}, now_iso=NOW)
    with pytest.raises(ValueError):
        presets.save_preset(tmp_path, 'x' * (presets.PRESET_NAME_MAX_LENGTH + 1), {'max_batch_size': 4}, now_iso=NOW)
    with pytest.raises(ValueError):
        presets.save_preset(tmp_path, 'only unknown keys', {'nonsense': 1, 'theme': 'onyx'}, now_iso=NOW)


def test_admin_preset_routes(monkeypatch) -> None:
    _cards(monkeypatch, 24576, 8192)
    client = make_client()
    try:
        listed = client.get('/api/admin/settings/presets').json()['presets']
        assert [preset['name'] for preset in listed] == TIER_NAMES
        assert listed[2]['values']['preferred_device'] == 'cuda:0'
        assert listed[0]['built_in'] is True and listed[0]['description']

        # The dashboard sends the whole settings draft; only preset keys are kept.
        draft = client.get('/api/admin/settings').json()
        response = client.put('/api/admin/settings/presets/lab', json={'values': {**draft, 'max_batch_size': 3}})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body['built_in'] is False
        assert set(body['values']) <= set(presets.PRESET_KEYS)
        assert body['values']['max_batch_size'] == 3
        assert 'vllm_base_url' not in body['values'] and 'model_directory' not in body['values']

        # A loaded preset applies through the normal settings PUT unchanged.
        applied = client.put('/api/admin/settings', json=body['values'])
        assert applied.status_code == 200, applied.text
        assert applied.json()['max_batch_size'] == 3

        assert client.put('/api/admin/settings/presets/lab', json={'values': {'max_batch_size': 0}}).status_code == 400
        assert client.put('/api/admin/settings/presets/32GB VRAM', json={'values': {'max_batch_size': 4}}).status_code == 400
        assert client.delete('/api/admin/settings/presets/24GB VRAM').status_code == 400
        assert client.delete('/api/admin/settings/presets/lab').status_code == 200
        assert client.delete('/api/admin/settings/presets/lab').status_code == 404
    finally:
        client.__exit__(None, None, None)
        client._temp_data_dir.cleanup()
