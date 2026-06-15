from __future__ import annotations

import io
import json
import tempfile
import time
import wave
from pathlib import Path

from fastapi.testclient import TestClient

from omnivoice_tts_server.app import create_app
from omnivoice_tts_server.config import Settings

def make_client(**overrides) -> TestClient:
    # Hermetic: both data_dir and models_root_dir live in temp dirs so finetune tests
    # (which write train clips, datasets, and promoted model dirs) never touch the repo.
    temp_data_dir = tempfile.TemporaryDirectory()
    temp_models_dir = tempfile.TemporaryDirectory()
    app = create_app(
        Settings(
            admin_api_key='test-admin-key',
            runtime_backend='mock',
            models_root_dir=Path(temp_models_dir.name),
            data_dir=Path(temp_data_dir.name),
            **overrides,
        )
    )
    client = TestClient(app)
    client.__enter__()
    client._temp_data_dir = temp_data_dir
    client._temp_models_dir = temp_models_dir
    return client


def auth_headers() -> dict[str, str]:
    return {'X-Admin-Key': 'test-admin-key'}


def make_wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24_000)
        wav_file.writeframes(b'\x00\x00' * 2400)
    return buffer.getvalue()


def _wait_run(client: TestClient, run_id: str, timeout: float = 8.0) -> dict:
    deadline = time.time() + timeout
    payload: dict = {}
    while time.time() < deadline:
        payload = client.get('/api/admin/finetune/generate/status', headers=auth_headers(), params={'run_id': run_id}).json()
        if payload['status'] != 'running':
            return payload
        time.sleep(0.05)
    raise AssertionError(f'datagen run did not finish: {payload}')


def _create_domain_with_sentences(client: TestClient, name: str, count: int) -> str:
    domain = client.post('/api/admin/finetune/domains', json={'name': name, 'description': 'x'}, headers=auth_headers()).json()
    domain_id = domain['domain_id']
    client.post(f'/api/admin/finetune/domains/{domain_id}/sentences', json={'count': count}, headers=auth_headers())
    return domain_id


def test_domain_crud_and_duplicate_rejection() -> None:
    client = make_client()
    r = client.post('/api/admin/finetune/domains', json={'name': 'GmbH Faelle', 'description': 'abk'}, headers=auth_headers())
    assert r.status_code == 200
    domain_id = r.json()['domain_id']

    # Case-insensitive duplicate name is rejected.
    dup = client.post('/api/admin/finetune/domains', json={'name': 'gmbh faelle'}, headers=auth_headers())
    assert dup.status_code == 400

    # Update + list + delete.
    upd = client.put(f'/api/admin/finetune/domains/{domain_id}', json={'description': 'neu'}, headers=auth_headers())
    assert upd.status_code == 200 and upd.json()['description'] == 'neu'
    listing = client.get('/api/admin/finetune/domains', headers=auth_headers()).json()
    assert any(d['domain_id'] == domain_id for d in listing['domains'])
    assert client.delete(f'/api/admin/finetune/domains/{domain_id}', headers=auth_headers()).status_code == 200
    assert client.get(f'/api/admin/finetune/domains/{domain_id}', headers=auth_headers()).status_code == 404


def test_sentence_generation_appends_to_domain() -> None:
    client = make_client()
    domain_id = _create_domain_with_sentences(client, 'Zahlen', 5)
    assert client.get(f'/api/admin/finetune/domains/{domain_id}', headers=auth_headers()).json()['sentence_count'] == 5
    again = client.post(f'/api/admin/finetune/domains/{domain_id}/sentences', json={'count': 3}, headers=auth_headers()).json()
    assert again['added'] == 3 and again['sentence_count'] == 8


def test_add_sentences_dedups_with_normalization() -> None:
    # Storage-level dedup is what guards every generation path (whitespace + case insensitive).
    client = make_client()
    from omnivoice_tts_server.finetune import domains as dstore

    data_dir = Path(client._temp_data_dir.name)
    domain = dstore.create_domain(data_dir, 'Dedup', '', now_iso='2026-01-01T00:00:00+00:00')
    added, skipped, total = dstore.add_sentences(
        data_dir, domain['domain_id'], ['Hallo Welt', 'hallo   welt', '  HALLO WELT', 'Neuer Satz', ''],
    )
    assert added == 2  # 'Hallo Welt' + 'Neuer Satz'; the two variants + empty are skipped
    assert skipped == 3 and total == 2


def test_datagen_auto_mode_creates_clips_and_delete_removes_pair() -> None:
    client = make_client()
    domain_id = _create_domain_with_sentences(client, 'Auto Domain', 4)
    start = client.post('/api/admin/finetune/generate', json={
        'domain_ids': [domain_id], 'voice_mode': 'auto', 'language': 'Deutsch',
        'wer_threshold': 0.0, 'max_attempts': 2, 'whisper_base_url': 'mock',
    }, headers=auth_headers())
    assert start.status_code == 200
    run = _wait_run(client, start.json()['run_id'])
    assert run['status'] == 'completed'
    assert run['accepted'] == 4 and run['rejected'] == 0

    clips = client.get('/api/admin/finetune/clips', headers=auth_headers()).json()
    assert clips['total'] == 4 and clips['voices'] == ['auto']

    summary = client.get('/api/admin/finetune/dataset/summary', headers=auth_headers()).json()
    assert summary['total_clips'] == 4

    cid = clips['clips'][0]['clip_id']
    removed = client.delete(f'/api/admin/finetune/clips/{cid}', headers=auth_headers()).json()
    assert removed['ok'] and any(name.endswith('.wav') for name in removed['removed'])
    assert any(name.endswith('.txt') for name in removed['removed'])
    assert client.get('/api/admin/finetune/clips', headers=auth_headers()).json()['total'] == 3


def test_datagen_clone_mode_writes_voice_folder_and_sample() -> None:
    client = make_client()
    voice = client.post(
        '/api/admin/voices',
        data={'name': 'Hanna Test', 'consent': 'true', 'ref_text': 'Hallo das ist ein Test.'},
        files={'audio_sample': ('sample.wav', make_wav_bytes(), 'audio/wav')},
        headers=auth_headers(),
    ).json()
    voice_id = voice['voice_id']
    domain_id = _create_domain_with_sentences(client, 'Clone Domain', 3)
    start = client.post('/api/admin/finetune/generate', json={
        'domain_ids': [domain_id], 'voice_mode': 'clone', 'voice_ids': [voice_id],
        'language': 'Deutsch', 'wer_threshold': 0.0, 'max_attempts': 2, 'whisper_base_url': 'mock',
    }, headers=auth_headers())
    assert start.status_code == 200
    run = _wait_run(client, start.json()['run_id'])
    assert run['status'] == 'completed' and run['accepted'] == 3

    clips = client.get('/api/admin/finetune/clips', headers=auth_headers()).json()
    assert clips['voices'] == ['Hanna_Test']  # sanitized folder name

    # The per-voice reference sample file is written alongside the clips.
    data_dir = Path(client._temp_data_dir.name)
    voice_dir = data_dir / 'finetune' / 'train' / 'Hanna_Test'
    assert (voice_dir / 'voice_sample_Hanna_Test.wav').exists()
    assert len(list(voice_dir.glob('clip_*.wav'))) == 3
    assert len(list(voice_dir.glob('clip_*.txt'))) == 3


def test_datagen_requires_sentences_and_voices() -> None:
    client = make_client()
    empty_domain = client.post('/api/admin/finetune/domains', json={'name': 'Leer'}, headers=auth_headers()).json()
    # Domain has no sentences -> nothing to generate.
    r = client.post('/api/admin/finetune/generate', json={
        'domain_ids': [empty_domain['domain_id']], 'voice_mode': 'auto',
    }, headers=auth_headers())
    assert r.status_code == 400


# --- Phase 2: trainer orchestration (no GPU) -------------------------------

def test_dataset_builder_writes_manifests_and_dev_split() -> None:
    from omnivoice_tts_server.finetune import dataset_builder, storage

    client = make_client()
    data_dir = Path(client._temp_data_dir.name)
    for i in range(4):
        storage.write_clip(data_dir, 'Sprecher A', wav_bytes=make_wav_bytes(), text=f'Satz {i}', meta={'language_id': 'de'})

    res = dataset_builder.build_manifests(data_dir, 'runX', voices=None, dev_fraction=0.25)
    assert res['train_count'] + res['dev_count'] == 4
    assert res['dev_count'] >= 1
    train_lines = (Path(res['train_jsonl'])).read_text(encoding='utf-8').strip().splitlines()
    first = json.loads(train_lines[0])
    assert set(first) >= {'id', 'audio_path', 'text', 'language_id'}
    assert first['id'].startswith('Sprecher_A__clip_')


def test_runtime_resolves_custom_model_dir(tmp_path) -> None:
    from omnivoice_tts_server.domain.state import InMemoryStore
    from omnivoice_tts_server.runtime_v2 import OmniVoiceSynthesizer

    settings = Settings(models_root_dir=tmp_path, data_dir=tmp_path / 'data')
    synth = OmniVoiceSynthesizer(settings, InMemoryStore(8))

    custom_dir = tmp_path / 'Custom-Demo'
    custom_dir.mkdir(parents=True)
    (custom_dir / 'config.json').write_text('{}', encoding='utf-8')
    source, extra = synth._resolve_model_source('local/Custom-Demo')
    assert Path(source) == custom_dir and extra == {}

    import pytest
    with pytest.raises(RuntimeError):
        synth._resolve_model_source('local/Does-Not-Exist')


def test_training_flow_mocked_subprocess_then_promote(monkeypatch) -> None:
    import time
    from omnivoice_tts_server.finetune import storage

    client = make_client()
    data_dir = Path(client._temp_data_dir.name)
    models_dir = client.app.state.settings.models_root_dir

    # A base model dir must exist for training to be allowed.
    (models_dir / 'OmniVoice').mkdir(parents=True, exist_ok=True)
    (models_dir / 'OmniVoice' / 'config.json').write_text('{}', encoding='utf-8')

    for i in range(4):
        storage.write_clip(data_dir, 'VoiceX', wav_bytes=make_wav_bytes(), text=f'Trainingssatz {i}', meta={'language_id': 'de'})

    trainer = client.app.state.finetune_trainer

    async def fake_subprocess(cmd, env, on_line, run):
        if 'omnivoice.scripts.extract_audio_tokens' in cmd:
            tar_pattern = cmd[cmd.index('--tar_output_pattern') + 1]
            split_dir = Path(tar_pattern).parent.parent
            split_dir.mkdir(parents=True, exist_ok=True)
            (split_dir / 'data.lst').write_text('shard.tar labels.jsonl 4 12.0\n', encoding='utf-8')
        elif 'omnivoice.cli.train' in cmd:
            out = Path(cmd[cmd.index('--output_dir') + 1])
            ckpt = out / 'checkpoint-20'
            ckpt.mkdir(parents=True, exist_ok=True)
            (ckpt / 'config.json').write_text('{}', encoding='utf-8')
            (ckpt / 'model.safetensors').write_bytes(b'\x00')
            for step in (10, 20):
                on_line(f'Step {step} | train/loss: {2.0 / step:.4f} | train/learning_rate: 0.00003 | train/steps_per_sec: 1.5')
        return 0

    monkeypatch.setattr(trainer, '_run_subprocess', fake_subprocess)

    start = client.post('/api/admin/finetune/train', json={'name': 'demo-ft', 'epochs': 1, 'dev_fraction': 0.25}, headers=auth_headers())
    assert start.status_code == 200, start.text
    run_id = start.json()['run_id']

    deadline = time.time() + 8
    status = {}
    while time.time() < deadline:
        status = client.get('/api/admin/finetune/train/status', headers=auth_headers(), params={'run_id': run_id}).json()
        if status['status'] in ('completed', 'failed', 'cancelled'):
            break
        time.sleep(0.05)
    assert status['status'] == 'completed', status
    assert status['checkpoint_dir'] and status['loss_curve']

    # Promote -> registered as a selectable custom model.
    promo = client.post(f'/api/admin/finetune/train/{run_id}/promote', json={'name': 'My FT'}, headers=auth_headers())
    assert promo.status_code == 200, promo.text
    model_id = promo.json()['model_id']
    assert model_id == 'local/Custom-My-FT'
    assert (models_dir / 'Custom-My-FT' / 'config.json').exists()

    models = client.get('/api/v1/models', headers=auth_headers()).json()
    assert any(m['model_id'] == model_id for m in models)
    assert client.get('/api/admin/finetune/checkpoints', headers=auth_headers()).json()['checkpoints']

    # The promoted model is accepted as a default-model setting.
    upd = client.put('/api/admin/settings', json={'default_model': model_id}, headers=auth_headers())
    assert upd.status_code == 200
