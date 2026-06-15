"""FinetuneTrainer: CPU/GPU preprocess (audio->codec tokens) + the omnivoice trainer,
launched as subprocesses, plus promotion of a checkpoint to a runtime-loadable model.

Conservative full-finetune path (no new deps): the official `omnivoice` package ships
the trainer (HF Accelerate) and the token-extraction script; we only orchestrate them,
generate the config/manifest JSON, parse progress, and wire the checkpoint into the
runtime model list. The GPU steps run in a child process so a crash never takes down the
FastAPI server and Accelerate can own its own process/seeds.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import shutil
import sys
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from ..config import Settings, save_runtime_settings
from ..domain.state import InMemoryStore, new_id, utcnow
from ..services_v2 import EventHub, spawn_tracked_task
from . import dataset_builder, registry
from .schemas import CheckpointItem, CheckpointListResponse, TrainRunResponse, TrainStartRequest

logger = logging.getLogger('omnivoice_tts_server.finetune.trainer')

_STEP_RE = re.compile(r'Step (\d+)')
_LOSS_RE = re.compile(r'train/loss:\s*([0-9.eE+-]+)')
_LR_RE = re.compile(r'train/learning_rate:\s*([0-9.eE+-]+)')
_SPS_RE = re.compile(r'train/steps_per_sec:\s*([0-9.eE+-]+)')
_EVAL_RE = re.compile(r'Eval Loss:\s*([0-9.eE+-]+)')

_TERMINAL = {'completed', 'failed', 'cancelled'}


class FinetuneTrainer:
    def __init__(self, store: InMemoryStore, settings: Settings, events: EventHub, synthesizer: Any) -> None:
        self.store = store
        self.settings = settings
        self.events = events
        self.synthesizer = synthesizer

    # --- public API --------------------------------------------------------

    def base_model_dir(self) -> Path:
        return self.settings.models_root_dir / 'OmniVoice'

    async def start_training(self, req: TrainStartRequest) -> TrainRunResponse:
        base_dir = self.base_model_dir()
        if not (base_dir / 'config.json').exists():
            raise ValueError(
                f'Base model not found at {base_dir}. Place the OmniVoice checkpoint there before finetuning.'
            )
        run_id = new_id('fttrain')
        run: dict[str, Any] = {
            'run_id': run_id,
            'status': 'queued',
            'phase': 'queued',
            'created_at': utcnow(),
            'completed_at': None,
            'base_model': str(base_dir),
            'dataset_dir': None,
            'output_dir': str(self.settings.models_root_dir / 'finetune' / run_id),
            'train_count': 0,
            'dev_count': 0,
            'total_steps': 0,
            'current_step': 0,
            'loss': None,
            'eval_loss': None,
            'lr': None,
            'steps_per_sec': None,
            'eta_ms': None,
            'loss_curve': [],
            'checkpoint_dir': None,
            'log_tail': [],
            'error_message': None,
            'cancelled': False,
            'subprocess_pid': None,
        }
        self.store.finetune_train_runs.clear()
        self.store.finetune_train_runs[run_id] = run
        spawn_tracked_task(self.store, self._execute(run, req), label=f'finetune-train:{run_id}')
        await self._publish(run)
        return _train_response(run)

    def get_run(self, run_id: str | None = None) -> TrainRunResponse | None:
        if run_id:
            run = self.store.finetune_train_runs.get(run_id)
        else:
            run = next(iter(sorted(self.store.finetune_train_runs.values(), key=lambda r: r['created_at'], reverse=True)), None)
        return _train_response(run) if run else None

    def cancel_run(self, run_id: str) -> bool:
        run = self.store.finetune_train_runs.get(run_id)
        if not run:
            return False
        run['cancelled'] = True
        pid = run.get('subprocess_pid')
        if pid:
            try:
                os.kill(pid, 9)
            except Exception:
                pass
        return True

    def list_checkpoints(self) -> CheckpointListResponse:
        items = []
        for entry in registry.list_checkpoints(self.settings.data_dir):
            dirname = entry.get('dirname') or ''
            exists = (self.settings.models_root_dir / dirname / 'config.json').exists()
            items.append(CheckpointItem(
                checkpoint_id=entry['checkpoint_id'], name=entry.get('name') or dirname,
                model_id=entry.get('model_id') or registry.model_id_for(dirname), dirname=dirname,
                base_model=entry.get('base_model'), steps=int(entry.get('steps') or 0),
                run_id=entry.get('run_id'), created_at=entry.get('created_at'), exists=exists,
            ))
        return CheckpointListResponse(checkpoints=items)

    def promote_run(self, run_id: str, name: str) -> CheckpointItem:
        run = self.store.finetune_train_runs.get(run_id)
        if not run or not run.get('checkpoint_dir'):
            raise KeyError(run_id)
        checkpoint_dir = Path(run['checkpoint_dir'])
        if not (checkpoint_dir / 'config.json').exists():
            raise ValueError(f'Checkpoint at {checkpoint_dir} is incomplete (missing config.json).')
        dirname = f'Custom-{registry.slug(name)}'
        target = self.settings.models_root_dir / dirname
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(checkpoint_dir, target)
        entry = registry.register_checkpoint(
            self.settings.data_dir, name=name, dirname=dirname, base_model=run.get('base_model'),
            steps=int(run.get('current_step') or 0), run_id=run_id, now_iso=utcnow().isoformat(),
        )
        self._add_supported_model(entry['model_id'])
        return CheckpointItem(
            checkpoint_id=entry['checkpoint_id'], name=entry['name'], model_id=entry['model_id'],
            dirname=dirname, base_model=entry.get('base_model'), steps=entry.get('steps') or 0,
            run_id=run_id, created_at=entry.get('created_at'), exists=True,
        )

    def delete_checkpoint(self, checkpoint_id: str, *, delete_files: bool = True) -> bool:
        entry = registry.delete_checkpoint(self.settings.data_dir, checkpoint_id)
        if entry is None:
            return False
        model_id = entry.get('model_id')
        if model_id and model_id in self.settings.supported_models:
            self.settings.supported_models.remove(model_id)
        if delete_files and entry.get('dirname'):
            target = self.settings.models_root_dir / entry['dirname']
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
        return True

    # --- execution ---------------------------------------------------------

    async def _execute(self, run: dict[str, Any], req: TrainStartRequest) -> None:
        run_id = run['run_id']
        data_dir = self.settings.data_dir
        try:
            # Free VRAM for the child process before any GPU work.
            try:
                await self.synthesizer.unload()
            except Exception as exc:  # noqa: BLE001
                logger.warning('unload before training failed (non-fatal): %s', exc)

            run['status'] = 'preprocessing'
            run['phase'] = 'building manifests'
            await self._publish(run)
            manifests = dataset_builder.build_manifests(
                data_dir, run_id, voices=req.voices or None, dev_fraction=req.dev_fraction,
            )
            run['dataset_dir'] = manifests['dataset_dir']
            run['train_count'] = manifests['train_count']
            run['dev_count'] = manifests['dev_count']
            await self._publish(run)

            env = self._child_env()
            ds_dir = Path(manifests['dataset_dir'])

            run['phase'] = 'extracting audio tokens (train)'
            await self._publish(run)
            rc = await self._extract_tokens(ds_dir / 'train', env, run)
            if rc != 0 and not run['cancelled']:
                raise RuntimeError(f'Token extraction (train) failed with exit code {rc}.')

            if manifests.get('dev_jsonl') and not run['cancelled']:
                run['phase'] = 'extracting audio tokens (dev)'
                await self._publish(run)
                await self._extract_tokens(ds_dir / 'dev', env, run)

            if run['cancelled']:
                run['status'] = 'cancelled'
                return

            data_config = dataset_builder.write_data_config(data_dir, run_id)

            # Build + write the training config.
            total_steps = self._estimate_steps(req, manifests['train_count'])
            run['total_steps'] = total_steps
            train_config_path = self._write_train_config(ds_dir, req, total_steps)

            run['status'] = 'training'
            run['phase'] = 'training'
            await self._publish(run)
            cmd = [
                sys.executable, '-m', 'omnivoice.cli.train',
                '--train_config', str(train_config_path),
                '--data_config', str(data_config),
                '--output_dir', run['output_dir'],
            ]
            rc = await self._run_subprocess(cmd, env, self._make_train_line_handler(run), run)
            if run['cancelled']:
                run['status'] = 'cancelled'
                return
            if rc != 0:
                raise RuntimeError(f'Training process exited with code {rc}. See log tail.')

            run['checkpoint_dir'] = self._latest_checkpoint(Path(run['output_dir']))
            run['status'] = 'completed'
            run['phase'] = 'done'
        except Exception as exc:  # noqa: BLE001
            run['status'] = 'failed'
            run['error_message'] = str(exc)
            logger.error('training run %s failed: %s', run_id, exc, exc_info=exc)
        finally:
            run['completed_at'] = utcnow()
            run['subprocess_pid'] = None
            # Best-effort: bring the base model back so TTS works after training.
            try:
                await self.synthesizer.ensure_model(self.settings.active_model)
            except Exception:
                pass
            await self._publish(run)

    async def _extract_tokens(self, split_dir: Path, env: dict[str, str], run: dict[str, Any]) -> int:
        raw_jsonl = split_dir / 'raw.jsonl'
        if not raw_jsonl.exists():
            return 0
        cmd = [
            sys.executable, '-m', 'omnivoice.scripts.extract_audio_tokens',
            '--input_jsonl', str(raw_jsonl),
            '--tar_output_pattern', str(split_dir / 'audios' / 'shard-%06d.tar'),
            '--jsonl_output_pattern', str(split_dir / 'txts' / 'shard-%06d.jsonl'),
            '--tokenizer_path', self.settings.finetune_audio_tokenizer,
            '--num_machines', '1', '--nj_per_gpu', '1', '--loader_workers', '4', '--skip_errors',
        ]
        return await self._run_subprocess(cmd, env, lambda line: self._append_log(run, line), run)

    def _child_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env['PYTHONUNBUFFERED'] = '1'
        env.setdefault('HF_HOME', str(self.settings.models_root_dir / '.hf'))
        device = (self.settings.preferred_device or 'cuda:0')
        if device.startswith('cuda:'):
            env['CUDA_VISIBLE_DEVICES'] = device.split(':', 1)[1]
        return env

    # Overridable seam: tests monkeypatch this to avoid launching real subprocesses.
    async def _run_subprocess(self, cmd: list[str], env: dict[str, str], on_line: Callable[[str], None], run: dict[str, Any]) -> int:
        proc = await asyncio.create_subprocess_exec(
            *cmd, env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            cwd=str(self.settings.models_root_dir.parent),
        )
        run['subprocess_pid'] = proc.pid
        assert proc.stdout is not None
        while True:
            raw = await proc.stdout.readline()
            if not raw:
                break
            line = raw.decode('utf-8', 'replace').rstrip()
            if line:
                on_line(line)
            if run.get('cancelled'):
                try:
                    proc.kill()
                except Exception:
                    pass
                break
        rc = await proc.wait()
        run['subprocess_pid'] = None
        return rc

    def _make_train_line_handler(self, run: dict[str, Any]) -> Callable[[str], None]:
        def handler(line: str) -> None:
            self._append_log(run, line)
            step_m = _STEP_RE.search(line)
            if step_m:
                run['current_step'] = int(step_m.group(1))
            loss_m = _LOSS_RE.search(line)
            if loss_m:
                run['loss'] = float(loss_m.group(1))
                run['loss_curve'] = (run.get('loss_curve') or [])[-199:] + [float(loss_m.group(1))]
            lr_m = _LR_RE.search(line)
            if lr_m:
                run['lr'] = float(lr_m.group(1))
            sps_m = _SPS_RE.search(line)
            if sps_m:
                sps = float(sps_m.group(1))
                run['steps_per_sec'] = sps
                remaining = max(0, int(run.get('total_steps') or 0) - int(run.get('current_step') or 0))
                run['eta_ms'] = int(remaining / sps * 1000) if sps > 0 else None
            eval_m = _EVAL_RE.search(line)
            if eval_m:
                run['eval_loss'] = float(eval_m.group(1))
        return handler

    def _append_log(self, run: dict[str, Any], line: str) -> None:
        tail = run.get('log_tail') or []
        tail.append(line)
        run['log_tail'] = tail[-60:]

    def _estimate_steps(self, req: TrainStartRequest, train_count: int) -> int:
        if req.steps_override > 0:
            return req.steps_override
        per_epoch = max(1, math.ceil(max(1, train_count) / max(1, req.max_batch_size)))
        return max(20, per_epoch * req.epochs)

    def _write_train_config(self, ds_dir: Path, req: TrainStartRequest, total_steps: int) -> Path:
        save_steps = max(10, total_steps // 4)
        eval_steps = max(10, total_steps // 4)
        logging_steps = max(1, total_steps // 20)
        config = {
            'init_from_checkpoint': str(self.base_model_dir()),
            'learning_rate': req.learning_rate,
            'weight_decay': req.weight_decay,
            'max_grad_norm': 1.0,
            'steps': total_steps,
            'seed': req.seed,
            'lr_scheduler_type': 'cosine',
            'warmup_type': 'ratio',
            'warmup_ratio': req.warmup_ratio,
            'batch_tokens': req.batch_tokens,
            'gradient_accumulation_steps': req.gradient_accumulation_steps,
            'num_workers': 4,
            'mixed_precision': req.mixed_precision,
            'allow_tf32': True,
            'attn_implementation': req.attn_implementation,
            'max_batch_size': req.max_batch_size,
            'logging_steps': logging_steps,
            'eval_steps': eval_steps,
            'save_steps': save_steps,
            'keep_last_n_checkpoints': req.keep_last_n_checkpoints,
        }
        path = ds_dir / 'train_config.json'
        path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
        return path

    @staticmethod
    def _latest_checkpoint(output_dir: Path) -> str | None:
        if not output_dir.exists():
            return None
        checkpoints = [
            p for p in output_dir.iterdir()
            if p.is_dir() and p.name.startswith('checkpoint-') and (p / 'config.json').exists()
        ]
        if not checkpoints:
            return None
        latest = max(checkpoints, key=lambda p: int(p.name.split('-')[-1]) if p.name.split('-')[-1].isdigit() else 0)
        return str(latest)

    def _add_supported_model(self, model_id: str) -> None:
        if model_id not in self.settings.supported_models:
            self.settings.supported_models.append(model_id)

    async def _publish(self, run: dict[str, Any]) -> None:
        await self.events.publish('dashboard.snapshot', {'reason': 'finetune.train', 'run_id': run['run_id']})


def _train_response(run: dict[str, Any]) -> TrainRunResponse:
    total = int(run.get('total_steps') or 0)
    current = int(run.get('current_step') or 0)
    pct = (current / total * 100.0) if total else 0.0
    return TrainRunResponse(
        run_id=run['run_id'], status=run['status'], phase=run.get('phase') or '',
        created_at=run['created_at'], completed_at=run.get('completed_at'),
        base_model=run.get('base_model'), dataset_dir=run.get('dataset_dir'), output_dir=run.get('output_dir'),
        train_count=int(run.get('train_count') or 0), dev_count=int(run.get('dev_count') or 0),
        total_steps=total, current_step=current, loss=run.get('loss'), eval_loss=run.get('eval_loss'),
        lr=run.get('lr'), steps_per_sec=run.get('steps_per_sec'), eta_ms=run.get('eta_ms'),
        pct=round(pct, 1), loss_curve=list(run.get('loss_curve') or []),
        checkpoint_dir=run.get('checkpoint_dir'), log_tail=list(run.get('log_tail') or []),
        error_message=run.get('error_message'),
    )
