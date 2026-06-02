"""Measure where OmniVoice VRAM goes and how much each lever reclaims, using the REAL
synthesizer (faithful: same load/compile/generate path as the server).

Reports allocated vs reserved vs nvidia-smi at each step, for compile OFF and compile ON,
and how much a cache trim (cuda_memory_trim_after_batch / free_memory) gives back. The
model params are ~2 GB; the rest is reserved pool + compile, which trim/eager can free.
"""
import asyncio
import subprocess
import torch

from omnivoice_tts_server.config import get_settings
from omnivoice_tts_server.domain.state import InMemoryStore
from omnivoice_tts_server.domain.models import SpeechRequest, TaskType
from omnivoice_tts_server.runtime_v2 import OmniVoiceSynthesizer, BatchSynthesisItem, OMNIVOICE_AUTO_ALIAS

SENT = "Dies ist ein Satz fuer die VRAM Messung der Sprachsynthese auf der Grafikkarte."


def smi_used_mb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
        return int(out)
    except Exception:
        return -1


def snap(label):
    torch.cuda.synchronize()
    a = torch.cuda.memory_allocated() / 1e9
    r = torch.cuda.memory_reserved() / 1e9
    print(f"    {label:<28} alloc={a:5.2f}GB reserved={r:5.2f}GB  nvidia-smi(total GPU)={smi_used_mb()}MiB")


def make_items(n):
    items = []
    for i in range(n):
        req = SpeechRequest(input=SENT, model=OMNIVOICE_AUTO_ALIAS, task_type=TaskType.custom_voice,
                            voice="auto voice", language="Auto", instructions="")
        items.append(BatchSynthesisItem(job_id=f"j{i}", sentence_index=0, request=req, text=SENT))
    return items


def run_profile(compile_model: bool):
    print(f"\n=== compile_model={compile_model} ===")
    settings = get_settings().model_copy(deep=True)
    settings.runtime_backend = "omnivoice"
    settings.compile_model = compile_model
    settings.warmup_on_startup = False        # measure load vs batch separately
    settings.cuda_memory_trim_after_batch = False
    store = InMemoryStore(max_queue_size=8)
    synth = OmniVoiceSynthesizer(settings=settings, store=store)

    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    snap("before load")
    asyncio.run(synth.ensure_model(OMNIVOICE_AUTO_ALIAS))
    snap("after load (params)")

    # first batch (compile happens here if enabled -> grows reserved a lot)
    synth._generate_batch_sync(make_items(1))
    snap("after batch=1 (warm/compile)")
    synth._generate_batch_sync(make_items(4))
    snap("after batch=4 (peak)")

    # trim: what cuda_memory_trim_after_batch / the Free-memory button reclaims
    synth._trim_cuda_cache_sync(reset_compile_cache=False)
    snap("after trim (empty_cache)")
    synth._trim_cuda_cache_sync(reset_compile_cache=True)
    snap("after trim+compile reset")

    # release model entirely
    synth._release_model()
    del synth
    torch.cuda.empty_cache(); torch.cuda.synchronize()
    snap("after release model")


print(f"torch {torch.__version__}; {torch.cuda.get_device_name(0)}")
print(f"baseline nvidia-smi (other processes incl. your running server/whisper): {smi_used_mb()}MiB of 32607")
run_profile(compile_model=False)
# Only profile compile=True if there is comfortable headroom (avoid OOM-ing the live stack).
free_before = 32607 - smi_used_mb()
if free_before > 12000:
    run_profile(compile_model=True)
else:
    print(f"\n[skipped compile=True profile: only {free_before}MiB free, too risky alongside the running stack]")
