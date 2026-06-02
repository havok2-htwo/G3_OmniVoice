"""Verify trim-after-warmup: load the REAL synthesizer with compile + warmup on (the
server's startup path) and check reserved VRAM AFTER load. If low, the startup pool is
released immediately and the operator just needs to restart with the new code.
"""
import asyncio, subprocess, torch
from omnivoice_tts_server.config import get_settings
from omnivoice_tts_server.domain.state import InMemoryStore
from omnivoice_tts_server.runtime_v2 import OmniVoiceSynthesizer, OMNIVOICE_AUTO_ALIAS

def smi():
    return int(subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],
                              capture_output=True,text=True).stdout.strip().splitlines()[0])

s = get_settings().model_copy(deep=True)
s.runtime_backend = 'omnivoice'
s.compile_model = True          # same as server
s.warmup_on_startup = True       # triggers warmup + compile autotuning, then trim-after-warmup
s.cuda_memory_trim_after_batch = True
store = InMemoryStore(max_queue_size=8)
synth = OmniVoiceSynthesizer(settings=s, store=store)

torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
print('smi before load:', smi(), 'MiB')
asyncio.run(synth.ensure_model(OMNIVOICE_AUTO_ALIAS))   # load -> compile -> warmup -> trim-after-warmup
torch.cuda.synchronize()
print(f'AFTER load+warmup+trim:  this-process reserved={torch.cuda.memory_reserved()/1e9:.2f}GB '
      f'allocated={torch.cuda.memory_allocated()/1e9:.2f}GB  peak_alloc={torch.cuda.max_memory_allocated()/1e9:.2f}GB')
print('smi after (incl. running server):', smi(), 'MiB')
