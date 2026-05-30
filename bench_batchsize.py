"""Isolate the batch-size -> latency curve of the OmniVoice worker.

For each target batch size B, fire B concurrent SINGLE-sentence requests (same
short text, same voice -> all batch-eligible) so the worker forms one batch of B,
and measure how long that batch takes. Reads max_batch_size_seen to confirm the
batch actually formed at size B. Samples nvidia-smi for peak VRAM + GPU load.

This tells us where batching stops paying off so we can set max_batch_size sanely.
"""
import asyncio
import statistics
import subprocess
import threading
import time

import httpx

BASE = "http://127.0.0.1:8091"
MODEL = "k2-fsa/OmniVoice-AutoVoice"
VOICE = "auto voice"
TASK = "CustomVoice"
SENTENCE = "Hallo das ist ein kurzer Satz fuer den Batchgroessen Test"  # no '.' -> 1 sentence
SIZES = [1, 2, 4, 8, 12, 16]


class GpuSampler(threading.Thread):
    def __init__(self, interval=0.1):
        super().__init__(daemon=True)
        self.interval = interval
        self.samples = []
        self._evt = threading.Event()

    def run(self):
        while not self._evt.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                ).stdout.strip().splitlines()[0]
                mem, util = (int(x.strip()) for x in out.split(","))
                self.samples.append((mem, util))
            except Exception:
                pass
            self._evt.wait(self.interval)

    def stop(self):
        self._evt.set()

    def peak(self):
        if not self.samples:
            return (0, 0, 0)
        mems = [m for m, _ in self.samples]
        utils = [u for _, u in self.samples]
        return (max(mems), max(utils), round(statistics.mean(utils)))


def payload():
    return {"input": SENTENCE, "model": MODEL, "voice": VOICE, "task_type": TASK,
            "language": "Auto", "instructions": "", "response_format": "wav", "stream": False}


async def one(client):
    t = time.time()
    r = await client.post(f"{BASE}/api/v1/synthesize", json=payload())
    r.raise_for_status()
    m = r.json().get("metrics", {})
    return {"wall_s": time.time() - t, "job_wall_ms": m.get("job_wall_ms"),
            "max_batch_seen": m.get("max_batch_size_seen"), "last_batch": m.get("last_batch_size"),
            "audio_ms": m.get("audio_duration_ms"), "rt": m.get("realtime_x")}


async def measure(B):
    sampler = GpuSampler(); sampler.start()
    t0 = time.time()
    limits = httpx.Limits(max_connections=B + 4, max_keepalive_connections=B + 4)
    async with httpx.AsyncClient(timeout=180, limits=limits) as client:
        res = await asyncio.gather(*(one(client) for _ in range(B)), return_exceptions=True)
    wall = time.time() - t0
    sampler.stop(); sampler.join(timeout=2)
    ok = [r for r in res if isinstance(r, dict)]
    if not ok:
        print(f"  B={B:2d}: ALL FAILED {res[:2]}"); return
    jw = [r["job_wall_ms"] for r in ok if r["job_wall_ms"]]
    mbs = [r["max_batch_seen"] for r in ok if r["max_batch_seen"]]
    audio = [r["audio_ms"] for r in ok if r["audio_ms"]]
    pk_mem, pk_util, avg_util = sampler.peak()
    batch_med = statistics.median(jw) if jw else 0
    per_item = batch_med / max(B, 1)
    total_audio_s = sum(audio) / 1000 if audio else 0
    agg_rt = total_audio_s / wall if wall else 0
    print(f"  B={B:2d}: batch_wall(med)={batch_med:6.0f}ms  per-item={per_item:6.0f}ms  "
          f"wall={wall:5.1f}s  formed_batch(max_seen)={max(mbs) if mbs else '?'}  "
          f"agg_realtime={agg_rt:4.1f}x  GPU avg/peak={avg_util}/{pk_util}%  VRAMpeak={pk_mem}MiB")


async def main():
    print(f"Batch-size scaling: single-sentence requests, model={MODEL}")
    print("(per-item = batch wall / B; lower per-item = batching pays off)")
    for B in SIZES:
        await measure(B)
        await asyncio.sleep(1.5)


asyncio.run(main())
