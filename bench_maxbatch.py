"""Find the max usable batch size on the RTX 5090 by stepping batch size up and
watching the VRAM peak, getting cautious as we approach ~30 GB.

Requires the server running with raised limits:
  max_parallel_requests >= max tested B   (active requests feed the batch)
  max_batch_size        >= max tested B   (sentences per generate())
  max_queue_size        >= max tested B
  batch_wait_ms         > 0               (so concurrent requests coalesce)

Each wave fires B concurrent single-sentence requests; with the above, the worker
forms ONE batch of B. VRAM is sampled via a streaming nvidia-smi (-lms 50).
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
# A longer sentence so VRAM per batch item is realistic (not a trivially short clip).
SENTENCE = (
    "Dies ist ein bewusst laengerer Beispielsatz fuer den VRAM Lasttest auf der RTX 5090 "
    "der mehr Tokens erzeugt damit der Speicherbedarf pro Element im Batch realistisch bleibt"
)
CANDIDATES = [16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96]
VRAM_CAUTION_MIB = 29000   # stop escalating once a wave peaks above this
VRAM_TOTAL_MIB = 32607


class VramSampler(threading.Thread):
    """Streams nvidia-smi at 50ms and records the (mem_used, util) timeline."""
    def __init__(self):
        super().__init__(daemon=True)
        self.samples = []
        self._stop_evt = threading.Event()
        self._proc = None

    def run(self):
        try:
            self._proc = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                 "--format=csv,noheader,nounits", "-lms", "50"],
                stdout=subprocess.PIPE, text=True, bufsize=1,
            )
        except Exception:
            return
        for line in self._proc.stdout:
            if self._stop_evt.is_set():
                break
            try:
                mem, util = (int(x.strip()) for x in line.split(","))
                self.samples.append((mem, util))
            except Exception:
                pass

    def stop(self):
        self._stop_evt.set()
        if self._proc:
            try:
                self._proc.terminate()
            except Exception:
                pass

    def stats(self):
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
    try:
        r = await client.post(f"{BASE}/api/v1/synthesize", json=payload())
        if r.status_code != 200:
            return {"ok": False, "err": f"HTTP {r.status_code}: {r.text[:80]}"}
        m = r.json().get("metrics", {})
        return {"ok": True, "wall_s": time.time() - t, "job_wall_ms": m.get("job_wall_ms"),
                "max_batch_seen": m.get("max_batch_size_seen"), "audio_ms": m.get("audio_duration_ms")}
    except Exception as e:
        return {"ok": False, "err": f"{type(e).__name__}: {e}"}


async def wave(B, *, warmup=False):
    sampler = VramSampler(); sampler.start()
    time.sleep(0.3)  # let sampler capture pre-wave baseline
    t0 = time.time()
    limits = httpx.Limits(max_connections=B + 8, max_keepalive_connections=B + 8)
    async with httpx.AsyncClient(timeout=300, limits=limits) as client:
        res = await asyncio.gather(*(one(client) for _ in range(B)))
    wall = time.time() - t0
    sampler.stop(); sampler.join(timeout=2)
    ok = [r for r in res if r.get("ok")]
    fails = [r for r in res if not r.get("ok")]
    vram_peak, gpu_peak, gpu_avg = sampler.stats()
    if warmup:
        print(f"  [warmup B={B}] done wall={wall:.1f}s ok={len(ok)} fails={len(fails)} VRAMpeak={vram_peak}MiB")
        return None
    jw = [r["job_wall_ms"] for r in ok if r.get("job_wall_ms")]
    mbs = [r["max_batch_seen"] for r in ok if r.get("max_batch_seen")]
    audio = [r["audio_ms"] for r in ok if r.get("audio_ms")]
    batch_med = statistics.median(jw) if jw else 0
    per_item = batch_med / max(B, 1)
    agg_rt = (sum(audio) / 1000 / wall) if (audio and wall) else 0
    formed = max(mbs) if mbs else "?"
    headroom = VRAM_TOTAL_MIB - vram_peak
    flag = "  <-- CAUTION near limit" if vram_peak >= VRAM_CAUTION_MIB else ""
    print(f"  B={B:3d}: formed_batch={formed:<3} VRAMpeak={vram_peak:5d}MiB (free {headroom:5d}) "
          f"GPU avg/peak={gpu_avg:3d}/{gpu_peak:3d}%  batch_wall={batch_med:6.0f}ms per-item={per_item:5.0f}ms "
          f"agg_rt={agg_rt:4.1f}x fails={len(fails)}{flag}")
    if fails:
        print(f"       first error: {fails[0].get('err')}")
    return {"B": B, "vram_peak": vram_peak, "formed": formed, "fails": len(fails),
            "agg_rt": agg_rt, "per_item": per_item}


async def main():
    print(f"Idle VRAM baseline + warmup (triggers dynamic compile once)...")
    # idle reading
    s = VramSampler(); s.start(); time.sleep(0.6); s.stop(); s.join(timeout=2)
    print(f"  idle VRAM ~{s.stats()[0]} MiB of {VRAM_TOTAL_MIB} MiB")
    await wave(8, warmup=True)
    await asyncio.sleep(1.0)

    print(f"\nStepping batch size up (stop when VRAM peak >= {VRAM_CAUTION_MIB} MiB or a wave fails):")
    results = []
    for B in CANDIDATES:
        r = await wave(B)
        if r:
            results.append(r)
            if r["fails"] > 0 or r["vram_peak"] >= VRAM_CAUTION_MIB:
                print(f"\n  -> stopping escalation at B={B} (fails={r['fails']}, VRAMpeak={r['vram_peak']}MiB)")
                break
        await asyncio.sleep(1.2)

    print("\n=== SUMMARY ===")
    safe = [r for r in results if r["fails"] == 0]
    if safe:
        best_rt = max(safe, key=lambda r: r["agg_rt"])
        biggest = max(safe, key=lambda r: r["B"])
        print(f"  Largest batch run WITHOUT failure: B={biggest['B']} "
              f"(VRAMpeak {biggest['vram_peak']}MiB, {biggest['agg_rt']:.1f}x realtime)")
        print(f"  Best aggregate throughput: B={best_rt['B']} at {best_rt['agg_rt']:.1f}x realtime "
              f"(VRAMpeak {best_rt['vram_peak']}MiB)")


asyncio.run(main())
