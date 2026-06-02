"""Batch + GPU benchmark for the OmniVoice server.

Fires N concurrent /api/v1/synthesize requests (stream=False, returns JSON
metrics incl. max_batch_size_seen / last_batch_size) against the loaded model,
while sampling nvidia-smi for VRAM + GPU load. Then round-trips one synthesized
WAV through the local Whisper server (7861) to confirm intelligible audio.

Reads NO admin key: the public /api/v1/synthesize response already carries the
batch-size metrics we need to see whether requests are really batched together.
"""
import asyncio
import json
import statistics
import subprocess
import threading
import time

import httpx

BASE = "http://127.0.0.1:8091"
WHISPER = "http://127.0.0.1:7861"

# Already-loaded model -> no reload skews timing/VRAM.
MODEL = "k2-fsa/OmniVoice-AutoVoice"
VOICE = "auto voice"
TASK = "CustomVoice"
TEXT = (
    "Das ist der erste Satz fuer den Batch-Test. "
    "Hier kommt ein zweiter, etwas laengerer Satz, der die Auslastung erhoeht. "
    "Und ein dritter Satz schliesst die Anfrage ab."
)

WAVES = [1, 4, 8, 16]   # concurrent requests per wave


class GpuSampler(threading.Thread):
    def __init__(self, interval=0.15):
        super().__init__(daemon=True)
        self.interval = interval
        self.samples = []      # (mem_used_mb, util_pct)
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
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
            self._stop_event.wait(self.interval)

    def stop(self):
        self._stop_event.set()

    def summary(self):
        if not self.samples:
            return "no GPU samples"
        mems = [m for m, _ in self.samples]
        utils = [u for _, u in self.samples]
        return (f"VRAM used {min(mems)}->{max(mems)} MiB (peak {max(mems)}), "
                f"GPU load avg {statistics.mean(utils):.0f}% peak {max(utils)}% "
                f"({len(self.samples)} samples)")


def synth_payload():
    return {"input": TEXT, "model": MODEL, "voice": VOICE, "task_type": TASK,
            "language": "Auto", "instructions": "", "response_format": "wav",
            "stream": False}


async def one_request(client, idx):
    t = time.time()
    r = await client.post(f"{BASE}/api/v1/synthesize", json=synth_payload())
    r.raise_for_status()
    m = r.json().get("metrics", {})
    return {
        "idx": idx,
        "wall_s": time.time() - t,
        "max_batch_seen": m.get("max_batch_size_seen"),
        "last_batch": m.get("last_batch_size"),
        "batch_count": m.get("batch_count"),
        "ttfa_ms": m.get("ttfa_ms"),
        "job_wall_ms": m.get("job_wall_ms"),
        "queue_wait_ms": m.get("queue_wait_ms"),
        "audio_ms": m.get("audio_duration_ms"),
        "realtime_x": m.get("realtime_x"),
    }


async def run_wave(n):
    sampler = GpuSampler()
    sampler.start()
    t0 = time.time()
    limits = httpx.Limits(max_connections=n + 4, max_keepalive_connections=n + 4)
    async with httpx.AsyncClient(timeout=120, limits=limits) as client:
        results = await asyncio.gather(*(one_request(client, i) for i in range(n)),
                                       return_exceptions=True)
    wall = time.time() - t0
    sampler.stop(); sampler.join(timeout=2)

    ok = [r for r in results if isinstance(r, dict)]
    errs = [r for r in results if not isinstance(r, dict)]
    print(f"\n========== WAVE: {n} concurrent ==========")
    print(f"  wall (all done): {wall:.2f}s   ok={len(ok)} err={len(errs)}")
    if errs:
        print(f"  errors: {[repr(e)[:120] for e in errs[:3]]}")
    if ok:
        mbs = [r['max_batch_seen'] for r in ok if r['max_batch_seen'] is not None]
        lbs = [r['last_batch'] for r in ok if r['last_batch'] is not None]
        ttfa = [r['ttfa_ms'] for r in ok if r['ttfa_ms'] is not None]
        jw = [r['job_wall_ms'] for r in ok if r['job_wall_ms'] is not None]
        qw = [r['queue_wait_ms'] for r in ok if r['queue_wait_ms'] is not None]
        rt = [r['realtime_x'] for r in ok if r['realtime_x'] is not None]
        print(f"  max_batch_size_seen: min={min(mbs)} med={statistics.median(mbs):.0f} max={max(mbs)}  (cap=8)")
        print(f"  last_batch_size:     {sorted(lbs)}")
        print(f"  batch_count/job:     {[r['batch_count'] for r in ok]}")
        print(f"  ttfa_ms:   p50={statistics.median(ttfa):.0f} max={max(ttfa):.0f}" if ttfa else "  ttfa: n/a")
        print(f"  queue_wait_ms: p50={statistics.median(qw):.0f} max={max(qw):.0f}" if qw else "")
        print(f"  job_wall_ms: p50={statistics.median(jw):.0f} max={max(jw):.0f}" if jw else "")
        print(f"  realtime_x: p50={statistics.median(rt):.1f}x" if rt else "")
    print(f"  GPU: {sampler.summary()}")
    return ok


async def whisper_roundtrip():
    print("\n========== WHISPER ROUND-TRIP (audio sanity) ==========")
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{BASE}/v1/audio/speech", json=synth_payload())
        r.raise_for_status()
        wav = r.content
        print(f"  synthesized WAV: {len(wav)} bytes")
        candidates = [
            (f"{WHISPER}/transcribe/", {"engine": "local", "voice_ident": "false"}),
            (f"{WHISPER}/v1/audio/transcriptions", {"model": "whisper-1"}),
            (f"{WHISPER}/audio/transcriptions", {"model": "whisper-1"}),
        ]
        for url, data in candidates:
            try:
                wr = await client.post(url, files={"file": ("test.wav", wav, "audio/wav")}, data=data)
                if wr.status_code >= 400:
                    print(f"  {url} -> HTTP {wr.status_code}")
                    continue
                try:
                    body = wr.json()
                    text = body.get("text") or body.get("transcription") or body.get("transcript") or json.dumps(body)[:200]
                except Exception:
                    text = wr.text[:200]
                print(f"  OK via {url}")
                print(f"  TRANSCRIPT: {text!r}")
                print(f"  (Original : {TEXT!r})")
                return
            except Exception as e:
                print(f"  {url} -> {type(e).__name__}: {e}")
        print("  Whisper round-trip failed on all candidates.")


async def main():
    print(f"Input text: {len(TEXT)} chars, ~3 sentences. Model={MODEL} voice={VOICE}")
    for n in WAVES:
        await run_wave(n)
        await asyncio.sleep(1.0)   # let the queue drain between waves
    await whisper_roundtrip()


asyncio.run(main())
