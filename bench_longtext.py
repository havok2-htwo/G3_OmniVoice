"""How does VRAM scale with a SINGLE long text (no sentence chunking, internal audio
chunking disabled), and how many minutes of audio can we generate at once within the
32 GB card (model resident)?

Sends ONE request whose text is a clause repeated K times -> one long sequence into
generate(). Steps K up, watching VRAM peak + produced audio duration + wall time,
until VRAM nears 30 GB, the request fails (OOM), or it gets impractically slow.
"""
import statistics
import subprocess
import threading
import time

import httpx

BASE = "http://127.0.0.1:8091"
MODEL, VOICE, TASK = "k2-fsa/OmniVoice-AutoVoice", "auto voice", "CustomVoice"
CLAUSE = "Das Wetter heute ist freundlich und die Sonne scheint angenehm warm. "
VRAM_TOTAL = 32607
VRAM_STOP = 30000
KS = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]


class VramSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True); self.samples=[]; self._evt=threading.Event(); self._proc=None
    def run(self):
        try:
            self._proc=subprocess.Popen(["nvidia-smi","--query-gpu=memory.used,utilization.gpu","--format=csv,noheader,nounits","-lms","50"],stdout=subprocess.PIPE,text=True,bufsize=1)
        except Exception: return
        for line in self._proc.stdout:
            if self._evt.is_set(): break
            try:
                m,u=(int(x.strip()) for x in line.split(",")); self.samples.append((m,u))
            except Exception: pass
    def stop(self):
        self._evt.set()
        if self._proc:
            try: self._proc.terminate()
            except Exception: pass
    def peak(self):
        if not self.samples: return (0,0)
        return (max(m for m,_ in self.samples), max(u for _,u in self.samples))


def gen(K):
    text = CLAUSE * K
    s = VramSampler(); s.start(); time.sleep(0.3)
    t0 = time.time()
    try:
        with httpx.Client(timeout=400) as c:
            r = c.post(f"{BASE}/api/v1/synthesize", json={"input": text, "model": MODEL, "voice": VOICE,
                       "task_type": TASK, "language": "Auto", "instructions": "", "response_format": "wav", "stream": False})
        wall = time.time() - t0; s.stop(); s.join(timeout=2)
        vram, gpu = s.peak()
        if r.status_code != 200:
            print(f"  K={K:3d} chars={len(text):6d}: FAILED HTTP {r.status_code} after {wall:.1f}s  VRAMpeak={vram}MiB  {r.text[:90]}")
            return vram, False, wall
        m = r.json().get("metrics", {})
        audio_s = (m.get("audio_duration_ms") or 0) / 1000
        rt = audio_s / wall if wall else 0
        print(f"  K={K:3d} chars={len(text):6d}: audio={audio_s/60:5.2f}min ({audio_s:6.1f}s)  "
              f"VRAMpeak={vram:5d}MiB (free {VRAM_TOTAL-vram:5d})  wall={wall:6.1f}s  rt={rt:5.1f}x  GPUpeak={gpu}%")
        return vram, True, wall
    except Exception as e:
        wall = time.time() - t0; s.stop(); s.join(timeout=2); vram, _ = s.peak()
        print(f"  K={K:3d} chars={len(CLAUSE*K):6d}: EXCEPTION after {wall:.1f}s {type(e).__name__}: {e}  VRAMpeak={vram}MiB")
        return vram, False, wall


def main():
    print("Single long text, no chunking. Stepping length up toward the VRAM ceiling:")
    for K in KS:
        vram, ok, wall = gen(K)
        if not ok or vram >= VRAM_STOP:
            print(f"  -> stop at K={K} ({'fail' if not ok else 'VRAM>=30GB'})")
            break
        if wall > 240:
            print(f"  -> stop at K={K} (wall>{wall:.0f}s, impractical)")
            break
        time.sleep(1.5)


main()
