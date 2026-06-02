"""Force LARGE single batches to find the real VRAM ceiling.

Instead of many 1-sentence requests (which don't all coalesce into one generate),
send a few requests each with many sentences: N requests x S sentences are grabbed
round-robin into ONE batch of up to max_batch_size. This reliably forms a big batch
so we can watch VRAM climb toward the 32 GB card limit.
"""
import asyncio
import statistics
import subprocess
import threading
import time

import httpx

BASE = "http://127.0.0.1:8091"
MODEL, VOICE, TASK = "k2-fsa/OmniVoice-AutoVoice", "auto voice", "CustomVoice"
ONE = "Dies ist ein Beispielsatz fuer den VRAM Lasttest auf der Grafikkarte."
VRAM_TOTAL = 32607
# (num_requests, sentences_each) -> target single-batch size = product (capped by max_batch_size)
PLANS = [(4, 12), (6, 12), (8, 12)]   # 48, 72, 96


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
    def stats(self):
        if not self.samples: return (0,0,0)
        mems=[m for m,_ in self.samples]; utils=[u for _,u in self.samples]
        return (max(mems),max(utils),round(statistics.mean(utils)))


def payload(n_sent):
    text=" ".join(ONE for _ in range(n_sent))
    return {"input":text,"model":MODEL,"voice":VOICE,"task_type":TASK,"language":"Auto","instructions":"","response_format":"wav","stream":False}


async def one(client,n_sent):
    t=time.time()
    try:
        r=await client.post(f"{BASE}/api/v1/synthesize",json=payload(n_sent))
        if r.status_code!=200: return {"ok":False,"err":f"HTTP {r.status_code}: {r.text[:80]}"}
        m=r.json().get("metrics",{})
        return {"ok":True,"wall_s":time.time()-t,"job_wall_ms":m.get("job_wall_ms"),"max_batch_seen":m.get("max_batch_size_seen"),"audio_ms":m.get("audio_duration_ms")}
    except Exception as e:
        return {"ok":False,"err":f"{type(e).__name__}: {e}"}


async def run_plan(n_req,s_each):
    target=n_req*s_each
    s=VramSampler(); s.start(); time.sleep(0.3)
    t0=time.time()
    async with httpx.AsyncClient(timeout=600,limits=httpx.Limits(max_connections=n_req+4,max_keepalive_connections=n_req+4)) as c:
        res=await asyncio.gather(*(one(c,s_each) for _ in range(n_req)))
    wall=time.time()-t0; s.stop(); s.join(timeout=2)
    ok=[r for r in res if r.get("ok")]; fails=[r for r in res if not r.get("ok")]
    vram,gpu_pk,gpu_avg=s.stats()
    mbs=[r["max_batch_seen"] for r in ok if r.get("max_batch_seen")]
    audio=[r["audio_ms"] for r in ok if r.get("audio_ms")]
    agg=(sum(audio)/1000/wall) if (audio and wall) else 0
    print(f"  target~{target:3d} ({n_req}x{s_each}): formed_batch={max(mbs) if mbs else '?'}  "
          f"VRAMpeak={vram:5d}MiB (free {VRAM_TOTAL-vram:5d})  GPU avg/peak={gpu_avg}/{gpu_pk}%  "
          f"wall={wall:.1f}s agg_rt={agg:.1f}x fails={len(fails)}")
    if fails: print(f"      err: {fails[0].get('err')}")
    return vram,len(fails)


async def main():
    print("Forcing large single batches (few requests x many sentences):")
    for n_req,s_each in PLANS:
        vram,f=await run_plan(n_req,s_each)
        if f>0 or vram>=30000:
            print(f"  -> stop ({'fail' if f else 'VRAM>=30GB'})"); break
        await asyncio.sleep(1.5)

asyncio.run(main())
