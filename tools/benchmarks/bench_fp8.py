"""A/B test: load OmniVoice in fp8 (FineGrainedFP8Config) vs bf16, generate the same
sentence, measure per-process VRAM (torch.cuda) + gen speed, and transcribe each output
via the local Whisper server (:7861) to verify audio quality survives fp8 quantization.

Standalone so it does not touch the running server; loads its OWN model instance.
"""
import gc
import io
import os
import time

# Mirror the server's model resolution: repo id + cache_dir + HF_HOME (no local plain dir).
os.environ.setdefault("HF_HOME", r"X:\dev\G3_OmniVoice\models\.hf")

import soundfile as sf
import torch
import httpx

MODEL_ID = "k2-fsa/OmniVoice"
CACHE_DIR = r"X:\dev\G3_OmniVoice\models"
WHISPER = "http://127.0.0.1:7861/transcribe/"
SENTENCE = "Die Quantisierung auf acht Bit soll die Sprachqualitaet moeglichst nicht verschlechtern."
GEN_KWARGS = dict(num_step=18, guidance_scale=2.0, t_shift=0.1, denoise=True)


def load_model(fp8: bool):
    from omnivoice import OmniVoice
    kwargs = dict(device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa",
                  load_asr=False, cache_dir=CACHE_DIR)
    if fp8:
        from transformers import FineGrainedFP8Config
        kwargs["quantization_config"] = FineGrainedFP8Config()
    return OmniVoice.from_pretrained(MODEL_ID, **kwargs)


def to_wav_bytes(wav, sr: int) -> bytes:
    arr = wav.detach().float().cpu().numpy() if hasattr(wav, "detach") else wav
    arr = arr.reshape(-1)
    buf = io.BytesIO()
    sf.write(buf, arr, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def transcribe(wav_bytes: bytes) -> str:
    try:
        r = httpx.post(WHISPER, files={"file": ("fp8.wav", wav_bytes, "audio/wav")},
                       data={"engine": "local", "voice_ident": "false"}, timeout=120)
        if r.status_code != 200:
            return f"<whisper HTTP {r.status_code}>"
        b = r.json()
        return (b.get("text") or b.get("transcription") or b.get("transcript") or str(b)).strip()
    except Exception as e:
        return f"<whisper error: {e}>"


def run(mode: str, fp8: bool):
    print(f"\n=== {mode} ===")
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        model = load_model(fp8)
    except Exception as e:
        print(f"  LOAD FAILED: {type(e).__name__}: {e}")
        return
    load_s = time.time() - t0
    sr = int(getattr(model, "sampling_rate", 24000) or 24000)
    footprint_gb = torch.cuda.memory_allocated() / 1e9

    # warmup (excluded), then timed gen
    try:
        with torch.inference_mode():
            _ = model.generate(text=["Aufwaermen."], language=[None], **GEN_KWARGS)
        torch.cuda.synchronize()
        tg = time.time()
        with torch.inference_mode():
            wavs = model.generate(text=[SENTENCE], language=[None], **GEN_KWARGS)
        torch.cuda.synchronize()
        gen_s = time.time() - tg
        wav = list(wavs)[0]
        wav_bytes = to_wav_bytes(wav, sr)
        audio_s = len(wav_bytes) / 2 / sr  # 16-bit mono
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        rt = audio_s / gen_s if gen_s else 0
        print(f"  load={load_s:.1f}s  model_footprint={footprint_gb:.2f}GB  gen_peak_alloc={peak_gb:.2f}GB")
        print(f"  audio={audio_s:.1f}s  gen={gen_s:.2f}s  realtime={rt:.1f}x  bytes={len(wav_bytes)}")
        print(f"  TRANSCRIPT: {transcribe(wav_bytes)!r}")
        print(f"  ORIGINAL  : {SENTENCE!r}")
    except Exception as e:
        print(f"  GENERATE FAILED: {type(e).__name__}: {e}")
    finally:
        del model
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()


print(f"OmniVoice fp8 vs bf16 A/B (model: {MODEL_ID}, cache: {CACHE_DIR})")
print(f"torch {torch.__version__}; device {torch.cuda.get_device_name(0)}")
run("fp8 (FineGrainedFP8Config)", fp8=True)
run("bf16 (baseline)", fp8=False)
