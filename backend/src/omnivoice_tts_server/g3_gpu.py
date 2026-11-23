"""G3 shared GPU identity and performance snapshot (STANDARD.md section 4.3, schema 1).

Copy verbatim into the backend package (for example backend/g3_gpu.py) and do not edit per repo.

Why: on Optimus every service runs with CUDA_VISIBLE_DEVICES=<RTX 3090 UUID>. Inside the process the
3090 is cuda:0, but nvidia-smi/NVML index 0 is the RTX 2070 SUPER. Mapping "cuda index == nvidia-smi
index" therefore shows the wrong card. This module matches cards by UUID (fallback: PCI bus order).

Usage in a FastAPI route (never call it directly on the event loop; nvidia-smi can take ~100 ms):

    from .g3_gpu import perf_payload
    @app.get("/api/admin/perf")
    async def admin_perf(_: dict = Depends(require_admin)):
        metrics = [
            {"key": "throughput", "label": "Documents/s", "unit": "docs/s", "value": stats.rate_30s(), "window_s": 30},
            {"key": "queue", "label": "Queue depth", "unit": "", "value": scheduler.queue_depth(), "decimals": 0, "integer": True},
        ]
        return await asyncio.to_thread(perf_payload, "G3 Rerank", metrics)
"""
from __future__ import annotations

import os
import subprocess
import threading
import time
from typing import Any

_CACHE_TTL_S = 1.0
_lock = threading.Lock()
_cache: tuple[float, list[dict[str, Any]]] | None = None
_nvml: Any = None
_nvml_failed = False


def _norm_uuid(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text[4:] if text.startswith("gpu-") else text


def _service_uuids(all_gpus: list[dict[str, Any]]) -> list[str]:
    """UUIDs (normalised, no 'GPU-' prefix) of the cards this process can use, in cuda index order."""
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            uuids = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                uuid = getattr(props, "uuid", None)
                if uuid is None:
                    uuids = []
                    break
                uuids.append(_norm_uuid(uuid))
            if uuids:
                return uuids
    except Exception:
        pass
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return [_norm_uuid(g["uuid"]) for g in all_gpus]
    out: list[str] = []
    by_index = {g["nvml_index"]: g for g in all_gpus}
    for entry in (e.strip() for e in visible.split(",") if e.strip()):
        if entry.lower().startswith("gpu-"):
            out.append(_norm_uuid(entry))
        elif entry.isdigit() and int(entry) in by_index:
            # integer ids follow nvidia-smi numbering only with CUDA_DEVICE_ORDER=PCI_BUS_ID (set it in __init__)
            out.append(_norm_uuid(by_index[int(entry)]["uuid"]))
    return out


def _service_pids() -> set[int]:
    pids = {os.getpid()}
    try:
        import psutil  # noqa: PLC0415

        pids.update(child.pid for child in psutil.Process().children(recursive=True))
    except Exception:
        pass
    return pids


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None  # NaN -> None


def _read_nvml(pids: set[int]) -> list[dict[str, Any]] | None:
    global _nvml, _nvml_failed
    if _nvml_failed:
        return None
    try:
        if _nvml is None:
            import pynvml  # noqa: PLC0415  (package nvidia-ml-py)

            pynvml.nvmlInit()
            _nvml = pynvml
        nv = _nvml
        gpus = []
        for i in range(nv.nvmlDeviceGetCount()):
            h = nv.nvmlDeviceGetHandleByIndex(i)

            def safe(fn, *args, _h=h):
                try:
                    return fn(_h, *args)
                except Exception:
                    return None

            mem = safe(nv.nvmlDeviceGetMemoryInfo)
            util = safe(nv.nvmlDeviceGetUtilizationRates)
            pci = safe(nv.nvmlDeviceGetPciInfo)
            power = safe(nv.nvmlDeviceGetPowerUsage)
            limit = safe(nv.nvmlDeviceGetEnforcedPowerLimit)
            procs = safe(nv.nvmlDeviceGetComputeRunningProcesses) or []
            own = [p.usedGpuMemory for p in procs if p.pid in pids and p.usedGpuMemory]
            name = safe(nv.nvmlDeviceGetName) or "GPU"
            bus = getattr(pci, "busId", None) if pci else None
            gpus.append({
                "nvml_index": i,
                "uuid": str(safe(nv.nvmlDeviceGetUUID) or "") or None,
                "pci_bus_id": (bus.decode() if isinstance(bus, bytes) else bus) or None,
                "name": name.decode() if isinstance(name, bytes) else str(name),
                "util_pct": _num(util.gpu) if util else None,
                "mem_total_mib": round(mem.total / 1048576) if mem else None,
                "mem_used_mib": round(mem.used / 1048576) if mem else None,
                "process_mib": round(sum(own) / 1048576) if procs is not None else None,
                "temp_c": _num(safe(nv.nvmlDeviceGetTemperature, nv.NVML_TEMPERATURE_GPU)),
                "power_w": power / 1000 if power is not None else None,
                "power_limit_w": limit / 1000 if limit is not None else None,
            })
        return gpus
    except Exception:
        _nvml_failed = True
        return None


def _read_nvidia_smi(pids: set[int]) -> list[dict[str, Any]]:
    fields = "index,uuid,pci.bus_id,name,utilization.gpu,memory.total,memory.used,temperature.gpu,power.draw,power.limit"
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False,
        ).stdout
    except Exception:
        return []
    own: dict[str, float] = {}
    for line in apps.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3 and parts[0].isdigit() and int(parts[0]) in pids:
            own[_norm_uuid(parts[1])] = own.get(_norm_uuid(parts[1]), 0.0) + (_num(parts[2]) or 0.0)
    gpus = []
    for line in out.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) < 10:
            continue
        gpus.append({
            "nvml_index": int(p[0]) if p[0].isdigit() else None,
            "uuid": p[1] or None,
            "pci_bus_id": p[2] or None,
            "name": p[3],
            "util_pct": _num(p[4]),
            "mem_total_mib": _num(p[5]),
            "mem_used_mib": _num(p[6]),
            "process_mib": own.get(_norm_uuid(p[1]), 0.0),
            "temp_c": _num(p[7]),
            "power_w": _num(p[8]),
            "power_limit_w": _num(p[9]),
        })
    return gpus


def gpu_snapshot() -> list[dict[str, Any]]:
    """All NVIDIA cards on the host, with this service's cards flagged. Cached for 1 s. Blocking."""
    global _cache
    with _lock:
        now = time.monotonic()
        if _cache is not None and now - _cache[0] < _CACHE_TTL_S:
            return _cache[1]
        pids = _service_pids()
        gpus = _read_nvml(pids)
        if gpus is None:
            gpus = _read_nvidia_smi(pids)
        service = _service_uuids(gpus)
        for gpu in gpus:
            key = _norm_uuid(gpu.get("uuid"))
            gpu["cuda_index"] = service.index(key) if key in service else None
            gpu["used_by_service"] = gpu["cuda_index"] is not None or bool(gpu.get("process_mib"))
        _cache = (now, gpus)
        return gpus


def gpu_for_cuda_index(cuda_index: int) -> dict[str, Any] | None:
    """The card behind torch's cuda:<n> in this process (use this instead of gpus[n])."""
    return next((g for g in gpu_snapshot() if g.get("cuda_index") == cuda_index), None)


def perf_payload(service: str, metrics: list[dict[str, Any]], error: str | None = None) -> dict[str, Any]:
    """Body of GET /api/admin/perf (schema 1). `metrics` values must be plain floats/ints or None."""
    try:
        gpus = gpu_snapshot()
    except Exception as exc:  # never fail the endpoint because of the GPU query
        gpus, error = [], error or f"GPU query failed: {exc}"
    return {
        "schema": 1,
        "service": service,
        "sampled_at_ms": int(time.time() * 1000),
        "interval_hint_ms": 2000,
        "gpus": gpus,
        "metrics": metrics,
        "error": error,
    }


class RateWindow:
    """Sliding-window rate for the throughput metric (drops to 0 when idle; never 'last batch')."""

    def __init__(self, window_s: float = 30.0) -> None:
        self.window_s = window_s
        self._events: list[tuple[float, float]] = []
        self._lock = threading.Lock()

    def add(self, amount: float) -> None:
        """Call when work completes: amount = documents, texts, or audio seconds."""
        with self._lock:
            now = time.monotonic()
            self._events.append((now, float(amount)))
            self._trim(now)

    def rate(self) -> float:
        """Units per second over the window."""
        with self._lock:
            now = time.monotonic()
            self._trim(now)
            return sum(a for _, a in self._events) / self.window_s

    def _trim(self, now: float) -> None:
        cutoff = now - self.window_s
        while self._events and self._events[0][0] < cutoff:
            self._events.pop(0)
