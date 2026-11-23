// G3 shared performance graph (STANDARD.md section 4). Copy verbatim to frontend/src/PerfPanel.tsx,
// together with perf.css and usePolling.ts. Do not edit per repo: wire it from App with
//   usePerfFeed(() => api.getPerf(), loggedIn)       // once, at App level (keeps history across tabs)
//   <PerfPanel />                                     // on the Dashboard tab
//   clearPerfHistory()                                // on logout
// Axis mechanics follow the G3_RERANK ScalingChart (charts.tsx): nice ticks, ResizeObserver, crosshair + tooltip.
import { useEffect, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";
import { useVisiblePolling } from "./usePolling";

// ---------- contract types (GET /api/admin/perf, schema 1) ----------
export interface PerfGpu {
  uuid: string | null;
  pci_bus_id: string | null;
  nvml_index: number | null; // nvidia-smi numbering, shown in the UI as "GPU <n>"
  cuda_index: number | null; // index inside this process (after CUDA_VISIBLE_DEVICES)
  name: string;
  used_by_service: boolean;
  util_pct: number | null;
  mem_total_mib: number | null;
  mem_used_mib: number | null; // whole card, all processes
  process_mib: number | null; // this service's processes on this card
  temp_c: number | null;
  power_w: number | null;
  power_limit_w: number | null;
}

export interface PerfMetric {
  key: string; // "throughput" | "queue" | "latency" | ...
  label: string; // "Documents/s", "Queue depth", "Realtime factor"
  unit: string; // "docs/s", "", "x", "ms"
  value: number | null;
  decimals?: number; // default 1 (0 for counts)
  integer?: boolean; // integer axis ticks (queue depth)
  window_s?: number; // averaging window of a rate (throughput: 30)
}

export interface PerfSnapshot {
  schema: 1;
  service: string;
  sampled_at_ms: number;
  interval_hint_ms?: number;
  gpus: PerfGpu[];
  metrics: PerfMetric[];
  error?: string | null;
}

// ---------- module-level history store (survives tab switches, cleared on logout) ----------
export const PERF_INTERVAL_MS = 2000;
export const PERF_WINDOW_MS = 5 * 60 * 1000;
const GAP_FACTOR = 3; // a gap longer than 3 intervals breaks the line

interface Sample { t: number; snap: PerfSnapshot }
let samples: Sample[] = [];
let lastError: string | null = null;
const listeners = new Set<() => void>();
const emit = () => listeners.forEach((l) => l());

export function pushPerfSample(snap: PerfSnapshot) {
  const t = Date.now(); // client clock: avoids skew between browser and server
  samples = [...samples.filter((s) => s.t >= t - PERF_WINDOW_MS - PERF_INTERVAL_MS), { t, snap }];
  lastError = snap.error ?? null;
  emit();
}
export function setPerfError(message: string | null) { lastError = message; emit(); }
export function clearPerfHistory() { samples = []; lastError = null; emit(); }

function usePerfStore() {
  const [, setTick] = useState(0);
  useEffect(() => {
    const l = () => setTick((n) => n + 1);
    listeners.add(l);
    return () => { listeners.delete(l); };
  }, []);
  return { samples, lastError };
}

/** Latest snapshot (or null) for the header GPU chip: usePerfLatest()?.gpus.filter(g => g.used_by_service). */
export function usePerfLatest(): PerfSnapshot | null {
  const { samples: hist } = usePerfStore();
  return hist.length ? hist[hist.length - 1].snap : null;
}

/** Mount once in App while logged in. Polls every 2 s, pauses in hidden tabs, never overlaps. */
export function usePerfFeed(fetcher: (signal: AbortSignal) => Promise<PerfSnapshot>, enabled: boolean) {
  useVisiblePolling(async (signal) => {
    try {
      const snap = await fetcher(signal);
      if (!signal.aborted) pushPerfSample(snap);
    } catch (err) {
      if (!signal.aborted) setPerfError(err instanceof Error ? err.message : String(err));
      throw err;
    }
  }, PERF_INTERVAL_MS, enabled);
}

// ---------- formatting ----------
const nf = (d: number) => new Intl.NumberFormat("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
export const mibToGb = (mib: number | null | undefined) => (mib == null ? null : mib / 1024);
export function fmt(v: number | null | undefined, decimals = 1, unit = "") {
  if (v == null || !Number.isFinite(v)) return "—";
  const s = nf(decimals).format(v);
  if (!unit) return s;
  return unit === "%" || unit === "x" ? `${s}${unit === "x" ? "×" : "%"}` : `${s} ${unit}`;
}
export const gpuTitle = (g: PerfGpu) => `${g.nvml_index != null ? `GPU ${g.nvml_index} · ` : ""}${g.name.replace(/^NVIDIA\s+/i, "")}`;

/** Axis tick label: trailing zeros dropped ("0", "1.25", "2.5"), unit only for % and ×. */
function fmtTick(v: number, decimals: number, unit: string) {
  const s = new Intl.NumberFormat("en-US", { maximumFractionDigits: decimals }).format(v);
  return unit === "%" ? `${s}%` : unit === "x" ? `${s}×` : s;
}

function niceCeil(x: number) {
  if (!(x > 0)) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(x)));
  for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= x - 1e-9) return m * p;
  return 10 * p;
}

/** Fewest decimals (0-2) that print every tick within 0.5% of the axis: 0.25 stays "0.25", 2.75 GB becomes "2.8", 5.997 GB "6". */
function tickDecimals(ticks: number[], yMax: number) {
  for (let d = 0; d < 2; d++) {
    const f = Math.pow(10, d);
    if (ticks.every((v) => Math.abs(Math.round(v * f) / f - v) <= yMax * 0.005)) return d;
  }
  return 2;
}

/** Integer axis (queues): whole-number step 1/2/5 x 10^n, at most 4 steps, floor 4. Peak 5 gives 0/2/4/6, peak 9 gives 0/5/10. */
function integerAxis(peak: number) {
  const target = Math.max(4, Math.ceil(peak));
  let step = 1;
  for (let p = 1; step * 4 < target; p *= 10) {
    step = [1, 2, 5].map((m) => m * p).find((s) => s * 4 >= target) ?? 10 * p;
  }
  const yMax = Math.ceil(target / step) * step;
  const ticks: number[] = [];
  for (let v = 0; v <= yMax; v += step) ticks.push(v);
  return { yMax, ticks };
}

// ---------- chart ----------
interface Series { key: string; label: string; color: string; area?: boolean; values: (number | null)[]; decimals: number; unit: string }
interface ChartProps {
  title: string;
  subtitle?: string;
  times: number[];
  series: Series[];
  yMax: number;
  yTicks: number[];
  tickDecimals: number;
  yUnit: string;
  now: number;
}

const H = 160;
const M = { l: 44, r: 10, t: 8, b: 22 };

function TimeChart({ title, subtitle, times, series, yMax, yTicks, tickDecimals, yUnit, now }: ChartProps) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(480);
  const [hover, setHover] = useState<number | null>(null);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => setWidth(Math.max(200, Math.round(entries[0].contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const pw = width - M.l - M.r;
  const ph = H - M.t - M.b;
  const t0 = now - PERF_WINDOW_MS;
  const x = (t: number) => M.l + ((t - t0) / PERF_WINDOW_MS) * pw;
  const y = (v: number) => M.t + ph - (Math.min(Math.max(v, 0), yMax) / yMax) * ph;

  const pathFor = (values: (number | null)[], area: boolean) => {
    let d = "";
    let segStart: number | null = null;
    let prevT: number | null = null;
    const close = (lastIdx: number) => {
      if (area && segStart != null) d += ` L${x(times[lastIdx]).toFixed(1)},${y(0).toFixed(1)} L${x(times[segStart]).toFixed(1)},${y(0).toFixed(1)} Z`;
    };
    values.forEach((v, i) => {
      const t = times[i];
      const broken = v == null || prevT == null || t - prevT > GAP_FACTOR * PERF_INTERVAL_MS;
      if (v == null) {
        if (segStart != null) close(i - 1);
        segStart = null; prevT = t; return;
      }
      if (broken || segStart == null) {
        if (segStart != null) close(i - 1);
        d += ` M${x(t).toFixed(1)},${y(v).toFixed(1)}`;
        segStart = i;
      } else {
        d += ` L${x(t).toFixed(1)},${y(v).toFixed(1)}`;
      }
      prevT = t;
    });
    if (segStart != null) close(values.length - 1);
    return d.trim();
  };

  const onMove = (e: ReactPointerEvent<SVGSVGElement>) => {
    if (!times.length) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const t = t0 + ((e.clientX - rect.left - M.l) / pw) * PERF_WINDOW_MS;
    let best = 0;
    times.forEach((tt, i) => { if (Math.abs(tt - t) < Math.abs(times[best] - t)) best = i; });
    setHover(best);
  };

  const last = times.length - 1;
  const xTicks = [5, 4, 3, 2, 1, 0];
  const hoverT = hover != null ? times[hover] : null;

  return (
    <figure className="perf-chart">
      <figcaption className="perf-chart-head">
        <span className="perf-chart-title">{title}</span>
        {subtitle && <span className="hint">{subtitle}</span>}
        <span className="perf-legend">
          {series.map((s) => (
            <span key={s.key} className="perf-legend-item">
              <i style={{ background: s.color }} />
              {s.label} <b>{fmt(last >= 0 ? s.values[last] : null, s.decimals, s.unit)}</b>
            </span>
          ))}
        </span>
      </figcaption>
      <div className="perf-plot" ref={wrapRef}>
        <svg width={width} height={H} role="img" aria-label={`${title}, last 5 minutes`}
          onPointerMove={onMove} onPointerLeave={() => setHover(null)}>
          {yTicks.map((v) => (
            <g key={v}>
              <line x1={M.l} x2={width - M.r} y1={y(v)} y2={y(v)} className="perf-grid-line" />
              <text x={M.l - 6} y={y(v)} className="perf-tick" textAnchor="end" dominantBaseline="middle">
                {fmtTick(v, tickDecimals, yUnit)}
              </text>
            </g>
          ))}
          {xTicks.map((m) => (
            <text key={m} x={x(now - m * 60000)} y={H - 6} className="perf-tick"
              textAnchor={m === 5 ? "start" : m === 0 ? "end" : "middle"}>
              {m === 0 ? "now" : `−${m} min`}
            </text>
          ))}
          {series.map((s) => (
            <path key={s.key} d={pathFor(s.values, !!s.area)} fill={s.area ? s.color : "none"}
              fillOpacity={s.area ? 0.22 : undefined} stroke={s.color} strokeWidth={s.area ? 1.25 : 2}
              strokeLinejoin="round" strokeLinecap="round" />
          ))}
          {hoverT != null && (
            <g>
              <line x1={x(hoverT)} x2={x(hoverT)} y1={M.t} y2={M.t + ph} className="perf-crosshair" />
              {series.map((s) => s.values[hover!] != null && (
                <circle key={s.key} cx={x(hoverT)} cy={y(s.values[hover!]!)} r={3.5} fill={s.color} />
              ))}
            </g>
          )}
          {times.length < 2 && (
            <text x={M.l + pw / 2} y={M.t + ph / 2} className="perf-empty" textAnchor="middle">Collecting data…</text>
          )}
        </svg>
        {hoverT != null && (
          <div className="perf-tooltip" style={{ left: Math.min(Math.max(x(hoverT) + 10, 0), width - 180) }}>
            <div className="perf-tooltip-time">
              {new Date(hoverT).toLocaleTimeString("en-GB", { hour12: false })} · {Math.round((now - hoverT) / 1000)} s ago
            </div>
            {series.map((s) => (
              <div key={s.key} className="perf-legend-item">
                <i style={{ background: s.color }} />{s.label} <b>{fmt(s.values[hover!], s.decimals, s.unit)}</b>
              </div>
            ))}
          </div>
        )}
      </div>
    </figure>
  );
}

const css = (name: string) => `var(${name})`;
// "Latency (ms)", but no "Documents/s (docs/s)": a rate label already names its unit.
const withUnit = (label: string, unit: string) =>
  (unit && unit !== "%" && unit !== "x" && !(unit.endsWith("/s") && label.endsWith("/s")) ? `${label} (${unit})` : label);

/** Dashboard section "Performance". Renders from the shared store; no fetching here. */
export function PerfPanel({ title = "Performance" }: { title?: string }) {
  const { samples: hist, lastError: error } = usePerfStore();
  const [now, setNow] = useState(Date.now());
  useEffect(() => { setNow(Date.now()); }, [hist]);

  const latest = hist.length ? hist[hist.length - 1].snap : null;
  const times = hist.map((s) => s.t);
  const gpus = (latest?.gpus ?? []).filter((g) => g.used_by_service);
  const findGpu = (snap: PerfSnapshot, g: PerfGpu) =>
    snap.gpus.find((o) => (g.uuid && o.uuid === g.uuid) || (!g.uuid && o.nvml_index === g.nvml_index)) ?? null;

  return (
    <section className="panel" aria-labelledby="perf-title">
      <div className="panel-head">
        <h2 id="perf-title">{title}</h2>
        <span className="hint">Last 5 minutes · every 2 s · paused while this browser tab is hidden</span>
      </div>
      {error && <div className="banner danger" role="status">Performance data unavailable: {error}</div>}
      {!latest && !error && <div className="state"><span className="spinner" />Loading performance data…</div>}
      {latest && gpus.length === 0 && (
        <div className="banner info">No GPU in use by this service (CPU mode or no NVIDIA driver). GPU charts are hidden.</div>
      )}
      <div className="perf-grid">
        {gpus.map((g) => {
          const totalGb = mibToGb(g.mem_total_mib) ?? 0;
          const vramTicks = totalGb > 0 ? [0, 0.25, 0.5, 0.75, 1].map((f) => f * totalGb) : [0, 1];
          const meta = [
            g.temp_c != null ? `${Math.round(g.temp_c)} °C` : null,
            g.power_w != null ? `${Math.round(g.power_w)}${g.power_limit_w ? ` / ${Math.round(g.power_limit_w)}` : ""} W` : null,
          ].filter(Boolean).join(" · ");
          const per = hist.map((s) => findGpu(s.snap, g));
          return [
            <TimeChart key={`${g.uuid ?? g.nvml_index}-util`} now={now} times={times}
              title={`GPU utilization · ${gpuTitle(g)}`} subtitle={meta || undefined}
              yMax={100} yTicks={[0, 25, 50, 75, 100]} tickDecimals={0} yUnit="%"
              series={[{ key: "util", label: "Whole card", color: css("--series-util"), decimals: 0, unit: "%",
                values: per.map((p) => p?.util_pct ?? null) }]} />,
            <TimeChart key={`${g.uuid ?? g.nvml_index}-vram`} now={now} times={times}
              title={`VRAM (GB) · ${gpuTitle(g)}`} subtitle={`Axis = card total ${fmt(totalGb, 1, "GB")}`}
              yMax={totalGb || 1} yTicks={vramTicks} tickDecimals={tickDecimals(vramTicks, totalGb || 1)} yUnit="GB"
              series={[
                { key: "card", label: "Card used (all processes)", color: css("--series-card"), area: true, decimals: 1, unit: "GB",
                  values: per.map((p) => mibToGb(p?.mem_used_mib)) },
                { key: "svc", label: `This service (${latest!.service})`, color: css("--series-service"), decimals: 1, unit: "GB",
                  values: per.map((p) => mibToGb(p?.process_mib)) },
              ]} />,
          ];
        })}
        {(latest?.metrics ?? []).map((m) => {
          const values = hist.map((s) => s.snap.metrics.find((o) => o.key === m.key)?.value ?? null);
          const peak = Math.max(0, ...values.map((v) => v ?? 0));
          const axis = m.integer
            ? integerAxis(peak)
            : (() => { const max = niceCeil(Math.max(peak, 1)); return { yMax: max, ticks: [0, 0.25, 0.5, 0.75, 1].map((f) => f * max) }; })();
          const { yMax, ticks } = axis;
          const decimals = m.decimals ?? (m.integer ? 0 : 1);
          const color = m.key === "queue" ? css("--series-queue") : m.key === "latency" ? css("--series-latency") : css("--series-service");
          return (
            <TimeChart key={m.key} now={now} times={times} title={withUnit(m.label, m.unit)}
              yMax={yMax} yTicks={ticks} tickDecimals={tickDecimals(ticks, yMax)} yUnit={m.unit}
              series={[{ key: m.key, label: m.window_s ? `${m.window_s} s average` : "Current", color, decimals, unit: m.unit, values }]} />
          );
        })}
      </div>
    </section>
  );
}
