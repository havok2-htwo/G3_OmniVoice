import { useCallback, useEffect, useState } from "react";

import { apiFetch } from "../shared/api";

interface DatasetVoiceSummary { voice: string; clips: number; seconds: number }
interface DatasetSummaryResponse { voices: DatasetVoiceSummary[]; total_clips: number }
interface TrainRunResponse {
  run_id: string;
  status: string;
  phase: string;
  base_model?: string | null;
  output_dir?: string | null;
  train_count: number;
  dev_count: number;
  total_steps: number;
  current_step: number;
  loss?: number | null;
  eval_loss?: number | null;
  lr?: number | null;
  steps_per_sec?: number | null;
  eta_ms?: number | null;
  pct: number;
  loss_curve: number[];
  checkpoint_dir?: string | null;
  log_tail: string[];
  error_message?: string | null;
}
interface CheckpointItem { checkpoint_id: string; name: string; model_id: string; dirname: string; base_model?: string | null; steps: number; created_at?: string | null; exists: boolean }
interface CheckpointListResponse { checkpoints: CheckpointItem[] }

const RUN_TERMINAL = ["completed", "failed", "cancelled"];

interface Props {
  adminKey: string;
  onMessage: (text: string) => void;
  onError: (text: string) => void;
}

export function TrainingPanel({ adminKey, onMessage, onError }: Props) {
  const [summary, setSummary] = useState<DatasetSummaryResponse>({ voices: [], total_clips: 0 });
  const [checkpoints, setCheckpoints] = useState<CheckpointItem[]>([]);
  const [run, setRun] = useState<TrainRunResponse | null>(null);
  const [busy, setBusy] = useState("");
  const [promoteName, setPromoteName] = useState("");

  // Config
  const [name, setName] = useState("omnivoice-finetune");
  const [epochs, setEpochs] = useState(3);
  const [stepsOverride, setStepsOverride] = useState(0);
  const [learningRate, setLearningRate] = useState(0.00003);
  const [devFraction, setDevFraction] = useState(0.05);
  const [batchTokens, setBatchTokens] = useState(8192);
  const [maxBatchSize, setMaxBatchSize] = useState(16);
  const [gradAccum, setGradAccum] = useState(4);
  const [attn, setAttn] = useState<"sdpa" | "flex_attention">("sdpa");
  const [keepLastN, setKeepLastN] = useState(2);

  const fail = useCallback((e: unknown, fb: string) => onError(e instanceof Error ? e.message : fb), [onError]);

  const loadSummary = useCallback(async () => {
    try { setSummary(await apiFetch<DatasetSummaryResponse>("/api/admin/finetune/dataset/summary", { adminKey })); }
    catch (e) { fail(e, "Dataset-Übersicht fehlgeschlagen."); }
  }, [adminKey, fail]);

  const loadCheckpoints = useCallback(async () => {
    try { setCheckpoints((await apiFetch<CheckpointListResponse>("/api/admin/finetune/checkpoints", { adminKey })).checkpoints); }
    catch (e) { fail(e, "Checkpoints konnten nicht geladen werden."); }
  }, [adminKey, fail]);

  const loadStatus = useCallback(async () => {
    try { setRun(await apiFetch<TrainRunResponse>("/api/admin/finetune/train/status", { adminKey })); }
    catch { /* no run yet */ }
  }, [adminKey]);

  useEffect(() => { void loadSummary(); void loadCheckpoints(); void loadStatus(); }, [loadSummary, loadCheckpoints, loadStatus]);

  useEffect(() => {
    if (!run || RUN_TERMINAL.includes(run.status)) return;
    const timer = window.setInterval(async () => {
      try {
        const next = await apiFetch<TrainRunResponse>(`/api/admin/finetune/train/status?run_id=${run.run_id}`, { adminKey });
        setRun(next);
        if (RUN_TERMINAL.includes(next.status)) { void loadCheckpoints(); }
      } catch { /* keep */ }
    }, 2000);
    return () => window.clearInterval(timer);
  }, [run, adminKey, loadCheckpoints]);

  async function handleStart() {
    setBusy("start");
    try {
      const res = await apiFetch<TrainRunResponse>("/api/admin/finetune/train", {
        adminKey, method: "POST",
        body: {
          name, epochs, steps_override: stepsOverride, learning_rate: learningRate, dev_fraction: devFraction,
          batch_tokens: batchTokens, max_batch_size: maxBatchSize, gradient_accumulation_steps: gradAccum,
          attn_implementation: attn, keep_last_n_checkpoints: keepLastN,
        },
      });
      setRun(res);
      onMessage("Training gestartet (Preprocess → Trainer als Subprozess).");
    } catch (e) { fail(e, "Training konnte nicht gestartet werden."); }
    finally { setBusy(""); }
  }

  async function handleCancel() {
    if (!run) return;
    try { await apiFetch(`/api/admin/finetune/train/${run.run_id}/cancel`, { adminKey, method: "POST" }); }
    catch (e) { fail(e, "Abbruch fehlgeschlagen."); }
  }

  async function handlePromote() {
    if (!run || !promoteName.trim()) return;
    setBusy("promote");
    try {
      const item = await apiFetch<CheckpointItem>(`/api/admin/finetune/train/${run.run_id}/promote`, {
        adminKey, method: "POST", body: { name: promoteName.trim() },
      });
      setPromoteName("");
      await loadCheckpoints();
      onMessage(`Modell promotet: ${item.model_id}. Im Overview-Tab unter Model Control auswählbar (Reload lädt es).`);
    } catch (e) { fail(e, "Promotion fehlgeschlagen."); }
    finally { setBusy(""); }
  }

  async function handleDeleteCheckpoint(id: string) {
    try { await apiFetch(`/api/admin/finetune/checkpoints/${id}`, { adminKey, method: "DELETE" }); await loadCheckpoints(); }
    catch (e) { fail(e, "Checkpoint konnte nicht gelöscht werden."); }
  }

  const runActive = run && !RUN_TERMINAL.includes(run.status);
  const lossMax = Math.max(1e-6, ...(run?.loss_curve || [1]));

  return (
    <section className="widget-grid">
      <section className="widget span-6">
        <div className="widget-header"><h2>Datensatz</h2>
          <div className="button-row compact"><button className="secondary-button" type="button" onClick={() => void loadSummary()}>Aktualisieren</button></div>
        </div>
        <div className="metric-list"><div className="metric-row"><span>Clips gesamt</span><strong>{summary.total_clips}</strong></div></div>
        <div className="job-list" style={{ marginTop: 8 }}>
          {summary.voices.map((v) => (
            <article key={v.voice} className="job-card"><strong>{v.voice}</strong><div className="inline-pills"><span className="pill">{v.clips} Clips</span><span className="pill">{v.seconds.toFixed(0)}s</span></div></article>
          ))}
          {!summary.voices.length ? <p className="widget-copy">Noch kein Material — erst im Data-Generation-Tab Clips erzeugen.</p> : null}
        </div>
      </section>

      <section className="widget span-6">
        <div className="widget-header"><h2>Training-Konfiguration</h2></div>
        <p className="widget-copy">Voll-Finetune vom Basismodell (sdpa, bf16, niedrige LR). Preprocess (Audio→Codec-Token-Cache) läuft automatisch davor.</p>
        <div className="field-grid two">
          <label>Name<input value={name} onChange={(e) => setName(e.target.value)} /></label>
          <label>Epochen<input type="number" min={1} max={100} value={epochs} onChange={(e) => setEpochs(Number(e.target.value))} /></label>
          <label>Steps (überschreibt Epochen, 0=aus)<input type="number" min={0} value={stepsOverride} onChange={(e) => setStepsOverride(Number(e.target.value))} /></label>
          <label>Learning Rate<input type="number" step={0.00001} min={0} value={learningRate} onChange={(e) => setLearningRate(Number(e.target.value))} /></label>
          <label>Dev-Anteil<input type="number" step={0.01} min={0} max={0.5} value={devFraction} onChange={(e) => setDevFraction(Number(e.target.value))} /></label>
          <label>batch_tokens<input type="number" min={512} value={batchTokens} onChange={(e) => setBatchTokens(Number(e.target.value))} /></label>
          <label>max_batch_size<input type="number" min={1} value={maxBatchSize} onChange={(e) => setMaxBatchSize(Number(e.target.value))} /></label>
          <label>grad_accum<input type="number" min={1} value={gradAccum} onChange={(e) => setGradAccum(Number(e.target.value))} /></label>
          <label>Attention
            <select value={attn} onChange={(e) => setAttn(e.target.value as typeof attn)}>
              <option value="sdpa">sdpa (Windows-sicher)</option>
              <option value="flex_attention">flex_attention (schneller, Kernel nötig)</option>
            </select>
          </label>
          <label>keep_last_n<input type="number" min={1} max={20} value={keepLastN} onChange={(e) => setKeepLastN(Number(e.target.value))} /></label>
        </div>
        <div className="button-row" style={{ marginTop: 12 }}>
          <button className="primary-button" type="button" disabled={busy === "start" || !!runActive} onClick={() => void handleStart()}>Training starten</button>
          {runActive ? <button className="ghost-button danger-button" type="button" onClick={() => void handleCancel()}>Stoppen</button> : null}
        </div>
      </section>

      <section className="widget span-12">
        <div className="widget-header"><h2>Trainingslauf</h2></div>
        {!run ? <p className="widget-copy">Noch kein Trainingslauf gestartet.</p> : (
          <>
            <div className="field-grid four">
              <div className="metric-row"><span>Status</span><strong>{run.status} · {run.phase}</strong></div>
              <div className="metric-row"><span>Step</span><strong>{run.current_step}/{run.total_steps}</strong></div>
              <div className="metric-row"><span>Loss</span><strong>{run.loss != null ? run.loss.toFixed(4) : "-"}</strong></div>
              <div className="metric-row"><span>Eval-Loss</span><strong>{run.eval_loss != null ? run.eval_loss.toFixed(4) : "-"}</strong></div>
              <div className="metric-row"><span>LR</span><strong>{run.lr != null ? run.lr.toExponential(1) : "-"}</strong></div>
              <div className="metric-row"><span>Steps/s</span><strong>{run.steps_per_sec != null ? run.steps_per_sec.toFixed(2) : "-"}</strong></div>
              <div className="metric-row"><span>ETA</span><strong>{run.eta_ms != null ? `${Math.round(run.eta_ms / 1000)}s` : "-"}</strong></div>
              <div className="metric-row"><span>Train/Dev</span><strong>{run.train_count}/{run.dev_count}</strong></div>
            </div>
            <div className="ft-progress" style={{ marginTop: 10 }}><div className="ft-progress-fill" style={{ width: `${run.pct}%` }} /></div>
            {run.loss_curve.length ? (
              <svg viewBox="0 0 300 60" style={{ width: "100%", height: 60, marginTop: 10 }} aria-label="Loss-Kurve">
                <polyline fill="none" stroke="#ff9a46" strokeWidth="2"
                  points={run.loss_curve.map((v, i) => `${(i / Math.max(1, run.loss_curve.length - 1)) * 300},${60 - (v / lossMax) * 56 - 2}`).join(" ")} />
              </svg>
            ) : null}
            {run.error_message ? <div className="message error" style={{ marginTop: 8 }}>{run.error_message}</div> : null}
            {run.checkpoint_dir ? (
              <div className="voice-card" style={{ marginTop: 10 }}>
                <strong>Checkpoint bereit</strong>
                <p className="widget-copy mono">{run.checkpoint_dir}</p>
                <div className="button-row">
                  <input placeholder="Modellname" value={promoteName} onChange={(e) => setPromoteName(e.target.value)} />
                  <button className="primary-button" type="button" disabled={busy === "promote" || !promoteName.trim()} onClick={() => void handlePromote()}>Als Modell übernehmen</button>
                </div>
              </div>
            ) : null}
            {run.log_tail.length ? (
              <details style={{ marginTop: 10 }}>
                <summary>Log ({run.log_tail.length} Zeilen)</summary>
                <pre className="mono" style={{ maxHeight: 220, overflow: "auto", fontSize: 12 }}>{run.log_tail.join("\n")}</pre>
              </details>
            ) : null}
          </>
        )}
      </section>

      <section className="widget span-12">
        <div className="widget-header"><h2>Custom-Modelle</h2>
          <div className="button-row compact"><button className="secondary-button" type="button" onClick={() => void loadCheckpoints()}>Aktualisieren</button></div>
        </div>
        <p className="widget-copy">Promotete Checkpoints erscheinen im Overview-Tab unter „Model Control" und werden per Reload geladen.</p>
        <div className="job-list">
          {checkpoints.map((c) => (
            <article key={c.checkpoint_id} className="job-card">
              <div className="inline-pills" style={{ justifyContent: "space-between" }}>
                <strong>{c.name}</strong>
                <span className="pill">{c.exists ? c.model_id : "fehlt auf Disk"}</span>
              </div>
              <div className="inline-pills"><span className="pill">{c.steps} Steps</span><span className="pill mono">{c.dirname}</span></div>
              <div className="button-row compact"><button className="ghost-button danger-button" type="button" onClick={() => void handleDeleteCheckpoint(c.checkpoint_id)}>Löschen</button></div>
            </article>
          ))}
          {!checkpoints.length ? <p className="widget-copy">Noch keine Custom-Modelle.</p> : null}
        </div>
      </section>
    </section>
  );
}
