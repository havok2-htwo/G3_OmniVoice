import { useCallback, useEffect, useState } from "react";

import { apiFetch, formatSeconds } from "../shared/api";

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
    catch (e) { fail(e, "Could not load the dataset summary."); }
  }, [adminKey, fail]);

  const loadCheckpoints = useCallback(async () => {
    try { setCheckpoints((await apiFetch<CheckpointListResponse>("/api/admin/finetune/checkpoints", { adminKey })).checkpoints); }
    catch (e) { fail(e, "Could not load the checkpoints."); }
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
      onMessage("Training started (preprocess, then the trainer as a subprocess).");
    } catch (e) { fail(e, "Could not start the training."); }
    finally { setBusy(""); }
  }

  async function handleCancel() {
    if (!run) return;
    if (!window.confirm("Stop the training run?\n\nThe current step is lost; saved checkpoints stay.")) return;
    try { await apiFetch(`/api/admin/finetune/train/${run.run_id}/cancel`, { adminKey, method: "POST" }); }
    catch (e) { fail(e, "Could not stop the training."); }
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
      onMessage(`Model promoted: ${item.model_id}. Pick it on the Models tab (Reload loads it).`);
    } catch (e) { fail(e, "Could not promote the checkpoint."); }
    finally { setBusy(""); }
  }

  async function handleDeleteCheckpoint(id: string) {
    const item = checkpoints.find((c) => c.checkpoint_id === id);
    if (!window.confirm(`Delete custom model "${item?.name ?? id}"?\n\nIts checkpoint folder is removed from disk.`)) return;
    try { await apiFetch(`/api/admin/finetune/checkpoints/${id}`, { adminKey, method: "DELETE" }); await loadCheckpoints(); }
    catch (e) { fail(e, "Could not delete the checkpoint."); }
  }

  const runActive = run && !RUN_TERMINAL.includes(run.status);
  const lossMax = Math.max(1e-6, ...(run?.loss_curve || [1]));

  return (
    <div className="grid grid-2 tab-grid">
      <section className="panel" aria-labelledby="tr-dataset">
        <div className="panel-head">
          <h2 id="tr-dataset">Dataset</h2>
          <button className="secondary-button" type="button" onClick={() => void loadSummary()}>Reload</button>
        </div>
        <p className="hint">{summary.total_clips} clips in total.</p>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Voice</th><th className="num">Clips</th><th className="num">Audio</th></tr></thead>
            <tbody>
              {summary.voices.map((v) => (
                <tr key={v.voice}><td>{v.voice}</td><td className="num">{v.clips}</td><td className="num">{formatSeconds(v.seconds)}</td></tr>
              ))}
              {!summary.voices.length ? <tr><td colSpan={3} className="empty">No material yet. Create clips on the Data generation tab first.</td></tr> : null}
            </tbody>
          </table>
        </div>
      </section>

      <section className="panel" aria-labelledby="tr-config">
        <div className="panel-head">
          <h2 id="tr-config">Training configuration</h2>
          <p className="hint">Full finetune from the base model (sdpa, bf16, low learning rate). The preprocess step (audio to codec-token cache) runs first automatically.</p>
        </div>
        <form className="field-grid two" onSubmit={(e) => { e.preventDefault(); void handleStart(); }}>
          <label>Name<input value={name} onChange={(e) => setName(e.target.value)} /></label>
          <label>Epochs<input type="number" min={1} max={100} value={epochs} onChange={(e) => setEpochs(Number(e.target.value))} /></label>
          <label>Steps<input type="number" min={0} value={stepsOverride} onChange={(e) => setStepsOverride(Number(e.target.value))} /><span className="hint">Overrides the epochs; 0 = off.</span></label>
          <label>Learning rate<input type="number" step={0.00001} min={0} value={learningRate} onChange={(e) => setLearningRate(Number(e.target.value))} /></label>
          <label>Dev share<input type="number" step={0.01} min={0} max={0.5} value={devFraction} onChange={(e) => setDevFraction(Number(e.target.value))} /></label>
          <label>batch_tokens<input type="number" min={512} value={batchTokens} onChange={(e) => setBatchTokens(Number(e.target.value))} /></label>
          <label>max_batch_size<input type="number" min={1} value={maxBatchSize} onChange={(e) => setMaxBatchSize(Number(e.target.value))} /></label>
          <label>grad_accum<input type="number" min={1} value={gradAccum} onChange={(e) => setGradAccum(Number(e.target.value))} /></label>
          <label>Attention
            <select value={attn} onChange={(e) => setAttn(e.target.value as typeof attn)}>
              <option value="sdpa">sdpa (safe on Windows)</option>
              <option value="flex_attention">flex_attention (faster, needs kernels)</option>
            </select>
          </label>
          <label>keep_last_n<input type="number" min={1} max={20} value={keepLastN} onChange={(e) => setKeepLastN(Number(e.target.value))} /></label>
          <div className="button-row span-all">
            <button type="submit" disabled={busy === "start" || !!runActive} aria-busy={busy === "start"}>{busy === "start" ? "Starting…" : "Start training"}</button>
            {runActive ? <button className="danger-button" type="button" onClick={() => void handleCancel()}>Stop</button> : null}
          </div>
        </form>
      </section>

      <section className="panel span-all" aria-labelledby="tr-run">
        <div className="panel-head"><h2 id="tr-run">Training run</h2></div>
        {!run ? <div className="state">No training run started yet.</div> : (
          <>
            <div className="grid grid-auto">
              <div className="card"><span className="card-label">Status</span><span className="card-value small">{run.status} · {run.phase}</span></div>
              <div className="card"><span className="card-label">Step</span><span className="card-value">{run.current_step}/{run.total_steps}</span></div>
              <div className="card"><span className="card-label">Loss</span><span className="card-value">{run.loss != null ? run.loss.toFixed(4) : "—"}</span></div>
              <div className="card"><span className="card-label">Eval loss</span><span className="card-value">{run.eval_loss != null ? run.eval_loss.toFixed(4) : "—"}</span></div>
              <div className="card"><span className="card-label">Learning rate</span><span className="card-value">{run.lr != null ? run.lr.toExponential(1) : "—"}</span></div>
              <div className="card"><span className="card-label">Steps/s</span><span className="card-value">{run.steps_per_sec != null ? run.steps_per_sec.toFixed(2) : "—"}</span></div>
              <div className="card"><span className="card-label">ETA</span><span className="card-value">{run.eta_ms != null ? formatSeconds(run.eta_ms / 1000) : "—"}</span></div>
              <div className="card"><span className="card-label">Train / dev</span><span className="card-value">{run.train_count}/{run.dev_count}</span></div>
            </div>
            <div className="bar" role="progressbar" aria-valuenow={run.pct} aria-valuemin={0} aria-valuemax={100}><span className="seg-service" style={{ width: `${run.pct}%` }} /></div>
            {run.loss_curve.length ? (
              <figure className="perf-chart loss-chart">
                <figcaption className="perf-chart-head"><span className="perf-chart-title">Training loss</span><span className="hint">max {lossMax.toFixed(4)}</span></figcaption>
                <svg viewBox="0 0 300 60" preserveAspectRatio="none" role="img" aria-label="Training loss curve">
                  <polyline fill="none" stroke="var(--series-service)" strokeWidth="2" vectorEffect="non-scaling-stroke"
                    points={run.loss_curve.map((v, i) => `${(i / Math.max(1, run.loss_curve.length - 1)) * 300},${60 - (v / lossMax) * 56 - 2}`).join(" ")} />
                </svg>
              </figure>
            ) : null}
            {run.error_message ? <div className="banner danger">{run.error_message}</div> : null}
            {run.checkpoint_dir ? (
              <div className="card">
                <span className="card-label">Checkpoint ready</span>
                <code className="mono">{run.checkpoint_dir}</code>
                <form className="button-row" onSubmit={(e) => { e.preventDefault(); void handlePromote(); }}>
                  <label className="sr-only" htmlFor="promote-name">Model name</label>
                  <input id="promote-name" className="grow" placeholder="Model name" value={promoteName} onChange={(e) => setPromoteName(e.target.value)} />
                  <button type="submit" disabled={busy === "promote" || !promoteName.trim()} aria-busy={busy === "promote"}>Promote to model</button>
                </form>
              </div>
            ) : null}
            {run.log_tail.length ? (
              <details>
                <summary className="dim">Log ({run.log_tail.length} lines)</summary>
                <pre className="log-pre">{run.log_tail.join("\n")}</pre>
              </details>
            ) : null}
          </>
        )}
      </section>

      <section className="panel span-all" aria-labelledby="tr-models">
        <div className="panel-head">
          <h2 id="tr-models">Custom models</h2>
          <button className="secondary-button" type="button" onClick={() => void loadCheckpoints()}>Reload</button>
          <p className="hint">Promoted checkpoints appear on the Models tab and load with Reload.</p>
        </div>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Name</th><th>Model id</th><th className="num">Steps</th><th>Folder</th><th className="actions"><span className="sr-only">Actions</span></th></tr></thead>
            <tbody>
              {checkpoints.map((c) => (
                <tr key={c.checkpoint_id}>
                  <td>{c.name}</td>
                  <td>{c.exists ? <span className="mono">{c.model_id}</span> : <span className="badge danger">Missing on disk</span>}</td>
                  <td className="num">{c.steps}</td>
                  <td><span className="mono">{c.dirname}</span></td>
                  <td className="actions"><button className="danger-button" type="button" onClick={() => void handleDeleteCheckpoint(c.checkpoint_id)}>Delete</button></td>
                </tr>
              ))}
              {!checkpoints.length ? <tr><td colSpan={5} className="empty">No custom models yet. Promote a finished training run above.</td></tr> : null}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
