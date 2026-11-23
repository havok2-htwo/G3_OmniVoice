import { useState } from "react";

import type { EngineInfo, EngineListResponse } from "../shared/api";

const ENGINE_SELECT_HELP = "The model this server runs. Switching needs no restart: running jobs finish, waiting and new requests go to the new model. Clients keep using the same endpoints.";
const DOWNLOAD_HELP = "Downloads the model files in the background, also for a model that is not active yet.";

const STATUS_LABELS: Record<string, string> = {
  active: "Active",
  ready: "Ready",
  downloadable: "Downloads on first load",
  downloading: "Downloading",
  missing_package: "Packages missing",
  missing_model: "Model missing",
};

/** One card per engine. A card only marks the choice, "Load model" starts the live switch. */
export function ModelPicker({ list, target, switchRunning, downloadingId, onSelect, onLoad, onDownload, onRefresh }: {
  list: EngineListResponse;
  target: string;
  switchRunning: boolean;
  /** Engine id or model id of a download being started, from the click until the server answers. */
  downloadingId: string | null;
  onSelect: (engineId: string) => void;
  onLoad: (engineId: string) => void;
  onDownload: (engine: EngineInfo) => void;
  onRefresh: () => void;
}) {
  const [confirmId, setConfirmId] = useState("");
  const status = list.switch;
  const engines = list.engines.filter((engine) => !engine.test_only || engine.active);

  function requestLoad(engine: EngineInfo) {
    if (engine.uses_voice_profiles) onLoad(engine.id);
    else setConfirmId(engine.id);
  }

  return (
    <section className="panel" aria-labelledby="model-picker-title">
      <div className="panel-head">
        <h2 id="model-picker-title">Engine</h2>
        <button className="secondary-button" type="button" onClick={onRefresh} disabled={switchRunning}>Reload status</button>
        <p className="hint">{ENGINE_SELECT_HELP}</p>
      </div>
      {status.state !== "idle" ? (
        <div className={`banner ${status.state === "failed" ? "danger" : "info"}`}>
          {status.from_backend ?? "—"} → {status.to_backend ?? "—"}: {status.state}
          {status.message ? ` - ${status.message}` : ""}
          {status.drained_jobs || status.cancelled_jobs || status.requeued_jobs
            ? ` (finished ${status.drained_jobs}, cancelled ${status.cancelled_jobs}, requeued ${status.requeued_jobs})`
            : ""}
          {status.error ? ` - ${status.error}` : ""}
        </div>
      ) : null}
      {list.env_override ? (
        <div className="banner warn">OMNIVOICE_TTS_RUNTIME_BACKEND={list.env_override} is set: a switch lasts until the next restart, then this engine starts again.</div>
      ) : null}
      <div className="model-bubble-grid">
        {engines.map((engine) => {
          const selectable = (engine.available || engine.active) && !switchRunning;
          const selected = target === engine.id;
          const canLoad = selected && !engine.active;
          return (
            <article key={engine.id} className={`model-bubble engine-card ${engine.active ? "active" : ""} ${selected ? "selected" : ""}`}>
              <label className={`engine-card-select ${selectable ? "" : "disabled"}`} title={ENGINE_SELECT_HELP}>
                <input
                  type="radio"
                  name="engine-target"
                  value={engine.id}
                  checked={selected}
                  disabled={!selectable}
                  onChange={() => {
                    setConfirmId("");
                    onSelect(engine.id);
                  }}
                />
                <span className="model-bubble-head">
                  <span>
                    <strong>{engine.label}</strong>
                    <span className="mono">{engine.model_id || engine.id}</span>
                  </span>
                  <span className="badge-row">
                    <span className={`badge ${engine.active || engine.available ? "ok" : engine.status === "downloading" ? "warn" : "plain"}`}>{STATUS_LABELS[engine.status] ?? engine.status}</span>
                    {engine.active ? <span className={`badge ${engine.loaded ? "ok" : "plain"}`}>{engine.loaded ? "Loaded" : "Not loaded"}</span> : null}
                    {engine.cpu_only ? <span className="badge info">CPU</span> : null}
                    {!engine.uses_voice_profiles ? <span className="badge plain">No voice profiles</span> : null}
                    {engine.approx_size_gb && !engine.model_present ? <span className="badge plain">~{engine.approx_size_gb.toLocaleString("en-US")} GB</span> : null}
                  </span>
                </span>
                <span className="widget-copy">{engine.description}</span>
                {engine.license ? <span className={`engine-license ${engine.license_restricted ? "restricted" : ""}`}>License: {engine.license}</span> : null}
                {engine.detail ? <span className="model-error">{engine.detail}</span> : null}
              </label>
              {confirmId === engine.id ? (
                <div className="banner danger" role="alert">
                  Load {engine.label}? It has no voice profiles: clients that request a profile voice get 404.
                  <div className="button-row compact engine-card-confirm">
                    <button
                      className="primary-button"
                      type="button"
                      disabled={switchRunning}
                      onClick={() => {
                        setConfirmId("");
                        onLoad(engine.id);
                      }}
                    >
                      Load anyway
                    </button>
                    <button className="ghost-button" type="button" onClick={() => setConfirmId("")}>Cancel</button>
                  </div>
                </div>
              ) : null}
              {canLoad || engine.downloadable ? (
                <div className="button-row compact">
                  {canLoad ? (
                    <button type="button" onClick={() => requestLoad(engine)} disabled={switchRunning || !engine.available || confirmId === engine.id} title={ENGINE_SELECT_HELP}>
                      {switchRunning ? "Switching…" : "Load model"}
                    </button>
                  ) : null}
                  {engine.downloadable ? (
                    <button className="secondary-button" type="button" onClick={() => onDownload(engine)} disabled={switchRunning || downloadingId === engine.id} title={DOWNLOAD_HELP}>
                      {downloadingId === engine.id ? "Starting…" : engine.model_present ? "Download again" : "Download"}
                    </button>
                  ) : null}
                </div>
              ) : null}
            </article>
          );
        })}
      </div>
    </section>
  );
}
