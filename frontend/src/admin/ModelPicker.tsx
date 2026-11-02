import { useState } from "react";

import type { EngineInfo, EngineListResponse } from "../shared/api";

const ENGINE_SELECT_HELP = "Modell des Servers. Der Wechsel laeuft ohne Neustart: laufende Jobs werden fertig, wartende und neue Anfragen gehen an das neue Modell. Clients nutzen dieselben Endpunkte weiter.";
const DOWNLOAD_HELP = "Startet den Download der Modelldateien im Hintergrund, auch fuer ein Modell, das noch nicht aktiv ist.";

const STATUS_LABELS: Record<string, string> = {
  active: "aktiv",
  ready: "bereit",
  downloadable: "laedt beim ersten Start",
  downloading: "Download laeuft",
  missing_package: "Pakete fehlen",
  missing_model: "Modell fehlt",
};

/** Step 1: one card per engine. A card only marks the choice, "Modell laden" starts the live switch. */
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
    <>
      <div className="widget-header">
        <h2>1. Modell</h2>
        <div className="button-row compact">
          <button className="secondary-button" type="button" onClick={onRefresh} disabled={switchRunning}>Status aktualisieren</button>
        </div>
      </div>
      {status.state !== "idle" ? (
        <div className={`message ${status.state === "failed" ? "error" : ""}`}>
          {status.from_backend ?? "-"} -&gt; {status.to_backend ?? "-"}: {status.state}
          {status.message ? ` - ${status.message}` : ""}
          {status.drained_jobs || status.cancelled_jobs || status.requeued_jobs
            ? ` (fertiggestellt ${status.drained_jobs}, abgebrochen ${status.cancelled_jobs}, neu eingereiht ${status.requeued_jobs})`
            : ""}
          {status.error ? ` - ${status.error}` : ""}
        </div>
      ) : null}
      {list.env_override ? (
        <div className="message">OMNIVOICE_TTS_RUNTIME_BACKEND={list.env_override} ist gesetzt: ein Wechsel gilt bis zum naechsten Neustart, danach startet wieder diese Engine.</div>
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
                  <span className="inline-pills">
                    <span className={`pill ${engine.active || engine.available ? "active" : ""}`}>{STATUS_LABELS[engine.status] ?? engine.status}</span>
                    {engine.active ? <span className={`pill ${engine.loaded ? "active" : ""}`}>{engine.loaded ? "geladen" : "nicht geladen"}</span> : null}
                    {engine.cpu_only ? <span className="pill">CPU</span> : null}
                    {!engine.uses_voice_profiles ? <span className="pill">keine Stimmprofile</span> : null}
                    {engine.approx_size_gb && !engine.model_present ? <span className="pill">~{engine.approx_size_gb.toLocaleString("de-DE")} GB</span> : null}
                  </span>
                </span>
                <span className="widget-copy">{engine.description}</span>
                {engine.license ? <span className={`engine-license ${engine.license_restricted ? "restricted" : ""}`}>Lizenz: {engine.license}</span> : null}
                {engine.detail ? <span className="model-error">{engine.detail}</span> : null}
              </label>
              {confirmId === engine.id ? (
                <div className="message error" role="alert">
                  {engine.label} laden? Stimmprofile sind dort nicht verfügbar und Clients mit Profilstimmen bekommen 404.
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
                      Trotzdem laden
                    </button>
                    <button className="ghost-button" type="button" onClick={() => setConfirmId("")}>Abbrechen</button>
                  </div>
                </div>
              ) : null}
              {canLoad || engine.downloadable ? (
                <div className="button-row compact">
                  {canLoad ? (
                    <button className="primary-button" type="button" onClick={() => requestLoad(engine)} disabled={switchRunning || !engine.available || confirmId === engine.id} title={ENGINE_SELECT_HELP}>
                      {switchRunning ? "Wechsel laeuft..." : "Modell laden"}
                    </button>
                  ) : null}
                  {engine.downloadable ? (
                    <button className="secondary-button" type="button" onClick={() => onDownload(engine)} disabled={switchRunning || downloadingId === engine.id} title={DOWNLOAD_HELP}>
                      {downloadingId === engine.id ? "Starting..." : engine.model_present ? "Download Again" : "Download"}
                    </button>
                  ) : null}
                </div>
              ) : null}
            </article>
          );
        })}
      </div>
    </>
  );
}
