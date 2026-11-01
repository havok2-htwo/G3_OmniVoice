import type { DashboardSnapshot, EngineInfo, ModelDownloadStatus } from "../shared/api";

const HELP = {
  modelSelect: "Checkpoint dieser Engine fuer Preload, Warmup, Reload und Default. Wirkt fuer den naechsten Serverstart erst nach dem Speichern.",
  preload: "Laedt das gewaehlte Modell ohne extra Benchmark-Run in den Speicher.",
  warmup: "Fuehrt eine kurze Testgenerierung aus, damit CUDA/Triton/Kernels und Caches vorbereitet sind.",
  reload: "Entlaedt und laedt das gewaehlte Modell neu. Noetig nach dtype/compile/graph-Aenderungen.",
  unload: "Entfernt das Modell aus dem Speicher und leert CUDA Cache soweit moeglich.",
  freeMemory: "Gibt ungenutzten VRAM frei. OmniVoice bleibt geladen, Ming wird im Leerlauf geschlossen und laedt mit der naechsten Anfrage neu.",
  saveDefault: "Speichert das gewaehlte Modell als Default fuer neue Requests und Startups.",
  refreshCache: "Liest den lokalen Modellcache neu ein.",
  download: "Startet einen Hugging-Face-Download im Hintergrund. Danach das Modell mit Preload/Warmup in den Speicher laden.",
  deleteCache: "Loescht den lokalen Cache dieses Modells im Model directory.",
};

function formatModelSize(model: ModelDownloadStatus) {
  const diskSize = Number(model.size_on_disk_gb);
  if (Number.isFinite(diskSize) && diskSize > 0) {
    return `${diskSize.toFixed(diskSize >= 10 ? 1 : 2)} GB`;
  }
  const approxSize = Number(model.approx_size_gb);
  if (Number.isFinite(approxSize) && approxSize > 0) {
    return `~${approxSize.toFixed(approxSize >= 10 ? 1 : 2)} GB`;
  }
  return "-";
}

function formatModelStatus(status: string) {
  if (status === "ready") return "Ready";
  if (status === "downloading") return "Downloading";
  if (status === "partial") return "Partial";
  if (status === "error") return "Error";
  return "Missing";
}

function resolveModelPath(model: ModelDownloadStatus) {
  return model.local_path || model.cache_path || model.storage_root || "-";
}

/** The former Model Control, for the one model that runs: only what can work on this engine. */
export function ActiveModelPanel({ engine, snapshot, selectedModel, download, deviceLabel, busy, memoryBusy, cacheRefreshing, downloadingId, onPickModel, onSetDefault, onOperation, onDownload, onDelete, onRefreshCache, onFreeMemory }: {
  engine: EngineInfo;
  snapshot: DashboardSnapshot;
  selectedModel: string;
  download?: ModelDownloadStatus;
  deviceLabel: string;
  busy: boolean;
  memoryBusy: boolean;
  cacheRefreshing: boolean;
  downloadingId: string | null;
  onPickModel: (modelId: string) => void;
  onSetDefault: (modelId: string) => void;
  onOperation: (kind: "preload" | "warmup" | "unload" | "reload") => void;
  onDownload: (modelId: string) => void;
  onDelete: (modelId: string) => void;
  onRefreshCache: () => void;
  onFreeMemory: () => void;
}) {
  const settings = snapshot.settings;
  const overview = snapshot.overview;
  const models = snapshot.selectable_models;
  const fields = engine.settings_fields;
  // An engine without a device setting runs on the CPU, so nothing of it sits in VRAM.
  const usesGpu = fields.includes("preferred_device");
  const hasDtype = fields.includes("torch_dtype");
  const hasCompile = fields.includes("compile_model");
  // The server only deletes what lives in the managed model directory; other engines keep their own folder. That
  // cache is the base weights, so a promoted local/* has none here: it is deleted in the Training tab.
  const canDeleteCache = fields.includes("model_directory") && !selectedModel.startsWith("local/");
  const attention = settings.attention_backend ?? (hasCompile ? settings.attention_implementation : null);
  const downloadStatus = download?.status || "missing";
  const actionBusy = busy || downloadingId === selectedModel;
  // A failure of the engine before a switch says nothing about this one.
  const lastJob = snapshot.jobs[0];
  const failedJob = lastJob?.status === "failed" && (!lastJob.model || snapshot.models.some((model) => model.model_id === lastJob.model)) ? lastJob.error_message : null;

  return (
    <>
      <div className="widget-header">
        <h3>Aktives Modell: {engine.label}</h3>
        <div className="inline-pills">
          <span className={`pill ${engine.loaded ? "active" : ""}`}>{engine.loaded ? "geladen" : "nicht geladen"}</span>
          <span className={`pill ${downloadStatus === "ready" ? "active" : ""}`}>{formatModelStatus(downloadStatus)}</span>
        </div>
      </div>
      {models.length > 1 ? (
        <div className="field-grid four">
          <label className="with-help" title={HELP.modelSelect}>Modell<select value={selectedModel} onChange={(event) => onPickModel(event.target.value)}>{models.map((model) => <option key={model} value={model}>{model}</option>)}</select></label>
        </div>
      ) : null}
      <div className="metric-list compact model-control-summary">
        <div className="metric-row"><span>Modell</span><strong className="mono path-cell">{selectedModel || "-"}</strong></div>
        <div className="metric-row"><span>Storage</span><strong className="mono path-cell">{download ? resolveModelPath(download) : engine.model_dir || "-"}</strong></div>
        <div className="metric-row"><span>Size</span><strong>{download ? formatModelSize(download) : "-"}</strong></div>
        <div className="metric-row"><span>Runtime</span><strong title={deviceLabel}>{[engine.id, usesGpu ? settings.preferred_device : null, hasDtype ? settings.torch_dtype : null].filter(Boolean).join(" / ")}</strong></div>
        {attention ? <div className="metric-row"><span>Attention</span><strong title={settings.attention_backend_note ?? ""}>{attention}{hasCompile && settings.attention_backend_note ? " (Fallback)" : ""}</strong></div> : null}
        {usesGpu ? <div className="metric-row"><span>VRAM</span><strong>{overview.gpu_memory_used_mb} / {overview.gpu_memory_total_mb} MB</strong></div> : null}
        {hasCompile ? <div className="metric-row"><span>Compile</span><strong>{settings.compile_model ? "llm on" : "llm off"}</strong></div> : null}
      </div>
      {download?.error ? <p className="model-error">{download.error}</p> : null}
      {failedJob ? <div className="message error">{failedJob}</div> : null}
      <div className="button-row compact">
        <button className="secondary-button" type="button" onClick={() => onOperation("preload")} disabled={actionBusy} title={HELP.preload}>Preload</button>
        <button className="primary-button" type="button" onClick={() => onOperation("warmup")} disabled={actionBusy} title={HELP.warmup}>Warmup</button>
        <button className="secondary-button" type="button" onClick={() => onOperation("reload")} disabled={actionBusy} title={HELP.reload}>Reload</button>
        <button className="ghost-button" type="button" onClick={() => onOperation("unload")} disabled={actionBusy} title={HELP.unload}>Unload</button>
        {usesGpu ? <button className="secondary-button" type="button" onClick={onFreeMemory} disabled={memoryBusy || busy} title={HELP.freeMemory}>Free VRAM</button> : null}
        {models.length > 1 ? <button className="secondary-button" type="button" onClick={() => onSetDefault(selectedModel)} disabled={actionBusy || selectedModel === settings.default_model} title={HELP.saveDefault}>Als Default speichern</button> : null}
        <button className="secondary-button" type="button" onClick={onRefreshCache} disabled={cacheRefreshing} title={HELP.refreshCache}>{cacheRefreshing ? "Refreshing..." : "Refresh Cache"}</button>
        {engine.downloadable ? (
          <button className="secondary-button" type="button" onClick={() => onDownload(selectedModel)} disabled={downloadStatus === "downloading" || actionBusy} title={HELP.download}>
            {downloadStatus === "downloading" ? "Downloading..." : downloadingId === selectedModel ? "Starting..." : downloadStatus === "ready" ? "Download Again" : "Download"}
          </button>
        ) : null}
        {canDeleteCache ? <button className="ghost-button" type="button" onClick={() => onDelete(selectedModel)} disabled={!download?.cache_path || downloadStatus === "downloading" || actionBusy} title={HELP.deleteCache}>Delete Cache</button> : null}
      </div>
    </>
  );
}
