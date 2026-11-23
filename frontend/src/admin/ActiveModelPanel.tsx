import { formatMibAsGb, type DashboardSnapshot, type EngineInfo, type ModelDownloadStatus } from "../shared/api";

const HELP = {
  modelSelect: "Checkpoint of this engine for Preload, Warmup, Reload and the default. The default for the next server start changes only after Save as default.",
  preload: "Loads the selected model into memory without a test run.",
  warmup: "Runs a short test generation so CUDA kernels and caches are ready.",
  reload: "Unloads and loads the selected model again. Needed after dtype, compile or graph changes.",
  unload: "Removes the model from memory and empties the CUDA cache as far as possible.",
  freeMemory: "Releases unused VRAM. OmniVoice stays loaded; Ming and Audio8 close while idle and load again with the next request.",
  saveDefault: "Saves the selected model as the default for new requests and server starts.",
  refreshCache: "Reads the local model cache again.",
  download: "Starts a Hugging Face download in the background. Afterwards load the model with Preload or Warmup.",
  deleteCache: "Deletes the local cache of this model in the model directory.",
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
  return "—";
}

function formatModelStatus(status: string) {
  if (status === "ready") return "Ready";
  if (status === "downloading") return "Downloading";
  if (status === "partial") return "Partial";
  if (status === "error") return "Error";
  return "Missing";
}

function resolveModelPath(model: ModelDownloadStatus) {
  return model.local_path || model.cache_path || model.storage_root || "—";
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

  const processMb = overview.gpu_process_memory_mb;
  const vramText = usesGpu && overview.gpu_memory_total_mb > 0
    ? `This service ${formatMibAsGb(processMb)} · card ${formatMibAsGb(overview.gpu_memory_used_mb)} / ${formatMibAsGb(overview.gpu_memory_total_mb)}`
    : "—";
  const confirmUnload = () => {
    if (window.confirm(`Unload model "${selectedModel}"?

Requests wait until it is loaded again (the next request or Preload loads it).`)) onOperation("unload");
  };
  const confirmDelete = () => {
    if (window.confirm(`Delete cached model "${selectedModel}"?

It must be downloaded again before it can be loaded.`)) onDelete(selectedModel);
  };

  return (
    <div className="panel" aria-labelledby="active-model-title">
      <div className="panel-head">
        <h2 id="active-model-title">Active model · {engine.label}</h2>
        <div className="button-row">
          <span className={`badge ${engine.loaded ? "ok" : "plain"}`}>{engine.loaded ? "Loaded" : "Not loaded"}</span>
          <span className={`badge ${downloadStatus === "ready" ? "ok" : downloadStatus === "downloading" ? "warn" : downloadStatus === "error" ? "danger" : "plain"}`}>{formatModelStatus(downloadStatus)}</span>
        </div>
      </div>
      {models.length > 1 ? (
        <div className="field-grid four">
          <label>Model<select value={selectedModel} onChange={(event) => onPickModel(event.target.value)}>{models.map((model) => <option key={model} value={model}>{model}</option>)}</select><span className="hint">{HELP.modelSelect}</span></label>
        </div>
      ) : null}
      <dl className="kv-list">
        <div><dt>Model</dt><dd className="mono">{selectedModel || "—"}</dd></div>
        <div><dt>Storage</dt><dd className="mono">{download ? resolveModelPath(download) : engine.model_dir || "—"}</dd></div>
        <div><dt>Size</dt><dd>{download ? formatModelSize(download) : "—"}</dd></div>
        <div><dt>Runtime</dt><dd title={deviceLabel}>{[engine.id, usesGpu ? deviceLabel : null, hasDtype ? settings.torch_dtype : null].filter(Boolean).join(" · ")}</dd></div>
        {attention ? <div><dt>Attention</dt><dd title={settings.attention_backend_note ?? ""}>{attention}{hasCompile && settings.attention_backend_note ? " (fallback)" : ""}</dd></div> : null}
        {usesGpu ? <div><dt>VRAM</dt><dd>{vramText}</dd></div> : null}
        {hasCompile ? <div><dt>Compile</dt><dd>{settings.compile_model ? "LLM on" : "LLM off"}</dd></div> : null}
      </dl>
      {download?.error ? <div className="banner danger">{download.error}</div> : null}
      {failedJob ? <div className="banner danger">Last job failed: {failedJob}</div> : null}
      <div className="button-row">
        <button type="button" onClick={() => onOperation("warmup")} disabled={actionBusy} title={HELP.warmup}>Warmup</button>
        <button className="secondary-button" type="button" onClick={() => onOperation("preload")} disabled={actionBusy} title={HELP.preload}>Preload</button>
        <button className="secondary-button" type="button" onClick={() => onOperation("reload")} disabled={actionBusy} title={HELP.reload}>Reload</button>
        {usesGpu ? <button className="secondary-button" type="button" onClick={onFreeMemory} disabled={memoryBusy || busy} aria-busy={memoryBusy} title={HELP.freeMemory}>{memoryBusy ? "Freeing…" : "Free VRAM"}</button> : null}
        {models.length > 1 ? <button className="secondary-button" type="button" onClick={() => onSetDefault(selectedModel)} disabled={actionBusy || selectedModel === settings.default_model} title={HELP.saveDefault}>Save as default</button> : null}
        <button className="secondary-button" type="button" onClick={onRefreshCache} disabled={cacheRefreshing} aria-busy={cacheRefreshing} title={HELP.refreshCache}>{cacheRefreshing ? "Refreshing…" : "Refresh cache"}</button>
        {engine.downloadable ? (
          <button className="secondary-button" type="button" onClick={() => onDownload(selectedModel)} disabled={downloadStatus === "downloading" || actionBusy} title={HELP.download}>
            {downloadStatus === "downloading" ? "Downloading…" : downloadingId === selectedModel ? "Starting…" : downloadStatus === "ready" ? "Download again" : "Download"}
          </button>
        ) : null}
        <button className="danger-button" type="button" onClick={confirmUnload} disabled={actionBusy} title={HELP.unload}>Unload</button>
        {canDeleteCache ? <button className="danger-button" type="button" onClick={confirmDelete} disabled={!download?.cache_path || downloadStatus === "downloading" || actionBusy} title={HELP.deleteCache}>Delete cache</button> : null}
      </div>
      <p className="hint">Warmup runs a short test generation; Preload only loads the weights; Reload is needed after dtype, compile or graph changes. Unload frees the GPU until the next request; Free VRAM only releases unused memory.</p>
    </div>
  );
}
