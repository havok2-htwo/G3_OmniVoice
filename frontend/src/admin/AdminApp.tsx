import { useEffect, useMemo, useRef, useState } from "react";

import {
  apiFetch,
  clearStoredAdminKey,
  createWavBlobFromInt16Chunks,
  decodePcm16Base64,
  formatDate,
  formatMs,
  formatRealtime,
  formatSeconds,
  readStoredAdminKey,
  streamNdjson,
  streamSse,
  writeStoredAdminKey,
  type BenchmarkRunResponse,
  type DashboardSnapshot,
  type JobMetrics,
  type MemoryCleanupResponse,
  type ModelDownloadListResponse,
  type ModelDownloadStatus,
  type ModelOperationResponse,
  type ServerSettings,
  type SynthStreamEvent,
  type TaskType,
  type VllmModelsResponse,
  type WerBenchmarkRunResponse,
} from "../shared/api";
import {
  DEFAULT_VOICE_DESIGN_INSTRUCT,
  VOICE_DESIGN_GROUPS,
  setVoiceDesignValue,
  voiceDesignInstructOrDefault,
  voiceDesignValueForGroup,
} from "../shared/voiceDesign";

function inferTaskType(modelId: string): TaskType {
  if (modelId.endsWith("VoiceDesign")) return "VoiceDesign";
  if (modelId.endsWith("Base")) return "Base";
  return "CustomVoice";
}

const OMNIVOICE_DEFAULTS = {
  num_step: 32,
  guidance_scale: 2.0,
  t_shift: 0.1,
  denoise: true,
  preprocess_prompt: true,
  postprocess_output: true,
  audio_chunk_duration: 15.0,
  audio_chunk_threshold: 30.0,
  position_temperature: 5.0,
  class_temperature: 0.0,
} as const;

const WER_LANGUAGE_OPTIONS = [
  "Deutsch",
  "English",
  "Français",
  "Español",
  "Italiano",
  "Nederlands",
  "Polski",
  "Português",
  "Türkçe",
  "Русский",
  "Українська",
  "中文",
  "日本語",
  "한국어",
] as const;

const ADMIN_HELP = {
  adminKey: "Admin-Key fuer geschuetzte /api/admin Endpoints. Wird lokal im Browser gespeichert.",
  modelOpsModel: "Zielmodell fuer Download, Preload, Warmup, Reload und Default-Speicherung.",
  allowDownloads: "Erlaubt dem Backend, k2-fsa/OmniVoice aus Hugging Face in das lokale models-Verzeichnis zu laden.",
  downloadPreload: "Schaltet Downloads frei, laedt fehlende Modelldateien und haelt das Modell im Speicher.",
  preload: "Laedt das gewaehlte Modell ohne extra Benchmark-Run in den Speicher.",
  warmup: "Fuehrt eine kurze Testgenerierung aus, damit CUDA/Triton/Kernels und Caches vorbereitet sind.",
  reload: "Entlaedt und laedt das gewaehlte Modell neu. Noetig nach dtype/compile/graph-Aenderungen.",
  unload: "Entfernt das Modell aus dem Speicher und leert CUDA Cache soweit moeglich.",
  freeMemory: "Leert Python-GC, Prompt-Cache und PyTorch CUDA Cache, ohne das geladene Modell zu entladen.",
  saveDefault: "Speichert das gewaehlte Modell als Default fuer neue Requests und Startups.",
  modelDownloads: "Lokaler OmniVoice-Modellcache im Stil des TADA3B Panels: Status, Pfad und Download-Aktionen.",
  refreshDownloads: "Liest den lokalen Modellcache im aktuell eingetragenen Model directory neu ein.",
  downloadModel: "Startet einen Hugging-Face-Download im Hintergrund. Danach das Modell mit Preload/Warmup in den Speicher laden.",
  deleteModel: "Loescht den lokalen Cache fuer OmniVoice im gewaehlten Model directory.",
  quickModel: "Modell fuer die schnelle Admin-Synthese. AutoVoice, VoiceDesign und Base haben unterschiedliche Voice-Regeln.",
  quickVoice: "Stimme fuer Quick Synthesis. Custom Voices laufen mit OmniVoice-Base, VoiceDesign braucht keine feste Stimme.",
  quickText: "Text fuer den Streaming-Test. Der erste Satz bekommt einen Fast-Path, danach werden kompatible Saetze wieder gebatcht.",
  quickInstructions: "VoiceDesign akzeptiert nur feste Tags aus den Dropdowns. Freitext wird von OmniVoice abgelehnt.",
  quickSeed: "Seed fuer reproduzierbarere Runs. Leer lassen fuer zufaellige Ausgabe.",
  defaultModel: "Default-Modell fuer neue Jobs, Model Ops und den naechsten Serverstart.",
  defaultVoice: "Default-Stimme passend zum Default-Modell. Base braucht eine gespeicherte Custom Voice.",
  queueLimit: "Maximale Anzahl wartender Jobs, bevor neue Requests mit 429 abgelehnt werden.",
  activeRequests: "Maximale Anzahl paralleler Worker-Requests. Hoeher kann Durchsatz steigern, kostet aber RAM/VRAM.",
  batchSize: "Maximale Anzahl kompatibler Saetze/Requests pro echtem OmniVoice generate(text=[...]) Batch.",
  batchWait: "Wartezeit, um kompatible Queue-Items fuer einen groesseren Batch zu sammeln.",
  streamPrebuffer: "Audio-Puffer vor dem ersten Playback-Chunk. Hoeher stabilisiert Playback, erhoeht aber Latenz.",
  modelDirectory: "Lokaler Ordner fuer OmniVoice-Modelldateien und Hugging-Face-Cache.",
  whisperUrl: "Einzige Whisper URL. Host/Port reicht, z.B. http://192.168.0.200:7861; ein voller Endpoint wie /transcribe/ geht auch.",
  vllmBaseUrl: "OpenAI-kompatible vLLM Base URL fuer zufaellige WER-Referenzsaetze. Default: http://192.168.20.126:8000.",
  vllmModel: "Optionales vLLM Modell fuer WER-Saetze. Leer lassen nutzt das erste Modell aus /v1/models.",
  refreshVllmModels: "Liest die Modellliste aus dem konfigurierten vLLM Server neu aus.",
  werConcurrency: "TTS-Wellengroesse fuer WER-Benchmarks. Eine Welle wird zusammen in die Queue gelegt, damit OmniVoice echte Batches sieht.",
  werTranscriptionConcurrency: "Whisper-Parallelitaet nach der Audio-Erzeugung. Auf demselben GPU-System meist 1 lassen, damit ASR die TTS-Batches nicht ausbremst.",
  dtype: "Torch dtype beim Modell-Laden: fp16 meist schnell, bf16 oft stabil, fp32 langsam und speicherhungrig.",
  numStep: "OmniVoice Diffusion-/Sampling-Schritte. Mehr kann Qualitaet verbessern, kostet Zeit. Default: 32.",
  guidance: "Guidance Scale fuer die Steuerstaerke. Hoeher folgt Prompts staerker, kann aber Artefakte erzeugen. Default: 2.0.",
  duration: "Optionale Ziel-Dauer in Sekunden. Leer lassen fuer OmniVoice-Automatik. Default: auto.",
  tShift: "OmniVoice Zeitverschiebungsparameter. Nur aendern, wenn du gezielt Sampling-Verhalten testest. Default: 0.1.",
  positionTemperature: "Temperatur fuer die Positionsauswahl beim Token-Unmasking. Default: 5.0.",
  classTemperature: "Temperatur fuer die Token-Klassenwahl. 0 bedeutet greedy. Default: 0.0.",
  audioChunkDuration: "Interne OmniVoice-Ziellaenge fuer lange Audio-Chunks in Sekunden. Default: 15.0.",
  audioChunkThreshold: "Ab welcher geschaetzten Dauer OmniVoice internes Audio-Chunking nutzt. Default: 30.0.",
  sentenceChunking: "Zerlegt lange Texte in Saetze. Das macht Streaming frueher hoerbar und erlaubt Satz-Batching.",
  compileModel: "Aktiviert torch.compile fuer das LLM im OmniVoice-Modell. Erst nach Reload/Restart wirksam.",
  cudagraphSkip: "Laesst Inductor dynamische CUDAGraph-Shapes ueberspringen und unterdrueckt viele Shape-Warnungen.",
  autoVramTrim: "Leert nach jedem Batch den PyTorch CUDA Cache. Hilft nach zu grossen Batches, kann Durchsatz minimal senken.",
  warmupStartup: "Fuehrt beim Start automatisch Warmup aus. Gut fuer erste Latenz, verlaengert Startup.",
  denoise: "Fuegt den OmniVoice-Denoise-Token in die Generierung ein. Default: an.",
  preprocessPrompt: "Bereitet Referenzprompt und Clone-Audio vor dem Generieren auf. Default: an.",
  postprocessOutput: "Entfernt/saeubert Stille und fuegt kleine Fades/Pads an. Default: an.",
  voiceName: "Name fuer die gespeicherte Clone-Stimme im Voice-Katalog.",
  voiceRefText: "Exakter gesprochener Text des Samples. OmniVoice-Base braucht ihn fuer Voice Clone Prompts.",
  voiceSample: "Audio-Sample der Stimme. Backend normalisiert es nach 24 kHz WAV.",
  transcribeSample: "Schickt das Sample an den konfigurierten G3 Whisper Server und fuellt den Referenztext.",
  voiceConsent: "Bestaetigt, dass diese Stimme gespeichert und fuer lokale Tests genutzt werden darf.",
  benchmarkMode: "Traffic simuliert echte Nutzer-Requests mit zufaelligen Ankunftszeiten. Iterations ist der alte feste Paralleltest.",
  benchmarkText: "Satzpool fuer Traffic. Das Benchmark waehlt pro Request zufaellig mehrere Saetze daraus aus.",
  benchmarkDuration: "Traffic-Fenster, in dem neue Requests zufaellig eintreffen. Offene Generierungen duerfen danach noch bis zu 180s fertig laufen.",
  benchmarkRpm: "Zielrate fuer Traffic: wie viele Nutzer-Requests pro Minute eintreffen.",
  benchmarkSentenceRange: "Zufaellige Anzahl Saetze pro Nutzer-Request. Beispiel: mal 3, mal 5, mal 10 Saetze.",
  benchmarkSeed: "Optionaler Seed fuer reproduzierbare Request-Verteilung und Satz-Auswahl.",
  benchmarkIterations: "Gemessene Wiederholungen pro Case im Iterationsmodus. Warmups werden nicht in die Mittelwerte gerechnet.",
  benchmarkWarmups: "Nicht gemessene Vorlaeufe zum Fuellen von Modell-, CUDA- und Prompt-Caches.",
  benchmarkParallel: "Iterationsmodus: Anzahl gleichzeitig gestarteter Jobs. Trafficmodus ignoriert dieses Feld.",
  werCount: "Anzahl zufaelliger Saetze, die vLLM erzeugt und OmniVoice danach vertont.",
  werLanguage: "Sprache fuer die von vLLM generierten Referenzsaetze. TTS laeuft weiter mit Auto-Language.",
  werWords: "Wortbereich fuer vLLM-Saetze. Kuerzere Saetze machen Fehler leichter sichtbar.",
  werTolerance: "Levenshtein-Toleranz pro Wort. Bis zu 2 Buchstaben Unterschied gelten als Schreibweise, Einfuegungen/Loeschungen bleiben Fehler.",
  werTimeout: "Maximale Wartezeit pro TTS+Whisper Sample. Hilft, haengende Einzelrequests abzubrechen.",
  werSeed: "TTS-Seed fuer den WER-Run. Der Satzpool bleibt gleich, damit Seeds fair vergleichbar sind.",
  werSeedRange: "Wenn groesser 0, testet der WER-Benchmark alle Seeds von Seed bis Seed + Range und zeigt eine Bestenliste.",
  werPrompt: "Optionaler eigener vLLM Prompt. Er sollte ein JSON-Array von Saetzen erzeugen.",
};

function preferredBaseModel(models: DashboardSnapshot["models"]) {
  return (
    models.find((model) => model.model_id.endsWith("OmniVoice-Base"))?.model_id ||
    ""
  );
}

function voiceOptionsForTask(voices: DashboardSnapshot["voices"], taskType: TaskType) {
  if (taskType === "Base") return voices.filter((voice) => voice.source === "custom");
  if (taskType === "CustomVoice") return voices.filter((voice) => voice.source !== "custom");
  return [];
}

function firstVoiceValueForTask(voices: DashboardSnapshot["voices"], taskType: TaskType) {
  const options = voiceOptionsForTask(voices, taskType);
  const first = options[0];
  if (!first) return "";
  return first.source === "custom" ? first.voice_id : first.name;
}

function voiceMatchesValue(voice: DashboardSnapshot["voices"][number], value: string) {
  return voice.voice_id === value || voice.name === value;
}

function applyOmniVoiceDefaults(settings: ServerSettings): ServerSettings {
  return {
    ...settings,
    num_step: settings.num_step ?? OMNIVOICE_DEFAULTS.num_step,
    guidance_scale: settings.guidance_scale ?? OMNIVOICE_DEFAULTS.guidance_scale,
    t_shift: settings.t_shift ?? OMNIVOICE_DEFAULTS.t_shift,
    denoise: settings.denoise ?? OMNIVOICE_DEFAULTS.denoise,
    preprocess_prompt: settings.preprocess_prompt ?? OMNIVOICE_DEFAULTS.preprocess_prompt,
    postprocess_output: settings.postprocess_output ?? OMNIVOICE_DEFAULTS.postprocess_output,
    cuda_memory_trim_after_batch: settings.cuda_memory_trim_after_batch ?? false,
    audio_chunk_duration: settings.audio_chunk_duration ?? OMNIVOICE_DEFAULTS.audio_chunk_duration,
    audio_chunk_threshold: settings.audio_chunk_threshold ?? OMNIVOICE_DEFAULTS.audio_chunk_threshold,
    position_temperature: settings.position_temperature ?? OMNIVOICE_DEFAULTS.position_temperature,
    class_temperature: settings.class_temperature ?? OMNIVOICE_DEFAULTS.class_temperature,
  };
}

async function copyToClipboard(value: string) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(value);
  }
}

const SPARK_CHART = {
  width: 260,
  height: 112,
  left: 42,
  right: 10,
  top: 10,
  bottom: 24,
} as const;

function sparkY(value: number, max: number) {
  const plotHeight = SPARK_CHART.height - SPARK_CHART.top - SPARK_CHART.bottom;
  return SPARK_CHART.top + (1 - Math.max(0, value) / Math.max(max, 1)) * plotHeight;
}

function buildSparkPath(values: number[]) {
  const max = Math.max(1, ...values.map((value) => Math.max(0, Number(value) || 0)));
  const plotWidth = SPARK_CHART.width - SPARK_CHART.left - SPARK_CHART.right;
  return values
    .map((value, index) => {
      const x = SPARK_CHART.left + (values.length <= 1 ? 0 : (index / (values.length - 1)) * plotWidth);
      const y = sparkY(Math.max(0, Number(value) || 0), max);
      return `${index === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(" ");
}

function formatGraphValue(value: number | undefined, label: string, suffix: string) {
  if (value === undefined || Number.isNaN(Number(value))) return "-";
  const numeric = Number(value);
  if (label.includes("Realtime")) return `${numeric.toFixed(numeric >= 10 ? 0 : 2)}x`;
  if (suffix.trim() === "ms") return `${Math.round(numeric)} ms`;
  if (suffix === "%") return `${Math.round(numeric)}%`;
  return `${Math.round(numeric)}${suffix}`;
}

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

function formatPercent(value?: number | null) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return `${(Number(value) * 100).toFixed(Number(value) >= 0.1 ? 1 : 2)}%`;
}

function resolveModelPath(model: ModelDownloadStatus) {
  return model.local_path || model.cache_path || model.storage_root || "-";
}

function MiniGraph({ label, values, color, suffix = "" }: { label: string; values: number[]; color: string; suffix?: string }) {
  const chartValues = values.slice(-80).map((value) => Math.max(0, Number(value) || 0));
  const max = Math.max(1, ...chartValues);
  const path = buildSparkPath(chartValues);
  const latest = values.at(-1);
  const yTicks = [max, max / 2, 0];
  const xTicks = [0, 0.5, 1];
  return (
    <article className="model-card">
      <div className="metric-row">
        <span>{label}</span>
        <strong>{formatGraphValue(latest, label, suffix)}</strong>
      </div>
      <svg className="sparkline-graph" viewBox={`0 0 ${SPARK_CHART.width} ${SPARK_CHART.height}`} aria-label={label}>
        {xTicks.map((tick) => {
          const x = SPARK_CHART.left + tick * (SPARK_CHART.width - SPARK_CHART.left - SPARK_CHART.right);
          return <line key={tick} x1={x} y1={SPARK_CHART.top} x2={x} y2={SPARK_CHART.height - SPARK_CHART.bottom} className="sparkline-grid vertical" />;
        })}
        {yTicks.map((tick) => {
          const y = sparkY(tick, max);
          return (
            <g key={tick}>
              <line x1={SPARK_CHART.left} y1={y} x2={SPARK_CHART.width - SPARK_CHART.right} y2={y} className="sparkline-grid" />
              <text x={SPARK_CHART.left - 7} y={y + 3} textAnchor="end" className="sparkline-label">{formatGraphValue(tick, label, suffix)}</text>
            </g>
          );
        })}
        <line x1={SPARK_CHART.left} y1={SPARK_CHART.top} x2={SPARK_CHART.left} y2={SPARK_CHART.height - SPARK_CHART.bottom} className="sparkline-axis" />
        <line x1={SPARK_CHART.left} y1={SPARK_CHART.height - SPARK_CHART.bottom} x2={SPARK_CHART.width - SPARK_CHART.right} y2={SPARK_CHART.height - SPARK_CHART.bottom} className="sparkline-axis" />
        <text x={SPARK_CHART.left} y={SPARK_CHART.height - 5} className="sparkline-label">alt</text>
        <text x={SPARK_CHART.width - SPARK_CHART.right} y={SPARK_CHART.height - 5} textAnchor="end" className="sparkline-label">jetzt</text>
        {path ? <path d={path} fill="none" stroke={color} strokeWidth="2.8" strokeLinecap="round" strokeLinejoin="round" className="sparkline-path" /> : null}
      </svg>
    </article>
  );
}

export function AdminApp() {
  const [adminKeyInput, setAdminKeyInput] = useState(() => readStoredAdminKey());
  const [adminKey, setAdminKey] = useState(() => readStoredAdminKey());
  const [snapshot, setSnapshot] = useState<DashboardSnapshot | null>(null);
  const [settingsDraft, setSettingsDraft] = useState<ServerSettings | null>(null);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [authLoading, setAuthLoading] = useState(false);
  const [quickModel, setQuickModel] = useState("");
  const [quickVoice, setQuickVoice] = useState("");
  const [modelOpsModel, setModelOpsModel] = useState("");
  const [quickText, setQuickText] = useState("Das neue Adminpanel nutzt dieselbe Streaming-Pipeline wie die offene Demo.");
  const [quickInstructions, setQuickInstructions] = useState("");
  const [quickSeed, setQuickSeed] = useState("");
  const [quickMetrics, setQuickMetrics] = useState<JobMetrics | null>(null);
  const [quickAudioUrl, setQuickAudioUrl] = useState("");
  const [quickRunning, setQuickRunning] = useState(false);
  const [voiceName, setVoiceName] = useState("");
  const [voiceRefText, setVoiceRefText] = useState("");
  const [voiceConsent, setVoiceConsent] = useState(false);
  const [voiceFile, setVoiceFile] = useState<File | null>(null);
  const [voiceUploadPreviewUrl, setVoiceUploadPreviewUrl] = useState("");
  const [jobAudioUrls, setJobAudioUrls] = useState<Record<string, string>>({});
  const [voiceAudioUrls, setVoiceAudioUrls] = useState<Record<string, string>>({});
  const [overviewHistory, setOverviewHistory] = useState<Array<DashboardSnapshot["overview"] & { recorded_at: string }>>([]);
  const [benchmarks, setBenchmarks] = useState<BenchmarkRunResponse[]>([]);
  const [werBenchmarks, setWerBenchmarks] = useState<WerBenchmarkRunResponse[]>([]);
  const [modelDownloads, setModelDownloads] = useState<ModelDownloadStatus[]>([]);
  const [vllmModels, setVllmModels] = useState<string[]>([]);
  const [vllmModelsError, setVllmModelsError] = useState("");
  const [vllmModelsLoading, setVllmModelsLoading] = useState(false);
  const [benchmarkMode, setBenchmarkMode] = useState<"traffic" | "iterations">("traffic");
  const [benchmarkText, setBenchmarkText] = useState("Satz eins fuer den Lasttest. Satz zwei ist etwas laenger und simuliert einen echten Nutzer. Satz drei prueft die Queue. Satz vier testet Batching. Satz fuenf ist kurz. Satz sechs ist wieder ein bisschen ausfuehrlicher. Satz sieben nutzt dieselbe Stimme. Satz acht kommt zufaellig rein. Satz neun macht den p99-Wert spannend. Satz zehn beendet den Pool.");
  const [benchmarkDuration, setBenchmarkDuration] = useState(60);
  const [benchmarkRequestsPerMinute, setBenchmarkRequestsPerMinute] = useState(30);
  const [benchmarkMinSentences, setBenchmarkMinSentences] = useState(3);
  const [benchmarkMaxSentences, setBenchmarkMaxSentences] = useState(10);
  const [benchmarkSeed, setBenchmarkSeed] = useState("");
  const [benchmarkIterations, setBenchmarkIterations] = useState(3);
  const [benchmarkWarmups, setBenchmarkWarmups] = useState(1);
  const [benchmarkParallel, setBenchmarkParallel] = useState(8);
  const [werCount, setWerCount] = useState(100);
  const [werLanguage, setWerLanguage] = useState("Deutsch");
  const [werMinWords, setWerMinWords] = useState(5);
  const [werMaxWords, setWerMaxWords] = useState(16);
  const [werTolerance, setWerTolerance] = useState(2);
  const [werTimeout, setWerTimeout] = useState(180);
  const [werPrompt, setWerPrompt] = useState("");
  const [werSeed, setWerSeed] = useState("");
  const [werSeedRange, setWerSeedRange] = useState(0);
  const [modelOpsBusy, setModelOpsBusy] = useState(false);
  const [modelDownloadAction, setModelDownloadAction] = useState<{ kind: "refresh" | "download" | "delete"; modelId: string } | null>(null);
  const [memoryCleanupBusy, setMemoryCleanupBusy] = useState(false);
  const [benchmarkBusy, setBenchmarkBusy] = useState(false);
  const [werBenchmarkBusy, setWerBenchmarkBusy] = useState(false);
  const [voiceTranscribing, setVoiceTranscribing] = useState(false);
  const audioContextRef = useRef<AudioContext | null>(null);
  const nextPlaybackTimeRef = useRef(0);
  const quickAbortRef = useRef<AbortController | null>(null);
  const jobAudioUrlsRef = useRef<Record<string, string>>({});
  const voiceAudioUrlsRef = useRef<Record<string, string>>({});
  const quickAudioUrlRef = useRef("");
  const voiceUploadPreviewUrlRef = useRef("");

  useEffect(() => {
    jobAudioUrlsRef.current = jobAudioUrls;
  }, [jobAudioUrls]);

  useEffect(() => {
    voiceAudioUrlsRef.current = voiceAudioUrls;
  }, [voiceAudioUrls]);

  useEffect(() => {
    quickAudioUrlRef.current = quickAudioUrl;
  }, [quickAudioUrl]);

  useEffect(() => {
    voiceUploadPreviewUrlRef.current = voiceUploadPreviewUrl;
  }, [voiceUploadPreviewUrl]);

  useEffect(() => {
    return () => {
      Object.values(jobAudioUrlsRef.current).forEach((url) => URL.revokeObjectURL(url));
      Object.values(voiceAudioUrlsRef.current).forEach((url) => URL.revokeObjectURL(url));
      if (quickAudioUrlRef.current) URL.revokeObjectURL(quickAudioUrlRef.current);
      if (voiceUploadPreviewUrlRef.current) URL.revokeObjectURL(voiceUploadPreviewUrlRef.current);
      if (audioContextRef.current) void audioContextRef.current.close().catch(() => undefined);
    };
  }, []);

  async function loadSnapshot(key: string) {
    const data = await apiFetch<DashboardSnapshot>("/api/admin/snapshot", { adminKey: key });
    setSnapshot(data);
    setSettingsDraft(applyOmniVoiceDefaults(data.settings));
    setQuickModel((current) => current || data.settings.default_model);
    setModelOpsModel((current) => current || data.settings.default_model);
    setOverviewHistory((current) => [...current.slice(-119), { ...data.overview, recorded_at: new Date().toISOString() }]);
    void loadBenchmarks(key);
    void loadWerBenchmarks(key);
    void loadModelDownloads(key, data.settings.model_directory);
    void loadVllmModels(key, data.settings.vllm_base_url);
  }

  async function loadBenchmarks(key = adminKey) {
    if (!key) return;
    try {
      setBenchmarks(await apiFetch<BenchmarkRunResponse[]>("/api/admin/benchmarks/runs", { adminKey: key }));
    } catch {
      // Benchmarks are auxiliary; keep the dashboard usable if this endpoint is mid-run or unavailable.
    }
  }

  async function loadWerBenchmarks(key = adminKey) {
    if (!key) return;
    try {
      const runs = await apiFetch<WerBenchmarkRunResponse[]>("/api/admin/wer-benchmarks/runs", { adminKey: key });
      setWerBenchmarks(runs.slice(0, 1));
    } catch {
      // WER runs are auxiliary; keep the dashboard usable if this endpoint is mid-run or unavailable.
    }
  }

  async function loadModelDownloads(key = adminKey, modelDirectory = settingsDraft?.model_directory || "") {
    if (!key) return;
    try {
      const query = modelDirectory ? `?storage_path=${encodeURIComponent(modelDirectory)}` : "";
      const payload = await apiFetch<ModelDownloadListResponse>(`/api/admin/models${query}`, { adminKey: key });
      setModelDownloads(payload.models || []);
    } catch {
      // Model status is an ops aid; the rest of the admin panel should stay usable.
    }
  }

  async function loadVllmModels(key = adminKey, baseUrl = settingsDraft?.vllm_base_url || "") {
    if (!key || !baseUrl.trim()) return;
    setVllmModelsLoading(true);
    try {
      const query = `?base_url=${encodeURIComponent(baseUrl)}`;
      const payload = await apiFetch<VllmModelsResponse>(`/api/admin/vllm/models${query}`, { adminKey: key });
      setVllmModels(payload.models || []);
      setVllmModelsError(payload.error || "");
      setSettingsDraft((current) => {
        if (!current || current.vllm_base_url !== baseUrl) return current;
        if (current.vllm_model || !payload.models?.length) return current;
        return { ...current, vllm_model: payload.models[0] };
      });
    } catch (loadError) {
      setVllmModels([]);
      setVllmModelsError(loadError instanceof Error ? loadError.message : "vLLM Modelle konnten nicht geladen werden.");
    } finally {
      setVllmModelsLoading(false);
    }
  }

  useEffect(() => {
    if (!adminKey) return;
    const controller = new AbortController();
    void streamSse("/api/admin/dashboard/stream", {
      adminKey,
      signal: controller.signal,
      onEvent: async (eventName, payload) => {
        if (eventName === "dashboard.snapshot") {
          const next = payload as DashboardSnapshot;
          setSnapshot(next);
          setSettingsDraft((current) => current ?? applyOmniVoiceDefaults(next.settings));
          setOverviewHistory((current) => [...current.slice(-119), { ...next.overview, recorded_at: new Date().toISOString() }]);
        }
      },
    }).catch((streamError) => {
      if (controller.signal.aborted) return;
      if ((streamError as { status?: number }).status === 401) {
        clearStoredAdminKey();
        setAdminKey("");
        setAdminKeyInput("");
        setSnapshot(null);
      } else {
        setError(streamError instanceof Error ? streamError.message : "Dashboard-Stream getrennt.");
      }
    });
    return () => controller.abort();
  }, [adminKey]);

  useEffect(() => {
    if (!adminKey) return;
    void loadBenchmarks(adminKey);
    void loadWerBenchmarks(adminKey);
    const timer = window.setInterval(() => {
      void loadBenchmarks(adminKey);
      void loadWerBenchmarks(adminKey);
    }, 1000);
    return () => window.clearInterval(timer);
  }, [adminKey]);

  useEffect(() => {
    if (!adminKey || !modelDownloads.some((model) => model.status === "downloading")) return;
    const timer = window.setInterval(() => {
      void loadModelDownloads(adminKey, settingsDraft?.model_directory || "");
    }, 2000);
    return () => window.clearInterval(timer);
  }, [adminKey, modelDownloads, settingsDraft?.model_directory]);

  useEffect(() => {
    if (!adminKey || !settingsDraft?.vllm_base_url) return;
    const timer = window.setTimeout(() => {
      void loadVllmModels(adminKey, settingsDraft.vllm_base_url);
    }, 500);
    return () => window.clearTimeout(timer);
  }, [adminKey, settingsDraft?.vllm_base_url]);

  const models = snapshot?.models ?? [];
  const voices = snapshot?.voices ?? [];
  const quickTaskType = useMemo(() => inferTaskType(quickModel || snapshot?.settings.default_model || ""), [quickModel, snapshot]);
  const quickVoices = useMemo(() => voiceOptionsForTask(voices, quickTaskType), [quickTaskType, voices]);
  const defaultTaskType = useMemo(() => inferTaskType(settingsDraft?.default_model || ""), [settingsDraft?.default_model]);
  const defaultVoiceOptions = useMemo(() => voiceOptionsForTask(voices, defaultTaskType), [defaultTaskType, voices]);
  const selectedDefaultVoice = useMemo(() => {
    const match = defaultVoiceOptions.find((voice) => voiceMatchesValue(voice, settingsDraft?.default_voice || ""));
    if (!match) return "";
    return match.source === "custom" ? match.voice_id : match.name;
  }, [defaultVoiceOptions, settingsDraft?.default_voice]);
  const customVoices = useMemo(() => voices.filter((voice) => voice.source === "custom"), [voices]);
  const libraryVoices = customVoices;
  const resolvedModelOpsModel = modelOpsModel || settingsDraft?.default_model || "";
  const modelControlCards = useMemo(
    () =>
      models.map((model) => ({
        model,
        download: modelDownloads.find((download) => download.id === model.model_id),
        selected: model.model_id === resolvedModelOpsModel,
        isDefault: model.model_id === settingsDraft?.default_model,
      })),
    [modelDownloads, models, resolvedModelOpsModel, settingsDraft?.default_model],
  );

  useEffect(() => {
    if (!quickVoices.length) {
      setQuickVoice("");
      if (quickTaskType === "VoiceDesign") {
        setQuickInstructions((current) => voiceDesignInstructOrDefault(current));
      }
      return;
    }
    setQuickVoice((current) =>
      current && quickVoices.some((voice) => voiceMatchesValue(voice, current))
        ? current
        : firstVoiceValueForTask(voices, quickTaskType),
    );
  }, [quickTaskType, quickVoices, voices]);

  useEffect(() => {
    if (quickTaskType === "VoiceDesign") {
      setQuickInstructions((current) => voiceDesignInstructOrDefault(current));
    }
  }, [quickTaskType]);

  useEffect(() => {
    if (!settingsDraft) return;
    if (defaultTaskType === "VoiceDesign") {
      if (settingsDraft.default_voice) {
        setSettingsDraft({ ...settingsDraft, default_voice: "" });
      }
      return;
    }
    if (!defaultVoiceOptions.length) return;
    if (!defaultVoiceOptions.some((voice) => voiceMatchesValue(voice, settingsDraft.default_voice))) {
      setSettingsDraft({ ...settingsDraft, default_voice: firstVoiceValueForTask(voices, defaultTaskType) });
    }
  }, [defaultTaskType, defaultVoiceOptions, settingsDraft, voices]);

  useEffect(() => {
    if (!models.length) return;
    setModelOpsModel((current) => (current && models.some((model) => model.model_id === current) ? current : models[0].model_id));
  }, [models]);

  function patchDefaultModel(modelId: string) {
    const nextTaskType = inferTaskType(modelId);
    setSettingsDraft((current) =>
      current
        ? {
            ...current,
            default_model: modelId,
            default_voice: firstVoiceValueForTask(voices, nextTaskType),
          }
        : current,
    );
  }

  function handleQuickVoiceChange(voiceId: string) {
    const voice = voices.find((item) => item.voice_id === voiceId || item.name === voiceId);
    if (voice?.source === "custom" && quickTaskType !== "Base") {
      const baseModel = preferredBaseModel(models);
      if (baseModel) setQuickModel(baseModel);
    }
    setQuickVoice(voiceId);
  }

  async function handleAuthenticate() {
    if (!adminKeyInput.trim()) return;
    setAuthLoading(true);
    setError("");
    try {
      await loadSnapshot(adminKeyInput.trim());
      writeStoredAdminKey(adminKeyInput.trim());
      setAdminKey(adminKeyInput.trim());
      setMessage("Adminpanel verbunden.");
    } catch (authError) {
      setError(authError instanceof Error ? authError.message : "Authentifizierung fehlgeschlagen.");
    } finally {
      setAuthLoading(false);
    }
  }

  async function ensureAudioContext(nextSampleRate: number) {
    const AudioContextImpl = window.AudioContext || (window as typeof window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!AudioContextImpl) throw new Error("Web Audio ist nicht verfuegbar.");
    if (audioContextRef.current && audioContextRef.current.sampleRate !== nextSampleRate) {
      await audioContextRef.current.close().catch(() => undefined);
      audioContextRef.current = null;
      nextPlaybackTimeRef.current = 0;
    }
    if (!audioContextRef.current) {
      audioContextRef.current = new AudioContextImpl({ sampleRate: nextSampleRate });
      nextPlaybackTimeRef.current = 0;
    }
    if (audioContextRef.current.state === "suspended") await audioContextRef.current.resume();
    return audioContextRef.current;
  }

  async function queuePlayback(float32: Float32Array, nextSampleRate: number) {
    const context = await ensureAudioContext(nextSampleRate);
    const buffer = context.createBuffer(1, float32.length, nextSampleRate);
    buffer.getChannelData(0).set(float32);
    const source = context.createBufferSource();
    source.buffer = buffer;
    source.connect(context.destination);
    const startAt = Math.max(context.currentTime + 0.06, nextPlaybackTimeRef.current);
    source.start(startAt);
    nextPlaybackTimeRef.current = startAt + buffer.duration;
  }

  async function handleQuickRun() {
    if (quickRunning) {
      quickAbortRef.current?.abort();
      return;
    }
    if (!quickModel || (quickTaskType !== "VoiceDesign" && !quickVoice)) return;
    if (quickAudioUrl) URL.revokeObjectURL(quickAudioUrl);
    setQuickAudioUrl("");
    setQuickMetrics(null);
    setQuickRunning(true);
    setError("");
    nextPlaybackTimeRef.current = 0;
    const controller = new AbortController();
    let sampleRate = snapshot?.settings.sample_rate ?? 24000;
    const chunks: Int16Array[] = [];
    quickAbortRef.current = controller;
    const parsedSeed = quickSeed.trim() ? Number(quickSeed) : null;

    try {
      await streamNdjson("/api/v1/synthesize/stream", {
        signal: controller.signal,
        body: {
          input: quickText,
          model: quickModel,
          voice: quickTaskType === "VoiceDesign" ? null : quickVoice,
          task_type: quickTaskType,
          instructions: quickTaskType === "VoiceDesign" ? voiceDesignInstructOrDefault(quickInstructions) : quickInstructions,
          language: "Auto",
          stream: true,
          response_format: "pcm",
          seed: parsedSeed !== null && Number.isFinite(parsedSeed) ? parsedSeed : null,
        },
        onEvent: async (event: SynthStreamEvent) => {
          if (event.type === "chunk") {
            sampleRate = event.sample_rate;
            const decoded = decodePcm16Base64(event.pcm16_b64);
            chunks.push(decoded.int16);
            await queuePlayback(decoded.float32, event.sample_rate);
          }
          if (event.type === "done") setQuickMetrics(event.result.metrics);
          if (event.type === "error") throw new Error(event.message);
        },
      });
      if (chunks.length) setQuickAudioUrl(URL.createObjectURL(createWavBlobFromInt16Chunks(chunks, sampleRate)));
      setMessage("Quick-Synthesis fertig.");
    } catch (quickError) {
      if (!controller.signal.aborted) setError(quickError instanceof Error ? quickError.message : "Quick-Synthesis fehlgeschlagen.");
    } finally {
      if (quickAbortRef.current === controller) quickAbortRef.current = null;
      setQuickRunning(false);
    }
  }

  async function handleRotateKey() {
    if (!adminKey) return;
    try {
      const payload = await apiFetch<{ token: string }>("/api/admin/keys", { method: "POST", adminKey });
      writeStoredAdminKey(payload.token);
      setAdminKey(payload.token);
      setAdminKeyInput(payload.token);
      await copyToClipboard(payload.token);
      setMessage("Admin-Key rotiert und kopiert.");
    } catch (rotateError) {
      setError(rotateError instanceof Error ? rotateError.message : "Key-Rotation fehlgeschlagen.");
    }
  }

  async function handleSaveSettings() {
    if (!adminKey || !settingsDraft) return;
    try {
      const updated = await apiFetch<ServerSettings>("/api/admin/settings", { method: "PUT", adminKey, body: settingsDraft });
      setSettingsDraft(applyOmniVoiceDefaults(updated));
      await loadSnapshot(adminKey);
      await loadModelDownloads(adminKey, updated.model_directory);
      setMessage("Settings gespeichert.");
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : "Settings konnten nicht gespeichert werden.");
    }
  }

  async function handleSetOpsModelAsDefault(modelId = resolvedModelOpsModel) {
    if (!adminKey || !settingsDraft || !modelId) return;
    const nextTaskType = inferTaskType(modelId);
    const nextSettings = {
      ...settingsDraft,
      default_model: modelId,
      default_voice: firstVoiceValueForTask(voices, nextTaskType),
    };
    try {
      const updated = await apiFetch<ServerSettings>("/api/admin/settings", { method: "PUT", adminKey, body: nextSettings });
      setSettingsDraft(applyOmniVoiceDefaults(updated));
      await loadSnapshot(adminKey);
      setMessage("Default-Modell gespeichert.");
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : "Default-Modell konnte nicht gespeichert werden.");
    }
  }

  async function handleModelOperation(kind: "preload" | "warmup" | "unload" | "reload", modelId = resolvedModelOpsModel) {
    if (!adminKey || !settingsDraft) return;
    const targetModel = modelId || settingsDraft.default_model;
    const targetTaskType = inferTaskType(targetModel);
    const warmupVoice =
      targetTaskType === "VoiceDesign"
        ? null
        : targetTaskType === "Base"
          ? customVoices[0]?.voice_id || settingsDraft.default_voice || null
          : settingsDraft.default_voice || null;
    setModelOpsBusy(true);
    setError("");
    try {
      const payload = await apiFetch<ModelOperationResponse>(`/api/admin/models/${kind}`, {
        method: "POST",
        adminKey,
        body: {
          model: targetModel,
          task_type: targetTaskType,
          voice: warmupVoice,
          instructions: kind === "warmup" && targetTaskType === "VoiceDesign" ? DEFAULT_VOICE_DESIGN_INSTRUCT : "",
          language: "Auto",
        },
      });
      await loadSnapshot(adminKey);
      setMessage(`${payload.model}: ${payload.message} (${formatMs(payload.warm_ms)})`);
    } catch (operationError) {
      setError(operationError instanceof Error ? operationError.message : "Model operation failed.");
    } finally {
      setModelOpsBusy(false);
    }
  }

  async function handleModelDownloadRefresh() {
    if (!adminKey) return;
    setModelDownloadAction({ kind: "refresh", modelId: "__refresh__" });
    setError("");
    try {
      await loadModelDownloads(adminKey, settingsDraft?.model_directory || "");
      setMessage("Model cache aktualisiert.");
    } finally {
      setModelDownloadAction(null);
    }
  }

  async function handleModelDownload(modelId: string) {
    if (!adminKey || !settingsDraft) return;
    setModelDownloadAction({ kind: "download", modelId });
    setError("");
    try {
      const payload = await apiFetch<ModelDownloadListResponse>("/api/admin/models/download", {
        method: "POST",
        adminKey,
        body: {
          model_id: modelId,
          storage_path: settingsDraft.model_directory,
        },
      });
      setModelDownloads(payload.models || []);
      setMessage(`Download fuer ${modelId} gestartet.`);
    } catch (downloadError) {
      setError(downloadError instanceof Error ? downloadError.message : "Model-Download fehlgeschlagen.");
    } finally {
      setModelDownloadAction(null);
    }
  }

  async function handleModelDelete(modelId: string) {
    if (!adminKey || !settingsDraft) return;
    setModelDownloadAction({ kind: "delete", modelId });
    setError("");
    try {
      const payload = await apiFetch<ModelDownloadListResponse>("/api/admin/models/delete", {
        method: "POST",
        adminKey,
        body: {
          model_id: modelId,
          storage_path: settingsDraft.model_directory,
        },
      });
      setModelDownloads(payload.models || []);
      setMessage(payload.removed ? `Model-Cache fuer ${modelId} geloescht.` : `Kein lokaler Cache fuer ${modelId} gefunden.`);
    } catch (deleteError) {
      setError(deleteError instanceof Error ? deleteError.message : "Model-Cache konnte nicht geloescht werden.");
    } finally {
      setModelDownloadAction(null);
    }
  }

  async function handleFreeMemory() {
    if (!adminKey) return;
    setMemoryCleanupBusy(true);
    setError("");
    try {
      const payload = await apiFetch<MemoryCleanupResponse>("/api/admin/runtime/free-memory", {
        method: "POST",
        adminKey,
      });
      await loadSnapshot(adminKey);
      const released = payload.released_mb === null || payload.released_mb === undefined ? "unbekannt" : `${payload.released_mb} MB`;
      const before = payload.memory_before_mb === null || payload.memory_before_mb === undefined ? "?" : `${payload.memory_before_mb} MB`;
      const after = payload.memory_after_mb === null || payload.memory_after_mb === undefined ? "?" : `${payload.memory_after_mb} MB`;
      setMessage(`${payload.message} VRAM: ${before} -> ${after}, frei: ${released}.`);
    } catch (cleanupError) {
      setError(cleanupError instanceof Error ? cleanupError.message : "VRAM Cleanup fehlgeschlagen.");
    } finally {
      setMemoryCleanupBusy(false);
    }
  }

  async function handleBenchmarkRun() {
    if (!adminKey || !settingsDraft) return;
    setBenchmarkBusy(true);
    setError("");
    try {
      const taskType = inferTaskType(settingsDraft.default_model);
      const parsedBenchmarkSeed = benchmarkSeed.trim() ? Number(benchmarkSeed) : null;
      await apiFetch<BenchmarkRunResponse>("/api/admin/benchmarks/runs", {
        method: "POST",
        adminKey,
        body: {
          name: benchmarkMode === "traffic" ? "G3_OmniVoice traffic benchmark" : "G3_OmniVoice iteration benchmark",
          text: benchmarkText,
          mode: benchmarkMode,
          iterations: benchmarkIterations,
          warmup_iterations: benchmarkWarmups,
          parallel_requests: benchmarkParallel,
          duration_seconds: benchmarkDuration,
          requests_per_minute: benchmarkRequestsPerMinute,
          min_sentences_per_request: Math.min(benchmarkMinSentences, benchmarkMaxSentences),
          max_sentences_per_request: Math.max(benchmarkMinSentences, benchmarkMaxSentences),
          random_seed: parsedBenchmarkSeed !== null && Number.isFinite(parsedBenchmarkSeed) ? parsedBenchmarkSeed : null,
          exclusive: true,
          cases: [
            {
              label: settingsDraft.default_model,
              request: {
                model: settingsDraft.default_model,
                voice: taskType === "VoiceDesign" ? null : settingsDraft.default_voice,
                task_type: taskType,
                language: "Auto",
                instructions: taskType === "VoiceDesign" ? voiceDesignInstructOrDefault(quickInstructions) : quickInstructions,
                response_format: "wav",
              },
            },
          ],
        },
      });
      await loadBenchmarks(adminKey);
      setMessage("Benchmark gestartet.");
    } catch (benchmarkError) {
      setError(benchmarkError instanceof Error ? benchmarkError.message : "Benchmark konnte nicht gestartet werden.");
    } finally {
      setBenchmarkBusy(false);
    }
  }

  async function handleWerBenchmarkRun() {
    if (!adminKey || !settingsDraft) return;
    setWerBenchmarkBusy(true);
    setError("");
    const taskType = inferTaskType(settingsDraft.default_model);
    const parsedSeed = werSeed.trim() ? Number(werSeed) : null;
    try {
      await apiFetch<WerBenchmarkRunResponse>("/api/admin/wer-benchmarks/runs", {
        method: "POST",
        adminKey,
        body: {
          name: "G3_OmniVoice WER benchmark",
          count: werCount,
          concurrency: settingsDraft.wer_concurrency,
          transcription_concurrency: settingsDraft.wer_transcription_concurrency,
          vllm_base_url: settingsDraft.vllm_base_url,
          vllm_model: settingsDraft.vllm_model.trim() || null,
          whisper_base_url: settingsDraft.whisper_base_url || "",
          whisper_path: null,
          language: werLanguage,
          prompt: werPrompt.trim() || null,
          min_words: Math.min(werMinWords, werMaxWords),
          max_words: Math.max(werMinWords, werMaxWords),
          tolerance_letters_per_word: werTolerance,
          completion_timeout_seconds: werTimeout,
          random_seed: parsedSeed !== null && Number.isFinite(parsedSeed) ? parsedSeed : null,
          seed_range: werSeedRange,
          exclusive: true,
          request: {
            model: settingsDraft.default_model,
            voice: taskType === "VoiceDesign" ? null : settingsDraft.default_voice,
            task_type: taskType,
            language: "Auto",
            instructions: taskType === "VoiceDesign" ? voiceDesignInstructOrDefault(quickInstructions) : quickInstructions,
            response_format: "wav",
          },
        },
      });
      await loadWerBenchmarks(adminKey);
      setMessage("WER-Benchmark gestartet.");
    } catch (werError) {
      setError(werError instanceof Error ? werError.message : "WER-Benchmark konnte nicht gestartet werden.");
    } finally {
      setWerBenchmarkBusy(false);
    }
  }

  function handleVoiceFileChange(file: File | null) {
    setVoiceFile(file);
    setVoiceUploadPreviewUrl((current) => {
      if (current) URL.revokeObjectURL(current);
      return file ? URL.createObjectURL(file) : "";
    });
  }

  async function handleUploadVoice() {
    if (!adminKey || !voiceFile || !voiceName.trim()) return;
    const form = new FormData();
    form.append("audio_sample", voiceFile);
    form.append("name", voiceName.trim());
    form.append("consent", String(voiceConsent));
    form.append("ref_text", voiceRefText);
    try {
      await apiFetch("/api/admin/voices", { method: "POST", adminKey, body: form });
      setVoiceName("");
      setVoiceRefText("");
      setVoiceConsent(false);
      handleVoiceFileChange(null);
      await loadSnapshot(adminKey);
      setMessage("Voice gespeichert.");
    } catch (uploadError) {
      setError(uploadError instanceof Error ? uploadError.message : "Voice-Upload fehlgeschlagen.");
    }
  }

  async function handleTranscribeVoice() {
    if (!adminKey || !voiceFile) return;
    setVoiceTranscribing(true);
    setError("");
    const form = new FormData();
    form.append("file", voiceFile);
    try {
      const payload = await apiFetch<{ transcription?: string; text?: string }>("/api/admin/voices/transcribe", {
        method: "POST",
        adminKey,
        body: form,
      });
      const transcript = (payload.transcription || payload.text || "").trim();
      if (!transcript) throw new Error("Whisper hat keinen Text zurueckgegeben.");
      setVoiceRefText(transcript);
      setMessage("Referenztext aus Whisper uebernommen.");
    } catch (transcribeError) {
      setError(transcribeError instanceof Error ? transcribeError.message : "Whisper-Transkription fehlgeschlagen.");
    } finally {
      setVoiceTranscribing(false);
    }
  }

  async function handleDeleteVoice(voiceId: string) {
    if (!adminKey) return;
    await apiFetch(`/api/admin/voices/${voiceId}`, { method: "DELETE", adminKey });
    setVoiceAudioUrls((current) => {
      const existing = current[voiceId];
      if (existing) URL.revokeObjectURL(existing);
      const next = { ...current };
      delete next[voiceId];
      return next;
    });
    await loadSnapshot(adminKey);
  }

  async function handleLoadVoiceSample(voiceId: string) {
    if (!adminKey) return;
    const blob = await apiFetch<Blob>(`/api/admin/voices/${encodeURIComponent(voiceId)}/audio`, { adminKey, responseType: "blob" });
    setVoiceAudioUrls((current) => {
      if (current[voiceId]) URL.revokeObjectURL(current[voiceId]);
      return { ...current, [voiceId]: URL.createObjectURL(blob) };
    });
  }

  async function handleLoadJobAudio(jobId: string) {
    if (!adminKey) return;
    const blob = await apiFetch<Blob>(`/api/admin/jobs/${jobId}/audio`, { adminKey, responseType: "blob" });
    setJobAudioUrls((current) => {
      if (current[jobId]) URL.revokeObjectURL(current[jobId]);
      return { ...current, [jobId]: URL.createObjectURL(blob) };
    });
  }

  async function handleDeleteJob(jobId: string) {
    if (!adminKey) return;
    await apiFetch(`/api/admin/jobs/${jobId}`, { method: "DELETE", adminKey });
    await loadSnapshot(adminKey);
  }

  function handleLogout() {
    clearStoredAdminKey();
    setAdminKey("");
    setAdminKeyInput("");
    setSnapshot(null);
  }

  if (!adminKey || !snapshot || !settingsDraft) {
    return (
      <main className="gate-shell">
        <section className="gate-card">
          <p className="eyebrow">Private Access</p>
          <h1>G3_OmniVoice Adminpanel</h1>
          <p className="widget-copy">Nur das Adminpanel ist geschuetzt. Die Demo bleibt offen.</p>
          <label className="with-help" title={ADMIN_HELP.adminKey}>
            Admin-Key
            <input value={adminKeyInput} onChange={(event) => setAdminKeyInput(event.target.value)} placeholder="omnivoice_tts_..." />
          </label>
          <div className="button-row">
            <button className="primary-button" type="button" onClick={handleAuthenticate} disabled={authLoading}>
              {authLoading ? "Pruefe..." : "Adminpanel oeffnen"}
            </button>
            <a className="ghost-button" href="/demo">Zur Demo</a>
          </div>
          {error ? <div className="message error">{error}</div> : null}
        </section>
      </main>
    );
  }

  return (
    <main className="app-shell admin-shell">
      <section className="hero-card">
        <div className="hero-copy">
          <p className="eyebrow">Admin Panel</p>
          <h1>G3_OmniVoice Runtime Control</h1>
          <p>Ein Panel fuer Queue, Stimmen, Einstellungen, Models und History. Mehr braucht die App nicht.</p>
        </div>
        <div className="status-grid">
          <div className="status-pill"><span>Model</span><strong>{snapshot.overview.active_model || "-"}</strong></div>
          <div className="status-pill"><span>Queue</span><strong>{snapshot.overview.queue_depth}</strong></div>
          <div className="status-pill"><span>Active requests</span><strong>{snapshot.overview.active_requests}</strong></div>
          <div className="status-pill"><span>Realtime avg</span><strong>{formatRealtime(snapshot.overview.realtime_x_avg)}</strong></div>
        </div>
        <div className="button-row">
          <a className="ghost-button" href="/">Landing</a>
          <a className="secondary-button" href="/demo">Demo</a>
          <button className="secondary-button" type="button" onClick={handleRotateKey}>Rotate API Key</button>
          <button className="ghost-button" type="button" onClick={handleLogout}>Logout</button>
        </div>
      </section>

      {message ? <div className="message success">{message}</div> : null}
      {error ? <div className="message error">{error}</div> : null}

      <section className="widget-grid">
        <section className="widget span-12"><div className="widget-header"><h2>Performance Graph</h2></div>
          <div className="field-grid four">
            <MiniGraph label="Realtime" values={overviewHistory.map((entry) => Number(entry.realtime_x_avg || 0))} color="#90f0b7" suffix="x" />
            <MiniGraph label="TTFA" values={overviewHistory.map((entry) => Number(entry.ttfa_ms_avg || 0))} color="#ff9a46" suffix=" ms" />
            <MiniGraph label="Queue" values={overviewHistory.map((entry) => Number(entry.queue_depth || 0))} color="#86c9ff" />
            <MiniGraph label="GPU" values={overviewHistory.map((entry) => Number(entry.gpu_utilization_pct || 0))} color="#ff6e6e" suffix="%" />
          </div>
        </section>

        <section className="widget span-6"><div className="widget-header"><h2>Live Queue</h2></div>
          <div className="metric-list">
            <div className="metric-row"><span>Worker</span><strong>{snapshot.overview.worker_state}</strong></div>
            <div className="metric-row"><span>TTFA avg</span><strong>{formatMs(snapshot.overview.ttfa_ms_avg)}</strong></div>
            <div className="metric-row"><span>Queue wait avg</span><strong>{formatMs(snapshot.overview.queue_wait_ms_avg)}</strong></div>
            <div className="metric-row"><span>Audio total</span><strong>{formatSeconds(snapshot.overview.audio_seconds_total)}</strong></div>
            <div className="metric-row"><span>GPU</span><strong>{snapshot.overview.gpu_utilization_pct}%</strong></div>
          </div>
          {snapshot.current_batch ? <div className="voice-card"><strong>{snapshot.current_batch.batch_id}</strong><div className="inline-pills"><span className="pill">{snapshot.current_batch.model_id}</span><span className="pill">{snapshot.current_batch.task_type}</span><span className="pill">size {snapshot.current_batch.size}</span></div></div> : null}
          {snapshot.recent_batches.length ? (
            <div className="job-list">
              {snapshot.recent_batches.slice(-4).reverse().map((batch) => (
                <article key={batch.batch_id} className="job-card">
                  <strong>{batch.batch_id}</strong>
                  <div className="inline-pills"><span className="pill">size {batch.size}</span><span className="pill">{batch.model_id}</span><span className="pill">{batch.task_type}</span></div>
                </article>
              ))}
            </div>
          ) : null}
        </section>

        <section className="widget span-6"><div className="widget-header"><h2>Admin Key</h2></div>
          <div className="metric-list">
            <div className="metric-row"><span>Created</span><strong>{formatDate(snapshot.admin_key.created_at)}</strong></div>
            <div className="metric-row"><span>Last used</span><strong>{formatDate(snapshot.admin_key.last_used_at)}</strong></div>
            <div className="metric-row"><span>Current</span><strong>{adminKey.slice(0, 14)}...</strong></div>
          </div>
          <div className="button-row"><button className="secondary-button" type="button" onClick={() => void copyToClipboard(adminKey)}>Copy</button></div>
        </section>

        <section className="widget span-12">
          <div className="widget-header">
            <h2>Model Control</h2>
            <div className="button-row compact">
              <button className="secondary-button" type="button" onClick={() => void handleModelDownloadRefresh()} disabled={modelDownloadAction?.kind === "refresh"} title={ADMIN_HELP.refreshDownloads}>{modelDownloadAction?.kind === "refresh" ? "Refreshing..." : "Refresh Cache"}</button>
              <button className="secondary-button" type="button" onClick={() => void handleFreeMemory()} disabled={memoryCleanupBusy || modelOpsBusy} title={ADMIN_HELP.freeMemory}>Free VRAM</button>
            </div>
          </div>
          <div className="metric-list compact model-control-summary">
            <div className="metric-row"><span>Storage</span><strong className="mono path-cell">{settingsDraft.model_directory || "-"}</strong></div>
            <div className="metric-row"><span>Runtime</span><strong>{settingsDraft.runtime_backend} / {settingsDraft.preferred_device} / {settingsDraft.torch_dtype}</strong></div>
            <div className="metric-row"><span>VRAM</span><strong>{snapshot.overview.gpu_memory_used_mb} / {snapshot.overview.gpu_memory_total_mb} MB</strong></div>
            <div className="metric-row"><span>Compile</span><strong>{settingsDraft.compile_model ? "llm on" : "llm off"}</strong></div>
          </div>
          <label className="checkbox-row with-help" title={ADMIN_HELP.allowDownloads}><input type="checkbox" checked={settingsDraft.allow_model_downloads} onChange={(event) => setSettingsDraft({ ...settingsDraft, allow_model_downloads: event.target.checked })} />HF-Downloads erlauben</label>
          {snapshot.jobs[0]?.status === "failed" && snapshot.jobs[0]?.error_message ? (
            <div className="message error">{snapshot.jobs[0].error_message}</div>
          ) : null}
          <div className="model-bubble-grid">
            {modelControlCards.map(({ model, download, selected, isDefault }) => {
              const downloadStatus = download?.status || "missing";
              const actionBusy = modelOpsBusy || modelDownloadAction?.modelId === model.model_id;
              return (
                <article key={model.model_id} className={`model-bubble ${model.active ? "active" : ""} ${selected ? "selected" : ""}`}>
                  <div className="model-bubble-head">
                    <div>
                      <strong>{download?.label || model.model_id.split("/").pop() || model.model_id}</strong>
                      <span className="mono">{model.model_id}</span>
                    </div>
                    <div className="inline-pills">
                      <span className={`pill ${model.active ? "active" : ""}`}>{model.active ? "active" : "idle"}</span>
                      <span className={`pill ${model.loaded ? "active" : ""}`}>{model.loaded ? "loaded" : "cold"}</span>
                      <span className={`pill ${downloadStatus === "ready" ? "active" : ""}`}>{formatModelStatus(downloadStatus)}</span>
                    </div>
                  </div>
                  <div className="metric-list compact">
                    <div className="metric-row"><span>Default</span><strong>{isDefault ? "yes" : "no"}</strong></div>
                    <div className="metric-row"><span>Type</span><strong>{download?.kind || model.task_types.join(", ")}</strong></div>
                    <div className="metric-row"><span>Size</span><strong>{download ? formatModelSize(download) : "-"}</strong></div>
                    <div className="metric-row"><span>Path</span><strong className="mono path-cell">{download ? resolveModelPath(download) : settingsDraft.model_directory}</strong></div>
                  </div>
                  {download?.error ? <p className="model-error">{download.error}</p> : null}
                  <div className="button-row compact model-bubble-actions">
                    <button className="secondary-button" type="button" onClick={() => setModelOpsModel(model.model_id)} title={ADMIN_HELP.modelOpsModel}>Select</button>
                    <button className="secondary-button" type="button" onClick={() => void handleModelDownload(model.model_id)} disabled={downloadStatus === "downloading" || actionBusy} title={ADMIN_HELP.downloadModel}>
                      {downloadStatus === "downloading" ? "Downloading..." : modelDownloadAction?.kind === "download" && modelDownloadAction.modelId === model.model_id ? "Starting..." : downloadStatus === "ready" ? "Download Again" : "Download"}
                    </button>
                    <button className="secondary-button" type="button" onClick={() => void handleModelOperation("preload", model.model_id)} disabled={actionBusy} title={ADMIN_HELP.preload}>Preload</button>
                    <button className="primary-button" type="button" onClick={() => void handleModelOperation("warmup", model.model_id)} disabled={actionBusy} title={ADMIN_HELP.warmup}>Warmup</button>
                    <button className="secondary-button" type="button" onClick={() => void handleModelOperation("reload", model.model_id)} disabled={actionBusy} title={ADMIN_HELP.reload}>Reload</button>
                    <button className="ghost-button" type="button" onClick={() => void handleModelOperation("unload", model.model_id)} disabled={actionBusy} title={ADMIN_HELP.unload}>Unload</button>
                    <button className="secondary-button" type="button" onClick={() => void handleSetOpsModelAsDefault(model.model_id)} disabled={actionBusy || isDefault} title={ADMIN_HELP.saveDefault}>Default</button>
                    <button className="ghost-button" type="button" onClick={() => void handleModelDelete(model.model_id)} disabled={!download?.cache_path || downloadStatus === "downloading" || actionBusy} title={ADMIN_HELP.deleteModel}>Delete Cache</button>
                  </div>
                </article>
              );
            })}
          </div>
        </section>

        <section className="widget span-12"><div className="widget-header"><h2>Runtime Settings</h2></div>
          <div className="field-grid four">
            <label className="with-help" title={ADMIN_HELP.defaultModel}>Default model<select value={settingsDraft.default_model} onChange={(event) => patchDefaultModel(event.target.value)}>{models.map((model) => <option key={model.model_id} value={model.model_id}>{model.model_id}</option>)}</select></label>
            <label className="with-help" title={ADMIN_HELP.defaultVoice}>Default voice<select value={selectedDefaultVoice} onChange={(event) => setSettingsDraft({ ...settingsDraft, default_voice: event.target.value })} disabled={defaultTaskType === "VoiceDesign" || !defaultVoiceOptions.length}>
              {defaultTaskType === "VoiceDesign" ? <option value="">Prompt-defined voice</option> : null}
              {defaultTaskType !== "VoiceDesign" && !defaultVoiceOptions.length ? <option value="">Keine passende Stimme</option> : null}
              {defaultVoiceOptions.map((voice) => <option key={voice.voice_id} value={voice.source === "custom" ? voice.voice_id : voice.name}>{voice.name}</option>)}
            </select></label>
            <label className="with-help" title={ADMIN_HELP.modelDirectory}>Model directory<input value={settingsDraft.model_directory} onChange={(event) => setSettingsDraft({ ...settingsDraft, model_directory: event.target.value })} /></label>
            <label className="with-help" title={ADMIN_HELP.whisperUrl}>Whisper URL<input value={settingsDraft.whisper_base_url || ""} onChange={(event) => setSettingsDraft({ ...settingsDraft, whisper_base_url: event.target.value, whisper_path: "" })} placeholder="http://192.168.0.200:7861" /></label>
            <label className="with-help" title={ADMIN_HELP.vllmBaseUrl}>vLLM Base URL<input value={settingsDraft.vllm_base_url} onChange={(event) => setSettingsDraft({ ...settingsDraft, vllm_base_url: event.target.value })} placeholder="http://192.168.20.126:8000" /></label>
            <label className="with-help" title={ADMIN_HELP.vllmModel}>vLLM Model
              {vllmModels.length > 1 ? (
                <select value={settingsDraft.vllm_model || vllmModels[0]} onChange={(event) => setSettingsDraft({ ...settingsDraft, vllm_model: event.target.value })}>
                  {vllmModels.map((model) => <option key={model} value={model}>{model}</option>)}
                </select>
              ) : (
                <input value={settingsDraft.vllm_model || vllmModels[0] || ""} onChange={(event) => setSettingsDraft({ ...settingsDraft, vllm_model: event.target.value })} placeholder={vllmModelsLoading ? "lade Modelle..." : "auto aus /v1/models"} />
              )}
            </label>
            <label className="with-help" title={ADMIN_HELP.werConcurrency}>WER TTS wave<input type="number" min="1" max="64" value={settingsDraft.wer_concurrency} onChange={(event) => setSettingsDraft({ ...settingsDraft, wer_concurrency: Number(event.target.value) || 1 })} /></label>
            <label className="with-help" title={ADMIN_HELP.werTranscriptionConcurrency}>WER ASR concurrency<input type="number" min="1" max="64" value={settingsDraft.wer_transcription_concurrency} onChange={(event) => setSettingsDraft({ ...settingsDraft, wer_transcription_concurrency: Number(event.target.value) || 1 })} /></label>
            <label className="with-help" title={ADMIN_HELP.queueLimit}>Queue limit<input type="number" value={settingsDraft.queue_limit} onChange={(event) => setSettingsDraft({ ...settingsDraft, queue_limit: Number(event.target.value) || 1 })} /></label>
            <label className="with-help" title={ADMIN_HELP.activeRequests}>Active requests<input type="number" value={settingsDraft.max_parallel_requests} onChange={(event) => setSettingsDraft({ ...settingsDraft, max_parallel_requests: Number(event.target.value) || 1 })} /></label>
            <label className="with-help" title={ADMIN_HELP.batchSize}>Batch size<input type="number" value={settingsDraft.max_batch_size} onChange={(event) => setSettingsDraft({ ...settingsDraft, max_batch_size: Number(event.target.value) || 1 })} /></label>
            <label className="with-help" title={ADMIN_HELP.batchWait}>Batch wait ms<input type="number" value={settingsDraft.batch_wait_ms} onChange={(event) => setSettingsDraft({ ...settingsDraft, batch_wait_ms: Number(event.target.value) || 0 })} /></label>
            <label className="with-help" title={ADMIN_HELP.streamPrebuffer}>Stream prebuffer ms<input type="number" value={settingsDraft.stream_prebuffer_ms} onChange={(event) => setSettingsDraft({ ...settingsDraft, stream_prebuffer_ms: Number(event.target.value) || 0 })} /></label>
            <label className="with-help" title={ADMIN_HELP.dtype}>Dtype<select value={settingsDraft.torch_dtype} onChange={(event) => setSettingsDraft({ ...settingsDraft, torch_dtype: event.target.value })}>
              <option value="float16">fp16</option>
              <option value="bfloat16">bf16</option>
              <option value="float32">fp32</option>
            </select></label>
            <label className="with-help" title={ADMIN_HELP.numStep}>Num step<input type="number" value={settingsDraft.num_step ?? ""} onChange={(event) => setSettingsDraft({ ...settingsDraft, num_step: event.target.value ? Number(event.target.value) : null })} /></label>
            <label className="with-help" title={ADMIN_HELP.guidance}>Guidance<input type="number" step="0.1" value={settingsDraft.guidance_scale ?? ""} onChange={(event) => setSettingsDraft({ ...settingsDraft, guidance_scale: event.target.value ? Number(event.target.value) : null })} /></label>
            <label className="with-help" title={ADMIN_HELP.duration}>Duration<input type="number" step="0.1" value={settingsDraft.duration ?? ""} onChange={(event) => setSettingsDraft({ ...settingsDraft, duration: event.target.value ? Number(event.target.value) : null })} placeholder="auto" /></label>
            <label className="with-help" title={ADMIN_HELP.tShift}>T shift<input type="number" step="0.1" value={settingsDraft.t_shift ?? ""} onChange={(event) => setSettingsDraft({ ...settingsDraft, t_shift: event.target.value ? Number(event.target.value) : null })} /></label>
            <label className="with-help" title={ADMIN_HELP.positionTemperature}>Position temp<input type="number" step="0.1" value={settingsDraft.position_temperature ?? ""} onChange={(event) => setSettingsDraft({ ...settingsDraft, position_temperature: event.target.value ? Number(event.target.value) : null })} /></label>
            <label className="with-help" title={ADMIN_HELP.classTemperature}>Class temp<input type="number" step="0.1" value={settingsDraft.class_temperature ?? ""} onChange={(event) => setSettingsDraft({ ...settingsDraft, class_temperature: event.target.value ? Number(event.target.value) : null })} /></label>
            <label className="with-help" title={ADMIN_HELP.audioChunkDuration}>Audio chunk sec<input type="number" step="0.1" value={settingsDraft.audio_chunk_duration ?? ""} onChange={(event) => setSettingsDraft({ ...settingsDraft, audio_chunk_duration: event.target.value ? Number(event.target.value) : null })} /></label>
            <label className="with-help" title={ADMIN_HELP.audioChunkThreshold}>Chunk threshold sec<input type="number" step="0.1" value={settingsDraft.audio_chunk_threshold ?? ""} onChange={(event) => setSettingsDraft({ ...settingsDraft, audio_chunk_threshold: event.target.value ? Number(event.target.value) : null })} /></label>
          </div>
          <div className="inline-pills">
            <button className="link-chip" type="button" title={ADMIN_HELP.refreshVllmModels} onClick={() => void loadVllmModels(adminKey, settingsDraft.vllm_base_url)} disabled={vllmModelsLoading}>{vllmModelsLoading ? "vLLM laedt..." : "vLLM Modelle aktualisieren"}</button>
            <span className={`pill ${vllmModels.length ? "active" : ""}`}>{vllmModels.length ? `${vllmModels.length} vLLM Modelle` : "keine vLLM Modelle"}</span>
            {vllmModelsError ? <span className="pill">{vllmModelsError}</span> : null}
          </div>
          <div className="field-grid two runtime-toggle-grid">
            <label className="checkbox-row with-help" title={ADMIN_HELP.sentenceChunking}><input type="checkbox" checked={settingsDraft.sentence_chunking} onChange={(event) => setSettingsDraft({ ...settingsDraft, sentence_chunking: event.target.checked })} />Sentence chunking aktiv</label>
            <label className="checkbox-row with-help" title={ADMIN_HELP.allowDownloads}><input type="checkbox" checked={settingsDraft.allow_model_downloads} onChange={(event) => setSettingsDraft({ ...settingsDraft, allow_model_downloads: event.target.checked })} />Model-Downloads erlauben</label>
            <label className="checkbox-row with-help" title={ADMIN_HELP.compileModel}><input type="checkbox" checked={settingsDraft.compile_model} onChange={(event) => setSettingsDraft({ ...settingsDraft, compile_model: event.target.checked })} />torch.compile fuer llm</label>
            <label className="checkbox-row with-help" title={ADMIN_HELP.cudagraphSkip}><input type="checkbox" checked={settingsDraft.cudagraph_skip_dynamic_graphs} onChange={(event) => setSettingsDraft({ ...settingsDraft, cudagraph_skip_dynamic_graphs: event.target.checked })} />CUDAGraph dynamic shapes skippen</label>
            <label className="checkbox-row with-help" title={ADMIN_HELP.autoVramTrim}><input type="checkbox" checked={settingsDraft.cuda_memory_trim_after_batch} onChange={(event) => setSettingsDraft({ ...settingsDraft, cuda_memory_trim_after_batch: event.target.checked })} />Auto VRAM trim nach Batch</label>
            <label className="checkbox-row with-help" title={ADMIN_HELP.warmupStartup}><input type="checkbox" checked={settingsDraft.warmup_on_startup} onChange={(event) => setSettingsDraft({ ...settingsDraft, warmup_on_startup: event.target.checked })} />Warmup beim Start</label>
            <label className="checkbox-row with-help" title={ADMIN_HELP.denoise}><input type="checkbox" checked={Boolean(settingsDraft.denoise)} onChange={(event) => setSettingsDraft({ ...settingsDraft, denoise: event.target.checked })} />Denoise</label>
            <label className="checkbox-row with-help" title={ADMIN_HELP.preprocessPrompt}><input type="checkbox" checked={Boolean(settingsDraft.preprocess_prompt)} onChange={(event) => setSettingsDraft({ ...settingsDraft, preprocess_prompt: event.target.checked })} />Preprocess prompt</label>
            <label className="checkbox-row with-help" title={ADMIN_HELP.postprocessOutput}><input type="checkbox" checked={Boolean(settingsDraft.postprocess_output)} onChange={(event) => setSettingsDraft({ ...settingsDraft, postprocess_output: event.target.checked })} />Postprocess output</label>
          </div>
          <div className="button-row"><button className="primary-button" type="button" onClick={handleSaveSettings}>Settings speichern</button></div>
        </section>

        <section className="widget span-6"><div className="widget-header"><h2>Benchmark</h2></div>
          <div className="field-grid two">
            <label className="with-help" title={ADMIN_HELP.benchmarkText}>Satzpool<textarea value={benchmarkText} onChange={(event) => setBenchmarkText(event.target.value)} /></label>
            <div className="field-grid">
              <label className="with-help" title={ADMIN_HELP.benchmarkMode}>Mode<select value={benchmarkMode} onChange={(event) => setBenchmarkMode(event.target.value as "traffic" | "iterations")}>
                <option value="traffic">Traffic simulation</option>
                <option value="iterations">Fixed iterations</option>
              </select></label>
              <div className="field-grid two">
                <label className="with-help" title={ADMIN_HELP.benchmarkDuration}>Duration sec<input type="number" min="1" max="3600" value={benchmarkDuration} onChange={(event) => setBenchmarkDuration(Number(event.target.value) || 1)} /></label>
                <label className="with-help" title={ADMIN_HELP.benchmarkRpm}>Requests/min<input type="number" min="1" max="6000" value={benchmarkRequestsPerMinute} onChange={(event) => setBenchmarkRequestsPerMinute(Number(event.target.value) || 1)} /></label>
                <label className="with-help" title={ADMIN_HELP.benchmarkSentenceRange}>Min Saetze<input type="number" min="1" max="100" value={benchmarkMinSentences} onChange={(event) => setBenchmarkMinSentences(Number(event.target.value) || 1)} /></label>
                <label className="with-help" title={ADMIN_HELP.benchmarkSentenceRange}>Max Saetze<input type="number" min="1" max="100" value={benchmarkMaxSentences} onChange={(event) => setBenchmarkMaxSentences(Number(event.target.value) || 1)} /></label>
                <label className="with-help" title={ADMIN_HELP.benchmarkWarmups}>Warmups<input type="number" min="0" max="20" value={benchmarkWarmups} onChange={(event) => setBenchmarkWarmups(Number(event.target.value) || 0)} /></label>
                <label className="with-help" title={ADMIN_HELP.benchmarkSeed}>Seed<input type="number" min="0" max="2147483647" value={benchmarkSeed} onChange={(event) => setBenchmarkSeed(event.target.value)} placeholder="leer = random" /></label>
              </div>
              {benchmarkMode === "iterations" ? (
                <div className="field-grid two">
                  <label className="with-help" title={ADMIN_HELP.benchmarkIterations}>Iterations<input type="number" min="1" max="50" value={benchmarkIterations} onChange={(event) => setBenchmarkIterations(Number(event.target.value) || 1)} /></label>
                  <label className="with-help" title={ADMIN_HELP.benchmarkParallel}>Parallel<input type="number" min="1" max="64" value={benchmarkParallel} onChange={(event) => setBenchmarkParallel(Number(event.target.value) || 1)} /></label>
                </div>
              ) : null}
              <div className="button-row"><button className="primary-button" type="button" onClick={handleBenchmarkRun} disabled={benchmarkBusy}>Benchmark starten</button></div>
            </div>
          </div>
          <div className="job-list scroll-list benchmark-results">
            {benchmarks.length === 0 ? <p className="widget-copy">Noch keine Benchmark-Runs.</p> : null}
            {benchmarks.map((run) => (
              <article key={run.run_id} className="job-card">
                <strong>{run.name}</strong>
                <div className="inline-pills"><span className={`pill ${run.status === "completed" ? "active" : ""}`}>{run.status}</span><span className="pill">{run.mode}</span><span className="pill">{run.total_requests || run.iterations} requests</span><span className="pill">{run.duration_seconds}s window</span><span className="pill">{run.requests_per_minute}/min</span><span className="pill">{run.completion_timeout_seconds || 180}s timeout</span><span className="pill">{run.warmup_iterations} warmup</span></div>
                {run.cases.map((item) => (
                  <div key={item.label} className="metric-list">
                    <div className="metric-row"><span>{item.label}</span><strong>{item.success_count}/{item.success_count + item.failure_count} ok</strong></div>
                    <div className="metric-row"><span>TTFA avg / p99</span><strong>{formatMs(item.ttfa_ms_avg)} / {formatMs(item.ttfa_ms_p99)}</strong></div>
                    <div className="metric-row"><span>TTFA best / worst</span><strong>{formatMs(item.ttfa_ms_min)} / {formatMs(item.ttfa_ms_max)}</strong></div>
                    <div className="metric-row"><span>Wall p50 / p99</span><strong>{formatMs(item.job_wall_ms_p50)} / {formatMs(item.job_wall_ms_p99)}</strong></div>
                    <div className="metric-row"><span>Queue p95 / p99</span><strong>{formatMs(item.queue_wait_ms_p95)} / {formatMs(item.queue_wait_ms_p99)}</strong></div>
                    <div className="metric-row"><span>Realtime avg</span><strong>{formatRealtime(item.realtime_x_avg)}</strong></div>
                  </div>
                ))}
              </article>
            ))}
          </div>
        </section>

        <section className="widget span-6"><div className="widget-header"><h2>WER Benchmark</h2></div>
          <div className="metric-list">
            <div className="metric-row"><span>vLLM</span><strong>{settingsDraft.vllm_model || "auto"} @ {settingsDraft.vllm_base_url}</strong></div>
            <div className="metric-row"><span>Whisper</span><strong>{settingsDraft.whisper_base_url || "-"}</strong></div>
            <div className="metric-row"><span>TTS / ASR concurrency</span><strong>{settingsDraft.wer_concurrency} / {settingsDraft.wer_transcription_concurrency}</strong></div>
          </div>
          <div className="field-grid two">
            <label className="with-help" title={ADMIN_HELP.werCount}>Samples<input type="number" min="1" max="1000" value={werCount} onChange={(event) => setWerCount(Number(event.target.value) || 1)} /></label>
            <label className="with-help" title={ADMIN_HELP.werLanguage}>Sprache<select value={werLanguage} onChange={(event) => setWerLanguage(event.target.value)}>
              {WER_LANGUAGE_OPTIONS.map((language) => <option key={language} value={language}>{language}</option>)}
            </select></label>
            <label className="with-help" title={ADMIN_HELP.werWords}>Min words<input type="number" min="1" max="80" value={werMinWords} onChange={(event) => setWerMinWords(Number(event.target.value) || 1)} /></label>
            <label className="with-help" title={ADMIN_HELP.werWords}>Max words<input type="number" min="1" max="120" value={werMaxWords} onChange={(event) => setWerMaxWords(Number(event.target.value) || 1)} /></label>
            <label className="with-help" title={ADMIN_HELP.werTolerance}>Tolerance<input type="number" min="0" max="8" value={werTolerance} onChange={(event) => setWerTolerance(Number(event.target.value) || 0)} /></label>
            <label className="with-help" title={ADMIN_HELP.werTimeout}>Timeout sec<input type="number" min="1" max="3600" value={werTimeout} onChange={(event) => setWerTimeout(Number(event.target.value) || 1)} /></label>
            <label className="with-help" title={ADMIN_HELP.werSeed}>Seed<input type="number" min="0" max="2147483647" value={werSeed} onChange={(event) => setWerSeed(event.target.value)} placeholder="leer = random" /></label>
            <label className="with-help" title={ADMIN_HELP.werSeedRange}>Seed range<input type="number" min="0" max="1024" value={werSeedRange} onChange={(event) => setWerSeedRange(Number(event.target.value) || 0)} /></label>
          </div>
          <label className="with-help" title={ADMIN_HELP.werPrompt}>Optionaler vLLM Prompt<textarea value={werPrompt} onChange={(event) => setWerPrompt(event.target.value)} placeholder="Leer lassen fuer den eingebauten JSON-Satzgenerator." /></label>
          <div className="button-row"><button className="primary-button" type="button" onClick={handleWerBenchmarkRun} disabled={werBenchmarkBusy}>WER Benchmark starten</button></div>
          <div className="job-list scroll-list benchmark-results wer-results">
            {werBenchmarks.length === 0 ? <p className="widget-copy">Noch keine WER-Benchmark-Runs.</p> : null}
            {werBenchmarks.map((run) => (
              <article key={run.run_id} className="job-card">
                <strong>{run.name}</strong>
                <div className="inline-pills">
                  <span className={`pill ${run.status === "completed" ? "active" : ""}`}>{run.status}</span>
                  <span className="pill">{run.summary.completed}/{run.summary.total} done</span>
                  <span className="pill">WER avg {formatPercent(run.summary.wer_avg)}</span>
                  <span className="pill">median {formatPercent(run.summary.wer_p50)}</span>
                  <span className="pill">p95 {formatPercent(run.summary.wer_p95)}</span>
                  <span className="pill">worst {formatPercent(run.summary.wer_max)}</span>
                  <span className="pill">exact {formatPercent(run.summary.exact_rate)}</span>
                  <span className={`pill ${run.sentence_cache_hit ? "active" : ""}`}>{run.sentence_cache_hit ? "cached sentences" : "new sentences"}</span>
                </div>
                {run.seed_leaderboard.length > 1 ? (
                  <div className="table-wrap wer-leaderboard-wrap">
                    <table className="downloads-table wer-leaderboard-table">
                      <thead>
                        <tr>
                          <th>#</th>
                          <th>Seed</th>
                          <th>WER avg</th>
                          <th>Median</th>
                          <th>Worst</th>
                          {run.seed_leaderboard.some((item) => item.failure_count > 0) ? <th>Fail</th> : null}
                        </tr>
                      </thead>
                      <tbody>
                        {run.seed_leaderboard.map((item, index) => (
                          <tr key={item.seed ?? "random"}>
                            <td>{index + 1}</td>
                            <td>{item.seed ?? "random"}</td>
                            <td>{formatPercent(item.wer_avg)}</td>
                            <td>{formatPercent(item.wer_p50)}</td>
                            <td>{formatPercent(item.wer_max)}</td>
                            {run.seed_leaderboard.some((entry) => entry.failure_count > 0) ? <td>{item.failure_count}</td> : null}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : null}
                <div className="metric-list">
                  <div className="metric-row"><span>Global WER avg / median</span><strong>{formatPercent(run.summary.wer_avg)} / {formatPercent(run.summary.wer_p50)}</strong></div>
                  <div className="metric-row"><span>Global WER p95 / worst</span><strong>{formatPercent(run.summary.wer_p95)} / {formatPercent(run.summary.wer_max)}</strong></div>
                  {run.summary.failure_count > 0 ? <div className="metric-row"><span>OK / Fail</span><strong>{run.summary.success_count} / {run.summary.failure_count}</strong></div> : null}
                </div>
                <div className="wer-sample-list">
                  {run.results.filter((item) => item.success && (item.wer ?? 0) > 0).length === 0 ? (
                    <p className="widget-copy">Keine WER-Abweichungen im aktuellen Run.</p>
                  ) : null}
                  {run.results.filter((item) => item.success && (item.wer ?? 0) > 0).map((item) => (
                    <div key={`${item.seed ?? "random"}-${item.index}-${item.job_id ?? ""}`} className="wer-sample">
                      <div className="metric-row">
                        <span>{item.seed !== null && item.seed !== undefined ? `Seed ${item.seed} #${item.index}` : `#${item.index}`} WER</span>
                        <strong>{item.success ? formatPercent(item.wer) : "failed"}</strong>
                      </div>
                      <div className="button-row compact">
                        <button
                          className="secondary-button"
                          type="button"
                          disabled={!item.job_id}
                          onClick={() => item.job_id && void handleLoadJobAudio(item.job_id)}
                        >
                          Audio
                        </button>
                      </div>
                      {item.job_id && jobAudioUrls[item.job_id] ? <audio controls src={jobAudioUrls[item.job_id]} /> : null}
                      <div className="inline-pills">
                        <span className="pill">TTS {formatMs(item.synthesis_ms)}</span>
                        <span className="pill">ASR {formatMs(item.transcription_ms)}</span>
                        <span className="pill">Total {formatMs(item.total_ms)}</span>
                      </div>
                      <p><span>Ref</span>{item.source_text}</p>
                      <p><span>ASR</span>{item.transcript || item.error_message || "-"}</p>
                    </div>
                  ))}
                </div>
              </article>
            ))}
          </div>
        </section>

        <section className="widget span-6 audio-card"><div className="widget-header"><h2>Quick Synthesis</h2></div>
          <div className="field-grid two">
            <label className="with-help" title={ADMIN_HELP.quickModel}>Modell<select value={quickModel} onChange={(event) => setQuickModel(event.target.value)}>{models.map((model) => <option key={model.model_id} value={model.model_id}>{model.model_id}</option>)}</select></label>
            <label className="with-help" title={ADMIN_HELP.quickVoice}>Stimme<select value={quickVoice} onChange={(event) => handleQuickVoiceChange(event.target.value)} disabled={quickTaskType === "VoiceDesign"}>{quickVoices.map((voice) => <option key={voice.voice_id} value={voice.source === "custom" ? voice.voice_id : voice.name}>{voice.name} ({voice.source})</option>)}</select></label>
          </div>
          <label className="with-help" title={ADMIN_HELP.quickText}>Text<textarea value={quickText} onChange={(event) => setQuickText(event.target.value)} /></label>
          <label className="with-help" title={ADMIN_HELP.quickInstructions}>Instructions<textarea value={quickInstructions} onChange={(event) => setQuickInstructions(event.target.value)} readOnly={quickTaskType === "VoiceDesign"} /></label>
          {quickTaskType === "VoiceDesign" ? (
            <div className="voice-design-builder" title={ADMIN_HELP.quickInstructions}>
              {VOICE_DESIGN_GROUPS.map((group) => (
                <label key={group.id} className="with-help" title={ADMIN_HELP.quickInstructions}>
                  {group.label}
                  <select
                    value={voiceDesignValueForGroup(quickInstructions, group.id)}
                    onChange={(event) => setQuickInstructions(setVoiceDesignValue(quickInstructions, group.id, event.target.value))}
                  >
                    <option value="">Auto</option>
                    {group.options.map((option) => (
                      <option key={option.value} value={option.value}>{option.label}</option>
                    ))}
                  </select>
                </label>
              ))}
            </div>
          ) : null}
          <label className="with-help" title={ADMIN_HELP.quickSeed}>Seed<input type="number" min="0" max="2147483647" value={quickSeed} onChange={(event) => setQuickSeed(event.target.value)} placeholder="leer = zufaellig" /></label>
          <div className="button-row"><button className="primary-button" type="button" onClick={handleQuickRun}>{quickRunning ? "Stoppen" : "Stream starten"}</button></div>
          {quickAudioUrl ? <audio controls src={quickAudioUrl} /> : null}
          {quickMetrics ? <div className="metric-list"><div className="metric-row"><span>TTFA</span><strong>{formatMs(quickMetrics.ttfa_ms)}</strong></div><div className="metric-row"><span>Realtime</span><strong>{formatRealtime(quickMetrics.realtime_x)}</strong></div><div className="metric-row"><span>Duration</span><strong>{formatSeconds((quickMetrics.audio_duration_ms || 0) / 1000)}</strong></div></div> : null}
        </section>

        <section className="widget span-6"><div className="widget-header"><h2>Voice Library</h2></div>
          <div className="field-grid">
            <label className="with-help" title={ADMIN_HELP.voiceName}>Name<input value={voiceName} onChange={(event) => setVoiceName(event.target.value)} /></label>
            <label className="with-help" title={ADMIN_HELP.voiceRefText}>Referenztext<textarea value={voiceRefText} onChange={(event) => setVoiceRefText(event.target.value)} /></label>
            <label className="with-help" title={ADMIN_HELP.voiceSample}>Sample<input type="file" accept="audio/*" onChange={(event) => handleVoiceFileChange(event.target.files?.[0] ?? null)} /></label>
            <div className="voice-consent-preview">
              <label className="checkbox-row with-help" title={ADMIN_HELP.voiceConsent}><input type="checkbox" checked={voiceConsent} onChange={(event) => setVoiceConsent(event.target.checked)} />Zustimmung bestaetigt</label>
              {voiceUploadPreviewUrl ? <audio className="inline-audio" controls src={voiceUploadPreviewUrl} /> : null}
            </div>
            <div className="button-row"><button className="secondary-button" type="button" onClick={() => void handleTranscribeVoice()} disabled={!voiceFile || voiceTranscribing} title={ADMIN_HELP.transcribeSample}>{voiceTranscribing ? "Transkribiere..." : "Mit Whisper transkribieren"}</button><button className="primary-button" type="button" onClick={handleUploadVoice}>Voice speichern</button></div>
          </div>
          <div className="voice-list">
            {libraryVoices.length === 0 ? <p className="widget-copy">Noch keine Custom Voices gespeichert.</p> : null}
            {libraryVoices.map((voice) => (
              <article key={voice.voice_id} className="voice-card">
                <div className="voice-card-head">
                  <div className="voice-card-title">
                    <strong>{voice.name}</strong>
                    <div className="inline-pills">
                      <span className="pill">{voice.source}</span>
                      <span className="pill">{formatDate(voice.created_at)}</span>
                      <span className="pill">{voice.filename || (voice.has_audio ? `${voice.voice_id}.wav` : "-")}</span>
                    </div>
                  </div>
                  <div className="voice-card-buttons">
                    <button className="secondary-button" type="button" disabled={!voice.has_audio} onClick={() => void handleLoadVoiceSample(voice.voice_id)}>
                      Original
                    </button>
                    <button className="ghost-button danger-button" type="button" onClick={() => void handleDeleteVoice(voice.voice_id)}>Delete</button>
                  </div>
                </div>
                <div className="voice-ref-text">
                  <span>Referenztext</span>
                  <p>{voice.ref_text || "-"}</p>
                </div>
                {voiceAudioUrls[voice.voice_id] ? <audio controls src={voiceAudioUrls[voice.voice_id]} /> : null}
              </article>
            ))}
          </div>
        </section>

        <section className="widget span-12"><div className="widget-header"><h2>History</h2></div>
          <div className="job-list">{snapshot.jobs.map((job) => <article key={job.job_id} className="job-card"><strong>{job.input_preview}</strong><div className="inline-pills"><span className={`pill ${job.status === "completed" ? "active" : ""}`}>{job.status}</span><span className="pill">{job.model || "-"}</span><span className="pill">{job.voice || "-"}</span><span className="pill">{formatMs(job.metrics.ttfa_ms)}</span></div><div className="button-row">{job.status === "completed" ? <button className="secondary-button" type="button" onClick={() => void handleLoadJobAudio(job.job_id)}>Audio</button> : null}<button className="ghost-button" type="button" onClick={() => void handleDeleteJob(job.job_id)}>{job.status === "completed" ? "Entfernen" : "Stornieren"}</button></div>{jobAudioUrls[job.job_id] ? <audio controls src={jobAudioUrls[job.job_id]} /> : null}</article>)}</div>
        </section>
      </section>
    </main>
  );
}
