import { useEffect, useMemo, useRef, useState } from "react";

import {
  apiFetch,
  changePassword,
  clearStoredAdminKey,
  createApiKey,
  createWavBlobFromInt16Chunks,
  decodePcm16Base64,
  deleteApiKey,
  formatDate,
  formatMs,
  formatRealtime,
  formatSeconds,
  listApiKeys,
  login,
  logout,
  streamNdjson,
  streamSse,
  whoami,
  type ApiKeyInfo,
  type BenchmarkRunResponse,
  type CreatedApiKey,
  type DashboardSnapshot,
  type EngineInfo,
  type EngineListResponse,
  type JobMetrics,
  type MemoryCleanupResponse,
  type ModelDownloadListResponse,
  type ModelDownloadStatus,
  type ModelOperationResponse,
  type RuntimeDeviceInfo,
  type RuntimeDeviceListResponse,
  type SeedEngine,
  type ServerSettings,
  type SynthStreamEvent,
  type TaskType,
  type TextOptimizationMode,
  type TextOptimizeResponse,
  type VllmModelsResponse,
  type VoiceItem,
  type WerBenchmarkDeviationsResponse,
  type WerBenchmarkRunResponse,
} from "../shared/api";
import { SYNTH_LANGUAGE_OPTIONS } from "../shared/languages";
import {
  EMPTY_MING_DELIVERY,
  MING_EMOTION_OPTIONS,
  MING_PITCH_OPTIONS,
  MING_RATE_OPTIONS,
  MING_TAGS,
  resolvedReferenceMode,
  MING_VOLUME_OPTIONS,
  REFERENCE_MODE_OPTIONS,
  hasStructuredMingControl,
  mingDeliveryFields,
  type MingDeliveryDraft,
} from "../shared/mingDelivery";
import { ActiveModelPanel } from "./ActiveModelPanel";
import { DataGenerationPanel } from "./DataGenerationPanel";
import { SettingsFields, pickFields } from "./EngineSettingsFields";
import { ModelPicker } from "./ModelPicker";
import { TrainingPanel } from "./TrainingPanel";

type AdminTab = "overview" | "finetune_data" | "finetune_train";

function inferTaskType(modelId: string): TaskType {
  if (modelId.endsWith("VoiceDesign")) return "VoiceDesign";
  if (modelId.endsWith("Base")) return "Base";
  return "CustomVoice";
}

const TERMINAL_RUN_STATUSES = ["completed", "failed", "cancelled", "error"];
function isRunActive(status: string | undefined): boolean {
  return !!status && !TERMINAL_RUN_STATUSES.includes(status);
}

// Built-in GPU tiers ("32GB VRAM", ...) come first and are read-only; saved presets follow by name.
type SettingsPreset = {
  name: string;
  values: Partial<ServerSettings>;
  /** The engine this preset tunes; the admin lists the active engine's presets only. */
  engine: string;
  created_at?: string | null;
  built_in?: boolean;
  description?: string;
  target_vram_gb?: number | null;
  note?: string | null;
};

/** The engine changes only through the model picker: neither a stale draft nor a loaded preset may carry it along. */
function withoutEngine<T extends Partial<ServerSettings>>(values: T): Omit<T, "runtime_backend"> {
  const { runtime_backend: _engine, ...rest } = values;
  return rest;
}

const OMNIVOICE_DEFAULTS = {
  num_step: 25,
  guidance_scale: 2.0,
  t_shift: 0.1,
  denoise: true,
  preprocess_prompt: true,
  postprocess_output: true,
  audio_chunk_duration: 30.1,
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

const MAX_TTS_SEED = 2_147_483_647;

function parseWerSeedInput(value: string): { randomSeed: number | null; seedValues: number[] | null; error?: string } {
  const trimmed = value.trim();
  if (!trimmed) {
    return { randomSeed: null, seedValues: null };
  }
  const parts = trimmed.split(",").map((part) => part.trim());
  if (parts.length === 1) {
    if (!/^\d+$/.test(parts[0])) {
      return { randomSeed: null, seedValues: null, error: "Seed muss eine ganze Zahl sein." };
    }
    const seed = Number(parts[0]);
    if (!Number.isSafeInteger(seed) || seed > MAX_TTS_SEED) {
      return { randomSeed: null, seedValues: null, error: `Seed muss zwischen 0 und ${MAX_TTS_SEED} liegen.` };
    }
    return { randomSeed: seed, seedValues: null };
  }
  const seeds: number[] = [];
  const seen = new Set<number>();
  for (const part of parts) {
    if (!part || !/^\d+$/.test(part)) {
      return { randomSeed: null, seedValues: null, error: "Seed-Liste bitte als ganze Zahlen mit Kommas eingeben, z.B. 8888,8890,2912." };
    }
    const seed = Number(part);
    if (!Number.isSafeInteger(seed) || seed > MAX_TTS_SEED) {
      return { randomSeed: null, seedValues: null, error: `Alle Seeds muessen zwischen 0 und ${MAX_TTS_SEED} liegen.` };
    }
    if (!seen.has(seed)) {
      seeds.push(seed);
      seen.add(seed);
    }
  }
  if (seeds.length > 1025) {
    return { randomSeed: null, seedValues: null, error: "Seed-Liste darf maximal 1025 Werte enthalten." };
  }
  return { randomSeed: seeds[0] ?? null, seedValues: seeds };
}

const ADMIN_HELP = {
  adminSession: "Admin-Zugang laeuft ueber Login und httpOnly Session-Cookie.",
  quickModel: "Modell fuer die schnelle Admin-Synthese.",
  quickVoice: "Stimme fuer Quick Synthesis. Custom Voices laufen mit dem Base-Modell.",
  quickText: "Text fuer den Streaming-Test. Der erste Satz bekommt einen Fast-Path, danach werden kompatible Saetze wieder gebatcht.",
  quickLanguage: "Sprachhinweis fuer OmniVoice. Auto laesst das Modell die Sprache erkennen; eine feste Wahl kann Aussprache und Qualitaet leicht verbessern.",
  quickInstructions: "Optionale Anweisung, die mit dem Request an die Engine geht.",
  quickMingInstruction: "Ming: freie Anweisung zur Sprechweise (z.B. \"spricht ruhig und warm, leicht laechelnd\"). Nicht mit Emotion/Tempo/Tonhoehe/Lautstaerke kombinierbar; solange eines davon gesetzt ist, wird sie nicht gesendet.",
  mingEmotion: "Ming: Emotion der Ausgabe. Geht zusammen mit Tempo/Tonhoehe/Lautstaerke als eine strukturierte Anweisung an das Modell.",
  mingRate: "Ming: Sprechtempo. Ist einer von Tempo/Tonhoehe/Lautstaerke gesetzt, fuellt der Server die anderen mit 'mittel'.",
  mingPitch: "Ming: Tonhoehe der Ausgabe.",
  mingVolume: "Ming: Lautstaerke der Ausgabe.",
  mingReferenceMode: "Ming: wie die Klon-Stimme genutzt wird. voll = Referenz-Audio + Transkript + Sprecher-Embedding (normales Klonen). nur Sprecher = nur das Sprecher-Embedding, damit die Sprechweise der Referenz die Anweisung nicht ueberdeckt. auto = nur Sprecher, sobald Emotion/Tempo/Tonhoehe/Lautstaerke gesetzt ist, sonst voll.",
  mingTags: "Ming: fuegt einen Geraeusch-Tag an der Cursorposition im Text ein. Tags werden nicht vorgelesen; eine Engine ohne diesen Tag entfernt ihn.",
  quickSeed: "Seed fuer reproduzierbarere Runs. Leer lassen fuer zufaellige Ausgabe.",
  quickTextOptimization: "Textoptimierung fuer diesen Run: Datum, Uhrzeit, Zahlen, Einheiten, URLs und Markdown werden fuer Deutsch/Englisch ausgeschrieben. Default = Server-Einstellung. Vorschau zeigt den gesprochenen Text ohne Synthese (bei 'off' den Originaltext). Sprache Auto: Deutsch/Englisch wird an Schreibweise (24.09.2026, 14:30 Uhr, 3:30 PM) oder an typischen Woertern erkannt; andere Sprachen bleiben unveraendert.",
  refreshVllmModels: "Liest die Modellliste aus dem konfigurierten vLLM Server neu aus.",
  refreshRuntimeDevices: "Liest CPU/CUDA-Geraete neu ein.",
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
  werSeed: "TTS-Seed fuer den WER-Run. Einzelwert oder kommagetrennte Liste wie 8888,8890,2912. Bei einer Liste wird Seed range ignoriert.",
  werSeedRange: "Wenn groesser 0, testet der WER-Benchmark alle Seeds von Seed bis Seed + Range. Wird ignoriert, wenn Seed eine kommagetrennte Liste ist.",
  werVoice: "Stimme, mit der der WER-Benchmark rechnet. Standard = Stimme aus den Einstellungen. Nur eigene Stimmen (Klon-Profile) stehen zur Wahl.",
  werCancel: "Stoppt den laufenden WER-Benchmark: laufende und wartende TTS-Jobs werden abgebrochen, der Lauf endet mit Status cancelled. Bereits fertige Ergebnisse bleiben sichtbar.",
  voiceSeed: "Bester Seed dieser Stimme pro Engine (aus dem WER-Benchmark). Ohne eigenen Seed im Request und ohne Seed im WER-Benchmark nutzt die aktive Engine ihren Wert; ein Seed passt nur zu der Engine, auf der er gemessen wurde. Leer = kein Seed, dann zufaellig.",
  werSaveSeed: "Speichert diesen Seed als besten Seed der Stimme des Runs, fuer die Engine, auf der der Run lief.",
  werPrompt: "Optionaler eigener vLLM Prompt. Er sollte ein JSON-Array von Saetzen erzeugen.",
};

const SEED_ENGINES: Array<{ id: SeedEngine; label: string }> = [
  { id: "omnivoice", label: "OmniVoice" },
  { id: "ming", label: "Ming" },
  { id: "phonon", label: "Phonon" },
];

/** The seed slot an engine reads; mock (tests) renders like OmniVoice. */
function seedEngineKey(engine?: string | null): SeedEngine {
  return engine === "ming" || engine === "phonon" ? engine : "omnivoice";
}

function seedEngineLabel(engine?: string | null) {
  return SEED_ENGINES.find((item) => item.id === seedEngineKey(engine))?.label ?? "OmniVoice";
}

function voiceEngineSeed(voice: VoiceItem, engine?: string | null): number | null {
  const key = seedEngineKey(engine);
  return voice.engine_seeds?.[key] ?? (key === "omnivoice" ? voice.seed ?? null : null);
}

function preferredBaseModel(models: string[]) {
  return models.find((model) => model.endsWith("OmniVoice-Base")) || "";
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
    cuda_memory_trim_after_batch: settings.cuda_memory_trim_after_batch ?? true,
    audio_chunk_duration: settings.audio_chunk_duration ?? OMNIVOICE_DEFAULTS.audio_chunk_duration,
    audio_chunk_threshold: settings.audio_chunk_threshold ?? OMNIVOICE_DEFAULTS.audio_chunk_threshold,
    position_temperature: settings.position_temperature ?? OMNIVOICE_DEFAULTS.position_temperature,
    class_temperature: settings.class_temperature ?? OMNIVOICE_DEFAULTS.class_temperature,
  };
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

function formatPercent(value?: number | null) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return `${(Number(value) * 100).toFixed(Number(value) >= 0.1 ? 1 : 2)}%`;
}

function fallbackRuntimeDevice(deviceId: string): RuntimeDeviceInfo {
  return {
    id: deviceId,
    label: `${deviceId} - configured`,
    kind: deviceId.startsWith("cuda") ? "cuda" : "other",
    name: "Configured device",
    index: null,
    memory_total_mb: null,
    available: false,
    selected: true,
  };
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
  const [adminKey, setAdminKey] = useState("");
  const [bootLoading, setBootLoading] = useState(true);
  const [loginUsername, setLoginUsername] = useState("admin");
  const [loginPassword, setLoginPassword] = useState("");
  const [currentUser, setCurrentUser] = useState("");
  const [mustChange, setMustChange] = useState(false);
  const [pwCurrent, setPwCurrent] = useState("");
  const [pwNew, setPwNew] = useState("");
  const [pwConfirm, setPwConfirm] = useState("");
  const [apiKeys, setApiKeys] = useState<ApiKeyInfo[]>([]);
  const [newKeyAlias, setNewKeyAlias] = useState("");
  const [createdKey, setCreatedKey] = useState<CreatedApiKey | null>(null);
  const [snapshot, setSnapshot] = useState<DashboardSnapshot | null>(null);
  const [settingsDraft, setSettingsDraft] = useState<ServerSettings | null>(null);
  const [presets, setPresets] = useState<SettingsPreset[]>([]);
  const [presetName, setPresetName] = useState("");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [authLoading, setAuthLoading] = useState(false);
  const [quickModel, setQuickModel] = useState("");
  const [quickVoice, setQuickVoice] = useState("");
  // True once the admin picked another model: only then a settings save may send default_model.
  const [defaultModelPicked, setDefaultModelPicked] = useState(false);
  const [adminTab, setAdminTab] = useState<AdminTab>("overview");
  const [quickText, setQuickText] = useState("Das neue Adminpanel nutzt dieselbe Streaming-Pipeline wie die offene Demo.");
  const [quickLanguage, setQuickLanguage] = useState("Auto");
  const [quickInstructions, setQuickInstructions] = useState("");
  const [quickSeed, setQuickSeed] = useState("");
  const [quickMingDelivery, setQuickMingDelivery] = useState<MingDeliveryDraft>(EMPTY_MING_DELIVERY);
  const quickTextRef = useRef<HTMLTextAreaElement | null>(null);
  const [quickTextOptimization, setQuickTextOptimization] = useState<"" | TextOptimizationMode>("");
  const [quickOptimizePreview, setQuickOptimizePreview] = useState<TextOptimizeResponse | null>(null);
  const [quickMetrics, setQuickMetrics] = useState<JobMetrics | null>(null);
  // Shown inside the Quick Synthesis card: on a phone the page banners sit far above it.
  const [quickNote, setQuickNote] = useState<{ error: boolean; text: string } | null>(null);
  const [quickAudioUrl, setQuickAudioUrl] = useState("");
  const [quickAudioDownloadName, setQuickAudioDownloadName] = useState("omnivoice-admin-quick.mp3");
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
  const [runtimeDevices, setRuntimeDevices] = useState<RuntimeDeviceInfo[]>([]);
  const [engineList, setEngineList] = useState<EngineListResponse | null>(null);
  const [engineTarget, setEngineTarget] = useState("");
  const [engineSwitchBusy, setEngineSwitchBusy] = useState(false);
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
  const [werVoice, setWerVoice] = useState("");
  const [werCancelling, setWerCancelling] = useState("");
  const [voiceSeedEdits, setVoiceSeedEdits] = useState<Record<string, string>>({});
  const [werDeviations, setWerDeviations] = useState<{ runId: string; total: number; items: WerBenchmarkRunResponse["results"] } | null>(null);
  const latestWerRun = werBenchmarks[0];
  useEffect(() => {
    // Only the first page of deviations is fetched; the run list itself carries no per-sample results.
    if (!latestWerRun || !(latestWerRun.results_count ?? 0)) return;
    void loadWerDeviations(latestWerRun.run_id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [latestWerRun?.run_id, latestWerRun?.results_count]);
  const [modelOpsBusy, setModelOpsBusy] = useState(false);
  const [modelDownloadAction, setModelDownloadAction] = useState<{ kind: "refresh" | "download" | "delete"; modelId: string } | null>(null);
  const [memoryCleanupBusy, setMemoryCleanupBusy] = useState(false);
  const [benchmarkBusy, setBenchmarkBusy] = useState(false);
  const [werBenchmarkBusy, setWerBenchmarkBusy] = useState(false);
  const [voiceTranscribing, setVoiceTranscribing] = useState(false);
  const audioContextRef = useRef<AudioContext | null>(null);
  const nextPlaybackTimeRef = useRef(0);
  const quickSourcesRef = useRef(new Set<AudioBufferSourceNode>());
  // The rate the live context was asked for; a fallback context runs at the hardware rate instead.
  const audioContextRequestedRateRef = useRef(0);
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

  // Bootstrap: resolve the admin session cookie on mount.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const me = await whoami();
        if (cancelled) return;
        setCurrentUser(me.username);
        setMustChange(me.must_change_password);
        setAdminKey("session");
      } catch {
        if (!cancelled) setAdminKey("");
      } finally {
        if (!cancelled) setBootLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (adminKey !== "session" || mustChange) return;
    void refreshApiKeys();
    void loadSnapshot(adminKey);
  }, [adminKey, mustChange]);

  async function refreshApiKeys() {
    try {
      const result = await listApiKeys();
      setApiKeys(result.keys);
    } catch {
      // ignore (surfaced elsewhere)
    }
  }

  async function loadSnapshot(key: string) {
    const data = await apiFetch<DashboardSnapshot>("/api/admin/snapshot", { adminKey: key });
    setSnapshot(data);
    setSettingsDraft(applyOmniVoiceDefaults(data.settings));
    setDefaultModelPicked(false);
    setQuickModel((current) => current || data.settings.default_model);
    setOverviewHistory((current) => [...current.slice(-119), { ...data.overview, recorded_at: new Date().toISOString() }]);
    void loadBenchmarks(key);
    void loadWerBenchmarks(key);
    void loadModelDownloads(key, data.settings.model_directory);
    void loadRuntimeDevices(key);
    void loadVllmModels(key, data.settings.vllm_base_url);
    void loadEngines(key);
  }

  async function loadBenchmarks(key = adminKey) {
    if (!key) return;
    try {
      setBenchmarks(await apiFetch<BenchmarkRunResponse[]>("/api/admin/benchmarks/runs", { adminKey: key }));
    } catch {
      // Benchmarks are auxiliary; keep the dashboard usable if this endpoint is mid-run or unavailable.
    }
  }

  function deviationsFor(runId: string): { total: number; items: WerBenchmarkRunResponse["results"] } {
    return werDeviations && werDeviations.runId === runId ? werDeviations : { total: 0, items: [] };
  }

  async function loadWerDeviations(runId: string, more = false) {
    if (!adminKey) return;
    const current = deviationsFor(runId).items;
    const offset = more ? current.length : 0;
    try {
      const page = await apiFetch<WerBenchmarkDeviationsResponse>(
        `/api/admin/wer-benchmarks/runs/${encodeURIComponent(runId)}/deviations?offset=${offset}&limit=100`,
        { adminKey },
      );
      setWerDeviations({ runId, total: page.total, items: more ? [...current, ...page.items] : page.items });
    } catch {
      // Deviations are auxiliary; the run card keeps working without them.
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

  async function loadRuntimeDevices(key = adminKey) {
    if (!key) return;
    try {
      const payload = await apiFetch<RuntimeDeviceListResponse>("/api/admin/runtime/devices", { adminKey: key });
      setRuntimeDevices(payload.devices || []);
    } catch {
      setRuntimeDevices([]);
    }
  }

  async function loadEngines(key = adminKey): Promise<EngineListResponse | null> {
    if (!key) return null;
    try {
      const payload = await apiFetch<EngineListResponse>("/api/admin/runtime/engines", { adminKey: key });
      setEngineList(payload);
      setEngineTarget((current) => current || payload.active);
      return payload;
    } catch {
      return null;
    }
  }

  async function handleEngineSwitch(target: string) {
    if (!adminKey || !target || !engineList || target === engineList.active) return;
    setError("");
    setEngineSwitchBusy(true);
    try {
      const started = await apiFetch<EngineListResponse>("/api/admin/runtime/backend", {
        method: "POST",
        adminKey,
        body: { backend: target },
      });
      setEngineList(started);
      setMessage(`Engine-Wechsel zu ${target} gestartet.`);
    } catch (switchError) {
      setEngineSwitchBusy(false);
      setError(switchError instanceof Error ? switchError.message : "Engine-Wechsel fehlgeschlagen.");
    }
  }

  async function finishEngineSwitch(result: EngineListResponse) {
    // Stays locked until the new engine's draft is in: the old one must not be edited or saved any more.
    setEngineSwitchBusy(true);
    setEngineTarget(result.active);
    // The model list, default model, voices and presets belong to the new engine now.
    setQuickModel("");
    setPresetName("");
    setQuickNote(null);
    try {
      await loadSnapshot(adminKey);
    } catch {
      // The stale-draft effect below retries.
    }
    setEngineSwitchBusy(false);
    void loadPresets(adminKey);
    // The "started" line must not stand next to the error.
    if (result.switch.state === "failed") {
      setMessage("");
      setError(`Engine-Wechsel fehlgeschlagen: ${result.switch.error || "unbekannter Fehler"}`);
    } else if (result.switch.error) {
      setMessage("");
      setError(result.switch.error);
    } else {
      setMessage(result.switch.message || `Engine ${result.active} aktiv.`);
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
    if (!adminKey || mustChange) return;
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
        setSnapshot(null);
      } else {
        setError(streamError instanceof Error ? streamError.message : "Dashboard-Stream getrennt.");
      }
    });
    return () => controller.abort();
  }, [adminKey, mustChange]);

  // Load benchmark/WER history + settings presets once when the key changes (e.g. on page reload).
  useEffect(() => {
    if (!adminKey || mustChange) return;
    void loadBenchmarks(adminKey);
    void loadWerBenchmarks(adminKey);
    void loadPresets(adminKey);
    void loadRuntimeDevices(adminKey);
    void loadEngines(adminKey);
  }, [adminKey, mustChange]);

  // A live engine switch (started here or elsewhere): poll its progress until it is done.
  const engineSwitchRunning =
    engineSwitchBusy || ["draining", "unloading", "loading"].includes(engineList?.switch.state ?? "") || Boolean(snapshot?.settings.runtime_backend_pending);
  useEffect(() => {
    if (!adminKey || mustChange || !engineSwitchRunning) return;
    const timer = window.setInterval(() => {
      void (async () => {
        const payload = await loadEngines(adminKey);
        if (payload && !["draining", "unloading", "loading"].includes(payload.switch.state)) {
          window.clearInterval(timer);
          await finishEngineSwitch(payload);
        }
      })();
    }, 700);
    return () => window.clearInterval(timer);
  }, [adminKey, mustChange, engineSwitchRunning]);

  // Another tab or an API call switched the engine and this tab missed the progress: the draft holds the old engine's
  // values, and the stream never replaces an existing draft.
  const draftEngineStale = Boolean(settingsDraft && snapshot && settingsDraft.runtime_backend !== snapshot.settings.runtime_backend);
  useEffect(() => {
    if (!adminKey || mustChange || engineSwitchRunning || !draftEngineStale) return;
    void (async () => {
      const payload = await loadEngines(adminKey);
      if (payload) await finishEngineSwitch(payload);
    })();
  }, [adminKey, mustChange, engineSwitchRunning, draftEngineStale]);

  // The stream does not carry the engine list, and a request can load a model on its own (Phonon does so on the first).
  const residentModels = snapshot?.models.filter((model) => model.loaded).map((model) => model.model_id).join();
  useEffect(() => {
    if (!adminKey || mustChange || residentModels === undefined) return;
    void loadEngines(adminKey);
  }, [adminKey, mustChange, residentModels]);

  // Poll the benchmark endpoints ONLY while a run is actually in progress, and at the
  // configured poll interval. Previously this polled both list endpoints (which return
  // every run's full results[]) every 1000ms unconditionally for as long as a key was
  // set, which flooded the server with ~2 req/sec/tab even when idle.
  useEffect(() => {
    if (!adminKey || mustChange) return;
    const active =
      benchmarkBusy ||
      werBenchmarkBusy ||
      benchmarks.some((run) => isRunActive(run.status)) ||
      werBenchmarks.some((run) => isRunActive(run.status));
    if (!active) return;
    const pollMs = Math.min(Math.max(snapshot?.settings.poll_interval_ms ?? 1000, 750), 5000);
    const timer = window.setInterval(() => {
      void loadBenchmarks(adminKey);
      void loadWerBenchmarks(adminKey);
    }, pollMs);
    return () => window.clearInterval(timer);
  }, [adminKey, mustChange, benchmarkBusy, werBenchmarkBusy, benchmarks, werBenchmarks, snapshot?.settings.poll_interval_ms]);

  const modelDownloadRunning =
    modelDownloads.some((model) => model.status === "downloading") || Boolean(engineList?.engines.some((engine) => engine.status === "downloading"));
  useEffect(() => {
    if (!adminKey || mustChange || !modelDownloadRunning) return;
    const timer = window.setInterval(() => {
      void loadModelDownloads(adminKey, settingsDraft?.model_directory || "");
      void loadEngines(adminKey);
    }, 2000);
    return () => {
      window.clearInterval(timer);
      // The last poll can predate the end of the download: the cards must not keep showing it as running.
      void loadEngines(adminKey);
    };
  }, [adminKey, mustChange, modelDownloadRunning, settingsDraft?.model_directory]);

  useEffect(() => {
    if (!adminKey || mustChange || !settingsDraft?.vllm_base_url) return;
    const timer = window.setTimeout(() => {
      void loadVllmModels(adminKey, settingsDraft.vllm_base_url);
    }, 500);
    return () => window.clearTimeout(timer);
  }, [adminKey, mustChange, settingsDraft?.vllm_base_url]);

  const voices = snapshot?.voices ?? [];
  const selectableModels = snapshot?.selectable_models ?? [];
  // A running server older than this bundle (dist is served from disk, a build needs no restart) sends no field lists:
  // treat its engine as not known yet instead of crashing the whole page on them.
  const activeEngine = engineList?.engines.find((engine) => engine.active && engine.settings_fields) ?? null;
  const enginePresets = presets.filter((preset) => preset.engine === activeEngine?.id);
  const selectedPreset = enginePresets.find((preset) => preset.name === presetName.trim()) ?? null;
  const quickTaskType = useMemo(() => inferTaskType(quickModel || snapshot?.settings.default_model || ""), [quickModel, snapshot]);
  const quickVoices = useMemo(() => voiceOptionsForTask(voices, quickTaskType), [quickTaskType, voices]);
  const defaultTaskType = useMemo(() => inferTaskType(settingsDraft?.default_model || ""), [settingsDraft?.default_model]);
  const defaultVoiceOptions = useMemo(() => voiceOptionsForTask(voices, defaultTaskType), [defaultTaskType, voices]);
  const runtimeDeviceOptions = useMemo(() => {
    const byId = new Map<string, RuntimeDeviceInfo>();
    runtimeDevices.forEach((device) => byId.set(device.id, device));
    if (settingsDraft?.preferred_device && !byId.has(settingsDraft.preferred_device)) {
      byId.set(settingsDraft.preferred_device, fallbackRuntimeDevice(settingsDraft.preferred_device));
    }
    if (!byId.size) {
      byId.set("cuda:0", fallbackRuntimeDevice("cuda:0"));
    }
    return Array.from(byId.values());
  }, [runtimeDevices, settingsDraft?.preferred_device]);
  const selectedRuntimeDeviceLabel = useMemo(() => {
    return runtimeDeviceOptions.find((device) => device.id === settingsDraft?.preferred_device)?.label || settingsDraft?.preferred_device || "-";
  }, [runtimeDeviceOptions, settingsDraft?.preferred_device]);
  const customVoices = useMemo(() => voices.filter((voice) => voice.source === "custom"), [voices]);
  // The engine whose voice seeds apply right now (requests and WER runs without a seed of their own).
  const activeEngineId = engineList?.active ?? snapshot?.settings.runtime_backend;
  const activeSeedEngine = seedEngineKey(activeEngineId);
  // Ming reads delivery controls and a free-form instruction; the other engines ignore them.
  const mingEngineActive = activeEngineId === "ming";
  const quickMingStructured = mingEngineActive && hasStructuredMingControl(quickMingDelivery);
  // Same rule as finetune_supported in the backend's engines.py; no engine list carries it.
  const finetuneSupported = activeEngineId === "omnivoice" || activeEngineId === "mock";
  const libraryVoices = customVoices;
  const settingsFieldContext = { voiceOptions: defaultVoiceOptions, devices: runtimeDeviceOptions, vllmModels, vllmLoading: vllmModelsLoading };

  useEffect(() => {
    if (!quickVoices.length) {
      setQuickVoice("");
      return;
    }
    setQuickVoice((current) =>
      current && quickVoices.some((voice) => voiceMatchesValue(voice, current))
        ? current
        : firstVoiceValueForTask(voices, quickTaskType),
    );
  }, [quickTaskType, quickVoices, voices]);

  useEffect(() => {
    const models = snapshot?.selectable_models;
    const defaultModel = snapshot?.settings.default_model;
    if (!models?.length || !defaultModel) return;
    // With a single model the select is hidden, so nothing else could correct a model that is gone.
    setQuickModel((current) => (models.includes(current) ? current : defaultModel));
  }, [snapshot?.selectable_models, snapshot?.settings.default_model]);

  useEffect(() => {
    if (activeEngineId && !finetuneSupported) setAdminTab("overview");
  }, [activeEngineId, finetuneSupported]);

  function patchDraft(patch: Partial<ServerSettings>) {
    setSettingsDraft((current) => (current ? { ...current, ...patch } : current));
  }

  function patchDefaultModel(modelId: string) {
    setDefaultModelPicked(true);
    patchDraft({ default_model: modelId });
  }

  function handleQuickVoiceChange(voiceId: string) {
    const voice = voices.find((item) => item.voice_id === voiceId || item.name === voiceId);
    if (voice?.source === "custom" && quickTaskType !== "Base") {
      const baseModel = preferredBaseModel(selectableModels);
      if (baseModel) setQuickModel(baseModel);
    }
    setQuickVoice(voiceId);
  }

  function patchQuickMingDelivery(patch: Partial<MingDeliveryDraft>) {
    setQuickMingDelivery((current) => ({ ...current, ...patch }));
  }

  /** Inserts a Ming tag at the cursor of the text field (replacing a selection), padded with spaces. */
  function insertQuickTag(tag: string) {
    const field = quickTextRef.current;
    const start = field?.selectionStart ?? quickText.length;
    const end = field?.selectionEnd ?? quickText.length;
    const before = quickText.slice(0, start);
    const after = quickText.slice(end);
    const inserted = `${before && !/\s$/.test(before) ? " " : ""}${tag}${after && !/^\s/.test(after) ? " " : ""}`;
    setQuickText(`${before}${inserted}${after}`);
    const caret = before.length + inserted.length;
    window.requestAnimationFrame(() => {
      field?.focus();
      field?.setSelectionRange(caret, caret);
    });
  }

  async function handleAuthenticate() {
    setAuthLoading(true);
    setError("");
    try {
      const me = await login(loginUsername.trim(), loginPassword);
      setCurrentUser(me.username);
      setMustChange(me.must_change_password);
      setLoginPassword("");
      setAdminKey("session");
      setMessage("Adminpanel verbunden.");
    } catch (authError) {
      setError(authError instanceof Error ? authError.message : "Login fehlgeschlagen.");
    } finally {
      setAuthLoading(false);
    }
  }

  async function handleChangePassword() {
    setError("");
    if (pwNew.length < 4) {
      setError("Passwort muss mindestens 4 Zeichen haben.");
      return;
    }
    if (pwNew !== pwConfirm) {
      setError("Passwoerter stimmen nicht ueberein.");
      return;
    }
    try {
      await changePassword(pwCurrent, pwNew);
      setPwCurrent("");
      setPwNew("");
      setPwConfirm("");
      setMustChange(false);
      setMessage("Passwort aktualisiert.");
    } catch (pwError) {
      setError(pwError instanceof Error ? pwError.message : "Passwortwechsel fehlgeschlagen.");
    }
  }

  async function handleCreateApiKey() {
    setError("");
    try {
      const created = await createApiKey(newKeyAlias.trim());
      setCreatedKey(created);
      setNewKeyAlias("");
      setMessage(`API-Key "${created.alias}" erstellt — jetzt kopieren, er wird nur einmal gezeigt.`);
      await refreshApiKeys();
    } catch (keyError) {
      setError(keyError instanceof Error ? keyError.message : "API-Key konnte nicht erstellt werden.");
    }
  }

  async function handleDeleteApiKey(id: string, alias: string) {
    setError("");
    try {
      await deleteApiKey(id);
      setCreatedKey((current) => (current?.id === id ? null : current));
      setMessage(`API-Key "${alias}" geloescht.`);
      await refreshApiKeys();
    } catch (keyError) {
      setError(keyError instanceof Error ? keyError.message : "API-Key konnte nicht geloescht werden.");
    }
  }

  // Mobile browsers (every iOS browser, Chrome included) only let an AudioContext start inside the tap that asked for
  // it; a resume() requested later never settles there. So the context is created and resumed synchronously in the
  // click handler, and chunk playback only schedules onto it without ever waiting for it.
  function unlockAudioPlayback(sampleRate: number) {
    const AudioContextImpl = window.AudioContext || (window as typeof window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!AudioContextImpl) return;
    // iOS 17+: play through the ringer switch instead of staying silent.
    const audioSession = (navigator as Navigator & { audioSession?: { type: string } }).audioSession;
    if (audioSession) {
      try {
        audioSession.type = "playback";
      } catch {
        // optional
      }
    }
    // A context that is not running (iOS "interrupted" after a call, or never unlocked) is rebuilt while the tap
    // still allows it, rather than trusting a resume() that may never take effect.
    const current = audioContextRef.current;
    if (current && (current.state !== "running" || audioContextRequestedRateRef.current !== sampleRate)) {
      if (current.state !== "closed") void current.close().catch(() => undefined);
      audioContextRef.current = null;
    }
    if (!audioContextRef.current) {
      audioContextRequestedRateRef.current = sampleRate;
      try {
        audioContextRef.current = new AudioContextImpl({ sampleRate });
      } catch {
        // Some browsers reject a rate the hardware cannot run; their default rate resamples the chunks instead.
        try {
          audioContextRef.current = new AudioContextImpl();
        } catch {
          return;
        }
      }
    }
    const context = audioContextRef.current;
    if (context.state !== "running") void context.resume().catch(() => undefined);
    // Older iOS only unlocks once something was started within the gesture.
    const silence = context.createBufferSource();
    silence.buffer = context.createBuffer(1, 1, context.sampleRate);
    silence.connect(context.destination);
    silence.start();
  }

  function stopQuickPlayback() {
    for (const source of quickSourcesRef.current) {
      try {
        source.stop();
      } catch {
        // already ended
      }
    }
    quickSourcesRef.current.clear();
    nextPlaybackTimeRef.current = 0;
  }

  function queuePlayback(float32: Float32Array, chunkSampleRate: number) {
    const context = audioContextRef.current;
    if (!context || context.state === "closed") return;
    const buffer = context.createBuffer(1, float32.length, chunkSampleRate);
    buffer.getChannelData(0).set(float32);
    const source = context.createBufferSource();
    source.buffer = buffer;
    source.connect(context.destination);
    const startAt = Math.max(context.currentTime + 0.06, nextPlaybackTimeRef.current);
    source.onended = () => quickSourcesRef.current.delete(source);
    quickSourcesRef.current.add(source);
    source.start(startAt);
    nextPlaybackTimeRef.current = startAt + buffer.duration;
  }

  async function handleQuickOptimizePreview() {
    setError("");
    const effectiveMode = quickTextOptimization || snapshot?.settings.text_optimization_default || "off";
    if (effectiveMode === "off") {
      // The run would speak the raw text: show that instead of a deterministic rewrite.
      setQuickOptimizePreview({ text: quickText.trim(), applied_rules: [], language: null, binding_origin: "Textoptimierung aus", sentences: [] });
      return;
    }
    try {
      const preview = await apiFetch<TextOptimizeResponse>("/api/admin/text/optimize", {
        method: "POST",
        adminKey,
        body: { input: quickText, language: quickLanguage },
      });
      setQuickOptimizePreview(preview);
    } catch (previewError) {
      setError(previewError instanceof Error ? previewError.message : "Textoptimierung fehlgeschlagen.");
    }
  }

  async function handleQuickRun() {
    if (quickRunning) {
      // Stop must work at once, whatever the run is waiting on: the aborted run only cleans up after itself.
      quickAbortRef.current?.abort();
      quickAbortRef.current = null;
      stopQuickPlayback();
      setQuickRunning(false);
      setQuickNote(null);
      setMessage("Quick-Synthesis gestoppt.");
      return;
    }
    if (!quickModel || !quickVoice) {
      setQuickNote({ error: true, text: "Keine passende Stimme fuer dieses Modell." });
      return;
    }
    let sampleRate = (mingEngineActive ? snapshot?.settings.ming_output_sample_rate : undefined) ?? snapshot?.settings.sample_rate ?? 24000;
    stopQuickPlayback();
    unlockAudioPlayback(sampleRate);
    if (quickAudioUrl) URL.revokeObjectURL(quickAudioUrl);
    setQuickAudioUrl("");
    setQuickAudioDownloadName("omnivoice-admin-quick.mp3");
    setQuickMetrics(null);
    setQuickNote(null);
    setQuickRunning(true);
    setError("");
    const controller = new AbortController();
    const chunks: Int16Array[] = [];
    let completedJobId = "";
    quickAbortRef.current = controller;
    const parsedSeed = quickSeed.trim() ? Number(quickSeed) : null;

    try {
      await streamNdjson("/api/v1/synthesize/stream", {
        signal: controller.signal,
        body: {
          input: quickText,
          model: quickModel,
          voice: quickVoice,
          task_type: quickTaskType,
          // On Ming a structured control replaces the free instruction (the server rejects both together).
          instructions: quickMingStructured ? "" : quickInstructions,
          language: quickLanguage,
          stream: true,
          response_format: "pcm",
          seed: parsedSeed !== null && Number.isFinite(parsedSeed) ? parsedSeed : null,
          text_optimization: quickTextOptimization || null,
          ...(mingEngineActive ? mingDeliveryFields(quickMingDelivery) : {}),
        },
        onEvent: (event: SynthStreamEvent) => {
          if (controller.signal.aborted) return;
          if (event.type === "start") {
            completedJobId = event.job_id;
            const info = event.text_optimization;
            setQuickOptimizePreview(
              info
                ? { text: info.spoken_text ?? "", applied_rules: info.applied_rules, language: info.language, binding_origin: info.binding_origin ?? "none", sentences: [] }
                : null,
            );
          }
          if (event.type === "chunk") {
            sampleRate = event.sample_rate;
            const decoded = decodePcm16Base64(event.pcm16_b64);
            chunks.push(decoded.int16);
            queuePlayback(decoded.float32, event.sample_rate);
          }
          if (event.type === "done") {
            completedJobId = event.result.job_id || completedJobId;
            setQuickMetrics(event.result.metrics);
          }
          if (event.type === "error") throw new Error(event.message);
        },
      });
      if (controller.signal.aborted) return;
      // Blocked live playback (no tap reached the audio context) leaves the player below as the way to listen.
      const liveBlocked = chunks.length > 0 && audioContextRef.current?.state !== "running";
      const liveNote = liveBlocked ? " Live-Wiedergabe hat der Browser blockiert, bitte im Player abspielen." : "";
      if (liveBlocked) setQuickNote({ error: false, text: liveNote.trim() });
      if (completedJobId) {
        try {
          const mp3Blob = await apiFetch<Blob>(
            `/api/admin/jobs/${encodeURIComponent(completedJobId)}/audio?format=mp3`,
            { adminKey, responseType: "blob" },
          );
          if (controller.signal.aborted) return;
          setQuickAudioDownloadName(`${completedJobId}.mp3`);
          setQuickAudioUrl(URL.createObjectURL(mp3Blob));
          setMessage(`Quick-Synthesis fertig. MP3 ist bereit.${liveNote}`);
        } catch (mp3Error) {
          if (controller.signal.aborted) return;
          if (chunks.length) {
            setQuickAudioDownloadName(`${completedJobId || "omnivoice-admin-quick"}.wav`);
            setQuickAudioUrl(URL.createObjectURL(createWavBlobFromInt16Chunks(chunks, sampleRate)));
          }
          const mp3Message = mp3Error instanceof Error ? `MP3-Export fehlgeschlagen: ${mp3Error.message}` : "MP3-Export fehlgeschlagen.";
          setError(mp3Message);
          setQuickNote({ error: true, text: `${mp3Message}${liveNote}` });
          setMessage(`Quick-Synthesis fertig. WAV-Fallback ist bereit.${liveNote}`);
        }
      } else if (chunks.length) {
        setQuickAudioDownloadName("omnivoice-admin-quick.wav");
        setQuickAudioUrl(URL.createObjectURL(createWavBlobFromInt16Chunks(chunks, sampleRate)));
        setMessage(`Quick-Synthesis fertig. WAV-Fallback ist bereit.${liveNote}`);
      } else {
        setMessage("Quick-Synthesis fertig.");
      }
    } catch (quickError) {
      if (!controller.signal.aborted) {
        const quickMessage = quickError instanceof Error ? quickError.message : "Quick-Synthesis fehlgeschlagen.";
        setError(quickMessage);
        setQuickNote({ error: true, text: quickMessage });
      }
    } finally {
      // A stopped run already reset the button; it must not flip a newer run back to idle.
      if (quickAbortRef.current === controller) {
        quickAbortRef.current = null;
        setQuickRunning(false);
      }
    }
  }

  async function loadPresets(key = adminKey) {
    if (!key) return;
    try {
      const payload = await apiFetch<{ presets: SettingsPreset[] }>("/api/admin/settings/presets", { adminKey: key });
      setPresets(payload.presets || []);
    } catch {
      // Presets are optional; keep the panel usable if the endpoint is unavailable.
    }
  }

  function handleApplyPreset(name: string) {
    const preset = presets.find((item) => item.name === name);
    if (!preset || !settingsDraft) return;
    // Load preset values into the draft only; the user reviews and clicks Save to apply.
    setSettingsDraft(applyOmniVoiceDefaults({ ...settingsDraft, ...withoutEngine(preset.values) } as ServerSettings));
    const note = preset.note ? ` Hinweis: ${preset.note}` : "";
    setMessage(`Preset "${name}" geladen — zum Anwenden "Settings speichern" klicken.${note}`);
  }

  async function handleSavePreset() {
    const name = presetName.trim();
    if (!adminKey || !settingsDraft || !activeEngine || !name) return;
    if (presets.some((preset) => preset.built_in && preset.name === name)) {
      setError(`"${name}" ist ein eingebautes Preset; bitte unter einem anderen Namen speichern.`);
      return;
    }
    // The list shows the active engine only, so a same-named preset of another engine would be overwritten unseen.
    const foreign = presets.find((preset) => preset.name === name && preset.engine !== activeEngine.id);
    if (foreign) {
      setError(`"${name}" gehoert zu ${foreign.engine}; bitte unter einem anderen Namen speichern.`);
      return;
    }
    setError("");
    try {
      await apiFetch<SettingsPreset>(`/api/admin/settings/presets/${encodeURIComponent(name)}`, {
        method: "PUT",
        adminKey,
        body: { values: { ...pickFields(settingsDraft, activeEngine.settings_fields), runtime_backend: activeEngine.id } },
      });
      await loadPresets(adminKey);
      setMessage(`Preset "${name}" gespeichert.`);
    } catch (presetError) {
      setError(presetError instanceof Error ? presetError.message : "Preset konnte nicht gespeichert werden.");
    }
  }

  async function handleDeletePreset() {
    const name = presetName.trim();
    if (!adminKey || !name) return;
    setError("");
    try {
      await apiFetch<{ ok: boolean }>(`/api/admin/settings/presets/${encodeURIComponent(name)}`, { method: "DELETE", adminKey });
      setPresetName("");
      await loadPresets(adminKey);
      setMessage(`Preset "${name}" geloescht.`);
    } catch (presetError) {
      setError(presetError instanceof Error ? presetError.message : "Preset konnte nicht geloescht werden.");
    }
  }

  async function handleSaveSettings() {
    if (!adminKey || !settingsDraft) return;
    const previousDevice = snapshot?.settings.preferred_device;
    const { default_model: defaultModel, ...values } = withoutEngine(settingsDraft);
    setError("");
    try {
      const updated = await apiFetch<ServerSettings>("/api/admin/settings", {
        method: "PUT",
        adminKey,
        body: defaultModelPicked ? { ...values, default_model: defaultModel } : values,
      });
      setSettingsDraft(applyOmniVoiceDefaults(updated));
      await loadSnapshot(adminKey);
      await loadModelDownloads(adminKey, updated.model_directory);
      setMessage(
        previousDevice && previousDevice !== updated.preferred_device
          ? "Settings gespeichert. GPU-Wechsel wird nach Model Reload, Engine-Wechsel oder Server-Neustart aktiv."
          : "Settings gespeichert.",
      );
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : "Settings konnten nicht gespeichert werden.");
    }
  }

  async function handleSetOpsModelAsDefault(modelId: string) {
    if (!adminKey || !modelId) return;
    setError("");
    try {
      const updated = await apiFetch<ServerSettings>("/api/admin/settings", { method: "PUT", adminKey, body: { default_model: modelId } });
      // Only the model was saved: other unsaved edits stay in the draft.
      setDefaultModelPicked(false);
      patchDraft({ default_model: updated.default_model });
      setMessage("Default-Modell gespeichert.");
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : "Default-Modell konnte nicht gespeichert werden.");
    }
  }

  async function handleModelOperation(kind: "preload" | "warmup" | "unload" | "reload") {
    if (!adminKey || !settingsDraft) return;
    const targetModel = settingsDraft.default_model;
    const targetTaskType = inferTaskType(targetModel);
    const warmupVoice = (targetTaskType === "Base" ? customVoices[0]?.voice_id : undefined) || settingsDraft.default_voice || null;
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
          instructions: "",
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

  /** `label` names the download in the UI; `target` is a model id, or an engine that is not active yet. */
  async function startModelDownload(label: string, target: { model_id: string } | { engine: string }) {
    if (!adminKey || !settingsDraft) return;
    setModelDownloadAction({ kind: "download", modelId: label });
    setError("");
    try {
      const payload = await apiFetch<ModelDownloadListResponse>("/api/admin/models/download", {
        method: "POST",
        adminKey,
        body: {
          ...target,
          storage_path: settingsDraft.model_directory,
        },
      });
      setModelDownloads(payload.models || []);
      await loadEngines(adminKey);
      setMessage(`Download fuer ${label} gestartet.`);
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
                voice: settingsDraft.default_voice,
                task_type: taskType,
                language: "Auto",
                instructions: benchmarkInstructions(),
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

  // Benchmark and WER runs measure the plain voice. On Ming the quick-synthesis textarea is the free-form
  // instruction (a caption the engine applies), so it is not sent with them.
  function benchmarkInstructions(): string {
    return mingEngineActive ? "" : quickInstructions;
  }

  async function handleWerBenchmarkRun() {
    if (!adminKey || !settingsDraft) return;
    setError("");
    const taskType = inferTaskType(settingsDraft.default_model);
    const parsedSeed = parseWerSeedInput(werSeed);
    if (parsedSeed.error) {
      setError(parsedSeed.error);
      return;
    }
    setWerBenchmarkBusy(true);
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
          random_seed: parsedSeed.randomSeed,
          seed_range: parsedSeed.seedValues ? 0 : werSeedRange,
          seed_values: parsedSeed.seedValues,
          exclusive: true,
          request: {
            model: settingsDraft.default_model,
            voice: werVoice || settingsDraft.default_voice,
            task_type: taskType,
            language: "Auto",
            instructions: benchmarkInstructions(),
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

  async function handleCancelWerBenchmark(runId: string) {
    if (!adminKey) return;
    setError("");
    setWerCancelling(runId);
    try {
      await apiFetch<WerBenchmarkRunResponse>(`/api/admin/wer-benchmarks/runs/${encodeURIComponent(runId)}/cancel`, { method: "POST", adminKey });
      setMessage("WER-Benchmark abgebrochen.");
      await loadWerBenchmarks(adminKey);
    } catch (cancelError) {
      setError(cancelError instanceof Error ? cancelError.message : "WER-Benchmark konnte nicht abgebrochen werden.");
    } finally {
      setWerCancelling("");
    }
  }

  async function saveVoiceSeed(voice: VoiceItem, engine: SeedEngine, seed: number | null) {
    if (!adminKey) return false;
    setError("");
    try {
      await apiFetch(`/api/admin/voices/${encodeURIComponent(voice.voice_id)}`, { method: "PATCH", adminKey, body: { seed, engine } });
      const label = seedEngineLabel(engine);
      setMessage(seed === null ? `${label}-Seed fuer ${voice.name} entfernt.` : `Bester ${label}-Seed fuer ${voice.name}: ${seed}.`);
      await loadSnapshot(adminKey);
      return true;
    } catch (seedError) {
      setError(seedError instanceof Error ? seedError.message : "Seed konnte nicht gespeichert werden.");
      return false;
    }
  }

  async function handleSaveVoiceSeed(voice: VoiceItem, engine: SeedEngine) {
    if (!adminKey) return;
    const editKey = `${voice.voice_id}:${engine}`;
    const raw = (voiceSeedEdits[editKey] ?? String(voiceEngineSeed(voice, engine) ?? "")).trim();
    let seed: number | null = null;
    if (raw !== "") {
      const parsed = Number(raw);
      if (!Number.isInteger(parsed) || parsed < 0 || parsed > 2147483647) {
        setError("Seed muss eine ganze Zahl zwischen 0 und 2147483647 sein.");
        return;
      }
      seed = parsed;
    }
    if (await saveVoiceSeed(voice, engine, seed)) {
      setVoiceSeedEdits((current) => {
        const next = { ...current };
        delete next[editKey];
        return next;
      });
    }
  }

  /** The voice profile a WER run rendered with (resolved by the server at its start), or undefined. */
  function werRunVoice(run: WerBenchmarkRunResponse): VoiceItem | undefined {
    const name = (run.resolved_voice || "").trim();
    if (!name) return undefined;
    const lowered = name.toLowerCase();
    return (
      customVoices.find((voice) => voice.voice_id === name || voice.name === name) ??
      customVoices.find((voice) => voice.voice_id.toLowerCase() === lowered || voice.name.toLowerCase() === lowered)
    );
  }

  /** Leaderboard cell: save the row's seed as the best seed of the run's voice for the run's engine. */
  function renderSaveSeedCell(run: WerBenchmarkRunResponse, seed: number | null | undefined) {
    const runVoice = werRunVoice(run);
    if (!runVoice || seed === null || seed === undefined) return "-";
    if (voiceEngineSeed(runVoice, run.engine) === seed) return <span className="pill active">gespeichert</span>;
    return (
      <button className="secondary-button" type="button" title={ADMIN_HELP.werSaveSeed} onClick={() => void saveVoiceSeed(runVoice, seedEngineKey(run.engine), seed)}>
        Fuer {runVoice.name} ({seedEngineLabel(run.engine)}) speichern
      </button>
    );
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
    const blob = await apiFetch<Blob>(`/api/admin/jobs/${jobId}/audio?format=mp3`, { adminKey, responseType: "blob" });
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

  async function handleLogout() {
    try {
      await logout();
    } catch {
      // clear locally regardless
    }
    clearStoredAdminKey();
    setAdminKey("");
    setCurrentUser("");
    setMustChange(false);
    setSnapshot(null);
  }

  if (bootLoading) {
    return (
      <main className="gate-shell">
        <section className="gate-card">
          <p className="widget-copy">Laedt...</p>
        </section>
      </main>
    );
  }

  if (!adminKey) {
    return (
      <main className="gate-shell">
        <section className="gate-card">
          <p className="eyebrow">Private Access</p>
          <h1>G3_OmniVoice Adminpanel</h1>
          <p className="widget-copy">
            Login mit Benutzername und Passwort. Standardzugang <code>admin</code> / <code>admin</code> &ndash; das
            Passwort muss beim ersten Login geaendert werden.
          </p>
          <label>
            Benutzername
            <input value={loginUsername} autoComplete="username" onChange={(event) => setLoginUsername(event.target.value)} placeholder="admin" />
          </label>
          <label>
            Passwort
            <input type="password" value={loginPassword} autoComplete="current-password" onChange={(event) => setLoginPassword(event.target.value)} placeholder="admin" />
          </label>
          <div className="button-row">
            <button className="primary-button" type="button" onClick={handleAuthenticate} disabled={authLoading || !loginUsername.trim() || !loginPassword}>
              {authLoading ? "Pruefe..." : "Anmelden"}
            </button>
          </div>
          {error ? <div className="message error">{error}</div> : null}
        </section>
      </main>
    );
  }

  if (mustChange) {
    return (
      <main className="gate-shell">
        <section className="gate-card">
          <p className="eyebrow">Security</p>
          <h1>Passwort aendern</h1>
          <p className="widget-copy">Angemeldet als {currentUser || "admin"}. Bitte ein neues Passwort setzen.</p>
          <label>
            Aktuelles Passwort
            <input type="password" value={pwCurrent} autoComplete="current-password" onChange={(event) => setPwCurrent(event.target.value)} />
          </label>
          <label>
            Neues Passwort
            <input type="password" value={pwNew} autoComplete="new-password" onChange={(event) => setPwNew(event.target.value)} />
          </label>
          <label>
            Neues Passwort bestaetigen
            <input type="password" value={pwConfirm} autoComplete="new-password" onChange={(event) => setPwConfirm(event.target.value)} />
          </label>
          <div className="button-row">
            <button className="primary-button" type="button" onClick={handleChangePassword} disabled={!pwCurrent || !pwNew || !pwConfirm}>
              Speichern
            </button>
            <button className="ghost-button" type="button" onClick={handleLogout}>Abmelden</button>
          </div>
          {error ? <div className="message error">{error}</div> : null}
        </section>
      </main>
    );
  }

  if (!snapshot || !settingsDraft) {
    return (
      <main className="gate-shell">
        <section className="gate-card">
          <p className="widget-copy">Dashboard laedt...</p>
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
          <button className="ghost-button" type="button" onClick={handleLogout}>Logout</button>
        </div>
      </section>

      <section className="hero-card">
        <div className="hero-copy" style={{ display: "grid", gap: "24px", gridTemplateColumns: "1fr 1.4fr" }}>
          <div style={{ display: "grid", gap: "10px", alignContent: "start" }}>
            <p className="eyebrow">Account</p>
            <h2>Admin-Zugang</h2>
            <p className="widget-copy">Angemeldet als <strong>{currentUser || "admin"}</strong>.</p>
            <label>Aktuelles Passwort<input type="password" value={pwCurrent} onChange={(event) => setPwCurrent(event.target.value)} /></label>
            <label>Neues Passwort<input type="password" value={pwNew} onChange={(event) => setPwNew(event.target.value)} /></label>
            <label>Bestaetigen<input type="password" value={pwConfirm} onChange={(event) => setPwConfirm(event.target.value)} /></label>
            <div className="button-row">
              <button className="secondary-button" type="button" onClick={handleChangePassword} disabled={!pwCurrent || !pwNew || !pwConfirm}>Passwort aendern</button>
            </div>
          </div>
          <div style={{ display: "grid", gap: "10px", alignContent: "start" }}>
            <p className="eyebrow">API Keys</p>
            <h2>Public API</h2>
            <p className="widget-copy">Ohne Key ist die oeffentliche TTS-API offen. Sobald ein Key existiert, ist <code>X-API-Key</code> Pflicht. Erzeugte Sekunden werden pro Key gezaehlt.</p>
            {createdKey ? (
              <div className="message success" style={{ wordBreak: "break-all" }}>
                <strong>{createdKey.alias}</strong> &ndash; jetzt kopieren (nur einmal sichtbar): <code>{createdKey.token}</code>
              </div>
            ) : null}
            <div className="button-row">
              <input value={newKeyAlias} onChange={(event) => setNewKeyAlias(event.target.value)} placeholder="Alias, z.B. Key fuer Projekt X" />
              <button className="secondary-button" type="button" onClick={handleCreateApiKey} disabled={!newKeyAlias.trim()}>Key erstellen</button>
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr><th>Alias</th><th>Erstellt</th><th>Audio (s)</th><th>Requests</th><th>Zuletzt</th><th></th></tr>
                </thead>
                <tbody>
                  {apiKeys.length === 0 ? (
                    <tr><td colSpan={6}>Keine API-Keys &ndash; oeffentliche API ist offen.</td></tr>
                  ) : null}
                  {apiKeys.map((key) => (
                    <tr key={key.id}>
                      <td>{key.alias}</td>
                      <td>{formatDate(key.created_at)}</td>
                      <td>{key.usage.total_seconds_processed.toFixed(1)}</td>
                      <td>{key.usage.request_count}</td>
                      <td>{formatDate(key.usage.last_used_at)}</td>
                      <td><button className="ghost-button" type="button" onClick={() => handleDeleteApiKey(key.id, key.alias)}>Loeschen</button></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </section>

      <nav className="ft-tab-rail">
        <button type="button" className={`ft-tab ${adminTab === "overview" ? "active" : ""}`} onClick={() => setAdminTab("overview")}>Overview</button>
        {finetuneSupported ? (
          <>
            <button type="button" className={`ft-tab ${adminTab === "finetune_data" ? "active" : ""}`} onClick={() => setAdminTab("finetune_data")}>Data Generation</button>
            <button type="button" className={`ft-tab ${adminTab === "finetune_train" ? "active" : ""}`} onClick={() => setAdminTab("finetune_train")}>Training</button>
          </>
        ) : null}
      </nav>

      {message ? <div className="message success">{message}</div> : null}
      {error ? <div className="message error">{error}</div> : null}

      {adminTab === "finetune_data" ? (
        <DataGenerationPanel adminKey={adminKey} voices={voices} onMessage={setMessage} onError={setError} />
      ) : null}

      {adminTab === "finetune_train" ? (
        <TrainingPanel adminKey={adminKey} onMessage={setMessage} onError={setError} />
      ) : null}

      {adminTab === "overview" ? (
      <section className="widget-grid">
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

        <section className="widget span-6"><div className="widget-header"><h2>Performance Graph</h2></div>
          <div className="field-grid two">
            <MiniGraph label="Durchsatz 30 s" values={overviewHistory.map((entry) => Number(entry.realtime_x_live || 0))} color="#90f0b7" suffix="x" />
            <MiniGraph label="TTFA 30 s" values={overviewHistory.map((entry) => Number(entry.ttfa_ms_live || 0))} color="#ff9a46" suffix=" ms" />
            <MiniGraph label="Queue" values={overviewHistory.map((entry) => Number(entry.queue_depth || 0))} color="#86c9ff" />
            <MiniGraph label="GPU" values={overviewHistory.map((entry) => Number(entry.gpu_utilization_pct || 0))} color="#ff6e6e" suffix="%" />
          </div>
        </section>

        <section className="widget span-12">
          {engineList ? (
            <ModelPicker
              list={engineList}
              target={engineTarget || engineList.active}
              switchRunning={engineSwitchRunning}
              downloadingId={modelDownloadAction?.kind === "download" ? modelDownloadAction.modelId : null}
              onSelect={setEngineTarget}
              onLoad={(engineId) => void handleEngineSwitch(engineId)}
              onDownload={(engine: EngineInfo) => void startModelDownload(engine.id, { engine: engine.id })}
              onRefresh={() => void loadEngines(adminKey)}
            />
          ) : (
            <>
              <div className="widget-header"><h2>1. Modell</h2></div>
              <p>Engine-Status wird geladen...</p>
            </>
          )}
          {activeEngine ? (
            <fieldset className="switch-lock" disabled={engineSwitchRunning}>
              <ActiveModelPanel
                engine={activeEngine}
                snapshot={snapshot}
                selectedModel={settingsDraft.default_model}
                download={modelDownloads.find((download) => download.id === settingsDraft.default_model)}
                deviceLabel={selectedRuntimeDeviceLabel}
                busy={modelOpsBusy || modelDownloadAction?.kind === "delete"}
                memoryBusy={memoryCleanupBusy}
                cacheRefreshing={modelDownloadAction?.kind === "refresh"}
                downloadingId={modelDownloadAction?.kind === "download" ? modelDownloadAction.modelId : null}
                onPickModel={patchDefaultModel}
                onSetDefault={(modelId) => void handleSetOpsModelAsDefault(modelId)}
                onOperation={(kind) => void handleModelOperation(kind)}
                onDownload={(modelId) => void startModelDownload(modelId, { model_id: modelId })}
                onDelete={(modelId) => void handleModelDelete(modelId)}
                onRefreshCache={() => void handleModelDownloadRefresh()}
                onFreeMemory={() => void handleFreeMemory()}
              />
            </fieldset>
          ) : null}
        </section>

        <section className="widget span-12">
          <div className="widget-header"><h2>2. Presets{activeEngine ? ` fuer ${activeEngine.label}` : ""}</h2></div>
          <fieldset className="switch-lock" disabled={engineSwitchRunning}>
            <div className="button-row" style={{ flexWrap: "wrap", gap: "8px", alignItems: "center" }}>
              <select
                value={selectedPreset ? selectedPreset.name : ""}
                onChange={(event) => setPresetName(event.target.value)}
                title={selectedPreset?.description || "Preset auswaehlen"}
                style={{ maxWidth: "280px" }}
              >
                <option value="">{enginePresets.length ? "— Preset waehlen —" : "— Keine Presets fuer dieses Modell —"}</option>
                {enginePresets.map((preset) => <option key={preset.name} value={preset.name} title={preset.description}>{preset.name}{preset.built_in ? " (eingebaut)" : ""}</option>)}
              </select>
              <button className="link-chip" type="button" onClick={() => selectedPreset && handleApplyPreset(selectedPreset.name)} disabled={!selectedPreset} title="Werte in den Entwurf uebernehmen; angewendet wird erst mit Settings speichern">In Entwurf laden</button>
              <input
                value={presetName}
                onChange={(event) => setPresetName(event.target.value)}
                placeholder="Preset-Name"
                style={{ maxWidth: "170px" }}
              />
              <button className="link-chip" type="button" onClick={() => void handleSavePreset()} disabled={!presetName.trim() || Boolean(selectedPreset?.built_in)} title={selectedPreset?.built_in ? "Eingebaute Presets sind schreibgeschuetzt" : "Werte dieses Modells aus dem Entwurf als Preset speichern"}>Als Preset speichern</button>
              <button className="link-chip" type="button" onClick={() => void handleDeletePreset()} disabled={!selectedPreset || Boolean(selectedPreset.built_in)} title={selectedPreset?.built_in ? "Eingebaute Presets koennen nicht geloescht werden" : "Gespeichertes Preset loeschen"}>Preset loeschen</button>
              {selectedPreset?.built_in ? <span className="pill" title={selectedPreset.description}>eingebaut</span> : null}
              {selectedPreset?.note ? <span className="pill" title={selectedPreset.note}>keine passende GPU</span> : null}
            </div>
            {selectedPreset?.description || selectedPreset?.note ? (
              <p className="muted" style={{ margin: "4px 0 0" }}>{[selectedPreset.description, selectedPreset.note].filter(Boolean).join(" ")}</p>
            ) : null}
          </fieldset>
        </section>

        <section className="widget span-12">
          <div className="widget-header"><h2>3. Einstellungen{activeEngine ? ` fuer ${activeEngine.label}` : ""}</h2></div>
          <fieldset className="switch-lock" disabled={engineSwitchRunning}>
            {activeEngine ? (
              <SettingsFields fields={activeEngine.settings_fields} draft={settingsDraft} context={settingsFieldContext} onPatch={patchDraft} />
            ) : (
              <p>Engine-Status wird geladen...</p>
            )}
            {activeEngine?.settings_fields.includes("preferred_device") ? (
              <div className="inline-pills">
                <button className="link-chip" type="button" title={ADMIN_HELP.refreshRuntimeDevices} onClick={() => void loadRuntimeDevices(adminKey)}>GPU-Liste aktualisieren</button>
                <span className={`pill ${runtimeDeviceOptions.some((device) => device.id === settingsDraft.preferred_device && device.available) ? "active" : ""}`}>{selectedRuntimeDeviceLabel}</span>
              </div>
            ) : null}
            <div className="button-row"><button className="primary-button" type="button" onClick={handleSaveSettings}>Settings speichern</button></div>
          </fieldset>
        </section>

        <section className="widget span-12">
          <div className="widget-header"><h2>Serverweite Einstellungen</h2></div>
          <fieldset className="switch-lock" disabled={engineSwitchRunning}>
            <SettingsFields fields={engineList?.server_settings_fields ?? []} draft={settingsDraft} context={settingsFieldContext} onPatch={patchDraft} />
            <div className="inline-pills">
              <button className="link-chip" type="button" title={ADMIN_HELP.refreshVllmModels} onClick={() => void loadVllmModels(adminKey, settingsDraft.vllm_base_url)} disabled={vllmModelsLoading}>{vllmModelsLoading ? "vLLM laedt..." : "vLLM Modelle aktualisieren"}</button>
              <span className={`pill ${vllmModels.length ? "active" : ""}`}>{vllmModels.length ? `${vllmModels.length} vLLM Modelle` : "keine vLLM Modelle"}</span>
              {vllmModelsError ? <span className="pill">{vllmModelsError}</span> : null}
            </div>
            <div className="button-row"><button className="primary-button" type="button" onClick={handleSaveSettings}>Settings speichern</button></div>
          </fieldset>
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
            <label className="with-help" title={ADMIN_HELP.werVoice}>Stimme<select value={werVoice} onChange={(event) => setWerVoice(event.target.value)}>
              <option value="">Standard-Stimme aus den Einstellungen</option>
              {customVoices.map((voice) => {
                const seed = voiceEngineSeed(voice, activeSeedEngine);
                return <option key={voice.voice_id} value={voice.voice_id}>{voice.name}{seed != null ? ` (bester ${seedEngineLabel(activeSeedEngine)}-Seed ${seed})` : ` (ohne ${seedEngineLabel(activeSeedEngine)}-Seed)`}</option>;
              })}
            </select></label>
            <label className="with-help" title={ADMIN_HELP.werSeed}>Seed<input type="text" value={werSeed} onChange={(event) => setWerSeed(event.target.value)} placeholder="leer = bester Seed der Stimme, 8888 oder 8888,8890,2912" /></label>
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
                  {run.fallback_sentence_count ? <span className="pill">{run.fallback_sentence_count} Vorlagen-Saetze (LLM lieferte zu wenige)</span> : null}
                  <span className="pill">Stimme: {run.voice ? voices.find((voice) => voice.voice_id === run.voice)?.name ?? run.voice : `Standard${werRunVoice(run) ? ` (${werRunVoice(run)?.name})` : ""}`}</span>
                  <span className="pill">Engine: {seedEngineLabel(run.engine)}</span>
                </div>
                {isRunActive(run.status) ? (
                  <div className="button-row">
                    <button className="ghost-button danger-button" type="button" disabled={werCancelling === run.run_id} onClick={() => void handleCancelWerBenchmark(run.run_id)} title={ADMIN_HELP.werCancel}>
                      {werCancelling === run.run_id ? "Breche ab..." : "WER-Benchmark abbrechen"}
                    </button>
                  </div>
                ) : null}
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
                          <th>Bester Seed</th>
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
                            <td>{renderSaveSeedCell(run, item.seed)}</td>
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
                  {deviationsFor(run.run_id).items.length < deviationsFor(run.run_id).total ? (
                    <div className="button-row">
                      <button className="secondary-button" type="button" onClick={() => void loadWerDeviations(run.run_id, true)}>
                        Weitere laden ({deviationsFor(run.run_id).items.length}/{deviationsFor(run.run_id).total} Abweichungen, schlechteste zuerst)
                      </button>
                    </div>
                  ) : null}
                  {werDeviations?.runId === run.run_id && werDeviations.items.length === 0 ? (
                    <p className="widget-copy">Keine WER-Abweichungen im aktuellen Run.</p>
                  ) : null}
                  {deviationsFor(run.run_id).items.map((item) => (
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
                          MP3
                        </button>
                      </div>
                      {item.job_id && jobAudioUrls[item.job_id] ? (
                        <>
                          <audio controls src={jobAudioUrls[item.job_id]} />
                          <div className="button-row compact">
                            <a className="secondary-button" href={jobAudioUrls[item.job_id]} download={`${item.job_id}.mp3`}>MP3 herunterladen</a>
                          </div>
                        </>
                      ) : null}
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
            {selectableModels.length > 1 ? (
              <label className="with-help" title={ADMIN_HELP.quickModel}>Modell<select value={quickModel} onChange={(event) => setQuickModel(event.target.value)}>{selectableModels.map((model) => <option key={model} value={model}>{model}</option>)}</select></label>
            ) : null}
            <label className="with-help" title={ADMIN_HELP.quickVoice}>Stimme<select value={quickVoice} onChange={(event) => handleQuickVoiceChange(event.target.value)}>{!quickVoices.length ? <option value="">Keine passende Stimme</option> : null}{quickVoices.map((voice) => <option key={voice.voice_id} value={voice.source === "custom" ? voice.voice_id : voice.name}>{voice.name} ({voice.source})</option>)}</select></label>
            <label className="with-help" title={ADMIN_HELP.quickLanguage}>Sprache<select value={quickLanguage} onChange={(event) => setQuickLanguage(event.target.value)}>{SYNTH_LANGUAGE_OPTIONS.map((option) => <option key={option} value={option}>{option}</option>)}</select></label>
            <label className="with-help" title={ADMIN_HELP.quickSeed}>Seed<input type="number" min="0" max="2147483647" value={quickSeed} onChange={(event) => setQuickSeed(event.target.value)} placeholder="leer = zufaellig" /></label>
          </div>
          <label className="with-help" title={ADMIN_HELP.quickText}>Text<textarea ref={quickTextRef} value={quickText} onChange={(event) => setQuickText(event.target.value)} /></label>
          {mingEngineActive ? (
            <div className="tag-insert" title={ADMIN_HELP.mingTags}>
              <div className="tag-insert-row">
                <span className="tag-insert-label">Geraeusch einfuegen</span>
                {MING_TAGS.filter((item) => item.common).map((item) => (
                  <button key={item.tag} className="secondary-button" type="button" title={item.tag} onClick={() => insertQuickTag(item.tag)}>{item.label}</button>
                ))}
              </div>
              <details className="tag-insert-more">
                <summary>mehr Laute</summary>
                <div className="tag-insert-row">
                  {MING_TAGS.filter((item) => !item.common).map((item) => (
                    <button key={item.tag} className="secondary-button" type="button" title={item.tag} onClick={() => insertQuickTag(item.tag)}>{item.label}</button>
                  ))}
                </div>
              </details>
            </div>
          ) : null}
          {mingEngineActive ? (
            <>
              <fieldset className="ming-delivery">
                <legend>Ming-Ausdruck</legend>
                <p className="ming-delivery-hint">Emotion und Tempo wirken am staerksten, Tonhoehe und Lautstaerke kaum. Mit Emotion nutzt &quot;auto&quot; nur das Sprecher-Embedding: mehr Ausdruck, die Stimme klingt aber etwas weniger nach der Referenz.</p>
                <div className="ming-delivery-grid">
                <label className="with-help" title={ADMIN_HELP.mingEmotion}>Emotion
                  <select value={quickMingDelivery.emotion} onChange={(event) => patchQuickMingDelivery({ emotion: event.target.value as MingDeliveryDraft["emotion"] })}>
                    <option value="">keine</option>
                    {MING_EMOTION_OPTIONS.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
                  </select>
                </label>
                <label className="with-help" title={ADMIN_HELP.mingRate}>Tempo
                  <select value={quickMingDelivery.speaking_rate} onChange={(event) => patchQuickMingDelivery({ speaking_rate: event.target.value as MingDeliveryDraft["speaking_rate"] })}>
                    <option value="">Standard</option>
                    {MING_RATE_OPTIONS.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
                  </select>
                </label>
                <label className="with-help" title={ADMIN_HELP.mingPitch}>Tonhoehe
                  <select value={quickMingDelivery.pitch} onChange={(event) => patchQuickMingDelivery({ pitch: event.target.value as MingDeliveryDraft["pitch"] })}>
                    <option value="">Standard</option>
                    {MING_PITCH_OPTIONS.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
                  </select>
                </label>
                <label className="with-help" title={ADMIN_HELP.mingVolume}>Lautstaerke
                  <select value={quickMingDelivery.volume} onChange={(event) => patchQuickMingDelivery({ volume: event.target.value as MingDeliveryDraft["volume"] })}>
                    <option value="">Standard</option>
                    {MING_VOLUME_OPTIONS.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
                  </select>
                </label>
                <label className="with-help" title={ADMIN_HELP.mingReferenceMode}>Referenzmodus
                  <select value={quickMingDelivery.reference_mode === "auto" ? "" : quickMingDelivery.reference_mode} onChange={(event) => patchQuickMingDelivery({ reference_mode: event.target.value as MingDeliveryDraft["reference_mode"] })}>
                    <option value="">auto ({resolvedReferenceMode(quickMingDelivery) === "speaker" ? "nur Sprecher" : "voll"})</option>
                    {REFERENCE_MODE_OPTIONS.filter((option) => option.value !== "auto").map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
                  </select>
                </label>
                </div>
              <label className="with-help" title={ADMIN_HELP.quickMingInstruction}>Freie Anweisung (statt Emotion/Tempo)
                <textarea
                  value={quickInstructions}
                  onChange={(event) => setQuickInstructions(event.target.value)}
                  disabled={quickMingStructured}
                  placeholder={quickMingStructured ? "Aus, solange Emotion/Tempo/Tonhoehe/Lautstaerke gesetzt ist." : "z.B. spricht ruhig und warm, leicht laechelnd"}
                />
              </label>
              {quickMingStructured && quickInstructions.trim() ? (
                <div className="message">Die freie Anweisung wird nicht gesendet, solange Emotion, Tempo, Tonhoehe oder Lautstaerke gesetzt ist.</div>
              ) : null}
              {hasStructuredMingControl(quickMingDelivery) || quickMingDelivery.reference_mode || quickInstructions ? (
                <div className="button-row">
                  <button className="secondary-button" type="button" onClick={() => setQuickMingDelivery(EMPTY_MING_DELIVERY)} disabled={!hasStructuredMingControl(quickMingDelivery) && !quickMingDelivery.reference_mode}>Ming-Steuerung zuruecksetzen</button>
                  <button className="secondary-button" type="button" onClick={() => setQuickInstructions("")} disabled={!quickInstructions}>Freie Anweisung leeren</button>
                </div>
              ) : null}
              </fieldset>
            </>
          ) : (
            <label className="with-help" title={ADMIN_HELP.quickInstructions}>Instructions<textarea value={quickInstructions} onChange={(event) => setQuickInstructions(event.target.value)} /></label>
          )}
          <label className="with-help" title={ADMIN_HELP.quickTextOptimization}>Textoptimierung<select value={quickTextOptimization} onChange={(event) => setQuickTextOptimization(event.target.value as "" | TextOptimizationMode)}>
            <option value="">Server-Default ({snapshot?.settings.text_optimization_default ?? "off"})</option>
            <option value="off">off</option>
            <option value="deterministic">deterministic</option>
          </select></label>
          <div className="button-row">
            <button className="primary-button" type="button" onClick={handleQuickRun}>{quickRunning ? "Stoppen" : "Stream starten"}</button>
            <button className="secondary-button" type="button" title={ADMIN_HELP.quickTextOptimization} onClick={() => void handleQuickOptimizePreview()}>Text-Vorschau</button>
          </div>
          {quickNote ? <div className={`message ${quickNote.error ? "error" : "success"}`} role="status">{quickNote.text}</div> : null}
          {quickOptimizePreview ? (
            <div className="metric-list" title="Gesprochener Text nach der Textoptimierung">
              <div className="metric-row"><span>Sprache</span><strong>{quickOptimizePreview.language ?? "keine"} ({quickOptimizePreview.binding_origin})</strong></div>
              <div className="metric-row"><span>Regeln</span><strong>{quickOptimizePreview.applied_rules.length ? quickOptimizePreview.applied_rules.join(", ") : "keine"}</strong></div>
              <p style={{ whiteSpace: "pre-wrap", margin: "6px 0 0" }}>{quickOptimizePreview.text}</p>
            </div>
          ) : null}
          {quickAudioUrl ? (
            <>
              <audio controls src={quickAudioUrl} />
              <div className="button-row">
                <a className="secondary-button" href={quickAudioUrl} download={quickAudioDownloadName}>
                  {quickAudioDownloadName.endsWith(".mp3") ? "MP3 herunterladen" : "WAV herunterladen"}
                </a>
              </div>
            </>
          ) : null}
          {quickMetrics ? <div className="metric-list"><div className="metric-row"><span>TTFA</span><strong>{formatMs(quickMetrics.ttfa_ms)}</strong></div><div className="metric-row"><span>Realtime</span><strong>{formatRealtime(quickMetrics.realtime_x)}</strong></div><div className="metric-row"><span>Duration</span><strong>{formatSeconds((quickMetrics.audio_duration_ms || 0) / 1000)}</strong></div></div> : null}
        </section>

        <section className="widget span-6"><div className="widget-header"><h2>Voice Library</h2></div>
          {activeEngine?.uses_voice_profiles ? (
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
          ) : null}
          <div className="voice-list">
            {libraryVoices.length === 0 ? <p className="widget-copy">{activeEngine && !activeEngine.uses_voice_profiles ? "Dieses Modell nutzt keine Stimmprofile." : "Noch keine Custom Voices gespeichert."}</p> : null}
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
                <div className="voice-seed-list">
                  {SEED_ENGINES.filter((engine) => engine.id === activeSeedEngine).map((engine) => {
                    const editKey = `${voice.voice_id}:${engine.id}`;
                    return (
                      <div key={engine.id} className="voice-seed-row">
                        <label className="with-help" title={ADMIN_HELP.voiceSeed}>Bester Seed {engine.label}
                          <input
                            type="number"
                            min="0"
                            max="2147483647"
                            placeholder="kein Seed"
                            value={voiceSeedEdits[editKey] ?? (voiceEngineSeed(voice, engine.id) ?? "")}
                            onChange={(event) => setVoiceSeedEdits((current) => ({ ...current, [editKey]: event.target.value }))}
                          />
                        </label>
                        <button className="secondary-button" type="button" onClick={() => void handleSaveVoiceSeed(voice, engine.id)}>Seed speichern</button>
                      </div>
                    );
                  })}
                </div>
                {voiceAudioUrls[voice.voice_id] ? <audio controls src={voiceAudioUrls[voice.voice_id]} /> : null}
              </article>
            ))}
          </div>
        </section>

        <section className="widget span-12"><div className="widget-header"><h2>History</h2></div>
          <div className="job-list">{snapshot.jobs.map((job) => <article key={job.job_id} className="job-card"><strong>{job.input_preview}</strong><div className="inline-pills"><span className={`pill ${job.status === "completed" ? "active" : ""}`}>{job.status}</span><span className="pill">{job.model || "-"}</span><span className="pill">{job.voice || "-"}</span><span className="pill">{formatMs(job.metrics.ttfa_ms)}</span></div><div className="button-row">{job.status === "completed" ? <button className="secondary-button" type="button" onClick={() => void handleLoadJobAudio(job.job_id)}>MP3</button> : null}<button className="ghost-button" type="button" onClick={() => void handleDeleteJob(job.job_id)}>{job.status === "completed" ? "Entfernen" : "Stornieren"}</button></div>{jobAudioUrls[job.job_id] ? <><audio controls src={jobAudioUrls[job.job_id]} /><div className="button-row"><a className="secondary-button" href={jobAudioUrls[job.job_id]} download={`${job.job_id}.mp3`}>MP3 herunterladen</a></div></> : null}</article>)}</div>
        </section>
      </section>
      ) : null}
    </main>
  );
}
