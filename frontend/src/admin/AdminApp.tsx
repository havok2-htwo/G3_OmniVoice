import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { FormEvent, KeyboardEvent as ReactKeyboardEvent } from "react";

import {
  apiFetch,
  authSetup,
  changePassword,
  clearStoredAdminKey,
  createApiKey,
  createWavBlobFromInt16Chunks,
  decodePcm16Base64,
  deleteApiKey,
  formatDate,
  formatMs,
  formatNumber,
  formatRealtime,
  formatSeconds,
  listApiKeys,
  login,
  logout,
  setUnauthorizedHandler,
  streamNdjson,
  streamSse,
  whoami,
  type ApiKeyInfo,
  type BenchmarkRunResponse,
  type DashboardSnapshot,
  type EngineCapabilities,
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
import { ApiKeysPanel, type ApiKeyRow } from "../ApiKeysPanel";
import { PerfPanel, clearPerfHistory, gpuTitle, usePerfFeed, usePerfLatest, type PerfSnapshot } from "../PerfPanel";
import { ToastHost, toast } from "../Toasts";

const TABS = [
  { id: "dashboard", label: "Dashboard" },
  { id: "playground", label: "Playground" },
  { id: "voices", label: "Voices" },
  { id: "models", label: "Models" },
  { id: "settings", label: "Settings" },
  { id: "benchmark", label: "Benchmark" },
  { id: "data-generation", label: "Data generation" },
  { id: "training", label: "Training" },
  { id: "history", label: "History" },
  { id: "api-keys", label: "API keys" },
  { id: "account", label: "Account" },
] as const;
type AdminTab = (typeof TABS)[number]["id"];
const FINETUNE_TABS: AdminTab[] = ["data-generation", "training"];

function tabFromHash(): AdminTab {
  const id = window.location.hash.replace(/^#/, "");
  return (TABS.find((tab) => tab.id === id)?.id ?? "dashboard") as AdminTab;
}

// Stable module-level notifiers (children keep them in useCallback deps without re-running effects).
function setMessage(text: string) {
  if (text) toast.ok(text);
}
function setError(text: string) {
  if (text) toast.error(text);
}
function errorText(error: unknown, fallback: string) {
  return error instanceof Error && error.message ? error.message : fallback;
}

const SETTINGS_NOT_COMPARED = new Set(["runtime_backend", "runtime_backend_pending", "attention_backend", "attention_backend_note"]);
function settingsChanges(draft: ServerSettings | null, saved: ServerSettings | null): string[] {
  if (!draft || !saved) return [];
  const keys = new Set([...Object.keys(draft), ...Object.keys(saved)]);
  return [...keys].filter((key) => {
    if (SETTINGS_NOT_COMPARED.has(key)) return false;
    const a = (draft as unknown as Record<string, unknown>)[key];
    const b = (saved as unknown as Record<string, unknown>)[key];
    return JSON.stringify(a ?? null) !== JSON.stringify(b ?? null);
  });
}

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
      return { randomSeed: null, seedValues: null, error: "The seed must be a whole number." };
    }
    const seed = Number(parts[0]);
    if (!Number.isSafeInteger(seed) || seed > MAX_TTS_SEED) {
      return { randomSeed: null, seedValues: null, error: `The seed must be between 0 and ${MAX_TTS_SEED}.` };
    }
    return { randomSeed: seed, seedValues: null };
  }
  const seeds: number[] = [];
  const seen = new Set<number>();
  for (const part of parts) {
    if (!part || !/^\d+$/.test(part)) {
      return { randomSeed: null, seedValues: null, error: "Enter the seed list as whole numbers separated by commas, for example 8888,8890,2912." };
    }
    const seed = Number(part);
    if (!Number.isSafeInteger(seed) || seed > MAX_TTS_SEED) {
      return { randomSeed: null, seedValues: null, error: `Every seed must be between 0 and ${MAX_TTS_SEED}.` };
    }
    if (!seen.has(seed)) {
      seeds.push(seed);
      seen.add(seed);
    }
  }
  if (seeds.length > 1025) {
    return { randomSeed: null, seedValues: null, error: "The seed list may hold at most 1025 values." };
  }
  return { randomSeed: seeds[0] ?? null, seedValues: seeds };
}

const ADMIN_HELP = {
  quickModel: "Model for this test synthesis.",
  quickVoice: "Voice for the test. Custom voices run on the Base model.",
  quickText: "Text for the streaming test. The first sentence takes a fast path; later compatible sentences are batched again.",
  quickLanguage: "Language hint for the engine. Auto lets the model detect it; a fixed choice can improve pronunciation slightly.",
  quickLanguageTextOnly: "This engine has no language parameter: the language only affects the text optimization.",
  quickInstructions: "Optional instruction sent with the request to the engine.",
  quickMingInstruction: "Ming: free instruction for the delivery (for example \"speaks calmly and warmly, slightly smiling\"). It cannot be combined with emotion, rate, pitch or volume; while one of them is set it is not sent.",
  mingEmotion: "Ming: emotion of the output. Sent together with rate, pitch and volume as one structured instruction.",
  mingRate: "Ming: speaking rate. When one of rate, pitch or volume is set, the server fills the others with 'medium'.",
  mingPitch: "Ming: pitch of the output.",
  mingVolume: "Ming: volume of the output.",
  mingReferenceMode: "Ming: how the clone voice is used. full = reference audio + transcript + speaker embedding (normal cloning). speaker only = only the speaker embedding, so the reference's delivery does not override the instruction. auto = speaker only as soon as emotion, rate, pitch or volume is set, otherwise full.",
  mingTags: "Ming: inserts a sound tag at the cursor. Tags are not read aloud; an engine without the tag removes it.",
  quickSeed: "Seed for more reproducible runs. Leave empty for random output.",
  quickTextOptimization: "Text optimization for this run: dates, times, numbers, units, URLs and Markdown are written out for German and English. Default = server setting. Preview shows the spoken text without synthesis (with 'off' the original text). With language Auto, German or English is detected from the spelling (24.09.2026, 14:30 Uhr, 3:30 PM) or typical words; other languages stay unchanged.",
  refreshVllmModels: "Reads the model list from the configured vLLM server again.",
  refreshRuntimeDevices: "Reads the CPU and CUDA devices again.",
  voiceName: "Name of the saved clone voice in the voice catalog.",
  voiceRefText: "Exact text spoken in the sample. The clone models need it for the voice prompt.",
  voiceSample: "Audio sample of the voice. The server converts it to 24 kHz WAV.",
  transcribeSample: "Sends the sample to the configured G3 Whisper server and fills in the reference text.",
  voiceConsent: "Confirms that this voice may be stored and used for local tests.",
  benchmarkMode: "Traffic simulates real users with random arrival times. Fixed iterations is the classic parallel test.",
  benchmarkText: "Sentence pool for traffic mode. Each request picks several random sentences from it.",
  benchmarkDuration: "Traffic window in which new requests arrive at random. Open generations may finish for up to 180 s afterwards.",
  benchmarkRpm: "Target rate in traffic mode: user requests per minute.",
  benchmarkSentenceRange: "Random number of sentences per user request, for example 3, 5 or 10.",
  benchmarkSeed: "Optional seed for a reproducible request schedule and sentence choice.",
  benchmarkIterations: "Measured repetitions per case in iteration mode. Warmups are not averaged.",
  benchmarkWarmups: "Unmeasured runs that fill model, CUDA and prompt caches first.",
  benchmarkParallel: "Iteration mode: jobs started at the same time. Traffic mode ignores it.",
  werCount: "Number of random sentences vLLM writes and the active engine then speaks.",
  werLanguage: "Language of the reference sentences from vLLM. TTS still runs with Auto language.",
  werWords: "Word range of the vLLM sentences. Shorter sentences make errors easier to see.",
  werTolerance: "Levenshtein tolerance per word. Up to this many letters of difference count as spelling; insertions and deletions stay errors.",
  werTimeout: "Maximum wait per TTS + Whisper sample. Stops single requests that hang.",
  werSeed: "TTS seed of the WER run: one value or a comma-separated list such as 8888,8890,2912. With a list the seed range is ignored.",
  werSeedRange: "Above 0 the benchmark tests every seed from Seed to Seed + range. Ignored when Seed is a list.",
  werVoice: "Voice the WER benchmark uses. Default = the voice from the settings. Only your own clone voices can be chosen.",
  werCancel: "Stops the running WER benchmark: running and waiting TTS jobs are cancelled and the run ends as cancelled. Finished results stay visible.",
  voiceSeed: "Best seed of this voice for the active engine (from the WER benchmark). Requests without their own seed use it; a seed only fits the engine it was measured on. Empty = no seed (random).",
  werSaveSeed: "Saves this seed as the best seed of the run's voice for the engine the run used.",
  werPrompt: "Optional custom vLLM prompt. It should produce a JSON array of sentences.",
};

const SEED_ENGINES: Array<{ id: SeedEngine; label: string }> = [
  { id: "omnivoice", label: "OmniVoice" },
  { id: "ming", label: "Ming" },
  { id: "phonon", label: "Phonon" },
  { id: "audio8_realistic", label: "Audio8 Realistic 0.6B" },
  { id: "audio8_preview", label: "Audio8 Preview 0.6B" },
  { id: "audio8_mini", label: "Audio8 Preview 0.1B" },
  { id: "audio8_onnx", label: "Audio8 0.1B ONNX INT8" },
];

/** The seed slot an engine reads; mock (tests) renders like OmniVoice. */
function seedEngineKey(engine?: string | null): SeedEngine {
  return SEED_ENGINES.find((item) => item.id === engine)?.id ?? "omnivoice";
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

function formatPercent(value?: number | null) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return `${(Number(value) * 100).toFixed(Number(value) >= 0.1 ? 1 : 2)}%`;
}

function fallbackRuntimeDevice(deviceId: string): RuntimeDeviceInfo {
  return {
    id: deviceId,
    label: `${deviceId} (configured)`,
    kind: deviceId.startsWith("cuda") ? "cuda" : "other",
    name: "Configured device",
    index: null,
    memory_total_mb: null,
    available: false,
    selected: true,
  };
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
  const [apiKeys, setApiKeys] = useState<ApiKeyInfo[] | null>(null);
  const [apiKeysError, setApiKeysError] = useState<string | null>(null);
  const [snapshot, setSnapshot] = useState<DashboardSnapshot | null>(null);
  const [snapshotError, setSnapshotError] = useState("");
  const [liveReconnecting, setLiveReconnecting] = useState(false);
  const [setupPending, setSetupPending] = useState(false);
  const [loginError, setLoginError] = useState("");
  const [pwError, setPwError] = useState("");
  const [pwBusy, setPwBusy] = useState(false);
  const [settingsSaving, setSettingsSaving] = useState(false);
  // The settings as the server last reported them: the base of the dirty check.
  const savedSettingsRef = useRef<ServerSettings | null>(null);
  const [savedSettings, setSavedSettings] = useState<ServerSettings | null>(null);
  const [settingsDraft, setSettingsDraft] = useState<ServerSettings | null>(null);
  const [presets, setPresets] = useState<SettingsPreset[]>([]);
  const [presetName, setPresetName] = useState("");
  const [presetSaveName, setPresetSaveName] = useState("");
  const [authLoading, setAuthLoading] = useState(false);
  const [quickModel, setQuickModel] = useState("");
  const [quickVoice, setQuickVoice] = useState("");
  // True once the admin picked another model: only then a settings save may send default_model.
  const [defaultModelPicked, setDefaultModelPicked] = useState(false);
  const [adminTab, setAdminTabState] = useState<AdminTab>(() => tabFromHash());
  const [quickText, setQuickText] = useState("Hallo! Das ist ein kurzer Test der Streaming-Ausgabe. Der zweite Satz kommt gleich hinterher.");
  const [quickLanguage, setQuickLanguage] = useState("Auto");
  const [quickInstructions, setQuickInstructions] = useState("");
  const [capabilities, setCapabilities] = useState<EngineCapabilities | null>(null);
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
  const [benchmarkText, setBenchmarkText] = useState("Satz eins für den Lasttest. Satz zwei ist etwas länger und simuliert einen echten Nutzer. Satz drei prüft die Queue. Satz vier testet Batching. Satz fünf ist kurz. Satz sechs ist wieder ein bisschen ausführlicher. Satz sieben nutzt dieselbe Stimme. Satz acht kommt zufällig rein. Satz neun macht den p99-Wert spannend. Satz zehn beendet den Pool.");
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
        try {
          const setup = await authSetup();
          if (!cancelled) setSetupPending(setup.setup_pending);
        } catch {
          // older server: no hint
        }
      } finally {
        if (!cancelled) setBootLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const loggedIn = adminKey === "session" && !mustChange;

  useEffect(() => {
    if (!loggedIn) return;
    void refreshApiKeys();
    void initialLoad();
  }, [loggedIn]);

  // Hash routing: #dashboard, #playground, ... (unknown -> dashboard).
  useEffect(() => {
    const onHash = () => setAdminTabState(tabFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  const setAdminTab = useCallback((tab: AdminTab) => {
    setAdminTabState(tab);
    if (window.location.hash !== `#${tab}`) window.history.replaceState(null, "", `#${tab}`);
  }, []);

  // Performance graph: one feed for the whole app, so the history keeps filling on every tab.
  const fetchPerf = useCallback((signal: AbortSignal) => apiFetch<PerfSnapshot>("/api/admin/perf", { signal }), []);
  usePerfFeed(fetchPerf, loggedIn);
  const perfLatest = usePerfLatest();

  // Clear everything that belongs to the session (perf history, toasts, created key via unmount) and show the login.
  const resetSession = useCallback((expired: boolean) => {
    clearStoredAdminKey();
    clearPerfHistory();
    toast.clear();
    setAdminKey("");
    setCurrentUser("");
    setMustChange(false);
    setSnapshot(null);
    setSettingsDraft(null);
    savedSettingsRef.current = null;
    setSavedSettings(null);
    setApiKeys(null);
    setSnapshotError("");
    setLiveReconnecting(false);
    if (expired) toast.warn("Session expired. Please log in again.");
  }, []);

  useEffect(() => {
    if (adminKey !== "session") return;
    setUnauthorizedHandler(() => resetSession(true));
    return () => setUnauthorizedHandler(null);
  }, [adminKey, resetSession]);

  async function refreshApiKeys() {
    try {
      const result = await listApiKeys();
      setApiKeys(result.keys);
      setApiKeysError(null);
    } catch (keyError) {
      setApiKeysError(errorText(keyError, "Could not load the API keys."));
    }
  }

  async function initialLoad() {
    setSnapshotError("");
    try {
      await loadSnapshot(adminKey || "session", true);
    } catch (loadError) {
      setSnapshotError(errorText(loadError, "Could not load the dashboard."));
    }
  }

  /** Reads the snapshot. The settings draft is replaced only when it has no unsaved edits (or `forceDraft`). */
  async function loadSnapshot(key: string, forceDraft = false) {
    const data = await apiFetch<DashboardSnapshot>("/api/admin/snapshot", { adminKey: key });
    const fresh = applyOmniVoiceDefaults(data.settings);
    const previousSaved = savedSettingsRef.current;
    setSnapshot(data);
    setSettingsDraft((current) => {
      if (forceDraft || !current || settingsChanges(current, previousSaved).length === 0) return fresh;
      return current;
    });
    savedSettingsRef.current = fresh;
    setSavedSettings(fresh);
    if (forceDraft) setDefaultModelPicked(false);
    setQuickModel((current) => current || data.settings.default_model);
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
    setEngineSwitchBusy(true);
    try {
      const started = await apiFetch<EngineListResponse>("/api/admin/runtime/backend", {
        method: "POST",
        adminKey,
        body: { backend: target },
      });
      setEngineList(started);
      setMessage(`Switching the engine to ${target}.`);
    } catch (switchError) {
      setEngineSwitchBusy(false);
      setError(`Could not switch the engine: ${errorText(switchError, "unknown error")}`);
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
      await loadSnapshot(adminKey, true);
    } catch {
      // The stale-draft effect below retries.
    }
    setEngineSwitchBusy(false);
    void loadPresets(adminKey);
    // The "started" line must not stand next to the error.
    if (result.switch.state === "failed") {
      setError(`Could not switch the engine: ${result.switch.error || "unknown error"}`);
    } else if (result.switch.error) {
      setError(result.switch.error);
    } else {
      setMessage(result.switch.message || `Engine ${result.active} is active.`);
    }
  }

  async function loadVllmModels(key = adminKey, baseUrl = settingsDraft?.vllm_base_url || "") {
    if (!key || !baseUrl.trim()) return;
    setVllmModelsLoading(true);
    try {
      const query = `?base_url=${encodeURIComponent(baseUrl)}`;
      const payload = await apiFetch<VllmModelsResponse>(`/api/admin/vllm/models${query}`, { adminKey: key });
      // An empty vllm_model means "the first model" on the server too, so the draft stays untouched (not dirty).
      setVllmModels(payload.models || []);
      setVllmModelsError(payload.error || "");
    } catch (loadError) {
      setVllmModels([]);
      setVllmModelsError(errorText(loadError, "Could not load the vLLM models."));
    } finally {
      setVllmModelsLoading(false);
    }
  }

  // Live dashboard stream. It reconnects after a disconnect (backend restart, network) with a 1, 2, 5, 10 s backoff.
  useEffect(() => {
    if (!loggedIn) return;
    const controller = new AbortController();
    const delays = [1000, 2000, 5000, 10000];
    void (async () => {
      let attempt = 0;
      while (!controller.signal.aborted) {
        try {
          await streamSse("/api/admin/dashboard/stream", {
            adminKey,
            signal: controller.signal,
            onEvent: async (eventName, payload) => {
              if (eventName !== "dashboard.snapshot") return;
              attempt = 0;
              setLiveReconnecting(false);
              const next = payload as DashboardSnapshot;
              setSnapshot(next);
              setSnapshotError("");
              setSettingsDraft((current) => current ?? applyOmniVoiceDefaults(next.settings));
            },
          });
        } catch (streamError) {
          if (controller.signal.aborted) return;
          if ((streamError as { status?: number }).status === 401) {
            resetSession(true);
            return;
          }
        }
        if (controller.signal.aborted) return;
        setLiveReconnecting(true);
        const delay = delays[Math.min(attempt, delays.length - 1)];
        attempt += 1;
        await new Promise((resolve) => window.setTimeout(resolve, delay));
      }
    })();
    return () => controller.abort();
  }, [loggedIn, adminKey, resetSession]);

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
  // Quick Synthesis shows what the engine reads. While the capabilities are unknown (first load, older server) nothing hides.
  const quickCapabilities = capabilities?.engine === activeEngineId ? capabilities : null;
  const quickReadsSeed = quickCapabilities?.options.seed ?? true;
  // The OmniVoice flag means clone requests; its built-in voices read instructions.
  const quickReadsInstructions = (quickCapabilities?.options.instructions ?? true) || (activeEngineId === "omnivoice" && quickTaskType !== "Base");
  const quickOptimizationMode = quickTextOptimization || snapshot?.settings.text_optimization_default || "off";
  // With a single language the model takes no choice; the text optimization still reads one.
  const quickHasLanguageChoice = (quickCapabilities?.languages.length ?? 2) > 1;
  const quickReadsLanguage = quickHasLanguageChoice || quickOptimizationMode === "deterministic";
  const quickRequestLanguage = quickReadsLanguage ? quickLanguage : "Auto";
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
    if (activeEngineId && !finetuneSupported && FINETUNE_TABS.includes(adminTab)) setAdminTab("dashboard");
  }, [activeEngineId, finetuneSupported, adminTab, setAdminTab]);

  useEffect(() => {
    if (!adminKey || mustChange || !activeEngineId) return;
    let stale = false;
    apiFetch<EngineCapabilities>("/api/v1/capabilities")
      .then((payload) => {
        if (!stale) setCapabilities(payload);
      })
      .catch(() => {
        // Only hides fields: without it Quick Synthesis shows them all.
        if (!stale) setCapabilities(null);
      });
    return () => {
      stale = true;
    };
  }, [adminKey, mustChange, activeEngineId, snapshot?.settings.default_model]);

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

  async function handleAuthenticate(event?: FormEvent) {
    event?.preventDefault();
    setAuthLoading(true);
    setLoginError("");
    try {
      const me = await login(loginUsername.trim(), loginPassword);
      setCurrentUser(me.username);
      setMustChange(me.must_change_password);
      setLoginPassword("");
      setAdminKey("session");
    } catch (authError) {
      setLoginError(errorText(authError, "Login failed."));
    } finally {
      setAuthLoading(false);
    }
  }

  async function handleChangePassword(event?: FormEvent) {
    event?.preventDefault();
    setPwError("");
    if (pwNew.length < 8) {
      setPwError("The new password needs at least 8 characters.");
      return;
    }
    if (pwNew !== pwConfirm) {
      setPwError("The new passwords do not match.");
      return;
    }
    setPwBusy(true);
    try {
      await changePassword(pwCurrent, pwNew);
      setPwCurrent("");
      setPwNew("");
      setPwConfirm("");
      setMustChange(false);
      setSetupPending(false);
      setMessage("Password changed.");
    } catch (pwChangeError) {
      setPwError(errorText(pwChangeError, "Could not change the password."));
    } finally {
      setPwBusy(false);
    }
  }

  const apiKeyRows: ApiKeyRow[] | null = useMemo(
    () =>
      apiKeys?.map((key) => ({
        id: key.id,
        alias: key.alias,
        hint: key.hint ?? null,
        created_at: key.created_at,
        last_used_at: key.usage.last_used_at,
        requests: key.usage.request_count,
        usage: key.usage.total_seconds_processed,
      })) ?? null,
    [apiKeys],
  );

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
    if (quickOptimizationMode === "off") {
      // The run would speak the raw text: show that instead of a deterministic rewrite.
      setQuickOptimizePreview({ text: quickText.trim(), applied_rules: [], language: null, binding_origin: "text optimization off", sentences: [] });
      return;
    }
    try {
      const preview = await apiFetch<TextOptimizeResponse>("/api/admin/text/optimize", {
        method: "POST",
        adminKey,
        body: { input: quickText, language: quickRequestLanguage },
      });
      setQuickOptimizePreview(preview);
    } catch (previewError) {
      setError(`Text optimization failed: ${errorText(previewError, "unknown error")}`);
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
      setMessage("Synthesis stopped.");
      return;
    }
    if (!quickModel || !quickVoice) {
      setQuickNote({ error: true, text: "No voice fits this model. Add a voice on the Voices tab." });
      return;
    }
    let sampleRate = snapshot?.settings.output_sample_rate ?? snapshot?.settings.sample_rate ?? 24000;
    stopQuickPlayback();
    unlockAudioPlayback(sampleRate);
    if (quickAudioUrl) URL.revokeObjectURL(quickAudioUrl);
    setQuickAudioUrl("");
    setQuickAudioDownloadName("omnivoice-admin-quick.mp3");
    setQuickMetrics(null);
    setQuickNote(null);
    setQuickRunning(true);
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
          instructions: quickMingStructured || !quickReadsInstructions ? "" : quickInstructions,
          language: quickRequestLanguage,
          stream: true,
          response_format: "pcm",
          seed: quickReadsSeed && parsedSeed !== null && Number.isFinite(parsedSeed) ? parsedSeed : null,
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
      const liveNote = liveBlocked ? " The browser blocked live playback; use the player below." : "";
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
          setMessage(`Synthesis finished. The MP3 is ready.${liveNote}`);
        } catch (mp3Error) {
          if (controller.signal.aborted) return;
          if (chunks.length) {
            setQuickAudioDownloadName(`${completedJobId || "omnivoice-admin-quick"}.wav`);
            setQuickAudioUrl(URL.createObjectURL(createWavBlobFromInt16Chunks(chunks, sampleRate)));
          }
          const mp3Message = `MP3 export failed: ${errorText(mp3Error, "unknown error")}`;
          setQuickNote({ error: true, text: `${mp3Message}${liveNote}` });
          setMessage(`Synthesis finished. A WAV file is ready instead.${liveNote}`);
        }
      } else if (chunks.length) {
        setQuickAudioDownloadName("omnivoice-admin-quick.wav");
        setQuickAudioUrl(URL.createObjectURL(createWavBlobFromInt16Chunks(chunks, sampleRate)));
        setMessage(`Synthesis finished. A WAV file is ready instead.${liveNote}`);
      } else {
        setMessage("Synthesis finished.");
      }
    } catch (quickError) {
      if (!controller.signal.aborted) {
        const quickMessage = `Synthesis failed: ${errorText(quickError, "unknown error")}`;
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
    const note = preset.note ? ` Note: ${preset.note}` : "";
    toast.info(`Preset "${name}" loaded into the form. Click "Save changes" to apply it.${note}`);
  }

  async function handleSavePreset(event?: FormEvent) {
    event?.preventDefault();
    const name = presetSaveName.trim();
    if (!adminKey || !settingsDraft || !activeEngine || !name) return;
    if (presets.some((preset) => preset.built_in && preset.name === name)) {
      setError(`"${name}" is a built-in preset. Save under another name.`);
      return;
    }
    // The list shows the active engine only, so a same-named preset of another engine would be overwritten unseen.
    const foreign = presets.find((preset) => preset.name === name && preset.engine !== activeEngine.id);
    if (foreign) {
      setError(`"${name}" belongs to ${foreign.engine}. Save under another name.`);
      return;
    }
    try {
      await apiFetch<SettingsPreset>(`/api/admin/settings/presets/${encodeURIComponent(name)}`, {
        method: "PUT",
        adminKey,
        body: { values: { ...pickFields(settingsDraft, activeEngine.settings_fields), runtime_backend: activeEngine.id } },
      });
      await loadPresets(adminKey);
      setPresetName(name);
      setPresetSaveName("");
      setMessage(`Preset "${name}" saved.`);
    } catch (presetError) {
      setError(`Could not save the preset: ${errorText(presetError, "unknown error")}`);
    }
  }

  async function handleDeletePreset() {
    const name = presetName.trim();
    if (!adminKey || !name) return;
    if (!window.confirm(`Delete preset "${name}"?\n\nThe saved values are removed; the current settings stay.`)) return;
    try {
      await apiFetch<{ ok: boolean }>(`/api/admin/settings/presets/${encodeURIComponent(name)}`, { method: "DELETE", adminKey });
      setPresetName("");
      await loadPresets(adminKey);
      setMessage(`Preset "${name}" deleted.`);
    } catch (presetError) {
      setError(`Could not delete the preset: ${errorText(presetError, "unknown error")}`);
    }
  }

  async function handleSaveSettings() {
    if (!adminKey || !settingsDraft) return;
    const previousDevice = snapshot?.settings.preferred_device;
    const { default_model: defaultModel, ...values } = withoutEngine(settingsDraft);
    setSettingsSaving(true);
    try {
      const updated = await apiFetch<ServerSettings>("/api/admin/settings", {
        method: "PUT",
        adminKey,
        body: defaultModelPicked ? { ...values, default_model: defaultModel } : values,
      });
      const adjusted = Object.keys(values).filter((key) => {
        if (SETTINGS_NOT_COMPARED.has(key)) return false;
        const sent = (values as Record<string, unknown>)[key];
        const got = (updated as unknown as Record<string, unknown>)[key];
        return sent !== undefined && sent !== null && got !== undefined && JSON.stringify(sent) !== JSON.stringify(got);
      });
      setSettingsDraft(applyOmniVoiceDefaults(updated));
      await loadSnapshot(adminKey, true);
      await loadModelDownloads(adminKey, updated.model_directory);
      setMessage(
        previousDevice && previousDevice !== updated.preferred_device
          ? "Settings saved. The new GPU takes effect after a model reload, an engine switch or a server restart."
          : "Settings saved.",
      );
      if (adjusted.length) toast.info(`The server adjusted: ${adjusted.join(", ")}.`);
    } catch (saveError) {
      setError(`Could not save settings: ${errorText(saveError, "unknown error")}`);
    } finally {
      setSettingsSaving(false);
    }
  }

  async function handleSetOpsModelAsDefault(modelId: string) {
    if (!adminKey || !modelId) return;
    try {
      const updated = await apiFetch<ServerSettings>("/api/admin/settings", { method: "PUT", adminKey, body: { default_model: modelId } });
      // Only the model was saved: other unsaved edits stay in the draft.
      setDefaultModelPicked(false);
      patchDraft({ default_model: updated.default_model });
      setSavedSettings((current) => (current ? { ...current, default_model: updated.default_model } : current));
      if (savedSettingsRef.current) savedSettingsRef.current = { ...savedSettingsRef.current, default_model: updated.default_model };
      setMessage("Default model saved.");
    } catch (saveError) {
      setError(`Could not save the default model: ${errorText(saveError, "unknown error")}`);
    }
  }

  async function handleModelOperation(kind: "preload" | "warmup" | "unload" | "reload") {
    if (!adminKey || !settingsDraft) return;
    const targetModel = settingsDraft.default_model;
    const targetTaskType = inferTaskType(targetModel);
    const warmupVoice = (targetTaskType === "Base" ? customVoices[0]?.voice_id : undefined) || settingsDraft.default_voice || null;
    setModelOpsBusy(true);
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
      setError(`Model ${kind} failed: ${errorText(operationError, "unknown error")}`);
    } finally {
      setModelOpsBusy(false);
    }
  }

  async function handleModelDownloadRefresh() {
    if (!adminKey) return;
    setModelDownloadAction({ kind: "refresh", modelId: "__refresh__" });
    try {
      await loadModelDownloads(adminKey, savedSettings?.model_directory || settingsDraft?.model_directory || "");
      setMessage("Model cache reloaded.");
    } finally {
      setModelDownloadAction(null);
    }
  }

  /** `label` names the download in the UI; `target` is a model id, or an engine that is not active yet. */
  async function startModelDownload(label: string, target: { model_id: string } | { engine: string }) {
    if (!adminKey || !settingsDraft) return;
    setModelDownloadAction({ kind: "download", modelId: label });
    try {
      const payload = await apiFetch<ModelDownloadListResponse>("/api/admin/models/download", {
        method: "POST",
        adminKey,
        body: {
          ...target,
          storage_path: savedSettings?.model_directory ?? settingsDraft.model_directory,
        },
      });
      setModelDownloads(payload.models || []);
      await loadEngines(adminKey);
      setMessage(`Download of ${label} started.`);
    } catch (downloadError) {
      setError(`Could not start the download: ${errorText(downloadError, "unknown error")}`);
    } finally {
      setModelDownloadAction(null);
    }
  }

  async function handleModelDelete(modelId: string) {
    if (!adminKey || !settingsDraft) return;
    setModelDownloadAction({ kind: "delete", modelId });
    try {
      const payload = await apiFetch<ModelDownloadListResponse>("/api/admin/models/delete", {
        method: "POST",
        adminKey,
        body: {
          model_id: modelId,
          storage_path: savedSettings?.model_directory ?? settingsDraft.model_directory,
        },
      });
      setModelDownloads(payload.models || []);
      setMessage(payload.removed ? `Cache of ${modelId} deleted.` : `No local cache of ${modelId} found.`);
    } catch (deleteError) {
      setError(`Could not delete the cache: ${errorText(deleteError, "unknown error")}`);
    } finally {
      setModelDownloadAction(null);
    }
  }

  async function handleFreeMemory() {
    if (!adminKey) return;
    setMemoryCleanupBusy(true);
    try {
      const payload = await apiFetch<MemoryCleanupResponse>("/api/admin/runtime/free-memory", {
        method: "POST",
        adminKey,
      });
      await loadSnapshot(adminKey);
      const gb = (mb?: number | null) => (mb === null || mb === undefined ? "—" : `${(mb / 1024).toFixed(1)} GB`);
      setMessage(`${payload.message} Card VRAM: ${gb(payload.memory_before_mb)} → ${gb(payload.memory_after_mb)}, released ${gb(payload.released_mb)}.`);
    } catch (cleanupError) {
      setError(`Could not free VRAM: ${errorText(cleanupError, "unknown error")}`);
    } finally {
      setMemoryCleanupBusy(false);
    }
  }

  async function handleBenchmarkRun() {
    if (!adminKey || !settingsDraft) return;
    setBenchmarkBusy(true);
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
      setMessage("Benchmark started.");
    } catch (benchmarkError) {
      setError(`Could not start the benchmark: ${errorText(benchmarkError, "unknown error")}`);
    } finally {
      setBenchmarkBusy(false);
    }
  }

  // Benchmark and WER runs measure the plain voice. On Ming the quick-synthesis textarea is the free-form
  // instruction (a caption the engine applies), so it is not sent with them; an engine that reads none hides it.
  function benchmarkInstructions(): string {
    return mingEngineActive || !quickReadsInstructions ? "" : quickInstructions;
  }

  async function handleWerBenchmarkRun() {
    if (!adminKey || !settingsDraft) return;
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
      setMessage("WER benchmark started.");
    } catch (werError) {
      setError(`Could not start the WER benchmark: ${errorText(werError, "unknown error")}`);
    } finally {
      setWerBenchmarkBusy(false);
    }
  }

  async function handleCancelWerBenchmark(runId: string) {
    if (!adminKey) return;
    if (!window.confirm("Stop the WER benchmark?\n\nRunning and waiting TTS jobs are cancelled; finished results stay.")) return;
    setWerCancelling(runId);
    try {
      await apiFetch<WerBenchmarkRunResponse>(`/api/admin/wer-benchmarks/runs/${encodeURIComponent(runId)}/cancel`, { method: "POST", adminKey });
      setMessage("WER benchmark stopped.");
      await loadWerBenchmarks(adminKey);
    } catch (cancelError) {
      setError(`Could not stop the WER benchmark: ${errorText(cancelError, "unknown error")}`);
    } finally {
      setWerCancelling("");
    }
  }

  async function saveVoiceSeed(voice: VoiceItem, engine: SeedEngine, seed: number | null) {
    if (!adminKey) return false;
    try {
      await apiFetch(`/api/admin/voices/${encodeURIComponent(voice.voice_id)}`, { method: "PATCH", adminKey, body: { seed, engine } });
      const label = seedEngineLabel(engine);
      setMessage(seed === null ? `${label} seed of ${voice.name} removed.` : `Best ${label} seed of ${voice.name}: ${seed}.`);
      await loadSnapshot(adminKey);
      return true;
    } catch (seedError) {
      setError(`Could not save the seed: ${errorText(seedError, "unknown error")}`);
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
        setError("The seed must be a whole number between 0 and 2147483647.");
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
    if (!runVoice || seed === null || seed === undefined) return "—";
    if (voiceEngineSeed(runVoice, run.engine) === seed) return <span className="badge ok">Saved</span>;
    return (
      <button className="secondary-button" type="button" title={ADMIN_HELP.werSaveSeed} onClick={() => void saveVoiceSeed(runVoice, seedEngineKey(run.engine), seed)}>
        Save for {runVoice.name} ({seedEngineLabel(run.engine)})
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

  async function handleUploadVoice(event?: FormEvent) {
    event?.preventDefault();
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
      setMessage("Voice saved.");
    } catch (uploadError) {
      setError(`Could not save the voice: ${errorText(uploadError, "unknown error")}`);
    }
  }

  async function handleTranscribeVoice() {
    if (!adminKey || !voiceFile) return;
    setVoiceTranscribing(true);
    const form = new FormData();
    form.append("file", voiceFile);
    try {
      const payload = await apiFetch<{ transcription?: string; text?: string }>("/api/admin/voices/transcribe", {
        method: "POST",
        adminKey,
        body: form,
      });
      const transcript = (payload.transcription || payload.text || "").trim();
      if (!transcript) throw new Error("Whisper returned no text.");
      setVoiceRefText(transcript);
      setMessage("Reference text filled in from Whisper.");
    } catch (transcribeError) {
      setError(`Transcription failed: ${errorText(transcribeError, "unknown error")}`);
    } finally {
      setVoiceTranscribing(false);
    }
  }

  async function handleDeleteVoice(voice: VoiceItem) {
    if (!adminKey) return;
    if (!window.confirm(`Delete voice "${voice.name}"?\n\nRequests that use it fail until it is added again.`)) return;
    try {
      await apiFetch(`/api/admin/voices/${encodeURIComponent(voice.voice_id)}`, { method: "DELETE", adminKey });
      setVoiceAudioUrls((current) => {
        const existing = current[voice.voice_id];
        if (existing) URL.revokeObjectURL(existing);
        const next = { ...current };
        delete next[voice.voice_id];
        return next;
      });
      setMessage(`Voice "${voice.name}" deleted.`);
      await loadSnapshot(adminKey);
    } catch (deleteError) {
      setError(`Could not delete the voice: ${errorText(deleteError, "unknown error")}`);
    }
  }

  async function handleLoadVoiceSample(voiceId: string) {
    if (!adminKey) return;
    try {
      const blob = await apiFetch<Blob>(`/api/admin/voices/${encodeURIComponent(voiceId)}/audio`, { adminKey, responseType: "blob" });
      setVoiceAudioUrls((current) => {
        if (current[voiceId]) URL.revokeObjectURL(current[voiceId]);
        return { ...current, [voiceId]: URL.createObjectURL(blob) };
      });
    } catch (loadError) {
      setError(`Could not load the sample: ${errorText(loadError, "unknown error")}`);
    }
  }

  async function handleLoadJobAudio(jobId: string) {
    if (!adminKey) return;
    try {
      const blob = await apiFetch<Blob>(`/api/admin/jobs/${encodeURIComponent(jobId)}/audio?format=mp3`, { adminKey, responseType: "blob" });
      setJobAudioUrls((current) => {
        if (current[jobId]) URL.revokeObjectURL(current[jobId]);
        return { ...current, [jobId]: URL.createObjectURL(blob) };
      });
    } catch (loadError) {
      setError(`Could not load the audio: ${errorText(loadError, "unknown error")}`);
    }
  }

  async function handleDeleteJob(jobId: string, completed: boolean) {
    if (!adminKey) return;
    if (!completed && !window.confirm("Cancel this job?\n\nThe client waiting for it gets an error.")) return;
    try {
      await apiFetch(`/api/admin/jobs/${encodeURIComponent(jobId)}`, { method: "DELETE", adminKey });
      await loadSnapshot(adminKey);
    } catch (deleteError) {
      setError(`Could not ${completed ? "remove" : "cancel"} the job: ${errorText(deleteError, "unknown error")}`);
    }
  }

  async function handleLogout() {
    try {
      await logout();
    } catch {
      // clear locally regardless
    }
    resetSession(false);
  }

  function onTabKeyDown(event: ReactKeyboardEvent<HTMLButtonElement>) {
    const visible = visibleTabs;
    const index = visible.findIndex((tab) => tab.id === adminTab);
    let next = index;
    if (event.key === "ArrowRight") next = (index + 1) % visible.length;
    else if (event.key === "ArrowLeft") next = (index - 1 + visible.length) % visible.length;
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = visible.length - 1;
    else return;
    event.preventDefault();
    const id = visible[next].id;
    setAdminTab(id);
    window.requestAnimationFrame(() => document.getElementById(`tab-${id}`)?.focus());
  }

  const visibleTabs = TABS.filter((tab) => finetuneSupported || !FINETUNE_TABS.includes(tab.id));
  const settingsDirty = settingsChanges(settingsDraft, savedSettings);
  const keyCount = apiKeys?.length ?? 0;
  const serviceGpus = (perfLatest?.gpus ?? []).filter((gpu) => gpu.used_by_service);

  if (bootLoading) {
    return (
      <div className="shell">
        <div className="state" role="status"><span className="spinner" />Loading…</div>
      </div>
    );
  }

  if (!adminKey) {
    return (
      <div className="gate-shell">
        <ToastHost />
        <main className="panel gate-panel" aria-labelledby="login-title">
          <p className="eyebrow">G3 GPU service · Admin</p>
          <h1 id="login-title">G3 OmniVoice</h1>
          <form className="form-stack" onSubmit={(event) => void handleAuthenticate(event)}>
            <label>
              Username
              <input value={loginUsername} autoComplete="username" onChange={(event) => setLoginUsername(event.target.value)} required />
            </label>
            <label>
              Password
              <input type="password" value={loginPassword} autoComplete="current-password" onChange={(event) => setLoginPassword(event.target.value)} required />
            </label>
            {setupPending ? (
              <p className="hint">First start: log in with <code>admin</code> / <code>admin</code>, then choose a new password.</p>
            ) : null}
            {loginError ? <div className="banner danger" role="alert">{loginError}</div> : null}
            <div className="button-row form-actions">
              <button type="submit" disabled={authLoading || !loginUsername.trim() || !loginPassword} aria-busy={authLoading}>
                {authLoading ? "Logging in…" : "Log in"}
              </button>
            </div>
          </form>
        </main>
      </div>
    );
  }

  if (mustChange) {
    return (
      <div className="gate-shell">
        <ToastHost />
        <main className="panel gate-panel" aria-labelledby="pw-title">
          <p className="eyebrow">G3 OmniVoice · Security</p>
          <h1 id="pw-title">Change password</h1>
          <p className="dim">Signed in as {currentUser || "admin"}. Choose a new password (at least 8 characters) to continue.</p>
          <form className="form-stack" onSubmit={(event) => void handleChangePassword(event)}>
            <input type="text" autoComplete="username" value={currentUser || "admin"} readOnly hidden />
            <label>Current password<input type="password" value={pwCurrent} autoComplete="current-password" onChange={(event) => setPwCurrent(event.target.value)} required /></label>
            <label>New password<input type="password" value={pwNew} autoComplete="new-password" minLength={8} onChange={(event) => setPwNew(event.target.value)} required /></label>
            <label>Confirm new password<input type="password" value={pwConfirm} autoComplete="new-password" minLength={8} onChange={(event) => setPwConfirm(event.target.value)} required /></label>
            {pwError ? <div className="banner danger" role="alert">{pwError}</div> : null}
            <div className="button-row form-actions">
              <button type="submit" disabled={pwBusy || !pwCurrent || !pwNew || !pwConfirm} aria-busy={pwBusy}>{pwBusy ? "Saving…" : "Change password"}</button>
              <button className="ghost-button" type="button" onClick={() => void handleLogout()}>Log out</button>
            </div>
          </form>
        </main>
      </div>
    );
  }

  if (!snapshot || !settingsDraft) {
    return (
      <div className="shell">
        <ToastHost />
        {snapshotError ? (
          <div className="banner danger" role="alert">
            <span>Could not load the dashboard: {snapshotError}</span>
            <span className="button-row">
              <button className="secondary-button" type="button" onClick={() => void initialLoad()}>Retry</button>
              <button className="ghost-button" type="button" onClick={() => void handleLogout()}>Log out</button>
            </span>
          </div>
        ) : (
          <div className="state" role="status"><span className="spinner" />Loading…</div>
        )}
      </div>
    );
  }

  const overview = snapshot.overview;
  const switchFailed = engineList?.switch.state === "failed";
  const modelLoaded = activeEngine ? activeEngine.loaded : snapshot.models.some((model) => model.loaded);
  const stateBadge = engineSwitchRunning
    ? { cls: "warn", label: "Loading", title: "An engine switch or model load is running." }
    : switchFailed
      ? { cls: "danger", label: "Error", title: engineList?.switch.error || "The last engine switch failed." }
      : modelLoaded
        ? { cls: "ok", label: "Ready", title: "The model is loaded." }
        : { cls: "plain", label: "Idle · model unloaded", title: "The model loads with the next request or with Preload." };
  const gpuText = serviceGpus.length ? serviceGpus.map(gpuTitle).join(", ") : perfLatest ? "CPU" : "—";
  const runtimeVram =
    overview.gpu_memory_total_mb > 0
      ? `This service ${overview.gpu_process_memory_mb != null ? `${(overview.gpu_process_memory_mb / 1024).toFixed(1)} GB` : "—"} · card ${(overview.gpu_memory_used_mb / 1024).toFixed(1)} / ${(overview.gpu_memory_total_mb / 1024).toFixed(1)} GB`
      : "—";
  const protectedRoutes = "POST /v1/audio/speech, /v1/jobs, /api/v1/synthesize(/stream), /api/v1/text/optimize and the model, voice, capability and job-audio routes";

  const dashboardTab = (
    <>
      <div className="grid grid-auto">
        <div className="card">
          <span className="card-label">Model</span>
          <span className="card-value small" title={overview.active_model || undefined}>{overview.active_model || "None"}</span>
          <span className="card-sub">{activeEngine?.label ?? activeEngineId ?? "—"}</span>
        </div>
        <div className="card">
          <span className="card-label">State</span>
          <span><span className={`badge ${stateBadge.cls}`} title={stateBadge.title}>{stateBadge.label}</span></span>
          <span className="card-sub">Queue {formatNumber(overview.queue_depth)} · active {formatNumber(overview.active_requests)}</span>
        </div>
        <div className="card">
          <span className="card-label">Requests</span>
          <span className="card-value">{formatNumber(overview.jobs_total)}</span>
          <span className="card-sub">since start · {formatSeconds(overview.audio_seconds_total)} audio</span>
        </div>
        <div className="card">
          <span className="card-label">Errors</span>
          <span className="card-value">{formatNumber(overview.jobs_failed_total)}</span>
          <span className="card-sub">failed jobs since start</span>
        </div>
      </div>

      <PerfPanel />

      <div className="grid grid-2">
        <section className="panel" aria-labelledby="queue-title">
          <div className="panel-head"><h2 id="queue-title">Live queue</h2></div>
          <dl className="kv-list">
            <div><dt>Worker</dt><dd>{overview.worker_state}</dd></div>
            <div><dt>Queue depth</dt><dd>{formatNumber(overview.queue_depth)}</dd></div>
            <div><dt>Active requests</dt><dd>{formatNumber(overview.active_requests)}</dd></div>
            <div><dt>TTFA (average)</dt><dd>{formatMs(overview.ttfa_ms_avg)}</dd></div>
            <div><dt>Queue wait (average)</dt><dd>{formatMs(overview.queue_wait_ms_avg)}</dd></div>
            <div><dt>Job time (average)</dt><dd>{formatMs(overview.job_wall_ms_avg)}</dd></div>
            <div><dt>Realtime factor (average)</dt><dd>{formatRealtime(overview.realtime_x_avg)}</dd></div>
          </dl>
          <div className="table-wrap">
            <table>
              <thead><tr><th>Batch</th><th>Model</th><th>Task</th><th className="num">Size</th></tr></thead>
              <tbody>
                {snapshot.current_batch ? (
                  <tr className="row-selected">
                    <td><span className="badge warn">Running</span> <span className="mono">{snapshot.current_batch.batch_id}</span></td>
                    <td className="wrap-cell">{snapshot.current_batch.model_id}</td>
                    <td>{snapshot.current_batch.task_type}</td>
                    <td className="num">{snapshot.current_batch.size}</td>
                  </tr>
                ) : null}
                {snapshot.recent_batches.slice(-5).reverse().map((batch) => (
                  <tr key={batch.batch_id}>
                    <td><span className="mono">{batch.batch_id}</span></td>
                    <td className="wrap-cell">{batch.model_id}</td>
                    <td>{batch.task_type}</td>
                    <td className="num">{batch.size}</td>
                  </tr>
                ))}
                {!snapshot.current_batch && !snapshot.recent_batches.length ? (
                  <tr><td colSpan={4} className="empty">No batches yet. Run a test on the Playground tab.</td></tr>
                ) : null}
              </tbody>
            </table>
          </div>
        </section>

        <section className="panel" aria-labelledby="runtime-title">
          <div className="panel-head"><h2 id="runtime-title">Runtime</h2></div>
          <dl className="kv-list">
            <div><dt>Engine</dt><dd>{activeEngine?.label ?? activeEngineId ?? "—"}</dd></div>
            <div><dt>Default model</dt><dd className="mono">{snapshot.settings.default_model || "—"}</dd></div>
            <div><dt>Device</dt><dd>{selectedRuntimeDeviceLabel}</dd></div>
            <div><dt>GPU</dt><dd>{gpuText}</dd></div>
            <div><dt>VRAM</dt><dd>{runtimeVram}</dd></div>
            <div><dt>GPU temperature</dt><dd>{overview.gpu_temperature_c != null ? `${overview.gpu_temperature_c} °C` : "—"}</dd></div>
            <div><dt>Output sample rate</dt><dd>{formatNumber(snapshot.settings.output_sample_rate ?? snapshot.settings.sample_rate)} Hz</dd></div>
            <div><dt>Voices</dt><dd>{formatNumber(customVoices.length)} custom</dd></div>
          </dl>
        </section>
      </div>
    </>
  );

  const playgroundTab = (
    <section className="panel" aria-labelledby="quick-title">
      <div className="panel-head">
        <h2 id="quick-title">Quick synthesis</h2>
        <p className="hint">Streams through the public API with your admin login (no API key needed). The first sentence plays while the rest is generated.</p>
      </div>
      <form className="form-stack" onSubmit={(event) => { event.preventDefault(); void handleQuickRun(); }}>
        <div className="field-grid four">
          {selectableModels.length > 1 ? (
            <label>Model<select value={quickModel} onChange={(event) => setQuickModel(event.target.value)}>{selectableModels.map((model) => <option key={model} value={model}>{model}</option>)}</select><span className="hint">{ADMIN_HELP.quickModel}</span></label>
          ) : null}
          <label>Voice<select value={quickVoice} onChange={(event) => handleQuickVoiceChange(event.target.value)}>{!quickVoices.length ? <option value="">No matching voice</option> : null}{quickVoices.map((voice) => <option key={voice.voice_id} value={voice.source === "custom" ? voice.voice_id : voice.name}>{voice.name} ({voice.source})</option>)}</select><span className="hint">{ADMIN_HELP.quickVoice}</span></label>
          {quickReadsLanguage ? (
            <label>{quickHasLanguageChoice ? "Language" : "Language (text optimization)"}<select value={quickLanguage} onChange={(event) => setQuickLanguage(event.target.value)}>{SYNTH_LANGUAGE_OPTIONS.map((option) => <option key={option} value={option}>{option}</option>)}</select><span className="hint">{quickHasLanguageChoice ? ADMIN_HELP.quickLanguage : ADMIN_HELP.quickLanguageTextOnly}</span></label>
          ) : null}
          {quickReadsSeed ? (
            <label>Seed<input type="number" min="0" max="2147483647" value={quickSeed} onChange={(event) => setQuickSeed(event.target.value)} placeholder="Empty = random" /><span className="hint">{ADMIN_HELP.quickSeed}</span></label>
          ) : null}
          <label>Text optimization<select value={quickTextOptimization} onChange={(event) => setQuickTextOptimization(event.target.value as "" | TextOptimizationMode)}>
            <option value="">Server default ({snapshot.settings.text_optimization_default ?? "off"})</option>
            <option value="off">off</option>
            <option value="deterministic">deterministic</option>
          </select></label>
        </div>
        <label>Text<textarea ref={quickTextRef} value={quickText} onChange={(event) => setQuickText(event.target.value)} /><span className="hint">{ADMIN_HELP.quickText}</span></label>
        {mingEngineActive ? (
          <div className="tag-insert">
            <div className="button-row">
              <span className="card-label">Insert sound</span>
              {MING_TAGS.filter((item) => item.common).map((item) => (
                <button key={item.tag} className="secondary-button small-button" type="button" title={item.tag} onClick={() => insertQuickTag(item.tag)}>{item.label}</button>
              ))}
            </div>
            <details>
              <summary className="dim">More sounds</summary>
              <div className="button-row">
                {MING_TAGS.filter((item) => !item.common).map((item) => (
                  <button key={item.tag} className="secondary-button small-button" type="button" title={item.tag} onClick={() => insertQuickTag(item.tag)}>{item.label}</button>
                ))}
              </div>
            </details>
            <p className="hint">{ADMIN_HELP.mingTags}</p>
          </div>
        ) : null}
        {mingEngineActive ? (
          <fieldset className="choice-group">
            <legend>Ming delivery</legend>
            <p className="hint">Emotion and rate have the strongest effect, pitch and volume hardly any. With an emotion, "auto" uses only the speaker embedding: more expression, but the voice sounds a little less like the reference.</p>
            <div className="field-grid four">
              <label>Emotion
                <select value={quickMingDelivery.emotion} onChange={(event) => patchQuickMingDelivery({ emotion: event.target.value as MingDeliveryDraft["emotion"] })}>
                  <option value="">none</option>
                  {MING_EMOTION_OPTIONS.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
                </select>
                <span className="hint">{ADMIN_HELP.mingEmotion}</span>
              </label>
              <label>Rate
                <select value={quickMingDelivery.speaking_rate} onChange={(event) => patchQuickMingDelivery({ speaking_rate: event.target.value as MingDeliveryDraft["speaking_rate"] })}>
                  <option value="">default</option>
                  {MING_RATE_OPTIONS.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
                </select>
                <span className="hint">{ADMIN_HELP.mingRate}</span>
              </label>
              <label>Pitch
                <select value={quickMingDelivery.pitch} onChange={(event) => patchQuickMingDelivery({ pitch: event.target.value as MingDeliveryDraft["pitch"] })}>
                  <option value="">default</option>
                  {MING_PITCH_OPTIONS.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
                </select>
                <span className="hint">{ADMIN_HELP.mingPitch}</span>
              </label>
              <label>Volume
                <select value={quickMingDelivery.volume} onChange={(event) => patchQuickMingDelivery({ volume: event.target.value as MingDeliveryDraft["volume"] })}>
                  <option value="">default</option>
                  {MING_VOLUME_OPTIONS.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
                </select>
                <span className="hint">{ADMIN_HELP.mingVolume}</span>
              </label>
              <label>Reference mode
                <select value={quickMingDelivery.reference_mode === "auto" ? "" : quickMingDelivery.reference_mode} onChange={(event) => patchQuickMingDelivery({ reference_mode: event.target.value as MingDeliveryDraft["reference_mode"] })}>
                  <option value="">auto ({resolvedReferenceMode(quickMingDelivery) === "speaker" ? "speaker only" : "full"})</option>
                  {REFERENCE_MODE_OPTIONS.filter((option) => option.value !== "auto").map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
                </select>
                <span className="hint">{ADMIN_HELP.mingReferenceMode}</span>
              </label>
            </div>
            <label>Free instruction (instead of emotion and rate)
              <textarea
                value={quickInstructions}
                onChange={(event) => setQuickInstructions(event.target.value)}
                disabled={quickMingStructured}
                placeholder={quickMingStructured ? "Off while emotion, rate, pitch or volume is set." : "For example: speaks calmly and warmly, slightly smiling"}
              />
              <span className="hint">{ADMIN_HELP.quickMingInstruction}</span>
            </label>
            {quickMingStructured && quickInstructions.trim() ? (
              <div className="banner info">The free instruction is not sent while emotion, rate, pitch or volume is set.</div>
            ) : null}
            {hasStructuredMingControl(quickMingDelivery) || quickMingDelivery.reference_mode || quickInstructions ? (
              <div className="button-row">
                <button className="ghost-button" type="button" onClick={() => setQuickMingDelivery(EMPTY_MING_DELIVERY)} disabled={!hasStructuredMingControl(quickMingDelivery) && !quickMingDelivery.reference_mode}>Reset Ming controls</button>
                <button className="ghost-button" type="button" onClick={() => setQuickInstructions("")} disabled={!quickInstructions}>Clear free instruction</button>
              </div>
            ) : null}
          </fieldset>
        ) : quickReadsInstructions ? (
          <label>Instructions<textarea value={quickInstructions} onChange={(event) => setQuickInstructions(event.target.value)} /><span className="hint">{ADMIN_HELP.quickInstructions}</span></label>
        ) : null}
        <p className="hint">{ADMIN_HELP.quickTextOptimization}</p>
        <div className="button-row">
          <button type="submit" className={quickRunning ? "danger-button" : undefined}>{quickRunning ? "Stop" : "Run synthesis"}</button>
          <button className="secondary-button" type="button" onClick={() => void handleQuickOptimizePreview()}>Preview text</button>
        </div>
      </form>
      {quickNote ? <div className={`banner ${quickNote.error ? "danger" : "info"}`} role="status">{quickNote.text}</div> : null}
      {quickOptimizePreview ? (
        <div className="card">
          <span className="card-label">Spoken text after text optimization</span>
          <dl className="kv-list">
            <div><dt>Language</dt><dd>{quickOptimizePreview.language ?? "none"} ({quickOptimizePreview.binding_origin})</dd></div>
            <div><dt>Rules</dt><dd>{quickOptimizePreview.applied_rules.length ? quickOptimizePreview.applied_rules.join(", ") : "none"}</dd></div>
          </dl>
          <p className="preview-text">{quickOptimizePreview.text}</p>
        </div>
      ) : null}
      {quickAudioUrl ? (
        <div className="button-row audio-row">
          <audio controls src={quickAudioUrl} />
          <a className="button-link secondary" href={quickAudioUrl} download={quickAudioDownloadName}>
            {quickAudioDownloadName.endsWith(".mp3") ? "Download MP3" : "Download WAV"}
          </a>
        </div>
      ) : null}
      {quickMetrics ? (
        <div className="grid grid-auto">
          <div className="card"><span className="card-label">TTFA</span><span className="card-value">{formatMs(quickMetrics.ttfa_ms)}</span></div>
          <div className="card"><span className="card-label">Realtime factor</span><span className="card-value">{formatRealtime(quickMetrics.realtime_x)}</span></div>
          <div className="card"><span className="card-label">Audio</span><span className="card-value">{formatSeconds((quickMetrics.audio_duration_ms || 0) / 1000)}</span></div>
        </div>
      ) : null}
    </section>
  );

  const voicesTab = (
    <div className="grid grid-2 tab-grid">
      {activeEngine?.uses_voice_profiles ? (
        <section className="panel" aria-labelledby="voice-add-title">
          <div className="panel-head"><h2 id="voice-add-title">Add voice</h2></div>
          <form className="form-stack" onSubmit={(event) => void handleUploadVoice(event)}>
            <label>Name<input value={voiceName} onChange={(event) => setVoiceName(event.target.value)} required /><span className="hint">{ADMIN_HELP.voiceName}</span></label>
            <label>Sample<input type="file" accept="audio/*" onChange={(event) => handleVoiceFileChange(event.target.files?.[0] ?? null)} /><span className="hint">{ADMIN_HELP.voiceSample}</span></label>
            {voiceUploadPreviewUrl ? <audio controls src={voiceUploadPreviewUrl} /> : null}
            <label>Reference text<textarea value={voiceRefText} onChange={(event) => setVoiceRefText(event.target.value)} /><span className="hint">{ADMIN_HELP.voiceRefText}</span></label>
            <label className="checkbox-row"><input type="checkbox" checked={voiceConsent} onChange={(event) => setVoiceConsent(event.target.checked)} /><span className="checkbox-text">Consent confirmed<span className="hint">{ADMIN_HELP.voiceConsent}</span></span></label>
            <div className="button-row">
              <button type="submit" disabled={!voiceFile || !voiceName.trim()}>Save voice</button>
              <button className="secondary-button" type="button" onClick={() => void handleTranscribeVoice()} disabled={!voiceFile || voiceTranscribing} aria-busy={voiceTranscribing} title={ADMIN_HELP.transcribeSample}>{voiceTranscribing ? "Transcribing…" : "Transcribe with Whisper"}</button>
            </div>
          </form>
        </section>
      ) : null}
      <section className={`panel ${activeEngine?.uses_voice_profiles ? "" : "span-all"}`} aria-labelledby="voice-lib-title">
        <div className="panel-head">
          <h2 id="voice-lib-title">Voice library</h2>
          <p className="hint">{ADMIN_HELP.voiceSeed}</p>
        </div>
        {libraryVoices.length === 0 ? (
          <div className="state">{activeEngine && !activeEngine.uses_voice_profiles ? "This model uses no voice profiles." : "No custom voices yet. Add one with a short, clean sample."}</div>
        ) : null}
        <div className="voice-list">
          {libraryVoices.map((voice) => {
            const engine = SEED_ENGINES.find((item) => item.id === activeSeedEngine) ?? SEED_ENGINES[0];
            const editKey = `${voice.voice_id}:${engine.id}`;
            return (
              <article key={voice.voice_id} className="card voice-card">
                <div className="voice-card-head">
                  <div>
                    <h3>{voice.name}</h3>
                    <p className="hint">{voice.source} · {formatDate(voice.created_at)} · <span className="mono">{voice.filename || (voice.has_audio ? `${voice.voice_id}.wav` : "—")}</span></p>
                  </div>
                  <div className="button-row">
                    <button className="secondary-button small-button" type="button" disabled={!voice.has_audio} onClick={() => void handleLoadVoiceSample(voice.voice_id)}>Play sample</button>
                    <button className="danger-button small-button" type="button" onClick={() => void handleDeleteVoice(voice)}>Delete</button>
                  </div>
                </div>
                <p className="voice-ref-text"><span className="card-label">Reference text</span> {voice.ref_text || "—"}</p>
                <form className="button-row" onSubmit={(event) => { event.preventDefault(); void handleSaveVoiceSeed(voice, engine.id); }}>
                  <label className="inline-field">Best {engine.label} seed
                    <input
                      type="number"
                      min="0"
                      max="2147483647"
                      placeholder="No seed"
                      value={voiceSeedEdits[editKey] ?? (voiceEngineSeed(voice, engine.id) ?? "")}
                      onChange={(event) => setVoiceSeedEdits((current) => ({ ...current, [editKey]: event.target.value }))}
                    />
                  </label>
                  <button className="secondary-button small-button" type="submit">Save seed</button>
                </form>
                {voiceAudioUrls[voice.voice_id] ? <audio controls src={voiceAudioUrls[voice.voice_id]} /> : null}
              </article>
            );
          })}
        </div>
      </section>
    </div>
  );

  const modelsTab = (
    <>
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
        <section className="panel"><div className="state"><span className="spinner" />Loading engines…</div></section>
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
    </>
  );

  const settingsTab = (
    <div className="settings-tab">
      <section className="panel" aria-labelledby="presets-title">
        <div className="panel-head">
          <h2 id="presets-title">Presets{activeEngine ? ` · ${activeEngine.label}` : ""}</h2>
          <p className="hint">Loading a preset only fills the form below; nothing changes until you click "Save changes".</p>
        </div>
        <fieldset className="switch-lock" disabled={engineSwitchRunning}>
          <div className="button-row">
            <label className="inline-field grow">Preset
              <select value={selectedPreset ? selectedPreset.name : ""} onChange={(event) => setPresetName(event.target.value)}>
                <option value="">{enginePresets.length ? "Choose a preset" : "No presets for this model"}</option>
                {enginePresets.map((preset) => <option key={preset.name} value={preset.name}>{preset.name}{preset.built_in ? " (built-in)" : ""}</option>)}
              </select>
            </label>
            <button className="secondary-button" type="button" onClick={() => selectedPreset && handleApplyPreset(selectedPreset.name)} disabled={!selectedPreset}>Load into form</button>
            <button className="danger-button" type="button" onClick={() => void handleDeletePreset()} disabled={!selectedPreset || Boolean(selectedPreset.built_in)} title={selectedPreset?.built_in ? "Built-in presets cannot be deleted." : undefined}>Delete</button>
            {selectedPreset?.built_in ? <span className="badge info">Built-in</span> : null}
            {selectedPreset?.note ? <span className="badge warn">No matching GPU</span> : null}
          </div>
          {selectedPreset?.description || selectedPreset?.note ? (
            <p className="hint">{[selectedPreset.description, selectedPreset.note].filter(Boolean).join(" ")}</p>
          ) : null}
          <form className="button-row" onSubmit={(event) => void handleSavePreset(event)}>
            <label className="inline-field grow">Save as<input value={presetSaveName} onChange={(event) => setPresetSaveName(event.target.value)} placeholder="New preset name" /></label>
            <button className="secondary-button" type="submit" disabled={!presetSaveName.trim() || !activeEngine}>Save as preset</button>
          </form>
        </fieldset>
      </section>

      <section className="panel" aria-labelledby="engine-settings-title">
        <div className="panel-head"><h2 id="engine-settings-title">Engine settings{activeEngine ? ` · ${activeEngine.label}` : ""}</h2></div>
        <fieldset className="switch-lock" disabled={engineSwitchRunning}>
          {activeEngine ? (
            <SettingsFields fields={activeEngine.settings_fields} draft={settingsDraft} context={settingsFieldContext} onPatch={patchDraft} />
          ) : (
            <div className="state"><span className="spinner" />Loading engine…</div>
          )}
          {activeEngine?.settings_fields.includes("preferred_device") ? (
            <div className="button-row">
              <button className="secondary-button small-button" type="button" title={ADMIN_HELP.refreshRuntimeDevices} onClick={() => void loadRuntimeDevices(adminKey)}>Reload GPU list</button>
              <span className={`badge ${runtimeDeviceOptions.some((device) => device.id === settingsDraft.preferred_device && device.available) ? "ok" : "warn"}`}>{selectedRuntimeDeviceLabel}</span>
            </div>
          ) : null}
        </fieldset>
      </section>

      <section className="panel" aria-labelledby="server-settings-title">
        <div className="panel-head"><h2 id="server-settings-title">Server-wide settings</h2></div>
        <fieldset className="switch-lock" disabled={engineSwitchRunning}>
          <SettingsFields fields={engineList?.server_settings_fields ?? []} draft={settingsDraft} context={settingsFieldContext} onPatch={patchDraft} />
          <div className="button-row">
            <button className="secondary-button small-button" type="button" title={ADMIN_HELP.refreshVllmModels} onClick={() => void loadVllmModels(adminKey, settingsDraft.vllm_base_url)} disabled={vllmModelsLoading} aria-busy={vllmModelsLoading}>{vllmModelsLoading ? "Loading…" : "Reload vLLM models"}</button>
            <span className={`badge ${vllmModels.length ? "ok" : "plain"}`}>{vllmModels.length ? `${vllmModels.length} vLLM model${vllmModels.length === 1 ? "" : "s"}` : "No vLLM models"}</span>
            {vllmModelsError ? <span className="badge danger" title={vllmModelsError}>vLLM error</span> : null}
          </div>
          {vllmModelsError ? <p className="field-error">{vllmModelsError}</p> : null}
        </fieldset>
      </section>

      <div className={`save-bar ${settingsDirty.length ? "dirty" : ""}`} role="region" aria-label="Unsaved settings">
        <span className="save-bar-text">
          {settingsDirty.length
            ? `${settingsDirty.length} unsaved change${settingsDirty.length === 1 ? "" : "s"}`
            : "All changes saved"}
          {settingsDirty.length ? <span className="hint"> · {settingsDirty.slice(0, 4).join(", ")}{settingsDirty.length > 4 ? ", …" : ""}</span> : null}
        </span>
        <span className="button-row">
          <button className="ghost-button" type="button" disabled={!settingsDirty.length || settingsSaving} onClick={() => { setSettingsDraft(savedSettings); setDefaultModelPicked(false); }}>Discard</button>
          <button className="secondary-button" type="button" disabled={settingsSaving} onClick={() => void loadSnapshot(adminKey, true).catch((reloadError) => setError(`Could not reload settings: ${errorText(reloadError, "unknown error")}`))}>Reload</button>
          <button type="button" disabled={!settingsDirty.length || settingsSaving || engineSwitchRunning} aria-busy={settingsSaving} onClick={() => void handleSaveSettings()}>{settingsSaving ? "Saving…" : "Save changes"}</button>
        </span>
      </div>
    </div>
  );

  const benchmarkTab = (
    <div className="grid grid-2 tab-grid">
      <section className="panel" aria-labelledby="bench-title">
        <div className="panel-head">
          <h2 id="bench-title">Benchmark</h2>
          <p className="hint">Runs exclusively: other requests wait while it measures. Uses the saved default model and voice.</p>
        </div>
        <form className="form-stack" onSubmit={(event) => { event.preventDefault(); void handleBenchmarkRun(); }}>
          <label>Mode<select value={benchmarkMode} onChange={(event) => setBenchmarkMode(event.target.value as "traffic" | "iterations")}>
            <option value="traffic">Traffic simulation</option>
            <option value="iterations">Fixed iterations</option>
          </select><span className="hint">{ADMIN_HELP.benchmarkMode}</span></label>
          <label>Sentence pool<textarea value={benchmarkText} onChange={(event) => setBenchmarkText(event.target.value)} /><span className="hint">{ADMIN_HELP.benchmarkText}</span></label>
          <div className="field-grid two">
            <label>Duration (s)<input type="number" min="1" max="3600" value={benchmarkDuration} onChange={(event) => setBenchmarkDuration(Number(event.target.value) || 1)} /><span className="hint">{ADMIN_HELP.benchmarkDuration}</span></label>
            <label>Requests per minute<input type="number" min="1" max="6000" value={benchmarkRequestsPerMinute} onChange={(event) => setBenchmarkRequestsPerMinute(Number(event.target.value) || 1)} /><span className="hint">{ADMIN_HELP.benchmarkRpm}</span></label>
            <label>Min sentences<input type="number" min="1" max="100" value={benchmarkMinSentences} onChange={(event) => setBenchmarkMinSentences(Number(event.target.value) || 1)} /></label>
            <label>Max sentences<input type="number" min="1" max="100" value={benchmarkMaxSentences} onChange={(event) => setBenchmarkMaxSentences(Number(event.target.value) || 1)} /><span className="hint">{ADMIN_HELP.benchmarkSentenceRange}</span></label>
            <label>Warmups<input type="number" min="0" max="20" value={benchmarkWarmups} onChange={(event) => setBenchmarkWarmups(Number(event.target.value) || 0)} /><span className="hint">{ADMIN_HELP.benchmarkWarmups}</span></label>
            <label>Seed<input type="number" min="0" max="2147483647" value={benchmarkSeed} onChange={(event) => setBenchmarkSeed(event.target.value)} placeholder="Empty = random" /><span className="hint">{ADMIN_HELP.benchmarkSeed}</span></label>
            {benchmarkMode === "iterations" ? (
              <>
                <label>Iterations<input type="number" min="1" max="50" value={benchmarkIterations} onChange={(event) => setBenchmarkIterations(Number(event.target.value) || 1)} /><span className="hint">{ADMIN_HELP.benchmarkIterations}</span></label>
                <label>Parallel<input type="number" min="1" max="64" value={benchmarkParallel} onChange={(event) => setBenchmarkParallel(Number(event.target.value) || 1)} /><span className="hint">{ADMIN_HELP.benchmarkParallel}</span></label>
              </>
            ) : null}
          </div>
          <div className="button-row"><button type="submit" disabled={benchmarkBusy} aria-busy={benchmarkBusy}>{benchmarkBusy ? "Starting…" : "Run benchmark"}</button></div>
        </form>
        {benchmarks.length === 0 ? <div className="state">No benchmark runs yet.</div> : null}
        {benchmarks.map((run) => (
          <article key={run.run_id} className="card">
            <div className="voice-card-head">
              <h3>{run.name}</h3>
              <span className={`badge ${run.status === "completed" ? "ok" : run.status === "failed" ? "danger" : "warn"}`}>{run.status}</span>
            </div>
            <p className="hint">{run.mode} · {run.total_requests || run.iterations} requests · {run.duration_seconds} s window · {run.requests_per_minute}/min · {run.completion_timeout_seconds || 180} s timeout · {run.warmup_iterations} warmup</p>
            <div className="table-wrap">
              <table>
                <thead><tr><th>Case</th><th className="num">OK</th><th className="num">TTFA avg / p99</th><th className="num">TTFA best / worst</th><th className="num">Job p50 / p99</th><th className="num">Queue p95 / p99</th><th className="num">Realtime</th></tr></thead>
                <tbody>
                  {run.cases.map((item) => (
                    <tr key={item.label}>
                      <td className="wrap-cell">{item.label}</td>
                      <td className="num">{item.success_count}/{item.success_count + item.failure_count}</td>
                      <td className="num">{formatMs(item.ttfa_ms_avg)} / {formatMs(item.ttfa_ms_p99)}</td>
                      <td className="num">{formatMs(item.ttfa_ms_min)} / {formatMs(item.ttfa_ms_max)}</td>
                      <td className="num">{formatMs(item.job_wall_ms_p50)} / {formatMs(item.job_wall_ms_p99)}</td>
                      <td className="num">{formatMs(item.queue_wait_ms_p95)} / {formatMs(item.queue_wait_ms_p99)}</td>
                      <td className="num">{formatRealtime(item.realtime_x_avg)}</td>
                    </tr>
                  ))}
                  {!run.cases.length ? <tr><td colSpan={7} className="empty">No results yet.</td></tr> : null}
                </tbody>
              </table>
            </div>
          </article>
        ))}
      </section>

      <section className="panel" aria-labelledby="wer-title">
        <div className="panel-head">
          <h2 id="wer-title">WER benchmark</h2>
          <p className="hint">vLLM writes sentences, the active engine speaks them, G3 Whisper transcribes them back; the word error rate compares both.</p>
        </div>
        <dl className="kv-list">
          <div><dt>vLLM</dt><dd className="mono">{settingsDraft.vllm_model || "auto"} @ {settingsDraft.vllm_base_url || "—"}</dd></div>
          <div><dt>Whisper</dt><dd className="mono">{settingsDraft.whisper_base_url || "—"}</dd></div>
          <div><dt>TTS / ASR concurrency</dt><dd>{settingsDraft.wer_concurrency} / {settingsDraft.wer_transcription_concurrency}</dd></div>
        </dl>
        <form className="form-stack" onSubmit={(event) => { event.preventDefault(); void handleWerBenchmarkRun(); }}>
          <div className="field-grid two">
            <label>Samples<input type="number" min="1" max="1000" value={werCount} onChange={(event) => setWerCount(Number(event.target.value) || 1)} /><span className="hint">{ADMIN_HELP.werCount}</span></label>
            <label>Language<select value={werLanguage} onChange={(event) => setWerLanguage(event.target.value)}>
              {WER_LANGUAGE_OPTIONS.map((language) => <option key={language} value={language}>{language}</option>)}
            </select><span className="hint">{ADMIN_HELP.werLanguage}</span></label>
            <label>Min words<input type="number" min="1" max="80" value={werMinWords} onChange={(event) => setWerMinWords(Number(event.target.value) || 1)} /></label>
            <label>Max words<input type="number" min="1" max="120" value={werMaxWords} onChange={(event) => setWerMaxWords(Number(event.target.value) || 1)} /><span className="hint">{ADMIN_HELP.werWords}</span></label>
            <label>Tolerance (letters)<input type="number" min="0" max="8" value={werTolerance} onChange={(event) => setWerTolerance(Number(event.target.value) || 0)} /><span className="hint">{ADMIN_HELP.werTolerance}</span></label>
            <label>Timeout (s)<input type="number" min="1" max="3600" value={werTimeout} onChange={(event) => setWerTimeout(Number(event.target.value) || 1)} /><span className="hint">{ADMIN_HELP.werTimeout}</span></label>
            <label>Voice<select value={werVoice} onChange={(event) => setWerVoice(event.target.value)}>
              <option value="">Default voice from the settings</option>
              {customVoices.map((voice) => {
                const seed = voiceEngineSeed(voice, activeSeedEngine);
                return <option key={voice.voice_id} value={voice.voice_id}>{voice.name}{seed != null ? ` (best ${seedEngineLabel(activeSeedEngine)} seed ${seed})` : ` (no ${seedEngineLabel(activeSeedEngine)} seed)`}</option>;
              })}
            </select><span className="hint">{ADMIN_HELP.werVoice}</span></label>
            <label>Seed<input type="text" value={werSeed} onChange={(event) => setWerSeed(event.target.value)} placeholder="Empty = voice's best seed" /><span className="hint">{ADMIN_HELP.werSeed}</span></label>
            <label>Seed range<input type="number" min="0" max="1024" value={werSeedRange} onChange={(event) => setWerSeedRange(Number(event.target.value) || 0)} /><span className="hint">{ADMIN_HELP.werSeedRange}</span></label>
          </div>
          <label>Custom vLLM prompt<textarea value={werPrompt} onChange={(event) => setWerPrompt(event.target.value)} placeholder="Empty = the built-in JSON sentence generator" /><span className="hint">{ADMIN_HELP.werPrompt}</span></label>
          <div className="button-row"><button type="submit" disabled={werBenchmarkBusy} aria-busy={werBenchmarkBusy}>{werBenchmarkBusy ? "Starting…" : "Run WER benchmark"}</button></div>
        </form>
        {werBenchmarks.length === 0 ? <div className="state">No WER benchmark runs yet.</div> : null}
        {werBenchmarks.map((run) => (
          <article key={run.run_id} className="card">
            <div className="voice-card-head">
              <h3>{run.name}</h3>
              <span className={`badge ${run.status === "completed" ? "ok" : run.status === "failed" ? "danger" : "warn"}`}>{run.status}</span>
            </div>
            <p className="hint">
              {run.summary.completed}/{run.summary.total} done · {run.sentence_cache_hit ? "cached sentences" : "new sentences"}
              {run.fallback_sentence_count ? ` · ${run.fallback_sentence_count} template sentences (the LLM returned too few)` : ""}
              {" · voice "}{run.voice ? voices.find((voice) => voice.voice_id === run.voice)?.name ?? run.voice : `default${werRunVoice(run) ? ` (${werRunVoice(run)?.name})` : ""}`}
              {" · engine "}{seedEngineLabel(run.engine)}
            </p>
            <div className="grid grid-auto">
              <div className="card"><span className="card-label">WER average</span><span className="card-value">{formatPercent(run.summary.wer_avg)}</span></div>
              <div className="card"><span className="card-label">Median</span><span className="card-value">{formatPercent(run.summary.wer_p50)}</span></div>
              <div className="card"><span className="card-label">p95</span><span className="card-value">{formatPercent(run.summary.wer_p95)}</span></div>
              <div className="card"><span className="card-label">Worst</span><span className="card-value">{formatPercent(run.summary.wer_max)}</span></div>
              <div className="card"><span className="card-label">Exact</span><span className="card-value">{formatPercent(run.summary.exact_rate)}</span></div>
              {run.summary.failure_count > 0 ? <div className="card"><span className="card-label">OK / failed</span><span className="card-value">{run.summary.success_count} / {run.summary.failure_count}</span></div> : null}
            </div>
            {isRunActive(run.status) ? (
              <div className="button-row">
                <button className="danger-button" type="button" disabled={werCancelling === run.run_id} aria-busy={werCancelling === run.run_id} onClick={() => void handleCancelWerBenchmark(run.run_id)} title={ADMIN_HELP.werCancel}>
                  {werCancelling === run.run_id ? "Stopping…" : "Stop WER benchmark"}
                </button>
              </div>
            ) : null}
            {run.seed_leaderboard.length > 1 ? (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th className="num">#</th>
                      <th className="num">Seed</th>
                      <th className="num">WER avg</th>
                      <th className="num">Median</th>
                      <th className="num">Worst</th>
                      {run.seed_leaderboard.some((item) => item.failure_count > 0) ? <th className="num">Failed</th> : null}
                      <th className="actions">Best seed</th>
                    </tr>
                  </thead>
                  <tbody>
                    {run.seed_leaderboard.map((item, index) => (
                      <tr key={item.seed ?? "random"}>
                        <td className="num">{index + 1}</td>
                        <td className="num">{item.seed ?? "random"}</td>
                        <td className="num">{formatPercent(item.wer_avg)}</td>
                        <td className="num">{formatPercent(item.wer_p50)}</td>
                        <td className="num">{formatPercent(item.wer_max)}</td>
                        {run.seed_leaderboard.some((entry) => entry.failure_count > 0) ? <td className="num">{item.failure_count}</td> : null}
                        <td className="actions">{renderSaveSeedCell(run, item.seed)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : null}
            <details open={deviationsFor(run.run_id).items.length > 0 && deviationsFor(run.run_id).items.length <= 5}>
              <summary className="dim">Deviations ({deviationsFor(run.run_id).items.length}/{deviationsFor(run.run_id).total}, worst first)</summary>
              {werDeviations?.runId === run.run_id && werDeviations.items.length === 0 ? <div className="state">No WER deviations in this run.</div> : null}
              <div className="wer-sample-list">
                {deviationsFor(run.run_id).items.map((item) => (
                  <div key={`${item.seed ?? "random"}-${item.index}-${item.job_id ?? ""}`} className="card wer-sample">
                    <div className="voice-card-head">
                      <strong>{item.seed !== null && item.seed !== undefined ? `Seed ${item.seed} #${item.index}` : `#${item.index}`}</strong>
                      <span className={`badge ${item.success ? "warn" : "danger"}`}>{item.success ? `WER ${formatPercent(item.wer)}` : "Failed"}</span>
                    </div>
                    <p className="hint">TTS {formatMs(item.synthesis_ms)} · ASR {formatMs(item.transcription_ms)} · total {formatMs(item.total_ms)}</p>
                    <p><span className="card-label">Reference</span> {item.source_text}</p>
                    <p><span className="card-label">Whisper</span> {item.transcript || item.error_message || "—"}</p>
                    <div className="button-row">
                      <button className="secondary-button small-button" type="button" disabled={!item.job_id} onClick={() => item.job_id && void handleLoadJobAudio(item.job_id)}>Load MP3</button>
                      {item.job_id && jobAudioUrls[item.job_id] ? (
                        <>
                          <audio controls src={jobAudioUrls[item.job_id]} />
                          <a className="button-link secondary small-button" href={jobAudioUrls[item.job_id]} download={`${item.job_id}.mp3`}>Download</a>
                        </>
                      ) : null}
                    </div>
                  </div>
                ))}
              </div>
              {deviationsFor(run.run_id).items.length < deviationsFor(run.run_id).total ? (
                <div className="button-row">
                  <button className="secondary-button" type="button" onClick={() => void loadWerDeviations(run.run_id, true)}>Load more</button>
                </div>
              ) : null}
            </details>
          </article>
        ))}
      </section>
    </div>
  );

  const historyTab = (
    <section className="panel" aria-labelledby="history-title">
      <div className="panel-head">
        <h2 id="history-title">History</h2>
        <p className="hint">The 24 latest jobs (updates live). Finished audio is kept for a while and can be played here.</p>
      </div>
      <div className="table-wrap">
        <table>
          <thead><tr><th>Text</th><th>Status</th><th>Model</th><th>Voice</th><th className="num">TTFA</th><th>Created</th><th className="actions"><span className="sr-only">Actions</span></th></tr></thead>
          <tbody>
            {snapshot.jobs.map((job) => (
              <tr key={job.job_id}>
                <td className="wrap-cell">
                  {job.input_preview}
                  {job.error_message ? <div className="field-error">{job.error_message}</div> : null}
                  {jobAudioUrls[job.job_id] ? <audio controls src={jobAudioUrls[job.job_id]} /> : null}
                </td>
                <td><span className={`badge ${job.status === "completed" ? "ok" : job.status === "failed" ? "danger" : job.status === "cancelled" ? "plain" : "warn"}`}>{job.status}</span></td>
                <td className="wrap-cell">{job.model || "—"}</td>
                <td>{job.voice || "—"}</td>
                <td className="num">{formatMs(job.metrics.ttfa_ms)}</td>
                <td>{formatDate(job.created_at)}</td>
                <td className="actions">
                  {job.status === "completed" ? (
                    jobAudioUrls[job.job_id] ? (
                      <a className="button-link secondary" href={jobAudioUrls[job.job_id]} download={`${job.job_id}.mp3`}>Download</a>
                    ) : (
                      <button className="secondary-button" type="button" onClick={() => void handleLoadJobAudio(job.job_id)}>Load MP3</button>
                    )
                  ) : null}
                  <button className={job.status === "completed" || job.status === "failed" || job.status === "cancelled" ? "ghost-button" : "danger-button"} type="button" onClick={() => void handleDeleteJob(job.job_id, ["completed", "failed", "cancelled"].includes(job.status))}>
                    {["completed", "failed", "cancelled"].includes(job.status) ? "Remove" : "Cancel"}
                  </button>
                </td>
              </tr>
            ))}
            {!snapshot.jobs.length ? <tr><td colSpan={7} className="empty">No jobs yet. Run a test on the Playground tab or send a request to /v1/audio/speech.</td></tr> : null}
          </tbody>
        </table>
      </div>
    </section>
  );

  const apiKeysTab = (
    <ApiKeysPanel
      keys={apiKeyRows}
      loadError={apiKeysError}
      usageLabel="Audio (s)"
      formatUsage={(value) => formatNumber(value, 1)}
      baseUrl={window.location.origin}
      exampleCurl={(key) =>
        `curl -X POST ${window.location.origin}/v1/audio/speech \\\n  -H "Authorization: Bearer ${key}" \\\n  -H "Content-Type: application/json" \\\n  -d '{"model": "${snapshot.settings.default_model}", "input": "Hello", "voice": "${snapshot.settings.default_voice || "Auto Voice"}", "response_format": "wav"}' \\\n  -o speech.wav`
      }
      routesText={protectedRoutes}
      onCreate={async (alias) => createApiKey(alias)}
      onRevoke={async (id) => { await deleteApiKey(id); }}
      onChanged={() => void refreshApiKeys()}
    />
  );

  const accountTab = (
    <div className="grid grid-2 tab-grid">
      <section className="panel" aria-labelledby="account-title">
        <div className="panel-head"><h2 id="account-title">Account</h2></div>
        <dl className="kv-list">
          <div><dt>Signed in as</dt><dd>{currentUser || "admin"}</dd></div>
          <div><dt>Session</dt><dd>httpOnly cookie, valid for 7 days</dd></div>
        </dl>
        <div className="button-row"><button className="ghost-button" type="button" onClick={() => void handleLogout()}>Log out</button></div>
      </section>
      <section className="panel" aria-labelledby="change-pw-title">
        <div className="panel-head"><h2 id="change-pw-title">Change password</h2><p className="hint">At least 8 characters. Other sessions of this account are logged out.</p></div>
        <form className="form-stack" onSubmit={(event) => void handleChangePassword(event)}>
          <input type="text" autoComplete="username" value={currentUser || "admin"} readOnly hidden />
          <label>Current password<input type="password" autoComplete="current-password" value={pwCurrent} onChange={(event) => setPwCurrent(event.target.value)} required /></label>
          <label>New password<input type="password" autoComplete="new-password" minLength={8} value={pwNew} onChange={(event) => setPwNew(event.target.value)} required /></label>
          <label>Confirm new password<input type="password" autoComplete="new-password" minLength={8} value={pwConfirm} onChange={(event) => setPwConfirm(event.target.value)} required /></label>
          {pwError ? <div className="banner danger" role="alert">{pwError}</div> : null}
          <div className="button-row form-actions">
            <button type="submit" disabled={pwBusy || !pwCurrent || !pwNew || !pwConfirm} aria-busy={pwBusy}>{pwBusy ? "Saving…" : "Change password"}</button>
          </div>
        </form>
      </section>
    </div>
  );

  return (
    <>
      <ToastHost />
      <div className="shell">
        <header className="app-header">
          <div className="titles">
            <p className="eyebrow">G3 GPU service · Admin</p>
            <h1>G3 OmniVoice</h1>
            <div className="status-strip">
              <span className={`badge ${stateBadge.cls}`} title={stateBadge.title}>{stateBadge.label}</span>
              <span className="chip"><span className="card-label">Model</span><span className="value" title={overview.active_model || undefined}>{overview.active_model || "None"}</span></span>
              <span className="chip"><span className="card-label">GPU</span><span className="value" title={gpuText}>{gpuText}</span></span>
              <button type="button" className="chip chip-button" onClick={() => setAdminTab("api-keys")} title="Manage API keys">
                <span className="card-label">API</span>
                <span className={`badge ${keyCount ? "ok" : "warn"}`}>{apiKeys === null ? "—" : keyCount ? "Protected" : "Open"}</span>
              </button>
            </div>
          </div>
          <div className="actions">
            <a className="button-link ghost" href="/docs" target="_blank" rel="noreferrer">API docs</a>
            <button className="ghost-button" type="button" onClick={() => void handleLogout()}>Log out</button>
          </div>
        </header>

        <nav className="tab-bar" role="tablist" aria-label="Sections">
          {visibleTabs.map((tab) => {
            const selected = tab.id === adminTab;
            return (
              <button
                key={tab.id}
                id={`tab-${tab.id}`}
                type="button"
                role="tab"
                className="tab-button"
                aria-selected={selected}
                aria-controls={`panel-${tab.id}`}
                tabIndex={selected ? 0 : -1}
                onClick={() => setAdminTab(tab.id)}
                onKeyDown={onTabKeyDown}
              >
                {tab.label}
                {tab.id === "settings" && settingsDirty.length ? <span className="tab-dot" aria-label="unsaved changes" /> : null}
              </button>
            );
          })}
        </nav>

        {liveReconnecting ? <div className="banner warn" role="status">Live updates reconnecting…</div> : null}

        <main className="tab-main" role="tabpanel" id={`panel-${adminTab}`} aria-labelledby={`tab-${adminTab}`}>
          {adminTab === "dashboard" ? dashboardTab : null}
          {adminTab === "playground" ? playgroundTab : null}
          {adminTab === "voices" ? voicesTab : null}
          {adminTab === "models" ? modelsTab : null}
          {adminTab === "settings" ? settingsTab : null}
          {adminTab === "benchmark" ? benchmarkTab : null}
          {adminTab === "data-generation" && finetuneSupported ? <DataGenerationPanel adminKey={adminKey} voices={voices} onMessage={setMessage} onError={setError} /> : null}
          {adminTab === "training" && finetuneSupported ? <TrainingPanel adminKey={adminKey} onMessage={setMessage} onError={setError} /> : null}
          {adminTab === "history" ? historyTab : null}
          {adminTab === "api-keys" ? apiKeysTab : null}
          {adminTab === "account" ? accountTab : null}
        </main>
      </div>
    </>
  );
}
