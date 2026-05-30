export type TaskType = "CustomVoice" | "VoiceDesign" | "Base";
export type JobStatus = "queued" | "warming" | "running" | "streaming" | "cancelling" | "completed" | "failed" | "cancelled";

export interface VoiceItem {
  voice_id: string;
  name: string;
  source: string;
  created_at?: string | null;
  ref_text?: string | null;
  filename?: string | null;
  has_audio?: boolean;
}

export interface ModelInfo {
  model_id: string;
  loaded: boolean;
  active: boolean;
  task_types: TaskType[];
}

export interface JobMetrics {
  queue_wait_ms?: number | null;
  model_warm_ms?: number | null;
  ttfa_ms?: number | null;
  job_wall_ms?: number | null;
  audio_duration_ms?: number | null;
  realtime_x?: number | null;
  output_bytes?: number | null;
  batch_count?: number | null;
  max_batch_size_seen?: number | null;
  last_batch_size?: number | null;
  sentences_total?: number | null;
  sentences_rendered?: number | null;
}

export interface GenerationControls {
  seed?: number | null;
}

export interface JobRecord {
  job_id: string;
  status: JobStatus;
  model?: string | null;
  task_type?: TaskType | null;
  voice?: string | null;
  input_preview: string;
  queue_position: number;
  eta_ms: number;
  created_at: string;
  updated_at: string;
  metrics: JobMetrics;
  error_message?: string | null;
}

export interface AdminKeyMeta {
  key_id: string;
  label: string;
  created_at: string;
  last_used_at?: string | null;
}

export interface ServerSettings {
  model_directory: string;
  default_model: string;
  default_voice: string;
  whisper_base_url?: string | null;
  whisper_path: string;
  vllm_base_url: string;
  vllm_model: string;
  wer_concurrency: number;
  wer_transcription_concurrency: number;
  retention_days: number;
  queue_limit: number;
  runtime_backend: string;
  allow_model_downloads: boolean;
  preferred_device: string;
  attention_implementation: string;
  torch_dtype: string;
  compile_model: boolean;
  compile_cudagraphs: boolean;
  cudagraph_skip_dynamic_graphs: boolean;
  cuda_memory_trim_after_batch: boolean;
  warmup_on_startup: boolean;
  sample_rate: number;
  poll_interval_ms: number;
  theme: string;
  built_in_voices: string[];
  sentence_chunking: boolean;
  short_sentence_merge_max_chars: number;
  following_sentence_merge_min_chars: number;
  max_parallel_requests: number;
  max_batch_size: number;
  batch_wait_ms: number;
  vram_budget_mb: number;
  max_input_chars: number;
  max_batch_audio_seconds?: number;
  max_chars_per_chunk?: number;
  estimated_peak_vram_mb?: number;
  stream_chunk_ms: number;
  stream_prebuffer_ms: number;
  num_step?: number | null;
  guidance_scale?: number | null;
  duration?: number | null;
  t_shift?: number | null;
  denoise?: boolean | null;
  preprocess_prompt?: boolean | null;
  postprocess_output?: boolean | null;
  audio_chunk_duration?: number | null;
  audio_chunk_threshold?: number | null;
  position_temperature?: number | null;
  class_temperature?: number | null;
}

export interface DashboardSnapshot {
  overview: {
    active_model: string;
    queue_depth: number;
    active_requests: number;
    worker_state: string;
    ttfa_ms_avg?: number | null;
    queue_wait_ms_avg?: number | null;
    job_wall_ms_avg?: number | null;
    realtime_x_avg?: number | null;
    jobs_total: number;
    audio_seconds_total: number;
    gpu_name: string;
    gpu_memory_used_mb: number;
    gpu_memory_total_mb: number;
    gpu_utilization_pct: number;
    gpu_temperature_c?: number | null;
  };
  settings: ServerSettings;
  models: ModelInfo[];
  voices: VoiceItem[];
  jobs: JobRecord[];
  admin_key: AdminKeyMeta;
  current_batch?: {
    batch_id: string;
    model_id: string;
    task_type: TaskType;
    voice?: string | null;
    language?: string | null;
    size: number;
    started_at: string;
    request_ids: string[];
    sentence_indices: number[];
  } | null;
  recent_batches: Array<{
    batch_id: string;
    model_id: string;
    task_type: TaskType;
    voice?: string | null;
    language?: string | null;
    size: number;
    started_at: string;
    request_ids: string[];
    sentence_indices: number[];
  }>;
}

export interface StreamChunkEvent {
  type: "chunk";
  job_id: string;
  sentence_index: number;
  chunk_index: number;
  sample_rate: number;
  pcm16_b64: string;
  emitted_audio_ms: number;
  preview?: boolean;
  final_chunk_of_sentence?: boolean;
  progress_step?: number;
  native_stream?: boolean;
  batch_id?: string;
}

export interface StreamDoneEvent {
  type: "done";
  result: {
    job_id: string;
    status: JobStatus;
    sample_rate: number;
    metrics: JobMetrics;
  };
}

export interface StreamStartEvent {
  type: "start";
  job_id: string;
  sentence_count: number;
  queue_position: number;
}

export interface StreamBatchEvent {
  type: "batch";
  job_id: string;
  batch_id: string;
  sentence_index: number;
  batch_size: number;
}

export interface StreamErrorEvent {
  type: "error";
  message: string;
}

export type SynthStreamEvent = StreamChunkEvent | StreamDoneEvent | StreamStartEvent | StreamBatchEvent | StreamErrorEvent;

export interface ModelOperationResponse {
  ok: boolean;
  model: string;
  warm_ms: number;
  message: string;
}

export interface ModelDownloadStatus {
  id: string;
  label: string;
  kind: string;
  status: "missing" | "partial" | "downloading" | "ready" | "error" | string;
  local_path?: string | null;
  cache_path?: string | null;
  error?: string | null;
  updated_at?: string | null;
  storage_root: string;
  approx_size_gb?: number | null;
  size_on_disk_gb?: number | null;
}

export interface ModelDownloadListResponse {
  models: ModelDownloadStatus[];
  ok?: boolean;
  job?: Record<string, unknown> | null;
  removed?: boolean | null;
  removed_path?: string | null;
}

export interface MemoryCleanupResponse {
  ok: boolean;
  message: string;
  memory_before_mb?: number | null;
  memory_after_mb?: number | null;
  memory_total_mb?: number | null;
  released_mb?: number | null;
}

export interface VllmModelsResponse {
  ok: boolean;
  base_url: string;
  models: string[];
  selected_model?: string | null;
  error?: string | null;
}

export interface BenchmarkRunResponse {
  run_id: string;
  name: string;
  status: string;
  created_at: string;
  completed_at?: string | null;
  mode: "iterations" | "traffic" | string;
  iterations: number;
  warmup_iterations: number;
  parallel_requests: number;
  duration_seconds: number;
  requests_per_minute: number;
  completion_timeout_seconds?: number;
  total_requests: number;
  exclusive: boolean;
  cases: Array<{
    label: string;
    iterations: number;
    success_count: number;
    failure_count: number;
    ttfa_ms_avg?: number | null;
    ttfa_ms_min?: number | null;
    ttfa_ms_p50?: number | null;
    ttfa_ms_p95?: number | null;
    ttfa_ms_p99?: number | null;
    ttfa_ms_max?: number | null;
    queue_wait_ms_avg?: number | null;
    queue_wait_ms_p95?: number | null;
    queue_wait_ms_p99?: number | null;
    job_wall_ms_avg?: number | null;
    job_wall_ms_min?: number | null;
    job_wall_ms_p50?: number | null;
    job_wall_ms_p95?: number | null;
    job_wall_ms_p99?: number | null;
    job_wall_ms_max?: number | null;
    realtime_x_avg?: number | null;
    audio_duration_ms_avg?: number | null;
  }>;
  results: Array<{
    iteration: number;
    warmup: boolean;
    label: string;
    success: boolean;
    metrics: JobMetrics;
    request_id?: string | null;
    scheduled_at_ms?: number | null;
    submitted_at_ms?: number | null;
    completed_at_ms?: number | null;
    sentence_count?: number | null;
    text_preview?: string | null;
    error_message?: string | null;
  }>;
  error_message?: string | null;
}

export interface WerBenchmarkRunResponse {
  run_id: string;
  name: string;
  status: string;
  created_at: string;
  completed_at?: string | null;
  count: number;
  concurrency: number;
  transcription_concurrency: number;
  seed_start?: number | null;
  seed_range: number;
  vllm_base_url: string;
  vllm_model?: string | null;
  whisper_base_url: string;
  language: string;
  min_words: number;
  max_words: number;
  tolerance_letters_per_word: number;
  completion_timeout_seconds: number;
  sentence_cache_hit: boolean;
  sentence_cache_key?: string | null;
  exclusive: boolean;
  summary: {
    total: number;
    completed: number;
    success_count: number;
    failure_count: number;
    wer_avg?: number | null;
    wer_p50?: number | null;
    wer_p95?: number | null;
    wer_max?: number | null;
    exact_count: number;
    exact_rate?: number | null;
  };
  seed_leaderboard: Array<{
    seed?: number | null;
    total: number;
    completed: number;
    success_count: number;
    failure_count: number;
    wer_avg?: number | null;
    wer_p50?: number | null;
    wer_p95?: number | null;
    wer_max?: number | null;
    exact_count: number;
    exact_rate?: number | null;
  }>;
  results: Array<{
    index: number;
    seed?: number | null;
    success: boolean;
    source_text: string;
    transcript?: string | null;
    normalized_source?: string | null;
    normalized_transcript?: string | null;
    wer?: number | null;
    word_count?: number | null;
    word_errors?: number | null;
    substitutions?: number | null;
    insertions?: number | null;
    deletions?: number | null;
    job_id?: string | null;
    metrics: JobMetrics;
    synthesis_ms?: number | null;
    transcription_ms?: number | null;
    total_ms?: number | null;
    error_message?: string | null;
  }>;
  error_message?: string | null;
}

export class ApiError extends Error {
  status?: number;
  payload?: unknown;

  constructor(message: string, status?: number, payload?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.payload = payload;
  }
}

function buildHeaders(adminKey?: string, headers: HeadersInit = {}) {
  const nextHeaders = new Headers(headers);
  if (adminKey?.trim()) {
    nextHeaders.set("X-Admin-Key", adminKey.trim());
  }
  return nextHeaders;
}

export async function apiFetch<T>(url: string, options: RequestInit & { adminKey?: string; responseType?: "json" | "blob" | "text" } = {}): Promise<T> {
  const { adminKey, responseType = "json", headers = {}, body, ...rest } = options;
  const nextHeaders = buildHeaders(adminKey, headers);
  const requestInit: RequestInit = {
    ...rest,
    headers: nextHeaders,
  };

  if (body instanceof FormData || typeof body === "string" || body === undefined) {
    requestInit.body = body;
  } else {
    nextHeaders.set("Content-Type", "application/json");
    requestInit.body = JSON.stringify(body);
  }

  const response = await fetch(url, requestInit);
  if (!response.ok) {
    const text = await response.text();
    let payload: unknown = undefined;
    let message = text || `Request failed with ${response.status}`;
    try {
      payload = text ? JSON.parse(text) : undefined;
      if (typeof (payload as { detail?: unknown })?.detail === "string") {
        message = (payload as { detail: string }).detail;
      }
    } catch {
      // keep raw text
    }
    throw new ApiError(message, response.status, payload);
  }

  if (responseType === "blob") {
    return (await response.blob()) as T;
  }
  if (responseType === "text") {
    return (await response.text()) as T;
  }
  return (await response.json()) as T;
}

export async function streamNdjson(
  url: string,
  options: {
    adminKey?: string;
    body?: unknown;
    signal?: AbortSignal;
    onEvent: (event: SynthStreamEvent) => Promise<void> | void;
  },
) {
  const headers = buildHeaders(options.adminKey);
  headers.set("Content-Type", "application/json");

  const response = await fetch(url, {
    method: "POST",
    headers,
    body: JSON.stringify(options.body ?? {}),
    signal: options.signal,
  });

  if (!response.ok) {
    const text = await response.text();
    let message = text || "Streaming request failed";
    try {
      const payload = text ? JSON.parse(text) : undefined;
      if (typeof (payload as { detail?: unknown })?.detail === "string") {
        message = (payload as { detail: string }).detail;
      }
    } catch {
      // keep raw text
    }
    throw new ApiError(message, response.status);
  }
  if (!response.body) {
    throw new Error("Streaming response does not include a readable body.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    let newlineIndex = buffer.indexOf("\n");
    while (newlineIndex >= 0) {
      const line = buffer.slice(0, newlineIndex).trim();
      buffer = buffer.slice(newlineIndex + 1);
      if (line) {
        await options.onEvent(JSON.parse(line) as SynthStreamEvent);
      }
      newlineIndex = buffer.indexOf("\n");
    }
  }

  const tail = buffer.trim();
  if (tail) {
    await options.onEvent(JSON.parse(tail) as SynthStreamEvent);
  }
}

export async function streamSse(
  url: string,
  options: {
    adminKey?: string;
    signal?: AbortSignal;
    onEvent: (event: string, payload: unknown) => Promise<void> | void;
  },
) {
  const headers = buildHeaders(options.adminKey);
  const response = await fetch(url, { headers, signal: options.signal });
  if (!response.ok) {
    const text = await response.text();
    throw new ApiError(text || "SSE request failed", response.status);
  }
  if (!response.body) {
    throw new Error("SSE response does not include a readable body.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    let separatorIndex = buffer.indexOf("\n\n");
    while (separatorIndex >= 0) {
      const block = buffer.slice(0, separatorIndex);
      buffer = buffer.slice(separatorIndex + 2);
      let eventName = "message";
      let data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) {
          eventName = line.slice(6).trim();
        }
        if (line.startsWith("data:")) {
          data += line.slice(5).trim();
        }
      }
      if (data) {
        await options.onEvent(eventName, JSON.parse(data));
      }
      separatorIndex = buffer.indexOf("\n\n");
    }
  }
}

export function decodePcm16Base64(value: string) {
  const raw = atob(value);
  const bytes = new Uint8Array(raw.length);
  for (let index = 0; index < raw.length; index += 1) {
    bytes[index] = raw.charCodeAt(index);
  }
  const int16 = new Int16Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  const float32 = new Float32Array(int16.length);
  for (let index = 0; index < int16.length; index += 1) {
    float32[index] = int16[index] / 32768;
  }
  return { bytes, int16, float32 };
}

export function createWavBlobFromInt16Chunks(chunks: Int16Array[], sampleRate: number) {
  const totalSamples = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const buffer = new ArrayBuffer(44 + totalSamples * 2);
  const view = new DataView(buffer);
  const headerText = (offset: number, text: string) => {
    for (let index = 0; index < text.length; index += 1) {
      view.setUint8(offset + index, text.charCodeAt(index));
    }
  };
  headerText(0, "RIFF");
  view.setUint32(4, 36 + totalSamples * 2, true);
  headerText(8, "WAVE");
  headerText(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  headerText(36, "data");
  view.setUint32(40, totalSamples * 2, true);
  let offset = 44;
  for (const chunk of chunks) {
    for (let index = 0; index < chunk.length; index += 1) {
      view.setInt16(offset, chunk[index], true);
      offset += 2;
    }
  }
  return new Blob([buffer], { type: "audio/wav" });
}

export function formatDate(value?: string | null) {
  if (!value) {
    return "-";
  }
  return new Intl.DateTimeFormat("de-DE", {
    year: "2-digit",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

export function formatMs(value?: number | null) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return `${Math.round(Number(value))} ms`;
}

export function formatSeconds(value?: number | null) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return `${Number(value).toFixed(1)} s`;
}

export function formatRealtime(value?: number | null) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return `${Number(value).toFixed(Number(value) >= 10 ? 0 : 2)}x`;
}

export const ADMIN_KEY_STORAGE = "g3-omnivoice-admin-key";

export function readStoredAdminKey() {
  try {
    return localStorage.getItem(ADMIN_KEY_STORAGE) || sessionStorage.getItem(ADMIN_KEY_STORAGE) || "";
  } catch {
    return "";
  }
}

export function writeStoredAdminKey(value: string) {
  try {
    localStorage.setItem(ADMIN_KEY_STORAGE, value);
    sessionStorage.setItem(ADMIN_KEY_STORAGE, value);
  } catch {
    // ignore storage errors
  }
}

export function clearStoredAdminKey() {
  try {
    localStorage.removeItem(ADMIN_KEY_STORAGE);
    sessionStorage.removeItem(ADMIN_KEY_STORAGE);
  } catch {
    // ignore storage errors
  }
}
