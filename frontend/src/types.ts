export type TabKey =
  | 'overview'
  | 'jobs'
  | 'compose'
  | 'voices'
  | 'keys'
  | 'settings';

export type BackendMode = 'mock' | 'http';
export type TaskType = 'CustomVoice' | 'VoiceDesign' | 'Base';
export type ResponseFormat = 'mp3' | 'wav' | 'pcm';
export type JobStatus = 'queued' | 'warming' | 'running' | 'streaming' | 'cancelling' | 'completed' | 'failed' | 'cancelled';
export type KeyStatus = 'active' | 'disabled';
export type VoiceKind = 'built-in' | 'custom';

export interface MetricSummary {
  avg: number;
  min: number;
  max: number;
}

export interface OverviewStats {
  activeModel: string;
  queueDepth: number;
  workerState: string;
  ttfaMsAvg: number;
  queueWaitMsAvg: number;
  jobWallMsAvg: number;
  realtimeXAvg: number;
  jobsTotal: number;
  audioSecondsTotal: number;
  gpuMemoryUsedMb: number;
  gpuMemoryTotalMb: number;
  gpuUtilizationPct: number;
  gpuTemperatureC: number;
}

export interface OverviewHistoryPoint {
  recordedAt: string;
  queueDepth: number;
  realtimeXAvg: number;
  gpuUtilizationPct: number;
  gpuMemoryUsedMb: number;
}

export interface ModelInfo {
  id: string;
  label: string;
  loaded: boolean;
  taskTypes: TaskType[];
  status: 'warm' | 'warming' | 'cold';
}

export interface JobMetrics {
  queueWaitMs: number;
  ttfaMs: number;
  audioDurationMs: number;
  jobWallMs: number;
  realtimeX: number;
}

export interface JobRecord {
  id: string;
  createdAt: string;
  updatedAt: string;
  status: JobStatus;
  model: string;
  taskType: TaskType;
  voice: string;
  language: string;
  input: string;
  instructions: string;
  responseFormat: ResponseFormat;
  stream: boolean;
  queuePosition: number;
  etaMs: number;
  progress: number;
  metrics?: JobMetrics;
  errorMessage?: string;
  audioUrl?: string;
}

export interface VoiceProfile {
  id: string;
  name: string;
  kind: VoiceKind;
  model: string;
  language: string;
  style: string;
  sampleLabel: string;
  consent: boolean;
}

export interface ApiKeyRecord {
  id: string;
  name: string;
  status: KeyStatus;
  lastUsedAt: string;
  createdAt: string;
  tokenPreview: string;
  rawKey?: string;
}

export interface ServerSettings {
  mode: BackendMode;
  baseUrl: string;
  modelDirectory: string;
  defaultModel: string;
  defaultVoice: string;
  whisperUrl: string;
  retentionDays: number;
  queueLimit: number;
  runtimeBackend: string;
  allowModelDownloads: boolean;
  preferredDevice: string;
  attentionImplementation: string;
  torchDtype: string;
  sampleRate: number;
  pollIntervalMs: number;
  theme: 'ember' | 'onyx';
  builtInVoices: string[];
  sentenceChunking: boolean;
  shortSentenceMergeMaxChars: number;
  followingSentenceMergeMinChars: number;
  maxParallelRequests: number;
  maxBatchSize: number;
  batchWaitMs: number;
  streamChunkMs: number;
  streamPrebufferMs: number;
}

export interface ComposeRequest {
  input: string;
  model: string;
  voice: string;
  taskType: TaskType;
  language: string;
  instructions: string;
  responseFormat: ResponseFormat;
  speed: number;
  stream: boolean;
  refText: string;
  refAudioLabel: string;
  xVectorOnlyMode: boolean;
}



export interface Snapshot {
  overview: OverviewStats;
  models: ModelInfo[];
  jobs: JobRecord[];
  voices: VoiceProfile[];
  keys: ApiKeyRecord[];
  settings: ServerSettings;
  transcriptionHint: string;
}

export interface DashboardClientConfig {
  mode: BackendMode;
  baseUrl: string;
  adminKey: string;
}

export interface DashboardClient {
  readonly mode: BackendMode;
  updateConfig(config: DashboardClientConfig): void;
  subscribe?(listener: () => void): () => void;
  verifyAdminKey(adminKey: string): Promise<ApiKeyRecord>;
  rotateAdminKey(): Promise<ApiKeyRecord>;
  getSnapshot(): Promise<Snapshot>;
  createSpeechJob(request: ComposeRequest): Promise<JobRecord>;
  createVoiceProfile(input: {
    name: string;
    model: string;
    language: string;
    style: string;
    consent: boolean;
    sampleLabel: string;
    refText: string;
    sampleFileName: string;
    file: File | null;
  }): Promise<VoiceProfile>;
  transcribeSample(input: { file: File | null; fileName: string }): Promise<{ transcription: string; voiceVector: number[] }>;
  createApiKey(name: string): Promise<ApiKeyRecord>;
  deleteApiKey(id: string): Promise<void>;
  cancelJob(id: string): Promise<void>;
  getJobAudio(id: string): Promise<Blob>;
  updateSettings(settings: ServerSettings): Promise<ServerSettings>;
}

export const tabLabels: Record<TabKey, string> = {
  overview: 'Overview',
  jobs: 'History',
  compose: 'Synthesis',
  voices: 'Voice Lab',
  keys: 'API Keys',
  settings: 'Settings'
};
