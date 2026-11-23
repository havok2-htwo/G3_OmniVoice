import { Fragment } from "react";

import type { RuntimeDeviceInfo, ServerSettings, VoiceItem } from "../shared/api";

// Mirrors backend capacity.py so the VRAM estimate updates live as the budget is edited.
// Batch seconds count generated audio plus each item's clone prompt (a typical 8.2 s reference).
const VRAM_BASE_MB = 2700;
const VRAM_MB_PER_AUDIO_S = 15.5;
const VRAM_EST_S_PER_CHAR = 0.07;
const VRAM_MB_PER_SEQUENCE_S = 42;
const VRAM_CHUNK_PROMPT_ALLOWANCE_S = 15;
const VRAM_TYPICAL_ITEM_S = 110 * VRAM_EST_S_PER_CHAR + 8.2; // ~110-char sentence with an 8.2 s reference
function vramEstimate(budgetMb: number): { peakGb: string; budgetS: number; chars: number; items: number } | null {
  if (!budgetMb || budgetMb <= 0) return null;
  const budgetS = Math.max(15, (budgetMb - VRAM_BASE_MB) / VRAM_MB_PER_AUDIO_S);
  const sequenceS = (budgetMb - VRAM_BASE_MB) / VRAM_MB_PER_SEQUENCE_S - VRAM_CHUNK_PROMPT_ALLOWANCE_S;
  const chars = Math.max(240, Math.floor(sequenceS / VRAM_EST_S_PER_CHAR));
  const items = Math.max(1, Math.floor(budgetS / VRAM_TYPICAL_ITEM_S));
  return { peakGb: ((VRAM_BASE_MB + budgetS * VRAM_MB_PER_AUDIO_S) / 1024).toFixed(1), budgetS: Math.round(budgetS), chars, items };
}

const VRAM_ESTIMATE_HELP = "Estimated VRAM peak and the derived limits (audio seconds per batch, characters per chunk) from the VRAM budget. Rule of thumb: about 2.7 GB model + about 15.5 MB per second of audio in the batch; every batch item brings its generated audio plus the voice's reference prompt. \"Sentences\" = batch items of about 110 characters.";

type SettingKey = keyof ServerSettings;

type FieldDef = { label: string; help: string } & (
  | { kind: "number"; fallback: number; shown?: number; min?: number; max?: number; step?: number; clamp?: boolean }
  | { kind: "optional-number"; min?: number; max?: number; step?: number; placeholder?: string }
  | { kind: "text"; placeholder?: string; also?: Partial<ServerSettings> }
  | { kind: "select"; options: Array<{ value: string | number; label: string }>; fallback: string | number }
  | { kind: "toggle"; fallback: boolean }
  | { kind: "voice" | "device" | "vllm-model" }
);

// Label, help and input of every settings field. Which of them show comes from the backend's lists
// (settings_fields of the active engine, server_settings_fields), never from this table.
const FIELDS: Partial<Record<SettingKey, FieldDef>> = {
  default_voice: { kind: "voice", label: "Default voice", help: "Voice used when a request names none; it must fit the default model. Base needs a saved custom voice." },
  preferred_device: { kind: "device", label: "GPU", help: "CUDA device of the active engine (OmniVoice, Ming and Audio8; Phonon and Audio8 ONNX run on the CPU). Numbered like nvidia-smi. Takes effect after a model reload, an engine switch or a server restart." },
  model_directory: { kind: "text", label: "Model directory", help: "Local folder for the OmniVoice model files and the Hugging Face cache." },
  allow_model_downloads: { kind: "toggle", label: "Allow model downloads", help: "Lets the server download k2-fsa/OmniVoice from Hugging Face into the local model directory.", fallback: false },
  max_parallel_requests: { kind: "number", label: "Active requests", help: "Maximum number of requests the worker runs in parallel. Higher can raise throughput but costs RAM and VRAM.", fallback: 1 },
  max_batch_size: { kind: "number", label: "Batch size", help: "Maximum number of compatible sentences or requests per real generate(text=[...]) batch.", fallback: 1 },
  batch_wait_ms: { kind: "number", label: "Batch wait (ms)", help: "How long to wait for compatible queue items to build a larger batch.", fallback: 0 },
  vram_budget_mb: { kind: "number", label: "VRAM budget (MB)", help: "Target VRAM of this process in MB. Limits audio seconds per batch and characters per sequence so large batches and long texts do not run out of memory. 0 = off (no limit). Default 7000 (batch 8 fits); 5000 for batch 4 on a shared 24 GB card. Above batch 4 the throughput with clone voices hardly grows.", fallback: 0, min: 0, step: 1000 },
  stream_chunk_ms: { kind: "number", label: "Stream chunk (ms)", help: "Length of one audio chunk in the stream. Smaller is finer but costs more overhead.", fallback: 140, min: 20, max: 1000 },
  stream_prebuffer_ms: { kind: "number", label: "Stream prebuffer (ms)", help: "Audio buffered before the first playback chunk. Higher is steadier but adds latency.", fallback: 0 },
  torch_dtype: {
    kind: "select",
    label: "Dtype",
    help: "Torch dtype when loading: fp16 is usually fast, bf16 often stable, fp32 slow and memory hungry. fp8 is experimental (fp8 tensor cores, less VRAM) and falls back to bf16 on error. Takes effect after a reload; check the audio quality with the WER benchmark.",
    fallback: "bfloat16",
    options: [
      { value: "float16", label: "fp16" },
      { value: "bfloat16", label: "bf16" },
      { value: "float32", label: "fp32" },
      { value: "fp8", label: "fp8 (experimental)" },
    ],
  },
  num_step: { kind: "optional-number", label: "Steps", help: "OmniVoice diffusion/sampling steps. More can improve quality and costs time. Default 25." },
  guidance_scale: { kind: "optional-number", label: "Guidance", help: "Guidance scale. Higher follows the prompt more closely but can add artifacts. Default 2.0.", step: 0.1 },
  duration: { kind: "optional-number", label: "Duration (s)", help: "Optional target duration in seconds. Leave empty for OmniVoice's own estimate.", step: 0.1, placeholder: "auto" },
  t_shift: { kind: "optional-number", label: "T shift", help: "OmniVoice time-shift parameter. Change it only to test sampling behaviour. Default 0.1.", step: 0.1 },
  position_temperature: { kind: "optional-number", label: "Position temperature", help: "Temperature of the position choice during token unmasking. Default 5.0.", step: 0.1 },
  class_temperature: { kind: "optional-number", label: "Class temperature", help: "Temperature of the token class choice. 0 means greedy. Default 0.0.", step: 0.1 },
  audio_chunk_duration: { kind: "optional-number", label: "Audio chunk (s)", help: "Internal OmniVoice target length of long audio chunks in seconds. Default 30.1.", step: 0.1 },
  audio_chunk_threshold: { kind: "optional-number", label: "Chunk threshold (s)", help: "Estimated duration from which OmniVoice cuts internally. Set it high for audiobooks (for example 1800) so the text is one continuous sequence; VRAM is the limit (about 7 to 9 minutes at once). Default 30.", step: 0.1 },
  compile_model: { kind: "toggle", label: "torch.compile for the LLM", help: "Compiles the LLM inside the OmniVoice model. Takes effect after a reload or restart.", fallback: false },
  sdpa_prefer_cudnn: { kind: "toggle", label: "Prefer cuDNN attention", help: "Puts cuDNN first in the SDPA kernel choice (otherwise memory-efficient). About 15 to 22% faster on recent GPUs, audio numerically slightly different. Takes effect after a reload or restart.", fallback: true },
  packed_cfg: { kind: "toggle", label: "Packed CFG (no padding)", help: "Computes one row [with voice | without voice] per item instead of two padded rows and saves the CFG padding tokens. About 20 to 33% faster on recent GPUs, audio numerically slightly different. Takes effect after a reload or restart.", fallback: true },
  stream_first_preempt: { kind: "toggle", label: "Start streams early", help: "When a streaming request arrives while a batch runs, its first sentence runs alone at the next diffusion step instead of after the batch. First audio arrives much sooner; the running batch stays bit-identical and only pauses briefly.", fallback: true },
  cudagraph_skip_dynamic_graphs: { kind: "toggle", label: "Skip dynamic CUDA graph shapes", help: "Lets Inductor skip dynamic CUDA graph shapes and silences many shape warnings.", fallback: false },
  cuda_memory_trim_after_batch: { kind: "toggle", label: "Trim VRAM when the queue is empty", help: "Empties the PyTorch CUDA cache as soon as the queue is empty after a batch, so VRAM goes back to the other services after a busy phase.", fallback: true },
  warmup_on_startup: { kind: "toggle", label: "Warmup on start", help: "Runs a warmup when the server starts. Good for the first latency, makes the start longer.", fallback: false },
  denoise: { kind: "toggle", label: "Denoise", help: "Adds the OmniVoice denoise token to the generation. Default on.", fallback: false },
  preprocess_prompt: { kind: "toggle", label: "Preprocess prompt", help: "Prepares the reference prompt and the clone audio before generating. Default on.", fallback: false },
  postprocess_output: { kind: "toggle", label: "Postprocess output", help: "Trims silence and adds short fades and padding. Default on.", fallback: false },
  ming_max_slots: { kind: "number", label: "Ming slots", help: "Ming: rows decoded at the same time (continuous batching; new requests join at the next step). More slots = more throughput and VRAM. Takes effect when Ming loads next. The quality parameters (10 flow steps, CFG 2.0) are fixed.", fallback: 1, shown: 16, min: 1, max: 64, clamp: true },
  ming_first_chunk_latents: { kind: "number", label: "First chunk (latents)", help: "Ming: latents (80 ms each) in the first stream chunk of a sentence. Small = faster first audio (TTFA).", fallback: 1, shown: 2, min: 1, max: 8, clamp: true },
  ming_max_chunk_latents: { kind: "number", label: "Later chunks (latents)", help: "Ming: latents (80 ms each) in the following stream chunks. Larger = less overhead, slightly later delivery.", fallback: 1, shown: 8, min: 1, max: 32, clamp: true },
  ming_output_sample_rate: {
    kind: "select",
    label: "Ming output rate",
    help: "Ming output sample rate. 24000 keeps the client contract (raw PCM from /v1/audio/speech is 24 kHz); 44100 is the native rate.",
    fallback: 24000,
    options: [
      { value: 24000, label: "24000 Hz (client contract)" },
      { value: 44100, label: "44100 Hz (native)" },
    ],
  },
  ming_model_dir: { kind: "text", label: "Ming model directory", help: "Folder with the Ming-3.0-tts-0.5B weights (hf download inclusionAI/Ming-3.0-tts-0.5B --local-dir ...)." },
  ming_cuda_graphs: { kind: "toggle", label: "Ming CUDA graphs", help: "Ming: CUDA graphs for the step loop (the eager loop is launch-bound). Turn off only for debugging.", fallback: true },
  phonon_lang: { kind: "text", label: "Phonon language", help: "Phonon: language of the engine, for example de. Requests cannot change it. Takes effect when Phonon loads next.", placeholder: "de" },
  phonon_quant: { kind: "text", label: "Phonon quantization", help: "Phonon: quantization of the model files, for example q8. Takes effect when Phonon loads next.", placeholder: "q8" },
  phonon_workers: { kind: "number", label: "Phonon workers", help: "Phonon: parallel engine instances on the CPU. More workers = more sentences at once and more RAM. Takes effect when Phonon loads next.", fallback: 1, shown: 2, min: 1, max: 16, clamp: true },
  phonon_threads: { kind: "number", label: "Phonon threads", help: "Phonon: Rust threads per engine. 0 = automatic from the CPU cores and the worker count. Takes effect when Phonon loads next.", fallback: 0, min: 0, max: 64, clamp: true },
  audio8_max_slots: { kind: "number", label: "Audio8 slots", help: "Audio8: rows rendered at the same time (continuous batching; new requests join at the next step). More slots = more throughput and VRAM. Takes effect when the model loads next.", fallback: 1, shown: 16, min: 1, max: 64, clamp: true },
  audio8_first_chunk_frames: { kind: "number", label: "First chunk (frames)", help: "Audio8: frames (46 ms each) in the first stream chunk of a sentence. Small = faster first audio (TTFA). Default 4. Takes effect when the model loads next.", fallback: 1, shown: 4, min: 1, max: 12, clamp: true },
  audio8_chunk_frames: { kind: "number", label: "Later chunks (frames)", help: "Audio8: frames (46 ms each) in the following stream chunks. Larger = less overhead, slightly later delivery. Default 12. Takes effect when the model loads next.", fallback: 1, shown: 12, min: 1, max: 32, clamp: true },
  audio8_output_sample_rate: {
    kind: "select",
    label: "Audio8 output rate",
    help: "Audio8 output sample rate. 44100 is native; 24000 resamples and keeps the client contract (raw PCM from /v1/audio/speech is 24 kHz). Takes effect when the model loads next.",
    fallback: 44100,
    options: [
      { value: 44100, label: "44100 Hz (native)" },
      { value: 24000, label: "24000 Hz (client contract)" },
    ],
  },
  audio8_fragment_max_chars: { kind: "number", label: "Max characters per piece", help: "Audio8: longest text piece rendered in one go (24 to 400). The model sometimes drops a clause in long sentences, so they are split first; very small pieces split sentences at awkward places. Default 200.", fallback: 200, min: 24, max: 400 },
  audio8_temperature: { kind: "optional-number", label: "Temperature", help: "Audio8: sampling temperature (above 0 up to 2). Lower = more even, higher = more variation. Default 0.8. Takes effect when the model loads next.", min: 0.05, max: 2, step: 0.05, placeholder: "0.8" },
  audio8_top_p: { kind: "optional-number", label: "Top P", help: "Audio8: share of the probability mass sampled from (above 0 up to 1). Default 0.95. Takes effect when the model loads next.", min: 0.05, max: 1, step: 0.05, placeholder: "0.95" },
  audio8_top_k: { kind: "number", label: "Top K", help: "Audio8: number of most likely tokens sampled from (1 to 1024). Default 50. Takes effect when the model loads next.", fallback: 50, min: 1, max: 1024, clamp: true },
  audio8_cuda_graphs: { kind: "toggle", label: "Audio8 CUDA graphs", help: "Audio8: CUDA graphs for the step loop. Turn off only for debugging. Takes effect when the model loads next.", fallback: true },
  audio8_onnx_workers: { kind: "number", label: "ONNX workers", help: "Audio8 ONNX: sentences rendered at the same time on the CPU. Each worker has one LM thread and its own codec threads. More workers = more sentences at once and more RAM. Takes effect when the model loads next.", fallback: 1, shown: 2, min: 1, max: 16, clamp: true },
  audio8_onnx_threads: { kind: "number", label: "ONNX codec threads", help: "Audio8 ONNX: codec threads per worker. 0 = automatic from the physical cores left next to the LM threads. Takes effect when the model loads next.", fallback: 0, min: 0, max: 64, clamp: true },
  audio8_onnx_first_chunk_frames: { kind: "number", label: "First chunk (frames)", help: "Audio8 ONNX: frames (46 ms each) in the first stream chunk. The CPU decodes much slower than a GPU, so the first chunk must last until the next one. Default 6. Takes effect when the model loads next.", fallback: 1, shown: 6, min: 1, max: 32, clamp: true },
  audio8_onnx_chunk_frames: { kind: "number", label: "Later chunks (frames)", help: "Audio8 ONNX: minimum frames (46 ms each) in the following stream chunks. Default 8. Takes effect when the model loads next.", fallback: 1, shown: 8, min: 1, max: 64, clamp: true },
  whisper_base_url: { kind: "text", label: "Whisper URL", help: "The G3 Whisper server. Host and port are enough, for example http://127.0.0.1:7861 (G3 Whisper on the same machine); a full endpoint such as /transcribe/ works too.", placeholder: "http://127.0.0.1:7861", also: { whisper_path: "" } },
  vllm_base_url: { kind: "text", label: "vLLM base URL", help: "OpenAI-compatible vLLM base URL that writes random WER reference sentences.", placeholder: "http://vllm-host:8000" },
  vllm_model: { kind: "vllm-model", label: "vLLM model", help: "vLLM model for the WER sentences. Empty = the first model from /v1/models." },
  wer_concurrency: { kind: "number", label: "WER TTS wave", help: "TTS wave size of WER benchmarks. One wave is queued together so the engine sees real batches.", fallback: 1, min: 1, max: 64 },
  wer_transcription_concurrency: { kind: "number", label: "WER ASR concurrency", help: "Whisper parallelism after the audio is generated. On a shared GPU usually 1, so ASR does not slow the TTS batches down.", fallback: 1, min: 1, max: 64 },
  queue_limit: { kind: "number", label: "Queue limit", help: "Maximum number of waiting jobs before new requests are refused with 429.", fallback: 1 },
  max_input_chars: { kind: "number", label: "Max input characters", help: "Hard limit of characters in one request; above it the server answers 413. 0 = off.", fallback: 0, min: 0, step: 1000 },
  text_optimization_default: {
    kind: "select",
    label: "Text optimization (default)",
    help: "Default for API requests without their own text_optimization field. 'deterministic' writes out dates, numbers, units, URLs and Markdown for German and English. It acts only when the request says de/en or the text is detected as German or English (spelling or typical words); other languages stay unchanged. Benchmarks, WER and finetuning always run without it.",
    fallback: "off",
    options: [
      { value: "off", label: "off" },
      { value: "deterministic", label: "deterministic (de/en)" },
    ],
  },
  sentence_chunking: { kind: "toggle", label: "Sentence chunking", help: "Splits long texts into sentences. Streams become audible sooner and sentences can be batched.", fallback: true },
};

/** The draft values of `fields`, for a preset that holds nothing of another engine. */
export function pickFields(draft: ServerSettings, fields: string[]): Partial<ServerSettings> {
  return Object.fromEntries(fields.filter((name) => name in draft).map((name) => [name, draft[name as SettingKey]]));
}

export interface SettingsFieldContext {
  voiceOptions: VoiceItem[];
  devices: RuntimeDeviceInfo[];
  vllmModels: string[];
  vllmLoading: boolean;
}

function voiceValue(voice: VoiceItem) {
  return voice.source === "custom" ? voice.voice_id : voice.name;
}

/** Renders the named settings; anything not listed stays untouched in the draft. */
export function SettingsFields({ fields, draft, context, onPatch }: {
  fields: string[];
  draft: ServerSettings;
  context: SettingsFieldContext;
  onPatch: (patch: Partial<ServerSettings>) => void;
}) {
  const known = fields.flatMap((name) => {
    const def = FIELDS[name as SettingKey];
    return def ? [{ name: name as SettingKey, def }] : [];
  });
  const inputs = known.filter(({ def }) => def.kind !== "toggle");
  const toggles = known.filter(({ def }) => def.kind === "toggle");
  // The estimate is a GPU formula: an engine without a device setting has no VRAM peak, even when the budget caps its text.
  const showEstimate = fields.includes("preferred_device");

  function renderControl(name: SettingKey, def: FieldDef) {
    const patch = (value: unknown) => onPatch({ [name]: value } as Partial<ServerSettings>);
    const raw = draft[name];
    switch (def.kind) {
      case "number": {
        const parse = (text: string) => {
          const value = Number(text) || def.fallback;
          return def.clamp ? Math.min(def.max ?? value, Math.max(def.min ?? value, value)) : value;
        };
        return <input type="number" min={def.min} max={def.max} step={def.step} value={(raw as number | undefined) ?? def.shown ?? def.fallback} onChange={(event) => patch(parse(event.target.value))} />;
      }
      case "optional-number":
        return <input type="number" min={def.min} max={def.max} step={def.step} placeholder={def.placeholder} value={(raw as number | null | undefined) ?? ""} onChange={(event) => patch(event.target.value ? Number(event.target.value) : null)} />;
      case "text":
        return <input value={String(raw ?? "")} placeholder={def.placeholder} onChange={(event) => onPatch({ [name]: event.target.value, ...def.also } as Partial<ServerSettings>)} />;
      case "select":
        return (
          <select value={(raw as string | number | undefined) ?? def.fallback} onChange={(event) => patch(typeof def.fallback === "number" ? Number(event.target.value) : event.target.value)}>
            {def.options.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
          </select>
        );
      case "voice": {
        const current = draft.default_voice;
        const match = context.voiceOptions.find((voice) => voice.voice_id === current || voice.name === current);
        return (
          <select value={match ? voiceValue(match) : current} onChange={(event) => patch(event.target.value)} disabled={!context.voiceOptions.length}>
            {/* A saved voice outside the list stays what it is; nothing rewrites it behind the admin's back. */}
            {!match ? <option value={current}>{current ? `${current} (not in the list)` : "No matching voice"}</option> : null}
            {context.voiceOptions.map((voice) => <option key={voice.voice_id} value={voiceValue(voice)}>{voice.name}</option>)}
          </select>
        );
      }
      case "device":
        return (
          <select value={draft.preferred_device} onChange={(event) => patch(event.target.value)}>
            {context.devices.map((device) => <option key={device.id} value={device.id}>{device.label}{device.available ? "" : " (not available)"}</option>)}
          </select>
        );
      case "vllm-model":
        return context.vllmModels.length > 1 ? (
          <select value={draft.vllm_model || context.vllmModels[0]} onChange={(event) => patch(event.target.value)}>
            {context.vllmModels.map((model) => <option key={model} value={model}>{model}</option>)}
          </select>
        ) : (
          <input value={draft.vllm_model || context.vllmModels[0] || ""} onChange={(event) => patch(event.target.value)} placeholder={context.vllmLoading ? "Loading models…" : "auto (first of /v1/models)"} />
        );
      default:
        return null;
    }
  }

  const estimate = vramEstimate(draft.vram_budget_mb ?? 0);
  return (
    <>
      {inputs.length ? (
        <div className="field-grid four">
          {inputs.map(({ name, def }) => (
            <Fragment key={name}>
              <label>{def.label}{renderControl(name, def)}<span className="hint">{def.help}</span></label>
              {name === "vram_budget_mb" && showEstimate ? (
                <label>Estimated VRAM peak<input type="text" readOnly value={estimate ? `~${estimate.peakGb} GB · ${estimate.budgetS} s/batch (~${estimate.items} sentences) · ${estimate.chars} chars/chunk` : "Budget off (no limit)"} /><span className="hint">{VRAM_ESTIMATE_HELP}</span></label>
              ) : null}
            </Fragment>
          ))}
        </div>
      ) : null}
      {toggles.length ? (
        <div className="field-grid two runtime-toggle-grid">
          {toggles.map(({ name, def }) => (
            <label key={name} className="checkbox-row">
              <input type="checkbox" checked={Boolean(draft[name] ?? (def.kind === "toggle" && def.fallback))} onChange={(event) => onPatch({ [name]: event.target.checked } as Partial<ServerSettings>)} />
              <span className="checkbox-text">{def.label}<span className="hint">{def.help}</span></span>
            </label>
          ))}
        </div>
      ) : null}
    </>
  );
}
