import { Fragment } from "react";

import type { RuntimeDeviceInfo, ServerSettings, VoiceItem } from "../shared/api";

// Mirrors backend capacity.py so the VRAM estimate updates live as the budget is edited.
// Batch seconds count generated audio plus each item's clone prompt (SonjaV2 8.2 s).
const VRAM_BASE_MB = 2700;
const VRAM_MB_PER_AUDIO_S = 15.5;
const VRAM_EST_S_PER_CHAR = 0.07;
const VRAM_MB_PER_SEQUENCE_S = 42;
const VRAM_CHUNK_PROMPT_ALLOWANCE_S = 15;
const VRAM_TYPICAL_ITEM_S = 110 * VRAM_EST_S_PER_CHAR + 8.2; // ~110-char sentence with SonjaV2
function vramEstimate(budgetMb: number): { peakGb: string; budgetS: number; chars: number; items: number } | null {
  if (!budgetMb || budgetMb <= 0) return null;
  const budgetS = Math.max(15, (budgetMb - VRAM_BASE_MB) / VRAM_MB_PER_AUDIO_S);
  const sequenceS = (budgetMb - VRAM_BASE_MB) / VRAM_MB_PER_SEQUENCE_S - VRAM_CHUNK_PROMPT_ALLOWANCE_S;
  const chars = Math.max(240, Math.floor(sequenceS / VRAM_EST_S_PER_CHAR));
  const items = Math.max(1, Math.floor(budgetS / VRAM_TYPICAL_ITEM_S));
  return { peakGb: ((VRAM_BASE_MB + budgetS * VRAM_MB_PER_AUDIO_S) / 1024).toFixed(1), budgetS: Math.round(budgetS), chars, items };
}

const VRAM_ESTIMATE_HELP = "Geschaetzter VRAM-Peak und die abgeleiteten Limits (Audio-Sekunden pro Batch, Zeichen pro Chunk) aus dem aktuellen VRAM-Budget. Faustformel (gemessen auf der RTX 5090): ~2,7 GB Modell + ~15,5 MB pro Sekunde Audio im Batch, wobei jeder Batch-Eintrag sein erzeugtes Audio plus den Referenz-Prompt der Stimme mitbringt (SonjaV2 8,2 s, Sonja 14,4 s). \"Saetze\" = Batch-Eintraege mit ~110 Zeichen und SonjaV2.";

type SettingKey = keyof ServerSettings;

type FieldDef = { label: string; help: string } & (
  | { kind: "number"; fallback: number; shown?: number; min?: number; max?: number; step?: number; clamp?: boolean }
  | { kind: "optional-number"; step?: number; placeholder?: string }
  | { kind: "text"; placeholder?: string; also?: Partial<ServerSettings> }
  | { kind: "select"; options: Array<{ value: string | number; label: string }>; fallback: string | number }
  | { kind: "toggle"; fallback: boolean }
  | { kind: "voice" | "device" | "vllm-model" }
);

// Label, help and input of every settings field. Which of them show comes from the backend's lists
// (settings_fields of the active engine, server_settings_fields), never from this table.
const FIELDS: Partial<Record<SettingKey, FieldDef>> = {
  default_voice: { kind: "voice", label: "Default voice", help: "Default-Stimme passend zum Default-Modell. Base braucht eine gespeicherte Custom Voice." },
  preferred_device: { kind: "device", label: "GPU", help: "CUDA-Geraet fuer die aktive Engine (OmniVoice und Ming; Phonon laeuft auf der CPU). Windows-GPU-Nummern und CUDA-Nummern koennen sich unterscheiden; relevant ist der NVIDIA/CUDA-Name. Erst nach Model Reload, Engine-Wechsel oder Server-Neustart aktiv." },
  model_directory: { kind: "text", label: "Model directory", help: "Lokaler Ordner fuer OmniVoice-Modelldateien und Hugging-Face-Cache." },
  allow_model_downloads: { kind: "toggle", label: "Model-Downloads erlauben", help: "Erlaubt dem Backend, k2-fsa/OmniVoice aus Hugging Face in das lokale models-Verzeichnis zu laden.", fallback: false },
  max_parallel_requests: { kind: "number", label: "Active requests", help: "Maximale Anzahl paralleler Worker-Requests. Hoeher kann Durchsatz steigern, kostet aber RAM/VRAM.", fallback: 1 },
  max_batch_size: { kind: "number", label: "Batch size", help: "Maximale Anzahl kompatibler Saetze/Requests pro echtem generate(text=[...]) Batch.", fallback: 1 },
  batch_wait_ms: { kind: "number", label: "Batch wait ms", help: "Wartezeit, um kompatible Queue-Items fuer einen groesseren Batch zu sammeln.", fallback: 0 },
  vram_budget_mb: { kind: "number", label: "VRAM budget MB", help: "Ziel-VRAM-Budget in MB. Daraus werden max. Audio-Sekunden pro Batch und max. Zeichen pro Sequenz abgeleitet, damit grosse Batches/lange Texte nicht ins OOM laufen. 0 = aus (unbegrenzt). Obergrenze fuer den VRAM dieses Prozesses. Default: 7000 (Batch 8 passt; gemessen ~5 GB Peak mit Sonja, SonjaV2 braucht weniger); 5000 fuer Batch 4 auf geteilten 24-GB-Karten. Ab Batch 4 steigt der Durchsatz mit Klon-Stimmen kaum noch (~35x Echtzeit).", fallback: 0, min: 0, step: 1000 },
  stream_chunk_ms: { kind: "number", label: "Stream chunk ms", help: "Laenge eines Audio-Chunks im Stream. Kleiner liefert feiner, kostet aber mehr Overhead.", fallback: 140, min: 20, max: 1000 },
  stream_prebuffer_ms: { kind: "number", label: "Stream prebuffer ms", help: "Audio-Puffer vor dem ersten Playback-Chunk. Hoeher stabilisiert Playback, erhoeht aber Latenz.", fallback: 0 },
  torch_dtype: {
    kind: "select",
    label: "Dtype",
    help: "Torch dtype beim Modell-Laden: fp16 meist schnell, bf16 oft stabil, fp32 langsam und speicherhungrig. fp8 = experimentell (RTX 50xx fp8-Tensor-Cores, weniger VRAM); faellt bei Fehler automatisch auf bf16 zurueck. Erst nach Reload aktiv; Audioqualitaet unbedingt per WER-Benchmark pruefen.",
    fallback: "bfloat16",
    options: [
      { value: "float16", label: "fp16" },
      { value: "bfloat16", label: "bf16" },
      { value: "float32", label: "fp32" },
      { value: "fp8", label: "fp8 (experimentell)" },
    ],
  },
  num_step: { kind: "optional-number", label: "Num step", help: "OmniVoice Diffusion-/Sampling-Schritte. Mehr kann Qualitaet verbessern, kostet Zeit. Default: 25." },
  guidance_scale: { kind: "optional-number", label: "Guidance", help: "Guidance Scale fuer die Steuerstaerke. Hoeher folgt Prompts staerker, kann aber Artefakte erzeugen. Default: 2.0.", step: 0.1 },
  duration: { kind: "optional-number", label: "Duration", help: "Optionale Ziel-Dauer in Sekunden. Leer lassen fuer OmniVoice-Automatik. Default: auto.", step: 0.1, placeholder: "auto" },
  t_shift: { kind: "optional-number", label: "T shift", help: "OmniVoice Zeitverschiebungsparameter. Nur aendern, wenn du gezielt Sampling-Verhalten testest. Default: 0.1.", step: 0.1 },
  position_temperature: { kind: "optional-number", label: "Position temp", help: "Temperatur fuer die Positionsauswahl beim Token-Unmasking. Default: 5.0.", step: 0.1 },
  class_temperature: { kind: "optional-number", label: "Class temp", help: "Temperatur fuer die Token-Klassenwahl. 0 bedeutet greedy. Default: 0.0.", step: 0.1 },
  audio_chunk_duration: { kind: "optional-number", label: "Audio chunk sec", help: "Interne OmniVoice-Ziellaenge fuer lange Audio-Chunks in Sekunden. Default: 30.1.", step: 0.1 },
  audio_chunk_threshold: { kind: "optional-number", label: "Chunk threshold sec", help: "Ab welcher geschaetzten Dauer (Sek.) OmniVoice intern in Stuecke schneidet. Fuer Hoerbuecher hoch setzen (z.B. 1800), damit der Text als EINE durchgehende Sequenz mit Betonung erzeugt wird. Grenze ist der VRAM (~7-9 min am Stueck). Default: 30.", step: 0.1 },
  compile_model: { kind: "toggle", label: "torch.compile fuer llm", help: "Aktiviert torch.compile fuer das LLM im OmniVoice-Modell. Erst nach Reload/Restart wirksam.", fallback: false },
  sdpa_prefer_cudnn: { kind: "toggle", label: "cuDNN-Attention bevorzugen", help: "Setzt cuDNN an die erste Stelle der SDPA-Kernelwahl (sonst mem-efficient). Auf der RTX 5090 etwa 15-22 % schneller, Audio numerisch minimal anders. Erst nach Reload/Restart wirksam.", fallback: true },
  packed_cfg: { kind: "toggle", label: "CFG gepackt (ohne Padding)", help: "Rechnet pro Item eine Zeile [mit Stimme | ohne Stimme] statt zwei aufgefuellter Zeilen, spart die Fuell-Tokens der CFG. Auf der RTX 5090 etwa 20-33 % schneller, Audio numerisch minimal anders. Erst nach Reload/Restart wirksam.", fallback: true },
  stream_first_preempt: { kind: "toggle", label: "Stream-Start vorziehen (Aufspringen)", help: "Kommt eine Streaming-Anfrage, waehrend ein Batch rechnet, wird ihr erster Satz am naechsten Diffusionsschritt solo dazwischengeschoben statt nach dem Batch. Erstes Audio deutlich frueher, der laufende Batch bleibt bitgleich und pausiert nur kurz.", fallback: true },
  cudagraph_skip_dynamic_graphs: { kind: "toggle", label: "CUDAGraph dynamic shapes skippen", help: "Laesst Inductor dynamische CUDAGraph-Shapes ueberspringen und unterdrueckt viele Shape-Warnungen.", fallback: false },
  cuda_memory_trim_after_batch: { kind: "toggle", label: "Auto VRAM trim wenn Queue leer", help: "Leert den PyTorch CUDA Cache, sobald die Queue nach einem Batch leer ist (nicht mehr nach jedem Batch, das kostete 8-11 ms pro Batch). Gibt VRAM nach einer Lastphase an die anderen Dienste zurueck.", fallback: true },
  warmup_on_startup: { kind: "toggle", label: "Warmup beim Start", help: "Fuehrt beim Start automatisch Warmup aus. Gut fuer erste Latenz, verlaengert Startup.", fallback: false },
  denoise: { kind: "toggle", label: "Denoise", help: "Fuegt den OmniVoice-Denoise-Token in die Generierung ein. Default: an.", fallback: false },
  preprocess_prompt: { kind: "toggle", label: "Preprocess prompt", help: "Bereitet Referenzprompt und Clone-Audio vor dem Generieren auf. Default: an.", fallback: false },
  postprocess_output: { kind: "toggle", label: "Postprocess output", help: "Entfernt/saeubert Stille und fuegt kleine Fades/Pads an. Default: an.", fallback: false },
  ming_max_slots: { kind: "number", label: "Ming slots", help: "Ming: Zeilen, die gleichzeitig decodiert werden (kontinuierliches Batching, neue Anfragen steigen am naechsten Schritt ein). Mehr Slots = mehr Durchsatz und VRAM. Aktiv beim naechsten Laden von Ming. Die Qualitaetsparameter (flow steps 10, CFG 2.0) sind fest.", fallback: 1, shown: 16, min: 1, max: 64, clamp: true },
  ming_first_chunk_latents: { kind: "number", label: "Erster Chunk (Latents)", help: "Ming: Latents (je 80 ms) im ersten Stream-Chunk eines Satzes. Klein = schnelle erste Audiodaten (TTFA).", fallback: 1, shown: 2, min: 1, max: 8, clamp: true },
  ming_max_chunk_latents: { kind: "number", label: "Folge-Chunks (Latents)", help: "Ming: Latents (je 80 ms) in den folgenden Stream-Chunks. Groesser = weniger Overhead, etwas spaetere Lieferung.", fallback: 1, shown: 8, min: 1, max: 32, clamp: true },
  ming_output_sample_rate: {
    kind: "select",
    label: "Ming Ausgaberate",
    help: "Ming-Ausgaberate. 24000 haelt den Client-Vertrag (Raw-PCM von /v1/audio/speech ist 24 kHz); 44100 ist die native Rate.",
    fallback: 24000,
    options: [
      { value: 24000, label: "24000 Hz (Client-Vertrag)" },
      { value: 44100, label: "44100 Hz (nativ)" },
    ],
  },
  ming_model_dir: { kind: "text", label: "Ming model dir", help: "Ordner mit den Ming-3.0-tts-0.5B Gewichten (hf download inclusionAI/Ming-3.0-tts-0.5B --local-dir ...)." },
  ming_cuda_graphs: { kind: "toggle", label: "Ming CUDA Graphs", help: "Ming: CUDA Graphs fuer die Schritt-Schleife (die eager-Schleife ist launch-gebunden). Nur zum Debuggen abschalten.", fallback: true },
  phonon_lang: { kind: "text", label: "Phonon Sprache", help: "Phonon: Sprache der Engine, z.B. de. Requests koennen sie nicht aendern. Aktiv beim naechsten Laden von Phonon.", placeholder: "de" },
  phonon_quant: { kind: "text", label: "Phonon Quantisierung", help: "Phonon: Quantisierung der Modelldateien, z.B. q8. Aktiv beim naechsten Laden von Phonon.", placeholder: "q8" },
  phonon_workers: { kind: "number", label: "Phonon Worker", help: "Phonon: parallele Engine-Instanzen auf der CPU. Mehr Worker = mehr gleichzeitige Saetze und mehr RAM. Aktiv beim naechsten Laden von Phonon.", fallback: 1, shown: 2, min: 1, max: 16, clamp: true },
  phonon_threads: { kind: "number", label: "Phonon Threads", help: "Phonon: Rust-Threads pro Engine. 0 = automatisch aus den CPU-Kernen und der Worker-Zahl. Aktiv beim naechsten Laden von Phonon.", fallback: 0, min: 0, max: 64, clamp: true },
  whisper_base_url: { kind: "text", label: "Whisper URL", help: "Einzige Whisper URL. Host/Port reicht, z.B. http://127.0.0.1:7861 (Default, G3_WHISPER auf demselben Rechner); ein voller Endpoint wie /transcribe/ geht auch.", placeholder: "http://127.0.0.1:7861", also: { whisper_path: "" } },
  vllm_base_url: { kind: "text", label: "vLLM Base URL", help: "OpenAI-kompatible vLLM Base URL fuer zufaellige WER-Referenzsaetze. Default: http://192.168.20.126:8000.", placeholder: "http://192.168.20.126:8000" },
  vllm_model: { kind: "vllm-model", label: "vLLM Model", help: "vLLM Modell fuer WER-Saetze. Default: Qwen/Qwen3.6-35B-A3B-FP8. Leer lassen nutzt das erste Modell aus /v1/models." },
  wer_concurrency: { kind: "number", label: "WER TTS wave", help: "TTS-Wellengroesse fuer WER-Benchmarks. Eine Welle wird zusammen in die Queue gelegt, damit die Engine echte Batches sieht.", fallback: 1, min: 1, max: 64 },
  wer_transcription_concurrency: { kind: "number", label: "WER ASR concurrency", help: "Whisper-Parallelitaet nach der Audio-Erzeugung. Auf demselben GPU-System meist 1 lassen, damit ASR die TTS-Batches nicht ausbremst.", fallback: 1, min: 1, max: 64 },
  queue_limit: { kind: "number", label: "Queue limit", help: "Maximale Anzahl wartender Jobs, bevor neue Requests mit 429 abgelehnt werden.", fallback: 1 },
  max_input_chars: { kind: "number", label: "Max input chars", help: "Harte Obergrenze fuer die Zeichenzahl eines einzelnen Requests. Darueber kommt 413. 0 = aus.", fallback: 0, min: 0, step: 1000 },
  text_optimization_default: {
    kind: "select",
    label: "Textoptimierung (Default)",
    help: "Standard fuer API-Requests ohne eigenes text_optimization-Feld. 'deterministic' schreibt Datum, Zahlen, Einheiten, URLs und Markdown fuer de/en aus. Wirkt nur, wenn der Request de/en angibt oder der Text bei Auto/ohne Sprache als Deutsch/Englisch erkannt wird (Schreibweise oder typische Woerter); andere Sprachen bleiben unveraendert. Benchmarks, WER und Finetune bleiben immer aus.",
    fallback: "off",
    options: [
      { value: "off", label: "off" },
      { value: "deterministic", label: "deterministic (de/en)" },
    ],
  },
  sentence_chunking: { kind: "toggle", label: "Sentence chunking aktiv", help: "Zerlegt lange Texte in Saetze. Das macht Streaming frueher hoerbar und erlaubt Satz-Batching.", fallback: true },
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
        return <input type="number" step={def.step} placeholder={def.placeholder} value={(raw as number | null | undefined) ?? ""} onChange={(event) => patch(event.target.value ? Number(event.target.value) : null)} />;
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
            {!match ? <option value={current}>{current ? `${current} (nicht in der Liste)` : "Keine passende Stimme"}</option> : null}
            {context.voiceOptions.map((voice) => <option key={voice.voice_id} value={voiceValue(voice)}>{voice.name}</option>)}
          </select>
        );
      }
      case "device":
        return (
          <select value={draft.preferred_device} onChange={(event) => patch(event.target.value)}>
            {context.devices.map((device) => <option key={device.id} value={device.id}>{device.label}{device.available ? "" : " (nicht verfuegbar)"}</option>)}
          </select>
        );
      case "vllm-model":
        return context.vllmModels.length > 1 ? (
          <select value={draft.vllm_model || context.vllmModels[0]} onChange={(event) => patch(event.target.value)}>
            {context.vllmModels.map((model) => <option key={model} value={model}>{model}</option>)}
          </select>
        ) : (
          <input value={draft.vllm_model || context.vllmModels[0] || ""} onChange={(event) => patch(event.target.value)} placeholder={context.vllmLoading ? "lade Modelle..." : "auto aus /v1/models"} />
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
              <label className="with-help" title={def.help}>{def.label}{renderControl(name, def)}</label>
              {name === "vram_budget_mb" && showEstimate ? (
                <label className="with-help" title={VRAM_ESTIMATE_HELP}>Est. VRAM peak<input type="text" readOnly value={estimate ? `~${estimate.peakGb} GB · ${estimate.budgetS}s/Batch (~${estimate.items} Saetze) · ${estimate.chars} Zeichen/Chunk` : "Budget aus (unbegrenzt)"} /></label>
              ) : null}
            </Fragment>
          ))}
        </div>
      ) : null}
      {toggles.length ? (
        <div className="field-grid two runtime-toggle-grid">
          {toggles.map(({ name, def }) => (
            <label key={name} className="checkbox-row with-help" title={def.help}>
              <input type="checkbox" checked={Boolean(draft[name] ?? (def.kind === "toggle" && def.fallback))} onChange={(event) => onPatch({ [name]: event.target.checked } as Partial<ServerSettings>)} />
              {def.label}
            </label>
          ))}
        </div>
      ) : null}
    </>
  );
}
