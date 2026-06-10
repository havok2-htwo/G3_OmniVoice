import { useEffect, useMemo, useRef, useState } from "react";

import {
  apiFetch,
  createWavBlobFromInt16Chunks,
  decodePcm16Base64,
  formatMs,
  formatRealtime,
  formatSeconds,
  type ModelInfo,
  type StreamDoneEvent,
  type SynthStreamEvent,
  type TaskType,
  type VoiceItem,
  streamNdjson,
} from "../shared/api";
import {
  DEFAULT_VOICE_DESIGN_INSTRUCT,
  VOICE_DESIGN_GROUPS,
  setVoiceDesignValue,
  voiceDesignInstructOrDefault,
  voiceDesignValueForGroup,
} from "../shared/voiceDesign";
import { SYNTH_LANGUAGE_OPTIONS } from "../shared/languages";

function inferTaskType(modelId: string): TaskType {
  if (modelId.endsWith("VoiceDesign")) {
    return "VoiceDesign";
  }
  if (modelId.endsWith("Base")) {
    return "Base";
  }
  return "CustomVoice";
}

function preferredDemoModel(models: ModelInfo[]) {
  return (
    preferredCustomVoiceModel(models) ||
    preferredBaseModel(models) ||
    models[0]?.model_id ||
    ""
  );
}

function preferredBaseModel(models: ModelInfo[]) {
  return (
    models.find((model) => model.model_id.endsWith("OmniVoice-Base"))?.model_id ||
    ""
  );
}

function preferredCustomVoiceModel(models: ModelInfo[]) {
  return (
    models.find((model) => model.model_id.endsWith("OmniVoice-AutoVoice"))?.model_id ||
    ""
  );
}

const DEMO_HELP = {
  model: "Waehlt den OmniVoice-Modus: AutoVoice fuer freie Stimmen, VoiceDesign fuer Instructs, Base fuer gespeicherte Clone-Stimmen.",
  voice: "Waehlt die Stimme fuer AutoVoice oder Base. Bei VoiceDesign wird die Stimme ueber Instructions beschrieben.",
  text: "Der Text wird in Saetze zerlegt, in die Queue gestellt und je nach Settings gebatcht.",
  language: "Sprachhinweis fuer OmniVoice. Auto laesst das Modell die Sprache aus dem Text erkennen; eine feste Wahl kann Aussprache und Qualitaet leicht verbessern.",
  instructions: "VoiceDesign akzeptiert nur feste Tags aus den Dropdowns. Freitext wie 'calm neutral voice' wird von OmniVoice abgelehnt.",
  seed: "Gleicher Seed kann aehnlichere Ergebnisse erzeugen. Leer lassen fuer zufaellige Generierung.",
  customVoiceButton: "Wechselt automatisch auf OmniVoice-Base und waehlt die erste gespeicherte Custom Voice.",
  streamButton: "Startet oder stoppt den Public-Streaming-Request. Audio wird live als PCM gestreamt und danach als MP3 bereitgestellt.",
};

export default function DemoApp() {
  const [voices, setVoices] = useState<VoiceItem[]>([]);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [selectedVoice, setSelectedVoice] = useState("");
  const [text, setText] = useState("Hallo! Das ist die neue offene OmniVoice-Demo mit Batch-Scheduler und Live-Streaming.");
  const [language, setLanguage] = useState("Auto");
  const [instructions, setInstructions] = useState("");
  const [seed, setSeed] = useState("");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [running, setRunning] = useState(false);
  const [queuePosition, setQueuePosition] = useState<number | null>(null);
  const [lastMetrics, setLastMetrics] = useState<StreamDoneEvent["result"]["metrics"] | null>(null);
  const [audioUrl, setAudioUrl] = useState("");
  const [sampleRate, setSampleRate] = useState(24000);
  const [ttfaMs, setTtfaMs] = useState<number | null>(null);
  const [firstPlaybackMs, setFirstPlaybackMs] = useState<number | null>(null);
  const [audioDownloadName, setAudioDownloadName] = useState("omnivoice-demo.mp3");

  const audioContextRef = useRef<AudioContext | null>(null);
  const nextPlaybackTimeRef = useRef(0);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    let cancelled = false;
    void Promise.all([
      apiFetch<{ voices: VoiceItem[] }>("/api/v1/voices"),
      apiFetch<ModelInfo[]>("/api/v1/models"),
    ])
      .then(([voicePayload, nextModels]) => {
        if (cancelled) {
          return;
        }
        setVoices(voicePayload.voices || []);
        setModels(nextModels);
        const model = preferredDemoModel(nextModels);
        setSelectedModel((current) => current || model);
      })
      .catch((loadError) => setError(loadError instanceof Error ? loadError.message : "Could not load demo data."));
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    return () => {
      if (audioContextRef.current) {
        void audioContextRef.current.close().catch(() => undefined);
      }
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
      }
    };
  }, [audioUrl]);

  const taskType = useMemo(() => inferTaskType(selectedModel), [selectedModel]);
  const visibleVoices = useMemo(() => {
    if (taskType === "Base") {
      return voices.filter((voice) => voice.source === "custom");
    }
    if (taskType === "CustomVoice") {
      return voices.filter((voice) => voice.source !== "custom");
    }
    return [];
  }, [taskType, voices]);

  const customVoices = useMemo(() => voices.filter((voice) => voice.source === "custom"), [voices]);

  useEffect(() => {
    if (taskType === "VoiceDesign") {
      setSelectedVoice("");
      setInstructions((current) => voiceDesignInstructOrDefault(current));
      return;
    }
    if (!visibleVoices.length) {
      setSelectedVoice("");
      return;
    }
    setSelectedVoice((current) => {
      if (current && visibleVoices.some((voice) => voice.voice_id === current || voice.name === current)) {
        return current;
      }
      return visibleVoices[0].source === "custom" ? visibleVoices[0].voice_id : visibleVoices[0].name;
    });
  }, [taskType, visibleVoices]);

  function handleVoiceChange(voiceId: string) {
    const voice = voices.find((item) => item.voice_id === voiceId || item.name === voiceId);
    if (voice?.source === "custom" && taskType !== "Base") {
      const baseModel = preferredBaseModel(models);
      if (baseModel) {
        setSelectedModel(baseModel);
      }
    }
    setSelectedVoice(voiceId);
  }

  async function refreshModels() {
    try {
      setModels(await apiFetch<ModelInfo[]>("/api/v1/models"));
    } catch {
      // The stream itself is the primary result; stale model badges are non-fatal.
    }
  }

  async function ensureAudioContext(nextSampleRate: number) {
    const AudioContextImpl = window.AudioContext || (window as typeof window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!AudioContextImpl) {
      throw new Error("Web Audio is not available in this browser.");
    }
    if (audioContextRef.current && audioContextRef.current.sampleRate !== nextSampleRate) {
      await audioContextRef.current.close().catch(() => undefined);
      audioContextRef.current = null;
      nextPlaybackTimeRef.current = 0;
    }
    if (!audioContextRef.current) {
      audioContextRef.current = new AudioContextImpl({ sampleRate: nextSampleRate });
      nextPlaybackTimeRef.current = 0;
    }
    if (audioContextRef.current.state === "suspended") {
      await audioContextRef.current.resume();
    }
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

  async function handleRun() {
    if (running) {
      abortRef.current?.abort();
      return;
    }

    if (!selectedModel) {
      setError("Bitte zuerst ein Modell waehlen.");
      return;
    }
    if (taskType !== "VoiceDesign" && !selectedVoice) {
      setError("Bitte zuerst eine Stimme waehlen.");
      return;
    }

    setRunning(true);
    setMessage("");
    setError("");
    setQueuePosition(null);
    setLastMetrics(null);
    setTtfaMs(null);
    setFirstPlaybackMs(null);
    nextPlaybackTimeRef.current = 0;
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
      setAudioUrl("");
    }
    setAudioDownloadName("omnivoice-demo.mp3");

    const startedAt = performance.now();
    const chunks: Int16Array[] = [];
    const controller = new AbortController();
    let seenFirstChunk = false;
    let seenFirstPlayback = false;
    let resolvedSampleRate = sampleRate;
    let completedJobId = "";
    abortRef.current = controller;
    const parsedSeed = seed.trim() ? Number(seed) : null;

    try {
      await streamNdjson("/api/v1/synthesize/stream", {
        signal: controller.signal,
        body: {
          input: text,
          model: selectedModel,
          voice: taskType === "VoiceDesign" ? null : selectedVoice,
          task_type: taskType,
          instructions: taskType === "VoiceDesign" ? voiceDesignInstructOrDefault(instructions) : instructions,
          language,
          stream: true,
          response_format: "pcm",
          seed: parsedSeed !== null && Number.isFinite(parsedSeed) ? parsedSeed : null,
        },
        onEvent: async (event: SynthStreamEvent) => {
          if (event.type === "start") {
            setQueuePosition(event.queue_position);
            completedJobId = event.job_id;
            return;
          }
          if (event.type === "chunk") {
            if (!seenFirstChunk) {
              seenFirstChunk = true;
              setTtfaMs(performance.now() - startedAt);
              void refreshModels();
            }
            resolvedSampleRate = event.sample_rate;
            setSampleRate(event.sample_rate);
            const decoded = decodePcm16Base64(event.pcm16_b64);
            chunks.push(decoded.int16);
            await queuePlayback(decoded.float32, event.sample_rate);
            if (!seenFirstPlayback) {
              seenFirstPlayback = true;
              setFirstPlaybackMs(performance.now() - startedAt);
            }
            return;
          }
          if (event.type === "done") {
            setLastMetrics(event.result.metrics);
            completedJobId = event.result.job_id || completedJobId;
            void refreshModels();
            return;
          }
          if (event.type === "error") {
            throw new Error(event.message);
          }
        },
      });

      if (completedJobId) {
        try {
          const mp3Blob = await apiFetch<Blob>(
            `/v1/jobs/${encodeURIComponent(completedJobId)}/audio?format=mp3`,
            { responseType: "blob" },
          );
          setAudioDownloadName(`${completedJobId}.mp3`);
          setAudioUrl(URL.createObjectURL(mp3Blob));
          setMessage("Stream abgeschlossen. MP3 ist bereit.");
        } catch (mp3Error) {
          if (chunks.length) {
            const wavBlob = createWavBlobFromInt16Chunks(chunks, resolvedSampleRate);
            setAudioDownloadName(`${completedJobId}.wav`);
            setAudioUrl(URL.createObjectURL(wavBlob));
          }
          setError(mp3Error instanceof Error ? `MP3-Export fehlgeschlagen: ${mp3Error.message}` : "MP3-Export fehlgeschlagen.");
          setMessage("Stream abgeschlossen. WAV-Fallback ist bereit.");
        }
      } else if (chunks.length) {
        const wavBlob = createWavBlobFromInt16Chunks(chunks, resolvedSampleRate);
        setAudioDownloadName("omnivoice-demo.wav");
        setAudioUrl(URL.createObjectURL(wavBlob));
        setMessage("Stream abgeschlossen. WAV-Fallback ist bereit.");
      } else {
        setMessage("Stream abgeschlossen.");
      }
    } catch (streamError) {
      if (!controller.signal.aborted) {
        setError(streamError instanceof Error ? streamError.message : "Streaming fehlgeschlagen.");
      }
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null;
      }
      setRunning(false);
    }
  }

  return (
    <main className="app-shell demo-shell">
      <section className="hero-card">
        <div className="hero-copy">
          <p className="eyebrow">Public Demo</p>
          <h1>OmniVoice Live Stream</h1>
          <p>
            Diese Seite nutzt nur die offenen Public-Endpoints. Kein API-Key, keine Adminfunktionen,
            einfach Modell, Stimme und Text waehlen.
          </p>
        </div>
        <div className="status-grid">
          <div className="status-pill">
            <span>Modelle</span>
            <strong>{models.length || "-"}</strong>
          </div>
          <div className="status-pill">
            <span>Voices</span>
            <strong>{visibleVoices.length || "-"}</strong>
          </div>
          <div className="status-pill">
            <span>Queue Position</span>
            <strong>{queuePosition ?? "-"}</strong>
          </div>
          <div className="status-pill">
            <span>TTFA</span>
            <strong>{formatMs(ttfaMs)}</strong>
          </div>
        </div>
        <div className="button-row">
          <a className="ghost-button" href="/">
            Landing
          </a>
          <a className="secondary-button" href="/admin">
            Adminpanel
          </a>
        </div>
      </section>

      {message ? <div className="message success">{message}</div> : null}
      {error ? <div className="message error">{error}</div> : null}

      <section className="widget-grid">
        <section className="widget span-7">
          <div className="widget-header">
            <h2>Synthese</h2>
          </div>
          <div className="field-grid two">
            <label className="with-help" title={DEMO_HELP.model}>
              Modell
              <select value={selectedModel} onChange={(event) => setSelectedModel(event.target.value)}>
                {models.map((model) => (
                  <option key={model.model_id} value={model.model_id}>
                    {model.model_id}
                  </option>
                ))}
              </select>
            </label>
            <label className="with-help" title={DEMO_HELP.voice}>
              Stimme
              <select
                value={selectedVoice}
                onChange={(event) => handleVoiceChange(event.target.value)}
                disabled={taskType === "VoiceDesign"}
              >
                <option value="">{taskType === "VoiceDesign" ? "nicht benoetigt" : "Bitte waehlen"}</option>
                {visibleVoices.map((voice) => (
                  <option key={voice.voice_id} value={voice.source === "custom" ? voice.voice_id : voice.name}>
                    {voice.name} ({voice.source})
                  </option>
                ))}
              </select>
            </label>
          </div>
          <label className="with-help" title={DEMO_HELP.language}>
            Sprache
            <select value={language} onChange={(event) => setLanguage(event.target.value)}>
              {SYNTH_LANGUAGE_OPTIONS.map((option) => (
                <option key={option} value={option}>{option}</option>
              ))}
            </select>
          </label>
          {customVoices.length > 0 && taskType !== "Base" ? (
            <div className="button-row">
              <button
                className="secondary-button"
                type="button"
                title={DEMO_HELP.customVoiceButton}
                onClick={() => {
                  const baseModel = preferredBaseModel(models);
                  if (baseModel) {
                    setSelectedModel(baseModel);
                    setSelectedVoice(customVoices[0].voice_id);
                  }
                }}
              >
                Custom Voice mit Base nutzen
              </button>
            </div>
          ) : null}
          <label className="with-help" title={DEMO_HELP.text}>
            Text
            <textarea value={text} onChange={(event) => setText(event.target.value)} />
          </label>
          <label className="with-help" title={DEMO_HELP.instructions}>
            Instructions
            <textarea
              value={instructions}
              onChange={(event) => setInstructions(event.target.value)}
              placeholder={taskType === "VoiceDesign" ? DEFAULT_VOICE_DESIGN_INSTRUCT : "Optional fuer AutoVoice."}
              readOnly={taskType === "VoiceDesign"}
            />
          </label>
          {taskType === "VoiceDesign" ? (
            <div className="voice-design-builder" title={DEMO_HELP.instructions}>
              {VOICE_DESIGN_GROUPS.map((group) => (
                <label key={group.id} className="with-help" title={DEMO_HELP.instructions}>
                  {group.label}
                  <select
                    value={voiceDesignValueForGroup(instructions, group.id)}
                    onChange={(event) => setInstructions(setVoiceDesignValue(instructions, group.id, event.target.value))}
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
          <label className="with-help" title={DEMO_HELP.seed}>
            Seed
            <input
              type="number"
              min="0"
              max="2147483647"
              value={seed}
              onChange={(event) => setSeed(event.target.value)}
              placeholder="leer = zufaellig"
            />
          </label>
          <div className="button-row">
            <button className="primary-button" type="button" onClick={handleRun} title={DEMO_HELP.streamButton}>
              {running ? "Stream stoppen" : "Live streamen"}
            </button>
          </div>
        </section>

        <section className="widget span-5 audio-card">
          <div className="widget-header">
            <h2>Ergebnis</h2>
          </div>
          <div className="metric-list">
            <div className="metric-row">
              <span>First audio</span>
              <strong>{formatMs(ttfaMs)}</strong>
            </div>
            <div className="metric-row">
              <span>First playback</span>
              <strong>{formatMs(firstPlaybackMs)}</strong>
            </div>
            <div className="metric-row">
              <span>Realtime</span>
              <strong>{formatRealtime(lastMetrics?.realtime_x)}</strong>
            </div>
            <div className="metric-row">
              <span>Audio duration</span>
              <strong>{formatSeconds((lastMetrics?.audio_duration_ms || 0) / 1000)}</strong>
            </div>
            <div className="metric-row">
              <span>Batches</span>
              <strong>{lastMetrics?.batch_count ?? "-"}</strong>
            </div>
          </div>
          {audioUrl ? (
            <>
              <audio controls src={audioUrl} />
              <div className="button-row">
                <a className="secondary-button" href={audioUrl} download={audioDownloadName}>
                  {audioDownloadName.endsWith(".mp3") ? "MP3 herunterladen" : "WAV herunterladen"}
                </a>
              </div>
            </>
          ) : <p className="widget-copy">Hier landet die fertige MP3 nach dem Stream.</p>}
        </section>

        <section className="widget span-12">
          <div className="widget-header">
            <h2>Verfuegbare Modelle</h2>
          </div>
          <div className="model-list">
            {models.map((model) => (
              <article key={model.model_id} className="model-card">
                <strong>{model.model_id}</strong>
                <div className="inline-pills">
                  <span className={`pill ${model.active ? "active" : ""}`}>{model.active ? "active" : "idle"}</span>
                  <span className={`pill ${model.loaded ? "active" : ""}`}>{model.loaded ? "loaded" : "cold"}</span>
                  {model.task_types.map((type) => (
                    <span key={type} className="pill">
                      {type}
                    </span>
                  ))}
                </div>
              </article>
            ))}
          </div>
        </section>
      </section>
    </main>
  );
}
