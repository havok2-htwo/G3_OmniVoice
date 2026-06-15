import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { apiFetch, type DashboardSnapshot } from "../shared/api";
import { SYNTH_LANGUAGE_OPTIONS } from "../shared/languages";

// --- API shapes (mirror backend finetune/schemas.py) -----------------------

interface DomainItem {
  domain_id: string;
  name: string;
  description: string;
  created_at?: string | null;
  sentence_count: number;
  sentences?: string[] | null;
}
interface DomainListResponse { domains: DomainItem[] }
interface DomainGenerateResponse { created: DomainItem[]; skipped_duplicates: number }
interface SentenceGenerateResponse { domain_id: string; added: number; skipped_duplicates: number; sentence_count: number; sample: string[] }
interface DatagenVoiceProgress { voice: string; planned: number; accepted: number; rejected: number; attempts: number }
interface DatagenRunResponse {
  run_id: string;
  status: string;
  phase: string;
  planned: number;
  accepted: number;
  rejected: number;
  attempts: number;
  pct: number;
  current?: string | null;
  voices: DatagenVoiceProgress[];
  wer_threshold: number;
  max_attempts: number;
  error_message?: string | null;
}
interface ClipItem { clip_id: string; voice: string; text: string; wer?: number | null; filename: string; size_bytes: number; created_at?: string | null }
interface ClipListResponse { voices: string[]; total: number; clips: ClipItem[] }

const RUN_TERMINAL = ["completed", "failed", "cancelled"];

interface Props {
  adminKey: string;
  voices: DashboardSnapshot["voices"];
  onMessage: (text: string) => void;
  onError: (text: string) => void;
}

export function DataGenerationPanel({ adminKey, voices, onMessage, onError }: Props) {
  const [domains, setDomains] = useState<DomainItem[]>([]);
  const [selectedDomainId, setSelectedDomainId] = useState<string>("");
  const [domainDetail, setDomainDetail] = useState<DomainItem | null>(null);

  const [newDomainName, setNewDomainName] = useState("");
  const [newDomainDesc, setNewDomainDesc] = useState("");
  const [domainGenCount, setDomainGenCount] = useState(20);
  const [sentenceGenCount, setSentenceGenCount] = useState(50);
  const [busy, setBusy] = useState<string>("");

  // Run config
  const [voiceMode, setVoiceMode] = useState<"clone" | "auto" | "both">("clone");
  const [selectedVoiceIds, setSelectedVoiceIds] = useState<string[]>([]);
  const [language, setLanguage] = useState("Deutsch");
  const [werThreshold, setWerThreshold] = useState(0);
  const [maxAttempts, setMaxAttempts] = useState(10);
  const [ttsConcurrency, setTtsConcurrency] = useState(4);
  const [asrConcurrency, setAsrConcurrency] = useState(8);
  const [tolerance, setTolerance] = useState(0);
  const [runDomainIds, setRunDomainIds] = useState<string[]>([]);

  const [run, setRun] = useState<DatagenRunResponse | null>(null);

  // Clip browser
  const [clips, setClips] = useState<ClipListResponse>({ voices: [], total: 0, clips: [] });
  const [clipFilter, setClipFilter] = useState<string>("");
  const [clipAudioUrls, setClipAudioUrls] = useState<Record<string, string>>({});
  const clipAudioUrlsRef = useRef<Record<string, string>>({});

  const customVoices = useMemo(() => voices.filter((v) => v.source === "custom"), [voices]);

  useEffect(() => { clipAudioUrlsRef.current = clipAudioUrls; }, [clipAudioUrls]);
  useEffect(() => () => { Object.values(clipAudioUrlsRef.current).forEach((url) => URL.revokeObjectURL(url)); }, []);

  const fail = useCallback((e: unknown, fallback: string) => onError(e instanceof Error ? e.message : fallback), [onError]);

  const loadDomains = useCallback(async () => {
    try {
      const data = await apiFetch<DomainListResponse>("/api/admin/finetune/domains", { adminKey });
      setDomains(data.domains);
    } catch (e) { fail(e, "Domänen konnten nicht geladen werden."); }
  }, [adminKey, fail]);

  const loadClips = useCallback(async (voice = clipFilter) => {
    try {
      const query = voice ? `?voice=${encodeURIComponent(voice)}` : "";
      setClips(await apiFetch<ClipListResponse>(`/api/admin/finetune/clips${query}`, { adminKey }));
    } catch (e) { fail(e, "Clips konnten nicht geladen werden."); }
  }, [adminKey, clipFilter, fail]);

  useEffect(() => { void loadDomains(); void loadClips(""); }, [loadDomains, loadClips]);

  // Load the selected domain's sentences.
  useEffect(() => {
    if (!selectedDomainId) { setDomainDetail(null); return; }
    void (async () => {
      try {
        setDomainDetail(await apiFetch<DomainItem>(`/api/admin/finetune/domains/${selectedDomainId}`, { adminKey }));
      } catch (e) { fail(e, "Domänendetails konnten nicht geladen werden."); }
    })();
  }, [selectedDomainId, adminKey, fail]);

  // Poll the run while it is active.
  useEffect(() => {
    if (!run || RUN_TERMINAL.includes(run.status)) return;
    const timer = window.setInterval(async () => {
      try {
        const next = await apiFetch<DatagenRunResponse>(`/api/admin/finetune/generate/status?run_id=${run.run_id}`, { adminKey });
        setRun(next);
        if (RUN_TERMINAL.includes(next.status)) { void loadClips(); }
      } catch { /* keep last */ }
    }, 1500);
    return () => window.clearInterval(timer);
  }, [run, adminKey, loadClips]);

  async function withBusy(key: string, fn: () => Promise<void>) {
    setBusy(key);
    try { await fn(); } finally { setBusy(""); }
  }

  async function handleAddDomain() {
    if (!newDomainName.trim()) return;
    await withBusy("add-domain", async () => {
      try {
        await apiFetch<DomainItem>("/api/admin/finetune/domains", {
          adminKey, method: "POST", body: { name: newDomainName.trim(), description: newDomainDesc.trim() },
        });
        setNewDomainName(""); setNewDomainDesc("");
        await loadDomains();
        onMessage("Domäne angelegt.");
      } catch (e) { fail(e, "Domäne konnte nicht angelegt werden."); }
    });
  }

  async function handleGenerateDomains() {
    await withBusy("gen-domains", async () => {
      try {
        const res = await apiFetch<DomainGenerateResponse>("/api/admin/finetune/domains/generate", {
          adminKey, method: "POST", body: { count: domainGenCount, language },
        });
        await loadDomains();
        onMessage(`${res.created.length} Domänen erzeugt (${res.skipped_duplicates} Duplikate übersprungen).`);
      } catch (e) { fail(e, "LLM-Domänengenerierung fehlgeschlagen."); }
    });
  }

  async function handleDeleteDomain(id: string) {
    await withBusy(`del-domain-${id}`, async () => {
      try {
        await apiFetch(`/api/admin/finetune/domains/${id}`, { adminKey, method: "DELETE" });
        if (selectedDomainId === id) setSelectedDomainId("");
        setRunDomainIds((cur) => cur.filter((d) => d !== id));
        await loadDomains();
      } catch (e) { fail(e, "Domäne konnte nicht gelöscht werden."); }
    });
  }

  async function handleGenerateSentences() {
    if (!selectedDomainId) return;
    await withBusy("gen-sentences", async () => {
      try {
        const res = await apiFetch<SentenceGenerateResponse>(`/api/admin/finetune/domains/${selectedDomainId}/sentences`, {
          adminKey, method: "POST", body: { count: sentenceGenCount, language },
        });
        setDomainDetail(await apiFetch<DomainItem>(`/api/admin/finetune/domains/${selectedDomainId}`, { adminKey }));
        await loadDomains();
        onMessage(`${res.added} Sätze hinzugefügt (${res.skipped_duplicates} Duplikate).`);
      } catch (e) { fail(e, "Satzgenerierung fehlgeschlagen."); }
    });
  }

  function toggleVoice(voiceId: string) {
    setSelectedVoiceIds((cur) => (cur.includes(voiceId) ? cur.filter((v) => v !== voiceId) : [...cur, voiceId]));
  }
  function toggleRunDomain(id: string) {
    setRunDomainIds((cur) => (cur.includes(id) ? cur.filter((d) => d !== id) : [...cur, id]));
  }

  async function handleStartRun() {
    if (!runDomainIds.length) { onError("Bitte mindestens eine Domäne für den Lauf auswählen."); return; }
    if (voiceMode !== "auto" && !selectedVoiceIds.length) { onError("Bitte mindestens eine Stimme auswählen (oder AutoVoice-Modus)."); return; }
    await withBusy("start-run", async () => {
      try {
        const res = await apiFetch<DatagenRunResponse>("/api/admin/finetune/generate", {
          adminKey, method: "POST",
          body: {
            domain_ids: runDomainIds,
            voice_mode: voiceMode,
            voice_ids: selectedVoiceIds,
            language,
            wer_threshold: werThreshold,
            max_attempts: maxAttempts,
            tts_concurrency: ttsConcurrency,
            transcription_concurrency: asrConcurrency,
            tolerance_letters_per_word: tolerance,
          },
        });
        setRun(res);
        onMessage(`Generierung gestartet: ${res.planned} Clips geplant.`);
      } catch (e) { fail(e, "Generierung konnte nicht gestartet werden."); }
    });
  }

  async function handleCancelRun() {
    if (!run) return;
    try { await apiFetch(`/api/admin/finetune/generate/${run.run_id}/cancel`, { adminKey, method: "POST" }); }
    catch (e) { fail(e, "Abbruch fehlgeschlagen."); }
  }

  async function handlePlayClip(clipId: string) {
    if (clipAudioUrls[clipId]) return;
    try {
      const blob = await apiFetch<Blob>(`/api/admin/finetune/clips/${clipId}/audio`, { adminKey, responseType: "blob" });
      setClipAudioUrls((cur) => ({ ...cur, [clipId]: URL.createObjectURL(blob) }));
    } catch (e) { fail(e, "Clip konnte nicht geladen werden."); }
  }

  async function handleDeleteClip(clipId: string) {
    try {
      await apiFetch(`/api/admin/finetune/clips/${clipId}`, { adminKey, method: "DELETE" });
      setClipAudioUrls((cur) => {
        if (cur[clipId]) URL.revokeObjectURL(cur[clipId]);
        const next = { ...cur }; delete next[clipId]; return next;
      });
      await loadClips();
    } catch (e) { fail(e, "Clip konnte nicht gelöscht werden."); }
  }

  const runActive = run && !RUN_TERMINAL.includes(run.status);

  return (
    <section className="widget-grid">
      {/* Domain list */}
      <section className="widget span-6">
        <div className="widget-header"><h2>Domänen</h2>
          <div className="button-row compact">
            <input type="number" min={1} max={200} value={domainGenCount} onChange={(e) => setDomainGenCount(Number(e.target.value))} style={{ width: 72 }} />
            <button className="secondary-button" type="button" disabled={busy === "gen-domains"} onClick={() => void handleGenerateDomains()}>{busy === "gen-domains" ? "..." : "N per LLM"}</button>
          </div>
        </div>
        <p className="widget-copy">Themen, an denen das Modell strauchelt (z. B. „Sätze mit GmbH"). Manuell oder per LLM (ohne Duplikate).</p>
        <div className="field-grid two">
          <label>Name<input value={newDomainName} onChange={(e) => setNewDomainName(e.target.value)} placeholder="Sätze mit GmbH" /></label>
          <label>Beschreibung<input value={newDomainDesc} onChange={(e) => setNewDomainDesc(e.target.value)} placeholder="Deutsche Sätze mit der Abkürzung GmbH" /></label>
        </div>
        <div className="button-row"><button className="primary-button" type="button" disabled={busy === "add-domain" || !newDomainName.trim()} onClick={() => void handleAddDomain()}>Domäne hinzufügen</button></div>
        <div className="job-list" style={{ marginTop: 12 }}>
          {domains.map((d) => (
            <article key={d.domain_id} className={`job-card ${selectedDomainId === d.domain_id ? "active" : ""}`}>
              <div className="inline-pills" style={{ justifyContent: "space-between" }}>
                <strong>{d.name}</strong>
                <span className="pill">{d.sentence_count} Sätze</span>
              </div>
              {d.description ? <p className="widget-copy">{d.description}</p> : null}
              <div className="button-row compact">
                <label className="pill" style={{ cursor: "pointer" }}>
                  <input type="checkbox" checked={runDomainIds.includes(d.domain_id)} onChange={() => toggleRunDomain(d.domain_id)} /> Im Lauf
                </label>
                <button className="link-chip" type="button" onClick={() => setSelectedDomainId(d.domain_id)}>Sätze</button>
                <button className="ghost-button danger-button" type="button" disabled={busy === `del-domain-${d.domain_id}`} onClick={() => void handleDeleteDomain(d.domain_id)}>Löschen</button>
              </div>
            </article>
          ))}
          {!domains.length ? <p className="widget-copy">Noch keine Domänen.</p> : null}
        </div>
      </section>

      {/* Sentences for selected domain */}
      <section className="widget span-6">
        <div className="widget-header"><h2>Sätze {domainDetail ? `· ${domainDetail.name}` : ""}</h2>
          <div className="button-row compact">
            <input type="number" min={1} max={2000} value={sentenceGenCount} onChange={(e) => setSentenceGenCount(Number(e.target.value))} style={{ width: 80 }} />
            <button className="secondary-button" type="button" disabled={!selectedDomainId || busy === "gen-sentences"} onClick={() => void handleGenerateSentences()}>{busy === "gen-sentences" ? "..." : "Sätze erzeugen"}</button>
          </div>
        </div>
        {!domainDetail ? <p className="widget-copy">Wähle links eine Domäne („Sätze"), um ihre Sätze zu sehen und per LLM zu erweitern.</p> : (
          <>
            <div className="metric-list"><div className="metric-row"><span>Gespeichert</span><strong>{domainDetail.sentence_count}</strong></div></div>
            <div className="scroll-list" style={{ maxHeight: 320, overflowY: "auto", marginTop: 8 }}>
              {(domainDetail.sentences || []).map((s, i) => (
                <div key={i} className="metric-row"><span style={{ opacity: 0.5 }}>{i + 1}</span><span style={{ textAlign: "left", flex: 1 }}>{s}</span></div>
              ))}
            </div>
          </>
        )}
      </section>

      {/* Run config */}
      <section className="widget span-6">
        <div className="widget-header"><h2>Datengenerierung starten</h2></div>
        <div className="field-grid two">
          <label>Stimm-Modus
            <select value={voiceMode} onChange={(e) => setVoiceMode(e.target.value as typeof voiceMode)}>
              <option value="clone">Geklonte Stimmen</option>
              <option value="auto">AutoVoice</option>
              <option value="both">Beides</option>
            </select>
          </label>
          <label>Sprache
            <select value={language} onChange={(e) => setLanguage(e.target.value)}>
              {SYNTH_LANGUAGE_OPTIONS.filter((o) => o !== "Auto").map((o) => <option key={o} value={o}>{o}</option>)}
            </select>
          </label>
          <label>WER-Schwelle<input type="number" step={0.01} min={0} max={1} value={werThreshold} onChange={(e) => setWerThreshold(Number(e.target.value))} /></label>
          <label>Max. Versuche<input type="number" min={1} max={50} value={maxAttempts} onChange={(e) => setMaxAttempts(Number(e.target.value))} /></label>
          <label>TTS-Parallelität<input type="number" min={1} max={64} value={ttsConcurrency} onChange={(e) => setTtsConcurrency(Number(e.target.value))} /></label>
          <label>ASR-Parallelität<input type="number" min={1} max={64} value={asrConcurrency} onChange={(e) => setAsrConcurrency(Number(e.target.value))} /></label>
          <label>WER-Toleranz (Buchst./Wort)<input type="number" min={0} max={8} value={tolerance} onChange={(e) => setTolerance(Number(e.target.value))} /></label>
        </div>
        {voiceMode !== "auto" ? (
          <div style={{ marginTop: 8 }}>
            <span className="field-label">Stimmen ({selectedVoiceIds.length} gewählt)</span>
            <div className="inline-pills" style={{ flexWrap: "wrap" }}>
              {customVoices.map((v) => (
                <label key={v.voice_id} className={`pill ${selectedVoiceIds.includes(v.voice_id) ? "active" : ""}`} style={{ cursor: "pointer" }}>
                  <input type="checkbox" checked={selectedVoiceIds.includes(v.voice_id)} onChange={() => toggleVoice(v.voice_id)} /> {v.name}
                </label>
              ))}
              {!customVoices.length ? <span className="widget-copy">Keine gespeicherten Stimmen — lege im Overview-Tab welche an oder nutze AutoVoice.</span> : null}
            </div>
          </div>
        ) : null}
        <div className="button-row" style={{ marginTop: 12 }}>
          <button className="primary-button" type="button" disabled={busy === "start-run" || !!runActive} onClick={() => void handleStartRun()}>Lauf starten ({runDomainIds.length} Domänen)</button>
          {runActive ? <button className="ghost-button danger-button" type="button" onClick={() => void handleCancelRun()}>Stoppen</button> : null}
        </div>
      </section>

      {/* Run progress */}
      <section className="widget span-6">
        <div className="widget-header"><h2>Lauf-Fortschritt</h2></div>
        {!run ? <p className="widget-copy">Noch kein Lauf gestartet.</p> : (
          <>
            <div className="metric-list">
              <div className="metric-row"><span>Status</span><strong>{run.status} · {run.phase}</strong></div>
              <div className="metric-row"><span>Fortschritt</span><strong>{run.pct}% ({run.accepted + run.rejected}/{run.planned})</strong></div>
              <div className="metric-row"><span>Akzeptiert</span><strong>{run.accepted}</strong></div>
              <div className="metric-row"><span>Verworfen</span><strong>{run.rejected}</strong></div>
              <div className="metric-row"><span>Versuche gesamt</span><strong>{run.attempts}</strong></div>
              {run.current ? <div className="metric-row"><span>Aktuell</span><strong style={{ fontWeight: 400 }}>{run.current}</strong></div> : null}
              {run.error_message ? <div className="metric-row"><span>Fehler</span><strong>{run.error_message}</strong></div> : null}
            </div>
            <div className="ft-progress" style={{ marginTop: 10 }}><div className="ft-progress-fill" style={{ width: `${run.pct}%` }} /></div>
            <div className="job-list" style={{ marginTop: 10 }}>
              {run.voices.map((v) => (
                <article key={v.voice} className="job-card">
                  <strong>{v.voice}</strong>
                  <div className="inline-pills"><span className="pill">{v.accepted}/{v.planned} ok</span><span className="pill">{v.rejected} verworfen</span><span className="pill">{v.attempts} Versuche</span></div>
                </article>
              ))}
            </div>
          </>
        )}
      </section>

      {/* Human-eval clip browser */}
      <section className="widget span-12">
        <div className="widget-header"><h2>Clip-Browser (Human-Eval)</h2>
          <div className="button-row compact">
            <select value={clipFilter} onChange={(e) => { setClipFilter(e.target.value); void loadClips(e.target.value); }}>
              <option value="">Alle Stimmen</option>
              {clips.voices.map((v) => <option key={v} value={v}>{v}</option>)}
            </select>
            <button className="secondary-button" type="button" onClick={() => void loadClips()}>Aktualisieren</button>
          </div>
        </div>
        <p className="widget-copy">Anhören und falsch klingende Clips löschen — das entfernt Audio + Transkript (.txt). {clips.total} Clips.</p>
        <div className="job-list">
          {clips.clips.map((clip) => (
            <article key={clip.clip_id} className="job-card">
              <div className="inline-pills" style={{ justifyContent: "space-between" }}>
                <strong>{clip.voice} · {clip.filename}</strong>
                <span className="pill">{clip.wer != null ? `WER ${(clip.wer * 100).toFixed(0)}%` : "WER -"}</span>
              </div>
              <p style={{ textAlign: "left" }}>{clip.text}</p>
              {clipAudioUrls[clip.clip_id] ? <audio controls src={clipAudioUrls[clip.clip_id]} /> : null}
              <div className="button-row compact">
                {!clipAudioUrls[clip.clip_id] ? <button className="secondary-button" type="button" onClick={() => void handlePlayClip(clip.clip_id)}>Abspielen</button> : null}
                <button className="ghost-button danger-button" type="button" onClick={() => void handleDeleteClip(clip.clip_id)}>Löschen</button>
              </div>
            </article>
          ))}
          {!clips.clips.length ? <p className="widget-copy">Noch keine Clips erzeugt.</p> : null}
        </div>
      </section>
    </section>
  );
}
