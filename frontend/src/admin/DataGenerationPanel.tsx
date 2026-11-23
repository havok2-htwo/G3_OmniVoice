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
    } catch (e) { fail(e, "Could not load the domains."); }
  }, [adminKey, fail]);

  const loadClips = useCallback(async (voice = clipFilter) => {
    try {
      const query = voice ? `?voice=${encodeURIComponent(voice)}` : "";
      setClips(await apiFetch<ClipListResponse>(`/api/admin/finetune/clips${query}`, { adminKey }));
    } catch (e) { fail(e, "Could not load the clips."); }
  }, [adminKey, clipFilter, fail]);

  useEffect(() => { void loadDomains(); void loadClips(""); }, [loadDomains, loadClips]);

  // Load the selected domain's sentences.
  useEffect(() => {
    if (!selectedDomainId) { setDomainDetail(null); return; }
    void (async () => {
      try {
        setDomainDetail(await apiFetch<DomainItem>(`/api/admin/finetune/domains/${selectedDomainId}`, { adminKey }));
      } catch (e) { fail(e, "Could not load the domain."); }
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
        onMessage("Domain added.");
      } catch (e) { fail(e, "Could not add the domain."); }
    });
  }

  async function handleGenerateDomains() {
    await withBusy("gen-domains", async () => {
      try {
        const res = await apiFetch<DomainGenerateResponse>("/api/admin/finetune/domains/generate", {
          adminKey, method: "POST", body: { count: domainGenCount, language },
        });
        await loadDomains();
        onMessage(`${res.created.length} domains created (${res.skipped_duplicates} duplicates skipped).`);
      } catch (e) { fail(e, "Could not generate domains with the LLM."); }
    });
  }

  async function handleDeleteDomain(id: string) {
    const domain = domains.find((item) => item.domain_id === id);
    if (!window.confirm(`Delete domain "${domain?.name ?? id}"?\n\nIts ${domain?.sentence_count ?? 0} sentences are deleted too.`)) return;
    await withBusy(`del-domain-${id}`, async () => {
      try {
        await apiFetch(`/api/admin/finetune/domains/${id}`, { adminKey, method: "DELETE" });
        if (selectedDomainId === id) setSelectedDomainId("");
        setRunDomainIds((cur) => cur.filter((d) => d !== id));
        await loadDomains();
      } catch (e) { fail(e, "Could not delete the domain."); }
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
        onMessage(`${res.added} sentences added (${res.skipped_duplicates} duplicates skipped).`);
      } catch (e) { fail(e, "Could not generate sentences."); }
    });
  }

  function toggleVoice(voiceId: string) {
    setSelectedVoiceIds((cur) => (cur.includes(voiceId) ? cur.filter((v) => v !== voiceId) : [...cur, voiceId]));
  }
  function toggleRunDomain(id: string) {
    setRunDomainIds((cur) => (cur.includes(id) ? cur.filter((d) => d !== id) : [...cur, id]));
  }

  async function handleStartRun() {
    if (!runDomainIds.length) { onError("Select at least one domain for the run."); return; }
    if (voiceMode !== "auto" && !selectedVoiceIds.length) { onError("Select at least one voice (or use the AutoVoice mode)."); return; }
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
        onMessage(`Generation started: ${res.planned} clips planned.`);
      } catch (e) { fail(e, "Could not start the generation."); }
    });
  }

  async function handleCancelRun() {
    if (!run) return;
    if (!window.confirm("Stop the data generation run?\n\nClips accepted so far stay.")) return;
    try { await apiFetch(`/api/admin/finetune/generate/${run.run_id}/cancel`, { adminKey, method: "POST" }); }
    catch (e) { fail(e, "Could not stop the run."); }
  }

  async function handlePlayClip(clipId: string) {
    if (clipAudioUrls[clipId]) return;
    try {
      const blob = await apiFetch<Blob>(`/api/admin/finetune/clips/${clipId}/audio`, { adminKey, responseType: "blob" });
      setClipAudioUrls((cur) => ({ ...cur, [clipId]: URL.createObjectURL(blob) }));
    } catch (e) { fail(e, "Could not load the clip."); }
  }

  async function handleDeleteClip(clipId: string) {
    const clip = clips.clips.find((item) => item.clip_id === clipId);
    if (!window.confirm(`Delete clip "${clip?.filename ?? clipId}"?\n\nThe audio and its transcript are removed from the dataset.`)) return;
    try {
      await apiFetch(`/api/admin/finetune/clips/${clipId}`, { adminKey, method: "DELETE" });
      setClipAudioUrls((cur) => {
        if (cur[clipId]) URL.revokeObjectURL(cur[clipId]);
        const next = { ...cur }; delete next[clipId]; return next;
      });
      await loadClips();
    } catch (e) { fail(e, "Could not delete the clip."); }
  }

  const runActive = run && !RUN_TERMINAL.includes(run.status);

  return (
    <div className="grid grid-2 tab-grid">
      <section className="panel" aria-labelledby="dg-domains">
        <div className="panel-head">
          <h2 id="dg-domains">Domains</h2>
          <p className="hint">Topics the model stumbles on (for example "sentences with GmbH"). Add them by hand or let the LLM suggest new ones (no duplicates).</p>
        </div>
        <form className="field-grid two" onSubmit={(e) => { e.preventDefault(); void handleAddDomain(); }}>
          <label>Name<input value={newDomainName} onChange={(e) => setNewDomainName(e.target.value)} placeholder="Sentences with GmbH" /></label>
          <label>Description<input value={newDomainDesc} onChange={(e) => setNewDomainDesc(e.target.value)} placeholder="German sentences with the abbreviation GmbH" /></label>
          <div className="button-row span-all">
            <button type="submit" disabled={busy === "add-domain" || !newDomainName.trim()} aria-busy={busy === "add-domain"}>Add domain</button>
          </div>
        </form>
        <div className="button-row">
          <label className="inline-field">Count<input type="number" min={1} max={200} value={domainGenCount} onChange={(e) => setDomainGenCount(Number(e.target.value))} /></label>
          <button className="secondary-button" type="button" disabled={busy === "gen-domains"} aria-busy={busy === "gen-domains"} onClick={() => void handleGenerateDomains()}>{busy === "gen-domains" ? "Generating…" : "Generate with LLM"}</button>
        </div>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Domain</th><th className="num">Sentences</th><th>In run</th><th className="actions"><span className="sr-only">Actions</span></th></tr></thead>
            <tbody>
              {domains.map((d) => (
                <tr key={d.domain_id} className={selectedDomainId === d.domain_id ? "row-selected" : ""}>
                  <td><strong>{d.name}</strong>{d.description ? <div className="hint">{d.description}</div> : null}</td>
                  <td className="num">{d.sentence_count}</td>
                  <td><input type="checkbox" aria-label={`Use ${d.name} in the run`} checked={runDomainIds.includes(d.domain_id)} onChange={() => toggleRunDomain(d.domain_id)} /></td>
                  <td className="actions">
                    <button className="secondary-button" type="button" onClick={() => setSelectedDomainId(d.domain_id)}>Sentences</button>
                    <button className="danger-button" type="button" disabled={busy === `del-domain-${d.domain_id}`} onClick={() => void handleDeleteDomain(d.domain_id)}>Delete</button>
                  </td>
                </tr>
              ))}
              {!domains.length ? <tr><td colSpan={4} className="empty">No domains yet. Add one above or generate some with the LLM.</td></tr> : null}
            </tbody>
          </table>
        </div>
      </section>

      <section className="panel" aria-labelledby="dg-sentences">
        <div className="panel-head">
          <h2 id="dg-sentences">Sentences{domainDetail ? ` · ${domainDetail.name}` : ""}</h2>
        </div>
        <div className="button-row">
          <label className="inline-field">Count<input type="number" min={1} max={2000} value={sentenceGenCount} onChange={(e) => setSentenceGenCount(Number(e.target.value))} /></label>
          <button className="secondary-button" type="button" disabled={!selectedDomainId || busy === "gen-sentences"} aria-busy={busy === "gen-sentences"} onClick={() => void handleGenerateSentences()}>{busy === "gen-sentences" ? "Generating…" : "Generate sentences"}</button>
        </div>
        {!domainDetail ? <div className="state">Pick "Sentences" on a domain to see its sentences and extend them with the LLM.</div> : (
          <>
            <p className="hint">{domainDetail.sentence_count} sentences saved.</p>
            <ol className="sentence-list">
              {(domainDetail.sentences || []).map((s, i) => <li key={i}>{s}</li>)}
            </ol>
          </>
        )}
      </section>

      <section className="panel" aria-labelledby="dg-run">
        <div className="panel-head"><h2 id="dg-run">Start data generation</h2></div>
        <div className="field-grid two">
          <label>Voice mode
            <select value={voiceMode} onChange={(e) => setVoiceMode(e.target.value as typeof voiceMode)}>
              <option value="clone">Cloned voices</option>
              <option value="auto">AutoVoice</option>
              <option value="both">Both</option>
            </select>
          </label>
          <label>Language
            <select value={language} onChange={(e) => setLanguage(e.target.value)}>
              {SYNTH_LANGUAGE_OPTIONS.filter((o) => o !== "Auto").map((o) => <option key={o} value={o}>{o}</option>)}
            </select>
          </label>
          <label>WER threshold<input type="number" step={0.01} min={0} max={1} value={werThreshold} onChange={(e) => setWerThreshold(Number(e.target.value))} /><span className="hint">Clips above this word error rate are rejected (0 = only exact).</span></label>
          <label>Max attempts<input type="number" min={1} max={50} value={maxAttempts} onChange={(e) => setMaxAttempts(Number(e.target.value))} /></label>
          <label>TTS concurrency<input type="number" min={1} max={64} value={ttsConcurrency} onChange={(e) => setTtsConcurrency(Number(e.target.value))} /></label>
          <label>ASR concurrency<input type="number" min={1} max={64} value={asrConcurrency} onChange={(e) => setAsrConcurrency(Number(e.target.value))} /></label>
          <label>WER tolerance (letters per word)<input type="number" min={0} max={8} value={tolerance} onChange={(e) => setTolerance(Number(e.target.value))} /></label>
        </div>
        {voiceMode !== "auto" ? (
          <fieldset className="choice-group">
            <legend>Voices ({selectedVoiceIds.length} selected)</legend>
            <div className="button-row">
              {customVoices.map((v) => (
                <label key={v.voice_id} className={`choice-chip ${selectedVoiceIds.includes(v.voice_id) ? "selected" : ""}`}>
                  <input type="checkbox" checked={selectedVoiceIds.includes(v.voice_id)} onChange={() => toggleVoice(v.voice_id)} /> {v.name}
                </label>
              ))}
              {!customVoices.length ? <span className="hint">No saved voices. Add some on the Voices tab or use AutoVoice.</span> : null}
            </div>
          </fieldset>
        ) : null}
        <div className="button-row">
          <button type="button" disabled={busy === "start-run" || !!runActive} aria-busy={busy === "start-run"} onClick={() => void handleStartRun()}>Start run ({runDomainIds.length} domain{runDomainIds.length === 1 ? "" : "s"})</button>
          {runActive ? <button className="danger-button" type="button" onClick={() => void handleCancelRun()}>Stop</button> : null}
        </div>
      </section>

      <section className="panel" aria-labelledby="dg-progress">
        <div className="panel-head"><h2 id="dg-progress">Run progress</h2></div>
        {!run ? <div className="state">No run started yet.</div> : (
          <>
            <dl className="kv-list">
              <div><dt>Status</dt><dd>{run.status} · {run.phase}</dd></div>
              <div><dt>Progress</dt><dd>{run.pct}% ({run.accepted + run.rejected}/{run.planned})</dd></div>
              <div><dt>Accepted</dt><dd>{run.accepted}</dd></div>
              <div><dt>Rejected</dt><dd>{run.rejected}</dd></div>
              <div><dt>Attempts</dt><dd>{run.attempts}</dd></div>
              {run.current ? <div><dt>Current</dt><dd>{run.current}</dd></div> : null}
            </dl>
            {run.error_message ? <div className="banner danger">{run.error_message}</div> : null}
            <div className="bar" role="progressbar" aria-valuenow={run.pct} aria-valuemin={0} aria-valuemax={100}><span className="seg-service" style={{ width: `${run.pct}%` }} /></div>
            <div className="table-wrap">
              <table>
                <thead><tr><th>Voice</th><th className="num">Accepted</th><th className="num">Rejected</th><th className="num">Attempts</th></tr></thead>
                <tbody>
                  {run.voices.map((v) => (
                    <tr key={v.voice}><td>{v.voice}</td><td className="num">{v.accepted}/{v.planned}</td><td className="num">{v.rejected}</td><td className="num">{v.attempts}</td></tr>
                  ))}
                  {!run.voices.length ? <tr><td colSpan={4} className="empty">No voice has started yet.</td></tr> : null}
                </tbody>
              </table>
            </div>
          </>
        )}
      </section>

      <section className="panel span-all" aria-labelledby="dg-clips">
        <div className="panel-head">
          <h2 id="dg-clips">Clip review</h2>
          <div className="button-row">
            <select aria-label="Filter by voice" value={clipFilter} onChange={(e) => { setClipFilter(e.target.value); void loadClips(e.target.value); }}>
              <option value="">All voices</option>
              {clips.voices.map((v) => <option key={v} value={v}>{v}</option>)}
            </select>
            <button className="secondary-button" type="button" onClick={() => void loadClips()}>Reload</button>
          </div>
          <p className="hint">Listen and delete clips that sound wrong; that removes the audio and its transcript (.txt). {clips.total} clips.</p>
        </div>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Clip</th><th>Text</th><th className="num">WER</th><th className="actions"><span className="sr-only">Actions</span></th></tr></thead>
            <tbody>
              {clips.clips.map((clip) => (
                <tr key={clip.clip_id}>
                  <td><strong>{clip.voice}</strong><div className="hint mono">{clip.filename}</div></td>
                  <td className="wrap-cell">{clip.text}{clipAudioUrls[clip.clip_id] ? <audio controls src={clipAudioUrls[clip.clip_id]} /> : null}</td>
                  <td className="num">{clip.wer != null ? `${(clip.wer * 100).toFixed(0)}%` : "—"}</td>
                  <td className="actions">
                    {!clipAudioUrls[clip.clip_id] ? <button className="secondary-button" type="button" onClick={() => void handlePlayClip(clip.clip_id)}>Play</button> : null}
                    <button className="danger-button" type="button" onClick={() => void handleDeleteClip(clip.clip_id)}>Delete</button>
                  </td>
                </tr>
              ))}
              {!clips.clips.length ? <tr><td colSpan={4} className="empty">No clips yet. Start a data generation run above.</td></tr> : null}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
