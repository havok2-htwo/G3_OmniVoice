// G3 shared "API keys" tab (STANDARD.md section 5). Copy verbatim to frontend/src/ApiKeysPanel.tsx.
// The repo maps its own API rows to ApiKeyRow and passes create/revoke functions from its api.ts.
// Keep the created token in this component only; App must unmount it (or pass a new `sessionKey`) on logout.
import { useState } from "react";
import type { FormEvent } from "react";
import { CopyButton, toast } from "./Toasts";

export interface ApiKeyRow {
  id: string;
  alias: string;
  hint: string | null; // masked preview "rerank_AbCd…wXyZ" (null for keys created before the hint existed)
  created_at: string | null; // ISO
  last_used_at: string | null; // ISO
  requests: number;
  usage: number | null; // service unit (documents, texts, audio seconds, characters)
}

export interface ApiKeysPanelProps {
  keys: ApiKeyRow[] | null; // null = still loading
  loadError?: string | null;
  usageLabel: string; // column header: "Documents", "Texts", "Audio (s)", "Audio (s)"
  formatUsage?: (value: number) => string;
  baseUrl: string; // e.g. window.location.origin
  exampleCurl: (key: string) => string; // full example request, key inserted
  routesText: string; // e.g. "POST /v1/rerank, POST /rerank, GET /v1/models"
  onCreate: (alias: string) => Promise<{ token: string }>;
  onRevoke: (id: string) => Promise<void>;
  onChanged: () => void; // reload the list
}

const fmtDate = (iso: string | null) => {
  if (!iso) return "Never";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
};
const nf = new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 });

export function ApiKeysPanel(props: ApiKeysPanelProps) {
  const { keys, loadError, usageLabel, formatUsage, baseUrl, exampleCurl, routesText, onCreate, onRevoke, onChanged } = props;
  const [alias, setAlias] = useState("");
  const [creating, setCreating] = useState(false);
  const [created, setCreated] = useState<{ alias: string; token: string } | null>(null);
  const [revoking, setRevoking] = useState<string | null>(null);
  const count = keys?.length ?? 0;
  const isProtected = count > 0;

  const create = async (e: FormEvent) => {
    e.preventDefault();
    const name = alias.trim();
    if (!name) return;
    setCreating(true);
    try {
      const res = await onCreate(name);
      setCreated({ alias: name, token: res.token });
      setAlias("");
      toast.ok(`Key "${name}" created.`);
      onChanged();
    } catch (err) {
      toast.error(`Could not create the key: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setCreating(false);
    }
  };

  const revoke = async (row: ApiKeyRow) => {
    const last = count === 1;
    const text = `Revoke key "${row.alias}"${row.hint ? ` (${row.hint})` : ""}?\n\nClients using it stop working immediately.`
      + (last ? "\n\nThis is the last key: the API will be OPEN again (no key required)." : "");
    if (!window.confirm(text)) return;
    setRevoking(row.id);
    try {
      await onRevoke(row.id);
      toast.ok(last ? `Key "${row.alias}" revoked. The API is open again.` : `Key "${row.alias}" revoked.`);
      onChanged();
    } catch (err) {
      toast.error(`Could not revoke the key: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setRevoking(null);
    }
  };

  return (
    <section className="panel" aria-labelledby="api-keys-title">
      <div className="panel-head">
        <h2 id="api-keys-title">API keys</h2>
        <span className={`badge ${isProtected ? "ok" : "warn"}`}>{isProtected ? `Protected · ${count} key${count === 1 ? "" : "s"}` : "Open"}</span>
        <p className="hint">
          {isProtected
            ? `Requests to ${routesText} need "Authorization: Bearer <key>" or "X-API-Key: <key>". The admin pages use your login instead.`
            : `No key exists, so ${routesText} accept requests without a key. Create a key to require one.`}
        </p>
      </div>

      <form className="button-row" onSubmit={create}>
        <label className="sr-only" htmlFor="api-key-alias">Key name</label>
        <input id="api-key-alias" style={{ flex: "1 1 240px", width: "auto" }} maxLength={64} placeholder="Name, e.g. Genesis3 on Desktop"
          value={alias} onChange={(e) => setAlias(e.target.value)} autoComplete="off" />
        <button type="submit" disabled={creating || !alias.trim()} aria-busy={creating}>{creating ? "Creating…" : "Create key"}</button>
      </form>

      {created && (
        <div className="key-reveal" role="status">
          <strong>New key "{created.alias}". It is shown only once; store it now.</strong>
          <code>{created.token}</code>
          <div className="button-row">
            <CopyButton text={created.token} label="Copy key" className="small-button" />
            <CopyButton text={exampleCurl(created.token)} label="Copy curl example" />
            <button type="button" className="ghost-button small-button" onClick={() => setCreated(null)}>Done, hide it</button>
          </div>
        </div>
      )}

      {loadError && <div className="banner danger">Could not load keys: {loadError}</div>}

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Name</th><th>Key</th><th>Created</th><th>Last used</th>
              <th className="num">Requests</th><th className="num">{usageLabel}</th><th className="actions"><span className="sr-only">Actions</span></th>
            </tr>
          </thead>
          <tbody>
            {keys === null && <tr><td colSpan={7} className="empty"><span className="spinner" /> Loading…</td></tr>}
            {keys !== null && keys.length === 0 && (
              <tr><td colSpan={7} className="empty">No API keys. The API is open to everyone who can reach this server.</td></tr>
            )}
            {(keys ?? []).map((row) => (
              <tr key={row.id}>
                <td>{row.alias}</td>
                <td><span className="mono">{row.hint ?? "—"}</span></td>
                <td>{fmtDate(row.created_at)}</td>
                <td>{fmtDate(row.last_used_at)}</td>
                <td className="num">{nf.format(row.requests)}</td>
                <td className="num">{row.usage == null ? "—" : formatUsage ? formatUsage(row.usage) : nf.format(row.usage)}</td>
                <td className="actions">
                  <button type="button" className="danger-button" disabled={revoking === row.id} aria-busy={revoking === row.id}
                    onClick={() => void revoke(row)}>{revoking === row.id ? "Revoking…" : "Revoke"}</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <details>
        <summary className="dim">Connection details</summary>
        <div className="grid" style={{ marginTop: "var(--sp-3)" }}>
          <div className="hint">Base URL</div>
          <code className="mono">{baseUrl}</code>
          <div className="hint">Example</div>
          <pre>{exampleCurl("<your key>")}</pre>
        </div>
      </details>
    </section>
  );
}
