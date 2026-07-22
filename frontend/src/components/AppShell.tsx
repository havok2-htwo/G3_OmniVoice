import type { ReactNode } from 'react';
import type { ApiKeyRecord, DashboardClientConfig, ServerSettings, Snapshot, TabKey } from '../types';
import { tabLabels } from '../types';
import { InfoTip } from './InfoTip';
import { StatusPill } from './StatusPill';

interface AppShellProps {
  activeTab: TabKey;
  onTabChange: (tab: TabKey) => void;
  children: ReactNode;
  snapshot: Snapshot | null;
  settings: ServerSettings;
  clientMode: DashboardClientConfig['mode'];
  onModelChange?: (modelId: string) => void;
  onRefresh: () => void;
  onRotateAdminKey: () => void;
  onLogout: () => void;
  onOpenLanding: () => void;
  adminRecord: ApiKeyRecord | null;
}

export function AppShell({
  activeTab,
  onTabChange,
  children,
  snapshot,
  settings,
  clientMode,
  onModelChange,
  onRefresh,
  onRotateAdminKey,
  onLogout,
  onOpenLanding,
  adminRecord,
}: AppShellProps) {
  const models = snapshot?.models ?? [];
  const activeModel = snapshot?.overview.activeModel || settings.defaultModel;
  const overview = snapshot?.overview;
  const activeModelInfo = models.find((model) => model.id === activeModel);
  const accessState = adminRecord ? 'Session Ready' : 'Login Required';

  return (
    <div className={`app-shell theme-${settings.theme}`}>
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />
      <header className="topbar hero-shell">
        <div className="hero-copy-block">
          <p className="eyebrow">Powered by SONS</p>
          <h1>G3_OmniVoice Control Room</h1>
          <p className="hero-copy">
            Private admin surface for the local OmniVoice runtime, voice assets, queue history, benchmarks, and operator handover.
          </p>
          <div className="hero-actions">
            <button className="secondary-button" type="button" onClick={onRefresh}>
              Refresh Snapshot
            </button>
            <button className="secondary-button" type="button" onClick={onRotateAdminKey}>
              Manage API Keys
            </button>
            <button className="ghost-button" type="button" onClick={onOpenLanding}>
              Open Landing
            </button>
            <button className="ghost-button" type="button" onClick={onLogout}>
              Sign Out
            </button>
          </div>
        </div>
        <div className="topbar-meta hero-pills">
          <StatusPill tone={clientMode === 'mock' ? 'warning' : 'success'} label={clientMode === 'mock' ? 'Mock Backend' : 'Live Admin Backend'} />
          <StatusPill tone="neutral" label={accessState} />
          <StatusPill tone="neutral" label={settings.preferredDevice} />
          <StatusPill tone="neutral" label={settings.baseUrl} />
          {models.length > 0 && onModelChange ? (
            <select
              className="model-select"
              value={activeModel}
              onChange={(e) => onModelChange(e.target.value)}
            >
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label || m.id}
                </option>
              ))}
            </select>
          ) : (
            <StatusPill tone="neutral" label={activeModel || 'No model'} />
          )}
        </div>
      </header>

      <section className="hero-status-grid">
        <article className="hero-status-card">
          <span className="field-label">
            Active model
            <InfoTip text="Current default model for new synthesis requests. Changing the selector in the header updates the backend routing target." />
          </span>
          <strong>{activeModel || 'No model selected'}</strong>
          <p>{activeModelInfo?.taskTypes.join(' / ') || 'Model routing is available after the first snapshot.'}</p>
        </article>
        <article className="hero-status-card">
          <span className="field-label">
            Runtime
            <InfoTip text="Quick runtime summary with backend family, preferred device, torch dtype, and queue capacity." />
          </span>
          <strong>{settings.runtimeBackend}</strong>
          <p>{settings.preferredDevice} with {settings.torchDtype} and queue limit {settings.queueLimit}.</p>
        </article>
        <article className="hero-status-card">
          <span className="field-label">
            Worker state
            <InfoTip text="Live worker condition from the latest snapshot, including queue depth and how many jobs have already been tracked." />
          </span>
          <strong>{overview?.workerState || 'Waiting for backend'}</strong>
          <p>Queue depth {overview?.queueDepth ?? 0} with {overview?.jobsTotal ?? 0} total jobs recorded.</p>
        </article>
        <article className="hero-status-card">
          <span className="field-label">
            Throughput
            <InfoTip text="Realtime generation speed together with average time to first audio and queue wait from the rolling overview snapshot." />
          </span>
          <strong>{overview ? `${overview.realtimeXAvg.toFixed(2)}x realtime` : 'No throughput yet'}</strong>
          <p>Mean TTFA {overview ? `${Math.round(overview.ttfaMsAvg)} ms` : 'pending'} and queue wait {overview ? `${Math.round(overview.queueWaitMsAvg)} ms` : 'pending'}.</p>
        </article>
        <article className="hero-status-card">
          <span className="field-label">
            Access
            <InfoTip text="Shows which admin credential is active in this browser session and when it was last used." />
          </span>
          <strong>{adminRecord?.name || 'Admin Session'}</strong>
          <p>{adminRecord ? `Last used ${new Date(adminRecord.lastUsedAt).toLocaleString('en-US')}` : 'The browser keeps the admin session until you sign out.'}</p>
        </article>
      </section>

      <nav className="tab-rail" aria-label="Main sections">
        {(Object.keys(tabLabels) as TabKey[]).map((tab) => (
          <button key={tab} className={`tab-chip ${activeTab === tab ? 'active' : ''}`} onClick={() => onTabChange(tab)} type="button">
            {tabLabels[tab]}
          </button>
        ))}
      </nav>

      <main className="main-stage">{children}</main>
    </div>
  );
}
