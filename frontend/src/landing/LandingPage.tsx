export function LandingPage() {
  return (
    <div className="shell landing-shell">
      <main className="panel landing-panel">
        <p className="eyebrow">G3 GPU service</p>
        <h1>G3 OmniVoice</h1>
        <p className="dim">
          Text-to-speech server with an OpenAI-compatible API (<code>/v1/audio/speech</code>), streaming and voice
          cloning. Manage models, voices, settings and API keys in the admin pages.
        </p>
        <div className="button-row">
          <a className="button-link" href="/admin">
            Open admin
          </a>
          <a className="button-link ghost" href="/docs">
            API docs
          </a>
        </div>
      </main>
    </div>
  );
}
