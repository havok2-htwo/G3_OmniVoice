export function LandingPage() {
  return (
    <main className="app-shell">
      <section className="hero-card hero-landing">
        <div className="hero-copy">
          <p className="eyebrow">Powered by SONS</p>
          <h1>G3_OmniVoice</h1>
          <p>
            Eine aufgeraeumte Front fuer OmniVoice mit Adminpanel und API-Dokumentation.
          </p>
          <div className="hero-actions">
            <a className="primary-button" href="/admin">
              Adminpanel
            </a>
            <a className="ghost-button" href="/docs">
              API Docs
            </a>
          </div>
        </div>
      </section>
    </main>
  );
}
