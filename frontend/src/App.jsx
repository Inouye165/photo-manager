import { useEffect, useState } from 'react'
import './App.css'

const EMPTY_DASHBOARD = {
  hero: {
    title: 'Identity Atlas',
    summary: 'Loading your photo memory...',
    momentum: null,
  },
  stats: {
    originalCount: 0,
    processedCount: 0,
    identityCount: 0,
    peopleSignalCount: 0,
    animalSignalCount: 0,
  },
  identityCollections: [],
  lanes: {},
}

function App() {
  const [dashboard, setDashboard] = useState(EMPTY_DASHBOARD)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    let isActive = true

    async function loadDashboard() {
      try {
        setLoading(true)
        const response = await fetch('/api/dashboard')
        if (!response.ok) {
          throw new Error('Could not load dashboard data')
        }
        const payload = await response.json()
        if (isActive) {
          setDashboard(payload)
          setError('')
        }
      } catch (requestError) {
        if (isActive) {
          setError(requestError.message || 'Could not load dashboard data')
        }
      } finally {
        if (isActive) {
          setLoading(false)
        }
      }
    }

    loadDashboard()

    return () => {
      isActive = false
    }
  }, [])

  const statCards = [
    {
      label: 'Original Photos',
      value: dashboard.stats.originalCount,
      note: 'Photos at the root of the working set.',
    },
    {
      label: 'Processed Frames',
      value: dashboard.stats.processedCount,
      note: 'Images already pushed through detection.',
    },
    {
      label: 'Named Identities',
      value: dashboard.stats.identityCount,
      note: 'People and pets with confirmed labels.',
    },
    {
      label: 'People Signals',
      value: dashboard.stats.peopleSignalCount,
      note: 'Frames that can accelerate known-face coverage.',
    },
    {
      label: 'Animal Signals',
      value: dashboard.stats.animalSignalCount,
      note: 'Frames that can become named pet collections.',
    },
  ]

  const lanes = Object.entries(dashboard.lanes || {})

  return (
    <main className="app-shell">
      <section className="hero-panel">
        <div>
          <div className="eyebrow">React Frontend</div>
          <h1>{dashboard.hero.title}</h1>
          <p>{dashboard.hero.summary}</p>
          <div className="hero-actions">
            <a className="action action-primary" href="http://127.0.0.1:5000/label">
              Open Identity Lab
            </a>
            <a className="action action-secondary" href="http://127.0.0.1:5000/">
              Open Flask Atlas
            </a>
          </div>
        </div>
        <aside className="hero-sidebar">
          <div className="hero-chip">Momentum</div>
          <h2>{dashboard.hero.momentum || 'Start naming the obvious wins'}</h2>
          <p>
            This React shell reads from the Flask backend and turns the current project into a proper
            frontend-ready product surface.
          </p>
        </aside>
      </section>

      <section className="stats-grid">
        {statCards.map((card) => (
          <article className="stat-card" key={card.label}>
            <div className="stat-value">{card.value}</div>
            <div className="stat-label">{card.label}</div>
            <div className="stat-note">{card.note}</div>
          </article>
        ))}
      </section>

      {error ? <section className="status-panel error">{error}</section> : null}
      {loading ? <section className="status-panel">Loading the atlas snapshot...</section> : null}

      <section className="section-panel">
        <div className="section-heading">
          <div>
            <div className="eyebrow accent">Identity Constellation</div>
            <h2>Named collections with visible momentum</h2>
          </div>
          <a className="text-link" href="http://127.0.0.1:5000/label">
            Keep labeling
          </a>
        </div>

        {dashboard.identityCollections.length ? (
          <div className="identity-grid">
            {dashboard.identityCollections.map((identity) => (
              <article className="identity-card" key={identity.name}>
                <div className="identity-cover">
                  {identity.coverUrl ? (
                    <img src={identity.coverUrl} alt={identity.name} />
                  ) : (
                    <div className="identity-fallback">{identity.name.slice(0, 1)}</div>
                  )}
                </div>
                <div className="identity-body">
                  <div className="identity-row">
                    <h3>{identity.name}</h3>
                    <span className={`stage-pill ${identity.stage.tone}`}>{identity.stage.title}</span>
                  </div>
                  <div className="chip-row">
                    <span>{identity.sampleCount} samples</span>
                    <span>{identity.peopleCount} people</span>
                    <span>{identity.animalCount} animals</span>
                  </div>
                  <div className="progress-track">
                    <div className="progress-fill" style={{ width: `${identity.stage.progress}%` }} />
                  </div>
                  <p>{identity.stage.summary}</p>
                  {identity.stage.target ? (
                    <p>{identity.missingToNext} more examples to reach {identity.stage.target}.</p>
                  ) : (
                    <p>This identity is already in the reliable zone.</p>
                  )}
                </div>
              </article>
            ))}
          </div>
        ) : (
          <div className="empty-panel">No named identities yet. Tag a few photos in the lab and this React dashboard will start filling out automatically.</div>
        )}
      </section>

      <section className="section-panel">
        <div className="section-heading">
          <div>
            <div className="eyebrow accent-coral">Detection Lanes</div>
            <h2>Signals worth turning into names</h2>
          </div>
        </div>

        <div className="lane-grid">
          {lanes.map(([key, lane]) => (
            <article className="lane-card" key={key}>
              <div className="lane-header">
                <div>
                  <h3>{lane.title}</h3>
                  <p>{lane.count} frames</p>
                </div>
              </div>
              {lane.items.length ? (
                <div className="thumb-grid">
                  {lane.items.map((item) => (
                    <div className="thumb-card" key={`${key}-${item.filename}`}>
                      <img src={item.url} alt={item.filename} />
                      <div className="thumb-meta">
                        <strong>{item.filename}</strong>
                        <span>{item.size}</span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="empty-panel compact">No preview items yet in this lane.</div>
              )}
            </article>
          ))}
        </div>
      </section>
    </main>
  )
}

export default App
