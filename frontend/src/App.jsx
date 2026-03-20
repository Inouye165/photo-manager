import { useEffect, useState } from 'react'
import './App.css'

const EMPTY_DASHBOARD = {
  hero: { title: 'Identity Atlas', summary: 'Loading your photo memory...', momentum: null },
  stats: { originalCount: 0, processedCount: 0, identityCount: 0, peopleSignalCount: 0, animalSignalCount: 0 },
  identityCollections: [],
  lanes: {},
}

const EMPTY_LAB = {
  detections: [],
  stats: { total_detections: 0, confirmed_labels: 0, pending_labels: 0, identity_collections: 0, ready_identities: 0 },
  quick_labels: [],
  identity_collections: [],
  lab_insights: { momentum: '', action_prompt: '', suggestion_count: 0, focus_detection: null },
}

function App() {
  const isLabView = window.location.pathname.startsWith('/lab') || window.location.pathname.startsWith('/label')
  const [dashboard, setDashboard] = useState(EMPTY_DASHBOARD)
  const [lab, setLab] = useState(EMPTY_LAB)
  const [drafts, setDrafts] = useState({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [toast, setToast] = useState('')

  useEffect(() => {
    let isActive = true

    async function loadView() {
      try {
        setLoading(true)
        const response = await fetch(isLabView ? '/api/lab' : '/api/dashboard')
        if (!response.ok) {
          throw new Error('Could not load app data')
        }

        const payload = await response.json()
        if (!isActive) {
          return
        }

        if (isLabView) {
          setLab(payload)
        } else {
          setDashboard(payload)
        }

        setError('')
      } catch (requestError) {
        if (isActive) {
          setError(requestError.message || 'Could not load app data')
        }
      } finally {
        if (isActive) {
          setLoading(false)
        }
      }
    }

    loadView()

    return () => {
      isActive = false
    }
  }, [isLabView])

  useEffect(() => {
    if (!toast) {
      return undefined
    }

    const timer = window.setTimeout(() => setToast(''), 2600)
    return () => window.clearTimeout(timer)
  }, [toast])

  async function refreshLab() {
    const response = await fetch('/api/lab')
    if (!response.ok) {
      throw new Error('Could not refresh the lab queue')
    }
    const payload = await response.json()
    setLab(payload)
  }

  async function saveLabel(imagePath, detectionIndex, assignedLabel, status = 'confirmed') {
    const response = await fetch('/api/label', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_path: imagePath, detection_index: detectionIndex, assigned_label: assignedLabel, status }),
    })

    if (!response.ok) {
      const result = await response.json().catch(() => ({}))
      throw new Error(result.error || 'Could not save label')
    }

    return response.json()
  }

  async function handleConfirm(detection, preferredLabel) {
    const detectionKey = `${detection.image_path}:${detection.detection_index}`
    const assignedLabel = (preferredLabel ?? drafts[detectionKey] ?? '').trim()
    if (!assignedLabel) {
      setToast('Type a name or tap a suggestion first.')
      return
    }

    try {
      await saveLabel(detection.image_path, detection.detection_index, assignedLabel)
      setDrafts((currentDrafts) => {
        const nextDrafts = { ...currentDrafts }
        delete nextDrafts[detectionKey]
        return nextDrafts
      })
      await refreshLab()
      setToast(`Saved ${assignedLabel}. Named folders were refreshed.`)
    } catch (requestError) {
      setToast(requestError.message || 'Could not save label')
    }
  }

  async function handleReject(detection) {
    try {
      await saveLabel(detection.image_path, detection.detection_index, '', 'rejected')
      await refreshLab()
      setToast('Rejected. Clean data matters more than volume.')
    } catch (requestError) {
      setToast(requestError.message || 'Could not reject detection')
    }
  }

  function updateDraft(detection, value) {
    const detectionKey = `${detection.image_path}:${detection.detection_index}`
    setDrafts((currentDrafts) => ({ ...currentDrafts, [detectionKey]: value }))
  }

  return (
    <main className={`app-shell ${isLabView ? 'lab-view' : 'atlas-view'}`}>
      {isLabView ? (
        <LabView
          lab={lab}
          drafts={drafts}
          loading={loading}
          error={error}
          onUpdateDraft={updateDraft}
          onConfirm={handleConfirm}
          onReject={handleReject}
        />
      ) : (
        <AtlasView dashboard={dashboard} loading={loading} error={error} />
      )}
      {toast ? <div className="toast-banner">{toast}</div> : null}
    </main>
  )
}

function AtlasView({ dashboard, loading, error }) {
  const statCards = [
    { label: 'Original Photos', value: dashboard.stats.originalCount, note: 'Photos at the root of the working set.' },
    { label: 'Processed Frames', value: dashboard.stats.processedCount, note: 'Images already pushed through detection.' },
    { label: 'Named Identities', value: dashboard.stats.identityCount, note: 'People and pets with confirmed labels.' },
    { label: 'People Signals', value: dashboard.stats.peopleSignalCount, note: 'Frames that can accelerate known-face coverage.' },
    { label: 'Animal Signals', value: dashboard.stats.animalSignalCount, note: 'Frames that can become named pet collections.' },
  ]

  const lanes = Object.entries(dashboard.lanes || {})

  return (
    <>
      <section className="hero-panel">
        <div>
          <div className="eyebrow">Identity Atlas</div>
          <h1>{dashboard.hero.title}</h1>
          <p>{dashboard.hero.summary}</p>
          <div className="hero-actions">
            <a className="action action-primary" href="/lab">Open Identity Lab</a>
            <a className="action action-secondary" href="https://github.com/Inouye165/photo-manager" target="_blank" rel="noreferrer">View GitHub Repo</a>
          </div>
        </div>
        <aside className="hero-sidebar">
          <div className="hero-chip">Momentum</div>
          <h2>{dashboard.hero.momentum || 'Start naming the obvious wins'}</h2>
          <p>The frontend now lives in React while Flask focuses on API responses, file serving, and image delivery.</p>
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
          <a className="text-link" href="/lab">Keep labeling</a>
        </div>

        {dashboard.identityCollections.length ? (
          <div className="identity-grid">
            {dashboard.identityCollections.map((identity) => (
              <article className="identity-card" key={identity.name}>
                <div className="identity-cover">
                  {identity.coverUrl ? <img src={identity.coverUrl} alt={identity.name} /> : <div className="identity-fallback">{identity.name.slice(0, 1)}</div>}
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
                  <div className="progress-track"><div className="progress-fill" style={{ width: `${identity.stage.progress}%` }} /></div>
                  <p>{identity.stage.summary}</p>
                  {identity.stage.target ? <p>{identity.missingToNext} more examples to reach {identity.stage.target}.</p> : <p>This identity is already in the reliable zone.</p>}
                </div>
              </article>
            ))}
          </div>
        ) : (
          <div className="empty-panel">No named identities yet. Tag a few photos in the lab and this dashboard will start filling out automatically.</div>
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
              <div className="lane-header"><div><h3>{lane.title}</h3><p>{lane.count} frames</p></div></div>
              {lane.items.length ? (
                <div className="thumb-grid">
                  {lane.items.map((item) => (
                    <div className="thumb-card" key={`${key}-${item.filename}`}>
                      <img src={item.url} alt={item.filename} />
                      <div className="thumb-meta"><strong>{item.filename}</strong><span>{item.size}</span></div>
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
    </>
  )
}

function LabView({ lab, drafts, loading, error, onUpdateDraft, onConfirm, onReject }) {
  return (
    <>
      <section className="hero-panel lab-hero">
        <div>
          <div className="eyebrow">Identity Lab</div>
          <h1>Name the memories. Grow the signal.</h1>
          <p>{lab.lab_insights.momentum || 'Every confirmation strengthens the identity engine.'}</p>
          <div className="hero-actions">
            <a className="action action-primary" href="/">Back to Atlas</a>
            <a className="action action-secondary" href="#queue">Jump to Queue</a>
          </div>
        </div>
        <aside className="hero-sidebar">
          <div className="hero-chip">Prediction Pulse</div>
          <h2>{lab.lab_insights.suggestion_count} likely matches ready</h2>
          <p>{lab.lab_insights.action_prompt || 'The next strongest detections are waiting for a name.'}</p>
        </aside>
      </section>

      <section className="stats-grid lab-stats-grid">
        <article className="stat-card"><div className="stat-value">{lab.stats.total_detections}</div><div className="stat-label">Visible Detections</div><div className="stat-note">Everything currently in the review queue.</div></article>
        <article className="stat-card"><div className="stat-value">{lab.stats.confirmed_labels}</div><div className="stat-label">Confirmed</div><div className="stat-note">These already power named collections and suggestions.</div></article>
        <article className="stat-card"><div className="stat-value">{lab.stats.pending_labels}</div><div className="stat-label">Pending</div><div className="stat-note">Clean, obvious labels here will help the most.</div></article>
        <article className="stat-card"><div className="stat-value">{lab.stats.ready_identities}</div><div className="stat-label">Ready Identities</div><div className="stat-note">Collections with enough examples to start learning shape.</div></article>
        <article className="stat-card"><div className="stat-value">{lab.lab_insights.suggestion_count}</div><div className="stat-label">Likely Matches</div><div className="stat-note">Pending detections where the engine already sees a probable name.</div></article>
      </section>

      {error ? <section className="status-panel error">{error}</section> : null}
      {loading ? <section className="status-panel">Refreshing the identity lab...</section> : null}

      <div className="lab-layout">
        <aside className="lab-sidebar">
          <section className="section-panel compact-panel">
            <div className="section-heading compact-heading"><div><div className="eyebrow accent">Next Best Label</div><h2>High-value target</h2></div></div>
            {lab.lab_insights.focus_detection ? (
              <div className="focus-card">
                <img src={lab.lab_insights.focus_detection.crop_path ? `/image/${lab.lab_insights.focus_detection.crop_path}` : `/image/${lab.lab_insights.focus_detection.image_path}`} alt="Focus detection" />
                <div className="focus-body">
                  <strong>{lab.lab_insights.focus_detection.detected_class} candidate</strong>
                  <span>{lab.lab_insights.focus_detection.image_path.split('/').slice(-1)[0]}</span>
                  <span>Confidence {Math.round(lab.lab_insights.focus_detection.confidence * 100)}%. A strong place to spend the next label.</span>
                </div>
              </div>
            ) : (
              <div className="empty-panel compact">No pending detections left.</div>
            )}
          </section>

          <section className="section-panel compact-panel">
            <div className="section-heading compact-heading"><div><div className="eyebrow accent-coral">Identity Roster</div><h2>Who the system knows</h2></div></div>
            {lab.identity_collections.length ? (
              <div className="identity-stack">
                {lab.identity_collections.map((identity) => (
                  <article className="identity-mini-card" key={identity.name}>
                    <div className="identity-row"><h3>{identity.name}</h3><span className={`stage-pill ${identity.stage.tone}`}>{identity.stage.title}</span></div>
                    <div className="chip-row"><span>{identity.sample_count} samples</span><span>{identity.people_count} people</span><span>{identity.animal_count} animals</span></div>
                  </article>
                ))}
              </div>
            ) : (
              <div className="empty-panel compact">No named identities yet.</div>
            )}
          </section>
        </aside>

        <section className="section-panel queue-panel" id="queue">
          <div className="section-heading">
            <div>
              <div className="eyebrow accent">Label Queue</div>
              <h2>Turn detections into known characters</h2>
            </div>
          </div>

          {lab.detections.length ? (
            <div className="detection-grid">
              {lab.detections.map((detection) => {
                const detectionKey = `${detection.image_path}:${detection.detection_index}`
                const draftValue = drafts[detectionKey] ?? detection.assigned_label ?? ''

                return (
                  <article className="detection-card" key={detectionKey}>
                    <div className="detection-image">
                      <img src={detection.crop_path ? `/image/${detection.crop_path}` : `/image/${detection.image_path}`} alt={detection.detected_class} />
                      <span className="subject-pill">{detection.detected_class}</span>
                      <span className="confidence-pill">{Math.round(detection.confidence * 100)}%</span>
                      <span className="status-ribbon">{detection.status}</span>
                    </div>
                    <div className="detection-body">
                      <div className="detection-topline">
                        <div>
                          <div className="detection-class">{detection.detected_class}</div>
                          <div className="detection-filename">{detection.image_path.split('/').slice(-1)[0]}</div>
                        </div>
                      </div>

                      {detection.status === 'confirmed' ? (
                        <div className="current-label confirmed">Confirmed as <strong>{detection.assigned_label}</strong>.</div>
                      ) : detection.status === 'rejected' ? (
                        <div className="current-label rejected">Rejected from the training set.</div>
                      ) : (
                        <>
                          {detection.suggestion ? (
                            <div className="suggestion-box">
                              <div className="suggestion-head">
                                <div>
                                  <div className="suggestion-label">Likely {detection.suggestion.label}</div>
                                  <div className="suggestion-meta">{detection.suggestion.confidence}% confidence from {detection.suggestion.sample_count} labeled samples.</div>
                                </div>
                                <button className="accept-suggestion" type="button" onClick={() => onConfirm(detection, detection.suggestion.label)}>Use suggestion</button>
                              </div>
                              {detection.suggestion.alternatives?.length ? (
                                <div className="suggestion-meta">
                                  Also nearby: {detection.suggestion.alternatives.map((alternative) => `${alternative.name} (${Math.round(alternative.confidence * 100)}%)`).join(', ')}
                                </div>
                              ) : null}
                            </div>
                          ) : null}

                          {lab.quick_labels.length ? (
                            <div className="quick-labels">
                              {lab.quick_labels.map((label) => (
                                <button key={`${detectionKey}-${label}`} type="button" className="quick-label" onClick={() => onUpdateDraft(detection, label)}>{label}</button>
                              ))}
                            </div>
                          ) : null}

                          <div className="label-row">
                            <input value={draftValue} onChange={(event) => onUpdateDraft(detection, event.target.value)} placeholder="Enter name, pet name, or nickname" />
                            <button type="button" className="btn-confirm" onClick={() => onConfirm(detection)}>Save</button>
                            <button type="button" className="btn-reject" onClick={() => onReject(detection)}>Skip</button>
                          </div>
                        </>
                      )}
                    </div>
                  </article>
                )
              })}
            </div>
          ) : (
            <div className="empty-panel">No detections found.</div>
          )}
        </section>
      </div>
    </>
  )
}

export default App
