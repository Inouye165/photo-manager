import { memo, startTransition, useCallback, useDeferredValue, useEffect, useMemo, useRef, useState } from 'react'
import './App.css'
import IdentityLab from './IdentityLab.jsx'
import UploadPanel from './UploadPanel.jsx'

const INITIAL_QUEUE_ITEMS = 12
const QUEUE_STEP = 12
const INITIAL_REVIEW_GROUPS = 4
const REVIEW_STEP = 4

const REVIEW_FILTERS = [
  { key: 'all', label: 'All' },
  { key: 'likely', label: 'Likely' },
  { key: 'review', label: 'Review' },
  { key: 'queue', label: 'Queue' },
  { key: 'backlog', label: 'Backlog' },
  { key: 'done', label: 'Done' },
]

const SUBJECT_FILTERS = [
  { key: 'all', label: 'All Subjects' },
  { key: 'people', label: 'People' },
  { key: 'animals', label: 'Animals' },
]

const LAB_MODES = [
  { key: 'queue', label: 'Queue' },
  { key: 'review', label: 'Review by Identity' },
]

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
  queue_summary: { all: 0, likely: 0, review: 0, queue: 0, backlog: 0, done: 0, people: 0, animals: 0, suggested: 0 },
  lab_insights: { momentum: '', action_prompt: '', suggestion_count: 0, focus_detection: null },
}

function matchesReviewFilter(detection, reviewFilter) {
  if (reviewFilter === 'all') {
    return true
  }

  if (reviewFilter === 'done') {
    return detection.status !== 'pending'
  }

  return detection.review_bucket === reviewFilter
}

function matchesSubjectFilter(detection, subjectFilter) {
  return subjectFilter === 'all' || detection.subject_group === subjectFilter
}

function matchesSearch(detection, searchTerm) {
  if (!searchTerm) {
    return true
  }

  const normalizedSearch = searchTerm.trim().toLowerCase()
  const haystack = [
    detection.detected_class,
    detection.assigned_label,
    detection.suggestion?.label,
    detection.image_path,
  ]
    .filter(Boolean)
    .join(' ')
    .toLowerCase()

  return haystack.includes(normalizedSearch)
}

function groupSuggestedDetections(detections) {
  const groups = new Map()

  for (const detection of detections) {
    if (detection.status !== 'pending' || !detection.suggestion) {
      continue
    }

    const groupKey = detection.suggestion.label
    if (!groups.has(groupKey)) {
      groups.set(groupKey, {
        label: groupKey,
        confidence: detection.suggestion.confidence,
        sampleCount: detection.suggestion.sample_count,
        detections: [],
      })
    }

    const group = groups.get(groupKey)
    group.detections.push(detection)
    group.confidence = Math.max(group.confidence, detection.suggestion.confidence)
    group.sampleCount = Math.max(group.sampleCount, detection.suggestion.sample_count)
  }

  return Array.from(groups.values()).sort((left, right) => (
    right.detections.length - left.detections.length
    || right.confidence - left.confidence
    || left.label.localeCompare(right.label)
  ))
}

const SmartImage = memo(function SmartImage({ src, alt, className, eager = false }) {
  return <img className={className} src={src} alt={alt} loading={eager ? 'eager' : 'lazy'} decoding="async" fetchPriority={eager ? 'high' : 'low'} />
})

function buildMediaUrl(path, scope = 'output', variant = 'thumb') {
  const routeBase = scope === 'input' ? '/working_dir' : '/image'
  return `${routeBase}/${path}?variant=${variant}`
}

function ReviewGroups({ groups, batchBusy, onBatchAccept, onConfirm }) {
  const [visibleCount, setVisibleCount] = useState(INITIAL_REVIEW_GROUPS)
  const sentinelRef = useRef(null)
  const visibleGroups = useMemo(() => groups.slice(0, visibleCount), [groups, visibleCount])

  useEffect(() => {
    if (visibleCount >= groups.length) {
      return undefined
    }

    const node = sentinelRef.current
    if (!node) {
      return undefined
    }

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0]?.isIntersecting) {
          setVisibleCount((currentCount) => Math.min(groups.length, currentCount + REVIEW_STEP))
        }
      },
      { rootMargin: '500px 0px' }
    )

    observer.observe(node)
    return () => observer.disconnect()
  }, [visibleCount, groups.length])

  return (
    <div className="suggestion-group-grid">
      {visibleGroups.map((group) => (
        <article className="suggestion-group-card" key={group.label}>
          <div className="suggestion-group-head">
            <div>
              <div className="eyebrow accent">Suggested Identity</div>
              <h3>{group.label}</h3>
              <p>{group.detections.length} detections at up to {group.confidence}% confidence from {group.sampleCount} labeled samples.</p>
            </div>
            <button type="button" className="accept-suggestion" disabled={batchBusy} onClick={() => onBatchAccept(group.detections)}>
              Accept group
            </button>
          </div>
          <div className="group-thumb-row">
            {group.detections.map((detection) => (
              <button key={`${group.label}-${detection.image_path}-${detection.detection_index}`} type="button" className="group-thumb" onClick={() => onConfirm(detection, detection.suggestion.label)}>
                <SmartImage src={detection.crop_path ? buildMediaUrl(detection.crop_path, 'output', 'thumb') : buildMediaUrl(detection.image_path, 'output', 'thumb')} alt={detection.detected_class} />
                <span>{detection.image_path.split('/').slice(-1)[0]}</span>
              </button>
            ))}
          </div>
        </article>
      ))}
      {visibleCount < groups.length ? <div className="render-sentinel" ref={sentinelRef}>Loading more review groups...</div> : null}
    </div>
  )
}

const DetectionCard = memo(function DetectionCard({ detection, draftValue, quickLabels, onUpdateDraft, onConfirm, onReject }) {
  const detectionKey = `${detection.image_path}:${detection.detection_index}`

  return (
    <article className="detection-card">
      <div className="detection-image">
        <SmartImage src={detection.crop_path ? buildMediaUrl(detection.crop_path, 'output', 'thumb') : buildMediaUrl(detection.image_path, 'output', 'thumb')} alt={detection.detected_class} />
        <span className="subject-pill">{detection.detected_class}</span>
        <span className="confidence-pill">{Math.round(detection.confidence * 100)}%</span>
        <span className={`status-ribbon bucket-${detection.review_bucket}`}>{detection.status === 'pending' ? detection.review_bucket : detection.status}</span>
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

            {quickLabels.length ? (
              <div className="quick-labels">
                {quickLabels.map((label) => (
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
})

function QueueDetections({ detections, drafts, lab, onUpdateDraft, onConfirm, onReject }) {
  const [visibleCount, setVisibleCount] = useState(INITIAL_QUEUE_ITEMS)
  const sentinelRef = useRef(null)
  const visibleDetections = useMemo(() => detections.slice(0, visibleCount), [detections, visibleCount])

  useEffect(() => {
    if (visibleCount >= detections.length) {
      return undefined
    }

    const node = sentinelRef.current
    if (!node) {
      return undefined
    }

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0]?.isIntersecting) {
          setVisibleCount((currentCount) => Math.min(detections.length, currentCount + QUEUE_STEP))
        }
      },
      { rootMargin: '500px 0px' }
    )

    observer.observe(node)
    return () => observer.disconnect()
  }, [visibleCount, detections.length])

  return (
    <div className="detection-grid">
      {visibleDetections.map((detection) => {
        const detectionKey = `${detection.image_path}:${detection.detection_index}`
        const draftValue = drafts[detectionKey] ?? detection.assigned_label ?? ''

        return (
          <DetectionCard
            key={detectionKey}
            detection={detection}
            draftValue={draftValue}
            quickLabels={lab.quick_labels}
            onUpdateDraft={onUpdateDraft}
            onConfirm={onConfirm}
            onReject={onReject}
          />
        )
      })}
      {visibleCount < detections.length ? <div className="render-sentinel" ref={sentinelRef}>Loading more detections...</div> : null}
    </div>
  )
}

function App() {
  const isIdentityLabView = window.location.pathname.startsWith('/identity-lab')
  const isLabView = window.location.pathname.startsWith('/lab') || window.location.pathname.startsWith('/label')
  const [dashboard, setDashboard] = useState(EMPTY_DASHBOARD)
  const [lab, setLab] = useState(EMPTY_LAB)
  const [drafts, setDrafts] = useState({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [toast, setToast] = useState('')
  const [reviewFilter, setReviewFilter] = useState('all')
  const [subjectFilter, setSubjectFilter] = useState('all')
  const [searchTerm, setSearchTerm] = useState('')
  const deferredSearchTerm = useDeferredValue(searchTerm)
  const [batchBusy, setBatchBusy] = useState(false)
  const [labMode, setLabMode] = useState('queue')
  const [lastAction, setLastAction] = useState(null)

  useEffect(() => {
    if (isIdentityLabView) {
      setLoading(false)
      return undefined
    }

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
          startTransition(() => setLab(payload))
        } else {
          startTransition(() => setDashboard(payload))
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
  }, [isLabView, isIdentityLabView])

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
    startTransition(() => setLab(payload))
  }

  async function refreshDashboard() {
    const response = await fetch('/api/dashboard')
    if (!response.ok) {
      throw new Error('Could not refresh the dashboard')
    }
    const payload = await response.json()
    startTransition(() => setDashboard(payload))
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

  const applyActions = useCallback(async function applyActions(actions, successMessage, actionLabel) {
    if (!actions.length) {
      return false
    }

    try {
      setBatchBusy(true)
      for (const action of actions) {
        await saveLabel(action.imagePath, action.detectionIndex, action.assignedLabel ?? '', action.status)
      }

      setDrafts((currentDrafts) => {
        const nextDrafts = { ...currentDrafts }
        for (const action of actions) {
          delete nextDrafts[`${action.imagePath}:${action.detectionIndex}`]
        }
        return nextDrafts
      })

      setLastAction({ label: actionLabel, actions })
      await refreshLab()
      setToast(successMessage)
      return true
    } catch (requestError) {
      setToast(requestError.message || 'Could not apply changes')
      return false
    } finally {
      setBatchBusy(false)
    }
  }, [])

  const handleConfirm = useCallback(async function handleConfirm(detection, preferredLabel) {
    const detectionKey = `${detection.image_path}:${detection.detection_index}`
    let assignedLabel = (preferredLabel ?? '').trim()
    if (!assignedLabel) {
      // Read from drafts via functional update to avoid stale closure
      await new Promise((resolve) => {
        setDrafts((currentDrafts) => {
          assignedLabel = (currentDrafts[detectionKey] ?? '').trim()
          resolve()
          return currentDrafts
        })
      })
    }
    if (!assignedLabel) {
      setToast('Type a name or tap a suggestion first.')
      return
    }

    await applyActions(
      [{ imagePath: detection.image_path, detectionIndex: detection.detection_index, assignedLabel, status: 'confirmed' }],
      `Saved ${assignedLabel}. Named folders were refreshed.`,
      `Saved ${assignedLabel}`
    )
  }, [applyActions])

  const handleReject = useCallback(async function handleReject(detection) {
    await applyActions(
      [{ imagePath: detection.image_path, detectionIndex: detection.detection_index, assignedLabel: '', status: 'rejected' }],
      'Rejected. Clean data matters more than volume.',
      'Rejected detection'
    )
  }, [applyActions])

  const updateDraft = useCallback(function updateDraft(detection, value) {
    const detectionKey = `${detection.image_path}:${detection.detection_index}`
    setDrafts((currentDrafts) => ({ ...currentDrafts, [detectionKey]: value }))
  }, [])

  async function handleBatchAccept(detections) {
    const suggestedDetections = detections.filter((detection) => detection.status === 'pending' && detection.suggestion)
    if (!suggestedDetections.length) {
      setToast('No visible suggestions to accept.')
      return
    }

    await applyActions(
      suggestedDetections.map((detection) => ({
        imagePath: detection.image_path,
        detectionIndex: detection.detection_index,
        assignedLabel: detection.suggestion.label,
        status: 'confirmed',
      })),
      `Accepted ${suggestedDetections.length} visible suggestions.`,
      `Accepted ${suggestedDetections.length} suggestions`
    )
  }

  async function handleBatchReject(detections) {
    const pendingDetections = detections.filter((detection) => detection.status === 'pending')
    if (!pendingDetections.length) {
      setToast('No visible pending detections to reject.')
      return
    }

    await applyActions(
      pendingDetections.map((detection) => ({
        imagePath: detection.image_path,
        detectionIndex: detection.detection_index,
        assignedLabel: '',
        status: 'rejected',
      })),
      `Rejected ${pendingDetections.length} visible detections.`,
      `Rejected ${pendingDetections.length} detections`
    )
  }

  async function handleUndo() {
    if (!lastAction?.actions?.length) {
      setToast('Nothing to undo yet.')
      return
    }

    const reverted = await applyActions(
      lastAction.actions.map((action) => ({
        imagePath: action.imagePath,
        detectionIndex: action.detectionIndex,
        assignedLabel: '',
        status: 'pending',
      })),
      `Undid ${lastAction.label}.`,
      'Undo'
    )

    if (reverted) {
      setLastAction(null)
    }
  }

  return (
    <main className={`app-shell ${isLabView || isIdentityLabView ? 'lab-view' : 'atlas-view'}`}>
      {isIdentityLabView ? (
        <IdentityLab onToast={setToast} />
      ) : isLabView ? (
        <LabView
          lab={lab}
          drafts={drafts}
          loading={loading}
          error={error}
          reviewFilter={reviewFilter}
          subjectFilter={subjectFilter}
          searchTerm={searchTerm}
          deferredSearchTerm={deferredSearchTerm}
          batchBusy={batchBusy}
          labMode={labMode}
          lastAction={lastAction}
          onUpdateDraft={updateDraft}
          onConfirm={handleConfirm}
          onReject={handleReject}
          onReviewFilterChange={setReviewFilter}
          onSubjectFilterChange={setSubjectFilter}
          onSearchTermChange={setSearchTerm}
          onBatchAccept={handleBatchAccept}
          onBatchReject={handleBatchReject}
          onLabModeChange={setLabMode}
          onUndo={handleUndo}
        />
      ) : (
        <AtlasView dashboard={dashboard} loading={loading} error={error} onRefresh={refreshDashboard} onToast={setToast} />
      )}
      {toast ? <div className="toast-banner">{toast}</div> : null}
    </main>
  )
}

function AtlasView({ dashboard, loading, error, onRefresh, onToast }) {
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
            <a className="action action-secondary" href="/identity-lab">Open Verification Queue</a>
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

      <UploadPanel onUploadComplete={onRefresh} onToast={onToast} />

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
                  {identity.coverUrl ? <SmartImage src={identity.coverUrl} alt={identity.name} /> : <div className="identity-fallback">{identity.name.slice(0, 1)}</div>}
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
                      <SmartImage src={item.url} alt={item.filename} />
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

function LabView({
  lab,
  drafts,
  loading,
  error,
  reviewFilter,
  subjectFilter,
  searchTerm,
  deferredSearchTerm,
  batchBusy,
  labMode,
  lastAction,
  onUpdateDraft,
  onConfirm,
  onReject,
  onReviewFilterChange,
  onSubjectFilterChange,
  onSearchTermChange,
  onBatchAccept,
  onBatchReject,
  onLabModeChange,
  onUndo,
}) {
  const filteredDetections = useMemo(
    () => lab.detections.filter(
      (detection) => matchesReviewFilter(detection, reviewFilter)
        && matchesSubjectFilter(detection, subjectFilter)
        && matchesSearch(detection, deferredSearchTerm)
    ),
    [lab.detections, reviewFilter, subjectFilter, deferredSearchTerm]
  )
  const visibleSuggestions = useMemo(
    () => filteredDetections.filter((detection) => detection.status === 'pending' && detection.suggestion).length,
    [filteredDetections]
  )
  const pendingVisible = useMemo(
    () => filteredDetections.filter((detection) => detection.status === 'pending').length,
    [filteredDetections]
  )
  const suggestionGroups = useMemo(
    () => groupSuggestedDetections(filteredDetections),
    [filteredDetections]
  )
  const queueWindowKey = `${reviewFilter}:${subjectFilter}:${deferredSearchTerm}:${filteredDetections.length}`
  const reviewWindowKey = `${reviewFilter}:${subjectFilter}:${deferredSearchTerm}:${suggestionGroups.length}`

  return (
    <>
      <section className="hero-panel lab-hero">
        <div>
          <div className="eyebrow">Identity Lab</div>
          <h1>Name the memories. Grow the signal.</h1>
          <p>{lab.lab_insights.momentum || 'Every confirmation strengthens the identity engine.'}</p>
          <div className="hero-actions">
            <a className="action action-primary" href="/">Back to Atlas</a>
            <a className="action action-secondary" href="/identity-lab">Open Verification Queue</a>
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
                <SmartImage eager src={lab.lab_insights.focus_detection.crop_path ? buildMediaUrl(lab.lab_insights.focus_detection.crop_path, 'output', 'full') : buildMediaUrl(lab.lab_insights.focus_detection.image_path, 'output', 'full')} alt="Focus detection" />
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

          <div className="queue-toolbar">
            <div className="filter-group mode-group">
              {LAB_MODES.map((mode) => (
                <button
                  key={mode.key}
                  type="button"
                  className={`filter-chip ${labMode === mode.key ? 'active' : ''}`}
                  onClick={() => onLabModeChange(mode.key)}
                >
                  {mode.label}
                  <span>{mode.key === 'review' ? suggestionGroups.length : filteredDetections.length}</span>
                </button>
              ))}
            </div>

            <div className="filter-row">
              <div className="filter-group">
                {REVIEW_FILTERS.map((filter) => (
                  <button
                    key={filter.key}
                    type="button"
                    className={`filter-chip ${reviewFilter === filter.key ? 'active' : ''}`}
                    onClick={() => onReviewFilterChange(filter.key)}
                  >
                    {filter.label}
                    <span>{filter.key === 'all' ? lab.queue_summary.all : lab.queue_summary[filter.key] ?? 0}</span>
                  </button>
                ))}
              </div>

              <div className="filter-group secondary-group">
                {SUBJECT_FILTERS.map((filter) => (
                  <button
                    key={filter.key}
                    type="button"
                    className={`filter-chip subtle ${subjectFilter === filter.key ? 'active' : ''}`}
                    onClick={() => onSubjectFilterChange(filter.key)}
                  >
                    {filter.label}
                    <span>{filter.key === 'all' ? lab.queue_summary.all : lab.queue_summary[filter.key] ?? 0}</span>
                  </button>
                ))}
              </div>
            </div>

            <div className="queue-search-row">
              <input
                className="queue-search"
                value={searchTerm}
                onChange={(event) => onSearchTermChange(event.target.value)}
                placeholder="Search by file, label, suggestion, or subject"
              />
              <div className="queue-actions">
                <button type="button" className="accept-suggestion batch-button" disabled={!visibleSuggestions || batchBusy} onClick={() => onBatchAccept(filteredDetections)}>
                  {batchBusy ? 'Applying...' : `Accept ${visibleSuggestions}`}
                </button>
                <button type="button" className="btn-reject batch-button" disabled={!pendingVisible || batchBusy} onClick={() => onBatchReject(filteredDetections)}>
                  {batchBusy ? 'Applying...' : `Reject ${pendingVisible}`}
                </button>
                <button type="button" className="quick-label undo-button" disabled={!lastAction || batchBusy} onClick={onUndo}>
                  Undo last action
                </button>
              </div>
            </div>

            <div className="queue-summary">
              <span className="summary-pill"><strong>{filteredDetections.length}</strong> visible</span>
              <span className="summary-pill"><strong>{lab.queue_summary.suggested}</strong> suggested total</span>
              <span className="summary-pill"><strong>{lab.queue_summary.done}</strong> done</span>
              {lastAction ? <span className="summary-pill"><strong>Undo ready</strong> {lastAction.label}</span> : null}
            </div>
          </div>

          {labMode === 'review' ? (
            suggestionGroups.length ? (
              <ReviewGroups key={reviewWindowKey} groups={suggestionGroups} batchBusy={batchBusy} onBatchAccept={onBatchAccept} onConfirm={onConfirm} />
            ) : (
              <div className="empty-panel">No grouped suggestions match the current filters.</div>
            )
          ) : filteredDetections.length ? (
            <QueueDetections key={queueWindowKey} detections={filteredDetections} drafts={drafts} lab={lab} onUpdateDraft={onUpdateDraft} onConfirm={onConfirm} onReject={onReject} />
          ) : (
            <div className="empty-panel">No detections match the current filters.</div>
          )}
        </section>
      </div>
    </>
  )
}

export default App
