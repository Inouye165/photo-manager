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
  hero: { title: 'Photo browser', summary: 'Loading your photos...', momentum: null },
  stats: { originalCount: 0, processedCount: 0, identityCount: 0, peopleSignalCount: 0, animalSignalCount: 0, sourceVaultCount: 0, duplicateUploadCount: 0 },
  identityCollections: [],
  lanes: {},
  vault: {
    sourceDir: '',
    sourceCount: 0,
    uniqueHashCount: 0,
    integrityStatus: 'clean',
    sourceSizeBytes: 0,
    sourceSizeLabel: '0 B',
    uploadAttempts: 0,
    newUploads: 0,
    blockedDuplicateAttempts: 0,
    lastUploadAt: null,
    recentBlockedAttempts: [],
    legacySourceMode: false,
    legacyMigratedCount: 0,
  },
}

function formatRelativeTimestamp(value) {
  if (!value) {
    return 'No uploads yet'
  }

  const date = new Date(value)
  if (Number.isNaN(date.getTime())) {
    return 'Unknown'
  }

  return new Intl.DateTimeFormat(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(date)
}

const EMPTY_LAB = {
  detections: [],
  stats: { total_detections: 0, confirmed_labels: 0, pending_labels: 0, identity_collections: 0, ready_identities: 0 },
  quick_labels: [],
  identity_collections: [],
  queue_summary: { all: 0, likely: 0, review: 0, queue: 0, backlog: 0, done: 0, people: 0, animals: 0, suggested: 0 },
  lab_insights: { momentum: '', action_prompt: '', suggestion_count: 0, focus_detection: null },
}

function GlobalNav({ isLabView, isIdentityLabView }) {
  const currentPath = window.location.pathname

  const links = [
    { label: 'All Images', href: '/#all-images', active: currentPath === '/' },
    { label: 'Tagged', href: '/#tagged', active: currentPath === '/' },
    { label: 'Lab', href: '/lab#queue', active: isLabView },
    { label: 'Search', href: '/identity-lab#semantic-search', active: isIdentityLabView },
  ]

  return (
    <nav className="global-nav" aria-label="Primary navigation">
      {links.map((link) => (
        <a key={link.label} className={`global-nav-link ${link.active ? 'active' : ''}`} href={link.href}>{link.label}</a>
      ))}
    </nav>
  )
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
  if (subjectFilter === 'all') {
    return true
  }

  if (subjectFilter === 'people') {
    return detection.subject_group === 'person'
  }

  if (subjectFilter === 'animals') {
    return detection.subject_group !== 'person' && detection.subject_group !== 'unknown'
  }

  return detection.subject_group === subjectFilter
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
              <div className="eyebrow accent">Suggested name</div>
              <h3>{group.label}</h3>
              <p>{group.detections.length} detections, up to {group.confidence}% confidence, based on {group.sampleCount} saved examples.</p>
            </div>
            <button type="button" className="accept-suggestion" disabled={batchBusy} onClick={() => onBatchAccept(group.detections)}>
              Save all
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
      {visibleCount < groups.length ? <div className="render-sentinel" ref={sentinelRef}>Loading more groups...</div> : null}
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
          <div className="current-label confirmed">Saved as <strong>{detection.assigned_label}</strong>.</div>
        ) : detection.status === 'rejected' ? (
          <div className="current-label rejected">Skipped.</div>
        ) : (
          <>
            {detection.suggestion ? (
              <div className="suggestion-box">
                <div className="suggestion-head">
                  <div>
                    <div className="suggestion-label">Best match: {detection.suggestion.label}</div>
                    <div className="suggestion-meta">{detection.suggestion.confidence}% confidence from {detection.suggestion.sample_count} saved examples.</div>
                  </div>
                  <button className="accept-suggestion" type="button" onClick={() => onConfirm(detection, detection.suggestion.label)}>Use name</button>
                </div>
                {detection.suggestion.alternatives?.length ? (
                  <div className="suggestion-meta">
                    Also close: {detection.suggestion.alternatives.map((alternative) => `${alternative.name} (${Math.round(alternative.confidence * 100)}%)`).join(', ')}
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
              <input value={draftValue} onChange={(event) => onUpdateDraft(detection, event.target.value)} placeholder="Enter a person or pet name" />
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

function ImageInspectModal({ image, onClose, onSave, onToast }) {
  const [payload, setPayload] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [drafts, setDrafts] = useState({})
  const [selectedIndex, setSelectedIndex] = useState(0)
  const [displaySize, setDisplaySize] = useState(null)
  const [savingKey, setSavingKey] = useState('')
  const requestPath = image?.imagePath || ''

  const loadInspection = useCallback(async () => {
    if (!requestPath) {
      return
    }

    try {
      setLoading(true)
      const response = await fetch(`/api/image/inspect?image_path=${encodeURIComponent(requestPath)}`)
      if (!response.ok) {
        const result = await response.json().catch(() => ({}))
        throw new Error(result.error || 'Could not open this image')
      }

      const nextPayload = await response.json()
      const firstPendingIndex = nextPayload.detections.findIndex((detection) => detection.status === 'pending')
      setPayload(nextPayload)
      setSelectedIndex(firstPendingIndex >= 0 ? firstPendingIndex : 0)
      setDrafts({})
      setError('')
      setDisplaySize(null)
    } catch (requestError) {
      setPayload(null)
      setError(requestError.message || 'Could not open this image')
    } finally {
      setLoading(false)
    }
  }, [requestPath])

  useEffect(() => {
    if (!image) {
      return undefined
    }

    void loadInspection()
    return undefined
  }, [image, loadInspection])

  useEffect(() => {
    if (!image) {
      return undefined
    }

    function handleKeyDown(event) {
      if (event.key === 'Escape') {
        onClose()
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [image, onClose])

  if (!image) {
    return null
  }

  const detections = payload?.detections || []
  const referenceWidth = payload?.sourceWidth || displaySize?.width || 0
  const referenceHeight = payload?.sourceHeight || displaySize?.height || 0

  function getDetectionStatusText(detection) {
    if (detection.status === 'confirmed' && detection.assignedLabel) {
      return {
        badge: 'Saved',
        headline: `Saved as ${detection.assignedLabel}`,
        detail: `Detected as ${detection.detectedClass}.`,
      }
    }
    if (detection.status === 'rejected') {
      return {
        badge: 'Skipped',
        headline: 'Skipped during review',
        detail: `Detector saw ${detection.detectedClass}, but this box is not being used as a saved label.`,
      }
    }
    return {
      badge: 'Pending',
      headline: 'No saved name yet',
      detail: `Detected as ${detection.detectedClass}. Save a name if this box is correct.`,
    }
  }

  async function handleSaveDetection(detection, status) {
    if (!payload) {
      return
    }

    const draftKey = `${payload.imagePath}:${detection.detectionIndex}`
    const assignedLabel = (drafts[draftKey] ?? detection.assignedLabel ?? '').trim()
    if (status === 'confirmed' && !assignedLabel) {
      onToast?.('Enter a name before saving.')
      return
    }

    try {
      setSavingKey(draftKey)
      await onSave({
        imagePath: payload.imagePath,
        detectionIndex: detection.detectionIndex,
        assignedLabel: status === 'confirmed' ? assignedLabel : '',
        status,
      })
      await loadInspection()
    } catch (requestError) {
      onToast?.(requestError.message || 'Could not save this detection')
    } finally {
      setSavingKey('')
    }
  }

  return (
    <div className="compare-modal-backdrop" role="dialog" aria-modal="true" onClick={onClose}>
      <section className="compare-modal image-inspect-modal" onClick={(event) => event.stopPropagation()}>
        <div className="compare-modal-header">
          <div>
            <div className="eyebrow accent">Image review</div>
            <h2>{payload?.filename || image.filename || 'Image'}</h2>
            <p>{detections.length ? `${detections.length} detection${detections.length === 1 ? '' : 's'} in this image.` : 'No saved detections for this image yet.'}</p>
          </div>
          <div className="compare-modal-actions">
            {payload?.originalFullUrl ? <a href={payload.originalFullUrl} target="_blank" rel="noopener noreferrer" className="action action-secondary">Open image</a> : null}
            <button type="button" className="verify-button verify-secondary compact" onClick={onClose}>Close</button>
          </div>
        </div>

        {error ? <div className="status-panel error">{error}</div> : null}
        {loading ? <div className="status-panel">Opening image...</div> : null}

        {!loading && payload ? (
          <div className="image-inspect-layout">
            <div className="image-inspect-stage">
              {payload.originalPreviewUrl ? (
                <div className="image-inspect-frame">
                  <img
                    src={payload.originalPreviewUrl}
                    alt={payload.filename}
                    loading="eager"
                    onLoad={(event) => {
                      setDisplaySize({
                        width: event.currentTarget.naturalWidth,
                        height: event.currentTarget.naturalHeight,
                      })
                    }}
                  />
                  {detections.map((detection) => {
                    const bbox = detection.bbox || []
                    const overlayStyle = bbox.length === 4 && referenceWidth > 0 && referenceHeight > 0 ? {
                      left: `${(bbox[0] / referenceWidth) * 100}%`,
                      top: `${(bbox[1] / referenceHeight) * 100}%`,
                      width: `${((bbox[2] - bbox[0]) / referenceWidth) * 100}%`,
                      height: `${((bbox[3] - bbox[1]) / referenceHeight) * 100}%`,
                    } : null

                    if (!overlayStyle) {
                      return null
                    }

                    return (
                      <button
                        key={`${payload.imagePath}-${detection.detectionIndex}`}
                        type="button"
                        className={`image-inspect-box ${selectedIndex === detection.detectionIndex ? 'active' : ''} status-${detection.status}`}
                        style={overlayStyle}
                        onClick={() => setSelectedIndex(detection.detectionIndex)}
                        aria-label={`Open detection ${detection.detectionIndex + 1}`}
                      >
                        <span>{detection.detectionIndex + 1}</span>
                      </button>
                    )
                  })}
                </div>
              ) : <div className="empty-panel compact">No image preview available.</div>}
            </div>

            <div className="image-inspect-sidebar">
              {detections.length ? detections.map((detection) => {
                const draftKey = `${payload.imagePath}:${detection.detectionIndex}`
                const isSaving = savingKey === draftKey
                const draftValue = drafts[draftKey] ?? detection.assignedLabel ?? ''
                const statusText = getDetectionStatusText(detection)

                return (
                  <article key={draftKey} className={`image-detection-card ${selectedIndex === detection.detectionIndex ? 'active' : ''}`}>
                    <button type="button" className="image-detection-summary" onClick={() => setSelectedIndex(detection.detectionIndex)}>
                      <div>
                        <strong>Detection {detection.detectionIndex + 1}</strong>
                        <span>{detection.detectedClass} · {Math.round(Number(detection.confidence || 0) * 100)}%</span>
                      </div>
                      <span className={`stage-pill ${detection.status === 'confirmed' ? 'dialed' : detection.status === 'rejected' ? 'warming' : 'spark'}`}>
                        {statusText.badge}
                      </span>
                    </button>
                    <div className="image-detection-body">
                      {detection.cropPreviewUrl ? <img src={detection.cropPreviewUrl} alt={`${detection.detectedClass} crop`} loading="lazy" /> : <div className="identity-task-fallback">No crop</div>}
                      <div className="identity-task-suggestion">
                        <strong>{statusText.headline}</strong>
                        <span>{statusText.detail}</span>
                      </div>
                      <input
                        value={draftValue}
                        onChange={(event) => setDrafts((currentDrafts) => ({ ...currentDrafts, [draftKey]: event.target.value }))}
                        placeholder="Type a name"
                      />
                      <div className="image-detection-actions">
                        <button type="button" className="verify-button verify-yes compact" disabled={isSaving} onClick={() => handleSaveDetection(detection, 'confirmed')}>
                          {isSaving ? 'Saving...' : 'Save name'}
                        </button>
                        <button type="button" className="verify-button verify-no compact" disabled={isSaving} onClick={() => handleSaveDetection(detection, 'rejected')}>
                          Skip
                        </button>
                      </div>
                    </div>
                  </article>
                )
              }) : <div className="empty-panel compact">This image opens correctly, but there are no detection boxes saved for it yet.</div>}
            </div>
          </div>
        ) : null}
      </section>
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
  const [inspectedImage, setInspectedImage] = useState(null)

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
      throw new Error('Could not refresh the queue')
    }
    const payload = await response.json()
    startTransition(() => setLab(payload))
  }

  async function refreshDashboard() {
    const response = await fetch('/api/dashboard')
    if (!response.ok) {
      throw new Error('Could not refresh the photo browser')
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

  async function handleInspectionSave({ imagePath, detectionIndex, assignedLabel, status }) {
    await saveLabel(imagePath, detectionIndex, assignedLabel, status)
    await refreshDashboard()
    setToast(status === 'rejected' ? 'Skipped.' : `Saved ${assignedLabel}.`)
  }

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
      `Saved ${assignedLabel}.`,
      `Saved ${assignedLabel}`
    )
  }, [applyActions])

  const handleReject = useCallback(async function handleReject(detection) {
    await applyActions(
      [{ imagePath: detection.image_path, detectionIndex: detection.detection_index, assignedLabel: '', status: 'rejected' }],
      'Skipped.',
      'Skipped detection'
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
      `Saved ${suggestedDetections.length} suggested match${suggestedDetections.length === 1 ? '' : 'es'}.`,
      `Saved ${suggestedDetections.length} suggestions`
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
      `Skipped ${pendingDetections.length} detection${pendingDetections.length === 1 ? '' : 's'}.`,
      `Skipped ${pendingDetections.length} detections`
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
      <GlobalNav isLabView={isLabView} isIdentityLabView={isIdentityLabView} />
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
        <AtlasView
          dashboard={dashboard}
          loading={loading}
          error={error}
          onRefresh={refreshDashboard}
          onToast={setToast}
          onOpenImage={setInspectedImage}
        />
      )}
      {toast ? <div className="toast-banner">{toast}</div> : null}
      <ImageInspectModal image={inspectedImage} onClose={() => setInspectedImage(null)} onSave={handleInspectionSave} onToast={setToast} />
    </main>
  )
}

function AtlasView({ dashboard, loading, error, onRefresh, onToast, onOpenImage }) {
  const statCards = [
    { label: 'Original photos', value: dashboard.stats.originalCount, note: 'Photos in the source vault.' },
    { label: 'Processed images', value: dashboard.stats.processedCount, note: 'Images that already went through detection.' },
    { label: 'Saved names', value: dashboard.stats.identityCount, note: 'People and pets with confirmed labels.' },
    { label: 'Photos with people', value: dashboard.stats.peopleSignalCount, note: 'Images where at least one person was detected.' },
    { label: 'Photos with animals', value: dashboard.stats.animalSignalCount, note: 'Images where at least one animal was detected.' },
    { label: 'Blocked uploads', value: dashboard.stats.duplicateUploadCount, note: 'Duplicate uploads stopped before they entered the vault.' },
  ]

  const lanes = Object.entries(dashboard.lanes || {})

  return (
    <>
      <section className="hero-panel">
        <div>
          <div className="eyebrow">Photo browser</div>
          <h1>{dashboard.hero.title}</h1>
          <p>{dashboard.hero.summary}</p>
          <div className="hero-actions">
            <a className="action action-primary" href="/lab">Open queue</a>
            <a className="action action-secondary" href="/identity-lab">Open labeling tools</a>
            <a className="action action-secondary" href="https://github.com/Inouye165/photo-manager" target="_blank" rel="noreferrer">View GitHub Repo</a>
          </div>
        </div>
        <aside className="hero-sidebar">
          <div className="hero-chip">Next step</div>
          <h2>{dashboard.hero.momentum || 'Open an image and label what you see'}</h2>
          <p>Click any image below to open the full photo and review every detection box on it.</p>
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
      {loading ? <section className="status-panel">Loading your photo browser...</section> : null}

      <UploadPanel onUploadComplete={onRefresh} onToast={onToast} />

      <VaultAdminPanel vault={dashboard.vault} onOpenImage={onOpenImage} />

      <section className="section-panel" id="identity-constellation">
        <div className="section-heading">
          <div>
            <div className="eyebrow accent">Saved names</div>
            <h2>People and pets you have already labeled</h2>
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
                    <span>{identity.sampleCount} photos</span>
                    <span>{identity.peopleCount} people</span>
                    <span>{identity.animalCount} animals</span>
                  </div>
                  <div className="progress-track"><div className="progress-fill" style={{ width: `${identity.stage.progress}%` }} /></div>
                  <p>{identity.stage.summary}</p>
                  {identity.stage.target ? <p>{identity.missingToNext} more photo{identity.missingToNext === 1 ? '' : 's'} to reach {identity.stage.target}.</p> : <p>This name already has enough examples.</p>}
                </div>
              </article>
            ))}
          </div>
        ) : (
          <div className="empty-panel">No saved names yet. Label a few detections in the queue and they will show up here.</div>
        )}
      </section>

      <section className="section-panel" id="signal-lanes">
        <div className="section-heading">
          <div>
            <div className="eyebrow accent-coral">Browse by result</div>
            <h2>Open any image to see all detections on it</h2>
          </div>
        </div>
        <div className="lane-grid">
          {lanes.map(([key, lane]) => (
            <article className="lane-card" key={key}>
              <div className="lane-header"><div><h3>{lane.title}</h3><p>{lane.count} image{lane.count === 1 ? '' : 's'}</p></div></div>
              {lane.items.length ? (
                <div className="thumb-grid">
                  {lane.items.map((item) => (
                    <button
                      type="button"
                      className="thumb-card thumb-card-button"
                      key={`${key}-${item.relativePath || item.url || item.filename}`}
                      onClick={() => onOpenImage({ imagePath: item.inspectPath || item.relativePath, filename: item.filename })}
                    >
                      <SmartImage src={item.url} alt={item.filename} />
                      <div className="thumb-meta"><strong>{item.filename}</strong><span>{item.size}</span></div>
                    </button>
                  ))}
                </div>
              ) : (
                <div className="empty-panel compact">No images in this section yet.</div>
              )}
            </article>
          ))}
        </div>
      </section>
    </>
  )
}

function VaultAdminPanel({ vault, onOpenImage }) {
  const [browserPayload, setBrowserPayload] = useState({ folders: [], tagged: { count: 0, groups: [] }, sourceCount: 0, uniqueHashCount: 0, duplicateHashGroups: 0 })
  const [browserLoading, setBrowserLoading] = useState(true)
  const [browserError, setBrowserError] = useState('')
  const [activeBrowserTab, setActiveBrowserTab] = useState('vault')

  useEffect(() => {
    function syncTabFromHash() {
      const hash = window.location.hash.toLowerCase()
      if (hash === '#tagged') {
        setActiveBrowserTab('tagged')
        return
      }
      if (hash === '#all-images') {
        setActiveBrowserTab('vault')
      }
    }

    syncTabFromHash()
    window.addEventListener('hashchange', syncTabFromHash)
    return () => window.removeEventListener('hashchange', syncTabFromHash)
  }, [])

  useEffect(() => {
    let isActive = true

    async function loadVaultBrowser() {
      try {
        setBrowserLoading(true)
        const response = await fetch('/api/vault/browser')
        if (!response.ok) {
          throw new Error('Could not load vault browser')
        }

        const payload = await response.json()
        if (!isActive) {
          return
        }

        setBrowserPayload(payload)
        setBrowserError('')
      } catch (requestError) {
        if (isActive) {
          setBrowserError(requestError.message || 'Could not load vault browser')
        }
      } finally {
        if (isActive) {
          setBrowserLoading(false)
        }
      }
    }

    loadVaultBrowser()
    return () => {
      isActive = false
    }
  }, [])

  const miniStats = [
    { label: 'Unique originals', value: vault.sourceCount, note: vault.sourceSizeLabel },
    { label: 'Upload attempts', value: vault.uploadAttempts, note: `${vault.newUploads} new accepted` },
    { label: 'Blocked attempts', value: vault.blockedDuplicateAttempts, note: 'Matched by SHA-256, not filename' },
    { label: 'Vault integrity', value: vault.integrityStatus === 'clean' ? 'Clean' : 'Check', note: `${vault.uniqueHashCount} unique hashes across ${vault.sourceCount} originals` },
    { label: 'Legacy migration', value: vault.legacyMigratedCount, note: vault.legacySourceMode ? 'Single-folder installs were backfilled into the vault' : 'Dedicated source vault mode' },
  ]
  const taggedGroups = browserPayload.tagged?.groups || []
  const taggedCount = browserPayload.tagged?.count || 0

  return (
    <section className="section-panel compact-panel vault-panel" id="vault-browser">
      <div className="section-heading compact-heading">
        <div>
          <div className="eyebrow accent">Source vault</div>
          <h2>Original files and upload history</h2>
        </div>
        <div className="vault-last-upload">Last upload: {formatRelativeTimestamp(vault.lastUploadAt)}</div>
      </div>

      <div className="vault-mini-grid">
        {miniStats.map((card) => (
          <article className="vault-mini-card" key={card.label}>
            <div className="vault-mini-value">{card.value}</div>
            <div className="vault-mini-label">{card.label}</div>
            <div className="vault-mini-note">{card.note}</div>
          </article>
        ))}
      </div>

      <div className="vault-meta-row">
        <div>
          <strong>Source vault</strong>
          <span>{vault.sourceDir || 'Not configured'}</span>
        </div>
      </div>

      <div className="vault-format-note">
        <strong>Display format</strong>
        <span>Originals stay in the vault as HEIC, JPG, or PNG. The app shows WebP previews so every image opens reliably in the browser.</span>
      </div>

      {vault.recentBlockedAttempts?.length ? (
        <div className="vault-duplicates-list">
          {vault.recentBlockedAttempts.map((item) => (
            <div className="vault-duplicate-item" key={`${item.sha256}-${item.timestamp}`}>
              <strong>{item.original_filename || 'Unnamed upload'}</strong>
              <span>Blocked because the vault already contains {item.matched_filename}.</span>
              <span className="vault-hash">{item.sha256.slice(0, 12)}...</span>
            </div>
          ))}
        </div>
      ) : (
        <div className="empty-panel compact">No duplicate uploads have been blocked yet.</div>
      )}

      <div className="section-heading compact-heading">
        <div>
          <div className="eyebrow accent-coral">Browse photos</div>
          <h2>All originals and all saved tags</h2>
        </div>
      </div>

      <div className="vault-browser-tabs" role="tablist" aria-label="Vault browser views">
        <button id="all-images" type="button" className={`vault-browser-tab ${activeBrowserTab === 'vault' ? 'active' : ''}`} onClick={() => { setActiveBrowserTab('vault'); window.history.replaceState(null, '', '#all-images') }}>
          All photos
          <span>{browserPayload.sourceCount || 0}</span>
        </button>
        <button id="tagged" type="button" className={`vault-browser-tab ${activeBrowserTab === 'tagged' ? 'active' : ''}`} onClick={() => { setActiveBrowserTab('tagged'); window.history.replaceState(null, '', '#tagged') }}>
          Tagged
          <span>{taggedCount}</span>
        </button>
      </div>

      {browserError ? <div className="status-panel error">{browserError}</div> : null}
      {browserLoading ? <div className="status-panel">Loading your photos...</div> : null}

      {!browserLoading && !browserError && activeBrowserTab === 'vault' ? (
        <div className="vault-folder-grid">
          {browserPayload.folders.map((folder) => (
            <article className="vault-folder-card" key={folder.path}>
              <div className="vault-folder-head">
                <div>
                  <h3>{folder.path}</h3>
                  <p>{folder.count} file{folder.count === 1 ? '' : 's'}</p>
                </div>
              </div>
              <div className="vault-item-grid">
                {folder.items.map((item) => (
                  <article className="vault-item-card" key={item.relativePath}>
                    <div className="vault-item-preview-grid">
                      <div className="vault-item-preview">
                        <button type="button" className="vault-preview-link vault-preview-button" onClick={() => onOpenImage({ imagePath: item.relativePath, filename: item.filename })}>
                          <SmartImage src={item.previewUrl} alt={item.filename} />
                        </button>
                      </div>
                      <div className="vault-item-preview debug">
                        {item.debugUrl ? (
                          <a href={item.debugUrl} target="_blank" rel="noreferrer" className="vault-preview-link">
                            <SmartImage src={item.debugUrl} alt={`${item.filename} debug boxes`} />
                          </a>
                        ) : <div className="vault-preview-empty">No boxes</div>}
                      </div>
                    </div>
                    <div className="vault-item-body">
                      <strong>{item.filename}</strong>
                      <span>Format: {item.mimeType || 'unknown'} original, WebP preview in app</span>
                      <span>{item.width} × {item.height} • {item.sizeLabel}</span>
                      <span>SHA-256: {item.sha256}</span>
                      <span>Detected: {item.detectedClasses?.length ? item.detectedClasses.join(', ') : 'none'} ({item.detectionCount})</span>
                      <span>Captured: {item.capturedAt || 'unknown'}</span>
                      <span>Original name: {item.originalFilename || item.filename}</span>
                      <div className="vault-item-actions">
                        <button type="button" className="text-link text-link-button" onClick={() => onOpenImage({ imagePath: item.relativePath, filename: item.filename })}>Open in app</button>
                        <a className="text-link" href={item.fullUrl} target="_blank" rel="noreferrer">Open file</a>
                        {item.debugUrl ? <a className="text-link" href={item.debugUrl} target="_blank" rel="noreferrer">Open debug boxes</a> : null}
                      </div>
                    </div>
                  </article>
                ))}
              </div>
            </article>
          ))}
        </div>
      ) : null}

      {!browserLoading && !browserError && activeBrowserTab === 'tagged' ? (
        taggedGroups.length ? (
          <div className="tagged-group-grid">
            {taggedGroups.map((group) => (
              <article className="tagged-group-card" key={group.label}>
                <div className="tagged-group-head">
                  <div>
                    <div className="eyebrow accent">Confirmed Label</div>
                    <h3>{group.label}</h3>
                    <p>{group.count} tagged subject{group.count === 1 ? '' : 's'}</p>
                  </div>
                </div>
                <div className="tagged-item-grid">
                  {group.items.map((item) => (
                    <article className="tagged-item-card" key={item.labelId || `${item.imagePath}-${item.detectionIndex}-${group.label}`}>
                      <button type="button" className="tagged-preview-link tagged-preview-button" onClick={() => onOpenImage({ imagePath: item.imagePath, filename: item.filename })}>
                        <SmartImage src={item.previewUrl} alt={`${group.label} in ${item.filename}`} />
                      </button>
                      <div className="tagged-item-body">
                        <strong>{item.filename}</strong>
                        <span>{item.detectedClass || 'unknown'} · detection #{Number(item.detectionIndex) + 1}</span>
                        <span>Captured: {item.capturedAt || 'unknown'}</span>
                        <div className="vault-item-actions">
                          <a className="text-link" href={item.fullUrl} target="_blank" rel="noreferrer">Open tagged crop</a>
                          <button type="button" className="text-link text-link-button" onClick={() => onOpenImage({ imagePath: item.imagePath, filename: item.filename })}>Open full image</button>
                          <a className="text-link" href={item.originalUrl || item.fullUrl} target="_blank" rel="noreferrer">Open original file</a>
                        </div>
                      </div>
                    </article>
                  ))}
                </div>
              </article>
            ))}
          </div>
        ) : (
          <div className="empty-panel compact">No saved tags yet. Save labels in the queue and they will appear here.</div>
        )
      ) : null}
    </section>
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
          <div className="eyebrow">Label queue</div>
          <h1>Review detections and save clear names</h1>
          <p>{lab.lab_insights.momentum || 'Each saved label gives the app a better example to learn from.'}</p>
          <div className="hero-actions">
            <a className="action action-primary" href="/">Back to photos</a>
            <a className="action action-secondary" href="/identity-lab">Open labeling tools</a>
            <a className="action action-secondary" href="#queue">Jump to Queue</a>
          </div>
        </div>
        <aside className="hero-sidebar">
          <div className="hero-chip">Ready now</div>
          <h2>{lab.lab_insights.suggestion_count} likely matches</h2>
          <p>{lab.lab_insights.action_prompt || 'The next strongest detections are waiting for a name.'}</p>
        </aside>
      </section>

      <section className="stats-grid lab-stats-grid">
        <article className="stat-card"><div className="stat-value">{lab.stats.total_detections}</div><div className="stat-label">Visible Detections</div><div className="stat-note">Everything currently in the review queue.</div></article>
        <article className="stat-card"><div className="stat-value">{lab.stats.confirmed_labels}</div><div className="stat-label">Saved</div><div className="stat-note">These already power named groups and suggestions.</div></article>
        <article className="stat-card"><div className="stat-value">{lab.stats.pending_labels}</div><div className="stat-label">Pending</div><div className="stat-note">These still need a name or a skip decision.</div></article>
        <article className="stat-card"><div className="stat-value">{lab.stats.ready_identities}</div><div className="stat-label">Ready groups</div><div className="stat-note">Names with enough examples to be useful.</div></article>
        <article className="stat-card"><div className="stat-value">{lab.lab_insights.suggestion_count}</div><div className="stat-label">Suggested matches</div><div className="stat-note">Pending detections that already look close to a saved name.</div></article>
      </section>

      {error ? <section className="status-panel error">{error}</section> : null}
      {loading ? <section className="status-panel">Refreshing the queue...</section> : null}

      <div className="lab-layout">
        <aside className="lab-sidebar">
          <section className="section-panel compact-panel">
            <div className="section-heading compact-heading"><div><div className="eyebrow accent">Start here</div><h2>Good next label</h2></div></div>
            {lab.lab_insights.focus_detection ? (
              <div className="focus-card">
                <SmartImage eager src={lab.lab_insights.focus_detection.crop_path ? buildMediaUrl(lab.lab_insights.focus_detection.crop_path, 'output', 'full') : buildMediaUrl(lab.lab_insights.focus_detection.image_path, 'output', 'full')} alt="Focus detection" />
                <div className="focus-body">
                  <strong>{lab.lab_insights.focus_detection.detected_class} candidate</strong>
                  <span>{lab.lab_insights.focus_detection.image_path.split('/').slice(-1)[0]}</span>
                  <span>Confidence {Math.round(lab.lab_insights.focus_detection.confidence * 100)}%. This is a good place to label next.</span>
                </div>
              </div>
            ) : (
              <div className="empty-panel compact">No pending detections left.</div>
            )}
          </section>

          <section className="section-panel compact-panel">
            <div className="section-heading compact-heading"><div><div className="eyebrow accent-coral">Saved names</div><h2>People and pets already known</h2></div></div>
            {lab.identity_collections.length ? (
              <div className="identity-stack">
                {lab.identity_collections.map((identity) => (
                  <article className="identity-mini-card" key={identity.name}>
                    <div className="identity-row"><h3>{identity.name}</h3><span className={`stage-pill ${identity.stage.tone}`}>{identity.stage.title}</span></div>
                    <div className="chip-row"><span>{identity.sample_count} photos</span><span>{identity.people_count} people</span><span>{identity.animal_count} animals</span></div>
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
              <div className="eyebrow accent">Queue</div>
              <h2>Label detections</h2>
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
                  {batchBusy ? 'Saving...' : `Save ${visibleSuggestions}`}
                </button>
                <button type="button" className="btn-reject batch-button" disabled={!pendingVisible || batchBusy} onClick={() => onBatchReject(filteredDetections)}>
                  {batchBusy ? 'Saving...' : `Skip ${pendingVisible}`}
                </button>
                <button type="button" className="quick-label undo-button" disabled={!lastAction || batchBusy} onClick={onUndo}>
                  Undo last action
                </button>
              </div>
            </div>

            <div className="queue-summary">
              <span className="summary-pill"><strong>{filteredDetections.length}</strong> visible</span>
              <span className="summary-pill"><strong>{lab.queue_summary.suggested}</strong> suggested</span>
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
