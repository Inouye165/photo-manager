import { startTransition, useEffect, useMemo, useState } from 'react'

function buildBatchItem(task, assignedLabel, status = 'confirmed', extra = {}) {
  return {
    image_path: task.image_path,
    detection_index: task.detection_index,
    assigned_label: assignedLabel,
    status,
    ...extra,
  }
}

function SearchResults({ results }) {
  if (!results.length) {
    return null
  }

  return (
    <section className="semantic-results-panel">
      <div className="section-heading compact-heading">
        <div>
          <div className="eyebrow accent-coral">Semantic Search</div>
          <h2>Nearest labeled matches</h2>
        </div>
      </div>
      <div className="semantic-results-grid">
        {results.map((result) => (
          <article className="semantic-result-card" key={result.record_id}>
            <div className="semantic-result-media">
              {result.preview_url ? <img src={result.preview_url} alt={result.label} loading="lazy" /> : <div className="semantic-result-fallback">No preview</div>}
            </div>
            <div className="semantic-result-body">
              <div className="semantic-result-topline">
                <strong>{result.label || 'Unknown'}</strong>
                <span>{Math.round(result.confidence)}%</span>
              </div>
              <p>{result.detected_class || result.subject_type}</p>
              <p>{result.image_path || result.relative_path}</p>
            </div>
          </article>
        ))}
      </div>
    </section>
  )
}

function BatchSuggestions({ suggestions, clusterDrafts, onClusterDraftChange, onApplyCluster, busy }) {
  if (!suggestions.length) {
    return null
  }

  return (
    <section className="batch-suggestions-panel">
      <div className="section-heading compact-heading">
        <div>
          <div className="eyebrow accent">Batch Suggest</div>
          <h2>Similar detections grouped for one-pass labeling</h2>
        </div>
      </div>
      <div className="batch-suggestion-grid">
        {suggestions.map((cluster) => {
          const draftValue = clusterDrafts[cluster.cluster_id] ?? cluster.suggested_label ?? ''

          return (
            <article className="batch-suggestion-card" key={cluster.cluster_id}>
              <div className="batch-suggestion-head">
                <div>
                  <div className="hero-chip">{cluster.detected_class}</div>
                  <h3>{cluster.member_count} similar crops</h3>
                  <p>{cluster.suggested_label ? `Suggested: ${cluster.suggested_label}` : 'No label suggestion yet'}</p>
                </div>
                {cluster.preview_url ? <img src={cluster.preview_url} alt={cluster.detected_class} /> : null}
              </div>
              <div className="batch-suggestion-actions">
                <input
                  value={draftValue}
                  onChange={(event) => onClusterDraftChange(cluster.cluster_id, event.target.value)}
                  placeholder="Apply one name to this cluster"
                />
                <button type="button" className="verify-button verify-yes compact" disabled={busy} onClick={() => onApplyCluster(cluster, draftValue)}>
                  Label cluster
                </button>
              </div>
            </article>
          )
        })}
      </div>
    </section>
  )
}

function TaskCard({ task, draftValue, clusterInfo, nameOptionsId, busy, onDraftChange, onSave, onReject, onApplyCluster }) {
  const suggestedLabel = task.proposed_label || ''

  return (
    <article className="identity-task-card">
      <div className="identity-task-media">
        {task.candidate_image_url ? <img src={task.candidate_image_url} alt={task.detected_class} loading="lazy" /> : <div className="identity-task-fallback">No crop</div>}
        <div className="identity-task-badges">
          <span className="subject-pill">{task.detected_class}</span>
          <span className={`stage-pill ${task.status === 'new_identity' ? 'spark' : task.status === 'auto_accept' ? 'dialed' : 'warming'}`}>
            {task.status === 'new_identity' ? 'New identity' : `${Math.round(task.confidence * 100)}%`}
          </span>
        </div>
      </div>

      <div className="identity-task-body">
        <div className="identity-task-header">
          <div>
            <h3>{task.image_path.split('/').slice(-1)[0]}</h3>
            <p>Detection #{task.detection_index + 1}</p>
          </div>
          {clusterInfo ? <span className="hero-chip">Cluster of {clusterInfo.member_count}</span> : null}
        </div>

        {suggestedLabel ? (
          <div className="identity-task-suggestion">
            <strong>{suggestedLabel}</strong>
            <span>{task.hits?.length ? `${Math.round(((task.hits[0].score + 1) / 2) * 100)}% nearest match` : 'Suggested from memory'}</span>
          </div>
        ) : (
          <div className="identity-task-suggestion muted">No strong suggestion yet. Type a new or existing name.</div>
        )}

        <div className="identity-task-inputs">
          <input
            list={nameOptionsId}
            value={draftValue}
            onChange={(event) => onDraftChange(task.detection_key, event.target.value)}
            placeholder="Start typing a person or pet name"
          />
        </div>

        {task.known_gallery?.length ? (
          <div className="identity-task-gallery">
            {task.known_gallery.slice(0, 3).map((item, index) => (
              <figure key={`${task.detection_key}-${index}`}>
                <img src={item.imageUrl} alt={item.label} loading="lazy" />
                <figcaption>{item.label}</figcaption>
              </figure>
            ))}
          </div>
        ) : null}

        <div className="identity-task-actions">
          <button type="button" className="verify-button verify-yes compact" disabled={busy} onClick={() => onSave(task, draftValue || suggestedLabel)}>
            Save label
          </button>
          {clusterInfo ? (
            <button type="button" className="verify-button verify-secondary compact" disabled={busy} onClick={() => onApplyCluster(clusterInfo, draftValue || suggestedLabel)}>
              Apply to {clusterInfo.member_count}
            </button>
          ) : null}
          <button type="button" className="verify-button verify-no compact" disabled={busy} onClick={() => onReject(task)}>
            {suggestedLabel ? `Not ${suggestedLabel}` : 'Reject'}
          </button>
        </div>
      </div>
    </article>
  )
}

function ConfirmedTaskCard({ task, draftValue, nameOptionsId, busy, editing, onStartEdit, onDraftChange, onSaveEdit }) {
  return (
    <article className="identity-task-card confirmed-task-card">
      <div className="identity-task-media">
        {task.candidate_image_url ? <img src={task.candidate_image_url} alt={task.detected_class} loading="lazy" /> : <div className="identity-task-fallback">No crop</div>}
        <div className="identity-task-badges">
          <span className="subject-pill">{task.detected_class}</span>
          <span className="stage-pill dialed">Confirmed</span>
        </div>
      </div>
      <div className="identity-task-body">
        <div className="identity-task-header">
          <div>
            <h3>{task.image_path.split('/').slice(-1)[0]}</h3>
            <p>Detection #{task.detection_index + 1}</p>
          </div>
          <button type="button" className="verify-button verify-secondary compact" disabled={busy} onClick={() => onStartEdit(task.record_id)}>
            Edit
          </button>
        </div>

        <div className="identity-task-suggestion">
          <strong>{task.assigned_label}</strong>
          <span>{task.captured_at ? `Captured ${task.captured_at}` : 'Confirmed training example'}</span>
        </div>

        {editing ? (
          <>
            <div className="identity-task-inputs">
              <input
                list={nameOptionsId}
                value={draftValue}
                onChange={(event) => onDraftChange(task.record_id, event.target.value)}
                placeholder="Rename this detection"
              />
            </div>
            <div className="identity-task-actions">
              <button type="button" className="verify-button verify-yes compact" disabled={busy} onClick={() => onSaveEdit(task, draftValue)}>
                Save edit
              </button>
            </div>
          </>
        ) : null}
      </div>
    </article>
  )
}

export default function IdentityLab({ onToast }) {
  const [payload, setPayload] = useState({ tasks: [], confirmed_tasks: [], name_options: [], batch_suggestions: [], stats: { task_count: 0, suggested_count: 0, cluster_count: 0, confirmed_count: 0 } })
  const [drafts, setDrafts] = useState({})
  const [clusterDrafts, setClusterDrafts] = useState({})
  const [editingConfirmed, setEditingConfirmed] = useState({})
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [searchQuery, setSearchQuery] = useState('')
  const [searching, setSearching] = useState(false)
  const [searchError, setSearchError] = useState('')
  const [searchResults, setSearchResults] = useState([])

  const nameOptionsId = 'identity-lab-name-options'
  const clusterMap = useMemo(() => new Map(payload.batch_suggestions.map((cluster) => [cluster.cluster_id, cluster])), [payload.batch_suggestions])

  useEffect(() => {
    void loadTasks()
  }, [])

  async function loadTasks() {
    try {
      setLoading(true)
      const response = await fetch('/api/identity/tasks')
      if (!response.ok) {
        throw new Error('Could not load batch labeling tasks')
      }

      const nextPayload = await response.json()
      startTransition(() => setPayload(nextPayload))
      setError('')
    } catch (requestError) {
      setError(requestError.message || 'Could not load batch labeling tasks')
    } finally {
      setLoading(false)
    }
  }

  function updateDraft(detectionKey, value) {
    setDrafts((currentDrafts) => ({ ...currentDrafts, [detectionKey]: value }))
  }

  function updateClusterDraft(clusterId, value) {
    setClusterDrafts((currentDrafts) => ({ ...currentDrafts, [clusterId]: value }))
  }

  async function submitBatch(items, successMessage) {
    if (!items.length) {
      return
    }

    try {
      setBusy(true)
      const response = await fetch('/api/label/batch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ items }),
      })

      if (!response.ok) {
        const result = await response.json().catch(() => ({}))
        throw new Error(result.error || 'Could not save labels')
      }

      await loadTasks()
      onToast?.(successMessage)
    } catch (requestError) {
      const message = requestError.message || 'Could not save labels'
      setError(message)
      onToast?.(message)
    } finally {
      setBusy(false)
    }
  }

  async function handleSave(task, labelValue) {
    const assignedLabel = (labelValue || '').trim()
    if (!assignedLabel) {
      onToast?.('Type a name or pick an existing one first.')
      return
    }

    await submitBatch([buildBatchItem(task, assignedLabel)], `Saved ${assignedLabel}.`)
  }

  async function handleReject(task) {
    await submitBatch([buildBatchItem(task, '', 'rejected', { suggested_label: task.proposed_label || null })], 'Rejected detection.')
  }

  async function handleApplyCluster(cluster, labelValue) {
    const assignedLabel = (labelValue || '').trim()
    if (!assignedLabel) {
      onToast?.('Add a name before applying it to the cluster.')
      return
    }

    const items = payload.tasks
      .filter((task) => cluster.detection_keys.includes(task.detection_key))
      .map((task) => buildBatchItem(task, assignedLabel))

    await submitBatch(items, `Applied ${assignedLabel} to ${items.length} detections.`)
  }

  function startEditConfirmed(recordId) {
    setEditingConfirmed((currentState) => ({ ...currentState, [recordId]: true }))
  }

  async function handleSaveConfirmedEdit(task, labelValue) {
    const assignedLabel = (labelValue || '').trim()
    if (!assignedLabel) {
      onToast?.('Type the replacement name before saving the edit.')
      return
    }

    await submitBatch([buildBatchItem(task, assignedLabel)], `Renamed ${task.assigned_label} to ${assignedLabel}.`)
    setEditingConfirmed((currentState) => ({ ...currentState, [task.record_id]: false }))
  }

  async function handleSemanticSearch(event) {
    event.preventDefault()
    const query = searchQuery.trim()
    if (!query) {
      setSearchResults([])
      setSearchError('')
      return
    }

    try {
      setSearching(true)
      const response = await fetch(`/api/search/semantic?q=${encodeURIComponent(query)}&limit=18`)
      if (!response.ok) {
        const result = await response.json().catch(() => ({}))
        throw new Error(result.error || 'Semantic search failed')
      }

      const result = await response.json()
      setSearchResults(result.results || [])
      setSearchError('')
    } catch (requestError) {
      setSearchResults([])
      setSearchError(requestError.message || 'Semantic search failed')
    } finally {
      setSearching(false)
    }
  }

  return (
    <section className="identity-lab-shell batch-lab-shell">
      <section className="hero-panel identity-lab-hero">
        <div>
          <div className="eyebrow">Batch Labeling</div>
          <h1>Label crops in parallel. Search the memory in plain language.</h1>
          <p>Every card is a detection crop. Type once, accept suggestions fast, and push cluster labels across visually similar detections in one pass.</p>
          <div className="hero-actions">
            <a className="action action-primary" href="/lab">Classic Queue</a>
            <a className="action action-secondary" href="/">Atlas Home</a>
          </div>
        </div>
        <aside className="hero-sidebar">
          <div className="hero-chip">Session State</div>
          <h2>{payload.stats.task_count} crops ready</h2>
          <p>{payload.stats.suggested_count} already have model label suggestions. {payload.stats.cluster_count} similarity clusters are ready for one-pass labeling. {payload.stats.confirmed_count} confirmed labels can be edited in place.</p>
        </aside>
      </section>

      <section className="semantic-search-panel">
        <form className="semantic-search-form" onSubmit={handleSemanticSearch}>
          <div>
            <div className="eyebrow accent">Semantic Search</div>
            <h2>Find labeled photos by description</h2>
          </div>
          <div className="semantic-search-controls">
            <input value={searchQuery} onChange={(event) => setSearchQuery(event.target.value)} placeholder="Try: small brown dog on grass, Ron in blue shirt, bird on branch" />
            <button type="submit" className="verify-button verify-yes compact" disabled={searching}>{searching ? 'Searching...' : 'Search'}</button>
          </div>
        </form>
        {searchError ? <div className="status-panel error">{searchError}</div> : null}
        <SearchResults results={searchResults} />
      </section>

      {error ? <section className="status-panel error">{error}</section> : null}
      {loading ? <section className="status-panel">Loading batch labeling tasks...</section> : null}

      <BatchSuggestions
        suggestions={payload.batch_suggestions}
        clusterDrafts={clusterDrafts}
        onClusterDraftChange={updateClusterDraft}
        onApplyCluster={handleApplyCluster}
        busy={busy}
      />

      <datalist id={nameOptionsId}>
        {payload.name_options.map((label) => <option value={label} key={label} />)}
      </datalist>

      {!loading && payload.tasks.length ? (
        <section className="identity-task-grid">
          {payload.tasks.map((task) => (
            <TaskCard
              key={task.detection_key}
              task={task}
              draftValue={drafts[task.detection_key] ?? task.proposed_label ?? ''}
              clusterInfo={task.batch_cluster_id ? clusterMap.get(task.batch_cluster_id) : null}
              nameOptionsId={nameOptionsId}
              busy={busy}
              onDraftChange={updateDraft}
              onSave={handleSave}
              onReject={handleReject}
              onApplyCluster={handleApplyCluster}
            />
          ))}
        </section>
      ) : null}

      {!loading && !payload.tasks.length ? (
        <section className="empty-panel">No pending detections are waiting for labels right now.</section>
      ) : null}

      {!loading && payload.confirmed_tasks.length ? (
        <section className="semantic-results-panel">
          <div className="section-heading compact-heading">
            <div>
              <div className="eyebrow accent">Confirmed Labels</div>
              <h2>Edit labels without leaving Identity Lab</h2>
            </div>
          </div>
          <section className="identity-task-grid confirmed-task-grid">
            {payload.confirmed_tasks.map((task) => (
              <ConfirmedTaskCard
                key={task.record_id}
                task={task}
                draftValue={drafts[task.record_id] ?? task.assigned_label ?? ''}
                nameOptionsId={nameOptionsId}
                busy={busy}
                editing={Boolean(editingConfirmed[task.record_id])}
                onStartEdit={startEditConfirmed}
                onDraftChange={updateDraft}
                onSaveEdit={handleSaveConfirmedEdit}
              />
            ))}
          </section>
        </section>
      ) : null}
    </section>
  )
}