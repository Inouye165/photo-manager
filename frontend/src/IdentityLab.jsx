import { startTransition, useEffect, useRef, useState } from 'react'

function preloadTask(task) {
  if (!task) {
    return
  }

  const urls = [
    task.candidate_image_url,
    task.candidate_full_url,
    ...(task.known_gallery || []).flatMap((item) => [item.imageUrl, item.fullUrl]),
  ].filter(Boolean)

  for (const url of urls) {
    const image = new Image()
    image.src = url
  }
}

function mergeTaskQueues(currentTasks, nextTasks) {
  const merged = [...currentTasks]
  const existingKeys = new Set(currentTasks.map((task) => task.relative_path))

  for (const task of nextTasks) {
    if (!existingKeys.has(task.relative_path)) {
      merged.push(task)
    }
  }

  return merged
}

export default function IdentityLab({ onToast }) {
  const [tasks, setTasks] = useState([])
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')
  const [learningMessage, setLearningMessage] = useState('')
  const currentTask = tasks[0] ?? null
  const nextTask = tasks[1] ?? null
  const isMountedRef = useRef(true)

  useEffect(() => {
    return () => {
      isMountedRef.current = false
    }
  }, [])

  useEffect(() => {
    preloadTask(currentTask)
  }, [currentTask])

  useEffect(() => {
    preloadTask(nextTask)
  }, [nextTask])

  useEffect(() => {
    void loadTasks(false)
  }, [])

  useEffect(() => {
    if (tasks.length <= 2 && !refreshing && !loading) {
      void loadTasks(true)
    }
  }, [tasks.length, refreshing, loading])

  async function loadTasks(append) {
    try {
      if (!append) {
        setLoading(true)
      } else {
        setRefreshing(true)
      }

      const response = await fetch('/api/identity/tasks')
      if (!response.ok) {
        throw new Error('Could not load the verification queue')
      }

      const payload = await response.json()
      if (!isMountedRef.current) {
        return
      }

      startTransition(() => {
        setTasks((currentTasks) => (append ? mergeTaskQueues(currentTasks, payload) : payload))
      })
      setError('')
    } catch (requestError) {
      if (isMountedRef.current) {
        setError(requestError.message || 'Could not load the verification queue')
      }
    } finally {
      if (isMountedRef.current) {
        setLoading(false)
        setRefreshing(false)
      }
    }
  }

  async function submitDecision(accepted) {
    if (!currentTask || submitting) {
      return
    }

    const decisionLabel = accepted ? currentTask.proposed_label : null
    const pendingMessage = accepted ? `Learning ${decisionLabel || 'identity'}...` : 'Removing incorrect suggestion...'
    setSubmitting(true)
    setLearningMessage(pendingMessage)
    onToast?.(pendingMessage)

    try {
      const response = await fetch('/api/identity/verify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          relative_path: currentTask.relative_path,
          subject_type: currentTask.subject_type,
          accepted,
          confirmed_label: accepted ? currentTask.proposed_label : null,
          schedule_fine_tune: accepted,
          metadata: currentTask.metadata,
        }),
      })

      if (!response.ok) {
        const result = await response.json().catch(() => ({}))
        throw new Error(result.error || 'Could not save the verification decision')
      }

      startTransition(() => {
        setTasks((currentTasks) => currentTasks.slice(1))
      })
      setLearningMessage('')
      onToast?.(accepted ? `${decisionLabel || 'Identity'} updated.` : 'Suggestion marked incorrect.')
    } catch (requestError) {
      setError(requestError.message || 'Could not save the verification decision')
      setLearningMessage('')
      onToast?.(requestError.message || 'Could not save the verification decision')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <section className="identity-lab-shell">
      <section className="hero-panel identity-lab-hero">
        <div>
          <div className="eyebrow">Identity Verification</div>
          <h1>Confirm the model. Tighten the memory.</h1>
          <p>Each Yes updates the live vector memory immediately. Each No clears weak guesses so the queue stays trustworthy.</p>
          <div className="hero-actions">
            <a className="action action-primary" href="/lab">Back to Label Queue</a>
            <a className="action action-secondary" href="/">Atlas Home</a>
          </div>
        </div>
        <aside className="hero-sidebar">
          <div className="hero-chip">Queue State</div>
          <h2>{tasks.length} task{tasks.length === 1 ? '' : 's'} ready</h2>
          <p>{refreshing ? 'Preloading the next set of candidates in the background.' : 'The next candidate is prefetched while you review the current one.'}</p>
        </aside>
      </section>

      {error ? <section className="status-panel error">{error}</section> : null}
      {learningMessage ? <section className="status-panel learning-status">{learningMessage}</section> : null}
      {loading ? <section className="status-panel">Loading the verification queue...</section> : null}

      {currentTask ? (
        <section className="identity-verify-grid">
          <article className="identity-candidate-card">
            <div className="section-heading compact-heading">
              <div>
                <div className="eyebrow accent">Candidate Image</div>
                <h2>{currentTask.proposed_label || 'Unknown identity'}</h2>
              </div>
            </div>

            <div className="identity-candidate-image-wrap">
              <img src={currentTask.candidate_full_url || currentTask.candidate_image_url} alt={currentTask.proposed_label || 'Candidate image'} />
            </div>

            <div className="identity-verify-meta">
              <span className={`stage-pill ${currentTask.status === 'auto_accept' ? 'dialed' : 'warming'}`}>{Math.round(currentTask.confidence * 100)}% confidence</span>
              <span className="hero-chip">{currentTask.subject_type}</span>
            </div>

            <p className="identity-verify-copy">
              {currentTask.status === 'auto_accept'
                ? 'The model is highly confident. Confirm to lock the embedding into the live memory.'
                : `The model is unsure enough to ask. Is this ${currentTask.proposed_label}?`}
            </p>

            <div className="identity-verify-actions">
              <button type="button" className="verify-button verify-yes" disabled={submitting || !currentTask.proposed_label} onClick={() => submitDecision(true)}>
                {submitting ? 'Learning...' : `Yes, this is ${currentTask.proposed_label}`}
              </button>
              <button type="button" className="verify-button verify-no" disabled={submitting} onClick={() => submitDecision(false)}>
                No, incorrect match
              </button>
            </div>
          </article>

          <aside className="identity-gallery-card">
            <div className="section-heading compact-heading">
              <div>
                <div className="eyebrow accent-coral">Known Gallery</div>
                <h2>What the model already knows</h2>
              </div>
            </div>

            {currentTask.known_gallery?.length ? (
              <div className="identity-gallery-grid">
                {currentTask.known_gallery.map((item, index) => (
                  <figure className="identity-gallery-item" key={`${item.label}-${index}`}>
                    <img src={item.imageUrl} alt={`${item.label} example ${index + 1}`} />
                    <figcaption>{item.label}</figcaption>
                  </figure>
                ))}
              </div>
            ) : (
              <div className="empty-panel compact">No gallery examples are available for this suggestion yet.</div>
            )}

            {currentTask.hits?.length ? (
              <div className="identity-score-list">
                {currentTask.hits.slice(0, 4).map((hit) => (
                  <div className="identity-score-row" key={hit.record_id}>
                    <strong>{hit.identity_label}</strong>
                    <span>{Math.round(((hit.score + 1) / 2) * 100)}%</span>
                  </div>
                ))}
              </div>
            ) : null}
          </aside>
        </section>
      ) : !loading ? (
        <section className="empty-panel">The verification queue is empty right now. Label a few more people or animals and the active-learning loop will refill automatically.</section>
      ) : null}
    </section>
  )
}