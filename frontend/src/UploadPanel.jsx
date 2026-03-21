import { useRef, useState } from 'react'

const MAX_BATCH_BYTES = 48 * 1024 * 1024
const MAX_BATCH_FILES = 12

export default function UploadPanel({ onUploadComplete, onToast }) {
  const inputRef = useRef(null)
  const folderInputRef = useRef(null)
  const [dragging, setDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)

  function openFilePicker() {
    inputRef.current?.click()
  }

  function openFolderPicker() {
    folderInputRef.current?.click()
  }

  async function handleFiles(fileList) {
    const files = Array.from(fileList || [])
    if (!files.length || uploading) {
      return
    }

    const isBatchUpload = files.length > 1
    const uploadLabel = isBatchUpload ? `${files.length} photos` : files[0].name

    setUploading(true)
    setProgress(0)
    setError('')
    setResult(null)
    onToast?.(`Uploading ${uploadLabel}...`)

    try {
      const payload = await uploadFiles(files, setProgress)
      const summary = normalizeUploadResult(payload, files)
      setResult(summary)
      onToast?.(buildUploadToast(summary))
      onUploadComplete?.()
    } catch (requestError) {
      setError(requestError.message || 'Upload failed')
      onToast?.(requestError.message || 'Upload failed')
    } finally {
      setUploading(false)
    }
  }

  function onDrop(event) {
    event.preventDefault()
    setDragging(false)
    void handleFiles(event.dataTransfer.files)
  }

  return (
    <section className="section-panel upload-panel">
      <div className="section-heading compact-heading">
        <div>
          <div className="eyebrow accent">Upload To Vault</div>
          <h2>Add Photos Without Touching Folders</h2>
        </div>
      </div>

      <div
        className={`upload-dropzone ${dragging ? 'dragging' : ''} ${uploading ? 'busy' : ''}`}
        onDragOver={(event) => {
          event.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <input
          ref={inputRef}
          className="upload-file-input"
          type="file"
          accept=".jpg,.jpeg,.png,.bmp,.tiff,.tif,.heic,.heif,.webp"
          multiple
          onChange={(event) => void handleFiles(event.target.files)}
        />
        <input
          ref={folderInputRef}
          className="upload-file-input"
          type="file"
          accept=".jpg,.jpeg,.png,.bmp,.tiff,.tif,.heic,.heif,.webp"
          multiple
          webkitdirectory=""
          directory=""
          onChange={(event) => void handleFiles(event.target.files)}
        />

        <strong>Drop photos here, upload several at once, or pick a folder.</strong>
        <p>The app hashes every file, blocks duplicates, stores originals in the source vault, mirrors them instantly, and indexes detections into the vector store right after ingest.</p>
        <div className="upload-actions-row">
          <button type="button" className="action action-primary upload-trigger" onClick={openFilePicker} disabled={uploading}>
            {uploading ? 'Uploading...' : 'Upload Photos'}
          </button>
          <button type="button" className="action action-secondary upload-trigger" onClick={openFolderPicker} disabled={uploading}>
            Choose Folder
          </button>
        </div>
      </div>

      {uploading ? (
        <div className="upload-progress-card">
          <div className="upload-progress-row">
            <strong>Upload progress</strong>
            <span>{progress}%</span>
          </div>
          <div className="progress-track upload-track"><div className="progress-fill upload-fill" style={{ width: `${progress}%` }} /></div>
        </div>
      ) : null}

      {error ? <div className="status-panel error upload-status">{error}</div> : null}

      {result ? (
        <div className="upload-result-card">
          <div className="upload-result-copy">
            <strong>{result.title}</strong>
            <span>{result.summary}</span>
            <span>{result.detection_count} detections found. {result.uploaded_task_count} new verification task{result.uploaded_task_count === 1 ? '' : 's'} ready.</span>
            {result.duplicate_count ? <span>{result.duplicate_count} upload attempt{result.duplicate_count === 1 ? '' : 's'} were blocked because those files already exist in the vault.</span> : null}
            {result.failure_count ? <span>{result.failure_count} file{result.failure_count === 1 ? '' : 's'} failed and were skipped.</span> : null}
          </div>
          <div className="upload-result-actions">
            <a className="action action-primary" href={result.identity_lab_url}>View in Identity Lab</a>
          </div>
          {result.items?.filter((item) => !item.duplicate).length ? (
            <div className="upload-result-list">
              {result.items.filter((item) => !item.duplicate).slice(0, 6).map((item) => (
                <div className="upload-result-list-item" key={`${item.clean_name}-${item.sha256 || item.original_filename || item.clean_name}`}>
                  <strong>{item.original_filename || item.clean_name}</strong>
                  <span>Indexed as {item.clean_name} • {item.detection_count} detections</span>
                </div>
              ))}
              {result.items.filter((item) => !item.duplicate).length > 6 ? <div className="upload-result-list-item">+ {result.items.filter((item) => !item.duplicate).length - 6} more accepted file{result.items.filter((item) => !item.duplicate).length - 6 === 1 ? '' : 's'}</div> : null}
            </div>
          ) : result.duplicate_count ? <div className="empty-panel compact">No new files were added. Existing vault files blocked those upload attempts before ingest.</div> : null}
        </div>
      ) : null}
    </section>
  )
}

function uploadFiles(files, onProgress) {
  const batches = splitUploadBatches(files)
  const totalBytes = Math.max(1, files.reduce((sum, file) => sum + Math.max(file.size || 0, 1), 0))
  let completedBytes = 0

  return batches.reduce(
    (promise, batch) => promise.then(async (payloads) => {
      const batchBytes = Math.max(1, batch.reduce((sum, file) => sum + Math.max(file.size || 0, 1), 0))
      const batchPayload = await uploadBatch(batch, (loaded, total) => {
        const safeTotal = Math.max(total || batchBytes, 1)
        const overallLoaded = completedBytes + Math.min(loaded, safeTotal)
        onProgress(Math.max(1, Math.round((overallLoaded / totalBytes) * 100)))
      })
      completedBytes += batchBytes
      return [...payloads, batchPayload]
    }),
    Promise.resolve([])
  ).then((payloads) => {
    onProgress(100)
    return mergeBatchPayloads(payloads)
  })
}

function uploadBatch(files, onProgress) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest()
    request.open('POST', '/api/ingest/upload')

    request.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        onProgress(event.loaded, event.total)
      }
    }

    request.onload = () => {
      try {
        const payload = JSON.parse(request.responseText || '{}')
        if (request.status >= 200 && request.status < 300) {
          resolve(payload)
          return
        }
        reject(new Error(payload.error || 'Upload failed'))
      } catch {
        reject(new Error('Upload failed'))
      }
    }

    request.onerror = () => reject(new Error('Upload connection dropped. Try again; the previous batch may have exceeded the server limit or the local server restarted.'))

    const formData = new FormData()
    if (files.length === 1) {
      formData.append('file', files[0])
    } else {
      for (const file of files) {
        formData.append('files', file, file.webkitRelativePath || file.name)
      }
    }
    request.send(formData)
  })
}

function splitUploadBatches(files) {
  const batches = []
  let currentBatch = []
  let currentBytes = 0

  for (const file of files) {
    const fileBytes = Math.max(file.size || 0, 1)
    const shouldFlush = currentBatch.length > 0 && (
      currentBatch.length >= MAX_BATCH_FILES || currentBytes + fileBytes > MAX_BATCH_BYTES
    )

    if (shouldFlush) {
      batches.push(currentBatch)
      currentBatch = []
      currentBytes = 0
    }

    currentBatch.push(file)
    currentBytes += fileBytes
  }

  if (currentBatch.length) {
    batches.push(currentBatch)
  }

  return batches
}

function normalizeBatchPayload(payload) {
  if (payload?.items) {
    return payload
  }

  return {
    count: 1,
    uploaded_count: payload?.duplicate ? 0 : 1,
    duplicate_count: payload?.duplicate ? 1 : 0,
    failure_count: 0,
    detection_count: payload?.detection_count || 0,
    uploaded_task_count: payload?.uploaded_task_count || 0,
    identity_lab_url: payload?.identity_lab_url || '/identity-lab',
    items: payload ? [payload] : [],
    duplicate_items: payload?.duplicate ? [payload] : [],
    failures: [],
  }
}

function mergeBatchPayloads(payloads) {
  const normalized = payloads.map(normalizeBatchPayload)
  const items = normalized.flatMap((payload) => payload.items || [])
  const failures = normalized.flatMap((payload) => payload.failures || [])
  const duplicateItems = normalized.flatMap((payload) => payload.duplicate_items || [])

  return {
    count: items.length,
    uploaded_count: normalized.reduce((sum, payload) => sum + (payload.uploaded_count || 0), 0),
    duplicate_count: normalized.reduce((sum, payload) => sum + (payload.duplicate_count || 0), 0),
    failure_count: failures.length,
    detection_count: normalized.reduce((sum, payload) => sum + (payload.detection_count || 0), 0),
    uploaded_task_count: normalized.reduce((sum, payload) => sum + (payload.uploaded_task_count || 0), 0),
    identity_lab_url: normalized[normalized.length - 1]?.identity_lab_url || '/identity-lab',
    items,
    duplicate_items: duplicateItems,
    failures,
  }
}

function normalizeUploadResult(payload, files) {
  if (payload?.items) {
    return {
      ...payload,
      title: payload.count === 1 ? 'Upload complete' : `${payload.count} photos processed`,
      summary: payload.count === 1
        ? payload.items[0]?.duplicate
          ? `${payload.items[0]?.original_filename || files[0]?.name || '1 photo'} was blocked because the vault already contains that file content`
          : payload.items[0]?.clean_name || files[0]?.name || '1 photo'
        : `${payload.uploaded_count} new, ${payload.duplicate_count} blocked duplicate attempt${payload.duplicate_count === 1 ? '' : 's'}`,
    }
  }

  return {
    ...payload,
    count: 1,
    uploaded_count: payload.duplicate ? 0 : 1,
    duplicate_count: payload.duplicate ? 1 : 0,
    failure_count: 0,
    items: [payload],
    title: payload.duplicate ? 'Upload blocked' : 'Upload complete',
    summary: payload.duplicate
      ? `${payload.original_filename || payload.clean_name} was blocked because the vault already contains that file content`
      : payload.clean_name,
  }
}

function buildUploadToast(result) {
  if (result.count === 1 && result.items[0]) {
    return result.items[0].duplicate
      ? `${result.items[0].original_filename || result.items[0].clean_name} was blocked because that file is already in the vault.`
      : `${result.items[0].clean_name} added to the vault and indexed.`
  }

  return `Processed ${result.count} photos. ${result.uploaded_count} new, ${result.duplicate_count} blocked duplicate attempt${result.duplicate_count === 1 ? '' : 's'}, ${result.failure_count} failed.`
}