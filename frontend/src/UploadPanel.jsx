import { useRef, useState } from 'react'

export default function UploadPanel({ onUploadComplete, onToast }) {
  const inputRef = useRef(null)
  const [dragging, setDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)

  function openFilePicker() {
    inputRef.current?.click()
  }

  async function handleFiles(fileList) {
    const file = fileList?.[0]
    if (!file || uploading) {
      return
    }

    setUploading(true)
    setProgress(0)
    setError('')
    setResult(null)
    onToast?.(`Uploading ${file.name}...`)

    try {
      const payload = await uploadFile(file, setProgress)
      setResult(payload)
      onToast?.(payload.duplicate ? `${payload.clean_name} is already in the vault.` : `${payload.clean_name} added to the vault.`)
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
          onChange={(event) => void handleFiles(event.target.files)}
        />

        <strong>Drop a photo here or upload from your machine.</strong>
        <p>The app hashes the file, blocks duplicates, stores the original in the source vault, mirrors it instantly, and starts identity review.</p>
        <button type="button" className="action action-primary upload-trigger" onClick={openFilePicker} disabled={uploading}>
          {uploading ? 'Uploading...' : 'Upload Photo'}
        </button>
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
            <strong>{result.duplicate ? 'Already in vault' : 'Upload complete'}</strong>
            <span>{result.clean_name}</span>
            <span>{result.detection_count} detections found. {result.uploaded_task_count} new verification task{result.uploaded_task_count === 1 ? '' : 's'} ready.</span>
          </div>
          <div className="upload-result-actions">
            <a className="action action-primary" href={result.identity_lab_url}>View in Identity Lab</a>
          </div>
        </div>
      ) : null}
    </section>
  )
}

function uploadFile(file, onProgress) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest()
    request.open('POST', '/api/ingest/upload')

    request.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        onProgress(Math.max(1, Math.round((event.loaded / event.total) * 100)))
      }
    }

    request.onload = () => {
      try {
        const payload = JSON.parse(request.responseText || '{}')
        if (request.status >= 200 && request.status < 300) {
          onProgress(100)
          resolve(payload)
          return
        }
        reject(new Error(payload.error || 'Upload failed'))
      } catch {
        reject(new Error('Upload failed'))
      }
    }

    request.onerror = () => reject(new Error('Upload failed'))

    const formData = new FormData()
    formData.append('file', file)
    request.send(formData)
  })
}