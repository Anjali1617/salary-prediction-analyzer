import React, { useEffect, useState, useRef } from 'react'
import { getDatasetSummary, uploadDataset, trainModel } from '../services/api'
import { fmtNum } from '../services/utils'

function fmtAmt(value) {
  if (value == null || isNaN(value)) return '—'
  return Number(value).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
}

export default function DatasetPage() {
  const [summary, setSummary] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [uploadMsg, setUploadMsg] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [training, setTraining] = useState(false)
  const [trainMsg, setTrainMsg] = useState(null)
  const fileRef = useRef(null)

  const load = () => {
    setLoading(true)
    getDatasetSummary()
      .then(r => setSummary(r.data))
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }
  useEffect(load, [])

  const handleUpload = async e => {
    const file = e.target.files?.[0]
    if (!file) return
    setUploading(true)
    setUploadMsg(null)
    try {
      const { data } = await uploadDataset(file)
      setUploadMsg({ ok: true, text: `${data.message} (${data.records} records)` })
      load()
    } catch (err) {
      const msg = err.response?.data?.detail || err.message
      setUploadMsg({ ok: false, text: typeof msg === 'string' ? msg : JSON.stringify(msg) })
    } finally {
      setUploading(false)
      if (fileRef.current) fileRef.current.value = ''
    }
  }

  const handleTrain = async () => {
    setTraining(true)
    setTrainMsg(null)
    try {
      const { data } = await trainModel()
      setTrainMsg({ ok: true, text: `Training complete: ${data.model_name}, R²=${data.metrics?.r2?.toFixed(4)}` })
    } catch (err) {
      setTrainMsg({ ok: false, text: err.response?.data?.detail || err.message })
    } finally {
      setTraining(false)
    }
  }

  if (loading) return (
    <div className="loading-overlay">
      <div className="spinner" style={{ borderTopColor: 'var(--ibm-blue)', borderColor: '#c6c6c6' }} />
      <span>Loading dataset info…</span>
    </div>
  )
  if (error) return <div className="error-msg">{error}</div>

  return (
    <div>
      {/* Summary cards */}
      <div className="kpi-grid" style={{ marginBottom: 24 }}>
        <div className="card"><div className="card-title">Total Records</div>
          <div className="card-value">{summary?.total_records}</div></div>
        <div className="card"><div className="card-title">Total Columns</div>
          <div className="card-value">{summary?.total_columns}</div></div>
        <div className="card"><div className="card-title">Missing Values</div>
          <div className="card-value">{summary?.missing_values}</div></div>
        <div className="card"><div className="card-title">Duplicates</div>
          <div className="card-value">{summary?.duplicate_records}</div></div>
        <div className="card"><div className="card-title">Target Column</div>
          <div className="card-value" style={{ fontSize: 18 }}>{summary?.target_column}</div></div>
      </div>

      {/* Salary stats */}
      <div className="chart-card" style={{ marginBottom: 20 }}>
        <div className="chart-title">Salary Statistics (from dataset)</div>
        <table className="data-table">
          <thead><tr><th>Metric</th><th>Value</th></tr></thead>
          <tbody>
            <tr><td>Mean</td><td>{fmtAmt(summary?.salary_mean)}</td></tr>
            <tr><td>Median</td><td>{fmtAmt(summary?.salary_median)}</td></tr>
            <tr><td>Minimum</td><td>{fmtAmt(summary?.salary_min)}</td></tr>
            <tr><td>Maximum</td><td>{fmtAmt(summary?.salary_max)}</td></tr>
            <tr><td>Std Deviation</td><td>{fmtAmt(summary?.salary_std)}</td></tr>
          </tbody>
        </table>
      </div>

      {/* Feature details */}
      <div className="charts-2col">
        <div className="chart-card">
          <div className="chart-title">Categorical Features — Unique Values</div>
          <table className="data-table">
            <thead><tr><th>Feature</th><th>Unique Values</th></tr></thead>
            <tbody>
              {Object.entries(summary?.categorical_uniques ?? {}).map(([col, vals]) => (
                <tr key={col}>
                  <td><strong>{col}</strong></td>
                  <td style={{ fontSize: 12 }}>{vals.join(', ')}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="chart-card">
          <div className="chart-title">Numerical Features — Ranges</div>
          <table className="data-table">
            <thead><tr><th>Feature</th><th>Min</th><th>Max</th><th>Mean</th></tr></thead>
            <tbody>
              {Object.entries(summary?.numerical_ranges ?? {}).map(([col, r]) => (
                <tr key={col}>
                  <td><strong>{col}</strong></td>
                  <td>{r.min}</td><td>{r.max}</td><td>{r.mean?.toFixed(1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* All columns */}
      <div className="chart-card" style={{ marginTop: 20 }}>
        <div className="chart-title">All Dataset Columns</div>
        <table className="data-table">
          <thead><tr><th>Column</th><th>Type</th><th>Role</th></tr></thead>
          <tbody>
            {(summary?.feature_columns ?? []).map(col => (
              <tr key={col}>
                <td>{col}</td>
                <td>{['Experience','Age'].includes(col) ? 'Numerical' : 'Categorical'}</td>
                <td><span className="badge badge-blue">Feature</span></td>
              </tr>
            ))}
            <tr>
              <td><strong>{summary?.target_column}</strong></td>
              <td>Numerical</td>
              <td><span className="badge badge-green">Target</span></td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* Admin: upload + retrain */}
      <div className="card" style={{ marginTop: 24, borderLeft: '4px solid var(--ibm-orange)' }}>
        <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 12 }}>
          Administrator: Upload New Dataset &amp; Retrain
        </div>
        <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 16 }}>
          Upload a replacement CSV. It must contain all 7 required columns.
          Upload does NOT automatically retrain — click "Train New Model" separately.
        </p>
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'center' }}>
          <label className="btn btn-secondary" style={{ cursor: 'pointer' }}>
            {uploading ? 'Uploading…' : 'Upload CSV'}
            <input ref={fileRef} type="file" accept=".csv" hidden onChange={handleUpload} disabled={uploading} />
          </label>
          <button className="btn btn-primary" onClick={handleTrain} disabled={training}>
            {training ? 'Training…' : 'Train New Model'}
          </button>
        </div>
        {uploadMsg && (
          <div className={uploadMsg.ok ? 'badge badge-green' : 'badge badge-red'}
            style={{ marginTop: 12, display: 'inline-block', padding: '6px 12px', fontSize: 12 }}>
            {uploadMsg.text}
          </div>
        )}
        {trainMsg && (
          <div className={trainMsg.ok ? 'badge badge-green' : 'badge badge-red'}
            style={{ marginTop: 8, display: 'inline-block', padding: '6px 12px', fontSize: 12 }}>
            {trainMsg.text}
          </div>
        )}
      </div>
    </div>
  )
}
