import React, { useEffect, useState } from 'react'
import { getModelInfo } from '../services/api'
import { fmtNum } from '../services/utils'

/** Plain number formatter — no currency symbol */
function fmtAmt(value) {
  if (value == null || isNaN(value)) return '—'
  return Number(value).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
}

export default function ModelInfoPage() {
  const [info, setInfo] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    getModelInfo()
      .then(r => setInfo(r.data))
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return (
    <div className="loading-overlay">
      <div className="spinner" style={{ borderTopColor: 'var(--ibm-blue)', borderColor: '#c6c6c6' }} />
      <span>Loading model info…</span>
    </div>
  )
  if (error) return <div className="error-msg">{error}</div>

  return (
    <div style={{ maxWidth: 900 }}>
      {info ? (
        <>
          <div className="card" style={{ marginBottom: 20 }}>
            <div className="chart-title">Local Model Metadata</div>
            <table className="data-table" style={{ marginTop: 12 }}>
              <tbody>
                <tr><td><strong>Model Name</strong></td><td>{info.model_name}</td></tr>
                <tr><td><strong>Version</strong></td><td>{info.model_version}</td></tr>
                <tr><td><strong>Training Date</strong></td>
                  <td>{new Date(info.training_date).toLocaleString()}</td></tr>
                <tr><td><strong>Dataset</strong></td><td>{info.dataset_name}</td></tr>
                <tr><td><strong>Target Column</strong></td>
                  <td><span className="badge badge-green">{info.target_column}</span></td></tr>
                <tr><td><strong>Training Records</strong></td><td>{info.train_records}</td></tr>
                <tr><td><strong>Testing Records</strong></td><td>{info.test_records}</td></tr>
                <tr><td><strong>Deployment Status</strong></td>
                  <td><span className={`badge badge-${info.deployment_status === 'watson' ? 'green' : 'orange'}`}>
                    {info.deployment_status === 'watson' ? 'Watson Deployed' : 'Local Only'}
                  </span></td></tr>
              </tbody>
            </table>
          </div>

          <div className="card" style={{ marginBottom: 20 }}>
            <div className="chart-title">Evaluation Metrics (test set)</div>
            <table className="data-table" style={{ marginTop: 12 }}>
              <tbody>
                <tr><td><strong>R² Score</strong></td><td>{fmtNum(info.metrics?.r2, 6)}</td></tr>
                <tr><td><strong>Adjusted R²</strong></td><td>{fmtNum(info.metrics?.adjusted_r2, 6)}</td></tr>
                <tr><td><strong>MAE</strong></td><td>{fmtAmt(info.metrics?.mae)}</td></tr>
                <tr><td><strong>MSE</strong></td><td>{fmtNum(info.metrics?.mse, 2)}</td></tr>
                <tr><td><strong>RMSE</strong></td><td>{fmtAmt(info.metrics?.rmse)}</td></tr>
                <tr><td><strong>MAPE</strong></td><td>{fmtNum(info.metrics?.mape, 2)} %</td></tr>
                {info.metrics?.cv_mean != null && (
                  <tr><td><strong>CV R² (5-fold)</strong></td>
                    <td>{fmtNum(info.metrics.cv_mean, 4)} ± {fmtNum(info.metrics.cv_std, 4)}</td></tr>
                )}
              </tbody>
            </table>
          </div>

          <div className="card" style={{ marginBottom: 20 }}>
            <div className="chart-title">Feature List</div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 12 }}>
              {info.feature_list.map(f => (
                <span key={f} className="badge badge-blue" style={{ fontSize: 12 }}>{f}</span>
              ))}
            </div>
          </div>

          <div className="card">
            <div className="chart-title">Feature Importance</div>
            <table className="data-table" style={{ marginTop: 12 }}>
              <thead><tr><th>Rank</th><th>Feature</th><th>Importance (normalised)</th></tr></thead>
              <tbody>
                {(info.feature_importance ?? []).map((fi, i) => (
                  <tr key={fi.feature}>
                    <td>{i + 1}</td>
                    <td>{fi.feature}</td>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <div style={{
                          height: 10,
                          width: `${Math.round(fi.importance * 200)}px`,
                          background: 'var(--ibm-blue)',
                          borderRadius: 3,
                          minWidth: 4,
                        }} />
                        {(fi.importance * 100).toFixed(2)} %
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      ) : (
        <div className="empty-state">No trained model found. Go to Model Performance and click Retrain.</div>
      )}
    </div>
  )
}
