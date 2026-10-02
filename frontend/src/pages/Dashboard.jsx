import React, { useEffect, useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from 'recharts'
import { getDatasetSummary, getModelInfo } from '../services/api'
import { fmtNum, CHART_COLORS } from '../services/utils'

/** Format a number with commas, no currency symbol */
function fmtAmt(value) {
  if (value == null || isNaN(value)) return '—'
  return Number(value).toLocaleString('en-US', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })
}

function KpiCard({ title, value, sub, accent }) {
  // Measure value length to shrink font if it would overflow
  const strLen = String(value).length
  const valueFontSize = strLen > 14 ? 16 : strLen > 10 ? 20 : 28

  return (
    <div className="card" style={{
      borderTop: `3px solid ${accent || 'var(--ibm-blue)'}`,
      overflow: 'hidden',
      minWidth: 0,         /* allow flex/grid shrink */
    }}>
      <div className="card-title" style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
        {title}
      </div>
      <div className="card-value" style={{
        fontSize: valueFontSize,
        wordBreak: 'break-word',
        overflowWrap: 'anywhere',
        lineHeight: 1.25,
      }}>
        {value}
      </div>
      {sub && <div className="card-sub">{sub}</div>}
    </div>
  )
}

export default function Dashboard() {
  const [summary, setSummary] = useState(null)
  const [modelInfo, setModelInfo] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([getDatasetSummary(), getModelInfo().catch(() => null)])
      .then(([s, m]) => {
        setSummary(s.data)
        setModelInfo(m?.data ?? null)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return (
    <div className="loading-overlay">
      <div className="spinner" style={{ borderTopColor: 'var(--ibm-blue)', borderColor: '#c6c6c6' }} />
      <span>Loading dashboard…</span>
    </div>
  )
  if (error) return <div className="error-msg">{error}</div>

  return (
    <div>
      {/* Dataset KPIs */}
      <div style={{ marginBottom: 8, fontSize: 12, fontWeight: 600, textTransform: 'uppercase',
        letterSpacing: '.05em', color: 'var(--text-muted)' }}>Dataset Overview</div>
      <div className="kpi-grid">
        <KpiCard title="Total Records" value={summary?.total_records ?? '—'} accent="#0f62fe" />
        <KpiCard title="Features" value={summary?.feature_columns?.length ?? '—'} accent="#009d9a" />
        <KpiCard title="Missing Values" value={summary?.missing_values ?? '—'} accent="#198038" />
        <KpiCard title="Duplicates" value={summary?.duplicate_records ?? '—'} accent="#198038" />
        <KpiCard title="Target" value="Salary" accent="#8a3ffc" />
      </div>

      {/* Salary KPIs */}
      <div style={{ marginBottom: 8, fontSize: 12, fontWeight: 600, textTransform: 'uppercase',
        letterSpacing: '.05em', color: 'var(--text-muted)' }}>Salary Statistics</div>
      <div className="kpi-grid">
        <KpiCard title="Mean Salary" value={fmtAmt(summary?.salary_mean)} accent="#0f62fe" />
        <KpiCard title="Median Salary" value={fmtAmt(summary?.salary_median)} accent="#0f62fe" />
        <KpiCard title="Min Salary" value={fmtAmt(summary?.salary_min)} accent="#198038" />
        <KpiCard title="Max Salary" value={fmtAmt(summary?.salary_max)} accent="#ff832b" />
        <KpiCard title="Std Deviation" value={fmtAmt(summary?.salary_std)} accent="#6f6f6f" />
      </div>

      {/* Model KPIs */}
      {modelInfo && (
        <>
          <div style={{ marginBottom: 8, fontSize: 12, fontWeight: 600, textTransform: 'uppercase',
            letterSpacing: '.05em', color: 'var(--text-muted)' }}>Model Performance</div>
          <div className="kpi-grid">
            <KpiCard title="Best Model" value={modelInfo.model_name} accent="#8a3ffc" />
            <KpiCard title="R² Score" value={fmtNum(modelInfo.metrics?.r2, 4)} accent="#0f62fe"
              sub="Higher is better (max 1.0)" />
            <KpiCard title="RMSE" value={fmtAmt(modelInfo.metrics?.rmse)} accent="#ff832b"
              sub="Root Mean Squared Error" />
            <KpiCard title="MAE" value={fmtAmt(modelInfo.metrics?.mae)} accent="#009d9a"
              sub="Mean Absolute Error" />
            <KpiCard title="Model Version" value={modelInfo.model_version} accent="#6f6f6f"
              sub={`Trained: ${new Date(modelInfo.training_date).toLocaleDateString()}`} />
          </div>
        </>
      )}

      {/* Categorical uniques info */}
      <div className="charts-2col">
        <div className="chart-card">
          <div className="chart-title">Dataset Column Info</div>
          <table className="data-table">
            <thead>
              <tr><th>Column</th><th>Type</th><th>Role</th></tr>
            </thead>
            <tbody>
              {['Education','Experience','Location','Job_Title','Age','Gender'].map(col => (
                <tr key={col}>
                  <td>{col}</td>
                  <td>{['Experience','Age'].includes(col) ? 'Numerical' : 'Categorical'}</td>
                  <td><span className="badge badge-blue">Feature</span></td>
                </tr>
              ))}
              <tr>
                <td><strong>Salary</strong></td>
                <td>Numerical</td>
                <td><span className="badge badge-green">Target</span></td>
              </tr>
            </tbody>
          </table>
        </div>

        <div className="chart-card">
          <div className="chart-title">Unique Categorical Values</div>
          <table className="data-table">
            <thead>
              <tr><th>Feature</th><th>Unique Values</th></tr>
            </thead>
            <tbody>
              {Object.entries(summary?.categorical_uniques ?? {}).map(([col, vals]) => (
                <tr key={col}>
                  <td>{col}</td>
                  <td style={{ fontSize: 12 }}>{vals.join(', ')}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
