import React, { useEffect, useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
  ScatterChart, Scatter, LineChart, Line, Legend,
} from 'recharts'
import { getAnalytics } from '../services/api'
import { CHART_COLORS } from '../services/utils'

function fmtAmt(value) {
  if (value == null || isNaN(value)) return '—'
  return Number(value).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
}

/* ── Reusable box-summary bar chart ────────────────────────────────────────── */
function MedianBarChart({ data, label }) {
  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={data} margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
        <XAxis dataKey="group" tick={{ fontSize: 12 }} />
        <YAxis tickFormatter={v => `${(v/1000).toFixed(0)}k`} tick={{ fontSize: 11 }} />
        <Tooltip
          formatter={(v) => fmtAmt(v)}
          labelFormatter={l => `${label}: ${l}`}
        />
        <Bar dataKey="median" name="Median Salary" radius={[4,4,0,0]}>
          {(data || []).map((_, i) => (
            <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />
          ))}
        </Bar>
        <Bar dataKey="mean" name="Mean Salary" fill="#c6c6c6" radius={[4,4,0,0]} />
      </BarChart>
    </ResponsiveContainer>
  )
}

/* ── Salary distribution histogram ────────────────────────────────────────── */
function SalaryHistogram({ data }) {
  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={data} margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
        <XAxis dataKey="label" tick={{ fontSize: 9 }} interval={3} />
        <YAxis tick={{ fontSize: 11 }} />
        <Tooltip formatter={(v) => `${v} employees`} labelFormatter={l => `Salary: ${l}`} />
        <Bar dataKey="count" name="Employees" fill="#0f62fe" radius={[3,3,0,0]} />
      </BarChart>
    </ResponsiveContainer>
  )
}

/* ── Scatter ───────────────────────────────────────────────────────────────── */
function SalaryScatter({ data, xKey, xLabel }) {
  return (
    <ResponsiveContainer width="100%" height={260}>
      <ScatterChart margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
        <XAxis dataKey={xKey} name={xLabel} type="number" tick={{ fontSize: 11 }}
          label={{ value: xLabel, position: 'insideBottom', offset: -2, fontSize: 11 }} />
        <YAxis dataKey="salary" name="Salary" tickFormatter={v => `${(v/1000).toFixed(0)}k`}
          tick={{ fontSize: 11 }} />
        <Tooltip
          formatter={(v, name) => name === 'Salary' ? fmtAmt(v) : v}
          cursor={{ strokeDasharray: '3 3' }}
        />
        <Scatter data={data} fill="#0f62fe" opacity={0.6} />
      </ScatterChart>
    </ResponsiveContainer>
  )
}

/* ── Feature importance bar ─────────────────────────────────────────────── */
function FeatureImportanceChart({ data }) {
  if (!data || data.length === 0) return (
    <div className="empty-state">Feature importance not available — train the model first.</div>
  )
  const sorted = [...data].sort((a, b) => b.importance - a.importance)
  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={sorted} layout="vertical" margin={{ top: 5, right: 30, left: 80, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
        <XAxis type="number" tickFormatter={v => `${(v*100).toFixed(1)}%`} tick={{ fontSize: 11 }} />
        <YAxis type="category" dataKey="feature" tick={{ fontSize: 12 }} width={80} />
        <Tooltip formatter={v => `${(v*100).toFixed(2)}%`} />
        <Bar dataKey="importance" name="Importance" radius={[0,4,4,0]}>
          {sorted.map((_, i) => <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

export default function AnalyticsPage() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    getAnalytics()
      .then(r => setData(r.data))
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return (
    <div className="loading-overlay">
      <div className="spinner" style={{ borderTopColor: 'var(--ibm-blue)', borderColor: '#c6c6c6' }} />
      <span>Loading analytics…</span>
    </div>
  )
  if (error) return <div className="error-msg">{error}</div>

  return (
    <div>
      {/* Salary Distribution */}
      <div className="charts-1col">
        <div className="chart-card">
          <div className="chart-title">Salary Distribution (all records)</div>
          <SalaryHistogram data={data.salary_distribution} />
        </div>
      </div>

      {/* Experience and Age scatter */}
      <div className="charts-2col">
        <div className="chart-card">
          <div className="chart-title">Salary vs Experience</div>
          <SalaryScatter data={data.salary_vs_experience} xKey="experience" xLabel="Experience (years)" />
        </div>
        <div className="chart-card">
          <div className="chart-title">Salary vs Age</div>
          <SalaryScatter data={data.salary_vs_age} xKey="age" xLabel="Age" />
        </div>
      </div>

      {/* Categorical charts */}
      <div className="charts-2col">
        <div className="chart-card">
          <div className="chart-title">Salary by Education (Median &amp; Mean)</div>
          <MedianBarChart data={data.salary_by_education} label="Education" />
        </div>
        <div className="chart-card">
          <div className="chart-title">Salary by Job Title (Median &amp; Mean)</div>
          <MedianBarChart data={data.salary_by_job_title} label="Job Title" />
        </div>
      </div>

      <div className="charts-2col">
        <div className="chart-card">
          <div className="chart-title">Salary by Location (Median &amp; Mean)</div>
          <MedianBarChart data={data.salary_by_location} label="Location" />
        </div>
        <div className="chart-card">
          <div className="chart-title">
            Salary by Gender — <em style={{ fontWeight: 400, fontSize: 12, color: 'var(--text-muted)' }}>
              Observed dataset distribution only. Not a causal claim.
            </em>
          </div>
          <MedianBarChart data={data.salary_by_gender} label="Gender" />
        </div>
      </div>

      {/* Feature Importance */}
      <div className="charts-1col">
        <div className="chart-card">
          <div className="chart-title">Feature Importance (normalised, from trained model)</div>
          <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12 }}>
            Shows how much each feature contributed to the model's predictions.
            Importance is aggregated across one-hot-encoded categories.
            A high value means the model relies heavily on that feature — not that it causes salary changes.
          </p>
          <FeatureImportanceChart data={data.feature_importance} />
        </div>
      </div>
    </div>
  )
}
