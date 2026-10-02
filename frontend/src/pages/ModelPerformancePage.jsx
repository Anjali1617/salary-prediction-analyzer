import React, { useEffect, useState } from 'react'
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, BarChart, Bar, Cell,
} from 'recharts'
import {
  getModelInfo, getModelComparison, getActualVsPredicted, trainModel
} from '../services/api'
import { fmtNum, CHART_COLORS } from '../services/utils'

/** Plain number formatter — no currency symbol */
function fmtAmt(value) {
  if (value == null || isNaN(value)) return '—'
  return Number(value).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
}

function MetricCard({ label, value, sub, color }) {
  return (
    <div className="card" style={{ borderLeft: `4px solid ${color || '#0f62fe'}` }}>
      <div className="card-title">{label}</div>
      <div className="card-value" style={{ fontSize: 22 }}>{value}</div>
      {sub && <div className="card-sub">{sub}</div>}
    </div>
  )
}

export default function ModelPerformancePage() {
  const [info, setInfo]   = useState(null)
  const [avp, setAvp]     = useState([])
  const [loading, setLoading] = useState(true)
  const [training, setTraining] = useState(false)
  const [trainMsg, setTrainMsg] = useState(null)
  const [error, setError] = useState(null)

  const loadData = () => {
    setLoading(true)
    Promise.all([
      getModelInfo().catch(() => null),
      getActualVsPredicted().catch(() => []),
    ]).then(([inf, a]) => {
      setInfo(inf?.data ?? null)
      setAvp(Array.isArray(a?.data) ? a.data : [])
    }).catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }

  useEffect(() => { loadData() }, [])

  const handleTrain = async () => {
    setTraining(true)
    setTrainMsg(null)
    try {
      const { data } = await trainModel(false)
      setTrainMsg(`✓ ${data.message}  (R²=${fmtNum(data.metrics?.r2, 4)})`)
      loadData()
    } catch (err) {
      setTrainMsg(`✗ Training failed: ${err.response?.data?.detail || err.message}`)
    } finally {
      setTraining(false)
    }
  }

  if (loading) return (
    <div className="loading-overlay">
      <div className="spinner" style={{ borderTopColor: 'var(--ibm-blue)', borderColor: '#c6c6c6' }} />
      <span>Loading model info…</span>
    </div>
  )

  if (!info) return (
    <div>
      <div className="empty-state">
        <p>No trained model found.</p>
        <p style={{ margin: '12px 0' }}>Click the button below to train the model now.</p>
        <button className="btn btn-primary" onClick={handleTrain} disabled={training}>
          {training ? 'Training…' : 'Train Model Now'}
        </button>
      </div>
    </div>
  )

  const m = info.metrics

  // Residual distribution (histogram)
  const residuals = avp.map(r => r.residual)
  const minR = Math.min(...residuals)
  const maxR = Math.max(...residuals)
  const BINS = 15
  const binWidth = (maxR - minR) / BINS
  const resHistogram = Array.from({ length: BINS }, (_, i) => {
    const lo = minR + i * binWidth
    const hi = lo + binWidth
    const count = residuals.filter(r => r >= lo && (i === BINS - 1 ? r <= hi : r < hi)).length
    return { bin: `${(lo/1000).toFixed(0)}k`, count }
  })

  return (
    <div>
      {/* Train button */}
      <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 20 }}>
        <button className="btn btn-primary" onClick={handleTrain} disabled={training}>
          {training ? (
            <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span className="spinner" /> Training…
            </span>
          ) : 'Retrain Model'}
        </button>
        {trainMsg && (
          <span style={{
            fontSize: 13,
            color: trainMsg.startsWith('✓') ? 'var(--ibm-green)' : 'var(--ibm-red)'
          }}>{trainMsg}</span>
        )}
      </div>

      {/* Header info */}
      <div className="card" style={{ marginBottom: 20, display: 'flex', flexWrap: 'wrap', gap: 24 }}>
        <div>
          <div className="card-title">Selected Model</div>
          <div style={{ fontSize: 18, fontWeight: 700 }}>{info.model_name}</div>
        </div>
        <div>
          <div className="card-title">Version</div>
          <div style={{ fontSize: 15, fontWeight: 600 }}>{info.model_version}</div>
        </div>
        <div>
          <div className="card-title">Training Date</div>
          <div style={{ fontSize: 15 }}>{new Date(info.training_date).toLocaleString()}</div>
        </div>
        <div>
          <div className="card-title">Training Records</div>
          <div style={{ fontSize: 15, fontWeight: 600 }}>{info.train_records}</div>
        </div>
        <div>
          <div className="card-title">Testing Records</div>
          <div style={{ fontSize: 15, fontWeight: 600 }}>{info.test_records}</div>
        </div>
      </div>

      {/* Metrics KPIs */}
      <div className="kpi-grid" style={{ marginBottom: 24 }}>
        <MetricCard label="R² Score" value={fmtNum(m.r2, 4)} color="#0f62fe" sub="Closer to 1.0 = better" />
        <MetricCard label="Adjusted R²" value={fmtNum(m.adjusted_r2, 4)} color="#009d9a" />
        <MetricCard label="MAE" value={fmtAmt(m.mae)} color="#8a3ffc" sub="Mean Absolute Error" />
        <MetricCard label="MSE" value={fmtNum(m.mse, 0)} color="#ff832b" sub="Mean Squared Error" />
        <MetricCard label="RMSE" value={fmtAmt(m.rmse)} color="#da1e28" sub="Root MSE" />
        {m.mape != null && <MetricCard label="MAPE" value={`${fmtNum(m.mape, 2)} %`} color="#6929c4" sub="Mean Abs % Error" />}
        {m.cv_mean != null && (
          <MetricCard
            label={`CV R² (${5}-fold)`}
            value={`${fmtNum(m.cv_mean, 4)} ± ${fmtNum(m.cv_std, 4)}`}
            color="#198038"
            sub="Cross-validation score"
          />
        )}
      </div>

      {/* Actual vs Predicted */}
      {avp.length > 0 && (
        <div className="charts-2col">
          <div className="chart-card">
            <div className="chart-title">Actual vs Predicted Salary</div>
            <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 8 }}>
              Points on the diagonal line = perfect prediction.
            </p>
            <ResponsiveContainer width="100%" height={280}>
              <ScatterChart margin={{ top: 10, right: 20, left: 10, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                <XAxis dataKey="actual" name="Actual"
                  tickFormatter={v => `${(v/1000).toFixed(0)}k`} tick={{ fontSize: 10 }}
                  label={{ value: 'Actual', position: 'insideBottom', offset: -4, fontSize: 11 }} />
                <YAxis dataKey="predicted" name="Predicted"
                  tickFormatter={v => `${(v/1000).toFixed(0)}k`} tick={{ fontSize: 10 }}
                  label={{ value: 'Predicted', angle: -90, position: 'insideLeft', fontSize: 11 }} />
                <Tooltip formatter={(v) => fmtAmt(v)} />
                <ReferenceLine
                  segment={[
                    { x: Math.min(...avp.map(r=>r.actual)), y: Math.min(...avp.map(r=>r.actual)) },
                    { x: Math.max(...avp.map(r=>r.actual)), y: Math.max(...avp.map(r=>r.actual)) },
                  ]}
                  stroke="#da1e28" strokeDasharray="4 4" />
                <Scatter data={avp} fill="#0f62fe" opacity={0.5} />
              </ScatterChart>
            </ResponsiveContainer>
          </div>

          <div className="chart-card">
            <div className="chart-title">Residual Distribution</div>
            <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 8 }}>
              Residual = Actual − Predicted. Centered near 0 is ideal.
            </p>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={resHistogram} margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                <XAxis dataKey="bin" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip formatter={v => `${v} records`} />
                <Bar dataKey="count" name="Count" fill="#0f62fe" radius={[3,3,0,0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Model comparison table */}
      {info.comparison?.length > 0 && (
        <div className="chart-card" style={{ marginTop: 20 }}>
          <div className="chart-title">Model Comparison (all candidates evaluated on test set)</div>
          <div style={{ overflowX: 'auto' }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>R²</th>
                  <th>MAE</th>
                  <th>RMSE</th>
                  <th>MAPE %</th>
                  <th>Selected</th>
                </tr>
              </thead>
              <tbody>
                {[...info.comparison]
                  .sort((a, b) => b.r2 - a.r2)
                  .map(row => (
                    <tr key={row.name} style={row.name === info.model_name
                      ? { background: '#edf5ff' } : {}}>
                      <td><strong>{row.name}</strong></td>
                      <td>{fmtNum(row.r2, 4)}</td>
                      <td>{fmtAmt(row.mae)}</td>
                      <td>{fmtAmt(row.rmse)}</td>
                      <td>{fmtNum(row.mape, 2)}</td>
                      <td>
                        {row.name === info.model_name
                          ? <span className="badge badge-green">✓ Best</span>
                          : ''}
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Comparison bar chart */}
      {info.comparison?.length > 0 && (
        <div className="chart-card" style={{ marginTop: 20 }}>
          <div className="chart-title">R² Comparison — All Models</div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart
              data={[...info.comparison].sort((a,b) => b.r2 - a.r2)}
              margin={{ top: 10, right: 20, left: 10, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
              <XAxis dataKey="name" tick={{ fontSize: 11 }} />
              <YAxis domain={[0, 1]} tickFormatter={v => v.toFixed(2)} tick={{ fontSize: 11 }} />
              <Tooltip formatter={v => fmtNum(v, 4)} />
              <Bar dataKey="r2" name="R² Score" radius={[4,4,0,0]}>
                {info.comparison.map((row, i) => (
                  <Cell
                    key={i}
                    fill={row.name === info.model_name ? '#0f62fe' : '#c6c6c6'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
}
