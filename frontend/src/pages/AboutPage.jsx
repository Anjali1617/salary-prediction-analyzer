import React from 'react'

export default function AboutPage() {
  return (
    <div style={{ maxWidth: 780 }}>
      <div className="card" style={{ marginBottom: 20 }}>
        <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 8 }}>
          Salary Predictive Analyzer
        </h2>
        <p style={{ fontSize: 14, color: 'var(--text-muted)', marginBottom: 16 }}>
          An end-to-end salary prediction application built using a scikit-learn regression pipeline.
        </p>

        <div style={{ borderTop: '1px solid var(--border)', paddingTop: 16, marginTop: 8 }}>
          <h3 style={{ fontSize: 14, fontWeight: 700, marginBottom: 10 }}>What this app does</h3>
          <ul style={{ paddingLeft: 20, fontSize: 13, color: 'var(--text-muted)', lineHeight: 2 }}>
            <li>Loads and validates the <code>salary_data.csv</code> dataset (249 records, 7 columns)</li>
            <li>Trains multiple regression models (Linear, Ridge, Random Forest, Gradient Boosting, Extra Trees, XGBoost)</li>
            <li>Selects the best model by R² on a held-out 20% test set</li>
            <li>Deploys to a remote ML endpoint if credentials are configured, or runs locally as fallback</li>
            <li>Predicts salary from 6 input features — predictions are never hardcoded</li>
            <li>Shows interactive analytics charts built from the actual CSV data</li>
          </ul>
        </div>
      </div>

      <div className="card" style={{ marginBottom: 20 }}>
        <h3 style={{ fontSize: 14, fontWeight: 700, marginBottom: 12 }}>Machine Learning Architecture</h3>
        <div style={{ fontFamily: 'monospace', fontSize: 12, background: '#f4f4f4',
          padding: 16, borderRadius: 6, lineHeight: 1.8 }}>
          salary_data.csv<br/>
          ↓ Data Validation<br/>
          ↓ Train/Test Split (80/20, random_state=42)<br/>
          ↓ Preprocessing Pipeline<br/>
          &nbsp;&nbsp;&nbsp;• Numerical (Experience, Age) → Median Imputation → StandardScaler<br/>
          &nbsp;&nbsp;&nbsp;• Categorical (Education, Location, Job_Title, Gender) → OneHotEncoder<br/>
          ↓ Candidate Models Trained &amp; Evaluated<br/>
          ↓ Best Model Selected by R²<br/>
          ↓ 5-Fold Cross-Validation<br/>
          ↓ Artifacts Saved (model.pkl, pipeline.pkl, metadata.json)<br/>
          ↓ Remote ML Deployment (when credentials configured)<br/>
          ↓ Prediction API (FastAPI)<br/>
          ↓ React Frontend
        </div>
      </div>

      <div className="chart-card" style={{ marginBottom: 20 }}>
        <div className="chart-title">Technology Stack</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(160px,1fr))', gap: 12, marginTop: 12 }}>
          {[
            ['IBM', 'watsonx.ai / Watson ML'],
            ['ML', 'scikit-learn, pandas, numpy'],
            ['Backend', 'FastAPI + Uvicorn'],
            ['Validation', 'Pydantic v2'],
            ['Frontend', 'React + Vite'],
            ['Charts', 'Recharts'],
            ['Language', 'Python 3.11 / JS ES2022'],
            ['Tests', 'pytest + httpx'],
          ].map(([cat, tech]) => (
            <div key={cat} style={{ padding: 12, background: '#f4f4f4', borderRadius: 4 }}>
              <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '.05em', marginBottom: 4 }}>{cat}</div>
              <div style={{ fontSize: 13 }}>{tech}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="card" style={{ marginBottom: 20 }}>
        <div className="chart-title">Dataset</div>
        <table className="data-table" style={{ marginTop: 10 }}>
          <tbody>
            <tr><td><strong>File</strong></td><td>salary_data.csv</td></tr>
            <tr><td><strong>Records</strong></td><td>249</td></tr>
            <tr><td><strong>Columns</strong></td><td>7 (6 features + 1 target)</td></tr>
            <tr><td><strong>Target</strong></td><td>Salary (continuous, regression)</td></tr>
            <tr><td><strong>Missing values</strong></td><td>0</td></tr>
            <tr><td><strong>Duplicates</strong></td><td>0</td></tr>
          </tbody>
        </table>
      </div>

      <div className="card" style={{ borderLeft: '4px solid var(--ibm-orange)', fontSize: 13,
        color: 'var(--text-muted)' }}>
        <strong>Important:</strong> This application does not make automated employment decisions.
        Salary predictions are estimates from a statistical model trained on the provided dataset.
        Results should be interpreted in context and reviewed by qualified personnel.
      </div>
    </div>
  )
}
