/**
 * Centralised API client.
 * The IBM_API_KEY is NEVER sent to the frontend.
 * All calls go to the FastAPI backend which holds the credentials.
 */
import axios from 'axios'

const BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: BASE,
  timeout: 60_000,
  headers: { 'Content-Type': 'application/json' },
})

// ── Endpoints ────────────────────────────────────────────────────────────────

export const getHealth         = () => api.get('/api/health')
export const getDatasetSummary = () => api.get('/api/dataset-summary')
export const getFeatures       = () => api.get('/api/features')
export const getAnalytics      = () => api.get('/api/analytics')
export const getModelInfo      = () => api.get('/api/model-info')
export const getModelMetrics   = () => api.get('/api/model-metrics')
export const getModelComparison = () => api.get('/api/model-comparison')
export const getActualVsPredicted = () => api.get('/api/actual-vs-predicted')
export const getWatsonStatus   = () => api.get('/api/watson-status')

export const predictSalary = (payload) => api.post('/api/predict', payload)
export const trainModel    = (pushToWatson = false) =>
  api.post(`/api/train?push_to_watson=${pushToWatson}`)

export const uploadDataset = (file) => {
  const form = new FormData()
  form.append('file', file)
  return api.post('/api/upload-dataset', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
}

export default api
