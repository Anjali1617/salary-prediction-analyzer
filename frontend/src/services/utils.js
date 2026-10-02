/**
 * Shared utility: format salary number.
 * Uses the currency symbol returned by the backend (configurable via CURRENCY env var).
 */
export function formatSalary(value, symbol = '$') {
  if (value == null || isNaN(value)) return '—'
  return `${symbol}${Number(value).toLocaleString('en-US', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`
}

export function fmtNum(value, decimals = 2) {
  if (value == null || isNaN(value)) return '—'
  return Number(value).toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  })
}

export function fmtPct(value) {
  if (value == null || isNaN(value)) return '—'
  return `${(value * 100).toFixed(2)} %`
}

/** IBM Blue palette for charts */
export const CHART_COLORS = [
  '#0f62fe', '#009d9a', '#8a3ffc', '#ff832b',
  '#198038', '#da1e28', '#6929c4', '#1192e8',
]
