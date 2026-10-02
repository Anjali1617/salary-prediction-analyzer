import React, { useState, useEffect } from 'react'
import { BrowserRouter, Routes, Route, NavLink, Navigate } from 'react-router-dom'

import Dashboard          from './pages/Dashboard'
import PredictPage        from './pages/PredictPage'
import AnalyticsPage      from './pages/AnalyticsPage'
import ModelPerformancePage from './pages/ModelPerformancePage'
import DatasetPage        from './pages/DatasetPage'
import ModelInfoPage      from './pages/ModelInfoPage'
import AboutPage          from './pages/AboutPage'

// Simple SVG icons (inline — no external dependency for icons)
const Icon = ({ d, size = 18 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
    stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d={d} />
  </svg>
)

const NAV = [
  { to: '/dashboard',   label: 'Dashboard',         d: 'M3 3h7v7H3zm11 0h7v7h-7zM3 14h7v7H3zm11 0h7v7h-7z' },
  { to: '/predict',     label: 'Predict Salary',     d: 'M12 2a10 10 0 1 0 0 20A10 10 0 0 0 12 2zm0 6v6l4 2' },
  { to: '/analytics',   label: 'Analytics',          d: 'M18 20V10M12 20V4M6 20v-6' },
  { to: '/performance', label: 'Model Performance',  d: 'M2 20h20M5 20V10l7-7 7 7v10' },
  { to: '/dataset',     label: 'Dataset',            d: 'M4 6h16M4 10h16M4 14h16M4 18h16' },
  { to: '/model-info',  label: 'Model Information',  d: 'M9 3H5a2 2 0 0 0-2 2v4m6-6h10a2 2 0 0 1 2 2v4M9 3v18m0 0h10a2 2 0 0 0 2-2V9M9 21H5a2 2 0 0 1-2-2V9m0 0h18' },
  { to: '/about',       label: 'About',              d: 'M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z' },
]

const PAGE_TITLES = {
  '/dashboard':   'Dashboard',
  '/predict':     'Predict Salary',
  '/analytics':   'Analytics',
  '/performance': 'Model Performance',
  '/dataset':     'Dataset',
  '/model-info':  'Model Information',
  '/about':       'About',
}

function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <h1>Salary Predictor</h1>
      </div>
      <nav>
        {NAV.map(({ to, label, d }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}
          >
            <Icon d={d} size={16} />
            {label}
          </NavLink>
        ))}
      </nav>
    </aside>
  )
}

function TopBar({ path }) {
  const title = Object.entries(PAGE_TITLES).find(([k]) => path.startsWith(k))?.[1] ?? 'App'
  return (
    <header className="top-bar">
      <h2>{title}</h2>
      <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
        Salary Predictive Analyzer v1.0
      </span>
    </header>
  )
}

function AppShell() {
  const [path, setPath] = useState(window.location.pathname)
  useEffect(() => {
    const handler = () => setPath(window.location.pathname)
    window.addEventListener('popstate', handler)
    return () => window.removeEventListener('popstate', handler)
  }, [])

  // Also update on link clicks via MutationObserver isn't ideal;
  // react-router provides useLocation but we keep this simple.
  // For the TopBar title, we use a lightweight approach.
  return (
    <div className="app-layout">
      <Sidebar />
      <div className="main-area">
        <TopBar path={window.location.pathname} />
        <main className="page-content">
          <Routes>
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            <Route path="/dashboard"   element={<Dashboard />} />
            <Route path="/predict"     element={<PredictPage />} />
            <Route path="/analytics"   element={<AnalyticsPage />} />
            <Route path="/performance" element={<ModelPerformancePage />} />
            <Route path="/dataset"     element={<DatasetPage />} />
            <Route path="/model-info"  element={<ModelInfoPage />} />
            <Route path="/about"       element={<AboutPage />} />
          </Routes>
        </main>
      </div>
    </div>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <AppShell />
    </BrowserRouter>
  )
}
