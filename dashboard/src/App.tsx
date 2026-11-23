import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom';
import { useState, useEffect } from 'react';
import RunsPage from './pages/RunsPage';
import RunDetailPage from './pages/RunDetailPage';
import ModelsPage from './pages/ModelsPage';
import BenchmarkPage from './pages/BenchmarkPage';
import SettingsPage from './pages/SettingsPage';
import ComparePage from './pages/ComparePage';
import './index.css';

function App() {
  const [theme, setTheme] = useState<'dark' | 'light'>('dark');

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  // Keyboard shortcut: g r for Runs
  useEffect(() => {
    let lastKey = '';
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (lastKey === 'g' && e.key === 'r') {
        window.location.hash = '#/';
      }
      lastKey = e.key;
      setTimeout(() => { lastKey = ''; }, 500);
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  return (
    <BrowserRouter>
      <div className="sidebar">
        <div className="sidebar-logo">
          <span>⊞ PredGate</span>
        </div>
        <nav className="sidebar-nav">
          <NavLink to="/" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`} end>
            📊 Runs
          </NavLink>
          <NavLink to="/models" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
            📦 Models
          </NavLink>
          <NavLink to="/benchmark" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
            🧪 Benchmark
          </NavLink>
          <NavLink to="/settings" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
            ⚙️ Settings
          </NavLink>
        </nav>
      </div>

      <div className="main-content">
        <div className="topbar">
          <div className="topbar-left">
            <span className="topbar-project">
              Project: <strong>credit-risk-demo</strong>
            </span>
            <span className="champion-badge">🏆 Champion: v14</span>
          </div>
          <button
            className="theme-toggle"
            onClick={() => setTheme(t => t === 'dark' ? 'light' : 'dark')}
          >
            {theme === 'dark' ? '☀️' : '🌙'}
          </button>
        </div>

        <div className="page-content">
          <Routes>
            <Route path="/" element={<RunsPage />} />
            <Route path="/runs/:id" element={<RunDetailPage />} />
            <Route path="/compare" element={<ComparePage />} />
            <Route path="/models" element={<ModelsPage />} />
            <Route path="/benchmark" element={<BenchmarkPage />} />
            <Route path="/settings" element={<SettingsPage />} />
          </Routes>
        </div>
      </div>
    </BrowserRouter>
  );
}

export default App;
