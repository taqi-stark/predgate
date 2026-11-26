import { useState, useEffect } from 'react';
import { api, getApiToken, setApiToken } from '../api';

export default function SettingsPage() {
  const [settings, setSettings] = useState<any>(null);
  const [editing, setEditing] = useState(false);
  const [editText, setEditText] = useState('');
  const [apiToken, setApiTokenState] = useState(() => getApiToken());
  const [loading, setLoading] = useState(true);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    api.getSettings().then(data => {
      setSettings(data);
      setEditText(JSON.stringify(data.config || {}, null, 2));
    }).catch(() => {}).finally(() => setLoading(false));
  }, []);

  const handleSave = async () => {
    try {
      const parsed = JSON.parse(editText);
      await api.updateSettings(parsed);
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
      setEditing(false);
    } catch (err: any) {
      alert(`Invalid JSON: ${err.message}`);
    }
  };

  if (loading) return <div className="skeleton" style={{ height: 300 }} />;

  return (
    <div>
      <h1 className="page-title">Settings</h1>

      {saved && (
        <div style={{ background: 'var(--pass-bg)', color: 'var(--pass)',
                      padding: '8px 16px', borderRadius: 6, marginBottom: 16, fontSize: 13 }}>
          ✓ Settings saved successfully
        </div>
      )}

      <div className="card">
        <div className="card-title">API Authentication (Bearer Token)</div>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 8 }}>
          <input
            type="password"
            className="filter-input"
            placeholder="Paste engineer / reviewer / admin bearer token"
            value={apiToken}
            onChange={e => {
              setApiTokenState(e.target.value);
              setApiToken(e.target.value);
            }}
            style={{ width: '100%' }}
          />
        </div>
      </div>

      <div className="card">
        <div className="card-title">Thresholds & Budgets</div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
          <div>
            <label style={{ fontSize: 12, color: 'var(--text-muted)', display: 'block', marginBottom: 4 }}>
              Drift PSI Warn Threshold
            </label>
            <input className="filter-input" defaultValue="0.10" style={{ width: '100%' }} />
          </div>
          <div>
            <label style={{ fontSize: 12, color: 'var(--text-muted)', display: 'block', marginBottom: 4 }}>
              Drift PSI Fail Threshold
            </label>
            <input className="filter-input" defaultValue="0.25" style={{ width: '100%' }} />
          </div>
          <div>
            <label style={{ fontSize: 12, color: 'var(--text-muted)', display: 'block', marginBottom: 4 }}>
              Max Slice Drop
            </label>
            <input className="filter-input" defaultValue="0.03" style={{ width: '100%' }} />
          </div>
          <div>
            <label style={{ fontSize: 12, color: 'var(--text-muted)', display: 'block', marginBottom: 4 }}>
              P95 Latency Budget (ms)
            </label>
            <input className="filter-input" defaultValue="25" style={{ width: '100%' }} />
          </div>
          <div>
            <label style={{ fontSize: 12, color: 'var(--text-muted)', display: 'block', marginBottom: 4 }}>
              Max Model Size (MB)
            </label>
            <input className="filter-input" defaultValue="50" style={{ width: '100%' }} />
          </div>
          <div>
            <label style={{ fontSize: 12, color: 'var(--text-muted)', display: 'block', marginBottom: 4 }}>
              Fairness DPD Warn
            </label>
            <input className="filter-input" defaultValue="0.05" style={{ width: '100%' }} />
          </div>
        </div>

        <div className="card-title">Raw Configuration (JSON)</div>
        <textarea
          style={{
            width: '100%', minHeight: 200, padding: 12,
            border: '1px solid var(--border)', borderRadius: 6,
            background: 'var(--bg-primary)', color: 'var(--text-primary)',
            fontFamily: "'JetBrains Mono', monospace", fontSize: 12,
            resize: 'vertical',
          }}
          value={editText}
          onChange={e => { setEditText(e.target.value); setEditing(true); }}
        />

        {editing && (
          <div style={{ marginTop: 12, display: 'flex', gap: 10 }}>
            <button className="btn btn-primary" onClick={handleSave}>Save Changes</button>
            <button className="btn" onClick={() => {
              setEditText(JSON.stringify(settings?.config || {}, null, 2));
              setEditing(false);
            }}>Cancel</button>
          </div>
        )}
      </div>

      <div className="card">
        <div className="card-title">Enabled Checks</div>
        {['data_contract', 'drift', 'slices', 'behavioral', 'budgets', 'fairness'].map(check => (
          <label key={check} style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8, cursor: 'pointer' }}>
            <input type="checkbox" defaultChecked style={{ accentColor: 'var(--text-primary)' }} />
            <span>{check.replace('_', ' ')}</span>
          </label>
        ))}
      </div>
    </div>
  );
}
