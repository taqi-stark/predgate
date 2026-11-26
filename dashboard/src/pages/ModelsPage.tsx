import { useState, useEffect } from 'react';
import { api, type Model } from '../api';

export default function ModelsPage() {
  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.listModels().then(setModels).catch(() => {}).finally(() => setLoading(false));
  }, []);

  const [copyText, setCopyText] = useState('');

  const handleReproduce = (runId: number) => {
    const cmd = `predgate reproduce ${runId}`;
    navigator.clipboard.writeText(cmd).then(() => {
      setCopyText(`Copied: ${cmd}`);
      setTimeout(() => setCopyText(''), 2000);
    });
  };

  return (
    <div>
      <h1 className="page-title">Models</h1>

      {copyText && (
        <div style={{ background: 'var(--pass-bg)', color: 'var(--pass)',
                      padding: '8px 16px', borderRadius: 6, marginBottom: 16, fontSize: 13 }}>
          {copyText}
        </div>
      )}

      {loading ? (
        <div>{[1, 2, 3].map(i => <div key={i} className="skeleton" style={{ height: 44, marginBottom: 4 }} />)}</div>
      ) : models.length === 0 ? (
        <div className="empty-state">
          <h3>No models registered</h3>
          <p>Run <code>predgate run</code> to create and register models.</p>
        </div>
      ) : (
        <table className="data-table">
          <thead>
            <tr>
              <th>Version</th><th>Run</th><th>Verdict</th><th>Branch</th>
              <th>Created</th><th>Champion</th><th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {models.map(m => (
              <tr key={m.version}>
                <td className="mono">{m.version}</td>
                <td>
                  <a
                    href={`/runs/${m.run_id}`}
                    className="mono"
                    style={{ color: 'var(--text-primary)', textDecoration: 'underline', textUnderlineOffset: '3px' }}
                  >
                    #{m.run_id}
                  </a>
                </td>
                <td><span className={`status-badge ${m.status}`}>{m.status}</span></td>
                <td>{m.branch || '—'}</td>
                <td className="time-ago">{new Date(m.created_at).toLocaleDateString()}</td>
                <td>{m.is_champion ? <span className="champion-badge">Champion</span> : '—'}</td>
                <td>
                  <button className="btn" style={{ marginRight: 8, padding: '4px 10px', fontSize: 12 }}
                          onClick={() => api.promoteRun(m.run_id).then(() => window.location.reload())}>
                    Promote
                  </button>
                  <button className="btn" style={{ padding: '4px 10px', fontSize: 12 }}
                          onClick={() => handleReproduce(m.run_id)}>
                    Reproduce
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
