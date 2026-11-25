import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { api, type Run } from '../api';

function timeAgo(dateStr: string): string {
  const now = Date.now();
  const then = new Date(dateStr).getTime();
  const diff = now - then;
  if (diff < 60000) return 'just now';
  if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
  if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
  return `${Math.floor(diff / 86400000)}d ago`;
}

export default function RunsPage() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [loading, setLoading] = useState(true);
  const [statusFilter, setStatusFilter] = useState('');
  const [branchFilter, setBranchFilter] = useState('');
  const [search, setSearch] = useState('');
  const navigate = useNavigate();

  const fetchRuns = async () => {
    try {
      const data = await api.listRuns({
        status: statusFilter || undefined,
        branch: branchFilter || undefined,
      });
      setRuns(data);
    } catch {
      setRuns([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchRuns(); }, [statusFilter, branchFilter]);

  // Auto-refresh every 15 seconds
  useEffect(() => {
    const interval = setInterval(fetchRuns, 15000);
    return () => clearInterval(interval);
  }, [statusFilter, branchFilter]);

  const filtered = runs.filter(r =>
    !search || r.branch?.toLowerCase().includes(search.toLowerCase()) ||
    r.git_sha?.toLowerCase().includes(search.toLowerCase())
  );

  const counts = {
    pass: runs.filter(r => r.status === 'pass').length,
    warn: runs.filter(r => r.status === 'warn').length,
    fail: runs.filter(r => r.status === 'fail').length,
  };

  return (
    <div>
      <h1 className="page-title">Runs</h1>

      <div className="status-chips">
        <div className="status-chip" onClick={() => setStatusFilter('')}>
          All <span className="count">{runs.length}</span>
        </div>
        <div className="status-chip" onClick={() => setStatusFilter('pass')}>
          <span style={{ color: 'var(--pass)' }}>●</span> Pass <span className="count">{counts.pass}</span>
        </div>
        <div className="status-chip" onClick={() => setStatusFilter('warn')}>
          <span style={{ color: 'var(--warn)' }}>●</span> Warn <span className="count">{counts.warn}</span>
        </div>
        <div className="status-chip" onClick={() => setStatusFilter('fail')}>
          <span style={{ color: 'var(--fail)' }}>●</span> Fail <span className="count">{counts.fail}</span>
        </div>
      </div>

      <div className="filters">
        <select className="filter-select" value={statusFilter} onChange={e => setStatusFilter(e.target.value)}>
          <option value="">All statuses</option>
          <option value="pass">Pass</option>
          <option value="warn">Warn</option>
          <option value="fail">Fail</option>
        </select>
        <input
          className="filter-input"
          placeholder="Filter branch..."
          value={branchFilter}
          onChange={e => setBranchFilter(e.target.value)}
        />
        <input
          className="filter-input"
          placeholder="Search branch or commit..."
          value={search}
          onChange={e => setSearch(e.target.value)}
        />
      </div>

      {loading ? (
        <div>
          {[1, 2, 3].map(i => (
            <div key={i} className="skeleton" style={{ height: 44, marginBottom: 4 }} />
          ))}
        </div>
      ) : filtered.length === 0 ? (
        <div className="empty-state">
          <h3>No runs yet</h3>
          <p>Run <code>predgate run</code> to create your first quality gate run.</p>
        </div>
      ) : (
        <table className="data-table">
          <thead>
            <tr>
              <th>Status</th>
              <th>PR</th>
              <th>Branch</th>
              <th>Commit</th>
              <th>Summary</th>
              <th>Time</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map(run => (
              <tr key={run.id} onClick={() => navigate(`/runs/${run.id}`)}>
                <td><span className={`status-badge ${run.status}`}>{run.status}</span></td>
                <td>{run.pr_number ? `#${run.pr_number}` : '—'}</td>
                <td>{run.branch || '—'}</td>
                <td className="mono">{run.git_sha?.slice(0, 7) || '—'}</td>
                <td style={{ maxWidth: 400, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {run.summary}
                  {(run.metrics?.slices_n_inconclusive_slices || 0) > 0 && (
                    <span
                      className="status-badge warn"
                      style={{ marginLeft: 8, fontSize: 10, padding: '1px 6px' }}
                    >
                      {Math.round(run.metrics.slices_n_inconclusive_slices)} inconclusive
                    </span>
                  )}
                </td>
                <td className="time-ago">{timeAgo(run.created_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
