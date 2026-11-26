import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { api, type Run } from '../api';

function fmtSigned(val: number | undefined | null): string {
  if (val === undefined || val === null || Number.isNaN(val)) return '—';
  const sign = val >= 0 ? '+' : '';
  return `${sign}${val.toFixed(3)}`;
}

export default function RunDetailPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [run, setRun] = useState<Run | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('data_contract');
  const [showApproval, setShowApproval] = useState(false);
  const [approvalComment, setApprovalComment] = useState('');

  useEffect(() => {
    if (!id) return;
    api.getRun(Number(id)).then(setRun).catch(() => {}).finally(() => setLoading(false));
  }, [id]);

  if (loading) return <div><div className="skeleton" style={{ height: 200 }} /></div>;
  if (!run) return <div className="empty-state"><h3>Run not found</h3></div>;

  const checks = run.check_results || [];
  const reasons = run.verdict?.reasons || [];
  const modelFamilyChange = run.verdict?.model_family_change;
  const slicesCheck = checks.find(c => c.name === 'slices');
  const inconclusiveEntries: any[] = slicesCheck?.details?.inconclusive_entries || [];
  const inconclusiveNotes: string[] = checks.flatMap(
    c => (c.details?.inconclusive_notes || []) as string[]
  );
  const hasInconclusive = inconclusiveEntries.length > 0 || inconclusiveNotes.length > 0;

  const tabs = [
    'data_contract', 'drift', 'leakage', 'label_shift',
    'slices', 'behavioral', 'budgets', 'fairness',
  ];
  const tabLabels: Record<string, string> = {
    data_contract: 'Data', drift: 'Drift', leakage: 'Leakage', label_shift: 'Label Shift',
    slices: 'Slices', behavioral: 'Behavioral', budgets: 'Budgets', fairness: 'Fairness',
  };

  const activeCheck = checks.find(c => c.name === activeTab);

  const handleApprove = async (decision: string) => {
    if (!id) return;
    try {
      await api.approveRun(Number(id), {
        reviewer: 'dashboard-user',
        decision,
        comment: approvalComment,
      });
      setShowApproval(false);
      const updated = await api.getRun(Number(id));
      setRun(updated);
    } catch (err: any) {
      alert(err.message);
    }
  };

  return (
    <div>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
        <h1 className="page-title" style={{ marginBottom: 0 }}>
          Run #{run.id}
        </h1>
        <span className={`status-badge ${run.status}`}>{run.status}</span>
        {run.branch && <span style={{ color: 'var(--text-secondary)' }}>branch {run.branch}</span>}
        {run.git_sha && <span className="mono" style={{ color: 'var(--text-muted)' }}>commit {run.git_sha.slice(0, 7)}</span>}
      </div>

      {/* Model Family Change Banner */}
      {modelFamilyChange?.changed && (
        <div className="banner-section">
          <div style={{ fontWeight: 600, marginBottom: 4 }}>
            Model family changed: <code>{modelFamilyChange.champion_family}</code> &rarr;{' '}
            <code>{modelFamilyChange.candidate_family}</code>
          </div>
          <div style={{ color: 'var(--text-secondary)', fontSize: 13 }}>
            Algorithm swaps can shift single-row inference latency and subgroup selection rates even without data or label bugs.
          </div>
        </div>
      )}

      {/* Prominent Inconclusive Slices Banner */}
      {hasInconclusive && (
        <div className="banner-section banner-warn">
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
            <span className="status-badge warn">INCONCLUSIVE SLICES</span>
            <span style={{ fontWeight: 500 }}>Wide confidence intervals straddling max_drop — a real drop may go undetected</span>
          </div>
          {inconclusiveEntries.length > 0 ? (
            <table className="data-table">
              <thead>
                <tr>
                  <th>Slice</th><th>N</th><th>Drop</th><th>95% CI (Drop)</th><th>CI Width</th>
                </tr>
              </thead>
              <tbody>
                {inconclusiveEntries.slice(0, 10).map((e: any, i: number) => (
                  <tr key={i}>
                    <td><code>{e.slice}</code></td>
                    <td className="mono">{e.n_rows}</td>
                    <td className="mono">{fmtSigned(e.diff)}</td>
                    <td className="mono">[{fmtSigned(e.ci_lower)}, {fmtSigned(e.ci_upper)}]</td>
                    <td className="mono" style={{ color: 'var(--warn)', fontWeight: 600 }}>
                      {e.ci_width?.toFixed(3)} (&gt; {e.max_ci_width?.toFixed(2)})
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <ul style={{ paddingLeft: 18 }}>
              {inconclusiveNotes.map((n, i) => <li key={i}>{n}</li>)}
            </ul>
          )}
        </div>
      )}

      {/* Reasons */}
      {reasons.length > 0 && (
        <div className={`reasons-list ${run.status === 'warn' ? 'warn-bg' : ''}`}>
          <h4>Reasons</h4>
          <ol>
            {reasons.map((r: any, i: number) => (
              <li key={i}>{r.message}</li>
            ))}
          </ol>
        </div>
      )}

      {/* Tabs */}
      <div className="tabs">
        {tabs.map(tab => (
          <div
            key={tab}
            className={`tab ${activeTab === tab ? 'active' : ''}`}
            onClick={() => setActiveTab(tab)}
          >
            {tabLabels[tab]}
            {checks.find(c => c.name === tab) && (
              <span className={`status-badge ${checks.find(c => c.name === tab)?.status}`}
                    style={{ marginLeft: 6, fontSize: 9, padding: '1px 6px' }}>
                {checks.find(c => c.name === tab)?.status}
              </span>
            )}
          </div>
        ))}
      </div>

      {/* Tab content */}
      {activeCheck && (
        <div className="card">
          <div className="card-title">{activeCheck.name.replace('_', ' ')} Check</div>
          <p style={{ marginBottom: 12 }}>{activeCheck.summary}</p>
          <p className="time-ago">Duration: {activeCheck.duration_s.toFixed(2)}s</p>

          {/* Metrics */}
          {Object.keys(activeCheck.metrics).length > 0 && (
            <table className="data-table" style={{ marginTop: 16 }}>
              <thead>
                <tr><th>Metric</th><th>Value</th></tr>
              </thead>
              <tbody>
                {Object.entries(activeCheck.metrics).map(([k, v]) => (
                  <tr key={k}>
                    <td>{k}</td>
                    <td className="mono">{typeof v === 'number' ? v.toFixed(4) : String(v)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          {/* Details - slice table */}
          {activeCheck.details?.slices && (
            <table className="data-table" style={{ marginTop: 16 }}>
              <thead>
                <tr>
                  <th>Slice</th><th>N</th><th>Champion</th><th>Candidate</th>
                  <th>Drop</th><th>95% CI (Drop)</th><th>Width</th><th>Status</th>
                </tr>
              </thead>
              <tbody>
                {activeCheck.details.slices.slice(0, 20).map((s: any, i: number) => (
                  <tr key={i}>
                    <td className="mono">{s.slice}</td>
                    <td className="mono">{s.n_rows}</td>
                    <td className="mono">{s.champion_metric?.toFixed(3)}</td>
                    <td className="mono">{s.candidate_metric?.toFixed(3)}</td>
                    <td className="mono">{fmtSigned(s.diff)}</td>
                    <td className="mono">[{fmtSigned(s.ci_lower)}, {fmtSigned(s.ci_upper)}]</td>
                    <td className="mono">{s.ci_width !== undefined ? s.ci_width.toFixed(3) : '—'}</td>
                    <td style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                      <span className={`status-badge ${s.verdict || 'pass'}`}>{s.verdict || 'pass'}</span>
                      {s.inconclusive && (
                        <span className="status-badge warn" title="Confidence interval width exceeds max_ci_width and straddles max_drop">
                          inconclusive
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          {/* Details - violations */}
          {activeCheck.details?.violations && activeCheck.details.violations.length > 0 && (
            <table className="data-table" style={{ marginTop: 16 }}>
              <thead>
                <tr><th>Dataset</th><th>Column</th><th>Check</th><th>Failure</th></tr>
              </thead>
              <tbody>
                {activeCheck.details.violations.map((v: any, i: number) => (
                  <tr key={i}>
                    <td>{v.dataset}</td><td>{v.column}</td>
                    <td>{v.check}</td><td>{v.failure_case}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          {/* Details - tests */}
          {activeCheck.details?.tests && (
            <table className="data-table" style={{ marginTop: 16 }}>
              <thead>
                <tr><th>Type</th><th>Feature</th><th>Result</th><th>Status</th></tr>
              </thead>
              <tbody>
                {activeCheck.details.tests.map((t: any, i: number) => (
                  <tr key={i}>
                    <td>{t.type}</td>
                    <td>{t.feature || '—'}</td>
                    <td className="mono">
                      {t.flip_rate !== undefined ? `flip: ${(t.flip_rate * 100).toFixed(1)}%` :
                       t.compliance_rate !== undefined ? `comply: ${(t.compliance_rate * 100).toFixed(1)}%` : '—'}
                    </td>
                    <td><span className={`status-badge ${t.status}`}>{t.status}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          {/* Details - fairness groups */}
          {activeCheck.details?.groups && (
            <table className="data-table" style={{ marginTop: 16 }}>
              <thead>
                <tr><th>Attribute</th><th>Metric</th><th>Candidate</th><th>Champion</th></tr>
              </thead>
              <tbody>
                {activeCheck.details.groups.map((g: any, i: number) => (
                  <tr key={i}>
                    <td>{g.protected_attribute}</td>
                    <td>{g.metric}</td>
                    <td className="mono">{g.candidate_value?.toFixed(4)}</td>
                    <td className="mono">{g.champion_value?.toFixed(4) || '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      {/* Lineage */}
      {run.lineage && Object.keys(run.lineage).length > 0 && (
        <div className="card">
          <div className="card-title">Lineage</div>
          <table className="data-table">
            <tbody>
              {run.lineage.git_sha && (
                <tr><td>Git SHA</td><td className="mono">{run.lineage.git_sha}</td></tr>
              )}
              <tr><td>Seed</td><td className="mono">{run.seed}</td></tr>
              {run.lineage.data_hashes && Object.entries(run.lineage.data_hashes).map(([k, v]) => (
                <tr key={k}><td>Data hash ({k})</td><td className="mono">{String(v).slice(0, 16)}...</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Actions */}
      <div style={{ display: 'flex', gap: 10, marginTop: 16 }}>
        <button className="btn" onClick={() => navigate(`/compare?a=${Number(id) - 1}&b=${id}`)}>
          Compare with champion
        </button>
        <button className="btn btn-primary" onClick={() => setShowApproval(true)}>
          Approve
        </button>
        <button className="btn btn-danger" onClick={() => setShowApproval(true)}>
          Reject
        </button>
        <button className="btn" disabled={run.status === 'fail'}
                onClick={() => api.promoteRun(Number(id)).then(() => alert('Promoted!')).catch(e => alert(e.message))}>
          Promote
        </button>
      </div>

      {/* Approval Modal */}
      {showApproval && (
        <div className="modal-overlay" onClick={() => setShowApproval(false)}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <h3>Approval Decision</h3>
            {reasons.length > 0 && (
              <div className={`reasons-list ${run.status === 'warn' ? 'warn-bg' : ''}`} style={{ marginBottom: 16 }}>
                <h4>Flagged issues</h4>
                <ol>
                  {reasons.map((r: any, i: number) => <li key={i}>{r.message}</li>)}
                </ol>
              </div>
            )}
            <textarea
              placeholder="Required comment (min 20 chars for FAIL override)..."
              value={approvalComment}
              onChange={e => setApprovalComment(e.target.value)}
            />
            <div className="modal-actions">
              <button className="btn" onClick={() => setShowApproval(false)}>Cancel</button>
              <button className="btn btn-danger" onClick={() => handleApprove('reject')}>Reject</button>
              <button className="btn btn-primary" onClick={() => handleApprove('approve')}>
                {run.status === 'fail' ? 'Approve with override' : 'Approve'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
