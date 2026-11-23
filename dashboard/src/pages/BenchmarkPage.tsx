import { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { api } from '../api';

export default function BenchmarkPage() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getBenchmark().then(setData).catch(() => {}).finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="skeleton" style={{ height: 400 }} />;

  if (!data?.summary) {
    return (
      <div className="empty-state">
        <h3>No benchmark results</h3>
        <p>Run <code>predgate bench</code> to generate the fault injection benchmark.</p>
      </div>
    );
  }

  const chartData = Object.entries(data.summary).map(([fault, s]: [string, any]) => ({
    name: fault.replace('_', ' '),
    PredGate: Math.round(s.predgate_detection_rate * 100),
    Baseline: Math.round(s.baseline_detection_rate * 100),
  }));

  return (
    <div>
      <h1 className="page-title">Benchmark</h1>
      <p style={{ color: 'var(--text-secondary)', marginBottom: 24 }}>
        Fault injection benchmark comparing PredGate vs accuracy-only baseline.
      </p>

      {/* Chart */}
      <div className="card" style={{ marginBottom: 24 }}>
        <div className="card-title">Detection Rates (%)</div>
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis dataKey="name" tick={{ fill: 'var(--text-secondary)', fontSize: 11 }} />
            <YAxis tick={{ fill: 'var(--text-secondary)', fontSize: 11 }} domain={[0, 100]} />
            <Tooltip
              contentStyle={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)',
                             borderRadius: 8, color: 'var(--text-primary)' }}
            />
            <Legend />
            <Bar dataKey="PredGate" fill="#3b82f6" radius={[4, 4, 0, 0]} />
            <Bar dataKey="Baseline" fill="#6b7280" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Table */}
      <table className="data-table">
        <thead>
          <tr>
            <th>Fault</th><th>Description</th><th>PredGate</th>
            <th>Baseline (Acc)</th><th>Trials</th><th>Winner</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(data.summary).map(([fault, s]: [string, any]) => {
            const isNoop = fault === 'noop';
            const mgWins = isNoop
              ? s.predgate_detection_rate <= s.baseline_detection_rate
              : s.predgate_wins;

            return (
              <tr key={fault}>
                <td style={{ fontWeight: 600 }}>{fault}</td>
                <td style={{ color: 'var(--text-secondary)' }}>{s.description}</td>
                <td className="mono">
                  {(s.predgate_detection_rate * 100).toFixed(0)}%
                  {isNoop && ' FP'}
                </td>
                <td className="mono">
                  {(s.baseline_detection_rate * 100).toFixed(0)}%
                  {isNoop && ' FP'}
                </td>
                <td className="mono">{s.n_trials}</td>
                <td>
                  {mgWins
                    ? <span style={{ color: 'var(--pass)' }}>✅ PredGate</span>
                    : <span style={{ color: 'var(--text-muted)' }}>Baseline</span>
                  }
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
