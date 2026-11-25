const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface Run {
  id: number;
  project: string;
  branch: string;
  pr_number: number | null;
  git_sha: string;
  status: string;
  summary: string;
  created_at: string;
  metrics: Record<string, number>;
  seed?: number;
  verdict?: Record<string, any>;
  lineage?: Record<string, any>;
  check_results?: CheckResult[];
  approvals?: Approval[];
}

export interface CheckResult {
  name: string;
  status: string;
  summary: string;
  metrics: Record<string, number>;
  details: Record<string, any>;
  duration_s: number;
}

export interface Approval {
  reviewer: string;
  decision: string;
  comment: string;
  is_override: boolean;
  created_at: string;
}

export interface SliceData {
  slice: string;
  column: string;
  value: string;
  n_rows: number;
  champion_metric: number;
  candidate_metric: number;
  diff: number;
  ci_lower: number;
  ci_upper: number;
  ci_width?: number;
  inconclusive?: boolean;
  verdict?: string;
}

export interface CompareData {
  run_a: { id: number; status: string; branch: string };
  run_b: { id: number; status: string; branch: string };
  metrics: { name: string; value_a: number; value_b: number; delta: number }[];
}

export interface Model {
  version: string;
  run_id: number;
  status: string;
  created_at: string;
  branch: string;
  git_sha: string;
  is_champion: boolean;
}

const TOKEN_STORAGE_KEY = 'predgate_api_token';

export function getApiToken(): string {
  return localStorage.getItem(TOKEN_STORAGE_KEY) || '';
}

export function setApiToken(token: string): void {
  if (token.trim()) {
    localStorage.setItem(TOKEN_STORAGE_KEY, token.trim());
  } else {
    localStorage.removeItem(TOKEN_STORAGE_KEY);
  }
}

async function fetchJSON<T>(path: string, options?: RequestInit): Promise<T> {
  const token = getApiToken();
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...((options?.headers as Record<string, string>) || {}),
  };
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  const resp = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers,
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail || resp.statusText);
  }
  return resp.json();
}

export const api = {
  listRuns: (params?: { status?: string; branch?: string }) => {
    const qs = new URLSearchParams();
    if (params?.status) qs.set('status', params.status);
    if (params?.branch) qs.set('branch', params.branch);
    return fetchJSON<Run[]>(`/runs?${qs}`);
  },

  getRun: (id: number) => fetchJSON<Run>(`/runs/${id}`),

  getRunSlices: (id: number) => fetchJSON<{ slices: SliceData[] }>(`/runs/${id}/slices`),

  compareRuns: (a: number, b: number) => fetchJSON<CompareData>(`/compare?a=${a}&b=${b}`),

  approveRun: (id: number, data: { reviewer: string; decision: string; comment: string }) =>
    fetchJSON<any>(`/runs/${id}/approval`, {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  promoteRun: (id: number) =>
    fetchJSON<any>(`/runs/${id}/promote`, { method: 'POST' }),

  listModels: () => fetchJSON<Model[]>('/models'),

  getSettings: () => fetchJSON<any>('/settings'),
  updateSettings: (config: any) =>
    fetchJSON<any>('/settings', {
      method: 'PUT',
      body: JSON.stringify({ config }),
    }),

  getBenchmark: () => fetchJSON<any>('/bench'),

  health: () => fetchJSON<{ status: string }>('/health'),
};
