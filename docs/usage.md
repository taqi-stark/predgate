# ModelGate — User Guide & Usage

This guide covers getting started, CLI commands, configuration options, GitHub Actions integration, and local Docker deployment.

---

## 1. Quickstart

Install ModelGate in a virtual environment:

```bash
git clone https://github.com/your-org/ModelGate.git
cd ModelGate
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the quality gate on the included credit-risk demo:

```bash
# 1. Generate synthetic dataset and train demo model
python demo/train.py

# 2. Run the full ModelGate quality check suite
modelgate run --config modelgate.yaml

# 3. View the generated report
open modelgate_output/report.html
```

---

## 2. CLI Reference

### `modelgate run`
Executes all configured quality checks against candidate and champion models.

```bash
modelgate run [OPTIONS]

Options:
  --config PATH      Path to modelgate.yaml (default: modelgate.yaml)
  --candidate PATH   Path or reference to candidate model artifact
  --champion PATH    Path or reference to champion model artifact
  --output-dir PATH  Directory to write verdict.json and report.html (default: modelgate_output)
  --post-pr-comment  Post verdict table to active GitHub PR (requires GITHUB_TOKEN)
```

Outputs:
- `modelgate_output/verdict.json`: Machine-readable check verdicts and scores.
- `modelgate_output/report.html`: Self-contained HTML report with interactive cards.
- Terminal output: Rich-formatted summary table and exit code (0 for PASS/WARN, 1 for FAIL).

### `modelgate bench`
Executes the fault-injection benchmark suite across multiple seeds and fault modes.

```bash
modelgate bench [OPTIONS]

Options:
  --seeds INT        Number of random seeds per fault (default: 30)
  --output-dir PATH  Directory to store benchmark results (default: docs/)
```

### `modelgate reproduce`
Reproduces a previous run's evaluation using recorded Git commit SHA, DVC data hash, and model parameters.

```bash
modelgate reproduce --run-id <RUN_ID>
# or
modelgate reproduce --data-hash <DVC_DATA_HASH>
```

---

## 3. Configuration (`modelgate.yaml`)

ModelGate is configured via a single declarative `modelgate.yaml` file:

```yaml
version: "1"

# MLflow Tracking Configuration
mlflow:
  tracking_uri: "file:./mlruns"
  experiment_name: "credit-risk-gate"

# Data contract validation (Pandera)
data_contract:
  enabled: true
  schema_path: "demo/contracts/schema.py"

# Drift detection on features & target
drift:
  enabled: true
  method: "psi"               # psi, ks, or wasserstein
  warn_at: 0.10
  fail_at: 0.25

# Performance slices with bootstrap confidence intervals
slices:
  enabled: true
  metric: "f1"
  by: ["region", "gender"]
  min_slice_size: 20
  max_drop: 0.03
  confidence: 0.95
  n_bootstrap: 50

# Behavioral & metamorphic tests
behavioral:
  enabled: true
  invariance_features: ["region"]
  directional_features:
    - feature: "income"
      direction: "increasing"

# Resource and latency budgets (interleaved batched median ratio + paired bootstrap CI; absolute ms optional)
budgets:
  enabled: true
  max_latency_ratio: 1.5
  latency_samples: 20
  latency_batch_size: 5
  latency_confidence: 0.95
  latency_n_bootstrap: 1000
  # p95_latency_ms: 100.0     # optional absolute ceiling in ms
  max_peak_mem_mb: 500.0
  max_model_mb: 50.0

# Target leakage detection
leakage:
  enabled: true
  warn_corr: 0.50
  fail_corr: 0.75
  warn_auc: 0.85
  fail_auc: 0.92

# Overall & subgroup label-shift detection
label_shift:
  enabled: true
  by: ["region", "gender", "age_bucket"]
  min_slice_size: 30
  warn_delta: 0.05
  fail_delta: 0.10
  alpha: 0.01

# Algorithmic fairness (paired bootstrap gap CI default + two-sided per-group selection-rate shift WARN; absolute DP opt-in relative to champion)
fairness:
  enabled: true
  protected: ["gender"]
  metric: "demographic_parity_diff"
  enforce_absolute: false
  warn_at: 0.05
  fail_at: 0.10
  max_gap_increase: 0.02
  fail_gap_increase: 0.05
  selection_rate_shift_warn: 0.02
  selection_rate_n_bootstrap: 1000
  min_group_size: 100
  min_group_positives: 30
  confidence: 0.95
  n_bootstrap: 200
```

---

## 4. GitHub Actions Integration

Use ModelGate's composite action in `.github/workflows/modelgate.yml`:

```yaml
name: ModelGate Evaluation

on:
  pull_request:
    paths:
      - 'src/**'
      - 'demo/**'
      - 'champion/**'
      - 'champion.dvc'
      - 'modelgate.yaml'
      - 'data/**'

jobs:
  modelgate:
    name: Run ModelGate
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pull-requests: write
      statuses: write
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Run ModelGate Quality Gate
        uses: ./
        with:
          config: 'modelgate.yaml'
          baseline-dir: 'champion'
          setup-data: 'python demo/train.py'
```

**Champion Integrity, Branch Protection & Fork PRs**:
- **Fail-Closed Verification & Pinned Tooling**: Both `.github/workflows/modelgate.yml` and `action.yml` fetch `origin/${BASE_REF}` (exiting non-zero if fetch fails), install ModelGate from the trusted committed base branch (`git archive`) or an explicit pinned `modelgate-spec` (never `pip install -e .` on the PR head), and verify `champion/model.pkl` against the base-branch pinned SHA-256 before running any PR code.
- **No Custom Approval Bypass**: Updates to `/champion/` and `/champion.dvc` are governed by GitHub Branch Protection + `.github/CODEOWNERS` (`@org/team` handles), not a spoofable workflow bypass flag.
- Require the **`Run ModelGate`** GitHub Actions check run in branch protection (not solely the `modelgate` commit status) if your repository accepts pull requests from forks.
- On fork PRs triggered by `on: pull_request`, GitHub automatically downgrades `GITHUB_TOKEN` to read-only (`contents: read`): `Run ModelGate` still verifies base-branch champion integrity, runs all 8 checks, fails the check run on regressions, and uploads `modelgate_output/report.html`, while `Post PR comment` and `Set commit status` gracefully emit warnings on HTTP 403 (`Resource not accessible by integration`) rather than crashing. Requiring `modelgate` commit status for fork PRs without a secondary `workflow_run` poster will leave passing fork PRs blocked waiting for that commit status.

---

## 5. Docker Compose Services

ModelGate includes a production-ready Docker Compose environment:

```bash
docker compose up -d
```

Services:
- **API**: http://localhost:8000 (FastAPI Swagger docs at `/docs`)
- **Dashboard**: http://localhost:3000 (React + Vite web interface)
- **MLflow Tracking Server**: http://localhost:5000
- **Seed Script**: Automatically seeds demo runs on startup.

---

## 6. Authentication, RBAC, Audit Log & HTTPS Deployment

ModelGate enforces bearer-token authentication and role-based access control (RBAC) across mutating API endpoints:

- **No Default Admin Token**: ModelGate never ships with a default or hardcoded admin token. Provision tokens explicitly via the CLI:
  ```bash
  modelgate users add alice --role engineer
  modelgate users add bob --role reviewer
  modelgate users add carol --role admin
  modelgate users list
  ```
  Alternatively, set `MODELGATE_ADMIN_TOKEN` in the server environment at startup to bootstrap the initial `admin` user. Plaintext tokens are displayed once at creation; only their SHA-256 hex digests are stored in the `users` table.
- **Role Permissions**:
  - `POST /runs`: `engineer`, `reviewer`, or `admin`
  - `POST /runs/{id}/approve` (or `/approval`): `reviewer` or `admin` (reviewer identity is bound from the authenticated token, not the request body). Overriding a `FAIL` run requires `admin` role and a justification comment of at least 20 characters.
  - `POST /runs/{id}/promote`: `admin`
  - `PUT /settings`: `admin`
  - `GET /audit`, `POST /users`, `GET /users`: `admin`
  - Read endpoints (`GET /runs`, `GET /models`, `GET /compare`, `GET /settings`, `GET /bench`, `GET /health`) remain unauthenticated for read-only dashboard access.
- **Append-Only Audit Log (`GET /audit`)**: Every allowed and denied (`401`/`403`/`400`) action on protected endpoints is recorded in the immutable `audit_log` table with `timestamp`, `actor`, `role`, `action`, `target`, `outcome`, and `detail`.
- **Restricted CORS & HTTPS Requirement**:
  - CORS origins are restricted to `http://localhost:3000,http://localhost:5173` by default (override via `MODELGATE_CORS_ORIGINS`), with methods restricted to `GET, POST, PUT, OPTIONS` and headers restricted to `Authorization, Content-Type, Accept`.
  - **HTTPS in Production**: Because bearer tokens are sent via `Authorization: Bearer <token>`, any production or non-localhost deployment **must** run behind TLS/HTTPS (e.g. Nginx/Caddy/ALB terminating TLS and setting `MODELGATE_CORS_ORIGINS=https://modelgate.example.com`) so tokens are encrypted in transit.
