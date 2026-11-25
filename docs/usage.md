# PredGate — User Guide & Usage

This guide covers getting started, CLI commands, configuration options, threshold presets, GitHub Actions integration, and local Docker deployment.

---

## 1. Quickstart

Install PredGate in a virtual environment (requires Python 3.10+; on Debian/Ubuntu install `git`, `python3-pip`, and `python3-venv` if missing):

```bash
git clone https://github.com/taqi-stark/predgate.git
cd predgate
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the quality gate on the included credit-risk demo:

```bash
# 1. Generate synthetic dataset and train demo model
python demo/train.py

# 2. Verify environment and configuration
predgate doctor --config predgate.yaml

# 3. Run the full PredGate quality check suite (default / balanced thresholds)
predgate run --config predgate.yaml

# 4. View the generated report (macOS: open | Linux desktop: xdg-open | or open in a browser)
open predgate_output/report.html   # Linux: xdg-open predgate_output/report.html
```

---

## 2. CLI Reference

### `predgate run`
Executes all configured quality checks against candidate and champion models.

```bash
predgate run [OPTIONS]

Options:
  -c, --config PATH    Path to predgate.yaml (default: predgate.yaml)
  --preset TEXT        Optional threshold preset override: strict, balanced, or quiet
  -b, --branch TEXT    Branch name
  --pr INT             Pull request number
  -o, --output PATH    Directory to write verdict.json, pr_comment.md, report.html, payload.json
                       (default: predgate_output)
```

Outputs:
- `predgate_output/verdict.json`: Machine-readable check verdicts, metrics, inconclusive slice entries, and `model_family_change` metadata.
- `predgate_output/pr_comment.md`: GitHub-flavored Markdown PR comment with prominent **Model family changed** and **Inconclusive Slices** callouts.
- `predgate_output/report.html`: Self-contained HTML report with highlighted callout banners.
- `predgate_output/payload.json`: Full API ingestion payload consumed by the FastAPI backend and React dashboard.
- Terminal output: Rich-formatted summary table and exit code (`0` for `PASS`/`WARN`, `1` for `FAIL`).

### `predgate doctor`
Verifies Python version (`>= 3.10`), `git`, `dvc`, `mlflow`, `predgate.yaml` validity, model entrypoint import, Pandera schema import, and dataset CSV paths, printing suggested fixes for any failure.

```bash
predgate doctor --config predgate.yaml
predgate doctor --config predgate.yaml --json
```

### `predgate bench`
Executes the synthetic 7-fault benchmark suite across multiple seeds and writes `docs/benchmark_results.json` and `docs/benchmark.md`.

```bash
predgate bench
```

### `predgate reproduce`
Reproduces a previous run's evaluation using recorded Git commit SHA, DVC data hash, and model parameters.

```bash
predgate reproduce <RUN_ID> --config predgate.yaml
```

### `predgate baseline`
Exports, imports, or verifies the persisted champion baseline (`champion/model.pkl`, `metadata.json`, `metrics.json`) with fail-closed SHA-256 verification:

```bash
predgate baseline export --config predgate.yaml --dir champion
predgate baseline verify --dir champion --dvc champion.dvc --pinned-sha256 <SHA256>
predgate baseline import --config predgate.yaml --dir champion --pinned-sha256 <SHA256>
```

---

## 3. Configuration (`predgate.yaml`)

PredGate is configured via a declarative `predgate.yaml` file validated with Pydantic v2:

```yaml
project: credit-risk-demo
seed: 42
# Optional threshold preset: strict | balanced | quiet
# Omit or use "balanced" for unchanged default thresholds.
# Explicit fields under `checks:` always override preset values.
# preset: balanced

model:
  entrypoint: demo/train.py:build_model
  task: classification

data:
  train: data/train.csv
  test: data/test.csv
  reference: data/reference.csv
  target: default
  schema: demo/contracts/schema.py:schema

champion:
  mlflow_alias: champion
  baseline_dir: champion

mlflow:
  tracking_uri: sqlite:///mlflow.db
  artifact_location: ./mlruns

checks:
  # 1. Data contract validation (Pandera)
  data_contract:
    enabled: true
    on_violation: fail

  # 2. Drift detection on shared features (reference-quantile PSI + KS + numeric point-mass spike)
  drift:
    enabled: true
    method: psi                 # psi or ks
    warn_at: 0.1
    fail_at: 0.25
    on_new_feature: info        # "info" or "warn" for features absent from reference
    point_mass_warn_delta: 0.10 # WARN on exact numeric value-spike jump >= +10 pp
    point_mass_fail_delta: 0.20 # FAIL on exact numeric value-spike jump >= +20 pp

  # 3. Target leakage detection
  leakage:
    enabled: true
    warn_corr: 0.50
    fail_corr: 0.75
    warn_auc: 0.85
    fail_auc: 0.92

  # 4. Overall & subgroup label-shift detection (Holm-Bonferroni two-proportion z-test)
  label_shift:
    enabled: true
    by: [region, gender, age_bucket]
    min_slice_size: 30
    warn_delta: 0.05
    fail_delta: 0.10
    alpha: 0.01

  # 5. Performance slices with paired bootstrap CIs
  # FAIL iff ci_lower > max_drop; WARN iff diff > max_drop (or inconclusive slice when on_inconclusive=warn)
  slices:
    enabled: true
    metric: f1
    by: [region, age_bucket, gender]
    min_slice_size: 100
    max_drop: 0.03
    max_ci_width: 0.10          # mark slices with wider 95% CI as "inconclusive"
    on_inconclusive: warn       # "warn" (default) elevates passing check to WARN; "info" keeps PASS while surfacing banner
    confidence: 0.95
    n_bootstrap: 1000

  # 6. Behavioral & metamorphic tests
  behavioral:
    enabled: true
    invariance:
      - feature: applicant_id
        perturb: random_id
        max_flip_rate: 0.0
    directional:
      - feature: income
        direction: increase
        expect: prob_not_increase

  # 7. Resource and latency budgets (interleaved batched median ratio + paired bootstrap CI)
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

  # 8. Algorithmic fairness (paired bootstrap gap CI default + per-group selection-rate shift WARN)
  fairness:
    enabled: true
    protected: [gender]
    metric: demographic_parity_diff
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

verdict:
  fail_if_any_fail: true
  warn_requires_approval: true
```

### Inconclusive Slices (`slices.on_inconclusive`)

When a slice's 95% paired bootstrap confidence interval is wider than `slices.max_ci_width` (default `0.10`), a real drop may go undetected due to small sample size (see [`docs/power.md`](power.md)). PredGate always surfaces inconclusive slices prominently in:
- The GitHub PR comment (`### ⚠️ Inconclusive Slices (Wide Confidence Intervals)` callout + slice table)
- The standalone HTML report (amber warning callout banner + table)
- The React dashboard (top-of-page **INCONCLUSIVE SLICES** banner card on Run Detail, highlighted slice rows with an `inconclusive` badge, and `{N} inconclusive` count badge on the Runs list)

Configure `slices.on_inconclusive`:
- `warn` (**default**): elevates an otherwise-passing `slices` check to `WARN` so small-sample uncertainty cannot silently pass.
- `info`: keeps an otherwise-passing `slices` check at `PASS` while still displaying the inconclusive slice callout across all reports.

### Model Family Change Note

Whenever the candidate's underlying algorithm differs from the champion's (e.g., `GradientBoostingClassifier` → `RandomForestClassifier`, unwrapping scikit-learn `Pipeline`, `GridSearchCV`, `CalibratedClassifierCV`, and custom wrappers without unwrapping ensemble base learners), PredGate automatically records `model_family_change` in `verdict.json` and displays a **Model family changed** note in the CLI summary, GitHub PR comment, HTML report, and Dashboard Run Detail view. This gives reviewers immediate context when an intentional architecture swap changes single-row latency or subgroup selection rates.

---

## 4. Config Presets (`strict`, `balanced`, `quiet`)

Set `preset: strict | balanced | quiet` in `predgate.yaml` or pass `--preset <name>` to `predgate run`. Any explicit threshold set under `checks:` in `predgate.yaml` overrides the preset value.

Default thresholds are **unchanged** and match `balanced` exactly.

### Preset Thresholds

| Setting | `strict` | `balanced` (default) | `quiet` |
|---|---|---|---|
| `drift.warn_at` / `fail_at` | `0.08` / `0.20` | `0.10` / `0.25` | `0.15` / `0.30` |
| `drift.on_new_feature` | `warn` | `info` | `info` |
| `drift.point_mass_warn_delta` / `fail_delta` | `0.08` / `0.15` | `0.10` / `0.20` | `0.15` / `0.25` |
| `leakage.warn_corr` / `fail_corr` | `0.40` / `0.65` | `0.50` / `0.75` | `0.55` / `0.80` |
| `leakage.warn_auc` / `fail_auc` | `0.80` / `0.88` | `0.85` / `0.92` | `0.88` / `0.94` |
| `label_shift.warn_delta` / `fail_delta` | `0.03` / `0.07` | `0.05` / `0.10` | `0.07` / `0.12` |
| `slices.max_drop` / `max_ci_width` | `0.02` / `0.08` | `0.03` / `0.10` | `0.05` / `0.15` |
| `slices.on_inconclusive` | `warn` | `warn` | `info` |
| `budgets.max_latency_ratio` | `1.25` | `1.50` | `2.00` |
| `fairness.max_gap_increase` / `fail_gap_increase` | `0.015` / `0.03` | `0.02` / `0.05` | `0.04` / `0.08` |
| `fairness.selection_rate_shift_warn` | `0.015` | `0.02` | `0.05` |

### Measured Benign-Change WARN / FAIL Rates (Fresh Seeds `11000..11029`)

Measured independently via `python scripts/bench_presets.py --n-seeds 30 --start-seed 11000` across **all 5 datasets × 9 benign changes × 30 fresh seeds (`1,350` trials per preset)** and saved in [`docs/preset_benchmark_results.json`](preset_benchmark_results.json) (95% Wilson CIs):

| Preset | Quality FAIL (Hard Block) | Quality WARN-Only (Alert Fatigue) | Latency FAIL (`budgets`) | Combined FAIL | Combined WARN-Only | Non-Swap Quality FAIL (N=1,050) |
|---|---|---|---|---|---|---|
| **`strict`** | 83/1350 (**6.1%**, [5.0%, 7.6%]) | 725/1350 (**53.7%**, [51.0%, 56.3%]) | 161/1350 (**11.9%**, [10.3%, 13.8%]) | 205/1350 (**15.2%**, [13.4%, 17.2%]) | 716/1350 (**53.0%**, [50.4%, 55.7%]) | 14/1050 (**1.3%**, [0.8%, 2.2%]) |
| **`balanced`** *(default)* | 49/1350 (**3.6%**, [2.8%, 4.8%]) | 247/1350 (**18.3%**, [16.3%, 20.4%]) | 135/1350 (**10.0%**, [8.5%, 11.7%]) | 163/1350 (**12.1%**, [10.4%, 13.9%]) | 180/1350 (**13.3%**, [11.6%, 15.3%]) | 5/1050 (**0.5%**, [0.2%, 1.1%]) |
| **`quiet`** | 25/1350 (**1.9%**, [1.3%, 2.7%]) | 99/1350 (**7.3%**, [6.1%, 8.8%]) | 59/1350 (**4.4%**, [3.4%, 5.6%]) | 81/1350 (**6.0%**, [4.9%, 7.4%]) | 110/1350 (**8.1%**, [6.8%, 9.7%]) | 1/1050 (**0.1%**, [0.0%, 0.5%]) |

**Key takeaways when picking a preset**:
- **`balanced` (recommended default)**: `0.5%` quality hard-block rate on routine non-swap changes (`hyperparam_change`, `more_data_20pct`, `irrelevant_feature`, `drop_irrelevant_feature`, `preprocessing_change`, `change_imputation`, `rescale_inputs`), `0.0%` benign drift/leakage/label-shift FPs, and `18.3%` quality WARN-only (mostly algorithm-swap selection-rate shifts and small-slice inconclusive warnings).
- **`strict`**: best for regulated / high-stakes models where reviewers want early warnings on new features (`drift.on_new_feature: warn`, `11.1%` WARN), moderately high single-feature associations (`leakage.warn_auc: 0.80`, `34.7%` WARN on strong tabular signals), and tighter `2%` slice drops (`slices.max_drop: 0.02`). Expect ~53% WARN rate on routine PRs.
- **`quiet`**: best for high-velocity teams or small evaluation sets where alert fatigue blocks adoption—cuts combined WARN-only from `13.3%` → `8.1%` and non-swap Quality FAIL to `0.1%` (`1/1050`) by setting `slices.on_inconclusive: info` and widening latency/fairness margins, at the cost of missing smaller 3–5 pt regressions.

---

## 5. GitHub Actions Integration

Use PredGate's composite action in `.github/workflows/predgate.yml`:

```yaml
name: PredGate Evaluation

on:
  pull_request:
    paths:
      - 'src/**'
      - 'demo/**'
      - 'champion/**'
      - 'champion.dvc'
      - 'predgate.yaml'
      - 'data/**'

jobs:
  predgate:
    name: Run PredGate
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pull-requests: write
      statuses: write
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Run PredGate Quality Gate
        uses: ./
        with:
          config: 'predgate.yaml'
          baseline-dir: 'champion'
          setup-data: 'python demo/train.py'
```

**Champion Integrity, Branch Protection & Fork PRs**:
- **Fail-Closed Verification & Pinned Tooling**: Both `.github/workflows/predgate.yml` and `action.yml` fetch `origin/${BASE_REF}` (exiting non-zero if fetch fails), install PredGate from the trusted committed base branch (`git archive`) or an explicit pinned `predgate-spec` (never `pip install -e .` on the PR head), and verify `champion/model.pkl` against the base-branch pinned SHA-256 before running any PR code.
- **No Custom Approval Bypass**: Updates to `/champion/` and `/champion.dvc` are governed by GitHub Branch Protection + `.github/CODEOWNERS` (`@<username>` or `@<org>/<team>` handles), not a spoofable workflow bypass flag.
- Require the **`Run PredGate`** GitHub Actions check run in branch protection (not solely the `predgate` commit status) if your repository accepts pull requests from forks.
- On fork PRs triggered by `on: pull_request`, GitHub automatically downgrades `GITHUB_TOKEN` to read-only (`contents: read`): `Run PredGate` still verifies base-branch champion integrity, runs all 8 checks, fails the check run on regressions, and uploads `predgate_output/report.html`, while `Post PR comment` and `Set commit status` gracefully emit warnings on HTTP 403 (`Resource not accessible by integration`) rather than crashing.

---

## 6. Docker Compose Services

PredGate includes a Docker Compose stack for local API, MLflow, and Dashboard exploration:

```bash
docker compose up -d
```

Services:
- **API**: http://localhost:8000 (FastAPI Swagger docs at `/docs`)
- **Dashboard**: http://localhost:3000 (React + Vite web interface)
- **MLflow Tracking Server**: http://localhost:5000
- **Seed Script**: Automatically seeds demo runs on startup.

---

## 7. Authentication, RBAC, Audit Log & HTTPS Deployment

PredGate enforces bearer-token authentication and role-based access control (RBAC) across mutating API endpoints:

- **No Default Admin Token**: PredGate never ships with a default or hardcoded admin token. Provision tokens explicitly via the CLI:
  ```bash
  predgate users add alice --role engineer
  predgate users add bob --role reviewer
  predgate users add carol --role admin
  predgate users list
  ```
  Alternatively, set `PREDGATE_ADMIN_TOKEN` in the server environment at startup to bootstrap the initial `admin` user. Plaintext tokens are displayed once at creation; only their SHA-256 hex digests are stored in the `users` table.
- **Role Permissions**:
  - `POST /runs`: `engineer`, `reviewer`, or `admin`
  - `POST /runs/{id}/approve` (or `/approval`): `reviewer` or `admin` (reviewer identity is bound from the authenticated token, not the request body). Overriding a `FAIL` run requires `admin` role and a justification comment of at least 20 characters.
  - `POST /runs/{id}/promote`: `admin`
  - `PUT /settings`: `admin`
  - `GET /audit`, `POST /users`, `GET /users`: `admin`
  - Read endpoints (`GET /runs`, `GET /models`, `GET /compare`, `GET /settings`, `GET /bench`, `GET /health`) remain unauthenticated for read-only dashboard access.
- **Append-Only Audit Log (`GET /audit`)**: Every allowed and denied (`401`/`403`/`400`) action on protected endpoints is recorded in the immutable `audit_log` table with `timestamp`, `actor`, `role`, `action`, `target`, `outcome`, and `detail`.
- **Restricted CORS & HTTPS Requirement**:
  - CORS origins are restricted to `http://localhost:3000,http://localhost:5173` by default (override via `PREDGATE_CORS_ORIGINS`), with methods restricted to `GET, POST, PUT, OPTIONS` and headers restricted to `Authorization, Content-Type, Accept`.
  - **HTTPS in Production**: Because bearer tokens are sent via `Authorization: Bearer <token>`, any production or non-localhost deployment **must** run behind TLS/HTTPS (e.g. Nginx/Caddy/ALB terminating TLS and setting `PREDGATE_CORS_ORIGINS=https://predgate.example.com`) so tokens are encrypted in transit.
