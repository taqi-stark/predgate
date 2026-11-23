# PredGate — Architecture

## Overview

PredGate is a **CI/CD quality gate for ML models**. It intercepts every PR that changes model code, data, or configuration, executes a configurable set of checks, and blocks the merge if the model doesn't meet production standards.

```
┌──────────────────────────────────────────────────────────────────┐
│                       Developer Workflow                         │
│                                                                  │
│  git push  →  GitHub Actions  →  predgate run  →  PR comment   │
│                                        │                         │
│                              verdict.json / report.html         │
└──────────────────────────────────────────────────────────────────┘
```

---

## Repository Layout

```
PredGate/
├── src/predgate/          # Core Python package
│   ├── cli.py              # Typer CLI (predgate run / bench / reproduce …)
│   ├── config.py           # Pydantic models for predgate.yaml
│   ├── loader.py           # Dynamic import of model entrypoints
│   ├── lineage.py          # Git SHA + DVC hash capture
│   ├── registry.py         # MLflow champion/challenger registry
│   ├── verdict.py          # Aggregates CheckResults → PASS/WARN/FAIL
│   ├── report.py           # HTML + Markdown PR comment generation
│   ├── stats.py            # Bootstrap CI, PSI, paired-bootstrap helpers
│   ├── benchmark.py        # Fault-injection benchmark suite
│   ├── api/                # FastAPI REST layer
│   │   ├── main.py         # Endpoints: runs, models, settings, health
│   │   └── models.py       # Pydantic request/response schemas
│   ├── checks/             # Individual quality-gate checks (8 checks)
│   │   ├── data_contract.py   # 1. Pandera schema validation
│   │   ├── drift.py           # 2. PSI / KS feature drift detection
│   │   ├── leakage.py         # 3. Target leakage (|r| & univariate ROC-AUC)
│   │   ├── label_shift.py     # 4. Overall & subgroup label-shift z-test
│   │   ├── slices.py          # 5. Per-slice metric regression (bootstrap CI)
│   │   ├── behavioral.py      # 6. Invariance & directional tests
│   │   ├── budgets.py         # 7. Latency, model size, peak memory budgets
│   │   └── fairness.py        # 8. Parity, equalized odds & calibration gap CI
│   └── integrations/
│       └── github/
│           └── pr_bot.py      # GitHub PR comment posting
├── dashboard/              # Vite + React TypeScript dashboard
│   ├── src/pages/          # Runs, Models, Compare, Benchmark, Settings
│   └── Dockerfile          # Multi-stage nginx build
├── demo/                   # End-to-end credit-risk demo
│   ├── train.py            # Generates synthetic data + trains model
│   └── contracts/schema.py # Pandera schema for demo dataset
├── data/                   # CSV datasets (tracked by DVC)
├── data.dvc                # DVC lock file (SHA-256 hash of data/)
├── tests/
│   ├── unit/               # Pure-Python, no I/O
│   └── integration/        # Smoke tests that call the API
├── scripts/seed.py         # Docker Compose seed script
├── .github/workflows/
│   ├── ci.yml              # Ruff, mypy, pytest, dashboard build
│   └── predgate.yml       # PredGate quality-gate on PRs
├── action.yml              # Reusable GitHub composite action
├── docker-compose.yml      # api + mlflow + dashboard + seed
├── Dockerfile              # API container
└── predgate.yaml          # Project-level configuration
```

---

## Core Data Flow

```
predgate run
     │
     ▼
config.py ──► loader.py
                  │  load model entrypoint + datasets
                  ▼
            ┌─────────────────────────────────────┐
            │            Check pipeline           │
            │  ┌──────────────┐                  │
            │  │ data_contract│  Pandera schema   │
            │  ├──────────────┤                  │
            │  │ drift        │  PSI / KS / Wass  │
            │  ├──────────────┤                  │
            │  │ slices       │  Bootstrap CI     │
            │  ├──────────────┤                  │
            │  │ behavioral   │  Invariance/Dir.  │
            │  ├──────────────┤                  │
            │  │ budgets      │  Lat/Mem/Size     │
            │  ├──────────────┤                  │
            │  │ fairness     │  Dem. parity      │
            │  └──────────────┘                  │
            └──────────┬──────────────────────────┘
                       │  list[CheckResult]
                       ▼
                  verdict.py  ──► PASS / WARN / FAIL
                       │
                       ├──► report.py ──► report.html
                       │               └► pr_comment.md
                       ├──► registry.py ──► MLflow alias update
                       └──► api/main.py ──► POST /runs
```

---

## Checks (8 Quality Gate Checks)

| Check | Trigger | Default Action |
|-------|---------|----------------|
| **1. Data Contract** | Schema column types, ranges, nullable constraints (Pandera) | `fail` on violation |
| **2. Drift** | Feature PSI > threshold vs reference dataset | `warn` at 0.10, `fail` at 0.25 |
| **3. Target Leakage** | Univariate feature-target `\|r\|` or rank ROC-AUC above threshold | `warn` at `\|r\| >= 0.50` / `AUC >= 0.85`, `fail` at `\|r\| >= 0.75` / `AUC >= 0.92` |
| **4. Label Shift** | Overall or subgroup target rate shift (Holm-Bonferroni two-proportion z-test) | `warn` at `Δ >= 0.05`, `fail` at `Δ >= 0.10` (`α = 0.01`) |
| **5. Slice Regression** | Per-group metric drop with vectorized paired bootstrap 95% CI | `fail` if drop > `max_drop` |
| **6. Behavioral** | Invariance (protected/ID attr flip) & directional metamorphic tests | `fail` if flip > `max_flip_rate` |
| **7. Budgets** | Interleaved batched median latency ratio with paired bootstrap 95% CI (optional absolute `p95_latency_ms`), serialized size, peak memory | `fail` if ratio CI lower > `max_latency_ratio` (or size/mem/optional p95 exceeded); `warn` if ratio > budget and CI lower > 1.0 |
| **8. Fairness** | Champion-vs-candidate paired bootstrap gap CI (DP, EO, eq. odds, calibration) + two-sided per-group selection-rate shift + 80%-power MDE notes; opt-in absolute DP relative to champion (`enforce_absolute`) | `fail` on significant gap regression (`+0.05` or CI lower `>= 0.02`); `warn` on moderate gap increase or significant per-group selection-rate shift (`\|Δ\| >= 0.02`) |

---

## Verdict Aggregation

`verdict.py` collects all `CheckResult(status, check, details)` objects and applies:

1. Any `FAIL` → overall verdict is **FAIL**  
2. Any `WARN` (no `FAIL`) → overall verdict is **WARN**  
3. All `PASS` / `SKIP` → overall verdict is **PASS**

The `fail_if_any_fail` flag in `predgate.yaml` controls whether the CI step exits non-zero.

---

## REST API

The FastAPI layer (`api/main.py`) exposes:

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Liveness check |
| `POST` | `/runs` | Create a gate run record |
| `GET` | `/runs` | List all runs (with optional filters) |
| `GET` | `/runs/{id}` | Fetch a single run |
| `GET` | `/models` | List registered models |
| `GET` | `/settings` | Read project settings |
| `PUT` | `/settings` | Update project settings |
| `GET` | `/stats/summary` | Aggregate pass/warn/fail counts |

Data is persisted in a SQLite file (`predgate.db`). In Docker Compose this file is stored in a named volume.

---

## MLflow Integration

PredGate uses a **single tracking URI** configured in `predgate.yaml`:

```yaml
mlflow:
  tracking_uri: sqlite:///mlflow.db   # local dev
  artifact_location: ./mlruns
```

When running via Docker Compose the API passes `MLFLOW_TRACKING_URI=http://mlflow:5000` so all services use the same MLflow server. The `registry.py` module promotes a run to the `champion` alias after a PASS verdict.

---

## DVC Data Tracking

`data/` is tracked via DVC (`data.dvc`). To reproduce a run against the exact data used in CI:

```bash
dvc pull          # downloads data/ matching data.dvc SHA
predgate reproduce
```

`predgate reproduce` calls `dvc repro` before running the gate, ensuring data integrity.

---

## Docker Compose Services

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   dashboard  │────▶│     api      │────▶│    mlflow    │
│  :3000 nginx │     │  :8000 uvicorn│     │ :5000 server │
└──────────────┘     └──────────────┘     └──────────────┘
                              ▲
                      ┌───────┴───────┐
                      │     seed      │
                      │ (one-shot run)│
                      └───────────────┘
```

- **api** — FastAPI service; reads/writes `predgate.db` via a named volume.  
- **mlflow** — MLflow tracking server; persists runs in `mlflow-data` volume.  
- **dashboard** — Vite/React app served via nginx; proxies API calls.  
- **seed** — Runs once on startup; generates demo data and posts results to the API.

Start everything:

```bash
docker compose up
```

---

## GitHub Actions

### CI (`ci.yml`)

Runs on every push and PR to `main`:

1. **Ruff** — lint and format check  
2. **Mypy** — static type checking  
3. **Pytest** — unit tests with coverage gate (≥ 55% total)  
4. **Dashboard build** — `npm run build` in `dashboard/`

### Quality Gate (`predgate.yml`)

Triggered on PRs that touch `src/`, `demo/`, `data/`, or `predgate.yaml`:

1. Runs `predgate run`  
2. Uploads `report.html` as an artifact  
3. Posts/updates a PR comment with the verdict table  
4. Sets a commit status (`success` / `failure`)

### Reusable Composite Action (`action.yml`)

Any downstream repository can use PredGate as a reusable action:

```yaml
- uses: your-org/PredGate@main
  with:
    config: predgate.yaml
    output-dir: predgate_output
    fail-on-verdict: 'true'
```

---

## Extending PredGate

To add a new check:

1. Create `src/predgate/checks/mycheck.py` with a `run_mycheck(…) -> CheckResult` function.  
2. Add a `MyCheckConfig` Pydantic model in `config.py`.  
3. Wire it into `cli.py` (inside `run_gate`).  
4. Add a unit test under `tests/unit/test_checks.py`.

The `CheckResult` dataclass:

```python
@dataclass
class CheckResult:
    check: str          # e.g. "drift"
    status: str         # "pass" | "warn" | "fail" | "skip"
    details: dict       # arbitrary payload for the report
    reason: str = ""    # human-readable summary line
```
