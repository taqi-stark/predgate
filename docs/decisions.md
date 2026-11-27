# PredGate — Architecture Decision Records (ADRs)

This document captures the key architectural and design decisions made during the development of PredGate, along with the context, options considered, and tradeoffs.

---

## ADR-001: CI/CD Quality Gate Model vs Offline Batch Monitoring

### Context
Machine learning monitoring tools (e.g., Evidently, Whylabs, Arize) traditionally focus on post-deployment batch monitoring in production. By the time drift or slice regressions are detected, damaged predictions have already affected end users.

### Decision
PredGate is built as a **pre-deployment gate** integrated directly into the CI/CD pull request lifecycle. Models cannot be merged to master or deployed without passing all 8 quality gate checks: data contract, feature drift, target leakage, subgroup label shift, slice regression, behavioral tests, performance budgets, and algorithmic fairness.

### Consequences
- **Positive**: Prevents silent regressions from ever reaching production; binds model evaluation to Git SHAs and PR commits.
- **Tradeoff**: Evaluation must execute quickly (under ~3 minutes) to maintain an acceptable developer feedback loop in CI.

---

## ADR-002: Champion-Challenger Paired Bootstrap vs Absolute Thresholds

### Context
Static accuracy thresholds (e.g., `accuracy > 0.85`) fail to catch subtle regressions. Furthermore, natural sampling variance can trigger false alarms when testing minority slices.

### Decision
PredGate implements **paired bootstrap confidence intervals (CI)** comparing candidate models directly against the current champion model. A slice regression is only flagged if the drop is statistically significant at $\alpha = 0.05$ over $B=50$ to $B=1000$ bootstrap iterations.

### Consequences
- **Positive**: Virtually eliminates false-positive alerts caused by noise on small demographic slices; provides statistically grounded pass/fail verdicts.
- **Tradeoff**: Bootstrap resampling introduces CPU overhead during slice evaluation.

---

## ADR-003: Consolidated MLflow Tracking URI

### Context
Early iterations of PredGate stored MLflow artifacts across disparate local directories (`./mlruns`, `/tmp/mlruns`, subfolders), leading to fragmented experiment histories and broken UI dashboards.

### Decision
Consolidate MLflow tracking to a single configurable tracking URI specified in `predgate.yaml` (defaulting to `file:./mlruns`), shared across the CLI, FastAPI backend, and Docker Compose services.

### Consequences
- **Positive**: Single source of truth for all runs, metrics, artifacts (`verdict.json`, `report.html`), and model registries.
- **Tradeoff**: In multi-developer or remote setups, a shared remote SQLite or PostgreSQL backend is required instead of local file storage.

---

## ADR-004: Data Lineage with DVC & Git Integration

### Context
Evaluating a model requires knowing the exact training and evaluation dataset used. Tracking large CSVs or Parquet files directly in Git bloats the repository.

### Decision
Support **DVC (Data Version Control)** optionally via `.dvc` pointer files and SHA-256 hashes while keeping included demo examples self-contained with deterministic dataset generators. The CLI captures `git_sha`, file SHA-256 `data_hashes`, and optional `dvc_hash` at run-time, allowing `predgate reproduce <RUN_ID>` to check out DVC-tracked data when present and verify reproduction.

### Consequences
- **Positive**: Full audit trail and deterministic reproducibility for compliance and debugging.
- **Tradeoff**: Requires DVC CLI installation in developer and CI environments.

---

## ADR-005: Pandera for Data Contracts

### Context
Input schema changes and type coercion errors frequently break models in silent ways. We needed an expressive schema definition tool that integrates seamlessly with pandas.

### Decision
Adopt **Pandera** for schema validation. Data contracts define strict column dtypes, value ranges, category sets, and nullability constraints.

### Consequences
- **Positive**: Clear schema contracts that can be versioned as code and tested in CI; descriptive error messages on schema violations.
- **Tradeoff**: Slight dataframe inspection overhead on very large validation sets.

---

## ADR-006: Reusable Composite GitHub Action (`action.yml`)

### Context
Users should be able to integrate PredGate into their GitHub workflows without manually copying complex bash scripts or python boilerplate.

### Decision
Provide a reusable composite action in the repository root (`action.yml`) that installs dependencies, executes `predgate run`, archives reports, and automatically comments on the pull request with a rich summary table.

### Consequences
- **Positive**: 1-step adoption for GitHub Actions users; standardized comment formatting and error handling across teams.
- **Tradeoff**: Requires `contents: read` and `pull-requests: write` permissions in GitHub Actions workflow.

---

## ADR-007: Decoupled Dashboard via React + Vite & Nginx Multi-Stage Build

### Context
The dashboard needs to display real-time runs, model comparisons, and interactive benchmark visualizations without coupling frontend build toolchains to the Python package.

### Decision
Build the dashboard with React, Vite, and Tailwind/CSS as an independent SPA under `dashboard/`. Containerize it using a multi-stage Dockerfile that builds static assets and serves them via Nginx, reverse-proxying API requests to FastAPI.

### Consequences
- **Positive**: Clean separation of concerns; fast frontend build times; minimal container footprint.
- **Tradeoff**: Two distinct toolchains (Python/Poetry or pip, and Node.js/npm) in development.
