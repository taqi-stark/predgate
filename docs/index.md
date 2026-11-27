# PredGate Documentation

**PredGate** is an open-source CI/CD quality gate for tabular machine learning models. On every pull request, it compares a candidate model against your SHA-256-verified champion across eight automated checks and produces a `PASS`, `WARN`, or `FAIL` verdict before broken models merge.

## Start Here

- **[Getting Started](getting-started.md)** — Install PredGate, initialize a project with `predgate init`, run the gate locally, and try the included examples.
- **[Quality Checks](concepts/checks.md)** — How each of the 8 checks works, how verdicts aggregate, inconclusive slices, and model-family change notes.

## Guides

- **[GitHub Actions](guides/github.md)** — Workflow setup, reusable composite action, branch protection, CODEOWNERS, fork-PR behavior, and optional DVC tracking.
- **[Docker Compose](guides/docker.md)** — Run the API, MLflow tracking server, seeded demo runs, and React dashboard locally.
- **[Dashboard](guides/dashboard.md)** — Walkthrough of the Runs list, Run Detail, Slice Explorer, Compare, and Models views, plus token authentication.
- **[Config Presets](guides/presets.md)** — Compare `strict`, `balanced`, and `quiet` thresholds and measured benign-change WARN/FAIL rates.

## Reference & Evaluation

- **Reference:** [Configuration (`predgate.yaml`)](reference/config.md) · [CLI Commands](reference/cli.md) · [REST API](reference/api.md)
- **Evaluation:** [Benchmark Results](benchmark.md) · [Statistical Power](power.md) · [Runtime & Memory](performance.md)
- **Operations & Design:** [Security & HTTPS](security.md) · [Architecture](architecture.md) · [Decisions (ADRs)](decisions.md) · [Releasing](RELEASING.md) · [Release Notes (`v0.1.0-alpha`)](RELEASE_NOTES_v0.1.0-alpha.md)
