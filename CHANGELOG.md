# Changelog

All notable changes to PredGate are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) (with PEP 440 pre-release tags on PyPI).

## [0.1.0-alpha] - Unreleased (`predgate==0.1.0a0`)

### Added
- **8-check ML quality gate suite**: Data contracts (Pandera), shared-feature drift (reference-quantile PSI, KS, numeric point-mass spike check, clean `new feature, no reference` reporting), target leakage (`|r|` & univariate ROC-AUC), subgroup label shift (Holm-Bonferroni two-proportion z-test), slice regression (vectorized paired bootstrap lower-bound CI gate), behavioral invariance/directional tests, interleaved batched median latency ratio & resource budgets, and relative algorithmic fairness (DP, EO, EqOdds, calibration + per-group selection-rate shift WARN + 80%-power MDE notes).
- **Configurable inconclusive slice handling**: `slices.on_inconclusive: warn|info` (default `warn`) with prominent banners and tables in GitHub PR comments, standalone HTML reports, REST API, and the React dashboard.
- **Config presets**: `strict`, `balanced` (identical to unchanged defaults), and `quiet` via `predgate.yaml` (`preset:`) or CLI (`--preset`), documented with benign-change WARN/FAIL rates measured on 30 fresh seeds (`11000..11029`).
- **Model family change detection**: Unwraps pipelines and single-model wrappers (without misidentifying ensembles) and surfaces a prominent **Model family changed** banner across CLI, `verdict.json`, PR comments, HTML reports, and dashboard.
- **Environment diagnostics**: `predgate doctor` verifying Python version, git, DVC, MLflow, config schema, model entrypoint, Pandera schema import, and dataset paths with actionable fixes.
- **Fail-closed champion integrity & RBAC**: Base-branch pinned SHA-256 verification before unpickling, GitHub Branch Protection + `.github/CODEOWNERS`, SHA-256 hashed bearer tokens with role checks (`engineer`, `reviewer`, `admin`), and append-only `/audit` log.
- **Documentation & adoption assets**: `docs/power.md`, `docs/performance.md`, `docs/benchmark.md` (with 5-dataset development disclosure, exact Evidently `0.7.23` config, and reproduction commands), `examples/minimal/`, `examples/github-workflow/`, MkDocs Material config, OIDC Trusted Publishing release workflow, and `scripts/check_benchmark_fresh.sh`.
