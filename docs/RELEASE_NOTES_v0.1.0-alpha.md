# PredGate `v0.1.0-alpha` Release Notes (Pre-Tag Release Prep)

> **Release status**: Prepared for **`v0.1.0-alpha`** (`predgate==0.1.0a0` in PEP 440) — **NOT yet git-tagged** (`git tag` intentionally withheld until final review).
> **Evaluated Release Code Commit**: `f40f0e7` (`git diff f40f0e7..HEAD -- src/predgate/checks/ src/predgate/stats.py src/predgate/config.py src/predgate/verdict.py` is empty, enforced in CI by `scripts/check_benchmark_fresh.sh`; fresh seeds `13000..13029`, frozen default thresholds, zero post-hoc tuning).
> **Preset Benchmark Evaluation**: Fresh seeds `11000..11029` (1,350 trials per preset across `strict`, `balanced`, and `quiet`; default thresholds unchanged).
> **TestPyPI Build Status**: `dist/predgate-0.1.0a0.tar.gz` and `dist/predgate-0.1.0a0-py3-none-any.whl` built cleanly with Hatchling, verified via `twine check --strict dist/*`, and smoke-tested in an isolated temporary virtualenv.

---

## What's New in `v0.1.0-alpha`

### Pre-Release Polish & Usability
- **Configurable Inconclusive Slice Handling (`slices.on_inconclusive: warn | info`, default `warn`)**: Wide paired-bootstrap confidence intervals (`ci_width > max_ci_width`, default `0.10`) now elevate an otherwise-passing `slices` check to `WARN` by default (`info` preserves `PASS`), and are surfaced prominently in:
  - GitHub PR comments (`### ⚠️ Inconclusive Slices (Wide Confidence Intervals)` alert + Markdown table)
  - Standalone HTML reports (amber `.callout-warn` banner + table)
  - React Dashboard (top-of-page **INCONCLUSIVE SLICES** banner on Run Detail, highlighted slice rows with `inconclusive` badge, and `{N} inconclusive` count badges on Runs list)
  - REST API (`GET /runs/{id}/slices` and `GET /runs/{id}`)
- **Config Presets (`strict`, `balanced`, `quiet`)**: Supported via `preset:` in `predgate.yaml` or `--preset` in `predgate run`, with explicit YAML fields always overriding preset defaults. Default thresholds are **100% unchanged** and equal `balanced`. Documented in [`docs/usage.md`](usage.md#4-config-presets-strict-balanced-quiet) and [`docs/benchmark.md`](benchmark.md) with benign-change WARN/FAIL rates measured on 30 fresh seeds (`11000..11029`, `1,350` trials/preset).
- **"Model family changed" Report Note**: `detect_model_family_change` unwraps scikit-learn `Pipeline`, `GridSearchCV`, `CalibratedClassifierCV`, `TransformedTargetRegressor`, and custom wrappers (without misidentifying ensembles like `RandomForestClassifier`, `AdaBoostClassifier`, or `BaggingClassifier` as their base learner) and emits a prominent **Model family changed** banner in CLI output, `verdict.json`, PR comments, HTML reports, and the React dashboard whenever the candidate algorithm differs from the champion's.
- **Statistical Power & Minimum Sample Size Guide ([`docs/power.md`](power.md))**: Exact formulas and tables for minimum rows per group needed to detect 3-point and 5-point drops at 80% power (`α = 0.05`) for both paired per-slice tests and two-group fairness proportion differences, mapped directly onto the five benchmark datasets.
- **Full Benchmark Transparency & Reproducibility ([`docs/benchmark.md`](benchmark.md) & [`README.md`](../README.md))**: Explicitly discloses that **all five benchmark datasets were used during development**, documents the exact `evidently==0.7.23` (`pandas==2.3.3`) comparator configuration with verbatim API calls, and adds a complete **How to Reproduce** guide.
- **Streamlined README ([`README.md`](../README.md))**: Concise 3-command quickstart, 60-second demo section with GIF placeholder (`docs/assets/demo.gif`), honest "What it catches / What it doesn't" table, preset summary table, and honest benchmark summary.

### Core Platform Capabilities
- **8-Check Quality Gate Suite**: Data contracts (Pandera), shared-feature drift (reference-quantile PSI, KS, numeric point-mass spike check, and clean `new feature, no reference` reporting), target leakage (`|r|` & univariate ROC-AUC), subgroup label shift (Holm-Bonferroni two-proportion z-test), slice regression (paired bootstrap lower-bound CI gate + `inconclusive` reporting), behavioral invariance/directional tests, interleaved batched median latency ratio + size/memory budgets, and relative algorithmic fairness (DP, EO, EqOdds, calibration + per-group selection-rate shift WARN + 80%-power MDE notes).
- **Fail-Closed Champion Integrity**: `.github/workflows/predgate.yml` and `action.yml` fetch `origin/${BASE_REF}` (failing closed on error), install PredGate from the trusted committed base branch (`git archive`) or an explicit pinned `predgate-spec`, and verify `champion/model.pkl` against the base-branch pinned SHA-256 before executing any PR training code. Champion updates are governed by GitHub Branch Protection + `.github/CODEOWNERS`.
- **RBAC & Audit Logging**: No hardcoded default admin token; SHA-256 hashed bearer tokens, role enforcement (`engineer`, `reviewer`, `admin`), restricted CORS, and append-only `/audit` log.

---

## Empirical Summary

> **Dataset-Use Disclosure**: **All five benchmark datasets** (UCI Adult, German Credit, Synthetic Credit, Bank Marketing, and COMPAS) **were used during development**. None should be treated as an unseen held-out benchmark.

### 1. Frozen-Default Evaluation (Release Code Commit `f40f0e7`, Fresh Seeds `13000..13029`)

| Evaluation Suite | Key Result (FAIL-only / Flagged WARN+FAIL, 95% Wilson CI) | Comparator Highlights |
|---|---|---|
| **Synthetic 7-Fault (`13000..13029`)** | **30/30 (100.0%, [88.6%, 100.0%])** FAIL on all 6 faults; **0/30 (0.0% FP, [0.0%, 11.4%])** on `noop` | Evidently `0.7.23` defaults had **11/30 (36.7% FP)** on `noop`; Overall Acc missed `degraded_slice` (`0/30`) & `latency_regression` (`0/30`). |
| **Benign-Change Control — 9 Changes × 5 Datasets (`13000..13029`, 1,350 trials)** | **Quality-Only FAIL**: **56/1350 (4.1%, [3.2%, 5.3%])**; **Quality WARN-Only**: **234/1350 (17.3%, [15.4%, 19.4%])**; **Latency FAIL (`budgets`)**: **139/1350 (10.3%, [8.8%, 12.0%])** (all on `algo_swap_rf`); **Combined FAIL**: **168/1350 (12.4%, [10.8%, 14.3%])** | `drift`, `leakage`, and `label_shift` had **0/1350 (0.0%)** benign FAILs/WARNs; `slices` WARN dropped `132 -> 127` with straddling-only inconclusive intervals (`ci_lower <= max_drop < ci_upper`). Evidently `0.7.23` defaults flagged **382/1350 (28.3%)**. |
| **All-5-Dataset Faults (`13000..13029`, 30 seeds/fault)** | **`silent_null_imputation`**, **`fairness_proxy_feature`**, **`fairness_regression`**: **30/30 (100%) FAIL** on all 5 datasets; **`cohort_feature_corruption`**: **30/30** Adult & COMPAS, **28/30** Bank (**30/30** Flagged), **27/30** Synth (**29/30** Flagged), **8/30 FAIL / 13/30 Flagged** German Credit; **`noop`**: **0/30 (0.0% FP)** FAIL and Flagged on all 5 datasets | Fixed DP `>=0.10` had **30/30 (100% FP)** on Adult & COMPAS `noop` and **4/30 (13.3% FP)** on German `noop`. |

### 2. Config Presets Benign-Change Comparison (Fresh Seeds `11000..11029`, 1,350 trials/preset)

| Preset | Quality FAIL (95% CI) | Quality WARN-Only (95% CI) | Latency FAIL (`budgets`) | Combined FAIL (95% CI) | Combined WARN-Only (95% CI) | Non-Swap Quality FAIL (N=1,050) |
|---|---|---|---|---|---|---|
| **`strict`** | 83/1350 (6.1%, [5.0%, 7.6%]) | 725/1350 (53.7%, [51.0%, 56.3%]) | 161/1350 (11.9%, [10.3%, 13.8%]) | 205/1350 (15.2%, [13.4%, 17.2%]) | 716/1350 (53.0%, [50.4%, 55.7%]) | 14/1050 (1.3%, [0.8%, 2.2%]) |
| **`balanced`** *(default)* | 49/1350 (3.6%, [2.8%, 4.8%]) | 247/1350 (18.3%, [16.3%, 20.4%]) | 135/1350 (10.0%, [8.5%, 11.7%]) | 163/1350 (12.1%, [10.4%, 13.9%]) | 180/1350 (13.3%, [11.6%, 15.3%]) | 5/1050 (0.5%, [0.2%, 1.1%]) |
| **`quiet`** | 25/1350 (1.9%, [1.3%, 2.7%]) | 99/1350 (7.3%, [6.1%, 8.8%]) | 59/1350 (4.4%, [3.4%, 5.6%]) | 81/1350 (6.0%, [4.9%, 7.4%]) | 110/1350 (8.1%, [6.8%, 9.7%]) | 1/1050 (0.1%, [0.0%, 0.5%]) |

---

## Honest Limitations Table

| Area | Limitation / Observed Failure Mode | Empirical Evidence | Mitigation / Follow-Up |
|---|---|---|---|
| **Small-Sample Slice Power (`ci_lower > max_drop`)** | On tiny test cohorts (e.g., German Credit `N_test=300`, `min_n=93`), wide paired-bootstrap CIs prevent localized degradations from clearing `ci_lower > 0.03`. | German Credit `cohort_feature_corruption` (`f40f0e7`, seeds `13000..13029`): **8/30 (26.7%) FAIL / 13/30 (43.3%) Flagged**. | PredGate defaults to `slices.on_inconclusive: warn`, renders prominent inconclusive slice tables across PR/HTML/dashboard, and documents exact sample sizes in [`docs/power.md`](power.md). |
| **Benign Model-Family Swaps Trigger Alerts** | Swapping GBDT → Random Forest (`n_estimators=60`) increases single-row sklearn latency >1.5× and shifts subgroup TPR/FPR and selection rates; GBDT → Logistic Regression shifts selection rates. | `balanced` (`11000..11029`, 300 algo-swap trials): Quality FAIL **44/300 (14.7%)**, Quality WARN-only **149/300 (49.7%)**, Latency FAIL **135/300 (45.0%)** (vs. **0.5%** non-swap Quality FAIL). | Reports now surface a prominent **Model family changed** banner (`Champion → Candidate`) so reviewers immediately recognize expected architecture-swap deltas; use `preset: quiet` or update baselines via CODEOWNERS PRs when migrating model families. |
| **Rare-Event Threshold Shifts & Fairness MDE** | When baseline positive prediction rate is near zero (~0.2%) or protected groups are small (`min_n = 93–594`), small shifts fall below 80%-power MDE (`MDE_DP = 0.066–0.160`). | Synthetic `τ=0.75`: **0/30**; Synthetic `τ=0.25`: **2/30 FAIL / 7/30 Flagged**; MDE low-power note emitted `30/30` on Adult, German, Bank, COMPAS. | PredGate emits explicit `minimum detectable gap / insufficient power` notes rather than silently asserting fairness. |
| **Selection-Rate Shift Is WARN-Only** | Group-specific threshold shifts that *narrow* DP (`τ=0.25`) trigger primarily `WARN` rather than exit code `1`. | Full-gate `τ=0.25` (`f40f0e7`): Adult **3/30 FAIL / 30/30 Flagged**, Bank **13/30 FAIL / 29/30 Flagged**, COMPAS **30/30 FAIL**. | Intentional alert-fatigue control; teams wanting hard blocks on any selection-rate shift can fail CI on `WARN`. |
| **Pickle Champion Serialization** | Champion baselines use SHA-256 verified `pickle` (failing closed on corruption or Python/sklearn mismatch); `skops` / ONNX not yet supported. | Verified fail-closed in `tests/unit/test_security.py`. | Use `on: pull_request` (never `pull_request_target`) with branch protection + CODEOWNERS on `/champion/`. |
| **Tabular Binary Classification Focus** | Core checks target `sklearn`-compatible tabular binary classifiers; multi-class, regression, NLP, and vision are only partially supported. | Verified across 5 tabular binary datasets. | Expand regression and multi-class check coverage in `v0.2.x`. |
| **Demo GIF Pending** | README includes a placeholder (`docs/assets/demo.gif`) for the recorded 60-second PR demo GIF. | Intentionally left as placeholder for maintainer recording. | Drop recorded GIF into `docs/assets/demo.gif` before tagging `v0.1.0-alpha`. |

---

## Publishing to TestPyPI (When Ready)

Tagging is intentionally deferred (`git tag` has **not** been created). When ready to publish `0.1.0a0` to TestPyPI:

```bash
# 1. Verify clean build artifacts
rm -rf dist/
.venv/bin/python -m build
.venv/bin/twine check --strict dist/*

# 2. Upload to TestPyPI (requires TestPyPI API token)
.venv/bin/twine upload --repository testpypi dist/*

# 3. Verify install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ predgate==0.1.0a0

# 4. Tag when ready
# git tag -a v0.1.0-alpha -m "PredGate v0.1.0-alpha"
# git push origin v0.1.0-alpha
```
