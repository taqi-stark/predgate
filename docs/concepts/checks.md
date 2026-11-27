# Quality Checks & Verdict Engine

PredGate evaluates every candidate model across eight modular checks in `src/predgate/checks/` and aggregates their `CheckResult` objects into a single `PASS`, `WARN`, or `FAIL` verdict.

## The 8 Checks

| # | Check | Method | Default Trigger |
|---|---|---|---|
| 1 | **Data Contract** | Validates train and test DataFrames against a user-supplied Pandera `DataFrameSchema` | `FAIL` on any schema violation |
| 2 | **Feature Drift** | Reference-quantile PSI (or KS) on shared columns, numeric point-mass spike detection (catches silent null-to-zero/sentinel corruption), and new-column detection | `WARN` at PSI `>= 0.10` or point-mass delta `>= 0.10`; `FAIL` at PSI `>= 0.25` or point-mass delta `>= 0.20`; new columns `info` (`warn` under `strict`) |
| 3 | **Target Leakage** | Univariate linear correlation (`\|r\|`) and single-feature rank ROC-AUC against the target | `WARN` at `\|r\| >= 0.50` or `AUC >= 0.85`; `FAIL` at `\|r\| >= 0.75` or `AUC >= 0.92` |
| 4 | **Label Shift** | Holm-Bonferroni corrected two-proportion z-test on overall and subgroup target rates (`train` vs `reference`) | `WARN` at significant delta `>= 0.05`; `FAIL` at significant delta `>= 0.10` (`alpha = 0.01`) |
| 5 | **Slice Regression** | Vectorized paired bootstrap 95% CI on `champion - candidate` metric drop per subgroup (`by` columns) | `FAIL` when `ci_lower > max_drop` (`0.03`); `WARN` when point drop `> max_drop` or slice is inconclusive (`on_inconclusive: warn`) |
| 6 | **Behavioral** | Metamorphic invariance tests (e.g., perturbing `applicant_id`) and directional monotonicity tests (e.g., increasing `income`) | `FAIL` when flip rate exceeds `max_flip_rate` or directional compliance falls below threshold |
| 7 | **Budgets** | Interleaved batched single-row median latency ratio (`cand / champ`) with paired bootstrap 95% CI, serialized model size (MB), peak Python memory (MB), and optional absolute `p95_latency_ms` | `FAIL` when ratio CI lower bound `> max_latency_ratio` (`1.5x`) or size/RAM/p95 ceiling exceeded; `WARN` when point ratio exceeds budget and CI lower `> 1.0` |
| 8 | **Fairness** | Cell-count multinomial paired bootstrap CIs for demographic parity, equal opportunity, equalized odds, and calibration gaps relative to champion, plus two-sided per-group selection-rate shift and 80%-power MDE notes | `FAIL` on statistically significant gap increase (`fail_gap_increase: 0.05` or CI lower `>= 0.02`); `WARN` on moderate increase or significant selection-rate shift (`>= 0.02`); optional `enforce_absolute: true` |

## Verdict Aggregation

`src/predgate/verdict.py` aggregates all enabled `CheckResult` statuses:

1. Any check with `status == "fail"` → overall verdict **`FAIL`** (CLI exits with status `1` when `verdict.fail_if_any_fail: true`).
2. No `fail`, at least one `warn` → overall verdict **`WARN`** (CLI exits `0`, surfaces reasons for human review).
3. All checks `pass` or `skip` → overall verdict **`PASS`**.

## Inconclusive Slices (`slices.on_inconclusive`)

A slice is marked **inconclusive** only when its 95% bootstrap confidence interval is wider than `slices.max_ci_width` (default `0.10`) **and** straddles `max_drop` (`ci_lower <= max_drop < ci_upper`). A slice whose CI lies entirely above `max_drop` (`ci_lower > max_drop`) is a conclusive regression (`FAIL`), never downgraded to inconclusive.

When inconclusive slices are present, PredGate highlights them across the PR comment, HTML report, and React dashboard:

- `slices.on_inconclusive: warn` *(default)* — elevates an otherwise-passing `slices` check to `WARN` so underpowered slices cannot silently pass.
- `slices.on_inconclusive: info` — keeps an otherwise-passing `slices` check at `PASS` while still displaying the inconclusive banner.

See [Statistical Power](../power.md) for the exact minimum rows per slice required for 80% power.

## Model Family Change Note

Whenever the candidate's underlying estimator class differs from the champion's (for example, `GradientBoostingClassifier` → `RandomForestClassifier`, unwrapping `Pipeline`, `GridSearchCV`, `CalibratedClassifierCV`, and custom wrappers without unwrapping ensemble base learners), PredGate records `model_family_change` in `verdict.json` and displays a **Model family changed** banner in the CLI output, PR comment, HTML report, and Dashboard Run Detail view. This gives reviewers immediate context when an intentional architecture swap alters single-row latency or subgroup selection rates.
