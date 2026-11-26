# Runtime & Memory Performance (`predgate run`)

Measured wall-clock runtime and peak resident memory (`ru_maxrss`) of `predgate run` running the **full 8-check suite with default settings** (`slices.n_bootstrap=1000`, `fairness.n_bootstrap=200`, `fairness.selection_rate_n_bootstrap=1000`, `budgets.latency_samples=20`, `budgets.latency_batch_size=5`) against a persisted GradientBoosting champion on synthetic credit data (3,000 train rows, 1,000 reference rows, varying test rows).

Reproduced with [`scripts/bench_perf.py`](../scripts/bench_perf.py) on a 4-vCPU Linux host (Python 3.12.3, `scikit-learn 1.9.1`, `mlflow 3.18.0`):

| Test Rows | Wall Time (Before → After) | Sum of 8 Checks | Peak RSS (Before → After) | `fairness` (Before → After) | `slices` | `budgets` | `behavioral` | `data_contract` | `drift` + `leakage` + `label_shift` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1,000** | 53.7 s → **34.6 s** *(18.8 s warm SQLite)* | 6.1 s | 336 MB → **334 MB** | 0.34 s → **0.24 s** | 0.27 s | 4.63 s | 0.11 s | 0.61 s | 0.26 s |
| **10,000** | 55.8 s → **36.5 s** | 8.2 s | 463 MB → **335 MB** | 2.50 s → **1.34 s** | 0.44 s | 5.16 s | 0.32 s | 0.59 s | 0.34 s |
| **100,000** | 129.2 s → **50.3 s** *(2.6× faster)* | 18.4 s | 1,749 MB → **355 MB** *(4.9× less RAM)* | 78.47 s → **10.46 s** *(7.5× faster)* | 1.81 s | 3.15 s | 1.66 s | 0.87 s | 0.48 s |

## Overhead Breakdown & Optimizations Applied

- **Eliminated ~19–29 s of avoidable startup/logging overhead at 1,000 rows**:
  - Replaced MLflow's default subprocess-based pip-environment inference during `mlflow.sklearn.log_model` (~18.4 s) with explicit `pip_requirements` resolved via `importlib.metadata` (~0.57 s) and batched `log_params` / `log_metrics` / `set_tags` (1.19 s → 0.15 s).
  - Replaced `subprocess.run([python, "-m", "pip", "freeze"])` (~4.2 s) in lineage capture with `importlib.metadata.version()` over core runtime packages (~0.07 s).
  - Skipped initializing a fresh SQLite MLflow DB during champion lookup when the DB file does not yet exist (~4.5 s saved when falling back to verified `champion/model.pkl`).
  - Remaining non-check time (~12 s warm SQLite / ~18–28 s fresh SQLite) is dominated by Python + MLflow/scikit-learn/Pandera cold imports (~8–11 s) and one-time SQLite Alembic schema creation on a brand-new database (~11 s).
- **Cell-count multinomial bootstrap for fairness gaps (100k rows: `fairness` 78.5 s → 10.5 s, RSS 1.75 GB → 355 MB)**:
  - `paired_bootstrap_gap_ci` now uses exact `≤ 8G`-cell multinomial resampling over `(group, y, cand_pred, champ_pred)` contingency counts for demographic parity, equal opportunity, and equalized odds (identical distribution in `O(n + B·8G)` time and `< 50 KB` memory) and exact group-partitioned streaming resampling for continuous `calibration_gap_diff` (eliminating `(B, n)` dense matrix allocations).
- **Latency budget check (~3–5 s, constant in test rows)**: `budgets` runs 20 interleaved single-row batches of 5 predictions (`200` total calls plus warmup), so its runtime is independent of test set size.
