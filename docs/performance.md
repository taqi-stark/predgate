# Runtime & Memory Performance (`predgate run`)

Measured wall-clock runtime and peak resident memory (`ru_maxrss`) of `predgate run` running the **full 8-check suite with default settings** (`slices.n_bootstrap=1000`, `fairness.n_bootstrap=200`, `fairness.selection_rate_n_bootstrap=1000`, `budgets.latency_samples=20`, `budgets.latency_batch_size=5`) against a persisted GradientBoosting champion on synthetic credit data (3,000 train rows, 1,000 reference rows, varying test rows).

Reproduced with [`scripts/bench_perf.py`](../scripts/bench_perf.py) on a 4-vCPU Linux host (Python 3.12.3, `scikit-learn 1.9.1`, `mlflow 3.17.2`):

| Test Rows | Total Wall Time (`predgate run`) | Sum of 8 Checks | Peak RSS Memory | `slices` | `fairness` | `budgets` | `behavioral` | `data_contract` | `drift` + `leakage` + `label_shift` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1,000** | **53.7 s** | 6.6 s | **336 MB** | 0.31 s | 0.34 s | 4.90 s | 0.13 s | 0.65 s | 0.27 s |
| **10,000** | **55.8 s** | 9.3 s | **463 MB** | 0.44 s | 2.50 s | 5.04 s | 0.28 s | 0.71 s | 0.29 s |
| **100,000** | **129.2 s** | 91.1 s | **1,749 MB** | 2.64 s | 78.47 s | 6.46 s | 2.23 s | 0.76 s | 0.54 s |

## Bottlenecks & Practical Scaling Notes

- **Fixed startup & MLflow logging (~38–47 s)**: Cold-starting Python, importing `mlflow` / `pandera` / `scikit-learn`, capturing `pip freeze` package versions for lineage, and logging the candidate model to SQLite MLflow account for ~40–47 s regardless of test-set size.
- **Latency budget check (~5–6 s, constant in test rows)**: `budgets` runs 20 interleaved single-row batches of 5 predictions (`200` total calls plus warmup), so its runtime is independent of test set size.
- **100k-row scaling (`fairness` = 78.5 s, Peak RSS = 1.75 GB)**: At 100,000 test rows, `fairness` dominates both compute and RAM because `selection_rate_n_bootstrap=1000` and `n_bootstrap=200` materialize `(n_bootstrap, n_group)` integer index matrices in NumPy.
- **Tuning for very large test sets (`>= 50k` rows)**: On 2 GB CI runners with 100k+ test rows, either evaluate on a stratified 20k–50k held-out sample (which already far exceeds 80%-power sample sizes in [`docs/power.md`](power.md)) or lower `fairness.selection_rate_n_bootstrap` (e.g., `500`) and `slices.n_bootstrap` (e.g., `500`).
