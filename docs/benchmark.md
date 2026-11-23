# ModelGate Benchmark Results

Single source of truth for the ModelGate fault-injection benchmark, generated strictly from [`docs/benchmark_results.json`](benchmark_results.json).

## Reproducibility Metadata

- **Command**: `python scripts/bench_public.py --n-seeds 30 --synth-start-seed 1000 --fairness-start-seed 5000`
- **Commit Hash**: `df94431`
- **Evaluated Seeds**: `1000..1029` (N=30: `1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029`)
- **No-op False-Positive Rate (ModelGate)**: **0/30 (0.0% observed FP; Wilson 95% CI [0.0%, 11.4%], i.e. an 11.4% 95% upper bound at N=30)**
- **No-op False-Positive Rate (Baseline)**: **0/30 (0.0% observed FP; Wilson 95% CI [0.0%, 11.4%], i.e. an 11.4% 95% upper bound at N=30)**

## Detection Rates (Synthetic Credit Benchmark, Seeds `1000..1029`)

A winner is declared only when the 95% Wilson confidence intervals do not overlap; overlapping 95% CIs are reported as a statistical **Tie (overlapping 95% CIs)**.

| Fault | Injected Defect | ModelGate (Count, Rate, Wilson 95% CI) | Baseline (Count, Rate, Wilson 95% CI) | Winner |
|-------|-----------------|----------------------------------------|---------------------------------------|--------|
| `feature_leakage` | Add a column derived from the target | 30/30 (100.0%, [88.6%, 100.0%]) | 30/30 (100.0%, [88.6%, 100.0%]) | **Tie (overlapping 95% CIs)** |
| `label_shift` | Corrupt labels in one subgroup of the train set | 30/30 (100.0%, [88.6%, 100.0%]) | 28/30 (93.3%, [78.7%, 98.2%]) | **Tie (overlapping 95% CIs)** |
| `degraded_slice` | Remove most of one subgroup from training data | 30/30 (100.0%, [88.6%, 100.0%]) | 0/30 (0.0%, [0.0%, 11.4%]) | **ModelGate** |
| `schema_drift` | Change a column's scale or unit | 30/30 (100.0%, [88.6%, 100.0%]) | 17/30 (56.7%, [39.2%, 72.6%]) | **ModelGate** |
| `latency_regression` | Swap in a much larger model | 30/30 (100.0%, [88.6%, 100.0%]) | 0/30 (0.0%, [0.0%, 11.4%]) | **ModelGate** |
| `fairness_regression` | Bias sampling against one protected group | 30/30 (100.0%, [88.6%, 100.0%]) | 9/30 (30.0%, [16.7%, 47.9%]) | **ModelGate** |
| `noop` | Retrain with a different seed (should PASS) | 0/30 (0.0% FP, [0.0%, 11.4%]) | 0/30 (0.0% FP, [0.0%, 11.4%]) | **Tie (11.4% 95% CI upper bound)** |

## Baseline Definitions

1. **Two-Sided Accuracy-Change Baseline (`|acc_champ - acc_cand| > 0.01`, shown in the table above)**: Flags any trial where overall test accuracy changes by more than 1 percentage point in either direction (drop or increase).
2. **One-Sided Accuracy-Drop Baseline (`acc_champ - acc_cand > 0.01`)**: Flags only when overall test accuracy *drops* by more than 1 percentage point (misses `feature_leakage`, which artificially increases accuracy, while ModelGate's `run_leakage_check` catches it directly).

## No-op False-Positive Rate & Upper Bound

For the `noop` control (retraining the identical pipeline on clean data with candidate seed `s + 500`), ModelGate produced **0/30 (0.0%)** and the baseline produced **0/30 (0.0%)** observed false positives across seeds `1000..1029` (Wilson 95% CI `[0.0%, 11.4%]`).

## Independent Fairness Validation on Public Datasets (30 Fresh Seeds `5000..5029`, Frozen Defaults)

- **Command**: `python scripts/bench_public.py --n-seeds 30 --fairness-start-seed 5000`
- **Commit Hash**: `df94431`
- **Frozen `FairnessConfig` Settings**: `enforce_absolute=False`, `max_gap_increase=0.02`, `fail_gap_increase=0.05`, `selection_rate_shift_warn=0.02`, `confidence=0.95`, `n_bootstrap=200`, `selection_rate_n_bootstrap=1000`

Evaluated using the **full `run_fairness_check` gate** (`status in ('warn', 'fail')`: paired bootstrap gap CI across demographic parity, equal opportunity, equalized odds, and per-group calibration + two-sided paired bootstrap per-group selection-rate shift at `WARN` level; static absolute DP opt-in via `enforce_absolute=True` relative to champion):

| Dataset (`5000..5029`) | Fault | Full Fairness Check (Count, Rate, Wilson 95% CI; Fail/Warn) | Accuracy Baseline (`\|Δacc\| > 1%`) | Mean DP (`champ -> cand`) | Power / MDE Note (`80% power`) |
|------------------------|-------|------------------------------------------------------------|--------------------------------------|---------------------------|--------------------------------|
| **UCI Adult** | `fairness_threshold_shift` (`τ=0.25`) | 30/30 (100.0%, [88.6%, 100.0%]; 2F/28W) | 7/30 (23.3%, [11.8%, 40.9%]) | `0.167 -> 0.102` | `min_n=495, MDE_DP=0.066, MDE_EO=0.212` (note emitted 30/30) |
| **UCI Adult** | `fairness_threshold_shift_hi` (`τ=0.75`) | 30/30 (100.0%, [88.6%, 100.0%]; 29F/1W) | 1/30 (3.3%, [0.6%, 16.7%]) | `0.167 -> 0.199` | `min_n=495, MDE_DP=0.066, MDE_EO=0.212` (note emitted 30/30) |
| **UCI Adult** | `fairness_proxy_feature` | 30/30 (100.0%, [88.6%, 100.0%]; 30F/0W) | 22/30 (73.3%, [55.6%, 85.8%]) | `0.167 -> 0.081` | `min_n=495, MDE_DP=0.066, MDE_EO=0.212` (note emitted 30/30) |
| **UCI Adult** | `fairness_regression` | 30/30 (100.0%, [88.6%, 100.0%]; 30F/0W) | 16/30 (53.3%, [36.1%, 69.8%]) | `0.167 -> 0.100` | `min_n=495, MDE_DP=0.066, MDE_EO=0.212` (note emitted 30/30) |
| **UCI Adult** | `noop` (FP rate) | **0/30 (0.0% FP, [0.0%, 11.4%]; 0F/0W)** | **0/30 (0.0% FP, [0.0%, 11.4%])** | `0.167 -> 0.168` | `min_n=495, MDE_DP=0.066, MDE_EO=0.212` (note emitted 30/30) |
| **German Credit** | `fairness_threshold_shift` (`τ=0.25`) | 30/30 (100.0%, [88.6%, 100.0%]; 30F/0W) | 18/30 (60.0%, [42.3%, 75.4%]) | `0.043 -> 0.351` | `min_n=109, MDE_DP=0.148, MDE_EO=0.285` (note emitted 30/30) |
| **German Credit** | `fairness_threshold_shift_hi` (`τ=0.75`) | 30/30 (100.0%, [88.6%, 100.0%]; 21F/9W) | 18/30 (60.0%, [42.3%, 75.4%]) | `0.043 -> 0.157` | `min_n=109, MDE_DP=0.148, MDE_EO=0.285` (note emitted 30/30) |
| **German Credit** | `fairness_proxy_feature` | 30/30 (100.0%, [88.6%, 100.0%]; 30F/0W) | 27/30 (90.0%, [74.4%, 96.5%]) | `0.043 -> 0.383` | `min_n=109, MDE_DP=0.148, MDE_EO=0.285` (note emitted 30/30) |
| **German Credit** | `fairness_regression` | 30/30 (100.0%, [88.6%, 100.0%]; 30F/0W) | 22/30 (73.3%, [55.6%, 85.8%]) | `0.043 -> 0.348` | `min_n=109, MDE_DP=0.148, MDE_EO=0.285` (note emitted 30/30) |
| **German Credit** | `noop` (FP rate) | **0/30 (0.0% FP, [0.0%, 11.4%]; 0F/0W)** | **1/30 (3.3% FP, [0.6%, 16.7%])** | `0.043 -> 0.043` | `min_n=109, MDE_DP=0.148, MDE_EO=0.285` (note emitted 30/30) |
| **Synthetic Credit** | `fairness_threshold_shift` (`τ=0.25`) | 13/30 (43.3%, [27.4%, 60.8%]; 2F/11W) | 1/30 (3.3%, [0.6%, 16.7%]) | `0.002 -> 0.013` | `min_n=674, MDE_DP=0.042, MDE_EO=0.243` (note emitted 0/30) |
| **Synthetic Credit** | `fairness_threshold_shift_hi` (`τ=0.75`) | 0/30 (0.0%, [0.0%, 11.4%]; 0F/0W) | 0/30 (0.0%, [0.0%, 11.4%]) | `0.002 -> 0.003` | `min_n=674, MDE_DP=0.042, MDE_EO=0.243` (note emitted 0/30) |
| **Synthetic Credit** | `fairness_proxy_feature` | 30/30 (100.0%, [88.6%, 100.0%]; 30F/0W) | 30/30 (100.0%, [88.6%, 100.0%]) | `0.002 -> 0.064` | `min_n=674, MDE_DP=0.042, MDE_EO=0.243` (note emitted 0/30) |
| **Synthetic Credit** | `fairness_regression` | 30/30 (100.0%, [88.6%, 100.0%]; 30F/0W) | 9/30 (30.0%, [16.7%, 47.9%]) | `0.002 -> 0.028` | `min_n=674, MDE_DP=0.042, MDE_EO=0.243` (note emitted 0/30) |
| **Synthetic Credit** | `noop` (FP rate) | **0/30 (0.0% FP, [0.0%, 11.4%]; 0F/0W)** | **0/30 (0.0% FP, [0.0%, 11.4%])** | `0.002 -> 0.002` | `min_n=674, MDE_DP=0.042, MDE_EO=0.243` (note emitted 0/30) |

## Methodology

Each synthetic fault is injected across 30 independent random seeds (`1000..1029`). For each seed `s`, training data (`N=3000`) is generated with `RandomState(s)` and held-out test data (`N=1500`) with `RandomState(s + 1000)`. Public fairness datasets (UCI Adult `adult` v2 and German Credit `credit-g` v1) are fetched directly from OpenML via `scripts/bench_public.py`. Confidence intervals are calculated using the Wilson score interval at the 95% confidence level (`z = 1.95996`).
