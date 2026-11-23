# PredGate Benchmark Results

Single source of truth for all PredGate benchmark evaluations, generated strictly from [`docs/benchmark_results.json`](benchmark_results.json) (aggregate summary; no hardcoded numbers).

## Reproducibility Metadata

- **Command**: `python scripts/bench_public.py --n-seeds 30 --synth-start-seed 1000 --fairness-start-seed 5000 --benign-start-seed 6000 --heldout-start-seed 7000`
- **Evaluated Code Commit Hash**: `8467d37`
- **Synthetic Suite Seeds**: `1000..1029` (N=30)
- **No-op FP (PredGate FAIL-only)**: **0/30 (0.0% FP, [0.0%, 11.4%])**
- **No-op FP (PredGate Flagged WARN+FAIL)**: **0/30 (0.0% FP, [0.0%, 11.4%])**
- **No-op FP (Overall Accuracy Baseline)**: **0/30 (0.0% FP, [0.0%, 11.4%])**
- **Per-Seed Records Release Artifact**: `release_artifacts/benchmark_per_seed_8467d37.json` (excluded from repo JSON to keep git lean; attach to GitHub Release)

## 1. Synthetic Credit 7-Fault Benchmark (Seeds `1000..1029`)

Reports **FAIL-only** (hard merge block) and **Flagged (WARN+FAIL)** (blocked or flagged for human review) separately alongside stronger baselines.

| Fault | PredGate FAIL-only (95% CI) | PredGate Flagged WARN+FAIL (95% CI) | Overall Acc (`\|Δacc\|>1%`) | Per-Group Acc Drop (`>2%`) | Fixed DP (`>=0.10`) | Evidently 0.7 Defaults |
|-------|-----------------------------|-------------------------------------|------------------------------|----------------------------|---------------------|------------------------|
| `feature_leakage` | 30/30 (100.0%, [88.6%, 100.0%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 0/30 (0.0%, [0.0%, 11.4%]) | 0/30 (0.0%, [0.0%, 11.4%]) | 30/30 (100.0%, [88.6%, 100.0%]) |
| `label_shift` | 30/30 (100.0%, [88.6%, 100.0%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 28/30 (93.3%, [78.7%, 98.2%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 0/30 (0.0%, [0.0%, 11.4%]) | 30/30 (100.0%, [88.6%, 100.0%]) |
| `degraded_slice` | 30/30 (100.0%, [88.6%, 100.0%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 0/30 (0.0%, [0.0%, 11.4%]) | 0/30 (0.0%, [0.0%, 11.4%]) | 0/30 (0.0%, [0.0%, 11.4%]) | 30/30 (100.0%, [88.6%, 100.0%]) |
| `schema_drift` | 30/30 (100.0%, [88.6%, 100.0%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 17/30 (56.7%, [39.2%, 72.6%]) | 15/30 (50.0%, [33.2%, 66.8%]) | 0/30 (0.0%, [0.0%, 11.4%]) | 30/30 (100.0%, [88.6%, 100.0%]) |
| `latency_regression` | 30/30 (100.0%, [88.6%, 100.0%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 0/30 (0.0%, [0.0%, 11.4%]) | 0/30 (0.0%, [0.0%, 11.4%]) | 0/30 (0.0%, [0.0%, 11.4%]) | 27/30 (90.0%, [74.4%, 96.5%]) |
| `fairness_regression` | 30/30 (100.0%, [88.6%, 100.0%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 9/30 (30.0%, [16.7%, 47.9%]) | 12/30 (40.0%, [24.6%, 57.7%]) | 0/30 (0.0%, [0.0%, 11.4%]) | 30/30 (100.0%, [88.6%, 100.0%]) |
| `noop` | **0/30 (0.0% FP, [0.0%, 11.4%])** | **0/30 (0.0% FP, [0.0%, 11.4%])** | **0/30 (0.0% FP, [0.0%, 11.4%])** | 0/30 (0.0% FP, [0.0%, 11.4%]) | 0/30 (0.0% FP, [0.0%, 11.4%]) | 10/30 (33.3% FP, [19.2%, 51.2%]) |

## 2. Benign-Change Control (Harmless Retrains, Seeds `6000..6029`, Frozen Defaults)

Six realistic harmless retrain scenarios (`hyperparam_change`, `algo_swap_rf`, `algo_swap_logreg`, `more_data_20pct`, `irrelevant_feature`, `preprocessing_change`) across 30 seeds × 3 datasets (**UCI Adult**, **German Credit**, **Synthetic Credit**; 540 total trials), running all 6 schema-agnostic checks (`drift`, `leakage`, `label_shift`, `slices`, `budgets`, `fairness`) with zero threshold tuning.

- **Overall Benign FAIL Rate (Hard Block FP)**: **282/540 (52.2%, [48.0%, 56.4%])**
- **Overall Benign WARN-Only Rate (Alert Fatigue)**: **14/540 (2.6%, [1.6%, 4.3%])**
- **Overall Benign Flagged Rate (WARN+FAIL)**: **296/540 (54.8%, [50.6%, 59.0%])**

| Dataset | Benign Change | PredGate FAIL (95% CI) | PredGate WARN-Only (Alert Fatigue) | PredGate Flagged (WARN+FAIL) | Per-Group Acc Drop | Fixed DP (`>=0.10`) | Evidently 0.7 Defaults |
|---------|---------------|------------------------|------------------------------------|------------------------------|--------------------|---------------------|------------------------|
| **UCI Adult** | `hyperparam_change` | 4/30 (13.3%, [5.3%, 29.7%]) | 6/30 (20.0%, [9.5%, 37.3%]) | 10/30 (33.3%, [19.2%, 51.2%]) | 4/30 (13.3%, [5.3%, 29.7%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 8/30 (26.7%, [14.2%, 44.4%]) |
| **UCI Adult** | `algo_swap_rf` | 30/30 (100.0%, [88.6%, 100.0%]) | 0/30 (0.0%, [0.0%, 11.4%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 17/30 (56.7%, [39.2%, 72.6%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 16/30 (53.3%, [36.1%, 69.8%]) |
| **UCI Adult** | `algo_swap_logreg` | 22/30 (73.3%, [55.6%, 85.8%]) | 6/30 (20.0%, [9.5%, 37.3%]) | 28/30 (93.3%, [78.7%, 98.2%]) | 19/30 (63.3%, [45.5%, 78.1%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 26/30 (86.7%, [70.3%, 94.7%]) |
| **UCI Adult** | `more_data_20pct` | 5/30 (16.7%, [7.3%, 33.6%]) | 1/30 (3.3%, [0.6%, 16.7%]) | 6/30 (20.0%, [9.5%, 37.3%]) | 4/30 (13.3%, [5.3%, 29.7%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 1/30 (3.3%, [0.6%, 16.7%]) |
| **UCI Adult** | `irrelevant_feature` | 30/30 (100.0%, [88.6%, 100.0%]) | 0/30 (0.0%, [0.0%, 11.4%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 2/30 (6.7%, [1.8%, 21.3%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 0/30 (0.0%, [0.0%, 11.4%]) |
| **UCI Adult** | `preprocessing_change` | 1/30 (3.3%, [0.6%, 16.7%]) | 0/30 (0.0%, [0.0%, 11.4%]) | 1/30 (3.3%, [0.6%, 16.7%]) | 0/30 (0.0%, [0.0%, 11.4%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 0/30 (0.0%, [0.0%, 11.4%]) |
| **German Credit** | `hyperparam_change` | 19/30 (63.3%, [45.5%, 78.1%]) | 0/30 (0.0%, [0.0%, 11.4%]) | 19/30 (63.3%, [45.5%, 78.1%]) | 16/30 (53.3%, [36.1%, 69.8%]) | 12/30 (40.0%, [24.6%, 57.7%]) | 18/30 (60.0%, [42.3%, 75.4%]) |
| **German Credit** | `algo_swap_rf` | 30/30 (100.0%, [88.6%, 100.0%]) | 0/30 (0.0%, [0.0%, 11.4%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 26/30 (86.7%, [70.3%, 94.7%]) | 3/30 (10.0%, [3.5%, 25.6%]) | 30/30 (100.0%, [88.6%, 100.0%]) |
| **German Credit** | `algo_swap_logreg` | 27/30 (90.0%, [74.4%, 96.5%]) | 1/30 (3.3%, [0.6%, 16.7%]) | 28/30 (93.3%, [78.7%, 98.2%]) | 18/30 (60.0%, [42.3%, 75.4%]) | 16/30 (53.3%, [36.1%, 69.8%]) | 19/30 (63.3%, [45.5%, 78.1%]) |
| **German Credit** | `more_data_20pct` | 22/30 (73.3%, [55.6%, 85.8%]) | 0/30 (0.0%, [0.0%, 11.4%]) | 22/30 (73.3%, [55.6%, 85.8%]) | 11/30 (36.7%, [21.9%, 54.5%]) | 7/30 (23.3%, [11.8%, 40.9%]) | 11/30 (36.7%, [21.9%, 54.5%]) |
| **German Credit** | `irrelevant_feature` | 30/30 (100.0%, [88.6%, 100.0%]) | 0/30 (0.0%, [0.0%, 11.4%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 21/30 (70.0%, [52.1%, 83.3%]) | 6/30 (20.0%, [9.5%, 37.3%]) | 6/30 (20.0%, [9.5%, 37.3%]) |
| **German Credit** | `preprocessing_change` | 2/30 (6.7%, [1.8%, 21.3%]) | 0/30 (0.0%, [0.0%, 11.4%]) | 2/30 (6.7%, [1.8%, 21.3%]) | 3/30 (10.0%, [3.5%, 25.6%]) | 6/30 (20.0%, [9.5%, 37.3%]) | 0/30 (0.0%, [0.0%, 11.4%]) |
| **Synthetic Credit** | `hyperparam_change` | 0/30 (0.0%, [0.0%, 11.4%]) | 0/30 (0.0%, [0.0%, 11.4%]) | 0/30 (0.0%, [0.0%, 11.4%]) | 0/30 (0.0%, [0.0%, 11.4%]) | 0/30 (0.0%, [0.0%, 11.4%]) | 28/30 (93.3%, [78.7%, 98.2%]) |
| **Synthetic Credit** | `algo_swap_rf` | 30/30 (100.0%, [88.6%, 100.0%]) | 0/30 (0.0%, [0.0%, 11.4%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 0/30 (0.0%, [0.0%, 11.4%]) | 0/30 (0.0%, [0.0%, 11.4%]) | 28/30 (93.3%, [78.7%, 98.2%]) |
| **Synthetic Credit** | `algo_swap_logreg` | 0/30 (0.0%, [0.0%, 11.4%]) | 0/30 (0.0%, [0.0%, 11.4%]) | 0/30 (0.0%, [0.0%, 11.4%]) | 0/30 (0.0%, [0.0%, 11.4%]) | 0/30 (0.0%, [0.0%, 11.4%]) | 28/30 (93.3%, [78.7%, 98.2%]) |
| **Synthetic Credit** | `more_data_20pct` | 0/30 (0.0%, [0.0%, 11.4%]) | 0/30 (0.0%, [0.0%, 11.4%]) | 0/30 (0.0%, [0.0%, 11.4%]) | 0/30 (0.0%, [0.0%, 11.4%]) | 0/30 (0.0%, [0.0%, 11.4%]) | 24/30 (80.0%, [62.7%, 90.5%]) |
| **Synthetic Credit** | `irrelevant_feature` | 30/30 (100.0%, [88.6%, 100.0%]) | 0/30 (0.0%, [0.0%, 11.4%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 0/30 (0.0%, [0.0%, 11.4%]) | 0/30 (0.0%, [0.0%, 11.4%]) | 23/30 (76.7%, [59.1%, 88.2%]) |
| **Synthetic Credit** | `preprocessing_change` | 0/30 (0.0%, [0.0%, 11.4%]) | 0/30 (0.0%, [0.0%, 11.4%]) | 0/30 (0.0%, [0.0%, 11.4%]) | 0/30 (0.0%, [0.0%, 11.4%]) | 0/30 (0.0%, [0.0%, 11.4%]) | 11/30 (36.7%, [21.9%, 54.5%]) |

### Per-Check Benign-Change FAIL & WARN Rates (All Datasets Combined)

| Check | FAIL Count & Rate (95% CI) | WARN Count & Rate (Alert Fatigue, 95% CI) | Flagged WARN+FAIL (95% CI) |
|-------|----------------------------|-------------------------------------------|----------------------------|
| `drift` | 90/540 (16.7%, [13.8%, 20.0%]) | 0/540 (0.0%, [0.0%, 0.7%]) | 90/540 (16.7%, [13.8%, 20.0%]) |
| `leakage` | 0/540 (0.0%, [0.0%, 0.7%]) | 0/540 (0.0%, [0.0%, 0.7%]) | 0/540 (0.0%, [0.0%, 0.7%]) |
| `label_shift` | 0/540 (0.0%, [0.0%, 0.7%]) | 0/540 (0.0%, [0.0%, 0.7%]) | 0/540 (0.0%, [0.0%, 0.7%]) |
| `slices` | 184/540 (34.1%, [30.2%, 38.2%]) | 0/540 (0.0%, [0.0%, 0.7%]) | 184/540 (34.1%, [30.2%, 38.2%]) |
| `budgets` | 87/540 (16.1%, [13.3%, 19.4%]) | 3/540 (0.6%, [0.2%, 1.6%]) | 90/540 (16.7%, [13.8%, 20.0%]) |
| `fairness` | 22/540 (4.1%, [2.7%, 6.1%]) | 93/540 (17.2%, [14.3%, 20.6%]) | 115/540 (21.3%, [18.1%, 24.9%]) |

## 3. Held-Out Validation (Unseen Datasets & Unseen Fault Families, Seeds `7000..7029`, Frozen Defaults)

Evaluated with **frozen thresholds** (zero tuning) on two held-out OpenML datasets never used during development — **Bank Marketing** (`data_id=1461`, protected=`marital`) and **COMPAS Recidivism** (`data_id=42192`, protected=`race`) — including two brand-new fault families not used in design (`silent_null_imputation`: silent 40% null-to-zero corruption of primary numeric feature in training data; `cohort_feature_corruption`: cohort-localized 2σ feature corruption at inference time with clean training data).

| Held-Out Dataset | Fault | PredGate FAIL-only (95% CI) | PredGate Flagged WARN+FAIL (95% CI) | Overall Acc (`\|Δacc\|>1%`) | Per-Group Acc Drop (`>2%`) | Fixed DP (`>=0.10`) | Evidently 0.7 Defaults | Power / MDE Note |
|------------------|-------|-----------------------------|-------------------------------------|------------------------------|----------------------------|---------------------|------------------------|------------------|
| **Bank Marketing** | `silent_null_imputation` (new fault: 40% null->0 in train) | 21/30 (70.0%, [52.1%, 83.3%]) | 25/30 (83.3%, [66.4%, 92.7%]) | 0/30 (0.0%, [0.0%, 11.4%]) | 9/30 (30.0%, [16.7%, 47.9%]) | 0/30 (0.0%, [0.0%, 11.4%]) | 30/30 (100.0%, [88.6%, 100.0%]) | `min_n=172, MDE_DP=0.075` (note 30/30) |
| **Bank Marketing** | `cohort_feature_corruption` (new fault: 2σ cohort noise) | 30/30 (100.0%, [88.6%, 100.0%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 27/30 (90.0%, [74.4%, 96.5%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 28/30 (93.3%, [78.7%, 98.2%]) | 30/30 (100.0%, [88.6%, 100.0%]) | `min_n=172, MDE_DP=0.075` (note 30/30) |
| **Bank Marketing** | `fairness_threshold_shift` (`τ=0.25`) | 21/30 (70.0%, [52.1%, 83.3%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 0/30 (0.0%, [0.0%, 11.4%]) | 8/30 (26.7%, [14.2%, 44.4%]) | 14/30 (46.7%, [30.2%, 63.9%]) | 18/30 (60.0%, [42.3%, 75.4%]) | `min_n=172, MDE_DP=0.075` (note 30/30) |
| **Bank Marketing** | `fairness_proxy_feature` | 30/30 (100.0%, [88.6%, 100.0%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 10/30 (33.3%, [19.2%, 51.2%]) | 28/30 (93.3%, [78.7%, 98.2%]) | 23/30 (76.7%, [59.1%, 88.2%]) | 19/30 (63.3%, [45.5%, 78.1%]) | `min_n=172, MDE_DP=0.075` (note 30/30) |
| **Bank Marketing** | `fairness_regression` | 30/30 (100.0%, [88.6%, 100.0%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 3/30 (10.0%, [3.5%, 25.6%]) | 23/30 (76.7%, [59.1%, 88.2%]) | 17/30 (56.7%, [39.2%, 72.6%]) | 12/30 (40.0%, [24.6%, 57.7%]) | `min_n=172, MDE_DP=0.075` (note 30/30) |
| **Bank Marketing** | `noop` (FP rate) | **1/30 (3.3% FP, [0.6%, 16.7%])** | **1/30 (3.3% FP, [0.6%, 16.7%])** | 0/30 (0.0% FP, [0.0%, 11.4%]) | 1/30 (3.3% FP, [0.6%, 16.7%]) | 0/30 (0.0% FP, [0.0%, 11.4%]) | 0/30 (0.0% FP, [0.0%, 11.4%]) | `min_n=172, MDE_DP=0.075` (note 30/30) |
| **COMPAS** | `silent_null_imputation` (new fault: 40% null->0 in train) | 27/30 (90.0%, [74.4%, 96.5%]) | 29/30 (96.7%, [83.3%, 99.4%]) | 14/30 (46.7%, [30.2%, 63.9%]) | 9/30 (30.0%, [16.7%, 47.9%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 30/30 (100.0%, [88.6%, 100.0%]) | `min_n=593, MDE_DP=0.074` (note 30/30) |
| **COMPAS** | `cohort_feature_corruption` (new fault: 2σ cohort noise) | 30/30 (100.0%, [88.6%, 100.0%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 0/30 (0.0%, [0.0%, 11.4%]) | 30/30 (100.0%, [88.6%, 100.0%]) | `min_n=593, MDE_DP=0.074` (note 30/30) |
| **COMPAS** | `fairness_threshold_shift` (`τ=0.25`) | 30/30 (100.0%, [88.6%, 100.0%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 30/30 (100.0%, [88.6%, 100.0%]) | `min_n=593, MDE_DP=0.074` (note 30/30) |
| **COMPAS** | `fairness_proxy_feature` | 30/30 (100.0%, [88.6%, 100.0%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 30/30 (100.0%, [88.6%, 100.0%]) | `min_n=593, MDE_DP=0.074` (note 30/30) |
| **COMPAS** | `fairness_regression` | 30/30 (100.0%, [88.6%, 100.0%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 29/30 (96.7%, [83.3%, 99.4%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 13/30 (43.3%, [27.4%, 60.8%]) | 30/30 (100.0%, [88.6%, 100.0%]) | `min_n=593, MDE_DP=0.074` (note 30/30) |
| **COMPAS** | `noop` (FP rate) | **0/30 (0.0% FP, [0.0%, 11.4%])** | **0/30 (0.0% FP, [0.0%, 11.4%])** | 0/30 (0.0% FP, [0.0%, 11.4%]) | 0/30 (0.0% FP, [0.0%, 11.4%]) | 30/30 (100.0% FP, [88.6%, 100.0%]) | 0/30 (0.0% FP, [0.0%, 11.4%]) | `min_n=593, MDE_DP=0.074` (note 30/30) |

## 4. Public Fairness Validation (Adult, German Credit & Synthetic, Seeds `5000..5029`, Frozen Defaults)

- **Command**: `python scripts/bench_public.py --n-seeds 30 --fairness-start-seed 5000`
- **Evaluated Code Commit Hash**: `8467d37`
- **Frozen `FairnessConfig` Settings**: `enforce_absolute=False`, `max_gap_increase=0.02`, `fail_gap_increase=0.05`, `selection_rate_shift_warn=0.02`, `confidence=0.95`, `n_bootstrap=200`, `selection_rate_n_bootstrap=1000`

| Dataset (`5000..5029`) | Fault | Fairness FAIL-only (95% CI) | Fairness Flagged WARN+FAIL (95% CI) | Overall Acc (`\|Δacc\|>1%`) | Per-Group Acc Drop (`>2%`) | Fixed DP (`>=0.10`) | Mean DP (`champ -> cand`) | Power / MDE Note |
|------------------------|-------|-----------------------------|-------------------------------------|------------------------------|----------------------------|---------------------|---------------------------|------------------|
| **UCI Adult** | `fairness_threshold_shift` (`τ=0.25`) | 2/30 (6.7%, [1.8%, 21.3%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 7/30 (23.3%, [11.8%, 40.9%]) | 10/30 (33.3%, [19.2%, 51.2%]) | 16/30 (53.3%, [36.1%, 69.8%]) | `0.167 -> 0.102` | `min_n=495, MDE_DP=0.066, MDE_EO=0.212` (note 30/30) |
| **UCI Adult** | `fairness_threshold_shift_hi` (`τ=0.75`) | 29/30 (96.7%, [83.3%, 99.4%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 1/30 (3.3%, [0.6%, 16.7%]) | 4/30 (13.3%, [5.3%, 29.7%]) | 30/30 (100.0%, [88.6%, 100.0%]) | `0.167 -> 0.199` | `min_n=495, MDE_DP=0.066, MDE_EO=0.212` (note 30/30) |
| **UCI Adult** | `fairness_proxy_feature` | 30/30 (100.0%, [88.6%, 100.0%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 22/30 (73.3%, [55.6%, 85.8%]) | 25/30 (83.3%, [66.4%, 92.7%]) | 8/30 (26.7%, [14.2%, 44.4%]) | `0.167 -> 0.081` | `min_n=495, MDE_DP=0.066, MDE_EO=0.212` (note 30/30) |
| **UCI Adult** | `fairness_regression` | 30/30 (100.0%, [88.6%, 100.0%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 16/30 (53.3%, [36.1%, 69.8%]) | 17/30 (56.7%, [39.2%, 72.6%]) | 16/30 (53.3%, [36.1%, 69.8%]) | `0.167 -> 0.100` | `min_n=495, MDE_DP=0.066, MDE_EO=0.212` (note 30/30) |
| **UCI Adult** | `noop` (FP rate) | **0/30 (0.0% FP, [0.0%, 11.4%])** | **0/30 (0.0% FP, [0.0%, 11.4%])** | **0/30 (0.0% FP, [0.0%, 11.4%])** | 0/30 (0.0% FP, [0.0%, 11.4%]) | 30/30 (100.0% FP, [88.6%, 100.0%]) | `0.167 -> 0.168` | `min_n=495, MDE_DP=0.066, MDE_EO=0.212` (note 30/30) |
| **German Credit** | `fairness_threshold_shift` (`τ=0.25`) | 30/30 (100.0%, [88.6%, 100.0%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 18/30 (60.0%, [42.3%, 75.4%]) | 22/30 (73.3%, [55.6%, 85.8%]) | 30/30 (100.0%, [88.6%, 100.0%]) | `0.043 -> 0.351` | `min_n=109, MDE_DP=0.148, MDE_EO=0.285` (note 30/30) |
| **German Credit** | `fairness_threshold_shift_hi` (`τ=0.75`) | 21/30 (70.0%, [52.1%, 83.3%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 18/30 (60.0%, [42.3%, 75.4%]) | 22/30 (73.3%, [55.6%, 85.8%]) | 29/30 (96.7%, [83.3%, 99.4%]) | `0.043 -> 0.157` | `min_n=109, MDE_DP=0.148, MDE_EO=0.285` (note 30/30) |
| **German Credit** | `fairness_proxy_feature` | 30/30 (100.0%, [88.6%, 100.0%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 27/30 (90.0%, [74.4%, 96.5%]) | 28/30 (93.3%, [78.7%, 98.2%]) | 30/30 (100.0%, [88.6%, 100.0%]) | `0.043 -> 0.383` | `min_n=109, MDE_DP=0.148, MDE_EO=0.285` (note 30/30) |
| **German Credit** | `fairness_regression` | 30/30 (100.0%, [88.6%, 100.0%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 22/30 (73.3%, [55.6%, 85.8%]) | 26/30 (86.7%, [70.3%, 94.7%]) | 30/30 (100.0%, [88.6%, 100.0%]) | `0.043 -> 0.348` | `min_n=109, MDE_DP=0.148, MDE_EO=0.285` (note 30/30) |
| **German Credit** | `noop` (FP rate) | **0/30 (0.0% FP, [0.0%, 11.4%])** | **0/30 (0.0% FP, [0.0%, 11.4%])** | **1/30 (3.3% FP, [0.6%, 16.7%])** | 0/30 (0.0% FP, [0.0%, 11.4%]) | 4/30 (13.3% FP, [5.3%, 29.7%]) | `0.043 -> 0.043` | `min_n=109, MDE_DP=0.148, MDE_EO=0.285` (note 30/30) |
| **Synthetic Credit** | `fairness_threshold_shift` (`τ=0.25`) | 2/30 (6.7%, [1.8%, 21.3%]) | 13/30 (43.3%, [27.4%, 60.8%]) | 1/30 (3.3%, [0.6%, 16.7%]) | 2/30 (6.7%, [1.8%, 21.3%]) | 0/30 (0.0%, [0.0%, 11.4%]) | `0.002 -> 0.013` | `min_n=674, MDE_DP=0.042, MDE_EO=0.243` (note 0/30) |
| **Synthetic Credit** | `fairness_threshold_shift_hi` (`τ=0.75`) | 0/30 (0.0%, [0.0%, 11.4%]) | 0/30 (0.0%, [0.0%, 11.4%]) | 0/30 (0.0%, [0.0%, 11.4%]) | 0/30 (0.0%, [0.0%, 11.4%]) | 0/30 (0.0%, [0.0%, 11.4%]) | `0.002 -> 0.003` | `min_n=674, MDE_DP=0.042, MDE_EO=0.243` (note 0/30) |
| **Synthetic Credit** | `fairness_proxy_feature` | 30/30 (100.0%, [88.6%, 100.0%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 2/30 (6.7%, [1.8%, 21.3%]) | `0.002 -> 0.064` | `min_n=674, MDE_DP=0.042, MDE_EO=0.243` (note 0/30) |
| **Synthetic Credit** | `fairness_regression` | 30/30 (100.0%, [88.6%, 100.0%]) | 30/30 (100.0%, [88.6%, 100.0%]) | 9/30 (30.0%, [16.7%, 47.9%]) | 11/30 (36.7%, [21.9%, 54.5%]) | 0/30 (0.0%, [0.0%, 11.4%]) | `0.002 -> 0.028` | `min_n=674, MDE_DP=0.042, MDE_EO=0.243` (note 0/30) |
| **Synthetic Credit** | `noop` (FP rate) | **0/30 (0.0% FP, [0.0%, 11.4%])** | **0/30 (0.0% FP, [0.0%, 11.4%])** | **0/30 (0.0% FP, [0.0%, 11.4%])** | 0/30 (0.0% FP, [0.0%, 11.4%]) | 0/30 (0.0% FP, [0.0%, 11.4%]) | `0.002 -> 0.002` | `min_n=674, MDE_DP=0.042, MDE_EO=0.243` (note 0/30) |

## 5. Baseline & Comparator Definitions

1. **Overall Accuracy Baseline (`|acc_champ - acc_cand| > 0.01`)**: Flags any trial where overall test accuracy changes by more than 1 percentage point in either direction.
2. **Per-Group Accuracy-Drop Baseline (`max_g (acc_champ(g) - acc_cand(g)) > 0.02`, `n_g >= 30`)**: Flags any trial where any subgroup's accuracy drops by more than 2 percentage points compared to the champion.
3. **Fixed Demographic-Parity-Limit Baseline (`DP_cand >= 0.10`)**: Classic static absolute fairness threshold flagging any candidate with demographic parity difference `>= 0.10` regardless of champion baseline.
4. **Evidently Off-the-Shelf Default Comparator (`evidently==0.7.23`, `pandas==2.3.3`)**: Runs Report([DataDriftPreset()], include_tests=True) on candidate vs reference training data and Report([ClassificationPreset()], include_tests=True) on candidate vs champion test predictions with 100% Evidently default thresholds; flagged if any test returns FAIL.

## Methodology

Each suite uses 30 independent random seeds (Synthetic 7-fault: `1000..1029`; Public fairness: `5000..5029`; Benign-change control: `6000..6029`; Held-out validation: `7000..7029`). All PredGate thresholds are frozen at default values and never tuned against benign controls or held-out datasets. Public datasets (UCI Adult `adult` v2, German Credit `credit-g` v1, Bank Marketing `1461`, COMPAS `42192`) are fetched directly from OpenML via `scripts/bench_public.py`. Confidence intervals use the 95% Wilson score interval (`z = 1.95996`).
