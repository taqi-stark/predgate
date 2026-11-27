# Config Presets (`strict`, `balanced`, `quiet`)

Set `preset: strict | balanced | quiet` in `predgate.yaml` or pass `--preset <name>` to `predgate run`. Any explicit threshold set under `checks:` in `predgate.yaml` overrides the preset value.

Default thresholds are unchanged and match `balanced` exactly.

## Preset Threshold Matrix

| Setting | `strict` | `balanced` *(default)* | `quiet` |
|---|---|---|---|
| `drift.warn_at` / `fail_at` | `0.08` / `0.20` | `0.10` / `0.25` | `0.15` / `0.30` |
| `drift.on_new_feature` | `warn` | `info` | `info` |
| `drift.point_mass_warn_delta` / `fail_delta` | `0.08` / `0.15` | `0.10` / `0.20` | `0.15` / `0.25` |
| `leakage.warn_corr` / `fail_corr` | `0.40` / `0.65` | `0.50` / `0.75` | `0.55` / `0.80` |
| `leakage.warn_auc` / `fail_auc` | `0.80` / `0.88` | `0.85` / `0.92` | `0.88` / `0.94` |
| `label_shift.warn_delta` / `fail_delta` | `0.03` / `0.07` | `0.05` / `0.10` | `0.07` / `0.12` |
| `slices.max_drop` / `max_ci_width` | `0.02` / `0.08` | `0.03` / `0.10` | `0.05` / `0.15` |
| `slices.on_inconclusive` | `warn` | `warn` | `info` |
| `budgets.max_latency_ratio` | `1.25` | `1.50` | `2.00` |
| `fairness.max_gap_increase` / `fail_gap_increase` | `0.015` / `0.03` | `0.02` / `0.05` | `0.04` / `0.08` |
| `fairness.selection_rate_shift_warn` | `0.015` | `0.02` | `0.05` |

## Measured Benign-Change WARN / FAIL Rates

Measured independently on 30 fresh seeds (`11000..11029`) across all 5 benchmark datasets × 9 benign retrains (`1,350` trials per preset) via `scripts/bench_presets.py` and recorded in [`preset_benchmark_results.json`](../preset_benchmark_results.json) (95% Wilson CIs):

| Preset | Quality FAIL (Hard Block) | Quality WARN-Only (Alert Fatigue) | Latency FAIL (`budgets`) | Combined FAIL | Combined WARN-Only | Non-Swap Quality FAIL (N=1,050) |
|---|---|---|---|---|---|---|
| **`strict`** | 83/1350 (**6.1%**, [5.0%, 7.6%]) | 725/1350 (**53.7%**, [51.0%, 56.3%]) | 161/1350 (**11.9%**, [10.3%, 13.8%]) | 205/1350 (**15.2%**, [13.4%, 17.2%]) | 716/1350 (**53.0%**, [50.4%, 55.7%]) | 14/1050 (**1.3%**, [0.8%, 2.2%]) |
| **`balanced`** *(default)* | 49/1350 (**3.6%**, [2.8%, 4.8%]) | 247/1350 (**18.3%**, [16.3%, 20.4%]) | 135/1350 (**10.0%**, [8.5%, 11.7%]) | 163/1350 (**12.1%**, [10.4%, 13.9%]) | 180/1350 (**13.3%**, [11.6%, 15.3%]) | 5/1050 (**0.5%**, [0.2%, 1.1%]) |
| **`quiet`** | 25/1350 (**1.9%**, [1.3%, 2.7%]) | 99/1350 (**7.3%**, [6.1%, 8.8%]) | 59/1350 (**4.4%**, [3.4%, 5.6%]) | 81/1350 (**6.0%**, [4.9%, 7.4%]) | 110/1350 (**8.1%**, [6.8%, 9.7%]) | 1/1050 (**0.1%**, [0.0%, 0.5%]) |

## Choosing a Preset

- **`balanced` (default):** `0.5%` quality hard-block rate on routine non-swap retrains, `0.0%` benign drift/leakage/label-shift false positives, and `slices.on_inconclusive: warn`.
- **`strict`:** Best for regulated or high-stakes models wanting early warnings on new features (`drift.on_new_feature: warn`), tighter `2%` slice margins, and lower univariate AUC thresholds — expect ~53% WARN rate on routine PRs.
- **`quiet`:** Best for high-velocity teams or small evaluation sets where alert fatigue blocks adoption — sets `slices.on_inconclusive: info` and widens latency/fairness margins, cutting non-swap Quality FAIL to `0.1%` at the cost of missing smaller 3–5 pt regressions.
