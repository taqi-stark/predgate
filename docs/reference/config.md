# Configuration Reference (`predgate.yaml`)

PredGate loads and validates `predgate.yaml` with Pydantic v2 (`src/predgate/config.py`). Any explicit check field overrides an optional `preset:` (`strict`, `balanced`, or `quiet`).

```yaml
project: credit-risk-demo
seed: 42
# Optional threshold preset: strict | balanced | quiet
# Omit or use "balanced" for unchanged defaults.
# preset: balanced

model:
  entrypoint: examples/credit-risk-demo/train.py:build_model
  task: classification

data:
  train: examples/credit-risk-demo/data/train.csv
  test: examples/credit-risk-demo/data/test.csv
  reference: examples/credit-risk-demo/data/reference.csv
  target: default
  schema: examples/credit-risk-demo/contracts/schema.py:schema

champion:
  mlflow_alias: champion
  baseline_dir: examples/credit-risk-demo/champion

mlflow:
  tracking_uri: sqlite:///mlflow.db
  artifact_location: ./mlruns

checks:
  data_contract:
    enabled: true
    on_violation: fail              # fail | warn

  drift:
    enabled: true
    method: psi                     # psi | ks
    warn_at: 0.10
    fail_at: 0.25
    on_new_feature: info            # info | warn
    point_mass_warn_delta: 0.10
    point_mass_fail_delta: 0.20

  leakage:
    enabled: true
    warn_corr: 0.50
    fail_corr: 0.75
    warn_auc: 0.85
    fail_auc: 0.92

  label_shift:
    enabled: true
    by: [region, gender, age_bucket]
    min_slice_size: 30
    warn_delta: 0.05
    fail_delta: 0.10
    alpha: 0.01

  slices:
    enabled: true
    metric: f1                      # f1 | accuracy | precision | recall | roc_auc | log_loss | brier | rmse | mae
    by: [region, age_bucket, gender]
    min_slice_size: 100
    max_drop: 0.03
    max_ci_width: 0.10
    on_inconclusive: warn           # warn | info
    confidence: 0.95
    n_bootstrap: 1000

  behavioral:
    enabled: true
    invariance:
      - feature: applicant_id
        perturb: random_id          # random_id | shuffle | swap_category
        max_flip_rate: 0.0
    directional:
      - feature: income
        direction: increase         # increase | decrease
        expect: prob_not_increase   # prob_not_increase | prob_not_decrease

  budgets:
    enabled: true
    max_latency_ratio: 1.5
    latency_samples: 20
    latency_batch_size: 5
    latency_confidence: 0.95
    latency_n_bootstrap: 1000
    # p95_latency_ms: 100.0         # optional absolute ms ceiling
    max_peak_mem_mb: 500.0
    max_model_mb: 50.0

  fairness:
    enabled: true
    protected: [gender]
    metric: demographic_parity_diff # demographic_parity_diff | equal_opportunity_diff | equalized_odds_diff
    enforce_absolute: false
    warn_at: 0.05
    fail_at: 0.10
    max_gap_increase: 0.02
    fail_gap_increase: 0.05
    selection_rate_shift_warn: 0.02
    selection_rate_n_bootstrap: 1000
    min_group_size: 100
    min_group_positives: 30
    confidence: 0.95
    n_bootstrap: 200

verdict:
  fail_if_any_fail: true
  warn_requires_approval: true
```

## Environment Variable Overrides

| Variable | Effect |
|---|---|
| `MLFLOW_TRACKING_URI` | Overrides `mlflow.tracking_uri` |
| `API_URL` | PredGate API URL for `predgate run` posting (default `http://localhost:8000`) |
| `PREDGATE_API_TOKEN` | Bearer token used by `predgate run` when posting to the API |
| `PREDGATE_ADMIN_TOKEN` | Bootstraps an initial `admin` API user on server startup |
| `PREDGATE_CORS_ORIGINS` | Comma-separated allowed CORS origins for the FastAPI server |
| `DATABASE_URL` | SQLAlchemy URL for the API database (default `sqlite:///predgate.db`) |
| `CHAMPION_PINNED_SHA256` | Pinned SHA-256 hash verified before unpickling `champion/model.pkl` |
