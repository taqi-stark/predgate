# Credit Risk Demo

Full end-to-end PredGate demo with Pandera schema validation, drift, leakage, label-shift, subgroup slice regression (`region`, `age_bucket`, `gender`), behavioral invariance/directional tests, latency/size budgets, fairness checks, and a SHA-256 verified champion baseline.

## Run Locally

```bash
# 1. Generate deterministic train / test / reference datasets (5,000 rows, seed 42)
python examples/credit-risk-demo/train.py

# 2. Run the PredGate quality gate against the persisted champion baseline
predgate run --config examples/credit-risk-demo/predgate.yaml
```
