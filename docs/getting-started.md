# Getting Started

This guide walks through installing PredGate, scaffolding a new project, locking your first champion baseline, and running the included examples.

## Prerequisites

- Python 3.10+ (`git`, `python3-pip`, and `python3-venv` on Ubuntu/Debian)

## 1. Install PredGate

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install predgate
```

## 2. Scaffold a Project and Run Your First Gate

Create a project directory, scaffold `predgate.yaml` and a starter training script, generate a small synthetic dataset, and run the gate:

```bash
predgate init my-gate
cd my-gate
python3 -c "import os, numpy as np, pandas as pd; os.makedirs('data', exist_ok=True); r = np.random.RandomState(42); df = pd.DataFrame({'x': r.randn(600), 'region': r.randint(0, 2, 600), 'age_bucket': r.randint(0, 2, 600), 'default': r.randint(0, 2, 600)}); df[:300].to_csv('data/train.csv', index=False); df[300:450].to_csv('data/test.csv', index=False); df[450:].to_csv('data/reference.csv', index=False)"
predgate doctor --config predgate.yaml
predgate run --config predgate.yaml
```

On the very first run, no champion exists yet, so comparison checks (`slices`, champion latency ratio, relative fairness) are skipped while data/drift/leakage/label-shift checks execute.

## 3. Export a Champion Baseline and Compare Candidates

Lock the current model as your verified champion baseline (`champion/model.pkl`, `metadata.json`, `metrics.json`, and `champion.dvc`), then re-run the gate to execute champion-vs-candidate comparisons:

```bash
predgate baseline export --config predgate.yaml --dir champion
predgate baseline verify --dir champion --dvc champion.dvc
predgate run --config predgate.yaml
```

Inspect the generated artifacts in `predgate_output/`:

- `verdict.json` — machine-readable verdict, metrics, inconclusive slices, and model-family metadata
- `pr_comment.md` — GitHub-flavored Markdown PR comment
- `report.html` — standalone HTML report
- `payload.json` — API ingestion payload

## 4. Try the Included Credit-Risk Demo (From Source)

Clone the repository to run the full 8-check credit-risk demo with Pandera schema validation, behavioral tests, fairness checks, and a pre-exported champion baseline:

```bash
git clone https://github.com/taqi-stark/predgate.git
cd predgate
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

python examples/credit-risk-demo/train.py
predgate doctor --config examples/credit-risk-demo/predgate.yaml
predgate run --config examples/credit-risk-demo/predgate.yaml
```

You can also try a minimal self-contained scikit-learn Diabetes example in `examples/minimal/`.

## Next Steps

- Understand the statistical guarantees behind [the 8 quality checks](concepts/checks.md).
- Pick a threshold profile in [Config Presets](guides/presets.md) (`strict`, `balanced`, or `quiet`).
- Wire PredGate into pull requests with the [GitHub Actions guide](guides/github.md).
- Explore runs interactively with [Docker Compose](guides/docker.md) and the [Dashboard](guides/dashboard.md).
