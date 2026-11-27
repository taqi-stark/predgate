"""`predgate init` command and starter templates."""

from __future__ import annotations

from pathlib import Path

import typer

from predgate.cli.common import app, console


@app.command()
def init(
    path: str = typer.Argument(".", help="Directory to initialize"),
) -> None:
    """Scaffold a predgate.yaml and starter training script."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)

    config_file = target / "predgate.yaml"
    if config_file.exists():
        console.print("[yellow]predgate.yaml already exists, skipping.[/yellow]")
    else:
        config_file.write_text(_SCAFFOLD_CONFIG)
        console.print(f"[green]✓[/green] Created {config_file}")

    demo_dir = target / "demo"
    demo_dir.mkdir(exist_ok=True)

    train_file = demo_dir / "train.py"
    if not train_file.exists():
        train_file.write_text(_SCAFFOLD_TRAIN)
        console.print(f"[green]✓[/green] Created {train_file}")

    console.print("\n[bold green]PredGate project initialized![/bold green]")
    console.print("Next steps:")
    console.print("  1. Add your data to data/")
    console.print("  2. Edit predgate.yaml")
    console.print("  3. Run: predgate run")


_SCAFFOLD_CONFIG = """\
project: my-project
seed: 42
# Optional threshold preset: strict | balanced | quiet (default thresholds match balanced)
# preset: balanced

model:
  entrypoint: demo/train.py:build_model
  task: classification

data:
  train: data/train.csv
  test: data/test.csv
  reference: data/reference.csv
  target: default

champion:
  mlflow_alias: champion

checks:
  data_contract:
    enabled: true
    on_violation: fail
  drift:
    enabled: true
    method: psi
    warn_at: 0.1
    fail_at: 0.25
  slices:
    enabled: true
    metric: f1
    by: [region, age_bucket]
    min_slice_size: 100
    max_drop: 0.03
    max_ci_width: 0.10
    on_inconclusive: warn
    confidence: 0.95
    n_bootstrap: 1000
  behavioral:
    enabled: true
    invariance: []
    directional: []
  budgets:
    max_latency_ratio: 1.5
    # Optional absolute ms ceiling:
    # p95_latency_ms: 25
    max_model_mb: 50
    max_peak_mem_mb: 500
  fairness:
    enabled: true
    protected: []
    metric: demographic_parity_diff
    warn_at: 0.05
    fail_at: 0.10

verdict:
  fail_if_any_fail: true
  warn_requires_approval: true
"""

_SCAFFOLD_TRAIN = """\
\"\"\"Example model training script.\"\"\"

from sklearn.ensemble import GradientBoostingClassifier


def build_model(X_train, y_train, seed=42):
    \"\"\"Build and train a gradient boosting classifier.\"\"\"
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        random_state=seed,
    )
    model.fit(X_train, y_train)
    return model
"""
