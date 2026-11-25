"""Measure wall-clock runtime and memory of `predgate run` for 1k, 10k, and 100k test rows.

Runs the full 8-check PredGate suite with default settings (identical to `predgate.yaml`
defaults: n_bootstrap=1000 for slices, 200 for fairness, 1000 for selection-rate shift,
20x5 interleaved latency batches) on synthetic credit datasets with 3,000 train/reference rows
and 1,000 / 10,000 / 100,000 test rows. Prints JSON and Markdown summary rows.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


def measure_scale(n_test: int, repo_root: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix=f"predgate_perf_{n_test}_") as tmp:
        tmp_dir = Path(tmp)
        data_dir = tmp_dir / "data"
        data_dir.mkdir()

        # Generate train (3,000), reference (1,000), and test (n_test)
        sys.path.insert(0, str(repo_root))
        from demo.train import generate_dataset

        base_df = generate_dataset(n_samples=4000, seed=42)
        base_df.iloc[:3000].to_csv(data_dir / "train.csv", index=False)
        base_df.iloc[3000:4000].to_csv(data_dir / "reference.csv", index=False)
        test_df = generate_dataset(n_samples=n_test, seed=99)
        test_df.to_csv(data_dir / "test.csv", index=False)

        # Config with default check settings (default SliceConfig n_bootstrap=1000, etc.)
        cfg = {
            "project": f"perf-{n_test}",
            "seed": 42,
            "model": {
                "entrypoint": str(repo_root / "demo" / "train.py") + ":build_model",
                "task": "classification",
            },
            "data": {
                "train": str(data_dir / "train.csv"),
                "test": str(data_dir / "test.csv"),
                "reference": str(data_dir / "reference.csv"),
                "target": "default",
                "schema": str(repo_root / "demo" / "contracts" / "schema.py") + ":schema",
            },
            "champion": {
                "mlflow_alias": "champion",
                "baseline_dir": str(repo_root / "champion"),
            },
            "mlflow": {
                "tracking_uri": f"sqlite:///{tmp_dir / 'mlflow.db'}",
                "artifact_location": str(tmp_dir / "mlruns"),
            },
            "checks": {
                "data_contract": {"enabled": True},
                "drift": {"enabled": True},
                "leakage": {"enabled": True},
                "label_shift": {"enabled": True, "by": ["region", "gender", "age_bucket"]},
                "slices": {"enabled": True, "by": ["region", "age_bucket", "gender"]},
                "behavioral": {
                    "enabled": True,
                    "invariance": [
                        {"feature": "applicant_id", "perturb": "random_id", "max_flip_rate": 0.0}
                    ],
                    "directional": [
                        {
                            "feature": "income",
                            "direction": "increase",
                            "expect": "prob_not_decrease",
                        }
                    ],
                },
                "budgets": {"enabled": True},
                "fairness": {"enabled": True, "protected": ["gender"]},
            },
        }
        cfg_path = tmp_dir / "predgate.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))
        out_dir = tmp_dir / "out"

        # Measure in a fresh subprocess so peak RSS (ru_maxrss) is isolated per scale
        measure_cmd = [
            sys.executable,
            "-c",
            (
                "import json, resource, subprocess, sys, time; "
                "t0 = time.perf_counter(); "
                "p = subprocess.run([sys.executable, '-m', 'predgate.cli', 'run', "
                f"'--config', {str(cfg_path)!r}, '--output', {str(out_dir)!r}], "
                "capture_output=True, text=True); "
                "elapsed = time.perf_counter() - t0; "
                "ru = resource.getrusage(resource.RUSAGE_CHILDREN); "
                "print(json.dumps({'exit_code': p.returncode, 'wall_s': round(elapsed, 2), "
                "'max_rss_mb': round(ru.ru_maxrss / 1024.0, 1)}))"
            ),
        ]
        res = subprocess.run(measure_cmd, capture_output=True, text=True, check=True)
        # Parse last JSON line
        metrics = json.loads(res.stdout.strip().splitlines()[-1])
        verdict = json.loads((out_dir / "verdict.json").read_text())
        per_check_s = {
            cr["name"]: round(float(cr["duration_s"]), 2) for cr in verdict["check_results"]
        }
        return {
            "n_test_rows": n_test,
            "wall_s": metrics["wall_s"],
            "max_rss_mb": metrics["max_rss_mb"],
            "verdict": verdict["status"],
            "per_check_s": per_check_s,
        }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    rows = []
    for n_test in (1_000, 10_000, 100_000):
        print(f"Measuring predgate run with {n_test:,} test rows (default settings)...", flush=True)
        r = measure_scale(n_test, repo_root)
        rows.append(r)
        print(f"  -> {json.dumps(r)}", flush=True)
    print("\nFINAL_JSON:")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
