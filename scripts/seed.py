"""Seed script for PredGate Docker Compose and local demo environments.

Waits for API and MLflow services to become healthy, generates demo data,
loads or registers the champion baseline, provisions an ephemeral runner token
(never a hardcoded default admin token), and executes 12 real `predgate run`
evaluations across realistic PASS, WARN, and FAIL scenarios — including
PR #42 with a genuine slice regression on `region=North`, an inconclusive-slice
warning run, and model-family swaps — with zero fabricated metrics.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import yaml
from sqlmodel import Session, SQLModel

from predgate.api.main import engine
from predgate.auth import create_user

API_URL = os.getenv("API_URL", "http://api:8000").rstrip("/")
MLFLOW_URL = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000").rstrip("/")


def wait_for_service(url: str, name: str, max_retries: int = 45) -> bool:
    print(f"Waiting for {name} ({url})...", flush=True)
    for _ in range(1, max_retries + 1):
        try:
            resp = httpx.get(f"{url}/health", timeout=2.0)
            if resp.status_code in (200, 301, 302):
                print(f"✓ {name} is ready.", flush=True)
                return True
        except Exception:
            pass
        time.sleep(1)
    print(f"⚠️ {name} did not become ready after {max_retries}s, proceeding anyway.", flush=True)
    return False


def _provision_seed_token() -> str:
    """Create or rotate an ephemeral engineer token for the seed runner (no static default)."""
    env_token = os.getenv("PREDGATE_API_TOKEN", "").strip()
    if env_token:
        return env_token
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        _, token = create_user(session, name="docker-seed-runner", role="engineer")
        return token


def _ensure_champion_in_mlflow(base_cfg_path: Path, run_env: dict[str, str]) -> None:
    """Ensure MLflow has a working @champion model matching the current Python/sklearn runtime.

    First attempts `predgate baseline import` from the verified `champion/` folder.
    If that fails (for instance, Python/sklearn pickle ABI mismatch in a different
    container image), trains the real baseline model from `demo/train.py:build_model`
    and registers it directly as `@champion` in MLflow.
    """
    imp = subprocess.run(
        [
            "predgate",
            "baseline",
            "import",
            "--config",
            str(base_cfg_path),
            "--dir",
            "champion",
        ],
        env=run_env,
        capture_output=True,
        text=True,
        check=False,
    )
    if imp.returncode == 0:
        print("✓ Imported verified champion baseline into MLflow.", flush=True)
        return

    print(
        f"Note: baseline import returned {imp.returncode} ({imp.stderr.strip() or imp.stdout.strip()}); "
        "training & registering real demo baseline model in MLflow as @champion...",
        flush=True,
    )
    import mlflow

    from predgate.config import load_config
    from predgate.loader import import_entrypoint
    from predgate.registry import log_run_to_mlflow, promote_to_champion
    from predgate.verdict import CheckResult, Verdict

    cfg = load_config(base_cfg_path)
    build_model = import_entrypoint(cfg.model.entrypoint)
    train_df = pd.read_csv(cfg.data.train)
    X_train = train_df.drop(columns=[cfg.data.target])
    y_train = train_df[cfg.data.target]
    champ_model = build_model(X_train, y_train, seed=cfg.seed)

    active_uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip() or cfg.mlflow.tracking_uri
    mlflow.set_tracking_uri(active_uri)
    dummy_verdict = Verdict(
        status="pass",
        reasons=[],
        check_results=[CheckResult(name="baseline", status="pass", summary="Baseline champion")],
        summary="Baseline champion registration",
    )
    run_id = log_run_to_mlflow(
        cfg.project,
        dummy_verdict,
        {"git_sha": "baseline", "seed": cfg.seed},
        champ_model,
        cfg.seed,
        tracking_uri=active_uri,
        artifact_location=cfg.mlflow.artifact_location,
    )
    if run_id:
        promote_to_champion(
            cfg.project,
            run_id,
            alias=cfg.champion.mlflow_alias,
            tracking_uri=active_uri,
        )
        print(f"✓ Registered and promoted baseline run {run_id} as @champion.", flush=True)


_SCENARIO_MODELS_PY = textwrap.dedent("""\
    from __future__ import annotations

    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler


    def _make_preprocessor(X_train: pd.DataFrame, scale_num: bool = False) -> ColumnTransformer:
        cat_cols = [c for c in ["region", "age_bucket", "gender"] if c in X_train.columns]
        num_cols = [
            c
            for c in ["income", "age", "credit_score", "debt_ratio", "employment_years"]
            if c in X_train.columns
        ]
        num_tf = StandardScaler() if scale_num else "passthrough"
        return ColumnTransformer(
            transformers=[
                ("num", num_tf, num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ],
            remainder="drop",
        )


    def build_baseline(X_train: pd.DataFrame, y_train: pd.Series, seed: int = 42) -> Pipeline:
        pipe = Pipeline([
            ("preprocessor", _make_preprocessor(X_train)),
            ("classifier", GradientBoostingClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, subsample=0.8, random_state=seed)),
        ])
        pipe.fit(X_train, y_train)
        return pipe


    def build_tuned_lr(X_train: pd.DataFrame, y_train: pd.Series, seed: int = 42) -> Pipeline:
        pipe = Pipeline([
            ("preprocessor", _make_preprocessor(X_train)),
            ("classifier", GradientBoostingClassifier(n_estimators=55, max_depth=3, learning_rate=0.08, subsample=0.8, random_state=seed)),
        ])
        pipe.fit(X_train, y_train)
        return pipe


    def build_regularized_depth(X_train: pd.DataFrame, y_train: pd.Series, seed: int = 42) -> Pipeline:
        pipe = Pipeline([
            ("preprocessor", _make_preprocessor(X_train)),
            ("classifier", GradientBoostingClassifier(n_estimators=50, max_depth=3, min_samples_leaf=8, learning_rate=0.09, subsample=0.85, random_state=seed)),
        ])
        pipe.fit(X_train, y_train)
        return pipe


    def build_scaled_gbdt(X_train: pd.DataFrame, y_train: pd.Series, seed: int = 42) -> Pipeline:
        pipe = Pipeline([
            ("preprocessor", _make_preprocessor(X_train, scale_num=True)),
            ("classifier", GradientBoostingClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, subsample=0.8, random_state=seed)),
        ])
        pipe.fit(X_train, y_train)
        return pipe


    def build_subsample_tweak(X_train: pd.DataFrame, y_train: pd.Series, seed: int = 42) -> Pipeline:
        pipe = Pipeline([
            ("preprocessor", _make_preprocessor(X_train)),
            ("classifier", GradientBoostingClassifier(n_estimators=48, max_depth=3, learning_rate=0.1, subsample=0.9, random_state=seed)),
        ])
        pipe.fit(X_train, y_train)
        return pipe


    def build_random_forest(X_train: pd.DataFrame, y_train: pd.Series, seed: int = 42) -> Pipeline:
        pipe = Pipeline([
            ("preprocessor", _make_preprocessor(X_train)),
            ("classifier", RandomForestClassifier(n_estimators=60, max_depth=8, min_samples_leaf=5, random_state=seed)),
        ])
        pipe.fit(X_train, y_train)
        return pipe


    def build_logreg(X_train: pd.DataFrame, y_train: pd.Series, seed: int = 42) -> Pipeline:
        pipe = Pipeline([
            ("preprocessor", _make_preprocessor(X_train, scale_num=True)),
            ("classifier", LogisticRegression(max_iter=500, random_state=seed)),
        ])
        pipe.fit(X_train, y_train)
        return pipe


    def build_north_degraded(X_train: pd.DataFrame, y_train: pd.Series, seed: int = 42) -> Pipeline:
        \"\"\"Over-weight positive labels in region=North so North F1 drops moderately (~0.08-0.09, FAIL on slices only).\"\"\"
        pipe = Pipeline([
            ("preprocessor", _make_preprocessor(X_train)),
            ("classifier", GradientBoostingClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, subsample=0.8, random_state=seed)),
        ])
        sample_weight = np.where((X_train["region"] == "North") & (y_train == 1), 9.0, 1.0)
        pipe.fit(X_train, y_train, classifier__sample_weight=sample_weight)
        return pipe


    class NoisyWestModel:
        \"\"\"Adds mild 8% prediction noise on region=West so slice drop overlaps 0.03 (WARN).\"\"\"

        def __init__(self, base_model: Pipeline) -> None:
            self.base_model = base_model

        def predict(self, X: pd.DataFrame) -> np.ndarray:
            preds = np.asarray(self.base_model.predict(X), dtype=int).copy()
            if "region" in X.columns:
                rng = np.random.RandomState(7)
                mask = (X["region"].astype(str).values == "West") & (rng.random(len(X)) < 0.08)
                preds[mask] = 1 - preds[mask]
            return preds

        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            return np.asarray(self.base_model.predict_proba(X), dtype=float)


    def build_noisy_west(X_train: pd.DataFrame, y_train: pd.Series, seed: int = 42) -> NoisyWestModel:
        return NoisyWestModel(build_baseline(X_train, y_train, seed=seed))


    def build_id_sensitive(X_train: pd.DataFrame, y_train: pd.Series, seed: int = 42) -> Pipeline:
        cat_cols = [c for c in ["region", "age_bucket", "gender"] if c in X_train.columns]
        num_cols = [
            c
            for c in ["applicant_id", "income", "age", "credit_score", "debt_ratio", "employment_years"]
            if c in X_train.columns
        ]
        prep = ColumnTransformer(
            transformers=[
                ("num", "passthrough", num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ],
            remainder="drop",
        )
        pipe = Pipeline([
            ("preprocessor", prep),
            ("classifier", GradientBoostingClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=seed)),
        ])
        pipe.fit(X_train, y_train)
        return pipe
""")


def _write_scenario_data(tmp_dir: Path, base_train: pd.DataFrame) -> dict[str, Path]:
    """Generate real perturbed training CSVs for drift, leakage, and fairness/label-shift runs."""
    paths: dict[str, Path] = {}
    rng = np.random.RandomState(42)

    # 1. Drift / scale corruption (income divided by 1000)
    drift_df = base_train.copy()
    drift_df["income"] = (drift_df["income"] / 1000.0).clip(lower=0.01)
    p_drift = tmp_dir / "train_drift.csv"
    drift_df.to_csv(p_drift, index=False)
    paths["drift"] = p_drift

    # 2. Target leakage (new column strongly derived from target)
    leak_df = base_train.copy()
    leak_df["leaked_risk_flag"] = leak_df["default"] * 0.9 + rng.normal(0.0, 0.1, len(leak_df))
    p_leak = tmp_dir / "train_leakage.csv"
    leak_df.to_csv(p_leak, index=False)
    paths["leakage"] = p_leak

    # 3. Fairness / subgroup label shift (flip 35% of non-default female labels to 1)
    fair_df = base_train.copy()
    fem_neg = fair_df[(fair_df["gender"] == "female") & (fair_df["default"] == 0)]
    flip_idx = fem_neg.sample(frac=0.35, random_state=42).index
    fair_df.loc[flip_idx, "default"] = 1
    p_fair = tmp_dir / "train_fairness.csv"
    fair_df.to_csv(p_fair, index=False)
    paths["fairness"] = p_fair

    # 4. Moderate feature drift (+18% income shift -> PSI ~0.17, genuine drift WARN)
    mild_df = base_train.copy()
    mild_df["income"] = mild_df["income"] * 1.18
    p_mild = tmp_dir / "train_mild_drift.csv"
    mild_df.to_csv(p_mild, index=False)
    paths["mild_drift"] = p_mild

    # 5. Mildly correlated proxy feature (|r| ~0.55, genuine leakage WARN under defaults)
    mild_leak_df = base_train.copy()
    mild_leak_df["credit_review_hint"] = mild_leak_df["default"] * 0.5 + rng.normal(
        0.0, 0.22, len(mild_leak_df)
    )
    p_mild_leak = tmp_dir / "train_mild_leakage.csv"
    mild_leak_df.to_csv(p_mild_leak, index=False)
    paths["mild_leakage"] = p_mild_leak

    return paths


def main() -> None:
    wait_for_service(API_URL, "PredGate API")
    wait_for_service(MLFLOW_URL, "MLflow Server")

    if not Path("data/train.csv").exists():
        print("\n1. Generating synthetic demo dataset...", flush=True)
        subprocess.run([sys.executable, "demo/train.py"], check=True)
    else:
        print("\n1. Demo dataset already present in data/.", flush=True)

    print("\n2. Provisioning ephemeral seed runner token (no default static token)...", flush=True)
    token = _provision_seed_token()
    run_env = {**os.environ, "API_URL": API_URL, "PREDGATE_API_TOKEN": token}

    base_cfg_path = Path("predgate.yaml")
    print("\n3. Ensuring champion baseline is registered in MLflow...", flush=True)
    _ensure_champion_in_mlflow(base_cfg_path, run_env)

    base_cfg = yaml.safe_load(base_cfg_path.read_text())
    base_train = pd.read_csv(base_cfg["data"]["train"])

    with tempfile.TemporaryDirectory(prefix="predgate_seed_") as tmp:
        tmp_dir = Path(tmp)
        models_mod = tmp_dir / "seed_models.py"
        models_mod.write_text(_SCENARIO_MODELS_PY)
        perturbed_data = _write_scenario_data(tmp_dir, base_train)

        scenarios: list[dict[str, object]] = [
            {
                "branch": "main",
                "pr": None,
                "entrypoint": f"{models_mod}:build_baseline",
                "train": None,
                "overrides": {},
                "desc": "Baseline champion re-verification (PASS)",
            },
            {
                "branch": "feature/tune-learning-rate",
                "pr": 35,
                "entrypoint": f"{models_mod}:build_tuned_lr",
                "train": None,
                "overrides": {},
                "desc": "Harmless LR=0.08 / n_estimators=55 tweak (PASS)",
            },
            {
                "branch": "feature/regularize-leaves",
                "pr": 36,
                "entrypoint": f"{models_mod}:build_regularized_depth",
                "train": None,
                "overrides": {},
                "desc": "Min-samples-leaf regularization (PASS)",
            },
            {
                "branch": "feature/standard-scaler",
                "pr": 37,
                "entrypoint": f"{models_mod}:build_scaled_gbdt",
                "train": None,
                "overrides": {},
                "desc": "Add StandardScaler to numeric pipeline (PASS)",
            },
            {
                "branch": "feature/credit-review-hint",
                "pr": 38,
                "entrypoint": f"{models_mod}:build_baseline",
                "train": str(perturbed_data["mild_leakage"]),
                "overrides": {},
                "desc": "Moderately correlated proxy feature (WARN leakage)",
            },
            {
                "branch": "feature/logreg-challenger",
                "pr": 39,
                "entrypoint": f"{models_mod}:build_logreg",
                "train": None,
                "overrides": {},
                "desc": "Model-family swap GBDT -> LogisticRegression",
            },
            {
                "branch": "feature/west-threshold-tweak",
                "pr": 40,
                "entrypoint": f"{models_mod}:build_noisy_west",
                "train": None,
                "overrides": {},
                "desc": "Mild region=West point drop overlapping threshold (WARN slices)",
            },
            {
                "branch": "feature/income-refresh-q4",
                "pr": 41,
                "entrypoint": f"{models_mod}:build_baseline",
                "train": str(perturbed_data["mild_drift"]),
                "overrides": {},
                "desc": "Moderate +18% income distribution shift (WARN drift)",
            },
            {
                "branch": "bad/degrade-region-north",
                "pr": 42,
                "entrypoint": f"{models_mod}:build_north_degraded",
                "train": None,
                "overrides": {},
                "desc": "PR #42 — real slice regression on region=North (FAIL)",
            },
            {
                "branch": "feature/rf-model-swap",
                "pr": 43,
                "entrypoint": f"{models_mod}:build_random_forest",
                "train": None,
                "overrides": {},
                "desc": "Model-family swap GBDT -> RandomForestClassifier",
            },
            {
                "branch": "bad/income-unit-bug",
                "pr": 44,
                "entrypoint": f"{models_mod}:build_baseline",
                "train": str(perturbed_data["drift"]),
                "overrides": {},
                "desc": "Silent income unit scaling bug (FAIL drift)",
            },
            {
                "branch": "bad/target-leakage",
                "pr": 45,
                "entrypoint": f"{models_mod}:build_baseline",
                "train": str(perturbed_data["leakage"]),
                "overrides": {},
                "desc": "Target leakage column in train.csv (FAIL leakage)",
            },
        ]

        print(f"\n4. Executing {len(scenarios)} real `predgate run` evaluations...", flush=True)
        for idx, sc in enumerate(scenarios, 1):
            cfg_copy = yaml.safe_load(yaml.safe_dump(base_cfg))
            cfg_copy["model"]["entrypoint"] = sc["entrypoint"]
            if sc["train"]:
                cfg_copy["data"]["train"] = sc["train"]
            overrides = sc["overrides"]
            if isinstance(overrides, dict):
                for chk_name, fields in overrides.items():
                    cfg_copy.setdefault("checks", {}).setdefault(chk_name, {}).update(fields)

            sc_cfg_path = tmp_dir / f"predgate_sc_{idx}.yaml"
            sc_cfg_path.write_text(yaml.safe_dump(cfg_copy))
            out_dir = tmp_dir / f"out_{idx}"

            cmd = [
                "predgate",
                "run",
                "--config",
                str(sc_cfg_path),
                "--branch",
                str(sc["branch"]),
                "--output",
                str(out_dir),
            ]
            if sc["pr"] is not None:
                cmd.extend(["--pr", str(sc["pr"])])

            res = subprocess.run(cmd, env=run_env, capture_output=True, text=True, check=False)
            verdict_line = ""
            for line in res.stdout.splitlines():
                if line.strip().startswith("Verdict:"):
                    verdict_line = line.strip()
                    break
            print(
                f"  [{idx:2d}/{len(scenarios)}] {sc['branch']} "
                f"(PR #{sc['pr'] if sc['pr'] else '-'}) -> {verdict_line or f'exit={res.returncode}'} | {sc['desc']}",
                flush=True,
            )

    resp = httpx.get(f"{API_URL}/runs", timeout=10.0)
    runs = resp.json() if resp.status_code == 200 else []
    counts = {st: sum(1 for r in runs if r.get("status") == st) for st in ("pass", "warn", "fail")}
    print(
        f"\n✓ Demo data seeding completed. Runs in API ({API_URL}/runs): {len(runs)} "
        f"(PASS={counts['pass']}, WARN={counts['warn']}, FAIL={counts['fail']})",
        flush=True,
    )


if __name__ == "__main__":
    main()
