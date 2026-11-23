"""Seed script for PredGate Docker Compose environment.

Waits for API and MLflow services to become healthy, generates demo data,
imports the persisted champion baseline into MLflow, provisions an ephemeral
runner token, and executes PredGate quality checks so demo runs appear in
the API, dashboard, and MLflow registry.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import httpx
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
    """Create or rotate an ephemeral engineer token for the seed runner (no default static token)."""
    env_token = os.getenv("PREDGATE_API_TOKEN", "").strip()
    if env_token:
        return env_token
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        _, token = create_user(session, name="docker-seed-runner", role="engineer")
        return token


def main() -> None:
    wait_for_service(API_URL, "PredGate API")
    wait_for_service(MLFLOW_URL, "MLflow Server")

    if not Path("data/train.csv").exists():
        print("\n1. Generating synthetic demo dataset...", flush=True)
        subprocess.run(["python", "demo/train.py"], check=True)
    else:
        print("\n1. Demo dataset already present in /app/data.", flush=True)

    print("\n2. Importing persisted champion baseline into MLflow...", flush=True)
    subprocess.run(
        ["predgate", "baseline", "import", "--config", "predgate.yaml", "--dir", "champion"],
        check=False,
    )

    print(
        "\n3. Provisioning ephemeral seed runner token & executing PredGate quality gate...",
        flush=True,
    )
    token = _provision_seed_token()
    run_env = {**os.environ, "API_URL": API_URL, "PREDGATE_API_TOKEN": token}
    subprocess.run(
        ["predgate", "run", "--config", "predgate.yaml", "--branch", "main"],
        env=run_env,
        check=False,
    )

    resp = httpx.get(f"{API_URL}/runs", timeout=5.0)
    runs = resp.json() if resp.status_code == 200 else []
    print(f"\n✓ Demo data seeding completed. Runs in API ({API_URL}/runs): {len(runs)}", flush=True)


if __name__ == "__main__":
    main()
