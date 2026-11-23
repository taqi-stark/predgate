"""Unit tests for PredGate FastAPI endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel

from predgate.api.main import app, engine
from predgate.auth import create_user


@pytest.fixture(name="client")
def client_fixture() -> TestClient:
    SQLModel.metadata.drop_all(engine)
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        _, token = create_user(session, name="test_admin", role="admin")
    with TestClient(app) as client:
        client.headers["Authorization"] = f"Bearer {token}"
        yield client


def test_health_check(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_create_and_get_run(client: TestClient) -> None:
    payload = {
        "project": "test-project",
        "branch": "feat/test",
        "git_sha": "abc1234",
        "seed": 42,
        "verdict": {
            "status": "pass",
            "summary": "All checks passed",
            "reasons": [],
            "check_results": [
                {
                    "name": "data_contract",
                    "status": "pass",
                    "summary": "0 violations",
                    "metrics": {"violations": 0},
                    "details": {},
                    "duration_s": 0.5,
                }
            ],
        },
        "metrics": {"accuracy": 0.95},
        "lineage": {"git_sha": "abc1234", "seed": 42},
    }

    create_resp = client.post("/runs", json=payload)
    assert create_resp.status_code == 200
    created = create_resp.json()
    assert created["status"] == "pass"
    assert "id" in created

    run_id = created["id"]
    get_resp = client.get(f"/runs/{run_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == run_id
    assert get_resp.json()["project"] == "test-project"

    # List runs
    list_resp = client.get("/runs?project=test-project")
    assert list_resp.status_code == 200
    assert len(list_resp.json()) == 1


def test_run_approval_flow(client: TestClient) -> None:
    # Create a WARN run
    payload = {
        "project": "test-project",
        "branch": "feat/warn",
        "git_sha": "warn123",
        "seed": 42,
        "verdict": {
            "status": "warn",
            "summary": "1 warning",
            "reasons": [{"check_name": "drift", "severity": "warn", "message": "minor drift"}],
            "check_results": [],
        },
    }
    run = client.post("/runs", json=payload).json()
    run_id = run["id"]

    # Approve with comment
    approve_resp = client.post(
        f"/runs/{run_id}/approve",
        json={
            "reviewer": "reviewer1",
            "decision": "approve",
            "comment": "Approved after reviewing slice metrics",
        },
    )
    assert approve_resp.status_code == 200
    assert approve_resp.json()["decision"] == "approve"


def test_settings_endpoints(client: TestClient) -> None:
    # Get settings
    get_resp = client.get("/settings")
    assert get_resp.status_code == 200

    # Update settings
    update_resp = client.put(
        "/settings", json={"config": {"checks": {"budgets": {"p95_latency_ms": 75.0}}}}
    )
    assert update_resp.status_code == 200
    assert update_resp.json()["status"] == "saved"

    # Second update modifies existing settings record
    update_resp2 = client.put(
        "/settings", json={"config": {"checks": {"budgets": {"p95_latency_ms": 50.0}}}}
    )
    assert update_resp2.status_code == 200
    get_resp2 = client.get("/settings")
    assert get_resp2.json()["config"]["checks"]["budgets"]["p95_latency_ms"] == 50.0


def test_slices_compare_models_bench_and_users_endpoints(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PREDGATE_ADMIN_TOKEN", "env-boot-token-123")
    # Create two runs (one with slices check, one fail run)
    run1 = client.post(
        "/runs",
        json={
            "project": "p1",
            "branch": "main",
            "verdict": {
                "status": "pass",
                "summary": "ok",
                "check_results": [
                    {
                        "name": "slices",
                        "status": "pass",
                        "summary": "ok",
                        "metrics": {"f1": 0.90},
                        "details": {"slices": [{"slice": "region=North"}], "worst_5": []},
                    }
                ],
            },
        },
    ).json()
    run2 = client.post(
        "/runs",
        json={
            "project": "p1",
            "branch": "feat",
            "verdict": {
                "status": "fail",
                "summary": "bad",
                "check_results": [
                    {
                        "name": "slices",
                        "status": "fail",
                        "summary": "drop",
                        "metrics": {"f1": 0.80},
                        "details": {"slices": [], "worst_5": []},
                    }
                ],
            },
        },
    ).json()

    # Filters on list_runs
    assert len(client.get("/runs?status=pass&branch=main&project=p1").json()) == 1
    assert client.get("/runs/99999").status_code == 404

    # Slices endpoint
    s_resp = client.get(f"/runs/{run1['id']}/slices")
    assert s_resp.status_code == 200
    assert len(s_resp.json()["slices"]) == 1
    assert client.get("/runs/99999/slices").status_code == 404

    # Compare endpoint
    c_resp = client.get(f"/compare?a={run1['id']}&b={run2['id']}")
    assert c_resp.status_code == 200
    assert abs(c_resp.json()["metrics"][0]["delta"] - (-0.10)) < 1e-6
    assert client.get("/compare?a=1&b=99999").status_code == 404

    # Override comment too short (<20 chars) -> 400
    short_override = client.post(
        f"/runs/{run2['id']}/approve",
        json={"decision": "approve", "comment": "too short"},
    )
    assert short_override.status_code == 400
    assert (
        client.post(
            "/runs/99999/approve", json={"decision": "approve", "comment": "ok"}
        ).status_code
        == 404
    )

    # Promote FAIL run without approval -> 400; promote PASS run -> 200; promote 404
    assert client.post(f"/runs/{run2['id']}/promote").status_code == 400
    assert client.post(f"/runs/{run1['id']}/promote").status_code == 200
    assert client.post("/runs/99999/promote").status_code == 404

    # Models & bench endpoints
    models_resp = client.get("/models?project=p1")
    assert models_resp.status_code == 200
    assert len(models_resp.json()) == 1
    assert client.get("/bench").status_code == 200

    # Users endpoints
    u_create = client.post("/users", json={"name": "new_rev", "role": "reviewer"})
    assert u_create.status_code == 200
    assert "token" in u_create.json()
    # Update existing user role
    u_update = client.post("/users", json={"name": "new_rev", "role": "admin"})
    assert u_update.status_code == 200
    assert client.post("/users", json={"name": "x", "role": "bad_role"}).status_code == 400
    u_list = client.get("/users")
    assert u_list.status_code == 200
    assert any(u["name"] == "new_rev" for u in u_list.json())
