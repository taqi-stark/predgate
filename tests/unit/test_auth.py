"""Unit tests for authentication, role-based access control (RBAC), and append-only audit logging."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel

from predgate.api.main import app, engine
from predgate.auth import create_user


@pytest.fixture(name="auth_env")
def auth_env_fixture() -> dict[str, object]:
    """Reset test database and provision engineer, reviewer, and admin users."""
    SQLModel.metadata.drop_all(engine)
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        _, eng_token = create_user(session, name="alice_eng", role="engineer")
        _, rev_token = create_user(session, name="bob_rev", role="reviewer")
        _, adm_token = create_user(session, name="carol_adm", role="admin")

    with TestClient(app) as client:
        yield {
            "client": client,
            "engineer": eng_token,
            "reviewer": rev_token,
            "admin": adm_token,
        }


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_unauthenticated_and_invalid_token_rejected_401(auth_env: dict[str, object]) -> None:
    client: TestClient = auth_env["client"]  # type: ignore[assignment]
    adm_token: str = auth_env["admin"]  # type: ignore[assignment]

    # Missing token -> 401
    r1 = client.post("/runs", json={"project": "p", "verdict": {"status": "pass"}})
    assert r1.status_code == 401

    # Bad token -> 401
    r2 = client.put(
        "/settings",
        json={"config": {"a": 1}},
        headers=_headers("not-a-valid-token"),
    )
    assert r2.status_code == 401

    # Both denied attempts must be recorded in audit log
    audit_resp = client.get("/audit", headers=_headers(adm_token))
    assert audit_resp.status_code == 200
    rows = audit_resp.json()
    denied = [r for r in rows if r["outcome"] == "denied"]
    assert len(denied) >= 2
    assert any(r["action"] == "run.create" and r["actor"] == "anonymous" for r in denied)
    assert any(r["action"] == "settings.update" and r["actor"] == "anonymous" for r in denied)


def test_role_enforcement_on_approve_override_promote_and_settings(
    auth_env: dict[str, object],
) -> None:
    client: TestClient = auth_env["client"]  # type: ignore[assignment]
    eng_token: str = auth_env["engineer"]  # type: ignore[assignment]
    rev_token: str = auth_env["reviewer"]  # type: ignore[assignment]
    adm_token: str = auth_env["admin"]  # type: ignore[assignment]

    # Engineer can create a WARN run and a FAIL run
    warn_run = client.post(
        "/runs",
        json={"project": "p", "verdict": {"status": "warn", "summary": "warn"}},
        headers=_headers(eng_token),
    ).json()
    fail_run = client.post(
        "/runs",
        json={"project": "p", "verdict": {"status": "fail", "summary": "fail"}},
        headers=_headers(eng_token),
    ).json()

    # 1. Engineer cannot approve a run -> 403
    r_eng_app = client.post(
        f"/runs/{warn_run['id']}/approve",
        json={"decision": "approve", "comment": "looks good"},
        headers=_headers(eng_token),
    )
    assert r_eng_app.status_code == 403

    # 2. Reviewer can approve a WARN run, and reviewer identity comes from token (bob_rev), not body
    r_rev_app = client.post(
        f"/runs/{warn_run['id']}/approve",
        json={"reviewer": "spoofed_name", "decision": "approve", "comment": "Reviewed slice CIs"},
        headers=_headers(rev_token),
    )
    assert r_rev_app.status_code == 200
    assert r_rev_app.json()["reviewer"] == "bob_rev"

    run_detail = client.get(f"/runs/{warn_run['id']}").json()
    assert run_detail["approvals"][0]["reviewer"] == "bob_rev"

    # 3. Reviewer cannot override a FAIL run -> 403
    r_rev_override = client.post(
        f"/runs/{fail_run['id']}/approve",
        json={
            "decision": "approve",
            "comment": "Override justification that is more than twenty characters",
        },
        headers=_headers(rev_token),
    )
    assert r_rev_override.status_code == 403

    # 4. Admin can override a FAIL run with >= 20 char justification -> 200
    r_adm_override = client.post(
        f"/runs/{fail_run['id']}/approve",
        json={
            "decision": "approve",
            "comment": "Emergency hotfix approved by risk committee ticket #402",
        },
        headers=_headers(adm_token),
    )
    assert r_adm_override.status_code == 200
    assert r_adm_override.json()["is_override"] is True
    assert r_adm_override.json()["reviewer"] == "carol_adm"

    # 5. Reviewer cannot promote a run -> 403; Admin can promote -> 200
    r_rev_prom = client.post(f"/runs/{fail_run['id']}/promote", headers=_headers(rev_token))
    assert r_rev_prom.status_code == 403

    r_adm_prom = client.post(f"/runs/{fail_run['id']}/promote", headers=_headers(adm_token))
    assert r_adm_prom.status_code == 200
    assert r_adm_prom.json()["promoted"] is True

    # 6. Engineer and Reviewer cannot update settings -> 403; Admin can -> 200
    assert (
        client.put("/settings", json={"config": {"x": 1}}, headers=_headers(eng_token)).status_code
        == 403
    )
    assert (
        client.put("/settings", json={"config": {"x": 1}}, headers=_headers(rev_token)).status_code
        == 403
    )
    assert (
        client.put("/settings", json={"config": {"x": 1}}, headers=_headers(adm_token)).status_code
        == 200
    )

    # 7. Non-admin cannot read /audit -> 403; Admin can -> 200
    assert client.get("/audit", headers=_headers(rev_token)).status_code == 403
    audit_rows = client.get("/audit", headers=_headers(adm_token)).json()
    actions_outcomes = [(r["actor"], r["action"], r["outcome"]) for r in audit_rows]
    assert ("alice_eng", "run.approve", "denied") in actions_outcomes
    assert ("bob_rev", "run.approve", "allowed") in actions_outcomes
    assert ("bob_rev", "run.override", "denied") in actions_outcomes
    assert ("carol_adm", "run.override", "allowed") in actions_outcomes
    assert ("bob_rev", "run.promote", "denied") in actions_outcomes
    assert ("carol_adm", "run.promote", "allowed") in actions_outcomes
    assert ("carol_adm", "settings.update", "allowed") in actions_outcomes
