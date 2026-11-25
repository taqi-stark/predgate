"""PredGate FastAPI application.

Provides REST API for run ingestion, querying, comparison, approval,
model promotion, role-based access control, and append-only audit logging.
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Body, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlmodel import Session, SQLModel, col, create_engine, select

from predgate import __version__
from predgate.api.models import Approval, AuditLog, CheckResultRecord, Run, Settings, User
from predgate.auth import (
    VALID_ROLES,
    authenticate_and_authorize,
    bootstrap_admin_from_env,
    create_user,
    record_audit,
)

UTC = timezone.utc

# Database setup (configurable via DATABASE_URL environment variable)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///predgate.db")
engine = create_engine(DATABASE_URL, echo=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create database tables and bootstrap optional admin user on startup."""
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        bootstrap_admin_from_env(session)
    yield


app = FastAPI(
    title="PredGate API",
    description="CI/CD quality gate for ML models",
    version=__version__,
    lifespan=lifespan,
)

# CORS for dashboard
_cors_raw = (
    os.getenv("PREDGATE_CORS_ORIGINS")
    or os.getenv("MODELGATE_CORS_ORIGINS")
    or "http://localhost:3000,http://localhost:5173"
)
_cors_origins = [origin.strip() for origin in _cors_raw.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept"],
)


# ──── Request/Response models ────


class RunCreateRequest(BaseModel):
    project: str = ""
    branch: str = ""
    pr_number: int | None = None
    timestamp: str = ""
    git_sha: str = ""
    seed: int = 42
    verdict: dict = {}
    lineage: dict = {}
    metrics: dict = {}
    run_id: int | str | None = None


class ApprovalRequest(BaseModel):
    reviewer: str = ""
    decision: str  # "approve" or "reject"
    comment: str


class SettingsUpdateRequest(BaseModel):
    config: dict


class UserCreateRequest(BaseModel):
    name: str
    role: str


# ──── Routes ────


@app.post("/runs")
def create_run(
    req: RunCreateRequest,
    authorization: str | None = Header(default=None),
) -> dict:
    """Ingest a run result (requires engineer, reviewer, or admin role)."""
    verdict = req.verdict or {}
    metrics: dict[str, float] = {}

    for cr in verdict.get("check_results", []):
        for k, v in cr.get("metrics", {}).items():
            metrics[f"{cr['name']}_{k}"] = v

    with Session(engine) as session:
        user = authenticate_and_authorize(
            session,
            authorization,
            allowed_roles=("engineer", "reviewer", "admin"),
            action="run.create",
            target=f"project:{req.project or 'default'}",
        )

        run = Run(
            project=req.project,
            branch=req.branch,
            pr_number=req.pr_number,
            git_sha=req.lineage.get("git_sha", req.git_sha) if req.lineage else req.git_sha,
            seed=req.lineage.get("seed", req.seed) if req.lineage else req.seed,
            status=verdict.get("status", ""),
            summary=verdict.get("summary", ""),
            verdict_json=json.dumps(verdict),
            lineage_json=json.dumps(req.lineage or {}),
            metrics_json=json.dumps(metrics),
        )
        session.add(run)
        session.commit()
        session.refresh(run)
        run_id = int(run.id or 0)

        for cr in verdict.get("check_results", []):
            record = CheckResultRecord(
                run_id=run_id,
                name=cr.get("name", ""),
                status=cr.get("status", ""),
                summary=cr.get("summary", ""),
                metrics_json=json.dumps(cr.get("metrics", {})),
                details_json=json.dumps(cr.get("details", {})),
                duration_s=cr.get("duration_s", 0.0),
            )
            session.add(record)
        session.commit()

        record_audit(
            session,
            actor=user.name,
            role=user.role,
            action="run.create",
            target=f"run:{run_id}",
            outcome="allowed",
            detail={"project": req.project, "status": verdict.get("status", "")},
        )

    return {"id": run_id, "status": verdict.get("status", "")}


@app.get("/runs")
def list_runs(
    status: str | None = Query(None),
    branch: str | None = Query(None),
    project: str | None = Query(None),
    limit: int = Query(50, le=200),
    offset: int = Query(0),
) -> list[dict]:
    """List runs with optional filters."""
    with Session(engine) as session:
        query = select(Run).order_by(col(Run.id).desc())

        if status:
            query = query.where(Run.status == status)
        if branch:
            query = query.where(Run.branch == branch)
        if project:
            query = query.where(Run.project == project)

        query = query.offset(offset).limit(limit)
        runs = session.exec(query).all()

        return [
            {
                "id": r.id,
                "project": r.project,
                "branch": r.branch,
                "pr_number": r.pr_number,
                "git_sha": r.git_sha,
                "status": r.status,
                "summary": r.summary,
                "created_at": r.created_at,
                "metrics": r.metrics,
            }
            for r in runs
        ]


@app.get("/runs/{run_id}")
def get_run(run_id: int) -> dict:
    """Get full run detail with all check results."""
    with Session(engine) as session:
        run = session.get(Run, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

        checks = session.exec(
            select(CheckResultRecord).where(CheckResultRecord.run_id == run_id)
        ).all()
        approvals = session.exec(select(Approval).where(Approval.run_id == run_id)).all()

        return {
            "id": run.id,
            "project": run.project,
            "branch": run.branch,
            "pr_number": run.pr_number,
            "git_sha": run.git_sha,
            "seed": run.seed,
            "status": run.status,
            "summary": run.summary,
            "created_at": run.created_at,
            "verdict": run.verdict,
            "lineage": run.lineage,
            "metrics": run.metrics,
            "check_results": [
                {
                    "name": cr.name,
                    "status": cr.status,
                    "summary": cr.summary,
                    "metrics": json.loads(cr.metrics_json),
                    "details": json.loads(cr.details_json),
                    "duration_s": cr.duration_s,
                }
                for cr in checks
            ],
            "approvals": [
                {
                    "reviewer": a.reviewer,
                    "decision": a.decision,
                    "comment": a.comment,
                    "is_override": a.is_override,
                    "created_at": a.created_at,
                }
                for a in approvals
            ],
        }


@app.get("/runs/{run_id}/slices")
def get_run_slices(run_id: int) -> dict:
    """Get slice data for the explorer."""
    with Session(engine) as session:
        checks = session.exec(
            select(CheckResultRecord)
            .where(CheckResultRecord.run_id == run_id)
            .where(CheckResultRecord.name == "slices")
        ).all()

        if not checks:
            raise HTTPException(status_code=404, detail="No slice data found")

        details = json.loads(checks[0].details_json)
        return {
            "run_id": run_id,
            "slices": details.get("slices", []),
            "worst_5": details.get("worst_5", []),
            "inconclusive_slices": details.get("inconclusive_slices", []),
            "inconclusive_entries": details.get("inconclusive_entries", []),
            "on_inconclusive": details.get("on_inconclusive", "warn"),
        }


@app.get("/compare")
def compare_runs(
    a: int = Query(..., description="First run ID"),
    b: int = Query(..., description="Second run ID"),
) -> dict:
    """Side-by-side comparison of two runs."""
    with Session(engine) as session:
        run_a = session.get(Run, a)
        run_b = session.get(Run, b)

        if not run_a or not run_b:
            raise HTTPException(status_code=404, detail="Run not found")

        metrics_a = run_a.metrics
        metrics_b = run_b.metrics

        all_keys = sorted(set(list(metrics_a.keys()) + list(metrics_b.keys())))
        comparison = []
        for key in all_keys:
            val_a = metrics_a.get(key)
            val_b = metrics_b.get(key)
            delta = None
            if val_a is not None and val_b is not None:
                try:
                    delta = float(val_b) - float(val_a)
                except (ValueError, TypeError):
                    pass
            comparison.append(
                {
                    "name": key,
                    "value_a": val_a,
                    "value_b": val_b,
                    "delta": delta,
                }
            )

        return {
            "run_a": {"id": a, "status": run_a.status, "branch": run_a.branch},
            "run_b": {"id": b, "status": run_b.status, "branch": run_b.branch},
            "metrics": comparison,
        }


@app.post("/runs/{run_id}/approval")
@app.post("/runs/{run_id}/approve")
def approve_run(
    run_id: int,
    req: ApprovalRequest,
    authorization: str | None = Header(default=None),
) -> dict:
    """Approve or reject a run (requires reviewer or admin; FAIL override requires admin)."""
    with Session(engine) as session:
        user = authenticate_and_authorize(
            session,
            authorization,
            allowed_roles=("reviewer", "admin"),
            action="run.approve",
            target=f"run:{run_id}",
        )

        run = session.get(Run, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

        is_override = run.status == "fail" and req.decision == "approve"
        audit_action = "run.override" if is_override else "run.approve"

        if is_override and user.role != "admin":
            record_audit(
                session,
                actor=user.name,
                role=user.role,
                action="run.override",
                target=f"run:{run_id}",
                outcome="denied",
                detail={"status_code": 403, "reason": "override_requires_admin"},
            )
            raise HTTPException(
                status_code=403,
                detail="Overriding a FAIL run requires 'admin' role",
            )

        if is_override and len(req.comment) < 20:
            record_audit(
                session,
                actor=user.name,
                role=user.role,
                action="run.override",
                target=f"run:{run_id}",
                outcome="denied",
                detail={"status_code": 400, "reason": "comment_too_short"},
            )
            raise HTTPException(
                status_code=400,
                detail="Override approval requires a justification of at least 20 characters",
            )

        approval = Approval(
            run_id=run_id,
            reviewer=user.name,
            decision=req.decision,
            comment=req.comment,
            is_override=is_override,
        )

        session.add(approval)
        session.commit()
        session.refresh(approval)

        record_audit(
            session,
            actor=user.name,
            role=user.role,
            action=audit_action,
            target=f"run:{run_id}",
            outcome="allowed",
            detail={
                "decision": req.decision,
                "is_override": is_override,
                "approval_id": approval.id,
            },
        )

        return {
            "id": approval.id,
            "run_id": run_id,
            "reviewer": user.name,
            "decision": req.decision,
            "is_override": is_override,
        }


@app.post("/runs/{run_id}/promote")
def promote_run(
    run_id: int,
    authorization: str | None = Header(default=None),
) -> dict:
    """Promote a run's model to champion (requires admin role)."""
    with Session(engine) as session:
        user = authenticate_and_authorize(
            session,
            authorization,
            allowed_roles=("admin",),
            action="run.promote",
            target=f"run:{run_id}",
        )

        run = session.get(Run, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

        if run.status == "fail":
            approvals = session.exec(
                select(Approval)
                .where(Approval.run_id == run_id)
                .where(Approval.decision == "approve")
            ).all()
            if not approvals:
                record_audit(
                    session,
                    actor=user.name,
                    role=user.role,
                    action="run.promote",
                    target=f"run:{run_id}",
                    outcome="denied",
                    detail={"status_code": 400, "reason": "missing_override_approval"},
                )
                raise HTTPException(
                    status_code=400,
                    detail="Cannot promote a FAIL run without override approval",
                )

        record_audit(
            session,
            actor=user.name,
            role=user.role,
            action="run.promote",
            target=f"run:{run_id}",
            outcome="allowed",
            detail={"run_status": run.status},
        )

        return {
            "run_id": run_id,
            "promoted": True,
            "promoted_by": user.name,
            "message": f"Model from run {run_id} promoted to champion",
        }


@app.get("/models")
def list_models(
    project: str | None = Query(None),
) -> list[dict]:
    """Registry view with champion marked."""
    with Session(engine) as session:
        query = (
            select(Run).where(col(Run.status).in_(["pass", "warn"])).order_by(col(Run.id).desc())
        )
        if project:
            query = query.where(Run.project == project)
        runs = session.exec(query.limit(50)).all()

        return [
            {
                "version": f"v{r.id}",
                "run_id": r.id,
                "status": r.status,
                "created_at": r.created_at,
                "branch": r.branch,
                "git_sha": r.git_sha,
                "is_champion": i == 0,
            }
            for i, r in enumerate(runs)
        ]


@app.get("/settings")
def get_settings(project: str = Query("default")) -> dict:
    """Get current settings."""
    with Session(engine) as session:
        settings = session.exec(select(Settings).where(Settings.project == project)).first()

        if not settings:
            return {"project": project, "config": {}}

        return {
            "project": settings.project,
            "config": json.loads(settings.config_json),
            "updated_at": settings.updated_at,
        }


@app.put("/settings")
def update_settings(
    project: str = Query("default"),
    req: SettingsUpdateRequest = Body(...),
    authorization: str | None = Header(default=None),
) -> dict:
    """Update settings (requires admin role)."""
    with Session(engine) as session:
        user = authenticate_and_authorize(
            session,
            authorization,
            allowed_roles=("admin",),
            action="settings.update",
            target=f"project:{project}",
        )

        settings = session.exec(select(Settings).where(Settings.project == project)).first()

        if settings:
            settings.config_json = json.dumps(req.config)
            settings.updated_at = datetime.now(UTC).isoformat()
        else:
            settings = Settings(
                project=project,
                config_json=json.dumps(req.config),
            )
            session.add(settings)

        session.commit()

        record_audit(
            session,
            actor=user.name,
            role=user.role,
            action="settings.update",
            target=f"project:{project}",
            outcome="allowed",
            detail={"keys": sorted(req.config.keys())},
        )

        return {"project": project, "status": "saved", "updated_by": user.name}


@app.get("/audit")
def list_audit_logs(
    limit: int = Query(100, le=500),
    offset: int = Query(0),
    authorization: str | None = Header(default=None),
) -> list[dict]:
    """List append-only audit log entries (requires admin role)."""
    with Session(engine) as session:
        authenticate_and_authorize(
            session,
            authorization,
            allowed_roles=("admin",),
            action="audit.read",
            target="audit_log",
        )
        rows = session.exec(
            select(AuditLog).order_by(col(AuditLog.id).desc()).offset(offset).limit(limit)
        ).all()
        return [
            {
                "id": r.id,
                "timestamp": r.timestamp,
                "actor": r.actor,
                "role": r.role,
                "action": r.action,
                "target": r.target,
                "outcome": r.outcome,
                "detail": r.detail,
            }
            for r in rows
        ]


@app.post("/users")
def create_api_user(
    req: UserCreateRequest,
    authorization: str | None = Header(default=None),
) -> dict:
    """Create an API user with role in (engineer, reviewer, admin) (requires admin role)."""
    with Session(engine) as session:
        admin = authenticate_and_authorize(
            session,
            authorization,
            allowed_roles=("admin",),
            action="user.create",
            target=f"user:{req.name}",
        )
        if req.role not in VALID_ROLES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid role '{req.role}'. Must be one of {list(VALID_ROLES)}",
            )
        user, token = create_user(session, name=req.name, role=req.role)
        record_audit(
            session,
            actor=admin.name,
            role=admin.role,
            action="user.create",
            target=f"user:{user.name}",
            outcome="allowed",
            detail={"role": user.role},
        )
        return {
            "id": user.id,
            "name": user.name,
            "role": user.role,
            "token": token,
            "created_at": user.created_at,
        }


@app.get("/users")
def list_api_users(
    authorization: str | None = Header(default=None),
) -> list[dict]:
    """List API users without exposing token hashes (requires admin role)."""
    with Session(engine) as session:
        authenticate_and_authorize(
            session,
            authorization,
            allowed_roles=("admin",),
            action="user.list",
            target="users",
        )
        users = session.exec(select(User).order_by(col(User.id).asc())).all()
        return [
            {
                "id": u.id,
                "name": u.name,
                "role": u.role,
                "created_at": u.created_at,
            }
            for u in users
        ]


@app.get("/bench")
def get_benchmark() -> dict:
    """Get latest benchmark results."""
    results_path = Path("docs/benchmark_results.json")
    if results_path.exists():
        return json.loads(results_path.read_text())
    return {"message": "No benchmark results found. Run 'predgate bench' first."}


@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "service": "predgate-api"}
