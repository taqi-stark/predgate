"""PredGate API database models.

SQLModel tables for runs, check_results, approvals, and settings.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlmodel import Field, SQLModel

UTC = timezone.utc


class Run(SQLModel, table=True):
    """A PredGate run record."""

    id: int | None = Field(default=None, primary_key=True)
    project: str = ""
    branch: str = ""
    pr_number: int | None = None
    git_sha: str = ""
    seed: int = 42
    status: str = ""  # pass, warn, fail
    summary: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    # JSON-encoded fields
    verdict_json: str = "{}"
    lineage_json: str = "{}"
    metrics_json: str = "{}"

    @property
    def verdict(self) -> dict:
        return json.loads(self.verdict_json) if self.verdict_json else {}

    @property
    def lineage(self) -> dict:
        return json.loads(self.lineage_json) if self.lineage_json else {}

    @property
    def metrics(self) -> dict:
        return json.loads(self.metrics_json) if self.metrics_json else {}


class CheckResultRecord(SQLModel, table=True):
    """A check result associated with a run."""

    __tablename__ = "check_results"

    id: int | None = Field(default=None, primary_key=True)
    run_id: int = Field(foreign_key="run.id")
    name: str = ""
    status: str = ""
    summary: str = ""
    metrics_json: str = "{}"
    details_json: str = "{}"
    duration_s: float = 0.0


class Approval(SQLModel, table=True):
    """An approval or rejection record for a run."""

    id: int | None = Field(default=None, primary_key=True)
    run_id: int = Field(foreign_key="run.id")
    reviewer: str = ""
    decision: str = ""  # "approve" or "reject"
    comment: str = ""
    is_override: bool = False
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class Settings(SQLModel, table=True):
    """Project settings (thresholds, budgets, checks)."""

    id: int | None = Field(default=None, primary_key=True)
    project: str = Field(unique=True, default="")
    config_json: str = "{}"
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class User(SQLModel, table=True):
    """Authenticated API user with role-based permissions (engineer, reviewer, admin)."""

    __tablename__ = "users"

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(unique=True, index=True, default="")
    role: str = ""  # "engineer", "reviewer", or "admin"
    token_hash: str = Field(unique=True, index=True, default="")
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class AuditLog(SQLModel, table=True):
    """Append-only audit log of allowed and denied API actions."""

    __tablename__ = "audit_log"

    id: int | None = Field(default=None, primary_key=True)
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    actor: str = "anonymous"
    role: str = ""
    action: str = ""
    target: str = ""
    outcome: str = ""  # "allowed" or "denied"
    detail_json: str = "{}"

    @property
    def detail(self) -> dict:
        return json.loads(self.detail_json) if self.detail_json else {}
