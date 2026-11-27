"""Authentication, role-based access control (RBAC), and append-only audit logging."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from typing import Any

from fastapi import HTTPException
from sqlmodel import Session, select

from predgate.api.models import AuditLog, User

VALID_ROLES: tuple[str, ...] = ("engineer", "reviewer", "admin")


def hash_token(token: str) -> str:
    """Return SHA-256 hex digest of a bearer token (plaintext tokens are never stored)."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def generate_token() -> str:
    """Generate a cryptographically random URL-safe bearer token."""
    return secrets.token_urlsafe(32)


def record_audit(
    session: Session,
    *,
    actor: str,
    role: str,
    action: str,
    target: str,
    outcome: str,
    detail: dict[str, Any] | None = None,
) -> AuditLog:
    """Append an immutable audit log record to the database."""
    entry = AuditLog(
        actor=actor,
        role=role,
        action=action,
        target=target,
        outcome=outcome,
        detail_json=json.dumps(detail or {}),
    )
    session.add(entry)
    session.commit()
    session.refresh(entry)
    return entry


def create_user(
    session: Session,
    name: str,
    role: str,
    raw_token: str | None = None,
) -> tuple[User, str]:
    """Create or update a user with the given role and return (User, plaintext_token)."""
    if role not in VALID_ROLES:
        raise ValueError(f"Invalid role '{role}'. Must be one of {VALID_ROLES}.")

    token = raw_token or generate_token()
    t_hash = hash_token(token)

    existing = session.exec(select(User).where(User.name == name)).first()
    if existing:
        existing.role = role
        existing.token_hash = t_hash
        session.add(existing)
        session.commit()
        session.refresh(existing)
        return existing, token

    user = User(name=name, role=role, token_hash=t_hash)
    session.add(user)
    session.commit()
    session.refresh(user)
    return user, token


def bootstrap_admin_from_env(session: Session) -> None:
    """If PREDGATE_ADMIN_TOKEN is set, ensure an admin user exists."""
    admin_token = os.getenv("PREDGATE_ADMIN_TOKEN", "").strip()
    if not admin_token:
        return
    t_hash = hash_token(admin_token)
    existing_by_hash = session.exec(select(User).where(User.token_hash == t_hash)).first()
    if existing_by_hash:
        return
    create_user(session, name="admin", role="admin", raw_token=admin_token)


def _extract_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    parts = authorization.strip().split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    token = parts[1].strip()
    return token or None


def authenticate_and_authorize(
    session: Session,
    authorization: str | None,
    allowed_roles: tuple[str, ...],
    action: str,
    target: str,
) -> User:
    """Verify bearer token and check role membership, logging denied attempts to AuditLog."""
    bootstrap_admin_from_env(session)
    token = _extract_bearer_token(authorization)
    if not token:
        record_audit(
            session,
            actor="anonymous",
            role="",
            action=action,
            target=target,
            outcome="denied",
            detail={"status_code": 401, "reason": "missing_bearer_token"},
        )
        raise HTTPException(
            status_code=401, detail="Missing or malformed Authorization Bearer token"
        )

    t_hash = hash_token(token)
    user = session.exec(select(User).where(User.token_hash == t_hash)).first()
    if not user:
        record_audit(
            session,
            actor="anonymous",
            role="",
            action=action,
            target=target,
            outcome="denied",
            detail={"status_code": 401, "reason": "invalid_token"},
        )
        raise HTTPException(status_code=401, detail="Invalid bearer token")

    if user.role not in allowed_roles:
        record_audit(
            session,
            actor=user.name,
            role=user.role,
            action=action,
            target=target,
            outcome="denied",
            detail={"status_code": 403, "required_roles": list(allowed_roles)},
        )
        raise HTTPException(
            status_code=403,
            detail=f"Role '{user.role}' is not permitted to perform '{action}'; requires {list(allowed_roles)}",
        )

    return user
