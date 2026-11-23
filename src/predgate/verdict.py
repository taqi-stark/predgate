"""PredGate verdict engine.

Merges all CheckResults into a final verdict: PASS, WARN, or FAIL.
Produces a ranked reasons list and machine-readable JSON output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from predgate.config import VerdictConfig


@dataclass
class CheckResult:
    """Result of a single quality check.

    Every check module produces one of these.
    """

    name: str
    status: Literal["pass", "warn", "fail", "skip"]
    summary: str
    metrics: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0


@dataclass
class Reason:
    """A human-readable reason for a non-PASS verdict."""

    check_name: str
    severity: Literal["warn", "fail"]
    message: str


@dataclass
class Verdict:
    """The final verdict from the PredGate run."""

    status: Literal["pass", "warn", "fail"]
    reasons: list[Reason]
    check_results: list[CheckResult]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the verdict to a dictionary."""
        return {
            "status": self.status,
            "summary": self.summary,
            "reasons": [
                {
                    "check_name": r.check_name,
                    "severity": r.severity,
                    "message": r.message,
                }
                for r in self.reasons
            ],
            "check_results": [
                {
                    "name": cr.name,
                    "status": cr.status,
                    "summary": cr.summary,
                    "metrics": cr.metrics,
                    "details": cr.details,
                    "duration_s": cr.duration_s,
                }
                for cr in self.check_results
            ],
        }


# Severity ordering for sorting reasons (most severe first)
_SEVERITY_ORDER: dict[str, int] = {"fail": 0, "warn": 1}


def compute_verdict(
    results: list[CheckResult],
    config: VerdictConfig | None = None,
) -> Verdict:
    """Merge all check results into a final verdict.

    Rules:
    - FAIL if any check fails (when fail_if_any_fail is True, which is default).
    - WARN if any check warns (and none fail).
    - PASS otherwise.

    Reasons are ranked by severity (FAIL first, then WARN), then alphabetically
    by check name for deterministic output.

    Args:
        results: List of CheckResult from all checks.
        config: Optional VerdictConfig for verdict rules.

    Returns:
        A Verdict with overall status and ranked reasons.
    """
    if config is None:
        config = VerdictConfig()

    reasons: list[Reason] = []
    has_fail = False
    has_warn = False

    for cr in results:
        if cr.status == "fail":
            has_fail = True
            reasons.append(
                Reason(
                    check_name=cr.name,
                    severity="fail",
                    message=cr.summary,
                )
            )
        elif cr.status == "warn":
            has_warn = True
            reasons.append(
                Reason(
                    check_name=cr.name,
                    severity="warn",
                    message=cr.summary,
                )
            )

    # Sort reasons: FAIL first, then WARN, then alphabetically by check name
    reasons.sort(key=lambda r: (_SEVERITY_ORDER.get(r.severity, 99), r.check_name))

    # Determine overall status
    if has_fail and config.fail_if_any_fail:
        overall_status: Literal["pass", "warn", "fail"] = "fail"
    elif has_warn:
        overall_status = "warn"
    else:
        overall_status = "pass"

    # Build summary
    total = len(results)
    pass_count = sum(1 for r in results if r.status == "pass")
    warn_count = sum(1 for r in results if r.status == "warn")
    fail_count = sum(1 for r in results if r.status == "fail")
    skip_count = sum(1 for r in results if r.status == "skip")

    summary_parts = [f"{total} checks run"]
    if pass_count:
        summary_parts.append(f"{pass_count} passed")
    if warn_count:
        summary_parts.append(f"{warn_count} warnings")
    if fail_count:
        summary_parts.append(f"{fail_count} failed")
    if skip_count:
        summary_parts.append(f"{skip_count} skipped")

    summary = ", ".join(summary_parts)

    return Verdict(
        status=overall_status,
        reasons=reasons,
        check_results=results,
        summary=summary,
    )
