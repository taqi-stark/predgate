"""Unit tests for PredGate report builder."""

from __future__ import annotations

from predgate.lineage import LineageInfo
from predgate.report import build_html_report, build_json_payload, build_pr_comment
from predgate.verdict import CheckResult, compute_verdict


def test_build_pr_comment() -> None:
    results = [
        CheckResult(
            name="data_contract", status="pass", summary="0 violations", metrics={"violations": 0}
        ),
        CheckResult(
            name="drift", status="warn", summary="income PSI 0.14", metrics={"psi_max": 0.14}
        ),
        CheckResult(
            name="leakage",
            status="pass",
            summary="Max |r| 0.12, max single-feature AUC 0.64",
            metrics={"max_abs_corr": 0.12},
        ),
        CheckResult(
            name="slices",
            status="fail",
            summary="region=North F1 dropped 0.07",
            metrics={"worst_drop": 0.07},
        ),
    ]
    verdict = compute_verdict(results)
    comment = build_pr_comment(verdict, candidate_sha="a1b2c3d", seed=42, data_hash="9f3a123")

    assert "PredGate verdict: FAIL" in comment
    assert "Candidate" in comment
    assert "Data Contract" in comment
    assert "Drift" in comment
    assert "Slices" in comment
    assert r"Max \|r\| 0.12, max single-feature AUC 0.64" in comment


def test_build_html_report() -> None:
    results = [
        CheckResult(
            name="data_contract", status="pass", summary="0 violations", metrics={"violations": 0}
        ),
    ]
    verdict = compute_verdict(results)
    lineage = LineageInfo(git_sha="1234567", seed=42)
    html = build_html_report(verdict, lineage, project="test-proj")

    assert "<!DOCTYPE html>" in html
    assert "PredGate Report" in html
    assert "test-proj" in html


def test_build_json_payload() -> None:
    results = [
        CheckResult(
            name="data_contract", status="pass", summary="0 violations", metrics={"violations": 0}
        ),
    ]
    verdict = compute_verdict(results)
    lineage = LineageInfo(git_sha="1234567", seed=42)
    payload = build_json_payload(verdict, lineage, project="test-proj", branch="main")

    assert payload["project"] == "test-proj"
    assert payload["branch"] == "main"
    assert payload["verdict"]["status"] == "pass"
