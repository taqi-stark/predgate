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


def test_model_family_changed_note_in_reports() -> None:
    """PR comment, HTML report, and Verdict JSON must surface 'Model family changed' when algorithms differ."""
    import numpy as np
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import (
        AdaBoostClassifier,
        BaggingClassifier,
        GradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from predgate.loader import detect_model_family, detect_model_family_change

    X = np.array([[0.0], [0.1], [0.9], [1.0], [0.2], [0.8]] * 4)
    y = np.array([0, 0, 1, 1, 0, 1] * 4)

    champ_pipe = Pipeline([("scale", StandardScaler()), ("clf", GradientBoostingClassifier())])
    cand_rf_pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=5, random_state=0)),
        ]
    ).fit(X, y)
    cand_same_pipe = Pipeline([("clf", GradientBoostingClassifier())])

    # Ensembles must NOT unwrap into their template base learner (DecisionTreeClassifier)
    assert detect_model_family(cand_rf_pipe)["name"] == "RandomForestClassifier"
    assert (
        detect_model_family(AdaBoostClassifier(n_estimators=3, random_state=0).fit(X, y))["name"]
        == "AdaBoostClassifier"
    )
    assert (
        detect_model_family(BaggingClassifier(n_estimators=3, random_state=0).fit(X, y))["name"]
        == "BaggingClassifier"
    )
    # Single-model wrappers unwrap cleanly to underlying estimator
    assert (
        detect_model_family(CalibratedClassifierCV(LogisticRegression(), cv=2).fit(X, y))["name"]
        == "LogisticRegression"
    )
    assert (
        detect_model_family(GridSearchCV(LogisticRegression(), {"C": [1.0]}, cv=2).fit(X, y))[
            "name"
        ]
        == "LogisticRegression"
    )

    # Different algorithms -> changed=True and prominent report banners
    mfc = detect_model_family_change(champ_pipe, cand_rf_pipe)
    assert mfc is not None
    assert mfc["changed"] is True
    assert mfc["champion_family"] == "GradientBoostingClassifier"
    assert mfc["candidate_family"] == "RandomForestClassifier"
    assert "Model family changed" in mfc["note"]

    results = [CheckResult(name="drift", status="pass", summary="Max PSI 0.01")]
    verdict = compute_verdict(results, model_family_change=mfc)
    assert verdict.to_dict()["model_family_change"]["changed"] is True

    pr_md = build_pr_comment(verdict, candidate_sha="abc1234")
    html = build_html_report(verdict, project="demo")
    assert "Model family changed" in pr_md
    assert "GradientBoostingClassifier" in pr_md
    assert "RandomForestClassifier" in pr_md
    assert "Model family changed" in html
    assert "GradientBoostingClassifier" in html
    assert "RandomForestClassifier" in html

    # Same algorithm (even with different preprocessing) -> changed=False, no banner
    mfc_same = detect_model_family_change(champ_pipe, cand_same_pipe)
    assert mfc_same is not None
    assert mfc_same["changed"] is False
    verdict_same = compute_verdict(results, model_family_change=mfc_same)
    assert "Model family changed" not in build_pr_comment(verdict_same)
    assert "Model family changed" not in build_html_report(verdict_same)
