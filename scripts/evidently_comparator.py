"""Off-the-shelf Evidently default-configuration comparator worker.

Runs inside an isolated Python environment with `evidently==0.7.23` and `pandas<3`
(since Evidently 0.7.x depends on `pandas.core.dtypes.common.infer_dtype_from_object`,
which was removed in pandas 3.0).

Documented configuration (100% Evidently 0.7.23 defaults, zero threshold tuning):
- Training-data drift: `Report([DataDriftPreset()], include_tests=True)` on
  `(current=candidate_train[common_cols], reference=champion_train[common_cols])`.
- Evaluation classification: `Report([ClassificationPreset()], include_tests=True)` on
  `(current=test_with_candidate_preds, reference=test_with_champion_preds)` with
  `BinaryClassification(target=target_col, prediction_labels="__pred__")`.
- Decision rule: flagged (`failed=True`) if ANY default Evidently test returns status `FAIL`.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from typing import Any

warnings.filterwarnings("ignore")

import evidently  # noqa: E402
import pandas as pd  # noqa: E402
from evidently import BinaryClassification, DataDefinition, Dataset, Report  # noqa: E402
from evidently.presets import ClassificationPreset, DataDriftPreset  # noqa: E402


def evaluate_pair(req: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a single (train_ref, train_cur, test_ref, test_cur) trial with Evidently defaults."""
    target = str(req["target"])
    pred_col = str(req.get("pred_col", "__pred__"))
    failed_tests: list[str] = []
    drift_failed = False
    clf_failed = False

    train_ref_path = req.get("train_ref")
    train_cur_path = req.get("train_cur")
    if train_ref_path and train_cur_path:
        tr_ref = pd.read_csv(train_ref_path)
        tr_cur = pd.read_csv(train_cur_path)
        common_cols = [c for c in tr_ref.columns if c in tr_cur.columns]
        if common_cols:
            drift_report = Report([DataDriftPreset()], include_tests=True)
            drift_snap = drift_report.run(
                Dataset.from_pandas(tr_cur[common_cols]),
                Dataset.from_pandas(tr_ref[common_cols]),
            )
            for t in drift_snap.dict().get("tests", []):
                status = str(t.get("status", "")).upper()
                if status.endswith("FAIL"):
                    drift_failed = True
                    failed_tests.append(f"drift:{t.get('name', 'unknown')}")

    test_ref_path = req.get("test_ref")
    test_cur_path = req.get("test_cur")
    if test_ref_path and test_cur_path:
        te_ref = pd.read_csv(test_ref_path)
        te_cur = pd.read_csv(test_cur_path)
        keep_cols = [c for c in te_ref.columns if c in te_cur.columns]
        te_ref = te_ref[keep_cols]
        te_cur = te_cur[keep_cols]
        dd = DataDefinition(
            classification=[BinaryClassification(target=target, prediction_labels=pred_col)]
        )
        clf_report = Report([ClassificationPreset()], include_tests=True)
        clf_snap = clf_report.run(
            Dataset.from_pandas(te_cur, data_definition=dd),
            Dataset.from_pandas(te_ref, data_definition=dd),
        )
        for t in clf_snap.dict().get("tests", []):
            status = str(t.get("status", "")).upper()
            if status.endswith("FAIL"):
                clf_failed = True
                failed_tests.append(f"clf:{t.get('name', 'unknown')}")

    return {
        "failed": bool(drift_failed or clf_failed),
        "drift_failed": drift_failed,
        "clf_failed": clf_failed,
        "n_failed_tests": len(failed_tests),
        "failed_tests": failed_tests[:10],
        "evidently_version": str(evidently.__version__),
        "pandas_version": str(pd.__version__),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evidently default comparator worker.")
    parser.add_argument("--worker", action="store_true", help="Run JSONL stdin/stdout worker loop")
    parser.add_argument("--version", action="store_true", help="Print version metadata JSON")
    args = parser.parse_args()

    if args.version:
        print(
            json.dumps(
                {
                    "evidently_version": str(evidently.__version__),
                    "pandas_version": str(pd.__version__),
                }
            )
        )
        return

    if args.worker:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                res = evaluate_pair(req)
            except Exception as exc:
                res = {
                    "failed": False,
                    "drift_failed": False,
                    "clf_failed": False,
                    "n_failed_tests": 0,
                    "failed_tests": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            sys.stdout.write(json.dumps(res) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
