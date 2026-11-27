"""Markdown benchmark report generator backed by docs/benchmark_results.json."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from predgate.bench.render_sections import (
    fmt_rate_ci,
    render_benign_section,
    render_presets_section,
    render_synthetic_section,
)


def _render_fault_validation_section(held: dict[str, Any]) -> list[str]:
    h_seeds: list[int] = list(held.get("seeds", []))
    h_range = f"{h_seeds[0]}..{h_seeds[-1]}" if h_seeds else str(held.get("seed_range", "unknown"))
    h_has_ev = any("evidently_detected" in r for r in held["rows"])
    lines = [
        "",
        f"## 3. Full Fault Validation Across All Datasets (Seeds `{h_range}`, Frozen Defaults)",
        "",
        "Evaluated with **frozen thresholds** (zero tuning) across all datasets (**UCI Adult**, **German Credit**, **Synthetic Credit**, **Bank Marketing** `data_id=1461`, **COMPAS** `data_id=42192`) covering all fault families including `silent_null_imputation` (40% null-to-zero corruption of primary numeric feature in training data) and `cohort_feature_corruption` (cohort-localized 2σ feature corruption at inference time with clean training data).",
        "",
    ]
    if h_has_ev:
        lines.append(
            "| Dataset | Fault | PredGate FAIL-only (95% CI) | PredGate Flagged WARN+FAIL (95% CI) | Overall Acc (`\\|Δacc\\|>1%`) | Per-Group Acc Drop (`>2%`) | Fixed DP (`>=0.10`) | Evidently 0.7 Defaults | Power / MDE Note |"
        )
        lines.append(
            "|---------|-------|-----------------------------|-------------------------------------|------------------------------|----------------------------|---------------------|------------------------|------------------|"
        )
    else:
        lines.append(
            "| Dataset | Fault | PredGate FAIL-only (95% CI) | PredGate Flagged WARN+FAIL (95% CI) | Overall Acc (`\\|Δacc\\|>1%`) | Per-Group Acc Drop (`>2%`) | Fixed DP (`>=0.10`) | Power / MDE Note |"
        )
        lines.append(
            "|---------|-------|-----------------------------|-------------------------------------|------------------------------|----------------------------|---------------------|------------------|"
        )

    for r in held["rows"]:
        n_t = int(r["n_trials"])
        suf = "FP" if r["fault"] == "noop" else ""
        fail_cell = fmt_rate_ci(
            int(r["detected_fail"]), n_t, float(r["fail_rate"]), r["fail_ci_95"], suf
        )
        flag_cell = fmt_rate_ci(
            int(r["detected_total"]), n_t, float(r["detection_rate"]), r["ci_95"], suf
        )
        if r["fault"] == "noop":
            fail_cell = f"**{fail_cell}**"
            flag_cell = f"**{flag_cell}**"
        mde_str = (
            f"`min_n={r['mean_min_n']:.0f}, MDE_DP={r['mean_mde_dp']:.3f}` "
            f"(note {int(r['power_note_count'])}/{n_t})"
        )
        cells = [
            f"**{r['dataset']}**",
            r["fault_label"],
            fail_cell,
            flag_cell,
            fmt_rate_ci(
                int(r["baseline_detected"]),
                n_t,
                float(r["baseline_rate"]),
                r["baseline_ci_95"],
                suf,
            ),
            fmt_rate_ci(
                int(r["per_group_acc_detected"]),
                n_t,
                float(r["per_group_acc_rate"]),
                r["per_group_acc_ci_95"],
                suf,
            ),
            fmt_rate_ci(
                int(r["fixed_dp_detected"]),
                n_t,
                float(r["fixed_dp_rate"]),
                r["fixed_dp_ci_95"],
                suf,
            ),
        ]
        if h_has_ev:
            cells.append(
                fmt_rate_ci(
                    int(r["evidently_detected"]),
                    n_t,
                    float(r["evidently_rate"]),
                    r["evidently_ci_95"],
                    suf,
                )
            )
        cells.append(mde_str)
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _render_fairness_section(pub: dict[str, Any], commit_hash: str) -> list[str]:
    from predgate.bench.runner import wilson_ci

    pub_seeds: list[int] = list(pub.get("seeds", []))
    pub_range = (
        f"{pub_seeds[0]}..{pub_seeds[-1]}" if pub_seeds else str(pub.get("seed_range", "unknown"))
    )
    pub_cmd = str(pub.get("command", "python scripts/bench_public.py"))
    pub_commit = str(pub.get("commit_hash", commit_hash))
    frozen = pub.get("frozen_settings", {})
    frozen_str = ", ".join(f"`{k}={v}`" for k, v in frozen.items())
    pub_has_stronger = any("per_group_acc_detected" in r for r in pub["rows"])

    lines = [
        "",
        f"## 4. Standalone Fairness Check Across All Datasets (Seeds `{pub_range}`, Frozen Defaults)",
        "",
        f"- **Command**: `{pub_cmd}`",
        f"- **Evaluated Code Commit Hash**: `{pub_commit}`",
        f"- **Frozen `FairnessConfig` Settings**: {frozen_str}",
        "",
    ]
    if pub_has_stronger:
        lines.append(
            f"| Dataset (`{pub_range}`) | Fault | Fairness FAIL-only (95% CI) | Fairness Flagged WARN+FAIL (95% CI) | Overall Acc (`\\|Δacc\\|>1%`) | Per-Group Acc Drop (`>2%`) | Fixed DP (`>=0.10`) | Mean DP (`champ -> cand`) | Power / MDE Note |"
        )
        lines.append(
            "|------------------------|-------|-----------------------------|-------------------------------------|------------------------------|----------------------------|---------------------|---------------------------|------------------|"
        )
    else:
        lines.append(
            f"| Dataset (`{pub_range}`) | Fault | Fairness FAIL-only (95% CI) | Fairness Flagged WARN+FAIL (95% CI) | Overall Acc (`\\|Δacc\\|>1%`) | Mean DP (`champ -> cand`) | Power / MDE Note |"
        )
        lines.append(
            "|------------------------|-------|-----------------------------|-------------------------------------|------------------------------|---------------------------|------------------|"
        )

    for row in pub["rows"]:
        ds = row["dataset"]
        flt = row["fault_label"]
        n_t = int(row["n_trials"])
        suf = "FP" if row["fault"] == "noop" else ""
        det_fail = int(row["detected_fail"])
        fail_rate = float(row.get("fail_rate", det_fail / n_t if n_t else 0.0))
        fail_ci = row.get("fail_ci_95", list(wilson_ci(det_fail, n_t)))
        det = int(row["detected_total"])
        rate = float(row["detection_rate"])
        ci = row["ci_95"]
        bl_det = int(row["baseline_detected"])
        bl_rate = float(row["baseline_rate"])
        bl_ci = row["baseline_ci_95"]
        dp_str = f"`{row['mean_champ_dp']:.3f} -> {row['mean_cand_dp']:.3f}`"
        note_cnt = int(row["power_note_count"])
        mde_str = (
            f"`min_n={row['mean_min_n']:.0f}, MDE_DP={row['mean_mde_dp']:.3f}, "
            f"MDE_EO={row['mean_mde_eo']:.3f}` (note {note_cnt}/{n_t})"
        )
        fail_cell = fmt_rate_ci(det_fail, n_t, fail_rate, fail_ci, suf)
        flag_cell = fmt_rate_ci(det, n_t, rate, ci, suf)
        bl_cell = fmt_rate_ci(bl_det, n_t, bl_rate, bl_ci, suf)
        if row["fault"] == "noop":
            fail_cell = f"**{fail_cell}**"
            flag_cell = f"**{flag_cell}**"
            bl_cell = f"**{bl_cell}**"

        cells = [f"**{ds}**", flt, fail_cell, flag_cell, bl_cell]
        if pub_has_stronger:
            cells.append(
                fmt_rate_ci(
                    int(row["per_group_acc_detected"]),
                    n_t,
                    float(row["per_group_acc_rate"]),
                    row["per_group_acc_ci_95"],
                    suf,
                )
            )
            cells.append(
                fmt_rate_ci(
                    int(row["fixed_dp_detected"]),
                    n_t,
                    float(row["fixed_dp_rate"]),
                    row["fixed_dp_ci_95"],
                    suf,
                )
            )
        cells.extend([dp_str, mde_str])
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _render_methodology_section(
    comp: dict[str, Any], command_str: str, seeds: list[int]
) -> list[str]:
    lines = [
        "",
        "## 5. Baseline & Comparator Definitions",
        "",
        "1. **Overall Accuracy Baseline (`|acc_champ - acc_cand| > 0.01`)**: Flags any trial where overall test accuracy changes by more than 1 percentage point in either direction.",
        "2. **Per-Group Accuracy-Drop Baseline (`max_g (acc_champ(g) - acc_cand(g)) > 0.02`, `n_g >= 30`)**: Flags any trial where any subgroup's accuracy drops by more than 2 percentage points compared to the champion.",
        "3. **Fixed Demographic-Parity-Limit Baseline (`DP_cand >= 0.10`)**: Classic static absolute fairness threshold flagging any candidate with demographic parity difference `>= 0.10` regardless of champion baseline.",
    ]
    if comp:
        lines.append(
            f"4. **Evidently Off-the-Shelf Default Comparator (`evidently=={comp.get('evidently_version', '0.7.23')}`, `pandas=={comp.get('pandas_version', '2.3.3')}`)**: "
            f"{comp.get('description', 'Runs Report([DataDriftPreset()], include_tests=True) on candidate vs reference training data and Report([ClassificationPreset()], include_tests=True) on candidate vs champion test predictions with 100% Evidently default thresholds; flagged if any test returns FAIL.')}"
        )

    lines.extend(
        [
            "",
            "### Exact Evidently Comparator Configuration",
            "",
            f"- **Package versions (isolated venv `/tmp/evidently-venv`)**: `evidently=={comp.get('evidently_version', '0.7.23')}`, `pandas=={comp.get('pandas_version', '2.3.3')}`, `numpy<2.3`, `scikit-learn>=1.5` (isolated because Evidently `0.7.x` imports `pandas.core.dtypes.common.infer_dtype_from_object`, which was removed in pandas `3.0`).",
            "- **Worker script**: [`scripts/evidently_comparator.py`](../scripts/evidently_comparator.py) (`--worker` JSONL mode, invoked by [`scripts/bench_public.py`](../scripts/bench_public.py)).",
            "- **Training-data drift report**: `Report([DataDriftPreset()], include_tests=True)` run on `(current=candidate_train[common_cols], reference=champion_train[common_cols])` (all shared columns including target) with **100% default Evidently thresholds** (skipped only when `candidate_train[common_cols].equals(champion_train[common_cols])`, which is mathematically guaranteed to pass).",
            "- **Test-prediction classification report**: `Report([ClassificationPreset()], include_tests=True)` run on `(current=test_df[[target_col, '__pred__']] (candidate), reference=test_df[[target_col, '__pred__']] (champion))` with `DataDefinition(classification=[BinaryClassification(target=target_col, prediction_labels='__pred__')])` and **100% default Evidently thresholds**.",
            "- **Decision rule**: `failed = True` if **any** auto-generated test in either report returns status ending in `FAIL`.",
            "",
            "```python",
            "from evidently import BinaryClassification, DataDefinition, Dataset, Report",
            "from evidently.presets import ClassificationPreset, DataDriftPreset",
            "",
            "# 1. Training-data drift (100% default thresholds)",
            "drift_report = Report([DataDriftPreset()], include_tests=True)",
            "drift_snap = drift_report.run(",
            "    Dataset.from_pandas(tr_cur[common_cols]),",
            "    Dataset.from_pandas(tr_ref[common_cols]),",
            ")",
            "",
            "# 2. Test-prediction classification comparison (100% default thresholds)",
            "dd = DataDefinition(",
            "    classification=[BinaryClassification(target=target_col, prediction_labels='__pred__')]",
            ")",
            "clf_report = Report([ClassificationPreset()], include_tests=True)",
            "clf_snap = clf_report.run(",
            "    Dataset.from_pandas(te_cur, data_definition=dd),",
            "    Dataset.from_pandas(te_ref, data_definition=dd),",
            ")",
            "",
            "# Flagged iff any test in drift_snap.dict()['tests'] or clf_snap.dict()['tests'] ends with 'FAIL'",
            "```",
            "",
            "## How to Reproduce",
            "",
            "```bash",
            "# 1. Create the PredGate environment",
            "python -m venv .venv",
            ".venv/bin/pip install -e '.[dev]'",
            "",
            "# 2. (Optional) Create the isolated Evidently 0.7.23 comparator venv",
            "python -m venv /tmp/evidently-venv",
            "/tmp/evidently-venv/bin/pip install 'evidently==0.7.23' 'pandas==2.3.3' 'numpy<2.3' 'scikit-learn>=1.5'",
            "",
            "# 3. Run the full 30-seed fresh evaluation across all 5 datasets, 9 benign changes, and all fault families",
            "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\",
            f"  .venv/bin/python {command_str.replace('python ', '')} --workers 4",
            "",
            "# 4. (Optional) Run the fresh-seed config-preset benign-change comparison (strict / balanced / quiet)",
            "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\",
            "  .venv/bin/python scripts/bench_presets.py --n-seeds 30 --start-seed 11000 --workers 4",
            "```",
            "",
            "Pass `--evidently-python ''` to run PredGate and the three built-in baselines without Evidently.",
            "",
            "## Methodology",
            "",
            f"All suites use {len(seeds)} independent fresh random seeds (`{seeds[0]}..{seeds[-1]}`). "
            "**All five datasets** (UCI Adult `adult` v2, German Credit `credit-g` v1, Synthetic Credit, Bank Marketing `1461`, and COMPAS `42192`) **were used during development**: "
            "seeds `6000..6029` and `7000..7029` from commit `8467d37` were inspected to diagnose the `slices.py`, `drift.py`, and algo-swap defects, after which all thresholds were frozen and re-evaluated on fresh seeds `9000..9029` with zero post-hoc tuning. "
            "Public datasets are fetched directly from OpenML via `scripts/bench_public.py`. "
            "Confidence intervals use the 95% Wilson score interval (`z = 1.95996`). "
            "See [`docs/power.md`](power.md) for minimum sample-size calculations at 80% power.",
        ]
    )
    return lines


def generate_benchmark_markdown_from_json(json_path: Path, md_path: Path) -> str:
    """Render `docs/benchmark.md` strictly from an aggregate `benchmark_results.json` file."""
    data: dict[str, Any] = json.loads(json_path.read_text())
    summary: dict[str, dict[str, Any]] = data.get("summary", {})
    commit_hash = str(data.get("commit_hash", "unknown"))
    command_str = str(data.get("command", "predgate bench"))
    per_seed_artifact = str(data.get("per_seed_artifact", ""))
    seeds: list[int] = list(data.get("seeds", []))
    if not seeds:
        first_entry: dict[str, Any] = next(iter(summary.values()), {})
        n_default = int(first_entry.get("n_trials", 30))
        seeds = [1000 + i for i in range(n_default)]

    seed_range_str = f"`{seeds[0]}..{seeds[-1]}` (N={len(seeds)})"
    noop = summary.get("noop", {})
    noop_n = noop.get("n_trials", len(seeds))
    noop_fail_k = noop.get("pg_fail", noop.get("pg_detected", 0))
    noop_fail_rate = noop.get("predgate_fail_rate", noop.get("predgate_detection_rate", 0.0))
    noop_fail_ci = noop.get("predgate_fail_ci_95", noop.get("predgate_ci_95", [0.0, 0.114]))
    noop_pg_k = noop.get("pg_detected", 0)
    noop_pg_rate = noop.get("predgate_detection_rate", 0.0)
    noop_pg_ci = noop.get("predgate_ci_95", [0.0, 0.114])
    noop_bl_k = noop.get("bl_detected", 0)
    noop_bl_rate = noop.get("baseline_detection_rate", 0.0)
    noop_bl_ci = noop.get("baseline_ci_95", [0.0, 0.114])

    lines = [
        "# PredGate Benchmark Results",
        "",
        "Single source of truth for all PredGate benchmark evaluations, generated strictly from [`docs/benchmark_results.json`](benchmark_results.json) (aggregate summary; no hardcoded numbers).",
        "",
        "> **Dataset-Use Disclosure**: **All five benchmark datasets** (**UCI Adult**, **German Credit**, **Synthetic Credit**, **Bank Marketing**, and **COMPAS**) **were used during development**. Bank Marketing and COMPAS were initially evaluated as held-out datasets at commit `8467d37`, then inspected to diagnose the `slices.py` CI-bound and `drift.py` quantile-PSI/point-mass defects, and re-evaluated on fresh seeds (`9000..9029`) with frozen thresholds and zero post-hoc tuning. None of the five datasets should be treated as an unseen held-out benchmark.",
        "",
        "## Reproducibility Metadata",
        "",
        f"- **Command**: `{command_str}`",
        f"- **Evaluated Code Commit Hash**: `{commit_hash}`",
        f"- **Synthetic Suite Seeds**: {seed_range_str}",
        f"- **No-op FP (PredGate FAIL-only)**: **{fmt_rate_ci(noop_fail_k, noop_n, noop_fail_rate, noop_fail_ci, 'FP')}**",
        f"- **No-op FP (PredGate Flagged WARN+FAIL)**: **{fmt_rate_ci(noop_pg_k, noop_n, noop_pg_rate, noop_pg_ci, 'FP')}**",
        f"- **No-op FP (Overall Accuracy Baseline)**: **{fmt_rate_ci(noop_bl_k, noop_n, noop_bl_rate, noop_bl_ci, 'FP')}**",
    ]
    if per_seed_artifact:
        lines.append(
            f"- **Per-Seed Records Release Artifact**: `{per_seed_artifact}` (excluded from repo JSON to keep git lean; attach to GitHub Release)"
        )

    lines.extend(render_synthetic_section(summary, seeds))

    benign = data.get("benign_change_control")
    if isinstance(benign, dict) and benign.get("rows"):
        lines.extend(render_benign_section(benign))

    lines.extend(
        render_presets_section(json_path.parent / "preset_benchmark_results.json", commit_hash)
    )

    held = data.get("held_out_validation")
    if isinstance(held, dict) and held.get("rows"):
        lines.extend(_render_fault_validation_section(held))

    pub = data.get("public_fairness_benchmark")
    if isinstance(pub, dict) and pub.get("rows"):
        lines.extend(_render_fairness_section(pub, commit_hash))

    lines.extend(_render_methodology_section(data.get("comparator_config", {}), command_str, seeds))

    content = "\n".join(lines) + "\n"
    md_path.write_text(content)
    return content


def _generate_benchmark_markdown(
    summary: dict[str, dict[str, Any]],
    path: Path,
    commit_hash: str = "unknown",
    seeds: list[int] | None = None,
    command_str: str = "predgate bench",
) -> None:
    """Render benchmark markdown from an in-memory summary dict via the JSON renderer."""
    if seeds is None:
        first_entry: dict[str, Any] = next(iter(summary.values()), {})
        n_default = int(first_entry.get("n_trials", 30))
        seeds = [1000 + i for i in range(n_default)]
    payload = {
        "command": command_str,
        "commit_hash": commit_hash,
        "n_seeds": len(seeds),
        "seed_range": f"{seeds[0]}..{seeds[-1]}" if seeds else "",
        "seeds": seeds,
        "summary": summary,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(payload, tf, indent=2)
        tmp_json = Path(tf.name)
    try:
        generate_benchmark_markdown_from_json(tmp_json, path)
    finally:
        tmp_json.unlink(missing_ok=True)
