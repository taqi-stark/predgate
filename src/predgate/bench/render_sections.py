"""Markdown table builders for benchmark report sections."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any


def fmt_rate_ci(k: int, n: int, rate: float, ci: list[float], suffix: str = "") -> str:
    """Format a count, percentage rate, and 95% Wilson CI cell."""
    suf = f" {suffix}" if suffix else ""
    return f"{k}/{n} ({rate:.1%}{suf}, [{ci[0]:.1%}, {ci[1]:.1%}])"


def render_synthetic_section(summary: dict[str, dict[str, Any]], seeds: list[int]) -> list[str]:
    """Render Section 1 (Synthetic 7-Fault Benchmark) lines."""
    has_synth_evidently = any("evidently_detected" in s for s in summary.values())
    has_synth_stronger = any("per_group_acc_detected" in s for s in summary.values())
    lines = [
        "",
        f"## 1. Synthetic Credit 7-Fault Benchmark (Seeds `{seeds[0]}..{seeds[-1]}`)",
        "",
        "Reports **FAIL-only** (hard merge block) and **Flagged (WARN+FAIL)** (blocked or flagged for human review) separately alongside stronger baselines.",
        "",
    ]
    if has_synth_stronger and has_synth_evidently:
        lines.append(
            "| Fault | PredGate FAIL-only (95% CI) | PredGate Flagged WARN+FAIL (95% CI) | Overall Acc (`\\|Δacc\\|>1%`) | Per-Group Acc Drop (`>2%`) | Fixed DP (`>=0.10`) | Evidently 0.7 Defaults |"
        )
        lines.append(
            "|-------|-----------------------------|-------------------------------------|------------------------------|----------------------------|---------------------|------------------------|"
        )
    elif has_synth_stronger:
        lines.append(
            "| Fault | PredGate FAIL-only (95% CI) | PredGate Flagged WARN+FAIL (95% CI) | Overall Acc (`\\|Δacc\\|>1%`) | Per-Group Acc Drop (`>2%`) | Fixed DP (`>=0.10`) |"
        )
        lines.append(
            "|-------|-----------------------------|-------------------------------------|------------------------------|----------------------------|---------------------|"
        )
    else:
        lines.append(
            "| Fault | PredGate FAIL-only (95% CI) | PredGate Flagged WARN+FAIL (95% CI) | Overall Acc (`\\|Δacc\\|>1%`) |"
        )
        lines.append(
            "|-------|-----------------------------|-------------------------------------|------------------------------|"
        )

    for fault, s in summary.items():
        n = int(s["n_trials"])
        suf = "FP" if fault == "noop" else ""
        fail_k = int(s.get("pg_fail", s["pg_detected"]))
        fail_r = float(s.get("predgate_fail_rate", s["predgate_detection_rate"]))
        fail_ci = s.get("predgate_fail_ci_95", s["predgate_ci_95"])
        flag_k = int(s["pg_detected"])
        flag_r = float(s["predgate_detection_rate"])
        flag_ci = s["predgate_ci_95"]
        bl_k = int(s["bl_detected"])
        bl_r = float(s["baseline_detection_rate"])
        bl_ci = s["baseline_ci_95"]

        fail_cell = fmt_rate_ci(fail_k, n, fail_r, fail_ci, suf)
        flag_cell = fmt_rate_ci(flag_k, n, flag_r, flag_ci, suf)
        bl_cell = fmt_rate_ci(bl_k, n, bl_r, bl_ci, suf)
        if fault == "noop":
            fail_cell = f"**{fail_cell}**"
            flag_cell = f"**{flag_cell}**"
            bl_cell = f"**{bl_cell}**"

        row_cells = [f"`{fault}`", fail_cell, flag_cell, bl_cell]
        if has_synth_stronger:
            pg_k = int(s.get("per_group_acc_detected", 0))
            pg_r = float(s.get("per_group_acc_rate", 0.0))
            pg_ci = s.get("per_group_acc_ci_95", [0.0, 0.0])
            dp_k = int(s.get("fixed_dp_detected", 0))
            dp_r = float(s.get("fixed_dp_rate", 0.0))
            dp_ci = s.get("fixed_dp_ci_95", [0.0, 0.0])
            row_cells.append(fmt_rate_ci(pg_k, n, pg_r, pg_ci, suf))
            row_cells.append(fmt_rate_ci(dp_k, n, dp_r, dp_ci, suf))
        if has_synth_evidently:
            ev_k = int(s.get("evidently_detected", 0))
            ev_r = float(s.get("evidently_rate", 0.0))
            ev_ci = s.get("evidently_ci_95", [0.0, 0.0])
            row_cells.append(fmt_rate_ci(ev_k, n, ev_r, ev_ci, suf))
        lines.append("| " + " | ".join(row_cells) + " |")
    return lines


def render_benign_section(benign: dict[str, Any]) -> list[str]:
    """Render Section 2 (Benign-Change Control) lines."""
    b_seeds: list[int] = list(benign.get("seeds", []))
    b_range = (
        f"{b_seeds[0]}..{b_seeds[-1]}" if b_seeds else str(benign.get("seed_range", "unknown"))
    )
    ov = benign.get("overall", {})
    b_has_ev = any("evidently_detected" in r for r in benign["rows"])
    b_has_ql = any("quality_fail_count" in r for r in benign["rows"])
    n_changes = len(benign.get("changes", [])) or len({r["change"] for r in benign["rows"]})
    n_datasets = len({r["dataset"] for r in benign["rows"]})
    lines = [
        "",
        f"## 2. Benign-Change Control (Harmless Retrains, Seeds `{b_range}`, Frozen Defaults)",
        "",
        f"{n_changes} realistic harmless retrain scenarios across {len(b_seeds) or 'N'} fresh seeds × {n_datasets} datasets "
        f"({int(ov.get('n_trials', len(benign['rows'])))} total trials), running all 6 schema-agnostic checks with **zero threshold tuning**. "
        "**Quality findings** (`drift`, `leakage`, `label_shift`, `slices`, `fairness`) are shown **separately from latency findings** (`budgets`).",
        "",
    ]
    if ov:
        ov_n = int(ov.get("n_trials", 0))
        if "quality_fail_count" in ov:
            lines.extend(
                [
                    f"- **Quality-Only Benign FAIL Rate (Hard Block FP)**: **{fmt_rate_ci(int(ov['quality_fail_count']), ov_n, float(ov['quality_fail_rate']), ov['quality_fail_ci_95'])}**",
                    f"- **Quality-Only Benign WARN-Only Rate (Alert Fatigue)**: **{fmt_rate_ci(int(ov['quality_warn_only_count']), ov_n, float(ov['quality_warn_only_rate']), ov['quality_warn_only_ci_95'])}**",
                    f"- **Latency-Only Benign FAIL Rate (`budgets`)**: **{fmt_rate_ci(int(ov['latency_fail_count']), ov_n, float(ov['latency_fail_rate']), ov['latency_fail_ci_95'])}**",
                    f"- **Latency-Only Benign WARN Rate (`budgets`)**: **{fmt_rate_ci(int(ov['latency_warn_count']), ov_n, float(ov['latency_warn_rate']), ov['latency_warn_ci_95'])}**",
                    f"- **Combined Benign FAIL Rate (Quality + Latency)**: **{fmt_rate_ci(int(ov['fail_count']), ov_n, float(ov['fail_rate']), ov['fail_ci_95'])}** | **Combined WARN-Only**: **{fmt_rate_ci(int(ov['warn_only_count']), ov_n, float(ov['warn_only_rate']), ov['warn_only_ci_95'])}**",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    f"- **Overall Benign FAIL Rate (Hard Block FP)**: **{fmt_rate_ci(int(ov['fail_count']), ov_n, float(ov['fail_rate']), ov['fail_ci_95'])}**",
                    f"- **Overall Benign WARN-Only Rate (Alert Fatigue)**: **{fmt_rate_ci(int(ov['warn_only_count']), ov_n, float(ov['warn_only_rate']), ov['warn_only_ci_95'])}**",
                    f"- **Overall Benign Flagged Rate (WARN+FAIL)**: **{fmt_rate_ci(int(ov['flagged_count']), ov_n, float(ov['flagged_rate']), ov['flagged_ci_95'])}**",
                    "",
                ]
            )

    if b_has_ql:
        header_cols = [
            "Dataset",
            "Benign Change",
            "Quality FAIL (95% CI)",
            "Quality WARN-Only (Alert Fatigue)",
            "Latency FAIL (95% CI)",
            "Latency WARN",
            "Per-Group Acc Drop",
            "Fixed DP (`>=0.10`)",
        ]
    else:
        header_cols = [
            "Dataset",
            "Benign Change",
            "PredGate FAIL (95% CI)",
            "PredGate WARN-Only (Alert Fatigue)",
            "PredGate Flagged (WARN+FAIL)",
            "Per-Group Acc Drop",
            "Fixed DP (`>=0.10`)",
        ]
    if b_has_ev:
        header_cols.append("Evidently 0.7 Defaults")
    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("|" + "|".join("---" for _ in header_cols) + "|")

    for r in benign["rows"]:
        n_t = int(r["n_trials"])
        if b_has_ql:
            cells = [
                f"**{r['dataset']}**",
                f"`{r['change']}`",
                fmt_rate_ci(
                    int(r["quality_fail_count"]),
                    n_t,
                    float(r["quality_fail_rate"]),
                    r["quality_fail_ci_95"],
                ),
                fmt_rate_ci(
                    int(r["quality_warn_only_count"]),
                    n_t,
                    float(r["quality_warn_only_rate"]),
                    r["quality_warn_only_ci_95"],
                ),
                fmt_rate_ci(
                    int(r["latency_fail_count"]),
                    n_t,
                    float(r["latency_fail_rate"]),
                    r["latency_fail_ci_95"],
                ),
                fmt_rate_ci(
                    int(r["latency_warn_count"]),
                    n_t,
                    float(r["latency_warn_rate"]),
                    r["latency_warn_ci_95"],
                ),
                fmt_rate_ci(
                    int(r["per_group_acc_detected"]),
                    n_t,
                    float(r["per_group_acc_rate"]),
                    r["per_group_acc_ci_95"],
                ),
                fmt_rate_ci(
                    int(r["fixed_dp_detected"]),
                    n_t,
                    float(r["fixed_dp_rate"]),
                    r["fixed_dp_ci_95"],
                ),
            ]
        else:
            cells = [
                f"**{r['dataset']}**",
                f"`{r['change']}`",
                fmt_rate_ci(int(r["fail_count"]), n_t, float(r["fail_rate"]), r["fail_ci_95"]),
                fmt_rate_ci(
                    int(r["warn_only_count"]),
                    n_t,
                    float(r["warn_only_rate"]),
                    r["warn_only_ci_95"],
                ),
                fmt_rate_ci(
                    int(r["flagged_count"]),
                    n_t,
                    float(r["flagged_rate"]),
                    r["flagged_ci_95"],
                ),
                fmt_rate_ci(
                    int(r["per_group_acc_detected"]),
                    n_t,
                    float(r["per_group_acc_rate"]),
                    r["per_group_acc_ci_95"],
                ),
                fmt_rate_ci(
                    int(r["fixed_dp_detected"]),
                    n_t,
                    float(r["fixed_dp_rate"]),
                    r["fixed_dp_ci_95"],
                ),
            ]
        if b_has_ev:
            cells.append(
                fmt_rate_ci(
                    int(r["evidently_detected"]),
                    n_t,
                    float(r["evidently_rate"]),
                    r["evidently_ci_95"],
                )
            )
        lines.append("| " + " | ".join(cells) + " |")

    per_check = benign.get("per_check", [])
    if per_check:
        lines.extend(
            [
                "",
                "### Per-Check Benign-Change FAIL & WARN Rates (All Datasets Combined)",
                "",
                "| Check | Category | FAIL Count & Rate (95% CI) | WARN Count & Rate (Alert Fatigue, 95% CI) | Flagged WARN+FAIL (95% CI) |",
                "|-------|----------|----------------------------|-------------------------------------------|----------------------------|",
            ]
        )
        for pc in per_check:
            n_pc = int(pc["n_trials"])
            cat = "Latency" if pc["check"] == "budgets" else "Quality"
            lines.append(
                f"| `{pc['check']}` | {cat} | "
                f"{fmt_rate_ci(int(pc['fail_count']), n_pc, float(pc['fail_rate']), pc['fail_ci_95'])} | "
                f"{fmt_rate_ci(int(pc['warn_count']), n_pc, float(pc['warn_rate']), pc['warn_ci_95'])} | "
                f"{fmt_rate_ci(int(pc['flagged_count']), n_pc, float(pc['flagged_rate']), pc['flagged_ci_95'])} |"
            )

    algo_fair = benign.get("algo_swap_fairness_metrics")
    if isinstance(algo_fair, dict) and algo_fair.get("n_trials"):
        fail_m_str = (
            ", ".join(
                f"`{k}`: {v}"
                for k, v in sorted(algo_fair.get("fail_metrics", {}).items(), key=lambda x: -x[1])
            )
            or "none"
        )
        warn_m_str = (
            ", ".join(
                f"`{k}`: {v}"
                for k, v in sorted(algo_fair.get("warn_metrics", {}).items(), key=lambda x: -x[1])
            )
            or "none"
        )
        lines.extend(
            [
                "",
                "### Fairness Metric Breakdown on Benign Algorithm Swaps (`algo_swap_rf`, `algo_swap_logreg`)",
                "",
                f"- **Trials**: {algo_fair['n_trials']} (`fairness` FAIL: {algo_fair['n_fail']}, WARN: {algo_fair['n_warn']})",
                f"- **Triggering FAIL metrics**: {fail_m_str}",
                f"- **Triggering WARN metrics**: {warn_m_str}",
            ]
        )
    return lines


def render_presets_section(preset_json_path: Path, commit_hash: str) -> list[str]:
    """Render optional Section 2b (Config Presets Benign-Change Comparison)."""
    lines: list[str] = []
    if not preset_json_path.exists():
        return lines
    with contextlib.suppress(Exception):
        pdata: dict[str, Any] = json.loads(preset_json_path.read_text())
        presets_map: dict[str, Any] = pdata.get("presets", {})
        if not presets_map:
            return lines
        p_range = str(pdata.get("seed_range", "unknown"))
        p_cmd = str(pdata.get("command", "python scripts/bench_presets.py"))
        p_commit = str(pdata.get("commit_hash", commit_hash))
        lines.extend(
            [
                "",
                f"### Config Presets Benign-Change Comparison (Fresh Seeds `{p_range}`, Code `{p_commit}`)",
                "",
                f"Measured independently on fresh seeds `{p_range}` via `{p_cmd}` across all 5 datasets × 9 benign changes (1,350 trials per preset), with default thresholds unchanged (`balanced` == default; includes `slices.on_inconclusive: warn` for `strict`/`balanced` and `info` for `quiet`).",
                "",
                "| Preset | Quality FAIL (95% CI) | Quality WARN-Only (95% CI) | Latency FAIL (`budgets`) | Combined FAIL (95% CI) | Combined WARN-Only (95% CI) | Non-Swap Quality FAIL (N=1,050) |",
                "|---|---|---|---|---|---|---|",
            ]
        )
        for p_name in ("strict", "balanced", "quiet"):
            ps = presets_map.get(p_name)
            if not isinstance(ps, dict):
                continue
            pn = int(ps["n_trials"])
            pns = int(ps["non_swap_n_trials"])
            qf = ps["quality_fail"]
            qw = ps["quality_warn_only"]
            lf = ps["latency_fail"]
            cf = ps["combined_fail"]
            cw = ps["combined_warn_only"]
            nsqf = ps["non_swap_quality_fail"]
            lines.append(
                f"| **`{p_name}`** | "
                f"{fmt_rate_ci(int(qf['count']), pn, float(qf['rate']), qf['ci_95'])} | "
                f"{fmt_rate_ci(int(qw['count']), pn, float(qw['rate']), qw['ci_95'])} | "
                f"{fmt_rate_ci(int(lf['count']), pn, float(lf['rate']), lf['ci_95'])} | "
                f"{fmt_rate_ci(int(cf['count']), pn, float(cf['rate']), cf['ci_95'])} | "
                f"{fmt_rate_ci(int(cw['count']), pn, float(cw['rate']), cw['ci_95'])} | "
                f"{fmt_rate_ci(int(nsqf['count']), pns, float(nsqf['rate']), nsqf['ci_95'])} |"
            )
    return lines
