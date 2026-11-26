"""PredGate report builder.

Produces Markdown PR comments, JSON payloads, and HTML reports.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from predgate.loader import LineageInfo
from predgate.verdict import Verdict

UTC = timezone.utc


def build_pr_comment(
    verdict: Verdict,
    candidate_sha: str = "",
    champion_version: str = "",
    seed: int = 42,
    data_hash: str = "",
    run_id: int | str = "",
    dashboard_url: str = "http://localhost:3000",
) -> str:
    """Build a Markdown PR comment from a verdict.

    Args:
        verdict: The PredGate verdict.
        candidate_sha: Git SHA of the candidate commit.
        champion_version: Version string for the champion model.
        seed: Random seed used.
        data_hash: Short data hash.
        run_id: Run identifier.
        dashboard_url: URL to the dashboard.

    Returns:
        Markdown string for the PR comment.
    """
    status_emoji = {
        "pass": "✅",
        "warn": "⚠️",
        "fail": "❌",
        "skip": "⏭️",
    }

    lines: list[str] = []
    lines.append(f"## PredGate verdict: {verdict.status.upper()}")
    lines.append("")

    # Header info
    header_parts = []
    if candidate_sha:
        header_parts.append(f"**Candidate** `{candidate_sha}`")
    if champion_version:
        header_parts.append(f"vs **Champion** {champion_version}")
    header_parts.append(f"seed {seed}")
    if data_hash:
        header_parts.append(f"data `{data_hash[:8]}...`")
    lines.append(" | ".join(header_parts))
    lines.append("")

    # Prominent model-family change banner
    mfc = verdict.model_family_change
    if isinstance(mfc, dict) and mfc.get("changed"):
        champ_fam = mfc.get("champion_family", "unknown")
        cand_fam = mfc.get("candidate_family", "unknown")
        lines.extend(
            [
                "> [!NOTE]",
                f"> **Model family changed:** Champion `{champ_fam}` → Candidate `{cand_fam}`. "
                "Algorithm swaps can shift single-row inference latency and subgroup selection rates even without data or label bugs.",
                "",
            ]
        )

    # Check results table
    lines.append("| Check | Status | Summary |")
    lines.append("|---|---|---|")
    for cr in verdict.check_results:
        emoji = status_emoji.get(cr.status, "")
        safe_summary = cr.summary.replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {cr.name.replace('_', ' ').title()} | {emoji} {cr.status.upper()} | {safe_summary} |"
        )
    lines.append("")

    # Prominent inconclusive slices callout & table
    inconclusive_entries: list[dict[str, Any]] = []
    inconclusive_notes: list[str] = []
    power_notes: list[str] = []
    for cr in verdict.check_results:
        for entry in cr.details.get("inconclusive_entries", []):
            inconclusive_entries.append(entry)
        for note in cr.details.get("inconclusive_notes", []):
            inconclusive_notes.append(f"[{cr.name}] {note}")
        for note in cr.details.get("power_notes", []):
            power_notes.append(f"[{cr.name}] {note}")

    if inconclusive_entries or inconclusive_notes:
        lines.append("### ⚠️ Inconclusive Slices (Wide Confidence Intervals)")
        lines.append("")
        lines.append("> [!WARNING]")
        lines.append(
            "> One or more evaluation slices have confidence intervals wider than `slices.max_ci_width` — "
            "a real regression may go undetected. See `docs/power.md` for minimum slice sizes, or set "
            "`slices.on_inconclusive: info` if you do not want inconclusive slices to trigger `WARN`."
        )
        lines.append("")
        if inconclusive_entries:
            lines.append("| Slice | N | Drop | 95% CI (Drop) | CI Width | Limit |")
            lines.append("|---|---|---|---|---|---|")
            for e in inconclusive_entries[:10]:
                lines.append(
                    f"| `{e['slice']}` | {e['n_rows']} | {e['diff']:+.3f} | "
                    f"[{e['ci_lower']:+.3f}, {e['ci_upper']:+.3f}] | "
                    f"**{e['ci_width']:.3f}** | `{e['max_ci_width']:.2f}` |"
                )
        else:
            for note in inconclusive_notes:
                lines.append(f"- ⚠️ {note}")
        lines.append("")

    # Top reasons
    if verdict.reasons:
        lines.append("**Top reasons**")
        for i, reason in enumerate(verdict.reasons[:5], 1):
            lines.append(f"{i}. {reason.message}")
        lines.append("")

    # Statistical power / minimum detectable gap notes
    if power_notes:
        lines.append("**Statistical power notes**")
        for note in power_notes:
            lines.append(f"- ⚠️ {note}")
        lines.append("")

    # Footer
    links = []
    if run_id:
        links.append(f"[Open full report in dashboard]({dashboard_url}/runs/{run_id})")
        links.append(f"Reproduce: `predgate reproduce {run_id}`")
    lines.append(" | ".join(links))

    return "\n".join(lines)


def build_json_payload(
    verdict: Verdict,
    lineage: LineageInfo | None = None,
    run_id: int | str | None = None,
    project: str = "",
    branch: str = "",
    pr_number: int | None = None,
) -> dict[str, Any]:
    """Build a JSON payload for the API.

    Args:
        verdict: The PredGate verdict.
        lineage: Optional lineage information.
        run_id: Run identifier.
        project: Project name.
        branch: Branch name.
        pr_number: PR number.

    Returns:
        Dictionary suitable for JSON serialization.
    """
    payload: dict[str, Any] = {
        "project": project,
        "branch": branch,
        "pr_number": pr_number,
        "timestamp": datetime.now(UTC).isoformat(),
        "verdict": verdict.to_dict(),
    }

    if run_id is not None:
        payload["run_id"] = run_id

    if lineage is not None:
        safe_hyperparams = json.loads(json.dumps(lineage.hyperparameters, default=str))
        payload["lineage"] = {
            "git_sha": lineage.git_sha,
            "data_hashes": lineage.data_hashes,
            "dvc_hash": lineage.dvc_hash,
            "python_version": lineage.python_version,
            "seed": lineage.seed,
            "hyperparameters": safe_hyperparams,
        }

    return payload


def build_html_report(
    verdict: Verdict,
    lineage: LineageInfo | None = None,
    project: str = "",
) -> str:
    """Build an HTML report file.

    Args:
        verdict: The PredGate verdict.
        lineage: Optional lineage information.
        project: Project name.

    Returns:
        HTML string.
    """
    status_colors = {
        "pass": "#22c55e",
        "warn": "#f59e0b",
        "fail": "#ef4444",
        "skip": "#9ca3af",
    }

    color = status_colors.get(verdict.status, "#9ca3af")

    checks_html = ""
    for cr in verdict.check_results:
        cr_color = status_colors.get(cr.status, "#9ca3af")
        checks_html += f"""
        <tr>
            <td>{cr.name.replace("_", " ").title()}</td>
            <td><span style="color:{cr_color};font-weight:bold">{cr.status.upper()}</span></td>
            <td>{cr.summary}</td>
            <td>{cr.duration_s:.2f}s</td>
        </tr>"""

    reasons_html = ""
    if verdict.reasons:
        reasons_html = "<h3>Reasons</h3><ol>"
        for r in verdict.reasons:
            reasons_html += (
                f"<li><strong>{r.severity.upper()}</strong> [{r.check_name}] {r.message}</li>"
            )
        reasons_html += "</ol>"

    family_html = ""
    mfc = verdict.model_family_change
    if isinstance(mfc, dict) and mfc.get("changed"):
        champ_fam = mfc.get("champion_family", "unknown")
        cand_fam = mfc.get("candidate_family", "unknown")
        family_html = (
            '<div class="callout callout-info">'
            f"<strong>Model family changed:</strong> Champion <code>{champ_fam}</code> &rarr; "
            f"Candidate <code>{cand_fam}</code>. Algorithm swaps can shift single-row inference "
            "latency and subgroup selection rates even without data or label bugs."
            "</div>"
        )

    inconclusive_html = ""
    html_inconclusive_entries: list[dict[str, Any]] = []
    html_power_notes: list[str] = []
    html_inconclusive_notes: list[str] = []
    for cr in verdict.check_results:
        for entry in cr.details.get("inconclusive_entries", []):
            html_inconclusive_entries.append(entry)
        for note in cr.details.get("power_notes", []):
            html_power_notes.append(f"[{cr.name}] {note}")
        for note in cr.details.get("inconclusive_notes", []):
            html_inconclusive_notes.append(f"[{cr.name}] {note}")

    if html_inconclusive_entries or html_inconclusive_notes:
        if html_inconclusive_entries:
            inc_rows = ""
            for e in html_inconclusive_entries[:15]:
                inc_rows += (
                    f"<tr><td><code>{e['slice']}</code></td><td>{e['n_rows']}</td>"
                    f"<td>{e['diff']:+.3f}</td>"
                    f"<td>[{e['ci_lower']:+.3f}, {e['ci_upper']:+.3f}]</td>"
                    f"<td><strong>{e['ci_width']:.3f}</strong> (&gt; {e['max_ci_width']:.2f})</td></tr>"
                )
            inc_body = (
                "<table><tr><th>Slice</th><th>N</th><th>Drop</th>"
                f"<th>95% CI (Drop)</th><th>CI Width</th></tr>{inc_rows}</table>"
            )
        else:
            inc_body = "<ul>" + "".join(f"<li>{n}</li>" for n in html_inconclusive_notes) + "</ul>"
        inconclusive_html = (
            '<div class="callout callout-warn">'
            "<h3>&#9888;&#65039; Inconclusive Slices (Wide Confidence Intervals)</h3>"
            "<p>One or more slices have confidence intervals wider than <code>slices.max_ci_width</code>; "
            "a real drop may not be detected with sufficient power (see <code>docs/power.md</code>).</p>"
            f"{inc_body}</div>"
        )

    power_html = ""
    if html_power_notes:
        power_html = "<h3>Statistical Power Notes</h3><ul>"
        for note in html_power_notes:
            power_html += f"<li>{note}</li>"
        power_html += "</ul>"

    lineage_html = ""
    if lineage:
        dvc_row = (
            f"<tr><td>DVC Hash</td><td><code>{lineage.dvc_hash}</code></td></tr>"
            if lineage.dvc_hash
            else ""
        )
        lineage_html = f"""
        <h3>Lineage</h3>
        <table>
            <tr><td>Git SHA</td><td><code>{lineage.git_sha}</code></td></tr>
            <tr><td>Seed</td><td>{lineage.seed}</td></tr>
            <tr><td>Python</td><td>{lineage.python_version}</td></tr>
            {dvc_row}
            <tr><td>Data Hashes</td><td><code>{json.dumps(lineage.data_hashes)}</code></td></tr>
        </table>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>PredGate Report — {project}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
               max-width: 900px; margin: 40px auto; padding: 0 20px;
               background: #0f172a; color: #e2e8f0; }}
        h1 {{ color: {color}; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #334155; padding: 8px 12px; text-align: left; }}
        th {{ background: #1e293b; }}
        code {{ background: #1e293b; padding: 2px 6px; border-radius: 4px;
                font-family: 'JetBrains Mono', monospace; }}
        .badge {{ display: inline-block; padding: 4px 12px; border-radius: 4px;
                  font-weight: bold; color: white; background: {color}; }}
        .callout {{ padding: 14px 18px; border-radius: 6px; margin: 16px 0; }}
        .callout-warn {{ background: rgba(245, 158, 11, 0.12); border-left: 4px solid #f59e0b; }}
        .callout-warn h3 {{ margin-top: 0; color: #fbbf24; }}
        .callout-info {{ background: rgba(59, 130, 246, 0.12); border-left: 4px solid #3b82f6; }}
    </style>
</head>
<body>
    <h1>PredGate Report</h1>
    <p>Project: <strong>{project}</strong></p>
    <p>Verdict: <span class="badge">{verdict.status.upper()}</span></p>
    <p>{verdict.summary}</p>

    {family_html}
    {inconclusive_html}
    {reasons_html}
    {power_html}

    <h3>Check Results</h3>
    <table>
        <tr><th>Check</th><th>Status</th><th>Summary</th><th>Duration</th></tr>
        {checks_html}
    </table>

    {lineage_html}

    <hr>
    <p style="color:#64748b;font-size:0.85em">
        Generated by PredGate at {datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")}
    </p>
</body>
</html>"""

    return html
