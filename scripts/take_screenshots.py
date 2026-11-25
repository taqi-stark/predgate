"""Capture live PredGate dashboard screenshots into docs/assets/ using Playwright.

Requires a running API + seeded runs and a running dashboard (e.g. `docker compose up --build`).
Captures:
  1. docs/assets/dashboard_runs.png           (Runs list with PASS/WARN/FAIL + badges)
  2. docs/assets/dashboard_run_detail.png     (PR #42 run detail with reasons)
  3. docs/assets/dashboard_slice_explorer.png (PR #42 Slices tab open)
  4. docs/assets/dashboard_compare.png        (Compare baseline PASS vs PR #42 FAIL)
  5. docs/assets/dashboard_models.png         (Models registry view)
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import httpx
from playwright.sync_api import sync_playwright


def _find_chromium_executable() -> str | None:
    env_exe = os.getenv("PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH", "").strip()
    if env_exe and Path(env_exe).exists():
        return env_exe
    home = Path.home()
    for pat in (
        str(home / ".cache/ms-playwright/chromium-*/chrome-linux*/chrome"),
        str(home / ".cache/ms-playwright/chromium_headless_shell-*/chrome-linux*/headless_shell"),
    ):
        matches = sorted(glob.glob(pat))
        if matches:
            return matches[-1]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture PredGate dashboard screenshots.")
    parser.add_argument("--dashboard-url", default="http://localhost:3000")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = httpx.get(f"{args.api_url.rstrip('/')}/runs", timeout=10.0).json()
    if not runs:
        raise RuntimeError(f"No seeded runs found at {args.api_url}/runs")

    pr42 = next((r for r in runs if r.get("pr_number") == 42), None)
    if pr42 is None:
        pr42 = next((r for r in runs if r.get("status") == "fail"), runs[0])
    pass_run = next((r for r in runs if r.get("status") == "pass"), runs[-1])

    pr42_id = pr42["id"]
    pass_id = pass_run["id"]
    dash = args.dashboard_url.rstrip("/")

    launch_kwargs: dict[str, object] = {"headless": True}
    exe = _find_chromium_executable()
    if exe:
        launch_kwargs["executable_path"] = exe

    with sync_playwright() as p:
        browser = p.chromium.launch(**launch_kwargs)  # type: ignore[arg-type]
        page = browser.new_page(viewport={"width": 1440, "height": 900})

        # 1. Runs list
        page.goto(f"{dash}/", wait_until="networkidle")
        page.wait_for_selector("table.data-table tbody tr", timeout=15000)
        p1 = args.output_dir / "dashboard_runs.png"
        page.screenshot(path=str(p1), full_page=True)
        print(f"Saved {p1}")

        # 2. Run detail with reasons (PR #42)
        page.goto(f"{dash}/runs/{pr42_id}", wait_until="networkidle")
        page.wait_for_selector(".reasons-list", timeout=15000)
        p2 = args.output_dir / "dashboard_run_detail.png"
        page.screenshot(path=str(p2), full_page=True)
        print(f"Saved {p2}")

        # 3. Slice explorer (PR #42 Slices tab)
        page.locator(".tab", has_text="Slices").click()
        page.wait_for_selector("table.data-table th:has-text('Champion')", timeout=15000)
        p3 = args.output_dir / "dashboard_slice_explorer.png"
        page.screenshot(path=str(p3), full_page=True)
        print(f"Saved {p3}")

        # 4. Compare page
        page.goto(f"{dash}/compare?a={pass_id}&b={pr42_id}", wait_until="networkidle")
        page.wait_for_selector("table.data-table tbody tr", timeout=15000)
        p4 = args.output_dir / "dashboard_compare.png"
        page.screenshot(path=str(p4), full_page=True)
        print(f"Saved {p4}")

        # 5. Models page
        page.goto(f"{dash}/models", wait_until="networkidle")
        page.wait_for_selector("table.data-table tbody tr", timeout=15000)
        p5 = args.output_dir / "dashboard_models.png"
        page.screenshot(path=str(p5), full_page=True)
        print(f"Saved {p5}")

        browser.close()


if __name__ == "__main__":
    main()
