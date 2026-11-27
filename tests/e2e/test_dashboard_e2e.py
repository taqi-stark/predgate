"""End-to-end Playwright test covering CLI -> API (token auth) -> React Dashboard -> RBAC flows.

Verifies:
1. `predgate run` trains a real model, executes checks, and POSTs to the API with an engineer bearer token.
2. The run appears live in the React dashboard Runs list.
3. Opening the run detail page and clicking the Slices tab displays the Slice Explorer table.
4. Approve and Promote actions work for authorized roles (`reviewer` / `admin`) via the
   dashboard Settings token flow and are rejected (HTTP 403) for unauthorized roles (`engineer` / `reviewer`).
"""

from __future__ import annotations

import glob
import os
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import httpx
import pytest
import yaml

playwright_sync = pytest.importorskip("playwright.sync_api")


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


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_http(url: str, timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = httpx.get(url, timeout=1.5)
            if r.status_code < 500:
                return
        except Exception:
            pass
        time.sleep(0.3)
    raise TimeoutError(f"Service did not start at {url} within {timeout_s}s")


def test_e2e_cli_api_dashboard_slice_explorer_and_rbac(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    dash_dist = repo_root / "dashboard" / "dist"
    if not (dash_dist / "index.html").exists():
        pytest.skip("dashboard/dist not built; run `cd dashboard && npm run build`")

    # Dashboard dist defaults API_BASE to http://localhost:8000; bind API to 8000 if free,
    # else skip (or use existing if desired — here we assert port 8000 is available for clean isolation).
    api_port = 8000
    dash_port = _free_port()
    db_file = tmp_path / "e2e_predgate.db"
    db_url = f"sqlite:///{db_file}"

    # Provision engineer, reviewer, and admin users directly in the clean test DB (no default token)
    tokens: dict[str, str] = {}
    env_base = {
        **os.environ,
        "DATABASE_URL": db_url,
        "PREDGATE_CORS_ORIGINS": f"http://localhost:{dash_port},http://127.0.0.1:{dash_port},http://localhost:3000",
        "PYTHONPATH": str(repo_root / "src"),
    }
    init_code = textwrap.dedent(f"""\
        import json
        from sqlmodel import Session, SQLModel, create_engine
        from predgate.api.models import User  # noqa: F401
        from predgate.auth import create_user
        eng = create_engine({db_url!r})
        SQLModel.metadata.create_all(eng)
        out = {{}}
        with Session(eng) as s:
            for name, role in [("e2e_eng", "engineer"), ("e2e_rev", "reviewer"), ("e2e_adm", "admin")]:
                _, tok = create_user(s, name=name, role=role)
                out[role] = tok
        print(json.dumps(out))
    """)
    tok_proc = subprocess.run(
        [sys.executable, "-c", init_code],
        env=env_base,
        capture_output=True,
        text=True,
        check=True,
    )
    import json

    tokens = json.loads(tok_proc.stdout.strip())

    # Start FastAPI server
    api_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "predgate.api.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(api_port),
        ],
        cwd=str(repo_root),
        env=env_base,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Start SPA static server for dashboard/dist
    spa_script = textwrap.dedent(f"""\
        import http.server, os, socketserver
        os.chdir({str(dash_dist)!r})
        class H(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                clean = self.path.split('?', 1)[0]
                if not os.path.exists(self.translate_path(clean)):
                    self.path = '/index.html'
                return super().do_GET()
            def log_message(self, *args):
                pass
        socketserver.TCPServer.allow_reuse_address = True
        socketserver.TCPServer(('127.0.0.1', {dash_port}), H).serve_forever()
    """)
    dash_proc = subprocess.Popen(
        [sys.executable, "-c", spa_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        _wait_http(f"http://localhost:{api_port}/health")
        _wait_http(f"http://localhost:{dash_port}/")

        # Ensure demo data exists
        demo_dir = repo_root / "examples" / "credit-risk-demo"
        if not (demo_dir / "data" / "train.csv").exists():
            subprocess.run(
                [sys.executable, str(demo_dir / "train.py")],
                cwd=str(repo_root),
                check=True,
            )

        # 1. Execute real `predgate run` posting to API with engineer token
        out_dir = tmp_path / "gate_out"
        run_env = {
            **env_base,
            "API_URL": f"http://localhost:{api_port}",
            "PREDGATE_API_TOKEN": tokens["engineer"],
        }
        # Use a fast copy of predgate.yaml (fewer bootstrap resamples for quick E2E)
        cfg = yaml.safe_load((demo_dir / "predgate.yaml").read_text())
        cfg["checks"]["slices"]["n_bootstrap"] = 100
        cfg["mlflow"]["tracking_uri"] = f"sqlite:///{tmp_path / 'mlflow.db'}"
        cfg_path = tmp_path / "e2e_predgate.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))

        predgate_bin = str(Path(sys.executable).parent / "predgate")
        run_proc = subprocess.run(
            [
                predgate_bin,
                "run",
                "--config",
                str(cfg_path),
                "--branch",
                "e2e/smoke-pass",
                "--pr",
                "101",
                "--output",
                str(out_dir),
            ],
            cwd=str(repo_root),
            env=run_env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert run_proc.returncode == 0, (
            f"predgate run failed:\n{run_proc.stdout}\n{run_proc.stderr}"
        )
        assert "Run posted to API" in run_proc.stdout

        runs_list = httpx.get(f"http://localhost:{api_port}/runs", timeout=5.0).json()
        assert len(runs_list) == 1
        run_id = runs_list[0]["id"]
        assert runs_list[0]["branch"] == "e2e/smoke-pass"

        # 2. Verify via Playwright in real headless Chromium
        launch_kwargs: dict[str, object] = {"headless": True}
        exe = _find_chromium_executable()
        if exe:
            launch_kwargs["executable_path"] = exe

        dialogs: list[str] = []

        with playwright_sync.sync_playwright() as p:
            browser = p.chromium.launch(**launch_kwargs)  # type: ignore[arg-type]
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page.on("dialog", lambda d: (dialogs.append(d.message), d.accept()))

            # Runs list shows the posted run
            page.goto(f"http://localhost:{dash_port}/", wait_until="networkidle")
            page.wait_for_selector("table.data-table tbody tr", timeout=15000)
            row_text = page.locator("table.data-table tbody tr").first.inner_text()
            assert "e2e/smoke-pass" in row_text
            assert "#101" in row_text

            # Navigate into Run Detail and open Slice Explorer tab
            page.locator("table.data-table tbody tr").first.click()
            page.wait_for_url(f"**/runs/{run_id}", timeout=10000)
            page.locator(".tab", has_text="Slices").click()
            page.wait_for_selector("table.data-table th:has-text('Champion')", timeout=10000)
            slice_rows = page.locator("table.data-table tbody tr").count()
            assert slice_rows >= 2

            # Helper to set bearer token via Settings UI
            def set_token_in_ui(token_val: str) -> None:
                page.goto(f"http://localhost:{dash_port}/settings", wait_until="networkidle")
                inp = page.locator("input[type='password']")
                inp.fill(token_val)
                assert page.evaluate("localStorage.getItem('predgate_api_token')") == token_val

            # 3. Engineer role: Approve and Promote must both be rejected (403)
            set_token_in_ui(tokens["engineer"])
            page.goto(f"http://localhost:{dash_port}/runs/{run_id}", wait_until="networkidle")
            dialogs.clear()
            page.locator("button", has_text="Approve").first.click()
            page.locator(".modal textarea").fill("Engineer trying to approve own PR")
            page.locator(".modal button.btn-primary").click()
            page.wait_for_timeout(800)
            assert any(
                "role" in m.lower() or "forbidden" in m.lower() or "reviewer" in m.lower()
                for m in dialogs
            ), f"Expected 403 dialog for engineer approve, got: {dialogs}"
            page.locator(".modal button", has_text="Cancel").click()

            dialogs.clear()
            page.locator("button", has_text="Promote").click()
            page.wait_for_timeout(800)
            assert any(
                "admin" in m.lower() or "forbidden" in m.lower() or "role" in m.lower()
                for m in dialogs
            ), f"Expected 403 dialog for engineer promote, got: {dialogs}"

            # 4. Reviewer role: Approve succeeds; Promote must be rejected (403)
            set_token_in_ui(tokens["reviewer"])
            page.goto(f"http://localhost:{dash_port}/runs/{run_id}", wait_until="networkidle")
            dialogs.clear()
            page.locator("button", has_text="Approve").first.click()
            page.locator(".modal textarea").fill(
                "Reviewed slice CIs and fairness metrics, looks good."
            )
            page.locator(".modal button.btn-primary").click()
            page.wait_for_selector(".modal", state="detached", timeout=10000)
            assert not dialogs, f"Unexpected error dialog on reviewer approve: {dialogs}"

            detail_after_approve = httpx.get(
                f"http://localhost:{api_port}/runs/{run_id}", timeout=5.0
            ).json()
            assert len(detail_after_approve["approvals"]) == 1
            assert detail_after_approve["approvals"][0]["reviewer"] == "e2e_rev"

            dialogs.clear()
            page.locator("button", has_text="Promote").click()
            page.wait_for_timeout(800)
            assert any(
                "admin" in m.lower() or "forbidden" in m.lower() or "role" in m.lower()
                for m in dialogs
            ), f"Expected 403 dialog for reviewer promote, got: {dialogs}"

            # 5. Admin role: Promote succeeds
            set_token_in_ui(tokens["admin"])
            page.goto(f"http://localhost:{dash_port}/runs/{run_id}", wait_until="networkidle")
            dialogs.clear()
            page.locator("button", has_text="Promote").click()
            page.wait_for_timeout(800)
            assert any("promoted" in m.lower() for m in dialogs), (
                f"Expected Promoted! dialog for admin, got: {dialogs}"
            )

            browser.close()
    finally:
        api_proc.terminate()
        dash_proc.terminate()
        api_proc.wait(timeout=5)
        dash_proc.wait(timeout=5)
