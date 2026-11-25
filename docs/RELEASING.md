# Releasing PredGate (PyPI Trusted Publishing / OIDC)

PredGate publishes to **TestPyPI** and **PyPI** automatically on version tags (`v*`) via [`.github/workflows/release.yml`](../.github/workflows/release.yml) using **OpenID Connect (OIDC) Trusted Publishing** — no long-lived API tokens or repository secrets are required.

---

## One-Time Setup on TestPyPI and PyPI

Complete these two pending-publisher registrations once before pushing your first version tag:

### 1. TestPyPI (`https://test.pypi.org`)
1. Sign in to `https://test.pypi.org` (enable 2FA if not already enabled).
2. Navigate to **Your Account → Publishing → Add a new pending publisher** (`https://test.pypi.org/manage/account/publishing/`).
3. Enter the exact values below:
   - **PyPI Project Name**: `predgate`
   - **Owner**: `taqi-stark` (your GitHub username or org)
   - **Repository name**: `predgate`
   - **Workflow name**: `release.yml`
   - **Environment name**: `testpypi`
4. Click **Add**.

### 2. PyPI (`https://pypi.org`)
1. Sign in to `https://pypi.org` (enable 2FA).
2. Navigate to **Your Account → Publishing → Add a new pending publisher** (`https://pypi.org/manage/account/publishing/`).
3. Enter the exact values below:
   - **PyPI Project Name**: `predgate`
   - **Owner**: `taqi-stark`
   - **Repository name**: `predgate`
   - **Workflow name**: `release.yml`
   - **Environment name**: `pypi`
4. Click **Add**.

### 3. GitHub Repository Environments
In your GitHub repository (**Settings → Environments**):
1. Create environment **`testpypi`** (no required reviewers needed if you want every `v*` tag to auto-publish to TestPyPI).
2. Create environment **`pypi`** and optionally enable **Required reviewers** (add yourself) so production PyPI publication waits for one-click manual approval after you inspect TestPyPI.

---

## Cutting a Release

1. Ensure `pyproject.toml`, `src/predgate/__init__.py`, `CHANGELOG.md`, and `docs/RELEASE_NOTES_<tag>.md` reflect the target version.
2. Verify benchmark freshness (`bash scripts/check_benchmark_fresh.sh`) and full CI checks.
3. Tag and push when ready:
   ```bash
   git tag -a v0.1.0-alpha -m "PredGate v0.1.0-alpha"
   git push origin v0.1.0-alpha
   ```
4. The `github-release` job in `.github/workflows/release.yml` automatically unpacks `docs/benchmark_per_seed_<commit>.json.gz` and attaches `benchmark_per_seed_<commit>.json` alongside the `dist/*` wheel and sdist to the created GitHub Release.
