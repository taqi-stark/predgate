# Releasing PredGate (PyPI Trusted Publishing / OIDC)

PredGate publishes to **TestPyPI** and **PyPI** automatically on version tags (`v*`) via [`.github/workflows/release.yml`](https://github.com/taqi-stark/predgate/blob/main/.github/workflows/release.yml) using **OpenID Connect (OIDC) Trusted Publishing** — no long-lived API tokens or repository secrets are required.

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

### 3. GitHub Repository Environments & Solo-Maintainer Branch Protection
In your GitHub repository:

1. **Environments (**Settings → Environments**)**:
   - Create environment **`testpypi`** (no required reviewers needed if you want every `v*` tag to auto-publish to TestPyPI).
   - Create environment **`pypi`** and optionally enable **Required reviewers** (add yourself) so production PyPI publication waits for one-click manual approval after you inspect TestPyPI.
2. **Solo-Maintainer Branch Protection (**Settings → Branches → Branch protection rules** for `main`)**:
   - Enable **Require status checks to pass before merging** and select **`Run PredGate`** (plus CI checks: `Lint & Type Check`, `Tests & Coverage`, `Benchmark Freshness`, `Dashboard Build`).
   - **Do NOT enable** *Require a pull request before merging → Require approvals* or *Require review from Code Owners* while maintaining the repository solo — GitHub does not allow a solo author to approve their own PR, so enabling required reviews or Code Owner enforcement blocks normal merges. Keep [`.github/CODEOWNERS`](https://github.com/taqi-stark/predgate/blob/main/.github/CODEOWNERS) in place and enable Code Owner review enforcement once a second maintainer/reviewer joins.

---

## Cutting a Release

1. Ensure `pyproject.toml`, `src/predgate/__init__.py`, `CHANGELOG.md`, and `docs/RELEASE_NOTES_<tag>.md` reflect the target version. Note that `.github/workflows/release.yml` verifies via PEP 440 normalization that the pushed git tag matches `pyproject.toml` (e.g., `v0.1.0-alpha` normalizes to `0.1.0a0`) and fails the release job on mismatch.
2. Verify benchmark freshness (`bash scripts/check_benchmark_fresh.sh`) and full CI checks.
3. Tag and push when ready:
   ```bash
   git tag -a v0.1.0-alpha -m "PredGate v0.1.0-alpha"
   git push origin v0.1.0-alpha
   ```
4. The `github-release` job in `.github/workflows/release.yml` automatically unpacks `docs/benchmark_per_seed_<commit>.json.gz` and attaches `benchmark_per_seed_<commit>.json` alongside the `dist/*` wheel and sdist to the created GitHub Release.
