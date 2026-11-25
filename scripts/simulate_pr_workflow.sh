#!/usr/bin/env bash
# Locally simulate `.github/workflows/predgate.yml` on a pull_request checkout
# using a temporary bare remote + clean clone (no network push, no credentials).
#
# Usage:
#   bash scripts/simulate_pr_workflow.sh <base_branch> <head_branch> [pr_number]
set -euo pipefail

BASE_REF="${1:-main}"
HEAD_REF="${2:-feature/tune-learning-rate}"
PR_NUM="${3:-101}"
REPO_ROOT="$(git rev-parse --show-toplevel)"
SIM_ROOT="$(mktemp -d /tmp/predgate-pr-sim-XXXXXX)"
trap 'rm -rf "$SIM_ROOT"' EXIT

echo "=== Simulating GitHub Actions predgate.yml (base=${BASE_REF}, head=${HEAD_REF}, PR=#${PR_NUM}) ==="

# 1. Create a local bare origin clone + checkout PR head (mirroring actions/checkout@v4 fetch-depth: 0)
git clone --bare --quiet "$REPO_ROOT" "$SIM_ROOT/origin.git"
git clone --quiet "$SIM_ROOT/origin.git" "$SIM_ROOT/workspace"
cd "$SIM_ROOT/workspace"
git checkout --quiet "$HEAD_REF"

# 2. Step: Fetch trusted base branch & install pinned PredGate (using offline wheelhouse from .venv if available)
echo "--- [Step 1] Fetch trusted base branch & archive pinned PredGate ---"
git fetch --no-tags origin "+refs/heads/${BASE_REF}:refs/remotes/origin/${BASE_REF}"
BASE_COMMIT=$(git rev-parse --verify "origin/${BASE_REF}^{commit}")
TRUSTED_DIR="$SIM_ROOT/predgate-trusted"
mkdir -p "$TRUSTED_DIR"
git archive "${BASE_COMMIT}" | tar -x -C "$TRUSTED_DIR"
echo "Base commit archived: ${BASE_COMMIT}"

# Use an isolated venv copying site-packages via --system-site-packages from repo .venv OR fresh install
python3 -m venv "$SIM_ROOT/runner-venv"
# Install trusted base-branch PredGate package without touching untrusted PR head src/
"$SIM_ROOT/runner-venv/bin/pip" install -q --no-build-isolation --no-deps "$TRUSTED_DIR" 2>/dev/null || true
# Use repo .venv deps + trusted base-branch src/ on PYTHONPATH to simulate exact runner python 3.12
export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
export PYTHONPATH="${TRUSTED_DIR}/src"

# 3. Step: Verify champion baseline from base branch
echo "--- [Step 2] Verify champion baseline from origin/${BASE_REF} ---"
if git diff --name-only "origin/${BASE_REF}...HEAD" | grep -Eq '^(champion/|champion\.dvc$)'; then
  echo "::notice::PR modifies champion baseline; evaluating against trusted base-branch champion."
fi

PINNED_SHA=$(git show "origin/${BASE_REF}:champion.dvc" 2>/dev/null | python -c "import sys, yaml; d = yaml.safe_load(sys.stdin) or {}; print((d.get('outs') or [{}])[0].get('sha256', ''))" 2>/dev/null || true)
if [ -z "$PINNED_SHA" ]; then
  PINNED_SHA=$(git show "origin/${BASE_REF}:champion/metadata.json" 2>/dev/null | python -c "import sys, json; print(json.load(sys.stdin).get('model_sha256', ''))" 2>/dev/null || true)
fi
if [ -z "$PINNED_SHA" ]; then
  echo "::error::Could not determine pinned champion SHA-256 from origin/${BASE_REF}; failing closed" >&2
  exit 1
fi
git checkout "origin/${BASE_REF}" -- champion champion.dvc
python -m predgate.cli baseline verify --dir champion --dvc champion.dvc --pinned-sha256 "$PINNED_SHA"

# 4. Step: Generate demo data
echo "--- [Step 3] Generate demo data ---"
python demo/train.py

# 5. Step: Import verified champion baseline into MLflow
echo "--- [Step 4] Import verified champion baseline into MLflow ---"
python -m predgate.cli baseline import --config predgate.yaml --dir champion --pinned-sha256 "$PINNED_SHA"

# 6. Step: Run PredGate
echo "--- [Step 5] Run PredGate ---"
export CHAMPION_PINNED_SHA256="$PINNED_SHA"
set +e
python -m predgate.cli run \
  --config predgate.yaml \
  --branch "$HEAD_REF" \
  --pr "$PR_NUM" \
  --output predgate_output
GATE_EXIT=$?
set -e

echo "--- [Step 6] Inspect generated workflow artifacts ---"
ls -lh predgate_output/
VERDICT_STATUS=$(python -c "import json; print(json.load(open('predgate_output/verdict.json'))['status'])")
echo "Simulated workflow verdict=${VERDICT_STATUS} (gate exit=${GATE_EXIT})"
if [ "$GATE_EXIT" -ne 0 ]; then
  echo "Final step 'Fail if verdict is FAIL' would exit 1 (blocking PR merge)."
else
  echo "Final step 'Fail if verdict is FAIL' would exit 0 (allowing PR merge)."
fi
