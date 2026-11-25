#!/usr/bin/env bash
# Verify that docs/benchmark_results.json was evaluated on the exact current src/ tree.
# Fails with exit code 1 if `git diff <recorded-hash>..HEAD -- src/` is non-empty
# or if there are uncommitted changes under src/.
set -euo pipefail

JSON_PATH="${1:-docs/benchmark_results.json}"

if [ ! -f "$JSON_PATH" ]; then
  echo "::error::Benchmark results file not found: $JSON_PATH" >&2
  exit 1
fi

RECORDED_HASH=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1])).get('commit_hash', ''))" "$JSON_PATH")

if [ -z "$RECORDED_HASH" ] || [ "$RECORDED_HASH" = "unknown" ]; then
  echo "::error::No valid commit_hash recorded in $JSON_PATH" >&2
  exit 1
fi

if ! git cat-file -e "${RECORDED_HASH}^{commit}" 2>/dev/null; then
  echo "::error::Recorded benchmark commit '${RECORDED_HASH}' not found in git history (ensure checkout uses fetch-depth: 0)." >&2
  exit 1
fi

SRC_DIFF=$(git diff --name-only "${RECORDED_HASH}..HEAD" -- src/)
if [ -n "$SRC_DIFF" ]; then
  echo "::error::Benchmark is stale! src/ changed since recorded commit ${RECORDED_HASH}:" >&2
  echo "$SRC_DIFF" >&2
  echo "Re-run scripts/bench_public.py on fresh seeds and commit updated docs/benchmark_results.json and docs/benchmark.md." >&2
  exit 1
fi

UNCOMMITTED_SRC=$(git status --porcelain -- src/)
if [ -n "$UNCOMMITTED_SRC" ]; then
  echo "::error::Uncommitted changes detected under src/ relative to recorded commit ${RECORDED_HASH}:" >&2
  echo "$UNCOMMITTED_SRC" >&2
  exit 1
fi

echo "OK: Benchmark results (${JSON_PATH}) are fresh for src/ (recorded commit ${RECORDED_HASH}, git diff ${RECORDED_HASH}..HEAD -- src/ is empty)."
