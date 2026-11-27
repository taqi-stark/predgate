## Summary
- What changed and why?

## Pre-Merge Checklist
- [ ] `ruff check src tests scripts` and `ruff format --check src tests` pass
- [ ] `mypy src/predgate --ignore-missing-imports` passes
- [ ] `pytest tests/unit/` passes with >= 85% coverage
- [ ] If `dashboard/src/` changed: `(cd dashboard && npm run build)` succeeded
- [ ] If `src/` changed: benchmark freshness verified (`bash scripts/check_benchmark_fresh.sh`) or fresh-seed benchmark re-run with zero post-hoc threshold tuning
- [ ] If `examples/credit-risk-demo/champion/` or `champion.dvc` changed: CODEOWNERS review requested and SHA-256 verified
- [ ] Docs updated (`README.md`, `docs/`, `CHANGELOG.md`) where applicable
