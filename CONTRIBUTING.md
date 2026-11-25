# Contributing to PredGate

Thank you for your interest in contributing to PredGate!

## Development Setup

```bash
git clone https://github.com/taqi-stark/predgate.git
cd predgate
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
python demo/train.py
.venv/bin/predgate doctor --config predgate.yaml
```

Optional dashboard development (Node.js 20+):

```bash
cd dashboard
npm ci
npm run dev    # or npm run build
```

Optional pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

## Running Tests & Quality Checks

```bash
# Lint & format check
.venv/bin/ruff check src tests scripts
.venv/bin/ruff format --check src tests

# Static type check
.venv/bin/mypy src/predgate --ignore-missing-imports

# Unit tests with coverage (CI requires >= 85%)
.venv/bin/pytest tests/unit/ -v

# Dashboard production build
(cd dashboard && npm run build)

# Benchmark freshness check (ensures docs/benchmark_results.json matches src/)
bash scripts/check_benchmark_fresh.sh
```

## Style & Statistical Integrity Rules

- Follow Ruff (`line-length = 100`, target Python 3.10+) and strict Mypy checks.
- **No silent threshold tuning**: Never tune default check thresholds on benchmark seeds and report results on the same seeds. Any change to `src/` must either preserve benchmark freshness or re-run `scripts/bench_public.py` on fresh, unused seeds and update `docs/benchmark_results.json` and `docs/benchmark.md`.
- Keep all benchmark tables generated strictly from JSON (no hand-typed benchmark numbers).

## How to Add a New Quality Check

1. **Configuration model**: Add a Pydantic v2 config class (with `enabled: bool = True` and sensible defaults) in `src/predgate/config.py`, attach it to `ChecksConfig`, and add explicit entries to `PRESET_CONFIGS` (`strict`, `balanced`, `quiet`) while keeping `balanced` identical to the defaults.
2. **Check implementation**: Create `src/predgate/checks/<check_name>.py` exporting `run_<check_name>_check(...) -> CheckResult`. Return `status` in `{"pass", "warn", "fail", "skip"}`, clear `summary`, numeric `metrics`, and structured `details`.
3. **Wire into CLI & pipeline**: Invoke the check in `src/predgate/cli.py` inside `predgate run`, and update `src/predgate/report.py` if the check emits callout banners or notes.
4. **Dashboard tab**: Add the check name to `tabs` and `tabLabels` in `dashboard/src/pages/RunDetailPage.tsx` and rebuild `dashboard/dist/`.
5. **Unit tests**: Add deterministic unit tests covering `pass`, `warn`, `fail`, `skip`, and edge cases in `tests/unit/test_checks.py` and config validation tests in `tests/unit/test_config.py`.
6. **Documentation & benchmark**: Document the new config fields in `docs/usage.md`, update the "What it catches / What it doesn't" table in `README.md`, and re-run fresh-seed benchmarks per `scripts/check_benchmark_fresh.sh`.
