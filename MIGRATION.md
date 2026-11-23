# Migration Guide: `ModelGate` → `PredGate` (`predgate`)

> **Why the rename?** The PyPI package name `modelgate` is already taken by an unrelated package (`modelgate` 0.2.0, an LLM provider adapter layer). To ship cleanly on PyPI with a single, unambiguous name across PyPI, Python imports, and the CLI, this project has been renamed to **PredGate** (`predgate`, verified available on PyPI).

---

## Summary of Changes

| Surface | Old (`ModelGate`) | New (`PredGate`) |
|---|---|---|
| **PyPI Package** | `modelgate` | `predgate` |
| **Python Import** | `import modelgate` | `import predgate` |
| **CLI Command** | `modelgate <subcommand>` | `predgate <subcommand>` |
| **Config Class** | `modelgate.config.ModelGateConfig` | `predgate.config.PredGateConfig` |
| **Config File** | `modelgate.yaml` | `predgate.yaml` |
| **Output Directory** | `modelgate_output/` | `predgate_output/` |
| **SQLite Database** | `modelgate.db` | `predgate.db` |
| **GitHub Workflow** | `.github/workflows/modelgate.yml` | `.github/workflows/predgate.yml` |
| **GitHub Check Run Name** | `Run ModelGate` | `Run PredGate` |
| **GitHub Commit Status Context** | `modelgate` | `predgate` |
| **PR Comment Marker** | `<!-- modelgate-verdict -->` | `<!-- predgate-verdict -->` |
| **Action Input (Pinned Package)** | `modelgate-spec` | `predgate-spec` |
| **CODEOWNERS Teams** | `@your-org/modelgate-admins` | `@your-org/predgate-admins` |
| **Env: Admin Bootstrap Token** | `MODELGATE_ADMIN_TOKEN` | `PREDGATE_ADMIN_TOKEN` *(legacy fallback supported)* |
| **Env: Runner API Token** | `MODELGATE_API_TOKEN` | `PREDGATE_API_TOKEN` *(legacy fallback supported)* |
| **Env: CORS Origins** | `MODELGATE_CORS_ORIGINS` | `PREDGATE_CORS_ORIGINS` *(legacy fallback supported)* |
| **Dashboard Token Storage Key** | `modelgate_api_token` | `predgate_api_token` |
| **Benchmark JSON Keys** | `modelgate_detection_rate`, `modelgate_ci_95`, `modelgate_wins` | `predgate_detection_rate`, `predgate_ci_95`, `predgate_wins` |
| **Docker Compose Volumes** | `modelgate-data`, `modelgate-db` | `predgate-data`, `predgate-db` |

---

## Step-by-Step Upgrade Checklist

1. **Reinstall the package**:
   ```bash
   pip uninstall -y modelgate
   pip install -e ".[dev]"   # or: pip install predgate
   ```
2. **Rename your project configuration file**:
   ```bash
   git mv modelgate.yaml predgate.yaml
   ```
3. **Update Python imports** in any custom scripts or tests:
   ```python
   # Before
   from modelgate.config import ModelGateConfig, load_config

   # After
   from predgate.config import PredGateConfig, load_config
   ```
4. **Update CI workflows and GitHub branch protection**:
   - Rename `.github/workflows/modelgate.yml` → `.github/workflows/predgate.yml`.
   - Update required branch protection status check from `Run ModelGate` to `Run PredGate`.
   - Update `.github/CODEOWNERS` handles if referencing `@<org>/modelgate-admins`.
   - If passing an explicit pinned package input to `action.yml`, update `modelgate-spec` → `predgate-spec`.
5. **Update environment variables & local artifacts**:
   - Prefer `PREDGATE_ADMIN_TOKEN`, `PREDGATE_API_TOKEN`, and `PREDGATE_CORS_ORIGINS` (temporary fallback to `MODELGATE_*` environment variables is retained for zero-downtime migration).
   - If reusing an existing SQLite DB, rename `modelgate.db` → `predgate.db` (or set `DATABASE_URL`).
   - Delete stale `modelgate_output/` folders; reports are now written to `predgate_output/`.
6. **Re-run the gate**:
   ```bash
   predgate run --config predgate.yaml
   ```
