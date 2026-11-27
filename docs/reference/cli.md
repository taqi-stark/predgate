# CLI Reference

All commands are available via `predgate <command>` or `python -m predgate.cli <command>`.

## `predgate init`

Scaffold a starter `predgate.yaml` and training script in `PATH` (default `.`).

```bash
predgate init [PATH]
```

## `predgate run`

Train the candidate model entrypoint, load the verified champion, execute all configured checks, write reports (`verdict.json`, `pr_comment.md`, `report.html`, `payload.json`), log lineage to MLflow, and post to the API if running. Exits `0` on `PASS`/`WARN` and `1` on `FAIL`.

```bash
predgate run [OPTIONS]
  -c, --config PATH    Config file path (default: predgate.yaml)
  --preset TEXT        Optional preset override: strict, balanced, or quiet
  -b, --branch TEXT    Branch name recorded in payload
  --pr INT             Pull request number recorded in payload
  -o, --output PATH    Output directory (default: predgate_output)
```

## `predgate doctor`

Verify Python version (`>= 3.10`), `git`, `dvc`, `mlflow`, `predgate.yaml` validity, model entrypoint import, Pandera schema import, and dataset CSV paths. Prints actionable fixes for any failure.

```bash
predgate doctor --config predgate.yaml [--json]
```

## `predgate baseline`

Export, verify, or import a persisted champion baseline (`model.pkl`, `metadata.json`, `metrics.json`, and `champion.dvc`). Verification always fails closed on hash mismatch, corruption, or missing manifest.

```bash
predgate baseline export --config predgate.yaml --dir champion
predgate baseline verify --dir champion --dvc champion.dvc --pinned-sha256 <SHA256>
predgate baseline import --config predgate.yaml --dir champion --pinned-sha256 <SHA256>
```

## `predgate users`

Provision and list API users (`engineer`, `reviewer`, `admin`). Plaintext bearer tokens are printed once on creation; only SHA-256 digests are stored.

```bash
predgate users add <NAME> --role <engineer|reviewer|admin>
predgate users list
```

## `predgate compare`

Print a side-by-side metric delta table for two API run IDs.

```bash
predgate compare <RUN_A> <RUN_B>
```

## `predgate promote`

Promote an MLflow run's model to the `@champion` alias (rejects unapproved `FAIL` runs when the API reports a failing verdict).

```bash
predgate promote <RUN_ID> --config predgate.yaml
```

## `predgate reproduce`

Inspect recorded lineage (`seed`, `git_sha`, `dvc_hash`, original metrics) for an MLflow run and check out DVC-tracked data when present.

```bash
predgate reproduce <RUN_ID> --config predgate.yaml
```

## `predgate bench`

Run the synthetic 7-fault benchmark suite (`predgate.bench.run_benchmark`) and regenerate `docs/benchmark_results.json` and `docs/benchmark.md`.

```bash
predgate bench
```

## `predgate serve`

Start the FastAPI backend server (`uvicorn predgate.api.main:app`, OpenAPI docs at `/docs`). Note: `predgate serve` serves the REST API only; run the React dashboard via [Docker Compose](../guides/docker.md) or `cd dashboard && npm run dev`.

```bash
predgate serve --host 0.0.0.0 --port 8000
```
