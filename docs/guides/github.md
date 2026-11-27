# GitHub Actions Integration

PredGate provides a reusable composite GitHub Action (`action.yml`) and a reference repository workflow (`.github/workflows/predgate.yml`).

## Recommended Workflow

Copy `examples/github-workflow/predgate.yml` into `.github/workflows/predgate.yml` in your repository:

```yaml
name: PredGate Quality Gate

on:
  pull_request:
    paths:
      - 'model/**'
      - 'src/**'
      - 'data/**'
      - 'champion/**'
      - 'champion.dvc'
      - 'predgate.yaml'

permissions:
  contents: read
  pull-requests: write
  statuses: write

jobs:
  predgate:
    name: Run PredGate
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Run PredGate Quality Gate
        id: gate
        uses: taqi-stark/predgate@v0.1.0-alpha
        with:
          config: 'predgate.yaml'
          python-version: '3.12'
          predgate-spec: 'predgate==0.1.0a0'
          baseline-dir: 'champion'
          setup-data: 'python train.py'
          output-dir: 'predgate_output'
          fail-on-verdict: 'false'

      - name: Upload HTML report
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: predgate-report
          path: predgate_output/report.html

      - name: Fail job if verdict is FAIL
        if: steps.gate.outputs.verdict == 'fail'
        run: exit 1
```

To also post or update the verdict table as a PR comment, copy the `Post PR comment` step from [`examples/github-workflow/predgate.yml`](https://github.com/taqi-stark/predgate/blob/main/examples/github-workflow/predgate.yml).

## Action Inputs & Outputs

| Input | Default | Description |
|---|---|---|
| `config` | `predgate.yaml` | Path to `predgate.yaml` |
| `python-version` | `3.12` | Python version |
| `predgate-spec` | `''` | Pinned pip package spec (e.g. `predgate==0.1.0a0`). When omitted, installs from the committed base branch via `git archive` (never the PR head). |
| `baseline-dir` | `champion` | Directory containing persisted `model.pkl`, `metadata.json`, `metrics.json` |
| `base-ref` | `''` | Optional base branch override (defaults to `github.base_ref`) |
| `pinned-sha256` | `''` | Optional explicit champion SHA-256 (auto-read from base branch when omitted) |
| `setup-data` | `''` | Optional command to prepare datasets when `data.train` is missing and no DVC remote is configured |
| `output-dir` | `predgate_output` | Output directory for reports |
| `fail-on-verdict` | `'true'` | Exit non-zero when verdict is `fail` |

Outputs: `verdict` (`pass`/`warn`/`fail`), `summary`, `report-html`, `pr-comment`.

## Branch Protection & Required Check Name

1. Go to **Settings → Branches → Branch protection rules** for `main`.
2. Enable **Require status checks to pass before merging** and search for **`Run PredGate`** (the GitHub Actions check run produced by `jobs.predgate.name`).
3. **Solo maintainers:** Do **not** enable *Require approvals* or *Require review from Code Owners* until a second reviewer joins, because GitHub blocks solo authors from approving their own PRs.
4. **Teams:** Enable *Require review from Code Owners* and protect your champion folder in `.github/CODEOWNERS`:
   ```text
   /champion* @<your-org>/<ml-reviewers-team>
   ```

## Fork-PR Behavior & Security

- Always trigger on `pull_request` — **never** `pull_request_target` — because candidate training executes PR code.
- On fork PRs, GitHub automatically scopes `GITHUB_TOKEN` to read-only. `Run PredGate` still verifies the base-branch champion SHA-256, executes all checks, uploads `report.html`, and blocks merge on `FAIL`, while PR comment / commit status writes log a non-fatal warning on HTTP 403.
- Require **`Run PredGate`** (the GitHub Actions check run) rather than the legacy `predgate` commit status so fork PRs cannot bypass protection.

## Optional DVC Data Tracking

For larger datasets stored outside Git, track `data/` with DVC and configure remote credentials as repository secrets:

```bash
dvc init
dvc add data
dvc remote add -d storage s3://my-bucket/predgate-data
dvc push
```

In GitHub Actions, export your remote credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, or GCP/Azure equivalents) before running the action; when `.dvc/config` is present and `data.train` is missing, `action.yml` runs `dvc pull` automatically. PredGate also captures `dvc_hash` in lineage so `predgate reproduce <RUN_ID>` can check out the exact dataset snapshot.

## Testing the Workflow Locally

Simulate `.github/workflows/predgate.yml` end-to-end against local branches without pushing:

```bash
bash scripts/simulate_pr_workflow.sh main feature/tune-learning-rate 101
bash scripts/simulate_pr_workflow.sh main bad/degrade-region-north 42
```
