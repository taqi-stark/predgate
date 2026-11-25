# Drop-In GitHub Workflow

Copy [`predgate.yml`](predgate.yml) to `.github/workflows/predgate.yml` in your ML repository.

1. Update `setup-data` (e.g., `python train.py` or `dvc pull`) and `paths` to match your repository layout.
2. Commit your initial champion baseline (`predgate baseline export --config predgate.yaml --dir champion`).
3. Add `/champion* @<your-github-username>` to `.github/CODEOWNERS`.
4. In GitHub **Settings → Branches → Branch protection rules**, require the **`Run PredGate`** status check and Code Owner reviews.
