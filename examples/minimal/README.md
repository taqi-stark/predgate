# Minimal PredGate Example (5-Minute Quickstart)

Self-contained example using scikit-learn's built-in public Diabetes dataset. Copy this folder into any repository and run:

```bash
# 1. Install PredGate
python3 -m venv .venv && source .venv/bin/activate
pip install predgate==0.1.0a0   # or: pip install -e ../..

# 2. Generate train / reference / test CSVs
python train.py

# 3. Verify environment, config, schema, and data paths
predgate doctor --config predgate.yaml

# 4. Run PredGate (first run skips champion-comparison checks)
predgate run --config predgate.yaml

# 5. Export a verified champion baseline, then re-run full 8-check gate
predgate baseline export --config predgate.yaml --dir champion
predgate run --config predgate.yaml
```

Inspect `predgate_output/pr_comment.md` and `predgate_output/report.html`.
Try `--preset strict` or `--preset quiet` to test threshold presets.

Files included:
- `train.py` — public dataset split generator + `build_model` entrypoint
- `schema.py` — Pandera data contract schema
- `predgate.yaml` — ready-to-edit quality gate config
