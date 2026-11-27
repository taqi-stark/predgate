# Security & HTTPS Deployment

## Zero Telemetry

PredGate collects **no telemetry**. It never sends analytics, crash reports, model metrics, or usage pings to any external server.

## Champion Integrity & Pickle Deserialization Risk

Persisted champion baselines (`champion/model.pkl`) currently use Python `pickle`, which can execute arbitrary code if untrusted files are deserialized. PredGate enforces four defense-in-depth controls in `v0.1.0-alpha`:

1. **Fail-closed SHA-256 verification:** `predgate baseline verify` and `load_candidate_and_champion()` verify `model.pkl` against `metadata.json`, `champion.dvc`, and `CHAMPION_PINNED_SHA256` **before** unpickling. Missing manifests, hash mismatches, or corrupt pickles raise a hard error — PredGate never silently retrains the champion from the candidate entrypoint.
2. **Trusted base-branch installation:** `.github/workflows/predgate.yml` and `action.yml` install PredGate from the committed base branch (`git archive`) or an explicit pinned `predgate-spec` — never `pip install -e .` from an untrusted PR head.
3. **Never use `pull_request_target`:** Because candidate training executes PR code (`model.entrypoint`), always trigger PredGate with `on: pull_request` so fork PRs run with a read-only `GITHUB_TOKEN` and zero repository secrets.
4. **CODEOWNERS + Branch Protection:** Govern changes to `champion/` and `champion.dvc` via GitHub branch protection and `.github/CODEOWNERS` rather than custom CI bypass flags.

Safer model serialization (`skops` / ONNX) is planned for a future release.

## API Authentication, RBAC & Audit Log

- **No default admin token:** PredGate never ships with a hardcoded or default admin token. Provision users explicitly:
  ```bash
  predgate users add alice --role engineer
  predgate users add bob --role reviewer
  predgate users add carol --role admin
  ```
  Or set `PREDGATE_ADMIN_TOKEN` in the API environment at startup to bootstrap an initial `admin` user.
- **Hashed tokens only:** Plaintext bearer tokens are displayed once at creation; only their SHA-256 hex digests are stored in the `users` table.
- **Role-based access control:**
  - `engineer`, `reviewer`, `admin`: `POST /runs`
  - `reviewer`, `admin`: `POST /runs/{id}/approve` (reviewer identity is bound from the authenticated token, not the request body)
  - `admin` only: overriding a `FAIL` run (requires a `>= 20`-character justification), `POST /runs/{id}/promote`, `PUT /settings`, `POST/GET /users`, `GET /audit`
- **Append-only audit trail:** Every allowed and denied (`401`/`403`/`400`) mutating request is recorded in `audit_log` (`GET /audit`, admin only).

## HTTPS / TLS & CORS in Non-Localhost Deployments

Bearer tokens travel in the `Authorization: Bearer <token>` header. Whenever deploying the PredGate API outside `localhost`:

1. Terminate TLS/HTTPS in front of `predgate serve` using a reverse proxy or load balancer (Nginx, Caddy, Envoy, or cloud ALB).
2. Restrict `PREDGATE_CORS_ORIGINS` to explicit `https://` dashboard origins:
   ```bash
   export PREDGATE_CORS_ORIGINS="https://predgate.internal.example.com"
   ```
3. Never expose the unencrypted HTTP port (`8000`) on untrusted networks.

## Reporting a Vulnerability

Please do **not** open a public issue for security vulnerabilities. Report privately via [GitHub Private Vulnerability Reporting](https://github.com/taqi-stark/predgate/security/advisories/new) (see [`SECURITY.md`](https://github.com/taqi-stark/predgate/blob/main/SECURITY.md) for response timelines).
