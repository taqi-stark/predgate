# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| `0.1.x-alpha` (`0.1.0a0`) | Yes (best-effort security fixes during alpha) |

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

1. Use **GitHub Private Vulnerability Reporting** via the repository's **Security → Advisories → Report a vulnerability** page:
   `https://github.com/taqi-stark/predgate/security/advisories/new`
2. Include a clear description, reproduction steps or minimal PoC, affected component/version, and potential impact.
3. We aim to acknowledge reports within **72 hours** and provide a remediation plan or status update within **7 days**.

## Security Architecture & Known Risks

### 1. Pickle Deserialization Risk (`champion/model.pkl`)
PredGate currently loads persisted champion baselines using Python `pickle`. Untrusted pickle files can execute arbitrary code upon deserialization.

Mitigations enforced in `v0.1.0-alpha`:
- **Fail-closed SHA-256 verification**: `predgate baseline verify` and `load_candidate_and_champion()` verify `champion/model.pkl` against the base-branch pinned SHA-256 (`champion.dvc` / `champion/metadata.json` / `CHAMPION_PINNED_SHA256`) **before** unpickling and raise a hard error on mismatch, corruption, or missing manifest (no silent retrain or bare-pickle fallback).
- **Trusted base-branch installation**: `.github/workflows/predgate.yml` and `action.yml` install PredGate from the committed base branch (`git archive`) or an explicit pinned `predgate-spec`—never `pip install -e .` from an untrusted PR head.
- **No `pull_request_target`**: Because candidate training executes PR code (`model.entrypoint`), always trigger PredGate with `on: pull_request` (read-only `GITHUB_TOKEN` on forks, no repository secrets exposed). Never use `pull_request_target` with PR code checkout.
- **CODEOWNERS + Branch Protection**: Changes to `/champion/` and `/champion.dvc` must require Code Owner review via GitHub branch protection.
- **Roadmap**: Safer model serialization (`skops` / ONNX) is planned for a future release.

### 2. API Authentication Tokens & RBAC
- PredGate **never** ships with a hardcoded or default admin token.
- Provision users explicitly via `predgate users add <name> --role <engineer|reviewer|admin>` or bootstrap an initial admin at startup via `PREDGATE_ADMIN_TOKEN`.
- Plaintext bearer tokens are displayed once at creation; only their **SHA-256 hex digests** are stored in the SQLite/SQLModel `users` table.
- All allowed and denied (`401`/`403`/`400`) requests on mutating endpoints are recorded in the append-only `audit_log` table (`GET /audit`, admin only).

### 3. HTTPS / TLS Requirement in Production
- Bearer tokens are transmitted in the `Authorization: Bearer <token>` HTTP header.
- Any non-localhost deployment **must** terminate TLS/HTTPS behind a reverse proxy or load balancer (Nginx, Caddy, Envoy, ALB) and restrict `PREDGATE_CORS_ORIGINS` to explicit `https://` dashboard origins. Never expose the API over plain HTTP on untrusted networks.
