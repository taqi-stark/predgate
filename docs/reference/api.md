# REST API Reference

The FastAPI application (`predgate.api.main:app`, started via `predgate serve` or Docker Compose) persists runs, check results, approvals, settings, users, and append-only audit logs in SQLite (`DATABASE_URL`, default `sqlite:///predgate.db`). Interactive Swagger docs are served at `http://localhost:8000/docs`.

## Endpoints

| Method | Path | Auth / Role | Purpose |
|---|---|---|---|
| `GET` | `/health` | Public | Liveness check (`{"status": "ok", "service": "predgate-api"}`) |
| `POST` | `/runs` | `engineer`, `reviewer`, `admin` | Ingest a `predgate run` payload |
| `GET` | `/runs` | Public | List runs (query filters: `status`, `branch`, `project`, `limit`, `offset`) |
| `GET` | `/runs/{run_id}` | Public | Full run detail including check results, lineage, and approvals |
| `GET` | `/runs/{run_id}/slices` | Public | Slice explorer payload including `inconclusive_entries` and `on_inconclusive` |
| `GET` | `/compare?a=<id>&b=<id>` | Public | Side-by-side metric deltas between two runs |
| `POST` | `/runs/{run_id}/approve` *(alias `/approval`)* | `reviewer`, `admin` | Record approval or rejection (`decision`, `comment`). Overriding a `FAIL` run requires `admin` role and a justification `>= 20` characters. |
| `POST` | `/runs/{run_id}/promote` | `admin` | Promote a run to champion (requires override approval if run status is `fail`) |
| `GET` | `/models` | Public | Passing/warning runs ordered newest-first with champion marker |
| `GET` | `/settings?project=<name>` | Public | Read stored project settings |
| `PUT` | `/settings?project=<name>` | `admin` | Update stored project settings |
| `POST` | `/users` | `admin` | Create or rotate an API user and return the bearer token once |
| `GET` | `/users` | `admin` | List provisioned users (never exposes token hashes) |
| `GET` | `/audit` | `admin` | Read append-only audit log (`limit`, `offset`) |
| `GET` | `/bench` | Public | Return `docs/benchmark_results.json` summary |

## Authentication Header

Send bearer tokens on protected endpoints using the standard HTTP `Authorization` header:

```bash
curl -X POST http://localhost:8000/runs \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d @predgate_output/payload.json
```

All allowed and denied (`401`, `403`, `400`) requests on protected endpoints are recorded in the append-only `audit_log` table. See [Security & HTTPS](../security.md) for deployment hardening.
