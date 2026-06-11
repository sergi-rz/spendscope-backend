# SpendScope Backend

AI proxy for the [SpendScope](https://github.com/sergi-rz/spendscope-app) iOS app. It exposes
three endpoints that sit between the app and the LLM providers, so the app never builds prompts
or talks to a model directly (SPECS §11).

- `POST /api/v1/parse` — universal bank-statement parsing (any format → canonical transactions). Free.
- `POST /api/v1/categorize` — AI categorization fallback when no on-device rule matches. Free.
- `POST /api/v1/enrich` — receipt → line items. **Premium** (validated against RevenueCat).
- `GET /health` — liveness probe.

## Design principles (from the spec)

- **The app sends structured data and gets structured data back.** Prompts live here, server-side.
- **No user data is stored.** Statements, transactions and receipts are processed and discarded.
  MySQL holds only a categorization cache, per-user rate-limit counters, and operational metrics.
- **Provider routing with fallback.** Primary: nan.builders (EU, zero prompt logging). Fallback:
  OpenAI. Both are OpenAI-compatible Chat Completions endpoints. Which one answered (and why a
  fallback happened) is logged per request (SPECS §11.5).
- **Degrades gracefully.** If MySQL is down, cache/rate-limit/logging become no-ops and the API
  keeps serving. If a provider fails, the next one is tried.

## Stack

Python 3.11+ · FastAPI · uvicorn · httpx · SQLAlchemy + PyMySQL · PyMuPDF (PDF) · openpyxl (Excel).

## Local development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

cp .env.example .env          # fill in PRIMARY_API_KEY / FALLBACK_API_KEY (leave DATABASE_URL empty to skip MySQL)
uvicorn app.main:app --reload --port 8077
```

Then `curl http://127.0.0.1:8077/health`.

### Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite mocks the LLM and RevenueCat and forces `DATABASE_URL=""`, so it runs offline with no
database.

## Configuration

All settings come from environment variables / `.env` (see `.env.example`). Key ones:

| Var | Purpose |
|---|---|
| `PRIMARY_BASE_URL` / `PRIMARY_API_KEY` / `PRIMARY_MODEL_TEXT` / `PRIMARY_MODEL_VISION` | nan.builders provider |
| `FALLBACK_BASE_URL` / `FALLBACK_API_KEY` / `FALLBACK_MODEL_TEXT` / `FALLBACK_MODEL_VISION` | OpenAI fallback |
| `DATABASE_URL` | `mysql+pymysql://user:pass@host:3306/spendscope?charset=utf8mb4` (empty = no DB) |
| `REQUIRE_PREMIUM` | `true` in prod; gates `/enrich` |
| `REVENUECAT_SECRET_KEY` | RevenueCat **server secret** (`sk_...`), never the public iOS key |
| `RATE_LIMIT_*` | per-user fixed-window limits |

## Deployment (Hetzner VPS, Apache, Let's Encrypt)

Files in `deploy/`:

1. **MySQL**: `mysql < deploy/schema.sql` (or let the app create the tables on first boot).
2. **App**: clone to `/var/www/spendscope`, create `.venv`, `pip install -r requirements.txt`,
   write `/var/www/spendscope/.env`.
3. **Service**: `deploy/spendscope.service` → `/etc/systemd/system/`, then
   `systemctl enable --now spendscope` (uvicorn on `127.0.0.1:8077`).
4. **Apache**: `deploy/apache-spendscope.conf` reverse-proxies `spendscope.hostsrg.com` → `:8077`.
   Enable mods (`proxy proxy_http headers ssl`), then `certbot --apache` for TLS.

The app's `Config.swift` already points at `https://spendscope.hostsrg.com/api/v1`; flip
`useMockAPI` to `false` once this is live.

## Endpoint contracts

See `SPECS.md` §11 in the app repo, and `app/schemas.py` here — the request/response models are
the source of truth and match the app's `APIModels.swift` field-for-field (snake_case on the wire).
