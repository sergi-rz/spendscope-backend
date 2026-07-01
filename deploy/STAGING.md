# Staging backend (for /enrich eval)

A second, isolated instance of the backend so the premium-gated `/enrich` flow (tickets, Amazon
screenshots) can be exercised with `REQUIRE_PREMIUM=false` **without touching production**.

- Prod:    `spendscope.hostsrg.com`            → uvicorn `127.0.0.1:8077` → `/opt/spendscope`
- Staging: `staging.spendscope.hostsrg.com`    → uvicorn `127.0.0.1:8078` → `/opt/spendscope-staging`

Prod and staging share nothing except the machine. Staging uses its own checkout, `.env`, port and
(recommended) its own database, so its request_log and any test data never mix with prod.

## One thing only you can do

Create a DNS **A record**: `staging.spendscope.hostsrg.com` → the VPS IP (same IP as prod).

## VPS setup (run once, as a sudoer on the VPS)

```bash
# 1. Checkout (own branch optional; main is fine for staging)
sudo git clone https://github.com/sergi-rz/spendscope-backend.git /opt/spendscope-staging
cd /opt/spendscope-staging
sudo python3 -m venv .venv
sudo .venv/bin/pip install -r requirements.txt

# 2. (recommended) a separate DB so staging never pollutes prod stats
sudo mysql -e "CREATE DATABASE IF NOT EXISTS spendscope_staging CHARACTER SET utf8mb4;
               CREATE USER IF NOT EXISTS 'spendscope_stg'@'127.0.0.1' IDENTIFIED BY 'CHANGE_ME';
               GRANT ALL ON spendscope_staging.* TO 'spendscope_stg'@'127.0.0.1'; FLUSH PRIVILEGES;"
# (schema is auto-created by the app on first startup — init_schema())

# 3. .env — copy prod's and flip the gate + DB. Keep the SAME LLM keys as prod.
sudo cp /opt/spendscope/.env /opt/spendscope-staging/.env
sudoedit /opt/spendscope-staging/.env
#   REQUIRE_PREMIUM=false
#   DATABASE_URL=mysql+pymysql://spendscope_stg:CHANGE_ME@127.0.0.1:3306/spendscope_staging?charset=utf8mb4
#   ENVIRONMENT=staging

sudo chown -R www-data:www-data /opt/spendscope-staging

# 4. systemd service on port 8078
sudo cp deploy/spendscope-staging.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now spendscope-staging
curl -fsS http://127.0.0.1:8078/health && echo "  <- staging up"

# 5. Apache vhost + TLS (needs the DNS record above to resolve first)
sudo cp deploy/apache-spendscope-staging.conf /etc/apache2/sites-available/
sudo a2ensite apache-spendscope-staging && sudo apache2ctl configtest
sudo systemctl reload apache2
sudo certbot --apache -d staging.spendscope.hostsrg.com
```

## Redeploy staging later

```bash
cd /opt/spendscope-staging && sudo git pull --ff-only origin main \
  && sudo .venv/bin/pip install -q -r requirements.txt \
  && sudo systemctl restart spendscope-staging
```

## Verify from outside

```bash
curl -fsS -H "User-Agent: Mozilla/5.0" https://staging.spendscope.hostsrg.com/api/v1/health
```

Once this returns `{"status":"ok",...}`, point the eval harness at it (`BASE` in `scripts/llm_eval.py`)
to exercise `/enrich` with the real ticket photo + Carrefour PDF already in `samples/`.
