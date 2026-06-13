#!/bin/bash
# Server-side deploy for the SpendScope backend. Runs ON the VPS, triggered by the
# DCT deploy hook (or manually). Pulls main, syncs deps, restarts the service.
#
# The .env is gitignored, so `git pull` never touches the secrets on the server.
set -euo pipefail

APP_DIR="/opt/spendscope"
SERVICE="spendscope"
BRANCH="main"

cd "$APP_DIR"

# The repo is owned by www-data; allow git to operate regardless of the hook user.
git config --global --add safe.directory "$APP_DIR" 2>/dev/null || true

echo "==> git pull ($BRANCH)"
git fetch --quiet origin "$BRANCH"
git reset --hard "origin/$BRANCH"

echo "==> sync dependencies"
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements.txt

echo "==> restart $SERVICE"
systemctl restart "$SERVICE"

echo "==> health check"
sleep 2
if curl -fsS http://127.0.0.1:8077/health > /dev/null; then
    echo "OK — backend healthy after deploy"
else
    echo "WARNING — health check failed, inspect: journalctl -u $SERVICE -n 40" >&2
    exit 1
fi
