#!/bin/bash
set -euo pipefail

PROJECT_DIR="/opt/spendscope"
LOG_DIR="/var/log/www/spendscope"
LOG_FILE="$LOG_DIR/deploy.log"
LOCK_FILE="/var/www/crons/deploys/deploy_spendscope.lock"
SERVICE="spendscope"

export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
export HOME=/home/www-data

KEY=/home/www-data/.ssh/id_ed25519_spendscope
KNOWN=/home/www-data/.ssh/known_hosts
OPTS="-o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"
export GIT_SSH_COMMAND="ssh -i $KEY -o UserKnownHostsFile=$KNOWN $OPTS"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "==== Deploy started: $(date -Iseconds) ===="

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "ERROR: another deploy is already running"
    exit 1
fi

cd "$PROJECT_DIR"

echo "-- git pull --"
git pull --ff-only origin main

echo "-- sync dependencies --"
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements.txt

echo "-- restart $SERVICE --"
sudo systemctl restart "$SERVICE"

echo "-- health check --"
sleep 2
if curl -fsS http://127.0.0.1:8077/health > /dev/null; then
    echo "✓ Deployed $(git rev-parse --short HEAD) — backend healthy"
else
    echo "ERROR: health check failed after restart"
    exit 1
fi

echo "==== Deploy finished OK: $(date -Iseconds) ===="
