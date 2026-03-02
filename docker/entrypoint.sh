#!/bin/bash
set -e

echo "╔══════════════════════════════════════════╗"
echo "║  Myanmar ASR — Presentation + Demo       ║"
echo "║                                          ║"
echo "║  Slides:  http://localhost/              ║"
echo "║  Demo:    http://localhost/demo           ║"
echo "╚══════════════════════════════════════════╝"

# Defaults for env vars supervisord will pass through
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://host.docker.internal:5050}"
export MLFLOW_S3_ENDPOINT_URL="${MLFLOW_S3_ENDPOINT_URL:-http://host.docker.internal:9002}"
export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-minioadmin}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-minioadmin123}"
export CLOUD_INFERENCE_URL="${CLOUD_INFERENCE_URL:-}"

# Remove default nginx site if present
rm -f /etc/nginx/sites-enabled/default 2>/dev/null || true
ln -sf /etc/nginx/sites-available/default /etc/nginx/sites-enabled/default

exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
