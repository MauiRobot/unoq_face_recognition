#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/arduino/unoq_face_recognition"
COMPOSE_FILE="${PROJECT_DIR}/standalone-ai/docker-compose.yml"

echo "=== Containers ==="
docker compose -f "${COMPOSE_FILE}" ps

echo
echo "=== AI runner recent log ==="
docker logs --tail 25 unoq-face-ai-runner 2>&1 || true

echo
echo "=== Camera sender recent log ==="
docker logs --tail 25 unoq-face-ai-camera 2>&1 || true
