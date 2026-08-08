#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/arduino/unoq_face_recognition"
COMPOSE_FILE="${PROJECT_DIR}/standalone-ai/docker-compose.yml"

docker compose -f "${COMPOSE_FILE}" down
echo "Standalone AI runtime stopped."
