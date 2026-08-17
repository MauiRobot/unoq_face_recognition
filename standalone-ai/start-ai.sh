#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/arduino/unoq_face_recognition"
COMPOSE_FILE="${PROJECT_DIR}/standalone-ai/docker-compose.yml"

echo "UNO Q standalone AI runtime"
echo
echo "Checking cached Docker images..."

docker image inspect ghcr.io/arduino/app-bricks/ei-models-runner:0.12.1 >/dev/null
docker image inspect ghcr.io/arduino/app-bricks/python-apps-base:0.12.0 >/dev/null

echo "Cached images found."
echo
echo "Stopping App Lab containers if present..."

docker rm -f test-of-face-detector-on-camera-main-1 >/dev/null 2>&1 || true
docker rm -f test-of-face-detector-on-camera-ei-video-obj-detection-runner-1 >/dev/null 2>&1 || true

echo
echo "Starting standalone AI containers..."

docker compose -f "${COMPOSE_FILE}" up -d --pull never

echo
docker compose -f "${COMPOSE_FILE}" ps
