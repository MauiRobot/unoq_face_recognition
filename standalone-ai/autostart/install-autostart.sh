#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/arduino/unoq_face_recognition"
SOURCE_DIR="${PROJECT_DIR}/standalone-ai/autostart"
SYSTEMD_DIR="/etc/systemd/system"
FLASK_SERVICE="unoq-face-recognition.service"
AI_SERVICE="unoq-face-ai.service"

echo "UNO Q Face Recognition - Milestone 7.3.2"
echo "Installing automatic startup services..."
echo

test -x "${PROJECT_DIR}/.venv/bin/python" || { echo "ERROR: .venv Python missing."; exit 1; }
test -f "${PROJECT_DIR}/app.py" || { echo "ERROR: app.py missing."; exit 1; }
test -x "${PROJECT_DIR}/standalone-ai/start-ai.sh" || { echo "ERROR: start-ai.sh missing/not executable."; exit 1; }
test -x "${PROJECT_DIR}/standalone-ai/stop-ai.sh" || { echo "ERROR: stop-ai.sh missing/not executable."; exit 1; }

docker image inspect ghcr.io/arduino/app-bricks/ei-models-runner:0.11.2 >/dev/null
docker image inspect ghcr.io/arduino/app-bricks/python-apps-base:0.11.0 >/dev/null

echo "Required files and cached Docker images found."

sudo install -m 0644 "${SOURCE_DIR}/${FLASK_SERVICE}" "${SYSTEMD_DIR}/${FLASK_SERVICE}"
sudo install -m 0644 "${SOURCE_DIR}/${AI_SERVICE}" "${SYSTEMD_DIR}/${AI_SERVICE}"

sudo systemctl daemon-reload
sudo systemctl enable "${FLASK_SERVICE}"
sudo systemctl enable "${AI_SERVICE}"

echo
echo "Starting Flask service..."
sudo systemctl restart "${FLASK_SERVICE}"
sleep 5

echo "Starting standalone AI service..."
sudo systemctl restart "${AI_SERVICE}"

echo
echo "Installed and started."
echo
sudo systemctl --no-pager --full status "${FLASK_SERVICE}" || true
echo
sudo systemctl --no-pager --full status "${AI_SERVICE}" || true
echo
echo "Dashboard: http://192.168.4.124:5000"
echo "After verification, reboot with: sudo reboot"
