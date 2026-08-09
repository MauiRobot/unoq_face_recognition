#!/usr/bin/env bash
set -euo pipefail

FLASK_SERVICE="unoq-face-recognition.service"
AI_SERVICE="unoq-face-ai.service"

sudo systemctl disable --now "${AI_SERVICE}" 2>/dev/null || true
sudo systemctl disable --now "${FLASK_SERVICE}" 2>/dev/null || true
sudo rm -f "/etc/systemd/system/${AI_SERVICE}"
sudo rm -f "/etc/systemd/system/${FLASK_SERVICE}"
sudo systemctl daemon-reload
sudo systemctl reset-failed

echo "Automatic startup removed."
