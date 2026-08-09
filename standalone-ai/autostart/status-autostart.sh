#!/usr/bin/env bash
set -euo pipefail

echo "=== Flask recognition service ==="
systemctl --no-pager --full status unoq-face-recognition.service || true

echo
echo "=== Standalone AI service ==="
systemctl --no-pager --full status unoq-face-ai.service || true

echo
echo "=== AI containers ==="
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" --filter "name=unoq-face-ai"

echo
echo "=== Port 5000 ==="
ss -ltnp 2>/dev/null | grep ':5000' || true

echo
echo "=== Recent Flask log ==="
journalctl -u unoq-face-recognition.service -n 25 --no-pager || true

echo
echo "=== Recent AI log ==="
journalctl -u unoq-face-ai.service -n 25 --no-pager || true
