#!/usr/bin/env python3
import base64
import json
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request

PROJECT_ROOT = Path.home() / "unoq_face_recognition"
OUTPUT_DIR = PROJECT_ROOT / "ai_bridge"
LATEST_FRAME = OUTPUT_DIR / "latest_frame.jpg"
LATEST_FACE = OUTPUT_DIR / "latest_face.jpg"
LATEST_METADATA = OUTPUT_DIR / "latest_detection.json"

app = Flask(__name__)
write_lock = threading.Lock()


def decode_jpeg_base64(encoded: str):
    try:
        jpeg_bytes = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError):
        return None

    array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def clamp_box(box, width: int, height: int):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None

    try:
        x1, y1, x2, y2 = (
            int(round(float(value))) for value in box
        )
    except (TypeError, ValueError):
        return None

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


@app.get("/health")
def health():
    return jsonify(success=True, service="UNO Q AI detector bridge")


@app.post("/ai_detection")
def ai_detection():
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify(success=False, message="JSON body required."), 400

    if payload.get("label") != "face":
        return jsonify(success=False, message="Only face accepted."), 400

    frame = decode_jpeg_base64(payload.get("frame_jpeg_base64"))

    if frame is None:
        return jsonify(success=False, message="Invalid JPEG frame."), 400

    height, width = frame.shape[:2]
    box = clamp_box(payload.get("bounding_box_xyxy"), width, height)

    if box is None:
        return jsonify(success=False, message="Invalid bounding box."), 400

    x1, y1, x2, y2 = box
    face = frame[y1:y2, x1:x2].copy()

    if face.size == 0:
        return jsonify(success=False, message="Empty face crop."), 400

    metadata = {
        "received_at": datetime.now().isoformat(timespec="milliseconds"),
        "label": "face",
        "confidence": payload.get("confidence"),
        "bounding_box_xyxy": list(box),
        "frame_width": width,
        "frame_height": height,
        "face_width": x2 - x1,
        "face_height": y2 - y1,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with write_lock:
        frame_ok = cv2.imwrite(str(LATEST_FRAME), frame)
        face_ok = cv2.imwrite(str(LATEST_FACE), face)
        LATEST_METADATA.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

    if not frame_ok or not face_ok:
        return jsonify(success=False, message="Unable to save images."), 500

    print("AI BRIDGE RECEIVED:", metadata, flush=True)

    return jsonify(success=True, metadata=metadata)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("UNO Q AI Detector Bridge")
    print("Listening on http://0.0.0.0:5055")
    print(f"Output directory: {OUTPUT_DIR}")
    app.run(host="0.0.0.0", port=5055, debug=False, threaded=True)
