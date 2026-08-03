#!/usr/bin/env python3

import atexit
from pathlib import Path

import cv2
from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    send_file,
)

import config
from camera import Camera


app = Flask(__name__)
camera = Camera()


def generate_frames():
    while True:
        frame = camera.get_frame()

        if frame is None:
            continue

        encoded, jpeg = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 80],
        )

        if not encoded:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpeg.tobytes()
            + b"\r\n"
        )


@app.route("/")
def index():
    return render_template(
        "index.html",
        camera_status="Streaming",
        flask_status="Running",
        opencv_status="Loaded",
        preview_available=config.FACE_IMAGE.exists(),
    )


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/capture", methods=["POST"])
def capture():
    raw_frame = camera.get_latest_raw_frame()
    face_crop = camera.get_latest_face_crop()

    if raw_frame is None:
        return jsonify(
            success=False,
            message="No camera frame is available yet.",
        ), 503

    if face_crop is None:
        return jsonify(
            success=False,
            message="No face is currently detected.",
        ), 400

    config.CAPTURED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame_saved = cv2.imwrite(
        str(config.FRAME_IMAGE),
        raw_frame,
    )

    face_saved = cv2.imwrite(
        str(config.FACE_IMAGE),
        face_crop,
    )

    if not frame_saved or not face_saved:
        return jsonify(
            success=False,
            message="Unable to save the captured images.",
        ), 500

    return jsonify(
        success=True,
        message="Face captured successfully.",
        preview_url="/captured_face",
    )


@app.route("/captured_face")
def captured_face():
    face_file = Path(config.FACE_IMAGE)

    if not face_file.exists():
        return jsonify(
            success=False,
            message="No captured face is available.",
        ), 404

    response = send_file(
        face_file,
        mimetype="image/jpeg",
        conditional=False,
    )

    response.headers["Cache-Control"] = (
        "no-store, no-cache, must-revalidate, max-age=0"
    )
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"

    return response


@atexit.register
def shutdown():
    camera.release()


if __name__ == "__main__":
    print("Arduino UNO Q Face Recognition")
    print(f"Camera device: {config.CAMERA_DEVICE}")
    print("Open http://192.168.4.124:5000")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True,
    )