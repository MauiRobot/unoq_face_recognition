#!/usr/bin/env python3

import atexit

import cv2
from flask import Flask, Response, jsonify, render_template

import config
from camera import Camera


app = Flask(__name__)
camera = Camera()


def generate_frames():
    while True:
        frame = camera.get_frame()

        if frame is None:
            continue

        ok, jpeg = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 80],
        )

        if not ok:
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
        frame_file=str(config.FRAME_IMAGE),
        face_file=str(config.FACE_IMAGE),
    )


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