#!/usr/bin/env python3

import atexit

import cv2
from flask import Flask, Response, render_template

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


@atexit.register
def shutdown():
    camera.release()


if __name__ == "__main__":
    print("Arduino UNO Q Face Recognition")
    print("Camera device: /dev/video2")
    print("Open http://192.168.4.124:5000")
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True,
    )