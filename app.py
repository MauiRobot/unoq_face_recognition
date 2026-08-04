#!/usr/bin/env python3

import atexit
import re
from datetime import datetime
from pathlib import Path

import cv2
from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_file,
)

import config
from camera import Camera


app = Flask(__name__)
camera = Camera()

event_log: list[dict[str, str]] = []


def add_event(message: str) -> None:
    event_log.insert(
        0,
        {
            "time": datetime.now().strftime("%H:%M:%S"),
            "message": message,
        },
    )

    del event_log[20:]


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


def clean_person_name(name: str) -> str:
    name = name.strip()

    if not name:
        return ""

    # Allow letters, numbers, spaces, hyphens, and underscores.
    name = re.sub(r"[^A-Za-z0-9 _-]", "", name)
    name = re.sub(r"\s+", " ", name)

    return name[:60].strip()


def person_folder_name(person_name: str) -> str:
    folder_name = person_name.replace(" ", "_")
    return folder_name


def next_image_path(person_dir: Path) -> Path:
    highest_number = 0

    for image_path in person_dir.glob("*.jpg"):
        try:
            number = int(image_path.stem)
        except ValueError:
            continue

        highest_number = max(highest_number, number)

    return person_dir / f"{highest_number + 1:04d}.jpg"


def count_person_images(person_dir: Path) -> int:
    return sum(1 for path in person_dir.glob("*.jpg"))


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

    add_event("Face captured")

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


@app.route("/enroll", methods=["POST"])
def enroll():
    request_data = request.get_json(silent=True) or {}
    person_name = clean_person_name(
        str(request_data.get("name", ""))
    )

    if not person_name:
        return jsonify(
            success=False,
            message="Enter a valid person name.",
        ), 400

    if not config.FACE_IMAGE.exists():
        return jsonify(
            success=False,
            message="Capture a face before saving the person.",
        ), 400

    config.FACES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    person_dir = (
        config.FACES_DIR
        / person_folder_name(person_name)
    )

    person_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination = next_image_path(person_dir)

    face_image = cv2.imread(
        str(config.FACE_IMAGE)
    )

    if face_image is None:
        return jsonify(
            success=False,
            message="Unable to read the captured face image.",
        ), 500

    if not cv2.imwrite(
        str(destination),
        face_image,
    ):
        return jsonify(
            success=False,
            message="Unable to save the enrolled image.",
        ), 500

    image_count = count_person_images(person_dir)

    add_event(
        f"Saved {person_name} "
        f"(image {image_count})"
    )

    return jsonify(
        success=True,
        message=(
            f"Saved {person_name} "
            f"as image {image_count}."
        ),
        person_name=person_name,
        image_count=image_count,
        image_file=str(destination),
    )


@app.route("/person_status")
def person_status():
    person_name = clean_person_name(
        request.args.get("name", "")
    )

    if not person_name:
        return jsonify(
            success=True,
            person_name="",
            image_count=0,
        )

    person_dir = (
        config.FACES_DIR
        / person_folder_name(person_name)
    )

    image_count = (
        count_person_images(person_dir)
        if person_dir.exists()
        else 0
    )

    return jsonify(
        success=True,
        person_name=person_name,
        image_count=image_count,
    )


@app.route("/events")
def events():
    return jsonify(
        success=True,
        events=event_log,
    )


@atexit.register
def shutdown():
    camera.release()


if __name__ == "__main__":
    add_event("Application started")

    print("Arduino UNO Q Face Recognition")
    print(f"Camera device: {config.CAMERA_DEVICE}")
    print("Open http://192.168.4.124:5000")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True,
    )