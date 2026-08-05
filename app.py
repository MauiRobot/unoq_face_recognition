#!/usr/bin/env python3

import atexit
import re
import threading
import time
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
from gpio_output import GPIOOutput
from recognition import FaceRecognitionEngine


GPIO_TEST_LINE = 41
GPIO_PULSE_SECONDS = 5.0
TRIGGER_COOLDOWN_SECONDS = 15.0

SNAPSHOT_LIMIT_PER_CATEGORY = 100
RECOGNITIONS_DIR = config.PROJECT_ROOT / "recognitions"
ACCEPTED_DIR = RECOGNITIONS_DIR / "accepted"
UNKNOWN_DIR = RECOGNITIONS_DIR / "unknown"

RECOGNITION_INTERVAL_SECONDS = 0.35


app = Flask(__name__)
camera = Camera()
recognition_engine = FaceRecognitionEngine()

event_log: list[dict[str, str]] = []
event_log_lock = threading.Lock()

recognition_lock = threading.Lock()
recognition_stop_event = threading.Event()

trigger_lock = threading.Lock()
last_trigger_time = 0.0

latest_recognition: dict[str, object] = {
    "ready": False,
    "face_detected": False,
    "name": "—",
    "distance": None,
    "threshold": recognition_engine.threshold,
    "result": "MODEL NOT READY",
    "people": 0,
    "updated_at": "",
    "cooldown_remaining": 0.0,
    "gpio_active": False,
}

gpio_output: GPIOOutput | None = None


def add_event(message: str) -> None:
    event = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "message": message,
    }

    with event_log_lock:
        event_log.insert(0, event)
        del event_log[20:]


def get_events() -> list[dict[str, str]]:
    with event_log_lock:
        return [event.copy() for event in event_log]


def set_recognition_status(**updates: object) -> None:
    with recognition_lock:
        latest_recognition.update(updates)
        latest_recognition["updated_at"] = (
            datetime.now().strftime("%H:%M:%S")
        )


def get_cooldown_remaining() -> float:
    with trigger_lock:
        elapsed = time.monotonic() - last_trigger_time

    return max(
        0.0,
        TRIGGER_COOLDOWN_SECONDS - elapsed,
    )


def get_recognition_status() -> dict[str, object]:
    with recognition_lock:
        status = latest_recognition.copy()

    status["cooldown_remaining"] = round(
        get_cooldown_remaining(),
        1,
    )

    status["gpio_active"] = (
        gpio_output.is_active()
        if gpio_output is not None
        else False
    )

    if gpio_output is not None:
        status["leds"] = gpio_output.get_status()

    return status


def load_recognition_model() -> bool:
    try:
        with recognition_lock:
            recognition_engine.load()
            engine_status = recognition_engine.get_status()

            latest_recognition.update(
                {
                    "ready": True,
                    "face_detected": False,
                    "name": "—",
                    "distance": None,
                    "threshold": recognition_engine.threshold,
                    "result": "WAITING",
                    "people": engine_status["people"],
                    "updated_at": datetime.now().strftime("%H:%M:%S"),
                }
            )

        add_event(
            "Recognition model loaded "
            f"({engine_status['people']} people)"
        )

        return True

    except RuntimeError as error:
        set_recognition_status(
            ready=False,
            face_detected=False,
            name="—",
            distance=None,
            result="MODEL NOT READY",
            people=0,
        )

        add_event(f"Recognition unavailable: {error}")
        return False


def retrain_recognition_model() -> int:
    with recognition_lock:
        image_count = recognition_engine.train()
        engine_status = recognition_engine.get_status()

        latest_recognition.update(
            {
                "ready": True,
                "face_detected": False,
                "name": "—",
                "distance": None,
                "threshold": recognition_engine.threshold,
                "result": "WAITING",
                "people": engine_status["people"],
                "updated_at": datetime.now().strftime("%H:%M:%S"),
            }
        )

    return image_count


def ensure_snapshot_directories() -> None:
    ACCEPTED_DIR.mkdir(parents=True, exist_ok=True)
    UNKNOWN_DIR.mkdir(parents=True, exist_ok=True)


def jpeg_files_oldest_first(directory: Path) -> list[Path]:
    return sorted(
        directory.glob("*.jpg"),
        key=lambda path: (
            path.stat().st_mtime,
            path.name,
        ),
    )


def rotate_snapshot_directory(directory: Path) -> None:
    files = jpeg_files_oldest_first(directory)

    while len(files) >= SNAPSHOT_LIMIT_PER_CATEGORY:
        oldest = files.pop(0)

        try:
            oldest.unlink()
            add_event(
                f"Deleted oldest snapshot: {oldest.name}"
            )
        except OSError as error:
            add_event(
                f"Unable to delete old snapshot "
                f"{oldest.name}: {error}"
            )
            break


def safe_filename_component(value: str) -> str:
    cleaned = re.sub(
        r"[^A-Za-z0-9_-]",
        "_",
        value.strip(),
    )

    return cleaned or "UNKNOWN"


def draw_audit_overlay(
    frame,
    name: str,
    distance: float,
    result: str,
):
    output = frame.copy()
    height, width = output.shape[:2]

    panel_height = 112
    panel_top = max(0, height - panel_height)

    dark_panel = output.copy()

    cv2.rectangle(
        dark_panel,
        (0, panel_top),
        (width, height),
        (0, 0, 0),
        thickness=-1,
    )

    cv2.addWeighted(
        dark_panel,
        0.72,
        output,
        0.28,
        0,
        output,
    )

    color = (
        (0, 255, 0)
        if result == "AUTHORIZED"
        else (0, 0, 255)
    )

    cv2.putText(
        output,
        name,
        (16, panel_top + 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        output,
        (
            f"Distance: {distance:.2f}   "
            f"Threshold: "
            f"{recognition_engine.threshold:.2f}   "
            f"{result}"
        ),
        (16, panel_top + 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        output,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        (16, panel_top + 94),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return output


def save_recognition_snapshot(
    name: str,
    distance: float,
    result: str,
) -> Path | None:
    raw_frame = camera.get_latest_raw_frame()

    if raw_frame is None:
        add_event("Snapshot skipped: no camera frame available")
        return None

    if result == "AUTHORIZED":
        directory = ACCEPTED_DIR
        category_name = "accepted"
        identity = safe_filename_component(name)
    else:
        directory = UNKNOWN_DIR
        category_name = "unknown"
        identity = "UNKNOWN"

    directory.mkdir(parents=True, exist_ok=True)
    rotate_snapshot_directory(directory)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination = directory / f"{timestamp}_{identity}.jpg"

    audit_frame = draw_audit_overlay(
        raw_frame,
        name,
        distance,
        result,
    )

    if not cv2.imwrite(str(destination), audit_frame):
        add_event(f"Unable to save {category_name} snapshot")
        return None

    add_event(
        f"Saved {category_name} snapshot: {destination.name}"
    )

    return destination


def trigger_action_if_ready(
    name: str,
    distance: float,
    matched: bool,
) -> None:
    global last_trigger_time

    now = time.monotonic()

    with trigger_lock:
        if (
            now - last_trigger_time
            < TRIGGER_COOLDOWN_SECONDS
        ):
            return

        last_trigger_time = now

    result = "AUTHORIZED" if matched else "UNKNOWN"

    save_recognition_snapshot(
        name=name,
        distance=distance,
        result=result,
    )

    if matched:
        add_event(f"Access accepted for {name}")

        if gpio_output is not None:
            gpio_output.pulse_authorized(
                GPIO_PULSE_SECONDS
            )

    else:
        add_event("Unknown face trigger")

        if gpio_output is not None:
            gpio_output.pulse_unknown(
                GPIO_PULSE_SECONDS
            )


def recognition_worker() -> None:
    while not recognition_stop_event.is_set():
        face_crop = camera.get_latest_face_crop()

        with recognition_lock:
            model_ready = recognition_engine.is_ready()

        if face_crop is None:
            set_recognition_status(
                ready=model_ready,
                face_detected=False,
                name="—",
                distance=None,
                result=(
                    "WAITING"
                    if model_ready
                    else "MODEL NOT READY"
                ),
            )

        elif not model_ready:
            set_recognition_status(
                ready=False,
                face_detected=True,
                name="Unknown",
                distance=None,
                result="MODEL NOT READY",
            )

        else:
            try:
                with recognition_lock:
                    name, distance, matched = (
                        recognition_engine.predict(face_crop)
                    )

                result = "AUTHORIZED" if matched else "UNKNOWN"

                set_recognition_status(
                    ready=True,
                    face_detected=True,
                    name=name,
                    distance=round(distance, 2),
                    threshold=recognition_engine.threshold,
                    result=result,
                )

                trigger_action_if_ready(
                    name=name,
                    distance=distance,
                    matched=matched,
                )

            except Exception as error:
                set_recognition_status(
                    ready=True,
                    face_detected=True,
                    name="—",
                    distance=None,
                    result="ERROR",
                )

                add_event(f"Recognition error: {error}")

        recognition_stop_event.wait(
            RECOGNITION_INTERVAL_SECONDS
        )


def draw_recognition_overlay(frame):
    status = get_recognition_status()

    height, width = frame.shape[:2]
    panel_height = 96
    panel_top = max(0, height - panel_height)

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, panel_top),
        (width, height),
        (0, 0, 0),
        thickness=-1,
    )

    cv2.addWeighted(
        overlay,
        0.68,
        frame,
        0.32,
        0,
        frame,
    )

    result = str(status.get("result", "WAITING"))
    name = str(status.get("name", "—"))
    distance = status.get("distance")
    threshold = float(status.get("threshold", 0.0))
    cooldown = float(status.get("cooldown_remaining", 0.0))

    if result == "AUTHORIZED":
        color = (0, 255, 0)
        title = name
    elif result in ("UNKNOWN", "ERROR"):
        color = (0, 0, 255)
        title = "Unknown"
    elif result == "MODEL NOT READY":
        color = (0, 165, 255)
        title = "Recognition model not ready"
    else:
        color = (0, 215, 255)
        title = "Waiting for face"

    cv2.putText(
        frame,
        title,
        (16, panel_top + 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.76,
        color,
        2,
        cv2.LINE_AA,
    )

    distance_text = (
        "Distance: --"
        if distance is None
        else f"Distance: {float(distance):.2f}"
    )

    cv2.putText(
        frame,
        (
            f"{distance_text}   "
            f"Threshold: {threshold:.2f}   "
            f"Result: {result}"
        ),
        (16, panel_top + 59),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.49,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        (
            f"Trigger cooldown: {cooldown:.1f}s   "
            f"Status LEDs active: "
            f"{'YES' if status['gpio_active'] else 'NO'}"
        ),
        (16, panel_top + 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return frame


def generate_frames():
    while True:
        frame = camera.get_frame()

        if frame is None:
            time.sleep(0.01)
            continue

        frame = draw_recognition_overlay(frame)

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
    name = re.sub(
        r"[^A-Za-z0-9 _-]",
        "",
        name.strip(),
    )
    name = re.sub(r"\s+", " ", name)
    return name[:60].strip()


def person_folder_name(person_name: str) -> str:
    return person_name.replace(" ", "_")


def next_image_path(person_dir: Path) -> Path:
    highest = 0

    for image_path in person_dir.glob("*.jpg"):
        try:
            highest = max(highest, int(image_path.stem))
        except ValueError:
            continue

    return person_dir / f"{highest + 1:04d}.jpg"


def count_person_images(person_dir: Path) -> int:
    return sum(1 for path in person_dir.glob("*.jpg"))


@app.route("/")
def index():
    status = get_recognition_status()

    return render_template(
        "index.html",
        camera_status="Streaming",
        flask_status="Running",
        opencv_status="Loaded",
        preview_available=config.FACE_IMAGE.exists(),
        recognition_ready=status["ready"],
        recognition_threshold=status["threshold"],
        recognition_people=status["people"],
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

    config.CAPTURED_DIR.mkdir(parents=True, exist_ok=True)

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
    data = request.get_json(silent=True) or {}
    person_name = clean_person_name(
        str(data.get("name", ""))
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

    config.FACES_DIR.mkdir(parents=True, exist_ok=True)

    person_dir = (
        config.FACES_DIR
        / person_folder_name(person_name)
    )
    person_dir.mkdir(parents=True, exist_ok=True)

    destination = next_image_path(person_dir)
    face_image = cv2.imread(str(config.FACE_IMAGE))

    if face_image is None:
        return jsonify(
            success=False,
            message="Unable to read the captured face image.",
        ), 500

    if not cv2.imwrite(str(destination), face_image):
        return jsonify(
            success=False,
            message="Unable to save the enrolled image.",
        ), 500

    image_count = count_person_images(person_dir)

    try:
        training_count = retrain_recognition_model()
        model_message = (
            f" Recognition model rebuilt using "
            f"{training_count} image(s)."
        )
    except Exception as error:
        model_message = (
            " Image saved, but model training failed: "
            f"{error}"
        )

    add_event(
        f"Saved {person_name} "
        f"(image {image_count})"
    )

    return jsonify(
        success=True,
        message=(
            f"Saved {person_name} "
            f"as image {image_count}."
            f"{model_message}"
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


@app.route("/recognition_status")
def recognition_status():
    return jsonify(
        success=True,
        **get_recognition_status(),
    )


@app.route("/events")
def events():
    return jsonify(
        success=True,
        events=get_events(),
    )


@atexit.register
def shutdown():
    recognition_stop_event.set()

    if gpio_output is not None:
        gpio_output.close()

    camera.release()


if __name__ == "__main__":
    add_event("Application started")

    ensure_snapshot_directories()

    gpio_output = GPIOOutput(
        gpio_line=GPIO_TEST_LINE,
        led_name="unoq:user-red1",
        event_callback=add_event,
        heartbeat_enabled=True,
        heartbeat_period_seconds=2.0,
        heartbeat_on_seconds=0.15,
    )

    load_recognition_model()

    threading.Thread(
        target=recognition_worker,
        name="recognition-worker",
        daemon=True,
    ).start()

    print("Arduino UNO Q Face Recognition")
    print(f"Camera device: {config.CAMERA_DEVICE}")
    print(
        f"Recognition threshold: "
        f"{recognition_engine.threshold:.2f}"
    )
    print(
        f"Trigger cooldown: "
        f"{TRIGGER_COOLDOWN_SECONDS:.1f} seconds"
    )
    print(
        f"Status LED pulse: "
        f"{GPIO_PULSE_SECONDS:.1f} seconds"
    )
    print(
        "LED mapping: "
        "green=authorized, "
        "red=unknown, "
        "blue=heartbeat"
    )
    print("Open http://192.168.4.124:5000")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True,
    )
