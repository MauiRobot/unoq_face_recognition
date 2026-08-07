#!/usr/bin/env python3

import atexit
import base64
import json
import re
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_file,
)

import config
from gpio_output import GPIOOutput
from recognition import (
    FaceRecognitionEngine,
    LABELS_FILE,
    MODEL_FILE,
)


# ----------------------------------------------------------------------
# Milestone 7.1 settings
# ----------------------------------------------------------------------

GPIO_TEST_LINE = 41
GPIO_PULSE_SECONDS = 5.0
TRIGGER_COOLDOWN_SECONDS = 15.0

SNAPSHOT_LIMIT_PER_CATEGORY = 100
RECOGNITIONS_DIR = config.PROJECT_ROOT / "recognitions"
ACCEPTED_DIR = RECOGNITIONS_DIR / "accepted"
UNKNOWN_DIR = RECOGNITIONS_DIR / "unknown"

AI_FRAME_STALE_SECONDS = 4.0
MJPEG_FRAME_INTERVAL_SECONDS = 0.10

DEFAULT_AI_ENROLLMENT_IMAGES = 5
MIN_AI_ENROLLMENT_CONFIDENCE = 0.80
MIN_AI_FACE_WIDTH = 120
MIN_AI_FACE_HEIGHT = 120
MIN_AI_ENROLLMENT_INTERVAL_SECONDS = 1.5
AI_ENROLLMENT_EDGE_MARGIN = 8


# ----------------------------------------------------------------------
# Flask and shared state
# ----------------------------------------------------------------------

app = Flask(__name__)
recognition_engine = FaceRecognitionEngine()

event_log: list[dict[str, str]] = []
event_log_lock = threading.Lock()

recognition_lock = threading.RLock()
trigger_lock = threading.Lock()
frame_lock = threading.Lock()
enrollment_lock = threading.Lock()

last_trigger_time = 0.0

latest_raw_frame: np.ndarray | None = None
latest_face_crop: np.ndarray | None = None
latest_detector_box: tuple[int, int, int, int] | None = None
latest_detector_confidence: float | None = None
latest_frame_received_at = 0.0
latest_frame_sequence = 0

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
    "detector": "Arduino AI",
    "detector_confidence": None,
}

ai_enrollment: dict[str, object] = {
    "active": False,
    "person_name": "",
    "folder_name": "",
    "target_images": DEFAULT_AI_ENROLLMENT_IMAGES,
    "saved_images": 0,
    "last_saved_at": 0.0,
    "last_message": "AI enrollment is idle.",
    "started_at": "",
    "completed_at": "",
}

gpio_output: GPIOOutput | None = None


# ----------------------------------------------------------------------
# Event log
# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------
# Recognition model
# ----------------------------------------------------------------------

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

    with frame_lock:
        age = (
            time.monotonic() - latest_frame_received_at
            if latest_frame_received_at > 0
            else None
        )

    status["ai_stream_connected"] = (
        age is not None
        and age <= AI_FRAME_STALE_SECONDS
    )
    status["ai_frame_age"] = (
        None if age is None else round(age, 1)
    )

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


def reset_recognition_model() -> None:
    global recognition_engine

    with recognition_lock:
        recognition_engine = FaceRecognitionEngine()

        latest_recognition.update(
            {
                "ready": False,
                "face_detected": False,
                "name": "—",
                "distance": None,
                "threshold": recognition_engine.threshold,
                "result": "MODEL NOT READY",
                "people": 0,
                "updated_at": datetime.now().strftime("%H:%M:%S"),
            }
        )

    for model_path in (MODEL_FILE, LABELS_FILE):
        try:
            model_path.unlink(missing_ok=True)
        except OSError as error:
            add_event(
                f"Unable to delete model file "
                f"{model_path.name}: {error}"
            )


# ----------------------------------------------------------------------
# People and enrollment helpers
# ----------------------------------------------------------------------

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


def person_display_name(folder_name: str) -> str:
    return folder_name.replace("_", " ")


def count_person_images(person_dir: Path) -> int:
    return sum(1 for path in person_dir.glob("*.jpg"))


def next_image_path(person_dir: Path) -> Path:
    highest = 0

    for image_path in person_dir.glob("*.jpg"):
        try:
            highest = max(
                highest,
                int(image_path.stem),
            )
        except ValueError:
            continue

    return person_dir / f"{highest + 1:04d}.jpg"


def list_enrolled_people() -> list[dict[str, object]]:
    if not config.FACES_DIR.exists():
        return []

    people: list[dict[str, object]] = []

    for person_dir in sorted(
        (
            path
            for path in config.FACES_DIR.iterdir()
            if path.is_dir()
        ),
        key=lambda path: path.name.lower(),
    ):
        image_count = count_person_images(person_dir)

        if image_count <= 0:
            continue

        people.append(
            {
                "name": person_display_name(person_dir.name),
                "folder": person_dir.name,
                "image_count": image_count,
            }
        )

    return people


def resolve_person_directory(person_name: str) -> Path | None:
    cleaned = clean_person_name(person_name)

    if not cleaned:
        return None

    candidate = (
        config.FACES_DIR
        / person_folder_name(cleaned)
    )

    try:
        faces_root = config.FACES_DIR.resolve()
        resolved = candidate.resolve()
    except OSError:
        return None

    if resolved.parent != faces_root:
        return None

    return resolved


# ----------------------------------------------------------------------
# AI frame helpers
# ----------------------------------------------------------------------

def decode_jpeg_base64(encoded: str) -> np.ndarray | None:
    try:
        jpeg_bytes = base64.b64decode(
            encoded,
            validate=True,
        )
    except (ValueError, TypeError):
        return None

    image_array = np.frombuffer(
        jpeg_bytes,
        dtype=np.uint8,
    )

    return cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR,
    )


def clamp_box(
    bounding_box,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    if (
        not isinstance(bounding_box, (list, tuple))
        or len(bounding_box) != 4
    ):
        return None

    try:
        x1, y1, x2, y2 = (
            int(round(float(value)))
            for value in bounding_box
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


def update_latest_ai_frame(
    frame: np.ndarray,
    face_crop: np.ndarray,
    box: tuple[int, int, int, int],
    confidence: float,
) -> None:
    global latest_raw_frame
    global latest_face_crop
    global latest_detector_box
    global latest_detector_confidence
    global latest_frame_received_at
    global latest_frame_sequence

    with frame_lock:
        latest_raw_frame = frame.copy()
        latest_face_crop = face_crop.copy()
        latest_detector_box = box
        latest_detector_confidence = confidence
        latest_frame_received_at = time.monotonic()
        latest_frame_sequence += 1


def get_latest_frames() -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    tuple[int, int, int, int] | None,
    float | None,
]:
    with frame_lock:
        raw = (
            None
            if latest_raw_frame is None
            else latest_raw_frame.copy()
        )
        face = (
            None
            if latest_face_crop is None
            else latest_face_crop.copy()
        )
        box = latest_detector_box
        confidence = latest_detector_confidence

    return raw, face, box, confidence


# ----------------------------------------------------------------------
# Snapshot rotation and access action
# ----------------------------------------------------------------------

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
    frame: np.ndarray,
    name: str,
    distance: float,
    result: str,
) -> np.ndarray:
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
    frame: np.ndarray,
    name: str,
    distance: float,
    result: str,
) -> Path | None:
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

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    destination = (
        directory
        / f"{timestamp}_{identity}.jpg"
    )

    audit_frame = draw_audit_overlay(
        frame,
        name,
        distance,
        result,
    )

    if not cv2.imwrite(
        str(destination),
        audit_frame,
    ):
        add_event(
            f"Unable to save {category_name} snapshot"
        )
        return None

    add_event(
        f"Saved {category_name} snapshot: "
        f"{destination.name}"
    )
    return destination


def trigger_action_if_ready(
    frame: np.ndarray,
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

    result = (
        "AUTHORIZED"
        if matched
        else "UNKNOWN"
    )

    save_recognition_snapshot(
        frame=frame,
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


# ----------------------------------------------------------------------
# AI multi-image enrollment API
# ----------------------------------------------------------------------

def get_ai_enrollment_status() -> dict[str, object]:
    with enrollment_lock:
        status = ai_enrollment.copy()

    status["remaining_images"] = max(
        0,
        int(status["target_images"])
        - int(status["saved_images"]),
    )
    return status


def update_ai_enrollment(**updates: object) -> None:
    with enrollment_lock:
        ai_enrollment.update(updates)


def ai_box_touches_edge(
    box: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
) -> bool:
    x1, y1, x2, y2 = box

    return (
        x1 <= AI_ENROLLMENT_EDGE_MARGIN
        or y1 <= AI_ENROLLMENT_EDGE_MARGIN
        or x2 >= frame_width - AI_ENROLLMENT_EDGE_MARGIN
        or y2 >= frame_height - AI_ENROLLMENT_EDGE_MARGIN
    )


def process_ai_enrollment(
    frame: np.ndarray,
    face_crop: np.ndarray,
    box: tuple[int, int, int, int],
    detector_confidence: float,
) -> dict[str, object] | None:
    status = get_ai_enrollment_status()

    if not bool(status["active"]):
        return None

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    face_width = x2 - x1
    face_height = y2 - y1

    if detector_confidence < MIN_AI_ENROLLMENT_CONFIDENCE:
        update_ai_enrollment(
            last_message=(
                "Hold still: detector confidence "
                f"{detector_confidence:.2f} is below "
                f"{MIN_AI_ENROLLMENT_CONFIDENCE:.2f}."
            )
        )
        return {
            "saved": False,
            "status": get_ai_enrollment_status(),
        }

    if (
        face_width < MIN_AI_FACE_WIDTH
        or face_height < MIN_AI_FACE_HEIGHT
    ):
        update_ai_enrollment(
            last_message=(
                "Move closer: face crop is "
                f"{face_width}x{face_height}."
            )
        )
        return {
            "saved": False,
            "status": get_ai_enrollment_status(),
        }

    if ai_box_touches_edge(
        box,
        width,
        height,
    ):
        update_ai_enrollment(
            last_message=(
                "Center your full face inside the frame."
            )
        )
        return {
            "saved": False,
            "status": get_ai_enrollment_status(),
        }

    elapsed = (
        time.monotonic()
        - float(status["last_saved_at"])
    )

    if elapsed < MIN_AI_ENROLLMENT_INTERVAL_SECONDS:
        return {
            "saved": False,
            "status": status,
        }

    person_name = str(status["person_name"])
    person_dir = (
        config.FACES_DIR
        / str(status["folder_name"])
    )
    person_dir.mkdir(parents=True, exist_ok=True)

    destination = next_image_path(person_dir)

    if not cv2.imwrite(
        str(destination),
        face_crop,
    ):
        update_ai_enrollment(
            active=False,
            last_message=(
                "Unable to save AI enrollment image."
            ),
            completed_at=datetime.now().isoformat(
                timespec="seconds"
            ),
        )
        return {
            "saved": False,
            "error": "Unable to save enrollment image.",
            "status": get_ai_enrollment_status(),
        }

    saved_images = int(status["saved_images"]) + 1
    target_images = int(status["target_images"])

    update_ai_enrollment(
        saved_images=saved_images,
        last_saved_at=time.monotonic(),
        last_message=(
            f"Saved image {saved_images} "
            f"of {target_images} for {person_name}."
        ),
    )

    add_event(
        f"AI enrollment saved {person_name} "
        f"image {saved_images}/{target_images}"
    )

    completed = saved_images >= target_images
    training_images = None

    if completed:
        try:
            training_images = (
                retrain_recognition_model()
            )

            update_ai_enrollment(
                active=False,
                completed_at=datetime.now().isoformat(
                    timespec="seconds"
                ),
                last_message=(
                    f"Enrollment completed for "
                    f"{person_name}. Model rebuilt "
                    f"using {training_images} image(s)."
                ),
            )

            add_event(
                f"AI enrollment completed for "
                f"{person_name}"
            )

        except Exception as error:
            update_ai_enrollment(
                active=False,
                completed_at=datetime.now().isoformat(
                    timespec="seconds"
                ),
                last_message=(
                    "Images saved, but model training "
                    f"failed: {error}"
                ),
            )
            add_event(
                f"AI enrollment training failed: {error}"
            )

    return {
        "saved": True,
        "completed": completed,
        "training_images": training_images,
        "status": get_ai_enrollment_status(),
    }


# ----------------------------------------------------------------------
# Recognition and live overlay
# ----------------------------------------------------------------------

def recognize_ai_crop(
    frame: np.ndarray,
    face_crop: np.ndarray,
    detector_confidence: float,
) -> dict[str, object]:
    with recognition_lock:
        model_ready = recognition_engine.is_ready()

    if not model_ready:
        set_recognition_status(
            ready=False,
            face_detected=True,
            name="Unknown",
            distance=None,
            result="MODEL NOT READY",
            detector_confidence=round(
                detector_confidence,
                3,
            ),
        )

        return {
            "name": "Unknown",
            "distance": None,
            "matched": False,
            "result": "MODEL NOT READY",
        }

    try:
        with recognition_lock:
            name, distance, matched = (
                recognition_engine.predict(
                    face_crop
                )
            )

        result = (
            "AUTHORIZED"
            if matched
            else "UNKNOWN"
        )

        set_recognition_status(
            ready=True,
            face_detected=True,
            name=name,
            distance=round(distance, 2),
            threshold=recognition_engine.threshold,
            result=result,
            detector_confidence=round(
                detector_confidence,
                3,
            ),
        )

        trigger_action_if_ready(
            frame=frame,
            name=name,
            distance=distance,
            matched=matched,
        )

        return {
            "name": name,
            "distance": round(distance, 2),
            "matched": matched,
            "result": result,
        }

    except Exception as error:
        set_recognition_status(
            ready=True,
            face_detected=True,
            name="—",
            distance=None,
            result="ERROR",
            detector_confidence=round(
                detector_confidence,
                3,
            ),
        )
        add_event(f"Recognition error: {error}")

        return {
            "name": "Unknown",
            "distance": None,
            "matched": False,
            "result": "ERROR",
            "error": str(error),
        }


def draw_recognition_overlay(
    frame: np.ndarray,
) -> np.ndarray:
    status = get_recognition_status()
    output = frame.copy()

    height, width = output.shape[:2]

    with frame_lock:
        box = latest_detector_box
        detector_confidence = (
            latest_detector_confidence
        )

    if box is not None:
        x1, y1, x2, y2 = box

        result = str(
            status.get("result", "WAITING")
        )

        if result == "AUTHORIZED":
            box_color = (0, 255, 0)
        elif result in ("UNKNOWN", "ERROR"):
            box_color = (0, 0, 255)
        else:
            box_color = (0, 215, 255)

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            box_color,
            2,
        )

        if detector_confidence is not None:
            cv2.putText(
                output,
                (
                    "AI face "
                    f"{detector_confidence:.2f}"
                ),
                (
                    x1,
                    max(22, y1 - 8),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                box_color,
                2,
                cv2.LINE_AA,
            )

    panel_height = 96
    panel_top = max(0, height - panel_height)
    overlay = output.copy()

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
        output,
        0.32,
        0,
        output,
    )

    result = str(
        status.get("result", "WAITING")
    )
    name = str(status.get("name", "—"))
    distance = status.get("distance")
    threshold = float(
        status.get("threshold", 0.0)
    )
    cooldown = float(
        status.get("cooldown_remaining", 0.0)
    )

    if result == "AUTHORIZED":
        color = (0, 255, 0)
        title = name
    elif result in ("UNKNOWN", "ERROR"):
        color = (0, 0, 255)
        title = "Unknown"
    elif result == "MODEL NOT READY":
        color = (0, 165, 255)
        title = "Recognition model not ready"
    elif result == "ENROLLING":
        color = (255, 180, 0)
        title = "AI enrollment active"
    else:
        color = (0, 215, 255)
        title = "Waiting for AI face"

    cv2.putText(
        output,
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
        output,
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
        output,
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

    return output


def waiting_frame() -> np.ndarray:
    frame = np.zeros(
        (480, 640, 3),
        dtype=np.uint8,
    )

    cv2.putText(
        frame,
        "Waiting for Arduino AI detector...",
        (72, 220),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "Start the App Lab face detector.",
        (120, 260),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    return frame


def generate_frames():
    while True:
        with frame_lock:
            frame = (
                None
                if latest_raw_frame is None
                else latest_raw_frame.copy()
            )

        if frame is None:
            frame = waiting_frame()
        else:
            frame = draw_recognition_overlay(
                frame
            )

        encoded, jpeg = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 80],
        )

        if encoded:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpeg.tobytes()
                + b"\r\n"
            )

        time.sleep(
            MJPEG_FRAME_INTERVAL_SECONDS
        )


# ----------------------------------------------------------------------
# Flask routes used by App Lab AI detector
# ----------------------------------------------------------------------

@app.post("/ai_detection")
def ai_detection():
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify(
            success=False,
            message="JSON body is required.",
        ), 400

    if payload.get("label") != "face":
        return jsonify(
            success=False,
            message="Only face detections are accepted.",
        ), 400

    encoded_frame = payload.get(
        "frame_jpeg_base64"
    )

    if not isinstance(encoded_frame, str):
        return jsonify(
            success=False,
            message="frame_jpeg_base64 is required.",
        ), 400

    frame = decode_jpeg_base64(
        encoded_frame
    )

    if frame is None:
        return jsonify(
            success=False,
            message="Unable to decode JPEG frame.",
        ), 400

    height, width = frame.shape[:2]

    box = clamp_box(
        payload.get("bounding_box_xyxy"),
        width,
        height,
    )

    if box is None:
        return jsonify(
            success=False,
            message="Invalid face bounding box.",
        ), 400

    x1, y1, x2, y2 = box
    face_crop = frame[
        y1:y2,
        x1:x2,
    ].copy()

    if face_crop.size == 0:
        return jsonify(
            success=False,
            message="Face crop is empty.",
        ), 400

    try:
        detector_confidence = float(
            payload.get("confidence", 0.0)
        )
    except (TypeError, ValueError):
        detector_confidence = 0.0

    update_latest_ai_frame(
        frame=frame,
        face_crop=face_crop,
        box=box,
        confidence=detector_confidence,
    )

    enrollment_result = process_ai_enrollment(
        frame=frame,
        face_crop=face_crop,
        box=box,
        detector_confidence=detector_confidence,
    )

    if enrollment_result is not None:
        set_recognition_status(
            ready=recognition_engine.is_ready(),
            face_detected=True,
            name="Enrollment active",
            distance=None,
            result="ENROLLING",
            detector_confidence=round(
                detector_confidence,
                3,
            ),
        )

        recognition_result = {
            "name": "Enrollment active",
            "distance": None,
            "matched": False,
            "result": "ENROLLING",
        }
    else:
        recognition_result = recognize_ai_crop(
            frame=frame,
            face_crop=face_crop,
            detector_confidence=detector_confidence,
        )

    return jsonify(
        success=True,
        detection={
            "confidence": detector_confidence,
            "bounding_box_xyxy": list(box),
            "frame_width": width,
            "frame_height": height,
            "face_width": x2 - x1,
            "face_height": y2 - y1,
        },
        recognition=recognition_result,
        enrollment=enrollment_result,
    )


@app.get("/ai_health")
def ai_health():
    return jsonify(
        success=True,
        service="UNO Q integrated AI face recognition",
        recognition=get_recognition_status(),
        enrollment=get_ai_enrollment_status(),
    )


@app.post("/enrollment/start")
def ai_enrollment_start():
    data = request.get_json(silent=True) or {}
    person_name = clean_person_name(
        str(data.get("name", ""))
    )

    if not person_name:
        return jsonify(
            success=False,
            message="Enter a valid person name.",
        ), 400

    try:
        target_images = int(
            data.get(
                "images",
                DEFAULT_AI_ENROLLMENT_IMAGES,
            )
        )
    except (TypeError, ValueError):
        return jsonify(
            success=False,
            message="Image count must be a number.",
        ), 400

    target_images = max(
        1,
        min(target_images, 25),
    )

    update_ai_enrollment(
        active=True,
        person_name=person_name,
        folder_name=person_folder_name(
            person_name
        ),
        target_images=target_images,
        saved_images=0,
        last_saved_at=0.0,
        last_message=(
            "Enrollment started. "
            "Look straight at the camera."
        ),
        started_at=datetime.now().isoformat(
            timespec="seconds"
        ),
        completed_at="",
    )

    add_event(
        f"AI enrollment started for "
        f"{person_name}"
    )

    return jsonify(
        success=True,
        enrollment=get_ai_enrollment_status(),
    )


@app.post("/enrollment/cancel")
def ai_enrollment_cancel():
    update_ai_enrollment(
        active=False,
        last_message="Enrollment cancelled.",
        completed_at=datetime.now().isoformat(
            timespec="seconds"
        ),
    )

    add_event("AI enrollment cancelled")

    return jsonify(
        success=True,
        enrollment=get_ai_enrollment_status(),
    )


@app.get("/enrollment/status")
def ai_enrollment_status():
    return jsonify(
        success=True,
        enrollment=get_ai_enrollment_status(),
    )


# ----------------------------------------------------------------------
# Existing dashboard routes
# ----------------------------------------------------------------------

@app.route("/")
def index():
    return render_template(
        "index.html",
        preview_available=(
            config.FACE_IMAGE.exists()
        ),
    )


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        ),
    )


@app.route("/capture", methods=["POST"])
def capture():
    raw_frame, face_crop, _, _ = (
        get_latest_frames()
    )

    if raw_frame is None:
        return jsonify(
            success=False,
            message=(
                "No AI camera frame is available yet. "
                "Start the App Lab detector."
            ),
        ), 503

    if face_crop is None:
        return jsonify(
            success=False,
            message=(
                "No AI face is currently detected."
            ),
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
            message=(
                "Unable to save the captured images."
            ),
        ), 500

    add_event("AI face captured")

    return jsonify(
        success=True,
        message="AI face captured successfully.",
        preview_url="/captured_face",
    )


@app.route("/captured_face")
def captured_face():
    face_file = Path(config.FACE_IMAGE)

    if not face_file.exists():
        return jsonify(
            success=False,
            message=(
                "No captured face is available."
            ),
        ), 404

    response = send_file(
        face_file,
        mimetype="image/jpeg",
        conditional=False,
    )

    response.headers["Cache-Control"] = (
        "no-store, no-cache, "
        "must-revalidate, max-age=0"
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
            message=(
                "Capture an AI face before "
                "saving the person."
            ),
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
            message=(
                "Unable to read the captured face."
            ),
        ), 500

    if not cv2.imwrite(
        str(destination),
        face_image,
    ):
        return jsonify(
            success=False,
            message=(
                "Unable to save the enrolled image."
            ),
        ), 500

    image_count = count_person_images(
        person_dir
    )

    try:
        training_count = (
            retrain_recognition_model()
        )
        model_message = (
            " Recognition model rebuilt using "
            f"{training_count} image(s)."
        )
    except Exception as error:
        model_message = (
            " Image saved, but model training "
            f"failed: {error}"
        )

    add_event(
        f"Saved {person_name} "
        f"(AI image {image_count})"
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


@app.route("/people")
def people():
    return jsonify(
        success=True,
        people=list_enrolled_people(),
    )


@app.route("/remove_person", methods=["POST"])
def remove_person():
    data = request.get_json(silent=True) or {}

    person_name = clean_person_name(
        str(data.get("name", ""))
    )

    person_dir = resolve_person_directory(
        person_name
    )

    if (
        person_dir is None
        or not person_dir.exists()
        or not person_dir.is_dir()
    ):
        return jsonify(
            success=False,
            message=(
                "Enrolled person was not found."
            ),
        ), 404

    image_count = count_person_images(
        person_dir
    )

    try:
        shutil.rmtree(person_dir)
    except OSError as error:
        return jsonify(
            success=False,
            message=(
                f"Unable to remove "
                f"{person_name}: {error}"
            ),
        ), 500

    remaining_people = list_enrolled_people()

    if remaining_people:
        try:
            training_count = (
                retrain_recognition_model()
            )
            model_message = (
                " Recognition model rebuilt using "
                f"{training_count} image(s)."
            )
        except Exception as error:
            return jsonify(
                success=False,
                message=(
                    f"{person_name} was removed, "
                    "but model retraining failed: "
                    f"{error}"
                ),
            ), 500
    else:
        reset_recognition_model()
        model_message = (
            " No enrolled people remain; "
            "recognition is now disabled."
        )

    add_event(
        f"Removed {person_name} "
        f"({image_count} image(s))"
    )

    return jsonify(
        success=True,
        message=(
            f"Removed {person_name} "
            f"and {image_count} image(s)."
            f"{model_message}"
        ),
        people=remaining_people,
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


# ----------------------------------------------------------------------
# Startup and shutdown
# ----------------------------------------------------------------------

@atexit.register
def shutdown():
    if gpio_output is not None:
        gpio_output.close()


if __name__ == "__main__":
    add_event(
        "Application started with Arduino AI detector"
    )

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

    print(
        "Arduino UNO Q AI Face Recognition"
    )
    print(
        "Camera owner: Arduino App Lab "
        "VideoObjectDetection"
    )
    print(
        "Detector input endpoint: "
        "http://192.168.4.124:5000/ai_detection"
    )
    print(
        f"Recognition threshold: "
        f"{recognition_engine.threshold:.2f}"
    )
    print(
        f"Trigger cooldown: "
        f"{TRIGGER_COOLDOWN_SECONDS:.1f} seconds"
    )
    print(
        "LED mapping: "
        "green=authorized, "
        "red=unknown, "
        "blue=heartbeat"
    )
    print(
        "Dashboard: "
        "http://192.168.4.124:5000"
    )

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True,
    )
