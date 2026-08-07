#!/usr/bin/env python3

import base64
import json
import re
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request

import config
from recognition import FaceRecognitionEngine


# ----------------------------------------------------------------------
# Paths and settings
# ----------------------------------------------------------------------

PROJECT_ROOT = Path.home() / "unoq_face_recognition"

OUTPUT_DIR = PROJECT_ROOT / "ai_bridge"
LATEST_FRAME = OUTPUT_DIR / "latest_frame.jpg"
LATEST_FACE = OUTPUT_DIR / "latest_face.jpg"
LATEST_METADATA = OUTPUT_DIR / "latest_detection.json"
LATEST_RECOGNITION = OUTPUT_DIR / "latest_recognition.json"
ENROLLMENT_STATUS_FILE = OUTPUT_DIR / "enrollment_status.json"

DEFAULT_ENROLLMENT_IMAGES = 5
MIN_ENROLLMENT_CONFIDENCE = 0.80
MIN_FACE_WIDTH = 120
MIN_FACE_HEIGHT = 120
MIN_SAVE_INTERVAL_SECONDS = 1.5

# Reject enrollment crops that touch the frame edge.
EDGE_MARGIN_PIXELS = 8


# ----------------------------------------------------------------------
# Application state
# ----------------------------------------------------------------------

app = Flask(__name__)

write_lock = threading.Lock()
recognition_lock = threading.Lock()
enrollment_lock = threading.Lock()

recognition_engine = FaceRecognitionEngine()
recognition_ready = False
recognition_load_error = ""

enrollment_state: dict[str, object] = {
    "active": False,
    "person_name": "",
    "folder_name": "",
    "target_images": DEFAULT_ENROLLMENT_IMAGES,
    "saved_images": 0,
    "last_saved_at": 0.0,
    "last_message": "Enrollment is idle.",
    "started_at": "",
    "completed_at": "",
}


# ----------------------------------------------------------------------
# Recognition model
# ----------------------------------------------------------------------

def load_recognition_model() -> bool:
    global recognition_ready
    global recognition_load_error

    try:
        with recognition_lock:
            recognition_engine.load()

        recognition_ready = True
        recognition_load_error = ""

        print(
            "LBPH MODEL LOADED:",
            {
                "threshold": recognition_engine.threshold,
                "labels": recognition_engine.label_to_name,
            },
            flush=True,
        )

        return True

    except Exception as error:
        recognition_ready = False
        recognition_load_error = str(error)

        print(
            "LBPH MODEL NOT READY:",
            recognition_load_error,
            flush=True,
        )

        return False


def retrain_recognition_model() -> int:
    global recognition_ready
    global recognition_load_error

    with recognition_lock:
        image_count = recognition_engine.train()

    recognition_ready = True
    recognition_load_error = ""

    print(
        "LBPH MODEL RETRAINED:",
        {
            "training_images": image_count,
            "labels": recognition_engine.label_to_name,
            "threshold": recognition_engine.threshold,
        },
        flush=True,
    )

    return image_count


def recognize_face(
    face_crop: np.ndarray,
) -> dict[str, object]:
    if not recognition_ready:
        return {
            "ready": False,
            "name": "Unknown",
            "distance": None,
            "threshold": recognition_engine.threshold,
            "authorized": False,
            "result": "MODEL NOT READY",
            "error": recognition_load_error,
        }

    try:
        with recognition_lock:
            name, distance, matched = (
                recognition_engine.predict(
                    face_crop
                )
            )

        return {
            "ready": True,
            "name": name,
            "distance": round(
                float(distance),
                2,
            ),
            "threshold": recognition_engine.threshold,
            "authorized": bool(matched),
            "result": (
                "AUTHORIZED"
                if matched
                else "UNKNOWN"
            ),
            "error": "",
        }

    except Exception as error:
        return {
            "ready": True,
            "name": "Unknown",
            "distance": None,
            "threshold": recognition_engine.threshold,
            "authorized": False,
            "result": "ERROR",
            "error": str(error),
        }


# ----------------------------------------------------------------------
# Image helpers
# ----------------------------------------------------------------------

def decode_jpeg_base64(
    encoded: str,
) -> np.ndarray | None:
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
        not isinstance(
            bounding_box,
            (list, tuple),
        )
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

    x1 = max(
        0,
        min(x1, width - 1),
    )
    y1 = max(
        0,
        min(y1, height - 1),
    )
    x2 = max(
        0,
        min(x2, width),
    )
    y2 = max(
        0,
        min(y2, height),
    )

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def clean_person_name(
    name: str,
) -> str:
    name = re.sub(
        r"[^A-Za-z0-9 _-]",
        "",
        name.strip(),
    )
    name = re.sub(
        r"\s+",
        " ",
        name,
    )
    return name[:60].strip()


def person_folder_name(
    person_name: str,
) -> str:
    return person_name.replace(
        " ",
        "_",
    )


def next_image_path(
    person_dir: Path,
) -> Path:
    highest = 0

    for image_path in person_dir.glob(
        "*.jpg"
    ):
        try:
            highest = max(
                highest,
                int(image_path.stem),
            )
        except ValueError:
            continue

    return (
        person_dir
        / f"{highest + 1:04d}.jpg"
    )


def box_touches_edge(
    box: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
) -> bool:
    x1, y1, x2, y2 = box

    return (
        x1 <= EDGE_MARGIN_PIXELS
        or y1 <= EDGE_MARGIN_PIXELS
        or x2 >= frame_width - EDGE_MARGIN_PIXELS
        or y2 >= frame_height - EDGE_MARGIN_PIXELS
    )


# ----------------------------------------------------------------------
# Enrollment state
# ----------------------------------------------------------------------

def get_enrollment_status() -> dict[str, object]:
    with enrollment_lock:
        state = enrollment_state.copy()

    state["remaining_images"] = max(
        0,
        int(state["target_images"])
        - int(state["saved_images"]),
    )

    return state


def save_enrollment_status_file() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ENROLLMENT_STATUS_FILE.write_text(
        json.dumps(
            get_enrollment_status(),
            indent=2,
        ),
        encoding="utf-8",
    )


def update_enrollment_state(
    **updates: object,
) -> None:
    with enrollment_lock:
        enrollment_state.update(updates)

    save_enrollment_status_file()


def begin_enrollment(
    person_name: str,
    target_images: int,
) -> dict[str, object]:
    cleaned_name = clean_person_name(
        person_name
    )

    if not cleaned_name:
        raise ValueError(
            "Enter a valid person name."
        )

    target_images = max(
        1,
        min(
            int(target_images),
            25,
        ),
    )

    update_enrollment_state(
        active=True,
        person_name=cleaned_name,
        folder_name=person_folder_name(
            cleaned_name
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

    print(
        "AI ENROLLMENT STARTED:",
        get_enrollment_status(),
        flush=True,
    )

    return get_enrollment_status()


def cancel_enrollment() -> dict[str, object]:
    update_enrollment_state(
        active=False,
        last_message="Enrollment cancelled.",
        completed_at=datetime.now().isoformat(
            timespec="seconds"
        ),
    )

    print(
        "AI ENROLLMENT CANCELLED",
        flush=True,
    )

    return get_enrollment_status()


def enrollment_quality_check(
    detector_confidence: float,
    box: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
) -> tuple[bool, str]:
    x1, y1, x2, y2 = box
    face_width = x2 - x1
    face_height = y2 - y1

    if (
        detector_confidence
        < MIN_ENROLLMENT_CONFIDENCE
    ):
        return (
            False,
            (
                "Hold still: detector confidence "
                f"{detector_confidence:.2f} is below "
                f"{MIN_ENROLLMENT_CONFIDENCE:.2f}."
            ),
        )

    if (
        face_width < MIN_FACE_WIDTH
        or face_height < MIN_FACE_HEIGHT
    ):
        return (
            False,
            (
                "Move closer: face crop is "
                f"{face_width}x{face_height}."
            ),
        )

    if box_touches_edge(
        box,
        frame_width,
        frame_height,
    ):
        return (
            False,
            "Center your full face inside the frame.",
        )

    with enrollment_lock:
        elapsed = (
            time.monotonic()
            - float(
                enrollment_state[
                    "last_saved_at"
                ]
            )
        )

    if (
        elapsed
        < MIN_SAVE_INTERVAL_SECONDS
    ):
        return (
            False,
            "Hold position for the next image.",
        )

    return True, "Enrollment crop accepted."


def save_enrollment_crop(
    face_crop: np.ndarray,
) -> dict[str, object]:
    with enrollment_lock:
        person_name = str(
            enrollment_state[
                "person_name"
            ]
        )
        folder_name = str(
            enrollment_state[
                "folder_name"
            ]
        )
        target_images = int(
            enrollment_state[
                "target_images"
            ]
        )
        saved_images = int(
            enrollment_state[
                "saved_images"
            ]
        )

    person_dir = (
        config.FACES_DIR
        / folder_name
    )

    person_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination = next_image_path(
        person_dir
    )

    if not cv2.imwrite(
        str(destination),
        face_crop,
    ):
        raise RuntimeError(
            "Unable to save enrollment image."
        )

    saved_images += 1

    update_enrollment_state(
        saved_images=saved_images,
        last_saved_at=time.monotonic(),
        last_message=(
            f"Saved image {saved_images} "
            f"of {target_images} for "
            f"{person_name}."
        ),
    )

    print(
        "AI ENROLLMENT IMAGE SAVED:",
        {
            "person": person_name,
            "image": saved_images,
            "target": target_images,
            "file": str(destination),
        },
        flush=True,
    )

    completed = (
        saved_images >= target_images
    )

    training_images = None

    if completed:
        try:
            training_images = (
                retrain_recognition_model()
            )

            update_enrollment_state(
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

            print(
                "AI ENROLLMENT COMPLETED:",
                get_enrollment_status(),
                flush=True,
            )

        except Exception as error:
            update_enrollment_state(
                active=False,
                completed_at=datetime.now().isoformat(
                    timespec="seconds"
                ),
                last_message=(
                    "Images were saved, but model "
                    f"training failed: {error}"
                ),
            )

            print(
                "AI ENROLLMENT TRAINING FAILED:",
                error,
                flush=True,
            )

    return {
        "saved": True,
        "completed": completed,
        "training_images": training_images,
        "status": get_enrollment_status(),
    }


# ----------------------------------------------------------------------
# HTTP routes
# ----------------------------------------------------------------------

@app.get("/health")
def health():
    return jsonify(
        success=True,
        service=(
            "UNO Q AI detector + LBPH bridge"
        ),
        recognition_ready=recognition_ready,
        recognition_error=(
            recognition_load_error
        ),
        threshold=(
            recognition_engine.threshold
        ),
        enrollment=get_enrollment_status(),
    )


@app.post("/reload_model")
def reload_model():
    loaded = load_recognition_model()

    return jsonify(
        success=loaded,
        recognition_ready=(
            recognition_ready
        ),
        recognition_error=(
            recognition_load_error
        ),
        threshold=(
            recognition_engine.threshold
        ),
    ), 200 if loaded else 503


@app.post("/enrollment/start")
def enrollment_start():
    data = request.get_json(
        silent=True
    ) or {}

    try:
        status = begin_enrollment(
            person_name=str(
                data.get(
                    "name",
                    "",
                )
            ),
            target_images=int(
                data.get(
                    "images",
                    DEFAULT_ENROLLMENT_IMAGES,
                )
            ),
        )

    except (TypeError, ValueError) as error:
        return jsonify(
            success=False,
            message=str(error),
        ), 400

    return jsonify(
        success=True,
        enrollment=status,
    )


@app.post("/enrollment/cancel")
def enrollment_cancel():
    return jsonify(
        success=True,
        enrollment=cancel_enrollment(),
    )


@app.get("/enrollment/status")
def enrollment_status():
    return jsonify(
        success=True,
        enrollment=get_enrollment_status(),
    )


@app.post("/ai_detection")
def ai_detection():
    payload = request.get_json(
        silent=True
    )

    if not isinstance(payload, dict):
        return jsonify(
            success=False,
            message="JSON body is required.",
        ), 400

    if payload.get("label") != "face":
        return jsonify(
            success=False,
            message=(
                "Only face detections are accepted."
            ),
        ), 400

    encoded_frame = payload.get(
        "frame_jpeg_base64"
    )

    if not isinstance(
        encoded_frame,
        str,
    ):
        return jsonify(
            success=False,
            message=(
                "frame_jpeg_base64 is required."
            ),
        ), 400

    frame = decode_jpeg_base64(
        encoded_frame
    )

    if frame is None:
        return jsonify(
            success=False,
            message=(
                "Unable to decode JPEG frame."
            ),
        ), 400

    frame_height, frame_width = (
        frame.shape[:2]
    )

    box = clamp_box(
        payload.get(
            "bounding_box_xyxy"
        ),
        frame_width,
        frame_height,
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

    detector_confidence = float(
        payload.get(
            "confidence",
            0.0,
        )
    )

    enrollment = get_enrollment_status()
    enrollment_result = None

    if bool(enrollment["active"]):
        quality_ok, quality_message = (
            enrollment_quality_check(
                detector_confidence,
                box,
                frame_width,
                frame_height,
            )
        )

        if quality_ok:
            try:
                enrollment_result = (
                    save_enrollment_crop(
                        face_crop
                    )
                )

            except Exception as error:
                update_enrollment_state(
                    active=False,
                    last_message=(
                        "Enrollment failed: "
                        f"{error}"
                    ),
                    completed_at=(
                        datetime.now().isoformat(
                            timespec="seconds"
                        )
                    ),
                )

                enrollment_result = {
                    "saved": False,
                    "completed": False,
                    "error": str(error),
                    "status": (
                        get_enrollment_status()
                    ),
                }

        else:
            update_enrollment_state(
                last_message=quality_message
            )

            enrollment_result = {
                "saved": False,
                "completed": False,
                "message": quality_message,
                "status": (
                    get_enrollment_status()
                ),
            }

        recognition = {
            "ready": recognition_ready,
            "name": "Enrollment active",
            "distance": None,
            "threshold": (
                recognition_engine.threshold
            ),
            "authorized": False,
            "result": "ENROLLING",
            "error": "",
        }

    else:
        recognition = recognize_face(
            face_crop
        )

    received_at = datetime.now().isoformat(
        timespec="milliseconds"
    )

    detection_metadata = {
        "received_at": received_at,
        "label": "face",
        "confidence": detector_confidence,
        "bounding_box_xyxy": list(box),
        "frame_width": frame_width,
        "frame_height": frame_height,
        "face_width": x2 - x1,
        "face_height": y2 - y1,
    }

    recognition_metadata = {
        "received_at": received_at,
        "detector_confidence": (
            detector_confidence
        ),
        **recognition,
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with write_lock:
        frame_saved = cv2.imwrite(
            str(LATEST_FRAME),
            frame,
        )

        face_saved = cv2.imwrite(
            str(LATEST_FACE),
            face_crop,
        )

        LATEST_METADATA.write_text(
            json.dumps(
                detection_metadata,
                indent=2,
            ),
            encoding="utf-8",
        )

        LATEST_RECOGNITION.write_text(
            json.dumps(
                recognition_metadata,
                indent=2,
            ),
            encoding="utf-8",
        )

    if not frame_saved or not face_saved:
        return jsonify(
            success=False,
            message=(
                "Unable to save bridge images."
            ),
        ), 500

    if bool(enrollment["active"]):
        print(
            "AI ENROLLMENT STATUS:",
            get_enrollment_status(),
            flush=True,
        )
    else:
        print(
            "AI + LBPH RESULT:",
            {
                "detector_confidence": (
                    detector_confidence
                ),
                "name": recognition["name"],
                "distance": (
                    recognition["distance"]
                ),
                "threshold": (
                    recognition["threshold"]
                ),
                "result": (
                    recognition["result"]
                ),
            },
            flush=True,
        )

    return jsonify(
        success=True,
        detection=detection_metadata,
        recognition=(
            recognition_metadata
        ),
        enrollment=enrollment_result,
    )


# ----------------------------------------------------------------------
# Startup
# ----------------------------------------------------------------------

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_enrollment_status_file()
    load_recognition_model()

    print(
        "UNO Q AI Detector + "
        "LBPH Enrollment Bridge"
    )
    print(
        "Listening on "
        "http://0.0.0.0:5055"
    )
    print(
        f"Output directory: {OUTPUT_DIR}"
    )
    print(
        "Enrollment API:"
    )
    print(
        "  POST /enrollment/start"
    )
    print(
        "  POST /enrollment/cancel"
    )
    print(
        "  GET  /enrollment/status"
    )

    app.run(
        host="0.0.0.0",
        port=5055,
        debug=False,
        threaded=True,
    )
