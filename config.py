#!/usr/bin/env python3

from pathlib import Path

# ----------------------------------------------------------------------
# Project directories
# ----------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent

CAPTURED_DIR = PROJECT_ROOT / "captured"
FACES_DIR = PROJECT_ROOT / "faces"
DATABASE_DIR = PROJECT_ROOT / "database"

# ----------------------------------------------------------------------
# Camera
# ----------------------------------------------------------------------

CAMERA_DEVICE = "/dev/video3"

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ----------------------------------------------------------------------
# Face Detection
# ----------------------------------------------------------------------

HAAR_CASCADE = (
    "/usr/share/opencv4/haarcascades/"
    "haarcascade_frontalface_default.xml"
)

# ----------------------------------------------------------------------
# Image filenames
# ----------------------------------------------------------------------

FRAME_IMAGE = CAPTURED_DIR / "frame.jpg"
FACE_IMAGE = CAPTURED_DIR / "face.jpg"