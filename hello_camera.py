#!/usr/bin/env python3

import sys
import cv2

CAMERA_DEVICE = "/dev/video2"
OUTPUT_FILE = "hello_camera.jpg"


def main():
    print("Arduino UNO Q - Hello Camera")

    camera = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)

    if not camera.isOpened():
        print("Could not open camera.")
        return 1

    ok, frame = camera.read()

    camera.release()

    if not ok:
        print("Failed to capture frame.")
        return 2

    h, w = frame.shape[:2]

    print(f"Captured {w} x {h}")

    cv2.putText(
        frame,
        "Hello Camera",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    cv2.imwrite(OUTPUT_FILE, frame)

    print("Saved", OUTPUT_FILE)

    return 0


if __name__ == "__main__":
    sys.exit(main())
