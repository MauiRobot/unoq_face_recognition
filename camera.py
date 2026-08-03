#!/usr/bin/env python3

import time

import cv2


class Camera:
    def __init__(
        self,
        device: str = "/dev/video2",
        width: int = 640,
        height: int = 480,
    ):
        self.device = device
        self.cap = cv2.VideoCapture(
            self.device,
            cv2.CAP_V4L2,
        )

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Unable to open camera {self.device}"
            )

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            width,
        )
        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            height,
        )

        cascade_path = (
            "/usr/share/opencv4/haarcascades/"
            "haarcascade_frontalface_default.xml"
        )

        self.face_detector = cv2.CascadeClassifier(
            cascade_path
        )

        if self.face_detector.empty():
            self.cap.release()
            raise RuntimeError(
                f"Unable to load Haar cascade: {cascade_path}"
            )

        self.last_frame_time = time.perf_counter()
        self.smoothed_fps = 0.0

        print(f"Camera opened: {self.device}")
        print(f"Face detector loaded: {cascade_path}")

    def get_frame(self):
        frame_start = time.perf_counter()

        ok, frame = self.cap.read()

        if not ok or frame is None:
            return None

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        gray = cv2.equalizeHist(gray)

        detection_start = time.perf_counter()

        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )

        detection_end = time.perf_counter()

        for x, y, width, height in faces:
            cv2.rectangle(
                frame,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                "Face",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        now = time.perf_counter()
        elapsed_since_last_frame = now - self.last_frame_time
        self.last_frame_time = now

        instantaneous_fps = (
            1.0 / elapsed_since_last_frame
            if elapsed_since_last_frame > 0
            else 0.0
        )

        if self.smoothed_fps == 0.0:
            self.smoothed_fps = instantaneous_fps
        else:
            self.smoothed_fps = (
                0.9 * self.smoothed_fps
                + 0.1 * instantaneous_fps
            )

        detection_ms = (
            detection_end - detection_start
        ) * 1000.0

        processing_ms = (
            time.perf_counter() - frame_start
        ) * 1000.0

        metrics = [
            f"FPS: {self.smoothed_fps:.1f}",
            f"Frame: {processing_ms:.1f} ms",
            f"Detect: {detection_ms:.1f} ms",
            f"Faces: {len(faces)}",
        ]

        y_position = 28

        for metric in metrics:
            cv2.putText(
                frame,
                metric,
                (15, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            y_position += 26

        return frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None