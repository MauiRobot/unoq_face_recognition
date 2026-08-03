#!/usr/bin/env python3

import threading
import time

import cv2

import config


class Camera:
    def __init__(self):
        self.device = config.CAMERA_DEVICE

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
            config.FRAME_WIDTH,
        )
        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            config.FRAME_HEIGHT,
        )

        self.face_detector = cv2.CascadeClassifier(
            config.HAAR_CASCADE
        )

        if self.face_detector.empty():
            self.cap.release()
            raise RuntimeError(
                f"Unable to load Haar cascade: "
                f"{config.HAAR_CASCADE}"
            )

        self.last_frame_time = time.perf_counter()
        self.smoothed_fps = 0.0

        self.frame_lock = threading.Lock()
        self.last_raw_frame = None
        self.last_face_crop = None
        self.last_face_box = None

        print(f"Camera opened: {self.device}")
        print(
            "Face detector loaded: "
            f"{config.HAAR_CASCADE}"
        )

    def get_frame(self):
        frame_start = time.perf_counter()

        ok, frame = self.cap.read()

        if not ok or frame is None:
            return None

        raw_frame = frame.copy()

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

        largest_face = None
        largest_face_box = None

        if len(faces) > 0:
            largest_face_box = max(
                faces,
                key=lambda item: item[2] * item[3],
            )

            x, y, width, height = largest_face_box

            largest_face = raw_frame[
                y:y + height,
                x:x + width,
            ].copy()

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

        with self.frame_lock:
            self.last_raw_frame = raw_frame
            self.last_face_crop = largest_face
            self.last_face_box = largest_face_box

        now = time.perf_counter()
        elapsed_since_last_frame = (
            now - self.last_frame_time
        )
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

        detected_face_count = (
            1 if largest_face is not None else 0
        )

        metrics = [
            f"FPS: {self.smoothed_fps:.1f}",
            f"Frame: {processing_ms:.1f} ms",
            f"Detect: {detection_ms:.1f} ms",
            f"Faces: {detected_face_count}",
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

    def get_latest_raw_frame(self):
        with self.frame_lock:
            if self.last_raw_frame is None:
                return None

            return self.last_raw_frame.copy()

    def get_latest_face_crop(self):
        with self.frame_lock:
            if self.last_face_crop is None:
                return None

            return self.last_face_crop.copy()

    def has_detected_face(self):
        with self.frame_lock:
            return self.last_face_crop is not None

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None