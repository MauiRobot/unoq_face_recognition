#!/usr/bin/env python3

import json
from pathlib import Path

import cv2
import numpy as np

import config


MODEL_FILE = config.DATABASE_DIR / "lbph_model.yml"
LABELS_FILE = config.DATABASE_DIR / "labels.json"

FACE_SIZE = (200, 200)

# LBPH uses a distance score:
# lower = better match
# higher = weaker match
LBPH_UNKNOWN_THRESHOLD = 65.0


class FaceRecognitionEngine:
    def __init__(
        self,
        threshold: float = LBPH_UNKNOWN_THRESHOLD,
    ):
        if not hasattr(cv2, "face"):
            raise RuntimeError(
                "This OpenCV build does not include cv2.face."
            )

        if not hasattr(
            cv2.face,
            "LBPHFaceRecognizer_create",
        ):
            raise RuntimeError(
                "LBPH face recognition is unavailable."
            )

        self.recognizer = (
            cv2.face.LBPHFaceRecognizer_create()
        )

        self.label_to_name: dict[int, str] = {}
        self.threshold = float(threshold)
        self.model_loaded = False

    def prepare_face(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        if image is None:
            raise ValueError("Face image is empty.")

        if len(image.shape) == 3:
            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY,
            )
        else:
            gray = image.copy()

        gray = cv2.equalizeHist(gray)

        prepared = cv2.resize(
            gray,
            FACE_SIZE,
            interpolation=cv2.INTER_AREA,
        )

        return prepared

    def load_training_images(
        self,
    ) -> tuple[list[np.ndarray], list[int]]:
        images: list[np.ndarray] = []
        labels: list[int] = []

        self.label_to_name.clear()

        if not config.FACES_DIR.exists():
            return images, labels

        person_dirs = sorted(
            path
            for path in config.FACES_DIR.iterdir()
            if path.is_dir()
        )

        label_id = 0

        for person_dir in person_dirs:
            image_paths = sorted(
                person_dir.glob("*.jpg")
            )

            valid_images = 0

            for image_path in image_paths:
                image = cv2.imread(
                    str(image_path)
                )

                if image is None:
                    print(
                        "Skipping unreadable image:",
                        image_path,
                    )
                    continue

                prepared = self.prepare_face(
                    image
                )

                images.append(prepared)
                labels.append(label_id)
                valid_images += 1

            if valid_images > 0:
                person_name = (
                    person_dir.name.replace(
                        "_",
                        " ",
                    )
                )

                self.label_to_name[
                    label_id
                ] = person_name

                print(
                    f"Loaded {valid_images} image(s) "
                    f"for {person_name}"
                )

                label_id += 1

        return images, labels

    def train(self) -> int:
        images, labels = (
            self.load_training_images()
        )

        if not images:
            raise RuntimeError(
                "No enrolled face images were found."
            )

        config.DATABASE_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.recognizer.train(
            images,
            np.array(labels),
        )

        self.recognizer.write(
            str(MODEL_FILE)
        )

        label_data = {
            str(label): name
            for label, name
            in self.label_to_name.items()
        }

        LABELS_FILE.write_text(
            json.dumps(
                label_data,
                indent=2,
            ),
            encoding="utf-8",
        )

        self.model_loaded = True

        return len(images)

    def load(self) -> None:
        if not MODEL_FILE.exists():
            raise RuntimeError(
                f"Recognition model not found: "
                f"{MODEL_FILE}"
            )

        if not LABELS_FILE.exists():
            raise RuntimeError(
                f"Label map not found: "
                f"{LABELS_FILE}"
            )

        self.recognizer.read(
            str(MODEL_FILE)
        )

        label_data = json.loads(
            LABELS_FILE.read_text(
                encoding="utf-8"
            )
        )

        self.label_to_name = {
            int(label): name
            for label, name
            in label_data.items()
        }

        self.model_loaded = True

    def is_ready(self) -> bool:
        return self.model_loaded

    def predict(
        self,
        image: np.ndarray,
    ) -> tuple[str, float, bool]:
        if not self.model_loaded:
            raise RuntimeError(
                "Recognition model is not loaded."
            )

        prepared = self.prepare_face(
            image
        )

        label, distance = (
            self.recognizer.predict(
                prepared
            )
        )

        matched_name = (
            self.label_to_name.get(
                label,
                "Unknown",
            )
        )

        is_match = (
            matched_name != "Unknown"
            and distance <= self.threshold
        )

        if not is_match:
            return (
                "Unknown",
                float(distance),
                False,
            )

        return (
            matched_name,
            float(distance),
            True,
        )

    def get_status(self) -> dict[str, object]:
        return {
            "ready": self.model_loaded,
            "threshold": self.threshold,
            "people": len(self.label_to_name),
            "model_file": str(MODEL_FILE),
            "labels_file": str(LABELS_FILE),
        }


def main() -> int:
    engine = FaceRecognitionEngine()

    print("Training LBPH recognition model...")

    image_count = engine.train()

    print(
        f"Training complete using "
        f"{image_count} image(s)."
    )

    print(f"Saved model: {MODEL_FILE}")
    print(f"Saved labels: {LABELS_FILE}")
    print(
        f"Unknown threshold: "
        f"{engine.threshold:.2f}"
    )

    if not config.FACE_IMAGE.exists():
        print(
            "No captured face is available "
            "for a recognition test."
        )
        return 0

    test_image = cv2.imread(
        str(config.FACE_IMAGE)
    )

    if test_image is None:
        print(
            "Unable to read captured face image."
        )
        return 1

    engine.load()

    name, distance, is_match = engine.predict(
        test_image
    )

    print(f"Test result: {name}")
    print(f"LBPH distance: {distance:.2f}")
    print(
        "Decision:",
        "AUTHORIZED" if is_match else "UNKNOWN",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())