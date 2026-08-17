# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l.
#
# SPDX-License-Identifier: MPL-2.0

from arduino.app_utils import App
from arduino.app_utils.image import get_image_bytes
from arduino.app_bricks.web_ui import WebUI
from arduino.app_bricks.video_objectdetection import VideoObjectDetection

from datetime import datetime, UTC

import base64
import json
import urllib.request


FLASK_AI_ENDPOINT = "http://192.168.4.124:5000/ai_detection"


ui = WebUI()

detection_stream = VideoObjectDetection(
    confidence=0.5,
    debounce_sec=0.0,
    camera_preview=True,
)


ui.on_message(
    "override_th",
    lambda sid, threshold:
        detection_stream.override_threshold(
            threshold
        )
)


def post_detection_to_flask(
    label,
    detection,
    frame,
):
    try:
        if frame is None:
            return

        jpeg_bytes = get_image_bytes(
            frame
        )

        encoded_frame = base64.b64encode(
            jpeg_bytes
        ).decode("ascii")

        box = detection.get(
            "bounding_box_xyxy"
        )

        if box is None:
            return

        payload = {
            "label": label,
            "confidence": detection.get(
                "confidence",
                0.0,
            ),
            "bounding_box_xyxy": list(
                box
            ),
            "frame_jpeg_base64":
                encoded_frame,
        }

        body = json.dumps(
            payload
        ).encode("utf-8")

        request = urllib.request.Request(
            FLASK_AI_ENDPOINT,
            data=body,
            headers={
                "Content-Type":
                    "application/json"
            },
            method="POST",
        )

        with urllib.request.urlopen(
            request,
            timeout=2.0,
        ) as response:

            result = response.read()

            print(
                "MAIN APP RESPONSE:",
                response.status,
                result.decode(
                    "utf-8",
                    errors="replace",
                ),
                flush=True,
            )

    except Exception as error:
        print(
            "Flask forwarding error:",
            error,
            flush=True,
        )


def on_all_detections(
    detections: dict,
    frame: bytes,
):
    for key, values in detections.items():

        for value in values:

            entry = {
                "content": key,
                "confidence":
                    value.get(
                        "confidence"
                    ),
                "timestamp":
                    datetime.now(
                        UTC
                    ).isoformat(),
            }

            ui.send_message(
                "detection",
                message=entry,
            )

            if key == "face":

                print(
                    "SENDING TO MAIN APP:",
                    {
                        "confidence":
                            value.get(
                                "confidence"
                            ),
                        "bounding_box_xyxy":
                            value.get(
                                "bounding_box_xyxy"
                            ),
                        "frame_bytes":
                            len(frame)
                            if frame
                            else 0,
                    },
                    flush=True,
                )

                post_detection_to_flask(
                    key,
                    value,
                    frame,
                )


detection_stream.on_detect_all(
    on_all_detections
)


App.run()