import cv2
from camera import Camera

camera = Camera()

frame = camera.get_frame()

if frame is None:
    print("Failed to capture frame")
else:
    print("Captured:", frame.shape)

    cv2.imwrite(
        "camera_test.jpg",
        frame
    )

    print("Saved camera_test.jpg")

camera.release()