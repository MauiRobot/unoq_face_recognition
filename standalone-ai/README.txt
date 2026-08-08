Milestone 7.3.1 - Standalone Offline AI Runtime

Goal:
Run the Arduino AI face detector without opening Arduino App Lab.

Required local Docker images:
- ghcr.io/arduino/app-bricks/ei-models-runner:0.11.2
- ghcr.io/arduino/app-bricks/python-apps-base:0.11.0

Face model inside cached runner image:
- /models/ootb/ei/lw-face-det.eim

Install these files into:
  /home/arduino/unoq_face_recognition/standalone-ai/

Then:
  chmod +x standalone-ai/*.sh

The local source directory:
  /home/arduino/ArduinoApps/test-of-face-detector-on-camera

is still mounted into the camera container, but Arduino App Lab itself does not need to be open or running.

The main.py in that local directory must be the verified version that sends detections to:
  http://192.168.4.124:5000/ai_detection

FIRST TEST
1. Stop/close the App Lab project.
2. Start Flask:
   cd ~/unoq_face_recognition
   source .venv/bin/activate
   python app.py

3. In another SSH window:
   cd ~/unoq_face_recognition
   ./standalone-ai/start-ai.sh

4. Check:
   ./standalone-ai/status-ai.sh

5. Open:
   http://192.168.4.124:5000

6. Verify live AI feed, recognition, LEDs, snapshots, and enrollment.

OFFLINE TEST
After the standalone test works:
- remove Internet access but keep the local network,
- restart Flask and the standalone AI runtime,
- repeat the same verification.

Do not configure boot auto-start until the offline test passes.
