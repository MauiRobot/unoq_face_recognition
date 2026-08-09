Milestone 7.3.2 - Automatic Startup at Boot

Install this folder as:
  /home/arduino/unoq_face_recognition/standalone-ai/autostart/

Then run:
  cd ~/unoq_face_recognition
  chmod +x standalone-ai/autostart/*.sh
  ./standalone-ai/autostart/install-autostart.sh

Verify:
  ./standalone-ai/autostart/status-autostart.sh
  Open http://192.168.4.124:5000

Then reboot:
  sudo reboot

After reboot, do not start App Lab and do not manually run app.py.
Open the dashboard and verify live video, recognition, LEDs, snapshots, and enrollment.

Rollback:
  ./standalone-ai/autostart/uninstall-autostart.sh

IMPORTANT:
The verified AI sender currently posts to:
  http://192.168.4.124:5000/ai_detection

For reliable appliance-style startup, the UNO Q should keep the local IP
192.168.4.124 (for example with a DHCP reservation or static LAN address).
