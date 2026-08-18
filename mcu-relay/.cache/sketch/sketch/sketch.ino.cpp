#include <Arduino.h>
#line 1 "/home/arduino/unoq_face_recognition/mcu-relay/sketch/sketch.ino"
#include "Arduino_RouterBridge.h"

#define RELAY_PIN 9
#define MONITOR_PIN 8


#line 7 "/home/arduino/unoq_face_recognition/mcu-relay/sketch/sketch.ino"
void relay_on();
#line 19 "/home/arduino/unoq_face_recognition/mcu-relay/sketch/sketch.ino"
void relay_off();
#line 31 "/home/arduino/unoq_face_recognition/mcu-relay/sketch/sketch.ino"
void setup();
#line 55 "/home/arduino/unoq_face_recognition/mcu-relay/sketch/sketch.ino"
void loop();
#line 7 "/home/arduino/unoq_face_recognition/mcu-relay/sketch/sketch.ino"
void relay_on() {
    digitalWrite(RELAY_PIN, HIGH);

    delay(50);

    int state = digitalRead(MONITOR_PIN);

    Monitor.print("Relay ON, D8 reads: ");
    Monitor.println(state);
}


void relay_off() {
    digitalWrite(RELAY_PIN, LOW);

    delay(50);

    int state = digitalRead(MONITOR_PIN);

    Monitor.print("Relay OFF, D8 reads: ");
    Monitor.println(state);
}


void setup() {
    Bridge.begin();
    Monitor.begin();

    pinMode(RELAY_PIN, OUTPUT);
    pinMode(MONITOR_PIN, INPUT);

    // Safe startup state
    digitalWrite(RELAY_PIN, LOW);

    Bridge.provide_safe(
        "relay_on",
        relay_on
    );

    Bridge.provide_safe(
        "relay_off",
        relay_off
    );

    Monitor.println("D9 Relay Bridge ready");
}


void loop() {
    // Relay is controlled through Bridge RPC.
}
