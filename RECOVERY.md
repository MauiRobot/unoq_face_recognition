# Arduino UNO Q Face Recognition - Recovery Guide

## Known-Working Configuration

This document describes the known-working Arduino UNO Q face-recognition
system and the procedure required to recover it after a fresh installation
or system failure.

The system has been verified to start automatically after an UNO Q reboot.

## System Architecture

Boot sequence:

    arduino-router.service
            |
            v
    unoq-face-relay.service
            |
            v
    MCU D9 Relay Arduino App
    relay_on / relay_off RPC registered
            |
            v
    unoq-face-ai.service
            |
            v
    AI detector containers
            |
            v
    unoq-face-recognition.service
            |
            v
    Flask Face Recognition Dashboard
            |
            v
    AUTHORIZED FACE
            |
            v
    D9 HIGH for 5 seconds
            |
            v
    D9 LOW

## Project Location

Main project:

    /home/arduino/unoq_face_recognition

The Git repository contains the face-recognition application, standalone AI
configuration, MCU relay Arduino App, and systemd service files.

Dashboard:

    http://192.168.4.124:5000

## MCU Relay Hardware

The relay is controlled by the UNO Q STM32U5 microcontroller.

Relay output:

    D9

Monitor/verification input:

    D8

Authorized recognition pulses D9 HIGH for 5 seconds and then returns it LOW.

Expected physical verification:

    Relay OFF, D8 reads: 0
    Relay ON, D8 reads: 1
    Relay OFF, D8 reads: 0

## MCU Arduino App

The repository-managed MCU relay Arduino App is located at:

    /home/arduino/unoq_face_recognition/mcu-relay

Structure:

    mcu-relay/
        app.yaml
        sketch/
            sketch.ino
            sketch.yaml
        python/
            main.py

The sketch registers these RouterBridge RPC methods:

    relay_on
    relay_off

IMPORTANT:

Do not rely on raw arduino-flash alone for this system.

The known-working method is to start the MCU sketch through the Arduino App
lifecycle so the RouterBridge RPC methods are registered correctly.

Known-working command:

    arduino-app-cli app start /home/arduino/unoq_face_recognition/mcu-relay

Under normal operation this command is executed automatically by systemd.
Do not manually start another copy while the systemd service owns the app.

## Automatic Startup Services

The system uses these project services:

    unoq-face-relay.service
    unoq-face-ai.service
    unoq-face-recognition.service

The relay service starts first, after arduino-router.

The AI service starts after the relay service.

The face-recognition service starts after the AI service.

## Boot Verification

After reboot, do not manually start anything.

Check:

    systemctl is-active arduino-router.service
    systemctl is-active unoq-face-relay.service
    systemctl is-active unoq-face-ai.service
    systemctl is-active unoq-face-recognition.service

Expected result:

    active
    active
    active
    active

## Relay Troubleshooting

If this appears:

    Bridge RPC error: [2, 'method relay_on not available']

or:

    Bridge RPC error: [2, 'method relay_off not available']

check:

    systemctl status unoq-face-relay.service

The relay Arduino App must be running through the Arduino App lifecycle.

## Known-Good Final Test

After a full reboot, the system restored automatically.

Authorized recognition produced:

    D9 relay ON for 5.0 seconds
    D9 relay OFF

MCU verification:

    Relay ON, D8 reads: 1
    Relay OFF, D8 reads: 0

This confirms the complete automatic startup chain is working.
