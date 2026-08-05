#!/usr/bin/env python3

import threading
import time
from pathlib import Path
from typing import Callable, Optional


class LedChannel:
    def __init__(
        self,
        led_name: str,
        event_callback: Optional[Callable[[str], None]] = None,
    ):
        self.led_name = led_name
        self.event_callback = event_callback

        self.led_directory = Path("/sys/class/leds") / led_name
        self.brightness_file = self.led_directory / "brightness"
        self.max_brightness_file = self.led_directory / "max_brightness"
        self.trigger_file = self.led_directory / "trigger"

        self._lock = threading.Lock()
        self._active = False
        self._available = False
        self._max_brightness = 1
        self._last_error = ""

        self._initialize()

    def _log(self, message: str) -> None:
        if self.event_callback is not None:
            self.event_callback(message)

    def _initialize(self) -> None:
        if not self.brightness_file.exists():
            self._last_error = (
                f"LED interface not found: {self.brightness_file}"
            )
            self._log(
                f"LED {self.led_name} unavailable: {self._last_error}"
            )
            return

        try:
            if self.max_brightness_file.exists():
                value = self.max_brightness_file.read_text(
                    encoding="utf-8"
                ).strip()
                self._max_brightness = max(1, int(value))

            if self.trigger_file.exists():
                try:
                    self.trigger_file.write_text(
                        "none",
                        encoding="utf-8",
                    )
                except OSError:
                    pass

            self._write(False)
            self._available = True
            self._log(f"LED ready: {self.led_name}")

        except (OSError, ValueError) as error:
            self._last_error = str(error)
            self._available = False
            self._log(
                f"LED {self.led_name} initialization failed: {error}"
            )

    def _write(self, active: bool) -> None:
        value = self._max_brightness if active else 0
        self.brightness_file.write_text(
            str(value),
            encoding="utf-8",
        )

    def is_available(self) -> bool:
        return self._available

    def is_active(self) -> bool:
        with self._lock:
            return self._active

    def set(self, active: bool) -> bool:
        if not self._available:
            return False

        with self._lock:
            try:
                self._write(active)
                self._active = active
                return True
            except OSError as error:
                self._last_error = str(error)
                self._available = False
                self._active = False
                self._log(
                    f"LED {self.led_name} write failed: {error}"
                )
                return False

    def pulse(self, duration_seconds: float) -> bool:
        if not self._available:
            self._log(
                f"LED {self.led_name} pulse skipped: unavailable"
            )
            return False

        with self._lock:
            if self._active:
                return False
            self._active = True

        threading.Thread(
            target=self._pulse_worker,
            args=(float(duration_seconds),),
            name=f"pulse-{self.led_name}",
            daemon=True,
        ).start()

        return True

    def _pulse_worker(self, duration_seconds: float) -> None:
        try:
            self._write(True)
            self._log(
                f"LED {self.led_name} ON "
                f"for {duration_seconds:.1f} seconds"
            )
            time.sleep(max(0.0, duration_seconds))

        except OSError as error:
            self._last_error = str(error)
            self._available = False
            self._log(
                f"LED {self.led_name} pulse failed: {error}"
            )

        finally:
            try:
                if self.brightness_file.exists():
                    self._write(False)
            except OSError as error:
                self._last_error = str(error)
                self._available = False
                self._log(
                    f"LED {self.led_name} OFF failed: {error}"
                )

            with self._lock:
                self._active = False

            self._log(f"LED {self.led_name} OFF")

    def close(self) -> None:
        self.set(False)


class GPIOOutput:
    """
    UNO Q three-LED status controller.

    Green:
        Authorized recognition pulse.

    Red:
        Unknown recognition pulse.

    Blue:
        Heartbeat while the application is running.
    """

    def __init__(
        self,
        gpio_line: int = 41,
        led_name: str = "unoq:user-red1",
        event_callback: Optional[Callable[[str], None]] = None,
        heartbeat_enabled: bool = True,
        heartbeat_period_seconds: float = 2.0,
        heartbeat_on_seconds: float = 0.15,
    ):
        # gpio_line and led_name are retained for compatibility with app.py.
        self.gpio_line = int(gpio_line)
        self.legacy_led_name = led_name
        self.event_callback = event_callback

        self.green = LedChannel(
            "unoq:user-green1",
            event_callback,
        )
        self.red = LedChannel(
            "unoq:user-red1",
            event_callback,
        )
        self.blue = LedChannel(
            "unoq:user-blue1",
            event_callback,
        )

        self.heartbeat_enabled = heartbeat_enabled
        self.heartbeat_period_seconds = max(
            0.5,
            float(heartbeat_period_seconds),
        )
        self.heartbeat_on_seconds = max(
            0.05,
            min(
                float(heartbeat_on_seconds),
                self.heartbeat_period_seconds,
            ),
        )

        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None

        if self.heartbeat_enabled and self.blue.is_available():
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_worker,
                name="blue-led-heartbeat",
                daemon=True,
            )
            self._heartbeat_thread.start()

    def _log(self, message: str) -> None:
        if self.event_callback is not None:
            self.event_callback(message)

    def _heartbeat_worker(self) -> None:
        self._log("Blue heartbeat started")

        while not self._heartbeat_stop.is_set():
            self.blue.set(True)

            if self._heartbeat_stop.wait(
                self.heartbeat_on_seconds
            ):
                break

            self.blue.set(False)

            remaining = (
                self.heartbeat_period_seconds
                - self.heartbeat_on_seconds
            )

            if self._heartbeat_stop.wait(remaining):
                break

        self.blue.set(False)
        self._log("Blue heartbeat stopped")

    def pulse_authorized(
        self,
        duration_seconds: float = 5.0,
    ) -> bool:
        return self.green.pulse(duration_seconds)

    def pulse_unknown(
        self,
        duration_seconds: float = 5.0,
    ) -> bool:
        return self.red.pulse(duration_seconds)

    def pulse(
        self,
        duration_seconds: float = 5.0,
    ) -> bool:
        """
        Backward-compatible default pulse.

        Existing code that calls pulse() still produces an authorized
        indication on the green LED.
        """
        return self.pulse_authorized(duration_seconds)

    def is_available(self) -> bool:
        return (
            self.green.is_available()
            or self.red.is_available()
            or self.blue.is_available()
        )

    def is_active(self) -> bool:
        return (
            self.green.is_active()
            or self.red.is_active()
        )

    def get_status(self) -> dict[str, object]:
        return {
            "gpio_line": self.gpio_line,
            "green_available": self.green.is_available(),
            "green_active": self.green.is_active(),
            "red_available": self.red.is_available(),
            "red_active": self.red.is_active(),
            "blue_available": self.blue.is_available(),
            "blue_active": self.blue.is_active(),
            "heartbeat_enabled": self.heartbeat_enabled,
        }

    def set_low(self) -> None:
        self.green.set(False)
        self.red.set(False)

    def close(self) -> None:
        self._heartbeat_stop.set()

        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=2.0)

        self.green.close()
        self.red.close()
        self.blue.close()
