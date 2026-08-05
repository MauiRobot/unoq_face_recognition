#!/usr/bin/env python3

import threading
import time
from pathlib import Path
from typing import Callable, Optional


class GPIOOutput:
    """Non-blocking UNO Q LED-backed GPIO test output."""

    LED_ALIASES = {
        "red:user": "unoq:user-red1",
        "red": "unoq:user-red1",
        "green": "unoq:user-green1",
        "blue": "unoq:user-blue1",
        "unoq:user-red1": "unoq:user-red1",
        "unoq:user-green1": "unoq:user-green1",
        "unoq:user-blue1": "unoq:user-blue1",
    }

    def __init__(
        self,
        gpio_line: int = 41,
        led_name: str = "unoq:user-red1",
        event_callback: Optional[Callable[[str], None]] = None,
    ):
        self.gpio_line = int(gpio_line)
        self.requested_led_name = led_name
        self.led_name = self.LED_ALIASES.get(led_name, led_name)
        self.event_callback = event_callback

        self.led_directory = Path("/sys/class/leds") / self.led_name
        self.brightness_file = self.led_directory / "brightness"
        self.max_brightness_file = self.led_directory / "max_brightness"
        self.trigger_file = self.led_directory / "trigger"

        self._lock = threading.Lock()
        self._pulse_thread: threading.Thread | None = None
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
            self._last_error = f"LED interface not found: {self.brightness_file}"
            self._log(
                f"GPIO {self.gpio_line} unavailable: {self._last_error}"
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
                    self.trigger_file.write_text("none", encoding="utf-8")
                except OSError:
                    pass

            self._write_value(0)
            self._available = True

            alias_note = ""
            if self.requested_led_name != self.led_name:
                alias_note = f" (mapped from {self.requested_led_name})"

            self._log(
                f"GPIO {self.gpio_line} test output ready through LED "
                f"{self.led_name}{alias_note}"
            )

        except (OSError, ValueError) as error:
            self._last_error = str(error)
            self._available = False
            self._log(
                f"GPIO {self.gpio_line} initialization failed: {error}"
            )

    def _write_value(self, value: int) -> None:
        output_value = self._max_brightness if value else 0
        self.brightness_file.write_text(str(output_value), encoding="utf-8")

    def is_available(self) -> bool:
        return self._available

    def is_active(self) -> bool:
        with self._lock:
            return self._active

    def get_status(self) -> dict[str, object]:
        return {
            "gpio_line": self.gpio_line,
            "requested_led_name": self.requested_led_name,
            "led_name": self.led_name,
            "available": self._available,
            "active": self.is_active(),
            "last_error": self._last_error,
        }

    def set_low(self) -> None:
        if not self._available:
            return

        with self._lock:
            try:
                self._write_value(0)
            except OSError as error:
                self._last_error = str(error)
                self._available = False
                self._log(f"GPIO {self.gpio_line} LOW failed: {error}")
            finally:
                self._active = False

    def pulse(self, duration_seconds: float = 5.0) -> bool:
        if not self._available:
            self._log(
                f"GPIO {self.gpio_line} pulse skipped: output unavailable"
            )
            return False

        with self._lock:
            if self._active:
                return False
            self._active = True

        self._pulse_thread = threading.Thread(
            target=self._pulse_worker,
            args=(float(duration_seconds),),
            name="gpio-output-pulse",
            daemon=True,
        )
        self._pulse_thread.start()
        return True

    def _pulse_worker(self, duration_seconds: float) -> None:
        try:
            self._write_value(1)
            self._log(
                f"GPIO {self.gpio_line} HIGH through {self.led_name} "
                f"for {duration_seconds:.1f} seconds"
            )
            time.sleep(max(0.0, duration_seconds))

        except OSError as error:
            self._last_error = str(error)
            self._available = False
            self._log(f"GPIO {self.gpio_line} pulse failed: {error}")

        finally:
            try:
                if self.brightness_file.exists():
                    self._write_value(0)
            except OSError as error:
                self._last_error = str(error)
                self._available = False
                self._log(f"GPIO {self.gpio_line} LOW failed: {error}")

            with self._lock:
                self._active = False

            self._log(
                f"GPIO {self.gpio_line} LOW through {self.led_name}"
            )

    def close(self) -> None:
        self.set_low()
