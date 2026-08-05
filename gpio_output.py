#!/usr/bin/env python3

import threading
import time
from pathlib import Path
from typing import Callable, Optional


class GPIOOutput:
    """Non-blocking test output for UNO Q GPIO line 41.

    The current UNO Q Linux image exposes gpiochip1 line 41 through the
    kernel LED consumer named ``red:user``. This driver uses the Linux LED
    class instead of attempting to request the busy line directly.
    """

    def __init__(
        self,
        gpio_line: int = 41,
        led_name: str = "red:user",
        event_callback: Optional[Callable[[str], None]] = None,
    ):
        self.gpio_line = gpio_line
        self.led_name = led_name
        self.event_callback = event_callback
        self.led_directory = Path("/sys/class/leds") / led_name
        self.brightness_file = self.led_directory / "brightness"
        self.max_brightness_file = self.led_directory / "max_brightness"
        self._lock = threading.Lock()
        self._active = False
        self._available = False
        self._max_brightness = 1
        self._last_error = ""
        self._initialize()

    def _log(self, message: str) -> None:
        if self.event_callback:
            self.event_callback(message)

    def _write_value(self, value: int) -> None:
        output = self._max_brightness if value else 0
        self.brightness_file.write_text(str(output), encoding="utf-8")

    def _initialize(self) -> None:
        if not self.brightness_file.exists():
            self._last_error = f"LED interface not found: {self.brightness_file}"
            self._log(f"GPIO {self.gpio_line} unavailable: {self._last_error}")
            return

        try:
            if self.max_brightness_file.exists():
                value = self.max_brightness_file.read_text(encoding="utf-8").strip()
                self._max_brightness = max(1, int(value))
            self._write_value(0)
            self._available = True
            self._log(
                f"GPIO {self.gpio_line} test output ready through LED {self.led_name}"
            )
        except (OSError, ValueError) as error:
            self._last_error = str(error)
            self._log(f"GPIO {self.gpio_line} initialization failed: {error}")

    def is_available(self) -> bool:
        return self._available

    def is_active(self) -> bool:
        with self._lock:
            return self._active

    def get_status(self) -> dict[str, object]:
        return {
            "gpio_line": self.gpio_line,
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
            self._log(f"GPIO {self.gpio_line} pulse skipped: output unavailable")
            return False

        with self._lock:
            if self._active:
                return False
            self._active = True

        threading.Thread(
            target=self._pulse_worker,
            args=(duration_seconds,),
            name="gpio-output-pulse",
            daemon=True,
        ).start()
        return True

    def _pulse_worker(self, duration_seconds: float) -> None:
        try:
            self._write_value(1)
            self._log(
                f"GPIO {self.gpio_line} HIGH for {duration_seconds:.1f} seconds"
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
            self._log(f"GPIO {self.gpio_line} LOW")

    def close(self) -> None:
        self.set_low()
