#!/usr/bin/env python3

import socket
import threading
import time

import msgpack


ROUTER_SOCKET = "/var/run/arduino-router.sock"

RPC_RELAY_ON = "relay_on"
RPC_RELAY_OFF = "relay_off"


class RelayOutput:
    def __init__(
        self,
        event_callback=None,
        pulse_seconds=5.0,
        timeout_seconds=2.0,
    ):
        self.event_callback = event_callback
        self.pulse_seconds = float(pulse_seconds)
        self.timeout_seconds = float(timeout_seconds)

        self._lock = threading.Lock()
        self._active = False
        self._available = False
        self._last_error = ""
        self._message_id = 0

        # Safety default: make sure D9 starts OFF.
        if self.off():
            self._log("D9 relay output ready (OFF)")
        else:
            self._log(
                "D9 relay output not ready: "
                f"{self._last_error}"
            )

    def _log(self, message):
        if self.event_callback is not None:
            self.event_callback(message)
        else:
            print(message, flush=True)

    def _next_message_id(self):
        with self._lock:
            self._message_id += 1
            return self._message_id

    def _rpc_call(self, method_name):
        message_id = self._next_message_id()

        request = [
            0,
            message_id,
            method_name,
            [],
        ]

        try:
            packed = msgpack.packb(
                request,
                use_bin_type=True,
            )

            with socket.socket(
                socket.AF_UNIX,
                socket.SOCK_STREAM,
            ) as client:
                client.settimeout(
                    self.timeout_seconds
                )

                client.connect(
                    ROUTER_SOCKET
                )

                client.sendall(packed)

                unpacker = msgpack.Unpacker(
                    raw=False
                )

                while True:
                    chunk = client.recv(4096)

                    if not chunk:
                        raise RuntimeError(
                            "Router closed connection "
                            "before sending response"
                        )

                    unpacker.feed(chunk)

                    for response in unpacker:
                        if (
                            not isinstance(response, list)
                            or len(response) != 4
                        ):
                            continue

                        # MessagePack RPC response:
                        # [1, msgid, error, result]
                        if response[0] != 1:
                            continue

                        if response[1] != message_id:
                            continue

                        error = response[2]

                        if error not in (None, ""):
                            raise RuntimeError(
                                f"Bridge RPC error: {error}"
                            )

                        self._available = True
                        self._last_error = ""

                        return True

        except Exception as error:
            self._available = False
            self._last_error = str(error)
            return False

    def on(self):
        if self._rpc_call(RPC_RELAY_ON):
            with self._lock:
                self._active = True
            return True

        return False

    def off(self):
        if self._rpc_call(RPC_RELAY_OFF):
            with self._lock:
                self._active = False
            return True

        return False

    def pulse(self, duration_seconds=None):
        if duration_seconds is None:
            duration_seconds = self.pulse_seconds

        duration_seconds = float(
            duration_seconds
        )

        with self._lock:
            if self._active:
                self._log(
                    "D9 relay pulse skipped: "
                    "already active"
                )
                return False

            self._active = True

        thread = threading.Thread(
            target=self._pulse_worker,
            args=(duration_seconds,),
            name="d9-relay-pulse",
            daemon=True,
        )

        thread.start()

        return True

    def _pulse_worker(self, duration_seconds):
        relay_started = False

        try:
            if not self._rpc_call(RPC_RELAY_ON):
                self._log(
                    "D9 relay ON failed: "
                    f"{self._last_error}"
                )
                return

            relay_started = True

            self._log(
                "D9 relay ON for "
                f"{duration_seconds:.1f} seconds"
            )

            time.sleep(duration_seconds)

        finally:
            if relay_started:
                if self._rpc_call(RPC_RELAY_OFF):
                    self._log("D9 relay OFF")
                else:
                    self._log(
                        "D9 relay OFF failed: "
                        f"{self._last_error}"
                    )

            with self._lock:
                self._active = False

    def is_active(self):
        with self._lock:
            return self._active

    def get_status(self):
        return {
            "available": self._available,
            "active": self.is_active(),
            "socket": ROUTER_SOCKET,
            "relay_on_method": RPC_RELAY_ON,
            "relay_off_method": RPC_RELAY_OFF,
            "last_error": self._last_error,
        }

    def close(self):
        self.off()