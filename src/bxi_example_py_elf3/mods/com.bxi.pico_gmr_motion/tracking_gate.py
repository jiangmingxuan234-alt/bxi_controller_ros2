"""Thread-safe PICO A+X tracking toggle shared by SDK workers."""

from __future__ import annotations

from threading import Lock


class TrackingGate:
    """Toggle live tracking on the rising edge of the PICO A+X combo."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._enabled = False
        self._combo_pressed = False

    def update(self, a_pressed: bool, x_pressed: bool) -> bool | None:
        combo_pressed = bool(a_pressed) and bool(x_pressed)
        with self._lock:
            changed = combo_pressed and not self._combo_pressed
            self._combo_pressed = combo_pressed
            if not changed:
                return None
            self._enabled = not self._enabled
            return self._enabled

    def is_enabled(self) -> bool:
        with self._lock:
            return self._enabled


__all__ = ["TrackingGate"]
