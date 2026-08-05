"""Compatibility wrapper for returning and blocking XRoboToolkit SDK builds."""

from __future__ import annotations

from threading import Event, Thread
from typing import Protocol


class XrtSdk(Protocol):
    def init(self) -> object:
        ...

    def close(self) -> object:
        ...


class XrtBackgroundSession:
    """Run ``xrt.init()`` off the data loop because some SDK builds block.

    XRoboToolkit releases exist with both behaviours: some return after
    connecting, while others keep ``init()`` inside the client stream until
    ``close()`` is called.  In both cases the SDK query functions are usable
    after the initializer thread has entered ``init()``.
    """

    def __init__(self, sdk: XrtSdk) -> None:
        self._sdk = sdk
        self._started = Event()
        self._closing = Event()
        self._thread: Thread | None = None
        self._error: BaseException | None = None

    @property
    def error(self) -> BaseException | None:
        return self._error

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("XRoboToolkit session is already started")

        def initialize() -> None:
            self._started.set()
            try:
                self._sdk.init()
            except BaseException as exc:
                if not self._closing.is_set():
                    self._error = exc

        self._thread = Thread(
            target=initialize,
            name="pico-gmr-xrt-init",
            daemon=True,
        )
        self._thread.start()
        self._started.wait(timeout=1.0)

    def close(self) -> None:
        thread = self._thread
        if thread is None:
            return
        self._thread = None
        self._closing.set()
        try:
            self._sdk.close()
        finally:
            thread.join(timeout=3.0)
        if thread.is_alive():
            raise RuntimeError("xrt.init() did not stop after xrt.close()")


__all__ = ["XrtBackgroundSession", "XrtSdk"]
