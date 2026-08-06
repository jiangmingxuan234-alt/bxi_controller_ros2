"""Non-blocking receiver and immutable snapshots for the control state."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import socket
from threading import Event, Lock, Thread
import time

import numpy as np

from .protocol import (
    LiveReferenceFrame,
    PACKET_SIZE,
    REFERENCE_WINDOW_SIZE,
    decode_reference_frame,
)


WINDOW_SIZE = REFERENCE_WINDOW_SIZE


@dataclass(frozen=True, slots=True)
class ReferenceWindow:
    session_id: int
    latest_sequence: int
    received_monotonic: float
    joint_pos: np.ndarray
    head_joint_pos: np.ndarray
    anchor_quat_wxyz: np.ndarray
    anchor_lin_vel_w: np.ndarray
    anchor_ang_vel_w: np.ndarray


class ReferenceWindowBuffer:
    """Thread-safe unique-frame accumulator, separated for deterministic tests."""

    def __init__(self) -> None:
        self._frames: deque[tuple[LiveReferenceFrame, float]] = deque(maxlen=WINDOW_SIZE)
        self._lock = Lock()

    def accept(
        self,
        frame: LiveReferenceFrame,
        received_monotonic: float | None = None,
    ) -> bool:
        received = (
            time.monotonic()
            if received_monotonic is None
            else float(received_monotonic)
        )
        with self._lock:
            if self._frames:
                previous = self._frames[-1][0]
                if frame.session_id != previous.session_id:
                    self._frames.clear()
                elif frame.sequence <= previous.sequence:
                    return False
            self._frames.append((frame, received))
        return True

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()

    def snapshot_window(
        self,
        *,
        max_age_s: float | None = None,
        now: float | None = None,
    ) -> ReferenceWindow | None:
        with self._lock:
            if len(self._frames) < WINDOW_SIZE:
                return None
            if max_age_s is not None:
                current = time.monotonic() if now is None else float(now)
                if current - self._frames[-1][1] > float(max_age_s):
                    return None
            items = tuple(self._frames)
        frames = tuple(item[0] for item in items)
        return ReferenceWindow(
            session_id=frames[-1].session_id,
            latest_sequence=frames[-1].sequence,
            received_monotonic=items[-1][1],
            joint_pos=np.stack([frame.joint_pos for frame in frames]),
            head_joint_pos=np.stack([frame.head_joint_pos for frame in frames]),
            anchor_quat_wxyz=np.stack([frame.anchor_quat_wxyz for frame in frames]),
            anchor_lin_vel_w=np.stack([frame.anchor_lin_vel_w for frame in frames]),
            anchor_ang_vel_w=np.stack([frame.anchor_ang_vel_w for frame in frames]),
        )

    def latest_age(self, now: float | None = None) -> float | None:
        with self._lock:
            if not self._frames:
                return None
            received = self._frames[-1][1]
        return (time.monotonic() if now is None else float(now)) - received


class LiveReferenceReceiver:
    """Receive and validate unique reference frames on a background thread."""

    def __init__(self, host: str, port: int) -> None:
        if not host:
            raise ValueError("reference receiver host must not be empty")
        if not 0 < int(port) < 65536:
            raise ValueError("reference receiver port must be in [1,65535]")
        self.host = host
        self.port = int(port)
        self._buffer = ReferenceWindowBuffer()
        self._stop = Event()
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((host, self.port))
        self._socket.settimeout(0.1)
        self._thread = Thread(target=self._receive_loop, name="pico-gmr-reference", daemon=False)
        self._thread.start()

    def _receive_loop(self) -> None:
        while not self._stop.is_set():
            try:
                packet, _ = self._socket.recvfrom(PACKET_SIZE + 1)
            except socket.timeout:
                continue
            except OSError:
                if self._stop.is_set():
                    return
                continue
            try:
                frame = decode_reference_frame(packet)
            except (TypeError, ValueError):
                continue
            self._buffer.accept(frame)

    def clear(self) -> None:
        self._buffer.clear()

    def snapshot_window(
        self,
        *,
        max_age_s: float | None = None,
        now: float | None = None,
    ) -> ReferenceWindow | None:
        return self._buffer.snapshot_window(max_age_s=max_age_s, now=now)

    def latest_age(self, now: float | None = None) -> float | None:
        return self._buffer.latest_age(now)

    def close(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        self._socket.close()
        self._thread.join(timeout=1.0)
        if self._thread.is_alive():
            raise RuntimeError("PICO GMR reference receiver did not stop")


__all__ = [
    "LiveReferenceReceiver",
    "ReferenceWindow",
    "ReferenceWindowBuffer",
    "WINDOW_SIZE",
]
