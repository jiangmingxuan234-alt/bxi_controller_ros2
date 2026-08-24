from dataclasses import dataclass, replace
import socket
import time
from typing import Callable

from .protocol import PACKET_SIZE


_SOCKET_EMPTY = object()
_REJECTED = object()


@dataclass(frozen=True)
class ReceivedDatagram:
    payload: bytes
    receive_timestamp_ns: int
    local_frame_index: int
    sender_address: tuple[str, int]


@dataclass(frozen=True)
class ReceiverStats:
    received: int = 0
    accepted: int = 0
    invalid_size: int = 0
    unexpected_sender: int = 0


class ZeroLabUdpReceiver:
    def __init__(
        self,
        bind_host: str = "0.0.0.0",
        port: int = 18000,
        allowed_sender_host: str | None = None,
        *,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        sock: socket.socket | None = None,
    ) -> None:
        if not 0 <= port <= 65535:
            raise ValueError("port must be in 0..65535")
        self._socket = (
            sock
            if sock is not None
            else socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        )
        self._socket.setblocking(False)
        self._socket.bind((bind_host, port))
        self.local_address = self._socket.getsockname()
        self._allowed_sender_host = allowed_sender_host
        self._clock_ns = clock_ns
        self._stats = ReceiverStats()
        self._next_frame_index = 0
        self._closed = False

    @property
    def stats(self) -> ReceiverStats:
        return replace(self._stats)

    def _poll_once(self):
        try:
            payload, sender_address = self._socket.recvfrom(65535)
        except BlockingIOError:
            return _SOCKET_EMPTY
        receive_timestamp_ns = self._clock_ns()
        self._stats = replace(
            self._stats, received=self._stats.received + 1
        )
        if (
            self._allowed_sender_host is not None
            and sender_address[0] != self._allowed_sender_host
        ):
            self._stats = replace(
                self._stats,
                unexpected_sender=self._stats.unexpected_sender + 1,
            )
            return _REJECTED
        if len(payload) != PACKET_SIZE:
            self._stats = replace(
                self._stats, invalid_size=self._stats.invalid_size + 1
            )
            return _REJECTED
        datagram = ReceivedDatagram(
            payload=bytes(payload),
            receive_timestamp_ns=receive_timestamp_ns,
            local_frame_index=self._next_frame_index,
            sender_address=(
                str(sender_address[0]), int(sender_address[1])
            ),
        )
        self._next_frame_index += 1
        self._stats = replace(
            self._stats, accepted=self._stats.accepted + 1
        )
        return datagram

    def poll(self) -> ReceivedDatagram | None:
        while True:
            result = self._poll_once()
            if result is _SOCKET_EMPTY:
                return None
            if result is not _REJECTED:
                return result

    def drain(self, limit: int = 256) -> list[ReceivedDatagram]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        datagrams = []
        for _ in range(limit):
            result = self._poll_once()
            if result is _SOCKET_EMPTY:
                break
            if result is not _REJECTED:
                datagrams.append(result)
        return datagrams

    def close(self) -> None:
        if self._closed:
            return
        self._socket.close()
        self._closed = True
