from collections import deque
from dataclasses import FrozenInstanceError
import socket

import pytest

from zerolab.udp_receiver import ZeroLabUdpReceiver


class FakeSocket:
    def __init__(self, datagrams, events=None):
        self.datagrams = deque(datagrams)
        self.events = events if events is not None else []
        self.bound = None
        self.blocking = None
        self.closed = False
        self.receive_sizes = []

    def setblocking(self, value):
        self.blocking = value

    def bind(self, address):
        self.bound = address

    def getsockname(self):
        return self.bound

    def recvfrom(self, size):
        self.receive_sizes.append(size)
        self.events.append("recvfrom")
        if not self.datagrams:
            raise BlockingIOError
        return self.datagrams.popleft()

    def close(self):
        self.closed = True


def test_receiver_filters_sender_and_size_before_assigning_frames():
    events = []
    fake = FakeSocket([
        (bytes(992), ("10.0.0.9", 4000)),
        (bytes(991), ("10.0.0.2", 4000)),
        (bytes([7]) * 992, ("10.0.0.2", 5000)),
    ], events)
    ticks = iter([111, 222, 123456])

    def clock_ns():
        events.append("clock")
        return next(ticks)

    receiver = ZeroLabUdpReceiver(
        "0.0.0.0", 18000,
        allowed_sender_host="10.0.0.2",
        clock_ns=clock_ns,
        sock=fake,
    )

    datagrams = receiver.drain()

    assert fake.blocking is False
    assert fake.bound == ("0.0.0.0", 18000)
    assert fake.receive_sizes == [65535, 65535, 65535, 65535]
    assert events == [
        "recvfrom", "clock", "recvfrom", "clock", "recvfrom", "clock",
        "recvfrom",
    ]
    assert len(datagrams) == 1
    assert datagrams[0].payload == bytes([7]) * 992
    assert datagrams[0].receive_timestamp_ns == 123456
    assert datagrams[0].local_frame_index == 0
    assert datagrams[0].sender_address == ("10.0.0.2", 5000)
    assert receiver.stats.received == 3
    assert receiver.stats.unexpected_sender == 1
    assert receiver.stats.invalid_size == 1
    assert receiver.stats.accepted == 1
    with pytest.raises(FrozenInstanceError):
        receiver.stats.accepted = 20


def test_receiver_rejects_oversized_datagram_without_truncating_it():
    fake = FakeSocket([(bytes(993), ("10.0.0.2", 4000))])
    receiver = ZeroLabUdpReceiver(sock=fake)

    assert receiver.poll() is None
    assert fake.receive_sizes == [65535, 65535]
    assert receiver.stats.invalid_size == 1
    assert receiver.stats.accepted == 0


def test_drain_obeys_limit_and_rejects_invalid_limit():
    fake = FakeSocket([
        (bytes([1]) * 992, ("10.0.0.2", 4000)),
        (bytes([2]) * 992, ("10.0.0.2", 4000)),
    ])
    receiver = ZeroLabUdpReceiver(sock=fake)

    datagrams = receiver.drain(limit=1)

    assert [datagram.local_frame_index for datagram in datagrams] == [0]
    assert receiver.poll().local_frame_index == 1
    with pytest.raises(ValueError, match="limit"):
        receiver.drain(limit=0)


@pytest.mark.parametrize("port", [-1, 65536])
def test_receiver_rejects_ports_outside_udp_range(port):
    with pytest.raises(ValueError, match="port"):
        ZeroLabUdpReceiver(port=port, sock=FakeSocket([]))


def test_close_is_idempotent_and_releases_udp_port():
    first = ZeroLabUdpReceiver("127.0.0.1", 0)
    port = first.local_address[1]
    first.close()
    first.close()
    second = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        second.bind(("127.0.0.1", port))
    finally:
        second.close()
