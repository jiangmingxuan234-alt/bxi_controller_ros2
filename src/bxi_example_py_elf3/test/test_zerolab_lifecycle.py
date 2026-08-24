from collections import deque
from copy import deepcopy
from pathlib import Path
import socket
import time

import numpy as np
import pytest
import rclpy
import zmq

from bxi_example_py_elf3.framework.mod_api import NodeBuildContext
from bxi_example_py_elf3.framework.runtime.logging import (
    LoggingConfig,
    ScopedLoggers,
)
from bxi_example_py_elf3.framework.runtime.mod_nodes import (
    ModNodeManager,
    ModNodeSpec,
)
from pico.pose_to_smpl_ref_bridge import SmplRefBridgeNode
from zerolab.converter import ZeroLabMotionConverter
from zerolab.resampler import ZeroLabPoseResampler
from zerolab.source_node import (
    ZeroLabSourceCore,
    ZeroLabSourceNode,
    validate_source_params,
)
from zerolab.udp_receiver import DrainBatch, ReceivedDatagram, ZeroLabUdpReceiver


MOD_ROOT = Path(__file__).resolve().parents[1] / "mods" / "com.bxi.sonic"


def free_port(sock_type):
    sock = socket.socket(socket.AF_INET, sock_type)
    try:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
    finally:
        sock.close()


@pytest.fixture
def rclpy_runtime():
    started_here = not rclpy.ok()
    if started_here:
        rclpy.init(args=[])
    yield
    if started_here and rclpy.ok():
        rclpy.shutdown()


def source_context(udp_port, pose_port, node_name):
    root = Path(__file__).resolve().parents[1] / "mods" / "com.bxi.sonic"
    return NodeBuildContext(
        mod_id="com.bxi.sonic",
        node_id=f"com.bxi.sonic/{node_name}",
        node_name=node_name,
        mod_root=root,
        params={
            "udp_bind_host": "127.0.0.1",
            "udp_port": udp_port,
            "allowed_sender": "",
            "pose_host": "127.0.0.1",
            "pose_port": pose_port,
            "pose_topic": "pose",
            "rate_hz": 50.0,
            "window_frames": 10,
            "stale_seconds": 0.5,
            "jitter_buffer_seconds": 0.08,
            "short_recovery_blend_seconds": 0.2,
            "recovery_real_frames": 10,
            "record_path": "",
        },
    )


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"mystery": 1}, "unknown"),
        ({"udp_port": 0}, "udp_port"),
        ({"udp_port": True}, "udp_port"),
        ({"pose_port": 65536}, "pose_port"),
        ({"pose_port": False}, "pose_port"),
        ({"rate_hz": 49.0}, "rate_hz"),
        ({"rate_hz": True}, "rate_hz"),
        ({"window_frames": 9}, "window_frames"),
        ({"window_frames": True}, "window_frames"),
        ({"stale_seconds": 0.0}, "stale_seconds"),
        ({"stale_seconds": False}, "stale_seconds"),
        ({"udp_bind_host": ""}, "udp_bind_host"),
        ({"pose_host": "0.0.0.0"}, "pose_host"),
        ({"pose_topic": ""}, "pose_topic"),
        ({"allowed_sender": 7}, "allowed_sender"),
        ({"record_path": Path("capture")}, "record_path"),
    ],
)
def test_source_rejects_unknown_or_unsafe_params(params, message):
    with pytest.raises(ValueError, match=message):
        validate_source_params(params)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("jitter_buffer_seconds", 0.0),
        ("jitter_buffer_seconds", float("nan")),
        ("short_recovery_blend_seconds", -0.1),
        ("short_recovery_blend_seconds", True),
        ("recovery_real_frames", 0),
        ("recovery_real_frames", True),
    ],
)
def test_source_rejects_invalid_resampling_params(name, value):
    with pytest.raises(ValueError, match=name):
        validate_source_params({name: value})


def test_source_accepts_valid_resampling_params():
    params = validate_source_params(
        {
            "jitter_buffer_seconds": 0.08,
            "short_recovery_blend_seconds": 0.2,
            "recovery_real_frames": 10,
        }
    )
    assert params["jitter_buffer_seconds"] == 0.08
    assert params["short_recovery_blend_seconds"] == 0.2
    assert params["recovery_real_frames"] == 10


def test_source_rejects_recording_path_inside_mod_root():
    root = Path(__file__).resolve().parents[1] / "mods" / "com.bxi.sonic"
    with pytest.raises(ValueError, match="outside"):
        validate_source_params(
            {"record_path": str(root / "capture")}, mod_root=root
        )


def test_repeated_source_enter_exit_releases_udp_and_pose_ports(rclpy_runtime):
    udp_port = free_port(socket.SOCK_DGRAM)
    pose_port = free_port(socket.SOCK_STREAM)
    first = ZeroLabSourceNode(source_context(udp_port, pose_port, "zerolab_a"))
    first.destroy_node()
    second = ZeroLabSourceNode(
        source_context(udp_port, pose_port, "zerolab_b")
    )
    second.destroy_node()


def test_pose_bind_failure_immediately_releases_udp_port(rclpy_runtime):
    udp_port = free_port(socket.SOCK_DGRAM)
    pose_port = free_port(socket.SOCK_STREAM)
    blocker_context = zmq.Context()
    blocker = blocker_context.socket(zmq.PUB)
    blocker.setsockopt(zmq.LINGER, 0)
    blocker.bind(f"tcp://127.0.0.1:{pose_port}")
    try:
        with pytest.raises(zmq.ZMQError):
            ZeroLabSourceNode(
                source_context(udp_port, pose_port, "zerolab_fail")
            )
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.bind(("127.0.0.1", udp_port))
        finally:
            probe.close()
    finally:
        blocker.close(linger=0)
        blocker_context.term()


def test_existing_bridge_releases_its_output_port(rclpy_runtime):
    input_port = free_port(socket.SOCK_STREAM)
    output_port = free_port(socket.SOCK_STREAM)
    params = {
        "pico_host": "127.0.0.1",
        "pico_port": input_port,
        "pico_topic": "pose",
        "out_host": "127.0.0.1",
        "out_port": output_port,
        "out_topic": "smpl_ref",
        "rate_hz": 50.0,
        "stale_warning_seconds": 0.5,
    }
    root = Path(__file__).resolve().parents[1] / "mods" / "com.bxi.sonic"
    first = SmplRefBridgeNode(
        NodeBuildContext("com.bxi.sonic", "bridge_a", "bridge_a", root, params)
    )
    first.destroy_node()
    second = SmplRefBridgeNode(
        NodeBuildContext("com.bxi.sonic", "bridge_b", "bridge_b", root, params)
    )
    second.destroy_node()


class ControlledReceiver:
    def __init__(self, payload, *, drain_limit=256):
        self.payload = payload
        self.pending = []
        self.drain_limit = drain_limit
        self.drain_calls = 0

    def queue(self, frame_index, timestamp_ns):
        self.pending.append(
            ReceivedDatagram(
                payload=self.payload,
                receive_timestamp_ns=timestamp_ns,
                local_frame_index=frame_index,
                sender_address=("127.0.0.1", 50000),
            )
        )

    def drain(self):
        self.drain_calls += 1
        exhausted = len(self.pending) < self.drain_limit
        pending = self.pending[: self.drain_limit]
        del self.pending[: self.drain_limit]
        return DrainBatch(tuple(pending), exhausted)


class QueuedUdpSocket:
    def __init__(self, datagrams):
        self.datagrams = deque(datagrams)
        self.bound = None

    def setblocking(self, _value):
        pass

    def bind(self, address):
        self.bound = address

    def getsockname(self):
        return self.bound

    def recvfrom(self, _size):
        if not self.datagrams:
            raise BlockingIOError
        return self.datagrams.popleft()

    def close(self):
        pass


class CapturingPublisher:
    def __init__(self):
        self.messages = []

    def send(self, fields):
        self.messages.append(deepcopy(fields))


class CapturingLogger:
    def __init__(self):
        self.infos = []
        self.warnings = []
        self.events = []

    def info(self, message):
        self.infos.append(message)
        self.events.append(("info", message))

    def warning(self, message):
        self.warnings.append(message)
        self.events.append(("warning", message))


def identity_payload():
    root = np.zeros(3, dtype="<f4")
    quaternions = np.zeros((47, 4), dtype="<f4")
    quaternions[:, 3] = 1.0
    hands = np.zeros(6, dtype="<u2")
    positions = np.zeros((17, 3), dtype="<f4")
    return b"".join(
        (
            root.tobytes(),
            quaternions.tobytes(),
            hands.tobytes(),
            hands.tobytes(),
            positions.tobytes(),
        )
    )


def make_test_core():
    generations = iter((101, 202, 303))
    return ZeroLabSourceCore(
        ZeroLabMotionConverter(),
        resampler=ZeroLabPoseResampler(
            jitter_buffer_seconds=0.08,
            short_recovery_blend_seconds=0.2,
            output_rate_hz=50.0,
        ),
        window_frames=10,
        stale_seconds=0.5,
        recovery_real_frames=10,
        generation_factory=lambda _previous=None: next(generations),
    )


def make_controlled_node(monkeypatch):
    clock_ns = [0]
    monkeypatch.setattr(
        "zerolab.source_node.time.monotonic_ns", lambda: clock_ns[0]
    )
    receiver = ControlledReceiver(identity_payload())
    publisher = CapturingPublisher()
    logger = CapturingLogger()
    node = ZeroLabSourceNode.__new__(ZeroLabSourceNode)
    node._closed = False
    node._receiver = receiver
    node._converter = ZeroLabMotionConverter()
    node._core = make_test_core()
    node._publisher = publisher
    node._writer = None
    node._recording_enabled = False
    node._recording_error_reported = False
    node._invalid_packets = 0
    node._dropped_publications = 0
    node._stream_state = None
    node._last_stale_log_generation = None
    node._last_stats_log_ns = 0
    node.get_logger = lambda: logger
    return node, receiver, publisher, clock_ns


def ready_controlled_node(monkeypatch):
    node, receiver, publisher, clock_ns = make_controlled_node(monkeypatch)
    for index in range(10):
        clock_ns[0] = 80_000_000 + index * 20_000_000
        receiver.queue(index, index * 20_000_000)
        node._tick()
    assert any(int(msg["real_stream_ready"][0]) for msg in publisher.messages)
    return node, receiver, publisher, clock_ns


def create_stale_then_ten_real_recovery(node, receiver, clock_ns):
    clock_ns[0] += 500_000_001
    node._tick()
    recovery_start = clock_ns[0] + 20_000_000
    for offset in range(10):
        timestamp_ns = recovery_start + offset * 20_000_000
        receiver.queue(100 + offset, timestamp_ns)
        clock_ns[0] = timestamp_ns + 80_000_000
        node._tick()


def test_node_drains_burst_but_publishes_at_most_once_per_tick(monkeypatch):
    node, receiver, publisher, clock = ready_controlled_node(monkeypatch)
    initial_publications = len(publisher.messages)
    for index in range(10, 40):
        receiver.queue(index, index * 20_000_000)
    clock[0] = 860_000_000
    node._tick()
    assert receiver.pending == []
    assert node._core.stats.real_valid_packets == 40
    assert len(publisher.messages) == initial_publications + 1


def test_node_drains_more_than_receiver_batch_limit_before_one_sample(
    monkeypatch,
):
    node, receiver, publisher, clock = ready_controlled_node(monkeypatch)
    initial_drain_calls = receiver.drain_calls
    initial_publications = len(publisher.messages)
    for index in range(10, 310):
        receiver.queue(index, index * 20_000_000)
    clock[0] = 6_260_000_000

    node._tick()

    assert receiver.drain_calls == initial_drain_calls + 2
    assert receiver.pending == []
    assert node._core.stats.real_valid_packets == 310
    assert len(publisher.messages) == initial_publications + 1


def test_node_continues_after_full_rejected_batch_until_socket_exhausted(
    monkeypatch,
):
    node, _receiver, publisher, clock = ready_controlled_node(monkeypatch)
    valid_payload = identity_payload()
    wrong_sender = (valid_payload, ("127.0.0.2", 50000))
    wrong_size = (bytes(991), ("127.0.0.1", 50000))
    valid = (valid_payload, ("127.0.0.1", 50000))
    fake_socket = QueuedUdpSocket(
        [wrong_sender] * 128 + [wrong_size] * 128 + [valid] * 30
    )
    receive_timestamps = iter(
        [0] * 256 + [index * 20_000_000 for index in range(10, 40)]
    )
    receiver = ZeroLabUdpReceiver(
        allowed_sender_host="127.0.0.1",
        clock_ns=lambda: next(receive_timestamps),
        sock=fake_socket,
    )
    receiver._next_frame_index = 10
    node._receiver = receiver
    initial_publications = len(publisher.messages)
    clock[0] = 860_000_000

    node._tick()

    assert fake_socket.datagrams == deque()
    assert receiver.stats.received == 286
    assert receiver.stats.unexpected_sender == 128
    assert receiver.stats.invalid_size == 128
    assert receiver.stats.accepted == 30
    assert node._core.stats.real_valid_packets == 40
    assert len(publisher.messages) == initial_publications + 1


def test_node_keeps_50_hz_stale_held_publication_with_stale_metadata(
    monkeypatch,
):
    node, receiver, publisher, clock = ready_controlled_node(monkeypatch)
    initial_publications = len(publisher.messages)
    clock[0] += 500_000_001
    node._tick()
    for _ in range(12):
        clock[0] += 20_000_000
        node._tick()
    stale_messages = publisher.messages[initial_publications:]
    assert len(stale_messages) == 13
    assert [
        int(message["frame_index"][-1]) for message in stale_messages
    ] == list(range(10, 23))
    assert all(
        int(msg["real_stream_ready"][0]) == 0
        and int(msg["source_stale"][0]) == 1
        and int(msg["playout_kind"][0]) == 2
        for msg in stale_messages
    )


def test_stale_log_precedes_recovery_ready_and_is_rate_limited(monkeypatch):
    node, receiver, publisher, clock = ready_controlled_node(monkeypatch)
    create_stale_then_ten_real_recovery(node, receiver, clock)
    stale = [
        event for event in node.get_logger().events if "input stale" in event[1]
    ]
    ready = [
        event for event in node.get_logger().events if "stream ready" in event[1]
    ]
    assert len(stale) == 1
    assert node.get_logger().events.index(
        stale[0]
    ) < node.get_logger().events.index(ready[-1])


def test_stale_transition_logs_once_for_each_source_generation(monkeypatch):
    node, receiver, _publisher, clock = ready_controlled_node(monkeypatch)
    stats_before = sum(
        "source stats" in message for message in node.get_logger().infos
    )
    first_gap_packet_ns = (
        node._core.latest_real_receive_timestamp_ns + 500_000_001
    )
    receiver.queue(10, first_gap_packet_ns)
    clock[0] = first_gap_packet_ns + 80_000_000
    node._tick()
    assert node._core.source_generation == 202

    clock[0] = first_gap_packet_ns + 500_000_001
    node._tick()

    assert node._core.source_generation == 303
    stale = [
        event for event in node.get_logger().events if "input stale" in event[1]
    ]
    assert len(stale) == 2
    stats_after = sum(
        "source stats" in message for message in node.get_logger().infos
    )
    assert stats_after == stats_before + 2


def test_source_statistics_are_bounded_to_five_seconds_and_include_counters(
    monkeypatch,
):
    node, receiver, _publisher, clock = ready_controlled_node(monkeypatch)
    node.get_logger().infos.clear()
    node.get_logger().events.clear()
    stats_start_ns = clock[0]
    node._last_stats_log_ns = stats_start_ns
    latest_real_ns = node._core.latest_real_receive_timestamp_ns

    for offset in range(1, 13):
        timestamp_ns = latest_real_ns + offset * 400_000_000
        receiver.queue(9 + offset, timestamp_ns)
        clock[0] = timestamp_ns + 80_000_000
        node._tick()
    assert not any("source stats" in message for message in node.get_logger().infos)

    clock[0] = stats_start_ns + 5_000_000_000
    node._tick()
    stats = [
        message
        for message in node.get_logger().infos
        if "source stats" in message
    ]
    assert len(stats) == 1
    for counter in (
        "real_valid_packets=22",
        "maximum_real_arrival_gap_ms=400.000",
        "stale_events=0",
        "interpolated_output_frames=",
        "held_output_frames=",
        "dropped_backlog_frames=",
        "invalid_packets=0",
        "dropped_publications=0",
    ):
        assert counter in stats[0]

    clock[0] += 20_000_000
    node._tick()
    assert sum(
        "source stats" in message for message in node.get_logger().infos
    ) == 1


class RejectMalformedCore:
    def accept(self, _packet):
        raise AssertionError("malformed packet reached core.accept")

    def check_stale(self, _now_ns):
        return False

    def consume_stale_event(self):
        return False

    def sample(self, _now_ns):
        return None


def test_malformed_packet_never_reaches_core_accept(monkeypatch):
    monkeypatch.setattr("zerolab.source_node.time.monotonic_ns", lambda: 0)
    receiver = ControlledReceiver(b"malformed")
    receiver.queue(0, 0)
    logger = CapturingLogger()
    node = ZeroLabSourceNode.__new__(ZeroLabSourceNode)
    node._closed = False
    node._receiver = receiver
    node._core = RejectMalformedCore()
    node._publisher = CapturingPublisher()
    node._writer = None
    node._recording_enabled = False
    node._invalid_packets = 0
    node._dropped_publications = 0
    node._stream_state = "collecting"
    node._last_stats_log_ns = 0
    node.get_logger = lambda: logger

    node._tick()

    assert node._invalid_packets == 1
    assert len(logger.warnings) == 1


def test_executor_gap_logs_stale_once_then_ready_once_after_refill(
    monkeypatch,
):
    node, receiver, publisher, clock_ns = ready_controlled_node(monkeypatch)

    create_stale_then_ten_real_recovery(node, receiver, clock_ns)

    stale_message = "ZeroLab input stale; holding playout with source_stale=1"
    assert node.get_logger().warnings.count(stale_message) == 1
    ready = [
        message
        for message in node.get_logger().infos
        if "ZeroLab stream ready" in message
    ]
    assert len(ready) == 2
    assert "generation=202 real_frames=10" in ready[-1]
    assert int(publisher.messages[-1]["source_generation"][0]) == 202
    assert int(publisher.messages[-1]["real_stream_ready"][0]) == 1


def test_executor_gap_backlog_logs_stale_before_same_tick_ready(monkeypatch):
    node, receiver, publisher, clock_ns = ready_controlled_node(monkeypatch)
    initial_publications = len(publisher.messages)

    pre_gap_timestamp_ns = clock_ns[0] + 20_000_000
    receiver.queue(10, pre_gap_timestamp_ns)
    gap_timestamp_ns = pre_gap_timestamp_ns + 500_000_001
    for frame_index in range(11, 21):
        timestamp_ns = gap_timestamp_ns + (frame_index - 11) * 20_000_000
        receiver.queue(frame_index, timestamp_ns)
        clock_ns[0] = timestamp_ns
    clock_ns[0] += 80_000_000
    node._tick()
    assert len(publisher.messages) <= initial_publications + 1
    for _ in range(9):
        clock_ns[0] += 20_000_000
        node._tick()

    stale_message = "ZeroLab input stale; holding playout with source_stale=1"
    assert node.get_logger().warnings.count(stale_message) == 1
    assert int(publisher.messages[-1]["latest_real_frame_index"][0]) == 20
    assert int(publisher.messages[-1]["source_generation"][0]) == 202
    assert int(publisher.messages[-1]["real_stream_ready"][0]) == 1
    stale_event = node.get_logger().events.index(("warning", stale_message))
    ready_event = next(
        index
        for index, event in enumerate(node.get_logger().events)
        if index > stale_event and "stream ready" in event[1]
    )
    assert stale_event < ready_event
    assert node._stream_state == "ready"


class NullLogger:
    def get_child(self, _name):
        return self

    def set_level(self, _level):
        pass

    def debug(self, _message):
        pass

    def info(self, _message):
        pass

    def warning(self, _message):
        pass

    def error(self, _message):
        pass

    def fatal(self, _message):
        pass


class FakeExecutor:
    def add_node(self, _node):
        return True

    def remove_node(self, _node):
        return True


class BindingNode:
    def __init__(self, endpoint):
        self.destroy_count = 0
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(endpoint)

    def destroy_node(self):
        self.destroy_count += 1
        self.socket.close(linger=0)
        self.context.term()


def make_binding_spec(local_name, state_name, factory):
    return ModNodeSpec(
        id=f"com.bxi.sonic/{local_name}",
        mod_id="com.bxi.sonic",
        local_name=local_name,
        node_name=local_name,
        mod_root=MOD_ROOT,
        manifest_path=MOD_ROOT / "mod.yaml",
        entrypoint="test:factory",
        execution="in_process",
        lifecycle="state",
        states=(state_name,),
        params={},
        manifest={},
        restart_max_attempts=0,
        restart_delay=0.0,
        factory=factory,
    )


def wait_until_real_zmq_bind_succeeds(manager, endpoint, timeout_s):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        manager.poll()
        context = zmq.Context()
        probe = context.socket(zmq.PUB)
        probe.setsockopt(zmq.LINGER, 0)
        try:
            probe.bind(endpoint)
            return
        except zmq.ZMQError as exc:
            if exc.errno != zmq.EADDRINUSE:
                raise
        finally:
            probe.close(linger=0)
            context.term()
        time.sleep(0.02)
    raise AssertionError(
        f"endpoint was not released within {timeout_s}s: {endpoint}"
    )


def test_manager_poll_releases_shared_bridge_port_after_each_state_stops():
    endpoint = f"tcp://127.0.0.1:{free_port(socket.SOCK_STREAM)}"
    instances = []

    def pico_factory(_context):
        instance = BindingNode(endpoint)
        instances.append(instance)
        return instance

    def zero_factory(_context):
        instance = BindingNode(endpoint)
        instances.append(instance)
        return instance

    root_logger = NullLogger()
    loggers = ScopedLoggers(
        root_logger,
        LoggingConfig(),
        logger_factory=lambda _name: root_logger,
    )
    manager = ModNodeManager(
        (
            make_binding_spec(
                "pico_test_bridge", "com.bxi.sonic/sonic_teleop", pico_factory
            ),
            make_binding_spec(
                "zero_test_bridge", "com.bxi.sonic/sonic_zerolab", zero_factory
            ),
        ),
        loggers=loggers,
    )
    fake_executor = FakeExecutor()
    try:
        manager.start()
        manager.attach_executor(fake_executor)
        manager.activate_initial_state("com.bxi.basic_actions/normal")

        manager.prepare_state("com.bxi.sonic/sonic_teleop")
        manager.finish_transition(
            "com.bxi.basic_actions/normal", "com.bxi.sonic/sonic_teleop"
        )
        manager.prepare_state("com.bxi.basic_actions/normal")
        manager.finish_transition(
            "com.bxi.sonic/sonic_teleop", "com.bxi.basic_actions/normal"
        )
        wait_until_real_zmq_bind_succeeds(manager, endpoint, timeout_s=2.0)

        manager.prepare_state("com.bxi.sonic/sonic_zerolab")
        manager.cancel_prepared_state("com.bxi.sonic/sonic_zerolab")
        wait_until_real_zmq_bind_succeeds(manager, endpoint, timeout_s=2.0)
    finally:
        manager.close()

    assert len(instances) == 2
    assert all(instance.destroy_count == 1 for instance in instances)
