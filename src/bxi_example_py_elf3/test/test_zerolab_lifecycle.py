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
from zerolab.source_node import (
    ZeroLabSourceCore,
    ZeroLabSourceNode,
    validate_source_params,
)
from zerolab.udp_receiver import ReceivedDatagram


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
    def __init__(self, payload):
        self.payload = payload
        self.pending = []

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
        pending, self.pending = self.pending, []
        return pending


class CapturingPublisher:
    def __init__(self):
        self.frame_indices = []
        self.frame_windows = []

    def send(self, fields):
        self.frame_indices.append(int(fields["frame_index"][-1]))
        self.frame_windows.append(np.asarray(fields["frame_index"]).copy())


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


class RejectMalformedCore:
    def accept(self, _packet):
        raise AssertionError("malformed packet reached core.accept")

    def check_stale(self, _now_ns):
        return False

    def consume_stale_event(self):
        return False


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
    node._stream_state = None
    node.get_logger = lambda: logger

    node._tick()

    assert node._invalid_packets == 1
    assert len(logger.warnings) == 1


def test_executor_gap_logs_stale_once_then_ready_once_after_refill(
    monkeypatch,
):
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
    node._core = ZeroLabSourceCore(node._converter)
    node._publisher = publisher
    node._writer = None
    node._recording_enabled = False
    node._invalid_packets = 0
    node._dropped_publications = 0
    node._stream_state = None
    node.get_logger = lambda: logger

    for frame_index in range(10):
        clock_ns[0] = frame_index * 20_000_000
        receiver.queue(frame_index, clock_ns[0])
        node._tick()

    assert publisher.frame_indices == [9]
    assert logger.infos.count("ZeroLab stream ready; frame=9") == 1

    clock_ns[0] += 500_000_001
    receiver.queue(10, clock_ns[0])
    node._tick()
    node._tick()

    stale_message = "ZeroLab input stale; live pose publication stopped"
    assert logger.warnings.count(stale_message) == 1

    gap_timestamp_ns = clock_ns[0]
    for frame_index in range(11, 20):
        clock_ns[0] = gap_timestamp_ns + (frame_index - 10) * 20_000_000
        receiver.queue(frame_index, clock_ns[0])
        node._tick()
    node._tick()

    assert publisher.frame_indices == [9, 19]
    assert logger.warnings.count(stale_message) == 1
    assert logger.infos.count("ZeroLab stream ready; frame=19") == 1
    assert logger.infos.count("ZeroLab waiting for 10-frame stream window") == 2


def test_executor_gap_backlog_logs_stale_before_same_tick_ready(monkeypatch):
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
    node._core = ZeroLabSourceCore(node._converter)
    node._publisher = publisher
    node._writer = None
    node._recording_enabled = False
    node._invalid_packets = 0
    node._dropped_publications = 0
    node._stream_state = None
    node.get_logger = lambda: logger

    for frame_index in range(10):
        clock_ns[0] = frame_index * 20_000_000
        receiver.queue(frame_index, clock_ns[0])
        node._tick()

    pre_gap_timestamp_ns = clock_ns[0] + 20_000_000
    receiver.queue(10, pre_gap_timestamp_ns)
    gap_timestamp_ns = pre_gap_timestamp_ns + 500_000_001
    for frame_index in range(11, 21):
        timestamp_ns = gap_timestamp_ns + (frame_index - 11) * 20_000_000
        receiver.queue(frame_index, timestamp_ns)
        clock_ns[0] = timestamp_ns
    node._tick()
    node._tick()

    stale_message = "ZeroLab input stale; live pose publication stopped"
    assert publisher.frame_indices == [9, 20]
    np.testing.assert_array_equal(
        publisher.frame_windows[-1], np.arange(11, 21)
    )
    assert logger.warnings.count(stale_message) == 1
    ready_message = "ZeroLab stream ready; frame=20"
    assert logger.infos.count(ready_message) == 1
    stale_event = logger.events.index(("warning", stale_message))
    ready_event = logger.events.index(("info", ready_message))
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
