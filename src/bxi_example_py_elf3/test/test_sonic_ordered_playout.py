from __future__ import annotations

import importlib.util
import queue
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from zerolab.resampler import PlayoutKind


MOD_ROOT = (
    Path(__file__).resolve().parents[1] / "mods" / "com.bxi.sonic"
)
PACKAGE = "_sonic_ordered_playout_test_mod"


def _load_module(name: str, relative_path: str):
    if PACKAGE not in sys.modules:
        package = ModuleType(PACKAGE)
        package.__path__ = [str(MOD_ROOT)]
        package.__package__ = PACKAGE
        sys.modules[PACKAGE] = package
    full_name = f"{PACKAGE}.{name}"
    loaded = sys.modules.get(full_name)
    if loaded is not None:
        return loaded
    path = MOD_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(full_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


merger_module = _load_module(
    "pico.streamed_smpl_ref",
    "pico/streamed_smpl_ref.py",
)
policy_module = _load_module("policy", "policy.py")
state_module = _load_module("state", "state.py")
bridge_module = _load_module(
    "pico.pose_to_smpl_ref_bridge",
    "pico/pose_to_smpl_ref_bridge.py",
)

IncomingChunk = merger_module.IncomingChunk
StreamedSmplRefMerger = merger_module.StreamedSmplRefMerger
SonicTeleopPolicy = policy_module.SonicTeleopPolicy
WINDOW = merger_module.WINDOW
NUM_JOINTS = policy_module.NUM_JOINTS
MODEL_INPUT_DIM = policy_module.MODEL_INPUT_DIM
ACTION_CLIP = policy_module.ACTION_CLIP
SonicTeleopState = state_module.SonicTeleopState


ZERO_LAB_METADATA_NAMES = {
    "source_generation",
    "latest_real_frame_index",
    "latest_real_receive_timestamp_ns",
    "real_valid_frames_in_generation",
    "real_stream_ready",
    "playout_kind",
    "source_stale",
}


class _FakeBackend:
    def __init__(self) -> None:
        self.fail = False
        self.inputs = []

    def run(self, inputs):
        self.inputs.append(inputs["obs_dict"].copy())
        if self.fail:
            raise RuntimeError("synthetic inference failure")
        return {"action": np.ones((1, NUM_JOINTS), dtype=np.float32)}

    def close(self) -> None:
        pass


class _Logger:
    def __init__(self) -> None:
        self.warnings = []

    def info(self, _message: str) -> None:
        pass

    def warning(self, message: str) -> None:
        self.warnings.append(message)


def _chunk(start: int, count: int = WINDOW) -> IncomingChunk:
    frames = np.arange(start, start + count, dtype=np.int64)
    term1 = np.zeros((count, 72), dtype=np.float32)
    term1[:, 0] = frames
    root = np.zeros((count, 4), dtype=np.float32)
    root[:, 0] = 1.0
    wrist = np.zeros((count, 6), dtype=np.float32)
    wrist[:, 0] = frames
    head = np.zeros((count, 2), dtype=np.float32)
    head[:, 0] = frames
    return IncomingChunk(frames, term1, root, wrist, head)


def _publish_tick(merger: StreamedSmplRefMerger):
    reference = merger.build_smpl_ref()
    assert reference is not None
    merger.advance_after_successful_tick()
    return reference


def _source_fields(start: int, *, epoch: int = 1):
    chunk = _chunk(start)
    return {
        "source_chunk": np.asarray([1], dtype=np.uint8),
        "source_stream_epoch": np.asarray([epoch], dtype=np.int64),
        "frame_index": chunk.frame_indices,
        "term1_local": chunk.term1_local,
        "root_quat": chunk.root_quat,
        "wrist": chunk.wrist,
        "head_joint_pos": chunk.head_joint_pos,
        "valid_horizon": np.asarray([WINDOW], dtype=np.int32),
        "clamp_slots": np.asarray([0], dtype=np.int32),
    }


def _zerolab_metadata(
    *, generation, latest_real_ns=None, real_frames=10, ready=True,
    playout_kind=PlayoutKind.REAL, stale=False,
):
    return {
        "source_generation": np.asarray([generation], dtype=np.int64),
        "latest_real_frame_index": np.asarray([real_frames - 1], dtype=np.int64),
        "latest_real_receive_timestamp_ns": np.asarray(
            [time.monotonic_ns() if latest_real_ns is None else latest_real_ns],
            dtype=np.int64,
        ),
        "real_valid_frames_in_generation": np.asarray(
            [real_frames], dtype=np.int32
        ),
        "real_stream_ready": np.asarray([int(ready)], dtype=np.uint8),
        "playout_kind": np.asarray([int(playout_kind)], dtype=np.uint8),
        "source_stale": np.asarray([int(stale)], dtype=np.uint8),
    }


def _zerolab_source_fields(start, *, generation):
    fields = _source_fields(start, epoch=generation)
    fields.update(_zerolab_metadata(generation=generation))
    return fields


def _manager_fields(start: int, count: int = WINDOW):
    frames = np.arange(start, start + count, dtype=np.int64)
    smpl = np.zeros((count, 24, 3), dtype=np.float32)
    smpl[:, 0, 0] = frames
    root = np.zeros((count, 4), dtype=np.float32)
    root[:, 0] = 1.0
    joints = np.zeros((count, 29), dtype=np.float32)
    joints[:, 19] = frames
    head = np.zeros((count, 2), dtype=np.float32)
    head[:, 0] = frames
    return {
        "frame_index": frames,
        "smpl_joints": smpl,
        "body_quat_w": root,
        "joint_pos": joints,
        "head_joint_pos": head,
    }


def _make_policy(monkeypatch) -> SonicTeleopPolicy:
    def fake_load_reference(self):
        self.ref_term1 = np.zeros((WINDOW, 72), dtype=np.float32)
        self.ref_term1[:, 0] = 42.0
        self.ref_root_quat = np.zeros((WINDOW, 4), dtype=np.float32)
        self.ref_root_quat[:, 0] = 1.0
        self.ref_wrist = np.zeros((WINDOW, 6), dtype=np.float32)
        self.ref_anchor_quat = None
        self.idle_frame_start = 0

    def fake_init_backend(self, _backend):
        self._backend = _FakeBackend()
        self.input_buffer = np.zeros((1, MODEL_INPUT_DIM), dtype=np.float32)
        self._inputs = {"obs_dict": self.input_buffer}

    def fake_init_zmq(self):
        self._reference_messages = queue.Queue(maxsize=64)
        self._zmq_stop = SimpleNamespace(set=lambda: None)
        self._zmq_thread = None
        self._zmq_error = None

    monkeypatch.setattr(
        SonicTeleopPolicy,
        "_load_stream_reference",
        fake_load_reference,
    )
    monkeypatch.setattr(SonicTeleopPolicy, "_init_backend", fake_init_backend)
    monkeypatch.setattr(SonicTeleopPolicy, "_init_zmq", fake_init_zmq)
    policy = SonicTeleopPolicy(
        "unused.onnx",
        "unused.npz",
        use_smpl_ref_zmq=False,
        yaw_bias_rad=0.0,
        source_blend_duration_s=0.0,
    )
    policy.bind_logger(_Logger())
    return policy


def _observation():
    return (
        np.zeros(NUM_JOINTS, dtype=np.float32),
        np.zeros(NUM_JOINTS, dtype=np.float32),
        np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        np.zeros(3, dtype=np.float32),
    )


def test_record_applied_joint_target_normalizes_clips_and_preserves_output(
    monkeypatch,
):
    policy = _make_policy(monkeypatch)
    output_before = policy.target_dof_pos.copy()
    normalized = np.linspace(-25.0, 25.0, NUM_JOINTS, dtype=np.float32)
    applied = policy.default_dof_pos + normalized * policy.action_scale

    policy.record_applied_joint_target(applied)

    np.testing.assert_allclose(
        policy.last_action,
        np.clip(normalized, -ACTION_CLIP, ACTION_CLIP),
        atol=1e-6,
    )
    np.testing.assert_array_equal(policy.target_dof_pos, output_before)


@pytest.mark.parametrize(
    "value",
    [
        np.zeros(NUM_JOINTS - 1, dtype=np.float32),
        np.zeros(NUM_JOINTS + 1, dtype=np.float32),
        np.full(NUM_JOINTS, np.nan, dtype=np.float32),
        np.full(NUM_JOINTS, np.inf, dtype=np.float32),
    ],
)
def test_record_applied_joint_target_rejects_invalid_input(monkeypatch, value):
    policy = _make_policy(monkeypatch)
    before = policy.last_action.copy()

    with pytest.raises(ValueError, match="29 finite"):
        policy.record_applied_joint_target(value)

    np.testing.assert_array_equal(policy.last_action, before)


def test_missing_canfd_packet_disables_only_optional_gripper(monkeypatch):
    monkeypatch.delattr(state_module.bxi_msg, "CANFDPacket", raising=False)
    monkeypatch.delattr(state_module.bxi_msg, "CanfdPacket", raising=False)

    state = SonicTeleopState.__new__(SonicTeleopState)
    state.hardware_gripper = True
    state._gripper_available = False
    state._policy = SimpleNamespace(status="unloaded")
    logger = _Logger()
    state._bind_logger(logger)

    state.on_bind(SimpleNamespace())

    assert state.hardware_gripper is False
    assert state._gripper_available is True
    assert state.is_available(SimpleNamespace()) is True
    assert logger.warnings == [
        "SONIC夹爪已禁用：缺少communication.msg.CANFDPacket；"
        "全身遥操仍可用"
    ]


def test_merger_never_clamps_and_advances_at_most_one_frame():
    merger = StreamedSmplRefMerger()
    merger.merge(_chunk(0))
    first = _publish_tick(merger)
    assert int(first["frame_index"][0]) == 0
    assert int(first["valid_horizon"][0]) == WINDOW
    assert int(first["clamp_slots"][0]) == 0
    np.testing.assert_array_equal(first["term1_local"][:, 0], np.arange(10))
    assert merger.current_frame == 0

    merger.merge(_chunk(1))
    second = _publish_tick(merger)
    assert int(second["frame_index"][0]) == 0
    assert merger.current_frame == 0

    merger.merge(_chunk(2))
    third = _publish_tick(merger)
    assert int(third["frame_index"][0]) == 0
    assert merger.current_frame == 1


def test_bridge_schema_is_one_way_complete_chunk_without_ack():
    chunk = bridge_module._parse_incoming_chunk(_manager_fields(100))
    fields = bridge_module._source_chunk_fields(
        chunk,
        source_stream_epoch=7,
        received_monotonic_ns=123,
    )
    assert int(fields["source_chunk"][0]) == 1
    assert int(fields["source_stream_epoch"][0]) == 7
    np.testing.assert_array_equal(fields["frame_index"], np.arange(100, 110))
    np.testing.assert_array_equal(
        fields["head_joint_pos"][:, 0],
        np.arange(100, 110),
    )
    assert int(fields["valid_horizon"][0]) == WINDOW
    assert int(fields["clamp_slots"][0]) == 0
    assert not any(
        key.startswith("ack_")
        or key in {"playout_seq", "consumer_session", "reset_stream"}
        for key in fields
    )


def test_bridge_preserves_complete_zerolab_source_metadata():
    incoming = _manager_fields(100)
    incoming.update(
        _zerolab_metadata(generation=77, real_frames=10, ready=True)
    )
    chunk = bridge_module._parse_incoming_chunk(incoming)
    metadata = bridge_module._parse_optional_source_metadata(incoming)
    fields = bridge_module._source_chunk_fields(
        chunk,
        source_stream_epoch=77,
        received_monotonic_ns=999,
        source_metadata=metadata,
    )
    for name in ZERO_LAB_METADATA_NAMES:
        np.testing.assert_array_equal(fields[name], incoming[name])


def test_bridge_rejects_partial_or_invalid_zerolab_metadata():
    fields = _manager_fields(0)
    fields["source_generation"] = np.asarray([7], dtype=np.int64)
    with pytest.raises(ValueError, match="metadata"):
        bridge_module._parse_optional_source_metadata(fields)


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("source_generation", np.asarray([0], dtype=np.int64)),
        ("latest_real_frame_index", np.asarray([-1], dtype=np.int64)),
        (
            "latest_real_receive_timestamp_ns",
            np.asarray([0], dtype=np.int64),
        ),
        (
            "real_valid_frames_in_generation",
            np.asarray([-1], dtype=np.int32),
        ),
        ("real_stream_ready", np.asarray([2], dtype=np.uint8)),
        ("playout_kind", np.asarray([4], dtype=np.uint8)),
        ("source_stale", np.asarray([2], dtype=np.uint8)),
        ("source_generation", np.asarray([7], dtype=np.int32)),
        ("source_generation", np.asarray([7, 8], dtype=np.int64)),
    ),
)
def test_bridge_rejects_invalid_zerolab_metadata_scalar(name, value):
    fields = _manager_fields(0)
    fields.update(_zerolab_metadata(generation=7))
    fields[name] = value
    with pytest.raises(ValueError, match="metadata"):
        bridge_module._parse_optional_source_metadata(fields)


def test_metadata_free_pico_schema_is_byte_compatible_in_fields():
    chunk = bridge_module._parse_incoming_chunk(_manager_fields(0))
    fields = bridge_module._source_chunk_fields(
        chunk,
        source_stream_epoch=7,
        received_monotonic_ns=123,
        source_metadata=None,
    )
    assert not any(name in fields for name in ZERO_LAB_METADATA_NAMES)


def test_bridge_rejects_incomplete_or_nonconsecutive_source_windows():
    with pytest.raises(ValueError, match="need at least 10"):
        bridge_module._parse_incoming_chunk(_manager_fields(0, WINDOW - 1))

    fields = _manager_fields(0)
    fields["frame_index"][3] += 1
    with pytest.raises(ValueError, match="must be consecutive"):
        bridge_module._parse_incoming_chunk(fields)


def test_bridge_duplicate_does_not_revoke_ready_gate():
    gate = bridge_module.PicoSourceReadinessGate(required_consecutive=3)

    def fields(newest: int):
        value = _manager_fields(newest - WINDOW + 1)
        value["stream_mode"] = np.asarray([1], dtype=np.int32)
        value["calibration_ready"] = np.asarray([1], dtype=np.uint8)
        return value

    assert gate.observe(fields(9), 1.00, 0.5) is False
    assert gate.observe(fields(10), 1.02, 0.5) is False
    assert gate.observe(fields(11), 1.04, 0.5) is True
    ready_mono = gate.last_ready_mono

    assert gate.observe(fields(11), 1.06, 0.5) is True
    assert gate.last_ready_mono == ready_mono
    assert gate.observe(fields(12), 1.08, 0.5) is True


def test_burst_chunks_are_consumed_in_order_instead_of_jumping_latest():
    merger = StreamedSmplRefMerger()
    for start in range(6):
        merger.merge(_chunk(start))
    starts = [int(_publish_tick(merger)["frame_index"][0]) for _ in range(8)]
    assert starts == [0, 1, 2, 3, 4, 4, 4, 4]


def test_policy_holds_last_complete_window_after_disconnect(monkeypatch):
    policy = _make_policy(monkeypatch)
    now = time.monotonic()
    for start in range(6):
        assert policy._merge_source_fields(_source_fields(start), now + start * 0.001)

    observed = []
    for _ in range(8):
        policy.inference_step(*_observation())
        observed.append(int(policy._backend.inputs[-1][0, 0]))
    assert observed == [0, 1, 2, 3, 4, 4, 4, 4]

    policy.last_source_rx_mono = time.monotonic() - 1.0
    policy.inference_step(*_observation())
    assert policy.last_status == "stale_hold"
    assert int(policy._backend.inputs[-1][0, 0]) == 4


def test_zerolab_freshness_uses_real_udp_timestamp_not_zmq_arrival(
    monkeypatch,
):
    clock_ns = [2_000_000_000]
    monkeypatch.setattr(policy_module.time, "monotonic_ns", lambda: clock_ns[0])
    policy = _make_policy(monkeypatch)
    fields = _source_fields(0, epoch=77)
    fields.update(
        _zerolab_metadata(
            generation=77,
            latest_real_ns=1_000_000_000,
            real_frames=10,
            ready=True,
            stale=False,
        )
    )
    policy._merge_source_fields(fields, received_mono=time.monotonic())
    policy.poll_reference()
    assert policy.has_fresh_live_reference(0.5) is False


def test_interpolated_and_held_chunks_cannot_make_recovery_ready(monkeypatch):
    policy = _make_policy(monkeypatch)
    fields = _source_fields(0, epoch=77)
    fields.update(
        _zerolab_metadata(
            generation=77,
            real_frames=9,
            ready=False,
            playout_kind=PlayoutKind.HELD,
        )
    )
    policy._merge_source_fields(fields, time.monotonic())
    assert policy.live_reference_recovery_ready(10) is False


def test_ten_real_fresh_frames_make_recovery_ready(monkeypatch):
    policy = _make_policy(monkeypatch)
    fields = _source_fields(0, epoch=77)
    fields.update(_zerolab_metadata(generation=77, real_frames=10, ready=True))
    policy._merge_source_fields(fields, time.monotonic())
    assert policy.live_reference_recovery_ready(10) is True


def test_policy_rejects_delayed_message_from_previously_seen_generation(
    monkeypatch,
):
    policy = _make_policy(monkeypatch)
    assert policy._merge_source_fields(
        _zerolab_source_fields(0, generation=11), 1.0
    )
    assert policy._merge_source_fields(
        _zerolab_source_fields(10, generation=22), 2.0
    )
    assert (
        policy._merge_source_fields(
            _zerolab_source_fields(20, generation=11), 3.0
        )
        is False
    )
    assert policy.source_generation == 22


def test_policy_reset_clears_zerolab_generation_tracking(monkeypatch):
    policy = _make_policy(monkeypatch)
    assert policy._merge_source_fields(
        _zerolab_source_fields(0, generation=11), 1.0
    )

    policy.reset()

    assert policy.source_generation is None
    assert policy._merge_source_fields(
        _zerolab_source_fields(20, generation=11), 3.0
    )


def test_policy_hold_keeps_reference_but_continues_inference(monkeypatch):
    policy = _make_policy(monkeypatch)
    now = time.monotonic()
    policy._merge_source_fields(_source_fields(0), now)
    policy.inference_step(*_observation())
    held_term = policy._backend.inputs[-1][0, 0]
    calls = len(policy._backend.inputs)

    assert policy.hold_live_reference() is True
    policy._merge_source_fields(_source_fields(100), now + 0.1)
    policy.inference_step(*_observation())

    assert len(policy._backend.inputs) == calls + 1
    assert policy._backend.inputs[-1][0, 0] == held_term
    assert policy.has_fresh_live_reference(0.5)
    assert policy.last_status == "held_reference"


def test_policy_rearm_exposes_interpolated_reference(monkeypatch):
    policy = _make_policy(monkeypatch)
    now = time.monotonic()
    policy._merge_source_fields(_source_fields(0), now)
    policy.inference_step(*_observation())
    policy.hold_live_reference()
    policy._merge_source_fields(_source_fields(10, epoch=2), now + 0.1)
    assert policy.begin_live_reference_rearm() is True
    policy.set_live_reference_rearm_progress(0.5)

    policy.inference_step(*_observation())

    assert policy._backend.inputs[-1][0, 0] == pytest.approx(5.0)
    assert policy.last_status == "rearming_reference"
    policy.complete_live_reference_rearm()
    assert policy._live_reference_gate.mode.name == "LIVE"


def test_policy_reset_returns_reference_gate_to_live(monkeypatch):
    policy = _make_policy(monkeypatch)
    policy._merge_source_fields(_source_fields(0), time.monotonic())
    policy.inference_step(*_observation())
    assert policy.hold_live_reference() is True
    assert policy._live_reference_gate.mode.name == "HOLD"

    policy.reset()

    assert policy._live_reference_gate.mode.name == "LIVE"


def test_backend_failure_does_not_advance_policy_cursor(monkeypatch):
    policy = _make_policy(monkeypatch)
    now = time.monotonic()
    for start in range(3):
        policy._merge_source_fields(_source_fields(start), now + start * 0.001)
    policy._backend.fail = True

    with pytest.raises(RuntimeError, match="synthetic inference failure"):
        policy.inference_step(*_observation())
    assert policy.stream_merger.current_frame == 0
    assert policy.latest_playback_telemetry is None

    policy._backend.fail = False
    policy.inference_step(*_observation())
    assert policy.stream_merger.current_frame == 1
    assert policy.latest_playback_telemetry.frame_index == 0


def test_source_epoch_restart_resets_merger_and_yaw(monkeypatch):
    policy = _make_policy(monkeypatch)
    now = time.monotonic()
    policy._merge_source_fields(_source_fields(50, epoch=1), now)
    policy.inference_step(*_observation())
    old_merger_epoch = policy.stream_merger.stream_epoch
    policy.yaw_aligned = True

    policy._merge_source_fields(_source_fields(500, epoch=2), now + 1.0)
    policy.inference_step(*_observation())
    assert policy.stream_merger.stream_epoch != old_merger_epoch
    assert policy.latest_playback_telemetry.frame_index == 500
    assert policy.source_stream_epoch == 2


def test_policy_reset_rejects_receive_handoff_from_previous_generation(
    monkeypatch,
):
    policy = _make_policy(monkeypatch)
    old_generation = policy._reference_generation
    old_received_mono = time.monotonic()
    fields = _source_fields(100)
    fields["source_received_monotonic_ns"] = np.asarray(
        [int(old_received_mono * 1.0e9)],
        dtype=np.int64,
    )
    message = bridge_module.pack_pose_message(
        fields,
        topic=policy.smpl_ref_zmq_topic,
        version=5,
    )

    policy.reset()
    policy._queue_reference_message(
        message,
        old_received_mono,
        old_generation,
    )
    assert policy.poll_reference() is None
    assert policy.stream_merger.timesteps == 0
    assert policy.has_seen_live_reference is False
    assert policy.source_pre_reset_drops == 1


def test_telemetry_sequence_has_no_skips(monkeypatch):
    policy = _make_policy(monkeypatch)
    now = time.monotonic()
    for start in range(20):
        policy._merge_source_fields(_source_fields(start), now + start * 0.001)

    consumed = []
    for _ in range(18):
        policy.inference_step(*_observation())
        consumed.append(policy.latest_playback_telemetry.frame_index)
    assert all(curr - prev in (0, 1) for prev, curr in zip(consumed, consumed[1:]))
    assert consumed[:5] == [0, 1, 2, 3, 4]
