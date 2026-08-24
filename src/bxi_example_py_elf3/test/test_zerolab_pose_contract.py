from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from zerolab.converter import ConvertedPoseFrame, ZeroLabMotionConverter
from zerolab.protocol import ZeroLabPacket
from zerolab.resampler import PlayoutKind, ZeroLabPoseResampler
from zerolab.source_node import PoseChunkWindow, ZeroLabSourceCore
from pico.pose_to_smpl_ref_bridge import (
    PicoSourceReadinessGate,
    _decode_packed_message,
    _parse_incoming_chunk,
)
from pico.zmq_messages import pack_pose_message


SOURCE_METADATA_DTYPES = {
    "source_generation": np.int64,
    "latest_real_frame_index": np.int64,
    "latest_real_receive_timestamp_ns": np.int64,
    "real_valid_frames_in_generation": np.int32,
    "real_stream_ready": np.uint8,
    "playout_kind": np.uint8,
    "source_stale": np.uint8,
}


def converted(index, *, dtype=np.float32):
    return ConvertedPoseFrame(
        frame_index=index,
        receive_timestamp_ns=index * 20_000_000,
        smpl_body_pose=np.full((21, 3), index, dtype=dtype),
        smpl_joints=np.full((24, 3), index, dtype=dtype),
        body_quat_w=np.array([1.0, 0.0, 0.0, 0.0], dtype=dtype),
        joint_pos=np.arange(29, dtype=dtype) + index,
    )


def identity_packet(index, *, timestamp_ns=None):
    quaternions = np.zeros((47, 4), dtype=np.float32)
    quaternions[:, 3] = 1.0
    return ZeroLabPacket(
        receive_timestamp_ns=(
            index * 20_000_000 if timestamp_ns is None else timestamp_ns
        ),
        local_frame_index=index,
        root_translation=np.zeros(3, dtype=np.float32),
        joint_quat_world_xyzw=quaternions,
        left_hand_values=np.zeros(6, dtype=np.uint16),
        right_hand_values=np.zeros(6, dtype=np.uint16),
        joint_position=np.zeros((17, 3), dtype=np.float32),
        raw_payload=bytes(992),
        sender_address=("127.0.0.1", 50000),
    )


def make_core(*, generations=(101, 202, 303)):
    values = iter(generations)
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
        generation_factory=lambda _previous=None: next(values),
    )


def core_with_fake_generation(generation):
    return make_core(generations=(generation, generation + 1))


def ready_core(*, generations=(101, 202)):
    core = make_core(generations=generations)
    fields = None
    for index in range(10):
        core.accept(identity_packet(index, timestamp_ns=index * 20_000_000))
        fields = core.sample(80_000_000 + index * 20_000_000)
    assert fields is not None
    assert int(fields["real_stream_ready"][0]) == 1
    return core


def stale_core():
    core = ready_core()
    assert core.check_stale(
        core.latest_real_receive_timestamp_ns + 500_000_001
    )
    return core


def collect_next_window(core, *, start_ns):
    now_ns = start_ns
    fields = None
    for _ in range(10):
        fields = core.sample(now_ns)
        now_ns += 20_000_000
    assert fields is not None
    return fields


def test_window_returns_no_chunk_until_ten_distinct_frames():
    window = PoseChunkWindow(10)
    for index in range(9):
        assert window.append(converted(index, dtype=np.float64)) is None

    fields = window.append(converted(9, dtype=np.float64))

    assert fields["frame_index"].shape == (10,)
    assert fields["frame_index"].dtype == np.int64
    np.testing.assert_array_equal(fields["frame_index"], np.arange(10))
    assert fields["smpl_joints"].shape == (10, 24, 3)
    assert fields["smpl_joints"].dtype == np.float32
    assert fields["body_quat_w"].shape == (10, 4)
    assert fields["body_quat_w"].dtype == np.float32
    assert fields["joint_pos"].shape == (10, 29)
    assert fields["joint_pos"].dtype == np.float32
    assert fields["head_joint_pos"].shape == (10, 2)
    assert fields["head_joint_pos"].dtype == np.float32
    np.testing.assert_array_equal(
        fields["head_joint_pos"],
        np.zeros((10, 2), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        fields["stream_mode"], np.array([1], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        fields["calibration_ready"], np.array([True], dtype=bool)
    )


def test_window_rolls_real_frames_instead_of_tiling_the_latest_frame():
    window = PoseChunkWindow(10)
    for index in range(10):
        window.append(converted(index))

    fields = window.append(converted(10))

    np.testing.assert_array_equal(fields["frame_index"], np.arange(1, 11))
    np.testing.assert_array_equal(
        fields["smpl_joints"][:, 0, 0], np.arange(1, 11)
    )


@pytest.mark.parametrize(
    "replacement",
    [
        lambda frame: frame.__class__(
            **{
                **frame.__dict__,
                "smpl_body_pose": np.zeros((20, 3), dtype=np.float32),
            }
        ),
        lambda frame: frame.__class__(
            **{
                **frame.__dict__,
                "smpl_joints": np.full((24, 3), np.nan, dtype=np.float32),
            }
        ),
        lambda frame: frame.__class__(
            **{**frame.__dict__, "body_quat_w": np.zeros(5, dtype=np.float32)}
        ),
        lambda frame: frame.__class__(
            **{
                **frame.__dict__,
                "joint_pos": np.full(29, np.inf, dtype=np.float32),
            }
        ),
    ],
)
def test_window_rejects_malformed_or_nonfinite_converted_frames(replacement):
    with pytest.raises(ValueError):
        PoseChunkWindow(10).append(replacement(converted(0)))


def test_duplicate_is_ignored_without_revoking_partial_window():
    window = PoseChunkWindow(10)
    for index in range(5):
        assert window.append(converted(index)) is None
    assert window.append(converted(4)) is None
    for index in range(5, 10):
        fields = window.append(converted(index))
    np.testing.assert_array_equal(fields["frame_index"], np.arange(10))


def test_forward_gap_starts_a_new_consecutive_window():
    window = PoseChunkWindow(10)
    for index in range(5):
        window.append(converted(index))
    assert window.append(converted(8)) is None
    for index in range(9, 18):
        fields = window.append(converted(index))
    np.testing.assert_array_equal(fields["frame_index"], np.arange(8, 18))


def test_forward_gap_discards_a_nearly_ready_window():
    window = PoseChunkWindow(10)
    for index in range(9):
        assert window.append(converted(index)) is None

    assert window.append(converted(10)) is None
    assert window.ready is False


def test_backward_index_clears_and_is_rejected():
    window = PoseChunkWindow(10)
    window.append(converted(5))
    with pytest.raises(ValueError, match="backward"):
        window.append(converted(4))

    assert window.ready is False


def test_synthetic_outputs_do_not_increment_real_readiness():
    core = core_with_fake_generation(101)
    for index in range(9):
        assert core.accept(identity_packet(index))
    for tick in range(20):
        core.sample(200_000_000 + tick * 20_000_000)
    assert core.real_valid_frames_in_generation == 9
    assert core.latest_real_receive_timestamp_ns == 160_000_000
    assert core.stats.real_valid_packets == 9
    assert core.real_stream_ready is False


def test_tenth_real_packet_and_complete_output_window_set_ready_metadata():
    core = core_with_fake_generation(101)
    fields = None
    for index in range(10):
        assert core.accept(identity_packet(index))
        fields = core.sample(80_000_000 + index * 20_000_000)
    assert fields is not None
    assert int(fields["source_generation"][0]) == 101
    assert int(fields["real_valid_frames_in_generation"][0]) == 10
    assert int(fields["real_stream_ready"][0]) == 1


def test_source_metadata_uses_exact_scalar_dtypes():
    fields = ready_core().sample(280_000_000)
    assert fields is not None
    for name, dtype in SOURCE_METADATA_DTYPES.items():
        assert fields[name].shape == (1,)
        assert fields[name].dtype == np.dtype(dtype)


def test_stale_is_strictly_greater_than_half_second_and_changes_generation():
    core = make_core(generations=(101, 202))
    core.accept(identity_packet(0, timestamp_ns=1_000_000_000))
    assert core.check_stale(1_500_000_000) is False
    assert core.check_stale(1_500_000_001) is True
    assert core.source_generation == 202
    assert core.real_valid_frames_in_generation == 0
    assert core.stats.stale_events == 1


@pytest.mark.parametrize("gap_s", [0.10, 0.49])
def test_short_gap_does_not_cross_source_stale_barrier(gap_s):
    core = ready_core(generations=(101, 202))
    last_real_ns = core.latest_real_receive_timestamp_ns
    now_ns = last_real_ns + int(gap_s * 1.0e9)
    assert core.check_stale(now_ns) is False
    fields = core.sample(now_ns)
    assert fields is not None
    assert int(fields["source_generation"][0]) == 101
    assert int(fields["source_stale"][0]) == 0


@pytest.mark.parametrize("gap_s", [0.51, 2.0, 30.0])
def test_long_gap_crosses_one_stale_generation_barrier(gap_s):
    core = ready_core(generations=(101, 202))
    last_real_ns = core.latest_real_receive_timestamp_ns
    assert core.check_stale(last_real_ns + int(gap_s * 1.0e9)) is True
    assert core.source_generation == 202
    assert core.real_valid_frames_in_generation == 0
    assert core.stats.stale_events == 1


def test_conversion_exception_preserves_last_real_authority():
    core = core_with_fake_generation(101)
    assert core.accept(identity_packet(0))
    invalid = identity_packet(1)
    invalid = ZeroLabPacket(
        receive_timestamp_ns=invalid.receive_timestamp_ns,
        local_frame_index=invalid.local_frame_index,
        root_translation=np.full(3, np.nan, dtype=np.float32),
        joint_quat_world_xyzw=invalid.joint_quat_world_xyzw,
        left_hand_values=invalid.left_hand_values,
        right_hand_values=invalid.right_hand_values,
        joint_position=invalid.joint_position,
        raw_payload=invalid.raw_payload,
        sender_address=invalid.sender_address,
    )

    with pytest.raises(ValueError, match="root translation"):
        core.accept(invalid)

    assert core.latest_real_receive_timestamp_ns == 0
    assert core.real_valid_frames_in_generation == 1
    assert core.stats.real_valid_packets == 1


def test_real_packet_stats_count_and_track_maximum_arrival_gap():
    core = core_with_fake_generation(101)
    assert core.accept(identity_packet(0, timestamp_ns=1_000_000_000))
    assert core.accept(identity_packet(1, timestamp_ns=1_020_000_000))
    assert core.accept(identity_packet(2, timestamp_ns=1_070_000_000))
    assert core.stats.real_valid_packets == 3
    assert core.stats.maximum_real_arrival_gap_ms == 50.0


def test_duplicate_real_packet_is_not_accepted_or_counted_twice():
    core = core_with_fake_generation(101)
    packet = identity_packet(0)
    assert core.accept(packet)
    assert core.accept(packet) is False
    assert core.real_valid_frames_in_generation == 1
    assert core.stats.real_valid_packets == 1


def test_real_readiness_count_is_capped_at_configured_recovery_frames():
    core = core_with_fake_generation(101)
    for index in range(12):
        assert core.accept(identity_packet(index))
    assert core.real_valid_frames_in_generation == 10
    assert core.stats.real_valid_packets == 12


def test_source_stats_are_immutable_snapshots():
    core = core_with_fake_generation(101)
    stats = core.stats
    with pytest.raises(FrozenInstanceError):
        stats.real_valid_packets = 3


def test_prearrival_stale_gap_resets_generation_before_accepting_real_packet():
    core = make_core(generations=(101, 202))
    assert core.accept(identity_packet(0, timestamp_ns=0))
    assert core.accept(identity_packet(1, timestamp_ns=500_000_001))
    assert core.source_generation == 202
    assert core.real_valid_frames_in_generation == 1
    assert core.latest_real_receive_timestamp_ns == 500_000_001
    assert core.stats.stale_events == 1
    assert core.consume_stale_event() is True


def test_stale_hold_is_published_but_cannot_be_ready():
    core = ready_core(generations=(101, 202))
    stale_ns = core.latest_real_receive_timestamp_ns + 500_000_001
    core.check_stale(stale_ns)
    fields = collect_next_window(core, start_ns=stale_ns)
    assert int(fields["source_generation"][0]) != 101
    assert int(fields["playout_kind"][0]) == int(PlayoutKind.HELD)
    assert int(fields["source_stale"][0]) == 1
    assert int(fields["real_stream_ready"][0]) == 0


def test_sample_computes_stale_metadata_directly_from_real_age():
    core = ready_core(generations=(101, 202))
    stale_ns = core.latest_real_receive_timestamp_ns + 500_000_001
    fields = core.sample(stale_ns)
    assert fields is not None
    assert int(fields["source_stale"][0]) == 1
    assert int(fields["real_stream_ready"][0]) == 0
    assert core.source_generation == 101
    assert core.real_valid_frames_in_generation == 10


def test_recovery_requires_ten_real_valid_packets_in_new_generation():
    core = stale_core()
    for index in range(9):
        core.accept(
            identity_packet(
                100 + index,
                timestamp_ns=2_000_000_000 + index * 20_000_000,
            )
        )
        fields = core.sample(2_080_000_000 + index * 20_000_000)
        if fields is not None:
            assert int(fields["real_stream_ready"][0]) == 0
    core.accept(identity_packet(109, timestamp_ns=2_180_000_000))
    fields = core.sample(2_260_000_000)
    assert int(fields["real_valid_frames_in_generation"][0]) == 10
    assert int(fields["real_stream_ready"][0]) == 1


def test_existing_bridge_accepts_three_progressing_zerolab_chunks():
    gate = PicoSourceReadinessGate(required_consecutive=3)
    ready = []
    for offset in range(3):
        window = PoseChunkWindow(10)
        fields = None
        for index in range(offset, offset + 10):
            fields = window.append(converted(index))
        message = pack_pose_message(fields, topic="pose")
        decoded = _decode_packed_message(message, "pose")
        incoming = _parse_incoming_chunk(decoded)
        assert incoming.term1_local.shape == (10, 72)
        assert incoming.root_quat.shape == (10, 4)
        assert incoming.wrist.shape == (10, 6)
        ready.append(
            gate.observe(decoded, now_mono=offset * 0.02, stale_seconds=0.5)
        )
    assert ready == [False, False, True]
