import numpy as np
import pytest

from zerolab.converter import ConvertedPoseFrame, ZeroLabMotionConverter
from zerolab.protocol import ZeroLabPacket
from zerolab.source_node import PoseChunkWindow, ZeroLabSourceCore
from pico.pose_to_smpl_ref_bridge import (
    PicoSourceReadinessGate,
    _decode_packed_message,
    _parse_incoming_chunk,
)
from pico.zmq_messages import pack_pose_message


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


def test_core_first_ready_chunk_is_source_frames_zero_through_nine():
    core = ZeroLabSourceCore(ZeroLabMotionConverter())
    for index in range(9):
        assert core.accept(identity_packet(index)) is None

    fields = core.accept(identity_packet(9))

    np.testing.assert_array_equal(fields["frame_index"], np.arange(10))


def test_completed_stream_refills_after_stale_gap():
    core = ZeroLabSourceCore(ZeroLabMotionConverter())
    for index in range(10):
        fields = core.accept(identity_packet(index))
    assert fields is not None

    last_timestamp_ns = 9 * 20_000_000
    assert core.check_stale(last_timestamp_ns + 500_000_000) is False
    assert core.check_stale(last_timestamp_ns + 500_000_001) is True
    assert core.check_stale(last_timestamp_ns + 600_000_000) is False

    fresh_start_ns = last_timestamp_ns + 500_000_001
    for index in range(10, 19):
        assert core.accept(
            identity_packet(
                index,
                timestamp_ns=fresh_start_ns + (index - 10) * 20_000_000,
            )
        ) is None
    fields = core.accept(
        identity_packet(19, timestamp_ns=fresh_start_ns + 180_000_000)
    )

    np.testing.assert_array_equal(
        fields["frame_index"], np.arange(10, 20, dtype=np.int64)
    )


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
