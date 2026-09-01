import struct

import numpy as np
import pytest

from zerolab.protocol import PACKET_SIZE, ZeroLabProtocolError, parse_zerolab_packet


def make_payload():
    root = np.array([1.25, -2.5, 3.75], dtype="<f4")
    quats = (
        np.arange(1, 189, dtype=np.float32).reshape(47, 4) / 10
    ).astype("<f4")
    left = np.arange(100, 106, dtype="<u2")
    right = np.arange(200, 206, dtype="<u2")
    positions = (np.arange(51, dtype=np.float32).reshape(17, 3) / 10).astype("<f4")
    payload = root.tobytes() + quats.tobytes() + left.tobytes() + right.tobytes() + positions.tobytes()
    return payload, root, quats, left, right, positions


def test_exact_packet_parses_all_documented_offsets():
    payload, root, quats, left, right, positions = make_payload()
    packet = parse_zerolab_packet(
        payload,
        receive_timestamp_ns=123456789,
        local_frame_index=7,
        sender_address=("192.168.1.20", 50000),
    )
    assert len(payload) == PACKET_SIZE == 992
    np.testing.assert_array_equal(packet.root_translation, root)
    np.testing.assert_array_equal(packet.joint_quat_world_xyzw, quats)
    assert packet.joint_quat_world_xyzw.shape == (47, 4)
    assert packet.joint_quat_world_xyzw.dtype == np.float32
    np.testing.assert_array_equal(packet.left_hand_values, left)
    np.testing.assert_array_equal(packet.right_hand_values, right)
    np.testing.assert_array_equal(packet.joint_position, positions)
    assert packet.raw_payload == payload
    assert packet.receive_timestamp_ns == 123456789
    assert packet.local_frame_index == 7
    assert packet.sender_address == ("192.168.1.20", 50000)


@pytest.mark.parametrize("size", [991, 993])
def test_wrong_packet_size_is_rejected(size):
    with pytest.raises(ZeroLabProtocolError, match="exactly 992"):
        parse_zerolab_packet(
            bytes(size),
            receive_timestamp_ns=1,
            local_frame_index=0,
            sender_address=("127.0.0.1", 1),
        )


@pytest.mark.parametrize(
    "offset,value",
    [
        (0, float("nan")),
        (12 + 46 * 16, float("inf")),
        (788 + 50 * 4, float("-inf")),
    ],
)
def test_non_finite_float_is_rejected(offset, value):
    payload, *_ = make_payload()
    corrupted = bytearray(payload)
    struct.pack_into("<f", corrupted, offset, value)
    with pytest.raises(ZeroLabProtocolError, match="non-finite"):
        parse_zerolab_packet(
            bytes(corrupted),
            receive_timestamp_ns=1,
            local_frame_index=0,
            sender_address=("127.0.0.1", 1),
        )


def test_near_zero_quaternion_is_rejected():
    payload, *_ = make_payload()
    corrupted = bytearray(payload)
    corrupted[12:28] = bytes(16)
    with pytest.raises(ZeroLabProtocolError, match="norm"):
        parse_zerolab_packet(
            bytes(corrupted),
            receive_timestamp_ns=1,
            local_frame_index=0,
            sender_address=("127.0.0.1", 1),
        )


def test_all_47_quaternion_norms_are_checked():
    payload, *_ = make_payload()
    corrupted = bytearray(payload)
    last_quaternion_offset = 12 + 46 * 16
    corrupted[last_quaternion_offset:last_quaternion_offset + 16] = bytes(16)
    with pytest.raises(ZeroLabProtocolError, match="norm"):
        parse_zerolab_packet(
            bytes(corrupted),
            receive_timestamp_ns=1,
            local_frame_index=0,
            sender_address=("127.0.0.1", 1),
        )
