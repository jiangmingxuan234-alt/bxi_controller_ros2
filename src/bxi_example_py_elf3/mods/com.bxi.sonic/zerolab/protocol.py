from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


PACKET_SIZE = 992
ROOT_AND_QUAT_END = 764
LEFT_HAND_END = 776
RIGHT_HAND_END = 788
QUATERNION_MIN_NORM = 1e-6


class ZeroLabProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class ZeroLabPacket:
    receive_timestamp_ns: int
    local_frame_index: int
    root_translation: NDArray[np.float32]
    joint_quat_world_xyzw: NDArray[np.float32]
    left_hand_values: NDArray[np.uint16]
    right_hand_values: NDArray[np.uint16]
    joint_position: NDArray[np.float32]
    raw_payload: bytes
    sender_address: tuple[str, int]


def parse_zerolab_packet(
    payload: bytes,
    *,
    receive_timestamp_ns: int,
    local_frame_index: int,
    sender_address: tuple[str, int],
) -> ZeroLabPacket:
    raw = bytes(payload)
    if len(raw) != PACKET_SIZE:
        raise ZeroLabProtocolError(
            f"ZeroLab datagram must be exactly {PACKET_SIZE} bytes, got {len(raw)}"
        )
    attitudes = np.frombuffer(raw[:ROOT_AND_QUAT_END], dtype="<f4").copy()
    root = attitudes[:3]
    quats = attitudes[3:].reshape(47, 4)
    left = np.frombuffer(raw[ROOT_AND_QUAT_END:LEFT_HAND_END], dtype="<u2").copy()
    right = np.frombuffer(raw[LEFT_HAND_END:RIGHT_HAND_END], dtype="<u2").copy()
    positions = np.frombuffer(raw[RIGHT_HAND_END:], dtype="<f4").copy().reshape(17, 3)
    if not all(np.isfinite(value).all() for value in (root, quats, positions)):
        raise ZeroLabProtocolError("ZeroLab datagram contains non-finite float values")
    norms = np.linalg.norm(quats.astype(np.float64), axis=1)
    if np.any(norms < QUATERNION_MIN_NORM):
        raise ZeroLabProtocolError("ZeroLab quaternion norm is below 1e-6")
    return ZeroLabPacket(
        receive_timestamp_ns=int(receive_timestamp_ns),
        local_frame_index=int(local_frame_index),
        root_translation=np.ascontiguousarray(root, dtype=np.float32),
        joint_quat_world_xyzw=np.ascontiguousarray(quats, dtype=np.float32),
        left_hand_values=np.ascontiguousarray(left, dtype=np.uint16),
        right_hand_values=np.ascontiguousarray(right, dtype=np.uint16),
        joint_position=np.ascontiguousarray(positions, dtype=np.float32),
        raw_payload=raw,
        sender_address=(str(sender_address[0]), int(sender_address[1])),
    )
