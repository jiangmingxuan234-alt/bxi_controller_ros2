"""Versioned local UDP protocol for live PICO/GMR reference frames."""

from __future__ import annotations

from dataclasses import dataclass
import struct
import zlib

import numpy as np

from bxi_example_py_elf3.policies.joints import ELF3_ISAAC_JOINTS


MAGIC = b"PGMR"
VERSION = 1
REFERENCE_WINDOW_SIZE = 21
REFERENCE_JOINT_NAMES = ELF3_ISAAC_JOINTS.names
REFERENCE_FLOAT_COUNT = len(REFERENCE_JOINT_NAMES) + 4 + 3 + 3
LAYOUT_CRC32 = zlib.crc32("\0".join(REFERENCE_JOINT_NAMES).encode("utf-8"))
_HEADER = struct.Struct("!4sHHIQQQ")
PACKET_SIZE = _HEADER.size + REFERENCE_FLOAT_COUNT * 4


@dataclass(frozen=True, slots=True)
class LiveReferenceFrame:
    session_id: int
    sequence: int
    source_timestamp_ns: int
    joint_pos: np.ndarray
    anchor_quat_wxyz: np.ndarray
    anchor_lin_vel_w: np.ndarray
    anchor_ang_vel_w: np.ndarray


def _finite(name: str, value: object, shape: tuple[int, ...]) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return result


def encode_reference_frame(frame: LiveReferenceFrame) -> bytes:
    joints = _finite("joint_pos", frame.joint_pos, (len(REFERENCE_JOINT_NAMES),))
    quat = _finite("anchor_quat_wxyz", frame.anchor_quat_wxyz, (4,))
    norm = float(np.linalg.norm(quat))
    if norm <= np.finfo(np.float32).eps:
        raise ValueError("anchor_quat_wxyz has zero length")
    quat = quat / norm
    lin_vel = _finite("anchor_lin_vel_w", frame.anchor_lin_vel_w, (3,))
    ang_vel = _finite("anchor_ang_vel_w", frame.anchor_ang_vel_w, (3,))
    payload = np.concatenate((joints, quat, lin_vel, ang_vel)).astype(">f4", copy=False)
    header = _HEADER.pack(
        MAGIC,
        VERSION,
        REFERENCE_FLOAT_COUNT,
        LAYOUT_CRC32,
        int(frame.session_id),
        int(frame.sequence),
        int(frame.source_timestamp_ns),
    )
    return header + payload.tobytes()


def decode_reference_frame(packet: bytes) -> LiveReferenceFrame:
    if len(packet) != PACKET_SIZE:
        raise ValueError(f"PICO GMR packet has {len(packet)} bytes, expected {PACKET_SIZE}")
    magic, version, count, layout_crc, session_id, sequence, source_ns = _HEADER.unpack_from(packet)
    if magic != MAGIC:
        raise ValueError("PICO GMR packet magic mismatch")
    if version != VERSION:
        raise ValueError(f"unsupported PICO GMR protocol version {version}")
    if count != REFERENCE_FLOAT_COUNT:
        raise ValueError(f"PICO GMR payload count mismatch: {count}")
    if layout_crc != LAYOUT_CRC32:
        raise ValueError("PICO GMR named-joint layout mismatch")
    values = np.frombuffer(packet, dtype=">f4", offset=_HEADER.size).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("PICO GMR packet contains NaN or infinity")
    joint_end = len(REFERENCE_JOINT_NAMES)
    quat_end = joint_end + 4
    lin_end = quat_end + 3
    quat = values[joint_end:quat_end].copy()
    norm = float(np.linalg.norm(quat))
    if norm <= np.finfo(np.float32).eps:
        raise ValueError("PICO GMR packet contains a zero-length quaternion")
    quat /= norm
    return LiveReferenceFrame(
        session_id=session_id,
        sequence=sequence,
        source_timestamp_ns=source_ns,
        joint_pos=values[:joint_end].copy(),
        anchor_quat_wxyz=quat,
        anchor_lin_vel_w=values[quat_end:lin_end].copy(),
        anchor_ang_vel_w=values[lin_end:].copy(),
    )


__all__ = [
    "LiveReferenceFrame",
    "PACKET_SIZE",
    "REFERENCE_JOINT_NAMES",
    "REFERENCE_WINDOW_SIZE",
    "decode_reference_frame",
    "encode_reference_frame",
]
