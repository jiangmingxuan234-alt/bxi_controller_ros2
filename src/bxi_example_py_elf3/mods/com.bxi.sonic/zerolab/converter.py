"""Quaternion calibration and ZeroLab-to-SONIC pose conversion."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation, Slerp

from .protocol import ZeroLabPacket

if __package__ == "zerolab":
    from pico.gear_sonic.trl.utils.elf3_wrist import build_elf3_joint_pos
    from pico.gear_sonic.trl.utils.numpy_smpl import compute_from_body_poses
else:
    from ..pico.gear_sonic.trl.utils.elf3_wrist import build_elf3_joint_pos
    from ..pico.gear_sonic.trl.utils.numpy_smpl import compute_from_body_poses


BODY_JOINT_COUNT = 17
SMPL24_PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21,
]

_BODY_QUATERNION_SHAPE = (BODY_JOINT_COUNT, 4)
_PACKET_QUATERNION_SHAPE = (47, 4)


@dataclass(frozen=True)
class ConvertedPoseFrame:
    """One ZeroLab frame converted to the arrays expected by SONIC."""

    frame_index: int
    receive_timestamp_ns: int
    smpl_body_pose: NDArray[np.float32]
    smpl_joints: NDArray[np.float32]
    body_quat_w: NDArray[np.float32]
    joint_pos: NDArray[np.float32]


def _normalize_quaternions(quats, expected_shape):
    """Return finite, nonzero scalar-last quaternions as contiguous float32."""
    source = np.asarray(quats)
    if not np.issubdtype(source.dtype, np.number) or np.issubdtype(
        source.dtype, np.complexfloating
    ):
        raise ValueError("quaternion array must use a real numeric dtype")
    values = source.astype(np.float64, copy=False)
    if values.shape != expected_shape or not np.isfinite(values).all():
        raise ValueError(
            f"quaternion array must be finite with shape {expected_shape}"
        )
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norms < 1e-6):
        raise ValueError("quaternion norm is below 1e-6")
    return np.ascontiguousarray(values / norms, dtype=np.float32)


def unity_world_quaternions_to_xrt(quats_xyzw, expected_shape):
    """Reflect Unity world quaternions into the XRT coordinate system."""
    converted = _normalize_quaternions(quats_xyzw, expected_shape)
    converted[:, :2] *= -1.0
    return np.ascontiguousarray(converted, dtype=np.float32)


def align_quaternion_signs(current_xyzw, previous_xyzw=None):
    """Choose quaternion representatives closest to a previous frame."""
    current_values = np.asarray(current_xyzw)
    if current_values.ndim != 2 or current_values.shape[1] != 4:
        raise ValueError("quaternion array must have shape (N, 4)")
    current = _normalize_quaternions(current_values, current_values.shape)
    if previous_xyzw is None:
        return current
    previous = _normalize_quaternions(previous_xyzw, current.shape)
    flip = np.einsum("ij,ij->i", current, previous) < 0.0
    current[flip] *= -1.0
    return current


def apply_rest_alignment(raw_xyzw, rest_xyzw):
    """Express raw world rotations relative to calibrated rest rotations."""
    raw = _normalize_quaternions(raw_xyzw, _BODY_QUATERNION_SHAPE)
    rest = _normalize_quaternions(rest_xyzw, _BODY_QUATERNION_SHAPE)
    aligned = (
        Rotation.from_quat(raw) * Rotation.from_quat(rest).inv()
    ).as_quat()
    return np.ascontiguousarray(aligned, dtype=np.float32)


class TPoseCalibrator:
    """Collect a stable quaternion window and derive its T-pose rest frame."""

    def __init__(
        self,
        required_frames: int = 100,
        max_deviation_degrees: float = 5.0,
    ) -> None:
        if isinstance(required_frames, bool) or required_frames < 1:
            raise ValueError("required_frames must be at least 1")
        if (
            not np.isfinite(max_deviation_degrees)
            or max_deviation_degrees < 0.0
        ):
            raise ValueError(
                "max_deviation_degrees must be finite and non-negative"
            )
        self._required_frames = int(required_frames)
        self._max_deviation_degrees = float(max_deviation_degrees)
        self._window = []
        self._rest_quats_xyzw = None

    @property
    def frames_collected(self) -> int:
        return len(self._window)

    @property
    def is_calibrated(self) -> bool:
        return self._rest_quats_xyzw is not None

    @property
    def rest_quats_xyzw(self):
        if self._rest_quats_xyzw is None:
            return None
        return self._rest_quats_xyzw.copy()

    def reset(self) -> None:
        self._window = []
        self._rest_quats_xyzw = None

    def observe(self, frame_xyzw) -> bool:
        frame = _normalize_quaternions(frame_xyzw, _BODY_QUATERNION_SHAPE)
        if self.is_calibrated:
            return True

        previous = self._window[-1] if self._window else None
        frame = align_quaternion_signs(frame, previous)
        candidate = self._window + [frame]
        mean = _normalize_quaternions(
            np.mean(candidate, axis=0, dtype=np.float64),
            _BODY_QUATERNION_SHAPE,
        )
        dots = np.sum(np.asarray(candidate) * mean, axis=2)
        angular_distances = np.degrees(
            2.0 * np.arccos(np.clip(np.abs(dots), 0.0, 1.0))
        )
        if np.any(angular_distances > self._max_deviation_degrees):
            self._window = [frame]
        else:
            self._window = candidate

        if len(self._window) == self._required_frames:
            self._rest_quats_xyzw = mean
            return True
        return False


def _shortest_path_slerp(start_xyzw, end_xyzw, fraction):
    start = _normalize_quaternions(
        np.asarray(start_xyzw)[None, :], (1, 4)
    )[0]
    end = _normalize_quaternions(np.asarray(end_xyzw)[None, :], (1, 4))[0]
    if np.dot(start, end) < 0.0:
        end *= -1.0
    rotations = Rotation.from_quat(np.stack((start, end)))
    result = Slerp([0.0, 1.0], rotations)([fraction]).as_quat()[0]
    return np.ascontiguousarray(result, dtype=np.float32)


def synthesize_smpl_world_quats(aligned_body_xyzw):
    """Map 17 aligned ZeroLab world rotations to 24 SMPL world rotations."""
    body = _normalize_quaternions(aligned_body_xyzw, _BODY_QUATERNION_SHAPE)
    smpl = np.empty((24, 4), dtype=np.float32)

    smpl[0] = body[10]
    smpl[1] = body[11]
    smpl[2] = body[14]
    smpl[3] = _shortest_path_slerp(body[10], body[1], 1.0 / 3.0)
    smpl[4] = body[12]
    smpl[5] = body[15]
    smpl[6] = _shortest_path_slerp(body[10], body[1], 2.0 / 3.0)
    smpl[7] = body[13]
    smpl[8] = body[16]
    smpl[9] = body[1]
    smpl[10] = smpl[7]
    smpl[11] = smpl[8]
    smpl[12] = _shortest_path_slerp(body[1], body[0], 0.5)
    smpl[13] = body[2]
    smpl[14] = body[6]
    smpl[15] = body[0]
    smpl[16] = body[3]
    smpl[17] = body[7]
    smpl[18] = body[4]
    smpl[19] = body[8]
    smpl[20] = body[5]
    smpl[21] = body[9]
    smpl[22] = smpl[20]
    smpl[23] = smpl[21]
    return smpl


def _validated_root_translation(root_translation):
    source = np.asarray(root_translation)
    if not np.issubdtype(source.dtype, np.number) or np.issubdtype(
        source.dtype, np.complexfloating
    ):
        raise ValueError("root translation must use a real numeric dtype")
    values = source.astype(np.float64, copy=False)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("root translation must be finite with shape (3,)")
    return np.ascontiguousarray(values, dtype=np.float32)


def _validated_converted_array(
    name, values, expected_shape, *, quaternion=False
):
    array = np.asarray(values)
    if array.shape != expected_shape or not np.isfinite(array).all():
        raise ValueError(
            f"derived {name} must be finite with shape {expected_shape}"
        )
    result = np.ascontiguousarray(array, dtype=np.float32)
    if quaternion:
        norm = float(np.linalg.norm(result.astype(np.float64)))
        if not np.isfinite(norm) or not np.isclose(norm, 1.0, atol=1e-5):
            raise ValueError(f"derived {name} must be a unit quaternion")
    return result


class ZeroLabMotionConverter:
    """Calibrate ZeroLab rest quaternions and emit SONIC pose frames."""

    def __init__(self) -> None:
        self._calibrator = TPoseCalibrator()
        self._previous_raw_quats_xyzw = None

    def mark_stale(self) -> None:
        """Clear continuity and any incomplete rest-calibration window."""
        self._previous_raw_quats_xyzw = None
        if not self._calibrator.is_calibrated:
            self._calibrator.reset()

    def reset_session(self) -> None:
        """Clear quaternion continuity and all session calibration."""
        self._previous_raw_quats_xyzw = None
        self._calibrator.reset()

    def observe(self, packet: ZeroLabPacket) -> ConvertedPoseFrame | None:
        """Observe a packet and return a frame after rest calibration."""
        raw_quats = unity_world_quaternions_to_xrt(
            packet.joint_quat_world_xyzw, _PACKET_QUATERNION_SHAPE
        )
        raw_quats = align_quaternion_signs(
            raw_quats, self._previous_raw_quats_xyzw
        )

        if not self._calibrator.is_calibrated:
            self._calibrator.observe(raw_quats[:BODY_JOINT_COUNT])
            self._previous_raw_quats_xyzw = raw_quats
            return None

        aligned_body = apply_rest_alignment(
            raw_quats[:BODY_JOINT_COUNT],
            self._calibrator.rest_quats_xyzw,
        )
        smpl_world_quats = synthesize_smpl_world_quats(aligned_body)
        body_poses = np.zeros((24, 7), dtype=np.float32)
        body_poses[:, 3:] = smpl_world_quats
        root_translation = _validated_root_translation(packet.root_translation)
        root_translation[2] *= -1.0
        body_poses[0, :3] = root_translation

        result = compute_from_body_poses(SMPL24_PARENTS, body_poses)
        smpl_body_pose = _validated_converted_array(
            "smpl_body_pose",
            result["smpl_pose"][0, :63].reshape(21, 3),
            (21, 3),
        )
        smpl_joints = _validated_converted_array(
            "smpl_joints", result["smpl_joints_local"][0], (24, 3)
        )
        body_quat_w = _validated_converted_array(
            "body_quat_w",
            result["global_orient_quat"][0],
            (4,),
            quaternion=True,
        )
        joint_pos = _validated_converted_array(
            "joint_pos",
            build_elf3_joint_pos(smpl_body_pose[None, ...])[0],
            (29,),
        )

        self._previous_raw_quats_xyzw = raw_quats
        return ConvertedPoseFrame(
            frame_index=int(packet.local_frame_index),
            receive_timestamp_ns=int(packet.receive_timestamp_ns),
            smpl_body_pose=smpl_body_pose,
            smpl_joints=smpl_joints,
            body_quat_w=body_quat_w,
            joint_pos=joint_pos,
        )
