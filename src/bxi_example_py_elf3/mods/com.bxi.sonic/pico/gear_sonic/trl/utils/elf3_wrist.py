"""Source-neutral SMPL-to-ELF3 wrist conversion."""

import numpy as np
from scipy.spatial.transform import Rotation


_SMPL_BODY_POSE_SHAPE = (21, 3)
_ELF3_JOINT_COUNT = 29
_TWIST_AXIS = np.array([0.0, 1.0, 0.0], dtype=np.float64)
_SUPPORTED_DTYPES = (np.dtype(np.float32), np.dtype(np.float64))


def _validated_pose_and_dtype(smpl_body_pose, dtype):
    source = np.asarray(smpl_body_pose)
    if source.ndim != 3 or source.shape[1:] != _SMPL_BODY_POSE_SHAPE:
        raise ValueError(
            "SMPL body pose must have shape (N, 21, 3), "
            f"got {source.shape}"
        )
    if source.shape[0] == 0:
        raise ValueError("SMPL body pose batch must not be empty")
    if not np.issubdtype(source.dtype, np.number) or np.issubdtype(
        source.dtype, np.complexfloating
    ):
        raise ValueError("SMPL body pose must use a real numeric dtype")
    values = source.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("SMPL body pose must contain only finite values")

    try:
        output_dtype = np.dtype(dtype)
    except TypeError as error:
        raise ValueError("dtype must be np.float32 or np.float64") from error
    if output_dtype not in _SUPPORTED_DTYPES:
        raise ValueError("dtype must be np.float32 or np.float64")
    return values, output_dtype


def _decompose_rotation_axis_angle(rotation_axis_angle, twist_axis):
    """Split rotations into twist and swing quaternions in wxyz order."""
    rotations = np.asarray(rotation_axis_angle, dtype=np.float64)
    axis = np.asarray(twist_axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    quaternions = Rotation.from_rotvec(rotations).as_quat(scalar_first=True)
    twist = np.concatenate(
        (
            quaternions[:, :1],
            (quaternions[:, 1:] @ axis)[:, None] * axis,
        ),
        axis=1,
    )
    norms = np.linalg.norm(twist, axis=1, keepdims=True)
    degenerate = norms[:, 0] < 1e-12
    twist[~degenerate] /= norms[~degenerate]
    twist[degenerate] = np.array([1.0, 0.0, 0.0, 0.0])
    twist_inverse = twist * np.array([1.0, -1.0, -1.0, -1.0])
    swing = (
        Rotation.from_quat(twist_inverse, scalar_first=True)
        * Rotation.from_quat(quaternions, scalar_first=True)
    ).as_quat(scalar_first=True)
    return twist, swing


def _compute_wrist_angles_float64(body_pose):
    _, left_elbow_swing = _decompose_rotation_axis_angle(
        body_pose[:, 17], _TWIST_AXIS
    )
    _, right_elbow_swing = _decompose_rotation_axis_angle(
        body_pose[:, 18], _TWIST_AXIS
    )
    left_elbow_swing_xyz = Rotation.from_quat(
        left_elbow_swing[:, [1, 2, 3, 0]]
    ).as_euler("XYZ", degrees=False)
    right_elbow_swing_xyz = Rotation.from_quat(
        right_elbow_swing[:, [1, 2, 3, 0]]
    ).as_euler("XYZ", degrees=False)
    left_wrist_xyz = Rotation.from_rotvec(body_pose[:, 19]).as_euler(
        "XYZ", degrees=False
    )
    right_wrist_xyz = Rotation.from_rotvec(body_pose[:, 20]).as_euler(
        "XYZ", degrees=False
    )
    return np.column_stack(
        (
            left_elbow_swing_xyz[:, 0] + left_wrist_xyz[:, 0],
            left_wrist_xyz[:, 1],
            left_elbow_swing_xyz[:, 2] + left_wrist_xyz[:, 2],
            -(right_elbow_swing_xyz[:, 0] + right_wrist_xyz[:, 0]),
            -right_wrist_xyz[:, 1],
            right_elbow_swing_xyz[:, 2] + right_wrist_xyz[:, 2],
        )
    )


def compute_elf3_wrist_angles(smpl_body_pose, *, dtype=np.float32):
    """Return ELF3 left/right wrist XYZ angles with shape ``(N, 6)``."""
    body_pose, output_dtype = _validated_pose_and_dtype(
        smpl_body_pose, dtype
    )
    wrist_angles = _compute_wrist_angles_float64(body_pose)
    return np.ascontiguousarray(wrist_angles, dtype=output_dtype)


def build_elf3_joint_pos(smpl_body_pose, *, dtype=np.float32):
    """Return 29 ELF3 joint positions with only the wrist slots populated."""
    body_pose, output_dtype = _validated_pose_and_dtype(
        smpl_body_pose, dtype
    )
    wrist_angles = _compute_wrist_angles_float64(body_pose)
    joint_pos = np.zeros(
        (body_pose.shape[0], _ELF3_JOINT_COUNT), dtype=np.float64
    )
    joint_pos[:, [19, 20, 21, 26, 27, 28]] = wrist_angles
    return np.ascontiguousarray(joint_pos, dtype=output_dtype)


__all__ = ["build_elf3_joint_pos", "compute_elf3_wrist_angles"]
