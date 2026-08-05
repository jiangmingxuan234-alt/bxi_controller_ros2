"""Portable PICO-to-ELF3 GMR solver matching MoCapLab's Mink objective.

The original project uses Mink and DAQP.  This module keeps the same local
SE(3) frame tasks, cost convention, Levenberg-Marquardt damping, configuration
limits and warm-started two-stage solve, but implements the small box QP with
NumPy so the Mod remains portable across x86_64 and aarch64.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np


def quat_conjugate_wxyz(value: object) -> np.ndarray:
    q = np.asarray(value, dtype=np.float64).copy()
    q[1:] *= -1.0
    return q


def quat_multiply_wxyz(left: object, right: object) -> np.ndarray:
    w1, x1, y1, z1 = np.asarray(left, dtype=np.float64)
    w2, x2, y2, z2 = np.asarray(right, dtype=np.float64)
    return np.asarray(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dtype=np.float64,
    )


def normalize_quaternion_wxyz(value: object) -> np.ndarray:
    q = np.asarray(value, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("quaternion must contain four finite WXYZ values")
    norm = float(np.linalg.norm(q))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("quaternion must not have zero length")
    return q / norm


def rotate_vector_wxyz(quaternion: object, vector: object) -> np.ndarray:
    q = normalize_quaternion_wxyz(quaternion)
    v = np.asarray(vector, dtype=np.float64)
    return v + 2.0 * np.cross(q[1:], np.cross(q[1:], v) + q[0] * v)


def quaternion_rotation_vector_wxyz(quaternion: object) -> np.ndarray:
    q = normalize_quaternion_wxyz(quaternion)
    if q[0] < 0.0:
        q = -q
    vector_norm = float(np.linalg.norm(q[1:]))
    if vector_norm < 1.0e-10:
        return 2.0 * q[1:]
    angle = 2.0 * math.atan2(vector_norm, max(float(q[0]), 0.0))
    return q[1:] * (angle / vector_norm)


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.asarray(
        ((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)),
        dtype=np.float64,
    )


def _mink_rotation_vector_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """Match Mink's ``SO3.log()``, including its exact half-turn convention."""

    q = np.asarray(quaternion, dtype=np.float64).copy()
    q *= np.sign(q[0])
    vector = q[1:]
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm < 1.0e-10:
        return np.zeros(3, dtype=np.float64)
    return 2.0 * math.atan2(vector_norm, float(q[0])) * vector / vector_norm


def _se3_log(quaternion: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Return Mink-compatible ``SE3.log()`` in ``[v, omega]`` order."""

    omega = _mink_rotation_vector_wxyz(quaternion)
    theta = float(np.linalg.norm(omega))
    theta_squared = theta * theta
    skew_omega = _skew(omega)
    skew_omega_squared = skew_omega @ skew_omega
    if theta_squared < 1.0e-10:
        inverse_v = (
            np.eye(3, dtype=np.float64)
            - 0.5 * skew_omega
            + skew_omega_squared / 12.0
        )
    else:
        half_theta = 0.5 * theta
        inverse_v = (
            np.eye(3, dtype=np.float64)
            - 0.5 * skew_omega
            + (
                1.0
                - 0.5 * theta * math.cos(half_theta) / math.sin(half_theta)
            )
            / theta_squared
            * skew_omega_squared
        )
    return np.concatenate((inverse_v @ translation, omega))


def _so3_left_jacobian_inverse(rotation_vector: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rotation_vector))
    theta_squared = theta * theta
    if theta < 1.0e-10:
        beta = (1.0 / 12.0) * (
            1.0
            + theta_squared
            / 60.0
            * (1.0 + theta_squared / 42.0 * (1.0 + theta_squared / 40.0))
        )
    else:
        beta = (1.0 / theta_squared) * (
            1.0
            - theta
            * math.sin(theta)
            / (2.0 * (1.0 - math.cos(theta)))
        )
    skew_rotation = _skew(rotation_vector)
    return (
        np.eye(3, dtype=np.float64)
        - 0.5 * skew_rotation
        + beta * (skew_rotation @ skew_rotation)
    )


def _se3_q(tangent: np.ndarray) -> np.ndarray:
    omega = tangent[3:]
    theta = float(np.linalg.norm(omega))
    theta_squared = theta * theta
    if theta_squared < 1.0e-10:
        b = (1.0 / 6.0) + (1.0 / 120.0) * theta_squared
        c = -(1.0 / 24.0) + (1.0 / 720.0) * theta_squared
        d = -(1.0 / 60.0)
    else:
        theta_fourth = theta_squared * theta_squared
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)
        b = (theta - sin_theta) / (theta_squared * theta)
        c = (1.0 - 0.5 * theta_squared - cos_theta) / theta_fourth
        d = (
            2.0 * theta - 3.0 * sin_theta + theta * cos_theta
        ) / (2.0 * theta_fourth * theta)
    v = _skew(tangent[:3])
    w = _skew(omega)
    vw = v @ w
    wv = vw.T
    wvw = wv @ w
    vww = vw @ w
    return (
        0.5 * v
        + b * (wv + vw + wvw)
        - c * (vww - vww.T - 3.0 * wvw)
        + d * (wvw @ w + w @ wvw)
    )


def _se3_left_jacobian_inverse(tangent: np.ndarray) -> np.ndarray:
    """Port of ``mink.SE3.ljacinv`` used by ``FrameTask``."""

    if float(tangent[3:] @ tangent[3:]) < 1.0e-10:
        return np.eye(6, dtype=np.float64)
    rotation_inverse = _so3_left_jacobian_inverse(tangent[3:])
    result = np.zeros((6, 6), dtype=np.float64)
    result[:3, :3] = rotation_inverse
    result[:3, 3:] = (
        -rotation_inverse @ _se3_q(tangent) @ rotation_inverse
    )
    result[3:, 3:] = rotation_inverse
    return result


def _solve_linear(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(matrix, vector)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(matrix, vector, rcond=None)[0]


def _solve_box_qp(
    hessian: np.ndarray,
    linear: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Solve a strictly convex box QP with a compact active-set method."""

    unconstrained = _solve_linear(hessian, -linear)
    solution = np.clip(unconstrained, lower, upper)
    active_lower = unconstrained < lower
    active_upper = unconstrained > upper
    tolerance = 1.0e-10

    for _ in range(4 * solution.size + 16):
        free = ~(active_lower | active_upper)
        if np.any(free):
            active = ~free
            rhs = -linear[free]
            if np.any(active):
                rhs -= hessian[np.ix_(free, active)] @ solution[active]
            candidate = _solve_linear(hessian[np.ix_(free, free)], rhs)
            free_indices = np.flatnonzero(free)
            current = solution[free]
            direction = candidate - current
            alpha = 1.0
            hit_index = -1
            hit_lower = False
            for local_index, global_index in enumerate(free_indices):
                if direction[local_index] > tolerance:
                    distance = (
                        upper[global_index] - current[local_index]
                    ) / direction[local_index]
                    if distance < alpha:
                        alpha = max(0.0, float(distance))
                        hit_index = int(global_index)
                        hit_lower = False
                elif direction[local_index] < -tolerance:
                    distance = (
                        lower[global_index] - current[local_index]
                    ) / direction[local_index]
                    if distance < alpha:
                        alpha = max(0.0, float(distance))
                        hit_index = int(global_index)
                        hit_lower = True
            solution[free] = current + alpha * direction
            if hit_index >= 0 and alpha < 1.0 - tolerance:
                if hit_lower:
                    solution[hit_index] = lower[hit_index]
                    active_lower[hit_index] = True
                else:
                    solution[hit_index] = upper[hit_index]
                    active_upper[hit_index] = True
                continue

        gradient = hessian @ solution + linear
        lower_violation = np.where(active_lower, -gradient, -np.inf)
        upper_violation = np.where(active_upper, gradient, -np.inf)
        lower_index = int(np.argmax(lower_violation))
        upper_index = int(np.argmax(upper_violation))
        if max(lower_violation[lower_index], upper_violation[upper_index]) <= tolerance:
            return solution
        if lower_violation[lower_index] >= upper_violation[upper_index]:
            active_lower[lower_index] = False
        else:
            active_upper[upper_index] = False

    raise RuntimeError("box-constrained GMR QP did not converge")


def unity_pose_to_right_handed(position: object, quaternion_wxyz: object) -> tuple[np.ndarray, np.ndarray]:
    """Match MoCapLab's Unity conversion: ``(x,y,z)->(x,-z,y)``."""

    pos = np.asarray(position, dtype=np.float64)
    if pos.shape != (3,) or not np.isfinite(pos).all():
        raise ValueError("Unity position must contain three finite values")
    source_q = normalize_quaternion_wxyz(quaternion_wxyz)
    basis_rotation = np.asarray((math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0))
    converted_q = normalize_quaternion_wxyz(
        quat_multiply_wxyz(basis_rotation, source_q)
    )
    return np.asarray((pos[0], -pos[2], pos[1])), converted_q


class PicoGmrRetargeter:
    """Warm-started two-stage weighted IK retargeter for the current ELF3 XML."""

    def __init__(
        self,
        model_xml: str | Path,
        config_json: str | Path,
        joint_names: tuple[str, ...],
        *,
        actual_human_height: float = 1.75,
        damping: float = 0.5,
        max_iterations: int = 10,
    ) -> None:
        if not math.isfinite(actual_human_height) or actual_human_height <= 0.0:
            raise ValueError("actual_human_height must be positive and finite")
        if not math.isfinite(damping) or damping <= 0.0:
            raise ValueError("damping must be positive and finite")
        self.model = mujoco.MjModel.from_xml_path(str(Path(model_xml).resolve()))
        self.data = mujoco.MjData(self.model)
        if self.model.nkey:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        with Path(config_json).open("r", encoding="utf-8") as stream:
            config = json.load(stream)
        self.human_root_name = str(config["human_root_name"])
        assumption = float(config["human_height_assumption"])
        ratio = actual_human_height / assumption
        self.human_scale = {
            str(name): float(scale) * ratio
            for name, scale in config["human_scale_table"].items()
        }
        self._task_tables = tuple(
            self._compile_tasks(config[name])
            for name in ("ik_match_table1", "ik_match_table2")
            if bool(config.get("use_" + name, True))
        )
        if not self._task_tables:
            raise ValueError("GMR configuration has no enabled IK task table")

        self.joint_names = tuple(joint_names)
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("GMR output joint names contain duplicates")
        self._joint_ids = np.asarray(
            [self._required_id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.joint_names],
            dtype=np.int32,
        )
        self._qpos_addresses = self.model.jnt_qposadr[self._joint_ids].astype(np.intp)
        raw_warm_start = config.get("warm_start_joint_positions")
        if raw_warm_start is None:
            self._warm_start_qpos = np.zeros(len(self.joint_names), dtype=np.float64)
        else:
            if not isinstance(raw_warm_start, dict):
                raise ValueError("warm_start_joint_positions must be an object")
            missing_warm_start = set(self.joint_names) - set(raw_warm_start)
            unknown_warm_start = set(raw_warm_start) - set(self.joint_names)
            if missing_warm_start or unknown_warm_start:
                details = []
                if missing_warm_start:
                    details.append("missing " + ", ".join(sorted(missing_warm_start)))
                if unknown_warm_start:
                    details.append("unknown " + ", ".join(sorted(unknown_warm_start)))
                raise ValueError("invalid warm_start_joint_positions: " + "; ".join(details))
            self._warm_start_qpos = np.asarray(
                [float(raw_warm_start[name]) for name in self.joint_names],
                dtype=np.float64,
            )
            if not np.isfinite(self._warm_start_qpos).all():
                raise ValueError("warm_start_joint_positions must contain finite values")
        for output_index, joint_id in enumerate(self._joint_ids):
            if self.model.jnt_limited[joint_id]:
                low, high = self.model.jnt_range[joint_id]
                value = self._warm_start_qpos[output_index]
                if value < low or value > high:
                    raise ValueError(
                        f"warm start for {self.joint_names[output_index]!r} is outside "
                        f"the model range [{low}, {high}]: {value}"
                    )
        joint_dofs = self.model.jnt_dofadr[self._joint_ids].astype(np.intp)
        root_id = self._required_id(mujoco.mjtObj.mjOBJ_JOINT, "world_joint")
        if self.model.jnt_type[root_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("ELF3 world_joint must be a free joint")
        root_dof = int(self.model.jnt_dofadr[root_id])
        self._solve_dofs = np.concatenate(
            (np.arange(root_dof, root_dof + 6, dtype=np.intp), joint_dofs)
        )
        self._root_qpos_address = int(self.model.jnt_qposadr[root_id])
        self.damping = float(damping)
        self.max_iterations = max(1, int(max_iterations))
        self._jac_pos = np.zeros((3, self.model.nv), dtype=np.float64)
        self._jac_rot = np.zeros((3, self.model.nv), dtype=np.float64)
        self._velocity = np.zeros(self.model.nv, dtype=np.float64)
        self._identity = np.eye(len(self._solve_dofs), dtype=np.float64)
        self._delta_lower = np.full(len(self._solve_dofs), -np.inf, dtype=np.float64)
        self._delta_upper = np.full(len(self._solve_dofs), np.inf, dtype=np.float64)
        self.reset()

    def reset(self) -> None:
        """Reset the IK warm start to the ELF3 keyframe."""

        if self.model.nkey:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._qpos_addresses] = self._warm_start_qpos
        mujoco.mj_forward(self.model, self.data)

    def _required_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        result = int(mujoco.mj_name2id(self.model, object_type, name))
        if result < 0:
            raise ValueError(f"ELF3 model is missing required name {name!r}")
        return result

    def _compile_tasks(self, raw_table: dict[str, object]) -> tuple[tuple[object, ...], ...]:
        tasks = []
        for robot_body, raw in raw_table.items():
            human_body, position_weight, rotation_weight, position_offset, rotation_offset = raw
            position_weight = float(position_weight)
            rotation_weight = float(rotation_weight)
            if position_weight == 0.0 and rotation_weight == 0.0:
                continue
            tasks.append(
                (
                    self._required_id(mujoco.mjtObj.mjOBJ_BODY, str(robot_body)),
                    str(human_body),
                    position_weight,
                    rotation_weight,
                    np.asarray(position_offset, dtype=np.float64),
                    normalize_quaternion_wxyz(rotation_offset),
                )
            )
        return tuple(tasks)

    def _scaled_human_pose(
        self,
        human_data: dict[str, tuple[np.ndarray, np.ndarray]],
        name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.human_root_name not in human_data:
            raise ValueError(f"PICO frame is missing {self.human_root_name!r}")
        if name not in human_data:
            raise ValueError(f"PICO frame is missing {name!r}")
        root_pos = human_data[self.human_root_name][0]
        pos, quat = human_data[name]
        scaled_root = root_pos * self.human_scale[self.human_root_name]
        if name == self.human_root_name:
            scaled_pos = scaled_root
        else:
            scaled_pos = (pos - root_pos) * self.human_scale[name] + scaled_root
        return scaled_pos, quat

    def _targets(
        self,
        human_data: dict[str, tuple[np.ndarray, np.ndarray]],
        tasks: tuple[tuple[object, ...], ...],
    ) -> tuple[tuple[int, float, float, np.ndarray, np.ndarray], ...]:
        result = []
        for body_id, human_name, pos_weight, rot_weight, pos_offset, rot_offset in tasks:
            position, quaternion = self._scaled_human_pose(human_data, human_name)
            target_quaternion = normalize_quaternion_wxyz(
                quat_multiply_wxyz(quaternion, rot_offset)
            )
            target_position = position + rotate_vector_wxyz(target_quaternion, pos_offset)
            result.append(
                (body_id, pos_weight, rot_weight, target_position, target_quaternion)
            )
        return tuple(result)

    def _qp_objective(
        self,
        targets: tuple[tuple[int, float, float, np.ndarray, np.ndarray], ...],
    ) -> tuple[np.ndarray, np.ndarray, float]:
        dimension = len(self._solve_dofs)
        hessian = self.damping * self._identity.copy()
        linear = np.zeros(dimension, dtype=np.float64)
        squared_error = 0.0
        for body_id, pos_weight, rot_weight, target_pos, target_quat in targets:
            self._jac_pos.fill(0.0)
            self._jac_rot.fill(0.0)
            mujoco.mj_jacBody(
                self.model,
                self.data,
                self._jac_pos,
                self._jac_rot,
                body_id,
            )
            current_rotation = self.data.xmat[body_id].reshape(3, 3)
            current_quaternion = np.empty(4, dtype=np.float64)
            mujoco.mju_mat2Quat(
                current_quaternion,
                current_rotation.reshape(-1),
            )
            rotation_body_to_target = quat_multiply_wxyz(
                quat_conjugate_wxyz(current_quaternion),
                target_quat,
            )
            translation_body_to_target = current_rotation.T @ (
                target_pos - self.data.xpos[body_id]
            )
            error = _se3_log(
                rotation_body_to_target,
                translation_body_to_target,
            )

            body_jacobian = np.vstack((self._jac_pos, self._jac_rot))
            body_jacobian[:3] = current_rotation.T @ body_jacobian[:3]
            body_jacobian[3:] = current_rotation.T @ body_jacobian[3:]
            error_jacobian = (
                -_se3_left_jacobian_inverse(error)
                @ body_jacobian[:, self._solve_dofs]
            )
            costs = np.asarray(
                (
                    pos_weight,
                    pos_weight,
                    pos_weight,
                    rot_weight,
                    rot_weight,
                    rot_weight,
                ),
                dtype=np.float64,
            )
            weighted_jacobian = costs[:, None] * error_jacobian
            weighted_feedback = -costs * error
            lm_damping = float(weighted_feedback @ weighted_feedback)
            hessian += weighted_jacobian.T @ weighted_jacobian
            hessian.flat[:: dimension + 1] += lm_damping
            linear -= weighted_feedback @ weighted_jacobian
            squared_error += float(error @ error)
        return hessian, linear, math.sqrt(squared_error)

    def _configuration_delta_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        self._delta_lower.fill(-np.inf)
        self._delta_upper.fill(np.inf)
        for output_index, (joint_id, qpos_address) in enumerate(
            zip(self._joint_ids, self._qpos_addresses)
        ):
            if self.model.jnt_limited[joint_id]:
                low, high = self.model.jnt_range[joint_id]
                current = self.data.qpos[qpos_address]
                self._delta_lower[6 + output_index] = 0.95 * (low - current)
                self._delta_upper[6 + output_index] = 0.95 * (high - current)
        return self._delta_lower, self._delta_upper

    def _clip_joint_ranges(self) -> None:
        for joint_id, qpos_address in zip(self._joint_ids, self._qpos_addresses):
            if self.model.jnt_limited[joint_id]:
                low, high = self.model.jnt_range[joint_id]
                self.data.qpos[qpos_address] = np.clip(
                    self.data.qpos[qpos_address], low, high
                )

    def _solve_phase(
        self,
        targets: tuple[tuple[int, float, float, np.ndarray, np.ndarray], ...],
    ) -> None:
        previous_error = math.inf
        for _ in range(self.max_iterations + 1):
            mujoco.mj_forward(self.model, self.data)
            hessian, linear, current_error = self._qp_objective(targets)
            if previous_error - current_error <= 1.0e-3 and previous_error < math.inf:
                break
            previous_error = current_error
            lower, upper = self._configuration_delta_bounds()
            delta = _solve_box_qp(hessian, linear, lower, upper)
            self._velocity.fill(0.0)
            self._velocity[self._solve_dofs] = delta
            mujoco.mj_integratePos(self.model, self.data.qpos, self._velocity, 1.0)
            self._clip_joint_ranges()

    def retarget(
        self,
        human_data: dict[str, tuple[np.ndarray, np.ndarray]],
        *,
        dt: float = 0.02,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dt = float(dt)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("GMR dt must be positive and finite")
        normalized: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, pose in human_data.items():
            position = np.asarray(pose[0], dtype=np.float64)
            quaternion = normalize_quaternion_wxyz(pose[1])
            if position.shape != (3,) or not np.isfinite(position).all():
                raise ValueError(f"PICO body {name!r} has an invalid position")
            normalized[name] = (position, quaternion)
        for tasks in self._task_tables:
            self._solve_phase(self._targets(normalized, tasks))
        mujoco.mj_forward(self.model, self.data)
        root = self._root_qpos_address
        return (
            self.data.qpos[self._qpos_addresses].astype(np.float32, copy=True),
            self.data.qpos[root : root + 3].astype(np.float32, copy=True),
            normalize_quaternion_wxyz(
                self.data.qpos[root + 3 : root + 7]
            ).astype(np.float32),
        )


__all__ = [
    "PicoGmrRetargeter",
    "quat_conjugate_wxyz",
    "quat_multiply_wxyz",
    "quaternion_rotation_vector_wxyz",
    "unity_pose_to_right_handed",
]
