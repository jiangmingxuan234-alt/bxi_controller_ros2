"""Private RGMT policy implementation for ``com.bxi.pico_gmr_motion``.

The ONNX graph is intentionally motion agnostic: it maps one flat RGMT
observation to one residual joint action.  This module owns reference-motion
loading/windowing, proprioceptive history, and the reference-residual action
conversion.  It therefore supports both an offline NPZ and a live 21-frame
reference window without changing or re-exporting the policy graph.

Quaternion inputs use WXYZ order.  Robot angular velocity must already be in
the robot base frame, matching Isaac Lab's ``base_ang_vel`` observation.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

from bxi_example_py_elf3.framework.inference import (
    InferenceFrame,
    InferenceRuntime,
    JointPolicy,
    ModelSpec,
    PolicyJointContract,
    PolicyOutput,
    default_runtime,
)
from bxi_example_py_elf3.framework.inference.history import HistoryBuffer
from bxi_example_py_elf3.framework.joints import JointLayout, JointParameterSet
from bxi_example_py_elf3.policies.joints import (
    ELF3_ISAAC_JOINTS,
    ELF3_ISAAC_PARAMETERS,
)


RGMT_NUM_JOINTS = 29
RGMT_COMMAND_WINDOW_OFFSETS = np.arange(-10, 11, dtype=np.int64)
RGMT_COMMAND_WINDOW_SIZE = 21
RGMT_COMMAND_TOKEN_DIM = 38
RGMT_PROPRIO_TOKEN_DIM = 93
RGMT_PROPRIO_HISTORY_LENGTH = 10
RGMT_OBSERVATION_DIM = 1734


def _as_finite_array(name: str, value, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return array


def _parse_csv_floats(metadata: dict[str, str], key: str, length: int) -> np.ndarray:
    if key not in metadata:
        raise KeyError(f"ONNX metadata is missing required key {key!r}.")
    try:
        values = np.asarray(
            [float(item) for item in metadata[key].split(",") if item.strip()],
            dtype=np.float32,
        )
    except ValueError as error:
        raise ValueError(f"ONNX metadata {key!r} is not a numeric CSV vector.") from error
    if values.shape != (length,) or not np.isfinite(values).all():
        raise ValueError(
            f"ONNX metadata {key!r} must contain {length} finite values, got {values.shape}."
        )
    return values


def _parse_csv_ints(metadata: dict[str, str], key: str) -> np.ndarray:
    if key not in metadata:
        raise KeyError(f"ONNX metadata is missing required key {key!r}.")
    try:
        values = np.asarray(
            [int(float(item)) for item in metadata[key].split(",") if item.strip()],
            dtype=np.int64,
        )
    except ValueError as error:
        raise ValueError(f"ONNX metadata {key!r} is not an integer CSV vector.") from error
    return values


def _normalize_quaternion_wxyz(name: str, value, expected_shape: tuple[int, ...]) -> np.ndarray:
    quaternion = _as_finite_array(name, value, expected_shape)
    norms = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norms <= np.finfo(np.float32).eps):
        raise ValueError(f"{name} contains a zero-length quaternion.")
    return quaternion / norms


def quaternion_conjugate_wxyz(quaternion: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternion, dtype=np.float32).copy()
    result[..., 1:] *= -1.0
    return result


def quaternion_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    w1, x1, y1, z1 = np.moveaxis(left, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        axis=-1,
    ).astype(np.float32, copy=False)


def yaw_quaternion_wxyz(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float32)
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    zeros = np.zeros_like(yaw)
    return np.stack((np.cos(0.5 * yaw), zeros, zeros, np.sin(0.5 * yaw)), axis=-1).astype(
        np.float32, copy=False
    )


def quaternion_to_rotation_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float32)
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norm <= np.finfo(np.float32).eps):
        raise ValueError("Cannot convert a zero-length quaternion to a rotation matrix.")
    q = quaternion / norm
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3)).astype(np.float32, copy=False)


def quat_rotate_inverse_wxyz(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate world-frame vectors into the WXYZ quaternion's local frame."""

    quaternion = np.asarray(quaternion, dtype=np.float32)
    vector = np.asarray(vector, dtype=np.float32)
    if (
        quaternion.shape[:-1] != vector.shape[:-1]
        or quaternion.shape[-1:] != (4,)
        or vector.shape[-1:] != (3,)
    ):
        raise ValueError(
            "quaternion/vector shapes must be [...,4] and [...,3] with "
            "identical leading dimensions, "
            f"got {quaternion.shape} and {vector.shape}."
        )
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norm <= np.finfo(np.float32).eps):
        raise ValueError("quaternion contains a zero-length value.")
    unit = quaternion / norm
    scalar = unit[..., :1]
    xyz = unit[..., 1:]
    cross = 2.0 * np.cross(xyz, vector)
    return (vector - scalar * cross + np.cross(xyz, cross)).astype(np.float32, copy=False)


def build_rgmt_command_window(
    reference_anchor_lin_vel_w,
    reference_anchor_ang_vel_w,
    reference_anchor_quat_w,
    reference_joint_pos,
) -> np.ndarray:
    """Build a clean 21x38 command window with the training-time field order."""

    lin_vel = _as_finite_array(
        "reference_anchor_lin_vel_w", reference_anchor_lin_vel_w, (RGMT_COMMAND_WINDOW_SIZE, 3)
    )
    ang_vel = _as_finite_array(
        "reference_anchor_ang_vel_w", reference_anchor_ang_vel_w, (RGMT_COMMAND_WINDOW_SIZE, 3)
    )
    quaternion = _normalize_quaternion_wxyz(
        "reference_anchor_quat_w", reference_anchor_quat_w, (RGMT_COMMAND_WINDOW_SIZE, 4)
    )
    joint_pos = _as_finite_array(
        "reference_joint_pos", reference_joint_pos, (RGMT_COMMAND_WINDOW_SIZE, RGMT_NUM_JOINTS)
    )
    gravity_w = np.broadcast_to(
        np.asarray((0.0, 0.0, -1.0), dtype=np.float32),
        (RGMT_COMMAND_WINDOW_SIZE, 3),
    )
    tokens = np.concatenate(
        (
            quat_rotate_inverse_wxyz(quaternion, lin_vel),
            quat_rotate_inverse_wxyz(quaternion, ang_vel),
            quat_rotate_inverse_wxyz(quaternion, gravity_w),
            joint_pos,
        ),
        axis=-1,
    ).astype(np.float32, copy=False)
    if tokens.shape != (RGMT_COMMAND_WINDOW_SIZE, RGMT_COMMAND_TOKEN_DIM):
        raise RuntimeError(f"Internal command layout error: got {tokens.shape}.")
    return tokens


class RgmtExternalReferencePolicy(JointPolicy):
    """RGMT policy using named joints and the shared inference runtime.

    Robot state may arrive in any framework ``JointLayout``; ``JointPolicy``
    maps it into :data:`ELF3_ISAAC_JOINTS` by name.  Model inputs, model
    outputs, policy parameters and reference joint arrays all use that one
    declared layout.  There are no public Isaac/MuJoCo index arrays.

    An NPZ without ``joint_names`` is interpreted as ``motion_joint_layout``.
    New NPZ files should include ``joint_names`` so a wrong order is rejected
    instead of silently producing a wrong command.
    """

    joint_contract = PolicyJointContract(
        observation=ELF3_ISAAC_JOINTS,
        action=ELF3_ISAAC_JOINTS,
    )

    def __init__(
        self,
        motion_npz_path: str | None,
        model_onnx_path: str | ModelSpec,
        start_frame: int = 0,
        *,
        reference_yaw_mode: str = "initial",
        anchor_body_index: int | None = None,
        motion_joint_layout: JointLayout = ELF3_ISAAC_JOINTS,
        calibration_offset: object | None = None,
        runtime: InferenceRuntime | None = None,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        if reference_yaw_mode not in {"none", "initial", "continuous"}:
            raise ValueError(
                "reference_yaw_mode must be 'none', 'initial', or 'continuous', "
                f"got {reference_yaw_mode!r}"
            )
        if not isinstance(motion_joint_layout, JointLayout):
            raise TypeError("motion_joint_layout must be a JointLayout")

        self.motion_npz_path = (
            None if motion_npz_path is None else str(motion_npz_path)
        )
        self.reference_yaw_mode = reference_yaw_mode
        self._explicit_anchor_body_index = anchor_body_index
        self._motion_joint_layout = motion_joint_layout
        self._runtime = runtime or default_runtime()
        self._policy_name = "rgmt"

        spec = (
            model_onnx_path
            if isinstance(model_onnx_path, ModelSpec)
            else ModelSpec.portable_onnx(
                model_onnx_path,
                input_names=("obs",),
                output_names=("actions",),
            )
        )
        self._backend = self._runtime.open_backend(spec, backend=backend)
        self.metadata = dict(self._backend.metadata)
        self._load_metadata_contract(calibration_offset)
        self._validate_model_io()

        self.num_obs = RGMT_OBSERVATION_DIM
        self.num_actions = RGMT_NUM_JOINTS
        self._obs = np.zeros((1, RGMT_OBSERVATION_DIM), dtype=np.float32)
        self.obs = self._obs[0]
        self._inputs = {"obs": self._obs}
        self.raw_action = np.zeros(RGMT_NUM_JOINTS, dtype=np.float32)
        self.action_buffer = np.zeros(RGMT_NUM_JOINTS, dtype=np.float32)
        self.action = self.action_buffer.copy()
        self._clipped_action = np.empty(RGMT_NUM_JOINTS, dtype=np.float32)
        self._scaled_action = np.empty(RGMT_NUM_JOINTS, dtype=np.float32)
        self._reference_center = np.empty(RGMT_NUM_JOINTS, dtype=np.float32)
        self._target = self._target_buffer.position
        self._proprio = np.empty(RGMT_PROPRIO_TOKEN_DIM, dtype=np.float32)
        self._proprio_history = HistoryBuffer(
            RGMT_PROPRIO_HISTORY_LENGTH,
            RGMT_PROPRIO_TOKEN_DIM,
            dtype=np.float32,
        )
        self._reference_yaw_delta: np.ndarray | None = None
        self._reference_stream = deque(maxlen=RGMT_COMMAND_WINDOW_SIZE)

        self.motioninputpos: np.ndarray | None = None
        self.motionquat: np.ndarray | None = None
        self.motionpos: np.ndarray | None = None
        self.motion_anchor_quat_w: np.ndarray | None = None
        self.motion_anchor_lin_vel_w: np.ndarray | None = None
        self.motion_anchor_ang_vel_w: np.ndarray | None = None
        self.motion_fps: float | None = None
        self._start_frame = 0
        self._end_frame = -1
        self._frame = 0.0

        np.copyto(self._target, self._parameters.default_position)
        self.publish_output(
            self._target,
            self._parameters.kp,
            self._parameters.kd,
        )
        self._backend.warmup(self._inputs, self._runtime.options.warmup_runs)

        if self.motion_npz_path is not None:
            self.load_motion_npz(
                self.motion_npz_path,
                start_frame=start_frame,
                joint_layout=motion_joint_layout,
            )
        elif start_frame != 0:
            raise ValueError("start_frame requires an NPZ motion source")

    @classmethod
    def for_live_reference(
        cls,
        model_onnx_path: str | ModelSpec,
        **kwargs,
    ) -> "RgmtExternalReferencePolicy":
        return cls(None, model_onnx_path, **kwargs)

    def _load_metadata_contract(self, calibration_offset: object | None) -> None:
        metadata = self.metadata
        if metadata.get("export_format") != "external_reference_actor":
            raise ValueError(
                "RGMT requires an actor-only external-reference model; "
                "re-export with play.py --export_actor_onnx"
            )
        if metadata.get("quaternion_convention") != "wxyz":
            raise ValueError("RGMT requires quaternion_convention=wxyz")
        if metadata.get("history_order") != "oldest_to_newest":
            raise ValueError("RGMT requires history_order=oldest_to_newest")
        if metadata.get("action_semantics") != "reference_joint_position_residual":
            raise ValueError("RGMT model actions are not reference-joint residuals")
        if int(metadata.get("policy_observation_dim", -1)) != RGMT_OBSERVATION_DIM:
            raise ValueError("RGMT model observation dimension must be 1734")
        if int(metadata.get("policy_action_dim", -1)) != RGMT_NUM_JOINTS:
            raise ValueError("RGMT model action dimension must be 29")

        observation_names = tuple(
            item.strip() for item in metadata.get("observation_names", "").split(",")
        )
        expected_observations = (
            "rgmt_command",
            "motion_anchor_ori_b",
            "rgmt_proprio",
        )
        if observation_names != expected_observations:
            raise ValueError(f"unexpected RGMT observation order: {observation_names}")
        history_lengths = _parse_csv_ints(
            metadata,
            "observation_history_lengths",
        )
        if not np.array_equal(history_lengths, np.asarray((1, 1, 10))):
            raise ValueError(
                f"unexpected RGMT history lengths: {history_lengths.tolist()}"
            )
        command_offsets = _parse_csv_ints(metadata, "command_window_offsets")
        if not np.array_equal(command_offsets, RGMT_COMMAND_WINDOW_OFFSETS):
            raise ValueError(
                f"unexpected RGMT command offsets: {command_offsets.tolist()}"
            )

        model_joint_names = tuple(
            item.strip() for item in metadata.get("joint_names", "").split(",")
            if item.strip()
        )
        if model_joint_names != self.joint_contract.observation.names:
            raise ValueError(
                "RGMT model joint_names does not match the declared "
                f"'{self.joint_contract.observation.label}' layout"
            )
        self.joint_name = model_joint_names

        default_position = _parse_csv_floats(
            metadata,
            "policy_default_joint_pos"
            if "policy_default_joint_pos" in metadata
            else "default_joint_pos",
            RGMT_NUM_JOINTS,
        )
        action_scale = _parse_csv_floats(
            metadata,
            "policy_action_scale"
            if "policy_action_scale" in metadata
            else "action_scale",
            RGMT_NUM_JOINTS,
        )
        metadata_offset = _parse_csv_floats(
            metadata,
            "reference_action_calibration_offset",
            RGMT_NUM_JOINTS,
        )
        self.calibration_offset = (
            metadata_offset
            if calibration_offset is None
            else _as_finite_array(
                "calibration_offset",
                calibration_offset,
                (RGMT_NUM_JOINTS,),
            ).copy()
        )
        actual_default = default_position + self.calibration_offset
        self._parameters = JointParameterSet.from_arrays(
            self.joint_contract.action,
            default_position=actual_default,
            kp=ELF3_ISAAC_PARAMETERS.kp,
            kd=ELF3_ISAAC_PARAMETERS.kd,
            action_scale=action_scale,
        )

        action_clip_text = metadata.get("policy_action_clip")
        if action_clip_text is None:
            raise KeyError("RGMT metadata is missing policy_action_clip")
        if action_clip_text == "none":
            self.action_clip: float | None = None
        else:
            self.action_clip = float(action_clip_text)
            if not np.isfinite(self.action_clip) or self.action_clip <= 0.0:
                raise ValueError("policy_action_clip must be positive or 'none'")

        self.anchor_body_name = metadata.get("anchor_body_name", "torso_link")
        model_anchor_index = int(metadata.get("motion_anchor_body_index_full", -1))
        if model_anchor_index < 0:
            raise ValueError("RGMT metadata has no valid anchor body index")
        if self._explicit_anchor_body_index is not None:
            anchor = self._explicit_anchor_body_index
            if isinstance(anchor, bool) or not isinstance(anchor, int):
                raise TypeError("anchor_body_index must be an integer")
            if anchor != model_anchor_index:
                raise ValueError(
                    f"anchor_body_index={anchor} conflicts with model index "
                    f"{model_anchor_index}"
                )
        self.anchor_body_index = (
            model_anchor_index
            if self._explicit_anchor_body_index is None
            else self._explicit_anchor_body_index
        )
        self.policy_fps = float(metadata["policy_fps"])
        self.policy_control_dt = float(metadata["policy_control_dt"])
        if not np.isfinite(self.policy_fps) or self.policy_fps <= 0.0:
            raise ValueError("policy_fps must be positive and finite")
        if not np.isfinite(self.policy_control_dt) or self.policy_control_dt <= 0.0:
            raise ValueError("policy_control_dt must be positive and finite")

    def _validate_model_io(self) -> None:
        input_shape = tuple(self._backend.input_shape("obs"))
        output_shape = tuple(self._backend.output_shape("actions"))
        if len(input_shape) != 2 or input_shape[-1] != RGMT_OBSERVATION_DIM:
            raise ValueError(f"RGMT obs shape must be [batch,1734], got {input_shape}")
        if len(output_shape) != 2 or output_shape[-1] != RGMT_NUM_JOINTS:
            raise ValueError(f"RGMT actions shape must be [batch,29], got {output_shape}")

    @staticmethod
    def _decode_names(values: np.ndarray) -> tuple[str, ...]:
        return tuple(
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in np.asarray(values).reshape(-1)
        )

    def _map_reference_joints(
        self,
        values: object,
        source_layout: JointLayout,
        *,
        name: str,
    ) -> np.ndarray:
        if not isinstance(source_layout, JointLayout):
            raise TypeError(f"{name} source layout must be a JointLayout")
        array = np.asarray(values, dtype=np.float32)
        if array.ndim not in (1, 2) or array.shape[-1] != source_layout.dof_num:
            raise ValueError(
                f"{name} must end with {source_layout.dof_num} joints for "
                f"'{source_layout.label}', got {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or infinite values")
        missing = self.joint_contract.observation.missing_from(source_layout)
        if missing:
            raise ValueError(f"{name} layout is missing RGMT joints: {missing}")
        if source_layout.names == self.joint_contract.observation.names:
            return np.ascontiguousarray(array, dtype=np.float32)
        indices = tuple(
            source_layout.index(joint_name)
            for joint_name in self.joint_contract.observation.names
        )
        return np.ascontiguousarray(array[..., indices], dtype=np.float32)

    def load_motion_npz(
        self,
        motion_npz_path: str,
        *,
        start_frame: int = 0,
        joint_layout: JointLayout | None = None,
    ) -> None:
        """Load an offline reference and map its joints by declared names."""

        path = Path(motion_npz_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"motion NPZ does not exist: {path}")
        required = (
            "fps",
            "joint_pos",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        )
        with np.load(path, allow_pickle=False) as data:
            missing = tuple(name for name in required if name not in data)
            if missing:
                raise KeyError(f"motion NPZ is missing fields: {missing}")
            fps_values = np.asarray(data["fps"]).reshape(-1)
            if fps_values.size != 1:
                raise ValueError("motion fps must contain exactly one value")
            motion_fps = float(fps_values[0])
            raw_joint_pos = np.asarray(data["joint_pos"], dtype=np.float32).copy()
            body_pos = np.asarray(data["body_pos_w"], dtype=np.float32).copy()
            body_quat = np.asarray(data["body_quat_w"], dtype=np.float32).copy()
            body_lin_vel = np.asarray(
                data["body_lin_vel_w"], dtype=np.float32
            ).copy()
            body_ang_vel = np.asarray(
                data["body_ang_vel_w"], dtype=np.float32
            ).copy()
            body_names = (
                self._decode_names(data["body_names"])
                if "body_names" in data
                else None
            )
            stored_joint_names = (
                self._decode_names(data["joint_names"])
                if "joint_names" in data
                else None
            )

        source_layout = joint_layout or self._motion_joint_layout
        if stored_joint_names is not None:
            stored_layout = JointLayout(
                stored_joint_names,
                label=f"{path.name} reference joints",
            )
            # Self-describing data wins over the fallback used for legacy NPZ
            # files.  This keeps the boundary name-based even when a caller's
            # default layout differs from the recorded order.
            source_layout = stored_layout
        joint_pos = self._map_reference_joints(
            raw_joint_pos,
            source_layout,
            name="joint_pos",
        )
        if joint_pos.ndim != 2:
            raise ValueError(
                f"joint_pos must have shape [frames,29], got {joint_pos.shape}"
            )

        if not np.isclose(motion_fps, self.policy_fps, rtol=0.0, atol=1.0e-6):
            raise ValueError(
                f"motion fps {motion_fps:g} does not match policy fps "
                f"{self.policy_fps:g}; resample first"
            )
        frame_count = joint_pos.shape[0]
        body_count = body_quat.shape[1] if body_quat.ndim == 3 else -1
        expected_shapes = {
            "body_pos_w": (frame_count, body_count, 3),
            "body_quat_w": (frame_count, body_count, 4),
            "body_lin_vel_w": (frame_count, body_count, 3),
            "body_ang_vel_w": (frame_count, body_count, 3),
        }
        for name, array in (
            ("body_pos_w", body_pos),
            ("body_quat_w", body_quat),
            ("body_lin_vel_w", body_lin_vel),
            ("body_ang_vel_w", body_ang_vel),
        ):
            if array.shape != expected_shapes[name]:
                raise ValueError(
                    f"{name} must have shape {expected_shapes[name]}, got {array.shape}"
                )
            if not np.isfinite(array).all():
                raise ValueError(f"{name} contains NaN or infinite values")
        if frame_count < 1 or body_count < 1:
            raise ValueError("motion NPZ must contain at least one frame and body")
        quaternion_norms = np.linalg.norm(body_quat, axis=-1)
        if np.any(quaternion_norms <= np.finfo(np.float32).eps):
            raise ValueError("body_quat_w contains a zero-length quaternion")
        if np.max(np.abs(quaternion_norms - 1.0)) > 1.0e-3:
            raise ValueError("body_quat_w is not normalized within tolerance 1e-3")

        if body_names is not None:
            if len(body_names) != body_count:
                raise ValueError("body_names length does not match body arrays")
            if self.anchor_body_name not in body_names:
                raise ValueError(
                    f"motion body_names does not contain {self.anchor_body_name!r}"
                )
            if body_names.index(self.anchor_body_name) != self.anchor_body_index:
                raise ValueError(
                    f"motion anchor {self.anchor_body_name!r} does not match "
                    f"model body index {self.anchor_body_index}"
                )
        if not 0 <= self.anchor_body_index < body_count:
            raise IndexError(
                f"anchor body index {self.anchor_body_index} is outside "
                f"[0,{body_count - 1}]"
            )
        if isinstance(start_frame, bool) or not isinstance(start_frame, int):
            raise TypeError("start_frame must be an integer")
        if not 0 <= start_frame < frame_count:
            raise IndexError(
                f"start_frame must be inside [0,{frame_count - 1}], got {start_frame}"
            )

        self.motion_npz_path = str(path)
        self.motioninputpos = joint_pos
        self.motionpos = body_pos
        self.motionquat = body_quat
        self.motion_anchor_quat_w = body_quat[:, self.anchor_body_index]
        self.motion_anchor_lin_vel_w = body_lin_vel[:, self.anchor_body_index]
        self.motion_anchor_ang_vel_w = body_ang_vel[:, self.anchor_body_index]
        self.motion_fps = motion_fps
        self._start_frame = int(start_frame)
        self._end_frame = frame_count - 1
        self._frame = float(self._start_frame)
        self._clear_recurrent_state(clear_reference_stream=True)

    @property
    def timestep(self) -> int:
        return int(self._frame)

    @timestep.setter
    def timestep(self, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError("timestep must be an integer")
        value = int(value)
        if value < 0 or (self._end_frame >= 0 and value > self._end_frame):
            raise ValueError(f"timestep is outside the loaded motion: {value}")
        if value < int(self._frame):
            self._clear_recurrent_state(clear_reference_stream=False)
        self._frame = float(value)

    @property
    def start_frame(self) -> int:
        return self._start_frame

    @property
    def end_frame(self) -> int:
        return self._end_frame

    def configure_range(
        self,
        start_frame: int | None = None,
        end_frame: int | None = None,
    ) -> None:
        if self.motioninputpos is None:
            raise RuntimeError("configure_range requires an NPZ motion")
        motion_end = self.motioninputpos.shape[0] - 1
        new_start = self._start_frame if start_frame is None else int(start_frame)
        new_end = self._end_frame if end_frame is None else int(end_frame)
        if not 0 <= new_start <= new_end <= motion_end:
            raise ValueError(
                f"invalid RGMT range [{new_start},{new_end}] for [0,{motion_end}]"
            )
        self._start_frame = new_start
        self._end_frame = new_end

    def _clear_recurrent_state(self, *, clear_reference_stream: bool) -> None:
        self.raw_action.fill(0.0)
        self.action_buffer.fill(0.0)
        self.action.fill(0.0)
        self._proprio_history.clear()
        self._reference_yaw_delta = None
        if clear_reference_stream:
            self._reference_stream.clear()

    def reset(self, frame: InferenceFrame) -> None:
        self.bind_joints(frame)
        self._frame = float(self._start_frame)
        self._clear_recurrent_state(clear_reference_stream=True)
        np.copyto(self._target, self._parameters.default_position)
        self.publish_output(
            self._target,
            self._parameters.kp,
            self._parameters.kd,
            completed=False,
        )

    def reset_reference_yaw_alignment(self) -> None:
        """Align the next standing/live reference to the robot's current yaw."""
        self._reference_yaw_delta = None

    def _npz_reference_window(self) -> tuple[np.ndarray, ...]:
        if self.motioninputpos is None:
            raise RuntimeError(
                "RGMT has no NPZ reference; call step_with_reference_window()"
            )
        assert self.motion_anchor_quat_w is not None
        assert self.motion_anchor_lin_vel_w is not None
        assert self.motion_anchor_ang_vel_w is not None
        center = min(max(int(self._frame), self._start_frame), self._end_frame)
        frame_indices = np.clip(
            center + RGMT_COMMAND_WINDOW_OFFSETS,
            0,
            self._end_frame,
        )
        return (
            self.motioninputpos[frame_indices],
            self.motion_anchor_quat_w[frame_indices],
            self.motion_anchor_lin_vel_w[frame_indices],
            self.motion_anchor_ang_vel_w[frame_indices],
        )

    def _aligned_reference_quaternion(
        self,
        robot_quat_w: np.ndarray,
        reference_quat_w: np.ndarray,
        *,
        advance: bool,
    ) -> np.ndarray:
        if self.reference_yaw_mode == "none":
            return reference_quat_w
        yaw_delta = self._reference_yaw_delta
        if self.reference_yaw_mode == "continuous" or yaw_delta is None:
            yaw_delta = quaternion_multiply_wxyz(
                yaw_quaternion_wxyz(robot_quat_w),
                quaternion_conjugate_wxyz(yaw_quaternion_wxyz(reference_quat_w)),
            )
            if advance and self.reference_yaw_mode == "initial":
                self._reference_yaw_delta = yaw_delta.copy()
        aligned = quaternion_multiply_wxyz(yaw_delta, reference_quat_w)
        return aligned / np.linalg.norm(aligned).clip(
            min=np.finfo(np.float32).eps
        )

    def _anchor_orientation_6d(
        self,
        robot_quat_w: np.ndarray,
        reference_quat_w: np.ndarray,
        *,
        advance: bool,
    ) -> np.ndarray:
        aligned_reference = self._aligned_reference_quaternion(
            robot_quat_w,
            reference_quat_w,
            advance=advance,
        )
        relative = quaternion_multiply_wxyz(
            quaternion_conjugate_wxyz(robot_quat_w),
            aligned_reference,
        )
        relative /= np.linalg.norm(relative).clip(
            min=np.finfo(np.float32).eps
        )
        return quaternion_to_rotation_matrix_wxyz(relative)[:, :2].reshape(-1)

    def _build_observation(
        self,
        frame: InferenceFrame,
        reference_joint_pos: np.ndarray,
        reference_quat: np.ndarray,
        reference_lin_vel: np.ndarray,
        reference_ang_vel: np.ndarray,
        *,
        advance: bool,
    ) -> None:
        joints = self.bind_joints(frame)
        robot_quat = _normalize_quaternion_wxyz(
            "frame.quat_wxyz",
            frame.quat_wxyz,
            (4,),
        )
        base_ang_vel = _as_finite_array(
            "frame.angular_velocity",
            frame.angular_velocity,
            (3,),
        )
        reference_quat = _normalize_quaternion_wxyz(
            "reference_anchor_quat_window_w",
            reference_quat,
            (RGMT_COMMAND_WINDOW_SIZE, 4),
        )
        command = build_rgmt_command_window(
            reference_lin_vel,
            reference_ang_vel,
            reference_quat,
            reference_joint_pos,
        )
        anchor_orientation = self._anchor_orientation_6d(
            robot_quat,
            reference_quat[RGMT_COMMAND_WINDOW_SIZE // 2],
            advance=advance,
        )
        projected_gravity = quat_rotate_inverse_wxyz(
            robot_quat,
            np.asarray((0.0, 0.0, -1.0), dtype=np.float32),
        )

        proprio = self._proprio
        proprio[0:3] = projected_gravity
        proprio[3:6] = base_ang_vel
        np.subtract(
            joints.position,
            self._parameters.default_position,
            out=proprio[6:35],
        )
        proprio[35:64] = joints.velocity
        proprio[64:93] = self.action_buffer

        command_end = RGMT_COMMAND_WINDOW_SIZE * RGMT_COMMAND_TOKEN_DIM
        orientation_end = command_end + 6
        self.obs[:command_end] = command.reshape(-1)
        self.obs[command_end:orientation_end] = anchor_orientation
        history_target = self.obs[orientation_end:]
        if not self._proprio_history.initialized:
            if advance:
                self._proprio_history.fill(proprio)
                self._proprio_history.write_into(history_target)
            else:
                history_target.reshape(
                    RGMT_PROPRIO_HISTORY_LENGTH,
                    RGMT_PROPRIO_TOKEN_DIM,
                )[...] = proprio
        elif advance:
            self._proprio_history.append(proprio)
            self._proprio_history.write_into(history_target)
        else:
            self._proprio_history.preview_append_into(proprio, history_target)

    def _decode_actor(
        self,
        outputs: object,
        reference_center: np.ndarray,
        *,
        advance: bool,
    ) -> None:
        raw = np.asarray(outputs["actions"], dtype=np.float32).reshape(-1)
        if raw.shape != (RGMT_NUM_JOINTS,) or not np.isfinite(raw).all():
            raise ValueError(f"RGMT backend produced invalid actions: {raw.shape}")
        np.copyto(self.raw_action, raw)
        if self.action_clip is None:
            np.copyto(self._clipped_action, raw)
        else:
            np.clip(
                raw,
                -self.action_clip,
                self.action_clip,
                out=self._clipped_action,
            )
        np.multiply(
            self._clipped_action,
            self._parameters.action_scale,
            out=self._scaled_action,
        )
        np.add(reference_center, self.calibration_offset, out=self._target)
        np.add(self._target, self._scaled_action, out=self._target)
        if advance:
            np.copyto(self.action_buffer, self._clipped_action)
            np.copyto(self.action, self._clipped_action)

    def decode_into(self, outputs) -> None:
        """Decode a committed backend result using the current reference center."""

        self._decode_actor(outputs, self._reference_center, advance=True)

    def step(
        self,
        frame: InferenceFrame,
        dt: float,
        *,
        advance: bool = True,
    ) -> PolicyOutput:
        dt = float(dt)
        if not np.isfinite(dt) or dt < 0.0:
            raise ValueError("RGMT step dt must be finite and non-negative")
        reference = self._npz_reference_window()
        output = self.step_with_reference_window(
            frame,
            dt,
            reference_joint_pos_window=reference[0],
            reference_anchor_quat_window_w=reference[1],
            reference_anchor_lin_vel_window_w=reference[2],
            reference_anchor_ang_vel_window_w=reference[3],
            reference_joint_layout=self.joint_contract.observation,
            advance=advance,
        )
        if advance:
            self._frame += self.policy_fps * dt
        output.completed = self.finished()
        return output

    def step_with_reference_window(
        self,
        frame: InferenceFrame,
        dt: float,
        *,
        reference_joint_pos_window: object,
        reference_anchor_quat_window_w: object,
        reference_anchor_lin_vel_window_w: object,
        reference_anchor_ang_vel_window_w: object,
        reference_joint_layout: JointLayout = ELF3_ISAAC_JOINTS,
        advance: bool = True,
    ) -> PolicyOutput:
        """Run one framework inference step from a centered reference window."""

        del dt
        joint_pos = self._map_reference_joints(
            reference_joint_pos_window,
            reference_joint_layout,
            name="reference_joint_pos_window",
        )
        if joint_pos.shape != (RGMT_COMMAND_WINDOW_SIZE, RGMT_NUM_JOINTS):
            raise ValueError(
                "reference_joint_pos_window must contain exactly 21 frames"
            )
        reference_quat = _normalize_quaternion_wxyz(
            "reference_anchor_quat_window_w",
            reference_anchor_quat_window_w,
            (RGMT_COMMAND_WINDOW_SIZE, 4),
        )
        reference_lin_vel = _as_finite_array(
            "reference_anchor_lin_vel_window_w",
            reference_anchor_lin_vel_window_w,
            (RGMT_COMMAND_WINDOW_SIZE, 3),
        )
        reference_ang_vel = _as_finite_array(
            "reference_anchor_ang_vel_window_w",
            reference_anchor_ang_vel_window_w,
            (RGMT_COMMAND_WINDOW_SIZE, 3),
        )
        np.copyto(
            self._reference_center,
            joint_pos[RGMT_COMMAND_WINDOW_SIZE // 2],
        )

        self._build_observation(
            frame,
            joint_pos,
            reference_quat,
            reference_lin_vel,
            reference_ang_vel,
            advance=advance,
        )
        outputs = self._backend.run(self._inputs)
        self._decode_actor(outputs, self._reference_center, advance=advance)
        return self.output

    def step_streaming(
        self,
        frame: InferenceFrame,
        dt: float,
        *,
        reference_joint_pos: object,
        reference_anchor_quat_w: object,
        reference_anchor_lin_vel_w: object,
        reference_anchor_ang_vel_w: object,
        reference_joint_layout: JointLayout = ELF3_ISAAC_JOINTS,
        advance: bool = True,
    ) -> PolicyOutput | None:
        """Push a live frame and infer once the 21-frame window is full."""

        reference_position = self._map_reference_joints(
            reference_joint_pos,
            reference_joint_layout,
            name="reference_joint_pos",
        )
        live_frame = (
            reference_position.copy(),
            _normalize_quaternion_wxyz(
                "reference_anchor_quat_w",
                reference_anchor_quat_w,
                (4,),
            ).copy(),
            _as_finite_array(
                "reference_anchor_lin_vel_w",
                reference_anchor_lin_vel_w,
                (3,),
            ).copy(),
            _as_finite_array(
                "reference_anchor_ang_vel_w",
                reference_anchor_ang_vel_w,
                (3,),
            ).copy(),
        )
        if advance:
            self._reference_stream.append(live_frame)
            frames = tuple(self._reference_stream)
        else:
            frames = tuple(self._reference_stream) + (live_frame,)
            frames = frames[-RGMT_COMMAND_WINDOW_SIZE:]
        if len(frames) < RGMT_COMMAND_WINDOW_SIZE:
            return None
        return self.step_with_reference_window(
            frame,
            dt,
            reference_joint_pos_window=np.stack([item[0] for item in frames]),
            reference_anchor_quat_window_w=np.stack([item[1] for item in frames]),
            reference_anchor_lin_vel_window_w=np.stack([item[2] for item in frames]),
            reference_anchor_ang_vel_window_w=np.stack([item[3] for item in frames]),
            reference_joint_layout=self.joint_contract.observation,
            advance=advance,
        )

    def finished(self, trim: int = 0) -> bool:
        if self._end_frame < 0:
            return False
        return self._frame > self._end_frame - int(trim)

    def close(self) -> None:
        self._backend.close()


__all__ = [
    "RgmtExternalReferencePolicy",
]
