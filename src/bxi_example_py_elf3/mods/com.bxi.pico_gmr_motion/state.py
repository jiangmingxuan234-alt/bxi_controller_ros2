from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import numpy as np

from bxi_example_py_elf3.framework.joints import CompiledJointMap
from bxi_example_py_elf3.framework.mod_api import (
    JointCommandComposer,
    JointCommandLayer,
    JointLayout,
    JointTargetBuffer,
    ResourceHandle,
    RobotControlState,
    StateBehavior,
)
from bxi_example_py_elf3.framework.mod_api.transition import (
    EntryFrameProvider,
    MotorFrame,
    RunningFrameProvider,
)
from bxi_example_py_elf3.policies.joints import ELF3_ISAAC_JOINTS

from .head_tracking import HEAD_JOINT_NAMES
from .reference import LiveReferenceReceiver, ReferenceWindow
from .rgmt_policy import RgmtExternalReferencePolicy

if TYPE_CHECKING:
    from bxi_example_py_elf3.framework.mod_api import RobotControlContext

PICO_HEAD_JOINTS = JointLayout(HEAD_JOINT_NAMES, label="PICO head command")
PICO_GMR_OUTPUT_JOINTS = JointLayout(
    (*ELF3_ISAAC_JOINTS.names, *PICO_HEAD_JOINTS.names),
    label="PICO GMR state output",
)


@dataclass(frozen=True, slots=True)
class PicoGmrMotionParams:
    host: str = "127.0.0.1"
    port: int = 5568
    stale_timeout_s: float = 0.4
    reference_yaw_mode: str = "initial"
    backend: str = "auto"
    head_pitch_limit_rad: float = 0.5
    head_yaw_limit_rad: float = 1.0
    head_pitch_speed_rad_s: float = 1.5
    head_yaw_speed_rad_s: float = 2.0
    head_deadband_rad: float = 0.015

    def __post_init__(self) -> None:
        if not self.host:
            raise ValueError("PICO GMR host must not be empty")
        if not 0 < self.port < 65536:
            raise ValueError("PICO GMR port must be in [1,65535]")
        if not math.isfinite(self.stale_timeout_s) or self.stale_timeout_s <= 0.0:
            raise ValueError("PICO GMR stale_timeout_s must be positive")
        if self.reference_yaw_mode not in {"none", "initial", "continuous"}:
            raise ValueError("PICO GMR reference_yaw_mode is invalid")
        if self.backend not in {"auto", "rknn", "openvino", "onnxruntime"}:
            raise ValueError("PICO GMR backend is invalid")
        for name, value in (
            ("head_pitch_limit_rad", self.head_pitch_limit_rad),
            ("head_yaw_limit_rad", self.head_yaw_limit_rad),
            ("head_pitch_speed_rad_s", self.head_pitch_speed_rad_s),
            ("head_yaw_speed_rad_s", self.head_yaw_speed_rad_s),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"PICO GMR {name} must be positive and finite")
        if not math.isfinite(self.head_deadband_rad) or self.head_deadband_rad < 0.0:
            raise ValueError("PICO GMR head_deadband_rad must be finite and non-negative")


class PicoGmrMotionState(RobotControlState, EntryFrameProvider, RunningFrameProvider):
    """Keep RGMT balancing and consume live PICO references only while fresh."""

    HEAD_KP = 16.747
    HEAD_KD = 1.066

    def __init__(
        self,
        name: str,
        state_id: int,
        *,
        policy: ResourceHandle[RgmtExternalReferencePolicy],
        receiver: ResourceHandle[LiveReferenceReceiver],
        params: PicoGmrMotionParams,
    ) -> None:
        super().__init__(name, state_id, resources=(policy, receiver))
        self._policy = policy
        self._receiver = receiver
        self.params = params
        self._hold_position = np.zeros(ELF3_ISAAC_JOINTS.dof_num, dtype=np.float32)
        self._hold_map: CompiledJointMap | None = None
        self._last_session_id: int | None = None
        self._last_sequence: int | None = None
        self._has_inferred = False
        self._reference_source: str | None = None
        self._fallback_joint_pos = np.zeros((21, 29), dtype=np.float32)
        self._fallback_quat_wxyz = np.zeros((21, 4), dtype=np.float32)
        self._fallback_lin_vel_w = np.zeros((21, 3), dtype=np.float32)
        self._fallback_ang_vel_w = np.zeros((21, 3), dtype=np.float32)
        self._hold_command = JointTargetBuffer(ELF3_ISAAC_JOINTS)
        self._head_command = JointTargetBuffer(PICO_HEAD_JOINTS)
        self._policy_composer: JointCommandComposer | None = None
        self._hold_composer: JointCommandComposer | None = None

    @property
    def policy(self) -> RgmtExternalReferencePolicy:
        return self._policy.get()

    @property
    def receiver(self) -> LiveReferenceReceiver:
        return self._receiver.get()

    def _capture_hold(self, ctx: RobotControlContext) -> None:
        source = ctx.robot_joints.layout
        mapping = self._hold_map
        if mapping is None or mapping.source != source:
            mapping = CompiledJointMap.compile(source, ELF3_ISAAC_JOINTS)
            self._hold_map = mapping
        mapping.map_into(ctx.robot_joints.position, self._hold_position)

    def _align_fallback_orientation(self, ctx: RobotControlContext) -> None:
        quaternion = np.asarray(ctx.inference_frame.quat_wxyz, dtype=np.float32)
        norm = float(np.linalg.norm(quaternion))
        if not np.isfinite(quaternion).all() or norm <= np.finfo(np.float32).eps:
            raise ValueError("robot quaternion is invalid while preparing PICO GMR")
        self._fallback_quat_wxyz[...] = quaternion / norm
        self._fallback_lin_vel_w.fill(0.0)
        self._fallback_ang_vel_w.fill(0.0)

    def _prepare_fallback(self, ctx: RobotControlContext) -> None:
        self._fallback_joint_pos[...] = self.policy.output.joints.position
        self._align_fallback_orientation(ctx)

    def _prepare_command_sources(self) -> None:
        policy_target = self.policy.output.joints
        self._hold_command.update(
            self._hold_position,
            policy_target.kp,
            policy_target.kd,
        )
        self._head_command.position.fill(0.0)
        self._head_command.kp.fill(self.HEAD_KP)
        self._head_command.kd.fill(self.HEAD_KD)
        self._policy_composer = JointCommandComposer(
            PICO_GMR_OUTPUT_JOINTS,
            (
                JointCommandLayer("rgmt_policy", policy_target),
                JointCommandLayer("pico_head", self._head_command.view),
            ),
        )
        self._hold_composer = JointCommandComposer(
            PICO_GMR_OUTPUT_JOINTS,
            (
                JointCommandLayer("entry_hold", self._hold_command.view),
                JointCommandLayer("pico_head", self._head_command.view),
            ),
        )

    def _update_head_command(
        self,
        desired: object,
        dt: float,
        *,
        advance: bool,
    ) -> None:
        if not advance:
            return
        target = np.asarray(desired, dtype=np.float32)
        if target.shape != (2,) or not np.isfinite(target).all():
            raise ValueError("PICO head target must contain two finite joint angles")
        target = np.clip(
            target,
            (-self.params.head_pitch_limit_rad, -self.params.head_yaw_limit_rad),
            (self.params.head_pitch_limit_rad, self.params.head_yaw_limit_rad),
        )
        target[np.abs(target) < self.params.head_deadband_rad] = 0.0
        max_step = np.asarray(
            (
                self.params.head_pitch_speed_rad_s * dt,
                self.params.head_yaw_speed_rad_s * dt,
            ),
            dtype=np.float32,
        )
        delta = np.clip(
            target - self._head_command.position,
            -max_step,
            max_step,
        )
        self._head_command.position += delta

    def _compose_policy_frame(self) -> MotorFrame:
        if self._policy_composer is None:
            raise RuntimeError("PICO GMR policy command composer is not prepared")
        return self._policy_composer.compose()

    def _hold_frame(self) -> MotorFrame:
        if self._hold_composer is None:
            raise RuntimeError("PICO GMR hold command composer is not prepared")
        return self._hold_composer.compose()

    def on_prepare(
        self,
        ctx: RobotControlContext,
        from_state: StateBehavior[RobotControlContext],
    ) -> None:
        self.receiver.clear()
        self.policy.reset(ctx.inference_frame)
        self._capture_hold(ctx)
        self._prepare_fallback(ctx)
        self._prepare_command_sources()
        self._last_session_id = None
        self._last_sequence = None
        self._has_inferred = False
        self._reference_source = None

    def on_enter(self, ctx: RobotControlContext) -> None:
        self.logger.info(
            "PICO GMR已启动并保持RGMT站立平衡；PICO同时按A+X开始/停止实时跟踪"
        )

    def get_entry_frame(self, ctx: RobotControlContext) -> MotorFrame:
        return self._hold_frame()

    def _is_new(self, window: ReferenceWindow) -> bool:
        return (
            window.session_id != self._last_session_id
            or window.latest_sequence != self._last_sequence
        )

    def _select_reference_source(
        self,
        ctx: RobotControlContext,
        source: str,
        *,
        session_id: int | None,
        advance: bool,
    ) -> None:
        if not advance:
            return
        source_changed = source != self._reference_source
        session_changed = (
            source == "live"
            and session_id is not None
            and session_id != self._last_session_id
        )
        if source_changed or session_changed:
            self.policy.reset_reference_yaw_alignment()
        if source_changed and source == "standing":
            self._align_fallback_orientation(ctx)
            self._last_session_id = None
            self._last_sequence = None
        self._reference_source = source

    def sample_running_frame(
        self,
        ctx: RobotControlContext,
        dt: float,
        *,
        advance: bool,
    ) -> MotorFrame:
        window = self.receiver.snapshot_window(
            max_age_s=self.params.stale_timeout_s,
        )
        if window is None:
            self._select_reference_source(
                ctx,
                "standing",
                session_id=None,
                advance=advance,
            )
            self.policy.step_with_reference_window(
                ctx.inference_frame,
                dt,
                reference_joint_pos_window=self._fallback_joint_pos,
                reference_anchor_quat_window_w=self._fallback_quat_wxyz,
                reference_anchor_lin_vel_window_w=self._fallback_lin_vel_w,
                reference_anchor_ang_vel_window_w=self._fallback_ang_vel_w,
                reference_joint_layout=ELF3_ISAAC_JOINTS,
                advance=advance,
            )
            if advance:
                self._has_inferred = True
            self._update_head_command((0.0, 0.0), dt, advance=advance)
            return self._compose_policy_frame()
        self._select_reference_source(
            ctx,
            "live",
            session_id=window.session_id,
            advance=advance,
        )
        self._update_head_command(
            window.head_joint_pos[-1],
            dt,
            advance=advance,
        )
        if not self._is_new(window):
            if self._has_inferred:
                return self._compose_policy_frame()
            return self._hold_frame()
        self.policy.step_with_reference_window(
            ctx.inference_frame,
            dt,
            reference_joint_pos_window=window.joint_pos,
            reference_anchor_quat_window_w=window.anchor_quat_wxyz,
            reference_anchor_lin_vel_window_w=window.anchor_lin_vel_w,
            reference_anchor_ang_vel_window_w=window.anchor_ang_vel_w,
            reference_joint_layout=ELF3_ISAAC_JOINTS,
            advance=advance,
        )
        if advance:
            self._last_session_id = window.session_id
            self._last_sequence = window.latest_sequence
            self._has_inferred = True
        return self._compose_policy_frame()

    def on_update(self, ctx: RobotControlContext, dt: float) -> None:
        previous_source = self._reference_source
        frame = self.sample_running_frame(ctx, dt, advance=True)
        self._apply_frame(ctx, frame)
        if self._reference_source == previous_source:
            return
        if self._reference_source == "live":
            self.logger.info("已收满21帧新鲜PICO参考，开始实时GMR跟踪")
        else:
            self.logger.info("实时参考未启用或已断流，继续使用RGMT站立平衡参考")


__all__ = [
    "PICO_GMR_OUTPUT_JOINTS",
    "PICO_HEAD_JOINTS",
    "PicoGmrMotionParams",
    "PicoGmrMotionState",
]
