from __future__ import annotations

from enum import Enum
import math
from typing import TYPE_CHECKING

import numpy as np

from bxi_example_py_elf3.framework.joints import CompiledJointMap, JointLayout
from bxi_example_py_elf3.framework.mod_api import ResourceHandle
from bxi_example_py_elf3.framework.mod_api.transition import MotorFrame
from bxi_example_py_elf3.policies import HumanoidGaitPolicyLiteIsaaclab
from bxi_example_py_elf3.policies.joints import ELF3_POLICY_JOINTS

from ..state import SonicTeleopState

if TYPE_CHECKING:
    from bxi_example_py_elf3.framework.mod_api import (
        RobotControlContext,
        StateBehavior,
    )


ARM_ACTION = "arm_zerolab"


class ZeroLabArmPhase(str, Enum):
    WAIT_STREAM = "wait_stream"
    WAIT_ARM = "wait_arm"
    BLENDING = "blending"
    ARMED = "armed"
    DISARMING = "disarming"
    HOLD_REFERENCE = "hold_reference"
    REARMING = "rearming"


class ZeroLabArmedTeleopState(SonicTeleopState):
    def __init__(
        self,
        name,
        state_id,
        policy,
        *,
        normal_policy: ResourceHandle[HumanoidGaitPolicyLiteIsaaclab],
        arm_blend_seconds=2.0,
        auto_rearm_on_recovery=True,
        auto_rearm_blend_seconds=2.0,
        recovery_real_frames=10,
        **kwargs,
    ):
        super().__init__(
            name,
            state_id,
            policy,
            additional_resources=(normal_policy,),
            **kwargs,
        )
        self._normal_policy = normal_policy
        value = float(arm_blend_seconds)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("arm_blend_seconds must be finite and positive")
        self.arm_blend_seconds = value
        if not isinstance(auto_rearm_on_recovery, bool):
            raise ValueError("auto_rearm_on_recovery must be a boolean")
        self.auto_rearm_on_recovery = auto_rearm_on_recovery
        if isinstance(auto_rearm_blend_seconds, bool):
            raise ValueError(
                "auto_rearm_blend_seconds must be finite and positive"
            )
        value = float(auto_rearm_blend_seconds)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                "auto_rearm_blend_seconds must be finite and positive"
            )
        self.auto_rearm_blend_seconds = value
        if (
            isinstance(recovery_real_frames, bool)
            or not isinstance(recovery_real_frames, int)
            or recovery_real_frames < 1
        ):
            raise ValueError(
                "recovery_real_frames must be a positive integer"
            )
        self.recovery_real_frames = recovery_real_frames
        self._arm_phase = ZeroLabArmPhase.WAIT_STREAM
        self._entry_frame = None
        self._applied_frame = None
        self._live_frame = None
        self._normal_frame = None
        self._disarm_frame = None
        self._blend_elapsed_s = 0.0
        self._initial_arm_completed = False
        self._auto_rearm_count = 0
        self._recovery_notice_logged = False
        self._phase_logged = False
        self._applied_target_map = None
        self._applied_policy_qpos = np.empty(
            ELF3_POLICY_JOINTS.dof_num, dtype=np.float32
        )

    @property
    def arm_phase(self) -> ZeroLabArmPhase:
        return self._arm_phase

    @property
    def auto_rearm_count(self) -> int:
        return self._auto_rearm_count

    @staticmethod
    def _copy_frame(target: MotorFrame, source: MotorFrame) -> MotorFrame:
        return target.update(
            source.qpos,
            source.kp,
            source.kd,
            vel=source.vel,
            torque=source.torque,
        )

    def _prepare_applied_target_mapping(self, source_layout: JointLayout) -> None:
        self._applied_target_map = CompiledJointMap.compile(
            source_layout,
            ELF3_POLICY_JOINTS,
        )

    def _record_previous_applied_target(self, frame: MotorFrame) -> None:
        mapping = self._applied_target_map
        if mapping is None:
            raise RuntimeError("ZeroLab applied-target mapping is not prepared")
        if frame.layout != mapping.source:
            raise ValueError("previous motor frame layout changed during ZeroLab")
        mapping.map_into(frame.qpos, self._applied_policy_qpos)
        self.policy.record_applied_joint_target(self._applied_policy_qpos)

    def _set_phase(
        self,
        phase: ZeroLabArmPhase,
        message: str,
        *,
        warning: bool = False,
    ) -> None:
        if phase is self._arm_phase and self._phase_logged:
            return
        self._arm_phase = phase
        self._phase_logged = True
        if warning:
            self.logger.warning(message)
        else:
            self.logger.info(message)

    def on_prepare(
        self,
        ctx: RobotControlContext,
        from_state: StateBehavior[RobotControlContext],
    ) -> None:
        super().on_prepare(ctx, from_state)
        self._entry_frame = MotorFrame.empty(ctx.robot_layout)
        self._applied_frame = MotorFrame.empty(ctx.robot_layout)
        self._live_frame = MotorFrame.empty(ctx.robot_layout)
        self._normal_frame = MotorFrame.empty(ctx.robot_layout)
        self._disarm_frame = MotorFrame.empty(ctx.robot_layout)
        self._prepare_applied_target_mapping(ctx.last_motor_frame.layout)
        self._copy_frame(self._entry_frame, ctx.last_motor_frame)
        self._copy_frame(self._applied_frame, ctx.last_motor_frame)
        self._blend_elapsed_s = 0.0
        self._initial_arm_completed = False
        self._auto_rearm_count = 0
        self._recovery_notice_logged = False
        self._set_phase(
            ZeroLabArmPhase.WAIT_STREAM,
            "ZeroLab ARM phase: WAIT_STREAM",
        )
        self.logger.info(
            "ZeroLab pre-ARM output: live zero-command Normal policy"
        )

    def get_entry_frame(self, ctx: RobotControlContext) -> MotorFrame:
        del ctx
        assert self._entry_frame is not None
        return self._entry_frame

    @staticmethod
    def _smoothstep(progress: float) -> float:
        p = min(max(float(progress), 0.0), 1.0)
        return p * p * (3.0 - 2.0 * p)

    @staticmethod
    def _blend_frames(source, target, output, alpha):
        for start, end, destination in (
            (source.qpos, target.qpos, output.qpos),
            (source.kp, target.kp, output.kp),
            (source.kd, target.kd, output.kd),
            (source.vel, target.vel, output.vel),
            (source.torque, target.torque, output.torque),
        ):
            np.subtract(end, start, out=destination)
            destination *= alpha
            destination += start
        return output

    def _sample_normal_frame(self, ctx, dt, *, advance):
        assert self._normal_frame is not None
        self.get_cmd_vel(ctx)
        output = self._normal_policy.get().step(
            ctx.inference_frame,
            dt,
            advance=advance,
        )
        natural = self._motor_frame_from_target(ctx, output.joints)
        return ctx.resolve_motor_frame(natural, self._normal_frame)

    def _normal_balance_active(self) -> bool:
        return self._arm_phase in (
            ZeroLabArmPhase.WAIT_STREAM,
            ZeroLabArmPhase.WAIT_ARM,
            ZeroLabArmPhase.BLENDING,
            ZeroLabArmPhase.DISARMING,
        )

    def on_update(self, ctx: RobotControlContext, dt: float) -> None:
        if self._normal_balance_active() and ctx.is_orientation_unsafe(
            ctx.current_quat_xyzw
        ):
            ctx.request_state(
                "com.bxi.basic_actions/zero_torque",
                trigger="safety",
            )
            return
        super().on_update(ctx, dt)

    def sample_running_frame(
        self,
        ctx: RobotControlContext,
        dt: float,
        *,
        advance: bool,
    ) -> MotorFrame:
        assert self._applied_frame is not None
        if not advance:
            return self._applied_frame

        assert self._live_frame is not None
        if self._arm_phase is ZeroLabArmPhase.REARMING:
            self._blend_elapsed_s += max(dt, 0.0)
            alpha = self._smoothstep(
                self._blend_elapsed_s / self.auto_rearm_blend_seconds
            )
            self.policy.set_live_reference_rearm_progress(alpha)

        self._record_previous_applied_target(ctx.last_motor_frame)
        natural_frame = super().sample_running_frame(ctx, dt, advance=True)
        ctx.resolve_motor_frame(natural_frame, self._live_frame)
        fresh_reference = self.policy.has_fresh_live_reference(
            self.live_reference_timeout_s
        )
        if (
            self._arm_phase is ZeroLabArmPhase.WAIT_STREAM
            and fresh_reference
        ):
            self._set_phase(
                ZeroLabArmPhase.WAIT_ARM,
                "ZeroLab ARM phase: WAIT_ARM",
            )

        if (
            self._arm_phase is ZeroLabArmPhase.WAIT_ARM
            and not fresh_reference
        ):
            self._set_phase(
                ZeroLabArmPhase.WAIT_STREAM,
                "ZeroLab ARM phase: WAIT_STREAM",
                warning=True,
            )

        if (
            self._arm_phase is ZeroLabArmPhase.BLENDING
            and not fresh_reference
        ):
            self._blend_elapsed_s = 0.0
            self._set_phase(
                ZeroLabArmPhase.WAIT_STREAM,
                "ZeroLab initial ARM cancelled; reference stale; "
                "returning to live Normal",
                warning=True,
            )
            normal = self._sample_normal_frame(ctx, dt, advance=True)
            return self._copy_frame(self._applied_frame, normal)

        if (
            self._arm_phase
            in (ZeroLabArmPhase.ARMED, ZeroLabArmPhase.REARMING)
            and not fresh_reference
        ):
            self.policy.hold_live_reference()
            self._blend_elapsed_s = 0.0
            self._recovery_notice_logged = False
            self._set_phase(
                ZeroLabArmPhase.HOLD_REFERENCE,
                "ZeroLab reference stale; holding human reference while "
                "SONIC balance continues",
                warning=True,
            )
            return self._copy_frame(self._applied_frame, self._live_frame)

        if self._arm_phase is ZeroLabArmPhase.HOLD_REFERENCE:
            if (
                self.auto_rearm_on_recovery
                and self._initial_arm_completed
                and self.policy.live_reference_recovery_ready(
                    self.recovery_real_frames
                )
                and self.policy.begin_live_reference_rearm()
            ):
                self._blend_elapsed_s = 0.0
                self._auto_rearm_count += 1
                self._set_phase(
                    ZeroLabArmPhase.REARMING,
                    "ZeroLab automatic recovery; ARM phase: REARMING for "
                    f"{self.auto_rearm_blend_seconds:.3f} s",
                )
            elif (
                not self.auto_rearm_on_recovery
                and fresh_reference
                and not self._recovery_notice_logged
            ):
                self.logger.info(
                    "ZeroLab reference recovered; fresh input pending; "
                    "send btn_10=12 to rearm"
                )
                self._recovery_notice_logged = True
            return self._copy_frame(self._applied_frame, self._live_frame)

        if self._arm_phase is ZeroLabArmPhase.DISARMING:
            assert self._disarm_frame is not None
            self._blend_elapsed_s += max(dt, 0.0)
            alpha = self._smoothstep(
                self._blend_elapsed_s / self.arm_blend_seconds
            )
            normal = self._sample_normal_frame(ctx, dt, advance=True)
            self._blend_frames(
                self._disarm_frame,
                normal,
                self._applied_frame,
                alpha,
            )
            if self._blend_elapsed_s >= self.arm_blend_seconds:
                self._copy_frame(self._applied_frame, normal)
                if fresh_reference:
                    self._set_phase(
                        ZeroLabArmPhase.WAIT_ARM,
                        "ZeroLab ARM phase: WAIT_ARM; manual pause complete",
                    )
                else:
                    self._set_phase(
                        ZeroLabArmPhase.WAIT_STREAM,
                        "ZeroLab ARM phase: WAIT_STREAM; manual pause "
                        "complete; reference stale",
                        warning=True,
                    )
            return self._applied_frame

        if self._arm_phase is ZeroLabArmPhase.BLENDING:
            self._blend_elapsed_s += max(dt, 0.0)
            alpha = self._smoothstep(
                self._blend_elapsed_s / self.arm_blend_seconds
            )
            blend_source = self._sample_normal_frame(
                ctx, dt, advance=True
            )
            self._blend_frames(
                blend_source,
                self._live_frame,
                self._applied_frame,
                alpha,
            )
            if self._blend_elapsed_s >= self.arm_blend_seconds:
                self._initial_arm_completed = True
                self._set_phase(
                    ZeroLabArmPhase.ARMED,
                    "ZeroLab ARM phase: ARMED",
                )
            return self._applied_frame

        if self._arm_phase is ZeroLabArmPhase.ARMED:
            return self._copy_frame(self._applied_frame, self._live_frame)

        if self._arm_phase is ZeroLabArmPhase.REARMING:
            if self._blend_elapsed_s >= self.auto_rearm_blend_seconds:
                self.policy.complete_live_reference_rearm()
                message = "ZeroLab recovery complete; ARM phase: ARMED"
                if self.auto_rearm_on_recovery:
                    message = (
                        "ZeroLab automatic recovery complete; "
                        "ARM phase: ARMED"
                    )
                self._set_phase(
                    ZeroLabArmPhase.ARMED,
                    message,
                )
            return self._copy_frame(self._applied_frame, self._live_frame)

        if self._arm_phase in (
            ZeroLabArmPhase.WAIT_STREAM,
            ZeroLabArmPhase.WAIT_ARM,
        ):
            normal = self._sample_normal_frame(ctx, dt, advance=True)
            return self._copy_frame(self._applied_frame, normal)

        return self._applied_frame

    def on_exit(self, ctx: RobotControlContext) -> None:
        super().on_exit(ctx)
        self._set_phase(
            ZeroLabArmPhase.WAIT_STREAM,
            "ZeroLab ARM phase: WAIT_STREAM",
        )
        self._entry_frame = None
        self._applied_frame = None
        self._live_frame = None
        self._normal_frame = None
        self._disarm_frame = None
        self._applied_target_map = None
        self._applied_policy_qpos.fill(0.0)
        self._blend_elapsed_s = 0.0
        self._initial_arm_completed = False
        self._auto_rearm_count = 0
        self._recovery_notice_logged = False

    def on_action(self, ctx: RobotControlContext, action_name: str) -> bool:
        if action_name != ARM_ACTION:
            return super().on_action(ctx, action_name)
        fresh_reference = self.policy.has_fresh_live_reference(
            self.live_reference_timeout_s
        )
        if self._arm_phase is ZeroLabArmPhase.WAIT_ARM and fresh_reference:
            self._blend_elapsed_s = 0.0
            self._set_phase(
                ZeroLabArmPhase.BLENDING,
                "ZeroLab ARM accepted; blending live Normal -> SONIC for "
                f"{self.arm_blend_seconds:.3f} s",
            )
        elif self._arm_phase in (
            ZeroLabArmPhase.BLENDING,
            ZeroLabArmPhase.ARMED,
            ZeroLabArmPhase.HOLD_REFERENCE,
            ZeroLabArmPhase.REARMING,
        ):
            assert self._applied_frame is not None
            assert self._disarm_frame is not None
            self._copy_frame(self._disarm_frame, self._applied_frame)
            self.policy.release_live_reference_hold()
            self._blend_elapsed_s = 0.0
            self._initial_arm_completed = False
            self._set_phase(
                ZeroLabArmPhase.DISARMING,
                "ZeroLab ARM phase: DISARMING; blending SONIC -> live "
                "Normal for "
                f"{self.arm_blend_seconds:.3f} s",
            )
        elif self._arm_phase is ZeroLabArmPhase.DISARMING:
            self.logger.info("ZeroLab pause ignored; disarming in progress")
        else:
            self.logger.warning("ZeroLab ARM refused; wait for a fresh reference")
        return True


__all__ = [
    "ARM_ACTION",
    "ZeroLabArmPhase",
    "ZeroLabArmedTeleopState",
]
