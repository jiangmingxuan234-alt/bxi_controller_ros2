from __future__ import annotations

from enum import Enum
import math
from typing import TYPE_CHECKING

import numpy as np

from bxi_example_py_elf3.framework.mod_api.transition import MotorFrame

from ..state import SonicTeleopState

if TYPE_CHECKING:
    from bxi_example_py_elf3.framework.mod_api import (
        RobotControlContext,
        StateBehavior,
    )


ARM_ACTION = "arm_zerolab"


class ZeroLabArmPhase(str, Enum):
    WAIT_CALIBRATION = "wait_calibration"
    WAIT_ARM = "wait_arm"
    BLENDING = "blending"
    ARMED = "armed"
    HOLD_STALE = "hold_stale"


class ZeroLabArmedTeleopState(SonicTeleopState):
    def __init__(self, name, state_id, policy, *, arm_blend_seconds=2.0, **kwargs):
        super().__init__(name, state_id, policy, **kwargs)
        value = float(arm_blend_seconds)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("arm_blend_seconds must be finite and positive")
        self.arm_blend_seconds = value
        self._arm_phase = ZeroLabArmPhase.WAIT_CALIBRATION
        self._hold_frame = None
        self._applied_frame = None
        self._live_frame = None
        self._blend_start_frame = None
        self._blend_elapsed_s = 0.0
        self._recovery_notice_logged = False
        self._phase_logged = False

    @property
    def arm_phase(self) -> ZeroLabArmPhase:
        return self._arm_phase

    @staticmethod
    def _copy_frame(target: MotorFrame, source: MotorFrame) -> MotorFrame:
        return target.update(
            source.qpos,
            source.kp,
            source.kd,
            vel=source.vel,
            torque=source.torque,
        )

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
        self._hold_frame = MotorFrame.empty(ctx.robot_layout)
        self._applied_frame = MotorFrame.empty(ctx.robot_layout)
        self._live_frame = MotorFrame.empty(ctx.robot_layout)
        self._blend_start_frame = MotorFrame.empty(ctx.robot_layout)
        self._copy_frame(self._hold_frame, ctx.last_motor_frame)
        self._copy_frame(self._applied_frame, ctx.last_motor_frame)
        self._blend_elapsed_s = 0.0
        self._recovery_notice_logged = False
        self._set_phase(
            ZeroLabArmPhase.WAIT_CALIBRATION,
            "ZeroLab ARM phase: WAIT_CALIBRATION",
        )

    def get_entry_frame(self, ctx: RobotControlContext) -> MotorFrame:
        del ctx
        assert self._hold_frame is not None
        return self._hold_frame

    @staticmethod
    def _smoothstep(progress: float) -> float:
        p = min(max(float(progress), 0.0), 1.0)
        return p * p * (3.0 - 2.0 * p)

    def _begin_blend(self) -> None:
        assert self._blend_start_frame is not None
        assert self._applied_frame is not None
        self._copy_frame(self._blend_start_frame, self._applied_frame)
        self._blend_elapsed_s = 0.0
        self._set_phase(
            ZeroLabArmPhase.BLENDING,
            f"ZeroLab ARM accepted; blending for {self.arm_blend_seconds:.3f} s",
        )

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

        assert self._hold_frame is not None
        assert self._live_frame is not None
        natural_frame = super().sample_running_frame(ctx, dt, advance=True)
        ctx.resolve_motor_frame(natural_frame, self._live_frame)
        fresh_reference = self.policy.has_fresh_live_reference(
            self.live_reference_timeout_s
        )
        if (
            self._arm_phase is ZeroLabArmPhase.WAIT_CALIBRATION
            and fresh_reference
        ):
            self._set_phase(
                ZeroLabArmPhase.WAIT_ARM,
                "ZeroLab ARM phase: WAIT_ARM",
            )

        if (
            self._arm_phase
            in (ZeroLabArmPhase.BLENDING, ZeroLabArmPhase.ARMED)
            and not fresh_reference
        ):
            self._copy_frame(self._hold_frame, self._applied_frame)
            self._blend_elapsed_s = 0.0
            self._recovery_notice_logged = False
            self._set_phase(
                ZeroLabArmPhase.HOLD_STALE,
                "ZeroLab reference stale; holding last motor frame and ARM cancelled",
                warning=True,
            )
            return self._applied_frame

        if self._arm_phase is ZeroLabArmPhase.HOLD_STALE:
            if fresh_reference and not self._recovery_notice_logged:
                self.logger.info(
                    "ZeroLab reference recovered; send btn_10=12 to resume"
                )
                self._recovery_notice_logged = True
            return self._hold_frame

        if self._arm_phase is ZeroLabArmPhase.BLENDING:
            assert self._blend_start_frame is not None
            self._blend_elapsed_s += max(dt, 0.0)
            alpha = self._smoothstep(
                self._blend_elapsed_s / self.arm_blend_seconds
            )
            for target, start, output in (
                (
                    self._live_frame.qpos,
                    self._blend_start_frame.qpos,
                    self._applied_frame.qpos,
                ),
                (
                    self._live_frame.kp,
                    self._blend_start_frame.kp,
                    self._applied_frame.kp,
                ),
                (
                    self._live_frame.kd,
                    self._blend_start_frame.kd,
                    self._applied_frame.kd,
                ),
                (
                    self._live_frame.vel,
                    self._blend_start_frame.vel,
                    self._applied_frame.vel,
                ),
                (
                    self._live_frame.torque,
                    self._blend_start_frame.torque,
                    self._applied_frame.torque,
                ),
            ):
                np.subtract(target, start, out=output)
                output *= alpha
                output += start
            if self._blend_elapsed_s >= self.arm_blend_seconds:
                self._set_phase(
                    ZeroLabArmPhase.ARMED,
                    "ZeroLab ARM phase: ARMED",
                )
            return self._applied_frame

        if self._arm_phase is ZeroLabArmPhase.ARMED:
            return self._copy_frame(self._applied_frame, self._live_frame)

        self._copy_frame(self._applied_frame, self._hold_frame)
        return self._applied_frame

    def on_exit(self, ctx: RobotControlContext) -> None:
        super().on_exit(ctx)
        self._set_phase(
            ZeroLabArmPhase.WAIT_CALIBRATION,
            "ZeroLab ARM phase: WAIT_CALIBRATION",
        )
        self._hold_frame = None
        self._applied_frame = None
        self._live_frame = None
        self._blend_start_frame = None
        self._blend_elapsed_s = 0.0
        self._recovery_notice_logged = False

    def on_action(self, ctx: RobotControlContext, action_name: str) -> bool:
        if action_name != ARM_ACTION:
            return super().on_action(ctx, action_name)
        if (
            self._arm_phase
            in (ZeroLabArmPhase.WAIT_ARM, ZeroLabArmPhase.HOLD_STALE)
            and self.policy.has_fresh_live_reference(
                self.live_reference_timeout_s
            )
        ):
            self._begin_blend()
        elif self._arm_phase in (
            ZeroLabArmPhase.BLENDING,
            ZeroLabArmPhase.ARMED,
        ):
            self.logger.info("ZeroLab ARM ignored; already active")
        else:
            self.logger.warning("ZeroLab ARM refused; wait for a fresh reference")
        return True


__all__ = ["ARM_ACTION", "ZeroLabArmPhase", "ZeroLabArmedTeleopState"]
