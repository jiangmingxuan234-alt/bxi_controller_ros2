from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import TYPE_CHECKING

from bxi_example_py_elf3.framework.mod_api import (
    ResourceHandle,
    RobotControlState,
    StateBehavior,
)
from bxi_example_py_elf3.framework.mod_api.transition import (
    EntryFrameProvider,
    MotorFrame,
    RunningFrameProvider,
)

from .rgmt import RgmtExternalReferencePolicy

if TYPE_CHECKING:
    from bxi_example_py_elf3.framework.mod_api import RobotControlContext


@dataclass(frozen=True, slots=True)
class AnyMotionParams:
    """Per-motion settings; the RGMT ONNX model is owned by the Mod."""

    npz: str = "assets/motion.npz"
    start_frame: int = 0
    end_frame: int = -1
    end_frame_trim: int = 0
    reference_yaw_mode: str = "initial"
    backend: str = "auto"
    finish_state: str = "com.bxi.basic_actions/normal"
    end_transition_seconds: float = 0.6

    def __post_init__(self) -> None:
        if not self.npz:
            raise ValueError("any_motion npz must not be empty")
        if self.start_frame < 0:
            raise ValueError("any_motion start_frame must be non-negative")
        if self.end_frame < -1:
            raise ValueError("any_motion end_frame must be -1 or non-negative")
        if self.end_frame >= 0 and self.end_frame < self.start_frame:
            raise ValueError("any_motion end_frame must not precede start_frame")
        if self.end_frame_trim < 0:
            raise ValueError("any_motion end_frame_trim must be non-negative")
        if self.reference_yaw_mode not in {"none", "initial", "continuous"}:
            raise ValueError(
                "any_motion reference_yaw_mode must be none, initial or continuous"
            )
        if self.backend not in {"auto", "rknn", "openvino", "onnxruntime"}:
            raise ValueError(
                "any_motion backend must be auto, rknn, openvino or onnxruntime"
            )
        if not self.finish_state:
            raise ValueError("any_motion finish_state must not be empty")
        if (
            not math.isfinite(self.end_transition_seconds)
            or self.end_transition_seconds < 0.0
        ):
            raise ValueError(
                "any_motion end_transition_seconds must be finite and non-negative"
            )


def resolve_mod_asset(mod_root: Path, relative_path: str, suffix: str) -> Path:
    """Resolve one configured file while keeping it inside the Mod assets."""

    if Path(relative_path).is_absolute():
        raise ValueError("any_motion asset paths must be relative to the Mod")
    assets_root = (mod_root / "assets").resolve()
    path = (mod_root / relative_path).resolve()
    if assets_root not in path.parents:
        raise ValueError(
            f"any_motion asset must be below {assets_root}: {relative_path}"
        )
    if path.suffix.lower() != suffix:
        raise ValueError(f"any_motion asset must end in {suffix}: {relative_path}")
    return path


class AnyMotionState(
    RobotControlState,
    EntryFrameProvider,
    RunningFrameProvider,
):
    """MotionReplay-style state backed by the fixed RGMT model and one NPZ."""

    def __init__(
        self,
        name: str,
        state_id: int,
        *,
        policy: ResourceHandle[RgmtExternalReferencePolicy],
        model_path: Path,
        motion_path: Path,
        params: AnyMotionParams,
    ) -> None:
        super().__init__(name, state_id, resources=(policy,))
        self._policy = policy
        self.model_path = model_path
        self.motion_path = motion_path
        self.params = params
        self.playing = True

    @property
    def policy(self) -> RgmtExternalReferencePolicy:
        return self._policy.get()

    def is_available(self, ctx: RobotControlContext) -> bool:
        return self.model_path.is_file() and self.motion_path.is_file()

    def on_prepare(
        self,
        ctx: RobotControlContext,
        from_state: StateBehavior[RobotControlContext],
    ) -> None:
        del from_state
        self.playing = True
        # The on-demand policy instance survives state exits.  Reset the
        # reference cursor and all recurrent state before every transition so
        # re-entering always replays from the configured start_frame.
        self.policy.reset(ctx.inference_frame)

    def on_enter(self, ctx: RobotControlContext) -> None:
        self.playing = True

    def get_entry_frame(self, ctx: RobotControlContext) -> MotorFrame:
        return self._motor_frame_from_target(ctx, self.policy.output.joints)

    def sample_running_frame(
        self,
        ctx: RobotControlContext,
        dt: float,
        *,
        advance: bool,
    ) -> MotorFrame:
        output = self.policy.step(
            ctx.inference_frame,
            dt,
            advance=self.playing and advance,
        )
        return self._motor_frame_from_target(ctx, output.joints)

    def on_update(self, ctx: RobotControlContext, dt: float) -> None:
        self._apply_frame(ctx, self.sample_running_frame(ctx, dt, advance=True))
        # if self.policy.finished(self.params.end_frame_trim):
        #     ctx.request_state(
        #         self.params.finish_state,
        #         trigger="any_motion_finished",
        #         transition={
        #             "profile": "dual_running_blend",
        #             "duration": self.params.end_transition_seconds,
        #             "curve": "smootherstep",
        #             "sample_from": True,
        #         },
        #     )

    def on_action(self, ctx: RobotControlContext, action_name: str) -> bool:
        if action_name != "toggle_pause":
            return False
        self.playing = not self.playing
        return True


__all__ = ["AnyMotionParams", "AnyMotionState", "resolve_mod_asset"]
