"""Human-reference hold and recovery interpolation for SONIC teleoperation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import Optional

import numpy as np


@dataclass
class SmplReferenceFrame:
    term1_local: np.ndarray
    root_quat: np.ndarray
    wrist: np.ndarray
    head_joint_pos: np.ndarray
    anchor_quat: Optional[np.ndarray] = None
    frame_index: int = -1
    sequence: int = 0
    stream_epoch: Optional[int] = None
    source_stale: bool = False
    source_age_ms: Optional[float] = None
    playback_hold: bool = False
    newest_frame_index: int = -1
    lead_frames: int = -1
    valid_horizon: int = 0
    clamp_slots: int = -1


class ReferenceGateMode(Enum):
    LIVE = auto()
    HOLD = auto()
    REARMING = auto()


def copy_smpl_reference(frame: SmplReferenceFrame) -> SmplReferenceFrame:
    """Return a frame whose array storage is independent from ``frame``."""

    return replace(
        frame,
        term1_local=frame.term1_local.copy(),
        root_quat=frame.root_quat.copy(),
        wrist=frame.wrist.copy(),
        head_joint_pos=frame.head_joint_pos.copy(),
        anchor_quat=(
            None if frame.anchor_quat is None else frame.anchor_quat.copy()
        ),
    )


def _normalized_wxyz(quaternions: np.ndarray) -> np.ndarray:
    values = np.asarray(quaternions)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    safe_norms = np.where(norms > 1.0e-12, norms, 1.0)
    normalized = values / safe_norms
    identity = np.zeros_like(normalized)
    identity[..., 0] = 1.0
    return np.where(norms > 1.0e-12, normalized, identity)


def _slerp_wxyz(start, end, alpha):
    start = _normalized_wxyz(start)
    end = _normalized_wxyz(end)
    dots = np.sum(start * end, axis=-1, keepdims=True)
    end = np.where(dots < 0.0, -end, end)
    dots = np.clip(np.abs(dots), 0.0, 1.0)
    theta = np.arccos(dots)
    sin_theta = np.sin(theta)
    linear = sin_theta < 1.0e-6
    left = np.sin((1.0 - alpha) * theta) / np.where(linear, 1.0, sin_theta)
    right = np.sin(alpha * theta) / np.where(linear, 1.0, sin_theta)
    result = left * start + right * end
    result = np.where(linear, (1.0 - alpha) * start + alpha * end, result)
    return _normalized_wxyz(result)


def interpolate_smpl_reference(
    start: SmplReferenceFrame,
    end: SmplReferenceFrame,
    alpha: float,
) -> SmplReferenceFrame:
    """Blend human-reference positions and orientations in reference space."""

    value = float(alpha)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("alpha must be finite in [0, 1]")
    if (start.anchor_quat is None) != (end.anchor_quat is None):
        raise ValueError("anchor quaternion presence must match")

    anchor_quat = None
    if start.anchor_quat is not None and end.anchor_quat is not None:
        anchor_quat = _slerp_wxyz(start.anchor_quat, end.anchor_quat, value)

    return replace(
        end,
        term1_local=(1.0 - value) * start.term1_local + value * end.term1_local,
        root_quat=_slerp_wxyz(start.root_quat, end.root_quat, value),
        wrist=(1.0 - value) * start.wrist + value * end.wrist,
        head_joint_pos=(
            (1.0 - value) * start.head_joint_pos
            + value * end.head_joint_pos
        ),
        anchor_quat=anchor_quat,
    )


class LiveReferenceGate:
    def __init__(self):
        self._mode = ReferenceGateMode.LIVE
        self._observed = None
        self._observed_received_mono = 0.0
        self._latched = None
        self._rearm_progress = 0.0

    @property
    def mode(self) -> ReferenceGateMode:
        return self._mode

    @property
    def observed_reference(self) -> SmplReferenceFrame | None:
        return self._observed

    def observe(self, frame: SmplReferenceFrame, received_mono: float) -> None:
        if not np.isfinite(received_mono):
            raise ValueError("received_mono must be finite")
        self._observed = copy_smpl_reference(frame)
        self._observed_received_mono = float(received_mono)

    def has_fresh_observed(self, now_mono: float, timeout_s: float) -> bool:
        return bool(
            self._observed is not None
            and np.isfinite(now_mono)
            and np.isfinite(timeout_s)
            and timeout_s > 0.0
            and max(0.0, now_mono - self._observed_received_mono) <= timeout_s
        )

    def active_reference(self) -> SmplReferenceFrame | None:
        if self._mode is ReferenceGateMode.LIVE:
            return self._observed
        if self._mode is ReferenceGateMode.HOLD:
            return self._latched
        if self._latched is None or self._observed is None:
            return None
        return interpolate_smpl_reference(
            self._latched,
            self._observed,
            self._rearm_progress,
        )

    def hold(self) -> bool:
        active = self.active_reference()
        if active is None:
            return False
        self._latched = copy_smpl_reference(active)
        self._mode = ReferenceGateMode.HOLD
        self._rearm_progress = 0.0
        return True

    def begin_rearm(self) -> bool:
        if (
            self._mode is not ReferenceGateMode.HOLD
            or self._latched is None
            or self._observed is None
        ):
            return False
        self._mode = ReferenceGateMode.REARMING
        self._rearm_progress = 0.0
        return True

    def set_rearm_progress(self, alpha: float) -> None:
        value = float(alpha)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("rearm progress must be finite in [0, 1]")
        if self._mode is not ReferenceGateMode.REARMING:
            raise RuntimeError("reference gate is not rearming")
        self._rearm_progress = value

    def complete_rearm(self) -> None:
        if self._mode is not ReferenceGateMode.REARMING:
            raise RuntimeError("reference gate is not rearming")
        if self._observed is None:
            raise RuntimeError("reference gate has no observed reference")
        self._mode = ReferenceGateMode.LIVE
        self._latched = None
        self._rearm_progress = 0.0

    def reset(self) -> None:
        self._mode = ReferenceGateMode.LIVE
        self._observed = None
        self._observed_received_mono = 0.0
        self._latched = None
        self._rearm_progress = 0.0
