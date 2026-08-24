"""Pure bounded-jitter playout for converted ZeroLab poses."""

from collections import deque
from dataclasses import dataclass, replace
from enum import IntEnum
from numbers import Integral

import numpy as np

from .converter import ConvertedPoseFrame


class PlayoutKind(IntEnum):
    REAL = 0
    INTERPOLATED = 1
    HELD = 2
    SHORT_RECOVERY_BLEND = 3


@dataclass(frozen=True)
class ResampledPose:
    frame: ConvertedPoseFrame
    kind: PlayoutKind
    latest_real_frame_index: int
    latest_real_receive_timestamp_ns: int


@dataclass(frozen=True)
class ResamplerStats:
    interpolated_output_frames: int = 0
    held_output_frames: int = 0
    dropped_backlog_frames: int = 0


_ARRAY_SPECS = (
    ("smpl_body_pose", (21, 3)),
    ("smpl_joints", (24, 3)),
    ("body_quat_w", (4,)),
    ("joint_pos", (29,)),
)


def _immutable_array(values, expected_shape, name):
    source = np.asarray(values)
    if (
        source.shape != expected_shape
        or not np.issubdtype(source.dtype, np.number)
        or np.issubdtype(source.dtype, np.complexfloating)
    ):
        raise ValueError(f"{name} must be a real numeric array with shape {expected_shape}")
    result = np.array(source, dtype=np.float32, copy=True, order="C")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {expected_shape}")
    result.setflags(write=False)
    return result


def _validated_frame(frame):
    if not isinstance(frame, ConvertedPoseFrame):
        raise ValueError("frame must be a ConvertedPoseFrame")
    if isinstance(frame.frame_index, bool) or not isinstance(frame.frame_index, Integral):
        raise ValueError("frame_index must be an integer")
    if (
        isinstance(frame.receive_timestamp_ns, bool)
        or not isinstance(frame.receive_timestamp_ns, Integral)
    ):
        raise ValueError("receive_timestamp_ns must be an integer")
    arrays = {
        name: _immutable_array(getattr(frame, name), shape, name)
        for name, shape in _ARRAY_SPECS
    }
    if float(np.linalg.norm(arrays["body_quat_w"].astype(np.float64))) < 1e-6:
        raise ValueError("body_quat_w must have a nonzero norm")
    return ConvertedPoseFrame(
        frame_index=int(frame.frame_index),
        receive_timestamp_ns=int(frame.receive_timestamp_ns),
        **arrays,
    )


def _slerp_wxyz(start, end, fraction):
    start64 = np.asarray(start, dtype=np.float64)
    end64 = np.asarray(end, dtype=np.float64)
    start64 /= np.linalg.norm(start64)
    end64 /= np.linalg.norm(end64)
    dot = float(np.dot(start64, end64))
    if dot < 0.0:
        end64 *= -1.0
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        result = start64 + fraction * (end64 - start64)
    else:
        angle = np.arccos(dot)
        sine = np.sin(angle)
        result = (
            np.sin((1.0 - fraction) * angle) / sine * start64
            + np.sin(fraction * angle) / sine * end64
        )
    result /= np.linalg.norm(result)
    return np.ascontiguousarray(result, dtype=np.float32)


def _interpolate_frame(left, right, fraction):
    fraction = min(1.0, max(0.0, float(fraction)))
    return ConvertedPoseFrame(
        frame_index=left.frame_index,
        receive_timestamp_ns=left.receive_timestamp_ns,
        smpl_body_pose=np.ascontiguousarray(
            left.smpl_body_pose + fraction * (right.smpl_body_pose - left.smpl_body_pose),
            dtype=np.float32,
        ),
        smpl_joints=np.ascontiguousarray(
            left.smpl_joints + fraction * (right.smpl_joints - left.smpl_joints),
            dtype=np.float32,
        ),
        body_quat_w=_slerp_wxyz(left.body_quat_w, right.body_quat_w, fraction),
        joint_pos=np.ascontiguousarray(
            left.joint_pos + fraction * (right.joint_pos - left.joint_pos),
            dtype=np.float32,
        ),
    )


class ZeroLabPoseResampler:
    """Resample received converted poses onto a bounded-delay playout clock."""

    def __init__(
        self,
        *,
        jitter_buffer_seconds: float,
        short_recovery_blend_seconds: float,
        output_rate_hz: float,
    ) -> None:
        if jitter_buffer_seconds < 0.0:
            raise ValueError("jitter_buffer_seconds must be nonnegative")
        if short_recovery_blend_seconds <= 0.0:
            raise ValueError("short_recovery_blend_seconds must be positive")
        if output_rate_hz <= 0.0:
            raise ValueError("output_rate_hz must be positive")
        self._jitter_buffer_ns = int(jitter_buffer_seconds * 1_000_000_000)
        self._short_blend_ns = int(short_recovery_blend_seconds * 1_000_000_000)
        self._output_period_ns = int(1_000_000_000 / output_rate_hz)
        self.reset()

    @property
    def stats(self):
        return self._stats

    def observe(self, frame: ConvertedPoseFrame) -> bool:
        """Validate and enqueue a newer real pose, ignoring exact duplicates."""
        frame = _validated_frame(frame)
        if self._latest_real_frame_index is not None:
            if (
                frame.frame_index == self._latest_real_frame_index
                and frame.receive_timestamp_ns == self._latest_real_receive_timestamp_ns
            ):
                return False
            if frame.frame_index <= self._latest_real_frame_index:
                raise ValueError("frame_index must be strictly increasing")
            if frame.receive_timestamp_ns < self._latest_real_receive_timestamp_ns:
                raise ValueError("receive_timestamp_ns must be monotonic")
        self._frames.append(frame)
        self._latest_real_frame_index = frame.frame_index
        self._latest_real_receive_timestamp_ns = frame.receive_timestamp_ns
        return True

    def sample(self, now_ns: int) -> ResampledPose | None:
        """Emit at most one 50 Hz playout frame for the current clock tick."""
        if isinstance(now_ns, bool) or not isinstance(now_ns, Integral):
            raise ValueError("now_ns must be an integer")
        now_ns = int(now_ns)
        if (
            self._last_output_now_ns is not None
            and now_ns - self._last_output_now_ns < self._output_period_ns
        ):
            return None
        target_ns = now_ns - self._jitter_buffer_ns
        target_frame, target_kind, left = self._select_target(target_ns)
        if left is not None:
            self._discard_before(left)
        if target_frame is None:
            if self._last_output is None:
                return None
            return self._emit(
                self._last_output.frame, PlayoutKind.HELD, target_ns, now_ns
            )

        if self._catchup_start is None and self._last_output is not None:
            if self._last_output.kind is PlayoutKind.HELD and not self._suppress_short_recovery:
                self._catchup_start = self._last_output.frame
                self._catchup_started_ns = now_ns
        if self._suppress_short_recovery and target_kind in (
            PlayoutKind.REAL,
            PlayoutKind.INTERPOLATED,
        ):
            self._suppress_short_recovery = False

        if self._catchup_start is not None:
            alpha = min(
                1.0,
                (now_ns - self._catchup_started_ns) / self._short_blend_ns,
            )
            pose = _interpolate_frame(self._catchup_start, target_frame, alpha)
            kind = (
                PlayoutKind.SHORT_RECOVERY_BLEND
                if alpha < 1.0
                else target_kind
            )
            if alpha == 1.0:
                self._catchup_start = None
                self._catchup_started_ns = None
            return self._emit(pose, kind, target_ns, now_ns)
        return self._emit(target_frame, target_kind, target_ns, now_ns)

    def mark_stale(self) -> None:
        """Drop live brackets while retaining the last output as a diagnostic hold."""
        self._frames.clear()
        self._catchup_start = None
        self._catchup_started_ns = None
        self._suppress_short_recovery = True

    def reset(self) -> None:
        """Clear all state, including stream identity and diagnostics."""
        self._frames = deque()
        self._last_output = None
        self._latest_real_frame_index = None
        self._latest_real_receive_timestamp_ns = None
        self._catchup_start = None
        self._catchup_started_ns = None
        self._suppress_short_recovery = False
        self._next_output_index = 0
        self._last_output_now_ns = None
        self._stats = ResamplerStats()

    def _select_target(self, target_ns):
        if not self._frames:
            return None, PlayoutKind.HELD, None
        first = self._frames[0]
        if target_ns < first.receive_timestamp_ns:
            return None, PlayoutKind.HELD, None
        left = first
        for right in list(self._frames)[1:]:
            if target_ns == right.receive_timestamp_ns:
                return right, PlayoutKind.REAL, right
            if target_ns < right.receive_timestamp_ns:
                if target_ns == left.receive_timestamp_ns:
                    return left, PlayoutKind.REAL, left
                span = right.receive_timestamp_ns - left.receive_timestamp_ns
                if span == 0:
                    return right, PlayoutKind.REAL, right
                return (
                    _interpolate_frame(
                        left,
                        right,
                        (target_ns - left.receive_timestamp_ns) / span,
                    ),
                    PlayoutKind.INTERPOLATED,
                    left,
                )
            left = right
        if target_ns == left.receive_timestamp_ns:
            return left, PlayoutKind.REAL, left
        if self._catchup_start is not None:
            return left, PlayoutKind.HELD, left
        if self._last_output is None:
            return left, PlayoutKind.HELD, left
        return None, PlayoutKind.HELD, left

    def _discard_before(self, left):
        discarded = 0
        while self._frames and self._frames[0] is not left:
            self._frames.popleft()
            discarded += 1
        if discarded:
            self._stats = replace(
                self._stats,
                dropped_backlog_frames=self._stats.dropped_backlog_frames + discarded,
            )

    def _emit(self, source, kind, target_ns, now_ns):
        frame = ConvertedPoseFrame(
            frame_index=self._next_output_index,
            receive_timestamp_ns=target_ns,
            smpl_body_pose=_immutable_array(source.smpl_body_pose, (21, 3), "smpl_body_pose"),
            smpl_joints=_immutable_array(source.smpl_joints, (24, 3), "smpl_joints"),
            body_quat_w=_immutable_array(source.body_quat_w, (4,), "body_quat_w"),
            joint_pos=_immutable_array(source.joint_pos, (29,), "joint_pos"),
        )
        self._next_output_index += 1
        if kind is PlayoutKind.INTERPOLATED:
            self._stats = replace(
                self._stats,
                interpolated_output_frames=self._stats.interpolated_output_frames + 1,
            )
        elif kind is PlayoutKind.HELD:
            self._stats = replace(
                self._stats,
                held_output_frames=self._stats.held_output_frames + 1,
            )
        output = ResampledPose(
            frame=frame,
            kind=kind,
            latest_real_frame_index=self._latest_real_frame_index,
            latest_real_receive_timestamp_ns=self._latest_real_receive_timestamp_ns,
        )
        self._last_output = output
        self._last_output_now_ns = now_ns
        return output
