"""ZeroLab pose-window state and the live ROS/ZMQ source node."""

from collections import deque
from collections.abc import Mapping
from datetime import datetime, timezone
import math
import operator
from pathlib import Path
import time

import numpy as np
import zmq

from rclpy.node import Node

from bxi_example_py_elf3.framework.mod_api import NodeBuildContext

from .converter import ConvertedPoseFrame, ZeroLabMotionConverter
from .protocol import ZeroLabProtocolError, parse_zerolab_packet
from .recording import (
    RawRecord,
    RawRecordingWriter,
    build_recording_metadata,
)
from .udp_receiver import ZeroLabUdpReceiver

if __package__ == "zerolab":
    from pico.zmq_messages import pack_pose_message
else:
    from ..pico.zmq_messages import pack_pose_message


SOURCE_DEFAULTS: dict[str, object] = {
    "udp_bind_host": "0.0.0.0",
    "udp_port": 18000,
    "allowed_sender": "",
    "pose_host": "127.0.0.1",
    "pose_port": 5558,
    "pose_topic": "pose",
    "rate_hz": 50.0,
    "window_frames": 10,
    "stale_seconds": 0.5,
    "record_path": "",
}


def validate_source_params(
    raw: Mapping[str, object], *, mod_root: Path | None = None
) -> dict[str, object]:
    """Merge and strictly validate ZeroLab source configuration."""
    unknown = set(raw) - set(SOURCE_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown ZeroLab source params: {sorted(unknown)}")
    params = {
        name: raw.get(name, default)
        for name, default in SOURCE_DEFAULTS.items()
    }

    for name in ("udp_bind_host", "pose_topic"):
        if not isinstance(params[name], str) or not params[name]:
            raise ValueError(f"{name} must be a non-empty string")
    if params["pose_host"] != "127.0.0.1":
        raise ValueError("pose_host must be 127.0.0.1")
    for name in ("allowed_sender", "record_path"):
        if not isinstance(params[name], str):
            raise ValueError(f"{name} must be a string")

    for name in ("udp_port", "pose_port"):
        value = params[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= 65535
        ):
            raise ValueError(f"{name} must be an integer from 1 to 65535")

    rate_hz = params["rate_hz"]
    if (
        isinstance(rate_hz, bool)
        or not isinstance(rate_hz, (int, float))
        or not math.isfinite(rate_hz)
        or float(rate_hz) != 50.0
    ):
        raise ValueError("rate_hz must be exactly 50.0")
    window_frames = params["window_frames"]
    if (
        isinstance(window_frames, bool)
        or not isinstance(window_frames, int)
        or window_frames != 10
    ):
        raise ValueError("window_frames must be exactly 10")
    stale_seconds = params["stale_seconds"]
    if (
        isinstance(stale_seconds, bool)
        or not isinstance(stale_seconds, (int, float))
        or not math.isfinite(stale_seconds)
        or stale_seconds <= 0
    ):
        raise ValueError("stale_seconds must be greater than zero")

    record_path = params["record_path"]
    if record_path and mod_root is not None:
        recording = Path(record_path).expanduser().resolve()
        root = Path(mod_root).expanduser().resolve()
        if recording == root or root in recording.parents:
            raise ValueError("record_path must resolve outside mod_root")
    return params


_POSE_FIELDS = {
    "smpl_body_pose": (21, 3),
    "smpl_joints": (24, 3),
    "body_quat_w": (4,),
    "joint_pos": (29,),
}


def _as_finite_float32(name, values, expected_shape):
    array = np.asarray(values)
    if (
        array.shape != expected_shape
        or not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.complexfloating)
    ):
        raise ValueError(
            f"{name} must use a real numeric dtype with shape {expected_shape}"
        )
    result = np.ascontiguousarray(array, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {expected_shape}")
    return result


class PoseChunkWindow:
    """Build a rolling fixed-size SONIC pose chunk from converted frames."""

    def __init__(self, window_frames: int = 10) -> None:
        if isinstance(window_frames, bool) or window_frames < 1:
            raise ValueError("window_frames must be at least 1")
        self._window_frames = int(window_frames)
        self._frames = deque(maxlen=self._window_frames)
        self._last_frame_index = None

    @property
    def ready(self) -> bool:
        return len(self._frames) == self._window_frames

    def clear(self) -> None:
        self._frames.clear()
        self._last_frame_index = None

    def append(self, frame: ConvertedPoseFrame):
        try:
            frame_index = operator.index(frame.frame_index)
        except TypeError as error:
            raise ValueError("frame_index must be an integer") from error
        if isinstance(frame.frame_index, bool):
            raise ValueError("frame_index must be an integer")

        validated = {
            name: _as_finite_float32(name, getattr(frame, name), shape)
            for name, shape in _POSE_FIELDS.items()
        }
        if self._last_frame_index is not None:
            delta = frame_index - self._last_frame_index
            if delta == 0:
                return None
            if delta < 0:
                self.clear()
                raise ValueError("frame_index moved backward")
            if delta > 1:
                self.clear()
        self._frames.append((frame_index, validated))
        self._last_frame_index = frame_index
        if not self.ready:
            return None

        frames = tuple(self._frames)
        return {
            "frame_index": np.asarray(
                [frame_index for frame_index, _ in frames], dtype=np.int64
            ),
            "smpl_joints": np.stack(
                [values["smpl_joints"] for _, values in frames]
            ).astype(np.float32),
            "body_quat_w": np.stack(
                [values["body_quat_w"] for _, values in frames]
            ).astype(np.float32),
            "joint_pos": np.stack(
                [values["joint_pos"] for _, values in frames]
            ).astype(np.float32),
            "head_joint_pos": np.zeros(
                (len(frames), 2), dtype=np.float32
            ),
            "stream_mode": np.array([1], dtype=np.int32),
            "calibration_ready": np.array([True], dtype=bool),
        }


class ZeroLabSourceCore:
    """Apply stale-stream semantics around ZeroLab conversion and chunking."""

    def __init__(
        self,
        converter: ZeroLabMotionConverter,
        window_frames: int = 10,
        stale_seconds: float = 0.5,
    ) -> None:
        if not np.isfinite(stale_seconds) or stale_seconds < 0.0:
            raise ValueError("stale_seconds must be finite and non-negative")
        self._converter = converter
        self._window = PoseChunkWindow(window_frames)
        self._stale_ns = int(float(stale_seconds) * 1_000_000_000)
        self._last_timestamp_ns = None
        self._stale_handled = False
        self._stale_event_pending = False

    def _mark_stale(self) -> None:
        self._window.clear()
        self._converter.mark_stale()
        self._stale_handled = True
        self._stale_event_pending = True

    def consume_stale_event(self) -> bool:
        """Return and clear a pending stale transition."""
        pending = self._stale_event_pending
        self._stale_event_pending = False
        return pending

    def accept(self, packet):
        timestamp_ns = operator.index(packet.receive_timestamp_ns)
        if (
            self._last_timestamp_ns is not None
            and not self._stale_handled
            and timestamp_ns - self._last_timestamp_ns > self._stale_ns
        ):
            self._mark_stale()

        frame = self._converter.observe(packet)
        self._last_timestamp_ns = timestamp_ns
        self._stale_handled = False
        return self._window.append(frame)

    def check_stale(self, now_ns: int) -> bool:
        now_timestamp_ns = operator.index(now_ns)
        if (
            self._last_timestamp_ns is None
            or self._stale_handled
            or now_timestamp_ns - self._last_timestamp_ns <= self._stale_ns
        ):
            return False
        self._mark_stale()
        return True


class ZeroLabPosePublisher:
    """Own the PUB socket used only for the ZeroLab ``pose`` stream."""

    def __init__(self, host: str, port: int, topic: str) -> None:
        self._topic = topic
        self._context = zmq.Context()
        self._socket = None
        self._closed = False
        try:
            self._socket = self._context.socket(zmq.PUB)
            self._socket.setsockopt(zmq.LINGER, 0)
            self._socket.setsockopt(zmq.SNDHWM, 2)
            self._socket.bind(f"tcp://{host}:{port}")
        except Exception:
            if self._socket is not None:
                try:
                    self._socket.close(linger=0)
                finally:
                    self._context.term()
            else:
                self._context.term()
            self._closed = True
            raise

    def send(self, fields) -> None:
        message = pack_pose_message(fields, topic=self._topic)
        self._socket.send(message, flags=zmq.NOBLOCK)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._socket.close(linger=0)
        finally:
            self._context.term()


class ZeroLabSourceNode(Node):
    """Receive ZeroLab UDP frames and publish PICO-compatible pose chunks."""

    def __init__(self, context: NodeBuildContext) -> None:
        params = validate_source_params(
            context.params, mod_root=context.mod_root
        )
        super().__init__(
            context.node_name, namespace=context.namespace or None
        )

        self._receiver = None
        self._converter = None
        self._core = None
        self._publisher = None
        self._timer = None
        self._writer = None
        self._record_path = str(params["record_path"])
        self._recording_enabled = bool(self._record_path)
        self._recording_error_reported = False
        self._invalid_packets = 0
        self._dropped_publications = 0
        self._stream_state = None
        self._closed = False
        self._destroy_result = True

        try:
            allowed_sender = str(params["allowed_sender"])
            self._receiver = ZeroLabUdpReceiver(
                bind_host=str(params["udp_bind_host"]),
                port=int(params["udp_port"]),
                allowed_sender_host=allowed_sender or None,
            )
            self._converter = ZeroLabMotionConverter()
            self._core = ZeroLabSourceCore(
                self._converter,
                window_frames=int(params["window_frames"]),
                stale_seconds=float(params["stale_seconds"]),
            )
            self._publisher = ZeroLabPosePublisher(
                str(params["pose_host"]),
                int(params["pose_port"]),
                str(params["pose_topic"]),
            )
            self._timer = self.create_timer(
                1.0 / float(params["rate_hz"]), self._tick
            )
        except Exception:
            self._cleanup_failed_construction()
            raise

    def _cleanup_failed_construction(self) -> None:
        if self._timer is not None:
            try:
                self.destroy_timer(self._timer)
            except Exception:
                pass
            self._timer = None
        if self._publisher is not None:
            try:
                self._publisher.close()
            except Exception:
                pass
            self._publisher = None
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass
            self._writer = None
        if self._converter is not None:
            try:
                self._converter.reset_session()
            except Exception:
                pass
        if self._receiver is not None:
            try:
                self._receiver.close()
            except Exception:
                pass
            self._receiver = None
        self._closed = True
        super().destroy_node()

    def _disable_recording(self, exc: Exception) -> None:
        if not self._recording_error_reported:
            self.get_logger().warning(
                f"ZeroLab recording disabled after error: {exc}"
            )
            self._recording_error_reported = True
        writer, self._writer = self._writer, None
        self._recording_enabled = False
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass

    def _record_valid_packet_if_enabled(self, packet) -> None:
        if not self._recording_enabled:
            return
        try:
            if self._writer is None:
                metadata = build_recording_metadata(
                    packet.sender_address,
                    datetime.now(timezone.utc).isoformat(),
                )
                self._writer = RawRecordingWriter(
                    Path(self._record_path), metadata
                )
            self._writer.append(
                RawRecord(
                    packet.receive_timestamp_ns,
                    packet.local_frame_index,
                    packet.raw_payload,
                )
            )
        except Exception as exc:
            self._disable_recording(exc)

    def _set_stream_state(
        self, state: str, *, frame_index: int | None = None
    ) -> None:
        if state == self._stream_state:
            return
        self._stream_state = state
        if state == "collecting":
            self.get_logger().info("ZeroLab waiting for 10-frame stream window")
        elif state == "ready":
            self.get_logger().info(
                f"ZeroLab stream ready; frame={frame_index}"
            )
        elif state == "stale":
            self.get_logger().warning(
                "ZeroLab input stale; live pose publication stopped"
            )

    def _tick(self) -> None:
        if self._closed:
            return
        latest_fields = None
        received_valid = False
        became_stale = False
        for datagram in self._receiver.drain():
            try:
                packet = parse_zerolab_packet(
                    datagram.payload,
                    receive_timestamp_ns=datagram.receive_timestamp_ns,
                    local_frame_index=datagram.local_frame_index,
                    sender_address=datagram.sender_address,
                )
            except ZeroLabProtocolError as exc:
                self._invalid_packets += 1
                self.get_logger().warning(
                    f"skipped invalid ZeroLab packet: {exc}"
                )
                continue
            received_valid = True
            self._record_valid_packet_if_enabled(packet)
            fields = self._core.accept(packet)
            if self._core.consume_stale_event():
                became_stale = True
                latest_fields = None
            if fields is not None:
                latest_fields = fields

        self._core.check_stale(time.monotonic_ns())
        if self._core.consume_stale_event():
            became_stale = True
            latest_fields = None
        if became_stale:
            self._set_stream_state("stale")
        if latest_fields is not None:
            try:
                self._publisher.send(latest_fields)
            except zmq.Again:
                self._dropped_publications += 1

        if latest_fields is not None:
            self._set_stream_state(
                "ready", frame_index=int(latest_fields["frame_index"][-1])
            )
        elif (
            not became_stale
            and (
                self._stream_state is None
                or (received_valid and self._stream_state == "stale")
            )
        ):
            self._set_stream_state("collecting")

    def _close_with_warning(self, name: str, close_resource) -> None:
        try:
            close_resource()
        except Exception as exc:
            self.get_logger().warning(f"failed to close {name}: {exc}")

    def destroy_node(self):
        if self._closed:
            return self._destroy_result
        self._closed = True
        if self._timer is not None:
            timer, self._timer = self._timer, None
            self._close_with_warning(
                "ZeroLab timer", lambda: self.destroy_timer(timer)
            )
        if self._receiver is not None:
            receiver, self._receiver = self._receiver, None
            self._close_with_warning("ZeroLab UDP receiver", receiver.close)
        if self._writer is not None:
            writer, self._writer = self._writer, None
            self._close_with_warning("ZeroLab recording", writer.close)
        if self._publisher is not None:
            publisher, self._publisher = self._publisher, None
            self._close_with_warning("ZeroLab pose publisher", publisher.close)
        if self._converter is not None:
            self._close_with_warning(
                "ZeroLab converter session", self._converter.reset_session
            )
        self._destroy_result = super().destroy_node()
        return self._destroy_result


def create_node(context: NodeBuildContext) -> ZeroLabSourceNode:
    return ZeroLabSourceNode(context)


__all__ = [
    "PoseChunkWindow",
    "SOURCE_DEFAULTS",
    "ZeroLabPosePublisher",
    "ZeroLabSourceCore",
    "ZeroLabSourceNode",
    "create_node",
    "validate_source_params",
]
