#!/usr/bin/env python3
"""Read PICO body tracking, retarget it to ELF3, and publish local references."""

from __future__ import annotations

import argparse
import json
import secrets
import signal
import socket
import sys
from threading import Event, Thread
import time
from pathlib import Path

import numpy as np
import xrobotoolkit_sdk as xrt

from gmr import (
    PicoGmrRetargeter,
    quat_conjugate_wxyz,
    quat_multiply_wxyz,
    quaternion_rotation_vector_wxyz,
    unity_pose_to_right_handed,
)
from protocol import (
    LiveReferenceFrame,
    REFERENCE_JOINT_NAMES,
    REFERENCE_WINDOW_SIZE,
    encode_reference_frame,
)
from tracking_gate import TrackingGate
from xrt_session import XrtBackgroundSession
from xrt_service import ManagedRoboticsService


BODY_JOINT_NAMES = (
    "Pelvis",
    "Left_Hip",
    "Right_Hip",
    "Spine1",
    "Left_Knee",
    "Right_Knee",
    "Spine2",
    "Left_Ankle",
    "Right_Ankle",
    "Spine3",
    "Left_Foot",
    "Right_Foot",
    "Neck",
    "Left_Collar",
    "Right_Collar",
    "Head",
    "Left_Shoulder",
    "Right_Shoulder",
    "Left_Elbow",
    "Right_Elbow",
    "Left_Wrist",
    "Right_Wrist",
    "Left_Hand",
    "Right_Hand",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5568)
    parser.add_argument("--rate-hz", type=float, default=50.0)
    parser.add_argument("--human-height", type=float, default=1.75)
    parser.add_argument(
        "--capture-frame",
        type=Path,
        help="write the first converted PICO/GMR frame to JSON for offline diagnosis",
    )
    return parser.parse_args()


def _model_paths() -> tuple[Path, Path]:
    mod_root = Path(__file__).resolve().parent
    package_share = mod_root.parents[1]
    return (
        package_share / "data" / "mujoco_simulation" / "elf3.xml",
        mod_root / "assets" / "pico_to_elf3.json",
    )


def _read_body_frame() -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], int] | None:
    if not xrt.is_body_data_available():
        return None
    poses = xrt.get_body_joints_pose()
    if len(poses) < len(BODY_JOINT_NAMES):
        raise ValueError(
            f"PICO returned {len(poses)} body joints, expected {len(BODY_JOINT_NAMES)}"
        )
    result = {}
    for name, raw in zip(BODY_JOINT_NAMES, poses):
        if len(raw) < 7:
            raise ValueError(f"PICO body {name} has only {len(raw)} pose values")
        result[name] = unity_pose_to_right_handed(
            raw[0:3],
            (raw[6], raw[3], raw[4], raw[5]),
        )
    source_ns = int(xrt.get_body_timestamp_ns())
    return result, source_ns


def _root_velocity(
    previous: tuple[int, float, np.ndarray, np.ndarray] | None,
    source_ns: int,
    now: float,
    position: np.ndarray,
    quaternion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if previous is None:
        return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
    previous_ns, previous_time, previous_position, previous_quaternion = previous
    dt = (source_ns - previous_ns) * 1.0e-9
    if not 0.002 <= dt <= 0.2:
        dt = now - previous_time
    if not 0.002 <= dt <= 0.2:
        return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
    linear = (position - previous_position) / dt
    delta_q = quat_multiply_wxyz(quaternion, quat_conjugate_wxyz(previous_quaternion))
    angular = quaternion_rotation_vector_wxyz(delta_q) / dt
    return linear.astype(np.float32), angular.astype(np.float32)


def _capture_diagnostic_frame(
    path: Path,
    human_data: dict[str, tuple[np.ndarray, np.ndarray]],
    joints: np.ndarray,
    source_ns: int,
) -> None:
    payload = {
        "format": "bxi_pico_gmr_frame_v1",
        "coordinate_system": "mocaplab_unity_to_right_handed",
        "source_timestamp_ns": int(source_ns),
        "human_data": {
            name: {
                "position": np.asarray(pose[0], dtype=float).tolist(),
                "quaternion_wxyz": np.asarray(pose[1], dtype=float).tolist(),
            }
            for name, pose in human_data.items()
        },
        "gmr_joint_names": list(REFERENCE_JOINT_NAMES),
        "gmr_joint_pos": np.asarray(joints, dtype=float).tolist(),
    }
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


def _run_controller_loop(
    *,
    stop_event: Event,
    tracking_gate: TrackingGate,
    rate_hz: float = 20.0,
) -> None:
    period = 1.0 / rate_hz
    next_tick = time.monotonic()
    read_errors = 0
    last_buttons: tuple[bool, bool] | None = None
    while not stop_event.is_set():
        now = time.monotonic()
        if now < next_tick:
            stop_event.wait(min(next_tick - now, 0.01))
            continue
        next_tick = max(next_tick + period, now)
        try:
            a_pressed = bool(xrt.get_A_button())
            x_pressed = bool(xrt.get_X_button())
            buttons = (a_pressed, x_pressed)
            if buttons != last_buttons:
                if last_buttons is not None or a_pressed or x_pressed:
                    print(
                        "PICO controller buttons: "
                        f"A={'DOWN' if a_pressed else 'UP'}, "
                        f"X={'DOWN' if x_pressed else 'UP'}",
                        flush=True,
                    )
                if a_pressed and x_pressed:
                    print(
                        "PICO A+X raw combo detected by XRoboToolkit",
                        flush=True,
                    )
                last_buttons = buttons
            changed = tracking_gate.update(a_pressed, x_pressed)
            read_errors = 0
        except Exception as exc:
            read_errors += 1
            if read_errors == 1 or read_errors % 100 == 0:
                print(
                    f"PICO controller input unavailable ({read_errors}): {exc}",
                    file=sys.stderr,
                    flush=True,
                )
            continue
        if changed is True:
            print(
                "PICO A+X: live GMR tracking enabled; collecting 21 fresh frames",
                flush=True,
            )
        elif changed is False:
            print(
                "PICO A+X: live GMR tracking disabled; RGMT standing reference active",
                flush=True,
            )


def _run_reference_loop(
    *,
    stop_event: Event,
    tracking_gate: TrackingGate,
    retargeter: PicoGmrRetargeter,
    udp: socket.socket,
    destination: tuple[str, int],
    rate_hz: float,
    capture_frame: Path | None,
) -> None:
    session_id = 0
    sequence = 0
    previous_source_ns: int | None = None
    previous_root = None
    was_enabled = False
    period = 1.0 / rate_hz
    next_tick = time.monotonic()
    invalid_frames = 0
    static_timestamp_frames = 0
    capture_written = False

    while not stop_event.is_set():
        now = time.monotonic()
        if now < next_tick:
            stop_event.wait(min(next_tick - now, 0.005))
            continue
        next_tick = max(next_tick + period, now)
        enabled = tracking_gate.is_enabled()
        if not enabled:
            was_enabled = False
            continue
        if not was_enabled:
            session_id = secrets.randbits(64)
            sequence = 0
            previous_source_ns = None
            previous_root = None
            invalid_frames = 0
            static_timestamp_frames = 0
            retargeter.reset()
            was_enabled = True
        try:
            body_frame = _read_body_frame()
            if body_frame is None:
                continue
            if not tracking_gate.is_enabled():
                was_enabled = False
                continue
            human_data, source_ns = body_frame
            if source_ns == previous_source_ns:
                static_timestamp_frames += 1
                if static_timestamp_frames == 1:
                    print(
                        "PICO body timestamp is static; continuing with local 50 Hz "
                        "sequence and monotonic velocity timing",
                        flush=True,
                    )
            else:
                static_timestamp_frames = 0
            joints, root_position, root_quaternion = retargeter.retarget(
                human_data,
                dt=period,
            )
            if capture_frame is not None and not capture_written:
                _capture_diagnostic_frame(
                    capture_frame,
                    human_data,
                    joints,
                    source_ns,
                )
                print(f"Captured converted PICO frame: {capture_frame.resolve()}", flush=True)
                capture_written = True
            sample_time = time.monotonic()
            linear_velocity, angular_velocity = _root_velocity(
                previous_root,
                source_ns,
                sample_time,
                root_position,
                root_quaternion,
            )
            frame = LiveReferenceFrame(
                session_id=session_id,
                sequence=sequence,
                source_timestamp_ns=source_ns,
                joint_pos=joints,
                anchor_quat_wxyz=root_quaternion,
                anchor_lin_vel_w=linear_velocity,
                anchor_ang_vel_w=angular_velocity,
            )
            udp.sendto(encode_reference_frame(frame), destination)
            if sequence == 0:
                left_elbow = joints[REFERENCE_JOINT_NAMES.index("l_elbow_y_joint")]
                right_elbow = joints[REFERENCE_JOINT_NAMES.index("r_elbow_y_joint")]
                print(
                    f"PICO GMR stream active: {len(REFERENCE_JOINT_NAMES)} joints, "
                    f"{rate_hz:.1f} Hz -> udp://{destination[0]}:{destination[1]}; "
                    f"first elbows: left={np.degrees(left_elbow):.2f} deg, "
                    f"right={np.degrees(right_elbow):.2f} deg"
                )
            if sequence + 1 == REFERENCE_WINDOW_SIZE:
                print(
                    f"PICO GMR source window ready: {REFERENCE_WINDOW_SIZE} frames sent",
                    flush=True,
                )
            previous_source_ns = source_ns
            previous_root = (
                source_ns,
                sample_time,
                root_position.copy(),
                root_quaternion.copy(),
            )
            sequence += 1
            invalid_frames = 0
        except Exception as exc:
            invalid_frames += 1
            if invalid_frames == 1 or invalid_frames % 100 == 0:
                print(
                    f"PICO GMR frame rejected ({invalid_frames}): {exc}",
                    file=sys.stderr,
                )


def main() -> int:
    args = _parse_args()
    if not 0 < args.port < 65536 or not np.isfinite(args.rate_hz) or args.rate_hz <= 0.0:
        print("invalid host UDP port or rate", file=sys.stderr)
        return 64
    model_xml, config_json = _model_paths()
    if not model_xml.is_file() or not config_json.is_file():
        print(f"missing GMR asset: {model_xml} or {config_json}", file=sys.stderr)
        return 78

    try:
        retargeter = PicoGmrRetargeter(
            model_xml,
            config_json,
            REFERENCE_JOINT_NAMES,
            actual_human_height=args.human_height,
        )
    except Exception as exc:
        print(f"cannot initialize PICO GMR: {exc}", file=sys.stderr)
        return 78

    stop_event = Event()

    def stop(_signum, _frame) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    destination = (args.host, args.port)
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    service = ManagedRoboticsService()
    xrt_session = XrtBackgroundSession(xrt)
    tracking_gate = TrackingGate()
    workers: list[Thread] = []
    exit_code = 0
    try:
        try:
            service.start()
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"XRoboToolkit PC Service启动失败: {exc}", file=sys.stderr)
            return 78
        xrt_session.start()
        print(
            "XRoboToolkit初始化已进入后台线程，正在等待PICO body tracking；"
            "此方式兼容阻塞式xrt.init()"
        )
        print("PICO GMR当前使用站立平衡参考；同时按A+X开始/停止实时跟踪")
        controller_worker = Thread(
            target=_run_controller_loop,
            kwargs={
                "stop_event": stop_event,
                "tracking_gate": tracking_gate,
            },
            name="pico-gmr-controller-loop",
            daemon=True,
        )
        reference_worker = Thread(
            target=_run_reference_loop,
            kwargs={
                "stop_event": stop_event,
                "tracking_gate": tracking_gate,
                "retargeter": retargeter,
                "udp": udp,
                "destination": destination,
                "rate_hz": args.rate_hz,
                "capture_frame": args.capture_frame,
            },
            name="pico-gmr-reference-loop",
            daemon=True,
        )
        workers.extend((controller_worker, reference_worker))
        for worker in workers:
            worker.start()
        while not stop_event.wait(timeout=0.1):
            if xrt_session.error is not None:
                print(
                    f"XRoboToolkit初始化失败: {xrt_session.error}",
                    file=sys.stderr,
                )
                exit_code = 78
                stop_event.set()
    finally:
        stop_event.set()
        try:
            xrt_session.close()
        except Exception as exc:
            print(f"XRoboToolkit关闭失败: {exc}", file=sys.stderr)
        for worker in workers:
            worker.join(timeout=3.0)
            if worker.is_alive():
                print(f"{worker.name}未能及时退出", file=sys.stderr)
        udp.close()
        service.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
