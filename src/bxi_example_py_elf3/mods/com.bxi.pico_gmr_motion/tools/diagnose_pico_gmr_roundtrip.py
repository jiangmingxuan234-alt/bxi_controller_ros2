#!/usr/bin/env python3
"""Diagnose ELF3 FK -> pseudo-PICO -> GMR joint round-trip errors.

The reference pose comes from the RGMT ONNX ``policy_default_joint_pos``
metadata used by ``com.bxi.pico_gmr_motion`` in standing mode.  The script:

1. applies that named 29-joint pose to the current ELF3 MuJoCo model;
2. computes the task-body poses with MuJoCo forward kinematics;
3. analytically inverts ``pico_to_elf3.json`` into a pseudo-PICO frame;
4. retargets the pseudo frame from both a cold and reference-pose warm start;
5. reports joint and task-space round-trip errors.

Use ``--view standing``, ``--view cold`` or ``--view warm`` to put the chosen
joint vector directly into MuJoCo for visual inspection.  The viewer never
runs the RGMT policy and therefore separates GMR/model errors from policy
tracking errors.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import socket
import sys
import tempfile
import time
from typing import Iterable

import mujoco
import numpy as np
import onnx


MOD_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = MOD_ROOT.parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(MOD_ROOT))

import gmr  # noqa: E402
import protocol  # noqa: E402


ARM_JOINTS = (
    "l_shoulder_y_joint",
    "l_shoulder_x_joint",
    "l_shoulder_z_joint",
    "l_elbow_y_joint",
    "l_wrist_x_joint",
    "l_wrist_y_joint",
    "l_wrist_z_joint",
    "r_shoulder_y_joint",
    "r_shoulder_x_joint",
    "r_shoulder_z_joint",
    "r_elbow_y_joint",
    "r_wrist_x_joint",
    "r_wrist_y_joint",
    "r_wrist_z_joint",
)

CALIBRATION_CANDIDATES = (
    "current",
    "previous_identity",
    "legacy_hybrid",
    "mocaplab_backup_right_arm",
    "right_xml_frame_corrected",
    "all_arm_xml_frame_corrected",
)


def _metadata(path: Path) -> dict[str, str]:
    model = onnx.load(str(path), load_external_data=False)
    return {item.key: item.value for item in model.metadata_props}


def _csv_floats(value: str, *, key: str) -> np.ndarray:
    result = np.asarray([float(item.strip()) for item in value.split(",")], dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"ONNX metadata {key!r} contains non-finite values")
    return result


def _csv_names(value: str, *, key: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if len(set(result)) != len(result):
        raise ValueError(f"ONNX metadata {key!r} contains duplicate names")
    return result


def load_standing_reference(model_path: Path) -> np.ndarray:
    metadata = _metadata(model_path)
    names = _csv_names(metadata["joint_names"], key="joint_names")
    key = (
        "policy_default_joint_pos"
        if "policy_default_joint_pos" in metadata
        else "default_joint_pos"
    )
    values = _csv_floats(metadata[key], key=key)
    if values.shape != (len(names),):
        raise ValueError(f"{key} has {values.size} values for {len(names)} joints")
    by_name = dict(zip(names, values))
    missing = set(protocol.REFERENCE_JOINT_NAMES) - set(by_name)
    if missing:
        raise ValueError("standing reference is missing joints: " + ", ".join(sorted(missing)))
    return np.asarray([by_name[name] for name in protocol.REFERENCE_JOINT_NAMES])


def _joint_qpos_addresses(model: mujoco.MjModel, names: Iterable[str]) -> np.ndarray:
    addresses = []
    for name in names:
        joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
        if joint_id < 0:
            raise ValueError(f"MuJoCo model is missing joint {name!r}")
        addresses.append(int(model.jnt_qposadr[joint_id]))
    return np.asarray(addresses, dtype=np.intp)


def set_named_joint_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_positions: np.ndarray,
) -> None:
    values = np.asarray(joint_positions, dtype=np.float64)
    if values.shape != (len(protocol.REFERENCE_JOINT_NAMES),):
        raise ValueError(f"joint pose has shape {values.shape}, expected (29,)")
    data.qpos[_joint_qpos_addresses(model, protocol.REFERENCE_JOINT_NAMES)] = values
    mujoco.mj_forward(model, data)


def _body_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
    if body_id < 0:
        raise ValueError(f"MuJoCo model is missing body {name!r}")
    return data.xpos[body_id].copy(), gmr.normalize_quaternion_wxyz(data.xquat[body_id])


def _task_definitions(config: dict[str, object]) -> dict[str, tuple[object, ...]]:
    definitions: dict[str, tuple[object, ...]] = {}
    for table_name in ("ik_match_table1", "ik_match_table2"):
        if not bool(config.get("use_" + table_name, True)):
            continue
        table = config[table_name]
        if not isinstance(table, dict):
            raise ValueError(f"{table_name} must be an object")
        for robot_body, raw in table.items():
            definition = tuple(raw)
            previous = definitions.setdefault(str(robot_body), definition)
            if previous[0] != definition[0] or previous[3:] != definition[3:]:
                raise ValueError(f"inconsistent task mapping for {robot_body!r}")
    return definitions


def pseudo_pico_from_fk(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config: dict[str, object],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    definitions = _task_definitions(config)
    human_root = str(config["human_root_name"])
    assumption = float(config["human_height_assumption"])
    scale = {
        str(name): float(value) * (assumption / assumption)
        for name, value in config["human_scale_table"].items()
    }

    scaled_positions: dict[str, np.ndarray] = {}
    quaternions: dict[str, np.ndarray] = {}
    for robot_body, definition in definitions.items():
        human_body = str(definition[0])
        position_offset = np.asarray(definition[3], dtype=np.float64)
        rotation_offset = gmr.normalize_quaternion_wxyz(definition[4])
        robot_position, robot_quaternion = _body_pose(model, data, robot_body)
        scaled_position = robot_position - gmr.rotate_vector_wxyz(
            robot_quaternion,
            position_offset,
        )
        human_quaternion = gmr.normalize_quaternion_wxyz(
            gmr.quat_multiply_wxyz(
                robot_quaternion,
                gmr.quat_conjugate_wxyz(rotation_offset),
            )
        )
        if human_body in scaled_positions:
            if not np.allclose(scaled_positions[human_body], scaled_position, atol=1.0e-9):
                raise ValueError(f"multiple robot bodies map to incompatible {human_body!r} positions")
            if abs(float(quaternions[human_body] @ human_quaternion)) < 1.0 - 1.0e-9:
                raise ValueError(f"multiple robot bodies map to incompatible {human_body!r} rotations")
        scaled_positions[human_body] = scaled_position
        quaternions[human_body] = human_quaternion

    if human_root not in scaled_positions:
        raise ValueError(f"no task maps the human root {human_root!r}")
    scaled_root = scaled_positions[human_root]
    root_position = scaled_root / scale[human_root]
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, scaled_position in scaled_positions.items():
        if name == human_root:
            position = root_position
        else:
            position = root_position + (scaled_position - scaled_root) / scale[name]
        result[name] = (position, quaternions[name])
    return result


def seed_retargeter(solver: gmr.PicoGmrRetargeter, joints: np.ndarray) -> None:
    solver.data.qpos[solver._qpos_addresses] = np.asarray(joints, dtype=np.float64)
    mujoco.mj_forward(solver.model, solver.data)


def load_captured_pico_frame(
    path: Path,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], np.ndarray | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "bxi_pico_gmr_frame_v1":
        raise ValueError("unsupported captured PICO frame format")
    raw_human = payload.get("human_data")
    if not isinstance(raw_human, dict):
        raise ValueError("captured PICO frame has no human_data object")
    human_data = {
        str(name): (
            np.asarray(pose["position"], dtype=np.float64),
            gmr.normalize_quaternion_wxyz(pose["quaternion_wxyz"]),
        )
        for name, pose in raw_human.items()
    }
    captured_joints = None
    if "gmr_joint_pos" in payload:
        names = tuple(str(name) for name in payload.get("gmr_joint_names", ()))
        values = np.asarray(payload["gmr_joint_pos"], dtype=np.float64)
        if names != protocol.REFERENCE_JOINT_NAMES or values.shape != (len(names),):
            raise ValueError("captured GMR joint layout does not match the current named layout")
        captured_joints = values
    return human_data, captured_joints


def _candidate_config(base: dict[str, object], candidate: str) -> dict[str, object]:
    config = copy.deepcopy(base)
    legacy_right = [0.0, math.sqrt(0.5), 0.0, -math.sqrt(0.5)]
    right_xml_frame = [math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0]
    left_xml_frame = [math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0]
    if candidate == "current":
        return config
    if candidate == "previous_identity":
        for table_name in ("ik_match_table1", "ik_match_table2"):
            config[table_name]["r_elbow_y_link"][4] = [1.0, 0.0, 0.0, 0.0]
            config[table_name]["r_wrist_z_link"][4] = [1.0, 0.0, 0.0, 0.0]
        return config
    if candidate == "legacy_hybrid":
        for table_name in ("ik_match_table1", "ik_match_table2"):
            config[table_name]["r_elbow_y_link"][4] = legacy_right
            config[table_name]["r_wrist_z_link"][4] = legacy_right
        return config
    if candidate == "mocaplab_backup_right_arm":
        for table_name in ("ik_match_table1", "ik_match_table2"):
            config[table_name]["r_elbow_y_link"][4] = legacy_right
            config[table_name]["r_wrist_z_link"][4] = legacy_right
        config["ik_match_table1"]["r_elbow_y_link"][1:3] = [20, 30]
        config["ik_match_table1"]["r_wrist_z_link"][1:3] = [30, 30]
        config["ik_match_table2"]["r_elbow_y_link"][1:3] = [0, 0]
        config["ik_match_table2"]["r_wrist_z_link"][1:3] = [40, 0]
        return config
    if candidate == "right_xml_frame_corrected":
        for table_name in ("ik_match_table1", "ik_match_table2"):
            config[table_name]["r_elbow_y_link"][4] = right_xml_frame
            config[table_name]["r_wrist_z_link"][4] = right_xml_frame
        return config
    if candidate == "all_arm_xml_frame_corrected":
        for table_name in ("ik_match_table1", "ik_match_table2"):
            config[table_name]["l_elbow_y_link"][4] = left_xml_frame
            config[table_name]["l_wrist_z_link"][4] = left_xml_frame
            config[table_name]["r_elbow_y_link"][4] = right_xml_frame
            config[table_name]["r_wrist_z_link"][4] = right_xml_frame
        return config
    raise ValueError(f"unknown calibration candidate {candidate!r}")


def solve_captured_frame_candidates(
    model_xml: Path,
    base_config: dict[str, object],
    human_data: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    max_iterations: int,
) -> dict[str, np.ndarray]:
    results: dict[str, np.ndarray] = {}
    with tempfile.TemporaryDirectory(prefix="pico_gmr_candidates_") as directory:
        root = Path(directory)
        for candidate in CALIBRATION_CANDIDATES:
            config_path = root / f"{candidate}.json"
            config_path.write_text(
                json.dumps(_candidate_config(base_config, candidate), indent=2) + "\n",
                encoding="utf-8",
            )
            solver = gmr.PicoGmrRetargeter(
                model_xml,
                config_path,
                protocol.REFERENCE_JOINT_NAMES,
                max_iterations=max_iterations,
            )
            results[candidate] = solver.retarget(human_data)[0]
    return results


def print_captured_frame_comparison(
    human_data: dict[str, tuple[np.ndarray, np.ndarray]],
    results: dict[str, np.ndarray],
    captured_joints: np.ndarray | None,
) -> None:
    print("Captured PICO arm geometry:")
    for title in ("Left", "Right"):
        shoulder = human_data[f"{title}_Shoulder"][0]
        elbow = human_data[f"{title}_Elbow"][0]
        wrist = human_data[f"{title}_Wrist"][0]
        upper = elbow - shoulder
        lower = wrist - elbow
        cosine = float(upper @ lower) / float(np.linalg.norm(upper) * np.linalg.norm(lower))
        straightness = math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))
        print(
            f"  {title:5s}: upper={np.linalg.norm(upper):.4f} m  "
            f"lower={np.linalg.norm(lower):.4f} m  bone-angle={straightness:.3f} deg"
        )

    selected_names = (
        "r_shoulder_y_joint",
        "r_shoulder_x_joint",
        "r_shoulder_z_joint",
        "r_elbow_y_joint",
        "r_wrist_x_joint",
        "r_wrist_y_joint",
        "r_wrist_z_joint",
    )
    print("\nRight-arm candidate outputs (degrees):")
    print("  candidate                         " + " ".join(f"{name[2:-6]:>10s}" for name in selected_names))
    for candidate, joints in results.items():
        values = [
            math.degrees(joints[protocol.REFERENCE_JOINT_NAMES.index(name)])
            for name in selected_names
        ]
        print(f"  {candidate:33s}" + " ".join(f"{value:10.2f}" for value in values))
    if captured_joints is not None:
        difference = np.degrees(results["current"] - captured_joints)
        print(
            "\nCurrent-config replay check: "
            f"max abs difference from captured output={np.max(np.abs(difference)):.6f} deg"
        )


def task_space_errors(
    reference_model: mujoco.MjModel,
    reference_data: mujoco.MjData,
    candidate_joints: np.ndarray,
    robot_bodies: Iterable[str],
) -> tuple[float, float, dict[str, tuple[float, float]]]:
    candidate_data = mujoco.MjData(reference_model)
    mujoco.mj_resetData(reference_model, candidate_data)
    set_named_joint_pose(reference_model, candidate_data, candidate_joints)
    details: dict[str, tuple[float, float]] = {}
    for body in robot_bodies:
        ref_pos, ref_quat = _body_pose(reference_model, reference_data, body)
        got_pos, got_quat = _body_pose(reference_model, candidate_data, body)
        position_mm = float(np.linalg.norm(got_pos - ref_pos) * 1000.0)
        relative = gmr.quat_multiply_wxyz(
            gmr.quat_conjugate_wxyz(ref_quat),
            got_quat,
        )
        rotation_deg = math.degrees(
            float(np.linalg.norm(gmr.quaternion_rotation_vector_wxyz(relative)))
        )
        details[body] = (position_mm, rotation_deg)
    max_position = max(value[0] for value in details.values())
    max_rotation = max(value[1] for value in details.values())
    return max_position, max_rotation, details


def print_joint_errors(reference: np.ndarray, candidate: np.ndarray, *, label: str) -> None:
    errors_deg = np.degrees(np.asarray(candidate) - np.asarray(reference))
    print(f"\n[{label}] joint round-trip errors")
    print(f"  max abs: {np.max(np.abs(errors_deg)):.3f} deg")
    print(f"  rms:     {np.sqrt(np.mean(np.square(errors_deg))):.3f} deg")
    print("  arms:")
    for name in ARM_JOINTS:
        index = protocol.REFERENCE_JOINT_NAMES.index(name)
        print(
            f"    {name:24s} ref={math.degrees(reference[index]):8.3f} deg"
            f"  got={math.degrees(candidate[index]):8.3f} deg"
            f"  err={errors_deg[index]:+8.3f} deg"
        )


def view_pose(model: mujoco.MjModel, joints: np.ndarray, *, label: str) -> None:
    try:
        import mujoco.viewer
    except ImportError as exc:
        raise RuntimeError("mujoco.viewer is unavailable") from exc
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    set_named_joint_pose(model, data, joints)
    print(f"Opening MuJoCo viewer with {label} joint positions; close the window to exit.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(1.0 / 60.0)


def _parse_endpoint(value: str) -> tuple[str, int]:
    host, separator, raw_port = value.rpartition(":")
    if not separator or not host:
        raise argparse.ArgumentTypeError("endpoint must be HOST:PORT")
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("endpoint port must be an integer") from exc
    if not 0 < port < 65536:
        raise argparse.ArgumentTypeError("endpoint port must be in [1,65535]")
    return host, port


def listen_and_view(
    model: mujoco.MjModel,
    standing: np.ndarray,
    endpoint: tuple[str, int],
) -> None:
    try:
        import mujoco.viewer
    except ImportError as exc:
        raise RuntimeError("mujoco.viewer is unavailable") from exc

    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    udp.bind(endpoint)
    udp.setblocking(False)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    set_named_joint_pose(model, data, standing)
    last_report = 0.0
    latest_sequence: int | None = None
    print(
        f"Listening for raw GMR references on udp://{endpoint[0]}:{endpoint[1]}; "
        "MuJoCo is initialized with the RGMT standing pose."
    )
    print("This path bypasses the RGMT policy completely. Close the viewer to exit.")
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                newest = None
                while True:
                    try:
                        packet, _source = udp.recvfrom(protocol.PACKET_SIZE + 1)
                    except BlockingIOError:
                        break
                    try:
                        newest = protocol.decode_reference_frame(packet)
                    except (TypeError, ValueError) as exc:
                        print(f"Ignored invalid GMR packet: {exc}", file=sys.stderr)
                if newest is not None:
                    set_named_joint_pose(model, data, newest.joint_pos)
                    latest_sequence = newest.sequence
                now = time.monotonic()
                if latest_sequence is not None and now - last_report >= 1.0:
                    left_index = protocol.REFERENCE_JOINT_NAMES.index("l_elbow_y_joint")
                    right_index = protocol.REFERENCE_JOINT_NAMES.index("r_elbow_y_joint")
                    qpos = data.qpos[_joint_qpos_addresses(model, protocol.REFERENCE_JOINT_NAMES)]
                    print(
                        f"sequence={latest_sequence}  "
                        f"l_elbow={math.degrees(qpos[left_index]):7.2f} deg  "
                        f"r_elbow={math.degrees(qpos[right_index]):7.2f} deg"
                    )
                    last_report = now
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(1.0 / 120.0)
    finally:
        udp.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xml",
        type=Path,
        default=PACKAGE_ROOT / "data" / "mujoco_simulation" / "elf3.xml",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=MOD_ROOT / "assets" / "pico_to_elf3.json",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=MOD_ROOT / "assets" / "rgmt.onnx",
    )
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--view", choices=("none", "standing", "cold", "warm"), default="none")
    parser.add_argument(
        "--listen",
        type=_parse_endpoint,
        metavar="HOST:PORT",
        help="bypass policy and display live GMR UDP joint references directly in MuJoCo",
    )
    parser.add_argument(
        "--replay-pico",
        type=Path,
        help="replay a --capture-frame JSON through several right-arm calibrations",
    )
    parser.add_argument(
        "--view-candidate",
        choices=CALIBRATION_CANDIDATES,
        help="with --replay-pico, display this candidate joint solution in MuJoCo",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.iterations < 1:
        raise ValueError("--iterations must be at least 1")
    with args.config.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    standing = load_standing_reference(args.onnx)

    model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    if args.listen is not None:
        listen_and_view(model, standing, args.listen)
        return 0
    if args.replay_pico is not None:
        human_data, captured_joints = load_captured_pico_frame(args.replay_pico)
        results = solve_captured_frame_candidates(
            args.xml,
            config,
            human_data,
            max_iterations=args.iterations,
        )
        print_captured_frame_comparison(human_data, results, captured_joints)
        if args.view_candidate is not None:
            view_pose(model, results[args.view_candidate], label=args.view_candidate)
        return 0
    if args.view_candidate is not None:
        raise ValueError("--view-candidate requires --replay-pico")
    reference_data = mujoco.MjData(model)
    mujoco.mj_resetData(model, reference_data)
    set_named_joint_pose(model, reference_data, standing)
    pseudo_pico = pseudo_pico_from_fk(model, reference_data, config)
    robot_bodies = tuple(_task_definitions(config))

    cold_solver = gmr.PicoGmrRetargeter(
        args.xml,
        args.config,
        protocol.REFERENCE_JOINT_NAMES,
        max_iterations=args.iterations,
    )
    seed_retargeter(cold_solver, np.zeros_like(standing))
    cold, _, _ = cold_solver.retarget(pseudo_pico)

    warm_solver = gmr.PicoGmrRetargeter(
        args.xml,
        args.config,
        protocol.REFERENCE_JOINT_NAMES,
        max_iterations=args.iterations,
    )
    seed_retargeter(warm_solver, standing)
    warm, _, _ = warm_solver.retarget(pseudo_pico)

    elbow_indices = {
        side: protocol.REFERENCE_JOINT_NAMES.index(f"{side}_elbow_y_joint")
        for side in ("l", "r")
    }
    print("RGMT standing reference elbows:")
    for side, index in elbow_indices.items():
        print(f"  {side}: {standing[index]:.6f} rad = {math.degrees(standing[index]):.3f} deg")
    print_joint_errors(standing, cold, label="zero start")
    print_joint_errors(standing, warm, label="standing warm start")

    for label, candidate in (("zero start", cold), ("standing warm start", warm)):
        max_pos, max_rot, details = task_space_errors(
            model,
            reference_data,
            candidate,
            robot_bodies,
        )
        print(f"\n[{label}] task-space FK difference: max={max_pos:.3f} mm, {max_rot:.3f} deg")
        for body in (
            "l_shoulder_y_link",
            "l_elbow_y_link",
            "l_wrist_z_link",
            "r_shoulder_y_link",
            "r_elbow_y_link",
            "r_wrist_z_link",
        ):
            position_mm, rotation_deg = details[body]
            print(f"  {body:20s} pos={position_mm:8.3f} mm  rot={rotation_deg:8.3f} deg")

    choices = {"standing": standing, "cold": cold, "warm": warm}
    if args.view != "none":
        view_pose(model, choices[args.view], label=args.view)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
