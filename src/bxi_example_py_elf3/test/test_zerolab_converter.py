import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import zerolab.converter as converter_module
from zerolab.converter import (
    BODY_JOINT_COUNT,
    SMPL24_PARENTS,
    TPoseCalibrator,
    ZeroLabMotionConverter,
    align_quaternion_signs,
    apply_rest_alignment,
    synthesize_smpl_world_quats,
    unity_world_quaternions_to_xrt,
)
from zerolab.protocol import ZeroLabPacket
from pico.gear_sonic.trl.utils.elf3_wrist import (
    build_elf3_joint_pos,
    compute_elf3_wrist_angles,
)
from pico.gear_sonic.trl.utils.numpy_smpl import compute_from_body_poses


DIRECT_CASES = [
    (10, 0), (11, 1), (14, 2), (12, 4), (15, 5),
    (13, 7), (16, 8), (2, 13), (6, 14), (0, 15),
    (3, 16), (7, 17), (4, 18), (8, 19), (5, 20), (9, 21),
]


def identity_body():
    quats = np.zeros((BODY_JOINT_COUNT, 4), dtype=np.float32)
    quats[:, 3] = 1.0
    return quats


def make_packet(index, quats47, root=None):
    return ZeroLabPacket(
        receive_timestamp_ns=index * 20_000_000,
        local_frame_index=index,
        root_translation=(
            np.zeros(3, dtype=np.float32) if root is None else root
        ),
        joint_quat_world_xyzw=np.asarray(quats47, dtype=np.float32),
        left_hand_values=np.zeros(6, dtype=np.uint16),
        right_hand_values=np.zeros(6, dtype=np.uint16),
        joint_position=np.zeros((17, 3), dtype=np.float32),
        raw_payload=bytes(992),
        sender_address=("127.0.0.1", 50000),
    )


def identity47():
    result = np.zeros((47, 4), dtype=np.float32)
    result[:, 3] = 1.0
    return result


def test_unity_world_quaternions_to_xrt_reflects_normalized_components():
    unity = np.array(
        [[2.0, -3.0, 4.0, -5.0], [-6.0, 8.0, -10.0, 12.0]],
        dtype=np.float64,
    )
    original = unity.copy()

    actual = unity_world_quaternions_to_xrt(unity, (2, 4))

    expected = original / np.linalg.norm(original, axis=1, keepdims=True)
    expected[:, :2] *= -1.0
    np.testing.assert_allclose(actual, expected.astype(np.float32), atol=1e-7)
    np.testing.assert_array_equal(unity, original)
    assert actual.dtype == np.float32
    assert actual.flags.c_contiguous
    np.testing.assert_allclose(
        np.linalg.norm(actual.astype(np.float64), axis=1), 1.0, atol=1e-7
    )


def test_quaternion_sign_flip_is_continuous():
    previous = identity_body()
    current = -previous

    aligned = align_quaternion_signs(current, previous)

    np.testing.assert_array_equal(aligned, previous)
    assert aligned.dtype == np.float32


def test_stable_t_pose_completes_on_frame_100():
    calibrator = TPoseCalibrator()
    for index in range(99):
        frame = identity_body() if index % 2 == 0 else -identity_body()
        assert calibrator.observe(frame) is False
        assert calibrator.frames_collected == index + 1

    assert calibrator.observe(-identity_body()) is True
    assert calibrator.is_calibrated
    np.testing.assert_allclose(
        calibrator.rest_quats_xyzw, identity_body(), atol=1e-7
    )
    assert calibrator.rest_quats_xyzw.dtype == np.float32


def test_motion_over_five_degrees_restarts_calibration_window():
    calibrator = TPoseCalibrator(required_frames=4, max_deviation_degrees=5.0)
    for _ in range(3):
        calibrator.observe(identity_body())
    moved = identity_body()
    moved[4] = Rotation.from_euler("x", 10.0, degrees=True).as_quat()

    assert calibrator.observe(moved) is False
    assert calibrator.frames_collected == 1


def test_motion_at_five_degrees_stays_in_calibration_window():
    boundary = identity_body()
    boundary[4] = Rotation.from_euler("x", 10.0, degrees=True).as_quat()
    normalized = boundary.astype(np.float64)
    normalized /= np.linalg.norm(normalized, axis=1, keepdims=True)
    normalized = normalized.astype(np.float32)
    expected_mean = Rotation.from_euler(
        "x", 5.0, degrees=True
    ).as_quat().astype(np.float32)
    endpoint_dots = np.array(
        [normalized[0] @ expected_mean, normalized[4] @ expected_mean]
    )
    boundary_angle = np.max(
        np.degrees(
            2.0 * np.arccos(np.clip(np.abs(endpoint_dots), 0.0, 1.0))
        )
    )
    calibrator = TPoseCalibrator(
        required_frames=3,
        max_deviation_degrees=float(boundary_angle),
    )
    calibrator.observe(identity_body())

    assert calibrator.observe(boundary) is False
    assert calibrator.frames_collected == 2


def test_motion_is_compared_to_every_frame_in_candidate_window():
    calibrator = TPoseCalibrator(required_frames=5, max_deviation_degrees=5.0)
    calibrator.observe(identity_body())
    four_degrees = identity_body()
    four_degrees[4] = Rotation.from_euler("x", 4.0, degrees=True).as_quat()
    eight_degrees = identity_body()
    eight_degrees[4] = Rotation.from_euler("x", 8.0, degrees=True).as_quat()
    ten_degrees = identity_body()
    ten_degrees[4] = Rotation.from_euler("x", 10.0, degrees=True).as_quat()

    assert calibrator.observe(four_degrees) is False
    assert calibrator.frames_collected == 2
    assert calibrator.observe(eight_degrees) is False
    assert calibrator.frames_collected == 3
    assert calibrator.observe(ten_degrees) is False
    assert calibrator.frames_collected == 1


def test_gradual_motion_cannot_chase_the_calibration_window_mean():
    calibrator = TPoseCalibrator(
        required_frames=5, max_deviation_degrees=5.0
    )

    for expected_count, angle in enumerate((0.0, 4.0, 6.0, 8.0), 1):
        frame = identity_body()
        frame[4] = Rotation.from_euler(
            "x", angle, degrees=True
        ).as_quat()
        assert calibrator.observe(frame) is False
        assert calibrator.frames_collected == expected_count

    moved = identity_body()
    moved[4] = Rotation.from_euler("x", 9.0, degrees=True).as_quat()

    assert calibrator.observe(moved) is False
    assert calibrator.frames_collected == 1
    assert calibrator.is_calibrated is False


def test_rest_alignment_uses_raw_times_inverse_rest():
    rest = np.repeat(
        Rotation.from_euler("x", 25.0, degrees=True).as_quat()[None, :],
        BODY_JOINT_COUNT,
        axis=0,
    )
    yaw = Rotation.from_euler("y", 30.0, degrees=True)
    raw = (yaw * Rotation.from_quat(rest)).as_quat()

    aligned = apply_rest_alignment(raw, rest)
    expected = np.repeat(yaw.as_quat()[None, :], BODY_JOINT_COUNT, axis=0)

    np.testing.assert_allclose(
        Rotation.from_quat(aligned).as_matrix(),
        Rotation.from_quat(expected).as_matrix(),
        atol=1e-6,
    )
    assert aligned.dtype == np.float32


def test_approved_mapping_slerps_spine_neck_and_copies_toes():
    body = identity_body()
    body[10] = Rotation.from_euler("y", 0.0, degrees=True).as_quat()
    body[1] = Rotation.from_euler("y", 60.0, degrees=True).as_quat()
    body[0] = Rotation.from_euler("y", 100.0, degrees=True).as_quat()
    body[13] = Rotation.from_euler("x", 12.0, degrees=True).as_quat()
    body[16] = Rotation.from_euler("x", -12.0, degrees=True).as_quat()

    smpl = synthesize_smpl_world_quats(body)

    for index, angle in zip(
        (3, 6, 9, 12, 15), (20.0, 40.0, 60.0, 80.0, 100.0)
    ):
        np.testing.assert_allclose(
            Rotation.from_quat(smpl[index]).as_matrix(),
            Rotation.from_euler("y", angle, degrees=True).as_matrix(),
            atol=1e-6,
        )
    np.testing.assert_allclose(smpl[10], smpl[7])
    np.testing.assert_allclose(smpl[11], smpl[8])
    np.testing.assert_allclose(smpl[22], smpl[20])
    np.testing.assert_allclose(smpl[23], smpl[21])
    assert smpl.dtype == np.float32
    assert SMPL24_PARENTS == [
        -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
        9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21,
    ]


@pytest.mark.parametrize("source_index,target_index", DIRECT_CASES)
def test_each_measured_joint_uses_its_approved_direct_smpl_target(
    source_index, target_index
):
    body = identity_body()
    angle = 7.0 + source_index
    body[source_index] = Rotation.from_euler(
        "z", angle, degrees=True
    ).as_quat()

    smpl = synthesize_smpl_world_quats(body)

    np.testing.assert_allclose(
        Rotation.from_quat(smpl[target_index]).as_matrix(),
        Rotation.from_euler("z", angle, degrees=True).as_matrix(),
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "bad_quaternions",
    [
        np.zeros((BODY_JOINT_COUNT - 1, 4), dtype=np.float32),
        np.full((BODY_JOINT_COUNT, 4), np.nan, dtype=np.float32),
        np.zeros((BODY_JOINT_COUNT, 4), dtype=np.float32),
    ],
)
def test_body_quaternion_inputs_require_finite_nonzero_17_by_4_arrays(
    bad_quaternions,
):
    with pytest.raises(ValueError):
        synthesize_smpl_world_quats(bad_quaternions)


@pytest.mark.parametrize(
    "bad_quaternions",
    [
        identity_body().astype("U32"),
        identity_body().astype(np.complex128) + 1j,
    ],
)
def test_all_converter_entry_points_reject_non_real_numeric_dtypes(
    bad_quaternions,
):
    with pytest.raises(ValueError):
        align_quaternion_signs(bad_quaternions)
    with pytest.raises(ValueError):
        apply_rest_alignment(bad_quaternions, identity_body())
    with pytest.raises(ValueError):
        TPoseCalibrator().observe(bad_quaternions)
    with pytest.raises(ValueError):
        synthesize_smpl_world_quats(bad_quaternions)


def test_wrist_zero_and_axis_sign_conventions():
    zero = build_elf3_joint_pos(
        np.zeros((1, 21, 3), dtype=np.float32)
    )[0]
    np.testing.assert_array_equal(zero, np.zeros(29, dtype=np.float32))

    pose = np.zeros((1, 21, 3), dtype=np.float32)
    pose[0, 19, 0] = 0.2
    pose[0, 20, 1] = 0.3
    joints = build_elf3_joint_pos(pose)[0]
    np.testing.assert_allclose(joints[[19, 27]], [0.2, -0.3], atol=1e-6)
    np.testing.assert_array_equal(
        np.delete(joints, [19, 20, 21, 26, 27, 28]),
        np.zeros(23, dtype=np.float32),
    )


def test_wrist_matches_current_pico_golden_vector():
    pose = np.zeros((1, 21, 3), dtype=np.float64)
    pose[0, 17] = [0.20, -0.10, 0.30]
    pose[0, 19] = [-0.15, 0.25, 0.05]
    pose[0, 18] = [-0.30, 0.20, 0.10]
    pose[0, 20] = [0.12, -0.22, 0.18]
    expected = np.array([
        0.052426428216, 0.245093826303, 0.357268492264,
        0.166555238766, 0.207292703914, 0.263749028654,
    ], dtype=np.float32)

    actual = compute_elf3_wrist_angles(pose)

    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual[0], expected, rtol=1e-6, atol=1e-7)
    pico_joint_pos = build_elf3_joint_pos(pose, dtype=np.float64)
    assert pico_joint_pos.dtype == np.float64


def test_mirrored_wrist_inputs_have_symmetric_approved_right_side_signs():
    pose = np.zeros((1, 21, 3), dtype=np.float64)
    pose[0, [17, 18]] = [0.20, -0.10, 0.30]
    pose[0, [19, 20]] = [-0.15, 0.25, 0.05]

    wrists = compute_elf3_wrist_angles(pose, dtype=np.float64)[0]

    assert np.isfinite(wrists).all()
    np.testing.assert_allclose(np.abs(wrists[:3]), np.abs(wrists[3:]))
    np.testing.assert_allclose(
        wrists[3:], [-wrists[0], -wrists[1], wrists[2]]
    )


@pytest.mark.parametrize(
    "bad_pose",
    [
        np.zeros((21, 3), dtype=np.float32),
        np.zeros((0, 21, 3), dtype=np.float32),
        np.zeros((1, 20, 3), dtype=np.float32),
        np.full((1, 21, 3), np.nan, dtype=np.float32),
    ],
)
def test_wrist_helper_rejects_bad_shape_empty_batch_and_nonfinite(bad_pose):
    with pytest.raises(ValueError):
        compute_elf3_wrist_angles(bad_pose)
    with pytest.raises(ValueError):
        build_elf3_joint_pos(bad_pose)


@pytest.mark.parametrize("bad_dtype", [np.float16, np.int64, object])
def test_wrist_helper_restricts_output_dtype(bad_dtype):
    pose = np.zeros((1, 21, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        compute_elf3_wrist_angles(pose, dtype=bad_dtype)
    with pytest.raises(ValueError):
        build_elf3_joint_pos(pose, dtype=bad_dtype)


def test_converter_uses_100_frames_only_for_rest_then_emits_frame_101():
    converter = ZeroLabMotionConverter()
    for index in range(100):
        assert converter.observe(make_packet(index, identity47())) is None

    output = converter.observe(make_packet(100, identity47()))

    assert output.frame_index == 100
    assert output.receive_timestamp_ns == 2_000_000_000
    assert output.smpl_body_pose.shape == (21, 3)
    assert output.smpl_joints.shape == (24, 3)
    assert output.body_quat_w.shape == (4,)
    assert output.joint_pos.shape == (29,)
    assert output.smpl_body_pose.dtype == np.float32
    assert output.smpl_joints.dtype == np.float32
    assert output.body_quat_w.dtype == np.float32
    assert output.joint_pos.dtype == np.float32
    assert np.isfinite(output.smpl_joints).all()
    np.testing.assert_allclose(output.smpl_body_pose, 0.0, atol=1e-6)


def test_unity_yaw_direction_maps_physical_left_to_positive_sonic_yaw():
    converter = ZeroLabMotionConverter()
    rest = identity47()
    for index in range(100):
        assert converter.observe(make_packet(index, rest)) is None
    baseline = converter.observe(make_packet(100, rest))

    unity_left = identity47()
    unity_left[:17] = [0.0, -0.25881904, 0.0, 0.9659258]
    left = converter.observe(make_packet(101, unity_left))

    unity_right = identity47()
    unity_right[:17] = [0.0, 0.25881904, 0.0, 0.9659258]
    right = converter.observe(make_packet(102, unity_right))

    baseline_yaw = Rotation.from_quat(
        baseline.body_quat_w[[1, 2, 3, 0]]
    ).as_euler("xyz")[2]
    left_yaw = Rotation.from_quat(left.body_quat_w[[1, 2, 3, 0]]).as_euler(
        "xyz"
    )[2]
    right_yaw = Rotation.from_quat(
        right.body_quat_w[[1, 2, 3, 0]]
    ).as_euler("xyz")[2]

    assert left_yaw - baseline_yaw > 0.0
    assert right_yaw - baseline_yaw < 0.0


def test_rigid_yaw_matches_existing_fk_and_preserves_pelvis_relative_shape():
    converter = ZeroLabMotionConverter()
    rest = identity47()
    for index in range(100):
        converter.observe(make_packet(index, rest))
    t_pose = converter.observe(make_packet(100, rest))

    unity_yaw = Rotation.from_euler("y", 30.0, degrees=True)
    unity_yawed = identity47()
    unity_yawed[:17] = (
        unity_yaw * Rotation.from_quat(rest[:17])
    ).as_quat()
    output = converter.observe(make_packet(101, unity_yawed))

    xrt_yawed = unity_yawed.astype(np.float64, copy=True)
    xrt_yawed /= np.linalg.norm(xrt_yawed, axis=1, keepdims=True)
    xrt_yawed[:, :2] *= -1.0
    xrt_yawed = np.ascontiguousarray(xrt_yawed, dtype=np.float32)
    virtual = synthesize_smpl_world_quats(
        apply_rest_alignment(xrt_yawed[:17], rest[:17])
    )
    body_poses = np.zeros((24, 7), dtype=np.float32)
    body_poses[:, 3:] = virtual
    expected = compute_from_body_poses(SMPL24_PARENTS, body_poses)
    np.testing.assert_allclose(
        output.smpl_joints, expected["smpl_joints_local"][0], atol=1e-6
    )
    np.testing.assert_allclose(
        output.smpl_joints - output.smpl_joints[0],
        t_pose.smpl_joints - t_pose.smpl_joints[0],
        atol=1e-6,
    )
    assert not np.allclose(output.body_quat_w, t_pose.body_quat_w)


def test_converter_reflects_only_local_root_translation_for_fk(monkeypatch):
    converter = ZeroLabMotionConverter()
    rest = identity47()
    for index in range(100):
        assert converter.observe(make_packet(index, rest)) is None

    root = np.array([1.25, -2.5, 3.75], dtype=np.float32)
    packet = make_packet(100, rest, root=root)
    original_root = packet.root_translation.copy()
    original_payload = packet.raw_payload
    captured = {}
    real_fk = converter_module.compute_from_body_poses

    def capture_fk(parents, body_poses):
        captured["root"] = body_poses[0, :3].copy()
        return real_fk(parents, body_poses)

    monkeypatch.setattr(converter_module, "compute_from_body_poses", capture_fk)

    assert converter.observe(packet) is not None

    np.testing.assert_array_equal(captured["root"], [1.25, -2.5, -3.75])
    np.testing.assert_array_equal(packet.root_translation, original_root)
    assert packet.raw_payload == original_payload


def test_left_elbow_motion_changes_only_left_wrist_chain():
    converter = ZeroLabMotionConverter()
    rest = identity47()
    for index in range(100):
        converter.observe(make_packet(index, rest))
    t_pose = converter.observe(make_packet(100, rest))

    moved = identity47()
    left_rotation = Rotation.from_euler("z", 30.0, degrees=True).as_quat()
    moved[[4, 5]] = left_rotation
    output = converter.observe(make_packet(101, moved))

    assert not np.allclose(output.smpl_joints[20], t_pose.smpl_joints[20])
    assert not np.allclose(output.smpl_joints[22], t_pose.smpl_joints[22])
    np.testing.assert_allclose(
        output.smpl_joints[0], t_pose.smpl_joints[0], atol=1e-6
    )
    np.testing.assert_allclose(
        output.smpl_joints[[17, 19, 21, 23]],
        t_pose.smpl_joints[[17, 19, 21, 23]],
        atol=1e-6,
    )


def test_converter_checks_47_quaternions_but_ignores_last_30():
    converter = ZeroLabMotionConverter()
    rest = identity47()
    for index in range(100):
        converter.observe(make_packet(index, rest))
    baseline = converter.observe(make_packet(100, rest))

    unused_changed = identity47()
    unused_changed[17:] = Rotation.random(30, random_state=123).as_quat()
    output = converter.observe(make_packet(101, unused_changed))
    np.testing.assert_array_equal(
        output.smpl_body_pose, baseline.smpl_body_pose
    )
    np.testing.assert_array_equal(output.smpl_joints, baseline.smpl_joints)
    np.testing.assert_array_equal(output.body_quat_w, baseline.body_quat_w)
    np.testing.assert_array_equal(output.joint_pos, baseline.joint_pos)

    invalid_unused = identity47()
    invalid_unused[-1] = 0.0
    with pytest.raises(ValueError):
        converter.observe(make_packet(102, invalid_unused))


def test_stale_and_reset_session_have_distinct_calibration_semantics():
    converter = ZeroLabMotionConverter()
    rest = identity47()
    for index in range(99):
        assert converter.observe(make_packet(index, rest)) is None
    converter.mark_stale()
    for index in range(99, 199):
        assert converter.observe(make_packet(index, rest)) is None
    assert converter.observe(make_packet(199, rest)) is not None

    converter.mark_stale()
    assert converter.observe(make_packet(200, rest)) is not None

    converter.reset_session()
    for index in range(201, 301):
        assert converter.observe(make_packet(index, rest)) is None
    assert converter.observe(make_packet(301, rest)) is not None
