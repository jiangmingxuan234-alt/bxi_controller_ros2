import numpy as np
import pytest

from zerolab.converter import ConvertedPoseFrame
from zerolab.resampler import PlayoutKind, ZeroLabPoseResampler


def frame(index, timestamp_ns, value, *, root_wxyz=(1.0, 0.0, 0.0, 0.0)):
    return ConvertedPoseFrame(
        frame_index=index,
        receive_timestamp_ns=timestamp_ns,
        smpl_body_pose=np.full((21, 3), value, dtype=np.float32),
        smpl_joints=np.full((24, 3), value, dtype=np.float32),
        body_quat_w=np.asarray(root_wxyz, dtype=np.float32),
        joint_pos=np.full(29, value, dtype=np.float32),
    )


def make_resampler():
    return ZeroLabPoseResampler(
        jitter_buffer_seconds=0.08,
        short_recovery_blend_seconds=0.2,
        output_rate_hz=50.0,
    )


def test_regular_input_plays_at_target_delayed_by_80_ms():
    resampler = ZeroLabPoseResampler(
        jitter_buffer_seconds=0.08,
        short_recovery_blend_seconds=0.2,
        output_rate_hz=50.0,
    )
    for index in range(7):
        assert resampler.observe(frame(index, index * 20_000_000, float(index)))

    output = resampler.sample(120_000_000)  # target = 40 ms, real frame 2

    assert output.kind is PlayoutKind.REAL
    assert output.frame.frame_index == 0
    assert output.frame.receive_timestamp_ns == 40_000_000
    np.testing.assert_allclose(output.frame.smpl_joints, 2.0)
    assert output.latest_real_frame_index == 6
    assert output.latest_real_receive_timestamp_ns == 120_000_000


def test_jittered_samples_interpolate_continuous_fields():
    resampler = make_resampler()
    resampler.observe(frame(0, 0, 0.0))
    resampler.observe(frame(1, 40_000_000, 4.0))

    output = resampler.sample(100_000_000)  # target = 20 ms

    assert output.kind is PlayoutKind.INTERPOLATED
    np.testing.assert_allclose(output.frame.smpl_body_pose, 2.0)
    np.testing.assert_allclose(output.frame.smpl_joints, 2.0)
    np.testing.assert_allclose(output.frame.joint_pos, 2.0)


def test_quaternion_interpolation_corrects_hemisphere_and_normalizes():
    resampler = make_resampler()
    resampler.observe(frame(0, 0, 0.0, root_wxyz=(1.0, 0.0, 0.0, 0.0)))
    resampler.observe(frame(1, 40_000_000, 4.0,
                            root_wxyz=(-2**-0.5, 0.0, 0.0, -2**-0.5)))

    root = resampler.sample(100_000_000).frame.body_quat_w

    np.testing.assert_allclose(np.linalg.norm(root), 1.0, atol=1e-6)
    np.testing.assert_allclose(
        root, np.array([0.9238795, 0.0, 0.0, 0.3826834]), atol=1e-6
    )


def test_missing_right_bracket_holds_and_never_extrapolates():
    resampler = make_resampler()
    resampler.observe(frame(0, 0, 0.0))
    first = resampler.sample(80_000_000)
    held = resampler.sample(200_000_000)
    assert held.kind is PlayoutKind.HELD
    np.testing.assert_array_equal(held.frame.smpl_joints, first.frame.smpl_joints)
    assert resampler.stats.held_output_frames == 1


def test_burst_discards_obsolete_brackets_without_output_catchup():
    resampler = make_resampler()
    for index in range(100):
        resampler.observe(frame(index, index * 20_000_000, float(index)))
    assert resampler.sample(2_000_000_000).frame.frame_index == 0
    assert resampler.stats.dropped_backlog_frames >= 95


def test_duplicate_backward_and_nonmonotonic_inputs_are_defensive_errors():
    resampler = make_resampler()
    assert resampler.observe(frame(3, 60_000_000, 3.0))
    assert resampler.observe(frame(3, 60_000_000, 3.0)) is False
    with pytest.raises(ValueError, match="frame_index"):
        resampler.observe(frame(2, 80_000_000, 2.0))
    resampler.reset()
    resampler.observe(frame(4, 100_000_000, 4.0))
    with pytest.raises(ValueError, match="receive_timestamp_ns"):
        resampler.observe(frame(5, 90_000_000, 5.0))


def test_short_gap_recovery_blends_from_pose_actually_held():
    resampler = make_resampler()
    resampler.observe(frame(0, 0, 0.0))
    resampler.observe(frame(1, 20_000_000, 1.0))
    held = resampler.sample(200_000_000)
    assert held.kind is PlayoutKind.HELD
    resampler.observe(frame(2, 140_000_000, 10.0))
    resampler.observe(frame(3, 180_000_000, 14.0))

    start = resampler.sample(220_000_000)
    middle = resampler.sample(320_000_000)

    assert start.kind is PlayoutKind.SHORT_RECOVERY_BLEND
    assert middle.kind is PlayoutKind.SHORT_RECOVERY_BLEND
    np.testing.assert_allclose(start.frame.smpl_joints, held.frame.smpl_joints)
    assert np.all(middle.frame.smpl_joints > start.frame.smpl_joints)


def test_mark_stale_clears_brackets_but_preserves_last_output_for_stale_hold():
    resampler = make_resampler()
    resampler.observe(frame(0, 0, 3.0))
    live = resampler.sample(80_000_000)
    resampler.mark_stale()
    stale = resampler.sample(100_000_000)
    assert stale.kind is PlayoutKind.HELD
    np.testing.assert_array_equal(stale.frame.smpl_joints, live.frame.smpl_joints)


def test_post_stale_input_does_not_use_short_recovery_blend():
    resampler = make_resampler()
    resampler.observe(frame(0, 0, 1.0))
    resampler.sample(80_000_000)
    resampler.mark_stale()
    resampler.observe(frame(1, 1_000_000_000, 10.0))
    resampler.observe(frame(2, 1_040_000_000, 14.0))
    recovered = resampler.sample(1_100_000_000)
    assert recovered.kind in (PlayoutKind.REAL, PlayoutKind.INTERPOLATED)
