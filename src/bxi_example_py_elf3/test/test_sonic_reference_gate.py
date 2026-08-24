from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest


MOD_ROOT = Path(__file__).resolve().parents[1] / "mods" / "com.bxi.sonic"
PACKAGE = "_sonic_reference_gate_test_mod"


def _load_module(name: str, relative_path: str):
    if PACKAGE not in sys.modules:
        package = ModuleType(PACKAGE)
        package.__path__ = [str(MOD_ROOT)]
        package.__package__ = PACKAGE
        sys.modules[PACKAGE] = package
    full_name = f"{PACKAGE}.{name}"
    loaded = sys.modules.get(full_name)
    if loaded is not None:
        return loaded
    path = MOD_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(full_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


reference_gate_module = _load_module("reference_gate", "reference_gate.py")
LiveReferenceGate = reference_gate_module.LiveReferenceGate
ReferenceGateMode = reference_gate_module.ReferenceGateMode
SmplReferenceFrame = reference_gate_module.SmplReferenceFrame
copy_smpl_reference = reference_gate_module.copy_smpl_reference
interpolate_smpl_reference = reference_gate_module.interpolate_smpl_reference


def frame(value, *, root=None):
    root_quat = np.zeros((10, 4), dtype=np.float32)
    root_quat[:, 0] = 1.0 if root is None else root[0]
    if root is not None:
        root_quat[:] = root
    return SmplReferenceFrame(
        term1_local=np.full((10, 72), value, dtype=np.float32),
        root_quat=root_quat,
        wrist=np.full((10, 6), value, dtype=np.float32),
        head_joint_pos=np.full((10, 2), value, dtype=np.float32),
        frame_index=int(value),
        newest_frame_index=int(value) + 9,
        valid_horizon=10,
        clamp_slots=0,
        source_generation=77,
        latest_real_frame_index=int(value) + 9,
        latest_real_receive_timestamp_ns=1_000_000_000 + int(value),
        real_valid_frames_in_generation=10,
        real_stream_ready=True,
        playout_kind=0,
    )


def test_hold_separates_observed_and_active_references():
    gate = LiveReferenceGate()
    gate.observe(frame(1), received_mono=1.0)
    assert gate.hold() is True
    gate.observe(frame(9), received_mono=2.0)

    assert gate.mode is ReferenceGateMode.HOLD
    assert gate.active_reference().frame_index == 1
    assert gate.observed_reference.frame_index == 9
    assert gate.has_fresh_observed(now_mono=2.4, timeout_s=0.5)


def test_rearm_blends_positions_and_shortest_path_quaternions():
    gate = LiveReferenceGate()
    gate.observe(frame(0), received_mono=1.0)
    gate.hold()
    end_root = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    gate.observe(frame(2, root=end_root), received_mono=2.0)
    assert gate.begin_rearm() is True
    gate.set_rearm_progress(0.5)

    active = gate.active_reference()
    np.testing.assert_allclose(active.term1_local, 1.0)
    np.testing.assert_allclose(np.linalg.norm(active.root_quat, axis=1), 1.0)
    np.testing.assert_allclose(
        np.abs(active.root_quat[0]),
        np.array([2**-0.5, 0.0, 2**-0.5, 0.0]),
        atol=1e-6,
    )


def test_interpolation_corrects_quaternion_hemisphere():
    start = frame(0)
    end = frame(
        2,
        root=np.array([-2**-0.5, 0.0, 0.0, -2**-0.5], dtype=np.float32),
    )

    blended = interpolate_smpl_reference(start, end, 0.5)

    np.testing.assert_allclose(
        blended.root_quat[0],
        np.array([0.9238795, 0.0, 0.0, 0.3826834]),
        atol=1e-6,
    )
    assert blended.source_generation == end.source_generation
    assert blended.latest_real_frame_index == end.latest_real_frame_index
    assert (
        blended.latest_real_receive_timestamp_ns
        == end.latest_real_receive_timestamp_ns
    )
    assert (
        blended.real_valid_frames_in_generation
        == end.real_valid_frames_in_generation
    )
    assert blended.real_stream_ready is end.real_stream_ready
    assert blended.playout_kind == end.playout_kind


def test_copy_smpl_reference_does_not_alias_arrays():
    original = frame(3)
    original.anchor_quat = np.tile(
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (10, 1)
    )

    copied = copy_smpl_reference(original)

    for name in (
        "term1_local",
        "root_quat",
        "wrist",
        "head_joint_pos",
        "anchor_quat",
    ):
        assert getattr(copied, name) is not getattr(original, name)
        np.testing.assert_array_equal(getattr(copied, name), getattr(original, name))
    for name in (
        "source_generation",
        "latest_real_frame_index",
        "latest_real_receive_timestamp_ns",
        "real_valid_frames_in_generation",
        "real_stream_ready",
        "playout_kind",
        "source_stale",
    ):
        assert getattr(copied, name) == getattr(original, name)


def test_hold_retains_source_metadata_without_aliasing_arrays():
    gate = LiveReferenceGate()
    gate.observe(frame(3), received_mono=1.0)
    observed = gate.observed_reference

    assert gate.hold() is True
    held = gate.active_reference()

    assert held is not None and observed is not None
    for name in (
        "term1_local",
        "root_quat",
        "wrist",
        "head_joint_pos",
    ):
        assert getattr(held, name) is not getattr(observed, name)
        np.testing.assert_array_equal(getattr(held, name), getattr(observed, name))
    for name in (
        "source_generation",
        "latest_real_frame_index",
        "latest_real_receive_timestamp_ns",
        "real_valid_frames_in_generation",
        "real_stream_ready",
        "playout_kind",
        "source_stale",
    ):
        assert getattr(held, name) == getattr(observed, name)


@pytest.mark.parametrize("alpha", [-0.01, 1.01, np.nan, np.inf])
def test_interpolation_rejects_invalid_alpha(alpha):
    with pytest.raises(ValueError):
        interpolate_smpl_reference(frame(0), frame(1), alpha)


def test_interpolation_rejects_anchor_quaternion_presence_mismatch():
    start = frame(0)
    end = frame(1)
    start.anchor_quat = np.tile(
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (10, 1)
    )

    with pytest.raises(ValueError):
        interpolate_smpl_reference(start, end, 0.5)


def test_hold_during_rearming_latches_current_interpolated_reference():
    gate = LiveReferenceGate()
    gate.observe(frame(0), received_mono=1.0)
    gate.hold()
    gate.observe(frame(4), received_mono=2.0)
    gate.begin_rearm()
    gate.set_rearm_progress(0.25)

    assert gate.hold() is True
    gate.observe(frame(9), received_mono=3.0)

    assert gate.mode is ReferenceGateMode.HOLD
    np.testing.assert_allclose(gate.active_reference().term1_local, 1.0)


def test_complete_rearm_returns_to_live_with_newest_observed_reference():
    gate = LiveReferenceGate()
    gate.observe(frame(0), received_mono=1.0)
    gate.hold()
    gate.observe(frame(2), received_mono=2.0)
    gate.begin_rearm()
    gate.set_rearm_progress(0.5)
    gate.observe(frame(8), received_mono=3.0)

    gate.complete_rearm()

    assert gate.mode is ReferenceGateMode.LIVE
    assert gate.active_reference().frame_index == 8
