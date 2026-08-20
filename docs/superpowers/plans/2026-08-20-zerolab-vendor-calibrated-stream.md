# ZeroLab Vendor-Calibrated Stream Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the duplicate ZeroLab runtime T-pose calibration, consume vendor-calibrated world orientations directly, and replace stale motor-frame freezing with a gated human-reference hold that keeps SONIC balance closed loop.

**Architecture:** Convert every valid vendor packet immediately, then use the existing 10-frame source window as the only stream-readiness gate. Add a small reference-gate unit between observed ZMQ windows and the active SONIC reference: armed stale input latches the human reference while inference keeps using current proprioception, recovered windows remain pending, and an explicit two-second reference-space rearm opens the gate. Keep live zero-command Normal before initial ARM and retain all established emergency routes.

**Tech Stack:** Ubuntu 22.04, ROS 2 Humble, Python 3.10, NumPy, SciPy quaternion math, pytest, YAML Mod manifests, colcon, MuJoCo.

## Global Constraints

- Treat ZeroLab `joint_quat_world` as vendor-calibrated world orientations after the vendor N-pose procedure.
- Apply only quaternion validation/normalization, cross-frame sign continuity, `(-qx, -qy, qz, qw)`, fixed ZeroLab-to-SMPL mapping, and parent-relative localization.
- Remove `TPoseCalibrator`, `apply_rest_alignment()`, `sampled_rest`, and the 100-frame runtime calibration delay from the ZeroLab path.
- Preserve exactly 10 consecutive source frames as the only application stream-readiness window.
- Preserve `btn_10=11` for ZeroLab entry and `btn_10=12` for explicit initial ARM and recovery ARM.
- Preserve source `stale_seconds=0.5`, policy `live_reference_timeout_s=0.5`, and initial/recovery blend duration `2.0` seconds.
- Apply live zero-command Normal during `WAIT_STREAM` and `WAIT_ARM`; apply a live-Normal/live-SONIC smoothstep during the first ARM blend.
- On stale input after full ARM, hold the last complete human reference, never a fixed motor frame, and continue SONIC inference against current robot feedback every control tick.
- Do not automatically activate recovered human input. Require a fresh pending 10-frame window and another `btn_10=12`.
- Perform recovery interpolation in reference space: linear for position fields and normalized hemisphere-corrected spherical interpolation for quaternion fields.
- Keep the PICO source, PICO calibration, PICO conversion, and ordinary `sonic_teleop` behavior unchanged.
- Do not change SONIC weights, action scale, joint gains, joint limits, MuJoCo XML, hardware drivers, or CAN handling.
- Do not deploy to or ARM robot hardware in this plan. MuJoCo acceptance and an independent CAN/`motor_timeout` clearance are mandatory later gates.
- Preserve the existing untracked `build-safe-arm/`, `install-safe-arm/`, `log-safe-arm/`, `build-live-normal/`, `install-live-normal/`, and `log-live-normal/` directories. Use new `*-vendor-stream` artifacts.

## File Structure

- Modify `src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/converter.py`: remove session rest calibration and emit a converted frame on the first valid packet.
- Modify `src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/source_node.py`: enforce 10 consecutive indices and replace calibration wording with stream-window wording.
- Create `src/bxi_example_py_elf3/mods/com.bxi.sonic/reference_gate.py`: own `SmplReferenceFrame`, deep-copy/interpolation helpers, and the LIVE/HOLD/REARMING reference gate.
- Modify `src/bxi_example_py_elf3/mods/com.bxi.sonic/policy.py`: feed decoded observed references through the gate and expose the narrow ZeroLab hold/rearm interface.
- Modify `src/bxi_example_py_elf3/mods/com.bxi.sonic/state.py`: extend the `SonicPolicy` protocol with the gate methods without changing ordinary PICO construction.
- Modify `src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/state.py`: rename phases, cancel an initial blend back to live Normal on stale, and keep live SONIC output during reference hold/rearm.
- Modify `src/bxi_example_py_elf3/mods/com.bxi.sonic/mod.yaml`: replace runtime T-pose prompts with the external N-pose and neutral-pose prerequisites.
- Modify `src/bxi_example_py_elf3/mods/com.bxi.sonic/README.md`: document the corrected workflow, state transitions, stale semantics, MuJoCo gates, and hardware block.
- Modify `src/bxi_example_py_elf3/test/test_zerolab_converter.py`: replace calibration tests with direct-conversion and continuity tests.
- Modify `src/bxi_example_py_elf3/test/test_zerolab_pose_contract.py`: cover strict 10-frame readiness, discontinuities, duplicate handling, stale refill, and source logs.
- Create `src/bxi_example_py_elf3/test/test_sonic_reference_gate.py`: test reference cloning, quaternion interpolation, observed/active separation, and recovery state.
- Modify `src/bxi_example_py_elf3/test/test_sonic_ordered_playout.py`: verify policy integration while preserving ordered PICO/ZeroLab playout behavior.
- Modify `src/bxi_example_py_elf3/test/test_zerolab_arming.py`: cover `WAIT_STREAM`, initial-stale return to Normal, closed-loop `HOLD_REFERENCE`, and explicit reference-space rearm.
- Modify `src/bxi_example_py_elf3/test/test_zerolab_manifest.py`: verify new prompts, external N-pose wording, and PICO isolation.

---

### Task 1: Replace Runtime T-Pose Calibration with Direct Vendor Conversion

**Files:**
- Modify: `src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/converter.py:1-290`
- Modify: `src/bxi_example_py_elf3/test/test_zerolab_converter.py:1-430`
- Read-only evidence: `/tmp/zerolab-arm-direction-20260819-002/{metadata.json,records.bin}`

**Interfaces:**
- Consumes: `ZeroLabPacket.joint_quat_world_xyzw: ndarray[(47,4), float32]`, already calibrated by the vendor N-pose flow.
- Produces: `ZeroLabMotionConverter.observe(packet: ZeroLabPacket) -> ConvertedPoseFrame` on the first valid packet.
- Preserves: `mark_stale()` and `reset_session()` as sign-continuity resets; neither method performs pose calibration.

- [ ] **Step 1: Capture the legacy recording baseline before changing converter code**

Run from `src/bxi_example_py_elf3`:

```bash
PYTHONPATH="$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
python3 - <<'PY'
from pathlib import Path
import numpy as np
from zerolab.converter import ZeroLabMotionConverter
from zerolab.protocol import parse_zerolab_packet
from zerolab.recording import iter_raw_records

recording = Path('/tmp/zerolab-arm-direction-20260819-002')
converter = ZeroLabMotionConverter()
indices, poses, joints, roots = [], [], [], []
for record in iter_raw_records(recording):
    packet = parse_zerolab_packet(
        record.payload,
        receive_timestamp_ns=record.receive_timestamp_ns,
        local_frame_index=record.local_frame_index,
        sender_address=('recording', 0),
    )
    frame = converter.observe(packet)
    if frame is None:
        continue
    indices.append(frame.frame_index)
    poses.append(frame.smpl_body_pose)
    joints.append(frame.smpl_joints)
    roots.append(frame.body_quat_w)
np.savez_compressed(
    '/tmp/zerolab-legacy-conversion.npz',
    frame_index=np.asarray(indices, dtype=np.int64),
    smpl_body_pose=np.asarray(poses, dtype=np.float32),
    smpl_joints=np.asarray(joints, dtype=np.float32),
    body_quat_w=np.asarray(roots, dtype=np.float32),
)
print(f'LEGACY_OUTPUT_FRAMES={len(indices)} FIRST={indices[0]} LAST={indices[-1]}')
PY
```

Expected: `LEGACY_OUTPUT_FRAMES` is positive and `/tmp/zerolab-legacy-conversion.npz` exists. The legacy converter requires a 100-frame stable window, but its >5-degree stability check can reset that window, so the first output may be later than frame 100; this supplied recording legitimately produces `FIRST=184` (the recording begins at frame 0).

- [ ] **Step 2: Write direct-conversion failing tests**

Remove imports and tests for `TPoseCalibrator` and `apply_rest_alignment`. Replace `test_converter_uses_100_frames_only_for_rest_then_emits_frame_101` with:

```python
def test_converter_emits_first_vendor_calibrated_packet():
    converter = ZeroLabMotionConverter()

    output = converter.observe(make_packet(0, identity47()))

    assert output.frame_index == 0
    assert output.receive_timestamp_ns == 0
    assert output.smpl_body_pose.shape == (21, 3)
    assert output.smpl_joints.shape == (24, 3)
    assert output.body_quat_w.shape == (4,)
    assert output.joint_pos.shape == (29,)
    np.testing.assert_allclose(output.smpl_body_pose, 0.0, atol=1e-6)


def test_converter_maps_vendor_world_pose_without_sampled_rest_inverse():
    vendor = identity47()
    vendor[3] = Rotation.from_euler("z", 30.0, degrees=True).as_quat()
    converter = ZeroLabMotionConverter()

    output = converter.observe(make_packet(0, vendor))

    xrt = unity_world_quaternions_to_xrt(vendor, (47, 4))
    expected_world = synthesize_smpl_world_quats(xrt[:BODY_JOINT_COUNT])
    body_poses = np.zeros((24, 7), dtype=np.float32)
    body_poses[:, 3:] = expected_world
    expected = compute_from_body_poses(SMPL24_PARENTS, body_poses)
    np.testing.assert_allclose(
        output.smpl_body_pose,
        expected["smpl_pose"][0, :63].reshape(21, 3),
        atol=1e-6,
    )
```

Change yaw, rigid-yaw, root-translation, elbow, and unused-joint tests so their first `observe()` result is used directly; remove every 100-frame warm-up loop. Replace the calibration reset test with:

```python
def test_stale_and_reset_session_clear_only_sign_continuity():
    converter = ZeroLabMotionConverter()
    first = converter.observe(make_packet(0, identity47()))
    assert first is not None

    converter.mark_stale()
    after_stale = converter.observe(make_packet(1, -identity47()))
    assert after_stale is not None

    converter.reset_session()
    after_reset = converter.observe(make_packet(2, identity47()))
    assert after_reset is not None
```

- [ ] **Step 3: Run focused tests and verify RED**

```bash
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
python3 -m pytest -q test/test_zerolab_converter.py
```

Expected: FAIL because the first packet still returns `None` and the deleted calibration symbols still exist in production.

- [ ] **Step 4: Implement the minimal direct converter**

Delete `apply_rest_alignment`, `TPoseCalibrator`, `_calibrator`, and all calibrator branches. Make the converter session state exactly:

```python
class ZeroLabMotionConverter:
    """Convert vendor-calibrated ZeroLab world poses to SONIC frames."""

    def __init__(self) -> None:
        self._previous_raw_quats_xyzw = None

    def mark_stale(self) -> None:
        self._previous_raw_quats_xyzw = None

    def reset_session(self) -> None:
        self._previous_raw_quats_xyzw = None

    def observe(self, packet: ZeroLabPacket) -> ConvertedPoseFrame:
        raw_quats = unity_world_quaternions_to_xrt(
            packet.joint_quat_world_xyzw, _PACKET_QUATERNION_SHAPE
        )
        raw_quats = align_quaternion_signs(
            raw_quats, self._previous_raw_quats_xyzw
        )
        smpl_world_quats = synthesize_smpl_world_quats(
            raw_quats[:BODY_JOINT_COUNT]
        )
```

Keep the existing root conversion, FK call, array validation, wrist conversion, output construction, and final assignment `self._previous_raw_quats_xyzw = raw_quats` unchanged after this new prefix. Update the module docstring so it no longer says calibration.

- [ ] **Step 5: Run focused tests and verify GREEN**

```bash
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
python3 -m pytest -q test/test_zerolab_converter.py
```

Expected: PASS, with no test importing or mentioning `TPoseCalibrator`, `sampled_rest`, or `apply_rest_alignment`.

- [ ] **Step 6: Commit the direct-conversion change**

```bash
git add \
  src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/converter.py \
  src/bxi_example_py_elf3/test/test_zerolab_converter.py
git commit -m "fix: consume vendor-calibrated ZeroLab poses"
```

---

### Task 2: Make the 10-Frame Window the Only Stream-Readiness Gate

**Files:**
- Modify: `src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/source_node.py:134-250,414-475`
- Modify: `src/bxi_example_py_elf3/test/test_zerolab_pose_contract.py:1-360`
- Modify: `src/bxi_example_py_elf3/test/test_zerolab_lifecycle.py` source-log assertions

**Interfaces:**
- Consumes: one `ConvertedPoseFrame` per valid packet.
- Produces: a source chunk only after 10 consecutive frame indices; the legacy wire field `calibration_ready=True` remains for bridge compatibility but means window-ready, not a new human calibration.
- Duplicate contract: same index returns `None` without advancing or clearing.
- Discontinuity contract: backward index is rejected after clearing; forward gap clears and uses the current valid frame as the first frame of a new sequence.

- [ ] **Step 1: Rewrite source-window tests for immediate conversion**

Change the first-ready assertion to:

```python
def test_core_first_ready_chunk_is_source_frames_zero_through_nine():
    core = ZeroLabSourceCore(ZeroLabMotionConverter())
    for index in range(9):
        assert core.accept(identity_packet(index)) is None

    fields = core.accept(identity_packet(9))

    np.testing.assert_array_equal(fields["frame_index"], np.arange(10))
```

Add exact index behavior:

```python
def test_duplicate_is_ignored_without_revoking_partial_window():
    window = PoseChunkWindow(10)
    for index in range(5):
        assert window.append(converted(index)) is None
    assert window.append(converted(4)) is None
    for index in range(5, 10):
        fields = window.append(converted(index))
    np.testing.assert_array_equal(fields["frame_index"], np.arange(10))


def test_forward_gap_starts_a_new_consecutive_window():
    window = PoseChunkWindow(10)
    for index in range(5):
        window.append(converted(index))
    assert window.append(converted(8)) is None
    for index in range(9, 18):
        fields = window.append(converted(index))
    np.testing.assert_array_equal(fields["frame_index"], np.arange(8, 18))


def test_backward_index_clears_and_is_rejected():
    window = PoseChunkWindow(10)
    window.append(converted(5))
    with pytest.raises(ValueError, match="backward"):
        window.append(converted(4))
    assert window.ready is False
```

Update stale tests to feed 10 frames before the first ready result and 10 new frames after `500_000_001` ns. Remove the obsolete test for restarting a 100-frame calibration window. Update lifecycle expectations from `ZeroLab collecting T-pose calibration` to `ZeroLab waiting for 10-frame stream window`.

- [ ] **Step 2: Run focused source tests and verify RED**

```bash
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
python3 -m pytest -q \
  test/test_zerolab_pose_contract.py \
  test/test_zerolab_lifecycle.py
```

Expected: duplicate and gap tests FAIL because the current window rejects duplicates and accepts forward gaps; log assertions FAIL on T-pose wording.

- [ ] **Step 3: Implement strict consecutive-window semantics**

In `PoseChunkWindow.append`, validate the frame first, then apply this index logic before appending:

```python
if self._last_frame_index is not None:
    delta = frame_index - self._last_frame_index
    if delta == 0:
        return None
    if delta < 0:
        self.clear()
        raise ValueError("frame_index moved backward")
    if delta > 1:
        self.clear()
```

Keep invalid-frame validation before mutating `_frames`, so an invalid index cannot advance the window. Remove the now-unreachable `if frame is None` branch from `ZeroLabSourceCore.accept`.

Change the source state message to:

```python
if state == "collecting":
    self.get_logger().info("ZeroLab waiting for 10-frame stream window")
```

Do not rename or remove `calibration_ready` in the packed source schema; the shared bridge still consumes that compatibility field.

- [ ] **Step 4: Run focused tests and verify GREEN**

```bash
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
python3 -m pytest -q \
  test/test_zerolab_pose_contract.py \
  test/test_zerolab_lifecycle.py
```

Expected: PASS; the first ready chunk is indices `0..9`, and every stale recovery requires 10 new frames.

- [ ] **Step 5: Commit stream-readiness behavior**

```bash
git add \
  src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/source_node.py \
  src/bxi_example_py_elf3/test/test_zerolab_pose_contract.py \
  src/bxi_example_py_elf3/test/test_zerolab_lifecycle.py
git commit -m "fix: gate ZeroLab readiness on consecutive frames"
```

---

### Task 3: Add an Isolated Human-Reference Gate and Interpolator

**Files:**
- Create: `src/bxi_example_py_elf3/mods/com.bxi.sonic/reference_gate.py`
- Create: `src/bxi_example_py_elf3/test/test_sonic_reference_gate.py`
- Modify: `src/bxi_example_py_elf3/mods/com.bxi.sonic/policy.py:90-130,1190-1210` to import and re-export `SmplReferenceFrame`

**Interfaces:**
- Produces: `ReferenceGateMode {LIVE,HOLD,REARMING}`.
- Produces: `copy_smpl_reference(frame) -> SmplReferenceFrame` with non-aliased arrays.
- Produces: `interpolate_smpl_reference(start, end, alpha) -> SmplReferenceFrame`.
- Produces: `LiveReferenceGate.observe`, `hold`, `begin_rearm`, `set_rearm_progress`, `complete_rearm`, `active_reference`, `has_fresh_observed`, and `reset`.
- Does not import `policy.py`; `policy.py` imports this focused module, avoiding a circular dependency.

- [ ] **Step 1: Write failing gate and interpolation tests**

Create the test with a dynamic loader matching `test_sonic_ordered_playout.py`, then add:

```python
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
```

Also test deep-copy non-aliasing, invalid alpha, anchor-quaternion presence mismatch, `hold()` during REARMING latching the current interpolated result, and `complete_rearm()` returning to LIVE with the newest observed reference.

- [ ] **Step 2: Run the new test and verify RED**

```bash
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
python3 -m pytest -q test/test_sonic_reference_gate.py
```

Expected: collection FAIL because `reference_gate.py` does not exist.

- [ ] **Step 3: Implement the reference data type and interpolation helpers**

Move the complete existing `SmplReferenceFrame` dataclass from `policy.py` into `reference_gate.py`. Implement copies with `dataclasses.replace` and copies of `term1_local`, `root_quat`, `wrist`, `head_joint_pos`, and optional `anchor_quat`.

Use this batch quaternion interpolation core for WXYZ arrays:

```python
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
```

`interpolate_smpl_reference` linearly blends `term1_local`, `wrist`, and `head_joint_pos`, slerps `root_quat` and matching optional `anchor_quat`, and takes metadata from `end` via `replace`. Require `0.0 <= alpha <= 1.0` and matching anchor presence.

- [ ] **Step 4: Implement the gate state machine**

Use this state ownership and these exact public methods:

```python
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
```

LIVE makes active follow observed. HOLD deep-copies the current active reference. REARMING returns `interpolate_smpl_reference(latched, observed, progress)` on every call, so a moving fresh endpoint is approached without a final jump. Calling `hold()` in REARMING first captures the current interpolated reference.

Import these names into `policy.py` and retain `SmplReferenceFrame` in `policy.__all__` so existing imports remain compatible.

- [ ] **Step 5: Run focused tests and verify GREEN**

```bash
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
python3 -m pytest -q \
  test/test_sonic_reference_gate.py \
  test/test_sonic_ordered_playout.py
```

Expected: PASS; moving the dataclass does not alter existing ordered playout.

- [ ] **Step 6: Commit the isolated gate**

```bash
git add \
  src/bxi_example_py_elf3/mods/com.bxi.sonic/reference_gate.py \
  src/bxi_example_py_elf3/mods/com.bxi.sonic/policy.py \
  src/bxi_example_py_elf3/test/test_sonic_reference_gate.py
git commit -m "feat: add SONIC live reference gate"
```

---

### Task 4: Integrate Observed and Active References into SONIC Policy

**Files:**
- Modify: `src/bxi_example_py_elf3/mods/com.bxi.sonic/policy.py:315-410,610-760,944-970,1080-1130`
- Modify: `src/bxi_example_py_elf3/mods/com.bxi.sonic/state.py:35-75`
- Modify: `src/bxi_example_py_elf3/test/test_sonic_ordered_playout.py:130-360`

**Interfaces:**
- `has_fresh_live_reference()` reports observed-source freshness even while active input is held.
- `hold_live_reference() -> bool` latches the active human window.
- `begin_live_reference_rearm() -> bool`, `set_live_reference_rearm_progress(alpha)`, and `complete_live_reference_rearm()` control reference-space recovery.
- Ordinary PICO never calls hold/rearm, so its gate remains LIVE and behavior is unchanged.

- [ ] **Step 1: Add failing policy-gate integration tests**

Extend ordered playout tests:

```python
def test_policy_hold_keeps_reference_but_continues_inference(monkeypatch):
    policy = _make_policy(monkeypatch)
    now = time.monotonic()
    policy._merge_source_fields(_source_fields(0), now)
    policy.inference_step(*_observation())
    held_term = policy._backend.inputs[-1][0, 0]
    calls = len(policy._backend.inputs)

    assert policy.hold_live_reference() is True
    policy._merge_source_fields(_source_fields(100), now + 0.1)
    policy.inference_step(*_observation())

    assert len(policy._backend.inputs) == calls + 1
    assert policy._backend.inputs[-1][0, 0] == held_term
    assert policy.has_fresh_live_reference(0.5)


def test_policy_rearm_exposes_interpolated_reference(monkeypatch):
    policy = _make_policy(monkeypatch)
    now = time.monotonic()
    policy._merge_source_fields(_source_fields(0), now)
    policy.inference_step(*_observation())
    policy.hold_live_reference()
    policy._merge_source_fields(_source_fields(10, epoch=2), now + 0.1)
    assert policy.begin_live_reference_rearm() is True
    policy.set_live_reference_rearm_progress(0.5)

    policy.inference_step(*_observation())

    assert policy._backend.inputs[-1][0, 0] == pytest.approx(5.0)
```

Add a reset test asserting gate mode returns LIVE, and preserve the existing `test_policy_holds_last_complete_window_after_disconnect` assertion.

- [ ] **Step 2: Run policy tests and verify RED**

```bash
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
python3 -m pytest -q test/test_sonic_ordered_playout.py
```

Expected: FAIL because `SonicTeleopPolicy` exposes no gate methods and always makes the newest decoded reference active.

- [ ] **Step 3: Route decoded observations through `LiveReferenceGate`**

Initialize `self._live_reference_gate = LiveReferenceGate()` next to `latest_live_ref`, and call `reset()` on it from `SonicTeleopPolicy.reset`.

Whenever `poll_reference()` creates or rebuilds `latest_live_ref`, keep that property as the observed reference for compatibility and add:

```python
self._live_reference_gate.observe(
    self.latest_live_ref,
    self.latest_live_ref_time,
)
```

Change `_active_reference()` to poll first and then select:

```python
self.poll_reference()
live = self._live_reference_gate.active_reference()
now_mono = time.monotonic()
if live is not None:
    # retain epoch/yaw handling and stale calculation
    return live, "live", now_mono
return self._offline_frame(), "idle", now_mono
```

Keep freshness based on `latest_live_ref_time`, which is observed receive time, not the latched active frame's metadata.

- [ ] **Step 4: Expose the narrow gate-control API**

Add to `SonicTeleopPolicy`:

```python
def hold_live_reference(self) -> bool:
    self.poll_reference()
    return self._live_reference_gate.hold()

def begin_live_reference_rearm(self) -> bool:
    self.poll_reference()
    return self._live_reference_gate.begin_rearm()

def set_live_reference_rearm_progress(self, alpha: float) -> None:
    self._live_reference_gate.set_rearm_progress(alpha)

def complete_live_reference_rearm(self) -> None:
    self._live_reference_gate.complete_rearm()
```

Add the same signatures to `SonicPolicy` in `state.py`. Update policy status strings so HOLD reports `held_reference` and REARMING reports `rearming_reference`, while source freshness remains independently available through `has_fresh_live_reference`.

- [ ] **Step 5: Run policy, gate, and PICO regression tests**

```bash
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
python3 -m pytest -q \
  test/test_sonic_reference_gate.py \
  test/test_sonic_ordered_playout.py \
  test/test_sonic_python_runtime.py
```

Expected: PASS. Existing ordered consumption, backend-failure cursor behavior, epoch restart, and PICO runtime tests remain unchanged.

- [ ] **Step 6: Commit policy integration**

```bash
git add \
  src/bxi_example_py_elf3/mods/com.bxi.sonic/policy.py \
  src/bxi_example_py_elf3/mods/com.bxi.sonic/state.py \
  src/bxi_example_py_elf3/test/test_sonic_ordered_playout.py
git commit -m "feat: gate active SONIC human references"
```

---

### Task 5: Replace Frozen-Motor Stale Hold in the ZeroLab State Machine

**Files:**
- Modify: `src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/state.py:20-330`
- Modify: `src/bxi_example_py_elf3/test/test_zerolab_arming.py:20-430`

**Interfaces:**
- Produces phases: `WAIT_STREAM`, `WAIT_ARM`, `BLENDING`, `ARMED`, `HOLD_REFERENCE`, `REARMING`.
- Initial stale: `BLENDING -> WAIT_STREAM`, output live Normal, new explicit ARM required.
- Armed stale: `ARMED -> HOLD_REFERENCE`, output every newly computed SONIC motor frame.
- Recovery: `HOLD_REFERENCE + fresh observed + btn_10=12 -> REARMING -> ARMED` using policy reference progress, not motor-frame interpolation.

- [ ] **Step 1: Extend `FakePolicy` with reference-gate behavior**

In `test_zerolab_arming.py`, add:

```python
self.held = False
self.rearming = False
self.rearm_progress = 0.0

def hold_live_reference(self):
    self.held = True
    self.rearming = False
    return True

def begin_live_reference_rearm(self):
    if not self.fresh or not self.held:
        return False
    self.rearming = True
    return True

def set_live_reference_rearm_progress(self, alpha):
    self.rearm_progress = float(alpha)

def complete_live_reference_rearm(self):
    self.rearming = False
    self.held = False
```

Keep `step()` incrementing on every advancing tick; tests use that counter to prove closed-loop execution continues.

- [ ] **Step 2: Replace frozen-frame tests with closed-loop state tests**

Rename every `WAIT_CALIBRATION` assertion to `WAIT_STREAM`. Replace frozen stale tests with:

```python
def test_initial_blend_stale_cancels_to_live_normal():
    state, sonic, normal, ctx = armed_state_at_half_blend()
    sonic.fresh = False
    normal.target.position.fill(0.75)

    frame = state.sample_running_frame(ctx, 0.02, advance=True)

    assert state.arm_phase is ZeroLabArmPhase.WAIT_STREAM
    np.testing.assert_allclose(frame.qpos, 0.75)
    assert normal.step_calls >= 1


def test_armed_stale_holds_reference_but_keeps_sonic_closed_loop():
    state, sonic, normal, ctx = fully_armed_state()
    sonic.fresh = False
    sonic.target.position.fill(3.0)
    calls = sonic.step_calls

    first = copy_motor_frame(state.sample_running_frame(ctx, 0.02, advance=True))
    sonic.target.position.fill(4.0)
    second = copy_motor_frame(state.sample_running_frame(ctx, 0.02, advance=True))

    assert state.arm_phase is ZeroLabArmPhase.HOLD_REFERENCE
    assert sonic.held is True
    assert sonic.step_calls == calls + 2
    np.testing.assert_allclose(first.qpos, 3.0)
    np.testing.assert_allclose(second.qpos, 4.0)


def test_recovered_reference_stays_gated_until_explicit_rearm():
    state, sonic, normal, ctx = reference_hold_state()
    sonic.fresh = True
    state.sample_running_frame(ctx, 0.02, advance=True)
    assert state.arm_phase is ZeroLabArmPhase.HOLD_REFERENCE
    assert sonic.rearming is False

    state.on_action(ctx, "arm_zerolab")
    state.sample_running_frame(ctx, 1.0, advance=True)
    assert state.arm_phase is ZeroLabArmPhase.REARMING
    assert sonic.rearm_progress == pytest.approx(0.5)
```

Add tests for `WAIT_ARM -> WAIT_STREAM` on stale, rearm completion at 2 seconds, stale during REARMING returning to HOLD_REFERENCE, duplicate ARM ignore, one log per stale/recovery session, exit resetting to WAIT_STREAM, and emergency/orientation routes in every new phase.

- [ ] **Step 3: Run arming tests and verify RED**

```bash
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
python3 -m pytest -q test/test_zerolab_arming.py
```

Expected: FAIL on missing phase names and because existing HOLD_STALE returns `_hold_frame` instead of live SONIC output.

- [ ] **Step 4: Simplify phase and buffer ownership**

Define:

```python
class ZeroLabArmPhase(str, Enum):
    WAIT_STREAM = "wait_stream"
    WAIT_ARM = "wait_arm"
    BLENDING = "blending"
    ARMED = "armed"
    HOLD_REFERENCE = "hold_reference"
    REARMING = "rearming"
```

Remove `ZeroLabBlendSource.FROZEN`, `_blend_start_frame`, and every stale copy into `_hold_frame`. Keep one entry frame for `get_entry_frame`, plus `_applied_frame`, `_live_frame`, and `_normal_frame`.

- [ ] **Step 5: Implement live output for every stale branch**

Before SONIC inference in REARMING, advance progress using control `dt`:

```python
if self._arm_phase is ZeroLabArmPhase.REARMING:
    self._blend_elapsed_s += max(dt, 0.0)
    alpha = self._smoothstep(
        self._blend_elapsed_s / self.arm_blend_seconds
    )
    self.policy.set_live_reference_rearm_progress(alpha)
```

After resolving the live SONIC frame and freshness:

```python
if self._arm_phase is ZeroLabArmPhase.WAIT_ARM and not fresh_reference:
    self._set_phase(
        ZeroLabArmPhase.WAIT_STREAM,
        "ZeroLab ARM phase: WAIT_STREAM",
        warning=True,
    )

if self._arm_phase is ZeroLabArmPhase.BLENDING and not fresh_reference:
    self._blend_elapsed_s = 0.0
    self._set_phase(
        ZeroLabArmPhase.WAIT_STREAM,
        "ZeroLab initial ARM cancelled; reference stale; returning to live Normal",
        warning=True,
    )
    normal = self._sample_normal_frame(ctx, dt, advance=True)
    return self._copy_frame(self._applied_frame, normal)

if self._arm_phase in (ZeroLabArmPhase.ARMED, ZeroLabArmPhase.REARMING) \
        and not fresh_reference:
    self.policy.hold_live_reference()
    self._blend_elapsed_s = 0.0
    self._set_phase(
        ZeroLabArmPhase.HOLD_REFERENCE,
        "ZeroLab reference stale; holding human reference while SONIC balance continues",
        warning=True,
    )
    return self._copy_frame(self._applied_frame, self._live_frame)
```

In `HOLD_REFERENCE`, always return `_live_frame`; log fresh pending data once but do not open the gate. In REARMING, return `_live_frame`, then call `complete_live_reference_rearm()` and enter ARMED when elapsed reaches two seconds.

- [ ] **Step 6: Implement explicit recovery ARM**

Split initial and recovery actions:

```python
if self._arm_phase is ZeroLabArmPhase.WAIT_ARM and fresh:
    self._blend_elapsed_s = 0.0
    self._set_phase(
        ZeroLabArmPhase.BLENDING,
        f"ZeroLab ARM accepted; blending live Normal -> SONIC for {self.arm_blend_seconds:.3f} s",
    )
elif self._arm_phase is ZeroLabArmPhase.HOLD_REFERENCE and fresh:
    if not self.policy.begin_live_reference_rearm():
        self.logger.warning("ZeroLab recovery ARM refused; no pending reference")
    else:
        self._blend_elapsed_s = 0.0
        self._set_phase(
            ZeroLabArmPhase.REARMING,
            f"ZeroLab recovery ARM accepted; blending human reference for {self.arm_blend_seconds:.3f} s",
        )
```

Retain bounded refusal/ignore logs for every other phase and retain existing live-Normal orientation handling for WAIT_STREAM, WAIT_ARM, and initial BLENDING.

- [ ] **Step 7: Run arming and manifest lifecycle tests**

```bash
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
python3 -m pytest -q \
  test/test_zerolab_arming.py \
  test/test_zerolab_manifest.py \
  test/test_zerolab_lifecycle.py
```

Expected: PASS, with no assertion expecting a repeated motor frame or `HOLD_STALE`.

- [ ] **Step 8: Commit the state-machine correction**

```bash
git add \
  src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/state.py \
  src/bxi_example_py_elf3/test/test_zerolab_arming.py
git commit -m "fix: keep SONIC balance closed loop on stale input"
```

---

### Task 6: Replace Operator Prompts and Document the Correct Workflow

**Files:**
- Modify: `src/bxi_example_py_elf3/mods/com.bxi.sonic/mod.yaml:295-315`
- Modify: `src/bxi_example_py_elf3/mods/com.bxi.sonic/README.md:520-650`
- Modify: `src/bxi_example_py_elf3/test/test_zerolab_manifest.py:35-250`

**Interfaces:**
- `confirm_message`: external vendor N-pose is complete before ZeroLab entry.
- `operator_prompt`: robot remains live Normal until stream ready; safety observer explicitly authorizes ARM from neutral.
- No active runtime text asks for application-level T-pose calibration.

- [ ] **Step 1: Write failing manifest wording assertions**

Use these exact strings:

```python
assert manifest["states"]["sonic_zerolab"]["manifest"]["confirm_message"] == (
    "请先在ZeroLab厂家软件完成N-pose标定并回到中立姿势；"
    "进入后等待ZeroLab stream ready"
)
assert params["operator_prompt"] == (
    "机器人保持实时Normal直到stream ready；确认操作者处于中立姿势后，"
    "由安全员发送btn_10=12"
)
```

Update lifecycle logger expectations to `WAIT_STREAM` and the new prompt. Add a source-tree text assertion that `converter.py`, `source_node.py`, `zerolab/state.py`, and the ZeroLab section of `mod.yaml` contain neither `TPoseCalibrator` nor `T-pose标定`.

- [ ] **Step 2: Run manifest tests and verify RED**

```bash
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
python3 -m pytest -q test/test_zerolab_manifest.py
```

Expected: FAIL on old T-pose prompts and old phase logs.

- [ ] **Step 3: Update Mod prompts and README**

Set `mod.yaml` to:

```yaml
confirm_message: >-
  请先在ZeroLab厂家软件完成N-pose标定并回到中立姿势；进入后等待ZeroLab stream ready
params:
  operator_prompt: >-
    机器人保持实时Normal直到stream ready；确认操作者处于中立姿势后，由安全员发送btn_10=12
```

Rewrite the README ZeroLab workflow to show:

```text
vendor N-pose -> operator neutral -> btn_10=11 -> WAIT_STREAM
-> WAIT_ARM -> btn_10=12 -> 2 s BLENDING -> ARMED

ARMED + stale -> HOLD_REFERENCE
HOLD_REFERENCE + recovered stream -> still gated
HOLD_REFERENCE + btn_10=12 -> 2 s REARMING -> ARMED
```

State explicitly that the legacy wire field `calibration_ready` means a complete source window for compatibility, not a new operator calibration. Document 0.6 s, 2 s, and 30 s MuJoCo dropout tests and the CAN/`motor_timeout` hardware block.

- [ ] **Step 4: Run wording checks and tests**

```bash
if rg -n 'TPoseCalibrator|collecting T-pose|T-pose标定|WAIT_CALIBRATION|HOLD_STALE|holding last motor frame' \
  mods/com.bxi.sonic/zerolab \
  mods/com.bxi.sonic/mod.yaml \
  mods/com.bxi.sonic/README.md
then
  echo 'Unexpected legacy ZeroLab wording remains'
  false
fi

PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
python3 -m pytest -q \
  test/test_zerolab_manifest.py \
  test/test_zerolab_arming.py
```

Expected: `rg` prints no matches and pytest passes. PICO's separate vendor calibration wording outside these ZeroLab paths is not changed.

- [ ] **Step 5: Commit operator-facing behavior**

```bash
git add \
  src/bxi_example_py_elf3/mods/com.bxi.sonic/mod.yaml \
  src/bxi_example_py_elf3/mods/com.bxi.sonic/README.md \
  src/bxi_example_py_elf3/test/test_zerolab_manifest.py
git commit -m "docs: correct ZeroLab calibrated-stream workflow"
```

---

### Task 7: Full Regression, Recorded Comparison, Build, and MuJoCo Gate

**Files:**
- Verify only: all modified production and test files above.
- Create isolated artifacts: `build-vendor-stream/`, `install-vendor-stream/`, `log-vendor-stream/`.
- Evidence outputs only: `/tmp/zerolab-direct-conversion.npz`, `/tmp/zerolab-vendor-stream-pytest.log`, `/tmp/zerolab-vendor-stream-colcon.log`, `/tmp/zerolab-vendor-stream-mujoco.log`.

**Interfaces:**
- Produces no hardware deployment artifact or robot-side mutation.
- Produces a tested local install at `install-vendor-stream` only after all source tests pass.

- [ ] **Step 1: Run the focused ZeroLab and shared SONIC regression suite**

From `src/bxi_example_py_elf3`:

```bash
set -o pipefail
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
python3 -m pytest -q \
  test/test_zerolab_protocol.py \
  test/test_zerolab_udp_receiver.py \
  test/test_zerolab_converter.py \
  test/test_zerolab_recording.py \
  test/test_zerolab_pose_contract.py \
  test/test_zerolab_lifecycle.py \
  test/test_zerolab_manifest.py \
  test/test_zerolab_arming.py \
  test/test_sonic_reference_gate.py \
  test/test_sonic_ordered_playout.py \
  test/test_sonic_python_runtime.py 2>&1 | \
  tee /tmp/zerolab-vendor-stream-pytest.log
```

Expected: all selected tests PASS; no skipped test may hide converter, gate, state, or PICO coverage.

- [ ] **Step 2: Run the complete package test suite**

```bash
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
python3 -m pytest -q test
```

Expected: PASS. Investigate any unrelated failure before proceeding; do not waive it as a ZeroLab exception.

- [ ] **Step 3: Export the revised recording conversion**

```bash
PYTHONPATH="$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
python3 - <<'PY'
from pathlib import Path
import numpy as np
from zerolab.converter import ZeroLabMotionConverter
from zerolab.protocol import parse_zerolab_packet
from zerolab.recording import iter_raw_records

recording = Path('/tmp/zerolab-arm-direction-20260819-002')
converter = ZeroLabMotionConverter()
indices, poses, joints, roots = [], [], [], []
for record in iter_raw_records(recording):
    packet = parse_zerolab_packet(
        record.payload,
        receive_timestamp_ns=record.receive_timestamp_ns,
        local_frame_index=record.local_frame_index,
        sender_address=('recording', 0),
    )
    frame = converter.observe(packet)
    indices.append(frame.frame_index)
    poses.append(frame.smpl_body_pose)
    joints.append(frame.smpl_joints)
    roots.append(frame.body_quat_w)
np.savez_compressed(
    '/tmp/zerolab-direct-conversion.npz',
    frame_index=np.asarray(indices, dtype=np.int64),
    smpl_body_pose=np.asarray(poses, dtype=np.float32),
    smpl_joints=np.asarray(joints, dtype=np.float32),
    body_quat_w=np.asarray(roots, dtype=np.float32),
)
print(f'DIRECT_OUTPUT_FRAMES={len(indices)} FIRST={indices[0]} LAST={indices[-1]}')
PY
```

Expected: direct output includes the recording's first packet rather than dropping 100 frames.

- [ ] **Step 4: Compare common recorded indices and shoulder chains**

```bash
python3 - <<'PY'
import numpy as np
old = np.load('/tmp/zerolab-legacy-conversion.npz')
new = np.load('/tmp/zerolab-direct-conversion.npz')
common, old_i, new_i = np.intersect1d(
    old['frame_index'], new['frame_index'], return_indices=True
)
assert common.size > 0
delta = np.linalg.norm(
    old['smpl_body_pose'][old_i] - new['smpl_body_pose'][new_i], axis=2
)
print(f'COMMON_FRAMES={common.size}')
print(f'LEFT_SHOULDER_MEAN_DELTA_RAD={delta[:, 15].mean():.6f}')
print(f'RIGHT_SHOULDER_MEAN_DELTA_RAD={delta[:, 16].mean():.6f}')
print(f'LEFT_ELBOW_MEAN_DELTA_RAD={delta[:, 17].mean():.6f}')
print(f'RIGHT_ELBOW_MEAN_DELTA_RAD={delta[:, 18].mean():.6f}')
assert np.isfinite(delta).all()
PY
```

Expected: finite common-frame comparison with nonzero shoulder deltas, demonstrating that the duplicate sampled-rest transform was removed. Direction correctness is decided by the known direct-conversion unit expectations and MuJoCo observation, not by assuming the old output was correct.

- [ ] **Step 5: Build into isolated vendor-stream directories**

From the worktree root:

```bash
set -eo pipefail
source /opt/ros/humble/setup.bash
source /home/fazepurple/ros2_ws/bxi_ros2_pkg/setup.bash

colcon --log-base log-vendor-stream build \
  --merge-install \
  --base-paths src \
  --packages-select bxi_example_py_elf3 \
  --allow-overriding bxi_example_py_elf3 \
  --build-base build-vendor-stream \
  --install-base install-vendor-stream \
  --cmake-args -DCMAKE_BUILD_TYPE=Release 2>&1 | \
  tee /tmp/zerolab-vendor-stream-colcon.log

source install-vendor-stream/setup.bash
test "$(ros2 pkg prefix bxi_example_py_elf3)" = "$PWD/install-vendor-stream"
```

Expected: build succeeds and package prefix is the isolated install.

- [ ] **Step 6: Run MuJoCo without any hardware process**

Terminal 1, from the worktree root:

```bash
set -eo pipefail
source /opt/ros/humble/setup.bash
source /home/fazepurple/ros2_ws/bxi_ros2_pkg/setup.bash
source "$PWD/install-vendor-stream/setup.bash"
export ROS_DOMAIN_ID=42

if pgrep -af '[h]ardware_elf3|[r]os2 launch.*example_demo_hw'; then
  echo 'STOP: hardware process detected on simulation host'
  exit 1
fi

ros2 launch bxi_example_py_elf3 example_demo.launch.py 2>&1 | \
  tee /tmp/zerolab-vendor-stream-mujoco.log
```

Terminal 2 uses the same three setup files and `ROS_DOMAIN_ID=42`, then sends one pulse at a time:

```bash
ros2 topic pub --once \
  /motion_commands communication/msg/MotionCommands '{btn_10: 11}'
ros2 topic pub --once \
  /motion_commands communication/msg/MotionCommands '{}'
```

After `WAIT_ARM`, operator neutral, and safety confirmation:

```bash
ros2 topic pub --once \
  /motion_commands communication/msg/MotionCommands '{btn_10: 12}'
ros2 topic pub --once \
  /motion_commands communication/msg/MotionCommands '{}'
```

Expected: `WAIT_STREAM -> WAIT_ARM -> BLENDING -> ARMED`, neutral arms agree with the operator, and small left/right/down/forward motions have correct direction without joint-limit contact.

- [ ] **Step 7: Exercise stale hold and explicit recovery in MuJoCo**

Pause the ZeroLab sender for 0.6 seconds, 2 seconds, and 30 seconds in separate trials. For each trial verify:

```bash
rg -n 'HOLD_REFERENCE|REARMING|holding last motor frame|reference stale' \
  /tmp/zerolab-vendor-stream-mujoco.log | tail -n 40
```

Expected before recovery ARM: `HOLD_REFERENCE` appears, `holding last motor frame` never appears, the robot remains standing, and resumed UDP data does not resume operator motion. Then send the same `btn_10=12` pulse and verify `REARMING -> ARMED` after two seconds.

- [ ] **Step 8: Run final repository checks**

```bash
git diff --check
git status --short
git log --oneline -8
```

Expected: no tracked uncommitted implementation changes; only the known isolated build/install/log directories may remain untracked. Do not copy, build, or launch anything on `192.168.88.172` as part of this plan.
