# ZeroLab Applied Action History Synchronization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every ZeroLab SONIC inference consume the previous motor target actually applied by the controller, eliminating the unexecuted action history accumulated before and during initial ARM.

**Architecture:** Add one narrowly scoped policy method that converts a named, policy-order applied qpos target into the existing normalized action representation. At the ZeroLab state boundary, compile one named joint map from `ctx.last_motor_frame.layout` to `ELF3_POLICY_JOINTS`; immediately before each advancing SONIC step, map the previous applied frame and overwrite `last_action`, which `_update_history()` then appends causally.

**Tech Stack:** Ubuntu 22.04, ROS 2 Humble, Python 3.10, NumPy, pytest, ONNX Runtime, named `JointLayout`/`CompiledJointMap`, colcon, MuJoCo.

## Global Constraints

- The change is ZeroLab-only; ordinary SONIC and PICO behavior remain unchanged.
- Do not change `sonic.onnx`, `stream_reference.npz`, `yaw_bias_rad`, gains, action scale, joint limits, pose conversion, resampling, stale hold, or automatic recovery.
- Initial activation still requires `btn_10=12` and still uses the existing two-second smoothstep blend.
- The previous applied frame must be mapped by joint name; never assume numeric order.
- Every applied target must be exactly 29 finite policy-order values after mapping and must be clipped to the existing `[-20, 20]` model action domain.
- A non-advancing scheduler call must not record or shift history.
- No hardware process may start before focused tests, full package regression, real-ONNX replay, isolated build, and MuJoCo neutral ARM gates pass.

## File Structure

- Modify `src/bxi_example_py_elf3/mods/com.bxi.sonic/policy.py`: expose the validated applied-qpos-to-`last_action` conversion; keep inference and output ownership unchanged.
- Modify `src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/state.py`: own the named mapping from the previous resolved motor frame and synchronize it immediately before advancing SONIC.
- Modify `src/bxi_example_py_elf3/test/test_sonic_ordered_playout.py`: unit-test the policy conversion, clipping, validation, and absence of output mutation.
- Modify `src/bxi_example_py_elf3/test/test_zerolab_arming.py`: component-test named mapping, causal call order, all phases, blends, and non-advancing calls.
- Evidence only `/tmp/zerolab_real_onnx_ab.py` and `/tmp/zerolab-applied-history-onnx.log`: replay captured neutral data through the real model without becoming runtime dependencies.
- Runtime only `build-applied-history/`, `install-applied-history/`, `log-applied-history/`: isolated simulation candidate; never overwrite the wired or wireless candidates.

---

### Task 1: Validated Applied-Target Policy Contract

**Files:**
- Modify: `src/bxi_example_py_elf3/mods/com.bxi.sonic/policy.py:1167-1186`
- Modify: `src/bxi_example_py_elf3/test/test_sonic_ordered_playout.py:178-225`

**Interfaces:**
- Consumes: one `applied_qpos` object representing 29 joint-position targets in `ELF3_POLICY_JOINTS` order.
- Produces: `SonicTeleopPolicy.record_applied_joint_target(self, applied_qpos: object) -> None`; updates only `last_action` in existing normalized/clipped model coordinates.

- [ ] **Step 1: Write the failing conversion and isolation test**

Add to `test_sonic_ordered_playout.py` after `_observation()`:

```python
def test_record_applied_joint_target_normalizes_clips_and_preserves_output(
    monkeypatch,
):
    policy = _make_policy(monkeypatch)
    output_before = policy.target_dof_pos.copy()
    normalized = np.linspace(-25.0, 25.0, NUM_JOINTS, dtype=np.float32)
    applied = policy.default_dof_pos + normalized * policy.action_scale

    policy.record_applied_joint_target(applied)

    np.testing.assert_allclose(
        policy.last_action,
        np.clip(normalized, -ACTION_CLIP, ACTION_CLIP),
        atol=1e-6,
    )
    np.testing.assert_array_equal(policy.target_dof_pos, output_before)
```

Import `ACTION_CLIP` from the dynamically loaded policy module beside the
existing constants:

```python
ACTION_CLIP = policy_module.ACTION_CLIP
```

- [ ] **Step 2: Write the failing shape and finiteness tests**

```python
@pytest.mark.parametrize(
    "value",
    [
        np.zeros(NUM_JOINTS - 1, dtype=np.float32),
        np.zeros(NUM_JOINTS + 1, dtype=np.float32),
        np.full(NUM_JOINTS, np.nan, dtype=np.float32),
        np.full(NUM_JOINTS, np.inf, dtype=np.float32),
    ],
)
def test_record_applied_joint_target_rejects_invalid_input(monkeypatch, value):
    policy = _make_policy(monkeypatch)
    before = policy.last_action.copy()

    with pytest.raises(ValueError, match="29 finite"):
        policy.record_applied_joint_target(value)

    np.testing.assert_array_equal(policy.last_action, before)
```

- [ ] **Step 3: Run the focused tests and verify RED**

From `src/bxi_example_py_elf3`:

```bash
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
python3 -m pytest -q \
  test/test_sonic_ordered_playout.py \
  -k 'record_applied_joint_target'
```

Expected: failures report that `SonicTeleopPolicy` has no
`record_applied_joint_target` method.

- [ ] **Step 4: Implement the minimal policy method**

Add immediately before `_update_history()` in `policy.py`:

```python
def record_applied_joint_target(self, applied_qpos: object) -> None:
    """Record the preceding applied qpos in model action coordinates."""
    qpos = np.asarray(applied_qpos)
    if qpos.shape != (NUM_JOINTS,) or not np.isfinite(qpos).all():
        raise ValueError(
            "applied joint target must contain exactly 29 finite values"
        )
    normalized = (
        qpos.astype(np.float32, copy=False) - self.default_dof_pos
    ) / self.action_scale
    if not np.isfinite(normalized).all():
        raise ValueError("normalized applied joint target must be finite")
    np.copyto(
        self.last_action,
        np.clip(normalized, -ACTION_CLIP, ACTION_CLIP),
        casting="same_kind",
    )
```

Do not update `target_dof_pos`, `action_history`, or any other history in this
method. `_update_history()` remains the only history shifter.

- [ ] **Step 5: Run focused and adjacent policy tests and verify GREEN**

```bash
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
python3 -m pytest -q \
  test/test_sonic_ordered_playout.py \
  test/test_sonic_reference_gate.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit the policy contract**

```bash
git add \
  src/bxi_example_py_elf3/mods/com.bxi.sonic/policy.py \
  src/bxi_example_py_elf3/test/test_sonic_ordered_playout.py
git commit -m "fix: record applied SONIC action target"
```

---

### Task 2: ZeroLab Previous-Applied-Frame Synchronization

**Files:**
- Modify: `src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/state.py:1-225,350-370`
- Modify: `src/bxi_example_py_elf3/test/test_zerolab_arming.py:20-320`

**Interfaces:**
- Consumes: Task 1 `SonicTeleopPolicy.record_applied_joint_target(applied_qpos)` and `ctx.last_motor_frame`.
- Produces: lifecycle-owned `CompiledJointMap` and `_record_previous_applied_target(frame: MotorFrame) -> None`, called before every advancing ZeroLab SONIC step.

- [ ] **Step 1: Extend the fake policy with an observable synchronization contract**

In `FakePolicy.__init__` add:

```python
self.applied_targets = []
self.call_order = []
```

Add to `FakePolicy`:

```python
def record_applied_joint_target(self, qpos):
    self.applied_targets.append(np.asarray(qpos, dtype=np.float32).copy())
    self.call_order.append("record")
```

At the beginning of the advancing branch in `FakePolicy.step()` add:

```python
self.call_order.append("step")
```

Clear both lists from `FakePolicy.reset()` so lifecycle tests remain isolated.

- [ ] **Step 2: Write the failing causal-order and non-advancing tests**

Add to `test_zerolab_arming.py`:

```python
def test_advancing_tick_records_previous_applied_target_before_sonic_step():
    state, sonic, _normal, ctx = prepared_state(entry_value=0.25)
    previous = ctx.last_motor_frame.qpos.copy()

    state.sample_running_frame(ctx, 0.02, advance=True)

    assert sonic.call_order[:2] == ["record", "step"]
    np.testing.assert_array_equal(sonic.applied_targets[-1], previous)


def test_non_advancing_tick_does_not_record_or_step():
    state, sonic, _normal, ctx = prepared_state(entry_value=0.25)
    state.sample_running_frame(ctx, 0.02, advance=True)
    recorded = len(sonic.applied_targets)
    steps = sonic.step_calls

    state.sample_running_frame(ctx, 0.02, advance=False)

    assert len(sonic.applied_targets) == recorded
    assert sonic.step_calls == steps
```

- [ ] **Step 3: Write the failing named-layout mapping test**

Import `JointLayout` beside `JointTargetBuffer`, then add:

```python
def test_applied_target_mapping_uses_joint_names_and_allows_extra_joint():
    state, sonic, _normal, ctx = prepared_state()
    names = tuple(reversed(ELF3_POLICY_JOINTS.names)) + ("extra_joint",)
    layout = JointLayout(names, label="reordered robot")
    values = np.asarray(
        [
            float(ELF3_POLICY_JOINTS.index(name))
            if name in ELF3_POLICY_JOINTS.names
            else 999.0
            for name in names
        ],
        dtype=np.float32,
    )
    frame = MotorFrame.create(
        layout,
        values,
        np.ones(layout.dof_num, dtype=np.float32),
        np.ones(layout.dof_num, dtype=np.float32),
    )

    state._prepare_applied_target_mapping(frame.layout)
    state._record_previous_applied_target(frame)

    np.testing.assert_array_equal(
        sonic.applied_targets[-1],
        np.arange(ELF3_POLICY_JOINTS.dof_num, dtype=np.float32),
    )
```

Add a missing-joint test:

```python
def test_applied_target_mapping_rejects_missing_policy_joint():
    state, _sonic, _normal, _ctx = prepared_state()
    incomplete = JointLayout(
        ELF3_POLICY_JOINTS.names[:-1], label="incomplete robot"
    )

    with pytest.raises(ValueError, match="missing joints"):
        state._prepare_applied_target_mapping(incomplete)
```

- [ ] **Step 4: Write the failing all-phase synchronization test**

```python
@pytest.mark.parametrize(
    "phase_name",
    [
        "wait_stream",
        "wait_arm",
        "blending",
        "armed",
        "hold_reference",
        "rearming",
    ],
)
def test_every_zerolab_phase_records_previous_applied_target(phase_name):
    state, sonic, _normal, ctx = state_in_phase(phase_name)
    marker = np.linspace(0.01, 0.29, 29, dtype=np.float32)
    ctx.last_motor_frame.qpos[:] = marker

    state.sample_running_frame(ctx, 0.001, advance=True)

    np.testing.assert_array_equal(sonic.applied_targets[-1], marker)
```

Add the two-cycle blend causal check:

```python
def test_blending_records_the_preceding_applied_blend():
    state, sonic, _normal, ctx = waiting_arm_state()
    assert state.on_action(ctx, "arm_zerolab") is True
    sonic.target.position.fill(2.0)
    previous_blend = copy_motor_frame(
        state.sample_running_frame(ctx, 1.0, advance=True)
    )
    ctx.last_motor_frame.qpos[:] = previous_blend.qpos

    state.sample_running_frame(ctx, 0.1, advance=True)

    np.testing.assert_allclose(
        sonic.applied_targets[-1], previous_blend.qpos
    )
```

- [ ] **Step 5: Run the new component tests and verify RED**

```bash
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
python3 -m pytest -q test/test_zerolab_arming.py \
  -k 'applied_target or advancing_tick or non_advancing_tick or preceding_applied_blend'
```

Expected: failures show that mapping/pre-step synchronization methods do not
exist or that `record` is absent before `step`.

- [ ] **Step 6: Implement lifecycle-owned named mapping**

In `zerolab/state.py` import:

```python
from bxi_example_py_elf3.framework.joints import CompiledJointMap, JointLayout
from bxi_example_py_elf3.policies.joints import ELF3_POLICY_JOINTS
```

In `__init__` add:

```python
self._applied_target_map = None
self._applied_policy_qpos = np.empty(
    ELF3_POLICY_JOINTS.dof_num, dtype=np.float32
)
```

Add these methods:

```python
def _prepare_applied_target_mapping(self, source_layout: JointLayout) -> None:
    self._applied_target_map = CompiledJointMap.compile(
        source_layout,
        ELF3_POLICY_JOINTS,
    )

def _record_previous_applied_target(self, frame: MotorFrame) -> None:
    mapping = self._applied_target_map
    if mapping is None:
        raise RuntimeError("ZeroLab applied-target mapping is not prepared")
    if frame.layout != mapping.source:
        raise ValueError("previous motor frame layout changed during ZeroLab")
    mapping.map_into(frame.qpos, self._applied_policy_qpos)
    self.policy.record_applied_joint_target(self._applied_policy_qpos)
```

At the end of `on_prepare()`, after all four frames have been created, add:

```python
self._prepare_applied_target_mapping(ctx.last_motor_frame.layout)
```

In `sample_running_frame()`, retain the existing early return for
`advance=False`, then insert immediately before the call to
`super().sample_running_frame(...)`:

```python
self._record_previous_applied_target(ctx.last_motor_frame)
```

This placement must be after the non-advancing return and before SONIC shifts
history. Do not add phase-specific synchronization branches.

In `on_exit()` add:

```python
self._applied_target_map = None
self._applied_policy_qpos.fill(0.0)
```

- [ ] **Step 7: Run focused state tests and verify GREEN**

```bash
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
python3 -m pytest -q \
  test/test_zerolab_arming.py \
  test/test_sonic_ordered_playout.py
```

Expected: all tests pass, including every existing ARM and recovery test.

- [ ] **Step 8: Commit the ZeroLab synchronization**

```bash
git add \
  src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/state.py \
  src/bxi_example_py_elf3/test/test_zerolab_arming.py
git commit -m "fix: synchronize ZeroLab applied action history"
```

---

### Task 3: Regression and Real-ONNX Evidence

**Files:**
- Verify: all Task 1 and Task 2 files.
- Evidence only: `/tmp/zerolab_real_onnx_ab.py`
- Evidence only: `/tmp/zerolab-applied-history-focused.log`
- Evidence only: `/tmp/zerolab-applied-history-onnx.log`

**Interfaces:**
- Consumes: the completed policy and ZeroLab state synchronization.
- Produces: deterministic test and real-model evidence that synchronized waiting history matches applied Normal targets instead of hidden SONIC targets.

- [ ] **Step 1: Run the focused deterministic regression with saved evidence**

From `src/bxi_example_py_elf3`:

```bash
set -o pipefail
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
python3 -m pytest -q \
  test/test_sonic_ordered_playout.py \
  test/test_zerolab_arming.py \
  test/test_sonic_reference_gate.py \
  test/test_zerolab_manifest.py 2>&1 | \
  tee /tmp/zerolab-applied-history-focused.log
```

Expected: all tests pass; no test is deselected by an accidental collection
failure.

- [ ] **Step 2: Extend the temporary real-ONNX diagnostic with the corrected path**

In `/tmp/zerolab_real_onnx_ab.py`, add this history branch beside
`executed_static` and `hidden_policy`:

```python
elif history_kind == "applied_sync":
    for _ in range(hidden_ticks):
        policy.record_applied_joint_target(q)
        policy.inference_step(q, dq, quat, omega)
    # inference_step() stores its new candidate in last_action; restore the
    # actually applied stationary target before the measured inference tick.
    policy.record_applied_joint_target(q)
```

Change the main case loop to include the synchronized history:

```python
for reference_kind in ("idle", "zerolab"):
    for history_kind in (
        "executed_static",
        "hidden_policy",
        "applied_sync",
    ):
        result = measure(
            reference_kind,
            history_kind,
            zero_ref,
            q,
            dq,
            quat,
            omega,
        )
        results.append(result)
```

Then compare its final target to `zerolab+executed_static` using the existing
`delta_metrics()` helper:

```python
"SYNC_EFFECT_ON_ZEROLAB": delta_metrics(
    by_name["zerolab+applied_sync"],
    by_name["zerolab+executed_static"],
),
```

The script remains under `/tmp`; do not add captured robot state or packet
captures to Git.

- [ ] **Step 3: Run the real model replay twice and verify determinism**

```bash
python3 /tmp/zerolab_real_onnx_ab.py \
  > /tmp/zerolab-applied-history-onnx.run1.log
python3 /tmp/zerolab_real_onnx_ab.py \
  > /tmp/zerolab-applied-history-onnx.run2.log
cmp \
  /tmp/zerolab-applied-history-onnx.run1.log \
  /tmp/zerolab-applied-history-onnx.run2.log
cp \
  /tmp/zerolab-applied-history-onnx.run1.log \
  /tmp/zerolab-applied-history-onnx.log
rg 'SYNC_EFFECT_ON_ZEROLAB|HISTORY_EFFECT_ON_ZEROLAB' \
  /tmp/zerolab-applied-history-onnx.log
sha256sum /tmp/zerolab-applied-history-onnx.log
```

Expected:

- `cmp` succeeds.
- `SYNC_EFFECT_ON_ZEROLAB` is numerically near zero within float32 inference
  tolerance (`max_abs_target_delta_rad <= 1e-5`).
- The old `HISTORY_EFFECT_ON_ZEROLAB` remains large, demonstrating that the
  diagnostic can still detect the original mismatch.

- [ ] **Step 4: Run the complete package suite**

```bash
set -o pipefail
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
python3 -m pytest -q test 2>&1 | \
  tee /tmp/zerolab-applied-history-pytest.log
```

Expected: all tests pass, including ZeroLab, ordinary SONIC, and PICO tests.

- [ ] **Step 5: Review the implementation diff before simulation**

From the worktree root:

```bash
git diff --check HEAD~2..HEAD
git diff --stat HEAD~2..HEAD
git status --short
```

Expected: the only tracked implementation changes are the two source files and
two test files in this plan. Existing untracked isolated build/install/log
directories remain untouched.

---

### Task 4: Isolated Build and MuJoCo Neutral-ARM Gate

**Files:**
- Verify only: all tracked Task 1 and Task 2 files.
- Runtime only: `build-applied-history/`, `install-applied-history/`, `log-applied-history/`.
- Evidence only: `/tmp/zerolab-applied-history-colcon.log`, `/tmp/zerolab-applied-history-mujoco.log`.

**Interfaces:**
- Consumes: green focused/full tests and real-ONNX replay from Task 3.
- Produces: a simulation-tested isolated candidate. It neither deploys to nor launches robot hardware.

- [ ] **Step 1: Build a new isolated candidate**

From the worktree root:

```bash
set -eo pipefail
set +u
source /opt/ros/humble/setup.bash
source /home/fazepurple/ros2_ws/bxi_ros2_pkg/setup.bash
colcon --log-base log-applied-history build \
  --merge-install \
  --base-paths src \
  --packages-select bxi_example_py_elf3 \
  --allow-overriding bxi_example_py_elf3 \
  --build-base build-applied-history \
  --install-base install-applied-history \
  --cmake-args -DCMAKE_BUILD_TYPE=Release 2>&1 | \
  tee /tmp/zerolab-applied-history-colcon.log
source install-applied-history/setup.bash
test "$(ros2 pkg prefix bxi_example_py_elf3)" = \
  "$PWD/install-applied-history"
```

Expected: build succeeds and the package prefix is the new isolated install.

- [ ] **Step 2: Prove no hardware process is running before MuJoCo**

```bash
if pgrep -af '[h]ardware_elf3|[r]os2 launch.*example_demo_hw'; then
  echo 'STOP: hardware process detected'
  exit 1
fi
echo 'NO_HARDWARE_PROCESS=PASS'
```

- [ ] **Step 3: Launch MuJoCo in Domain 42**

Terminal 1, from the worktree root:

```bash
set -eo pipefail
set +u
source /opt/ros/humble/setup.bash
source /home/fazepurple/ros2_ws/bxi_ros2_pkg/setup.bash
source "$PWD/install-applied-history/setup.bash"
export ROS_DOMAIN_ID=42
ros2 launch bxi_example_py_elf3 example_demo.launch.py 2>&1 | \
  tee /tmp/zerolab-applied-history-mujoco.log
```

- [ ] **Step 4: Exercise short and long neutral waits before ARM**

Terminal 2 sources the same environments and defines:

```bash
set +u
source /opt/ros/humble/setup.bash
source /home/fazepurple/ros2_ws/bxi_ros2_pkg/setup.bash
source "$PWD/install-applied-history/setup.bash"
export ROS_DOMAIN_ID=42

pulse_btn10() {
  ros2 topic pub --once /motion_commands \
    communication/msg/MotionCommands "{btn_10: $1}"
  ros2 topic pub --once /motion_commands \
    communication/msg/MotionCommands '{}'
}
```

Run two fresh ZeroLab lifecycles with the human reference neutral:

1. `pulse_btn10 11`, wait for `WAIT_ARM`, wait another `0.2 s`, then
   `pulse_btn10 12` once.
2. Exit to PD Brake, re-enter with `pulse_btn10 11`, wait for `WAIT_ARM`, wait
   another `5 s`, then `pulse_btn10 12` once.

For both trials, observe the full two-second `BLENDING -> ARMED` transition.
Expected: no fixed waist/leg yaw transient, no discontinuous motor target, no
fall, and no second ARM press.

- [ ] **Step 5: Verify phase and safety evidence**

```bash
grep -nE \
  'ZeroLab ARM phase|ARM accepted|reference status|state transition|safety|Traceback|exception|fatal' \
  /tmp/zerolab-applied-history-mujoco.log | tail -n 240
```

Expected for both trials:

```text
WAIT_STREAM -> WAIT_ARM -> BLENDING -> ARMED
```

No traceback, inference exception, fatal error, safety transition, or
unexpected phase reset is present.

- [ ] **Step 6: Recheck stale hold and automatic recovery without hardware**

After one successful neutral ARM, pause the ZeroLab sender for `0.51 s`, return
the reference to neutral, and restore it. Expected:

```text
ARMED -> HOLD_REFERENCE -> REARMING -> ARMED
```

No second `btn_10=12` is sent, and the reference-space two-second recovery
behavior remains unchanged.

- [ ] **Step 7: Stop simulation and record the candidate identity**

After stopping both MuJoCo terminals:

```bash
git diff --check
git status --short
git rev-parse HEAD
sha256sum \
  /tmp/zerolab-applied-history-pytest.log \
  /tmp/zerolab-applied-history-onnx.log \
  /tmp/zerolab-applied-history-colcon.log \
  /tmp/zerolab-applied-history-mujoco.log
```

Expected: all four evidence files exist, the tracked worktree is clean, and
only known isolated runtime directories are untracked. Hardware deployment
instructions are withheld until every MuJoCo condition above passes.
