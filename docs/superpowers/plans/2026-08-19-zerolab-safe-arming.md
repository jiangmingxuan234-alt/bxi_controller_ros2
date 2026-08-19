# ZeroLab Safe ARM Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the ELF3 on its captured Normal motor command while ZeroLab calibrates, then require a safety operator's `btn_10=12` event before a two-second smooth handover to the live SONIC policy.

**Architecture:** Add a ZeroLab-only `ZeroLabArmedTeleopState` subclass that advances the existing policy in the background while owning a full-layout held motor frame. The subclass implements an internal five-phase ARM session, smoothstep blending, stale hold, and explicit re-ARM; the existing `SonicTeleopState`, PICO data path, converter, policy, and hardware stack remain unchanged.

**Tech Stack:** Ubuntu 22.04, ROS 2 Humble, Python 3.10, NumPy, rclpy, pytest, YAML Mod manifests, colcon, MuJoCo replay tooling.

## Global Constraints

- Work only in `/home/fazepurple/ros2_ws/bxi_rl_controller_ros2_example_dev/.worktrees/konodoki-dev` until the isolated deployment task.
- Do not modify `/home/bxi/bxi_ws/bxi_rl_controller_ros2_example` or any of its tracked, modified, deleted, or untracked files.
- Do not modify `mods/com.bxi.sonic/state.py`, the SONIC policy, ZeroLab converter/protocol/source/recording/UDP code, PICO code, model assets, message definitions, base state machine, or hardware drivers.
- Preserve `btn_10=9` for PICO SONIC and `btn_10=11` for entering ZeroLab; add `btn_10=12` only as a `sonic_zerolab` action.
- Do not add automatic human shoulder/elbow pose checks, a ROS service, a new control topic, or a per-frame ACK.
- Use `arm_blend_seconds=2.0`, the existing `live_reference_timeout_s=0.5`, and smoothstep `p*p*(3-2*p)`.
- A stale live reference must hold the last applied full motor frame, cancel ARM, and require another `btn_10=12`; it must not automatically request Normal, PD Brake, or zero torque.
- Exiting and re-entering `sonic_zerolab` must start a new T-pose calibration. A short stale gap in the same state must retain the converter's completed calibration.
- Keep `btn_3=1` PD Brake, `btn_1=1` Normal, and `btn_2=1` zero torque available through existing routes.
- Do not launch a second `hardware_elf3` or `bxi_example_py_elf3_demo` process, and do not perform live testing while another person is using the robot.
- Follow TDD: add the named failing test, run it and observe the expected failure, implement only that behavior, rerun the focused test, then run the related regression set.

## File Structure

- Create `src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/state.py`: ZeroLab-only ARM phases, full-frame snapshots, smoothstep blending, stale hold, action handling, and lifecycle cleanup.
- Create `src/bxi_example_py_elf3/test/test_zerolab_arming.py`: fake policy/context tests for every ARM phase and all five `MotorFrame` fields.
- Modify `src/bxi_example_py_elf3/mods/com.bxi.sonic/plugin.py`: construct `ZeroLabArmedTeleopState` only for `sonic_zerolab`; leave `sonic_teleop` on `SonicTeleopState`.
- Modify `src/bxi_example_py_elf3/mods/com.bxi.sonic/mod.yaml`: declare `arm_zerolab`, bind it as a ZeroLab action, set `arm_blend_seconds: 2.0`, and update operator text.
- Modify `src/bxi_example_py_elf3/test/test_zerolab_manifest.py`: assert event isolation, factory class selection, parameters, and prompts.
- Modify `src/bxi_example_py_elf3/mods/com.bxi.sonic/README.md`: document the two-button calibration/ARM flow, stale recovery, emergency action, and exit behavior.

---

### Task 1: ZeroLab Waiting Gate and Two-Second ARM Blend

**Files:**
- Create: `src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/state.py`
- Create: `src/bxi_example_py_elf3/test/test_zerolab_arming.py`

**Interfaces:**
- Consumes: `SonicTeleopState`, `MotorFrame`, `RobotControlContext.last_motor_frame`, `RobotControlContext.resolve_motor_frame()`, and `SonicPolicy.has_fresh_live_reference(timeout_s)`.
- Produces: `ZeroLabArmPhase`, `ZeroLabArmedTeleopState(..., arm_blend_seconds: float = 2.0)`, `ZeroLabArmedTeleopState.arm_phase`, and the action name constant `ARM_ACTION = "arm_zerolab"`.

- [ ] **Step 1: Add a deterministic test module loader and fake runtime objects**

Create `test/test_zerolab_arming.py`. Load `com.bxi.sonic/state.py` and
`com.bxi.sonic/zerolab/state.py` under one synthetic package so their relative imports match the Mod loader:

```python
MOD_ROOT = Path(__file__).resolve().parents[1] / "mods" / "com.bxi.sonic"
PACKAGE = "_zerolab_arming_test_mod"


def _load_module(name: str, relative_path: str):
    if PACKAGE not in sys.modules:
        package = ModuleType(PACKAGE)
        package.__path__ = [str(MOD_ROOT)]
        package.__package__ = PACKAGE
        sys.modules[PACKAGE] = package
    if f"{PACKAGE}.zerolab" not in sys.modules:
        package = ModuleType(f"{PACKAGE}.zerolab")
        package.__path__ = [str(MOD_ROOT / "zerolab")]
        package.__package__ = f"{PACKAGE}.zerolab"
        sys.modules[f"{PACKAGE}.zerolab"] = package
    full_name = f"{PACKAGE}.{name}"
    spec = importlib.util.spec_from_file_location(full_name, MOD_ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


base_state_module = _load_module("state", "state.py")
arming_module = _load_module("zerolab.state", "zerolab/state.py")
ZeroLabArmPhase = arming_module.ZeroLabArmPhase
ZeroLabArmedTeleopState = arming_module.ZeroLabArmedTeleopState
```

Add `FakePolicy`, `FakeHandle`, `FakeContext`, and `CaptureLogger` with these fixed behaviors:

```python
class FakePolicy:
    def __init__(self):
        target = JointTargetBuffer(ELF3_POLICY_JOINTS)
        target.kp.fill(40.0)
        target.kd.fill(2.0)
        self.output = PolicyOutput(target.view)
        self.target = target
        self.fresh = False
        self.step_calls = 0
        self.head_joint_target = np.zeros(2, dtype=np.float32)
        self.last_status = "idle_reference"

    def bind_logger(self, _logger):
        pass

    def configure_runtime(self, **_kwargs):
        pass

    def reset(self, _frame=None):
        self.step_calls = 0

    def step(self, _frame, _dt, *, advance=True):
        if advance:
            self.step_calls += 1
        return self.output

    def has_fresh_live_reference(self, _timeout_s=None):
        return self.fresh

    def reset_yaw_alignment(self):
        pass


class FakeHandle:
    status = "ready"

    def __init__(self, policy):
        self.policy = policy

    def get(self):
        return self.policy


class FakeContext:
    def __init__(self, last_motor_frame):
        self.robot_layout = ELF3_POLICY_JOINTS
        self.last_motor_frame = last_motor_frame
        self.inference_frame = object()
        self.applied = None

    def resolve_motor_frame(self, frame, output):
        assert frame.layout == self.robot_layout
        return output.update(
            frame.qpos, frame.kp, frame.kd,
            vel=frame.vel, torque=frame.torque,
        )

    def set_motor_target(self, frame):
        self.applied = MotorFrame.create(
            frame.layout, frame.qpos, frame.kp, frame.kd,
            vel=frame.vel, torque=frame.torque,
        )
```

The shared `make_state()` helper must instantiate the real constructor with
`head_control_enabled=False`, `hardware_gripper=False`,
`live_reference_timeout_s=0.5`, and a caller-selected `arm_blend_seconds`.
Its entry frame uses `qpos=entry_value`, `kp=40.0`, `kd=2.0`, `vel=0.4`, and
`torque=0.8` for every joint, so the midpoint assertions for all five fields
are numeric and deterministic.

- [ ] **Step 2: Write failing waiting and validation tests**

Add these tests before the source file exists:

```python
def test_arm_blend_seconds_must_be_finite_and_positive():
    for value in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="arm_blend_seconds"):
            make_state(arm_blend_seconds=value)


def test_prepare_deep_copies_entry_frame_and_waiting_advances_policy():
    state, policy, ctx = prepared_state(entry_value=0.25)
    ctx.last_motor_frame.qpos.fill(9.0)
    policy.target.position.fill(1.0)

    frame = state.sample_running_frame(ctx, 0.02, advance=True)

    assert state.arm_phase is ZeroLabArmPhase.WAIT_CALIBRATION
    assert policy.step_calls == 1
    np.testing.assert_allclose(frame.qpos, 0.25)
    np.testing.assert_allclose(state.get_entry_frame(ctx).qpos, 0.25)


def test_fresh_reference_waits_for_explicit_arm():
    state, policy, ctx = prepared_state(entry_value=0.25)
    policy.fresh = True
    policy.target.position.fill(1.0)

    frame = state.sample_running_frame(ctx, 0.02, advance=True)

    assert state.arm_phase is ZeroLabArmPhase.WAIT_ARM
    np.testing.assert_allclose(frame.qpos, 0.25)
```

- [ ] **Step 3: Run the new module and verify the import failure**

Run:

```bash
cd /home/fazepurple/ros2_ws/bxi_rl_controller_ros2_example_dev/.worktrees/konodoki-dev/src/bxi_example_py_elf3
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m pytest -q test/test_zerolab_arming.py
```

Expected: collection fails because `zerolab/state.py` and
`ZeroLabArmedTeleopState` do not exist.

- [ ] **Step 4: Implement phases, validation, full-frame buffers, and waiting output**

Create `zerolab/state.py` with these public declarations:

```python
from __future__ import annotations

from enum import Enum
import math
from typing import TYPE_CHECKING

import numpy as np

from bxi_example_py_elf3.framework.mod_api.transition import MotorFrame

from ..state import SonicTeleopState

if TYPE_CHECKING:
    from bxi_example_py_elf3.framework.mod_api import RobotControlContext, StateBehavior


ARM_ACTION = "arm_zerolab"


class ZeroLabArmPhase(str, Enum):
    WAIT_CALIBRATION = "wait_calibration"
    WAIT_ARM = "wait_arm"
    BLENDING = "blending"
    ARMED = "armed"
    HOLD_STALE = "hold_stale"


class ZeroLabArmedTeleopState(SonicTeleopState):
    def __init__(self, name, state_id, policy, *, arm_blend_seconds=2.0, **kwargs):
        super().__init__(name, state_id, policy, **kwargs)
        value = float(arm_blend_seconds)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("arm_blend_seconds must be finite and positive")
        self.arm_blend_seconds = value
        self._arm_phase = ZeroLabArmPhase.WAIT_CALIBRATION
        self._hold_frame = None
        self._applied_frame = None
        self._live_frame = None
        self._blend_start_frame = None
        self._blend_elapsed_s = 0.0

    @property
    def arm_phase(self) -> ZeroLabArmPhase:
        return self._arm_phase
```

Implement one private `_copy_frame(target, source)` method using
`MotorFrame.update()` for all five fields. In `on_prepare`, call
`super().on_prepare(...)`, allocate four `MotorFrame.empty(ctx.robot_layout)`
buffers, and deep-copy `ctx.last_motor_frame` into both `_hold_frame` and
`_applied_frame`.

Override `get_entry_frame()` to return `_hold_frame`. Override
`sample_running_frame()` so `advance=False` returns `_applied_frame` without
calling the policy. For `advance=True`, call
`super().sample_running_frame(..., advance=True)`, resolve the natural policy
frame into `_live_frame`, advance `WAIT_CALIBRATION -> WAIT_ARM` only when
`has_fresh_live_reference(0.5)` is true, and otherwise copy `_hold_frame` into
`_applied_frame`.

- [ ] **Step 5: Run waiting tests and verify they pass**

Run the Task 1 pytest command again.

Expected: validation, deep-copy, background advancement, and explicit waiting
tests pass. ARM blend tests have not been added yet.

- [ ] **Step 6: Write failing explicit ARM and smoothstep tests**

Add:

```python
def test_arm_is_rejected_until_fresh_reference_exists():
    state, policy, ctx = prepared_state(entry_value=0.0)
    assert state.on_action(ctx, "arm_zerolab") is True
    assert state.arm_phase is ZeroLabArmPhase.WAIT_CALIBRATION


def test_arm_blends_all_motor_fields_for_exactly_two_seconds():
    state, policy, ctx = prepared_state(entry_value=0.0)
    policy.fresh = True
    state.sample_running_frame(ctx, 0.02, advance=True)
    assert state.on_action(ctx, "arm_zerolab") is True

    policy.target.position.fill(2.0)
    policy.target.kp.fill(80.0)
    policy.target.kd.fill(4.0)
    first = state.sample_running_frame(ctx, 0.0, advance=True)
    middle = state.sample_running_frame(ctx, 1.0, advance=True)
    final = state.sample_running_frame(ctx, 1.0, advance=True)

    np.testing.assert_allclose(first.qpos, 0.0)
    np.testing.assert_allclose(middle.qpos, 1.0)
    np.testing.assert_allclose(final.qpos, 2.0)
    np.testing.assert_allclose(middle.kp, 60.0)
    np.testing.assert_allclose(middle.kd, 3.0)
    assert state.arm_phase is ZeroLabArmPhase.ARMED


def test_duplicate_arm_does_not_restart_blend():
    state, policy, ctx = armed_state_at_half_blend()
    before = state._blend_elapsed_s
    assert state.on_action(ctx, "arm_zerolab") is True
    assert state._blend_elapsed_s == before
    assert state.arm_phase is ZeroLabArmPhase.BLENDING
```

Construct the entry test frame with nonzero `vel` and `torque`, and add matching
assertions `middle.vel == 0.2` and `middle.torque == 0.4`, proving all five
fields use the same alpha while the natural policy target supplies zero
velocity and feed-forward torque.

- [ ] **Step 7: Run the focused tests and observe blend failures**

Run:

```bash
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m pytest -q test/test_zerolab_arming.py \
  -k 'arm or blend'
```

Expected: failures show that `arm_zerolab` does not yet start `BLENDING` and
that the output remains the hold frame.

- [ ] **Step 8: Implement ARM action and smoothstep blending**

Add `_smoothstep(progress)`, `_begin_blend()`, and the `BLENDING`/`ARMED`
branches:

```python
@staticmethod
def _smoothstep(progress: float) -> float:
    p = min(max(float(progress), 0.0), 1.0)
    return p * p * (3.0 - 2.0 * p)


def _begin_blend(self) -> None:
    self._copy_frame(self._blend_start_frame, self._applied_frame)
    self._blend_elapsed_s = 0.0
    self._arm_phase = ZeroLabArmPhase.BLENDING
```

`on_action()` must delegate unknown actions to `super().on_action()`. For
`ARM_ACTION`, it must always return `True`; accept only `WAIT_ARM` or
`HOLD_STALE` with a fresh reference, reject `WAIT_CALIBRATION` or stale data,
and ignore duplicates in `BLENDING`/`ARMED`.

During `BLENDING`, increment elapsed by `max(dt, 0.0)`, calculate alpha, and
write all five arrays using:

```python
np.subtract(target, start, out=output)
output *= alpha
output += start
```

At elapsed `>= arm_blend_seconds`, set phase to `ARMED`. In `ARMED`, deep-copy
the resolved live frame into `_applied_frame` and return it. Do not add a
persistent post-ARM filter.

Export exactly:

```python
__all__ = ["ARM_ACTION", "ZeroLabArmPhase", "ZeroLabArmedTeleopState"]
```

- [ ] **Step 9: Run Task 1 tests and the existing ordered-playout regression**

Run:

```bash
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m pytest -q \
  test/test_zerolab_arming.py \
  test/test_sonic_ordered_playout.py
```

Expected: all tests pass, and the ordered SONIC policy tests remain unchanged.

- [ ] **Step 10: Commit the waiting and blend implementation**

```bash
git add \
  src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/state.py \
  src/bxi_example_py_elf3/test/test_zerolab_arming.py
git commit -m "feat(sonic): gate ZeroLab takeover behind explicit arm"
```

---

### Task 2: Stale Hold, Re-ARM, Lifecycle Cleanup, and Logs

**Files:**
- Modify: `src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/state.py`
- Modify: `src/bxi_example_py_elf3/test/test_zerolab_arming.py`

**Interfaces:**
- Consumes: Task 1's `ZeroLabArmPhase`, `_applied_frame`, `_hold_frame`, `_begin_blend()`, and `ARM_ACTION`.
- Produces: stale transition to `HOLD_STALE`, fresh-reference recovery without automatic output, re-ARM from the frozen frame, and phase-transition logs.

- [ ] **Step 1: Write failing stale and recovery tests**

Add these tests:

```python
@pytest.mark.parametrize("starting_phase", [
    ZeroLabArmPhase.BLENDING,
    ZeroLabArmPhase.ARMED,
])
def test_stale_reference_freezes_last_applied_frame_and_cancels_arm(starting_phase):
    state, policy, ctx = state_in_active_phase(starting_phase)
    before = copy_motor_frame(state.sample_running_frame(ctx, 0.02, advance=True))
    policy.fresh = False
    policy.target.position.fill(9.0)

    after = state.sample_running_frame(ctx, 0.02, advance=True)

    assert state.arm_phase is ZeroLabArmPhase.HOLD_STALE
    assert_motor_frames_equal(after, before)


def test_recovery_does_not_resume_until_rearmed():
    state, policy, ctx = stale_hold_state()
    frozen = copy_motor_frame(state.sample_running_frame(ctx, 0.02, advance=True))
    policy.fresh = True
    policy.target.position.fill(2.0)

    waiting = state.sample_running_frame(ctx, 0.02, advance=True)
    assert state.arm_phase is ZeroLabArmPhase.HOLD_STALE
    assert_motor_frames_equal(waiting, frozen)

    state.on_action(ctx, "arm_zerolab")
    resumed = state.sample_running_frame(ctx, 0.0, advance=True)
    assert state.arm_phase is ZeroLabArmPhase.BLENDING
    assert_motor_frames_equal(resumed, frozen)


def test_wait_arm_gap_never_changes_held_normal_frame():
    state, policy, ctx = waiting_arm_state()
    policy.fresh = False
    frame = state.sample_running_frame(ctx, 1.0, advance=True)
    assert state.arm_phase is ZeroLabArmPhase.WAIT_ARM
    assert_motor_frames_equal(frame, state.get_entry_frame(ctx))


def test_exit_clears_arm_session_and_next_prepare_requires_calibration():
    state, policy, ctx = fully_armed_state()
    state.on_exit(ctx)
    policy.fresh = False
    state.on_prepare(ctx, SimpleNamespace())
    assert state.arm_phase is ZeroLabArmPhase.WAIT_CALIBRATION
```

- [ ] **Step 2: Run stale tests and verify they fail**

Run:

```bash
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m pytest -q test/test_zerolab_arming.py \
  -k 'stale or recovery or gap or exit'
```

Expected: active phases continue returning live output or retain `ARMED`, proving
the stale gate is not implemented.

- [ ] **Step 3: Implement stale freeze and lifecycle reset**

After resolving the live target but before applying `BLENDING` or `ARMED`, test
freshness. If false, copy `_applied_frame` into `_hold_frame`, set
`HOLD_STALE`, reset `_blend_elapsed_s`, and return the unchanged
`_applied_frame`.

In `HOLD_STALE`, keep advancing the policy but return the frozen hold frame even
after freshness returns. Only `_begin_blend()` from `on_action()` can leave
this phase.

Override `on_exit()` to call `super().on_exit(ctx)`, reset the phase to
`WAIT_CALIBRATION`, set all four frame buffers to `None`, and reset blend
elapsed time. Do not call converter or source internals; the existing
state-scoped node lifecycle owns calibration destruction on state exit.

- [ ] **Step 4: Add phase-transition and refusal logs without control-loop spam**

Add `_set_phase(phase, message, *, warning=False)` that logs only when the
phase actually changes. Add one boolean recovery notice latch so
`HOLD_STALE -> fresh-but-waiting` produces exactly one recovery message while
remaining in `HOLD_STALE`.

The observable messages must contain these stable substrings:

```text
ZeroLab ARM phase: WAIT_CALIBRATION
ZeroLab ARM phase: WAIT_ARM
ZeroLab ARM accepted; blending for 2.000 s
ZeroLab ARM phase: ARMED
ZeroLab reference stale; holding last motor frame and ARM cancelled
ZeroLab reference recovered; send btn_10=12 to resume
```

Add logger assertions proving stale and recovery messages occur once even
after 100 update cycles. Refused and duplicate button presses may log once per
button event because they are not generated by the 50 Hz loop.

- [ ] **Step 5: Run all ARM tests**

Run:

```bash
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m pytest -q test/test_zerolab_arming.py
```

Expected: all waiting, blend, stale, re-ARM, cleanup, and log-count tests pass.

- [ ] **Step 6: Commit stale safety behavior**

```bash
git add \
  src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/state.py \
  src/bxi_example_py_elf3/test/test_zerolab_arming.py
git commit -m "feat(sonic): hold and rearm ZeroLab after stale input"
```

---

### Task 3: Isolated Plugin and Manifest Wiring

**Files:**
- Modify: `src/bxi_example_py_elf3/mods/com.bxi.sonic/plugin.py`
- Modify: `src/bxi_example_py_elf3/mods/com.bxi.sonic/mod.yaml`
- Modify: `src/bxi_example_py_elf3/test/test_zerolab_manifest.py`

**Interfaces:**
- Consumes: `ZeroLabArmedTeleopState` and `ARM_ACTION` from Tasks 1-2.
- Produces: `btn_10=12 -> arm_zerolab` only inside `sonic_zerolab`, a 2.0-second manifest parameter, and unchanged PICO factory selection.

- [ ] **Step 1: Write failing manifest isolation assertions**

Extend `test_zerolab_event_state_and_routes_are_safe()`:

```python
assert manifest["events"]["arm_zerolab"] == {
    "slot": "btn_10",
    "value": 12,
}
params = manifest["states"]["sonic_zerolab"]["params"]
assert params["arm_blend_seconds"] == 2.0

actions = {
    (item["from"], item["event"], item["action"])
    for item in manifest["actions"]
}
assert ("sonic_zerolab", "arm_zerolab", "arm_zerolab") in actions
assert not any(
    source == "sonic_teleop" and event == "arm_zerolab"
    for source, event, _action in actions
)
```

Extend the dynamic factory test to assert:

```python
assert type(pico).__name__ == "SonicTeleopState"
assert type(zero).__name__ == "ZeroLabArmedTeleopState"
assert zero.arm_blend_seconds == 2.0
assert zero._policy is pico._policy
```

- [ ] **Step 2: Run manifest tests and verify missing event/parameter/class failures**

Run:

```bash
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m pytest -q test/test_zerolab_manifest.py
```

Expected: failures identify absent `arm_zerolab`, absent
`arm_blend_seconds`, and the old `SonicTeleopState` ZeroLab factory.

- [ ] **Step 3: Refactor plugin construction without changing PICO arguments**

Import `ZeroLabArmedTeleopState` from `.zerolab.state`. Extract the existing
parameter reads verbatim into `_common_state_kwargs(state) -> dict[str, object]`.
Do not rename, remove, or change defaults for any existing parameter.

Keep:

```python
def _build_state(state, policy) -> SonicTeleopState:
    return SonicTeleopState(
        state.name,
        state.state_id,
        policy,
        **_common_state_kwargs(state),
    )
```

Add:

```python
def _build_zerolab_state(state, policy) -> ZeroLabArmedTeleopState:
    return ZeroLabArmedTeleopState(
        state.name,
        state.state_id,
        policy,
        arm_blend_seconds=state.float_param("arm_blend_seconds", 2.0),
        **_common_state_kwargs(state),
    )
```

Register `sonic_teleop` with `_build_state` and `sonic_zerolab` with
`_build_zerolab_state`. Both must continue sharing the single startup
`SONIC_POLICY` handle.

- [ ] **Step 4: Add manifest event, action, parameter, and operator text**

Add `arm_zerolab` under `events`. Under `sonic_zerolab.params`, set:

```yaml
operator_prompt: >-
  T-pose标定期间机器人保持Normal；stream ready后回到中立姿势，等待安全员发送btn_10=12
arm_blend_seconds: 2.0
```

Append the `sonic_zerolab` ARM action without altering existing routes or the
`sonic_teleop/reset_alignment` action.

- [ ] **Step 5: Update exact prompt and logger expectations in manifest tests**

The ZeroLab factory test currently expects one generic `on_enter` message.
Update it to assert the generic message plus the new
`WAIT_CALIBRATION` phase message, while the PICO prompt remains exactly:

```text
PICO同时按住A+B+X+Y请求校准，再按A+X切入实时POSE
```

- [ ] **Step 6: Run manifest, ARM, lifecycle, and PICO policy regressions**

Run:

```bash
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m pytest -q \
  test/test_zerolab_manifest.py \
  test/test_zerolab_arming.py \
  test/test_zerolab_lifecycle.py \
  test/test_sonic_ordered_playout.py
```

Expected: all pass. In particular, PICO uses `SonicTeleopState`, ZeroLab uses
`ZeroLabArmedTeleopState`, and source/bridge lifecycle tests still release
18000/5558/5557.

- [ ] **Step 7: Commit plugin and manifest wiring**

```bash
git add \
  src/bxi_example_py_elf3/mods/com.bxi.sonic/plugin.py \
  src/bxi_example_py_elf3/mods/com.bxi.sonic/mod.yaml \
  src/bxi_example_py_elf3/test/test_zerolab_manifest.py
git commit -m "feat(sonic): wire ZeroLab explicit arm event"
```

---

### Task 4: Operator Documentation and Full Local Verification

**Files:**
- Modify: `src/bxi_example_py_elf3/mods/com.bxi.sonic/README.md`
- Verify: all files changed in Tasks 1-3

**Interfaces:**
- Consumes: the installed `btn_10=11` entry event, `btn_10=12` ARM action, `btn_3=1` PD Brake event, and ARM logs.
- Produces: copyable two-terminal instructions and local evidence for tests, build, installed manifest, and unchanged MuJoCo replay metrics.

- [ ] **Step 1: Replace the old immediate-takeover instructions**

In `README.md`'s `## ZeroLab 实时 MuJoCo 遥操` section, document this exact
sequence:

1. Enter PD Brake, then Normal, and wait for stable idle standing.
2. Send `btn_10=11`; the robot holds the captured Normal command.
3. Hold T-pose until source/bridge readiness and `WAIT_ARM` are logged.
4. Return to the operator's neutral pose; there is no automatic human-pose gate.
5. The safety operator prepares the PD Brake command, then sends `btn_10=12`.
6. Keep the neutral pose during the two-second handover.
7. On stale input, the robot holds the last command and requires another
   `btn_10=12`; it does not automatically enter PD Brake or resume.
8. Leaving and re-entering ZeroLab requires a new T-pose.

Include the exact ARM command:

```bash
ros2 topic pub --once /motion_commands \
  communication/msg/MotionCommands '{btn_10: 12}'
ros2 topic pub --once /motion_commands \
  communication/msg/MotionCommands '{}'
```

Keep the existing PD Brake command adjacent to the ARM instructions. State
that the two-second blend reduces a jump but cannot correct a wrong target.

- [ ] **Step 2: Run all ZeroLab tests including the new ARM module**

From `src/bxi_example_py_elf3`:

```bash
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m pytest -q \
  test/test_zerolab_protocol.py \
  test/test_zerolab_udp_receiver.py \
  test/test_zerolab_converter.py \
  test/test_zerolab_recording.py \
  test/test_zerolab_pose_contract.py \
  test/test_zerolab_lifecycle.py \
  test/test_zerolab_manifest.py \
  test/test_zerolab_arming.py
```

Expected: the original 102 tests still pass and the new ARM tests add only
passing cases.

- [ ] **Step 3: Run the complete package test directory**

```bash
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m pytest -q test
```

Expected: all tests pass with no collection errors.

- [ ] **Step 4: Build into separate local artifacts**

From the worktree root:

```bash
source /opt/ros/humble/setup.bash
source /opt/bxi/bxi_ros2_pkg/setup.bash
colcon --log-base log-safe-arm build \
  --merge-install \
  --symlink-install \
  --base-paths src \
  --packages-select bxi_example_py_elf3 \
  --build-base build-safe-arm \
  --install-base install-safe-arm \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
```

Expected: `Summary: 1 package finished`.

- [ ] **Step 5: Verify the installed class and manifest**

```bash
source install-safe-arm/setup.bash
test "$(ros2 pkg prefix bxi_example_py_elf3)" = "$PWD/install-safe-arm"
grep -nE 'arm_zerolab|value: 12|arm_blend_seconds: 2.0' \
  install-safe-arm/share/bxi_example_py_elf3/mods/com.bxi.sonic/mod.yaml
test -f \
  install-safe-arm/share/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/state.py
```

Expected: prefix equals the isolated install, the three ARM manifest entries
are present, and the new state file is installed.

- [ ] **Step 6: Replay the known capture headlessly through SONIC/MuJoCo**

Run from any directory. The output directory is new and does not replace the
previous evidence:

```bash
OFFLINE_ROOT=/home/fazepurple/ros2_ws/bxi_rl_controller_ros2_example_dev/.worktrees/zerolab-offline-evaluation
SAFE_ARM_ROOT=/home/fazepurple/ros2_ws/bxi_rl_controller_ros2_example_dev/.worktrees/konodoki-dev
REPLAY_OUT=$(mktemp -d /tmp/zerolab-safe-arm-replay-XXXXXX)
PYTHONPATH="$OFFLINE_ROOT/src/bxi_example_py_elf3/mods/com.bxi.sonic:$OFFLINE_ROOT/src/bxi_example_py_elf3${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m zerolab.mujoco_replay \
  --canonical /tmp/zerolab-arm-direction-20260819-002-canonical.npz \
  --mod-root "$SAFE_ARM_ROOT/src/bxi_example_py_elf3/mods/com.bxi.sonic" \
  --xml "$SAFE_ARM_ROOT/src/bxi_example_py_elf3/data/mujoco_simulation/elf3.xml" \
  --output "$REPLAY_OUT"
python3 -m json.tool "$REPLAY_OUT/report.json"
```

Expected report invariants:

```text
reference.source = live
reference.device = zerolab
metrics.samples = 718
metrics.duration_s = 14.34
metrics.limits.samples_outside = 0
metrics.stability.fell = false
metrics.tracking.rmse_rad = 0.05898461942317773
```

The ARM state wraps motor-frame publication and must not change canonical
conversion or policy/MuJoCo replay metrics.

- [ ] **Step 7: Review the final source diff for forbidden files**

```bash
git diff --name-only 46e3724..HEAD -- src/bxi_example_py_elf3
git diff --check 46e3724..HEAD -- src/bxi_example_py_elf3
```

Expected changed runtime/test files are limited to the six paths listed in
the File Structure section. No PICO, converter, policy,
asset, hardware, or base-framework path may appear.

- [ ] **Step 8: Commit operator documentation**

```bash
git add src/bxi_example_py_elf3/mods/com.bxi.sonic/README.md
git commit -m "docs(sonic): document ZeroLab safe arming flow"
```

---

### Task 5: Deploy Only to the Robot Candidate and Rebuild Separately

**Files:**
- Read only: local implementation commits after `46e3724`
- Modify only: `/home/bxi/zerolab-sim2real-candidate-20260818`
- Prohibited: any write under `/home/bxi/bxi_ws/bxi_rl_controller_ros2_example`

**Interfaces:**
- Consumes: verified local source diff and SSH route `bxi@192.168.88.172`.
- Produces: a separately built robot install at `/home/bxi/zerolab-sim2real-candidate-20260818/install-safe-arm` and before/after proof that the original worktree status is identical.

- [ ] **Step 1: Create and inspect a source-only binary patch locally**

```bash
LOCAL_ROOT=/home/fazepurple/ros2_ws/bxi_rl_controller_ros2_example_dev/.worktrees/konodoki-dev
PATCH_LOCAL=$(mktemp /tmp/zerolab-safe-arm-XXXXXX.patch)
git -C "$LOCAL_ROOT" diff --binary 46e3724..HEAD \
  -- src/bxi_example_py_elf3 > "$PATCH_LOCAL"
git -C "$LOCAL_ROOT" diff --name-only 46e3724..HEAD \
  -- src/bxi_example_py_elf3
sha256sum "$PATCH_LOCAL"
```

Expected list: only `zerolab/state.py`, `plugin.py`, `mod.yaml`, `README.md`,
`test_zerolab_arming.py`, and `test_zerolab_manifest.py`.

- [ ] **Step 2: Transfer by the verified direct robot address**

```bash
PATCH_REMOTE=/home/bxi/zerolab-safe-arm-46e3724-to-head.patch
ssh bxi@192.168.88.172 "test ! -e $PATCH_REMOTE"
scp "$PATCH_LOCAL" "bxi@192.168.88.172:$PATCH_REMOTE"
ssh bxi@192.168.88.172 "sha256sum $PATCH_REMOTE"
```

Expected: local and remote SHA-256 values match. If the remote target already
exists, stop and inspect it; do not overwrite or delete it as part of this
plan.

- [ ] **Step 3: Snapshot the original robot worktree before candidate changes**

On the robot as `bxi`:

```bash
set -euo pipefail
BASE_WS=/home/bxi/bxi_ws/bxi_rl_controller_ros2_example
CANDIDATE_WS=/home/bxi/zerolab-sim2real-candidate-20260818
PATCH_REMOTE=/home/bxi/zerolab-safe-arm-46e3724-to-head.patch
git -C "$BASE_WS" rev-parse --is-inside-work-tree
test -d "$CANDIDATE_WS/src/bxi_example_py_elf3"
test -f "$PATCH_REMOTE"
git -C "$BASE_WS" status --short --branch > \
  "$CANDIDATE_WS/original-worktree-status-safe-arm-before.txt"
```

- [ ] **Step 4: Check and apply only to the candidate source**

```bash
cd /home/bxi/zerolab-sim2real-candidate-20260818/src
git apply --no-index -p2 --check "$PATCH_REMOTE"
git apply --no-index -p2 "$PATCH_REMOTE"
grep -nE 'arm_zerolab|value: 12|arm_blend_seconds: 2.0' \
  bxi_example_py_elf3/mods/com.bxi.sonic/mod.yaml
```

Expected: check and apply succeed once. If check fails, stop and inspect; do
not use `--reject`, `--3way`, force, reset, checkout, or deletion commands.

- [ ] **Step 5: Run candidate source tests before building**

```bash
cd /home/bxi/zerolab-sim2real-candidate-20260818
source /opt/ros/humble/setup.bash
source /opt/bxi/bxi_ros2_pkg/setup.bash
source /home/bxi/bxi_ws/bxi_rl_controller_ros2_example/install/setup.bash
PYTHONPATH="$PWD/src/bxi_example_py_elf3:$PWD/src/bxi_example_py_elf3/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m pytest -q \
  src/bxi_example_py_elf3/test/test_zerolab_arming.py \
  src/bxi_example_py_elf3/test/test_zerolab_manifest.py \
  src/bxi_example_py_elf3/test/test_zerolab_lifecycle.py \
  src/bxi_example_py_elf3/test/test_sonic_ordered_playout.py
```

Expected: all selected tests pass.

- [ ] **Step 6: Build to new candidate artifact directories**

```bash
cd /home/bxi/zerolab-sim2real-candidate-20260818
colcon --log-base log-safe-arm build \
  --merge-install \
  --base-paths src \
  --packages-select bxi_example_py_elf3 \
  --allow-overriding bxi_example_py_elf3 \
  --build-base build-safe-arm \
  --install-base install-safe-arm \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
```

Expected: `Summary: 1 package finished`. This must not overwrite the existing
candidate `build`, `install`, or `log` directories.

- [ ] **Step 7: Verify the separate install and original-worktree proof**

```bash
source /home/bxi/zerolab-sim2real-candidate-20260818/install-safe-arm/setup.bash
test "$(ros2 pkg prefix bxi_example_py_elf3)" = \
  /home/bxi/zerolab-sim2real-candidate-20260818/install-safe-arm
grep -nE 'arm_zerolab|value: 12|arm_blend_seconds: 2.0' \
  /home/bxi/zerolab-sim2real-candidate-20260818/install-safe-arm/share/bxi_example_py_elf3/mods/com.bxi.sonic/mod.yaml
git -C "$BASE_WS" status --short --branch > \
  "$CANDIDATE_WS/original-worktree-status-safe-arm-after.txt"
cmp \
  "$CANDIDATE_WS/original-worktree-status-safe-arm-before.txt" \
  "$CANDIDATE_WS/original-worktree-status-safe-arm-after.txt"
echo 'ORIGINAL_WORKTREE_UNCHANGED=PASS'
```

Expected: candidate prefix, installed ARM entries, and unchanged original
worktree all pass.

---

### Task 6: Root-Only Hardware Acceptance with a Separate Safety Terminal

**Files:**
- Read only: `/home/bxi/zerolab-sim2real-candidate-20260818/install-safe-arm`
- Runtime output only: ROS logs and terminal output

**Interfaces:**
- Consumes: the verified candidate install from Task 5 and live ZeroLab UDP from `192.168.89.171`.
- Produces: observed evidence for wait-before-ARM, two-second takeover, stale hold, explicit re-ARM, emergency routing, and clean exit.

- [ ] **Step 1: Confirm exclusive access and stop if any old controller exists**

On the robot as `bxi`:

```bash
ps -eo user:12,pid,etimes,args | \
  grep -E '[h]ardware_elf3|[b]xi_example_py_elf3_demo|[r]os2 launch.*example_demo_hw' \
  || echo 'NO_EXISTING_CONTROLLER=PASS'
```

If any process is listed, do not launch another controller. Identify its owner
and wait until the current operator shuts it down normally.

- [ ] **Step 2: Start the only hardware stack in root terminal 1**

```bash
sudo -i
source /opt/ros/humble/setup.bash
source /opt/bxi/bxi_ros2_pkg/setup.bash
source /home/bxi/bxi_ws/bxi_rl_controller_ros2_example/install/setup.bash
source /home/bxi/zerolab-sim2real-candidate-20260818/install-safe-arm/setup.bash
export ROS_DOMAIN_ID=42
test "$(ros2 pkg prefix bxi_example_py_elf3)" = \
  /home/bxi/zerolab-sim2real-candidate-20260818/install-safe-arm
cd /home/bxi/zerolab-sim2real-candidate-20260818
ros2 launch bxi_example_py_elf3 example_demo_hw.launch.py
```

Expected: exactly one `hardware_elf3` and one candidate
`bxi_example_py_elf3_demo` stay running. Keep this terminal visible.

- [ ] **Step 3: Prepare root terminal 2 and the emergency command**

In a second terminal:

```bash
sudo -i
source /opt/ros/humble/setup.bash
source /opt/bxi/bxi_ros2_pkg/setup.bash
source /home/bxi/bxi_ws/bxi_rl_controller_ros2_example/install/setup.bash
source /home/bxi/zerolab-sim2real-candidate-20260818/install-safe-arm/setup.bash
export ROS_DOMAIN_ID=42
```

Keep this PD Brake command ready in terminal history:

```bash
ros2 topic pub --once /motion_commands communication/msg/MotionCommands '{btn_3: 1}'
ros2 topic pub --once /motion_commands communication/msg/MotionCommands '{}'
```

- [ ] **Step 4: Enter PD Brake, then stable Normal**

```bash
ros2 topic pub --once /motion_commands communication/msg/MotionCommands '{btn_3: 1}'
ros2 topic pub --once /motion_commands communication/msg/MotionCommands '{}'
sleep 3
ros2 topic pub --once /motion_commands communication/msg/MotionCommands '{btn_1: 1}'
ros2 topic pub --once /motion_commands communication/msg/MotionCommands '{}'
sleep 5
timeout 5 ros2 topic echo --no-daemon --once --full-length --field data \
  /hardware/state_machine_info std_msgs/msg/String
```

Expected: state is `com.bxi.basic_actions/normal`, the robot is standing still,
and commanded walking velocity is zero.

- [ ] **Step 5: Enter ZeroLab and verify calibration cannot move the robot**

```bash
ros2 topic pub --once /motion_commands communication/msg/MotionCommands '{btn_10: 11}'
ros2 topic pub --once /motion_commands communication/msg/MotionCommands '{}'
```

The operator holds T-pose. Terminal 1 must show, in order, source/bridge ready
and:

```text
ZeroLab ARM phase: WAIT_ARM
```

Before `btn_10=12`, the robot must remain on the captured Normal motor command
and must not move toward T-pose. If it moves, issue PD Brake immediately and
stop acceptance.

- [ ] **Step 6: Return to neutral and perform the explicit two-second ARM**

After the human returns to the agreed neutral pose, the safety operator sends:

```bash
ros2 topic pub --once /motion_commands communication/msg/MotionCommands '{btn_10: 12}'
ros2 topic pub --once /motion_commands communication/msg/MotionCommands '{}'
```

Expected terminal 1 messages:

```text
ZeroLab ARM accepted; blending for 2.000 s
ZeroLab ARM phase: ARMED
```

Hold the neutral pose for the full two seconds. Verify no instantaneous shoulder
rotation, then test only small-amplitude left arm, right arm, both arms, and arm
lowering. Any wrong direction, rapid movement, loss of balance, or unexpected
shoulder plane requires immediate PD Brake and termination of this run.

- [ ] **Step 7: Verify stale hold and explicit recovery**

Stop the Windows ZeroLab UDP sender without stopping the ROS hardware stack.
After more than 0.5 seconds, terminal 1 must show:

```text
ZeroLab reference stale; holding last motor frame and ARM cancelled
```

The robot must hold the last command and must not switch state automatically.
Restart the sender. Terminal 1 must show recovery but the robot must not resume
motion. Send `btn_10=12` again and verify another complete two-second blend.

- [ ] **Step 8: Exit to Normal and verify a new session requires T-pose**

```bash
ros2 topic pub --once /motion_commands communication/msg/MotionCommands '{btn_1: 1}'
ros2 topic pub --once /motion_commands communication/msg/MotionCommands '{}'
sleep 3
```

Terminal 1 must return to Normal. Re-enter `btn_10=11` once more and verify the
phase starts at `WAIT_CALIBRATION`, not `WAIT_ARM` or `ARMED`. Return to Normal
without ARM.

- [ ] **Step 9: Shut down normally and confirm process/port cleanup**

Press `Ctrl-C` once in terminal 1 and wait for launch shutdown. Back in terminal
2 as root:

```bash
ps -eo pid=,user=,stat=,etime=,args= | \
  grep -E '[h]ardware_elf3|[b]xi_example_py_elf3_demo' \
  || echo 'CONTROLLERS_STOPPED=PASS'
ss -H -lunp 'sport = :18000'
ss -H -ltnp 'sport = :5558'
ss -H -ltnp 'sport = :5557'
```

Expected: no controller process and no listener on 18000, 5558, or 5557.

## Completion Evidence

Before declaring the implementation complete, collect:

- Commit IDs for Tasks 1-4.
- `git diff --check 46e3724..HEAD` with no output.
- Focused and full pytest summaries.
- Local and robot colcon summaries.
- Local and installed manifest grep output showing `btn_10=12` and `2.0`.
- MuJoCo replay report with 718 samples, no joint-limit violations, no fall, and unchanged RMSE.
- Matching original robot worktree status snapshots.
- Root-terminal logs for `WAIT_CALIBRATION`, `WAIT_ARM`, `BLENDING`, `ARMED`, stale hold, recovery, re-ARM, and clean exit.

Do not merge, push, overwrite the original robot workspace, or delete deployment evidence as part of this plan.
