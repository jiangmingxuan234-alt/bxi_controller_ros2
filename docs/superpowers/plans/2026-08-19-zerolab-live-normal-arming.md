# ZeroLab Live-Normal Safe Arming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace ZeroLab's pre-ARM captured-frame output with a live zero-command Normal balancing policy, then perform the first two-second handover between current Normal and current SONIC outputs.

**Architecture:** Keep the existing `ZeroLabArmedTeleopState`, `btn_10=11` entry, `btn_10=12` action, and state-scoped source/bridge lifecycle. Inject the existing `com.bxi.basic_actions/normal_policy` resource only into the ZeroLab state; advance both policies while waiting and during the first blend, stop Normal inference once armed, and retain the frozen-frame recovery blend after stale input.

**Tech Stack:** Ubuntu 22.04, ROS 2 Humble, Python 3.10, NumPy, pytest, YAML Mod manifests, colcon, MuJoCo.

## Global Constraints

- Preserve `btn_10=9` for PICO SONIC, `btn_10=11` for ZeroLab entry, and `btn_10=12` for explicit ZeroLab ARM.
- Keep `live_reference_timeout_s=0.5` and source `stale_seconds=0.5` unchanged.
- Do not change PICO `sonic_teleop`, ZeroLab packet parsing, quaternion conversion, T-pose calibration, SONIC reference playback, policy weights, MuJoCo XML, or hardware drivers.
- During `WAIT_CALIBRATION` and `WAIT_ARM`, apply a live Normal policy frame with zero commanded velocity on every advancing control tick.
- During the first two-second ARM blend, mix the current live Normal frame with the current live SONIC frame using smoothstep across `qpos`, `kp`, `kd`, `vel`, and `torque`.
- After `ARMED`, stop advancing Normal and apply only live SONIC.
- After stale input in `BLENDING` or `ARMED`, preserve complete-frame hold and explicit re-ARM; recovery blends from the frozen frame, not from Normal.
- Do not deploy to or run on robot hardware in this plan. MuJoCo acceptance is the terminal gate.
- Preserve existing untracked `build-safe-arm/`, `install-safe-arm/`, and `log-safe-arm/`; use new `*-live-normal` artifact directories.

## File Structure

- Modify `src/bxi_example_py_elf3/mods/com.bxi.sonic/state.py`: allow a ZeroLab-only additional required policy resource without changing default PICO construction.
- Modify `src/bxi_example_py_elf3/mods/com.bxi.sonic/plugin.py`: request the existing Normal policy resource and inject it only into `sonic_zerolab`.
- Modify `src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/state.py`: sample live Normal, distinguish first and recovery blends, enforce pre-ARM orientation safety, and manage new buffers.
- Modify `src/bxi_example_py_elf3/test/test_zerolab_manifest.py`: verify resource wiring and PICO isolation.
- Modify `src/bxi_example_py_elf3/test/test_zerolab_arming.py`: provide independent fake Normal and SONIC policies and cover waiting, blending, stale, safety, and inference counts.
- Modify `src/bxi_example_py_elf3/mods/com.bxi.sonic/README.md`: replace captured-frame wording with live-Normal behavior and document the MuJoCo gate.
- Create isolated runtime artifacts only: `build-live-normal/`, `install-live-normal/`, and `log-live-normal/` (never commit them).

---

### Task 1: Wire the Existing Normal Policy into ZeroLab Only

**Files:**
- Modify: `src/bxi_example_py_elf3/mods/com.bxi.sonic/state.py:101-160`
- Modify: `src/bxi_example_py_elf3/mods/com.bxi.sonic/plugin.py:1-160`
- Modify: `src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/state.py:30-50`
- Test: `src/bxi_example_py_elf3/test/test_zerolab_manifest.py:150-230`

**Interfaces:**
- Consumes: global resource ID `com.bxi.basic_actions/normal_policy` owned by the required `com.bxi.basic_actions` Mod.
- Produces: `ZeroLabArmedTeleopState(..., normal_policy: ResourceHandle[HumanoidGaitPolicyLiteIsaaclab], ...)`; its `required_resources` contains SONIC and Normal handles. `SonicTeleopState` keeps only SONIC.

- [ ] **Step 1: Write the failing resource-isolation assertions**

The SONIC plugin will now resolve a resource owned by Basic Actions, so extend
the test fixture to load that required Mod before SONIC:

```python
BASIC_ACTIONS_ROOT = MOD_ROOT.parent / "com.bxi.basic_actions"


def test_source_prompts_and_zerolab_availability_without_live_data():
    resources = ResourceManager()
    module_prefixes = []
    try:
        discovered = _discover_mods((BASIC_ACTIONS_ROOT, MOD_ROOT))
        basic_definition, basic_module = _load_definition(
            discovered["com.bxi.basic_actions"], resources
        )
        module_prefixes.append(basic_module.__name__.split(".", 1)[0])
        definition, module = _load_definition(
            discovered["com.bxi.sonic"], resources
        )
        module_prefixes.append(module.__name__.split(".", 1)[0])
    finally:
        resources.close()
        _remove_module_prefixes(tuple(module_prefixes))
```

Replace the existing singular `module_prefix` setup/cleanup with the code
above, leaving the assertions between SONIC loading and `finally` in their
current order. Construct the Basic Actions Normal state next to PICO and
ZeroLab:

```python
        normal_context = StateBuildContext(
            "com.bxi.basic_actions/normal", 0, {}
        )
        normal = basic_definition.state_factories["normal"](normal_context)
        normal_context.finish()
```

Then add the resource IDs and assertions after constructing all three states:

```python
        sonic_policy_id = "com.bxi.sonic/policy"
        normal_policy_id = "com.bxi.basic_actions/normal_policy"
        assert [handle.key.id for handle in pico.required_resources] == [
            sonic_policy_id
        ]
        assert [handle.key.id for handle in zero.required_resources] == [
            sonic_policy_id,
            normal_policy_id,
        ]
        assert zero._normal_policy.key.id == normal_policy_id
        assert zero._normal_policy.key == normal._policy.key
```

Keep the existing `type(pico).__name__`, `type(zero).__name__`, prompt, and shared SONIC policy assertions.

- [ ] **Step 2: Run the focused test and verify RED**

Run from `src/bxi_example_py_elf3`:

```bash
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m pytest -q \
  test/test_zerolab_manifest.py::test_source_prompts_and_zerolab_availability_without_live_data
```

Expected: FAIL because ZeroLab has no Normal resource and `SonicTeleopState`
accepts no additional resources. It must not fail with `no loaded Mod provides
resource`; that indicates the Basic Actions test fixture was not loaded first.

- [ ] **Step 3: Add optional required resources to the SONIC base state**

In `state.py`, import `Any`, add the keyword, and include it in the base resource tuple:

```python
from typing import Any, TYPE_CHECKING, Optional, Protocol

    def __init__(
        self,
        name: str,
        state_id: int,
        policy: ResourceHandle[SonicPolicy],
        *,
        additional_resources: tuple[ResourceHandle[Any], ...] = (),
        require_live_reference: bool = False,
    ) -> None:
        super().__init__(
            name,
            state_id,
            resources=(policy, *additional_resources),
        )
```

Insert `additional_resources` immediately before the existing
`require_live_reference` keyword and retain every later constructor keyword.
Do not pass `additional_resources` from `_build_state`; its default keeps PICO
unchanged.

- [ ] **Step 4: Request and inject the existing Normal resource**

In `plugin.py`, define the same globally unique key used by Basic Actions and pass its handle only to ZeroLab:

```python
from bxi_example_py_elf3.policies import HumanoidGaitPolicyLiteIsaaclab

NORMAL_POLICY = ResourceKey[HumanoidGaitPolicyLiteIsaaclab](
    "com.bxi.basic_actions/normal_policy"
)

def _build_zerolab_state(
    state: StateBuildContext,
    policy,
    normal_policy,
) -> ZeroLabArmedTeleopState:
    return ZeroLabArmedTeleopState(
        state.name,
        state.state_id,
        policy,
        normal_policy=normal_policy,
        arm_blend_seconds=state.float_param("arm_blend_seconds", 2.0),
        **_common_state_kwargs(state),
    )

def create_mod(context: ModLoadContext) -> ModDefinition:
    context.register_resource(SONIC_POLICY, _load_policy, policy="startup")
    policy = context.resource(SONIC_POLICY)
    normal_policy = context.resource(NORMAL_POLICY)
    return ModDefinition(
        state_factories={
            "sonic_teleop": lambda state: _build_state(state, policy),
            "sonic_zerolab": lambda state: _build_zerolab_state(
                state, policy, normal_policy
            ),
        }
    )

__all__ = ["NORMAL_POLICY", "SONIC_POLICY", "create_mod"]
```

In `zerolab/state.py`, accept and store the handle while registering it with the base state:

```python
from bxi_example_py_elf3.framework.mod_api import ResourceHandle
from bxi_example_py_elf3.policies import HumanoidGaitPolicyLiteIsaaclab

    def __init__(
        self,
        name,
        state_id,
        policy,
        *,
        normal_policy: ResourceHandle[HumanoidGaitPolicyLiteIsaaclab],
        arm_blend_seconds=2.0,
        **kwargs,
    ):
        super().__init__(
            name,
            state_id,
            policy,
            additional_resources=(normal_policy,),
            **kwargs,
        )
        self._normal_policy = normal_policy
```

- [ ] **Step 5: Run focused manifest tests and verify GREEN**

```bash
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m pytest -q test/test_zerolab_manifest.py
```

Expected: all manifest tests pass and PICO still has exactly one required resource.

- [ ] **Step 6: Commit Task 1**

```bash
git add \
  src/bxi_example_py_elf3/mods/com.bxi.sonic/state.py \
  src/bxi_example_py_elf3/mods/com.bxi.sonic/plugin.py \
  src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/state.py \
  src/bxi_example_py_elf3/test/test_zerolab_manifest.py
git commit -m "refactor(sonic): inject Normal policy into ZeroLab"
```

---

### Task 2: Apply Live Zero-Command Normal While Waiting

**Files:**
- Modify: `src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/state.py:35-225`
- Test: `src/bxi_example_py_elf3/test/test_zerolab_arming.py:35-370`
- Test: `src/bxi_example_py_elf3/test/test_zerolab_manifest.py:185-230`

**Interfaces:**
- Consumes: `self._normal_policy` from Task 1 and the inherited `get_cmd_vel()`/`_motor_frame_from_target()` APIs.
- Produces: `_sample_normal_frame(ctx, dt, *, advance) -> MotorFrame`; waiting phases return a copied current Normal frame and still advance SONIC/reference state.

- [ ] **Step 1: Add an independent fake Normal policy and command-aware context**

Extend `test_zerolab_arming.py` with:

```python
class FakeNormalPolicy:
    def __init__(self):
        target = JointTargetBuffer(ELF3_POLICY_JOINTS)
        target.kp.fill(40.0)
        target.kd.fill(2.0)
        self.output = PolicyOutput(target.view)
        self.target = target
        self.step_calls = 0

    def step(self, _frame, _dt, *, advance=True):
        if advance:
            self.step_calls += 1
        return self.output


class FakeInferenceFrame:
    def __init__(self, command):
        self.command = command
```

Update `FakeContext.__init__` so the command array used by the inference frame is shared:

```python
        self.current_raw_cmd_vel = np.array([0.8, -0.4, 0.6], dtype=np.float32)
        self.current_cmd_vel = np.zeros(3, dtype=np.float32)
        self.speed_profiles = {}
        self.inference_frame = FakeInferenceFrame(self.current_cmd_vel)
```

Update `make_state` to construct `normal_policy = FakeNormalPolicy()`, pass
`normal_policy=FakeHandle(normal_policy)`, and return
`state, policy, normal_policy, ctx`. Change the return tuples and unpacking in
`prepared_state`, `armed_state_at_half_blend`, `fully_armed_state`,
`state_in_active_phase`, `stale_hold_state`, and `waiting_arm_state` to that
exact four-item order; test-local variables should use the names
`state, sonic, normal, ctx`.

- [ ] **Step 2: Replace captured-frame waiting tests with live-Normal tests**

Replace `test_prepare_deep_copies_entry_frame_and_waiting_advances_policy` and
`test_wait_arm_gap_never_changes_held_normal_frame` with:

```python
def test_waiting_applies_changing_live_normal_with_zero_command():
    state, sonic, normal, ctx = prepared_state(entry_value=0.25)
    normal.target.position.fill(0.4)

    first = copy_motor_frame(state.sample_running_frame(ctx, 0.02, advance=True))
    normal.target.position.fill(0.7)
    second = copy_motor_frame(state.sample_running_frame(ctx, 0.02, advance=True))

    assert state.arm_phase is ZeroLabArmPhase.WAIT_CALIBRATION
    np.testing.assert_allclose(first.qpos, 0.4)
    np.testing.assert_allclose(second.qpos, 0.7)
    np.testing.assert_allclose(ctx.inference_frame.command, 0.0)
    assert sonic.step_calls == 2
    assert normal.step_calls == 2
    np.testing.assert_allclose(state.get_entry_frame(ctx).qpos, 0.25)


def test_wait_arm_stale_gap_continues_live_normal():
    state, sonic, normal, ctx = waiting_arm_state()
    sonic.fresh = False
    normal.target.position.fill(0.6)

    frame = state.sample_running_frame(ctx, 1.0, advance=True)

    assert state.arm_phase is ZeroLabArmPhase.WAIT_ARM
    np.testing.assert_allclose(frame.qpos, 0.6)
    assert normal.step_calls >= 2
```

Update `test_fresh_reference_waits_for_explicit_arm` to assert the current
Normal value `0.0`, not the captured `0.25` entry value.

Replace the old fixed-entry all-field blend test with this temporary green
checkpoint; Task 3 will replace it with the dynamic-source and explicit
all-field tests:

```python
def test_first_arm_snapshot_blend_still_completes_two_seconds():
    state, sonic, normal, ctx = prepared_state(entry_value=0.0)
    sonic.fresh = True
    state.sample_running_frame(ctx, 0.02, advance=True)
    state.on_action(ctx, "arm_zerolab")
    sonic.target.position.fill(2.0)

    first = copy_motor_frame(state.sample_running_frame(ctx, 0.0, advance=True))
    middle = copy_motor_frame(state.sample_running_frame(ctx, 1.0, advance=True))
    final = copy_motor_frame(state.sample_running_frame(ctx, 1.0, advance=True))

    np.testing.assert_allclose(first.qpos, 0.0)
    np.testing.assert_allclose(middle.qpos, 1.0)
    np.testing.assert_allclose(final.qpos, 2.0)
    assert state.arm_phase is ZeroLabArmPhase.ARMED
```

Because `on_prepare` gains one bounded log, update the exact manifest lifecycle
expectation to:

```python
        assert logger.messages == [
            "ZeroLab ARM phase: WAIT_CALIBRATION",
            "ZeroLab pre-ARM output: live zero-command Normal policy",
            "SONIC遥操已启动；头部跟踪已关闭；" + prompt,
        ]
```

- [ ] **Step 3: Run waiting tests and verify RED**

```bash
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m pytest -q test/test_zerolab_arming.py \
  -k 'waiting or wait_arm or fresh_reference'
```

Expected: FAIL because waiting still returns `_hold_frame` and never advances Normal.

- [ ] **Step 4: Implement live Normal sampling and buffers**

In `ZeroLabArmedTeleopState`:

```python
        self._normal_frame = None

    def _sample_normal_frame(self, ctx, dt, *, advance):
        assert self._normal_frame is not None
        self.get_cmd_vel(ctx)
        output = self._normal_policy.get().step(
            ctx.inference_frame,
            dt,
            advance=advance,
        )
        natural = self._motor_frame_from_target(ctx, output.joints)
        return ctx.resolve_motor_frame(natural, self._normal_frame)
```

Allocate `_normal_frame = MotorFrame.empty(ctx.robot_layout)` in `on_prepare`
and clear it in `on_exit`. After advancing/resolving SONIC and updating
freshness, replace the final waiting branch with:

```python
        if self._arm_phase in (
            ZeroLabArmPhase.WAIT_CALIBRATION,
            ZeroLabArmPhase.WAIT_ARM,
        ):
            normal = self._sample_normal_frame(ctx, dt, advance=True)
            return self._copy_frame(self._applied_frame, normal)
```

Retain `get_entry_frame()` as the captured frame for the framework's short
entry transition only. Add one `on_prepare` info log:

```python
self.logger.info("ZeroLab pre-ARM output: live zero-command Normal policy")
```

- [ ] **Step 5: Run the complete arming test module and verify GREEN**

```bash
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m pytest -q \
  test/test_zerolab_arming.py \
  test/test_zerolab_manifest.py
```

Expected: all updated arming and manifest tests pass with no skipped tests.

- [ ] **Step 6: Commit Task 2**

```bash
git add \
  src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/state.py \
  src/bxi_example_py_elf3/test/test_zerolab_arming.py \
  src/bxi_example_py_elf3/test/test_zerolab_manifest.py
git commit -m "fix(sonic): balance ZeroLab pre-arm with live Normal"
```

---

### Task 3: Blend Current Normal into Current SONIC on First ARM

**Files:**
- Modify: `src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/state.py:20-225`
- Test: `src/bxi_example_py_elf3/test/test_zerolab_arming.py:130-340`

**Interfaces:**
- Consumes: `_sample_normal_frame()` and `_normal_frame` from Task 2.
- Produces: private `ZeroLabBlendSource` with `LIVE_NORMAL` and `FROZEN`; `_blend_frames(source, target, output, alpha) -> MotorFrame`; first ARM uses live Normal, recovery ARM uses the frozen frame.

- [ ] **Step 1: Write failing first-blend and inference-count tests**

Replace `test_first_arm_snapshot_blend_still_completes_two_seconds` with:

```python
def test_first_arm_blends_current_normal_and_current_sonic():
    state, sonic, normal, ctx = waiting_arm_state()
    normal.target.position.fill(0.5)
    sonic.target.position.fill(2.0)
    state.on_action(ctx, "arm_zerolab")

    normal.target.position.fill(1.0)
    first = copy_motor_frame(state.sample_running_frame(ctx, 0.0, advance=True))
    normal.target.position.fill(1.2)
    middle = copy_motor_frame(state.sample_running_frame(ctx, 1.0, advance=True))
    final = copy_motor_frame(state.sample_running_frame(ctx, 1.0, advance=True))

    np.testing.assert_allclose(first.qpos, 1.0)
    np.testing.assert_allclose(middle.qpos, 1.6)
    np.testing.assert_allclose(final.qpos, 2.0)
    assert state.arm_phase is ZeroLabArmPhase.ARMED

    normal_calls = normal.step_calls
    state.sample_running_frame(ctx, 0.02, advance=True)
    assert normal.step_calls == normal_calls
```

Add an all-field helper test using arbitrary complete frames:

```python
def test_blend_frames_interpolates_all_complete_motor_fields():
    source = full_motor_frame(qpos=0.0, kp=10.0, kd=1.0, vel=2.0, torque=3.0)
    target = full_motor_frame(qpos=2.0, kp=30.0, kd=5.0, vel=6.0, torque=7.0)
    output = MotorFrame.empty(ELF3_POLICY_JOINTS)

    ZeroLabArmedTeleopState._blend_frames(source, target, output, 0.5)

    np.testing.assert_allclose(output.qpos, 1.0)
    np.testing.assert_allclose(output.kp, 20.0)
    np.testing.assert_allclose(output.kd, 3.0)
    np.testing.assert_allclose(output.vel, 4.0)
    np.testing.assert_allclose(output.torque, 5.0)
```

Define the helper explicitly:

```python
def full_motor_frame(*, qpos, kp, kd, vel, torque):
    size = ELF3_POLICY_JOINTS.dof_num
    values = lambda value: np.full(size, value, dtype=np.float32)
    return MotorFrame.create(
        ELF3_POLICY_JOINTS,
        values(qpos),
        values(kp),
        values(kd),
        vel=values(vel),
        torque=values(torque),
    )
```

- [ ] **Step 2: Run the focused blend tests and verify RED**

```bash
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m pytest -q test/test_zerolab_arming.py \
  -k 'first_arm or blend_frames or recovery'
```

Expected: FAIL because first ARM still mixes from a fixed snapshot and `_blend_frames` does not exist.

- [ ] **Step 3: Distinguish initial and recovery blend sources**

Add:

```python
class ZeroLabBlendSource(str, Enum):
    LIVE_NORMAL = "live_normal"
    FROZEN = "frozen"
```

Initialize `_blend_source = ZeroLabBlendSource.LIVE_NORMAL`. In
`_begin_blend`, inspect the phase before calling `_set_phase`:

```python
        initial_arm = self._arm_phase is ZeroLabArmPhase.WAIT_ARM
        self._blend_source = (
            ZeroLabBlendSource.LIVE_NORMAL
            if initial_arm
            else ZeroLabBlendSource.FROZEN
        )
        if not initial_arm:
            self._copy_frame(self._blend_start_frame, self._applied_frame)
```

Log `blending live Normal -> SONIC` for the initial ARM and
`blending frozen frame -> SONIC` for recovery, each including the configured
duration. Reset `_blend_source` to `LIVE_NORMAL` in `on_prepare` and `on_exit`
so every new ZeroLab session starts with explicit initial-blend semantics.

- [ ] **Step 4: Extract complete-frame interpolation and use a dynamic source**

Add:

```python
    @staticmethod
    def _blend_frames(source, target, output, alpha):
        for start, end, destination in (
            (source.qpos, target.qpos, output.qpos),
            (source.kp, target.kp, output.kp),
            (source.kd, target.kd, output.kd),
            (source.vel, target.vel, output.vel),
            (source.torque, target.torque, output.torque),
        ):
            np.subtract(end, start, out=destination)
            destination *= alpha
            destination += start
        return output
```

In `BLENDING`, choose the source each tick:

```python
            if self._blend_source is ZeroLabBlendSource.LIVE_NORMAL:
                blend_source = self._sample_normal_frame(
                    ctx, dt, advance=True
                )
            else:
                blend_source = self._blend_start_frame
            self._blend_frames(
                blend_source,
                self._live_frame,
                self._applied_frame,
                alpha,
            )
```

Do not sample Normal in `ARMED`, `HOLD_STALE`, or frozen recovery blend.

- [ ] **Step 5: Run arming tests and verify GREEN**

```bash
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m pytest -q test/test_zerolab_arming.py
```

Expected: all arming tests pass, including existing stale/recovery coverage.

- [ ] **Step 6: Commit Task 3**

```bash
git add \
  src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/state.py \
  src/bxi_example_py_elf3/test/test_zerolab_arming.py
git commit -m "feat(sonic): blend live Normal into ZeroLab"
```

---

### Task 4: Preserve Pre-ARM Normal Safety and Stale Recovery Semantics

**Files:**
- Modify: `src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/state.py:120-270`
- Test: `src/bxi_example_py_elf3/test/test_zerolab_arming.py:250-430`

**Interfaces:**
- Consumes: `ZeroLabBlendSource` from Task 3.
- Produces: `_normal_balance_active() -> bool` and an `on_update` guard that requests `com.bxi.basic_actions/zero_torque` only while live Normal is part of the applied control path.

- [ ] **Step 1: Add safety methods to the fake context and write failing tests**

Extend `FakeContext`:

```python
        self.current_quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0])
        self.orientation_unsafe = False
        self.requested_states = []

    def is_orientation_unsafe(self, _quat):
        return self.orientation_unsafe

    def request_state(self, name, *, trigger):
        self.requested_states.append((name, trigger))
        return True
```

Add:

```python
@pytest.mark.parametrize(
    "phase_name",
    ["wait_calibration", "wait_arm", "initial_blend"],
)
def test_live_normal_phases_keep_normal_orientation_safety(phase_name):
    if phase_name == "wait_calibration":
        state, sonic, normal, ctx = prepared_state()
    else:
        state, sonic, normal, ctx = waiting_arm_state()
    if phase_name == "initial_blend":
        state.on_action(ctx, "arm_zerolab")
    ctx.orientation_unsafe = True

    state.on_update(ctx, 0.02)

    assert ctx.requested_states == [
        ("com.bxi.basic_actions/zero_torque", "safety")
    ]
    assert ctx.applied is None


def test_fully_armed_keeps_existing_sonic_orientation_behavior():
    state, sonic, normal, ctx = fully_armed_state()
    ctx.orientation_unsafe = True

    state.on_update(ctx, 0.02)

    assert ctx.requested_states == []
    assert ctx.applied is not None


def test_frozen_recovery_blend_keeps_existing_sonic_orientation_behavior():
    state, sonic, normal, ctx = stale_hold_state()
    sonic.fresh = True
    state.on_action(ctx, "arm_zerolab")
    ctx.orientation_unsafe = True

    state.on_update(ctx, 0.02)

    assert state._blend_source is arming_module.ZeroLabBlendSource.FROZEN
    assert ctx.requested_states == []
    assert ctx.applied is not None
```

- [ ] **Step 2: Run the safety tests and verify RED**

```bash
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m pytest -q test/test_zerolab_arming.py -k orientation
```

Expected: FAIL because ZeroLab currently inherits the disabled SONIC orientation guard.

- [ ] **Step 3: Implement phase-specific Normal orientation safety**

In `ZeroLabArmedTeleopState`:

```python
    def _normal_balance_active(self) -> bool:
        return self._arm_phase in (
            ZeroLabArmPhase.WAIT_CALIBRATION,
            ZeroLabArmPhase.WAIT_ARM,
        ) or (
            self._arm_phase is ZeroLabArmPhase.BLENDING
            and self._blend_source is ZeroLabBlendSource.LIVE_NORMAL
        )

    def on_update(self, ctx, dt):
        if self._normal_balance_active() and ctx.is_orientation_unsafe(
            ctx.current_quat_xyzw
        ):
            ctx.request_state(
                "com.bxi.basic_actions/zero_torque",
                trigger="safety",
            )
            return
        super().on_update(ctx, dt)
```

Do not alter the established fully armed SONIC behavior in this change.

- [ ] **Step 4: Run stale, recovery, lifecycle, and safety regression tests**

```bash
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m pytest -q \
  test/test_zerolab_arming.py \
  test/test_zerolab_manifest.py \
  test/test_zerolab_lifecycle.py \
  test/test_sonic_ordered_playout.py
```

Expected: all selected tests pass. Confirm there are no skips added to hide failures.

- [ ] **Step 5: Commit Task 4**

```bash
git add \
  src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/state.py \
  src/bxi_example_py_elf3/test/test_zerolab_arming.py
git commit -m "fix(sonic): retain Normal safety before ZeroLab ARM"
```

---

### Task 5: Update Operator Documentation and Run Source Regressions

**Files:**
- Modify: `src/bxi_example_py_elf3/mods/com.bxi.sonic/README.md:536-640`
- Verify: all runtime and test files changed in Tasks 1-4

**Interfaces:**
- Consumes: the final live-Normal state behavior and log text.
- Produces: copyable simulation instructions that never describe a captured frame as ongoing balance control.

- [ ] **Step 1: Update the ZeroLab MuJoCo instructions**

Replace captured-frame wording with these exact behavioral statements:

```text
btn_10=11 后，ZeroLab source/bridge 与 SONIC policy 在后台运行；电机输出由零速度 Normal policy 每周期更新，不是重复一张 Normal 快照。
WAIT_CALIBRATION 和 WAIT_ARM 中的数据短停不会停止 Normal 平衡，也不会允许 SONIC 接管。
首次 btn_10=12 在两秒内动态混合当前 Normal 与当前 SONIC 输出；ARMED 后停止 Normal inference。
BLENDING/ARMED 断流仍冻结最后实际输出并取消 ARM；恢复后必须再次发送 btn_10=12，恢复 blend 从冻结帧开始。
```

Keep the existing exact `btn_10=11`, `btn_10=12`, PD Brake, Normal exit, and
port checks. Change both simulation terminals from `source install/setup.bash`
to `source install-live-normal/setup.bash`, add the exact prefix assertion
`test "$(ros2 pkg prefix bxi_example_py_elf3)" = "$PWD/install-live-normal"`,
and state that hardware testing remains forbidden until every MuJoCo
acceptance item in Task 7 passes.

- [ ] **Step 2: Run all ZeroLab tests**

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
  test/test_zerolab_arming.py \
  test/test_sonic_ordered_playout.py
```

Expected: all selected tests pass.

- [ ] **Step 3: Run the complete package test directory**

```bash
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m pytest -q test
```

Expected: all tests pass with no collection errors, new skips, or warnings caused by missing Normal resources.

- [ ] **Step 4: Review the source diff against forbidden paths**

From the worktree root:

```bash
git diff --name-only 155040f..HEAD -- src/bxi_example_py_elf3
git diff --check 155040f..HEAD -- src/bxi_example_py_elf3
```

Expected runtime/test/doc changes are limited to the six source paths in the
File Structure section. No PICO node, converter, policy, model asset, MuJoCo
XML, hardware, or base-framework path may appear.

- [ ] **Step 5: Commit Task 5**

```bash
git add src/bxi_example_py_elf3/mods/com.bxi.sonic/README.md
git commit -m "docs(sonic): document live-Normal ZeroLab arming"
```

---

### Task 6: Build an Isolated Candidate and Verify Installed Wiring

**Files:**
- Create runtime-only: `build-live-normal/`
- Create runtime-only: `install-live-normal/`
- Create runtime-only: `log-live-normal/`
- Verify installed files under `install-live-normal/share/bxi_example_py_elf3`

**Interfaces:**
- Consumes: all source and tests from Tasks 1-5.
- Produces: an isolated local install for MuJoCo acceptance; does not overwrite `install-safe-arm` or the default install.

- [ ] **Step 1: Build into new artifact directories**

From the worktree root:

```bash
source /opt/ros/humble/setup.bash
source /home/fazepurple/ros2_ws/bxi_ros2_pkg/setup.bash
colcon --log-base log-live-normal build \
  --merge-install \
  --symlink-install \
  --base-paths src \
  --packages-select bxi_example_py_elf3 \
  --build-base build-live-normal \
  --install-base install-live-normal \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
```

Expected: `Summary: 1 package finished`.

- [ ] **Step 2: Verify the isolated prefix and installed source**

```bash
source install-live-normal/setup.bash
test "$(ros2 pkg prefix bxi_example_py_elf3)" = "$PWD/install-live-normal"
grep -nE 'arm_zerolab|value: 12|arm_blend_seconds: 2.0' \
  install-live-normal/share/bxi_example_py_elf3/mods/com.bxi.sonic/mod.yaml
grep -n 'live zero-command Normal policy' \
  install-live-normal/lib/python3.10/site-packages/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/state.py \
  install-live-normal/share/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/state.py \
  2>/dev/null || true
```

Expected: prefix equals `install-live-normal`; the installed manifest contains
the ARM event and two-second parameter; at least one installed state path
contains the live-Normal log. If package layout differs, use
`python3 -c 'import bxi_example_py_elf3, pathlib; print(pathlib.Path(bxi_example_py_elf3.__file__).resolve())'`
to locate it and verify the same source text.

- [ ] **Step 3: Confirm runtime factory resource isolation from the install**

```bash
INSTALL_MODS="$PWD/install-live-normal/share/bxi_example_py_elf3/mods" \
python3 - <<'PY'
import os
from pathlib import Path

from bxi_example_py_elf3.framework.mod_api import StateBuildContext
from bxi_example_py_elf3.framework.runtime.mod_loader import (
    _discover_mods,
    _load_definition,
    _remove_module_prefixes,
)
from bxi_example_py_elf3.framework.runtime.resource_manager import ResourceManager

mods = Path(os.environ["INSTALL_MODS"])
resources = ResourceManager()
prefixes = []
try:
    discovered = _discover_mods(
        (mods / "com.bxi.basic_actions", mods / "com.bxi.sonic")
    )
    _, basic_module = _load_definition(
        discovered["com.bxi.basic_actions"], resources
    )
    prefixes.append(basic_module.__name__.split(".", 1)[0])
    sonic, sonic_module = _load_definition(
        discovered["com.bxi.sonic"], resources
    )
    prefixes.append(sonic_module.__name__.split(".", 1)[0])

    pico_ctx = StateBuildContext("com.bxi.sonic/sonic_teleop", 1, {})
    pico = sonic.state_factories["sonic_teleop"](pico_ctx)
    pico_ctx.finish()
    zero_ctx = StateBuildContext("com.bxi.sonic/sonic_zerolab", 2, {})
    zero = sonic.state_factories["sonic_zerolab"](zero_ctx)
    zero_ctx.finish()

    assert [item.key.id for item in pico.required_resources] == [
        "com.bxi.sonic/policy"
    ]
    assert [item.key.id for item in zero.required_resources] == [
        "com.bxi.sonic/policy",
        "com.bxi.basic_actions/normal_policy",
    ]
    print("INSTALLED_FACTORY_RESOURCE_ISOLATION=PASS")
finally:
    resources.close()
    _remove_module_prefixes(tuple(prefixes))
PY
```

Expected: `INSTALLED_FACTORY_RESOURCE_ISOLATION=PASS`. This script loads both
factories from `install-live-normal/share`, rather than accidentally reusing
the source-tree `MOD_ROOT` hard-coded by the pytest module.

- [ ] **Step 4: Confirm the tracked tree is clean**

No runtime artifacts are committed. Confirm:

```bash
git status --short
```

Expected: only untracked build/install/log artifact directories appear; no tracked source changes remain uncommitted.

---

### Task 7: MuJoCo Live Acceptance and Stop Gate

**Files:**
- Read only: `install-live-normal/`
- Runtime output only: terminal logs and `/home/fazepurple/.ros/log`

**Interfaces:**
- Consumes: isolated install from Task 6 and live ZeroLab UDP from `192.168.89.171` to local `192.168.88.161:18000`.
- Produces: evidence for ten-second live-Normal waiting, first ARM blend, active motion, stale hold, explicit recovery, timing, and clean shutdown. It produces no robot deployment.

- [ ] **Step 1: Confirm simulation-only exclusivity and free local ports**

```bash
pgrep -af \
  '[h]ardware_elf3|[b]xi_example_py_elf3_demo|[m]od_node_runner.*zerolab|[z]erolab.record_cli' \
  || echo 'PROCESS_CHECK=PASS'
ss -H -lunp 'sport = :18000'
ss -H -ltnp 'sport = :5558'
ss -H -ltnp 'sport = :5557'
```

Expected: no `hardware_elf3`, controller, standalone ZeroLab process, or port
listener. If `code` owns `5557/5558`, stop those forwards in VS Code's Ports
panel; do not kill the entire editor.

- [ ] **Step 2: Start MuJoCo from the isolated install in terminal 1**

```bash
cd /home/fazepurple/ros2_ws/bxi_rl_controller_ros2_example_dev/.worktrees/konodoki-dev
source /opt/ros/humble/setup.bash
source /home/fazepurple/ros2_ws/bxi_ros2_pkg/setup.bash
source install-live-normal/setup.bash
export ROS_DOMAIN_ID=42
test "$(ros2 pkg prefix bxi_example_py_elf3)" = "$PWD/install-live-normal"
ros2 launch bxi_example_py_elf3 example_demo.launch.py
```

- [ ] **Step 3: Prepare terminal 2 and reach stable Normal**

```bash
cd /home/fazepurple/ros2_ws/bxi_rl_controller_ros2_example_dev/.worktrees/konodoki-dev
source /opt/ros/humble/setup.bash
source /home/fazepurple/ros2_ws/bxi_ros2_pkg/setup.bash
source install-live-normal/setup.bash
export ROS_DOMAIN_ID=42

ros2 topic pub --once /motion_commands communication/msg/MotionCommands '{}'
ros2 topic pub --once /motion_commands communication/msg/MotionCommands '{btn_3: 1}'
ros2 topic pub --once /motion_commands communication/msg/MotionCommands '{}'
sleep 3
ros2 topic pub --once /motion_commands communication/msg/MotionCommands '{btn_1: 1}'
ros2 topic pub --once /motion_commands communication/msg/MotionCommands '{}'
sleep 5
```

Expected: MuJoCo stands stably in Normal.

- [ ] **Step 4: Enter ZeroLab and prove ten seconds of live-Normal waiting**

With the operator holding T-pose:

```bash
ros2 topic pub --once /motion_commands communication/msg/MotionCommands '{btn_10: 11}'
ros2 topic pub --once /motion_commands communication/msg/MotionCommands '{}'
sleep 10
```

Terminal 1 must show:

```text
ZeroLab pre-ARM output: live zero-command Normal policy
ZeroLab ARM phase: WAIT_CALIBRATION
ZeroLab stream ready; frame=...
PICO source chunks ready; sent=...
ZeroLab ARM phase: WAIT_ARM
```

Expected: the simulated robot remains standing for the full ten seconds and
does not move toward T-pose. Any fall, joint-limit violation, unexpected
SONIC motion, or hardware process fails acceptance and ends this plan.

- [ ] **Step 5: Return neutral and verify the first live-Normal blend**

After the operator returns to a neutral pose and holds it:

```bash
ros2 topic pub --once /motion_commands communication/msg/MotionCommands '{btn_10: 12}'
ros2 topic pub --once /motion_commands communication/msg/MotionCommands '{}'
sleep 3
```

Expected logs identify a two-second live-Normal-to-SONIC blend followed by:

```text
ZeroLab ARM phase: ARMED
```

Then test only small left-arm, right-arm, bilateral, and arm-lowering motions.
Any wrong direction, rapid shoulder motion, loss of balance, or missing
`ARMED` fails acceptance.

- [ ] **Step 6: Verify stale hold and frozen-frame recovery**

Stop Windows UDP for more than `0.5 s`. Expected:

```text
ZeroLab reference stale; holding last motor frame and ARM cancelled
```

Restore UDP. Confirm no automatic motion. Return the operator to neutral and
send:

```bash
ros2 topic pub --once /motion_commands communication/msg/MotionCommands '{btn_10: 12}'
ros2 topic pub --once /motion_commands communication/msg/MotionCommands '{}'
sleep 3
```

Expected: log identifies a frozen-frame-to-SONIC recovery blend and reaches
`ARMED` again.

- [ ] **Step 7: Return to Normal, shut down, and collect timing evidence**

```bash
ros2 topic pub --once /motion_commands communication/msg/MotionCommands '{btn_1: 1}'
ros2 topic pub --once /motion_commands communication/msg/MotionCommands '{}'
sleep 3
```

Press `Ctrl-C` once in terminal 1. Then verify:

```bash
pgrep -af '[b]xi_example_py_elf3_demo|[m]od_node_runner.*zerolab' \
  || echo 'SIM_CLEANUP=PASS'
ss -H -lunp 'sport = :18000'
ss -H -ltnp 'sport = :5558'
ss -H -ltnp 'sport = :5557'
```

Review the terminal-1 performance summary and record total cycles, deadline
misses, skipped periods, budget overruns, P99/max execution time, stale events,
fall status, and joint-limit status. `SIM_CLEANUP=PASS` and released ports are
required. Do not prepare a robot patch or hardware commands under this plan.

---

### Task 8: Final Review of the Simulation Candidate

**Files:**
- Verify: all tracked changes since `155040f`
- Verify: design `docs/superpowers/specs/2026-08-19-zerolab-live-normal-arming-design.md`
- Verify: plan `docs/superpowers/plans/2026-08-19-zerolab-live-normal-arming.md`

**Interfaces:**
- Consumes: passing automated tests, isolated build, and MuJoCo evidence.
- Produces: a reviewed simulation-only candidate decision; hardware deployment remains a separate future task.

- [ ] **Step 1: Run final diff and repository checks**

```bash
git diff --check 155040f..HEAD
git diff --stat 155040f..HEAD
git log --oneline 155040f..HEAD
git status --short
```

Expected: no whitespace errors, focused commits, and no tracked uncommitted
changes. Build artifacts may remain untracked.

- [ ] **Step 2: Confirm every deployment gate explicitly**

Record PASS/FAIL for:

```text
Automated ZeroLab tests
Complete package tests
Isolated colcon build and installed factory
10-second WAIT_CALIBRATION/WAIT_ARM standing
Two-second live-Normal initial blend
ARMED small-motion direction
Stale hold without automatic recovery
Explicit frozen-frame re-ARM
No fall
No joint-limit violation
No hardware process
Clean process and port shutdown
```

Any FAIL leaves the branch incomplete and forbids hardware deployment.

- [ ] **Step 3: Request final code review**

Invoke `superpowers:requesting-code-review` against the full diff from
`155040f` to `HEAD`, including automated test output and MuJoCo evidence. Fix
all blocking findings with focused tests before claiming completion.
