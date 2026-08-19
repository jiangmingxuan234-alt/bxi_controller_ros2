from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from bxi_example_py_elf3.framework.inference import PolicyOutput
from bxi_example_py_elf3.framework.joints import JointTargetBuffer
from bxi_example_py_elf3.framework.mod_api.transition import MotorFrame
from bxi_example_py_elf3.policies.joints import ELF3_POLICY_JOINTS


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


class CaptureLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(("info", message))

    def warning(self, message):
        self.messages.append(("warning", message))

    def error(self, message):
        self.messages.append(("error", message))


def make_state(*, entry_value=0.0, arm_blend_seconds=2.0):
    policy = FakePolicy()
    entry = MotorFrame.create(
        ELF3_POLICY_JOINTS,
        np.full(ELF3_POLICY_JOINTS.dof_num, entry_value, dtype=np.float32),
        np.full(ELF3_POLICY_JOINTS.dof_num, 40.0, dtype=np.float32),
        np.full(ELF3_POLICY_JOINTS.dof_num, 2.0, dtype=np.float32),
        vel=np.full(ELF3_POLICY_JOINTS.dof_num, 0.4, dtype=np.float32),
        torque=np.full(ELF3_POLICY_JOINTS.dof_num, 0.8, dtype=np.float32),
    )
    ctx = FakeContext(entry)
    state = ZeroLabArmedTeleopState(
        "zerolab",
        1,
        FakeHandle(policy),
        head_control_enabled=False,
        hardware_gripper=False,
        live_reference_timeout_s=0.5,
        arm_blend_seconds=arm_blend_seconds,
    )
    state._bind_logger(CaptureLogger())
    return state, policy, ctx


def prepared_state(*, entry_value=0.0, arm_blend_seconds=2.0):
    state, policy, ctx = make_state(
        entry_value=entry_value,
        arm_blend_seconds=arm_blend_seconds,
    )
    state.on_prepare(ctx, object())
    return state, policy, ctx


def armed_state_at_half_blend():
    state, policy, ctx = prepared_state(entry_value=0.0)
    policy.fresh = True
    state.sample_running_frame(ctx, 0.02, advance=True)
    assert state.on_action(ctx, "arm_zerolab") is True
    policy.target.position.fill(2.0)
    policy.target.kp.fill(80.0)
    policy.target.kd.fill(4.0)
    state.sample_running_frame(ctx, 1.0, advance=True)
    return state, policy, ctx


def fully_armed_state():
    state, policy, ctx = prepared_state(entry_value=0.0)
    policy.fresh = True
    state.sample_running_frame(ctx, 0.02, advance=True)
    assert state.on_action(ctx, "arm_zerolab") is True
    policy.target.position.fill(2.0)
    state.sample_running_frame(ctx, 2.0, advance=True)
    assert state.arm_phase is ZeroLabArmPhase.ARMED
    return state, policy, ctx


def state_in_active_phase(phase):
    state, policy, ctx = fully_armed_state()
    if phase is ZeroLabArmPhase.BLENDING:
        state, policy, ctx = armed_state_at_half_blend()
    return state, policy, ctx


def stale_hold_state():
    state, policy, ctx = fully_armed_state()
    policy.fresh = False
    state.sample_running_frame(ctx, 0.02, advance=True)
    assert state.arm_phase is ZeroLabArmPhase.HOLD_STALE
    return state, policy, ctx


def waiting_arm_state():
    state, policy, ctx = prepared_state(entry_value=0.0)
    policy.fresh = True
    state.sample_running_frame(ctx, 0.02, advance=True)
    assert state.arm_phase is ZeroLabArmPhase.WAIT_ARM
    return state, policy, ctx


def copy_motor_frame(frame):
    return MotorFrame.create(
        frame.layout,
        frame.qpos,
        frame.kp,
        frame.kd,
        vel=frame.vel,
        torque=frame.torque,
    )


def assert_motor_frames_equal(actual, expected):
    np.testing.assert_allclose(actual.qpos, expected.qpos)
    np.testing.assert_allclose(actual.kp, expected.kp)
    np.testing.assert_allclose(actual.kd, expected.kd)
    np.testing.assert_allclose(actual.vel, expected.vel)
    np.testing.assert_allclose(actual.torque, expected.torque)


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


def test_prepare_logs_initial_wait_calibration_phase_once():
    state, _policy, ctx = prepared_state(entry_value=0.0)
    state.on_prepare(ctx, object())

    messages = [
        message for _level, message in state.logger.messages
        if "ZeroLab ARM phase: WAIT_CALIBRATION" in message
    ]
    assert messages == ["ZeroLab ARM phase: WAIT_CALIBRATION"]


def test_fresh_reference_waits_for_explicit_arm():
    state, policy, ctx = prepared_state(entry_value=0.25)
    policy.fresh = True
    policy.target.position.fill(1.0)

    frame = state.sample_running_frame(ctx, 0.02, advance=True)

    assert state.arm_phase is ZeroLabArmPhase.WAIT_ARM
    np.testing.assert_allclose(frame.qpos, 0.25)


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
    first = copy_motor_frame(state.sample_running_frame(ctx, 0.0, advance=True))
    middle = copy_motor_frame(state.sample_running_frame(ctx, 1.0, advance=True))
    final = copy_motor_frame(state.sample_running_frame(ctx, 1.0, advance=True))

    np.testing.assert_allclose(first.qpos, 0.0)
    np.testing.assert_allclose(middle.qpos, 1.0)
    np.testing.assert_allclose(final.qpos, 2.0)
    np.testing.assert_allclose(middle.kp, 60.0)
    np.testing.assert_allclose(middle.kd, 3.0)
    np.testing.assert_allclose(middle.vel, 0.2)
    np.testing.assert_allclose(middle.torque, 0.4)
    assert state.arm_phase is ZeroLabArmPhase.ARMED


def test_duplicate_arm_does_not_restart_blend():
    state, policy, ctx = armed_state_at_half_blend()
    before = state._blend_elapsed_s
    assert state.on_action(ctx, "arm_zerolab") is True
    assert state._blend_elapsed_s == before
    assert state.arm_phase is ZeroLabArmPhase.BLENDING


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


def test_stale_and_recovery_logs_are_emitted_once_per_stale_session():
    state, policy, ctx = fully_armed_state()
    logger = state.logger
    policy.fresh = False
    for _ in range(100):
        state.sample_running_frame(ctx, 0.02, advance=True)
    policy.fresh = True
    for _ in range(100):
        state.sample_running_frame(ctx, 0.02, advance=True)

    stale_messages = [
        message for _, message in logger.messages
        if "ZeroLab reference stale; holding last motor frame and ARM cancelled" in message
    ]
    recovery_messages = [
        message for _, message in logger.messages
        if "ZeroLab reference recovered; send btn_10=12 to resume" in message
    ]
    assert len(stale_messages) == 1
    assert len(recovery_messages) == 1
