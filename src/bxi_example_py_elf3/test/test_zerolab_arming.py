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
from zerolab.converter import ZeroLabMotionConverter
from zerolab.protocol import ZeroLabPacket
from zerolab.resampler import PlayoutKind, ZeroLabPoseResampler
from zerolab.source_node import ZeroLabSourceCore


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


policy_module = _load_module("policy", "policy.py")
base_state_module = _load_module("state", "state.py")
arming_module = _load_module("zerolab.state", "zerolab/state.py")
reference_gate_module = sys.modules[f"{PACKAGE}.reference_gate"]
SonicTeleopPolicy = policy_module.SonicTeleopPolicy
LiveReferenceGate = reference_gate_module.LiveReferenceGate
ReferenceGateMode = reference_gate_module.ReferenceGateMode
SmplReferenceFrame = reference_gate_module.SmplReferenceFrame
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
        self.recovery_ready = False
        self.step_calls = 0
        self.held = False
        self.rearming = False
        self.rearm_attempts = 0
        self.rearm_progress = 0.0
        self.held_rearm_progress = None
        self.rearm_progress_at_step = []
        self.head_joint_target = np.zeros(2, dtype=np.float32)
        self.last_status = "idle_reference"

    def bind_logger(self, _logger):
        pass

    def configure_runtime(self, **_kwargs):
        pass

    def reset(self, _frame=None):
        self.step_calls = 0
        self.rearm_progress_at_step.clear()

    def step(self, _frame, _dt, *, advance=True):
        if advance:
            self.step_calls += 1
            self.rearm_progress_at_step.append(self.rearm_progress)
        return self.output

    def has_fresh_live_reference(self, _timeout_s=None):
        return self.fresh

    def live_reference_recovery_ready(self, required_real_frames):
        assert required_real_frames == 10
        return self.fresh and self.recovery_ready

    def hold_live_reference(self):
        self.held_rearm_progress = self.rearm_progress
        self.held = True
        self.rearming = False
        return True

    def begin_live_reference_rearm(self):
        if not self.fresh or not self.held:
            return False
        self.rearming = True
        self.rearm_attempts += 1
        return True

    def set_live_reference_rearm_progress(self, alpha):
        self.rearm_progress = float(alpha)

    def complete_live_reference_rearm(self):
        self.rearming = False
        self.held = False

    def reset_yaw_alignment(self):
        pass


class SourceBackedPolicy(FakePolicy):
    def __init__(self):
        super().__init__()
        self._live_reference_gate = LiveReferenceGate()
        self.live_ref_timeout_s = 0.5
        self.source_now_ns = None

    def observe_source(self, fields, *, now_ns):
        assert fields is not None
        self.source_now_ns = int(now_ns)
        root_quat = np.zeros((10, 4), dtype=np.float32)
        root_quat[:, 0] = 1.0
        self._live_reference_gate.observe(
            SmplReferenceFrame(
                term1_local=np.zeros((10, 72), dtype=np.float32),
                root_quat=root_quat,
                wrist=np.zeros((10, 6), dtype=np.float32),
                head_joint_pos=np.zeros((10, 2), dtype=np.float32),
                frame_index=int(fields["frame_index"][-1]),
                newest_frame_index=int(fields["frame_index"][-1]),
                valid_horizon=10,
                clamp_slots=0,
                source_generation=int(fields["source_generation"][0]),
                latest_real_frame_index=int(
                    fields["latest_real_frame_index"][0]
                ),
                latest_real_receive_timestamp_ns=int(
                    fields["latest_real_receive_timestamp_ns"][0]
                ),
                real_valid_frames_in_generation=int(
                    fields["real_valid_frames_in_generation"][0]
                ),
                real_stream_ready=bool(fields["real_stream_ready"][0]),
                playout_kind=int(fields["playout_kind"][0]),
                source_stale=bool(fields["source_stale"][0]),
            ),
            received_mono=self.source_now_ns / 1.0e9,
        )

    @property
    def reference_mode(self):
        return self._live_reference_gate.mode

    def poll_reference(self):
        return self._live_reference_gate.observed_reference

    has_fresh_live_reference = SonicTeleopPolicy.has_fresh_live_reference
    live_reference_recovery_ready = (
        SonicTeleopPolicy.live_reference_recovery_ready
    )
    hold_live_reference = SonicTeleopPolicy.hold_live_reference
    begin_live_reference_rearm = SonicTeleopPolicy.begin_live_reference_rearm
    set_live_reference_rearm_progress = (
        SonicTeleopPolicy.set_live_reference_rearm_progress
    )
    complete_live_reference_rearm = (
        SonicTeleopPolicy.complete_live_reference_rearm
    )


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


class FakeHandle:
    status = "ready"

    def __init__(self, policy):
        self.policy = policy

    def get(self):
        return self.policy


class FakeInferenceFrame:
    def __init__(self, command):
        self.command = command


class FakeContext:
    def __init__(self, last_motor_frame):
        self.robot_layout = ELF3_POLICY_JOINTS
        self.last_motor_frame = last_motor_frame
        self.current_raw_cmd_vel = np.array([0.8, -0.4, 0.6], dtype=np.float32)
        self.current_cmd_vel = np.zeros(3, dtype=np.float32)
        self.speed_profiles = {}
        self.inference_frame = FakeInferenceFrame(self.current_cmd_vel)
        self.applied = None
        self.current_quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0])
        self.orientation_unsafe = False
        self.requested_states = []

    def is_orientation_unsafe(self, _quat):
        return self.orientation_unsafe

    def request_state(self, name, *, trigger):
        self.requested_states.append((name, trigger))
        return True

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


_UNSET = object()


def make_state(
    *,
    entry_value=0.0,
    arm_blend_seconds=2.0,
    sonic_policy=None,
    auto_rearm_on_recovery=_UNSET,
    auto_rearm_blend_seconds=_UNSET,
    recovery_real_frames=_UNSET,
):
    policy = FakePolicy() if sonic_policy is None else sonic_policy
    normal_policy = FakeNormalPolicy()
    entry = MotorFrame.create(
        ELF3_POLICY_JOINTS,
        np.full(ELF3_POLICY_JOINTS.dof_num, entry_value, dtype=np.float32),
        np.full(ELF3_POLICY_JOINTS.dof_num, 40.0, dtype=np.float32),
        np.full(ELF3_POLICY_JOINTS.dof_num, 2.0, dtype=np.float32),
        vel=np.full(ELF3_POLICY_JOINTS.dof_num, 0.4, dtype=np.float32),
        torque=np.full(ELF3_POLICY_JOINTS.dof_num, 0.8, dtype=np.float32),
    )
    ctx = FakeContext(entry)
    recovery_kwargs = {}
    for name, value in (
        ("auto_rearm_on_recovery", auto_rearm_on_recovery),
        ("auto_rearm_blend_seconds", auto_rearm_blend_seconds),
        ("recovery_real_frames", recovery_real_frames),
    ):
        if value is not _UNSET:
            recovery_kwargs[name] = value
    state = ZeroLabArmedTeleopState(
        "zerolab",
        1,
        FakeHandle(policy),
        normal_policy=FakeHandle(normal_policy),
        head_control_enabled=False,
        hardware_gripper=False,
        live_reference_timeout_s=0.5,
        arm_blend_seconds=arm_blend_seconds,
        **recovery_kwargs,
    )
    state._bind_logger(CaptureLogger())
    return state, policy, normal_policy, ctx


def prepared_state(
    *,
    entry_value=0.0,
    arm_blend_seconds=2.0,
    **recovery_kwargs,
):
    state, sonic, normal, ctx = make_state(
        entry_value=entry_value,
        arm_blend_seconds=arm_blend_seconds,
        **recovery_kwargs,
    )
    state.on_prepare(ctx, object())
    return state, sonic, normal, ctx


def armed_state_at_half_blend():
    state, sonic, normal, ctx = prepared_state(entry_value=0.0)
    sonic.fresh = True
    state.sample_running_frame(ctx, 0.02, advance=True)
    assert state.on_action(ctx, "arm_zerolab") is True
    sonic.target.position.fill(2.0)
    sonic.target.kp.fill(80.0)
    sonic.target.kd.fill(4.0)
    state.sample_running_frame(ctx, 1.0, advance=True)
    return state, sonic, normal, ctx


def fully_armed_state(**recovery_kwargs):
    state, sonic, normal, ctx = prepared_state(
        entry_value=0.0,
        **recovery_kwargs,
    )
    sonic.fresh = True
    state.sample_running_frame(ctx, 0.02, advance=True)
    assert state.on_action(ctx, "arm_zerolab") is True
    sonic.target.position.fill(2.0)
    state.sample_running_frame(ctx, 2.0, advance=True)
    assert state.arm_phase is ZeroLabArmPhase.ARMED
    return state, sonic, normal, ctx


def reference_hold_state(**recovery_kwargs):
    state, sonic, normal, ctx = fully_armed_state(**recovery_kwargs)
    sonic.fresh = False
    state.sample_running_frame(ctx, 0.02, advance=True)
    assert state.arm_phase is ZeroLabArmPhase.HOLD_REFERENCE
    return state, sonic, normal, ctx


def source_packet(index, *, timestamp_ns):
    quaternions = np.zeros((47, 4), dtype=np.float32)
    quaternions[:, 3] = 1.0
    return ZeroLabPacket(
        receive_timestamp_ns=timestamp_ns,
        local_frame_index=index,
        root_translation=np.zeros(3, dtype=np.float32),
        joint_quat_world_xyzw=quaternions,
        left_hand_values=np.zeros(6, dtype=np.uint16),
        right_hand_values=np.zeros(6, dtype=np.uint16),
        joint_position=np.zeros((17, 3), dtype=np.float32),
        raw_payload=bytes(992),
        sender_address=("127.0.0.1", 50000),
    )


def source_backed_armed_state(monkeypatch):
    generations = iter((101, 202))
    core = ZeroLabSourceCore(
        ZeroLabMotionConverter(),
        resampler=ZeroLabPoseResampler(
            jitter_buffer_seconds=0.08,
            short_recovery_blend_seconds=0.2,
            output_rate_hz=50.0,
        ),
        window_frames=10,
        stale_seconds=0.5,
        recovery_real_frames=10,
        generation_factory=lambda _previous=None: next(generations),
    )
    sonic = SourceBackedPolicy()
    monkeypatch.setattr(
        policy_module,
        "time",
        SimpleNamespace(
            monotonic=lambda: sonic.source_now_ns / 1.0e9,
            monotonic_ns=lambda: sonic.source_now_ns,
        ),
    )
    state, sonic, normal, ctx = prepared_state(sonic_policy=sonic)
    fields = None
    sample_now_ns = None
    for index in range(10):
        timestamp_ns = index * 20_000_000
        assert core.accept(source_packet(index, timestamp_ns=timestamp_ns))
        sample_now_ns = 80_000_000 + timestamp_ns
        fields = core.sample(sample_now_ns)
    assert fields is not None
    sonic.observe_source(fields, now_ns=sample_now_ns)
    state.sample_running_frame(ctx, 0.02, advance=True)
    assert state.arm_phase is ZeroLabArmPhase.WAIT_ARM
    assert state.on_action(ctx, "arm_zerolab") is True
    state.sample_running_frame(ctx, 2.0, advance=True)
    assert state.arm_phase is ZeroLabArmPhase.ARMED
    return state, sonic, normal, ctx, core


def accept_source_packet(
    core,
    sonic,
    index,
    timestamp_ns,
    *,
    sample_now_ns=None,
):
    assert core.accept(source_packet(index, timestamp_ns=timestamp_ns))
    now_ns = timestamp_ns if sample_now_ns is None else sample_now_ns
    fields = core.sample(now_ns)
    assert fields is not None
    sonic.observe_source(fields, now_ns=now_ns)
    return fields


def waiting_arm_state():
    state, sonic, normal, ctx = prepared_state(entry_value=0.0)
    sonic.fresh = True
    state.sample_running_frame(ctx, 0.02, advance=True)
    assert state.arm_phase is ZeroLabArmPhase.WAIT_ARM
    return state, sonic, normal, ctx


def auto_rearming_state():
    state, sonic, normal, ctx = reference_hold_state()
    sonic.fresh = True
    sonic.recovery_ready = True
    state.sample_running_frame(ctx, 0.02, advance=True)
    assert state.arm_phase is ZeroLabArmPhase.REARMING
    return state, sonic, normal, ctx


def rearming_state():
    return auto_rearming_state()


def state_in_phase(phase_name):
    if phase_name == "wait_stream":
        return prepared_state()
    if phase_name == "wait_arm":
        return waiting_arm_state()
    if phase_name == "blending":
        return armed_state_at_half_blend()
    if phase_name == "armed":
        return fully_armed_state()
    if phase_name == "hold_reference":
        return reference_hold_state()
    if phase_name == "rearming":
        return rearming_state()
    raise AssertionError(f"unexpected phase {phase_name}")


def copy_motor_frame(frame):
    return MotorFrame.create(
        frame.layout,
        frame.qpos,
        frame.kp,
        frame.kd,
        vel=frame.vel,
        torque=frame.torque,
    )


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


def test_arm_blend_seconds_must_be_finite_and_positive():
    for value in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="arm_blend_seconds"):
            make_state(arm_blend_seconds=value)


@pytest.mark.parametrize("value", [0, 1, 0.0, 1.0])
def test_auto_rearm_on_recovery_rejects_numeric_booleans(value):
    with pytest.raises(ValueError, match="auto_rearm_on_recovery"):
        make_state(auto_rearm_on_recovery=value)


@pytest.mark.parametrize(
    "value",
    [False, True, 0.0, -1.0, float("inf"), float("nan")],
)
def test_auto_rearm_blend_seconds_must_be_finite_and_positive(value):
    with pytest.raises(ValueError, match="auto_rearm_blend_seconds"):
        make_state(auto_rearm_blend_seconds=value)


@pytest.mark.parametrize("value", [False, True, 0, -1, 1.0, 1.5, "10", None])
def test_recovery_real_frames_must_be_a_positive_integer(value):
    with pytest.raises(ValueError, match="recovery_real_frames"):
        make_state(recovery_real_frames=value)


def test_waiting_applies_changing_live_normal_with_zero_command():
    state, sonic, normal, ctx = prepared_state(entry_value=0.25)
    normal.target.position.fill(0.4)

    first = copy_motor_frame(state.sample_running_frame(ctx, 0.02, advance=True))
    normal.target.position.fill(0.7)
    second = copy_motor_frame(state.sample_running_frame(ctx, 0.02, advance=True))

    assert state.arm_phase is ZeroLabArmPhase.WAIT_STREAM
    np.testing.assert_allclose(first.qpos, 0.4)
    np.testing.assert_allclose(second.qpos, 0.7)
    np.testing.assert_allclose(ctx.inference_frame.command, 0.0)
    assert sonic.step_calls == 2
    assert normal.step_calls == 2
    np.testing.assert_allclose(state.get_entry_frame(ctx).qpos, 0.25)


def test_prepare_logs_initial_wait_stream_phase_once():
    state, _sonic, _normal, ctx = prepared_state(entry_value=0.0)
    state.on_prepare(ctx, object())

    messages = [
        message for _level, message in state.logger.messages
        if "ZeroLab ARM phase: WAIT_STREAM" in message
    ]
    assert messages == ["ZeroLab ARM phase: WAIT_STREAM"]


def test_fresh_reference_waits_for_explicit_arm():
    state, sonic, normal, ctx = prepared_state(entry_value=0.25)
    sonic.fresh = True
    sonic.target.position.fill(1.0)

    frame = state.sample_running_frame(ctx, 0.02, advance=True)

    assert state.arm_phase is ZeroLabArmPhase.WAIT_ARM
    np.testing.assert_allclose(frame.qpos, 0.0)


def test_initial_stream_never_auto_arms_without_btn10_12():
    state, sonic, _normal, ctx = prepared_state()
    sonic.fresh = True
    sonic.recovery_ready = True

    for _ in range(200):
        state.sample_running_frame(ctx, 0.02, advance=True)

    assert state.arm_phase is ZeroLabArmPhase.WAIT_ARM


def test_arm_is_rejected_until_fresh_reference_exists():
    state, sonic, normal, ctx = prepared_state(entry_value=0.0)
    assert state.on_action(ctx, "arm_zerolab") is True
    assert state.arm_phase is ZeroLabArmPhase.WAIT_STREAM


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


@pytest.mark.parametrize("phase_name", ["blending", "armed", "rearming"])
def test_duplicate_arm_does_not_restart_active_phase(phase_name):
    state, sonic, normal, ctx = state_in_phase(phase_name)
    elapsed_before = state._blend_elapsed_s
    progress_before = sonic.rearm_progress

    assert state.on_action(ctx, "arm_zerolab") is True

    assert state.arm_phase.value == phase_name
    assert state._blend_elapsed_s == elapsed_before
    assert sonic.rearm_progress == progress_before


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

    assert sonic.step_calls == calls + 2
    np.testing.assert_allclose(first.qpos, 3.0)
    np.testing.assert_allclose(second.qpos, 4.0)
    assert state.arm_phase is ZeroLabArmPhase.HOLD_REFERENCE
    assert sonic.held is True


def test_previously_armed_hold_auto_rearms_after_ten_real_frames():
    state, sonic, _normal, ctx = reference_hold_state()
    sonic.fresh = True
    sonic.recovery_ready = False
    state.sample_running_frame(ctx, 0.02, advance=True)
    assert state.arm_phase is ZeroLabArmPhase.HOLD_REFERENCE
    assert sonic.rearming is False

    sonic.recovery_ready = True
    state.sample_running_frame(ctx, 0.02, advance=True)
    assert state.arm_phase is ZeroLabArmPhase.REARMING
    assert sonic.rearming is True
    assert sonic.rearm_attempts == 1


@pytest.mark.parametrize("gap_s", [0.10, 0.49])
def test_short_gap_remains_armed_and_uses_short_recovery(gap_s, monkeypatch):
    state, sonic, _normal, ctx, core = source_backed_armed_state(monkeypatch)
    last_real_ns = core.latest_real_receive_timestamp_ns
    gap_now_ns = last_real_ns + int(gap_s * 1.0e9)

    assert core.check_stale(gap_now_ns) is False
    held_fields = core.sample(gap_now_ns)
    assert held_fields is not None
    assert int(held_fields["playout_kind"][0]) == int(PlayoutKind.HELD)
    sonic.observe_source(held_fields, now_ns=gap_now_ns)
    state.sample_running_frame(ctx, 0.02, advance=True)

    assert state.arm_phase is ZeroLabArmPhase.ARMED
    assert state.auto_rearm_count == 0
    assert sonic.reference_mode is ReferenceGateMode.LIVE

    recovered_fields = accept_source_packet(
        core,
        sonic,
        10,
        gap_now_ns,
        sample_now_ns=gap_now_ns + 20_000_000,
    )
    assert int(recovered_fields["playout_kind"][0]) == int(
        PlayoutKind.SHORT_RECOVERY_BLEND
    )
    state.sample_running_frame(ctx, 0.02, advance=True)
    assert state.arm_phase is ZeroLabArmPhase.ARMED
    assert state.auto_rearm_count == 0
    assert sonic.reference_mode is ReferenceGateMode.LIVE


@pytest.mark.parametrize("gap_s", [0.51, 2.0, 30.0])
def test_long_gap_holds_until_tenth_real_packet_then_auto_rearms(
    gap_s, monkeypatch
):
    state, sonic, _normal, ctx, core = source_backed_armed_state(monkeypatch)
    last_real_ns = core.latest_real_receive_timestamp_ns
    gap_now_ns = last_real_ns + int(gap_s * 1.0e9)

    assert core.check_stale(gap_now_ns) is True
    stale_fields = core.sample(gap_now_ns)
    assert stale_fields is not None
    sonic.observe_source(stale_fields, now_ns=gap_now_ns)
    state.sample_running_frame(ctx, 0.02, advance=True)

    assert core.real_valid_frames_in_generation == 0
    assert state.arm_phase is ZeroLabArmPhase.HOLD_REFERENCE
    assert state.auto_rearm_count == 0
    assert sonic.reference_mode is ReferenceGateMode.HOLD

    packet_index = 10
    packet_timestamp_ns = gap_now_ns
    for real_frames in range(1, 10):
        packet_timestamp_ns += 20_000_000
        fields = accept_source_packet(
            core,
            sonic,
            packet_index,
            packet_timestamp_ns,
        )
        packet_index += 1
        state.sample_running_frame(ctx, 0.02, advance=True)
        assert int(fields["real_valid_frames_in_generation"][0]) == real_frames
        assert int(fields["real_stream_ready"][0]) == 0
        assert state.arm_phase is ZeroLabArmPhase.HOLD_REFERENCE
        assert state.auto_rearm_count == 0
        assert sonic.reference_mode is ReferenceGateMode.HOLD

    packet_timestamp_ns += 20_000_000
    tenth_fields = accept_source_packet(
        core,
        sonic,
        packet_index,
        packet_timestamp_ns,
    )
    packet_index += 1
    state.sample_running_frame(ctx, 0.02, advance=True)

    assert int(tenth_fields["real_valid_frames_in_generation"][0]) == 10
    assert int(tenth_fields["real_stream_ready"][0]) == 1
    assert state.arm_phase is ZeroLabArmPhase.REARMING
    assert state.auto_rearm_count == 1
    assert sonic.reference_mode is ReferenceGateMode.REARMING

    for _ in range(99):
        packet_timestamp_ns += 20_000_000
        accept_source_packet(
            core,
            sonic,
            packet_index,
            packet_timestamp_ns,
        )
        packet_index += 1
        state.sample_running_frame(ctx, 0.02, advance=True)
    assert state.arm_phase is ZeroLabArmPhase.REARMING

    packet_timestamp_ns += 20_000_000
    accept_source_packet(
        core,
        sonic,
        packet_index,
        packet_timestamp_ns,
    )
    state.sample_running_frame(ctx, 0.02, advance=True)
    assert state.arm_phase is ZeroLabArmPhase.ARMED
    assert state.auto_rearm_count == 1
    assert sonic.reference_mode is ReferenceGateMode.LIVE


def test_auto_rearm_count_increments_only_after_gate_rearm_succeeds():
    state, sonic, _normal, ctx = reference_hold_state()
    sonic.fresh = True
    sonic.recovery_ready = True
    sonic.held = False

    state.sample_running_frame(ctx, 0.02, advance=True)

    assert state.arm_phase is ZeroLabArmPhase.HOLD_REFERENCE
    assert state.auto_rearm_count == 0
    assert sonic.rearm_attempts == 0

    sonic.held = True
    state.sample_running_frame(ctx, 0.02, advance=True)

    assert state.arm_phase is ZeroLabArmPhase.REARMING
    assert state.auto_rearm_count == 1
    assert sonic.rearm_attempts == 1


def test_explicit_recovery_remains_available_when_auto_rearm_is_disabled():
    state, sonic, _normal, ctx = reference_hold_state(
        auto_rearm_on_recovery=False
    )
    sonic.fresh = True
    sonic.recovery_ready = True

    state.sample_running_frame(ctx, 0.02, advance=True)
    assert state.arm_phase is ZeroLabArmPhase.HOLD_REFERENCE

    assert state.on_action(ctx, "arm_zerolab") is True
    assert state.arm_phase is ZeroLabArmPhase.REARMING
    assert sonic.rearm_attempts == 1


def test_wait_arm_stale_returns_to_wait_stream_with_live_normal():
    state, sonic, normal, ctx = waiting_arm_state()
    sonic.fresh = False
    normal.target.position.fill(0.6)

    frame = state.sample_running_frame(ctx, 1.0, advance=True)

    assert state.arm_phase is ZeroLabArmPhase.WAIT_STREAM
    np.testing.assert_allclose(frame.qpos, 0.6)
    assert normal.step_calls >= 2


def test_rearm_completes_after_two_seconds_with_live_sonic_output():
    state, sonic, normal, ctx = rearming_state()
    sonic.target.position.fill(3.0)
    calls = sonic.step_calls
    progress_samples = len(sonic.rearm_progress_at_step)

    first = copy_motor_frame(state.sample_running_frame(ctx, 1.0, advance=True))
    sonic.target.position.fill(4.0)
    second = copy_motor_frame(state.sample_running_frame(ctx, 1.0, advance=True))

    np.testing.assert_allclose(first.qpos, 3.0)
    np.testing.assert_allclose(second.qpos, 4.0)
    assert sonic.step_calls == calls + 2
    assert sonic.rearm_progress_at_step[progress_samples:] == [0.5, 1.0]
    assert sonic.rearm_progress == pytest.approx(1.0)
    assert sonic.rearming is False
    assert sonic.held is False
    assert state.arm_phase is ZeroLabArmPhase.ARMED


def test_auto_rearm_uses_separate_two_second_duration():
    state, sonic, _normal, ctx = auto_rearming_state()
    state.arm_blend_seconds = 9.0

    state.sample_running_frame(ctx, 1.99, advance=True)
    assert state.arm_phase is ZeroLabArmPhase.REARMING
    state.sample_running_frame(ctx, 0.01, advance=True)
    assert state.arm_phase is ZeroLabArmPhase.ARMED


def test_stale_during_rearm_returns_to_reference_hold():
    state, sonic, normal, ctx = rearming_state()
    state.sample_running_frame(ctx, 0.5, advance=True)
    sonic.fresh = False
    sonic.target.position.fill(5.0)
    calls = sonic.step_calls

    frame = state.sample_running_frame(ctx, 0.1, advance=True)

    np.testing.assert_allclose(frame.qpos, 5.0)
    assert sonic.step_calls == calls + 1
    assert sonic.held is True
    assert sonic.rearming is False
    assert sonic.held_rearm_progress == pytest.approx(0.216)
    assert state.arm_phase is ZeroLabArmPhase.HOLD_REFERENCE


def test_stale_during_rearm_holds_then_retries_after_new_ready_generation():
    state, sonic, _normal, ctx = auto_rearming_state()
    sonic.fresh = False
    sonic.recovery_ready = False

    state.sample_running_frame(ctx, 0.1, advance=True)

    assert state.arm_phase is ZeroLabArmPhase.HOLD_REFERENCE
    assert sonic.held_rearm_progress == pytest.approx(0.00725)
    sonic.fresh = True
    sonic.recovery_ready = True
    state.sample_running_frame(ctx, 0.02, advance=True)
    assert state.arm_phase is ZeroLabArmPhase.REARMING
    assert state.auto_rearm_count == 2
    assert sonic.rearm_attempts == 2


def test_btn10_12_does_not_restart_hold_rearming_or_armed_when_auto_enabled():
    for phase in ("hold_reference", "rearming", "armed"):
        state, sonic, _normal, ctx = state_in_phase(phase)
        before_phase = state.arm_phase
        before_elapsed = state._blend_elapsed_s
        before_attempts = sonic.rearm_attempts
        before_messages = len(state.logger.messages)

        assert state.on_action(ctx, "arm_zerolab") is True

        assert state.arm_phase is before_phase
        assert state._blend_elapsed_s == before_elapsed
        assert sonic.rearm_attempts == before_attempts
        assert state.logger.messages[before_messages:] == [
            (
                "info",
                "ZeroLab ARM ignored; automatic recovery owns "
                "post-ARM transitions",
            )
        ]


@pytest.mark.parametrize(
    "phase_name",
    ["wait_stream", "wait_arm", "blending"],
)
def test_live_normal_phases_keep_normal_orientation_safety(phase_name):
    state, sonic, normal, ctx = state_in_phase(phase_name)
    ctx.orientation_unsafe = True

    state.on_update(ctx, 0.02)

    assert ctx.requested_states == [
        ("com.bxi.basic_actions/zero_torque", "safety")
    ]
    assert ctx.applied is None


@pytest.mark.parametrize(
    "phase_name",
    ["armed", "hold_reference", "rearming"],
)
def test_live_sonic_phases_keep_existing_sonic_orientation_behavior(phase_name):
    state, sonic, normal, ctx = state_in_phase(phase_name)
    ctx.orientation_unsafe = True

    state.on_update(ctx, 0.02)

    assert ctx.requested_states == []
    assert ctx.applied is not None


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
def test_emergency_routes_remain_unconsumed_in_every_phase(phase_name):
    state, sonic, normal, ctx = state_in_phase(phase_name)
    phase_before = state.arm_phase

    assert state.on_action(ctx, "com.bxi.basic_actions/zero_torque") is False
    assert state.arm_phase is phase_before


def test_exit_clears_arm_session_and_next_prepare_waits_for_stream():
    state, sonic, normal, ctx = fully_armed_state()
    state.on_exit(ctx)
    assert state.arm_phase is ZeroLabArmPhase.WAIT_STREAM

    sonic.fresh = False
    state.on_prepare(ctx, object())
    assert state.arm_phase is ZeroLabArmPhase.WAIT_STREAM


def test_stale_and_recovery_logs_are_emitted_once_per_stale_session():
    state, sonic, normal, ctx = fully_armed_state()
    logger = state.logger
    sonic.fresh = False
    for _ in range(100):
        state.sample_running_frame(ctx, 0.02, advance=True)
    sonic.fresh = True
    sonic.recovery_ready = True
    for _ in range(101):
        state.sample_running_frame(ctx, 0.02, advance=True)

    stale_messages = [
        message for _, message in logger.messages
        if "ZeroLab reference stale; holding human reference while SONIC balance continues"
        in message
    ]
    rearming_messages = [
        message for _, message in logger.messages
        if message.startswith(
            "ZeroLab automatic recovery; ARM phase: REARMING"
        )
    ]
    completion_messages = [
        message for _, message in logger.messages
        if message == "ZeroLab automatic recovery complete; ARM phase: ARMED"
    ]
    assert len(stale_messages) == 1
    assert rearming_messages == [
        "ZeroLab automatic recovery; ARM phase: REARMING for 2.000 s"
    ]
    assert completion_messages == [
        "ZeroLab automatic recovery complete; ARM phase: ARMED"
    ]
