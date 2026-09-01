# ZeroLab Applied Action History Synchronization Design

**Date:** 2026-08-26
**Status:** Approved design
**Scope:** ZeroLab-only correction of SONIC action history before and during
the initial ARM transition on ELF3 simulation and real hardware

## Relationship to Existing ZeroLab Behavior

This design is a narrowly scoped correction to the initial ARM behavior in
`2026-08-24-zerolab-wireless-jitter-auto-recovery-design.md`. All input
resampling, freshness, stale hold, automatic recovery, safety routes, and
configuration defined there remain unchanged.

The following behavior also remains unchanged:

- Entering ZeroLab requires `btn_10=11`.
- Initial SONIC activation requires one explicit `btn_10=12`.
- The initial Normal-to-SONIC blend remains two seconds.
- SONIC continues closed-loop inference during an armed stale-reference hold.
- SONIC weights, gains, action scale, limits, yaw bias, pose conversion, PICO,
  and motor safety routing do not change.

## Problem and Evidence

`ZeroLabArmedTeleopState` currently runs SONIC inference every control tick in
`WAIT_STREAM` and `WAIT_ARM`, while the state sends the live Normal policy
frame to the motors. `SonicTeleopPolicy.inference_step()` stores its generated
target in `last_action`, and the following tick appends that value to the
ten-frame `action_history`. The model therefore receives a history claiming
that previous SONIC targets were applied even though the motors executed
Normal targets.

The same mismatch continues during `BLENDING`: the model records the complete
SONIC candidate while the motors execute only the current Normal-to-SONIC
mixture. Pressing ARM gradually exposes a policy state conditioned on actions
the robot never executed.

A read-only diagnostic used the production converter, the captured neutral
ZeroLab packets, the new robot's measured 29-joint state and IMU, and the real
`sonic.onnx` model. With the same stationary ZeroLab reference:

```text
history effect on lower-body target L2:       2.7423 rad
largest single-joint target difference:       2.0364 rad (right ankle)
waist-z target difference:                   -0.1476 rad (-8.46 degrees)
```

The effect remained present across four different captured ten-frame windows
and simulated pre-ARM waits of 10, 25, 50, 100, and 250 inference ticks. In
contrast, changing only the neutral reference from the official idle sample
to the captured ZeroLab sample changed the waist-z target by approximately
`-0.0072 rad`. The evidence identifies the action-history mismatch as the
primary cause candidate for the observed fixed rotation after ARM. The
converted neutral skeleton asymmetry remains a secondary input-quality issue
and is outside this correction.

## Goals

1. Make the action history consumed by SONIC describe the joint-position
   target actually applied in the preceding motor cycle.
2. Keep the causal order used during training: previous applied command,
   current proprioception and reference, then current inference.
3. Cover `WAIT_STREAM`, `WAIT_ARM`, `BLENDING`, `ARMED`, `HOLD_REFERENCE`, and
   `REARMING` without changing their transition semantics.
4. Include commands produced by live Normal, the initial blend, SONIC, and any
   framework command resolution visible through the previous applied motor
   frame.
5. Preserve fixed-rate inference and the ten-frame observation history.

## Non-Goals

- Do not change `yaw_bias_rad` or add a new heading calibration.
- Do not change ZeroLab-to-SMPL conversion or neutral-pose calibration.
- Do not disable SONIC inference while waiting for ARM.
- Do not reset every observation history when ARM is pressed.
- Do not change reference-space stale recovery or automatic rearming.
- Do not change Normal, PICO, the ONNX model, joint gains, action scale, or
  hardware configuration.
- Do not claim that this correction repairs wireless packet timing.

## Chosen Design

### Source of truth

For inference tick `N`, `ctx.last_motor_frame` is the source of truth for the
joint-position command applied on tick `N-1`. ZeroLab maps that frame from its
named source layout into the 29-joint `ELF3_POLICY_JOINTS` order before SONIC
advances.

Using the previous applied frame is preferable to copying the state-local
candidate because it preserves causal order and includes the frame selected
by Normal, the ARM blend, or SONIC. It also observes framework resolution that
has already become the controller's previous motor frame.

### Policy contract

`SonicTeleopPolicy` gains a ZeroLab-used method that accepts one finite
29-joint applied position target in policy order. It converts the target to
the existing normalized model action coordinates:

```text
applied_action = (applied_qpos - default_dof_pos) / action_scale
```

The result is clipped to the existing `[-ACTION_CLIP, ACTION_CLIP]` model
action domain and stored in `last_action`. Shape and finiteness violations are
hard errors; they must not silently poison model history.

This method changes only the value that `_update_history()` will append on the
next inference build. It does not run inference, change `target_dof_pos`,
publish a motor command, or alter any other history channel.

### State sequencing

On every advancing ZeroLab control tick:

```text
previous ctx.last_motor_frame
  -> named mapping into ELF3_POLICY_JOINTS
  -> policy records previous applied target as last_action
  -> SONIC builds its observation and appends that action
  -> SONIC produces the current candidate
  -> ZeroLab selects Normal, blend, SONIC, hold, or rearm output
  -> framework applies the selected motor frame
```

The mapping and reusable 29-joint buffer are prepared once per ZeroLab state
lifecycle. Mapping requires every SONIC policy joint but permits unrelated
extra robot joints. A missing required joint fails state preparation rather
than applying an incorrectly ordered history.

Non-advancing scheduler calls neither record another applied action nor shift
history. State exit clears the lifecycle-owned mapping and buffer along with
the existing ZeroLab lifecycle state.

### Phase behavior

- `WAIT_STREAM` and `WAIT_ARM`: history records the preceding live Normal
  frame, while SONIC may continue preparing its current candidate.
- `BLENDING`: history records the preceding smoothstep blend, not the full
  unexecuted SONIC candidate.
- `ARMED`: history records the preceding applied SONIC frame, preserving the
  intended closed loop.
- `HOLD_REFERENCE`: SONIC still runs with current proprioception and records
  the preceding applied SONIC frame; only the human reference remains held.
- `REARMING`: reference-space rearm behavior is unchanged and action history
  records the preceding applied SONIC frame.

No button or phase semantics change.

## Alternatives Rejected

### Stop SONIC inference until ARM

This avoids hidden policy actions but enters ARM with empty or stale
proprioception histories and introduces a first-inference latency/discontinuity.
It also changes the existing readiness behavior more broadly than necessary.

### Reset action history when `btn_10=12` arrives

This removes accumulated hidden targets but replaces them with zeros, which
still do not describe the preceding Normal commands. It also leaves the same
mismatch throughout the two-second blend.

### Change yaw bias or subtract the observed rotation

The diagnostic shows a much larger history effect than reference-only waist
yaw effect. A fixed angle correction would mask one captured posture and
could make other operators, robots, or headings worse.

## Error and Safety Handling

- Applied targets must be exactly 29 finite values after named mapping.
- Normalization must remain finite, and action scale remains the established
  nonzero SONIC parameter set.
- Mapping errors fail before ARM and follow existing framework exception
  handling; they must not fall back to index-based copying.
- PD Brake, zero torque, Normal, orientation safety, emergency transitions,
  policy exceptions, CAN faults, and `motor_timeout` retain priority.
- No new ROS topic, service, motor publisher, or operator button is added.

## Validation Strategy

### Unit and component tests

Tests cover:

1. The policy converts a finite applied qpos target into normalized,
   action-clipped `last_action` without changing its current output target.
2. Wrong shape and non-finite applied targets are rejected.
3. A reordered source joint layout maps by joint name, not numeric position.
4. Extra non-SONIC joints are permitted; a missing SONIC joint is rejected.
5. Before each advancing SONIC step, ZeroLab supplies the preceding
   `ctx.last_motor_frame` target.
6. `WAIT_STREAM` and `WAIT_ARM` record Normal, not the hidden SONIC candidate.
7. Consecutive `BLENDING` ticks record the preceding applied blend.
8. `ARMED`, `HOLD_REFERENCE`, and `REARMING` preserve closed-loop applied
   SONIC action history.
9. A non-advancing tick does not shift or duplicate action history.
10. Existing initial ARM, automatic recovery, emergency-preemption, ZeroLab,
    SONIC, and PICO regression suites remain green.

### Real ONNX regression

The captured neutral input and measured robot state are replayed read-only
through the real ONNX backend. The corrected pre-ARM replay must build the
same action-history channel as the applied Normal targets and must not build
the previously observed hidden-policy history. Diagnostic artifacts remain
test evidence and are not packaged as runtime dependencies.

### MuJoCo gate

With operator reference held neutral and no commanded locomotion:

1. Enter ZeroLab and wait at least five seconds before ARM.
2. Record Normal output, applied action history, SONIC candidate, and applied
   blend through the complete two-second transition.
3. Repeat with 0.2-second and 5-second pre-ARM waits.
4. Confirm every action-history sample matches the preceding applied motor
   target in policy coordinates.
5. Confirm the ARM transition does not reproduce the previous fixed waist/leg
   rotation and does not introduce a discontinuous motor target.
6. Confirm stale hold and automatic recovery behavior remain unchanged.

Any action-history mismatch, unexpected whole-body rotation, safety route
failure, inference exception, or regression test failure blocks hardware use.

### Real-hardware gate

Hardware retest is allowed only after all automated and MuJoCo gates pass. It
uses a support rig or equivalent fall protection, a working physical emergency
stop, and a dedicated safety observer. The operator and robot begin parallel,
facing the same direction, with the operator neutral.

The sequence is PD Brake verification, ZeroLab entry, neutral WAIT_ARM hold,
one explicit ARM, and observation through the two-second blend. Testing stops
immediately on unexpected yaw, asymmetric stepping, loss of balance, safety
route failure, policy exception, or hardware fault. Dropout and automatic
recovery tests remain blocked until this initial neutral ARM gate passes.

## Acceptance Criteria

The correction is accepted only when:

- Every advancing ZeroLab inference consumes the previous applied motor target
  in its action-history channel.
- Initial waiting and blending never record an unexecuted full SONIC target.
- Named joint mapping prevents order-dependent action corruption.
- Non-advancing ticks do not mutate observation history.
- Initial ARM still requires `btn_10=12` and still blends for two seconds.
- Stale hold, automatic recovery, safety transitions, Normal, and PICO are
  unchanged.
- Unit, component, real-ONNX replay, and MuJoCo gates pass before hardware use.
- Neutral supported hardware ARM no longer reproduces the observed fixed
  rotation before dropout testing resumes.
