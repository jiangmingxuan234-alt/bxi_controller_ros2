# ZeroLab Vendor-Calibrated Stream and Closed-Loop Stale Hold Design

**Date:** 2026-08-20
**Status:** Approved design
**Scope:** ZeroLab input conversion, readiness/arming semantics, and stale-input
behavior for `com.bxi.sonic/sonic_zerolab`

## Relationship to Earlier Designs

This design corrects two assumptions in the earlier ZeroLab safe-arming work:

1. `2026-08-19-zerolab-safe-arming-design.md` treated a runtime human T-pose as
   an application-level calibration input.
2. `2026-08-19-zerolab-live-normal-arming-design.md` froze the last complete
   motor frame after an armed reference became stale.

Those behaviors are superseded for the ZeroLab path by this document. The
existing explicit ARM action, live-Normal waiting output, two-second initial
handover, emergency routes, and PICO path remain in force unless this document
explicitly changes them.

## Confirmed Vendor Contract

The ZeroLab vendor confirmed the following interface contract:

- The operator completes the standard N-pose calibration in the vendor
  software before starting this application path.
- UDP `joint_quat_world` values are already the final calibrated skeleton
  world orientations.
- The consumer may directly perform the Unity/XRT coordinate conversion,
  fixed skeleton mapping, and parent-relative localization.
- The consumer does not need to sample another operator T-pose to locate the
  sensors.

The current runtime T-pose calibration violates this contract. It samples a
second rest orientation and applies `raw * sampled_rest^-1`, which changes the
meaning of the already calibrated world orientations. The observed shoulder
offset, raised arms, and inability to lower the arms are consistent with that
duplicate rest removal.

## Goals

1. Consume vendor-calibrated world orientations without a second human rest
   calibration.
2. Preserve the fixed coordinate and skeleton conversion needed by SONIC.
3. Keep the robot on live zero-command Normal until a complete, fresh 10-frame
   ZeroLab reference is ready and the operator explicitly arms it.
4. Preserve closed-loop whole-body balance when an armed human reference goes
   stale.
5. Prevent recovered human input from automatically regaining control.
6. Make all operator prompts and state names match the actual workflow.

## Non-Goals

- Do not change the vendor N-pose calibration procedure.
- Do not automatically classify N-pose or the operator's neutral pose.
- Do not change the PICO acquisition, calibration, conversion, or arming path.
- Do not tune SONIC weights, joint gains, action scale, or robot joint limits.
- Do not mask CAN errors, `motor_timeout`, inference exceptions, or framework
  safety transitions.
- Do not authorize a new hardware test until the simulation gates pass and the
  independent CAN fault is cleared or accepted by the hardware owner.

## Operator Workflow

The operator and safety observer perform the following sequence:

1. Complete the vendor's standard N-pose calibration.
2. Return to the robot-compatible neutral pose: upper arms down, elbows near
   90 degrees, and forearms pointing forward.
3. Send `btn_10=11` to enter ZeroLab.
4. Wait for `ZeroLab stream ready` and `ZeroLab ARM phase: WAIT_ARM`.
5. Confirm that the operator is still in the safe neutral pose.
6. Send `btn_10=12`.
7. Wait for the two-second handover and `ZeroLab ARM phase: ARMED` before
   starting small motions.

N-pose completion and neutral-pose readiness are explicit external
preconditions checked by people, not inferred by the application.

## Conversion Pipeline

For every valid UDP frame, `ZeroLabMotionConverter` performs exactly this
pipeline:

```text
vendor-calibrated joint_quat_world
  -> finite-value and quaternion-norm validation
  -> quaternion normalization
  -> per-joint cross-frame hemisphere/sign continuity
  -> Unity/XRT to application coordinate conversion
     (-qx, -qy, qz, qw)
  -> fixed ZeroLab-to-SMPL skeleton synthesis/mapping
  -> SMPL parent-relative localization
  -> ConvertedPoseFrame
```

The converter must not capture or apply a session rest orientation.
`TPoseCalibrator`, `apply_rest_alignment()`, `sampled_rest`, the 100-frame
calibration delay, and calibration-only constructor arguments are removed from
the ZeroLab runtime path. The first valid UDP frame therefore produces a
`ConvertedPoseFrame`.

Hemisphere/sign continuity is not a pose calibration. It only selects the
equivalent quaternion sign closest to the previous frame. A source-stale event
or stream epoch restart clears that history so a new stream cannot inherit sign
state from an old stream.

## Source Readiness

`PoseChunkWindow` remains the authority for policy-window completeness:

- It accepts converted frames with consecutive frame indices.
- It becomes ready only after 10 complete frames.
- `ZeroLab stream ready` is emitted only when that 10-frame window exists.
- A duplicate frame is ignored without advancing or clearing the window.
- A backward frame index, forward gap, or epoch restart clears the window; the
  next valid frame starts a new consecutive sequence.
- An invalid frame is rejected without advancing the window. Because its index
  is then missing, the following forward gap also starts a new sequence.
- The existing 0.5-second source/reference stale thresholds are unchanged.
- A stale source clears the source window and converter sign-continuity state.
- Recovery must collect 10 new consecutive frames before it can be considered
  fresh and ready again.
- No synthetic frame, repeated packet, or pre-stale frame may be used to fill
  the recovered window.

The state formerly named `WAIT_CALIBRATION` becomes `WAIT_STREAM`. It describes
window readiness and never implies that the application is calibrating the
human skeleton.

## Control State Machine

### Normal entry and initial ARM

```text
btn_10=11
    |
    v
WAIT_STREAM
  applied output: live zero-command Normal
  background: advance SONIC and observe ZeroLab readiness
    | complete fresh 10-frame reference
    v
WAIT_ARM
  applied output: live zero-command Normal
  background: advance SONIC and keep the reference current
    | btn_10=12 and reference fresh
    v
BLENDING
  applied output: live Normal <-> live SONIC smoothstep
  duration: 2.0 seconds
    |
    v
ARMED
  applied output: live SONIC
```

Both sides of the initial blend remain closed-loop policies. Normal feedback is
not replaced by a fixed entry frame.

If the reference becomes stale during `BLENDING`, the initial ARM is cancelled
and output returns to live Normal. The state returns to `WAIT_STREAM`; recovery
must rebuild 10 consecutive frames and the operator must explicitly ARM again.
It must not freeze the partially blended motor frame.

### Armed stale handling

When the source becomes stale in `ARMED`, the controller enters
`HOLD_REFERENCE`:

1. Latch the exact last complete SMPL reference window used by inference.
2. Stop new or recovered human windows from becoming the policy's active
   reference.
3. Continue SONIC inference every control tick using the latched human
   reference and current robot joint, velocity, IMU, and history feedback.
4. Apply each newly computed complete motor frame.
5. Mark human control disarmed and require a new `btn_10=12`.

`HOLD_REFERENCE` freezes the human target, not the motor output. This is a
critical distinction for a biped: repeating one motor frame removes the active
balance corrections even if its position targets and gains are complete.

The existing log demonstrates the failure mode that must not recur:

```text
1787134348.842  ARM blend accepted
1787134349.527  reference stale
1787134349.532  holding last motor frame
```

The motor frame was frozen only 0.685 seconds into the handover. The revised
path must continue producing feedback-dependent SONIC outputs instead.

### Reference gate and recovery ARM

The policy/reference boundary separates two concepts:

- **Observed reference:** the newest validated source window and its receive
  time, used to detect source recovery.
- **Active reference:** the window currently consumed by SONIC inference.

In normal live operation they are the same. In `HOLD_REFERENCE`, the active
reference remains latched while recovered observed windows are retained as
pending input. Merely receiving a fresh pending window must not move the robot.

When a fresh pending 10-frame window exists and the operator sends
`btn_10=12`, the controller enters `REARMING`. Over two seconds it applies a
smoothstep interpolation from the latched reference window to the newest live
window while SONIC continues to run against current robot feedback on every
tick. Continuous position-like reference fields are interpolated linearly;
quaternion fields use normalized, hemisphere-corrected spherical
interpolation. Slot `i` is blended with slot `i` across the two complete
10-frame windows.

At the end of the reference blend, the gate opens, the newest observed window
becomes active, and the state returns to `ARMED`. If the source becomes stale
again during `REARMING`, the current interpolated reference is latched and the
state returns to `HOLD_REFERENCE`. Recovery never resumes automatically.

```text
ARMED + source stale
    -> HOLD_REFERENCE (latched human target, live SONIC balance)

HOLD_REFERENCE + source recovered
    -> HOLD_REFERENCE (fresh window pending, still gated)

HOLD_REFERENCE + fresh pending window + btn_10=12
    -> REARMING (2-second reference-space blend)
    -> ARMED
```

This recovery is deliberately a reference-space blend rather than a blend
from a frozen motor frame. SONIC therefore continues to receive current
proprioception and compute balance corrections throughout recovery.

## Invalid Input and Epoch Handling

- Non-finite data, invalid quaternion norms, malformed shapes, and invalid
  source metadata are rejected before conversion or policy publication.
- Duplicate and out-of-order frames do not advance `PoseChunkWindow`.
- A stream epoch change clears the source window, converter sign-continuity
  state, observed pending reference, and readiness state.
- Invalid data cannot be replaced with synthetic pose frames.
- An inference/backend exception follows existing framework error handling; it
  is not relabeled as source stale and silently held.

## Button Semantics

- `btn_10=11` selects ZeroLab and starts stream acquisition. It does not ARM
  and does not start application-level calibration.
- `btn_10=12` is accepted only in `WAIT_ARM`, or in `HOLD_REFERENCE` when a
  fresh pending 10-frame window exists.
- `btn_10=12` is refused with an explicit bounded warning in `WAIT_STREAM` or
  `HOLD_REFERENCE` without fresh pending data.
- Repeated ARM in `BLENDING`, `REARMING`, or `ARMED` is ignored with a bounded
  informational log.
- Normal, PD Brake, zero torque, recover, and established orientation-safety
  routes remain reachable from every internal ZeroLab phase.

## Logging and User-Facing Text

Remove all ZeroLab text that asks the operator to hold a T-pose or says that
the application is collecting/calibrating T-pose. Replace it with bounded
messages for:

- external vendor N-pose prerequisite;
- `WAIT_STREAM`, including current consecutive-frame count when useful;
- `ZeroLab stream ready` only after a complete window;
- `WAIT_ARM`, `BLENDING`, `ARMED`, `HOLD_REFERENCE`, and `REARMING`;
- source stale, reference latched, source recovered but gated, ARM refused,
  recovery ARM accepted, and recovery completed.

The logs must distinguish source freshness from reference activation so an
operator can see that a recovered stream is pending rather than controlling
the robot.

## Automated Tests

### Converter tests

1. The first valid packet returns `ConvertedPoseFrame`; there is no 100-frame
   warm-up.
2. Known vendor world quaternions produce the expected coordinate-converted,
   mapped, and parent-local SMPL orientations.
3. No sampled-rest inverse is present in the expected result.
4. Quaternion sign flips do not create a motion discontinuity.
5. Stale and epoch-reset events clear sign-continuity state.
6. Invalid shapes, non-finite values, and zero-norm quaternions are rejected.

### Source tests

1. Nine consecutive valid frames do not publish a ready window; the tenth
   does.
2. Stale, epoch restart, and discontinuity handling require 10 new consecutive
   frames.
3. Duplicate, invalid, and pre-stale frames cannot complete a window.
4. No T-pose collection log or status remains.

### State and policy tests

1. `WAIT_STREAM` and `WAIT_ARM` apply changing live-Normal outputs.
2. Initial ARM uses a two-second live-Normal/live-SONIC smoothstep.
3. Initial-blend stale cancels ARM and returns to live Normal without emitting
   a repeated motor frame.
4. Armed stale enters `HOLD_REFERENCE`, keeps the active reference identical,
   and keeps running SONIC inference.
5. With the reference held, changing joint/IMU feedback changes the applied
   motor output. This is the regression assertion that proves motor output is
   not frozen.
6. Recovered observed windows do not change the active reference or applied
   human target before a new ARM.
7. Recovery ARM performs the two-second reference-space blend and then opens
   the gate.
8. A second stale event during `REARMING` latches the interpolated reference
   and returns to `HOLD_REFERENCE`.
9. Existing emergency and orientation-safety routes remain valid.
10. Existing PICO tests and behavior remain unchanged.

## Offline and MuJoCo Acceptance

### Recorded-data comparison

Replay the same raw ZeroLab recording through the old and revised converters.
Inspect neutral pose, both arms forward, both arms down, and independent small
left/right movements. The revised output must remove the session-rest shoulder
offset and match the direct vendor-world-to-parent-local calculation.

### MuJoCo sequence

Run the complete state flow:

```text
Normal -> btn_10=11 -> WAIT_STREAM -> WAIT_ARM
       -> btn_10=12 -> 2 s BLENDING -> ARMED
```

Acceptance requires:

- neutral robot arms agree with the defined operator neutral pose;
- both arms can move down and forward, with correct left/right direction and
  no shoulder joint-limit violation;
- unilateral and bilateral low-amplitude motions remain stable;
- induced source gaps of 0.6 seconds, 2 seconds, and 30 seconds do not repeat a
  fixed motor frame and do not cause a fall;
- recovered data does not resume human control before a new ARM;
- recovery ARM completes the reference-space blend without a discontinuity;
- no hardware process is running on the simulation host.

Any fall, automatic recovery, fixed motor-frame hold, missing state phase,
joint-limit violation, or direction mismatch fails simulation acceptance.

## Hardware Gate

Passing converter and MuJoCo tests does not clear the current hardware fault.
Before another real-robot ARM attempt:

1. `motor_timeout` and CAN transmit/receive errors must be absent, or the
   hardware owner must explicitly accept and supervise the condition.
2. The robot must use a support rig, working emergency stop, clear test area,
   and a dedicated safety observer.
3. The first motions must be neutral hold followed by small, single-arm
   movements before bilateral or downward motions.

The converter change and CAN diagnosis remain separate acceptance tracks.
