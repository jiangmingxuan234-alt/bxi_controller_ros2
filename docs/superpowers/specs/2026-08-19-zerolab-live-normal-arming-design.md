# ZeroLab Live-Normal Safe Arming Design

**Date:** 2026-08-19  
**Status:** Approved for implementation planning
**Supersedes:** The captured-Normal-frame waiting and fixed-start first-ARM blend
defined in `2026-08-19-zerolab-safe-arming-design.md`

## Context

The first ZeroLab safe-arming implementation keeps the robot on a deep copy of
the last complete Normal `MotorFrame` while ZeroLab calibrates and waits for the
operator's explicit ARM event. That frame contains `qpos`, `kp`, `kd`, `vel`,
and `torque`, but it is static.

MuJoCo acceptance on 2026-08-19 disproved the assumption that replaying that
frame is equivalent to remaining in Normal. The observed sequence was:

```text
1787135930.263  Normal -> sonic_zerolab
1787135930.267  WAIT_CALIBRATION
1787135937.312  WAIT_ARM
```

No `ZeroLab ARM accepted` or `ZeroLab ARM phase: ARMED` message occurred. The
simulated robot fell before ARM because the Normal feedback policy stopped
updating while the state repeatedly emitted one captured command. The ARM event
sent before `btn_10=11` was correctly ignored and was not queued.

The run also reproduced independent approximately 500 ms ZeroLab reference
gaps. This design does not change their timeout or attempt to hide them.

## Goals

1. Keep the existing `btn_10=11` entry and `btn_10=12` explicit ARM workflow.
2. Run the existing Normal policy with a zero velocity command on every control
   cycle during `WAIT_CALIBRATION` and `WAIT_ARM`.
3. Continue advancing the existing SONIC policy in the background so ZeroLab
   calibration, source chunks, and reference freshness continue to work.
4. On the first accepted ARM, blend for two seconds between the current live
   Normal output and the current live SONIC output.
5. Preserve the existing complete-frame stale hold, explicit recovery ARM, and
   emergency routes.
6. Prove in MuJoCo that the robot can wait for at least ten seconds without
   falling before any hardware retest.

## Non-goals

- Do not change the PICO `sonic_teleop` state or its activation event.
- Do not change ZeroLab packet parsing, quaternion conversion, T-pose
  calibration, SONIC reference playback, policy weights, MuJoCo model, or the
  hardware driver.
- Do not change the `0.5 s` source/reference stale thresholds in this change.
- Do not make recovery automatic after stale input.
- Do not deploy to or test on hardware until the new MuJoCo acceptance passes.

## Considered Approaches

### 1. Separate ZeroLab preparation state

Add a `zerolab_prepare` framework state based on Normal, scope the ZeroLab
source and bridge to both preparation and active states, and transition to the
active state on `btn_10=12`.

This makes preparation explicit, but it adds a state, routes, lifecycle
handoff, readiness transfer, and SONIC policy reset semantics. It is a larger
change than required.

### 2. Dual-policy ZeroLab state

Keep the current ZeroLab state and node lifecycle. Give only
`ZeroLabArmedTeleopState` a handle to the existing Normal policy. During the
waiting phases it advances both policies and applies only Normal. During the
first blend it dynamically mixes both current outputs.

This retains the operator workflow, limits changes to the ZeroLab path, and
uses the existing tested ARM/stale state machine. This is the selected
approach.

### 3. Forced SONIC idle reference while waiting

Suppress live references and apply the SONIC idle policy until ARM. This avoids
a second inference, but it does not satisfy the requirement that the robot
remain on the real Normal balancing policy and may reintroduce the shoulder and
stance differences that motivated the ARM gate.

## Resource and Component Design

`com.bxi.sonic` already declares `com.bxi.basic_actions` as a required Mod.
Its plugin will request the existing resource key
`com.bxi.basic_actions/normal_policy` and pass that handle only to the
`sonic_zerolab` factory. The PICO `sonic_teleop` factory remains unchanged.

`SonicTeleopState` will accept an optional tuple of additional required
resources, defaulting to empty. It will include those handles alongside its
SONIC policy in the base `RobotControlState` resource list. Only
`ZeroLabArmedTeleopState` supplies the Normal policy handle. This preserves
resource readiness checks without changing PICO behavior.

`ZeroLabArmedTeleopState` owns resolved frame buffers for:

- the current Normal policy frame;
- the current SONIC policy frame;
- the last frame actually applied;
- the stale hold frame;
- the blend output.

The old captured entry frame remains available only for the very short
framework entry transition and as initialization before the first dynamic
sample. It is no longer the steady output of either waiting phase.

## State and Data Flow

```text
WAIT_CALIBRATION
  output: live zero-command Normal policy
  background: advance SONIC and poll ZeroLab reference
      | fresh complete reference
      v
WAIT_ARM
  output: live zero-command Normal policy
  background: advance SONIC and keep reference current
      | btn_10=12 and fresh reference
      v
BLENDING
  output: live Normal * (1-alpha) + live SONIC * alpha
  alpha: smoothstep(elapsed / 2.0 s)
      | elapsed >= 2.0 s
      v
ARMED
  output: live SONIC only

BLENDING or ARMED + stale reference
      -> HOLD_STALE

HOLD_STALE + fresh reference + btn_10=12
      -> two-second recovery blend from the frozen frame
```

### Waiting phases

Every advancing control tick:

1. Publish a zero command into the shared inference frame. The ZeroLab state has
   no speed profile, so its normal `get_cmd_vel()` result is zero regardless of
   the command that was active before entry.
2. Advance the existing Normal policy and resolve its output into the Normal
   frame buffer.
3. Advance the SONIC policy and resolve its output into the SONIC frame buffer.
4. Update reference freshness and the `WAIT_CALIBRATION -> WAIT_ARM` phase.
5. Apply the current Normal frame.

The Normal policy resource is shared with the preceding Normal state, so its
feedback history continues instead of being reinitialized at ZeroLab entry.

### First ARM blend

The first accepted ARM does not use a fixed blend start frame. On each tick it
advances and resolves both policies, then applies smoothstep interpolation to
all five complete-frame fields:

```text
alpha = smoothstep(clamp(elapsed / arm_blend_seconds, 0, 1))
output = normal_current + alpha * (sonic_current - normal_current)
```

This keeps dynamic Normal feedback dominant at the beginning of handover and
allows SONIC feedback to take over continuously. At `alpha == 1`, the phase
becomes `ARMED`.

### Armed phase

After the first blend completes, Normal inference stops. Only SONIC advances
and its resolved frame is applied. This avoids a permanent dual-inference cost.

### Recovery ARM

Once stale input has caused `HOLD_STALE`, the robot may no longer be in a pose
that the Normal policy can safely assume. Recovery therefore preserves the
existing behavior: a fresh reference plus a new `btn_10=12` performs a
two-second smoothstep from the frozen complete frame to the current live SONIC
frame. It does not switch back to Normal.

The implementation must distinguish the initial live-Normal blend from this
frozen-frame recovery blend.

## Stale and Safety Behavior

- In `WAIT_CALIBRATION` or `WAIT_ARM`, stale ZeroLab input does not affect the
  applied live Normal output. ARM is refused until a complete fresh reference
  returns.
- In `BLENDING` or `ARMED`, stale input deep-copies the last frame actually
  applied, enters `HOLD_STALE`, cancels ARM, and stops automatic motion.
- Recovery never resumes automatically. The operator must return to a safe
  neutral pose and send `btn_10=12` again.
- Existing Normal, PD Brake, zero torque, and recover routes remain available
  in every internal phase.
- The waiting and initial-blend phases retain the Normal state's unsafe
  orientation response. The established SONIC behavior remains unchanged once
  fully `ARMED`.
- An inference exception is not converted into a stale event or silently
  ignored; existing framework error handling remains authoritative.

The known 500 ms source gaps remain visible. If they occur while waiting, the
robot continues balancing in Normal. If they occur after takeover begins, the
existing hold and re-ARM rule remains intentionally conservative.

## Logging

Retain the existing phase and reference logs. Add one bounded entry message
that states the waiting output source, for example:

```text
ZeroLab pre-ARM output: live zero-command Normal policy
```

The first ARM log identifies the live-Normal blend. Recovery identifies the
frozen-frame blend. Logs must not print every control tick.

## Test Design

### Unit tests

Write failing regression tests before changing production code:

1. Feed changing Normal policy outputs through multiple
   `WAIT_CALIBRATION`/`WAIT_ARM` ticks and assert the applied output changes
   with them instead of matching the captured entry frame.
2. Make the reference stale in each waiting phase and assert live Normal output
   continues.
3. Assert no SONIC frame reaches the applied output before an accepted ARM.
4. Assert ARM is refused without a fresh complete reference.
5. At half of the first blend, assert every complete-frame field equals the
   smoothstep mix of the current Normal and current SONIC frames, including
   when the Normal frame differs from the previous tick.
6. At two seconds, assert `ARMED` and direct SONIC output, and assert Normal is
   no longer advanced.
7. Assert stale input during initial blend and armed operation freezes the last
   applied complete frame and enters `HOLD_STALE`.
8. Assert recovery requires another ARM and uses the frozen-frame recovery
   blend, not Normal.
9. Assert only `sonic_zerolab` requires the Normal policy resource and the PICO
   state construction and behavior remain unchanged.

### Existing regression suites

Run all ZeroLab tests, the complete `bxi_example_py_elf3/test` directory, and a
clean local `colcon` build/install. The installed manifest and factory must be
tested, not only source imports.

### MuJoCo acceptance

Using the candidate install and the live ZeroLab sender:

1. Reach a stable Normal state.
2. Send `btn_10=11` and hold the preparation state for at least ten seconds.
3. Confirm the robot does not fall or violate joint limits while waiting.
4. Hold T-pose until `stream ready`, source chunks, and `WAIT_ARM` are logged.
5. Return to a neutral operator pose and send `btn_10=12`.
6. Confirm the live-Normal blend lasts two seconds and reaches `ARMED`.
7. Check small left, right, and bilateral arm movements for correct direction.
8. Deliberately stop input, verify frozen output and cancelled ARM, restore
   input, verify no automatic motion, and explicitly re-ARM.
9. Return to Normal and shut down cleanly.

Record phase logs, fall status, joint-limit status, control-period statistics,
and source/bridge stale events. Any fall, joint-limit violation, missing phase,
automatic stale recovery, or hardware process on the simulation host fails
acceptance.

## Deployment Gate

No robot workspace update or hardware run is authorized by this design alone.
Hardware deployment requires passing unit, package, build/install, and MuJoCo
acceptance, followed by a separate review of the 500 ms data gaps and control
timing evidence.
