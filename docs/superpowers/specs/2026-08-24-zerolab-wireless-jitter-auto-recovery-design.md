# ZeroLab Wireless Jitter Hiding and Automatic Recovery Design

**Date:** 2026-08-24
**Status:** Approved design
**Scope:** ZeroLab-only wireless input resampling, stale-reference handling, and
automatic recovery for `com.bxi.sonic/sonic_zerolab` on ELF3 simulation and
real hardware

## Relationship to Existing ZeroLab Behavior

This design extends the vendor-calibrated stream and closed-loop stale-hold
behavior documented in
`2026-08-20-zerolab-vendor-calibrated-stream-design.md`.

The following existing behavior remains in force:

- Vendor N-pose calibration and the ZeroLab-to-SMPL coordinate conversion are
  unchanged.
- Entering ZeroLab still requires `btn_10=11`.
- The first transition from live Normal to SONIC still requires one explicit
  `btn_10=12` and a two-second blend.
- A stale armed reference freezes the human target, not the motor output;
  SONIC continues closed-loop inference using current robot proprioception.
- Normal, PD Brake, zero torque, and the existing framework safety routes can
  interrupt ZeroLab at any time.
- PICO acquisition, buffering, recovery, and control behavior are unchanged.

This design supersedes only the earlier requirement that every recovered
ZeroLab stream needs another `btn_10=12`. After a session has been armed once,
valid recovered input automatically re-enters through a controlled
reference-space blend.

## Evidence and Protocol Limits

The direct wired measurement at the receiving ELF3 computer was:

```text
PACKETS=2399
AVERAGE_HZ=40.41
MAX_GAP_MS=41.8
GAPS_GT_100MS=0
GAPS_GT_500MS=0
GAPS_GT_1000MS=0
```

Earlier wireless measurements contained gaps up to approximately 1.69
seconds, while the Windows sender capture was normally near 20 ms. The Linux
kernel capture reported no kernel packet drops. These observations justify a
receiver-side jitter-hiding experiment, but they do not prove that the robot
receiver itself caused the wireless gap.

The vendor UDP datagram is exactly 992 bytes and has no sender sequence number,
capture timestamp, or stream epoch. The receiver assigns a local index only
after a datagram arrives. Consequently this implementation cannot determine
the original sampling time, reconstruct lost packets, reliably reorder vendor
packets, or distinguish delayed old packets from newly sampled packets. An ACK
layer at this receiver would not fix those omissions. True loss/reorder
recovery requires a vendor protocol revision containing at least a monotonic
sequence number and sender capture timestamp.

The design therefore hides bounded arrival jitter, rejects invalid input,
drops excessive backlog, and controls the robot safely across gaps. It does
not claim to repair the network or reproduce missing human motion.

## Goals

1. Publish a bounded-latency ZeroLab pose stream at a fixed 50 Hz despite
   ordinary wireless arrival jitter.
2. Smooth small timing variation without extrapolating human motion.
3. Hold the last human reference across short gaps while SONIC keeps using
   live proprioceptive feedback.
4. After a stale event, automatically recover any previously armed session
   once ten new real valid packets arrive, regardless of dropout duration.
5. Blend recovered human motion in reference space so the robot never snaps
   directly to the latest recovered pose.
6. Keep real-packet freshness and recovery readiness independent from
   interpolated or held output frames.
7. Make buffering, stale events, and automatic recovery observable in logs
   and counters.

## Non-Goals

- Do not change the vendor sender or its 992-byte packet format.
- Do not add a receiver ACK protocol that the vendor sender does not support.
- Do not extrapolate human position or orientation through a missing interval.
- Do not replay an old burst until all queued poses have been consumed.
- Do not add a pose-difference gate before automatic recovery. The operator
  explicitly selected unconditional automatic recovery.
- Do not remove the initial explicit ARM action.
- Do not change SONIC weights, gains, action scale, joint limits, or PICO.
- Do not hide malformed packets, policy exceptions, CAN faults,
  `motor_timeout`, or framework safety transitions as network jitter.
- Do not make the wireless configuration replace the known-good direct-wired
  configuration.

## Configuration

The ZeroLab source gains these strictly validated parameters:

```yaml
jitter_buffer_seconds: 0.08
short_recovery_blend_seconds: 0.2
stale_seconds: 0.5
auto_rearm_on_recovery: true
auto_rearm_blend_seconds: 2.0
recovery_real_frames: 10
```

Their meanings are fixed as follows:

- `jitter_buffer_seconds` is the target playout delay used by the 50 Hz pose
  resampler. It is not added to the stale timeout.
- `short_recovery_blend_seconds` blends from a temporary held pose to live
  buffered output when real input returns before the stale threshold.
- `stale_seconds` is measured from the most recent real, correctly sized,
  allowed-sender, successfully parsed, finite, converted UDP packet. The
  comparison is strict: age greater than 0.5 seconds is stale.
- `auto_rearm_on_recovery` affects only a session that reached `ARMED` after
  an explicit initial ARM. It never bypasses initial arming.
- `auto_rearm_blend_seconds` controls the reference-space recovery blend after
  a stale event.
- `recovery_real_frames` counts only new real valid packets received after the
  most recent stale transition.

All durations must be finite and positive. `recovery_real_frames` must be an
integer of at least 1. The output rate remains exactly 50 Hz and the SONIC
window remains exactly ten frames.

## Architecture and Data Flow

The ZeroLab-only path becomes:

```text
vendor UDP datagram
  -> sender/size/protocol/finite validation
  -> coordinate conversion and parent localization
  -> ZeroLabPoseResampler
  -> ten-frame SONIC pose window
  -> observed/active reference gate
  -> SONIC policy with current robot feedback
  -> motor frame
```

`ZeroLabPoseResampler` is an isolated component. It owns only pose playout and
its statistics; it does not decide robot control phases, call the policy, or
interpret buttons. The source core remains responsible for real-packet
freshness and stale transitions. The reference gate remains responsible for
choosing the human reference consumed by SONIC. The armed teleoperation state
remains responsible for initial ARM, automatic recovery phase timing, and
emergency interruption.

The resampler input contains the converted pose plus its local monotonic UDP
arrival time. Its output contains a pose and metadata identifying the output
as one of:

- `real`: exactly a converted real sample selected for playout;
- `interpolated`: computed between two real samples;
- `held`: a copy of the last emitted pose because no safe interpolation is
  available.

Output metadata must retain the newest contributing real local frame index and
real arrival time for diagnostics. A resampled output index may advance at 50
Hz for downstream window construction, but it must never be confused with a
vendor sequence number or used as evidence of real source freshness.

### Freshness metadata contract

Every source-to-bridge pose message carries these source-authoritative fields
alongside its pose window:

- `source_generation`: changes on source startup and every stale transition;
- `latest_real_frame_index`: the newest accepted local UDP receive index;
- `latest_real_receive_timestamp_ns`: its monotonic receive time;
- `real_valid_frames_in_generation`: count of valid real packets after the
  current startup/stale readiness barrier, capped at the readiness threshold;
- `real_stream_ready`: true only after ten such real packets and a complete
  usable output window exist;
- `playout_kind`: `real`, `interpolated`, `held`, or `short_recovery_blend`;
- `source_stale`: true once real-packet age is greater than the configured
  stale threshold.

The pose bridge and policy preserve these fields through the observed
reference. `has_fresh_live_reference()` uses the source real-packet timestamp
and `source_stale`, never the ZMQ message receive time. Initial and recovery
readiness use `source_generation` plus `real_stream_ready`, never the number of
pose messages received by the bridge. A delayed message from an older source
generation is rejected. This contract prevents a 50 Hz stream of held or
interpolated messages from keeping human control falsely fresh.

## Resampler Semantics

### Clock and bounded playout

The timer calls the resampler at exactly 50 Hz using a monotonic clock. At
timer time `now`, the target playout time is:

```text
target = now - jitter_buffer_seconds
```

The resampler keeps only the smallest ordered set of real converted samples
needed to bracket the current target and produce the next output. Arrival time
is used as the sample time because the vendor packet has no sender timestamp.
When two real samples bracket the target, position-like fields use linear
interpolation. Quaternion fields use normalized interpolation after flipping
the second quaternion when necessary so both quaternions occupy the same
hemisphere. Every emitted quaternion is normalized.

No pose is extrapolated beyond the newest real sample. If the target cannot be
bracketed, the resampler emits the last pose it emitted and marks it `held`.
Before the first safe output exists, it publishes nothing.

### Backlog policy

After each UDP drain, samples older than the last sample needed to bracket the
next target are removed. If a burst advances the newest real sample so far
ahead that replaying all arrivals would increase latency beyond the bounded
playout delay, intermediate backlog is discarded while preserving the two
samples needed for the next possible interpolation. Discarded samples increase
`dropped_backlog_frames`.

The receiver never accelerates output above 50 Hz to catch up. This avoids
turning an old wireless burst into delayed robot motion.

### Short-gap behavior

If interpolation becomes impossible but the real-packet age is no more than
0.5 seconds, the resampler holds the last emitted human pose and the control
phase remains `ARMED`. SONIC still runs every control tick against current
robot feedback.

When real input returns before a stale transition, the active pose moves from
the held pose to the current buffered resampler pose over 0.2 seconds using
the same position and quaternion interpolation rules. Another hold during this
catch-up restarts the catch-up from the pose actually active at that moment.
This is playout recovery inside `ARMED`; it is not `REARMING` and does not
require ten recovery packets.

### Real-packet authority

Only an accepted, successfully parsed, finite, converted UDP packet may:

- update the last-real-packet arrival time;
- increase the post-stale recovery count;
- update the measured arrival gap;
- satisfy initial or recovery readiness.

Interpolated, held, and short-recovery blend frames may fill the regular
fixed-rate downstream pose window only after initial real-stream readiness,
but they cannot keep the source fresh or complete the ten-real-packet gate.
Malformed, wrong-sized, wrong-sender, non-finite, or conversion-failing
datagrams do not count.

## Control State Machine

### Entry and initial ARM

Initial entry remains explicit:

```text
Normal
  -> btn_10=11
WAIT_STREAM
  -> ten real valid packets and usable resampler output
WAIT_ARM
  -> initial btn_10=12
BLENDING (live Normal -> live SONIC, 2.0 s)
  -> ARMED
```

`WAIT_STREAM` and `WAIT_ARM` apply live zero-command Normal. If input becomes
stale during initial `BLENDING`, the blend is cancelled and the controller
returns to `WAIT_STREAM`. Because the session has not yet reached `ARMED`, it
must again collect ten real valid packets and receive explicit `btn_10=12`.
Automatic recovery is not enabled merely by entering `BLENDING`.

### Armed operation and short gaps

While the latest real-packet age is at most 0.5 seconds, the phase remains
`ARMED`. An unavailable interpolation produces a held human pose; return of
real data invokes the 0.2-second short-recovery blend described above. These
synthetic outputs never refresh the stale clock.

### Stale hold and unconditional automatic recovery

When real-packet age becomes greater than 0.5 seconds in `ARMED` or
`REARMING`, the controller enters `HOLD_REFERENCE`:

1. Latch the exact human reference active at the transition.
2. Keep running SONIC on that reference and current robot feedback.
3. Clear the resampler backlog and its interpolation continuity.
4. Reset the post-stale real-valid-packet counter to zero.
5. Prevent all recovered input from becoming the active reference until the
   recovery gate opens.

The source timer may continue emitting the last output at 50 Hz during this
state, but every such message is marked `held` and `source_stale`. Clearing
interpolation continuity does not discard the last output value needed for
that diagnostic playout. The reference gate ignores these messages as
freshness evidence and continues using its independently latched active
reference.

For a session that previously reached `ARMED`, recovery is:

```text
HOLD_REFERENCE
  -> receive 10 new real valid packets
REARMING (latched reference -> newest live reference, 2.0 s)
  -> ARMED
```

No new `btn_10=12` is required. There is no maximum dropout duration and no
pose-difference test. A one-second, two-second, or thirty-second gap follows
the same ten-real-packet and two-second blend procedure.

During `REARMING`, SONIC continues running with current robot feedback. The
reference gate blends continuous position-like fields linearly and
quaternion fields with normalized, hemisphere-corrected interpolation.
Smoothstep timing is used from zero to one. Slot `i` in the latched ten-frame
window is blended with slot `i` in the newest observed ten-frame window.

If real input becomes stale again during `REARMING`, the exact currently
interpolated human reference is latched and the state returns immediately to
`HOLD_REFERENCE`. The counter resets. Ten further real valid packets start a
new automatic two-second `REARMING` attempt.

An explicit `btn_10=12` received in `HOLD_REFERENCE`, `REARMING`, or `ARMED`
is informationally ignored because automatic recovery owns those transitions.

## Epoch, Invalid Input, and Lifecycle Handling

- Starting a new ZeroLab state lifecycle creates a new local stream epoch and
  clears the receiver index, resampler, downstream window, freshness time,
  recovery count, observed reference, and active gate.
- Leaving ZeroLab for Normal, PD Brake, zero torque, or another state disables
  automatic recovery for the old session. Re-entering ZeroLab always requires
  the normal initial ARM sequence.
- A stale transition clears resampler interpolation continuity and downstream
  readiness data but preserves only the explicitly latched active human
  reference used for closed-loop hold.
- Invalid input is rejected before it reaches the resampler and never changes
  freshness or readiness.
- Policy/backend exceptions follow existing framework error handling. They do
  not trigger or masquerade as the UDP recovery state machine.
- Established orientation-safety handling and emergency transitions have
  priority over all buffering and recovery actions.

## Observability

The ZeroLab path exposes monotonically increasing counters for the current
lifecycle. Source-owned counters are carried in periodic source logs; the
control-state-owned automatic-rearm counter is reported in state logs:

- `real_valid_packets`
- `maximum_real_arrival_gap_ms`
- `interpolated_output_frames`
- `held_output_frames`
- `dropped_backlog_frames`
- `stale_events`
- `auto_rearm_count`

Existing invalid-size, unexpected-sender, parse-error, and dropped-publication
counters remain separate. Periodic bounded logs report these values without
logging every 50 Hz frame.

State transition logs must distinguish:

```text
ZeroLab ARM phase: WAIT_STREAM
ZeroLab ARM phase: WAIT_ARM
ZeroLab ARM accepted; blending live Normal -> SONIC for 2.000 s
ZeroLab ARM phase: ARMED
ZeroLab short input hold
ZeroLab short input recovered; blending buffered reference for 0.200 s
ZeroLab reference stale; ARM phase: HOLD_REFERENCE
ZeroLab recovery ready; 10 real valid packets received
ZeroLab automatic recovery; ARM phase: REARMING for 2.000 s
ZeroLab automatic recovery complete; ARM phase: ARMED
```

Repeated 50 Hz hold messages must be rate-limited. Logs must never claim that
an interpolated or held frame is a real packet.

## Wired and Wireless Packaging

The committed direct-wired adapter remains the known-good variant with:

```yaml
allowed_sender: 192.168.1.52
```

Wireless testing uses a separate branch/install variant with:

```yaml
allowed_sender: 192.168.89.171
```

The wireless build must have a distinct install directory and log path so it
cannot silently overwrite or be confused with the wired candidate. Switching
variants requires stopping all `hardware_elf3`,
`bxi_example_py_elf3_demo`, and `zerolab_source` processes first. The sender
address difference must remain covered by manifest tests for each packaged
variant.

## Validation Strategy

### Unit and component tests

Tests use a fake monotonic clock and deterministic converted poses. They cover:

1. Regular 50 Hz arrivals produce fixed 50 Hz output after the 80 ms playout
   delay.
2. Bounded jitter selects correct brackets and interpolates position and
   quaternion fields correctly.
3. Quaternion sign changes do not create long-path rotations.
4. Bursts drop obsolete backlog instead of replaying it or exceeding 50 Hz.
5. Missing brackets produce held frames and never extrapolate.
6. Held and interpolated frames do not update real freshness or recovery
   counts.
7. Wrong sender, wrong size, malformed, non-finite, and conversion-failing
   packets do not count as valid recovery frames.
8. The stale boundary is exact: 0.5 seconds is fresh and greater than 0.5
   seconds is stale.
9. A short gap recovers through a 0.2-second blend without leaving `ARMED`.
10. A stale gap latches the current reference and resets recovery readiness.
11. Exactly ten post-stale real valid packets automatically enter `REARMING`.
12. No initial automatic ARM occurs before the first explicit `btn_10=12`.
13. A second stale event during `REARMING` returns to `HOLD_REFERENCE`, resets
    the counter, and permits another automatic attempt.
14. State exit and a new lifecycle clear all automatic-recovery eligibility.
15. A directly injected duplicate, backward, or non-increasing local input
    index is rejected or resets readiness according to the component
    contract. This is defensive API validation, not a claim that vendor
    network reordering can be detected.
16. Emergency and established safety transitions preempt `HOLD_REFERENCE` and
    `REARMING`.

Because production local indices are assigned in receive order and the packet
lacks sender sequencing, tests must not claim to detect true network
reordering. The defensive index test covers only inconsistent inputs injected
directly at the component boundary.

### MuJoCo gates

Before hardware use, deterministic input interruption tests cover gaps of:

```text
0.10 s, 0.49 s, 0.51 s, 2.0 s, and 30.0 s
```

The gates are:

- Output pose publication remains fixed at 50 Hz once initially ready.
- The 0.10-second and 0.49-second cases remain `ARMED`, hold rather than
  extrapolate, and use the 0.2-second short-recovery blend.
- The 0.51-second, 2.0-second, and 30.0-second cases enter
  `HOLD_REFERENCE`, require ten real packets, and automatically transition
  `HOLD_REFERENCE -> REARMING -> ARMED` without another button press.
- A forced stale event during `REARMING` returns to `HOLD_REFERENCE` and later
  retries only after another ten real packets.
- No burst replay causes growing end-to-end latency.
- Normal, PD Brake, and zero torque interrupt every ZeroLab phase.
- Existing ZeroLab and PICO regression suites remain green.

### Real-hardware gates

Real-hardware testing begins only after all unit and MuJoCo gates pass. It
requires a support rig or equivalent fall protection, a working emergency
stop, and a dedicated safety observer. The sequence is:

1. Verify the selected install, allowed sender, interface route, and live UDP
   gap statistics before enabling motors.
2. Enter ZeroLab with the operator neutral and perform the one explicit
   initial ARM.
3. Hold neutral through the first controlled short interruption.
4. Test one small single-arm motion with no intentional interruption.
5. Introduce controlled 0.1-second, then 0.49-second interruptions.
6. Only after those pass, test one controlled stale interruption while the
   operator returns to and remains neutral before data resumes.
7. Confirm logs and video show the expected hold and smooth automatic re-entry.

Testing must stop immediately on unexpected whole-body motion, loss of balance,
failure of an emergency route, a policy exception, or a new hardware fault.
The two-second automatic recovery is a test configuration, not evidence that
unconditional recovery is safe for unattended operation.

## Acceptance Criteria

The feature is accepted only when:

- Initial ZeroLab control cannot activate without one explicit `btn_10=12`.
- Regular and jittered input produce bounded fixed-rate 50 Hz playout.
- No synthetic output refreshes real-packet freshness or recovery readiness.
- Short gaps hold the human reference and recover with the 0.2-second blend.
- Every stale, previously armed session holds the human reference, waits for
  ten real valid packets, and automatically uses a two-second reference-space
  rearm without another button press.
- Stale input during rearming returns to hold and can retry automatically.
- Backlog is dropped instead of replayed with accumulating latency.
- Safety transitions preempt buffering and recovery.
- PICO behavior is unchanged.
- The direct-wired build remains available independently from the wireless
  test build.
