# ZeroLab Wireless Jitter Hiding and Automatic Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add bounded 50 Hz jitter-hiding playout to the ZeroLab wireless path and automatically, smoothly rearm every previously armed session after ten recovered real packets, while preserving the first explicit `btn_10=12` and all robot safety exits.

**Architecture:** Insert a pure `ZeroLabPoseResampler` between calibrated packet conversion and the existing ten-frame source window. Carry real-packet freshness and recovery-generation metadata through the shared bridge into `SmplReferenceFrame`, then let the ZeroLab state automatically drive the existing reference gate from `HOLD_REFERENCE` through a two-second reference-space blend. PICO uses the same bridge and policy files but follows the legacy metadata-free branch unchanged.

**Tech Stack:** Ubuntu 22.04, ROS 2 Humble, Python 3.10, NumPy, SciPy quaternion math, pyzmq, pytest, YAML Mod manifests, colcon, MuJoCo, ROS 2 CLI.

## Global Constraints

- Use `jitter_buffer_seconds=0.08`, `short_recovery_blend_seconds=0.2`, source `stale_seconds=0.5`, `auto_rearm_on_recovery=true`, `auto_rearm_blend_seconds=2.0`, and `recovery_real_frames=10` exactly.
- Keep ZeroLab pose output configured at exactly 50 Hz and each SONIC window at exactly ten frames.
- Only a correctly sized, allowed-sender, parsed, finite, successfully converted UDP datagram updates freshness, maximum arrival gap, or recovery readiness.
- Interpolated, held, and short-recovery-blend outputs never refresh source freshness and never count toward the ten-real-packet recovery gate.
- Interpolate continuous arrays linearly and quaternions with normalization plus hemisphere correction. Never extrapolate human motion.
- Drop obsolete burst backlog. Never raise the output rate above 50 Hz to replay queued motion.
- Preserve `btn_10=11` for ZeroLab entry and require one explicit `btn_10=12` before the first transition to SONIC.
- Enable automatic recovery only after the same state lifecycle has completed initial `BLENDING` and reached `ARMED`.
- After stale input, hold the human reference while SONIC continues inference with current robot proprioception; never freeze a motor frame.
- After any-duration stale gap, require ten new real valid packets, then blend the latched and newest human references for exactly 2.0 seconds without another button press.
- A new stale event during `REARMING` returns to `HOLD_REFERENCE`, resets the real-packet gate, and permits another automatic attempt after ten more real valid packets.
- Do not add a pose-difference gate, ACK channel, loss reconstruction, sender-clock estimate, or human-motion extrapolator.
- Preserve Normal, PD Brake, zero torque, orientation safety, policy exception handling, and state-exit priority over every recovery phase.
- Keep PICO acquisition, PICO readiness, PICO calibration, ordinary `sonic_teleop`, SONIC model assets, gains, action scale, joint limits, MuJoCo XML, hardware drivers, and CAN handling unchanged.
- The wireless branch manifest uses `allowed_sender: 192.168.89.171`. Preserve the known-good direct-wired version at commit `a82e5f4`, whose manifest test expects `192.168.1.52`.
- Use only new `build-wireless-auto-recovery/`, `install-wireless-auto-recovery/`, and `log-wireless-auto-recovery/` artifacts. Never commit or delete any existing `build-*`, `install-*`, or `log-*` directory.
- Do not start a hardware process until focused tests, the complete package suite, isolated build checks, and every MuJoCo dropout gate pass.

## File Structure

- Create `src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/resampler.py`: own playout kinds, immutable output/stat records, bounded sample storage, interpolation, hold, short recovery, and backlog accounting.
- Modify `src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/source_node.py`: validate new source parameters, make real packets authoritative, create source generations, build resampled ten-frame windows, publish fixed-rate freshness metadata, and log bounded statistics.
- Modify `src/bxi_example_py_elf3/mods/com.bxi.sonic/pico/pose_to_smpl_ref_bridge.py`: strictly validate and forward optional ZeroLab metadata while leaving metadata-free PICO messages unchanged.
- Modify `src/bxi_example_py_elf3/mods/com.bxi.sonic/reference_gate.py`: extend `SmplReferenceFrame` with optional source-authoritative freshness/readiness fields without changing reference interpolation.
- Modify `src/bxi_example_py_elf3/mods/com.bxi.sonic/policy.py`: preserve ZeroLab metadata across the source merger, reject previously seen old generations, compute freshness from the real receive timestamp, and expose recovery readiness.
- Modify `src/bxi_example_py_elf3/mods/com.bxi.sonic/state.py`: extend the narrow `SonicPolicy` protocol with `live_reference_recovery_ready()`.
- Modify `src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/state.py`: track whether initial ARM completed, start automatic rearm, use the separate two-second recovery duration, and make ARM presses informational after initial ARM.
- Modify `src/bxi_example_py_elf3/mods/com.bxi.sonic/plugin.py`: pass the new ZeroLab-only state parameters.
- Modify `src/bxi_example_py_elf3/mods/com.bxi.sonic/mod.yaml`: add exact source/state configuration and select the wireless sender.
- Modify `src/bxi_example_py_elf3/mods/com.bxi.sonic/README.md`: replace explicit recovery-ARM instructions with jitter, hold, and automatic recovery operations.
- Create `src/bxi_example_py_elf3/test/test_zerolab_resampler.py`: deterministic fake-clock tests for regular, jittered, burst, hold, quaternion, and short-recovery behavior.
- Modify `src/bxi_example_py_elf3/test/test_zerolab_pose_contract.py`: test real-packet authority, source generations, stale boundaries, recovery counts, and wire metadata.
- Modify `src/bxi_example_py_elf3/test/test_zerolab_lifecycle.py`: test one-output-per-tick publication, stale held output, transition log order, counters, and resource cleanup.
- Modify `src/bxi_example_py_elf3/test/test_sonic_ordered_playout.py`: test optional metadata forwarding, PICO compatibility, source-authoritative policy freshness, and old-generation rejection.
- Modify `src/bxi_example_py_elf3/test/test_sonic_reference_gate.py`: test metadata copying through hold and reference-space rearming.
- Modify `src/bxi_example_py_elf3/test/test_zerolab_arming.py`: test initial explicit ARM, automatic rearm, repeated stale, button behavior, phase durations, and emergency exits.
- Modify `src/bxi_example_py_elf3/test/test_zerolab_manifest.py`: lock the exact wireless source/state parameters and PICO isolation.

---

### Task 1: Implement the Pure ZeroLab Pose Resampler

**Files:**
- Create: `src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/resampler.py`
- Create: `src/bxi_example_py_elf3/test/test_zerolab_resampler.py`
- Read: `src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/converter.py`

**Interfaces:**
- Consumes: `ConvertedPoseFrame` with a strictly increasing local real `frame_index` and monotonic `receive_timestamp_ns`.
- Produces: `ZeroLabPoseResampler.observe(frame: ConvertedPoseFrame) -> bool`, `sample(now_ns: int) -> ResampledPose | None`, `mark_stale() -> None`, `reset() -> None`, and immutable `stats: ResamplerStats`.
- Wire-independent types:

```python
class PlayoutKind(IntEnum):
    REAL = 0
    INTERPOLATED = 1
    HELD = 2
    SHORT_RECOVERY_BLEND = 3


@dataclass(frozen=True)
class ResampledPose:
    frame: ConvertedPoseFrame
    kind: PlayoutKind
    latest_real_frame_index: int
    latest_real_receive_timestamp_ns: int


@dataclass(frozen=True)
class ResamplerStats:
    interpolated_output_frames: int = 0
    held_output_frames: int = 0
    dropped_backlog_frames: int = 0
```

- Generated `ResampledPose.frame.frame_index` is a consecutive 50 Hz output index. Generated `receive_timestamp_ns` is the playout target time; the authoritative real timestamp stays in `latest_real_receive_timestamp_ns`.

- [ ] **Step 1: Write failing regular-playout and interpolation tests**

Create these helpers, which make continuous fields equal to `value` and root quaternions use WXYZ form:

```python
def frame(index, timestamp_ns, value, *, root_wxyz=(1.0, 0.0, 0.0, 0.0)):
    return ConvertedPoseFrame(
        frame_index=index,
        receive_timestamp_ns=timestamp_ns,
        smpl_body_pose=np.full((21, 3), value, dtype=np.float32),
        smpl_joints=np.full((24, 3), value, dtype=np.float32),
        body_quat_w=np.asarray(root_wxyz, dtype=np.float32),
        joint_pos=np.full(29, value, dtype=np.float32),
    )


def make_resampler():
    return ZeroLabPoseResampler(
        jitter_buffer_seconds=0.08,
        short_recovery_blend_seconds=0.2,
        output_rate_hz=50.0,
    )


def test_regular_input_plays_at_target_delayed_by_80_ms():
    resampler = ZeroLabPoseResampler(
        jitter_buffer_seconds=0.08,
        short_recovery_blend_seconds=0.2,
        output_rate_hz=50.0,
    )
    for index in range(7):
        assert resampler.observe(frame(index, index * 20_000_000, float(index)))

    output = resampler.sample(120_000_000)  # target = 40 ms, real frame 2

    assert output.kind is PlayoutKind.REAL
    assert output.frame.frame_index == 0
    assert output.frame.receive_timestamp_ns == 40_000_000
    np.testing.assert_allclose(output.frame.smpl_joints, 2.0)
    assert output.latest_real_frame_index == 6
    assert output.latest_real_receive_timestamp_ns == 120_000_000


def test_jittered_samples_interpolate_continuous_fields():
    resampler = make_resampler()
    resampler.observe(frame(0, 0, 0.0))
    resampler.observe(frame(1, 40_000_000, 4.0))

    output = resampler.sample(100_000_000)  # target = 20 ms

    assert output.kind is PlayoutKind.INTERPOLATED
    np.testing.assert_allclose(output.frame.smpl_body_pose, 2.0)
    np.testing.assert_allclose(output.frame.smpl_joints, 2.0)
    np.testing.assert_allclose(output.frame.joint_pos, 2.0)
```

- [ ] **Step 2: Write failing quaternion, hold, backlog, and defensive-index tests**

```python
def test_quaternion_interpolation_corrects_hemisphere_and_normalizes():
    resampler = make_resampler()
    resampler.observe(frame(0, 0, 0.0, root_wxyz=(1.0, 0.0, 0.0, 0.0)))
    resampler.observe(frame(1, 40_000_000, 4.0,
                            root_wxyz=(-2**-0.5, 0.0, 0.0, -2**-0.5)))

    root = resampler.sample(100_000_000).frame.body_quat_w

    np.testing.assert_allclose(np.linalg.norm(root), 1.0, atol=1e-6)
    np.testing.assert_allclose(
        root, np.array([0.9238795, 0.0, 0.0, 0.3826834]), atol=1e-6
    )


def test_missing_right_bracket_holds_and_never_extrapolates():
    resampler = make_resampler()
    resampler.observe(frame(0, 0, 0.0))
    first = resampler.sample(80_000_000)
    held = resampler.sample(200_000_000)
    assert held.kind is PlayoutKind.HELD
    np.testing.assert_array_equal(held.frame.smpl_joints, first.frame.smpl_joints)
    assert resampler.stats.held_output_frames == 1


def test_burst_discards_obsolete_brackets_without_output_catchup():
    resampler = make_resampler()
    for index in range(100):
        resampler.observe(frame(index, index * 20_000_000, float(index)))
    assert resampler.sample(2_000_000_000).frame.frame_index == 0
    assert resampler.stats.dropped_backlog_frames >= 95


def test_duplicate_backward_and_nonmonotonic_inputs_are_defensive_errors():
    resampler = make_resampler()
    assert resampler.observe(frame(3, 60_000_000, 3.0))
    assert resampler.observe(frame(3, 60_000_000, 3.0)) is False
    with pytest.raises(ValueError, match="frame_index"):
        resampler.observe(frame(2, 80_000_000, 2.0))
    resampler.reset()
    resampler.observe(frame(4, 100_000_000, 4.0))
    with pytest.raises(ValueError, match="receive_timestamp_ns"):
        resampler.observe(frame(5, 90_000_000, 5.0))
```

- [ ] **Step 3: Write failing short-recovery and stale-reset tests**

Use the resampler's actual held output as the start of the catch-up, not the oldest newly arrived packet:

```python
def test_short_gap_recovery_blends_from_pose_actually_held():
    resampler = make_resampler()
    resampler.observe(frame(0, 0, 0.0))
    resampler.observe(frame(1, 20_000_000, 1.0))
    held = resampler.sample(200_000_000)
    assert held.kind is PlayoutKind.HELD
    resampler.observe(frame(2, 140_000_000, 10.0))
    resampler.observe(frame(3, 180_000_000, 14.0))

    start = resampler.sample(220_000_000)
    middle = resampler.sample(320_000_000)

    assert start.kind is PlayoutKind.SHORT_RECOVERY_BLEND
    assert middle.kind is PlayoutKind.SHORT_RECOVERY_BLEND
    np.testing.assert_allclose(start.frame.smpl_joints, held.frame.smpl_joints)
    assert np.all(middle.frame.smpl_joints > start.frame.smpl_joints)


def test_mark_stale_clears_brackets_but_preserves_last_output_for_stale_hold():
    resampler = make_resampler()
    resampler.observe(frame(0, 0, 3.0))
    live = resampler.sample(80_000_000)
    resampler.mark_stale()
    stale = resampler.sample(100_000_000)
    assert stale.kind is PlayoutKind.HELD
    np.testing.assert_array_equal(stale.frame.smpl_joints, live.frame.smpl_joints)


def test_post_stale_input_does_not_use_short_recovery_blend():
    resampler = make_resampler()
    resampler.observe(frame(0, 0, 1.0))
    resampler.sample(80_000_000)
    resampler.mark_stale()
    resampler.observe(frame(1, 1_000_000_000, 10.0))
    resampler.observe(frame(2, 1_040_000_000, 14.0))
    recovered = resampler.sample(1_100_000_000)
    assert recovered.kind in (PlayoutKind.REAL, PlayoutKind.INTERPOLATED)
```

- [ ] **Step 4: Run the new test module and verify RED**

From `src/bxi_example_py_elf3`:

```bash
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
python3 -m pytest -q test/test_zerolab_resampler.py
```

Expected: collection fails because `zerolab.resampler` does not exist.

- [ ] **Step 5: Implement the minimal pure component**

Implement strict finite/shape validation by reusing the converted-frame shapes, a deque ordered by real receive time, `target_ns = now_ns - jitter_buffer_ns`, hemisphere-corrected normalized quaternion interpolation, one consecutive output index per `sample()` call, and immutable copied output arrays. Use this catch-up state:

```python
self._catchup_start: ConvertedPoseFrame | None = None
self._catchup_started_ns: int | None = None

alpha = min(1.0, (now_ns - self._catchup_started_ns) / self._short_blend_ns)
pose = _interpolate_frame(self._catchup_start, target_pose, alpha)
kind = PlayoutKind.SHORT_RECOVERY_BLEND if alpha < 1.0 else target_kind
```

Delete all real samples strictly older than the left bracket after each sample. If more than two obsolete samples exist before the selected bracket, count each discarded sample. `mark_stale()` clears the real deque and catch-up state, retains the last emitted pose for stale diagnostics, and sets a one-shot `suppress_short_recovery` flag. The first safely bracketed post-stale output clears that flag and is `REAL`/`INTERPOLATED`, never `SHORT_RECOVERY_BLEND`; the reference gate owns the long two-second recovery. `reset()` clears everything including the last pose, output index, flags, and statistics.

- [ ] **Step 6: Run focused tests and commit**

```bash
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
python3 -m pytest -q test/test_zerolab_resampler.py
git add \
  mods/com.bxi.sonic/zerolab/resampler.py \
  test/test_zerolab_resampler.py
git commit -m "feat: add ZeroLab pose jitter resampler"
```

Expected: all resampler tests pass; the commit contains only the new component and its tests.

---

### Task 2: Make Real UDP Packets Authoritative in the Source Core

**Files:**
- Modify: `src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/source_node.py:32-252,517-529`
- Modify: `src/bxi_example_py_elf3/test/test_zerolab_pose_contract.py`
- Modify: `src/bxi_example_py_elf3/test/test_zerolab_lifecycle.py:20-103`

**Interfaces:**
- Consumes: Task 1 `ZeroLabPoseResampler`.
- Produces: `ZeroLabSourceCore.accept(packet: ZeroLabPacket) -> bool`, `sample(now_ns: int) -> dict[str, np.ndarray] | None`, `check_stale(now_ns: int) -> bool`, `consume_stale_event() -> bool`, and immutable `stats: ZeroLabSourceStats`.
- Adds source fields with exact dtypes:

```python
SOURCE_METADATA_DTYPES = {
    "source_generation": np.int64,
    "latest_real_frame_index": np.int64,
    "latest_real_receive_timestamp_ns": np.int64,
    "real_valid_frames_in_generation": np.int32,
    "real_stream_ready": np.uint8,
    "playout_kind": np.uint8,
    "source_stale": np.uint8,
}
```

- `source_generation` is a positive random epoch from `new_stream_epoch(previous)`. It changes at construction and every stale transition. Real-frame readiness is capped at `recovery_real_frames` and is false until both the real count and output window are ready.

Use these concrete test helpers throughout this task:

```python
def make_core(*, generations=(101, 202, 303)):
    values = iter(generations)
    return ZeroLabSourceCore(
        ZeroLabMotionConverter(),
        resampler=ZeroLabPoseResampler(
            jitter_buffer_seconds=0.08,
            short_recovery_blend_seconds=0.2,
            output_rate_hz=50.0,
        ),
        window_frames=10,
        stale_seconds=0.5,
        recovery_real_frames=10,
        generation_factory=lambda _previous=None: next(values),
    )


def core_with_fake_generation(generation):
    return make_core(generations=(generation, generation + 1))


def ready_core(*, generations=(101, 202)):
    core = make_core(generations=generations)
    fields = None
    for index in range(10):
        core.accept(identity_packet(index, timestamp_ns=index * 20_000_000))
        fields = core.sample(80_000_000 + index * 20_000_000)
    assert fields is not None
    assert int(fields["real_stream_ready"][0]) == 1
    return core


def stale_core():
    core = ready_core()
    assert core.check_stale(
        core.latest_real_receive_timestamp_ns + 500_000_001
    )
    return core


def collect_next_window(core, *, start_ns):
    now_ns = start_ns
    fields = None
    for _ in range(10):
        fields = core.sample(now_ns)
        now_ns += 20_000_000
    assert fields is not None
    return fields
```

- [ ] **Step 1: Extend parameter-validation tests and verify RED**

Add the new defaults to every `source_context()` fixture and assert strict rejection:

```python
@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("jitter_buffer_seconds", 0.0),
        ("jitter_buffer_seconds", float("nan")),
        ("short_recovery_blend_seconds", -0.1),
        ("short_recovery_blend_seconds", True),
        ("recovery_real_frames", 0),
        ("recovery_real_frames", True),
    ],
)
def test_source_rejects_invalid_resampling_params(name, value):
    with pytest.raises(ValueError, match=name):
        validate_source_params({name: value})
```

Run:

```bash
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
python3 -m pytest -q \
  test/test_zerolab_lifecycle.py::test_source_rejects_invalid_resampling_params
```

Expected: FAIL because the parameters are unknown.

- [ ] **Step 2: Write source-authority and exact stale-boundary tests**

Replace direct `accept()`-returns-window assumptions with `accept()` followed by `sample()` and add:

```python
def test_synthetic_outputs_do_not_increment_real_readiness():
    core = core_with_fake_generation(101)
    for index in range(9):
        assert core.accept(identity_packet(index))
    for tick in range(20):
        core.sample(200_000_000 + tick * 20_000_000)
    assert core.real_valid_frames_in_generation == 9
    assert core.real_stream_ready is False


def test_tenth_real_packet_and_complete_output_window_set_ready_metadata():
    core = core_with_fake_generation(101)
    fields = None
    for index in range(10):
        assert core.accept(identity_packet(index))
        fields = core.sample(80_000_000 + index * 20_000_000)
    assert fields is not None
    assert int(fields["source_generation"][0]) == 101
    assert int(fields["real_valid_frames_in_generation"][0]) == 10
    assert int(fields["real_stream_ready"][0]) == 1


def test_stale_is_strictly_greater_than_half_second_and_changes_generation():
    core = make_core(generations=(101, 202))
    core.accept(identity_packet(0, timestamp_ns=1_000_000_000))
    assert core.check_stale(1_500_000_000) is False
    assert core.check_stale(1_500_000_001) is True
    assert core.source_generation == 202
    assert core.real_valid_frames_in_generation == 0
    assert core.stats.stale_events == 1
```

Also assert malformed packets never call `core.accept()` in the node test and a conversion exception leaves last-real time/count unchanged.

- [ ] **Step 3: Write stale-held and ten-real recovery metadata tests**

```python
def test_stale_hold_is_published_but_cannot_be_ready():
    core = ready_core(generations=(101, 202))
    stale_ns = core.latest_real_receive_timestamp_ns + 500_000_001
    core.check_stale(stale_ns)
    fields = collect_next_window(core, start_ns=stale_ns)
    assert int(fields["source_generation"][0]) != 101
    assert int(fields["playout_kind"][0]) == int(PlayoutKind.HELD)
    assert int(fields["source_stale"][0]) == 1
    assert int(fields["real_stream_ready"][0]) == 0


def test_recovery_requires_ten_real_valid_packets_in_new_generation():
    core = stale_core()
    for index in range(9):
        core.accept(identity_packet(100 + index, timestamp_ns=2_000_000_000 + index * 20_000_000))
        fields = core.sample(2_080_000_000 + index * 20_000_000)
        if fields is not None:
            assert int(fields["real_stream_ready"][0]) == 0
    core.accept(identity_packet(109, timestamp_ns=2_180_000_000))
    fields = core.sample(2_260_000_000)
    assert int(fields["real_valid_frames_in_generation"][0]) == 10
    assert int(fields["real_stream_ready"][0]) == 1
```

- [ ] **Step 4: Run the source-contract tests and verify RED**

```bash
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
python3 -m pytest -q \
  test/test_zerolab_pose_contract.py \
  test/test_zerolab_lifecycle.py -k 'param or source or stale or recovery'
```

Expected: FAIL because `ZeroLabSourceCore` lacks resampling, generations, metadata, and statistics.

- [ ] **Step 5: Implement the core and strict configuration**

Add these source defaults:

```python
"jitter_buffer_seconds": 0.08,
"short_recovery_blend_seconds": 0.2,
"recovery_real_frames": 10,
```

Implement `ZeroLabSourceStats(real_valid_packets, maximum_real_arrival_gap_ms, stale_events)` as an immutable dataclass. In `accept()`, detect a pre-arrival stale gap before conversion, convert, call `resampler.observe()`, then update the real timestamp, real count, total count, and maximum real arrival gap only after every prior step succeeds. In `_mark_stale()`, perform exactly:

```python
self._window.clear()
self._converter.mark_stale()
self._resampler.mark_stale()
self._source_generation = self._generation_factory(self._source_generation)
self._real_valid_frames_in_generation = 0
self._source_stale = True
self._stale_event_pending = True
```

In `sample()`, append only `resampled.frame` to `PoseChunkWindow`, then add the seven metadata arrays. `source_stale` is computed from real age on every call; held/interpolated output never changes the real timestamp or count.

- [ ] **Step 6: Run the focused suite and commit**

```bash
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
python3 -m pytest -q \
  test/test_zerolab_resampler.py \
  test/test_zerolab_pose_contract.py \
  test/test_zerolab_lifecycle.py -k 'param or source or stale or recovery'
git add \
  mods/com.bxi.sonic/zerolab/source_node.py \
  test/test_zerolab_pose_contract.py \
  test/test_zerolab_lifecycle.py
git commit -m "feat: make ZeroLab real packets authoritative"
```

Expected: focused tests pass and Task 1 remains green.

---

### Task 3: Publish Fixed-Rate Held Frames and Bounded Source Statistics

**Files:**
- Modify: `src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/source_node.py:294-529`
- Modify: `src/bxi_example_py_elf3/test/test_zerolab_lifecycle.py:164-327`

**Interfaces:**
- Consumes: Task 2 `ZeroLabSourceCore.accept()` and `sample()`.
- Produces: at most one pose message per ROS timer tick, including stale held messages, plus transition logs and a five-second bounded statistics log.
- Preserves: the receiver drains every available UDP packet before the single playout sample; recording still stores only real valid UDP packets.

Add these lifecycle helpers by extracting the existing repeated `__new__` node setup:

```python
def make_test_core():
    generations = iter((101, 202, 303))
    return ZeroLabSourceCore(
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


def make_controlled_node(monkeypatch):
    clock_ns = [0]
    monkeypatch.setattr(
        "zerolab.source_node.time.monotonic_ns", lambda: clock_ns[0]
    )
    receiver = ControlledReceiver(identity_payload())
    publisher = CapturingPublisher()
    logger = CapturingLogger()
    node = ZeroLabSourceNode.__new__(ZeroLabSourceNode)
    node._closed = False
    node._receiver = receiver
    node._converter = ZeroLabMotionConverter()
    node._core = make_test_core()
    node._publisher = publisher
    node._writer = None
    node._recording_enabled = False
    node._recording_error_reported = False
    node._invalid_packets = 0
    node._dropped_publications = 0
    node._stream_state = None
    node._last_stats_log_ns = 0
    node.get_logger = lambda: logger
    return node, receiver, publisher, clock_ns


def ready_controlled_node(monkeypatch):
    node, receiver, publisher, clock_ns = make_controlled_node(monkeypatch)
    for index in range(10):
        clock_ns[0] = 80_000_000 + index * 20_000_000
        receiver.queue(index, index * 20_000_000)
        node._tick()
    assert any(int(msg["real_stream_ready"][0]) for msg in publisher.messages)
    return node, receiver, publisher, clock_ns


def create_stale_then_ten_real_recovery(node, receiver, clock_ns):
    clock_ns[0] += 500_000_001
    node._tick()
    recovery_start = clock_ns[0] + 20_000_000
    for offset in range(10):
        timestamp_ns = recovery_start + offset * 20_000_000
        receiver.queue(100 + offset, timestamp_ns)
        clock_ns[0] = timestamp_ns + 80_000_000
        node._tick()
```

Tests must not import helpers from another test module.

- [ ] **Step 1: Replace lifecycle test doubles with metadata capture and write RED tests**

Make `CapturingPublisher.send()` store a deep copy of the entire field mapping. Initialize test nodes with `_last_stats_log_ns = 0`. Add:

```python
def test_node_drains_burst_but_publishes_at_most_once_per_tick(monkeypatch):
    node, receiver, publisher, clock = make_controlled_node(monkeypatch)
    for index in range(30):
        receiver.queue(index, index * 20_000_000)
    clock[0] = 600_000_000
    node._tick()
    assert len(publisher.messages) <= 1


def test_node_keeps_50_hz_stale_held_publication_with_stale_metadata(monkeypatch):
    node, receiver, publisher, clock = ready_controlled_node(monkeypatch)
    clock[0] += 500_000_001
    node._tick()
    for _ in range(12):
        clock[0] += 20_000_000
        node._tick()
    assert any(int(msg["source_stale"][0]) == 1 for msg in publisher.messages)
    assert all(int(msg["real_stream_ready"][0]) == 0
               for msg in publisher.messages if int(msg["source_stale"][0]))


def test_stale_log_precedes_recovery_ready_and_is_rate_limited(monkeypatch):
    node, receiver, publisher, clock = ready_controlled_node(monkeypatch)
    create_stale_then_ten_real_recovery(node, receiver, clock)
    stale = [event for event in node.get_logger().events if "input stale" in event[1]]
    ready = [event for event in node.get_logger().events if "stream ready" in event[1]]
    assert len(stale) == 1
    assert node.get_logger().events.index(stale[0]) < node.get_logger().events.index(ready[-1])
```

- [ ] **Step 2: Run lifecycle tests and verify RED**

```bash
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
python3 -m pytest -q test/test_zerolab_lifecycle.py
```

Expected: FAIL because `_tick()` still publishes only the last real-built window and stops on stale.

- [ ] **Step 3: Rework `_tick()` around one timer-owned sample**

Use this order exactly:

```python
for datagram in self._receiver.drain():
    packet = parse_and_validate(datagram)
    self._record_valid_packet_if_enabled(packet)
    self._core.accept(packet)

now_ns = time.monotonic_ns()
self._core.check_stale(now_ns)
became_stale = self._core.consume_stale_event()
fields = self._core.sample(now_ns)
if fields is not None:
    self._publisher.send(fields)
```

Keep the existing explicit parse exception branch instead of introducing a literal `parse_and_validate()` helper. Log `ZeroLab input stale; holding playout with source_stale=1` once per stale generation. Log `ZeroLab stream ready; frame=... generation=... real_frames=10` once when `real_stream_ready` first becomes true in a generation. Emit all source/resampler counters every five monotonic seconds and on transition, never every frame.

- [ ] **Step 4: Run lifecycle/resource tests and commit**

```bash
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
python3 -m pytest -q \
  test/test_zerolab_lifecycle.py \
  test/test_zerolab_recording.py \
  test/test_zerolab_udp_receiver.py
git add \
  mods/com.bxi.sonic/zerolab/source_node.py \
  test/test_zerolab_lifecycle.py
git commit -m "feat: publish bounded ZeroLab playout telemetry"
```

Expected: node lifecycle, repeated port release, recording, and receiver tests pass.

---

### Task 4: Carry Source Freshness Through the Bridge and Policy

**Files:**
- Modify: `src/bxi_example_py_elf3/mods/com.bxi.sonic/pico/pose_to_smpl_ref_bridge.py:243-350,381-530`
- Modify: `src/bxi_example_py_elf3/mods/com.bxi.sonic/reference_gate.py:13-49`
- Modify: `src/bxi_example_py_elf3/mods/com.bxi.sonic/policy.py:330-375,600-920,954-980`
- Modify: `src/bxi_example_py_elf3/mods/com.bxi.sonic/state.py:45-102`
- Modify: `src/bxi_example_py_elf3/test/test_sonic_ordered_playout.py`
- Modify: `src/bxi_example_py_elf3/test/test_sonic_reference_gate.py`

**Interfaces:**
- Consumes: the seven optional source metadata fields from Task 2.
- Produces: `SmplReferenceFrame.source_generation`, `latest_real_frame_index`, `latest_real_receive_timestamp_ns`, `real_valid_frames_in_generation`, `real_stream_ready`, and `playout_kind`, all optional for PICO compatibility; existing `source_stale` remains.
- Produces: `SonicTeleopPolicy.live_reference_recovery_ready(required_real_frames: int) -> bool` and the same method in `SonicPolicy` protocol.
- Freshness rule: if `source_generation` is present, use real timestamp/readiness/stale metadata; otherwise preserve current PICO/local-receive-time behavior exactly.

Add these exact test helpers to `test_sonic_ordered_playout.py`:

```python
ZERO_LAB_METADATA_NAMES = {
    "source_generation",
    "latest_real_frame_index",
    "latest_real_receive_timestamp_ns",
    "real_valid_frames_in_generation",
    "real_stream_ready",
    "playout_kind",
    "source_stale",
}


def _zerolab_metadata(
    *, generation, latest_real_ns=None, real_frames=10, ready=True,
    playout_kind=PlayoutKind.REAL, stale=False,
):
    return {
        "source_generation": np.asarray([generation], dtype=np.int64),
        "latest_real_frame_index": np.asarray([real_frames - 1], dtype=np.int64),
        "latest_real_receive_timestamp_ns": np.asarray(
            [time.monotonic_ns() if latest_real_ns is None else latest_real_ns],
            dtype=np.int64,
        ),
        "real_valid_frames_in_generation": np.asarray(
            [real_frames], dtype=np.int32
        ),
        "real_stream_ready": np.asarray([int(ready)], dtype=np.uint8),
        "playout_kind": np.asarray([int(playout_kind)], dtype=np.uint8),
        "source_stale": np.asarray([int(stale)], dtype=np.uint8),
    }


def _zerolab_source_fields(start, *, generation):
    fields = _source_fields(start, epoch=generation)
    fields.update(_zerolab_metadata(generation=generation))
    return fields
```

- [ ] **Step 1: Write bridge schema and PICO-compatibility tests**

```python
def test_bridge_preserves_complete_zerolab_source_metadata():
    incoming = _manager_fields(100)
    incoming.update(_zerolab_metadata(generation=77, real_frames=10, ready=True))
    chunk = bridge_module._parse_incoming_chunk(incoming)
    metadata = bridge_module._parse_optional_source_metadata(incoming)
    fields = bridge_module._source_chunk_fields(
        chunk,
        source_stream_epoch=77,
        received_monotonic_ns=999,
        source_metadata=metadata,
    )
    for name in ZERO_LAB_METADATA_NAMES:
        np.testing.assert_array_equal(fields[name], incoming[name])


def test_bridge_rejects_partial_or_invalid_zerolab_metadata():
    fields = _manager_fields(0)
    fields["source_generation"] = np.asarray([7], dtype=np.int64)
    with pytest.raises(ValueError, match="metadata"):
        bridge_module._parse_optional_source_metadata(fields)


def test_metadata_free_pico_schema_is_byte_compatible_in_fields():
    chunk = bridge_module._parse_incoming_chunk(_manager_fields(0))
    fields = bridge_module._source_chunk_fields(
        chunk, source_stream_epoch=7, received_monotonic_ns=123,
        source_metadata=None,
    )
    assert not any(name in fields for name in ZERO_LAB_METADATA_NAMES)
```

Use integer `PlayoutKind` wire values because `pack_pose_message()` supports numeric NumPy arrays, not strings.

- [ ] **Step 2: Write policy real-freshness and old-generation tests**

```python
def test_zerolab_freshness_uses_real_udp_timestamp_not_zmq_arrival(monkeypatch):
    clock_ns = [2_000_000_000]
    monkeypatch.setattr(policy_module.time, "monotonic_ns", lambda: clock_ns[0])
    policy = _make_policy(monkeypatch)
    fields = _source_fields(0, epoch=77)
    fields.update(_zerolab_metadata(
        generation=77,
        latest_real_ns=1_000_000_000,
        real_frames=10,
        ready=True,
        stale=False,
    ))
    policy._merge_source_fields(fields, received_mono=time.monotonic())
    policy.poll_reference()
    assert policy.has_fresh_live_reference(0.5) is False


def test_interpolated_and_held_chunks_cannot_make_recovery_ready(monkeypatch):
    policy = _make_policy(monkeypatch)
    fields = _source_fields(0, epoch=77)
    fields.update(_zerolab_metadata(
        generation=77, real_frames=9, ready=False,
        playout_kind=PlayoutKind.HELD,
    ))
    policy._merge_source_fields(fields, time.monotonic())
    assert policy.live_reference_recovery_ready(10) is False


def test_ten_real_fresh_frames_make_recovery_ready(monkeypatch):
    policy = _make_policy(monkeypatch)
    fields = _source_fields(0, epoch=77)
    fields.update(_zerolab_metadata(generation=77, real_frames=10, ready=True))
    policy._merge_source_fields(fields, time.monotonic())
    assert policy.live_reference_recovery_ready(10) is True


def test_policy_rejects_delayed_message_from_previously_seen_generation(monkeypatch):
    policy = _make_policy(monkeypatch)
    assert policy._merge_source_fields(_zerolab_source_fields(0, generation=11), 1.0)
    assert policy._merge_source_fields(_zerolab_source_fields(10, generation=22), 2.0)
    assert policy._merge_source_fields(_zerolab_source_fields(20, generation=11), 3.0) is False
    assert policy.source_generation == 22
```

- [ ] **Step 3: Write reference-copy metadata tests and run RED**

Extend the `frame()` fixture and assert `copy_smpl_reference()`, `hold()`, and `interpolate_smpl_reference()` retain the end frame's source metadata without aliasing arrays. Then run:

```bash
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
python3 -m pytest -q \
  test/test_sonic_reference_gate.py \
  test/test_sonic_ordered_playout.py
```

Expected: FAIL because the bridge, frame, policy, and protocol method do not contain the new contract.

- [ ] **Step 4: Implement optional metadata validation and forwarding**

Add `ZERO_LAB_SOURCE_METADATA` mapping each field to its exact scalar dtype and range. `_parse_optional_source_metadata()` returns `None` when all fields are absent, rejects a partial set, requires positive generation/timestamp, nonnegative real count, known `PlayoutKind`, and boolean 0/1 flags. For metadata-bearing messages, use `source_generation` as `source_stream_epoch`; maintain a set of seen generations and reject a different generation already in that set. For metadata-free PICO, retain `_source_stream_epoch` and every current readiness/counter rule.

- [ ] **Step 5: Implement source-authoritative policy freshness**

Store the newest validated metadata mapping alongside merged source chunks and add it to `stream_merger.build_smpl_ref()` fields before `_frame_from_fields()`. Implement:

```python
def has_fresh_live_reference(self, timeout_s=None):
    self.poll_reference()
    frame = self._live_reference_gate.observed_reference
    timeout = self.live_ref_timeout_s if timeout_s is None else float(timeout_s)
    if frame is None:
        return False
    if frame.source_generation is None:
        return time.monotonic() - self.latest_live_ref_time <= timeout
    age_ns = max(0, time.monotonic_ns() - frame.latest_real_receive_timestamp_ns)
    return bool(frame.real_stream_ready and not frame.source_stale
                and age_ns <= int(timeout * 1_000_000_000))


def live_reference_recovery_ready(self, required_real_frames):
    self.poll_reference()
    frame = self._live_reference_gate.observed_reference
    return bool(
        frame is not None
        and frame.source_generation is not None
        and frame.real_stream_ready
        and frame.real_valid_frames_in_generation >= int(required_real_frames)
        and self.has_fresh_live_reference()
    )
```

Reset seen generations and stored source metadata in `reset()`. Do not alter metadata-free PICO freshness.

- [ ] **Step 6: Run shared SONIC/PICO tests and commit**

```bash
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
python3 -m pytest -q \
  test/test_sonic_reference_gate.py \
  test/test_sonic_ordered_playout.py \
  test/test_sonic_python_runtime.py \
  test/test_zerolab_pose_contract.py
git add \
  mods/com.bxi.sonic/pico/pose_to_smpl_ref_bridge.py \
  mods/com.bxi.sonic/reference_gate.py \
  mods/com.bxi.sonic/policy.py \
  mods/com.bxi.sonic/state.py \
  test/test_sonic_reference_gate.py \
  test/test_sonic_ordered_playout.py
git commit -m "feat: propagate ZeroLab real-packet freshness"
```

Expected: shared SONIC and PICO regression tests pass, including the original one-way no-ACK schema test.

---

### Task 5: Automate Previously Armed Reference Recovery

**Files:**
- Modify: `src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/state.py`
- Modify: `src/bxi_example_py_elf3/mods/com.bxi.sonic/plugin.py:130-148`
- Modify: `src/bxi_example_py_elf3/mods/com.bxi.sonic/mod.yaml:176-316`
- Modify: `src/bxi_example_py_elf3/test/test_zerolab_arming.py`
- Modify: `src/bxi_example_py_elf3/test/test_zerolab_manifest.py`

**Interfaces:**
- Consumes: `policy.live_reference_recovery_ready(recovery_real_frames)` from Task 4 and existing reference-gate hold/rearm methods.
- Produces: `ZeroLabArmedTeleopState(..., auto_rearm_on_recovery=True, auto_rearm_blend_seconds=2.0, recovery_real_frames=10)`.
- Produces: read-only `auto_rearm_count: int` for tests and bounded state telemetry.
- Preserves: `arm_blend_seconds=2.0` independently for initial live-Normal-to-SONIC blending.

- [ ] **Step 1: Extend the fake policy and write automatic-recovery tests**

Add `recovery_ready = False`, `rearm_attempts = 0`, and:

```python
def live_reference_recovery_ready(self, required_real_frames):
    assert required_real_frames == 10
    return self.fresh and self.recovery_ready
```

Update `begin_live_reference_rearm()` to increment `rearm_attempts` only when it succeeds. Add:

```python
def auto_rearming_state():
    state, sonic, normal, ctx = reference_hold_state()
    sonic.fresh = True
    sonic.recovery_ready = True
    state.sample_running_frame(ctx, 0.02, advance=True)
    assert state.arm_phase is ZeroLabArmPhase.REARMING
    return state, sonic, normal, ctx
```

Replace the existing `rearming_state()` body with `return auto_rearming_state()` so all pre-existing phase, orientation, emergency, and completion tests exercise the automatic path. Update `test_stale_and_recovery_logs_are_emitted_once_per_stale_session` to set `recovery_ready=True` only after the held loop and assert exactly one `ZeroLab automatic recovery; ARM phase: REARMING` plus one completion message; remove the old `send btn_10=12 to rearm` assertion.

Replace the explicit-recovery expectation with:

```python
def test_previously_armed_hold_auto_rearms_after_ten_real_frames():
    state, sonic, _normal, ctx = reference_hold_state()
    sonic.fresh = True
    sonic.recovery_ready = False
    state.sample_running_frame(ctx, 0.02, advance=True)
    assert state.arm_phase is ZeroLabArmPhase.HOLD_REFERENCE

    sonic.recovery_ready = True
    state.sample_running_frame(ctx, 0.02, advance=True)
    assert state.arm_phase is ZeroLabArmPhase.REARMING
    assert sonic.rearming is True
```

Add:

```python
def test_initial_stream_never_auto_arms_without_btn10_12():
    state, sonic, _normal, ctx = prepared_state()
    sonic.fresh = True
    sonic.recovery_ready = True
    for _ in range(200):
        state.sample_running_frame(ctx, 0.02, advance=True)
    assert state.arm_phase is ZeroLabArmPhase.WAIT_ARM


def test_auto_rearm_uses_separate_two_second_duration():
    state, sonic, _normal, ctx = auto_rearming_state()
    state.sample_running_frame(ctx, 1.99, advance=True)
    assert state.arm_phase is ZeroLabArmPhase.REARMING
    state.sample_running_frame(ctx, 0.01, advance=True)
    assert state.arm_phase is ZeroLabArmPhase.ARMED


def test_stale_during_rearm_holds_then_retries_after_new_ready_generation():
    state, sonic, _normal, ctx = auto_rearming_state()
    sonic.fresh = False
    sonic.recovery_ready = False
    state.sample_running_frame(ctx, 0.1, advance=True)
    assert state.arm_phase is ZeroLabArmPhase.HOLD_REFERENCE
    sonic.fresh = True
    sonic.recovery_ready = True
    state.sample_running_frame(ctx, 0.02, advance=True)
    assert state.arm_phase is ZeroLabArmPhase.REARMING
    assert state.auto_rearm_count == 2


def test_btn10_12_does_not_restart_hold_rearming_or_armed_when_auto_enabled():
    for phase in ("hold_reference", "rearming", "armed"):
        state, sonic, _normal, ctx = state_in_phase(phase)
        before = state.arm_phase
        state.on_action(ctx, "arm_zerolab")
        assert state.arm_phase is before
```

- [ ] **Step 2: Write config validation and manifest tests**

Assert source and state values exactly:

```python
source = manifest["nodes"]["zerolab_source"]["params"]
assert source["allowed_sender"] == "192.168.89.171"
assert source["jitter_buffer_seconds"] == 0.08
assert source["short_recovery_blend_seconds"] == 0.2
assert source["stale_seconds"] == 0.5
assert source["recovery_real_frames"] == 10

state = manifest["states"]["sonic_zerolab"]["params"]
assert state["auto_rearm_on_recovery"] is True
assert state["auto_rearm_blend_seconds"] == 2.0
assert state["recovery_real_frames"] == 10
assert state["arm_blend_seconds"] == 2.0
```

Add constructor rejection tests for false numeric booleans, nonfinite/nonpositive recovery duration, and noninteger/less-than-one real-frame count.

- [ ] **Step 3: Run state/manifest tests and verify RED**

```bash
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
python3 -m pytest -q \
  test/test_zerolab_arming.py \
  test/test_zerolab_manifest.py
```

Expected: FAIL because recovery is still button-driven and the manifest still selects the wired sender.

- [ ] **Step 4: Implement automatic state transitions**

Track `_initial_arm_completed`, set it only when initial `BLENDING` reaches `ARMED`, and clear it in `on_prepare()`/`on_exit()`. In `HOLD_REFERENCE`, perform:

```python
if (
    self.auto_rearm_on_recovery
    and self._initial_arm_completed
    and self.policy.live_reference_recovery_ready(self.recovery_real_frames)
    and self.policy.begin_live_reference_rearm()
):
    self._blend_elapsed_s = 0.0
    self._auto_rearm_count += 1
    self._set_phase(
        ZeroLabArmPhase.REARMING,
        "ZeroLab automatic recovery; ARM phase: REARMING for "
        f"{self.auto_rearm_blend_seconds:.3f} s",
    )
```

Use `auto_rearm_blend_seconds` for REARMING progress/completion and `arm_blend_seconds` only for initial BLENDING. If stale during rearming, call `hold_live_reference()` before changing phase so the exact current interpolated reference is latched. With automatic recovery enabled, ARM actions in `HOLD_REFERENCE`, `REARMING`, and `ARMED` log once-bounded informational text and do not change timers or phase. Increment `auto_rearm_count` only after `begin_live_reference_rearm()` succeeds.

When `auto_rearm_on_recovery` is false, preserve the existing explicit fresh-reference `btn_10=12` recovery path. The shipped wireless manifest is true, so its post-initial ARM presses are informational only.

- [ ] **Step 5: Wire exact parameters and wireless sender**

In `plugin.py`, pass three state params explicitly. In `mod.yaml`, add the exact source/state values from Global Constraints and change only the ZeroLab source sender from `192.168.1.52` to `192.168.89.171`. Do not change PICO node/state params or events/routes.

- [ ] **Step 6: Run state, manifest, lifecycle tests and commit**

```bash
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
python3 -m pytest -q \
  test/test_zerolab_arming.py \
  test/test_zerolab_manifest.py \
  test/test_zerolab_lifecycle.py
git add \
  mods/com.bxi.sonic/zerolab/state.py \
  mods/com.bxi.sonic/plugin.py \
  mods/com.bxi.sonic/mod.yaml \
  test/test_zerolab_arming.py \
  test/test_zerolab_manifest.py
git commit -m "feat: auto recover armed ZeroLab references"
```

Expected: state/manifest/lifecycle tests pass; initial ARM remains explicit; PICO manifest assertions remain unchanged.

---

### Task 6: Add the Deterministic Dropout Matrix and Operator Documentation

**Files:**
- Modify: `src/bxi_example_py_elf3/test/test_zerolab_pose_contract.py`
- Modify: `src/bxi_example_py_elf3/test/test_zerolab_arming.py`
- Modify: `src/bxi_example_py_elf3/mods/com.bxi.sonic/README.md:540-670`

**Interfaces:**
- Consumes: Tasks 1-5 production interfaces.
- Produces: deterministic source-boundary evidence for `0.10`, `0.49`, `0.51`, `2.0`, and `30.0` second gaps plus copyable operating guidance. Task 5 tests prove the associated control phases.
- These tests use fake monotonic timestamps and never open robot hardware or a real network interface.

- [ ] **Step 1: Write the source and state dropout matrices**

Add to `test_zerolab_pose_contract.py` using its existing `identity_packet()` and Task 2 `ready_core()` helpers:

```python
@pytest.mark.parametrize("gap_s", [0.10, 0.49])
def test_short_gap_does_not_cross_source_stale_barrier(gap_s):
    core = ready_core(generations=(101, 202))
    last_real_ns = core.latest_real_receive_timestamp_ns
    now_ns = last_real_ns + int(gap_s * 1.0e9)
    assert core.check_stale(now_ns) is False
    fields = core.sample(now_ns)
    assert fields is not None
    assert int(fields["source_generation"][0]) == 101
    assert int(fields["source_stale"][0]) == 0


@pytest.mark.parametrize("gap_s", [0.51, 2.0, 30.0])
def test_long_gap_crosses_one_stale_generation_barrier(gap_s):
    core = ready_core(generations=(101, 202))
    last_real_ns = core.latest_real_receive_timestamp_ns
    assert core.check_stale(last_real_ns + int(gap_s * 1.0e9)) is True
    assert core.source_generation == 202
    assert core.real_valid_frames_in_generation == 0
    assert core.stats.stale_events == 1
```

Add to `test_zerolab_arming.py` using its Task 5 fake-policy fields:

```python
@pytest.mark.parametrize("real_frames", range(10))
def test_hold_does_not_auto_rearm_before_ten_real_frames(real_frames):
    state, sonic, _normal, ctx = reference_hold_state()
    sonic.fresh = True
    sonic.recovery_ready = real_frames >= 10
    state.sample_running_frame(ctx, 0.02, advance=True)
    assert state.arm_phase is ZeroLabArmPhase.HOLD_REFERENCE


def test_tenth_real_frame_auto_rearms_without_second_arm_action():
    state, sonic, _normal, ctx = reference_hold_state()
    sonic.fresh = True
    sonic.recovery_ready = True
    state.sample_running_frame(ctx, 0.02, advance=True)
    assert state.arm_phase is ZeroLabArmPhase.REARMING
    assert sonic.rearm_attempts == 1
```

- [ ] **Step 2: Run the deterministic dropout matrix**

```bash
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
python3 -m pytest -q \
  test/test_zerolab_pose_contract.py \
  test/test_zerolab_arming.py
```

Expected: all source-boundary and state recovery tests pass.

- [ ] **Step 3: Update the ZeroLab README workflow**

Replace explicit recovery ARM text with this state flow and exact caveat:

```text
vendor N-pose -> operator neutral -> btn_10=11 -> WAIT_STREAM
-> WAIT_ARM -> initial btn_10=12 -> 2 s BLENDING -> ARMED

gap <= 0.5 s: 80 ms buffered playout -> held reference if needed
-> automatic 0.2 s short recovery -> remains ARMED

gap > 0.5 s: HOLD_REFERENCE -> 10 new real valid UDP packets
-> automatic 2 s REARMING -> ARMED
```

Document that no second button is required, no pose-difference gate exists, the human must return to neutral before deliberately restoring a long-disconnected sender, held/interpolated messages do not prove network health, and unconditional automatic recovery is forbidden without support rig, emergency stop, and safety observer.

For this wireless branch, replace the direct-wired endpoint text with: Windows sender `192.168.89.171` sends 992-byte UDP to robot wireless address `192.168.88.172:18000`. State separately that direct wired remains available at `a82e5f4` with sender `192.168.1.52` and receiver `192.168.1.51:18000`; do not present the two routes as simultaneously active.

- [ ] **Step 4: Run all focused tests and commit**

```bash
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
python3 -m pytest -q \
  test/test_zerolab_protocol.py \
  test/test_zerolab_udp_receiver.py \
  test/test_zerolab_converter.py \
  test/test_zerolab_recording.py \
  test/test_zerolab_resampler.py \
  test/test_zerolab_pose_contract.py \
  test/test_zerolab_lifecycle.py \
  test/test_zerolab_manifest.py \
  test/test_zerolab_arming.py \
  test/test_sonic_reference_gate.py \
  test/test_sonic_ordered_playout.py \
  test/test_sonic_python_runtime.py
git add \
  test/test_zerolab_pose_contract.py \
  test/test_zerolab_arming.py \
  mods/com.bxi.sonic/README.md
git commit -m "test: cover ZeroLab wireless dropout recovery"
```

Expected: all focused tests pass and documentation matches automatic recovery.

---

### Task 7: Full Regression, Isolated Build, and MuJoCo Dropout Gate

**Files:**
- Verify only: all tracked implementation files from Tasks 1-6.
- Create runtime-only: `build-wireless-auto-recovery/`, `install-wireless-auto-recovery/`, `log-wireless-auto-recovery/`.
- Evidence only: `/tmp/zerolab-wireless-auto-recovery-pytest.log`, `/tmp/zerolab-wireless-auto-recovery-colcon.log`, `/tmp/zerolab-wireless-auto-recovery-mujoco.log`.

**Interfaces:**
- Produces: a simulation-tested isolated install. It does not deploy to or launch robot hardware.

- [ ] **Step 1: Run the complete package test suite**

From `src/bxi_example_py_elf3`:

```bash
set -o pipefail
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
python3 -m pytest -q test 2>&1 | \
  tee /tmp/zerolab-wireless-auto-recovery-pytest.log
```

Expected: all tests pass; no skip hides ZeroLab resampling, metadata, state, or PICO coverage.

- [ ] **Step 2: Build only the isolated wireless candidate**

From the worktree root:

```bash
set -eo pipefail
source /opt/ros/humble/setup.bash
source /home/fazepurple/ros2_ws/bxi_ros2_pkg/setup.bash
colcon --log-base log-wireless-auto-recovery build \
  --merge-install \
  --base-paths src \
  --packages-select bxi_example_py_elf3 \
  --allow-overriding bxi_example_py_elf3 \
  --build-base build-wireless-auto-recovery \
  --install-base install-wireless-auto-recovery \
  --cmake-args -DCMAKE_BUILD_TYPE=Release 2>&1 | \
  tee /tmp/zerolab-wireless-auto-recovery-colcon.log
source install-wireless-auto-recovery/setup.bash
test "$(ros2 pkg prefix bxi_example_py_elf3)" = \
  "$PWD/install-wireless-auto-recovery"
grep -nE \
  'allowed_sender: 192.168.89.171|jitter_buffer_seconds: 0.08|short_recovery_blend_seconds: 0.2|auto_rearm_on_recovery: true|auto_rearm_blend_seconds: 2.0|recovery_real_frames: 10' \
  install-wireless-auto-recovery/share/bxi_example_py_elf3/mods/com.bxi.sonic/mod.yaml
```

Expected: package prefix is the isolated install and all six exact values appear in the installed manifest.

- [ ] **Step 3: Launch MuJoCo with no hardware process**

Terminal 1:

```bash
set -eo pipefail
source /opt/ros/humble/setup.bash
source /home/fazepurple/ros2_ws/bxi_ros2_pkg/setup.bash
source "$PWD/install-wireless-auto-recovery/setup.bash"
export ROS_DOMAIN_ID=42
if pgrep -af '[h]ardware_elf3|[r]os2 launch.*example_demo_hw'; then
  echo 'STOP: hardware process detected'
  exit 1
fi
ros2 launch bxi_example_py_elf3 example_demo.launch.py 2>&1 | \
  tee /tmp/zerolab-wireless-auto-recovery-mujoco.log
```

Terminal 2 sources the same three setup files and `ROS_DOMAIN_ID=42`, then defines:

```bash
pulse_btn10() {
  ros2 topic pub --once /motion_commands \
    communication/msg/MotionCommands "{btn_10: $1}"
  ros2 topic pub --once /motion_commands \
    communication/msg/MotionCommands '{}'
}
pulse_btn10 11
```

Wait for `WAIT_ARM`, keep the operator neutral, then run `pulse_btn10 12` once. Expected: `WAIT_STREAM -> WAIT_ARM -> BLENDING -> ARMED`. No later trial sends `btn_10=12` again.

- [ ] **Step 4: Exercise all deterministic dropout gates**

Pause the ZeroLab sender separately for `0.10`, `0.49`, `0.51`, `2.0`, and `30.0` seconds. Return the operator to neutral before restoring every stale trial. After each trial inspect:

```bash
grep -E \
  'short input hold|short input recovered|HOLD_REFERENCE|recovery ready|REARMING|automatic recovery complete|input stale' \
  /tmp/zerolab-wireless-auto-recovery-mujoco.log | tail -n 120
```

Expected:

- `0.10` and `0.49` seconds remain `ARMED`, show held/short-recovery behavior, and never show `HOLD_REFERENCE` for that trial.
- `0.51`, `2.0`, and `30.0` seconds show `HOLD_REFERENCE -> recovery ready -> REARMING -> ARMED` without a second ARM pulse.
- The robot never replays an old burst, snaps to the recovered pose, freezes a motor frame, falls, or loses emergency routes.

- [ ] **Step 5: Force stale during REARMING**

During one two-second `REARMING`, stop the sender for more than 0.5 seconds, then restore it with the operator neutral. Expected: immediate return to `HOLD_REFERENCE`, no partial-resume continuation, ten new real packets, and a new automatic two-second `REARMING`.

- [ ] **Step 6: Verify emergency interruption and clean exit**

From Terminal 2, while in `HOLD_REFERENCE` or `REARMING`:

```bash
ros2 topic pub --once /motion_commands communication/msg/MotionCommands '{btn_3: 1}'
ros2 topic pub --once /motion_commands communication/msg/MotionCommands '{}'
sleep 2
ros2 topic pub --once /motion_commands communication/msg/MotionCommands '{btn_1: 1}'
ros2 topic pub --once /motion_commands communication/msg/MotionCommands '{}'
```

Expected: PD Brake preempts ZeroLab immediately, Normal remains reachable, and re-entering ZeroLab starts at `WAIT_STREAM` with automatic recovery eligibility cleared.

- [ ] **Step 7: Run final repository checks**

```bash
git diff --check
git status --short
git log --oneline -10
```

Expected: no tracked uncommitted implementation changes; only known/new isolated runtime directories may be untracked. Stop here if any MuJoCo condition failed.

---

### Task 8: Gated Real-Hardware Candidate and Acceptance

**Files:**
- Runtime candidate root: `/home/bxi/zerolab-wireless-auto-recovery-20260824`
- Runtime-only build: `build-wireless-auto-recovery/`, `install-wireless-auto-recovery/`, `log-wireless-auto-recovery/`
- Runtime evidence: `/tmp/zerolab-wireless-auto-recovery-hw.log`

**Interfaces:**
- Consumes: the exact reviewed commit that passed Task 7 and live wireless UDP from `192.168.89.171` to the robot receiver on port `18000`.
- Produces: supervised real-hardware evidence only. It does not push, merge, alter firmware, suppress CAN faults, or change the 0.5-second stale threshold.

- [ ] **Step 1: Confirm the hardware gate and candidate identity**

On the robot as `bxi`, do not continue unless the support rig/fall protection, physical emergency stop, and separate safety observer are present:

```bash
cd /home/bxi/zerolab-wireless-auto-recovery-20260824
git status --short
git rev-parse HEAD
pgrep -af '[h]ardware_elf3|[b]xi_example_py_elf3_demo|[z]erolab_source' \
  || echo 'CONTROLLERS_STOPPED=PASS'
```

Expected: the commit equals the Task 7 reviewed commit, tracked status is clean, and no controller exists. If a controller is listed, stop and have its current operator shut it down normally.

- [ ] **Step 2: Re-run focused tests and build on the robot**

```bash
source /opt/ros/humble/setup.bash
source /opt/bxi/bxi_ros2_pkg/setup.bash
source /home/bxi/bxi_ws/bxi_rl_controller_ros2_example/install/setup.bash
cd /home/bxi/zerolab-wireless-auto-recovery-20260824/src/bxi_example_py_elf3
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
python3 -m pytest -q \
  test/test_zerolab_resampler.py \
  test/test_zerolab_pose_contract.py \
  test/test_zerolab_lifecycle.py \
  test/test_zerolab_manifest.py \
  test/test_zerolab_arming.py \
  test/test_sonic_reference_gate.py \
  test/test_sonic_ordered_playout.py
cd /home/bxi/zerolab-wireless-auto-recovery-20260824
colcon --log-base log-wireless-auto-recovery build \
  --merge-install \
  --base-paths src \
  --packages-select bxi_example_py_elf3 \
  --allow-overriding bxi_example_py_elf3 \
  --build-base build-wireless-auto-recovery \
  --install-base install-wireless-auto-recovery \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
```

Expected: focused tests and build pass. Any failure blocks hardware launch.

- [ ] **Step 3: Verify installed identity and live wireless packet quality before motors**

```bash
source /home/bxi/zerolab-wireless-auto-recovery-20260824/install-wireless-auto-recovery/setup.bash
test "$(ros2 pkg prefix bxi_example_py_elf3)" = \
  /home/bxi/zerolab-wireless-auto-recovery-20260824/install-wireless-auto-recovery
MANIFEST=/home/bxi/zerolab-wireless-auto-recovery-20260824/install-wireless-auto-recovery/share/bxi_example_py_elf3/mods/com.bxi.sonic/mod.yaml
grep -nE \
  'allowed_sender: 192.168.89.171|jitter_buffer_seconds: 0.08|stale_seconds: 0.5|auto_rearm_on_recovery: true|auto_rearm_blend_seconds: 2.0|recovery_real_frames: 10' \
  "$MANIFEST"
ip route get 192.168.89.171
timeout 10 tcpdump -ni any -nn -tt -c 20 \
  'udp and src host 192.168.89.171 and dst port 18000'
```

Expected: exact install prefix/configuration, a valid route, 992-byte UDP packets, and no multi-second gap during preflight. No packet blocks hardware launch.

- [ ] **Step 4: Start the only hardware stack in root terminal 1**

```bash
sudo -i
source /opt/ros/humble/setup.bash
source /opt/bxi/bxi_ros2_pkg/setup.bash
source /home/bxi/bxi_ws/bxi_rl_controller_ros2_example/install/setup.bash
source /home/bxi/zerolab-wireless-auto-recovery-20260824/install-wireless-auto-recovery/setup.bash
export ROS_DOMAIN_ID=42
test "$(ros2 pkg prefix bxi_example_py_elf3)" = \
  /home/bxi/zerolab-wireless-auto-recovery-20260824/install-wireless-auto-recovery
cd /home/bxi/zerolab-wireless-auto-recovery-20260824
ros2 launch bxi_example_py_elf3 example_demo_hw.launch.py 2>&1 | \
  tee /tmp/zerolab-wireless-auto-recovery-hw.log
```

Expected: exactly one hardware driver and one candidate controller run. Keep this terminal visible.

- [ ] **Step 5: Prepare root terminal 2 and emergency command**

```bash
sudo -i
source /opt/ros/humble/setup.bash
source /opt/bxi/bxi_ros2_pkg/setup.bash
source /home/bxi/bxi_ws/bxi_rl_controller_ros2_example/install/setup.bash
source /home/bxi/zerolab-wireless-auto-recovery-20260824/install-wireless-auto-recovery/setup.bash
export ROS_DOMAIN_ID=42
```

Keep this command ready:

```bash
ros2 topic pub --once /motion_commands communication/msg/MotionCommands '{btn_3: 1}'
ros2 topic pub --once /motion_commands communication/msg/MotionCommands '{}'
```

- [ ] **Step 6: Enter stable Normal, ZeroLab, and one explicit ARM**

```bash
ros2 topic pub --once /motion_commands communication/msg/MotionCommands '{btn_3: 1}'
ros2 topic pub --once /motion_commands communication/msg/MotionCommands '{}'
sleep 3
ros2 topic pub --once /motion_commands communication/msg/MotionCommands '{btn_1: 1}'
ros2 topic pub --once /motion_commands communication/msg/MotionCommands '{}'
sleep 5
ros2 topic pub --once /motion_commands communication/msg/MotionCommands '{btn_10: 11}'
ros2 topic pub --once /motion_commands communication/msg/MotionCommands '{}'
```

Wait for `WAIT_ARM`. With the operator neutral and safety observer ready, send exactly one initial ARM pulse:

```bash
ros2 topic pub --once /motion_commands communication/msg/MotionCommands '{btn_10: 12}'
ros2 topic pub --once /motion_commands communication/msg/MotionCommands '{}'
```

Expected: two-second `BLENDING -> ARMED`, stable neutral hold, then only one small single-arm motion. Any unexpected whole-body motion, wrong direction, balance loss, policy exception, or new hardware fault requires immediate PD Brake and termination.

- [ ] **Step 7: Run only the staged hardware recovery checks**

First hold neutral through one controlled 0.10-second interruption, then one 0.49-second interruption. Only if both remain stable, stop the sender for 0.51 seconds while the operator returns to and remains neutral; restore it without sending another ARM pulse.

Expected: short cases remain `ARMED`; the stale case logs `HOLD_REFERENCE -> recovery ready -> REARMING -> ARMED`, uses a visibly smooth two-second recovery, and never snaps or replays old motion. Do not test a 2-second or 30-second hardware dropout in this first acceptance run.

- [ ] **Step 8: Exit and verify cleanup**

```bash
ros2 topic pub --once /motion_commands communication/msg/MotionCommands '{btn_1: 1}'
ros2 topic pub --once /motion_commands communication/msg/MotionCommands '{}'
```

Press `Ctrl-C` once in terminal 1, then verify:

```bash
pgrep -af '[h]ardware_elf3|[b]xi_example_py_elf3_demo|[z]erolab_source' \
  || echo 'CONTROLLERS_STOPPED=PASS'
ss -H -lunp 'sport = :18000'
ss -H -ltnp 'sport = :5558'
ss -H -ltnp 'sport = :5557'
```

Expected: controller processes and all three listeners are gone. Preserve `/tmp/zerolab-wireless-auto-recovery-hw.log` with the test notes; do not declare hardware acceptance without the safety observer's confirmation.

## Completion Evidence

- Task commits show one independently reviewable change per component boundary.
- `test/test_zerolab_resampler.py` proves interpolation, hemisphere correction, no extrapolation, hold, short recovery, backlog drop, and defensive input handling.
- Source/bridge/policy tests prove synthetic frames cannot refresh real-packet freshness or recovery readiness.
- State tests prove initial ARM stays explicit and every previously armed stale generation can automatically rearm only after ten real packets.
- The complete package suite passes.
- The isolated installed manifest contains the exact wireless sender and six timing/recovery values.
- MuJoCo evidence covers `0.10`, `0.49`, `0.51`, `2.0`, and `30.0` second gaps plus stale-during-rearming and emergency interruption.
- Real-hardware evidence, if authorized after MuJoCo, covers neutral hold, one small arm motion, short interruptions, one controlled stale interruption, automatic smooth recovery, and clean shutdown.
