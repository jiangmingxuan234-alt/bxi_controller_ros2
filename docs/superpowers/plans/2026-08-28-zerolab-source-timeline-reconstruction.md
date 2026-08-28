# ZeroLab Source Timeline Reconstruction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent UDP packet bursts from compressing multiple ZeroLab body samples into nearly the same resampling timestamp while the vendor protocol still lacks capture timestamps and source sequence numbers.

**Architecture:** Keep each packet's robot receive timestamp authoritative for stale detection, source age, recording, and network diagnostics. Add a PC-side burst timeline reconstructor that only redistributes valid packets drained together: it anchors the newest packet to its actual arrival time and spaces earlier burst members backward at the configured 50 Hz source cadence, then passes those reconstructed timestamps to the existing jitter resampler. This is explicitly a bounded fallback until the sender provides capture timestamps and source sequence numbers.

**Tech Stack:** Python 3, NumPy, pytest, ROS 2 Humble, existing ZeroLab UDP/source/resampler pipeline.

## Global Constraints

- Preserve the existing `zerolab-jitter-80ms` and `zerolab-jitter-40ms` Git tags.
- Keep `jitter_buffer_seconds: 0.04`, ARM behavior, stale threshold, recovery behavior, pose conversion, and SONIC policy unchanged.
- Never substitute reconstructed sample time for actual receive time in freshness or stale decisions.
- Reconstructed timestamps must be strictly increasing and must not exceed the newest actual receive timestamp in a drained batch.
- Do not claim equivalence to sender capture timestamps; this fallback cannot identify packets lost before the robot.

---

### Task 1: Reconstruct timestamps inside UDP bursts

**Files:**
- Create: `src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/timeline.py`
- Test: `src/bxi_example_py_elf3/test/test_zerolab_timeline.py`

**Interfaces:**
- Consumes: `BurstTimelineReconstructor(rate_hz: float)` and a sequence of monotonic robot receive timestamps.
- Produces: `reconstruct_batch(receive_timestamps_ns) -> tuple[int, ...]`, plus read-only redistribution counters for runtime telemetry.

- [x] **Step 1: Write failing burst reconstruction tests**

```python
def test_clustered_arrivals_are_spread_back_from_newest_arrival():
    timeline = BurstTimelineReconstructor(rate_hz=50.0)
    actual = (1_000_000_000, 1_000_100_000, 1_000_200_000)
    assert timeline.reconstruct_batch(actual) == (
        960_200_000,
        980_200_000,
        1_000_200_000,
    )


def test_regular_arrivals_are_not_changed():
    timeline = BurstTimelineReconstructor(rate_hz=50.0)
    actual = (1_000_000_000, 1_020_000_000, 1_040_000_000)
    assert timeline.reconstruct_batch(actual) == actual
```

- [x] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic" \
python3 -m pytest -q test/test_zerolab_timeline.py
```

Expected: FAIL because `zerolab.timeline.BurstTimelineReconstructor` does not exist.

- [x] **Step 3: Implement the minimum bounded reconstructor**

Implement validation for finite positive rate, integer monotonic timestamps, empty batches, backward spacing from the newest actual arrival, and a compressed monotonic fallback when a batch cannot fit between the preceding emitted sample and its newest arrival.

- [x] **Step 4: Run the focused tests and verify GREEN**

Run the command from Step 2. Expected: all tests pass.

### Task 2: Preserve freshness time while feeding reconstructed sample time to the resampler

**Files:**
- Modify: `src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/converter.py`
- Modify: `src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/source_node.py`
- Modify: `src/bxi_example_py_elf3/test/test_zerolab_lifecycle.py`
- Modify: `src/bxi_example_py_elf3/test/test_zerolab_converter.py`

**Interfaces:**
- Consumes: actual `ZeroLabPacket.receive_timestamp_ns` and optional reconstructed `sample_timestamp_ns`.
- Produces: actual receive time for freshness/recording/statistics and reconstructed sample time only in `ConvertedPoseFrame.receive_timestamp_ns`, which remains the resampler's time coordinate.

- [x] **Step 1: Write a failing integration test**

Create one controlled node tick containing three valid packets whose receive timestamps differ by only 0.1 ms. Assert that the converter/resampler receives 20 ms-spaced timestamps while `latest_real_receive_timestamp_ns` and `maximum_real_arrival_gap_ms` still use the original packet arrivals.

- [x] **Step 2: Run the integration test and verify RED**

Run:

```bash
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic" \
python3 -m pytest -q \
  test/test_zerolab_lifecycle.py::test_node_reconstructs_burst_sample_timeline_without_falsifying_freshness
```

Expected: FAIL because the node currently forwards UDP arrival timestamps directly to the resampler.

- [x] **Step 3: Wire the reconstructor into the source node**

Drain and parse all currently available valid datagrams, preserve raw recordings with actual receive timestamps, reconstruct one timestamp batch, and call:

```python
self._core.accept(packet, sample_timestamp_ns=sample_timestamp_ns)
```

Update `ZeroLabMotionConverter.observe()` so its optional sample timestamp controls only the converted frame's resampling coordinate. Keep the core's stale and arrival-gap calculations on `packet.receive_timestamp_ns`.

- [x] **Step 4: Expose bounded diagnostics**

Append `timeline_redistributed_packets` and `maximum_timeline_adjustment_ms` to the existing five-second `ZeroLab source stats` log line.

- [x] **Step 5: Run focused tests and verify GREEN**

Run the integration and timeline test files. Expected: all tests pass.

### Task 3: Regression verification and candidate commit

**Files:**
- Modify if required: `src/bxi_example_py_elf3/mods/com.bxi.sonic/README.md`
- Verify: all ZeroLab and SONIC reference tests.

**Interfaces:**
- Consumes: the completed timeline reconstruction behavior.
- Produces: one new commit based on `zerolab-jitter-40ms`, without moving either existing rollback tag.

- [x] **Step 1: Run focused ZeroLab tests**

```bash
PYTHONPATH="$PWD:$PWD/mods/com.bxi.sonic" \
python3 -m pytest -q \
  test/test_zerolab_resampler.py \
  test/test_zerolab_timeline.py \
  test/test_zerolab_pose_contract.py \
  test/test_zerolab_lifecycle.py \
  test/test_zerolab_manifest.py \
  test/test_zerolab_arming.py \
  test/test_sonic_reference_gate.py \
  test/test_sonic_ordered_playout.py
```

- [x] **Step 2: Inspect the exact diff and verify rollback tags**

```bash
git diff --check
git diff --stat
git rev-list -n1 zerolab-jitter-40ms
git rev-list -n1 zerolab-jitter-80ms
```

- [x] **Step 3: Commit the isolated change**

```bash
git add \
  docs/superpowers/plans/2026-08-28-zerolab-source-timeline-reconstruction.md \
  src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/timeline.py \
  src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/converter.py \
  src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/source_node.py \
  src/bxi_example_py_elf3/test/test_zerolab_timeline.py \
  src/bxi_example_py_elf3/test/test_zerolab_lifecycle.py \
  src/bxi_example_py_elf3/test/test_zerolab_converter.py
git commit -m "fix: reconstruct ZeroLab UDP burst timeline"
```

- [x] **Step 4: Verify the commit and working tree**

```bash
git show --stat --oneline HEAD
git status --short
```
