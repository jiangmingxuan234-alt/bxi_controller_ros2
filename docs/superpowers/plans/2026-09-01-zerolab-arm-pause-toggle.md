# ZeroLab ARM Pause Toggle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `btn_10=12` arm ZeroLab from `WAIT_ARM` and smoothly pause an active ZeroLab session back to live Normal without stopping the ZeroLab source or bridge.

**Architecture:** Add a `DISARMING` phase to the existing ZeroLab state. A manual pause releases any latched reference, blends the last applied SONIC motor frame into the continuously sampled Normal frame over the existing `arm_blend_seconds`, then lands in `WAIT_ARM` when the stream is fresh or `WAIT_STREAM` when stale.

**Tech Stack:** Python 3, NumPy, pytest, existing SONIC state machine and live-reference gate.

## Global Constraints

- `btn_10=12` remains the only ZeroLab ARM/pause action.
- A second Y press must not leave `sonic_zerolab`; source/bridge remain alive.
- Manual pause must never freeze or replay an old human pose.
- `RB+X` remains the full exit to Normal and `RB+B` remains PD Brake.
- Preserve the uncommitted ZeroLab resampler work.

---

### Task 1: Reference-gate manual release

**Files:**
- Modify: `src/bxi_example_py_elf3/mods/com.bxi.sonic/reference_gate.py`
- Modify: `src/bxi_example_py_elf3/mods/com.bxi.sonic/policy.py`
- Test: `src/bxi_example_py_elf3/test/test_sonic_reference_gate.py`

**Interfaces:**
- Produces: `LiveReferenceGate.release_to_live() -> None`
- Produces: `SonicTeleopPolicy.release_live_reference_hold() -> None`

- [ ] **Step 1: Write failing tests**

Add literal-behavior tests showing that release from HOLD and REARMING selects the newest observed frame, returns to `LIVE`, and clears interpolation state.

- [ ] **Step 2: Verify RED**

Run:

```bash
pytest -q src/bxi_example_py_elf3/test/test_sonic_reference_gate.py
```

Expected: failure because `release_to_live` does not exist.

- [ ] **Step 3: Implement the minimum release API**

Implement `release_to_live` by setting mode to `LIVE`, clearing `_latched`, and resetting `_rearm_progress`; expose it through `SonicTeleopPolicy.release_live_reference_hold`.

- [ ] **Step 4: Verify GREEN**

Run the reference-gate test file and expect all tests to pass.

### Task 2: State-level ARM/pause toggle

**Files:**
- Modify: `src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/state.py`
- Modify: `src/bxi_example_py_elf3/test/test_zerolab_arming.py`

**Interfaces:**
- Consumes: `SonicTeleopPolicy.release_live_reference_hold() -> None`
- Produces: `ZeroLabArmPhase.DISARMING`
- Produces: state-dependent `arm_zerolab`: `WAIT_ARM -> BLENDING`, active phases -> `DISARMING`

- [ ] **Step 1: Write failing state tests**

Cover these externally visible behaviors with hand-derived motor targets:

```text
ARMED + Y -> DISARMING -> half blend -> live Normal -> WAIT_ARM
HOLD_REFERENCE + Y -> DISARMING -> live Normal -> WAIT_STREAM if stale
paused WAIT_ARM + Y -> BLENDING -> ARMED
Y during DISARMING -> no restart or phase reversal
```

- [ ] **Step 2: Verify RED**

Run:

```bash
pytest -q src/bxi_example_py_elf3/test/test_zerolab_arming.py
```

Expected: failure because active phases currently ignore `arm_zerolab`.

- [ ] **Step 3: Implement the minimum phase logic**

Capture the current applied frame at pause request, release any reference hold, blend it to live Normal using smoothstep over `arm_blend_seconds`, and choose `WAIT_ARM`/`WAIT_STREAM` based on current freshness when the blend completes.

- [ ] **Step 4: Verify GREEN**

Run the arming test file and expect all tests to pass.

### Task 3: Manifest and operator documentation

**Files:**
- Modify: `src/bxi_example_py_elf3/mods/com.bxi.sonic/mod.yaml`
- Modify: `src/bxi_example_py_elf3/mods/com.bxi.sonic/README.md`
- Modify: `src/bxi_example_py_elf3/test/test_zerolab_manifest.py`

**Interfaces:**
- Produces: user-visible `ZeroLab ARM / 暂停` action label and documented tablet sequence.

- [ ] **Step 1: Update the manifest test expectation**
- [ ] **Step 2: Verify the test fails on the old label**
- [ ] **Step 3: Update the label and ARM/pause lifecycle documentation**
- [ ] **Step 4: Run manifest and ZeroLab regression tests**

Run:

```bash
pytest -q \
  src/bxi_example_py_elf3/test/test_zerolab_manifest.py \
  src/bxi_example_py_elf3/test/test_zerolab_arming.py \
  src/bxi_example_py_elf3/test/test_sonic_reference_gate.py
```
