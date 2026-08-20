# ZeroLab 1.0-Second Simulation Overlay Design

## Goal

Create a local MuJoCo-only install of the current ZeroLab vendor-stream
candidate with a coordinated 1.0-second reference-stale threshold. This
overlay exists only to observe live motion despite the measured 616.4 ms UDP
arrival gap. It must not change the source tree's 0.5-second hardware safety
contract or any existing install.

## Scope and isolation

- Keep the tracked `mod.yaml` and the existing `install-vendor-stream/` at
  0.5 seconds.
- Stage a complete copy of `src/bxi_example_py_elf3` under `/tmp` and modify
  only that copy.
- Build the staged package into new, clearly named artifacts:
  `build-vendor-stream-sim1s/`, `install-vendor-stream-sim1s/`, and
  `log-vendor-stream-sim1s/`.
- Use the already-built `install-vendor-stream/` as the underlay so the
  isolated overlay can resolve `bxi_depth_camera` without rebuilding or
  modifying existing artifacts.
- Do not create, copy, or launch anything on robot hardware.

## Coordinated override

Only these ZeroLab values in the staged `mod.yaml` change from `0.5` to
`1.0`:

1. `nodes.zerolab_source.params.stale_seconds`
2. `nodes.zerolab_bridge.params.stale_warning_seconds`
3. `states.sonic_zerolab.params.live_reference_timeout_s`

The ordinary PICO values remain unchanged, including
`nodes.smpl_bridge.params.stale_warning_seconds=0.5` and
`states.sonic_teleop.params.live_reference_timeout_s=0.5`.

## Verification

Before launching MuJoCo:

- parse the tracked, staged, and installed manifests and assert the exact
  values above;
- assert the tracked source still uses 0.5 seconds for all three ZeroLab
  values;
- assert the overlay package prefix resolves to
  `install-vendor-stream-sim1s`;
- confirm no hardware controller or hardware launch process is running;
- confirm ports 5557, 5558, and 18000 are free.

Launch only `example_demo.launch.py`. The operator still follows the existing
explicit sequence: `btn_10=11`, wait for `WAIT_ARM`, then `btn_10=12`.
Recovery after a stale event still requires another explicit `btn_10=12`.

## Acceptance and limitations

The overlay succeeds when MuJoCo reaches `ARMED`, remains live across the
observed 616.4 ms gap, and responds to small operator motions with the correct
direction. Any gap above 1.0 second must still enter `HOLD_REFERENCE`.

This is a diagnostic overlay, not a real-hardware acceptance artifact. It
does not make bursty UDP delivery correct, does not supersede the 0.5-second
design, and must never be sourced before `example_demo_hw.launch.py`.

