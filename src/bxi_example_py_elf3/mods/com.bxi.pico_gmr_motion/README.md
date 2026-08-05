# PICO GMR Motion Mod

This self-contained Mod replaces a fixed NPZ reference with live PICO body
tracking while retaining the RGMT actor and named 29-joint control contract.
Its RGMT policy implementation and ONNX/RKNN assets are copied into this Mod;
it does not require `com.bxi.any_motion` at load time or runtime.

Runtime pipeline:

1. The state-scoped `pico_gmr_source` node starts XRoboToolkit PC Service and
   the SDK client automatically.
2. `pico_gmr_process.py` reads the 24 XRoboToolkit body poses.
3. Unity poses are converted to the right-handed convention used by MoCapLab.
4. A compact two-stage GMR solver applies `pico_to_elf3.json` to the current
   package `data/mujoco_simulation/elf3.xml` model.
5. Versioned UDP packets carry named-layout 29-joint poses plus torso
   orientation and finite-difference velocities.
6. The state starts in RGMT standing-reference mode and waits indefinitely;
   connecting or wearing the headset does not trigger a state transition.
7. Pressing PICO `A+X` enables tracking and starts a new 21-frame session.
   While those frames are collected, the private RGMT actor continues using
   its standing reference instead of statically locking the joints.
8. Once the live window is complete, the same private
   `RgmtExternalReferencePolicy.step_with_reference_window()` consumes the
   real-time PICO/GMR reference on every new frame.
9. Pressing PICO `A+X` again disables live publication. The stale reference is
   discarded and the actor returns to its standing reference without leaving
   the Pico-GMR state.

The user does not need to start XRoboToolkit PC Service manually. The runtime
is selected in this order:

1. `PICO_GMR_XRT_SERVICE_DIR`, when explicitly configured;
2. `/opt/apps/roboticsservice`, when installed on the host;
3. the current-platform copy under
   `runtime/<platform>/roboticsservice` inside this Mod.

The launcher prefers an importable host `xrobotoolkit_sdk`; otherwise it adds
the matching bundled CPython binding under `vendor/python`. The node owns only
the service process it creates and terminates that exact process when the
state exits. Switching away therefore releases the SDK and TCP port 60061.

There is no connection or startup timeout that returns to `normal`. If live
tracking has not been enabled, fewer than 21 fresh frames are available, or a
running stream is stale for more than 0.4 seconds, RGMT keeps running against
the standing reference. Reconnecting the headset can therefore take as long
as needed. `normal`, PD brake and zero-torque remain explicit operator exits.

The `A+X` buttons are read directly from the PICO controllers through
`xrobotoolkit_sdk`, matching SONIC's POSE/PLANNER toggle style. They do not use
the Xbox/CRSF `MotionCommands.btn_*` mapping. Each transition from standing to
live creates a new session, so old frames cannot be mixed into the new
21-frame window.

The PICO headset still needs to run the XRoboToolkit sender with body tracking
enabled. Automatic PC Service startup cannot create body data when the headset
is disconnected, asleep, or not publishing.

First hardware validation must use a suspension, conservative gains and an
immediately reachable emergency stop. Automatic process management and the
standing fallback reduce two startup failure modes; they do not make an
unvalidated retargeted motion intrinsically safe.

The PICO joint mapping and two-stage task configuration are adapted from
Yanjie Ze's General Motion Retargeting (GMR), as carried by
`BXI_Elf3_MoCapLab`. See `vendor/licenses/GMR.LICENSE`.

The right elbow and wrist orientation targets use `Rx(180 deg)`. This is the
fixed-frame calibration measured from a real, converted PICO body frame for
the current URDF-aligned ELF3 XML. The original MoCapLab primary config's
identity offsets make the current model twist `r_shoulder_z_joint` by roughly
130 degrees, while the alternative offsets from `pico_to_elf3（复件）.json`
make a naturally hanging right arm converge to an L-shaped elbow branch. Do
not mix either old calibration with the current XML.

The portable IK follows MoCapLab's actual Mink formulation: body-frame SE(3)
errors and Jacobians, squared FrameTask costs, per-task error-dependent LM
damping, global damping `0.5`, two stages of up to 11 solves, and the same
`0.95` configuration-limit gain. It intentionally has no output low-pass,
previous-pose regularizer or artificial arm velocity clamp; those additions
made the live reference lag behind PICO. A small NumPy active-set box solver
replaces DAQP without changing the objective, keeping the Mod portable to both
x86_64 and aarch64. Starting a new `A+X` tracking session resets the IK warm
start to the complete named 29-joint RGMT standing reference copied from the
ONNX `policy_default_joint_pos` metadata. The current XML has no MuJoCo
keyframe, and resetting to its all-zero configuration would place both elbow
joints at an L-shaped branch instead of the standing values near `1.28 rad`.
The standing pose is only an IK initial condition; it is never added to the
retargeted output.

## Mod-local diagnostic tools

All PICO-GMR capture and diagnosis entry points are contained in this Mod:

- `pico_gmr_launcher.py` prepares the packaged XR runtime and starts the
  worker;
- `pico_gmr_process.py --capture-frame` records one converted real PICO frame;
- `tools/diagnose_pico_gmr_roundtrip.py` performs FK round trips, captured-frame
  candidate comparisons and direct MuJoCo policy-bypass viewing.

The examples below run from the repository root. Stop the ROS demo before
starting a standalone PICO source, because both processes own XRoboToolkit PC
Service and its TCP port.

### Capture a real PICO frame

Wear the headset, stand in the pose to diagnose, press PICO `A+X`, and wait for
`Captured converted PICO frame`:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=src/bxi_example_py_elf3 \
/usr/bin/python3 -B \
  src/bxi_example_py_elf3/mods/com.bxi.pico_gmr_motion/pico_gmr_launcher.py \
  --host 127.0.0.1 \
  --port 5569 \
  --capture-frame /tmp/pico_gmr_real.json
```

The JSON contains the 24 body poses after the same Unity-to-right-handed
conversion used online, the source timestamp, named 29-joint GMR output and
layout. It contains no camera image. One process writes only its first enabled
tracking frame; use a different output name and restart it to capture another
pose.

### Replay a captured frame and compare calibrations

```bash
PYTHONDONTWRITEBYTECODE=1 \
/usr/bin/python3 -B \
  src/bxi_example_py_elf3/mods/com.bxi.pico_gmr_motion/tools/diagnose_pico_gmr_roundtrip.py \
  --replay-pico /tmp/pico_gmr_real.json
```

The table reports the current calibration and retained diagnostic alternatives.
`current` is the production `Rx(180 deg)` right-elbow/right-wrist calibration;
the other candidates reproduce previous identity, hybrid, backup and XML-frame
hypotheses without changing the production JSON.

To put one candidate's 29 joint angles directly into MuJoCo, bypassing RGMT:

```bash
PYTHONDONTWRITEBYTECODE=1 \
/usr/bin/python3 -B \
  src/bxi_example_py_elf3/mods/com.bxi.pico_gmr_motion/tools/diagnose_pico_gmr_roundtrip.py \
  --replay-pico /tmp/pico_gmr_real.json \
  --view-candidate current
```

Available names are printed by `--help` and currently include `current`,
`previous_identity`, `legacy_hybrid`, `mocaplab_backup_right_arm`,
`right_xml_frame_corrected` and `all_arm_xml_frame_corrected`.

### View live GMR joints without the policy

Start the standalone source in terminal 1:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=src/bxi_example_py_elf3 \
/usr/bin/python3 -B \
  src/bxi_example_py_elf3/mods/com.bxi.pico_gmr_motion/pico_gmr_launcher.py \
  --host 127.0.0.1 --port 5569
```

Start the direct MuJoCo viewer in terminal 2:

```bash
PYTHONDONTWRITEBYTECODE=1 \
/usr/bin/python3 -B \
  src/bxi_example_py_elf3/mods/com.bxi.pico_gmr_motion/tools/diagnose_pico_gmr_roundtrip.py \
  --listen 127.0.0.1:5569
```

Press PICO `A+X` to publish. If the bad pose is visible here, the fault is in
PICO coordinates, GMR calibration or the MuJoCo model. If this viewer is
correct while the controlled robot is wrong, inspect the reference-window and
policy-consumption boundary.

### Check standing-reference FK/GMR round trip

With no live hardware, the tool extracts the RGMT standing pose from ONNX,
applies MuJoCo FK, synthesizes a matching pseudo-PICO task frame, solves it
again and reports joint and task-space errors:

```bash
PYTHONDONTWRITEBYTECODE=1 \
/usr/bin/python3 -B \
  src/bxi_example_py_elf3/mods/com.bxi.pico_gmr_motion/tools/diagnose_pico_gmr_roundtrip.py
```

Use `--view standing`, `--view cold` or `--view warm` for visual comparison.
Run `--help` for model/config overrides and all viewer options.
