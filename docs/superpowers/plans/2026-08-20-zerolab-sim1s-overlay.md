# ZeroLab 1.0-Second Simulation Overlay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a MuJoCo-only ZeroLab overlay whose coordinated stale threshold is 1.0 second while every tracked and hardware-capable configuration remains at 0.5 seconds.

**Architecture:** Assert the tracked manifest first, copy only `bxi_example_py_elf3` into a fresh `/tmp` staging directory, and change three ZeroLab-only manifest values in that disposable copy. Build the staged package over the existing `install-vendor-stream/` underlay into uniquely named `*-vendor-stream-sim1s` artifacts, then verify the installed manifest and launch only `example_demo.launch.py`.

**Tech Stack:** Ubuntu 22.04, ROS 2 Humble, Python 3.10, PyYAML, colcon, MuJoCo, ROS 2 CLI.

## Global Constraints

- Keep tracked `src/bxi_example_py_elf3/mods/com.bxi.sonic/mod.yaml` and existing `install-vendor-stream/` unchanged at 0.5 seconds.
- Change only staged `nodes.zerolab_source.params.stale_seconds`, `nodes.zerolab_bridge.params.stale_warning_seconds`, and `states.sonic_zerolab.params.live_reference_timeout_s` from `0.5` to `1.0`.
- Keep `nodes.smpl_bridge.params.stale_warning_seconds=0.5` and `states.sonic_teleop.params.live_reference_timeout_s=0.5`.
- Build only `build-vendor-stream-sim1s/`, `install-vendor-stream-sim1s/`, and `log-vendor-stream-sim1s/` from a fresh `/tmp` staging tree.
- Use existing `install-vendor-stream/` only as the `bxi_depth_camera` underlay; do not modify or rebuild it.
- Preserve all existing untracked build, install, and log directories.
- Launch only `example_demo.launch.py`; never source this overlay before or launch `example_demo_hw.launch.py`.
- Preserve explicit `btn_10=11` entry and `btn_10=12` initial/recovery arming.
- Treat 1.0 seconds as a diagnostic MuJoCo threshold, not a fix for the measured bursty UDP sender.

---

### Task 1: Build and Validate the Simulation-Only Overlay

**Files:**
- Create: `docs/superpowers/plans/2026-08-20-zerolab-sim1s-overlay.md`
- Read-only: `src/bxi_example_py_elf3/mods/com.bxi.sonic/mod.yaml`
- Stage and modify: `/tmp/zerolab-vendor-stream-sim1s.XXXXXX/bxi_example_py_elf3/mods/com.bxi.sonic/mod.yaml`
- Create build artifact: `build-vendor-stream-sim1s/`
- Create install artifact: `install-vendor-stream-sim1s/`
- Create log artifact: `log-vendor-stream-sim1s/`

**Interfaces:**
- Consumes: the current tracked `bxi_example_py_elf3` source and the existing `install-vendor-stream/` underlay.
- Produces: `install-vendor-stream-sim1s/setup.bash`, whose `bxi_example_py_elf3` package prefix is the isolated overlay and whose installed manifest has only the three ZeroLab thresholds set to `1.0`.
- Safety boundary: no tracked runtime source, ordinary PICO timeout, hardware process, hardware launch file, or existing install tree is modified.

- [ ] **Step 1: Assert the tracked safety contract before staging**

Run from the worktree root:

```bash
python3 - <<'PY'
from pathlib import Path
import yaml

path = Path('src/bxi_example_py_elf3/mods/com.bxi.sonic/mod.yaml')
data = yaml.safe_load(path.read_text())
checks = {
    'zerolab_source': data['nodes']['zerolab_source']['params']['stale_seconds'],
    'zerolab_bridge': data['nodes']['zerolab_bridge']['params']['stale_warning_seconds'],
    'sonic_zerolab': data['states']['sonic_zerolab']['params']['live_reference_timeout_s'],
    'smpl_bridge': data['nodes']['smpl_bridge']['params']['stale_warning_seconds'],
    'sonic_teleop': data['states']['sonic_teleop']['params']['live_reference_timeout_s'],
}
assert checks == {
    'zerolab_source': 0.5,
    'zerolab_bridge': 0.5,
    'sonic_zerolab': 0.5,
    'smpl_bridge': 0.5,
    'sonic_teleop': 0.5,
}, checks
print('TRACKED_MANIFEST_0_5=PASS', checks)
PY
```

Expected: `TRACKED_MANIFEST_0_5=PASS`; otherwise stop without building.

- [ ] **Step 2: Copy the package into a fresh disposable staging directory**

```bash
STAGE_ROOT=$(mktemp -d /tmp/zerolab-vendor-stream-sim1s.XXXXXX)
cp -a src/bxi_example_py_elf3 "$STAGE_ROOT/"
export STAGE_ROOT
echo "STAGE_ROOT=$STAGE_ROOT"
test -f "$STAGE_ROOT/bxi_example_py_elf3/mods/com.bxi.sonic/mod.yaml"
```

Expected: `STAGE_ROOT` names a new `/tmp/zerolab-vendor-stream-sim1s.*` directory and the copied manifest exists.

- [ ] **Step 3: Patch only the three staged ZeroLab values**

```bash
python3 - <<'PY'
import os
from pathlib import Path
import yaml

path = Path(os.environ['STAGE_ROOT']) / 'bxi_example_py_elf3/mods/com.bxi.sonic/mod.yaml'
data = yaml.safe_load(path.read_text())
assert data['nodes']['zerolab_source']['params']['stale_seconds'] == 0.5
assert data['nodes']['zerolab_bridge']['params']['stale_warning_seconds'] == 0.5
assert data['states']['sonic_zerolab']['params']['live_reference_timeout_s'] == 0.5
data['nodes']['zerolab_source']['params']['stale_seconds'] = 1.0
data['nodes']['zerolab_bridge']['params']['stale_warning_seconds'] = 1.0
data['states']['sonic_zerolab']['params']['live_reference_timeout_s'] = 1.0
path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
print(f'STAGED_PATCHED={path}')
PY
```

Expected: `STAGED_PATCHED` points only inside `$STAGE_ROOT`.

- [ ] **Step 4: Assert staged ZeroLab values are 1.0 and PICO values remain 0.5**

```bash
python3 - <<'PY'
import os
from pathlib import Path
import yaml

path = Path(os.environ['STAGE_ROOT']) / 'bxi_example_py_elf3/mods/com.bxi.sonic/mod.yaml'
data = yaml.safe_load(path.read_text())
zerolab = {
    'source': data['nodes']['zerolab_source']['params']['stale_seconds'],
    'bridge': data['nodes']['zerolab_bridge']['params']['stale_warning_seconds'],
    'state': data['states']['sonic_zerolab']['params']['live_reference_timeout_s'],
}
pico = {
    'bridge': data['nodes']['smpl_bridge']['params']['stale_warning_seconds'],
    'state': data['states']['sonic_teleop']['params']['live_reference_timeout_s'],
}
assert zerolab == {'source': 1.0, 'bridge': 1.0, 'state': 1.0}, zerolab
assert pico == {'bridge': 0.5, 'state': 0.5}, pico
print('STAGED_ZEROLAB_1_0=PASS', zerolab)
print('STAGED_PICO_0_5=PASS', pico)
PY
```

Expected: both `PASS` lines.

- [ ] **Step 5: Build the staged package into isolated sim1s artifacts**

```bash
source /opt/ros/humble/setup.bash
source "$PWD/install-vendor-stream/setup.bash"

colcon --log-base log-vendor-stream-sim1s build \
  --merge-install \
  --base-paths "$STAGE_ROOT/bxi_example_py_elf3" \
  --packages-select bxi_example_py_elf3 \
  --allow-overriding bxi_example_py_elf3 \
  --build-base build-vendor-stream-sim1s \
  --install-base install-vendor-stream-sim1s \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
```

Expected: `Summary: 1 package finished` and no package failure.

- [ ] **Step 6: Assert package resolution selects the sim1s overlay**

```bash
source /opt/ros/humble/setup.bash
source "$PWD/install-vendor-stream/setup.bash"
source "$PWD/install-vendor-stream-sim1s/setup.bash"

EXPECTED_PREFIX="$PWD/install-vendor-stream-sim1s"
ACTUAL_PREFIX=$(ros2 pkg prefix bxi_example_py_elf3)
test "$ACTUAL_PREFIX" = "$EXPECTED_PREFIX"
echo "SIM1S_PACKAGE_PREFIX=PASS $ACTUAL_PREFIX"
```

Expected: the printed prefix ends exactly in `install-vendor-stream-sim1s`.

- [ ] **Step 7: Assert both installed and tracked manifest contracts**

```bash
python3 - <<'PY'
from pathlib import Path
import yaml

def values(path):
    data = yaml.safe_load(path.read_text())
    return {
        'zerolab_source': data['nodes']['zerolab_source']['params']['stale_seconds'],
        'zerolab_bridge': data['nodes']['zerolab_bridge']['params']['stale_warning_seconds'],
        'sonic_zerolab': data['states']['sonic_zerolab']['params']['live_reference_timeout_s'],
        'smpl_bridge': data['nodes']['smpl_bridge']['params']['stale_warning_seconds'],
        'sonic_teleop': data['states']['sonic_teleop']['params']['live_reference_timeout_s'],
    }

tracked = values(Path('src/bxi_example_py_elf3/mods/com.bxi.sonic/mod.yaml'))
installed = values(Path('install-vendor-stream-sim1s/share/bxi_example_py_elf3/mods/com.bxi.sonic/mod.yaml'))
assert tracked == {
    'zerolab_source': 0.5,
    'zerolab_bridge': 0.5,
    'sonic_zerolab': 0.5,
    'smpl_bridge': 0.5,
    'sonic_teleop': 0.5,
}, tracked
assert installed == {
    'zerolab_source': 1.0,
    'zerolab_bridge': 1.0,
    'sonic_zerolab': 1.0,
    'smpl_bridge': 0.5,
    'sonic_teleop': 0.5,
}, installed
print('TRACKED_SAFETY_CONTRACT=PASS', tracked)
print('INSTALLED_SIM1S_CONTRACT=PASS', installed)
PY
```

Expected: both contract assertions pass.

- [ ] **Step 8: Run the simulation preflight before launch**

```bash
if pgrep -af \
  '[h]ardware_elf3|[b]xi_example_py_elf3_demo|[r]os2 launch.*example_demo(_hw)?\.launch\.py|[z]erolab_source|[p]ose_to_smpl_ref_bridge'
then
  echo 'STOP: existing controller or pose process found'
  return 1 2>/dev/null || exit 1
fi

for port in 5557 5558 18000
do
  if ss -H -ltnp "sport = :$port" | grep -q . || \
     ss -H -lunp "sport = :$port" | grep -q .
  then
    echo "STOP: port $port is occupied"
    return 1 2>/dev/null || exit 1
  fi
done
echo 'SIMULATION_PREFLIGHT=PASS'
```

Expected: no matched processes or listeners and `SIMULATION_PREFLIGHT=PASS`.

- [ ] **Step 9: Launch only the MuJoCo demo and exercise explicit arming**

Terminal 1:

```bash
cd /home/fazepurple/ros2_ws/bxi_rl_controller_ros2_example_dev/.worktrees/konodoki-dev
source /opt/ros/humble/setup.bash
source "$PWD/install-vendor-stream/setup.bash"
source "$PWD/install-vendor-stream-sim1s/setup.bash"
export ROS_DOMAIN_ID=42

ros2 launch bxi_example_py_elf3 example_demo.launch.py 2>&1 | \
  tee /tmp/zerolab-vendor-stream-sim1s-mujoco.log
```

Terminal 2, after the launch is stable and the operator has completed the vendor N-pose procedure and returned to neutral:

```bash
cd /home/fazepurple/ros2_ws/bxi_rl_controller_ros2_example_dev/.worktrees/konodoki-dev
source /opt/ros/humble/setup.bash
source "$PWD/install-vendor-stream/setup.bash"
source "$PWD/install-vendor-stream-sim1s/setup.bash"
export ROS_DOMAIN_ID=42

pulse_btn10() {
  ros2 topic pub --once \
    /motion_commands communication/msg/MotionCommands \
    "{btn_10: $1}"
  ros2 topic pub --once \
    /motion_commands communication/msg/MotionCommands '{}'
}

pulse_btn10 11
grep -E 'stream ready|WAIT_ARM' /tmp/zerolab-vendor-stream-sim1s-mujoco.log | tail -n 10
pulse_btn10 12
```

Expected: the log reaches `WAIT_ARM`, then `ARM accepted` and `ARMED`; small operator motions move MuJoCo in the correct direction. Never run `example_demo_hw.launch.py` with this overlay sourced.

- [ ] **Step 10: Monitor stale behavior and verify the 1.0-second boundary**

Terminal 3:

```bash
tail -F /tmp/zerolab-vendor-stream-sim1s-mujoco.log | \
  grep --line-buffered -E \
  'WAIT_STREAM|WAIT_ARM|ARM accepted|ARMED|held_reference|fresh input pending|REARMING|input stale|stream ready'
```

Acceptance observations:

1. The previously measured `616.4 ms` UDP gap must no longer cause `input stale` or `held_reference`.
2. A deliberate sender pause longer than `1.0 s` must produce `input stale` and `held_reference`/`HOLD_REFERENCE`.
3. After stream recovery, the robot remains held until the safety operator explicitly runs `pulse_btn10 12` again.
4. If ordinary small operator motion still does not move MuJoCo while continuously `ARMED`, collect the three terminal logs; do not increase the timeout again.

- [ ] **Step 11: Verify source and existing artifacts remain preserved**

After stopping MuJoCo with Ctrl-C:

```bash
git status --short
python3 - <<'PY'
from pathlib import Path
import yaml

path = Path('src/bxi_example_py_elf3/mods/com.bxi.sonic/mod.yaml')
data = yaml.safe_load(path.read_text())
assert data['nodes']['zerolab_source']['params']['stale_seconds'] == 0.5
assert data['nodes']['zerolab_bridge']['params']['stale_warning_seconds'] == 0.5
assert data['states']['sonic_zerolab']['params']['live_reference_timeout_s'] == 0.5
print('FINAL_TRACKED_SAFETY_CONTRACT=PASS')
PY

for name in \
  build-live-normal install-live-normal log-live-normal \
  build-safe-arm install-safe-arm log-safe-arm \
  build-vendor-stream install-vendor-stream log-vendor-stream \
  build-vendor-stream-sim1s install-vendor-stream-sim1s log-vendor-stream-sim1s
do
  test -e "$name" && echo "$name=PRESENT" || echo "$name=ABSENT"
done
```

Expected: tracked runtime source still reports `0.5`; all pre-existing artifacts remain present; only the three new sim1s artifact directories are added as untracked local build outputs.

- [ ] **Step 12: Commit only the implementation plan**

```bash
git add docs/superpowers/plans/2026-08-20-zerolab-sim1s-overlay.md
git commit -m "docs: plan ZeroLab sim-only 1s overlay"
```

Expected: no build, install, log, `/tmp`, or tracked runtime source is included in the commit.
