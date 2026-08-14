# ZeroLab Independent SONIC State Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a lifecycle-managed `com.bxi.sonic/sonic_zerolab` state to `konodoki/dev` so live ZeroLab UDP motion drives the current SONIC v5 policy in the rich MuJoCo universal test scene without changing the existing PICO state or either motion algorithm.

**Architecture:** Preserve the verified ZeroLab parser, T-pose calibrator, coordinate conversion and 10-frame pose source byte-for-byte, except for adding the v5-required all-zero `head_joint_pos` transport field. Publish ZeroLab pose chunks on 5558, reuse the current `pico.pose_to_smpl_ref_bridge` as a second state-scoped bridge to produce v5 source chunks on 5557, and register a separate SONIC state that shares the existing policy resource but owns neither head nor gripper joints.

**Tech Stack:** Ubuntu 22.04, ROS 2 Humble, Python 3.10, rclpy, NumPy, SciPy, pyzmq, pytest, YAML Mod manifests, MuJoCo MJCF, colcon.

## Global Constraints

- Work only in `/home/fazepurple/ros2_ws/bxi_rl_controller_ros2_example_dev/.worktrees/konodoki-dev` on branch `test/konodoki-dev`.
- Preserve `konodoki/dev@efbce8d` PICO activation `btn_10=9` and all existing PICO, RoboticsService and RTSP lifecycle behavior.
- Add ZeroLab activation as `btn_10=11`; `btn_10=4` remains owned by `com.bxi.pico_gmr_motion`.
- Do not change ZeroLab calibration, quaternion, coordinate, left/right mapping, wrist mapping or SMPL conversion algorithms.
- Keep the original 100 stable-frame T-pose calibration, 50 Hz source rate, 10-frame chunks and 0.5-second stale threshold.
- Add only `head_joint_pos: float32[10,2] == 0` to the ZeroLab pose contract; configure `head_control_enabled=false` and `hardware_gripper=false`.
- Reuse the current v5 bridge and policy-owned ordered playback; do not create another merger, cursor, ACK channel or legacy v4 bridge.
- Keep `require_live_reference=false` because state-scoped nodes start only after state availability succeeds.
- Restrict the runtime UDP sender to `192.168.89.171` and bind UDP port 18000.
- Use `example_demo.launch.py` and its existing `elf3.xml`, which includes `terrains/universal_test/universal_test_terrain.xml`.
- Do not add dependencies, scripts or abstractions when existing framework components cover the requirement.
- Make production edits with `apply_patch`; preserve unrelated files and do not push.

---

### Task 1: Restore the verified ZeroLab core without algorithm changes

**Files:**
- Create: `src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/__init__.py`
- Create: `src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/protocol.py`
- Create: `src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/udp_receiver.py`
- Create: `src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/converter.py`
- Create: `src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/recording.py`
- Create: `src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/source_node.py`
- Create: `src/bxi_example_py_elf3/test/test_zerolab_protocol.py`
- Create: `src/bxi_example_py_elf3/test/test_zerolab_udp_receiver.py`
- Create: `src/bxi_example_py_elf3/test/test_zerolab_converter.py`
- Create: `src/bxi_example_py_elf3/test/test_zerolab_recording.py`

**Interfaces:**
- Consumes: the current Mod loader's dynamic package path, existing `pico.gear_sonic.trl` NumPy/SMPL helpers, and `pico.zmq_messages.pack_pose_message`.
- Produces: `parse_zerolab_packet`, `ZeroLabUdpReceiver`, `ZeroLabMotionConverter`, `PoseChunkWindow`, `ZeroLabSourceCore`, `ZeroLabSourceNode`, and `create_node(context)` for later manifest integration.

- [ ] **Step 1: Add the four existing baseline test files before production files**

Read these immutable Git blobs and use `apply_patch` to add their exact contents:

```text
f47ae5e0c9832c0bcd7ff4e05c55553b7f47fa4a  test_zerolab_protocol.py
6b7e11e61a14526bb2feeed5aea291356ee6f3b1  test_zerolab_udp_receiver.py
0f68401a3f04e8bbcf90ff7ebe54cca61dfdaf04  test_zerolab_converter.py
f441db55dfbc293e2bf0d99e55cad48d70ea65f1  test_zerolab_recording.py
```

Use `git show <blob>` only to read source content. Do not use checkout, restore, shell redirection or copying that changes the index.

- [ ] **Step 2: Run the baseline tests and verify RED**

```bash
source /opt/ros/humble/setup.bash
source /home/fazepurple/ros2_ws/bxi_ros2_pkg/setup.bash
cd /home/fazepurple/ros2_ws/bxi_rl_controller_ros2_example_dev/.worktrees/konodoki-dev
python3 -m pytest -q \
  src/bxi_example_py_elf3/test/test_zerolab_protocol.py \
  src/bxi_example_py_elf3/test/test_zerolab_udp_receiver.py \
  src/bxi_example_py_elf3/test/test_zerolab_converter.py \
  src/bxi_example_py_elf3/test/test_zerolab_recording.py
```

Expected: collection fails with `ModuleNotFoundError: No module named 'zerolab'`.

- [ ] **Step 3: Add the verified production files exactly**

Read these immutable blobs and use `apply_patch` to add their exact content:

```text
c4f9a28f5f5375b45d122839782f033280b22e80  zerolab/__init__.py
4006fac509c88d3f46886ce5a3ba8adff9f0183a  zerolab/protocol.py
3f1361f91f9022a268e3e851ca3560c68503673e  zerolab/udp_receiver.py
31b93cafd45af80019244f82e7f1a05b73ff100f  zerolab/converter.py
dcbf49a6a972941fefefc1027f0408f047b21e9e  zerolab/recording.py
dff39c4347e73b8222ec1ca9a448b4fc7f39edf6  zerolab/source_node.py
```

Do not include `record_cli.py`, paired recording or offline evaluation code; the realtime state does not use them.

- [ ] **Step 4: Prove byte identity before the transport change**

```bash
git hash-object \
  src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/__init__.py \
  src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/protocol.py \
  src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/udp_receiver.py \
  src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/converter.py \
  src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/recording.py \
  src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/source_node.py
```

Expected in order: `c4f9a28f...`, `4006fac5...`, `3f1361f9...`, `31b93caf...`, `dcbf49a6...`, `dff39c43...`.

- [ ] **Step 5: Run the baseline tests and verify GREEN**

Run the Step 2 command again. All selected tests must pass; converter golden-vector failures block implementation rather than justify algorithm edits.

- [ ] **Step 6: Commit the restored core**

```bash
git add \
  src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab \
  src/bxi_example_py_elf3/test/test_zerolab_protocol.py \
  src/bxi_example_py_elf3/test/test_zerolab_udp_receiver.py \
  src/bxi_example_py_elf3/test/test_zerolab_converter.py \
  src/bxi_example_py_elf3/test/test_zerolab_recording.py
git commit -m "feat(sonic): restore ZeroLab realtime source core"
```

---

### Task 2: Adapt only the pose transport contract to the v5 bridge

**Files:**
- Create: `src/bxi_example_py_elf3/test/test_zerolab_pose_contract.py`
- Create: `src/bxi_example_py_elf3/test/test_zerolab_lifecycle.py`
- Modify: `src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/source_node.py`

**Interfaces:**
- Consumes: `PoseChunkWindow.append(ConvertedPoseFrame) -> dict[str, np.ndarray]` and current bridge `_parse_incoming_chunk(fields) -> IncomingChunk`.
- Produces: a ten-frame ZeroLab pose chunk with finite all-zero `head_joint_pos` accepted by the unchanged v5 bridge.

- [ ] **Step 1: Add pose and lifecycle tests before changing the source**

Use blob `be808598224e8580a78bd1a416524d417beec229` for `test_zerolab_pose_contract.py`. Add these assertions after its `joint_pos` assertions:

```python
    assert fields["head_joint_pos"].shape == (10, 2)
    assert fields["head_joint_pos"].dtype == np.float32
    np.testing.assert_array_equal(
        fields["head_joint_pos"],
        np.zeros((10, 2), dtype=np.float32),
    )
```

Keep `test_existing_bridge_accepts_three_progressing_zerolab_chunks` unchanged so it calls the current v5 parser.

Use blob `41fd1f311be24bcf07c0f0c8374c09ebde04d850` for `test_zerolab_lifecycle.py`, removing these obsolete v4 bridge parameters:

```python
        "history_frames": 5,
        "max_gap_frames": 200,
        "catch_up_enabled": True,
```

- [ ] **Step 2: Run the v5 compatibility test and verify RED**

```bash
python3 -m pytest -q \
  src/bxi_example_py_elf3/test/test_zerolab_pose_contract.py::test_window_returns_no_chunk_until_ten_distinct_frames \
  src/bxi_example_py_elf3/test/test_zerolab_pose_contract.py::test_existing_bridge_accepts_three_progressing_zerolab_chunks
```

Expected: missing `head_joint_pos`; bridge parsing reports `PICO pose missing fields: ['head_joint_pos']`.

- [ ] **Step 3: Add the single transport-only field**

In `PoseChunkWindow.append`, add this returned item after `joint_pos`:

```python
            "head_joint_pos": np.zeros(
                (len(frames), 2), dtype=np.float32
            ),
```

Do not modify `_POSE_FIELDS`, `ConvertedPoseFrame`, `ZeroLabMotionConverter`, quaternion handling or joint mapping.

- [ ] **Step 4: Run contract and lifecycle tests and verify GREEN**

```bash
python3 -m pytest -q \
  src/bxi_example_py_elf3/test/test_zerolab_pose_contract.py \
  src/bxi_example_py_elf3/test/test_zerolab_lifecycle.py
```

Expected: all tests pass, including repeated UDP/5558/5557 bind/release and stale/refill checks.

- [ ] **Step 5: Prove algorithm files remain unchanged**

```bash
git hash-object \
  src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/protocol.py \
  src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/udp_receiver.py \
  src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/converter.py \
  src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/recording.py
```

Expected: `4006fac5...`, `3f1361f9...`, `31b93caf...`, `dcbf49a6...`. The `source_node.py` diff may contain only the zero head field.

- [ ] **Step 6: Commit the v5 transport adaptation**

```bash
git add \
  src/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/source_node.py \
  src/bxi_example_py_elf3/test/test_zerolab_pose_contract.py \
  src/bxi_example_py_elf3/test/test_zerolab_lifecycle.py
git commit -m "feat(sonic): adapt ZeroLab pose chunks to v5 bridge"
```

---

### Task 3: Register the independent state and lifecycle-managed nodes

**Files:**
- Create: `src/bxi_example_py_elf3/test/test_zerolab_manifest.py`
- Modify: `src/bxi_example_py_elf3/mods/com.bxi.sonic/mod.yaml`
- Modify: `src/bxi_example_py_elf3/mods/com.bxi.sonic/plugin.py`
- Modify: `src/bxi_example_py_elf3/mods/com.bxi.sonic/state.py`

**Interfaces:**
- Consumes: `zerolab.source_node:create_node`, existing `pico.pose_to_smpl_ref_bridge:create_node`, `SONIC_POLICY`, and framework state-scoped node lifecycle.
- Produces: `activate_zerolab` on `btn_10=11`, state factory `sonic_zerolab`, UDP 18000, pose 5558 and v5 smpl_ref 5557, automatically started/stopped with that state.

- [ ] **Step 1: Add an adapted manifest/state test before production edits**

Use blob `e3efe073523b04937551fd25ab6f84e5b5543e58` as the starting test. Require these exact values:

```python
    assert nodes["zerolab_source"]["params"]["allowed_sender"] == (
        "192.168.89.171"
    )
    assert manifest["events"]["activate_zerolab"] == {
        "slot": "btn_10",
        "value": 11,
    }
    params = manifest["states"]["sonic_zerolab"]["params"]
    assert params["require_live_reference"] is False
    assert params["head_control_enabled"] is False
    assert params["hardware_gripper"] is False
```

Use this prompt and log expectation:

```python
    prompt = "请保持T-pose，直到ZeroLab stream ready后再开始动作"
```

```python
        assert logger.messages == [
            "SONIC遥操已启动；头部跟踪已关闭；" + prompt
        ]
```

Keep assertions for disjoint PICO/ZeroLab node sets, mutually exclusive 5557 bridges, a shared policy handle and dynamic process import of `zerolab_source`.

- [ ] **Step 2: Run the manifest test and verify RED**

```bash
python3 -m pytest -q src/bxi_example_py_elf3/test/test_zerolab_manifest.py
```

Expected: failure because ZeroLab nodes, event, state and factory do not exist.

- [ ] **Step 3: Parameterize the operator prompt without changing PICO behavior**

In `state.py`, define:

```python
DEFAULT_OPERATOR_PROMPT = (
    "PICO同时按住A+B+X+Y请求校准，再按A+X切入实时POSE"
)
```

Add after `source_blend_seconds` in `SonicTeleopState.__init__`:

```python
        operator_prompt: str = DEFAULT_OPERATOR_PROMPT,
```

Store and validate:

```python
        self.operator_prompt = str(operator_prompt)
        if not self.operator_prompt:
            raise ValueError("SONIC operator_prompt must not be empty")
```

Replace the hard-coded PICO suffix in `on_enter`:

```python
        self.logger.info(
            f"{mode}已启动；{head_status}；{self.operator_prompt}"
        )
```

The default preserves the PICO log exactly.

- [ ] **Step 4: Register both factories against the shared policy**

Import `DEFAULT_OPERATOR_PROMPT` in `plugin.py` and pass:

```python
        operator_prompt=state.string_param(
            "operator_prompt",
            DEFAULT_OPERATOR_PROMPT,
        ),
```

Return:

```python
    return ModDefinition(
        state_factories={
            "sonic_teleop": lambda state: _build_state(state, policy),
            "sonic_zerolab": lambda state: _build_state(state, policy),
        }
    )
```

Do not allocate a second policy and do not add a direct PICO-to-ZeroLab route.

- [ ] **Step 5: Add `zerolab_source` to `mod.yaml`**

```yaml
  zerolab_source:
    runtime: python
    entrypoint: zerolab.source_node:create_node
    execution: process
    runtime_profile: host_ros
    lifecycle: state
    states: [sonic_zerolab]
    params:
      udp_bind_host: 0.0.0.0
      udp_port: 18000
      allowed_sender: 192.168.89.171
      pose_host: 127.0.0.1
      pose_port: 5558
      pose_topic: pose
      rate_hz: 50.0
      window_frames: 10
      stale_seconds: 0.5
      record_path: ""
    manifest:
      label: ZeroLab姿态源
    runtime_requirements:
      python:
        - import: numpy
        - import: scipy
        - import: zmq
      ros:
        - package: rclpy
      system: []
    shutdown:
      signal: SIGINT
      terminate_after: 3.0
      kill_after: 5.0
```

- [ ] **Step 6: Add the reused v5 `zerolab_bridge` to `mod.yaml`**

```yaml
  zerolab_bridge:
    runtime: python
    entrypoint: pico.pose_to_smpl_ref_bridge:create_node
    execution: in_process
    runtime_profile: host_ros
    lifecycle: state
    states: [sonic_zerolab]
    depends_on: [zerolab_source]
    params:
      pico_host: 127.0.0.1
      pico_port: 5558
      pico_topic: pose
      out_host: 127.0.0.1
      out_port: 5557
      out_topic: smpl_ref
      rate_hz: 50.0
      stale_warning_seconds: 0.5
    manifest:
      label: ZeroLab SMPL参考桥
    runtime_requirements:
      python:
        - import: zmq
      ros:
        - package: rclpy
        - package: std_msgs
      system: []
```

Do not add v4 merger parameters.

- [ ] **Step 7: Add the event, state and routes to `mod.yaml`**

```yaml
  activate_zerolab:
    slot: btn_10
    value: 11
```

```yaml
  sonic_zerolab:
    manifest:
      label: SONIC ZeroLab遥操
      priority: 839
      group: Advanced
      icon: accessibility_new
      confirm: true
      confirm_message: 请先摆好T-pose；进入后保持不动，直到日志显示ZeroLab stream ready
    params:
      operator_prompt: 请保持T-pose，直到ZeroLab stream ready后再开始动作
      require_live_reference: false
      yaw_bias_rad: 1.57079632679
      live_reference_timeout_s: 0.5
      idle_frame_start: 3509
      source_blend_seconds: 0.4
      head_control_enabled: false
      hardware_gripper: false
```

```yaml
  - from: com.bxi.basic_actions/normal
    event: activate_zerolab
    to: sonic_zerolab
    transition: soft_switch
  - from: sonic_zerolab
    event: com.bxi.basic_actions/normal
    to: com.bxi.basic_actions/normal
    transition: soft_switch
  - from: sonic_zerolab
    event: com.bxi.basic_actions/zero_torque
    to: com.bxi.basic_actions/zero_torque
  - from: sonic_zerolab
    event: com.bxi.basic_actions/pd_brake
    to: com.bxi.basic_actions/pd_brake
  - from: sonic_zerolab
    event: com.bxi.basic_actions/recover
    to: com.bxi.basic_actions/recover
    transition: soft_switch
```

- [ ] **Step 8: Run state, manifest, loader and PICO regressions**

```bash
python3 -m pytest -q \
  src/bxi_example_py_elf3/test/test_zerolab_manifest.py \
  src/bxi_example_py_elf3/test/test_zerolab_pose_contract.py \
  src/bxi_example_py_elf3/test/test_zerolab_lifecycle.py \
  src/bxi_example_py_elf3/test/test_sonic_ordered_playout.py \
  src/bxi_example_py_elf3/test/test_sonic_python_runtime.py
```

Expected: all pass; PICO remains 9, ZeroLab is 11, node groups are disjoint and states share one policy.

- [ ] **Step 9: Commit the state integration**

```bash
git add \
  src/bxi_example_py_elf3/mods/com.bxi.sonic/mod.yaml \
  src/bxi_example_py_elf3/mods/com.bxi.sonic/plugin.py \
  src/bxi_example_py_elf3/mods/com.bxi.sonic/state.py \
  src/bxi_example_py_elf3/test/test_zerolab_manifest.py
git commit -m "feat(sonic): add independent ZeroLab teleop state"
```

---

### Task 4: Document, build and verify the complete realtime Sim2Sim path

**Files:**
- Modify: `src/bxi_example_py_elf3/mods/com.bxi.sonic/README.md`
- Verify: `src/bxi_example_py_elf3/data/mujoco_simulation/elf3.xml`
- Verify: `src/bxi_example_py_elf3/data/mujoco_simulation/terrains/universal_test/universal_test_terrain.xml`
- Verify: `install/share/bxi_example_py_elf3/mods/com.bxi.sonic/mod.yaml`
- Verify: `install/share/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/source_node.py`

**Interfaces:**
- Consumes: the installed `sonic_zerolab` manifest, ROS event messages on `/motion_commands`, Windows UDP `192.168.89.171 -> 192.168.88.161:18000`, and the existing `example_demo.launch.py`.
- Produces: copyable two-terminal operator instructions and evidence that the installed Mod drives the rich `elf3.xml` universal-test MuJoCo scene.

- [ ] **Step 1: Add exact ZeroLab realtime operating instructions to the SONIC README**

Append a `## ZeroLab 实时 MuJoCo 遥操` section containing these fixed prerequisites:

```text
Windows MotionCaptureMaster：镜像关闭
Windows sender：192.168.89.171
Ubuntu receiver：192.168.88.161:18000
UDP rate：50 Hz
UDP payload：992 bytes
ZeroLab activation：btn_10=11
```

Document terminal 1 exactly as:

```bash
cd /home/fazepurple/ros2_ws/bxi_rl_controller_ros2_example_dev/.worktrees/konodoki-dev
source /opt/ros/humble/setup.bash
source /home/fazepurple/ros2_ws/bxi_ros2_pkg/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=42
ros2 launch bxi_example_py_elf3 example_demo.launch.py
```

Document terminal 2 setup exactly as:

```bash
cd /home/fazepurple/ros2_ws/bxi_rl_controller_ros2_example_dev/.worktrees/konodoki-dev
source /opt/ros/humble/setup.bash
source /home/fazepurple/ros2_ws/bxi_ros2_pkg/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=42
```

Document state entry as short, independently copyable commands. Before the last three commands the operator must already hold a standard T-pose:

```bash
ros2 topic pub --once /motion_commands communication/msg/MotionCommands '{}'
ros2 topic pub --once /motion_commands communication/msg/MotionCommands '{btn_3: 1}'
ros2 topic pub --once /motion_commands communication/msg/MotionCommands '{}'
sleep 3

ros2 topic pub --once /motion_commands communication/msg/MotionCommands '{btn_1: 1}'
ros2 topic pub --once /motion_commands communication/msg/MotionCommands '{}'
sleep 5

ros2 topic pub --once /motion_commands communication/msg/MotionCommands '{btn_10: 11}'
ros2 topic pub --once /motion_commands communication/msg/MotionCommands '{}'
sleep 1
```

Document the readiness rule exactly: hold T-pose until terminal 1 has shown both `ZeroLab stream ready; frame=...` and `PICO source chunks ready; sent=...`; MuJoCo is expected to remain on the idle policy pose during the 100-frame calibration, so a displayed T-pose is not the readiness signal.

Document diagnostics:

```bash
ros2 topic echo --no-daemon --once --full-length --field data /simulation/state_machine_info std_msgs/msg/String
ss -H -lunp 'sport = :18000'
ss -H -ltnp 'sport = :5558'
ss -H -ltnp 'sport = :5557'
```

Document normal exit and emergency actions as individual commands:

```bash
# Return to normal and release 18000/5558/5557.
ros2 topic pub --once /motion_commands communication/msg/MotionCommands '{btn_1: 1}'
ros2 topic pub --once /motion_commands communication/msg/MotionCommands '{}'

# Emergency PD brake.
ros2 topic pub --once /motion_commands communication/msg/MotionCommands '{btn_3: 1}'
ros2 topic pub --once /motion_commands communication/msg/MotionCommands '{}'

# Zero torque only when the operator intentionally requests it.
ros2 topic pub --once /motion_commands communication/msg/MotionCommands '{btn_2: 1}'
ros2 topic pub --once /motion_commands communication/msg/MotionCommands '{}'
```

State that `zerolab.record_cli`, UDP replay programs, standalone `zerolab_source`, and standalone bridge processes must not run concurrently with `sonic_zerolab`.

- [ ] **Step 2: Run the complete focused test suite**

```bash
source /opt/ros/humble/setup.bash
source /home/fazepurple/ros2_ws/bxi_ros2_pkg/setup.bash
cd /home/fazepurple/ros2_ws/bxi_rl_controller_ros2_example_dev/.worktrees/konodoki-dev
python3 -m pytest -q \
  src/bxi_example_py_elf3/test/test_zerolab_protocol.py \
  src/bxi_example_py_elf3/test/test_zerolab_udp_receiver.py \
  src/bxi_example_py_elf3/test/test_zerolab_converter.py \
  src/bxi_example_py_elf3/test/test_zerolab_recording.py \
  src/bxi_example_py_elf3/test/test_zerolab_pose_contract.py \
  src/bxi_example_py_elf3/test/test_zerolab_lifecycle.py \
  src/bxi_example_py_elf3/test/test_zerolab_manifest.py \
  src/bxi_example_py_elf3/test/test_sonic_ordered_playout.py \
  src/bxi_example_py_elf3/test/test_sonic_python_runtime.py \
  src/bxi_example_py_elf3/test/test_state_machine_inspector.py
```

Expected: all selected tests pass. Any golden-vector, v5 parser, lifecycle, PICO regression or manifest failure blocks the build.

- [ ] **Step 3: Build the worktree installation**

```bash
source /opt/ros/humble/setup.bash
source /home/fazepurple/ros2_ws/bxi_ros2_pkg/setup.bash
cd /home/fazepurple/ros2_ws/bxi_rl_controller_ros2_example_dev/.worktrees/konodoki-dev
bash build.sh
```

Expected: `colcon build --merge-install` exits 0. Do not source the original checkout's `install/local_setup.bash` for this verification.

- [ ] **Step 4: Verify the installed Mod and universal-test scene**

```bash
test -f install/share/bxi_example_py_elf3/mods/com.bxi.sonic/zerolab/source_node.py
test -f install/share/bxi_example_py_elf3/mods/com.bxi.sonic/mod.yaml
test -f install/share/bxi_example_py_elf3/data/mujoco_simulation/elf3.xml
test -f install/share/bxi_example_py_elf3/data/mujoco_simulation/terrains/universal_test/universal_test_terrain.xml
rg -n 'entrypoint: zerolab.source_node:create_node|value: 11|sonic_zerolab' \
  install/share/bxi_example_py_elf3/mods/com.bxi.sonic/mod.yaml
rg -n 'terrains/universal_test/universal_test_terrain.xml' \
  install/share/bxi_example_py_elf3/data/mujoco_simulation/elf3.xml
```

Expected: all `test` commands exit 0; the installed manifest reports the ZeroLab source, state and event value 11; installed `elf3.xml` includes the universal test terrain.

- [ ] **Step 5: Compile the installed rich scene headlessly with MuJoCo**

```bash
python3 - <<'PY'
from pathlib import Path
import mujoco

xml = Path(
    "install/share/bxi_example_py_elf3/data/mujoco_simulation/elf3.xml"
).resolve()
model = mujoco.MjModel.from_xml_path(str(xml))
assert model.nbody > 1
assert model.ngeom > 1
print(f"compiled={xml}")
print(f"nbody={model.nbody} ngeom={model.ngeom} nq={model.nq} nv={model.nv}")
PY
```

Expected: compilation succeeds without opening a GUI and prints nontrivial model counts.

- [ ] **Step 6: Run the manual Windows-to-rich-MuJoCo acceptance**

Use the README terminal 1 and terminal 2 commands. Verify in order:

1. Windows packets are visible with `sudo tcpdump -ni enp5s0 -c 5 'udp dst port 18000'` and each packet is 992 bytes.
2. `btn_10=11` changes the state to `com.bxi.sonic/sonic_zerolab` and starts listeners on 18000, 5558 and 5557.
3. The MuJoCo window contains the universal test scene, not the old empty floor.
4. Hold T-pose until both readiness logs appear, then test left arm, right arm, both arms, squat, left/right turn and marching in place.
5. Return to normal and prove `ss` has no listeners on 18000, 5558 or 5557.
6. Re-enter `sonic_zerolab` once and repeat a left/right arm check to prove lifecycle restart and port reuse.
7. Return to normal, activate the unchanged PICO state with `btn_10=9`, and confirm its manager/bridge lifecycle still starts; do not require PICO body tracking for this lifecycle-only regression if the headset is unavailable.

Record pass/fail for every item. Visual motion acceptance must remain pending until the user observes the GUI; automated tests alone cannot mark it passed.

- [ ] **Step 7: Commit documentation and verification evidence**

```bash
git add src/bxi_example_py_elf3/mods/com.bxi.sonic/README.md
git commit -m "docs(sonic): document ZeroLab realtime MuJoCo operation"
```

Do not commit generated `build/`, `install/`, `log/`, trace directories or test caches.

---

## Final verification checklist

- [ ] `git status --short --branch` shows only the intended committed branch history and no generated artifacts.
- [ ] `git diff konodoki/dev...HEAD --check` produces no whitespace errors.
- [ ] `git diff --stat konodoki/dev...HEAD` contains only the design/plan docs, verified ZeroLab package/tests, SONIC README, manifest, plugin and state changes.
- [ ] `git hash-object` confirms `protocol.py`, `udp_receiver.py`, `converter.py` and `recording.py` still equal their immutable source blobs.
- [ ] Full focused pytest command passes after the build.
- [ ] `bash build.sh` passes and the installed tree contains the ZeroLab source and rich terrain.
- [ ] Installed `elf3.xml` compiles with MuJoCo.
- [ ] `btn_10=9` remains PICO; `btn_10=11` is ZeroLab; `btn_10=4` remains untouched.
- [ ] ZeroLab state owns neither head joints nor hardware grippers.
- [ ] Manual GUI acceptance and post-exit port release are recorded separately from automated verification.
- [ ] No push, merge or Sim2Real action is performed before the user visually accepts the MuJoCo result.
