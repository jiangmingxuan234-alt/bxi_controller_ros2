# ZeroLab Boot Services Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist the ZeroLab receive address and Wi-Fi ARP policy, then boot exactly one candidate hardware stack and the updated tablet controller in ROS Domain 111.

**Architecture:** A root oneshot unit owns `192.168.88.213/32` and `arp_ignore=1`. A separate hardware unit requires that network unit and launches the candidate overlay; the existing remote-controller unit sources the same overlay and delegates its Start/Stop hardware command to systemd instead of directly spawning a second controller.

**Tech Stack:** systemd, bash, ROS 2 Humble, CycloneDDS, remote-controller YAML.

## Global Constraints

- Robot interface names are `enp86s0` and `wlo1`.
- ZeroLab receive address is exactly `192.168.88.213/32`.
- All ROS processes use `ROS_DOMAIN_ID=111` and the existing loopback CycloneDDS URI.
- Candidate prefix is `/home/bxi/zerolab-wireless-auto-recovery-20260824/install-wireless-auto-recovery`.
- Never run two `hardware_elf3`/`bxi_example_py_elf3_demo` stacks concurrently.
- Keep BMS and voice Start/Stop commands unchanged.

---

### Task 1: Persistent receive-network unit

**Files:**
- Create: `script/zerolab-network.service`

**Interfaces:**
- Produces: systemd unit `zerolab-network.service`.

- [ ] **Step 1: Create the oneshot unit**

The unit must run `ip address replace 192.168.88.213/32 dev enp86s0` and `sysctl -w net.ipv4.conf.wlo1.arp_ignore=1`, remain active, and undo both settings when explicitly stopped.

- [ ] **Step 2: Validate unit syntax**

Run:

```bash
systemd-analyze verify script/zerolab-network.service
```

Expected: exit code 0 with no unit-file errors.

### Task 2: Candidate hardware service and remote-controller ownership

**Files:**
- Create: `script/zerolab-hardware.service`
- Modify: `script/ros_elf_launch.service`
- Modify: `src/remote_controller/config/xbox_default.yaml`

**Interfaces:**
- Consumes: `zerolab-network.service`.
- Produces: `zerolab-hardware.service` as the only hardware launcher.
- Produces: tablet Start/Stop delegation to `systemctl start/stop zerolab-hardware.service`.

- [ ] **Step 1: Create the hardware unit**

Source ROS Humble, BXI underlays, and the fixed candidate prefix; set Domain 111 and the existing CycloneDDS URI; launch `example_demo_hw.launch.py`; use a control-group SIGINT shutdown and restart only on failure.

- [ ] **Step 2: Point the remote service at the candidate overlay**

Keep `ros_elf_launch.service` remote-controller-only, set Domain 111, and source the candidate overlay so the deployed A/Y mapping is used.

- [ ] **Step 3: Replace only direct hardware process management**

In `xbox_default.yaml`, replace the direct hardware `ros2 launch` with `systemctl start zerolab-hardware.service`; replace the three hardware `killall` entries with `systemctl stop zerolab-hardware.service`; retain BMS, camera, and voice commands.

- [ ] **Step 4: Validate configs and unit syntax**

Run the remote-controller config tests and `systemd-analyze verify` for all three units.

### Task 3: Deployment documentation and full verification

**Files:**
- Create: `script/README-zerolab-services.md`

**Interfaces:**
- Produces: exact build, install, enable, rollback, and reboot-validation commands for `elf3-81`.

- [ ] **Step 1: Document candidate build including both packages**

Build `bxi_example_py_elf3` and `remote_controller` into `install-wireless-auto-recovery`.

- [ ] **Step 2: Document service installation and enablement**

Install the three unit files, daemon-reload, and enable network, hardware, and remote services without launching any direct second hardware process.

- [ ] **Step 3: Document reboot validation**

Verify one hardware controller, one `/motion_commands` subscriber, Domain 111 discovery, the `/32` address, `arp_ignore=1`, and strict UDP arrival before hardware ARM.

- [ ] **Step 4: Run focused and full repository tests/build**

Preserve existing build/install/log directories and report the exact results without claiming robot deployment.
