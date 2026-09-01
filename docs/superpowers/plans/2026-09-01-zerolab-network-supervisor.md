# ZeroLab Network Self-Healing Supervisor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the one-shot ZeroLab network setup with a rollback-safe supervisor that restores `192.168.88.213/32` and `wlo1.arp_ignore=1` within two seconds after NetworkManager or link reconfiguration removes them.

**Architecture:** Keep all temporary network ownership in `zerolab-network-config`, add a `run` action that performs initial transactional setup and then reconciles state once per second, and let systemd supervise that long-running process. Preserve the existing `start` and `stop` actions for focused tests and recovery, make cleanup idempotent, and never modify a NetworkManager connection profile.

**Tech Stack:** Bash, systemd, iproute2, sysctl, Python 3/pytest fake-process integration tests, ROS 2 Humble/colcon, CMake/CTest.

## Global Constraints

- Target robot remains `elf3-81`; Ethernet is `enp86s0`, Wi-Fi is `wlo1`, and the receive address is exactly `192.168.88.213/32`.
- Recovery interval defaults to exactly one second and must restore drift within two seconds.
- Do not modify persistent NetworkManager profiles, routes, DNS, DHCP configuration, ROS Domain 111 behavior, controller mappings, or robot safety protection.
- The network supervisor must not start a second hardware stack.
- Normal stop restores the exact pre-service address-presence state and numeric `arp_ignore` value.
- Runtime ownership state remains under `/run/zerolab-network`; test state may use `/tmp`.
- Production code is written only after a focused regression test has failed for the expected reason.
- Robot changes stay offline until local tests and builds pass; GitHub is updated only after robot validation passes.

## File map

- Modify `script/zerolab-network-config`: idempotent address ownership, transactional setup, one-second reconciliation loop, readiness notification, signal cleanup.
- Modify `script/zerolab-network.service`: long-running `Type=notify` unit with restart policy and no competing `ExecStop` helper.
- Modify `script/test_zerolab_services.py`: subprocess-level regression coverage for missing-address cleanup, drift repair, pre-existing address restoration, and unit lifecycle.
- Modify `script/README-zerolab-services.md`: document self-healing semantics, upgrade from failed one-shot state, validation, and rollback.
- Read-only verification of `src/remote_controller/test/control_rules_test.cpp`: ensure the unrelated A/Y mapping remains green.

---

### Task 1: Make address cleanup idempotent

**Files:**
- Modify: `script/test_zerolab_services.py:17-244`
- Modify: `script/zerolab-network-config:21-55`

**Interfaces:**
- Consumes: `ZEROLAB_IP_BIN`, `ZEROLAB_NETWORK_STATE_DIR`, `ZEROLAB_ETH`, and the existing `stop` CLI action.
- Produces: `address_present() -> shell status` and an idempotent `stop_network() -> shell status` used by startup recovery and the later supervisor.

- [ ] **Step 1: Add a failing missing-address cleanup test**

Add this focused behavior to `script/test_zerolab_services.py`, using the existing `_write_executable` helper:

```python
def test_network_helper_treats_missing_owned_address_as_clean(tmp_path):
    log = tmp_path / "commands.log"
    fake_ip = tmp_path / "ip"
    fake_sysctl = tmp_path / "sysctl"
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "address.added").touch()
    _write_executable(
        fake_ip,
        """printf 'ip %s\\n' "$*" >> "$ZEROLAB_TEST_LOG"
if [ "$1 $2 $3 $4" = "-4 address show dev" ]; then
  exit 0
fi
if [ "$1 $2" = "address delete" ]; then
  exit 25
fi
exit 0
""",
    )
    _write_executable(fake_sysctl, "exit 0\n")
    env = os.environ.copy()
    env.update(
        {
            "ZEROLAB_IP_BIN": str(fake_ip),
            "ZEROLAB_SYSCTL_BIN": str(fake_sysctl),
            "ZEROLAB_NETWORK_STATE_DIR": str(state_dir),
            "ZEROLAB_ETH": "enp-test",
            "ZEROLAB_WIFI": "wlan-test",
            "ZEROLAB_TEST_LOG": str(log),
        }
    )

    result = subprocess.run(
        [str(NETWORK_HELPER), "stop"], env=env, check=False
    )

    assert result.returncode == 0
    assert not state_dir.exists()
    assert "ip address delete" not in log.read_text()
```

- [ ] **Step 2: Run the new test and confirm RED**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python3 -m pytest -q \
  script/test_zerolab_services.py::test_network_helper_treats_missing_owned_address_as_clean
```

Expected: FAIL because the current helper calls `ip address delete` without first checking whether the address exists and returns status 25.

- [ ] **Step 3: Implement the smallest idempotent cleanup change**

Add a shared exact-address predicate before `stop_network()`:

```bash
address_present()
{
  "$IP_BIN" -4 address show dev "$ETH" |
    grep -Fq "inet $ADDRESS "
}
```

Change the `address.added` cleanup branch so deletion is attempted only when `address_present` succeeds. If it is already absent, remove `address.added` and continue successfully. Preserve the existing behavior that a genuine delete failure retains the marker and returns nonzero.

- [ ] **Step 4: Verify GREEN and existing retry behavior**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python3 -m pytest -q \
  script/test_zerolab_services.py::test_network_helper_treats_missing_owned_address_as_clean \
  script/test_zerolab_services.py::test_network_helper_retains_failed_cleanup_for_retry
```

Expected: both tests PASS.

- [ ] **Step 5: Commit the isolated cleanup fix**

```bash
git add script/zerolab-network-config script/test_zerolab_services.py
git commit -m "fix: make ZeroLab network cleanup idempotent"
```

---

### Task 2: Add long-running reconciliation and exact rollback

**Files:**
- Modify: `script/test_zerolab_services.py`
- Modify: `script/zerolab-network-config`

**Interfaces:**
- Consumes: `start_network()`, `stop_network()`, `address_present()`, injected `ZEROLAB_IP_BIN`, `ZEROLAB_SYSCTL_BIN`, `ZEROLAB_SYSTEMD_NOTIFY_BIN`, and `ZEROLAB_RECONCILE_SECONDS`.
- Produces: `reconcile_network() -> shell status`, `run_network() -> long-running process`, and the public CLI action `zerolab-network-config run`.

- [ ] **Step 1: Add reusable fake network state helpers to the test module**

Add `_wait_until(predicate, timeout=2.0)` using `time.monotonic()` and a 10 ms sleep. Add a `_supervisor_fixture(tmp_path, address_present=False, arp_value="0")` helper that creates:

- an `address-present` file representing the exact `/32`;
- an `arp-value` file containing the current numeric sysctl;
- a fake `ip` executable whose `show`, `add`, and `delete` actions read or update `address-present` and append to `commands.log`;
- a fake `sysctl` executable whose `-n` reads `arp-value` and whose `-w key=value` updates it;
- a fake `systemd-notify` executable that appends `--ready` to `notify.log`;
- an environment with `ZEROLAB_RECONCILE_SECONDS=0.05`.

The fake `ip` behavior must include the exact address output expected by production code:

```sh
if [ "$1 $2 $3 $4" = "-4 address show dev" ]; then
  if [ -f "$ZEROLAB_ADDRESS_STATE" ]; then
    printf '    inet 192.168.88.213/32 scope global enp-test\\n'
  fi
  exit 0
fi
```

- [ ] **Step 2: Add a failing drift-repair test**

```python
def test_network_supervisor_repairs_external_network_changes(tmp_path):
    fixture = _supervisor_fixture(tmp_path)
    process = subprocess.Popen(
        [str(NETWORK_HELPER), "run"], env=fixture["env"]
    )
    try:
        assert _wait_until(fixture["address_state"].exists)
        assert _wait_until(
            lambda: fixture["arp_state"].read_text().strip() == "1"
        )
        fixture["address_state"].unlink()
        fixture["arp_state"].write_text("0\n")
        assert _wait_until(fixture["address_state"].exists)
        assert _wait_until(
            lambda: fixture["arp_state"].read_text().strip() == "1"
        )
    finally:
        process.terminate()
        assert process.wait(timeout=2) == 0

    assert not fixture["address_state"].exists()
    assert fixture["arp_state"].read_text().strip() == "0"
    assert not fixture["state_dir"].exists()
```

- [ ] **Step 3: Add a failing pre-existing-address rollback test**

Start the fixture with `address_present=True`, wait for readiness, delete the simulated address while the supervisor is running, terminate it before another reconciliation cycle can be relied upon, and assert that graceful cleanup restores the address because it existed before startup. Also assert the original sysctl value, such as `3`, is restored exactly.

- [ ] **Step 4: Run both supervisor tests and confirm RED**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python3 -m pytest -q \
  script/test_zerolab_services.py -k 'supervisor_repairs or preexisting_address'
```

Expected: FAIL because the current helper rejects the `run` action with usage status 2.

- [ ] **Step 5: Refactor setup into explicit lifecycle functions**

Move the existing `start)` body into `start_network()`. During the initial snapshot:

- create `address.preexisting` when `address_present` is true;
- otherwise add the address and create `address.added` only after a successful add;
- snapshot `arp_ignore.old` before setting it to `1`;
- create `active` only after both desired settings succeed.

Extend `stop_network()` so:

- `address.added` means delete the address if present;
- `address.preexisting` means add the address if missing;
- each marker is removed only after its restore operation succeeds;
- the numeric sysctl baseline is restored exactly.

- [ ] **Step 6: Implement reconciliation and the `run` action**

Add these defaults:

```bash
SYSTEMD_NOTIFY_BIN=${ZEROLAB_SYSTEMD_NOTIFY_BIN:-/usr/bin/systemd-notify}
RECONCILE_SECONDS=${ZEROLAB_RECONCILE_SECONDS:-1}
```

Validate that the interval is a positive decimal before using it. Implement `reconcile_network()` to add the exact address when absent and set the sysctl to `1` only when its current value differs. Log only repairs and failures.

Implement `run_network()` with this lifecycle:

```bash
run_network()
{
  RUN_RC=0
  trap 'exit 0' INT TERM
  trap 'RUN_RC=$?; trap - EXIT; stop_network || RUN_RC=1; exit "$RUN_RC"' EXIT

  start_network
  "$SYSTEMD_NOTIFY_BIN" --ready

  while :; do
    sleep "$RECONCILE_SECONDS"
    if ! reconcile_network; then
      echo "ZeroLab network reconciliation failed; retrying" >&2
    fi
  done
}
```

Keep public `start` and `stop` actions for focused recovery and compatibility, and add `run)` to the CLI case statement.

- [ ] **Step 7: Run supervisor tests and all helper tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python3 -m pytest -q script/test_zerolab_services.py
```

Expected: all service tests PASS, including external address repair, sysctl repair, pre-existing address restoration, missing-address cleanup, and genuine failure retry.

- [ ] **Step 8: Commit the supervisor implementation**

```bash
git add script/zerolab-network-config script/test_zerolab_services.py
git commit -m "feat: supervise ZeroLab network runtime state"
```

---

### Task 3: Supervise the helper with systemd and update operator docs

**Files:**
- Modify: `script/test_zerolab_services.py:22-29`
- Modify: `script/zerolab-network.service:1-21`
- Modify: `script/README-zerolab-services.md:35-180`

**Interfaces:**
- Consumes: public helper action `/usr/local/libexec/zerolab-network-config run` and readiness notification.
- Produces: a restartable `zerolab-network.service` that becomes active only after initial configuration and remains ordered before `zerolab-hardware.service`.

- [ ] **Step 1: Replace the old unit assertions with failing supervisor assertions**

The unit test must require all of these exact properties:

```python
assert "Type=notify" in unit
assert "NotifyAccess=main" in unit
assert "Restart=on-failure" in unit
assert "RestartSec=1s" in unit
assert "Environment=ZEROLAB_RECONCILE_SECONDS=1" in unit
assert "ExecStart=/usr/local/libexec/zerolab-network-config run" in unit
assert "ExecStop=" not in unit
assert "RemainAfterExit=" not in unit
assert "Before=zerolab-hardware.service" in unit
```

- [ ] **Step 2: Run the unit test and confirm RED**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python3 -m pytest -q \
  script/test_zerolab_services.py::test_network_unit_uses_transactional_helper
```

Expected: FAIL because the current unit is `Type=oneshot` and runs `start`/`stop` actions.

- [ ] **Step 3: Implement the long-running unit**

Change `[Service]` to:

```ini
[Service]
Type=notify
NotifyAccess=main
Restart=on-failure
RestartSec=1s
TimeoutStopSec=10s
ExecStartPre=/usr/bin/test -d /sys/class/net/enp86s0
ExecStartPre=/usr/bin/test -d /sys/class/net/wlo1
Environment=ZEROLAB_IP_BIN=/usr/sbin/ip
Environment=ZEROLAB_SYSCTL_BIN=/usr/sbin/sysctl
Environment=ZEROLAB_SYSTEMD_NOTIFY_BIN=/usr/bin/systemd-notify
Environment=ZEROLAB_NETWORK_STATE_DIR=/run/zerolab-network
Environment=ZEROLAB_ETH=enp86s0
Environment=ZEROLAB_WIFI=wlo1
Environment=ZEROLAB_RECONCILE_SECONDS=1
ExecStart=/usr/local/libexec/zerolab-network-config run
```

Retain the existing unit ordering and install target. Do not add a dependency from the tablet service that implicitly starts hardware.

- [ ] **Step 4: Update deployment and rollback documentation**

Document that the network unit is a long-running self-healing process, that a failed older one-shot state is recovered by the updated idempotent helper, and that initial validation deliberately deletes only the `/32` and changes only `wlo1.arp_ignore` before waiting two seconds for repair. Preserve the command ordering that stops the legacy network unit before replacing its helper and unit.

- [ ] **Step 5: Verify service tests and unit syntax**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python3 -m pytest -q script/test_zerolab_services.py

systemd-analyze verify \
  script/zerolab-network.service \
  script/zerolab-hardware.service \
  script/ros_elf_launch.service
```

Expected: all service tests PASS; `systemd-analyze` returns zero. Unrelated warnings from an installed `snapd.service` do not fail verification.

- [ ] **Step 6: Commit the unit and documentation**

```bash
git add \
  script/zerolab-network.service \
  script/README-zerolab-services.md \
  script/test_zerolab_services.py
git commit -m "fix: auto-recover ZeroLab network state"
```

---

### Task 4: Full regression, Release build, and offline robot artifact

**Files:**
- Verify: `src/bxi_example_py_elf3/test`
- Verify: `src/remote_controller/test/control_rules_test.cpp`
- Runtime-only create: `build-network-supervisor/`, `install-network-supervisor/`, `log-network-supervisor/`
- Runtime-only create: `/tmp/zerolab-<short-sha>-incremental.bundle`

**Interfaces:**
- Consumes: all committed supervisor, unit, documentation, SONIC, and A/Y mapping changes.
- Produces: verified commit SHA, checksum-verified incremental Git bundle based on robot commit `dba187375a4f1bcd5696ddecffbc74227422768c`, and robot deployment commands that do not contact GitHub.

- [ ] **Step 1: Run the 292 SONIC tests and complete service suite**

Run with the installed ROS message environment sourced:

```bash
source /opt/ros/humble/setup.bash
source /opt/bxi/bxi_ros2_pkg/setup.bash
source /home/fazepurple/ros2_ws/bxi_rl_controller_ros2_example_dev/install/setup.bash

PYTHONPATH="$PWD/src/bxi_example_py_elf3:$PWD/src/bxi_example_py_elf3/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python3 -m pytest -q src/bxi_example_py_elf3/test

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python3 -m pytest -q script/test_zerolab_services.py
```

Expected: 292 SONIC tests PASS and every service test PASS.

- [ ] **Step 2: Build both packages with C++ tests enabled**

```bash
colcon --log-base log-network-supervisor build \
  --merge-install \
  --base-paths src \
  --packages-ignore bxi_depth_camera \
  --packages-select bxi_example_py_elf3 remote_controller \
  --allow-overriding bxi_example_py_elf3 remote_controller \
  --build-base build-network-supervisor \
  --install-base install-network-supervisor \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
```

Expected: both packages finish successfully.

- [ ] **Step 3: Run the focused remote-controller CTest**

```bash
ctest \
  --test-dir build-network-supervisor/remote_controller \
  --output-on-failure \
  -R remote_controller_control_rules_test
```

Expected: the A/Y edge and modifier-hardening test executable passes.

- [ ] **Step 4: Perform final repository verification**

```bash
git diff --check
git status --short
git log -8 --oneline
```

Expected: only the pre-existing untracked `build-*`, `install-*`, and `log-*` directories remain; no tracked changes are pending.

- [ ] **Step 5: Create and verify the incremental bundle**

```bash
BASE=dba187375a4f1bcd5696ddecffbc74227422768c
SHORT=$(git rev-parse --short HEAD)
BUNDLE="/tmp/zerolab-${SHORT}-incremental.bundle"

git bundle create "$BUNDLE" test/konodoki-dev "^$BASE"
git bundle verify "$BUNDLE"
git bundle list-heads "$BUNDLE"
ls -lh "$BUNDLE"
sha256sum "$BUNDLE"
```

Expected: the bundle head exactly equals the final local `test/konodoki-dev` HEAD, declares `dba1873...` as its prerequisite, and remains local until robot testing passes.

- [ ] **Step 6: Hand off the no-ARM robot validation checkpoint**

Provide commands to SCP the bundle to `bxi@192.168.89.152`, verify its checksum, fetch and detach the expected SHA, rerun tests/build, stop the failed old network unit, install only the updated helper/unit, reload systemd, and validate automatic repair of the `/32` and `arp_ignore` before starting any hardware service. Do not push the fork branch or update PR #1 until the robot completes network, single-controller, tablet, and teleoperation validation.
