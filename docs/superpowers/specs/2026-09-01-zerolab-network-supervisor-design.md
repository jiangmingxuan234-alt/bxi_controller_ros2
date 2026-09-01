# ZeroLab network self-healing supervisor design

## Context

`elf3-81` receives the ZeroLab UDP stream on Ethernet interface `enp86s0` at
`192.168.88.213:18000`. The robot's DHCP address remains independent, while the
ZeroLab receive address is an additional `/32` address. The Wi-Fi interface
`wlo1` must use `net.ipv4.conf.wlo1.arp_ignore=1` while ZeroLab networking is
active.

The first persistent-service deployment exposed a startup race:

1. At 16:16:50, `zerolab-network.service` added `192.168.88.213/32` while
   NetworkManager was still acquiring DHCP on `enp86s0`.
2. At 16:16:51, NetworkManager's first activation failed and it reapplied the
   connection after receiving `192.168.89.106/16`.
3. Reapplying the connection removed the additional `/32` address.
4. The service later tried to delete the already-missing address, treated
   `Address not found` as a cleanup failure, and retained an incomplete state
   directory that blocked the next start.

The network service therefore must not only persist across reboot. It must also
self-heal after Ethernet link changes, DHCP reconfiguration, or NetworkManager
restart, while retaining exact rollback semantics.

## Requirements

- Keep `192.168.88.213/32` present on `enp86s0` whenever the ZeroLab network
  service is active.
- Keep `net.ipv4.conf.wlo1.arp_ignore=1` while the service is active.
- Recover both settings within two seconds after NetworkManager or another
  local component removes or changes them.
- Continue waiting safely while the Ethernet link is absent, and recover after
  the link returns without operator intervention.
- Make cleanup idempotent. An already-missing address is a successful cleanup,
  not an error.
- On a normal stop, restore the address-presence state and `arp_ignore` value
  observed before the service started.
- On an unexpected supervisor failure, let systemd restart it automatically.
- Do not modify the persistent NetworkManager connection profile.
- Do not start a second hardware stack, alter the ROS Domain 111 arrangement,
  or bypass any robot safety protection.

## Chosen approach

Replace the one-shot network helper with a long-running systemd-managed
supervisor. The supervisor owns only the temporary runtime settings and checks
them once per second.

This is preferred over a NetworkManager dispatcher because it keeps ownership,
repair, and rollback in one process. It is preferred over modifying the
NetworkManager connection profile because it does not permanently alter a
vendor-managed profile or depend on a profile name or UUID.

## Service lifecycle

`zerolab-network.service` will run the helper's `run` action as its main process.
The unit will:

- use `Type=notify` so dependent hardware does not start until the first network
  application succeeds;
- use `Restart=on-failure` with a short restart delay;
- retain its ordering before `zerolab-hardware.service`;
- pass the Ethernet interface, Wi-Fi interface, state directory, and one-second
  reconciliation interval through explicit environment variables;
- stop cleanly through the supervisor's signal handler rather than a competing
  `ExecStop` process.

The supervisor performs these phases:

1. Validate interface names, state-directory scope, and reconciliation period.
2. Recover any state left by a previous abnormal exit using idempotent cleanup.
3. Snapshot whether the `/32` address existed and the original `arp_ignore`
   value.
4. Apply the desired address and `arp_ignore=1`.
5. Notify systemd that initial configuration is ready.
6. Reconcile the desired state once per second until asked to stop.
7. On `SIGTERM`, `SIGINT`, or normal exit, restore the snapshotted state.

The hardware service may remain running while the Ethernet cable is absent.
The existing ZeroLab stream gate remains in `WAIT_STREAM` or stale handling;
the network supervisor restores reception after the link returns.

## Ownership and rollback state

Runtime state remains under `/run/zerolab-network`, because IP addresses and
sysctl values are themselves runtime state and reset on reboot.

The state records:

- the original numeric `arp_ignore` value;
- whether `192.168.88.213/32` existed before startup;
- whether the supervisor added the address;
- whether the supervisor completed initial setup.

If the address did not exist before startup, stop removes it when present and
also succeeds when it is already absent. If the address existed before startup,
stop leaves it present; if another component removed it, stop restores it to
match the original state. The original `arp_ignore` value is restored exactly.

State files are deleted only after their corresponding rollback operation is
successful. A genuine rollback error remains retryable, but an absent address
is not classified as an error.

## Reconciliation behavior

Each reconciliation cycle:

- checks for the exact `inet 192.168.88.213/32` address on `enp86s0`;
- adds it if missing;
- reads `net.ipv4.conf.wlo1.arp_ignore` and resets it to `1` if necessary;
- emits a log only when it repairs state or an operation fails.

Transient repair failures do not discard the baseline snapshot. The supervisor
retries on the next cycle. An unrecoverable internal error exits nonzero so
systemd restarts the process. The loop does not change routes, DNS, the DHCP
address, or NetworkManager profiles.

## Testing strategy

Automated tests will use fake `ip`, `sysctl`, `systemd-notify`, and timing tools
to exercise the real shell helper.

Required regression tests:

- cleanup succeeds when a service-owned address is already absent;
- a missing address is re-added during a later reconciliation cycle;
- a changed `arp_ignore` value is repaired to `1`;
- graceful termination removes a service-added address and restores the
  original sysctl value;
- a pre-existing address is preserved or restored on stop;
- a genuine deletion failure retains state for a later retry;
- the systemd unit is long-running, restartable, readiness-notifying, and still
  ordered before the hardware service.

The complete verification set is:

- 292 SONIC Python tests;
- all ZeroLab service lifecycle tests, increasing the current seven-test set;
- focused remote-controller C++ A/Y mapping tests;
- Release builds of `bxi_example_py_elf3` and `remote_controller`;
- `systemd-analyze verify` for all installed unit files.

## Robot validation

Deployment to `elf3-81` remains offline through a checksum-verified incremental
Git bundle. Before installing the update, all hardware controller processes
must remain stopped.

Validation proceeds without ARM:

1. Install the updated helper and unit, reload systemd, and start only the
   network service.
2. Verify the service is active, the `/32` exists, `arp_ignore=1`, and strict
   ZeroLab UDP packets arrive.
3. Manually delete only `192.168.88.213/32` from `enp86s0`; verify the supervisor
   restores it within two seconds.
4. Set only `net.ipv4.conf.wlo1.arp_ignore=0`; verify it returns to `1` within
   two seconds.
5. Stop the service; verify the `/32` is absent, `arp_ignore` returns to `0`, and
   the runtime state directory is removed.
6. Start it again and repeat strict UDP verification.
7. Only after network validation passes, start one candidate hardware service
   and the Domain 111 tablet service, then validate one controller instance and
   one `/motion_commands` subscriber.

No `btn_10=12` ARM command is sent during network validation.

## Rollback

The original tablet service backup remains at
`/etc/systemd/system/ros_elf_launch.service.pre-zerolab`. Stopping the network
supervisor restores runtime networking before the candidate services are
disabled. Full rollback disables the three candidate services, restores the
original tablet unit, reloads systemd, and restarts the original service.

Git rollback remains available at the prior candidate commit
`dba187375a4f1bcd5696ddecffbc74227422768c`; the self-healing fix will be a new,
separate commit on `test/konodoki-dev` and a corresponding fork review branch.

## Acceptance criteria

- No duplicate hardware stack can start as part of network recovery.
- NetworkManager DHCP reconfiguration cannot leave the `/32` absent for more
  than two seconds while the service is active.
- Repeated start/stop cycles leave no stale runtime state.
- A stopped service restores the pre-service network state.
- All automated, build, unit-file, and robot network checks pass before any
  teleoperation test resumes.
