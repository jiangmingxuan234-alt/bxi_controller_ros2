import os
from pathlib import Path
import subprocess
import time


SCRIPT_DIR = Path(__file__).resolve().parent
NETWORK_HELPER = SCRIPT_DIR / "zerolab-network-config"


def test_remote_service_does_not_pull_in_hardware():
    unit = (SCRIPT_DIR / "ros_elf_launch.service").read_text()

    assert "Wants=zerolab-hardware.service" not in unit
    assert "After=zerolab-hardware.service" in unit


def _write_executable(path, body):
    path.write_text("#!/bin/sh\nset -eu\n" + body)
    path.chmod(0o755)


def _wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _supervisor_fixture(
    tmp_path, *, address_present=False, arp_value="0", interval="0.05"
):
    log = tmp_path / "commands.log"
    notify_log = tmp_path / "notify.log"
    address_state = tmp_path / "address-present"
    arp_state = tmp_path / "arp-value"
    fake_ip = tmp_path / "ip"
    fake_sysctl = tmp_path / "sysctl"
    fake_notify = tmp_path / "systemd-notify"
    state_dir = tmp_path / "state"

    if address_present:
        address_state.touch()
    arp_state.write_text(f"{arp_value}\n")

    _write_executable(
        fake_ip,
        """printf 'ip %s\\n' "$*" >> "$ZEROLAB_TEST_LOG"
if [ "$1 $2 $3 $4" = "-4 address show dev" ]; then
  if [ -f "$ZEROLAB_ADDRESS_STATE" ]; then
    printf '    inet 192.168.88.213/32 scope global enp-test\\n'
  fi
  exit 0
fi
if [ "$1 $2" = "address add" ]; then
  : > "$ZEROLAB_ADDRESS_STATE"
  exit 0
fi
if [ "$1 $2" = "address delete" ]; then
  rm -f "$ZEROLAB_ADDRESS_STATE"
  exit 0
fi
exit 0
""",
    )
    _write_executable(
        fake_sysctl,
        """printf 'sysctl %s\\n' "$*" >> "$ZEROLAB_TEST_LOG"
if [ "$1" = "-n" ]; then
  cat "$ZEROLAB_ARP_STATE"
  exit 0
fi
if [ "$1" = "-w" ]; then
  VALUE=${2#*=}
  printf '%s\\n' "$VALUE" > "$ZEROLAB_ARP_STATE"
  exit 0
fi
exit 2
""",
    )
    _write_executable(
        fake_notify,
        """printf '%s\\n' "$*" >> "$ZEROLAB_NOTIFY_LOG"
""",
    )

    env = os.environ.copy()
    env.update(
        {
            "ZEROLAB_IP_BIN": str(fake_ip),
            "ZEROLAB_SYSCTL_BIN": str(fake_sysctl),
            "ZEROLAB_SYSTEMD_NOTIFY_BIN": str(fake_notify),
            "ZEROLAB_NETWORK_STATE_DIR": str(state_dir),
            "ZEROLAB_ETH": "enp-test",
            "ZEROLAB_WIFI": "wlan-test",
            "ZEROLAB_RECONCILE_SECONDS": interval,
            "ZEROLAB_TEST_LOG": str(log),
            "ZEROLAB_NOTIFY_LOG": str(notify_log),
            "ZEROLAB_ADDRESS_STATE": str(address_state),
            "ZEROLAB_ARP_STATE": str(arp_state),
        }
    )
    return {
        "env": env,
        "log": log,
        "notify_log": notify_log,
        "address_state": address_state,
        "arp_state": arp_state,
        "state_dir": state_dir,
    }


def test_network_unit_uses_transactional_helper():
    unit = (SCRIPT_DIR / "zerolab-network.service").read_text()

    assert NETWORK_HELPER.exists()
    assert "ExecStart=/usr/local/libexec/zerolab-network-config start" in unit
    assert "ExecStop=/usr/local/libexec/zerolab-network-config stop" in unit
    assert "ExecStart=/usr/sbin/ip " not in unit
    assert "ExecStart=/usr/sbin/sysctl " not in unit


def test_network_helper_rolls_back_address_when_sysctl_fails(tmp_path):
    assert NETWORK_HELPER.exists()
    log = tmp_path / "commands.log"
    address_state = tmp_path / "address-present"
    fake_ip = tmp_path / "ip"
    fake_sysctl = tmp_path / "sysctl"
    state_dir = tmp_path / "state"
    _write_executable(
        fake_ip,
        """printf 'ip %s\\n' "$*" >> "$ZEROLAB_TEST_LOG"
if [ "$1 $2 $3 $4" = "-4 address show dev" ]; then
  if [ -f "$ZEROLAB_ADDRESS_STATE" ]; then
    printf '    inet 192.168.88.213/32 scope global enp-test\\n'
  fi
  exit 0
fi
if [ "$1 $2" = "address add" ]; then
  : > "$ZEROLAB_ADDRESS_STATE"
  exit 0
fi
if [ "$1 $2" = "address delete" ]; then
  rm -f "$ZEROLAB_ADDRESS_STATE"
  exit 0
fi
exit 0
""",
    )
    _write_executable(
        fake_sysctl,
        """printf 'sysctl %s\\n' "$*" >> "$ZEROLAB_TEST_LOG"
if [ "$1" = "-n" ]; then
  printf '0\\n'
  exit 0
fi
case "$2" in
  *=1) exit 42 ;;
  *) exit 0 ;;
esac
""",
    )
    env = os.environ.copy()
    env.update(
        {
            "ZEROLAB_IP_BIN": str(fake_ip),
            "ZEROLAB_SYSCTL_BIN": str(fake_sysctl),
            "ZEROLAB_NETWORK_STATE_DIR": str(state_dir),
            "ZEROLAB_ETH": "enp-test",
            "ZEROLAB_WIFI": "wlan-test",
            "ZEROLAB_TEST_LOG": str(log),
            "ZEROLAB_ADDRESS_STATE": str(address_state),
        }
    )

    result = subprocess.run(
        [str(NETWORK_HELPER), "start"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    commands = log.read_text().splitlines()
    assert "ip address add 192.168.88.213/32 dev enp-test" in commands
    assert "ip address delete 192.168.88.213/32 dev enp-test" in commands
    assert (
        "sysctl -w net.ipv4.conf.wlan-test.arp_ignore=0" in commands
    )
    assert not state_dir.exists()


def test_network_helper_leaves_no_state_when_snapshot_fails(tmp_path):
    log = tmp_path / "commands.log"
    fake_ip = tmp_path / "ip"
    fake_sysctl = tmp_path / "sysctl"
    state_dir = tmp_path / "state"
    _write_executable(fake_ip, "exit 0\n")
    _write_executable(
        fake_sysctl,
        """printf 'sysctl %s\\n' "$*" >> "$ZEROLAB_TEST_LOG"
if [ "$1" = "-n" ]; then
  exit 23
fi
exit 0
""",
    )
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
        [str(NETWORK_HELPER), "start"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert not state_dir.exists()


def test_network_helper_does_not_own_address_when_add_fails(tmp_path):
    log = tmp_path / "commands.log"
    fake_ip = tmp_path / "ip"
    fake_sysctl = tmp_path / "sysctl"
    state_dir = tmp_path / "state"
    _write_executable(
        fake_ip,
        """printf 'ip %s\\n' "$*" >> "$ZEROLAB_TEST_LOG"
if [ "$1 $2" = "address add" ]; then
  exit 24
fi
exit 0
""",
    )
    _write_executable(
        fake_sysctl,
        """printf 'sysctl %s\\n' "$*" >> "$ZEROLAB_TEST_LOG"
if [ "$1" = "-n" ]; then
  printf '1\\n'
fi
exit 0
""",
    )
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
        [str(NETWORK_HELPER), "start"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    commands = log.read_text().splitlines()
    assert "ip address add 192.168.88.213/32 dev enp-test" in commands
    assert "ip address delete 192.168.88.213/32 dev enp-test" not in commands
    assert not state_dir.exists()


def test_network_helper_retains_failed_cleanup_for_retry(tmp_path):
    log = tmp_path / "commands.log"
    delete_failed = tmp_path / "delete-failed"
    address_state = tmp_path / "address-present"
    fake_ip = tmp_path / "ip"
    fake_sysctl = tmp_path / "sysctl"
    state_dir = tmp_path / "state"
    _write_executable(
        fake_ip,
        """printf 'ip %s\\n' "$*" >> "$ZEROLAB_TEST_LOG"
if [ "$1 $2 $3 $4" = "-4 address show dev" ]; then
  if [ -f "$ZEROLAB_ADDRESS_STATE" ]; then
    printf '    inet 192.168.88.213/32 scope global enp-test\\n'
  fi
  exit 0
fi
if [ "$1 $2" = "address add" ]; then
  : > "$ZEROLAB_ADDRESS_STATE"
  exit 0
fi
if [ "$1 $2" = "address delete" ] && [ ! -f "$ZEROLAB_DELETE_FAILED" ]; then
  : > "$ZEROLAB_DELETE_FAILED"
  exit 25
fi
if [ "$1 $2" = "address delete" ]; then
  rm -f "$ZEROLAB_ADDRESS_STATE"
fi
exit 0
""",
    )
    _write_executable(
        fake_sysctl,
        """printf 'sysctl %s\\n' "$*" >> "$ZEROLAB_TEST_LOG"
if [ "$1" = "-n" ]; then
  printf '3\\n'
fi
exit 0
""",
    )
    env = os.environ.copy()
    env.update(
        {
            "ZEROLAB_IP_BIN": str(fake_ip),
            "ZEROLAB_SYSCTL_BIN": str(fake_sysctl),
            "ZEROLAB_NETWORK_STATE_DIR": str(state_dir),
            "ZEROLAB_ETH": "enp-test",
            "ZEROLAB_WIFI": "wlan-test",
            "ZEROLAB_TEST_LOG": str(log),
            "ZEROLAB_DELETE_FAILED": str(delete_failed),
            "ZEROLAB_ADDRESS_STATE": str(address_state),
        }
    )

    started = subprocess.run(
        [str(NETWORK_HELPER), "start"], env=env, check=False
    )
    first_stop = subprocess.run(
        [str(NETWORK_HELPER), "stop"], env=env, check=False
    )

    assert started.returncode == 0
    assert first_stop.returncode != 0
    assert (state_dir / "address.added").exists()

    second_stop = subprocess.run(
        [str(NETWORK_HELPER), "stop"], env=env, check=False
    )

    assert second_stop.returncode == 0
    assert not state_dir.exists()


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
        assert _wait_until(fixture["notify_log"].exists)

        fixture["address_state"].unlink()
        fixture["arp_state"].write_text("0\n")

        assert _wait_until(fixture["address_state"].exists)
        assert _wait_until(
            lambda: fixture["arp_state"].read_text().strip() == "1"
        )
    finally:
        if process.poll() is None:
            process.terminate()
        returncode = process.wait(timeout=2)

    assert returncode == 0
    assert not fixture["address_state"].exists()
    assert fixture["arp_state"].read_text().strip() == "0"
    assert not fixture["state_dir"].exists()


def test_network_supervisor_restores_preexisting_address_on_stop(tmp_path):
    fixture = _supervisor_fixture(
        tmp_path, address_present=True, arp_value="3"
    )
    started = subprocess.run(
        [str(NETWORK_HELPER), "start"], env=fixture["env"], check=False
    )
    assert started.returncode == 0
    assert fixture["address_state"].exists()
    assert fixture["arp_state"].read_text().strip() == "1"

    fixture["address_state"].unlink()
    stopped = subprocess.run(
        [str(NETWORK_HELPER), "stop"], env=fixture["env"], check=False
    )

    assert stopped.returncode == 0
    assert fixture["address_state"].exists()
    assert fixture["arp_state"].read_text().strip() == "3"
    assert not fixture["state_dir"].exists()


def test_upgrade_stops_legacy_network_unit_before_replacing_it():
    readme = (SCRIPT_DIR / "README-zerolab-services.md").read_text()
    assert "sudo systemctl stop zerolab-network.service" in readme
    stop = readme.index("sudo systemctl stop zerolab-network.service")
    install = readme.index(
        '"$CAND/script/zerolab-network.service"'
    )

    assert stop < install
