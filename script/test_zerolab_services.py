import os
from pathlib import Path
import subprocess


SCRIPT_DIR = Path(__file__).resolve().parent
NETWORK_HELPER = SCRIPT_DIR / "zerolab-network-config"


def test_remote_service_does_not_pull_in_hardware():
    unit = (SCRIPT_DIR / "ros_elf_launch.service").read_text()

    assert "Wants=zerolab-hardware.service" not in unit
    assert "After=zerolab-hardware.service" in unit


def _write_executable(path, body):
    path.write_text("#!/bin/sh\nset -eu\n" + body)
    path.chmod(0o755)


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
    fake_ip = tmp_path / "ip"
    fake_sysctl = tmp_path / "sysctl"
    state_dir = tmp_path / "state"
    _write_executable(
        fake_ip,
        """printf 'ip %s\\n' "$*" >> "$ZEROLAB_TEST_LOG"
if [ "$1 $2 $3 $4" = "-4 address show dev" ]; then
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
