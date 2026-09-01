# ZeroLab 开机服务部署（elf3-81）

这些服务将 ZeroLab 接收地址、候选硬件栈和遥控器统一到 ROS Domain 111。GitHub push
只负责传输版本；机器人仍必须检出目标提交、重新构建并安装服务后才会生效。

## 1. 构建候选包

在机器人候选目录执行：

```bash
set -e

CAND=/home/bxi/zerolab-wireless-auto-recovery-20260824
cd "$CAND"

source /opt/ros/humble/setup.bash
source /opt/bxi/bxi_ros2_pkg/setup.bash
source /home/bxi/bxi_ws/bxi_rl_controller_ros2_example/install/setup.bash

PYTHONPATH="$CAND/src/bxi_example_py_elf3:$CAND/src/bxi_example_py_elf3/mods/com.bxi.sonic${PYTHONPATH:+:$PYTHONPATH}" \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python3 -m pytest -q src/bxi_example_py_elf3/test

colcon --log-base log-wireless-auto-recovery build \
  --merge-install \
  --base-paths src \
  --packages-ignore bxi_depth_camera \
  --packages-select bxi_example_py_elf3 remote_controller \
  --allow-overriding bxi_example_py_elf3 remote_controller \
  --build-base build-wireless-auto-recovery \
  --install-base install-wireless-auto-recovery \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
```

必须同时构建 `remote_controller`，否则单独 A/Y 的 `btn_10=11/12` 映射不会安装到候选前缀。

## 2. 安装服务文件

机器人可靠支撑、物理急停可用、安全员就位后执行：

```bash
set -e

CAND=/home/bxi/zerolab-wireless-auto-recovery-20260824

sudo systemctl stop ros_elf_launch.service
sudo systemctl stop zerolab-hardware.service 2>/dev/null || true

if pgrep -af \
  '[h]ardware_elf3|[b]xi_example_py_elf3_demo|[z]erolab_source'
then
  echo 'STOP: 仍有旧硬件控制器运行'
  exit 1
fi

sudo cp -n \
  /etc/systemd/system/ros_elf_launch.service \
  /etc/systemd/system/ros_elf_launch.service.pre-zerolab

sudo install -m 0644 \
  "$CAND/script/zerolab-network.service" \
  /etc/systemd/system/zerolab-network.service
sudo install -m 0644 \
  "$CAND/script/zerolab-hardware.service" \
  /etc/systemd/system/zerolab-hardware.service
sudo install -m 0644 \
  "$CAND/script/ros_elf_launch.service" \
  /etc/systemd/system/ros_elf_launch.service

sudo systemctl daemon-reload
sudo systemctl enable zerolab-network.service
sudo systemctl enable zerolab-hardware.service
sudo systemctl enable ros_elf_launch.service
```

`enable` 只设置下次开机启动；上面的步骤不会立即启动电机控制器。

## 3. 首次受控启动

仍保持可靠支撑和急停准备：

```bash
set -e

sudo systemctl start zerolab-network.service

ip -4 -br addr show enp86s0
test "$(cat /proc/sys/net/ipv4/conf/wlo1/arp_ignore)" = 1

sudo systemctl start zerolab-hardware.service
sudo systemctl start ros_elf_launch.service

systemctl --no-pager --full status \
  zerolab-network.service \
  zerolab-hardware.service \
  ros_elf_launch.service
```

## 4. Domain 111 与单实例验证

```bash
set -e

export ROS_DOMAIN_ID=111
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_LOCALHOST_ONLY=0
export CYCLONEDDS_URI='<CycloneDDS><Domain Id="any"><General><Interfaces><NetworkInterface name="lo" multicast="true"/></Interfaces><AllowMulticast>true</AllowMulticast></General></Domain></CycloneDDS>'

source /opt/ros/humble/setup.bash
source /opt/bxi/bxi_ros2_pkg/setup.bash
source /home/bxi/bxi_ws/bxi_rl_controller_ros2_example/install/setup.bash
source /home/bxi/zerolab-wireless-auto-recovery-20260824/install-wireless-auto-recovery/setup.bash

echo '=== PROCESSES ==='
pgrep -af '[h]ardware_elf3|[b]xi_example_py_elf3_demo|[r]emote_controller'

echo '=== ROS ==='
ros2 node list --no-daemon
ros2 topic info /motion_commands

echo '=== NETWORK ==='
ip -4 -br addr show enp86s0
cat /proc/sys/net/ipv4/conf/wlo1/arp_ignore
```

期望只有一个 `hardware_elf3` 和一个 `bxi_example_py_elf3_demo`，并且
`/motion_commands` 的订阅者数量为 1。

ZeroLab 发送端开启后，在 ARM 前验证原始 UDP：

```bash
sudo timeout 10 tcpdump \
  -ni enp86s0 -nn -c 20 \
  'udp and src host 192.168.89.171 and dst host 192.168.88.213 and dst port 18000'
```

## 5. 平板操作

```text
RB+B  -> PD Brake
RB+X  -> Normal
A     -> 进入 ZeroLab，等待 WAIT_ARM
Y     -> 2 秒 ARM
Y     -> 2 秒暂停回实时 Normal；ZeroLab 链路保持运行
Y     -> 操作者回到中立且再次到 WAIT_ARM 后重新 ARM
RB+X  -> 完全退出 ZeroLab并回到 Normal
RB+B  -> 异常时立即进入 PD Brake
```

A/Y 都是上升沿单脉冲：按住不会重复触发，必须完全松开后下一次按下才会再次生效。

Start/Menu 现在调用 `systemctl start zerolab-hardware.service`，不会启动第二套硬件栈。
Stop/View 会停止整个候选硬件服务，仅用于维护；它不是退出 ARM 的按钮。

## 6. 回退服务配置

先用平板进入 PD Brake，确认可靠支撑，再执行：

```bash
set -e

sudo systemctl disable --now ros_elf_launch.service
sudo systemctl disable --now zerolab-hardware.service
sudo systemctl disable --now zerolab-network.service

sudo install -m 0644 \
  /etc/systemd/system/ros_elf_launch.service.pre-zerolab \
  /etc/systemd/system/ros_elf_launch.service
sudo systemctl daemon-reload
sudo systemctl enable --now ros_elf_launch.service
```

停止 `zerolab-network.service` 会删除 `192.168.88.213/32` 并将
`net.ipv4.conf.wlo1.arp_ignore` 恢复为 `0`。
