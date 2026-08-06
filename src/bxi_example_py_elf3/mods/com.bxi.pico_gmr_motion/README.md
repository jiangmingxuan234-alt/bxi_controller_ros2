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

## 头部相机 RTSP 推流

进入 `pico_gmr_motion` 状态时，Mod 会额外启动两个状态级子节点：

1. `mediamtx_server` 使用 `runtime/mediamtx.yml` 启动 MediaMTX；
2. `head_camera_rtsp` 等待 MediaMTX 的 TCP 2212 端口就绪，然后直接通过
   FFmpeg 的 `libav*` C API 编码并发布 RTSP，不会启动 `ffmpeg` 命令行子进程。

离开状态时框架按相反顺序关闭，先停止推流器，再停止 MediaMTX。节点只管理自己
启动的进程，不调用 `killall`。推流故障不会切换机器人状态或停止 GMR；节点会在
网络恢复后自动重新发布。

推流器同时订阅：

```text
/simulation/head_depth_camera/color/image_raw
/hardware/head_depth_camera/color/image_raw
```

两个话题都是 `sensor_msgs/msg/Image`，使用 Sensor Data QoS 和深度 1。默认
`source_mode=auto`：连续收到 3 帧真机图像后优先真机；真机超过 0.5 秒断流后自动
使用仿真。也可以设为 `simulation` 或 `hardware` 强制选择。支持 `rgb8`、`bgr8`、
`rgba8`、`bgra8`、`mono8` 和 `8UC1`，其他编码会被拒绝并限频告警。

输出固定缩放为 424x240、H.264/YUV420P、60 FPS 时间基、3 Mbps、无 B 帧，默认
使用 `libx264` 的 `ultrafast+zerolatency`。只保留每个来源的最新 ROS 消息，编码或
网络变慢时丢弃旧图，不排队制造越来越大的延迟。RTSP 发布地址是：

```text
rtsp://127.0.0.1:2212/video
```

这是机器人本机发布地址。PICO 应使用机器人实际局域网 IP：

```text
rtsp://<机器人IP>:2212/video
```

### 支持的环境与运行依赖

当前随 Mod 构建的 x86_64 推流器针对以下环境验证：

- Ubuntu 22.04；
- ROS 2 Humble；
- GCC 11，C++17；
- FFmpeg 4.4 ABI：`libavcodec.so.58`、`libavformat.so.58`、
  `libavutil.so.56`、`libswscale.so.5`；
- MediaMTX 1.15.6；
- `libx264` H.264 encoder。

推流器动态链接目标机器的 ROS 和 FFmpeg 系统库。若目标机器库的 SONAME/ABI 与
上述不同，必须在目标机器重新运行构建工具，不能直接复制 x86_64 可执行文件。
aarch64 没有使用 x86_64 产物的 fallback，必须在 aarch64 目标机安装依赖并本机构建。

Ubuntu 22.04 / ROS Humble 安装完整编译和运行依赖：

```bash
sudo apt update
sudo apt install -y \
  ca-certificates curl ffmpeg \
  build-essential cmake pkg-config \
  libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libx264-dev \
  ros-humble-ros-base ros-humble-ament-cmake \
  ros-humble-rclcpp ros-humble-sensor-msgs
```

上面的 ROS 包要求机器已经按 ROS 2 Humble 官方 Ubuntu 安装文档配置 ROS 2 apt
软件源；如果 `apt` 提示找不到 `ros-humble-*`，先完成该软件源配置再执行此命令。
`ffmpeg` 不是推流节点的运行方式，但提供 `ffmpeg`/`ffplay` 诊断工具，也便于确认
系统的 H.264 encoder 是否包含 `libx264`。

检查 FFmpeg 开发库和 x264 encoder：

```bash
pkg-config --modversion libavcodec libavformat libavutil libswscale
ffmpeg -hide_banner -encoders | grep libx264
```

运行前必须 source ROS 环境；从工作空间启动 demo 时还要 source 安装树：

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
```

### 随 Mod 打包的 MediaMTX

MediaMTX v1.15.6 已经随 Mod 打包，不需要在目标机器另外下载或安装：

```text
runtime/linux-x86_64/mediamtx
runtime/linux-aarch64/mediamtx
```

两者都是官方 release 的静态可执行文件。官方归档与二进制 SHA-256、上游地址和
许可证记录在 `vendor/licenses/MediaMTX.PROVENANCE.txt` 与
`vendor/licenses/MediaMTX.LICENSE`。完整复制或安装这个 Mod 目录时会一起部署。

启动器按以下顺序查找 MediaMTX：

1. 环境变量 `PICO_GMR_MEDIAMTX_BIN` 指向的覆盖版本；
2. Mod 内 `runtime/linux-x86_64/mediamtx` 或
   `runtime/linux-aarch64/mediamtx`；
3. 当前 `PATH` 中的 `mediamtx`，作为兼容 fallback。

正常运行会自动选择当前 CPU 架构对应的 Mod 内版本。可以直接检查：

```bash
src/bxi_example_py_elf3/mods/com.bxi.pico_gmr_motion/runtime/\
linux-x86_64/mediamtx --version
```

在 aarch64 机器上将目录替换为 `linux-aarch64`。只有需要临时测试其他版本时才设置：

```bash
export PICO_GMR_MEDIAMTX_BIN=/absolute/path/to/mediamtx
```

MediaMTX 配置允许局域网匿名 publish/read，并监听：

- TCP 2212：RTSP 握手和可选 RTP-over-TCP；
- UDP 8002：RTP；
- UDP 8003：RTCP。

如启用了 UFW：

```bash
sudo ufw allow 2212/tcp
sudo ufw allow 8002/udp
sudo ufw allow 8003/udp
```

该匿名配置仅适合受信任的机器人局域网，不能直接暴露到公网。生产网络需要在
`runtime/mediamtx.yml` 中配置发布/读取账号和允许的 IP。

### 构建原生 ROS/FFmpeg 推流器

构建工具只在临时目录生成 CMake 中间文件，并把最终程序写入当前架构目录：

```bash
source /opt/ros/humble/setup.bash
PYTHONDONTWRITEBYTECODE=1 \
/usr/bin/python3 -B \
  src/bxi_example_py_elf3/mods/com.bxi.pico_gmr_motion/tools/build_rtsp_streamer.py
```

输出位置：

```text
bin/linux-x86_64/head_camera_rtsp_node
bin/linux-aarch64/head_camera_rtsp_node
```

非默认 ROS 安装前缀或 Python 解释器可以显式指定：

```bash
/usr/bin/python3 \
  src/bxi_example_py_elf3/mods/com.bxi.pico_gmr_motion/tools/build_rtsp_streamer.py \
  --ros-prefix /opt/ros/humble \
  --python /usr/bin/python3
```

构建后检查动态依赖，不应出现 `not found`，也不应包含开发者 Home/Conda RPATH：

```bash
source /opt/ros/humble/setup.bash
ldd \
  src/bxi_example_py_elf3/mods/com.bxi.pico_gmr_motion/bin/linux-x86_64/head_camera_rtsp_node
readelf -d \
  src/bxi_example_py_elf3/mods/com.bxi.pico_gmr_motion/bin/linux-x86_64/head_camera_rtsp_node \
  | grep -E 'RPATH|RUNPATH' || true
```

### 参数与排障

RTSP 参数位于 `mod.yaml` 的 `nodes.head_camera_rtsp.params`。常用参数包括：

- `source_mode`：`auto`、`simulation` 或 `hardware`；
- `source_timeout_s`：来源断流判定时间；
- `rtsp_url`：向本机或外部 MediaMTX 发布的 URL；
- `rtsp_transport`：`udp` 或 `tcp`；
- `encoder`：默认 `libx264`；
- `output_width/output_height/output_fps`；
- `bitrate/gop_size/network_timeout_s`。

每 5 秒输出一次 `head camera RTSP perf`，包含两个来源的接收频率、编码频率、
覆盖丢帧、非法帧和网络错误。常见故障：

- `MediaMTX executable was not found`：按上文安装，或设置
  `PICO_GMR_MEDIAMTX_BIN`；
- `waiting for MediaMTX at 127.0.0.1:2212`：检查 MediaMTX 日志和端口占用；
- `FFmpeg encoder is unavailable: libx264`：安装 `libx264` 和包含 GPL/x264 的
  FFmpeg，重新构建；
- Mod node `unavailable`：目标架构缺少 `bin/<platform>/head_camera_rtsp_node`，
  或缺少清单声明的 ROS/FFmpeg 动态库；
- 接收频率为 0：用 `ros2 topic info -v` 检查话题名、消息类型和 QoS；
- PICO 能连接但无图：检查 UDP 8002/8003、防火墙、PICO URL 中是否使用机器人 IP；
- `Address already in use`：已有 MediaMTX 或其他程序占用 2212/8002/8003。

PC 侧快速拉流验证：

```bash
ffplay -fflags nobuffer -flags low_delay -framedrop \
  -rtsp_transport udp rtsp://127.0.0.1:2212/video
```

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
