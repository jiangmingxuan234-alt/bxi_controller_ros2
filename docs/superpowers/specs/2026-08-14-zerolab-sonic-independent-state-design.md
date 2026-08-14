# ZeroLab 独立 SONIC 状态设计

日期：2026-08-14  
目标分支：`test/konodoki-dev`，基于 `konodoki/dev@efbce8d`

## 目标

在保留现有 PICO `com.bxi.sonic/sonic_teleop` 的前提下，为 ZeroLab 增加独立的
`com.bxi.sonic/sonic_zerolab` 状态，使 Windows MotionCaptureMaster 的实时 UDP
动作能够通过当前 SONIC v5 source-chunk 协议驱动 MuJoCo ELF3。

MuJoCo 必须继续使用该分支 `example_demo.launch.py` 默认加载的
`data/mujoco_simulation/elf3.xml`。该模型已经包含
`terrains/universal_test/universal_test_terrain.xml`，即城市、工业、野外、救援、
综合障碍、耐久、建筑负障碍、室内仓储和 RGB-D 感知区组成的万能测试场。

## 明确不做

- 不修改 ZeroLab 的 T-pose 标定、四元数变换、坐标系转换、左右关节映射或腕部映射算法。
- 不修改 SONIC policy 的推理算法、10 帧窗口推进语义或模型资产。
- 不用 ZeroLab 替换现有 PICO 状态。
- 不让 ZeroLab 控制 ELF3 的两个头部关节。
- 不在 ZeroLab 状态启动 PICO manager、RoboticsService、MediaMTX 或 RTSP 推流。
- 不改实体机器人 Sim2Real 启动流程；本阶段验收目标是实时 Sim2Sim。

## 已验证的现状

1. `konodoki/dev@efbce8d` 的 SONIC Mod 版本为 `1.3.1`。
2. 现有 PICO 状态通过 `btn_10=9` 激活，不得改变。
3. 现有 bridge 向 `tcp://127.0.0.1:5557` 发布 v5 source-chunk；policy 同时保留
   `legacy_window` 解码路径，但新版 `SmplReferenceFrame` 强制需要
   `head_joint_pos`。
4. 旧 ZeroLab source 在 UDP `0.0.0.0:18000` 接收 992 字节数据，经原转换器产生
   10 帧 PICO-compatible pose chunk，并在 `tcp://127.0.0.1:5558` 发布 `pose`。
5. 旧 ZeroLab pose chunk 没有 `head_joint_pos`，因此不能原样送入当前 v5 bridge。
6. `btn_10=4` 已被 `com.bxi.pico_gmr_motion` 占用；`btn_10=1..10` 也均已有用途。

## 方案选择

采用“ZeroLab source 生成 PICO-compatible pose，复用当前 v5 bridge”的方案：

```text
Windows MotionCaptureMaster
  -> UDP 0.0.0.0:18000，50 Hz，992 bytes
  -> zerolab_source（原转换与 T-pose 标定）
  -> ZMQ tcp://127.0.0.1:5558，topic=pose
  -> 现有 pico.pose_to_smpl_ref_bridge 代码，独立 zerolab_bridge 实例
  -> ZMQ tcp://127.0.0.1:5557，topic=smpl_ref，version=5 source-chunk
  -> 共享 SonicTeleopPolicy
  -> sonic_zerolab
  -> MuJoCo ELF3 万能测试场
```

不采用以下方案：

- ZeroLab source 直接构造 v5 source-chunk：会复制 bridge 的帧连续性、epoch、去重、
  stale 和发布逻辑，增加与 PICO 链路再次漂移的风险。
- 外部运行旧 v4 bridge：它缺少 `head_joint_pos`，并会与 SONIC 状态内建节点竞争
  5557，生命周期也无法由状态机可靠回收。

## 组件设计

### ZeroLab Python 包

把已经验证过的 ZeroLab 接入包放入当前 SONIC Mod 的 `zerolab/` 目录。协议解析、UDP
接收、转换器、录制基础类型和 source 生命周期保持原实现。独立状态只使用实时 source；
录制 CLI 和离线评估工具不属于本阶段入口。

原 `PoseChunkWindow` 继续输出：

- `frame_index: int64[N]`
- `smpl_joints: float32[N,24,3]`
- `body_quat_w: float32[N,4]`，顺序为 wxyz
- `joint_pos: float32[N,29]`
- `stream_mode=1`
- `calibration_ready=true`

其中 `N=10`。为满足当前 v5 bridge 的输入契约，只增加：

```text
head_joint_pos: float32[10,2] = 0
```

这个字段只是传输兼容数据。`sonic_zerolab` 设置
`head_control_enabled=false`，因此两个零值不会获得头部关节命令所有权，也不会改变身体
姿态算法。

### Mod 节点

在 `mod.yaml` 中增加：

1. `zerolab_source`
   - `runtime: python`
   - `entrypoint: zerolab.source_node:create_node`
   - `execution: process`
   - `lifecycle: state`
   - 只关联 `sonic_zerolab`
   - UDP `0.0.0.0:18000`
   - 只接受发送者 `192.168.89.171`
   - ZeroLab pose PUB `127.0.0.1:5558`
   - 频率 50 Hz，窗口10帧，stale阈值0.5秒

2. `zerolab_bridge`
   - 复用 `pico.pose_to_smpl_ref_bridge:create_node`
   - `execution: in_process`
   - `lifecycle: state`
   - 只关联 `sonic_zerolab`
   - `depends_on: zerolab_source`
   - SUB `127.0.0.1:5558/pose`
   - PUB `127.0.0.1:5557/smpl_ref`

原 `pico_manager`、`smpl_bridge`、`mediamtx_server` 和 `head_camera_rtsp` 仍只关联
`sonic_teleop`。因此 PICO 与 ZeroLab 不会同时绑定 5557。

### 状态与事件

- 保留 `activate -> btn_10=9 -> sonic_teleop`。
- 新增 `activate_zerolab -> btn_10=11 -> sonic_zerolab`。
- `plugin.py` 为 `sonic_teleop` 和 `sonic_zerolab` 注册两个状态工厂；两者顺序使用同一个
  startup `SonicTeleopPolicy` resource，不并发运行。
- `sonic_zerolab` 从 `com.bxi.basic_actions/normal` 进入，并可返回 normal、zero_torque、
  pd_brake 或 recover。
- `sonic_zerolab` 使用 `soft_switch`，保持现有 SONIC 进入/退出语义。
- `head_control_enabled=false`、`hardware_gripper=false`。
- `require_live_reference=false`。原因是 state-scoped source/bridge 只有状态转换提交后才启动；
  若设为 true，状态可用性检查会在节点启动前等待实时参考，形成循环依赖。

## 标定与运行行为

操作者必须在发送 `btn_10=11` 前摆好 T-pose。状态转换提交后框架依次启动 source 和
bridge；source 使用原稳定窗口规则收集100帧，50 Hz下理论最短约2秒。标定期间 SONIC
使用 idle reference，MuJoCo 不显示人体 T-pose属于正常现象。

日志顺序应为：

```text
ZeroLab collecting T-pose calibration
ZeroLab stream ready; frame=...
PICO source chunks ready; sent=... newest=... epoch=...
```

第三条日志沿用通用 bridge 的 PICO 命名，但输入实际来自 ZeroLab。看到 source 与 bridge
都 ready 后，操作者才放下双臂开始动作。

离开 `sonic_zerolab` 时，框架按依赖逆序先停止 bridge，再关闭 source，最终释放
5557、5558和18000。必须先切回 normal 或 pd_brake，再关闭 MuJoCo；不能在
`sonic_zerolab` 中直接杀死数据源，因为 policy 会保持最后一个完整参考窗口。

## 错误处理与安全边界

- 非992字节、非法字段、非有限数或非 `192.168.89.171` 发送者的数据沿用 ZeroLab
  source 的校验与丢包逻辑。
- UDP超过0.5秒无新数据时，source清空窗口并停止发布；v5 bridge报告stale，不合成虚假帧。
- bridge继续执行帧号递增、重复chunk过滤、source epoch和完整10帧窗口校验。
- 18000、5558或5557绑定失败时，状态节点启动失败，状态机不得静默进入一个缺少输入的
  ZeroLab状态。
- PICO状态及其按钮、端口、厂商进程和RTSP行为必须保持现状。

## 测试设计

### 自动测试

1. 原 ZeroLab 协议、UDP、标定、转换、窗口与 stale 测试在新目录继续通过。
2. 新测试先证明 `PoseChunkWindow` 输出形状为 `head_joint_pos==(10,2)`、dtype为
   `float32` 且全零。
3. 把 ZeroLab chunk交给当前 `_parse_incoming_chunk`，证明 v5 bridge 接受所有字段且身体
   `term1_local/root_quat/wrist` 未被改写。
4. manifest测试证明：
   - `btn_10=9` 仍进入 PICO `sonic_teleop`；
   - `btn_10=11` 进入 `sonic_zerolab`；
   - 两组节点只在各自状态运行；
   - ZeroLab状态不拥有头部和夹爪。
5. plugin测试证明两个状态工厂共享同一policy resource且状态名正确。
6. 运行现有 SONIC ordered playout、状态、Mod loader和状态机回归测试。
7. `colcon build --merge-install` 成功，安装树包含 ZeroLab 包和万能测试地形。

### 手工 Sim2Sim 验收

1. Windows关闭镜像并以50 Hz向 `192.168.88.161:18000` 发送992字节UDP包。
2. Ubuntu启动 `example_demo.launch.py`，确认 MuJoCo显示万能测试场。
3. 状态按 `zero_torque -> pd_brake -> normal -> sonic_zerolab` 进入。
4. 保持T-pose直到source与bridge均ready。
5. 依次检查左右单臂、双臂、下蹲、左右转身和原地踏步。
6. 返回normal，确认18000、5558、5557监听全部释放。
7. 再进入原PICO `sonic_teleop`，确认PICO路径没有回归。

## 验收标准

- PICO和ZeroLab拥有两个独立、可重复进入和退出的状态。
- ZeroLab实时动作通过v5 source-chunk进入policy，不使用外部旧bridge或手工后台进程。
- ZeroLab身体转换结果除新增全零头部传输字段外逐项保持原算法输出。
- 标定、断流、退出和端口释放行为可从日志及端口检查明确验证。
- MuJoCo使用当前分支自带的万能测试场，而不是商业分支或旧dev安装树中的简单场景。
