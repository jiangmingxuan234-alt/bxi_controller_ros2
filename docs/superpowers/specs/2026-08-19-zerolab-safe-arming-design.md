# ZeroLab 真机安全 ARM 设计

日期：2026-08-19
目标分支：`test/konodoki-dev`，设计基线 `34bdd15`

## 背景

ZeroLab 独立 SONIC 状态已经完成构建、自动测试、原始数据录制和离线 MuJoCo
回放验证。现有结果包括：

- ZeroLab 测试 `102 passed`；
- 原始捕获 `/tmp/zerolab-arm-direction-20260819-002` 完整，共 911 帧；
- SONIC 到 MuJoCo 回放共 718 帧、14.34 秒，无跌倒和关节越界；
- 回放 RMSE 为 `0.058985 rad`；
- 实时 `pose -> smpl_ref` 的有效窗口、帧差和 clamp 指标正常；
- PICO 真机和 ZeroLab 离线 MuJoCo 的方向表现正常。

真机仍存在一个独立的接管问题：ZeroLab 完成水平 T-pose 标定后立即产生
`live_reference`，当前 `sonic_zerolab` 会立刻把策略结果发送给电机。真机肩关节因此在操作者
还保持 T-pose 时进入大角度姿态，运动平面与期望不同，之后双臂无法正常放下。

本设计不改变人体坐标转换。它把“传感器标定完成”和“允许人体参考控制真机”拆成两个明确
步骤，在二者之间增加由安全员操作的一次性 ARM 门。

## 目标

实现以下操作顺序：

```text
Normal/idle
  -> btn_10=11
ZeroLab T-pose 标定，真机保持进入时的 Normal 命令
  -> ZeroLab stream ready
等待操作者自行回到中立姿势
  -> 安全员在第二个 root 终端发送 btn_10=12
2 秒平滑接管
  -> ZeroLab 实时控制
```

同时满足：

1. 标定期间 ZeroLab 数据链和策略可以正常运行，但策略结果不能控制电机。
2. ARM 必须是安全员在第二个 root 终端执行的明确动作。
3. 不自动判断人体肩、肘是否处于精确中立角度。
4. 断流不自动切换 PD Brake，也不能在恢复后自动重新接管。
5. PICO SONIC 链路、原机器人工作区和硬件驱动保持不变。

## 明确不做

- 不修改 ZeroLab 的 T-pose 收集条件、100 帧窗口、四元数变换、坐标转换、父子局部化、
  左右映射或腕部映射。
- 不降低或重新解释 T-pose 标定标准。
- 不检查操作者的肩部、肘部或中立姿势角度。
- 不修改 `SonicTeleopState` 的 PICO 行为。
- 不修改 SONIC policy、ONNX 模型、PICO manager、PICO bridge 或厂商进程。
- 不增加 ROS service、额外控制 topic 或逐帧 ACK 链路。
- 不自动进入 Normal、PD Brake 或 zero torque。
- 不修改 `/home/bxi/bxi_ws/bxi_rl_controller_ros2_example`。
- 不在有人使用真机时启动第二套硬件控制进程。

## 方案选择

采用 `sonic_zerolab` 状态内部的独立 ARM 门。

不采用拆分成“标定状态”和“控制状态”的方案，因为它会扩大状态图和节点生命周期改动，
并可能在两态切换时重新启动 ZeroLab source 或 bridge。

不采用 source/bridge ROS service，因为阻止参考数据发布不能保证 SONIC 状态本身继续输出
Normal 命令，而且会增加新的通信面。

ARM 门只决定哪一帧电机命令被发送。ZeroLab source、bridge 和 policy 的数据处理方式保持
原样。

## 状态模型

`ZeroLabArmedTeleopState` 维护以下五个内部阶段。它们不是新的框架级机器人状态：

```text
WAIT_CALIBRATION
      | fresh live reference
      v
WAIT_ARM
      | btn_10=12
      v
BLENDING
      | 2.0 s complete
      v
ARMED
      | live reference stale > 0.5 s
      v
HOLD_STALE
      | fresh reference + btn_10=12
      +---------------------> BLENDING
```

从任意阶段离开 `sonic_zerolab` 都结束本次 ARM 会话。下次从 Normal 重新进入时，从
`WAIT_CALIBRATION` 开始。

### 进入状态

`btn_10=11` 保持为从 `com.bxi.basic_actions/normal` 进入
`com.bxi.sonic/sonic_zerolab` 的唯一事件。操作者必须先让机器人停止行走并稳定站立。

`on_prepare` 在状态转换创建前执行，因此新状态在此时把 `ctx.last_motor_frame` 复制到自己
拥有的完整机器人布局缓冲区。后续等待阶段始终输出该快照。快照必须深复制
`qpos/kp/kd/vel/torque`，不能持有框架复用缓冲区的引用。

状态的 `get_entry_frame` 也返回这份快照，使 Normal 到 ZeroLab 的现有 `soft_switch` 不会
把机器人拉向 SONIC idle reference。

与此同时，state-scoped `zerolab_source` 和 `zerolab_bridge` 按原生命周期启动。状态每个
控制周期仍推进 SONIC policy，以便 source-chunk、标定完成和 reference freshness 正常
更新；在 ARM 之前，这些策略输出只存入独立目标缓冲区，不调用到电机输出。

### 标定完成与等待 ARM

状态使用 policy 的现有
`has_fresh_live_reference(live_reference_timeout_s)` 判定完整参考已经到达。第一次返回 true
时从 `WAIT_CALIBRATION` 进入 `WAIT_ARM`，并只记录一次提示日志。

操作者在此阶段自行从 T-pose 回到预定中立姿势。系统不检查人体关节角；安全员目视确认后
在第二个 root 终端发送：

```bash
ros2 topic pub --once /motion_commands \
  communication/msg/MotionCommands '{btn_10: 12}'
ros2 topic pub --once /motion_commands \
  communication/msg/MotionCommands '{}'
```

第二条空消息用于释放一次性按钮值，保持现有遥控事件语义。

## ARM 事件

`mod.yaml` 增加：

```yaml
events:
  arm_zerolab:
    slot: btn_10
    value: 12

actions:
  - from: sonic_zerolab
    event: arm_zerolab
    action: arm_zerolab
```

`arm_zerolab` 是同一状态内的 action，不创建状态转换，也不是逐帧 ACK。

处理规则：

- `WAIT_CALIBRATION`：识别 action 但拒绝 ARM，记录“等待 ZeroLab 标定完成”。
- `WAIT_ARM` 且参考新鲜：接受 ARM，进入 `BLENDING`。
- `WAIT_ARM` 但参考已过期：拒绝 ARM，记录当前 reference age 或无有效参考。
- `HOLD_STALE` 且参考重新新鲜：接受 ARM，进入 `BLENDING`。
- `BLENDING` 或 `ARMED`：忽略重复 action，记录已经正在接管或已经接管。

无论接受还是拒绝，action 都由状态明确处理，不能让框架把合法但时机不对的 ARM 请求当成
“无处理器”异常。

## 两秒平滑接管

每次进入 `BLENDING` 时：

1. 深复制当前实际发送的最后一帧完整电机命令作为固定起点；
2. 每个控制周期推进并采样当前 ZeroLab policy 输出作为动态终点；
3. 令 `p = clamp(elapsed / 2.0, 0, 1)`；
4. 使用框架已有 running blend 默认曲线相同的 smoothstep：
   `alpha = p * p * (3 - 2 * p)`；
5. 对 `qpos/kp/kd/vel/torque` 五个字段逐项计算
   `output = start + alpha * (target - start)`；
6. `p == 1` 时进入 `ARMED`，之后直接使用实时 policy 输出。

平滑只作用于首次 ARM 和断流后的重新 ARM。完成后不增加额外延迟或永久速率过滤，不改变
原 SONIC policy 的实时响应。

如果目标本身错误，平滑只能避免瞬时跳变并为安全员提供反应时间，不能修正错误目标。
现有 `btn_3=1` PD Brake、`btn_1=1` Normal 和 `btn_2=1` zero torque 路由在所有内部阶段
保持可用，其中动作异常时优先使用 PD Brake。

## 断流行为

继续使用现有 `live_reference_timeout_s=0.5`。不增加第二套 timeout 定义。

- `WAIT_CALIBRATION` 中断流：继续保持进入时的 Normal 命令。未完成的 ZeroLab 标定窗口按
  source 现有语义清空。
- `WAIT_ARM` 中断流：继续保持进入时的 Normal 命令；恢复并重建完整参考后仍处于
  `WAIT_ARM`。
- `BLENDING` 或 `ARMED` 中断流：立即深复制并保持最后一次实际发送的完整电机命令，进入
  `HOLD_STALE`，取消 ARM，并周期性限频告警。
- 数据恢复：source 重建 10 帧窗口。已经完成的本次 T-pose 标定按 converter 现有语义
  保留，但机器人不自动恢复控制。
- 安全员重新发送 `btn_10=12`：从冻结命令重新执行完整 2 秒混合。

断流不自动切换 PD Brake、Normal 或 zero torque。退出 `sonic_zerolab` 后 source 进程关闭，
本次 converter 与 T-pose 标定被销毁；再次进入必须重新完成 T-pose。

## 代码边界

新增：

- `mods/com.bxi.sonic/zerolab/state.py`：`ZeroLabArmedTeleopState`、内部阶段和帧混合逻辑；
- ZeroLab ARM 状态的单元测试。

最小修改：

- `mods/com.bxi.sonic/plugin.py`：只有 `sonic_zerolab` 工厂改为构造新状态类；
- `mods/com.bxi.sonic/mod.yaml`：增加 `btn_10=12` action，以及
  `arm_blend_seconds: 2.0`；
- `mods/com.bxi.sonic/README.md`：更新真机标定、ARM、断流恢复和退出流程；
- manifest/lifecycle 测试：增加 ARM 事件和隔离性断言。

保持不变：

- `mods/com.bxi.sonic/state.py` 中原 `SonicTeleopState`；
- `zerolab/converter.py`、`protocol.py`、`recording.py`、`source_node.py` 和
  `udp_receiver.py`；
- PICO manager、PICO bridge、SONIC policy 和 ONNX/NPZ 资产；
- `communication` 消息定义、硬件驱动和基础状态机；
- 原机器人工作区。

参数 `arm_blend_seconds` 必须是有限且大于零的浮点数。它只属于
`ZeroLabArmedTeleopState`，不能加入 PICO 状态的构造参数或运行路径。

## 日志与可观测性

不增加新 ROS topic 或 service。终端日志必须明确显示：

```text
ZeroLab ARM phase: WAIT_CALIBRATION; holding entry motor frame
ZeroLab ARM phase: WAIT_ARM; return to neutral pose, then send btn_10=12
ZeroLab ARM accepted; blending for 2.000 s
ZeroLab ARM phase: ARMED
ZeroLab reference stale; holding last motor frame and ARM cancelled
ZeroLab reference recovered; send btn_10=12 to resume
```

拒绝、重复按钮和状态退出也必须有单次或限频日志，避免 50 Hz 控制循环刷屏。日志不得把
“数据恢复”描述为“已经恢复控制”。

## 自动测试

新增测试使用 fake policy、完整 `MotorFrame` 和可控时间步，不依赖真实 ROS 网络、UDP 或
硬件。至少覆盖：

1. `on_prepare` 深复制 `ctx.last_motor_frame`，后续修改源缓冲不会改变 hold frame。
2. `WAIT_CALIBRATION` 和 `WAIT_ARM` 的输出逐字段等于 hold frame，同时 policy 仍被推进。
3. reference 未就绪和过期时 ARM 被拒绝。
4. 新鲜 reference 下 ARM 被接受；混合在 0、1、2 秒的 alpha 分别符合 smoothstep。
5. 混合使用动态 policy 目标，但固定起点不漂移。
6. 2 秒完成后进入 `ARMED` 并直接输出 policy frame。
7. 重复 ARM 不重置混合计时、不改变当前输出。
8. `BLENDING` 和 `ARMED` 断流时冻结最后实际输出并进入 `HOLD_STALE`。
9. reference 恢复后不会自动接管；再次 ARM 从冻结帧开始新的 2 秒混合。
10. `on_exit` 清除阶段、hold、blend 和 ARM 会话数据。
11. `arm_blend_seconds` 的非法值构建失败。
12. manifest 中 `btn_10=9/11/12` 分别保持 PICO、进入 ZeroLab、ARM ZeroLab 的用途，且
    `btn_10=12` 只在 `sonic_zerolab` 内生效。
13. 原 `sonic_teleop` 仍构造 `SonicTeleopState`，参数和测试输出不变。

回归验证包括：

- 现有 102 项 ZeroLab 测试；
- SONIC state、policy playout、manifest、Mod loader 和状态机测试；
- `colcon build --merge-install --packages-select bxi_example_py_elf3`；
- 使用 `/tmp/zerolab-arm-direction-20260819-002` 的离线转换和 MuJoCo 回放，确认帧数、关节
  范围和既有结果没有回归。

## 隔离部署与真机验收

实现、测试和提交只发生在本地隔离 worktree：

```text
/home/fazepurple/ros2_ws/bxi_rl_controller_ros2_example_dev/.worktrees/konodoki-dev
```

部署只更新机器人候选目录：

```text
/home/bxi/zerolab-sim2real-candidate-20260818
```

部署前后必须保存并比较：

```text
/home/bxi/bxi_ws/bxi_rl_controller_ros2_example
```

的 `git status --short --branch`，结果必须逐字相同。不得覆盖、清理或提交该原工作区的任何
已有修改和未跟踪文件。

真机验收必须在确认无人使用机器人、没有第二套 `hardware_elf3` 或 demo 进程后进行：

1. root 主终端从候选安装树启动唯一一套硬件和 demo。
2. 机器人按 PD Brake、Normal 顺序稳定站立并停止行走。
3. 第二个 root 终端发送 `btn_10=11`。
4. 操作者保持 T-pose；验证机器人仍保持进入时的 Normal 命令。
5. 出现 `WAIT_ARM` 日志后，操作者自行回到中立姿势。
6. 安全员准备好 `btn_3=1`，再发送 `btn_10=12`。
7. 验证 2 秒内平滑接管，没有肩关节瞬时大角度跳变。
8. 低幅度验证左右单臂、双臂和放下手臂。
9. 人为停止 ZeroLab 数据；验证机器人保持最后命令、日志取消 ARM且不自动切状态。
10. 恢复数据；验证机器人不自动动作。再次发送 `btn_10=12` 后平滑恢复。
11. 返回 Normal，确认 ZeroLab 端口和节点释放；再次进入时确认重新要求 T-pose。
12. 退出后再次比较原工作区状态快照。

任何肩部方向异常、非预期快速运动、失稳或日志与状态不一致，都立即发送 PD Brake 并停止
该轮验收，不通过手动强拧带力矩关节继续测试。

## 验收标准

- T-pose 标定期间真机电机命令保持进入 ZeroLab 时的 Normal 快照。
- 没有 `btn_10=12` 时，任何 fresh live reference 都不能控制真机。
- ARM 只在参考新鲜时接受，并执行完整 2 秒 smoothstep 接管。
- 断流冻结最后输出、取消 ARM，恢复后不自动继续。
- 退出再进入必须重新 T-pose；同一次会话的短暂断流不要求重新 T-pose。
- `btn_3=1` 等原安全路由在全部内部阶段继续有效。
- PICO SONIC、ZeroLab 坐标转换、模型、硬件驱动和原机器人工作区无变化。
- 自动测试、构建、离线回放和候选目录真机验收全部通过后，才能考虑后续合并。
