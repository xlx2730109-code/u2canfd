# sim2real

目标：把 Isaac Lab 训出的策略部署到 Bennett 真机（达妙电机 + DM-IMU），Windows 与 Linux（Jetson）通用。本文件夹汇集部署运行库、动态库与工具，原位置的文件保留不动。

## 基准与结论

### A. 命令

在 `u2canfd/` 下运行（在 `sim2real/` 下运行时去掉 `sim2real/` 前缀）；用装有 torch 的环境（`env_isaaclab`）。**`--check_only` 与 `--dry_run` 不使能电机、不发运动命令；其余命令会连接并驱动真机，须用户当次明确同意。**

不开电机的校验与试运行：

```
python sim2real/bennett_1_crawl_deploy.py --check_only
python sim2real/bennett_1_crawl_deploy.py --dry_run --keyboard
python sim2real/bennett_1_free_deploy.py --check_only
python sim2real/tools/dev_sn.py
```

真机运行（开电机）：

```
python sim2real/bennett_1_crawl_deploy.py --keyboard
python sim2real/bennett_1_free_deploy.py --keyboard
python sim2real/quad_leg_free_gait3_deploy.py --keyboard
python sim2real/bennett_1_crawl_deploy.py --keyboard --dm_sdk_profile low_latency
```

ROS 2 手柄运行用两个终端。Jetson 的终端一启动手柄和比例映射命令桥：

```bash
bash sim2real/start_ros2_teleop.sh
```

终端二启动策略；Jetson 使用已配好的 `policy_python.sh`，已有正确环境时可用 `python`：

```bash
../policy_python.sh sim2real/bennett_1_crawl_deploy.py --ros2
```

程序仍会先使能电机并回默认姿态。准备完成后在策略终端按空格或 Enter 开始策略，按住手柄 `buttons[4]` 才接收速度，松开则速度归零；`buttons[2]` 请求退出策略，`q` 也可退出。两个按钮索引均可在桥接启动命令中配置，物理键位须按当前手柄确认。软件停止不是硬件急停。

链路为 `/joy → bennett_cmd_bridge 比例映射 → /cmd_vel → 本机 UDP → 策略程序 → 电机`。不再启动旧的 `wheeltec_joy` 映射节点。ROS 2 使用系统 Python，策略继续使用自己的 Python 环境；策略进程按入口的速度范围限幅，未收到新鲜命令时归零，默认超时 0.25 s。`--ros2` 与 `--keyboard` 二选一；ROS 模式仍保留终端空格授权和 `q` 退出。

| 摇杆操作 | 速度分量 | 默认满幅速度 |
| --- | --- | --- |
| 右摇杆前后 | `vx`，前进为正 | ±0.25 m/s |
| 右摇杆左右 | `vy`，左移为正 | ±0.12 m/s |
| 左摇杆左右 | `wz`，左转为正 | ±0.5 rad/s |

左摇杆前后不使用；中心死区为 5%，死区外按摇杆幅度线性增加速度，满推达到上表数值。当前轴索引为 `--axes 3 4 0`；已观察到右摇杆前推对应轴 3 正值，另外两项编号和方向仍待采集确认。`--scales` 可配置三个方向的满幅速度。此轮只更改摇杆映射，START 和 B 的新操作方案尚未实现。

入口脚本：

| 入口 | 策略 |
| --- | --- |
| `bennett_1_crawl_deploy.py` | bennett_1 爬行变体，缺省 T9，与 sim2sim 配置 `bennett_1_crawl` 一致 |
| `bennett_1_free_deploy.py` | bennett_1 自由步态，缺省 S1，与 sim2sim 配置 `bennett_1_free` 一致 |
| `quad_leg_free_gait3_deploy.py`、`quad_leg_free_gait4_deploy.py` | 旧的自由步态策略，取 `quad_leg_free_gait` 下最新导出 |
| `quad_leg_go2-10_deploy.py`、`quad_leg_trot1_deploy.py` | go2-10、trot1 |

常用参数（各入口通用，加在命令后面，顺序不限）：

| 参数 | 作用 |
| --- | --- |
| `--keyboard` | 键盘给速度指令；电机回到默认姿态并预热后，按空格或回车才启动策略 |
| `--check_only` | 只校验训练配置与 TorchScript 策略，不打开 IMU 与电机 |
| `--dry_run` | 读 IMU、跑策略，不打开电机 |
| `--dm_sdk_profile original` / `low_latency` | 选电机动态库：缺省 Windows 为 `low_latency`、Linux 为 `original`（2026-10-19 起，此前缺省 `original`）；`original` 为官方版，反馈约每 110–120 ms 一批；`low_latency` 为补丁版，反馈时龄均值约 7 ms，仅 Windows 有。用 T9 等训练时没加滞后的策略、真机迟钝或摔倒时再试 |
| `--imu_port <串口>` | IMU 串口；Windows 缺省 `COM6`，Linux 缺省 `/dev/ttyACM0`（推测），以实际枚举为准 |
| `--output_scale <0–1>` | 策略动作的缩放，首次试验取小值 |
| `--active_joints rr` | 只驱动 RR 的两个关节，缺省 `all` |
| `--duration_s <秒>` | 运行时长，缺省 0 为不限 |
| `--policy <路径>` | 指定 `exported/policy.pt`；运行目录须含 `params/env.yaml`、`params/agent.yaml` |

其他：
- 键盘：`w`/`s` 前后，`a`/`d` 侧移，`j`/`l` 转向，`0`/`x` 清零，空格或回车授权策略，`q` 或 Esc 退出；指令范围缺省取各策略的训练范围。
- 策略已打包在 `sim2real/policies/`，缺省不需要 `bennett_rl`，整个 `sim2real/` 拷到另一台电脑即可运行；另需 Python（含 torch）、pyserial、pyusb、dmcan-sdk，Linux 另需 libusb（见 `docs/WORKFLOW.md`）。
- 换 bennett_1 的策略：环境变量 `BENNETT1_CRAWL_RUN=T9|D4`、`BENNETT1_FREE_RUN=S1`（打包的别名），或 `bennett_rl/logs/rsl_rl/` 下的完整运行目录名，或 `--policy`。
- 打包的策略找不到时才去 `bennett_rl` 找：环境变量 `BENNETT_RL_ROOT`，否则取同级目录 `bennett_rl`，再否则取本仓库所在盘符下的 `Project/Isaaclab/bennett_rl`。

### B. 文件夹结构

```
sim2real/
├── bennett_deploy/                运行库
│   ├── __init__.py                导出契约类
│   ├── async_csv.py               异步 CSV 日志
│   ├── contract.py                go2-10 与 bennett_1 爬行的 50 维部署契约：校验 env.yaml、关节表与符号、动作缩放
│   ├── dm_can.py                  达妙电机 CAN 通信与 MIT 控制
│   ├── free_gait_contract.py      自由步态（33 维）契约
│   ├── free_gait_policy.py        自由步态观测与推理
│   ├── gait.py                    爬行相位时钟
│   ├── imu.py                     DM-IMU 串口读取与一阶低通
│   ├── input.py                   键盘速度指令（Windows 用 msvcrt，Linux 用 termios）
│   ├── paths.py                   路径解析：打包的策略优先，其次训练仓库
│   ├── policy.py                  50 维观测拼装与策略推理
│   ├── ros_command.py             ROS 本机速度接收、限幅、超时与手柄使能门控
│   ├── runtime.py                 主程序：参数、安全启动、1 kHz 控制环、50 Hz 策略
│   ├── tests.py                   部署核心单元测试
│   ├── trot1_contract.py          trot1 契约
│   ├── trot1_gait.py              trot1 步态时钟
│   └── trot1_policy.py            trot1 观测与推理
├── docs/
│   └── WORKFLOW.md                达妙 USB-CANFD Python 例程说明（安装与 Ubuntu 环境）
├── policies/                      已打包的策略，只含 exported/policy.pt 与 params/*.yaml，免去引用 bennett_rl
│   ├── bennett_1_crawl_D4/        bennett_1 爬行，D4（滞后 0–5 步训练）
│   ├── bennett_1_crawl_T9/        bennett_1 爬行，T9（缺省）
│   ├── bennett_1_free_S1/         bennett_1 自由步态，S1（缺省）
│   ├── free_gait3/                free_gait3
│   ├── go2-10/                    go2-10
│   └── trot1/                     trot1
├── sdk/                           达妙动态库
│   ├── dlls/                      官方版（反馈每约 110–120 ms 一批）与 Linux x86-64 `.so`
│   ├── dlls_linux_aarch64/        Linux aarch64（Jetson）版 `libdm_device.so`，1.x 接口，需要 libusb 1.0.27 以上
│   ├── dlls_lowlatency/           补丁版（批量延迟改为 1 ms，反馈时龄均值约 7 ms，无 Linux 版）
│   └── libusb-1.0.dll
├── tools/
│   ├── 0_Return_to_zero.py        8 电机限速安全回零
│   └── dev_sn.py                  列出 USB-CANFD 设备
├── bennett_1_crawl_deploy.py      bennett_1 爬行变体入口（缺省 T9）
├── bennett_1_free_deploy.py       bennett_1 自由步态入口（缺省 S1）
├── quad_leg_free_gait3_deploy.py  自由步态 gait3 入口
├── quad_leg_free_gait4_deploy.py  自由步态 gait4 入口
├── quad_leg_go2-10_deploy.py      go2-10 入口（缺省策略）
├── quad_leg_trot1_deploy.py       trot1 入口
├── README.md                      本文件
├── ros2_teleop.py                 ROS 2 手柄速度桥，可同时启动现有手柄节点
└── start_ros2_teleop.sh           加载 Jetson ROS 环境并启动手柄链路
```

### C. 迁移对照

| 来源（`u2canfd/` 下） | 去向 | 说明 |
| --- | --- | --- |
| `quad_leg_xu/bennett_deploy/` | `bennett_deploy/` | 复制后有改动，见 D |
| `quad_leg_xu/quad_leg_*_deploy.py`（4 个） | 同名，位于本文件夹根 | 只把写死的路径改为打包策略优先、其次 `bennett_rl` 的解析 |
| `dlls/`、`dlls_lowlatency/`、`libusb-1.0.dll` | `sdk/` | 无改动，`libdm_device.dll` 与原件逐字节相同；2026-10-20 新增 `dlls_linux_aarch64/` |
| `0_Return_to_zero.py` | `tools/` | 动态库目录改为 `sdk/` |
| `dev_sn.py` | `tools/` | 无改动 |
| `WORKFLOW.md` | `docs/` | 无改动 |
| 新写 | `bennett_1_crawl_deploy.py`、`bennett_1_free_deploy.py` | bennett_1 的 T9、S1 入口 |
| `bennett_rl/logs/rsl_rl/` 下的 6 个运行目录 | `policies/` | 只取 `exported/policy.pt`、`params/env.yaml`、`params/agent.yaml`，来源见 E |

### D. 相对原件的改动

| 文件 | 改动 |
| --- | --- |
| `runtime.py` | 动态库目录改为 `sdk/`；缺省策略改为打包的 `policies/go2-10`，没有再取 `bennett_rl`；IMU 缺省串口按平台取值；Linux 上选 `low_latency` 报错；打印训练与运行时的观测低通对比；退出时恢复终端 |
| `dm_can.py` | 动态库目录改为 `sdk/`；Linux 上把 dmcan 的后端库绑定到 `sdk/dlls/libdm_device.so`，不再依赖启动目录 |
| `contract.py`、`free_gait_contract.py` | 关节位置、速度观测同时接受官方函数与 bennett_1 的 `filtered_observation` 包装（内层函数仍须相同） |
| `input.py` | 新增 Linux 键盘读取（termios，按键与 Windows 一致）；`parse_posix_keys` 可单测 |
| `paths.py` | 不再写死盘符：环境变量、同级目录、本仓库所在盘符三级查找；新增 `resolve_policy`，打包的策略优先 |
| `tests.py` | 新增 3 个键盘解析测试；读取不存在的旧脚本的测试改为跳过 |
| 关节符号表 `sim_to_motor` | 未改动 |

### E. 打包的策略

| `policies/` 下的目录 | 来源运行目录（`bennett_rl/logs/rsl_rl/` 下） | 用于 |
| --- | --- | --- |
| `bennett_1_crawl_T9` | `bennett_1/crawl/2026-10-16_13-10-20_t9_same` | `bennett_1_crawl_deploy.py` 缺省 |
| `bennett_1_crawl_D4` | `bennett_1/crawl/2026-10-16_15-13-42_d4_delay5` | `BENNETT1_CRAWL_RUN=D4` |
| `bennett_1_free_S1` | `bennett_1/flat/2026-10-16_05-53-35` | `bennett_1_free_deploy.py` 缺省 |
| `free_gait3` | `quad_leg_free_gait/quad_leg_free_gait3/flat/2026-09-05_07-07-11` | `quad_leg_free_gait3_deploy.py`、`quad_leg_free_gait4_deploy.py` |
| `go2-10` | `quad_leg_go2/quad_leg_go2-10/flat/2026-07-19_18-41-26` | `quad_leg_go2-10_deploy.py` |
| `trot1` | `quad_leg_trot/quad_leg_trot1/flat/2026-07-27_20-08-08` | `quad_leg_trot1_deploy.py` |

每个目录约 0.5 MB，`policy.pt` 与原件逐字节相同；`quad_leg_free_gait4_deploy.py` 在 gait4 没有导出策略时取 `free_gait3`。

### F. 未迁移（原处保留）

| 内容 | 位置 | 原因 |
| --- | --- | --- |
| 早期单腿、四足试验脚本 | 根目录 `1_`、`2_`、`3_`、`4_`、`5_*.py`、`damiao.py`、`damiao_2.py`、`single_leg_policy_udp_runner_2.py` | 已被 `bennett_deploy` 取代 |
| 各策略独立部署脚本 | `quad_leg_xu/quad_leg_go2-*.py`、`quad_leg_crawl*.py`、`quad_leg_walk.py`、`quad_leg_trot/`、`quad_leg_free_gait/` | 早期版本，go2-10 之后统一走 `bennett_deploy` |
| DLL 测试、辨识输出、日志 | `_dlltest/`、`sysid_out/`、`logs/`，以及 `quad_leg_xu/*/logs/` | 数据与测试产物，不是部署代码 |

### G. 结论

- sim2real-K1：迁移与修改后验证（均不打开 IMU 与电机）：6 个入口的 `--check_only` 全部通过；`bennett_deploy.tests` 共 12 个测试，11 个通过、1 个跳过（旧脚本不存在）；动态库解析到 `sim2real/sdk/dlls`，哈希一致。
- sim2real-K2：bennett_1 的观测与训练端逐项一致（读代码对照）：50 维顺序、相位 = 回合时间 × 0.55 Hz 取小数、腿序偏移 (0, 0.5, 0.75, 0.25)、占空比 0.78、指令范数小于 0.025 时取站立值（相位 0、四脚着地、参数 0/1/0）；动作缩放 0.2，裁剪 3.5，kp 28，kd 2；角速度、重力、关节速度的 12 Hz 低通与训练一致（运行时会打印对比），关节位置不滤波。
- sim2real-K3：关节符号表 `sim_to_motor` 沿用原文件（FR_calf、RR_calf 为 −1，其余 +1），不改动。`free_gait_contract.py` 的注释记载：FL、RL 的 thigh 取 +1 在实机验证过（2026-09-05），`3_quad_leg_track.py` 的 −1 会使步态镜像、不适用于腿部策略；入口已由用户验证。
- sim2real-K4：训练配置差异：go2-10 的执行器为力矩限 7 N·m、饱和 12 N·m、速度限 30 rad/s，bennett_1 为 8、20、20。
- sim2real-K5：缺省策略的已知性质（来自 `bennett_rl/README.md`）：T9 训练时没有加反馈滞后，滞后 80 ms 开始变差、100 ms 失效；官方动态库的反馈约 109 ms 一批，所以用 T9 时建议用补丁版（Windows，`--dm_sdk_profile low_latency`），或换 D4（K54）。S1 前进时四脚抬高 27–41 mm，两脚同时着地约占 20%（K53）。
- sim2real-K6：Linux 部分只做了静态检查与键盘解析单测，未在 Linux 上运行：`.so` 绑定、termios 键盘、串口名、`low_latency` 无 Linux 版。
- sim2real-K7：`dm_sdk_profile` 缺省：Windows 为 `low_latency`，Linux 为 `original`（2026-10-19 起；此前缺省 `original`）；补丁版的反馈时龄见根 README 01-K27。
- sim2real-K8：把训练仓库藏起来（`BENNETT_RL_ROOT` 指向不存在的目录）后，6 个入口的 `--check_only` 与单元测试照常通过，契约指纹与此前一致，说明运行不依赖 `bennett_rl`；未在另一台电脑上验证，那里还需要 Python 环境与依赖。

- **sim2real-K9**：`--ros2` 从本机 `127.0.0.1:19761` 接收速度，策略进程按入口范围限幅；ROS 桥要求新鲜 `/joy`、按住使能键和新鲜 `/cmd_vel`，默认 0.25 s 超时归零；停止请求锁存并让策略退出。来源：`ros_command.py`、`ros2_teleop.py` 与无电机验证。失联归零针对 ROS/进程消息中断，不等于无线接收器能检测所有无线断连。

- **sim2real-K10**：摇杆映射已改为独立三轴比例输入，右摇杆控制 `vx/vy`，左摇杆左右控制 `wz`；旧的 A/B 切换模式和右摇杆调速行为不再参与默认手柄链路。来源：`JoystickVelocity` 与 `ros2_teleop.py`；实际轴编号、方向及推量响应待逐项验证。

- **sim2real-K11**：第一次右摇杆前推采集显示轴 3 从零增至约 +0.3 后回中；初设 `4/3/0` 将该动作错误映射成侧移，已将前后输入修正为轴 3。半推、满推及其他方向尚未验证。来源：2026-10-20 用户指定动作与 `JOY_AXES`、`CMD_VEL` 输出。

## 已完成

- [x] 2026-10-20 首次前推采集发现前后轴配置错误，修正后进入同方向复测；该次未覆盖满幅输入（sim2real-K11）。

- [x] 2026-10-20 按用户布局实现摇杆幅度到速度的线性映射，保留原有按住使能和超时归零；开始逐方向无电机验证（sim2real-K10）。

- [x] 2026-10-20 增加 `--ros2` 输入、ROS 速度桥和启动脚本；Windows 的门控/限幅验证及 Jetson 的真实 ROS 2 话题到 Python 3.13 运行库验证通过，未打开电机（sim2real-K9）。
  - 首次 Windows 连续发包发现粗粒度单调时钟可能使立即到来的停止包被判为重复；已改用跨进程一致的高精度 `perf_counter`，停止包与实际 ROS 链路验证通过。

- [X]  2026-10-16 建 `sim2real/`，复制部署运行库、入口、动态库、工具与文档，改动动态库路径并验证（sim2real-K1）。
- [X]  2026-10-17 写死的 `E:\` 路径改为跨平台解析；加 Linux 键盘与 `.so` 绑定；放宽契约校验（见 D）。
- [X]  2026-10-17 新增 bennett_1 的 T9、S1 入口并通过 `--check_only`（sim2real-K1、K2）。
- [X]  2026-10-17 策略打包进 `policies/`，入口优先用打包的策略（sim2real-K8）。

## 待完成

- [ ] **sim2real-T6** 逐项验证右摇杆前后、左右及左摇杆左右的轴编号、方向、幅度响应和回中归零；本轮不运行电机（sim2real-K10）。
- [ ] **sim2real-T7** 摇杆映射确认后，再实现并验证 START、B 的新控制流程；当前仍使用原有键盘启动与退出逻辑。

- [ ]  sim2real-T1：用户在真机上试 `bennett_1_crawl_deploy.py`（T9）与 `bennett_1_free_deploy.py`（S1），结果记入本节；首次上电按用户的流程，须当次明确同意。
- [ ]  sim2real-T2：在 Linux（Jetson）上跑 `--check_only`、`--dry_run`，确认 `.so` 绑定、键盘、IMU 串口名（sim2real-K6）。
- [ ]  sim2real-T3：官方动态库下 T9 对约 109 ms 滞后不稳，真机若出现迟钝或摔倒，改用补丁版或 D4（sim2real-K5）。
- [ ]  sim2real-T4：用户确认文件夹结构后，决定原位置文件是否删除。
- [ ]  sim2real-T5：在另一台电脑上拷入整个 `sim2real/`，运行 `--check_only`，确认 Python 依赖与动态库齐全（sim2real-K8）。
