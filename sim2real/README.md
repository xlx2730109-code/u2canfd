# sim2real

目标：把 Isaac Lab 训出的策略部署到 Bennett 真机（达妙电机 + DM-IMU）。本文件夹汇集部署运行库、动态库与工具，原位置的文件保留不动。

## 基准与结论

### A. 命令

在 `sim2real/` 下运行。**`--check_only` 与 `--dry_run` 不使能电机、不发运动命令；去掉它们就会连接并驱动真机，须用户当次明确同意。**

```
python quad_leg_go2-10_deploy.py --check_only
python quad_leg_go2-10_deploy.py --dry_run --keyboard --imu_port COM6
python tools/dev_sn.py
```

- `--check_only`：只校验训练配置与 TorchScript 策略，不打开 IMU 与电机。
- `--dry_run`：读 IMU、跑策略，不打开电机。
- `tools/dev_sn.py`：列出 USB-CANFD 设备。
- 真机运行与回零：`tools/0_Return_to_zero.py`、各 `*_deploy.py` 去掉上面两个开关。

### B. 文件夹结构

```
sim2real/
├── bennett_deploy/                运行库
│   ├── __init__.py                导出契约类
│   ├── async_csv.py               异步 CSV 日志
│   ├── contract.py                go2-10 的 50 维部署契约：校验 env.yaml、关节表与符号、动作缩放
│   ├── dm_can.py                  达妙电机 CAN 通信与 MIT 控制
│   ├── free_gait_contract.py      自由步态（33 维）契约
│   ├── free_gait_policy.py        自由步态观测与推理
│   ├── gait.py                    爬行相位时钟
│   ├── imu.py                     DM-IMU 串口读取与一阶低通
│   ├── input.py                   键盘速度指令（仅 Windows 有按键读取）
│   ├── paths.py                   训练仓库路径解析
│   ├── policy.py                  50 维观测拼装与策略推理
│   ├── runtime.py                 主程序：参数、安全启动、1 kHz 控制环、50 Hz 策略
│   ├── tests.py                   部署核心单元测试
│   ├── trot1_contract.py          trot1 契约
│   ├── trot1_gait.py              trot1 步态时钟
│   └── trot1_policy.py            trot1 观测与推理
├── docs/
│   └── WORKFLOW.md                达妙 USB-CANFD Python 例程说明（安装与 Ubuntu 环境）
├── sdk/                           达妙动态库
│   ├── dlls/                      官方版（反馈每约 109 ms 一批）与 Linux `.so`
│   ├── dlls_lowlatency/           补丁版（批量延迟改为 1 ms，真实延迟未测）
│   └── libusb-1.0.dll
├── tools/
│   ├── 0_Return_to_zero.py        8 电机限速安全回零
│   └── dev_sn.py                  列出 USB-CANFD 设备
├── quad_leg_free_gait3_deploy.py  自由步态 gait3 入口
├── quad_leg_free_gait4_deploy.py  自由步态 gait4 入口
├── quad_leg_go2-10_deploy.py      go2-10 入口（缺省策略）
├── quad_leg_trot1_deploy.py       trot1 入口
└── README.md                      本文件
```

### C. 迁移对照

| 来源（`u2canfd/` 下） | 去向 | 改动 |
| --- | --- | --- |
| `quad_leg_xu/bennett_deploy/` | `bennett_deploy/` | `runtime.py`、`dm_can.py` 各改一行：动态库目录由 `u2canfd/` 改为 `sim2real/sdk/` |
| `quad_leg_xu/quad_leg_*_deploy.py`（4 个） | 同名，位于本文件夹根 | 无 |
| `dlls/`、`dlls_lowlatency/`、`libusb-1.0.dll` | `sdk/` | 无，`libdm_device.dll` 与原件逐字节相同 |
| `0_Return_to_zero.py` | `tools/` | 动态库目录改为 `sdk/` |
| `dev_sn.py` | `tools/` | 无 |
| `WORKFLOW.md` | `docs/` | 无 |

### D. 未迁移（原处保留）

| 内容 | 位置 | 原因 |
| --- | --- | --- |
| 早期单腿、四足试验脚本 | 根目录 `1_`、`2_`、`3_`、`4_`、`5_*.py`、`damiao.py`、`damiao_2.py`、`single_leg_policy_udp_runner_2.py` | 已被 `bennett_deploy` 取代；`3_quad_leg_track.py` 是 FL、RL 的 thigh 符号为 −1 的实机验证来源（见 K3） |
| 各策略独立部署脚本 | `quad_leg_xu/quad_leg_go2-*.py`、`quad_leg_crawl*.py`、`quad_leg_walk.py`、`quad_leg_trot/`、`quad_leg_free_gait/` | 早期版本，go2-10 之后统一走 `bennett_deploy` |
| DLL 测试、辨识输出、日志 | `_dlltest/`、`sysid_out/`、`logs/`，以及 `quad_leg_xu/*/logs/` | 数据与测试产物，不是部署代码 |

### E. 结论

- sim2real-K1：迁移后验证（均不打开 IMU 与电机）：`quad_leg_go2-10_deploy.py --check_only` 通过，动态库解析到 `sim2real/sdk/dlls` 且哈希一致；`import bennett_deploy.dm_can` 成功。
- sim2real-K2：现有契约校验只接受官方原版的关节观测函数，bennett_1 的 E2 在 `--check_only` 下报 `unsupported joint_pos observation function`（bennett_1 用 `filtered_observation`）。要部署 bennett_1 须新增契约，其余项（观测顺序、动作缩放 0.2、裁剪 3.5、kp 28、kd 2、默认姿态、相位时钟 0.55 Hz、占空比 0.78）与现有一致。
- sim2real-K3：关节符号 `sim_to_motor` 在 `bennett_deploy/contract.py` 中为 FR_calf、RR_calf 取 −1，其余 +1；`3_quad_leg_track.py` 中 FL、RL 的 thigh 为 −1（实机验证过），两者冲突，未定论（`ROBOT.md` 第 7 节）。
- sim2real-K4：训练配置差异：go2-10 的执行器为力矩限 7 N·m、饱和 12 N·m、速度限 30 rad/s，bennett_1 为 8、20、20。
- sim2real-K5：`input.py` 在非 Windows 下不读取按键，指令固定为初始值且 `armed` 不会变真，Linux 上须用 `--auto_arm`。
- sim2real-K6：`tests.py` 有一个测试读取 `quad_leg_xu/quad_leg_go2-10.py`，该文件不存在，按代码判断此测试在原处同样会失败（未运行）。
- sim2real-K7：`runtime.py` 的缺省策略路径、`quad_leg_trot1_deploy.py`、`quad_leg_free_gait3_deploy.py` 写死了 `E:\` 路径，不符合双平台要求。

## 已完成

- [X]  2026-10-16 建 `sim2real/`，复制部署运行库、入口、动态库、工具与文档，改动动态库路径并验证（sim2real-K1）。

## 待完成

- [ ]  sim2real-T1：新增 bennett_1 的部署契约与入口，使 E2、D4 能通过 `--check_only`（sim2real-K2）。
- [ ]  sim2real-T2：键盘指令范围对齐训练（前后 ±0.25、侧移 ±0.12、转向 ±0.5）；补 Linux 键盘读取（sim2real-K5）。
- [ ]  sim2real-T3：写死的 `E:\` 路径改为相对路径或环境变量（sim2real-K7）。
- [ ]  sim2real-T4：确定 `sim_to_motor` 符号（sim2real-K3），需用户在真机上确认。
- [ ]  sim2real-T5：用户确认文件夹结构后，决定原位置文件是否删除。
