# 现有代码/配置排查（v0，2026-10-02，只读）

范围：根目录 `damiao*.py`、`quad_leg_xu/bennett_deploy/`（dm_can.py、runtime.py）、`dlls*`、旧 `System_Identification/` 的 G0 寄存器快照与延迟数据、Trot1 训练 `env.yaml`。
未运行任何真机代码。标注"待验证"的是推断，不是实测。

## A. 通信 / 时序

| # | 发现 | 证据 | 影响 |
|---|---|---|---|
| A1 | 官方 `dlls/libdm_device.dll` 里有一个 100 ms 常量（`mov edx,0x05F5E100`=1e8 ns），已有补丁版 `dlls_lowlatency/` 把它改成 1 ms（`0x000F4240`），仅此 4 字节不同 | `cmp` 对比 + 十六进制 | 补丁版对应你图里的问题；需要实测补丁版延迟，而不是假设它"好了" |
| A2 | 部署入口不统一：`quad_leg_trot/quad_leg_trot1.py`、`go2-11/12/13`、`free_gait2` 传 `--dm_sdk_profile low_latency`；但 `quad_leg_trot1_deploy.py`（文档称"稳定入口"）没传，默认 `original`，即 109 ms 批量交付 | `runtime.py:140` 默认值 + `quad_leg_trot1_deploy.py` | 用该入口上真机，反馈平均旧 55 ms，与训练（无延迟）严重不符 |
| A3 | 9 月的延迟测量脚本 `probe_feedback_age_1khz.py` 走的是根目录 `damiao.py` → 官方 DLL，不是补丁版 | 脚本 import 与 `os.chdir(PARENT)`、manifest 无 profile 字段 | 那份 109 ms 数据描述的是官方 DLL；部署实际用的补丁版延迟**未测过** |
| A4 | 根目录老脚本（`0_`–`5_`、`damiao.py`、`damiao_2.py`、`quad_leg_crawl*.py`）全部硬编码加载 `dlls/`（官方 DLL） | grep | 任何用它们采的数据都带 109 ms 批量延迟，不能混进新数据集 |
| A5 | `Motor` 只保存最新一帧，`updateTimeInterval()` 只更新间隔，不存"这一帧的接收时间戳/序号" | `dm_can.py` Motor 类 | 无法区分"新反馈"和"旧缓存"，正是新日志器要补的 |
| A6 | 部署主循环用 `time.sleep` 做 1 kHz 节拍，靠 `windows_timer_period(1)` 把 Windows 定时器设到 1 ms；日志里有 overrun 统计 | `runtime.py` 主循环 | Windows 抖动可能达数 ms，需实测 overrun 分布 |
| A7 | 回调里有参数读写模式标志：`read_write_save` 置位时，若普通反馈帧的 `data[2]` 恰好等于 0x33/0x55/0xAA 会被当参数应答吞掉 | `canframeCallback` | 只在读写寄存器期间可能丢个别帧，概率低；采集时避免同时读寄存器 |

## B. 电机 / 寄存器（`registers.csv`，9-16，不使能读取）

| # | 发现 | 影响 |
|---|---|---|
| B1 | `TIMEOUT=0`（8 台全部）：通信超时保护关闭 | 上位机崩溃/USB 掉线时电机会保持最后一条命令（带着 Kp=28 去追旧目标）。**安全问题**，也影响"掉线后行为"的一致性。需决定是否打开 |
| B2 | `OC_Value=0.8`（8 台）：含义待查（可能是过流保护的比例/阈值） | 若是峰值电流的 80%，实际峰值转矩约 16 N·m 而非 20；会改变转矩饱和模型 |
| B3 | `sub_ver`：前 5 台 =51，`RL_calf`、`RR_thigh`、`RR_calf` =52；`hw_ver`/`SN` 读出全是 0 | 固件小版本不一致，可能有行为差异；SN 读不到，设备身份只能靠 CAN ID |
| B4 | `FR_thigh` 与 `RL_thigh` 的 Rs、LS、Flux、Damp、Inertia 到 9 位有效数字完全相同 | 概率极低，疑似同一电机被读了两次（通道/ID 路由问题）或出厂默认值复制；待验证 |
| B5 | 固件内部辨识值：`Inertia≈1.2e-4`、`Damp≈5e-4–1.1e-3`；`Gr=6`，`NPP=21`，`Flux≈0.0055 Wb` | 若 `Inertia` 是转子侧：折算到输出轴 ×36 ≈ 4.4e-3 kg·m²（待验证）。用 1.5·NPP·Flux ≈ 0.172 N·m/A（转子侧），×6 ≈ 1.03 N·m/A（输出，未计效率），与 20 N·m/≈21 A 峰值基本自洽 |
| B6 | `KT_Value=0.0`、`GREF=1.0`、`KP_APR=54`、`I_BW=1000`、`V_BW=40` | KT=0 的含义需查手册；MIT 命令实际是 `Kp·Δq + Kd·Δv + τff`，其转矩精度取决于 KT 与电流环，**需台架标定** |
| B7 | MIT 编码量化：位置 16 bit/±12.5 rad（0.38 mrad）、速度 12 bit/±45（0.022 rad/s）、转矩 12 bit/±20（0.0098 N·m）、Kp 12 bit/0–500（0.122）、Kd 12 bit/0–5；编码用 `int()` 截断不是四舍五入 | Kp=28 实际发出约 27.96，转矩有约 −0.5 LSB 的系统偏置，量级小但应在仿真里复现或改为四舍五入 |

## C. 训练配置（Trot1 `env.yaml`）与真机的差距

| # | 训练里 | 真机现实 / 风险 |
|---|---|---|
| C1 | 执行器 `DCMotor`，`armature=null`、`friction=0`、`viscous_friction=null`、无延迟 | 真机有转子惯量折算（B5）、减速器摩擦、通信与控制延迟 → 这正是 PACE 要辨识的那几项 |
| C2 | `effort_limit=8`，`saturation_effort=20`，`velocity_limit=20 rad/s`，`damping=2`，`stiffness=28` | 部署默认 `software_effort_limit_nm=0`（关闭）；固件限幅是 20（TMAX），两边限幅逻辑不同 |
| C3 | 随机化只有：Kp×0.8–1.2、Kd×0.7–1.3、基座质量 ×0.9–1.1、基座 COM、足端摩擦 0.6–1.3、推力；**没有**腿部连杆质量/惯量随机化、关节摩擦/电枢/延迟随机化 | 样机质量与仿真不对齐时，没有随机化兜底 |
| C4 | 观测噪声：角速度 ±0.2、重力投影 ±0.05、关节位置 ±0.01、关节速度 ±1.5 rad/s | 真机编码器位置噪声约 0.38 mrad（量化）远小于 0.01；速度噪声 ±1.5 很宽，可能掩盖了延迟/滤波差异 |
| C5 | 部署端对关节速度加 12 Hz 一阶低通、IMU 加 12 Hz 低通（`--joint_vel_cutoff_hz 12`、`--imu_cutoff_hz 12`） | 训练里没有对应滤波（待核对 `env.yaml`/代码）。12 Hz 一阶滤波有 ~13 ms 相位滞后，且只在 50 Hz 策略步更新 |
| C6 | 动作 = 默认姿态 + 0.2·action，按关节裁剪；部署有 `--target_rate_limit_deg_s` 默认 60000 | 基本等于关闭，OK；但启动 ramp 默认 20°/s，需与训练初始状态对齐 |
| C7 | 机器人 USD：`Urdf_Bennett_3.usd`（路径里是 `Urdf_Bennett_4/`） | 质量/惯量来源未知（`mass_props: null` = 用 USD 里的值）；闭链机构的建模方式待查 |

## D. 部署代码逻辑（`runtime.py`）

| # | 发现 | 说明 |
|---|---|---|
| D1 | `default_mode=motor_zero` 且 `--motor_zero_positions` 默认全 0 | 假设"电机编码器零位 = 仿真零位 + 符号"。关节零位/方向没有独立验证（旧审计也指出） |
| D2 | 观测在策略 tick 时刻读"最新快照"，动作在下一个 1 kHz 循环发出，没有显式计算推理耗时 | 仿真里是"物理步 → 观测 → 动作 → 下一步"，真机有推理+IMU 年龄+反馈年龄；这些加起来才是等效延迟 |
| D3 | 反馈超时默认 0.5 s，IMU 超时 0.1 s | 0.5 s 的反馈超时在 109 ms 批量交付下不会触发，等于对"反馈很旧"没有保护 |

## E. 旧 sysid 结论的可信度

旧审计（`System_Identification/docs/BASELINE_AUDIT.md`）自己已指出：阶跃/Kp/阻尼/延迟/IMU 的旧数据大多不可用。补充：因为 A3/A4，旧数据还带官方 DLL 的 109 ms。结论：**旧数据一律不进新数据集**；只有 9-16 的寄存器快照和延迟分布可作为线索。

## F. 建议优先动作（只列方向，待你确认）

1. 先补测"补丁版 DLL"的反馈周期/年龄（零力矩、不使能），再决定是否需要更换驱动或重新打补丁。
2. 统一部署入口的 DLL profile，或在日志 manifest 里强制记录 profile 与 DLL SHA256。
3. 确认 `TIMEOUT=0` 是否改为非零（电机侧看门狗）。
4. 查清 `OC_Value`、`KT_Value=0` 的含义、`FR_thigh`/`RL_thigh` 寄存器相同的原因、`sub_ver` 差异。
5. 核对训练里有无速度/IMU 低通，使部署滤波与训练一致。

## G. 补充（2026-10-08）

1. 开题里的"约 36 ms"是批相位，不是常数；Trot1 `env.yaml` 无延迟。
2. Linux `.so` 里同样有 100 ms 常量（偏移 0x13628，与 Windows DLL 同值），未打补丁；Jetson 部署前必须单独处理。
3. 符号表冲突：`3_quad_leg_track.py` FL/RL_thigh=-1（实机验证过）vs `bennett_deploy/contract.py`=+1。
4. `技术文档_1.md` 的 ×6 系数已被实测推翻（关节端 1:1）。
5. 旧 kp 标定（23–31 N·m/rad）是静态、悬空、含重力柔度，不能当作"执行器已对齐"。
6. SW 装配体 Bennett_test5：质量 12.0098 kg（URDF 11.90 kg），质心 (1.35, 0.11, -14.6) mm，Ixx/Iyy/Izz ≈ 0.2765/0.5380/0.6490 kg·m²；电池型号与接线图不一致（5300 mAh vs 2200 mAh）。
7. 条件限制：腿已装好不能拆、无单电机台架 → 质量惯量用整机称重/悬吊/多点秤/扭摆，执行器用整机悬空辨识，KT/摩擦置信度单独标注。

> 2026-10-20 更新：A2（部署入口缺省走官方库）已处理，缺省改为 Windows `low_latency`，见根 README 01-K45；通信延迟的实测与结论见根 README 的 `01_communication` 一节，本文件的 109 ms 等数字是 9 月的旧结论。
