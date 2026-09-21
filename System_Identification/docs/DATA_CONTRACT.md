# 系统辨识数据记录规则

说白了，这份文件规定“一次实验必须留下什么，才能让几天后的我们仍然知道当时做了什么”。字段名保留英文，是为了以后能直接用于程序。看不懂的词可查 [常用词解释](GLOSSARY.md)。

## 小白先记住四件事

1. 原始数据永远不覆盖，处理后的数据另存。
2. 命令和反馈分开记录各自真正发生的时间。
3. 程序重复读取同一条旧反馈，不算电机又采了一次新数据。
4. 每次实验都要附一份 `manifest.yaml`，它就是这次实验的“身份证”。

## 1. 核心规则

1. 每个工作包的 `data/raw/` 只追加，不修改、不覆盖、不滤波、不插值。
2. 一次启动只产生一个唯一 `run_id`，也就是这次实验独有的名字。
3. 命令和反馈是两条独立事件流，之后才能对齐。
4. 只有接收回调触发的新帧才算新反馈；轮询缓存不能增加传感器采样数。
5. 所有数组必须显式保存 joint order、单位、坐标方向和零位定义。
6. 处理后的数据必须保存：由哪个脚本、哪些原始实验和哪些处理参数生成。

## 2. 目录与命名

```text
workstreams/<工作包>/data/raw/YYYY-MM-DD/<run_id>/
├── manifest.yaml
├── command.csv
├── feedback.csv
├── imu.csv                 # 若使用
├── power.csv               # 若使用
├── events.csv
└── notes.md

workstreams/<工作包>/data/processed/<dataset_id>/
├── manifest.yaml
├── aligned.parquet
├── quality_report.json
└── source_runs.txt
```

建议 `run_id`：

```text
YYYYMMDD_HHMMSS_<rig>_<motion>_<repeat>
```

示例：`20260915_143012_suspended_chirpA_r01`

## 3. 实验身份证 `manifest.yaml` 必填内容

### 这次实验是谁、什么时候、用什么代码做的

- `run_id`
- 本地时区和 UTC 开始时间
- 操作者
- git 仓库、commit、dirty 状态
- Python、驱动、Isaac Lab/Isaac Sim 版本
- 数据格式版本

### 硬件

- robot revision
- 电机完整型号、铭牌/序列号（可得时）
- 每个关节的 CAN channel、ESC ID、MST ID
- `sw_ver, Gr, Damp, Inertia, PMAX, VMAX, TMAX, CTRL_MODE, TIMEOUT`
- 电源额定值、起始/结束电压
- 固件补偿和滤波设置；未知时写 `unknown`
- 悬挂工装和接触状态

### 真机和仿真的控制方式

- joint order
- sim ↔ motor 符号和零位
- `sim_dt, decimation, policy_rate_hz, control_rate_hz`
- `kp, kd, tau_ff`
- effort、velocity、position、target-rate 限制
- action scale/clip 和默认关节位置
- 命令保持/插值方式

### 测试动作和安全条件

- 信号类型、频率、幅值、偏置、相位、时长、随机种子
- 训练或验证用途
- 预演配置和结果引用
- 所有停止阈值
- 实体急停检查结果

## 4. 每一条实际命令要记什么

建议一行表示一个关节的一次实际发送命令：

- `run_id`
- `cmd_seq`
- `t_host_mono_ns_enqueue`
- `t_host_mono_ns_tx_begin`
- `t_host_mono_ns_tx_end`
- `channel, can_id, joint_name, joint_index`
- `q_des_joint_rad, dq_des_joint_rad_s, tau_ff_nm`
- `q_des_motor_rad, dq_des_motor_rad_s`
- `kp_nm_rad, kd_nm_s_rad`
- `estimated_pd_tau_nm`
- `effort_limit_nm, velocity_limit_rad_s`
- `phase, signal_id`
- `raw_tx_hex`

若底层 API 无法获得真实总线发送完成时间，要把字段命名为主机 API 时间，不得暗示为线缆上的物理时间。

## 5. 每一条新反馈要记什么

建议一行表示一次真实接收回调：

- `run_id`
- `feedback_seq_global, feedback_seq_joint`
- `t_host_mono_ns_rx_callback`
- `channel, can_id, joint_name, joint_index`
- `status_code`
- `q_motor_rad, dq_motor_rad_s, tau_feedback_nm`
- `q_joint_rad, dq_joint_rad_s`
- `temp_mos_c, temp_rotor_c`（不可用时为空并在 manifest 说明）
- `raw_rx_hex, dlc`

反馈转矩必须注明来源：真实测量、电流换算还是固件估计。没有证据时写 `unknown_estimate`，不能作为独立真值。

## 6. IMU、电源和人工操作怎么记

### IMU

- 原始加速度、角速度、欧拉角/四元数分别保存；
- 保存设备时间戳、主机接收时间戳、寄存器和单位；
- projected gravity 是派生量，只能放 processed 层。

### 电源

- 母线电压、电流、SOC、测量设备与采样率；
- 若来自独立设备，记录时钟同步方法。

### 事件

- 使能、失能、急停、阶段切换、超限、丢帧和人工备注。

## 7. 后期怎么把命令和反馈放到同一条时间线上

- 使用不会因系统校时而倒退的单调时钟（monotonic clock）计算间隔；UTC 只用于跨设备和文件查找。
- 命令按真实发送时间保持到下一条命令，除非真实控制器明确使用其他插值方式。
- 反馈先按回调时间排序和去重，再对齐到仿真时间栅格。
- 保存插值掩码、数据空洞和丢帧区间。
- 写延迟时必须说明是哪一种：接口调用耗时、反馈有多旧、传输往返，还是曲线匹配得到的综合延迟。
- 不能将样本索引除以“目标频率”代替真实时间戳。

## 8. 什么样的数据才允许拿去找参数

进入拟合前至少检查：

- manifest 完整；
- 无未知 joint mapping；
- 无接触和机械碰撞；
- 无持续超限、错误码或急停；
- 命令和反馈序号单调；
- 反馈新鲜率和丢帧率已报告；
- 时间戳无倒退；
- 每个关节的运动范围和测试频率覆盖了以后真正关心的运动；
- 找参数的数据和用来考试的新数据没有重复实验。
