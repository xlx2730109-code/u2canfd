# Bennett 机器人系统辨识工作区

这里用来解决一件事：让同一条命令发给真实机器人和仿真机器人后，两边的运动尽量接近。

当前状态：**只整理好了资料和计划，还没有开始新的真机实验。**

## 第一次打开，从这里看

1. 想先弄懂“系统辨识到底在做什么”：看 [小白版总导航](docs/BEGINNER_GUIDE.md)。
2. 想知道每个文件夹分别干什么：看 [工作区地图](workstreams/README.md)。
3. 真正准备开工：按 [正式计划](Plan.md) 从 G0 开始，不要跳着做。
4. 遇到不认识的词：查 [常用词解释](docs/GLOSSARY.md)。

## 目前已经整理好的内容

- 旧代码和旧数据哪些能信、哪些不能信：[旧资料检查结果](docs/BASELINE_AUDIT.md)。
- 新实验应该记录哪些数据：[数据记录规则](docs/DATA_CONTRACT.md)。
- 真机通电和运动前必须检查什么：[真机安全检查表](docs/HARDWARE_SAFETY_CHECKLIST.md)。
- 达妙 24 V 官方曲线模型：[电机曲线工作包](workstreams/05_motor_datasheet_model/README.md)。
- 后续所有任务已经分进 12 个独立工作包，不再把脚本、数据和图片堆在一起。

## 这项工作不只是在“调电机参数”

要分清三类问题：

| 问题 | 例子 | 怎么解决 |
|---|---|---|
| 真机和仿真的定义没对上 | 关节顺序、正负方向、零位、频率、限幅不同 | 逐项检查和实测 |
| 机器人本身的参数不准 | 质量、惯量、摩擦、延迟、传感器偏差 | 测量和系统辨识 |
| 训练方法没设计好 | 奖励、PPO 参数、网络结构不合适 | 单独做训练对比 |

系统辨识不能包治所有 Sim-to-Real 问题。普通代码错误、IMU 方向错误、动作缩放不一致，也会让策略迁移失败。

## 文件夹地图

```text
System Identification/
├── README.md                  # 你现在看的总入口
├── Plan.md                    # 真正开工时按它执行
├── config/                    # 电脑可读取的实验配置模板
├── docs/
│   ├── README.md              # 说明文档索引
│   ├── BEGINNER_GUIDE.md      # 小白版说明
│   ├── GLOSSARY.md            # 常用词翻译
│   ├── BASELINE_AUDIT.md      # 旧代码、旧数据检查结果
│   ├── DATA_CONTRACT.md       # 数据应该怎么记
│   └── HARDWARE_SAFETY_CHECKLIST.md
├── references/                # 论文、官网和输入资料来源
└── workstreams/               # 每类工作各放各的
    ├── 00_control_contract/   # 先对齐命令、频率、增益和限幅
    ├── 01_hardware_inventory/ # 查清真实硬件和固件
    ├── 02_timing_latency/     # 测通信周期、抖动和综合延迟
    ├── 03_joint_mapping_zero/ # 对关节顺序、方向和零位
    ├── 04_rigid_body_properties/ # 尺寸、质量、质心和惯量
    ├── 05_motor_datasheet_model/ # 达妙官方曲线模型
    ├── 06_data_acquisition/   # 统一的数据记录工具
    ├── 07_actuator_identification/ # 辨识关节运动特性
    ├── 08_sensor_identification/   # 编码器和 IMU
    ├── 09_contact_identification/  # 脚和地面接触
    ├── 10_simulation_validation/   # 用新动作检验结果
    └── 11_training_deployment/     # 训练、部署和真机验收
```

文件夹名保留英文，是为了以后脚本路径稳定；每个入口文档都用中文解释。

## 三条不能破的规则

1. 原始数据只增加，不覆盖；处理后的数据另存。
2. 用来找参数的动作，不能再拿来证明参数很好，必须换一套新动作验证。
3. “整理资料”和“开始真机实验”是两件事。没有你当次明确同意，不使能电机、不发送运动命令。

## 下次开始工作的说法

你可以直接说：

> 开始系统辨识。先执行 Plan.md 的 G0，只做不使能电机的检查、代码审查和离线测试；关键问题随时问我。

到了需要电机运动的阶段，我会重新列出具体关节、动作、增益、幅值、速度、限矩、停止条件和风险，再由你决定是否开始。
